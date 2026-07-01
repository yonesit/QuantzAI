"""
src/data/mt5_connector.py
MT5Connector – Verbindung zu MetaTrader 5 und OHLCV-Datenabruf.

Zustaendigkeiten:
  - Verbindungsaufbau mit Retry-Logik
  - Health-Check im Hintergrund
  - OHLCV-Daten als standardisierter DataFrame
  - Keine Business-Logik – nur Transport

Voraussetzung: MetaTrader5-Package laeuft nur auf Windows.
"""

from __future__ import annotations

import time
import threading
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

import pandas as pd
from loguru import logger


# ─────────────────────────────────────────────
#  Exceptions
# ─────────────────────────────────────────────

class MT5ConnectionError(Exception):
    """Wird ausgeloest wenn die MT5-Verbindung fehlschlaegt oder verloren geht."""


class MT5DataError(Exception):
    """Wird ausgeloest wenn ein Datenabruf fehlschlaegt."""


# ─────────────────────────────────────────────
#  Timeframe-Enum
# ─────────────────────────────────────────────

class Timeframe(str, Enum):
    M1  = "M1"
    M5  = "M5"
    M15 = "M15"
    M30 = "M30"
    H1  = "H1"
    H4  = "H4"
    D1  = "D1"
    W1  = "W1"


# ─────────────────────────────────────────────
#  MT5-Konstanten (werden zur Laufzeit gemappt)
# ─────────────────────────────────────────────

_TIMEFRAME_MAP: dict[str, int] = {}
_MT5_MODULE = None


def _load_mt5():
    """Importiert MetaTrader5 einmalig und befuellt die Timeframe-Map."""
    global _MT5_MODULE, _TIMEFRAME_MAP
    if _MT5_MODULE is not None:
        return _MT5_MODULE
    try:
        import MetaTrader5 as mt5  # type: ignore
        _MT5_MODULE = mt5
        _TIMEFRAME_MAP = {
            "M1":  mt5.TIMEFRAME_M1,
            "M5":  mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1":  mt5.TIMEFRAME_H1,
            "H4":  mt5.TIMEFRAME_H4,
            "D1":  mt5.TIMEFRAME_D1,
            "W1":  mt5.TIMEFRAME_W1,
        }
        return mt5
    except ImportError as exc:
        raise MT5ConnectionError(
            "MetaTrader5-Package nicht installiert. "
            "Installiere es mit: pip install MetaTrader5  (nur Windows)"
        ) from exc


def read_stops_level(info) -> int:
    """Liest den Broker-Mindestabstand (Punkte) robust aus einem SymbolInfo.

    Das echte MetaTrader5-Paket nennt das Feld ``trade_stops_level``. Aeltere
    Stubs / Mocks nutzten faelschlich ``stops_level`` – deshalb beide Namen
    probieren. Faellt auf 0 zurueck (= kein Mindestabstand, kein Clamp) wenn
    keines vorhanden oder nicht numerisch ist. Wirft niemals eine Exception,
    damit ein fehlendes Feld den Live-Handel nicht blockiert.
    """
    for attr in ("trade_stops_level", "stops_level"):
        val = getattr(info, attr, None)
        try:
            ival = int(val)
        except (TypeError, ValueError):
            continue
        if ival >= 0:
            return ival
    return 0


# ─────────────────────────────────────────────
#  MT5Connector
# ─────────────────────────────────────────────

class MT5Connector:
    """
    Verwaltet die Verbindung zu MetaTrader 5 und liefert OHLCV-Daten.

    Parameters
    ----------
    login            : MT5-Kontonummer
    password         : MT5-Passwort
    server           : Broker-Server-Name
    path             : Pfad zur terminal64.exe (optional)
    max_retries      : Maximale Verbindungsversuche (Standard: 3)
    health_interval  : Sekunden zwischen Health-Checks (Standard: 60)
    """

    def __init__(
        self,
        login: int,
        password: str,
        server: str,
        path: Optional[str] = None,
        max_retries: int = 3,
        health_interval: int = 60,
    ) -> None:
        self._login           = login
        self._password        = password
        self._server          = server
        self._path            = path
        self._max_retries     = max_retries
        self._health_interval = health_interval

        self._connected       = False
        self._lock            = threading.Lock()
        self._health_thread:  Optional[threading.Thread] = None
        self._stop_health     = threading.Event()
        self._reconnect_callbacks: list = []

    def register_reconnect_callback(self, callback) -> None:
        """Registriert eine Funktion die nach jedem erfolgreichen (Re-)Connect aufgerufen wird."""
        self._reconnect_callbacks.append(callback)

    def _fire_reconnect_callbacks(self) -> None:
        for cb in self._reconnect_callbacks:
            try:
                cb()
            except Exception as exc:  # noqa: BLE001
                logger.error("Reconnect-Callback Fehler: {exc}", exc=exc)

    # ── Context Manager ──────────────────────────────

    def __enter__(self) -> "MT5Connector":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.disconnect()

    # ── Verbindung ───────────────────────────────────

    @property
    def is_connected(self) -> bool:
        """True wenn die Verbindung aktiv ist."""
        with self._lock:
            return self._connected

    def connect(self) -> bool:
        """
        Baut Verbindung zu MT5 auf. Versucht bis zu max_retries Mal
        mit exponentiellem Backoff (1s, 2s, 4s).

        Returns True bei Erfolg, wirft MT5ConnectionError bei Misserfolg.
        """
        mt5 = _load_mt5()

        for attempt in range(1, self._max_retries + 1):
            try:
                kwargs: dict = dict(
                    login=self._login,
                    password=self._password,
                    server=self._server,
                )
                if self._path:
                    kwargs["path"] = self._path

                if mt5.initialize(**kwargs):
                    with self._lock:
                        self._connected = True
                    logger.info(
                        "MT5 connected | server={server} login={login}",
                        server=self._server,
                        login=self._login,
                    )
                    self._start_health_check()
                    self._fire_reconnect_callbacks()
                    return True

                err = mt5.last_error()
                logger.warning(
                    "MT5 reconnect attempt {attempt}/{max} | error={err}",
                    attempt=attempt,
                    max=self._max_retries,
                    err=err,
                )

            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "MT5 reconnect attempt {attempt}/{max} | exception={exc}",
                    attempt=attempt,
                    max=self._max_retries,
                    exc=exc,
                )

            if attempt < self._max_retries:
                backoff = 2 ** (attempt - 1)
                time.sleep(backoff)

        raise MT5ConnectionError(
            f"MT5-Verbindung nach {self._max_retries} Versuchen gescheitert | "
            f"server={self._server} login={self._login}"
        )

    def disconnect(self) -> None:
        """Trennt die MT5-Verbindung und stoppt den Health-Check."""
        self._stop_health.set()
        mt5 = _load_mt5()
        mt5.shutdown()
        with self._lock:
            self._connected = False
        logger.info("MT5 disconnected | login={login}", login=self._login)

    # ── Health-Check ─────────────────────────────────

    def _start_health_check(self) -> None:
        """Startet einen Hintergrund-Thread der die Verbindung prueft."""
        self._stop_health.clear()
        self._health_thread = threading.Thread(
            target=self._health_loop,
            daemon=True,
            name="mt5-health",
        )
        self._health_thread.start()

    def _health_loop(self) -> None:
        while not self._stop_health.wait(timeout=self._health_interval):
            self._health_check()

    def _health_check(self) -> None:
        """Prueft die Verbindung und reconnected bei Bedarf. Testbar."""
        try:
            mt5 = _load_mt5()
            info = mt5.terminal_info()
            if info is None or not getattr(info, "connected", False):
                logger.warning("MT5 Health-Check: Verbindung verloren – reconnect...")
                with self._lock:
                    self._connected = False
                self._reconnect()
        except Exception as exc:  # noqa: BLE001
            logger.error("MT5 Health-Check Fehler: {exc}", exc=exc)

    def _reconnect(self) -> None:
        """Interner Reconnect ohne erneuten Health-Check-Start."""
        mt5 = _load_mt5()
        kwargs: dict = dict(
            login=self._login,
            password=self._password,
            server=self._server,
        )
        if self._path:
            kwargs["path"] = self._path

        if mt5.initialize(**kwargs):
            with self._lock:
                self._connected = True
            logger.info("MT5 reconnected | login={login}", login=self._login)
            self._fire_reconnect_callbacks()
        else:
            logger.error(
                "MT5ConnectionError | reconnect gescheitert | {error}",
                error=mt5.last_error(),
            )

    # ── Daten ────────────────────────────────────────

    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str | Timeframe,
        start: datetime,
        end: datetime,
        include_spread: bool = False,
    ) -> pd.DataFrame:
        """
        Liefert OHLCV-Daten als DataFrame mit UTC-DatetimeIndex.

        Parameters
        ----------
        symbol    : z.B. "EURUSD"
        timeframe : Timeframe-Enum oder String ("H1", "M15", ...)
        start     : Startzeit (naive datetimes werden als UTC behandelt)
        end       : Endzeit   (naive datetimes werden als UTC behandelt)
        include_spread : Wenn True wird die MT5-``spread``-Spalte (Spread in
                    POINTS je Bar, vom Broker gemeldet) mit ausgegeben. Standard
                    False, damit bestehende Aufrufer unveraendert nur OHLCV
                    erhalten. ACHTUNG: spread ist in Points, nicht Pips – die
                    Points->Pips-Umrechnung erfolgt bewusst erst beim Verbraucher.

        Returns
        -------
        pd.DataFrame  Index: UTC DatetimeIndex
                      Spalten: open, high, low, close, volume
                      (+ spread in Points, falls include_spread=True)
        """
        if not self.is_connected:
            raise MT5ConnectionError("Nicht verbunden – rufe zuerst connect() auf.")

        mt5 = _load_mt5()
        tf_str = timeframe.value if isinstance(timeframe, Timeframe) else str(timeframe).upper()

        if tf_str not in _TIMEFRAME_MAP:
            raise MT5DataError(
                f"Ungueltiger Timeframe: '{tf_str}'. "
                f"Erlaubt: {list(_TIMEFRAME_MAP.keys())}"
            )

        start_utc = start.replace(tzinfo=timezone.utc) if start.tzinfo is None else start
        end_utc   = end.replace(tzinfo=timezone.utc)   if end.tzinfo   is None else end

        rates = mt5.copy_rates_range(symbol, _TIMEFRAME_MAP[tf_str], start_utc, end_utc)

        if rates is None or len(rates) == 0:
            err = mt5.last_error()
            # Broker-Limit-Fallback: copy_rates_range versagt fuer grosse/alte Bereiche
            # (-2 = Invalid params).  copy_rates_from_pos liefert bis zu 99.999 Bars ab
            # der aktuellen Kerze – wir filtern anschliessend auf das gewuenschte Fenster.
            if err[0] == -2:
                logger.warning(
                    "copy_rates_range fehlgeschlagen fuer {sym} {tf} ({err}) – "
                    "Fallback: copy_rates_from_pos(99999) und Datums-Filter",
                    sym=symbol, tf=tf_str, err=err,
                )
                rates = mt5.copy_rates_from_pos(symbol, _TIMEFRAME_MAP[tf_str], 0, 99_999)
            if rates is None or len(rates) == 0:
                raise MT5DataError(
                    f"Keine Daten fuer {symbol} {tf_str} "
                    f"({start_utc} – {end_utc}) | MT5-Fehler: {err}"
                )

        df = pd.DataFrame(rates)

        df = df.rename(columns={"tick_volume": "volume"})
        if "real_volume" in df.columns and "volume" not in df.columns:
            df = df.rename(columns={"real_volume": "volume"})

        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.set_index("time")
        df.index.name = None

        cols = ["open", "high", "low", "close", "volume"]
        if include_spread and "spread" in df.columns:
            cols.append("spread")
        df = df[[c for c in cols if c in df.columns]]

        # Datums-Filter (relevant wenn Fallback copy_rates_from_pos genutzt wurde)
        df = df.loc[(df.index >= start_utc) & (df.index <= end_utc)]

        if df.empty:
            raise MT5DataError(
                f"Keine Daten fuer {symbol} {tf_str} im Zeitraum "
                f"{start_utc.date()} – {end_utc.date()} "
                f"(Broker haelt fruehestens ab {df.index.min() if not df.empty else 'N/A'})."
            )

        logger.info(
            "MT5 OHLCV | symbol={symbol} tf={tf} | {n} Candles | {s} – {e}",
            symbol=symbol, tf=tf_str, n=len(df),
            s=df.index.min().date(), e=df.index.max().date(),
        )
        return df

    def get_ohlcv_count(
        self,
        symbol: str,
        timeframe: str | Timeframe,
        count: int = 100,
        start_pos: int = 0,
    ) -> pd.DataFrame:
        """
        Liefert die letzten `count` OHLCV-Kerzen ab Position `start_pos`.

        Parameters
        ----------
        symbol    : z.B. "EURUSD"
        timeframe : Timeframe-Enum oder String
        count     : Anzahl Kerzen
        start_pos : Startposition (0 = aktuellste Kerze)
        """
        if not self.is_connected:
            raise MT5ConnectionError("Nicht verbunden – rufe zuerst connect() auf.")

        mt5 = _load_mt5()
        tf_str = timeframe.value if isinstance(timeframe, Timeframe) else str(timeframe).upper()

        if tf_str not in _TIMEFRAME_MAP:
            raise MT5DataError(f"Ungueltiger Timeframe: '{tf_str}'.")

        rates = mt5.copy_rates_from_pos(symbol, _TIMEFRAME_MAP[tf_str], start_pos, count)

        if rates is None or len(rates) == 0:
            err = mt5.last_error()
            raise MT5DataError(f"Keine Daten fuer {symbol} {tf_str} | {err}")

        df = pd.DataFrame(rates)
        df = df.rename(columns={"tick_volume": "volume"})
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.set_index("time")
        df.index.name = None

        cols = ["open", "high", "low", "close", "volume"]
        df = df[[c for c in cols if c in df.columns]]
        return df

    def get_symbol_info(self, symbol: str) -> dict:
        """
        Gibt Broker-Informationen zu einem Symbol zurueck.

        Returns dict mit: point, digits, spread, swap_long, swap_short, contract_size
        """
        if not self.is_connected:
            raise MT5ConnectionError("Nicht verbunden.")

        mt5 = _load_mt5()
        info = mt5.symbol_info(symbol)

        if info is None:
            err = mt5.last_error()
            raise MT5DataError(f"Symbol nicht gefunden: {symbol} | {err}")

        return {
            "point":         info.point,
            "digits":        info.digits,
            "spread":        info.spread,
            "stops_level":   read_stops_level(info),
            "swap_long":     info.swap_long,
            "swap_short":    info.swap_short,
            "contract_size": info.trade_contract_size,
        }

    def get_tick(self, symbol: str) -> dict:
        """
        Gibt den aktuellen Bid/Ask-Kurs fuer ein Symbol zurueck.

        Returns dict mit: bid, ask
        """
        if not self.is_connected:
            raise MT5ConnectionError("Nicht verbunden.")

        mt5 = _load_mt5()
        tick = mt5.symbol_info_tick(symbol)

        if tick is None:
            err = mt5.last_error()
            raise MT5DataError(f"Kein Tick fuer {symbol} | {err}")

        return {"bid": float(tick.bid), "ask": float(tick.ask)}

    def get_account_info(self) -> dict:
        """
        Gibt Kontoinformationen zurueck.

        Returns dict mit: login, name, server, balance, equity,
                          currency, leverage, is_demo
        """
        if not self.is_connected:
            raise MT5ConnectionError("Nicht verbunden.")

        mt5 = _load_mt5()
        acc = mt5.account_info()
        if acc is None:
            raise MT5DataError(
                f"account_info() fehlgeschlagen: {mt5.last_error()}"
            )

        demo_mode = getattr(mt5, "ACCOUNT_TRADE_MODE_DEMO", 0)
        return {
            "login":    acc.login,
            "name":     acc.name,
            "server":   acc.server,
            "balance":  acc.balance,
            "equity":   acc.equity,
            "currency": acc.currency,
            "leverage": acc.leverage,
            "is_demo":  acc.trade_mode == demo_mode,
        }

    def get_available_symbols(self) -> list[str]:
        """Gibt alle vom Broker angebotenen Symbole zurueck."""
        if not self.is_connected:
            raise MT5ConnectionError("Nicht verbunden.")

        mt5 = _load_mt5()
        symbols = mt5.symbols_get()
        if symbols is None:
            return []
        return [s.name for s in symbols]
