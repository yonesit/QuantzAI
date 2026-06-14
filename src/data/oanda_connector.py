"""
src/data/oanda_connector.py
OANDAConnector – REST-Fallback fuer MT5Connector.

Gleiche Schnittstelle wie MT5Connector:
  - connect() / disconnect() / is_connected
  - get_ohlcv(symbol, timeframe, start, end) -> pd.DataFrame  (UTC-Index)
  - get_ohlcv_count(symbol, timeframe, count) -> pd.DataFrame
  - get_symbol_info(symbol) -> dict
  - Context Manager (__enter__ / __exit__)

Symbol-Mapping: EURUSD -> EUR_USD  (aus config.yaml geladen)
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Optional

import requests
import pandas as pd
from loguru import logger


# ─────────────────────────────────────────────
#  Exceptions
# ─────────────────────────────────────────────

class OANDAConnectionError(Exception):
    """Verbindung zu OANDA fehlgeschlagen."""


class OANDADataError(Exception):
    """Datenabruf von OANDA fehlgeschlagen."""


# ─────────────────────────────────────────────
#  Konstanten
# ─────────────────────────────────────────────

_OANDA_DEMO_URL = "https://api-fxpractice.oanda.com/v3"
_OANDA_LIVE_URL = "https://api-fxtrade.oanda.com/v3"

# Timeframe-Mapping: QuantzAI -> OANDA Granularity
_TIMEFRAME_MAP: dict[str, str] = {
    "M1":  "M1",
    "M5":  "M5",
    "M15": "M15",
    "M30": "M30",
    "H1":  "H1",
    "H4":  "H4",
    "D1":  "D",
    "W1":  "W",
}

# Standard Symbol-Mapping (wird durch config.yaml ergaenzt)
_DEFAULT_SYMBOL_MAP: dict[str, str] = {
    "EURUSD": "EUR_USD",
    "GBPUSD": "GBP_USD",
    "USDJPY": "USD_JPY",
    "USDCHF": "USD_CHF",
    "AUDUSD": "AUD_USD",
    "NZDUSD": "NZD_USD",
    "USDCAD": "USD_CAD",
    "EURGBP": "EUR_GBP",
    "EURJPY": "EUR_JPY",
    "GBPJPY": "GBP_JPY",
    "XAUUSD": "XAU_USD",
    "XAGUSD": "XAG_USD",
}


# ─────────────────────────────────────────────
#  OANDAConnector
# ─────────────────────────────────────────────

class OANDAConnector:
    """
    REST-Connector fuer die OANDA v3 API.

    Parameters
    ----------
    api_key      : OANDA API-Token (aus .env: OANDA_API_KEY)
    account_id   : OANDA Account-ID (aus .env: OANDA_ACCOUNT_ID)
    demo         : True = fxpractice (Demo), False = fxtrade (Live)
    symbol_map   : Optionales Symbol-Mapping (ergaenzt den Standard)
    max_retries  : Anzahl Verbindungsversuche
    timeout      : HTTP-Timeout in Sekunden
    """

    def __init__(
        self,
        api_key: str,
        account_id: str,
        demo: bool = True,
        symbol_map: Optional[dict[str, str]] = None,
        max_retries: int = 3,
        timeout: int = 10,
    ) -> None:
        self._api_key    = api_key
        self._account_id = account_id
        self._base_url   = _OANDA_DEMO_URL if demo else _OANDA_LIVE_URL
        self._max_retries = max_retries
        self._timeout    = timeout
        self._connected  = False
        self._session: Optional[requests.Session] = None

        self._symbol_map = {**_DEFAULT_SYMBOL_MAP, **(symbol_map or {})}

    # ── Context Manager ──────────────────────────────

    def __enter__(self) -> "OANDAConnector":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.disconnect()

    # ── Verbindung ───────────────────────────────────

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> bool:
        """
        Oeffnet eine HTTP-Session und prueft die Verbindung
        mit einem leichten Account-Summary-Request.
        """
        for attempt in range(1, self._max_retries + 1):
            try:
                self._session = requests.Session()
                self._session.headers.update({
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type":  "application/json",
                    "Accept-Datetime-Format": "RFC3339",
                })

                url = f"{self._base_url}/accounts/{self._account_id}/summary"
                resp = self._session.get(url, timeout=self._timeout)

                if resp.status_code == 200:
                    self._connected = True
                    logger.info(
                        "OANDA connected | account={account} demo={demo}",
                        account=self._account_id,
                        demo=(self._base_url == _OANDA_DEMO_URL),
                    )
                    return True

                logger.warning(
                    "OANDA connect attempt {attempt}/{max} | status={status}",
                    attempt=attempt,
                    max=self._max_retries,
                    status=resp.status_code,
                )

            except requests.RequestException as exc:
                logger.warning(
                    "OANDA connect attempt {attempt}/{max} | error={exc}",
                    attempt=attempt,
                    max=self._max_retries,
                    exc=exc,
                )

            if attempt < self._max_retries:
                time.sleep(2 ** (attempt - 1))

        raise OANDAConnectionError(
            f"OANDA-Verbindung nach {self._max_retries} Versuchen gescheitert."
        )

    def disconnect(self) -> None:
        """Schliesst die HTTP-Session."""
        if self._session:
            self._session.close()
            self._session = None
        self._connected = False
        logger.info("OANDA disconnected | account={account}", account=self._account_id)

    # ── Symbol-Mapping ───────────────────────────────

    def map_symbol(self, symbol: str) -> str:
        """Konvertiert MT5-Symbol (EURUSD) in OANDA-Format (EUR_USD)."""
        mapped = self._symbol_map.get(symbol.upper())
        if mapped is None:
            # Fallback: falls schon im OANDA-Format
            if "_" in symbol:
                return symbol.upper()
            raise OANDADataError(
                f"Kein Symbol-Mapping fuer '{symbol}'. "
                f"Bitte in config.yaml eintragen."
            )
        return mapped

    # ── Daten ────────────────────────────────────────

    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """
        OHLCV-Daten fuer einen Zeitraum abrufen.

        Parameters
        ----------
        symbol    : MT5-Symbol z.B. "EURUSD"
        timeframe : "M1", "M5", "H1", "D1", ...
        start     : Startzeit (naive = UTC)
        end       : Endzeit   (naive = UTC)

        Returns
        -------
        pd.DataFrame  UTC-DatetimeIndex, Spalten: open, high, low, close, volume
        """
        if not self.is_connected:
            raise OANDAConnectionError("Nicht verbunden.")

        tf = timeframe.upper()
        if tf not in _TIMEFRAME_MAP:
            raise OANDADataError(f"Ungueltiger Timeframe: '{tf}'.")

        oanda_symbol = self.map_symbol(symbol)
        granularity  = _TIMEFRAME_MAP[tf]

        start_utc = start.replace(tzinfo=timezone.utc) if start.tzinfo is None else start
        end_utc   = end.replace(tzinfo=timezone.utc)   if end.tzinfo   is None else end

        url = f"{self._base_url}/instruments/{oanda_symbol}/candles"
        params = {
            "granularity": granularity,
            "from":        start_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "to":          end_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "price":       "M",   # Mid-Preise
        }

        resp = self._session.get(url, params=params, timeout=self._timeout)

        if resp.status_code != 200:
            raise OANDADataError(
                f"OANDA API Fehler {resp.status_code}: {resp.text[:200]}"
            )

        return self._parse_candles(resp.json())

    def get_ohlcv_count(
        self,
        symbol: str,
        timeframe: str,
        count: int = 100,
        start_pos: int = 0,
    ) -> pd.DataFrame:
        """
        Die letzten `count` Kerzen abrufen.

        Parameters
        ----------
        symbol    : MT5-Symbol z.B. "EURUSD"
        timeframe : "M1", "H1", ...
        count     : Anzahl Kerzen (max 5000 bei OANDA)
        start_pos : wird ignoriert (OANDA zaehlt von aktuell rueckwaerts)
        """
        if not self.is_connected:
            raise OANDAConnectionError("Nicht verbunden.")

        tf = timeframe.upper()
        if tf not in _TIMEFRAME_MAP:
            raise OANDADataError(f"Ungueltiger Timeframe: '{tf}'.")

        oanda_symbol = self.map_symbol(symbol)
        granularity  = _TIMEFRAME_MAP[tf]

        url = f"{self._base_url}/instruments/{oanda_symbol}/candles"
        params = {
            "granularity": granularity,
            "count":       min(count, 5000),
            "price":       "M",
        }

        resp = self._session.get(url, params=params, timeout=self._timeout)

        if resp.status_code != 200:
            raise OANDADataError(
                f"OANDA API Fehler {resp.status_code}: {resp.text[:200]}"
            )

        return self._parse_candles(resp.json())

    def get_symbol_info(self, symbol: str) -> dict:
        """
        Instrument-Informationen abrufen.

        Returns dict mit: point, digits, spread, swap_long, swap_short
        (spread/swap werden geschaetzt da OANDA keine direkten Werte liefert)
        """
        if not self.is_connected:
            raise OANDAConnectionError("Nicht verbunden.")

        oanda_symbol = self.map_symbol(symbol)
        url = f"{self._base_url}/instruments/{oanda_symbol}"
        params = {"fields": "displayPrecision,pipLocation,marginRate"}

        resp = self._session.get(url, params=params, timeout=self._timeout)

        if resp.status_code != 200:
            raise OANDADataError(
                f"Symbol nicht gefunden: {symbol} | {resp.status_code}"
            )

        data = resp.json().get("instrument", {})
        digits      = data.get("displayPrecision", 5)
        pip_location = data.get("pipLocation", -4)

        return {
            "point":      10 ** pip_location,
            "digits":     digits,
            "spread":     None,   # OANDA liefert keinen festen Spread
            "swap_long":  None,   # In OANDA: Financing-Rates, separat abrufbar
            "swap_short": None,
        }

    def get_latest_price(self, symbol: str) -> float:
        """
        Letzten Mid-Schlusskurs abrufen (fuer PriceValidator).

        Returns
        -------
        float: letzter Close-Preis
        """
        if not self.is_connected:
            raise OANDAConnectionError("Nicht verbunden.")

        oanda_symbol = self.map_symbol(symbol)
        url = f"{self._base_url}/instruments/{oanda_symbol}/candles"
        params = {"granularity": "M1", "count": 1, "price": "M"}

        resp = self._session.get(url, params=params, timeout=self._timeout)

        if resp.status_code != 200:
            raise OANDADataError(f"Preis-Abruf fehlgeschlagen: {resp.status_code}")

        candles = resp.json().get("candles", [])
        if not candles:
            raise OANDADataError(f"Keine Preis-Daten fuer {symbol}.")

        return float(candles[-1]["mid"]["c"])

    # ── Intern ───────────────────────────────────────

    def _parse_candles(self, data: dict) -> pd.DataFrame:
        """Wandelt OANDA-JSON in einen normierten DataFrame um."""
        candles = data.get("candles", [])
        if not candles:
            raise OANDADataError("Keine Kerzen in der API-Antwort.")

        rows = []
        for c in candles:
            if not c.get("complete", True):
                continue
            mid = c.get("mid", {})
            rows.append({
                "time":   c["time"],
                "open":   float(mid.get("o", 0)),
                "high":   float(mid.get("h", 0)),
                "low":    float(mid.get("l", 0)),
                "close":  float(mid.get("c", 0)),
                "volume": int(c.get("volume", 0)),
            })

        if not rows:
            raise OANDADataError("Alle Kerzen unvollstaendig.")

        df = pd.DataFrame(rows)
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.set_index("time")
        df.index.name = None
        return df
