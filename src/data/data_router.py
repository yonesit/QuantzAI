"""
src/data/data_router.py
DataRouter – waehlt automatisch den verfuegbaren Connector.

Logik:
  1. MT5 verbunden  → MT5Connector verwenden
  2. MT5 nicht erreichbar + Preisabgleich OK (<= 5 Pips Abweichung)
     → OANDAConnector als Fallback
  3. MT5 nicht erreichbar + Preisabweichung > 5 Pips
     → PriceDiscrepancyError (Emergency Mode)

Jeder Connector-Wechsel wird im Audit-Log festgehalten.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import pandas as pd
from loguru import logger


# ─────────────────────────────────────────────
#  Exceptions
# ─────────────────────────────────────────────

class PriceDiscrepancyError(Exception):
    """MT5 und OANDA-Preise weichen zu stark ab – Fallback nicht sicher."""


class EmergencyModeError(Exception):
    """Kein Connector verfuegbar und Preisabgleich nicht bestanden."""


# ─────────────────────────────────────────────
#  PriceValidator
# ─────────────────────────────────────────────

class PriceValidator:
    """
    Vergleicht den letzten Schlusskurs von MT5 und OANDA.

    Parameters
    ----------
    max_pips : Maximale erlaubte Abweichung in Pips (Standard: 5)
    pip_size : Pip-Groesse (Standard: 0.0001 fuer die meisten Forex-Paare)
    """

    def __init__(self, max_pips: float = 5.0, pip_size: float = 0.0001) -> None:
        self._max_pips = max_pips
        self._pip_size = pip_size

    def check(
        self,
        symbol: str,
        mt5_price: float,
        oanda_price: float,
    ) -> bool:
        """
        Gibt True zurueck wenn die Preisabweichung akzeptabel ist.

        Parameters
        ----------
        symbol      : Symbol-Name (nur fuer Logging)
        mt5_price   : Letzter MT5-Schlusskurs
        oanda_price : Letzter OANDA-Schlusskurs

        Returns
        -------
        bool: True = Abweichung ok, False = zu gross
        """
        diff_pips = abs(mt5_price - oanda_price) / self._pip_size

        logger.debug(
            "PriceValidator | {symbol} | MT5={mt5:.5f} OANDA={oanda:.5f} | diff={diff:.1f} pips",
            symbol=symbol,
            mt5=mt5_price,
            oanda=oanda_price,
            diff=diff_pips,
        )

        if diff_pips > self._max_pips:
            logger.error(
                "PriceDiscrepancy | {symbol} | diff={diff:.1f} pips > max={max} pips",
                symbol=symbol,
                diff=diff_pips,
                max=self._max_pips,
            )
            return False

        return True


# ─────────────────────────────────────────────
#  DataRouter
# ─────────────────────────────────────────────

class DataRouter:
    """
    Waehlt automatisch den verfuegbaren Connector.

    Parameters
    ----------
    mt5   : MT5Connector-Instanz
    oanda : OANDAConnector-Instanz
    validator       : PriceValidator (optional, wird erstellt falls None)
    audit_log_path  : Pfad fuer Audit-Log (optional)
    """

    def __init__(
        self,
        mt5,
        oanda,
        validator: Optional[PriceValidator] = None,
        audit_log_path: Optional[str] = None,
    ) -> None:
        self._mt5      = mt5
        self._oanda    = oanda
        self._validator = validator or PriceValidator()
        self._audit_log = audit_log_path
        self._active_connector_name: str = "none"

    # ── Hauptmethode ─────────────────────────────────

    def get_connector(self, symbol: Optional[str] = None):
        """
        Gibt den besten verfuegbaren Connector zurueck.

        Parameters
        ----------
        symbol : Symbol fuer den Preisabgleich (optional).
                 Wenn angegeben, wird bei MT5-Ausfall der Preis verglichen.

        Returns
        -------
        MT5Connector oder OANDAConnector

        Raises
        ------
        EmergencyModeError : Kein Connector verfuegbar oder Preisabweichung zu gross.
        """
        # 1. MT5 verfuegbar?
        if self._mt5.is_connected:
            if self._active_connector_name != "mt5":
                self._log_switch("mt5", "MT5 verbunden")
            return self._mt5

        # 2. OANDA verfuegbar?
        if not self._oanda.is_connected:
            raise EmergencyModeError(
                "Kein Connector verfuegbar: MT5 und OANDA nicht verbunden."
            )

        # 3. Preisabgleich wenn Symbol angegeben
        if symbol is not None:
            if not self._price_check_passes(symbol):
                raise PriceDiscrepancyError(
                    f"Preisabweichung zu gross fuer {symbol} – "
                    f"Fallback auf OANDA nicht sicher."
                )

        # 4. OANDA als Fallback
        if self._active_connector_name != "oanda":
            reason = "MT5 nicht erreichbar"
            if symbol:
                reason += f", Preisabgleich OK ({symbol})"
            self._log_switch("oanda", reason)

        return self._oanda

    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """OHLCV-Daten ueber den besten verfuegbaren Connector."""
        connector = self.get_connector(symbol=symbol)
        return connector.get_ohlcv(symbol, timeframe, start, end)

    def get_ohlcv_count(
        self,
        symbol: str,
        timeframe: str,
        count: int = 100,
    ) -> pd.DataFrame:
        """Letzte N Kerzen ueber den besten verfuegbaren Connector."""
        connector = self.get_connector(symbol=symbol)
        return connector.get_ohlcv_count(symbol, timeframe, count)

    def get_symbol_info(self, symbol: str) -> dict:
        """Symbol-Info ueber den besten verfuegbaren Connector."""
        connector = self.get_connector(symbol=symbol)
        return connector.get_symbol_info(symbol)

    @property
    def active_connector_name(self) -> str:
        """Name des aktuell aktiven Connectors: 'mt5', 'oanda' oder 'none'."""
        return self._active_connector_name

    # ── Intern ───────────────────────────────────────

    def _price_check_passes(self, symbol: str) -> bool:
        """Holt Preise von beiden Connectoren und validiert die Abweichung."""
        try:
            # MT5-Preis: letzter Schlusskurs aus 1 Kerze
            from datetime import timedelta
            end   = datetime.now(timezone.utc)
            start = end - timedelta(minutes=5)
            mt5_df = self._mt5.get_ohlcv(symbol, "M1", start, end)
            mt5_price = float(mt5_df["close"].iloc[-1])
        except Exception as exc:
            logger.warning("PriceCheck: MT5-Preis nicht abrufbar | {exc}", exc=exc)
            # Wenn MT5 nicht erreichbar ist kann kein Preisabgleich gemacht werden
            return False

        try:
            oanda_price = self._oanda.get_latest_price(symbol)
        except Exception as exc:
            logger.warning("PriceCheck: OANDA-Preis nicht abrufbar | {exc}", exc=exc)
            return False

        return self._validator.check(symbol, mt5_price, oanda_price)

    def _log_switch(self, connector_name: str, reason: str) -> None:
        """Schreibt einen Connector-Wechsel ins Audit-Log."""
        timestamp = datetime.now(timezone.utc).isoformat()
        old = self._active_connector_name
        self._active_connector_name = connector_name

        entry = (
            f"[{timestamp}] CONNECTOR_SWITCH "
            f"{old.upper()} -> {connector_name.upper()} | Grund: {reason}"
        )

        logger.info(entry)

        if self._audit_log:
            try:
                with open(self._audit_log, "a", encoding="utf-8") as f:
                    f.write(entry + "\n")
            except OSError as exc:
                logger.error("Audit-Log schreiben fehlgeschlagen: {exc}", exc=exc)
