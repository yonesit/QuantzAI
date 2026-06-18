"""
src/risk/pre_trade_check.py
PreTradeCheck – kombiniert Spread-Filter und EconomicCalendar-No-Trade-Zone
als abschliessende Vortrade-Pruefung vor jedem Orderversuch.

Schnittstellen:
  EconomicCalendar.is_no_trade_zone(symbol) -> bool
  MT5Connector.get_symbol_info(symbol)       -> {"spread": int (points),
                                                 "point": float, ...}

Spread-Umrechnung:
  spread_pips = spread_points * point / pip_size
  Standard pip_size=0.0001 (Majors ohne JPY).
  Fuer JPY-Paare (USDJPY, GBPJPY …) pip_size=0.01 uebergeben.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from src.data.calendar import EconomicCalendar
    from src.data.mt5_connector import MT5Connector


class PreTradeCheck:
    """
    Finale Vortrade-Pruefung: Spread-Filter + News-No-Trade-Zone.

    Reihenfolge:
      1. EconomicCalendar: No-Trade-Zone aktiv? -> sofortiger Block
      2. MT5Connector: aktueller Spread > Schwellwert? -> Block

    Parameters
    ----------
    calendar        : EconomicCalendar-Instanz (bereits initialisiert/gecacht)
    connector       : MT5Connector-Instanz (muss verbunden sein)
    max_spread_pips : Maximaler erlaubter Spread in Pips (Standard: 3.0)
    pip_size        : Pip-Groesse des Instruments (Standard: 0.0001 fuer Majors)
                      JPY-Paare: 0.01 uebergeben
    """

    def __init__(
        self,
        calendar: "EconomicCalendar",
        connector: "MT5Connector",
        max_spread_pips: float = 3.0,
        pip_size: float = 0.0001,
    ) -> None:
        self._calendar = calendar
        self._connector = connector
        self._max_spread_pips = max_spread_pips
        self._pip_size = pip_size

    def is_safe_to_trade(self, symbol: str) -> tuple[bool, str]:
        """
        Prueft ob ein Trade fuer das Symbol unter aktuellen Bedingungen
        sicher (erlaubt) ist.

        Parameters
        ----------
        symbol : z.B. "EURUSD"

        Returns
        -------
        tuple[bool, str]:
          (True,  "Trade erlaubt …")             – alle Checks bestanden
          (False, "No-Trade-Zone aktiv …")        – Kalender blockiert
          (False, "Spread zu hoch: X.X Pips …")  – Spread blockiert
        """
        # ── 1. Kalender-Pruefung ────────────────────────
        if self._calendar.is_no_trade_zone(symbol):
            reason = (
                f"No-Trade-Zone aktiv fuer {symbol} "
                f"(High-Impact-Event im Zeitfenster)"
            )
            logger.warning("PreTradeCheck: {reason}", reason=reason)
            return False, reason

        # ── 2. Spread-Pruefung ──────────────────────────
        info = self._connector.get_symbol_info(symbol)
        spread_pips = self._spread_to_pips(info["spread"], info["point"])

        if spread_pips > self._max_spread_pips:
            reason = (
                f"Spread zu hoch: {spread_pips:.1f} Pips "
                f"(Limit: {self._max_spread_pips:.1f} Pips) fuer {symbol}"
            )
            logger.warning("PreTradeCheck: {reason}", reason=reason)
            return False, reason

        reason = (
            f"Trade erlaubt fuer {symbol} "
            f"(Spread: {spread_pips:.1f} Pips, kein News-Fenster)"
        )
        logger.debug("PreTradeCheck: {reason}", reason=reason)
        return True, reason

    def _spread_to_pips(self, spread_points: int, point: float) -> float:
        """Rechnet Spread von MT5-Points in Pips um."""
        if self._pip_size <= 0:
            return 0.0
        return spread_points * point / self._pip_size
