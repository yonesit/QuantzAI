"""
src/risk/pre_trade_check.py
PreTradeCheck – kombiniert Spread-Filter und EconomicCalendar-No-Trade-Zone
als abschliessende Vortrade-Pruefung vor jedem Orderversuch.

Schnittstellen:
  EconomicCalendar.is_no_trade_zone(symbol) -> bool
  MT5Connector.get_symbol_info(symbol)       -> {"spread": int (points),
                                                 "point": float, ...}

Spread-Umrechnung (pro Symbol konfigurierbar):
  spread_pips = spread_points * point / pip_size

  Default:  pip_size=0.0001 (5-stellige Majors: EURUSD, GBPUSD …)
  USDJPY:   pip_size=0.01   (3-stellige JPY-Paare)
  XAUUSD:   pip_size=0.01   (2-stellige Gold-Notierung, point=0.01)

  Das Spread-Limit ist ebenfalls pro Symbol konfigurierbar:
  XAUUSD benoetigt ein hoeheres Limit als Forex-Paare (typisch 50-100 Pips = 0.50-1.00 USD).

  Uebergabe ueber symbol_overrides:
    symbol_overrides = {
        "XAUUSD": {"pip_size": 0.01, "max_spread_pips": 100.0},
        "USDJPY": {"pip_size": 0.01, "max_spread_pips": 3.0},
    }
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
    calendar         : EconomicCalendar-Instanz (bereits initialisiert/gecacht)
    connector        : MT5Connector-Instanz (muss verbunden sein)
    max_spread_pips  : Standard-Spread-Limit in Pips fuer alle Symbole (Standard: 3.0)
    pip_size         : Standard-Pip-Groesse fuer alle Symbole (Standard: 0.0001 fuer Majors)
    symbol_overrides : Pro-Symbol-Ueberschreibungen fuer pip_size und/oder max_spread_pips.
                       Fehlende Schluesse werden mit den Standardwerten befuellt.
                       Beispiel:
                         {"XAUUSD": {"pip_size": 0.01, "max_spread_pips": 100.0}}
    """

    def __init__(
        self,
        calendar: "EconomicCalendar",
        connector: "MT5Connector",
        max_spread_pips: float = 3.0,
        pip_size: float = 0.0001,
        symbol_overrides: dict[str, dict] | None = None,
    ) -> None:
        self._calendar = calendar
        self._connector = connector
        self._max_spread_pips = max_spread_pips
        self._pip_size = pip_size
        self._symbol_overrides: dict[str, dict] = symbol_overrides or {}

    def is_safe_to_trade(self, symbol: str) -> tuple[bool, str]:
        """
        Prueft ob ein Trade fuer das Symbol unter aktuellen Bedingungen
        sicher (erlaubt) ist.

        Parameters
        ----------
        symbol : z.B. "EURUSD" oder "XAUUSD"

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

        # ── 2. Spread-Pruefung (per Symbol konfigurierbar) ─────────────
        pip_size, max_spread_pips = self._get_symbol_params(symbol)
        info = self._connector.get_symbol_info(symbol)
        spread_pips = self._spread_to_pips(info["spread"], info["point"], pip_size)

        if spread_pips > max_spread_pips:
            reason = (
                f"Spread zu hoch: {spread_pips:.1f} Pips "
                f"(Limit: {max_spread_pips:.1f} Pips) fuer {symbol}"
            )
            logger.warning("PreTradeCheck: {reason}", reason=reason)
            return False, reason

        reason = (
            f"Trade erlaubt fuer {symbol} "
            f"(Spread: {spread_pips:.1f} Pips, kein News-Fenster)"
        )
        logger.debug("PreTradeCheck: {reason}", reason=reason)
        return True, reason

    # ── Hilfsmethoden ─────────────────────────────────────────────────────────

    def _get_symbol_params(self, symbol: str) -> tuple[float, float]:
        """
        Gibt (pip_size, max_spread_pips) fuer ein Symbol zurueck.

        Symbol-spezifische Werte aus symbol_overrides haben Vorrang;
        fehlende Schluessel werden mit den Konstruktor-Defaults befuellt.
        """
        override = self._symbol_overrides.get(symbol, {})
        pip_size = override.get("pip_size", self._pip_size)
        max_spread_pips = override.get("max_spread_pips", self._max_spread_pips)
        return pip_size, max_spread_pips

    @staticmethod
    def _spread_to_pips(spread_points: int, point: float, pip_size: float) -> float:
        """Rechnet Spread von MT5-Points in Pips um."""
        if pip_size <= 0:
            return 0.0
        return spread_points * point / pip_size
