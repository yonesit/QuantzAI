"""
src/risk/position_sizer.py
PositionSizer – ATR-basierte, dynamische Positionsgroessen-Berechnung.

Kein fester Lot-Wert: die Groesse ergibt sich aus Kontostand, Risiko pro
Trade und aktueller Volatilitaet (ATR).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from loguru import logger


# ─────────────────────────────────────────────
#  Exceptions
# ─────────────────────────────────────────────

class PositionSizeError(Exception):
    """Position kann nicht sinnvoll berechnet werden (z.B. zu klein)."""


# ─────────────────────────────────────────────
#  Ergebnis-Datenklasse
# ─────────────────────────────────────────────

@dataclass
class PositionSizeResult:
    symbol: str
    lot_size: float
    risk_amount: float
    stop_loss_distance: float
    is_valid: bool
    rejection_reason: Optional[str] = None


# ─────────────────────────────────────────────
#  PositionSizer
# ─────────────────────────────────────────────

class PositionSizer:
    """
    Berechnet die Positionsgroesse (Lot-Size) basierend auf:
      risk_amount = account_balance * risk_pct
      stop_loss_distance = atr * sl_atr_multiplier
      position_size = risk_amount / (stop_loss_distance * pip_value)

    Parameters
    ----------
    risk_per_trade_pct : Risiko pro Trade in Prozent des Kontostands (Standard: 1.0)
    sl_atr_multiplier   : Multiplikator fuer Stop-Loss-Distanz aus ATR (Standard: 1.5)
    min_lot_size        : Mindest-Lot-Groesse (Standard: 0.01)
    """

    def __init__(
        self,
        risk_per_trade_pct: float = 1.0,
        sl_atr_multiplier: float = 1.5,
        min_lot_size: float = 0.01,
    ) -> None:
        self._risk_pct        = risk_per_trade_pct
        self._sl_multiplier    = sl_atr_multiplier
        self._min_lot_size     = min_lot_size

    def calculate_lot_size(
        self,
        account_balance: float,
        atr: float,
        symbol: str,
        risk_pct: Optional[float] = None,
        pip_value: float = 10.0,
        pip_size: float = 0.0001,
        lot_step: float = 0.01,
        contract_size: float = 100_000,
    ) -> PositionSizeResult:
        """
        Berechnet die Lot-Groesse fuer einen Trade.

        Parameters
        ----------
        account_balance : Aktueller Kontostand
        atr              : ATR(14) des Symbols (in Preiseinheiten, nicht Pips)
        symbol           : Symbol-Name (fuer Logging/Ergebnis)
        risk_pct         : Ueberschreibt die Standard-Risiko-Prozentzahl (optional)
        pip_value        : Wert eines Pips pro Standard-Lot in Kontowaehrung (Standard: 10 USD)
        pip_size         : Pip-Groesse des Symbols (Standard: 0.0001 fuer die meisten Forex-Paare)
        lot_step         : Minimaler Lot-Schritt des Brokers (aus MT5 symbol_info)
        contract_size    : Kontraktgroesse pro Standard-Lot (Standard: 100000)

        Returns
        -------
        PositionSizeResult: enthaelt lot_size=0.0 und is_valid=False wenn die
        berechnete Groesse unter min_lot_size liegt (Trade wird abgelehnt,
        NICHT aufgerundet).
        """
        effective_risk_pct = risk_pct if risk_pct is not None else self._risk_pct

        if account_balance <= 0:
            return self._rejected(symbol, "Kontostand muss positiv sein.")

        if atr <= 0:
            return self._rejected(symbol, "ATR muss positiv sein (ATR=0 verhindert Berechnung).")

        if effective_risk_pct <= 0:
            return self._rejected(symbol, "Risiko-Prozentsatz muss positiv sein.")

        risk_amount = account_balance * (effective_risk_pct / 100.0)
        stop_loss_distance = atr * self._sl_multiplier

        # Stop-Loss-Distanz in Pips umrechnen
        sl_distance_pips = stop_loss_distance / pip_size

        if sl_distance_pips <= 0:
            return self._rejected(symbol, "Stop-Loss-Distanz in Pips ist 0 oder negativ.")

        raw_lot_size = risk_amount / (sl_distance_pips * pip_value)

        # Auf Symbol-Lot-Step runden (immer ABRUNDEN, nie aufrunden -
        # Risiko darf nie ueberschritten werden)
        rounded_lot_size = math.floor(raw_lot_size / lot_step) * lot_step
        rounded_lot_size = round(rounded_lot_size, 8)  # Floating-Point-Reste vermeiden

        if rounded_lot_size < self._min_lot_size:
            logger.warning(
                "PositionSizer: Lot-Groesse {raw:.5f} unter Minimum {min} | "
                "Symbol={symbol} -> Trade abgelehnt (kein Aufrunden!)",
                raw=raw_lot_size, min=self._min_lot_size, symbol=symbol,
            )
            return self._rejected(
                symbol,
                f"Berechnete Lot-Groesse ({rounded_lot_size:.5f}) unter Mindestgroesse "
                f"({self._min_lot_size}). Risiko zu klein fuer sinnvollen Trade.",
                risk_amount=risk_amount,
                stop_loss_distance=stop_loss_distance,
            )

        logger.info(
            "PositionSizer: {symbol} | lot={lot} | risk={risk:.2f} | sl_dist={sl:.5f}",
            symbol=symbol, lot=rounded_lot_size, risk=risk_amount, sl=stop_loss_distance,
        )

        return PositionSizeResult(
            symbol=symbol,
            lot_size=rounded_lot_size,
            risk_amount=risk_amount,
            stop_loss_distance=stop_loss_distance,
            is_valid=True,
        )

    @staticmethod
    def _rejected(
        symbol: str,
        reason: str,
        risk_amount: float = 0.0,
        stop_loss_distance: float = 0.0,
    ) -> PositionSizeResult:
        return PositionSizeResult(
            symbol=symbol,
            lot_size=0.0,
            risk_amount=risk_amount,
            stop_loss_distance=stop_loss_distance,
            is_valid=False,
            rejection_reason=reason,
        )
