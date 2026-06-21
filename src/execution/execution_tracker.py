"""
src/execution/execution_tracker.py
ExecutionTracker – Slippage- und Gebuehren-Tracking fuer ausgefuehrte Orders.

Erfasst pro Trade:
  - Slippage: Differenz zwischen erwartetem und tatsaechlichem Ausfuehrungspreis
  - Fees: Spread, Commission und Swap (aus MT5 oder manuell gesetzt)

Vergleichsdashboard:
  - Durchschnittliche Slippage (letzte 100 Trades) vs. Backtest-Annahmen
  - Durchschnittliche Gebuehren (letzte 100 Trades) vs. Backtest-Annahmen
  - Warnung bei signifikanter Abweichung (konfigurierbar, Standard: 50%)

Positive Slippage = schlechtere Ausfuehrung als erwartet (haeufig im Livebetrieb).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from loguru import logger


# ─────────────────────────────────────────────────────────────────────────────
#  Datenklassen
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SlippageRecord:
    """Slippage einer einzelnen Order."""
    ticket:          int
    symbol:          str
    direction:       str           # "buy" | "sell"
    expected_price:  float
    actual_price:    float
    slippage_pips:   float         # positiv = schlechter als erwartet
    timestamp:       datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


@dataclass
class FeeRecord:
    """Tatsaechliche Gebuehren einer einzelnen Order."""
    ticket:      int
    symbol:      str
    spread:      float             # in Preis-Einheiten
    commission:  float
    swap:        float = 0.0
    timestamp:   datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @property
    def total_fees(self) -> float:
        return self.spread + self.commission + self.swap


# ─────────────────────────────────────────────────────────────────────────────
#  ExecutionTracker
# ─────────────────────────────────────────────────────────────────────────────

class ExecutionTracker:
    """
    Erfasst Slippage und Gebuehren pro Order und vergleicht sie mit
    den Backtest-Annahmen.

    Parameters
    ----------
    pip_size : Pip-Groesse des Instruments (Standard: 0.0001 fuer Majors).
               Wird fuer Slippage-Berechnung in Pips verwendet.
    """

    def __init__(self, pip_size: float = 0.0001) -> None:
        if pip_size <= 0:
            raise ValueError("pip_size muss positiv sein.")
        self._pip_size = pip_size
        self._slippage_records: list[SlippageRecord] = []
        self._fee_records:      list[FeeRecord]      = []

    # ── Slippage ──────────────────────────────────────────────────────────────

    def record_slippage(
        self,
        ticket:         int,
        symbol:         str,
        direction:      str,
        expected_price: float,
        actual_price:   float,
    ) -> SlippageRecord:
        """
        Erfasst Slippage fuer eine ausgefuehrte Order.

        Buy:  positive Slippage = actual > expected (teurer als erwartet)
        Sell: positive Slippage = actual < expected (guenstiger als erwartet fuer Gegenpartei)
        """
        if direction == "buy":
            slippage_pips = (actual_price - expected_price) / self._pip_size
        else:
            slippage_pips = (expected_price - actual_price) / self._pip_size

        record = SlippageRecord(
            ticket=ticket,
            symbol=symbol,
            direction=direction,
            expected_price=expected_price,
            actual_price=actual_price,
            slippage_pips=slippage_pips,
        )
        self._slippage_records.append(record)
        logger.debug(
            "ExecutionTracker: Slippage ticket={t} {sym} {dir} "
            "expected={exp:.5f} actual={act:.5f} slippage={s:.2f} pips",
            t=ticket, sym=symbol, dir=direction,
            exp=expected_price, act=actual_price, s=slippage_pips,
        )
        return record

    # ── Gebuehren ────────────────────────────────────────────────────────────

    def record_fees(
        self,
        ticket:     int,
        symbol:     str,
        spread:     float,
        commission: float,
        swap:       float = 0.0,
    ) -> FeeRecord:
        """Erfasst tatsaechliche Gebuehren fuer eine ausgefuehrte Order."""
        record = FeeRecord(
            ticket=ticket,
            symbol=symbol,
            spread=spread,
            commission=commission,
            swap=swap,
        )
        self._fee_records.append(record)
        logger.debug(
            "ExecutionTracker: Fees ticket={t} {sym} "
            "spread={sp:.5f} commission={c:.5f} swap={sw:.5f} total={tot:.5f}",
            t=ticket, sym=symbol,
            sp=spread, c=commission, sw=swap, tot=record.total_fees,
        )
        return record

    # ── Abfragen ──────────────────────────────────────────────────────────────

    def get_slippage_records(self, n: int = 100) -> list[SlippageRecord]:
        """Gibt die letzten n Slippage-Datensaetze zurueck."""
        return self._slippage_records[-n:] if n > 0 else list(self._slippage_records)

    def get_fee_records(self, n: int = 100) -> list[FeeRecord]:
        """Gibt die letzten n Gebuehren-Datensaetze zurueck."""
        return self._fee_records[-n:] if n > 0 else list(self._fee_records)

    def get_avg_slippage_pips(self, n: int = 100) -> Optional[float]:
        """Durchschnittliche Slippage in Pips (letzte n Trades). None wenn keine Daten."""
        records = self.get_slippage_records(n)
        if not records:
            return None
        return sum(r.slippage_pips for r in records) / len(records)

    def get_avg_total_fees(self, n: int = 100) -> Optional[float]:
        """Durchschnittliche Gesamtgebuehren (letzte n Trades). None wenn keine Daten."""
        records = self.get_fee_records(n)
        if not records:
            return None
        return sum(r.total_fees for r in records) / len(records)

    # ── Vergleichs-Dashboard ──────────────────────────────────────────────────

    def compare_to_backtest(
        self,
        backtest_slippage_pips: float,
        backtest_fees:          float,
        n:                      int = 100,
    ) -> dict:
        """
        Vergleicht tatsaechliche Ausfuehrungsqualitaet mit Backtest-Annahmen.

        Returns
        -------
        dict mit:
          actual_avg_slippage_pips, backtest_slippage_pips, slippage_deviation_pct,
          actual_avg_fees, backtest_fees, fees_deviation_pct,
          n_slippage_trades, n_fee_trades
        """
        avg_slip  = self.get_avg_slippage_pips(n)
        avg_fees  = self.get_avg_total_fees(n)
        n_slip    = len(self.get_slippage_records(n))
        n_fees    = len(self.get_fee_records(n))

        slip_dev: Optional[float] = None
        if avg_slip is not None and backtest_slippage_pips != 0:
            slip_dev = ((avg_slip - backtest_slippage_pips) / abs(backtest_slippage_pips)) * 100.0

        fees_dev: Optional[float] = None
        if avg_fees is not None and backtest_fees != 0:
            fees_dev = ((avg_fees - backtest_fees) / abs(backtest_fees)) * 100.0

        return {
            "actual_avg_slippage_pips": avg_slip,
            "backtest_slippage_pips":   backtest_slippage_pips,
            "slippage_deviation_pct":   slip_dev,
            "actual_avg_fees":          avg_fees,
            "backtest_fees":            backtest_fees,
            "fees_deviation_pct":       fees_dev,
            "n_slippage_trades":        n_slip,
            "n_fee_trades":             n_fees,
        }

    def get_deviation_warning(
        self,
        backtest_slippage_pips: float,
        backtest_fees:          float,
        threshold_pct:          float = 50.0,
        n:                      int   = 100,
    ) -> Optional[str]:
        """
        Gibt eine Warnung zurueck wenn die tatsaechliche Ausfuehrung
        signifikant von den Backtest-Annahmen abweicht.

        Returns None wenn die Abweichung innerhalb der Toleranz liegt
        oder keine Daten vorhanden sind.
        """
        comparison = self.compare_to_backtest(backtest_slippage_pips, backtest_fees, n)
        parts: list[str] = []

        slip_dev = comparison["slippage_deviation_pct"]
        if slip_dev is not None and slip_dev > threshold_pct:
            parts.append(
                f"Slippage: {comparison['actual_avg_slippage_pips']:.1f} pips actual vs "
                f"{backtest_slippage_pips:.1f} pips backtest (+{slip_dev:.0f}%)"
            )

        fees_dev = comparison["fees_deviation_pct"]
        if fees_dev is not None and fees_dev > threshold_pct:
            parts.append(
                f"Fees: {comparison['actual_avg_fees']:.5f} actual vs "
                f"{backtest_fees:.5f} backtest (+{fees_dev:.0f}%)"
            )

        if parts:
            return "EXECUTION WARNING: " + " | ".join(parts)
        return None
