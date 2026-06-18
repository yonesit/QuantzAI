"""
tests/unit/test_risk_metrics.py
Unit-Tests fuer VaR, CVaR, Kelly-Kriterium und KellyPositionSizer.

Abgedeckt:
  calculate_var
    - Normalverteilung mit bekanntem theoretischem VaR (~1.645 bei N(0,1), 95%)
    - Gleichverteilung mit analytischem VaR
    - 99%-Konfidenz
    - Alle positiven Renditen -> VaR = 0
    - Leerer Array -> ValueError
    - Ungueltige Konfidenzniveaus -> ValueError

  calculate_cvar
    - N(0,1) 95%: CVaR > VaR (immer)
    - CVaR >= VaR fuer jede Verteilung
    - Gleiche Grenzwert-Faelle wie VaR

  calculate_kelly_fraction
    - Handberechnung: p=0.6, w=1.5, l=1.0 -> 1/3
    - Faire Muenze: p=0.5, w=1.0, l=1.0 -> 0.0
    - Unfavorable Spiel: p=0.4, w=1.0, l=2.0 -> negativ
    - Eingabe-Validierung: win_rate, avg_win, avg_loss

  portfolio_var
    - Skalierung mit Lot-Summe
    - Leere Positions -> 0.0
    - Positionen ohne lot_size-Key -> 0.0
    - Korrekte Weitergabe an calculate_var

  KellyPositionSizer
    - Schnittstellen-Kompatibilitaet mit PositionSizer
    - Half-Kelly: angewandte Fraktion = raw_kelly * 0.5
    - Negativer Kelly -> lot_size=0, is_valid=False
    - max_kelly_fraction Cap
    - calculate_lot_size: Lot-Berechnung mit Handwert
    - update_stats: Neuberechnung der Kelly-Fraktion
    - Validierung: ATR=0, balance<=0
    - Immer abgerundet (nie aufgerundet)
    - risk_pct-Parameter wird ignoriert
    - Gleiche Ergebnis-Struktur wie PositionSizer (PositionSizeResult)
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.risk.position_sizer import PositionSizeResult
from src.risk.risk_metrics import (
    KellyPositionSizer,
    calculate_cvar,
    calculate_kelly_fraction,
    calculate_var,
    portfolio_var,
)

# ─── Zufallsseed fuer reproduzierbare Tests ───────────────────────────────────
RNG = np.random.default_rng(42)


# ─────────────────────────────────────────────────────────────────────────────
#  calculate_var
# ─────────────────────────────────────────────────────────────────────────────

class TestCalculateVar:
    def test_normal_distribution_95_approx(self):
        """N(0,1)-VaR bei 95% Konfidenz ≈ 1.645 (theoretisch)."""
        returns = RNG.standard_normal(100_000)
        var = calculate_var(returns, confidence_level=0.95)
        # Toleranz 0.05 fuer endliche Stichprobe
        assert abs(var - 1.6449) < 0.05

    def test_normal_distribution_99(self):
        """N(0,1)-VaR bei 99% Konfidenz ≈ 2.326."""
        returns = RNG.standard_normal(100_000)
        var = calculate_var(returns, confidence_level=0.99)
        assert abs(var - 2.3263) < 0.05

    def test_uniform_distribution_analytical(self):
        """U(-1, 1)-VaR bei 95%: 5-Percentile = -0.9, also VaR=0.9."""
        returns = np.linspace(-1.0, 1.0, 10_001)   # exakt gleichverteilt
        var = calculate_var(returns, confidence_level=0.95)
        # 5-Percentile von U(-1,1) = -1 + 0.05*2 = -0.9  ->  VaR = 0.9
        assert abs(var - 0.9) < 0.01

    def test_all_positive_returns_var_zero(self):
        """Wenn alle Renditen positiv sind, gibt es keinen Verlust -> VaR=0."""
        returns = np.array([0.01, 0.02, 0.05, 0.10, 0.50])
        assert calculate_var(returns) == 0.0

    def test_all_negative_returns(self):
        """Alle Verluste -> VaR = Betrag des 5-Percentile."""
        returns = np.array([-0.1, -0.2, -0.3, -0.4, -0.5])
        var = calculate_var(returns, confidence_level=0.95)
        assert var > 0.0

    def test_returns_positive_number(self):
        """VaR muss immer eine nicht-negative Zahl sein."""
        returns = RNG.standard_normal(1000)
        assert calculate_var(returns) >= 0.0

    def test_higher_confidence_higher_var(self):
        """Hoehere Konfidenzstufe -> hoehrer VaR (bei typischer Verteilung)."""
        returns = RNG.standard_normal(10_000)
        var_95 = calculate_var(returns, 0.95)
        var_99 = calculate_var(returns, 0.99)
        assert var_99 >= var_95

    def test_single_element_returns(self):
        """Einzel-Rendite: VaR = max(0, -value)."""
        assert calculate_var(np.array([-0.05])) == pytest.approx(0.05)
        assert calculate_var(np.array([0.05])) == 0.0

    def test_scaled_returns(self):
        """Verdopplung der Renditen verdoppelt VaR."""
        returns = RNG.standard_normal(10_000)
        var1 = calculate_var(returns)
        var2 = calculate_var(returns * 2)
        assert abs(var2 - 2 * var1) < 1e-10

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="leer"):
            calculate_var(np.array([]))

    def test_confidence_zero_raises(self):
        with pytest.raises(ValueError, match="confidence_level"):
            calculate_var(np.array([0.1, -0.1]), confidence_level=0.0)

    def test_confidence_one_raises(self):
        with pytest.raises(ValueError, match="confidence_level"):
            calculate_var(np.array([0.1, -0.1]), confidence_level=1.0)

    def test_confidence_negative_raises(self):
        with pytest.raises(ValueError):
            calculate_var(np.array([0.1]), confidence_level=-0.1)

    def test_accepts_list_input(self):
        """Python-Listen als Eingabe muss funktionieren."""
        var = calculate_var([-0.05, -0.02, 0.01, 0.03])
        assert isinstance(var, float)

    def test_default_confidence_is_095(self):
        """Standardwert fuer confidence_level ist 0.95."""
        returns = np.array([-0.10, -0.05, 0.0, 0.05, 0.10])
        assert calculate_var(returns) == calculate_var(returns, 0.95)


# ─────────────────────────────────────────────────────────────────────────────
#  calculate_cvar
# ─────────────────────────────────────────────────────────────────────────────

class TestCalculateCVar:
    def test_normal_distribution_95_approx(self):
        """N(0,1)-CVaR bei 95% ≈ 2.063 (theoretisch)."""
        returns = RNG.standard_normal(100_000)
        cvar = calculate_cvar(returns, confidence_level=0.95)
        assert abs(cvar - 2.063) < 0.05

    def test_cvar_always_geq_var(self):
        """CVaR >= VaR fuer jede Verteilung (definitionsgemaess)."""
        returns = RNG.standard_normal(10_000)
        var  = calculate_var(returns, 0.95)
        cvar = calculate_cvar(returns, 0.95)
        assert cvar >= var - 1e-10

    def test_cvar_strictly_greater_for_fat_tails(self):
        """Fuer nicht-triviale Verlustverteilungen: CVaR > VaR."""
        returns = RNG.standard_normal(10_000)
        var  = calculate_var(returns, 0.95)
        cvar = calculate_cvar(returns, 0.95)
        assert cvar > var

    def test_cvar_at_99_greater_than_at_95(self):
        """Hoehere Konfidenzstufe -> hoehrer CVaR."""
        returns = RNG.standard_normal(10_000)
        cvar_95 = calculate_cvar(returns, 0.95)
        cvar_99 = calculate_cvar(returns, 0.99)
        assert cvar_99 >= cvar_95

    def test_all_positive_cvar_zero(self):
        returns = np.array([0.01, 0.02, 0.05, 0.10])
        assert calculate_cvar(returns) == 0.0

    def test_returns_non_negative(self):
        returns = RNG.standard_normal(1000)
        assert calculate_cvar(returns) >= 0.0

    def test_uniform_distribution(self):
        """U(-1, 1)-CVaR bei 95%: E[X | X <= -0.9] = -0.95  ->  CVaR = 0.95."""
        returns = np.linspace(-1.0, 1.0, 100_001)
        cvar = calculate_cvar(returns, 0.95)
        assert abs(cvar - 0.95) < 0.02

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="leer"):
            calculate_cvar(np.array([]))

    def test_confidence_invalid_raises(self):
        with pytest.raises(ValueError):
            calculate_cvar(np.array([0.1]), confidence_level=1.5)

    def test_accepts_list_input(self):
        cvar = calculate_cvar([-0.10, -0.05, 0.01, 0.05])
        assert isinstance(cvar, float)

    def test_default_confidence_is_095(self):
        returns = np.array([-0.10, -0.05, 0.0, 0.05])
        assert calculate_cvar(returns) == calculate_cvar(returns, 0.95)

    def test_scaled_returns(self):
        """Verdopplung der Renditen verdoppelt CVaR."""
        returns = RNG.standard_normal(10_000)
        cv1 = calculate_cvar(returns)
        cv2 = calculate_cvar(returns * 2)
        assert abs(cv2 - 2 * cv1) < 1e-10


# ─────────────────────────────────────────────────────────────────────────────
#  calculate_kelly_fraction
# ─────────────────────────────────────────────────────────────────────────────

class TestCalculateKellyFraction:
    def test_handcalculated_example(self):
        """p=0.6, w=1.5, l=1.0 -> f* = 1/3."""
        kelly = calculate_kelly_fraction(0.6, avg_win=1.5, avg_loss=1.0)
        assert kelly == pytest.approx(1 / 3, rel=1e-6)

    def test_fair_coin_zero_kelly(self):
        """Faire Muenze mit symmetrischem Payoff: f*=0."""
        kelly = calculate_kelly_fraction(0.5, avg_win=1.0, avg_loss=1.0)
        assert kelly == pytest.approx(0.0, abs=1e-10)

    def test_unfavorable_game_negative_kelly(self):
        """Unfavorables Spiel: p=0.4, w=1.0, l=2.0 -> f* < 0."""
        kelly = calculate_kelly_fraction(0.4, avg_win=1.0, avg_loss=2.0)
        assert kelly < 0.0
        assert kelly == pytest.approx(-0.8, rel=1e-6)

    def test_certain_win_kelly_one(self):
        """100% Gewinnrate -> f* = 1.0 (volles Kelly)."""
        kelly = calculate_kelly_fraction(1.0, avg_win=1.0, avg_loss=1.0)
        assert kelly == pytest.approx(1.0)

    def test_certain_loss_kelly_negative(self):
        """0% Gewinnrate -> f* < 0."""
        kelly = calculate_kelly_fraction(0.0, avg_win=1.0, avg_loss=1.0)
        assert kelly < 0.0

    def test_high_win_loss_ratio(self):
        """Grosses Win/Loss-Verhaeltnis ergibt hohe Kelly-Fraktion."""
        kelly = calculate_kelly_fraction(0.55, avg_win=3.0, avg_loss=1.0)
        assert kelly > 0.0

    def test_formula_symmetry(self):
        """f* = p - (1-p)/b, manuell nachgerechnet."""
        p, w, l = 0.55, 2.0, 1.0
        b = w / l
        expected = p - (1 - p) / b
        assert calculate_kelly_fraction(p, w, l) == pytest.approx(expected, rel=1e-9)

    def test_win_rate_negative_raises(self):
        with pytest.raises(ValueError, match="win_rate"):
            calculate_kelly_fraction(-0.1, 1.0, 1.0)

    def test_win_rate_above_one_raises(self):
        with pytest.raises(ValueError, match="win_rate"):
            calculate_kelly_fraction(1.1, 1.0, 1.0)

    def test_avg_win_zero_raises(self):
        with pytest.raises(ValueError, match="avg_win"):
            calculate_kelly_fraction(0.6, avg_win=0.0, avg_loss=1.0)

    def test_avg_win_negative_raises(self):
        with pytest.raises(ValueError, match="avg_win"):
            calculate_kelly_fraction(0.6, avg_win=-1.0, avg_loss=1.0)

    def test_avg_loss_zero_raises(self):
        with pytest.raises(ValueError, match="avg_loss"):
            calculate_kelly_fraction(0.6, avg_win=1.0, avg_loss=0.0)

    def test_avg_loss_negative_raises(self):
        with pytest.raises(ValueError, match="avg_loss"):
            calculate_kelly_fraction(0.6, avg_win=1.0, avg_loss=-0.5)

    def test_returns_float(self):
        kelly = calculate_kelly_fraction(0.6, 1.5, 1.0)
        assert isinstance(kelly, float)


# ─────────────────────────────────────────────────────────────────────────────
#  portfolio_var
# ─────────────────────────────────────────────────────────────────────────────

class TestPortfolioVar:
    def _returns(self, n=10_000) -> np.ndarray:
        return RNG.standard_normal(n) * 0.01  # std=1%

    def test_empty_positions_returns_zero(self):
        assert portfolio_var([], self._returns()) == 0.0

    def test_single_position_scales_var(self):
        """Einzel-Position mit lot=1.0: Portfolio-VaR = Unit-VaR."""
        returns = self._returns()
        positions = [{"symbol": "EURUSD", "lot_size": 1.0, "status": "open"}]
        unit_var = calculate_var(returns, 0.95)
        pvar = portfolio_var(positions, returns, 0.95)
        assert pvar == pytest.approx(unit_var, rel=1e-9)

    def test_two_positions_doubled_var(self):
        """Zwei Positionen mit je lot=1.0: Portfolio-VaR = 2 * Unit-VaR."""
        returns = self._returns()
        positions = [
            {"symbol": "EURUSD", "lot_size": 1.0},
            {"symbol": "GBPUSD", "lot_size": 1.0},
        ]
        unit_var = calculate_var(returns, 0.95)
        pvar = portfolio_var(positions, returns, 0.95)
        assert pvar == pytest.approx(2.0 * unit_var, rel=1e-9)

    def test_fractional_lots_scale_correctly(self):
        returns = self._returns()
        positions = [{"lot_size": 0.5}, {"lot_size": 0.3}]
        unit_var = calculate_var(returns, 0.95)
        pvar = portfolio_var(positions, returns, 0.95)
        assert pvar == pytest.approx(0.8 * unit_var, rel=1e-9)

    def test_positions_without_lot_size_key(self):
        """Positionen ohne lot_size-Schluessel zaehlen als 0."""
        returns = self._returns()
        positions = [{"symbol": "EURUSD", "status": "open"}]
        assert portfolio_var(positions, returns) == 0.0

    def test_result_non_negative(self):
        positions = [{"lot_size": 2.0}]
        pvar = portfolio_var(positions, self._returns())
        assert pvar >= 0.0

    def test_higher_confidence_higher_portfolio_var(self):
        returns = self._returns()
        positions = [{"lot_size": 1.0}]
        pvar_95 = portfolio_var(positions, returns, 0.95)
        pvar_99 = portfolio_var(positions, returns, 0.99)
        assert pvar_99 >= pvar_95

    def test_empty_returns_raises(self):
        positions = [{"lot_size": 1.0}]
        with pytest.raises(ValueError):
            portfolio_var(positions, np.array([]))

    def test_returns_float(self):
        positions = [{"lot_size": 1.0}]
        result = portfolio_var(positions, self._returns())
        assert isinstance(result, float)


# ─────────────────────────────────────────────────────────────────────────────
#  KellyPositionSizer
# ─────────────────────────────────────────────────────────────────────────────

class TestKellyPositionSizer:

    # ── Konstruktor & Kelly-Fraktion ──────────────────────────────────────────

    def test_half_kelly_default(self):
        """Standard-Multiplier 0.5 halbiert die rohe Kelly-Fraktion."""
        ks = KellyPositionSizer(win_rate=0.6, avg_win=1.5, avg_loss=1.0)
        raw = calculate_kelly_fraction(0.6, 1.5, 1.0)
        assert ks.kelly_fraction == pytest.approx(raw * 0.5, rel=1e-6)

    def test_full_kelly_multiplier_one(self):
        """kelly_multiplier=1.0 und max_kelly_fraction > raw: applied = raw Kelly."""
        ks = KellyPositionSizer(0.6, 1.5, 1.0, kelly_multiplier=1.0, max_kelly_fraction=1.0)
        raw = calculate_kelly_fraction(0.6, 1.5, 1.0)
        assert ks.kelly_fraction == pytest.approx(raw, rel=1e-6)

    def test_quarter_kelly(self):
        ks = KellyPositionSizer(0.6, 1.5, 1.0, kelly_multiplier=0.25)
        raw = calculate_kelly_fraction(0.6, 1.5, 1.0)
        assert ks.kelly_fraction == pytest.approx(raw * 0.25, rel=1e-6)

    def test_max_kelly_fraction_cap(self):
        """Kelly-Fraktion wird auf max_kelly_fraction gekappt."""
        ks = KellyPositionSizer(
            win_rate=0.9, avg_win=10.0, avg_loss=1.0,
            kelly_multiplier=1.0, max_kelly_fraction=0.10,
        )
        assert ks.kelly_fraction <= 0.10 + 1e-10

    def test_negative_raw_kelly_fraction_zero(self):
        """Negatives Kelly -> applied fraction = 0.0."""
        ks = KellyPositionSizer(
            win_rate=0.4, avg_win=1.0, avg_loss=2.0,
            kelly_multiplier=0.5,
        )
        assert ks.kelly_fraction == 0.0

    # ── calculate_lot_size – Grundlegendes ───────────────────────────────────

    def test_returns_position_size_result(self):
        ks = KellyPositionSizer(0.6, 1.5, 1.0)
        result = ks.calculate_lot_size(10_000, 0.002, "EURUSD")
        assert isinstance(result, PositionSizeResult)

    def test_valid_result_normal_inputs(self):
        ks = KellyPositionSizer(0.6, 1.5, 1.0)
        result = ks.calculate_lot_size(10_000, 0.002, "EURUSD")
        assert result.is_valid is True
        assert result.lot_size > 0.0

    def test_symbol_preserved_in_result(self):
        ks = KellyPositionSizer(0.6, 1.5, 1.0)
        result = ks.calculate_lot_size(10_000, 0.002, "GBPUSD")
        assert result.symbol == "GBPUSD"

    def test_lot_size_handcalculation(self):
        """
        Manuelle Berechnung:
          raw_kelly = 1/3, half-kelly = 1/6 ≈ 0.16667
          risk = 10000 * 0.16667 = 1666.7
          sl_dist = 0.002 * 1.5 = 0.003
          sl_pips = 0.003 / 0.0001 = 30
          raw_lot = 1666.7 / (30 * 10) = 5.5556
          rounded = floor(5.5556 / 0.01) * 0.01 = 5.55
        """
        ks = KellyPositionSizer(
            win_rate=0.6, avg_win=1.5, avg_loss=1.0,
            kelly_multiplier=0.5, sl_atr_multiplier=1.5,
        )
        result = ks.calculate_lot_size(
            account_balance=10_000, atr=0.002, symbol="EURUSD",
            pip_value=10.0, pip_size=0.0001, lot_step=0.01,
        )
        assert result.is_valid is True
        assert result.lot_size == pytest.approx(5.55, abs=0.01)

    def test_risk_amount_in_result(self):
        ks = KellyPositionSizer(0.6, 1.5, 1.0, kelly_multiplier=0.5)
        result = ks.calculate_lot_size(10_000, 0.002, "EURUSD")
        expected_risk = 10_000 * ks.kelly_fraction
        assert result.risk_amount == pytest.approx(expected_risk, rel=1e-6)

    def test_stop_loss_distance_in_result(self):
        ks = KellyPositionSizer(0.6, 1.5, 1.0, sl_atr_multiplier=1.5)
        result = ks.calculate_lot_size(10_000, 0.002, "EURUSD")
        assert result.stop_loss_distance == pytest.approx(0.003, rel=1e-6)

    # ── Immer abrunden ────────────────────────────────────────────────────────

    def test_lot_never_rounds_up(self):
        """Risiko darf nie ueberschritten werden – nur abrunden."""
        ks = KellyPositionSizer(0.6, 1.5, 1.0)
        result = ks.calculate_lot_size(10_000, 0.002, "EURUSD", lot_step=0.01)
        raw_lot = (10_000 * ks.kelly_fraction) / ((0.002 * 1.5 / 0.0001) * 10.0)
        assert result.lot_size <= raw_lot + 1e-9

    def test_lot_step_respected(self):
        ks = KellyPositionSizer(0.6, 1.5, 1.0)
        result = ks.calculate_lot_size(10_000, 0.002, "EURUSD", lot_step=0.01)
        remainder = round(result.lot_size / 0.01) * 0.01 - result.lot_size
        assert abs(remainder) < 1e-6

    # ── Ablehnungen ───────────────────────────────────────────────────────────

    def test_zero_balance_rejected(self):
        ks = KellyPositionSizer(0.6, 1.5, 1.0)
        result = ks.calculate_lot_size(0, 0.002, "EURUSD")
        assert result.is_valid is False
        assert result.lot_size == 0.0

    def test_negative_balance_rejected(self):
        ks = KellyPositionSizer(0.6, 1.5, 1.0)
        result = ks.calculate_lot_size(-1000, 0.002, "EURUSD")
        assert result.is_valid is False

    def test_zero_atr_rejected(self):
        ks = KellyPositionSizer(0.6, 1.5, 1.0)
        result = ks.calculate_lot_size(10_000, 0.0, "EURUSD")
        assert result.is_valid is False

    def test_negative_atr_rejected(self):
        ks = KellyPositionSizer(0.6, 1.5, 1.0)
        result = ks.calculate_lot_size(10_000, -0.001, "EURUSD")
        assert result.is_valid is False

    def test_negative_kelly_rejects_trade(self):
        """Unfavorable Spielparameter -> Trade wird abgelehnt."""
        ks = KellyPositionSizer(
            win_rate=0.4, avg_win=1.0, avg_loss=2.0
        )
        result = ks.calculate_lot_size(10_000, 0.002, "EURUSD")
        assert result.is_valid is False
        assert result.lot_size == 0.0
        assert result.rejection_reason is not None

    def test_small_balance_below_min_lot(self):
        # balance=10: risk=10*0.1667=1.667, sl_pips=30, lot=1.667/300=0.0056 < min(0.01)
        ks = KellyPositionSizer(0.6, 1.5, 1.0)
        result = ks.calculate_lot_size(10, 0.002, "EURUSD")
        assert result.is_valid is False

    def test_rejection_has_reason(self):
        ks = KellyPositionSizer(0.6, 1.5, 1.0)
        result = ks.calculate_lot_size(0, 0.002, "EURUSD")
        assert result.rejection_reason is not None
        assert len(result.rejection_reason) > 0

    # ── risk_pct wird ignoriert ───────────────────────────────────────────────

    def test_risk_pct_ignored(self):
        """risk_pct aendert das Ergebnis nicht – Kelly bestimmt den Anteil."""
        ks = KellyPositionSizer(0.6, 1.5, 1.0)
        r1 = ks.calculate_lot_size(10_000, 0.002, "EURUSD", risk_pct=0.5)
        r2 = ks.calculate_lot_size(10_000, 0.002, "EURUSD", risk_pct=5.0)
        assert r1.lot_size == r2.lot_size

    # ── update_stats ──────────────────────────────────────────────────────────

    def test_update_stats_recomputes_fraction(self):
        ks = KellyPositionSizer(0.5, 1.0, 1.0)  # faire Muenze -> kelly=0
        old_fraction = ks.kelly_fraction
        ks.update_stats(win_rate=0.6, avg_win=1.5, avg_loss=1.0)
        assert ks.kelly_fraction > old_fraction

    def test_update_stats_to_negative_clamps_zero(self):
        ks = KellyPositionSizer(0.6, 1.5, 1.0)
        ks.update_stats(win_rate=0.3, avg_win=1.0, avg_loss=2.0)
        assert ks.kelly_fraction == 0.0

    def test_update_stats_respects_max_cap(self):
        ks = KellyPositionSizer(0.6, 1.5, 1.0, max_kelly_fraction=0.10)
        ks.update_stats(0.9, 10.0, 1.0)
        assert ks.kelly_fraction <= 0.10 + 1e-10

    # ── Vergleich mit PositionSizer-Schnittstelle ─────────────────────────────

    def test_same_interface_as_position_sizer(self):
        """KellyPositionSizer.calculate_lot_size() hat dieselbe Signatur."""
        from src.risk.position_sizer import PositionSizer
        ks = KellyPositionSizer(0.6, 1.5, 1.0)
        ps = PositionSizer(risk_per_trade_pct=1.0)

        r_kelly = ks.calculate_lot_size(10_000, 0.002, "EURUSD")
        r_atr   = ps.calculate_lot_size(10_000, 0.002, "EURUSD")

        # Beide geben PositionSizeResult zurueck
        for r in (r_kelly, r_atr):
            assert hasattr(r, "lot_size")
            assert hasattr(r, "risk_amount")
            assert hasattr(r, "stop_loss_distance")
            assert hasattr(r, "is_valid")
            assert hasattr(r, "rejection_reason")

    def test_higher_win_rate_bigger_lot(self):
        """Hohere Gewinnrate -> groessere Kelly-Fraktion -> groesserer Lot."""
        ks_low  = KellyPositionSizer(0.55, 1.5, 1.0)
        ks_high = KellyPositionSizer(0.70, 1.5, 1.0)
        r_low  = ks_low.calculate_lot_size(10_000, 0.002, "EURUSD")
        r_high = ks_high.calculate_lot_size(10_000, 0.002, "EURUSD")
        if r_low.is_valid and r_high.is_valid:
            assert r_high.lot_size >= r_low.lot_size

    def test_larger_account_bigger_lot(self):
        ks = KellyPositionSizer(0.6, 1.5, 1.0)
        r_small = ks.calculate_lot_size(10_000,  0.002, "EURUSD")
        r_large = ks.calculate_lot_size(100_000, 0.002, "EURUSD")
        assert r_large.lot_size > r_small.lot_size
