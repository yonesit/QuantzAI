"""
Unit-Tests fuer PositionSizer.
"""

from __future__ import annotations

import pytest

from src.risk.position_sizer import PositionSizer, PositionSizeResult


# ---------------------------------------------------------------------------
# Tests: Normale Berechnung
# ---------------------------------------------------------------------------

class TestCalculateLotSize:

    def test_returns_valid_result_for_normal_inputs(self):
        sizer = PositionSizer(risk_per_trade_pct=1.0, sl_atr_multiplier=1.5)
        result = sizer.calculate_lot_size(
            account_balance=10_000, atr=0.0020, symbol="EURUSD"
        )
        assert result.is_valid is True
        assert result.lot_size > 0

    def test_risk_amount_correct(self):
        sizer = PositionSizer(risk_per_trade_pct=1.0)
        result = sizer.calculate_lot_size(
            account_balance=10_000, atr=0.0020, symbol="EURUSD"
        )
        assert result.risk_amount == pytest.approx(100.0)  # 1% von 10000

    def test_stop_loss_distance_correct(self):
        sizer = PositionSizer(sl_atr_multiplier=1.5)
        result = sizer.calculate_lot_size(
            account_balance=10_000, atr=0.0020, symbol="EURUSD"
        )
        assert result.stop_loss_distance == pytest.approx(0.0030)  # 1.5 * 0.0020

    def test_higher_risk_pct_gives_bigger_lot(self):
        sizer = PositionSizer(risk_per_trade_pct=1.0)
        small_risk = sizer.calculate_lot_size(10_000, 0.0020, "EURUSD", risk_pct=1.0)
        big_risk   = sizer.calculate_lot_size(10_000, 0.0020, "EURUSD", risk_pct=2.0)
        assert big_risk.lot_size > small_risk.lot_size

    def test_higher_atr_gives_smaller_lot(self):
        sizer = PositionSizer()
        low_atr  = sizer.calculate_lot_size(10_000, 0.0010, "EURUSD")
        high_atr = sizer.calculate_lot_size(10_000, 0.0050, "EURUSD")
        assert high_atr.lot_size < low_atr.lot_size

    def test_override_risk_pct_per_call(self):
        sizer = PositionSizer(risk_per_trade_pct=1.0)
        default_result  = sizer.calculate_lot_size(10_000, 0.0020, "EURUSD")
        override_result = sizer.calculate_lot_size(10_000, 0.0020, "EURUSD", risk_pct=0.5)
        assert override_result.lot_size < default_result.lot_size

    def test_lot_size_rounded_to_lot_step(self):
        sizer = PositionSizer(risk_per_trade_pct=1.0)
        result = sizer.calculate_lot_size(
            account_balance=10_000, atr=0.0020, symbol="EURUSD", lot_step=0.01
        )
        # Lot-Groesse muss ein ganzzahliges Vielfaches von lot_step sein
        remainder = round(result.lot_size / 0.01) * 0.01 - result.lot_size
        assert abs(remainder) < 1e-6

    def test_never_rounds_up(self):
        """Kritisch: Risiko darf nie ueber das gewuenschte Niveau steigen durch Aufrundung."""
        sizer = PositionSizer(risk_per_trade_pct=1.0)
        result = sizer.calculate_lot_size(
            account_balance=10_000, atr=0.0020, symbol="EURUSD", lot_step=0.01
        )
        raw_lot = (10_000 * 0.01) / ((0.0020 * 1.5 / 0.0001) * 10.0)
        assert result.lot_size <= raw_lot + 1e-9


# ---------------------------------------------------------------------------
# Tests: Edge Cases (explizit gefordert im Issue)
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_atr_zero_rejected(self):
        sizer = PositionSizer()
        result = sizer.calculate_lot_size(account_balance=10_000, atr=0.0, symbol="EURUSD")
        assert result.is_valid is False
        assert result.lot_size == 0.0
        assert "ATR" in result.rejection_reason

    def test_negative_atr_rejected(self):
        sizer = PositionSizer()
        result = sizer.calculate_lot_size(account_balance=10_000, atr=-0.001, symbol="EURUSD")
        assert result.is_valid is False

    def test_very_small_account_balance_rejected(self):
        sizer = PositionSizer(risk_per_trade_pct=1.0)
        # 10 Euro Kontostand -> Risikobetrag viel zu klein fuer min_lot_size
        result = sizer.calculate_lot_size(account_balance=10, atr=0.0020, symbol="EURUSD")
        assert result.is_valid is False
        assert result.lot_size == 0.0

    def test_zero_account_balance_rejected(self):
        sizer = PositionSizer()
        result = sizer.calculate_lot_size(account_balance=0, atr=0.0020, symbol="EURUSD")
        assert result.is_valid is False

    def test_negative_account_balance_rejected(self):
        sizer = PositionSizer()
        result = sizer.calculate_lot_size(account_balance=-500, atr=0.0020, symbol="EURUSD")
        assert result.is_valid is False

    def test_very_high_volatility_rejected_if_too_small(self):
        sizer = PositionSizer(risk_per_trade_pct=1.0)
        # Extrem hohe Volatilitaet -> winzige Lot-Groesse -> unter min_lot_size
        result = sizer.calculate_lot_size(account_balance=1_000, atr=0.0500, symbol="EURUSD")
        assert result.is_valid is False

    def test_zero_risk_pct_rejected(self):
        sizer = PositionSizer()
        result = sizer.calculate_lot_size(
            account_balance=10_000, atr=0.0020, symbol="EURUSD", risk_pct=0.0
        )
        assert result.is_valid is False

    def test_rejection_reason_present_when_invalid(self):
        sizer = PositionSizer()
        result = sizer.calculate_lot_size(account_balance=10_000, atr=0.0, symbol="EURUSD")
        assert result.rejection_reason is not None
        assert len(result.rejection_reason) > 0

    def test_no_rejection_reason_when_valid(self):
        sizer = PositionSizer()
        result = sizer.calculate_lot_size(account_balance=10_000, atr=0.0020, symbol="EURUSD")
        assert result.rejection_reason is None


# ---------------------------------------------------------------------------
# Tests: Verschiedene Symbole / Pip-Konfigurationen
# ---------------------------------------------------------------------------

class TestSymbolSpecifics:

    def test_jpy_pair_different_pip_size(self):
        sizer = PositionSizer(risk_per_trade_pct=1.0)
        result = sizer.calculate_lot_size(
            account_balance=10_000, atr=0.20, symbol="USDJPY",
            pip_size=0.01, pip_value=9.0,
        )
        assert result.is_valid is True

    def test_custom_lot_step_respected(self):
        sizer = PositionSizer(risk_per_trade_pct=5.0)  # hohes Risiko -> groessere Lot
        result = sizer.calculate_lot_size(
            account_balance=50_000, atr=0.0015, symbol="EURUSD", lot_step=0.1
        )
        remainder = round(result.lot_size / 0.1) * 0.1 - result.lot_size
        assert abs(remainder) < 1e-6

    def test_custom_min_lot_size(self):
        sizer = PositionSizer(risk_per_trade_pct=1.0, min_lot_size=0.1)
        # Lot-Groesse die normalerweise gueltig waere (>0.01) aber unter 0.1 liegt
        result = sizer.calculate_lot_size(account_balance=1_000, atr=0.0020, symbol="EURUSD")
        assert result.is_valid is False
