"""
tests/unit/test_modes.py
Unit-Tests fuer src/modes.py: TradingMode, ConfirmationCallback, is_autonomous_allowed.
"""

from __future__ import annotations

import os

import pytest

from src.modes import (
    AUTONOMOUS_ENV_VAL,
    AUTONOMOUS_ENV_VAR,
    ConfirmationCallback,
    TradingMode,
    is_autonomous_allowed,
)


# ─────────────────────────────────────────────────────────────────────────────
#  TradingMode Enum
# ─────────────────────────────────────────────────────────────────────────────

class TestTradingModeEnum:

    def test_has_suggest_only(self):
        assert TradingMode.SUGGEST_ONLY.value == "suggest_only"

    def test_has_confirm_required(self):
        assert TradingMode.CONFIRM_REQUIRED.value == "confirm_required"

    def test_has_autonomous(self):
        assert TradingMode.AUTONOMOUS.value == "autonomous"

    def test_enum_has_three_members(self):
        assert len(TradingMode) == 3

    def test_from_value_suggest_only(self):
        assert TradingMode("suggest_only") == TradingMode.SUGGEST_ONLY

    def test_from_value_confirm_required(self):
        assert TradingMode("confirm_required") == TradingMode.CONFIRM_REQUIRED

    def test_from_value_autonomous(self):
        assert TradingMode("autonomous") == TradingMode.AUTONOMOUS

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            TradingMode("unknown")

    def test_modes_are_distinct(self):
        modes = list(TradingMode)
        assert len(set(m.value for m in modes)) == 3


# ─────────────────────────────────────────────────────────────────────────────
#  is_autonomous_allowed
# ─────────────────────────────────────────────────────────────────────────────

class TestIsAutonomousAllowed:

    def test_returns_true_when_env_set(self, monkeypatch):
        monkeypatch.setenv(AUTONOMOUS_ENV_VAR, AUTONOMOUS_ENV_VAL)
        assert is_autonomous_allowed() is True

    def test_returns_false_when_env_missing(self, monkeypatch):
        monkeypatch.delenv(AUTONOMOUS_ENV_VAR, raising=False)
        assert is_autonomous_allowed() is False

    def test_returns_false_when_env_wrong_value(self, monkeypatch):
        monkeypatch.setenv(AUTONOMOUS_ENV_VAR, "no")
        assert is_autonomous_allowed() is False

    def test_returns_false_when_env_empty(self, monkeypatch):
        monkeypatch.setenv(AUTONOMOUS_ENV_VAR, "")
        assert is_autonomous_allowed() is False

    def test_case_insensitive_yes(self, monkeypatch):
        monkeypatch.setenv(AUTONOMOUS_ENV_VAR, "YES")
        assert is_autonomous_allowed() is True

    def test_case_insensitive_Yes(self, monkeypatch):
        monkeypatch.setenv(AUTONOMOUS_ENV_VAR, "Yes")
        assert is_autonomous_allowed() is True

    def test_strips_whitespace(self, monkeypatch):
        monkeypatch.setenv(AUTONOMOUS_ENV_VAR, "  yes  ")
        assert is_autonomous_allowed() is True

    def test_strips_whitespace_wrong_value(self, monkeypatch):
        monkeypatch.setenv(AUTONOMOUS_ENV_VAR, "  no  ")
        assert is_autonomous_allowed() is False

    def test_constant_env_var_name(self):
        assert AUTONOMOUS_ENV_VAR == "CONFIRM_AUTONOMOUS"

    def test_constant_env_val(self):
        assert AUTONOMOUS_ENV_VAL == "yes"


# ─────────────────────────────────────────────────────────────────────────────
#  ConfirmationCallback Protocol
# ─────────────────────────────────────────────────────────────────────────────

class TestConfirmationCallbackProtocol:

    def test_class_implementing_protocol_is_accepted(self):
        class _CB:
            def confirm_order(self, symbol, direction, lot_size, sl, tp) -> bool:
                return True

        assert isinstance(_CB(), ConfirmationCallback)

    def test_class_missing_method_is_rejected(self):
        class _NoCB:
            pass

        assert not isinstance(_NoCB(), ConfirmationCallback)

    def test_lambda_is_not_accepted(self):
        fn = lambda: True  # noqa: E731
        assert not isinstance(fn, ConfirmationCallback)

    def test_confirm_order_signature_returns_bool(self):
        class _CB:
            def confirm_order(self, symbol, direction, lot_size, sl, tp) -> bool:
                return False

        cb = _CB()
        result = cb.confirm_order("EURUSD", "buy", 0.1, 1.089, 1.095)
        assert isinstance(result, bool)
        assert result is False

    def test_confirm_order_can_return_true(self):
        class _CB:
            def confirm_order(self, symbol, direction, lot_size, sl, tp) -> bool:
                return True

        assert _CB().confirm_order("EURUSD", "sell", 0.2, 1.091, 1.085) is True
