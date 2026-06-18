"""
Unit-Tests fuer RiskGuard.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

import pytest

from src.risk.risk_guard import RiskGuard, RiskState


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def _guard(tmp_path, **kwargs) -> RiskGuard:
    return RiskGuard(state_path=str(tmp_path / "risk_state.json"), **kwargs)


# ---------------------------------------------------------------------------
# Tests: Grundfunktionen
# ---------------------------------------------------------------------------

class TestBasics:

    def test_first_update_initializes_state(self, tmp_path):
        guard = _guard(tmp_path)
        guard.update_balance(10_000)
        assert guard.state is not None
        assert guard.state.day_start_balance == 10_000
        assert guard.state.all_time_high == 10_000

    def test_trading_allowed_initially(self, tmp_path):
        guard = _guard(tmp_path)
        guard.update_balance(10_000)
        assert guard.is_trading_allowed() is True

    def test_all_time_high_tracked(self, tmp_path):
        guard = _guard(tmp_path)
        guard.update_balance(10_000)
        guard.update_balance(11_000)
        guard.update_balance(10_500)
        assert guard.state.all_time_high == 11_000

    def test_position_multiplier_default_is_one(self, tmp_path):
        guard = _guard(tmp_path)
        guard.update_balance(10_000)
        assert guard.get_position_size_multiplier() == 1.0


# ---------------------------------------------------------------------------
# Tests: Taegliches Verlustlimit
# ---------------------------------------------------------------------------

class TestDailyLimit:

    def test_daily_limit_not_hit_with_small_loss(self, tmp_path):
        guard = _guard(tmp_path, daily_loss_limit_pct=5.0)
        guard.update_balance(10_000)
        guard.update_balance(9_800)  # 2% Verlust
        assert guard.is_daily_limit_hit() is False
        assert guard.is_trading_allowed() is True

    def test_daily_limit_hit_blocks_trading(self, tmp_path):
        guard = _guard(tmp_path, daily_loss_limit_pct=5.0)
        guard.update_balance(10_000)
        guard.update_balance(9_400)  # 6% Verlust > 5% Limit
        assert guard.is_daily_limit_hit() is True
        assert guard.is_trading_allowed() is False

    def test_daily_limit_exact_boundary(self, tmp_path):
        guard = _guard(tmp_path, daily_loss_limit_pct=5.0)
        guard.update_balance(10_000)
        guard.update_balance(9_500)  # genau 5% Verlust
        assert guard.is_daily_limit_hit() is True

    def test_post_loss_multiplier_active_after_limit_hit(self, tmp_path):
        guard = _guard(tmp_path, daily_loss_limit_pct=5.0, post_loss_size_multiplier=0.5)
        guard.update_balance(10_000)
        guard.update_balance(9_000)  # Limit getroffen
        assert guard.get_position_size_multiplier() == 0.5

    def test_profit_does_not_trigger_limit(self, tmp_path):
        guard = _guard(tmp_path, daily_loss_limit_pct=5.0)
        guard.update_balance(10_000)
        guard.update_balance(10_500)  # Gewinn
        assert guard.is_daily_limit_hit() is False


# ---------------------------------------------------------------------------
# Tests: Maximaler Drawdown
# ---------------------------------------------------------------------------

class TestMaxDrawdown:

    def test_drawdown_not_hit_with_small_decline(self, tmp_path):
        guard = _guard(tmp_path, max_drawdown_pct=15.0)
        guard.update_balance(10_000)
        guard.update_balance(11_000)  # neues Hoch
        guard.update_balance(10_500)  # ~4.5% vom Hoch
        assert guard.is_max_drawdown_hit() is False

    def test_drawdown_hit_blocks_trading_globally(self, tmp_path):
        guard = _guard(tmp_path, max_drawdown_pct=15.0)
        guard.update_balance(10_000)
        guard.update_balance(8_400)  # 16% Drawdown vom Hoch 10000
        assert guard.is_max_drawdown_hit() is True
        assert guard.is_trading_allowed() is False

    def test_drawdown_uses_all_time_high_not_current(self, tmp_path):
        guard = _guard(tmp_path, max_drawdown_pct=15.0)
        guard.update_balance(10_000)
        guard.update_balance(12_000)  # neues Hoch
        guard.update_balance(10_300)  # ~14.2% vom Hoch 12000, nicht vom Start
        assert guard.is_max_drawdown_hit() is False

    def test_manual_reset_of_drawdown(self, tmp_path):
        guard = _guard(tmp_path, max_drawdown_pct=15.0)
        guard.update_balance(10_000)
        guard.update_balance(8_000)
        assert guard.is_max_drawdown_hit() is True

        guard.reset_max_drawdown()
        assert guard.is_max_drawdown_hit() is False


# ---------------------------------------------------------------------------
# Tests: Persistenz
# ---------------------------------------------------------------------------

class TestPersistence:

    def test_state_persisted_to_file(self, tmp_path):
        guard = _guard(tmp_path)
        guard.update_balance(10_000)

        state_file = tmp_path / "risk_state.json"
        assert state_file.exists()

        with open(state_file) as f:
            data = json.load(f)
        assert data["day_start_balance"] == 10_000

    def test_state_survives_restart(self, tmp_path):
        guard1 = _guard(tmp_path, daily_loss_limit_pct=5.0)
        guard1.update_balance(10_000)
        guard1.update_balance(9_400)  # Limit getroffen
        assert guard1.is_daily_limit_hit() is True

        # Neue Instanz simuliert Neustart
        guard2 = _guard(tmp_path, daily_loss_limit_pct=5.0)
        assert guard2.is_daily_limit_hit() is True

    def test_state_survives_restart_normal_case(self, tmp_path):
        guard1 = _guard(tmp_path)
        guard1.update_balance(10_000)
        guard1.update_balance(10_200)

        guard2 = _guard(tmp_path)
        assert guard2.state.current_balance == 10_200
        assert guard2.state.all_time_high == 10_200


# ---------------------------------------------------------------------------
# Tests: Tageswechsel / Reset um Mitternacht
# ---------------------------------------------------------------------------

class TestDailyReset:

    def test_new_day_resets_day_start_balance(self, tmp_path):
        guard = _guard(tmp_path)
        guard.update_balance(10_000)

        # Tag manuell auf "gestern" setzen, um Tageswechsel zu simulieren
        guard._state.day_start_date = "2020-01-01"
        guard.update_balance(9_000)

        # day_start_balance sollte jetzt 9000 sein (neuer Tag, neuer Referenzwert)
        assert guard._state.day_start_balance == 9_000
        assert guard._state.day_start_date == guard._today_str()

    def test_new_day_clears_daily_limit_block(self, tmp_path):
        guard = _guard(tmp_path, daily_loss_limit_pct=5.0)
        guard.update_balance(10_000)
        guard.update_balance(9_000)  # Limit getroffen
        assert guard.is_daily_limit_hit() is True

        # Tageswechsel simulieren
        guard._state.day_start_date = "2020-01-01"
        guard.update_balance(9_000)  # gleicher Kontostand, aber neuer Tag

        assert guard.is_daily_limit_hit() is False
        assert guard.is_trading_allowed() is True

    def test_post_loss_days_decrement_on_new_day(self, tmp_path):
        guard = _guard(tmp_path, daily_loss_limit_pct=5.0, post_loss_days=3)
        guard.update_balance(10_000)
        guard.update_balance(9_000)  # Limit getroffen -> post_loss_days_remaining = 3
        assert guard._state.post_loss_days_remaining == 3

        guard._state.day_start_date = "2020-01-01"
        guard.update_balance(9_500)  # Tag 1 nach Verlust
        assert guard._state.post_loss_days_remaining == 2

    def test_position_multiplier_normal_after_post_loss_period(self, tmp_path):
        guard = _guard(
            tmp_path, daily_loss_limit_pct=5.0, post_loss_days=1,
            post_loss_size_multiplier=0.5,
        )
        guard.update_balance(10_000)
        guard.update_balance(9_000)  # Limit getroffen
        assert guard.get_position_size_multiplier() == 0.5

        # Ein Tageswechsel reicht bei post_loss_days=1 um wieder normal zu werden
        guard._state.day_start_date = "2020-01-01"
        guard.update_balance(9_500)
        assert guard.get_position_size_multiplier() == 1.0
