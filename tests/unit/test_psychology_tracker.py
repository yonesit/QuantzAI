"""
tests/unit/test_psychology_tracker.py
Unit-Tests fuer PsychologyTracker und TradingOrchestrator.pause()/resume().

Abgedeckt:
  - Tilt-Erkennung: Verlust-Serien, Lot-Eskalation, Revenge-Mood
  - Notbremse: orchestrator.pause() wird bei Tilt ausgeloest
  - record_open / record_close: Datenspeicherung und Updates
  - analyze_mood_patterns: Win-Rate-Analyse mit genuegend Trades
  - TradingOrchestrator.pause() / resume() / is_paused
  - run_cycle gibt 'trading_paused' zurueck wenn Pause aktiv
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.journal.psychology_tracker import MoodState, PsychologyTracker, TradeRecord
from src.orchestrator import TradingOrchestrator
from src.risk.position_sizer import PositionSizeResult


# ─────────────────────────────────────────────────────────────────────────────
#  Hilfsfunktionen
# ─────────────────────────────────────────────────────────────────────────────

_NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _record(
    trade_id:       int | str   = 1,
    symbol:         str         = "EURUSD",
    lot_size:       float       = 0.10,
    mood_open:      MoodState   = MoodState.CALM,
    opening_reason: str         = "trend",
    pnl:            float | None = None,
    mood_close:     MoodState | None = None,
    plan_followed:  bool | None = None,
    close_reason:   str         = "",
) -> TradeRecord:
    r = TradeRecord(
        trade_id=trade_id,
        symbol=symbol,
        lot_size=lot_size,
        mood_open=mood_open,
        opening_reason=opening_reason,
        opened_at=_NOW,
    )
    if pnl is not None:
        r.pnl           = pnl
        r.mood_close    = mood_close or MoodState.CALM
        r.plan_followed = plan_followed if plan_followed is not None else True
        r.close_reason  = close_reason
        r.closed_at     = _NOW
    return r


def _loss(trade_id=1, lot_size=0.10, mood_open=MoodState.CALM) -> TradeRecord:
    return _record(trade_id=trade_id, lot_size=lot_size, mood_open=mood_open, pnl=-50.0)


def _win(trade_id=1, lot_size=0.10, mood_open=MoodState.CALM) -> TradeRecord:
    return _record(trade_id=trade_id, lot_size=lot_size, mood_open=mood_open, pnl=+50.0)


def _tracker(**kwargs) -> PsychologyTracker:
    return PsychologyTracker(_now_fn=lambda: _NOW, **kwargs)


def _make_orch() -> TradingOrchestrator:
    """Baut minimalen Orchestrator mit allen Mocks."""
    features = pd.DataFrame([{"close": 1.09, "atr": 0.001}])
    size_res  = PositionSizeResult(
        symbol="EURUSD", lot_size=0.1, risk_amount=100.0,
        stop_loss_distance=0.001, is_valid=True, rejection_reason=None,
    )
    pipeline   = MagicMock(); pipeline.run_batch.return_value = {}
    risk_guard = MagicMock()
    risk_guard.is_trading_allowed.return_value = True
    risk_guard.get_position_size_multiplier.return_value = 1.0
    pre_trade  = MagicMock(); pre_trade.is_safe_to_trade.return_value = (True, "ok")
    sig_model  = MagicMock(); sig_model.get_signal.return_value = "long"
    corr_guard = MagicMock(); corr_guard.can_open_position.return_value = True
    executor   = MagicMock()
    executor.get_open_positions.return_value = []
    executor.open_position.return_value = {"ticket": 1}
    pos_sizer  = MagicMock(); pos_sizer.calculate_lot_size.return_value = size_res

    return TradingOrchestrator(
        data_pipeline=pipeline,
        risk_guard=risk_guard,
        pre_trade_check=pre_trade,
        signal_model=sig_model,
        correlation_guard=corr_guard,
        position_sizer=pos_sizer,
        order_executor=executor,
        features_loader=lambda sym: features,
        balance_getter=lambda: 10_000.0,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  TestTiltDetection
# ─────────────────────────────────────────────────────────────────────────────

class TestTiltDetection:

    def test_no_tilt_empty_trades(self):
        assert _tracker().check_tilt_state([]) is False

    def test_no_tilt_below_threshold(self):
        trades = [_loss(i) for i in range(4)]  # 4 < default 5
        assert _tracker().check_tilt_state(trades) is False

    def test_tilt_consecutive_losses_default(self):
        trades = [_loss(i) for i in range(5)]
        assert _tracker().check_tilt_state(trades) is True

    def test_tilt_consecutive_losses_custom_threshold(self):
        trades = [_loss(i) for i in range(3)]
        assert _tracker(tilt_consecutive_losses=3).check_tilt_state(trades) is True

    def test_no_tilt_loss_streak_broken_by_win(self):
        # win followed by 4 losses — streak only 4, below threshold 5
        trades = [_win(0)] + [_loss(i + 1) for i in range(4)]
        assert _tracker().check_tilt_state(trades) is False

    def test_tilt_loss_streak_after_win(self):
        # win then 5 losses
        trades = [_win(0)] + [_loss(i + 1) for i in range(5)]
        assert _tracker().check_tilt_state(trades) is True

    def test_tilt_lot_escalation_after_loss(self):
        # lot goes from 0.10 to 0.12 (ratio 1.2) after loss
        trades = [
            _loss(1, lot_size=0.10),
            _win(2,  lot_size=0.12),
        ]
        assert _tracker(tilt_lot_increase_ratio=1.2).check_tilt_state(trades) is True

    def test_no_tilt_small_lot_increase_after_loss(self):
        # lot goes from 0.10 to 0.11 (ratio 1.1) – below threshold 1.2
        trades = [
            _loss(1, lot_size=0.10),
            _win(2,  lot_size=0.11),
        ]
        assert _tracker(tilt_lot_increase_ratio=1.2).check_tilt_state(trades) is False

    def test_tilt_lot_escalation_after_win_no_trigger(self):
        # lot escalation only triggers after a LOSS, not a win
        trades = [
            _win(1, lot_size=0.10),
            _win(2, lot_size=0.20),
        ]
        assert _tracker(tilt_lot_increase_ratio=1.2).check_tilt_state(trades) is False

    def test_tilt_revenge_mood_angry_after_loss(self):
        trades = [
            _loss(1),
            _win(2, mood_open=MoodState.ANGRY),
        ]
        assert _tracker().check_tilt_state(trades) is True

    def test_tilt_revenge_mood_fomo_after_loss(self):
        trades = [
            _loss(1),
            _loss(2, mood_open=MoodState.FOMO),
        ]
        assert _tracker().check_tilt_state(trades) is True

    def test_no_tilt_calm_mood_after_loss(self):
        trades = [
            _loss(1),
            _win(2, mood_open=MoodState.CALM),
        ]
        assert _tracker().check_tilt_state(trades) is False

    def test_no_tilt_focused_mood_after_loss(self):
        trades = [
            _loss(1),
            _win(2, mood_open=MoodState.FOCUSED),
        ]
        assert _tracker().check_tilt_state(trades) is False

    def test_overconfident_after_loss_no_tilt(self):
        # OVERCONFIDENT is not a revenge mood
        trades = [
            _loss(1),
            _win(2, mood_open=MoodState.OVERCONFIDENT),
        ]
        assert _tracker().check_tilt_state(trades) is False

    def test_check_uses_internal_trades_when_no_arg(self):
        t = _tracker()
        # Manually inject trades (simulating record_open+close flow)
        for i in range(5):
            t._trades.append(_loss(i))
        assert t.check_tilt_state() is True

    def test_open_only_trades_ignored(self):
        # Trades without pnl (still open) must not count
        trades = [_record(i) for i in range(5)]  # pnl=None → open
        assert _tracker().check_tilt_state(trades) is False


# ─────────────────────────────────────────────────────────────────────────────
#  TestEmergencyBrake
# ─────────────────────────────────────────────────────────────────────────────

class TestEmergencyBrake:

    def _make_tracker_with_mock_orch(self, **kwargs) -> tuple[PsychologyTracker, MagicMock]:
        orch = MagicMock()
        return _tracker(orchestrator=orch, **kwargs), orch

    def _add_losses_via_record(self, t: PsychologyTracker, n: int) -> None:
        for i in range(n):
            t.record_open(i, "EURUSD", MoodState.CALM, "test", 0.1)
            t.record_close(i, MoodState.CALM, True, "sl", -50.0)

    def test_pause_called_after_fifth_loss(self):
        t, orch = self._make_tracker_with_mock_orch()
        self._add_losses_via_record(t, 5)
        orch.pause.assert_called_once()

    def test_pause_not_called_after_four_losses(self):
        t, orch = self._make_tracker_with_mock_orch()
        self._add_losses_via_record(t, 4)
        orch.pause.assert_not_called()

    def test_pause_not_called_without_orchestrator(self):
        t = _tracker()  # no orchestrator
        # Should not raise even without orchestrator
        for i in range(5):
            t.record_open(i, "EURUSD", MoodState.CALM, "test", 0.1)
            t.record_close(i, MoodState.CALM, True, "sl", -50.0)

    def test_pause_called_on_lot_escalation(self):
        t, orch = self._make_tracker_with_mock_orch(tilt_lot_increase_ratio=1.2)
        t.record_open(1, "EURUSD", MoodState.CALM, "test", 0.10)
        t.record_close(1, MoodState.CALM, True, "sl", -50.0)
        t.record_open(2, "EURUSD", MoodState.CALM, "test", 0.12)
        t.record_close(2, MoodState.CALM, True, "tp", +50.0)
        orch.pause.assert_called_once()

    def test_pause_called_on_revenge_mood(self):
        t, orch = self._make_tracker_with_mock_orch()
        t.record_open(1, "EURUSD", MoodState.CALM, "test", 0.1)
        t.record_close(1, MoodState.CALM, True, "sl", -50.0)
        t.record_open(2, "EURUSD", MoodState.ANGRY, "revenge", 0.1)
        t.record_close(2, MoodState.ANGRY, False, "sl", -50.0)
        orch.pause.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
#  TestRecordOpenClose
# ─────────────────────────────────────────────────────────────────────────────

class TestRecordOpenClose:

    def test_record_open_stores_trade(self):
        t = _tracker()
        t.record_open(42, "USDJPY", MoodState.FOCUSED, "breakout", 0.2)
        assert len(t.all_trades) == 1
        tr = t.all_trades[0]
        assert tr.trade_id == 42
        assert tr.symbol == "USDJPY"
        assert tr.mood_open == MoodState.FOCUSED
        assert tr.opening_reason == "breakout"
        assert tr.lot_size == 0.2
        assert tr.pnl is None  # still open

    def test_record_open_adds_to_open_ids(self):
        t = _tracker()
        t.record_open(7, "EURUSD", MoodState.CALM, "signal", 0.1)
        assert 7 in t.open_trade_ids

    def test_record_close_updates_trade(self):
        t = _tracker()
        t.record_open(1, "EURUSD", MoodState.CALM, "trend", 0.1)
        t.record_close(1, MoodState.NERVOUS, True, "tp", +75.5)
        tr = t.all_trades[0]
        assert tr.pnl == +75.5
        assert tr.mood_close == MoodState.NERVOUS
        assert tr.plan_followed is True
        assert tr.close_reason == "tp"
        assert tr.closed_at is not None

    def test_record_close_removes_from_open_ids(self):
        t = _tracker()
        t.record_open(5, "EURUSD", MoodState.CALM, "trend", 0.1)
        t.record_close(5, MoodState.CALM, True, "sl", -30.0)
        assert 5 not in t.open_trade_ids

    def test_record_close_unknown_id_is_noop(self):
        t = _tracker()
        # Should not raise
        t.record_close(999, MoodState.CALM, True, "sl", -10.0)
        assert len(t.all_trades) == 0

    def test_multiple_open_trades_tracked(self):
        t = _tracker()
        t.record_open(1, "EURUSD", MoodState.CALM,    "trend", 0.1)
        t.record_open(2, "GBPUSD", MoodState.FOCUSED, "news",  0.2)
        assert set(t.open_trade_ids) == {1, 2}

    def test_record_close_only_updates_correct_trade(self):
        t = _tracker()
        t.record_open(1, "EURUSD", MoodState.CALM, "A", 0.1)
        t.record_open(2, "GBPUSD", MoodState.CALM, "B", 0.2)
        t.record_close(1, MoodState.CALM, True, "tp", +100.0)
        tr1, tr2 = t.all_trades
        assert tr1.pnl == +100.0
        assert tr2.pnl is None  # still open

    def test_string_trade_id(self):
        t = _tracker()
        t.record_open("abc-123", "EURUSD", MoodState.CALM, "test", 0.1)
        t.record_close("abc-123", MoodState.CALM, True, "tp", +10.0)
        assert t.all_trades[0].pnl == +10.0

    def test_all_trades_returns_copy(self):
        t = _tracker()
        t.record_open(1, "EURUSD", MoodState.CALM, "test", 0.1)
        copy = t.all_trades
        copy.clear()
        assert len(t.all_trades) == 1


# ─────────────────────────────────────────────────────────────────────────────
#  TestMoodPatternAnalysis
# ─────────────────────────────────────────────────────────────────────────────

class TestMoodPatternAnalysis:

    def _fill_trades(self, t: PsychologyTracker, n: int, mood: MoodState, win: bool) -> None:
        for i in range(n):
            t.record_open(i, "EURUSD", mood, "test", 0.1)
            pnl = +50.0 if win else -50.0
            t.record_close(i, MoodState.CALM, True, "tp" if win else "sl", pnl)

    def test_returns_empty_below_min_trades(self):
        t = _tracker(mood_pattern_min_trades=30)
        self._fill_trades(t, 29, MoodState.CALM, win=True)
        assert t.analyze_mood_patterns() == {}

    def test_returns_results_at_min_trades(self):
        t = _tracker(mood_pattern_min_trades=30)
        self._fill_trades(t, 30, MoodState.CALM, win=True)
        result = t.analyze_mood_patterns()
        assert MoodState.CALM in result

    def test_win_rate_all_wins(self):
        t = _tracker(mood_pattern_min_trades=5)
        self._fill_trades(t, 5, MoodState.FOCUSED, win=True)
        result = t.analyze_mood_patterns()
        assert result[MoodState.FOCUSED]["win_rate"] == pytest.approx(1.0)
        assert result[MoodState.FOCUSED]["n_trades"] == 5

    def test_win_rate_all_losses(self):
        t = _tracker(mood_pattern_min_trades=5)
        self._fill_trades(t, 5, MoodState.ANGRY, win=False)
        result = t.analyze_mood_patterns()
        assert result[MoodState.ANGRY]["win_rate"] == pytest.approx(0.0)

    def test_win_rate_mixed(self):
        t = _tracker(mood_pattern_min_trades=4)
        self._fill_trades(t, 2, MoodState.NERVOUS, win=True)
        self._fill_trades(t, 2, MoodState.NERVOUS, win=False)
        result = t.analyze_mood_patterns()
        assert result[MoodState.NERVOUS]["win_rate"] == pytest.approx(0.5)
        assert result[MoodState.NERVOUS]["n_trades"] == 4

    def test_multiple_moods_tracked_separately(self):
        t = _tracker(mood_pattern_min_trades=4)
        self._fill_trades(t, 2, MoodState.CALM,    win=True)
        self._fill_trades(t, 2, MoodState.NERVOUS, win=False)
        result = t.analyze_mood_patterns()
        assert result[MoodState.CALM]["win_rate"]    == pytest.approx(1.0)
        assert result[MoodState.NERVOUS]["win_rate"] == pytest.approx(0.0)

    def test_open_trades_excluded_from_analysis(self):
        t = _tracker(mood_pattern_min_trades=5)
        # 5 closed + 3 open (not counted)
        self._fill_trades(t, 5, MoodState.CALM, win=True)
        for i in range(100, 103):
            t.record_open(i, "EURUSD", MoodState.FOMO, "test", 0.1)
        result = t.analyze_mood_patterns()
        assert MoodState.FOMO not in result  # open trades not counted


# ─────────────────────────────────────────────────────────────────────────────
#  TestOrchestratorPause
# ─────────────────────────────────────────────────────────────────────────────

class TestOrchestratorPause:

    def test_not_paused_initially(self):
        orch = _make_orch()
        assert orch.is_paused is False

    def test_pause_sets_flag(self):
        orch = _make_orch()
        orch.pause()
        assert orch.is_paused is True

    def test_resume_clears_flag(self):
        orch = _make_orch()
        orch.pause()
        orch.resume()
        assert orch.is_paused is False

    def test_pause_with_reason(self):
        orch = _make_orch()
        orch.pause(reason="Tilt erkannt")
        assert orch.is_paused is True

    def test_run_cycle_returns_paused_reason(self):
        orch = _make_orch()
        orch.pause()
        result = orch.run_cycle("EURUSD")
        assert result["reason"]          == "trading_paused"
        assert result["step_stopped_at"] == "pause"
        assert result["action"]          == "skipped"

    def test_run_cycle_skips_all_steps_when_paused(self):
        orch = _make_orch()
        orch.pause()
        result = orch.run_cycle("EURUSD")
        # DataPipeline must NOT have been called
        orch._pipeline.run_batch.assert_not_called()

    def test_run_cycle_proceeds_after_resume(self):
        orch = _make_orch()
        orch.pause()
        orch.resume()
        result = orch.run_cycle("EURUSD")
        assert result["reason"] != "trading_paused"
        assert result["action"] in ("open_buy", "open_sell", "flat", "skipped")
