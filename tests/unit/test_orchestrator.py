"""
tests/unit/test_orchestrator.py
Unit-Tests fuer TradingOrchestrator.

Abgedeckt:
  - Jeder Schritt einzeln gemockt (happy path)
  - Reihenfolge der Schritt-Aufrufe
  - Abgebrochener Check verhindert alle Folgeschritte
  - run_loop: Mehrere Symbole, Stop, Exception-Weiterleitung
  - Graceful Shutdown via stop()
"""

from __future__ import annotations

import os
import threading
import time
from typing import Optional
from unittest.mock import MagicMock, call, patch

import pandas as pd
import pytest

from src.modes import TradingMode
from src.models.regime_detector import RegimeDetector
from src.orchestrator import TradingOrchestrator, _validate_mode_transition
from src.risk.position_sizer import PositionSizeResult


# ─────────────────────────────────────────────────────────────────────────────
#  Fixtures / Hilfsfunktionen
# ─────────────────────────────────────────────────────────────────────────────

def _make_size_result(lot=0.1, valid=True, rejection=None):
    return PositionSizeResult(
        symbol="EURUSD",
        lot_size=lot,
        risk_amount=100.0,
        stop_loss_distance=0.001,
        is_valid=valid,
        rejection_reason=rejection,
    )


def _make_features(close=1.09, atr=0.001) -> pd.DataFrame:
    return pd.DataFrame([{"close": close, "atr": atr}])


def _make_orch(
    signal="long",
    risk_allowed=True,
    pre_trade_safe=(True, "ok"),
    corr_allowed=True,
    positions=None,
    size_result=None,
    open_result=None,
    features=None,
    reconciler=True,
    audit_log=True,
    emergency=True,
    balance=10_000.0,
    mode: TradingMode = TradingMode.AUTONOMOUS,
    regime_detector: Optional[RegimeDetector] = None,
    price_getter=None,
    sl_atr_multiplier: float = 1.5,
    tp_atr_multiplier: float = 2.0,
    atr_col: str = "atr",
    close_col: str = "close",
) -> tuple[TradingOrchestrator, dict]:
    """Baut einen Orchestrator mit ausschliesslich Mocks."""
    pipeline    = MagicMock()
    risk_guard  = MagicMock()
    pre_trade   = MagicMock()
    sig_model   = MagicMock()
    corr_guard  = MagicMock()
    pos_sizer   = MagicMock()
    executor    = MagicMock()
    rec         = MagicMock() if reconciler else None
    alog        = MagicMock() if audit_log  else None
    emerg       = MagicMock() if emergency  else None

    pipeline.run_batch.return_value = {"status": "ok"}
    risk_guard.is_trading_allowed.return_value = risk_allowed
    risk_guard.get_position_size_multiplier.return_value = 1.0
    pre_trade.is_safe_to_trade.return_value = pre_trade_safe
    sig_model.get_signal.return_value = signal
    corr_guard.can_open_position.return_value = corr_allowed
    executor.get_open_positions.return_value = positions or []
    pos_sizer.calculate_lot_size.return_value = size_result or _make_size_result()

    if open_result is None:
        open_result = {"ticket": 42, "symbol": "EURUSD", "direction": "buy",
                       "lot_size": 0.1, "status": "open"}
    executor.open_position.return_value = open_result

    feat_df = features if features is not None else _make_features()

    # AUTONOMOUS erfordert Umgebungsvariable; für Tests ohne monkeypatch setzen wir sie direkt.
    if mode == TradingMode.AUTONOMOUS:
        os.environ.setdefault("CONFIRM_AUTONOMOUS", "yes")

    orch = TradingOrchestrator(
        data_pipeline=pipeline,
        risk_guard=risk_guard,
        pre_trade_check=pre_trade,
        signal_model=sig_model,
        correlation_guard=corr_guard,
        position_sizer=pos_sizer,
        order_executor=executor,
        position_reconciler=rec,
        audit_log=alog,
        emergency_handler=emerg,
        features_loader=lambda sym: feat_df,
        balance_getter=lambda: balance,
        regime_detector=regime_detector,
        price_getter=price_getter,
        sl_atr_multiplier=sl_atr_multiplier,
        tp_atr_multiplier=tp_atr_multiplier,
        atr_col=atr_col,
        close_col=close_col,
        mode=mode,
    )

    mocks = {
        "pipeline": pipeline,
        "risk_guard": risk_guard,
        "pre_trade": pre_trade,
        "sig_model": sig_model,
        "corr_guard": corr_guard,
        "pos_sizer": pos_sizer,
        "executor": executor,
        "reconciler": rec,
        "audit_log": alog,
        "emergency": emerg,
    }
    return orch, mocks


# ─────────────────────────────────────────────────────────────────────────────
#  Happy Path
# ─────────────────────────────────────────────────────────────────────────────

class TestRunCycleHappyPath:

    def test_returns_dict(self):
        orch, _ = _make_orch()
        result = orch.run_cycle("EURUSD")
        assert isinstance(result, dict)

    def test_result_has_expected_keys(self):
        orch, _ = _make_orch()
        result = orch.run_cycle("EURUSD")
        for key in ("symbol", "signal", "action", "reason", "ticket", "lot_size", "step_stopped_at"):
            assert key in result

    def test_long_signal_opens_buy(self):
        orch, _ = _make_orch(signal="long")
        result = orch.run_cycle("EURUSD")
        assert result["action"] == "open_buy"

    def test_short_signal_opens_sell(self):
        orch, mocks = _make_orch(signal="short")
        result = orch.run_cycle("EURUSD")
        assert result["action"] == "open_sell"

    def test_open_position_called_with_direction_buy(self):
        orch, mocks = _make_orch(signal="long")
        orch.run_cycle("EURUSD")
        args = mocks["executor"].open_position.call_args
        assert args[0][1] == "buy"  # direction positional arg

    def test_open_position_called_with_direction_sell(self):
        orch, mocks = _make_orch(signal="short")
        orch.run_cycle("EURUSD")
        args = mocks["executor"].open_position.call_args
        assert args[0][1] == "sell"

    def test_ticket_in_result(self):
        orch, _ = _make_orch()
        result = orch.run_cycle("EURUSD")
        assert result["ticket"] == 42

    def test_lot_size_in_result(self):
        orch, _ = _make_orch(size_result=_make_size_result(lot=0.2))
        result = orch.run_cycle("EURUSD")
        assert result["lot_size"] is not None
        assert result["lot_size"] > 0

    def test_lot_size_multiplied_by_risk_guard_factor(self):
        orch, mocks = _make_orch(size_result=_make_size_result(lot=0.2))
        mocks["risk_guard"].get_position_size_multiplier.return_value = 0.5
        result = orch.run_cycle("EURUSD")
        assert result["lot_size"] == pytest.approx(0.1, abs=1e-6)
    def test_lot_size_multiplied_by_regime_detector(self):
        regime = MagicMock()
        regime.name = "trending"
        regime_multiplier = 0.5
        regime_detector = MagicMock()
        regime_detector.detect_regime.return_value = regime
        regime_detector.get_position_size_multiplier.return_value = regime_multiplier

        orch, mocks = _make_orch(
            size_result=_make_size_result(lot=0.2),
            regime_detector=regime_detector,
        )
        result = orch.run_cycle("EURUSD")
        assert result["lot_size"] == pytest.approx(0.1, abs=1e-6)
        assert result["regime"] == regime
        assert any(check["name"] == "RegimeDetector" for check in result["checks"])

    def test_live_execution_does_not_mark_paper(self):
        orch, mocks = _make_orch(size_result=_make_size_result(lot=0.2))
        mocks["executor"]._live = True

        result = orch.run_cycle("EURUSD")

        assert result.get("is_paper") is None
        mocks["executor"].open_position.assert_called_once()

    def test_risk_guard_update_balance_called_with_balance(self):
        orch, mocks = _make_orch(balance=12_345.0)
        orch.run_cycle("EURUSD")
        mocks["risk_guard"].update_balance.assert_called_once_with(12_345.0)

    def test_risk_guard_update_balance_called_even_when_blocked(self):
        orch, mocks = _make_orch(risk_allowed=False, balance=5_000.0)
        orch.run_cycle("EURUSD")
        mocks["risk_guard"].update_balance.assert_called_once_with(5_000.0)

    def test_step_stopped_at_is_none_on_success(self):
        orch, _ = _make_orch()
        result = orch.run_cycle("EURUSD")
        assert result["step_stopped_at"] is None

    def test_reason_signal_executed_on_success(self):
        orch, _ = _make_orch()
        result = orch.run_cycle("EURUSD")
        assert result["reason"] == "signal_executed"

    def test_audit_log_log_order_called_on_success(self):
        orch, mocks = _make_orch()
        orch.run_cycle("EURUSD")
        mocks["audit_log"].log_order.assert_called_once()

    def test_reconciler_sync_called_on_success(self):
        orch, mocks = _make_orch()
        orch.run_cycle("EURUSD")
        mocks["reconciler"].sync.assert_called_once()

    def test_symbol_in_result(self):
        orch, _ = _make_orch()
        result = orch.run_cycle("GBPUSD")
        assert result["symbol"] == "GBPUSD"

    def test_flat_signal_returns_flat_action(self):
        orch, _ = _make_orch(signal="flat")
        result = orch.run_cycle("EURUSD")
        assert result["action"] == "flat"
        assert result["step_stopped_at"] == "flat_signal"

    def test_flat_signal_no_order_placed(self):
        orch, mocks = _make_orch(signal="flat")
        orch.run_cycle("EURUSD")
        mocks["executor"].open_position.assert_not_called()

    def test_neutral_signal_does_not_trade_as_sell(self):
        """Regression: 'neutral' darf NICHT zu 'sell' gemappt werden (Zeile 248)."""
        orch, mocks = _make_orch(signal="neutral")
        result = orch.run_cycle("EURUSD")
        assert result["action"] == "flat"
        assert result["step_stopped_at"] == "flat_signal"
        mocks["executor"].open_position.assert_not_called()

    def test_unknown_signal_does_not_trade(self):
        """Unerwartete Signalwerte fuehren ebenfalls zu keinem Trade."""
        orch, mocks = _make_orch(signal="something_weird")
        result = orch.run_cycle("EURUSD")
        assert result["action"] == "flat"
        mocks["executor"].open_position.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
#  Schritt-Reihenfolge
# ─────────────────────────────────────────────────────────────────────────────

class TestStepOrder:

    def test_pipeline_called_before_risk_guard(self):
        call_log = []
        orch, mocks = _make_orch()
        mocks["pipeline"].run_batch.side_effect = lambda *a, **k: call_log.append("pipeline")
        mocks["risk_guard"].is_trading_allowed.side_effect = lambda: call_log.append("risk") or True
        orch.run_cycle("EURUSD")
        assert call_log.index("pipeline") < call_log.index("risk")

    def test_risk_guard_before_pre_trade(self):
        call_log = []
        orch, mocks = _make_orch()
        mocks["risk_guard"].is_trading_allowed.side_effect = lambda: call_log.append("risk") or True
        mocks["pre_trade"].is_safe_to_trade.side_effect = lambda s: call_log.append("pre_trade") or (True, "ok")
        orch.run_cycle("EURUSD")
        assert call_log.index("risk") < call_log.index("pre_trade")

    def test_pre_trade_before_signal(self):
        call_log = []
        orch, mocks = _make_orch()
        mocks["pre_trade"].is_safe_to_trade.side_effect = lambda s: call_log.append("pre_trade") or (True, "ok")
        mocks["sig_model"].get_signal.side_effect = lambda *a, **k: call_log.append("signal") or "long"
        orch.run_cycle("EURUSD")
        assert call_log.index("pre_trade") < call_log.index("signal")

    def test_signal_before_corr_guard(self):
        call_log = []
        orch, mocks = _make_orch()
        mocks["sig_model"].get_signal.side_effect = lambda *a, **k: call_log.append("signal") or "long"
        mocks["corr_guard"].can_open_position.side_effect = lambda *a: call_log.append("corr") or True
        orch.run_cycle("EURUSD")
        assert call_log.index("signal") < call_log.index("corr")

    def test_corr_guard_before_position_sizer(self):
        call_log = []
        orch, mocks = _make_orch()
        mocks["corr_guard"].can_open_position.side_effect = lambda *a: call_log.append("corr") or True
        mocks["pos_sizer"].calculate_lot_size.side_effect = lambda *a, **k: (call_log.append("sizer") or _make_size_result())
        orch.run_cycle("EURUSD")
        assert call_log.index("corr") < call_log.index("sizer")

    def test_position_sizer_before_open_position(self):
        call_log = []
        orch, mocks = _make_orch()
        mocks["pos_sizer"].calculate_lot_size.side_effect = lambda *a, **k: (call_log.append("sizer") or _make_size_result())
        mocks["executor"].open_position.side_effect = lambda *a, **kw: (call_log.append("open") or {"ticket": 1})
        orch.run_cycle("EURUSD")
        assert call_log.index("sizer") < call_log.index("open")


# ─────────────────────────────────────────────────────────────────────────────
#  RiskGuard blockiert
# ─────────────────────────────────────────────────────────────────────────────

class TestRiskGuardBlocks:

    def test_returns_risk_guard_blocked_reason(self):
        orch, _ = _make_orch(risk_allowed=False)
        result = orch.run_cycle("EURUSD")
        assert result["reason"] == "risk_guard_blocked"

    def test_step_stopped_at_risk_guard(self):
        orch, _ = _make_orch(risk_allowed=False)
        result = orch.run_cycle("EURUSD")
        assert result["step_stopped_at"] == "risk_guard"

    def test_pre_trade_not_called(self):
        orch, mocks = _make_orch(risk_allowed=False)
        orch.run_cycle("EURUSD")
        mocks["pre_trade"].is_safe_to_trade.assert_not_called()

    def test_signal_model_not_called(self):
        orch, mocks = _make_orch(risk_allowed=False)
        orch.run_cycle("EURUSD")
        mocks["sig_model"].get_signal.assert_not_called()

    def test_corr_guard_not_called(self):
        orch, mocks = _make_orch(risk_allowed=False)
        orch.run_cycle("EURUSD")
        mocks["corr_guard"].can_open_position.assert_not_called()

    def test_position_sizer_not_called(self):
        orch, mocks = _make_orch(risk_allowed=False)
        orch.run_cycle("EURUSD")
        mocks["pos_sizer"].calculate_lot_size.assert_not_called()

    def test_order_executor_not_called(self):
        orch, mocks = _make_orch(risk_allowed=False)
        orch.run_cycle("EURUSD")
        mocks["executor"].open_position.assert_not_called()

    def test_action_is_skipped(self):
        orch, _ = _make_orch(risk_allowed=False)
        result = orch.run_cycle("EURUSD")
        assert result["action"] == "skipped"


# ─────────────────────────────────────────────────────────────────────────────
#  PreTradeCheck blockiert
# ─────────────────────────────────────────────────────────────────────────────

class TestPreTradeCheckBlocks:

    def test_returns_pre_trade_reason(self):
        orch, _ = _make_orch(pre_trade_safe=(False, "Spread zu hoch"))
        result = orch.run_cycle("EURUSD")
        assert "pre_trade_check_failed" in result["reason"]
        assert "Spread zu hoch" in result["reason"]

    def test_step_stopped_at_pre_trade(self):
        orch, _ = _make_orch(pre_trade_safe=(False, "News"))
        result = orch.run_cycle("EURUSD")
        assert result["step_stopped_at"] == "pre_trade_check"

    def test_signal_not_called(self):
        orch, mocks = _make_orch(pre_trade_safe=(False, "x"))
        orch.run_cycle("EURUSD")
        mocks["sig_model"].get_signal.assert_not_called()

    def test_order_not_placed(self):
        orch, mocks = _make_orch(pre_trade_safe=(False, "x"))
        orch.run_cycle("EURUSD")
        mocks["executor"].open_position.assert_not_called()

    def test_risk_guard_was_called(self):
        orch, mocks = _make_orch(pre_trade_safe=(False, "x"))
        orch.run_cycle("EURUSD")
        mocks["risk_guard"].is_trading_allowed.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
#  CorrelationGuard blockiert
# ─────────────────────────────────────────────────────────────────────────────

class TestCorrelationGuardBlocks:

    def test_returns_corr_blocked_reason(self):
        orch, _ = _make_orch(corr_allowed=False)
        result = orch.run_cycle("EURUSD")
        assert result["reason"] == "correlation_guard_blocked"

    def test_step_stopped_at_correlation_guard(self):
        orch, _ = _make_orch(corr_allowed=False)
        result = orch.run_cycle("EURUSD")
        assert result["step_stopped_at"] == "correlation_guard"

    def test_signal_was_computed(self):
        orch, mocks = _make_orch(corr_allowed=False, signal="long")
        orch.run_cycle("EURUSD")
        mocks["sig_model"].get_signal.assert_called_once()

    def test_order_not_placed(self):
        orch, mocks = _make_orch(corr_allowed=False)
        orch.run_cycle("EURUSD")
        mocks["executor"].open_position.assert_not_called()

    def test_position_sizer_not_called(self):
        orch, mocks = _make_orch(corr_allowed=False)
        orch.run_cycle("EURUSD")
        mocks["pos_sizer"].calculate_lot_size.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
#  PositionSizer ungueltig
# ─────────────────────────────────────────────────────────────────────────────

class TestPositionSizerInvalid:

    def test_returns_sizer_invalid_reason(self):
        orch, _ = _make_orch(size_result=_make_size_result(valid=False, rejection="Zu klein"))
        result = orch.run_cycle("EURUSD")
        assert "position_sizer_invalid" in result["reason"]

    def test_step_stopped_at_position_sizer(self):
        orch, _ = _make_orch(size_result=_make_size_result(valid=False))
        result = orch.run_cycle("EURUSD")
        assert result["step_stopped_at"] == "position_sizer"

    def test_order_not_placed_when_sizer_invalid(self):
        orch, mocks = _make_orch(size_result=_make_size_result(valid=False))
        orch.run_cycle("EURUSD")
        mocks["executor"].open_position.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
#  Kein Features-DataFrame (DataPipeline liefert nichts)
# ─────────────────────────────────────────────────────────────────────────────

class TestNoFeaturesAvailable:

    def test_returns_no_features_reason(self):
        orch, mocks = _make_orch()
        orch._features_loader = lambda sym: None
        result = orch.run_cycle("EURUSD")
        assert result["reason"] == "no_features_available"
        assert result["step_stopped_at"] == "data_pipeline"

    def test_risk_guard_not_called_when_no_features(self):
        orch, mocks = _make_orch()
        orch._features_loader = lambda sym: None
        orch.run_cycle("EURUSD")
        mocks["risk_guard"].is_trading_allowed.assert_not_called()

    def test_empty_dataframe_treated_as_no_features(self):
        orch, _ = _make_orch(features=pd.DataFrame())
        result = orch.run_cycle("EURUSD")
        assert result["step_stopped_at"] == "data_pipeline"


# ─────────────────────────────────────────────────────────────────────────────
#  SL/TP-Berechnung
# ─────────────────────────────────────────────────────────────────────────────

class TestSlTpCalculation:

    def test_buy_sl_below_close(self):
        orch, mocks = _make_orch(signal="long")
        orch.run_cycle("EURUSD")
        _, _, lot, sl, tp = mocks["executor"].open_position.call_args[0]
        assert sl < 1.09  # close=1.09

    def test_buy_tp_above_close(self):
        orch, mocks = _make_orch(signal="long")
        orch.run_cycle("EURUSD")
        _, _, lot, sl, tp = mocks["executor"].open_position.call_args[0]
        assert tp > 1.09

    def test_sell_sl_above_close(self):
        orch, mocks = _make_orch(signal="short")
        orch.run_cycle("EURUSD")
        _, _, lot, sl, tp = mocks["executor"].open_position.call_args[0]
        assert sl > 1.09

    def test_sell_tp_below_close(self):
        orch, mocks = _make_orch(signal="short")
        orch.run_cycle("EURUSD")
        _, _, lot, sl, tp = mocks["executor"].open_position.call_args[0]
        assert tp < 1.09


# ─────────────────────────────────────────────────────────────────────────────
#  run_loop
# ─────────────────────────────────────────────────────────────────────────────

class TestRunLoop:

    def test_calls_run_cycle_for_each_symbol(self):
        orch, _ = _make_orch()
        call_log = []
        orch.run_cycle = lambda sym: call_log.append(sym) or {}

        thread = threading.Thread(
            target=orch.run_loop,
            args=(["EURUSD", "GBPUSD"], 0),
        )
        thread.start()
        time.sleep(0.05)
        orch.stop()
        thread.join(timeout=2)

        assert "EURUSD" in call_log
        assert "GBPUSD" in call_log

    def test_stop_exits_loop(self):
        orch, _ = _make_orch()
        orch.run_cycle = lambda sym: {}

        done = threading.Event()
        def _run():
            orch.run_loop(["EURUSD"], interval_seconds=60)
            done.set()

        thread = threading.Thread(target=_run)
        thread.start()
        time.sleep(0.05)
        orch.stop()
        assert done.wait(timeout=3), "run_loop haette nach stop() enden sollen"

    def test_exception_delegated_to_emergency_handler(self):
        orch, mocks = _make_orch()
        orch.run_cycle = MagicMock(side_effect=RuntimeError("boom"))

        thread = threading.Thread(
            target=orch.run_loop,
            args=(["EURUSD"], 0),
        )
        thread.start()
        time.sleep(0.15)
        orch.stop()
        thread.join(timeout=2)

        mocks["emergency"].handle_unhandled_exception.assert_called()

    def test_loop_runs_multiple_iterations(self):
        orch, _ = _make_orch()
        counter = {"n": 0}

        def _cycle(sym):
            counter["n"] += 1
            if counter["n"] >= 3:
                orch.stop()
            return {}

        orch.run_cycle = _cycle
        orch.run_loop(["EURUSD"], interval_seconds=0)
        assert counter["n"] >= 3


# ─────────────────────────────────────────────────────────────────────────────
#  Graceful Shutdown
# ─────────────────────────────────────────────────────────────────────────────

class TestGracefulShutdown:

    def test_stop_sets_stop_event(self):
        orch, _ = _make_orch()
        assert not orch._stop_event.is_set()
        orch.stop()
        assert orch._stop_event.is_set()

    def test_second_stop_is_idempotent(self):
        orch, _ = _make_orch()
        orch.stop()
        orch.stop()
        assert orch._stop_event.is_set()


# ─────────────────────────────────────────────────────────────────────────────
#  Optionale Komponenten (None)
# ─────────────────────────────────────────────────────────────────────────────

class TestOptionalComponents:

    def test_no_reconciler_still_works(self):
        orch, _ = _make_orch(reconciler=False)
        result = orch.run_cycle("EURUSD")
        assert result["action"] == "open_buy"

    def test_no_audit_log_still_works(self):
        orch, _ = _make_orch(audit_log=False)
        result = orch.run_cycle("EURUSD")
        assert result["action"] == "open_buy"

    def test_no_emergency_handler_loop_continues(self):
        """Ohne EmergencyHandler darf run_loop() niemals re-raisen – der Loop muss weiterlaufen."""
        orch, _mocks = _make_orch(emergency=False)
        call_count = {"n": 0}

        def _fail_then_stop(sym):
            call_count["n"] += 1
            if call_count["n"] <= 2:
                raise RuntimeError("transient error")
            orch.stop()
            return {}

        orch.run_cycle = _fail_then_stop
        # Must NOT raise – loop continues after exceptions
        orch.run_loop(["EURUSD"], interval_seconds=0)
        assert call_count["n"] >= 3  # loop continued past the failures

    def test_default_balance_used_when_no_getter(self):
        orch, mocks = _make_orch()
        orch._balance_getter = None
        orch.run_cycle("EURUSD")
        # PositionSizer und RiskGuard muessen mit 10000.0 aufgerufen worden sein
        call_args = mocks["pos_sizer"].calculate_lot_size.call_args
        assert call_args[0][0] == 10_000.0
        mocks["risk_guard"].update_balance.assert_called_once_with(10_000.0)

    def test_corr_guard_receives_open_positions(self):
        pos = [{"ticket": 1, "symbol": "GBPUSD", "direction": "buy"}]
        orch, mocks = _make_orch(positions=pos)
        orch.run_cycle("EURUSD")
        call_args = mocks["corr_guard"].can_open_position.call_args
        assert call_args[0][2] == pos


# ─────────────────────────────────────────────────────────────────────────────
#  Pause / Resume
# ─────────────────────────────────────────────────────────────────────────────

class TestPauseResume:

    def test_pause_sets_paused(self):
        orch, _ = _make_orch()
        orch.pause()
        assert orch.is_paused is True

    def test_resume_clears_paused(self):
        orch, _ = _make_orch()
        orch.pause()
        orch.resume()
        assert orch.is_paused is False

    def test_paused_cycle_returns_trading_paused(self):
        orch, _ = _make_orch()
        orch.pause()
        result = orch.run_cycle("EURUSD")
        assert result["reason"] == "trading_paused"
        assert result["step_stopped_at"] == "pause"

    def test_paused_cycle_no_order(self):
        orch, mocks = _make_orch()
        orch.pause()
        orch.run_cycle("EURUSD")
        mocks["executor"].open_position.assert_not_called()

    def test_resume_allows_cycle_again(self):
        orch, _ = _make_orch()
        orch.pause()
        orch.resume()
        result = orch.run_cycle("EURUSD")
        assert result["action"] == "open_buy"


# ─────────────────────────────────────────────────────────────────────────────
#  TradingModes
# ─────────────────────────────────────────────────────────────────────────────

def _make_orch_mode(
    mode: TradingMode,
    confirmation_callback=None,
    signal: str = "long",
    monkeypatch=None,
) -> tuple[TradingOrchestrator, dict]:
    """Wie _make_orch() aber mit Mode-Parameter."""
    pipeline    = MagicMock()
    risk_guard  = MagicMock()
    pre_trade   = MagicMock()
    sig_model   = MagicMock()
    corr_guard  = MagicMock()
    pos_sizer   = MagicMock()
    executor    = MagicMock()
    alog        = MagicMock()
    emerg       = MagicMock()

    pipeline.run_batch.return_value = {"status": "ok"}
    risk_guard.is_trading_allowed.return_value = True
    risk_guard.get_position_size_multiplier.return_value = 1.0
    pre_trade.is_safe_to_trade.return_value = (True, "ok")
    sig_model.get_signal.return_value = signal
    corr_guard.can_open_position.return_value = True
    executor.get_open_positions.return_value = []
    pos_sizer.calculate_lot_size.return_value = _make_size_result()
    executor.open_position.return_value = {
        "ticket": 42, "symbol": "EURUSD", "direction": "buy",
        "lot_size": 0.1, "status": "open",
    }

    if monkeypatch is not None and mode == TradingMode.AUTONOMOUS:
        monkeypatch.setenv("CONFIRM_AUTONOMOUS", "yes")

    orch = TradingOrchestrator(
        data_pipeline=pipeline,
        risk_guard=risk_guard,
        pre_trade_check=pre_trade,
        signal_model=sig_model,
        correlation_guard=corr_guard,
        position_sizer=pos_sizer,
        order_executor=executor,
        audit_log=alog,
        emergency_handler=emerg,
        features_loader=lambda sym: _make_features(),
        balance_getter=lambda: 10_000.0,
        mode=mode,
        confirmation_callback=confirmation_callback,
    )

    mocks = {
        "pipeline": pipeline, "risk_guard": risk_guard, "pre_trade": pre_trade,
        "sig_model": sig_model, "corr_guard": corr_guard, "pos_sizer": pos_sizer,
        "executor": executor, "audit_log": alog, "emergency": emerg,
    }
    return orch, mocks


class TestTradingModes:

    # ── SUGGEST_ONLY ──────────────────────────────────────────────────────────

    def test_suggest_only_is_default_orchestrator_mode(self):
        orch, _ = _make_orch_mode(TradingMode.SUGGEST_ONLY)
        assert orch.mode == TradingMode.SUGGEST_ONLY

    def test_suggest_only_action_is_suggested(self):
        orch, _ = _make_orch_mode(TradingMode.SUGGEST_ONLY)
        result = orch.run_cycle("EURUSD")
        assert result["action"] == "suggested"

    def test_suggest_only_no_order_placed(self):
        orch, mocks = _make_orch_mode(TradingMode.SUGGEST_ONLY)
        orch.run_cycle("EURUSD")
        mocks["executor"].open_position.assert_not_called()

    def test_suggest_only_reason(self):
        orch, _ = _make_orch_mode(TradingMode.SUGGEST_ONLY)
        result = orch.run_cycle("EURUSD")
        assert result["reason"] == "suggest_only_mode"

    def test_suggest_only_step_stopped_at(self):
        orch, _ = _make_orch_mode(TradingMode.SUGGEST_ONLY)
        result = orch.run_cycle("EURUSD")
        assert result["step_stopped_at"] == "mode_suggest_only"

    def test_suggest_only_lot_size_computed(self):
        orch, _ = _make_orch_mode(TradingMode.SUGGEST_ONLY)
        result = orch.run_cycle("EURUSD")
        assert result["lot_size"] is not None and result["lot_size"] > 0

    def test_suggest_only_signal_computed(self):
        orch, mocks = _make_orch_mode(TradingMode.SUGGEST_ONLY, signal="long")
        orch.run_cycle("EURUSD")
        mocks["sig_model"].get_signal.assert_called_once()

    # ── CONFIRM_REQUIRED ──────────────────────────────────────────────────────

    def test_confirm_required_with_approval_opens_order(self):
        class _CB:
            def confirm_order(self, *a) -> bool:
                return True

        orch, mocks = _make_orch_mode(TradingMode.CONFIRM_REQUIRED, confirmation_callback=_CB())
        result = orch.run_cycle("EURUSD")
        assert result["action"] == "open_buy"
        mocks["executor"].open_position.assert_called_once()

    def test_confirm_required_with_rejection_skips_order(self):
        class _CB:
            def confirm_order(self, *a) -> bool:
                return False

        orch, mocks = _make_orch_mode(TradingMode.CONFIRM_REQUIRED, confirmation_callback=_CB())
        result = orch.run_cycle("EURUSD")
        assert result["action"] == "skipped"
        mocks["executor"].open_position.assert_not_called()

    def test_confirm_required_no_callback_skips_order(self):
        orch, mocks = _make_orch_mode(TradingMode.CONFIRM_REQUIRED, confirmation_callback=None)
        result = orch.run_cycle("EURUSD")
        assert result["action"] == "skipped"
        mocks["executor"].open_position.assert_not_called()

    def test_confirm_required_rejection_reason(self):
        class _CB:
            def confirm_order(self, *a) -> bool:
                return False

        orch, _ = _make_orch_mode(TradingMode.CONFIRM_REQUIRED, confirmation_callback=_CB())
        result = orch.run_cycle("EURUSD")
        assert result["reason"] == "order_not_confirmed"

    def test_confirm_required_rejection_step_stopped_at(self):
        class _CB:
            def confirm_order(self, *a) -> bool:
                return False

        orch, _ = _make_orch_mode(TradingMode.CONFIRM_REQUIRED, confirmation_callback=_CB())
        result = orch.run_cycle("EURUSD")
        assert result["step_stopped_at"] == "confirmation"

    def test_confirm_required_callback_exception_skips_order(self):
        class _CB:
            def confirm_order(self, *a) -> bool:
                raise RuntimeError("dialog closed")

        orch, mocks = _make_orch_mode(TradingMode.CONFIRM_REQUIRED, confirmation_callback=_CB())
        result = orch.run_cycle("EURUSD")
        assert result["action"] == "skipped"
        mocks["executor"].open_position.assert_not_called()

    def test_confirm_required_callback_receives_correct_symbol(self):
        received = {}

        class _CB:
            def confirm_order(self, symbol, direction, lot_size, sl, tp) -> bool:
                received["symbol"] = symbol
                return True

        orch, _ = _make_orch_mode(TradingMode.CONFIRM_REQUIRED, confirmation_callback=_CB())
        orch.run_cycle("GBPUSD")
        assert received["symbol"] == "GBPUSD"

    # ── AUTONOMOUS ────────────────────────────────────────────────────────────

    def test_autonomous_with_env_opens_order(self, monkeypatch):
        monkeypatch.setenv("CONFIRM_AUTONOMOUS", "yes")
        orch, mocks = _make_orch_mode(TradingMode.AUTONOMOUS, monkeypatch=monkeypatch)
        result = orch.run_cycle("EURUSD")
        assert result["action"] == "open_buy"
        mocks["executor"].open_position.assert_called_once()

    def test_autonomous_without_env_skips_order(self, monkeypatch):
        monkeypatch.setenv("CONFIRM_AUTONOMOUS", "yes")  # needed for __init__
        orch, mocks = _make_orch_mode(TradingMode.AUTONOMOUS, monkeypatch=monkeypatch)
        monkeypatch.delenv("CONFIRM_AUTONOMOUS")          # remove before run_cycle
        result = orch.run_cycle("EURUSD")
        assert result["action"] == "skipped"
        mocks["executor"].open_position.assert_not_called()

    def test_autonomous_without_env_skips_reason(self, monkeypatch):
        monkeypatch.setenv("CONFIRM_AUTONOMOUS", "yes")
        orch, _ = _make_orch_mode(TradingMode.AUTONOMOUS, monkeypatch=monkeypatch)
        monkeypatch.delenv("CONFIRM_AUTONOMOUS")
        result = orch.run_cycle("EURUSD")
        assert result["reason"] == "autonomous_not_confirmed_env"

    def test_autonomous_init_without_env_raises(self, monkeypatch):
        monkeypatch.delenv("CONFIRM_AUTONOMOUS", raising=False)
        with pytest.raises(EnvironmentError, match="CONFIRM_AUTONOMOUS"):
            _make_orch_mode(TradingMode.AUTONOMOUS)

    # ── mode property / set_mode ──────────────────────────────────────────────

    def test_mode_property_returns_current_mode(self):
        orch, _ = _make_orch_mode(TradingMode.SUGGEST_ONLY)
        assert orch.mode == TradingMode.SUGGEST_ONLY

    def test_set_mode_changes_mode(self):
        orch, _ = _make_orch()
        orch.set_mode(TradingMode.CONFIRM_REQUIRED)
        assert orch.mode == TradingMode.CONFIRM_REQUIRED

    def test_set_mode_to_autonomous_without_env_raises(self, monkeypatch):
        monkeypatch.delenv("CONFIRM_AUTONOMOUS", raising=False)
        orch, _ = _make_orch(mode=TradingMode.SUGGEST_ONLY)
        with pytest.raises(EnvironmentError, match="CONFIRM_AUTONOMOUS"):
            orch.set_mode(TradingMode.AUTONOMOUS)

    def test_set_mode_to_autonomous_with_env_succeeds(self, monkeypatch):
        monkeypatch.setenv("CONFIRM_AUTONOMOUS", "yes")
        orch, _ = _make_orch()
        orch.set_mode(TradingMode.AUTONOMOUS)
        assert orch.mode == TradingMode.AUTONOMOUS

    def test_set_mode_logs_to_audit(self):
        orch, mocks = _make_orch()
        orch.set_mode(TradingMode.CONFIRM_REQUIRED)
        mocks["audit_log"].log_error.assert_called()

    def test_set_mode_suggest_to_confirm_then_back(self):
        orch, _ = _make_orch()
        orch.set_mode(TradingMode.CONFIRM_REQUIRED)
        orch.set_mode(TradingMode.SUGGEST_ONLY)
        assert orch.mode == TradingMode.SUGGEST_ONLY

    # ── _validate_mode_transition ─────────────────────────────────────────────

    def test_validate_suggest_always_ok(self, monkeypatch):
        monkeypatch.delenv("CONFIRM_AUTONOMOUS", raising=False)
        _validate_mode_transition(TradingMode.SUGGEST_ONLY)  # kein Fehler

    def test_validate_confirm_always_ok(self, monkeypatch):
        monkeypatch.delenv("CONFIRM_AUTONOMOUS", raising=False)
        _validate_mode_transition(TradingMode.CONFIRM_REQUIRED)  # kein Fehler

    def test_validate_autonomous_without_env_raises(self, monkeypatch):
        monkeypatch.delenv("CONFIRM_AUTONOMOUS", raising=False)
        with pytest.raises(EnvironmentError):
            _validate_mode_transition(TradingMode.AUTONOMOUS)

    def test_validate_autonomous_with_env_ok(self, monkeypatch):
        monkeypatch.setenv("CONFIRM_AUTONOMOUS", "yes")
        _validate_mode_transition(TradingMode.AUTONOMOUS)  # kein Fehler


# ─────────────────────────────────────────────────────────────────────────────
#  EmergencyStop
# ─────────────────────────────────────────────────────────────────────────────

class TestEmergencyStop:

    def test_emergency_stop_pauses_trading(self):
        orch, _ = _make_orch()
        orch.emergency_stop()
        assert orch.is_paused is True

    def test_emergency_stop_calls_handler_drawdown_limit(self):
        orch, mocks = _make_orch()
        orch.emergency_stop()
        mocks["emergency"].handle_drawdown_limit.assert_called_once()

    def test_emergency_stop_without_handler_closes_via_executor(self):
        orch, mocks = _make_orch(emergency=False)
        mocks["executor"].get_open_positions.return_value = [
            {"ticket": 1}, {"ticket": 2},
        ]
        orch.emergency_stop()
        assert mocks["executor"].close_position.call_count == 2

    def test_emergency_stop_without_handler_calls_get_open_positions(self):
        orch, mocks = _make_orch(emergency=False)
        mocks["executor"].get_open_positions.return_value = []
        orch.emergency_stop()
        mocks["executor"].get_open_positions.assert_called_once()

    def test_emergency_stop_cycle_returns_paused_after(self):
        orch, _ = _make_orch()
        orch.emergency_stop()
        result = orch.run_cycle("EURUSD")
        assert result["reason"] == "trading_paused"

    def test_emergency_stop_logs_to_audit(self):
        orch, mocks = _make_orch()
        orch.emergency_stop()
        mocks["audit_log"].log_error.assert_called()

    def test_emergency_stop_handler_exception_does_not_propagate(self):
        orch, mocks = _make_orch()
        mocks["emergency"].handle_drawdown_limit.side_effect = RuntimeError("handler down")
        orch.emergency_stop()  # Kein Fehler erwartet
        assert orch.is_paused is True

    def test_emergency_stop_executor_exception_does_not_propagate(self):
        orch, mocks = _make_orch(emergency=False)
        mocks["executor"].get_open_positions.side_effect = RuntimeError("mt5 down")
        orch.emergency_stop()  # Kein Fehler erwartet
        assert orch.is_paused is True


# ─────────────────────────────────────────────────────────────────────────────
#  Regression: SL/TP-Distanzen vs. ATR-Multiplikatoren (Long UND Short)
#  Anlass: XAUUSD-Short Ticket 446740295 – SL/TP-Verhaeltnis verzerrt.
# ─────────────────────────────────────────────────────────────────────────────

def _xauusd_size_result(atr: float, sl_mult: float) -> PositionSizeResult:
    """Mimik des PositionSizer-Ergebnisses: stop_loss_distance = atr * sl_mult."""
    return PositionSizeResult(
        symbol="XAUUSD",
        lot_size=0.1,
        risk_amount=20.0,
        stop_loss_distance=atr * sl_mult,
        is_valid=True,
    )


class TestSLTPDistanceRegression:
    """SL = 1xATR, TP = 2xATR, jeweils in die richtige Richtung gespiegelt."""

    _ATR    = 11.07
    _CLOSE  = 4066.53
    _SLMULT = 1.0
    _TPMULT = 2.0

    def _run(self, signal: str) -> "tuple[float, float]":
        orch, mocks = _make_orch(
            signal=signal,
            size_result=_xauusd_size_result(self._ATR, self._SLMULT),
            features=_make_features(close=self._CLOSE, atr=self._ATR),
            sl_atr_multiplier=self._SLMULT,
            tp_atr_multiplier=self._TPMULT,
        )
        orch.run_cycle("XAUUSD")
        args = mocks["executor"].open_position.call_args.args
        return args[3], args[4]   # sl_price, tp_price

    def test_short_sl_above_tp_below(self):
        sl, tp = self._run("short")
        assert sl > self._CLOSE   # SL ueber Eroeffnung (Short)
        assert tp < self._CLOSE   # TP unter Eroeffnung (Short)

    def test_short_distances_match_atr_multipliers(self):
        sl, tp = self._run("short")
        sl_dist = sl - self._CLOSE
        tp_dist = self._CLOSE - tp
        assert sl_dist == pytest.approx(self._ATR * self._SLMULT, abs=1e-4)
        assert tp_dist == pytest.approx(self._ATR * self._TPMULT, abs=1e-4)
        # TP doppelt so weit wie SL – NICHT umgekehrt (Kern des Bugs)
        assert tp_dist == pytest.approx(2 * sl_dist, abs=1e-4)
        assert tp_dist > sl_dist

    def test_long_sl_below_tp_above(self):
        sl, tp = self._run("long")
        assert sl < self._CLOSE   # SL unter Eroeffnung (Long)
        assert tp > self._CLOSE   # TP ueber Eroeffnung (Long)

    def test_long_distances_match_atr_multipliers(self):
        sl, tp = self._run("long")
        sl_dist = self._CLOSE - sl
        tp_dist = tp - self._CLOSE
        assert sl_dist == pytest.approx(self._ATR * self._SLMULT, abs=1e-4)
        assert tp_dist == pytest.approx(self._ATR * self._TPMULT, abs=1e-4)
        assert tp_dist == pytest.approx(2 * sl_dist, abs=1e-4)

    def test_long_and_short_distances_are_symmetric(self):
        sl_s, tp_s = self._run("short")
        sl_l, tp_l = self._run("long")
        # gleiche Distanzen, nur Richtung gespiegelt
        assert (sl_s - self._CLOSE) == pytest.approx(self._CLOSE - sl_l, abs=1e-4)
        assert (self._CLOSE - tp_s) == pytest.approx(tp_l - self._CLOSE, abs=1e-4)


class TestLiveTickAnchor:
    """SL/TP muessen am aktuellen Tick verankert werden – auch LIVE, nicht am
    veralteten Candle-Close (Ursache des ~20-Punkte-Versatzes Ticket 446740295)."""

    def test_live_mode_uses_tick_not_stale_close(self):
        stale_close, live_tick, atr = 4086.79, 4066.53, 11.07
        orch, mocks = _make_orch(
            signal="short",
            size_result=_xauusd_size_result(atr, 1.0),
            features=_make_features(close=stale_close, atr=atr),
            price_getter=lambda sym: live_tick,
            sl_atr_multiplier=1.0,
            tp_atr_multiplier=2.0,
        )
        mocks["executor"]._live = True   # LIVE-Modus
        orch.run_cycle("XAUUSD")
        sl, tp = mocks["executor"].open_position.call_args.args[3:5]
        # Anker = live_tick, NICHT stale_close
        assert sl == pytest.approx(live_tick + atr, abs=1e-4)       # 4077.60
        assert tp == pytest.approx(live_tick - 2 * atr, abs=1e-4)   # 4044.39
        # Gegenprobe: am Bug-Anker (stale_close) waere SL ~4097.86 (>10 Punkte weg)
        assert abs(sl - (stale_close + atr)) > 10

    def test_paper_mode_also_uses_tick(self):
        stale_close, live_tick, atr = 4086.79, 4066.53, 11.07
        orch, mocks = _make_orch(
            signal="short",
            size_result=_xauusd_size_result(atr, 1.0),
            features=_make_features(close=stale_close, atr=atr),
            price_getter=lambda sym: live_tick,
            sl_atr_multiplier=1.0,
            tp_atr_multiplier=2.0,
        )
        mocks["executor"]._live = False  # Paper-Modus
        orch.run_cycle("XAUUSD")
        sl, _tp = mocks["executor"].open_position.call_args.args[3:5]
        assert sl == pytest.approx(live_tick + atr, abs=1e-4)
