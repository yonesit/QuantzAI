"""
tests/unit/test_emergency.py
Unit-Tests fuer EmergencyHandler (4 Fehlerreaktionen)
und Integration-Test fuer den Watchdog.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from src.execution.emergency import (
    AlertSender,
    EmergencyHandler,
    LogOnlyAlertSender,
)
from src.data.data_router import PriceDiscrepancyError, EmergencyModeError
from src.execution.order_executor import OrderError

# scripts/ liegt nicht im Python-Pfad, daher direkter Import via importlib
import importlib.util as _ilu

_watchdog_spec = _ilu.spec_from_file_location(
    "watchdog",
    Path(__file__).parents[2] / "scripts" / "watchdog.py",
)
_watchdog_mod = _ilu.module_from_spec(_watchdog_spec)
_watchdog_spec.loader.exec_module(_watchdog_mod)
Watchdog = _watchdog_mod.Watchdog


# ─────────────────────────────────────────────────────────────────────────────
#  Hilfsfunktion – baut einen EmergencyHandler mit gemockten Abhaengigkeiten
# ─────────────────────────────────────────────────────────────────────────────

def _make_handler(
    positions=None,
    close_side_effect=None,
    drawdown_hit=True,
    router_side_effect=None,
    exit_fn=None,
    alert_sender=None,
    audit_log_path=None,
):
    executor   = MagicMock()
    router     = MagicMock()
    risk_guard = MagicMock()

    executor.get_open_positions.return_value = positions or []

    if close_side_effect is not None:
        executor.close_position.side_effect = close_side_effect
    else:
        executor.close_position.return_value = {"status": "closed"}

    risk_guard.is_max_drawdown_hit.return_value = drawdown_hit

    if router_side_effect is not None:
        router.get_connector.side_effect = router_side_effect
    else:
        router.get_connector.return_value = MagicMock()

    alert = alert_sender or MagicMock()
    exit_mock = exit_fn or MagicMock()

    handler = EmergencyHandler(
        executor=executor,
        data_router=router,
        risk_guard=risk_guard,
        alert_sender=alert,
        audit_log_path=audit_log_path,
        _exit_fn=exit_mock,
    )
    return handler, executor, router, risk_guard, alert, exit_mock


# ─────────────────────────────────────────────────────────────────────────────
#  Fehler-Reaktion 1: handle_mt5_unreachable
# ─────────────────────────────────────────────────────────────────────────────

class TestHandleMT5Unreachable:

    def test_returns_dict_with_expected_keys(self):
        handler, *_ = _make_handler()
        result = handler.handle_mt5_unreachable()
        assert set(result.keys()) == {"open_positions", "oanda_fallback", "trading_paused"}

    def test_calls_data_router_get_connector(self):
        handler, _, router, *_ = _make_handler()
        handler.handle_mt5_unreachable()
        router.get_connector.assert_called_once()

    def test_passes_first_symbol_to_get_connector(self):
        handler, _, router, *_ = _make_handler()
        handler.handle_mt5_unreachable(symbols=["EURUSD"])
        router.get_connector.assert_called_once_with(symbol="EURUSD")

    def test_passes_none_when_no_symbols(self):
        handler, _, router, *_ = _make_handler()
        handler.handle_mt5_unreachable()
        router.get_connector.assert_called_once_with(symbol=None)

    def test_oanda_fallback_true_when_connector_ok(self):
        handler, *_ = _make_handler()
        result = handler.handle_mt5_unreachable()
        assert result["oanda_fallback"] is True

    def test_oanda_fallback_false_on_price_discrepancy(self):
        handler, *_ = _make_handler(
            router_side_effect=PriceDiscrepancyError("Zu gross")
        )
        result = handler.handle_mt5_unreachable()
        assert result["oanda_fallback"] is False

    def test_oanda_fallback_false_on_emergency_mode_error(self):
        handler, *_ = _make_handler(
            router_side_effect=EmergencyModeError("Kein Connector")
        )
        result = handler.handle_mt5_unreachable()
        assert result["oanda_fallback"] is False

    def test_trading_not_paused_when_oanda_ok(self):
        handler, *_ = _make_handler()
        result = handler.handle_mt5_unreachable()
        assert result["trading_paused"] is False
        assert handler.is_trading_paused is False

    def test_trading_paused_on_price_discrepancy(self):
        handler, *_ = _make_handler(
            router_side_effect=PriceDiscrepancyError("Zu gross")
        )
        result = handler.handle_mt5_unreachable()
        assert result["trading_paused"] is True
        assert handler.is_trading_paused is True

    def test_trading_paused_on_emergency_mode_error(self):
        handler, *_ = _make_handler(
            router_side_effect=EmergencyModeError("Kein Connector")
        )
        result = handler.handle_mt5_unreachable()
        assert result["trading_paused"] is True

    def test_open_positions_count_in_result(self):
        positions = [
            {"ticket": 1, "symbol": "EURUSD"},
            {"ticket": 2, "symbol": "GBPUSD"},
        ]
        handler, *_ = _make_handler(positions=positions)
        result = handler.handle_mt5_unreachable()
        assert result["open_positions"] == 2

    def test_open_positions_zero_when_none_open(self):
        handler, *_ = _make_handler(positions=[])
        result = handler.handle_mt5_unreachable()
        assert result["open_positions"] == 0

    def test_audit_log_written_to_file(self, tmp_path):
        audit_file = tmp_path / "audit.log"
        handler, *_ = _make_handler(audit_log_path=str(audit_file))
        handler.handle_mt5_unreachable()
        content = audit_file.read_text(encoding="utf-8")
        assert "MT5_UNREACHABLE" in content

    def test_executor_get_positions_called(self):
        handler, executor, *_ = _make_handler()
        handler.handle_mt5_unreachable()
        executor.get_open_positions.assert_called()


# ─────────────────────────────────────────────────────────────────────────────
#  Fehler-Reaktion 2: handle_bad_datafeed
# ─────────────────────────────────────────────────────────────────────────────

class TestHandleBadDatafeed:

    def test_pauses_trading_immediately(self):
        handler, *_ = _make_handler()
        assert handler.is_trading_paused is False
        handler.handle_bad_datafeed(symbol="EURUSD", reason="Leerer DataFrame")
        assert handler.is_trading_paused is True

    def test_pauses_without_symbol(self):
        handler, *_ = _make_handler()
        handler.handle_bad_datafeed()
        assert handler.is_trading_paused is True

    def test_pauses_without_reason(self):
        handler, *_ = _make_handler()
        handler.handle_bad_datafeed(symbol="GBPUSD")
        assert handler.is_trading_paused is True

    def test_audit_log_contains_bad_datafeed(self, tmp_path):
        audit_file = tmp_path / "audit.log"
        handler, *_ = _make_handler(audit_log_path=str(audit_file))
        handler.handle_bad_datafeed(symbol="EURUSD", reason="NaN-Werte")
        content = audit_file.read_text(encoding="utf-8")
        assert "BAD_DATAFEED" in content

    def test_returns_none(self):
        handler, *_ = _make_handler()
        result = handler.handle_bad_datafeed()
        assert result is None

    def test_not_paused_initially(self):
        handler, *_ = _make_handler()
        assert handler.is_trading_paused is False

    def test_executor_not_called(self):
        handler, executor, *_ = _make_handler()
        handler.handle_bad_datafeed()
        executor.close_position.assert_not_called()

    def test_resume_clears_pause(self):
        handler, *_ = _make_handler()
        handler.handle_bad_datafeed()
        assert handler.is_trading_paused is True
        handler.resume_trading()
        assert handler.is_trading_paused is False


# ─────────────────────────────────────────────────────────────────────────────
#  Fehler-Reaktion 3: handle_critical_drawdown
# ─────────────────────────────────────────────────────────────────────────────

class TestHandleCriticalDrawdown:

    def test_returns_dict_with_expected_keys(self):
        handler, *_ = _make_handler()
        result = handler.handle_critical_drawdown()
        assert set(result.keys()) == {"closed_tickets", "errors", "trading_paused"}

    def test_trading_paused_after_call(self):
        handler, *_ = _make_handler()
        handler.handle_critical_drawdown()
        assert handler.is_trading_paused is True

    def test_result_trading_paused_is_true(self):
        handler, *_ = _make_handler()
        result = handler.handle_critical_drawdown()
        assert result["trading_paused"] is True

    def test_calls_is_max_drawdown_hit(self):
        handler, _, _, risk_guard, *_ = _make_handler()
        handler.handle_critical_drawdown()
        risk_guard.is_max_drawdown_hit.assert_called_once()

    def test_calls_close_position_for_each_open_position(self):
        positions = [
            {"ticket": 101, "symbol": "EURUSD"},
            {"ticket": 202, "symbol": "GBPUSD"},
            {"ticket": 303, "symbol": "USDJPY"},
        ]
        handler, executor, *_ = _make_handler(positions=positions)
        handler.handle_critical_drawdown()
        assert executor.close_position.call_count == 3
        executor.close_position.assert_any_call(101)
        executor.close_position.assert_any_call(202)
        executor.close_position.assert_any_call(303)

    def test_returns_closed_tickets(self):
        positions = [
            {"ticket": 101, "symbol": "EURUSD"},
            {"ticket": 202, "symbol": "GBPUSD"},
        ]
        handler, *_ = _make_handler(positions=positions)
        result = handler.handle_critical_drawdown()
        assert sorted(result["closed_tickets"]) == [101, 202]

    def test_returns_empty_closed_tickets_when_no_positions(self):
        handler, *_ = _make_handler(positions=[])
        result = handler.handle_critical_drawdown()
        assert result["closed_tickets"] == []
        assert result["errors"] == []

    def test_errors_when_close_position_raises(self):
        positions = [{"ticket": 101, "symbol": "EURUSD"}]
        handler, *_ = _make_handler(
            positions=positions,
            close_side_effect=OrderError("Margin"),
        )
        result = handler.handle_critical_drawdown()
        assert result["closed_tickets"] == []
        assert len(result["errors"]) == 1
        assert "101" in result["errors"][0]

    def test_partial_close_failure(self):
        positions = [
            {"ticket": 101, "symbol": "EURUSD"},
            {"ticket": 202, "symbol": "GBPUSD"},
        ]
        side_effects = [None, OrderError("Margin-Fehler")]

        def _close(ticket):
            effect = side_effects.pop(0)
            if effect is not None:
                raise effect

        handler, executor, *_ = _make_handler(positions=positions)
        executor.close_position.side_effect = _close
        result = handler.handle_critical_drawdown()
        assert result["closed_tickets"] == [101]
        assert len(result["errors"]) == 1

    def test_audit_log_contains_drawdown_entry(self, tmp_path):
        audit_file = tmp_path / "audit.log"
        handler, *_ = _make_handler(audit_log_path=str(audit_file))
        handler.handle_critical_drawdown()
        content = audit_file.read_text(encoding="utf-8")
        assert "CRITICAL_DRAWDOWN" in content
        assert "ALL_POSITIONS_CLOSED" in content

    def test_resume_clears_drawdown_pause(self):
        handler, *_ = _make_handler()
        handler.handle_critical_drawdown()
        assert handler.is_trading_paused is True
        handler.resume_trading()
        assert handler.is_trading_paused is False


# ─────────────────────────────────────────────────────────────────────────────
#  Fehler-Reaktion 4: handle_unhandled_exception
# ─────────────────────────────────────────────────────────────────────────────

class TestHandleUnhandledException:

    def test_calls_exit_fn_with_1(self):
        handler, _, _, _, _, exit_mock = _make_handler()
        handler.handle_unhandled_exception(RuntimeError("Testfehler"))
        exit_mock.assert_called_once_with(1)

    def test_sends_alert(self):
        handler, _, _, _, alert, _ = _make_handler()
        handler.handle_unhandled_exception(ValueError("Etwas kaputt"))
        alert.send_alert.assert_called_once()

    def test_alert_contains_exception_type(self):
        handler, _, _, _, alert, _ = _make_handler()
        handler.handle_unhandled_exception(TypeError("Typfehler"))
        msg = alert.send_alert.call_args[0][0]
        assert "TypeError" in msg

    def test_alert_contains_exception_message(self):
        handler, _, _, _, alert, _ = _make_handler()
        handler.handle_unhandled_exception(RuntimeError("kritischer Fehler"))
        msg = alert.send_alert.call_args[0][0]
        assert "kritischer Fehler" in msg

    def test_closes_all_positions_before_exit(self):
        positions = [
            {"ticket": 77, "symbol": "EURUSD"},
            {"ticket": 88, "symbol": "GBPUSD"},
        ]
        handler, executor, *_ = _make_handler(positions=positions)
        handler.handle_unhandled_exception(RuntimeError("crash"))
        assert executor.close_position.call_count == 2
        executor.close_position.assert_any_call(77)
        executor.close_position.assert_any_call(88)

    def test_closes_before_sending_alert(self):
        order_log = []
        positions = [{"ticket": 55, "symbol": "EURUSD"}]
        handler, executor, _, _, alert, _ = _make_handler(positions=positions)
        executor.close_position.side_effect = lambda t: order_log.append(("close", t))
        alert.send_alert.side_effect = lambda m: order_log.append(("alert", m))

        handler.handle_unhandled_exception(RuntimeError("x"))
        actions = [a[0] for a in order_log]
        assert actions.index("close") < actions.index("alert")

    def test_alert_sent_even_with_no_positions(self):
        handler, _, _, _, alert, _ = _make_handler(positions=[])
        handler.handle_unhandled_exception(RuntimeError("leer"))
        alert.send_alert.assert_called_once()

    def test_exit_called_even_when_close_fails(self):
        positions = [{"ticket": 1, "symbol": "EURUSD"}]
        handler, _, _, _, _, exit_mock = _make_handler(
            positions=positions,
            close_side_effect=OrderError("Fehler"),
        )
        handler.handle_unhandled_exception(RuntimeError("crash"))
        exit_mock.assert_called_once_with(1)

    def test_audit_log_contains_unhandled_exception(self, tmp_path):
        audit_file = tmp_path / "audit.log"
        handler, *_ = _make_handler(audit_log_path=str(audit_file))
        handler.handle_unhandled_exception(RuntimeError("boom"))
        content = audit_file.read_text(encoding="utf-8")
        assert "UNHANDLED_EXCEPTION" in content
        assert "PROCESS_EXIT" in content


# ─────────────────────────────────────────────────────────────────────────────
#  AlertSender Protocol
# ─────────────────────────────────────────────────────────────────────────────

class TestAlertSenderProtocol:

    def test_log_only_alert_sender_satisfies_protocol(self):
        sender = LogOnlyAlertSender()
        assert isinstance(sender, AlertSender)

    def test_concrete_class_with_send_alert_satisfies_protocol(self):
        class ConcreteAlert:
            def send_alert(self, message: str) -> None:
                pass

        assert isinstance(ConcreteAlert(), AlertSender)

    def test_object_without_send_alert_fails_protocol(self):
        class NoAlert:
            pass
        assert not isinstance(NoAlert(), AlertSender)


# ─────────────────────────────────────────────────────────────────────────────
#  Watchdog – Unit-Tests (ohne echter Subprocess-Ausfuehrung)
# ─────────────────────────────────────────────────────────────────────────────

class TestWatchdogUnit:

    def test_restart_count_zero_initially(self):
        wd = Watchdog(["echo", "test"])
        assert wd.restart_count == 0

    def test_can_restart_when_no_restarts(self):
        wd = Watchdog(["echo", "test"], max_restarts=3)
        assert wd._can_restart() is True

    def test_can_restart_below_limit(self):
        wd = Watchdog(["echo"], max_restarts=3, restart_window_seconds=3600)
        wd._restart_times.append(datetime.now(timezone.utc))
        wd._restart_times.append(datetime.now(timezone.utc))
        assert wd._can_restart() is True  # 2 < 3

    def test_cannot_restart_at_limit(self):
        wd = Watchdog(["echo"], max_restarts=3, restart_window_seconds=3600)
        now = datetime.now(timezone.utc)
        wd._restart_times = [now, now, now]
        assert wd._can_restart() is False  # 3 >= 3

    def test_old_restarts_outside_window_dont_count(self):
        wd = Watchdog(["echo"], max_restarts=2, restart_window_seconds=60)
        old = datetime.now(timezone.utc) - timedelta(seconds=120)
        wd._restart_times = [old, old, old]  # alle ausserhalb des Fensters
        assert wd._can_restart() is True

    def test_run_returns_0_on_normal_exit(self):
        wd = Watchdog(["dummy"], max_restarts=3)
        mock_proc = MagicMock()
        mock_proc.wait.return_value = 0

        with patch.object(wd, "_start_process", return_value=mock_proc):
            result = wd.run()
        assert result == 0

    def test_run_returns_1_after_max_restarts(self):
        wd = Watchdog(["dummy"], max_restarts=2, restart_window_seconds=3600)
        mock_proc = MagicMock()
        mock_proc.wait.side_effect = [1, 1, 1]  # immer Crash

        with patch.object(wd, "_start_process", return_value=mock_proc):
            result = wd.run()
        assert result == 1

    def test_alert_sent_when_limit_reached(self):
        alert = MagicMock()
        wd = Watchdog(["dummy"], alert_sender=alert, max_restarts=1, restart_window_seconds=3600)
        mock_proc = MagicMock()
        mock_proc.wait.side_effect = [1, 1]

        with patch.object(wd, "_start_process", return_value=mock_proc):
            wd.run()

        alert.send_alert.assert_called_once()

    def test_no_alert_on_normal_exit(self):
        alert = MagicMock()
        wd = Watchdog(["dummy"], alert_sender=alert, max_restarts=3)
        mock_proc = MagicMock()
        mock_proc.wait.return_value = 0

        with patch.object(wd, "_start_process", return_value=mock_proc):
            wd.run()

        alert.send_alert.assert_not_called()

    def test_restart_count_incremented(self):
        wd = Watchdog(["dummy"], max_restarts=2, restart_window_seconds=3600)
        mock_proc = MagicMock()
        mock_proc.wait.side_effect = [1, 1, 1]

        with patch.object(wd, "_start_process", return_value=mock_proc):
            wd.run()

        assert wd.restart_count == 2


# ─────────────────────────────────────────────────────────────────────────────
#  Watchdog – Integration-Test (echter Subprocess)
# ─────────────────────────────────────────────────────────────────────────────

class TestWatchdogIntegration:

    def test_restarts_crashing_process_and_stops(self, tmp_path):
        """Watchdog startet einen crashenden Prozess und stoppt nach max_restarts."""
        crasher = tmp_path / "crasher.py"
        crasher.write_text("import sys\nsys.exit(1)\n")

        alert = MagicMock()
        wd = Watchdog(
            command=[sys.executable, str(crasher)],
            alert_sender=alert,
            max_restarts=2,
            restart_window_seconds=3600,
        )
        result = wd.run()

        assert result == 1
        assert wd.restart_count == 2
        alert.send_alert.assert_called_once()

    def test_does_not_restart_on_clean_exit(self, tmp_path):
        """Watchdog beendet sich sauber wenn Prozess mit Exit 0 endet."""
        normal = tmp_path / "normal.py"
        normal.write_text("import sys\nsys.exit(0)\n")

        alert = MagicMock()
        wd = Watchdog(
            command=[sys.executable, str(normal)],
            alert_sender=alert,
            max_restarts=3,
            restart_window_seconds=3600,
        )
        result = wd.run()

        assert result == 0
        assert wd.restart_count == 0
        alert.send_alert.assert_not_called()

    def test_alert_message_contains_command(self, tmp_path):
        """Die Alert-Nachricht nennt den gestoppten Prozess."""
        crasher = tmp_path / "crasher.py"
        crasher.write_text("import sys\nsys.exit(1)\n")

        alert = MagicMock()
        wd = Watchdog(
            command=[sys.executable, str(crasher)],
            alert_sender=alert,
            max_restarts=1,
            restart_window_seconds=3600,
        )
        wd.run()

        msg = alert.send_alert.call_args[0][0]
        assert "crasher.py" in msg

    def test_three_crashes_within_window_triggers_stop(self, tmp_path):
        """Standard-Konfiguration: 3 Crashes in 1h -> Stopp."""
        crasher = tmp_path / "crasher.py"
        crasher.write_text("import sys\nsys.exit(1)\n")

        alert = MagicMock()
        wd = Watchdog(
            command=[sys.executable, str(crasher)],
            alert_sender=alert,
            max_restarts=3,
            restart_window_seconds=3600,
        )
        result = wd.run()

        assert result == 1
        assert wd.restart_count == 3
        alert.send_alert.assert_called_once()
