"""
tests/gui/test_watchdog_service.py
Tests fuer Issue #61: Watchdog-Integration.

Prueft:
  - WatchdogService: Initialisierung, Enable/Disable
  - Crash-Erkennung via BotControlsWidget.error_occurred
  - Restart-Limit (max. 3 / Stunde) mit Zeitfenster-Logik
  - Sauberes Beenden via shutdown()
  - Telegram-Alert beim Crash und beim Limit
  - BotControlsWidget.start() oeffentliche Methode
  - SettingsView Watchdog-Tab (Checkbox, Spinbox, Combo)
  - TradingStatusBar Watchdog-Indikator
  - MainWindow-Integration (Eigenschaft, closeEvent, Einstellungs-Propagation)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional
from unittest.mock import MagicMock, call, patch

import pytest

pytest_plugins = ["pytestqt"]

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QCheckBox, QWidget

from gui.services.watchdog_service import (
    AlertSender,
    LogOnlyAlertSender,
    STATUS_DISABLED,
    STATUS_LIMIT_REACHED,
    STATUS_RUNNING,
    WatchdogService,
)
from gui.widgets.bot_controls_widget import BotControlsWidget, BotState


# ─────────────────────────────────────────────────────────────────────────────
#  Hilfsfunktionen
# ─────────────────────────────────────────────────────────────────────────────

def _make_bot_controls(qtbot) -> BotControlsWidget:
    w = BotControlsWidget()
    qtbot.addWidget(w)
    return w


def _make_watchdog(qtbot, max_restarts=3, window=3600, delay=0,
                   alert=None) -> tuple[WatchdogService, BotControlsWidget]:
    bc = _make_bot_controls(qtbot)
    wd = WatchdogService(bc, alert_sender=alert,
                         max_restarts=max_restarts,
                         window_seconds=window,
                         restart_delay_ms=delay)
    return wd, bc


# ─────────────────────────────────────────────────────────────────────────────
#  1.  LogOnlyAlertSender
# ─────────────────────────────────────────────────────────────────────────────

class TestLogOnlyAlertSender:
    def test_implements_protocol(self):
        sender = LogOnlyAlertSender()
        assert isinstance(sender, AlertSender)

    def test_send_alert_does_not_raise(self):
        LogOnlyAlertSender().send_alert("Test")


# ─────────────────────────────────────────────────────────────────────────────
#  2.  WatchdogService – Initialisierung
# ─────────────────────────────────────────────────────────────────────────────

class TestWatchdogServiceInit:
    def test_creates_without_crash(self, qtbot):
        wd, _ = _make_watchdog(qtbot)

    def test_enabled_by_default(self, qtbot):
        wd, _ = _make_watchdog(qtbot)
        assert wd.is_enabled is True

    def test_restart_count_zero(self, qtbot):
        wd, _ = _make_watchdog(qtbot)
        assert wd.restart_count == 0

    def test_can_restart_true_initially(self, qtbot):
        wd, _ = _make_watchdog(qtbot)
        assert wd.can_restart is True

    def test_status_running_initially(self, qtbot):
        wd, _ = _make_watchdog(qtbot)
        assert wd.status == STATUS_RUNNING


# ─────────────────────────────────────────────────────────────────────────────
#  3.  WatchdogService – Enable / Disable
# ─────────────────────────────────────────────────────────────────────────────

class TestWatchdogServiceEnableDisable:
    def test_disable_sets_enabled_false(self, qtbot):
        wd, _ = _make_watchdog(qtbot)
        wd.disable()
        assert wd.is_enabled is False

    def test_disable_emits_status_disabled(self, qtbot):
        wd, _ = _make_watchdog(qtbot)
        statuses = []
        wd.status_changed.connect(statuses.append)
        wd.disable()
        assert STATUS_DISABLED in statuses

    def test_enable_sets_enabled_true(self, qtbot):
        wd, _ = _make_watchdog(qtbot)
        wd.disable()
        wd.enable()
        assert wd.is_enabled is True

    def test_enable_emits_status_running(self, qtbot):
        wd, _ = _make_watchdog(qtbot)
        wd.disable()
        statuses = []
        wd.status_changed.connect(statuses.append)
        wd.enable()
        assert STATUS_RUNNING in statuses

    def test_double_disable_emits_once(self, qtbot):
        wd, _ = _make_watchdog(qtbot)
        statuses = []
        wd.status_changed.connect(statuses.append)
        wd.disable()
        wd.disable()
        assert statuses.count(STATUS_DISABLED) == 1

    def test_double_enable_emits_once(self, qtbot):
        wd, _ = _make_watchdog(qtbot)
        wd.disable()
        statuses = []
        wd.status_changed.connect(statuses.append)
        wd.enable()
        wd.enable()
        assert statuses.count(STATUS_RUNNING) == 1

    def test_status_property_reflects_disable(self, qtbot):
        wd, _ = _make_watchdog(qtbot)
        wd.disable()
        assert wd.status == STATUS_DISABLED

    def test_status_property_reflects_enable(self, qtbot):
        wd, _ = _make_watchdog(qtbot)
        wd.disable()
        wd.enable()
        assert wd.status == STATUS_RUNNING


# ─────────────────────────────────────────────────────────────────────────────
#  4.  WatchdogService – Crash-Erkennung und Neustart
# ─────────────────────────────────────────────────────────────────────────────

class TestWatchdogServiceCrash:
    def test_crash_when_disabled_no_restart_triggered(self, qtbot):
        wd, bc = _make_watchdog(qtbot, delay=0)
        wd.disable()
        restarted = []
        wd.restart_triggered.connect(restarted.append)
        bc.error_occurred.emit("Crash!")
        qtbot.wait(50)
        assert restarted == []

    def test_crash_when_enabled_triggers_restart_signal(self, qtbot):
        wd, bc = _make_watchdog(qtbot, delay=0)
        restarted = []
        wd.restart_triggered.connect(restarted.append)
        bc.error_occurred.emit("Crash!")
        qtbot.waitUntil(lambda: len(restarted) == 1, timeout=1000)

    def test_restart_triggered_carries_count(self, qtbot):
        wd, bc = _make_watchdog(qtbot, delay=0)
        restarted = []
        wd.restart_triggered.connect(restarted.append)
        bc.error_occurred.emit("Crash!")
        qtbot.waitUntil(lambda: len(restarted) == 1, timeout=1000)
        assert restarted[0] == 1

    def test_restart_count_increments(self, qtbot):
        wd, bc = _make_watchdog(qtbot, delay=0)
        bc.error_occurred.emit("Crash!")
        qtbot.waitUntil(lambda: wd.restart_count == 1, timeout=1000)

    def test_alert_sent_on_crash(self, qtbot):
        alert = MagicMock()
        wd, bc = _make_watchdog(qtbot, delay=0, alert=alert)
        bc.error_occurred.emit("Fehler XY")
        qtbot.waitUntil(lambda: alert.send_alert.called, timeout=1000)
        alert.send_alert.assert_called_once()
        assert "Fehler XY" in alert.send_alert.call_args[0][0]

    def test_alert_message_contains_restart_number(self, qtbot):
        alert = MagicMock()
        wd, bc = _make_watchdog(qtbot, delay=0, alert=alert)
        bc.error_occurred.emit("err")
        qtbot.waitUntil(lambda: alert.send_alert.called, timeout=1000)
        assert "#1" in alert.send_alert.call_args[0][0]

    def test_disabled_crash_no_alert(self, qtbot):
        alert = MagicMock()
        wd, bc = _make_watchdog(qtbot, delay=0, alert=alert)
        wd.disable()
        bc.error_occurred.emit("Crash!")
        qtbot.wait(100)
        alert.send_alert.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
#  5.  WatchdogService – Restart-Limit
# ─────────────────────────────────────────────────────────────────────────────

class TestWatchdogServiceRestartLimit:
    def test_can_restart_false_after_limit(self, qtbot):
        wd, bc = _make_watchdog(qtbot, max_restarts=2, delay=0)
        bc.error_occurred.emit("1")
        bc.error_occurred.emit("2")
        qtbot.waitUntil(lambda: wd.restart_count == 2, timeout=2000)
        assert not wd.can_restart

    def test_limit_reached_emits_status(self, qtbot):
        statuses = []
        wd, bc = _make_watchdog(qtbot, max_restarts=1, delay=0)
        wd.status_changed.connect(statuses.append)
        bc.error_occurred.emit("1st crash")
        bc.error_occurred.emit("2nd crash")  # should hit limit
        qtbot.waitUntil(lambda: STATUS_LIMIT_REACHED in statuses, timeout=2000)

    def test_limit_alert_sent(self, qtbot):
        alert = MagicMock()
        wd, bc = _make_watchdog(qtbot, max_restarts=1, delay=0, alert=alert)
        bc.error_occurred.emit("crash1")
        bc.error_occurred.emit("crash2")
        qtbot.waitUntil(lambda: alert.send_alert.call_count >= 2, timeout=2000)
        # Second call is the limit-reached alert
        combined = " ".join(c[0][0] for c in alert.send_alert.call_args_list)
        assert "Manuelle Pruefung" in combined

    def test_no_restart_after_limit(self, qtbot):
        wd, bc = _make_watchdog(qtbot, max_restarts=1, delay=0)
        restarted = []
        wd.restart_triggered.connect(restarted.append)
        bc.error_occurred.emit("crash1")
        bc.error_occurred.emit("crash2")
        qtbot.waitUntil(lambda: wd.restart_count == 1, timeout=2000)
        qtbot.wait(100)
        assert len(restarted) == 1  # no second restart

    def test_restart_count_does_not_exceed_limit(self, qtbot):
        wd, bc = _make_watchdog(qtbot, max_restarts=2, delay=0)
        for _ in range(5):
            bc.error_occurred.emit("crash")
        qtbot.waitUntil(lambda: wd.restart_count == 2, timeout=2000)
        qtbot.wait(100)
        assert wd.restart_count == 2

    def test_old_restarts_outside_window_not_counted(self, qtbot):
        wd, bc = _make_watchdog(qtbot, max_restarts=2, window=1, delay=0)
        # Add an old restart outside the 1-second window
        wd._restart_times.append(
            datetime.now(timezone.utc) - timedelta(seconds=2)
        )
        assert wd.can_restart  # old restart not in window

    def test_window_empty_means_can_restart(self, qtbot):
        wd, bc = _make_watchdog(qtbot, max_restarts=3, delay=0)
        assert wd.can_restart


# ─────────────────────────────────────────────────────────────────────────────
#  6.  WatchdogService – Shutdown
# ─────────────────────────────────────────────────────────────────────────────

class TestWatchdogServiceShutdown:
    def test_shutdown_disconnects_crash_signal(self, qtbot):
        wd, bc = _make_watchdog(qtbot, delay=0)
        wd.shutdown()
        restarted = []
        wd.restart_triggered.connect(restarted.append)
        bc.error_occurred.emit("Crash nach Shutdown")
        qtbot.wait(100)
        assert restarted == []

    def test_double_shutdown_does_not_raise(self, qtbot):
        wd, bc = _make_watchdog(qtbot, delay=0)
        wd.shutdown()
        wd.shutdown()  # must not raise

    def test_shutdown_while_disabled(self, qtbot):
        wd, bc = _make_watchdog(qtbot, delay=0)
        wd.disable()
        wd.shutdown()  # must not raise


# ─────────────────────────────────────────────────────────────────────────────
#  7.  BotControlsWidget.start() – oeffentliche Methode
# ─────────────────────────────────────────────────────────────────────────────

class TestBotControlsWidgetStart:
    def test_start_method_exists(self, qtbot):
        bc = _make_bot_controls(qtbot)
        assert hasattr(bc, "start") and callable(bc.start)

    def test_start_does_nothing_without_orchestrator(self, qtbot):
        bc = _make_bot_controls(qtbot)
        bc.start()  # must not raise, stays STOPPED
        assert bc.bot_state == BotState.STOPPED

    def test_start_delegates_to_on_start(self, qtbot):
        bc = _make_bot_controls(qtbot)
        called = []
        orig = bc._on_start
        bc._on_start = lambda: called.append(1) or orig()
        bc.start()
        assert called == [1]


# ─────────────────────────────────────────────────────────────────────────────
#  8.  SettingsView – Watchdog-Tab
# ─────────────────────────────────────────────────────────────────────────────

class TestSettingsViewWatchdogTab:
    def test_watchdog_tab_exists(self, qtbot):
        from gui.views.settings_view import SettingsView, TAB_WATCHDOG
        sv = SettingsView(_confirm_fn=lambda t, m: True)
        qtbot.addWidget(sv)
        assert sv._tabs.tabText(TAB_WATCHDOG) == "Watchdog"

    def test_watchdog_enabled_checkbox_exists(self, qtbot):
        from gui.views.settings_view import SettingsView
        sv = SettingsView(_confirm_fn=lambda t, m: True)
        qtbot.addWidget(sv)
        cb = sv.findChild(QWidget, "watchdog_enabled_cb")
        assert cb is not None

    def test_watchdog_enabled_default_true(self, qtbot):
        from gui.views.settings_view import SettingsView
        sv = SettingsView(_confirm_fn=lambda t, m: True)
        qtbot.addWidget(sv)
        assert sv._watchdog_enabled_cb.isChecked() is True

    def test_watchdog_enabled_false_when_unchecked(self, qtbot):
        from gui.views.settings_view import SettingsView
        sv = SettingsView(_confirm_fn=lambda t, m: True)
        qtbot.addWidget(sv)
        sv._watchdog_enabled_cb.setChecked(False)
        current = sv._collect_current()
        assert current["watchdog_enabled"] is False

    def test_settings_saved_includes_watchdog_enabled(self, qtbot):
        from gui.views.settings_view import SettingsView
        saved = []
        sv = SettingsView(_confirm_fn=lambda t, m: True)
        qtbot.addWidget(sv)
        sv.settings_saved.connect(saved.append)
        sv._on_save_clicked()
        assert len(saved) == 1
        assert "watchdog_enabled" in saved[0]

    def test_watchdog_max_restarts_spin_exists(self, qtbot):
        from gui.views.settings_view import SettingsView
        sv = SettingsView(_confirm_fn=lambda t, m: True)
        qtbot.addWidget(sv)
        spin = sv.findChild(QWidget, "watchdog_max_restarts_spin")
        assert spin is not None

    def test_watchdog_window_combo_exists(self, qtbot):
        from gui.views.settings_view import SettingsView
        sv = SettingsView(_confirm_fn=lambda t, m: True)
        qtbot.addWidget(sv)
        combo = sv.findChild(QWidget, "watchdog_window_combo")
        assert combo is not None

    def test_collect_includes_max_restarts(self, qtbot):
        from gui.views.settings_view import SettingsView
        sv = SettingsView(_confirm_fn=lambda t, m: True)
        qtbot.addWidget(sv)
        sv._watchdog_max_restarts_spin.setValue(5)
        current = sv._collect_current()
        assert current["watchdog_max_restarts"] == 5

    def test_collect_includes_window_seconds(self, qtbot):
        from gui.views.settings_view import SettingsView
        sv = SettingsView(_confirm_fn=lambda t, m: True)
        qtbot.addWidget(sv)
        assert "watchdog_window_seconds" in sv._collect_current()

    def test_load_settings_restores_watchdog_enabled_false(self, qtbot):
        from gui.views.settings_view import SettingsView, _DEFAULT_SETTINGS
        sv = SettingsView(_confirm_fn=lambda t, m: True)
        qtbot.addWidget(sv)
        settings = dict(_DEFAULT_SETTINGS)
        settings["watchdog_enabled"] = False
        sv._load_settings(settings)
        assert sv._watchdog_enabled_cb.isChecked() is False


# ─────────────────────────────────────────────────────────────────────────────
#  9.  TradingStatusBar – Watchdog-Indikator
# ─────────────────────────────────────────────────────────────────────────────

class TestTradingStatusBarWatchdog:
    def test_watchdog_label_exists(self, qtbot):
        from gui.app import TradingStatusBar
        sb = TradingStatusBar()
        qtbot.addWidget(sb)
        assert sb.findChild(QWidget, "watchdog_status_label") is not None

    def test_default_shows_running(self, qtbot):
        from gui.app import TradingStatusBar
        sb = TradingStatusBar()
        qtbot.addWidget(sb)
        assert "aktiv" in sb.watchdog_label.text().lower()

    def test_set_status_running(self, qtbot):
        from gui.app import TradingStatusBar
        sb = TradingStatusBar()
        qtbot.addWidget(sb)
        sb.set_watchdog_status("running")
        assert "aktiv" in sb.watchdog_label.text().lower()

    def test_set_status_disabled(self, qtbot):
        from gui.app import TradingStatusBar
        sb = TradingStatusBar()
        qtbot.addWidget(sb)
        sb.set_watchdog_status("disabled")
        assert "aus" in sb.watchdog_label.text().lower()

    def test_set_status_limit_reached(self, qtbot):
        from gui.app import TradingStatusBar
        sb = TradingStatusBar()
        qtbot.addWidget(sb)
        sb.set_watchdog_status("limit_reached")
        assert "limit" in sb.watchdog_label.text().lower()

    def test_watchdog_label_property(self, qtbot):
        from gui.app import TradingStatusBar
        from PySide6.QtWidgets import QLabel
        sb = TradingStatusBar()
        qtbot.addWidget(sb)
        assert isinstance(sb.watchdog_label, QLabel)


# ─────────────────────────────────────────────────────────────────────────────
#  10. MainWindow-Integration
# ─────────────────────────────────────────────────────────────────────────────

class TestMainWindowWatchdogIntegration:
    def test_has_watchdog_service_property(self, qtbot):
        from gui.app import MainWindow
        win = MainWindow()
        qtbot.addWidget(win)
        assert hasattr(win, "watchdog_service")

    def test_watchdog_service_is_watchdog_service(self, qtbot):
        from gui.app import MainWindow
        win = MainWindow()
        qtbot.addWidget(win)
        assert isinstance(win.watchdog_service, WatchdogService)

    def test_watchdog_service_enabled_by_default(self, qtbot):
        from gui.app import MainWindow
        win = MainWindow()
        qtbot.addWidget(win)
        assert win.watchdog_service.is_enabled is True

    def test_close_event_calls_shutdown(self, qtbot):
        from gui.app import MainWindow
        win = MainWindow()
        qtbot.addWidget(win)
        shutdown_called = []
        orig = win._watchdog_service.shutdown
        win._watchdog_service.shutdown = lambda: shutdown_called.append(1) or orig()
        win.close()
        assert shutdown_called == [1]

    def test_settings_saved_disables_watchdog(self, qtbot):
        from gui.app import MainWindow
        win = MainWindow()
        qtbot.addWidget(win)
        # Emit settings_saved with watchdog_enabled=False
        from gui.views.settings_view import _DEFAULT_SETTINGS
        settings = dict(_DEFAULT_SETTINGS)
        settings["watchdog_enabled"] = False
        win._settings_view.settings_saved.emit(settings)
        assert win.watchdog_service.is_enabled is False

    def test_settings_saved_enables_watchdog(self, qtbot):
        from gui.app import MainWindow
        win = MainWindow()
        qtbot.addWidget(win)
        win.watchdog_service.disable()
        from gui.views.settings_view import _DEFAULT_SETTINGS
        settings = dict(_DEFAULT_SETTINGS)
        settings["watchdog_enabled"] = True
        win._settings_view.settings_saved.emit(settings)
        assert win.watchdog_service.is_enabled is True

    def test_watchdog_status_reflected_in_status_bar(self, qtbot):
        from gui.app import MainWindow
        win = MainWindow()
        qtbot.addWidget(win)
        win.watchdog_service.disable()
        qtbot.wait(50)
        assert "aus" in win.trading_status_bar.watchdog_label.text().lower()
