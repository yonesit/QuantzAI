"""
tests/gui/test_console_widget.py
Tests fuer Issue #60: Bot-Logs und Fehler in der GUI sichtbar.

Prueft:
  - LogEntry-Datenhaltung
  - GuiLogSink: Eintrag-Erstellung, Ausnahme-Sicherheit
  - ConsoleWidget: Initialisierung, UI-Elemente
  - Farbcodierung nach Level
  - Filter nach Level und Modul
  - Auto-Scroll Toggle
  - Emergency-Banner (CRITICAL + EmergencyHandler)
  - Export als Textdatei
  - Clear / Ring-Puffer
  - Thread-sichere Zustellung via QueuedConnection
  - Loguru-Integration: logger.warning() erscheint im Widget
  - MainWindow hat console_widget-Property
"""

from __future__ import annotations

import os
import tempfile
import threading
from datetime import datetime
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

pytest_plugins = ["pytestqt"]

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget

from gui.widgets.console_widget import (
    ConsoleWidget,
    GuiLogSink,
    LogEntry,
    _LEVEL_ORDER,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Hilfsfunktionen
# ─────────────────────────────────────────────────────────────────────────────

def _entry(
    level:   str = "INFO",
    module:  str = "src.orchestrator",
    message: str = "Test-Nachricht",
) -> LogEntry:
    return LogEntry(
        timestamp=datetime.now(),
        level=level,
        module=module,
        message=message,
    )


def _wait_for_entry(widget: ConsoleWidget, qtbot, count: int = 1, timeout: int = 2000) -> None:
    qtbot.waitUntil(lambda: widget.entry_count >= count, timeout=timeout)


# ─────────────────────────────────────────────────────────────────────────────
#  1.  LogEntry
# ─────────────────────────────────────────────────────────────────────────────

class TestLogEntry:
    def test_creation(self):
        now = datetime.now()
        e = LogEntry(timestamp=now, level="INFO", module="src.foo", message="hello")
        assert e.level == "INFO"
        assert e.module == "src.foo"
        assert e.message == "hello"
        assert e.timestamp is now

    def test_all_levels_valid(self):
        for lvl in _LEVEL_ORDER:
            e = LogEntry(datetime.now(), lvl, "mod", "msg")
            assert e.level == lvl

    def test_empty_module(self):
        e = LogEntry(datetime.now(), "WARNING", "", "msg")
        assert e.module == ""

    def test_multiline_message(self):
        e = LogEntry(datetime.now(), "ERROR", "mod", "line1\nline2")
        assert "line1" in e.message


# ─────────────────────────────────────────────────────────────────────────────
#  2.  GuiLogSink
# ─────────────────────────────────────────────────────────────────────────────

class TestGuiLogSink:
    def _make_record(self, level="INFO", name="src.test", message="hello"):
        """Baut ein minimales loguru-aehnliches Message-Objekt."""
        record = {
            "level": MagicMock(name=level),
            "name":  name,
            "message": message,
            "time":  MagicMock(),
        }
        record["level"].name = level
        msg = MagicMock()
        msg.record = record
        return msg

    def test_callable(self):
        sink = GuiLogSink(lambda e: None)
        assert callable(sink)

    def test_creates_log_entry(self):
        received = []
        sink = GuiLogSink(received.append)
        sink(self._make_record())
        assert len(received) == 1
        assert isinstance(received[0], LogEntry)

    def test_level_passed_correctly(self):
        received = []
        sink = GuiLogSink(received.append)
        sink(self._make_record(level="WARNING"))
        assert received[0].level == "WARNING"

    def test_warn_normalized_to_warning(self):
        received = []
        sink = GuiLogSink(received.append)
        sink(self._make_record(level="WARN"))
        assert received[0].level == "WARNING"

    def test_module_passed_correctly(self):
        received = []
        sink = GuiLogSink(received.append)
        sink(self._make_record(name="src.risk.risk_guard"))
        assert received[0].module == "src.risk.risk_guard"

    def test_message_passed_correctly(self):
        received = []
        sink = GuiLogSink(received.append)
        sink(self._make_record(message="Drawdown-Limit erreicht"))
        assert received[0].message == "Drawdown-Limit erreicht"

    def test_callback_exception_does_not_raise(self):
        def _bad(e):
            raise RuntimeError("test")
        sink = GuiLogSink(_bad)
        sink(self._make_record())  # darf nicht crashen


# ─────────────────────────────────────────────────────────────────────────────
#  3.  ConsoleWidget – Initialisierung
# ─────────────────────────────────────────────────────────────────────────────

class TestConsoleWidgetInit:
    def test_creates_without_crash(self, qtbot):
        w = ConsoleWidget()
        qtbot.addWidget(w)

    def test_object_name(self, qtbot):
        w = ConsoleWidget()
        qtbot.addWidget(w)
        assert w.objectName() == "console_widget"

    def test_has_log_display(self, qtbot):
        w = ConsoleWidget()
        qtbot.addWidget(w)
        disp = w.findChild(QWidget, "console_log_display")
        assert disp is not None

    def test_has_level_combo(self, qtbot):
        w = ConsoleWidget()
        qtbot.addWidget(w)
        combo = w.findChild(QWidget, "console_level_combo")
        assert combo is not None

    def test_has_module_filter(self, qtbot):
        w = ConsoleWidget()
        qtbot.addWidget(w)
        flt = w.findChild(QWidget, "console_module_filter")
        assert flt is not None

    def test_has_autoscroll_checkbox(self, qtbot):
        w = ConsoleWidget()
        qtbot.addWidget(w)
        cb = w.findChild(QWidget, "console_autoscroll_checkbox")
        assert cb is not None

    def test_has_export_button(self, qtbot):
        w = ConsoleWidget()
        qtbot.addWidget(w)
        btn = w.findChild(QWidget, "console_export_btn")
        assert btn is not None

    def test_has_clear_button(self, qtbot):
        w = ConsoleWidget()
        qtbot.addWidget(w)
        btn = w.findChild(QWidget, "console_clear_btn")
        assert btn is not None

    def test_has_emergency_banner_hidden(self, qtbot):
        w = ConsoleWidget()
        qtbot.addWidget(w)
        banner = w.findChild(QWidget, "console_emergency_banner")
        assert banner is not None
        assert not banner.isVisible()

    def test_default_min_level_info(self, qtbot):
        w = ConsoleWidget()
        qtbot.addWidget(w)
        assert w.min_level == "INFO"

    def test_custom_min_level(self, qtbot):
        w = ConsoleWidget(min_level="WARNING")
        qtbot.addWidget(w)
        assert w.min_level == "WARNING"

    def test_autoscroll_on_by_default(self, qtbot):
        w = ConsoleWidget()
        qtbot.addWidget(w)
        assert w.auto_scroll is True

    def test_entry_count_zero_initially(self, qtbot):
        w = ConsoleWidget()
        qtbot.addWidget(w)
        assert w.entry_count == 0

    def test_display_empty_initially(self, qtbot):
        w = ConsoleWidget()
        qtbot.addWidget(w)
        assert w.display_text == ""

    def test_emergency_banner_not_visible_initially(self, qtbot):
        w = ConsoleWidget()
        qtbot.addWidget(w)
        assert not w.emergency_banner_visible


# ─────────────────────────────────────────────────────────────────────────────
#  4.  ConsoleWidget – Eintraege hinzufuegen
# ─────────────────────────────────────────────────────────────────────────────

class TestConsoleWidgetAppend:
    def test_append_entry_appears_in_display(self, qtbot):
        w = ConsoleWidget()
        qtbot.addWidget(w)
        w.append_entry(_entry(message="Hallo Welt"))
        _wait_for_entry(w, qtbot)
        assert "Hallo Welt" in w.display_text

    def test_append_entry_increments_count(self, qtbot):
        w = ConsoleWidget()
        qtbot.addWidget(w)
        w.append_entry(_entry())
        _wait_for_entry(w, qtbot)
        assert w.entry_count == 1

    def test_append_multiple_entries(self, qtbot):
        w = ConsoleWidget()
        qtbot.addWidget(w)
        for i in range(5):
            w.append_entry(_entry(message=f"msg{i}"))
        _wait_for_entry(w, qtbot, count=5)
        for i in range(5):
            assert f"msg{i}" in w.display_text

    def test_level_in_display(self, qtbot):
        w = ConsoleWidget()
        qtbot.addWidget(w)
        w.append_entry(_entry(level="WARNING", message="Warnung!"))
        _wait_for_entry(w, qtbot)
        assert "WARNING" in w.display_text

    def test_module_in_display(self, qtbot):
        w = ConsoleWidget()
        qtbot.addWidget(w)
        w.append_entry(_entry(module="src.risk.risk_guard"))
        _wait_for_entry(w, qtbot)
        assert "src.risk.risk_guard" in w.display_text

    def test_timestamp_in_display(self, qtbot):
        w = ConsoleWidget()
        qtbot.addWidget(w)
        w.append_entry(_entry())
        _wait_for_entry(w, qtbot)
        # Zeitstempel-Format [HH:MM:SS]
        import re
        assert re.search(r"\d{2}:\d{2}:\d{2}", w.display_text)

    def test_append_from_background_thread(self, qtbot):
        w = ConsoleWidget()
        qtbot.addWidget(w)

        def _send():
            w.append_entry(_entry(message="Thread-Nachricht"))

        t = threading.Thread(target=_send)
        t.start()
        t.join(timeout=2)
        _wait_for_entry(w, qtbot)
        assert "Thread-Nachricht" in w.display_text


# ─────────────────────────────────────────────────────────────────────────────
#  5.  ConsoleWidget – Level-Filter
# ─────────────────────────────────────────────────────────────────────────────

class TestConsoleWidgetLevelFilter:
    def test_debug_hidden_when_min_level_info(self, qtbot):
        w = ConsoleWidget(min_level="INFO")
        qtbot.addWidget(w)
        w.append_entry(_entry(level="DEBUG", message="debug-msg"))
        qtbot.wait(200)
        assert "debug-msg" not in w.display_text

    def test_debug_buffered_when_filtered(self, qtbot):
        w = ConsoleWidget(min_level="INFO")
        qtbot.addWidget(w)
        w.append_entry(_entry(level="DEBUG", message="debug-msg"))
        qtbot.wait(200)
        assert w.entry_count == 1  # gepuffert aber nicht angezeigt

    def test_warning_visible_when_min_info(self, qtbot):
        w = ConsoleWidget(min_level="INFO")
        qtbot.addWidget(w)
        w.append_entry(_entry(level="WARNING", message="warn-msg"))
        _wait_for_entry(w, qtbot)
        assert "warn-msg" in w.display_text

    def test_filter_change_rebuilds_display(self, qtbot):
        w = ConsoleWidget(min_level="WARNING")
        qtbot.addWidget(w)
        w.append_entry(_entry(level="INFO", message="info-msg"))
        qtbot.wait(200)
        assert "info-msg" not in w.display_text

        # Filter auf DEBUG setzen → info-msg erscheint
        w._level_combo.setCurrentText("DEBUG")
        qtbot.wait(100)
        assert "info-msg" in w.display_text

    def test_error_visible_at_any_level(self, qtbot):
        w = ConsoleWidget(min_level="ERROR")
        qtbot.addWidget(w)
        w.append_entry(_entry(level="ERROR", message="error-msg"))
        _wait_for_entry(w, qtbot)
        assert "error-msg" in w.display_text

    def test_critical_always_visible(self, qtbot):
        w = ConsoleWidget(min_level="CRITICAL")
        qtbot.addWidget(w)
        w.append_entry(_entry(level="CRITICAL", message="crit-msg"))
        _wait_for_entry(w, qtbot)
        assert "crit-msg" in w.display_text


# ─────────────────────────────────────────────────────────────────────────────
#  6.  ConsoleWidget – Modul-Filter
# ─────────────────────────────────────────────────────────────────────────────

class TestConsoleWidgetModuleFilter:
    def test_module_filter_hides_non_matching(self, qtbot):
        w = ConsoleWidget()
        qtbot.addWidget(w)
        w._module_filter.setText("risk")
        w.append_entry(_entry(module="src.orchestrator", message="orch-msg"))
        qtbot.wait(200)
        assert "orch-msg" not in w.display_text

    def test_module_filter_shows_matching(self, qtbot):
        w = ConsoleWidget()
        qtbot.addWidget(w)
        w._module_filter.setText("risk")
        w.append_entry(_entry(module="src.risk.risk_guard", message="risk-msg"))
        _wait_for_entry(w, qtbot)
        assert "risk-msg" in w.display_text

    def test_module_filter_case_insensitive(self, qtbot):
        w = ConsoleWidget()
        qtbot.addWidget(w)
        w._module_filter.setText("RISK")
        w.append_entry(_entry(module="src.risk.risk_guard", message="risk-msg"))
        _wait_for_entry(w, qtbot)
        assert "risk-msg" in w.display_text

    def test_clear_module_filter_shows_all(self, qtbot):
        w = ConsoleWidget()
        qtbot.addWidget(w)
        w._module_filter.setText("risk")
        w.append_entry(_entry(module="src.orchestrator", message="orch-msg"))
        qtbot.wait(200)
        assert "orch-msg" not in w.display_text

        w._module_filter.clear()
        qtbot.wait(100)
        assert "orch-msg" in w.display_text

    def test_module_filter_rebuilds_from_buffer(self, qtbot):
        w = ConsoleWidget()
        qtbot.addWidget(w)
        w.append_entry(_entry(module="src.orchestrator", message="orch"))
        w.append_entry(_entry(module="src.risk.guard",   message="risk"))
        _wait_for_entry(w, qtbot, count=2)

        w._module_filter.setText("risk")
        qtbot.wait(100)
        assert "risk" in w.display_text
        assert "orch" not in w.display_text


# ─────────────────────────────────────────────────────────────────────────────
#  7.  ConsoleWidget – Auto-Scroll
# ─────────────────────────────────────────────────────────────────────────────

class TestConsoleWidgetAutoScroll:
    def test_autoscroll_default_on(self, qtbot):
        w = ConsoleWidget()
        qtbot.addWidget(w)
        assert w.auto_scroll is True

    def test_autoscroll_toggle_off(self, qtbot):
        w = ConsoleWidget()
        qtbot.addWidget(w)
        w._autoscroll_cb.setChecked(False)
        assert w.auto_scroll is False

    def test_autoscroll_toggle_back_on(self, qtbot):
        w = ConsoleWidget()
        qtbot.addWidget(w)
        w._autoscroll_cb.setChecked(False)
        w._autoscroll_cb.setChecked(True)
        assert w.auto_scroll is True

    def test_entries_still_appended_without_autoscroll(self, qtbot):
        w = ConsoleWidget()
        qtbot.addWidget(w)
        w._autoscroll_cb.setChecked(False)
        w.append_entry(_entry(message="no-scroll"))
        _wait_for_entry(w, qtbot)
        assert "no-scroll" in w.display_text


# ─────────────────────────────────────────────────────────────────────────────
#  8.  ConsoleWidget – Emergency-Banner
# ─────────────────────────────────────────────────────────────────────────────

class TestConsoleWidgetEmergencyBanner:
    def test_banner_shown_on_critical(self, qtbot):
        w = ConsoleWidget()
        qtbot.addWidget(w)
        w.append_entry(_entry(level="CRITICAL", message="Notfall!"))
        qtbot.waitUntil(lambda: w.emergency_banner_visible, timeout=2000)

    def test_banner_message_set(self, qtbot):
        w = ConsoleWidget()
        qtbot.addWidget(w)
        w.append_entry(_entry(level="CRITICAL", message="Notfall-Texxt"))
        qtbot.waitUntil(lambda: w.emergency_banner_visible, timeout=2000)
        assert "Notfall-Texxt" in w._emergency_label.text()

    def test_banner_shown_on_emergency_handler_message(self, qtbot):
        w = ConsoleWidget()
        qtbot.addWidget(w)
        w.append_entry(_entry(level="ERROR", module="src.execution.emergency",
                              message="EmergencyHandler: Drawdown-Limit"))
        qtbot.waitUntil(lambda: w.emergency_banner_visible, timeout=2000)

    def test_banner_not_shown_for_normal_error(self, qtbot):
        w = ConsoleWidget()
        qtbot.addWidget(w)
        w.append_entry(_entry(level="ERROR", message="Normaler Fehler"))
        qtbot.wait(300)
        assert not w.emergency_banner_visible

    def test_banner_dismiss_hides_it(self, qtbot):
        w = ConsoleWidget()
        qtbot.addWidget(w)
        w.append_entry(_entry(level="CRITICAL", message="Notfall"))
        qtbot.waitUntil(lambda: w.emergency_banner_visible, timeout=2000)
        dismiss = w.findChild(QWidget, "console_emergency_dismiss_btn")
        qtbot.mouseClick(dismiss, Qt.MouseButton.LeftButton)
        assert not w.emergency_banner_visible

    def test_banner_shown_for_critical_drawdown_keyword(self, qtbot):
        w = ConsoleWidget()
        qtbot.addWidget(w)
        w.append_entry(_entry(level="ERROR", message="CRITICAL_DRAWDOWN erreicht"))
        qtbot.waitUntil(lambda: w.emergency_banner_visible, timeout=2000)

    def test_multiple_critical_updates_banner(self, qtbot):
        w = ConsoleWidget()
        qtbot.addWidget(w)
        w.append_entry(_entry(level="CRITICAL", message="Erster Notfall"))
        qtbot.waitUntil(lambda: w.emergency_banner_visible, timeout=2000)
        w.append_entry(_entry(level="CRITICAL", message="Zweiter Notfall"))
        qtbot.wait(200)
        assert "Zweiter Notfall" in w._emergency_label.text()


# ─────────────────────────────────────────────────────────────────────────────
#  9.  ConsoleWidget – Export
# ─────────────────────────────────────────────────────────────────────────────

class TestConsoleWidgetExport:
    def test_export_to_file_writes_content(self, tmp_path, qtbot):
        w = ConsoleWidget()
        qtbot.addWidget(w)
        w.append_entry(_entry(message="Export-Test"))
        _wait_for_entry(w, qtbot)
        path = str(tmp_path / "log.txt")
        w.export_to_file(path)
        content = open(path, encoding="utf-8").read()
        assert "Export-Test" in content

    def test_export_creates_file(self, tmp_path, qtbot):
        w = ConsoleWidget()
        qtbot.addWidget(w)
        path = str(tmp_path / "export.txt")
        w.export_to_file(path)
        assert os.path.exists(path)

    def test_export_contains_all_visible_lines(self, tmp_path, qtbot):
        w = ConsoleWidget()
        qtbot.addWidget(w)
        for i in range(3):
            w.append_entry(_entry(message=f"Zeile{i}"))
        _wait_for_entry(w, qtbot, count=3)
        path = str(tmp_path / "log.txt")
        w.export_to_file(path)
        content = open(path, encoding="utf-8").read()
        for i in range(3):
            assert f"Zeile{i}" in content

    def test_export_empty_widget(self, tmp_path, qtbot):
        w = ConsoleWidget()
        qtbot.addWidget(w)
        path = str(tmp_path / "empty.txt")
        w.export_to_file(path)
        assert os.path.exists(path)

    def test_export_respects_filter(self, tmp_path, qtbot):
        w = ConsoleWidget(min_level="WARNING")
        qtbot.addWidget(w)
        w.append_entry(_entry(level="INFO",    message="info-msg"))
        w.append_entry(_entry(level="WARNING", message="warn-msg"))
        _wait_for_entry(w, qtbot, count=2)
        path = str(tmp_path / "filtered.txt")
        w.export_to_file(path)
        content = open(path, encoding="utf-8").read()
        assert "warn-msg" in content
        assert "info-msg" not in content


# ─────────────────────────────────────────────────────────────────────────────
#  10. ConsoleWidget – Clear + Ring-Puffer
# ─────────────────────────────────────────────────────────────────────────────

class TestConsoleWidgetClearAndBuffer:
    def test_clear_empties_display(self, qtbot):
        w = ConsoleWidget()
        qtbot.addWidget(w)
        w.append_entry(_entry(message="vor-clear"))
        _wait_for_entry(w, qtbot)
        qtbot.mouseClick(w._clear_btn, Qt.MouseButton.LeftButton)
        assert w.display_text == ""

    def test_clear_empties_buffer(self, qtbot):
        w = ConsoleWidget()
        qtbot.addWidget(w)
        w.append_entry(_entry())
        _wait_for_entry(w, qtbot)
        qtbot.mouseClick(w._clear_btn, Qt.MouseButton.LeftButton)
        assert w.entry_count == 0

    def test_ring_buffer_max_lines(self, qtbot):
        w = ConsoleWidget(max_lines=5)
        qtbot.addWidget(w)
        for i in range(10):
            w.append_entry(_entry(message=f"m{i}"))
        _wait_for_entry(w, qtbot, count=5)
        assert w.entry_count == 5

    def test_ring_buffer_drops_oldest(self, qtbot):
        w = ConsoleWidget(max_lines=3)
        qtbot.addWidget(w)
        for i in range(5):
            w.append_entry(_entry(message=f"msg{i}"))
        _wait_for_entry(w, qtbot, count=3)
        # Nur die letzten 3 im Puffer
        messages_in_buffer = [e.message for e in w._buffer]
        assert "msg0" not in messages_in_buffer
        assert "msg4" in messages_in_buffer


# ─────────────────────────────────────────────────────────────────────────────
#  11. Loguru-Integration
# ─────────────────────────────────────────────────────────────────────────────

class TestLoguruIntegration:
    def test_loguru_warning_appears_in_widget(self, qtbot):
        from loguru import logger
        w = ConsoleWidget(min_level="DEBUG")
        qtbot.addWidget(w)
        sink = GuiLogSink(w.append_entry)
        sink_id = logger.add(sink, level="DEBUG")
        try:
            logger.warning("LOGURU-TEST-WARNUNG-59a3f")
            _wait_for_entry(w, qtbot, count=1)
            assert "LOGURU-TEST-WARNUNG-59a3f" in w.display_text
        finally:
            logger.remove(sink_id)

    def test_loguru_error_triggers_emergency_check(self, qtbot):
        from loguru import logger
        w = ConsoleWidget(min_level="DEBUG")
        qtbot.addWidget(w)
        sink = GuiLogSink(w.append_entry)
        sink_id = logger.add(sink, level="DEBUG")
        try:
            logger.critical("EmergencyHandler: CRITICAL_DRAWDOWN Test")
            qtbot.waitUntil(lambda: w.emergency_banner_visible, timeout=2000)
        finally:
            logger.remove(sink_id)

    def test_loguru_debug_filtered_at_info_level(self, qtbot):
        from loguru import logger
        w = ConsoleWidget(min_level="INFO")
        qtbot.addWidget(w)
        sink = GuiLogSink(w.append_entry)
        sink_id = logger.add(sink, level="DEBUG")
        try:
            logger.debug("DEBUG-NUR-INTERN-x9q7")
            qtbot.wait(300)
            assert "DEBUG-NUR-INTERN-x9q7" not in w.display_text
        finally:
            logger.remove(sink_id)

    def test_multiple_loguru_messages(self, qtbot):
        from loguru import logger
        w = ConsoleWidget(min_level="DEBUG")
        qtbot.addWidget(w)
        sink = GuiLogSink(w.append_entry)
        sink_id = logger.add(sink, level="DEBUG")
        try:
            for i in range(3):
                logger.info(f"LOGURU-MULTI-{i}")
            _wait_for_entry(w, qtbot, count=3)
            for i in range(3):
                assert f"LOGURU-MULTI-{i}" in w.display_text
        finally:
            logger.remove(sink_id)


# ─────────────────────────────────────────────────────────────────────────────
#  12. MainWindow-Integration
# ─────────────────────────────────────────────────────────────────────────────

class TestMainWindowIntegration:
    def test_main_window_has_console_widget_property(self, qtbot):
        from gui.app import MainWindow
        win = MainWindow()
        qtbot.addWidget(win)
        assert hasattr(win, "console_widget")

    def test_console_widget_is_console_widget_instance(self, qtbot):
        from gui.app import MainWindow
        win = MainWindow()
        qtbot.addWidget(win)
        assert isinstance(win.console_widget, ConsoleWidget)

    def test_activity_log_still_accessible(self, qtbot):
        from gui.app import MainWindow
        from gui.widgets.activity_log_widget import ActivityLogWidget
        win = MainWindow()
        qtbot.addWidget(win)
        assert isinstance(win.activity_log, ActivityLogWidget)
