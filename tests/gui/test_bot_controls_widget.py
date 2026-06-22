"""
tests/gui/test_bot_controls_widget.py
pytest-qt Tests fuer gui/widgets/bot_controls_widget.py
und die Integration in gui/app.py.

Abgedeckt:
  BotState
    - Enum-Werte vorhanden (STOPPED, RUNNING, PAUSED, STOPPING)

  BotWorker
    - run() ruft orchestrator.run_loop() auf
    - stopped-Signal nach normalem Ende
    - error_occurred-Signal bei Exception
    - stopped-Signal auch nach Exception

  BotControlsWidget
    - Erstellt ohne Absturz (mit und ohne Orchestrator)
    - Initaler Zustand: STOPPED
    - Buttons ohne Orchestrator alle deaktiviert
    - Buttons mit Orchestrator: Start=enabled, Stop/Pause=disabled
    - _on_start: Zustand RUNNING, Start-Btn disabled, Stop/Pause enabled
    - _on_stop:  Zustand STOPPING, alle Buttons disabled, orchestrator.stop() aufgerufen
    - STOPPING -> STOPPED nach Thread-Ende (worker_stopped)
    - _on_pause_resume: RUNNING -> PAUSED -> RUNNING
    - Pause/Resume-Button Text wechselt
    - state_changed emittiert bei jedem Uebergang
    - error_occurred wird von Worker weitergeleitet
    - Status-Dot Text und Farbe aendern sich
    - Status-Label Text spiegelt Zustand wider
    - set_orchestrator() waehrend STOPPED erlaubt
    - set_orchestrator() waehrend RUNNING wirft RuntimeError
    - _on_mode_changed ruft orchestrator.set_mode() auf
    - _on_mode_changed mit EnvironmentError setzt Combo zurueck
    - cleanup() ruft orchestrator.stop() auf wenn nicht gestoppt
    - cleanup() wartet auf Thread-Ende

  MainWindow-Integration
    - bot_controls Property vorhanden
    - state_changed verbindet mit update_bot_indicator
    - TradingStatusBar.update_bot_indicator aendert Bot-Label
    - closeEvent ruft cleanup() auf
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest
from pytestqt.qtbot import QtBot

from gui.app import MainWindow, TradingStatusBar
from gui.design.theme import ThemeManager, ThemeMode
from gui.widgets.bot_controls_widget import (
    BotControlsWidget,
    BotState,
    BotWorker,
    _MODE_LABELS,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Hilfsfunktionen / Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_orchestrator(run_loop_fn=None, mode_name="SUGGEST_ONLY"):
    """Erstellt einen Mock-Orchestrator mit konfigurierbarem run_loop."""
    orc = MagicMock()
    orc._stop_event = threading.Event()
    orc._paused     = False
    orc.is_paused   = False

    from src.modes import TradingMode
    orc.mode = TradingMode.SUGGEST_ONLY

    if run_loop_fn is not None:
        orc.run_loop.side_effect = run_loop_fn
    else:
        orc.run_loop.return_value = None  # sofort beenden

    def _pause(reason=""):
        orc._paused  = True
        orc.is_paused = True
    def _resume():
        orc._paused  = False
        orc.is_paused = False
    def _stop():
        orc._stop_event.set()
    def _set_mode(m):
        orc.mode = m

    orc.pause.side_effect   = _pause
    orc.resume.side_effect  = _resume
    orc.stop.side_effect    = _stop
    orc.set_mode.side_effect = _set_mode

    return orc


@pytest.fixture
def orc():
    return _make_orchestrator()


@pytest.fixture
def widget(qtbot: QtBot) -> BotControlsWidget:
    w = BotControlsWidget()
    qtbot.addWidget(w)
    return w


@pytest.fixture
def widget_with_orc(qtbot: QtBot, orc) -> BotControlsWidget:
    w = BotControlsWidget(orchestrator=orc, symbols=["EURUSD"])
    qtbot.addWidget(w)
    return w


@pytest.fixture
def theme() -> ThemeManager:
    return ThemeManager(mode=ThemeMode.DARK)


@pytest.fixture
def main_window(qtbot: QtBot, theme: ThemeManager) -> MainWindow:
    w = MainWindow(theme_manager=theme)
    qtbot.addWidget(w)
    return w


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: BotState
# ─────────────────────────────────────────────────────────────────────────────

class TestBotState:
    def test_stopped_exists(self):
        assert BotState.STOPPED.value == "Gestoppt"

    def test_running_exists(self):
        assert BotState.RUNNING.value == "Aktiv"

    def test_paused_exists(self):
        assert BotState.PAUSED.value == "Pausiert"

    def test_stopping_exists(self):
        assert BotState.STOPPING.value == "Stoppt..."

    def test_four_states(self):
        assert len(BotState) == 4


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: BotWorker
# ─────────────────────────────────────────────────────────────────────────────

class TestBotWorker:

    def test_run_calls_run_loop(self, qtbot: QtBot):
        orc = _make_orchestrator()
        worker = BotWorker(orc, ["EURUSD"], 60)
        worker.run()
        orc.run_loop.assert_called_once_with(["EURUSD"], interval_seconds=60)

    def test_stopped_emitted_after_normal_return(self, qtbot: QtBot):
        orc = _make_orchestrator()
        worker = BotWorker(orc, ["EURUSD"])
        with qtbot.waitSignal(worker.stopped, timeout=2000):
            worker.run()

    def test_stopped_emitted_after_exception(self, qtbot: QtBot):
        orc = _make_orchestrator(run_loop_fn=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("test")))
        worker = BotWorker(orc, ["EURUSD"])
        signals: list[str] = []
        worker.error_occurred.connect(signals.append)
        with qtbot.waitSignal(worker.stopped, timeout=2000):
            worker.run()
        assert len(signals) == 1
        assert "test" in signals[0]

    def test_error_occurred_not_emitted_on_success(self, qtbot: QtBot):
        orc = _make_orchestrator()
        worker = BotWorker(orc, ["EURUSD"])
        errors: list[str] = []
        worker.error_occurred.connect(errors.append)
        worker.run()
        assert errors == []


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: BotControlsWidget – Erstellung und Initialzustand
# ─────────────────────────────────────────────────────────────────────────────

class TestBotControlsWidgetInit:

    def test_creates_without_crash(self, widget: BotControlsWidget):
        assert widget is not None

    def test_creates_with_orchestrator(self, widget_with_orc: BotControlsWidget):
        assert widget_with_orc is not None

    def test_initial_state_stopped(self, widget: BotControlsWidget):
        assert widget.bot_state == BotState.STOPPED

    def test_buttons_disabled_without_orchestrator(self, widget: BotControlsWidget):
        assert not widget._start_btn.isEnabled()
        assert not widget._stop_btn.isEnabled()
        assert not widget._pause_resume_btn.isEnabled()

    def test_start_enabled_with_orchestrator(self, widget_with_orc: BotControlsWidget):
        assert widget_with_orc._start_btn.isEnabled()

    def test_stop_disabled_in_stopped_state(self, widget_with_orc: BotControlsWidget):
        assert not widget_with_orc._stop_btn.isEnabled()

    def test_pause_disabled_in_stopped_state(self, widget_with_orc: BotControlsWidget):
        assert not widget_with_orc._pause_resume_btn.isEnabled()

    def test_mode_combo_disabled_without_orchestrator(self, widget: BotControlsWidget):
        assert not widget._mode_combo.isEnabled()

    def test_mode_combo_enabled_with_orchestrator(self, widget_with_orc: BotControlsWidget):
        assert widget_with_orc._mode_combo.isEnabled()

    def test_status_label_shows_stopped(self, widget: BotControlsWidget):
        assert widget._status_label.text() == BotState.STOPPED.value

    def test_pause_resume_button_default_icon(self, widget_with_orc: BotControlsWidget):
        assert "⏸" in widget_with_orc._pause_resume_btn.text()

    def test_mode_combo_has_three_items(self, widget: BotControlsWidget):
        assert widget._mode_combo.count() == 3

    def test_object_name_set(self, widget: BotControlsWidget):
        assert widget.objectName() == "bot_controls_widget"


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: Start
# ─────────────────────────────────────────────────────────────────────────────

class TestStart:

    def test_state_running_after_start(self, qtbot: QtBot, widget_with_orc: BotControlsWidget):
        widget_with_orc._on_start()
        assert widget_with_orc.bot_state == BotState.RUNNING
        qtbot.waitUntil(
            lambda: widget_with_orc.bot_state == BotState.STOPPED, timeout=5000
        )

    def test_start_btn_disabled_when_running(self, qtbot: QtBot, widget_with_orc: BotControlsWidget):
        widget_with_orc._set_state(BotState.RUNNING)
        assert not widget_with_orc._start_btn.isEnabled()

    def test_stop_btn_enabled_when_running(self, qtbot: QtBot, widget_with_orc: BotControlsWidget):
        widget_with_orc._set_state(BotState.RUNNING)
        assert widget_with_orc._stop_btn.isEnabled()

    def test_pause_btn_enabled_when_running(self, qtbot: QtBot, widget_with_orc: BotControlsWidget):
        widget_with_orc._set_state(BotState.RUNNING)
        assert widget_with_orc._pause_resume_btn.isEnabled()

    def test_state_changed_signal_on_start(self, qtbot: QtBot, widget_with_orc: BotControlsWidget):
        with qtbot.waitSignal(widget_with_orc.state_changed, timeout=1000) as blocker:
            widget_with_orc._on_start()
        assert blocker.args[0] == BotState.RUNNING
        qtbot.waitUntil(
            lambda: widget_with_orc.bot_state == BotState.STOPPED, timeout=5000
        )

    def test_state_returns_to_stopped_when_worker_finishes(
        self, qtbot: QtBot, widget_with_orc: BotControlsWidget
    ):
        widget_with_orc._on_start()
        qtbot.waitUntil(
            lambda: widget_with_orc.bot_state == BotState.STOPPED, timeout=5000
        )

    def test_start_ignored_without_orchestrator(self, qtbot: QtBot, widget: BotControlsWidget):
        widget._on_start()
        assert widget.bot_state == BotState.STOPPED

    def test_start_ignored_when_already_running(
        self, qtbot: QtBot, widget_with_orc: BotControlsWidget
    ):
        # Direkt RUNNING setzen – kein echter Thread, kein Race
        widget_with_orc._set_state(BotState.RUNNING)
        widget_with_orc._on_start()  # Soll ignoriert werden
        assert widget_with_orc.bot_state == BotState.RUNNING


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: Stop
# ─────────────────────────────────────────────────────────────────────────────

class TestStop:

    def test_stop_sets_stopping_state(self, qtbot: QtBot, widget_with_orc: BotControlsWidget):
        widget_with_orc._set_state(BotState.RUNNING)
        with qtbot.waitSignal(widget_with_orc.state_changed, timeout=1000) as blocker:
            widget_with_orc._on_stop()
        assert blocker.args[0] == BotState.STOPPING

    def test_stop_calls_orchestrator_stop(self, qtbot: QtBot, widget_with_orc: BotControlsWidget):
        widget_with_orc._set_state(BotState.RUNNING)
        widget_with_orc._on_stop()
        widget_with_orc._orchestrator.stop.assert_called()

    def test_all_buttons_disabled_in_stopping(
        self, qtbot: QtBot, widget_with_orc: BotControlsWidget
    ):
        widget_with_orc._set_state(BotState.RUNNING)
        widget_with_orc._on_stop()
        assert not widget_with_orc._start_btn.isEnabled()
        assert not widget_with_orc._stop_btn.isEnabled()
        assert not widget_with_orc._pause_resume_btn.isEnabled()

    def test_stop_ignored_when_already_stopped(
        self, qtbot: QtBot, widget_with_orc: BotControlsWidget
    ):
        widget_with_orc._on_stop()
        widget_with_orc._orchestrator.stop.assert_not_called()

    def test_stop_transitions_to_stopped_after_worker_done(
        self, qtbot: QtBot, widget_with_orc: BotControlsWidget
    ):
        # run_loop gibt sofort zurueck -> RUNNING -> STOPPING -> STOPPED
        widget_with_orc._on_start()
        qtbot.waitUntil(
            lambda: widget_with_orc.bot_state == BotState.STOPPED, timeout=5000
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: Pause / Resume
# ─────────────────────────────────────────────────────────────────────────────

class TestPauseResume:
    """
    Testet Pause/Resume-Logik mit _set_state() um Thread-Timing zu vermeiden.
    """

    def test_pause_sets_paused_state(self, qtbot: QtBot, widget_with_orc: BotControlsWidget):
        widget_with_orc._set_state(BotState.RUNNING)
        with qtbot.waitSignal(widget_with_orc.state_changed, timeout=1000) as blocker:
            widget_with_orc._on_pause_resume()
        assert blocker.args[0] == BotState.PAUSED

    def test_pause_calls_orchestrator_pause(self, qtbot: QtBot, widget_with_orc: BotControlsWidget):
        widget_with_orc._set_state(BotState.RUNNING)
        widget_with_orc._on_pause_resume()
        widget_with_orc._orchestrator.pause.assert_called_once_with("GUI")

    def test_resume_sets_running_state(self, qtbot: QtBot, widget_with_orc: BotControlsWidget):
        widget_with_orc._set_state(BotState.RUNNING)
        widget_with_orc._on_pause_resume()   # -> PAUSED
        assert widget_with_orc.bot_state == BotState.PAUSED
        with qtbot.waitSignal(widget_with_orc.state_changed, timeout=1000) as blocker:
            widget_with_orc._on_pause_resume()   # -> RUNNING
        assert blocker.args[0] == BotState.RUNNING

    def test_resume_calls_orchestrator_resume(self, qtbot: QtBot, widget_with_orc: BotControlsWidget):
        widget_with_orc._set_state(BotState.RUNNING)
        widget_with_orc._on_pause_resume()   # -> PAUSED
        widget_with_orc._on_pause_resume()   # -> RUNNING
        widget_with_orc._orchestrator.resume.assert_called_once()

    def test_pause_resume_btn_icon_changes(self, qtbot: QtBot, widget_with_orc: BotControlsWidget):
        widget_with_orc._set_state(BotState.RUNNING)
        assert "⏸" in widget_with_orc._pause_resume_btn.text()   # Pause-Icon
        widget_with_orc._on_pause_resume()
        assert "▶" in widget_with_orc._pause_resume_btn.text()   # Resume-Icon
        widget_with_orc._on_pause_resume()
        assert "⏸" in widget_with_orc._pause_resume_btn.text()   # wieder Pause-Icon

    def test_stop_btn_enabled_when_paused(self, qtbot: QtBot, widget_with_orc: BotControlsWidget):
        widget_with_orc._set_state(BotState.PAUSED)
        assert widget_with_orc._stop_btn.isEnabled()

    def test_pause_ignored_without_orchestrator(self, qtbot: QtBot):
        w = BotControlsWidget()
        qtbot.addWidget(w)
        w._on_pause_resume()
        assert w.bot_state == BotState.STOPPED

    def test_stop_resumes_before_stopping_if_paused(
        self, qtbot: QtBot, widget_with_orc: BotControlsWidget
    ):
        widget_with_orc._set_state(BotState.PAUSED)
        widget_with_orc._on_stop()
        widget_with_orc._orchestrator.resume.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: state_changed Signal
# ─────────────────────────────────────────────────────────────────────────────

class TestStateChangedSignal:

    def test_signal_emitted_on_start(self, qtbot: QtBot, widget_with_orc: BotControlsWidget):
        states: list[BotState] = []
        widget_with_orc.state_changed.connect(states.append)
        widget_with_orc._on_start()
        assert BotState.RUNNING in states
        qtbot.waitUntil(
            lambda: widget_with_orc.bot_state == BotState.STOPPED, timeout=5000
        )

    def test_signal_not_emitted_twice_for_same_state(
        self, qtbot: QtBot, widget_with_orc: BotControlsWidget
    ):
        states: list[BotState] = []
        widget_with_orc.state_changed.connect(states.append)
        widget_with_orc._set_state(BotState.RUNNING)
        count_before = states.count(BotState.RUNNING)
        widget_with_orc._set_state(BotState.RUNNING)   # Gleicher Zustand – kein Signal
        assert states.count(BotState.RUNNING) == count_before

    def test_error_occurred_propagated_from_worker(
        self, qtbot: QtBot
    ):
        orc = _make_orchestrator(
            run_loop_fn=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("netz"))
        )
        w = BotControlsWidget(orchestrator=orc, symbols=["EURUSD"])
        errors: list[str] = []
        w.error_occurred.connect(errors.append)
        w._on_start()
        qtbot.waitUntil(lambda: len(errors) > 0, timeout=3000)
        assert "netz" in errors[0]


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: Status-Indicator
# ─────────────────────────────────────────────────────────────────────────────

class TestStatusIndicator:

    def test_label_stopped(self, widget_with_orc: BotControlsWidget):
        assert widget_with_orc._status_label.text() == "Gestoppt"

    def test_label_running_after_start(self, qtbot: QtBot, widget_with_orc: BotControlsWidget):
        widget_with_orc._set_state(BotState.RUNNING)
        assert widget_with_orc._status_label.text() == "Aktiv"

    def test_label_paused(self, qtbot: QtBot, widget_with_orc: BotControlsWidget):
        widget_with_orc._set_state(BotState.PAUSED)
        assert widget_with_orc._status_label.text() == "Pausiert"

    def test_dot_color_green_when_running(self, qtbot: QtBot, widget_with_orc: BotControlsWidget):
        widget_with_orc._set_state(BotState.RUNNING)
        style = widget_with_orc._status_dot.styleSheet()
        assert "#22c55e" in style

    def test_dot_color_gray_when_stopped(self, widget_with_orc: BotControlsWidget):
        style = widget_with_orc._status_dot.styleSheet()
        assert "#6b7280" in style

    def test_dot_color_amber_when_paused(self, qtbot: QtBot, widget_with_orc: BotControlsWidget):
        widget_with_orc._set_state(BotState.PAUSED)
        assert "#f59e0b" in widget_with_orc._status_dot.styleSheet()


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: set_orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class TestSetOrchestrator:

    def test_set_orchestrator_enables_buttons(self, widget: BotControlsWidget):
        orc = _make_orchestrator()
        widget.set_orchestrator(orc, ["EURUSD"])
        assert widget._start_btn.isEnabled()

    def test_set_orchestrator_raises_when_running(
        self, qtbot: QtBot, widget_with_orc: BotControlsWidget
    ):
        orc2 = _make_orchestrator()
        widget_with_orc._set_state(BotState.RUNNING)
        with pytest.raises(RuntimeError, match="nicht gewechselt"):
            widget_with_orc.set_orchestrator(orc2, ["GBPUSD"])

    def test_set_orchestrator_updates_symbols(self, widget: BotControlsWidget):
        orc = _make_orchestrator()
        widget.set_orchestrator(orc, ["GBPUSD", "USDJPY"])
        assert widget._symbols == ["GBPUSD", "USDJPY"]

    def test_set_orchestrator_updates_interval(self, widget: BotControlsWidget):
        orc = _make_orchestrator()
        widget.set_orchestrator(orc, ["EURUSD"], interval_seconds=60)
        assert widget._interval == 60


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: Modus-Wechsel
# ─────────────────────────────────────────────────────────────────────────────

class TestModeChange:

    def test_mode_change_calls_set_mode(self, qtbot: QtBot, widget_with_orc: BotControlsWidget):
        from src.modes import TradingMode
        widget_with_orc._on_mode_changed(1)  # CONFIRM_REQUIRED
        widget_with_orc._orchestrator.set_mode.assert_called_once_with(
            TradingMode.CONFIRM_REQUIRED
        )

    def test_mode_change_autonomous_calls_set_mode(
        self, qtbot: QtBot, widget_with_orc: BotControlsWidget
    ):
        from src.modes import TradingMode
        widget_with_orc._on_mode_changed(2)  # AUTONOMOUS
        widget_with_orc._orchestrator.set_mode.assert_called_with(TradingMode.AUTONOMOUS)

    def test_mode_change_environment_error_resets_combo(
        self, qtbot: QtBot
    ):
        from src.modes import TradingMode
        orc = _make_orchestrator()
        orc.set_mode.side_effect = EnvironmentError("CONFIRM_AUTONOMOUS fehlt")

        w = BotControlsWidget(orchestrator=orc, symbols=["EURUSD"])

        with patch("gui.widgets.bot_controls_widget.QMessageBox.warning"):
            w._on_mode_changed(2)  # Versucht AUTONOMOUS

        # Combo sollte wieder auf Index 0 (SUGGEST_ONLY) stehen
        assert w._mode_combo.currentIndex() == 0

    def test_mode_change_ignored_without_orchestrator(self, widget: BotControlsWidget):
        widget._on_mode_changed(1)  # kein Fehler, kein Orchestrator


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: cleanup
# ─────────────────────────────────────────────────────────────────────────────

class TestCleanup:
    """
    Nutzt _set_state() wo moeglich, um Thread-Timing-Probleme zu vermeiden.
    """

    def test_cleanup_when_stopped_no_crash(self, widget_with_orc: BotControlsWidget):
        widget_with_orc.cleanup()

    def test_cleanup_when_running_calls_stop(self, qtbot: QtBot, widget_with_orc: BotControlsWidget):
        widget_with_orc._set_state(BotState.RUNNING)
        widget_with_orc.cleanup()
        widget_with_orc._orchestrator.stop.assert_called()

    def test_cleanup_when_paused_calls_resume_then_stop(
        self, qtbot: QtBot, widget_with_orc: BotControlsWidget
    ):
        widget_with_orc._set_state(BotState.PAUSED)
        widget_with_orc.cleanup()
        widget_with_orc._orchestrator.resume.assert_called()
        widget_with_orc._orchestrator.stop.assert_called()

    def test_cleanup_with_real_thread_waits_for_end(
        self, qtbot: QtBot, widget_with_orc: BotControlsWidget
    ):
        # run_loop gibt sofort zurueck; cleanup() muss sauber beenden
        widget_with_orc._on_start()
        qtbot.waitUntil(
            lambda: widget_with_orc.bot_state == BotState.STOPPED, timeout=5000
        )
        # Nach Abschluss kein Absturz
        widget_with_orc.cleanup()


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: MainWindow-Integration
# ─────────────────────────────────────────────────────────────────────────────

class TestMainWindowIntegration:

    def test_bot_controls_property_exists(self, main_window: MainWindow):
        assert isinstance(main_window.bot_controls, BotControlsWidget)

    def test_bot_controls_in_sidebar(self, main_window: MainWindow):
        sidebar_layout = main_window.sidebar.layout()
        found = False
        for i in range(sidebar_layout.count()):
            item = sidebar_layout.itemAt(i)
            if item and item.widget() is main_window.bot_controls:
                found = True
                break
        assert found, "BotControlsWidget nicht in der Sidebar-Layout gefunden"

    def test_state_changed_updates_status_bar(self, main_window: MainWindow):
        main_window._on_bot_state_changed(BotState.RUNNING)
        text = main_window.trading_status_bar.bot_label.text()
        assert "aktiv" in text.lower() or "▶" in text

    def test_state_stopped_updates_status_bar(self, main_window: MainWindow):
        main_window._on_bot_state_changed(BotState.STOPPED)
        text = main_window.trading_status_bar.bot_label.text()
        assert "gestoppt" in text.lower() or "⏹" in text

    def test_state_paused_updates_status_bar(self, main_window: MainWindow):
        main_window._on_bot_state_changed(BotState.PAUSED)
        text = main_window.trading_status_bar.bot_label.text()
        assert "pausiert" in text.lower() or "⏸" in text

    def test_close_event_calls_cleanup(self, qtbot: QtBot, theme: ThemeManager):
        w = MainWindow(theme_manager=theme)
        qtbot.addWidget(w)

        cleanup_called: list[bool] = []
        w.bot_controls.cleanup = lambda: cleanup_called.append(True)

        from PySide6.QtGui import QCloseEvent
        event = QCloseEvent()
        w.closeEvent(event)

        assert len(cleanup_called) == 1


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: TradingStatusBar.update_bot_indicator
# ─────────────────────────────────────────────────────────────────────────────

class TestUpdateBotIndicator:

    @pytest.fixture
    def status_bar(self, qtbot: QtBot) -> TradingStatusBar:
        w = TradingStatusBar()
        qtbot.addWidget(w)
        return w

    def test_update_sets_text(self, status_bar: TradingStatusBar):
        status_bar.update_bot_indicator("▶  Bot aktiv", "#22c55e")
        assert status_bar.bot_label.text() == "▶  Bot aktiv"

    def test_update_sets_color(self, status_bar: TradingStatusBar):
        status_bar.update_bot_indicator("⏸  Bot pausiert", "#f59e0b")
        assert "#f59e0b" in status_bar.bot_label.styleSheet()

    def test_update_bold_for_active(self, status_bar: TradingStatusBar):
        status_bar.update_bot_indicator("▶  Bot aktiv", "#22c55e")
        style = status_bar.bot_label.styleSheet()
        assert "bold" in style

    def test_update_not_bold_for_stopped(self, status_bar: TradingStatusBar):
        status_bar.update_bot_indicator("⏹  Bot gestoppt", "#6b7280")
        style = status_bar.bot_label.styleSheet()
        assert "bold" not in style or "font-weight: bold" not in style

    def test_set_paused_still_works(self, status_bar: TradingStatusBar):
        status_bar.set_paused(True)
        assert "pausiert" in status_bar.bot_label.text().lower()

    def test_set_paused_false_shows_active(self, status_bar: TradingStatusBar):
        status_bar.set_paused(False)
        assert "aktiv" in status_bar.bot_label.text().lower()
