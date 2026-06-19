"""
tests/gui/test_app.py
GUI-Tests fuer gui/app.py via pytest-qt.

Gepruefte Bereiche:
  ThemeManager
    - Default Dark Mode
    - set_mode / toggle
    - Stylesheet nicht leer
    - Callback wird aufgerufen

  TradingStatusBar
    - Defaults: Disconnected, Suggest, Paused
    - set_connection / set_trading_mode / set_paused aktualisieren Zustand
    - Label-Inhalte spiegeln Zustand wider

  NavigationSidebar
    - Alle 6 Sektionen vorhanden
    - Dashboard ist Standard
    - navigate_to aendert current_section
    - section_changed Signal wird emittiert

  MainWindow
    - Erstellt ohne Absturz
    - 6 Views im StackedWidget
    - Navigate-To wechselt aktuellen View
    - Mindestgroesse >= 1366x768
    - Statusleiste zugaenglich
    - Dashboard ist Standard-View

  ConfirmationDialog
    - Erstellt ohne Absturz
    - Fenster hat Titel und Bestaetigung-Button
"""

from __future__ import annotations

import pytest
from pytestqt.qtbot import QtBot

from gui.app import (
    ConfirmationDialog,
    ConnectionStatus,
    MainWindow,
    NavigationSidebar,
    Section,
    TradingMode,
    TradingStatusBar,
)
from gui.design.theme import ThemeManager, ThemeMode


# ─────────────────────────────────────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def theme() -> ThemeManager:
    return ThemeManager(mode=ThemeMode.DARK)


@pytest.fixture
def main_window(qtbot: QtBot, theme: ThemeManager) -> MainWindow:
    w = MainWindow(theme_manager=theme)
    qtbot.addWidget(w)
    return w


@pytest.fixture
def sidebar(qtbot: QtBot) -> NavigationSidebar:
    w = NavigationSidebar()
    qtbot.addWidget(w)
    return w


@pytest.fixture
def status_bar(qtbot: QtBot) -> TradingStatusBar:
    w = TradingStatusBar()
    qtbot.addWidget(w)
    return w


# ─────────────────────────────────────────────────────────────────────────────
#  ThemeManager
# ─────────────────────────────────────────────────────────────────────────────

class TestThemeManager:
    def test_default_is_dark(self, theme: ThemeManager):
        assert theme.mode is ThemeMode.DARK

    def test_set_light_mode(self, theme: ThemeManager):
        theme.set_mode(ThemeMode.LIGHT)
        assert theme.mode is ThemeMode.LIGHT

    def test_set_dark_mode_again(self, theme: ThemeManager):
        theme.set_mode(ThemeMode.LIGHT)
        theme.set_mode(ThemeMode.DARK)
        assert theme.mode is ThemeMode.DARK

    def test_toggle_switches_to_light(self, theme: ThemeManager):
        theme.toggle()
        assert theme.mode is ThemeMode.LIGHT

    def test_toggle_twice_returns_to_dark(self, theme: ThemeManager):
        theme.toggle()
        theme.toggle()
        assert theme.mode is ThemeMode.DARK

    def test_stylesheet_is_nonempty(self, theme: ThemeManager):
        qss = theme.stylesheet()
        assert isinstance(qss, str)
        assert len(qss) > 100

    def test_stylesheet_contains_background(self, theme: ThemeManager):
        qss = theme.stylesheet()
        assert "background-color" in qss

    def test_callback_called_on_mode_change(self, theme: ThemeManager):
        received: list[str] = []
        theme.on_theme_changed(received.append)
        theme.set_mode(ThemeMode.LIGHT)
        assert len(received) == 1

    def test_callback_not_called_when_mode_unchanged(self, theme: ThemeManager):
        received: list[str] = []
        theme.on_theme_changed(received.append)
        theme.set_mode(ThemeMode.DARK)  # bereits DARK
        assert len(received) == 0

    def test_multiple_callbacks(self, theme: ThemeManager):
        calls_a: list[str] = []
        calls_b: list[str] = []
        theme.on_theme_changed(calls_a.append)
        theme.on_theme_changed(calls_b.append)
        theme.toggle()
        assert len(calls_a) == 1
        assert len(calls_b) == 1

    def test_colors_change_with_mode(self, theme: ThemeManager):
        dark_bg = theme.colors.bg_base
        theme.set_mode(ThemeMode.LIGHT)
        light_bg = theme.colors.bg_base
        assert dark_bg != light_bg


# ─────────────────────────────────────────────────────────────────────────────
#  TradingStatusBar
# ─────────────────────────────────────────────────────────────────────────────

class TestTradingStatusBar:
    def test_default_connection_disconnected(self, status_bar: TradingStatusBar):
        assert status_bar.connection_status is ConnectionStatus.DISCONNECTED

    def test_default_mode_suggest(self, status_bar: TradingStatusBar):
        assert status_bar.trading_mode is TradingMode.SUGGEST

    def test_default_paused_true(self, status_bar: TradingStatusBar):
        assert status_bar.is_paused is True

    def test_set_connection_connected(self, status_bar: TradingStatusBar):
        status_bar.set_connection(ConnectionStatus.CONNECTED)
        assert status_bar.connection_status is ConnectionStatus.CONNECTED

    def test_set_connection_error(self, status_bar: TradingStatusBar):
        status_bar.set_connection(ConnectionStatus.ERROR)
        assert status_bar.connection_status is ConnectionStatus.ERROR

    def test_set_mode_autonomous(self, status_bar: TradingStatusBar):
        status_bar.set_trading_mode(TradingMode.AUTONOMOUS)
        assert status_bar.trading_mode is TradingMode.AUTONOMOUS

    def test_set_mode_confirm(self, status_bar: TradingStatusBar):
        status_bar.set_trading_mode(TradingMode.CONFIRM)
        assert status_bar.trading_mode is TradingMode.CONFIRM

    def test_set_paused_false(self, status_bar: TradingStatusBar):
        status_bar.set_paused(False)
        assert status_bar.is_paused is False

    def test_connection_label_shows_status(self, status_bar: TradingStatusBar):
        status_bar.set_connection(ConnectionStatus.CONNECTED)
        assert ConnectionStatus.CONNECTED.value in status_bar.connection_label.text()

    def test_mode_label_shows_mode(self, status_bar: TradingStatusBar):
        status_bar.set_trading_mode(TradingMode.AUTONOMOUS)
        assert TradingMode.AUTONOMOUS.value in status_bar.mode_label.text()

    def test_bot_label_shows_paused(self, status_bar: TradingStatusBar):
        status_bar.set_paused(True)
        assert "pausiert" in status_bar.bot_label.text().lower()

    def test_bot_label_shows_active(self, status_bar: TradingStatusBar):
        status_bar.set_paused(False)
        assert "aktiv" in status_bar.bot_label.text().lower()

    def test_disconnected_label_text(self, status_bar: TradingStatusBar):
        assert ConnectionStatus.DISCONNECTED.value in status_bar.connection_label.text()


# ─────────────────────────────────────────────────────────────────────────────
#  NavigationSidebar
# ─────────────────────────────────────────────────────────────────────────────

class TestNavigationSidebar:
    def test_has_buttons_for_all_sections(self, sidebar: NavigationSidebar):
        for section in Section:
            btn = sidebar.button(section)
            assert btn is not None

    def test_default_section_is_dashboard(self, sidebar: NavigationSidebar):
        assert sidebar.current_section is Section.DASHBOARD

    def test_dashboard_button_checked_by_default(self, sidebar: NavigationSidebar):
        assert sidebar.button(Section.DASHBOARD).isChecked()

    def test_navigate_to_cockpit(self, sidebar: NavigationSidebar):
        sidebar.navigate_to(Section.COCKPIT)
        assert sidebar.current_section is Section.COCKPIT

    def test_navigate_to_settings(self, sidebar: NavigationSidebar):
        sidebar.navigate_to(Section.SETTINGS)
        assert sidebar.current_section is Section.SETTINGS

    def test_previous_button_unchecked_after_navigation(self, sidebar: NavigationSidebar):
        sidebar.navigate_to(Section.RISK)
        assert not sidebar.button(Section.DASHBOARD).isChecked()
        assert sidebar.button(Section.RISK).isChecked()

    def test_section_changed_signal_emitted(self, qtbot: QtBot, sidebar: NavigationSidebar):
        with qtbot.waitSignal(sidebar.section_changed, timeout=1000) as blocker:
            sidebar.navigate_to(Section.JOURNAL)
        assert blocker.args[0] is Section.JOURNAL

    def test_navigate_to_same_section_no_signal(self, qtbot: QtBot, sidebar: NavigationSidebar):
        """Nochmals zur aktuellen Sektion navigieren darf kein Signal emittieren."""
        signals_received = []
        sidebar.section_changed.connect(lambda s: signals_received.append(s))
        sidebar.navigate_to(Section.DASHBOARD)  # bereits Dashboard
        assert len(signals_received) == 0

    def test_six_sections_exist(self):
        assert len(list(Section)) == 6

    def test_all_section_labels_nonempty(self):
        for section in Section:
            assert len(section.label) > 0


# ─────────────────────────────────────────────────────────────────────────────
#  MainWindow
# ─────────────────────────────────────────────────────────────────────────────

class TestMainWindow:
    def test_creates_without_crash(self, main_window: MainWindow):
        assert main_window is not None

    def test_window_title(self, main_window: MainWindow):
        assert "QuantzAI" in main_window.windowTitle()

    def test_minimum_width_at_least_1366(self, main_window: MainWindow):
        assert main_window.minimumWidth() >= 1366

    def test_minimum_height_at_least_768(self, main_window: MainWindow):
        assert main_window.minimumHeight() >= 768

    def test_has_six_views_in_stack(self, main_window: MainWindow):
        assert main_window.content.count() == 6

    def test_default_view_is_dashboard(self, main_window: MainWindow):
        assert main_window.sidebar.current_section is Section.DASHBOARD

    def test_navigate_to_changes_view(self, main_window: MainWindow):
        initial = main_window.current_view()
        main_window.navigate_to(Section.BACKTEST)
        assert main_window.current_view() is not initial

    def test_navigate_through_all_sections(self, main_window: MainWindow):
        """Alle Sektionen sind erreichbar ohne Absturz."""
        for section in Section:
            main_window.navigate_to(section)
            assert main_window.sidebar.current_section is section

    def test_has_sidebar(self, main_window: MainWindow):
        assert main_window.sidebar is not None

    def test_has_trading_status_bar(self, main_window: MainWindow):
        assert main_window.trading_status_bar is not None

    def test_status_bar_default_paused(self, main_window: MainWindow):
        assert main_window.trading_status_bar.is_paused is True

    def test_status_bar_default_disconnected(self, main_window: MainWindow):
        assert (main_window.trading_status_bar.connection_status
                is ConnectionStatus.DISCONNECTED)

    def test_status_bar_update_propagates(self, main_window: MainWindow):
        main_window.trading_status_bar.set_connection(ConnectionStatus.CONNECTED)
        assert (main_window.trading_status_bar.connection_status
                is ConnectionStatus.CONNECTED)

    def test_theme_toggle_does_not_crash(self, main_window: MainWindow, theme: ThemeManager):
        """Theme-Wechsel darf nicht zu einem Absturz fuehren."""
        theme.toggle()
        assert main_window.styleSheet() != ""


# ─────────────────────────────────────────────────────────────────────────────
#  ConfirmationDialog
# ─────────────────────────────────────────────────────────────────────────────

class TestConfirmationDialog:
    def test_creates_without_crash(self, qtbot: QtBot):
        dlg = ConfirmationDialog(
            title="Test",
            message="Bist du sicher?",
        )
        qtbot.addWidget(dlg)
        assert dlg is not None

    def test_window_title_set(self, qtbot: QtBot):
        dlg = ConfirmationDialog(title="Live-Modus aktivieren",
                                 message="Irreversibel!")
        qtbot.addWidget(dlg)
        assert dlg.windowTitle() == "Live-Modus aktivieren"

    def test_minimum_width(self, qtbot: QtBot):
        dlg = ConfirmationDialog(title="T", message="M")
        qtbot.addWidget(dlg)
        assert dlg.minimumWidth() >= 400

    def test_confirm_button_exists(self, qtbot: QtBot):
        dlg = ConfirmationDialog(title="T", message="M",
                                 confirm_label="Jetzt aktivieren")
        qtbot.addWidget(dlg)
        confirm_btn = dlg.findChild(type(dlg), "dialog_confirm_btn")
        # Alternativ: Dialog hat mindestens einen Button
        from PySide6.QtWidgets import QPushButton
        buttons = dlg.findChildren(QPushButton)
        labels = [b.text() for b in buttons]
        assert "Jetzt aktivieren" in labels

    def test_cancel_button_exists(self, qtbot: QtBot):
        dlg = ConfirmationDialog(title="T", message="M")
        qtbot.addWidget(dlg)
        from PySide6.QtWidgets import QPushButton
        buttons = dlg.findChildren(QPushButton)
        labels = [b.text() for b in buttons]
        assert "Abbrechen" in labels
