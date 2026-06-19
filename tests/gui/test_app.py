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

import os
from unittest.mock import MagicMock

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
    _MT5AccountBackend,
    _load_env_file,
    _try_connect_mt5,
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

    def test_account_info_default_none(self, status_bar: TradingStatusBar):
        assert status_bar.account_info is None

    def test_account_label_hidden_by_default(self, status_bar: TradingStatusBar):
        assert status_bar.account_label.isHidden()

    def test_set_account_info_shows_label(self, status_bar: TradingStatusBar):
        status_bar.set_account_info({"login": 383619, "balance": 10_000.0, "currency": "EUR", "is_demo": True})
        assert not status_bar.account_label.isHidden()

    def test_set_account_info_none_hides_label(self, status_bar: TradingStatusBar):
        status_bar.set_account_info({"login": 1, "balance": 100.0, "currency": "EUR", "is_demo": True})
        status_bar.set_account_info(None)
        assert status_bar.account_label.isHidden()

    def test_set_account_info_shows_login(self, status_bar: TradingStatusBar):
        status_bar.set_account_info({"login": 99999, "balance": 100.0, "currency": "EUR", "is_demo": False})
        assert "99999" in status_bar.account_label.text()

    def test_set_account_info_shows_balance(self, status_bar: TradingStatusBar):
        status_bar.set_account_info({"login": 1, "balance": 12_345.67, "currency": "EUR", "is_demo": True})
        assert "12" in status_bar.account_label.text()

    def test_set_account_info_demo_tag(self, status_bar: TradingStatusBar):
        status_bar.set_account_info({"login": 1, "balance": 0.0, "currency": "EUR", "is_demo": True})
        assert "Demo" in status_bar.account_label.text()

    def test_set_account_info_live_tag(self, status_bar: TradingStatusBar):
        status_bar.set_account_info({"login": 1, "balance": 0.0, "currency": "EUR", "is_demo": False})
        assert "Live" in status_bar.account_label.text()

    def test_set_account_info_stores_info(self, status_bar: TradingStatusBar):
        info = {"login": 42, "balance": 500.0, "currency": "USD", "is_demo": False}
        status_bar.set_account_info(info)
        assert status_bar.account_info is info


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


# ─────────────────────────────────────────────────────────────────────────────
#  _load_env_file
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadEnvFile:

    def test_loads_key_value(self, tmp_path, monkeypatch):
        env_file = tmp_path / ".env"
        env_file.write_text("TEST_LOAD_KEY=hello\n", encoding="utf-8")
        monkeypatch.delenv("TEST_LOAD_KEY", raising=False)
        _load_env_file(str(env_file))
        assert os.environ.get("TEST_LOAD_KEY") == "hello"

    def test_skips_comment_lines(self, tmp_path, monkeypatch):
        env_file = tmp_path / ".env"
        env_file.write_text("# This is a comment\nKEY_NO_COMMENT=val\n", encoding="utf-8")
        monkeypatch.delenv("KEY_NO_COMMENT", raising=False)
        _load_env_file(str(env_file))
        assert os.environ.get("KEY_NO_COMMENT") == "val"

    def test_skips_empty_lines(self, tmp_path, monkeypatch):
        env_file = tmp_path / ".env"
        env_file.write_text("\n\nKEY_EMPTY=42\n\n", encoding="utf-8")
        monkeypatch.delenv("KEY_EMPTY", raising=False)
        _load_env_file(str(env_file))
        assert os.environ.get("KEY_EMPTY") == "42"

    def test_missing_file_no_crash(self):
        _load_env_file("/nonexistent/path/.env")

    def test_does_not_overwrite_existing(self, tmp_path, monkeypatch):
        env_file = tmp_path / ".env"
        env_file.write_text("EXISTING_KEY=fromfile\n", encoding="utf-8")
        monkeypatch.setenv("EXISTING_KEY", "original")
        _load_env_file(str(env_file))
        assert os.environ.get("EXISTING_KEY") == "original"

    def test_strips_double_quotes(self, tmp_path, monkeypatch):
        env_file = tmp_path / ".env"
        env_file.write_text('QUOTED_KEY="my value"\n', encoding="utf-8")
        monkeypatch.delenv("QUOTED_KEY", raising=False)
        _load_env_file(str(env_file))
        assert os.environ.get("QUOTED_KEY") == "my value"

    def test_strips_single_quotes(self, tmp_path, monkeypatch):
        env_file = tmp_path / ".env"
        env_file.write_text("SINGLE_KEY='my value'\n", encoding="utf-8")
        monkeypatch.delenv("SINGLE_KEY", raising=False)
        _load_env_file(str(env_file))
        assert os.environ.get("SINGLE_KEY") == "my value"

    def test_value_with_equals_sign(self, tmp_path, monkeypatch):
        env_file = tmp_path / ".env"
        env_file.write_text("EQ_KEY=a=b=c\n", encoding="utf-8")
        monkeypatch.delenv("EQ_KEY", raising=False)
        _load_env_file(str(env_file))
        assert os.environ.get("EQ_KEY") == "a=b=c"

    def test_multiple_keys(self, tmp_path, monkeypatch):
        env_file = tmp_path / ".env"
        env_file.write_text("MULTI_A=1\nMULTI_B=2\n", encoding="utf-8")
        monkeypatch.delenv("MULTI_A", raising=False)
        monkeypatch.delenv("MULTI_B", raising=False)
        _load_env_file(str(env_file))
        assert os.environ.get("MULTI_A") == "1"
        assert os.environ.get("MULTI_B") == "2"


# ─────────────────────────────────────────────────────────────────────────────
#  _try_connect_mt5
# ─────────────────────────────────────────────────────────────────────────────

class TestTryConnectMT5:

    @pytest.fixture(autouse=True)
    def no_env_file(self, monkeypatch):
        """Prevent the real .env from being loaded during any test in this class."""
        monkeypatch.setattr("gui.app._load_env_file", lambda *a, **kw: None)

    def _bar(self, qtbot):
        bar = TradingStatusBar()
        qtbot.addWidget(bar)
        return bar

    def test_no_credentials_returns_none(self, qtbot, monkeypatch):
        monkeypatch.delenv("MT5_LOGIN",    raising=False)
        monkeypatch.delenv("MT5_PASSWORD", raising=False)
        monkeypatch.delenv("MT5_SERVER",   raising=False)
        result = _try_connect_mt5(self._bar(qtbot))
        assert result is None

    def test_no_credentials_status_unchanged(self, qtbot, monkeypatch):
        monkeypatch.delenv("MT5_LOGIN",    raising=False)
        monkeypatch.delenv("MT5_PASSWORD", raising=False)
        monkeypatch.delenv("MT5_SERVER",   raising=False)
        bar = self._bar(qtbot)
        _try_connect_mt5(bar)
        assert bar.connection_status is ConnectionStatus.DISCONNECTED

    def test_missing_password_skips_connect(self, qtbot, monkeypatch):
        monkeypatch.setenv("MT5_LOGIN",  "12345")
        monkeypatch.delenv("MT5_PASSWORD", raising=False)
        monkeypatch.setenv("MT5_SERVER", "srv")
        calls = []
        _try_connect_mt5(self._bar(qtbot), _connector_factory=lambda *a: calls.append(a))
        assert calls == []

    def test_missing_server_skips_connect(self, qtbot, monkeypatch):
        monkeypatch.setenv("MT5_LOGIN",    "12345")
        monkeypatch.setenv("MT5_PASSWORD", "pw")
        monkeypatch.delenv("MT5_SERVER", raising=False)
        result = _try_connect_mt5(self._bar(qtbot))
        assert result is None

    def test_invalid_login_returns_none(self, qtbot, monkeypatch):
        monkeypatch.setenv("MT5_LOGIN",    "notanumber")
        monkeypatch.setenv("MT5_PASSWORD", "pw")
        monkeypatch.setenv("MT5_SERVER",   "srv")
        result = _try_connect_mt5(self._bar(qtbot))
        assert result is None

    def test_invalid_login_status_unchanged(self, qtbot, monkeypatch):
        monkeypatch.setenv("MT5_LOGIN",    "abc")
        monkeypatch.setenv("MT5_PASSWORD", "pw")
        monkeypatch.setenv("MT5_SERVER",   "srv")
        bar = self._bar(qtbot)
        _try_connect_mt5(bar)
        assert bar.connection_status is ConnectionStatus.DISCONNECTED

    def test_success_returns_connector(self, qtbot, monkeypatch):
        monkeypatch.setenv("MT5_LOGIN",    "12345")
        monkeypatch.setenv("MT5_PASSWORD", "secret")
        monkeypatch.setenv("MT5_SERVER",   "DemoServer")
        mock = MagicMock()
        result = _try_connect_mt5(self._bar(qtbot), _connector_factory=lambda *a: mock)
        assert result is mock

    def test_success_sets_connected(self, qtbot, monkeypatch):
        monkeypatch.setenv("MT5_LOGIN",    "12345")
        monkeypatch.setenv("MT5_PASSWORD", "secret")
        monkeypatch.setenv("MT5_SERVER",   "DemoServer")
        mock = MagicMock()
        bar = self._bar(qtbot)
        _try_connect_mt5(bar, _connector_factory=lambda *a: mock)
        assert bar.connection_status is ConnectionStatus.CONNECTED

    def test_connect_raises_sets_error(self, qtbot, monkeypatch):
        monkeypatch.setenv("MT5_LOGIN",    "12345")
        monkeypatch.setenv("MT5_PASSWORD", "secret")
        monkeypatch.setenv("MT5_SERVER",   "DemoServer")

        def factory(*args):
            raise RuntimeError("MT5 nicht erreichbar")

        bar = self._bar(qtbot)
        _try_connect_mt5(bar, _connector_factory=factory)
        assert bar.connection_status is ConnectionStatus.ERROR

    def test_connect_raises_returns_none(self, qtbot, monkeypatch):
        monkeypatch.setenv("MT5_LOGIN",    "12345")
        monkeypatch.setenv("MT5_PASSWORD", "secret")
        monkeypatch.setenv("MT5_SERVER",   "DemoServer")

        def factory(*args):
            raise RuntimeError("crash")

        result = _try_connect_mt5(self._bar(qtbot), _connector_factory=factory)
        assert result is None

    def test_exception_never_propagates(self, qtbot, monkeypatch):
        monkeypatch.setenv("MT5_LOGIN",    "12345")
        monkeypatch.setenv("MT5_PASSWORD", "pw")
        monkeypatch.setenv("MT5_SERVER",   "srv")

        def factory(*args):
            raise SystemError("hard crash")

        _try_connect_mt5(self._bar(qtbot), _connector_factory=factory)  # must not raise

    def test_factory_receives_correct_login(self, qtbot, monkeypatch):
        monkeypatch.setenv("MT5_LOGIN",    "99999")
        monkeypatch.setenv("MT5_PASSWORD", "mypass")
        monkeypatch.setenv("MT5_SERVER",   "ICMarkets-Demo")
        monkeypatch.delenv("MT5_PATH", raising=False)
        received: dict = {}

        def factory(login, password, server, path):
            received.update(login=login, password=password,
                            server=server, path=path)
            return MagicMock()

        _try_connect_mt5(self._bar(qtbot), _connector_factory=factory)
        assert received["login"]    == 99999
        assert received["password"] == "mypass"
        assert received["server"]   == "ICMarkets-Demo"
        assert received["path"]     is None

    def test_factory_receives_mt5_path(self, qtbot, monkeypatch):
        monkeypatch.setenv("MT5_LOGIN",    "1")
        monkeypatch.setenv("MT5_PASSWORD", "p")
        monkeypatch.setenv("MT5_SERVER",   "s")
        monkeypatch.setenv("MT5_PATH",     "C:/MT5/terminal64.exe")
        received: dict = {}

        def factory(login, password, server, path):
            received["path"] = path
            return MagicMock()

        _try_connect_mt5(self._bar(qtbot), _connector_factory=factory)
        assert received["path"] == "C:/MT5/terminal64.exe"

    def test_connect_method_called(self, qtbot, monkeypatch):
        monkeypatch.setenv("MT5_LOGIN",    "1")
        monkeypatch.setenv("MT5_PASSWORD", "p")
        monkeypatch.setenv("MT5_SERVER",   "s")
        mock = MagicMock()
        _try_connect_mt5(self._bar(qtbot), _connector_factory=lambda *a: mock)
        mock.connect.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
#  _MT5AccountBackend
# ─────────────────────────────────────────────────────────────────────────────

def _make_account_dict(**kw) -> dict:
    return {
        "login":    kw.get("login",    383619),
        "name":     kw.get("name",     "DEMO_007"),
        "server":   kw.get("server",   "FusionMarkets-Demo"),
        "balance":  kw.get("balance",  10_000.0),
        "equity":   kw.get("equity",   10_000.0),
        "currency": kw.get("currency", "EUR"),
        "leverage": kw.get("leverage", 100),
        "is_demo":  kw.get("is_demo",  True),
    }


class TestMT5AccountBackend:

    def _backend(self, account_dict=None):
        connector = MagicMock()
        if account_dict is None:
            account_dict = _make_account_dict()
        connector.get_account_info.return_value = account_dict
        return _MT5AccountBackend(connector)

    def test_fetch_snapshot_returns_snapshot(self):
        from gui.views.dashboard_view import DashboardSnapshot
        snap = self._backend().fetch_snapshot()
        assert isinstance(snap, DashboardSnapshot)

    def test_fetch_snapshot_balance(self):
        snap = self._backend(_make_account_dict(balance=12_500.0)).fetch_snapshot()
        assert snap.balance == 12_500.0

    def test_fetch_snapshot_equity(self):
        snap = self._backend(_make_account_dict(equity=11_000.0)).fetch_snapshot()
        assert snap.equity == 11_000.0

    def test_fetch_snapshot_currency(self):
        snap = self._backend(_make_account_dict(currency="USD")).fetch_snapshot()
        assert snap.currency == "USD"

    def test_fetch_snapshot_account_number(self):
        snap = self._backend(_make_account_dict(login=99999)).fetch_snapshot()
        assert snap.account_number == 99999

    def test_fetch_snapshot_server(self):
        snap = self._backend(_make_account_dict(server="ICMarkets-Live")).fetch_snapshot()
        assert snap.server == "ICMarkets-Live"

    def test_fetch_snapshot_leverage(self):
        snap = self._backend(_make_account_dict(leverage=500)).fetch_snapshot()
        assert snap.leverage == 500

    def test_fetch_snapshot_is_demo_true(self):
        snap = self._backend(_make_account_dict(is_demo=True)).fetch_snapshot()
        assert snap.is_demo is True

    def test_fetch_snapshot_is_demo_false(self):
        snap = self._backend(_make_account_dict(is_demo=False)).fetch_snapshot()
        assert snap.is_demo is False

    def test_fetch_snapshot_on_error_returns_empty(self):
        from gui.views.dashboard_view import DashboardSnapshot
        connector = MagicMock()
        connector.get_account_info.side_effect = RuntimeError("MT5 crashed")
        backend = _MT5AccountBackend(connector)
        snap = backend.fetch_snapshot()
        assert isinstance(snap, DashboardSnapshot)
        assert snap.balance is None

    def test_fetch_snapshot_calls_get_account_info(self):
        b = self._backend()
        b.fetch_snapshot()
        b._connector.get_account_info.assert_called_once()
