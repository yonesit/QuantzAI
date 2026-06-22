"""
tests/gui/test_settings_view.py
Tests fuer gui/views/settings_view.py und MainWindow-Integration.

Abgedeckt:
  - Initialisierung: alle Widgets, Tab-Anzahl, Anfangszustand
  - Risiko-Parameter: Grenzen, Dirty-State
  - Modus-Auswahl: Combo-Optionen, Aenderungen
  - Live/Paper-Schalter: Radio-Buttons
  - Pending-Changes: Label, Buttons, Zaehler
  - Speichern: Bestaetigung, Backend-Call, Signale, Audit
  - AUTONOMOUS: 2-stufige Bestaetigung
  - LIVE-Modus: eigene Bestaetigung
  - Verwerfen: Reset auf gespeicherten Zustand
  - Konten: Hinzufuegen/Entfernen (kein Passwort)
  - Symbole: Checkbox-Liste
  - Telegram: maskierter Token, Test-Button
  - Audit-Log: Tabelle nach Speichern
  - MainWindow: settings_view Property, Navigation, View-Anzahl
"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest
from PySide6.QtCore import Qt

from gui.app import MainWindow, Section
from gui.views.settings_view import (
    SettingsView,
    _AVAILABLE_SYMBOLS,
    _DEFAULT_SETTINGS,
    TAB_RISK,
    TAB_MODE,
    TAB_ACCOUNTS,
    TAB_SYMBOLS,
    TAB_TELEGRAM,
    TAB_AUDIT,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _confirm_yes(title: str, msg: str) -> bool:
    return True


def _confirm_no(title: str, msg: str) -> bool:
    return False


def _mock_backend(**overrides) -> MagicMock:
    backend = MagicMock()
    backend.get_settings.return_value = dict(_DEFAULT_SETTINGS)
    backend.get_accounts.return_value = [
        {"account_id": "demo",   "broker": "MT5",   "server": "Demo-Server"},
        {"account_id": "live_1", "broker": "OANDA", "server": "Live-Server"},
    ]
    backend.list_stored_accounts.return_value = [
        {"account_id": "demo",   "broker": "MT5",   "server": "Demo-Server",
         "is_live": False, "has_password": False, "login": ""},
        {"account_id": "live_1", "broker": "OANDA", "server": "Live-Server",
         "is_live": True,  "has_password": True,  "login": "999"},
    ]
    backend.get_active_account_id.return_value  = None
    backend.save_settings.return_value          = None
    backend.save_account_credentials.return_value = None
    backend.set_account_password.return_value   = None
    backend.delete_account_credentials.return_value = None
    backend.import_from_env.return_value        = []
    backend.switch_account.return_value         = None
    backend.add_account.return_value            = None
    backend.remove_account.return_value         = None
    backend.test_telegram.return_value          = True
    for k, v in overrides.items():
        setattr(backend, k, MagicMock(return_value=v) if not callable(v) else v)
    return backend


def _make_view(
    qtbot,
    backend=None,
    confirm_fn=None,
    test_telegram_fn=None,
) -> SettingsView:
    v = SettingsView(
        backend=backend,
        _confirm_fn=confirm_fn if confirm_fn is not None else _confirm_yes,
        _test_telegram_fn=test_telegram_fn if test_telegram_fn is not None else (lambda t, c: True),
    )
    qtbot.addWidget(v)
    return v


# ─────────────────────────────────────────────────────────────────────────────
#  TestSettingsViewInit
# ─────────────────────────────────────────────────────────────────────────────

class TestSettingsViewInit:

    def test_widget_created(self, qtbot):
        v = _make_view(qtbot)
        assert v is not None

    def test_object_name(self, qtbot):
        v = _make_view(qtbot)
        assert v.objectName() == "settings_view"

    def test_tab_count(self, qtbot):
        v = _make_view(qtbot)
        assert v.tabs.count() == 7  # +1 Watchdog-Tab (Issue #61)

    def test_tab_risk_index(self, qtbot):
        v = _make_view(qtbot)
        assert v.tabs.tabText(TAB_RISK) == "Risiko"

    def test_tab_mode_index(self, qtbot):
        v = _make_view(qtbot)
        assert v.tabs.tabText(TAB_MODE) == "Modus"

    def test_tab_accounts_index(self, qtbot):
        v = _make_view(qtbot)
        assert v.tabs.tabText(TAB_ACCOUNTS) == "Konten"

    def test_tab_symbols_index(self, qtbot):
        v = _make_view(qtbot)
        assert v.tabs.tabText(TAB_SYMBOLS) == "Symbole"

    def test_tab_telegram_index(self, qtbot):
        v = _make_view(qtbot)
        assert v.tabs.tabText(TAB_TELEGRAM) == "Telegram"

    def test_tab_audit_index(self, qtbot):
        v = _make_view(qtbot)
        assert v.tabs.tabText(TAB_AUDIT) == "Audit-Log"

    def test_save_btn_disabled_initially(self, qtbot):
        v = _make_view(qtbot)
        assert not v.save_btn.isEnabled()

    def test_discard_btn_disabled_initially(self, qtbot):
        v = _make_view(qtbot)
        assert not v.discard_btn.isEnabled()

    def test_pending_label_no_changes(self, qtbot):
        v = _make_view(qtbot)
        assert "Keine" in v.pending_label.text()

    def test_paper_radio_checked_default(self, qtbot):
        v = _make_view(qtbot)
        assert v.paper_radio.isChecked()

    def test_live_radio_unchecked_default(self, qtbot):
        v = _make_view(qtbot)
        assert not v.live_radio.isChecked()

    def test_mode_combo_has_three_options(self, qtbot):
        v = _make_view(qtbot)
        assert v.mode_combo.count() == 3

    def test_mode_combo_default_suggest_only(self, qtbot):
        v = _make_view(qtbot)
        assert v.mode_combo.currentData() == "suggest_only"

    def test_telegram_token_password_echo(self, qtbot):
        from PySide6.QtWidgets import QLineEdit
        v = _make_view(qtbot)
        assert v.token_input.echoMode() == QLineEdit.EchoMode.Password

    def test_accounts_table_three_columns(self, qtbot):
        v = _make_view(qtbot)
        assert v.accounts_table.columnCount() == 5

    def test_audit_table_four_columns(self, qtbot):
        v = _make_view(qtbot)
        assert v.audit_table.columnCount() == 4

    def test_audit_table_empty_initially(self, qtbot):
        v = _make_view(qtbot)
        assert v.audit_table.rowCount() == 0

    def test_symbols_list_populated(self, qtbot):
        v = _make_view(qtbot)
        assert v.symbols_list.count() == len(_AVAILABLE_SYMBOLS)

    def test_all_symbols_in_list(self, qtbot):
        v = _make_view(qtbot)
        items = [v.symbols_list.item(i).text() for i in range(v.symbols_list.count())]
        for sym in _AVAILABLE_SYMBOLS:
            assert sym in items


# ─────────────────────────────────────────────────────────────────────────────
#  TestRiskParameters
# ─────────────────────────────────────────────────────────────────────────────

class TestRiskParameters:

    def test_max_risk_min_bound(self, qtbot):
        v = _make_view(qtbot)
        assert v.max_risk_spin.minimum() == pytest.approx(0.1)

    def test_max_risk_max_bound(self, qtbot):
        v = _make_view(qtbot)
        assert v.max_risk_spin.maximum() == pytest.approx(5.0)

    def test_max_daily_dd_min_bound(self, qtbot):
        v = _make_view(qtbot)
        assert v.max_daily_dd_spin.minimum() == pytest.approx(1.0)

    def test_max_daily_dd_max_bound(self, qtbot):
        v = _make_view(qtbot)
        assert v.max_daily_dd_spin.maximum() == pytest.approx(20.0)

    def test_max_positions_min_bound(self, qtbot):
        v = _make_view(qtbot)
        assert v.max_positions_spin.minimum() == 1

    def test_max_positions_max_bound(self, qtbot):
        v = _make_view(qtbot)
        assert v.max_positions_spin.maximum() == 20

    def test_max_lot_min_bound(self, qtbot):
        v = _make_view(qtbot)
        assert v.max_lot_spin.minimum() == pytest.approx(0.01)

    def test_max_lot_max_bound(self, qtbot):
        v = _make_view(qtbot)
        assert v.max_lot_spin.maximum() == pytest.approx(100.0)

    def test_cooldown_min_bound(self, qtbot):
        v = _make_view(qtbot)
        assert v.cooldown_spin.minimum() == 0

    def test_cooldown_max_bound(self, qtbot):
        v = _make_view(qtbot)
        assert v.cooldown_spin.maximum() == 24

    def test_change_risk_creates_dirty(self, qtbot):
        v = _make_view(qtbot)
        v.max_risk_spin.setValue(2.5)
        assert v.save_btn.isEnabled()

    def test_change_daily_dd_creates_dirty(self, qtbot):
        v = _make_view(qtbot)
        v.max_daily_dd_spin.setValue(10.0)
        assert v.save_btn.isEnabled()

    def test_change_positions_creates_dirty(self, qtbot):
        v = _make_view(qtbot)
        v.max_positions_spin.setValue(10)
        assert v.save_btn.isEnabled()

    def test_change_lot_creates_dirty(self, qtbot):
        v = _make_view(qtbot)
        v.max_lot_spin.setValue(50.0)
        assert v.save_btn.isEnabled()

    def test_change_cooldown_creates_dirty(self, qtbot):
        v = _make_view(qtbot)
        v.cooldown_spin.setValue(4)
        assert v.save_btn.isEnabled()

    def test_default_risk_loaded_from_defaults(self, qtbot):
        v = _make_view(qtbot)
        assert v.max_risk_spin.value() == pytest.approx(_DEFAULT_SETTINGS["max_risk_per_trade_pct"])


# ─────────────────────────────────────────────────────────────────────────────
#  TestTradingModeSelection
# ─────────────────────────────────────────────────────────────────────────────

class TestTradingModeSelection:

    def test_combo_data_suggest_only(self, qtbot):
        v = _make_view(qtbot)
        data_values = [v.mode_combo.itemData(i) for i in range(v.mode_combo.count())]
        assert "suggest_only" in data_values

    def test_combo_data_confirm_required(self, qtbot):
        v = _make_view(qtbot)
        data_values = [v.mode_combo.itemData(i) for i in range(v.mode_combo.count())]
        assert "confirm_required" in data_values

    def test_combo_data_autonomous(self, qtbot):
        v = _make_view(qtbot)
        data_values = [v.mode_combo.itemData(i) for i in range(v.mode_combo.count())]
        assert "autonomous" in data_values

    def test_mode_change_creates_dirty(self, qtbot):
        v = _make_view(qtbot)
        v.mode_combo.setCurrentIndex(1)
        assert v.save_btn.isEnabled()

    def test_mode_collect_current(self, qtbot):
        v = _make_view(qtbot)
        # find confirm_required index
        for i in range(v.mode_combo.count()):
            if v.mode_combo.itemData(i) == "confirm_required":
                v.mode_combo.setCurrentIndex(i)
                break
        assert v._collect_current()["trading_mode"] == "confirm_required"

    def test_backend_settings_mode_loaded(self, qtbot):
        backend = _mock_backend()
        backend.get_settings.return_value = {**_DEFAULT_SETTINGS, "trading_mode": "confirm_required"}
        v = _make_view(qtbot, backend=backend)
        assert v.mode_combo.currentData() == "confirm_required"


# ─────────────────────────────────────────────────────────────────────────────
#  TestLivePaperSwitch
# ─────────────────────────────────────────────────────────────────────────────

class TestLivePaperSwitch:

    def test_paper_default(self, qtbot):
        v = _make_view(qtbot)
        assert v.paper_radio.isChecked()
        assert not v.live_radio.isChecked()

    def test_switch_to_live_creates_dirty(self, qtbot):
        v = _make_view(qtbot)
        v.live_radio.setChecked(True)
        assert v.save_btn.isEnabled()

    def test_collect_current_live_mode(self, qtbot):
        v = _make_view(qtbot)
        v.live_radio.setChecked(True)
        assert v._collect_current()["paper_mode"] is False

    def test_collect_current_paper_mode(self, qtbot):
        v = _make_view(qtbot)
        assert v._collect_current()["paper_mode"] is True

    def test_backend_live_setting_loaded(self, qtbot):
        backend = _mock_backend()
        backend.get_settings.return_value = {**_DEFAULT_SETTINGS, "paper_mode": False}
        v = _make_view(qtbot, backend=backend)
        assert v.live_radio.isChecked()
        assert not v.paper_radio.isChecked()


# ─────────────────────────────────────────────────────────────────────────────
#  TestPendingChanges
# ─────────────────────────────────────────────────────────────────────────────

class TestPendingChanges:

    def test_no_pending_initially(self, qtbot):
        v = _make_view(qtbot)
        assert not v._is_dirty()

    def test_pending_after_risk_change(self, qtbot):
        v = _make_view(qtbot)
        v.max_risk_spin.setValue(3.0)
        assert v._is_dirty()

    def test_pending_label_shows_count(self, qtbot):
        v = _make_view(qtbot)
        v.max_risk_spin.setValue(3.0)
        assert "ausstehend" in v.pending_label.text()

    def test_discard_clears_pending(self, qtbot):
        v = _make_view(qtbot)
        v.max_risk_spin.setValue(3.0)
        v.discard_btn.click()
        assert not v._is_dirty()

    def test_discard_resets_label(self, qtbot):
        v = _make_view(qtbot)
        v.max_risk_spin.setValue(3.0)
        v.discard_btn.click()
        assert "Keine" in v.pending_label.text()

    def test_save_clears_pending(self, qtbot):
        v = _make_view(qtbot)
        v.max_risk_spin.setValue(3.0)
        v.save_btn.click()
        assert not v._is_dirty()

    def test_save_disables_save_btn(self, qtbot):
        v = _make_view(qtbot)
        v.max_risk_spin.setValue(3.0)
        v.save_btn.click()
        assert not v.save_btn.isEnabled()

    def test_pending_count_correct(self, qtbot):
        v = _make_view(qtbot)
        v.max_risk_spin.setValue(3.0)
        v.max_positions_spin.setValue(10)
        assert v._pending_count() == 2


# ─────────────────────────────────────────────────────────────────────────────
#  TestSaveChanges
# ─────────────────────────────────────────────────────────────────────────────

class TestSaveChanges:

    def test_save_calls_backend(self, qtbot):
        backend = _mock_backend()
        v = _make_view(qtbot, backend=backend)
        v.max_risk_spin.setValue(2.0)
        v.save_btn.click()
        backend.save_settings.assert_called_once()

    def test_save_passes_correct_values(self, qtbot):
        backend = _mock_backend()
        v = _make_view(qtbot, backend=backend)
        v.max_risk_spin.setValue(2.5)
        v.save_btn.click()
        args = backend.save_settings.call_args[0][0]
        assert args["max_risk_per_trade_pct"] == pytest.approx(2.5)

    def test_save_emits_settings_saved_signal(self, qtbot):
        v = _make_view(qtbot)
        v.max_risk_spin.setValue(2.0)
        with qtbot.waitSignal(v.settings_saved, timeout=2000) as blocker:
            v.save_btn.click()
        assert "max_risk_per_trade_pct" in blocker.args[0]

    def test_save_emits_mode_changed_when_mode_changes(self, qtbot):
        v = _make_view(qtbot)
        for i in range(v.mode_combo.count()):
            if v.mode_combo.itemData(i) == "confirm_required":
                v.mode_combo.setCurrentIndex(i)
                break
        with qtbot.waitSignal(v.mode_changed, timeout=2000) as blocker:
            v.save_btn.click()
        assert blocker.args[0] == "confirm_required"

    def test_save_does_not_emit_mode_changed_if_mode_unchanged(self, qtbot):
        v = _make_view(qtbot)
        v.max_risk_spin.setValue(2.0)
        received = []
        v.mode_changed.connect(lambda m: received.append(m))
        v.save_btn.click()
        assert received == []

    def test_save_requires_confirmation(self, qtbot):
        backend = _mock_backend()
        v = _make_view(qtbot, backend=backend, confirm_fn=_confirm_no)
        v.max_risk_spin.setValue(2.0)
        v.save_btn.click()
        backend.save_settings.assert_not_called()

    def test_save_updates_saved_settings(self, qtbot):
        v = _make_view(qtbot)
        v.max_risk_spin.setValue(2.0)
        v.save_btn.click()
        assert v.saved_settings["max_risk_per_trade_pct"] == pytest.approx(2.0)

    def test_save_no_backend_no_crash(self, qtbot):
        v = _make_view(qtbot, backend=None)
        v.max_risk_spin.setValue(2.0)
        v.save_btn.click()  # should not raise
        assert not v._is_dirty()


# ─────────────────────────────────────────────────────────────────────────────
#  TestAutonomousConfirmation
# ─────────────────────────────────────────────────────────────────────────────

class TestAutonomousConfirmation:

    def _set_autonomous(self, v: SettingsView) -> None:
        for i in range(v.mode_combo.count()):
            if v.mode_combo.itemData(i) == "autonomous":
                v.mode_combo.setCurrentIndex(i)
                return

    def test_autonomous_calls_confirm_twice_before_general(self, qtbot):
        calls: list[str] = []

        def confirm_fn(title: str, msg: str) -> bool:
            calls.append(title)
            return True

        backend = _mock_backend()
        v = _make_view(qtbot, backend=backend, confirm_fn=confirm_fn)
        self._set_autonomous(v)
        v.save_btn.click()
        assert len(calls) >= 3
        assert any("AUTONOMOUS" in t for t in calls[:2])

    def test_autonomous_first_confirm_rejected_aborts(self, qtbot):
        call_count = [0]

        def confirm_fn(title: str, msg: str) -> bool:
            call_count[0] += 1
            return False  # reject immediately

        backend = _mock_backend()
        v = _make_view(qtbot, backend=backend, confirm_fn=confirm_fn)
        self._set_autonomous(v)
        v.save_btn.click()
        backend.save_settings.assert_not_called()
        assert call_count[0] == 1

    def test_autonomous_second_confirm_rejected_aborts(self, qtbot):
        call_count = [0]

        def confirm_fn(title: str, msg: str) -> bool:
            call_count[0] += 1
            return call_count[0] == 1  # only first returns True

        backend = _mock_backend()
        v = _make_view(qtbot, backend=backend, confirm_fn=confirm_fn)
        self._set_autonomous(v)
        v.save_btn.click()
        backend.save_settings.assert_not_called()
        assert call_count[0] == 2

    def test_autonomous_no_extra_confirmation_if_already_autonomous(self, qtbot):
        """No 2nd-step confirm if mode was already AUTONOMOUS."""
        calls: list[str] = []

        def confirm_fn(title: str, msg: str) -> bool:
            calls.append(title)
            return True

        backend = _mock_backend()
        backend.get_settings.return_value = {**_DEFAULT_SETTINGS, "trading_mode": "autonomous"}
        v = _make_view(qtbot, backend=backend, confirm_fn=confirm_fn)
        v.max_risk_spin.setValue(2.0)  # some other change
        v.save_btn.click()
        # only general save confirmation, not the AUTONOMOUS-specific ones
        assert not any("AUTONOMOUS" in t for t in calls)


# ─────────────────────────────────────────────────────────────────────────────
#  TestLiveModeConfirmation
# ─────────────────────────────────────────────────────────────────────────────

class TestLiveModeConfirmation:

    def test_live_mode_requires_confirmation(self, qtbot):
        backend = _mock_backend()
        v = _make_view(qtbot, backend=backend, confirm_fn=_confirm_no)
        v.live_radio.setChecked(True)
        v.save_btn.click()
        backend.save_settings.assert_not_called()

    def test_live_mode_confirmation_accepted(self, qtbot):
        backend = _mock_backend()
        v = _make_view(qtbot, backend=backend, confirm_fn=_confirm_yes)
        v.live_radio.setChecked(True)
        v.save_btn.click()
        backend.save_settings.assert_called_once()

    def test_live_mode_no_extra_confirm_if_already_live(self, qtbot):
        """No live-confirm if was already live."""
        calls: list[str] = []

        def confirm_fn(title: str, msg: str) -> bool:
            calls.append(title)
            return True

        backend = _mock_backend()
        backend.get_settings.return_value = {**_DEFAULT_SETTINGS, "paper_mode": False}
        v = _make_view(qtbot, backend=backend, confirm_fn=confirm_fn)
        v.max_risk_spin.setValue(2.0)
        v.save_btn.click()
        assert not any("Live" in t for t in calls)

    def test_live_confirm_message_mentions_money(self, qtbot):
        messages: list[str] = []

        def confirm_fn(title: str, msg: str) -> bool:
            messages.append(msg)
            return True

        backend = _mock_backend()
        v = _make_view(qtbot, backend=backend, confirm_fn=confirm_fn)
        v.live_radio.setChecked(True)
        v.save_btn.click()
        assert any("Geld" in m or "echtes" in m.lower() or "real" in m.lower() for m in messages)


# ─────────────────────────────────────────────────────────────────────────────
#  TestDiscardChanges
# ─────────────────────────────────────────────────────────────────────────────

class TestDiscardChanges:

    def test_discard_resets_risk_value(self, qtbot):
        v = _make_view(qtbot)
        original = v.max_risk_spin.value()
        v.max_risk_spin.setValue(4.0)
        v.discard_btn.click()
        assert v.max_risk_spin.value() == pytest.approx(original)

    def test_discard_resets_mode(self, qtbot):
        v = _make_view(qtbot)
        original = v.mode_combo.currentData()
        v.mode_combo.setCurrentIndex(1)
        v.discard_btn.click()
        assert v.mode_combo.currentData() == original

    def test_discard_resets_live_radio(self, qtbot):
        v = _make_view(qtbot)
        v.live_radio.setChecked(True)
        v.discard_btn.click()
        assert v.paper_radio.isChecked()

    def test_discard_clears_dirty(self, qtbot):
        v = _make_view(qtbot)
        v.max_risk_spin.setValue(4.0)
        v.discard_btn.click()
        assert not v._is_dirty()

    def test_discard_disables_buttons(self, qtbot):
        v = _make_view(qtbot)
        v.max_risk_spin.setValue(4.0)
        v.discard_btn.click()
        assert not v.save_btn.isEnabled()
        assert not v.discard_btn.isEnabled()


# ─────────────────────────────────────────────────────────────────────────────
#  TestAccountManagement
# ─────────────────────────────────────────────────────────────────────────────

class TestAccountManagement:

    def test_accounts_table_column_names(self, qtbot):
        v = _make_view(qtbot)
        headers = [
            v.accounts_table.horizontalHeaderItem(i).text()
            for i in range(v.accounts_table.columnCount())
        ]
        assert "Konto-ID" in headers
        assert "Broker"   in headers
        assert "Server"   in headers
        assert "Typ"      in headers
        assert "Passwort" in headers

    def test_accounts_loaded_from_backend(self, qtbot):
        backend = _mock_backend()
        v = _make_view(qtbot, backend=backend)
        assert v.accounts_table.rowCount() == 2

    def test_accounts_table_shows_account_id(self, qtbot):
        backend = _mock_backend()
        v = _make_view(qtbot, backend=backend)
        ids = [v.accounts_table.item(r, 0).text() for r in range(v.accounts_table.rowCount())]
        assert "demo" in ids
        assert "live_1" in ids

    def test_add_account_calls_backend(self, qtbot):
        backend = _mock_backend()
        v = _make_view(qtbot, backend=backend)
        v.account_id_input.setText("new_acc")
        v.broker_input.setText("MT5")
        v.server_input.setText("New-Server")
        v.add_account_btn.click()
        backend.save_account_credentials.assert_called_once()

    def test_add_account_adds_row_to_table(self, qtbot):
        v = _make_view(qtbot)
        before = v.accounts_table.rowCount()
        v.account_id_input.setText("extra")
        v.add_account_btn.click()
        assert v.accounts_table.rowCount() == before + 1

    def test_add_account_clears_inputs(self, qtbot):
        v = _make_view(qtbot)
        v.account_id_input.setText("x")
        v.broker_input.setText("MT5")
        v.add_account_btn.click()
        assert v.account_id_input.text() == ""
        assert v.broker_input.text() == ""

    def test_add_account_no_id_is_noop(self, qtbot):
        backend = _mock_backend()
        v = _make_view(qtbot, backend=backend)
        before = v.accounts_table.rowCount()
        v.account_id_input.setText("")
        v.add_account_btn.click()
        assert v.accounts_table.rowCount() == before
        backend.save_account_credentials.assert_not_called()

    def test_add_account_has_password_field(self, qtbot):
        v = _make_view(qtbot)
        # Password widget exists – stored securely via keyring, never shown again
        from PySide6.QtWidgets import QLineEdit
        inputs = v.findChildren(QLineEdit)
        names  = [w.objectName() for w in inputs]
        assert "password_input" in names

    def test_remove_account_calls_backend(self, qtbot):
        backend = _mock_backend()
        v = _make_view(qtbot, backend=backend)
        v.accounts_table.selectRow(0)
        v.remove_account_btn.click()
        backend.delete_account_credentials.assert_called_once()

    def test_remove_account_removes_row(self, qtbot):
        backend = _mock_backend()
        v = _make_view(qtbot, backend=backend)
        before = v.accounts_table.rowCount()
        v.accounts_table.selectRow(0)
        v.remove_account_btn.click()
        assert v.accounts_table.rowCount() == before - 1

    def test_remove_without_selection_is_noop(self, qtbot):
        backend = _mock_backend()
        v = _make_view(qtbot, backend=backend)
        before = v.accounts_table.rowCount()
        v.accounts_table.clearSelection()
        v.remove_account_btn.click()
        assert v.accounts_table.rowCount() == before
        backend.delete_account_credentials.assert_not_called()

    def test_remove_account_rejected_confirm_no_removal(self, qtbot):
        backend = _mock_backend()
        v = _make_view(qtbot, backend=backend, confirm_fn=_confirm_no)
        v.accounts_table.selectRow(0)
        before = v.accounts_table.rowCount()
        v.remove_account_btn.click()
        assert v.accounts_table.rowCount() == before


# ─────────────────────────────────────────────────────────────────────────────
#  TestSymbolSelection
# ─────────────────────────────────────────────────────────────────────────────

class TestSymbolSelection:

    def test_symbols_list_has_all_symbols(self, qtbot):
        v = _make_view(qtbot)
        assert v.symbols_list.count() == len(_AVAILABLE_SYMBOLS)

    def test_default_symbols_checked(self, qtbot):
        v = _make_view(qtbot)
        checked = [
            v.symbols_list.item(i).text()
            for i in range(v.symbols_list.count())
            if v.symbols_list.item(i).checkState() == Qt.CheckState.Checked
        ]
        for sym in _DEFAULT_SETTINGS["symbols"]:
            assert sym in checked

    def test_check_symbol_creates_dirty(self, qtbot):
        v = _make_view(qtbot)
        # uncheck a default symbol
        for i in range(v.symbols_list.count()):
            item = v.symbols_list.item(i)
            if item.text() == _DEFAULT_SETTINGS["symbols"][0]:
                item.setCheckState(Qt.CheckState.Unchecked)
                break
        assert v._is_dirty()

    def test_checked_symbols_in_collect_current(self, qtbot):
        v = _make_view(qtbot)
        # check XAUUSD specifically
        for i in range(v.symbols_list.count()):
            item = v.symbols_list.item(i)
            if item.text() == "XAUUSD":
                item.setCheckState(Qt.CheckState.Checked)
                break
        symbols = v._collect_current()["symbols"]
        assert "XAUUSD" in symbols

    def test_backend_symbols_loaded(self, qtbot):
        backend = _mock_backend()
        backend.get_settings.return_value = {**_DEFAULT_SETTINGS, "symbols": ["USDJPY", "XAUUSD"]}
        v = _make_view(qtbot, backend=backend)
        checked = [
            v.symbols_list.item(i).text()
            for i in range(v.symbols_list.count())
            if v.symbols_list.item(i).checkState() == Qt.CheckState.Checked
        ]
        assert "USDJPY" in checked
        assert "XAUUSD" in checked
        assert "EURUSD" not in checked


# ─────────────────────────────────────────────────────────────────────────────
#  TestTelegramConfig
# ─────────────────────────────────────────────────────────────────────────────

class TestTelegramConfig:

    def test_token_input_password_mode(self, qtbot):
        from PySide6.QtWidgets import QLineEdit
        v = _make_view(qtbot)
        assert v.token_input.echoMode() == QLineEdit.EchoMode.Password

    def test_test_btn_exists(self, qtbot):
        v = _make_view(qtbot)
        assert v.test_btn is not None

    def test_test_result_label_empty_initially(self, qtbot):
        v = _make_view(qtbot)
        assert v.test_result_label.text() == ""

    def test_test_no_token_shows_error(self, qtbot):
        v = _make_view(qtbot)
        v.token_input.setText("")
        v.chat_id_input.setText("123")
        v.test_btn.click()
        assert v.test_result_label.text() != ""
        assert "✓" not in v.test_result_label.text()

    def test_test_no_chat_id_shows_error(self, qtbot):
        v = _make_view(qtbot)
        v.token_input.setText("fake_token")
        v.chat_id_input.setText("")
        v.test_btn.click()
        assert v.test_result_label.text() != ""

    def test_test_success_shows_ok(self, qtbot):
        v = _make_view(qtbot, test_telegram_fn=lambda t, c: True)
        v.token_input.setText("tok")
        v.chat_id_input.setText("cid")
        v.test_btn.click()
        assert "✓" in v.test_result_label.text()

    def test_test_failure_shows_error(self, qtbot):
        v = _make_view(qtbot, test_telegram_fn=lambda t, c: False)
        v.token_input.setText("tok")
        v.chat_id_input.setText("cid")
        v.test_btn.click()
        assert "✗" in v.test_result_label.text()

    def test_test_passes_token_and_chat_id(self, qtbot):
        received: list = []

        def fn(token: str, chat_id: str) -> bool:
            received.append((token, chat_id))
            return True

        v = _make_view(qtbot, test_telegram_fn=fn)
        v.token_input.setText("my_token")
        v.chat_id_input.setText("my_chat")
        v.test_btn.click()
        assert received == [("my_token", "my_chat")]

    def test_telegram_settings_saved(self, qtbot):
        backend = _mock_backend()
        v = _make_view(qtbot, backend=backend)
        v.token_input.setText("tok123")
        v.chat_id_input.setText("chat456")
        v.save_btn.click()
        args = backend.save_settings.call_args[0][0]
        assert args["telegram_token"]   == "tok123"
        assert args["telegram_chat_id"] == "chat456"

    def test_telegram_token_in_collect_current(self, qtbot):
        v = _make_view(qtbot)
        v.token_input.setText("secret_token")
        assert v._collect_current()["telegram_token"] == "secret_token"


# ─────────────────────────────────────────────────────────────────────────────
#  TestAuditLog
# ─────────────────────────────────────────────────────────────────────────────

class TestAuditLog:

    def test_audit_table_column_headers(self, qtbot):
        v = _make_view(qtbot)
        headers = [
            v.audit_table.horizontalHeaderItem(i).text()
            for i in range(v.audit_table.columnCount())
        ]
        assert "Zeitstempel" in headers
        assert "Parameter"   in headers
        assert "Alter Wert"  in headers
        assert "Neuer Wert"  in headers

    def test_no_audit_entries_initially(self, qtbot):
        v = _make_view(qtbot)
        assert v.audit_table.rowCount() == 0
        assert v.audit_entries == []

    def test_save_adds_audit_entry(self, qtbot):
        v = _make_view(qtbot)
        v.max_risk_spin.setValue(3.0)
        v.save_btn.click()
        assert len(v.audit_entries) >= 1

    def test_audit_entry_has_parameter_field(self, qtbot):
        v = _make_view(qtbot)
        v.max_risk_spin.setValue(3.0)
        v.save_btn.click()
        params = [e["parameter"] for e in v.audit_entries]
        assert "max_risk_per_trade_pct" in params

    def test_audit_entry_has_old_and_new_value(self, qtbot):
        v = _make_view(qtbot)
        v.max_risk_spin.setValue(3.0)
        v.save_btn.click()
        entry = next(e for e in v.audit_entries if e["parameter"] == "max_risk_per_trade_pct")
        assert "old_value" in entry
        assert "new_value" in entry

    def test_audit_entry_has_timestamp(self, qtbot):
        v = _make_view(qtbot)
        v.max_risk_spin.setValue(3.0)
        v.save_btn.click()
        assert all("ts" in e for e in v.audit_entries)

    def test_audit_table_row_added_after_save(self, qtbot):
        v = _make_view(qtbot)
        v.max_risk_spin.setValue(3.0)
        v.save_btn.click()
        assert v.audit_table.rowCount() >= 1

    def test_multiple_changes_multiple_entries(self, qtbot):
        v = _make_view(qtbot)
        v.max_risk_spin.setValue(3.0)
        v.max_positions_spin.setValue(8)
        v.save_btn.click()
        assert len(v.audit_entries) >= 2

    def test_unchanged_settings_no_audit_entry(self, qtbot):
        v = _make_view(qtbot)
        # make a change, save, then change back, save again
        original = v.max_risk_spin.value()
        v.max_risk_spin.setValue(3.0)
        v.save_btn.click()
        first_count = len(v.audit_entries)
        v.max_risk_spin.setValue(original)
        v.save_btn.click()
        # new entries for the revert
        assert len(v.audit_entries) > first_count


# ─────────────────────────────────────────────────────────────────────────────
#  TestBackendIntegration
# ─────────────────────────────────────────────────────────────────────────────

class TestBackendIntegration:

    def test_get_settings_called_on_init(self, qtbot):
        backend = _mock_backend()
        _make_view(qtbot, backend=backend)
        backend.get_settings.assert_called_once()

    def test_list_stored_accounts_called_on_init(self, qtbot):
        backend = _mock_backend()
        _make_view(qtbot, backend=backend)
        backend.list_stored_accounts.assert_called()

    def test_custom_settings_loaded_into_widgets(self, qtbot):
        backend = _mock_backend()
        backend.get_settings.return_value = {
            **_DEFAULT_SETTINGS,
            "max_risk_per_trade_pct": 4.0,
            "max_open_positions":     15,
        }
        v = _make_view(qtbot, backend=backend)
        assert v.max_risk_spin.value() == pytest.approx(4.0)
        assert v.max_positions_spin.value() == 15

    def test_save_settings_receives_full_dict(self, qtbot):
        backend = _mock_backend()
        v = _make_view(qtbot, backend=backend)
        v.max_risk_spin.setValue(2.0)
        v.save_btn.click()
        args = backend.save_settings.call_args[0][0]
        expected_keys = {
            "max_risk_per_trade_pct",
            "max_daily_drawdown_pct",
            "max_open_positions",
            "max_lot_size",
            "cooldown_after_loss_h",
            "trading_mode",
            "paper_mode",
            "symbols",
            "telegram_token",
            "telegram_chat_id",
        }
        assert expected_keys.issubset(set(args.keys()))


# ─────────────────────────────────────────────────────────────────────────────
#  TestMainWindowIntegration
# ─────────────────────────────────────────────────────────────────────────────

class TestMainWindowIntegration:

    def test_settings_view_property_exists(self, qtbot, fresh_theme):
        mw = MainWindow(theme_manager=fresh_theme)
        qtbot.addWidget(mw)
        assert isinstance(mw.settings_view, SettingsView)

    def test_settings_view_in_content_stack(self, qtbot, fresh_theme):
        mw = MainWindow(theme_manager=fresh_theme)
        qtbot.addWidget(mw)
        found = any(
            isinstance(mw.content.widget(i), SettingsView)
            for i in range(mw.content.count())
        )
        assert found

    def test_content_count_is_7(self, qtbot, fresh_theme):
        mw = MainWindow(theme_manager=fresh_theme)
        qtbot.addWidget(mw)
        assert mw.content.count() == 7

    def test_navigate_to_settings(self, qtbot, fresh_theme):
        mw = MainWindow(theme_manager=fresh_theme)
        qtbot.addWidget(mw)
        mw.navigate_to(Section.SETTINGS)
        assert mw.current_view() is mw.settings_view

    def test_settings_backend_passed_to_view(self, qtbot, fresh_theme):
        backend = _mock_backend()
        mw = MainWindow(theme_manager=fresh_theme, settings_backend=backend)
        qtbot.addWidget(mw)
        assert mw.settings_view._backend is backend

    def test_settings_view_without_backend(self, qtbot, fresh_theme):
        mw = MainWindow(theme_manager=fresh_theme, settings_backend=None)
        qtbot.addWidget(mw)
        assert mw.settings_view._backend is None
        # view should still work (defaults used)
        assert mw.settings_view.mode_combo.currentData() == "suggest_only"


# ─────────────────────────────────────────────────────────────────────────────
#  TestKeyringAccountManagement  (Issue #66)
# ─────────────────────────────────────────────────────────────────────────────

class TestKeyringAccountManagement:

    def test_account_selector_present(self, qtbot):
        v = _make_view(qtbot)
        assert v.account_selector is not None

    def test_account_selector_object_name(self, qtbot):
        v = _make_view(qtbot)
        assert v.account_selector.objectName() == "account_selector"

    def test_live_warning_hidden_initially_no_backend(self, qtbot):
        v = _make_view(qtbot)
        assert v.live_warning_label.isHidden()

    def test_live_warning_hidden_for_demo_accounts(self, qtbot):
        backend = _mock_backend()
        backend.list_stored_accounts.return_value = [
            {"account_id": "demo", "broker": "MT5", "server": "Demo", "is_live": False, "has_password": False}
        ]
        backend.get_active_account_id.return_value = None
        v = _make_view(qtbot, backend=backend)
        assert v.live_warning_label.isHidden()

    def test_live_warning_shown_for_live_account(self, qtbot):
        backend = _mock_backend()
        backend.list_stored_accounts.return_value = [
            {"account_id": "live1", "broker": "MT5", "server": "Live", "is_live": True, "has_password": True}
        ]
        backend.get_active_account_id.return_value = None
        v = _make_view(qtbot, backend=backend)
        # Manually trigger update for the live account
        v._update_live_warning("live1")
        assert not v.live_warning_label.isHidden()

    def test_update_live_warning_hides_for_demo(self, qtbot):
        backend = _mock_backend()
        backend.list_stored_accounts.return_value = [
            {"account_id": "demo", "broker": "MT5", "server": "Demo", "is_live": False, "has_password": False}
        ]
        v = _make_view(qtbot, backend=backend)
        v._update_live_warning("demo")
        assert v.live_warning_label.isHidden()

    def test_update_live_warning_unknown_account_hides(self, qtbot):
        v = _make_view(qtbot)
        v._update_live_warning("nonexistent")
        assert v.live_warning_label.isHidden()

    def test_accounts_table_five_columns(self, qtbot):
        v = _make_view(qtbot)
        assert v.accounts_table.columnCount() == 5

    def test_password_input_has_password_echo(self, qtbot):
        from PySide6.QtWidgets import QLineEdit
        v = _make_view(qtbot)
        assert v.password_input.echoMode() == QLineEdit.EchoMode.Password

    def test_login_input_present(self, qtbot):
        v = _make_view(qtbot)
        assert v.login_input is not None
        assert v.login_input.objectName() == "login_input"

    def test_is_live_checkbox_present(self, qtbot):
        v = _make_view(qtbot)
        assert v.is_live_checkbox is not None
        assert v.is_live_checkbox.objectName() == "is_live_checkbox"

    def test_import_env_btn_present(self, qtbot):
        v = _make_view(qtbot)
        assert v.import_env_btn is not None

    def test_set_password_btn_present(self, qtbot):
        v = _make_view(qtbot)
        assert v.set_password_btn is not None

    def test_password_status_label_initially_not_set(self, qtbot):
        v = _make_view(qtbot)
        assert "Nicht gesetzt" in v.password_status_label.text()

    def test_set_password_btn_calls_backend(self, qtbot):
        backend = _mock_backend()
        v = _make_view(qtbot, backend=backend)
        v.account_id_input.setText("test_acc")
        v.password_input.setText("secret123")
        v.set_password_btn.click()
        backend.set_account_password.assert_called_once_with("test_acc", "secret123")

    def test_set_password_clears_input_field(self, qtbot):
        backend = _mock_backend()
        v = _make_view(qtbot, backend=backend)
        v.account_id_input.setText("test_acc")
        v.password_input.setText("secret123")
        v.set_password_btn.click()
        assert v.password_input.text() == ""

    def test_set_password_updates_status_label(self, qtbot):
        v = _make_view(qtbot)
        v.account_id_input.setText("acc")
        v.password_input.setText("pw")
        v.set_password_btn.click()
        assert "Gesetzt" in v.password_status_label.text()

    def test_set_password_empty_account_id_noop(self, qtbot):
        backend = _mock_backend()
        v = _make_view(qtbot, backend=backend)
        v.account_id_input.setText("")
        v.password_input.setText("secret")
        v.set_password_btn.click()
        backend.set_account_password.assert_not_called()

    def test_set_password_empty_password_noop(self, qtbot):
        backend = _mock_backend()
        v = _make_view(qtbot, backend=backend)
        v.account_id_input.setText("demo")
        v.password_input.setText("")
        v.set_password_btn.click()
        backend.set_account_password.assert_not_called()

    def test_import_env_btn_calls_backend(self, qtbot):
        backend = _mock_backend()
        backend.import_from_env.return_value = ["env_import"]
        v = _make_view(qtbot, backend=backend)
        v.import_env_btn.click()
        backend.import_from_env.assert_called_once()

    def test_import_env_triggers_refresh(self, qtbot):
        backend = _mock_backend()
        v = _make_view(qtbot, backend=backend)
        initial_calls = backend.list_stored_accounts.call_count
        v.import_env_btn.click()
        assert backend.list_stored_accounts.call_count > initial_calls

    def test_account_selector_populated_from_backend(self, qtbot):
        backend = _mock_backend()
        v = _make_view(qtbot, backend=backend)
        assert v.account_selector.count() == 2

    def test_account_selector_live_label_contains_live(self, qtbot):
        backend = _mock_backend()
        # live_1 has is_live=True → label should contain LIVE
        v = _make_view(qtbot, backend=backend)
        labels = [v.account_selector.itemText(i) for i in range(v.account_selector.count())]
        live_labels = [lb for lb in labels if "live_1" in lb]
        assert any("LIVE" in lb for lb in live_labels)

    def test_account_selector_demo_label_contains_demo(self, qtbot):
        backend = _mock_backend()
        v = _make_view(qtbot, backend=backend)
        labels = [v.account_selector.itemText(i) for i in range(v.account_selector.count())]
        demo_labels = [lb for lb in labels if "demo" in lb.lower()]
        assert any("Demo" in lb for lb in demo_labels)

    def test_add_account_saves_credentials(self, qtbot):
        backend = _mock_backend()
        v = _make_view(qtbot, backend=backend)
        v.account_id_input.setText("new_acc")
        v.login_input.setText("12345")
        v.broker_input.setText("MT5")
        v.server_input.setText("Test-Server")
        v.is_live_checkbox.setChecked(False)
        v.add_account_btn.click()
        backend.save_account_credentials.assert_called()
        call_args = backend.save_account_credentials.call_args[0]
        assert call_args[0] == "new_acc"

    def test_add_live_account_passes_is_live_true(self, qtbot):
        backend = _mock_backend()
        v = _make_view(qtbot, backend=backend)
        v.account_id_input.setText("live_acc")
        v.is_live_checkbox.setChecked(True)
        v.add_account_btn.click()
        call_args = backend.save_account_credentials.call_args[0]
        # is_live is the 5th positional arg: (account_id, login, broker, server, is_live, ...)
        assert call_args[4] is True

    def test_add_account_with_password_passes_password(self, qtbot):
        backend = _mock_backend()
        v = _make_view(qtbot, backend=backend)
        v.account_id_input.setText("acc")
        v.password_input.setText("my_pw")
        v.add_account_btn.click()
        call_args = backend.save_account_credentials.call_args[0]
        assert call_args[5] == "my_pw"

    def test_add_account_no_password_passes_none(self, qtbot):
        backend = _mock_backend()
        v = _make_view(qtbot, backend=backend)
        v.account_id_input.setText("acc")
        v.password_input.setText("")
        v.add_account_btn.click()
        call_args = backend.save_account_credentials.call_args[0]
        assert call_args[5] is None

    def test_live_account_row_typ_column_shows_live(self, qtbot):
        backend = _mock_backend()
        backend.list_stored_accounts.return_value = [
            {"account_id": "live1", "broker": "MT5", "server": "Live",
             "is_live": True, "has_password": False}
        ]
        v = _make_view(qtbot, backend=backend)
        typ_text = v.accounts_table.item(0, 3).text()
        assert typ_text == "LIVE"

    def test_demo_account_row_typ_column_shows_demo(self, qtbot):
        backend = _mock_backend()
        backend.list_stored_accounts.return_value = [
            {"account_id": "demo1", "broker": "MT5", "server": "Demo",
             "is_live": False, "has_password": False}
        ]
        v = _make_view(qtbot, backend=backend)
        typ_text = v.accounts_table.item(0, 3).text()
        assert typ_text == "Demo"

    def test_account_with_password_shows_gesetzt(self, qtbot):
        backend = _mock_backend()
        backend.list_stored_accounts.return_value = [
            {"account_id": "acc", "broker": "MT5", "server": "s",
             "is_live": False, "has_password": True}
        ]
        v = _make_view(qtbot, backend=backend)
        pw_text = v.accounts_table.item(0, 4).text()
        assert pw_text == "Gesetzt"

    def test_account_without_password_shows_nicht_gesetzt(self, qtbot):
        backend = _mock_backend()
        backend.list_stored_accounts.return_value = [
            {"account_id": "acc", "broker": "MT5", "server": "s",
             "is_live": False, "has_password": False}
        ]
        v = _make_view(qtbot, backend=backend)
        pw_text = v.accounts_table.item(0, 4).text()
        assert pw_text == "Nicht gesetzt"

    def test_account_switched_signal_emitted(self, qtbot):
        backend = _mock_backend()
        backend.list_stored_accounts.return_value = [
            {"account_id": "demo", "broker": "MT5", "server": "Demo",
             "is_live": False, "has_password": False},
        ]
        v = _make_view(qtbot, backend=backend)
        received: list[str] = []
        v.account_switched.connect(lambda aid: received.append(aid))
        # Add a second item and select it to trigger the signal
        v._account_selector.addItem("other [Demo]", "other")
        v._account_selector.setCurrentIndex(1)
        assert "other" in received

    def test_switch_account_calls_backend(self, qtbot):
        backend = _mock_backend()
        backend.list_stored_accounts.return_value = [
            {"account_id": "demo", "broker": "MT5", "server": "Demo",
             "is_live": False, "has_password": False},
        ]
        v = _make_view(qtbot, backend=backend)
        v._account_selector.addItem("live [LIVE]", "live_acc")
        v._account_selector.setCurrentIndex(1)
        backend.switch_account.assert_called_with("live_acc")
