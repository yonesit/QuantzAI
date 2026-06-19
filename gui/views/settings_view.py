"""
gui/views/settings_view.py
Zentrale Einstellungs-View fuer alle Bot-Parameter.

Tabs:
  0 – Risiko-Parameter  : Min/Max-Grenzen fuer Risiko-Werte
  1 – Trading-Modus     : Modus- und Live/Paper-Umschaltung (mehrstufige Bestätigung)
  2 – Konten            : Konto-Verwaltung ohne Klartext-Passwoerter
  3 – Symbole           : Symbol-Auswahl per Checkbox-Liste
  4 – Telegram          : Token (maskiert), Chat-ID, Test-Button
  5 – Audit-Log         : Aenderungshistorie der gespeicherten Einstellungen

Sicherheitsprinzipien:
  - Konto-Formulare enthalten keine Passwort-Felder (Passwoerter per .env)
  - Speichern erst nach Bestaetigung wirksam
  - AUTONOMOUS-Modus erfordert 2-stufige Bestaetigung
  - LIVE-Modus erfordert eigene Bestaetigung
  - Telegram-Token in Eingabefeld maskiert (Password-Echo)

Testbarkeit:
  _confirm_fn      : (title: str, message: str) -> bool  (ersetzt Dialoge)
  _test_telegram_fn: (token: str, chat_id: str) -> bool  (ersetzt HTTP-Aufruf)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Konstanten
# ─────────────────────────────────────────────────────────────────────────────

TAB_RISK     = 0
TAB_MODE     = 1
TAB_ACCOUNTS = 2
TAB_SYMBOLS  = 3
TAB_TELEGRAM = 4
TAB_AUDIT    = 5

_AVAILABLE_SYMBOLS: list[str] = [
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD",
    "XAUUSD", "US30", "NAS100", "SPX500", "BTCUSD",
]

_TRADING_MODE_OPTIONS: list[tuple[str, str]] = [
    ("Nur Vorschläge (SUGGEST_ONLY)",              "suggest_only"),
    ("Bestätigung erforderlich (CONFIRM_REQUIRED)", "confirm_required"),
    ("Autonom (AUTONOMOUS)",                        "autonomous"),
]

_DEFAULT_SETTINGS: dict[str, Any] = {
    "max_risk_per_trade_pct": 1.0,
    "max_daily_drawdown_pct": 5.0,
    "max_open_positions":     5,
    "max_lot_size":           10.0,
    "cooldown_after_loss_h":  0,
    "trading_mode":           "suggest_only",
    "paper_mode":             True,
    "symbols":                ["EURUSD", "GBPUSD"],
    "telegram_token":         "",
    "telegram_chat_id":       "",
}


# ─────────────────────────────────────────────────────────────────────────────
#  Backend-Protocol
# ─────────────────────────────────────────────────────────────────────────────

class SettingsBackend:
    """
    Protocol-aehnliches Interface fuer den Settings-Backend.

    Alle Methoden koennen als MagicMock injiziert werden (fuer Tests).
    """

    def get_settings(self) -> dict[str, Any]:
        return dict(_DEFAULT_SETTINGS)

    def save_settings(self, settings: dict[str, Any]) -> None:
        pass

    def get_accounts(self) -> list[dict[str, str]]:
        return []

    def add_account(self, account_id: str, broker: str, server: str) -> None:
        pass

    def remove_account(self, account_id: str) -> None:
        pass

    def test_telegram(self, token: str, chat_id: str) -> bool:
        return False


# ─────────────────────────────────────────────────────────────────────────────
#  Default-Hilfsfunktionen (werden in Tests ersetzt)
# ─────────────────────────────────────────────────────────────────────────────

def _default_confirm(title: str, message: str) -> bool:
    result = QMessageBox.question(
        None,
        title,
        message,
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
    )
    return result == QMessageBox.StandardButton.Yes


def _default_test_telegram(token: str, chat_id: str) -> bool:
    try:
        from src.monitoring.telegram_alerts import TelegramAlertSender  # type: ignore[import]
        sender = TelegramAlertSender(token=token, chat_id=chat_id)
        return sender.send_alert("QuantzAI: Test-Nachricht")
    except Exception:  # noqa: BLE001
        return False


# ─────────────────────────────────────────────────────────────────────────────
#  SettingsView
# ─────────────────────────────────────────────────────────────────────────────

class SettingsView(QWidget):
    """
    Zentrale Einstellungs-View.

    Signale
    -------
    settings_saved : emittiert nach erfolgreichem Speichern mit dem neuen settings-dict
    mode_changed   : emittiert wenn sich der Trading-Modus beim Speichern aendert

    Parameters
    ----------
    backend           : SettingsBackend-Instanz (optional).
    _confirm_fn       : (title, message) -> bool (ersetzt QMessageBox in Tests).
    _test_telegram_fn : (token, chat_id) -> bool (ersetzt HTTP-Aufruf in Tests).
    parent            : Eltern-Widget.
    """

    settings_saved = Signal(dict)
    mode_changed   = Signal(str)

    def __init__(
        self,
        backend:             Optional[SettingsBackend]           = None,
        _confirm_fn:         Optional[Callable[[str, str], bool]] = None,
        _test_telegram_fn:   Optional[Callable[[str, str], bool]] = None,
        parent:              Optional[QWidget]                    = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("settings_view")

        self._backend          = backend
        self._confirm_fn       = _confirm_fn  if _confirm_fn       is not None else _default_confirm
        self._test_telegram_fn = _test_telegram_fn if _test_telegram_fn is not None else _default_test_telegram

        self._saved_settings: dict[str, Any] = {}
        self._audit_entries:  list[dict]     = []
        self._loading:        bool           = False

        self._build()
        self._init_data()

    # ── Layout ───────────────────────────────────────────────────────────────

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 12)
        root.setSpacing(8)

        title = QLabel("⚙  Einstellungen")
        title.setObjectName("view_title")
        root.addWidget(title)

        self._tabs = QTabWidget()
        self._tabs.setObjectName("settings_tabs")
        self._tabs.addTab(self._build_risk_tab(),     "Risiko")
        self._tabs.addTab(self._build_mode_tab(),     "Modus")
        self._tabs.addTab(self._build_accounts_tab(), "Konten")
        self._tabs.addTab(self._build_symbols_tab(),  "Symbole")
        self._tabs.addTab(self._build_telegram_tab(), "Telegram")
        self._tabs.addTab(self._build_audit_tab(),    "Audit-Log")
        root.addWidget(self._tabs, stretch=1)

        root.addWidget(self._build_save_bar())

    def _build_risk_tab(self) -> QWidget:
        w = QWidget()
        w.setObjectName("risk_tab")
        form = QFormLayout(w)
        form.setContentsMargins(20, 20, 20, 20)
        form.setSpacing(12)

        self._max_risk_spin = QDoubleSpinBox()
        self._max_risk_spin.setObjectName("max_risk_spin")
        self._max_risk_spin.setRange(0.1, 5.0)
        self._max_risk_spin.setSingleStep(0.1)
        self._max_risk_spin.setDecimals(1)
        self._max_risk_spin.setSuffix(" %")
        self._max_risk_spin.setToolTip("Maximales Risiko pro Trade (0.1 – 5.0 %)")
        self._max_risk_spin.valueChanged.connect(self._on_widget_changed)
        form.addRow("Max. Risiko / Trade:", self._max_risk_spin)

        self._max_daily_dd_spin = QDoubleSpinBox()
        self._max_daily_dd_spin.setObjectName("max_daily_dd_spin")
        self._max_daily_dd_spin.setRange(1.0, 20.0)
        self._max_daily_dd_spin.setSingleStep(0.5)
        self._max_daily_dd_spin.setDecimals(1)
        self._max_daily_dd_spin.setSuffix(" %")
        self._max_daily_dd_spin.setToolTip("Maximaler Tages-Drawdown (1.0 – 20.0 %)")
        self._max_daily_dd_spin.valueChanged.connect(self._on_widget_changed)
        form.addRow("Max. Tages-Drawdown:", self._max_daily_dd_spin)

        self._max_positions_spin = QSpinBox()
        self._max_positions_spin.setObjectName("max_positions_spin")
        self._max_positions_spin.setRange(1, 20)
        self._max_positions_spin.setToolTip("Maximale Anzahl gleichzeitig offener Positionen (1 – 20)")
        self._max_positions_spin.valueChanged.connect(self._on_widget_changed)
        form.addRow("Max. offene Positionen:", self._max_positions_spin)

        self._max_lot_spin = QDoubleSpinBox()
        self._max_lot_spin.setObjectName("max_lot_spin")
        self._max_lot_spin.setRange(0.01, 100.0)
        self._max_lot_spin.setSingleStep(0.01)
        self._max_lot_spin.setDecimals(2)
        self._max_lot_spin.setToolTip("Maximale Lot-Groesse pro Trade (0.01 – 100.0)")
        self._max_lot_spin.valueChanged.connect(self._on_widget_changed)
        form.addRow("Max. Lot-Groesse:", self._max_lot_spin)

        self._cooldown_spin = QSpinBox()
        self._cooldown_spin.setObjectName("cooldown_spin")
        self._cooldown_spin.setRange(0, 24)
        self._cooldown_spin.setSuffix(" h")
        self._cooldown_spin.setToolTip("Pause nach Verlust-Trade (0 – 24 Stunden)")
        self._cooldown_spin.valueChanged.connect(self._on_widget_changed)
        form.addRow("Cooldown nach Verlust:", self._cooldown_spin)

        return w

    def _build_mode_tab(self) -> QWidget:
        w = QWidget()
        w.setObjectName("mode_tab")
        layout = QVBoxLayout(w)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        mode_group = QGroupBox("Trading-Modus")
        mode_form  = QFormLayout(mode_group)
        mode_form.setSpacing(10)

        self._mode_combo = QComboBox()
        self._mode_combo.setObjectName("mode_combo")
        for label, data in _TRADING_MODE_OPTIONS:
            self._mode_combo.addItem(label, data)
        self._mode_combo.setToolTip(
            "AUTONOMOUS erfordert 2-stufige Bestätigung und Umgebungsvariable CONFIRM_AUTONOMOUS=yes"
        )
        self._mode_combo.currentIndexChanged.connect(self._on_widget_changed)
        mode_form.addRow("Betriebsmodus:", self._mode_combo)

        mode_hint = QLabel(
            "⚠  AUTONOMOUS: Bot handelt vollständig selbstständig.\n"
            "   Erfordert CONFIRM_AUTONOMOUS=yes in .env."
        )
        mode_hint.setObjectName("mode_hint")
        mode_hint.setWordWrap(True)
        mode_form.addRow(mode_hint)
        layout.addWidget(mode_group)

        paper_group = QGroupBox("Handelsmodus")
        paper_layout = QVBoxLayout(paper_group)
        paper_layout.setSpacing(8)

        self._paper_radio = QRadioButton("Paper-Trading (simuliert, kein echtes Geld)")
        self._paper_radio.setObjectName("paper_radio")
        self._paper_radio.setChecked(True)

        self._live_radio = QRadioButton("Live-Trading (echtes Geld!)")
        self._live_radio.setObjectName("live_radio")

        self._paper_radio.toggled.connect(self._on_widget_changed)
        paper_layout.addWidget(self._paper_radio)
        paper_layout.addWidget(self._live_radio)

        live_warn = QLabel("⚠  Live-Modus: Alle Trades werden mit echtem Geld ausgefuehrt!")
        live_warn.setObjectName("live_warn")
        live_warn.setWordWrap(True)
        paper_layout.addWidget(live_warn)
        layout.addWidget(paper_group)

        layout.addStretch()
        return w

    def _build_accounts_tab(self) -> QWidget:
        w = QWidget()
        w.setObjectName("accounts_tab")
        layout = QVBoxLayout(w)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        info = QLabel(
            "Konten-Verwaltung: Passwoerter werden NICHT gespeichert.\n"
            "Login-Daten ausschliesslich ueber .env-Datei konfigurieren."
        )
        info.setObjectName("accounts_info")
        info.setWordWrap(True)
        layout.addWidget(info)

        self._accounts_table = QTableWidget(0, 3)
        self._accounts_table.setObjectName("accounts_table")
        self._accounts_table.setHorizontalHeaderLabels(["Konto-ID", "Broker", "Server"])
        self._accounts_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self._accounts_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._accounts_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self._accounts_table)

        form_group = QGroupBox("Konto hinzufügen")
        form_layout = QFormLayout(form_group)
        form_layout.setSpacing(8)

        self._account_id_input = QLineEdit()
        self._account_id_input.setObjectName("account_id_input")
        self._account_id_input.setPlaceholderText("z.B. demo, live_1")
        form_layout.addRow("Konto-ID:", self._account_id_input)

        self._broker_input = QLineEdit()
        self._broker_input.setObjectName("broker_input")
        self._broker_input.setPlaceholderText("z.B. MT5, OANDA")
        form_layout.addRow("Broker:", self._broker_input)

        self._server_input = QLineEdit()
        self._server_input.setObjectName("server_input")
        self._server_input.setPlaceholderText("z.B. ICMarkets-Demo")
        form_layout.addRow("Server:", self._server_input)

        btn_row = QHBoxLayout()
        self._add_account_btn = QPushButton("Hinzufügen")
        self._add_account_btn.setObjectName("add_account_btn")
        self._add_account_btn.clicked.connect(self._on_add_account_clicked)

        self._remove_account_btn = QPushButton("Entfernen")
        self._remove_account_btn.setObjectName("remove_account_btn")
        self._remove_account_btn.setProperty("danger", "true")
        self._remove_account_btn.clicked.connect(self._on_remove_account_clicked)

        btn_row.addWidget(self._add_account_btn)
        btn_row.addWidget(self._remove_account_btn)
        btn_row.addStretch()
        form_layout.addRow(btn_row)
        layout.addWidget(form_group)

        return w

    def _build_symbols_tab(self) -> QWidget:
        w = QWidget()
        w.setObjectName("symbols_tab")
        layout = QVBoxLayout(w)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)

        layout.addWidget(QLabel("Aktive Handelssymbole auswählen:"))

        self._symbols_list = QListWidget()
        self._symbols_list.setObjectName("symbols_list")
        for sym in _AVAILABLE_SYMBOLS:
            item = QListWidgetItem(sym)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            self._symbols_list.addItem(item)
        self._symbols_list.itemChanged.connect(self._on_widget_changed)
        layout.addWidget(self._symbols_list)

        return w

    def _build_telegram_tab(self) -> QWidget:
        w = QWidget()
        w.setObjectName("telegram_tab")
        layout = QVBoxLayout(w)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        form_group = QGroupBox("Telegram-Konfiguration")
        form_layout = QFormLayout(form_group)
        form_layout.setSpacing(10)

        self._token_input = QLineEdit()
        self._token_input.setObjectName("token_input")
        self._token_input.setEchoMode(QLineEdit.EchoMode.Password)
        self._token_input.setPlaceholderText("Bot-Token (aus BotFather)")
        self._token_input.textChanged.connect(self._on_widget_changed)
        form_layout.addRow("Bot-Token:", self._token_input)

        self._chat_id_input = QLineEdit()
        self._chat_id_input.setObjectName("chat_id_input")
        self._chat_id_input.setPlaceholderText("Chat-ID oder Kanal-ID")
        self._chat_id_input.textChanged.connect(self._on_widget_changed)
        form_layout.addRow("Chat-ID:", self._chat_id_input)

        test_row = QHBoxLayout()
        self._test_btn = QPushButton("Verbindung testen")
        self._test_btn.setObjectName("test_btn")
        self._test_btn.clicked.connect(self._on_test_telegram_clicked)

        self._test_result_label = QLabel("")
        self._test_result_label.setObjectName("test_result_label")

        test_row.addWidget(self._test_btn)
        test_row.addWidget(self._test_result_label, stretch=1)
        form_layout.addRow(test_row)

        layout.addWidget(form_group)
        layout.addStretch()
        return w

    def _build_audit_tab(self) -> QWidget:
        w = QWidget()
        w.setObjectName("audit_tab")
        layout = QVBoxLayout(w)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        layout.addWidget(QLabel("Aenderungshistorie (aktuelle Sitzung):"))

        self._audit_table = QTableWidget(0, 4)
        self._audit_table.setObjectName("audit_table")
        self._audit_table.setHorizontalHeaderLabels(
            ["Zeitstempel", "Parameter", "Alter Wert", "Neuer Wert"]
        )
        self._audit_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self._audit_table.horizontalHeader().setStretchLastSection(True)
        self._audit_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._audit_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        layout.addWidget(self._audit_table)

        return w

    def _build_save_bar(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("save_bar")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(8)

        self._pending_label = QLabel("Keine Änderungen")
        self._pending_label.setObjectName("pending_label")
        layout.addWidget(self._pending_label, stretch=1)

        self._discard_btn = QPushButton("Verwerfen")
        self._discard_btn.setObjectName("discard_btn")
        self._discard_btn.setProperty("secondary", "true")
        self._discard_btn.setEnabled(False)
        self._discard_btn.clicked.connect(self._on_discard_clicked)
        layout.addWidget(self._discard_btn)

        self._save_btn = QPushButton("Einstellungen speichern")
        self._save_btn.setObjectName("save_btn")
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self._on_save_clicked)
        layout.addWidget(self._save_btn)

        return bar

    # ── Daten laden ──────────────────────────────────────────────────────────

    def _init_data(self) -> None:
        settings = (
            self._backend.get_settings()
            if self._backend is not None
            else dict(_DEFAULT_SETTINGS)
        )
        self._saved_settings = dict(settings)
        self._load_settings(settings)
        self._refresh_accounts()
        self._update_pending_ui()

    def _load_settings(self, settings: dict[str, Any]) -> None:
        """Lädt settings-Dict in alle Widgets. Verhindert spurious-dirty via _loading."""
        self._loading = True

        self._max_risk_spin.setValue(settings.get("max_risk_per_trade_pct", 1.0))
        self._max_daily_dd_spin.setValue(settings.get("max_daily_drawdown_pct", 5.0))
        self._max_positions_spin.setValue(settings.get("max_open_positions", 5))
        self._max_lot_spin.setValue(settings.get("max_lot_size", 10.0))
        self._cooldown_spin.setValue(settings.get("cooldown_after_loss_h", 0))

        mode = settings.get("trading_mode", "suggest_only")
        for i in range(self._mode_combo.count()):
            if self._mode_combo.itemData(i) == mode:
                self._mode_combo.setCurrentIndex(i)
                break

        paper = settings.get("paper_mode", True)
        self._paper_radio.setChecked(paper)
        self._live_radio.setChecked(not paper)

        checked = set(settings.get("symbols", []))
        for i in range(self._symbols_list.count()):
            item = self._symbols_list.item(i)
            item.setCheckState(
                Qt.CheckState.Checked
                if item.text() in checked
                else Qt.CheckState.Unchecked
            )

        self._token_input.setText(settings.get("telegram_token", ""))
        self._chat_id_input.setText(settings.get("telegram_chat_id", ""))

        self._loading = False

    def _refresh_accounts(self) -> None:
        self._accounts_table.setRowCount(0)
        if self._backend is None:
            return
        for acc in self._backend.get_accounts():
            self._add_account_row(
                acc.get("account_id", ""),
                acc.get("broker", ""),
                acc.get("server", ""),
            )

    # ── Aktuellen Zustand sammeln ─────────────────────────────────────────────

    def _collect_current(self) -> dict[str, Any]:
        checked_syms = []
        for i in range(self._symbols_list.count()):
            item = self._symbols_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                checked_syms.append(item.text())
        return {
            "max_risk_per_trade_pct": self._max_risk_spin.value(),
            "max_daily_drawdown_pct": self._max_daily_dd_spin.value(),
            "max_open_positions":     self._max_positions_spin.value(),
            "max_lot_size":           self._max_lot_spin.value(),
            "cooldown_after_loss_h":  self._cooldown_spin.value(),
            "trading_mode":           self._mode_combo.currentData(),
            "paper_mode":             self._paper_radio.isChecked(),
            "symbols":                checked_syms,
            "telegram_token":         self._token_input.text(),
            "telegram_chat_id":       self._chat_id_input.text(),
        }

    def _is_dirty(self) -> bool:
        return self._collect_current() != self._saved_settings

    def _pending_count(self) -> int:
        current = self._collect_current()
        return sum(1 for k, v in current.items() if v != self._saved_settings.get(k))

    # ── Widget-Änderungs-Handler ──────────────────────────────────────────────

    def _on_widget_changed(self) -> None:
        if self._loading:
            return
        self._update_pending_ui()

    def _update_pending_ui(self) -> None:
        dirty = self._is_dirty()
        n     = self._pending_count()
        self._save_btn.setEnabled(dirty)
        self._discard_btn.setEnabled(dirty)
        if dirty:
            self._pending_label.setText(f"{n} Änderung(en) ausstehend")
        else:
            self._pending_label.setText("Keine Änderungen")

    # ── Speichern / Verwerfen ─────────────────────────────────────────────────

    def _on_save_clicked(self) -> None:
        current   = self._collect_current()
        old_mode  = self._saved_settings.get("trading_mode")
        new_mode  = current["trading_mode"]
        old_paper = self._saved_settings.get("paper_mode", True)
        new_paper = current["paper_mode"]

        # 1. Mehrstufige Bestaetigung fuer AUTONOMOUS
        if new_mode == "autonomous" and old_mode != "autonomous":
            if not self._confirm_fn(
                "Modus-Wechsel: AUTONOMOUS",
                "Wechsel in den AUTONOMOUS-Modus?\n\n"
                "Der Bot agiert vollständig selbstständig ohne Rückfrage.",
            ):
                return
            if not self._confirm_fn(
                "Letzte Bestätigung: AUTONOMOUS",
                "VORSICHT: Bot agiert völlig autonom. Alle Signale werden sofort ausgefuehrt.\n\n"
                "Umgebungsvariable CONFIRM_AUTONOMOUS=yes muss gesetzt sein.\n\n"
                "Wirklich aktivieren?",
            ):
                return

        # 2. Bestaetigung fuer LIVE-Modus
        if not new_paper and old_paper:
            if not self._confirm_fn(
                "Live-Modus aktivieren",
                "VORSICHT: Live-Modus verwendet echtes Geld!\n\n"
                "Alle Trades werden mit realem Kapital ausgefuehrt.\n"
                "Wirklich in den Live-Modus wechseln?",
            ):
                return

        # 3. Allgemeine Speicher-Bestaetigung
        if not self._confirm_fn(
            "Einstellungen speichern",
            f"{'1 Änderung' if self._pending_count() == 1 else f'{self._pending_count()} Änderungen'} "
            "jetzt speichern?",
        ):
            return

        # Audit-Eintraege anlegen
        for key, new_val in current.items():
            old_val = self._saved_settings.get(key)
            if new_val != old_val:
                entry: dict = {
                    "ts":        datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "parameter": key,
                    "old_value": str(old_val),
                    "new_value": str(new_val),
                }
                self._audit_entries.append(entry)
                self._add_audit_row_data(entry)

        # Backend-Call
        if self._backend is not None:
            self._backend.save_settings(current)

        mode_before = self._saved_settings.get("trading_mode")
        self._saved_settings = dict(current)
        self._update_pending_ui()
        self.settings_saved.emit(current)
        if mode_before != new_mode:
            self.mode_changed.emit(new_mode)

    def _on_discard_clicked(self) -> None:
        self._load_settings(self._saved_settings)
        self._update_pending_ui()

    # ── Konten ───────────────────────────────────────────────────────────────

    def _on_add_account_clicked(self) -> None:
        account_id = self._account_id_input.text().strip()
        broker     = self._broker_input.text().strip()
        server     = self._server_input.text().strip()
        if not account_id:
            return
        if self._backend is not None:
            try:
                self._backend.add_account(account_id, broker, server)
            except Exception:  # noqa: BLE001
                pass
        self._add_account_row(account_id, broker, server)
        self._account_id_input.clear()
        self._broker_input.clear()
        self._server_input.clear()

    def _on_remove_account_clicked(self) -> None:
        row = self._accounts_table.currentRow()
        if row < 0:
            return
        item = self._accounts_table.item(row, 0)
        account_id = item.text() if item else ""
        if not self._confirm_fn(
            "Konto entfernen",
            f"Konto '{account_id}' wirklich entfernen?",
        ):
            return
        if self._backend is not None:
            try:
                self._backend.remove_account(account_id)
            except Exception:  # noqa: BLE001
                pass
        self._accounts_table.removeRow(row)

    def _add_account_row(self, account_id: str, broker: str, server: str) -> None:
        row = self._accounts_table.rowCount()
        self._accounts_table.insertRow(row)
        self._accounts_table.setItem(row, 0, QTableWidgetItem(account_id))
        self._accounts_table.setItem(row, 1, QTableWidgetItem(broker))
        self._accounts_table.setItem(row, 2, QTableWidgetItem(server))

    # ── Telegram ─────────────────────────────────────────────────────────────

    def _on_test_telegram_clicked(self) -> None:
        token   = self._token_input.text().strip()
        chat_id = self._chat_id_input.text().strip()
        if not token or not chat_id:
            self._test_result_label.setText("Bitte Token und Chat-ID eingeben.")
            return
        ok = self._test_telegram_fn(token, chat_id)
        if ok:
            self._test_result_label.setText("✓  Verbindung erfolgreich!")
        else:
            self._test_result_label.setText("✗  Verbindung fehlgeschlagen.")

    # ── Audit ─────────────────────────────────────────────────────────────────

    def _add_audit_row_data(self, entry: dict) -> None:
        row = self._audit_table.rowCount()
        self._audit_table.insertRow(row)
        self._audit_table.setItem(row, 0, QTableWidgetItem(entry["ts"]))
        self._audit_table.setItem(row, 1, QTableWidgetItem(entry["parameter"]))
        self._audit_table.setItem(row, 2, QTableWidgetItem(entry["old_value"]))
        self._audit_table.setItem(row, 3, QTableWidgetItem(entry["new_value"]))

    # ── Oeffentliche Properties fuer Tests ───────────────────────────────────

    @property
    def tabs(self) -> QTabWidget:
        return self._tabs

    @property
    def max_risk_spin(self) -> QDoubleSpinBox:
        return self._max_risk_spin

    @property
    def max_daily_dd_spin(self) -> QDoubleSpinBox:
        return self._max_daily_dd_spin

    @property
    def max_positions_spin(self) -> QSpinBox:
        return self._max_positions_spin

    @property
    def max_lot_spin(self) -> QDoubleSpinBox:
        return self._max_lot_spin

    @property
    def cooldown_spin(self) -> QSpinBox:
        return self._cooldown_spin

    @property
    def mode_combo(self) -> QComboBox:
        return self._mode_combo

    @property
    def paper_radio(self) -> QRadioButton:
        return self._paper_radio

    @property
    def live_radio(self) -> QRadioButton:
        return self._live_radio

    @property
    def accounts_table(self) -> QTableWidget:
        return self._accounts_table

    @property
    def account_id_input(self) -> QLineEdit:
        return self._account_id_input

    @property
    def broker_input(self) -> QLineEdit:
        return self._broker_input

    @property
    def server_input(self) -> QLineEdit:
        return self._server_input

    @property
    def add_account_btn(self) -> QPushButton:
        return self._add_account_btn

    @property
    def remove_account_btn(self) -> QPushButton:
        return self._remove_account_btn

    @property
    def symbols_list(self) -> QListWidget:
        return self._symbols_list

    @property
    def token_input(self) -> QLineEdit:
        return self._token_input

    @property
    def chat_id_input(self) -> QLineEdit:
        return self._chat_id_input

    @property
    def test_btn(self) -> QPushButton:
        return self._test_btn

    @property
    def test_result_label(self) -> QLabel:
        return self._test_result_label

    @property
    def save_btn(self) -> QPushButton:
        return self._save_btn

    @property
    def discard_btn(self) -> QPushButton:
        return self._discard_btn

    @property
    def pending_label(self) -> QLabel:
        return self._pending_label

    @property
    def audit_table(self) -> QTableWidget:
        return self._audit_table

    @property
    def audit_entries(self) -> list[dict]:
        return list(self._audit_entries)

    @property
    def saved_settings(self) -> dict[str, Any]:
        return dict(self._saved_settings)
