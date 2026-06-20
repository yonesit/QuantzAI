"""
gui/app.py
QuantzAI Desktop-Anwendung – Hauptfenster und Grundgeruest.

Architektur-Prinzip: strikte Trennung von UI und Logik.
  - Kein Geschaeftslogik-Code in Widget-Klassen
  - Views rufen Backend-Module (TradingOrchestrator etc.) ueber oeffentliche APIs auf
  - MainWindow kennt nur Navigation und Statusweiterleitung

HCI-Prinzipien:
  - Sichtbarkeit des Systemstatus: TradingStatusBar zeigt immer Bot-Zustand
  - Fehlervermeidung: ConfirmationDialog fuer jede irreversible Aktion
  - Konsistenz: alle Views nutzen die gleichen Komponenten aus diesem Modul
  - Wiedererkennung: Sidebar ist immer sichtbar, Navigation immer zugaenglich
"""

from __future__ import annotations

import os
import sys
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Optional

from loguru import logger

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from gui.design.theme import ThemeManager, ThemeMode, get_theme_manager
from gui.views.backtest_view import BacktestBackend, BacktestView
from gui.views.cockpit_view import CockpitBackend, CockpitView
from gui.backends.backtest_backend import BacktestGUIBackend
from gui.views.dashboard_view import DashboardBackend, DashboardSnapshot, DashboardView
from gui.views.journal_view import JournalBackend, JournalView
from gui.views.risk_center_view import RiskCenterBackend, RiskCenterView
from gui.views.settings_view import SettingsBackend, SettingsView


# ─────────────────────────────────────────────────────────────────────────────
#  Domaenen-Enums (UI-Repr. der Bot-Zustaende)
# ─────────────────────────────────────────────────────────────────────────────

class Section(Enum):
    """Navigations-Sektionen der App."""
    DASHBOARD = ("Dashboard",      "📊", 0)
    COCKPIT   = ("Cockpit",        "🎮", 1)
    RISK      = ("Risiko",         "🛡",  2)
    JOURNAL   = ("Journal",        "📓", 3)
    BACKTEST  = ("Backtest",       "📈", 4)
    SETTINGS  = ("Einstellungen",  "⚙",  5)

    def __init__(self, label: str, icon: str, index: int) -> None:
        self.label = label
        self.icon  = icon
        self.index = index


class TradingMode(Enum):
    SUGGEST    = "Vorschlag"
    CONFIRM    = "Bestätigung"
    AUTONOMOUS = "Autonom"


class ConnectionStatus(Enum):
    CONNECTED    = "Verbunden"
    DISCONNECTED = "Getrennt"
    ERROR        = "Fehler"


# ─────────────────────────────────────────────────────────────────────────────
#  Placeholder Views
# ─────────────────────────────────────────────────────────────────────────────

class _PlaceholderView(QWidget):
    """Basis-Platzhalter bis Views in separaten Issues implementiert werden."""

    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("placeholder_view")
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(8)

        heading = QLabel(title)
        heading.setAlignment(Qt.AlignmentFlag.AlignCenter)
        heading.setObjectName("placeholder_title")
        f = heading.font()
        f.setPointSize(18)
        f.setBold(True)
        heading.setFont(f)

        sub = QLabel("Coming soon")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sub.setObjectName("placeholder_sub")
        sub.setProperty("secondary", "true")

        layout.addWidget(heading)
        layout.addWidget(sub)


class _CockpitPlaceholderView(_PlaceholderView):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("🎮  Cockpit", parent)


class _RiskPlaceholderView(_PlaceholderView):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("🛡  Risiko", parent)


class _JournalPlaceholderView(_PlaceholderView):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("📓  Journal", parent)


class _BacktestPlaceholderView(_PlaceholderView):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("📈  Backtest", parent)


class _SettingsPlaceholderView(_PlaceholderView):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("⚙  Einstellungen", parent)


_PLACEHOLDER_VIEWS: dict[Section, type[_PlaceholderView]] = {}


# ─────────────────────────────────────────────────────────────────────────────
#  NavigationSidebar
# ─────────────────────────────────────────────────────────────────────────────

class NavigationSidebar(QWidget):
    """
    Linke Sidebar mit Icon + Text fuer jede Sektion.
    Sendet `section_changed` wenn der Nutzer navigiert.
    """

    section_changed = Signal(Section)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("sidebar")
        self.setFixedWidth(190)

        self._current: Section = Section.DASHBOARD
        self._buttons: dict[Section, QPushButton] = {}
        self._build()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 18, 10, 18)
        layout.setSpacing(2)

        logo = QLabel("QuantzAI")
        logo.setObjectName("sidebar_logo")
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        f = logo.font()
        f.setPointSize(14)
        f.setBold(True)
        logo.setFont(f)
        layout.addWidget(logo)
        layout.addSpacing(18)

        for section in Section:
            btn = QPushButton(f"  {section.icon}  {section.label}")
            btn.setObjectName(f"nav_{section.name.lower()}")
            btn.setProperty("nav", "true")
            btn.setCheckable(True)
            btn.setMinimumHeight(38)
            btn.clicked.connect(lambda _checked, s=section: self._on_clicked(s))
            self._buttons[section] = btn
            layout.addWidget(btn)

        layout.addStretch()

        # Dashboard ist Standard
        self._buttons[Section.DASHBOARD].setChecked(True)

    def _on_clicked(self, section: Section) -> None:
        if section is self._current:
            # Sicherstellen dass Button gecheckt bleibt wenn erneut geklickt
            self._buttons[section].setChecked(True)
            return
        self._buttons[self._current].setChecked(False)
        self._current = section
        self._buttons[section].setChecked(True)
        self.section_changed.emit(section)

    @property
    def current_section(self) -> Section:
        return self._current

    def navigate_to(self, section: Section) -> None:
        """Programmatisch zu einer Sektion navigieren (emittiert section_changed)."""
        if section is self._current:
            return
        self._buttons[self._current].setChecked(False)
        self._current = section
        self._buttons[section].setChecked(True)
        self.section_changed.emit(section)

    def button(self, section: Section) -> QPushButton:
        """Gibt den Nav-Button fuer eine Sektion zurueck (fuer Tests)."""
        return self._buttons[section]


# ─────────────────────────────────────────────────────────────────────────────
#  TradingStatusBar
# ─────────────────────────────────────────────────────────────────────────────

class TradingStatusBar(QWidget):
    """
    Permanent sichtbare Statusleiste (HCI: Sichtbarkeit des Systemstatus).

    Zeigt jederzeit:
      - Verbindungsstatus MT5/OANDA
      - Aktueller Trading-Modus (Suggest / Confirm / Autonomous)
      - Ob der Bot pausiert ist

    Der Nutzer darf NIE im Unklaren sein, ob der Bot aktiv handelt.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("trading_status_bar")

        self._connection   = ConnectionStatus.DISCONNECTED
        self._mode         = TradingMode.SUGGEST
        self._paused       = True
        self._account_info: Optional[dict] = None

        self._build()
        self._refresh()

    def _build(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 0, 12, 0)
        layout.setSpacing(16)

        self._conn_dot    = QLabel()
        self._conn_label  = QLabel()
        self._account_lbl = QLabel()
        self._mode_label  = QLabel()
        self._bot_label   = QLabel()

        for lbl in (self._conn_dot, self._conn_label,
                    self._account_lbl, self._mode_label, self._bot_label):
            lbl.setObjectName("status_label")

        self._account_lbl.setVisible(False)

        layout.addWidget(self._conn_dot)
        layout.addWidget(self._conn_label)
        layout.addWidget(self._account_lbl)
        layout.addWidget(_vline())
        layout.addWidget(self._mode_label)
        layout.addWidget(_vline())
        layout.addWidget(self._bot_label)
        layout.addStretch()

    def _refresh(self) -> None:
        color_map = {
            ConnectionStatus.CONNECTED:    "#22c55e",
            ConnectionStatus.DISCONNECTED: "#6b7280",
            ConnectionStatus.ERROR:        "#ef4444",
        }
        dot_color = color_map[self._connection]
        self._conn_dot.setText("●")
        self._conn_dot.setStyleSheet(f"color: {dot_color}; font-size: 9pt;")
        self._conn_label.setText(f"MT5/OANDA: {self._connection.value}")

        if self._account_info is not None:
            info    = self._account_info
            login   = info.get("login") or "?"
            balance = info.get("balance")
            curr    = info.get("currency", "")
            is_demo = info.get("is_demo")
            bal_str = f"{curr}{balance:,.2f}" if balance is not None else "--"
            tag     = "Demo" if is_demo is True else ("Live" if is_demo is False else "")
            demo_part = f" {tag}" if tag else ""
            self._account_lbl.setText(f"#{login}{demo_part} | {bal_str}")
            self._account_lbl.setVisible(True)
        else:
            self._account_lbl.setVisible(False)

        self._mode_label.setText(f"Modus: {self._mode.value}")

        if self._paused:
            self._bot_label.setText("⏸  Bot pausiert")
            self._bot_label.setStyleSheet("color: #f59e0b;")
        else:
            self._bot_label.setText("▶  Bot aktiv")
            self._bot_label.setStyleSheet("color: #22c55e; font-weight: bold;")

    # ── Setter (sauber ohne direkten Widget-Zugriff von aussen) ──────────────

    def set_connection(self, status: ConnectionStatus) -> None:
        self._connection = status
        self._refresh()

    def set_account_info(self, info: Optional[dict]) -> None:
        self._account_info = info
        self._refresh()

    def set_trading_mode(self, mode: TradingMode) -> None:
        self._mode = mode
        self._refresh()

    def set_paused(self, paused: bool) -> None:
        self._paused = paused
        self._refresh()

    # ── Getter fuer Tests ─────────────────────────────────────────────────────

    @property
    def connection_status(self) -> ConnectionStatus:
        return self._connection

    @property
    def trading_mode(self) -> TradingMode:
        return self._mode

    @property
    def is_paused(self) -> bool:
        return self._paused

    @property
    def connection_label(self) -> QLabel:
        return self._conn_label

    @property
    def mode_label(self) -> QLabel:
        return self._mode_label

    @property
    def bot_label(self) -> QLabel:
        return self._bot_label

    @property
    def account_label(self) -> QLabel:
        return self._account_lbl

    @property
    def account_info(self) -> Optional[dict]:
        return self._account_info


def _vline() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.Shape.VLine)
    f.setObjectName("status_sep")
    return f


# ─────────────────────────────────────────────────────────────────────────────
#  ConfirmationDialog
# ─────────────────────────────────────────────────────────────────────────────

class ConfirmationDialog(QDialog):
    """
    Bestaetigung fuer irreversible Aktionen (HCI: Fehlervermeidung).

    Beschreibt in Worten die Konsequenz – kein blosses OK-Klicken ohne Kontext.

    Beispiele:
      - Live-Modus aktivieren
      - Position schliessen
      - Notfall-Stop ausloesen
    """

    def __init__(
        self,
        title: str,
        message: str,
        confirm_label: str = "Bestätigen",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(420)
        self.setObjectName("confirmation_dialog")

        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(28, 28, 28, 28)

        # Icon
        icon_lbl = QLabel("⚠")
        icon_lbl.setObjectName("dialog_icon")
        f = icon_lbl.font()
        f.setPointSize(26)
        icon_lbl.setFont(f)
        icon_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(icon_lbl)

        # Nachricht
        msg_lbl = QLabel(message)
        msg_lbl.setObjectName("dialog_message")
        msg_lbl.setWordWrap(True)
        msg_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(msg_lbl)

        # Buttons
        btn_box = QDialogButtonBox()
        cancel_btn  = QPushButton("Abbrechen")
        cancel_btn.setProperty("secondary", "true")
        confirm_btn = QPushButton(confirm_label)
        confirm_btn.setProperty("danger", "true")
        confirm_btn.setObjectName("dialog_confirm_btn")
        btn_box.addButton(cancel_btn,  QDialogButtonBox.ButtonRole.RejectRole)
        btn_box.addButton(confirm_btn, QDialogButtonBox.ButtonRole.AcceptRole)
        btn_box.rejected.connect(self.reject)
        btn_box.accepted.connect(self.accept)
        layout.addWidget(btn_box)

    @staticmethod
    def ask(
        title: str,
        message: str,
        confirm_label: str = "Bestätigen",
        parent: Optional[QWidget] = None,
    ) -> bool:
        """Zeigt Dialog und gibt True zurueck wenn der Nutzer bestaetigt hat."""
        dlg = ConfirmationDialog(title, message, confirm_label, parent)
        return dlg.exec() == QDialog.DialogCode.Accepted


# ─────────────────────────────────────────────────────────────────────────────
#  MainWindow
# ─────────────────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    """
    Hauptfenster der QuantzAI Desktop-Anwendung.

    Trennung UI / Logik:
      - Keine Geschaeftslogik hier
      - Views kommunizieren ueber ihre eigenen Signale mit dem Backend
      - MainWindow verwaltet nur Navigation und Statusanzeige

    Parameters
    ----------
    theme_manager      : ThemeManager-Instanz. Standard: globale Instanz.
                         Fuer Tests frische Instanz uebergeben.
    dashboard_backend  : Backend fuer den Dashboard-View (Optional).
                         Wenn None, zeigt Dashboard Platzhalterwerte.
    """

    def __init__(
        self,
        theme_manager:        Optional[ThemeManager]        = None,
        dashboard_backend:    Optional[DashboardBackend]    = None,
        cockpit_backend:      Optional[CockpitBackend]      = None,
        risk_center_backend:  Optional[RiskCenterBackend]   = None,
        journal_backend:      Optional[JournalBackend]      = None,
        backtest_backend:     Optional[BacktestBackend]     = None,
        settings_backend:     Optional[SettingsBackend]     = None,
        parent:               Optional[QWidget]             = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("QuantzAI")
        self.setMinimumSize(1366, 768)

        self._theme               = theme_manager if theme_manager is not None else get_theme_manager()
        self._dashboard_backend   = dashboard_backend
        self._cockpit_backend     = cockpit_backend
        self._risk_center_backend = risk_center_backend
        self._journal_backend     = journal_backend
        self._backtest_backend    = backtest_backend
        self._settings_backend    = settings_backend
        self._theme.on_theme_changed(self.setStyleSheet)

        self._build()
        self.setStyleSheet(self._theme.stylesheet())

    def _build(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)

        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Sidebar
        self._sidebar = NavigationSidebar()
        self._sidebar.section_changed.connect(self._on_section_changed)
        root.addWidget(self._sidebar)

        # Trennlinie Sidebar / Content
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setProperty("sidebar_sep", "true")
        root.addWidget(sep)

        # Content-Bereich
        self._content = QStackedWidget()
        self._views: dict[Section, QWidget] = {}

        # Dashboard: echter View mit optionalem Backend
        self._dashboard_view = DashboardView(backend=self._dashboard_backend)
        self._views[Section.DASHBOARD] = self._dashboard_view
        self._content.addWidget(self._dashboard_view)

        # Cockpit: echter View mit optionalem Backend
        self._cockpit_view = CockpitView(backend=self._cockpit_backend)
        self._views[Section.COCKPIT] = self._cockpit_view
        self._content.addWidget(self._cockpit_view)

        # Risiko-Zentrale: echter View mit optionalem Backend
        self._risk_center_view = RiskCenterView(backend=self._risk_center_backend)
        self._views[Section.RISK] = self._risk_center_view
        self._content.addWidget(self._risk_center_view)

        # Journal: echter View mit optionalem Backend
        self._journal_view = JournalView(backend=self._journal_backend)
        self._views[Section.JOURNAL] = self._journal_view
        self._content.addWidget(self._journal_view)

        # Backtest: echter View mit optionalem Backend
        self._backtest_view = BacktestView(backend=self._backtest_backend)
        self._views[Section.BACKTEST] = self._backtest_view
        self._content.addWidget(self._backtest_view)

        # Einstellungen: echter View mit optionalem Backend
        self._settings_view = SettingsView(backend=self._settings_backend)
        self._views[Section.SETTINGS] = self._settings_view
        self._content.addWidget(self._settings_view)

        root.addWidget(self._content, stretch=1)

        # Status-Bar (permanent am unteren Rand)
        self._trading_status = TradingStatusBar()
        status_bar = QStatusBar()
        status_bar.setSizeGripEnabled(True)
        status_bar.addPermanentWidget(self._trading_status, 1)
        self.setStatusBar(status_bar)

    @Slot(Section)
    def _on_section_changed(self, section: Section) -> None:
        self._content.setCurrentWidget(self._views[section])

    # ── Oeffentliche API ──────────────────────────────────────────────────────

    @property
    def sidebar(self) -> NavigationSidebar:
        return self._sidebar

    @property
    def trading_status_bar(self) -> TradingStatusBar:
        return self._trading_status

    @property
    def content(self) -> QStackedWidget:
        return self._content

    def current_view(self) -> QWidget:
        return self._content.currentWidget()

    @property
    def dashboard_view(self) -> DashboardView:
        return self._dashboard_view

    @property
    def cockpit_view(self) -> CockpitView:
        return self._cockpit_view

    @property
    def risk_center_view(self) -> RiskCenterView:
        return self._risk_center_view

    @property
    def journal_view(self) -> JournalView:
        return self._journal_view

    @property
    def backtest_view(self) -> BacktestView:
        return self._backtest_view

    @property
    def settings_view(self) -> SettingsView:
        return self._settings_view

    def navigate_to(self, section: Section) -> None:
        """Navigiert programmatisch zu einer Sektion."""
        self._sidebar.navigate_to(section)


# ─────────────────────────────────────────────────────────────────────────────
#  Einstiegspunkt
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
#  MT5-Startup-Hilfsfunktionen
# ─────────────────────────────────────────────────────────────────────────────

def _load_env_file(env_path: str = ".env") -> None:
    """
    Laedt eine .env-Datei in os.environ (setzt nur noch nicht gesetzte Variablen).

    Unterstuetzt:
      KEY=value          – einfache Zuweisung
      KEY="value"        – einfache/doppelte Anführungszeichen werden entfernt
      # Kommentar        – wird uebersprungen
      Leerzeilen         – werden uebersprungen

    Keine externe Abhaengigkeit (kein python-dotenv erforderlich).
    """
    path = Path(env_path)
    if not path.exists():
        return
    try:
        with open(path, encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip()
                if len(val) >= 2 and val[0] == val[-1] and val[0] in ('"', "'"):
                    val = val[1:-1]
                if key and key not in os.environ:
                    os.environ[key] = val
    except OSError as exc:
        logger.debug(".env konnte nicht gelesen werden: {exc}", exc=exc)


def _try_connect_mt5(
    status_bar: "TradingStatusBar",
    _connector_factory: Optional[Callable[..., Any]] = None,
    _env_path: str = ".env",
) -> Optional[Any]:
    """
    Versucht beim App-Start eine MT5-Verbindung aufzubauen.

    Liest MT5_LOGIN, MT5_PASSWORD, MT5_SERVER (und optional MT5_PATH)
    aus os.environ (zuvor wird .env geladen).

    Aktualisiert die TradingStatusBar:
      - CONNECTED  bei Erfolg
      - ERROR      wenn Zugangsdaten vorhanden aber Verbindung fehlschlaegt
      - (unveraendert DISCONNECTED wenn Zugangsdaten fehlen)

    Wirft nie – alle Ausnahmen werden abgefangen und geloggt.

    Parameters
    ----------
    status_bar          : TradingStatusBar-Instanz des MainWindow.
    _connector_factory  : Optionale Factory (login, password, server, path) -> connector.
                          Nur fuer Tests; Standard ist MT5Connector.

    Returns
    -------
    MT5Connector-Instanz bei Erfolg, None sonst.
    """
    _load_env_file(_env_path)

    login_str = os.environ.get("MT5_LOGIN",    "").strip()
    password  = os.environ.get("MT5_PASSWORD", "").strip()
    server    = os.environ.get("MT5_SERVER",   "").strip()
    path      = os.environ.get("MT5_PATH",     "").strip() or None

    if not login_str or not password or not server:
        logger.info("MT5-Startup: Zugangsdaten unvollstaendig – Verbindung uebersprungen.")
        return None

    try:
        login = int(login_str)
    except ValueError:
        logger.warning(
            "MT5-Startup: MT5_LOGIN ist keine gueltige Zahl: '{v}'", v=login_str
        )
        return None

    try:
        if _connector_factory is not None:
            connector = _connector_factory(login, password, server, path)
        else:
            from src.data.mt5_connector import MT5Connector  # lazy – kein Pflicht-Import
            connector = MT5Connector(
                login=login,
                password=password,
                server=server,
                path=path,
                max_retries=1,
            )

        connector.connect()
        status_bar.set_connection(ConnectionStatus.CONNECTED)
        logger.info(
            "MT5-Startup: Verbunden | server={s} login={l}", s=server, l=login
        )
        return connector

    except Exception as exc:  # noqa: BLE001
        status_bar.set_connection(ConnectionStatus.ERROR)
        logger.warning("MT5-Startup: Verbindung fehlgeschlagen: {exc}", exc=exc)
        return None


class _MT5AccountBackend:
    """
    Minimales DashboardBackend das Kontodaten live aus MT5Connector bezieht.
    Implementiert das DashboardBackend-Protokoll per duck-typing.
    """

    def __init__(self, connector: Any) -> None:
        self._connector = connector

    def fetch_snapshot(self) -> DashboardSnapshot:
        try:
            info = self._connector.get_account_info()
        except Exception as exc:  # noqa: BLE001
            logger.warning("MT5 Account-Info fehlgeschlagen: {exc}", exc=exc)
            return DashboardSnapshot()
        return DashboardSnapshot(
            balance=info.get("balance"),
            currency=info.get("currency", "€"),
            equity=info.get("equity"),
            account_number=info.get("login"),
            server=info.get("server"),
            leverage=info.get("leverage"),
            is_demo=info.get("is_demo"),
        )


def create_app(argv: list[str] | None = None) -> QApplication:
    """Erstellt und konfiguriert die QApplication."""
    app = QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("QuantzAI")
    app.setApplicationVersion("0.1.0")
    app.setOrganizationName("QuantzAI")
    return app


def main() -> int:
    """Haupteinstiegspunkt fuer die Desktop-Anwendung."""
    app = create_app()
    # Backtest-Backend einhaengen (kein externes Netzwerk noetig)
    backtest_backend = BacktestGUIBackend()
    window = MainWindow(backtest_backend=backtest_backend)
    window.show()
    connector = _try_connect_mt5(window.trading_status_bar)
    if connector is not None:
        backend = _MT5AccountBackend(connector)
        # Initiale Kontodaten sofort darstellen
        try:
            snap = backend.fetch_snapshot()
            window.dashboard_view.update_display(snap)
            window.trading_status_bar.set_account_info({
                "login":    snap.account_number,
                "balance":  snap.balance,
                "currency": snap.currency,
                "is_demo":  snap.is_demo,
            })
        except Exception as exc:  # noqa: BLE001
            logger.warning("Initialer Account-Abruf fehlgeschlagen: {exc}", exc=exc)
        # Polling einrichten – Dashboard aktualisiert sich alle 5 s
        window.dashboard_view.set_backend(backend)
        window.dashboard_view.start_polling()
        # Statusleiste bei jeder Dashboard-Aktualisierung mitziehen
        window.dashboard_view.data_refreshed.connect(
            lambda s: window.trading_status_bar.set_account_info({
                "login":    s.account_number,
                "balance":  s.balance,
                "currency": s.currency,
                "is_demo":  s.is_demo,
            })
        )
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
