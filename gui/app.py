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

import sys
from enum import Enum, auto
from typing import Optional

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
from gui.views.cockpit_view import CockpitBackend, CockpitView
from gui.views.dashboard_view import DashboardBackend, DashboardView
from gui.views.journal_view import JournalBackend, JournalView
from gui.views.risk_center_view import RiskCenterBackend, RiskCenterView


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


class BacktestView(_PlaceholderView):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("📈  Backtest", parent)


class SettingsView(_PlaceholderView):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("⚙  Einstellungen", parent)


_PLACEHOLDER_VIEWS: dict[Section, type[_PlaceholderView]] = {
    Section.BACKTEST: BacktestView,
    Section.SETTINGS: SettingsView,
}


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

        self._connection = ConnectionStatus.DISCONNECTED
        self._mode       = TradingMode.SUGGEST
        self._paused     = True

        self._build()
        self._refresh()

    def _build(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 0, 12, 0)
        layout.setSpacing(16)

        self._conn_dot    = QLabel()
        self._conn_label  = QLabel()
        self._mode_label  = QLabel()
        self._bot_label   = QLabel()

        for lbl in (self._conn_dot, self._conn_label,
                    self._mode_label, self._bot_label):
            lbl.setObjectName("status_label")

        layout.addWidget(self._conn_dot)
        layout.addWidget(self._conn_label)
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

        # Restliche Sektionen: Placeholder-Views
        for section, ViewClass in _PLACEHOLDER_VIEWS.items():
            view = ViewClass()
            self._views[section] = view
            self._content.addWidget(view)

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

    def navigate_to(self, section: Section) -> None:
        """Navigiert programmatisch zu einer Sektion."""
        self._sidebar.navigate_to(section)


# ─────────────────────────────────────────────────────────────────────────────
#  Einstiegspunkt
# ─────────────────────────────────────────────────────────────────────────────

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
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
