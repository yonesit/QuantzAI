"""
gui/views/dashboard_view.py
Dashboard-View – alle wichtigen Kennzahlen auf einen Blick.

HCI-Prinzipien:
  - Minimale Gedaechtnislast: kein Suchen, alles sichtbar ohne Klicken
  - Sichtbarkeit des Systemstatus: Echtzeit-Polling, Ampelstatus immer praesentiert
  - Kein Datenpunkt ohne Kontext: jede Zahl hat eine erklaerende Bezeichnung

Architektur (Trennung UI / Logik):
  - DashboardSnapshot: reines Daten-Objekt (keine Qt-Abhaengigkeit)
  - compute_risk_status(): pure Funktion, testbar ohne Qt
  - DashboardBackend: Protocol – jedes Objekt das fetch_snapshot() hat
  - DashboardView: nur Darstellung, keine Geschaeftslogik

Anbindung ans Backend:
  backend.fetch_snapshot() -> DashboardSnapshot
  Polling ueber QTimer (konfigurierbares Intervall, Standard 5 s).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Optional, Protocol, runtime_checkable

from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from gui.widgets.chart_widget import CandleData, ChartWidget, Timeframe


# ─────────────────────────────────────────────────────────────────────────────
#  Daten-Typen (pure Python – kein Qt)
# ─────────────────────────────────────────────────────────────────────────────

class RiskStatus(Enum):
    GREEN  = "grün"
    YELLOW = "gelb"
    RED    = "rot"


@dataclass
class SignalInfo:
    """Signal eines Symbol mit Konfidenz."""
    symbol:     str
    signal:     str    # 'long' | 'short' | 'flat'
    confidence: float  # 0.0–1.0


@dataclass
class PositionInfo:
    """Eine offene Position."""
    ticket:           int | str
    symbol:           str
    direction:        str
    lot_size:         float
    open_price:       float | None = None
    current_pnl:      float | None = None
    crv:              float | None = None
    sl_price:         float | None = None
    tp_price:         float | None = None
    break_even_active: bool = False


@dataclass
class DashboardSnapshot:
    """Vollstaendiger Snapshot aller Dashboard-Daten zu einem Zeitpunkt."""

    # Konto
    balance:           float | None = None
    day_start_balance: float | None = None
    all_time_high:     float | None = None
    currency:          str = "€"

    # Erweiterte Kontodaten (nach MT5-Verbindung befuellt)
    equity:         float | None = None
    account_number: int | None   = None
    server:         str | None   = None
    leverage:       int | None   = None
    is_demo:        bool | None  = None

    # Drawdown / Tagesverlust
    drawdown_pct:           float = 0.0
    drawdown_limit_pct:     float = 15.0
    daily_loss_pct:         float = 0.0
    daily_loss_limit_pct:   float = 5.0
    post_loss_days_remaining: int = 0

    # Risiko-Ampel (vorberechnet vom Backend)
    risk_status:  RiskStatus  = RiskStatus.GREEN
    risk_reasons: list[str]   = field(default_factory=list)

    # Offene Positionen
    positions: list[PositionInfo] = field(default_factory=list)

    # Tages-Statistiken (aus TradeJournal)
    today_trades:   int         = 0
    today_pnl:      float | None = None
    today_win_rate: float | None = None

    # Gesamt-Statistiken (realisierte P&L aus paper_trades.json, seit Teststart)
    total_gross_profit: float | None = None
    total_gross_loss:   float | None = None

    # Signale (aus SignalModel)
    signals: list[SignalInfo] = field(default_factory=list)

    # Zeitstempel der letzten Aktualisierung
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@runtime_checkable
class DashboardBackend(Protocol):
    """Protokoll fuer Backend-Objekte die Dashboard-Daten liefern."""
    def fetch_snapshot(self) -> DashboardSnapshot: ...


# ─────────────────────────────────────────────────────────────────────────────
#  Pure Logik (testbar ohne Qt)
# ─────────────────────────────────────────────────────────────────────────────

_RISK_WARNING_RATIO = 0.8  # Warnfarbe ab 80 % des Limits


def compute_risk_status(
    trading_allowed: bool,
    daily_limit_hit: bool,
    max_drawdown_hit: bool,
    post_loss_days: int = 0,
    drawdown_pct: float = 0.0,
    drawdown_limit_pct: float = 15.0,
    anomaly_detected: bool = False,
    warning_ratio: float = _RISK_WARNING_RATIO,
) -> tuple[RiskStatus, list[str]]:
    """
    Berechnet Risiko-Ampelstatus aus mehreren Quellen (RiskGuard, AnomalyDetector).

    Reihenfolge: erste treffende Bedingung bestimmt den Status.
      RED    : max_drawdown_hit | daily_limit_hit | anomaly_detected | not trading_allowed
      YELLOW : post_loss_days > 0 | drawdown >= warning_ratio * limit
      GREEN  : alles in Ordnung
    """
    reasons: list[str] = []

    if max_drawdown_hit:
        reasons.append("Maximaler Drawdown erreicht – Handel gesperrt")
        return RiskStatus.RED, reasons

    if daily_limit_hit:
        reasons.append("Tägliches Verlustlimit erreicht – Handel gesperrt")
        return RiskStatus.RED, reasons

    if anomaly_detected:
        reasons.append("Bot-Anomalie erkannt – Handel gesperrt")
        return RiskStatus.RED, reasons

    if not trading_allowed:
        reasons.append("Handel gesperrt")
        return RiskStatus.RED, reasons

    # YELLOW-Bedingungen
    if post_loss_days > 0:
        reasons.append(
            f"Post-Loss-Phase: {post_loss_days} Tag(e) reduzierte Positionsgröße"
        )

    if drawdown_limit_pct > 0:
        if drawdown_pct / drawdown_limit_pct >= warning_ratio:
            reasons.append(
                f"Drawdown {drawdown_pct:.1f}% nähert sich Limit {drawdown_limit_pct:.1f}%"
            )

    if reasons:
        return RiskStatus.YELLOW, reasons

    return RiskStatus.GREEN, ["Handel erlaubt"]


def _fmt_balance(amount: float | None, currency: str = "€") -> str:
    if amount is None:
        return "--"
    return f"{currency}{amount:,.2f}"


def _fmt_delta(amount: float | None, currency: str = "€") -> str:
    if amount is None:
        return "--"
    sign = "+" if amount > 0 else ""
    return f"{sign}{currency}{amount:,.2f}"


def _fmt_pct(value: float | None, decimals: int = 1) -> str:
    if value is None:
        return "--"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.{decimals}f}%"


def _profit_color(value: float | None) -> str:
    if value is None:
        return ""
    return "#22c55e" if value >= 0 else "#ef4444"


# ─────────────────────────────────────────────────────────────────────────────
#  Interne Widget-Helfer
# ─────────────────────────────────────────────────────────────────────────────

def _card(parent: QWidget | None = None) -> QFrame:
    """Erstellt eine einheitliche Karte (Panel mit Rahmen)."""
    f = QFrame(parent)
    f.setObjectName("dashboard_card")
    f.setFrameShape(QFrame.Shape.StyledPanel)
    return f


def _title_label(text: str, parent: QWidget | None = None) -> QLabel:
    lbl = QLabel(text, parent)
    lbl.setObjectName("card_title")
    lbl.setProperty("secondary", "true")
    return lbl


def _hline() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.Shape.HLine)
    f.setObjectName("dashboard_hline")
    return f


# ─────────────────────────────────────────────────────────────────────────────
#  Widget: Kontostand-Karte
# ─────────────────────────────────────────────────────────────────────────────

class _AccountCard(QFrame):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("dashboard_card")
        self.setFrameShape(QFrame.Shape.StyledPanel)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 12, 16, 14)
        lay.setSpacing(4)

        lay.addWidget(_title_label("Kontostand"))

        self._balance_lbl = QLabel("--")
        self._balance_lbl.setObjectName("account_balance")
        f = self._balance_lbl.font()
        f.setPointSize(20)
        f.setBold(True)
        self._balance_lbl.setFont(f)
        lay.addWidget(self._balance_lbl)

        self._day_lbl = QLabel("Heute: --")
        self._day_lbl.setObjectName("account_day_change")
        self._day_lbl.setToolTip(
            "Tagesveränderung: absoluter Betrag und Prozent gegenüber Tagesstart"
        )
        lay.addWidget(self._day_lbl)

        self._ath_lbl = QLabel("Allzeithoch: --")
        self._ath_lbl.setObjectName("account_ath")
        self._ath_lbl.setProperty("secondary", "true")
        self._ath_lbl.setToolTip("Höchster je erreichter Kontostand (All-Time-High)")
        lay.addWidget(self._ath_lbl)

        self._equity_lbl = QLabel("")
        self._equity_lbl.setObjectName("account_equity")
        self._equity_lbl.setProperty("secondary", "true")
        self._equity_lbl.setToolTip("Aktuelles Eigenkapital (Balance +/- offene Positionen)")
        self._equity_lbl.setVisible(False)
        lay.addWidget(self._equity_lbl)

        self._account_details_lbl = QLabel("")
        self._account_details_lbl.setObjectName("account_details")
        self._account_details_lbl.setProperty("secondary", "true")
        self._account_details_lbl.setToolTip("Kontonummer | Server | Kontotyp | Hebel")
        self._account_details_lbl.setVisible(False)
        lay.addWidget(self._account_details_lbl)

    def refresh(self, snap: DashboardSnapshot) -> None:
        self._balance_lbl.setText(_fmt_balance(snap.balance, snap.currency))

        if snap.balance is not None and snap.day_start_balance is not None:
            delta = snap.balance - snap.day_start_balance
            pct   = (delta / snap.day_start_balance * 100
                     if snap.day_start_balance else 0.0)
            self._day_lbl.setText(
                f"Heute: {_fmt_delta(delta, snap.currency)} ({_fmt_pct(pct)})"
            )
            self._day_lbl.setStyleSheet(f"color: {_profit_color(delta)};")
        else:
            self._day_lbl.setText("Heute: --")
            self._day_lbl.setStyleSheet("")

        self._ath_lbl.setText(
            f"Allzeithoch: {_fmt_balance(snap.all_time_high, snap.currency)}"
        )

        if snap.equity is not None:
            self._equity_lbl.setText(
                f"Equity: {_fmt_balance(snap.equity, snap.currency)}"
            )
            self._equity_lbl.setVisible(True)
        else:
            self._equity_lbl.setVisible(False)

        parts: list[str] = []
        if snap.account_number is not None:
            parts.append(f"#{snap.account_number}")
        if snap.server:
            parts.append(snap.server)
        if snap.is_demo is True:
            parts.append("Demo")
        elif snap.is_demo is False:
            parts.append("Live")
        if snap.leverage is not None:
            parts.append(f"1:{snap.leverage}")
        if parts:
            self._account_details_lbl.setText(" | ".join(parts))
            self._account_details_lbl.setVisible(True)
        else:
            self._account_details_lbl.setVisible(False)


# ─────────────────────────────────────────────────────────────────────────────
#  Widget: Drawdown-Gauge
# ─────────────────────────────────────────────────────────────────────────────

_DD_WARN_RATIO  = _RISK_WARNING_RATIO   # 80 %
_QSS_DD_NORMAL  = "QProgressBar::chunk { background-color: #6366f1; }"
_QSS_DD_WARNING = "QProgressBar::chunk { background-color: #f59e0b; }"
_QSS_DD_DANGER  = "QProgressBar::chunk { background-color: #ef4444; }"


class _DrawdownGauge(QFrame):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("dashboard_card")
        self.setFrameShape(QFrame.Shape.StyledPanel)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 12, 16, 14)
        lay.setSpacing(6)

        lay.addWidget(_title_label("Drawdown"))

        self._pct_lbl = QLabel("0.0% / 15.0% Limit")
        self._pct_lbl.setObjectName("drawdown_label")
        self._pct_lbl.setToolTip(
            "Aktueller Drawdown vom Allzeithoch in Prozent, bezogen auf das konfigurierte Limit"
        )
        lay.addWidget(self._pct_lbl)

        self._bar = QProgressBar()
        self._bar.setObjectName("drawdown_bar")
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        self._bar.setTextVisible(False)
        self._bar.setFixedHeight(12)
        self._bar.setStyleSheet(_QSS_DD_NORMAL)
        lay.addWidget(self._bar)

        self._daily_lbl = QLabel("Tagesverlust: --")
        self._daily_lbl.setProperty("secondary", "true")
        self._daily_lbl.setToolTip("Heutiger Verlust in % des Tagesstart-Kontostands")
        lay.addWidget(self._daily_lbl)

    def refresh(self, snap: DashboardSnapshot) -> None:
        self._pct_lbl.setText(
            f"{snap.drawdown_pct:.1f}% / {snap.drawdown_limit_pct:.1f}% Limit"
        )

        if snap.drawdown_limit_pct > 0:
            ratio   = snap.drawdown_pct / snap.drawdown_limit_pct
            bar_val = min(100, int(ratio * 100))
        else:
            ratio   = 0.0
            bar_val = 0

        self._bar.setValue(bar_val)

        if ratio >= 1.0:
            self._bar.setStyleSheet(_QSS_DD_DANGER)
        elif ratio >= _DD_WARN_RATIO:
            self._bar.setStyleSheet(_QSS_DD_WARNING)
        else:
            self._bar.setStyleSheet(_QSS_DD_NORMAL)

        self._daily_lbl.setText(
            f"Tagesverlust: {snap.daily_loss_pct:.1f}% / {snap.daily_loss_limit_pct:.1f}% Limit"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Widget: Risiko-Ampel
# ─────────────────────────────────────────────────────────────────────────────

_STATUS_COLOR = {
    RiskStatus.GREEN:  "#22c55e",
    RiskStatus.YELLOW: "#f59e0b",
    RiskStatus.RED:    "#ef4444",
}

_STATUS_LABEL = {
    RiskStatus.GREEN:  "Handel erlaubt",
    RiskStatus.YELLOW: "Warnung",
    RiskStatus.RED:    "Handel gesperrt",
}


class _RiskTrafficLight(QFrame):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("dashboard_card")
        self.setFrameShape(QFrame.Shape.StyledPanel)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 12, 16, 14)
        lay.setSpacing(4)
        lay.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        lay.addWidget(_title_label("Risiko-Ampel"))

        self._dot_lbl = QLabel("●")
        self._dot_lbl.setObjectName("risk_dot")
        self._dot_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        df = self._dot_lbl.font()
        df.setPointSize(28)
        self._dot_lbl.setFont(df)
        lay.addWidget(self._dot_lbl)

        self._status_lbl = QLabel("--")
        self._status_lbl.setObjectName("risk_status_label")
        self._status_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sf = self._status_lbl.font()
        sf.setBold(True)
        self._status_lbl.setFont(sf)
        lay.addWidget(self._status_lbl)

        self._reason_lbl = QLabel("")
        self._reason_lbl.setObjectName("risk_reason")
        self._reason_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._reason_lbl.setWordWrap(True)
        self._reason_lbl.setProperty("secondary", "true")
        lay.addWidget(self._reason_lbl)

    def refresh(self, snap: DashboardSnapshot) -> None:
        color  = _STATUS_COLOR[snap.risk_status]
        label  = _STATUS_LABEL[snap.risk_status]
        reason = "\n".join(snap.risk_reasons) if snap.risk_reasons else ""

        self._dot_lbl.setStyleSheet(f"color: {color};")
        self._status_lbl.setText(label)
        self._status_lbl.setStyleSheet(f"color: {color}; font-weight: bold;")
        self._reason_lbl.setText(reason)

    @property
    def status_label(self) -> QLabel:
        return self._status_lbl

    @property
    def dot_label(self) -> QLabel:
        return self._dot_lbl


# ─────────────────────────────────────────────────────────────────────────────
#  Widget: Signal-Panel
# ─────────────────────────────────────────────────────────────────────────────

_SIGNAL_COLOR = {
    "long":  "#22c55e",
    "short": "#ef4444",
    "flat":  "#6b7280",
}


class _SignalRow(QWidget):
    """Eine Zeile fuer ein einzelnes Symbol-Signal."""

    def __init__(self, info: SignalInfo, currency: str = "€",
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 2, 0, 2)
        lay.setSpacing(8)

        sym_lbl = QLabel(info.symbol)
        sym_lbl.setObjectName("signal_symbol")
        sym_lbl.setFixedWidth(80)
        sym_lbl.setToolTip(f"Signal fuer {info.symbol}")

        sig_lbl = QLabel(info.signal.upper())
        sig_lbl.setObjectName("signal_direction")
        color = _SIGNAL_COLOR.get(info.signal.lower(), "#6b7280")
        sig_lbl.setStyleSheet(f"color: {color}; font-weight: bold;")
        sig_lbl.setFixedWidth(60)

        conf_bar = QProgressBar()
        conf_bar.setObjectName("signal_confidence_bar")
        conf_bar.setRange(0, 100)
        conf_bar.setValue(int(info.confidence * 100))
        conf_bar.setTextVisible(False)
        conf_bar.setFixedHeight(10)
        conf_bar.setFixedWidth(100)
        conf_bar.setToolTip(f"Modell-Konfidenz: {info.confidence * 100:.0f}%")

        conf_lbl = QLabel(f"{info.confidence * 100:.0f}%")
        conf_lbl.setObjectName("signal_confidence_label")
        conf_lbl.setFixedWidth(36)
        conf_lbl.setToolTip("Konfidenz des Signal-Modells (0–100%)")

        lay.addWidget(sym_lbl)
        lay.addWidget(sig_lbl)
        lay.addWidget(conf_bar)
        lay.addWidget(conf_lbl)
        lay.addStretch()

        # Store for tests
        self._sym_lbl  = sym_lbl
        self._sig_lbl  = sig_lbl
        self._conf_bar = conf_bar
        self._conf_lbl = conf_lbl

    @property
    def signal_label(self) -> QLabel:
        return self._sig_lbl

    @property
    def confidence_label(self) -> QLabel:
        return self._conf_lbl


class _SignalPanel(QFrame):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("dashboard_card")
        self.setFrameShape(QFrame.Shape.StyledPanel)

        self._outer = QVBoxLayout(self)
        self._outer.setContentsMargins(16, 12, 16, 14)
        self._outer.setSpacing(4)

        self._outer.addWidget(_title_label("Aktuelle Signale"))

        self._empty_lbl = QLabel("Keine Signale verfügbar")
        self._empty_lbl.setProperty("secondary", "true")
        self._outer.addWidget(self._empty_lbl)

        self._rows_container = QWidget()
        self._rows_layout    = QVBoxLayout(self._rows_container)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(2)
        self._outer.addWidget(self._rows_container)

    def refresh(self, snap: DashboardSnapshot) -> None:
        # Alte Zeilen loeschen
        while self._rows_layout.count():
            item = self._rows_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not snap.signals:
            self._empty_lbl.setVisible(True)
            self._rows_container.setVisible(False)
            return

        self._empty_lbl.setVisible(False)
        self._rows_container.setVisible(True)
        for info in snap.signals:
            row = _SignalRow(info, snap.currency)
            self._rows_layout.addWidget(row)

    @property
    def empty_label(self) -> QLabel:
        return self._empty_lbl


# ─────────────────────────────────────────────────────────────────────────────
#  Hauptview
# ─────────────────────────────────────────────────────────────────────────────

class DashboardView(QScrollArea):
    """
    Dashboard-Hauptansicht – grobe Uebersicht des Konto- und Systemstatus.

    Zeigt auf einen Blick:
      - Kontostand mit Tagesveraenderung und Equity
      - Drawdown-Gauge mit konfigurierbarer Warnschwelle
      - Risiko-Ampel (gruen/gelb/rot)
      - Aktuelle Signale aus SignalModel
      - Candlestick-Chart fuer Marktbeobachtung

    Positionen, Tages- und Gesamtstatistiken befinden sich im Cockpit.

    Parameters
    ----------
    backend     : Objekt mit fetch_snapshot() -> DashboardSnapshot.
                  None = keine automatische Aktualisierung.
    interval_ms : Polling-Intervall in Millisekunden (Standard: 5 000).
    parent      : Eltern-Widget.
    """

    # Signal wenn Daten abgerufen wurden (fuer Tests und UI-Updates)
    data_refreshed = Signal(DashboardSnapshot)

    # Behalten fuer Rueckwaertskompatibilitaet mit run_gui_bot.py
    position_close_requested = Signal(int)

    def __init__(
        self,
        backend: Optional[DashboardBackend] = None,
        interval_ms: int = 5_000,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("dashboard_view")
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        self._backend     = backend
        self._interval_ms = interval_ms
        self._last_snap   = DashboardSnapshot()  # Initialzustand

        # Chart-Zustand
        self._chart_connector = None
        self._chart_symbol: str = "EURUSD"
        self._chart_tf: Timeframe = Timeframe.H1

        self._build()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_timer)

        # Separater Timer fuer Chart-Refresh (60 s)
        self._chart_timer = QTimer(self)
        self._chart_timer.timeout.connect(self._refresh_chart)

        # Initialen Zustand anzeigen
        self._apply_snapshot(self._last_snap)

    def _build(self) -> None:
        container = QWidget()
        container.setObjectName("dashboard_container")
        self.setWidget(container)

        root = QVBoxLayout(container)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        # ── Obere Reihe: Account | Drawdown | Risiko-Ampel ───────────────────
        top_row = QHBoxLayout()
        top_row.setSpacing(12)

        self._account   = _AccountCard()
        self._drawdown  = _DrawdownGauge()
        self._risk_light = _RiskTrafficLight()

        for w in (self._account, self._drawdown, self._risk_light):
            w.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
            )
            top_row.addWidget(w)

        root.addLayout(top_row)

        # ── Markt-Chart ──────────────────────────────────────────────────────
        chart_card = _card()
        chart_lay = QVBoxLayout(chart_card)
        chart_lay.setContentsMargins(16, 12, 16, 14)
        chart_lay.setSpacing(6)
        chart_lay.addWidget(_title_label("Markt-Chart"))
        self._chart = ChartWidget()
        self._chart.setMinimumHeight(280)
        self._chart.set_symbol(self._chart_symbol)
        self._chart.timeframe_changed.connect(self._on_chart_tf_changed)
        chart_lay.addWidget(self._chart)
        root.addWidget(chart_card)

        # ── Signal-Panel ─────────────────────────────────────────────────────
        self._signals = _SignalPanel()
        root.addWidget(self._signals)

        root.addStretch()

        # Zeitstempel-Label
        self._updated_lbl = QLabel("Letzte Aktualisierung: --")
        self._updated_lbl.setObjectName("dashboard_updated_at")
        self._updated_lbl.setProperty("secondary", "true")
        self._updated_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
        root.addWidget(self._updated_lbl)

    def _apply_snapshot(self, snap: DashboardSnapshot) -> None:
        """Aktualisiert alle Widgets mit neuen Daten."""
        self._last_snap = snap
        self._account.refresh(snap)
        self._drawdown.refresh(snap)
        self._risk_light.refresh(snap)
        self._signals.refresh(snap)
        self._updated_lbl.setText(f"Letzte Aktualisierung: {snap.updated_at[:19].replace('T', ' ')} UTC")

    @Slot()
    def _on_timer(self) -> None:
        if self._backend is None:
            return
        try:
            snap = self._backend.fetch_snapshot()
        except Exception:  # noqa: BLE001
            return
        self._apply_snapshot(snap)
        self.data_refreshed.emit(snap)

    # ── Oeffentliche API ──────────────────────────────────────────────────────

    def update_display(self, snapshot: DashboardSnapshot) -> None:
        """Manuell einen Snapshot einspielen (z.B. aus Tests oder Push-Updates)."""
        self._apply_snapshot(snapshot)

    def set_backend(self, backend: DashboardBackend) -> None:
        """Setzt oder ersetzt das Backend. Startet Polling nicht automatisch."""
        self._backend = backend

    def start_polling(self) -> None:
        """Startet automatisches Polling (backend muss gesetzt sein)."""
        if self._backend is not None and not self._timer.isActive():
            self._timer.start(self._interval_ms)

    def stop_polling(self) -> None:
        """Stoppt automatisches Polling."""
        self._timer.stop()

    @property
    def is_polling(self) -> bool:
        return self._timer.isActive()

    @property
    def account_card(self) -> _AccountCard:
        return self._account

    @property
    def drawdown_gauge(self) -> _DrawdownGauge:
        return self._drawdown

    @property
    def risk_light(self) -> _RiskTrafficLight:
        return self._risk_light

    @property
    def signal_panel(self) -> _SignalPanel:
        return self._signals

    @property
    def last_snapshot(self) -> DashboardSnapshot:
        return self._last_snap

    def set_chart_connector(self, connector, symbol: str) -> None:
        """
        Verbindet den MT5Connector fuer Chart-Daten.
        Startet automatischen Chart-Refresh (60 s).
        """
        self._chart_connector = connector
        self._chart_symbol = symbol
        self._chart.set_symbol(symbol)
        self._refresh_chart()
        if not self._chart_timer.isActive():
            self._chart_timer.start(60_000)

    @Slot()
    def _refresh_chart(self) -> None:
        """Holt Candles vom Connector und aktualisiert den Chart."""
        if self._chart_connector is None:
            return
        from datetime import timezone as _tz
        try:
            tf_str = self._chart_tf.label
            df = self._chart_connector.get_ohlcv_count(
                self._chart_symbol, tf_str, count=200
            )
            candles = []
            for ts, row in df.iterrows():
                ts_py = ts.to_pydatetime()
                if ts_py.tzinfo is None:
                    ts_py = ts_py.replace(tzinfo=_tz.utc)
                candles.append(CandleData(
                    timestamp=ts_py,
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row.get("volume", 0.0)),
                ))
            self._chart.set_candles(candles)
            # Bid/Ask aus letzter Kerze approximieren (~1.5 Pip Spread fuer Majors)
            if candles:
                mid = candles[-1].close
                self._chart.set_bid_ask(mid - 0.000075, mid + 0.000075)
        except Exception as exc:  # noqa: BLE001
            from loguru import logger
            logger.warning("Chart-Refresh fehlgeschlagen: {exc}", exc=exc)

    @Slot(object)
    def _on_chart_tf_changed(self, tf: Timeframe) -> None:
        self._chart_tf = tf
        self._refresh_chart()

    def connect_order_executor(self, relay) -> None:
        """
        Rueckwaertskompatibilitaet: Positionen werden jetzt im Cockpit verwaltet.
        Methode bleibt erhalten damit bestehende Aufrufer nicht crashen.
        """
        relay.order_opened.connect(self.on_order_opened)
        relay.order_closed.connect(self.on_order_closed)

    @Slot(dict)
    def on_order_opened(self, order: dict) -> None:  # noqa: ARG002
        """No-op: Positionen werden im Cockpit verwaltet."""

    @Slot(dict)
    def on_order_closed(self, order: dict) -> None:  # noqa: ARG002
        """No-op: Positionen werden im Cockpit verwaltet."""
