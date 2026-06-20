"""
gui/dialogs/trade_confirmation_dialog.py
Nicht-modaler Bestaetigungsdialog fuer den CONFIRM_REQUIRED-Modus.

Komponenten:
  TradeProposal            – Datenhaltung: alle relevanten Trade-Infos
  TradeConfirmationBanner  – Nicht-modales QWidget mit Countdown und drei Aktionen
  GuiConfirmationCallback  – Implementiert ConfirmationCallback-Protokoll fuer GUI;
                             Bruecke zwischen Worker-Thread und GUI-Hauptthread

Threading-Pattern:
  confirm_order() laeuft im Worker-Thread und blockiert via threading.Event.
  _request_signal (QueuedConnection) liefert den TradeProposal in den Hauptthread.
  Nutzer klickt Bestaetigen / Ablehnen -> _event.set() -> Worker laeuft weiter.
  Timeout (configurable, Standard 60 s): Event laeuft ab, keine Order.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable, Optional

from loguru import logger

from PySide6.QtCore import QObject, QTimer, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


# ─────────────────────────────────────────────────────────────────────────────
#  TradeProposal
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TradeProposal:
    """Alle relevanten Daten eines ausstehenden Trade-Signals."""

    symbol:      str
    direction:   str           # "buy" | "sell"
    lot_size:    float
    sl_price:    float
    tp_price:    float
    confidence:  Optional[float] = None
    spread_cost: Optional[float] = None


# ─────────────────────────────────────────────────────────────────────────────
#  TradeConfirmationBanner
# ─────────────────────────────────────────────────────────────────────────────

class TradeConfirmationBanner(QWidget):
    """
    Nicht-modales Widget fuer Trade-Bestaetigung mit Countdown.

    Zeigt Symbol, Richtung, Lots, SL/TP, Konfidenz, Spread-Kosten und
    einen Countdown. Drei Aktionen: Bestaetigen / Ablehnen / Lots anpassen.

    Signals
    -------
    confirmed(float)   – lot_size (original oder angepasst)
    rejected()
    timed_out()
    """

    confirmed = Signal(float)   # emittiert lot_size
    rejected  = Signal()
    timed_out = Signal()

    def __init__(
        self,
        proposal: TradeProposal,
        timeout_seconds: int = 60,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._proposal  = proposal
        self._remaining = timeout_seconds
        self._resolved  = False
        self._build()
        self._start_timer()

    # ─── Layout ──────────────────────────────────────────────────────────────

    def _build(self) -> None:
        p = self._proposal
        is_buy    = p.direction.lower() == "buy"
        dir_str   = "BUY  ▲" if is_buy else "SELL ▼"
        dir_color = "#22c55e" if is_buy else "#ef4444"

        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 10, 14, 10)
        outer.setSpacing(7)

        # ── Titelzeile ───────────────────────────────────────────────────────
        title_row = QHBoxLayout()

        icon = QLabel("⚡")
        icon.setObjectName("confirmation_icon")
        f = icon.font()
        f.setPointSize(14)
        icon.setFont(f)
        title_row.addWidget(icon)

        title = QLabel("Trade-Bestätigung erforderlich")
        title.setObjectName("confirmation_title")
        f = title.font()
        f.setBold(True)
        f.setPointSize(11)
        title.setFont(f)
        title_row.addWidget(title)
        title_row.addStretch()

        self._countdown_label = QLabel(f"⏱ {self._remaining}s")
        self._countdown_label.setObjectName("confirmation_countdown")
        self._countdown_label.setStyleSheet("color: #f59e0b; font-weight: bold;")
        title_row.addWidget(self._countdown_label)
        outer.addLayout(title_row)

        # ── Trennlinie ───────────────────────────────────────────────────────
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setObjectName("confirmation_separator")
        outer.addWidget(sep)

        # ── Detail-Zeile ─────────────────────────────────────────────────────
        detail_row = QHBoxLayout()
        detail_row.setSpacing(16)

        sym_lbl = QLabel(f"<b>{p.symbol}</b>")
        sym_lbl.setObjectName("confirmation_symbol")

        dir_lbl = QLabel(dir_str)
        dir_lbl.setObjectName("confirmation_direction")
        dir_lbl.setStyleSheet(f"color: {dir_color}; font-weight: bold;")

        lot_lbl = QLabel(f"Lots: <b>{p.lot_size:.2f}</b>")
        lot_lbl.setObjectName("confirmation_lot_display")

        sl_lbl = QLabel(f"SL: {p.sl_price:.5f}")
        sl_lbl.setObjectName("confirmation_sl")

        tp_lbl = QLabel(f"TP: {p.tp_price:.5f}")
        tp_lbl.setObjectName("confirmation_tp")

        for w in (sym_lbl, dir_lbl, lot_lbl, sl_lbl, tp_lbl):
            detail_row.addWidget(w)

        if p.confidence is not None:
            conf_lbl = QLabel(f"Konfidenz: {p.confidence:.0%}")
            conf_lbl.setObjectName("confirmation_confidence")
            detail_row.addWidget(conf_lbl)

        if p.spread_cost is not None:
            cost_lbl = QLabel(f"~Spread: ${p.spread_cost:.2f}")
            cost_lbl.setObjectName("confirmation_spread")
            detail_row.addWidget(cost_lbl)

        detail_row.addStretch()
        outer.addLayout(detail_row)

        # ── Lot-Anpassung + Buttons ──────────────────────────────────────────
        action_row = QHBoxLayout()
        action_row.setSpacing(8)

        action_row.addWidget(QLabel("Lots:"))

        self._lot_spin = QDoubleSpinBox()
        self._lot_spin.setObjectName("confirmation_lot_spin")
        self._lot_spin.setMinimum(0.01)
        self._lot_spin.setMaximum(100.0)
        self._lot_spin.setSingleStep(0.01)
        self._lot_spin.setDecimals(2)
        self._lot_spin.setValue(p.lot_size)
        action_row.addWidget(self._lot_spin)

        action_row.addSpacing(12)

        self._confirm_btn = QPushButton("✓  Bestätigen")
        self._confirm_btn.setObjectName("confirmation_confirm_btn")
        self._confirm_btn.setStyleSheet(
            "background: #22c55e; color: white; font-weight: bold; padding: 5px 16px;"
        )
        self._confirm_btn.clicked.connect(self._on_confirm)
        action_row.addWidget(self._confirm_btn)

        self._reject_btn = QPushButton("✗  Ablehnen")
        self._reject_btn.setObjectName("confirmation_reject_btn")
        self._reject_btn.setStyleSheet(
            "background: #ef4444; color: white; font-weight: bold; padding: 5px 16px;"
        )
        self._reject_btn.clicked.connect(self._on_reject)
        action_row.addWidget(self._reject_btn)

        action_row.addStretch()
        outer.addLayout(action_row)

        # ── Widget-Rahmen ────────────────────────────────────────────────────
        self.setObjectName("trade_confirmation_banner")
        self.setStyleSheet(
            "#trade_confirmation_banner {"
            "  border: 2px solid #f59e0b;"
            "  border-radius: 8px;"
            "  background: #1c1c1e;"
            "}"
        )
        self.setMinimumWidth(580)

    # ─── Timer ───────────────────────────────────────────────────────────────

    def _start_timer(self) -> None:
        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    @Slot()
    def _tick(self) -> None:
        self._remaining -= 1
        self._countdown_label.setText(f"⏱ {self._remaining}s")
        if self._remaining <= 10:
            self._countdown_label.setStyleSheet(
                "color: #ef4444; font-weight: bold;"
            )
        if self._remaining <= 0:
            self._timer.stop()
            if not self._resolved:
                self._resolved = True
                self.timed_out.emit()

    # ─── Aktions-Slots ───────────────────────────────────────────────────────

    def _resolve(self, ok: bool, lot: Optional[float] = None) -> None:
        if self._resolved:
            return
        self._resolved = True
        self._timer.stop()
        self._confirm_btn.setEnabled(False)
        self._reject_btn.setEnabled(False)
        if ok:
            self.confirmed.emit(lot if lot is not None else self._lot_spin.value())
        else:
            self.rejected.emit()

    @Slot()
    def _on_confirm(self) -> None:
        self._resolve(True, self._lot_spin.value())

    @Slot()
    def _on_reject(self) -> None:
        self._resolve(False)

    # ─── Properties ──────────────────────────────────────────────────────────

    @property
    def proposal(self) -> TradeProposal:
        return self._proposal

    @property
    def remaining_seconds(self) -> int:
        return self._remaining

    @property
    def is_resolved(self) -> bool:
        return self._resolved


# ─────────────────────────────────────────────────────────────────────────────
#  GuiConfirmationCallback
# ─────────────────────────────────────────────────────────────────────────────

class GuiConfirmationCallback(QObject):
    """
    Implementiert das ConfirmationCallback-Protokoll fuer die GUI.

    Wird von TradingOrchestrator aus dem Worker-Thread aufgerufen.
    Zeigt einen TradeConfirmationBanner im Hauptthread (nicht-modal).
    Blockiert den Worker-Thread via threading.Event bis zum Ergebnis.

    Parameters
    ----------
    parent_widget    : Eltern-QWidget fuer den Banner (z.B. MainWindow).
    timeout_seconds  : Automatischer Abbruch ohne Reaktion (Standard: 60).
    audit_fn         : Optionale Funktion (action: str, data: dict) -> None
                       fuer Audit-Log-Eintraege nach jeder Entscheidung.
    """

    # Emittiert aus dem Worker-Thread via QueuedConnection in den Hauptthread
    _request_signal: Signal = Signal(object)  # TradeProposal

    def __init__(
        self,
        parent_widget: Optional[QWidget] = None,
        timeout_seconds: int = 60,
        audit_fn: Optional[Callable[[str, dict], None]] = None,
    ) -> None:
        super().__init__(parent_widget)
        self._parent_widget = parent_widget
        self._timeout       = timeout_seconds
        self._audit_fn      = audit_fn

        self._banner:   Optional[TradeConfirmationBanner] = None
        self._event:    Optional[threading.Event]         = None
        self._result:   list[bool]           = [False]
        self._last_lot: list[Optional[float]] = [None]

        self._request_signal.connect(
            self._on_request, Qt.ConnectionType.QueuedConnection
        )

    # ─── ConfirmationCallback Protocol ──────────────────────────────────────

    def confirm_order(
        self,
        symbol: str,
        direction: str,
        lot_size: float,
        sl: float,
        tp: float,
    ) -> bool:
        """
        Wird aus dem Worker-Thread aufgerufen.
        Blockiert bis der Nutzer antwortet oder der Timeout ablaeuft.
        Gibt True zurueck wenn bestaetigt, False bei Ablehnen/Timeout.
        """
        self._event       = threading.Event()
        self._result[0]   = False
        self._last_lot[0] = None

        self._request_signal.emit(
            TradeProposal(
                symbol=symbol,
                direction=direction,
                lot_size=lot_size,
                sl_price=sl,
                tp_price=tp,
            )
        )

        fired     = self._event.wait(timeout=self._timeout)
        confirmed = fired and self._result[0]

        if self._audit_fn is not None:
            try:
                action = "ORDER_CONFIRMED" if confirmed else "ORDER_REJECTED"
                self._audit_fn(action, {
                    "symbol":    symbol,
                    "direction": direction,
                    "lot_size":  lot_size,
                    "sl":        sl,
                    "tp":        tp,
                    "timed_out": not fired,
                })
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "GuiConfirmationCallback: Audit-Fehler: {e}", e=exc
                )

        return confirmed

    # ─── Oeffentliche Properties ─────────────────────────────────────────────

    @property
    def last_confirmed_lot_size(self) -> Optional[float]:
        """Letzte vom Nutzer bestaetigt Lot-Groesse (ggf. angepasst)."""
        return self._last_lot[0]

    # ─── Hauptthread-Slots ───────────────────────────────────────────────────

    @Slot(object)
    def _on_request(self, proposal: TradeProposal) -> None:
        """Zeigt den Banner im GUI-Hauptthread."""
        try:
            QApplication.beep()
        except Exception:  # noqa: BLE001
            pass

        self._banner = TradeConfirmationBanner(
            proposal,
            timeout_seconds=self._timeout,
            parent=self._parent_widget,
        )
        self._banner.confirmed.connect(self._on_confirmed)
        self._banner.rejected.connect(self._on_rejected)
        self._banner.timed_out.connect(self._on_timed_out)

        if self._parent_widget is not None:
            pw = self._parent_widget
            bw = max(self._banner.minimumWidth(), 580)
            x  = max(0, (pw.width() - bw) // 2)
            self._banner.setGeometry(x, 10, bw, 130)
            self._banner.raise_()

        self._banner.show()
        logger.info(
            "GuiConfirmationCallback: Banner angezeigt | {sym} {dir}",
            sym=proposal.symbol,
            dir=proposal.direction,
        )

    @Slot(float)
    def _on_confirmed(self, lot_size: float) -> None:
        self._result[0]   = True
        self._last_lot[0] = lot_size
        if self._event is not None:
            self._event.set()

    @Slot()
    def _on_rejected(self) -> None:
        self._result[0] = False
        if self._event is not None:
            self._event.set()

    @Slot()
    def _on_timed_out(self) -> None:
        self._result[0] = False
        if self._event is not None:
            self._event.set()
