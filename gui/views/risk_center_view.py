"""
gui/views/risk_center_view.py
Risiko-Zentrale: alle risikorelevanten Informationen gebuendelt an einem Ort.

Zeigt:
  - Tages-/Wochen-/Monatsverlust vs. Limit (farbkodierter Balken)
  - Drawdown-Verlauf als Zeitreihen-Chart (QPainter)
  - Korrelationsmatrix offener Positionen als Heatmap (QPainter)
  - VaR/CVaR fuer das aktuelle Portfolio
  - Liste aktiver Warnungen/Eingriffe mit Zeitstempel und Klartext
  - Pause/Resume-, Notfall-Stop- und Drawdown-Freigabe-Button (je mit Bestaetigung)

Backend-Protocol (RiskCenterBackend):
  get_loss_summary()      -> dict  {daily/weekly/monthly: {loss_pct, limit_pct}}
  get_drawdown_history()  -> list[dict]  [{timestamp, drawdown_pct}]
  get_correlation_data()  -> dict  {symbols, matrix, threshold}
  get_var_cvar()          -> dict  {var, cvar, confidence}
  get_active_warnings()   -> list[dict]  [{timestamp, type, message}]
  is_trading_paused()     -> bool
  is_max_drawdown_hit()   -> bool
  pause_trading(reason)   -> None
  resume_trading()        -> None
  emergency_stop()        -> dict
  release_drawdown_stop() -> None

Testbarkeit:
  _confirm_fn injizierbar (ersetzt ConfirmationDialog.ask in Tests)
  Backend via set_backend() nachtraeglich setzbar.
"""

from __future__ import annotations

from datetime import datetime
from typing import Callable, Optional, Protocol, runtime_checkable

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Backend-Protocol
# ─────────────────────────────────────────────────────────────────────────────

@runtime_checkable
class RiskCenterBackend(Protocol):
    def get_loss_summary(self) -> dict: ...
    def get_drawdown_history(self) -> list[dict]: ...
    def get_correlation_data(self) -> dict: ...
    def get_var_cvar(self) -> dict: ...
    def get_active_warnings(self) -> list[dict]: ...
    def is_trading_paused(self) -> bool: ...
    def is_max_drawdown_hit(self) -> bool: ...
    def pause_trading(self, reason: str) -> None: ...
    def resume_trading(self) -> None: ...
    def emergency_stop(self) -> dict: ...
    def release_drawdown_stop(self) -> None: ...


# ─────────────────────────────────────────────────────────────────────────────
#  _LossGauge – farbkodierter Verlust-Balken
# ─────────────────────────────────────────────────────────────────────────────

class _LossGauge(QWidget):
    """Zeigt Verlust-% vs. Limit als farbkodierten Fortschrittsbalken."""

    _BG    = QColor("#1e1e2e")
    _GREEN = QColor("#22c55e")
    _AMBER = QColor("#f59e0b")
    _RED   = QColor("#ef4444")
    _TEXT  = QColor("#e2e8f0")
    _H     = 28

    def __init__(self, label: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._label    = label
        self._loss_pct = 0.0
        self._limit    = 5.0
        self.setFixedHeight(self._H)
        self.setMinimumWidth(180)

    def set_values(self, loss_pct: float, limit_pct: float) -> None:
        self._loss_pct = max(0.0, loss_pct)
        self._limit    = max(0.01, limit_pct)
        self.update()

    @property
    def loss_pct(self) -> float:
        return self._loss_pct

    @property
    def limit_pct(self) -> float:
        return self._limit

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        ratio = min(1.0, self._loss_pct / self._limit) if self._limit > 0 else 0.0
        p.fillRect(0, 0, w, h, self._BG)
        color = self._GREEN if ratio < 0.5 else (self._AMBER if ratio < 0.8 else self._RED)
        bar_w = int(w * ratio)
        if bar_w > 0:
            p.fillRect(0, 0, bar_w, h, color)
        p.setPen(QPen(self._TEXT))
        text = f"{self._label}: {self._loss_pct:.1f}% / {self._limit:.1f}%"
        p.drawText(8, 0, w - 16, h, Qt.AlignmentFlag.AlignVCenter, text)


# ─────────────────────────────────────────────────────────────────────────────
#  _DrawdownCanvas – Zeitreihen-Chart des Drawdown-Verlaufs
# ─────────────────────────────────────────────────────────────────────────────

class _DrawdownCanvas(QWidget):
    """Zeichnet den Drawdown-Verlauf als Flaechen-Chart (QPainter)."""

    _BG   = QColor("#0f0f11")
    _GRID = QColor("#2a2a3e")
    _LINE = QColor("#ef4444")
    _FILL = QColor(239, 68, 68, 60)
    _LIM  = QColor("#f59e0b")
    _TEXT = QColor("#94a3b8")

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("drawdown_canvas")
        self._points: list[tuple[float, float]] = []
        self._max_dd  = 15.0
        self.setMinimumHeight(140)

    def set_data(self, history: list[dict], max_drawdown_pct: float = 15.0) -> None:
        self._max_dd = max(1.0, max_drawdown_pct)
        if not history:
            self._points = []
            self.update()
            return

        raw_ts: list[float] = []
        for item in history:
            ts = item.get("timestamp")
            if isinstance(ts, str):
                try:
                    ts = datetime.fromisoformat(ts).timestamp()
                except ValueError:
                    ts = 0.0
            elif isinstance(ts, datetime):
                ts = ts.timestamp()
            else:
                ts = float(ts or 0)
            raw_ts.append(ts)

        t_min, t_max = min(raw_ts), max(raw_ts)
        t_range = (t_max - t_min) or 1.0
        self._points = [
            ((t - t_min) / t_range, float(h.get("drawdown_pct", 0.0)))
            for t, h in zip(raw_ts, history)
        ]
        self.update()

    @property
    def point_count(self) -> int:
        return len(self._points)

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        PL, PR, PT, PB = 44, 8, 8, 24
        cw, ch = w - PL - PR, h - PT - PB

        p.fillRect(0, 0, w, h, self._BG)

        # Grid + Y-labels
        steps = 5
        for i in range(steps + 1):
            gy = PT + int(ch * i / steps)
            p.setPen(QPen(self._GRID, 1))
            p.drawLine(PL, gy, PL + cw, gy)
            val = self._max_dd * (steps - i) / steps
            p.setPen(QPen(self._TEXT))
            p.drawText(0, gy - 8, PL - 4, 16,
                       Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                       f"{val:.0f}%")

        # Limit line at top
        p.setPen(QPen(self._LIM, 1, Qt.PenStyle.DashLine))
        p.drawLine(PL, PT, PL + cw, PT)

        if not self._points:
            p.setPen(QPen(self._TEXT))
            p.drawText(PL, PT, cw, ch, Qt.AlignmentFlag.AlignCenter, "Keine Daten")
            return

        def _xy(t_n: float, dd: float) -> tuple[int, int]:
            return (
                PL + int(t_n * cw),
                PT + int(min(1.0, dd / self._max_dd) * ch),
            )

        from PySide6.QtCore import QPointF
        pts = [QPointF(*_xy(t, dd)) for t, dd in self._points]
        closed = pts + [
            QPointF(_xy(self._points[-1][0], 0)[0], PT + ch),
            QPointF(_xy(self._points[0][0],  0)[0], PT + ch),
        ]
        p.setBrush(QBrush(self._FILL))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawPolygon(QPolygonF(closed))

        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(self._LINE, 2))
        for i in range(1, len(self._points)):
            x1, y1 = _xy(*self._points[i - 1])
            x2, y2 = _xy(*self._points[i])
            p.drawLine(x1, y1, x2, y2)


# ─────────────────────────────────────────────────────────────────────────────
#  _HeatmapCanvas – Korrelationsmatrix als farbkodierte Heatmap
# ─────────────────────────────────────────────────────────────────────────────

class _HeatmapCanvas(QWidget):
    """Zeichnet eine N×N Korrelationsmatrix als Heatmap (QPainter).

    Zellen oberhalb des Schwellwerts (nicht-Diagonale) werden
    mit einem Amber-Rahmen hervorgehoben.
    """

    _BG        = QColor("#0f0f11")
    _TEXT      = QColor("#e2e8f0")
    _BORDER_HI = QColor("#f59e0b")

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("heatmap_canvas")
        self._symbols:   list[str]                    = []
        self._matrix:    dict[tuple[str, str], float] = {}
        self._threshold: float                        = 0.8
        self.setMinimumSize(160, 140)

    def set_data(
        self,
        symbols:   list[str],
        matrix:    dict,
        threshold: float = 0.8,
    ) -> None:
        """
        matrix kann nested-dict {sym_a: {sym_b: float}} oder
        flat-dict {(sym_a, sym_b): float} sein.
        """
        self._symbols   = list(symbols)
        self._threshold = threshold
        self._matrix    = {}
        for a in symbols:
            for b in symbols:
                if a == b:
                    v = 1.0
                elif isinstance(matrix.get(a), dict):
                    v = float(matrix[a].get(b, 0.0))
                elif (a, b) in matrix:
                    v = float(matrix[(a, b)])
                else:
                    v = 0.0
                self._matrix[(a, b)] = v
        self.update()

    @property
    def symbols(self) -> list[str]:
        return list(self._symbols)

    @property
    def threshold(self) -> float:
        return self._threshold

    def get_correlation(self, a: str, b: str) -> float:
        return self._matrix.get((a, b), 0.0)

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, self._BG)

        n = len(self._symbols)
        if n == 0:
            p.setPen(QPen(self._TEXT))
            p.drawText(0, 0, w, h, Qt.AlignmentFlag.AlignCenter, "Keine Positionen")
            return

        PL, PT, PR, PB = 52, 20, 8, 52
        cell_w = (w - PL - PR) / n
        cell_h = (h - PT - PB) / n

        sf = QFont(p.font())
        sf.setPointSize(max(6, min(9, int(cell_w / 4))))
        p.setFont(sf)

        for i, sym_a in enumerate(self._symbols):
            # Column label (rotated, top)
            col_cx = PL + int(i * cell_w + cell_w / 2)
            p.setPen(QPen(self._TEXT))
            p.save()
            p.translate(col_cx, PT - 4)
            p.rotate(-45)
            p.drawText(0, 0, sym_a)
            p.restore()
            # Row label (left)
            row_cy = PT + int(i * cell_h + cell_h / 2)
            p.drawText(0, row_cy - 8, PL - 4, 16,
                       Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                       sym_a)

            for j, sym_b in enumerate(self._symbols):
                corr = self._matrix.get((sym_a, sym_b), 0.0)
                cx = PL + int(j * cell_w)
                cy = PT + int(i * cell_h)
                cw_px = max(1, int(cell_w) - 1)
                ch_px = max(1, int(cell_h) - 1)

                # green (low) → grey (0) → red (high)
                if corr >= 0:
                    r = min(255, int(60 + 195 * corr))
                    g = min(255, int(200 - 140 * corr))
                    b = 60
                else:
                    r = 60
                    g = min(255, int(200 + 140 * corr))
                    b = min(255, int(60 + 140 * (-corr)))
                p.fillRect(cx, cy, cw_px, ch_px, QColor(r, g, b))

                # Highlight above threshold (non-diagonal)
                if sym_a != sym_b and abs(corr) > self._threshold:
                    p.setPen(QPen(self._BORDER_HI, 2))
                    p.drawRect(cx, cy, cw_px, ch_px)

                # Value text
                if cell_w > 24:
                    p.setPen(QPen(QColor("#ffffff")))
                    p.drawText(cx, cy, cw_px, ch_px,
                               Qt.AlignmentFlag.AlignCenter, f"{corr:.2f}")

        p.setPen(Qt.PenStyle.NoPen)


# ─────────────────────────────────────────────────────────────────────────────
#  RiskCenterView
# ─────────────────────────────────────────────────────────────────────────────

_COL_TS  = 0
_COL_TYP = 1
_COL_MSG = 2


class RiskCenterView(QWidget):
    """
    Risiko-Zentrale: gebündelte Risikoansicht fuer den Operator.

    Signals
    -------
    trading_paused(bool)   – True nach Pause, False nach Resume
    emergency_stopped      – nach bestaetigendem Notfall-Stop
    drawdown_released      – nach manueller Drawdown-Freigabe

    Parameters
    ----------
    backend     : RiskCenterBackend-Implementierung (kann None sein)
    _confirm_fn : Callable[[str, str, str], bool] – ersetzt ConfirmationDialog
                  in Tests; Signatur: (title, message, confirm_label) -> bool
    """

    trading_paused    = Signal(bool)
    emergency_stopped = Signal()
    drawdown_released = Signal()

    def __init__(
        self,
        backend:     Optional[RiskCenterBackend] = None,
        _confirm_fn: Optional[Callable[[str, str, str], bool]] = None,
        parent:      Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("risk_center_view")
        self._backend    = backend
        self._confirm_fn = _confirm_fn
        self._build()

    # ── Layout ───────────────────────────────────────────────────────────────

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        # Row 1: Loss-Limit-Gauges
        loss_box = QGroupBox("Verlustlimits")
        loss_row = QHBoxLayout(loss_box)
        self._gauge_day   = _LossGauge("Täglich")
        self._gauge_week  = _LossGauge("Wöchentlich")
        self._gauge_month = _LossGauge("Monatlich")
        for g in (self._gauge_day, self._gauge_week, self._gauge_month):
            loss_row.addWidget(g, stretch=1)
        root.addWidget(loss_box)

        # Row 2: Charts (Drawdown | Heatmap | VaR/CVaR)
        charts = QSplitter(Qt.Orientation.Horizontal)

        dd_box = QGroupBox("Drawdown-Verlauf")
        dd_lay = QVBoxLayout(dd_box)
        self._drawdown_canvas = _DrawdownCanvas()
        dd_lay.addWidget(self._drawdown_canvas)
        charts.addWidget(dd_box)

        corr_box = QGroupBox("Korrelationsmatrix")
        corr_lay = QVBoxLayout(corr_box)
        self._heatmap_canvas = _HeatmapCanvas()
        corr_lay.addWidget(self._heatmap_canvas)
        charts.addWidget(corr_box)

        var_box = QGroupBox("VaR / CVaR")
        var_lay = QVBoxLayout(var_box)
        self._var_label      = QLabel("VaR  (95%): –")
        self._cvar_label     = QLabel("CVaR (95%): –")
        self._var_conf_label = QLabel("Konfidenzniveau: 95%")
        for lbl in (self._var_label, self._cvar_label, self._var_conf_label):
            lbl.setObjectName("var_label")
            var_lay.addWidget(lbl)
        var_lay.addStretch()
        charts.addWidget(var_box)

        charts.setStretchFactor(0, 2)
        charts.setStretchFactor(1, 2)
        charts.setStretchFactor(2, 1)
        root.addWidget(charts, stretch=3)

        # Row 3: Active warnings table
        warn_box = QGroupBox("Aktive Warnungen & Eingriffe")
        warn_lay = QVBoxLayout(warn_box)
        self._warnings_table = QTableWidget(0, 3)
        self._warnings_table.setObjectName("warnings_table")
        self._warnings_table.setHorizontalHeaderLabels(["Zeitstempel", "Typ", "Meldung"])
        self._warnings_table.horizontalHeader().setStretchLastSection(True)
        self._warnings_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._warnings_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._warnings_table.verticalHeader().setVisible(False)
        warn_lay.addWidget(self._warnings_table)
        root.addWidget(warn_box, stretch=2)

        # Row 4: Control buttons
        ctrl = QFrame()
        ctrl.setObjectName("risk_controls")
        ctrl_lay = QHBoxLayout(ctrl)
        ctrl_lay.setContentsMargins(0, 0, 0, 0)
        ctrl_lay.setSpacing(8)

        self._pause_btn = QPushButton("⏸  Pause")
        self._pause_btn.setObjectName("pause_btn")
        self._pause_btn.setCheckable(True)
        self._pause_btn.clicked.connect(self._on_pause_resume)

        self._emergency_btn = QPushButton("🚨  Notfall-Stop")
        self._emergency_btn.setObjectName("emergency_btn")
        self._emergency_btn.setProperty("danger", "true")
        self._emergency_btn.clicked.connect(self._on_emergency_stop)

        self._release_btn = QPushButton("✅  Drawdown-Freigabe")
        self._release_btn.setObjectName("release_btn")
        self._release_btn.setEnabled(False)
        self._release_btn.clicked.connect(self._on_release_drawdown)

        ctrl_lay.addWidget(self._pause_btn)
        ctrl_lay.addWidget(self._emergency_btn)
        ctrl_lay.addWidget(self._release_btn)
        ctrl_lay.addStretch()
        root.addWidget(ctrl)

    # ── Public API ────────────────────────────────────────────────────────────

    def set_backend(self, backend: RiskCenterBackend) -> None:
        self._backend = backend

    def refresh(self) -> None:
        """Alle Daten vom Backend neu laden und Anzeige aktualisieren."""
        if self._backend is None:
            return
        self._refresh_loss_limits()
        self._refresh_drawdown()
        self._refresh_correlation()
        self._refresh_var_cvar()
        self._refresh_warnings()
        self._refresh_button_states()

    # ── Private refresh ───────────────────────────────────────────────────────

    def _refresh_loss_limits(self) -> None:
        data  = self._backend.get_loss_summary()
        day   = data.get("daily",   {})
        week  = data.get("weekly",  {})
        month = data.get("monthly", {})
        self._gauge_day  .set_values(day  .get("loss_pct", 0.0), day  .get("limit_pct",  5.0))
        self._gauge_week .set_values(week .get("loss_pct", 0.0), week .get("limit_pct", 10.0))
        self._gauge_month.set_values(month.get("loss_pct", 0.0), month.get("limit_pct", 20.0))

    def _refresh_drawdown(self) -> None:
        self._drawdown_canvas.set_data(self._backend.get_drawdown_history())

    def _refresh_correlation(self) -> None:
        cd = self._backend.get_correlation_data()
        self._heatmap_canvas.set_data(
            symbols   = cd.get("symbols",   []),
            matrix    = cd.get("matrix",    {}),
            threshold = cd.get("threshold", 0.8),
        )

    def _refresh_var_cvar(self) -> None:
        data = self._backend.get_var_cvar()
        var  = data.get("var",        0.0)
        cvar = data.get("cvar",       0.0)
        conf = data.get("confidence", 0.95)
        pct  = int(conf * 100)
        self._var_label     .setText(f"VaR  ({pct}%): {var:.4f}")
        self._cvar_label    .setText(f"CVaR ({pct}%): {cvar:.4f}")
        self._var_conf_label.setText(f"Konfidenzniveau: {pct}%")

    def _refresh_warnings(self) -> None:
        warnings = self._backend.get_active_warnings()
        self._warnings_table.setRowCount(0)
        for w in warnings:
            row = self._warnings_table.rowCount()
            self._warnings_table.insertRow(row)
            ts_item  = QTableWidgetItem(str(w.get("timestamp", "")))
            typ_item = QTableWidgetItem(str(w.get("type",      "")))
            msg_item = QTableWidgetItem(str(w.get("message",   "")))
            typ_upper = str(w.get("type", "")).upper()
            if any(k in typ_upper for k in ("EMERGENCY", "CRITICAL", "STOP")):
                color = QColor("#ef4444")
            elif any(k in typ_upper for k in ("WARN", "DRAWDOWN")):
                color = QColor("#f59e0b")
            else:
                color = None
            if color:
                for item in (ts_item, typ_item, msg_item):
                    item.setForeground(color)
            self._warnings_table.setItem(row, _COL_TS,  ts_item)
            self._warnings_table.setItem(row, _COL_TYP, typ_item)
            self._warnings_table.setItem(row, _COL_MSG, msg_item)

    def _refresh_button_states(self) -> None:
        paused = self._backend.is_trading_paused()
        dd_hit = self._backend.is_max_drawdown_hit()
        self._pause_btn.setChecked(paused)
        self._pause_btn.setText("▶  Resume" if paused else "⏸  Pause")
        self._release_btn.setEnabled(dd_hit)

    # ── Button handlers ───────────────────────────────────────────────────────

    def _on_pause_resume(self, checked: bool) -> None:
        if checked:
            if self._backend is not None:
                self._backend.pause_trading(reason="Manuell durch Operator")
            self._pause_btn.setText("▶  Resume")
            self.trading_paused.emit(True)
        else:
            ok = self._show_confirmation(
                "Handel fortsetzen",
                "Möchten Sie den Handel wirklich fortsetzen?\n"
                "Stellen Sie sicher, dass alle Risikobedingungen erfüllt sind.",
                "Handel fortsetzen",
            )
            if ok:
                if self._backend is not None:
                    self._backend.resume_trading()
                self._pause_btn.setText("⏸  Pause")
                self.trading_paused.emit(False)
            else:
                # Abbrechen: Pause-Zustand beibehalten
                self._pause_btn.setChecked(True)

    def _on_emergency_stop(self) -> None:
        ok = self._show_confirmation(
            "Notfall-Stop",
            "ACHTUNG: Alle offenen Positionen werden sofort geschlossen!\n"
            "Der Handel wird bis zur manuellen Freigabe gestoppt.\n\n"
            "Sind Sie sicher?",
            "Notfall-Stop ausführen",
        )
        if ok:
            if self._backend is not None:
                self._backend.emergency_stop()
            self._pause_btn.setChecked(True)
            self._pause_btn.setText("▶  Resume")
            self.emergency_stopped.emit()

    def _on_release_drawdown(self) -> None:
        ok = self._show_confirmation(
            "Drawdown-Stop aufheben",
            "Der maximale Drawdown-Stop wurde ausgelöst.\n"
            "Die manuelle Freigabe erlaubt das erneute Öffnen von Positionen.\n\n"
            "Haben Sie die Ursache analysiert und sind bereit fortzufahren?",
            "Freigabe bestätigen",
        )
        if ok:
            if self._backend is not None:
                self._backend.release_drawdown_stop()
            self._release_btn.setEnabled(False)
            self.drawdown_released.emit()

    def _show_confirmation(self, title: str, message: str, label: str) -> bool:
        if self._confirm_fn is not None:
            return self._confirm_fn(title, message, label)
        from gui.app import ConfirmationDialog
        return ConfirmationDialog.ask(title, message, label, self)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def gauge_day(self) -> _LossGauge:
        return self._gauge_day

    @property
    def gauge_week(self) -> _LossGauge:
        return self._gauge_week

    @property
    def gauge_month(self) -> _LossGauge:
        return self._gauge_month

    @property
    def drawdown_canvas(self) -> _DrawdownCanvas:
        return self._drawdown_canvas

    @property
    def heatmap_canvas(self) -> _HeatmapCanvas:
        return self._heatmap_canvas

    @property
    def var_label(self) -> QLabel:
        return self._var_label

    @property
    def cvar_label(self) -> QLabel:
        return self._cvar_label

    @property
    def var_conf_label(self) -> QLabel:
        return self._var_conf_label

    @property
    def warnings_table(self) -> QTableWidget:
        return self._warnings_table

    @property
    def pause_button(self) -> QPushButton:
        return self._pause_btn

    @property
    def emergency_button(self) -> QPushButton:
        return self._emergency_btn

    @property
    def release_button(self) -> QPushButton:
        return self._release_btn
