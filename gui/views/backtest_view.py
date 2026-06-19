"""
gui/views/backtest_view.py
Backtest-View: Backtesting-Ergebnisse visuell nachvollziehbar machen.

Layout:
  QSplitter (horizontal)
  ├── Links: Konfigurationsmaske (Symbol, Zeitrahmen, Daten, IS-Split, Kapital)
  │          QProgressBar + [Starten] + [Export Markdown]
  └── Rechts: QTabWidget
              ├── Tab 0 "Ergebnis"   : _EquityCurveCanvas + _MetricsGrid
              ├── Tab 1 "Vergleich"  : _RunsTable (mehrere Laeufe)
              └── Tab 2 "Walk-Fwd"  : _WalkForwardPanel

Backend-Protocol (BacktestBackend):
  run_backtest(symbol, timeframe, start, end, is_split, init_cash) -> BacktestResult
  get_available_symbols() -> list[str]
  export_markdown(result, path) -> None

Testbarkeit:
  _run_fn injizierbar: ersetzt backend.run_backtest (kein echter Backtest in Tests).
  _export_fn injizierbar: ersetzt QFileDialog+Schreiben beim Export.
  _on_result_received() oeffentlich: direkte Ergebnis-Injektion ohne Worker-Thread.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Optional, Protocol, runtime_checkable

import pandas as pd
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.backtesting.vectorbt_runner import BacktestResult


# ─────────────────────────────────────────────────────────────────────────────
#  Backend-Protocol
# ─────────────────────────────────────────────────────────────────────────────

@runtime_checkable
class BacktestBackend(Protocol):
    def run_backtest(
        self,
        symbol:    str,
        timeframe: str,
        start:     str,
        end:       str,
        is_split:  Optional[str],
        init_cash: float,
    ) -> BacktestResult: ...

    def get_available_symbols(self) -> list[str]: ...

    def export_markdown(self, result: BacktestResult, path: str) -> None: ...


# ─────────────────────────────────────────────────────────────────────────────
#  Tooltip-Texte fuer Kennzahlen
# ─────────────────────────────────────────────────────────────────────────────

_METRIC_TOOLTIPS: dict[str, str] = {
    "Gesamtertrag":        "Prozentualer Gesamtertrag (Endwert / Startkapital – 1).",
    "Sharpe Ratio":        "Annualisierter Sharpe Ratio: mittlere Rendite / Stddev × √ann.",
    "Sortino Ratio":       "Wie Sharpe, aber nur Downside-Volatilität im Nenner.",
    "Max. Drawdown":       "Groesster Rueckgang vom Hochpunkt zum Tiefpunkt des Portfolios.",
    "Gewinnfaktor":        "Brutto-Gewinne / Brutto-Verluste. > 1.0 = profitabel.",
    "Win-Rate":            "Anteil der gewinnenden Trades (0 – 100 %).",
    "Ø Gewinn":            "Durchschnittlicher Gewinn je gewinnenden Trade.",
    "Ø Verlust":           "Durchschnittlicher Verlust (negativ) je verlierenden Trade.",
    "Trades":              "Anzahl abgeschlossener Trades im Backtest-Zeitraum.",
    "IS Sharpe":           "Sharpe Ratio im In-Sample-Zeitraum (Trainingsbereich).",
    "OOS Sharpe":          "Sharpe Ratio im Out-of-Sample-Zeitraum (Testbereich).",
}

_TIMEFRAMES = ["M1", "M5", "M15", "M30", "H1", "H4", "D1"]


# ─────────────────────────────────────────────────────────────────────────────
#  _EquityCurveCanvas
# ─────────────────────────────────────────────────────────────────────────────

_IS_COLOR  = QColor(59,  130, 246, 45)   # blau, halbtransparent
_OOS_COLOR = QColor(249, 115, 22,  45)   # orange, halbtransparent
_SPLIT_PEN = QPen(QColor(249, 115, 22), 1, Qt.PenStyle.DashLine)
_CURVE_PEN = QPen(QColor(34,  197, 94),  2)


class _EquityCurveCanvas(QWidget):
    """QPainter-basierte Equity-Curve mit IS/OOS-Hintergrund-Visualisierung."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(200)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._result:   Optional[BacktestResult] = None
        self._is_mask:  Optional[pd.Series]      = None

    # ── Oeffentliche Schnittstelle ────────────────────────────────────────────

    def set_result(
        self,
        result:  Optional[BacktestResult],
        is_mask: Optional[pd.Series] = None,
    ) -> None:
        self._result  = result
        self._is_mask = is_mask
        self.update()

    @property
    def has_data(self) -> bool:
        return self._result is not None and not self._result.equity_curve.empty

    @property
    def is_oos_split_index(self) -> Optional[int]:
        """Index im equity-Array, an dem IS->OOS wechselt (oder None)."""
        if self._is_mask is None or self._result is None:
            return None
        mask  = self._is_mask
        curve = self._result.equity_curve
        n     = len(curve)
        for i in range(n):
            if i < len(mask) and not mask.iloc[i]:
                return i
        return None

    # ── Painting ─────────────────────────────────────────────────────────────

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        bg = QColor(30, 30, 30) if self.palette().window().color().lightness() < 128 else QColor(248, 248, 248)
        p.fillRect(self.rect(), bg)

        if not self.has_data:
            p.setPen(QColor(120, 120, 120))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Noch kein Backtest")
            return

        curve = self._result.equity_curve  # type: ignore[union-attr]
        vals  = curve.values.tolist()
        n     = len(vals)
        if n < 2:
            return

        w, h   = self.width(), self.height()
        pad    = 8
        draw_w = w - 2 * pad
        draw_h = h - 2 * pad

        v_min = min(vals)
        v_max = max(vals)
        v_rng = (v_max - v_min) or 1.0

        def _px(i: int, v: float) -> tuple[float, float]:
            x = pad + i / (n - 1) * draw_w
            y = pad + (1.0 - (v - v_min) / v_rng) * draw_h
            return x, y

        split_idx = self.is_oos_split_index

        # IS-Hintergrund
        if split_idx is not None and split_idx > 0:
            x_split, _ = _px(split_idx, 0)
            p.fillRect(int(pad), int(pad), int(x_split - pad), int(draw_h), _IS_COLOR)
            p.fillRect(int(x_split), int(pad), int(w - x_split - pad), int(draw_h), _OOS_COLOR)
            p.setPen(_SPLIT_PEN)
            p.drawLine(int(x_split), int(pad), int(x_split), int(pad + draw_h))

        # Equity-Kurve
        p.setPen(_CURVE_PEN)
        for i in range(1, n):
            x0, y0 = _px(i - 1, vals[i - 1])
            x1, y1 = _px(i,     vals[i])
            p.drawLine(int(x0), int(y0), int(x1), int(y1))


# ─────────────────────────────────────────────────────────────────────────────
#  _MetricsGrid
# ─────────────────────────────────────────────────────────────────────────────

class _MetricsGrid(QWidget):
    """Raster mit Kennzahlen-Labels und Tooltips."""

    _METRIC_KEYS = [
        ("Gesamtertrag",  lambda r: f"{r.total_return:+.2%}"),
        ("Sharpe Ratio",  lambda r: f"{r.sharpe_ratio:.3f}"),
        ("Sortino Ratio", lambda r: f"{r.sortino_ratio:.3f}"),
        ("Max. Drawdown", lambda r: f"{r.max_drawdown:.2%}"),
        ("Gewinnfaktor",  lambda r: ("∞" if r.profit_factor == float("inf") else f"{r.profit_factor:.2f}")),
        ("Win-Rate",      lambda r: f"{r.win_rate:.1%}"),
        ("Ø Gewinn",      lambda r: f"{r.avg_win:.2f}"),
        ("Ø Verlust",     lambda r: f"{r.avg_loss:.2f}"),
        ("Trades",        lambda r: str(r.n_trades)),
        ("IS Sharpe",     lambda r: f"{r.is_sharpe:.3f}"  if r.is_sharpe  is not None else "–"),
        ("OOS Sharpe",    lambda r: f"{r.oos_sharpe:.3f}" if r.oos_sharpe is not None else "–"),
    ]

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._value_labels: dict[str, QLabel] = {}
        self._build()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        grid = QWidget()
        grid_layout = QFormLayout(grid)
        grid_layout.setSpacing(4)
        grid_layout.setContentsMargins(0, 0, 0, 0)

        for name, _ in self._METRIC_KEYS:
            name_lbl = QLabel(name + ":")
            name_lbl.setToolTip(_METRIC_TOOLTIPS.get(name, ""))
            val_lbl  = QLabel("–")
            val_lbl.setObjectName(f"metric_{name.lower().replace(' ', '_').replace('.', '').replace('ø', 'avg')}")
            val_lbl.setToolTip(_METRIC_TOOLTIPS.get(name, ""))
            val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self._value_labels[name] = val_lbl
            grid_layout.addRow(name_lbl, val_lbl)

        layout.addWidget(grid)
        layout.addStretch()

    # ── Oeffentliche Schnittstelle ────────────────────────────────────────────

    def set_result(self, result: Optional[BacktestResult]) -> None:
        if result is None:
            for lbl in self._value_labels.values():
                lbl.setText("–")
            return
        for name, fmt_fn in self._METRIC_KEYS:
            self._value_labels[name].setText(fmt_fn(result))

    def metric_label(self, name: str) -> QLabel:
        return self._value_labels[name]

    @property
    def metric_names(self) -> list[str]:
        return [name for name, _ in self._METRIC_KEYS]


# ─────────────────────────────────────────────────────────────────────────────
#  _RunsTable
# ─────────────────────────────────────────────────────────────────────────────

_RUN_COLS = ["Name", "Return", "Sharpe", "Sortino", "MaxDD", "WinRate", "Trades", "Overfitting"]


class _RunsTable(QWidget):
    """Tabelle zum Vergleich mehrerer Backtest-Laeufe."""

    run_selected = Signal(int)  # Zeilenindex

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._runs: list[tuple[str, BacktestResult]] = []
        self._build()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._table = QTableWidget(0, len(_RUN_COLS))
        self._table.setHorizontalHeaderLabels(_RUN_COLS)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self._table)

    def add_run(self, name: str, result: BacktestResult) -> None:
        row = self._table.rowCount()
        self._table.insertRow(row)
        self._runs.append((name, result))

        pf = "∞" if result.profit_factor == float("inf") else f"{result.profit_factor:.2f}"
        ov = "⚠ Ja" if result.overfitting_warning else "Nein"
        values = [
            name,
            f"{result.total_return:+.2%}",
            f"{result.sharpe_ratio:.3f}",
            f"{result.sortino_ratio:.3f}",
            f"{result.max_drawdown:.2%}",
            f"{result.win_rate:.1%}",
            str(result.n_trades),
            ov,
        ]
        for col, val in enumerate(values):
            item = QTableWidgetItem(val)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._table.setItem(row, col, item)

    def run_count(self) -> int:
        return self._table.rowCount()

    def clear_runs(self) -> None:
        self._table.setRowCount(0)
        self._runs.clear()

    def get_run(self, index: int) -> Optional[tuple[str, BacktestResult]]:
        if 0 <= index < len(self._runs):
            return self._runs[index]
        return None

    @property
    def table(self) -> QTableWidget:
        return self._table

    def _on_selection_changed(self) -> None:
        rows = self._table.selectionModel().selectedRows()
        if rows:
            self.run_selected.emit(rows[0].row())


# ─────────────────────────────────────────────────────────────────────────────
#  _WalkForwardPanel
# ─────────────────────────────────────────────────────────────────────────────

_WF_COLS = ["Fenster", "IS-Start", "IS-Ende", "OOS-Start", "OOS-Ende", "IS Sharpe", "OOS Sharpe", "Overfitting"]


class _WalkForwardPanel(QWidget):
    """Zeigt Walk-Forward-Fenster in einer Tabelle; Fenster koennen aus Laufen uebernommen werden."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._build()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        info = QLabel(
            "Speichere mehrere Backtest-Laeufe mit IS/OOS-Trennung und vergleiche "
            "die Walk-Forward-Fenster hier."
        )
        info.setWordWrap(True)
        info.setObjectName("wf_info_label")
        layout.addWidget(info)

        self._table = QTableWidget(0, len(_WF_COLS))
        self._table.setHorizontalHeaderLabels(_WF_COLS)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self._table)

    def add_window(
        self,
        window_nr: int,
        is_start:  str,
        is_end:    str,
        oos_start: str,
        oos_end:   str,
        result:    BacktestResult,
    ) -> None:
        row = self._table.rowCount()
        self._table.insertRow(row)
        ov  = "⚠ Ja" if result.overfitting_warning else "Nein"
        is_s  = f"{result.is_sharpe:.3f}"  if result.is_sharpe  is not None else "–"
        oos_s = f"{result.oos_sharpe:.3f}" if result.oos_sharpe is not None else "–"
        for col, val in enumerate([
            str(window_nr), is_start, is_end, oos_start, oos_end, is_s, oos_s, ov
        ]):
            item = QTableWidgetItem(val)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._table.setItem(row, col, item)

    def window_count(self) -> int:
        return self._table.rowCount()

    def clear_windows(self) -> None:
        self._table.setRowCount(0)

    @property
    def table(self) -> QTableWidget:
        return self._table

    @property
    def info_label(self) -> QLabel:
        return self._table.parent().findChild(QLabel, "wf_info_label")  # type: ignore[return-value]


# ─────────────────────────────────────────────────────────────────────────────
#  _BacktestWorker
# ─────────────────────────────────────────────────────────────────────────────

class _BacktestWorker(QThread):
    """Fuehrt BacktestRunner.run() in einem Hintergrund-Thread aus."""

    finished = Signal(object)  # BacktestResult
    failed   = Signal(str)

    def __init__(
        self,
        run_fn: Callable[..., BacktestResult],
        params: dict,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._run_fn = run_fn
        self._params = params

    def run(self) -> None:
        try:
            result = self._run_fn(**self._params)
            self.finished.emit(result)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


# ─────────────────────────────────────────────────────────────────────────────
#  Markdown-Export-Hilfsfunktion
# ─────────────────────────────────────────────────────────────────────────────

def _result_to_markdown(result: BacktestResult, name: str = "Backtest") -> str:
    pf     = "∞" if result.profit_factor == float("inf") else f"{result.profit_factor:.2f}"
    ov     = "Ja ⚠" if result.overfitting_warning else "Nein"
    is_s   = f"{result.is_sharpe:.3f}"  if result.is_sharpe  is not None else "–"
    oos_s  = f"{result.oos_sharpe:.3f}" if result.oos_sharpe is not None else "–"
    lines = [
        f"# {name}",
        "",
        "| Kennzahl | Wert |",
        "|----------|------|",
        f"| Gesamtertrag | {result.total_return:+.2%} |",
        f"| Sharpe Ratio | {result.sharpe_ratio:.3f} |",
        f"| Sortino Ratio | {result.sortino_ratio:.3f} |",
        f"| Max. Drawdown | {result.max_drawdown:.2%} |",
        f"| Gewinnfaktor | {pf} |",
        f"| Win-Rate | {result.win_rate:.1%} |",
        f"| Ø Gewinn | {result.avg_win:.2f} |",
        f"| Ø Verlust | {result.avg_loss:.2f} |",
        f"| Trades | {result.n_trades} |",
        f"| IS Sharpe | {is_s} |",
        f"| OOS Sharpe | {oos_s} |",
        f"| Overfitting-Warnung | {ov} |",
    ]
    return "\n".join(lines) + "\n"


# ─────────────────────────────────────────────────────────────────────────────
#  BacktestView
# ─────────────────────────────────────────────────────────────────────────────

class BacktestView(QWidget):
    """
    Backtest-Ansicht: Eingabemaske, asynchroner Lauf, Ergebnis-Darstellung.

    Parameters
    ----------
    backend    : BacktestBackend-Implementierung (optional).
    _run_fn    : Injectable fuer Tests: ersetzt backend.run_backtest.
                 Signatur: (symbol, timeframe, start, end, is_split, init_cash) -> BacktestResult.
    _export_fn : Injectable fuer Tests: ersetzt QFileDialog + Datei-Schreiben.
                 Signatur: (markdown_text: str) -> None.
    """

    backtest_started  = Signal()
    backtest_finished = Signal(object)   # BacktestResult
    backtest_failed   = Signal(str)

    def __init__(
        self,
        backend:    Optional[BacktestBackend]                   = None,
        _run_fn:    Optional[Callable[..., BacktestResult]]     = None,
        _export_fn: Optional[Callable[[str], None]]             = None,
        parent:     Optional[QWidget]                           = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("backtest_view")

        self._backend    = backend
        self._run_fn     = _run_fn or (backend.run_backtest if backend else None)
        self._export_fn  = _export_fn

        self._current_result: Optional[BacktestResult] = None
        self._run_counter: int = 0
        self._worker: Optional[_BacktestWorker] = None

        self._build()

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        splitter.addWidget(self._build_input_panel())
        splitter.addWidget(self._build_results_panel())
        splitter.setSizes([280, 900])

    def _build_input_panel(self) -> QWidget:
        box = QGroupBox("Konfiguration")
        box.setObjectName("config_panel")
        form = QFormLayout(box)
        form.setSpacing(8)
        form.setContentsMargins(12, 16, 12, 12)

        # Symbol
        self._symbol_input = QLineEdit("EURUSD")
        self._symbol_input.setObjectName("symbol_input")
        self._symbol_input.setPlaceholderText("z.B. EURUSD")
        form.addRow("Symbol:", self._symbol_input)

        # Zeitrahmen
        self._tf_combo = QComboBox()
        self._tf_combo.setObjectName("timeframe_combo")
        self._tf_combo.addItems(_TIMEFRAMES)
        self._tf_combo.setCurrentText("H1")
        form.addRow("Zeitrahmen:", self._tf_combo)

        # Start
        self._start_input = QLineEdit("2023-01-01")
        self._start_input.setObjectName("start_input")
        self._start_input.setPlaceholderText("YYYY-MM-DD")
        form.addRow("Start:", self._start_input)

        # Ende
        self._end_input = QLineEdit("2024-01-01")
        self._end_input.setObjectName("end_input")
        self._end_input.setPlaceholderText("YYYY-MM-DD")
        form.addRow("Ende:", self._end_input)

        # IS-Split
        self._is_split_input = QLineEdit()
        self._is_split_input.setObjectName("is_split_input")
        self._is_split_input.setPlaceholderText("YYYY-MM-DD (optional)")
        self._is_split_input.setToolTip(
            "Trenndate In-Sample / Out-of-Sample. "
            "Leer lassen wenn kein IS/OOS-Split gewuenscht."
        )
        form.addRow("IS-Ende:", self._is_split_input)

        # Startkapital
        self._cash_spinbox = QDoubleSpinBox()
        self._cash_spinbox.setObjectName("init_cash_spinbox")
        self._cash_spinbox.setRange(100.0, 10_000_000.0)
        self._cash_spinbox.setSingleStep(1000.0)
        self._cash_spinbox.setValue(10_000.0)
        self._cash_spinbox.setPrefix("€ ")
        self._cash_spinbox.setDecimals(0)
        form.addRow("Startkapital:", self._cash_spinbox)

        # Progress
        self._progress = QProgressBar()
        self._progress.setObjectName("progress_bar")
        self._progress.setRange(0, 0)   # indeterminate
        self._progress.setVisible(False)
        form.addRow(self._progress)

        # Fehler-Label
        self._error_label = QLabel("")
        self._error_label.setObjectName("error_label")
        self._error_label.setWordWrap(True)
        self._error_label.setStyleSheet("color: #ef4444;")
        self._error_label.setVisible(False)
        form.addRow(self._error_label)

        # Starten-Button
        self._run_btn = QPushButton("▶  Backtest starten")
        self._run_btn.setObjectName("run_button")
        self._run_btn.clicked.connect(self._on_start_clicked)
        form.addRow(self._run_btn)

        # Export-Button
        self._export_btn = QPushButton("⬇  Export Markdown")
        self._export_btn.setObjectName("export_button")
        self._export_btn.setEnabled(False)
        self._export_btn.clicked.connect(self._on_export_clicked)
        form.addRow(self._export_btn)

        return box

    def _build_results_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        # Overfitting-Warnung
        self._overfitting_label = QLabel("⚠  Overfitting-Warnung: IS-Sharpe deutlich besser als OOS-Sharpe.")
        self._overfitting_label.setObjectName("overfitting_label")
        self._overfitting_label.setStyleSheet(
            "background: #7c2d12; color: #fed7aa; padding: 6px; border-radius: 4px;"
        )
        self._overfitting_label.setWordWrap(True)
        self._overfitting_label.setVisible(False)
        layout.addWidget(self._overfitting_label)

        # Tabs
        self._tabs = QTabWidget()
        self._tabs.setObjectName("results_tabs")
        layout.addWidget(self._tabs, stretch=1)

        # Tab 0: Ergebnis
        result_tab = QWidget()
        result_layout = QVBoxLayout(result_tab)
        result_layout.setContentsMargins(4, 4, 4, 4)
        self._equity_canvas = _EquityCurveCanvas()
        self._equity_canvas.setObjectName("equity_canvas")
        self._metrics_grid = _MetricsGrid()
        self._metrics_grid.setObjectName("metrics_grid")
        result_layout.addWidget(self._equity_canvas, stretch=3)
        result_layout.addWidget(self._metrics_grid, stretch=2)
        self._tabs.addTab(result_tab, "Ergebnis")

        # Tab 1: Vergleich
        self._runs_table = _RunsTable()
        self._runs_table.setObjectName("runs_table")
        self._runs_table.run_selected.connect(self._on_run_selected)
        self._tabs.addTab(self._runs_table, "Vergleich")

        # Tab 2: Walk-Forward
        self._wf_panel = _WalkForwardPanel()
        self._wf_panel.setObjectName("walk_forward_panel")
        self._tabs.addTab(self._wf_panel, "Walk-Forward")

        return panel

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _on_start_clicked(self) -> None:
        self._error_label.setVisible(False)
        params = self._collect_params()
        if self._run_fn is None:
            self._show_error("Kein Backend konfiguriert.")
            return

        self._run_btn.setEnabled(False)
        self._progress.setVisible(True)
        self.backtest_started.emit()

        self._worker = _BacktestWorker(self._run_fn, params)
        self._worker.finished.connect(self._on_result_received)
        self._worker.failed.connect(self._on_run_failed)
        self._worker.start()

    def _on_export_clicked(self, _export_fn: Optional[Callable[[str], None]] = None) -> None:
        if self._current_result is None:
            return
        export_fn = _export_fn or self._export_fn
        md = _result_to_markdown(self._current_result, name=f"Backtest #{self._run_counter}")
        if export_fn is not None:
            export_fn(md)
            return
        # Kein _export_fn: QFileDialog (spaet importiert um zirkulaere Abhaengigkeiten zu vermeiden)
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(
            self, "Backtest exportieren", "backtest_report.md",
            "Markdown (*.md);;Alle Dateien (*)"
        )
        if path:
            try:
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(md)
            except OSError as exc:
                self._show_error(f"Export fehlgeschlagen: {exc}")

    # ── Oeffentliche Slot-Methode (direkt testbar ohne Worker) ────────────────

    def _on_result_received(self, result: BacktestResult) -> None:
        self._current_result = result
        self._run_counter += 1

        self._progress.setVisible(False)
        self._run_btn.setEnabled(True)

        # Equity-Canvas aktualisieren
        self._equity_canvas.set_result(result)

        # Metrics aktualisieren
        self._metrics_grid.set_result(result)

        # Overfitting-Warnung
        self._overfitting_label.setVisible(result.overfitting_warning)

        # Export freischalten
        self._export_btn.setEnabled(True)

        # Lauf zur Vergleichstabelle hinzufuegen
        sym = self._symbol_input.text() or "?"
        tf  = self._tf_combo.currentText()
        run_name = f"#{self._run_counter} {sym} {tf}"
        self._runs_table.add_run(run_name, result)

        # Walk-Forward-Fenster eintragen wenn IS/OOS vorhanden
        if result.is_sharpe is not None or result.oos_sharpe is not None:
            self._wf_panel.add_window(
                window_nr=self._run_counter,
                is_start=self._start_input.text(),
                is_end=self._is_split_input.text() or self._end_input.text(),
                oos_start=self._is_split_input.text() or "–",
                oos_end=self._end_input.text(),
                result=result,
            )

        self.backtest_finished.emit(result)

    def _on_run_failed(self, error: str) -> None:
        self._progress.setVisible(False)
        self._run_btn.setEnabled(True)
        self._show_error(f"Backtest-Fehler: {error}")
        self.backtest_failed.emit(error)

    def _on_run_selected(self, index: int) -> None:
        run = self._runs_table.get_run(index)
        if run is None:
            return
        _, result = run
        self._equity_canvas.set_result(result)
        self._metrics_grid.set_result(result)
        self._overfitting_label.setVisible(result.overfitting_warning)
        self._tabs.setCurrentIndex(0)

    # ── Hilfe ─────────────────────────────────────────────────────────────────

    def _collect_params(self) -> dict:
        is_split = self._is_split_input.text().strip() or None
        return {
            "symbol":    self._symbol_input.text().strip(),
            "timeframe": self._tf_combo.currentText(),
            "start":     self._start_input.text().strip(),
            "end":       self._end_input.text().strip(),
            "is_split":  is_split,
            "init_cash": self._cash_spinbox.value(),
        }

    def _show_error(self, msg: str) -> None:
        self._error_label.setText(msg)
        self._error_label.setVisible(True)

    # ── Properties fuer Tests ─────────────────────────────────────────────────

    @property
    def symbol_input(self) -> QLineEdit:
        return self._symbol_input

    @property
    def timeframe_combo(self) -> QComboBox:
        return self._tf_combo

    @property
    def start_input(self) -> QLineEdit:
        return self._start_input

    @property
    def end_input(self) -> QLineEdit:
        return self._end_input

    @property
    def is_split_input(self) -> QLineEdit:
        return self._is_split_input

    @property
    def init_cash_spinbox(self) -> QDoubleSpinBox:
        return self._cash_spinbox

    @property
    def run_button(self) -> QPushButton:
        return self._run_btn

    @property
    def export_button(self) -> QPushButton:
        return self._export_btn

    @property
    def progress_bar(self) -> QProgressBar:
        return self._progress

    @property
    def equity_canvas(self) -> _EquityCurveCanvas:
        return self._equity_canvas

    @property
    def metrics_grid(self) -> _MetricsGrid:
        return self._metrics_grid

    @property
    def runs_table(self) -> _RunsTable:
        return self._runs_table

    @property
    def walk_forward_panel(self) -> _WalkForwardPanel:
        return self._wf_panel

    @property
    def overfitting_label(self) -> QLabel:
        return self._overfitting_label

    @property
    def error_label(self) -> QLabel:
        return self._error_label

    @property
    def results_tabs(self) -> QTabWidget:
        return self._tabs

    @property
    def current_result(self) -> Optional[BacktestResult]:
        return self._current_result
