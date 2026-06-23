"""
gui/views/cockpit_view.py
CockpitView – zentrale Arbeitsansicht: Chart, Watchlist, Order-Kontrolle,
Positionsverwaltung und Performance-Zahlen.

Layout:
  ┌─────────────────────────────┬──────────────────┐
  │  ChartWidget                │  WatchlistWidget  │
  │  (Kerzen, EMA, BB, SL/TP)   │  (Filter, Signal) │
  ├─────────────────────────────┴──────────────────┤
  │  [Hinweis-Leiste: nicht-blockierende Anfragen] │
  ├────────────────────────────────────────────────┤
  │  Manuelle Order: Symbol | BUY/SELL | Lots | SL | TP | [Aufgeben]   │
  ├────────────────────────────────────────────────┤
  │  Heutige Statistiken | Gesamt seit Teststart   │
  ├────────────────────────────────────────────────┤
  │  Offene Positionen (Symbol | Richtg | Lots | Preis | P&L | [Schliessen]) │
  └────────────────────────────────────────────────┘

Backend-Protocol (CockpitBackend):
  fetch_candles(symbol, timeframe, limit) -> list[CandleData]
  get_open_positions()                    -> list[dict]
  get_lot_suggestion(symbol)              -> float
  close_position(ticket)                  -> dict
  update_sl_tp(ticket, sl, tp)           -> dict
  open_position(symbol, direction, lot, sl, tp) -> dict

Watchlist-Befuellung:
  set_watchlist_connector(connector, signal_providers) fuellt die
  4 Standard-Symbole (XAUUSD, EURUSD, USDJPY, GBPUSD) mit Live-Daten
  aus dem MT5Connector und optionalen Modell-Signal-Callbacks.

Testbarkeit:
  _confirm_fn injizierbar (ersetzt ConfirmationDialog.ask)
  Backend via set_backend() setzbar.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Protocol, runtime_checkable

from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFrame,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from gui.widgets.chart_widget import CandleData, ChartWidget, Timeframe
from gui.widgets.watchlist_widget import WatchlistEntry, WatchlistWidget
from gui.views.dashboard_view import (
    DashboardSnapshot,
    PositionInfo,
    _fmt_delta,
    _profit_color,
    _title_label,
)

# Symbole die immer in der Watchlist erscheinen
WATCHLIST_SYMBOLS = ["XAUUSD", "EURUSD", "USDJPY", "GBPUSD"]


# ─────────────────────────────────────────────────────────────────────────────
#  Backend-Protocol
# ─────────────────────────────────────────────────────────────────────────────

@runtime_checkable
class CockpitBackend(Protocol):
    def fetch_candles(
        self, symbol: str, timeframe: str, limit: int = 200
    ) -> list[CandleData]: ...

    def get_open_positions(self) -> list[dict]: ...

    def get_lot_suggestion(self, symbol: str) -> float: ...

    def close_position(self, ticket: Any) -> dict: ...

    def update_sl_tp(
        self, ticket: Any, sl: float | None, tp: float | None
    ) -> dict: ...

    def open_position(
        self,
        symbol:    str,
        direction: str,
        lot_size:  float,
        sl:        float | None,
        tp:        float | None,
    ) -> dict: ...


# ─────────────────────────────────────────────────────────────────────────────
#  Widget: Offene Positionen
# ─────────────────────────────────────────────────────────────────────────────

_POS_HEADERS = ["Symbol", "Richtung", "Lots", "Eröffnung", "P&L", ""]

_COL_CLOSE = len(_POS_HEADERS) - 1


class _PositionsTable(QFrame):
    close_requested = Signal(int)  # ticket

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("dashboard_card")
        self.setFrameShape(QFrame.Shape.StyledPanel)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 12, 16, 14)
        lay.setSpacing(6)

        lay.addWidget(_title_label("Offene Positionen"))

        self._table = QTableWidget(0, len(_POS_HEADERS))
        self._table.setObjectName("positions_table")
        self._table.setHorizontalHeaderLabels(_POS_HEADERS)
        hdr = self._table.horizontalHeader()
        for col in range(_COL_CLOSE - 1):
            hdr.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(_COL_CLOSE - 1, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(_COL_CLOSE, QHeaderView.ResizeMode.Fixed)
        self._table.setColumnWidth(_COL_CLOSE, 90)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setAlternatingRowColors(True)
        self._table.verticalHeader().setVisible(False)
        self._table.setMinimumHeight(100)
        lay.addWidget(self._table)

    def refresh(self, snap: DashboardSnapshot) -> None:
        positions = snap.positions
        self._table.setRowCount(len(positions))

        for row, pos in enumerate(positions):
            pnl_text = _fmt_delta(pos.current_pnl, snap.currency)

            direction_text = pos.direction.upper()
            if pos.break_even_active:
                direction_text += " [BE]"

            cells = [
                pos.symbol,
                direction_text,
                f"{pos.lot_size:.2f}",
                f"{pos.open_price:.5f}" if pos.open_price is not None else "—",
                pnl_text,
            ]
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if col == 0:
                    item.setData(Qt.ItemDataRole.UserRole, pos.ticket)
                    if pos.crv is not None:
                        sl_str = f"SL: {pos.sl_price:.5f}" if pos.sl_price is not None else ""
                        tp_str = f"TP: {pos.tp_price:.5f}" if pos.tp_price is not None else ""
                        item.setToolTip(
                            f"CRV: {pos.crv:.1f} | {sl_str} | {tp_str}".strip(" |")
                        )
                if col == 4 and pos.current_pnl is not None:
                    item.setForeground(QColor(_profit_color(pos.current_pnl)))
                self._table.setItem(row, col, item)

            ticket = pos.ticket
            btn = QPushButton("Schließen")
            btn.setStyleSheet(
                "QPushButton { background:#ef4444; color:white; font-size:11px;"
                " border-radius:3px; padding:2px 6px; }"
                "QPushButton:hover { background:#dc2626; }"
            )
            btn.clicked.connect(lambda _checked=False, t=ticket: self.close_requested.emit(t))
            self._table.setCellWidget(row, _COL_CLOSE, btn)

    def add_position(self, pos: dict) -> None:
        row = self._table.rowCount()
        self._table.insertRow(row)

        open_price = pos.get("open_price") or 0.0
        ticket = pos.get("ticket")
        cells = [
            pos.get("symbol", ""),
            (pos.get("direction", "")).upper(),
            f"{pos.get('lot_size', 0):.2f}",
            f"{open_price:.5f}",
            "–",
        ]
        for col, text in enumerate(cells):
            item = QTableWidgetItem(text)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if col == 0:
                item.setData(Qt.ItemDataRole.UserRole, ticket)
            self._table.setItem(row, col, item)

        btn = QPushButton("Schließen")
        btn.setStyleSheet(
            "QPushButton { background:#ef4444; color:white; font-size:11px;"
            " border-radius:3px; padding:2px 6px; }"
            "QPushButton:hover { background:#dc2626; }"
        )
        btn.clicked.connect(lambda _checked=False, t=ticket: self.close_requested.emit(t))
        self._table.setCellWidget(row, _COL_CLOSE, btn)

        self._highlight_row(row)

    def remove_position(self, ticket) -> None:
        for row in range(self._table.rowCount()):
            item = self._table.item(row, 0)
            if item is not None and item.data(Qt.ItemDataRole.UserRole) == ticket:
                self._table.removeRow(row)
                return

    def _highlight_row(self, row: int) -> None:
        highlight = QColor("#fef3c7")
        for col in range(self._table.columnCount()):
            item = self._table.item(row, col)
            if item is not None:
                item.setBackground(highlight)
        QTimer.singleShot(2000, lambda r=row: self._clear_row_highlight(r))

    def _clear_row_highlight(self, row: int) -> None:
        try:
            if row >= self._table.rowCount():
                return
            transparent = QColor("transparent")
            for col in range(self._table.columnCount()):
                item = self._table.item(row, col)
                if item is not None:
                    item.setBackground(transparent)
        except RuntimeError:
            pass

    @property
    def table(self) -> QTableWidget:
        return self._table


# ─────────────────────────────────────────────────────────────────────────────
#  Widget: Tages-Statistiken
# ─────────────────────────────────────────────────────────────────────────────

class _DailyStatsCard(QFrame):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("dashboard_card")
        self.setFrameShape(QFrame.Shape.StyledPanel)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 12, 16, 14)
        lay.setSpacing(6)

        lay.addWidget(_title_label("Heutige Statistiken"))

        row = QHBoxLayout()
        row.setSpacing(24)

        def _stat(label: str, obj_name: str, tooltip: str) -> QLabel:
            col = QVBoxLayout()
            lbl_title = QLabel(label)
            lbl_title.setProperty("secondary", "true")
            lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
            val = QLabel("--")
            val.setObjectName(obj_name)
            val.setAlignment(Qt.AlignmentFlag.AlignCenter)
            val.setToolTip(tooltip)
            f = val.font()
            f.setPointSize(14)
            f.setBold(True)
            val.setFont(f)
            col.addWidget(lbl_title)
            col.addWidget(val)
            row.addLayout(col)
            return val

        self._trades_lbl  = _stat("Trades heute", "stat_trades",   "Anzahl Trades heute")
        self._pnl_lbl     = _stat("Tages-P&L",    "stat_day_pnl",  "Realisierter P&L heute")
        self._winrate_lbl = _stat("Win-Rate",      "stat_win_rate", "Anteil erfolgreicher Trades heute")

        lay.addLayout(row)

    def refresh(self, snap: DashboardSnapshot) -> None:
        self._trades_lbl.setText(str(snap.today_trades))

        pnl_text = _fmt_delta(snap.today_pnl, snap.currency)
        self._pnl_lbl.setText(pnl_text)
        self._pnl_lbl.setStyleSheet(f"color: {_profit_color(snap.today_pnl)};")

        if snap.today_win_rate is not None:
            self._winrate_lbl.setText(f"{snap.today_win_rate * 100:.1f}%")
        else:
            self._winrate_lbl.setText("--")

    @property
    def trades_label(self) -> QLabel:
        return self._trades_lbl

    @property
    def pnl_label(self) -> QLabel:
        return self._pnl_lbl

    @property
    def winrate_label(self) -> QLabel:
        return self._winrate_lbl


# ─────────────────────────────────────────────────────────────────────────────
#  Widget: Gesamt-Statistiken (realisierter P&L seit Teststart)
# ─────────────────────────────────────────────────────────────────────────────

class _TotalStatsCard(QFrame):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("dashboard_card")
        self.setFrameShape(QFrame.Shape.StyledPanel)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 12, 16, 14)
        lay.setSpacing(6)

        lay.addWidget(_title_label("Gesamt seit Teststart"))

        row = QHBoxLayout()
        row.setSpacing(24)

        def _stat(label: str, obj_name: str, tooltip: str) -> QLabel:
            col = QVBoxLayout()
            lbl_title = QLabel(label)
            lbl_title.setProperty("secondary", "true")
            lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
            val = QLabel("--")
            val.setObjectName(obj_name)
            val.setAlignment(Qt.AlignmentFlag.AlignCenter)
            val.setToolTip(tooltip)
            f = val.font()
            f.setPointSize(14)
            f.setBold(True)
            val.setFont(f)
            col.addWidget(lbl_title)
            col.addWidget(val)
            row.addLayout(col)
            return val

        self._profit_lbl = _stat(
            "Gesamt-Gewinn", "stat_total_profit",
            "Summe aller realisierten Gewinne (geschlossene Positionen)",
        )
        self._loss_lbl = _stat(
            "Gesamt-Verlust", "stat_total_loss",
            "Summe aller realisierten Verluste (geschlossene Positionen)",
        )
        lay.addLayout(row)

    def refresh(self, snap: DashboardSnapshot) -> None:
        profit_text = _fmt_delta(snap.total_gross_profit, snap.currency)
        self._profit_lbl.setText(profit_text)
        self._profit_lbl.setStyleSheet(f"color: {_profit_color(snap.total_gross_profit)};")

        loss_text = _fmt_delta(snap.total_gross_loss, snap.currency)
        self._loss_lbl.setText(loss_text)
        self._loss_lbl.setStyleSheet(f"color: {_profit_color(snap.total_gross_loss)};")

    @property
    def profit_label(self) -> QLabel:
        return self._profit_lbl

    @property
    def loss_label(self) -> QLabel:
        return self._loss_lbl


# ─────────────────────────────────────────────────────────────────────────────
#  CockpitView
# ─────────────────────────────────────────────────────────────────────────────

class CockpitView(QWidget):
    """
    Zentrale Handels-Arbeitsansicht.

    Signals
    -------
    order_submitted(dict)           – emittiert wenn eine manuelle Order aufgegeben wurde.
    position_closed(object)         – emittiert wenn eine Position bestaetigt geschlossen wurde.
    position_close_requested(int)   – emittiert wenn der Nutzer den Schliessen-Button drueckt
                                      (ohne Dialog, fuer externe Handler wie run_gui_bot.py).

    Parameters
    ----------
    backend      : CockpitBackend-Implementierung (optional, via set_backend() setzbar).
    _confirm_fn  : Injectable fuer Tests: (title, message, label) -> bool.
                   Standard: ConfirmationDialog.ask() via spaeten Import.
    """

    order_submitted         = Signal(dict)
    position_closed         = Signal(object)
    position_close_requested = Signal(int)   # direkt aus Tabellen-Button, kein Dialog

    def __init__(
        self,
        backend:     Optional[Any]                          = None,
        _confirm_fn: Optional[Callable[[str, str, str], bool]] = None,
        parent:      Optional[QWidget]                      = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("cockpit_view")

        self._backend     = backend
        self._confirm_fn  = _confirm_fn
        self._active_sym  = ""

        # Watchlist-Datenquellen (optional, via set_watchlist_connector)
        self._watchlist_connector = None
        self._signal_providers: dict[str, Callable] = {}
        self._watchlist_timer = QTimer(self)
        self._watchlist_timer.timeout.connect(self._refresh_watchlist)

        self._build()

        if backend is not None:
            self._refresh_positions()

    # ── Aufbau ────────────────────────────────────────────────────────────────

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Haupt-Splitter: Chart (links) / Watchlist (rechts) ────────────────
        self._splitter = QSplitter(Qt.Orientation.Horizontal)

        # Chart
        self._chart = ChartWidget()
        self._chart.setMinimumHeight(280)
        self._chart.timeframe_changed.connect(self._on_timeframe_changed)
        self._splitter.addWidget(self._chart)

        # Watchlist
        self._watchlist = WatchlistWidget()
        self._watchlist.symbol_selected.connect(self._on_symbol_selected)
        self._splitter.addWidget(self._watchlist)

        self._splitter.setStretchFactor(0, 2)
        self._splitter.setStretchFactor(1, 1)
        root.addWidget(self._splitter, stretch=1)

        # ── Trennlinie ────────────────────────────────────────────────────────
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        root.addWidget(sep)

        # ── Nicht-blockierender Hinweis (z.B. Bestaetigungsanfrage) ──────────
        self._hint_frame = QFrame()
        self._hint_frame.setObjectName("cockpit_hint_frame")
        self._hint_frame.setFrameShape(QFrame.Shape.NoFrame)
        hint_row = QHBoxLayout(self._hint_frame)
        hint_row.setContentsMargins(12, 6, 12, 6)

        self._hint_label = QLabel()
        self._hint_label.setObjectName("cockpit_hint_label")
        self._hint_label.setWordWrap(True)
        hint_row.addWidget(self._hint_label, stretch=1)

        self._hint_dismiss_btn = QPushButton("✕")
        self._hint_dismiss_btn.setObjectName("hint_dismiss_btn")
        self._hint_dismiss_btn.setMaximumWidth(30)
        self._hint_dismiss_btn.clicked.connect(self.hide_pending_request)
        hint_row.addWidget(self._hint_dismiss_btn)

        self._hint_frame.hide()
        root.addWidget(self._hint_frame)

        # ── Order-Panel ───────────────────────────────────────────────────────
        order_frame = QFrame()
        order_frame.setObjectName("order_panel")
        order_row = QHBoxLayout(order_frame)
        order_row.setContentsMargins(12, 8, 12, 8)
        order_row.setSpacing(8)

        order_row.addWidget(QLabel("Symbol:"))
        self._sym_label = QLabel("–")
        self._sym_label.setObjectName("order_symbol_label")
        f = self._sym_label.font()
        f.setBold(True)
        self._sym_label.setFont(f)
        order_row.addWidget(self._sym_label)

        order_row.addSpacing(8)

        self._buy_btn = QPushButton("BUY")
        self._buy_btn.setObjectName("order_buy_btn")
        self._buy_btn.setCheckable(True)
        self._buy_btn.setChecked(True)
        self._buy_btn.clicked.connect(self._on_buy_clicked)
        order_row.addWidget(self._buy_btn)

        self._sell_btn = QPushButton("SELL")
        self._sell_btn.setObjectName("order_sell_btn")
        self._sell_btn.setCheckable(True)
        self._sell_btn.clicked.connect(self._on_sell_clicked)
        order_row.addWidget(self._sell_btn)

        order_row.addSpacing(8)

        order_row.addWidget(QLabel("Lots:"))
        self._lot_spin = QDoubleSpinBox()
        self._lot_spin.setObjectName("order_lot_spin")
        self._lot_spin.setRange(0.01, 100.0)
        self._lot_spin.setDecimals(2)
        self._lot_spin.setSingleStep(0.01)
        self._lot_spin.setValue(0.01)
        self._lot_spin.setMinimumWidth(70)
        order_row.addWidget(self._lot_spin)

        order_row.addWidget(QLabel("SL:"))
        self._sl_spin = QDoubleSpinBox()
        self._sl_spin.setObjectName("order_sl_spin")
        self._sl_spin.setRange(0.0, 999999.0)
        self._sl_spin.setDecimals(5)
        self._sl_spin.setValue(0.0)
        self._sl_spin.setMinimumWidth(80)
        self._sl_spin.setToolTip("Stop-Loss Preis (0 = kein SL)")
        order_row.addWidget(self._sl_spin)

        order_row.addWidget(QLabel("TP:"))
        self._tp_spin = QDoubleSpinBox()
        self._tp_spin.setObjectName("order_tp_spin")
        self._tp_spin.setRange(0.0, 999999.0)
        self._tp_spin.setDecimals(5)
        self._tp_spin.setValue(0.0)
        self._tp_spin.setMinimumWidth(80)
        self._tp_spin.setToolTip("Take-Profit Preis (0 = kein TP)")
        order_row.addWidget(self._tp_spin)

        order_row.addSpacing(8)

        self._submit_btn = QPushButton("Order aufgeben")
        self._submit_btn.setObjectName("order_submit_btn")
        self._submit_btn.clicked.connect(self._on_submit_order)
        order_row.addWidget(self._submit_btn)

        order_row.addStretch()
        root.addWidget(order_frame)

        # ── Statistiken: Heute + Gesamt nebeneinander ─────────────────────────
        stats_row = QHBoxLayout()
        stats_row.setSpacing(12)
        self._daily_stats = _DailyStatsCard()
        self._total_stats = _TotalStatsCard()
        stats_row.addWidget(self._daily_stats)
        stats_row.addWidget(self._total_stats)
        root.addLayout(stats_row)

        # ── Offene Positionen ─────────────────────────────────────────────────
        self._pos_rich = _PositionsTable()
        self._pos_rich.close_requested.connect(self.position_close_requested)
        root.addWidget(self._pos_rich)

    # ── Oeffentliche Methoden ─────────────────────────────────────────────────

    def set_backend(self, backend: Any) -> None:
        """Setzt oder ersetzt das Backend und aktualisiert die Positionsliste."""
        self._backend = backend
        self._refresh_positions()

    def update_watchlist(self, entries: list[WatchlistEntry]) -> None:
        """Aktualisiert die Watchlist-Daten (manuell, ohne Connector)."""
        self._watchlist.update_entries(entries)

    def update_trading_stats(self, snap: DashboardSnapshot) -> None:
        """
        Aktualisiert Positionen (mit P&L), Tages- und Gesamtstatistiken
        aus einem DashboardSnapshot.

        Wird typischerweise vom MainWindow nach jedem Dashboard-Polling-Tick
        aufgerufen, um Cockpit-Daten synchron zu halten.
        """
        self._pos_rich.refresh(snap)
        self._daily_stats.refresh(snap)
        self._total_stats.refresh(snap)

    def set_watchlist_connector(
        self,
        connector: Any,
        signal_providers: dict[str, Callable] | None = None,
    ) -> None:
        """
        Verbindet einen MT5Connector fuer Live-Watchlist-Daten.

        Parameters
        ----------
        connector        : MT5Connector-Instanz mit get_tick() und get_ohlcv_count().
        signal_providers : dict {symbol: callable() -> {"signal": str, "confidence": float}}
                           Symbole ohne Eintrag erhalten signal="kein modell".
        """
        self._watchlist_connector = connector
        self._signal_providers = signal_providers or {}
        self._refresh_watchlist()
        if not self._watchlist_timer.isActive():
            self._watchlist_timer.start(5_000)

    def show_pending_request(self, message: str) -> None:
        """Zeigt einen nicht-blockierenden Hinweis-Banner."""
        self._hint_label.setText(message)
        self._hint_frame.show()

    def hide_pending_request(self) -> None:
        """Versteckt den Hinweis-Banner."""
        self._hint_frame.hide()

    def refresh_positions(self) -> None:
        """Laedt offene Positionen vom Backend und aktualisiert die Tabelle."""
        self._refresh_positions()

    # ── Signal-Handler ────────────────────────────────────────────────────────

    def _on_symbol_selected(self, symbol: str) -> None:
        self._active_sym = symbol
        self._sym_label.setText(symbol)
        self._chart.set_symbol(symbol)
        if self._backend is not None:
            try:
                lot = self._backend.get_lot_suggestion(symbol)
                self._lot_spin.setValue(lot)
            except Exception:  # noqa: BLE001
                pass
            tf = self._chart.current_timeframe.label
            try:
                candles = self._backend.fetch_candles(symbol, tf, limit=200)
                self._chart.set_candles(candles)
                if candles:
                    last = candles[-1]
                    approx_spread = 0.00015
                    self._chart.set_bid_ask(
                        last.close - approx_spread / 2,
                        last.close + approx_spread / 2,
                    )
            except Exception:  # noqa: BLE001
                pass

    def _on_timeframe_changed(self, tf: Timeframe) -> None:
        if self._backend is not None and self._active_sym:
            try:
                candles = self._backend.fetch_candles(
                    self._active_sym, tf.label, limit=200
                )
                self._chart.set_candles(candles)
                if candles:
                    last = candles[-1]
                    approx_spread = 0.00015
                    self._chart.set_bid_ask(
                        last.close - approx_spread / 2,
                        last.close + approx_spread / 2,
                    )
            except Exception:  # noqa: BLE001
                pass

    def _on_buy_clicked(self) -> None:
        self._buy_btn.setChecked(True)
        self._sell_btn.setChecked(False)

    def _on_sell_clicked(self) -> None:
        self._sell_btn.setChecked(True)
        self._buy_btn.setChecked(False)

    def _on_submit_order(self) -> None:
        if not self._active_sym or self._backend is None:
            return
        direction = "buy" if self._buy_btn.isChecked() else "sell"
        lot       = self._lot_spin.value()
        sl        = self._sl_spin.value() or None
        tp        = self._tp_spin.value() or None

        result = self._backend.open_position(
            self._active_sym, direction, lot, sl, tp
        )
        self.order_submitted.emit(result)
        self._refresh_positions()

    def _on_close_position(self, ticket: Any, symbol: str) -> None:
        """Zeigt Bestaetigung und schliesst Position via Backend (fuer Tests)."""
        confirmed = self._show_confirmation(
            title="Position schliessen",
            message=(
                f"Position {ticket} ({symbol}) wirklich schliessen?\n"
                "Diese Aktion ist unwiderruflich."
            ),
            label="Schliessen",
        )
        if confirmed and self._backend is not None:
            self._backend.close_position(ticket)
            self.position_closed.emit(ticket)
            self._refresh_positions()

    def _show_confirmation(self, title: str, message: str, label: str) -> bool:
        if self._confirm_fn is not None:
            return self._confirm_fn(title, message, label)
        from gui.app import ConfirmationDialog  # noqa: PLC0415
        return ConfirmationDialog.ask(
            title=title, message=message, confirm_label=label, parent=self
        )

    def _refresh_positions(self) -> None:
        """Holt Positionen vom Backend und fuettert die reiche Positions-Tabelle."""
        if self._backend is None:
            return
        try:
            positions = self._backend.get_open_positions()
        except Exception:  # noqa: BLE001
            return
        pos_infos = [
            PositionInfo(
                ticket=p.get("ticket", ""),
                symbol=p.get("symbol", ""),
                direction=p.get("direction", ""),
                lot_size=float(p.get("lot_size", 0.0)),
                open_price=p.get("open_price"),
                current_pnl=p.get("current_pnl"),
                sl_price=p.get("sl_price"),
                tp_price=p.get("tp_price"),
                break_even_active=bool(p.get("break_even_active", False)),
            )
            for p in positions
        ]
        self._pos_rich.refresh(DashboardSnapshot(positions=pos_infos))

    @Slot()
    def _refresh_watchlist(self) -> None:
        """Holt Live-Bid/Ask, Tagesveraenderung und Modell-Signale fuer alle Watchlist-Symbole."""
        if self._watchlist_connector is None:
            return
        entries: list[WatchlistEntry] = []
        for sym in WATCHLIST_SYMBOLS:
            bid: float | None = None
            ask: float | None = None
            change: float | None = None
            signal = "flat"
            conf = 0.0

            try:
                tick = self._watchlist_connector.get_tick(sym)
                bid = float(tick.get("bid", 0)) or None
                ask = float(tick.get("ask", 0)) or None
            except Exception:  # noqa: BLE001
                pass

            try:
                df = self._watchlist_connector.get_ohlcv_count(sym, "D1", count=2)
                if df is not None and len(df) >= 1:
                    day_open = float(df.iloc[-1]["open"])
                    if day_open:
                        mid = ((bid or 0) + (ask or 0)) / 2 or day_open
                        change = (mid - day_open) / day_open * 100
            except Exception:  # noqa: BLE001
                pass

            if sym in self._signal_providers:
                try:
                    result = self._signal_providers[sym]()
                    signal = result.get("signal", "flat")
                    conf = float(result.get("confidence", 0.0))
                except Exception:  # noqa: BLE001
                    signal = "flat"
            else:
                signal = "kein modell"

            entries.append(WatchlistEntry(
                symbol=sym,
                bid=bid,
                ask=ask,
                daily_change_pct=change,
                signal=signal,
                signal_confidence=conf,
            ))
        self._watchlist.update_entries(entries)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def chart_widget(self) -> ChartWidget:
        return self._chart

    @property
    def watchlist_widget(self) -> WatchlistWidget:
        return self._watchlist

    @property
    def pending_hint_label(self) -> QLabel:
        return self._hint_label

    @property
    def pending_hint_frame(self) -> QFrame:
        return self._hint_frame

    @property
    def positions_table(self):
        """QTableWidget der reichen Positions-Tabelle (fuer Tests und externe Zugriffe)."""
        return self._pos_rich.table

    @property
    def daily_stats_card(self) -> _DailyStatsCard:
        return self._daily_stats

    @property
    def total_stats_card(self) -> _TotalStatsCard:
        return self._total_stats

    @property
    def lot_spinbox(self) -> QDoubleSpinBox:
        return self._lot_spin

    @property
    def sl_spinbox(self) -> QDoubleSpinBox:
        return self._sl_spin

    @property
    def tp_spinbox(self) -> QDoubleSpinBox:
        return self._tp_spin

    @property
    def buy_button(self) -> QPushButton:
        return self._buy_btn

    @property
    def sell_button(self) -> QPushButton:
        return self._sell_btn

    @property
    def submit_button(self) -> QPushButton:
        return self._submit_btn

    @property
    def active_symbol(self) -> str:
        return self._active_sym

    def connect_order_executor(self, relay) -> None:
        """
        Verbindet einen OrderEventRelay fuer sofortige Positions-Updates.

        Aktualisiert die Positions-Tabelle unmittelbar bei jeder Order-Aktion
        des Bots, ohne auf das naechste manuelle Refresh zu warten.
        """
        relay.order_opened.connect(self.on_order_opened)
        relay.order_closed.connect(self.on_order_closed)

    @Slot(dict)
    def on_order_opened(self, order: dict) -> None:  # noqa: ARG002
        """Aktualisiert die Positions-Tabelle nach einer neuen Bot-Order."""
        self._refresh_positions()

    @Slot(dict)
    def on_order_closed(self, order: dict) -> None:  # noqa: ARG002
        """Aktualisiert die Positions-Tabelle nach einer Bot-Schliessen."""
        self._refresh_positions()
