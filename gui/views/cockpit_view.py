"""
gui/views/cockpit_view.py
CockpitView – zentrale Arbeitsansicht: Chart, Watchlist und Order-Kontrolle.

Layout:
  ┌─────────────────────────────┬──────────────────┐
  │  ChartWidget                │  WatchlistWidget  │
  │  (Kerzen, EMA, BB, SL/TP)   │  (Filter, Signal) │
  ├─────────────────────────────┴──────────────────┤
  │  [Hinweis-Leiste: nicht-blockierende Anfragen] │
  ├────────────────────────────────────────────────┤
  │  Manuelle Order: Symbol | BUY/SELL | Lots | SL | TP | [Aufgeben]   │
  ├────────────────────────────────────────────────┤
  │  Offene Positionen (Ticket | Symbol | Richtung | Lots | [Schliessen])│
  └────────────────────────────────────────────────┘

Backend-Protocol (CockpitBackend):
  fetch_candles(symbol, timeframe, limit) -> list[CandleData]
  get_open_positions()                    -> list[dict]
  get_lot_suggestion(symbol)              -> float
  close_position(ticket)                  -> dict
  update_sl_tp(ticket, sl, tp)           -> dict
  open_position(symbol, direction, lot, sl, tp) -> dict

Testbarkeit:
  _confirm_fn injizierbar (ersetzt ConfirmationDialog.ask)
  Backend via set_backend() setzbar.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Protocol, runtime_checkable

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFrame,
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
#  Positions-Tabellen-Spalten
# ─────────────────────────────────────────────────────────────────────────────

_POS_COL_TICKET = 0
_POS_COL_SYMBOL = 1
_POS_COL_DIR    = 2
_POS_COL_LOTS   = 3
_POS_COL_PRICE  = 4
_POS_COL_SL     = 5
_POS_COL_TP     = 6
_POS_COL_ACTION = 7
_POS_HEADERS    = ["Ticket", "Symbol", "Richtung", "Lots", "Eröffnung", "SL", "TP", "Aktion"]


# ─────────────────────────────────────────────────────────────────────────────
#  CockpitView
# ─────────────────────────────────────────────────────────────────────────────

class CockpitView(QWidget):
    """
    Zentrale Handels-Arbeitsansicht.

    Signals
    -------
    order_submitted(dict)    – emittiert wenn eine manuelle Order aufgegeben wurde.
    position_closed(object)  – emittiert wenn eine Position geschlossen wurde (Ticket).

    Parameters
    ----------
    backend      : CockpitBackend-Implementierung (optional, via set_backend() setzbar).
    _confirm_fn  : Injectable fuer Tests: (title, message, label) -> bool.
                   Standard: ConfirmationDialog.ask() via spaeten Import.
    """

    order_submitted  = Signal(dict)
    position_closed  = Signal(object)

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

        # Richtungs-Buttons
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

        # Lots
        order_row.addWidget(QLabel("Lots:"))
        self._lot_spin = QDoubleSpinBox()
        self._lot_spin.setObjectName("order_lot_spin")
        self._lot_spin.setRange(0.01, 100.0)
        self._lot_spin.setDecimals(2)
        self._lot_spin.setSingleStep(0.01)
        self._lot_spin.setValue(0.01)
        self._lot_spin.setMinimumWidth(70)
        order_row.addWidget(self._lot_spin)

        # SL
        order_row.addWidget(QLabel("SL:"))
        self._sl_spin = QDoubleSpinBox()
        self._sl_spin.setObjectName("order_sl_spin")
        self._sl_spin.setRange(0.0, 999999.0)
        self._sl_spin.setDecimals(5)
        self._sl_spin.setValue(0.0)
        self._sl_spin.setMinimumWidth(80)
        self._sl_spin.setToolTip("Stop-Loss Preis (0 = kein SL)")
        order_row.addWidget(self._sl_spin)

        # TP
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

        # Submit-Button
        self._submit_btn = QPushButton("Order aufgeben")
        self._submit_btn.setObjectName("order_submit_btn")
        self._submit_btn.clicked.connect(self._on_submit_order)
        order_row.addWidget(self._submit_btn)

        order_row.addStretch()
        root.addWidget(order_frame)

        # ── Offene Positionen ─────────────────────────────────────────────────
        pos_label = QLabel("Offene Positionen")
        pos_label.setObjectName("positions_label")
        f2 = pos_label.font()
        f2.setBold(True)
        pos_label.setFont(f2)
        root.addWidget(pos_label)

        self._positions_table = QTableWidget(0, len(_POS_HEADERS))
        self._positions_table.setObjectName("positions_table")
        self._positions_table.setHorizontalHeaderLabels(_POS_HEADERS)
        self._positions_table.setMaximumHeight(160)
        self._positions_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )
        self._positions_table.verticalHeader().setVisible(False)
        self._positions_table.horizontalHeader().setStretchLastSection(True)
        root.addWidget(self._positions_table)

    # ── Oeffentliche Methoden ─────────────────────────────────────────────────

    def set_backend(self, backend: Any) -> None:
        """Setzt oder ersetzt das Backend und aktualisiert die Positionsliste."""
        self._backend = backend
        self._refresh_positions()

    def update_watchlist(self, entries: list[WatchlistEntry]) -> None:
        """Aktualisiert die Watchlist-Daten."""
        self._watchlist.update_entries(entries)

    def show_pending_request(self, message: str) -> None:
        """
        Zeigt einen nicht-blockierenden Hinweis (z.B. Bestaetigungsanfrage
        vom TradingOrchestrator im CONFIRM_REQUIRED-Modus).
        """
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
            # Kerzen laden
            tf = self._chart.current_timeframe.label
            try:
                candles = self._backend.fetch_candles(symbol, tf, limit=200)
                self._chart.set_candles(candles)
                # Approximierte Bid/Ask aus letzter Kerze (ca. 1.5 Pips Spread)
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
                # Approximierte Bid/Ask aus letzter Kerze (ca. 1.5 Pips Spread)
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
        # Spaeter Import um zirkulaere Abhaengigkeit mit gui.app zu vermeiden
        from gui.app import ConfirmationDialog  # noqa: PLC0415
        return ConfirmationDialog.ask(
            title=title, message=message, confirm_label=label, parent=self
        )

    def _refresh_positions(self) -> None:
        if self._backend is None:
            return
        try:
            positions = self._backend.get_open_positions()
        except Exception:  # noqa: BLE001
            return

        self._positions_table.setRowCount(0)
        for pos in positions:
            row = self._positions_table.rowCount()
            self._positions_table.insertRow(row)

            ticket = pos.get("ticket", "")
            symbol = pos.get("symbol", "")

            def _cell(text: str) -> QTableWidgetItem:
                item = QTableWidgetItem(str(text))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                return item

            self._positions_table.setItem(row, _POS_COL_TICKET, _cell(ticket))
            self._positions_table.setItem(row, _POS_COL_SYMBOL, _cell(symbol))
            self._positions_table.setItem(
                row, _POS_COL_DIR, _cell(pos.get("direction", ""))
            )
            self._positions_table.setItem(
                row, _POS_COL_LOTS, _cell(f"{pos.get('lot_size', 0):.2f}")
            )
            self._positions_table.setItem(
                row, _POS_COL_PRICE, _cell(f"{pos.get('open_price', 0):.5f}")
            )
            self._positions_table.setItem(
                row, _POS_COL_SL, _cell(
                    f"{pos.get('sl_price', 0):.5f}"
                    if pos.get("sl_price") else "–"
                )
            )
            self._positions_table.setItem(
                row, _POS_COL_TP, _cell(
                    f"{pos.get('tp_price', 0):.5f}"
                    if pos.get("tp_price") else "–"
                )
            )

            # Schliessen-Button
            close_btn = QPushButton("Schliessen")
            close_btn.setObjectName(f"close_btn_{ticket}")
            close_btn.clicked.connect(
                lambda _c, t=ticket, s=symbol: self._on_close_position(t, s)
            )
            self._positions_table.setCellWidget(row, _POS_COL_ACTION, close_btn)

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
    def positions_table(self) -> QTableWidget:
        return self._positions_table

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
