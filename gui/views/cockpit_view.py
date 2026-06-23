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
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from gui.widgets.chart_widget import CandleData, ChartWidget, Timeframe
from gui.widgets.watchlist_widget import WatchlistEntry, WatchlistWidget
from gui.views.dashboard_view import (
    _DailyStatsCard,
    _PositionsTable,
    _TotalStatsCard,
    DashboardSnapshot,
    PositionInfo,
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

        # ── Heutige Statistiken ───────────────────────────────────────────────
        self._daily_stats = _DailyStatsCard()
        root.addWidget(self._daily_stats)

        # ── Gesamt-Statistiken (seit Teststart) ───────────────────────────────
        self._total_stats = _TotalStatsCard()
        root.addWidget(self._total_stats)

        # ── Offene Positionen (reich: mit P&L + Schliessen-Button) ───────────
        self._pos_rich = _PositionsTable()
        # Schliessen-Button der Tabelle emittiert direkt position_close_requested
        # (kein interner Dialog – externer Handler wie run_gui_bot.py übernimmt)
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
