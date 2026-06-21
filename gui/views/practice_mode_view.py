"""
gui/views/practice_mode_view.py
PracticeModeView – Manueller Uebungsmodus auf historischen Daten.

Architektur (Backend-Protocol-Pattern):
  PracticeBackend : Protocol – jedes Objekt mit den noetigten Methoden.
  _DefaultPracticeBackend : No-Op-Implementierung (fuer Tests ohne Backend).
  PracticeModeView : reines Qt-Widget, keinerlei Geschaeftslogik.

Features:
  - Historischer Zeitraum frei waehlbar (letzte Woche / letzter Monat / benutzerdefiniert)
  - Vorwaerts-Wiedergabe: manuell (Kerze fuer Kerze) oder automatisch (1x, 5x, 10x)
  - Kauf/Verkauf-Buttons mit SL, TP, Lot-Eingabe
  - KEINE Verbindung zu echten Order-Systemen – rein simulierte Uebungs-Positionen
  - Ergebnis-Anzeige nach Schliessen (P&L + Counterfactual)
  - Eigene Statistiken (getrennt vom echten TradeJournal)
  - No-Lookahead: angezeigte Kerzenanzahl entspricht Cursor + 1, nie der Gesamtzahl
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional, Protocol, runtime_checkable

from PySide6.QtCore import QDate, Qt, QTimer, Signal, Slot
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QComboBox,
    QDateEdit,
    QDoubleSpinBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from loguru import logger


# ─────────────────────────────────────────────────────────────────────────────
#  Backend-Protocol
# ─────────────────────────────────────────────────────────────────────────────

@runtime_checkable
class PracticeBackend(Protocol):
    """
    Protocol fuer das Practice-Mode-Backend.

    Jede Klasse die diese Methoden implementiert kann als Backend verwendet
    werden – kein Erben notwendig (structural subtyping).
    """

    def load_session(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
    ) -> dict:
        """
        Laedt Kerzen fuer den angegebenen Zeitraum und bereitet die Session vor.

        Returns
        -------
        dict : {"total_candles": int, "cursor": int, "current_candle": dict | None}
        """
        ...  # pragma: no cover

    def advance(self, steps: int = 1) -> dict:
        """
        Rueckt die Session um `steps` Kerzen vor.

        Returns
        -------
        dict : {"cursor": int, "current_candle": dict, "is_at_end": bool,
                "auto_closed": list[dict]}
        """
        ...  # pragma: no cover

    def open_position(
        self,
        direction: str,
        lot_size: float,
        sl: float,
        tp: float,
    ) -> int:
        """Oeffnet eine simulierte Position. Returns position_id."""
        ...  # pragma: no cover

    def close_position(self, position_id: int) -> dict:
        """
        Schliesst eine Position.

        Returns
        -------
        dict : {"pnl": float, "counterfactual_pnl": float, "closed_by": str}
        """
        ...  # pragma: no cover

    def get_open_positions(self) -> list[dict]:
        """Returns list of open position dicts."""
        ...  # pragma: no cover

    def get_stats(self) -> dict:
        """Returns {"trade_count": int, "wins": int, "losses": int,
                    "win_rate": float, "total_pnl": float}."""
        ...  # pragma: no cover

    def suggest_lot_size(self, account_balance: float, atr: float) -> float:
        """Schlaegt eine Lot-Groesse via PositionSizer vor."""
        ...  # pragma: no cover


class _DefaultPracticeBackend:
    """No-Op-Backend fuer Tests und Standalone-Verwendung."""

    def load_session(self, symbol, timeframe, start_date, end_date):
        return {"total_candles": 0, "cursor": 0, "current_candle": None}

    def advance(self, steps=1):
        return {"cursor": 0, "current_candle": None, "is_at_end": True, "auto_closed": []}

    def open_position(self, direction, lot_size, sl, tp):
        return 0

    def close_position(self, position_id):
        return {"pnl": 0.0, "counterfactual_pnl": 0.0, "closed_by": "manual"}

    def get_open_positions(self):
        return []

    def get_stats(self):
        return {"trade_count": 0, "wins": 0, "losses": 0, "win_rate": 0.0, "total_pnl": 0.0}

    def suggest_lot_size(self, account_balance, atr):
        return 0.01


# ─────────────────────────────────────────────────────────────────────────────
#  Hilfsfunktionen
# ─────────────────────────────────────────────────────────────────────────────

def _pnl_color(pnl: float) -> str:
    return "#27ae60" if pnl >= 0 else "#c0392b"


def _fmt_price(v: Optional[float]) -> str:
    if v is None:
        return "—"
    return f"{v:.5f}"


def _fmt_pnl(v: Optional[float]) -> str:
    if v is None:
        return "—"
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.2f}"


# ─────────────────────────────────────────────────────────────────────────────
#  PracticeModeView
# ─────────────────────────────────────────────────────────────────────────────

class PracticeModeView(QWidget):
    """
    Haupt-Widget fuer den manuellen Uebungsmodus.

    Signale
    -------
    session_started     : Neue Session geladen.
    position_opened(int): position_id der geoeffneten Uebungs-Position.
    position_closed(dict): Ergebnis-Dict der geschlossenen Position.
    """

    session_started    = Signal()
    position_opened    = Signal(int)
    position_closed    = Signal(dict)

    # Verfuegbare Abspiel-Geschwindigkeiten: (Label, Millisekunden pro Tick)
    _SPEEDS: list[tuple[str, int]] = [
        ("1x",  1000),
        ("5x",   200),
        ("10x",  100),
    ]

    def __init__(
        self,
        backend: Optional[object] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._backend: PracticeBackend = backend or _DefaultPracticeBackend()  # type: ignore[assignment]
        self._session_loaded: bool = False
        self._auto_timer = QTimer(self)
        self._auto_timer.timeout.connect(self._on_auto_advance)
        self._open_position_ids: list[int] = []

        self._build_ui()
        self._update_controls_enabled()

    # ── UI-Aufbau ─────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(8)
        root.setContentsMargins(8, 8, 8, 8)

        root.addWidget(self._build_setup_panel())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_candle_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([420, 320])
        root.addWidget(splitter, 1)

        root.addWidget(self._build_positions_panel())
        root.addWidget(self._build_stats_panel())

    # ── Setup-Panel (Zeitraum, Symbol, Laden) ─────────────────────────────────

    def _build_setup_panel(self) -> QGroupBox:
        group = QGroupBox("Session-Setup")
        group.setObjectName("setup_panel")
        layout = QHBoxLayout(group)
        layout.setSpacing(8)

        layout.addWidget(QLabel("Symbol:"))
        self._symbol_input = QLineEdit("EURUSD")
        self._symbol_input.setObjectName("symbol_input")
        self._symbol_input.setMaximumWidth(100)
        layout.addWidget(self._symbol_input)

        layout.addWidget(QLabel("Timeframe:"))
        self._timeframe_combo = QComboBox()
        self._timeframe_combo.setObjectName("timeframe_combo")
        for tf in ("M1", "M5", "M15", "H1", "H4", "D1"):
            self._timeframe_combo.addItem(tf)
        self._timeframe_combo.setCurrentText("H1")
        layout.addWidget(self._timeframe_combo)

        layout.addWidget(QLabel("Zeitraum:"))
        self._range_combo = QComboBox()
        self._range_combo.setObjectName("range_combo")
        self._range_combo.addItem("Letzte Woche",  "week")
        self._range_combo.addItem("Letzter Monat", "month")
        self._range_combo.addItem("Benutzerdefiniert", "custom")
        self._range_combo.currentIndexChanged.connect(self._on_range_changed)
        layout.addWidget(self._range_combo)

        today = date.today()
        self._start_date_edit = QDateEdit()
        self._start_date_edit.setObjectName("start_date_edit")
        self._start_date_edit.setCalendarPopup(True)
        self._start_date_edit.setDate(QDate(today.year, today.month, today.day).addDays(-7))
        layout.addWidget(self._start_date_edit)

        layout.addWidget(QLabel("–"))

        self._end_date_edit = QDateEdit()
        self._end_date_edit.setObjectName("end_date_edit")
        self._end_date_edit.setCalendarPopup(True)
        self._end_date_edit.setDate(QDate(today.year, today.month, today.day))
        layout.addWidget(self._end_date_edit)

        self._load_btn = QPushButton("Session laden")
        self._load_btn.setObjectName("load_btn")
        self._load_btn.clicked.connect(self._on_load_session)
        layout.addWidget(self._load_btn)

        layout.addStretch()

        # Datumseingabe standardmaessig deaktiviert (Preset-Modus)
        self._start_date_edit.setEnabled(False)
        self._end_date_edit.setEnabled(False)

        return group

    # ── Kerzen-Panel (aktueller Kurs) ─────────────────────────────────────────

    def _build_candle_panel(self) -> QGroupBox:
        group = QGroupBox("Aktueller Kurs")
        group.setObjectName("candle_panel")
        layout = QVBoxLayout(group)

        # Kurs-Anzeige
        self._price_label = QLabel("—")
        self._price_label.setObjectName("price_label")
        font = QFont()
        font.setPointSize(20)
        font.setBold(True)
        self._price_label.setFont(font)
        self._price_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._price_label)

        # OHLCV-Detail-Tabelle
        self._candle_table = QTableWidget(5, 2)
        self._candle_table.setObjectName("candle_table")
        self._candle_table.setHorizontalHeaderLabels(["Feld", "Wert"])
        self._candle_table.verticalHeader().setVisible(False)
        self._candle_table.horizontalHeader().setStretchLastSection(True)
        self._candle_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._candle_table.setMaximumHeight(160)
        for i, label in enumerate(("Zeit", "Open", "High", "Low", "Volume")):
            self._candle_table.setItem(i, 0, QTableWidgetItem(label))
            self._candle_table.setItem(i, 1, QTableWidgetItem("—"))
        layout.addWidget(self._candle_table)

        # Fortschritts-Anzeige
        self._progress_label = QLabel("Kerze 0 / 0")
        self._progress_label.setObjectName("progress_label")
        self._progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._progress_label)

        layout.addStretch()
        return group

    # ── Rechtes Panel: Wiedergabe + Handels-Controls ──────────────────────────

    def _build_right_panel(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(6)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._build_playback_panel())
        layout.addWidget(self._build_trading_panel())
        layout.addStretch()
        return widget

    # ── Wiedergabe-Panel ──────────────────────────────────────────────────────

    def _build_playback_panel(self) -> QGroupBox:
        group = QGroupBox("Wiedergabe")
        group.setObjectName("playback_panel")
        layout = QVBoxLayout(group)

        btn_row = QHBoxLayout()

        self._next_btn = QPushButton("▶ Naechste Kerze")
        self._next_btn.setObjectName("next_btn")
        self._next_btn.clicked.connect(self._on_next_candle)
        btn_row.addWidget(self._next_btn)

        self._next5_btn = QPushButton("▶▶ +5 Kerzen")
        self._next5_btn.setObjectName("next5_btn")
        self._next5_btn.clicked.connect(self._on_next_5)
        btn_row.addWidget(self._next5_btn)

        layout.addLayout(btn_row)

        auto_row = QHBoxLayout()
        self._auto_btn = QPushButton("Auto-Play")
        self._auto_btn.setObjectName("auto_btn")
        self._auto_btn.setCheckable(True)
        self._auto_btn.clicked.connect(self._on_toggle_auto)
        auto_row.addWidget(self._auto_btn)

        auto_row.addWidget(QLabel("Geschwindigkeit:"))
        self._speed_combo = QComboBox()
        self._speed_combo.setObjectName("speed_combo")
        for label, _ in self._SPEEDS:
            self._speed_combo.addItem(label)
        auto_row.addWidget(self._speed_combo)

        layout.addLayout(auto_row)
        return group

    # ── Handels-Panel (Kauf/Verkauf) ──────────────────────────────────────────

    def _build_trading_panel(self) -> QGroupBox:
        group = QGroupBox("Handel (Uebung)")
        group.setObjectName("trading_panel")
        layout = QVBoxLayout(group)

        form = QHBoxLayout()
        form.addWidget(QLabel("SL:"))
        self._sl_input = QDoubleSpinBox()
        self._sl_input.setObjectName("sl_input")
        self._sl_input.setDecimals(5)
        self._sl_input.setRange(0.0, 999999.0)
        self._sl_input.setSingleStep(0.0001)
        form.addWidget(self._sl_input)

        form.addWidget(QLabel("TP:"))
        self._tp_input = QDoubleSpinBox()
        self._tp_input.setObjectName("tp_input")
        self._tp_input.setDecimals(5)
        self._tp_input.setRange(0.0, 999999.0)
        self._tp_input.setSingleStep(0.0001)
        form.addWidget(self._tp_input)

        form.addWidget(QLabel("Lot:"))
        self._lot_input = QDoubleSpinBox()
        self._lot_input.setObjectName("lot_input")
        self._lot_input.setDecimals(2)
        self._lot_input.setRange(0.01, 100.0)
        self._lot_input.setSingleStep(0.01)
        self._lot_input.setValue(0.01)
        form.addWidget(self._lot_input)

        layout.addLayout(form)

        btn_row = QHBoxLayout()
        self._buy_btn = QPushButton("KAUFEN (Buy)")
        self._buy_btn.setObjectName("buy_btn")
        self._buy_btn.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold;")
        self._buy_btn.clicked.connect(self._on_buy)
        btn_row.addWidget(self._buy_btn)

        self._sell_btn = QPushButton("VERKAUFEN (Sell)")
        self._sell_btn.setObjectName("sell_btn")
        self._sell_btn.setStyleSheet("background-color: #c0392b; color: white; font-weight: bold;")
        self._sell_btn.clicked.connect(self._on_sell)
        btn_row.addWidget(self._sell_btn)

        layout.addLayout(btn_row)
        return group

    # ── Offene Positionen ─────────────────────────────────────────────────────

    def _build_positions_panel(self) -> QGroupBox:
        group = QGroupBox("Offene Positionen")
        group.setObjectName("positions_panel")
        layout = QVBoxLayout(group)

        self._positions_table = QTableWidget(0, 5)
        self._positions_table.setObjectName("positions_table")
        self._positions_table.setHorizontalHeaderLabels(
            ["ID", "Richtung", "Entry", "SL", "TP"]
        )
        self._positions_table.verticalHeader().setVisible(False)
        self._positions_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._positions_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self._positions_table.horizontalHeader().setStretchLastSection(True)
        self._positions_table.setMaximumHeight(120)
        layout.addWidget(self._positions_table)

        self._close_position_btn = QPushButton("Position schliessen")
        self._close_position_btn.setObjectName("close_position_btn")
        self._close_position_btn.clicked.connect(self._on_close_position)
        layout.addWidget(self._close_position_btn)

        return group

    # ── Statistiken ──────────────────────────────────────────────────────────

    def _build_stats_panel(self) -> QGroupBox:
        group = QGroupBox("Uebungs-Statistiken (getrennt vom TradeJournal)")
        group.setObjectName("stats_panel")
        layout = QHBoxLayout(group)

        self._stats_trade_count_label = QLabel("Trades: 0")
        self._stats_trade_count_label.setObjectName("stats_trade_count_label")
        layout.addWidget(self._stats_trade_count_label)

        self._stats_win_rate_label = QLabel("Trefferquote: 0.0%")
        self._stats_win_rate_label.setObjectName("stats_win_rate_label")
        layout.addWidget(self._stats_win_rate_label)

        self._stats_pnl_label = QLabel("Gesamt P&L: 0.00")
        self._stats_pnl_label.setObjectName("stats_pnl_label")
        layout.addWidget(self._stats_pnl_label)

        layout.addStretch()
        return group

    # ── Slot: Session laden ───────────────────────────────────────────────────

    @Slot()
    def _on_load_session(self) -> None:
        symbol    = self._symbol_input.text().strip().upper()
        timeframe = self._timeframe_combo.currentText()
        start_d   = self._start_date_edit.date()
        end_d     = self._end_date_edit.date()

        if not symbol:
            QMessageBox.warning(self, "Eingabefehler", "Symbol darf nicht leer sein.")
            return

        start_str = f"{start_d.year():04d}-{start_d.month():02d}-{start_d.day():02d}"
        end_str   = f"{end_d.year():04d}-{end_d.month():02d}-{end_d.day():02d}"

        logger.debug(
            "PracticeModeView: Lade Session | {sym} {tf} | {s} – {e}",
            sym=symbol, tf=timeframe, s=start_str, e=end_str,
        )

        try:
            result = self._backend.load_session(symbol, timeframe, start_str, end_str)
            self._session_loaded = True
            total = result.get("total_candles", 0)
            cursor = result.get("cursor", 0)
            self._progress_label.setText(f"Kerze {cursor + 1} / {total}")
            self._refresh_candle_display(result.get("current_candle"))
            self._positions_table.setRowCount(0)
            self._open_position_ids.clear()
            self._update_controls_enabled()
            self._refresh_stats()
            self.session_started.emit()
            logger.info(
                "PracticeModeView: Session geladen | {sym} {tf} | {n} Kerzen",
                sym=symbol, tf=timeframe, n=total,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("PracticeModeView: Session laden fehlgeschlagen | {e}", e=exc)
            QMessageBox.critical(self, "Fehler", f"Session konnte nicht geladen werden:\n{exc}")

    # ── Slot: Naechste Kerze ──────────────────────────────────────────────────

    @Slot()
    def _on_next_candle(self) -> None:
        self._do_advance(steps=1)

    @Slot()
    def _on_next_5(self) -> None:
        self._do_advance(steps=5)

    def _do_advance(self, steps: int) -> None:
        if not self._session_loaded:
            return
        result = self._backend.advance(steps)
        cursor = result.get("cursor", 0)
        total  = result.get("total_candles", result.get("cursor", 0))
        self._progress_label.setText(f"Kerze {cursor + 1} / {total}")
        self._refresh_candle_display(result.get("current_candle"))
        if result.get("is_at_end"):
            self._auto_timer.stop()
            self._auto_btn.setChecked(False)
        self._refresh_positions()
        self._refresh_stats()

        for ac in result.get("auto_closed", []):
            self.position_closed.emit(ac)

    # ── Slot: Auto-Play ───────────────────────────────────────────────────────

    @Slot(bool)
    def _on_toggle_auto(self, checked: bool) -> None:
        if checked and self._session_loaded:
            speed_idx = self._speed_combo.currentIndex()
            ms = self._SPEEDS[speed_idx][1] if speed_idx < len(self._SPEEDS) else 1000
            self._auto_timer.start(ms)
        else:
            self._auto_timer.stop()

    @Slot()
    def _on_auto_advance(self) -> None:
        self._do_advance(steps=1)

    # ── Slot: Kaufen/Verkaufen ────────────────────────────────────────────────

    @Slot()
    def _on_buy(self) -> None:
        self._do_open_position("buy")

    @Slot()
    def _on_sell(self) -> None:
        self._do_open_position("sell")

    def _do_open_position(self, direction: str) -> None:
        if not self._session_loaded:
            return
        lot_size = self._lot_input.value()
        sl       = self._sl_input.value()
        tp       = self._tp_input.value()

        try:
            pid = self._backend.open_position(direction, lot_size, sl, tp)
            self._open_position_ids.append(pid)
            self._refresh_positions()
            self.position_opened.emit(pid)
        except Exception as exc:  # noqa: BLE001
            logger.error("PracticeModeView: Position oeffnen fehlgeschlagen | {e}", e=exc)
            QMessageBox.critical(self, "Fehler", f"Position konnte nicht geoeffnet werden:\n{exc}")

    # ── Slot: Position schliessen ─────────────────────────────────────────────

    @Slot()
    def _on_close_position(self) -> None:
        if not self._session_loaded:
            return
        row = self._positions_table.currentRow()
        if row < 0:
            return
        pid_item = self._positions_table.item(row, 0)
        if pid_item is None:
            return
        try:
            pid = int(pid_item.text())
            result = self._backend.close_position(pid)
            if pid in self._open_position_ids:
                self._open_position_ids.remove(pid)
            self._refresh_positions()
            self._refresh_stats()
            self._show_result_dialog(result)
            self.position_closed.emit(result)
        except Exception as exc:  # noqa: BLE001
            logger.error("PracticeModeView: Position schliessen fehlgeschlagen | {e}", e=exc)
            QMessageBox.critical(self, "Fehler", f"Position konnte nicht geschlossen werden:\n{exc}")

    # ── Slot: Zeitraum-Preset geaendert ───────────────────────────────────────

    @Slot(int)
    def _on_range_changed(self, _index: int) -> None:
        preset = self._range_combo.currentData()
        today  = date.today()
        if preset == "week":
            start = today - timedelta(days=7)
            self._start_date_edit.setDate(
                QDate(start.year, start.month, start.day)
            )
            self._end_date_edit.setDate(QDate(today.year, today.month, today.day))
            self._start_date_edit.setEnabled(False)
            self._end_date_edit.setEnabled(False)
        elif preset == "month":
            start = today - timedelta(days=30)
            self._start_date_edit.setDate(
                QDate(start.year, start.month, start.day)
            )
            self._end_date_edit.setDate(QDate(today.year, today.month, today.day))
            self._start_date_edit.setEnabled(False)
            self._end_date_edit.setEnabled(False)
        else:
            self._start_date_edit.setEnabled(True)
            self._end_date_edit.setEnabled(True)

    # ── Hilfsmethoden ────────────────────────────────────────────────────────

    def _refresh_candle_display(self, candle: Optional[dict]) -> None:
        if not candle:
            self._price_label.setText("—")
            for i in range(5):
                item = self._candle_table.item(i, 1)
                if item:
                    item.setText("—")
            return
        close_val = candle.get("close")
        self._price_label.setText(_fmt_price(close_val))
        values = [
            candle.get("time", "—"),
            _fmt_price(candle.get("open")),
            _fmt_price(candle.get("high")),
            _fmt_price(candle.get("low")),
            str(candle.get("volume", "—")),
        ]
        for i, v in enumerate(values):
            item = self._candle_table.item(i, 1)
            if item:
                item.setText(str(v))

    def _refresh_positions(self) -> None:
        open_pos = self._backend.get_open_positions()
        self._positions_table.setRowCount(len(open_pos))
        for row, pos in enumerate(open_pos):
            self._positions_table.setItem(row, 0, QTableWidgetItem(str(pos.get("position_id", ""))))
            self._positions_table.setItem(row, 1, QTableWidgetItem(pos.get("direction", "")))
            self._positions_table.setItem(row, 2, QTableWidgetItem(_fmt_price(pos.get("entry_price"))))
            self._positions_table.setItem(row, 3, QTableWidgetItem(_fmt_price(pos.get("sl"))))
            self._positions_table.setItem(row, 4, QTableWidgetItem(_fmt_price(pos.get("tp"))))

    def _refresh_stats(self) -> None:
        stats = self._backend.get_stats()
        self._stats_trade_count_label.setText(f"Trades: {stats.get('trade_count', 0)}")
        wr = stats.get("win_rate", 0.0) * 100
        self._stats_win_rate_label.setText(f"Trefferquote: {wr:.1f}%")
        pnl = stats.get("total_pnl", 0.0)
        self._stats_pnl_label.setText(f"Gesamt P&L: {_fmt_pnl(pnl)}")
        self._stats_pnl_label.setStyleSheet(f"color: {_pnl_color(pnl)};")

    def _show_result_dialog(self, result: dict) -> None:
        pnl  = result.get("pnl", 0.0)
        cpnl = result.get("counterfactual_pnl", 0.0)
        by   = result.get("closed_by", "manual")
        msg  = (
            f"<b>Ergebnis:</b> {_fmt_pnl(pnl)}<br>"
            f"<b>Geschlossen durch:</b> {by}<br>"
            f"<br>"
            f"<b>Counterfactual</b> (entgegengesetzte Richtung): {_fmt_pnl(cpnl)}"
        )
        dlg = QMessageBox(self)
        dlg.setWindowTitle("Position geschlossen")
        dlg.setText(msg)
        dlg.setTextFormat(Qt.TextFormat.RichText)
        dlg.open()  # non-blocking: kein Freeze des Event-Loops

    def _update_controls_enabled(self) -> None:
        for w in (
            self._next_btn, self._next5_btn, self._auto_btn,
            self._buy_btn, self._sell_btn, self._close_position_btn,
        ):
            w.setEnabled(self._session_loaded)

    # ── Oeffentliche Properties (fuer Tests) ──────────────────────────────────

    @property
    def symbol_input(self) -> QLineEdit:
        return self._symbol_input

    @property
    def timeframe_combo(self) -> QComboBox:
        return self._timeframe_combo

    @property
    def range_combo(self) -> QComboBox:
        return self._range_combo

    @property
    def start_date_edit(self) -> QDateEdit:
        return self._start_date_edit

    @property
    def end_date_edit(self) -> QDateEdit:
        return self._end_date_edit

    @property
    def load_btn(self) -> QPushButton:
        return self._load_btn

    @property
    def next_btn(self) -> QPushButton:
        return self._next_btn

    @property
    def next5_btn(self) -> QPushButton:
        return self._next5_btn

    @property
    def auto_btn(self) -> QPushButton:
        return self._auto_btn

    @property
    def speed_combo(self) -> QComboBox:
        return self._speed_combo

    @property
    def buy_btn(self) -> QPushButton:
        return self._buy_btn

    @property
    def sell_btn(self) -> QPushButton:
        return self._sell_btn

    @property
    def sl_input(self) -> QDoubleSpinBox:
        return self._sl_input

    @property
    def tp_input(self) -> QDoubleSpinBox:
        return self._tp_input

    @property
    def lot_input(self) -> QDoubleSpinBox:
        return self._lot_input

    @property
    def positions_table(self) -> QTableWidget:
        return self._positions_table

    @property
    def close_position_btn(self) -> QPushButton:
        return self._close_position_btn

    @property
    def price_label(self) -> QLabel:
        return self._price_label

    @property
    def progress_label(self) -> QLabel:
        return self._progress_label

    @property
    def stats_trade_count_label(self) -> QLabel:
        return self._stats_trade_count_label

    @property
    def stats_win_rate_label(self) -> QLabel:
        return self._stats_win_rate_label

    @property
    def stats_pnl_label(self) -> QLabel:
        return self._stats_pnl_label

    @property
    def candle_table(self) -> QTableWidget:
        return self._candle_table
