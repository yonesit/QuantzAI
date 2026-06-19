"""
gui/views/journal_view.py
Journal- & Psychologie-View: Trade-Historie, Stimmungs-Erfassung,
Trading-DNA-Profil, Replay, KI-Coach und automatischer Report.

Layout: QTabWidget mit 5 Tabs
  0 – Historie   : filterbare/durchsuchbare Trade-Tabelle
  1 – DNA-Profil : Heatmap (Wochentag x Stunde), Symbol-/Setup-Ranking
  2 – Replay     : dasselbe ChartWidget wie CockpitView
  3 – KI-Coach   : Eingabefeld mit Trade-ID-Kontext, Antwort-Anzeige
  4 – Report     : Markdown-Report, exportierbar

Backend-Protocol (JournalBackend):
  get_trades(symbol_filter, text_search, status_filter, limit) -> list[dict]
  generate_report(period)                                       -> str
  get_dna_profile()                                             -> dict
  get_hour_weekday_matrix()  -> list[list[dict]]  # 7 Wochentage × 24 Stunden
  get_replay_data(trade_id)  -> dict
  ask_coach(question, trade_ids)                                -> str
  record_mood_open(trade_id, mood, reason)                      -> None
  record_mood_close(trade_id, mood, plan_followed, reason, pnl) -> None

Testbarkeit:
  _mood_fn injizierbar: Callable[[str, str, bool], tuple[str, str, bool] | None]
    – ersetzt _MoodPopup in Tests; None = Abbruch/Skip
"""

from __future__ import annotations

import re
from typing import Callable, Optional, Protocol, runtime_checkable

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from gui.widgets.chart_widget import CandleData, ChartWidget


# ─────────────────────────────────────────────────────────────────────────────
#  Backend-Protocol
# ─────────────────────────────────────────────────────────────────────────────

@runtime_checkable
class JournalBackend(Protocol):
    def get_trades(
        self,
        symbol_filter:    str = "",
        text_search:      str = "",
        status_filter:    str = "",
        limit:            int = 500,
    ) -> list[dict]: ...

    def generate_report(self, period: str = "daily") -> str: ...
    def get_dna_profile(self) -> dict: ...
    def get_hour_weekday_matrix(self) -> list[list[dict]]: ...
    def get_replay_data(self, trade_id: int) -> dict: ...
    def ask_coach(self, question: str, trade_ids: list[int]) -> str: ...
    def record_mood_open(self, trade_id: int, mood: str, reason: str) -> None: ...
    def record_mood_close(
        self, trade_id: int, mood: str, plan_followed: bool, reason: str, pnl: float
    ) -> None: ...


# ─────────────────────────────────────────────────────────────────────────────
#  _MoodPopup
# ─────────────────────────────────────────────────────────────────────────────

_MOOD_OPTIONS = ["calm", "focused", "nervous", "fomo", "angry", "overconfident"]


class _MoodPopup(QDialog):
    """
    Unaufdringliches Stimmungs-Erfassungs-Popup (nicht-modal, kein Pflichtfeld).

    Signals
    -------
    mood_saved(str, str, bool)  – (mood, reason, plan_followed)
    mood_skipped()
    """

    mood_saved   = Signal(str, str, bool)
    mood_skipped = Signal()

    def __init__(
        self,
        title:         str,
        show_plan_cb:  bool = False,
        parent:        Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setObjectName("mood_popup")
        self.setModal(False)
        self.setMinimumWidth(340)

        lay = QVBoxLayout(self)
        lay.setSpacing(10)
        lay.setContentsMargins(20, 20, 20, 20)

        lay.addWidget(QLabel("Aktuelle Stimmung (optional):"))
        self._mood_combo = QComboBox()
        self._mood_combo.setObjectName("mood_combo")
        self._mood_combo.addItems([m.capitalize() for m in _MOOD_OPTIONS])
        lay.addWidget(self._mood_combo)

        lay.addWidget(QLabel("Begründung (optional):"))
        self._reason_edit = QLineEdit()
        self._reason_edit.setObjectName("mood_reason")
        self._reason_edit.setPlaceholderText("z.B. Setup entsprach System-Kriterien …")
        lay.addWidget(self._reason_edit)

        self._plan_cb: Optional[QCheckBox] = None
        if show_plan_cb:
            self._plan_cb = QCheckBox("Plan befolgt?")
            self._plan_cb.setObjectName("plan_followed_cb")
            lay.addWidget(self._plan_cb)

        btns = QDialogButtonBox()
        skip_btn = QPushButton("Überspringen")
        skip_btn.setObjectName("skip_btn")
        save_btn = QPushButton("Speichern")
        save_btn.setObjectName("save_btn")
        btns.addButton(skip_btn, QDialogButtonBox.ButtonRole.RejectRole)
        btns.addButton(save_btn, QDialogButtonBox.ButtonRole.AcceptRole)
        btns.rejected.connect(self._on_skip)
        btns.accepted.connect(self._on_save)
        lay.addWidget(btns)

    def _on_save(self) -> None:
        mood  = _MOOD_OPTIONS[self._mood_combo.currentIndex()]
        reason = self._reason_edit.text().strip()
        plan   = bool(self._plan_cb.isChecked()) if self._plan_cb else False
        self.mood_saved.emit(mood, reason, plan)
        self.accept()

    def _on_skip(self) -> None:
        self.mood_skipped.emit()
        self.reject()

    @property
    def mood_combo(self) -> QComboBox:
        return self._mood_combo

    @property
    def reason_edit(self) -> QLineEdit:
        return self._reason_edit

    @property
    def plan_checkbox(self) -> Optional[QCheckBox]:
        return self._plan_cb


# ─────────────────────────────────────────────────────────────────────────────
#  _DnaHeatmapCanvas  (7 Wochentage × 24 Stunden)
# ─────────────────────────────────────────────────────────────────────────────

_WEEKDAY_LABELS = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]


class _DnaHeatmapCanvas(QWidget):
    """
    7×24-Heatmap der Wochentag-x-Stunde-Performance (QPainter).

    Datenformat: list[list[dict|None]]  – shape (7, 24)
      Outer index: Wochentag (0=Mo … 6=So)
      Inner index: Stunde    (0–23)
      Dict-Keys:  win_rate (float), n_trades (int)
    """

    _BG   = QColor("#0f0f11")
    _NONE = QColor("#1e1e2e")
    _TEXT = QColor("#94a3b8")

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("dna_heatmap")
        self._matrix: list[list[Optional[dict]]] = [[None] * 24 for _ in range(7)]
        self.setMinimumSize(500, 160)

    def set_data(self, matrix: list[list[Optional[dict]]]) -> None:
        """matrix[weekday][hour] = {"win_rate": float, "n_trades": int} or None"""
        self._matrix = matrix
        self.update()

    @property
    def matrix(self) -> list[list[Optional[dict]]]:
        return self._matrix

    def get_cell(self, weekday: int, hour: int) -> Optional[dict]:
        if 0 <= weekday < 7 and 0 <= hour < 24:
            return self._matrix[weekday][hour]
        return None

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, self._BG)

        PL, PT, PR, PB = 28, 16, 8, 24
        cw = (w - PL - PR) / 24
        ch = (h - PT - PB) / 7

        # Hour labels (X axis)
        p.setPen(QPen(self._TEXT))
        for hr in range(0, 24, 4):
            x = PL + int(hr * cw + cw / 2)
            p.drawText(x - 10, h - PB + 4, 20, PB - 4,
                       Qt.AlignmentFlag.AlignCenter, str(hr))

        for day_i in range(7):
            # Weekday label (Y axis)
            y_center = PT + int(day_i * ch + ch / 2)
            p.setPen(QPen(self._TEXT))
            p.drawText(0, y_center - 8, PL - 2, 16,
                       Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                       _WEEKDAY_LABELS[day_i])

            for hr in range(24):
                cx = PL + int(hr * cw)
                cy = PT + int(day_i * ch)
                cw_px = max(1, int(cw) - 1)
                ch_px = max(1, int(ch) - 1)

                cell = self._matrix[day_i][hr] if day_i < len(self._matrix) and hr < len(self._matrix[day_i]) else None
                if cell is None or cell.get("n_trades", 0) == 0:
                    p.fillRect(cx, cy, cw_px, ch_px, self._NONE)
                else:
                    wr = max(0.0, min(1.0, float(cell.get("win_rate", 0.5))))
                    r = min(255, int(240 - 180 * wr))
                    g = min(255, int(80  + 160 * wr))
                    p.fillRect(cx, cy, cw_px, ch_px, QColor(r, g, 60))


# ─────────────────────────────────────────────────────────────────────────────
#  _NumericTableItem
# ─────────────────────────────────────────────────────────────────────────────

class _NumericTableItem(QTableWidgetItem):
    def __lt__(self, other: "QTableWidgetItem") -> bool:
        try:
            return float(self.data(Qt.ItemDataRole.UserRole)) < float(
                other.data(Qt.ItemDataRole.UserRole)
            )
        except (TypeError, ValueError):
            return super().__lt__(other)


# ─────────────────────────────────────────────────────────────────────────────
#  Tab 0 – Trade-Historie
# ─────────────────────────────────────────────────────────────────────────────

_HISTORY_COLS = ["ID", "Symbol", "Richtung", "Lots", "Einstieg", "Ausstieg",
                 "P&L", "Status", "Setup"]
_HC = {c: i for i, c in enumerate(_HISTORY_COLS)}


class _HistoryTab(QWidget):
    """Filterbare/durchsuchbare Trade-Historie."""

    replay_requested = Signal(int)  # trade_id
    coach_trade_added = Signal(int)  # trade_id -> KI-Coach-Kontext

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("history_tab")
        self._build()

    def _build(self) -> None:
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        # Filter-Leiste
        filt_row = QHBoxLayout()
        self._sym_filter  = QLineEdit()
        self._sym_filter.setObjectName("sym_filter")
        self._sym_filter.setPlaceholderText("Symbol …")
        self._sym_filter.setMaximumWidth(100)

        self._text_search = QLineEdit()
        self._text_search.setObjectName("text_search")
        self._text_search.setPlaceholderText("Suche (ID, Setup, Regime …)")

        self._status_combo = QComboBox()
        self._status_combo.setObjectName("status_combo")
        self._status_combo.addItems(["Alle", "open", "closed"])

        self._apply_btn = QPushButton("Anwenden")
        self._apply_btn.setObjectName("apply_filter_btn")

        filt_row.addWidget(QLabel("Symbol:"))
        filt_row.addWidget(self._sym_filter)
        filt_row.addWidget(QLabel("Suche:"))
        filt_row.addWidget(self._text_search, stretch=1)
        filt_row.addWidget(QLabel("Status:"))
        filt_row.addWidget(self._status_combo)
        filt_row.addWidget(self._apply_btn)
        lay.addLayout(filt_row)

        # Tabelle
        self._table = QTableWidget(0, len(_HISTORY_COLS))
        self._table.setObjectName("history_table")
        self._table.setHorizontalHeaderLabels(_HISTORY_COLS)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)
        self._table.horizontalHeader().setStretchLastSection(True)
        lay.addWidget(self._table, stretch=1)

        # Aktions-Leiste
        act_row = QHBoxLayout()
        self._replay_btn = QPushButton("▶  Replay")
        self._replay_btn.setObjectName("replay_btn")
        self._replay_btn.setEnabled(False)
        self._coach_btn  = QPushButton("🤖  Coach verknüpfen")
        self._coach_btn.setObjectName("coach_btn")
        self._coach_btn.setEnabled(False)
        act_row.addWidget(self._replay_btn)
        act_row.addWidget(self._coach_btn)
        act_row.addStretch()
        lay.addLayout(act_row)

        self._table.itemSelectionChanged.connect(self._on_selection_changed)
        self._replay_btn.clicked.connect(self._on_replay)
        self._coach_btn.clicked.connect(self._on_coach)

    # ── Public ──────────────────────────────────────────────────────────────

    def load_trades(self, trades: list[dict]) -> None:
        self._table.setSortingEnabled(False)
        self._table.setRowCount(0)
        for t in trades:
            row = self._table.rowCount()
            self._table.insertRow(row)

            tid   = t.get("id",        "")
            sym   = t.get("symbol",    "")
            direc = t.get("direction", "")
            lots  = t.get("lot_size")
            entry = t.get("entry_time", "")
            ex    = t.get("exit_time",  "") or ""
            pnl   = t.get("pnl")
            stat  = t.get("status",    "")
            setup = t.get("setup",     "") or ""

            def _item(val, numeric=False):
                it = QTableWidgetItem(str(val) if val is not None else "–")
                if numeric and val is not None:
                    ni = _NumericTableItem(str(val))
                    ni.setData(Qt.ItemDataRole.UserRole, float(val))
                    return ni
                return it

            self._table.setItem(row, _HC["ID"],       _item(tid))
            self._table.setItem(row, _HC["Symbol"],   _item(sym))
            self._table.setItem(row, _HC["Richtung"], _item(direc))
            self._table.setItem(row, _HC["Lots"],     _item(lots, numeric=True))
            self._table.setItem(row, _HC["Einstieg"], _item(entry))
            self._table.setItem(row, _HC["Ausstieg"], _item(ex))

            pnl_item = _item(f"{pnl:+.2f}" if pnl is not None else "–")
            if pnl is not None:
                pnl_item.setForeground(QColor("#22c55e") if pnl >= 0 else QColor("#ef4444"))
            self._table.setItem(row, _HC["P&L"],    pnl_item)
            self._table.setItem(row, _HC["Status"], _item(stat))
            self._table.setItem(row, _HC["Setup"],  _item(setup))

        self._table.setSortingEnabled(True)

    def apply_local_filter(self, text: str) -> None:
        """Blendet Zeilen aus, die den Suchtext nicht enthalten (alle Spalten)."""
        txt = text.strip().lower()
        for row in range(self._table.rowCount()):
            match = not txt or any(
                txt in (self._table.item(row, col).text().lower() if self._table.item(row, col) else "")
                for col in range(self._table.columnCount())
            )
            self._table.setRowHidden(row, not match)

    def _on_selection_changed(self) -> None:
        has = bool(self._table.selectedItems())
        self._replay_btn.setEnabled(has)
        self._coach_btn.setEnabled(has)

    def _selected_trade_id(self) -> Optional[int]:
        items = self._table.selectedItems()
        if not items:
            return None
        row = items[0].row()
        id_item = self._table.item(row, _HC["ID"])
        if id_item is None:
            return None
        try:
            return int(id_item.text())
        except ValueError:
            return None

    def _on_replay(self) -> None:
        tid = self._selected_trade_id()
        if tid is not None:
            self.replay_requested.emit(tid)

    def _on_coach(self) -> None:
        tid = self._selected_trade_id()
        if tid is not None:
            self.coach_trade_added.emit(tid)

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def sym_filter(self) -> QLineEdit:
        return self._sym_filter

    @property
    def text_search(self) -> QLineEdit:
        return self._text_search

    @property
    def status_combo(self) -> QComboBox:
        return self._status_combo

    @property
    def apply_btn(self) -> QPushButton:
        return self._apply_btn

    @property
    def table(self) -> QTableWidget:
        return self._table

    @property
    def replay_btn(self) -> QPushButton:
        return self._replay_btn

    @property
    def coach_btn(self) -> QPushButton:
        return self._coach_btn

    def visible_row_count(self) -> int:
        return sum(
            1 for r in range(self._table.rowCount())
            if not self._table.isRowHidden(r)
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Tab 1 – DNA-Profil
# ─────────────────────────────────────────────────────────────────────────────

class _DnaTab(QWidget):
    """Trading-DNA-Profil: Heatmap + Rankings + Schwaechen."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("dna_tab")
        self._build()

    def _build(self) -> None:
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)

        # Status-Label
        self._status_lbl = QLabel("Profil noch nicht geladen.")
        self._status_lbl.setObjectName("dna_status_lbl")
        lay.addWidget(self._status_lbl)

        # Heatmap
        hm_box = QGroupBox("Performance-Heatmap (Wochentag × Stunde)")
        hm_lay = QVBoxLayout(hm_box)
        self._heatmap = _DnaHeatmapCanvas()
        hm_lay.addWidget(self._heatmap)
        lay.addWidget(hm_box)

        # Rankings side by side
        rank_split = QSplitter(Qt.Orientation.Horizontal)

        sym_box = QGroupBox("Symbole (nach Gesamt-PnL)")
        sym_lay = QVBoxLayout(sym_box)
        self._sym_table = self._make_rank_table(["Symbol", "Win-Rate", "PnL", "Trades", "Konfidenz"])
        sym_lay.addWidget(self._sym_table)
        rank_split.addWidget(sym_box)

        setup_box = QGroupBox("Setups (nach Gesamt-PnL)")
        setup_lay = QVBoxLayout(setup_box)
        self._setup_table = self._make_rank_table(["Setup", "Win-Rate", "PnL", "Trades", "Konfidenz"])
        setup_lay.addWidget(self._setup_table)
        rank_split.addWidget(setup_box)

        lay.addWidget(rank_split, stretch=2)

        # Weaknesses
        weak_box = QGroupBox("Psychologische Schwächen")
        weak_lay = QVBoxLayout(weak_box)
        self._weakness_lbl = QLabel("–")
        self._weakness_lbl.setObjectName("weakness_lbl")
        self._weakness_lbl.setWordWrap(True)
        weak_lay.addWidget(self._weakness_lbl)
        lay.addWidget(weak_box)

    def _make_rank_table(self, headers: list[str]) -> QTableWidget:
        t = QTableWidget(0, len(headers))
        t.setHorizontalHeaderLabels(headers)
        t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        t.verticalHeader().setVisible(False)
        t.horizontalHeader().setStretchLastSection(True)
        return t

    def load_profile(self, profile: dict) -> None:
        status = profile.get("status", "insufficient_data")
        n      = profile.get("n_trades", 0)
        min_n  = profile.get("min_trades_required", 500)

        if status != "ready":
            self._status_lbl.setText(
                f"Unzureichend Daten: {n} / {min_n} Trades erforderlich."
            )
            return

        self._status_lbl.setText(
            f"Profil auf Basis von {n} Trades (Mindestanzahl: {min_n})."
        )
        self._load_rank_table(self._sym_table,   profile.get("symbols",  {}).get("ranked", []), "symbol")
        self._load_rank_table(self._setup_table, profile.get("setups",   {}).get("ranked", []), "setup")

        weaknesses = profile.get("psychological_weaknesses", [])
        self._weakness_lbl.setText(
            "\n".join(f"• {w}" for w in weaknesses) if weaknesses else "Keine erkannt."
        )

    def _load_rank_table(
        self, table: QTableWidget, ranked: list[dict], key: str
    ) -> None:
        table.setRowCount(0)
        for item in ranked:
            row = table.rowCount()
            table.insertRow(row)
            table.setItem(row, 0, QTableWidgetItem(str(item.get(key, ""))))
            wr = item.get("win_rate", 0.0)
            table.setItem(row, 1, QTableWidgetItem(f"{wr:.1%}"))
            table.setItem(row, 2, QTableWidgetItem(f"{item.get('total_pnl', 0.0):.2f}"))
            table.setItem(row, 3, QTableWidgetItem(str(item.get("n_trades", 0))))
            table.setItem(row, 4, QTableWidgetItem(str(item.get("confidence", ""))))

    def load_heatmap(self, matrix: list[list[Optional[dict]]]) -> None:
        self._heatmap.set_data(matrix)

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def heatmap(self) -> _DnaHeatmapCanvas:
        return self._heatmap

    @property
    def status_label(self) -> QLabel:
        return self._status_lbl

    @property
    def sym_table(self) -> QTableWidget:
        return self._sym_table

    @property
    def setup_table(self) -> QTableWidget:
        return self._setup_table

    @property
    def weakness_label(self) -> QLabel:
        return self._weakness_lbl


# ─────────────────────────────────────────────────────────────────────────────
#  Tab 2 – Replay
# ─────────────────────────────────────────────────────────────────────────────

class _ReplayTab(QWidget):
    """Replay-Ansicht: Chart + Trade-Metadaten."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("replay_tab")
        self._current_trade_id: Optional[int] = None
        self._build()

    def _build(self) -> None:
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        # Info-Leiste
        info_row = QHBoxLayout()
        self._trade_info_lbl = QLabel("Kein Trade ausgewählt.")
        self._trade_info_lbl.setObjectName("replay_trade_info")
        info_row.addWidget(self._trade_info_lbl, stretch=1)
        lay.addLayout(info_row)

        # Chart (dasselbe Widget wie CockpitView)
        self._chart = ChartWidget()
        self._chart.setObjectName("replay_chart")
        lay.addWidget(self._chart, stretch=1)

        # Marker-Info
        self._marker_lbl = QLabel("")
        self._marker_lbl.setObjectName("replay_marker_lbl")
        self._marker_lbl.setWordWrap(True)
        lay.addWidget(self._marker_lbl)

    def load_replay(self, data: dict) -> None:
        """Laedt Replay-Daten und aktualisiert Chart."""
        trade = data.get("trade", {})
        meta  = data.get("meta",  {})

        self._current_trade_id = trade.get("id")
        sym = trade.get("symbol", "?")
        direc = trade.get("direction", "?")
        entry_p = trade.get("entry_price")
        exit_p  = trade.get("exit_price")
        pnl     = trade.get("pnl")

        info = f"Trade #{self._current_trade_id} | {sym} {direc.upper()}"
        if entry_p is not None:
            info += f" | Entry: {entry_p:.5f}"
        if exit_p is not None:
            info += f" | Exit: {exit_p:.5f}"
        if pnl is not None:
            sign = "+" if pnl >= 0 else ""
            info += f" | PnL: {sign}{pnl:.2f}"
        self._trade_info_lbl.setText(info)

        candles_raw = data.get("candles", [])
        candles = [
            CandleData(
                timestamp=c.get("time", ""),
                open=float(c.get("open") or 0),
                high=float(c.get("high") or 0),
                low=float(c.get("low")  or 0),
                close=float(c.get("close") or 0),
                volume=float(c.get("volume") or 0),
            )
            for c in candles_raw
            if c.get("open") is not None
        ]
        self._chart.set_symbol(sym)
        self._chart.set_candles(candles)
        if entry_p:
            self._chart.set_position_levels(
                sl=entry_p,
                tp=exit_p or 0.0,
                trailing=0.0,
            )

        entry_marker = data.get("entry_marker", {})
        exit_marker  = data.get("exit_marker")
        marker_parts = [f"Entry: {entry_marker.get('time', '?')}"]
        if exit_marker:
            marker_parts.append(f"Exit: {exit_marker.get('time', '?')}")
        n_lookahead = meta.get("no_lookahead", False)
        if n_lookahead:
            marker_parts.append("✓ No-Lookahead garantiert")
        self._marker_lbl.setText(" | ".join(marker_parts))

    @property
    def chart_widget(self) -> ChartWidget:
        return self._chart

    @property
    def trade_info_label(self) -> QLabel:
        return self._trade_info_lbl

    @property
    def marker_label(self) -> QLabel:
        return self._marker_lbl

    @property
    def current_trade_id(self) -> Optional[int]:
        return self._current_trade_id


# ─────────────────────────────────────────────────────────────────────────────
#  Tab 3 – KI-Coach
# ─────────────────────────────────────────────────────────────────────────────

class _CoachTab(QWidget):
    """KI-Coach: Freitexteingabe mit Trade-ID-Kontext, Antwort-Anzeige."""

    ask_requested = Signal(str, list)  # question, trade_ids

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("coach_tab")
        self._trade_ids: list[int] = []
        self._build()

    def _build(self) -> None:
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        # Trade-Kontext
        ctx_row = QHBoxLayout()
        ctx_row.addWidget(QLabel("Verknüpfte Trades (IDs):"))
        self._ctx_lbl = QLabel("–")
        self._ctx_lbl.setObjectName("coach_ctx_lbl")
        ctx_row.addWidget(self._ctx_lbl, stretch=1)
        self._clear_ctx_btn = QPushButton("✕ Leeren")
        self._clear_ctx_btn.setObjectName("clear_ctx_btn")
        self._clear_ctx_btn.clicked.connect(self._on_clear_ctx)
        ctx_row.addWidget(self._clear_ctx_btn)
        lay.addLayout(ctx_row)

        # Chat-Verlauf
        self._chat_display = QTextBrowser()
        self._chat_display.setObjectName("coach_chat")
        self._chat_display.setOpenExternalLinks(False)
        lay.addWidget(self._chat_display, stretch=1)

        # Eingabe
        inp_row = QHBoxLayout()
        self._input_edit = QLineEdit()
        self._input_edit.setObjectName("coach_input")
        self._input_edit.setPlaceholderText("Frage an den KI-Coach …")
        self._input_edit.returnPressed.connect(self._on_send)
        self._send_btn = QPushButton("Senden")
        self._send_btn.setObjectName("coach_send_btn")
        self._send_btn.clicked.connect(self._on_send)
        inp_row.addWidget(self._input_edit, stretch=1)
        inp_row.addWidget(self._send_btn)
        lay.addLayout(inp_row)

    def add_trade_id(self, trade_id: int) -> None:
        if trade_id not in self._trade_ids:
            self._trade_ids.append(trade_id)
        self._refresh_ctx_lbl()

    def _on_clear_ctx(self) -> None:
        self._trade_ids.clear()
        self._refresh_ctx_lbl()

    def _refresh_ctx_lbl(self) -> None:
        self._ctx_lbl.setText(
            ", ".join(f"#{t}" for t in self._trade_ids) if self._trade_ids else "–"
        )

    def _on_send(self) -> None:
        q = self._input_edit.text().strip()
        if not q:
            return
        self._chat_display.append(f"<b>Du:</b> {q}")
        self._input_edit.clear()
        self.ask_requested.emit(q, list(self._trade_ids))

    def show_response(self, response: str) -> None:
        self._chat_display.append(f"<b>KI-Coach:</b> {response}")

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def input_edit(self) -> QLineEdit:
        return self._input_edit

    @property
    def send_btn(self) -> QPushButton:
        return self._send_btn

    @property
    def chat_display(self) -> QTextBrowser:
        return self._chat_display

    @property
    def ctx_label(self) -> QLabel:
        return self._ctx_lbl

    @property
    def clear_ctx_btn(self) -> QPushButton:
        return self._clear_ctx_btn

    @property
    def trade_ids(self) -> list[int]:
        return list(self._trade_ids)


# ─────────────────────────────────────────────────────────────────────────────
#  Tab 4 – Report
# ─────────────────────────────────────────────────────────────────────────────

class _ReportTab(QWidget):
    """Automatischer Report: Markdown-Ansicht + Export."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("report_tab")
        self._build()

    def _build(self) -> None:
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        ctrl_row = QHBoxLayout()
        self._period_combo = QComboBox()
        self._period_combo.setObjectName("period_combo")
        self._period_combo.addItems(["daily", "weekly"])
        self._load_btn   = QPushButton("Report laden")
        self._load_btn.setObjectName("load_report_btn")
        self._export_btn = QPushButton("Exportieren …")
        self._export_btn.setObjectName("export_report_btn")
        ctrl_row.addWidget(QLabel("Zeitraum:"))
        ctrl_row.addWidget(self._period_combo)
        ctrl_row.addWidget(self._load_btn)
        ctrl_row.addWidget(self._export_btn)
        ctrl_row.addStretch()
        lay.addLayout(ctrl_row)

        self._report_edit = QTextEdit()
        self._report_edit.setObjectName("report_edit")
        self._report_edit.setReadOnly(True)
        lay.addWidget(self._report_edit, stretch=1)

    def show_report(self, text: str) -> None:
        self._report_edit.setPlainText(text)

    def export_report(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self._report_edit.toPlainText())

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def period_combo(self) -> QComboBox:
        return self._period_combo

    @property
    def load_btn(self) -> QPushButton:
        return self._load_btn

    @property
    def export_btn(self) -> QPushButton:
        return self._export_btn

    @property
    def report_edit(self) -> QTextEdit:
        return self._report_edit


# ─────────────────────────────────────────────────────────────────────────────
#  JournalView
# ─────────────────────────────────────────────────────────────────────────────

_TAB_HISTORY = 0
_TAB_DNA     = 1
_TAB_REPLAY  = 2
_TAB_COACH   = 3
_TAB_REPORT  = 4


class JournalView(QWidget):
    """
    Journal- & Psychologie-View.

    Signals
    -------
    mood_recorded(dict)  – nach erfolgreicher Stimmungs-Erfassung

    Parameters
    ----------
    backend   : JournalBackend-Implementierung (kann None sein)
    _mood_fn  : Callable[[str, str, bool], tuple[str, str, bool] | None]
                Injizierbar fuer Tests; None-Rueckgabe = Skip.
                Parameter: (title, type_hint {"open"|"close"}, show_plan)
                Rueckgabe: (mood_str, reason_str, plan_followed) oder None
    """

    mood_recorded = Signal(dict)

    def __init__(
        self,
        backend:  Optional[JournalBackend] = None,
        _mood_fn: Optional[Callable] = None,
        parent:   Optional[QWidget]  = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("journal_view")
        self._backend  = backend
        self._mood_fn  = _mood_fn
        self._build()

    # ── Build ─────────────────────────────────────────────────────────────────

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        self._tabs = QTabWidget()
        self._tabs.setObjectName("journal_tabs")

        self._history_tab = _HistoryTab()
        self._dna_tab     = _DnaTab()
        self._replay_tab  = _ReplayTab()
        self._coach_tab   = _CoachTab()
        self._report_tab  = _ReportTab()

        self._tabs.addTab(self._history_tab, "📋  Historie")
        self._tabs.addTab(self._dna_tab,     "🧬  DNA-Profil")
        self._tabs.addTab(self._replay_tab,  "▶  Replay")
        self._tabs.addTab(self._coach_tab,   "🤖  KI-Coach")
        self._tabs.addTab(self._report_tab,  "📄  Report")

        root.addWidget(self._tabs)

        # Wire up internal signals
        self._history_tab.replay_requested.connect(self._on_replay_requested)
        self._history_tab.coach_trade_added.connect(self._coach_tab.add_trade_id)
        self._history_tab.apply_btn.clicked.connect(self._on_apply_filter)
        self._history_tab.text_search.textChanged.connect(
            self._history_tab.apply_local_filter
        )
        self._coach_tab.ask_requested.connect(self._on_ask_coach)
        self._report_tab.load_btn.clicked.connect(self._on_load_report)
        self._report_tab.export_btn.clicked.connect(self._on_export_report)

    # ── Public API ────────────────────────────────────────────────────────────

    def set_backend(self, backend: JournalBackend) -> None:
        self._backend = backend

    def refresh_history(self) -> None:
        if self._backend is None:
            return
        sym    = self._history_tab.sym_filter.text().strip()
        search = self._history_tab.text_search.text().strip()
        stat   = self._history_tab.status_combo.currentText()
        status = stat if stat != "Alle" else ""
        trades = self._backend.get_trades(
            symbol_filter=sym,
            text_search=search,
            status_filter=status,
        )
        self._history_tab.load_trades(trades)

    def refresh_dna(self) -> None:
        if self._backend is None:
            return
        profile = self._backend.get_dna_profile()
        self._dna_tab.load_profile(profile)
        matrix = self._backend.get_hour_weekday_matrix()
        self._dna_tab.load_heatmap(matrix)

    def show_mood_popup_open(
        self, trade_id: int, symbol: str = ""
    ) -> None:
        """Zeigt Stimmungs-Popup fuer Trade-Eroeffnung (nicht-blockierend)."""
        title = f"Stimmung – Trade Eröffnung{' ' + symbol if symbol else ''}"
        if self._mood_fn is not None:
            result = self._mood_fn(title, "open", False)
            if result is not None:
                mood, reason, _ = result
                if self._backend:
                    self._backend.record_mood_open(trade_id, mood, reason)
                self.mood_recorded.emit(
                    {"trade_id": trade_id, "mood": mood, "type": "open"}
                )
        else:
            popup = _MoodPopup(title, show_plan_cb=False, parent=self)
            popup.mood_saved.connect(
                lambda m, r, _: self._save_mood_open(trade_id, m, r)
            )
            popup.show()

    def show_mood_popup_close(
        self, trade_id: int, pnl: float = 0.0
    ) -> None:
        """Zeigt Stimmungs-Popup fuer Trade-Schliessung (nicht-blockierend)."""
        title = "Stimmung – Trade Schliessung"
        if self._mood_fn is not None:
            result = self._mood_fn(title, "close", True)
            if result is not None:
                mood, reason, plan_followed = result
                if self._backend:
                    self._backend.record_mood_close(
                        trade_id, mood, plan_followed, reason, pnl
                    )
                self.mood_recorded.emit(
                    {"trade_id": trade_id, "mood": mood, "type": "close"}
                )
        else:
            popup = _MoodPopup(title, show_plan_cb=True, parent=self)
            popup.mood_saved.connect(
                lambda m, r, plan: self._save_mood_close(trade_id, m, plan, r, pnl)
            )
            popup.show()

    def _save_mood_open(self, trade_id: int, mood: str, reason: str) -> None:
        if self._backend:
            self._backend.record_mood_open(trade_id, mood, reason)
        self.mood_recorded.emit({"trade_id": trade_id, "mood": mood, "type": "open"})

    def _save_mood_close(
        self, trade_id: int, mood: str, plan: bool, reason: str, pnl: float
    ) -> None:
        if self._backend:
            self._backend.record_mood_close(trade_id, mood, plan, reason, pnl)
        self.mood_recorded.emit({"trade_id": trade_id, "mood": mood, "type": "close"})

    # ── Internal handlers ─────────────────────────────────────────────────────

    def _on_apply_filter(self) -> None:
        self.refresh_history()

    def _on_replay_requested(self, trade_id: int) -> None:
        if self._backend is None:
            return
        try:
            data = self._backend.get_replay_data(trade_id)
            self._replay_tab.load_replay(data)
            self._tabs.setCurrentIndex(_TAB_REPLAY)
        except Exception:  # noqa: BLE001
            pass

    def _on_ask_coach(self, question: str, trade_ids: list[int]) -> None:
        if self._backend is None:
            self._coach_tab.show_response("Kein Backend verbunden.")
            return
        try:
            answer = self._backend.ask_coach(question, trade_ids)
            self._coach_tab.show_response(answer)
        except Exception as exc:  # noqa: BLE001
            self._coach_tab.show_response(f"Fehler: {exc}")

    def _on_load_report(self) -> None:
        if self._backend is None:
            return
        period = self._report_tab.period_combo.currentText()
        text   = self._backend.generate_report(period)
        self._report_tab.show_report(text)

    def _on_export_report(self, _export_fn=None) -> None:
        text = self._report_tab.report_edit.toPlainText()
        if not text:
            return
        if _export_fn is not None:
            _export_fn(text)
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Report exportieren", "journal_report.md",
            "Markdown (*.md);;Text (*.txt)"
        )
        if path:
            self._report_tab.export_report(path)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def tabs(self) -> QTabWidget:
        return self._tabs

    @property
    def history_tab(self) -> _HistoryTab:
        return self._history_tab

    @property
    def dna_tab(self) -> _DnaTab:
        return self._dna_tab

    @property
    def replay_tab(self) -> _ReplayTab:
        return self._replay_tab

    @property
    def coach_tab(self) -> _CoachTab:
        return self._coach_tab

    @property
    def report_tab(self) -> _ReportTab:
        return self._report_tab
