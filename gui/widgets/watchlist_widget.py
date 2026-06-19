"""
gui/widgets/watchlist_widget.py
WatchlistWidget – filterbare und sortierbare Echtzeit-Symbolliste.

Zeigt pro Symbol:
  Kurs (Bid/Ask), Tagesveraenderung %, KI-Signal und Signal-Konfidenz.

Interaktion:
  - Freitext-Filter ueber alle Symbole (Gross-/Kleinschreibung egal)
  - Klick auf Spaltenheader sortiert auf- oder absteigend
  - Zeilenauswahl emittiert symbol_selected(str)

Testbarkeit:
  visible_row_count, filter_widget, table als oeffentliche Properties.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Datenklasse (pure Python)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WatchlistEntry:
    """Snapshot-Daten fuer eine Zeile der Watchlist."""
    symbol:             str
    bid:                float | None = None
    ask:                float | None = None
    daily_change_pct:   float | None = None
    signal:             str          = "flat"   # 'long' | 'short' | 'flat'
    signal_confidence:  float        = 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  Hilfsfunktion: numerisches QTableWidgetItem
# ─────────────────────────────────────────────────────────────────────────────

class _NumericItem(QTableWidgetItem):
    """QTableWidgetItem mit korrekter numerischer Sortierung."""

    def __lt__(self, other: "QTableWidgetItem") -> bool:
        try:
            return float(self.data(Qt.ItemDataRole.UserRole)) < float(
                other.data(Qt.ItemDataRole.UserRole)
            )
        except (TypeError, ValueError):
            return super().__lt__(other)


# ─────────────────────────────────────────────────────────────────────────────
#  WatchlistWidget
# ─────────────────────────────────────────────────────────────────────────────

_SIGNAL_LABELS = {
    "long":  "▲ LONG",
    "short": "▼ SHORT",
    "flat":  "— FLAT",
}
_SIGNAL_COLORS = {
    "long":  "#22c55e",
    "short": "#ef4444",
}

_COL_SYMBOL = 0
_COL_BID    = 1
_COL_ASK    = 2
_COL_CHANGE = 3
_COL_SIGNAL = 4
_COL_CONF   = 5
_HEADERS    = ["Symbol", "Bid", "Ask", "% Tag", "Signal", "Konfidenz"]


class WatchlistWidget(QWidget):
    """
    Filterbare, sortierbare Symbol-Watchlist.

    Signals
    -------
    symbol_selected(str)  – emittiert wenn eine Zeile angeklickt wird.

    Properties
    ----------
    visible_row_count  – Anzahl sichtbarer Zeilen nach Filter
    selected_symbol    – aktuell markiertes Symbol (None wenn nichts markiert)
    filter_widget      – QLineEdit fuer Zugriff in Tests
    table              – QTableWidget fuer Zugriff in Tests
    """

    symbol_selected = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("watchlist_widget")
        self._entries: list[WatchlistEntry] = []
        self._build()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Filter-Zeile
        row = QHBoxLayout()
        row.addWidget(QLabel("Filter:"))
        self._filter_input = QLineEdit()
        self._filter_input.setObjectName("watchlist_filter")
        self._filter_input.setPlaceholderText("Symbol suchen …")
        self._filter_input.textChanged.connect(self._on_filter_changed)
        row.addWidget(self._filter_input, stretch=1)
        layout.addLayout(row)

        # Tabelle
        self._table = QTableWidget(0, len(_HEADERS))
        self._table.setObjectName("watchlist_table")
        self._table.setHorizontalHeaderLabels(_HEADERS)
        self._table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSortingEnabled(True)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.verticalHeader().setVisible(False)
        self._table.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self._table, stretch=1)

    # ── Oeffentliche Methoden ─────────────────────────────────────────────────

    def update_entries(self, entries: list[WatchlistEntry]) -> None:
        """Ersetzt alle Zeilen der Watchlist und wendet den aktuellen Filter an."""
        self._entries = list(entries)
        self._refresh_table()

    def set_filter(self, text: str) -> None:
        """Setzt den Filter-Text programmatisch (aktualisiert QLineEdit und Tabelle)."""
        self._filter_input.setText(text)
        # textChanged wird automatisch gefeuert

    # ── Interne Methoden ──────────────────────────────────────────────────────

    def _on_filter_changed(self, text: str) -> None:
        flt = text.strip().upper()
        for row in range(self._table.rowCount()):
            item = self._table.item(row, _COL_SYMBOL)
            hidden = bool(flt) and flt not in (item.text().upper() if item else "")
            self._table.setRowHidden(row, hidden)

    def _refresh_table(self) -> None:
        self._table.setSortingEnabled(False)
        self._table.setRowCount(0)

        for entry in self._entries:
            row = self._table.rowCount()
            self._table.insertRow(row)

            # Symbol
            sym = QTableWidgetItem(entry.symbol)
            sym.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._table.setItem(row, _COL_SYMBOL, sym)

            # Bid
            if entry.bid is not None:
                bid_item = _NumericItem(f"{entry.bid:.5f}")
                bid_item.setData(Qt.ItemDataRole.UserRole, entry.bid)
                bid_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            else:
                bid_item = QTableWidgetItem("–")
            self._table.setItem(row, _COL_BID, bid_item)

            # Ask
            if entry.ask is not None:
                ask_item = _NumericItem(f"{entry.ask:.5f}")
                ask_item.setData(Qt.ItemDataRole.UserRole, entry.ask)
                ask_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            else:
                ask_item = QTableWidgetItem("–")
            self._table.setItem(row, _COL_ASK, ask_item)

            # Tagesveraenderung
            chg = entry.daily_change_pct
            if chg is not None:
                chg_item = _NumericItem(f"{chg:+.2f}%")
                chg_item.setData(Qt.ItemDataRole.UserRole, chg)
                chg_item.setForeground(
                    QColor("#22c55e") if chg >= 0 else QColor("#ef4444")
                )
                chg_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            else:
                chg_item = QTableWidgetItem("–")
            self._table.setItem(row, _COL_CHANGE, chg_item)

            # Signal
            sig_text  = _SIGNAL_LABELS.get(entry.signal.lower(), entry.signal.upper())
            sig_item  = QTableWidgetItem(sig_text)
            sig_color = _SIGNAL_COLORS.get(entry.signal.lower())
            if sig_color:
                sig_item.setForeground(QColor(sig_color))
            sig_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._table.setItem(row, _COL_SIGNAL, sig_item)

            # Konfidenz
            conf_item = _NumericItem(f"{entry.signal_confidence:.0%}")
            conf_item.setData(Qt.ItemDataRole.UserRole, entry.signal_confidence)
            conf_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._table.setItem(row, _COL_CONF, conf_item)

        self._table.setSortingEnabled(True)
        # Bestehenden Filter erneut anwenden
        self._on_filter_changed(self._filter_input.text())

    def _on_selection_changed(self) -> None:
        items = self._table.selectedItems()
        if not items:
            return
        item = self._table.item(items[0].row(), _COL_SYMBOL)
        if item:
            self.symbol_selected.emit(item.text())

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def selected_symbol(self) -> str | None:
        items = self._table.selectedItems()
        if not items:
            return None
        item = self._table.item(items[0].row(), _COL_SYMBOL)
        return item.text() if item else None

    @property
    def visible_row_count(self) -> int:
        return sum(
            1 for row in range(self._table.rowCount())
            if not self._table.isRowHidden(row)
        )

    @property
    def filter_widget(self) -> QLineEdit:
        return self._filter_input

    @property
    def table(self) -> QTableWidget:
        return self._table
