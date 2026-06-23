"""
gui/widgets/activity_log_widget.py
ActivityLogWidget – Live-Anzeige der Bot-Aktivitaet.

Zeigt pro Zyklus: Zeitstempel, Symbol, KI-Signal + Konfidenz,
Check-Ergebnisse (RiskGuard / PreTradeCheck / CorrelationGuard), finale Aktion.

Ring-Puffer: letzte 200 Zyklen im RAM, aeltere fallen heraus.
Live-Update via Qt-QueuedConnection – thread-sicher aus dem Worker-Thread.

Farbkodierung (HCI):
  gruen  (#22c55e) = Trade ausgefuehrt
  blau   (#3b82f6) = Vorschlag (SUGGEST_ONLY-Modus)
  grau   (#6b7280) = neutral (flat / kein Signal)
  rot    (#ef4444) = Ablehnung durch Check
  dunkelrot (#dc2626) = Notfall / Emergency
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from loguru import logger

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Daten-Typen (pure Python, keine Qt-Abhaengigkeit)
# ─────────────────────────────────────────────────────────────────────────────

class LogFilter(Enum):
    ALL        = "Alle"
    TRADES     = "Nur Trades"
    REJECTIONS = "Nur Ablehnungen"


@dataclass
class CheckResult:
    """Ergebnis eines einzelnen Guards/Checks."""
    name:   str
    passed: bool
    reason: str = ""


@dataclass
class CycleLogEntry:
    """
    Zusammenfassung eines einzelnen Orchestrator-Zyklus fuer das ActivityLog.

    Wird aus dem run_cycle()-Ergebnis-Dict des TradingOrchestrators
    gebaut und im Ring-Puffer des ActivityLogWidget gehalten.
    """

    timestamp:  datetime
    symbol:     str
    signal:     Optional[str]        # "long" | "short" | "flat" | None
    action:     str                  # "open_buy" | "open_sell" | "flat" | "skipped" | "suggested"
    reason:     str
    checks:     list[CheckResult]
    confidence: Optional[float] = None
    ticket:     Optional[int]   = None
    lot_size:   Optional[float] = None
    is_paper:   bool            = False

    @property
    def category(self) -> str:
        """
        Kategorie fuer Farbkodierung:
        TRADE | SUGGESTED | NEUTRAL | REJECTION | EMERGENCY
        """
        if self.action.startswith("open_"):
            return "TRADE"
        if "emergency" in self.reason.lower():
            return "EMERGENCY"
        if self.action == "suggested":
            return "SUGGESTED"
        if self.action == "flat" or self.reason == "signal_flat":
            return "NEUTRAL"
        return "REJECTION"

    @staticmethod
    def from_cycle_result(
        result: dict,
        ts: Optional[datetime] = None,
    ) -> "CycleLogEntry":
        """
        Erstellt einen CycleLogEntry aus dem Ergebnis-Dict von run_cycle().

        Rueckwaertskompatibel: fehlende Schlussel (checks, confidence, timestamp)
        werden mit sinnvollen Defaults belegt.
        """
        timestamp = ts or result.get("timestamp") or datetime.now(timezone.utc)
        if isinstance(timestamp, datetime) and timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        raw_checks: list[dict] = result.get("checks") or []
        checks = [
            CheckResult(
                name=str(c.get("name", "")),
                passed=bool(c.get("passed", True)),
                reason=str(c.get("reason", "")),
            )
            for c in raw_checks
        ]

        return CycleLogEntry(
            timestamp=timestamp,
            symbol=str(result.get("symbol", "")),
            signal=result.get("signal"),
            action=str(result.get("action", "skipped")),
            reason=str(result.get("reason", "")),
            checks=checks,
            confidence=result.get("confidence"),
            ticket=result.get("ticket"),
            lot_size=result.get("lot_size"),
            is_paper=bool(result.get("is_paper", False)),
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Farb-Mapping
# ─────────────────────────────────────────────────────────────────────────────

_CATEGORY_COLORS: dict[str, str] = {
    "TRADE":     "#22c55e",
    "SUGGESTED": "#3b82f6",
    "NEUTRAL":   "#6b7280",
    "REJECTION": "#ef4444",
    "EMERGENCY": "#dc2626",
}

_TABLE_COLS = ("Zeit", "Symbol", "Signal", "Checks", "Aktion")


# ─────────────────────────────────────────────────────────────────────────────
#  ActivityLogWidget
# ─────────────────────────────────────────────────────────────────────────────

class ActivityLogWidget(QWidget):
    """
    Live-Aktivitaets-Log fuer den TradingOrchestrator.

    Empfaengt Zyklus-Ergebnisse thread-sicher via Qt-QueuedConnection
    und zeigt sie in einer gefilterten, farbcodierten Tabelle an.

    Parameter
    ---------
    max_entries : Groesse des Ring-Puffers (Standard: 200).
    parent      : Qt-Elternobjekt.

    Signale
    -------
    entry_appended(CycleLogEntry) – nach jedem neuen Eintrag im Hauptthread.
    """

    _raw_signal    = Signal(object)   # internes Queued-Signal: CycleLogEntry
    entry_appended = Signal(object)   # oeffentlich: CycleLogEntry

    def __init__(
        self,
        max_entries: int = 200,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("activity_log_widget")

        self._max_entries = max_entries
        self._buffer: deque[CycleLogEntry] = deque(maxlen=max_entries)
        self._filter = LogFilter.ALL

        # Queued-Connection: Slot immer im Hauptthread ausgefuehrt,
        # auch wenn _raw_signal aus dem Worker-Thread emittiert wird.
        self._raw_signal.connect(
            self._on_entry_received,
            Qt.ConnectionType.QueuedConnection,
        )
        self._build()

    # ── Builder ───────────────────────────────────────────────────────────────

    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        # Titelzeile + Filter-Combo
        header = QHBoxLayout()

        title = QLabel("📋  Bot-Aktivitaets-Log")
        title.setObjectName("activity_log_title")
        f = title.font()
        f.setBold(True)
        title.setFont(f)
        header.addWidget(title)
        header.addStretch()

        self._filter_combo = QComboBox()
        self._filter_combo.setObjectName("activity_log_filter")
        for flt in LogFilter:
            self._filter_combo.addItem(flt.value)
        self._filter_combo.currentIndexChanged.connect(self._on_filter_changed)
        header.addWidget(self._filter_combo)

        outer.addLayout(header)

        # Log-Tabelle
        self._table = QTableWidget(0, len(_TABLE_COLS))
        self._table.setObjectName("activity_log_table")
        self._table.setHorizontalHeaderLabels(list(_TABLE_COLS))
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setAlternatingRowColors(False)
        self._table.verticalHeader().setVisible(False)
        self._table.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        hdr = self._table.horizontalHeader()
        hdr.resizeSection(0, 85)    # Zeit
        hdr.resizeSection(1, 75)    # Symbol
        hdr.resizeSection(2, 95)    # Signal + Konfidenz
        hdr.resizeSection(3, 230)   # Checks
        # Aktion: stretch (setStretchLastSection=True)

        outer.addWidget(self._table)

    # ── Oeffentliche API ──────────────────────────────────────────────────────

    def append_cycle_result(self, result: dict) -> None:
        """
        Thread-sicher: kann aus dem Worker-Thread aufgerufen werden.
        Erstellt CycleLogEntry und emittiert _raw_signal (QueuedConnection).
        """
        entry = CycleLogEntry.from_cycle_result(result)
        self._raw_signal.emit(entry)

    def clear(self) -> None:
        """Leert Puffer und Tabelle."""
        self._buffer.clear()
        self._table.setRowCount(0)

    def set_filter(self, flt: LogFilter) -> None:
        """Setzt Filter programmatisch (loest currentIndexChanged aus)."""
        idx = list(LogFilter).index(flt)
        self._filter_combo.setCurrentIndex(idx)

    @property
    def current_filter(self) -> LogFilter:
        return self._filter

    @property
    def entry_count(self) -> int:
        """Anzahl gespeicherter Eintraege (max. max_entries)."""
        return len(self._buffer)

    def entries(self) -> list[CycleLogEntry]:
        """Kopie des aktuellen Puffers (aelteste zuerst)."""
        return list(self._buffer)

    # ── Interne Slots ─────────────────────────────────────────────────────────

    @Slot(object)
    def _on_entry_received(self, entry: CycleLogEntry) -> None:
        self._buffer.append(entry)
        self._rebuild_table()
        self.entry_appended.emit(entry)
        logger.debug(
            "ActivityLog: {sym} | {act} | {cat}",
            sym=entry.symbol, act=entry.action, cat=entry.category,
        )

    @Slot(int)
    def _on_filter_changed(self, index: int) -> None:
        self._filter = list(LogFilter)[index]
        self._rebuild_table()

    # ── Tabellenaufbau ────────────────────────────────────────────────────────

    def _visible_entries(self) -> list[CycleLogEntry]:
        if self._filter == LogFilter.ALL:
            return list(self._buffer)
        if self._filter == LogFilter.TRADES:
            return [e for e in self._buffer if e.category in ("TRADE", "SUGGESTED")]
        # REJECTIONS
        return [e for e in self._buffer if e.category == "REJECTION"]

    def _rebuild_table(self) -> None:
        visible = list(reversed(self._visible_entries()))  # Neuestes oben
        self._table.setRowCount(len(visible))

        for row, entry in enumerate(visible):
            color = _CATEGORY_COLORS.get(entry.category, "#6b7280")

            self._set_cell(row, 0, entry.timestamp.strftime("%H:%M:%S"), color)
            self._set_cell(row, 1, entry.symbol, color)

            sig_str = entry.signal or "–"
            if entry.confidence is not None:
                sig_str += f" {entry.confidence:.0%}"
            self._set_cell(row, 2, sig_str, color)

            parts: list[str] = []
            for chk in entry.checks:
                icon = "✓" if chk.passed else "✗"
                p = f"{icon}{chk.name}"
                if not chk.passed and chk.reason:
                    p += f":{chk.reason}"
                parts.append(p)
            self._set_cell(row, 3, "  ".join(parts) if parts else "–", color)

            action_text = entry.action
            if entry.is_paper and entry.action.startswith("open_"):
                action_text = f"{entry.action}  [PAPER-TRADE (simuliert)]"
            self._set_cell(row, 4, action_text, color)

    def _set_cell(self, row: int, col: int, text: str, color: str) -> None:
        item = QTableWidgetItem(text)
        item.setForeground(QColor(color))
        self._table.setItem(row, col, item)
