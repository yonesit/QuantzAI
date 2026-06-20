"""
gui/widgets/console_widget.py
ConsoleWidget – Bot-Logs und Fehler in der GUI.

Zeigt loguru-Log-Eintraege ab einem konfigurierbaren Mindest-Level in Echtzeit.
Unterstuetzt Farbcodierung nach Level, Filter nach Level/Modul,
abschaltbares Auto-Scroll, Export als Textdatei und einen auffaelligen
Emergency-Banner fuer CRITICAL-Meldungen und EmergencyHandler-Eingriffe.

Komponenten:
  LogEntry       – Datenhaltung (level, module, message, timestamp)
  GuiLogSink     – Loguru-kompatibler Sink-Adapter (kein Qt-Import noetig)
  ConsoleWidget  – QWidget mit Ring-Puffer, Filter, Farbcodierung

Thread-Sicherheit:
  GuiLogSink.__call__() wird aus beliebigem Thread aufgerufen.
  _raw_signal (QueuedConnection) stellt Eintraege immer im Hauptthread zu.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Optional

from loguru import logger

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QColor, QFont, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Konstanten
# ─────────────────────────────────────────────────────────────────────────────

_LEVEL_ORDER: list[str] = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

_LEVEL_COLORS: dict[str, Optional[str]] = {
    "DEBUG":    "#6b7280",   # grau
    "INFO":     None,        # Standard-Textfarbe
    "WARNING":  "#f59e0b",   # amber
    "ERROR":    "#ef4444",   # rot
    "CRITICAL": "#dc2626",   # dunkelrot + fett
}

_EMERGENCY_KEYWORDS = ("EmergencyHandler", "CRITICAL_DRAWDOWN", "MT5_UNREACHABLE")


# ─────────────────────────────────────────────────────────────────────────────
#  LogEntry
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LogEntry:
    """Einzelner normalisierter Log-Eintrag."""

    timestamp: datetime
    level:     str      # "DEBUG" | "INFO" | "WARNING" | "ERROR" | "CRITICAL"
    module:    str      # loguru record["name"]
    message:   str


# ─────────────────────────────────────────────────────────────────────────────
#  GuiLogSink
# ─────────────────────────────────────────────────────────────────────────────

class GuiLogSink:
    """
    Loguru-kompatibler Sink-Adapter.

    Wandelt loguru-Nachrichten in LogEntry-Objekte um und leitet sie
    per Callback weiter (typischerweise ConsoleWidget.append_entry).

    Verwendung:
        sink = GuiLogSink(widget.append_entry)
        sink_id = logger.add(sink, level="DEBUG")
        ...
        logger.remove(sink_id)

    Der Sink darf nie eine Exception werfen – loguru wuerde sonst die
    gesamte Log-Infrastruktur deaktivieren.
    """

    def __init__(self, callback: Callable[[LogEntry], None]) -> None:
        self._callback = callback

    def __call__(self, message) -> None:
        try:
            record     = message.record
            level_name = record["level"].name.upper()
            if level_name == "WARN":
                level_name = "WARNING"
            entry = LogEntry(
                timestamp=datetime.now(),
                level=level_name,
                module=record.get("name") or "",
                message=record["message"],
            )
            self._callback(entry)
        except Exception:  # noqa: BLE001
            pass  # Sink darf nie crashen


# ─────────────────────────────────────────────────────────────────────────────
#  ConsoleWidget
# ─────────────────────────────────────────────────────────────────────────────

class ConsoleWidget(QWidget):
    """
    Bot-Konsole: zeigt loguru-Log-Eintraege in der GUI an.

    Parameters
    ----------
    max_lines   : Maximale Anzahl gepufferter Eintraege (Ring-Puffer, Standard 1000).
    min_level   : Standard-Mindestlevel beim Start (Standard: "INFO").
    parent      : Eltern-Widget.
    """

    # Thread-sicheres Signal fuer Hauptthread-Dispatch
    _raw_signal: Signal = Signal(object)

    def __init__(
        self,
        max_lines: int = 1000,
        min_level: str = "INFO",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._buffer: deque[LogEntry] = deque(maxlen=max_lines)
        self._default_min_level = min_level if min_level in _LEVEL_ORDER else "INFO"
        self._build()
        self._raw_signal.connect(
            self._on_entry_received, Qt.ConnectionType.QueuedConnection
        )

    # ─── Layout ──────────────────────────────────────────────────────────────

    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(4)
        self.setObjectName("console_widget")

        # ── Emergency-Banner (standardmaessig versteckt) ──────────────────────
        self._emergency_frame = QFrame()
        self._emergency_frame.setObjectName("console_emergency_banner")
        self._emergency_frame.setStyleSheet(
            "background: #dc2626; border-radius: 4px;"
        )
        self._emergency_frame.hide()

        em_row = QHBoxLayout(self._emergency_frame)
        em_row.setContentsMargins(12, 8, 8, 8)

        self._emergency_label = QLabel()
        self._emergency_label.setObjectName("console_emergency_label")
        self._emergency_label.setStyleSheet("color: white; font-weight: bold;")
        self._emergency_label.setWordWrap(True)
        em_row.addWidget(self._emergency_label, stretch=1)

        em_dismiss = QPushButton("✕")
        em_dismiss.setObjectName("console_emergency_dismiss_btn")
        em_dismiss.setMaximumWidth(28)
        em_dismiss.clicked.connect(self._emergency_frame.hide)
        em_row.addWidget(em_dismiss)

        outer.addWidget(self._emergency_frame)

        # ── Toolbar ───────────────────────────────────────────────────────────
        toolbar = QHBoxLayout()
        toolbar.setSpacing(6)

        toolbar.addWidget(QLabel("Level:"))

        self._level_combo = QComboBox()
        self._level_combo.setObjectName("console_level_combo")
        for lvl in _LEVEL_ORDER:
            self._level_combo.addItem(lvl)
        self._level_combo.setCurrentText(self._default_min_level)
        self._level_combo.currentTextChanged.connect(self._rebuild_display)
        toolbar.addWidget(self._level_combo)

        toolbar.addWidget(QLabel("Modul:"))

        self._module_filter = QLineEdit()
        self._module_filter.setObjectName("console_module_filter")
        self._module_filter.setPlaceholderText("Filter…")
        self._module_filter.setMaximumWidth(160)
        self._module_filter.textChanged.connect(self._rebuild_display)
        toolbar.addWidget(self._module_filter)

        self._autoscroll_cb = QCheckBox("Auto-Scroll")
        self._autoscroll_cb.setObjectName("console_autoscroll_checkbox")
        self._autoscroll_cb.setChecked(True)
        toolbar.addWidget(self._autoscroll_cb)

        self._export_btn = QPushButton("Export…")
        self._export_btn.setObjectName("console_export_btn")
        self._export_btn.clicked.connect(self._on_export)
        toolbar.addWidget(self._export_btn)

        self._clear_btn = QPushButton("Leeren")
        self._clear_btn.setObjectName("console_clear_btn")
        self._clear_btn.clicked.connect(self._on_clear)
        toolbar.addWidget(self._clear_btn)

        toolbar.addStretch()
        outer.addLayout(toolbar)

        # ── Log-Anzeige ───────────────────────────────────────────────────────
        self._display = QTextEdit()
        self._display.setObjectName("console_log_display")
        self._display.setReadOnly(True)
        font = QFont("Courier New", 9)
        font.setStyleHint(QFont.StyleHint.Monospace)
        self._display.setFont(font)
        outer.addWidget(self._display)

    # ─── Oeffentliche API ─────────────────────────────────────────────────────

    def append_entry(self, entry: LogEntry) -> None:
        """
        Nimmt einen Log-Eintrag entgegen – thread-sicher aus beliebigem Thread.
        """
        self._raw_signal.emit(entry)

    def export_to_file(self, path: str) -> None:
        """Schreibt die aktuell sichtbaren Zeilen in eine Datei."""
        content = self._display.toPlainText()
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)

    # ─── Properties ──────────────────────────────────────────────────────────

    @property
    def entry_count(self) -> int:
        """Anzahl gepufferter Eintraege (inkl. herausgefilterte)."""
        return len(self._buffer)

    @property
    def display_text(self) -> str:
        """Aktuell sichtbarer Text der Log-Anzeige."""
        return self._display.toPlainText()

    @property
    def auto_scroll(self) -> bool:
        return self._autoscroll_cb.isChecked()

    @property
    def min_level(self) -> str:
        return self._level_combo.currentText()

    @property
    def emergency_banner_visible(self) -> bool:
        return not self._emergency_frame.isHidden()

    # ─── Hauptthread-Slots ───────────────────────────────────────────────────

    @Slot(object)
    def _on_entry_received(self, entry: LogEntry) -> None:
        self._buffer.append(entry)
        if self._passes_filter(entry):
            self._append_to_display(entry)
        if self._is_emergency(entry):
            self._show_emergency_banner(entry.message)

    # ─── Interna ─────────────────────────────────────────────────────────────

    def _passes_filter(self, entry: LogEntry) -> bool:
        min_level = self._level_combo.currentText()
        try:
            entry_idx = _LEVEL_ORDER.index(entry.level)
            min_idx   = _LEVEL_ORDER.index(min_level)
        except ValueError:
            return True
        if entry_idx < min_idx:
            return False
        module_text = self._module_filter.text().strip().lower()
        if module_text and module_text not in entry.module.lower():
            return False
        return True

    def _is_emergency(self, entry: LogEntry) -> bool:
        if entry.level == "CRITICAL":
            return True
        for kw in _EMERGENCY_KEYWORDS:
            if kw in entry.message or kw in entry.module:
                return True
        return False

    def _append_to_display(self, entry: LogEntry) -> None:
        ts   = entry.timestamp.strftime("%H:%M:%S")
        text = f"[{ts}] [{entry.level:<8}] [{entry.module}] {entry.message}"

        cursor = self._display.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)

        fmt   = QTextCharFormat()
        color = _LEVEL_COLORS.get(entry.level)
        if color:
            fmt.setForeground(QColor(color))
        if entry.level == "CRITICAL":
            fmt.setFontWeight(QFont.Weight.Bold)

        cursor.insertText(text + "\n", fmt)
        self._display.setTextCursor(cursor)

        if self._autoscroll_cb.isChecked():
            self._display.ensureCursorVisible()

    def _rebuild_display(self) -> None:
        self._display.clear()
        for entry in self._buffer:
            if self._passes_filter(entry):
                self._append_to_display(entry)

    def _show_emergency_banner(self, message: str) -> None:
        self._emergency_label.setText(f"⚠  NOTFALL: {message}")
        self._emergency_frame.show()

    @Slot()
    def _on_export(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Log exportieren",
            "",
            "Textdateien (*.txt);;Alle Dateien (*)",
        )
        if path:
            self.export_to_file(path)

    @Slot()
    def _on_clear(self) -> None:
        self._buffer.clear()
        self._display.clear()
