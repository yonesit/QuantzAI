"""
gui/views/attribution_view.py
Performance-Attribution-View – Top-Features und Drift-Warnung.

Zeigt:
  - Top-5-Features mit ihrem aktuellen SHAP-Beitrag (Balkendiagramm-aehnlich)
  - Drift-Warnung wenn ein Feature seinen Edge verliert
  - Letzter Update-Zeitstempel

Architektur (Trennung UI / Logik):
  - AttributionSnapshot    : reines Daten-Objekt (kein Qt)
  - format_importance_row  : pure Funktion, testbar ohne Qt
  - AttributionBackend     : Protocol
  - AttributionView        : nur Darstellung
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Protocol, runtime_checkable

from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Daten-Typen (pure Python)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AttributionSnapshot:
    """Momentaufnahme der Attribution-Daten fuer die GUI."""
    top_features:     list[tuple[str, float]]   # (name, mean_abs_shap)
    n_records:        int
    drift_warning:    Optional[str]             # None = kein Drift
    retrain_needed:   bool
    computed_at:      Optional[datetime]        = None
    label:            str                       = "live"


def format_importance_row(rank: int, name: str, value: float) -> tuple[str, str, str]:
    """
    Bereitet eine Feature-Importance-Zeile fuer die Tabelle auf.

    Returns
    -------
    (rank_str, name, value_str)  z.B. ("#1", "feat_atr", "0.0423")
    """
    return (f"#{rank}", name, f"{value:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
#  Backend-Protocol
# ─────────────────────────────────────────────────────────────────────────────

@runtime_checkable
class AttributionBackend(Protocol):
    def fetch_attribution_snapshot(self) -> AttributionSnapshot: ...


# ─────────────────────────────────────────────────────────────────────────────
#  Qt-Widget
# ─────────────────────────────────────────────────────────────────────────────

class AttributionView(QWidget):
    """
    Live-Sicht der Feature-Attribution.

    Kann in einen Tab der Journal-View oder als eigenstaendiges Widget
    eingebunden werden.
    """

    _COLUMNS = ["Rang", "Feature", "Ø |SHAP|"]

    def __init__(
        self,
        backend: AttributionBackend,
        poll_interval_ms: int = 30_000,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._backend = backend
        self._build_ui()

        self._timer = QTimer(self)
        self._timer.setInterval(poll_interval_ms)
        self._timer.timeout.connect(self.refresh)
        self._timer.start()

        self.refresh()

    # ── UI-Aufbau ─────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        header = QLabel("Feature-Attribution  –  Rolling SHAP")
        header.setStyleSheet("font-size: 15px; font-weight: bold;")
        root.addWidget(header)

        # Meta-Info
        self._lbl_meta = QLabel("Lade…")
        self._lbl_meta.setStyleSheet("color: gray; font-size: 11px;")
        root.addWidget(self._lbl_meta)

        # Drift-Warnung
        self._lbl_warning = QLabel()
        self._lbl_warning.setWordWrap(True)
        self._lbl_warning.setStyleSheet(
            "padding: 6px; border-radius: 4px; font-weight: bold;"
        )
        self._lbl_warning.hide()
        root.addWidget(self._lbl_warning)

        # Tabelle
        self._table = QTableWidget(0, len(self._COLUMNS))
        self._table.setHorizontalHeaderLabels(self._COLUMNS)
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        root.addWidget(self._table)

        btn = QPushButton("Aktualisieren")
        btn.clicked.connect(self.refresh)
        root.addWidget(btn, alignment=Qt.AlignmentFlag.AlignRight)

    # ── Aktualisierung ────────────────────────────────────────────────────────

    @Slot()
    def refresh(self) -> None:
        try:
            snap = self._backend.fetch_attribution_snapshot()
            self._apply_snapshot(snap)
        except Exception as exc:  # noqa: BLE001
            self._lbl_meta.setText(f"Fehler: {exc}")

    def _apply_snapshot(self, snap: AttributionSnapshot) -> None:
        ts = (
            snap.computed_at.strftime("%Y-%m-%d %H:%M UTC")
            if snap.computed_at else "–"
        )
        self._lbl_meta.setText(
            f"{snap.n_records} Trades im Window  |  Stand: {ts}  |  "
            f"Label: {snap.label}"
        )

        if snap.drift_warning:
            self._lbl_warning.setText(f"⚠ {snap.drift_warning}")
            color = "#c0392b" if snap.retrain_needed else "#e67e22"
            self._lbl_warning.setStyleSheet(
                f"padding: 6px; border-radius: 4px; font-weight: bold; "
                f"background: {color}; color: white;"
            )
            self._lbl_warning.show()
        else:
            self._lbl_warning.hide()

        self._table.setRowCount(len(snap.top_features))
        for idx, (name, val) in enumerate(snap.top_features):
            rank_s, name_s, val_s = format_importance_row(idx + 1, name, val)
            self._table.setItem(idx, 0, QTableWidgetItem(rank_s))
            self._table.setItem(idx, 1, QTableWidgetItem(name_s))
            self._table.setItem(idx, 2, QTableWidgetItem(val_s))
