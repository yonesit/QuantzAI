"""
gui/views/shadow_view.py
Shadow-Mode-Vergleichs-View – hypothetische Shadow-Performance vs. Live.

Architektur (Trennung UI / Logik):
  - ShadowSnapshot          : reines Daten-Objekt (keine Qt-Abhaengigkeit)
  - compute_go_live_recommendation() : pure Funktion, testbar ohne Qt
  - ShadowBackend           : Protocol – jedes Objekt das fetch_shadow_snapshot() hat
  - ShadowComparisonView    : nur Darstellung, keine Geschaeftslogik
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Protocol, runtime_checkable

from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtGui import QColor
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
#  Daten-Typen (pure Python – kein Qt)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ShadowSnapshot:
    """Momentaufnahme der Shadow-vs-Live-Vergleichsdaten."""
    n_shadow_trades:        int
    n_live_trades:          int
    shadow_avg_confidence:  Optional[float]
    shadow_sharpe:          Optional[float]
    go_live_eligible:       bool
    recommendation:         str
    period_start:           str
    period_end:             str
    shadow_trades_rows:     list[dict] = field(default_factory=list)
    label:                  str        = "shadow"


def compute_go_live_recommendation(
    n_shadow_trades:       int,
    shadow_sharpe:         Optional[float],
    shadow_avg_confidence: Optional[float],
    min_trades:            int   = 30,
    oos_sharpe_threshold:  float = 0.5,
) -> tuple[bool, str]:
    """
    Berechnet die Go-Live-Empfehlung aus Shadow-Metriken.

    Kriterien:
      1. Mindestens min_trades Shadow-Trades.
      2. shadow_sharpe > oos_sharpe_threshold.

    Returns
    -------
    (eligible: bool, recommendation: str)
    """
    problems: list[str] = []

    if n_shadow_trades < min_trades:
        problems.append(
            f"Zu wenig Trades: {n_shadow_trades}/{min_trades}"
        )

    if shadow_sharpe is None:
        problems.append("Kein Sharpe berechenbar (zu wenig Daten)")
    elif shadow_sharpe <= oos_sharpe_threshold:
        problems.append(
            f"Sharpe {shadow_sharpe:.2f} ≤ Schwelle {oos_sharpe_threshold:.2f}"
        )

    if problems:
        return False, "Nicht bereit: " + " | ".join(problems)

    conf_str = f"{shadow_avg_confidence:.3f}" if shadow_avg_confidence is not None else "–"
    return True, (
        f"Bereit für Live-Schaltung: {n_shadow_trades} Trades, "
        f"Sharpe={shadow_sharpe:.2f}, Konfidenz={conf_str}"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Backend-Protocol
# ─────────────────────────────────────────────────────────────────────────────

@runtime_checkable
class ShadowBackend(Protocol):
    def fetch_shadow_snapshot(
        self,
        start_date,
        end_date,
        min_trades:           int   = 30,
        oos_sharpe_threshold: float = 0.5,
    ) -> ShadowSnapshot: ...


# ─────────────────────────────────────────────────────────────────────────────
#  Qt-Widget
# ─────────────────────────────────────────────────────────────────────────────

class ShadowComparisonView(QWidget):
    """
    Vergleichs-View: Shadow-Trades vs. echte Live-Performance.

    Zeigt:
      - Zusammenfassung (Shadow-Trades, Live-Trades, Sharpe, Konfidenz)
      - Go-Live-Empfehlung (gruen/rot)
      - Tabelle der letzten Shadow-Trades
    """

    _COLUMNS = ["Zeit", "Symbol", "Richtung", "Lots", "Konfidenz", "Signal", "Label"]

    def __init__(
        self,
        backend:    ShadowBackend,
        start_date = None,
        end_date   = None,
        parent:     Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._backend    = backend
        self._start_date = start_date or datetime(2020, 1, 1, tzinfo=timezone.utc)
        self._end_date   = end_date   or datetime.now(timezone.utc)

        self._build_ui()
        self.refresh()

    # ── UI-Aufbau ─────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # Header
        header = QLabel("Shadow-Mode  –  Vergleich mit Live-Performance")
        header.setStyleSheet("font-size: 16px; font-weight: bold;")
        root.addWidget(header)

        # Zusammenfassung
        summary_frame = QFrame()
        summary_frame.setFrameShape(QFrame.Shape.StyledPanel)
        summary_layout = QHBoxLayout(summary_frame)
        summary_layout.setSpacing(24)

        self._lbl_shadow_n  = self._stat_label("Shadow-Trades", "–")
        self._lbl_live_n    = self._stat_label("Live-Trades", "–")
        self._lbl_sharpe    = self._stat_label("Shadow-Sharpe", "–")
        self._lbl_conf      = self._stat_label("Ø Konfidenz", "–")

        for w in (self._lbl_shadow_n, self._lbl_live_n, self._lbl_sharpe, self._lbl_conf):
            summary_layout.addWidget(w)
        summary_layout.addStretch()

        root.addWidget(summary_frame)

        # Go-Live-Empfehlung
        self._lbl_recommendation = QLabel()
        self._lbl_recommendation.setWordWrap(True)
        self._lbl_recommendation.setStyleSheet(
            "padding: 8px; border-radius: 4px; font-weight: bold;"
        )
        root.addWidget(self._lbl_recommendation)

        # Tabelle
        self._table = QTableWidget(0, len(self._COLUMNS))
        self._table.setHorizontalHeaderLabels(self._COLUMNS)
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        root.addWidget(self._table)

        # Refresh-Button
        btn = QPushButton("Aktualisieren")
        btn.clicked.connect(self.refresh)
        root.addWidget(btn, alignment=Qt.AlignmentFlag.AlignRight)

    @staticmethod
    def _stat_label(title: str, value: str) -> QLabel:
        lbl = QLabel(f"<b>{title}</b><br>{value}")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setMinimumWidth(110)
        return lbl

    # ── Aktualisierung ────────────────────────────────────────────────────────

    @Slot()
    def refresh(self) -> None:
        """Holt einen neuen Snapshot und aktualisiert die Anzeige."""
        try:
            snap = self._backend.fetch_shadow_snapshot(
                self._start_date, self._end_date
            )
            self._apply_snapshot(snap)
        except Exception as exc:  # noqa: BLE001
            self._lbl_recommendation.setText(f"Fehler beim Laden: {exc}")
            self._lbl_recommendation.setStyleSheet(
                "padding: 8px; border-radius: 4px; font-weight: bold; "
                "background: #c0392b; color: white;"
            )

    def _apply_snapshot(self, snap: ShadowSnapshot) -> None:
        sharpe_str = f"{snap.shadow_sharpe:.2f}" if snap.shadow_sharpe is not None else "–"
        conf_str   = f"{snap.shadow_avg_confidence:.3f}" if snap.shadow_avg_confidence is not None else "–"

        self._lbl_shadow_n.setText(
            f"<b>Shadow-Trades</b><br>{snap.n_shadow_trades}"
        )
        self._lbl_live_n.setText(
            f"<b>Live-Trades</b><br>{snap.n_live_trades}"
        )
        self._lbl_sharpe.setText(f"<b>Shadow-Sharpe</b><br>{sharpe_str}")
        self._lbl_conf.setText(f"<b>Ø Konfidenz</b><br>{conf_str}")

        self._lbl_recommendation.setText(snap.recommendation)
        if snap.go_live_eligible:
            self._lbl_recommendation.setStyleSheet(
                "padding: 8px; border-radius: 4px; font-weight: bold; "
                "background: #27ae60; color: white;"
            )
        else:
            self._lbl_recommendation.setStyleSheet(
                "padding: 8px; border-radius: 4px; font-weight: bold; "
                "background: #e67e22; color: white;"
            )

        self._populate_table(snap.shadow_trades_rows)

    def _populate_table(self, rows: list[dict]) -> None:
        self._table.setRowCount(len(rows))
        for r_idx, row in enumerate(rows):
            self._table.setItem(r_idx, 0, QTableWidgetItem(str(row.get("ts", ""))))
            self._table.setItem(r_idx, 1, QTableWidgetItem(str(row.get("symbol", ""))))
            self._table.setItem(r_idx, 2, QTableWidgetItem(str(row.get("direction", ""))))
            lots = row.get("lot_size")
            self._table.setItem(r_idx, 3, QTableWidgetItem(f"{lots:.2f}" if lots is not None else "–"))
            conf = row.get("confidence")
            self._table.setItem(r_idx, 4, QTableWidgetItem(f"{conf:.3f}" if conf is not None else "–"))
            self._table.setItem(r_idx, 5, QTableWidgetItem(str(row.get("signal", ""))))
            self._table.setItem(r_idx, 6, QTableWidgetItem(str(row.get("label", ""))))
