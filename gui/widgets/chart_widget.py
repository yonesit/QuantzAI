"""
gui/widgets/chart_widget.py
ChartWidget – Candlestick-Chart mit Zeitebenen-Umschaltung und Indikator-Overlays.

Komponenten:
  CandleData  – reines Daten-Objekt (OHLCV + Zeitstempel)
  Timeframe   – Enum M1 bis D1
  _ChartCanvas – innere Zeichen-Flaeche (paintEvent mit QPainter)
  ChartWidget  – oeffentliche Klasse: Toolbar + Canvas

Indikatoren (ein-/ausblendbar):
  EMA-20       – exponentieller gleitender Durchschnitt
  BB-20        – Bollinger Bands (SMA ± 2 Sigma)

Positionslinien:
  SL   – rote gestrichelte Horizontallinie
  TP   – gruene gestrichelte Horizontallinie
  Trailing-Stop – gelbe gepunktete Horizontallinie

Testbarkeit:
  set_candles(), set_indicator_visible(), set_position_levels()
  aendern nur internen Zustand und loesen update() aus.
  Rendering-Tests pruefen ausschliesslich, dass kein Absturz auftritt.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Enums & Datenklassen (pure Python)
# ─────────────────────────────────────────────────────────────────────────────

class Timeframe(Enum):
    M1  = ("M1",  1)
    M5  = ("M5",  5)
    M15 = ("M15", 15)
    M30 = ("M30", 30)
    H1  = ("H1",  60)
    H4  = ("H4",  240)
    D1  = ("D1",  1440)

    def __init__(self, label: str, minutes: int) -> None:
        self.label   = label
        self.minutes = minutes


@dataclass
class CandleData:
    timestamp: datetime
    open:      float
    high:      float
    low:       float
    close:     float
    volume:    float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  Indikator-Berechnungen (pure Python, testbar ohne Qt)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_ema(closes: list[float], period: int = 20) -> list[float | None]:
    """Exponentiell gleitender Durchschnitt. Liefert None fuer die ersten (period-1) Werte."""
    if not closes or period < 1:
        return []
    effective = min(period, len(closes))
    if effective == 1:
        return list(closes)
    result: list[float | None] = [None] * (effective - 1)
    ema = sum(closes[:effective]) / effective
    result.append(ema)
    k = 2.0 / (effective + 1)
    for price in closes[effective:]:
        ema = price * k + ema * (1.0 - k)
        result.append(ema)
    return result


def _compute_bollinger(
    closes:   list[float],
    period:   int   = 20,
    std_mult: float = 2.0,
) -> tuple[list[float | None], list[float | None], list[float | None]]:
    """Bollinger Bands (SMA ± std_mult * Sigma). Liefert (mid, upper, lower)."""
    n   = len(closes)
    eff = min(period, max(2, n))
    mid_b:   list[float | None] = [None] * n
    upper_b: list[float | None] = [None] * n
    lower_b: list[float | None] = [None] * n
    for i in range(eff - 1, n):
        window   = closes[i - eff + 1: i + 1]
        avg      = sum(window) / eff
        variance = sum((x - avg) ** 2 for x in window) / eff
        std      = variance ** 0.5
        mid_b[i]   = avg
        upper_b[i] = avg + std_mult * std
        lower_b[i] = avg - std_mult * std
    return mid_b, upper_b, lower_b


# ─────────────────────────────────────────────────────────────────────────────
#  _ChartCanvas – Zeichen-Flaeche
# ─────────────────────────────────────────────────────────────────────────────

_COLOR_BULL      = "#22c55e"
_COLOR_BEAR      = "#ef4444"
_COLOR_EMA       = "#6366f1"
_COLOR_BB_BAND   = "#f59e0b"
_COLOR_BB_MID    = "#6b7280"
_COLOR_SL        = "#ef4444"
_COLOR_TP        = "#22c55e"
_COLOR_TRAILING  = "#f59e0b"
_COLOR_AXIS      = "#6b7280"
_COLOR_BG        = "#0f0f11"
_COLOR_GRID      = "#1a1a1f"


class _ChartCanvas(QWidget):
    """Zeichnet Candlesticks, Indikatoren und Positionslinien via QPainter."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(250)

        self._candles:  list[CandleData] = []
        self._sl:       float | None = None
        self._tp:       float | None = None
        self._trailing: float | None = None
        self._show_ema: bool = False
        self._show_bb:  bool = False

    # ── paintEvent ────────────────────────────────────────────────────────────

    def paintEvent(self, event) -> None:
        w = self.width()
        h = self.height()
        if w == 0 or h == 0:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Hintergrund
        painter.fillRect(0, 0, w, h, QColor(_COLOR_BG))

        if not self._candles:
            painter.setPen(QColor(_COLOR_AXIS))
            painter.drawText(w // 2 - 50, h // 2, "Keine Daten vorhanden")
            return

        # Chart-Bereich
        m_left, m_right, m_top, m_bottom = 10, 65, 10, 25
        cw = w - m_left - m_right
        ch = h - m_top  - m_bottom

        # Preisbereich
        p_max = max(c.high  for c in self._candles)
        p_min = min(c.low   for c in self._candles)
        p_rng = p_max - p_min
        if p_rng < 1e-10:
            p_rng = 1.0

        def price_y(price: float) -> int:
            ratio = (price - p_min) / p_rng
            return int(m_top + ch * (1.0 - ratio))

        n             = len(self._candles)
        slot_w        = max(3, cw // max(n, 1))
        body_w        = max(1, slot_w - 2)

        # Gitter (3 horizontale Linien)
        painter.setPen(QPen(QColor(_COLOR_GRID), 1))
        for pct in (0.25, 0.5, 0.75):
            y = price_y(p_min + p_rng * pct)
            painter.drawLine(m_left, y, w - m_right, y)

        # Kerzen
        closes = [c.close for c in self._candles]
        for i, candle in enumerate(self._candles):
            x     = m_left + i * slot_w
            mid_x = x + body_w // 2
            is_bull = candle.close >= candle.open
            color   = QColor(_COLOR_BULL if is_bull else _COLOR_BEAR)

            painter.setPen(QPen(color, 1))
            painter.drawLine(mid_x, price_y(candle.high), mid_x, price_y(candle.low))

            top_y  = min(price_y(candle.open), price_y(candle.close))
            body_h = max(abs(price_y(candle.close) - price_y(candle.open)), 2)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(color))
            painter.drawRect(x, top_y, body_w, body_h)

        # EMA-Overlay
        if self._show_ema and closes:
            ema_vals = _compute_ema(closes, period=min(20, n))
            painter.setPen(QPen(QColor(_COLOR_EMA), 1))
            prev = None
            for i, v in enumerate(ema_vals):
                if v is None:
                    prev = None
                    continue
                px = m_left + i * slot_w + body_w // 2
                py = price_y(v)
                if prev is not None:
                    painter.drawLine(prev[0], prev[1], px, py)
                prev = (px, py)

        # Bollinger-Overlay
        if self._show_bb and closes:
            mid_b, upper_b, lower_b = _compute_bollinger(closes, period=min(20, n))
            for band, color_str, style in [
                (upper_b, _COLOR_BB_BAND, Qt.PenStyle.DotLine),
                (mid_b,   _COLOR_BB_MID,  Qt.PenStyle.DotLine),
                (lower_b, _COLOR_BB_BAND, Qt.PenStyle.DotLine),
            ]:
                painter.setPen(QPen(QColor(color_str), 1, style))
                prev = None
                for i, v in enumerate(band):
                    if v is None:
                        prev = None
                        continue
                    px = m_left + i * slot_w + body_w // 2
                    py = price_y(v)
                    if prev is not None:
                        painter.drawLine(prev[0], prev[1], px, py)
                    prev = (px, py)

        # Positionslinien (SL / TP / Trailing-Stop)
        for level, color_str, style in [
            (self._sl,       _COLOR_SL,       Qt.PenStyle.DashLine),
            (self._tp,       _COLOR_TP,        Qt.PenStyle.DashLine),
            (self._trailing, _COLOR_TRAILING,  Qt.PenStyle.DotLine),
        ]:
            if level is not None and p_min <= level <= p_max:
                py = price_y(level)
                painter.setPen(QPen(QColor(color_str), 1, style))
                painter.drawLine(m_left, py, w - m_right, py)

        # Preisachse (rechts)
        painter.setPen(QColor(_COLOR_AXIS))
        f = painter.font()
        f.setPointSize(7)
        painter.setFont(f)
        for pct in (0.0, 0.25, 0.5, 0.75, 1.0):
            price = p_min + p_rng * pct
            py    = price_y(price)
            painter.drawText(w - m_right + 4, py + 4, f"{price:.5g}")


# ─────────────────────────────────────────────────────────────────────────────
#  ChartWidget – oeffentliche Klasse
# ─────────────────────────────────────────────────────────────────────────────

class ChartWidget(QWidget):
    """
    Candlestick-Chart mit umschaltbaren Zeitebenen und Indikator-Overlays.

    Signals
    -------
    timeframe_changed(Timeframe)  – wird emittiert wenn Zeitebene geaendert wird.

    Properties
    ----------
    current_timeframe  – aktive Zeitebene (Standard: H1)
    ema_visible        – ob EMA-Overlay aktiv ist
    bb_visible         – ob Bollinger-Overlay aktiv ist
    candles_count      – Anzahl geladener Kerzen
    sl / tp / trailing_stop – aktuelle Positionslinien-Werte
    """

    timeframe_changed = Signal(object)   # Timeframe-Enum-Wert

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("chart_widget")

        self._timeframe:   Timeframe          = Timeframe.H1
        self._indicators:  dict[str, bool]    = {"ema": False, "bb": False}
        self._tf_buttons:  dict[Timeframe, QPushButton] = {}

        self._canvas = _ChartCanvas()
        self._build()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Toolbar
        toolbar = QWidget()
        toolbar.setObjectName("chart_toolbar")
        tb = QHBoxLayout(toolbar)
        tb.setContentsMargins(8, 4, 8, 4)
        tb.setSpacing(4)

        # Zeitebenen-Buttons
        for tf in Timeframe:
            btn = QPushButton(tf.label)
            btn.setObjectName(f"tf_btn_{tf.name.lower()}")
            btn.setCheckable(True)
            btn.setMaximumWidth(42)
            btn.setMinimumWidth(32)
            btn.clicked.connect(lambda _c, t=tf: self._on_tf_clicked(t))
            self._tf_buttons[tf] = btn
            tb.addWidget(btn)

        self._tf_buttons[Timeframe.H1].setChecked(True)
        tb.addSpacing(12)

        # Indikator-Checkboxen
        self._ema_cb = QCheckBox("EMA")
        self._ema_cb.setObjectName("ema_checkbox")
        self._ema_cb.setToolTip("EMA-20 ein-/ausblenden")
        self._ema_cb.toggled.connect(lambda on: self.set_indicator_visible("ema", on))
        tb.addWidget(self._ema_cb)

        self._bb_cb = QCheckBox("BB")
        self._bb_cb.setObjectName("bb_checkbox")
        self._bb_cb.setToolTip("Bollinger Bands ein-/ausblenden")
        self._bb_cb.toggled.connect(lambda on: self.set_indicator_visible("bb", on))
        tb.addWidget(self._bb_cb)

        tb.addStretch()

        # Symbol-Bezeichnung (wird extern gesetzt)
        self._symbol_label = QLabel("–")
        self._symbol_label.setObjectName("chart_symbol_label")
        f = self._symbol_label.font()
        f.setBold(True)
        self._symbol_label.setFont(f)
        tb.addWidget(self._symbol_label)

        layout.addWidget(toolbar)
        layout.addWidget(self._canvas, stretch=1)

    # ── Signalhandler ─────────────────────────────────────────────────────────

    def _on_tf_clicked(self, tf: Timeframe) -> None:
        if tf is self._timeframe:
            self._tf_buttons[tf].setChecked(True)
            return
        self._tf_buttons[self._timeframe].setChecked(False)
        self._timeframe = tf
        self._tf_buttons[tf].setChecked(True)
        self.timeframe_changed.emit(tf)

    # ── Oeffentliche Methoden ─────────────────────────────────────────────────

    def set_symbol(self, symbol: str) -> None:
        """Setzt den angezeigten Symbol-Namen in der Toolbar."""
        self._symbol_label.setText(symbol)

    def set_timeframe(self, tf: Timeframe) -> None:
        """Wechselt programmatisch die Zeitebene und emittiert timeframe_changed."""
        self._on_tf_clicked(tf)

    def set_candles(self, candles: list[CandleData]) -> None:
        """Ersetzt die Kerzendaten und loest einen Repaint aus."""
        self._canvas._candles = list(candles)
        self._canvas.update()

    def set_position_levels(
        self,
        sl:       float | None = None,
        tp:       float | None = None,
        trailing: float | None = None,
    ) -> None:
        """Setzt SL/TP/Trailing-Stop als Linien im Chart."""
        self._canvas._sl       = sl
        self._canvas._tp       = tp
        self._canvas._trailing = trailing
        self._canvas.update()

    def set_indicator_visible(self, name: str, visible: bool) -> None:
        """Schaltet einen Indikator ein oder aus ('ema' oder 'bb')."""
        if name not in self._indicators:
            return
        self._indicators[name]  = visible
        self._canvas._show_ema  = self._indicators["ema"]
        self._canvas._show_bb   = self._indicators["bb"]
        # Checkbox-Zustand synchronisieren wenn extern aufgerufen
        if name == "ema" and self._ema_cb.isChecked() != visible:
            self._ema_cb.blockSignals(True)
            self._ema_cb.setChecked(visible)
            self._ema_cb.blockSignals(False)
        if name == "bb" and self._bb_cb.isChecked() != visible:
            self._bb_cb.blockSignals(True)
            self._bb_cb.setChecked(visible)
            self._bb_cb.blockSignals(False)
        self._canvas.update()

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def current_timeframe(self) -> Timeframe:
        return self._timeframe

    @property
    def ema_visible(self) -> bool:
        return self._indicators["ema"]

    @property
    def bb_visible(self) -> bool:
        return self._indicators["bb"]

    @property
    def candles_count(self) -> int:
        return len(self._canvas._candles)

    @property
    def sl(self) -> float | None:
        return self._canvas._sl

    @property
    def tp(self) -> float | None:
        return self._canvas._tp

    @property
    def trailing_stop(self) -> float | None:
        return self._canvas._trailing

    @property
    def timeframe_buttons(self) -> dict[Timeframe, QPushButton]:
        return dict(self._tf_buttons)

    @property
    def ema_checkbox(self) -> QCheckBox:
        return self._ema_cb

    @property
    def bb_checkbox(self) -> QCheckBox:
        return self._bb_cb
