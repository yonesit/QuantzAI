"""
gui/widgets/chart_widget.py
ChartWidget – TradingView-naher Candlestick-Chart.

Neu (gegenueber Vorversion):
  - TradingView-Kerzenfarben (chart_bull/chart_bear aus Design-Tokens)
  - Volumen-Histogramm-Subplot (untere 20 % der Canvas-Hoehe)
  - Crosshair-Cursor mit Preis- und Zeit-Label
  - Bid/Ask-Preisboxen oben links
  - Zeitraum-Buttons (1D, 5D, 1M, 3M, 6M, YTD, 1Y, 5Y, All) unterhalb des Canvas
  - TradeAnnotation-Marker (▲ BUY / ▼ SELL) mit SHAP-Tooltip
  - compute_shap_labels() – berechnet Top-N SHAP-Features via interpretability.py

Bestehende Funktionalitaet erhalten:
  - Timeframe-Buttons M1-D1 (emit timeframe_changed)
  - EMA-20 / Bollinger-Band-Overlays
  - SL / TP / Trailing-Stop Positionslinien (set_position_levels)
  - set_candles(), set_symbol(), set_timeframe(), alle Properties
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QPainter,
    QPen,
    QPolygon,
)
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from gui.design.tokens import DARK


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


@dataclass
class TradeAnnotation:
    """Markierung eines Trade-Einstiegs im Chart."""
    trade_id:    str | int
    timestamp:   datetime
    price:       float
    direction:   str            # 'buy' | 'sell'
    sl:          float | None = None
    tp:          float | None = None
    shap_labels: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
#  SHAP-Hilfsfunktion
# ─────────────────────────────────────────────────────────────────────────────

def compute_shap_labels(model, features_row, direction: str, top_n: int = 3) -> list[str]:
    """
    Berechnet Top-N SHAP-Beschreibungen aus src/models/interpretability.explain_prediction().
    Gibt [] zurueck wenn SHAP nicht verfuegbar oder Fehler auftritt.
    """
    try:
        from src.models.interpretability import explain_prediction
        shap_dict = explain_prediction(model, features_row)
        cls_name = "long" if direction.lower() == "buy" else "short"
        vals = shap_dict.get(cls_name, {})
        top = sorted(vals.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_n]
        return [f"{feat}: {v:+.3f} ({'bullisch' if v > 0 else 'baerisch'})" for feat, v in top]
    except Exception:
        return []


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
    """Bollinger Bands (SMA +/- std_mult * Sigma). Liefert (mid, upper, lower)."""
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
#  Farb-Konstanten (chart-spezifisch, nicht aus Tokens)
# ─────────────────────────────────────────────────────────────────────────────

_COLOR_BG       = DARK.bg_base       # "#0f0f11"
_COLOR_GRID     = DARK.bg_surface    # "#1a1a1f"
_COLOR_AXIS     = DARK.text_secondary
_COLOR_EMA      = DARK.accent        # "#6366f1"
_COLOR_BB_BAND  = DARK.warning       # "#f59e0b"
_COLOR_BB_MID   = DARK.neutral       # "#6b7280"
_COLOR_SL       = DARK.loss          # "#ef4444"
_COLOR_TP       = DARK.profit        # "#22c55e"
_COLOR_TRAILING = DARK.warning       # "#f59e0b"

# TradingView-Kerzenfarben (aus Design-Tokens)
_COLOR_BULL     = DARK.chart_bull    # "#26a69a"
_COLOR_BEAR     = DARK.chart_bear    # "#ef5350"
_COLOR_BUY_BOX  = DARK.chart_buy_box  # "#1565c0"
_COLOR_SELL_BOX = DARK.chart_sell_box  # "#c62828"


# ─────────────────────────────────────────────────────────────────────────────
#  _Layout – Geometrie-Helfer
# ─────────────────────────────────────────────────────────────────────────────

class _Layout:
    """Berechnet Chart-Bereichs-Geometrie inkl. Volumen-Subplot."""

    M_LEFT   = 10
    M_RIGHT  = 70
    M_TOP    = 10
    M_BOTTOM = 24
    VOL_PCT  = 0.20
    DIV_H    = 2

    def __init__(self, w: int, h: int) -> None:
        self.w = w
        self.h = h
        inner_h = h - self.M_TOP - self.M_BOTTOM
        self.vol_h   = max(20, int(inner_h * self.VOL_PCT))
        self.chart_h = max(1, inner_h - self.vol_h - self.DIV_H)
        self.chart_x = self.M_LEFT
        self.chart_y = self.M_TOP
        self.chart_w = max(1, w - self.M_LEFT - self.M_RIGHT)
        self.vol_y   = self.M_TOP + self.chart_h + self.DIV_H

    def price_y(self, price: float, p_min: float, p_rng: float) -> int:
        ratio = (price - p_min) / max(p_rng, 1e-10)
        return int(self.chart_y + self.chart_h * (1.0 - ratio))

    def price_from_y(self, y: int, p_min: float, p_rng: float) -> float:
        ratio = 1.0 - (y - self.chart_y) / max(self.chart_h, 1)
        return p_min + p_rng * ratio

    def idx_from_x(self, x: int, slot_w: int, n: int) -> int | None:
        if slot_w <= 0 or n <= 0:
            return None
        i = (x - self.chart_x) // slot_w
        return i if 0 <= i < n else None


# ─────────────────────────────────────────────────────────────────────────────
#  _ChartCanvas – Zeichen-Flaeche
# ─────────────────────────────────────────────────────────────────────────────

class _ChartCanvas(QWidget):
    """Zeichnet Candlesticks, Indikatoren, Volumen-Subplot und Overlays."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(250)
        self.setMouseTracking(True)

        # Kerzen-Daten
        self._candles:  list[CandleData] = []

        # Positionslinien
        self._sl:       float | None = None
        self._tp:       float | None = None
        self._trailing: float | None = None

        # Indikatoren
        self._show_ema: bool = False
        self._show_bb:  bool = False

        # Bid/Ask
        self._bid:      float | None = None
        self._ask:      float | None = None

        # Trade-Annotations
        self._annotations:        list[TradeAnnotation] = []
        self._annotation_markers: dict[str | int, tuple[int, int]] = {}

        # Maus-Zustand
        self._mouse_pos:   tuple[int, int] | None = None
        self._hovered_ann: TradeAnnotation | None  = None

        # Anzeigebegrenzung (None = alle Kerzen)
        self._display_limit: int | None = None

    # ── Sichtbare Kerzen ──────────────────────────────────────────────────────

    def _visible_candles(self) -> list[CandleData]:
        if self._display_limit is not None:
            return self._candles[-self._display_limit:]
        return self._candles

    # ── Annotation-Hilfsmethoden ──────────────────────────────────────────────

    def _find_annotation_idx(self, candles: list[CandleData], ann: TradeAnnotation) -> int | None:
        """Findet den Index der Kerze mit dem naechsten Zeitstempel zur Annotation."""
        if not candles:
            return None

        ann_ts = ann.timestamp
        # Normalisiere zu UTC
        if ann_ts.tzinfo is None:
            ann_ts_utc = ann_ts.replace(tzinfo=timezone.utc)
        else:
            ann_ts_utc = ann_ts.astimezone(timezone.utc)

        best_idx  = None
        best_diff = float("inf")
        for i, c in enumerate(candles):
            c_ts = c.timestamp
            if c_ts.tzinfo is None:
                c_ts_utc = c_ts.replace(tzinfo=timezone.utc)
            else:
                c_ts_utc = c_ts.astimezone(timezone.utc)
            diff = abs((c_ts_utc - ann_ts_utc).total_seconds())
            if diff < best_diff:
                best_diff = diff
                best_idx  = i
        return best_idx

    def _update_hovered_ann(self) -> None:
        """Aktualisiert _hovered_ann basierend auf Mausposition und Marker-Positionen."""
        if self._mouse_pos is None:
            self._hovered_ann = None
            return
        mx, my = self._mouse_pos
        for ann in self._annotations:
            pos = self._annotation_markers.get(ann.trade_id)
            if pos is None:
                continue
            ax, ay = pos
            dist = ((mx - ax) ** 2 + (my - ay) ** 2) ** 0.5
            if dist <= 12:
                self._hovered_ann = ann
                return
        self._hovered_ann = None

    # ── Maus-Events ───────────────────────────────────────────────────────────

    def mouseMoveEvent(self, event) -> None:
        self._mouse_pos = (event.position().x().__int__(), event.position().y().__int__())
        self._update_hovered_ann()
        self.update()

    def leaveEvent(self, event) -> None:
        self._mouse_pos   = None
        self._hovered_ann = None
        self.update()

    # ── paintEvent ────────────────────────────────────────────────────────────

    def paintEvent(self, event) -> None:
        w = self.width()
        h = self.height()
        if w == 0 or h == 0:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # 1. Hintergrund fuellen
        painter.fillRect(0, 0, w, h, QColor(_COLOR_BG))

        # 2. Keine Daten
        if not self._candles:
            painter.setPen(QColor(_COLOR_AXIS))
            painter.drawText(w // 2 - 60, h // 2, "Keine Daten vorhanden")
            return

        # 3. Layout + sichtbare Kerzen
        lay = _Layout(w, h)
        candles = self._visible_candles()
        n = len(candles)
        if n == 0:
            return

        slot_w = max(3, lay.chart_w // n)
        body_w = max(1, slot_w - 2)

        # 4. Preisbereich
        p_max = max(c.high  for c in candles)
        p_min = min(c.low   for c in candles)
        p_rng = p_max - p_min
        if p_rng < 1e-10:
            p_rng = 1.0
        # 5 % Padding
        padding = p_rng * 0.05
        p_min -= padding
        p_max += padding
        p_rng = p_max - p_min

        # Volumen-Maximalwert
        vol_max = max((c.volume for c in candles), default=1.0)
        if vol_max <= 0:
            vol_max = 1.0

        # 6. Gitter (gestrichelte horizontale Linien)
        grid_pen = QPen(QColor(_COLOR_GRID), 1, Qt.PenStyle.DashLine)
        painter.setPen(grid_pen)
        for pct in (0.25, 0.50, 0.75):
            gy = lay.price_y(p_min + p_rng * pct, p_min, p_rng)
            painter.drawLine(lay.chart_x, gy, w - lay.M_RIGHT, gy)

        # 7. Bollinger Bands (unter Kerzen)
        closes = [c.close for c in candles]
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
                    px = lay.chart_x + i * slot_w + body_w // 2
                    py = lay.price_y(v, p_min, p_rng)
                    if prev is not None:
                        painter.drawLine(prev[0], prev[1], px, py)
                    prev = (px, py)

        # 8. Kerzen + Volumen-Balken
        for i, candle in enumerate(candles):
            x     = lay.chart_x + i * slot_w
            mid_x = x + body_w // 2
            is_bull = candle.close >= candle.open
            color   = QColor(_COLOR_BULL if is_bull else _COLOR_BEAR)

            # Docht
            painter.setPen(QPen(color, 1))
            painter.drawLine(mid_x, lay.price_y(candle.high, p_min, p_rng),
                             mid_x, lay.price_y(candle.low,  p_min, p_rng))

            # Koerper
            top_y  = min(lay.price_y(candle.open,  p_min, p_rng),
                         lay.price_y(candle.close, p_min, p_rng))
            body_h = max(abs(lay.price_y(candle.close, p_min, p_rng) -
                             lay.price_y(candle.open,  p_min, p_rng)), 2)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(color))
            painter.drawRect(x, top_y, body_w, body_h)

            # Volumen-Balken (semi-transparent)
            if candle.volume > 0 and lay.vol_h > 0:
                vol_ratio = candle.volume / vol_max
                bar_h = max(1, int(lay.vol_h * vol_ratio))
                vol_color = QColor(_COLOR_BULL if is_bull else _COLOR_BEAR)
                vol_color.setAlpha(100)
                painter.setBrush(QBrush(vol_color))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawRect(x, lay.vol_y + lay.vol_h - bar_h, body_w, bar_h)

        # 9. Volumen-Trennlinie
        painter.setPen(QPen(QColor(_COLOR_GRID), 1))
        painter.drawLine(lay.chart_x, lay.vol_y, w - lay.M_RIGHT, lay.vol_y)

        # 10. EMA-Overlay
        if self._show_ema and closes:
            ema_vals = _compute_ema(closes, period=min(20, n))
            painter.setPen(QPen(QColor(_COLOR_EMA), 1))
            prev = None
            for i, v in enumerate(ema_vals):
                if v is None:
                    prev = None
                    continue
                px = lay.chart_x + i * slot_w + body_w // 2
                py = lay.price_y(v, p_min, p_rng)
                if prev is not None:
                    painter.drawLine(prev[0], prev[1], px, py)
                prev = (px, py)

        # 11. Positionslinien (SL / TP / Trailing-Stop) - volle Breite
        for level, color_str, style in [
            (self._sl,       _COLOR_SL,       Qt.PenStyle.DashLine),
            (self._tp,       _COLOR_TP,        Qt.PenStyle.DashLine),
            (self._trailing, _COLOR_TRAILING,  Qt.PenStyle.DotLine),
        ]:
            if level is not None and p_min <= level <= p_max:
                py = lay.price_y(level, p_min, p_rng)
                painter.setPen(QPen(QColor(color_str), 1, style))
                painter.drawLine(lay.chart_x, py, w - lay.M_RIGHT, py)

        # 12. Annotationen zeichnen
        self._annotation_markers.clear()
        self._draw_annotations(painter, lay, candles, slot_w, body_w, p_min, p_rng, n, w)

        # 13. Preisachse (rechts, 5 Labels)
        painter.setPen(QColor(_COLOR_AXIS))
        f = painter.font()
        orig_f = painter.font()
        f.setPointSize(7)
        painter.setFont(f)
        for pct in (0.0, 0.25, 0.5, 0.75, 1.0):
            price = p_min + p_rng * pct
            py    = lay.price_y(price, p_min, p_rng)
            painter.drawText(w - lay.M_RIGHT + 4, py + 4, f"{price:.5g}")
        painter.setFont(orig_f)

        # 14. Zeitachse (unten, alle n//6 Kerzen)
        painter.setPen(QColor(_COLOR_AXIS))
        f2 = painter.font()
        f2.setPointSize(7)
        painter.setFont(f2)
        step = max(1, n // 6)
        for i in range(0, n, step):
            tx = lay.chart_x + i * slot_w + body_w // 2
            ts = candles[i].timestamp
            label = ts.strftime("%H:%M") if hasattr(ts, "strftime") else ""
            painter.drawText(tx - 15, h - 4, label)
        painter.setFont(orig_f)

        # 15. Volumen-Label oben links im Volumen-Bereich
        painter.setPen(QColor(_COLOR_AXIS))
        f3 = painter.font()
        f3.setPointSize(7)
        painter.setFont(f3)
        last_vol = candles[-1].volume if candles else 0.0
        painter.drawText(lay.chart_x + 2, lay.vol_y + 12, f"Vol  {last_vol:,.0f}")
        painter.setFont(orig_f)

        # 16. Bid/Ask-Boxen
        if self._bid is not None or self._ask is not None:
            self._draw_bid_ask(painter)

        # 17. Crosshair
        if self._mouse_pos is not None:
            self._draw_crosshair(painter, lay, p_min, p_rng, candles, slot_w, n, w, h)

        # 18. SHAP-Tooltip
        if self._hovered_ann is not None and self._hovered_ann.shap_labels:
            self._draw_shap_tooltip(painter, self._hovered_ann)

    # ── Bid/Ask-Boxen ─────────────────────────────────────────────────────────

    def _draw_bid_ask(self, painter: QPainter) -> None:
        """Zeichnet Bid/Ask-Preisboxen oben links."""
        f = painter.font()
        orig_f = f
        f_bold = painter.font()
        f_bold.setPointSize(8)
        f_bold.setBold(True)

        white = QColor(255, 255, 255)

        # SELL-Box (Bid)
        if self._bid is not None:
            sell_rect_x = 4
            sell_rect_y = 4
            sell_rect_w = 82
            sell_rect_h = 36
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor(_COLOR_SELL_BOX)))
            painter.drawRect(sell_rect_x, sell_rect_y, sell_rect_w, sell_rect_h)

            painter.setPen(white)
            painter.setFont(f_bold)
            painter.drawText(sell_rect_x, sell_rect_y, sell_rect_w, sell_rect_h // 2,
                             Qt.AlignmentFlag.AlignCenter, f"{self._bid:.5f}")

            f_sm = painter.font()
            f_sm.setPointSize(7)
            f_sm.setBold(False)
            painter.setFont(f_sm)
            painter.drawText(sell_rect_x, sell_rect_y + sell_rect_h // 2,
                             sell_rect_w, sell_rect_h // 2,
                             Qt.AlignmentFlag.AlignCenter, "SELL")

        # BUY-Box (Ask)
        if self._ask is not None:
            buy_rect_x = 90
            buy_rect_y = 4
            buy_rect_w = 82
            buy_rect_h = 36
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor(_COLOR_BUY_BOX)))
            painter.drawRect(buy_rect_x, buy_rect_y, buy_rect_w, buy_rect_h)

            painter.setPen(white)
            painter.setFont(f_bold)
            painter.drawText(buy_rect_x, buy_rect_y, buy_rect_w, buy_rect_h // 2,
                             Qt.AlignmentFlag.AlignCenter, f"{self._ask:.5f}")

            f_sm = painter.font()
            f_sm.setPointSize(7)
            f_sm.setBold(False)
            painter.setFont(f_sm)
            painter.drawText(buy_rect_x, buy_rect_y + buy_rect_h // 2,
                             buy_rect_w, buy_rect_h // 2,
                             Qt.AlignmentFlag.AlignCenter, "BUY")

        # Spread
        if self._bid is not None and self._ask is not None:
            spread = self._ask - self._bid
            painter.setPen(QColor(_COLOR_AXIS))
            f_sp = painter.font()
            f_sp.setPointSize(7)
            f_sp.setBold(False)
            painter.setFont(f_sp)
            painter.drawText(176, 22, f"Spread: {spread:.5f}")

        painter.setFont(orig_f)

    # ── Crosshair ─────────────────────────────────────────────────────────────

    def _draw_crosshair(
        self,
        painter: QPainter,
        lay: _Layout,
        p_min: float,
        p_rng: float,
        candles: list[CandleData],
        slot_w: int,
        n: int,
        w: int,
        h: int,
    ) -> None:
        """Zeichnet Crosshair-Cursor mit Preis- und Zeit-Label."""
        if self._mouse_pos is None:
            return
        mx, my = self._mouse_pos

        # Pruefen ob Maus im Chart- oder Volumen-Bereich ist
        in_chart_x = lay.chart_x <= mx <= w - lay.M_RIGHT
        in_chart_y = lay.chart_y <= my <= lay.vol_y + lay.vol_h
        if not (in_chart_x and in_chart_y):
            return

        cross_color = QColor(136, 136, 136, 160)
        cross_pen = QPen(cross_color, 1, Qt.PenStyle.DashLine)
        painter.setPen(cross_pen)

        # Horizontale Linie (nur im Preis-Chart-Bereich)
        if lay.chart_y <= my <= lay.vol_y:
            painter.drawLine(lay.chart_x, my, w - lay.M_RIGHT, my)

        # Vertikale Linie (gesamter Chart+Volumen-Bereich)
        painter.drawLine(mx, lay.chart_y, mx, lay.vol_y + lay.vol_h)

        # Preis-Label rechts
        if lay.chart_y <= my <= lay.vol_y:
            price_at_y = lay.price_from_y(my, p_min, p_rng)
            label_x = w - lay.M_RIGHT
            label_y = my - 9
            label_w = lay.M_RIGHT - 2
            label_h = 18
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor(_COLOR_SELL_BOX)))
            painter.drawRect(label_x, label_y, label_w, label_h)
            painter.setPen(QColor(255, 255, 255))
            f = painter.font()
            orig_f = f
            f.setPointSize(7)
            painter.setFont(f)
            painter.drawText(label_x, label_y, label_w, label_h,
                             Qt.AlignmentFlag.AlignCenter, f"{price_at_y:.5g}")
            painter.setFont(orig_f)

        # Zeit-Label unten
        idx = lay.idx_from_x(mx, slot_w, n)
        if idx is not None and 0 <= idx < len(candles):
            ts = candles[idx].timestamp
            time_label = ts.strftime("%Y-%m-%d %H:%M") if hasattr(ts, "strftime") else ""
            lbl_w = 120
            lbl_h = 16
            lbl_x = max(lay.chart_x, min(mx - lbl_w // 2, w - lay.M_RIGHT - lbl_w))
            lbl_y = lay.vol_y + lay.vol_h + 2
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor(_COLOR_SELL_BOX)))
            painter.drawRect(lbl_x, lbl_y, lbl_w, lbl_h)
            painter.setPen(QColor(255, 255, 255))
            f2 = painter.font()
            orig_f2 = f2
            f2.setPointSize(7)
            painter.setFont(f2)
            painter.drawText(lbl_x, lbl_y, lbl_w, lbl_h,
                             Qt.AlignmentFlag.AlignCenter, time_label)
            painter.setFont(orig_f2)

    # ── Trade-Annotations ─────────────────────────────────────────────────────

    def _draw_annotations(
        self,
        painter: QPainter,
        lay: _Layout,
        candles: list[CandleData],
        slot_w: int,
        body_w: int,
        p_min: float,
        p_rng: float,
        n: int,
        w: int,
    ) -> None:
        """Zeichnet Trade-Einstiegs-Marker und partielle SL/TP-Linien."""
        for ann in self._annotations:
            idx = self._find_annotation_idx(candles, ann)
            if idx is None:
                continue

            entry_x = lay.chart_x + idx * slot_w + body_w // 2
            ey      = lay.price_y(ann.price, p_min, p_rng)

            if ann.direction.lower() == "buy":
                # Gruenes Dreieck nach oben, unterhalb des Einstiegspreises
                color = QColor("#26a69a")
                pts = QPolygon([
                    QPoint(entry_x,     ey + 2),
                    QPoint(entry_x - 7, ey + 16),
                    QPoint(entry_x + 7, ey + 16),
                ])
            else:
                # Rotes Dreieck nach unten, oberhalb des Einstiegspreises
                color = QColor("#ef5350")
                pts = QPolygon([
                    QPoint(entry_x,     ey - 2),
                    QPoint(entry_x - 7, ey - 16),
                    QPoint(entry_x + 7, ey - 16),
                ])

            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(color))
            painter.drawPolygon(pts)

            # Partielles SL vom Einstieg nach rechts
            if ann.sl is not None and p_min <= ann.sl <= p_max_from_rng(p_min, p_rng):
                sl_y = lay.price_y(ann.sl, p_min, p_rng)
                sl_pen = QPen(QColor(_COLOR_SL), 1, Qt.PenStyle.DashLine)
                painter.setPen(sl_pen)
                painter.drawLine(entry_x, sl_y, w - lay.M_RIGHT, sl_y)

            # Partielles TP vom Einstieg nach rechts
            if ann.tp is not None and p_min <= ann.tp <= p_max_from_rng(p_min, p_rng):
                tp_y = lay.price_y(ann.tp, p_min, p_rng)
                tp_pen = QPen(QColor(_COLOR_TP), 1, Qt.PenStyle.DashLine)
                painter.setPen(tp_pen)
                painter.drawLine(entry_x, tp_y, w - lay.M_RIGHT, tp_y)

            # Marker-Position speichern
            self._annotation_markers[ann.trade_id] = (entry_x, ey)

    # ── SHAP-Tooltip ──────────────────────────────────────────────────────────

    def _draw_shap_tooltip(self, painter: QPainter, ann: TradeAnnotation) -> None:
        """Zeichnet SHAP-Informations-Tooltip fuer einen Trade."""
        pos = self._annotation_markers.get(ann.trade_id)
        if pos is None:
            return
        ax, ay = pos

        lines = [
            f"Trade #{ann.trade_id} | {ann.direction.upper()}",
            "Top-3 SHAP:",
        ] + ann.shap_labels

        # Font fuer Tooltip
        f = painter.font()
        orig_f = f
        f_tt = painter.font()
        f_tt.setPointSize(8)
        painter.setFont(f_tt)

        fm = painter.fontMetrics()
        line_h    = fm.height() + 2
        max_width = max(fm.horizontalAdvance(line) for line in lines)
        box_w     = max_width + 16
        box_h     = len(lines) * line_h + 10
        box_x     = ax + 14
        box_y     = ay - box_h // 2

        # Box bleibt im sichtbaren Bereich
        w = self.width()
        h = self.height()
        if box_x + box_w > w - _Layout.M_RIGHT:
            box_x = ax - box_w - 14
        if box_y < 0:
            box_y = 0
        if box_y + box_h > h:
            box_y = h - box_h

        # Hintergrund
        bg_color = QColor(DARK.bg_elevated)
        border_color = QColor(DARK.border)
        painter.setPen(QPen(border_color, 1))
        painter.setBrush(QBrush(bg_color))
        painter.drawRect(box_x, box_y, box_w, box_h)

        # Text
        painter.setPen(QColor(255, 255, 255))
        for i, line in enumerate(lines):
            ty = box_y + 6 + i * line_h + fm.ascent()
            painter.drawText(box_x + 8, ty, line)

        painter.setFont(orig_f)


def p_max_from_rng(p_min: float, p_rng: float) -> float:
    """Hilfsfunktion: berechnet p_max aus p_min und p_rng."""
    return p_min + p_rng


# ─────────────────────────────────────────────────────────────────────────────
#  Perioden-Konfiguration
# ─────────────────────────────────────────────────────────────────────────────

_PERIOD_DAYS: dict[str, int] = {
    "1D":  1,
    "5D":  5,
    "1M":  22,
    "3M":  66,
    "6M":  130,
    "1Y":  260,
    "5Y":  1300,
}

_PERIOD_LABELS = ["1D", "5D", "1M", "3M", "6M", "YTD", "1Y", "5Y", "All"]


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
    annotations        – Liste der Trade-Annotationen
    display_limit      – aktuelle Anzeigebeschraenkung (None = alle)
    bid / ask          – aktuelle Bid/Ask-Preise
    """

    timeframe_changed = Signal(object)   # Timeframe-Enum-Wert

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("chart_widget")

        self._timeframe:      Timeframe              = Timeframe.H1
        self._indicators:     dict[str, bool]        = {"ema": False, "bb": False}
        self._tf_buttons:     dict[Timeframe, QPushButton] = {}
        self._period_buttons: dict[str, QPushButton] = {}

        self._canvas = _ChartCanvas()
        self._build()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── Toolbar ───────────────────────────────────────────────────────────
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

        # ── Perioden-Buttons unterhalb des Canvas ─────────────────────────────
        period_bar = QWidget()
        period_bar.setObjectName("chart_period_bar")
        pb = QHBoxLayout(period_bar)
        pb.setContentsMargins(8, 2, 8, 2)
        pb.setSpacing(4)

        for label in _PERIOD_LABELS:
            btn = QPushButton(label)
            btn.setObjectName(f"period_btn_{label.lower()}")
            btn.setCheckable(True)
            btn.setMaximumWidth(42)
            btn.setMinimumWidth(32)
            btn.clicked.connect(lambda _c, lbl=label: self.set_display_period(lbl))
            self._period_buttons[label] = btn
            pb.addWidget(btn)

        pb.addStretch()
        # Standard: "All" ist aktiv
        self._period_buttons["All"].setChecked(True)

        layout.addWidget(period_bar)

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

    def set_bid_ask(self, bid: float | None, ask: float | None) -> None:
        """Setzt Bid/Ask-Preise fuer die Preisboxen."""
        self._canvas._bid = bid
        self._canvas._ask = ask
        self._canvas.update()

    def add_trade_annotation(self, annotation: TradeAnnotation) -> None:
        """Fuegt einen Trade-Einstiegs-Marker hinzu."""
        self._canvas._annotations.append(annotation)
        self._canvas.update()

    def clear_annotations(self) -> None:
        """Loescht alle Trade-Annotationen."""
        self._canvas._annotations.clear()
        self._canvas._annotation_markers.clear()
        self._canvas.update()

    def set_display_period(self, period_label: str) -> None:
        """Setzt den Anzeige-Zeitraum (z.B. '1D', '1M', 'All')."""
        tf_mins = self._timeframe.minutes
        trading_mins_per_day = 480  # 8h * 60min

        if period_label == "All":
            self._canvas._display_limit = None
        elif period_label == "YTD":
            from datetime import date
            today = date.today()
            ytd_days = max(1, int((today - date(today.year, 1, 1)).days * 0.71))
            self._canvas._display_limit = max(10, int(ytd_days * trading_mins_per_day / tf_mins))
        else:
            days = _PERIOD_DAYS.get(period_label, 0)
            if days:
                self._canvas._display_limit = max(10, int(days * trading_mins_per_day / tf_mins))
            else:
                self._canvas._display_limit = None

        # Button-Zustand aktualisieren
        for label, btn in self._period_buttons.items():
            btn.setChecked(label == period_label)

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

    @property
    def annotations(self) -> list[TradeAnnotation]:
        return list(self._canvas._annotations)

    @property
    def display_limit(self) -> int | None:
        return self._canvas._display_limit

    @property
    def bid(self) -> float | None:
        return self._canvas._bid

    @property
    def ask(self) -> float | None:
        return self._canvas._ask
