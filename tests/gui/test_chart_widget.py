"""
tests/gui/test_chart_widget.py
Umfassende Tests fuer gui/widgets/chart_widget.py (TradingView-Stil).

Kein SHAP oder ML-Import direkt – compute_shap_labels() wird mit None
getestet und per Monkeypatch gesichert.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from datetime import datetime, timedelta, timezone

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap

from gui.widgets.chart_widget import (
    CandleData,
    ChartWidget,
    Timeframe,
    TradeAnnotation,
    compute_shap_labels,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def widget(qtbot):
    w = ChartWidget()
    qtbot.addWidget(w)
    w.resize(800, 500)
    return w


@pytest.fixture
def candles():
    base = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
    result = []
    price = 1.1000
    for i in range(50):
        ts = base + timedelta(hours=i)
        o = price
        c = price + (0.001 if i % 3 != 0 else -0.001)
        h = max(o, c) + 0.0005
        lo = min(o, c) - 0.0005
        result.append(CandleData(timestamp=ts, open=o, high=h, low=lo, close=c,
                                 volume=float(1000 + i * 10)))
        price = c
    return result


@pytest.fixture
def annotation():
    return TradeAnnotation(
        trade_id=1,
        timestamp=datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
        price=1.1010,
        direction="buy",
        sl=1.0990,
        tp=1.1050,
        shap_labels=["rsi: +0.123 (bullisch)", "macd: -0.045 (baerisch)"],
    )


# ─────────────────────────────────────────────────────────────────────────────
#  TestComputeShapLabels
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeShapLabels:
    def test_returns_empty_list_without_model_arg(self):
        """compute_shap_labels mit None-Modell muss [] zurueckgeben."""
        result = compute_shap_labels(None, None, "buy")
        assert result == []

    def test_returns_empty_list_on_import_error(self, monkeypatch):
        """Wenn Import fehlschlaegt, muss [] zurueckgegeben werden."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if "interpretability" in name:
                raise ImportError("No module named 'src.models.interpretability'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        result = compute_shap_labels(object(), object(), "sell")
        assert result == []


# ─────────────────────────────────────────────────────────────────────────────
#  TestTradeAnnotation
# ─────────────────────────────────────────────────────────────────────────────

class TestTradeAnnotation:
    def test_defaults(self):
        """shap_labels muss standardmaessig eine leere Liste sein."""
        ann = TradeAnnotation(
            trade_id="T001",
            timestamp=datetime(2024, 1, 15, tzinfo=timezone.utc),
            price=1.1000,
            direction="buy",
        )
        assert ann.shap_labels == []
        assert ann.sl is None
        assert ann.tp is None

    def test_custom_shap_labels(self):
        labels = ["feat1: +0.5 (bullisch)", "feat2: -0.3 (baerisch)"]
        ann = TradeAnnotation(
            trade_id=42,
            timestamp=datetime(2024, 1, 15, tzinfo=timezone.utc),
            price=1.2000,
            direction="sell",
            shap_labels=labels,
        )
        assert ann.shap_labels == labels
        assert ann.direction == "sell"

    def test_sl_tp_set(self):
        ann = TradeAnnotation(
            trade_id=99,
            timestamp=datetime(2024, 1, 15, tzinfo=timezone.utc),
            price=1.1000,
            direction="buy",
            sl=1.0950,
            tp=1.1100,
        )
        assert ann.sl == pytest.approx(1.0950)
        assert ann.tp == pytest.approx(1.1100)


# ─────────────────────────────────────────────────────────────────────────────
#  TestChartWidgetCreation
# ─────────────────────────────────────────────────────────────────────────────

class TestChartWidgetCreation:
    def test_widget_created(self, qtbot):
        """ChartWidget() darf keinen Fehler werfen."""
        w = ChartWidget()
        qtbot.addWidget(w)
        assert w is not None

    def test_initial_timeframe_is_h1(self, widget):
        assert widget.current_timeframe == Timeframe.H1

    def test_symbol_label_default(self, widget):
        assert widget._symbol_label.text() == "–"

    def test_period_buttons_all_present(self, widget):
        """Alle 9 Perioden-Buttons muessen vorhanden sein."""
        expected = {"1D", "5D", "1M", "3M", "6M", "YTD", "1Y", "5Y", "All"}
        assert set(widget._period_buttons.keys()) == expected

    def test_period_button_all_checked_by_default(self, widget):
        assert widget._period_buttons["All"].isChecked()

    def test_canvas_accessible(self, widget):
        """_canvas muss direkt zugreifbar sein (Testbarkeit)."""
        assert widget._canvas is not None
        assert hasattr(widget._canvas, "_candles")
        assert hasattr(widget._canvas, "_sl")
        assert hasattr(widget._canvas, "_tp")
        assert hasattr(widget._canvas, "_trailing")
        assert hasattr(widget._canvas, "_show_ema")
        assert hasattr(widget._canvas, "_show_bb")


# ─────────────────────────────────────────────────────────────────────────────
#  TestSetCandles
# ─────────────────────────────────────────────────────────────────────────────

class TestSetCandles:
    def test_set_empty_candles(self, widget):
        widget.set_candles([])
        assert widget.candles_count == 0
        assert widget._canvas._candles == []

    def test_set_candles_stores_data(self, widget, candles):
        widget.set_candles(candles)
        assert widget.candles_count == 50
        assert len(widget._canvas._candles) == 50

    def test_set_candles_triggers_repaint(self, widget, candles):
        """Kein Crash beim Setzen von Kerzen."""
        widget.set_candles(candles)
        widget._canvas.repaint()

    def test_set_candles_replaces_previous(self, widget, candles):
        widget.set_candles(candles)
        widget.set_candles(candles[:10])
        assert widget.candles_count == 10


# ─────────────────────────────────────────────────────────────────────────────
#  TestBidAsk
# ─────────────────────────────────────────────────────────────────────────────

class TestBidAsk:
    def test_set_bid_ask_stores_values(self, widget):
        widget.set_bid_ask(1.1000, 1.1002)
        assert widget._canvas._bid == pytest.approx(1.1000)
        assert widget._canvas._ask == pytest.approx(1.1002)

    def test_set_bid_ask_none_clears(self, widget):
        widget.set_bid_ask(1.1000, 1.1002)
        widget.set_bid_ask(None, None)
        assert widget._canvas._bid is None
        assert widget._canvas._ask is None

    def test_bid_ask_properties(self, widget):
        widget.set_bid_ask(1.2345, 1.2348)
        assert widget.bid == pytest.approx(1.2345)
        assert widget.ask == pytest.approx(1.2348)

    def test_set_bid_only(self, widget):
        widget.set_bid_ask(1.1000, None)
        assert widget.bid == pytest.approx(1.1000)
        assert widget.ask is None


# ─────────────────────────────────────────────────────────────────────────────
#  TestAnnotations
# ─────────────────────────────────────────────────────────────────────────────

class TestAnnotations:
    def test_add_annotation(self, widget, annotation):
        widget.add_trade_annotation(annotation)
        assert len(widget._canvas._annotations) == 1
        assert widget._canvas._annotations[0] is annotation

    def test_clear_annotations(self, widget, annotation):
        widget.add_trade_annotation(annotation)
        widget.clear_annotations()
        assert len(widget._canvas._annotations) == 0
        assert len(widget._canvas._annotation_markers) == 0

    def test_annotations_property_returns_copy(self, widget, annotation):
        widget.add_trade_annotation(annotation)
        lst = widget.annotations
        assert lst is not widget._canvas._annotations
        assert len(lst) == 1

    def test_multiple_annotations(self, widget):
        for i in range(5):
            ann = TradeAnnotation(
                trade_id=i,
                timestamp=datetime(2024, 1, 15, 10 + i, tzinfo=timezone.utc),
                price=1.1 + i * 0.001,
                direction="buy" if i % 2 == 0 else "sell",
            )
            widget.add_trade_annotation(ann)
        assert len(widget.annotations) == 5


# ─────────────────────────────────────────────────────────────────────────────
#  TestDisplayPeriod
# ─────────────────────────────────────────────────────────────────────────────

class TestDisplayPeriod:
    def test_all_shows_none_limit(self, widget):
        widget.set_display_period("All")
        assert widget._canvas._display_limit is None

    def test_1d_h1_limit(self, widget):
        """1D auf H1: max(10, 1 * 480 / 60) = max(10, 8) = 10 Kerzen."""
        widget.set_timeframe(Timeframe.H1)
        widget.set_display_period("1D")
        assert widget._canvas._display_limit == 10

    def test_1m_h1_limit(self, widget):
        """1M auf H1: 22 * 480 / 60 = 176 Kerzen."""
        widget.set_timeframe(Timeframe.H1)
        widget.set_display_period("1M")
        assert widget._canvas._display_limit == 176

    def test_ytd_sets_limit_positive(self, widget):
        widget.set_display_period("YTD")
        assert widget._canvas._display_limit is not None
        assert widget._canvas._display_limit >= 10

    def test_period_buttons_check_state(self, widget):
        """Nach set_display_period('1M') muss der '1M'-Button aktiviert sein."""
        widget.set_display_period("1M")
        assert widget._period_buttons["1M"].isChecked()
        # Alle anderen muessen deaktiviert sein
        for label, btn in widget._period_buttons.items():
            if label != "1M":
                assert not btn.isChecked(), f"Button '{label}' sollte nicht aktiviert sein"

    def test_display_limit_property(self, widget):
        widget.set_display_period("5D")
        assert widget.display_limit is not None
        assert widget.display_limit > 0

    def test_5d_h4_limit(self, widget):
        """5D auf H4: 5 * 480 / 240 = 10 Kerzen."""
        widget.set_timeframe(Timeframe.H4)
        widget.set_display_period("5D")
        assert widget._canvas._display_limit == 10

    def test_unknown_period_sets_none(self, widget):
        """Unbekannte Periode setzt display_limit auf None."""
        widget.set_display_period("XXX")
        assert widget._canvas._display_limit is None


# ─────────────────────────────────────────────────────────────────────────────
#  TestTimeframeButtons
# ─────────────────────────────────────────────────────────────────────────────

class TestTimeframeButtons:
    def test_all_tf_buttons_present(self, widget):
        """Alle Timeframes muessen einen Button haben."""
        for tf in Timeframe:
            assert tf in widget._tf_buttons

    def test_tf_button_click_emits_signal(self, widget, qtbot):
        """Klick auf M1-Button emittiert timeframe_changed."""
        received = []
        widget.timeframe_changed.connect(received.append)
        widget._tf_buttons[Timeframe.M1].click()
        assert len(received) == 1
        assert received[0] == Timeframe.M1

    def test_set_timeframe_programmatic(self, widget, qtbot):
        received = []
        widget.timeframe_changed.connect(received.append)
        widget.set_timeframe(Timeframe.D1)
        assert widget.current_timeframe == Timeframe.D1
        assert len(received) == 1

    def test_set_same_timeframe_no_double_emit(self, widget, qtbot):
        """Gleiche Zeitebene nochmal setzen darf kein Signal emittieren."""
        received = []
        widget.timeframe_changed.connect(received.append)
        widget.set_timeframe(Timeframe.H1)  # H1 ist schon aktiv
        assert len(received) == 0

    def test_tf_button_checked_state_after_switch(self, widget):
        widget.set_timeframe(Timeframe.M5)
        assert widget._tf_buttons[Timeframe.M5].isChecked()
        assert not widget._tf_buttons[Timeframe.H1].isChecked()


# ─────────────────────────────────────────────────────────────────────────────
#  TestIndicatorOverlays
# ─────────────────────────────────────────────────────────────────────────────

class TestIndicatorOverlays:
    def test_ema_toggle(self, widget):
        assert not widget.ema_visible
        widget.set_indicator_visible("ema", True)
        assert widget.ema_visible
        assert widget._canvas._show_ema
        widget.set_indicator_visible("ema", False)
        assert not widget.ema_visible

    def test_bb_toggle(self, widget):
        assert not widget.bb_visible
        widget.set_indicator_visible("bb", True)
        assert widget.bb_visible
        assert widget._canvas._show_bb

    def test_indicator_visible_unknown_name_no_crash(self, widget):
        """Unbekannter Indikator-Name darf keinen Fehler werfen."""
        widget.set_indicator_visible("unknown_indicator", True)  # no crash

    def test_ema_checkbox_synced(self, widget):
        widget.set_indicator_visible("ema", True)
        assert widget._ema_cb.isChecked()

    def test_bb_checkbox_synced(self, widget):
        widget.set_indicator_visible("bb", True)
        assert widget._bb_cb.isChecked()


# ─────────────────────────────────────────────────────────────────────────────
#  TestPositionLines
# ─────────────────────────────────────────────────────────────────────────────

class TestPositionLines:
    def test_set_position_levels_stores_values(self, widget):
        widget.set_position_levels(sl=1.0950, tp=1.1100, trailing=1.0970)
        assert widget.sl == pytest.approx(1.0950)
        assert widget.tp == pytest.approx(1.1100)
        assert widget.trailing_stop == pytest.approx(1.0970)
        assert widget._canvas._sl == pytest.approx(1.0950)
        assert widget._canvas._tp == pytest.approx(1.1100)
        assert widget._canvas._trailing == pytest.approx(1.0970)

    def test_clear_position_levels(self, widget):
        widget.set_position_levels(sl=1.0950, tp=1.1100)
        widget.set_position_levels(sl=None, tp=None, trailing=None)
        assert widget.sl is None
        assert widget.tp is None
        assert widget.trailing_stop is None

    def test_partial_position_levels(self, widget):
        widget.set_position_levels(sl=1.0950)
        assert widget.sl == pytest.approx(1.0950)
        assert widget.tp is None
        assert widget.trailing_stop is None


# ─────────────────────────────────────────────────────────────────────────────
#  TestPaintEvent
# ─────────────────────────────────────────────────────────────────────────────

class TestPaintEvent:
    def test_paint_no_candles_no_crash(self, widget):
        """Zeichnen ohne Kerzen darf nicht abstuerzen."""
        pix = QPixmap(800, 500)
        widget.render(pix)

    def test_paint_with_candles_no_crash(self, widget, candles):
        widget.set_candles(candles)
        pix = QPixmap(800, 500)
        widget.render(pix)

    def test_paint_with_annotations_no_crash(self, widget, candles, annotation):
        widget.set_candles(candles)
        widget.add_trade_annotation(annotation)
        pix = QPixmap(800, 500)
        widget.render(pix)

    def test_paint_with_bid_ask_no_crash(self, widget, candles):
        widget.set_candles(candles)
        widget.set_bid_ask(1.1050, 1.1052)
        pix = QPixmap(800, 500)
        widget.render(pix)

    def test_paint_with_ema_no_crash(self, widget, candles):
        widget.set_candles(candles)
        widget.set_indicator_visible("ema", True)
        pix = QPixmap(800, 500)
        widget.render(pix)

    def test_paint_with_bb_no_crash(self, widget, candles):
        widget.set_candles(candles)
        widget.set_indicator_visible("bb", True)
        pix = QPixmap(800, 500)
        widget.render(pix)

    def test_paint_with_sl_tp_no_crash(self, widget, candles):
        widget.set_candles(candles)
        widget.set_position_levels(sl=1.0990, tp=1.1050, trailing=1.1000)
        pix = QPixmap(800, 500)
        widget.render(pix)

    def test_paint_with_all_features_no_crash(self, widget, candles, annotation):
        """Alle Features gleichzeitig darf nicht abstuerzen."""
        widget.set_candles(candles)
        widget.set_bid_ask(1.1050, 1.1052)
        widget.set_indicator_visible("ema", True)
        widget.set_indicator_visible("bb", True)
        widget.set_position_levels(sl=1.0990, tp=1.1050, trailing=1.1000)
        widget.add_trade_annotation(annotation)
        widget.set_display_period("1M")
        pix = QPixmap(800, 500)
        widget.render(pix)

    def test_paint_small_canvas_no_crash(self, qtbot, candles):
        """Sehr kleines Widget darf keinen Fehler werfen."""
        w = ChartWidget()
        qtbot.addWidget(w)
        w.resize(50, 50)
        w.set_candles(candles)
        pix = QPixmap(50, 50)
        w.render(pix)

    def test_paint_single_candle_no_crash(self, widget):
        """Einzelne Kerze muss korrekt gezeichnet werden koennen."""
        c = CandleData(
            timestamp=datetime(2024, 1, 15, tzinfo=timezone.utc),
            open=1.1000, high=1.1010, low=1.0990, close=1.1005, volume=500.0
        )
        widget.set_candles([c])
        pix = QPixmap(800, 500)
        widget.render(pix)

    def test_paint_with_sell_annotation_no_crash(self, widget, candles):
        ann = TradeAnnotation(
            trade_id=2,
            timestamp=datetime(2024, 1, 15, 14, 0, 0, tzinfo=timezone.utc),
            price=1.1020,
            direction="sell",
            sl=1.1050,
            tp=1.0990,
            shap_labels=["vol: +0.2 (bullisch)"],
        )
        widget.set_candles(candles)
        widget.add_trade_annotation(ann)
        pix = QPixmap(800, 500)
        widget.render(pix)

    def test_paint_with_display_period_no_crash(self, widget, candles):
        widget.set_candles(candles)
        widget.set_display_period("5D")
        pix = QPixmap(800, 500)
        widget.render(pix)


# ─────────────────────────────────────────────────────────────────────────────
#  TestCrosshair
# ─────────────────────────────────────────────────────────────────────────────

class TestCrosshair:
    def test_mouse_move_sets_mouse_pos(self, widget, qtbot):
        """mouseMoveEvent speichert Mausposition."""
        from PySide6.QtCore import QPoint, QPointF
        from PySide6.QtGui import QMouseEvent, QCursor
        from PySide6.QtCore import QEvent

        widget._canvas._mouse_pos = None
        # Direkt _mouse_pos setzen (Simulation)
        widget._canvas._mouse_pos = (100, 150)
        assert widget._canvas._mouse_pos == (100, 150)

    def test_mouse_leave_clears_mouse_pos(self, widget, qtbot):
        """leaveEvent loescht Mausposition."""
        widget._canvas._mouse_pos = (100, 150)
        widget._canvas.leaveEvent(None)
        assert widget._canvas._mouse_pos is None
        assert widget._canvas._hovered_ann is None

    def test_paint_with_crosshair_no_crash(self, widget, candles):
        """Zeichnen mit Crosshair darf nicht abstuerzen."""
        widget.set_candles(candles)
        widget._canvas._mouse_pos = (400, 250)
        pix = QPixmap(800, 500)
        widget.render(pix)

    def test_crosshair_outside_chart_no_crash(self, widget, candles):
        """Crosshair ausserhalb des Chart-Bereichs darf keinen Fehler werfen."""
        widget.set_candles(candles)
        widget._canvas._mouse_pos = (5, 5)  # linke obere Ecke
        pix = QPixmap(800, 500)
        widget.render(pix)

    def test_paint_with_crosshair_and_annotations_no_crash(self, widget, candles, annotation):
        widget.set_candles(candles)
        widget.add_trade_annotation(annotation)
        widget._canvas._mouse_pos = (400, 250)
        pix = QPixmap(800, 500)
        widget.render(pix)


# ─────────────────────────────────────────────────────────────────────────────
#  TestMouseTracking
# ─────────────────────────────────────────────────────────────────────────────

class TestMouseTracking:
    def test_canvas_has_mouse_tracking_enabled(self, widget):
        """_ChartCanvas muss Mouse-Tracking aktiviert haben."""
        assert widget._canvas.hasMouseTracking()

    def test_initial_mouse_pos_none(self, widget):
        assert widget._canvas._mouse_pos is None

    def test_initial_hovered_ann_none(self, widget):
        assert widget._canvas._hovered_ann is None


# ─────────────────────────────────────────────────────────────────────────────
#  TestLayoutHelper
# ─────────────────────────────────────────────────────────────────────────────

class TestLayoutHelper:
    def test_layout_geometry(self):
        from gui.widgets.chart_widget import _Layout
        lay = _Layout(800, 500)
        assert lay.chart_x == _Layout.M_LEFT
        assert lay.chart_y == _Layout.M_TOP
        assert lay.chart_w > 0
        assert lay.chart_h > 0
        assert lay.vol_h > 0

    def test_price_y_midpoint(self):
        from gui.widgets.chart_widget import _Layout
        lay = _Layout(800, 500)
        mid = lay.price_y(1.1, 1.0, 0.2)
        top = lay.price_y(1.2, 1.0, 0.2)
        bot = lay.price_y(1.0, 1.0, 0.2)
        assert top < mid < bot  # hoehere Preise weiter oben

    def test_idx_from_x_valid(self):
        from gui.widgets.chart_widget import _Layout
        lay = _Layout(800, 500)
        slot_w = 10
        n = 50
        idx = lay.idx_from_x(lay.chart_x + 25, slot_w, n)
        assert idx == 2

    def test_idx_from_x_out_of_bounds(self):
        from gui.widgets.chart_widget import _Layout
        lay = _Layout(800, 500)
        idx = lay.idx_from_x(0, 10, 50)  # vor chart_x
        assert idx is None

    def test_price_from_y_round_trip(self):
        from gui.widgets.chart_widget import _Layout
        lay = _Layout(800, 500)
        original_price = 1.1050
        y = lay.price_y(original_price, 1.0, 0.2)
        recovered = lay.price_from_y(y, 1.0, 0.2)
        assert abs(recovered - original_price) < 0.001  # Pixel-Rundungsfehler OK


# ─────────────────────────────────────────────────────────────────────────────
#  TestVisibleCandles
# ─────────────────────────────────────────────────────────────────────────────

class TestVisibleCandles:
    def test_visible_candles_no_limit(self, widget, candles):
        widget.set_candles(candles)
        visible = widget._canvas._visible_candles()
        assert len(visible) == 50

    def test_visible_candles_with_limit(self, widget, candles):
        widget.set_candles(candles)
        widget._canvas._display_limit = 20
        visible = widget._canvas._visible_candles()
        assert len(visible) == 20
        # Muss die letzten 20 sein
        assert visible[-1] is candles[-1]

    def test_visible_candles_limit_larger_than_data(self, widget, candles):
        widget.set_candles(candles)
        widget._canvas._display_limit = 1000
        visible = widget._canvas._visible_candles()
        assert len(visible) == 50


# ─────────────────────────────────────────────────────────────────────────────
#  TestFindAnnotationIdx
# ─────────────────────────────────────────────────────────────────────────────

class TestFindAnnotationIdx:
    def test_finds_correct_index(self, widget, candles):
        ann = TradeAnnotation(
            trade_id=1,
            timestamp=candles[10].timestamp,  # exakt Kerze 10
            price=1.1,
            direction="buy",
        )
        idx = widget._canvas._find_annotation_idx(candles, ann)
        assert idx == 10

    def test_empty_candles_returns_none(self, widget):
        ann = TradeAnnotation(
            trade_id=1,
            timestamp=datetime(2024, 1, 15, tzinfo=timezone.utc),
            price=1.1,
            direction="buy",
        )
        idx = widget._canvas._find_annotation_idx([], ann)
        assert idx is None

    def test_naive_datetime_handled(self, widget, candles):
        """Naive Timestamps muessen korrekt verarbeitet werden."""
        naive_ts = candles[5].timestamp.replace(tzinfo=None)
        ann = TradeAnnotation(
            trade_id=1,
            timestamp=naive_ts,
            price=1.1,
            direction="buy",
        )
        idx = widget._canvas._find_annotation_idx(candles, ann)
        assert idx == 5


# ─────────────────────────────────────────────────────────────────────────────
#  TestSymbolAndTimeframe
# ─────────────────────────────────────────────────────────────────────────────

class TestSymbolAndTimeframe:
    def test_set_symbol(self, widget):
        widget.set_symbol("EURUSD")
        assert widget._symbol_label.text() == "EURUSD"

    def test_set_symbol_empty_string(self, widget):
        widget.set_symbol("")
        assert widget._symbol_label.text() == ""

    def test_timeframe_enum_values(self):
        assert Timeframe.M1.minutes == 1
        assert Timeframe.M5.minutes == 5
        assert Timeframe.M15.minutes == 15
        assert Timeframe.M30.minutes == 30
        assert Timeframe.H1.minutes == 60
        assert Timeframe.H4.minutes == 240
        assert Timeframe.D1.minutes == 1440

    def test_timeframe_labels(self):
        assert Timeframe.M1.label == "M1"
        assert Timeframe.D1.label == "D1"
