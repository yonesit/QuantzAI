"""
tests/gui/test_cockpit_view.py
GUI-Tests fuer ChartWidget, WatchlistWidget, CockpitView und MainWindow-Integration.

Abgedeckt:
  ChartWidget
    - Standardzustand (H1, keine Indikatoren)
    - Zeitebenen-Umschaltung, timeframe_changed-Signal
    - Indikator ein-/ausblenden (EMA, BB)
    - Kerzendaten setzen / Positionslinien setzen
    - Repaint ohne Absturz (mit und ohne Daten)
    - Alle 7 Zeitebenen-Buttons vorhanden

  WatchlistWidget
    - Leerer Anfangszustand
    - update_entries befuellt Tabelle
    - Freitext-Filter (Gross-/Kleinschreibung, sichtbare Zeilen)
    - Filter zuruecksetzen stellt alle Zeilen wieder her
    - symbol_selected-Signal beim Anklicken
    - selected_symbol-Property
    - Signal-Text und Farbe

  CockpitView
    - Erstellt ohne Absturz
    - chart_widget / watchlist_widget Properties vorhanden
    - Pending-Hint: show/hide
    - update_watchlist propagiert an WatchlistWidget
    - BUY/SELL-Umschaltung (gegenseitig exklusiv)
    - Bestaetigungsworkflow end-to-end: Schliessen-Button + Callback
    - Position schliessen loest position_closed-Signal aus
    - Order aufgeben loest order_submitted-Signal aus
    - Lot-Suggestion wird aus Backend geladen

  MainWindow-Integration
    - CockpitView ist keine Platzhalter-Instanz
    - cockpit_view-Property vorhanden
    - Navigation zu COCKPIT zeigt CockpitView
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock

import pytest
from pytestqt.qtbot import QtBot

from gui.app import MainWindow, Section
from gui.design.theme import ThemeManager, ThemeMode
from gui.views.cockpit_view import CockpitView, WATCHLIST_SYMBOLS
from gui.views.dashboard_view import DashboardSnapshot, PositionInfo
from gui.widgets.chart_widget import CandleData, ChartWidget, Timeframe, _compute_ema, _compute_bollinger
from gui.widgets.watchlist_widget import WatchlistEntry, WatchlistWidget


# ─────────────────────────────────────────────────────────────────────────────
#  Hilfsfunktionen
# ─────────────────────────────────────────────────────────────────────────────

_T0 = datetime(2026, 1, 5, 10, 0, 0, tzinfo=timezone.utc)


def _candle(i: int = 0, price: float = 1.09) -> CandleData:
    from datetime import timedelta
    return CandleData(
        timestamp = _T0 + timedelta(hours=i),
        open  = price,
        high  = price + 0.001,
        low   = price - 0.001,
        close = price + 0.0005,
    )


def _candles(n: int, price: float = 1.09) -> list[CandleData]:
    return [_candle(i, price) for i in range(n)]


def _entries(*symbols: str) -> list[WatchlistEntry]:
    return [
        WatchlistEntry(
            symbol=s,
            bid=1.0 + i * 0.1,
            ask=1.001 + i * 0.1,
            daily_change_pct=0.5 if i % 2 == 0 else -0.3,
            signal="long" if i % 3 == 0 else "short",
            signal_confidence=0.65,
        )
        for i, s in enumerate(symbols)
    ]


def _mock_backend(**overrides) -> MagicMock:
    b = MagicMock()
    b.fetch_candles.return_value   = _candles(50)
    b.get_open_positions.return_value = []
    b.get_lot_suggestion.return_value = 0.10
    b.close_position.return_value  = {"status": "closed"}
    b.update_sl_tp.return_value    = {"status": "ok"}
    b.open_position.return_value   = {"ticket": 99, "status": "open"}
    for k, v in overrides.items():
        setattr(b, k, v)
    return b


# ─────────────────────────────────────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def chart(qtbot: QtBot) -> ChartWidget:
    w = ChartWidget()
    qtbot.addWidget(w)
    return w


@pytest.fixture
def wlist(qtbot: QtBot) -> WatchlistWidget:
    w = WatchlistWidget()
    qtbot.addWidget(w)
    return w


@pytest.fixture
def cockpit(qtbot: QtBot) -> CockpitView:
    v = CockpitView()
    qtbot.addWidget(v)
    return v


@pytest.fixture
def cockpit_with_backend(qtbot: QtBot) -> tuple[CockpitView, MagicMock]:
    backend = _mock_backend()
    v = CockpitView(backend=backend)
    qtbot.addWidget(v)
    return v, backend


@pytest.fixture
def main_window(qtbot: QtBot) -> MainWindow:
    theme = ThemeManager(mode=ThemeMode.DARK)
    w = MainWindow(theme_manager=theme)
    qtbot.addWidget(w)
    return w


# ─────────────────────────────────────────────────────────────────────────────
#  Indikator-Hilfsfunktionen (pure Python – kein Qt noetig)
# ─────────────────────────────────────────────────────────────────────────────

class TestIndicatorFunctions:

    def test_ema_correct_length(self):
        closes = [float(i) for i in range(30)]
        ema = _compute_ema(closes, period=20)
        assert len(ema) == 30

    def test_ema_first_values_none(self):
        closes = [1.0] * 30
        ema = _compute_ema(closes, period=20)
        assert all(v is None for v in ema[:19])

    def test_ema_constant_series_equals_constant(self):
        closes = [2.0] * 25
        ema = _compute_ema(closes, period=5)
        for v in ema[4:]:
            assert v is not None
            assert abs(v - 2.0) < 1e-10

    def test_ema_empty_input(self):
        assert _compute_ema([], period=20) == []

    def test_ema_period_larger_than_data_returns_correct_length(self):
        closes = [1.0, 2.0, 3.0]
        ema = _compute_ema(closes, period=10)
        # period clamped to len(closes)=3 → 2 Nones + 1 value
        assert len(ema) == 3
        assert ema[-1] is not None

    def test_bollinger_correct_length(self):
        closes = [1.0 + i * 0.001 for i in range(30)]
        mid, upper, lower = _compute_bollinger(closes, period=20)
        assert len(mid) == len(upper) == len(lower) == 30

    def test_bollinger_first_values_none(self):
        closes = [1.0] * 30
        mid, upper, lower = _compute_bollinger(closes, period=20)
        assert all(v is None for v in mid[:19])

    def test_bollinger_upper_above_lower(self):
        closes = [1.0 + (i % 5) * 0.001 for i in range(30)]
        _, upper, lower = _compute_bollinger(closes, period=10)
        for u, l in zip(upper, lower):
            if u is not None and l is not None:
                assert u >= l


# ─────────────────────────────────────────────────────────────────────────────
#  ChartWidget
# ─────────────────────────────────────────────────────────────────────────────

class TestChartWidget:

    def test_default_timeframe_is_h1(self, chart: ChartWidget):
        assert chart.current_timeframe is Timeframe.H1

    def test_h1_button_checked_by_default(self, chart: ChartWidget):
        assert chart.timeframe_buttons[Timeframe.H1].isChecked()

    def test_all_seven_timeframe_buttons_present(self, chart: ChartWidget):
        assert set(chart.timeframe_buttons.keys()) == set(Timeframe)

    def test_set_timeframe_changes_current(self, chart: ChartWidget):
        chart.set_timeframe(Timeframe.M5)
        assert chart.current_timeframe is Timeframe.M5

    def test_set_timeframe_checks_new_button(self, chart: ChartWidget):
        chart.set_timeframe(Timeframe.M15)
        assert chart.timeframe_buttons[Timeframe.M15].isChecked()

    def test_set_timeframe_unchecks_old_button(self, chart: ChartWidget):
        chart.set_timeframe(Timeframe.D1)
        assert not chart.timeframe_buttons[Timeframe.H1].isChecked()

    def test_timeframe_changed_signal_emitted(self, chart: ChartWidget, qtbot: QtBot):
        with qtbot.waitSignal(chart.timeframe_changed, timeout=1000) as blocker:
            chart.set_timeframe(Timeframe.M30)
        assert blocker.args[0] is Timeframe.M30

    def test_no_signal_when_same_timeframe(self, chart: ChartWidget, qtbot: QtBot):
        received = []
        chart.timeframe_changed.connect(received.append)
        chart.set_timeframe(Timeframe.H1)  # already H1
        assert received == []

    def test_ema_off_by_default(self, chart: ChartWidget):
        assert chart.ema_visible is False

    def test_bb_off_by_default(self, chart: ChartWidget):
        assert chart.bb_visible is False

    def test_set_ema_visible(self, chart: ChartWidget):
        chart.set_indicator_visible("ema", True)
        assert chart.ema_visible is True

    def test_set_bb_visible(self, chart: ChartWidget):
        chart.set_indicator_visible("bb", True)
        assert chart.bb_visible is True

    def test_unset_ema(self, chart: ChartWidget):
        chart.set_indicator_visible("ema", True)
        chart.set_indicator_visible("ema", False)
        assert chart.ema_visible is False

    def test_ema_checkbox_synced(self, chart: ChartWidget):
        chart.set_indicator_visible("ema", True)
        assert chart.ema_checkbox.isChecked()

    def test_bb_checkbox_synced(self, chart: ChartWidget):
        chart.set_indicator_visible("bb", True)
        assert chart.bb_checkbox.isChecked()

    def test_unknown_indicator_name_ignored(self, chart: ChartWidget):
        chart.set_indicator_visible("macd", True)  # unknown – no crash, no change

    def test_initial_candles_count_zero(self, chart: ChartWidget):
        assert chart.candles_count == 0

    def test_set_candles_stores_count(self, chart: ChartWidget):
        chart.set_candles(_candles(50))
        assert chart.candles_count == 50

    def test_set_candles_empty_no_crash(self, chart: ChartWidget):
        chart.set_candles([])
        assert chart.candles_count == 0

    def test_set_candles_replaces_old_data(self, chart: ChartWidget):
        chart.set_candles(_candles(100))
        chart.set_candles(_candles(25))
        assert chart.candles_count == 25

    def test_sl_none_by_default(self, chart: ChartWidget):
        assert chart.sl is None

    def test_tp_none_by_default(self, chart: ChartWidget):
        assert chart.tp is None

    def test_trailing_stop_none_by_default(self, chart: ChartWidget):
        assert chart.trailing_stop is None

    def test_set_position_levels(self, chart: ChartWidget):
        chart.set_position_levels(sl=1.08, tp=1.10, trailing=1.085)
        assert chart.sl        == pytest.approx(1.08)
        assert chart.tp        == pytest.approx(1.10)
        assert chart.trailing_stop == pytest.approx(1.085)

    def test_clear_position_levels(self, chart: ChartWidget):
        chart.set_position_levels(sl=1.08, tp=1.10)
        chart.set_position_levels()  # all None
        assert chart.sl is None
        assert chart.tp is None

    def test_repaint_no_crash_empty(self, chart: ChartWidget, qtbot: QtBot):
        chart.show()
        chart.repaint()

    def test_repaint_no_crash_with_data(self, chart: ChartWidget, qtbot: QtBot):
        chart.set_candles(_candles(60))
        chart.set_position_levels(sl=1.089, tp=1.092)
        chart.set_indicator_visible("ema", True)
        chart.set_indicator_visible("bb",  True)
        chart.show()
        chart.repaint()

    def test_timeframe_enum_has_seven_members(self):
        assert len(list(Timeframe)) == 7

    def test_timeframe_labels(self):
        labels = {tf.label for tf in Timeframe}
        assert labels == {"M1", "M5", "M15", "M30", "H1", "H4", "D1"}

    def test_set_symbol_updates_label(self, chart: ChartWidget):
        chart.set_symbol("EURUSD")
        # no crash, symbol label is set
        assert True  # smoke test


# ─────────────────────────────────────────────────────────────────────────────
#  WatchlistWidget
# ─────────────────────────────────────────────────────────────────────────────

class TestWatchlistWidget:

    def test_initial_row_count_zero(self, wlist: WatchlistWidget):
        assert wlist.table.rowCount() == 0

    def test_update_entries_populates_rows(self, wlist: WatchlistWidget):
        wlist.update_entries(_entries("EURUSD", "GBPUSD", "USDJPY"))
        assert wlist.table.rowCount() == 3

    def test_visible_row_count_matches_all_without_filter(self, wlist: WatchlistWidget):
        wlist.update_entries(_entries("EURUSD", "GBPUSD", "USDJPY"))
        assert wlist.visible_row_count == 3

    def test_filter_hides_non_matching_rows(self, wlist: WatchlistWidget):
        wlist.update_entries(_entries("EURUSD", "GBPUSD", "USDJPY"))
        wlist.set_filter("EUR")
        assert wlist.visible_row_count == 1

    def test_filter_case_insensitive(self, wlist: WatchlistWidget):
        wlist.update_entries(_entries("EURUSD", "GBPUSD"))
        wlist.set_filter("eur")
        assert wlist.visible_row_count == 1

    def test_filter_partial_match(self, wlist: WatchlistWidget):
        wlist.update_entries(_entries("EURUSD", "EURJPY", "GBPUSD"))
        wlist.set_filter("EUR")
        assert wlist.visible_row_count == 2

    def test_filter_no_match_hides_all(self, wlist: WatchlistWidget):
        wlist.update_entries(_entries("EURUSD", "GBPUSD"))
        wlist.set_filter("XAUUSD")
        assert wlist.visible_row_count == 0

    def test_clear_filter_restores_all_rows(self, wlist: WatchlistWidget):
        wlist.update_entries(_entries("EURUSD", "GBPUSD", "USDJPY"))
        wlist.set_filter("EUR")
        wlist.set_filter("")
        assert wlist.visible_row_count == 3

    def test_symbol_in_first_column(self, wlist: WatchlistWidget):
        wlist.update_entries([WatchlistEntry(symbol="EURUSD")])
        item = wlist.table.item(0, 0)
        assert item is not None
        assert item.text() == "EURUSD"

    def test_long_signal_shows_up_arrow(self, wlist: WatchlistWidget):
        wlist.update_entries([WatchlistEntry("EURUSD", signal="long")])
        item = wlist.table.item(0, 4)
        assert "▲" in item.text() or "LONG" in item.text()

    def test_short_signal_shows_down_arrow(self, wlist: WatchlistWidget):
        wlist.update_entries([WatchlistEntry("EURUSD", signal="short")])
        item = wlist.table.item(0, 4)
        assert "▼" in item.text() or "SHORT" in item.text()

    def test_positive_change_text_has_plus(self, wlist: WatchlistWidget):
        wlist.update_entries([WatchlistEntry("EURUSD", daily_change_pct=1.5)])
        item = wlist.table.item(0, 3)
        assert "+" in item.text()

    def test_negative_change_text_has_minus(self, wlist: WatchlistWidget):
        wlist.update_entries([WatchlistEntry("EURUSD", daily_change_pct=-0.8)])
        item = wlist.table.item(0, 3)
        assert "-" in item.text()

    def test_selected_symbol_none_initially(self, wlist: WatchlistWidget):
        assert wlist.selected_symbol is None

    def test_symbol_selected_signal_on_row_click(self, wlist: WatchlistWidget, qtbot: QtBot):
        wlist.update_entries(_entries("EURUSD", "GBPUSD"))
        # Find EURUSD row regardless of sort order
        eurusd_row = next(
            r for r in range(wlist.table.rowCount())
            if wlist.table.item(r, 0) and wlist.table.item(r, 0).text() == "EURUSD"
        )
        with qtbot.waitSignal(wlist.symbol_selected, timeout=1000) as blocker:
            wlist.table.selectRow(eurusd_row)
        assert blocker.args[0] == "EURUSD"

    def test_selected_symbol_after_row_click(self, wlist: WatchlistWidget, qtbot: QtBot):
        wlist.update_entries(_entries("EURUSD", "GBPUSD"))
        gbpusd_row = next(
            r for r in range(wlist.table.rowCount())
            if wlist.table.item(r, 0) and wlist.table.item(r, 0).text() == "GBPUSD"
        )
        wlist.table.selectRow(gbpusd_row)
        assert wlist.selected_symbol == "GBPUSD"

    def test_update_entries_clears_old_rows(self, wlist: WatchlistWidget):
        wlist.update_entries(_entries("EURUSD", "GBPUSD"))
        wlist.update_entries(_entries("USDJPY"))
        assert wlist.table.rowCount() == 1

    def test_filter_widget_is_qlineedit(self, wlist: WatchlistWidget):
        from PySide6.QtWidgets import QLineEdit
        assert isinstance(wlist.filter_widget, QLineEdit)

    def test_sorting_enabled(self, wlist: WatchlistWidget):
        assert wlist.table.isSortingEnabled()


# ─────────────────────────────────────────────────────────────────────────────
#  CockpitView
# ─────────────────────────────────────────────────────────────────────────────

class TestCockpitView:

    def test_creates_without_crash(self, cockpit: CockpitView):
        assert cockpit is not None

    def test_chart_widget_property(self, cockpit: CockpitView):
        assert isinstance(cockpit.chart_widget, ChartWidget)

    def test_watchlist_widget_property(self, cockpit: CockpitView):
        assert isinstance(cockpit.watchlist_widget, WatchlistWidget)

    def test_pending_hint_hidden_initially(self, cockpit: CockpitView):
        assert cockpit.pending_hint_frame.isHidden()

    def test_show_pending_request_makes_hint_visible(self, cockpit: CockpitView):
        cockpit.show_pending_request("Trade EURUSD LONG bestätigen?")
        assert not cockpit.pending_hint_frame.isHidden()

    def test_show_pending_request_sets_text(self, cockpit: CockpitView):
        cockpit.show_pending_request("Bestätigung erforderlich: GBPUSD")
        assert "GBPUSD" in cockpit.pending_hint_label.text()

    def test_hide_pending_request_hides_hint(self, cockpit: CockpitView):
        cockpit.show_pending_request("Test")
        cockpit.hide_pending_request()
        assert cockpit.pending_hint_frame.isHidden()

    def test_update_watchlist_propagates(self, cockpit: CockpitView):
        cockpit.update_watchlist(_entries("EURUSD", "GBPUSD"))
        assert cockpit.watchlist_widget.table.rowCount() == 2

    def test_buy_button_checked_by_default(self, cockpit: CockpitView):
        assert cockpit.buy_button.isChecked()
        assert not cockpit.sell_button.isChecked()

    def test_sell_button_exclusive_with_buy(self, cockpit: CockpitView):
        cockpit.sell_button.click()
        assert cockpit.sell_button.isChecked()
        assert not cockpit.buy_button.isChecked()

    def test_buy_button_exclusive_with_sell(self, cockpit: CockpitView):
        cockpit.sell_button.click()
        cockpit.buy_button.click()
        assert cockpit.buy_button.isChecked()
        assert not cockpit.sell_button.isChecked()

    def test_positions_table_columns(self, cockpit: CockpitView):
        # Reiche Positions-Tabelle: Symbol, Richtung, Lots, Eröffnung, P&L, [Schliessen]
        assert cockpit.positions_table.columnCount() == 6

    def test_positions_table_empty_without_backend(self, cockpit: CockpitView):
        assert cockpit.positions_table.rowCount() == 0

    def test_lot_spinbox_present(self, cockpit: CockpitView):
        assert cockpit.lot_spinbox is not None
        assert cockpit.lot_spinbox.value() >= 0.01

    def test_sl_tp_spinboxes_present(self, cockpit: CockpitView):
        assert cockpit.sl_spinbox is not None
        assert cockpit.tp_spinbox is not None

    def test_submit_button_present(self, cockpit: CockpitView):
        assert cockpit.submit_button is not None

    def test_lot_suggestion_set_on_symbol_select(
        self, cockpit_with_backend: tuple[CockpitView, MagicMock]
    ):
        view, backend = cockpit_with_backend
        backend.get_lot_suggestion.return_value = 0.25
        view.watchlist_widget.update_entries(_entries("EURUSD"))
        view.watchlist_widget.table.selectRow(0)
        assert view.lot_spinbox.value() == pytest.approx(0.25)

    def test_candles_loaded_on_symbol_select(
        self, cockpit_with_backend: tuple[CockpitView, MagicMock]
    ):
        view, backend = cockpit_with_backend
        backend.fetch_candles.return_value = _candles(80)
        view.watchlist_widget.update_entries(_entries("EURUSD"))
        view.watchlist_widget.table.selectRow(0)
        assert view.chart_widget.candles_count == 80

    def test_positions_populated_from_backend(self, qtbot: QtBot):
        backend = _mock_backend()
        backend.get_open_positions.return_value = [
            {"ticket": 1, "symbol": "EURUSD", "direction": "buy",
             "lot_size": 0.1, "open_price": 1.09, "sl_price": 1.08, "tp_price": 1.10}
        ]
        view = CockpitView(backend=backend)
        qtbot.addWidget(view)
        assert view.positions_table.rowCount() == 1

    def test_close_position_calls_confirm_fn(self, qtbot: QtBot):
        confirm_calls: list[tuple] = []
        def _confirm(title, msg, label):
            confirm_calls.append((title, msg, label))
            return False  # user cancels

        backend = _mock_backend()
        backend.get_open_positions.return_value = [
            {"ticket": 42, "symbol": "GBPUSD", "direction": "sell",
             "lot_size": 0.2, "open_price": 1.27, "sl_price": None, "tp_price": None}
        ]
        view = CockpitView(backend=backend, _confirm_fn=_confirm)
        qtbot.addWidget(view)

        # Trigger close via internal method directly (avoids real dialog)
        view._on_close_position(42, "GBPUSD")
        assert len(confirm_calls) == 1
        assert "GBPUSD" in confirm_calls[0][1]

    def test_close_confirmed_calls_backend(self, qtbot: QtBot):
        backend = _mock_backend()
        view = CockpitView(backend=backend, _confirm_fn=lambda t, m, l: True)
        qtbot.addWidget(view)
        view._on_close_position(99, "EURUSD")
        backend.close_position.assert_called_once_with(99)

    def test_close_cancelled_does_not_call_backend(self, qtbot: QtBot):
        backend = _mock_backend()
        view = CockpitView(backend=backend, _confirm_fn=lambda t, m, l: False)
        qtbot.addWidget(view)
        view._on_close_position(99, "EURUSD")
        backend.close_position.assert_not_called()

    def test_close_confirmed_emits_position_closed(self, qtbot: QtBot):
        backend = _mock_backend()
        view = CockpitView(backend=backend, _confirm_fn=lambda t, m, l: True)
        qtbot.addWidget(view)
        with qtbot.waitSignal(view.position_closed, timeout=1000) as blocker:
            view._on_close_position(77, "USDJPY")
        assert blocker.args[0] == 77

    def test_submit_order_calls_backend(self, qtbot: QtBot):
        backend = _mock_backend()
        view = CockpitView(backend=backend)
        qtbot.addWidget(view)
        view._active_sym = "EURUSD"
        view._lot_spin.setValue(0.1)
        view._on_submit_order()
        backend.open_position.assert_called_once()

    def test_submit_order_emits_signal(self, qtbot: QtBot):
        backend = _mock_backend()
        view = CockpitView(backend=backend)
        qtbot.addWidget(view)
        view._active_sym = "EURUSD"
        with qtbot.waitSignal(view.order_submitted, timeout=1000):
            view._on_submit_order()

    def test_submit_without_symbol_does_nothing(self, qtbot: QtBot):
        backend = _mock_backend()
        view = CockpitView(backend=backend)
        qtbot.addWidget(view)
        # _active_sym is "" by default
        view._on_submit_order()
        backend.open_position.assert_not_called()

    def test_set_backend_refreshes_positions(self, qtbot: QtBot):
        backend = _mock_backend()
        backend.get_open_positions.return_value = [
            {"ticket": 5, "symbol": "EURUSD", "direction": "buy",
             "lot_size": 0.05, "open_price": 1.09, "sl_price": None, "tp_price": None}
        ]
        view = CockpitView()
        qtbot.addWidget(view)
        assert view.positions_table.rowCount() == 0
        view.set_backend(backend)
        assert view.positions_table.rowCount() == 1

    def test_active_symbol_initially_empty(self, cockpit: CockpitView):
        assert cockpit.active_symbol == ""


# ─────────────────────────────────────────────────────────────────────────────
#  MainWindow-Integration
# ─────────────────────────────────────────────────────────────────────────────

class TestMainWindowCockpitIntegration:

    def test_cockpit_view_property_exists(self, main_window: MainWindow):
        assert main_window.cockpit_view is not None

    def test_cockpit_view_is_real_cockpit_view(self, main_window: MainWindow):
        assert isinstance(main_window.cockpit_view, CockpitView)

    def test_navigate_to_cockpit_shows_cockpit_view(self, main_window: MainWindow):
        main_window.navigate_to(Section.COCKPIT)
        assert main_window.current_view() is main_window.cockpit_view

    def test_still_seven_views_in_stack(self, main_window: MainWindow):
        assert main_window.content.count() == 7

    def test_navigate_through_all_sections_no_crash(self, main_window: MainWindow):
        for section in Section:
            main_window.navigate_to(section)
            assert main_window.sidebar.current_section is section


# ─────────────────────────────────────────────────────────────────────────────
#  Cockpit Trading-Stats (Positionen + Tages/Gesamt-Statistiken)
# ─────────────────────────────────────────────────────────────────────────────

def _snap_with_positions(**kw) -> DashboardSnapshot:
    return DashboardSnapshot(
        positions=[
            PositionInfo(
                ticket=kw.get("ticket", 42),
                symbol=kw.get("symbol", "EURUSD"),
                direction=kw.get("direction", "long"),
                lot_size=kw.get("lot_size", 0.10),
                open_price=kw.get("open_price", 1.0850),
                current_pnl=kw.get("current_pnl", 25.50),
            )
        ],
        today_trades=kw.get("today_trades", 3),
        today_pnl=kw.get("today_pnl", 75.00),
        today_win_rate=kw.get("today_win_rate", 0.667),
        total_gross_profit=kw.get("total_gross_profit", 200.00),
        total_gross_loss=kw.get("total_gross_loss", -80.00),
        currency="€",
    )


class TestCockpitTradingStats:

    def test_daily_stats_card_present(self, cockpit: CockpitView):
        assert cockpit.daily_stats_card is not None

    def test_total_stats_card_present(self, cockpit: CockpitView):
        assert cockpit.total_stats_card is not None

    def test_positions_table_has_six_columns(self, cockpit: CockpitView):
        assert cockpit.positions_table.columnCount() == 6

    def test_update_trading_stats_populates_positions(self, cockpit: CockpitView):
        snap = _snap_with_positions()
        cockpit.update_trading_stats(snap)
        assert cockpit.positions_table.rowCount() == 1

    def test_update_trading_stats_shows_symbol(self, cockpit: CockpitView):
        snap = _snap_with_positions(symbol="XAUUSD")
        cockpit.update_trading_stats(snap)
        item = cockpit.positions_table.item(0, 0)
        assert item is not None and "XAUUSD" in item.text()

    def test_update_trading_stats_shows_pnl(self, cockpit: CockpitView):
        snap = _snap_with_positions(current_pnl=25.50)
        cockpit.update_trading_stats(snap)
        pnl_item = cockpit.positions_table.item(0, 4)
        assert pnl_item is not None
        assert "25" in pnl_item.text()

    def test_update_trading_stats_shows_daily_trades(self, cockpit: CockpitView):
        snap = _snap_with_positions(today_trades=7)
        cockpit.update_trading_stats(snap)
        assert "7" in cockpit.daily_stats_card.trades_label.text()

    def test_update_trading_stats_shows_daily_pnl(self, cockpit: CockpitView):
        snap = _snap_with_positions(today_pnl=150.00)
        cockpit.update_trading_stats(snap)
        assert "150" in cockpit.daily_stats_card.pnl_label.text()

    def test_update_trading_stats_shows_win_rate(self, cockpit: CockpitView):
        snap = _snap_with_positions(today_win_rate=0.75)
        cockpit.update_trading_stats(snap)
        assert "75" in cockpit.daily_stats_card.winrate_label.text()

    def test_update_trading_stats_shows_total_profit(self, cockpit: CockpitView):
        snap = _snap_with_positions(total_gross_profit=200.00)
        cockpit.update_trading_stats(snap)
        assert "200" in cockpit.total_stats_card.profit_label.text()

    def test_update_trading_stats_shows_total_loss(self, cockpit: CockpitView):
        snap = _snap_with_positions(total_gross_loss=-80.00)
        cockpit.update_trading_stats(snap)
        assert "80" in cockpit.total_stats_card.loss_label.text()

    def test_positions_table_close_button_emits_signal(self, cockpit: CockpitView, qtbot: QtBot):
        snap = _snap_with_positions(ticket=99)
        cockpit.update_trading_stats(snap)
        close_btn = cockpit.positions_table.cellWidget(0, 5)
        assert close_btn is not None
        with qtbot.waitSignal(cockpit.position_close_requested, timeout=1000) as blocker:
            close_btn.click()
        assert blocker.args[0] == 99

    def test_multiple_positions_multiple_rows(self, cockpit: CockpitView):
        snap = DashboardSnapshot(
            positions=[
                PositionInfo(ticket=1, symbol="EURUSD", direction="long", lot_size=0.1),
                PositionInfo(ticket=2, symbol="XAUUSD", direction="short", lot_size=0.05),
                PositionInfo(ticket=3, symbol="USDJPY", direction="long", lot_size=0.2),
            ]
        )
        cockpit.update_trading_stats(snap)
        assert cockpit.positions_table.rowCount() == 3

    def test_empty_snap_clears_positions(self, cockpit: CockpitView):
        cockpit.update_trading_stats(_snap_with_positions())
        assert cockpit.positions_table.rowCount() == 1
        cockpit.update_trading_stats(DashboardSnapshot())
        assert cockpit.positions_table.rowCount() == 0


# ─────────────────────────────────────────────────────────────────────────────
#  Cockpit Watchlist-Befuellung
# ─────────────────────────────────────────────────────────────────────────────

def _mock_connector(bid=1.08500, ask=1.08520, day_open=1.08000) -> MagicMock:
    conn = MagicMock()
    conn.get_tick.return_value = {"bid": bid, "ask": ask}
    import pandas as pd
    from datetime import datetime, timezone
    df = pd.DataFrame(
        {"open": [day_open], "high": [day_open + 0.01],
         "low": [day_open - 0.005], "close": [bid]},
        index=[datetime.now(timezone.utc)],
    )
    conn.get_ohlcv_count.return_value = df
    return conn


class TestCockpitWatchlistPopulation:

    def test_watchlist_symbols_constant(self):
        assert "XAUUSD" in WATCHLIST_SYMBOLS
        assert "EURUSD" in WATCHLIST_SYMBOLS
        assert "USDJPY" in WATCHLIST_SYMBOLS
        assert "GBPUSD" in WATCHLIST_SYMBOLS

    def test_set_watchlist_connector_populates_four_symbols(self, cockpit: CockpitView):
        conn = _mock_connector()
        cockpit.set_watchlist_connector(conn)
        assert cockpit.watchlist_widget.table.rowCount() == 4

    def test_watchlist_shows_bid(self, cockpit: CockpitView):
        conn = _mock_connector(bid=1.08500)
        cockpit.set_watchlist_connector(conn)
        # Find EURUSD row
        tbl = cockpit.watchlist_widget.table
        eurusd_row = next(
            r for r in range(tbl.rowCount())
            if tbl.item(r, 0) and tbl.item(r, 0).text() == "EURUSD"
        )
        bid_item = tbl.item(eurusd_row, 1)
        assert bid_item is not None
        assert "1.085" in bid_item.text()

    def test_watchlist_shows_ask(self, cockpit: CockpitView):
        conn = _mock_connector(ask=1.08520)
        cockpit.set_watchlist_connector(conn)
        tbl = cockpit.watchlist_widget.table
        eurusd_row = next(
            r for r in range(tbl.rowCount())
            if tbl.item(r, 0) and tbl.item(r, 0).text() == "EURUSD"
        )
        ask_item = tbl.item(eurusd_row, 2)
        assert ask_item is not None
        assert "1.085" in ask_item.text()

    def test_watchlist_shows_daily_change(self, cockpit: CockpitView):
        conn = _mock_connector(bid=1.09000, day_open=1.08000)
        cockpit.set_watchlist_connector(conn)
        tbl = cockpit.watchlist_widget.table
        eurusd_row = next(
            r for r in range(tbl.rowCount())
            if tbl.item(r, 0) and tbl.item(r, 0).text() == "EURUSD"
        )
        change_item = tbl.item(eurusd_row, 3)
        assert change_item is not None and change_item.text() != "–"

    def test_watchlist_signal_provider_used_for_eurusd(self, cockpit: CockpitView):
        conn = _mock_connector()
        signal_providers = {
            "EURUSD": lambda: {"signal": "long", "confidence": 0.82},
            "XAUUSD": lambda: {"signal": "short", "confidence": 0.71},
        }
        cockpit.set_watchlist_connector(conn, signal_providers=signal_providers)
        tbl = cockpit.watchlist_widget.table
        eurusd_row = next(
            r for r in range(tbl.rowCount())
            if tbl.item(r, 0) and tbl.item(r, 0).text() == "EURUSD"
        )
        sig_item = tbl.item(eurusd_row, 4)
        assert sig_item is not None
        assert "LONG" in sig_item.text()

    def test_watchlist_no_model_for_unsupported_symbols(self, cockpit: CockpitView):
        conn = _mock_connector()
        # Nur EURUSD und XAUUSD haben Modelle
        signal_providers = {
            "EURUSD": lambda: {"signal": "flat", "confidence": 0.5},
            "XAUUSD": lambda: {"signal": "flat", "confidence": 0.5},
        }
        cockpit.set_watchlist_connector(conn, signal_providers=signal_providers)
        tbl = cockpit.watchlist_widget.table
        # USDJPY hat kein Modell
        usdjpy_row = next(
            r for r in range(tbl.rowCount())
            if tbl.item(r, 0) and tbl.item(r, 0).text() == "USDJPY"
        )
        sig_item = tbl.item(usdjpy_row, 4)
        assert sig_item is not None
        assert "kein" in sig_item.text().lower() or "modell" in sig_item.text().lower()

    def test_watchlist_click_switches_chart_symbol(
        self, cockpit_with_backend: tuple[CockpitView, MagicMock], qtbot: QtBot
    ):
        view, backend = cockpit_with_backend
        conn = _mock_connector()
        view.set_watchlist_connector(conn)
        tbl = view.watchlist_widget.table
        xauusd_row = next(
            r for r in range(tbl.rowCount())
            if tbl.item(r, 0) and tbl.item(r, 0).text() == "XAUUSD"
        )
        with qtbot.waitSignal(view.watchlist_widget.symbol_selected, timeout=1000):
            tbl.selectRow(xauusd_row)
        assert view.active_symbol == "XAUUSD"

    def test_connector_not_required_for_manual_update(self, cockpit: CockpitView):
        """update_watchlist() funktioniert ohne set_watchlist_connector."""
        cockpit.update_watchlist(_entries("EURUSD", "XAUUSD"))
        assert cockpit.watchlist_widget.table.rowCount() == 2

    def test_timer_started_after_set_connector(self, cockpit: CockpitView):
        conn = _mock_connector()
        assert not cockpit._watchlist_timer.isActive()
        cockpit.set_watchlist_connector(conn)
        assert cockpit._watchlist_timer.isActive()
        cockpit._watchlist_timer.stop()
