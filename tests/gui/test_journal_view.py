"""
tests/gui/test_journal_view.py
Tests fuer gui/views/journal_view.py und MainWindow-Integration.

Abgedeckt:
  - _MoodPopup: Save, Skip, Plan-Checkbox
  - _DnaHeatmapCanvas: Datensetting, Zellen-Zugriff
  - _HistoryTab: Laden, lokaler Filter, Replay/Coach-Signale
  - _ReplayTab: load_replay, ChartWidget-Integration
  - _CoachTab: Trade-ID-Verwaltung, Senden, Antwort-Anzeige
  - _ReportTab: Report laden, exportieren
  - JournalView: Mood-Popup (injectable), refresh, Signale, Backend-Delegation
  - MainWindow: JournalView integriert, Navigation, View-Anzahl
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from gui.app import MainWindow, Section
from gui.views.journal_view import (
    JournalView,
    _CoachTab,
    _DnaHeatmapCanvas,
    _DnaTab,
    _HistoryTab,
    _MoodPopup,
    _ReplayTab,
    _ReportTab,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _mock_backend(**overrides):
    b = MagicMock()
    b.get_trades.return_value = [
        {"id": 1, "symbol": "EURUSD", "direction": "buy",  "lot_size": 0.10,
         "entry_time": "2026-01-01T10:00:00", "exit_time": "2026-01-01T11:00:00",
         "pnl": 55.0,  "status": "closed", "setup": "Breakout"},
        {"id": 2, "symbol": "GBPUSD", "direction": "sell", "lot_size": 0.05,
         "entry_time": "2026-01-02T09:00:00", "exit_time": None,
         "pnl": None, "status": "open", "setup": "Trend"},
    ]
    b.generate_report.return_value = "# Report\n\n| Trades | 2 |"
    b.get_dna_profile.return_value = {
        "status": "ready", "n_trades": 600, "min_trades_required": 500,
        "symbols": {"ranked": [
            {"symbol": "EURUSD", "win_rate": 0.65, "total_pnl": 430.0, "n_trades": 120, "confidence": "high"},
        ], "best": [], "worst": []},
        "setups": {"ranked": [
            {"setup": "Breakout", "win_rate": 0.60, "total_pnl": 310.0, "n_trades": 80, "confidence": "medium"},
        ], "best": [], "worst": []},
        "psychological_weaknesses": ["FOMO: win_rate=28% (15 Trades)"],
    }
    b.get_hour_weekday_matrix.return_value = [
        [{"win_rate": 0.6, "n_trades": 10} if hr == 9 else None for hr in range(24)]
        for _ in range(7)
    ]
    b.get_replay_data.return_value = {
        "trade": {"id": 1, "symbol": "EURUSD", "direction": "buy",
                  "entry_price": 1.08500, "exit_price": 1.09000, "pnl": 55.0},
        "candles": [
            {"time": "2026-01-01T09:00:00", "open": 1.08400, "high": 1.08600,
             "low": 1.08300, "close": 1.08500, "volume": 100.0},
            {"time": "2026-01-01T10:00:00", "open": 1.08500, "high": 1.09100,
             "low": 1.08400, "close": 1.09000, "volume": 150.0},
        ],
        "entry_marker": {"time": "2026-01-01T10:00:00", "price": 1.08500, "direction": "buy"},
        "exit_marker":  {"time": "2026-01-01T11:00:00", "price": 1.09000},
        "indicators":   {},
        "news_events":  [],
        "meta": {"symbol": "EURUSD", "timeframe": "H1",
                 "lookback_candles": 100, "candles_found": 2, "no_lookahead": True,
                 "entry_time": "2026-01-01T10:00:00"},
    }
    b.ask_coach.return_value = "Trade #1 zeigt ein solides Risk/Reward-Verhältnis."
    for k, v in overrides.items():
        setattr(b, k, MagicMock(return_value=v) if not callable(v) else v)
    return b


def _mood_yes(title, kind, show_plan):
    return ("calm", "Alles gut", True)

def _mood_no(title, kind, show_plan):
    return None  # Skip


# ─────────────────────────────────────────────────────────────────────────────
#  TestMoodPopup
# ─────────────────────────────────────────────────────────────────────────────

class TestMoodPopup:
    def test_builds_without_plan_checkbox(self, qtbot):
        popup = _MoodPopup("Stimmung", show_plan_cb=False)
        qtbot.addWidget(popup)
        assert popup.plan_checkbox is None

    def test_builds_with_plan_checkbox(self, qtbot):
        popup = _MoodPopup("Stimmung", show_plan_cb=True)
        qtbot.addWidget(popup)
        assert popup.plan_checkbox is not None

    def test_mood_combo_has_6_options(self, qtbot):
        popup = _MoodPopup("Stimmung")
        qtbot.addWidget(popup)
        assert popup.mood_combo.count() == 6

    def test_save_emits_mood_saved(self, qtbot):
        popup = _MoodPopup("Stimmung", show_plan_cb=False)
        qtbot.addWidget(popup)
        signals = []
        popup.mood_saved.connect(lambda m, r, p: signals.append((m, r, p)))
        popup.reason_edit.setText("Test-Grund")
        popup._on_save()
        assert len(signals) == 1
        assert signals[0][1] == "Test-Grund"
        assert isinstance(signals[0][0], str)

    def test_skip_emits_mood_skipped(self, qtbot):
        popup = _MoodPopup("Stimmung")
        qtbot.addWidget(popup)
        signals = []
        popup.mood_skipped.connect(lambda: signals.append(True))
        popup._on_skip()
        assert signals == [True]

    def test_plan_followed_false_by_default(self, qtbot):
        popup = _MoodPopup("Stimmung", show_plan_cb=True)
        qtbot.addWidget(popup)
        signals = []
        popup.mood_saved.connect(lambda m, r, p: signals.append(p))
        popup._on_save()
        assert signals == [False]

    def test_plan_followed_true_when_checked(self, qtbot):
        popup = _MoodPopup("Stimmung", show_plan_cb=True)
        qtbot.addWidget(popup)
        popup.plan_checkbox.setChecked(True)
        signals = []
        popup.mood_saved.connect(lambda m, r, p: signals.append(p))
        popup._on_save()
        assert signals == [True]

    def test_non_modal(self, qtbot):
        popup = _MoodPopup("Stimmung")
        qtbot.addWidget(popup)
        assert not popup.isModal()

    def test_reason_placeholder(self, qtbot):
        popup = _MoodPopup("Stimmung")
        qtbot.addWidget(popup)
        assert popup.reason_edit.placeholderText() != ""

    def test_save_uses_selected_mood(self, qtbot):
        popup = _MoodPopup("Stimmung")
        qtbot.addWidget(popup)
        popup.mood_combo.setCurrentIndex(3)  # fomo
        signals = []
        popup.mood_saved.connect(lambda m, r, p: signals.append(m))
        popup._on_save()
        assert signals[0] == "fomo"


# ─────────────────────────────────────────────────────────────────────────────
#  TestDnaHeatmapCanvas
# ─────────────────────────────────────────────────────────────────────────────

class TestDnaHeatmapCanvas:
    def test_initial_matrix_all_none(self, qtbot):
        c = _DnaHeatmapCanvas()
        qtbot.addWidget(c)
        assert c.get_cell(0, 0) is None
        assert c.get_cell(6, 23) is None

    def test_set_data_stores_matrix(self, qtbot):
        c = _DnaHeatmapCanvas()
        qtbot.addWidget(c)
        matrix = [[None] * 24 for _ in range(7)]
        matrix[0][9] = {"win_rate": 0.7, "n_trades": 20}
        c.set_data(matrix)
        assert c.get_cell(0, 9) == {"win_rate": 0.7, "n_trades": 20}

    def test_get_cell_out_of_bounds_returns_none(self, qtbot):
        c = _DnaHeatmapCanvas()
        qtbot.addWidget(c)
        assert c.get_cell(-1, 0)  is None
        assert c.get_cell(7,  0)  is None
        assert c.get_cell(0, 24)  is None

    def test_paint_no_crash_empty(self, qtbot):
        c = _DnaHeatmapCanvas()
        qtbot.addWidget(c)
        c.show()
        c.resize(600, 200)
        qtbot.waitExposed(c)

    def test_paint_no_crash_with_data(self, qtbot):
        c = _DnaHeatmapCanvas()
        qtbot.addWidget(c)
        matrix = [[{"win_rate": 0.6, "n_trades": 5}] * 24 for _ in range(7)]
        c.set_data(matrix)
        c.show()
        c.resize(600, 200)
        qtbot.waitExposed(c)

    def test_matrix_property(self, qtbot):
        c = _DnaHeatmapCanvas()
        qtbot.addWidget(c)
        matrix = [[None] * 24 for _ in range(7)]
        c.set_data(matrix)
        assert len(c.matrix) == 7
        assert len(c.matrix[0]) == 24


# ─────────────────────────────────────────────────────────────────────────────
#  TestHistoryTab
# ─────────────────────────────────────────────────────────────────────────────

class TestHistoryTab:
    def test_initial_row_count_zero(self, qtbot):
        t = _HistoryTab()
        qtbot.addWidget(t)
        assert t.table.rowCount() == 0

    def test_load_trades_fills_table(self, qtbot):
        t = _HistoryTab()
        qtbot.addWidget(t)
        trades = _mock_backend().get_trades()
        t.load_trades(trades)
        assert t.table.rowCount() == 2

    def test_load_trades_symbol_in_row(self, qtbot):
        t = _HistoryTab()
        qtbot.addWidget(t)
        t.load_trades([{"id": 1, "symbol": "EURUSD", "direction": "buy",
                        "lot_size": 0.1, "entry_time": "", "exit_time": "",
                        "pnl": 10.0, "status": "closed", "setup": ""}])
        assert t.table.item(0, 1).text() == "EURUSD"

    def test_pnl_positive_green(self, qtbot):
        t = _HistoryTab()
        qtbot.addWidget(t)
        t.load_trades([{"id": 1, "symbol": "X", "direction": "buy",
                        "lot_size": 0.1, "entry_time": "", "exit_time": "",
                        "pnl": 50.0, "status": "closed", "setup": ""}])
        color = t.table.item(0, 6).foreground().color()
        assert color.green() > 150

    def test_pnl_negative_red(self, qtbot):
        t = _HistoryTab()
        qtbot.addWidget(t)
        t.load_trades([{"id": 1, "symbol": "X", "direction": "sell",
                        "lot_size": 0.1, "entry_time": "", "exit_time": "",
                        "pnl": -30.0, "status": "closed", "setup": ""}])
        color = t.table.item(0, 6).foreground().color()
        assert color.red() > 150

    def test_pnl_none_shows_dash(self, qtbot):
        t = _HistoryTab()
        qtbot.addWidget(t)
        t.load_trades([{"id": 2, "symbol": "X", "direction": "buy",
                        "lot_size": 0.1, "entry_time": "", "exit_time": "",
                        "pnl": None, "status": "open", "setup": ""}])
        assert t.table.item(0, 6).text() == "–"

    def test_local_filter_hides_rows(self, qtbot):
        t = _HistoryTab()
        qtbot.addWidget(t)
        trades = _mock_backend().get_trades()
        t.load_trades(trades)
        t.apply_local_filter("EURUSD")
        assert t.visible_row_count() == 1

    def test_local_filter_empty_shows_all(self, qtbot):
        t = _HistoryTab()
        qtbot.addWidget(t)
        t.load_trades(_mock_backend().get_trades())
        t.apply_local_filter("")
        assert t.visible_row_count() == 2

    def test_local_filter_case_insensitive(self, qtbot):
        t = _HistoryTab()
        qtbot.addWidget(t)
        t.load_trades(_mock_backend().get_trades())
        t.apply_local_filter("eurusd")
        assert t.visible_row_count() == 1

    def test_replay_btn_disabled_initially(self, qtbot):
        t = _HistoryTab()
        qtbot.addWidget(t)
        assert not t.replay_btn.isEnabled()

    def test_coach_btn_disabled_initially(self, qtbot):
        t = _HistoryTab()
        qtbot.addWidget(t)
        assert not t.coach_btn.isEnabled()

    def test_replay_btn_enabled_on_selection(self, qtbot):
        t = _HistoryTab()
        qtbot.addWidget(t)
        t.load_trades(_mock_backend().get_trades())
        t.table.selectRow(0)
        assert t.replay_btn.isEnabled()

    def test_coach_btn_enabled_on_selection(self, qtbot):
        t = _HistoryTab()
        qtbot.addWidget(t)
        t.load_trades(_mock_backend().get_trades())
        t.table.selectRow(0)
        assert t.coach_btn.isEnabled()

    def test_replay_requested_signal(self, qtbot):
        t = _HistoryTab()
        qtbot.addWidget(t)
        t.load_trades(_mock_backend().get_trades())
        # Find row with id=1
        for r in range(t.table.rowCount()):
            if t.table.item(r, 0).text() == "1":
                t.table.selectRow(r)
                break
        signals = []
        t.replay_requested.connect(signals.append)
        t.replay_btn.click()
        assert signals == [1]

    def test_coach_trade_added_signal(self, qtbot):
        t = _HistoryTab()
        qtbot.addWidget(t)
        t.load_trades(_mock_backend().get_trades())
        for r in range(t.table.rowCount()):
            if t.table.item(r, 0).text() == "1":
                t.table.selectRow(r)
                break
        signals = []
        t.coach_trade_added.connect(signals.append)
        t.coach_btn.click()
        assert signals == [1]

    def test_columns_count(self, qtbot):
        t = _HistoryTab()
        qtbot.addWidget(t)
        assert t.table.columnCount() == 9

    def test_status_combo_options(self, qtbot):
        t = _HistoryTab()
        qtbot.addWidget(t)
        texts = [t.status_combo.itemText(i) for i in range(t.status_combo.count())]
        assert "Alle" in texts
        assert "open"   in texts
        assert "closed" in texts


# ─────────────────────────────────────────────────────────────────────────────
#  TestReplayTab
# ─────────────────────────────────────────────────────────────────────────────

class TestReplayTab:
    def test_initial_trade_id_none(self, qtbot):
        r = _ReplayTab()
        qtbot.addWidget(r)
        assert r.current_trade_id is None

    def test_chart_widget_present(self, qtbot):
        r = _ReplayTab()
        qtbot.addWidget(r)
        from gui.widgets.chart_widget import ChartWidget
        assert isinstance(r.chart_widget, ChartWidget)

    def test_load_replay_sets_trade_id(self, qtbot):
        r = _ReplayTab()
        qtbot.addWidget(r)
        data = _mock_backend().get_replay_data(1)
        r.load_replay(data)
        assert r.current_trade_id == 1

    def test_load_replay_updates_info_label(self, qtbot):
        r = _ReplayTab()
        qtbot.addWidget(r)
        data = _mock_backend().get_replay_data(1)
        r.load_replay(data)
        assert "EURUSD" in r.trade_info_label.text()

    def test_load_replay_shows_direction(self, qtbot):
        r = _ReplayTab()
        qtbot.addWidget(r)
        data = _mock_backend().get_replay_data(1)
        r.load_replay(data)
        assert "BUY" in r.trade_info_label.text().upper()

    def test_load_replay_shows_pnl(self, qtbot):
        r = _ReplayTab()
        qtbot.addWidget(r)
        data = _mock_backend().get_replay_data(1)
        r.load_replay(data)
        assert "55" in r.trade_info_label.text()

    def test_load_replay_sets_candles(self, qtbot):
        r = _ReplayTab()
        qtbot.addWidget(r)
        data = _mock_backend().get_replay_data(1)
        r.load_replay(data)
        assert r.chart_widget.candles_count == 2

    def test_load_replay_marker_label_contains_entry(self, qtbot):
        r = _ReplayTab()
        qtbot.addWidget(r)
        data = _mock_backend().get_replay_data(1)
        r.load_replay(data)
        assert "Entry" in r.marker_label.text()

    def test_load_replay_no_lookahead_shown(self, qtbot):
        r = _ReplayTab()
        qtbot.addWidget(r)
        data = _mock_backend().get_replay_data(1)
        r.load_replay(data)
        assert "No-Lookahead" in r.marker_label.text()

    def test_load_replay_empty_candles_no_crash(self, qtbot):
        r = _ReplayTab()
        qtbot.addWidget(r)
        data = {
            "trade": {"id": 5, "symbol": "XAUUSD", "direction": "buy",
                      "entry_price": None, "exit_price": None, "pnl": None},
            "candles": [], "entry_marker": {}, "exit_marker": None,
            "indicators": {}, "news_events": [],
            "meta": {"no_lookahead": True},
        }
        r.load_replay(data)  # must not raise
        assert r.current_trade_id == 5


# ─────────────────────────────────────────────────────────────────────────────
#  TestCoachTab
# ─────────────────────────────────────────────────────────────────────────────

class TestCoachTab:
    def test_initial_trade_ids_empty(self, qtbot):
        c = _CoachTab()
        qtbot.addWidget(c)
        assert c.trade_ids == []

    def test_add_trade_id(self, qtbot):
        c = _CoachTab()
        qtbot.addWidget(c)
        c.add_trade_id(42)
        assert 42 in c.trade_ids

    def test_add_trade_id_no_duplicates(self, qtbot):
        c = _CoachTab()
        qtbot.addWidget(c)
        c.add_trade_id(1)
        c.add_trade_id(1)
        assert c.trade_ids.count(1) == 1

    def test_ctx_label_updates(self, qtbot):
        c = _CoachTab()
        qtbot.addWidget(c)
        c.add_trade_id(7)
        assert "#7" in c.ctx_label.text()

    def test_clear_ctx(self, qtbot):
        c = _CoachTab()
        qtbot.addWidget(c)
        c.add_trade_id(1)
        c.add_trade_id(2)
        c.clear_ctx_btn.click()
        assert c.trade_ids == []
        assert c.ctx_label.text() == "–"

    def test_send_emits_signal(self, qtbot):
        c = _CoachTab()
        qtbot.addWidget(c)
        signals = []
        c.ask_requested.connect(lambda q, ids: signals.append((q, ids)))
        c.input_edit.setText("Was kann ich verbessern?")
        c.send_btn.click()
        assert len(signals) == 1
        assert signals[0][0] == "Was kann ich verbessern?"

    def test_send_includes_trade_ids(self, qtbot):
        c = _CoachTab()
        qtbot.addWidget(c)
        c.add_trade_id(3)
        c.add_trade_id(5)
        signals = []
        c.ask_requested.connect(lambda q, ids: signals.append(ids))
        c.input_edit.setText("?")
        c.send_btn.click()
        assert 3 in signals[0]
        assert 5 in signals[0]

    def test_send_clears_input(self, qtbot):
        c = _CoachTab()
        qtbot.addWidget(c)
        c.input_edit.setText("test")
        c.send_btn.click()
        assert c.input_edit.text() == ""

    def test_empty_input_no_signal(self, qtbot):
        c = _CoachTab()
        qtbot.addWidget(c)
        signals = []
        c.ask_requested.connect(lambda q, ids: signals.append(q))
        c.input_edit.clear()
        c.send_btn.click()
        assert signals == []

    def test_show_response(self, qtbot):
        c = _CoachTab()
        qtbot.addWidget(c)
        c.show_response("Antwort des KI-Coach.")
        assert "Antwort des KI-Coach." in c.chat_display.toPlainText()

    def test_return_pressed_sends(self, qtbot):
        c = _CoachTab()
        qtbot.addWidget(c)
        signals = []
        c.ask_requested.connect(lambda q, ids: signals.append(q))
        c.input_edit.setText("Hallo Coach")
        c.input_edit.returnPressed.emit()
        assert signals == ["Hallo Coach"]


# ─────────────────────────────────────────────────────────────────────────────
#  TestReportTab
# ─────────────────────────────────────────────────────────────────────────────

class TestReportTab:
    def test_period_combo_options(self, qtbot):
        r = _ReportTab()
        qtbot.addWidget(r)
        texts = [r.period_combo.itemText(i) for i in range(r.period_combo.count())]
        assert "daily"  in texts
        assert "weekly" in texts

    def test_report_edit_readonly(self, qtbot):
        r = _ReportTab()
        qtbot.addWidget(r)
        assert r.report_edit.isReadOnly()

    def test_show_report(self, qtbot):
        r = _ReportTab()
        qtbot.addWidget(r)
        r.show_report("# Mein Report")
        assert "# Mein Report" in r.report_edit.toPlainText()

    def test_export_report_writes_file(self, qtbot, tmp_path):
        r = _ReportTab()
        qtbot.addWidget(r)
        r.show_report("Test-Inhalt")
        out = tmp_path / "report.md"
        r.export_report(str(out))
        assert out.read_text(encoding="utf-8") == "Test-Inhalt"

    def test_load_btn_exists(self, qtbot):
        r = _ReportTab()
        qtbot.addWidget(r)
        assert r.load_btn is not None

    def test_export_btn_exists(self, qtbot):
        r = _ReportTab()
        qtbot.addWidget(r)
        assert r.export_btn is not None


# ─────────────────────────────────────────────────────────────────────────────
#  TestDnaTab
# ─────────────────────────────────────────────────────────────────────────────

class TestDnaTab:
    def test_initial_status_label(self, qtbot):
        t = _DnaTab()
        qtbot.addWidget(t)
        assert t.status_label.text() != ""

    def test_load_profile_insufficient(self, qtbot):
        t = _DnaTab()
        qtbot.addWidget(t)
        t.load_profile({"status": "insufficient_data", "n_trades": 50, "min_trades_required": 500})
        assert "50" in t.status_label.text()
        assert "500" in t.status_label.text()

    def test_load_profile_ready(self, qtbot):
        t = _DnaTab()
        qtbot.addWidget(t)
        profile = _mock_backend().get_dna_profile()
        t.load_profile(profile)
        assert "600" in t.status_label.text()

    def test_load_profile_fills_sym_table(self, qtbot):
        t = _DnaTab()
        qtbot.addWidget(t)
        t.load_profile(_mock_backend().get_dna_profile())
        assert t.sym_table.rowCount() == 1
        assert t.sym_table.item(0, 0).text() == "EURUSD"

    def test_load_profile_fills_setup_table(self, qtbot):
        t = _DnaTab()
        qtbot.addWidget(t)
        t.load_profile(_mock_backend().get_dna_profile())
        assert t.setup_table.rowCount() == 1
        assert t.setup_table.item(0, 0).text() == "Breakout"

    def test_load_profile_shows_weaknesses(self, qtbot):
        t = _DnaTab()
        qtbot.addWidget(t)
        t.load_profile(_mock_backend().get_dna_profile())
        assert "FOMO" in t.weakness_label.text()

    def test_load_profile_no_weaknesses(self, qtbot):
        t = _DnaTab()
        qtbot.addWidget(t)
        profile = _mock_backend().get_dna_profile()
        profile["psychological_weaknesses"] = []
        t.load_profile(profile)
        assert "Keine" in t.weakness_label.text()

    def test_heatmap_widget_present(self, qtbot):
        t = _DnaTab()
        qtbot.addWidget(t)
        assert isinstance(t.heatmap, _DnaHeatmapCanvas)

    def test_load_heatmap_passes_to_canvas(self, qtbot):
        t = _DnaTab()
        qtbot.addWidget(t)
        matrix = [[None] * 24 for _ in range(7)]
        matrix[1][10] = {"win_rate": 0.8, "n_trades": 30}
        t.load_heatmap(matrix)
        assert t.heatmap.get_cell(1, 10) == {"win_rate": 0.8, "n_trades": 30}


# ─────────────────────────────────────────────────────────────────────────────
#  TestJournalView
# ─────────────────────────────────────────────────────────────────────────────

class TestJournalView:
    def test_object_name(self, qtbot):
        v = JournalView()
        qtbot.addWidget(v)
        assert v.objectName() == "journal_view"

    def test_has_5_tabs(self, qtbot):
        v = JournalView()
        qtbot.addWidget(v)
        assert v.tabs.count() == 5

    def test_sub_widgets_present(self, qtbot):
        v = JournalView()
        qtbot.addWidget(v)
        assert isinstance(v.history_tab, _HistoryTab)
        assert isinstance(v.dna_tab,     _DnaTab)
        assert isinstance(v.replay_tab,  _ReplayTab)
        assert isinstance(v.coach_tab,   _CoachTab)
        assert isinstance(v.report_tab,  _ReportTab)

    def test_set_backend(self, qtbot):
        v = JournalView()
        qtbot.addWidget(v)
        b = _mock_backend()
        v.set_backend(b)
        assert v._backend is b

    def test_refresh_history_noop_without_backend(self, qtbot):
        v = JournalView()
        qtbot.addWidget(v)
        v.refresh_history()  # must not raise

    def test_refresh_history_calls_get_trades(self, qtbot):
        b = _mock_backend()
        v = JournalView(backend=b)
        qtbot.addWidget(v)
        v.refresh_history()
        b.get_trades.assert_called_once()

    def test_refresh_history_fills_table(self, qtbot):
        b = _mock_backend()
        v = JournalView(backend=b)
        qtbot.addWidget(v)
        v.refresh_history()
        assert v.history_tab.table.rowCount() == 2

    def test_refresh_history_passes_filters(self, qtbot):
        b = _mock_backend()
        v = JournalView(backend=b)
        qtbot.addWidget(v)
        v.history_tab.sym_filter.setText("EURUSD")
        v.history_tab.status_combo.setCurrentIndex(2)  # closed
        v.refresh_history()
        call_kwargs = b.get_trades.call_args
        assert call_kwargs.kwargs.get("symbol_filter") == "EURUSD"
        assert call_kwargs.kwargs.get("status_filter") == "closed"

    def test_refresh_dna_noop_without_backend(self, qtbot):
        v = JournalView()
        qtbot.addWidget(v)
        v.refresh_dna()  # must not raise

    def test_refresh_dna_calls_backend(self, qtbot):
        b = _mock_backend()
        v = JournalView(backend=b)
        qtbot.addWidget(v)
        v.refresh_dna()
        b.get_dna_profile.assert_called_once()
        b.get_hour_weekday_matrix.assert_called_once()

    # ── Mood popup (injectable) ────────────────────────────────────────────

    def test_mood_open_injectable_saved(self, qtbot):
        b = _mock_backend()
        v = JournalView(backend=b, _mood_fn=_mood_yes)
        qtbot.addWidget(v)
        signals = []
        v.mood_recorded.connect(signals.append)
        v.show_mood_popup_open(trade_id=1, symbol="EURUSD")
        b.record_mood_open.assert_called_once_with(1, "calm", "Alles gut")
        assert len(signals) == 1
        assert signals[0]["type"] == "open"
        assert signals[0]["trade_id"] == 1

    def test_mood_open_injectable_skipped(self, qtbot):
        b = _mock_backend()
        v = JournalView(backend=b, _mood_fn=_mood_no)
        qtbot.addWidget(v)
        signals = []
        v.mood_recorded.connect(signals.append)
        v.show_mood_popup_open(trade_id=1)
        b.record_mood_open.assert_not_called()
        assert signals == []

    def test_mood_close_injectable_saved(self, qtbot):
        b = _mock_backend()
        v = JournalView(backend=b, _mood_fn=_mood_yes)
        qtbot.addWidget(v)
        signals = []
        v.mood_recorded.connect(signals.append)
        v.show_mood_popup_close(trade_id=2, pnl=-30.0)
        b.record_mood_close.assert_called_once_with(2, "calm", True, "Alles gut", -30.0)
        assert signals[0]["type"] == "close"

    def test_mood_close_injectable_skipped(self, qtbot):
        b = _mock_backend()
        v = JournalView(backend=b, _mood_fn=_mood_no)
        qtbot.addWidget(v)
        signals = []
        v.mood_recorded.connect(signals.append)
        v.show_mood_popup_close(trade_id=2)
        b.record_mood_close.assert_not_called()
        assert signals == []

    def test_mood_open_no_backend_no_crash(self, qtbot):
        v = JournalView(_mood_fn=_mood_yes)
        qtbot.addWidget(v)
        v.show_mood_popup_open(1)  # must not raise

    def test_mood_close_no_backend_no_crash(self, qtbot):
        v = JournalView(_mood_fn=_mood_yes)
        qtbot.addWidget(v)
        v.show_mood_popup_close(1)  # must not raise

    def test_mood_recorded_signal_type(self, qtbot):
        b = _mock_backend()
        v = JournalView(backend=b, _mood_fn=_mood_yes)
        qtbot.addWidget(v)
        with qtbot.waitSignal(v.mood_recorded, timeout=1000):
            v.show_mood_popup_open(1)

    # ── Replay ────────────────────────────────────────────────────────────

    def test_replay_request_switches_tab(self, qtbot):
        b = _mock_backend()
        v = JournalView(backend=b)
        qtbot.addWidget(v)
        v.history_tab.load_trades(b.get_trades())
        # Trigger replay for trade 1
        v._on_replay_requested(1)
        assert v.tabs.currentIndex() == 2  # _TAB_REPLAY

    def test_replay_request_no_backend_no_crash(self, qtbot):
        v = JournalView()
        qtbot.addWidget(v)
        v._on_replay_requested(1)  # must not raise

    def test_replay_loads_data(self, qtbot):
        b = _mock_backend()
        v = JournalView(backend=b)
        qtbot.addWidget(v)
        v._on_replay_requested(1)
        assert v.replay_tab.current_trade_id == 1

    # ── Coach ────────────────────────────────────────────────────────────

    def test_coach_trade_added_from_history(self, qtbot):
        v = JournalView()
        qtbot.addWidget(v)
        v.history_tab.coach_trade_added.emit(99)
        assert 99 in v.coach_tab.trade_ids

    def test_coach_ask_calls_backend(self, qtbot):
        b = _mock_backend()
        v = JournalView(backend=b)
        qtbot.addWidget(v)
        v._on_ask_coach("Meine Frage?", [1, 2])
        b.ask_coach.assert_called_once_with("Meine Frage?", [1, 2])

    def test_coach_response_displayed(self, qtbot):
        b = _mock_backend()
        v = JournalView(backend=b)
        qtbot.addWidget(v)
        v._on_ask_coach("?", [])
        assert "Trade #1" in v.coach_tab.chat_display.toPlainText()

    def test_coach_no_backend_shows_error(self, qtbot):
        v = JournalView()
        qtbot.addWidget(v)
        v._on_ask_coach("Frage", [])
        assert "Kein Backend" in v.coach_tab.chat_display.toPlainText()

    # ── Report ────────────────────────────────────────────────────────────

    def test_load_report_calls_backend(self, qtbot):
        b = _mock_backend()
        v = JournalView(backend=b)
        qtbot.addWidget(v)
        v._on_load_report()
        b.generate_report.assert_called_once()

    def test_load_report_displays_text(self, qtbot):
        b = _mock_backend()
        v = JournalView(backend=b)
        qtbot.addWidget(v)
        v._on_load_report()
        assert "Report" in v.report_tab.report_edit.toPlainText()

    def test_load_report_noop_without_backend(self, qtbot):
        v = JournalView()
        qtbot.addWidget(v)
        v._on_load_report()  # must not raise

    def test_export_report_with_fn(self, qtbot):
        b = _mock_backend()
        v = JournalView(backend=b)
        qtbot.addWidget(v)
        v._on_load_report()
        exported = []
        v._on_export_report(_export_fn=lambda text: exported.append(text))
        assert len(exported) == 1
        assert "Report" in exported[0]

    def test_export_empty_report_no_crash(self, qtbot):
        v = JournalView()
        qtbot.addWidget(v)
        v._on_export_report(_export_fn=lambda t: None)  # must not raise

    # ── Text search ───────────────────────────────────────────────────────

    def test_text_search_triggers_local_filter(self, qtbot):
        b = _mock_backend()
        v = JournalView(backend=b)
        qtbot.addWidget(v)
        v.refresh_history()
        v.history_tab.text_search.setText("EURUSD")
        assert v.history_tab.visible_row_count() == 1

    def test_apply_filter_refreshes_from_backend(self, qtbot):
        b = _mock_backend()
        v = JournalView(backend=b)
        qtbot.addWidget(v)
        v.history_tab.apply_btn.click()
        b.get_trades.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
#  TestMainWindowJournalIntegration
# ─────────────────────────────────────────────────────────────────────────────

class TestMainWindowJournalIntegration:
    @pytest.fixture
    def fresh_window(self, qtbot, fresh_theme):
        win = MainWindow(theme_manager=fresh_theme)
        qtbot.addWidget(win)
        return win

    def test_still_has_seven_views_in_stack(self, fresh_window):
        assert fresh_window.content.count() == 7

    def test_journal_view_property_exists(self, fresh_window):
        assert isinstance(fresh_window.journal_view, JournalView)

    def test_journal_section_registered(self, fresh_window):
        assert Section.JOURNAL in fresh_window._views

    def test_navigate_to_journal(self, fresh_window):
        fresh_window.navigate_to(Section.JOURNAL)
        assert fresh_window.current_view() is fresh_window.journal_view

    def test_journal_backend_passed_through(self, qtbot, fresh_theme):
        b = _mock_backend()
        win = MainWindow(theme_manager=fresh_theme, journal_backend=b)
        qtbot.addWidget(win)
        assert win.journal_view._backend is b

    def test_journal_view_object_name(self, fresh_window):
        assert fresh_window.journal_view.objectName() == "journal_view"
