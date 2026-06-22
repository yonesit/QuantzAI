"""
tests/gui/test_backtest_view.py
GUI-Tests fuer gui/views/backtest_view.py via pytest-qt.

Abgedeckt:
  BacktestView
    - Erstellt ohne Absturz
    - Alle Eingabefelder zugaenglich und mit Defaults
    - Run-Button loest Worker aus (via _run_fn-Injection)
    - _on_result_received() injiziert Ergebnis ohne Worker
    - Progress sichtbar/unsichtbar
    - Metrics werden nach Ergebnis befuellt
    - Equity-Canvas hat Daten nach Ergebnis
    - Overfitting-Warnung sichtbar/unsichtbar
    - Export-Button freigeschaltet nach Ergebnis
    - _on_export_clicked() nutzt _export_fn
    - Laeufe landen in Vergleichstabelle
    - Walk-Forward-Fenster bei IS/OOS-Ergebnis
    - Fehlerbehandlung: kein Backend, Worker-Fehler
    - Navigations-Tabs zugaenglich
    - Lauf-Auswahl in Vergleichstabelle wechselt Equity-Canvas
    - Mehrere Laeufe angehaeuft

  _EquityCurveCanvas
    - has_data False/True
    - set_result/clear
    - is_oos_split_index berechnet
    - paintEvent kein Absturz (leer & mit Daten)

  _MetricsGrid
    - Alle Metric-Labels vorhanden
    - Tooltips gesetzt
    - set_result befuellt Labels
    - set_result(None) zeigt Striche

  _RunsTable
    - add_run fuegt Zeile ein
    - run_count korrekt
    - clear_runs leert Tabelle
    - get_run gibt Tupel zurueck
    - run_selected Signal

  _WalkForwardPanel
    - add_window fuegt Zeile ein
    - window_count korrekt
    - clear_windows leert Tabelle

  Integration MainWindow
    - backtest_view Property vorhanden
    - Navigation zu Backtest funktioniert
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pandas as pd
import pytest
from pytestqt.qtbot import QtBot

from gui.views.backtest_view import (
    BacktestView,
    _EquityCurveCanvas,
    _MetricsGrid,
    _RunsTable,
    _WalkForwardPanel,
    _result_to_markdown,
    _METRIC_TOOLTIPS,
)
from src.backtesting.vectorbt_runner import BacktestResult


# ─────────────────────────────────────────────────────────────────────────────
#  Hilfsfunktionen
# ─────────────────────────────────────────────────────────────────────────────

def _make_equity(n: int = 50, start: float = 10_000.0, step: float = 100.0) -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.Series([start + i * step for i in range(n)], index=idx)


def _make_result(
    total_return: float = 0.12,
    sharpe: float = 1.5,
    sortino: float = 2.0,
    max_dd: float = -0.08,
    profit_factor: float = 1.8,
    win_rate: float = 0.6,
    avg_win: float = 120.0,
    avg_loss: float = -80.0,
    n_trades: int = 42,
    equity: pd.Series | None = None,
    is_sharpe: float | None = None,
    oos_sharpe: float | None = None,
    overfitting: bool = False,
) -> BacktestResult:
    return BacktestResult(
        total_return=total_return,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_dd,
        profit_factor=profit_factor,
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        n_trades=n_trades,
        equity_curve=equity if equity is not None else _make_equity(),
        is_sharpe=is_sharpe,
        oos_sharpe=oos_sharpe,
        overfitting_warning=overfitting,
    )


def _make_view(qtbot: QtBot, run_fn=None, export_fn=None) -> BacktestView:
    v = BacktestView(_run_fn=run_fn, _export_fn=export_fn)
    qtbot.addWidget(v)
    return v


# ─────────────────────────────────────────────────────────────────────────────
#  BacktestView – Initialisierung
# ─────────────────────────────────────────────────────────────────────────────

class TestBacktestViewInit:

    def test_creates_without_crash(self, qtbot: QtBot):
        v = _make_view(qtbot)
        assert v is not None

    def test_symbol_input_default(self, qtbot: QtBot):
        v = _make_view(qtbot)
        assert v.symbol_input.text() == "EURUSD"

    def test_timeframe_combo_default_h1(self, qtbot: QtBot):
        v = _make_view(qtbot)
        assert v.timeframe_combo.currentText() == "H1"

    def test_timeframe_combo_has_h4(self, qtbot: QtBot):
        v = _make_view(qtbot)
        items = [v.timeframe_combo.itemText(i) for i in range(v.timeframe_combo.count())]
        assert "H4" in items

    def test_start_input_has_value(self, qtbot: QtBot):
        v = _make_view(qtbot)
        assert v.start_input.text() != ""

    def test_end_input_has_value(self, qtbot: QtBot):
        v = _make_view(qtbot)
        assert v.end_input.text() != ""

    def test_is_split_input_empty_by_default(self, qtbot: QtBot):
        v = _make_view(qtbot)
        assert v.is_split_input.text() == ""

    def test_init_cash_default_10000(self, qtbot: QtBot):
        v = _make_view(qtbot)
        assert v.init_cash_spinbox.value() == 10_000.0

    def test_run_button_enabled(self, qtbot: QtBot):
        v = _make_view(qtbot)
        assert v.run_button.isEnabled()

    def test_export_button_disabled_initially(self, qtbot: QtBot):
        v = _make_view(qtbot)
        assert not v.export_button.isEnabled()

    def test_progress_bar_hidden_initially(self, qtbot: QtBot):
        v = _make_view(qtbot)
        assert v.progress_bar.isHidden()

    def test_overfitting_label_hidden_initially(self, qtbot: QtBot):
        v = _make_view(qtbot)
        assert v.overfitting_label.isHidden()

    def test_error_label_hidden_initially(self, qtbot: QtBot):
        v = _make_view(qtbot)
        assert v.error_label.isHidden()

    def test_results_tabs_has_three_tabs(self, qtbot: QtBot):
        v = _make_view(qtbot)
        assert v.results_tabs.count() == 3

    def test_current_result_none_initially(self, qtbot: QtBot):
        v = _make_view(qtbot)
        assert v.current_result is None


# ─────────────────────────────────────────────────────────────────────────────
#  BacktestView – Ergebnis empfangen (_on_result_received)
# ─────────────────────────────────────────────────────────────────────────────

class TestBacktestViewOnResult:

    def test_equity_canvas_has_data_after_result(self, qtbot: QtBot):
        v = _make_view(qtbot)
        v._on_result_received(_make_result())
        assert v.equity_canvas.has_data

    def test_export_button_enabled_after_result(self, qtbot: QtBot):
        v = _make_view(qtbot)
        v._on_result_received(_make_result())
        assert v.export_button.isEnabled()

    def test_run_button_re_enabled_after_result(self, qtbot: QtBot):
        v = _make_view(qtbot)
        v._run_btn.setEnabled(False)
        v._on_result_received(_make_result())
        assert v.run_button.isEnabled()

    def test_progress_hidden_after_result(self, qtbot: QtBot):
        v = _make_view(qtbot)
        v._progress.setVisible(True)
        v._on_result_received(_make_result())
        assert v.progress_bar.isHidden()

    def test_overfitting_shown_when_warning(self, qtbot: QtBot):
        v = _make_view(qtbot)
        v._on_result_received(_make_result(overfitting=True))
        assert not v.overfitting_label.isHidden()

    def test_overfitting_hidden_when_no_warning(self, qtbot: QtBot):
        v = _make_view(qtbot)
        v._on_result_received(_make_result(overfitting=False))
        assert v.overfitting_label.isHidden()

    def test_result_stored_in_current_result(self, qtbot: QtBot):
        v  = _make_view(qtbot)
        r  = _make_result()
        v._on_result_received(r)
        assert v.current_result is r

    def test_run_added_to_comparison_table(self, qtbot: QtBot):
        v = _make_view(qtbot)
        v._on_result_received(_make_result())
        assert v.runs_table.run_count() == 1

    def test_multiple_runs_accumulated(self, qtbot: QtBot):
        v = _make_view(qtbot)
        v._on_result_received(_make_result())
        v._on_result_received(_make_result())
        v._on_result_received(_make_result())
        assert v.runs_table.run_count() == 3

    def test_backtest_finished_signal_emitted(self, qtbot: QtBot):
        v   = _make_view(qtbot)
        r   = _make_result()
        with qtbot.waitSignal(v.backtest_finished, timeout=1000) as blocker:
            v._on_result_received(r)
        assert blocker.args[0] is r

    def test_wf_window_added_when_is_oos_split(self, qtbot: QtBot):
        v = _make_view(qtbot)
        v.is_split_input.setText("2023-07-01")
        r = _make_result(is_sharpe=1.2, oos_sharpe=0.8)
        v._on_result_received(r)
        assert v.walk_forward_panel.window_count() == 1

    def test_wf_window_not_added_when_no_split(self, qtbot: QtBot):
        v = _make_view(qtbot)
        r = _make_result(is_sharpe=None, oos_sharpe=None)
        v._on_result_received(r)
        assert v.walk_forward_panel.window_count() == 0


# ─────────────────────────────────────────────────────────────────────────────
#  BacktestView – Fehlerbehandlung
# ─────────────────────────────────────────────────────────────────────────────

class TestBacktestViewErrors:

    def test_error_shown_when_no_backend(self, qtbot: QtBot):
        v = BacktestView(_run_fn=None, _export_fn=None)
        qtbot.addWidget(v)
        v.run_button.click()
        assert not v.error_label.isHidden()
        assert v.error_label.text() != ""

    def test_run_btn_stays_enabled_when_no_backend(self, qtbot: QtBot):
        v = BacktestView(_run_fn=None)
        qtbot.addWidget(v)
        v.run_button.click()
        assert v.run_button.isEnabled()

    def test_on_run_failed_shows_error(self, qtbot: QtBot):
        v = _make_view(qtbot)
        v._on_run_failed("vectorbt Fehler XYZ")
        assert not v.error_label.isHidden()
        assert "vectorbt Fehler XYZ" in v.error_label.text()

    def test_on_run_failed_re_enables_run_btn(self, qtbot: QtBot):
        v = _make_view(qtbot)
        v._run_btn.setEnabled(False)
        v._on_run_failed("err")
        assert v.run_button.isEnabled()

    def test_on_run_failed_hides_progress(self, qtbot: QtBot):
        v = _make_view(qtbot)
        v._progress.setVisible(True)
        v._on_run_failed("err")
        assert v.progress_bar.isHidden()

    def test_on_run_failed_emits_signal(self, qtbot: QtBot):
        v = _make_view(qtbot)
        with qtbot.waitSignal(v.backtest_failed, timeout=1000) as blocker:
            v._on_run_failed("fail!")
        assert blocker.args[0] == "fail!"


# ─────────────────────────────────────────────────────────────────────────────
#  BacktestView – Worker-Thread via _run_fn
# ─────────────────────────────────────────────────────────────────────────────

class TestBacktestViewWorker:

    def test_start_triggers_run_fn(self, qtbot: QtBot):
        called = {}
        result = _make_result()

        def fake_run(**kwargs):
            called.update(kwargs)
            return result

        v = _make_view(qtbot, run_fn=fake_run)
        with qtbot.waitSignal(v.backtest_finished, timeout=5000):
            v.run_button.click()
        assert "symbol" in called

    def test_start_passes_symbol(self, qtbot: QtBot):
        called = {}

        def fake_run(**kwargs):
            called.update(kwargs)
            return _make_result()

        v = _make_view(qtbot, run_fn=fake_run)
        v.symbol_input.setText("GBPUSD")
        with qtbot.waitSignal(v.backtest_finished, timeout=5000):
            v.run_button.click()
        assert called.get("symbol") == "GBPUSD"

    def test_start_passes_timeframe(self, qtbot: QtBot):
        called = {}

        def fake_run(**kwargs):
            called.update(kwargs)
            return _make_result()

        v = _make_view(qtbot, run_fn=fake_run)
        v.timeframe_combo.setCurrentText("H4")
        with qtbot.waitSignal(v.backtest_finished, timeout=5000):
            v.run_button.click()
        assert called.get("timeframe") == "H4"

    def test_start_passes_none_for_empty_is_split(self, qtbot: QtBot):
        called = {}

        def fake_run(**kwargs):
            called.update(kwargs)
            return _make_result()

        v = _make_view(qtbot, run_fn=fake_run)
        v.is_split_input.setText("")
        with qtbot.waitSignal(v.backtest_finished, timeout=5000):
            v.run_button.click()
        assert called.get("is_split") is None

    def test_progress_shown_immediately_after_click(self, qtbot: QtBot):
        # Progress wird synchron gesetzt bevor der Worker-Thread startet
        v = _make_view(qtbot, run_fn=lambda **kwargs: _make_result())
        assert v.progress_bar.isHidden()
        with qtbot.waitSignal(v.backtest_finished, timeout=5000):
            v.run_button.click()
            assert not v.progress_bar.isHidden()

    def test_backtest_started_signal_emitted(self, qtbot: QtBot):
        received = []
        v = _make_view(qtbot, run_fn=lambda **kwargs: _make_result())
        v.backtest_started.connect(lambda: received.append("started"))
        # Warte auf finished damit Worker-Thread sauber beendet wird
        with qtbot.waitSignal(v.backtest_finished, timeout=5000):
            v.run_button.click()
        assert "started" in received


# ─────────────────────────────────────────────────────────────────────────────
#  BacktestView – Export
# ─────────────────────────────────────────────────────────────────────────────

class TestBacktestViewExport:

    def test_export_fn_called_with_markdown(self, qtbot: QtBot):
        received = {}
        v = _make_view(qtbot, export_fn=lambda md: received.update({"md": md}))
        v._on_result_received(_make_result())
        v._on_export_clicked()
        assert "md" in received
        assert "Backtest" in received["md"]

    def test_export_markdown_contains_return(self, qtbot: QtBot):
        received = {}
        v = _make_view(qtbot, export_fn=lambda md: received.update({"md": md}))
        v._on_result_received(_make_result(total_return=0.25))
        v._on_export_clicked()
        assert "25" in received["md"]

    def test_export_not_called_when_no_result(self, qtbot: QtBot):
        called = []
        v = _make_view(qtbot, export_fn=lambda md: called.append(md))
        v._on_export_clicked()
        assert len(called) == 0

    def test_export_fn_param_overrides_stored(self, qtbot: QtBot):
        stored_called = []
        param_called  = []
        v = _make_view(qtbot, export_fn=lambda md: stored_called.append(md))
        v._on_result_received(_make_result())
        v._on_export_clicked(_export_fn=lambda md: param_called.append(md))
        assert len(param_called) == 1
        assert len(stored_called) == 0


# ─────────────────────────────────────────────────────────────────────────────
#  BacktestView – Lauf-Auswahl im Vergleich
# ─────────────────────────────────────────────────────────────────────────────

class TestBacktestViewRunSelection:

    def test_run_selection_updates_equity_canvas(self, qtbot: QtBot):
        v  = _make_view(qtbot)
        r1 = _make_result(total_return=0.1)
        r2 = _make_result(total_return=0.3)
        v._on_result_received(r1)
        v._on_result_received(r2)
        v.runs_table.table.selectRow(0)
        # Equity canvas sollte auf r1 gesetzt sein
        assert v.equity_canvas.has_data

    def test_run_selection_switches_to_result_tab(self, qtbot: QtBot):
        v = _make_view(qtbot)
        v._on_result_received(_make_result())
        v.results_tabs.setCurrentIndex(1)   # Vergleich-Tab
        v._on_run_selected(0)
        assert v.results_tabs.currentIndex() == 0

    def test_run_selection_updates_metrics(self, qtbot: QtBot):
        v  = _make_view(qtbot)
        r1 = _make_result(n_trades=10)
        v._on_result_received(r1)
        v._on_run_selected(0)
        assert v.metrics_grid.metric_label("Trades").text() == "10"


# ─────────────────────────────────────────────────────────────────────────────
#  _EquityCurveCanvas
# ─────────────────────────────────────────────────────────────────────────────

class TestEquityCurveCanvas:

    def test_has_no_data_initially(self, qtbot: QtBot):
        c = _EquityCurveCanvas()
        qtbot.addWidget(c)
        assert not c.has_data

    def test_has_data_after_set_result(self, qtbot: QtBot):
        c = _EquityCurveCanvas()
        qtbot.addWidget(c)
        c.set_result(_make_result())
        assert c.has_data

    def test_has_no_data_after_clear(self, qtbot: QtBot):
        c = _EquityCurveCanvas()
        qtbot.addWidget(c)
        c.set_result(_make_result())
        c.set_result(None)
        assert not c.has_data

    def test_is_oos_split_none_without_mask(self, qtbot: QtBot):
        c = _EquityCurveCanvas()
        qtbot.addWidget(c)
        c.set_result(_make_result())
        assert c.is_oos_split_index is None

    def test_is_oos_split_with_mask(self, qtbot: QtBot):
        c = _EquityCurveCanvas()
        qtbot.addWidget(c)
        n    = 50
        eq   = _make_equity(n)
        mask = pd.Series([True] * 25 + [False] * 25, index=eq.index)
        c.set_result(_make_result(equity=eq), is_mask=mask)
        assert c.is_oos_split_index == 25

    def test_is_oos_split_all_is_returns_none(self, qtbot: QtBot):
        c = _EquityCurveCanvas()
        qtbot.addWidget(c)
        n    = 20
        eq   = _make_equity(n)
        mask = pd.Series([True] * n, index=eq.index)
        c.set_result(_make_result(equity=eq), is_mask=mask)
        assert c.is_oos_split_index is None

    def test_paint_event_no_crash_empty(self, qtbot: QtBot):
        c = _EquityCurveCanvas()
        qtbot.addWidget(c)
        c.show()
        qtbot.waitExposed(c)
        c.repaint()   # kein Absturz erwartet

    def test_paint_event_no_crash_with_data(self, qtbot: QtBot):
        c = _EquityCurveCanvas()
        qtbot.addWidget(c)
        c.set_result(_make_result())
        c.show()
        qtbot.waitExposed(c)
        c.repaint()

    def test_paint_event_no_crash_with_is_oos(self, qtbot: QtBot):
        c  = _EquityCurveCanvas()
        qtbot.addWidget(c)
        n  = 40
        eq = _make_equity(n)
        mask = pd.Series([True] * 20 + [False] * 20, index=eq.index)
        c.set_result(_make_result(equity=eq), is_mask=mask)
        c.show()
        qtbot.waitExposed(c)
        c.repaint()

    def test_minimum_height(self, qtbot: QtBot):
        c = _EquityCurveCanvas()
        qtbot.addWidget(c)
        assert c.minimumHeight() >= 150


# ─────────────────────────────────────────────────────────────────────────────
#  _MetricsGrid
# ─────────────────────────────────────────────────────────────────────────────

class TestMetricsGrid:

    def test_creates_without_crash(self, qtbot: QtBot):
        g = _MetricsGrid()
        qtbot.addWidget(g)
        assert g is not None

    def test_all_metric_names_present(self, qtbot: QtBot):
        g = _MetricsGrid()
        qtbot.addWidget(g)
        for name in g.metric_names:
            lbl = g.metric_label(name)
            assert lbl is not None

    def test_initial_values_are_dash(self, qtbot: QtBot):
        g = _MetricsGrid()
        qtbot.addWidget(g)
        for name in g.metric_names:
            assert g.metric_label(name).text() == "–"

    def test_set_result_updates_trades(self, qtbot: QtBot):
        g = _MetricsGrid()
        qtbot.addWidget(g)
        g.set_result(_make_result(n_trades=77))
        assert g.metric_label("Trades").text() == "77"

    def test_set_result_updates_win_rate(self, qtbot: QtBot):
        g = _MetricsGrid()
        qtbot.addWidget(g)
        g.set_result(_make_result(win_rate=0.65))
        assert "65" in g.metric_label("Win-Rate").text()

    def test_set_result_updates_sharpe(self, qtbot: QtBot):
        g = _MetricsGrid()
        qtbot.addWidget(g)
        g.set_result(_make_result(sharpe=2.345))
        assert "2.345" in g.metric_label("Sharpe Ratio").text()

    def test_set_result_updates_total_return(self, qtbot: QtBot):
        g = _MetricsGrid()
        qtbot.addWidget(g)
        g.set_result(_make_result(total_return=0.1))
        lbl = g.metric_label("Gesamtertrag").text()
        assert "10" in lbl or "0.10" in lbl

    def test_set_none_resets_to_dash(self, qtbot: QtBot):
        g = _MetricsGrid()
        qtbot.addWidget(g)
        g.set_result(_make_result())
        g.set_result(None)
        for name in g.metric_names:
            assert g.metric_label(name).text() == "–"

    def test_tooltips_set_for_sharpe(self, qtbot: QtBot):
        g = _MetricsGrid()
        qtbot.addWidget(g)
        lbl = g.metric_label("Sharpe Ratio")
        assert lbl.toolTip() != ""

    def test_tooltips_set_for_max_drawdown(self, qtbot: QtBot):
        g = _MetricsGrid()
        qtbot.addWidget(g)
        lbl = g.metric_label("Max. Drawdown")
        assert lbl.toolTip() != ""

    def test_infinite_profit_factor_shown(self, qtbot: QtBot):
        g = _MetricsGrid()
        qtbot.addWidget(g)
        g.set_result(_make_result(profit_factor=float("inf")))
        assert "∞" in g.metric_label("Gewinnfaktor").text()

    def test_is_sharpe_dash_when_none(self, qtbot: QtBot):
        g = _MetricsGrid()
        qtbot.addWidget(g)
        g.set_result(_make_result(is_sharpe=None))
        assert g.metric_label("IS Sharpe").text() == "–"

    def test_oos_sharpe_shown_when_set(self, qtbot: QtBot):
        g = _MetricsGrid()
        qtbot.addWidget(g)
        g.set_result(_make_result(oos_sharpe=0.987))
        assert "0.987" in g.metric_label("OOS Sharpe").text()


# ─────────────────────────────────────────────────────────────────────────────
#  _RunsTable
# ─────────────────────────────────────────────────────────────────────────────

class TestRunsTable:

    def test_empty_initially(self, qtbot: QtBot):
        t = _RunsTable()
        qtbot.addWidget(t)
        assert t.run_count() == 0

    def test_add_run_increases_count(self, qtbot: QtBot):
        t = _RunsTable()
        qtbot.addWidget(t)
        t.add_run("Run 1", _make_result())
        assert t.run_count() == 1

    def test_multiple_runs(self, qtbot: QtBot):
        t = _RunsTable()
        qtbot.addWidget(t)
        for i in range(5):
            t.add_run(f"Run {i}", _make_result())
        assert t.run_count() == 5

    def test_clear_runs(self, qtbot: QtBot):
        t = _RunsTable()
        qtbot.addWidget(t)
        t.add_run("Run", _make_result())
        t.clear_runs()
        assert t.run_count() == 0

    def test_get_run_returns_tuple(self, qtbot: QtBot):
        t = _RunsTable()
        qtbot.addWidget(t)
        r = _make_result()
        t.add_run("Test", r)
        name, stored = t.get_run(0)
        assert name == "Test"
        assert stored is r

    def test_get_run_invalid_index_returns_none(self, qtbot: QtBot):
        t = _RunsTable()
        qtbot.addWidget(t)
        assert t.get_run(99) is None

    def test_overfitting_shown_in_row(self, qtbot: QtBot):
        t = _RunsTable()
        qtbot.addWidget(t)
        t.add_run("OvRun", _make_result(overfitting=True))
        col = 7   # Overfitting-Spalte
        assert "Ja" in t.table.item(0, col).text()

    def test_no_overfitting_shown_in_row(self, qtbot: QtBot):
        t = _RunsTable()
        qtbot.addWidget(t)
        t.add_run("NormRun", _make_result(overfitting=False))
        col = 7
        assert "Nein" in t.table.item(0, col).text()

    def test_run_selected_signal(self, qtbot: QtBot):
        t = _RunsTable()
        qtbot.addWidget(t)
        t.add_run("R1", _make_result())
        t.add_run("R2", _make_result())
        with qtbot.waitSignal(t.run_selected, timeout=1000) as blocker:
            t.table.selectRow(1)
        assert blocker.args[0] == 1

    def test_table_has_correct_columns(self, qtbot: QtBot):
        t = _RunsTable()
        qtbot.addWidget(t)
        assert t.table.columnCount() == 8

    def test_name_in_first_cell(self, qtbot: QtBot):
        t = _RunsTable()
        qtbot.addWidget(t)
        t.add_run("MyBacktest", _make_result())
        assert t.table.item(0, 0).text() == "MyBacktest"


# ─────────────────────────────────────────────────────────────────────────────
#  _WalkForwardPanel
# ─────────────────────────────────────────────────────────────────────────────

class TestWalkForwardPanel:

    def test_empty_initially(self, qtbot: QtBot):
        p = _WalkForwardPanel()
        qtbot.addWidget(p)
        assert p.window_count() == 0

    def test_add_window_increases_count(self, qtbot: QtBot):
        p = _WalkForwardPanel()
        qtbot.addWidget(p)
        r = _make_result(is_sharpe=1.2, oos_sharpe=0.9)
        p.add_window(1, "2023-01-01", "2023-06-30", "2023-07-01", "2023-12-31", r)
        assert p.window_count() == 1

    def test_multiple_windows(self, qtbot: QtBot):
        p = _WalkForwardPanel()
        qtbot.addWidget(p)
        r = _make_result(is_sharpe=1.0, oos_sharpe=0.8)
        for i in range(3):
            p.add_window(i + 1, "2023-01-01", "2023-06-30", "2023-07-01", "2023-12-31", r)
        assert p.window_count() == 3

    def test_clear_windows(self, qtbot: QtBot):
        p = _WalkForwardPanel()
        qtbot.addWidget(p)
        r = _make_result(is_sharpe=1.0, oos_sharpe=0.8)
        p.add_window(1, "2023-01-01", "2023-06-30", "2023-07-01", "2023-12-31", r)
        p.clear_windows()
        assert p.window_count() == 0

    def test_table_has_correct_column_count(self, qtbot: QtBot):
        p = _WalkForwardPanel()
        qtbot.addWidget(p)
        assert p.table.columnCount() == 8

    def test_overfitting_shown_in_wf_row(self, qtbot: QtBot):
        p = _WalkForwardPanel()
        qtbot.addWidget(p)
        r = _make_result(overfitting=True, is_sharpe=2.0, oos_sharpe=0.3)
        p.add_window(1, "2023-01-01", "2023-06-30", "2023-07-01", "2023-12-31", r)
        ov_col = 7
        assert "Ja" in p.table.item(0, ov_col).text()

    def test_is_sharpe_dash_when_none(self, qtbot: QtBot):
        p = _WalkForwardPanel()
        qtbot.addWidget(p)
        r = _make_result(is_sharpe=None, oos_sharpe=None)
        p.add_window(1, "2023-01-01", "2023-06-30", "2023-07-01", "2023-12-31", r)
        assert p.table.item(0, 5).text() == "–"


# ─────────────────────────────────────────────────────────────────────────────
#  _result_to_markdown
# ─────────────────────────────────────────────────────────────────────────────

class TestResultToMarkdown:

    def test_returns_string(self):
        md = _result_to_markdown(_make_result())
        assert isinstance(md, str)

    def test_contains_heading(self):
        md = _result_to_markdown(_make_result(), name="MeinBacktest")
        assert "# MeinBacktest" in md

    def test_contains_sharpe(self):
        md = _result_to_markdown(_make_result(sharpe=1.234))
        assert "1.234" in md

    def test_contains_total_return(self):
        md = _result_to_markdown(_make_result(total_return=0.15))
        assert "15" in md

    def test_contains_overfitting_warning(self):
        md = _result_to_markdown(_make_result(overfitting=True))
        assert "Ja" in md

    def test_no_overfitting_in_markdown(self):
        md = _result_to_markdown(_make_result(overfitting=False))
        assert "Nein" in md

    def test_infinite_profit_factor_in_md(self):
        md = _result_to_markdown(_make_result(profit_factor=float("inf")))
        assert "∞" in md

    def test_ends_with_newline(self):
        md = _result_to_markdown(_make_result())
        assert md.endswith("\n")


# ─────────────────────────────────────────────────────────────────────────────
#  Integration MainWindow
# ─────────────────────────────────────────────────────────────────────────────

class TestMainWindowIntegration:

    def test_backtest_view_property_accessible(self, qtbot: QtBot, fresh_theme):
        from gui.app import MainWindow, Section
        w = MainWindow(theme_manager=fresh_theme)
        qtbot.addWidget(w)
        assert w.backtest_view is not None

    def test_backtest_view_is_real_view(self, qtbot: QtBot, fresh_theme):
        from gui.app import MainWindow
        w = MainWindow(theme_manager=fresh_theme)
        qtbot.addWidget(w)
        assert isinstance(w.backtest_view, BacktestView)

    def test_navigate_to_backtest(self, qtbot: QtBot, fresh_theme):
        from gui.app import MainWindow, Section
        w = MainWindow(theme_manager=fresh_theme)
        qtbot.addWidget(w)
        w.navigate_to(Section.BACKTEST)
        assert w.content.currentWidget() is w.backtest_view

    def test_stack_count_still_seven(self, qtbot: QtBot, fresh_theme):
        from gui.app import MainWindow
        w = MainWindow(theme_manager=fresh_theme)
        qtbot.addWidget(w)
        assert w.content.count() == 7
