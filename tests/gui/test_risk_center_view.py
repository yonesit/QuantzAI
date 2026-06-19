"""
tests/gui/test_risk_center_view.py
Tests fuer gui/views/risk_center_view.py und MainWindow-Integration.

Abgedeckt:
  - _LossGauge: Werte, Grenzen
  - _DrawdownCanvas: Daten setzen, Zeitstempel-Parsing
  - _HeatmapCanvas: Daten setzen, nested-dict und flat-dict Matrix
  - RiskCenterView: Refresh, Button-Handler, Signale, Bestaetigungs-Dialoge
  - MainWindow: RiskCenterView integriert, Navigation, View-Anzahl
"""

from __future__ import annotations

from unittest.mock import MagicMock, call

import pytest
from PySide6.QtCore import Qt

from gui.app import MainWindow, Section
from gui.views.risk_center_view import (
    RiskCenterView,
    _DrawdownCanvas,
    _HeatmapCanvas,
    _LossGauge,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _confirm_yes(title, msg, label):
    return True

def _confirm_no(title, msg, label):
    return False


def _mock_backend(**overrides):
    backend = MagicMock()
    backend.get_loss_summary.return_value = {
        "daily":   {"loss_pct": 2.0, "limit_pct": 5.0},
        "weekly":  {"loss_pct": 4.0, "limit_pct": 10.0},
        "monthly": {"loss_pct": 8.0, "limit_pct": 20.0},
    }
    backend.get_drawdown_history.return_value = [
        {"timestamp": "2026-01-01T10:00:00", "drawdown_pct": 1.0},
        {"timestamp": "2026-01-01T11:00:00", "drawdown_pct": 3.5},
        {"timestamp": "2026-01-01T12:00:00", "drawdown_pct": 2.0},
    ]
    backend.get_correlation_data.return_value = {
        "symbols":   ["EURUSD", "GBPUSD"],
        "matrix":    {"EURUSD": {"EURUSD": 1.0, "GBPUSD": 0.85},
                      "GBPUSD": {"EURUSD": 0.85, "GBPUSD": 1.0}},
        "threshold": 0.8,
    }
    backend.get_var_cvar.return_value = {
        "var": 0.0312, "cvar": 0.0481, "confidence": 0.95
    }
    backend.get_active_warnings.return_value = [
        {"timestamp": "2026-01-01T09:00:00", "type": "WARN_DAILY_LIMIT", "message": "Taegliches Limit 80% erreicht"},
        {"timestamp": "2026-01-01T09:30:00", "type": "EMERGENCY_STOP",   "message": "Notfall-Stop ausgeloest"},
    ]
    backend.is_trading_paused.return_value  = False
    backend.is_max_drawdown_hit.return_value = False
    backend.emergency_stop.return_value     = {"closed_tickets": [1, 2]}
    for k, v in overrides.items():
        setattr(backend, k, MagicMock(return_value=v) if not callable(v) else v)
    return backend


def _view(**kwargs) -> RiskCenterView:
    return RiskCenterView(**kwargs)


# ─────────────────────────────────────────────────────────────────────────────
#  TestLossGauge
# ─────────────────────────────────────────────────────────────────────────────

class TestLossGauge:
    def test_initial_values(self, qtbot):
        g = _LossGauge("Test")
        qtbot.addWidget(g)
        assert g.loss_pct == 0.0
        assert g.limit_pct == 5.0

    def test_set_values_stored(self, qtbot):
        g = _LossGauge("Test")
        qtbot.addWidget(g)
        g.set_values(2.5, 10.0)
        assert g.loss_pct == pytest.approx(2.5)
        assert g.limit_pct == pytest.approx(10.0)

    def test_loss_pct_clamped_to_zero(self, qtbot):
        g = _LossGauge("Test")
        qtbot.addWidget(g)
        g.set_values(-1.0, 5.0)
        assert g.loss_pct == 0.0

    def test_limit_zero_clamped(self, qtbot):
        g = _LossGauge("Test")
        qtbot.addWidget(g)
        g.set_values(0.0, 0.0)
        assert g.limit_pct == pytest.approx(0.01)

    def test_fixed_height(self, qtbot):
        g = _LossGauge("Test")
        qtbot.addWidget(g)
        assert g.height() == _LossGauge._H

    def test_label_stored(self, qtbot):
        g = _LossGauge("Täglich")
        qtbot.addWidget(g)
        assert g._label == "Täglich"

    def test_update_called_on_set_values(self, qtbot):
        g = _LossGauge("Test")
        qtbot.addWidget(g)
        g.show()
        g.set_values(3.0, 6.0)
        assert g.loss_pct == pytest.approx(3.0)


# ─────────────────────────────────────────────────────────────────────────────
#  TestDrawdownCanvas
# ─────────────────────────────────────────────────────────────────────────────

class TestDrawdownCanvas:
    def test_initial_empty(self, qtbot):
        c = _DrawdownCanvas()
        qtbot.addWidget(c)
        assert c.point_count == 0

    def test_set_data_empty_list(self, qtbot):
        c = _DrawdownCanvas()
        qtbot.addWidget(c)
        c.set_data([])
        assert c.point_count == 0

    def test_set_data_stores_points(self, qtbot):
        c = _DrawdownCanvas()
        qtbot.addWidget(c)
        c.set_data([
            {"timestamp": "2026-01-01T10:00:00", "drawdown_pct": 2.0},
            {"timestamp": "2026-01-01T11:00:00", "drawdown_pct": 5.0},
        ])
        assert c.point_count == 2

    def test_set_data_string_timestamps(self, qtbot):
        c = _DrawdownCanvas()
        qtbot.addWidget(c)
        c.set_data([
            {"timestamp": "2026-06-01T08:00:00", "drawdown_pct": 1.0},
            {"timestamp": "2026-06-01T09:00:00", "drawdown_pct": 3.0},
        ])
        assert c.point_count == 2
        # First point t_norm = 0, last = 1
        assert c._points[0][0] == pytest.approx(0.0)
        assert c._points[-1][0] == pytest.approx(1.0)

    def test_set_data_single_point(self, qtbot):
        c = _DrawdownCanvas()
        qtbot.addWidget(c)
        c.set_data([{"timestamp": "2026-01-01T10:00:00", "drawdown_pct": 7.0}])
        assert c.point_count == 1

    def test_max_dd_stored(self, qtbot):
        c = _DrawdownCanvas()
        qtbot.addWidget(c)
        c.set_data([], max_drawdown_pct=20.0)
        assert c._max_dd == pytest.approx(20.0)

    def test_max_dd_clamped_to_1(self, qtbot):
        c = _DrawdownCanvas()
        qtbot.addWidget(c)
        c.set_data([], max_drawdown_pct=0.0)
        assert c._max_dd == pytest.approx(1.0)

    def test_set_data_datetime_objects(self, qtbot):
        from datetime import datetime, timezone
        c = _DrawdownCanvas()
        qtbot.addWidget(c)
        c.set_data([
            {"timestamp": datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc), "drawdown_pct": 2.0},
            {"timestamp": datetime(2026, 1, 1, 11, 0, tzinfo=timezone.utc), "drawdown_pct": 4.0},
        ])
        assert c.point_count == 2

    def test_paint_does_not_crash(self, qtbot):
        c = _DrawdownCanvas()
        qtbot.addWidget(c)
        c.set_data([
            {"timestamp": "2026-01-01T10:00:00", "drawdown_pct": 2.0},
            {"timestamp": "2026-01-01T11:00:00", "drawdown_pct": 5.0},
        ])
        c.show()
        c.resize(400, 200)
        qtbot.waitExposed(c)

    def test_paint_empty_data_no_crash(self, qtbot):
        c = _DrawdownCanvas()
        qtbot.addWidget(c)
        c.show()
        c.resize(400, 200)
        qtbot.waitExposed(c)


# ─────────────────────────────────────────────────────────────────────────────
#  TestHeatmapCanvas
# ─────────────────────────────────────────────────────────────────────────────

class TestHeatmapCanvas:
    def test_initial_empty(self, qtbot):
        c = _HeatmapCanvas()
        qtbot.addWidget(c)
        assert c.symbols == []

    def test_set_data_symbols_stored(self, qtbot):
        c = _HeatmapCanvas()
        qtbot.addWidget(c)
        c.set_data(["EURUSD", "GBPUSD"], {}, 0.8)
        assert c.symbols == ["EURUSD", "GBPUSD"]

    def test_threshold_stored(self, qtbot):
        c = _HeatmapCanvas()
        qtbot.addWidget(c)
        c.set_data(["A"], {}, 0.75)
        assert c.threshold == pytest.approx(0.75)

    def test_diagonal_is_1(self, qtbot):
        c = _HeatmapCanvas()
        qtbot.addWidget(c)
        c.set_data(["EURUSD", "GBPUSD"], {}, 0.8)
        assert c.get_correlation("EURUSD", "EURUSD") == pytest.approx(1.0)
        assert c.get_correlation("GBPUSD", "GBPUSD") == pytest.approx(1.0)

    def test_nested_dict_matrix(self, qtbot):
        c = _HeatmapCanvas()
        qtbot.addWidget(c)
        matrix = {
            "EURUSD": {"EURUSD": 1.0, "GBPUSD": 0.85},
            "GBPUSD": {"EURUSD": 0.85, "GBPUSD": 1.0},
        }
        c.set_data(["EURUSD", "GBPUSD"], matrix, 0.8)
        assert c.get_correlation("EURUSD", "GBPUSD") == pytest.approx(0.85)
        assert c.get_correlation("GBPUSD", "EURUSD") == pytest.approx(0.85)

    def test_flat_tuple_dict_matrix(self, qtbot):
        c = _HeatmapCanvas()
        qtbot.addWidget(c)
        matrix = {("EURUSD", "GBPUSD"): 0.72, ("GBPUSD", "EURUSD"): 0.72}
        c.set_data(["EURUSD", "GBPUSD"], matrix, 0.8)
        assert c.get_correlation("EURUSD", "GBPUSD") == pytest.approx(0.72)

    def test_unknown_pair_defaults_to_zero(self, qtbot):
        c = _HeatmapCanvas()
        qtbot.addWidget(c)
        c.set_data(["EURUSD", "GBPUSD"], {}, 0.8)
        assert c.get_correlation("EURUSD", "GBPUSD") == pytest.approx(0.0)

    def test_paint_no_symbols_no_crash(self, qtbot):
        c = _HeatmapCanvas()
        qtbot.addWidget(c)
        c.show()
        c.resize(300, 200)
        qtbot.waitExposed(c)

    def test_paint_with_data_no_crash(self, qtbot):
        c = _HeatmapCanvas()
        qtbot.addWidget(c)
        matrix = {
            "EURUSD": {"EURUSD": 1.0, "GBPUSD": 0.9},
            "GBPUSD": {"EURUSD": 0.9, "GBPUSD": 1.0},
        }
        c.set_data(["EURUSD", "GBPUSD"], matrix, 0.8)
        c.show()
        c.resize(300, 200)
        qtbot.waitExposed(c)


# ─────────────────────────────────────────────────────────────────────────────
#  TestRiskCenterView
# ─────────────────────────────────────────────────────────────────────────────

class TestRiskCenterView:
    def test_initial_widgets_present(self, qtbot):
        v = _view()
        qtbot.addWidget(v)
        assert v.gauge_day   is not None
        assert v.gauge_week  is not None
        assert v.gauge_month is not None
        assert v.drawdown_canvas  is not None
        assert v.heatmap_canvas   is not None
        assert v.var_label        is not None
        assert v.cvar_label       is not None
        assert v.warnings_table   is not None
        assert v.pause_button     is not None
        assert v.emergency_button is not None
        assert v.release_button   is not None

    def test_pause_button_checkable(self, qtbot):
        v = _view()
        qtbot.addWidget(v)
        assert v.pause_button.isCheckable()

    def test_release_button_disabled_initially(self, qtbot):
        v = _view()
        qtbot.addWidget(v)
        assert not v.release_button.isEnabled()

    def test_set_backend(self, qtbot):
        v = _view()
        qtbot.addWidget(v)
        b = _mock_backend()
        v.set_backend(b)
        assert v._backend is b

    def test_refresh_noop_without_backend(self, qtbot):
        v = _view()
        qtbot.addWidget(v)
        v.refresh()  # must not raise

    # ── refresh_loss_limits ────────────────────────────────────────────────

    def test_refresh_loss_limits_day(self, qtbot):
        v = _view(backend=_mock_backend())
        qtbot.addWidget(v)
        v.refresh()
        assert v.gauge_day.loss_pct  == pytest.approx(2.0)
        assert v.gauge_day.limit_pct == pytest.approx(5.0)

    def test_refresh_loss_limits_week(self, qtbot):
        v = _view(backend=_mock_backend())
        qtbot.addWidget(v)
        v.refresh()
        assert v.gauge_week.loss_pct  == pytest.approx(4.0)
        assert v.gauge_week.limit_pct == pytest.approx(10.0)

    def test_refresh_loss_limits_month(self, qtbot):
        v = _view(backend=_mock_backend())
        qtbot.addWidget(v)
        v.refresh()
        assert v.gauge_month.loss_pct  == pytest.approx(8.0)
        assert v.gauge_month.limit_pct == pytest.approx(20.0)

    # ── refresh_drawdown ───────────────────────────────────────────────────

    def test_refresh_drawdown_loads_data(self, qtbot):
        v = _view(backend=_mock_backend())
        qtbot.addWidget(v)
        v.refresh()
        assert v.drawdown_canvas.point_count == 3

    def test_refresh_drawdown_empty(self, qtbot):
        b = _mock_backend()
        b.get_drawdown_history.return_value = []
        v = _view(backend=b)
        qtbot.addWidget(v)
        v.refresh()
        assert v.drawdown_canvas.point_count == 0

    # ── refresh_correlation ────────────────────────────────────────────────

    def test_refresh_correlation_symbols(self, qtbot):
        v = _view(backend=_mock_backend())
        qtbot.addWidget(v)
        v.refresh()
        assert "EURUSD" in v.heatmap_canvas.symbols
        assert "GBPUSD" in v.heatmap_canvas.symbols

    def test_refresh_correlation_threshold(self, qtbot):
        v = _view(backend=_mock_backend())
        qtbot.addWidget(v)
        v.refresh()
        assert v.heatmap_canvas.threshold == pytest.approx(0.8)

    def test_refresh_correlation_values(self, qtbot):
        v = _view(backend=_mock_backend())
        qtbot.addWidget(v)
        v.refresh()
        assert v.heatmap_canvas.get_correlation("EURUSD", "GBPUSD") == pytest.approx(0.85)

    # ── refresh_var_cvar ───────────────────────────────────────────────────

    def test_refresh_var_label(self, qtbot):
        v = _view(backend=_mock_backend())
        qtbot.addWidget(v)
        v.refresh()
        assert "VaR" in v.var_label.text()
        assert "95" in v.var_label.text()
        assert "0.0312" in v.var_label.text()

    def test_refresh_cvar_label(self, qtbot):
        v = _view(backend=_mock_backend())
        qtbot.addWidget(v)
        v.refresh()
        assert "CVaR" in v.cvar_label.text()
        assert "0.0481" in v.cvar_label.text()

    def test_refresh_var_conf_label(self, qtbot):
        v = _view(backend=_mock_backend())
        qtbot.addWidget(v)
        v.refresh()
        assert "95" in v.var_conf_label.text()

    def test_refresh_var_custom_confidence(self, qtbot):
        b = _mock_backend()
        b.get_var_cvar.return_value = {"var": 0.05, "cvar": 0.07, "confidence": 0.99}
        v = _view(backend=b)
        qtbot.addWidget(v)
        v.refresh()
        assert "99" in v.var_label.text()

    # ── refresh_warnings ──────────────────────────────────────────────────

    def test_refresh_warnings_row_count(self, qtbot):
        v = _view(backend=_mock_backend())
        qtbot.addWidget(v)
        v.refresh()
        assert v.warnings_table.rowCount() == 2

    def test_refresh_warnings_content(self, qtbot):
        v = _view(backend=_mock_backend())
        qtbot.addWidget(v)
        v.refresh()
        types = [v.warnings_table.item(r, 1).text()
                 for r in range(v.warnings_table.rowCount())]
        assert "WARN_DAILY_LIMIT" in types
        assert "EMERGENCY_STOP"   in types

    def test_refresh_warnings_emergency_color_red(self, qtbot):
        v = _view(backend=_mock_backend())
        qtbot.addWidget(v)
        v.refresh()
        # Find EMERGENCY_STOP row
        for r in range(v.warnings_table.rowCount()):
            if v.warnings_table.item(r, 1).text() == "EMERGENCY_STOP":
                color = v.warnings_table.item(r, 1).foreground().color()
                assert color.red() > 200
                assert color.green() < 100
                break

    def test_refresh_warnings_warn_color_amber(self, qtbot):
        v = _view(backend=_mock_backend())
        qtbot.addWidget(v)
        v.refresh()
        for r in range(v.warnings_table.rowCount()):
            if v.warnings_table.item(r, 1).text() == "WARN_DAILY_LIMIT":
                color = v.warnings_table.item(r, 1).foreground().color()
                assert color.red() > 200
                assert color.blue() < 100
                break

    def test_refresh_warnings_empty(self, qtbot):
        b = _mock_backend()
        b.get_active_warnings.return_value = []
        v = _view(backend=b)
        qtbot.addWidget(v)
        v.refresh()
        assert v.warnings_table.rowCount() == 0

    # ── refresh_button_states ─────────────────────────────────────────────

    def test_pause_button_not_checked_when_not_paused(self, qtbot):
        b = _mock_backend()
        b.is_trading_paused.return_value = False
        v = _view(backend=b)
        qtbot.addWidget(v)
        v.refresh()
        assert not v.pause_button.isChecked()
        assert "Pause" in v.pause_button.text()

    def test_pause_button_checked_when_paused(self, qtbot):
        b = _mock_backend()
        b.is_trading_paused.return_value = True
        v = _view(backend=b)
        qtbot.addWidget(v)
        v.refresh()
        assert v.pause_button.isChecked()
        assert "Resume" in v.pause_button.text()

    def test_release_button_enabled_when_drawdown_hit(self, qtbot):
        b = _mock_backend()
        b.is_max_drawdown_hit.return_value = True
        v = _view(backend=b)
        qtbot.addWidget(v)
        v.refresh()
        assert v.release_button.isEnabled()

    def test_release_button_disabled_when_no_drawdown(self, qtbot):
        b = _mock_backend()
        b.is_max_drawdown_hit.return_value = False
        v = _view(backend=b)
        qtbot.addWidget(v)
        v.refresh()
        assert not v.release_button.isEnabled()

    # ── Pause / Resume ────────────────────────────────────────────────────

    def test_pause_calls_backend_pause(self, qtbot):
        b = _mock_backend()
        v = _view(backend=b, _confirm_fn=_confirm_yes)
        qtbot.addWidget(v)
        # Start unchecked, click() toggles to checked -> calls _on_pause_resume(True)
        v.pause_button.setChecked(False)
        v.pause_button.click()
        b.pause_trading.assert_called_once()

    def test_pause_emits_signal_true(self, qtbot):
        b = _mock_backend()
        v = _view(backend=b, _confirm_fn=_confirm_yes)
        qtbot.addWidget(v)
        signals = []
        v.trading_paused.connect(signals.append)
        v.pause_button.setChecked(False)
        v.pause_button.click()
        assert signals == [True]

    def test_pause_changes_button_text(self, qtbot):
        b = _mock_backend()
        v = _view(backend=b, _confirm_fn=_confirm_yes)
        qtbot.addWidget(v)
        v.pause_button.setChecked(False)
        v.pause_button.click()
        assert "Resume" in v.pause_button.text()

    def test_resume_confirmed_calls_backend(self, qtbot):
        b = _mock_backend()
        v = _view(backend=b, _confirm_fn=_confirm_yes)
        qtbot.addWidget(v)
        # Start checked (paused), click() toggles to unchecked -> calls _on_pause_resume(False)
        v.pause_button.setChecked(True)
        v.pause_button.click()
        b.resume_trading.assert_called_once()

    def test_resume_confirmed_emits_signal_false(self, qtbot):
        b = _mock_backend()
        v = _view(backend=b, _confirm_fn=_confirm_yes)
        qtbot.addWidget(v)
        signals = []
        v.trading_paused.connect(signals.append)
        v.pause_button.setChecked(True)
        v.pause_button.click()
        assert signals == [False]

    def test_resume_cancelled_stays_paused(self, qtbot):
        b = _mock_backend()
        v = _view(backend=b, _confirm_fn=_confirm_no)
        qtbot.addWidget(v)
        v.pause_button.setChecked(True)
        v.pause_button.click()
        # Should revert to checked (paused)
        assert v.pause_button.isChecked()
        b.resume_trading.assert_not_called()

    def test_resume_cancelled_no_signal(self, qtbot):
        b = _mock_backend()
        v = _view(backend=b, _confirm_fn=_confirm_no)
        qtbot.addWidget(v)
        signals = []
        v.trading_paused.connect(signals.append)
        v.pause_button.setChecked(True)
        v.pause_button.click()
        assert signals == []

    # ── Emergency Stop ────────────────────────────────────────────────────

    def test_emergency_stop_confirmed_calls_backend(self, qtbot):
        b = _mock_backend()
        v = _view(backend=b, _confirm_fn=_confirm_yes)
        qtbot.addWidget(v)
        v.emergency_button.click()
        b.emergency_stop.assert_called_once()

    def test_emergency_stop_emits_signal(self, qtbot):
        b = _mock_backend()
        v = _view(backend=b, _confirm_fn=_confirm_yes)
        qtbot.addWidget(v)
        signals = []
        v.emergency_stopped.connect(lambda: signals.append(True))
        v.emergency_button.click()
        assert signals == [True]

    def test_emergency_stop_sets_pause_button(self, qtbot):
        b = _mock_backend()
        v = _view(backend=b, _confirm_fn=_confirm_yes)
        qtbot.addWidget(v)
        v.emergency_button.click()
        assert v.pause_button.isChecked()
        assert "Resume" in v.pause_button.text()

    def test_emergency_stop_cancelled_no_call(self, qtbot):
        b = _mock_backend()
        v = _view(backend=b, _confirm_fn=_confirm_no)
        qtbot.addWidget(v)
        v.emergency_button.click()
        b.emergency_stop.assert_not_called()

    def test_emergency_stop_cancelled_no_signal(self, qtbot):
        b = _mock_backend()
        v = _view(backend=b, _confirm_fn=_confirm_no)
        qtbot.addWidget(v)
        signals = []
        v.emergency_stopped.connect(lambda: signals.append(True))
        v.emergency_button.click()
        assert signals == []

    def test_emergency_stop_no_backend_no_crash(self, qtbot):
        v = _view(_confirm_fn=_confirm_yes)
        qtbot.addWidget(v)
        v.emergency_button.click()  # must not raise

    # ── Release Drawdown ──────────────────────────────────────────────────

    def test_release_drawdown_confirmed_calls_backend(self, qtbot):
        b = _mock_backend()
        b.is_max_drawdown_hit.return_value = True
        v = _view(backend=b, _confirm_fn=_confirm_yes)
        qtbot.addWidget(v)
        v.refresh()
        v.release_button.click()
        b.release_drawdown_stop.assert_called_once()

    def test_release_drawdown_emits_signal(self, qtbot):
        b = _mock_backend()
        b.is_max_drawdown_hit.return_value = True
        v = _view(backend=b, _confirm_fn=_confirm_yes)
        qtbot.addWidget(v)
        v.refresh()
        signals = []
        v.drawdown_released.connect(lambda: signals.append(True))
        v.release_button.click()
        assert signals == [True]

    def test_release_drawdown_disables_button(self, qtbot):
        b = _mock_backend()
        b.is_max_drawdown_hit.return_value = True
        v = _view(backend=b, _confirm_fn=_confirm_yes)
        qtbot.addWidget(v)
        v.refresh()
        v.release_button.click()
        assert not v.release_button.isEnabled()

    def test_release_drawdown_cancelled_no_call(self, qtbot):
        b = _mock_backend()
        b.is_max_drawdown_hit.return_value = True
        v = _view(backend=b, _confirm_fn=_confirm_no)
        qtbot.addWidget(v)
        v.refresh()
        v.release_button.click()
        b.release_drawdown_stop.assert_not_called()

    def test_release_drawdown_cancelled_button_still_enabled(self, qtbot):
        b = _mock_backend()
        b.is_max_drawdown_hit.return_value = True
        v = _view(backend=b, _confirm_fn=_confirm_no)
        qtbot.addWidget(v)
        v.refresh()
        v.release_button.click()
        assert v.release_button.isEnabled()

    def test_release_drawdown_no_backend_no_crash(self, qtbot):
        v = _view(_confirm_fn=_confirm_yes)
        qtbot.addWidget(v)
        v._release_btn.setEnabled(True)
        v.release_button.click()  # must not raise

    # ── Signals ──────────────────────────────────────────────────────────

    def test_trading_paused_signal_type(self, qtbot):
        b = _mock_backend()
        v = _view(backend=b, _confirm_fn=_confirm_yes)
        qtbot.addWidget(v)
        with qtbot.waitSignal(v.trading_paused, timeout=1000):
            v.pause_button.setChecked(False)
            v.pause_button.click()

    def test_emergency_stopped_signal_type(self, qtbot):
        b = _mock_backend()
        v = _view(backend=b, _confirm_fn=_confirm_yes)
        qtbot.addWidget(v)
        with qtbot.waitSignal(v.emergency_stopped, timeout=1000):
            v.emergency_button.click()

    def test_drawdown_released_signal_type(self, qtbot):
        b = _mock_backend()
        b.is_max_drawdown_hit.return_value = True
        v = _view(backend=b, _confirm_fn=_confirm_yes)
        qtbot.addWidget(v)
        v.refresh()
        with qtbot.waitSignal(v.drawdown_released, timeout=1000):
            v.release_button.click()

    # ── Warnings table columns ────────────────────────────────────────────

    def test_warnings_table_has_3_columns(self, qtbot):
        v = _view()
        qtbot.addWidget(v)
        assert v.warnings_table.columnCount() == 3

    def test_warnings_table_headers(self, qtbot):
        v = _view()
        qtbot.addWidget(v)
        headers = [v.warnings_table.horizontalHeaderItem(i).text()
                   for i in range(3)]
        assert headers[0] == "Zeitstempel"
        assert headers[1] == "Typ"
        assert headers[2] == "Meldung"

    def test_warnings_table_not_editable(self, qtbot):
        v = _view()
        qtbot.addWidget(v)
        from PySide6.QtWidgets import QAbstractItemView
        assert v.warnings_table.editTriggers() == QAbstractItemView.EditTrigger.NoEditTriggers


# ─────────────────────────────────────────────────────────────────────────────
#  TestMainWindowRiskIntegration
# ─────────────────────────────────────────────────────────────────────────────

class TestMainWindowRiskIntegration:
    @pytest.fixture
    def fresh_window(self, qtbot, fresh_theme):
        win = MainWindow(theme_manager=fresh_theme)
        qtbot.addWidget(win)
        return win

    def test_still_has_six_views_in_stack(self, fresh_window):
        assert fresh_window.content.count() == 6

    def test_risk_center_view_property_exists(self, fresh_window):
        view = fresh_window.risk_center_view
        from gui.views.risk_center_view import RiskCenterView
        assert isinstance(view, RiskCenterView)

    def test_risk_center_view_registered(self, fresh_window):
        assert Section.RISK in fresh_window._views

    def test_navigate_to_risk_switches_view(self, fresh_window):
        fresh_window.navigate_to(Section.RISK)
        assert fresh_window.current_view() is fresh_window.risk_center_view

    def test_risk_backend_passed_to_view(self, qtbot, fresh_theme):
        b = _mock_backend()
        win = MainWindow(theme_manager=fresh_theme, risk_center_backend=b)
        qtbot.addWidget(win)
        assert win.risk_center_view._backend is b

    def test_risk_center_view_objectname(self, fresh_window):
        assert fresh_window.risk_center_view.objectName() == "risk_center_view"
