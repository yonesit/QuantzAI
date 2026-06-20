"""
tests/gui/test_activity_log_widget.py
pytest-qt Tests fuer gui/widgets/activity_log_widget.py
und die Integration in TradingOrchestrator / BotControlsWidget / MainWindow.

Abgedeckt:
  CycleLogEntry
    - Erstellung (dataclass)
    - from_cycle_result: alle Felder, fehlende Keys, Checks
    - category-Property: TRADE / SUGGESTED / NEUTRAL / REJECTION / EMERGENCY

  LogFilter
    - Enum-Werte vorhanden

  ActivityLogWidget
    - Erstellung ohne Absturz
    - Initial leer
    - append_cycle_result fuegt Eintrag hinzu
    - entry_appended Signal emittiert
    - Ring-Puffer: max_entries begrenzt Groesse
    - clear() leert Puffer und Tabelle
    - Filter ALL / TRADES / REJECTIONS
    - set_filter() aendert Filter
    - Tabellenzeilen entsprechen gefilterten Eintraegen
    - Tabelleninhalt (Symbol, Signal, Aktion, Checks)
    - Farbkodierung per Kategorie
    - Confidence-Anzeige im Signal-Feld

  TradingOrchestrator-Integration
    - run_cycle() liefert checks-Liste
    - run_cycle() liefert timestamp
    - run_cycle() liefert confidence wenn Modell vorhanden
    - RiskGuard-Check wird in checks gefuehrt (passed/failed)
    - PreTradeCheck-Check wird in checks gefuehrt
    - CorrelationGuard-Check wird in checks gefuehrt
    - set_activity_callback() setzt Callback
    - Callback wird in run_loop() nach jedem Zyklus aufgerufen

  BotWorker-Integration
    - cycle_completed Signal existiert
    - Callback wird auf Orchestrator gesetzt in run()
    - cycle_completed wird emittiert

  BotControlsWidget-Integration
    - cycle_completed Signal existiert
    - cycle_completed wird von BotWorker weitergeleitet

  MainWindow-Integration
    - activity_log Property vorhanden
    - activity_log mit bot_controls.cycle_completed verbunden
    - Eintrag kommt im activity_log an
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from pytestqt.qtbot import QtBot

from gui.app import MainWindow
from gui.design.theme import ThemeManager, ThemeMode
from gui.widgets.activity_log_widget import (
    ActivityLogWidget,
    CheckResult,
    CycleLogEntry,
    LogFilter,
    _CATEGORY_COLORS,
)
from gui.widgets.bot_controls_widget import BotControlsWidget, BotWorker


# ─────────────────────────────────────────────────────────────────────────────
#  Hilfsfunktionen / Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_result(
    symbol="EURUSD",
    signal="long",
    action="open_buy",
    reason="signal_executed",
    checks=None,
    confidence=0.72,
    ticket=42,
    lot_size=0.1,
    step_stopped_at=None,
) -> dict:
    return {
        "symbol":          symbol,
        "signal":          signal,
        "confidence":      confidence,
        "action":          action,
        "reason":          reason,
        "ticket":          ticket,
        "lot_size":        lot_size,
        "step_stopped_at": step_stopped_at,
        "checks":          checks or [],
        "timestamp":       datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
    }


def _trade_result(**kw):
    return _make_result(signal="long", action="open_buy", **kw)


def _rejection_result(**kw):
    defaults = dict(
        signal=None, action="skipped",
        reason="risk_guard_blocked",
        checks=[{"name": "RiskGuard", "passed": False, "reason": "Handelssperre"}],
        confidence=None, ticket=None, lot_size=None,
    )
    defaults.update(kw)
    return _make_result(**defaults)


def _neutral_result(**kw):
    defaults = dict(signal="flat", action="flat", reason="signal_flat",
                    confidence=0.50, ticket=None, lot_size=None)
    defaults.update(kw)
    return _make_result(**defaults)


@pytest.fixture
def widget(qtbot: QtBot) -> ActivityLogWidget:
    w = ActivityLogWidget()
    qtbot.addWidget(w)
    return w


@pytest.fixture
def widget_small(qtbot: QtBot) -> ActivityLogWidget:
    w = ActivityLogWidget(max_entries=5)
    qtbot.addWidget(w)
    return w


@pytest.fixture
def theme() -> ThemeManager:
    return ThemeManager(mode=ThemeMode.DARK)


@pytest.fixture
def main_window(qtbot: QtBot, theme: ThemeManager) -> MainWindow:
    w = MainWindow(theme_manager=theme)
    qtbot.addWidget(w)
    return w


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: CycleLogEntry
# ─────────────────────────────────────────────────────────────────────────────

class TestCycleLogEntry:

    def test_create_minimal(self):
        now = datetime.now(timezone.utc)
        e = CycleLogEntry(
            timestamp=now, symbol="EURUSD",
            signal="long", action="open_buy", reason="", checks=[],
        )
        assert e.symbol == "EURUSD"
        assert e.action == "open_buy"

    def test_from_cycle_result_basic(self):
        r = _trade_result()
        e = CycleLogEntry.from_cycle_result(r)
        assert e.symbol == "EURUSD"
        assert e.signal == "long"
        assert e.action == "open_buy"
        assert e.confidence == pytest.approx(0.72)
        assert e.ticket == 42
        assert e.lot_size == pytest.approx(0.1)

    def test_from_cycle_result_timestamp_preserved(self):
        r = _trade_result()
        e = CycleLogEntry.from_cycle_result(r)
        assert e.timestamp.year == 2024
        assert e.timestamp.hour == 10

    def test_from_cycle_result_no_timestamp_uses_now(self):
        r = _trade_result()
        del r["timestamp"]
        before = datetime.now(timezone.utc)
        e = CycleLogEntry.from_cycle_result(r)
        assert e.timestamp >= before

    def test_from_cycle_result_checks_parsed(self):
        r = _make_result(checks=[
            {"name": "RiskGuard",    "passed": True,  "reason": ""},
            {"name": "PreTradeCheck","passed": False, "reason": "Spread hoch"},
        ])
        e = CycleLogEntry.from_cycle_result(r)
        assert len(e.checks) == 2
        assert e.checks[0].name == "RiskGuard"
        assert e.checks[0].passed is True
        assert e.checks[1].name == "PreTradeCheck"
        assert e.checks[1].passed is False
        assert e.checks[1].reason == "Spread hoch"

    def test_from_cycle_result_missing_checks_defaults_empty(self):
        r = {"symbol": "GBPUSD", "action": "skipped", "reason": "risk_guard_blocked"}
        e = CycleLogEntry.from_cycle_result(r)
        assert e.checks == []

    def test_from_cycle_result_missing_confidence_is_none(self):
        r = {"symbol": "GBPUSD", "action": "skipped", "reason": ""}
        e = CycleLogEntry.from_cycle_result(r)
        assert e.confidence is None

    def test_category_trade(self):
        e = CycleLogEntry.from_cycle_result(_make_result(action="open_buy"))
        assert e.category == "TRADE"

    def test_category_trade_sell(self):
        e = CycleLogEntry.from_cycle_result(_make_result(action="open_sell"))
        assert e.category == "TRADE"

    def test_category_suggested(self):
        e = CycleLogEntry.from_cycle_result(_make_result(action="suggested", reason="suggest_only_mode"))
        assert e.category == "SUGGESTED"

    def test_category_neutral_flat_action(self):
        e = CycleLogEntry.from_cycle_result(_neutral_result())
        assert e.category == "NEUTRAL"

    def test_category_neutral_signal_flat_reason(self):
        e = CycleLogEntry.from_cycle_result(_make_result(action="flat", reason="signal_flat"))
        assert e.category == "NEUTRAL"

    def test_category_rejection_skipped(self):
        e = CycleLogEntry.from_cycle_result(_rejection_result())
        assert e.category == "REJECTION"

    def test_category_emergency(self):
        e = CycleLogEntry.from_cycle_result(_make_result(action="skipped", reason="emergency_stop"))
        assert e.category == "EMERGENCY"

    def test_naive_timestamp_gets_utc_tzinfo(self):
        r = _trade_result()
        r["timestamp"] = datetime(2024, 6, 1, 8, 0, 0)  # naive
        e = CycleLogEntry.from_cycle_result(r)
        assert e.timestamp.tzinfo is not None


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: LogFilter
# ─────────────────────────────────────────────────────────────────────────────

class TestLogFilter:

    def test_all_exists(self):
        assert LogFilter.ALL.value == "Alle"

    def test_trades_exists(self):
        assert LogFilter.TRADES.value == "Nur Trades"

    def test_rejections_exists(self):
        assert LogFilter.REJECTIONS.value == "Nur Ablehnungen"

    def test_three_values(self):
        assert len(LogFilter) == 3


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: ActivityLogWidget – Erstellung und Initialzustand
# ─────────────────────────────────────────────────────────────────────────────

class TestActivityLogWidgetInit:

    def test_creates_without_crash(self, widget: ActivityLogWidget):
        assert widget is not None

    def test_initial_entry_count_zero(self, widget: ActivityLogWidget):
        assert widget.entry_count == 0

    def test_initial_filter_all(self, widget: ActivityLogWidget):
        assert widget.current_filter == LogFilter.ALL

    def test_table_has_five_columns(self, widget: ActivityLogWidget):
        assert widget._table.columnCount() == 5

    def test_table_initially_empty(self, widget: ActivityLogWidget):
        assert widget._table.rowCount() == 0

    def test_filter_combo_has_three_items(self, widget: ActivityLogWidget):
        assert widget._filter_combo.count() == 3

    def test_object_name_set(self, widget: ActivityLogWidget):
        assert widget.objectName() == "activity_log_widget"

    def test_custom_max_entries(self, qtbot: QtBot):
        w = ActivityLogWidget(max_entries=10)
        qtbot.addWidget(w)
        assert w._max_entries == 10


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: ActivityLogWidget – Eintraege hinzufuegen
# ─────────────────────────────────────────────────────────────────────────────

class TestActivityLogWidgetAppend:

    def test_append_increases_count(self, qtbot: QtBot, widget: ActivityLogWidget):
        with qtbot.waitSignal(widget.entry_appended, timeout=2000):
            widget.append_cycle_result(_trade_result())
        assert widget.entry_count == 1

    def test_entry_appended_signal_emitted(self, qtbot: QtBot, widget: ActivityLogWidget):
        received: list = []
        widget.entry_appended.connect(received.append)
        with qtbot.waitSignal(widget.entry_appended, timeout=2000):
            widget.append_cycle_result(_trade_result())
        assert len(received) == 1
        assert isinstance(received[0], CycleLogEntry)

    def test_entry_symbol_correct(self, qtbot: QtBot, widget: ActivityLogWidget):
        received: list = []
        widget.entry_appended.connect(received.append)
        with qtbot.waitSignal(widget.entry_appended, timeout=2000):
            widget.append_cycle_result(_trade_result(symbol="GBPUSD"))
        assert received[0].symbol == "GBPUSD"

    def test_table_row_added(self, qtbot: QtBot, widget: ActivityLogWidget):
        with qtbot.waitSignal(widget.entry_appended, timeout=2000):
            widget.append_cycle_result(_trade_result())
        assert widget._table.rowCount() == 1

    def test_multiple_entries(self, qtbot: QtBot, widget: ActivityLogWidget):
        for _ in range(3):
            with qtbot.waitSignal(widget.entry_appended, timeout=2000):
                widget.append_cycle_result(_trade_result())
        assert widget.entry_count == 3
        assert widget._table.rowCount() == 3

    def test_entries_returns_list(self, qtbot: QtBot, widget: ActivityLogWidget):
        with qtbot.waitSignal(widget.entry_appended, timeout=2000):
            widget.append_cycle_result(_trade_result())
        lst = widget.entries()
        assert isinstance(lst, list)
        assert len(lst) == 1


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: Ring-Puffer
# ─────────────────────────────────────────────────────────────────────────────

class TestActivityLogRingBuffer:

    def test_buffer_capped_at_max_entries(self, qtbot: QtBot, widget_small: ActivityLogWidget):
        for i in range(8):
            with qtbot.waitSignal(widget_small.entry_appended, timeout=2000):
                widget_small.append_cycle_result(_trade_result(symbol=f"SYM{i}"))
        assert widget_small.entry_count == 5  # maxlen=5

    def test_oldest_entries_dropped(self, qtbot: QtBot, widget_small: ActivityLogWidget):
        for i in range(7):
            with qtbot.waitSignal(widget_small.entry_appended, timeout=2000):
                widget_small.append_cycle_result(_trade_result(symbol=f"SYM{i}"))
        symbols = [e.symbol for e in widget_small.entries()]
        assert "SYM0" not in symbols  # wurde herausgedraengt
        assert "SYM6" in symbols

    def test_clear_empties_buffer(self, qtbot: QtBot, widget: ActivityLogWidget):
        with qtbot.waitSignal(widget.entry_appended, timeout=2000):
            widget.append_cycle_result(_trade_result())
        widget.clear()
        assert widget.entry_count == 0
        assert widget._table.rowCount() == 0


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: Filter
# ─────────────────────────────────────────────────────────────────────────────

class TestActivityLogWidgetFilter:

    def _fill(self, qtbot: QtBot, widget: ActivityLogWidget) -> None:
        for r in [_trade_result(), _rejection_result(), _neutral_result(), _make_result(signal="short", action="open_sell")]:
            with qtbot.waitSignal(widget.entry_appended, timeout=2000):
                widget.append_cycle_result(r)

    def test_filter_all_shows_all(self, qtbot: QtBot, widget: ActivityLogWidget):
        self._fill(qtbot, widget)
        widget.set_filter(LogFilter.ALL)
        assert widget._table.rowCount() == 4

    def test_filter_trades_shows_only_trades(self, qtbot: QtBot, widget: ActivityLogWidget):
        self._fill(qtbot, widget)
        widget.set_filter(LogFilter.TRADES)
        assert widget._table.rowCount() == 2   # open_buy + open_sell

    def test_filter_rejections_shows_only_rejections(self, qtbot: QtBot, widget: ActivityLogWidget):
        self._fill(qtbot, widget)
        widget.set_filter(LogFilter.REJECTIONS)
        assert widget._table.rowCount() == 1

    def test_set_filter_updates_current_filter(self, qtbot: QtBot, widget: ActivityLogWidget):
        widget.set_filter(LogFilter.TRADES)
        assert widget.current_filter == LogFilter.TRADES

    def test_filter_combo_index_matches_filter(self, qtbot: QtBot, widget: ActivityLogWidget):
        widget.set_filter(LogFilter.REJECTIONS)
        assert widget._filter_combo.currentIndex() == 2

    def test_new_entry_respects_active_filter(self, qtbot: QtBot, widget: ActivityLogWidget):
        widget.set_filter(LogFilter.TRADES)
        with qtbot.waitSignal(widget.entry_appended, timeout=2000):
            widget.append_cycle_result(_rejection_result())
        assert widget._table.rowCount() == 0  # Ablehnung, nicht Trade

    def test_suggested_visible_in_trades_filter(self, qtbot: QtBot, widget: ActivityLogWidget):
        widget.set_filter(LogFilter.TRADES)
        r = _make_result(action="suggested", reason="suggest_only_mode", ticket=None, lot_size=0.1)
        with qtbot.waitSignal(widget.entry_appended, timeout=2000):
            widget.append_cycle_result(r)
        assert widget._table.rowCount() == 1


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: Tabelleninhalt
# ─────────────────────────────────────────────────────────────────────────────

class TestActivityLogWidgetTableContent:

    def test_symbol_in_table(self, qtbot: QtBot, widget: ActivityLogWidget):
        with qtbot.waitSignal(widget.entry_appended, timeout=2000):
            widget.append_cycle_result(_trade_result(symbol="USDJPY"))
        assert widget._table.item(0, 1).text() == "USDJPY"

    def test_signal_in_table(self, qtbot: QtBot, widget: ActivityLogWidget):
        with qtbot.waitSignal(widget.entry_appended, timeout=2000):
            widget.append_cycle_result(_make_result(signal="short", action="open_sell"))
        text = widget._table.item(0, 2).text()
        assert "short" in text

    def test_confidence_in_signal_column(self, qtbot: QtBot, widget: ActivityLogWidget):
        with qtbot.waitSignal(widget.entry_appended, timeout=2000):
            widget.append_cycle_result(_trade_result(confidence=0.78))
        text = widget._table.item(0, 2).text()
        assert "78%" in text or "0.78" in text or "78" in text

    def test_no_confidence_shows_dash_or_signal(self, qtbot: QtBot, widget: ActivityLogWidget):
        r = _trade_result()
        r["confidence"] = None
        with qtbot.waitSignal(widget.entry_appended, timeout=2000):
            widget.append_cycle_result(r)
        text = widget._table.item(0, 2).text()
        # Kein % im Text wenn keine Konfidenz
        assert "%" not in text

    def test_action_in_table(self, qtbot: QtBot, widget: ActivityLogWidget):
        with qtbot.waitSignal(widget.entry_appended, timeout=2000):
            widget.append_cycle_result(_make_result(signal="short", action="open_sell"))
        assert widget._table.item(0, 4).text() == "open_sell"

    def test_check_passed_shows_checkmark(self, qtbot: QtBot, widget: ActivityLogWidget):
        r = _make_result(checks=[{"name": "RiskGuard", "passed": True, "reason": ""}])
        with qtbot.waitSignal(widget.entry_appended, timeout=2000):
            widget.append_cycle_result(r)
        checks_text = widget._table.item(0, 3).text()
        assert "✓" in checks_text or "RiskGuard" in checks_text

    def test_check_failed_shows_cross(self, qtbot: QtBot, widget: ActivityLogWidget):
        r = _rejection_result()
        with qtbot.waitSignal(widget.entry_appended, timeout=2000):
            widget.append_cycle_result(r)
        checks_text = widget._table.item(0, 3).text()
        assert "✗" in checks_text

    def test_check_failed_reason_in_table(self, qtbot: QtBot, widget: ActivityLogWidget):
        r = _make_result(checks=[{"name": "PreTradeCheck", "passed": False, "reason": "Spread zu hoch"}])
        with qtbot.waitSignal(widget.entry_appended, timeout=2000):
            widget.append_cycle_result(r)
        checks_text = widget._table.item(0, 3).text()
        assert "Spread zu hoch" in checks_text

    def test_no_checks_shows_dash(self, qtbot: QtBot, widget: ActivityLogWidget):
        r = _trade_result(checks=[])
        with qtbot.waitSignal(widget.entry_appended, timeout=2000):
            widget.append_cycle_result(r)
        checks_text = widget._table.item(0, 3).text()
        assert "–" in checks_text or "-" in checks_text

    def test_newest_entry_shown_first(self, qtbot: QtBot, widget: ActivityLogWidget):
        for sym in ["AAA", "BBB"]:
            with qtbot.waitSignal(widget.entry_appended, timeout=2000):
                widget.append_cycle_result(_trade_result(symbol=sym))
        assert widget._table.item(0, 1).text() == "BBB"  # BBB zuletzt -> oben


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: Farbkodierung
# ─────────────────────────────────────────────────────────────────────────────

class TestActivityLogWidgetColors:

    def _cell_color(self, widget: ActivityLogWidget, row: int, col: int) -> str:
        item = widget._table.item(row, col)
        return item.foreground().color().name() if item else ""

    def test_trade_row_green(self, qtbot: QtBot, widget: ActivityLogWidget):
        with qtbot.waitSignal(widget.entry_appended, timeout=2000):
            widget.append_cycle_result(_trade_result())
        color = self._cell_color(widget, 0, 1)
        assert color.lower() == _CATEGORY_COLORS["TRADE"].lower()

    def test_rejection_row_red(self, qtbot: QtBot, widget: ActivityLogWidget):
        with qtbot.waitSignal(widget.entry_appended, timeout=2000):
            widget.append_cycle_result(_rejection_result())
        color = self._cell_color(widget, 0, 1)
        assert color.lower() == _CATEGORY_COLORS["REJECTION"].lower()

    def test_neutral_row_gray(self, qtbot: QtBot, widget: ActivityLogWidget):
        with qtbot.waitSignal(widget.entry_appended, timeout=2000):
            widget.append_cycle_result(_neutral_result())
        color = self._cell_color(widget, 0, 1)
        assert color.lower() == _CATEGORY_COLORS["NEUTRAL"].lower()

    def test_suggested_row_blue(self, qtbot: QtBot, widget: ActivityLogWidget):
        r = _make_result(action="suggested", reason="suggest_only_mode", ticket=None, lot_size=0.1)
        with qtbot.waitSignal(widget.entry_appended, timeout=2000):
            widget.append_cycle_result(r)
        color = self._cell_color(widget, 0, 1)
        assert color.lower() == _CATEGORY_COLORS["SUGGESTED"].lower()

    def test_category_colors_dict_complete(self):
        for cat in ("TRADE", "SUGGESTED", "NEUTRAL", "REJECTION", "EMERGENCY"):
            assert cat in _CATEGORY_COLORS


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: TradingOrchestrator-Integration
# ─────────────────────────────────────────────────────────────────────────────

class TestOrchestratorChecks:

    def _make_orchestrator_mocks(self):
        from src.orchestrator import TradingOrchestrator
        pipeline    = MagicMock()
        risk_guard  = MagicMock()
        pre_trade   = MagicMock()
        signal_model = MagicMock()
        corr_guard  = MagicMock()
        sizer       = MagicMock()
        executor    = MagicMock()

        risk_guard.is_trading_allowed.return_value     = True
        risk_guard.get_position_size_multiplier.return_value = 1.0
        pre_trade.is_safe_to_trade.return_value        = (True, "")
        signal_model.get_signal.return_value           = "flat"
        signal_model.predict_proba.return_value        = {"long": 0.3, "short": 0.2, "neutral": 0.5}
        corr_guard.can_open_position.return_value      = True
        executor.get_open_positions.return_value       = []

        orc = TradingOrchestrator(
            data_pipeline=pipeline,
            risk_guard=risk_guard,
            pre_trade_check=pre_trade,
            signal_model=signal_model,
            correlation_guard=corr_guard,
            position_sizer=sizer,
            order_executor=executor,
            features_loader=lambda sym: __import__("pandas").DataFrame(
                {"f1": [1.0], "close": [1.1], "atr": [0.001]}
            ),
        )
        return orc, risk_guard, pre_trade, signal_model, corr_guard, executor

    def test_result_has_checks_key(self):
        orc, *_ = self._make_orchestrator_mocks()
        result = orc.run_cycle("EURUSD")
        assert "checks" in result

    def test_result_has_timestamp_key(self):
        orc, *_ = self._make_orchestrator_mocks()
        result = orc.run_cycle("EURUSD")
        assert "timestamp" in result
        assert isinstance(result["timestamp"], datetime)

    def test_result_timestamp_is_utc(self):
        orc, *_ = self._make_orchestrator_mocks()
        result = orc.run_cycle("EURUSD")
        assert result["timestamp"].tzinfo is not None

    def test_riskguard_passed_in_checks(self):
        orc, *_ = self._make_orchestrator_mocks()
        result = orc.run_cycle("EURUSD")
        names = [c["name"] for c in result["checks"]]
        assert "RiskGuard" in names
        rg = next(c for c in result["checks"] if c["name"] == "RiskGuard")
        assert rg["passed"] is True

    def test_riskguard_failed_in_checks(self):
        orc, risk_guard, *_ = self._make_orchestrator_mocks()
        risk_guard.is_trading_allowed.return_value = False
        result = orc.run_cycle("EURUSD")
        rg = next((c for c in result["checks"] if c["name"] == "RiskGuard"), None)
        assert rg is not None
        assert rg["passed"] is False

    def test_pretrade_check_passed_in_checks(self):
        orc, *_ = self._make_orchestrator_mocks()
        result = orc.run_cycle("EURUSD")
        names = [c["name"] for c in result["checks"]]
        assert "PreTradeCheck" in names
        pt = next(c for c in result["checks"] if c["name"] == "PreTradeCheck")
        assert pt["passed"] is True

    def test_pretrade_check_failed_in_checks(self):
        orc, _, pre_trade, *_ = self._make_orchestrator_mocks()
        pre_trade.is_safe_to_trade.return_value = (False, "Spread zu hoch")
        result = orc.run_cycle("EURUSD")
        pt = next((c for c in result["checks"] if c["name"] == "PreTradeCheck"), None)
        assert pt is not None
        assert pt["passed"] is False
        assert "Spread" in pt["reason"]

    def test_correlation_guard_checked_for_directional_signal(self):
        orc, _, _, signal_model, corr_guard, executor = self._make_orchestrator_mocks()
        # Mache signal = "long", damit CorrelationGuard geprueft wird
        signal_model.get_signal.return_value = "long"
        corr_guard.can_open_position.return_value = True
        # PositionSizer liefert ungueltig -> Zyklus stoppt nach CorrelationGuard-Check
        size_result = MagicMock()
        size_result.is_valid = False
        size_result.rejection_reason = "Zu klein"
        orc._position_sizer.calculate_lot_size.return_value = size_result
        result = orc.run_cycle("EURUSD")
        names = [c["name"] for c in result["checks"]]
        assert "CorrelationGuard" in names

    def test_correlation_guard_failed_in_checks(self):
        orc, _, _, signal_model, corr_guard, executor = self._make_orchestrator_mocks()
        signal_model.get_signal.return_value = "short"
        corr_guard.can_open_position.return_value = False
        result = orc.run_cycle("EURUSD")
        cg = next((c for c in result["checks"] if c["name"] == "CorrelationGuard"), None)
        assert cg is not None
        assert cg["passed"] is False

    def test_set_activity_callback(self):
        orc, *_ = self._make_orchestrator_mocks()
        cb = MagicMock()
        orc.set_activity_callback(cb)
        assert orc._activity_callback is cb

    def test_set_activity_callback_none(self):
        orc, *_ = self._make_orchestrator_mocks()
        orc.set_activity_callback(MagicMock())
        orc.set_activity_callback(None)
        assert orc._activity_callback is None

    def test_activity_callback_called_in_run_loop(self):
        orc, *_ = self._make_orchestrator_mocks()
        calls: list = []

        def stop_after_first(result):
            calls.append(result)
            orc.stop()   # setzt stop_event -> Loop endet nach aktuellem Zyklus

        orc.set_activity_callback(stop_after_first)
        t = threading.Thread(
            target=orc.run_loop, args=(["EURUSD"],), kwargs={"interval_seconds": 0}
        )
        t.start()
        t.join(timeout=5)
        assert len(calls) >= 1
        assert isinstance(calls[0], dict)
        assert "symbol" in calls[0]


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: BotWorker cycle_completed Signal
# ─────────────────────────────────────────────────────────────────────────────

class TestBotWorkerCycleSignal:

    def test_cycle_completed_signal_exists(self):
        orc = MagicMock()
        orc.run_loop.return_value = None
        worker = BotWorker(orc, ["EURUSD"])
        assert hasattr(worker, "cycle_completed")

    def test_worker_sets_activity_callback(self):
        orc = MagicMock()
        orc.run_loop.return_value = None
        worker = BotWorker(orc, ["EURUSD"])
        worker.run()
        orc.set_activity_callback.assert_called()

    def test_worker_clears_callback_in_finally(self):
        orc = MagicMock()
        orc.run_loop.return_value = None
        worker = BotWorker(orc, ["EURUSD"])
        worker.run()
        last_call = orc.set_activity_callback.call_args_list[-1]
        assert last_call.args[0] is None

    def test_cycle_completed_emitted_when_callback_called(self, qtbot: QtBot):
        """Callback-Emit aus dem 'Orchestrator-Thread' → cycle_completed Signal."""
        results: list = []
        dummy_result = {"symbol": "EURUSD", "action": "flat", "reason": "signal_flat"}

        def fake_run_loop(symbols, interval_seconds=300):
            # Simuliert einen Zyklus durch direktes Aufrufen des Callbacks
            if hasattr(fake_run_loop, "_cb") and fake_run_loop._cb:
                fake_run_loop._cb(dummy_result)

        orc = MagicMock()

        def set_cb(cb):
            fake_run_loop._cb = cb

        orc.set_activity_callback.side_effect = set_cb
        orc.run_loop.side_effect = fake_run_loop

        worker = BotWorker(orc, ["EURUSD"])
        worker.cycle_completed.connect(results.append)
        worker.run()

        assert len(results) == 1
        assert results[0]["symbol"] == "EURUSD"


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: BotControlsWidget cycle_completed Signal
# ─────────────────────────────────────────────────────────────────────────────

class TestBotControlsCycleSignal:

    def test_cycle_completed_signal_exists(self, qtbot: QtBot):
        w = BotControlsWidget()
        qtbot.addWidget(w)
        assert hasattr(w, "cycle_completed")

    def test_cycle_completed_forwarded_from_worker(self, qtbot: QtBot):
        """Startet echten Thread, wartet auf cycle_completed via Signal."""
        dummy_result = {"symbol": "TEST", "action": "flat", "reason": "signal_flat", "checks": []}

        def fake_run_loop(symbols, interval_seconds=300):
            pass  # gibt sofort zurueck

        orc = MagicMock()
        orc.run_loop.side_effect = fake_run_loop

        def set_cb(cb):
            if cb is not None:
                cb(dummy_result)  # sofort feuern

        orc.set_activity_callback.side_effect = set_cb

        w = BotControlsWidget(orchestrator=orc, symbols=["TEST"])
        qtbot.addWidget(w)

        received: list = []
        w.cycle_completed.connect(received.append)

        w._on_start()
        qtbot.waitUntil(lambda: w.bot_state.name == "STOPPED", timeout=5000)

        assert len(received) == 1
        assert received[0]["symbol"] == "TEST"


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: MainWindow-Integration
# ─────────────────────────────────────────────────────────────────────────────

class TestMainWindowActivityLog:

    def test_activity_log_property_exists(self, main_window: MainWindow):
        assert isinstance(main_window.activity_log, ActivityLogWidget)

    def test_activity_log_in_layout(self, main_window: MainWindow):
        # ActivityLogWidget muss irgendwo im Widget-Baum vorhanden sein
        log = main_window.activity_log
        assert log is not None
        assert log.parent() is not None

    def test_cycle_completed_connected_to_activity_log(self, qtbot: QtBot, theme: ThemeManager):
        w = MainWindow(theme_manager=theme)
        qtbot.addWidget(w)

        received: list = []
        w.activity_log.entry_appended.connect(received.append)

        # Direkt cycle_completed emittieren (simuliert Worker)
        dummy = {"symbol": "EURUSD", "action": "flat", "reason": "signal_flat", "checks": []}
        w.bot_controls.cycle_completed.emit(dummy)

        qtbot.waitUntil(lambda: len(received) > 0, timeout=3000)
        assert received[0].symbol == "EURUSD"

    def test_activity_log_receives_entry_via_signal_chain(self, qtbot: QtBot, theme: ThemeManager):
        w = MainWindow(theme_manager=theme)
        qtbot.addWidget(w)

        entry_received: list = []
        w.activity_log.entry_appended.connect(entry_received.append)

        trade_result = {
            "symbol": "GBPUSD", "signal": "short", "action": "open_sell",
            "reason": "signal_executed", "confidence": 0.81,
            "checks": [{"name": "RiskGuard", "passed": True, "reason": ""}],
            "ticket": 99, "lot_size": 0.2,
        }
        w.bot_controls.cycle_completed.emit(trade_result)
        qtbot.waitUntil(lambda: len(entry_received) > 0, timeout=3000)

        e = entry_received[0]
        assert e.symbol == "GBPUSD"
        assert e.action == "open_sell"
        assert e.category == "TRADE"
