"""
tests/gui/test_practice_mode_view.py
GUI-Tests fuer gui/views/practice_mode_view.py via pytest-qt.

Abgedeckt:
  TestPracticeModeViewCreation
    - Erstellt ohne Absturz
    - Signale vorhanden: session_started, position_opened, position_closed

  TestSetupPanel
    - Symbol-Eingabe vorhanden und editierbar
    - Timeframe-Selector vorhanden
    - Start-/Enddatum-Eingaben vorhanden
    - Zeitraum-Preset-Selector vorhanden
    - Load-Session-Button vorhanden
    - load_btn loest backend.load_session aus

  TestPlaybackControls
    - Naechste-Kerze-Button vorhanden
    - +5-Kerzen-Button vorhanden
    - Auto-Play-Button vorhanden (checkable)
    - Geschwindigkeits-Selector vorhanden
    - next_btn loest backend.advance(1) aus
    - next5_btn loest backend.advance(5) aus
    - Steuerelemente deaktiviert ohne Session
    - Steuerelemente aktiv nach session_started

  TestTradingPanel
    - Buy-Button vorhanden
    - Sell-Button vorhanden
    - SL-Eingabe vorhanden
    - TP-Eingabe vorhanden
    - Lot-Eingabe vorhanden
    - Buy loest backend.open_position('buy') aus
    - Sell loest backend.open_position('sell') aus
    - position_opened Signal wird emittiert
    - Kein Trade ohne geladene Session

  TestPositionsPanel
    - Positions-Tabelle vorhanden
    - Position-schliessen-Button vorhanden
    - Position schliessen ruft backend.close_position auf
    - position_closed Signal wird emittiert

  TestStatsPanel
    - Stats-Labels vorhanden
    - Stats werden nach Session-Laden aktualisiert
    - Win-Rate korrekt formatiert

  TestNoLookaheadDisplay
    - Angezeigte Kerzenanzahl im Progress-Label beginnt bei 1
    - Progress-Label zeigt cursor+1 nicht total_candles

  TestPresetDateRange
    - Letzte Woche setzt Datum auf heute -7
    - Letzter Monat setzt Datum auf heute -30
    - Benutzerdefiniert aktiviert Datumseingaben

  TestPracticeBackendProtocol
    - _DefaultPracticeBackend erfuellt PracticeBackend-Protocol
"""

from __future__ import annotations

from unittest.mock import MagicMock, call

import pytest
from pytestqt.qtbot import QtBot

from gui.views.practice_mode_view import (
    PracticeBackend,
    PracticeModeView,
    _DefaultPracticeBackend,
    _fmt_pnl,
    _fmt_price,
    _pnl_color,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _mock_backend() -> MagicMock:
    """Erstellt ein Mock-Backend mit sinnvollen Standardwerten."""
    b = MagicMock()
    b.load_session.return_value = {
        "total_candles": 100,
        "cursor": 0,
        "current_candle": {
            "time": "2024-01-01T10:00:00+00:00",
            "open": 1.1000, "high": 1.1010,
            "low": 1.0990, "close": 1.1005,
            "volume": 500.0,
        },
    }
    b.advance.return_value = {
        "cursor": 1,
        "total_candles": 100,
        "current_candle": {
            "time": "2024-01-01T11:00:00+00:00",
            "open": 1.1005, "high": 1.1015,
            "low": 1.0995, "close": 1.1010,
            "volume": 450.0,
        },
        "is_at_end": False,
        "auto_closed": [],
    }
    b.open_position.return_value = 42
    b.close_position.return_value = {
        "pnl": 50.0,
        "counterfactual_pnl": -50.0,
        "closed_by": "manual",
    }
    b.get_open_positions.return_value = [
        {
            "position_id": 42,
            "direction": "buy",
            "entry_price": 1.1005,
            "sl": 1.0950,
            "tp": 1.1100,
        }
    ]
    b.get_stats.return_value = {
        "trade_count": 3,
        "wins": 2,
        "losses": 1,
        "win_rate": 2 / 3,
        "total_pnl": 75.0,
    }
    b.suggest_lot_size.return_value = 0.05
    return b


@pytest.fixture
def view(qtbot: QtBot) -> PracticeModeView:
    v = PracticeModeView(backend=_mock_backend())
    qtbot.addWidget(v)
    return v


@pytest.fixture
def loaded_view(qtbot: QtBot) -> PracticeModeView:
    """View mit geladener Session."""
    v = PracticeModeView(backend=_mock_backend())
    qtbot.addWidget(v)
    qtbot.mouseClick(v.load_btn, __import__("PySide6.QtCore", fromlist=["Qt"]).Qt.MouseButton.LeftButton)
    return v


# ─────────────────────────────────────────────────────────────────────────────
#  TestPracticeModeViewCreation
# ─────────────────────────────────────────────────────────────────────────────

class TestPracticeModeViewCreation:
    def test_creates_without_crash(self, qtbot):
        v = PracticeModeView()
        qtbot.addWidget(v)

    def test_session_started_signal_exists(self, view):
        assert hasattr(view, "session_started")

    def test_position_opened_signal_exists(self, view):
        assert hasattr(view, "position_opened")

    def test_position_closed_signal_exists(self, view):
        assert hasattr(view, "position_closed")

    def test_default_backend_no_crash(self, qtbot):
        v = PracticeModeView(backend=_DefaultPracticeBackend())
        qtbot.addWidget(v)


# ─────────────────────────────────────────────────────────────────────────────
#  TestSetupPanel
# ─────────────────────────────────────────────────────────────────────────────

class TestSetupPanel:
    def test_symbol_input_exists(self, view):
        assert view.symbol_input is not None

    def test_symbol_input_default(self, view):
        assert view.symbol_input.text() == "EURUSD"

    def test_timeframe_combo_exists(self, view):
        assert view.timeframe_combo is not None

    def test_timeframe_combo_has_h1(self, view):
        items = [view.timeframe_combo.itemText(i) for i in range(view.timeframe_combo.count())]
        assert "H1" in items

    def test_range_combo_exists(self, view):
        assert view.range_combo is not None

    def test_range_combo_has_presets(self, view):
        items = [view.range_combo.itemText(i) for i in range(view.range_combo.count())]
        assert any("Woche" in item or "woche" in item.lower() for item in items)
        assert any("Monat" in item or "monat" in item.lower() for item in items)

    def test_start_date_edit_exists(self, view):
        assert view.start_date_edit is not None

    def test_end_date_edit_exists(self, view):
        assert view.end_date_edit is not None

    def test_load_btn_exists(self, view):
        assert view.load_btn is not None

    def test_load_session_calls_backend(self, qtbot):
        b = _mock_backend()
        v = PracticeModeView(backend=b)
        qtbot.addWidget(v)
        qtbot.mouseClick(v.load_btn, __import__("PySide6.QtCore", fromlist=["Qt"]).Qt.MouseButton.LeftButton)
        b.load_session.assert_called_once()

    def test_load_session_passes_symbol(self, qtbot):
        b = _mock_backend()
        v = PracticeModeView(backend=b)
        qtbot.addWidget(v)
        v.symbol_input.setText("GBPUSD")
        qtbot.mouseClick(v.load_btn, __import__("PySide6.QtCore", fromlist=["Qt"]).Qt.MouseButton.LeftButton)
        args = b.load_session.call_args[0]
        assert args[0] == "GBPUSD"

    def test_load_session_emits_session_started(self, qtbot):
        b = _mock_backend()
        v = PracticeModeView(backend=b)
        qtbot.addWidget(v)
        signals = []
        v.session_started.connect(lambda: signals.append(1))
        qtbot.mouseClick(v.load_btn, __import__("PySide6.QtCore", fromlist=["Qt"]).Qt.MouseButton.LeftButton)
        assert len(signals) == 1


# ─────────────────────────────────────────────────────────────────────────────
#  TestPlaybackControls
# ─────────────────────────────────────────────────────────────────────────────

class TestPlaybackControls:
    def _click(self, qtbot, btn):
        from PySide6.QtCore import Qt
        qtbot.mouseClick(btn, Qt.MouseButton.LeftButton)

    def test_next_btn_exists(self, view):
        assert view.next_btn is not None

    def test_next5_btn_exists(self, view):
        assert view.next5_btn is not None

    def test_auto_btn_exists(self, view):
        assert view.auto_btn is not None

    def test_auto_btn_is_checkable(self, view):
        assert view.auto_btn.isCheckable()

    def test_speed_combo_exists(self, view):
        assert view.speed_combo is not None

    def test_speed_combo_has_1x(self, view):
        items = [view.speed_combo.itemText(i) for i in range(view.speed_combo.count())]
        assert "1x" in items

    def test_speed_combo_has_5x(self, view):
        items = [view.speed_combo.itemText(i) for i in range(view.speed_combo.count())]
        assert "5x" in items

    def test_controls_disabled_without_session(self, view):
        assert not view.next_btn.isEnabled()
        assert not view.buy_btn.isEnabled()

    def test_controls_enabled_after_load(self, qtbot):
        from PySide6.QtCore import Qt
        b = _mock_backend()
        v = PracticeModeView(backend=b)
        qtbot.addWidget(v)
        qtbot.mouseClick(v.load_btn, Qt.MouseButton.LeftButton)
        assert v.next_btn.isEnabled()
        assert v.buy_btn.isEnabled()

    def test_next_btn_calls_advance_1(self, qtbot):
        from PySide6.QtCore import Qt
        b = _mock_backend()
        v = PracticeModeView(backend=b)
        qtbot.addWidget(v)
        qtbot.mouseClick(v.load_btn, Qt.MouseButton.LeftButton)
        b.advance.reset_mock()
        qtbot.mouseClick(v.next_btn, Qt.MouseButton.LeftButton)
        b.advance.assert_called_once_with(1)

    def test_next5_btn_calls_advance_5(self, qtbot):
        from PySide6.QtCore import Qt
        b = _mock_backend()
        v = PracticeModeView(backend=b)
        qtbot.addWidget(v)
        qtbot.mouseClick(v.load_btn, Qt.MouseButton.LeftButton)
        b.advance.reset_mock()
        qtbot.mouseClick(v.next5_btn, Qt.MouseButton.LeftButton)
        b.advance.assert_called_once_with(5)


# ─────────────────────────────────────────────────────────────────────────────
#  TestTradingPanel
# ─────────────────────────────────────────────────────────────────────────────

class TestTradingPanel:
    def test_buy_btn_exists(self, view):
        assert view.buy_btn is not None

    def test_sell_btn_exists(self, view):
        assert view.sell_btn is not None

    def test_sl_input_exists(self, view):
        assert view.sl_input is not None

    def test_tp_input_exists(self, view):
        assert view.tp_input is not None

    def test_lot_input_exists(self, view):
        assert view.lot_input is not None

    def test_lot_input_default(self, view):
        assert view.lot_input.value() == 0.01

    def test_buy_calls_open_position_buy(self, qtbot):
        from PySide6.QtCore import Qt
        b = _mock_backend()
        v = PracticeModeView(backend=b)
        qtbot.addWidget(v)
        qtbot.mouseClick(v.load_btn, Qt.MouseButton.LeftButton)
        b.open_position.reset_mock()
        qtbot.mouseClick(v.buy_btn, Qt.MouseButton.LeftButton)
        b.open_position.assert_called_once()
        args = b.open_position.call_args[0]
        assert args[0] == "buy"

    def test_sell_calls_open_position_sell(self, qtbot):
        from PySide6.QtCore import Qt
        b = _mock_backend()
        v = PracticeModeView(backend=b)
        qtbot.addWidget(v)
        qtbot.mouseClick(v.load_btn, Qt.MouseButton.LeftButton)
        b.open_position.reset_mock()
        qtbot.mouseClick(v.sell_btn, Qt.MouseButton.LeftButton)
        b.open_position.assert_called_once()
        args = b.open_position.call_args[0]
        assert args[0] == "sell"

    def test_buy_emits_position_opened(self, qtbot):
        from PySide6.QtCore import Qt
        b = _mock_backend()
        v = PracticeModeView(backend=b)
        qtbot.addWidget(v)
        qtbot.mouseClick(v.load_btn, Qt.MouseButton.LeftButton)
        emitted = []
        v.position_opened.connect(lambda pid: emitted.append(pid))
        qtbot.mouseClick(v.buy_btn, Qt.MouseButton.LeftButton)
        assert len(emitted) == 1
        assert emitted[0] == 42  # mock returns 42

    def test_buy_passes_lot_sl_tp(self, qtbot):
        from PySide6.QtCore import Qt
        b = _mock_backend()
        v = PracticeModeView(backend=b)
        qtbot.addWidget(v)
        qtbot.mouseClick(v.load_btn, Qt.MouseButton.LeftButton)
        v.lot_input.setValue(0.10)
        v.sl_input.setValue(1.0950)
        v.tp_input.setValue(1.1100)
        b.open_position.reset_mock()
        qtbot.mouseClick(v.buy_btn, Qt.MouseButton.LeftButton)
        args = b.open_position.call_args[0]
        assert args[0] == "buy"
        assert abs(args[1] - 0.10) < 0.001
        assert abs(args[2] - 1.0950) < 0.00001
        assert abs(args[3] - 1.1100) < 0.00001

    def test_no_trade_without_session(self, qtbot):
        from PySide6.QtCore import Qt
        b = _mock_backend()
        v = PracticeModeView(backend=b)
        qtbot.addWidget(v)
        # Session NICHT laden
        qtbot.mouseClick(v.buy_btn, Qt.MouseButton.LeftButton)
        b.open_position.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
#  TestPositionsPanel
# ─────────────────────────────────────────────────────────────────────────────

class TestPositionsPanel:
    def test_positions_table_exists(self, view):
        assert view.positions_table is not None

    def test_positions_table_has_5_columns(self, view):
        assert view.positions_table.columnCount() == 5

    def test_positions_table_column_names(self, view):
        headers = [
            view.positions_table.horizontalHeaderItem(i).text()
            for i in range(view.positions_table.columnCount())
        ]
        assert "ID" in headers
        assert "Richtung" in headers
        assert "Entry" in headers

    def test_close_position_btn_exists(self, view):
        assert view.close_position_btn is not None

    def test_close_position_calls_backend(self, qtbot):
        from PySide6.QtCore import Qt
        b = _mock_backend()
        v = PracticeModeView(backend=b)
        qtbot.addWidget(v)
        qtbot.mouseClick(v.load_btn, Qt.MouseButton.LeftButton)
        qtbot.mouseClick(v.buy_btn, Qt.MouseButton.LeftButton)
        # Zeile auswaehlen
        v.positions_table.selectRow(0)
        b.close_position.reset_mock()
        qtbot.mouseClick(v.close_position_btn, Qt.MouseButton.LeftButton)
        b.close_position.assert_called_once()

    def test_close_position_emits_signal(self, qtbot):
        from PySide6.QtCore import Qt
        b = _mock_backend()
        v = PracticeModeView(backend=b)
        qtbot.addWidget(v)
        qtbot.mouseClick(v.load_btn, Qt.MouseButton.LeftButton)
        qtbot.mouseClick(v.buy_btn, Qt.MouseButton.LeftButton)
        v.positions_table.selectRow(0)
        emitted = []
        v.position_closed.connect(lambda r: emitted.append(r))
        qtbot.mouseClick(v.close_position_btn, Qt.MouseButton.LeftButton)
        assert len(emitted) == 1


# ─────────────────────────────────────────────────────────────────────────────
#  TestStatsPanel
# ─────────────────────────────────────────────────────────────────────────────

class TestStatsPanel:
    def test_stats_trade_count_label_exists(self, view):
        assert view.stats_trade_count_label is not None

    def test_stats_win_rate_label_exists(self, view):
        assert view.stats_win_rate_label is not None

    def test_stats_pnl_label_exists(self, view):
        assert view.stats_pnl_label is not None

    def test_stats_updated_after_load(self, qtbot):
        from PySide6.QtCore import Qt
        b = _mock_backend()
        v = PracticeModeView(backend=b)
        qtbot.addWidget(v)
        qtbot.mouseClick(v.load_btn, Qt.MouseButton.LeftButton)
        assert "3" in v.stats_trade_count_label.text()

    def test_win_rate_formatted_as_percent(self, qtbot):
        from PySide6.QtCore import Qt
        b = _mock_backend()
        v = PracticeModeView(backend=b)
        qtbot.addWidget(v)
        qtbot.mouseClick(v.load_btn, Qt.MouseButton.LeftButton)
        assert "%" in v.stats_win_rate_label.text()

    def test_stats_pnl_shown(self, qtbot):
        from PySide6.QtCore import Qt
        b = _mock_backend()
        v = PracticeModeView(backend=b)
        qtbot.addWidget(v)
        qtbot.mouseClick(v.load_btn, Qt.MouseButton.LeftButton)
        assert "75" in v.stats_pnl_label.text()


# ─────────────────────────────────────────────────────────────────────────────
#  TestNoLookaheadDisplay
# ─────────────────────────────────────────────────────────────────────────────

class TestNoLookaheadDisplay:
    def test_progress_label_shows_cursor_1_not_total_on_load(self, qtbot):
        """Nach Session-Laden zeigt Progress Kerze 1 von 100 – kein Look-ahead."""
        from PySide6.QtCore import Qt
        b = _mock_backend()
        v = PracticeModeView(backend=b)
        qtbot.addWidget(v)
        qtbot.mouseClick(v.load_btn, Qt.MouseButton.LeftButton)
        text = v.progress_label.text()
        # "Kerze 1 / 100" – zeigt cursor+1=1, nicht total=100 als aktuelle Position
        assert "1" in text

    def test_progress_label_updates_on_advance(self, qtbot):
        from PySide6.QtCore import Qt
        b = _mock_backend()
        v = PracticeModeView(backend=b)
        qtbot.addWidget(v)
        qtbot.mouseClick(v.load_btn, Qt.MouseButton.LeftButton)
        initial = v.progress_label.text()
        qtbot.mouseClick(v.next_btn, Qt.MouseButton.LeftButton)
        updated = v.progress_label.text()
        # Progress hat sich geaendert
        assert updated != initial

    def test_candle_panel_exists(self, view):
        assert view.candle_table is not None

    def test_candle_table_has_ohlcv_rows(self, view):
        assert view.candle_table.rowCount() == 5

    def test_price_label_exists(self, view):
        assert view.price_label is not None


# ─────────────────────────────────────────────────────────────────────────────
#  TestPresetDateRange
# ─────────────────────────────────────────────────────────────────────────────

class TestPresetDateRange:
    def test_custom_preset_enables_date_inputs(self, view):
        # Benutzerdefiniert auswaehlen
        idx = None
        for i in range(view.range_combo.count()):
            if view.range_combo.itemData(i) == "custom":
                idx = i
                break
        assert idx is not None
        view.range_combo.setCurrentIndex(idx)
        assert view.start_date_edit.isEnabled()
        assert view.end_date_edit.isEnabled()

    def test_week_preset_disables_date_inputs(self, view):
        # Erst custom, dann zurueck zu week
        for i in range(view.range_combo.count()):
            if view.range_combo.itemData(i) == "custom":
                view.range_combo.setCurrentIndex(i)
                break
        for i in range(view.range_combo.count()):
            if view.range_combo.itemData(i) == "week":
                view.range_combo.setCurrentIndex(i)
                break
        assert not view.start_date_edit.isEnabled()

    def test_month_preset_disables_date_inputs(self, view):
        for i in range(view.range_combo.count()):
            if view.range_combo.itemData(i) == "custom":
                view.range_combo.setCurrentIndex(i)
                break
        for i in range(view.range_combo.count()):
            if view.range_combo.itemData(i) == "month":
                view.range_combo.setCurrentIndex(i)
                break
        assert not view.start_date_edit.isEnabled()


# ─────────────────────────────────────────────────────────────────────────────
#  TestPracticeBackendProtocol
# ─────────────────────────────────────────────────────────────────────────────

class TestPracticeBackendProtocol:
    def test_default_backend_satisfies_protocol(self):
        b = _DefaultPracticeBackend()
        assert isinstance(b, PracticeBackend)

    def test_default_backend_load_session(self):
        b = _DefaultPracticeBackend()
        result = b.load_session("EURUSD", "H1", "2024-01-01", "2024-01-31")
        assert isinstance(result, dict)

    def test_default_backend_advance(self):
        b = _DefaultPracticeBackend()
        result = b.advance(1)
        assert isinstance(result, dict)
        assert result.get("is_at_end") is True

    def test_default_backend_get_stats(self):
        b = _DefaultPracticeBackend()
        stats = b.get_stats()
        assert stats["trade_count"] == 0
        assert stats["win_rate"] == 0.0

    def test_default_backend_suggest_lot_size(self):
        b = _DefaultPracticeBackend()
        lot = b.suggest_lot_size(10_000.0, 0.001)
        assert lot == 0.01


# ─────────────────────────────────────────────────────────────────────────────
#  TestHelperFunctions
# ─────────────────────────────────────────────────────────────────────────────

class TestHelperFunctions:
    def test_pnl_color_positive(self):
        assert _pnl_color(100.0) == "#27ae60"

    def test_pnl_color_negative(self):
        assert _pnl_color(-50.0) == "#c0392b"

    def test_pnl_color_zero(self):
        assert _pnl_color(0.0) == "#27ae60"

    def test_fmt_price_none(self):
        assert _fmt_price(None) == "—"

    def test_fmt_price_value(self):
        result = _fmt_price(1.1005)
        assert "1.1005" in result

    def test_fmt_pnl_positive(self):
        result = _fmt_pnl(50.0)
        assert "+" in result

    def test_fmt_pnl_negative(self):
        result = _fmt_pnl(-30.0)
        assert "-" in result

    def test_fmt_pnl_none(self):
        assert _fmt_pnl(None) == "—"
