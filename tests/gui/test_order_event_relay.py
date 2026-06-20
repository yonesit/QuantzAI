"""
tests/gui/test_order_event_relay.py
Tests fuer Issue #59: Live-Order-Updates in der GUI.

Prueft:
  - OrderExecutor.set_order_callbacks() – Callbacks feuern korrekt
  - OrderEventRelay – Qt-Signale nach open/close
  - DashboardView.connect_order_executor() – sofortige Positions-Updates
  - _PositionsTable.add_position() / remove_position() / Highlight-Effekt
  - CockpitView.connect_order_executor() – Refresh bei Bot-Order
  - Paper-Modus funktioniert (keine echte MT5-Verbindung)
"""

from __future__ import annotations

import os
import tempfile
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

pytest_plugins = ["pytestqt"]

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QWidget

from gui.widgets.order_event_relay import OrderEventRelay
from gui.views.dashboard_view import DashboardSnapshot, DashboardView, PositionInfo


# ─────────────────────────────────────────────────────────────────────────────
#  Hilfsfunktionen
# ─────────────────────────────────────────────────────────────────────────────

def _make_executor(tmp_path=None):
    """Minimaler Paper-OrderExecutor ohne MT5-Verbindung."""
    from src.execution.order_executor import OrderExecutor
    path = tmp_path or tempfile.mktemp(suffix=".json")
    connector = MagicMock()
    connector.is_connected = False
    return OrderExecutor(connector=connector, paper_trades_path=path)


def _open_order() -> dict:
    return {
        "ticket":    1,
        "symbol":    "EURUSD",
        "direction": "buy",
        "lot_size":  0.10,
        "sl_price":  1.0800,
        "tp_price":  1.1200,
        "open_price": None,
        "status":    "open",
    }


def _close_order() -> dict:
    return {
        "ticket":      1,
        "symbol":      "EURUSD",
        "close_price": 1.0900,
        "status":      "closed",
    }


# ─────────────────────────────────────────────────────────────────────────────
#  1.  OrderExecutor – set_order_callbacks
# ─────────────────────────────────────────────────────────────────────────────

class TestOrderExecutorCallbacks:
    def test_set_order_callbacks_method_exists(self):
        from src.execution.order_executor import OrderExecutor
        assert hasattr(OrderExecutor, "set_order_callbacks")

    def test_on_open_callback_called_after_paper_open(self, tmp_path):
        executor = _make_executor(tmp_path / "trades.json")
        received = []
        executor.set_order_callbacks(on_open=received.append)
        executor.open_position("EURUSD", "buy", 0.10, 1.08, 1.12)
        assert len(received) == 1

    def test_on_open_callback_receives_correct_symbol(self, tmp_path):
        executor = _make_executor(tmp_path / "trades.json")
        received = []
        executor.set_order_callbacks(on_open=received.append)
        executor.open_position("GBPUSD", "sell", 0.05, 1.25, 1.20)
        assert received[0]["symbol"] == "GBPUSD"

    def test_on_open_callback_receives_direction(self, tmp_path):
        executor = _make_executor(tmp_path / "trades.json")
        received = []
        executor.set_order_callbacks(on_open=received.append)
        executor.open_position("EURUSD", "sell", 0.10, 1.12, 1.08)
        assert received[0]["direction"] == "sell"

    def test_on_open_callback_receives_lot_size(self, tmp_path):
        executor = _make_executor(tmp_path / "trades.json")
        received = []
        executor.set_order_callbacks(on_open=received.append)
        executor.open_position("EURUSD", "buy", 0.33, 1.08, 1.12)
        assert received[0]["lot_size"] == pytest.approx(0.33)

    def test_on_close_callback_called_after_paper_close(self, tmp_path):
        executor = _make_executor(tmp_path / "trades.json")
        close_received = []
        executor.set_order_callbacks(on_close=close_received.append)
        executor.open_position("EURUSD", "buy", 0.10, 1.08, 1.12)
        ticket = list(executor._paper_positions.keys())[0]
        executor.close_position(ticket)
        assert len(close_received) == 1

    def test_on_close_callback_receives_ticket(self, tmp_path):
        executor = _make_executor(tmp_path / "trades.json")
        close_received = []
        executor.set_order_callbacks(on_close=close_received.append)
        executor.open_position("EURUSD", "buy", 0.10, 1.08, 1.12)
        ticket = list(executor._paper_positions.keys())[0]
        executor.close_position(ticket)
        assert close_received[0]["ticket"] == ticket

    def test_on_close_callback_receives_closed_status(self, tmp_path):
        executor = _make_executor(tmp_path / "trades.json")
        close_received = []
        executor.set_order_callbacks(on_close=close_received.append)
        executor.open_position("EURUSD", "buy", 0.10, 1.08, 1.12)
        ticket = list(executor._paper_positions.keys())[0]
        executor.close_position(ticket)
        assert close_received[0]["status"] == "closed"

    def test_callback_exception_does_not_crash_executor(self, tmp_path):
        executor = _make_executor(tmp_path / "trades.json")

        def _bad(_):
            raise RuntimeError("test error")

        executor.set_order_callbacks(on_open=_bad)
        result = executor.open_position("EURUSD", "buy", 0.10, 1.08, 1.12)
        assert result is not None  # kein Absturz

    def test_no_callback_by_default(self, tmp_path):
        executor = _make_executor(tmp_path / "trades.json")
        assert executor._on_open_cb is None
        assert executor._on_close_cb is None

    def test_set_callbacks_to_none_removes_them(self, tmp_path):
        executor = _make_executor(tmp_path / "trades.json")
        executor.set_order_callbacks(on_open=lambda _: None)
        executor.set_order_callbacks(on_open=None)
        assert executor._on_open_cb is None

    def test_open_callback_not_called_when_none(self, tmp_path):
        executor = _make_executor(tmp_path / "trades.json")
        executor.set_order_callbacks(on_open=None)
        result = executor.open_position("EURUSD", "buy", 0.10, 1.08, 1.12)
        assert result["ticket"] is not None  # kein Crash

    def test_multiple_opens_each_trigger_callback(self, tmp_path):
        executor = _make_executor(tmp_path / "trades.json")
        received = []
        executor.set_order_callbacks(on_open=received.append)
        executor.open_position("EURUSD", "buy",  0.10, 1.08, 1.12)
        executor.open_position("GBPUSD", "sell", 0.05, 1.25, 1.20)
        assert len(received) == 2


# ─────────────────────────────────────────────────────────────────────────────
#  2.  OrderEventRelay
# ─────────────────────────────────────────────────────────────────────────────

class TestOrderEventRelay:
    def test_creates_without_crash(self, qtbot):
        relay = OrderEventRelay()
        relay.deleteLater()

    def test_has_order_opened_signal(self, qtbot):
        relay = OrderEventRelay()
        assert hasattr(relay, "order_opened")
        relay.deleteLater()

    def test_has_order_closed_signal(self, qtbot):
        relay = OrderEventRelay()
        assert hasattr(relay, "order_closed")
        relay.deleteLater()

    def test_attach_sets_executor_callbacks(self, tmp_path, qtbot):
        executor = _make_executor(tmp_path / "trades.json")
        relay = OrderEventRelay()
        relay.attach(executor)
        assert executor._on_open_cb is not None
        assert executor._on_close_cb is not None
        relay.deleteLater()

    def test_detach_clears_executor_callbacks(self, tmp_path, qtbot):
        executor = _make_executor(tmp_path / "trades.json")
        relay = OrderEventRelay()
        relay.attach(executor)
        relay.detach(executor)
        assert executor._on_open_cb is None
        assert executor._on_close_cb is None
        relay.deleteLater()

    def test_order_opened_signal_fires_on_paper_open(self, tmp_path, qtbot):
        executor = _make_executor(tmp_path / "trades.json")
        relay = OrderEventRelay()
        relay.attach(executor)
        with qtbot.waitSignal(relay.order_opened, timeout=2000) as blocker:
            executor.open_position("EURUSD", "buy", 0.10, 1.08, 1.12)
        assert blocker.args[0]["symbol"] == "EURUSD"
        relay.deleteLater()

    def test_order_closed_signal_fires_on_paper_close(self, tmp_path, qtbot):
        executor = _make_executor(tmp_path / "trades.json")
        relay = OrderEventRelay()
        relay.attach(executor)
        executor.open_position("EURUSD", "buy", 0.10, 1.08, 1.12)
        ticket = list(executor._paper_positions.keys())[0]
        with qtbot.waitSignal(relay.order_closed, timeout=2000) as blocker:
            executor.close_position(ticket)
        assert blocker.args[0]["ticket"] == ticket
        relay.deleteLater()

    def test_order_opened_signal_contains_direction(self, tmp_path, qtbot):
        executor = _make_executor(tmp_path / "trades.json")
        relay = OrderEventRelay()
        relay.attach(executor)
        with qtbot.waitSignal(relay.order_opened, timeout=2000) as blocker:
            executor.open_position("GBPUSD", "sell", 0.05, 1.25, 1.20)
        assert blocker.args[0]["direction"] == "sell"
        relay.deleteLater()

    def test_order_opened_signal_not_fired_without_attach(self, tmp_path, qtbot):
        executor = _make_executor(tmp_path / "trades.json")
        relay = OrderEventRelay()
        received = []
        relay.order_opened.connect(received.append)
        executor.open_position("EURUSD", "buy", 0.10, 1.08, 1.12)
        qtbot.wait(100)
        assert len(received) == 0
        relay.deleteLater()


# ─────────────────────────────────────────────────────────────────────────────
#  3.  _PositionsTable.add_position / remove_position
# ─────────────────────────────────────────────────────────────────────────────

class TestPositionsTableOperations:
    def _make_table(self, qtbot):
        from gui.views.dashboard_view import _PositionsTable
        widget = _PositionsTable()
        qtbot.addWidget(widget)
        return widget

    def test_add_position_inserts_row(self, qtbot):
        tbl = self._make_table(qtbot)
        assert tbl.table.rowCount() == 0
        tbl.add_position(_open_order())
        assert tbl.table.rowCount() == 1

    def test_add_position_shows_symbol(self, qtbot):
        tbl = self._make_table(qtbot)
        tbl.add_position(_open_order())
        assert tbl.table.item(0, 0).text() == "EURUSD"

    def test_add_position_shows_direction_uppercase(self, qtbot):
        tbl = self._make_table(qtbot)
        tbl.add_position(_open_order())
        assert tbl.table.item(0, 1).text() == "BUY"

    def test_add_position_shows_lot_size(self, qtbot):
        tbl = self._make_table(qtbot)
        tbl.add_position(_open_order())
        assert "0.10" in tbl.table.item(0, 2).text()

    def test_add_position_stores_ticket_in_userdata(self, qtbot):
        tbl = self._make_table(qtbot)
        tbl.add_position(_open_order())
        ticket = tbl.table.item(0, 0).data(Qt.ItemDataRole.UserRole)
        assert ticket == 1

    def test_add_position_sets_highlight_background(self, qtbot):
        tbl = self._make_table(qtbot)
        tbl.add_position(_open_order())
        bg = tbl.table.item(0, 0).background().color()
        assert bg != QColor("transparent")
        assert bg.isValid()

    def test_highlight_cleared_after_timer(self, qtbot):
        tbl = self._make_table(qtbot)
        tbl.add_position(_open_order())
        qtbot.wait(2200)
        bg = tbl.table.item(0, 0).background().color()
        assert bg == QColor("transparent") or bg.alpha() == 0

    def test_remove_position_by_ticket(self, qtbot):
        tbl = self._make_table(qtbot)
        tbl.add_position(_open_order())
        assert tbl.table.rowCount() == 1
        tbl.remove_position(1)
        assert tbl.table.rowCount() == 0

    def test_remove_position_unknown_ticket_no_crash(self, qtbot):
        tbl = self._make_table(qtbot)
        tbl.add_position(_open_order())
        tbl.remove_position(9999)  # nicht vorhanden
        assert tbl.table.rowCount() == 1  # unveraendert

    def test_remove_position_removes_correct_row(self, qtbot):
        tbl = self._make_table(qtbot)
        order1 = dict(_open_order(), ticket=1, symbol="EURUSD")
        order2 = dict(_open_order(), ticket=2, symbol="GBPUSD")
        tbl.add_position(order1)
        tbl.add_position(order2)
        tbl.remove_position(1)
        assert tbl.table.rowCount() == 1
        assert tbl.table.item(0, 0).text() == "GBPUSD"

    def test_add_multiple_positions(self, qtbot):
        tbl = self._make_table(qtbot)
        for i in range(3):
            tbl.add_position(dict(_open_order(), ticket=i + 1))
        assert tbl.table.rowCount() == 3

    def test_refresh_stores_ticket_in_userdata(self, qtbot):
        tbl = self._make_table(qtbot)
        snap = DashboardSnapshot(positions=[
            PositionInfo(ticket=42, symbol="EURUSD", direction="buy",
                         lot_size=0.10, open_price=1.085)
        ])
        tbl.refresh(snap)
        ticket = tbl.table.item(0, 0).data(Qt.ItemDataRole.UserRole)
        assert ticket == 42


# ─────────────────────────────────────────────────────────────────────────────
#  4.  DashboardView – connect_order_executor
# ─────────────────────────────────────────────────────────────────────────────

class TestDashboardViewOrderUpdate:
    def test_connect_order_executor_method_exists(self, qtbot):
        view = DashboardView()
        qtbot.addWidget(view)
        assert hasattr(view, "connect_order_executor")

    def test_on_order_opened_method_exists(self, qtbot):
        view = DashboardView()
        qtbot.addWidget(view)
        assert hasattr(view, "on_order_opened")

    def test_on_order_closed_method_exists(self, qtbot):
        view = DashboardView()
        qtbot.addWidget(view)
        assert hasattr(view, "on_order_closed")

    def test_on_order_opened_adds_row(self, qtbot):
        view = DashboardView()
        qtbot.addWidget(view)
        assert view.positions_table.table.rowCount() == 0
        view.on_order_opened(_open_order())
        assert view.positions_table.table.rowCount() == 1

    def test_on_order_opened_shows_symbol(self, qtbot):
        view = DashboardView()
        qtbot.addWidget(view)
        view.on_order_opened(_open_order())
        assert view.positions_table.table.item(0, 0).text() == "EURUSD"

    def test_on_order_closed_removes_row(self, qtbot):
        view = DashboardView()
        qtbot.addWidget(view)
        view.on_order_opened(_open_order())
        assert view.positions_table.table.rowCount() == 1
        view.on_order_closed(_close_order())
        assert view.positions_table.table.rowCount() == 0

    def test_connect_order_executor_wires_relay_signals(self, tmp_path, qtbot):
        executor = _make_executor(tmp_path / "trades.json")
        relay = OrderEventRelay()
        relay.attach(executor)
        view = DashboardView()
        qtbot.addWidget(view)
        view.connect_order_executor(relay)
        executor.open_position("EURUSD", "buy", 0.10, 1.08, 1.12)
        qtbot.waitUntil(lambda: view.positions_table.table.rowCount() == 1, timeout=2000)
        relay.deleteLater()

    def test_relay_close_removes_row_in_dashboard(self, tmp_path, qtbot):
        executor = _make_executor(tmp_path / "trades.json")
        relay = OrderEventRelay()
        relay.attach(executor)
        view = DashboardView()
        qtbot.addWidget(view)
        view.connect_order_executor(relay)
        executor.open_position("EURUSD", "buy", 0.10, 1.08, 1.12)
        qtbot.waitUntil(lambda: view.positions_table.table.rowCount() == 1, timeout=2000)
        ticket = list(executor._paper_positions.keys())[0]
        executor.close_position(ticket)
        qtbot.waitUntil(lambda: view.positions_table.table.rowCount() == 0, timeout=2000)
        relay.deleteLater()

    def test_polling_still_works_after_relay_connection(self, tmp_path, qtbot):
        backend = MagicMock()
        backend.fetch_snapshot.return_value = DashboardSnapshot(
            positions=[PositionInfo(ticket=99, symbol="XAUUSD",
                                    direction="buy", lot_size=0.1, open_price=1800.0)]
        )
        view = DashboardView(backend=backend, interval_ms=100)
        qtbot.addWidget(view)
        relay = OrderEventRelay()
        view.connect_order_executor(relay)
        view.start_polling()
        qtbot.waitUntil(
            lambda: backend.fetch_snapshot.call_count >= 1, timeout=1000
        )
        view.stop_polling()
        relay.deleteLater()

    def test_paper_mode_position_visible_in_dashboard(self, tmp_path, qtbot):
        executor = _make_executor(tmp_path / "trades.json")
        relay = OrderEventRelay()
        relay.attach(executor)
        view = DashboardView()
        qtbot.addWidget(view)
        view.connect_order_executor(relay)
        executor.open_position("USDJPY", "sell", 0.20, 145.0, 142.0)
        qtbot.waitUntil(lambda: view.positions_table.table.rowCount() == 1, timeout=2000)
        assert view.positions_table.table.item(0, 0).text() == "USDJPY"
        relay.deleteLater()

    def test_highlight_applied_after_relay_open(self, tmp_path, qtbot):
        executor = _make_executor(tmp_path / "trades.json")
        relay = OrderEventRelay()
        relay.attach(executor)
        view = DashboardView()
        qtbot.addWidget(view)
        view.connect_order_executor(relay)
        executor.open_position("EURUSD", "buy", 0.10, 1.08, 1.12)
        qtbot.waitUntil(lambda: view.positions_table.table.rowCount() == 1, timeout=2000)
        bg = view.positions_table.table.item(0, 0).background().color()
        assert bg != QColor("transparent")
        relay.deleteLater()


# ─────────────────────────────────────────────────────────────────────────────
#  5.  CockpitView – connect_order_executor
# ─────────────────────────────────────────────────────────────────────────────

class TestCockpitViewOrderUpdate:
    def _make_cockpit(self, qtbot):
        from gui.views.cockpit_view import CockpitView
        view = CockpitView()
        qtbot.addWidget(view)
        return view

    def _make_backend(self, positions=None):
        backend = MagicMock()
        backend.get_open_positions.return_value = positions or []
        backend.fetch_candles.return_value = []
        backend.get_lot_suggestion.return_value = 0.10
        return backend

    def test_connect_order_executor_method_exists(self, qtbot):
        view = self._make_cockpit(qtbot)
        assert hasattr(view, "connect_order_executor")

    def test_on_order_opened_method_exists(self, qtbot):
        view = self._make_cockpit(qtbot)
        assert hasattr(view, "on_order_opened")

    def test_on_order_closed_method_exists(self, qtbot):
        view = self._make_cockpit(qtbot)
        assert hasattr(view, "on_order_closed")

    def test_on_order_opened_calls_refresh_positions(self, qtbot):
        view = self._make_cockpit(qtbot)
        backend = self._make_backend(positions=[{
            "ticket": 1, "symbol": "EURUSD", "direction": "buy",
            "lot_size": 0.1, "open_price": 1.085,
            "sl_price": 1.08, "tp_price": 1.12,
        }])
        view.set_backend(backend)
        view.on_order_opened(_open_order())
        qtbot.wait(50)
        assert backend.get_open_positions.call_count >= 1

    def test_on_order_closed_calls_refresh_positions(self, qtbot):
        view = self._make_cockpit(qtbot)
        backend = self._make_backend()
        view.set_backend(backend)
        initial_calls = backend.get_open_positions.call_count
        view.on_order_closed(_close_order())
        qtbot.wait(50)
        assert backend.get_open_positions.call_count > initial_calls

    def test_connect_order_executor_wires_relay(self, tmp_path, qtbot):
        executor = _make_executor(tmp_path / "trades.json")
        relay = OrderEventRelay()
        relay.attach(executor)
        view = self._make_cockpit(qtbot)
        backend = self._make_backend(positions=[{
            "ticket": 1, "symbol": "EURUSD", "direction": "buy",
            "lot_size": 0.1, "open_price": 1.085,
            "sl_price": None, "tp_price": None,
        }])
        view.set_backend(backend)
        view.connect_order_executor(relay)
        executor.open_position("EURUSD", "buy", 0.10, 1.08, 1.12)
        qtbot.wait(100)
        assert backend.get_open_positions.call_count >= 1
        relay.deleteLater()

    def test_cockpit_refresh_on_paper_open(self, tmp_path, qtbot):
        executor = _make_executor(tmp_path / "trades.json")
        relay = OrderEventRelay()
        relay.attach(executor)
        view = self._make_cockpit(qtbot)
        backend = self._make_backend()
        view.set_backend(backend)
        view.connect_order_executor(relay)
        executor.open_position("GBPUSD", "sell", 0.05, 1.25, 1.20)
        qtbot.wait(100)
        assert backend.get_open_positions.call_count >= 1
        relay.deleteLater()

    def test_cockpit_no_crash_without_backend(self, qtbot):
        view = self._make_cockpit(qtbot)
        # on_order_opened ohne Backend darf nicht crashen
        view.on_order_opened(_open_order())
        view.on_order_closed(_close_order())


# ─────────────────────────────────────────────────────────────────────────────
#  6.  End-to-End: Executor → Relay → Dashboard + Cockpit
# ─────────────────────────────────────────────────────────────────────────────

class TestEndToEnd:
    def test_both_views_update_on_single_open(self, tmp_path, qtbot):
        from gui.views.cockpit_view import CockpitView
        executor = _make_executor(tmp_path / "trades.json")
        relay = OrderEventRelay()
        relay.attach(executor)

        dash = DashboardView()
        qtbot.addWidget(dash)
        dash.connect_order_executor(relay)

        cockpit = CockpitView()
        qtbot.addWidget(cockpit)
        cockpit_backend = MagicMock()
        cockpit_backend.get_open_positions.return_value = [{
            "ticket": 1, "symbol": "EURUSD", "direction": "buy",
            "lot_size": 0.1, "open_price": 1.085,
            "sl_price": None, "tp_price": None,
        }]
        cockpit_backend.fetch_candles.return_value = []
        cockpit.set_backend(cockpit_backend)
        cockpit.connect_order_executor(relay)

        executor.open_position("EURUSD", "buy", 0.10, 1.08, 1.12)
        qtbot.waitUntil(lambda: dash.positions_table.table.rowCount() == 1, timeout=2000)
        qtbot.wait(100)
        assert cockpit_backend.get_open_positions.call_count >= 1
        relay.deleteLater()

    def test_relay_reusable_after_detach_reattach(self, tmp_path, qtbot):
        executor = _make_executor(tmp_path / "trades.json")
        relay = OrderEventRelay()
        relay.attach(executor)
        dash = DashboardView()
        qtbot.addWidget(dash)
        dash.connect_order_executor(relay)

        executor.open_position("EURUSD", "buy", 0.10, 1.08, 1.12)
        qtbot.waitUntil(lambda: dash.positions_table.table.rowCount() == 1, timeout=2000)

        relay.detach(executor)
        # Nach detach – kein weiteres Signal
        received = []
        relay.order_opened.connect(received.append)
        executor.open_position("GBPUSD", "sell", 0.05, 1.25, 1.20)
        qtbot.wait(100)
        assert len(received) == 0
        relay.deleteLater()

    def test_multiple_open_close_cycles(self, tmp_path, qtbot):
        executor = _make_executor(tmp_path / "trades.json")
        relay = OrderEventRelay()
        relay.attach(executor)
        dash = DashboardView()
        qtbot.addWidget(dash)
        dash.connect_order_executor(relay)

        for _ in range(3):
            executor.open_position("EURUSD", "buy", 0.10, 1.08, 1.12)

        qtbot.waitUntil(lambda: dash.positions_table.table.rowCount() == 3, timeout=3000)

        for ticket in list(executor._paper_positions.keys()):
            executor.close_position(ticket)

        qtbot.waitUntil(lambda: dash.positions_table.table.rowCount() == 0, timeout=3000)
        relay.deleteLater()
