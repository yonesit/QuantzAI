"""
tests/unit/test_audit_log.py
Unit- und Integrationstests fuer AuditLog.

Unit-Tests: Schreiben/Lesen/Filtern
Integration-Tests: OrderExecutor, EmergencyHandler, PositionReconciler
erzeugen bei uebergebenem AuditLog tatsaechlich Eintraege.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, date, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.monitoring.audit_log import AuditLog
from src.data.data_router import PriceDiscrepancyError
from src.execution.order_executor import OrderExecutor
from src.execution.emergency import EmergencyHandler
from src.execution.reconciliation import PositionReconciler


# ─────────────────────────────────────────────────────────────────────────────
#  Hilfsfunktion
# ─────────────────────────────────────────────────────────────────────────────

def _raw_count(db_path: Path, table: str) -> int:
    conn = sqlite3.connect(str(db_path))
    n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    conn.close()
    return n


def _raw_rows(db_path: Path, table: str) -> list[dict]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(f"SELECT * FROM {table}").fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ─────────────────────────────────────────────────────────────────────────────
#  Unit-Tests: Initialisierung und Tabellen
# ─────────────────────────────────────────────────────────────────────────────

class TestAuditLogInit:

    def test_db_file_created(self, tmp_path):
        db = tmp_path / "test.db"
        AuditLog(db).close()
        assert db.exists()

    def test_parent_dir_auto_created(self, tmp_path):
        db = tmp_path / "sub" / "dir" / "audit.db"
        AuditLog(db).close()
        assert db.exists()

    def test_tables_exist_after_init(self, tmp_path):
        db = tmp_path / "test.db"
        AuditLog(db).close()
        conn = sqlite3.connect(str(db))
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        conn.close()
        assert {"orders", "errors", "emergencies"} <= tables

    def test_context_manager(self, tmp_path):
        db = tmp_path / "test.db"
        with AuditLog(db) as al:
            al.log_error("INIT_TEST", {"x": 1})
        assert _raw_count(db, "errors") == 1


# ─────────────────────────────────────────────────────────────────────────────
#  Unit-Tests: log_order
# ─────────────────────────────────────────────────────────────────────────────

class TestLogOrder:

    def test_returns_none(self, tmp_path):
        al = AuditLog(tmp_path / "test.db")
        result = al.log_order({"symbol": "EURUSD", "status": "open"})
        al.close()
        assert result is None

    def test_creates_one_row(self, tmp_path):
        db = tmp_path / "test.db"
        al = AuditLog(db)
        al.log_order({"symbol": "EURUSD"})
        al.close()
        assert _raw_count(db, "orders") == 1

    def test_symbol_stored(self, tmp_path):
        db = tmp_path / "test.db"
        al = AuditLog(db)
        al.log_order({"symbol": "GBPUSD", "direction": "buy"})
        al.close()
        rows = _raw_rows(db, "orders")
        assert rows[0]["symbol"] == "GBPUSD"

    def test_direction_stored(self, tmp_path):
        db = tmp_path / "test.db"
        al = AuditLog(db)
        al.log_order({"symbol": "EURUSD", "direction": "sell"})
        al.close()
        assert _raw_rows(db, "orders")[0]["direction"] == "sell"

    def test_ticket_stored(self, tmp_path):
        db = tmp_path / "test.db"
        al = AuditLog(db)
        al.log_order({"ticket": 42, "symbol": "EURUSD"})
        al.close()
        assert _raw_rows(db, "orders")[0]["ticket"] == 42

    def test_status_stored(self, tmp_path):
        db = tmp_path / "test.db"
        al = AuditLog(db)
        al.log_order({"status": "closed"})
        al.close()
        assert _raw_rows(db, "orders")[0]["status"] == "closed"

    def test_ts_is_set_automatically(self, tmp_path):
        db = tmp_path / "test.db"
        al = AuditLog(db)
        al.log_order({"symbol": "EURUSD"})
        al.close()
        ts = _raw_rows(db, "orders")[0]["ts"]
        assert ts and len(ts) > 10

    def test_extra_fields_in_json(self, tmp_path):
        db = tmp_path / "test.db"
        al = AuditLog(db)
        al.log_order({"symbol": "EURUSD", "custom_field": "wert"})
        al.close()
        import json
        extra = json.loads(_raw_rows(db, "orders")[0]["extra_json"])
        assert extra["custom_field"] == "wert"

    def test_no_extra_json_when_only_known_fields(self, tmp_path):
        db = tmp_path / "test.db"
        al = AuditLog(db)
        al.log_order({"symbol": "EURUSD", "direction": "buy", "status": "open"})
        al.close()
        assert _raw_rows(db, "orders")[0]["extra_json"] is None


# ─────────────────────────────────────────────────────────────────────────────
#  Unit-Tests: log_error
# ─────────────────────────────────────────────────────────────────────────────

class TestLogError:

    def test_returns_none(self, tmp_path):
        al = AuditLog(tmp_path / "test.db")
        result = al.log_error("SOME_ERROR", {"msg": "test"})
        al.close()
        assert result is None

    def test_creates_one_row(self, tmp_path):
        db = tmp_path / "test.db"
        al = AuditLog(db)
        al.log_error("MT5_CONN", {"retcode": -1})
        al.close()
        assert _raw_count(db, "errors") == 1

    def test_error_type_stored(self, tmp_path):
        db = tmp_path / "test.db"
        al = AuditLog(db)
        al.log_error("MY_ERROR_TYPE", {})
        al.close()
        assert _raw_rows(db, "errors")[0]["error_type"] == "MY_ERROR_TYPE"

    def test_details_stored_as_json(self, tmp_path):
        db = tmp_path / "test.db"
        al = AuditLog(db)
        al.log_error("FAIL", {"code": 99, "msg": "boom"})
        al.close()
        import json
        details = json.loads(_raw_rows(db, "errors")[0]["details_json"])
        assert details["code"] == 99

    def test_ts_set(self, tmp_path):
        db = tmp_path / "test.db"
        al = AuditLog(db)
        al.log_error("X", {})
        al.close()
        assert _raw_rows(db, "errors")[0]["ts"]


# ─────────────────────────────────────────────────────────────────────────────
#  Unit-Tests: log_emergency
# ─────────────────────────────────────────────────────────────────────────────

class TestLogEmergency:

    def test_returns_none(self, tmp_path):
        al = AuditLog(tmp_path / "test.db")
        result = al.log_emergency("DRAWDOWN", {"reason": "15% hit"})
        al.close()
        assert result is None

    def test_creates_one_row(self, tmp_path):
        db = tmp_path / "test.db"
        al = AuditLog(db)
        al.log_emergency("MT5_UNREACHABLE", {"reason": "3 Retries"})
        al.close()
        assert _raw_count(db, "emergencies") == 1

    def test_event_type_stored(self, tmp_path):
        db = tmp_path / "test.db"
        al = AuditLog(db)
        al.log_emergency("BAD_DATAFEED", {})
        al.close()
        assert _raw_rows(db, "emergencies")[0]["event_type"] == "BAD_DATAFEED"

    def test_reason_stored(self, tmp_path):
        db = tmp_path / "test.db"
        al = AuditLog(db)
        al.log_emergency("DRAWDOWN", {"reason": "Limit erreicht"})
        al.close()
        assert _raw_rows(db, "emergencies")[0]["reason"] == "Limit erreicht"

    def test_ts_set(self, tmp_path):
        db = tmp_path / "test.db"
        al = AuditLog(db)
        al.log_emergency("X", {})
        al.close()
        assert _raw_rows(db, "emergencies")[0]["ts"]


# ─────────────────────────────────────────────────────────────────────────────
#  Unit-Tests: query_orders
# ─────────────────────────────────────────────────────────────────────────────

class TestQueryOrders:

    def _make_al(self, tmp_path) -> AuditLog:
        return AuditLog(tmp_path / "test.db")

    def test_returns_dataframe(self, tmp_path):
        import pandas as pd
        al = self._make_al(tmp_path)
        df = al.query_orders(
            datetime(2020, 1, 1, tzinfo=timezone.utc),
            datetime(2030, 1, 1, tzinfo=timezone.utc),
        )
        al.close()
        assert isinstance(df, pd.DataFrame)

    def test_empty_when_no_orders(self, tmp_path):
        al = self._make_al(tmp_path)
        df = al.query_orders(
            datetime(2020, 1, 1, tzinfo=timezone.utc),
            datetime(2030, 1, 1, tzinfo=timezone.utc),
        )
        al.close()
        assert len(df) == 0

    def test_returns_order_within_range(self, tmp_path):
        al = self._make_al(tmp_path)
        al.log_order({"symbol": "EURUSD", "status": "open"})
        df = al.query_orders(
            datetime(2020, 1, 1, tzinfo=timezone.utc),
            datetime(2030, 1, 1, tzinfo=timezone.utc),
        )
        al.close()
        assert len(df) == 1
        assert df.iloc[0]["symbol"] == "EURUSD"

    def test_excludes_before_start(self, tmp_path):
        al = self._make_al(tmp_path)
        al.log_order({"symbol": "EURUSD"})
        # Start in the future → no results
        future = datetime.now(timezone.utc) + timedelta(days=1)
        df = al.query_orders(future, future + timedelta(days=1))
        al.close()
        assert len(df) == 0

    def test_excludes_after_end(self, tmp_path):
        al = self._make_al(tmp_path)
        al.log_order({"symbol": "EURUSD"})
        # End in the past → no results
        past = datetime.now(timezone.utc) - timedelta(days=365)
        df = al.query_orders(past - timedelta(days=1), past)
        al.close()
        assert len(df) == 0

    def test_filter_by_symbol(self, tmp_path):
        al = self._make_al(tmp_path)
        al.log_order({"symbol": "EURUSD"})
        al.log_order({"symbol": "GBPUSD"})
        df = al.query_orders(
            datetime(2020, 1, 1, tzinfo=timezone.utc),
            datetime(2030, 1, 1, tzinfo=timezone.utc),
            symbol="EURUSD",
        )
        al.close()
        assert len(df) == 1
        assert df.iloc[0]["symbol"] == "EURUSD"

    def test_symbol_none_returns_all(self, tmp_path):
        al = self._make_al(tmp_path)
        al.log_order({"symbol": "EURUSD"})
        al.log_order({"symbol": "GBPUSD"})
        al.log_order({"symbol": "USDJPY"})
        df = al.query_orders(
            datetime(2020, 1, 1, tzinfo=timezone.utc),
            datetime(2030, 1, 1, tzinfo=timezone.utc),
        )
        al.close()
        assert len(df) == 3

    def test_with_date_objects(self, tmp_path):
        al = self._make_al(tmp_path)
        al.log_order({"symbol": "EURUSD"})
        today = date.today()
        df = al.query_orders(
            date(today.year - 1, 1, 1),
            date(today.year + 1, 12, 31),
        )
        al.close()
        assert len(df) == 1

    def test_sorted_by_ts(self, tmp_path):
        al = self._make_al(tmp_path)
        al.log_order({"symbol": "A"})
        al.log_order({"symbol": "B"})
        al.log_order({"symbol": "C"})
        df = al.query_orders(
            datetime(2020, 1, 1, tzinfo=timezone.utc),
            datetime(2030, 1, 1, tzinfo=timezone.utc),
        )
        al.close()
        assert list(df["ts"]) == sorted(df["ts"])

    def test_unknown_symbol_returns_empty(self, tmp_path):
        al = self._make_al(tmp_path)
        al.log_order({"symbol": "EURUSD"})
        df = al.query_orders(
            datetime(2020, 1, 1, tzinfo=timezone.utc),
            datetime(2030, 1, 1, tzinfo=timezone.utc),
            symbol="XYZABC",
        )
        al.close()
        assert len(df) == 0

    def test_dataframe_has_expected_columns(self, tmp_path):
        al = self._make_al(tmp_path)
        al.log_order({"symbol": "EURUSD"})
        df = al.query_orders(
            datetime(2020, 1, 1, tzinfo=timezone.utc),
            datetime(2030, 1, 1, tzinfo=timezone.utc),
        )
        al.close()
        for col in ("id", "ts", "symbol", "direction", "status"):
            assert col in df.columns


# ─────────────────────────────────────────────────────────────────────────────
#  Integrationstests: OrderExecutor schreibt in DB
# ─────────────────────────────────────────────────────────────────────────────

class TestOrderExecutorIntegration:

    def test_open_position_writes_to_audit_log(self, tmp_path):
        db = tmp_path / "audit.db"
        al = AuditLog(db)
        connector = MagicMock()
        executor = OrderExecutor(connector, audit_log=al)

        executor.open_position("EURUSD", "buy", 0.1, 1.08, 1.10)

        df = al.query_orders(
            datetime(2020, 1, 1, tzinfo=timezone.utc),
            datetime(2030, 1, 1, tzinfo=timezone.utc),
        )
        al.close()
        assert len(df) == 1
        assert df.iloc[0]["symbol"] == "EURUSD"
        assert df.iloc[0]["direction"] == "buy"
        assert df.iloc[0]["status"] == "open"

    def test_close_position_writes_to_audit_log(self, tmp_path):
        db = tmp_path / "audit.db"
        al = AuditLog(db)
        connector = MagicMock()
        executor = OrderExecutor(connector, audit_log=al)

        executor.open_position("GBPUSD", "sell", 0.2, 1.30, 1.28)
        ticket = executor.get_open_positions()[0]["ticket"]
        executor.close_position(ticket)

        df = al.query_orders(
            datetime(2020, 1, 1, tzinfo=timezone.utc),
            datetime(2030, 1, 1, tzinfo=timezone.utc),
        )
        al.close()
        # open + close = 2 rows
        assert len(df) == 2
        statuses = set(df["status"])
        assert "open" in statuses
        assert "closed" in statuses

    def test_executor_without_audit_log_still_works(self, tmp_path):
        connector = MagicMock()
        executor = OrderExecutor(connector)  # kein audit_log
        result = executor.open_position("EURUSD", "buy", 0.1, 1.08, 1.10)
        assert result["status"] == "open"


# ─────────────────────────────────────────────────────────────────────────────
#  Integrationstests: EmergencyHandler schreibt in DB
# ─────────────────────────────────────────────────────────────────────────────

class TestEmergencyHandlerIntegration:

    def _make_handler(self, audit_log):
        executor   = MagicMock()
        router     = MagicMock()
        risk_guard = MagicMock()
        executor.get_open_positions.return_value = []
        risk_guard.is_max_drawdown_hit.return_value = True
        return EmergencyHandler(
            executor=executor,
            data_router=router,
            risk_guard=risk_guard,
            audit_log=audit_log,
            _exit_fn=MagicMock(),
        )

    def test_handle_bad_datafeed_writes_emergency(self, tmp_path):
        db = tmp_path / "audit.db"
        al = AuditLog(db)
        handler = self._make_handler(al)
        handler.handle_bad_datafeed(symbol="EURUSD", reason="leer")
        al.close()
        rows = _raw_rows(db, "emergencies")
        assert any(r["event_type"] == "BAD_DATAFEED" for r in rows)

    def test_handle_critical_drawdown_writes_emergencies(self, tmp_path):
        db = tmp_path / "audit.db"
        al = AuditLog(db)
        handler = self._make_handler(al)
        handler.handle_critical_drawdown()
        al.close()
        rows = _raw_rows(db, "emergencies")
        event_types = {r["event_type"] for r in rows}
        assert "CRITICAL_DRAWDOWN" in event_types

    def test_handle_unhandled_exception_writes_emergency(self, tmp_path):
        db = tmp_path / "audit.db"
        al = AuditLog(db)
        handler = self._make_handler(al)
        handler.handle_unhandled_exception(RuntimeError("boom"))
        al.close()
        rows = _raw_rows(db, "emergencies")
        event_types = {r["event_type"] for r in rows}
        assert "UNHANDLED_EXCEPTION" in event_types

    def test_handler_without_audit_log_still_works(self):
        executor   = MagicMock()
        router     = MagicMock()
        risk_guard = MagicMock()
        executor.get_open_positions.return_value = []
        handler = EmergencyHandler(
            executor=executor,
            data_router=router,
            risk_guard=risk_guard,
            _exit_fn=MagicMock(),
        )
        handler.handle_bad_datafeed()
        assert handler.is_trading_paused is True


# ─────────────────────────────────────────────────────────────────────────────
#  Integrationstests: PositionReconciler schreibt in DB
# ─────────────────────────────────────────────────────────────────────────────

class TestPositionReconcilerIntegration:

    def test_missing_locally_writes_to_errors(self, tmp_path):
        db = tmp_path / "audit.db"
        al = AuditLog(db)

        connector = MagicMock()
        connector.register_reconnect_callback = MagicMock()
        executor = MagicMock()
        executor.get_open_positions.return_value = []

        reconciler = PositionReconciler(connector, executor, audit_log=al)
        mt5_position = {
            "ticket": 999, "symbol": "EURUSD", "direction": "buy",
            "lot_size": 0.1, "sl_price": 1.08, "tp_price": 1.10,
            "open_price": 1.09, "status": "open",
        }
        with patch.object(reconciler, "_fetch_mt5_positions", return_value=[mt5_position]):
            reconciler.sync()

        al.close()
        rows = _raw_rows(db, "errors")
        assert any(r["error_type"] == "RECONCILIATION_MISSING_LOCALLY" for r in rows)

    def test_missing_at_mt5_writes_to_errors(self, tmp_path):
        db = tmp_path / "audit.db"
        al = AuditLog(db)

        connector = MagicMock()
        connector.register_reconnect_callback = MagicMock()
        executor = MagicMock()
        executor.get_open_positions.return_value = [
            {"ticket": 777, "symbol": "GBPUSD", "direction": "sell",
             "lot_size": 0.1, "sl_price": 1.30, "tp_price": 1.28, "status": "open"}
        ]

        reconciler = PositionReconciler(connector, executor, audit_log=al)
        with patch.object(reconciler, "_fetch_mt5_positions", return_value=[]):
            reconciler.sync()

        al.close()
        rows = _raw_rows(db, "errors")
        assert any(r["error_type"] == "RECONCILIATION_MISSING_AT_MT5" for r in rows)

    def test_in_sync_writes_no_errors(self, tmp_path):
        db = tmp_path / "audit.db"
        al = AuditLog(db)

        connector = MagicMock()
        connector.register_reconnect_callback = MagicMock()
        executor = MagicMock()
        executor.get_open_positions.return_value = []

        reconciler = PositionReconciler(connector, executor, audit_log=al)
        with patch.object(reconciler, "_fetch_mt5_positions", return_value=[]):
            reconciler.sync()

        al.close()
        assert _raw_count(db, "errors") == 0

    def test_reconciler_without_audit_log_still_works(self):
        connector = MagicMock()
        connector.register_reconnect_callback = MagicMock()
        executor = MagicMock()
        executor.get_open_positions.return_value = []

        reconciler = PositionReconciler(connector, executor)  # kein audit_log
        with patch.object(reconciler, "_fetch_mt5_positions", return_value=[]):
            result = reconciler.sync()
        assert result["in_sync"] is True
