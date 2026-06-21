"""
Unit-Tests fuer ExecutionTracker, SlippageRecord, FeeRecord
und die Erweiterungen des OrderExecutors (Limit-Orders, Partial-Fill,
Slippage-/Gebuehren-Tracking).
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from src.execution.execution_tracker import (
    ExecutionTracker,
    FeeRecord,
    SlippageRecord,
)
from src.execution.order_executor import OrderError, OrderExecutor


# ─────────────────────────────────────────────────────────────────────────────
#  Hilfsfunktionen
# ─────────────────────────────────────────────────────────────────────────────

def _connector(connected: bool = True) -> MagicMock:
    conn = MagicMock()
    type(conn).is_connected = PropertyMock(return_value=connected)
    conn.get_symbol_info.return_value = {
        "point":       0.00001,
        "digits":      5,
        "spread":      10,
        "commission":  0.0,
        "swap_long":  -7.0,
        "swap_short":  2.0,
    }
    return conn


def _paper_executor(tmp_path: Path, **kwargs) -> OrderExecutor:
    return OrderExecutor(
        connector=_connector(),
        live_trading_enabled=False,
        paper_trades_path=tmp_path / "paper_trades.json",
        **kwargs,
    )


def _live_executor(tmp_path: Path, **kwargs) -> OrderExecutor:
    return OrderExecutor(
        connector=_connector(),
        live_trading_enabled=True,
        paper_trades_path=tmp_path / "paper_trades.json",
        **kwargs,
    )


def _mt5_mock(
    ticket: int = 42,
    fill_price: float = 1.10000,
    filled_volume: float | None = None,
) -> MagicMock:
    mt5 = MagicMock()
    mt5.ORDER_TYPE_BUY       = 0
    mt5.ORDER_TYPE_SELL      = 1
    mt5.ORDER_TYPE_BUY_LIMIT = 2
    mt5.ORDER_TYPE_SELL_LIMIT = 3
    mt5.TRADE_ACTION_DEAL    = 1
    mt5.TRADE_ACTION_PENDING = 5
    mt5.TRADE_ACTION_REMOVE  = 4
    mt5.TRADE_ACTION_SLTP    = 6
    mt5.ORDER_FILLING_IOC    = 1
    mt5.TRADE_RETCODE_DONE   = 10009

    ok_result = MagicMock()
    ok_result.retcode  = 10009
    ok_result.order    = ticket
    ok_result.comment  = "Request completed"
    ok_result.price    = fill_price
    ok_result.volume   = filled_volume if filled_volume is not None else 0.1
    mt5.order_send.return_value = ok_result

    pos = MagicMock()
    pos.ticket     = ticket
    pos.symbol     = "EURUSD"
    pos.type       = 0
    pos.volume     = 0.1
    pos.price_open = fill_price
    pos.sl         = 1.0950
    pos.tp         = 1.1100
    mt5.positions_get.return_value = [pos]

    pending = MagicMock()
    pending.ticket = ticket
    mt5.orders_get.return_value = [pending]

    return mt5


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: SlippageRecord
# ─────────────────────────────────────────────────────────────────────────────

class TestSlippageRecord:

    def test_fields_stored(self):
        r = SlippageRecord(
            ticket=1, symbol="EURUSD", direction="buy",
            expected_price=1.10000, actual_price=1.10005,
            slippage_pips=0.5,
        )
        assert r.ticket         == 1
        assert r.symbol         == "EURUSD"
        assert r.direction      == "buy"
        assert r.expected_price == 1.10000
        assert r.actual_price   == 1.10005
        assert r.slippage_pips  == 0.5

    def test_timestamp_default_utc(self):
        before = datetime.now(timezone.utc)
        r = SlippageRecord(
            ticket=1, symbol="X", direction="buy",
            expected_price=1.0, actual_price=1.0, slippage_pips=0.0,
        )
        after = datetime.now(timezone.utc)
        assert before <= r.timestamp <= after

    def test_explicit_timestamp(self):
        ts = datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc)
        r = SlippageRecord(
            ticket=1, symbol="X", direction="buy",
            expected_price=1.0, actual_price=1.0, slippage_pips=0.0,
            timestamp=ts,
        )
        assert r.timestamp == ts


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: FeeRecord
# ─────────────────────────────────────────────────────────────────────────────

class TestFeeRecord:

    def test_fields_stored(self):
        r = FeeRecord(ticket=5, symbol="GBPUSD", spread=0.0002, commission=0.001)
        assert r.ticket     == 5
        assert r.symbol     == "GBPUSD"
        assert r.spread     == 0.0002
        assert r.commission == 0.001
        assert r.swap       == 0.0

    def test_total_fees_no_swap(self):
        r = FeeRecord(ticket=1, symbol="X", spread=0.0001, commission=0.0005)
        assert abs(r.total_fees - 0.0006) < 1e-10

    def test_total_fees_with_swap(self):
        r = FeeRecord(ticket=1, symbol="X", spread=0.0001, commission=0.0, swap=0.0003)
        assert abs(r.total_fees - 0.0004) < 1e-10

    def test_timestamp_default_utc(self):
        before = datetime.now(timezone.utc)
        r = FeeRecord(ticket=1, symbol="X", spread=0.0, commission=0.0)
        after = datetime.now(timezone.utc)
        assert before <= r.timestamp <= after


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: ExecutionTracker – Initialisierung
# ─────────────────────────────────────────────────────────────────────────────

class TestExecutionTrackerInit:

    def test_default_pip_size(self):
        et = ExecutionTracker()
        assert et._pip_size == 0.0001

    def test_custom_pip_size(self):
        et = ExecutionTracker(pip_size=0.001)
        assert et._pip_size == 0.001

    def test_invalid_pip_size_raises(self):
        with pytest.raises(ValueError, match="pip_size"):
            ExecutionTracker(pip_size=0.0)

    def test_negative_pip_size_raises(self):
        with pytest.raises(ValueError):
            ExecutionTracker(pip_size=-0.0001)

    def test_empty_records_initially(self):
        et = ExecutionTracker()
        assert et._slippage_records == []
        assert et._fee_records == []


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: ExecutionTracker – Slippage
# ─────────────────────────────────────────────────────────────────────────────

class TestExecutionTrackerSlippage:

    def test_record_slippage_buy_positive(self):
        et = ExecutionTracker(pip_size=0.0001)
        r = et.record_slippage(
            ticket=1, symbol="EURUSD", direction="buy",
            expected_price=1.10000, actual_price=1.10005,
        )
        assert isinstance(r, SlippageRecord)
        assert abs(r.slippage_pips - 0.5) < 1e-9

    def test_record_slippage_buy_negative(self):
        et = ExecutionTracker(pip_size=0.0001)
        r = et.record_slippage(
            ticket=2, symbol="EURUSD", direction="buy",
            expected_price=1.10010, actual_price=1.10000,
        )
        assert r.slippage_pips < 0

    def test_record_slippage_sell_positive(self):
        et = ExecutionTracker(pip_size=0.0001)
        r = et.record_slippage(
            ticket=3, symbol="EURUSD", direction="sell",
            expected_price=1.10010, actual_price=1.10000,
        )
        assert abs(r.slippage_pips - 1.0) < 1e-9

    def test_record_slippage_zero(self):
        et = ExecutionTracker(pip_size=0.0001)
        r = et.record_slippage(
            ticket=4, symbol="EURUSD", direction="buy",
            expected_price=1.10000, actual_price=1.10000,
        )
        assert r.slippage_pips == 0.0

    def test_records_appended(self):
        et = ExecutionTracker()
        et.record_slippage(1, "X", "buy", 1.0, 1.0001)
        et.record_slippage(2, "X", "sell", 1.0, 1.0)
        assert len(et._slippage_records) == 2

    def test_get_slippage_records_limit(self):
        et = ExecutionTracker()
        for i in range(10):
            et.record_slippage(i, "X", "buy", 1.0, 1.0001)
        assert len(et.get_slippage_records(n=5)) == 5
        assert len(et.get_slippage_records(n=5)) == 5  # last 5

    def test_get_slippage_records_n_zero_returns_all(self):
        et = ExecutionTracker()
        for i in range(5):
            et.record_slippage(i, "X", "buy", 1.0, 1.0)
        assert len(et.get_slippage_records(n=0)) == 5


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: ExecutionTracker – Gebuehren
# ─────────────────────────────────────────────────────────────────────────────

class TestExecutionTrackerFees:

    def test_record_fees_basic(self):
        et = ExecutionTracker()
        r = et.record_fees(1, "EURUSD", spread=0.0001, commission=0.0005)
        assert r.ticket == 1
        assert r.spread == 0.0001
        assert r.commission == 0.0005

    def test_record_fees_with_swap(self):
        et = ExecutionTracker()
        r = et.record_fees(2, "EURUSD", spread=0.0, commission=0.0, swap=0.0003)
        assert abs(r.total_fees - 0.0003) < 1e-10

    def test_records_appended(self):
        et = ExecutionTracker()
        et.record_fees(1, "X", 0.0001, 0.0)
        et.record_fees(2, "X", 0.0002, 0.0)
        assert len(et._fee_records) == 2

    def test_get_fee_records_limit(self):
        et = ExecutionTracker()
        for i in range(8):
            et.record_fees(i, "X", 0.0001, 0.0)
        assert len(et.get_fee_records(n=3)) == 3


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: ExecutionTracker – Durchschnitte
# ─────────────────────────────────────────────────────────────────────────────

class TestExecutionTrackerAverages:

    def test_avg_slippage_empty(self):
        et = ExecutionTracker()
        assert et.get_avg_slippage_pips() is None

    def test_avg_slippage_single(self):
        et = ExecutionTracker(pip_size=0.0001)
        et.record_slippage(1, "X", "buy", 1.0, 1.0002)
        avg = et.get_avg_slippage_pips()
        assert abs(avg - 2.0) < 1e-9

    def test_avg_slippage_multiple(self):
        et = ExecutionTracker(pip_size=0.0001)
        et.record_slippage(1, "X", "buy", 1.0, 1.0001)   # 1.0 pip
        et.record_slippage(2, "X", "buy", 1.0, 1.0003)   # 3.0 pips
        avg = et.get_avg_slippage_pips()
        assert abs(avg - 2.0) < 1e-9

    def test_avg_fees_empty(self):
        et = ExecutionTracker()
        assert et.get_avg_total_fees() is None

    def test_avg_fees_single(self):
        et = ExecutionTracker()
        et.record_fees(1, "X", spread=0.0002, commission=0.0003)
        avg = et.get_avg_total_fees()
        assert abs(avg - 0.0005) < 1e-10

    def test_avg_fees_multiple(self):
        et = ExecutionTracker()
        et.record_fees(1, "X", spread=0.0001, commission=0.0)
        et.record_fees(2, "X", spread=0.0003, commission=0.0)
        avg = et.get_avg_total_fees()
        assert abs(avg - 0.0002) < 1e-10

    def test_avg_uses_only_last_n(self):
        et = ExecutionTracker(pip_size=0.0001)
        # First 5 trades: 0 slippage
        for i in range(5):
            et.record_slippage(i, "X", "buy", 1.0, 1.0)
        # Last 5 trades: 10 pips slippage
        for i in range(5, 10):
            et.record_slippage(i, "X", "buy", 1.0, 1.001)  # 10 pips
        avg_all  = et.get_avg_slippage_pips(n=10)
        avg_last = et.get_avg_slippage_pips(n=5)
        assert avg_all < avg_last


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: ExecutionTracker – Vergleich mit Backtest
# ─────────────────────────────────────────────────────────────────────────────

class TestExecutionTrackerCompare:

    def _tracker_with_data(self) -> ExecutionTracker:
        et = ExecutionTracker(pip_size=0.0001)
        for i in range(5):
            et.record_slippage(i, "EURUSD", "buy", 1.0, 1.0003)  # 3 pips
            et.record_fees(i, "EURUSD", spread=0.0002, commission=0.0001)
        return et

    def test_compare_returns_dict_keys(self):
        et = self._tracker_with_data()
        result = et.compare_to_backtest(1.0, 0.0001)
        expected_keys = {
            "actual_avg_slippage_pips",
            "backtest_slippage_pips",
            "slippage_deviation_pct",
            "actual_avg_fees",
            "backtest_fees",
            "fees_deviation_pct",
            "n_slippage_trades",
            "n_fee_trades",
        }
        assert set(result.keys()) == expected_keys

    def test_compare_no_data_none_deviations(self):
        et = ExecutionTracker()
        result = et.compare_to_backtest(1.0, 0.0001)
        assert result["slippage_deviation_pct"] is None
        assert result["fees_deviation_pct"] is None
        assert result["n_slippage_trades"] == 0
        assert result["n_fee_trades"] == 0

    def test_compare_slippage_deviation(self):
        et = ExecutionTracker(pip_size=0.0001)
        for i in range(5):
            et.record_slippage(i, "X", "buy", 1.0, 1.0003)  # 3 pips
        result = et.compare_to_backtest(backtest_slippage_pips=1.0, backtest_fees=0.0)
        # deviation = (3 - 1) / 1 * 100 = 200%
        assert abs(result["slippage_deviation_pct"] - 200.0) < 0.01

    def test_compare_fees_deviation(self):
        et = ExecutionTracker()
        for i in range(3):
            et.record_fees(i, "X", spread=0.0003, commission=0.0)
        result = et.compare_to_backtest(backtest_slippage_pips=0.0, backtest_fees=0.0001)
        # deviation = (0.0003 - 0.0001) / 0.0001 * 100 = 200%
        assert abs(result["fees_deviation_pct"] - 200.0) < 0.01

    def test_compare_zero_backtest_slippage_no_deviation(self):
        et = ExecutionTracker(pip_size=0.0001)
        et.record_slippage(1, "X", "buy", 1.0, 1.0001)
        result = et.compare_to_backtest(backtest_slippage_pips=0.0, backtest_fees=0.0)
        assert result["slippage_deviation_pct"] is None

    def test_compare_n_counts(self):
        et = self._tracker_with_data()
        result = et.compare_to_backtest(1.0, 0.0001, n=3)
        assert result["n_slippage_trades"] == 3
        assert result["n_fee_trades"] == 3


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: ExecutionTracker – Deviation-Warnung
# ─────────────────────────────────────────────────────────────────────────────

class TestExecutionTrackerWarning:

    def test_no_data_returns_none(self):
        et = ExecutionTracker()
        assert et.get_deviation_warning(1.0, 0.0001) is None

    def test_within_threshold_returns_none(self):
        et = ExecutionTracker(pip_size=0.0001)
        for i in range(3):
            et.record_slippage(i, "X", "buy", 1.0, 1.0001)  # 1 pip
            et.record_fees(i, "X", spread=0.0001, commission=0.0)
        # backtest = 1 pip, actual = 1 pip -> 0% deviation
        result = et.get_deviation_warning(
            backtest_slippage_pips=1.0,
            backtest_fees=0.0001,
            threshold_pct=50.0,
        )
        assert result is None

    def test_slippage_exceeds_threshold(self):
        et = ExecutionTracker(pip_size=0.0001)
        for i in range(3):
            et.record_slippage(i, "X", "buy", 1.0, 1.0003)  # 3 pips
        warning = et.get_deviation_warning(
            backtest_slippage_pips=1.0,
            backtest_fees=0.0,
            threshold_pct=50.0,
        )
        assert warning is not None
        assert "EXECUTION WARNING" in warning
        assert "Slippage" in warning

    def test_fees_exceed_threshold(self):
        et = ExecutionTracker()
        for i in range(3):
            et.record_fees(i, "X", spread=0.0005, commission=0.0)
        warning = et.get_deviation_warning(
            backtest_slippage_pips=0.0,
            backtest_fees=0.0001,
            threshold_pct=50.0,
        )
        assert warning is not None
        assert "Fees" in warning

    def test_both_exceed_threshold(self):
        et = ExecutionTracker(pip_size=0.0001)
        for i in range(3):
            et.record_slippage(i, "X", "buy", 1.0, 1.0003)
            et.record_fees(i, "X", spread=0.0005, commission=0.0)
        warning = et.get_deviation_warning(1.0, 0.0001, threshold_pct=50.0)
        assert "Slippage" in warning
        assert "Fees" in warning

    def test_custom_threshold(self):
        et = ExecutionTracker(pip_size=0.0001)
        for i in range(3):
            et.record_slippage(i, "X", "buy", 1.0, 1.00015)  # 1.5 pips
        # 50% deviation over backtest of 1 pip; threshold=100%: no warning
        result_100 = et.get_deviation_warning(1.0, 0.0, threshold_pct=100.0)
        # threshold=10%: warning
        result_10 = et.get_deviation_warning(1.0, 0.0, threshold_pct=10.0)
        assert result_100 is None
        assert result_10 is not None


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: OrderExecutor – Limit-Orders (Paper)
# ─────────────────────────────────────────────────────────────────────────────

class TestPaperLimitOrder:

    def test_place_limit_returns_pending(self, tmp_path):
        ex = _paper_executor(tmp_path)
        result = ex.place_limit_order("EURUSD", "buy", 0.1, 1.09, 1.11, limit_price=1.095)
        assert result["status"] == "pending_limit"
        assert result["limit_price"] == 1.095
        assert result["ticket"] >= 1

    def test_place_limit_stored_in_paper_positions(self, tmp_path):
        ex = _paper_executor(tmp_path)
        result = ex.place_limit_order("EURUSD", "sell", 0.1, 1.11, 1.09, limit_price=1.105)
        ticket = result["ticket"]
        assert ticket in ex._paper_positions
        assert ex._paper_positions[ticket]["status"] == "pending_limit"

    def test_place_limit_sell_direction(self, tmp_path):
        ex = _paper_executor(tmp_path)
        result = ex.place_limit_order("EURUSD", "sell", 0.2, 1.11, 1.09, limit_price=1.105)
        assert result["direction"] == "sell"

    def test_place_limit_invalid_direction(self, tmp_path):
        ex = _paper_executor(tmp_path)
        with pytest.raises(ValueError, match="direction"):
            ex.place_limit_order("EURUSD", "hold", 0.1, 1.09, 1.11, limit_price=1.095)

    def test_place_limit_invalid_lot_size(self, tmp_path):
        ex = _paper_executor(tmp_path)
        with pytest.raises(ValueError, match="lot_size"):
            ex.place_limit_order("EURUSD", "buy", 0.0, 1.09, 1.11, limit_price=1.095)

    def test_place_limit_with_timeout_stores_deadline(self, tmp_path):
        ex = _paper_executor(tmp_path)
        before = datetime.now(timezone.utc)
        result = ex.place_limit_order("EURUSD", "buy", 0.1, 1.09, 1.11,
                                      limit_price=1.095, timeout_s=3600)
        ticket = result["ticket"]
        deadline_str = ex._paper_positions[ticket]["timeout_deadline"]
        assert deadline_str is not None
        deadline = datetime.fromisoformat(deadline_str)
        # Deadline should be ~1 hour from now
        diff = (deadline - before).total_seconds()
        assert 3595 < diff < 3605

    def test_place_limit_no_timeout_no_deadline(self, tmp_path):
        ex = _paper_executor(tmp_path)
        result = ex.place_limit_order("EURUSD", "buy", 0.1, 1.09, 1.11, limit_price=1.095)
        ticket = result["ticket"]
        assert ex._paper_positions[ticket]["timeout_deadline"] is None

    def test_cancel_limit_order(self, tmp_path):
        ex = _paper_executor(tmp_path)
        result = ex.place_limit_order("EURUSD", "buy", 0.1, 1.09, 1.11, limit_price=1.095)
        ticket = result["ticket"]
        cancelled = ex.cancel_limit_order(ticket)
        assert cancelled["status"] == "cancelled"
        assert ex._paper_positions[ticket]["status"] == "cancelled"

    def test_cancel_nonexistent_ticket_raises(self, tmp_path):
        ex = _paper_executor(tmp_path)
        with pytest.raises(OrderError):
            ex.cancel_limit_order(999)

    def test_cancel_already_cancelled_raises(self, tmp_path):
        ex = _paper_executor(tmp_path)
        result = ex.place_limit_order("EURUSD", "buy", 0.1, 1.09, 1.11, limit_price=1.095)
        ticket = result["ticket"]
        ex.cancel_limit_order(ticket)
        with pytest.raises(OrderError):
            ex.cancel_limit_order(ticket)

    def test_cancel_open_position_raises(self, tmp_path):
        ex = _paper_executor(tmp_path)
        result = ex.open_position("EURUSD", "buy", 0.1, 1.09, 1.11)
        with pytest.raises(OrderError):
            ex.cancel_limit_order(result["ticket"])

    def test_limit_order_not_in_open_positions(self, tmp_path):
        ex = _paper_executor(tmp_path)
        ex.place_limit_order("EURUSD", "buy", 0.1, 1.09, 1.11, limit_price=1.095)
        assert ex.get_open_positions() == []

    def test_multiple_limit_orders_unique_tickets(self, tmp_path):
        ex = _paper_executor(tmp_path)
        r1 = ex.place_limit_order("EURUSD", "buy", 0.1, 1.09, 1.11, limit_price=1.095)
        r2 = ex.place_limit_order("EURUSD", "sell", 0.1, 1.11, 1.09, limit_price=1.105)
        assert r1["ticket"] != r2["ticket"]

    def test_limit_order_written_to_json(self, tmp_path):
        ex = _paper_executor(tmp_path)
        ex.place_limit_order("EURUSD", "buy", 0.1, 1.09, 1.11, limit_price=1.095)
        import json
        data = json.loads((tmp_path / "paper_trades.json").read_text())
        assert any(t["status"] == "pending_limit" for t in data)


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: OrderExecutor – Limit-Order-Timeout (Paper)
# ─────────────────────────────────────────────────────────────────────────────

class TestLimitOrderTimeout:

    def test_expire_overdue_limit_order(self, tmp_path):
        ex = _paper_executor(tmp_path)
        result = ex.place_limit_order("EURUSD", "buy", 0.1, 1.09, 1.11,
                                      limit_price=1.095, timeout_s=60)
        ticket = result["ticket"]
        # Simulate time passing: use a future timestamp
        future = datetime.now(timezone.utc) + timedelta(seconds=120)
        cancelled = ex.check_and_expire_limit_orders(current_time=future)
        assert ticket in cancelled
        assert ex._paper_positions[ticket]["status"] == "cancelled"

    def test_not_expired_yet_not_cancelled(self, tmp_path):
        ex = _paper_executor(tmp_path)
        ex.place_limit_order("EURUSD", "buy", 0.1, 1.09, 1.11,
                             limit_price=1.095, timeout_s=3600)
        # Only 1 second has passed
        soon = datetime.now(timezone.utc) + timedelta(seconds=1)
        cancelled = ex.check_and_expire_limit_orders(current_time=soon)
        assert cancelled == []

    def test_no_timeout_never_expired(self, tmp_path):
        ex = _paper_executor(tmp_path)
        ex.place_limit_order("EURUSD", "buy", 0.1, 1.09, 1.11, limit_price=1.095)
        far_future = datetime.now(timezone.utc) + timedelta(days=365)
        cancelled = ex.check_and_expire_limit_orders(current_time=far_future)
        assert cancelled == []

    def test_multiple_orders_only_expired_cancelled(self, tmp_path):
        ex = _paper_executor(tmp_path)
        r1 = ex.place_limit_order("EURUSD", "buy", 0.1, 1.09, 1.11,
                                   limit_price=1.095, timeout_s=30)
        r2 = ex.place_limit_order("EURUSD", "buy", 0.1, 1.09, 1.11,
                                   limit_price=1.094, timeout_s=3600)
        future = datetime.now(timezone.utc) + timedelta(seconds=60)
        cancelled = ex.check_and_expire_limit_orders(current_time=future)
        assert r1["ticket"] in cancelled
        assert r2["ticket"] not in cancelled
        assert ex._paper_positions[r2["ticket"]]["status"] == "pending_limit"

    def test_already_cancelled_order_ignored(self, tmp_path):
        ex = _paper_executor(tmp_path)
        result = ex.place_limit_order("EURUSD", "buy", 0.1, 1.09, 1.11,
                                      limit_price=1.095, timeout_s=30)
        ticket = result["ticket"]
        ex.cancel_limit_order(ticket)
        # No error, just returns empty list
        future = datetime.now(timezone.utc) + timedelta(seconds=60)
        cancelled = ex.check_and_expire_limit_orders(current_time=future)
        assert ticket not in cancelled

    def test_default_current_time_uses_utc_now(self, tmp_path):
        ex = _paper_executor(tmp_path)
        # Order with 1 ms timeout (already expired)
        ex.place_limit_order("EURUSD", "buy", 0.1, 1.09, 1.11,
                             limit_price=1.095, timeout_s=0.001)
        import time
        time.sleep(0.01)
        cancelled = ex.check_and_expire_limit_orders()  # no current_time arg
        assert len(cancelled) == 1


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: OrderExecutor – Partial-Fill-Handling (Live)
# ─────────────────────────────────────────────────────────────────────────────

class TestPartialFillHandling:

    def test_partial_fill_recorded_in_lot_size(self, tmp_path):
        mt5 = _mt5_mock(ticket=7, fill_price=1.10000, filled_volume=0.05)
        with patch.dict(os.environ, {"CONFIRM_LIVE": "yes"}):
            with patch("src.execution.order_executor._load_mt5", return_value=mt5):
                ex = _live_executor(tmp_path)
                result = ex.open_position("EURUSD", "buy", 0.1, 1.09, 1.11)
        assert result["lot_size"] == 0.05
        assert result["requested_lots"] == 0.1
        assert result["partial_fill"] is True

    def test_full_fill_no_partial_fill_flag(self, tmp_path):
        mt5 = _mt5_mock(ticket=8, fill_price=1.10000, filled_volume=0.1)
        with patch.dict(os.environ, {"CONFIRM_LIVE": "yes"}):
            with patch("src.execution.order_executor._load_mt5", return_value=mt5):
                ex = _live_executor(tmp_path)
                result = ex.open_position("EURUSD", "buy", 0.1, 1.09, 1.11)
        assert result["partial_fill"] is False

    def test_partial_fill_logged_as_warning(self, tmp_path):
        mt5 = _mt5_mock(ticket=9, fill_price=1.10000, filled_volume=0.03)
        with patch.dict(os.environ, {"CONFIRM_LIVE": "yes"}):
            with patch("src.execution.order_executor._load_mt5", return_value=mt5):
                with patch("src.execution.order_executor.logger") as mock_logger:
                    ex = _live_executor(tmp_path)
                    ex.open_position("EURUSD", "buy", 0.1, 1.09, 1.11)
        warning_calls = [str(c) for c in mock_logger.warning.call_args_list]
        assert any("Partial" in c or "partial" in c.lower() for c in warning_calls)

    def test_partial_fill_journal_uses_filled_volume(self, tmp_path):
        mt5 = _mt5_mock(ticket=10, fill_price=1.10000, filled_volume=0.07)
        journal = MagicMock()
        journal.log_trade_open.return_value = 1
        with patch.dict(os.environ, {"CONFIRM_LIVE": "yes"}):
            with patch("src.execution.order_executor._load_mt5", return_value=mt5):
                ex = _live_executor(tmp_path, trade_journal=journal)
                ex.open_position("EURUSD", "buy", 0.1, 1.09, 1.11)
        call_kwargs = journal.log_trade_open.call_args[0][0]
        assert call_kwargs["lot_size"] == 0.07


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: OrderExecutor – Slippage/Fee-Tracking (Live)
# ─────────────────────────────────────────────────────────────────────────────

class TestLiveSlippageFeeTracking:

    def test_slippage_recorded_on_live_open(self, tmp_path):
        mt5 = _mt5_mock(ticket=1, fill_price=1.10005)
        et = ExecutionTracker(pip_size=0.0001)
        with patch.dict(os.environ, {"CONFIRM_LIVE": "yes"}):
            with patch("src.execution.order_executor._load_mt5", return_value=mt5):
                ex = _live_executor(tmp_path, execution_tracker=et)
                ex.open_position("EURUSD", "buy", 0.1, 1.09, 1.11)
        records = et.get_slippage_records()
        assert len(records) == 1
        assert records[0].actual_price == 1.10005

    def test_fees_recorded_on_live_open(self, tmp_path):
        mt5 = _mt5_mock(ticket=2, fill_price=1.10000)
        et = ExecutionTracker(pip_size=0.0001)
        with patch.dict(os.environ, {"CONFIRM_LIVE": "yes"}):
            with patch("src.execution.order_executor._load_mt5", return_value=mt5):
                ex = _live_executor(tmp_path, execution_tracker=et)
                ex.open_position("EURUSD", "buy", 0.1, 1.09, 1.11)
        records = et.get_fee_records()
        assert len(records) == 1
        # spread = 10 points * 0.0001 = 0.001
        assert abs(records[0].spread - 0.001) < 1e-9

    def test_no_tracker_no_error(self, tmp_path):
        mt5 = _mt5_mock(ticket=3, fill_price=1.10000)
        with patch.dict(os.environ, {"CONFIRM_LIVE": "yes"}):
            with patch("src.execution.order_executor._load_mt5", return_value=mt5):
                ex = _live_executor(tmp_path)  # no execution_tracker
                result = ex.open_position("EURUSD", "buy", 0.1, 1.09, 1.11)
        assert result["status"] == "open"

    def test_slippage_uses_expected_price_when_provided(self, tmp_path):
        mt5 = _mt5_mock(ticket=4, fill_price=1.10010)
        et = ExecutionTracker(pip_size=0.0001)
        with patch.dict(os.environ, {"CONFIRM_LIVE": "yes"}):
            with patch("src.execution.order_executor._load_mt5", return_value=mt5):
                ex = _live_executor(tmp_path, execution_tracker=et)
                # Call _open_live directly with expected_price
                ex._open_live("EURUSD", "buy", 0.1, 1.09, 1.11,
                              expected_price=1.10000)
        record = et.get_slippage_records()[0]
        assert record.expected_price == 1.10000
        assert record.actual_price   == 1.10010
        assert abs(record.slippage_pips - 1.0) < 1e-9

    def test_connector_info_error_does_not_break_trade(self, tmp_path):
        mt5 = _mt5_mock(ticket=5, fill_price=1.10000)
        conn = _connector()
        conn.get_symbol_info.side_effect = RuntimeError("info nicht verfuegbar")
        et = ExecutionTracker(pip_size=0.0001)
        with patch.dict(os.environ, {"CONFIRM_LIVE": "yes"}):
            with patch("src.execution.order_executor._load_mt5", return_value=mt5):
                ex = OrderExecutor(
                    connector=conn,
                    live_trading_enabled=True,
                    paper_trades_path=tmp_path / "pt.json",
                    execution_tracker=et,
                )
                result = ex.open_position("EURUSD", "buy", 0.1, 1.09, 1.11)
        assert result["status"] == "open"


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: OrderExecutor – Limit-Orders (Live)
# ─────────────────────────────────────────────────────────────────────────────

class TestLiveLimitOrder:

    def test_place_live_limit_buy(self, tmp_path):
        mt5 = _mt5_mock(ticket=20)
        with patch.dict(os.environ, {"CONFIRM_LIVE": "yes"}):
            with patch("src.execution.order_executor._load_mt5", return_value=mt5):
                ex = _live_executor(tmp_path)
                result = ex.place_limit_order("EURUSD", "buy", 0.1, 1.09, 1.11,
                                              limit_price=1.095)
        assert result["status"] == "pending_limit"
        assert result["ticket"] == 20
        # Verify TRADE_ACTION_PENDING was used
        call_args = mt5.order_send.call_args[0][0]
        assert call_args["action"] == mt5.TRADE_ACTION_PENDING
        assert call_args["type"] == mt5.ORDER_TYPE_BUY_LIMIT

    def test_place_live_limit_sell(self, tmp_path):
        mt5 = _mt5_mock(ticket=21)
        with patch.dict(os.environ, {"CONFIRM_LIVE": "yes"}):
            with patch("src.execution.order_executor._load_mt5", return_value=mt5):
                ex = _live_executor(tmp_path)
                result = ex.place_limit_order("EURUSD", "sell", 0.1, 1.11, 1.09,
                                              limit_price=1.105)
        call_args = mt5.order_send.call_args[0][0]
        assert call_args["type"] == mt5.ORDER_TYPE_SELL_LIMIT

    def test_place_live_limit_with_timeout_sets_expiration(self, tmp_path):
        mt5 = _mt5_mock(ticket=22)
        with patch.dict(os.environ, {"CONFIRM_LIVE": "yes"}):
            with patch("src.execution.order_executor._load_mt5", return_value=mt5):
                ex = _live_executor(tmp_path)
                ex.place_limit_order("EURUSD", "buy", 0.1, 1.09, 1.11,
                                     limit_price=1.095, timeout_s=3600)
        call_args = mt5.order_send.call_args[0][0]
        assert "expiration" in call_args

    def test_place_live_limit_mt5_rejected_raises(self, tmp_path):
        mt5 = _mt5_mock()
        mt5.order_send.return_value.retcode = 10004  # reject
        with patch.dict(os.environ, {"CONFIRM_LIVE": "yes"}):
            with patch("src.execution.order_executor._load_mt5", return_value=mt5):
                ex = _live_executor(tmp_path)
                with pytest.raises(OrderError, match="place_limit_order abgelehnt"):
                    ex.place_limit_order("EURUSD", "buy", 0.1, 1.09, 1.11,
                                         limit_price=1.095)

    def test_cancel_live_limit(self, tmp_path):
        mt5 = _mt5_mock(ticket=30)
        with patch.dict(os.environ, {"CONFIRM_LIVE": "yes"}):
            with patch("src.execution.order_executor._load_mt5", return_value=mt5):
                ex = _live_executor(tmp_path)
                result = ex.cancel_limit_order(30)
        assert result["status"] == "cancelled"
        call_args = mt5.order_send.call_args[0][0]
        assert call_args["action"] == mt5.TRADE_ACTION_REMOVE

    def test_cancel_live_limit_not_found_raises(self, tmp_path):
        mt5 = _mt5_mock()
        mt5.orders_get.return_value = []  # no pending order found
        with patch.dict(os.environ, {"CONFIRM_LIVE": "yes"}):
            with patch("src.execution.order_executor._load_mt5", return_value=mt5):
                ex = _live_executor(tmp_path)
                with pytest.raises(OrderError, match="nicht gefunden"):
                    ex.cancel_limit_order(999)

    def test_cancel_live_limit_mt5_rejected_raises(self, tmp_path):
        mt5 = _mt5_mock(ticket=31)
        ok_result = mt5.order_send.return_value
        # First call (if any) succeeds, cancel call fails
        fail_result = MagicMock()
        fail_result.retcode  = 10004
        fail_result.comment  = "rejected"
        mt5.order_send.return_value = fail_result
        with patch.dict(os.environ, {"CONFIRM_LIVE": "yes"}):
            with patch("src.execution.order_executor._load_mt5", return_value=mt5):
                ex = _live_executor(tmp_path)
                with pytest.raises(OrderError, match="cancel_limit_order abgelehnt"):
                    ex.cancel_limit_order(31)

    def test_live_not_connected_place_limit_raises(self, tmp_path):
        mt5 = _mt5_mock()
        conn = _connector(connected=False)
        with patch.dict(os.environ, {"CONFIRM_LIVE": "yes"}):
            with patch("src.execution.order_executor._load_mt5", return_value=mt5):
                ex = OrderExecutor(
                    connector=conn,
                    live_trading_enabled=True,
                    paper_trades_path=tmp_path / "pt.json",
                )
                with pytest.raises(OrderError, match="verbunden"):
                    ex.place_limit_order("EURUSD", "buy", 0.1, 1.09, 1.11,
                                         limit_price=1.095)

    def test_live_not_connected_cancel_limit_raises(self, tmp_path):
        mt5 = _mt5_mock()
        conn = _connector(connected=False)
        with patch.dict(os.environ, {"CONFIRM_LIVE": "yes"}):
            with patch("src.execution.order_executor._load_mt5", return_value=mt5):
                ex = OrderExecutor(
                    connector=conn,
                    live_trading_enabled=True,
                    paper_trades_path=tmp_path / "pt.json",
                )
                with pytest.raises(OrderError, match="verbunden"):
                    ex.cancel_limit_order(1)
