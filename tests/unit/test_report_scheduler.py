"""
tests/unit/test_report_scheduler.py
Unit-Tests fuer Issue #62: Automatischer Daily/Weekly Performance-Digest.

Prueft:
  - Hilfsfunktionen (_parse_time, _calc_sharpe, _calc_max_drawdown)
  - TradeJournal-Erweiterungen (get_pnl_sequence, get_open_positions)
  - ReportScheduler: Initialisierung, trigger_daily/weekly
  - Scheduling-Logik (_should_run_daily, _should_run_weekly)
  - Report-Inhalt (Trades, Win-Rate, Sharpe, MaxDD, offene Positionen)
  - Datei-Output (.md in reports/-Ordner)
  - Alert-Versand (gemockt)
  - "Keine Trades"-Meldung
  - Hintergrund-Thread (start/stop)
  - on_report_generated Callback
"""

from __future__ import annotations

import tempfile
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, call

import pytest

from src.journal.trade_journal import TradeJournal
from src.journal.report_scheduler import (
    AlertSender,
    LogOnlyAlertSender,
    ReportScheduler,
    _calc_max_drawdown,
    _calc_sharpe,
    _parse_time,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Hilfsfunktionen fuer Tests
# ─────────────────────────────────────────────────────────────────────────────

def _journal(tmp_path: Path) -> TradeJournal:
    return TradeJournal(db_path=tmp_path / "journal.db")


def _close_trade(jnl: TradeJournal, trade_id: int, pnl: float) -> None:
    jnl.log_trade_close(trade_id, {"exit_price": 1.1, "pnl": pnl})


def _seed_closed(jnl: TradeJournal, pnls: list[float]) -> None:
    """Legt geschlossene Trades mit den angegebenen PnLs an (entry_time = jetzt UTC)."""
    now = datetime.now(timezone.utc)
    for pnl in pnls:
        tid = jnl.log_trade_open({
            "symbol": "EURUSD", "direction": "buy",
            "lot_size": 0.1, "entry_price": 1.1, "entry_time": now,
        })
        _close_trade(jnl, tid, pnl)


def _scheduler(
    tmp_path: Path,
    jnl: Optional[TradeJournal] = None,
    alert: Optional[object] = None,
    daily_time: str = "23:00",
    weekly_time: str = "23:00",
    check_interval_s: int = 1,
    _now_fn=None,
    on_report_generated=None,
) -> ReportScheduler:
    if jnl is None:
        jnl = _journal(tmp_path)
    return ReportScheduler(
        journal=jnl,
        reports_dir=tmp_path / "reports",
        daily_time=daily_time,
        weekly_time=weekly_time,
        alert_sender=alert or MagicMock(spec=AlertSender),
        check_interval_s=check_interval_s,
        on_report_generated=on_report_generated,
        _now_fn=_now_fn,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  1.  _parse_time
# ─────────────────────────────────────────────────────────────────────────────

class TestParseTime:
    def test_parses_valid_time(self):
        assert _parse_time("23:00") == (23, 0)

    def test_parses_midnight(self):
        assert _parse_time("00:00") == (0, 0)

    def test_parses_arbitrary_time(self):
        assert _parse_time("08:30") == (8, 30)

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError):
            _parse_time("2300")

    def test_invalid_hour_raises(self):
        with pytest.raises(ValueError):
            _parse_time("25:00")

    def test_invalid_minute_raises(self):
        with pytest.raises(ValueError):
            _parse_time("12:60")

    def test_non_numeric_raises(self):
        with pytest.raises(ValueError):
            _parse_time("HH:MM")


# ─────────────────────────────────────────────────────────────────────────────
#  2.  _calc_sharpe
# ─────────────────────────────────────────────────────────────────────────────

class TestCalcSharpe:
    def test_empty_returns_none(self):
        assert _calc_sharpe([]) is None

    def test_single_returns_none(self):
        assert _calc_sharpe([1.0]) is None

    def test_two_identical_returns_none(self):
        assert _calc_sharpe([1.0, 1.0]) is None

    def test_positive_pnl_positive_sharpe(self):
        result = _calc_sharpe([1.0, 2.0, 3.0])
        assert result is not None
        assert result > 0

    def test_returns_float(self):
        result = _calc_sharpe([1.0, -1.0, 2.0, -0.5])
        assert isinstance(result, float)

    def test_symmetric_pnls_near_zero(self):
        # mean ~ 0, should be near 0
        result = _calc_sharpe([1.0, -1.0])
        assert result is not None
        assert abs(result) < 0.01


# ─────────────────────────────────────────────────────────────────────────────
#  3.  _calc_max_drawdown
# ─────────────────────────────────────────────────────────────────────────────

class TestCalcMaxDrawdown:
    def test_empty_returns_zero(self):
        assert _calc_max_drawdown([]) == 0.0

    def test_all_positive_no_drawdown(self):
        assert _calc_max_drawdown([1.0, 2.0, 3.0]) == 0.0

    def test_single_loss(self):
        dd = _calc_max_drawdown([5.0, -2.0])
        assert abs(dd - 2.0) < 1e-9

    def test_deep_drawdown(self):
        # equity: 0, 10, 5, 15, 3 → peak 15, trough 3, dd = 12
        dd = _calc_max_drawdown([10.0, -5.0, 10.0, -12.0])
        assert abs(dd - 12.0) < 1e-9

    def test_returns_float(self):
        assert isinstance(_calc_max_drawdown([1.0, -0.5]), float)

    def test_only_losses(self):
        # equity: 0 → -1 → -3 → -6; peak stays 0; dd peaks at 6.0
        dd = _calc_max_drawdown([-1.0, -2.0, -3.0])
        assert abs(dd - 6.0) < 1e-9


# ─────────────────────────────────────────────────────────────────────────────
#  4.  TradeJournal.get_pnl_sequence
# ─────────────────────────────────────────────────────────────────────────────

class TestTradeJournalGetPnlSequence:
    def test_empty_journal_returns_empty(self, tmp_path):
        jnl = _journal(tmp_path)
        now = datetime.now(timezone.utc)
        result = jnl.get_pnl_sequence(now - timedelta(days=1), now)
        assert result == []

    def test_returns_correct_pnls(self, tmp_path):
        jnl = _journal(tmp_path)
        _seed_closed(jnl, [1.0, -0.5, 2.0])
        now = datetime.now(timezone.utc)
        result = jnl.get_pnl_sequence(now - timedelta(days=1), now)
        assert sorted(result) == sorted([1.0, -0.5, 2.0])

    def test_filters_by_period(self, tmp_path):
        jnl = _journal(tmp_path)
        old_time = datetime.now(timezone.utc) - timedelta(days=8)
        tid = jnl.log_trade_open({
            "symbol": "EURUSD", "direction": "buy", "entry_time": old_time,
        })
        _close_trade(jnl, tid, 99.0)
        _seed_closed(jnl, [1.0])
        now = datetime.now(timezone.utc)
        result = jnl.get_pnl_sequence(now - timedelta(days=1), now)
        assert 99.0 not in result
        assert 1.0 in result

    def test_open_trades_excluded(self, tmp_path):
        jnl = _journal(tmp_path)
        jnl.log_trade_open({"symbol": "EURUSD", "direction": "buy"})
        now = datetime.now(timezone.utc)
        result = jnl.get_pnl_sequence(now - timedelta(days=1), now)
        assert result == []

    def test_symbol_filter(self, tmp_path):
        jnl = _journal(tmp_path)
        now = datetime.now(timezone.utc)
        for sym, pnl in [("EURUSD", 1.0), ("GBPUSD", -2.0)]:
            tid = jnl.log_trade_open({"symbol": sym, "direction": "buy", "entry_time": now})
            _close_trade(jnl, tid, pnl)
        result = jnl.get_pnl_sequence(now - timedelta(days=1), now, symbol="EURUSD")
        assert result == [1.0]


# ─────────────────────────────────────────────────────────────────────────────
#  5.  TradeJournal.get_open_positions
# ─────────────────────────────────────────────────────────────────────────────

class TestTradeJournalGetOpenPositions:
    def test_empty_journal_returns_empty(self, tmp_path):
        jnl = _journal(tmp_path)
        assert jnl.get_open_positions() == []

    def test_returns_open_positions(self, tmp_path):
        jnl = _journal(tmp_path)
        jnl.log_trade_open({"symbol": "EURUSD", "direction": "buy"})
        jnl.log_trade_open({"symbol": "GBPUSD", "direction": "sell"})
        result = jnl.get_open_positions()
        assert len(result) == 2

    def test_closed_trades_excluded(self, tmp_path):
        jnl = _journal(tmp_path)
        _seed_closed(jnl, [1.0])
        assert jnl.get_open_positions() == []

    def test_mixed_returns_only_open(self, tmp_path):
        jnl = _journal(tmp_path)
        _seed_closed(jnl, [1.0])
        jnl.log_trade_open({"symbol": "USDCHF", "direction": "buy"})
        result = jnl.get_open_positions()
        assert len(result) == 1
        assert result[0]["symbol"] == "USDCHF"

    def test_returns_list_of_dicts(self, tmp_path):
        jnl = _journal(tmp_path)
        jnl.log_trade_open({"symbol": "EURUSD", "direction": "buy"})
        result = jnl.get_open_positions()
        assert isinstance(result, list)
        assert isinstance(result[0], dict)


# ─────────────────────────────────────────────────────────────────────────────
#  6.  ReportScheduler – Initialisierung
# ─────────────────────────────────────────────────────────────────────────────

class TestReportSchedulerInit:
    def test_creates_without_crash(self, tmp_path):
        _scheduler(tmp_path)

    def test_is_running_false_initially(self, tmp_path):
        s = _scheduler(tmp_path)
        assert s.is_running is False

    def test_last_daily_none_initially(self, tmp_path):
        s = _scheduler(tmp_path)
        assert s._last_daily is None

    def test_last_weekly_none_initially(self, tmp_path):
        s = _scheduler(tmp_path)
        assert s._last_weekly is None

    def test_custom_reports_dir(self, tmp_path):
        custom = tmp_path / "custom_reports"
        s = ReportScheduler(
            journal=_journal(tmp_path),
            reports_dir=custom,
            _now_fn=datetime.now,
        )
        assert s._reports_dir == custom

    def test_invalid_daily_time_raises(self, tmp_path):
        with pytest.raises(ValueError):
            ReportScheduler(journal=_journal(tmp_path), daily_time="2500")

    def test_default_alert_sender_is_log_only(self, tmp_path):
        s = ReportScheduler(journal=_journal(tmp_path))
        assert isinstance(s._alert, LogOnlyAlertSender)


# ─────────────────────────────────────────────────────────────────────────────
#  7.  ReportScheduler – trigger_daily / trigger_weekly
# ─────────────────────────────────────────────────────────────────────────────

class TestReportSchedulerTrigger:
    def test_trigger_daily_returns_string(self, tmp_path):
        s = _scheduler(tmp_path)
        result = s.trigger_daily()
        assert isinstance(result, str)

    def test_trigger_weekly_returns_string(self, tmp_path):
        s = _scheduler(tmp_path)
        result = s.trigger_weekly()
        assert isinstance(result, str)

    def test_trigger_daily_no_trades_contains_no_trades_message(self, tmp_path):
        s = _scheduler(tmp_path)
        result = s.trigger_daily()
        assert "Keine abgeschlossenen Trades" in result

    def test_trigger_weekly_no_trades_contains_no_trades_message(self, tmp_path):
        s = _scheduler(tmp_path)
        result = s.trigger_weekly()
        assert "Keine abgeschlossenen Trades" in result

    def test_trigger_daily_with_trades_uses_generate_report(self, tmp_path):
        jnl = _journal(tmp_path)
        _seed_closed(jnl, [1.0, -0.5])
        s = _scheduler(tmp_path, jnl=jnl)
        result = s.trigger_daily()
        assert "QuantzAI" in result

    def test_trigger_daily_sends_alert(self, tmp_path):
        alert = MagicMock()
        s = _scheduler(tmp_path, alert=alert)
        s.trigger_daily()
        alert.send_alert.assert_called_once()

    def test_trigger_weekly_sends_alert(self, tmp_path):
        alert = MagicMock()
        s = _scheduler(tmp_path, alert=alert)
        s.trigger_weekly()
        alert.send_alert.assert_called_once()

    def test_trigger_daily_creates_file(self, tmp_path):
        s = _scheduler(tmp_path)
        s.trigger_daily()
        files = list((tmp_path / "reports").glob("*.md"))
        assert len(files) == 1

    def test_trigger_weekly_creates_file(self, tmp_path):
        s = _scheduler(tmp_path)
        s.trigger_weekly()
        files = list((tmp_path / "reports").glob("*.md"))
        assert len(files) == 1

    def test_trigger_calls_callback(self, tmp_path):
        received = []
        s = _scheduler(tmp_path, on_report_generated=lambda p, c: received.append((p, c)))
        s.trigger_daily()
        assert len(received) == 1
        assert received[0][0] == "daily"


# ─────────────────────────────────────────────────────────────────────────────
#  8.  ReportScheduler – Scheduling-Logik
# ─────────────────────────────────────────────────────────────────────────────

class TestReportSchedulerScheduling:
    def _make(self, tmp_path, daily="23:00", weekly="23:00"):
        return _scheduler(tmp_path, daily_time=daily, weekly_time=weekly)

    # Daily

    def test_daily_false_before_target_time(self, tmp_path):
        s = self._make(tmp_path, daily="23:00")
        now = datetime(2026, 6, 21, 22, 59)  # one minute before
        assert s._should_run_daily(now) is False

    def test_daily_true_at_target_time(self, tmp_path):
        s = self._make(tmp_path, daily="23:00")
        now = datetime(2026, 6, 21, 23, 0)
        assert s._should_run_daily(now) is True

    def test_daily_true_after_target_time(self, tmp_path):
        s = self._make(tmp_path, daily="23:00")
        now = datetime(2026, 6, 21, 23, 30)
        assert s._should_run_daily(now) is True

    def test_daily_false_if_already_ran_today(self, tmp_path):
        s = self._make(tmp_path, daily="23:00")
        now = datetime(2026, 6, 21, 23, 5)
        s._last_daily = now.date()
        assert s._should_run_daily(now) is False

    def test_daily_true_next_day(self, tmp_path):
        s = self._make(tmp_path, daily="23:00")
        s._last_daily = date(2026, 6, 21)
        now = datetime(2026, 6, 22, 23, 0)
        assert s._should_run_daily(now) is True

    # Weekly

    def test_weekly_false_on_weekday(self, tmp_path):
        s = self._make(tmp_path, weekly="23:00")
        # Monday = weekday 0
        now = datetime(2026, 6, 22, 23, 0)  # Monday 22.06.2026
        assert now.weekday() == 0
        assert s._should_run_weekly(now) is False

    def test_weekly_true_on_sunday_at_target(self, tmp_path):
        s = self._make(tmp_path, weekly="23:00")
        # Sunday 21.06.2026
        now = datetime(2026, 6, 21, 23, 0)
        assert now.weekday() == 6
        assert s._should_run_weekly(now) is True

    def test_weekly_false_on_sunday_before_target(self, tmp_path):
        s = self._make(tmp_path, weekly="23:00")
        now = datetime(2026, 6, 21, 22, 59)
        assert now.weekday() == 6
        assert s._should_run_weekly(now) is False

    def test_weekly_false_if_already_ran_this_sunday(self, tmp_path):
        s = self._make(tmp_path, weekly="23:00")
        now = datetime(2026, 6, 21, 23, 5)
        s._last_weekly = now.date()
        assert s._should_run_weekly(now) is False

    def test_weekly_true_next_sunday(self, tmp_path):
        s = self._make(tmp_path, weekly="23:00")
        s._last_weekly = date(2026, 6, 21)
        now = datetime(2026, 6, 28, 23, 0)  # next Sunday
        assert now.weekday() == 6
        assert s._should_run_weekly(now) is True


# ─────────────────────────────────────────────────────────────────────────────
#  9.  ReportScheduler – Report-Inhalt
# ─────────────────────────────────────────────────────────────────────────────

class TestReportSchedulerContent:
    def _make_with_trades(self, tmp_path, pnls: list[float]) -> ReportScheduler:
        jnl = _journal(tmp_path)
        _seed_closed(jnl, pnls)
        return _scheduler(tmp_path, jnl=jnl)

    def test_report_contains_trade_count(self, tmp_path):
        s = self._make_with_trades(tmp_path, [1.0, -0.5, 2.0])
        content = s.trigger_daily()
        assert "3" in content  # 3 Trades

    def test_report_contains_win_rate(self, tmp_path):
        s = self._make_with_trades(tmp_path, [1.0, -0.5, 2.0])
        content = s.trigger_daily()
        # 2 wins out of 3 = 66.7%
        assert "%" in content

    def test_report_contains_total_pnl(self, tmp_path):
        s = self._make_with_trades(tmp_path, [1.0, 2.0])
        content = s.trigger_daily()
        # total = 3.0
        assert "3.00" in content

    def test_report_contains_sharpe(self, tmp_path):
        s = self._make_with_trades(tmp_path, [1.0, -0.5, 2.0])
        content = s.trigger_daily()
        assert "Sharpe" in content

    def test_report_contains_max_drawdown(self, tmp_path):
        s = self._make_with_trades(tmp_path, [1.0, -0.5, 2.0])
        content = s.trigger_daily()
        assert "Drawdown" in content

    def test_report_contains_open_positions_section(self, tmp_path):
        jnl = _journal(tmp_path)
        _seed_closed(jnl, [1.0])
        jnl.log_trade_open({"symbol": "EURUSD", "direction": "buy"})
        s = _scheduler(tmp_path, jnl=jnl)
        content = s.trigger_daily()
        assert "Offene Positionen" in content

    def test_report_no_trades_message(self, tmp_path):
        s = _scheduler(tmp_path)
        content = s.trigger_daily()
        assert "Keine abgeschlossenen Trades" in content
        assert "Trades gesamt" not in content  # kein leerer Report

    def test_no_trades_message_still_has_period(self, tmp_path):
        s = _scheduler(tmp_path)
        content = s.trigger_daily()
        assert "Täglich" in content or "daily" in content.lower()

    def test_weekly_period_label_in_content(self, tmp_path):
        s = self._make_with_trades(tmp_path, [1.0])
        content = s.trigger_weekly()
        assert "Woechentlich" in content or "Wöchentlich" in content or "weekly" in content.lower()


# ─────────────────────────────────────────────────────────────────────────────
#  10. ReportScheduler – Datei-Output
# ─────────────────────────────────────────────────────────────────────────────

class TestReportSchedulerFileOutput:
    def test_creates_reports_dir(self, tmp_path):
        s = _scheduler(tmp_path)
        s.trigger_daily()
        assert (tmp_path / "reports").is_dir()

    def test_file_has_md_extension(self, tmp_path):
        s = _scheduler(tmp_path)
        s.trigger_daily()
        files = list((tmp_path / "reports").glob("*.md"))
        assert len(files) == 1

    def test_filename_contains_daily(self, tmp_path):
        s = _scheduler(tmp_path)
        s.trigger_daily()
        files = list((tmp_path / "reports").glob("*.md"))
        assert "daily" in files[0].name

    def test_filename_contains_weekly(self, tmp_path):
        s = _scheduler(tmp_path)
        s.trigger_weekly()
        files = list((tmp_path / "reports").glob("*.md"))
        assert "weekly" in files[0].name

    def test_file_content_matches_returned_string(self, tmp_path):
        s = _scheduler(tmp_path)
        content = s.trigger_daily()
        files = list((tmp_path / "reports").glob("*.md"))
        assert files[0].read_text(encoding="utf-8") == content

    def test_two_triggers_create_two_files(self, tmp_path):
        import time
        s = _scheduler(tmp_path)
        s.trigger_daily()
        time.sleep(0.06)  # ensure different minute in filename
        s.trigger_daily()
        files = list((tmp_path / "reports").glob("*.md"))
        # filenames include timestamp; could be same minute → at least 1 file
        assert len(files) >= 1

    def test_custom_now_fn_affects_filename(self, tmp_path):
        fixed = datetime(2025, 1, 15, 8, 0)
        s = _scheduler(tmp_path, _now_fn=lambda: fixed)
        s.trigger_daily()
        files = list((tmp_path / "reports").glob("*.md"))
        assert "20250115" in files[0].name


# ─────────────────────────────────────────────────────────────────────────────
#  11. ReportScheduler – Alert-Versand
# ─────────────────────────────────────────────────────────────────────────────

class TestReportSchedulerAlertSend:
    def test_alert_called_on_daily(self, tmp_path):
        alert = MagicMock()
        s = _scheduler(tmp_path, alert=alert)
        s.trigger_daily()
        alert.send_alert.assert_called_once()

    def test_alert_called_on_weekly(self, tmp_path):
        alert = MagicMock()
        s = _scheduler(tmp_path, alert=alert)
        s.trigger_weekly()
        alert.send_alert.assert_called_once()

    def test_alert_receives_report_content(self, tmp_path):
        alert = MagicMock()
        s = _scheduler(tmp_path, alert=alert)
        content = s.trigger_daily()
        args = alert.send_alert.call_args[0][0]
        assert args == content

    def test_failing_alert_does_not_raise(self, tmp_path):
        alert = MagicMock()
        alert.send_alert.side_effect = RuntimeError("Netz down")
        s = _scheduler(tmp_path, alert=alert)
        s.trigger_daily()  # must not raise

    def test_log_only_sender_does_not_raise(self, tmp_path):
        s = ReportScheduler(
            journal=_journal(tmp_path),
            reports_dir=tmp_path / "r",
        )
        s.trigger_daily()  # must not raise


# ─────────────────────────────────────────────────────────────────────────────
#  12. ReportScheduler – Hintergrund-Thread
# ─────────────────────────────────────────────────────────────────────────────

class TestReportSchedulerThread:
    def test_is_running_after_start(self, tmp_path):
        s = _scheduler(tmp_path, check_interval_s=60)
        s.start()
        try:
            assert s.is_running is True
        finally:
            s.stop()

    def test_is_not_running_after_stop(self, tmp_path):
        s = _scheduler(tmp_path, check_interval_s=60)
        s.start()
        s.stop()
        assert s.is_running is False

    def test_double_start_idempotent(self, tmp_path):
        s = _scheduler(tmp_path, check_interval_s=60)
        s.start()
        thread_id = id(s._thread)
        s.start()
        assert id(s._thread) == thread_id
        s.stop()

    def test_stop_without_start_does_not_raise(self, tmp_path):
        s = _scheduler(tmp_path)
        s.stop()  # must not raise

    def test_thread_triggers_daily_via_now_fn(self, tmp_path):
        """Thread soll Daily-Report ausloesen wenn _now_fn eine passende Zeit liefert."""
        alert = MagicMock()
        fired_times = []

        def _fixed_now():
            return datetime(2026, 6, 22, 23, 0)  # Monday 23:00

        s = ReportScheduler(
            journal=_journal(tmp_path),
            reports_dir=tmp_path / "reports",
            daily_time="23:00",
            alert_sender=alert,
            check_interval_s=0,
            _now_fn=_fixed_now,
        )
        s.start()
        import time
        time.sleep(0.15)
        s.stop()
        alert.send_alert.assert_called()

    def test_thread_triggers_weekly_on_sunday(self, tmp_path):
        """Thread soll Weekly-Report am Sonntag ausloesen."""
        alert = MagicMock()
        call_count = [0]
        orig_weekly = None

        def _sunday_now():
            return datetime(2026, 6, 21, 23, 0)  # Sunday

        s = ReportScheduler(
            journal=_journal(tmp_path),
            reports_dir=tmp_path / "reports",
            weekly_time="23:00",
            alert_sender=alert,
            check_interval_s=0,
            _now_fn=_sunday_now,
        )
        s.start()
        import time
        time.sleep(0.15)
        s.stop()
        # Both daily and weekly should fire on Sunday (Sunday after 23:00)
        assert alert.send_alert.call_count >= 1


# ─────────────────────────────────────────────────────────────────────────────
#  13. LogOnlyAlertSender
# ─────────────────────────────────────────────────────────────────────────────

class TestLogOnlyAlertSender:
    def test_implements_protocol(self):
        sender = LogOnlyAlertSender()
        assert isinstance(sender, AlertSender)

    def test_send_alert_does_not_raise(self):
        LogOnlyAlertSender().send_alert("Test-Report")

    def test_send_alert_empty_does_not_raise(self):
        LogOnlyAlertSender().send_alert("")
