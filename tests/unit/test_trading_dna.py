"""
tests/unit/test_trading_dna.py
Unit-Tests fuer TradingDNA.

Abgedeckt:
  - Mindestanzahl-Schwelle: insufficient_data unter Limit, ready ab Limit
  - Handelszeiten: beste/schlechteste Stunden und Wochentage nach PnL
  - Symbole / Setups: Ranking nach Gesamt-PnL, None-Werte ausgeschlossen
  - Positionsgroesse: Lot-Bucket-Analyse, optimaler Bereich identifiziert
  - Psychologische Schwaechen: via PsychologyTracker (Mock)
  - Konfidenz: low / medium / high nach Trade-Anzahl
  - Kantenfall: leere Eingaben, fehlende Felder, einzelne Kategorie
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.journal.trading_dna import TradingDNA, _confidence


# ─────────────────────────────────────────────────────────────────────────────
#  Hilfsfunktionen
# ─────────────────────────────────────────────────────────────────────────────

# 2026-01-05 ist ein Montag (weekday=0)
_MONDAY = datetime(2026, 1, 5, 10, 0, 0, tzinfo=timezone.utc)


def _make_trade(
    pnl:      float = 10.0,
    symbol:   str   = "EURUSD",
    setup:    str   = "trend",
    lot_size: float = 0.10,
    hour:     int   = 10,
    weekday:  int   = 0,    # 0=Monday
) -> dict:
    """Erstellt ein minimales Trade-Dict mit gesetztem entry_time."""
    # Basis-Montag-Datum + gewuenschter Wochentag
    base = _MONDAY.replace(hour=hour)
    dt   = base + timedelta(days=weekday)
    return {
        "symbol":           symbol,
        "direction":        "buy",
        "lot_size":         lot_size,
        "entry_price":      1.09,
        "entry_time":       dt.isoformat(),
        "exit_price":       1.10,
        "exit_time":        dt.isoformat(),
        "pnl":              pnl,
        "regime":           "TRENDING",
        "news_context":     None,
        "signal_confidence": 0.70,
        "setup":            setup,
        "status":           "closed",
        "extra_json":       None,
    }


def _make_n(n: int, **kwargs) -> list[dict]:
    """Erstellt n identische Trades."""
    return [_make_trade(**kwargs) for _ in range(n)]


def _dna(trades: list[dict], min_trades: int = 5, psychology_tracker=None) -> TradingDNA:
    """Erstellt TradingDNA mit injiziertem Trade-Loader."""
    return TradingDNA(
        min_trades=min_trades,
        psychology_tracker=psychology_tracker,
        _trade_loader=lambda: trades,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  TestConfidenceHelper
# ─────────────────────────────────────────────────────────────────────────────

class TestConfidenceHelper:

    def test_low_under_30(self):
        for n in (0, 1, 15, 29):
            assert _confidence(n) == "low"

    def test_medium_30_to_99(self):
        for n in (30, 50, 99):
            assert _confidence(n) == "medium"

    def test_high_100_plus(self):
        for n in (100, 200, 1000):
            assert _confidence(n) == "high"


# ─────────────────────────────────────────────────────────────────────────────
#  TestInsufficientData
# ─────────────────────────────────────────────────────────────────────────────

class TestInsufficientData:

    def test_zero_trades_insufficient(self):
        profile = _dna([], min_trades=500).generate_profile()
        assert profile["status"] == "insufficient_data"

    def test_below_threshold_insufficient(self):
        trades  = _make_n(499)
        profile = _dna(trades, min_trades=500).generate_profile()
        assert profile["status"] == "insufficient_data"

    def test_at_threshold_ready(self):
        trades  = _make_n(500)
        profile = _dna(trades, min_trades=500).generate_profile()
        assert profile["status"] == "ready"

    def test_above_threshold_ready(self):
        trades  = _make_n(501)
        profile = _dna(trades, min_trades=500).generate_profile()
        assert profile["status"] == "ready"

    def test_custom_min_trades_respected(self):
        trades  = _make_n(9)
        profile = _dna(trades, min_trades=10).generate_profile()
        assert profile["status"] == "insufficient_data"

    def test_custom_min_trades_exact(self):
        trades  = _make_n(10)
        profile = _dna(trades, min_trades=10).generate_profile()
        assert profile["status"] == "ready"

    def test_insufficient_data_contains_counts(self):
        trades  = _make_n(42)
        profile = _dna(trades, min_trades=500).generate_profile()
        assert profile["n_trades"]            == 42
        assert profile["min_trades_required"] == 500
        assert "42" in profile["message"]
        assert "500" in profile["message"]

    def test_insufficient_data_no_analysis_keys(self):
        trades  = _make_n(4)
        profile = _dna(trades, min_trades=5).generate_profile()
        assert "trading_hours" not in profile

    def test_ready_profile_has_n_trades(self):
        trades  = _make_n(10)
        profile = _dna(trades, min_trades=10).generate_profile()
        assert profile["n_trades"] == 10


# ─────────────────────────────────────────────────────────────────────────────
#  TestTradingHours
# ─────────────────────────────────────────────────────────────────────────────

class TestTradingHours:

    def _profile_hours(self, trades: list[dict]) -> dict:
        return _dna(trades, min_trades=len(trades)).generate_profile()["trading_hours"]

    def test_best_hour_has_highest_pnl(self):
        trades = (
            _make_n(10, pnl=+100.0, hour=10) +
            _make_n(10, pnl=-80.0,  hour=22)
        )
        hours = self._profile_hours(trades)
        assert hours["best"][0]["hour"] == 10

    def test_worst_hour_has_lowest_pnl(self):
        trades = (
            _make_n(10, pnl=+100.0, hour=10) +
            _make_n(10, pnl=-80.0,  hour=22)
        )
        hours = self._profile_hours(trades)
        assert hours["worst"][0]["hour"] == 22

    def test_ranked_descending_by_pnl(self):
        trades = (
            _make_n(5, pnl=+50.0, hour=9) +   # total=+250
            _make_n(5, pnl=+10.0, hour=14) +   # total=+50
            _make_n(5, pnl=-20.0, hour=17)     # total=-100
        )
        hours   = self._profile_hours(trades)
        ranked  = hours["ranked"]
        totals  = [r["total_pnl"] for r in ranked]
        assert totals == sorted(totals, reverse=True)

    def test_hour_win_rate_computed(self):
        trades = (
            _make_n(8, pnl=+10.0, hour=10) +
            _make_n(2, pnl=-10.0, hour=10)
        )
        hours = self._profile_hours(trades)
        h10   = next(r for r in hours["ranked"] if r["hour"] == 10)
        assert h10["win_rate"] == pytest.approx(0.8)

    def test_trade_with_no_entry_time_excluded(self):
        t = _make_trade(pnl=+10.0, hour=10)
        t["entry_time"] = None
        trades = [t] + _make_n(5, pnl=+5.0, hour=12)
        hours  = self._profile_hours(trades)
        hour_keys = {r["hour"] for r in hours["ranked"]}
        assert 10 not in hour_keys  # None entry_time excluded

    def test_hour_n_trades_counted(self):
        trades = _make_n(7, pnl=+10.0, hour=10)
        hours  = self._profile_hours(trades)
        h10    = hours["ranked"][0]
        assert h10["n_trades"] == 7

    def test_hour_confidence_low(self):
        trades = _make_n(5, pnl=+10.0, hour=10)
        hours  = self._profile_hours(trades)
        assert hours["ranked"][0]["confidence"] == "low"

    def test_hour_confidence_medium(self):
        trades = _make_n(50, pnl=+10.0, hour=10)
        hours  = self._profile_hours(trades)
        assert hours["ranked"][0]["confidence"] == "medium"

    def test_hour_confidence_high(self):
        trades = _make_n(100, pnl=+10.0, hour=10)
        hours  = self._profile_hours(trades)
        assert hours["ranked"][0]["confidence"] == "high"

    def test_best_and_worst_lists_present(self):
        trades = _make_n(10, pnl=+5.0, hour=10)
        hours  = self._profile_hours(trades)
        assert "best"   in hours
        assert "worst"  in hours
        assert "ranked" in hours


# ─────────────────────────────────────────────────────────────────────────────
#  TestTradingWeekdays
# ─────────────────────────────────────────────────────────────────────────────

class TestTradingWeekdays:

    def _profile_wdays(self, trades: list[dict]) -> dict:
        return _dna(trades, min_trades=len(trades)).generate_profile()["trading_weekdays"]

    def test_best_weekday_has_highest_pnl(self):
        trades = (
            _make_n(10, pnl=+100.0, weekday=0) +  # Monday
            _make_n(10, pnl=-50.0,  weekday=4)    # Friday
        )
        wdays = self._profile_wdays(trades)
        assert wdays["best"][0]["weekday"] == "Monday"

    def test_worst_weekday_has_lowest_pnl(self):
        trades = (
            _make_n(10, pnl=+100.0, weekday=0) +
            _make_n(10, pnl=-50.0,  weekday=4)
        )
        wdays = self._profile_wdays(trades)
        assert wdays["worst"][0]["weekday"] == "Friday"

    def test_weekday_entry_has_name_and_index(self):
        trades = _make_n(5, weekday=1)  # Tuesday
        wdays  = self._profile_wdays(trades)
        entry  = wdays["ranked"][0]
        assert entry["weekday"]     == "Tuesday"
        assert entry["weekday_idx"] == 1

    def test_ranked_descending_by_pnl(self):
        trades = (
            _make_n(5, pnl=+40.0, weekday=0) +
            _make_n(5, pnl=+10.0, weekday=2) +
            _make_n(5, pnl=-20.0, weekday=4)
        )
        wdays  = self._profile_wdays(trades)
        totals = [r["total_pnl"] for r in wdays["ranked"]]
        assert totals == sorted(totals, reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
#  TestSymbols
# ─────────────────────────────────────────────────────────────────────────────

class TestSymbols:

    def _profile_sym(self, trades: list[dict]) -> dict:
        return _dna(trades, min_trades=len(trades)).generate_profile()["symbols"]

    def test_best_symbol_highest_total_pnl(self):
        trades = (
            _make_n(10, pnl=+80.0, symbol="EURUSD") +
            _make_n(10, pnl=-30.0, symbol="GBPUSD")
        )
        syms = self._profile_sym(trades)
        assert syms["best"][0]["symbol"] == "EURUSD"

    def test_worst_symbol_lowest_total_pnl(self):
        trades = (
            _make_n(10, pnl=+80.0, symbol="EURUSD") +
            _make_n(10, pnl=-30.0, symbol="GBPUSD")
        )
        syms = self._profile_sym(trades)
        assert syms["worst"][0]["symbol"] == "GBPUSD"

    def test_ranked_descending_by_pnl(self):
        trades = (
            _make_n(5, pnl=+100.0, symbol="A") +
            _make_n(5, pnl=+20.0,  symbol="B") +
            _make_n(5, pnl=-10.0,  symbol="C")
        )
        syms   = self._profile_sym(trades)
        totals = [r["total_pnl"] for r in syms["ranked"]]
        assert totals == sorted(totals, reverse=True)

    def test_none_symbol_excluded(self):
        trades = _make_n(5, pnl=+10.0, symbol="EURUSD")
        t = _make_trade(pnl=+10.0); t["symbol"] = None
        trades.append(t)
        syms    = self._profile_sym(trades)
        symbols = {r["symbol"] for r in syms["ranked"]}
        assert None not in symbols
        assert "None" not in symbols

    def test_win_rate_per_symbol(self):
        trades = (
            _make_n(6, pnl=+10.0, symbol="EURUSD") +
            _make_n(4, pnl=-10.0, symbol="EURUSD")
        )
        syms  = self._profile_sym(trades)
        entry = syms["ranked"][0]
        assert entry["win_rate"] == pytest.approx(0.6)

    def test_n_trades_per_symbol(self):
        trades = _make_n(7, pnl=+5.0, symbol="USDJPY")
        syms   = self._profile_sym(trades)
        assert syms["ranked"][0]["n_trades"] == 7

    def test_single_symbol_is_both_best_and_worst(self):
        trades = _make_n(5, pnl=+10.0, symbol="EURUSD")
        syms   = self._profile_sym(trades)
        assert syms["best"][0]["symbol"]  == "EURUSD"
        assert syms["worst"][0]["symbol"] == "EURUSD"


# ─────────────────────────────────────────────────────────────────────────────
#  TestSetups
# ─────────────────────────────────────────────────────────────────────────────

class TestSetups:

    def _profile_setups(self, trades: list[dict]) -> dict:
        return _dna(trades, min_trades=len(trades)).generate_profile()["setups"]

    def test_best_setup_highest_pnl(self):
        trades = (
            _make_n(10, pnl=+60.0, setup="breakout") +
            _make_n(10, pnl=-20.0, setup="reversal")
        )
        setups = self._profile_setups(trades)
        assert setups["best"][0]["setup"] == "breakout"

    def test_none_setup_excluded(self):
        trades = _make_n(5, pnl=+10.0, setup="trend")
        t = _make_trade(pnl=+10.0); t["setup"] = None
        trades.append(t)
        setups  = self._profile_setups(trades)
        labels  = {r["setup"] for r in setups["ranked"]}
        assert None not in labels
        assert "None" not in labels

    def test_setup_win_rate_computed(self):
        trades = (
            _make_n(3, pnl=+10.0, setup="range") +
            _make_n(1, pnl=-10.0, setup="range")
        )
        setups = self._profile_setups(trades)
        entry  = next(r for r in setups["ranked"] if r["setup"] == "range")
        assert entry["win_rate"] == pytest.approx(0.75)


# ─────────────────────────────────────────────────────────────────────────────
#  TestPositionSizing
# ─────────────────────────────────────────────────────────────────────────────

class TestPositionSizing:

    def _profile_sizing(self, trades: list[dict]) -> dict:
        return _dna(trades, min_trades=len(trades)).generate_profile()["position_sizing"]

    def test_buckets_populated(self):
        trades  = _make_n(5, pnl=+10.0, lot_size=0.10)
        sizing  = self._profile_sizing(trades)
        assert len(sizing["lot_buckets"]) == 1
        assert sizing["lot_buckets"][0]["label"] == "0.10–0.20"

    def test_optimal_range_identified(self):
        # Bucket 0.10-0.20 hat 100% win rate, 0.05-0.10 hat 0%
        trades = (
            _make_n(5, pnl=+10.0, lot_size=0.15) +
            _make_n(5, pnl=-10.0, lot_size=0.07)
        )
        sizing = self._profile_sizing(trades)
        assert sizing["optimal_lot_range"][0] == pytest.approx(0.10)
        assert sizing["optimal_win_rate"]     == pytest.approx(1.0)
        assert sizing["optimal_label"]        == "0.10–0.20"

    def test_empty_bucket_not_in_results(self):
        trades  = _make_n(5, pnl=+10.0, lot_size=0.30)
        sizing  = self._profile_sizing(trades)
        labels  = {b["label"] for b in sizing["lot_buckets"]}
        assert "0.10–0.20" not in labels
        assert "0.20–0.50" in labels

    def test_large_lot_uses_last_bucket(self):
        trades = _make_n(5, pnl=+10.0, lot_size=1.0)
        sizing = self._profile_sizing(trades)
        assert sizing["lot_buckets"][0]["label"] == ">= 0.50"

    def test_lot_max_is_none_for_last_bucket(self):
        trades = _make_n(5, pnl=+10.0, lot_size=1.0)
        sizing = self._profile_sizing(trades)
        last   = sizing["lot_buckets"][-1]
        assert last["lot_max"] is None

    def test_no_trades_returns_none_optimal(self):
        # lot_size=None for all → no buckets
        trades = [_make_trade(pnl=+10.0) for _ in range(5)]
        for t in trades:
            t["lot_size"] = None
        sizing = self._profile_sizing(trades)
        assert sizing["lot_buckets"]       == []
        assert sizing["optimal_lot_range"] is None
        assert sizing["optimal_win_rate"]  is None

    def test_bucket_win_rate(self):
        trades = (
            _make_n(3, pnl=+10.0, lot_size=0.10) +
            _make_n(1, pnl=-10.0, lot_size=0.10)
        )
        sizing  = self._profile_sizing(trades)
        bucket  = sizing["lot_buckets"][0]
        assert bucket["win_rate"] == pytest.approx(0.75)

    def test_bucket_total_pnl(self):
        trades  = _make_n(4, pnl=+25.0, lot_size=0.10)
        sizing  = self._profile_sizing(trades)
        assert sizing["lot_buckets"][0]["total_pnl"] == pytest.approx(100.0)

    def test_multiple_buckets_optimal_is_best_win_rate(self):
        trades = (
            _make_n(10, pnl=+10.0, lot_size=0.07) +   # 0.05-0.10 → 100%
            _make_n(5,  pnl=+10.0, lot_size=0.15) +   # 0.10-0.20 → 80%
            _make_n(1,  pnl=-10.0, lot_size=0.15)
        )
        sizing = self._profile_sizing(trades)
        assert sizing["optimal_label"] == "0.05–0.10"


# ─────────────────────────────────────────────────────────────────────────────
#  TestPsychologicalWeaknesses
# ─────────────────────────────────────────────────────────────────────────────

class TestPsychologicalWeaknesses:

    def _mock_tracker(self, patterns: dict) -> Any:
        tracker = MagicMock()
        tracker.analyze_mood_patterns.return_value = patterns
        return tracker

    def _profile_weaknesses(self, trades: list[dict], tracker=None) -> list[str]:
        return (
            _dna(trades, min_trades=len(trades), psychology_tracker=tracker)
            .generate_profile()["psychological_weaknesses"]
        )

    def _mood_entry(self, win_rate: float, n_trades: int) -> dict:
        return {"win_rate": win_rate, "n_trades": n_trades}

    def test_no_tracker_returns_empty(self):
        trades = _make_n(5)
        result = self._profile_weaknesses(trades)
        assert result == []

    def test_low_win_rate_mood_is_weakness(self):
        class _Mood:
            value = "fomo"
        patterns = {_Mood(): self._mood_entry(0.30, 15)}
        tracker  = self._mock_tracker(patterns)
        trades   = _make_n(5)
        result   = self._profile_weaknesses(trades, tracker)
        assert len(result) == 1
        assert "FOMO" in result[0]
        assert "30%" in result[0]

    def test_high_win_rate_mood_not_weakness(self):
        class _Mood:
            value = "calm"
        patterns = {_Mood(): self._mood_entry(0.65, 50)}
        tracker  = self._mock_tracker(patterns)
        trades   = _make_n(5)
        result   = self._profile_weaknesses(trades, tracker)
        assert result == []

    def test_below_min_samples_not_weakness(self):
        class _Mood:
            value = "angry"
        # win_rate=0.20 but only 9 samples (< 10 threshold)
        patterns = {_Mood(): self._mood_entry(0.20, 9)}
        tracker  = self._mock_tracker(patterns)
        trades   = _make_n(5)
        result   = self._profile_weaknesses(trades, tracker)
        assert result == []

    def test_multiple_weaknesses_sorted(self):
        class _MoodA:
            value = "nervous"
        class _MoodB:
            value = "angry"
        patterns = {
            _MoodA(): self._mood_entry(0.25, 20),
            _MoodB(): self._mood_entry(0.35, 15),
        }
        tracker  = self._mock_tracker(patterns)
        trades   = _make_n(5)
        result   = self._profile_weaknesses(trades, tracker)
        assert len(result) == 2
        assert result == sorted(result)  # alphabetically sorted

    def test_weaknesses_contain_trade_count(self):
        class _Mood:
            value = "overconfident"
        patterns = {_Mood(): self._mood_entry(0.33, 25)}
        tracker  = self._mock_tracker(patterns)
        trades   = _make_n(5)
        result   = self._profile_weaknesses(trades, tracker)
        assert "25" in result[0]


# ─────────────────────────────────────────────────────────────────────────────
#  TestProfileStructure
# ─────────────────────────────────────────────────────────────────────────────

class TestProfileStructure:

    def _full_profile(self, n: int = 10) -> dict:
        trades = _make_n(n)
        return _dna(trades, min_trades=n).generate_profile()

    def test_ready_profile_has_all_top_level_keys(self):
        profile = self._full_profile()
        expected_keys = {
            "status", "n_trades", "min_trades_required",
            "trading_hours", "trading_weekdays",
            "symbols", "setups",
            "position_sizing", "psychological_weaknesses",
        }
        assert expected_keys.issubset(profile.keys())

    def test_trading_hours_has_ranked_best_worst(self):
        profile = self._full_profile()
        hours   = profile["trading_hours"]
        assert "ranked" in hours
        assert "best"   in hours
        assert "worst"  in hours

    def test_trading_weekdays_has_ranked_best_worst(self):
        profile = self._full_profile()
        wdays   = profile["trading_weekdays"]
        assert "ranked" in wdays
        assert "best"   in wdays
        assert "worst"  in wdays

    def test_symbols_has_ranked_best_worst(self):
        profile = self._full_profile()
        syms    = profile["symbols"]
        assert "ranked" in syms
        assert "best"   in syms
        assert "worst"  in syms

    def test_setups_has_ranked_best_worst(self):
        profile = self._full_profile()
        setups  = profile["setups"]
        assert "ranked" in setups
        assert "best"   in setups
        assert "worst"  in setups

    def test_position_sizing_has_required_keys(self):
        profile = self._full_profile()
        sizing  = profile["position_sizing"]
        assert "lot_buckets"       in sizing
        assert "optimal_lot_range" in sizing
        assert "optimal_win_rate"  in sizing
        assert "optimal_label"     in sizing

    def test_psychological_weaknesses_is_list(self):
        profile = self._full_profile()
        assert isinstance(profile["psychological_weaknesses"], list)

    def test_status_is_ready(self):
        assert self._full_profile()["status"] == "ready"


# ─────────────────────────────────────────────────────────────────────────────
#  TestEdgeCases
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_no_journal_no_loader_insufficient(self):
        dna     = TradingDNA(min_trades=1)
        profile = dna.generate_profile()
        assert profile["status"]   == "insufficient_data"
        assert profile["n_trades"] == 0

    def test_invalid_entry_time_excluded(self):
        trades = _make_n(5, pnl=+10.0, hour=10)
        trades[0]["entry_time"] = "not-a-date"
        dna     = _dna(trades, min_trades=5)
        profile = dna.generate_profile()
        # Should not raise; hour=10 may appear from the 4 valid trades
        hours   = profile["trading_hours"]["ranked"]
        total_n = sum(r["n_trades"] for r in hours)
        assert total_n == 4  # invalid trade excluded

    def test_none_pnl_excluded_from_analysis(self):
        trades = _make_n(5, pnl=+10.0)
        trades[0]["pnl"] = None  # open trade slipped in
        dna    = _dna(trades, min_trades=5)
        profile = dna.generate_profile()
        # Only 4 trades with real pnl, but n_trades counts loader output
        assert profile["n_trades"] == 5  # raw count
        # Hour analysis should only have 4 trades
        hours   = profile["trading_hours"]["ranked"]
        total_n = sum(r["n_trades"] for r in hours)
        assert total_n == 4

    def test_with_journal_integration(self, tmp_path):
        from src.journal.trade_journal import TradeJournal
        db_path = tmp_path / "test.db"
        with TradeJournal(db_path=str(db_path)) as journal:
            for _ in range(6):
                tid = journal.log_trade_open({"symbol": "EURUSD", "direction": "buy",
                                               "lot_size": 0.1, "entry_time": _MONDAY.isoformat()})
                journal.log_trade_close(tid, {"pnl": 10.0,
                                               "exit_time": _MONDAY.isoformat()})
            dna     = TradingDNA(journal=journal, min_trades=6)
            profile = dna.generate_profile()
        assert profile["status"]   == "ready"
        assert profile["n_trades"] == 6
