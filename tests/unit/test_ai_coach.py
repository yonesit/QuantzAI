"""
tests/unit/test_ai_coach.py
Unit-Tests fuer AICoach (src/journal/ai_coach.py).

Abgedeckt:
  - Datenaufbereitung: build_context(), _format_trades_table(),
    _format_stats(), _format_dna_summary(), _format_psychology_summary()
    alle isoliert (ohne LLM-Aufruf)
  - ask(): LLM gemockt via _llm_fn-Injection
  - Ehrliche Antwort bei wenig/keinen Daten
  - Fehlertoleranz (LLM-Fehler, fehlendes anthropic-Paket)
  - Optionale Abhaengigkeiten (dna=None, psychology_tracker=None, journal=None)
"""

from __future__ import annotations

import sqlite3
import tempfile
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.journal.ai_coach import (
    AICoach,
    _MIN_DATA_WARNING_TRADES,
    _STATS_LOOKBACK_DAYS,
    _SYSTEM_INTRO,
)
from src.journal.trade_journal import TradeJournal


# ─────────────────────────────────────────────────────────────────────────────
#  Hilfsfunktionen
# ─────────────────────────────────────────────────────────────────────────────

def _mock_llm(response: str = "Antwort vom Coach"):
    return lambda system, user: response


def _make_journal_with_trades(trades: list[dict]) -> TradeJournal:
    """Erstellt eine In-Memory-TradeJournal-Instanz mit den angegebenen Trades."""
    tmp = tempfile.mktemp(suffix=".db")
    journal = TradeJournal(db_path=tmp)
    for td in trades:
        tid = journal.log_trade_open(td)
        if "pnl" in td and td.get("status", "closed") == "closed":
            journal.log_trade_close(tid, {
                "exit_price": td.get("exit_price", 1.0),
                "exit_time":  "2026-01-02T10:00:00+00:00",
                "pnl":        td["pnl"],
            })
    return journal


def _make_sample_trades(n: int = 5) -> list[dict]:
    return [
        {
            "symbol": "EURUSD", "direction": "buy", "lot_size": 0.1,
            "entry_price": 1.08 + i * 0.001,
            "entry_time": f"2026-01-{i+1:02d}T09:00:00+00:00",
            "setup": "BreakoutA", "status": "closed",
            "pnl": 50.0 if i % 2 == 0 else -30.0,
        }
        for i in range(n)
    ]


def _make_coach(
    trades: list[dict] | None = None,
    llm_response: str = "Antwort",
    dna=None,
    psychology_tracker=None,
) -> AICoach:
    journal = _make_journal_with_trades(trades if trades is not None else _make_sample_trades())
    return AICoach(
        journal=journal,
        dna=dna,
        psychology_tracker=psychology_tracker,
        _llm_fn=_mock_llm(llm_response),
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Initialisierung
# ─────────────────────────────────────────────────────────────────────────────

class TestAICoachInit:

    def test_default_model_set(self):
        coach = AICoach()
        assert "claude" in coach._model

    def test_custom_model_accepted(self):
        coach = AICoach(model="claude-haiku-4-5-20251001")
        assert coach._model == "claude-haiku-4-5-20251001"

    def test_journal_none_accepted(self):
        coach = AICoach(journal=None, _llm_fn=_mock_llm())
        assert coach._journal is None

    def test_dna_none_accepted(self):
        coach = AICoach(_llm_fn=_mock_llm())
        assert coach._dna is None

    def test_psychology_tracker_none_accepted(self):
        coach = AICoach(_llm_fn=_mock_llm())
        assert coach._psychology is None

    def test_max_trades_in_context_default(self):
        coach = AICoach()
        assert coach._max_trades == 50

    def test_custom_max_trades(self):
        coach = AICoach(max_trades_in_context=20)
        assert coach._max_trades == 20

    def test_llm_fn_injectable(self):
        fn = _mock_llm("test")
        coach = AICoach(_llm_fn=fn)
        assert coach._llm_fn is fn


# ─────────────────────────────────────────────────────────────────────────────
#  ask() – LLM gemockt
# ─────────────────────────────────────────────────────────────────────────────

class TestAskLLMMocked:

    def test_ask_returns_llm_response(self):
        coach = _make_coach(llm_response="Dein bester Trade war #3.")
        result = coach.ask("Was war mein bester Trade?")
        assert result == "Dein bester Trade war #3."

    def test_ask_empty_question_returns_hint(self):
        coach = _make_coach()
        result = coach.ask("")
        assert "Frage" in result

    def test_ask_whitespace_only_returns_hint(self):
        coach = _make_coach()
        result = coach.ask("   ")
        assert "Frage" in result

    def test_ask_passes_question_to_llm(self):
        received = {}
        def _fn(system, user):
            received["user"] = user
            return "ok"
        journal = _make_journal_with_trades(_make_sample_trades())
        coach   = AICoach(journal=journal, _llm_fn=_fn)
        coach.ask("Wie ist meine Win-Rate?")
        assert received["user"] == "Wie ist meine Win-Rate?"

    def test_ask_passes_system_prompt_to_llm(self):
        received = {}
        def _fn(system, user):
            received["system"] = system
            return "ok"
        journal = _make_journal_with_trades(_make_sample_trades())
        coach   = AICoach(journal=journal, _llm_fn=_fn)
        coach.ask("Frage?")
        assert _SYSTEM_INTRO in received["system"]

    def test_ask_strips_whitespace_from_question(self):
        received = {}
        def _fn(system, user):
            received["user"] = user
            return "ok"
        journal = _make_journal_with_trades(_make_sample_trades())
        coach   = AICoach(journal=journal, _llm_fn=_fn)
        coach.ask("  Frage?  ")
        assert received["user"] == "Frage?"

    def test_ask_llm_error_returns_error_message(self):
        def _failing_fn(system, user):
            raise RuntimeError("Verbindungsfehler")
        journal = _make_journal_with_trades(_make_sample_trades())
        coach   = AICoach(journal=journal, _llm_fn=_failing_fn)
        result  = coach.ask("Frage?")
        assert "Fehler" in result or "fehler" in result

    def test_ask_no_anthropic_without_llm_fn(self):
        coach = AICoach(_llm_fn=None)
        with patch.dict("sys.modules", {"anthropic": None}):
            result = coach.ask("Frage?")
        assert "Fehler" in result or "fehler" in result or "Error" in result

    def test_ask_system_includes_context(self):
        received = {}
        def _fn(system, user):
            received["system"] = system
            return "ok"
        journal = _make_journal_with_trades(_make_sample_trades())
        coach   = AICoach(journal=journal, _llm_fn=_fn)
        coach.ask("Frage?")
        # Kontext-Daten muessen im System-Prompt sein
        assert "Handelshistorie" in received["system"]


# ─────────────────────────────────────────────────────────────────────────────
#  build_context() – Datenaufbereitung isoliert
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildContext:

    def test_returns_string(self):
        coach = _make_coach()
        assert isinstance(coach.build_context(), str)

    def test_contains_handelshistorie_header(self):
        coach = _make_coach()
        ctx   = coach.build_context()
        assert "Handelshistorie" in ctx

    def test_contains_statistiken_header(self):
        coach = _make_coach()
        ctx   = coach.build_context()
        assert "Statistiken" in ctx

    def test_no_journal_handelshistorie_shows_no_trades(self):
        coach = AICoach(journal=None, _llm_fn=_mock_llm())
        ctx   = coach.build_context()
        assert "Keine Trades" in ctx

    def test_no_journal_no_statistiken_data(self):
        coach = AICoach(journal=None, _llm_fn=_mock_llm())
        ctx   = coach.build_context()
        assert "Kein Journal" in ctx

    def test_dna_section_present_when_dna_given(self):
        dna   = MagicMock()
        dna.generate_profile.return_value = {"status": "insufficient_data", "n_trades": 0, "min_trades_required": 500, "message": "..."}
        coach = AICoach(journal=None, dna=dna, _llm_fn=_mock_llm())
        ctx   = coach.build_context()
        assert "DNA" in ctx

    def test_no_dna_section_when_dna_is_none(self):
        coach = AICoach(journal=None, dna=None, _llm_fn=_mock_llm())
        ctx   = coach.build_context()
        assert "TradingDNA" not in ctx

    def test_psychology_section_present_when_tracker_given(self):
        pt = MagicMock()
        pt.analyze_mood_patterns.return_value = {}
        coach = AICoach(journal=None, psychology_tracker=pt, _llm_fn=_mock_llm())
        ctx   = coach.build_context()
        assert "Psychologie" in ctx

    def test_no_psychology_section_when_tracker_is_none(self):
        coach = AICoach(journal=None, psychology_tracker=None, _llm_fn=_mock_llm())
        ctx   = coach.build_context()
        assert "Psychologie" not in ctx


# ─────────────────────────────────────────────────────────────────────────────
#  _format_trades_table()
# ─────────────────────────────────────────────────────────────────────────────

class TestFormatTradesTable:

    def test_empty_list_returns_no_trades_marker(self):
        coach = AICoach()
        result = coach._format_trades_table([])
        assert "Keine Trades" in result

    def test_table_has_header_row(self):
        coach  = AICoach()
        trades = [{"id": 1, "symbol": "EURUSD", "direction": "buy",
                   "lot_size": 0.1, "entry_price": 1.08, "exit_price": None,
                   "pnl": None, "status": "open", "setup": None}]
        result = coach._format_trades_table(trades)
        assert "| ID |" in result
        assert "| Symbol |" in result

    def test_table_contains_trade_id(self):
        coach  = AICoach()
        trades = [{"id": 42, "symbol": "GBPUSD", "direction": "sell",
                   "lot_size": 0.2, "entry_price": 1.27, "exit_price": 1.25,
                   "pnl": 200.0, "status": "closed", "setup": "Fib"}]
        result = coach._format_trades_table(trades)
        assert "#42" in result

    def test_table_contains_symbol(self):
        coach  = AICoach()
        trades = [{"id": 1, "symbol": "USDJPY", "direction": "buy",
                   "lot_size": 0.1, "entry_price": 150.0, "exit_price": None,
                   "pnl": None, "status": "open", "setup": None}]
        result = coach._format_trades_table(trades)
        assert "USDJPY" in result

    def test_pnl_formatted_with_sign(self):
        coach  = AICoach()
        trades = [{"id": 1, "symbol": "EURUSD", "direction": "buy",
                   "lot_size": 0.1, "entry_price": 1.08, "exit_price": 1.09,
                   "pnl": 100.0, "status": "closed", "setup": None}]
        result = coach._format_trades_table(trades)
        assert "+100.00" in result

    def test_negative_pnl_formatted(self):
        coach  = AICoach()
        trades = [{"id": 1, "symbol": "EURUSD", "direction": "buy",
                   "lot_size": 0.1, "entry_price": 1.08, "exit_price": 1.07,
                   "pnl": -50.0, "status": "closed", "setup": None}]
        result = coach._format_trades_table(trades)
        assert "-50.00" in result

    def test_open_trade_pnl_shows_offen(self):
        coach  = AICoach()
        trades = [{"id": 1, "symbol": "EURUSD", "direction": "buy",
                   "lot_size": 0.1, "entry_price": 1.08, "exit_price": None,
                   "pnl": None, "status": "open", "setup": None}]
        result = coach._format_trades_table(trades)
        assert "offen" in result

    def test_multiple_trades_all_ids_present(self):
        coach  = AICoach()
        trades = [{"id": i, "symbol": "EURUSD", "direction": "buy",
                   "lot_size": 0.1, "entry_price": 1.08, "exit_price": None,
                   "pnl": None, "status": "open", "setup": None}
                  for i in range(1, 4)]
        result = coach._format_trades_table(trades)
        for i in range(1, 4):
            assert f"#{i}" in result

    def test_setup_dash_when_none(self):
        coach  = AICoach()
        trades = [{"id": 1, "symbol": "EURUSD", "direction": "buy",
                   "lot_size": 0.1, "entry_price": 1.08, "exit_price": None,
                   "pnl": None, "status": "open", "setup": None}]
        result = coach._format_trades_table(trades)
        # There should be a dash placeholder for the missing setup
        assert "–" in result


# ─────────────────────────────────────────────────────────────────────────────
#  _format_stats()
# ─────────────────────────────────────────────────────────────────────────────

class TestFormatStats:

    def test_zero_trades_returns_no_trades_message(self):
        coach = AICoach()
        stats = {"n_trades": 0, "win_rate": 0.0, "profit_factor": 0.0,
                 "avg_win": 0.0, "avg_loss": 0.0, "total_pnl": 0.0,
                 "best_trade": None, "worst_trade": None}
        result = coach._format_stats(stats)
        assert "Keine" in result

    def test_includes_win_rate(self):
        coach = AICoach()
        stats = {"n_trades": 10, "win_rate": 0.6, "profit_factor": 1.5,
                 "avg_win": 100.0, "avg_loss": 50.0, "total_pnl": 300.0,
                 "best_trade": 200.0, "worst_trade": -80.0}
        result = coach._format_stats(stats)
        assert "60%" in result or "60,0%" in result or "Win-Rate" in result

    def test_includes_total_pnl(self):
        coach = AICoach()
        stats = {"n_trades": 10, "win_rate": 0.5, "profit_factor": 1.0,
                 "avg_win": 50.0, "avg_loss": 50.0, "total_pnl": 100.0,
                 "best_trade": 100.0, "worst_trade": -50.0}
        result = coach._format_stats(stats)
        assert "100" in result

    def test_infinite_profit_factor_shown(self):
        coach = AICoach()
        stats = {"n_trades": 5, "win_rate": 1.0, "profit_factor": float("inf"),
                 "avg_win": 50.0, "avg_loss": 0.0, "total_pnl": 250.0,
                 "best_trade": 100.0, "worst_trade": 20.0}
        result = coach._format_stats(stats)
        assert "inf" in result.lower()

    def test_includes_n_trades(self):
        coach = AICoach()
        stats = {"n_trades": 42, "win_rate": 0.5, "profit_factor": 1.2,
                 "avg_win": 80.0, "avg_loss": 60.0, "total_pnl": 200.0,
                 "best_trade": 200.0, "worst_trade": -100.0}
        result = coach._format_stats(stats)
        assert "42" in result


# ─────────────────────────────────────────────────────────────────────────────
#  _format_dna_summary()
# ─────────────────────────────────────────────────────────────────────────────

class TestFormatDnaSummary:

    def test_insufficient_data_message(self):
        coach   = AICoach()
        profile = {"status": "insufficient_data", "n_trades": 10, "min_trades_required": 500, "message": "..."}
        result  = coach._format_dna_summary(profile)
        assert "10" in result
        assert "500" in result

    def test_ready_profile_includes_best_hours(self):
        coach   = AICoach()
        profile = {
            "status": "ready",
            "trading_hours":    {"best": [{"hour": 9, "win_rate": 0.7, "total_pnl": 500}], "worst": []},
            "trading_weekdays": {"best": [], "worst": []},
            "symbols":          {"best": [], "worst": []},
            "setups":           {"best": [], "worst": []},
            "position_sizing":  {"lot_buckets": [], "optimal_label": None, "optimal_win_rate": None},
            "psychological_weaknesses": [],
        }
        result = coach._format_dna_summary(profile)
        assert "9:00" in result

    def test_ready_profile_includes_best_symbols(self):
        coach   = AICoach()
        profile = {
            "status": "ready",
            "trading_hours":    {"best": [], "worst": []},
            "trading_weekdays": {"best": [], "worst": []},
            "symbols":          {"best": [{"symbol": "EURUSD", "win_rate": 0.65, "total_pnl": 800}], "worst": []},
            "setups":           {"best": [], "worst": []},
            "position_sizing":  {"lot_buckets": [], "optimal_label": None, "optimal_win_rate": None},
            "psychological_weaknesses": [],
        }
        result = coach._format_dna_summary(profile)
        assert "EURUSD" in result

    def test_psychological_weaknesses_included(self):
        coach   = AICoach()
        profile = {
            "status": "ready",
            "trading_hours":    {"best": [], "worst": []},
            "trading_weekdays": {"best": [], "worst": []},
            "symbols":          {"best": [], "worst": []},
            "setups":           {"best": [], "worst": []},
            "position_sizing":  {"lot_buckets": [], "optimal_label": None, "optimal_win_rate": None},
            "psychological_weaknesses": ["ANGRY: win_rate=35% (15 Trades)"],
        }
        result = coach._format_dna_summary(profile)
        assert "ANGRY" in result

    def test_optimal_lot_range_included(self):
        coach   = AICoach()
        profile = {
            "status": "ready",
            "trading_hours":    {"best": [], "worst": []},
            "trading_weekdays": {"best": [], "worst": []},
            "symbols":          {"best": [], "worst": []},
            "setups":           {"best": [], "worst": []},
            "position_sizing":  {
                "lot_buckets": [], "optimal_label": "0.05–0.10",
                "optimal_win_rate": 0.72, "optimal_lot_range": [0.05, 0.10],
            },
            "psychological_weaknesses": [],
        }
        result = coach._format_dna_summary(profile)
        assert "0.05–0.10" in result

    def test_empty_ready_profile_no_crash(self):
        coach   = AICoach()
        profile = {
            "status": "ready",
            "trading_hours":    {"best": [], "worst": []},
            "trading_weekdays": {"best": [], "worst": []},
            "symbols":          {"best": [], "worst": []},
            "setups":           {"best": [], "worst": []},
            "position_sizing":  {"lot_buckets": [], "optimal_label": None, "optimal_win_rate": None},
            "psychological_weaknesses": [],
        }
        result = coach._format_dna_summary(profile)
        assert isinstance(result, str)


# ─────────────────────────────────────────────────────────────────────────────
#  _format_psychology_summary()
# ─────────────────────────────────────────────────────────────────────────────

class TestFormatPsychologySummary:

    def test_empty_patterns_returns_not_enough_message(self):
        coach  = AICoach()
        result = coach._format_psychology_summary({})
        assert "30" in result or "genug" in result or "Trades" in result

    def test_patterns_shown_as_table(self):
        from src.journal.psychology_tracker import MoodState
        coach    = AICoach()
        patterns = {MoodState.CALM: {"n_trades": 20, "win_rate": 0.65}}
        result   = coach._format_psychology_summary(patterns)
        assert "Calm" in result or "calm" in result
        assert "65%" in result or "65" in result

    def test_multiple_moods_all_shown(self):
        from src.journal.psychology_tracker import MoodState
        coach    = AICoach()
        patterns = {
            MoodState.CALM:  {"n_trades": 15, "win_rate": 0.7},
            MoodState.ANGRY: {"n_trades": 8,  "win_rate": 0.3},
        }
        result = coach._format_psychology_summary(patterns)
        assert "Calm" in result or "calm" in result
        assert "Angry" in result or "angry" in result


# ─────────────────────────────────────────────────────────────────────────────
#  _get_recent_trades()
# ─────────────────────────────────────────────────────────────────────────────

class TestGetRecentTrades:

    def test_returns_list(self):
        coach = _make_coach()
        assert isinstance(coach._get_recent_trades(10), list)

    def test_respects_limit(self):
        journal = _make_journal_with_trades(_make_sample_trades(20))
        coach   = AICoach(journal=journal, _llm_fn=_mock_llm())
        result  = coach._get_recent_trades(5)
        assert len(result) <= 5

    def test_returns_empty_when_no_journal(self):
        coach = AICoach(journal=None)
        assert coach._get_recent_trades(10) == []

    def test_trades_have_id_field(self):
        coach = _make_coach(_make_sample_trades(3))
        trades = coach._get_recent_trades(10)
        for t in trades:
            assert "id" in t

    def test_trades_in_chronological_order(self):
        journal = _make_journal_with_trades(_make_sample_trades(5))
        coach   = AICoach(journal=journal, _llm_fn=_mock_llm())
        trades  = coach._get_recent_trades(10)
        ids = [t["id"] for t in trades]
        assert ids == sorted(ids)


# ─────────────────────────────────────────────────────────────────────────────
#  _format_trades_section() – Datenmenge-Warnungen
# ─────────────────────────────────────────────────────────────────────────────

class TestFormatTradesSection:

    def test_no_trades_shows_empty_message(self):
        coach  = AICoach()
        result = coach._format_trades_section([])
        assert "Keine Trades" in result

    def test_few_trades_shows_warning(self):
        coach  = AICoach()
        trades = [{"id": 1, "symbol": "EURUSD", "direction": "buy",
                   "lot_size": 0.1, "entry_price": 1.08, "exit_price": None,
                   "pnl": None, "status": "open", "setup": None}]
        result = coach._format_trades_section(trades)
        assert "wenig" in result.lower() or "Hinweis" in result or str(_MIN_DATA_WARNING_TRADES) in result

    def test_enough_trades_no_warning(self):
        coach  = AICoach()
        trades = [{"id": i, "symbol": "EURUSD", "direction": "buy",
                   "lot_size": 0.1, "entry_price": 1.08, "exit_price": None,
                   "pnl": None, "status": "open", "setup": None}
                  for i in range(1, _MIN_DATA_WARNING_TRADES + 2)]
        result = coach._format_trades_section(trades)
        assert "Hinweis" not in result


# ─────────────────────────────────────────────────────────────────────────────
#  Integration: DNA und Psychologie via Mock
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegrationWithMocks:

    def test_dna_generate_profile_called(self):
        dna = MagicMock()
        dna.generate_profile.return_value = {
            "status": "insufficient_data", "n_trades": 0,
            "min_trades_required": 500, "message": "...",
        }
        coach = AICoach(dna=dna, _llm_fn=_mock_llm())
        coach.build_context()
        dna.generate_profile.assert_called_once()

    def test_psychology_analyze_mood_patterns_called(self):
        pt = MagicMock()
        pt.analyze_mood_patterns.return_value = {}
        coach = AICoach(psychology_tracker=pt, _llm_fn=_mock_llm())
        coach.build_context()
        pt.analyze_mood_patterns.assert_called_once()

    def test_dna_exception_does_not_propagate(self):
        dna = MagicMock()
        dna.generate_profile.side_effect = RuntimeError("DB down")
        coach = AICoach(dna=dna, _llm_fn=_mock_llm())
        ctx = coach.build_context()
        assert isinstance(ctx, str)

    def test_psychology_exception_does_not_propagate(self):
        pt = MagicMock()
        pt.analyze_mood_patterns.side_effect = RuntimeError("crash")
        coach = AICoach(psychology_tracker=pt, _llm_fn=_mock_llm())
        ctx = coach.build_context()
        assert isinstance(ctx, str)

    def test_full_pipeline_with_all_mocked(self):
        dna = MagicMock()
        dna.generate_profile.return_value = {
            "status": "ready", "n_trades": 600,
            "trading_hours":    {"best": [], "worst": []},
            "trading_weekdays": {"best": [], "worst": []},
            "symbols":          {"best": [], "worst": []},
            "setups":           {"best": [], "worst": []},
            "position_sizing":  {"lot_buckets": [], "optimal_label": None, "optimal_win_rate": None},
            "psychological_weaknesses": [],
        }
        pt = MagicMock()
        pt.analyze_mood_patterns.return_value = {}

        journal = _make_journal_with_trades(_make_sample_trades(5))
        coach   = AICoach(
            journal=journal, dna=dna, psychology_tracker=pt,
            _llm_fn=_mock_llm("Vollaendige Antwort."),
        )
        result = coach.ask("Wie kann ich mich verbessern?")
        assert result == "Vollaendige Antwort."


# ─────────────────────────────────────────────────────────────────────────────
#  _default_llm_fn – ImportError ohne anthropic-Paket
# ─────────────────────────────────────────────────────────────────────────────

class TestDefaultLlmFn:

    def test_import_error_when_anthropic_missing(self):
        coach = AICoach()
        with patch.dict("sys.modules", {"anthropic": None}):
            with pytest.raises((ImportError, Exception)):
                coach._default_llm_fn("system", "user")

    def test_ask_catches_import_error_gracefully(self):
        coach = AICoach(_llm_fn=None)
        with patch.dict("sys.modules", {"anthropic": None}):
            result = coach.ask("Frage?")
        # Muss eine Fehlermeldung enthalten, kein Absturz
        assert isinstance(result, str)
        assert len(result) > 0
