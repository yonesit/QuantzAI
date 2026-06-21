"""
tests/unit/test_sentiment_lookahead.py
Look-ahead-Praeventionstests fuer SentimentHistory.

Alle Tests verwenden synthetische Daten (kein Netz, kein GDELT-Download).
Kernprinzip:
  Fuer H1-Bar bei Zeitpunkt T darf AUSSCHLIESSLICH Sentiment aus
  Nachrichten mit bucket_time < T verwendet werden.
"""
from __future__ import annotations

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from pathlib import Path

from src.data.gdelt_sentiment import SentimentHistory, _is_eurusd_relevant, _parse_v2tone


# ── Hilfsfunktionen ──────────────────────────────────────────────────────────

def _make_history(rows: list[dict], tmp_path: Path, window_hours: float = 2.0) -> SentimentHistory:
    """Erstellt eine SentimentHistory aus synthetischen Rohdaten."""
    df = pd.DataFrame(rows)
    df["bucket_time"] = pd.to_datetime(df["bucket_time"], utc=True)
    path = tmp_path / "gdelt_EURUSD.parquet"
    df.to_parquet(path, index=False)
    return SentimentHistory(path, window_hours=window_hours)


def _ts(hour: int, minute: int = 0, date: str = "2023-03-15") -> datetime:
    return datetime.fromisoformat(f"{date}T{hour:02d}:{minute:02d}:00+00:00")


# ── TestIsEurusdRelevant ──────────────────────────────────────────────────────

class TestIsEurusdRelevant:
    def test_econ_theme_matches(self):
        assert _is_eurusd_relevant("ECON_INFLATION,12;WB_UNRELATED,5", "unknown.com")

    def test_central_bank_theme_matches(self):
        assert _is_eurusd_relevant("CENTRAL_BANK,45", "unknown.com")

    def test_wb613_theme_matches(self):
        assert _is_eurusd_relevant("WB_613_FINANCE,20", "unknown.com")

    def test_financial_domain_matches(self):
        assert _is_eurusd_relevant("", "bloomberg.com")

    def test_reuters_matches(self):
        assert _is_eurusd_relevant("", "reuters.com - Business")

    def test_irrelevant_does_not_match(self):
        assert not _is_eurusd_relevant("SPORTS_FOOTBALL,10", "espn.com")

    def test_empty_strings_do_not_match(self):
        assert not _is_eurusd_relevant("", "")


# ── TestParseV2Tone ───────────────────────────────────────────────────────────

class TestParseV2Tone:
    def test_normal_value(self):
        assert _parse_v2tone("2.35,4.21,1.86,6.07,0.15") == pytest.approx(2.35)

    def test_negative_value(self):
        assert _parse_v2tone("-5.10,2.00,7.10,9.10,0.20") == pytest.approx(-5.10)

    def test_empty_string_returns_none(self):
        assert _parse_v2tone("") is None

    def test_malformed_returns_none(self):
        assert _parse_v2tone("not_a_number,1.0") is None


# ── TestSentimentHistoryLookahead ─────────────────────────────────────────────

class TestSentimentHistoryLookahead:
    """Kerntest: kein Look-ahead."""

    def test_no_news_returns_zero(self, tmp_path):
        hist = _make_history(
            [{"bucket_time": _ts(14, 0), "avg_tone": 5.0, "n_articles": 10}],
            tmp_path,
        )
        # H1-Bar 30 Stunden spaeter – weit ausserhalb des 2h-Fensters
        result = hist.get_sentiment_series(pd.Series([_ts(20, 0)]))
        assert result[0] == pytest.approx(0.0)

    def test_news_exactly_at_bar_time_not_used(self, tmp_path):
        """bucket_time=T → darf NICHT fuer H1-Bar bei T verwendet werden."""
        hist = _make_history(
            [{"bucket_time": _ts(14, 0), "avg_tone": 20.0, "n_articles": 5}],
            tmp_path,
        )
        result = hist.get_sentiment_series(pd.Series([_ts(14, 0)]))
        # Die einzige News ist bei 14:00 – gleich wie Bar-Zeit → ausgeschlossen
        assert result[0] == pytest.approx(0.0), (
            "News mit bucket_time == bar_time darf nicht verwendet werden"
        )

    def test_news_after_bar_not_used(self, tmp_path):
        """bucket_time > T → kein Look-ahead."""
        hist = _make_history(
            [{"bucket_time": _ts(15, 0), "avg_tone": 20.0, "n_articles": 5}],
            tmp_path,
        )
        result = hist.get_sentiment_series(pd.Series([_ts(14, 0)]))
        assert result[0] == pytest.approx(0.0), (
            "News nach der Bar-Zeit darf nicht verwendet werden"
        )

    def test_news_one_minute_before_bar_is_used(self, tmp_path):
        """bucket_time = T - 1 min → SOLL verwendet werden."""
        hist = _make_history(
            [{"bucket_time": _ts(13, 59), "avg_tone": 15.0, "n_articles": 4}],
            tmp_path,
        )
        result = hist.get_sentiment_series(pd.Series([_ts(14, 0)]))
        # 15.0 / 30.0 = 0.5
        assert result[0] == pytest.approx(15.0 / 30.0)

    def test_news_exactly_2h_before_bar_is_used(self, tmp_path):
        """bucket_time = T - 2h → liegt am linken Rand des Fensters → verwendet."""
        hist = _make_history(
            [{"bucket_time": _ts(12, 0), "avg_tone": 9.0, "n_articles": 2}],
            tmp_path,
        )
        result = hist.get_sentiment_series(pd.Series([_ts(14, 0)]))
        assert result[0] == pytest.approx(9.0 / 30.0)

    def test_news_more_than_2h_before_not_used(self, tmp_path):
        """bucket_time = T - 2h - 1min → ausserhalb des Fensters."""
        hist = _make_history(
            [{"bucket_time": _ts(11, 59), "avg_tone": 9.0, "n_articles": 2}],
            tmp_path,
        )
        result = hist.get_sentiment_series(pd.Series([_ts(14, 0)]))
        assert result[0] == pytest.approx(0.0)

    def test_weighted_average_of_multiple_buckets(self, tmp_path):
        """Mehrere Buckets werden gewichtet gemittelt."""
        # bucket 13:00: tone=+30, n=1  → gewichtet 30
        # bucket 13:30: tone=-30, n=3  → gewichtet -90
        # Gesamt: (-90+30)/(1+3) = -60/4 = -15 → normiert: -15/30 = -0.5
        hist = _make_history([
            {"bucket_time": _ts(13, 0),  "avg_tone":  30.0, "n_articles": 1},
            {"bucket_time": _ts(13, 30), "avg_tone": -30.0, "n_articles": 3},
        ], tmp_path)
        result = hist.get_sentiment_series(pd.Series([_ts(14, 0)]))
        assert result[0] == pytest.approx(-0.5)

    def test_score_clipped_to_minus_one_plus_one(self, tmp_path):
        """Tone > 30 wird auf +1 geclippt."""
        hist = _make_history(
            [{"bucket_time": _ts(13, 0), "avg_tone": 90.0, "n_articles": 1}],
            tmp_path,
        )
        result = hist.get_sentiment_series(pd.Series([_ts(14, 0)]))
        assert result[0] == pytest.approx(1.0)

    def test_empty_db_returns_zeros(self, tmp_path):
        """Leere Datenbank → alle Scores sind 0."""
        hist = _make_history(
            [{"bucket_time": _ts(13, 0), "avg_tone": 5.0, "n_articles": 1}],
            tmp_path,
        )
        ts = pd.Series([_ts(13, 0), _ts(14, 0), _ts(15, 0)])
        result = hist.get_sentiment_series(ts)
        # Bar 13:00: keine News vor 13:00 → 0
        assert result[0] == pytest.approx(0.0)
        assert len(result) == 3

    def test_custom_window_hours(self, tmp_path):
        """window_hours=1 darf nur Nachrichten der letzten 1h verwenden."""
        # bucket 12:30 → liegt in 2h-Fenster aber NICHT im 1h-Fenster vor 14:00
        # bucket 13:15 → liegt in 1h-Fenster (14:00 - 1h = 13:00 <= 13:15 < 14:00) ✓
        hist = _make_history([
            {"bucket_time": _ts(12, 30), "avg_tone": 30.0, "n_articles": 1},
            {"bucket_time": _ts(13, 15), "avg_tone": -6.0, "n_articles": 1},
        ], tmp_path, window_hours=1.0)
        result = hist.get_sentiment_series(pd.Series([_ts(14, 0)]))
        # Nur 13:15: -6/30 = -0.2
        assert result[0] == pytest.approx(-6.0 / 30.0)

    def test_multiple_bars_independent_windows(self, tmp_path):
        """Jede H1-Bar hat ihr eigenes unabhaengiges Fenster."""
        # bucket 10:30: tone=+12
        # bucket 12:30: tone=-12
        # Bar 12:00 sieht nur 10:30 (12:30 ist nach 12:00)
        # Bar 14:00 sieht nur 12:30 (10:30 ist vor 2h-Fenster)
        hist = _make_history([
            {"bucket_time": _ts(10, 30), "avg_tone": 12.0, "n_articles": 1},
            {"bucket_time": _ts(12, 30), "avg_tone": -12.0, "n_articles": 1},
        ], tmp_path)
        ts = pd.Series([_ts(12, 0), _ts(14, 0)])
        result = hist.get_sentiment_series(ts)
        assert result[0] == pytest.approx(12.0 / 30.0)   # Bar 12:00
        assert result[1] == pytest.approx(-12.0 / 30.0)  # Bar 14:00

    def test_order_preserved_for_unsorted_input(self, tmp_path):
        """Eingabe-Reihenfolge der Zeitstempel wird im Ergebnis beibehalten."""
        hist = _make_history([
            {"bucket_time": _ts(10, 0), "avg_tone": 15.0, "n_articles": 1},
            {"bucket_time": _ts(13, 0), "avg_tone": -15.0, "n_articles": 1},
        ], tmp_path)
        # Absichtlich unsortiert eingeben: zuerst 14:00, dann 11:00
        ts = pd.Series([_ts(14, 0), _ts(11, 0)])
        result = hist.get_sentiment_series(ts)
        assert result[0] == pytest.approx(-15.0 / 30.0)  # Bar 14:00 sieht 13:00
        assert result[1] == pytest.approx(15.0 / 30.0)   # Bar 11:00 sieht 10:00

    def test_file_not_found_raises(self, tmp_path):
        """Fehlende Parquet-Datei loest FileNotFoundError aus."""
        with pytest.raises(FileNotFoundError, match="GDELT"):
            SentimentHistory(tmp_path / "nonexistent.parquet")
