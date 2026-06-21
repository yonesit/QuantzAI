"""
tests/unit/test_practice_session.py
Unit-Tests fuer src/journal/practice_session.py.

Abgedeckt:
  TestPracticeSessionNoLookahead
    - current_candles liefert niemals zukuenftige Kerzen
    - current_candles Laenge == cursor + 1
    - Nach advance() immer noch kein Look-ahead
    - _all_candles enthaelt alle Kerzen (interner Guard)
    - cursor startet bei 0

  TestPracticeSessionAdvance
    - advance(1) rueckt cursor um 1 vor
    - advance(5) rueckt cursor um 5 vor
    - advance() am Ende gibt 0 zurueck
    - advance() gibt tatsaechliche Schritte zurueck

  TestPracticeSessionPositions
    - open_position gibt gueltige ID zurueck
    - close_position gibt PracticeResult zurueck
    - P&L-Berechnung Buy korrekt
    - P&L-Berechnung Sell korrekt
    - Counterfactual ist entgegengesetzt
    - SL-Erreichung schliesst automatisch
    - TP-Erreichung schliesst automatisch
    - close_position(unbekannte ID) wirft KeyError
    - open_position(ungueltiges direction) wirft ValueError

  TestPracticeSessionStats
    - get_stats() bei leerer Session
    - get_stats() mit Gewinnen und Verlusten
    - win_rate korrekt berechnet
    - total_pnl korrekt berechnet
    - get_open_positions() gibt Kopie zurueck
    - get_closed_positions() gibt Kopie zurueck
    - get_results() gibt Kopie zurueck

  TestPracticeTradesNotInRealJournal
    - Uebungs-Trades landen NICHT im echten TradeJournal

  TestPracticeSessionFromRange
    - from_range mit gemockten Parquet-Daten
    - from_range wirft ReplayDataNotFoundError bei fehlenden Dateien

  TestPracticeSessionStore
    - save_result speichert in eigene SQLite-DB
    - get_global_stats korrekt
    - getrennte DB-Datei von TradeJournal
    - close_position landet NICHT im TradeJournal (Doppelsicherung)
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.journal.practice_session import (
    PracticePosition,
    PracticeResult,
    PracticeSession,
    PracticeSessionStore,
)
from src.journal.replay import ReplayDataNotFoundError


# ─────────────────────────────────────────────────────────────────────────────
#  Hilfsfunktionen
# ─────────────────────────────────────────────────────────────────────────────

def _make_candles(n: int, base_price: float = 1.1000) -> list[dict]:
    """Erstellt eine Liste von n Dummy-Candle-Dicts."""
    candles = []
    for i in range(n):
        p = base_price + i * 0.0001
        candles.append({
            "time":   f"2024-01-01T{i:02d}:00:00+00:00",
            "open":   p,
            "high":   p + 0.0005,
            "low":    p - 0.0005,
            "close":  p,
            "volume": 100.0,
        })
    return candles


def _make_session(n: int = 20, base_price: float = 1.1000) -> PracticeSession:
    return PracticeSession(
        symbol="EURUSD",
        timeframe="H1",
        all_candles=_make_candles(n, base_price),
    )


# ─────────────────────────────────────────────────────────────────────────────
#  TestPracticeSessionNoLookahead
# ─────────────────────────────────────────────────────────────────────────────

class TestPracticeSessionNoLookahead:
    def test_initial_cursor_is_zero(self):
        s = _make_session(10)
        assert s.cursor == 0

    def test_current_candles_length_at_start(self):
        s = _make_session(10)
        assert len(s.current_candles) == 1  # cursor=0 → [0:1]

    def test_current_candles_length_equals_cursor_plus_one(self):
        s = _make_session(10)
        s.advance(3)
        assert len(s.current_candles) == s.cursor + 1

    def test_current_candles_never_contains_future_candles(self):
        s = _make_session(10)
        s.advance(4)
        cursor = s.cursor
        visible = s.current_candles
        # Alle sichtbaren Kerzen haben Index < len(all_candles) und duerfen nicht
        # nach dem Cursor liegen
        assert len(visible) == cursor + 1

    def test_all_candles_has_total_data(self):
        s = _make_session(10)
        # intern alle Kerzen vorhanden
        assert s.total_candles == 10

    def test_current_candles_after_advance_excludes_future(self):
        candles = _make_candles(10)
        s = PracticeSession("EURUSD", "H1", candles)
        s.advance(2)
        assert s.current_candles[-1]["time"] == candles[2]["time"]
        # Zukuenftige Kerze nicht sichtbar
        assert candles[3]["time"] not in [c["time"] for c in s.current_candles]

    def test_no_future_data_exposed_in_current_candles(self):
        s = _make_session(20)
        for step in range(5):
            s.advance(1)
            visible = s.current_candles
            assert len(visible) == s.cursor + 1
            # keine Kerze ausserhalb des Cursors
            assert visible[-1] == s.current_candle


# ─────────────────────────────────────────────────────────────────────────────
#  TestPracticeSessionAdvance
# ─────────────────────────────────────────────────────────────────────────────

class TestPracticeSessionAdvance:
    def test_advance_one_step(self):
        s = _make_session(10)
        moved = s.advance(1)
        assert moved == 1
        assert s.cursor == 1

    def test_advance_five_steps(self):
        s = _make_session(20)
        moved = s.advance(5)
        assert moved == 5
        assert s.cursor == 5

    def test_advance_at_end_returns_zero(self):
        s = _make_session(5)
        s.advance(4)  # cursor=4 = end
        assert s.is_at_end
        moved = s.advance()
        assert moved == 0
        assert s.cursor == 4

    def test_advance_clips_to_end(self):
        s = _make_session(5)
        moved = s.advance(100)
        assert s.cursor == 4
        assert moved == 4  # 4 steps from 0 to 4

    def test_advance_returns_actual_steps(self):
        s = _make_session(5)
        s.advance(3)  # cursor=3
        moved = s.advance(3)  # nur 1 schritt moeglich
        assert moved == 1
        assert s.cursor == 4

    def test_is_at_end_false_initially(self):
        s = _make_session(5)
        assert not s.is_at_end

    def test_is_at_end_true_at_last_candle(self):
        s = _make_session(3)
        s.advance(2)
        assert s.is_at_end


# ─────────────────────────────────────────────────────────────────────────────
#  TestPracticeSessionPositions
# ─────────────────────────────────────────────────────────────────────────────

class TestPracticeSessionPositions:
    def test_open_position_returns_id(self):
        s = _make_session(10)
        pid = s.open_position("buy", 0.1, sl=1.0950, tp=1.1100)
        assert isinstance(pid, int)
        assert pid > 0

    def test_open_position_ids_unique(self):
        s = _make_session(10)
        pid1 = s.open_position("buy", 0.1, sl=1.09, tp=1.11)
        pid2 = s.open_position("sell", 0.1, sl=1.11, tp=1.09)
        assert pid1 != pid2

    def test_close_position_returns_practice_result(self):
        s = _make_session(10)
        pid = s.open_position("buy", 0.1, sl=1.09, tp=1.12)
        s.advance(3)
        result = s.close_position(pid)
        assert isinstance(result, PracticeResult)

    def test_pnl_buy_positive_when_price_rises(self):
        # Basis-Preis steigt von 1.1000 auf 1.1003 (nach 3 Schritten)
        s = _make_session(10, base_price=1.1000)
        pid = s.open_position("buy", lot_size=1.0, sl=1.090, tp=1.120)
        s.advance(3)  # close = 1.1003
        result = s.close_position(pid)
        assert result.position.pnl is not None
        assert result.position.pnl > 0

    def test_pnl_sell_positive_when_price_falls(self):
        # Sell: entry bei 1.1009 (nach advance(9)), close bei 1.1000 (weiter zurueck)
        # Wir simulieren manuell: entry_price > exit_price
        candles = _make_candles(10, base_price=1.1000)
        s = PracticeSession("EURUSD", "H1", candles)
        s.advance(9)  # cursor=9, close=1.1009
        pid = s.open_position("sell", lot_size=1.0, sl=1.1050, tp=1.0950)
        # Manuell Position schliessen mit niedrigerem Preis
        result = s.close_position(pid, exit_price=1.1000)
        assert result.position.pnl is not None
        assert result.position.pnl > 0  # sell: entry 1.1009 > exit 1.1000 → gewinn

    def test_pnl_calculation_buy_formula(self):
        # (exit - entry) * lot_size * 100000
        candles = _make_candles(2)
        s = PracticeSession("EURUSD", "H1", candles)
        pid = s.open_position("buy", lot_size=0.5, sl=1.0900, tp=1.1100)
        entry_price = s.current_candle["close"]
        exit_price = entry_price + 0.0010  # +10 pip
        result = s.close_position(pid, exit_price=exit_price)
        expected_pnl = 0.0010 * 0.5 * 100_000
        assert abs(result.position.pnl - expected_pnl) < 0.001

    def test_pnl_calculation_sell_formula(self):
        candles = _make_candles(2)
        s = PracticeSession("EURUSD", "H1", candles)
        pid = s.open_position("sell", lot_size=0.5, sl=1.1100, tp=1.0900)
        entry_price = s.current_candle["close"]
        exit_price = entry_price - 0.0010  # -10 pip → sell profit
        result = s.close_position(pid, exit_price=exit_price)
        expected_pnl = 0.0010 * 0.5 * 100_000
        assert abs(result.position.pnl - expected_pnl) < 0.001

    def test_counterfactual_is_opposite(self):
        candles = _make_candles(2)
        s = PracticeSession("EURUSD", "H1", candles)
        pid = s.open_position("buy", lot_size=1.0, sl=1.090, tp=1.120)
        result = s.close_position(pid, exit_price=candles[0]["close"] + 0.0010)
        # Counterfactual = negatives P&L
        assert abs(result.position.pnl + result.counterfactual_pnl) < 0.0001

    def test_close_unknown_position_raises_key_error(self):
        s = _make_session(5)
        with pytest.raises(KeyError):
            s.close_position(999)

    def test_open_invalid_direction_raises_value_error(self):
        s = _make_session(5)
        with pytest.raises(ValueError, match="direction"):
            s.open_position("long", 0.1, sl=1.09, tp=1.11)

    def test_sl_hit_auto_closes_buy_position(self):
        candles = [
            {"time": "2024-01-01T00:00:00+00:00", "open": 1.1010, "high": 1.1015,
             "low": 1.1005, "close": 1.1010, "volume": 100.0},
            # next candle: low goes below SL
            {"time": "2024-01-01T01:00:00+00:00", "open": 1.1010, "high": 1.1012,
             "low": 1.0950, "close": 1.0955, "volume": 100.0},
        ]
        s = PracticeSession("EURUSD", "H1", candles)
        pid = s.open_position("buy", lot_size=1.0, sl=1.1000, tp=1.1100)
        assert len(s.get_open_positions()) == 1
        s.advance(1)  # low=1.0950 < sl=1.1000
        assert len(s.get_open_positions()) == 0
        closed = s.get_closed_positions()
        assert len(closed) == 1
        assert closed[0].closed_by == "sl"

    def test_tp_hit_auto_closes_buy_position(self):
        candles = [
            {"time": "2024-01-01T00:00:00+00:00", "open": 1.1010, "high": 1.1015,
             "low": 1.1005, "close": 1.1010, "volume": 100.0},
            # next candle: high exceeds TP
            {"time": "2024-01-01T01:00:00+00:00", "open": 1.1010, "high": 1.1200,
             "low": 1.1005, "close": 1.1100, "volume": 100.0},
        ]
        s = PracticeSession("EURUSD", "H1", candles)
        pid = s.open_position("buy", lot_size=1.0, sl=1.0900, tp=1.1150)
        s.advance(1)  # high=1.1200 >= tp=1.1150
        closed = s.get_closed_positions()
        assert len(closed) == 1
        assert closed[0].closed_by == "tp"

    def test_sl_hit_auto_closes_sell_position(self):
        candles = [
            {"time": "2024-01-01T00:00:00+00:00", "open": 1.1010, "high": 1.1015,
             "low": 1.1005, "close": 1.1010, "volume": 100.0},
            # next candle: high exceeds sell SL
            {"time": "2024-01-01T01:00:00+00:00", "open": 1.1010, "high": 1.1100,
             "low": 1.1005, "close": 1.1010, "volume": 100.0},
        ]
        s = PracticeSession("EURUSD", "H1", candles)
        pid = s.open_position("sell", lot_size=1.0, sl=1.1050, tp=1.0900)
        s.advance(1)  # high=1.1100 >= sl=1.1050
        closed = s.get_closed_positions()
        assert len(closed) == 1
        assert closed[0].closed_by == "sl"

    def test_open_position_is_open_flag(self):
        s = _make_session(5)
        pid = s.open_position("buy", 0.1, sl=1.09, tp=1.12)
        open_pos = s.get_open_positions()
        assert len(open_pos) == 1
        assert open_pos[0].is_open is True

    def test_closed_position_is_not_open(self):
        s = _make_session(5)
        pid = s.open_position("buy", 0.1, sl=1.09, tp=1.12)
        s.close_position(pid)
        assert len(s.get_open_positions()) == 0
        closed = s.get_closed_positions()
        assert len(closed) == 1
        assert closed[0].is_open is False

    def test_closed_by_manual_when_closed_manually(self):
        s = _make_session(5)
        pid = s.open_position("buy", 0.1, sl=1.09, tp=1.12)
        result = s.close_position(pid)
        assert result.position.closed_by == "manual"


# ─────────────────────────────────────────────────────────────────────────────
#  TestPracticeSessionStats
# ─────────────────────────────────────────────────────────────────────────────

class TestPracticeSessionStats:
    def test_empty_stats(self):
        s = _make_session(5)
        stats = s.get_stats()
        assert stats["trade_count"] == 0
        assert stats["win_rate"] == 0.0
        assert stats["total_pnl"] == 0.0

    def test_stats_after_winning_trade(self):
        s = _make_session(10)
        pid = s.open_position("buy", 1.0, sl=1.090, tp=1.120)
        s.close_position(pid, exit_price=1.1020)  # +20 pip
        stats = s.get_stats()
        assert stats["trade_count"] == 1
        assert stats["wins"] == 1
        assert stats["losses"] == 0
        assert stats["win_rate"] == 1.0

    def test_stats_after_losing_trade(self):
        s = _make_session(10)
        pid = s.open_position("buy", 1.0, sl=1.0950, tp=1.1200)
        entry = s.current_candle["close"]
        s.close_position(pid, exit_price=entry - 0.0010)  # -10 pip
        stats = s.get_stats()
        assert stats["trade_count"] == 1
        assert stats["wins"] == 0
        assert stats["losses"] == 1
        assert stats["win_rate"] == 0.0

    def test_win_rate_calculation(self):
        s = _make_session(20)
        entry = s.current_candle["close"]
        p1 = s.open_position("buy", 1.0, sl=1.09, tp=1.12)
        s.close_position(p1, exit_price=entry + 0.001)  # win
        p2 = s.open_position("buy", 1.0, sl=1.09, tp=1.12)
        s.close_position(p2, exit_price=entry - 0.001)  # loss
        p3 = s.open_position("buy", 1.0, sl=1.09, tp=1.12)
        s.close_position(p3, exit_price=entry + 0.001)  # win
        stats = s.get_stats()
        assert stats["trade_count"] == 3
        assert stats["wins"] == 2
        assert abs(stats["win_rate"] - 2 / 3) < 0.001

    def test_total_pnl_sum(self):
        s = _make_session(20)
        entry = s.current_candle["close"]
        p1 = s.open_position("buy", 1.0, sl=1.09, tp=1.12)
        r1 = s.close_position(p1, exit_price=entry + 0.001)
        p2 = s.open_position("buy", 1.0, sl=1.09, tp=1.12)
        r2 = s.close_position(p2, exit_price=entry - 0.0005)
        stats = s.get_stats()
        expected = (r1.position.pnl or 0) + (r2.position.pnl or 0)
        assert abs(stats["total_pnl"] - expected) < 0.001

    def test_get_open_positions_returns_copy(self):
        s = _make_session(10)
        s.open_position("buy", 0.1, sl=1.09, tp=1.12)
        lst = s.get_open_positions()
        lst.clear()
        assert len(s.get_open_positions()) == 1

    def test_get_closed_positions_returns_copy(self):
        s = _make_session(10)
        pid = s.open_position("buy", 0.1, sl=1.09, tp=1.12)
        s.close_position(pid)
        lst = s.get_closed_positions()
        lst.clear()
        assert len(s.get_closed_positions()) == 1

    def test_get_results_returns_copy(self):
        s = _make_session(10)
        pid = s.open_position("buy", 0.1, sl=1.09, tp=1.12)
        s.close_position(pid)
        lst = s.get_results()
        lst.clear()
        assert len(s.get_results()) == 1


# ─────────────────────────────────────────────────────────────────────────────
#  TestPracticeTradesNotInRealJournal
# ─────────────────────────────────────────────────────────────────────────────

class TestPracticeTradesNotInRealJournal:
    def test_practice_result_has_no_trade_journal_reference(self):
        """PracticeResult enthaelt kein TradeJournal-Objekt."""
        s = _make_session(5)
        pid = s.open_position("buy", 0.1, sl=1.09, tp=1.12)
        result = s.close_position(pid)
        # Sicherstellen dass kein TradeJournal importiert oder referenziert wird
        import src.journal.practice_session as ps_mod
        assert not hasattr(ps_mod, "TradeJournal")

    def test_practice_session_does_not_write_to_trade_journal(self, tmp_path):
        """Kein Schreiben in eine TradeJournal-DB."""
        from src.journal.trade_journal import TradeJournal
        journal_db = tmp_path / "real_journal.db"
        journal = TradeJournal(db_path=str(journal_db))
        try:
            s = _make_session(5)
            pid = s.open_position("buy", 0.1, sl=1.09, tp=1.12)
            s.close_position(pid)
            # TradeJournal bleibt leer
            conn = __import__("sqlite3").connect(str(journal_db))
            rows = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
            conn.close()
            assert rows == 0
        finally:
            journal.close()

    def test_practice_session_store_uses_separate_db(self, tmp_path):
        """PracticeSessionStore schreibt in eigene DB, nicht in TradeJournal."""
        practice_db = tmp_path / "practice.db"
        journal_db  = tmp_path / "journal.db"
        store = PracticeSessionStore(db_path=str(practice_db))
        # TradeJournal-DB existiert nicht einmal
        assert not journal_db.exists()
        assert practice_db.exists()
        store.close()


# ─────────────────────────────────────────────────────────────────────────────
#  TestPracticeSessionFromRange
# ─────────────────────────────────────────────────────────────────────────────

class TestPracticeSessionFromRange:
    def _make_parquet(self, tmp_path: Path, n: int = 20) -> None:
        """Erstellt eine Dummy-Parquet-Datei fuer Tests."""
        idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        df = pd.DataFrame({
            "open":   [1.1000 + i * 0.0001 for i in range(n)],
            "high":   [1.1010 + i * 0.0001 for i in range(n)],
            "low":    [1.0990 + i * 0.0001 for i in range(n)],
            "close":  [1.1005 + i * 0.0001 for i in range(n)],
            "volume": [100.0] * n,
        }, index=idx)
        (tmp_path / "features").mkdir(exist_ok=True)
        df.to_parquet(tmp_path / "features" / "EURUSD_H1_20240101.parquet")

    def test_from_range_loads_candles(self, tmp_path):
        self._make_parquet(tmp_path, n=20)
        start = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
        end   = datetime(2024, 1, 1, 19, 0, tzinfo=timezone.utc)
        s = PracticeSession.from_range(
            symbol="EURUSD",
            timeframe="H1",
            start_dt=start,
            end_dt=end,
            features_dir=tmp_path / "features",
        )
        assert s.total_candles > 0
        assert s.symbol == "EURUSD"
        assert s.timeframe == "H1"

    def test_from_range_session_starts_at_cursor_zero(self, tmp_path):
        self._make_parquet(tmp_path, n=20)
        start = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
        end   = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
        s = PracticeSession.from_range(
            "EURUSD", "H1", start, end, features_dir=tmp_path / "features"
        )
        assert s.cursor == 0

    def test_from_range_no_parquet_raises(self, tmp_path):
        start = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
        end   = datetime(2024, 1, 2, 0, 0, tzinfo=timezone.utc)
        (tmp_path / "empty_features").mkdir()
        with pytest.raises(ReplayDataNotFoundError):
            PracticeSession.from_range(
                "EURUSD", "H1", start, end, features_dir=tmp_path / "empty_features"
            )

    def test_from_range_no_lookahead_after_load(self, tmp_path):
        self._make_parquet(tmp_path, n=20)
        start = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
        end   = datetime(2024, 1, 1, 19, 0, tzinfo=timezone.utc)
        s = PracticeSession.from_range(
            "EURUSD", "H1", start, end, features_dir=tmp_path / "features"
        )
        # Direkt nach Laden: nur 1 Kerze sichtbar (cursor=0)
        assert len(s.current_candles) == 1
        # Alle Daten trotzdem intern vorhanden
        assert s.total_candles > 1


# ─────────────────────────────────────────────────────────────────────────────
#  TestPracticeSessionStore
# ─────────────────────────────────────────────────────────────────────────────

class TestPracticeSessionStore:
    def _make_result(self, pnl: float = 50.0) -> PracticeResult:
        pos = PracticePosition(
            position_id=1,
            direction="buy",
            entry_price=1.1000,
            lot_size=0.1,
            sl=1.0950,
            tp=1.1100,
            entry_candle_idx=0,
            is_open=False,
            exit_price=1.1005,
            exit_candle_idx=3,
            pnl=pnl,
            closed_by="manual",
        )
        return PracticeResult(position=pos, counterfactual_pnl=-pnl)

    def test_save_result_stores_to_db(self, tmp_path):
        db = tmp_path / "practice.db"
        store = PracticeSessionStore(db_path=str(db))
        store.save_result(self._make_result(50.0), "EURUSD", "H1")
        store.close()
        import sqlite3
        conn = sqlite3.connect(str(db))
        row = conn.execute("SELECT COUNT(*) FROM practice_trades").fetchone()
        conn.close()
        assert row[0] == 1

    def test_save_result_returns_id(self, tmp_path):
        store = PracticeSessionStore(db_path=str(tmp_path / "p.db"))
        rid = store.save_result(self._make_result(100.0), "EURUSD", "H1")
        store.close()
        assert isinstance(rid, int)
        assert rid >= 1

    def test_get_global_stats_empty(self, tmp_path):
        store = PracticeSessionStore(db_path=str(tmp_path / "p.db"))
        stats = store.get_global_stats()
        store.close()
        assert stats["trade_count"] == 0
        assert stats["win_rate"] == 0.0

    def test_get_global_stats_with_wins_and_losses(self, tmp_path):
        store = PracticeSessionStore(db_path=str(tmp_path / "p.db"))
        store.save_result(self._make_result(pnl=100.0), "EURUSD", "H1")
        store.save_result(self._make_result(pnl=-50.0), "EURUSD", "H1")
        store.save_result(self._make_result(pnl=75.0), "EURUSD", "H1")
        stats = store.get_global_stats()
        store.close()
        assert stats["trade_count"] == 3
        assert stats["wins"] == 2
        assert stats["losses"] == 1
        assert abs(stats["win_rate"] - 2 / 3) < 0.001

    def test_global_stats_total_pnl(self, tmp_path):
        store = PracticeSessionStore(db_path=str(tmp_path / "p.db"))
        store.save_result(self._make_result(pnl=100.0), "EURUSD", "H1")
        store.save_result(self._make_result(pnl=50.0),  "EURUSD", "H1")
        stats = store.get_global_stats()
        store.close()
        assert abs(stats["total_pnl"] - 150.0) < 0.001

    def test_store_separate_from_trade_journal_path(self, tmp_path):
        practice_db = tmp_path / "practice.db"
        journal_db  = tmp_path / "journal.db"
        store = PracticeSessionStore(db_path=str(practice_db))
        store.save_result(self._make_result(), "EURUSD", "H1")
        store.close()
        # Journal-DB wurde nicht angefasst
        assert not journal_db.exists()
        # Practice-DB existiert
        assert practice_db.exists()

    def test_context_manager(self, tmp_path):
        db = tmp_path / "p.db"
        with PracticeSessionStore(db_path=str(db)) as store:
            store.save_result(self._make_result(), "EURUSD", "H1")
        # Nach __exit__: Verbindung geschlossen (kein Fehler)
        import sqlite3
        conn = sqlite3.connect(str(db))
        count = conn.execute("SELECT COUNT(*) FROM practice_trades").fetchone()[0]
        conn.close()
        assert count == 1
