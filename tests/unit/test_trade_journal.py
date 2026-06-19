"""
tests/unit/test_trade_journal.py
Unit-Tests fuer TradeJournal.

Abgedeckt:
  Initialisierung
    - DB wird erstellt, 'trades'-Tabelle existiert
    - Kontext-Manager (__enter__/__exit__)
    - Benutzerdefinierter db_path

  log_trade_open
    - Gibt int-ID zurueck
    - ID inkrementiert mit jedem neuen Trade
    - Pflichtfelder (symbol, direction) werden gespeichert
    - Optionale Felder (lot_size, entry_price, entry_time,
      regime, news_context, signal_confidence, setup)
    - Status ist nach Open immer 'open'
    - Extra-Felder landen in extra_json
    - entry_time Default = jetzt (ISO-String gesetzt)
    - entry_time als datetime-Objekt wird gespeichert
    - entry_time als ISO-String wird direkt uebernommen

  log_trade_close
    - Status wechselt auf 'closed'
    - exit_price, exit_time, pnl werden gesetzt
    - exit_time Default = jetzt
    - Unbekannte trade_id -> kein Fehler (UPDATE betrifft 0 Zeilen)
    - Kann mehrfach fuer denselben Trade aufgerufen werden (Idempotenz-Test)

  calculate_stats (bekannte Testdaten)
    - Leere DB -> alle Werte Null/None
    - Nur offene Trades -> 0 Trades
    - Trades ohne PnL (pnl=None) -> werden ignoriert
    - 5 Trades: 3 Gewinne, 2 Verluste -> Win-Rate, profit_factor, avg_win, avg_loss
    - Profit-Factor: nur Gewinne -> float('inf')
    - Profit-Factor: nur Verluste -> 0.0
    - Symbol-Filter: nur Trades fuer ein Symbol
    - Datums-Filter: Trades ausserhalb Zeitraum werden ausgeschlossen
    - best_trade und worst_trade korrekt
    - total_pnl = Summe aller PnL-Werte
    - n_trades zaehlt korrekt

  generate_report
    - 'daily' gibt Markdown-String zurueck
    - 'weekly' gibt Markdown-String zurueck
    - Unbekannte period -> ValueError
    - Report enthaelt # Ueberschrift
    - Report enthaelt Kennzahl-Tabelle
    - Report spiegelt calculate_stats korrekt wider
    - Kein-Trades-Fall: n_trades=0 erscheint im Report
    - profit_factor=inf erscheint als '∞'
    - case-insensitive period ('Daily', 'WEEKLY')

  OrderExecutor-Integration
    - trade_journal=None: kein Fehler bei open/close (Rueckwaertskompatibilitaet)
    - open_position ruft log_trade_open auf
    - close_position ruft log_trade_close mit der richtigen ID auf
    - Trade landet mit status='open' nach open, 'closed' nach close
"""

from __future__ import annotations

import math
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from src.journal.trade_journal import TradeJournal
from src.execution.order_executor import OrderExecutor


# ─── Hilfsfunktionen ─────────────────────────────────────────────────────────

def _journal(tmp_path: Path) -> TradeJournal:
    return TradeJournal(db_path=tmp_path / "journal.db")


def _open_trade(jnl: TradeJournal, **kwargs) -> int:
    defaults = {"symbol": "EURUSD", "direction": "buy", "lot_size": 1.0}
    defaults.update(kwargs)
    return jnl.log_trade_open(defaults)


def _close_trade(jnl: TradeJournal, trade_id: int, pnl: float) -> None:
    jnl.log_trade_close(trade_id, {"pnl": pnl})


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _dt(offset_days: float = 0) -> datetime:
    """Fester Zeitpunkt relativ zu heute."""
    return datetime(2026, 6, 18, 12, 0, 0, tzinfo=timezone.utc) + timedelta(days=offset_days)


# ─────────────────────────────────────────────────────────────────────────────
#  Initialisierung
# ─────────────────────────────────────────────────────────────────────────────

class TestInit:
    def test_db_file_created(self, tmp_path):
        db = tmp_path / "journal.db"
        TradeJournal(db_path=db).close()
        assert db.exists()

    def test_parent_dirs_created(self, tmp_path):
        db = tmp_path / "nested" / "dir" / "journal.db"
        TradeJournal(db_path=db).close()
        assert db.exists()

    def test_context_manager(self, tmp_path):
        with TradeJournal(db_path=tmp_path / "j.db") as jnl:
            tid = jnl.log_trade_open({"symbol": "EURUSD", "direction": "buy"})
        assert isinstance(tid, int)

    def test_trades_table_exists(self, tmp_path):
        import sqlite3
        db = tmp_path / "journal.db"
        with TradeJournal(db_path=db):
            pass
        conn = sqlite3.connect(str(db))
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        conn.close()
        assert "trades" in tables


# ─────────────────────────────────────────────────────────────────────────────
#  log_trade_open
# ─────────────────────────────────────────────────────────────────────────────

class TestLogTradeOpen:
    def test_returns_int(self, tmp_path):
        jnl = _journal(tmp_path)
        tid = _open_trade(jnl)
        jnl.close()
        assert isinstance(tid, int)

    def test_id_starts_at_one(self, tmp_path):
        jnl = _journal(tmp_path)
        tid = _open_trade(jnl)
        jnl.close()
        assert tid == 1

    def test_ids_increment(self, tmp_path):
        jnl = _journal(tmp_path)
        t1 = _open_trade(jnl)
        t2 = _open_trade(jnl)
        jnl.close()
        assert t2 == t1 + 1

    def test_status_is_open(self, tmp_path):
        import sqlite3
        db = tmp_path / "j.db"
        jnl = TradeJournal(db_path=db)
        _open_trade(jnl)
        jnl.close()
        conn = sqlite3.connect(str(db))
        row = conn.execute("SELECT status FROM trades WHERE id=1").fetchone()
        conn.close()
        assert row[0] == "open"

    def test_symbol_stored(self, tmp_path):
        import sqlite3
        db = tmp_path / "j.db"
        jnl = TradeJournal(db_path=db)
        jnl.log_trade_open({"symbol": "GBPUSD", "direction": "sell"})
        jnl.close()
        conn = sqlite3.connect(str(db))
        row = conn.execute("SELECT symbol FROM trades WHERE id=1").fetchone()
        conn.close()
        assert row[0] == "GBPUSD"

    def test_direction_stored(self, tmp_path):
        import sqlite3
        db = tmp_path / "j.db"
        jnl = TradeJournal(db_path=db)
        jnl.log_trade_open({"symbol": "EURUSD", "direction": "sell"})
        jnl.close()
        conn = sqlite3.connect(str(db))
        row = conn.execute("SELECT direction FROM trades WHERE id=1").fetchone()
        conn.close()
        assert row[0] == "sell"

    def test_optional_fields_stored(self, tmp_path):
        import sqlite3
        db = tmp_path / "j.db"
        jnl = TradeJournal(db_path=db)
        jnl.log_trade_open({
            "symbol": "EURUSD", "direction": "buy",
            "lot_size": 2.5,
            "entry_price": 1.08500,
            "regime": "TRENDING",
            "news_context": "NFP",
            "signal_confidence": 0.72,
            "setup": "EMA_CROSS",
        })
        jnl.close()
        conn = sqlite3.connect(str(db))
        row = conn.execute(
            "SELECT lot_size, entry_price, regime, news_context, signal_confidence, setup "
            "FROM trades WHERE id=1"
        ).fetchone()
        conn.close()
        assert row[0] == 2.5
        assert row[1] == pytest.approx(1.08500)
        assert row[2] == "TRENDING"
        assert row[3] == "NFP"
        assert row[4] == pytest.approx(0.72)
        assert row[5] == "EMA_CROSS"

    def test_entry_time_default_set(self, tmp_path):
        import sqlite3
        db = tmp_path / "j.db"
        jnl = TradeJournal(db_path=db)
        jnl.log_trade_open({"symbol": "EURUSD", "direction": "buy"})
        jnl.close()
        conn = sqlite3.connect(str(db))
        row = conn.execute("SELECT entry_time FROM trades WHERE id=1").fetchone()
        conn.close()
        assert row[0] is not None
        assert len(row[0]) > 10  # ISO-String

    def test_entry_time_as_datetime(self, tmp_path):
        import sqlite3
        db = tmp_path / "j.db"
        jnl = TradeJournal(db_path=db)
        dt = datetime(2026, 6, 18, 10, 30, 0, tzinfo=timezone.utc)
        jnl.log_trade_open({"symbol": "EURUSD", "direction": "buy", "entry_time": dt})
        jnl.close()
        conn = sqlite3.connect(str(db))
        row = conn.execute("SELECT entry_time FROM trades WHERE id=1").fetchone()
        conn.close()
        assert "2026-06-18" in row[0]

    def test_entry_time_as_string(self, tmp_path):
        import sqlite3
        db = tmp_path / "j.db"
        jnl = TradeJournal(db_path=db)
        jnl.log_trade_open({
            "symbol": "EURUSD", "direction": "buy",
            "entry_time": "2026-06-18T10:00:00+00:00",
        })
        jnl.close()
        conn = sqlite3.connect(str(db))
        row = conn.execute("SELECT entry_time FROM trades WHERE id=1").fetchone()
        conn.close()
        assert "2026-06-18" in row[0]

    def test_extra_fields_in_extra_json(self, tmp_path):
        import sqlite3, json
        db = tmp_path / "j.db"
        jnl = TradeJournal(db_path=db)
        jnl.log_trade_open({
            "symbol": "EURUSD", "direction": "buy",
            "custom_field": "value123",
        })
        jnl.close()
        conn = sqlite3.connect(str(db))
        row = conn.execute("SELECT extra_json FROM trades WHERE id=1").fetchone()
        conn.close()
        extra = json.loads(row[0])
        assert extra.get("custom_field") == "value123"


# ─────────────────────────────────────────────────────────────────────────────
#  log_trade_close
# ─────────────────────────────────────────────────────────────────────────────

class TestLogTradeClose:
    def test_status_becomes_closed(self, tmp_path):
        import sqlite3
        db = tmp_path / "j.db"
        jnl = TradeJournal(db_path=db)
        tid = _open_trade(jnl)
        _close_trade(jnl, tid, 100.0)
        jnl.close()
        conn = sqlite3.connect(str(db))
        row = conn.execute("SELECT status FROM trades WHERE id=?", (tid,)).fetchone()
        conn.close()
        assert row[0] == "closed"

    def test_pnl_stored(self, tmp_path):
        import sqlite3
        db = tmp_path / "j.db"
        jnl = TradeJournal(db_path=db)
        tid = _open_trade(jnl)
        _close_trade(jnl, tid, 250.75)
        jnl.close()
        conn = sqlite3.connect(str(db))
        row = conn.execute("SELECT pnl FROM trades WHERE id=?", (tid,)).fetchone()
        conn.close()
        assert row[0] == pytest.approx(250.75)

    def test_negative_pnl_stored(self, tmp_path):
        import sqlite3
        db = tmp_path / "j.db"
        jnl = TradeJournal(db_path=db)
        tid = _open_trade(jnl)
        _close_trade(jnl, tid, -150.50)
        jnl.close()
        conn = sqlite3.connect(str(db))
        row = conn.execute("SELECT pnl FROM trades WHERE id=?", (tid,)).fetchone()
        conn.close()
        assert row[0] == pytest.approx(-150.50)

    def test_exit_price_stored(self, tmp_path):
        import sqlite3
        db = tmp_path / "j.db"
        jnl = TradeJournal(db_path=db)
        tid = _open_trade(jnl)
        jnl.log_trade_close(tid, {"exit_price": 1.09000, "pnl": 50.0})
        jnl.close()
        conn = sqlite3.connect(str(db))
        row = conn.execute("SELECT exit_price FROM trades WHERE id=?", (tid,)).fetchone()
        conn.close()
        assert row[0] == pytest.approx(1.09000)

    def test_exit_time_default_set(self, tmp_path):
        import sqlite3
        db = tmp_path / "j.db"
        jnl = TradeJournal(db_path=db)
        tid = _open_trade(jnl)
        jnl.log_trade_close(tid, {"pnl": 10.0})
        jnl.close()
        conn = sqlite3.connect(str(db))
        row = conn.execute("SELECT exit_time FROM trades WHERE id=?", (tid,)).fetchone()
        conn.close()
        assert row[0] is not None

    def test_unknown_trade_id_no_error(self, tmp_path):
        """UPDATE auf nicht-existente ID darf keinen Fehler werfen."""
        jnl = _journal(tmp_path)
        jnl.log_trade_close(9999, {"pnl": 100.0})  # kein Fehler
        jnl.close()

    def test_exit_time_as_datetime(self, tmp_path):
        import sqlite3
        db = tmp_path / "j.db"
        jnl = TradeJournal(db_path=db)
        tid = _open_trade(jnl)
        dt = datetime(2026, 6, 18, 15, 0, 0, tzinfo=timezone.utc)
        jnl.log_trade_close(tid, {"exit_time": dt, "pnl": 0.0})
        jnl.close()
        conn = sqlite3.connect(str(db))
        row = conn.execute("SELECT exit_time FROM trades WHERE id=?", (tid,)).fetchone()
        conn.close()
        assert "2026-06-18" in row[0]


# ─────────────────────────────────────────────────────────────────────────────
#  calculate_stats – bekannte Testdaten
# ─────────────────────────────────────────────────────────────────────────────

def _seed_trades(jnl: TradeJournal, pnls: list, symbol: str = "EURUSD",
                 entry_time: datetime | None = None) -> None:
    """Legt abgeschlossene Trades mit den gegebenen PnL-Werten an."""
    base_time = entry_time or _dt(0)
    for i, pnl in enumerate(pnls):
        et = base_time + timedelta(minutes=i)
        tid = jnl.log_trade_open({
            "symbol": symbol, "direction": "buy",
            "entry_time": et,
        })
        jnl.log_trade_close(tid, {"pnl": pnl})


class TestCalculateStats:
    def _range(self):
        return _dt(-1), _dt(1)

    def test_empty_db_returns_zeros(self, tmp_path):
        jnl = _journal(tmp_path)
        stats = jnl.calculate_stats(*self._range())
        jnl.close()
        assert stats["n_trades"] == 0
        assert stats["win_rate"] == 0.0
        assert stats["profit_factor"] == 0.0
        assert stats["best_trade"] is None
        assert stats["worst_trade"] is None

    def test_open_trades_not_counted(self, tmp_path):
        jnl = _journal(tmp_path)
        _open_trade(jnl)  # bleibt offen
        stats = jnl.calculate_stats(*self._range())
        jnl.close()
        assert stats["n_trades"] == 0

    def test_trades_without_pnl_ignored(self, tmp_path):
        jnl = _journal(tmp_path)
        tid = _open_trade(jnl)
        jnl.log_trade_close(tid, {})  # kein pnl
        stats = jnl.calculate_stats(*self._range())
        jnl.close()
        assert stats["n_trades"] == 0

    def test_n_trades(self, tmp_path):
        jnl = _journal(tmp_path)
        _seed_trades(jnl, [100, -50, 200, -80, 150])
        stats = jnl.calculate_stats(*self._range())
        jnl.close()
        assert stats["n_trades"] == 5

    def test_win_rate(self, tmp_path):
        """3 Gewinne, 2 Verluste -> 60% Win-Rate."""
        jnl = _journal(tmp_path)
        _seed_trades(jnl, [100, -50, 200, -80, 150])
        stats = jnl.calculate_stats(*self._range())
        jnl.close()
        assert stats["win_rate"] == pytest.approx(0.6)

    def test_profit_factor(self, tmp_path):
        """Sum(Gewinne)=450, Sum(Verluste)=130 -> PF=450/130≈3.46."""
        jnl = _journal(tmp_path)
        _seed_trades(jnl, [100, -50, 200, -80, 150])
        stats = jnl.calculate_stats(*self._range())
        jnl.close()
        assert stats["profit_factor"] == pytest.approx(450 / 130, rel=1e-6)

    def test_avg_win(self, tmp_path):
        """Durchschn. Gewinn = (100+200+150)/3 = 150."""
        jnl = _journal(tmp_path)
        _seed_trades(jnl, [100, -50, 200, -80, 150])
        stats = jnl.calculate_stats(*self._range())
        jnl.close()
        assert stats["avg_win"] == pytest.approx(150.0)

    def test_avg_loss(self, tmp_path):
        """Durchschn. Verlust-Betrag = (50+80)/2 = 65."""
        jnl = _journal(tmp_path)
        _seed_trades(jnl, [100, -50, 200, -80, 150])
        stats = jnl.calculate_stats(*self._range())
        jnl.close()
        assert stats["avg_loss"] == pytest.approx(65.0)

    def test_total_pnl(self, tmp_path):
        """Gesamt-PnL = 100 - 50 + 200 - 80 + 150 = 320."""
        jnl = _journal(tmp_path)
        _seed_trades(jnl, [100, -50, 200, -80, 150])
        stats = jnl.calculate_stats(*self._range())
        jnl.close()
        assert stats["total_pnl"] == pytest.approx(320.0)

    def test_best_trade(self, tmp_path):
        jnl = _journal(tmp_path)
        _seed_trades(jnl, [100, -50, 200, -80, 150])
        stats = jnl.calculate_stats(*self._range())
        jnl.close()
        assert stats["best_trade"] == pytest.approx(200.0)

    def test_worst_trade(self, tmp_path):
        jnl = _journal(tmp_path)
        _seed_trades(jnl, [100, -50, 200, -80, 150])
        stats = jnl.calculate_stats(*self._range())
        jnl.close()
        assert stats["worst_trade"] == pytest.approx(-80.0)

    def test_profit_factor_all_wins(self, tmp_path):
        """Nur Gewinne -> profit_factor = inf."""
        jnl = _journal(tmp_path)
        _seed_trades(jnl, [100, 200, 300])
        stats = jnl.calculate_stats(*self._range())
        jnl.close()
        assert math.isinf(stats["profit_factor"])

    def test_profit_factor_all_losses(self, tmp_path):
        """Nur Verluste -> profit_factor = 0.0."""
        jnl = _journal(tmp_path)
        _seed_trades(jnl, [-100, -200])
        stats = jnl.calculate_stats(*self._range())
        jnl.close()
        assert stats["profit_factor"] == 0.0

    def test_symbol_filter(self, tmp_path):
        """Symbol-Filter: nur EURUSD-Trades zurueck."""
        jnl = _journal(tmp_path)
        _seed_trades(jnl, [100, 200], symbol="EURUSD")
        _seed_trades(jnl, [-50, -300], symbol="GBPUSD")
        stats = jnl.calculate_stats(*self._range(), symbol="EURUSD")
        jnl.close()
        assert stats["n_trades"] == 2
        assert stats["total_pnl"] == pytest.approx(300.0)

    def test_date_filter_excludes_old_trades(self, tmp_path):
        """Trades ausserhalb des Zeitraums werden nicht gezaehlt."""
        jnl = _journal(tmp_path)
        _seed_trades(jnl, [500], entry_time=_dt(-10))   # zu alt
        _seed_trades(jnl, [100, -50], entry_time=_dt(0))  # im Bereich
        stats = jnl.calculate_stats(_dt(-1), _dt(1))
        jnl.close()
        assert stats["n_trades"] == 2

    def test_single_trade(self, tmp_path):
        jnl = _journal(tmp_path)
        _seed_trades(jnl, [75.0])
        stats = jnl.calculate_stats(*self._range())
        jnl.close()
        assert stats["n_trades"] == 1
        assert stats["win_rate"] == pytest.approx(1.0)
        assert stats["best_trade"] == pytest.approx(75.0)
        assert stats["worst_trade"] == pytest.approx(75.0)

    def test_breakeven_trade(self, tmp_path):
        """PnL=0 zaehlt weder als Gewinn noch als Verlust."""
        jnl = _journal(tmp_path)
        _seed_trades(jnl, [0.0])
        stats = jnl.calculate_stats(*self._range())
        jnl.close()
        assert stats["n_trades"] == 1
        assert stats["win_rate"] == 0.0
        assert stats["profit_factor"] == 0.0

    def test_date_object_as_boundary(self, tmp_path):
        """date-Objekte (ohne Uhrzeit) werden als Tagesgrenzen interpretiert."""
        jnl = _journal(tmp_path)
        _seed_trades(jnl, [100], entry_time=datetime(2026, 6, 18, 12, 0, tzinfo=timezone.utc))
        stats = jnl.calculate_stats(date(2026, 6, 18), date(2026, 6, 18))
        jnl.close()
        assert stats["n_trades"] == 1


# ─────────────────────────────────────────────────────────────────────────────
#  generate_report
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerateReport:
    def test_daily_returns_string(self, tmp_path):
        jnl = _journal(tmp_path)
        report = jnl.generate_report("daily")
        jnl.close()
        assert isinstance(report, str)

    def test_weekly_returns_string(self, tmp_path):
        jnl = _journal(tmp_path)
        report = jnl.generate_report("weekly")
        jnl.close()
        assert isinstance(report, str)

    def test_unknown_period_raises(self, tmp_path):
        jnl = _journal(tmp_path)
        with pytest.raises(ValueError, match="period"):
            jnl.generate_report("monthly")
        jnl.close()

    def test_report_has_markdown_heading(self, tmp_path):
        jnl = _journal(tmp_path)
        report = jnl.generate_report("daily")
        jnl.close()
        assert report.startswith("# QuantzAI")

    def test_report_has_table(self, tmp_path):
        jnl = _journal(tmp_path)
        report = jnl.generate_report("daily")
        jnl.close()
        assert "|" in report

    def test_report_contains_n_trades(self, tmp_path):
        jnl = _journal(tmp_path)
        report = jnl.generate_report("daily")
        jnl.close()
        assert "0" in report   # 0 Trades bei leerer DB

    def test_report_reflects_stats(self, tmp_path):
        """Report soll die tatsaechlichen Stats widerspiegeln."""
        jnl = _journal(tmp_path)
        # Trades mit entry_time = jetzt damit sie im 'daily'-Report erscheinen
        now = datetime.now(timezone.utc)
        tid1 = jnl.log_trade_open({"symbol": "EURUSD", "direction": "buy", "entry_time": now})
        jnl.log_trade_close(tid1, {"pnl": 100.0})
        tid2 = jnl.log_trade_open({"symbol": "EURUSD", "direction": "sell", "entry_time": now})
        jnl.log_trade_close(tid2, {"pnl": -50.0})
        report = jnl.generate_report("daily")
        jnl.close()
        # 2 Trades und Win-Rate 50% muss irgendwo im Report stehen
        assert "2" in report
        assert "50" in report or "50.0" in report

    def test_report_inf_profit_factor(self, tmp_path):
        jnl = _journal(tmp_path)
        now = datetime.now(timezone.utc)
        tid = jnl.log_trade_open({"symbol": "EURUSD", "direction": "buy", "entry_time": now})
        jnl.log_trade_close(tid, {"pnl": 200.0})
        report = jnl.generate_report("daily")
        jnl.close()
        assert "∞" in report

    def test_case_insensitive_daily(self, tmp_path):
        jnl = _journal(tmp_path)
        report = jnl.generate_report("Daily")
        jnl.close()
        assert isinstance(report, str)

    def test_case_insensitive_weekly(self, tmp_path):
        jnl = _journal(tmp_path)
        report = jnl.generate_report("WEEKLY")
        jnl.close()
        assert isinstance(report, str)

    def test_daily_keyword_in_report(self, tmp_path):
        jnl = _journal(tmp_path)
        report = jnl.generate_report("daily")
        jnl.close()
        assert "Taeglich" in report

    def test_weekly_keyword_in_report(self, tmp_path):
        jnl = _journal(tmp_path)
        report = jnl.generate_report("weekly")
        jnl.close()
        assert "Woechentlich" in report


# ─────────────────────────────────────────────────────────────────────────────
#  OrderExecutor-Integration
# ─────────────────────────────────────────────────────────────────────────────

def _paper_executor(tmp_path: Path, trade_journal=None) -> OrderExecutor:
    conn = MagicMock()
    type(conn).is_connected = PropertyMock(return_value=True)
    return OrderExecutor(
        connector=conn,
        live_trading_enabled=False,
        paper_trades_path=tmp_path / "pt.json",
        trade_journal=trade_journal,
    )


class TestOrderExecutorIntegration:
    def test_no_journal_no_error(self, tmp_path):
        """Ohne trade_journal: open/close laufen fehlerfrei."""
        ex = _paper_executor(tmp_path)
        pos = ex.open_position("EURUSD", "buy", 1.0, 1.07, 1.10)
        ex.close_position(pos["ticket"])

    def test_open_calls_log_trade_open(self, tmp_path):
        jnl = MagicMock(spec=TradeJournal)
        jnl.log_trade_open.return_value = 1
        ex = _paper_executor(tmp_path, trade_journal=jnl)
        ex.open_position("EURUSD", "buy", 1.0, 1.07, 1.10)
        jnl.log_trade_open.assert_called_once()

    def test_open_passes_symbol(self, tmp_path):
        jnl = MagicMock(spec=TradeJournal)
        jnl.log_trade_open.return_value = 42
        ex = _paper_executor(tmp_path, trade_journal=jnl)
        ex.open_position("GBPUSD", "sell", 0.5, 1.25, 1.20)
        call_kwargs = jnl.log_trade_open.call_args[0][0]
        assert call_kwargs["symbol"] == "GBPUSD"

    def test_close_calls_log_trade_close(self, tmp_path):
        jnl = MagicMock(spec=TradeJournal)
        jnl.log_trade_open.return_value = 7
        ex = _paper_executor(tmp_path, trade_journal=jnl)
        pos = ex.open_position("EURUSD", "buy", 1.0, 1.07, 1.10)
        ex.close_position(pos["ticket"])
        jnl.log_trade_close.assert_called_once()

    def test_close_uses_correct_journal_id(self, tmp_path):
        """log_trade_close muss mit der ID aus log_trade_open aufgerufen werden."""
        jnl = MagicMock(spec=TradeJournal)
        jnl.log_trade_open.return_value = 99
        ex = _paper_executor(tmp_path, trade_journal=jnl)
        pos = ex.open_position("EURUSD", "buy", 1.0, 1.07, 1.10)
        ex.close_position(pos["ticket"])
        close_id = jnl.log_trade_close.call_args[0][0]
        assert close_id == 99

    def test_full_roundtrip_in_real_journal(self, tmp_path):
        """Echter End-to-End: open -> log_trade_open -> close -> log_trade_close."""
        jnl = TradeJournal(db_path=tmp_path / "j.db")
        ex = _paper_executor(tmp_path, trade_journal=jnl)

        pos = ex.open_position("EURUSD", "buy", 1.0, 1.07, 1.10)
        ex.close_position(pos["ticket"])

        import sqlite3
        conn = sqlite3.connect(str(tmp_path / "j.db"))
        row = conn.execute("SELECT status, symbol, direction FROM trades WHERE id=1").fetchone()
        conn.close()
        jnl.close()

        assert row is not None
        assert row[0] == "closed"
        assert row[1] == "EURUSD"
        assert row[2] == "buy"

    def test_two_simultaneous_positions_tracked(self, tmp_path):
        """Zwei offene Positionen: jede bekommt ihre eigene Journal-ID."""
        jnl = MagicMock(spec=TradeJournal)
        jnl.log_trade_open.side_effect = [1, 2]
        ex = _paper_executor(tmp_path, trade_journal=jnl)
        pos1 = ex.open_position("EURUSD", "buy", 1.0, 1.07, 1.10)
        pos2 = ex.open_position("GBPUSD", "sell", 0.5, 1.25, 1.20)
        ex.close_position(pos1["ticket"])
        ex.close_position(pos2["ticket"])

        close_calls = [c[0][0] for c in jnl.log_trade_close.call_args_list]
        assert 1 in close_calls
        assert 2 in close_calls
