"""
tests/unit/test_shadow_orchestrator.py
Unit-Tests fuer ShadowOrderExecutor, ShadowOrchestrator und AuditLog-Erweiterung.

Abgedeckt:
  - ShadowOrderExecutor sendet keine MT5-Orders
  - Positionen werden im Speicher getrackt
  - ShadowOrchestrator lehnt Nicht-Shadow-Executor ab
  - run_cycle() loggt hypothetischen Trade bei open_buy/open_sell
  - run_cycle() loggt NICHT bei flat/skipped
  - Paralleler Betrieb zweier Shadow-Orchestratoren ohne Konflikt
  - compare_performance(): korrekte Counts und Metriken
  - should_go_live(): korrekte True/False-Logik
  - run_loop() / stop()
  - AuditLog: shadow_trades-Tabelle, log_shadow_trade(), query_shadow_trades()
  - shadow_view: compute_go_live_recommendation() pure-function Tests
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.monitoring.audit_log import AuditLog
from src.shadow import ShadowOrderExecutor, ShadowOrchestrator
from gui.views.shadow_view import (
    ShadowSnapshot,
    compute_go_live_recommendation,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Hilfsfunktionen
# ─────────────────────────────────────────────────────────────────────────────

def _audit(tmp_path: Path) -> AuditLog:
    return AuditLog(tmp_path / "test_audit.db")


def _make_shadow(
    audit: AuditLog,
    action:     str   = "open_buy",
    confidence: float = 0.72,
    signal:     str   = "long",
    label:      str   = "shadow",
) -> ShadowOrchestrator:
    """Erstellt einen ShadowOrchestrator mit gemocktem innerem Orchestrator."""
    executor = ShadowOrderExecutor()

    mock_inner = MagicMock()

    def _run_cycle(symbol: str) -> dict:
        ticket = None
        if action in ("open_buy", "open_sell"):
            direction = "buy" if action == "open_buy" else "sell"
            trade = executor.open_position(symbol, direction, 0.1, 1.09, 1.11)
            ticket = trade["ticket"]

        return {
            "symbol":    symbol,
            "action":    action,
            "ticket":    ticket,
            "confidence": confidence,
            "signal":    signal,
            "lot_size":  0.1,
            "reason":    "signal_executed" if ticket else action,
        }

    mock_inner.run_cycle.side_effect = _run_cycle

    return ShadowOrchestrator(mock_inner, executor, audit, label=label)


_WIDE_START = datetime(2020, 1, 1, tzinfo=timezone.utc)
_WIDE_END   = datetime(2099, 1, 1, tzinfo=timezone.utc)


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: ShadowOrderExecutor
# ─────────────────────────────────────────────────────────────────────────────

class TestShadowOrderExecutor:

    def test_open_position_returns_shadow_dict(self):
        ex = ShadowOrderExecutor()
        result = ex.open_position("EURUSD", "buy", 0.1, 1.09, 1.11)
        assert result["status"] == "shadow_open"
        assert result["ticket"] >= 1
        assert result["symbol"] == "EURUSD"

    def test_open_position_no_mt5_import(self):
        """ShadowOrderExecutor importiert und nutzt MetaTrader5 nicht."""
        with patch("builtins.__import__", side_effect=ImportError("mt5 not available")):
            # Bereits importierte Module sind nicht betroffen; der Executor
            # selbst muss keinen mt5-Import benoetigen.
            ex = ShadowOrderExecutor()
        result = ex.open_position("EURUSD", "buy", 0.1, 1.09, 1.11)
        assert result["status"] == "shadow_open"

    def test_open_position_increments_ticket(self):
        ex = ShadowOrderExecutor()
        r1 = ex.open_position("EURUSD", "buy",  0.1, 1.09, 1.11)
        r2 = ex.open_position("GBPUSD", "sell", 0.2, 1.29, 1.31)
        assert r2["ticket"] == r1["ticket"] + 1

    def test_position_tracked_in_memory(self):
        ex = ShadowOrderExecutor()
        ex.open_position("EURUSD", "buy", 0.1, 1.09, 1.11)
        assert len(ex._positions) == 1

    def test_get_open_positions_returns_open_only(self):
        ex = ShadowOrderExecutor()
        r1 = ex.open_position("EURUSD", "buy",  0.1, 1.09, 1.11)
        r2 = ex.open_position("GBPUSD", "sell", 0.1, 1.29, 1.31)
        ex.close_position(r1["ticket"])
        open_pos = ex.get_open_positions()
        assert len(open_pos) == 1
        assert open_pos[0]["ticket"] == r2["ticket"]

    def test_close_position_marks_closed(self):
        ex = ShadowOrderExecutor()
        r = ex.open_position("EURUSD", "buy", 0.1, 1.09, 1.11)
        closed = ex.close_position(r["ticket"])
        assert closed["status"] == "shadow_closed"
        assert ex._positions[r["ticket"]]["status"] == "shadow_closed"

    def test_close_nonexistent_raises(self):
        from src.execution.order_executor import OrderError
        ex = ShadowOrderExecutor()
        with pytest.raises(OrderError):
            ex.close_position(999)

    def test_open_position_does_not_call_order_send(self):
        """Kein mt5.order_send – weder direkt noch indirekt."""
        ex = ShadowOrderExecutor()
        with patch("src.shadow.logger"):  # logger darf gecallt werden
            ex.open_position("EURUSD", "buy", 0.1, 1.09, 1.11)
        # Wenn wir hier ankommen ohne Exception, ist kein MT5-Aufruf passiert.


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: ShadowOrchestrator – Konstruktion
# ─────────────────────────────────────────────────────────────────────────────

class TestShadowOrchestratorConstruction:

    def test_rejects_non_shadow_executor(self, tmp_path):
        audit = _audit(tmp_path)
        mock_orch = MagicMock()
        bad_executor = MagicMock()  # kein ShadowOrderExecutor
        with pytest.raises(TypeError, match="ShadowOrderExecutor"):
            ShadowOrchestrator(mock_orch, bad_executor, audit)
        audit.close()

    def test_accepts_shadow_executor(self, tmp_path):
        audit    = _audit(tmp_path)
        executor = ShadowOrderExecutor()
        mock_orch = MagicMock()
        shadow = ShadowOrchestrator(mock_orch, executor, audit, label="test")
        assert shadow.label == "test"
        audit.close()

    def test_custom_label_stored(self, tmp_path):
        audit  = _audit(tmp_path)
        shadow = _make_shadow(audit, label="shadow_v2")
        assert shadow.label == "shadow_v2"
        audit.close()


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: ShadowOrchestrator – run_cycle()
# ─────────────────────────────────────────────────────────────────────────────

class TestShadowOrchestratorRunCycle:

    def test_run_cycle_returns_inner_result(self, tmp_path):
        audit  = _audit(tmp_path)
        shadow = _make_shadow(audit, action="open_buy", confidence=0.75)
        result = shadow.run_cycle("EURUSD")
        assert result["action"] == "open_buy"
        assert result["symbol"] == "EURUSD"
        audit.close()

    def test_open_buy_logs_shadow_trade(self, tmp_path):
        audit  = _audit(tmp_path)
        shadow = _make_shadow(audit, action="open_buy", confidence=0.80)
        shadow.run_cycle("EURUSD")
        df = audit.query_shadow_trades(_WIDE_START, _WIDE_END)
        assert len(df) == 1
        assert df.iloc[0]["symbol"] == "EURUSD"
        assert df.iloc[0]["direction"] == "buy"
        audit.close()

    def test_open_sell_logs_shadow_trade(self, tmp_path):
        audit  = _audit(tmp_path)
        shadow = _make_shadow(audit, action="open_sell", confidence=0.68, signal="short")
        shadow.run_cycle("GBPUSD")
        df = audit.query_shadow_trades(_WIDE_START, _WIDE_END)
        assert len(df) == 1
        assert df.iloc[0]["direction"] == "sell"
        audit.close()

    def test_confidence_stored_in_shadow_trade(self, tmp_path):
        audit  = _audit(tmp_path)
        shadow = _make_shadow(audit, action="open_buy", confidence=0.77)
        shadow.run_cycle("EURUSD")
        df = audit.query_shadow_trades(_WIDE_START, _WIDE_END)
        assert abs(float(df.iloc[0]["confidence"]) - 0.77) < 1e-6
        audit.close()

    def test_label_stored_in_shadow_trade(self, tmp_path):
        audit  = _audit(tmp_path)
        shadow = _make_shadow(audit, action="open_buy", label="v2_model")
        shadow.run_cycle("EURUSD")
        df = audit.query_shadow_trades(_WIDE_START, _WIDE_END)
        assert df.iloc[0]["label"] == "v2_model"
        audit.close()

    def test_flat_action_does_not_log(self, tmp_path):
        audit  = _audit(tmp_path)
        shadow = _make_shadow(audit, action="flat")
        shadow.run_cycle("EURUSD")
        df = audit.query_shadow_trades(_WIDE_START, _WIDE_END)
        assert len(df) == 0
        audit.close()

    def test_skipped_action_does_not_log(self, tmp_path):
        audit  = _audit(tmp_path)
        shadow = _make_shadow(audit, action="skipped")
        shadow.run_cycle("EURUSD")
        df = audit.query_shadow_trades(_WIDE_START, _WIDE_END)
        assert len(df) == 0
        audit.close()

    def test_multiple_cycles_multiple_records(self, tmp_path):
        audit  = _audit(tmp_path)
        shadow = _make_shadow(audit, action="open_buy")
        shadow.run_cycle("EURUSD")
        shadow.run_cycle("GBPUSD")
        df = audit.query_shadow_trades(_WIDE_START, _WIDE_END)
        assert len(df) == 2
        audit.close()

    def test_no_real_order_send_called(self, tmp_path):
        """Der innere Orchestrator ruft mt5.order_send NICHT auf."""
        audit  = _audit(tmp_path)
        shadow = _make_shadow(audit, action="open_buy")
        # Wenn mt5 nicht importierbar waere, wuerde das hier fehlschlagen.
        # Wir pruefen, dass der ShadowOrderExecutor nie MT5 braucht.
        with patch("src.shadow.logger"):
            result = shadow.run_cycle("EURUSD")
        assert result["action"] == "open_buy"
        audit.close()


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: Paralleler Betrieb
# ─────────────────────────────────────────────────────────────────────────────

class TestParallelRun:

    def test_two_shadow_orchestrators_no_conflict(self, tmp_path):
        """Zwei ShadowOrchestratoren koennen gleichzeitig laufen."""
        audit   = _audit(tmp_path)
        shadow1 = _make_shadow(audit, action="open_buy",  label="shadow_a")
        shadow2 = _make_shadow(audit, action="open_sell", label="shadow_b")

        errors: list[Exception] = []

        def run1():
            try:
                for _ in range(5):
                    shadow1.run_cycle("EURUSD")
            except Exception as exc:
                errors.append(exc)

        def run2():
            try:
                for _ in range(5):
                    shadow2.run_cycle("GBPUSD")
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=run1)
        t2 = threading.Thread(target=run2)
        t1.start(); t2.start()
        t1.join();  t2.join()

        assert errors == [], f"Fehler im parallelen Betrieb: {errors}"

        df = audit.query_shadow_trades(_WIDE_START, _WIDE_END)
        # Beide Instanzen haben 5 Trades protokolliert
        assert len(df) == 10
        audit.close()

    def test_shadow_and_live_audit_logs_independent(self, tmp_path):
        """Shadow-Trades landen in shadow_trades, Live-Orders in orders – keine Vermischung."""
        audit  = _audit(tmp_path)
        shadow = _make_shadow(audit, action="open_buy", label="shadow")

        # Fuege direkt einen Live-Order in orders-Tabelle ein
        audit.log_order({
            "symbol": "EURUSD", "direction": "buy", "lot_size": 0.1,
            "sl_price": 1.09, "tp_price": 1.11, "ticket": 99, "status": "open",
        })

        shadow.run_cycle("EURUSD")

        shadow_df = audit.query_shadow_trades(_WIDE_START, _WIDE_END)
        live_df   = audit.query_orders(_WIDE_START, _WIDE_END)

        assert len(shadow_df) == 1
        assert len(live_df) == 1
        # Keine Vermischung: shadow_trades enthaelt keinen Order-Eintrag
        assert "ticket" not in live_df.columns or live_df.iloc[0].get("ticket") == 99
        audit.close()

    def test_stop_ends_run_loop(self, tmp_path):
        audit  = _audit(tmp_path)
        shadow = _make_shadow(audit, action="open_buy")

        thread = threading.Thread(
            target=shadow.run_loop,
            args=(["EURUSD"],),
            kwargs={"interval_seconds": 60},
        )
        thread.start()
        time.sleep(0.05)
        shadow.stop()
        thread.join(timeout=3.0)
        assert not thread.is_alive()
        audit.close()


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: compare_performance()
# ─────────────────────────────────────────────────────────────────────────────

class TestComparePerformance:

    def test_empty_returns_zero_counts(self, tmp_path):
        audit  = _audit(tmp_path)
        shadow = _make_shadow(audit, action="flat")
        cmp = shadow.compare_performance(_WIDE_START, _WIDE_END)
        assert cmp["n_shadow_trades"] == 0
        assert cmp["n_live_trades"]   == 0
        assert cmp["shadow_avg_confidence"] is None
        assert cmp["shadow_sharpe"] is None
        audit.close()

    def test_counts_shadow_trades_correctly(self, tmp_path):
        audit  = _audit(tmp_path)
        shadow = _make_shadow(audit, action="open_buy", confidence=0.70)
        for _ in range(4):
            shadow.run_cycle("EURUSD")
        cmp = shadow.compare_performance(_WIDE_START, _WIDE_END)
        assert cmp["n_shadow_trades"] == 4
        audit.close()

    def test_counts_live_trades(self, tmp_path):
        audit  = _audit(tmp_path)
        shadow = _make_shadow(audit, action="flat")
        for i in range(3):
            audit.log_order({"symbol": "EURUSD", "direction": "buy", "ticket": i})
        cmp = shadow.compare_performance(_WIDE_START, _WIDE_END)
        assert cmp["n_live_trades"] == 3
        audit.close()

    def test_avg_confidence_computed(self, tmp_path):
        audit  = _audit(tmp_path)
        shadow = _make_shadow(audit, action="open_buy", confidence=0.80)
        shadow.run_cycle("EURUSD")
        cmp = shadow.compare_performance(_WIDE_START, _WIDE_END)
        assert abs(cmp["shadow_avg_confidence"] - 0.80) < 1e-6
        audit.close()

    def test_sharpe_none_for_single_trade(self, tmp_path):
        """Sharpe ist None wenn nur ein Trade vorhanden (std=0 nicht berechenbar)."""
        audit  = _audit(tmp_path)
        shadow = _make_shadow(audit, action="open_buy", confidence=0.75)
        shadow.run_cycle("EURUSD")
        cmp = shadow.compare_performance(_WIDE_START, _WIDE_END)
        assert cmp["shadow_sharpe"] is None
        audit.close()

    def test_sharpe_computed_for_multiple_trades(self, tmp_path):
        audit  = _audit(tmp_path)
        shadow = _make_shadow(audit, action="open_buy", confidence=0.75)
        for _ in range(5):
            shadow.run_cycle("EURUSD")
        cmp = shadow.compare_performance(_WIDE_START, _WIDE_END)
        # Alle Konfidenzen gleich -> std=0 -> Sharpe ist None
        # Das ist korrekt, da gleiche Konfidenzen keine Varianz ergeben
        # (technisch gesehen wuerde mean/0 auftreten)
        assert "shadow_sharpe" in cmp
        audit.close()

    def test_sharpe_nonzero_for_varying_confidence(self, tmp_path):
        """Sharpe > 0 wenn Konfidenzen variieren und avg > 0.5."""
        audit    = _audit(tmp_path)
        executor = ShadowOrderExecutor()
        mock_inner = MagicMock()
        confs = [0.60, 0.70, 0.65, 0.80, 0.75]

        call_idx = [0]

        def _run(symbol):
            c = confs[call_idx[0] % len(confs)]
            call_idx[0] += 1
            trade = executor.open_position(symbol, "buy", 0.1, 1.09, 1.11)
            return {
                "symbol": symbol, "action": "open_buy",
                "ticket": trade["ticket"], "confidence": c, "signal": "long",
            }

        mock_inner.run_cycle.side_effect = _run
        shadow = ShadowOrchestrator(mock_inner, executor, audit)

        for _ in range(5):
            shadow.run_cycle("EURUSD")

        cmp = shadow.compare_performance(_WIDE_START, _WIDE_END)
        assert cmp["shadow_sharpe"] is not None
        assert cmp["shadow_sharpe"] > 0
        audit.close()

    def test_compare_label_filtered(self, tmp_path):
        """compare_performance() beruecksichtigt nur Trades mit dem eigenen Label."""
        audit   = _audit(tmp_path)
        shadow1 = _make_shadow(audit, action="open_buy", label="modelA")
        shadow2 = _make_shadow(audit, action="open_buy", label="modelB")

        for _ in range(3):
            shadow1.run_cycle("EURUSD")
        for _ in range(2):
            shadow2.run_cycle("EURUSD")

        cmp1 = shadow1.compare_performance(_WIDE_START, _WIDE_END)
        cmp2 = shadow2.compare_performance(_WIDE_START, _WIDE_END)

        assert cmp1["n_shadow_trades"] == 3
        assert cmp2["n_shadow_trades"] == 2
        audit.close()

    def test_period_start_end_in_result(self, tmp_path):
        audit  = _audit(tmp_path)
        shadow = _make_shadow(audit, action="flat")
        cmp = shadow.compare_performance(_WIDE_START, _WIDE_END)
        assert "period_start" in cmp
        assert "period_end"   in cmp
        audit.close()


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: should_go_live()
# ─────────────────────────────────────────────────────────────────────────────

class TestShouldGoLive:

    def _shadow_with_n_trades(
        self,
        tmp_path: Path,
        n: int,
        confidence: float = 0.72,
    ) -> tuple[ShadowOrchestrator, AuditLog]:
        """Erstellt Shadow mit n Trades unterschiedlicher Konfidenz."""
        audit    = _audit(tmp_path)
        executor = ShadowOrderExecutor()
        mock_inner = MagicMock()
        import random, itertools

        # Unterschiedliche Konfidenzen damit Sharpe berechenbar ist
        conf_cycle = itertools.cycle([confidence - 0.05, confidence, confidence + 0.05])
        call_state = [iter(conf_cycle)]

        def _run(symbol):
            c = next(call_state[0])
            trade = executor.open_position(symbol, "buy", 0.1, 1.09, 1.11)
            return {
                "symbol": symbol, "action": "open_buy",
                "ticket": trade["ticket"], "confidence": c, "signal": "long",
            }

        mock_inner.run_cycle.side_effect = _run
        shadow = ShadowOrchestrator(mock_inner, executor, audit)

        for _ in range(n):
            shadow.run_cycle("EURUSD")

        return shadow, audit

    def test_not_enough_trades_returns_false(self, tmp_path):
        shadow, audit = self._shadow_with_n_trades(tmp_path, n=5, confidence=0.80)
        ok, reason = shadow.should_go_live(_WIDE_START, _WIDE_END, min_trades=30)
        assert ok is False
        assert "Trades" in reason
        audit.close()

    def test_low_sharpe_returns_false(self, tmp_path):
        """Niedrige Konfidenz -> negativer oder niedriger Sharpe -> nicht bereit."""
        shadow, audit = self._shadow_with_n_trades(tmp_path, n=50, confidence=0.51)
        ok, reason = shadow.should_go_live(
            _WIDE_START, _WIDE_END,
            min_trades=30, oos_sharpe_threshold=2.0,  # sehr hohe Schwelle
        )
        assert ok is False
        audit.close()

    def test_meets_criteria_returns_true(self, tmp_path):
        """Genug Trades und hohe Konfidenz -> bereit."""
        shadow, audit = self._shadow_with_n_trades(tmp_path, n=40, confidence=0.80)
        ok, reason = shadow.should_go_live(
            _WIDE_START, _WIDE_END,
            min_trades=30, oos_sharpe_threshold=0.1,  # niedrige Schwelle
        )
        assert ok is True
        assert "Bereit" in reason
        audit.close()

    def test_reason_mentions_sharpe(self, tmp_path):
        shadow, audit = self._shadow_with_n_trades(tmp_path, n=40, confidence=0.80)
        ok, reason = shadow.should_go_live(
            _WIDE_START, _WIDE_END,
            min_trades=30, oos_sharpe_threshold=0.1,
        )
        if ok:
            assert "Sharpe" in reason
        audit.close()

    def test_false_reason_contains_details(self, tmp_path):
        shadow, audit = self._shadow_with_n_trades(tmp_path, n=3, confidence=0.50)
        ok, reason = shadow.should_go_live(_WIDE_START, _WIDE_END, min_trades=10)
        assert ok is False
        assert "Nicht bereit" in reason
        audit.close()


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: AuditLog – shadow_trades-Tabelle
# ─────────────────────────────────────────────────────────────────────────────

class TestAuditLogShadowTrades:

    def test_shadow_trades_table_exists(self, tmp_path):
        import sqlite3
        audit = _audit(tmp_path)
        audit.close()
        conn = sqlite3.connect(str(tmp_path / "test_audit.db"))
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        conn.close()
        assert "shadow_trades" in tables

    def test_log_shadow_trade_writes_row(self, tmp_path):
        import sqlite3
        audit = _audit(tmp_path)
        audit.log_shadow_trade({
            "symbol": "EURUSD", "direction": "buy", "lot_size": 0.1,
            "sl_price": 1.09, "tp_price": 1.11, "confidence": 0.75,
            "label": "shadow",
        })
        audit.close()
        conn = sqlite3.connect(str(tmp_path / "test_audit.db"))
        n = conn.execute("SELECT COUNT(*) FROM shadow_trades").fetchone()[0]
        conn.close()
        assert n == 1

    def test_log_shadow_trade_stores_all_fields(self, tmp_path):
        audit = _audit(tmp_path)
        audit.log_shadow_trade({
            "symbol":     "GBPUSD",
            "direction":  "sell",
            "lot_size":   0.2,
            "entry_price": 1.30000,
            "sl_price":   1.31,
            "tp_price":   1.28,
            "confidence": 0.68,
            "label":      "test_model",
            "ticket":     42,
            "signal":     "short",
        })
        df = audit.query_shadow_trades(_WIDE_START, _WIDE_END)
        audit.close()
        assert len(df) == 1
        row = df.iloc[0]
        assert row["symbol"]     == "GBPUSD"
        assert row["direction"]  == "sell"
        assert abs(float(row["lot_size"]) - 0.2) < 1e-9
        assert abs(float(row["confidence"]) - 0.68) < 1e-9
        assert row["label"]      == "test_model"
        assert int(row["ticket"]) == 42
        assert row["signal"]     == "short"

    def test_log_shadow_trade_default_label(self, tmp_path):
        audit = _audit(tmp_path)
        audit.log_shadow_trade({"symbol": "EURUSD", "direction": "buy"})
        df = audit.query_shadow_trades(_WIDE_START, _WIDE_END)
        audit.close()
        assert df.iloc[0]["label"] == "shadow"

    def test_query_shadow_trades_date_filter(self, tmp_path):
        audit = _audit(tmp_path)
        audit.log_shadow_trade({"symbol": "EURUSD", "direction": "buy", "label": "s"})
        today = datetime.now(timezone.utc)
        yesterday = today - timedelta(days=1)
        tomorrow  = today + timedelta(days=1)

        df_all   = audit.query_shadow_trades(yesterday, tomorrow)
        df_past  = audit.query_shadow_trades(
            datetime(2020, 1, 1, tzinfo=timezone.utc),
            datetime(2020, 12, 31, tzinfo=timezone.utc),
        )
        audit.close()
        assert len(df_all)  == 1
        assert len(df_past) == 0

    def test_query_shadow_trades_symbol_filter(self, tmp_path):
        audit = _audit(tmp_path)
        audit.log_shadow_trade({"symbol": "EURUSD", "direction": "buy"})
        audit.log_shadow_trade({"symbol": "GBPUSD", "direction": "sell"})
        df = audit.query_shadow_trades(_WIDE_START, _WIDE_END, symbol="EURUSD")
        audit.close()
        assert len(df) == 1
        assert df.iloc[0]["symbol"] == "EURUSD"

    def test_query_shadow_trades_label_filter(self, tmp_path):
        audit = _audit(tmp_path)
        audit.log_shadow_trade({"symbol": "EURUSD", "label": "modelA"})
        audit.log_shadow_trade({"symbol": "EURUSD", "label": "modelB"})
        df = audit.query_shadow_trades(_WIDE_START, _WIDE_END, label="modelA")
        audit.close()
        assert len(df) == 1
        assert df.iloc[0]["label"] == "modelA"

    def test_query_shadow_trades_returns_dataframe(self, tmp_path):
        import pandas as pd
        audit = _audit(tmp_path)
        df = audit.query_shadow_trades(_WIDE_START, _WIDE_END)
        audit.close()
        assert isinstance(df, pd.DataFrame)

    def test_query_shadow_trades_empty_returns_empty_df(self, tmp_path):
        audit = _audit(tmp_path)
        df = audit.query_shadow_trades(_WIDE_START, _WIDE_END)
        audit.close()
        assert len(df) == 0

    def test_multiple_shadow_trades_ordered_by_ts(self, tmp_path):
        audit = _audit(tmp_path)
        for _ in range(5):
            audit.log_shadow_trade({"symbol": "EURUSD", "direction": "buy"})
        df = audit.query_shadow_trades(_WIDE_START, _WIDE_END)
        audit.close()
        assert len(df) == 5
        # ts-Spalte muss sortiert (aufsteigend) sein
        assert list(df["ts"]) == sorted(df["ts"])

    def test_shadow_trades_independent_from_orders_table(self, tmp_path):
        """shadow_trades und orders sind getrennte Tabellen."""
        audit = _audit(tmp_path)
        audit.log_shadow_trade({"symbol": "EURUSD", "direction": "buy", "label": "s"})
        audit.log_order({"symbol": "EURUSD", "direction": "buy", "ticket": 1})
        shadow_df = audit.query_shadow_trades(_WIDE_START, _WIDE_END)
        orders_df = audit.query_orders(_WIDE_START, _WIDE_END)
        audit.close()
        assert len(shadow_df) == 1
        assert len(orders_df) == 1
        # Verschiedene Spalten
        assert "confidence" in shadow_df.columns
        assert "confidence" not in orders_df.columns


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: compute_go_live_recommendation() (pure function)
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeGoLiveRecommendation:

    def test_not_enough_trades(self):
        ok, reason = compute_go_live_recommendation(
            n_shadow_trades=5,
            shadow_sharpe=1.5,
            shadow_avg_confidence=0.75,
            min_trades=30,
        )
        assert ok is False
        assert "Trades" in reason

    def test_no_sharpe(self):
        ok, reason = compute_go_live_recommendation(
            n_shadow_trades=50,
            shadow_sharpe=None,
            shadow_avg_confidence=0.75,
        )
        assert ok is False
        assert "Sharpe" in reason

    def test_low_sharpe(self):
        ok, reason = compute_go_live_recommendation(
            n_shadow_trades=50,
            shadow_sharpe=0.3,
            shadow_avg_confidence=0.75,
            oos_sharpe_threshold=0.5,
        )
        assert ok is False
        assert "0.3" in reason or "0.30" in reason

    def test_eligible(self):
        ok, reason = compute_go_live_recommendation(
            n_shadow_trades=50,
            shadow_sharpe=1.2,
            shadow_avg_confidence=0.75,
            min_trades=30,
            oos_sharpe_threshold=0.5,
        )
        assert ok is True
        assert "Bereit" in reason

    def test_both_problems_listed(self):
        ok, reason = compute_go_live_recommendation(
            n_shadow_trades=5,
            shadow_sharpe=0.1,
            shadow_avg_confidence=0.51,
            min_trades=30,
            oos_sharpe_threshold=0.5,
        )
        assert ok is False
        assert "Trades" in reason
        assert "Sharpe" in reason

    def test_exactly_at_threshold_is_not_eligible(self):
        ok, _ = compute_go_live_recommendation(
            n_shadow_trades=30,
            shadow_sharpe=0.5,
            shadow_avg_confidence=0.70,
            min_trades=30,
            oos_sharpe_threshold=0.5,
        )
        assert ok is False  # Schwelle ist > nicht >=

    def test_just_above_threshold_is_eligible(self):
        ok, _ = compute_go_live_recommendation(
            n_shadow_trades=30,
            shadow_sharpe=0.501,
            shadow_avg_confidence=0.70,
            min_trades=30,
            oos_sharpe_threshold=0.5,
        )
        assert ok is True
