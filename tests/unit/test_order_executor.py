"""
Unit-Tests fuer OrderExecutor.

MT5-Modul und Connector werden vollstaendig gemockt –
kein MT5-Terminal, kein Netzwerk noetig.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, call, patch, PropertyMock

import pytest

from src.execution.order_executor import OrderExecutor, OrderError


# ─────────────────────────────────────────────────────────────────────────────
#  Hilfsfunktionen
# ─────────────────────────────────────────────────────────────────────────────

def _connector(connected: bool = True) -> MagicMock:
    conn = MagicMock()
    type(conn).is_connected = PropertyMock(return_value=connected)
    conn.get_symbol_info.return_value = {
        "point": 0.00001, "digits": 5,
        "spread": 10, "swap_long": -7.0, "swap_short": 2.0,
    }
    return conn


def _paper_executor(tmp_path: Path, **kwargs) -> OrderExecutor:
    """Paper-Trading OrderExecutor mit tempoeraerer JSON-Datei."""
    paper_file = tmp_path / "paper_trades.json"
    return OrderExecutor(
        connector=_connector(),
        live_trading_enabled=False,
        paper_trades_path=paper_file,
        **kwargs,
    )


def _mt5_mock() -> MagicMock:
    """Minimal-Mock des mt5-Moduls fuer Live-Tests."""
    mt5 = MagicMock()
    mt5.ORDER_TYPE_BUY   = 0
    mt5.ORDER_TYPE_SELL  = 1
    mt5.TRADE_ACTION_DEAL = 1
    mt5.TRADE_ACTION_SLTP = 6
    mt5.ORDER_FILLING_IOC = 1
    mt5.TRADE_RETCODE_DONE = 10009

    # order_send gibt ein Erfolgs-Objekt zurueck
    ok_result = MagicMock()
    ok_result.retcode  = 10009  # TRADE_RETCODE_DONE
    ok_result.order    = 42
    ok_result.comment  = "Request completed"
    ok_result.price    = 1.10000
    ok_result.volume   = 0.1    # vollstaendige Fuellung
    mt5.order_send.return_value = ok_result

    # positions_get gibt eine offene Buy-Position zurueck
    pos = MagicMock()
    pos.ticket     = 42
    pos.symbol     = "EURUSD"
    pos.type       = 0        # BUY
    pos.volume     = 0.1
    pos.price_open = 1.10000
    pos.sl         = 1.0950
    pos.tp         = 1.1100
    mt5.positions_get.return_value = [pos]

    return mt5


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: Sicherheits-Flag
# ─────────────────────────────────────────────────────────────────────────────

class TestSafetyFlag:

    def test_paper_mode_default(self, tmp_path):
        """live_trading_enabled=False ist Default – kein Fehler."""
        ex = _paper_executor(tmp_path)
        assert ex._live is False

    def test_live_without_confirm_env_raises(self, tmp_path):
        """live_trading_enabled=True ohne CONFIRM_LIVE=yes wirft RuntimeError."""
        env = {k: v for k, v in os.environ.items() if k != "CONFIRM_LIVE"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(RuntimeError, match="CONFIRM_LIVE"):
                OrderExecutor(
                    connector=_connector(),
                    live_trading_enabled=True,
                )

    def test_live_with_wrong_confirm_raises(self, tmp_path):
        """CONFIRM_LIVE=no (falsch geschrieben) wirft ebenfalls RuntimeError."""
        with patch.dict(os.environ, {"CONFIRM_LIVE": "no"}):
            with pytest.raises(RuntimeError, match="CONFIRM_LIVE"):
                OrderExecutor(
                    connector=_connector(),
                    live_trading_enabled=True,
                )

    def test_live_with_correct_confirm_succeeds(self, tmp_path):
        """CONFIRM_LIVE=yes -> kein Fehler beim Init."""
        with patch.dict(os.environ, {"CONFIRM_LIVE": "yes"}):
            ex = OrderExecutor(
                connector=_connector(),
                live_trading_enabled=True,
                paper_trades_path=tmp_path / "pt.json",
            )
        assert ex._live is True

    def test_error_message_not_silent_fallback(self, tmp_path):
        """Fehlermeldung erklaert explizit: kein stiller Fallback."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(RuntimeError, match="(?i)kein stiller Fallback"):
                OrderExecutor(connector=_connector(), live_trading_enabled=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: Paper-Trading – mt5.order_send wird NIE aufgerufen
# ─────────────────────────────────────────────────────────────────────────────

class TestPaperNeverCallsMT5:

    def test_open_position_paper_never_calls_order_send(self, tmp_path):
        """Paper-Modus: open_position darf mt5.order_send NIEMALS aufrufen."""
        mt5 = _mt5_mock()
        with patch("src.execution.order_executor._load_mt5", return_value=mt5):
            ex = _paper_executor(tmp_path)
            ex.open_position("EURUSD", "buy", 0.1, 1.0950, 1.1100)
        mt5.order_send.assert_not_called()

    def test_close_position_paper_never_calls_order_send(self, tmp_path):
        """Paper-Modus: close_position darf mt5.order_send NIEMALS aufrufen."""
        mt5 = _mt5_mock()
        with patch("src.execution.order_executor._load_mt5", return_value=mt5):
            ex = _paper_executor(tmp_path)
            ex.open_position("EURUSD", "buy", 0.1, 1.0950, 1.1100)
            ex.close_position(1)
        mt5.order_send.assert_not_called()

    def test_update_trailing_stop_paper_never_calls_order_send(self, tmp_path):
        """Paper-Modus: update_trailing_stop darf mt5.order_send NIEMALS aufrufen."""
        mt5 = _mt5_mock()
        with patch("src.execution.order_executor._load_mt5", return_value=mt5):
            ex = _paper_executor(tmp_path)
            ex.open_position("EURUSD", "buy", 0.1, 1.0900, 1.1100)
            ex.update_trailing_stop(1, current_price=1.1050)
        mt5.order_send.assert_not_called()

    def test_paper_never_calls_load_mt5_at_all(self, tmp_path):
        """Paper-Modus laed das mt5-Modul ueberhaupt nicht."""
        with patch("src.execution.order_executor._load_mt5") as mock_loader:
            ex = _paper_executor(tmp_path)
            ex.open_position("EURUSD", "buy", 0.1, 1.0950, 1.1100)
            ex.close_position(1)
        mock_loader.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: Paper-Trades JSON-Datei
# ─────────────────────────────────────────────────────────────────────────────

class TestPaperTradesFile:

    def test_file_created_after_open(self, tmp_path):
        """paper_trades.json wird nach open_position erstellt."""
        paper_file = tmp_path / "paper_trades.json"
        ex = OrderExecutor(
            connector=_connector(),
            paper_trades_path=paper_file,
        )
        ex.open_position("EURUSD", "buy", 0.1, 1.0950, 1.1100)
        assert paper_file.exists()

    def test_file_contains_trade_entry(self, tmp_path):
        """JSON-Datei enthaelt den Trade-Eintrag mit korrekten Feldern."""
        paper_file = tmp_path / "paper_trades.json"
        ex = OrderExecutor(
            connector=_connector(),
            paper_trades_path=paper_file,
        )
        ex.open_position("EURUSD", "buy", 0.1, 1.0950, 1.1100)

        data = json.loads(paper_file.read_text(encoding="utf-8"))
        assert len(data) == 1
        trade = data[0]
        assert trade["symbol"]    == "EURUSD"
        assert trade["direction"] == "buy"
        assert trade["lot_size"]  == 0.1
        assert trade["sl_price"]  == 1.0950
        assert trade["tp_price"]  == 1.1100
        assert trade["status"]    == "open"
        assert "ticket" in trade
        assert "open_time" in trade

    def test_multiple_trades_written(self, tmp_path):
        """Mehrere offene Positionen erscheinen alle in der JSON-Datei."""
        paper_file = tmp_path / "paper_trades.json"
        ex = OrderExecutor(connector=_connector(), paper_trades_path=paper_file)
        ex.open_position("EURUSD", "buy",  0.1, 1.09, 1.11)
        ex.open_position("GBPUSD", "sell", 0.2, 1.28, 1.25)

        data = json.loads(paper_file.read_text(encoding="utf-8"))
        assert len(data) == 2
        symbols = {t["symbol"] for t in data}
        assert symbols == {"EURUSD", "GBPUSD"}

    def test_close_updates_status_in_file(self, tmp_path):
        """Nach close_position hat der Eintrag status='closed'."""
        paper_file = tmp_path / "paper_trades.json"
        ex = OrderExecutor(connector=_connector(), paper_trades_path=paper_file)
        result = ex.open_position("EURUSD", "buy", 0.1, 1.09, 1.11)
        ex.close_position(result["ticket"])

        data = json.loads(paper_file.read_text(encoding="utf-8"))
        assert data[0]["status"] == "closed"
        assert data[0]["close_time"] is not None

    def test_parent_dir_created_automatically(self, tmp_path):
        """Ausgabeverzeichnis wird automatisch erstellt falls nicht vorhanden."""
        paper_file = tmp_path / "nested" / "dir" / "paper_trades.json"
        ex = OrderExecutor(connector=_connector(), paper_trades_path=paper_file)
        ex.open_position("EURUSD", "buy", 0.1, 1.09, 1.11)
        assert paper_file.exists()

    def test_json_is_valid_after_operations(self, tmp_path):
        """Datei ist nach mehreren Operationen noch gueltiges JSON."""
        paper_file = tmp_path / "pt.json"
        ex = OrderExecutor(connector=_connector(), paper_trades_path=paper_file)
        t1 = ex.open_position("EURUSD", "buy", 0.1, 1.09, 1.11)["ticket"]
        ex.open_position("USDJPY", "sell", 0.3, 150.0, 148.0)
        ex.close_position(t1)
        data = json.loads(paper_file.read_text())
        assert isinstance(data, list)


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: open_position Paper-Modus (Rueckgabewerte)
# ─────────────────────────────────────────────────────────────────────────────

class TestOpenPositionPaper:

    def test_returns_dict(self, tmp_path):
        ex = _paper_executor(tmp_path)
        result = ex.open_position("EURUSD", "buy", 0.1, 1.09, 1.11)
        assert isinstance(result, dict)

    def test_result_has_ticket(self, tmp_path):
        ex = _paper_executor(tmp_path)
        result = ex.open_position("EURUSD", "buy", 0.1, 1.09, 1.11)
        assert "ticket" in result
        assert isinstance(result["ticket"], int)

    def test_tickets_increment(self, tmp_path):
        ex = _paper_executor(tmp_path)
        t1 = ex.open_position("EURUSD", "buy",  0.1, 1.09, 1.11)["ticket"]
        t2 = ex.open_position("GBPUSD", "sell", 0.1, 1.27, 1.24)["ticket"]
        assert t2 > t1

    def test_result_fields_correct(self, tmp_path):
        ex = _paper_executor(tmp_path)
        r = ex.open_position("EURUSD", "buy", 0.1, 1.0950, 1.1100)
        assert r["symbol"]    == "EURUSD"
        assert r["direction"] == "buy"
        assert r["lot_size"]  == 0.1
        assert r["sl_price"]  == 1.0950
        assert r["tp_price"]  == 1.1100
        assert r["status"]    == "open"

    def test_invalid_direction_raises(self, tmp_path):
        ex = _paper_executor(tmp_path)
        with pytest.raises(ValueError, match="direction"):
            ex.open_position("EURUSD", "long", 0.1, 1.09, 1.11)

    def test_invalid_lot_size_raises(self, tmp_path):
        ex = _paper_executor(tmp_path)
        with pytest.raises(ValueError, match="lot_size"):
            ex.open_position("EURUSD", "buy", -1.0, 1.09, 1.11)

    def test_direction_case_insensitive(self, tmp_path):
        ex = _paper_executor(tmp_path)
        r = ex.open_position("EURUSD", "BUY", 0.1, 1.09, 1.11)
        assert r["direction"] == "buy"


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: close_position Paper-Modus
# ─────────────────────────────────────────────────────────────────────────────

class TestClosePositionPaper:

    def test_close_returns_dict(self, tmp_path):
        ex = _paper_executor(tmp_path)
        t = ex.open_position("EURUSD", "buy", 0.1, 1.09, 1.11)["ticket"]
        result = ex.close_position(t)
        assert isinstance(result, dict)

    def test_close_sets_status_closed(self, tmp_path):
        ex = _paper_executor(tmp_path)
        t = ex.open_position("EURUSD", "buy", 0.1, 1.09, 1.11)["ticket"]
        result = ex.close_position(t)
        assert result["status"] == "closed"

    def test_close_nonexistent_ticket_raises(self, tmp_path):
        ex = _paper_executor(tmp_path)
        with pytest.raises(OrderError, match="nicht gefunden"):
            ex.close_position(999)

    def test_close_already_closed_raises(self, tmp_path):
        ex = _paper_executor(tmp_path)
        t = ex.open_position("EURUSD", "buy", 0.1, 1.09, 1.11)["ticket"]
        ex.close_position(t)
        with pytest.raises(OrderError, match="bereits geschlossen"):
            ex.close_position(t)

    def test_closed_position_disappears_from_open_positions(self, tmp_path):
        ex = _paper_executor(tmp_path)
        t = ex.open_position("EURUSD", "buy", 0.1, 1.09, 1.11)["ticket"]
        ex.close_position(t)
        open_pos = ex.get_open_positions()
        assert all(p["ticket"] != t for p in open_pos)


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: get_open_positions Paper-Modus
# ─────────────────────────────────────────────────────────────────────────────

class TestGetOpenPositionsPaper:

    def test_empty_initially(self, tmp_path):
        ex = _paper_executor(tmp_path)
        assert ex.get_open_positions() == []

    def test_returns_list(self, tmp_path):
        ex = _paper_executor(tmp_path)
        assert isinstance(ex.get_open_positions(), list)

    def test_one_open_after_open(self, tmp_path):
        ex = _paper_executor(tmp_path)
        ex.open_position("EURUSD", "buy", 0.1, 1.09, 1.11)
        assert len(ex.get_open_positions()) == 1

    def test_two_open_after_two_opens(self, tmp_path):
        ex = _paper_executor(tmp_path)
        ex.open_position("EURUSD", "buy",  0.1, 1.09, 1.11)
        ex.open_position("GBPUSD", "sell", 0.2, 1.28, 1.25)
        assert len(ex.get_open_positions()) == 2

    def test_count_decreases_after_close(self, tmp_path):
        ex = _paper_executor(tmp_path)
        t = ex.open_position("EURUSD", "buy", 0.1, 1.09, 1.11)["ticket"]
        ex.open_position("GBPUSD", "sell", 0.2, 1.28, 1.25)
        ex.close_position(t)
        assert len(ex.get_open_positions()) == 1


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: Trailing-Stop-Logik (Paper-Modus)
# ─────────────────────────────────────────────────────────────────────────────

class TestTrailingStopPaper:
    """
    Standardwerte: min_pips=10, step_pips=5, pip_size=0.0001
    d.h. min_dist = 0.0010, step = 0.0005
    """

    def _ex(self, tmp_path, **kwargs) -> OrderExecutor:
        defaults = dict(trailing_stop_min_pips=10.0, trailing_stop_step_pips=5.0, pip_size=0.0001)
        defaults.update(kwargs)
        return _paper_executor(tmp_path, **defaults)

    # ── LONG-Positionen ──────────────────────────────────────────────────────

    def test_long_sl_moves_up_when_price_rises(self, tmp_path):
        """LONG: SL steigt wenn Preis hoch genug steigt."""
        ex = self._ex(tmp_path)
        # open: SL=1.0900
        t = ex.open_position("EURUSD", "buy", 0.1, 1.0900, 1.1200)["ticket"]
        # Kurs steigt auf 1.1100 -> neuer SL = 1.1100 - 0.0010 = 1.1090
        # Bedingung: 1.1090 >= 1.0900 + 0.0005 -> True
        ex.update_trailing_stop(t, current_price=1.1100)
        pos = ex._paper_positions[t]
        assert pos["sl_price"] > 1.0900

    def test_long_sl_does_not_move_down(self, tmp_path):
        """LONG: SL darf sich nie nach unten bewegen."""
        ex = self._ex(tmp_path)
        t = ex.open_position("EURUSD", "buy", 0.1, 1.0950, 1.1200)["ticket"]
        # SL bereits bei 1.0950; Kurs faellt auf 1.0960 -> neues Kandidat 1.0950
        # Kandidat = 1.0960 - 0.0010 = 1.0950, Bedingung: 1.0950 >= 1.0950 + 0.0005 -> False
        ex.update_trailing_stop(t, current_price=1.0960)
        pos = ex._paper_positions[t]
        assert pos["sl_price"] == 1.0950  # unveraendert

    def test_long_no_update_if_step_too_small(self, tmp_path):
        """LONG: Kein Update wenn Preisanstieg kleiner als Schritt."""
        ex = self._ex(tmp_path)
        # SL=1.0900; Kurs=1.0914 -> Kandidat=1.0904; Bedingung: 1.0904 >= 1.0900+0.0005=1.0905 -> False
        t = ex.open_position("EURUSD", "buy", 0.1, 1.0900, 1.1200)["ticket"]
        ex.update_trailing_stop(t, current_price=1.0914)
        pos = ex._paper_positions[t]
        assert pos["sl_price"] == 1.0900

    def test_long_update_when_price_rises_enough(self, tmp_path):
        """LONG: Update wenn Preisanstieg >= step."""
        ex = self._ex(tmp_path)
        # SL=1.0900; Kurs=1.0920 -> Kandidat=1.0910; Bedingung: 1.0910 >= 1.0905 -> True
        t = ex.open_position("EURUSD", "buy", 0.1, 1.0900, 1.1200)["ticket"]
        ex.update_trailing_stop(t, current_price=1.0920)
        pos = ex._paper_positions[t]
        assert abs(pos["sl_price"] - 1.0910) < 1e-9

    # ── SHORT-Positionen ─────────────────────────────────────────────────────

    def test_short_sl_moves_down_when_price_falls(self, tmp_path):
        """SHORT: SL sinkt wenn Preis weit genug faellt."""
        ex = self._ex(tmp_path)
        # SL=1.1100; Kurs faellt auf 1.0900 -> Kandidat=1.0910; 1.0910 <= 1.1100-0.0005=1.1095 -> True
        t = ex.open_position("EURUSD", "sell", 0.1, 1.1100, 1.0800)["ticket"]
        ex.update_trailing_stop(t, current_price=1.0900)
        pos = ex._paper_positions[t]
        assert pos["sl_price"] < 1.1100

    def test_short_sl_does_not_move_up(self, tmp_path):
        """SHORT: SL darf sich nie nach oben bewegen."""
        ex = self._ex(tmp_path)
        # SL=1.1000; Kurs steigt auf 1.1050 -> Kandidat=1.1060; 1.1060 <= 1.1000-0.0005 -> False
        t = ex.open_position("EURUSD", "sell", 0.1, 1.1000, 1.0800)["ticket"]
        ex.update_trailing_stop(t, current_price=1.1050)
        pos = ex._paper_positions[t]
        assert pos["sl_price"] == 1.1000  # unveraendert

    def test_short_no_update_if_step_too_small(self, tmp_path):
        """SHORT: Kein Update wenn Preisrueckgang kleiner als Schritt."""
        ex = self._ex(tmp_path)
        # SL=1.1100; Kurs=1.1086 -> Kandidat=1.1096; 1.1096 <= 1.1100-0.0005=1.1095 -> False
        t = ex.open_position("EURUSD", "sell", 0.1, 1.1100, 1.0800)["ticket"]
        ex.update_trailing_stop(t, current_price=1.1086)
        pos = ex._paper_positions[t]
        assert pos["sl_price"] == 1.1100

    def test_short_update_when_price_falls_enough(self, tmp_path):
        """SHORT: Update wenn Preisrueckgang >= step."""
        ex = self._ex(tmp_path)
        # SL=1.1100; Kurs=1.1080 -> Kandidat=1.1090; 1.1090 <= 1.1095 -> True
        t = ex.open_position("EURUSD", "sell", 0.1, 1.1100, 1.0800)["ticket"]
        ex.update_trailing_stop(t, current_price=1.1080)
        pos = ex._paper_positions[t]
        assert abs(pos["sl_price"] - 1.1090) < 1e-9

    # ── Allgemein ────────────────────────────────────────────────────────────

    def test_trailing_stop_updates_file(self, tmp_path):
        """Trailing-Stop-Update persistiert in der JSON-Datei."""
        paper_file = tmp_path / "pt.json"
        ex = OrderExecutor(
            connector=_connector(),
            paper_trades_path=paper_file,
            trailing_stop_min_pips=10.0,
            trailing_stop_step_pips=5.0,
            pip_size=0.0001,
        )
        t = ex.open_position("EURUSD", "buy", 0.1, 1.0900, 1.1200)["ticket"]
        ex.update_trailing_stop(t, current_price=1.0920)

        data = json.loads(paper_file.read_text())
        assert abs(data[0]["sl_price"] - 1.0910) < 1e-9

    def test_trailing_stop_nonexistent_ticket_raises(self, tmp_path):
        ex = _paper_executor(tmp_path)
        with pytest.raises(OrderError):
            ex.update_trailing_stop(999, current_price=1.1000)

    def test_configurable_pip_size(self, tmp_path):
        """Pip-Groesse ist konfigurierbar (JPY-Paare: 0.01)."""
        # min_pips=10, step=5, pip_size=0.01 -> min_dist=0.10, step=0.05
        ex = _paper_executor(tmp_path, trailing_stop_min_pips=10.0,
                             trailing_stop_step_pips=5.0, pip_size=0.01)
        # SL=149.0; Kurs=148.0 -> Kandidat=148.0+0.10=148.10; 148.10 <= 149.0-0.05=148.95 -> True
        t = ex.open_position("USDJPY", "sell", 0.1, 149.0, 147.0)["ticket"]
        ex.update_trailing_stop(t, current_price=148.0)
        pos = ex._paper_positions[t]
        assert pos["sl_price"] < 149.0


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: Live-Modus (mt5 gemockt)
# ─────────────────────────────────────────────────────────────────────────────

class TestLiveMode:

    def _live_executor(self, tmp_path) -> tuple[OrderExecutor, MagicMock]:
        mt5 = _mt5_mock()
        with patch.dict(os.environ, {"CONFIRM_LIVE": "yes"}):
            ex = OrderExecutor(
                connector=_connector(),
                live_trading_enabled=True,
                paper_trades_path=tmp_path / "pt.json",
            )
        return ex, mt5

    def test_live_open_calls_order_send(self, tmp_path):
        """Live-Modus: open_position ruft mt5.order_send auf."""
        ex, mt5 = self._live_executor(tmp_path)
        with patch("src.execution.order_executor._load_mt5", return_value=mt5):
            ex.open_position("EURUSD", "buy", 0.1, 1.0950, 1.1100)
        mt5.order_send.assert_called_once()

    def test_live_open_uses_correct_action(self, tmp_path):
        """Live open_position nutzt TRADE_ACTION_DEAL."""
        ex, mt5 = self._live_executor(tmp_path)
        with patch("src.execution.order_executor._load_mt5", return_value=mt5):
            ex.open_position("EURUSD", "buy", 0.1, 1.0950, 1.1100)
        request = mt5.order_send.call_args[0][0]
        assert request["action"] == mt5.TRADE_ACTION_DEAL

    def test_live_open_buy_uses_correct_type(self, tmp_path):
        """Buy-Order nutzt ORDER_TYPE_BUY."""
        ex, mt5 = self._live_executor(tmp_path)
        with patch("src.execution.order_executor._load_mt5", return_value=mt5):
            ex.open_position("EURUSD", "buy", 0.1, 1.0950, 1.1100)
        request = mt5.order_send.call_args[0][0]
        assert request["type"] == mt5.ORDER_TYPE_BUY

    def test_live_open_sell_uses_correct_type(self, tmp_path):
        """Sell-Order nutzt ORDER_TYPE_SELL."""
        ex, mt5 = self._live_executor(tmp_path)
        with patch("src.execution.order_executor._load_mt5", return_value=mt5):
            ex.open_position("EURUSD", "sell", 0.1, 1.1050, 1.0950)
        request = mt5.order_send.call_args[0][0]
        assert request["type"] == mt5.ORDER_TYPE_SELL

    def test_live_open_rejected_raises_order_error(self, tmp_path):
        """MT5 lehnt Order ab -> OrderError wird geworfen."""
        ex, mt5 = self._live_executor(tmp_path)
        fail_result = MagicMock()
        fail_result.retcode = 10004  # REQUOTE
        fail_result.comment = "Requote"
        mt5.order_send.return_value = fail_result

        with patch("src.execution.order_executor._load_mt5", return_value=mt5):
            with pytest.raises(OrderError, match="abgelehnt"):
                ex.open_position("EURUSD", "buy", 0.1, 1.0950, 1.1100)

    def test_live_open_none_result_raises_order_error(self, tmp_path):
        """order_send gibt None zurueck -> OrderError."""
        ex, mt5 = self._live_executor(tmp_path)
        mt5.order_send.return_value = None

        with patch("src.execution.order_executor._load_mt5", return_value=mt5):
            with pytest.raises(OrderError):
                ex.open_position("EURUSD", "buy", 0.1, 1.0950, 1.1100)

    def test_live_open_returns_ticket(self, tmp_path):
        """open_position gibt Ticket aus MT5-Ergebnis zurueck."""
        ex, mt5 = self._live_executor(tmp_path)
        with patch("src.execution.order_executor._load_mt5", return_value=mt5):
            result = ex.open_position("EURUSD", "buy", 0.1, 1.0950, 1.1100)
        assert result["ticket"] == 42

    def test_live_close_calls_order_send(self, tmp_path):
        """close_position im Live-Modus ruft mt5.order_send auf."""
        ex, mt5 = self._live_executor(tmp_path)
        with patch("src.execution.order_executor._load_mt5", return_value=mt5):
            ex.close_position(42)
        mt5.order_send.assert_called_once()

    def test_live_close_not_found_raises(self, tmp_path):
        """close_position mit unbekanntem Ticket -> OrderError."""
        ex, mt5 = self._live_executor(tmp_path)
        mt5.positions_get.return_value = []  # keine Position gefunden

        with patch("src.execution.order_executor._load_mt5", return_value=mt5):
            with pytest.raises(OrderError, match="nicht gefunden"):
                ex.close_position(999)

    def test_live_get_open_positions(self, tmp_path):
        """get_open_positions im Live-Modus gibt MT5-Positionen zurueck."""
        ex, mt5 = self._live_executor(tmp_path)
        with patch("src.execution.order_executor._load_mt5", return_value=mt5):
            positions = ex.get_open_positions()
        assert len(positions) == 1
        assert positions[0]["ticket"] == 42
        assert positions[0]["symbol"] == "EURUSD"
        assert positions[0]["direction"] == "buy"

    def test_live_get_open_positions_empty(self, tmp_path):
        """get_open_positions gibt leere Liste zurueck wenn keine Positionen."""
        ex, mt5 = self._live_executor(tmp_path)
        mt5.positions_get.return_value = []

        with patch("src.execution.order_executor._load_mt5", return_value=mt5):
            positions = ex.get_open_positions()
        assert positions == []

    def test_live_disconnected_raises_on_open(self, tmp_path):
        """Kein verbundener Connector -> OrderError bei open_position."""
        mt5 = _mt5_mock()
        with patch.dict(os.environ, {"CONFIRM_LIVE": "yes"}):
            ex = OrderExecutor(
                connector=_connector(connected=False),
                live_trading_enabled=True,
                paper_trades_path=tmp_path / "pt.json",
            )
        with patch("src.execution.order_executor._load_mt5", return_value=mt5):
            with pytest.raises(OrderError, match="verbunden"):
                ex.open_position("EURUSD", "buy", 0.1, 1.09, 1.11)


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: Trailing-Stop Live-Modus
# ─────────────────────────────────────────────────────────────────────────────

class TestTrailingStopLive:

    def test_live_trailing_stop_calls_order_send_with_sltp(self, tmp_path):
        """Trailing-Stop im Live-Modus nutzt TRADE_ACTION_SLTP."""
        mt5 = _mt5_mock()
        # Position: BUY, SL=1.0900, Kurs steigt auf 1.0960
        # Kandidat = 1.0960 - 0.0010 = 1.0950 >= 1.0900 + 0.0005 -> Update
        pos = mt5.positions_get.return_value[0]
        pos.type = 0    # BUY
        pos.sl   = 1.0900
        pos.tp   = 1.1100

        with patch.dict(os.environ, {"CONFIRM_LIVE": "yes"}):
            ex = OrderExecutor(
                connector=_connector(),
                live_trading_enabled=True,
                paper_trades_path=tmp_path / "pt.json",
                trailing_stop_min_pips=10.0,
                trailing_stop_step_pips=5.0,
                pip_size=0.0001,
            )
        with patch("src.execution.order_executor._load_mt5", return_value=mt5):
            ex.update_trailing_stop(42, current_price=1.0960)

        request = mt5.order_send.call_args[0][0]
        assert request["action"] == mt5.TRADE_ACTION_SLTP
        assert request["position"] == 42

    def test_live_trailing_stop_no_call_when_no_update(self, tmp_path):
        """Kein order_send wenn Trailing-Stop nicht benoetigt."""
        mt5 = _mt5_mock()
        pos = mt5.positions_get.return_value[0]
        pos.type = 0        # BUY
        pos.sl   = 1.0950   # SL hoch genug, Kurs bewegt sich kaum
        pos.tp   = 1.1100

        with patch.dict(os.environ, {"CONFIRM_LIVE": "yes"}):
            ex = OrderExecutor(
                connector=_connector(),
                live_trading_enabled=True,
                paper_trades_path=tmp_path / "pt.json",
            )
        # Kurs=1.0960 -> Kandidat=1.0950, Bedingung: 1.0950 >= 1.0950+0.0005 -> False
        with patch("src.execution.order_executor._load_mt5", return_value=mt5):
            ex.update_trailing_stop(42, current_price=1.0960)

        mt5.order_send.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
#  check_paper_sl_tp
# ─────────────────────────────────────────────────────────────────────────────

def _executor_with_open_position(
    tmp_path: Path,
    *,
    direction: str,
    open_price: float,
    sl_price: float,
    tp_price: float,
    lot_size: float = 0.1,
    symbol: str = "EURUSD",
    contract_size: float = 100_000.0,
) -> tuple[OrderExecutor, MagicMock]:
    conn = MagicMock()
    type(conn).is_connected = PropertyMock(return_value=True)
    conn.get_symbol_info.return_value = {"contract_size": contract_size}
    ex = OrderExecutor(connector=conn, paper_trades_path=tmp_path / "pt.json")
    ex.open_position(symbol, direction, lot_size, sl_price, tp_price, open_price=open_price)
    return ex, conn


class TestCheckPaperSLTP:
    def test_no_hit_returns_empty(self, tmp_path: Path):
        ex, conn = _executor_with_open_position(
            tmp_path, direction="buy", open_price=1.1000,
            sl_price=1.0900, tp_price=1.1100,
        )
        conn.get_tick.return_value = {"bid": 1.1050, "ask": 1.1052}
        result = ex.check_paper_sl_tp()
        assert result == []
        assert ex.get_open_positions()

    def test_sl_hit_buy_closes_position(self, tmp_path: Path):
        ex, conn = _executor_with_open_position(
            tmp_path, direction="buy", open_price=1.1000,
            sl_price=1.0900, tp_price=1.1100,
        )
        conn.get_tick.return_value = {"bid": 1.0898, "ask": 1.0900}
        result = ex.check_paper_sl_tp()
        assert len(result) == 1
        assert result[0]["status"] == "closed"
        assert not ex.get_open_positions()

    def test_tp_hit_buy_closes_position(self, tmp_path: Path):
        ex, conn = _executor_with_open_position(
            tmp_path, direction="buy", open_price=1.1000,
            sl_price=1.0900, tp_price=1.1100,
        )
        conn.get_tick.return_value = {"bid": 1.1102, "ask": 1.1104}
        result = ex.check_paper_sl_tp()
        assert len(result) == 1
        assert result[0]["status"] == "closed"

    def test_sl_hit_sell_closes_position(self, tmp_path: Path):
        ex, conn = _executor_with_open_position(
            tmp_path, direction="sell", open_price=1.1000,
            sl_price=1.1100, tp_price=1.0900,
        )
        conn.get_tick.return_value = {"bid": 1.1098, "ask": 1.1102}
        result = ex.check_paper_sl_tp()
        assert len(result) == 1
        assert result[0]["status"] == "closed"

    def test_tp_hit_sell_closes_position(self, tmp_path: Path):
        ex, conn = _executor_with_open_position(
            tmp_path, direction="sell", open_price=1.1000,
            sl_price=1.1100, tp_price=1.0900,
        )
        conn.get_tick.return_value = {"bid": 1.0898, "ask": 1.0900}
        result = ex.check_paper_sl_tp()
        assert len(result) == 1
        assert result[0]["status"] == "closed"

    def test_pnl_positive_on_tp_buy(self, tmp_path: Path):
        ex, conn = _executor_with_open_position(
            tmp_path, direction="buy", open_price=1.1000,
            sl_price=1.0900, tp_price=1.1100,
            lot_size=0.1, contract_size=100_000.0,
        )
        conn.get_tick.return_value = {"bid": 1.1100, "ask": 1.1102}
        result = ex.check_paper_sl_tp()
        assert result[0]["pnl"] == pytest.approx(0.1 * 100_000.0 * (1.1100 - 1.1000), abs=0.01)

    def test_pnl_negative_on_sl_buy(self, tmp_path: Path):
        ex, conn = _executor_with_open_position(
            tmp_path, direction="buy", open_price=1.1000,
            sl_price=1.0900, tp_price=1.1100,
            lot_size=0.1, contract_size=100_000.0,
        )
        conn.get_tick.return_value = {"bid": 1.0900, "ask": 1.0902}
        result = ex.check_paper_sl_tp()
        assert result[0]["pnl"] == pytest.approx(0.1 * 100_000.0 * (1.0900 - 1.1000), abs=0.01)

    def test_live_mode_returns_empty(self, tmp_path: Path):
        conn = MagicMock()
        type(conn).is_connected = PropertyMock(return_value=True)
        with patch.dict(os.environ, {"CONFIRM_LIVE": "yes"}):
            ex = OrderExecutor(
                connector=conn, live_trading_enabled=True,
                paper_trades_path=tmp_path / "pt.json",
            )
        result = ex.check_paper_sl_tp()
        assert result == []
        conn.get_tick.assert_not_called()

    def test_tick_error_skips_position(self, tmp_path: Path):
        ex, conn = _executor_with_open_position(
            tmp_path, direction="buy", open_price=1.1000,
            sl_price=1.0900, tp_price=1.1100,
        )
        conn.get_tick.side_effect = RuntimeError("no connection")
        result = ex.check_paper_sl_tp()
        assert result == []
        assert ex.get_open_positions()

    def test_close_fires_on_close_callback(self, tmp_path: Path):
        ex, conn = _executor_with_open_position(
            tmp_path, direction="buy", open_price=1.1000,
            sl_price=1.0900, tp_price=1.1100,
        )
        cb = MagicMock()
        ex.set_order_callbacks(on_open=None, on_close=cb)
        conn.get_tick.return_value = {"bid": 1.1102, "ask": 1.1104}
        ex.check_paper_sl_tp()
        cb.assert_called_once()


class TestLoadPaperTrades:

    def test_loads_open_positions_on_init(self, tmp_path: Path):
        path = tmp_path / "t.json"
        path.write_text(json.dumps([{
            "ticket": 5, "symbol": "EURUSD", "direction": "buy",
            "lot_size": 0.1, "sl_price": 1.07, "tp_price": 1.10,
            "open_price": 1.08, "open_time": "2026-06-23T10:00:00+00:00",
            "close_price": None, "close_time": None, "status": "open",
        }]), encoding="utf-8")
        conn = MagicMock()
        ex = OrderExecutor(connector=conn, paper_trades_path=path)
        assert 5 in ex._paper_positions
        assert ex._paper_positions[5]["status"] == "open"

    def test_next_ticket_above_loaded(self, tmp_path: Path):
        path = tmp_path / "t.json"
        path.write_text(json.dumps([
            {"ticket": 10, "status": "open", "symbol": "EURUSD", "direction": "buy",
             "lot_size": 0.1, "open_price": 1.08, "sl_price": 1.07, "tp_price": 1.10,
             "open_time": None, "close_price": None, "close_time": None},
            {"ticket": 11, "status": "closed", "symbol": "EURUSD", "direction": "buy",
             "lot_size": 0.1, "open_price": 1.08, "sl_price": 1.07, "tp_price": 1.10,
             "open_time": None, "close_price": 1.09, "close_time": None, "pnl": 100.0},
        ]), encoding="utf-8")
        conn = MagicMock()
        ex = OrderExecutor(connector=conn, paper_trades_path=path)
        assert ex._next_ticket == 12

    def test_no_file_starts_empty(self, tmp_path: Path):
        conn = MagicMock()
        ex = OrderExecutor(connector=conn, paper_trades_path=tmp_path / "missing.json")
        assert ex._paper_positions == {}
        assert ex._next_ticket == 1

    def test_check_sl_tp_monitors_loaded_positions(self, tmp_path: Path):
        path = tmp_path / "t.json"
        path.write_text(json.dumps([{
            "ticket": 3, "symbol": "EURUSD", "direction": "buy",
            "lot_size": 0.1, "open_price": 1.0800,
            "sl_price": 1.0700, "tp_price": 1.0900,
            "open_time": "2026-06-23T10:00:00+00:00",
            "close_price": None, "close_time": None, "status": "open",
        }]), encoding="utf-8")
        conn = MagicMock()
        conn.get_tick.return_value = {"bid": 1.0905, "ask": 1.0907}
        conn.get_symbol_info.return_value = {"contract_size": 100_000.0}
        ex = OrderExecutor(connector=conn, paper_trades_path=path)
        closed = ex.check_paper_sl_tp()
        assert len(closed) == 1
        assert closed[0]["ticket"] == 3


class TestMarkProfitLock70:

    def test_sets_flag_on_open_position(self, tmp_path: Path):
        conn = MagicMock()
        ex = OrderExecutor(connector=conn, paper_trades_path=tmp_path / "t.json")
        result = ex.open_position("EURUSD", "buy", 0.1, 1.07, 1.10)
        ticket = result["ticket"]
        ex.mark_profit_lock_70(ticket)
        pos = ex._paper_positions[ticket]
        assert pos["profit_lock_70_triggered"] is True

    def test_persisted_to_json(self, tmp_path: Path):
        import json
        conn = MagicMock()
        path = tmp_path / "t.json"
        ex = OrderExecutor(connector=conn, paper_trades_path=path)
        result = ex.open_position("EURUSD", "buy", 0.1, 1.07, 1.10)
        ex.mark_profit_lock_70(result["ticket"])
        data = json.loads(path.read_text())
        assert data[0]["profit_lock_70_triggered"] is True

    def test_noop_on_closed_position(self, tmp_path: Path):
        conn = MagicMock()
        conn.get_tick.return_value = {"bid": 1.10, "ask": 1.1002}
        conn.get_symbol_info.return_value = {"contract_size": 100_000.0}
        ex = OrderExecutor(connector=conn, paper_trades_path=tmp_path / "t.json")
        result = ex.open_position("EURUSD", "buy", 0.1, 1.07, 1.12)
        ticket = result["ticket"]
        ex.close_position(ticket, close_price=1.10)
        ex.mark_profit_lock_70(ticket)
        pos = ex._paper_positions[ticket]
        assert not pos.get("profit_lock_70_triggered")

    def test_noop_in_live_mode(self, tmp_path: Path):
        import os
        from unittest.mock import patch, PropertyMock
        conn = MagicMock()
        type(conn).is_connected = PropertyMock(return_value=True)
        with patch.dict(os.environ, {"CONFIRM_LIVE": "yes"}):
            ex = OrderExecutor(
                connector=conn,
                live_trading_enabled=True,
                paper_trades_path=tmp_path / "t.json",
            )
        ex.mark_profit_lock_70(99)  # kein Crash, kein Effekt
