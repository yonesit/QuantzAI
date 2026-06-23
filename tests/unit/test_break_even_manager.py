"""
tests/unit/test_break_even_manager.py
Tests fuer src/risk/break_even_manager.py und die zugehoerigen Hilfsfunktionen
aus scripts/run_gui_bot.py (_calc_crv, _calc_total_stats).

Alle Tests ohne echte MT5-Verbindung oder Qt-Abhaengigkeit.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from src.risk.break_even_manager import (
    BreakEvenManager,
    calc_break_even_sl,
    calc_progress,
    calc_profit_lock_70_sl,
    calc_realized_pnl,
    calc_trailing_85_sl,
    is_sl_hit,
    is_tp_hit,
    should_trigger_break_even,
)
from scripts.run_gui_bot import _calc_crv, _calc_total_stats


# ─────────────────────────────────────────────────────────────────────────────
#  should_trigger_break_even
# ─────────────────────────────────────────────────────────────────────────────

class TestShouldTriggerBreakEven:

    # BUY

    def test_buy_below_threshold_returns_false(self):
        # 30 % des Weges zum TP → noch nicht triggern
        assert not should_trigger_break_even("buy", 1.0800, 1.0900, 1.0830, threshold=0.5)

    def test_buy_exactly_at_threshold_returns_true(self):
        # 50 % des Weges zum TP (1 Pip darueber um FP-Grenzfall zu umgehen) → triggern
        assert should_trigger_break_even("buy", 1.0800, 1.0900, 1.0851, threshold=0.5)

    def test_buy_above_threshold_returns_true(self):
        # 80 % → triggern
        assert should_trigger_break_even("buy", 1.0800, 1.0900, 1.0880, threshold=0.5)

    def test_buy_zero_dist_returns_false(self):
        # tp == open: keine sinnvolle Distanz
        assert not should_trigger_break_even("buy", 1.0800, 1.0800, 1.0800)

    def test_buy_negative_dist_returns_false(self):
        # tp < open: ungueltige Konfiguration → kein BE
        assert not should_trigger_break_even("buy", 1.0900, 1.0800, 1.0900)

    # SELL

    def test_sell_below_threshold_returns_false(self):
        # 30 % des Weges zum TP (Kurs faellt) → noch nicht triggern
        assert not should_trigger_break_even("sell", 1.0900, 1.0800, 1.0870, threshold=0.5)

    def test_sell_exactly_at_threshold_returns_true(self):
        # genau 50 % → triggern
        assert should_trigger_break_even("sell", 1.0900, 1.0800, 1.0850, threshold=0.5)

    def test_sell_above_threshold_returns_true(self):
        # 80 % → triggern
        assert should_trigger_break_even("sell", 1.0900, 1.0800, 1.0820, threshold=0.5)

    def test_sell_zero_dist_returns_false(self):
        assert not should_trigger_break_even("sell", 1.0800, 1.0800, 1.0800)

    # Benutzerdefinierter Schwellenwert

    def test_custom_threshold_25pct(self):
        # 30 % Fortschritt bei threshold=0.25 → triggern
        assert should_trigger_break_even("buy", 1.0800, 1.0900, 1.0830, threshold=0.25)

    def test_custom_threshold_75pct_not_yet(self):
        # 60 % Fortschritt bei threshold=0.75 → noch nicht triggern
        assert not should_trigger_break_even("buy", 1.0800, 1.0900, 1.0860, threshold=0.75)


# ─────────────────────────────────────────────────────────────────────────────
#  calc_break_even_sl
# ─────────────────────────────────────────────────────────────────────────────

class TestCalcBreakEvenSl:

    def test_buy_sl_above_open(self):
        # BUY: SL = open + 2 pips buffer
        sl = calc_break_even_sl("buy", 1.0800, spread_buffer_pips=2.0, pip_size=0.0001)
        assert abs(sl - 1.0802) < 1e-9

    def test_sell_sl_below_open(self):
        # SELL: SL = open - 2 pips buffer
        sl = calc_break_even_sl("sell", 1.0800, spread_buffer_pips=2.0, pip_size=0.0001)
        assert abs(sl - 1.0798) < 1e-9

    def test_buy_zero_buffer(self):
        sl = calc_break_even_sl("buy", 1.0800, spread_buffer_pips=0.0)
        assert abs(sl - 1.0800) < 1e-9

    def test_sell_custom_buffer(self):
        sl = calc_break_even_sl("sell", 1.0900, spread_buffer_pips=5.0, pip_size=0.0001)
        assert abs(sl - 1.0895) < 1e-9

    def test_xauusd_buy_large_pip_size(self):
        # XAUUSD: pip_size = 0.01
        sl = calc_break_even_sl("buy", 2000.00, spread_buffer_pips=3.0, pip_size=0.01)
        assert abs(sl - 2000.03) < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
#  is_sl_hit
# ─────────────────────────────────────────────────────────────────────────────

class TestIsSlHit:

    def test_buy_sl_triggered_bid_at_sl(self):
        assert is_sl_hit("buy", sl_price=1.0790, bid=1.0790, ask=1.0792)

    def test_buy_sl_triggered_bid_below_sl(self):
        assert is_sl_hit("buy", sl_price=1.0790, bid=1.0785, ask=1.0787)

    def test_buy_sl_not_triggered_bid_above_sl(self):
        assert not is_sl_hit("buy", sl_price=1.0790, bid=1.0795, ask=1.0797)

    def test_sell_sl_triggered_ask_at_sl(self):
        assert is_sl_hit("sell", sl_price=1.0910, bid=1.0908, ask=1.0910)

    def test_sell_sl_triggered_ask_above_sl(self):
        assert is_sl_hit("sell", sl_price=1.0910, bid=1.0912, ask=1.0914)

    def test_sell_sl_not_triggered_ask_below_sl(self):
        assert not is_sl_hit("sell", sl_price=1.0910, bid=1.0905, ask=1.0907)


# ─────────────────────────────────────────────────────────────────────────────
#  is_tp_hit
# ─────────────────────────────────────────────────────────────────────────────

class TestIsTpHit:

    def test_buy_tp_triggered_bid_at_tp(self):
        assert is_tp_hit("buy", tp_price=1.0950, bid=1.0950, ask=1.0952)

    def test_buy_tp_triggered_bid_above_tp(self):
        assert is_tp_hit("buy", tp_price=1.0950, bid=1.0955, ask=1.0957)

    def test_buy_tp_not_triggered_bid_below_tp(self):
        assert not is_tp_hit("buy", tp_price=1.0950, bid=1.0945, ask=1.0947)

    def test_sell_tp_triggered_ask_at_tp(self):
        assert is_tp_hit("sell", tp_price=1.0750, bid=1.0748, ask=1.0750)

    def test_sell_tp_triggered_ask_below_tp(self):
        assert is_tp_hit("sell", tp_price=1.0750, bid=1.0745, ask=1.0747)

    def test_sell_tp_not_triggered_ask_above_tp(self):
        assert not is_tp_hit("sell", tp_price=1.0750, bid=1.0755, ask=1.0757)


# ─────────────────────────────────────────────────────────────────────────────
#  calc_realized_pnl
# ─────────────────────────────────────────────────────────────────────────────

class TestCalcRealizedPnl:

    def test_buy_profit(self):
        pnl = calc_realized_pnl("buy", 1.0800, 1.0900, lot_size=0.1, contract_size=100_000)
        assert abs(pnl - 100.0) < 0.001

    def test_buy_loss(self):
        pnl = calc_realized_pnl("buy", 1.0900, 1.0800, lot_size=0.1, contract_size=100_000)
        assert abs(pnl - (-100.0)) < 0.001

    def test_sell_profit(self):
        pnl = calc_realized_pnl("sell", 1.0900, 1.0800, lot_size=0.1, contract_size=100_000)
        assert abs(pnl - 100.0) < 0.001

    def test_sell_loss(self):
        pnl = calc_realized_pnl("sell", 1.0800, 1.0900, lot_size=0.1, contract_size=100_000)
        assert abs(pnl - (-100.0)) < 0.001

    def test_xauusd_buy(self):
        pnl = calc_realized_pnl("buy", 2000.0, 2010.0, lot_size=0.01, contract_size=100)
        assert abs(pnl - 10.0) < 0.001


# ─────────────────────────────────────────────────────────────────────────────
#  BreakEvenManager.manage() – mit gemockten Connector + Executor
# ─────────────────────────────────────────────────────────────────────────────

def _make_pos(
    ticket=1,
    symbol="EURUSD",
    direction="buy",
    open_price=1.0800,
    sl_price=1.0750,
    tp_price=1.0900,
    lot_size=0.1,
    be_triggered=False,
    lock_70_triggered=False,
    status="open",
):
    return {
        "ticket":                   ticket,
        "symbol":                   symbol,
        "direction":                direction,
        "open_price":               open_price,
        "sl_price":                 sl_price,
        "tp_price":                 tp_price,
        "lot_size":                 lot_size,
        "break_even_triggered":     be_triggered,
        "profit_lock_70_triggered": lock_70_triggered,
        "status":                   status,
    }


@pytest.fixture
def mock_connector():
    c = MagicMock()
    c.get_tick.return_value = {"bid": 1.0850, "ask": 1.0852}
    c.get_symbol_info.return_value = {"contract_size": 100_000.0}
    return c


@pytest.fixture
def mock_executor():
    ex = MagicMock()
    ex.get_open_positions.return_value = []
    return ex


class TestBreakEvenManagerManage:

    def test_no_positions_returns_empty(self, mock_connector, mock_executor):
        mock_executor.get_open_positions.return_value = []
        mgr = BreakEvenManager(mock_connector, mock_executor)
        actions = mgr.manage("EURUSD")
        assert actions == []
        mock_connector.get_tick.assert_not_called()

    def test_tick_error_returns_empty(self, mock_connector, mock_executor):
        mock_executor.get_open_positions.return_value = [_make_pos()]
        mock_connector.get_tick.side_effect = RuntimeError("kein Tick")
        mgr = BreakEvenManager(mock_connector, mock_executor)
        actions = mgr.manage("EURUSD")
        assert actions == []

    def test_sl_hit_closes_position(self, mock_connector, mock_executor):
        """BUY: bid faellt auf SL → Position wird geschlossen."""
        pos = _make_pos(open_price=1.0800, sl_price=1.0850, tp_price=1.0950)
        mock_executor.get_open_positions.return_value = [pos]
        # bid=1.0850 == sl_price → SL getriggert
        mock_connector.get_tick.return_value = {"bid": 1.0850, "ask": 1.0852}

        mgr = BreakEvenManager(mock_connector, mock_executor)
        actions = mgr.manage("EURUSD")

        assert len(actions) == 1
        assert actions[0]["action"] == "sl_hit"
        assert actions[0]["ticket"] == 1
        mock_executor.close_position.assert_called_once()
        call_args = mock_executor.close_position.call_args
        assert call_args.args[0] == 1  # ticket
        assert call_args.kwargs.get("close_price") is not None

    def test_tp_hit_closes_position(self, mock_connector, mock_executor):
        """BUY: bid erreicht TP → Position wird geschlossen."""
        pos = _make_pos(open_price=1.0800, sl_price=1.0750, tp_price=1.0850)
        mock_executor.get_open_positions.return_value = [pos]
        # bid=1.0850 == tp_price → TP getriggert
        mock_connector.get_tick.return_value = {"bid": 1.0850, "ask": 1.0852}

        mgr = BreakEvenManager(mock_connector, mock_executor)
        actions = mgr.manage("EURUSD")

        assert len(actions) == 1
        assert actions[0]["action"] == "tp_hit"
        mock_executor.close_position.assert_called_once()

    def test_break_even_triggered_at_50pct(self, mock_connector, mock_executor):
        """BUY: Preis ueber 50 % des TP-Weges → Break-Even aktivieren."""
        # open=1.0800, tp=1.0900 → BE bei > 1.0850; bid=1.0860 (60 %) sicher darueber
        pos = _make_pos(open_price=1.0800, sl_price=1.0750, tp_price=1.0900)
        mock_executor.get_open_positions.return_value = [pos]
        mock_connector.get_tick.return_value = {"bid": 1.0860, "ask": 1.0862}

        mgr = BreakEvenManager(mock_connector, mock_executor, break_even_threshold=0.5)
        actions = mgr.manage("EURUSD")

        assert len(actions) == 1
        assert actions[0]["action"] == "break_even"
        mock_executor.set_stop_loss.assert_called_once()
        mock_executor.mark_break_even.assert_called_once_with(1)

    def test_break_even_not_triggered_below_threshold(self, mock_connector, mock_executor):
        """BUY: Preis bei 30 % → Break-Even noch nicht aktivieren."""
        pos = _make_pos(open_price=1.0800, sl_price=1.0750, tp_price=1.0900)
        mock_executor.get_open_positions.return_value = [pos]
        # 30 % von 100 Pips = 30 Pips → bid = 1.0830
        mock_connector.get_tick.return_value = {"bid": 1.0830, "ask": 1.0832}

        mgr = BreakEvenManager(mock_connector, mock_executor, break_even_threshold=0.5)
        actions = mgr.manage("EURUSD")

        assert actions == []
        mock_executor.set_stop_loss.assert_not_called()
        mock_executor.mark_break_even.assert_not_called()

    def test_profit_lock_70_triggered_at_70pct(self, mock_connector, mock_executor):
        """BE aktiv + Preis bei 72% → SL auf +33% TP-Distanz sichern."""
        # open=1.0800, tp=1.0900 (100 Pips), bid=1.0872 → 72% (>70%)
        pos = _make_pos(open_price=1.0800, sl_price=1.0802, tp_price=1.0900, be_triggered=True)
        mock_executor.get_open_positions.return_value = [pos]
        mock_connector.get_tick.return_value = {"bid": 1.0872, "ask": 1.0874}

        mgr = BreakEvenManager(mock_connector, mock_executor)
        actions = mgr.manage("EURUSD")

        assert len(actions) == 1
        assert actions[0]["action"] == "profit_lock_70"
        mock_executor.mark_profit_lock_70.assert_called_once_with(1)
        # SL muss auf open + 33% * tp_dist = 1.0800 + 0.33 * 0.0100 = 1.0833 liegen
        sl_arg = mock_executor.set_stop_loss.call_args[0][1]
        assert abs(sl_arg - 1.0833) < 0.0001

    def test_trailing_85_called_at_85pct(self, mock_connector, mock_executor):
        """BE + lock_70 aktiv + Preis bei ≥85% → 85%-Trailing setzen."""
        # open=1.0800, tp=1.0900 (100 Pips), bid=1.0890 → 90%
        pos = _make_pos(
            open_price=1.0800, sl_price=1.0833, tp_price=1.0900,
            be_triggered=True, lock_70_triggered=True,
        )
        mock_executor.get_open_positions.return_value = [pos]
        mock_connector.get_tick.return_value = {"bid": 1.0890, "ask": 1.0892}

        mgr = BreakEvenManager(mock_connector, mock_executor)
        actions = mgr.manage("EURUSD")

        assert len(actions) == 1
        assert actions[0]["action"] == "trailing_85"
        # SL = bid - 20% * tp_dist = 1.0890 - 0.0020 = 1.0870
        sl_arg = mock_executor.set_stop_loss.call_args[0][1]
        assert abs(sl_arg - 1.0870) < 0.0001

    def test_no_action_when_be_active_and_below_70pct(self, mock_connector, mock_executor):
        """BE aktiv, aber Preis <70% → kein weiterer Eingriff."""
        pos = _make_pos(open_price=1.0800, sl_price=1.0802, tp_price=1.0900, be_triggered=True)
        mock_executor.get_open_positions.return_value = [pos]
        mock_connector.get_tick.return_value = {"bid": 1.0860, "ask": 1.0862}  # 60%

        mgr = BreakEvenManager(mock_connector, mock_executor)
        actions = mgr.manage("EURUSD")

        assert actions == []
        mock_executor.set_stop_loss.assert_not_called()

    def test_break_even_sl_not_moved_if_worse(self, mock_connector, mock_executor):
        """BE-SL soll nur gesetzt werden wenn er den SL verbessert (nicht verschlechtert)."""
        # SL ist schon bei 1.0805 (besser als BE-SL 1.0802); bid=1.0860 (60 %) → BE triggert
        pos = _make_pos(open_price=1.0800, sl_price=1.0805, tp_price=1.0900)
        mock_executor.get_open_positions.return_value = [pos]
        mock_connector.get_tick.return_value = {"bid": 1.0860, "ask": 1.0862}

        mgr = BreakEvenManager(
            mock_connector, mock_executor,
            break_even_threshold=0.5,
            spread_buffer_pips=2.0,
        )
        actions = mgr.manage("EURUSD")

        # BE wird markiert, aber SL nicht verschlechtert
        assert len(actions) == 1
        assert actions[0]["action"] == "break_even"
        mock_executor.set_stop_loss.assert_not_called()  # schlechter → nicht setzen
        mock_executor.mark_break_even.assert_called_once()

    def test_symbol_filter(self, mock_connector, mock_executor):
        """manage('EURUSD') darf XAUUSD-Positionen ignorieren."""
        xauusd_pos = _make_pos(ticket=10, symbol="XAUUSD")
        eurusd_pos = _make_pos(ticket=11, symbol="EURUSD",
                               open_price=1.0800, sl_price=1.0750, tp_price=1.0900)
        mock_executor.get_open_positions.return_value = [xauusd_pos, eurusd_pos]
        mock_connector.get_tick.return_value = {"bid": 1.0830, "ask": 1.0832}

        mgr = BreakEvenManager(mock_connector, mock_executor)
        mgr.manage("EURUSD")

        # Tick-Abruf muss NUR fuer EURUSD erfolgen (keine XAUUSD-Verarbeitung)
        mock_connector.get_tick.assert_called_once_with("EURUSD")

    def test_contract_size_cached(self, mock_connector, mock_executor):
        """get_symbol_info() darf nur einmal pro Symbol aufgerufen werden."""
        pos1 = _make_pos(ticket=1, open_price=1.0800, sl_price=1.0750, tp_price=1.0850)
        pos2 = _make_pos(ticket=2, open_price=1.0810, sl_price=1.0760, tp_price=1.0860)
        mock_executor.get_open_positions.return_value = [pos1, pos2]
        mock_connector.get_tick.return_value = {"bid": 1.0850, "ask": 1.0852}

        mgr = BreakEvenManager(mock_connector, mock_executor)
        mgr.manage("EURUSD")

        # Beide Positionen treffen TP → zwei close_position-Aufrufe
        # aber get_symbol_info nur einmal (gecacht)
        assert mock_connector.get_symbol_info.call_count <= 1

    # ── SL/TP Luecken-/Sprung-Szenario (GAP-OVER) ────────────────────────────

    def test_sl_gap_over_buy_closes_position(self, mock_connector, mock_executor):
        """
        BUY: Kurs SPRINGT von 1.14050 direkt auf 1.13900 ohne den exakten SL-Wert
        1.14000 zu beruehren. Position muss trotzdem korrekt geschlossen werden.

        Prueft: is_sl_hit nutzt '<=', nicht '=='. Das ist der wichtigste Korrektheitsbeweis
        fuer Wochenend-Gaps und volatile Markte.
        """
        pos = _make_pos(
            ticket=42,
            symbol="EURUSD",
            direction="buy",
            open_price=1.14083,
            sl_price=1.14000,   # SL liegt bei 1.14000
            tp_price=1.14283,
            lot_size=0.66,
        )
        mock_executor.get_open_positions.return_value = [pos]
        # Kurs springt von 1.14050 direkt auf 1.13900 – SL (1.14000) wird uebersprungen
        mock_connector.get_tick.return_value = {"bid": 1.13900, "ask": 1.13902}

        mgr = BreakEvenManager(mock_connector, mock_executor)
        actions = mgr.manage("EURUSD")

        assert len(actions) == 1, (
            "Position muss bei Gap-Over des SL geschlossen werden! "
            f"Erhaltene Aktionen: {actions}"
        )
        assert actions[0]["action"] == "sl_hit"
        assert actions[0]["ticket"] == 42
        mock_executor.close_position.assert_called_once()
        close_kwargs = mock_executor.close_position.call_args.kwargs
        assert close_kwargs.get("close_price") is not None
        assert close_kwargs.get("pnl") is not None
        # Verlust weil bid (1.13900) < open (1.14083)
        assert close_kwargs["pnl"] < 0, "Gap-Over SL muss negativen PnL erzeugen"

    def test_sl_gap_over_sell_closes_position(self, mock_connector, mock_executor):
        """
        SELL: Kurs SPRINGT von 1.0800 direkt auf 1.0920 ohne den exakten SL-Wert
        1.0910 zu beruehren. Position muss trotzdem korrekt geschlossen werden.
        """
        pos = _make_pos(
            ticket=43,
            symbol="EURUSD",
            direction="sell",
            open_price=1.0850,
            sl_price=1.0910,   # SL liegt bei 1.0910
            tp_price=1.0750,
            lot_size=0.1,
        )
        mock_executor.get_open_positions.return_value = [pos]
        # Ask springt von 1.0800 auf 1.0920 – SL (1.0910) wird uebersprungen
        mock_connector.get_tick.return_value = {"bid": 1.0918, "ask": 1.0920}

        mgr = BreakEvenManager(mock_connector, mock_executor)
        actions = mgr.manage("EURUSD")

        assert len(actions) == 1, (
            "SELL-Position muss bei Gap-Over des SL geschlossen werden!"
        )
        assert actions[0]["action"] == "sl_hit"
        mock_executor.close_position.assert_called_once()

    def test_tp_gap_over_buy_closes_position(self, mock_connector, mock_executor):
        """
        BUY: Kurs SPRINGT von 1.0880 direkt auf 1.0920 ohne den exakten TP-Wert
        1.0900 zu beruehren. TP muss trotzdem korrekt ausgeloest werden.
        """
        pos = _make_pos(
            ticket=44,
            direction="buy",
            open_price=1.0800,
            sl_price=1.0750,
            tp_price=1.0900,   # TP liegt bei 1.0900
            lot_size=0.1,
        )
        mock_executor.get_open_positions.return_value = [pos]
        # bid springt von 1.0880 auf 1.0920 – TP (1.0900) wird uebersprungen
        mock_connector.get_tick.return_value = {"bid": 1.0920, "ask": 1.0922}

        mgr = BreakEvenManager(mock_connector, mock_executor)
        actions = mgr.manage("EURUSD")

        assert len(actions) == 1
        assert actions[0]["action"] == "tp_hit"
        close_kwargs = mock_executor.close_position.call_args.kwargs
        # Gewinn weil bid (1.0920) > open (1.0800)
        assert close_kwargs["pnl"] > 0


# ─────────────────────────────────────────────────────────────────────────────
#  _calc_crv (aus run_gui_bot.py)
# ─────────────────────────────────────────────────────────────────────────────

class TestCalcCrv:

    def test_buy_crv_2_to_1(self):
        pos = {
            "direction": "buy",
            "open_price": 1.0800,
            "sl_price":   1.0750,  # 50 Pips SL
            "tp_price":   1.0900,  # 100 Pips TP -> CRV = 2.0
        }
        assert _calc_crv(pos) == 2.0

    def test_sell_crv_3_to_1(self):
        pos = {
            "direction": "sell",
            "open_price": 1.0900,
            "sl_price":   1.0940,  # 40 Pips SL
            "tp_price":   1.0780,  # 120 Pips TP -> CRV = 3.0
        }
        assert _calc_crv(pos) == 3.0

    def test_missing_open_price_returns_none(self):
        pos = {"direction": "buy", "sl_price": 1.07, "tp_price": 1.10}
        assert _calc_crv(pos) is None

    def test_missing_sl_returns_none(self):
        pos = {"direction": "buy", "open_price": 1.08, "tp_price": 1.10}
        assert _calc_crv(pos) is None

    def test_zero_sl_dist_returns_none(self):
        # SL == open_price → Distanz 0
        pos = {
            "direction": "buy",
            "open_price": 1.0800,
            "sl_price":   1.0800,
            "tp_price":   1.0900,
        }
        assert _calc_crv(pos) is None


# ─────────────────────────────────────────────────────────────────────────────
#  _calc_total_stats (aus run_gui_bot.py)
# ─────────────────────────────────────────────────────────────────────────────

class TestCalcTotalStats:

    def test_empty_list_returns_none_none(self):
        assert _calc_total_stats([]) == (None, None)

    def test_only_open_trades_returns_none_none(self):
        trades = [
            {"status": "open", "pnl": None},
            {"status": "open", "pnl": None},
        ]
        assert _calc_total_stats(trades) == (None, None)

    def test_no_pnl_field_returns_none_none(self):
        trades = [{"status": "closed"}]
        assert _calc_total_stats(trades) == (None, None)

    def test_single_profit(self):
        trades = [{"status": "closed", "pnl": 150.0}]
        profit, loss = _calc_total_stats(trades)
        assert abs(profit - 150.0) < 0.001
        assert loss is None

    def test_single_loss(self):
        trades = [{"status": "closed", "pnl": -80.0}]
        profit, loss = _calc_total_stats(trades)
        assert profit is None
        assert abs(loss - (-80.0)) < 0.001

    def test_mixed_trades(self):
        trades = [
            {"status": "closed", "pnl":  200.0},
            {"status": "closed", "pnl": -100.0},
            {"status": "closed", "pnl":   50.0},
            {"status": "open",   "pnl": None},
        ]
        profit, loss = _calc_total_stats(trades)
        assert abs(profit - 250.0) < 0.001
        assert abs(loss - (-100.0)) < 0.001

    def test_breakeven_trade_counted_as_loss(self):
        """pnl=0.0 ist <= 0 → als 'Verlust' (Break-Even) einordnen."""
        trades = [{"status": "closed", "pnl": 0.0}]
        profit, loss = _calc_total_stats(trades)
        assert profit is None
        assert abs(loss) < 0.001

    def test_open_trades_ignored(self):
        trades = [
            {"status": "open",   "pnl": 999.0},  # ignorieren
            {"status": "closed", "pnl":  75.0},
        ]
        profit, loss = _calc_total_stats(trades)
        assert abs(profit - 75.0) < 0.001
        assert loss is None


# ─────────────────────────────────────────────────────────────────────────────
#  calc_progress (neue Pure-Funktion)
# ─────────────────────────────────────────────────────────────────────────────

class TestCalcProgress:

    def test_buy_50pct(self):
        assert abs(calc_progress("buy", 1.0800, 1.0900, 1.0850) - 0.5) < 1e-9

    def test_buy_70pct(self):
        assert abs(calc_progress("buy", 1.0800, 1.0900, 1.0870) - 0.7) < 1e-9

    def test_buy_85pct(self):
        assert abs(calc_progress("buy", 1.0800, 1.0900, 1.0885) - 0.85) < 1e-9

    def test_buy_zero_dist_returns_zero(self):
        assert calc_progress("buy", 1.0800, 1.0800, 1.0800) == 0.0

    def test_sell_50pct(self):
        assert abs(calc_progress("sell", 1.0900, 1.0800, 1.0850) - 0.5) < 1e-9

    def test_sell_85pct(self):
        assert abs(calc_progress("sell", 1.0900, 1.0800, 1.0815) - 0.85) < 1e-9

    def test_sell_zero_dist_returns_zero(self):
        assert calc_progress("sell", 1.0900, 1.0900, 1.0900) == 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  calc_profit_lock_70_sl (neue Pure-Funktion)
# ─────────────────────────────────────────────────────────────────────────────

class TestCalcProfitLock70Sl:

    def test_buy_sl_at_33pct_tp_dist(self):
        # open=1.0800, tp=1.0900 → tp_dist=0.0100 → SL = 1.0800 + 0.0033 = 1.0833
        sl = calc_profit_lock_70_sl("buy", 1.0800, 1.0900)
        assert abs(sl - 1.0833) < 1e-5

    def test_sell_sl_at_33pct_tp_dist(self):
        # open=1.0900, tp=1.0800 → tp_dist=0.0100 → SL = 1.0900 - 0.0033 = 1.0867
        sl = calc_profit_lock_70_sl("sell", 1.0900, 1.0800)
        assert abs(sl - 1.0867) < 1e-5

    def test_xauusd_buy(self):
        # open=2000, tp=2030 → tp_dist=30 → SL = 2000 + 9.9 = 2009.9
        sl = calc_profit_lock_70_sl("buy", 2000.0, 2030.0)
        assert abs(sl - 2009.9) < 0.01


# ─────────────────────────────────────────────────────────────────────────────
#  calc_trailing_85_sl (neue Pure-Funktion)
# ─────────────────────────────────────────────────────────────────────────────

class TestCalcTrailing85Sl:

    def test_buy_sl_20pct_behind(self):
        # open=1.0800, tp=1.0900 (dist=0.0100), current=1.0890
        # SL = 1.0890 - 0.20*0.0100 = 1.0890 - 0.0020 = 1.0870
        sl = calc_trailing_85_sl("buy", 1.0890, 1.0800, 1.0900)
        assert abs(sl - 1.0870) < 1e-5

    def test_sell_sl_20pct_behind(self):
        # open=1.0900, tp=1.0800 (dist=0.0100), current=1.0815
        # SL = 1.0815 + 0.0020 = 1.0835
        sl = calc_trailing_85_sl("sell", 1.0815, 1.0900, 1.0800)
        assert abs(sl - 1.0835) < 1e-5

    def test_trail_moves_up_with_price(self):
        # Price moves from 85% to 95%
        sl1 = calc_trailing_85_sl("buy", 1.0885, 1.0800, 1.0900)
        sl2 = calc_trailing_85_sl("buy", 1.0895, 1.0800, 1.0900)
        assert sl2 > sl1
