"""Tests fuer die reinen P&L-/Sharpe-/Sizing-Bausteine."""

from __future__ import annotations

import numpy as np
import pytest

from src.models import pnl_backtest as pnl

PIP = 0.0001


class TestSizing:

    def test_full_sl_loss_equals_risk_frac(self):
        # SL-Abstand = 1.0*k*ATR; ein Trade der -sl_distance macht = -risk_frac
        atr = np.array([0.0010])                 # 10 Pip ATR
        k, sl_mult, risk = 1.5, 1.0, 0.01
        sl_pips = pnl.sl_distance_pips(atr, k, sl_mult, PIP)[0]   # 15 Pip
        net = np.array([-sl_pips])               # exakt der SL-Verlust
        r = pnl.trade_returns(net, atr, k=k, sl_mult=sl_mult, pip=PIP, risk_frac=risk)
        assert r[0] == pytest.approx(-risk)      # -1%

    def test_return_scales_with_net_pips(self):
        atr = np.array([0.0010, 0.0010])
        net = np.array([15.0, 30.0])             # doppelter Move -> doppelte Rendite
        r = pnl.trade_returns(net, atr, k=1.5, sl_mult=1.0, pip=PIP, risk_frac=0.01)
        assert r[1] == pytest.approx(2 * r[0])

    def test_zero_atr_gives_zero_return(self):
        r = pnl.trade_returns(np.array([10.0]), np.array([0.0]),
                              k=1.5, sl_mult=1.0, pip=PIP)
        assert r[0] == 0.0


class TestSharpe:

    def test_positive_series(self):
        r = np.array([0.01, 0.02, 0.015, 0.005])
        assert pnl.sharpe(r, periods_per_year=252) > 0

    def test_zero_std_returns_zero(self):
        assert pnl.sharpe(np.array([0.01, 0.01, 0.01]), 252) == 0.0

    def test_too_few_returns_zero(self):
        assert pnl.sharpe(np.array([0.01]), 252) == 0.0

    def test_annualization_scales(self):
        r = np.array([0.01, -0.005, 0.02, -0.01, 0.015])
        s1 = pnl.sharpe(r, 1)
        s2 = pnl.sharpe(r, 100)
        assert s2 == pytest.approx(s1 * np.sqrt(100))


class TestMetrics:

    def test_max_drawdown_negative(self):
        r = np.array([0.1, -0.2, 0.05, -0.1])
        assert pnl.max_drawdown(r) < 0

    def test_profit_factor(self):
        r = np.array([2.0, -1.0, 1.0, -1.0])   # Gewinne 3, Verluste 2
        assert pnl.profit_factor(r) == pytest.approx(1.5)

    def test_profit_factor_no_losses_is_inf(self):
        assert pnl.profit_factor(np.array([1.0, 2.0])) == float("inf")

    def test_win_rate(self):
        assert pnl.win_rate(np.array([1.0, -1.0, 1.0, 1.0])) == pytest.approx(0.75)

    def test_periods_per_year_floor(self):
        assert pnl.periods_per_year_from_trades(0, 5) == 1.0
        assert pnl.periods_per_year_from_trades(500, 5) == pytest.approx(100.0)


class TestNonOverlapping:

    def test_blocks_overlapping_entries(self):
        # Positionen 0,5,10,30 bei Horizont 16: 0 akzeptiert (Exit 16),
        # 5 & 10 blockiert, 30 wieder akzeptiert
        pos = np.array([0, 5, 10, 30])
        keep = pnl.non_overlapping_mask(pos, horizon=16)
        assert list(keep) == [True, False, False, True]

    def test_all_kept_when_spaced(self):
        pos = np.array([0, 20, 40, 60])
        keep = pnl.non_overlapping_mask(pos, horizon=16)
        assert keep.all()

    def test_daily_sharpe_interpretable_scale(self):
        # 3 positive Trades an 3 Tagen -> endliches, positives Sharpe
        import pandas as pd
        ts = pd.to_datetime(["2024-01-01 10:00", "2024-01-02 10:00",
                             "2024-01-03 10:00"], utc=True).to_numpy()
        r = np.array([0.01, 0.012, 0.008])
        sh = pnl.daily_sharpe(r, ts, ts[0], ts[-1])
        assert np.isfinite(sh) and sh > 0
