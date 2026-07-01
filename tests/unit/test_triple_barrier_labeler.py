"""
Unit-Tests fuer src/data/triple_barrier_labeler.py – reine Logik, synthetische
Mini-Datensaetze, kein MT5, keine echten paper_trades.json.

Abgedeckte Faelle: TP-vor-SL, SL-vor-TP, Timeout, Kosten kippen knappen Gewinn
ins Minus, No-Trade-Zone, Overnight-Swap ja/nein, Datenluecke, Spike.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.triple_barrier_labeler import (
    CostConfig, BarrierConfig, label_dataframe,
    compute_atr, net_pips_long, crosses_rollover, _scan_barrier,
)

PIP = 0.0001


def _mk_df(bars, start="2024-03-04 08:00", spread_price=0.00003):
    """bars: Liste (open, high, low, close). Zeitstempel M15 fortlaufend."""
    ts = pd.date_range(start, periods=len(bars), freq="15min", tz="UTC")
    o, h, l, c = zip(*bars)
    return pd.DataFrame({
        "timestamp": ts, "open": o, "high": h, "low": l, "close": c,
        "spread_median": [spread_price] * len(bars),
    })


# Kosten fast aus: nur damit Netto-Vorzeichen leicht zu steuern ist.
def _cheap_cost(**kw):
    base = dict(pip=PIP, spread_factor=0.0, commission_roundturn_pips=0.0,
                slippage_per_side_pips=0.0, swap_long_pips=0.0)
    base.update(kw)
    return CostConfig(**base)


# ── reine Bausteine ──────────────────────────────────────────────────────────

class TestBuildingBlocks:

    def test_atr_positive_after_warmup(self):
        df = _mk_df([(1.10, 1.11, 1.09, 1.10)] * 20)
        atr = compute_atr(df, period=14)
        assert np.isnan(atr[0])
        assert atr[-1] > 0

    def test_net_pips_long_subtracts_costs(self):
        cost = CostConfig(pip=PIP, spread_factor=1.0, commission_roundturn_pips=0.464,
                          slippage_per_side_pips=0.2, swap_long_pips=0.6)
        # gross 5 Pips, Spread 0.3 Pips (0.5*(0.3+0.3)*1.0), Komm 0.464, Slip 0.4
        net, total = net_pips_long(5 * PIP, 0.3, 0.3, overnight=False, cost=cost)
        assert total == pytest.approx(0.3 + 0.464 + 0.4)
        assert net == pytest.approx(5 - (0.3 + 0.464 + 0.4))

    def test_net_pips_long_overnight_adds_swap(self):
        cost = CostConfig(pip=PIP, spread_factor=0.0, commission_roundturn_pips=0.0,
                          slippage_per_side_pips=0.0, swap_long_pips=0.6)
        net_no, tot_no = net_pips_long(5 * PIP, 0, 0, overnight=False, cost=cost)
        net_on, tot_on = net_pips_long(5 * PIP, 0, 0, overnight=True, cost=cost)
        assert tot_on - tot_no == pytest.approx(0.6)
        assert net_no - net_on == pytest.approx(0.6)

    def test_crosses_rollover_true_when_spanning_22utc(self):
        entry = pd.Timestamp("2024-03-04 20:00", tz="UTC").value
        exit_ = pd.Timestamp("2024-03-04 23:00", tz="UTC").value
        assert crosses_rollover(entry, exit_) is True

    def test_crosses_rollover_false_intraday(self):
        entry = pd.Timestamp("2024-03-04 10:00", tz="UTC").value
        exit_ = pd.Timestamp("2024-03-04 13:00", tz="UTC").value
        assert crosses_rollover(entry, exit_) is False

    def test_scan_tp_before_sl(self):
        highs = np.array([0, 1.2, 1.2]); lows = np.array([0, 0.9, 0.9])
        closes = np.array([0, 1.1, 1.1])
        out = _scan_barrier(highs, lows, closes, 1.0, tp=1.15, sl=0.85, first=1, last=2)
        assert out[0] == "TP"

    def test_scan_simultaneous_is_pessimistic_sl(self):
        highs = np.array([0, 1.2]); lows = np.array([0, 0.8]); closes = np.array([0, 1.0])
        out = _scan_barrier(highs, lows, closes, 1.0, tp=1.15, sl=0.85, first=1, last=1)
        assert out[0] == "SL"


# ── Ende-zu-Ende Label-Faelle ────────────────────────────────────────────────

class TestLabelCases:

    def _bar_cfg(self, **kw):
        base = dict(horizon=4, atr_period=2, k=1.0, no_trade_hours=(21, 22),
                    spike_z=15.0, gap_max_minutes=60.0)
        base.update(kw)
        return BarrierConfig(**base)

    def test_tp_before_sl_gives_label_1(self):
        # ruhige Warmup-Bars, dann Entry, dann klarer Aufwaerts-Move -> TP, netto>0
        bars = [(1.1000, 1.1005, 1.0995, 1.1000)] * 4       # ATR-Warmup
        bars += [(1.1000, 1.1002, 1.0998, 1.1000)]          # Entry-Signal (t=4)
        bars += [(1.1000, 1.1300, 1.0999, 1.1250)]          # naechster Bar: TP hoch
        bars += [(1.1250, 1.1300, 1.1200, 1.1250)] * 3
        df = _mk_df(bars)
        res = label_dataframe(df, cost=_cheap_cost(), barrier=self._bar_cfg())
        row = res.iloc[4]
        assert row["status"] == "labeled"
        assert row["outcome_symmetric"] == "TP"
        assert row["label_symmetric"] == 1

    def test_sl_before_tp_gives_label_minus1(self):
        bars = [(1.1000, 1.1005, 1.0995, 1.1000)] * 4
        bars += [(1.1000, 1.1002, 1.0998, 1.1000)]          # Entry t=4
        bars += [(1.1000, 1.1001, 1.0700, 1.0750)]          # scharf runter -> SL
        bars += [(1.0750, 1.0800, 1.0700, 1.0750)] * 3
        df = _mk_df(bars)
        res = label_dataframe(df, cost=_cheap_cost(), barrier=self._bar_cfg())
        row = res.iloc[4]
        assert row["outcome_symmetric"] == "SL"
        assert row["label_symmetric"] == -1

    def test_timeout_flat_market_label_0(self):
        # flacher Markt, keine Barriere getroffen -> Timeout -> Label 0
        bars = [(1.1000, 1.1005, 1.0995, 1.1000)] * 12
        df = _mk_df(bars)
        res = label_dataframe(df, cost=_cheap_cost(), barrier=self._bar_cfg())
        row = res.iloc[4]
        assert row["outcome_symmetric"] == "TIMEOUT"
        assert row["label_symmetric"] == 0

    def test_costs_flip_marginal_winner_to_zero(self):
        # Brutto knapp positiver Timeout-Move, aber Kosten fressen ihn auf -> 0
        bars = [(1.1000, 1.1005, 1.0995, 1.1000)] * 4
        bars += [(1.1000, 1.1001, 1.0999, 1.1000)]          # Entry t=4
        # kleiner Aufwaerts-Drift (+0.5 Pip je Bar), nie TP/SL -> Timeout knapp +
        bars += [(1.1000, 1.10007, 1.09999, 1.10005)]
        bars += [(1.10005, 1.10012, 1.10004, 1.10010)]
        bars += [(1.10010, 1.10017, 1.10009, 1.10015)]
        bars += [(1.10015, 1.10022, 1.10014, 1.10020)]
        df = _mk_df(bars)
        cfg = self._bar_cfg(k=5.0)  # weite Barrieren -> sicher Timeout
        # teuer: hohe Kommission kippt den knappen Gewinn
        expensive = _cheap_cost(commission_roundturn_pips=5.0)
        res = label_dataframe(df, cost=expensive, barrier=cfg)
        row = res.iloc[4]
        assert row["outcome_symmetric"] == "TIMEOUT"
        assert row["gross_pips_symmetric"] > 0        # brutto positiv
        assert row["net_pips_symmetric"] < 0          # netto negativ
        assert row["label_symmetric"] == 0            # Kosten kippen es

    def test_no_trade_zone_not_labeled(self):
        # Entry-Bar um 21:00 UTC -> no_trade
        bars = [(1.1000, 1.1005, 1.0995, 1.1000)] * 12
        df = _mk_df(bars, start="2024-03-04 19:00")   # Index 8 = 21:00 UTC
        res = label_dataframe(df, cost=_cheap_cost(), barrier=self._bar_cfg())
        r21 = res[res["timestamp"].dt.hour == 21].iloc[0]
        assert r21["status"] == "no_trade"
        assert pd.isna(r21["label_symmetric"])

    def test_overnight_swap_applied_only_when_spanning(self):
        # Entry-Ausfuehrung 20:00, Timeout nach 3 h -> ueberspannt 22:00 -> Swap.
        bars = [(1.1000, 1.1005, 1.0995, 1.1000)] * 20
        df = _mk_df(bars, start="2024-03-04 18:45")    # idx4 = 19:45, open[5] = 20:00
        cfg = self._bar_cfg(k=50.0, horizon=12)        # 12 Bars = 3 h -> bis 23:00
        cost = _cheap_cost(swap_long_pips=3.0)
        res = label_dataframe(df, cost=cost, barrier=cfg)
        row = res.iloc[4]
        assert row["outcome_symmetric"] == "TIMEOUT"
        assert row["cost_pips_symmetric"] == pytest.approx(3.0)   # nur Swap greift

    def test_no_swap_when_not_spanning_rollover(self):
        # Entry-Ausfuehrung 10:00, kurzer Horizont -> kein 22:00 im Fenster.
        bars = [(1.1000, 1.1005, 1.0995, 1.1000)] * 12
        df = _mk_df(bars, start="2024-03-04 08:45")    # open[5] = 10:00
        cfg = self._bar_cfg(k=50.0, horizon=4)
        cost = _cheap_cost(swap_long_pips=3.0)
        res = label_dataframe(df, cost=cost, barrier=cfg)
        assert res.iloc[4]["cost_pips_symmetric"] == pytest.approx(0.0)

    def test_gap_skip_on_missing_bars(self):
        bars = [(1.1000, 1.1005, 1.0995, 1.1000)] * 12
        df = _mk_df(bars)
        # kuenstliche Luecke: schiebe die Zeit ab Bar 6 um 1 Tag nach vorn
        df.loc[6:, "timestamp"] = df.loc[6:, "timestamp"] + pd.Timedelta(days=1)
        res = label_dataframe(df, cost=_cheap_cost(), barrier=self._bar_cfg())
        # Entry bei t=4: Fenster [4..8] ueberspannt die Luecke -> gap_skip
        assert res.iloc[4]["status"] == "gap_skip"

    def test_spike_skip(self):
        # leichtes Rauschen (mad>0), dann Extrem-Spike im Haltefenster von t=4
        base = 1.1000
        bars = []
        for i in range(5):                                # idx0-4 (Entry t=4)
            c = base + (0.0001 if i % 2 else -0.0001)
            bars.append((base, max(base, c) + 0.0002, min(base, c) - 0.0002, c))
            base = c
        bars.append((base, base + 0.0200, base - 0.0002, base + 0.0200))  # Spike idx5
        for _ in range(5):
            bars.append((base + 0.0200, base + 0.0205, base + 0.0195, base + 0.0200))
        df = _mk_df(bars)
        res = label_dataframe(df, cost=_cheap_cost(), barrier=self._bar_cfg(spike_z=10.0))
        assert res.iloc[4]["status"] == "spike_skip"

    def test_both_designs_present(self):
        bars = [(1.1000, 1.1005, 1.0995, 1.1000)] * 12
        df = _mk_df(bars)
        res = label_dataframe(df, cost=_cheap_cost(), barrier=self._bar_cfg())
        for col in ("label_symmetric", "label_asymmetric",
                    "outcome_symmetric", "outcome_asymmetric",
                    "net_pips_symmetric", "net_pips_asymmetric"):
            assert col in res.columns

    def test_asymmetric_tp_farther_than_symmetric(self):
        # Gleichmaessige 10-Pip-Ranges -> ATR ~= 0.0010. k=1: sym TP = +10 Pip,
        # asym TP = +20 Pip. Move +14 Pip trifft sym-TP, nicht asym-TP.
        bars = [(1.1000, 1.1005, 1.0995, 1.1000)] * 5   # inkl. Entry-Bar idx4
        bars += [(1.1000, 1.1014, 1.0999, 1.1012)]      # +14 Pip Hoch, kein Rueckfall
        bars += [(1.1012, 1.1013, 1.1006, 1.1012)] * 5
        df = _mk_df(bars)
        cfg = self._bar_cfg(k=1.0, horizon=6)
        res = label_dataframe(df, cost=_cheap_cost(), barrier=cfg)
        row = res.iloc[4]
        assert row["outcome_symmetric"] == "TP"
        assert row["outcome_asymmetric"] in ("TIMEOUT", "SL")
