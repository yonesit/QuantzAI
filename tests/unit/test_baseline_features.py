"""
Tests fuer die Baseline-Features – Fokus: KEIN Look-Ahead (Kausalitaet).
Synthetische Daten, kein MT5, keine echten Handelsdaten.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.baseline_trainer import build_features


def _synthetic(n=400, seed=0):
    rng = np.random.default_rng(seed)
    close = 1.10 + np.cumsum(rng.normal(0, 0.0005, n))
    high = close + rng.uniform(0.0001, 0.0006, n)
    low = close - rng.uniform(0.0001, 0.0006, n)
    open_ = close + rng.normal(0, 0.0002, n)
    vol = rng.uniform(100, 1000, n)
    ts = pd.date_range("2020-01-01", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame({"timestamp": ts, "open": open_, "high": high,
                         "low": low, "close": close, "volume": vol})


class TestCausality:

    def test_future_perturbation_does_not_change_past_features(self):
        """Aendert man einen ZUKUENFTIGEN Bar, duerfen Features frueherer Bars
        sich NICHT aendern (harte Kausalitaets-/Look-Ahead-Pruefung)."""
        df = _synthetic()
        frame_a, names = build_features(df)

        t = 200
        df2 = df.copy()
        # massive Stoerung aller Bars ab t+1
        df2.loc[t + 1:, ["open", "high", "low", "close"]] *= 1.05
        frame_b, _ = build_features(df2)

        a = frame_a.loc[:t, names].to_numpy()
        b = frame_b.loc[:t, names].to_numpy()
        # NaNs an gleicher Stelle, Werte sonst identisch
        both_nan = np.isnan(a) & np.isnan(b)
        equal = np.isclose(a, b, equal_nan=False) | both_nan
        assert equal.all(), "Feature eines vergangenen Bars haengt von der Zukunft ab!"

    def test_all_features_present(self):
        df = _synthetic()
        _, names = build_features(df)
        for expected in ("rsi_14", "atr_norm", "dist_ema20", "macd_diff_norm",
                         "adx", "bb_pos", "stoch_k", "hour", "dow"):
            assert expected in names

    def test_warmup_rows_are_nan_not_dropped(self):
        df = _synthetic()
        frame, names = build_features(df)
        # Frame behaelt volle Laenge (Alignment fuer Bar-Position-Purging)
        assert len(frame) == len(df)
        # erste Zeile hat NaN-Features (Warmup)
        assert frame.loc[0, names].isna().any()
