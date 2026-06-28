"""
tests/unit/test_wf_pnl.py
Tests fuer den echten P&L-Sharpe-Pfad (SCHRITT A):
  - pnl_sharpe(): annualisierter Sharpe aus echter Rendite-Serie
  - compute_wf_pnl_sharpe(): rollierende WF-Fenster mit vectorbt-P&L
  - aggregate_pnl_sharpe(): Mean/Median ueber Fenster

Deterministisch, kein MT5, keine echten paper_trades.json (synthetische Daten /
tmp_path), damit der Lauf reproduzierbar ist.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtesting.vectorbt_runner import BacktestConfig, pnl_sharpe
from src.backtesting.wf_pnl import aggregate_pnl_sharpe, compute_wf_pnl_sharpe


# ─────────────────────────────────────────────────────────────────────────────
#  pnl_sharpe (reine Rendite-Funktion)
# ─────────────────────────────────────────────────────────────────────────────

class TestPnlSharpe:
    def test_finite_on_known_returns(self, tmp_path):
        """Bekannter Renditedatensatz (ueber tmp_path geladen) -> endlicher Wert."""
        rng = np.random.default_rng(0)
        rets = pd.Series(rng.normal(0.001, 0.01, 500), name="r")
        csv = tmp_path / "returns.csv"
        rets.to_csv(csv, header=True)

        loaded = pd.read_csv(csv)["r"]
        s = pnl_sharpe(loaded, freq="4h")

        assert s is not None
        assert isinstance(s, float)
        assert np.isfinite(s)

    def test_positive_drift_positive_sharpe(self):
        rets = pd.Series([0.01, 0.011, 0.009, 0.012, 0.008, 0.0105])
        s = pnl_sharpe(rets, freq="4h")
        assert s is not None and s > 0

    def test_too_few_returns_none(self):
        assert pnl_sharpe(pd.Series([0.01]), freq="4h") is None

    def test_zero_std_none(self):
        assert pnl_sharpe(pd.Series([0.0, 0.0, 0.0, 0.0]), freq="4h") is None

    def test_nan_dropped(self):
        s = pnl_sharpe(pd.Series([0.01, np.nan, 0.012, 0.009, 0.011]), freq="4h")
        assert s is not None and np.isfinite(s)

    def test_annualization_scales_with_freq(self):
        rets = pd.Series([0.01, 0.012, 0.009, 0.011, 0.0105, 0.0095])
        s_4h = pnl_sharpe(rets, freq="4h")   # ann=2190
        s_1d = pnl_sharpe(rets, freq="1d")   # ann=252
        # Hoehere Frequenz -> hoeherer Annualisierungsfaktor -> groesserer Sharpe
        assert s_4h > s_1d > 0


# ─────────────────────────────────────────────────────────────────────────────
#  Synthetische Daten + Stub-Modell fuer compute_wf_pnl_sharpe
# ─────────────────────────────────────────────────────────────────────────────

class _AlwaysLong:
    """Deterministisches Stub-Modell: sagt immer Klasse 2 (long) voraus."""

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        n = len(X)
        p = np.zeros((n, 3), dtype=float)
        p[:, 2] = 1.0
        return p


def _synthetic_features(n_months: int = 8):
    """~n_months an 4h-Bars mit stetig steigendem Close (Long gewinnt)."""
    n = n_months * 30 * 6  # ~6 Bars/Tag
    idx = pd.date_range("2021-01-01", periods=n, freq="4h", tz="UTC")
    close = 1000.0 + np.arange(n) * 0.5
    df = pd.DataFrame({
        "timestamp": idx,
        "close": close,
        "feat_a": np.sin(np.arange(n) / 10.0),
    })
    labels = pd.Series(np.ones(n, dtype=int), index=df.index, name="label")
    return df, labels


class TestComputeWfPnlSharpe:
    def test_returns_windows_with_finite_sharpe(self):
        df, labels = _synthetic_features(n_months=9)
        cfg = BacktestConfig(freq="4h")

        results = compute_wf_pnl_sharpe(
            df, labels, feat_cols=["feat_a"],
            model_factory=_AlwaysLong, config=cfg,
        )

        assert len(results) >= 1
        for r in results:
            assert np.isfinite(r["oos_pnl_sharpe"])
        # Mindestens ein Fenster handelt tatsaechlich (immer-long -> 1 Trade/Fenster)
        assert any(r["n_trades"] >= 1 for r in results)

    def test_aggregate_returns_finite_mean(self):
        df, labels = _synthetic_features(n_months=9)
        results = compute_wf_pnl_sharpe(
            df, labels, feat_cols=["feat_a"],
            model_factory=_AlwaysLong, config=BacktestConfig(freq="4h"),
        )
        agg = aggregate_pnl_sharpe(results)
        assert agg["n_windows"] == len(results)
        assert agg["mean_pnl_sharpe"] is not None
        assert np.isfinite(agg["mean_pnl_sharpe"])

    def test_missing_timestamp_raises(self):
        df = pd.DataFrame({"close": [1.0, 2.0], "feat_a": [0.1, 0.2]})
        labels = pd.Series([1, 1])
        with pytest.raises(ValueError, match="timestamp"):
            compute_wf_pnl_sharpe(df, labels, ["feat_a"], _AlwaysLong)

    def test_missing_close_raises(self):
        idx = pd.date_range("2021-01-01", periods=2, freq="4h", tz="UTC")
        df = pd.DataFrame({"timestamp": idx, "feat_a": [0.1, 0.2]})
        labels = pd.Series([1, 1])
        with pytest.raises(ValueError, match="close"):
            compute_wf_pnl_sharpe(df, labels, ["feat_a"], _AlwaysLong)


class TestAggregateEmpty:
    def test_empty_results(self):
        agg = aggregate_pnl_sharpe([])
        assert agg["n_windows"] == 0
        assert agg["mean_pnl_sharpe"] is None
        assert agg["total_trades"] == 0
