"""
Unit-Tests fuer MeanReversionModel.

Synthetische OHLCV-Daten – kein MT5-Zugriff noetig.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.mean_reversion_model import MeanReversionModel
from src.models.label_builder import LabelBuilder


# ─────────────────────────────────────────────────────────────────────────────
#  Hilfsfunktionen
# ─────────────────────────────────────────────────────────────────────────────

def _make_ohlcv(n: int = 600, seed: int = 42) -> pd.DataFrame:
    """Erstellt synthetischen OHLCV-DataFrame mit Trend-artigem Kurs."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="4h")
    close = 1.1000 + np.cumsum(rng.normal(0, 0.0005, n))
    high  = close + rng.uniform(0.0001, 0.0010, n)
    low   = close - rng.uniform(0.0001, 0.0010, n)
    open_ = close + rng.normal(0, 0.0002, n)
    vol   = rng.integers(1000, 5000, n).astype(float)
    return pd.DataFrame({
        "timestamp": dates,
        "open":      open_,
        "high":      high,
        "low":       low,
        "close":     close,
        "volume":    vol,
    })


# ─────────────────────────────────────────────────────────────────────────────
#  Klassen-Konfiguration
# ─────────────────────────────────────────────────────────────────────────────

class TestMeanReversionModelConfig:
    def test_mr_label_params_tp(self):
        assert MeanReversionModel.MR_LABEL_PARAMS["tp_atr_mult"] == 1.0

    def test_mr_label_params_sl(self):
        assert MeanReversionModel.MR_LABEL_PARAMS["sl_atr_mult"] == 2.0

    def test_mr_label_params_max_candles(self):
        assert MeanReversionModel.MR_LABEL_PARAMS["max_candles"] == 10

    def test_mr_feature_names_count(self):
        assert len(MeanReversionModel.MR_FEATURE_NAMES) == 3

    def test_mr_feature_names_content(self):
        assert set(MeanReversionModel.MR_FEATURE_NAMES) == {
            "bb_pct_b", "dist_ema20_atr", "dist_sma50_atr"
        }

    def test_n_features_property(self):
        m = MeanReversionModel()
        assert m.n_features == 26

    def test_default_label_builder_returns_correct_type(self):
        lb = MeanReversionModel.default_label_builder()
        assert isinstance(lb, LabelBuilder)

    def test_default_label_builder_params_differ_from_standard(self):
        standard = LabelBuilder()
        mr = MeanReversionModel.default_label_builder()
        # tp muss enger sein als Standard (2.0)
        assert mr._tp_mult < standard._tp_mult
        # sl muss weiter sein als Standard (1.5)
        assert mr._sl_mult > standard._sl_mult
        # Horizont kuerzer
        assert mr._max_candles < standard._max_candles


# ─────────────────────────────────────────────────────────────────────────────
#  Feature-Engineering
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildFeatures:
    @pytest.fixture(scope="class")
    def features_df(self):
        df = _make_ohlcv(n=600)
        m = MeanReversionModel()
        return m.build_features(df, symbol="EURUSD", timeframe="H4")

    def test_returns_dataframe(self, features_df):
        assert isinstance(features_df, pd.DataFrame)

    def test_has_26_feature_columns(self, features_df):
        struct_cols = {"timestamp", "close", "high", "low"}
        feat_cols = [c for c in features_df.columns if c not in struct_cols]
        assert len(feat_cols) == 26, (
            f"Erwartet 26 Features, got {len(feat_cols)}: {feat_cols}"
        )

    def test_mr_features_present(self, features_df):
        for name in MeanReversionModel.MR_FEATURE_NAMES:
            assert name in features_df.columns, f"MR-Feature '{name}' fehlt"

    def test_no_h4_trend_or_d1_trend(self, features_df):
        assert "h4_trend" not in features_df.columns
        assert "d1_trend" not in features_df.columns

    def test_structural_columns_present(self, features_df):
        for col in ("timestamp", "close"):
            assert col in features_df.columns

    def test_bb_pct_b_no_all_nan_after_warmup(self, features_df):
        # Nach dem Warmup-Cut darf bb_pct_b nicht vollstaendig NaN sein
        non_nan = features_df["bb_pct_b"].notna().sum()
        assert non_nan > 0, "bb_pct_b ist nach Warmup komplett NaN"

    def test_dist_ema20_atr_finite_values(self, features_df):
        vals = features_df["dist_ema20_atr"].dropna()
        assert len(vals) > 0
        # Werte sollten im vernuenftigen Bereich liegen (nicht +/-inf)
        assert np.all(np.isfinite(vals))

    def test_dist_sma50_atr_finite_values(self, features_df):
        vals = features_df["dist_sma50_atr"].dropna()
        assert len(vals) > 0
        assert np.all(np.isfinite(vals))

    def test_bb_pct_b_range_plausible(self, features_df):
        vals = features_df["bb_pct_b"].dropna()
        # Extremwerte koennen >1 oder <0 liegen (ausserhalb BB), aber
        # der Median sollte um 0.5 liegen (Kurs nahe Mitte der Baender)
        median_val = vals.median()
        assert 0.1 < median_val < 0.9, (
            f"bb_pct_b Median {median_val:.3f} ausserhalb erwartetem Bereich [0.1, 0.9]"
        )

    def test_row_count_after_warmup(self, features_df):
        # 600 Bars - 200 Warmup = 400 max; nach NaN-Drop leicht weniger
        assert len(features_df) > 300


# ─────────────────────────────────────────────────────────────────────────────
#  Training und Walk-Forward
# ─────────────────────────────────────────────────────────────────────────────

class TestMrModelTraining:
    def _make_synthetic_features(self, n: int = 200) -> tuple[pd.DataFrame, pd.Series]:
        rng = np.random.default_rng(99)
        data = rng.standard_normal((n, 26))
        cols = [f"feat_{i}" for i in range(26)]
        X = pd.DataFrame(data, columns=cols)
        y = pd.Series(rng.choice([-1, 0, 1], size=n, p=[0.3, 0.4, 0.3]), name="label")
        return X, y

    def test_train_runs_without_error(self):
        m = MeanReversionModel()
        X, y = self._make_synthetic_features()
        result = m.train(X, y)
        assert isinstance(result, dict)

    def test_predict_proba_after_train(self):
        m = MeanReversionModel()
        X, y = self._make_synthetic_features()
        m.train(X, y)
        proba = m.predict_proba(X.iloc[[0]])
        assert isinstance(proba, dict)
        assert set(proba.keys()) >= {"long", "short", "neutral"}

    def test_get_signal_returns_valid_string(self):
        m = MeanReversionModel()
        X, y = self._make_synthetic_features()
        m.train(X, y)
        signal = m.get_signal(X.iloc[[0]])
        assert signal in {"long", "short", "flat"}

    def test_walk_forward_returns_list(self):
        rng = np.random.default_rng(7)
        n = 800
        data = rng.standard_normal((n, 5))
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        X = pd.DataFrame(data, columns=[f"f{i}" for i in range(5)])
        X["timestamp"] = dates
        y = pd.Series(rng.choice([-1, 0, 1], size=n), name="label")
        m = MeanReversionModel()
        results = m.walk_forward_validate(
            X, y, timestamp_col="timestamp", train_months=6, test_months=1
        )
        assert isinstance(results, list)
        assert len(results) > 0

    def test_walk_forward_window_has_oos_sharpe(self):
        rng = np.random.default_rng(7)
        n = 800
        data = rng.standard_normal((n, 5))
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        X = pd.DataFrame(data, columns=[f"f{i}" for i in range(5)])
        X["timestamp"] = dates
        y = pd.Series(rng.choice([-1, 0, 1], size=n), name="label")
        m = MeanReversionModel()
        results = m.walk_forward_validate(
            X, y, timestamp_col="timestamp", train_months=6, test_months=1
        )
        for r in results:
            assert "oos_sharpe" in r
            assert isinstance(r["oos_sharpe"], float)
