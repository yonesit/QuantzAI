"""
Unit-Tests fuer src/models/interpretability.py (SHAP).

Nutzt ein kleines synthetisches Modell (30 Samples, 4 Features) fuer schnelle Laeufe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.signal_model import SignalModel
from src.models.interpretability import (
    explain_prediction,
    explain_batch,
    get_expected_values,
    _normalize_shap,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Hilfsfunktionen
# ─────────────────────────────────────────────────────────────────────────────

N_SAMPLES = 60
N_FEATURES = 4
FEATURE_NAMES = [f"f{i}" for i in range(N_FEATURES)]


def _make_df(n: int = N_SAMPLES, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.standard_normal((n, N_FEATURES)), columns=FEATURE_NAMES)


def _make_labels(n: int = N_SAMPLES, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.choice([-1, 0, 1], size=n), name="label")


def _trained() -> tuple[SignalModel, pd.DataFrame, pd.Series]:
    model = SignalModel(lgbm_params={"n_estimators": 20, "num_leaves": 8})
    df = _make_df()
    labels = _make_labels()
    model.train(df, labels)
    return model, df, labels


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: explain_prediction
# ─────────────────────────────────────────────────────────────────────────────

class TestExplainPrediction:

    def test_returns_dict_with_three_classes(self):
        model, df, _ = _trained()
        result = explain_prediction(model, df.iloc[[0]])
        assert set(result.keys()) == {"short", "neutral", "long"}

    def test_each_class_has_all_features(self):
        model, df, _ = _trained()
        result = explain_prediction(model, df.iloc[[0]])
        for cls_name, shap_dict in result.items():
            assert set(shap_dict.keys()) == set(FEATURE_NAMES), (
                f"Klasse '{cls_name}' hat falsche Feature-Keys"
            )

    def test_values_are_floats(self):
        model, df, _ = _trained()
        result = explain_prediction(model, df.iloc[[0]])
        for cls_name, shap_dict in result.items():
            for feat, val in shap_dict.items():
                assert isinstance(val, float), f"{cls_name}/{feat} ist kein float"

    def test_accepts_series_input(self):
        model, df, _ = _trained()
        result = explain_prediction(model, df.iloc[0])
        assert set(result.keys()) == {"short", "neutral", "long"}

    def test_raises_if_model_not_trained(self):
        model = SignalModel()
        df = _make_df(n=1)
        with pytest.raises(RuntimeError, match="trainiert"):
            explain_prediction(model, df)


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: Konsistenz-Check (SHAP-Werte + E[f(x)] == Modell-Output)
# ─────────────────────────────────────────────────────────────────────────────

class TestShapConsistency:
    """
    Kerntest: SHAP-Werte summieren sich korrekt zur Modell-Vorhersage.

    Fuer jede Klasse k gilt:
      sum(shap_values[k]) + expected_value[k] == raw_model_output[k]

    LightGBM gibt raw outputs als Log-Odds zurueck (predict_raw / pred_raw_score).
    TreeExplainer garantiert diese Additivitaet (Shapley-Eigenschaft).
    """

    def test_shap_sum_plus_expected_equals_raw_output_single(self):
        """Einzelne Zeile: SHAP + E[f] == roher Modell-Output fuer alle Klassen."""
        model, df, _ = _trained()
        row = df.iloc[[0]]

        import shap as shap_lib
        lgbm = model._model
        explainer = shap_lib.TreeExplainer(lgbm)

        X = row[FEATURE_NAMES].values.astype(float)
        shap_vals = explainer.shap_values(X)

        from src.models.interpretability import _normalize_shap
        shap_arr = _normalize_shap(shap_vals)  # (1, n_feat, 3)
        ev = np.asarray(explainer.expected_value).flatten()

        raw_output = lgbm.predict(X, raw_score=True)  # shape (1, 3)

        for cls_idx in range(3):
            computed = shap_arr[0, :, cls_idx].sum() + ev[cls_idx]
            expected = float(raw_output[0, cls_idx])
            assert abs(computed - expected) < 1e-5, (
                f"Konsistenz verletzt fuer Klasse {cls_idx}: "
                f"SHAP+EV={computed:.6f} != raw={expected:.6f}"
            )

    def test_shap_sum_plus_expected_equals_raw_output_batch(self):
        """Batch: Konsistenz fuer alle Samples und alle Klassen."""
        model, df, _ = _trained()

        import shap as shap_lib
        lgbm = model._model
        explainer = shap_lib.TreeExplainer(lgbm)

        X = df[FEATURE_NAMES].values.astype(float)
        shap_vals = explainer.shap_values(X)
        from src.models.interpretability import _normalize_shap
        shap_arr = _normalize_shap(shap_vals)  # (n, n_feat, 3)
        ev = np.asarray(explainer.expected_value).flatten()

        raw_output = lgbm.predict(X, raw_score=True)  # (n, 3)

        for cls_idx in range(3):
            computed = shap_arr[:, :, cls_idx].sum(axis=1) + ev[cls_idx]
            expected = raw_output[:, cls_idx]
            max_err = np.abs(computed - expected).max()
            assert max_err < 1e-4, (
                f"Konsistenz verletzt fuer Klasse {cls_idx}: max_err={max_err:.2e}"
            )

    def test_shap_values_not_all_zero(self):
        """SHAP-Werte duerfen nicht alle null sein (wuerde auf Fehler hindeuten)."""
        model, df, _ = _trained()
        result = explain_prediction(model, df.iloc[[0]])
        all_zeros = all(
            abs(v) < 1e-12
            for cls_dict in result.values()
            for v in cls_dict.values()
        )
        assert not all_zeros, "Alle SHAP-Werte sind 0 – Berechnungsfehler?"


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: explain_batch
# ─────────────────────────────────────────────────────────────────────────────

class TestExplainBatch:

    def test_returns_dict_with_three_classes(self):
        model, df, _ = _trained()
        result = explain_batch(model, df)
        assert set(result.keys()) == {"short", "neutral", "long"}

    def test_each_class_is_dataframe(self):
        model, df, _ = _trained()
        result = explain_batch(model, df)
        for cls_name, shap_df in result.items():
            assert isinstance(shap_df, pd.DataFrame), f"{cls_name} ist kein DataFrame"

    def test_shape_matches_input(self):
        model, df, _ = _trained()
        result = explain_batch(model, df)
        for cls_name, shap_df in result.items():
            assert shap_df.shape == (len(df), N_FEATURES), (
                f"Shape fuer '{cls_name}' falsch: {shap_df.shape}"
            )

    def test_columns_match_features(self):
        model, df, _ = _trained()
        result = explain_batch(model, df)
        for cls_name, shap_df in result.items():
            assert list(shap_df.columns) == FEATURE_NAMES

    def test_index_preserved(self):
        model, df, _ = _trained()
        df_idx = df.copy()
        df_idx.index = range(100, 100 + len(df))
        result = explain_batch(model, df_idx)
        for cls_name, shap_df in result.items():
            assert list(shap_df.index) == list(df_idx.index)

    def test_raises_if_not_trained(self):
        model = SignalModel()
        with pytest.raises(RuntimeError):
            explain_batch(model, _make_df())

    def test_ignores_timestamp_column(self):
        """timestamp-Spalte im DataFrame stoert explain_batch nicht."""
        model, df, _ = _trained()
        df_ts = df.copy()
        df_ts["timestamp"] = pd.date_range("2024-01-01", periods=len(df), freq="h")
        result = explain_batch(model, df_ts)
        for cls_name, shap_df in result.items():
            assert "timestamp" not in shap_df.columns


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: get_expected_values
# ─────────────────────────────────────────────────────────────────────────────

class TestGetExpectedValues:

    def test_returns_dict_with_three_classes(self):
        model, _, _ = _trained()
        ev = get_expected_values(model)
        assert set(ev.keys()) == {"short", "neutral", "long"}

    def test_values_are_floats(self):
        model, _, _ = _trained()
        ev = get_expected_values(model)
        for k, v in ev.items():
            assert isinstance(v, float), f"{k}: kein float"

    def test_raises_if_not_trained(self):
        model = SignalModel()
        with pytest.raises(RuntimeError):
            get_expected_values(model)


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: _normalize_shap
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalizeShap:

    def test_list_input_produces_correct_shape(self):
        # Liste von 3 arrays (n, f) -> (n, f, 3)
        n, f = 5, 4
        arr_list = [np.zeros((n, f)) for _ in range(3)]
        result = _normalize_shap(arr_list)
        assert result.shape == (n, f, 3)

    def test_3d_array_passthrough(self):
        arr = np.zeros((5, 4, 3))
        result = _normalize_shap(arr)
        assert result.shape == (5, 4, 3)

    def test_invalid_shape_raises(self):
        arr = np.zeros((5, 4))  # 2D – kein gueltiger SHAP-Output
        with pytest.raises((ValueError, Exception)):
            _normalize_shap(arr)

    def test_values_preserved(self):
        n, f = 3, 2
        arrays = [np.ones((n, f)) * i for i in range(3)]
        result = _normalize_shap(arrays)
        for cls_idx in range(3):
            np.testing.assert_array_equal(result[:, :, cls_idx], arrays[cls_idx])
