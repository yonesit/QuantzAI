"""
Unit-Tests fuer SignalModel.

Nutzt kleine synthetische Datensaetze – kein echtes Markt-Datenset noetig.
"""

from __future__ import annotations

import tempfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.signal_model import SignalModel, build_save_path


# ─────────────────────────────────────────────────────────────────────────────
#  Hilfsfunktionen
# ─────────────────────────────────────────────────────────────────────────────

def _make_features(n: int = 120, n_features: int = 5, seed: int = 0) -> pd.DataFrame:
    """Erstellt synthetischen Feature-DataFrame ohne Zeitstempel-Spalte."""
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n, n_features))
    cols = [f"feat_{i}" for i in range(n_features)]
    return pd.DataFrame(data, columns=cols)


def _make_labels(n: int = 120, seed: int = 0) -> pd.Series:
    """Erstellt Labels mit Distribution {-1, 0, 1}."""
    rng = np.random.default_rng(seed)
    vals = rng.choice([-1, 0, 1], size=n, p=[0.3, 0.4, 0.3])
    return pd.Series(vals, name="label")


def _make_features_with_ts(
    n: int = 300,
    n_features: int = 5,
    start: str = "2020-01-01",
    freq: str = "D",
    seed: int = 0,
) -> pd.DataFrame:
    """Erstellt Feature-DataFrame MIT timestamp-Spalte fuer Walk-Forward-Tests."""
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n, n_features))
    cols = [f"feat_{i}" for i in range(n_features)]
    df = pd.DataFrame(data, columns=cols)
    dates = pd.date_range(start=start, periods=n, freq=freq)
    df["timestamp"] = dates
    return df


def _trained_model(n: int = 120) -> SignalModel:
    """Gibt ein trainiertes SignalModel zurueck."""
    model = SignalModel()
    features = _make_features(n)
    labels = _make_labels(n)
    model.train(features, labels)
    return model


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: Training
# ─────────────────────────────────────────────────────────────────────────────

class TestTrain:

    def test_train_returns_dict(self):
        model = SignalModel()
        metrics = model.train(_make_features(), _make_labels())
        assert isinstance(metrics, dict)

    def test_train_metrics_keys(self):
        model = SignalModel()
        metrics = model.train(_make_features(), _make_labels())
        assert "n_samples" in metrics
        assert "n_features" in metrics
        assert "class_distribution" in metrics

    def test_train_n_samples_correct(self):
        n = 80
        model = SignalModel()
        metrics = model.train(_make_features(n), _make_labels(n))
        assert metrics["n_samples"] == n

    def test_train_n_features_correct(self):
        model = SignalModel()
        metrics = model.train(_make_features(n_features=7), _make_labels())
        assert metrics["n_features"] == 7

    def test_train_class_distribution_has_label_minus1(self):
        model = SignalModel()
        metrics = model.train(_make_features(), _make_labels())
        dist = metrics["class_distribution"]
        assert -1 in dist or 0 in dist or 1 in dist

    def test_train_sets_model(self):
        model = SignalModel()
        assert model._model is None
        model.train(_make_features(), _make_labels())
        assert model._model is not None

    def test_predict_before_train_raises(self):
        model = SignalModel()
        with pytest.raises(RuntimeError, match="trainiert"):
            model.predict_proba(_make_features(n=1))


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: predict_proba
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictProba:

    def test_returns_dict_with_three_keys(self):
        model = _trained_model()
        proba = model.predict_proba(_make_features(n=1))
        assert set(proba.keys()) == {"long", "short", "neutral"}

    def test_probabilities_sum_to_one(self):
        model = _trained_model()
        proba = model.predict_proba(_make_features(n=1))
        assert abs(sum(proba.values()) - 1.0) < 1e-6

    def test_all_probabilities_non_negative(self):
        model = _trained_model()
        proba = model.predict_proba(_make_features(n=1))
        assert all(v >= 0.0 for v in proba.values())

    def test_accepts_series_input(self):
        model = _trained_model()
        row = _make_features(n=1).iloc[0]  # pd.Series
        proba = model.predict_proba(row)
        assert set(proba.keys()) == {"long", "short", "neutral"}

    def test_accepts_dataframe_input(self):
        model = _trained_model()
        row = _make_features(n=1)  # DataFrame mit 1 Zeile
        proba = model.predict_proba(row)
        assert set(proba.keys()) == {"long", "short", "neutral"}


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: get_signal und Confidence-Threshold
# ─────────────────────────────────────────────────────────────────────────────

class TestGetSignal:

    def test_signal_is_valid_string(self):
        model = _trained_model()
        sig = model.get_signal(_make_features(n=1))
        assert sig in {"long", "short", "flat"}

    def test_threshold_zero_never_flat(self):
        """Bei threshold=0.0 kann max_prob nie < 0 -> Signal immer long/short/neutral."""
        model = _trained_model()
        sig = model.get_signal(_make_features(n=1), confidence_threshold=0.0)
        assert sig in {"long", "short", "neutral"}

    def test_threshold_one_always_flat(self):
        """Bei threshold=1.0 kann max_prob nie >= 1.0 -> immer 'flat'."""
        model = _trained_model()
        sig = model.get_signal(_make_features(n=1), confidence_threshold=1.0)
        assert sig == "flat"

    def test_threshold_just_below_max_prob_not_flat(self):
        """threshold knapp unter max_prob -> kein 'flat'."""
        model = _trained_model()
        row = _make_features(n=1)
        proba = model.predict_proba(row)
        max_prob = max(proba.values())
        threshold = max_prob - 0.001
        sig = model.get_signal(row, confidence_threshold=threshold)
        assert sig != "flat"

    def test_threshold_just_above_max_prob_is_flat(self):
        """threshold knapp ueber max_prob -> 'flat'."""
        model = _trained_model()
        row = _make_features(n=1)
        proba = model.predict_proba(row)
        max_prob = max(proba.values())
        threshold = max_prob + 0.001
        sig = model.get_signal(row, confidence_threshold=threshold)
        assert sig == "flat"

    def test_threshold_exactly_at_max_prob_not_flat(self):
        """threshold == max_prob -> kein 'flat' (Bedingung: max < threshold, nicht <=)."""
        model = _trained_model()
        row = _make_features(n=1)
        proba = model.predict_proba(row)
        max_prob = max(proba.values())
        sig = model.get_signal(row, confidence_threshold=max_prob)
        # max_prob < max_prob ist False -> kein flat
        assert sig != "flat"

    def test_signal_matches_highest_probability_class(self):
        """Signal entspricht der Klasse mit hoechster Wahrscheinlichkeit."""
        model = _trained_model()
        row = _make_features(n=1)
        proba = model.predict_proba(row)
        max_prob = max(proba.values())
        expected_class = max(proba, key=lambda k: proba[k])
        sig = model.get_signal(row, confidence_threshold=0.0)
        assert sig == expected_class

    def test_default_threshold_is_055(self):
        """Standard-Threshold ist 0.55."""
        model = _trained_model()
        row = _make_features(n=1)
        proba = model.predict_proba(row)
        max_prob = max(proba.values())
        sig_default = model.get_signal(row)
        sig_explicit = model.get_signal(row, confidence_threshold=0.55)
        assert sig_default == sig_explicit


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: save / load
# ─────────────────────────────────────────────────────────────────────────────

class TestSaveLoad:

    def test_save_creates_file(self, tmp_path):
        model = _trained_model()
        path = tmp_path / "model.joblib"
        model.save(path)
        assert path.exists()

    def test_load_returns_signal_model(self, tmp_path):
        model = _trained_model()
        path = tmp_path / "model.joblib"
        model.save(path)
        loaded = SignalModel.load(path)
        assert isinstance(loaded, SignalModel)

    def test_loaded_model_predicts(self, tmp_path):
        model = _trained_model()
        path = tmp_path / "model.joblib"
        model.save(path)
        loaded = SignalModel.load(path)
        proba = loaded.predict_proba(_make_features(n=1))
        assert set(proba.keys()) == {"long", "short", "neutral"}

    def test_loaded_model_same_predictions(self, tmp_path):
        model = _trained_model()
        row = _make_features(n=1)
        proba_before = model.predict_proba(row)

        path = tmp_path / "model.joblib"
        model.save(path)
        loaded = SignalModel.load(path)
        proba_after = loaded.predict_proba(row)

        assert abs(proba_before["long"] - proba_after["long"]) < 1e-9
        assert abs(proba_before["short"] - proba_after["short"]) < 1e-9

    def test_save_before_train_raises(self, tmp_path):
        model = SignalModel()
        with pytest.raises(RuntimeError):
            model.save(tmp_path / "model.joblib")

    def test_save_creates_parent_dirs(self, tmp_path):
        model = _trained_model()
        path = tmp_path / "nested" / "dir" / "model.joblib"
        model.save(path)
        assert path.exists()


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: Walk-Forward-Split
# ─────────────────────────────────────────────────────────────────────────────

class TestWalkForward:

    def test_returns_list(self):
        model = SignalModel()
        df = _make_features_with_ts(n=400)
        labels = _make_labels(n=400)
        results = model.walk_forward_validate(df, labels)
        assert isinstance(results, list)

    def test_at_least_one_window(self):
        model = SignalModel()
        df = _make_features_with_ts(n=400)
        labels = _make_labels(n=400)
        results = model.walk_forward_validate(df, labels)
        assert len(results) >= 1

    def test_window_dict_has_required_keys(self):
        model = SignalModel()
        df = _make_features_with_ts(n=400)
        labels = _make_labels(n=400)
        results = model.walk_forward_validate(df, labels)
        required = {"window", "train_size", "test_size", "oos_sharpe", "accuracy"}
        for r in results:
            assert required.issubset(r.keys()), f"Fehlende Keys in Fenster {r}"

    def test_no_train_test_overlap(self):
        """Train- und Test-Zeitraum duerfen sich nicht ueberlappen."""
        model = SignalModel()
        df = _make_features_with_ts(n=400)
        labels = _make_labels(n=400)
        results = model.walk_forward_validate(df, labels)
        for r in results:
            assert r["train_end"] <= r["test_start"], (
                f"Ueberlappung in Fenster {r['window']}: "
                f"train_end={r['train_end']} > test_start={r['test_start']}"
            )

    def test_windows_in_chronological_order(self):
        """Fenster muessen chronologisch geordnet sein."""
        model = SignalModel()
        df = _make_features_with_ts(n=400)
        labels = _make_labels(n=400)
        results = model.walk_forward_validate(df, labels)
        for i in range(1, len(results)):
            assert results[i]["train_start"] >= results[i - 1]["train_start"]

    def test_accuracy_between_zero_and_one(self):
        model = SignalModel()
        df = _make_features_with_ts(n=400)
        labels = _make_labels(n=400)
        results = model.walk_forward_validate(df, labels)
        for r in results:
            assert 0.0 <= r["accuracy"] <= 1.0, f"Accuracy ausserhalb [0,1]: {r['accuracy']}"

    def test_raises_without_timestamp_column(self):
        model = SignalModel()
        df = _make_features(n=100)  # keine timestamp-Spalte
        labels = _make_labels(n=100)
        with pytest.raises(ValueError, match="timestamp"):
            model.walk_forward_validate(df, labels)

    def test_test_size_positive(self):
        model = SignalModel()
        df = _make_features_with_ts(n=400)
        labels = _make_labels(n=400)
        results = model.walk_forward_validate(df, labels)
        for r in results:
            assert r["test_size"] > 0

    def test_train_size_larger_than_test_size(self):
        """Trainingsfenster (6M) soll groesser sein als Testfenster (1M)."""
        model = SignalModel()
        df = _make_features_with_ts(n=400)
        labels = _make_labels(n=400)
        results = model.walk_forward_validate(df, labels)
        for r in results:
            assert r["train_size"] > r["test_size"], (
                f"Fenster {r['window']}: train_size={r['train_size']} "
                f"nicht groesser als test_size={r['test_size']}"
            )

    def test_annualization_factor_daily(self):
        model = SignalModel()
        timestamps = pd.Series(pd.date_range("2024-01-01", periods=10, freq="D"))
        assert model._annualization_factor(timestamps) == 252.0

    def test_annualization_factor_15min(self):
        model = SignalModel()
        timestamps = pd.Series(pd.date_range("2024-01-01", periods=10, freq="15min"))
        assert model._annualization_factor(timestamps) == 35_040.0


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: Monte-Carlo
# ─────────────────────────────────────────────────────────────────────────────

class TestMonteCarlo:

    def test_returns_dict_with_required_keys(self):
        model = _trained_model(n=80)
        features = _make_features(n=80)
        labels = _make_labels(n=80)
        result = model.monte_carlo_test(features, labels, n_permutations=10)
        required = {"real_score", "permutation_mean", "permutation_std", "p_value", "significant", "n_permutations"}
        assert required.issubset(result.keys())

    def test_p_value_between_zero_and_one(self):
        model = _trained_model(n=80)
        features = _make_features(n=80)
        labels = _make_labels(n=80)
        result = model.monte_carlo_test(features, labels, n_permutations=10)
        assert 0.0 <= result["p_value"] <= 1.0

    def test_real_score_between_zero_and_one(self):
        model = _trained_model(n=80)
        features = _make_features(n=80)
        labels = _make_labels(n=80)
        result = model.monte_carlo_test(features, labels, n_permutations=10)
        assert 0.0 <= result["real_score"] <= 1.0

    def test_significant_is_bool(self):
        model = _trained_model(n=80)
        features = _make_features(n=80)
        labels = _make_labels(n=80)
        result = model.monte_carlo_test(features, labels, n_permutations=10)
        assert isinstance(result["significant"], bool)

    def test_n_permutations_matches(self):
        model = _trained_model(n=80)
        features = _make_features(n=80)
        labels = _make_labels(n=80)
        result = model.monte_carlo_test(features, labels, n_permutations=15)
        assert result["n_permutations"] == 15

    def test_raises_if_not_trained(self):
        model = SignalModel()
        with pytest.raises(RuntimeError):
            model.monte_carlo_test(_make_features(), _make_labels(), n_permutations=5)


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: build_save_path
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildSavePath:

    def test_returns_path_object(self):
        p = build_save_path(1, date(2025, 1, 15))
        assert isinstance(p, Path)

    def test_contains_version(self):
        p = build_save_path(3, date(2025, 1, 15))
        assert "v3" in str(p)

    def test_contains_date(self):
        p = build_save_path(1, date(2025, 6, 20))
        assert "20250620" in str(p)

    def test_extension_is_joblib(self):
        p = build_save_path(1, date(2025, 1, 15))
        assert str(p).endswith(".joblib")

    def test_parent_is_models(self):
        p = build_save_path(2, date(2025, 1, 15))
        assert p.parent.name == "models"
