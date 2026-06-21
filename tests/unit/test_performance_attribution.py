"""
tests/unit/test_performance_attribution.py
Unit-Tests fuer PerformanceAttribution, SHAP-Aggregation und Drift-Erkennung.

Abgedeckt:
  - AttributionConfig: Standardwerte und angepasste Werte
  - FeatureImportanceSummary / DriftReport: Datenklassen
  - Rolling-Window: record_shap, record_shap_batch, window_size, maxlen
  - compute_attribution: Aggregation, Sortierung, leeres Fenster, Caching
  - get_top_features: n-Parameter, leeres Fenster
  - detect_drift: kein Drift, Drift ueber Schwelle, training_val=0, zu wenig Daten
  - get_retrain_warning: None bei kein Drift, String bei Drift
  - run_async_update: Thread gestartet, SHAP-Fn wird aufgerufen, Ergebnis gecacht
  - _async_worker: Fehler-Handling ohne Exception-Propagation
  - extract_training_importance: Hilfsfunktion
  - format_importance_row (GUI): pure function
  - AttributionSnapshot (GUI): Datenklasse
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.models.performance_attribution import (
    AttributionConfig,
    DriftReport,
    FeatureImportanceSummary,
    PerformanceAttribution,
    _default_shap_fn,
    extract_training_importance,
)
from gui.views.attribution_view import (
    AttributionSnapshot,
    format_importance_row,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Hilfsfunktionen
# ─────────────────────────────────────────────────────────────────────────────

def _shap_row(**kwargs: float) -> dict[str, float]:
    """Erstellt ein einfaches SHAP-Dict."""
    return dict(kwargs)


def _make_shap_rows(
    n: int,
    features: list[str] | None = None,
    seed: int = 42,
) -> list[dict[str, float]]:
    """Erstellt n zufaellige SHAP-Dicts."""
    rng = np.random.default_rng(seed)
    feats = features or ["feat_a", "feat_b", "feat_c"]
    return [
        {f: float(rng.standard_normal()) for f in feats}
        for _ in range(n)
    ]


def _mock_shap_fn(rows: list[dict[str, float]]):
    """Erstellt eine injizierbare _shap_fn die feste Rows zurueckgibt."""
    def fn(model, features_df):
        return rows[: len(features_df)]
    return fn


def _pa(config=None, shap_fn=None) -> PerformanceAttribution:
    return PerformanceAttribution(config=config, _shap_fn=shap_fn)


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: AttributionConfig
# ─────────────────────────────────────────────────────────────────────────────

class TestAttributionConfig:

    def test_defaults(self):
        cfg = AttributionConfig()
        assert cfg.window_size     == 100
        assert cfg.top_n           == 5
        assert cfg.drift_threshold == 0.5
        assert cfg.min_records     == 10

    def test_custom_values(self):
        cfg = AttributionConfig(window_size=50, top_n=3, drift_threshold=0.3, min_records=5)
        assert cfg.window_size     == 50
        assert cfg.top_n           == 3
        assert cfg.drift_threshold == 0.3
        assert cfg.min_records     == 5


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: Datenklassen
# ─────────────────────────────────────────────────────────────────────────────

class TestDataclasses:

    def test_feature_importance_summary_fields(self):
        fis = FeatureImportanceSummary(
            feature_names=["a", "b"],
            mean_abs_shap=[0.3, 0.1],
            n_records=10,
            top_features=[("a", 0.3)],
        )
        assert fis.n_records == 10
        assert fis.feature_names == ["a", "b"]
        assert fis.computed_at is not None

    def test_drift_report_fields(self):
        dr = DriftReport(
            drift_scores={"a": 0.6},
            drifted_features=["a"],
            max_drift=0.6,
            has_significant_drift=True,
            retrain_recommended=True,
            reason="drift!",
        )
        assert dr.retrain_recommended is True
        assert dr.computed_at is not None

    def test_feature_importance_timestamp_utc(self):
        before = datetime.now(timezone.utc)
        fis = FeatureImportanceSummary(
            feature_names=[], mean_abs_shap=[], n_records=0, top_features=[]
        )
        after = datetime.now(timezone.utc)
        assert before <= fis.computed_at <= after

    def test_drift_report_timestamp_utc(self):
        before = datetime.now(timezone.utc)
        dr = DriftReport(
            drift_scores={}, drifted_features=[], max_drift=0.0,
            has_significant_drift=False, retrain_recommended=False, reason=""
        )
        after = datetime.now(timezone.utc)
        assert before <= dr.computed_at <= after


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: Rolling-Window
# ─────────────────────────────────────────────────────────────────────────────

class TestRollingWindow:

    def test_initially_empty(self):
        pa = _pa()
        assert pa.window_size() == 0

    def test_record_shap_adds_entry(self):
        pa = _pa()
        pa.record_shap({"a": 0.1, "b": 0.2})
        assert pa.window_size() == 1

    def test_record_shap_batch_adds_multiple(self):
        pa = _pa()
        pa.record_shap_batch(_make_shap_rows(5))
        assert pa.window_size() == 5

    def test_window_capped_at_maxlen(self):
        cfg = AttributionConfig(window_size=10)
        pa  = _pa(config=cfg)
        pa.record_shap_batch(_make_shap_rows(20))
        assert pa.window_size() == 10

    def test_rolling_behavior_oldest_dropped(self):
        cfg = AttributionConfig(window_size=3)
        pa  = _pa(config=cfg)
        # Fuege 3 eindeutige Eintraege ein, dann einen weiteren
        pa.record_shap({"feat": 1.0})
        pa.record_shap({"feat": 2.0})
        pa.record_shap({"feat": 3.0})
        pa.record_shap({"feat": 4.0})  # schiebt {feat:1.0} heraus
        summary = pa.compute_attribution()
        # Mittelwert von 2.0, 3.0, 4.0 = 3.0 (abs)
        assert abs(summary.mean_abs_shap[0] - 3.0) < 1e-9

    def test_record_shap_dict_is_copied(self):
        pa = _pa()
        d  = {"a": 1.0}
        pa.record_shap(d)
        d["a"] = 99.0  # nachtraegliche Aenderung darf Window nicht betreffen
        summary = pa.compute_attribution()
        assert abs(summary.mean_abs_shap[0] - 1.0) < 1e-9

    def test_record_shap_batch_dicts_copied(self):
        pa   = _pa()
        rows = [{"a": 0.5}]
        pa.record_shap_batch(rows)
        rows[0]["a"] = 99.0
        summary = pa.compute_attribution()
        assert abs(summary.mean_abs_shap[0] - 0.5) < 1e-9


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: compute_attribution
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeAttribution:

    def test_empty_window_returns_empty_summary(self):
        pa      = _pa()
        summary = pa.compute_attribution()
        assert summary.n_records == 0
        assert summary.feature_names == []
        assert summary.top_features  == []

    def test_single_record(self):
        pa = _pa()
        pa.record_shap({"feat_a": 0.5, "feat_b": -0.3})
        summary = pa.compute_attribution()
        assert summary.n_records == 1
        feat_map = dict(zip(summary.feature_names, summary.mean_abs_shap))
        assert abs(feat_map["feat_a"] - 0.5) < 1e-9
        assert abs(feat_map["feat_b"] - 0.3) < 1e-9

    def test_uses_absolute_values(self):
        pa = _pa()
        pa.record_shap({"feat": -0.8})
        summary = pa.compute_attribution()
        assert abs(summary.mean_abs_shap[0] - 0.8) < 1e-9

    def test_mean_across_multiple_records(self):
        pa = _pa()
        pa.record_shap({"a": 0.2})
        pa.record_shap({"a": 0.6})
        summary = pa.compute_attribution()
        assert abs(summary.mean_abs_shap[0] - 0.4) < 1e-9

    def test_sorted_descending(self):
        pa = _pa()
        pa.record_shap({"low": 0.1, "high": 0.9, "mid": 0.5})
        summary = pa.compute_attribution()
        assert summary.feature_names[0] == "high"
        assert summary.feature_names[-1] == "low"
        assert summary.mean_abs_shap == sorted(summary.mean_abs_shap, reverse=True)

    def test_top_features_respects_top_n(self):
        cfg = AttributionConfig(top_n=2)
        pa  = _pa(config=cfg)
        pa.record_shap({"a": 0.9, "b": 0.7, "c": 0.5})
        summary = pa.compute_attribution()
        assert len(summary.top_features) == 2
        assert summary.top_features[0][0] == "a"

    def test_missing_feature_in_some_records(self):
        pa = _pa()
        pa.record_shap({"a": 1.0, "b": 0.5})
        pa.record_shap({"a": 0.8})        # "b" fehlt
        summary = pa.compute_attribution()
        feat_map = dict(zip(summary.feature_names, summary.mean_abs_shap))
        # "a": mean(1.0, 0.8) = 0.9
        assert abs(feat_map["a"] - 0.9) < 1e-9
        # "b": mean(0.5) = 0.5  (nur ein Record hat "b")
        assert abs(feat_map["b"] - 0.5) < 1e-9

    def test_caches_last_summary(self):
        pa = _pa()
        pa.record_shap({"x": 1.0})
        summary1 = pa.compute_attribution()
        cached   = pa.get_last_summary()
        assert cached is not None
        assert cached.n_records == summary1.n_records

    def test_get_last_summary_none_before_compute(self):
        pa = _pa()
        assert pa.get_last_summary() is None

    def test_empty_window_also_caches(self):
        pa = _pa()
        pa.compute_attribution()
        cached = pa.get_last_summary()
        assert cached is not None
        assert cached.n_records == 0


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: get_top_features
# ─────────────────────────────────────────────────────────────────────────────

class TestGetTopFeatures:

    def test_returns_top_n_by_default(self):
        cfg = AttributionConfig(top_n=3)
        pa  = _pa(config=cfg)
        pa.record_shap({"a": 0.9, "b": 0.7, "c": 0.5, "d": 0.3})
        top = pa.get_top_features()
        assert len(top) == 3
        assert top[0][0] == "a"

    def test_override_n(self):
        pa = _pa()
        pa.record_shap({"a": 0.9, "b": 0.7, "c": 0.5})
        top = pa.get_top_features(n=2)
        assert len(top) == 2

    def test_n_larger_than_available_features(self):
        pa = _pa()
        pa.record_shap({"a": 0.5})
        top = pa.get_top_features(n=10)
        assert len(top) == 1

    def test_empty_window_returns_empty(self):
        pa  = _pa()
        top = pa.get_top_features()
        assert top == []

    def test_sorted_descending(self):
        pa = _pa()
        pa.record_shap({"x": 0.3, "y": 0.9, "z": 0.1})
        top = pa.get_top_features(n=3)
        values = [v for _, v in top]
        assert values == sorted(values, reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: detect_drift
# ─────────────────────────────────────────────────────────────────────────────

class TestDetectDrift:

    def _pa_with_records(self, n: int = 15, val: float = 0.1) -> PerformanceAttribution:
        cfg = AttributionConfig(min_records=10)
        pa  = _pa(config=cfg)
        pa.record_shap_batch([{"feat_a": val, "feat_b": val * 0.5}] * n)
        return pa

    def test_no_drift_below_threshold(self):
        pa      = self._pa_with_records(15, 0.4)
        report  = pa.detect_drift({"feat_a": 0.4, "feat_b": 0.2}, threshold=0.5)
        assert report.has_significant_drift is False
        assert report.retrain_recommended   is False
        assert report.drifted_features      == []

    def test_drift_above_threshold(self):
        pa = self._pa_with_records(15, 0.1)   # live: 0.1
        # training hatte 0.4 fuer feat_a -> rel_change = |0.1-0.4|/0.4 = 0.75 > 0.5
        report = pa.detect_drift({"feat_a": 0.4}, threshold=0.5)
        assert report.has_significant_drift is True
        assert "feat_a" in report.drifted_features

    def test_max_drift_correct(self):
        cfg = AttributionConfig(min_records=10)
        pa  = _pa(config=cfg)
        pa.record_shap_batch([{"a": 0.1, "b": 0.9}] * 15)
        # training: a=0.4, b=0.9 -> drift_a=0.75, drift_b=0.0
        report = pa.detect_drift({"a": 0.4, "b": 0.9})
        assert abs(report.max_drift - 0.75) < 1e-6

    def test_drift_scores_per_feature(self):
        cfg = AttributionConfig(min_records=5)
        pa  = _pa(config=cfg)
        pa.record_shap_batch([{"feat": 0.1}] * 10)
        report = pa.detect_drift({"feat": 0.4}, threshold=0.5)
        # |0.1-0.4|/0.4 = 0.75
        assert abs(report.drift_scores["feat"] - 0.75) < 1e-6

    def test_training_val_zero(self):
        """train_val=0 -> keine Division durch Null."""
        cfg = AttributionConfig(min_records=5)
        pa  = _pa(config=cfg)
        pa.record_shap_batch([{"feat": 0.5}] * 10)
        report = pa.detect_drift({"feat": 0.0}, threshold=0.5)
        # live=0.5, train=0 -> rel = |0.5| = 0.5; bei threshold=0.5 ist > False
        assert report.drift_scores["feat"] == pytest.approx(0.5)

    def test_training_val_zero_above_threshold(self):
        cfg = AttributionConfig(min_records=5)
        pa  = _pa(config=cfg)
        pa.record_shap_batch([{"feat": 0.8}] * 10)
        report = pa.detect_drift({"feat": 0.0}, threshold=0.5)
        assert report.drift_scores["feat"] == pytest.approx(0.8)
        assert "feat" in report.drifted_features

    def test_too_few_records_no_retrain(self):
        cfg = AttributionConfig(min_records=20)
        pa  = _pa(config=cfg)
        pa.record_shap_batch([{"feat": 0.1}] * 5)   # nur 5 < 20
        report = pa.detect_drift({"feat": 0.9}, threshold=0.1)
        # Drift vorhanden, aber zu wenig Daten -> kein retrain
        assert report.retrain_recommended is False
        assert "wenig" in report.reason.lower() or "daten" in report.reason.lower()

    def test_retrain_recommended_when_drift_and_enough_records(self):
        cfg = AttributionConfig(min_records=5)
        pa  = _pa(config=cfg)
        pa.record_shap_batch([{"feat": 0.05}] * 10)
        report = pa.detect_drift({"feat": 0.9}, threshold=0.5)
        assert report.retrain_recommended is True
        assert report.has_significant_drift is True

    def test_reason_mentions_drifted_feature(self):
        cfg = AttributionConfig(min_records=5)
        pa  = _pa(config=cfg)
        pa.record_shap_batch([{"feat_x": 0.05}] * 10)
        report = pa.detect_drift({"feat_x": 0.9}, threshold=0.5)
        if report.has_significant_drift:
            assert "feat_x" in report.reason

    def test_no_drift_reason_message(self):
        pa     = self._pa_with_records(15, 0.4)
        report = pa.detect_drift({"feat_a": 0.4}, threshold=0.5)
        assert "kein" in report.reason.lower()

    def test_uses_config_threshold_by_default(self):
        cfg = AttributionConfig(min_records=5, drift_threshold=0.9)
        pa  = _pa(config=cfg)
        pa.record_shap_batch([{"feat": 0.1}] * 10)
        # rel_change = |0.1-0.4|/0.4 = 0.75 < 0.9 -> kein Drift bei cfg-Schwelle
        report = pa.detect_drift({"feat": 0.4})
        assert report.has_significant_drift is False

    def test_explicit_threshold_overrides_config(self):
        cfg = AttributionConfig(min_records=5, drift_threshold=0.9)
        pa  = _pa(config=cfg)
        pa.record_shap_batch([{"feat": 0.1}] * 10)
        # rel_change = 0.75 > 0.3 -> Drift mit expliziter Schwelle
        report = pa.detect_drift({"feat": 0.4}, threshold=0.3)
        assert report.has_significant_drift is True

    def test_empty_training_importance(self):
        pa     = _pa()
        report = pa.detect_drift({})
        assert report.drift_scores == {}
        assert report.max_drift == 0.0
        assert report.has_significant_drift is False


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: get_retrain_warning
# ─────────────────────────────────────────────────────────────────────────────

class TestGetRetrainWarning:

    def test_no_drift_returns_none(self):
        cfg = AttributionConfig(min_records=5)
        pa  = _pa(config=cfg)
        pa.record_shap_batch([{"feat": 0.4}] * 10)
        result = pa.get_retrain_warning({"feat": 0.4}, threshold=0.5)
        assert result is None

    def test_drift_returns_string(self):
        cfg = AttributionConfig(min_records=5)
        pa  = _pa(config=cfg)
        pa.record_shap_batch([{"feat": 0.05}] * 10)
        result = pa.get_retrain_warning({"feat": 0.9}, threshold=0.5)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_too_few_records_returns_none(self):
        cfg = AttributionConfig(min_records=50)
        pa  = _pa(config=cfg)
        pa.record_shap_batch([{"feat": 0.05}] * 5)
        result = pa.get_retrain_warning({"feat": 0.9}, threshold=0.1)
        assert result is None

    def test_warning_mentions_feature(self):
        cfg = AttributionConfig(min_records=5)
        pa  = _pa(config=cfg)
        pa.record_shap_batch([{"critical_feat": 0.05}] * 10)
        result = pa.get_retrain_warning({"critical_feat": 0.9}, threshold=0.2)
        if result is not None:
            assert "critical_feat" in result


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: run_async_update
# ─────────────────────────────────────────────────────────────────────────────

class TestAsyncUpdate:

    def test_returns_started_thread(self):
        shap_rows = _make_shap_rows(5)
        pa     = _pa(shap_fn=_mock_shap_fn(shap_rows))
        df     = pd.DataFrame(np.zeros((5, 2)), columns=["a", "b"])
        thread = pa.run_async_update(MagicMock(), df)
        assert isinstance(thread, threading.Thread)
        thread.join(timeout=5.0)
        assert not thread.is_alive()

    def test_async_populates_window(self):
        shap_rows = _make_shap_rows(8, ["feat_x", "feat_y"])
        pa     = _pa(shap_fn=_mock_shap_fn(shap_rows))
        df     = pd.DataFrame(np.zeros((8, 2)), columns=["feat_x", "feat_y"])
        thread = pa.run_async_update(MagicMock(), df)
        thread.join(timeout=5.0)
        assert pa.window_size() == 8

    def test_async_caches_summary(self):
        shap_rows = _make_shap_rows(5)
        pa     = _pa(shap_fn=_mock_shap_fn(shap_rows))
        df     = pd.DataFrame(np.zeros((5, 2)), columns=["feat_a", "feat_b"])
        thread = pa.run_async_update(MagicMock(), df)
        thread.join(timeout=5.0)
        cached = pa.get_last_summary()
        assert cached is not None
        assert cached.n_records == 5

    def test_async_caps_at_window_size(self):
        cfg       = AttributionConfig(window_size=10)
        shap_rows = _make_shap_rows(20)
        pa        = _pa(config=cfg, shap_fn=_mock_shap_fn(shap_rows))
        df        = pd.DataFrame(np.zeros((20, 3)), columns=["a", "b", "c"])
        thread    = pa.run_async_update(MagicMock(), df)
        thread.join(timeout=5.0)
        # _mock_shap_fn returns shap_rows[:len(df)]=shap_rows[:20]
        # But window maxlen=10, so only last 10 kept
        assert pa.window_size() == 10

    def test_async_daemon_thread(self):
        pa     = _pa(shap_fn=_mock_shap_fn([]))
        thread = pa.run_async_update(MagicMock(), pd.DataFrame())
        assert thread.daemon is True
        thread.join(timeout=3.0)

    def test_async_shap_fn_called_with_subset(self):
        """_shap_fn wird mit den letzten window_size Zeilen aufgerufen."""
        called_with: list[pd.DataFrame] = []

        def capturing_fn(model, df):
            called_with.append(df.copy())
            return []

        cfg = AttributionConfig(window_size=5)
        pa  = _pa(config=cfg, shap_fn=capturing_fn)
        df  = pd.DataFrame(np.zeros((10, 2)), columns=["a", "b"])
        thread = pa.run_async_update(MagicMock(), df)
        thread.join(timeout=5.0)
        assert len(called_with) == 1
        assert len(called_with[0]) == 5   # nur letzte 5 Zeilen

    def test_async_error_does_not_propagate(self):
        """Exception im Worker-Thread laeuft nicht im Haupt-Thread hoch."""
        def failing_fn(model, df):
            raise RuntimeError("SHAP-Fehler")

        pa     = _pa(shap_fn=failing_fn)
        df     = pd.DataFrame([[1, 2]], columns=["a", "b"])
        thread = pa.run_async_update(MagicMock(), df)
        thread.join(timeout=5.0)
        # Kein Exception, Window bleibt leer
        assert pa.window_size() == 0

    def test_thread_name(self):
        pa     = _pa(shap_fn=_mock_shap_fn([]))
        thread = pa.run_async_update(MagicMock(), pd.DataFrame())
        assert "PerformanceAttribution" in thread.name
        thread.join(timeout=3.0)


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: extract_training_importance
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractTrainingImportance:

    def test_returns_dict(self):
        shap_rows = _make_shap_rows(10, ["a", "b", "c"])

        def fn(model, df):
            return shap_rows

        result = extract_training_importance(MagicMock(), pd.DataFrame(), shap_fn=fn)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"a", "b", "c"}

    def test_values_are_mean_abs(self):
        # Alle Eintraege haben feat=0.4 -> mean_abs = 0.4
        rows = [{"feat": 0.4}] * 5

        def fn(model, df):
            return rows

        result = extract_training_importance(MagicMock(), pd.DataFrame(), shap_fn=fn)
        assert abs(result["feat"] - 0.4) < 1e-9

    def test_uses_default_shap_fn_when_none(self):
        """Ohne shap_fn-Argument wird _default_shap_fn verwendet (nur Smoke-Test)."""
        # Wir mocken _default_shap_fn weg um keinen echten LightGBM-Model zu brauchen
        rows = [{"f": 1.0}] * 3
        with patch(
            "src.models.performance_attribution._default_shap_fn",
            side_effect=lambda m, df: rows,
        ):
            result = extract_training_importance(MagicMock(), pd.DataFrame())
        assert "f" in result

    def test_empty_rows_returns_empty_dict(self):
        def fn(model, df):
            return []

        result = extract_training_importance(MagicMock(), pd.DataFrame(), shap_fn=fn)
        assert result == {}


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: Thread-Sicherheit
# ─────────────────────────────────────────────────────────────────────────────

class TestThreadSafety:

    def test_concurrent_record_shap(self):
        """Gleichzeitiges Schreiben aus mehreren Threads darf kein Race-Condition erzeugen."""
        pa     = _pa()
        errors: list[Exception] = []

        def writer():
            try:
                for _ in range(50):
                    pa.record_shap({"a": 1.0, "b": 2.0})
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=writer) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert pa.window_size() <= 100  # maxlen haelt

    def test_concurrent_compute_and_record(self):
        """compute_attribution() waehrend record_shap() darf nicht crashen."""
        pa     = _pa()
        errors: list[Exception] = []

        def record():
            try:
                for _ in range(30):
                    pa.record_shap({"x": 0.5})
            except Exception as exc:
                errors.append(exc)

        def compute():
            try:
                for _ in range(30):
                    pa.compute_attribution()
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=record)
        t2 = threading.Thread(target=compute)
        t1.start(); t2.start()
        t1.join();  t2.join()

        assert errors == []


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: _default_shap_fn (Integration, erfordert LightGBM + SHAP)
# ─────────────────────────────────────────────────────────────────────────────

class TestDefaultShapFn:
    """Integrations-Tests mit echtem LightGBM und SHAP."""

    @pytest.fixture(scope="class")
    def trained_model(self):
        """Gibt ein trainiertes SignalModel zurueck."""
        from src.models.signal_model import SignalModel
        import numpy as np, pandas as pd

        rng = np.random.default_rng(0)
        n, f = 80, 4
        X = pd.DataFrame(rng.standard_normal((n, f)), columns=[f"f{i}" for i in range(f)])
        y = pd.Series(rng.choice([-1, 0, 1], n))
        m = SignalModel(lgbm_params={"n_estimators": 10, "verbose": -1})
        m.train(X, y)
        return m, X

    def test_returns_list_of_dicts(self, trained_model):
        model, X = trained_model
        rows = _default_shap_fn(model, X.iloc[:3])
        assert isinstance(rows, list)
        assert len(rows) == 3
        for r in rows:
            assert isinstance(r, dict)

    def test_all_values_nonnegative(self, trained_model):
        """_default_shap_fn gibt absolute Werte zurueck."""
        model, X = trained_model
        rows = _default_shap_fn(model, X.iloc[:5])
        for r in rows:
            for v in r.values():
                assert v >= 0.0

    def test_feature_names_match_model(self, trained_model):
        model, X = trained_model
        rows = _default_shap_fn(model, X.iloc[:2])
        for r in rows:
            assert set(r.keys()) == set(model._feature_names)

    def test_extract_training_importance_with_real_model(self, trained_model):
        model, X = trained_model
        importance = extract_training_importance(model, X)
        assert len(importance) == len(model._feature_names)
        for v in importance.values():
            assert v >= 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: GUI-Hilfs-Funktionen (pure Python)
# ─────────────────────────────────────────────────────────────────────────────

class TestFormatImportanceRow:

    def test_rank_format(self):
        rank, _, _ = format_importance_row(1, "feat", 0.5)
        assert rank == "#1"

    def test_name_preserved(self):
        _, name, _ = format_importance_row(3, "atr_feature", 0.123)
        assert name == "atr_feature"

    def test_value_four_decimals(self):
        _, _, val = format_importance_row(1, "f", 0.042345)
        assert val == "0.0423"

    def test_value_zero(self):
        _, _, val = format_importance_row(1, "f", 0.0)
        assert val == "0.0000"

    def test_rank_five(self):
        rank, _, _ = format_importance_row(5, "f", 0.1)
        assert rank == "#5"


class TestAttributionSnapshot:

    def test_fields_stored(self):
        snap = AttributionSnapshot(
            top_features=[("feat_a", 0.5), ("feat_b", 0.3)],
            n_records=50,
            drift_warning="Re-Training empfohlen!",
            retrain_needed=True,
        )
        assert snap.n_records == 50
        assert snap.retrain_needed is True
        assert snap.drift_warning is not None

    def test_no_drift(self):
        snap = AttributionSnapshot(
            top_features=[("f", 0.1)],
            n_records=20,
            drift_warning=None,
            retrain_needed=False,
        )
        assert snap.drift_warning is None
        assert snap.retrain_needed is False

    def test_default_label(self):
        snap = AttributionSnapshot(
            top_features=[], n_records=0,
            drift_warning=None, retrain_needed=False,
        )
        assert snap.label == "live"

    def test_custom_label(self):
        snap = AttributionSnapshot(
            top_features=[], n_records=0,
            drift_warning=None, retrain_needed=False,
            label="shadow_v2",
        )
        assert snap.label == "shadow_v2"
