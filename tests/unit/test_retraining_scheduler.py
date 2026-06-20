"""
tests/unit/test_retraining_scheduler.py
Unit-Tests fuer RetrainingScheduler.

Abgedeckt:
  RetrainingConfig – Defaultwerte und Instanziierung
  RetrainingScheduler.is_due – Trigger-Logik (nie gelaufen, Intervall, Stunde)
  RetrainingScheduler._evaluate – Sharpe/Win-Rate-Berechnung
  RetrainingScheduler._should_promote – Vergleichslogik, Fallback
  RetrainingScheduler.run – End-to-End mit Mocks (deployed / rejected)
  RetrainingScheduler.get_active_model_path – Registry-Lese
  Alert- und Audit-Callbacks werden aufgerufen
  Fallback: bei Ablehnung bleibt altes Modell als aktiv
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.models.retraining_scheduler import (
    ModelMetrics,
    RetrainingConfig,
    RetrainingResult,
    RetrainingScheduler,
    _REGISTRY_FILENAME,
    _STRUCT_COLS,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Hilfsfunktionen
# ─────────────────────────────────────────────────────────────────────────────

def _utc(year=2024, month=1, day=15, hour=2):
    return datetime(year, month, day, hour, 0, 0, tzinfo=timezone.utc)


def _make_features(n=200, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "close":     rng.uniform(1.08, 1.10, n),
            "high":      rng.uniform(1.09, 1.11, n),
            "low":       rng.uniform(1.07, 1.09, n),
            "ema_9":     rng.uniform(1.08, 1.10, n),
            "rsi_14":    rng.uniform(30, 70, n),
            "atr_14":    rng.uniform(0.0005, 0.0015, n),
        },
        index=idx,
    )


def _make_labels(n=200, seed=99) -> pd.Series:
    rng = np.random.default_rng(seed)
    vals = rng.choice([-1, 0, 1], size=n, p=[0.4, 0.1, 0.5])
    return pd.Series(vals.astype(int), name="label")


def _make_config(tmp_path: Path) -> RetrainingConfig:
    return RetrainingConfig(
        models_dir=tmp_path,
        interval_days=7,
        preferred_hour_utc=2,
        min_sharpe_delta=-0.1,
        is_ratio=0.70,
    )


def _scheduler(tmp_path, now=None, alert_fn=None, audit_fn=None):
    cfg    = _make_config(tmp_path)
    now_fn = (lambda: now) if now is not None else None
    return RetrainingScheduler(
        config=cfg,
        alert_fn=alert_fn,
        audit_fn=audit_fn,
        _now_fn=now_fn,
    )


# Fake SignalModel fuer schnelle Tests (kein echtes Training)
def _fake_signal_model(proba_value=0.5):
    """Erzeugt ein SignalModel-Mock das immer gleiche Proba zurueckgibt."""
    mock_inner = MagicMock()
    n_classes  = 3
    mock_inner.predict_proba.side_effect = lambda X: np.full(
        (len(X), n_classes), proba_value / n_classes
    )
    mock_sm = MagicMock()
    mock_sm._model   = mock_inner
    mock_sm._feature_names = ["ema_9", "rsi_14", "atr_14"]
    return mock_sm


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: RetrainingConfig
# ─────────────────────────────────────────────────────────────────────────────

class TestRetrainingConfig:
    def test_defaults(self):
        cfg = RetrainingConfig()
        assert cfg.interval_days    == 7
        assert cfg.preferred_hour_utc == 2
        assert cfg.min_sharpe_delta == -0.1
        assert cfg.is_ratio         == 0.70
        assert cfg.confidence_threshold == 0.55

    def test_custom_values(self, tmp_path):
        cfg = RetrainingConfig(
            models_dir=tmp_path,
            interval_days=14,
            preferred_hour_utc=3,
            min_sharpe_delta=0.05,
        )
        assert cfg.interval_days      == 14
        assert cfg.preferred_hour_utc == 3
        assert cfg.min_sharpe_delta   == 0.05
        assert cfg.models_dir         == tmp_path


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: is_due
# ─────────────────────────────────────────────────────────────────────────────

class TestIsDue:

    def test_due_when_never_run(self, tmp_path):
        s = _scheduler(tmp_path, now=_utc())
        assert s.is_due(last_run=None) is True

    def test_not_due_within_interval(self, tmp_path):
        now      = _utc(day=15, hour=2)
        last_run = _utc(day=10, hour=2)   # 5 Tage – unter 7
        s = _scheduler(tmp_path, now=now)
        assert s.is_due(last_run=last_run) is False

    def test_due_after_interval_at_preferred_hour(self, tmp_path):
        now      = _utc(day=23, hour=2)   # 8 Tage nach last_run, bevorzugte Stunde
        last_run = _utc(day=15, hour=2)
        s = _scheduler(tmp_path, now=now)
        assert s.is_due(last_run=last_run) is True

    def test_not_due_after_interval_wrong_hour(self, tmp_path):
        now      = _utc(day=23, hour=10)  # 8 Tage – aber falsche Stunde
        last_run = _utc(day=15, hour=2)
        s = _scheduler(tmp_path, now=now)
        assert s.is_due(last_run=last_run) is False

    def test_exactly_at_interval_boundary(self, tmp_path):
        now      = _utc(day=22, hour=2)   # genau 7 Tage
        last_run = _utc(day=15, hour=2)
        s = _scheduler(tmp_path, now=now)
        assert s.is_due(last_run=last_run) is True

    def test_naive_last_run_treated_as_utc(self, tmp_path):
        now      = _utc(day=23, hour=2)
        last_run = datetime(2024, 1, 15, 2, 0, 0)  # naive
        s = _scheduler(tmp_path, now=now)
        assert s.is_due(last_run=last_run) is True

    def test_custom_interval(self, tmp_path):
        cfg = RetrainingConfig(models_dir=tmp_path, interval_days=30, preferred_hour_utc=2)
        s   = RetrainingScheduler(cfg, _now_fn=lambda: _utc(month=2, day=5, hour=2))
        last_run = _utc(month=1, day=15)   # 21 Tage – unter 30
        assert s.is_due(last_run=last_run) is False

    def test_custom_preferred_hour(self, tmp_path):
        cfg = RetrainingConfig(models_dir=tmp_path, interval_days=7, preferred_hour_utc=3)
        s   = RetrainingScheduler(cfg, _now_fn=lambda: _utc(day=23, hour=3))
        assert s.is_due(last_run=_utc(day=15, hour=3)) is True

    def test_not_due_at_wrong_custom_hour(self, tmp_path):
        cfg = RetrainingConfig(models_dir=tmp_path, interval_days=7, preferred_hour_utc=3)
        s   = RetrainingScheduler(cfg, _now_fn=lambda: _utc(day=23, hour=2))
        assert s.is_due(last_run=_utc(day=15, hour=3)) is False


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: _evaluate
# ─────────────────────────────────────────────────────────────────────────────

class TestEvaluate:

    def _sched(self, tmp_path):
        return _scheduler(tmp_path)

    def test_returns_model_metrics(self, tmp_path):
        s   = self._sched(tmp_path)
        m   = _fake_signal_model()
        X   = np.random.randn(50, 3)
        y   = np.array([1, -1, 0] * 16 + [1, -1])
        res = s._evaluate(m, X, y)
        assert isinstance(res, ModelMetrics)
        assert isinstance(res.sharpe, float)
        assert isinstance(res.win_rate, float)
        assert res.n_oos_rows == 50

    def test_win_rate_between_0_and_1(self, tmp_path):
        s = self._sched(tmp_path)
        m = _fake_signal_model()
        X = np.random.randn(100, 3)
        y = np.random.choice([-1, 0, 1], size=100)
        r = s._evaluate(m, X, y)
        assert 0.0 <= r.win_rate <= 1.0

    def test_all_neutral_returns_zero_metrics(self, tmp_path):
        s = self._sched(tmp_path)
        # Modell das immer neutral voraussagt (argmax=1)
        mock_inner = MagicMock()
        mock_inner.predict_proba.return_value = np.array([[0.1, 0.8, 0.1]] * 20)
        mock_sm = MagicMock()
        mock_sm._model = mock_inner
        y = np.ones(20, dtype=int)
        r = s._evaluate(mock_sm, np.zeros((20, 3)), y)
        assert r.sharpe   == 0.0
        assert r.win_rate == 0.0

    def test_perfect_model_high_win_rate(self, tmp_path):
        s = self._sched(tmp_path)
        # Modell das immer long voraussagt (class=2) auf Daten wo Label=1
        mock_inner = MagicMock()
        mock_inner.predict_proba.return_value = np.array([[0.1, 0.1, 0.8]] * 30)
        mock_sm = MagicMock()
        mock_sm._model = mock_inner
        y = np.ones(30, dtype=int)   # Label 1 = Long = class 2 -> korrekt
        r = s._evaluate(mock_sm, np.zeros((30, 3)), y)
        assert r.win_rate == 1.0

    def test_none_model_returns_zero(self, tmp_path):
        s      = self._sched(tmp_path)
        mock_m = MagicMock()
        mock_m._model = None
        y = np.array([1, -1, 1])
        r = s._evaluate(mock_m, np.zeros((3, 3)), y)
        assert r.sharpe   == 0.0
        assert r.win_rate == 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: _should_promote
# ─────────────────────────────────────────────────────────────────────────────

class TestShouldPromote:

    def _sched(self, tmp_path, min_delta=-0.1):
        cfg = RetrainingConfig(models_dir=tmp_path, min_sharpe_delta=min_delta)
        return RetrainingScheduler(cfg)

    def test_promoted_when_no_old_model(self, tmp_path):
        s = self._sched(tmp_path)
        promoted, reason = s._should_promote(
            ModelMetrics(1.0, 0.6, 100), old_m=None
        )
        assert promoted is True
        assert "Kein aktives Modell" in reason

    def test_promoted_when_new_better(self, tmp_path):
        s = self._sched(tmp_path)
        promoted, _ = s._should_promote(
            ModelMetrics(1.5, 0.65, 100),
            ModelMetrics(1.0, 0.60, 100),
        )
        assert promoted is True

    def test_promoted_when_delta_exactly_at_threshold(self, tmp_path):
        s = self._sched(tmp_path, min_delta=-0.1)
        # delta = 0.9 - 1.0 = -0.1, genau am Schwellwert -> promoted
        promoted, _ = s._should_promote(
            ModelMetrics(0.9, 0.58, 100),
            ModelMetrics(1.0, 0.60, 100),
        )
        assert promoted is True

    def test_rejected_when_new_much_worse(self, tmp_path):
        s = self._sched(tmp_path, min_delta=-0.1)
        promoted, reason = s._should_promote(
            ModelMetrics(0.5, 0.50, 100),
            ModelMetrics(1.0, 0.60, 100),
        )
        assert promoted is False
        assert "Abgelehnt" in reason

    def test_reason_contains_delta(self, tmp_path):
        s = self._sched(tmp_path)
        _, reason = s._should_promote(
            ModelMetrics(1.5, 0.65, 100),
            ModelMetrics(1.0, 0.60, 100),
        )
        assert "Delta" in reason or "delta" in reason.lower() or "Sharpe" in reason

    def test_strict_threshold_zero(self, tmp_path):
        s = self._sched(tmp_path, min_delta=0.0)
        promoted, _ = s._should_promote(
            ModelMetrics(0.99, 0.59, 100),
            ModelMetrics(1.0, 0.60, 100),
        )
        assert promoted is False

    def test_strict_threshold_positive(self, tmp_path):
        s = self._sched(tmp_path, min_delta=0.5)
        promoted, _ = s._should_promote(
            ModelMetrics(1.3, 0.62, 100),
            ModelMetrics(1.0, 0.60, 100),
        )
        # delta=0.3 < 0.5 -> rejected
        assert promoted is False


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: run() – End-zu-End mit gemocktem SignalModel
# ─────────────────────────────────────────────────────────────────────────────

class TestRun:

    def _patched_run(self, tmp_path, now, promoted_metrics, old_metrics=None,
                     alert_fn=None, audit_fn=None):
        """
        Fuehrt scheduler.run() mit gemocktem SignalModel aus.
        promoted_metrics: ModelMetrics fuer neues Modell
        old_metrics:      ModelMetrics fuer altes Modell (None = kein Vorgaenger)
        """
        features_df = _make_features(n=100)
        labels      = _make_labels(n=100)
        sched       = _scheduler(tmp_path, now=now,
                                  alert_fn=alert_fn, audit_fn=audit_fn)

        # Neues Modell
        new_sm = MagicMock()
        new_sm._model = MagicMock()
        new_sm._feature_names = ["ema_9", "rsi_14", "atr_14"]
        new_sm.save = MagicMock()

        old_sm = None
        if old_metrics is not None:
            old_sm = MagicMock()
            old_sm._model = MagicMock()
            old_sm._feature_names = ["ema_9", "rsi_14", "atr_14"]
            (tmp_path / "signal_model_v1_20231201.joblib").touch()

        date_str = now.strftime("%Y%m%d")

        with (
            patch("src.models.retraining_scheduler.SignalModel") as MockSM,
            patch("src.models.retraining_scheduler.build_save_path") as mock_bsp,
            patch.object(sched, "_evaluate") as mock_eval,
        ):
            mock_bsp.return_value = Path(f"signal_model_v1_{date_str}.joblib")
            MockSM.return_value   = new_sm
            if old_sm is not None:
                MockSM.load.return_value = old_sm

                def _side_effect(m, X, y):
                    return promoted_metrics if m is new_sm else old_metrics

                mock_eval.side_effect = _side_effect
            else:
                mock_eval.return_value = promoted_metrics

            result = sched.run("EURUSD", "H1", features_df, labels)

        return result

    def test_run_returns_result(self, tmp_path):
        result = self._patched_run(
            tmp_path, now=_utc(),
            promoted_metrics=ModelMetrics(1.5, 0.65, 30),
        )
        assert isinstance(result, RetrainingResult)

    def test_promoted_when_no_old_model(self, tmp_path):
        result = self._patched_run(
            tmp_path, now=_utc(),
            promoted_metrics=ModelMetrics(1.5, 0.65, 30),
        )
        assert result.promoted is True

    def test_registry_written_on_promotion(self, tmp_path):
        self._patched_run(
            tmp_path, now=_utc(),
            promoted_metrics=ModelMetrics(1.5, 0.65, 30),
        )
        registry_file = tmp_path / _REGISTRY_FILENAME
        assert registry_file.exists()
        data = json.loads(registry_file.read_text())
        assert "active" in data
        assert data["active"] is not None

    def test_registry_not_written_on_rejection(self, tmp_path):
        # Altes Modell besser: neue wird rejected
        self._patched_run(
            tmp_path, now=_utc(),
            promoted_metrics=ModelMetrics(0.3, 0.50, 30),
            old_metrics=ModelMetrics(1.0, 0.60, 30),
        )
        registry_file = tmp_path / _REGISTRY_FILENAME
        # Registry existiert nicht oder active zeigt noch auf altes Modell
        if registry_file.exists():
            data = json.loads(registry_file.read_text())
            assert data.get("active") != f"signal_model_v1_{_utc().strftime('%Y%m%d')}.joblib"

    def test_result_contains_symbol_timeframe(self, tmp_path):
        result = self._patched_run(
            tmp_path, now=_utc(),
            promoted_metrics=ModelMetrics(1.5, 0.65, 30),
        )
        assert result.symbol    == "EURUSD"
        assert result.timeframe == "H1"

    def test_result_metrics_set(self, tmp_path):
        m = ModelMetrics(1.23, 0.61, 30)
        result = self._patched_run(tmp_path, now=_utc(), promoted_metrics=m)
        assert result.new_metrics.sharpe   == pytest.approx(1.23)
        assert result.new_metrics.win_rate == pytest.approx(0.61)

    def test_alert_called_on_promotion(self, tmp_path):
        calls: list[str] = []
        self._patched_run(
            tmp_path, now=_utc(),
            promoted_metrics=ModelMetrics(1.5, 0.65, 30),
            alert_fn=calls.append,
        )
        assert len(calls) == 1
        assert "EURUSD" in calls[0]

    def test_alert_called_on_rejection(self, tmp_path):
        calls: list[str] = []
        self._patched_run(
            tmp_path, now=_utc(),
            promoted_metrics=ModelMetrics(0.3, 0.50, 30),
            old_metrics=ModelMetrics(1.0, 0.60, 30),
            alert_fn=calls.append,
        )
        assert len(calls) == 1

    def test_audit_called_with_event_type(self, tmp_path):
        events: list[tuple] = []
        self._patched_run(
            tmp_path, now=_utc(),
            promoted_metrics=ModelMetrics(1.5, 0.65, 30),
            audit_fn=lambda et, d: events.append((et, d)),
        )
        assert len(events) == 1
        assert events[0][0] == "MODEL_RETRAIN_DEPLOYED"

    def test_audit_rejected_event_type(self, tmp_path):
        events: list[tuple] = []
        self._patched_run(
            tmp_path, now=_utc(),
            promoted_metrics=ModelMetrics(0.3, 0.50, 30),
            old_metrics=ModelMetrics(1.0, 0.60, 30),
            audit_fn=lambda et, d: events.append((et, d)),
        )
        assert events[0][0] == "MODEL_RETRAIN_REJECTED"

    def test_audit_details_contain_metrics(self, tmp_path):
        events: list[tuple] = []
        self._patched_run(
            tmp_path, now=_utc(),
            promoted_metrics=ModelMetrics(1.5, 0.65, 30),
            audit_fn=lambda et, d: events.append((et, d)),
        )
        details = events[0][1]
        assert "new_sharpe"   in details
        assert "new_win_rate" in details
        assert "symbol"       in details

    def test_alert_error_does_not_raise(self, tmp_path):
        def _bad_alert(msg):
            raise RuntimeError("Telegram down")

        result = self._patched_run(
            tmp_path, now=_utc(),
            promoted_metrics=ModelMetrics(1.5, 0.65, 30),
            alert_fn=_bad_alert,
        )
        # Kein Exception; Ergebnis trotzdem zurueckgegeben
        assert isinstance(result, RetrainingResult)

    def test_fallback_model_preserved_on_rejection(self, tmp_path):
        """Altes Modell bleibt nach Ablehnung erhalten."""
        old_file = tmp_path / "signal_model_v1_20231201.joblib"
        old_file.touch()

        self._patched_run(
            tmp_path, now=_utc(),
            promoted_metrics=ModelMetrics(0.3, 0.50, 30),
            old_metrics=ModelMetrics(1.0, 0.60, 30),
        )
        assert old_file.exists()

    def test_timestamp_in_result(self, tmp_path):
        now = _utc(day=20, hour=2)
        result = self._patched_run(tmp_path, now=now,
                                    promoted_metrics=ModelMetrics(1.5, 0.65, 30))
        assert result.timestamp == now


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: get_active_model_path
# ─────────────────────────────────────────────────────────────────────────────

class TestGetActiveModelPath:

    def test_returns_none_when_no_models(self, tmp_path):
        s = _scheduler(tmp_path)
        assert s.get_active_model_path() is None

    def test_returns_most_recent_when_no_registry(self, tmp_path):
        (tmp_path / "signal_model_v1_20230101.joblib").touch()
        (tmp_path / "signal_model_v1_20240101.joblib").touch()
        s    = _scheduler(tmp_path)
        path = s.get_active_model_path()
        assert path is not None
        assert "20240101" in path.name

    def test_returns_registry_active(self, tmp_path):
        (tmp_path / "signal_model_v1_20230101.joblib").touch()
        (tmp_path / "signal_model_v1_20240101.joblib").touch()
        registry = {"active": "signal_model_v1_20230101.joblib"}
        (tmp_path / _REGISTRY_FILENAME).write_text(
            json.dumps(registry), encoding="utf-8"
        )
        s    = _scheduler(tmp_path)
        path = s.get_active_model_path()
        assert "20230101" in path.name

    def test_ignores_missing_registry_file(self, tmp_path):
        (tmp_path / "signal_model_v1_20240101.joblib").touch()
        registry = {"active": "signal_model_v1_NONEXISTENT.joblib"}
        (tmp_path / _REGISTRY_FILENAME).write_text(
            json.dumps(registry), encoding="utf-8"
        )
        s    = _scheduler(tmp_path)
        path = s.get_active_model_path()
        # Faellt auf neueste vorhandene Datei zurueck
        assert path is not None


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: _STRUCT_COLS (Konsistenz mit train_model.py)
# ─────────────────────────────────────────────────────────────────────────────

class TestStructCols:
    def test_contains_price_cols(self):
        assert "close" in _STRUCT_COLS
        assert "high"  in _STRUCT_COLS
        assert "low"   in _STRUCT_COLS

    def test_contains_metadata_cols(self):
        assert "timestamp" in _STRUCT_COLS
        assert "label"     in _STRUCT_COLS

    def test_get_feature_cols_excludes_struct(self):
        df = _make_features(n=10)
        cols = RetrainingScheduler._get_feature_cols(df)
        for c in _STRUCT_COLS:
            assert c not in cols

    def test_get_feature_cols_includes_indicators(self):
        df   = _make_features(n=10)
        cols = RetrainingScheduler._get_feature_cols(df)
        assert "ema_9"  in cols
        assert "rsi_14" in cols


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: Alert-Nachrichtenformat
# ─────────────────────────────────────────────────────────────────────────────

class TestAlertFormat:

    def _make_result(self, promoted=True) -> RetrainingResult:
        return RetrainingResult(
            timestamp=_utc(),
            symbol="EURUSD",
            timeframe="H1",
            new_model_path=Path("models/signal_model_v1_20240115.joblib"),
            old_model_path=Path("models/signal_model_v1_20240101.joblib"),
            new_metrics=ModelMetrics(sharpe=1.5, win_rate=0.65, n_oos_rows=100),
            old_metrics=ModelMetrics(sharpe=1.0, win_rate=0.60, n_oos_rows=100),
            promoted=promoted,
            reason="Deployed" if promoted else "Abgelehnt",
        )

    def test_alert_contains_symbol(self, tmp_path):
        messages: list[str] = []
        s = _scheduler(tmp_path, alert_fn=messages.append)
        s._send_alert(self._make_result())
        assert "EURUSD" in messages[0]

    def test_alert_contains_sharpe_values(self, tmp_path):
        messages: list[str] = []
        s = _scheduler(tmp_path, alert_fn=messages.append)
        s._send_alert(self._make_result())
        assert "1.5" in messages[0] or "1.500" in messages[0]

    def test_alert_contains_win_rate(self, tmp_path):
        messages: list[str] = []
        s = _scheduler(tmp_path, alert_fn=messages.append)
        s._send_alert(self._make_result())
        assert "65" in messages[0] or "0.65" in messages[0]

    def test_alert_no_fn_no_error(self, tmp_path):
        s = _scheduler(tmp_path, alert_fn=None)
        s._send_alert(self._make_result())   # kein Fehler

    def test_alert_contains_old_metrics(self, tmp_path):
        messages: list[str] = []
        s = _scheduler(tmp_path, alert_fn=messages.append)
        s._send_alert(self._make_result())
        assert "1.0" in messages[0] or "1.000" in messages[0]

    def test_alert_dash_when_no_old_model(self, tmp_path):
        messages: list[str] = []
        s = _scheduler(tmp_path, alert_fn=messages.append)
        result = self._make_result()
        result.old_model_path = None
        result.old_metrics    = None
        s._send_alert(result)
        assert "–" in messages[0]
