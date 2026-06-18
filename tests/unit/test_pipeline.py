"""
Unit-Tests fuer DataPipeline.
Router, Validator und FeatureBuilder werden vollstaendig gemockt.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.data.pipeline import DataPipeline, PipelineError


# ---------------------------------------------------------------------------
# Test-Doubles
# ---------------------------------------------------------------------------

@dataclass
class _FakeReport:
    symbol: str = "EURUSD"
    timeframe: str = "H1"
    total_candles: int = 100
    missing_candles: int = 0
    missing_pct: float = 0.0
    duplicates_removed: int = 0
    ohlc_violations: int = 0
    outliers_flagged: int = 0
    nan_rows_removed: int = 0
    quality_score: float = 1.0
    is_usable: bool = True
    warnings: list = field(default_factory=list)
    errors: list = field(default_factory=list)


def _make_ohlcv_df(n: int = 250) -> pd.DataFrame:
    idx = pd.date_range(
        end=datetime.now(timezone.utc), periods=n, freq="h", tz="UTC"
    )
    return pd.DataFrame({
        "open":   np.linspace(1.10, 1.11, n),
        "high":   np.linspace(1.101, 1.111, n),
        "low":    np.linspace(1.099, 1.109, n),
        "close":  np.linspace(1.1005, 1.1105, n),
        "volume": np.random.randint(100, 1000, n),
    }, index=idx)


def _make_features_df(n: int = 50) -> pd.DataFrame:
    idx = pd.date_range(end=datetime.now(timezone.utc), periods=n, freq="h", tz="UTC")
    return pd.DataFrame({"ema_20": np.linspace(1.1, 1.2, n)}, index=idx)


@pytest.fixture
def mock_router():
    router = MagicMock()
    router.get_ohlcv.return_value = _make_ohlcv_df()
    router.get_ohlcv_count.return_value = _make_ohlcv_df()
    return router


@pytest.fixture
def mock_validator():
    validator = MagicMock()
    validator.validate.return_value = (_FakeReport(), _make_ohlcv_df())
    return validator


@pytest.fixture
def mock_feature_builder():
    builder = MagicMock()
    builder.build.return_value = _make_features_df()
    return builder


@pytest.fixture
def pipeline(tmp_path, mock_router, mock_validator, mock_feature_builder) -> DataPipeline:
    return DataPipeline(
        router=mock_router,
        validator=mock_validator,
        feature_builder=mock_feature_builder,
        features_dir=str(tmp_path / "features"),
        reports_dir=str(tmp_path / "reports"),
        live_interval=99999,
    )


def _dt(offset_days: int = 0) -> datetime:
    return datetime.now(timezone.utc) - timedelta(days=offset_days)


# ---------------------------------------------------------------------------
# Tests: run_batch
# ---------------------------------------------------------------------------

class TestRunBatch:

    def test_returns_output_path(self, pipeline):
        result = pipeline.run_batch("EURUSD", "H1", _dt(7), _dt(0))
        assert "output_path" in result
        assert Path(result["output_path"]).exists()

    def test_calls_router_validator_builder_in_order(
        self, pipeline, mock_router, mock_validator, mock_feature_builder
    ):
        pipeline.run_batch("EURUSD", "H1", _dt(7), _dt(0))
        mock_router.get_ohlcv.assert_called_once()
        mock_validator.validate.assert_called_once()
        mock_feature_builder.build.assert_called_once()

    def test_saves_quality_report(self, pipeline):
        result = pipeline.run_batch("EURUSD", "H1", _dt(7), _dt(0))
        report_path = Path(result["report_path"])
        assert report_path.exists()
        with open(report_path) as f:
            data = json.load(f)
        assert data["symbol"] == "EURUSD"

    def test_raises_pipeline_error_on_quality_error(self, pipeline, mock_validator):
        from src.data.validator import DataQualityError
        mock_validator.validate.side_effect = DataQualityError("zu viele fehlende Daten")
        with pytest.raises(PipelineError):
            pipeline.run_batch("EURUSD", "H1", _dt(7), _dt(0))

    def test_idempotent_skips_unchanged(self, pipeline, mock_router):
        # Feste Zeitstempel: _dt(7)/_dt(0) wuerden bei zwei Aufrufen
        # unterschiedliche Mikrosekunden liefern -> anderer Hash -> falscher Test
        start, end = _dt(7), _dt(0)
        pipeline.run_batch("EURUSD", "H1", start, end)
        mock_router.get_ohlcv.reset_mock()

        result = pipeline.run_batch("EURUSD", "H1", start, end)
        assert result.get("skipped") is True
        mock_router.get_ohlcv.assert_not_called()

    def test_force_refetch_ignores_hash(self, pipeline, mock_router):
        start, end = _dt(7), _dt(0)
        pipeline.run_batch("EURUSD", "H1", start, end)
        mock_router.get_ohlcv.reset_mock()

        result = pipeline.run_batch("EURUSD", "H1", start, end, force_refetch=True)
        assert result.get("skipped") is False
        mock_router.get_ohlcv.assert_called_once()

    def test_different_date_range_not_skipped(self, pipeline, mock_router):
        pipeline.run_batch("EURUSD", "H1", _dt(7), _dt(0))
        mock_router.get_ohlcv.reset_mock()

        # Anderer Zeitraum -> anderer Hash -> kein Skip
        result = pipeline.run_batch("EURUSD", "H1", _dt(30), _dt(10))
        mock_router.get_ohlcv.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: run_batch_multi
# ---------------------------------------------------------------------------

class TestRunBatchMulti:

    def test_processes_all_symbols(self, pipeline):
        results = pipeline.run_batch_multi(
            ["EURUSD", "GBPUSD"], "H1", _dt(7), _dt(0), show_progress=False
        )
        assert set(results.keys()) == {"EURUSD", "GBPUSD"}

    def test_continues_after_single_failure(self, pipeline, mock_validator):
        from src.data.validator import DataQualityError

        call_count = {"n": 0}

        def side_effect(df, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise DataQualityError("Fehler bei erstem Symbol")
            return (_FakeReport(), _make_ohlcv_df())

        mock_validator.validate.side_effect = side_effect

        results = pipeline.run_batch_multi(
            ["EURUSD", "GBPUSD"], "H1", _dt(7), _dt(0), show_progress=False
        )
        assert "error" in results["EURUSD"]
        assert "error" not in results["GBPUSD"]


# ---------------------------------------------------------------------------
# Tests: Live-Update (einzelner Zyklus, kein echter Thread)
# ---------------------------------------------------------------------------

class TestLiveUpdate:

    def test_live_update_single_cycle(self, pipeline, mock_router, mock_feature_builder):
        result = pipeline._live_update("EURUSD", "M1", lookback_candles=300)
        mock_router.get_ohlcv_count.assert_called_once_with("EURUSD", "M1", count=300)
        assert "output_path" in result
        assert Path(result["output_path"]).exists()

    def test_start_and_stop_live(self, pipeline):
        pipeline.start_live("EURUSD", "M1")
        assert pipeline._live_thread is not None
        assert pipeline._live_thread.is_alive()
        pipeline.stop_live()
        assert not pipeline._live_thread.is_alive()


# ---------------------------------------------------------------------------
# Tests: Pfade
# ---------------------------------------------------------------------------

class TestPaths:

    def test_feature_path_format(self, pipeline):
        path = pipeline._feature_path("EURUSD", "H1", datetime(2024, 5, 1, tzinfo=timezone.utc))
        assert path.name == "EURUSD_H1_20240501.parquet"

    def test_directories_created(self, tmp_path, mock_router, mock_validator, mock_feature_builder):
        features_dir = tmp_path / "nested" / "features"
        reports_dir  = tmp_path / "nested" / "reports"
        DataPipeline(
            router=mock_router,
            validator=mock_validator,
            feature_builder=mock_feature_builder,
            features_dir=str(features_dir),
            reports_dir=str(reports_dir),
        )
        assert features_dir.exists()
        assert reports_dir.exists()
