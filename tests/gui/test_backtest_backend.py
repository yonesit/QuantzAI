"""
tests/gui/test_backtest_backend.py
Unit-Tests fuer gui/backends/backtest_backend.py.

Keine Qt-Abhaengigkeit – reine Python-Tests mit tmp_path + MagicMock.

Abgedeckt:
  BacktestSetupError
  _find_features_file
  _find_model_file
  BacktestGUIBackend.get_available_symbols
  BacktestGUIBackend._load_features (Fehler + Erfolg)
  BacktestGUIBackend._load_signal_func (Fehler + Erfolg)
  BacktestGUIBackend.run_backtest (Ende-zu-Ende mit Mocks)
  BacktestGUIBackend.export_markdown
  _result_to_markdown
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from gui.backends.backtest_backend import (
    BacktestGUIBackend,
    BacktestSetupError,
    _find_features_file,
    _find_model_file,
    _result_to_markdown,
)
from src.backtesting.vectorbt_runner import BacktestResult


# ─────────────────────────────────────────────────────────────────────────────
#  Hilfsfunktionen
# ─────────────────────────────────────────────────────────────────────────────

def _make_result() -> BacktestResult:
    idx = pd.date_range("2023-01-01", periods=10, freq="1h", tz="UTC")
    return BacktestResult(
        total_return=0.05,
        sharpe_ratio=1.2,
        sortino_ratio=1.5,
        max_drawdown=-0.08,
        profit_factor=1.8,
        win_rate=0.6,
        avg_win=50.0,
        avg_loss=-30.0,
        n_trades=20,
        equity_curve=pd.Series(np.linspace(10_000, 10_500, 10), index=idx),
    )


def _make_features_df(n: int = 200) -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "close":  np.random.uniform(1.08, 1.10, n),
            "rsi":    np.random.uniform(30, 70, n),
            "ma20":   np.random.uniform(1.08, 1.10, n),
        },
        index=idx,
    )


def _write_features(tmp_path: Path, symbol="EURUSD", tf="H1", n=200) -> Path:
    df   = _make_features_df(n)
    path = tmp_path / f"{symbol}_{tf}_20241231.parquet"
    df.to_parquet(path)
    return path


def _write_model(tmp_path: Path, version: int = 1) -> Path:
    path = tmp_path / f"signal_model_v{version}_20241231.joblib"
    path.touch()
    return path


# ─────────────────────────────────────────────────────────────────────────────
#  BacktestSetupError
# ─────────────────────────────────────────────────────────────────────────────

class TestBacktestSetupError:
    def test_is_exception(self):
        err = BacktestSetupError("Kein Modell")
        assert isinstance(err, Exception)

    def test_message_preserved(self):
        err = BacktestSetupError("Test-Meldung")
        assert "Test-Meldung" in str(err)


# ─────────────────────────────────────────────────────────────────────────────
#  Hilfsfunktionen
# ─────────────────────────────────────────────────────────────────────────────

class TestFindFeaturesFile:
    def test_returns_none_when_empty(self, tmp_path):
        assert _find_features_file(tmp_path, "EURUSD", "H1") is None

    def test_finds_file(self, tmp_path):
        _write_features(tmp_path)
        result = _find_features_file(tmp_path, "EURUSD", "H1")
        assert result is not None
        assert result.suffix == ".parquet"

    def test_case_insensitive_symbol(self, tmp_path):
        _write_features(tmp_path, symbol="EURUSD", tf="H1")
        assert _find_features_file(tmp_path, "eurusd", "h1") is not None

    def test_returns_most_recent_when_multiple(self, tmp_path):
        (tmp_path / "EURUSD_H1_20230101.parquet").touch()
        (tmp_path / "EURUSD_H1_20241231.parquet").touch()
        result = _find_features_file(tmp_path, "EURUSD", "H1")
        assert "20241231" in result.name

    def test_does_not_match_wrong_symbol(self, tmp_path):
        _write_features(tmp_path, symbol="GBPUSD", tf="H1")
        assert _find_features_file(tmp_path, "EURUSD", "H1") is None

    def test_does_not_match_wrong_timeframe(self, tmp_path):
        _write_features(tmp_path, symbol="EURUSD", tf="H4")
        assert _find_features_file(tmp_path, "EURUSD", "H1") is None


class TestFindModelFile:
    def test_returns_none_when_empty(self, tmp_path):
        assert _find_model_file(tmp_path) is None

    def test_finds_file(self, tmp_path):
        _write_model(tmp_path)
        result = _find_model_file(tmp_path)
        assert result is not None
        assert result.suffix == ".joblib"

    def test_returns_most_recent_when_multiple(self, tmp_path):
        (tmp_path / "signal_model_v1_20230101.joblib").touch()
        (tmp_path / "signal_model_v2_20241231.joblib").touch()
        result = _find_model_file(tmp_path)
        assert "v2" in result.name or "20241231" in result.name

    def test_ignores_non_matching_files(self, tmp_path):
        (tmp_path / "other_model.joblib").touch()
        assert _find_model_file(tmp_path) is None


# ─────────────────────────────────────────────────────────────────────────────
#  BacktestGUIBackend.get_available_symbols
# ─────────────────────────────────────────────────────────────────────────────

class TestGetAvailableSymbols:
    def _backend(self, tmp_path):
        return BacktestGUIBackend(features_dir=tmp_path, models_dir=tmp_path)

    def test_empty_when_no_files(self, tmp_path):
        assert self._backend(tmp_path).get_available_symbols() == []

    def test_single_symbol(self, tmp_path):
        _write_features(tmp_path, symbol="EURUSD")
        assert "EURUSD" in self._backend(tmp_path).get_available_symbols()

    def test_multiple_symbols(self, tmp_path):
        for sym in ("EURUSD", "GBPUSD", "USDJPY"):
            _write_features(tmp_path, symbol=sym)
        syms = self._backend(tmp_path).get_available_symbols()
        assert set(syms) == {"EURUSD", "GBPUSD", "USDJPY"}

    def test_deduplicates_multiple_files(self, tmp_path):
        (tmp_path / "EURUSD_H1_20230101.parquet").touch()
        (tmp_path / "EURUSD_H4_20230101.parquet").touch()
        syms = self._backend(tmp_path).get_available_symbols()
        assert syms.count("EURUSD") == 1

    def test_sorted_alphabetically(self, tmp_path):
        for sym in ("USDJPY", "EURUSD", "GBPUSD"):
            _write_features(tmp_path, symbol=sym)
        syms = self._backend(tmp_path).get_available_symbols()
        assert syms == sorted(syms)


# ─────────────────────────────────────────────────────────────────────────────
#  BacktestGUIBackend._load_features
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadFeatures:
    def _backend(self, tmp_path):
        return BacktestGUIBackend(features_dir=tmp_path, models_dir=tmp_path)

    def test_raises_setup_error_when_no_file(self, tmp_path):
        with pytest.raises(BacktestSetupError):
            self._backend(tmp_path)._load_features("EURUSD", "H1", "2023-01-01", "2024-01-01")

    def test_error_mentions_fetch_data(self, tmp_path):
        with pytest.raises(BacktestSetupError) as exc:
            self._backend(tmp_path)._load_features("EURUSD", "H1", "2023-01-01", "2024-01-01")
        assert "fetch_data" in str(exc.value) or "scripts" in str(exc.value)

    def test_error_mentions_symbol(self, tmp_path):
        with pytest.raises(BacktestSetupError) as exc:
            self._backend(tmp_path)._load_features("MYSYM", "H1", "2023-01-01", "2024-01-01")
        assert "MYSYM" in str(exc.value)

    def test_returns_dataframe_when_file_exists(self, tmp_path):
        _write_features(tmp_path)
        df = self._backend(tmp_path)._load_features(
            "EURUSD", "H1", "2023-01-01", "2024-01-01"
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_dataframe_has_close_column(self, tmp_path):
        _write_features(tmp_path)
        df = self._backend(tmp_path)._load_features(
            "EURUSD", "H1", "2023-01-01", "2024-01-01"
        )
        assert "close" in df.columns

    def test_index_is_datetime(self, tmp_path):
        _write_features(tmp_path)
        df = self._backend(tmp_path)._load_features(
            "EURUSD", "H1", "2023-01-01", "2024-01-01"
        )
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_date_filter_applied(self, tmp_path):
        _write_features(tmp_path, n=200)
        df = self._backend(tmp_path)._load_features(
            "EURUSD", "H1", "2023-01-01", "2023-01-08"
        )
        assert df.index.max() <= pd.Timestamp("2023-01-08", tz="UTC")

    def test_empty_range_raises_setup_error(self, tmp_path):
        _write_features(tmp_path, n=10)
        with pytest.raises(BacktestSetupError) as exc:
            self._backend(tmp_path)._load_features(
                "EURUSD", "H1", "2020-01-01", "2020-12-31"
            )
        assert "Zeitraum" in str(exc.value) or "keine Daten" in str(exc.value).lower()

    def test_empty_range_error_mentions_available_range(self, tmp_path):
        _write_features(tmp_path, n=10)
        with pytest.raises(BacktestSetupError) as exc:
            self._backend(tmp_path)._load_features(
                "EURUSD", "H1", "2020-01-01", "2020-12-31"
            )
        assert "2023" in str(exc.value)  # available range contains 2023

    def test_raises_when_close_missing(self, tmp_path):
        df = _make_features_df()
        df = df.drop(columns=["close"])
        (tmp_path / "EURUSD_H1_20241231.parquet").parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(tmp_path / "EURUSD_H1_20241231.parquet")
        with pytest.raises(BacktestSetupError) as exc:
            self._backend(tmp_path)._load_features(
                "EURUSD", "H1", "2023-01-01", "2024-01-01"
            )
        assert "close" in str(exc.value).lower()


# ─────────────────────────────────────────────────────────────────────────────
#  BacktestGUIBackend._load_signal_func
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadSignalFunc:
    def _backend(self, tmp_path):
        return BacktestGUIBackend(features_dir=tmp_path, models_dir=tmp_path)

    def test_raises_setup_error_when_no_model(self, tmp_path):
        with pytest.raises(BacktestSetupError):
            self._backend(tmp_path)._load_signal_func()

    def test_error_mentions_train(self, tmp_path):
        with pytest.raises(BacktestSetupError) as exc:
            self._backend(tmp_path)._load_signal_func()
        msg = str(exc.value).lower()
        assert "train" in msg or "modell" in msg

    def test_error_mentions_models_dir(self, tmp_path):
        with pytest.raises(BacktestSetupError) as exc:
            self._backend(tmp_path)._load_signal_func()
        assert "signal_model" in str(exc.value)

    def test_returns_callable_when_model_exists(self, tmp_path):
        _write_model(tmp_path)
        mock_model = MagicMock()
        mock_model.get_signal = MagicMock(return_value="flat")
        mock_sm_module = MagicMock()
        mock_sm_module.SignalModel.load.return_value = mock_model
        with patch.dict(sys.modules, {"src.models.signal_model": mock_sm_module, "lightgbm": MagicMock()}):
            result = self._backend(tmp_path)._load_signal_func()
        assert callable(result)

    def test_calls_signal_model_load(self, tmp_path):
        _write_model(tmp_path)
        mock_model = MagicMock()
        mock_sm_module = MagicMock()
        mock_sm_module.SignalModel.load.return_value = mock_model
        with patch.dict(sys.modules, {"src.models.signal_model": mock_sm_module, "lightgbm": MagicMock()}):
            self._backend(tmp_path)._load_signal_func()
        mock_sm_module.SignalModel.load.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
#  BacktestGUIBackend.run_backtest (Ende-zu-Ende mit Mocks)
# ─────────────────────────────────────────────────────────────────────────────

class TestRunBacktest:
    def _backend(self, tmp_path):
        return BacktestGUIBackend(features_dir=tmp_path, models_dir=tmp_path)

    def test_raises_when_no_features(self, tmp_path):
        with pytest.raises(BacktestSetupError):
            self._backend(tmp_path).run_backtest(
                "EURUSD", "H1", "2023-01-01", "2024-01-01", None, 10_000.0
            )

    def test_raises_when_no_model(self, tmp_path):
        _write_features(tmp_path)
        with pytest.raises(BacktestSetupError):
            self._backend(tmp_path).run_backtest(
                "EURUSD", "H1", "2023-01-01", "2024-01-01", None, 10_000.0
            )

    def _sm_modules(self, mock_model):
        """Fake sys.modules entries so `from src.models.signal_model import SignalModel` works."""
        mock_sm = MagicMock()
        mock_sm.SignalModel.load.return_value = mock_model
        return {"src.models.signal_model": mock_sm, "lightgbm": MagicMock()}

    def test_returns_backtest_result(self, tmp_path):
        _write_features(tmp_path)
        _write_model(tmp_path)
        expected = _make_result()
        mock_model = MagicMock()
        mock_model.get_signal = MagicMock(return_value="flat")
        with (
            patch.dict(sys.modules, self._sm_modules(mock_model)),
            patch("gui.backends.backtest_backend.BacktestRunner") as MockRunner,
        ):
            MockRunner.return_value.run_with_model.return_value = expected
            result = self._backend(tmp_path).run_backtest(
                "EURUSD", "H1", "2023-01-01", "2024-01-01", None, 10_000.0
            )
        assert result is expected

    def test_passes_init_cash_to_config(self, tmp_path):
        _write_features(tmp_path)
        _write_model(tmp_path)
        mock_model = MagicMock()
        captured_cfg = {}
        with (
            patch.dict(sys.modules, self._sm_modules(mock_model)),
            patch("gui.backends.backtest_backend.BacktestRunner") as MockRunner,
            patch("gui.backends.backtest_backend.BacktestConfig") as MockCfg,
        ):
            MockRunner.return_value.run_with_model.return_value = _make_result()
            MockCfg.side_effect = lambda **kw: (captured_cfg.update(kw) or MagicMock())
            self._backend(tmp_path).run_backtest(
                "EURUSD", "H1", "2023-01-01", "2024-01-01", None, 25_000.0
            )
        assert captured_cfg.get("init_cash") == 25_000.0

    def test_passes_is_split_to_runner(self, tmp_path):
        _write_features(tmp_path)
        _write_model(tmp_path)
        mock_model = MagicMock()
        with (
            patch.dict(sys.modules, self._sm_modules(mock_model)),
            patch("gui.backends.backtest_backend.BacktestRunner") as MockRunner,
        ):
            MockRunner.return_value.run_with_model.return_value = _make_result()
            self._backend(tmp_path).run_backtest(
                "EURUSD", "H1", "2023-01-01", "2024-01-01", "2023-12-31", 10_000.0
            )
            call_kwargs = MockRunner.return_value.run_with_model.call_args.kwargs
        assert call_kwargs.get("is_end") == "2023-12-31"

    def test_no_is_split_auto_computes_split(self, tmp_path):
        """Wenn is_split=None, berechnet Backend automatisch 70/30-Split."""
        _write_features(tmp_path)
        _write_model(tmp_path)
        mock_model = MagicMock()
        with (
            patch.dict(sys.modules, self._sm_modules(mock_model)),
            patch("gui.backends.backtest_backend.BacktestRunner") as MockRunner,
        ):
            MockRunner.return_value.run_with_model.return_value = _make_result()
            self._backend(tmp_path).run_backtest(
                "EURUSD", "H1", "2023-01-01", "2024-01-01", None, 10_000.0
            )
            call_kwargs = MockRunner.return_value.run_with_model.call_args.kwargs
        # Auto-split muss gesetzt sein (nicht None) und im Zeitraum liegen
        is_end = call_kwargs.get("is_end")
        assert is_end is not None
        assert "2023" in str(is_end)  # 70% von 2023-01-01..2024-01-01 liegt in 2023


# ─────────────────────────────────────────────────────────────────────────────
#  BacktestGUIBackend.export_markdown
# ─────────────────────────────────────────────────────────────────────────────

class TestExportMarkdown:
    def test_writes_file(self, tmp_path):
        backend = BacktestGUIBackend(features_dir=tmp_path, models_dir=tmp_path)
        result  = _make_result()
        outfile = tmp_path / "report.md"
        backend.export_markdown(result, str(outfile))
        assert outfile.exists()

    def test_file_contains_sharpe(self, tmp_path):
        backend = BacktestGUIBackend(features_dir=tmp_path, models_dir=tmp_path)
        result  = _make_result()
        outfile = tmp_path / "report.md"
        backend.export_markdown(result, str(outfile))
        content = outfile.read_text(encoding="utf-8")
        assert "Sharpe" in content

    def test_file_contains_total_return(self, tmp_path):
        backend = BacktestGUIBackend(features_dir=tmp_path, models_dir=tmp_path)
        result  = _make_result()
        outfile = tmp_path / "report.md"
        backend.export_markdown(result, str(outfile))
        content = outfile.read_text(encoding="utf-8")
        assert "5.00%" in content or "Gesamtertrag" in content


# ─────────────────────────────────────────────────────────────────────────────
#  _result_to_markdown
# ─────────────────────────────────────────────────────────────────────────────

class TestResultToMarkdown:
    def test_returns_string(self):
        assert isinstance(_result_to_markdown(_make_result()), str)

    def test_contains_header(self):
        md = _result_to_markdown(_make_result(), name="TestLauf")
        assert "TestLauf" in md

    def test_contains_total_return(self):
        md = _result_to_markdown(_make_result())
        assert "Gesamtertrag" in md

    def test_contains_sharpe(self):
        md = _result_to_markdown(_make_result())
        assert "Sharpe" in md

    def test_infinity_profit_factor(self):
        r   = _make_result()
        r.profit_factor = float("inf")
        md  = _result_to_markdown(r)
        assert "∞" in md

    def test_overfitting_warning_ja(self):
        r = _make_result()
        r.overfitting_warning = True
        assert "Ja" in _result_to_markdown(r)

    def test_overfitting_warning_nein(self):
        r = _make_result()
        r.overfitting_warning = False
        assert "Nein" in _result_to_markdown(r)

    def test_is_oos_sharpe_shown(self):
        r = _make_result()
        r.is_sharpe  = 1.23
        r.oos_sharpe = 0.87
        md = _result_to_markdown(r)
        assert "1.230" in md
        assert "0.870" in md

    def test_none_sharpe_shows_dash(self):
        r = _make_result()
        r.is_sharpe  = None
        r.oos_sharpe = None
        md = _result_to_markdown(r)
        assert "–" in md


# ─────────────────────────────────────────────────────────────────────────────
#  Integration: BacktestGUIBackend als BacktestBackend erkannt
# ─────────────────────────────────────────────────────────────────────────────

class TestProtocolCompliance:
    def test_implements_protocol(self, tmp_path):
        from gui.views.backtest_view import BacktestBackend
        backend = BacktestGUIBackend(features_dir=tmp_path, models_dir=tmp_path)
        assert isinstance(backend, BacktestBackend)
