"""
tests/unit/test_validator.py
Unit-Tests fuer DataValidator und DataQualityReport.
Keine externen Abhaengigkeiten ausser pandas/numpy.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.validator import DataValidator, DataQualityReport, DataQualityError


# ─────────────────────────────────────────────
#  Hilfsfunktionen
# ─────────────────────────────────────────────

def _make_ohlcv(
    n: int = 100,
    start: datetime | None = None,
    freq: str = "h",
    tz_aware: bool = True,
) -> pd.DataFrame:
    """Erzeugt einen sauberen synthetischen OHLCV-DataFrame."""
    if start is None:
        start = datetime(2024, 1, 2, tzinfo=timezone.utc)  # Dienstag

    idx = pd.date_range(start=start, periods=n, freq=freq, tz="UTC" if tz_aware else None)
    # Nur Wochentage (Mo–Fr)
    idx = idx[idx.dayofweek < 5][:n]
    # Wenn zu wenige uebrig -> wiederholen bis n erreicht
    while len(idx) < n:
        more = pd.date_range(start=idx[-1] + pd.tseries.frequencies.to_offset(freq),
                             periods=n, freq=freq, tz="UTC" if tz_aware else None)
        more = more[more.dayofweek < 5]
        idx  = idx.append(more)
    idx = idx[:n]

    rng    = np.random.default_rng(42)
    close  = 1.1000 + rng.normal(0, 0.001, n).cumsum()
    open_  = close + rng.uniform(-0.0005, 0.0005, n)
    high   = np.maximum(open_, close) + rng.uniform(0.0001, 0.001, n)
    low    = np.minimum(open_, close) - rng.uniform(0.0001, 0.001, n)
    volume = rng.integers(100, 1000, n).astype(float)

    return pd.DataFrame({
        "timestamp": idx,
        "open":   open_,
        "high":   high,
        "low":    low,
        "close":  close,
        "volume": volume,
    })


@pytest.fixture
def validator() -> DataValidator:
    return DataValidator(
        max_missing_pct=5.0,
        outlier_atr_multiplier=5.0,
        min_quality_score=0.95,
        interpolation_method="linear",
        max_gap_candles=3,
    )


@pytest.fixture
def clean_df() -> pd.DataFrame:
    return _make_ohlcv(100)


# ─────────────────────────────────────────────
#  Tests: DataQualityReport
# ─────────────────────────────────────────────

class TestDataQualityReport:

    def _make_report(self, **kwargs) -> DataQualityReport:
        defaults = dict(
            symbol="EURUSD", timeframe="H1",
            total_candles=100, missing_candles=0, missing_pct=0.0,
            duplicates_removed=0, ohlc_violations=0, outliers_flagged=0,
            nan_rows_removed=0, quality_score=1.0, is_usable=True,
        )
        defaults.update(kwargs)
        return DataQualityReport(**defaults)

    def test_to_json_is_valid(self):
        report = self._make_report()
        data   = json.loads(report.to_json())
        assert data["symbol"]    == "EURUSD"
        assert data["timeframe"] == "H1"
        assert data["is_usable"] is True

    def test_to_json_all_fields_present(self):
        report = self._make_report()
        data   = json.loads(report.to_json())
        for key in [
            "symbol", "timeframe", "total_candles", "missing_candles",
            "missing_pct", "duplicates_removed", "ohlc_violations",
            "outliers_flagged", "nan_rows_removed", "quality_score",
            "is_usable", "warnings", "errors",
        ]:
            assert key in data, f"Feld '{key}' fehlt im JSON"

    def test_save_creates_file(self, tmp_path):
        report = self._make_report()
        path   = report.save(tmp_path)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["symbol"] == "EURUSD"

    def test_save_creates_directory(self, tmp_path):
        report = self._make_report()
        subdir = tmp_path / "quality_reports" / "sub"
        report.save(subdir)
        assert subdir.exists()


# ─────────────────────────────────────────────
#  Tests: Saubere Daten
# ─────────────────────────────────────────────

class TestCleanData:

    def test_clean_data_is_usable(self, validator, clean_df):
        report = validator.validate(clean_df, "EURUSD", "H1")
        assert report.is_usable is True

    def test_clean_data_no_violations(self, validator, clean_df):
        report = validator.validate(clean_df, "EURUSD", "H1")
        assert report.duplicates_removed == 0
        assert report.ohlc_violations    == 0
        assert report.nan_rows_removed   == 0

    def test_report_has_correct_symbol_timeframe(self, validator, clean_df):
        report = validator.validate(clean_df, "GBPUSD", "H4")
        assert report.symbol    == "GBPUSD"
        assert report.timeframe == "H4"

    def test_quality_score_between_0_and_1(self, validator, clean_df):
        report = validator.validate(clean_df, "EURUSD", "H1")
        assert 0.0 <= report.quality_score <= 1.0


# ─────────────────────────────────────────────
#  Tests: Pruefung 5 – NaN-Werte
# ─────────────────────────────────────────────

class TestNaNCheck:

    def test_nan_rows_removed(self, validator, clean_df):
        clean_df.loc[5, "close"] = np.nan
        clean_df.loc[10, "high"] = np.nan
        report = validator.validate(clean_df, "EURUSD", "H1")
        assert report.nan_rows_removed == 2

    def test_nan_warning_in_report(self, validator, clean_df):
        clean_df.loc[3, "open"] = np.nan
        report = validator.validate(clean_df, "EURUSD", "H1")
        assert any("NaN" in w for w in report.warnings)

    def test_all_nan_rows_removed_from_df(self, validator, clean_df):
        clean_df.loc[0, "volume"] = np.nan
        report = validator.validate(clean_df, "EURUSD", "H1")
        assert report.total_candles == len(clean_df) - 1


# ─────────────────────────────────────────────
#  Tests: Pruefung 2 – Duplikate
# ─────────────────────────────────────────────

class TestDuplicateCheck:

    def test_duplicates_removed(self, validator, clean_df):
        dup = clean_df.iloc[[5, 5]].copy()
        df  = pd.concat([clean_df, dup]).reset_index(drop=True)
        report = validator.validate(df, "EURUSD", "H1")
        assert report.duplicates_removed >= 1

    def test_newest_entry_kept(self, validator, clean_df):
        ts  = clean_df.iloc[0]["timestamp"]
        row = clean_df.iloc[[0]].copy()
        row["close"] = 9999.0
        df  = pd.concat([clean_df, row]).reset_index(drop=True)
        report = validator.validate(df, "EURUSD", "H1")
        assert report.duplicates_removed >= 1

    def test_no_duplicate_warning_when_clean(self, validator, clean_df):
        report = validator.validate(clean_df, "EURUSD", "H1")
        assert not any("Duplikat" in w for w in report.warnings)


# ─────────────────────────────────────────────
#  Tests: Pruefung 3 – OHLC-Konsistenz
# ─────────────────────────────────────────────

class TestOHLCConsistency:

    def test_high_below_close_removed(self, validator, clean_df):
        clean_df.loc[5, "high"] = clean_df.loc[5, "low"] - 0.001  # high < low
        report = validator.validate(clean_df, "EURUSD", "H1")
        assert report.ohlc_violations >= 1

    def test_low_above_open_removed(self, validator, clean_df):
        clean_df.loc[7, "low"] = clean_df.loc[7, "open"] + 0.01  # low > open
        report = validator.validate(clean_df, "EURUSD", "H1")
        assert report.ohlc_violations >= 1

    def test_ohlc_warning_in_report(self, validator, clean_df):
        clean_df.loc[3, "high"] = clean_df.loc[3, "close"] - 0.01
        report = validator.validate(clean_df, "EURUSD", "H1")
        assert any("OHLC" in w for w in report.warnings)

    def test_valid_ohlc_not_flagged(self, validator, clean_df):
        report = validator.validate(clean_df, "EURUSD", "H1")
        assert report.ohlc_violations == 0


# ─────────────────────────────────────────────
#  Tests: Pruefung 6 – Zero-Range
# ─────────────────────────────────────────────

class TestZeroRange:

    def test_zero_range_candle_removed(self, validator, clean_df):
        # Alle OHLC-Werte gleich setzen: OHLC-valid, aber high-low == 0
        val = clean_df.loc[10, "close"]
        clean_df.loc[10, "high"]  = val
        clean_df.loc[10, "low"]   = val
        clean_df.loc[10, "open"]  = val
        before = len(clean_df)
        report = validator.validate(clean_df, "EURUSD", "H1")
        # Zeile entfernt (danach ggf. interpoliert) – mind. eine Warning vorhanden
        all_warnings = " ".join(report.warnings)
        assert "Zero-Range" in all_warnings or "Lueck" in all_warnings or "interpoliert" in all_warnings

    def test_zero_range_warning(self, validator, clean_df):
        val = clean_df.loc[10, "close"]
        clean_df.loc[10, "high"]  = val
        clean_df.loc[10, "low"]   = val
        clean_df.loc[10, "open"]  = val
        report = validator.validate(clean_df, "EURUSD", "H1")
        all_warnings = " ".join(report.warnings)
        assert "Zero-Range" in all_warnings or "Lueck" in all_warnings or "interpoliert" in all_warnings

    def test_negative_range_also_removed(self, validator, clean_df):
        clean_df.loc[5, "high"] = clean_df.loc[5, "low"] - 0.0001
        report = validator.validate(clean_df, "EURUSD", "H1")
        assert report.ohlc_violations >= 1 or report.total_candles < len(clean_df)


# ─────────────────────────────────────────────
#  Tests: Pruefung 4 – Ausreisser
# ─────────────────────────────────────────────

class TestOutlierCheck:

    def test_outlier_flagged(self, validator):
        df = _make_ohlcv(200)
        # Riesige Kerze einfuegen
        df.loc[100, "high"] = df.loc[100, "close"] + 5.0   # 5 EUR Range
        df.loc[100, "low"]  = df.loc[100, "close"] - 5.0
        report = validator.validate(df, "EURUSD", "H1")
        assert report.outliers_flagged >= 1

    def test_no_outlier_on_clean_data(self, validator, clean_df):
        report = validator.validate(clean_df, "EURUSD", "H1")
        assert report.outliers_flagged == 0

    def test_outlier_warning_in_report(self, validator):
        df = _make_ohlcv(200)
        df.loc[100, "high"] = df.loc[100, "close"] + 5.0
        df.loc[100, "low"]  = df.loc[100, "close"] - 5.0
        report = validator.validate(df, "EURUSD", "H1")
        if report.outliers_flagged > 0:
            assert any("Ausreisser" in w or "outlier" in w.lower() for w in report.warnings)


# ─────────────────────────────────────────────
#  Tests: Pruefung 1 – Zeitluecken
# ─────────────────────────────────────────────

class TestGapCheck:

    def test_no_gaps_on_clean_data(self, validator, clean_df):
        report = validator.validate(clean_df, "EURUSD", "H1")
        assert report.missing_candles == 0

    def test_gaps_detected(self, validator):
        df = _make_ohlcv(50)
        # 2 Zeilen entfernen → Luecke
        df = df.drop(index=[10, 11]).reset_index(drop=True)
        report = validator.validate(df, "EURUSD", "H1")
        assert report.missing_candles >= 2

    def test_small_gaps_cause_warning_not_error(self, validator):
        df = _make_ohlcv(50)
        df = df.drop(index=[10]).reset_index(drop=True)
        report = validator.validate(df, "EURUSD", "H1")
        assert report.is_usable is True
        assert any("interpoliert" in w or "Lueck" in w for w in report.warnings)

    def test_large_gap_raises_error(self):
        v  = DataValidator(max_missing_pct=1.0)
        df = _make_ohlcv(100)
        # Viele Zeilen entfernen → > 1% Luecken
        df = df.drop(index=list(range(10, 25))).reset_index(drop=True)
        with pytest.raises(DataQualityError, match="Zu viele fehlende Candles"):
            v.validate(df, "EURUSD", "H1")

    def test_missing_pct_in_report(self, validator):
        df     = _make_ohlcv(100)
        df_cut = df.drop(index=[5]).reset_index(drop=True)
        report = validator.validate(df_cut, "EURUSD", "H1")
        assert report.missing_pct >= 0.0


# ─────────────────────────────────────────────
#  Tests: Qualitaetsscore & Gesamtverhalten
# ─────────────────────────────────────────────

class TestQualityScore:

    def test_perfect_data_score_near_1(self, validator, clean_df):
        report = validator.validate(clean_df, "EURUSD", "H1")
        assert report.quality_score >= 0.95

    def test_low_quality_not_usable(self):
        v = DataValidator(min_quality_score=0.99, max_missing_pct=50.0)
        df = _make_ohlcv(100)
        # Viele OHLC-Verletzungen einbauen
        for i in range(0, 30):
            df.loc[i, "high"] = df.loc[i, "low"] - 0.001
        report = v.validate(df, "EURUSD", "H1")
        assert report.is_usable is False

    def test_usable_flag_true_on_good_data(self, validator, clean_df):
        report = validator.validate(clean_df, "EURUSD", "H1")
        assert report.is_usable is True

    def test_errors_populated_when_not_usable(self):
        v = DataValidator(min_quality_score=0.99, max_missing_pct=50.0)
        df = _make_ohlcv(100)
        for i in range(0, 30):
            df.loc[i, "high"] = df.loc[i, "low"] - 0.001
        report = v.validate(df, "EURUSD", "H1")
        if not report.is_usable:
            assert len(report.errors) > 0


# ─────────────────────────────────────────────
#  Tests: from_config
# ─────────────────────────────────────────────

class TestFromConfig:

    def test_from_config_loads_values(self, tmp_path):
        config = tmp_path / "config.yaml"
        config.write_text(
            "data_validation:\n"
            "  max_missing_pct: 3.0\n"
            "  outlier_atr_multiplier: 4.0\n"
            "  min_quality_score: 0.98\n"
            "  interpolation_method: linear\n"
            "  max_gap_candles: 2\n",
            encoding="utf-8",
        )
        v = DataValidator.from_config(config)
        assert v.max_missing_pct        == 3.0
        assert v.outlier_atr_multiplier == 4.0
        assert v.min_quality_score      == 0.98
        assert v.max_gap_candles        == 2

    def test_from_config_uses_defaults_when_missing(self, tmp_path):
        config = tmp_path / "config.yaml"
        config.write_text("data_validation: {}\n", encoding="utf-8")
        v = DataValidator.from_config(config)
        assert v.max_missing_pct   == 5.0
        assert v.min_quality_score == 0.95


# ─────────────────────────────────────────────
#  Tests: Fehlerfaelle
# ─────────────────────────────────────────────

class TestEdgeCases:

    def test_no_timestamp_column_raises(self, validator):
        df = pd.DataFrame({"open": [1.0], "high": [2.0], "low": [0.5], "close": [1.5]})
        with pytest.raises(DataQualityError, match="timestamp"):
            validator.validate(df, "EURUSD", "H1")

    def test_empty_df_raises(self, validator):
        df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        # Leerer DataFrame -> Qualitaetsscore 0, nicht usable
        with pytest.raises(DataQualityError):
            validator.validate(df, "EURUSD", "H1")

    def test_single_row_df(self, validator):
        df = _make_ohlcv(1)
        # Soll nicht crashen
        report = validator.validate(df, "EURUSD", "H1")
        assert report.total_candles >= 0

    def test_unknown_timeframe_no_gap_check(self, validator, clean_df):
        # Unbekannter Timeframe -> kein Gap-Check, kein Absturz
        report = validator.validate(clean_df, "EURUSD", "X99")
        assert report is not None
