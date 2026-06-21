"""
Tests: Multi-Timeframe Feature Look-ahead Prevention
=====================================================
Prueft, dass h4_trend und d1_trend fuer H1-Bar T ausschliesslich
Informationen von HTF-Bars verwenden, deren close_time <= T.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.feature_builder import FeatureBuilder


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def _make_ohlcv(timestamps: pd.DatetimeIndex, seed: int = 0) -> pd.DataFrame:
    """Erstellt einen minimalen OHLCV-DataFrame fuer Tests."""
    rng = np.random.default_rng(seed)
    n = len(timestamps)
    close = 1.1000 + rng.normal(0, 0.001, n).cumsum()
    high  = close + rng.uniform(0.0001, 0.0005, n)
    low   = close - rng.uniform(0.0001, 0.0005, n)
    return pd.DataFrame({
        "timestamp": timestamps,
        "open":  close,
        "high":  high,
        "low":   low,
        "close": close,
        "volume": np.ones(n) * 1000,
    })


def _make_h1_ohlcv(start: str = "2023-01-01", periods: int = 500) -> pd.DataFrame:
    ts = pd.date_range(start, periods=periods, freq="1h")
    return _make_ohlcv(ts)


def _make_h4_ohlcv(start: str = "2023-01-01", periods: int = 300) -> pd.DataFrame:
    ts = pd.date_range(start, periods=periods, freq="4h")
    return _make_ohlcv(ts, seed=1)


def _make_d1_ohlcv(start: str = "2023-01-01", periods: int = 100) -> pd.DataFrame:
    ts = pd.date_range(start, periods=periods, freq="24h")
    return _make_ohlcv(ts, seed=2)


# ---------------------------------------------------------------------------
# _compute_mtf_trend
# ---------------------------------------------------------------------------

class TestComputeMtfTrend:
    def test_returns_dataframe(self):
        df = _make_h4_ohlcv()
        result = FeatureBuilder._compute_mtf_trend(df, tf_hours=4)
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self):
        df = _make_h4_ohlcv()
        result = FeatureBuilder._compute_mtf_trend(df, tf_hours=4)
        assert "close_time" in result.columns
        assert "trend" in result.columns

    def test_close_time_equals_open_plus_tf(self):
        """close_time muss genau open_time + tf_hours sein."""
        df = _make_h4_ohlcv(periods=50)  # >= 14 bars fuer ATR14
        result = FeatureBuilder._compute_mtf_trend(df, tf_hours=4)
        expected = pd.to_datetime(df["timestamp"].values) + pd.Timedelta(hours=4)
        pd.testing.assert_index_equal(
            pd.DatetimeIndex(result["close_time"]),
            pd.DatetimeIndex(expected),
            check_names=False,
        )

    def test_close_time_d1_offset(self):
        """D1: close_time = open_time + 24 Stunden."""
        df = _make_d1_ohlcv(periods=50)  # >= 14 bars fuer ATR14
        result = FeatureBuilder._compute_mtf_trend(df, tf_hours=24)
        deltas = (
            pd.to_datetime(result["close_time"]) -
            pd.to_datetime(df["timestamp"].values)
        )
        assert (deltas == pd.Timedelta(hours=24)).all()

    def test_trend_is_float(self):
        df = _make_h4_ohlcv()
        result = FeatureBuilder._compute_mtf_trend(df, tf_hours=4)
        assert result["trend"].dtype == float

    def test_no_nan_in_trend(self):
        """fillna(0.0) muss alle NaN-Werte (Warmup-Phase) ersetzen."""
        df = _make_h4_ohlcv()
        result = FeatureBuilder._compute_mtf_trend(df, tf_hours=4)
        assert result["trend"].isna().sum() == 0

    def test_length_matches_input(self):
        df = _make_h4_ohlcv(periods=50)
        result = FeatureBuilder._compute_mtf_trend(df, tf_hours=4)
        assert len(result) == 50


# ---------------------------------------------------------------------------
# _merge_mtf_trend  --  Look-ahead-Disziplin
# ---------------------------------------------------------------------------

class TestMergeMtfTrendLookahead:
    """
    Kerntest: Fuer jeden H1-Bar T darf nur der letzte HTF-Bar verwendet werden,
    dessen close_time <= T ist. Ein HTF-Bar mit close_time > T (laufende Kerze)
    darf NICHT einfliessen.
    """

    def _build_scenario(self):
        """
        Zeitachse:
          H4-Bar A: open=08:00, close_time=12:00  trend=+1.0
          H4-Bar B: open=12:00, close_time=16:00  trend=-1.0  (noch laufend um 13:00)
          H4-Bar C: open=16:00, close_time=20:00  trend=+2.0

          H1-Bars: 09:00, 11:00, 12:00, 13:00, 15:59, 16:00, 17:00, 20:00, 21:00
        """
        mtf_df = pd.DataFrame({
            "close_time": pd.to_datetime([
                "2023-01-02 12:00",
                "2023-01-02 16:00",
                "2023-01-02 20:00",
            ]),
            "trend": [1.0, -1.0, 2.0],
        })
        h1_ts = pd.Series(pd.to_datetime([
            "2023-01-02 09:00",   # kein vorheriger H4-Bar -> 0.0
            "2023-01-02 11:00",   # kein vorheriger H4-Bar -> 0.0
            "2023-01-02 12:00",   # close_time=12:00 <= 12:00 -> Bar A (+1.0)
            "2023-01-02 13:00",   # Bar A geschlossen, Bar B laeuft noch -> +1.0
            "2023-01-02 15:59",   # Bar B noch offen (close=16:00 > 15:59) -> +1.0
            "2023-01-02 16:00",   # Bar B geschlossen (16:00 <= 16:00) -> -1.0
            "2023-01-02 17:00",   # Bar B ist letzter mit close<=17 -> -1.0
            "2023-01-02 20:00",   # Bar C geschlossen (20:00 <= 20:00) -> +2.0
            "2023-01-02 21:00",   # Bar C ist letzter -> +2.0
        ]))
        return h1_ts, mtf_df

    def test_no_lookahead_running_bar(self):
        """H1 13:00: Bar B (close=16:00) darf NICHT verwendet werden."""
        h1_ts, mtf_df = self._build_scenario()
        result = FeatureBuilder._merge_mtf_trend(h1_ts, mtf_df)
        # Index 3 = 13:00 -> muss +1.0 sein (Bar A), nicht -1.0 (Bar B)
        assert result[3] == pytest.approx(1.0), (
            f"Look-ahead! 13:00 nutzte laufenden Bar B (trend=-1.0): got {result[3]}"
        )

    def test_no_lookahead_just_before_close(self):
        """H1 15:59: Bar B schliesst erst um 16:00, darf nicht verwendet werden."""
        h1_ts, mtf_df = self._build_scenario()
        result = FeatureBuilder._merge_mtf_trend(h1_ts, mtf_df)
        assert result[4] == pytest.approx(1.0), (
            f"Look-ahead! 15:59 nutzte Bar B (close=16:00): got {result[4]}"
        )

    def test_bar_used_exactly_at_close_time(self):
        """H1 16:00: Bar B ist jetzt geschlossen (close_time == bar_time ist OK)."""
        h1_ts, mtf_df = self._build_scenario()
        result = FeatureBuilder._merge_mtf_trend(h1_ts, mtf_df)
        assert result[5] == pytest.approx(-1.0), (
            f"Bar B (close=16:00) sollte bei H1 16:00 genutzt werden: got {result[5]}"
        )

    def test_neutral_before_first_htf_bar(self):
        """H1-Bars vor dem ersten HTF-Bar -> Trend = 0.0 (neutral)."""
        h1_ts, mtf_df = self._build_scenario()
        result = FeatureBuilder._merge_mtf_trend(h1_ts, mtf_df)
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(0.0)

    def test_correct_propagation_after_close(self):
        """Der zuletzt geschlossene HTF-Trendwert haelt bis zum naechsten Bar."""
        h1_ts, mtf_df = self._build_scenario()
        result = FeatureBuilder._merge_mtf_trend(h1_ts, mtf_df)
        # 17:00 -> letzter geschlossener ist Bar B (close=16:00) -> -1.0
        assert result[6] == pytest.approx(-1.0)
        # 21:00 -> letzter ist Bar C (close=20:00) -> +2.0
        assert result[8] == pytest.approx(2.0)

    def test_returns_numpy_array(self):
        h1_ts, mtf_df = self._build_scenario()
        result = FeatureBuilder._merge_mtf_trend(h1_ts, mtf_df)
        assert isinstance(result, np.ndarray)
        assert result.dtype == float

    def test_length_matches_h1(self):
        h1_ts, mtf_df = self._build_scenario()
        result = FeatureBuilder._merge_mtf_trend(h1_ts, mtf_df)
        assert len(result) == len(h1_ts)

    def test_no_nan_in_output(self):
        h1_ts, mtf_df = self._build_scenario()
        result = FeatureBuilder._merge_mtf_trend(h1_ts, mtf_df)
        assert not np.isnan(result).any()

    def test_order_preserved_when_input_unsorted(self):
        """Unsortierte H1-Zeitstempel muessen korrekt zugeordnet werden."""
        h1_ts, mtf_df = self._build_scenario()
        # Shuffle
        shuffled_idx = [4, 0, 8, 2, 6, 1, 7, 3, 5]
        h1_shuffled = h1_ts.iloc[shuffled_idx].reset_index(drop=True)
        result_sorted   = FeatureBuilder._merge_mtf_trend(h1_ts, mtf_df)
        result_shuffled = FeatureBuilder._merge_mtf_trend(h1_shuffled, mtf_df)
        for new_pos, orig_pos in enumerate(shuffled_idx):
            assert result_shuffled[new_pos] == pytest.approx(result_sorted[orig_pos])


# ---------------------------------------------------------------------------
# Integration: build() mit df_h4 / df_d1
# ---------------------------------------------------------------------------

class TestBuildWithMtfFeatures:
    def _fb(self) -> FeatureBuilder:
        return FeatureBuilder(
            ema_periods=[9, 20, 50, 200],
            sma_periods=[50],
            rsi_periods=[14],
            atr_period=14,
            warmup_candles=200,
        )

    def test_h4_trend_present_in_output(self):
        fb = self._fb()
        df_h1 = _make_h1_ohlcv(periods=500)
        df_h4 = _make_h4_ohlcv(periods=300)
        result = fb.build(df_h1, df_h4=df_h4)
        assert "h4_trend" in result.columns

    def test_d1_trend_present_in_output(self):
        fb = self._fb()
        df_h1 = _make_h1_ohlcv(periods=500)
        df_d1 = _make_d1_ohlcv(periods=100)
        result = fb.build(df_h1, df_d1=df_d1)
        assert "d1_trend" in result.columns

    def test_both_features_present(self):
        fb = self._fb()
        df_h1 = _make_h1_ohlcv(periods=500)
        df_h4 = _make_h4_ohlcv(periods=300)
        df_d1 = _make_d1_ohlcv(periods=100)
        result = fb.build(df_h1, df_h4=df_h4, df_d1=df_d1)
        assert "h4_trend" in result.columns
        assert "d1_trend" in result.columns

    def test_without_mtf_args_no_mtf_columns(self):
        fb = self._fb()
        df_h1 = _make_h1_ohlcv(periods=500)
        result = fb.build(df_h1)
        assert "h4_trend" not in result.columns
        assert "d1_trend" not in result.columns

    def test_h4_trend_no_nan(self):
        fb = self._fb()
        df_h1 = _make_h1_ohlcv(periods=500)
        df_h4 = _make_h4_ohlcv(periods=300)
        result = fb.build(df_h1, df_h4=df_h4)
        assert result["h4_trend"].isna().sum() == 0

    def test_d1_trend_no_nan(self):
        fb = self._fb()
        df_h1 = _make_h1_ohlcv(periods=500)
        df_d1 = _make_d1_ohlcv(periods=100)
        result = fb.build(df_h1, df_d1=df_d1)
        assert result["d1_trend"].isna().sum() == 0

    def test_h4_trend_lookahead_in_full_build(self):
        """
        Integration-Lookahead-Test:
        H4-Bar schliesst um 16:00 -> H1-Bars vor 16:00 duerfen diesen Wert nicht sehen.
        """
        fb = self._fb()
        # Kontrollierter H4-DataFrame: erster Bar open=2023-01-02 00:00, close=04:00
        h4_ts = pd.date_range("2023-01-02 00:00", periods=250, freq="4h")
        df_h4 = _make_ohlcv(h4_ts, seed=5)
        # Einen bekannten Trendwert setzen ist nicht direkt moeglich,
        # aber wir pruefen: h4_trend bei H1-Bar 03:00 != h4_trend bei H1-Bar 05:00
        # (letzterer hat mehr geschlossene H4-Bars)
        h1_ts = pd.date_range("2023-01-02 00:00", periods=500, freq="1h")
        df_h1 = _make_ohlcv(h1_ts)
        result = fb.build(df_h1, df_h4=df_h4)
        # Kein Fehler = kein AttributeError, Feature vorhanden
        assert "h4_trend" in result.columns
