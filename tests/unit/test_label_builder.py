"""
Unit-Tests fuer LabelBuilder (Triple-Barrier-Methode).

Synthetische Preispfade mit bekanntem Ergebnis:
  - Eindeutig steigende Preise  -> Label  1 (TP getroffen)
  - Eindeutig fallende Preise   -> Label -1 (SL getroffen)
  - Seitwarts innerhalb Range   -> Label  0 (Zeitlimit)
  - Gemischte Sequenzen
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.label_builder import LabelBuilder


# ─────────────────────────────────────────────────────────────────────────────
#  Hilfsfunktionen fuer synthetische DataFrames
# ─────────────────────────────────────────────────────────────────────────────

ATR_VAL = 0.0100   # fester ATR fuer alle Testfaelle
TP_MULT = 2.0      # Standard-Multiplier
SL_MULT = 1.5


def _df(
    closes: list[float],
    highs:  list[float] | None = None,
    lows:   list[float] | None = None,
    atr:    float = ATR_VAL,
    atr_col: str = "atr_14",
) -> pd.DataFrame:
    """
    Baut einen minimalen OHLCV-DataFrame.
    Wenn highs/lows fehlen, wird high = close + 0.001 und low = close - 0.001
    gesetzt (engste moegliche Range ohne Barriereberuehrung bei ATR=0.01).
    """
    n = len(closes)
    if highs is None:
        highs = [c + 0.001 for c in closes]
    if lows is None:
        lows  = [c - 0.001 for c in closes]
    return pd.DataFrame({
        "close":  closes,
        "high":   highs,
        "low":    lows,
        "open":   closes,
        atr_col:  [atr] * n,
    })


def _builder(**kwargs) -> LabelBuilder:
    kw = {"tp_atr_mult": TP_MULT, "sl_atr_mult": SL_MULT, "max_candles": 5}
    kw.update(kwargs)
    return LabelBuilder(**kw)


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: Label-Grundlogik
# ─────────────────────────────────────────────────────────────────────────────

class TestTripleBarrierLabels:
    """Eindeutige Preispfade liefern vorhersagbare Labels."""

    def test_uptrend_yields_long_label(self):
        """Starker Aufwaertstrend: high ueberschreitet TP -> Label 1."""
        # close[0]=1.10, TP = 1.10 + 2*0.01 = 1.12, SL = 1.10 - 1.5*0.01 = 1.085
        # Naechste Kerze: high=1.13 > 1.12 -> TP getroffen
        df = _df(
            closes=[1.10, 1.115, 1.120, 1.125],
            highs =[1.10, 1.130, 1.135, 1.140],   # high[1]=1.13 > TP=1.12
            lows  =[1.09, 1.110, 1.115, 1.120],   # lows bleiben ueber SL=1.085
        )
        builder = _builder()
        labels = builder.build_labels(df)
        assert labels.iloc[0] == 1

    def test_downtrend_yields_short_label(self):
        """Starker Abwaertstrend: low unterschreitet SL -> Label -1."""
        # close[0]=1.10, SL = 1.10 - 1.5*0.01 = 1.085, TP = 1.12
        # Naechste Kerze: low=1.08 < 1.085 -> SL getroffen
        df = _df(
            closes=[1.10, 1.09, 1.08, 1.07],
            highs =[1.10, 1.09, 1.08, 1.08],   # highs bleiben unter TP=1.12
            lows  =[1.09, 1.08, 1.07, 1.07],   # low[1]=1.08 < SL=1.085
        )
        builder = _builder()
        labels = builder.build_labels(df)
        assert labels.iloc[0] == -1

    def test_sideways_yields_neutral_label(self):
        """Seitwartsbewegung: keine Barriere getroffen -> Label 0."""
        # close[0]=1.10, TP=1.12, SL=1.085
        # Alle Kerzen bleiben zwischen SL und TP
        n = 7   # max_candles=5, also 5 Kerzen nach t=0
        df = _df(
            closes=[1.10] * n,
            highs =[1.105] * n,   # < TP=1.12
            lows  =[1.095] * n,   # > SL=1.085
        )
        builder = _builder(max_candles=5)
        labels = builder.build_labels(df)
        assert labels.iloc[0] == 0

    def test_tp_hit_on_second_future_candle(self):
        """TP nicht sofort, aber in der zweiten Zukunftskerze getroffen."""
        # close[0]=1.10, TP=1.12
        # Kerze 1: high=1.11 < TP (kein Treffer)
        # Kerze 2: high=1.13 > TP -> Label 1
        df = _df(
            closes=[1.10, 1.11, 1.12, 1.13, 1.14],
            highs =[1.10, 1.11, 1.130, 1.13, 1.14],
            lows  =[1.09, 1.10, 1.110, 1.12, 1.13],
        )
        builder = _builder()
        labels = builder.build_labels(df)
        assert labels.iloc[0] == 1

    def test_sl_hit_on_second_future_candle(self):
        """SL nicht sofort, aber in der zweiten Zukunftskerze getroffen."""
        # close[0]=1.10, SL=1.085
        # Kerze 1: low=1.090 > SL (kein Treffer)
        # Kerze 2: low=1.080 < SL -> Label -1
        df = _df(
            closes=[1.10, 1.09, 1.085, 1.08, 1.07],
            highs =[1.10, 1.10, 1.090, 1.09, 1.08],
            lows  =[1.09, 1.090, 1.080, 1.08, 1.07],
        )
        builder = _builder()
        labels = builder.build_labels(df)
        assert labels.iloc[0] == -1

    def test_time_limit_reached_before_barrier(self):
        """Zeitlimit (max_candles=2) ohne Barriereberuehrung -> Label 0."""
        # close[0]=1.10, TP=1.12, SL=1.085
        # max_candles=2: nur Kerzen 1 und 2 betrachten
        df = _df(
            closes=[1.10, 1.105, 1.108, 1.13],  # Kerze 3 wuerde TP treffen, liegt aber ausserhalb
            highs =[1.10, 1.108, 1.110, 1.14],
            lows  =[1.09, 1.100, 1.102, 1.12],
        )
        builder = _builder(max_candles=2)
        labels = builder.build_labels(df)
        assert labels.iloc[0] == 0

    def test_simultaneous_tp_sl_hit_pessimistic(self):
        """Gleichzeitiger TP- und SL-Treffer -> pessimistisch Label -1."""
        # Kerze mit sehr hoher Range: low < SL und high > TP
        df = _df(
            closes=[1.10, 1.10],
            highs =[1.10, 1.15],   # > TP=1.12
            lows  =[1.10, 1.07],   # < SL=1.085
        )
        builder = _builder()
        labels = builder.build_labels(df)
        assert labels.iloc[0] == -1

    def test_exact_tp_boundary_is_hit(self):
        """High genau gleich TP -> TP getroffen (>=)."""
        # TP = 1.10 + 2*0.01 = 1.12; high = 1.12 genau
        df = _df(
            closes=[1.10, 1.11],
            highs =[1.10, 1.12],   # genau gleich TP
            lows  =[1.09, 1.10],
        )
        builder = _builder()
        labels = builder.build_labels(df)
        assert labels.iloc[0] == 1

    def test_exact_sl_boundary_is_hit(self):
        """Low genau gleich SL -> SL getroffen (<=)."""
        # SL = 1.10 - 1.5*0.01 = 1.085; low = 1.085 genau
        df = _df(
            closes=[1.10, 1.09],
            highs =[1.10, 1.09],
            lows  =[1.09, 1.085],   # genau gleich SL
        )
        builder = _builder()
        labels = builder.build_labels(df)
        assert labels.iloc[0] == -1


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: Rueckgabeformat
# ─────────────────────────────────────────────────────────────────────────────

class TestReturnFormat:

    def test_returns_series(self):
        df = _df([1.10, 1.11, 1.12])
        labels = _builder().build_labels(df)
        assert isinstance(labels, pd.Series)

    def test_series_name_is_label(self):
        df = _df([1.10, 1.11])
        labels = _builder().build_labels(df)
        assert labels.name == "label"

    def test_same_length_as_input(self):
        df = _df([1.0 + i * 0.01 for i in range(10)])
        labels = _builder().build_labels(df)
        assert len(labels) == len(df)

    def test_same_index_as_input(self):
        idx = pd.date_range("2024-01-01", periods=5, freq="h")
        df = _df([1.10] * 5)
        df.index = idx
        labels = _builder().build_labels(df)
        assert labels.index.equals(idx)

    def test_labels_only_contain_valid_values(self):
        n = 20
        closes = [1.10 + i * 0.002 for i in range(n)]
        df = _df(closes)
        labels = _builder().build_labels(df)
        assert set(labels.unique()).issubset({-1, 0, 1})

    def test_integer_dtype(self):
        df = _df([1.10, 1.11, 1.12])
        labels = _builder().build_labels(df)
        assert np.issubdtype(labels.dtype, np.integer)


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: Kein Look-ahead-Bias
# ─────────────────────────────────────────────────────────────────────────────

class TestNoLookaheadBias:

    def test_last_row_gets_zero_if_no_future_candles(self):
        """Letzte Kerze hat keine Zukunftskerzen -> Label 0."""
        df = _df([1.10, 1.11, 1.12])
        labels = _builder().build_labels(df)
        assert labels.iloc[-1] == 0

    def test_second_to_last_row_only_sees_one_future_candle(self):
        """Vorletzte Kerze bei max_candles=5 schaut nur eine Kerze voraus."""
        # close[-2]=1.10, TP=1.12 - naechste Kerze high=1.13 -> Label 1
        df = _df(
            closes=[1.10, 1.10, 1.115],
            highs =[1.10, 1.10, 1.130],
            lows  =[1.09, 1.09, 1.110],
        )
        builder = _builder(max_candles=5)
        labels = builder.build_labels(df)
        assert labels.iloc[1] == 1   # close[1]=1.10, high[2]=1.13 > TP=1.12


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: NaN und Randwerte
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_nan_atr_yields_zero_label(self):
        """NaN-ATR-Eintrag -> Label 0 (kein Barriereversuch)."""
        df = pd.DataFrame({
            "close":  [1.10, 1.10],
            "high":   [1.10, 1.15],
            "low":    [1.09, 1.05],
            "open":   [1.10, 1.10],
            "atr_14": [float("nan"), 0.01],
        })
        labels = _builder().build_labels(df)
        assert labels.iloc[0] == 0

    def test_zero_atr_yields_zero_label(self):
        """ATR=0 -> keine sinnvolle Barriere -> Label 0."""
        df = pd.DataFrame({
            "close":  [1.10, 1.20],
            "high":   [1.10, 1.30],
            "low":    [1.09, 1.00],
            "open":   [1.10, 1.10],
            "atr_14": [0.0, 0.01],
        })
        labels = _builder().build_labels(df)
        assert labels.iloc[0] == 0

    def test_single_row_df(self):
        """Einzeiliger DataFrame -> Label 0 (keine Zukunftskerzen)."""
        df = _df([1.10])
        labels = _builder().build_labels(df)
        assert labels.iloc[0] == 0

    def test_custom_atr_col_name(self):
        """Benutzerdefinierter ATR-Spaltenname wird korrekt verwendet."""
        df = pd.DataFrame({
            "close":   [1.10, 1.11],
            "high":    [1.10, 1.13],
            "low":     [1.09, 1.10],
            "open":    [1.10, 1.10],
            "my_atr":  [0.01, 0.01],
        })
        builder = LabelBuilder(tp_atr_mult=2.0, sl_atr_mult=1.5, max_candles=5, atr_col="my_atr")
        labels = builder.build_labels(df)
        assert labels.iloc[0] == 1  # high[1]=1.13 > TP=1.12


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: Validierung
# ─────────────────────────────────────────────────────────────────────────────

class TestValidation:

    def test_missing_close_column_raises(self):
        df = pd.DataFrame({"high": [1.1], "low": [1.0], "atr_14": [0.01]})
        with pytest.raises(ValueError, match="close"):
            _builder().build_labels(df)

    def test_missing_atr_column_raises(self):
        df = pd.DataFrame({"close": [1.1], "high": [1.1], "low": [1.0]})
        with pytest.raises(ValueError, match="atr_14"):
            _builder().build_labels(df)

    def test_empty_df_raises(self):
        df = pd.DataFrame({"close": [], "high": [], "low": [], "atr_14": []})
        with pytest.raises(ValueError, match="leer"):
            _builder().build_labels(df)

    def test_negative_tp_mult_raises(self):
        with pytest.raises(ValueError):
            LabelBuilder(tp_atr_mult=-1.0)

    def test_zero_sl_mult_raises(self):
        with pytest.raises(ValueError):
            LabelBuilder(sl_atr_mult=0.0)

    def test_zero_max_candles_raises(self):
        with pytest.raises(ValueError):
            LabelBuilder(max_candles=0)


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: Klassenverteilung
# ─────────────────────────────────────────────────────────────────────────────

class TestClassDistribution:

    def test_pure_uptrend_all_long(self):
        """Starker Aufwaertstrend: alle Labels sollten 1 oder 0 sein."""
        n = 10
        # Jede Kerze: high der naechsten Kerze ueberschreitet TP
        closes = [1.10 + i * 0.05 for i in range(n)]
        highs  = [c + 0.03 for c in closes]   # > TP = close + 0.02
        lows   = [c - 0.001 for c in closes]  # weit ueber SL
        df = _df(closes, highs, lows, atr=0.01)
        labels = _builder(max_candles=1).build_labels(df)
        # Alle ausser der letzten Kerze sollten 1 sein
        assert (labels.iloc[:-1] == 1).all()
        assert labels.iloc[-1] == 0

    def test_pure_downtrend_all_short(self):
        """Starker Abwaertstrend: alle Labels sollten -1 oder 0 sein."""
        n = 10
        closes = [1.10 - i * 0.05 for i in range(n)]
        highs  = [c + 0.001 for c in closes]  # weit unter TP
        lows   = [c - 0.02 for c in closes]   # < SL = close - 0.015
        df = _df(closes, highs, lows, atr=0.01)
        labels = _builder(max_candles=1).build_labels(df)
        assert (labels.iloc[:-1] == -1).all()
        assert labels.iloc[-1] == 0

    def test_flat_market_all_neutral(self):
        """Flacher Markt innerhalb TP/SL: alle Labels 0."""
        n = 10
        closes = [1.10] * n
        highs  = [1.105] * n   # < TP=1.12
        lows   = [1.095] * n   # > SL=1.085
        df = _df(closes, highs, lows, atr=0.01)
        labels = _builder(max_candles=5).build_labels(df)
        assert (labels == 0).all()

    def test_mixed_sequence_contains_all_classes(self):
        """Gemischte Sequenz enthaelt alle drei Label-Klassen."""
        # Aufbau: Kerze 0 -> TP, Kerze 3 -> SL, Rest neutral
        # close=1.10, ATR=0.01, TP=1.12, SL=1.085
        df = pd.DataFrame({
            "close":  [1.10, 1.115, 1.115, 1.10,  1.09,  1.09,  1.09,  1.09],
            "high":   [1.10, 1.130, 1.115, 1.105, 1.095, 1.095, 1.095, 1.095],
            "low":    [1.09, 1.110, 1.110, 1.090, 1.080, 1.088, 1.088, 1.088],
            "open":   [1.10] * 8,
            "atr_14": [0.01] * 8,
        })
        builder = _builder(max_candles=2)
        labels = builder.build_labels(df)
        unique = set(labels.unique())
        assert 1  in unique, "Keine Long-Labels gefunden"
        assert -1 in unique, "Keine Short-Labels gefunden"
        assert 0  in unique, "Keine Neutral-Labels gefunden"

    def test_distribution_logging_does_not_raise(self, caplog):
        """Logging der Klassenverteilung loest keinen Fehler aus."""
        import logging
        df = _df([1.10 + i * 0.001 for i in range(10)])
        with caplog.at_level(logging.INFO):
            _builder().build_labels(df)
        # Kein AssertionError oder Exception


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: Konfiguration
# ─────────────────────────────────────────────────────────────────────────────

class TestConfiguration:

    def test_custom_tp_multiplier(self):
        """Groesserer TP-Multiplier -> TP schwieriger zu erreichen."""
        # Mit tp_mult=5.0: TP = 1.10 + 5*0.01 = 1.15 -> high=1.13 reicht nicht
        df = _df(
            closes=[1.10, 1.11],
            highs =[1.10, 1.13],
            lows  =[1.09, 1.09],
        )
        builder = LabelBuilder(tp_atr_mult=5.0, sl_atr_mult=1.5, max_candles=1)
        labels = builder.build_labels(df)
        assert labels.iloc[0] == 0  # TP nicht erreicht, kein SL -> Zeitlimit

    def test_custom_sl_multiplier(self):
        """Groesserer SL-Multiplier -> SL schwieriger zu unterschreiten."""
        # Mit sl_mult=5.0: SL = 1.10 - 5*0.01 = 1.05 -> low=1.08 > SL
        df = _df(
            closes=[1.10, 1.09],
            highs =[1.10, 1.09],
            lows  =[1.09, 1.08],
        )
        builder = LabelBuilder(tp_atr_mult=2.0, sl_atr_mult=5.0, max_candles=1)
        labels = builder.build_labels(df)
        assert labels.iloc[0] == 0  # SL nicht unterschritten, kein TP -> Zeitlimit

    def test_max_candles_one_limits_lookahead(self):
        """max_candles=1 schaut nur eine Kerze voraus."""
        # close[0]=1.10, TP=1.12; Kerze 1: high=1.11 < TP; Kerze 2: high=1.15 > TP
        df = _df(
            closes=[1.10, 1.11, 1.12],
            highs =[1.10, 1.11, 1.15],
            lows  =[1.09, 1.10, 1.11],
        )
        builder = _builder(max_candles=1)
        labels = builder.build_labels(df)
        assert labels.iloc[0] == 0   # Kerze 2 liegt ausserhalb max_candles=1
