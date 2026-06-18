"""
tests/unit/test_regime_detector.py
Unit-Tests fuer RegimeDetector.

Abgedeckt:
  - Regime-Konstanten (Enum-Werte, String-Kompatibilitaet)
  - Konstruktor: Standardwerte, ungueltige Parameter
  - detect_regime: TRENDING, RANGING, HIGH_VOLATILITY je eindeutig
  - Prioritaet: HIGH_VOLATILITY schlaegt TRENDING (hoher ADX + hoher ATR)
  - ADX-Grenzwert: genau am Schwellwert -> TRENDING
  - NaN-Behandlung: ADX-NaN -> 0.0 (RANGING), ATR-NaN -> kein HIGH_VOL
  - Leerer DataFrame / fehlende Spalten -> ValueError
  - Eigene Spaltennamen (adx_col / atr_col konfigurierbar)
  - atr_window: Durchschnitt nur aus letzten N Zeilen
  - atr_window groesser als DataFrame -> kein Fehler
  - Regimewechsel-Logging: last_regime-Property
  - Kein Log bei gleichem Regime (kein doppeltes Protokoll)
  - get_position_size_multiplier: HIGH_VOL=0.5, sonst 1.0
  - Sequenz mehrerer Erkennungen: korrekte Zustandsaenderungen
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

from src.models.regime_detector import Regime, RegimeDetector


# ─────────────────────────────────────────────────────────────────────────────
#  Synthetische Feature-DataFrames
# ─────────────────────────────────────────────────────────────────────────────

def _df(
    adx: float,
    atr_last: float,
    atr_history: list[float] | None = None,
    adx_col: str = "adx",
    atr_col: str = "atr_14",
    n_extra: int = 0,
) -> pd.DataFrame:
    """
    Hilfsfunktion: baut einen minimalen features-DataFrame.

    atr_history: Liste historischer ATR-Werte VOR atr_last.
                 Wenn None: nur atr_last ohne Geschichte.
    n_extra: n_extra zusaetzliche Zeilen mit atr = atr_last / 2 am Anfang
             (zur Pruefung des atr_window-Verhaltens).
    """
    rows = []
    if atr_history:
        for h in atr_history:
            rows.append({adx_col: adx, atr_col: h})
    if n_extra:
        for _ in range(n_extra):
            rows.append({adx_col: adx, atr_col: atr_last / 2})
    rows.append({adx_col: adx, atr_col: atr_last})
    return pd.DataFrame(rows)


def _trending_df(adx: float = 30.0, atr: float = 0.001) -> pd.DataFrame:
    """ADX klar über Schwellwert, ATR normal -> TRENDING."""
    history = [atr] * 49  # 49 Eintraege mit normalem ATR
    return _df(adx=adx, atr_last=atr, atr_history=history)


def _ranging_df(adx: float = 10.0, atr: float = 0.001) -> pd.DataFrame:
    """ADX klar unter Schwellwert, ATR normal -> RANGING."""
    history = [atr] * 49
    return _df(adx=adx, atr_last=atr, atr_history=history)


def _high_vol_df(
    adx: float = 10.0,
    atr_last: float = 0.003,
    atr_normal: float = 0.001,
    multiplier_factor: float = 2.0,
) -> pd.DataFrame:
    """ATR deutlich ueber historischem Durchschnitt -> HIGH_VOLATILITY."""
    history = [atr_normal] * 49
    return _df(adx=adx, atr_last=atr_last, atr_history=history)


# ─────────────────────────────────────────────────────────────────────────────
#  Regime-Enum
# ─────────────────────────────────────────────────────────────────────────────

class TestRegimeEnum:
    def test_string_values(self):
        assert Regime.TRENDING.value        == "TRENDING"
        assert Regime.RANGING.value         == "RANGING"
        assert Regime.HIGH_VOLATILITY.value == "HIGH_VOLATILITY"

    def test_str_enum_comparison(self):
        assert Regime.TRENDING == "TRENDING"
        assert Regime.RANGING  == "RANGING"
        assert Regime.HIGH_VOLATILITY == "HIGH_VOLATILITY"

    def test_all_three_exist(self):
        regimes = {r.value for r in Regime}
        assert regimes == {"TRENDING", "RANGING", "HIGH_VOLATILITY"}


# ─────────────────────────────────────────────────────────────────────────────
#  Konstruktor
# ─────────────────────────────────────────────────────────────────────────────

class TestConstructor:
    def test_defaults(self):
        rd = RegimeDetector()
        assert rd._adx_threshold  == 25.0
        assert rd._atr_multiplier == 1.5
        assert rd._atr_window     == 50
        assert rd._adx_col        == "adx"
        assert rd._atr_col        == "atr_14"
        assert rd.last_regime is None

    def test_custom_parameters(self):
        rd = RegimeDetector(
            adx_trending_threshold=20.0,
            atr_volatility_multiplier=2.0,
            atr_window=30,
            adx_col="my_adx",
            atr_col="my_atr",
        )
        assert rd._adx_threshold  == 20.0
        assert rd._atr_multiplier == 2.0
        assert rd._atr_window     == 30
        assert rd._adx_col        == "my_adx"
        assert rd._atr_col        == "my_atr"

    def test_negative_adx_threshold_raises(self):
        with pytest.raises(ValueError, match="adx_trending_threshold"):
            RegimeDetector(adx_trending_threshold=-1.0)

    def test_zero_atr_multiplier_raises(self):
        with pytest.raises(ValueError, match="atr_volatility_multiplier"):
            RegimeDetector(atr_volatility_multiplier=0.0)

    def test_negative_atr_multiplier_raises(self):
        with pytest.raises(ValueError, match="atr_volatility_multiplier"):
            RegimeDetector(atr_volatility_multiplier=-0.5)

    def test_zero_atr_window_raises(self):
        with pytest.raises(ValueError, match="atr_window"):
            RegimeDetector(atr_window=0)

    def test_adx_threshold_zero_allowed(self):
        rd = RegimeDetector(adx_trending_threshold=0.0)
        assert rd._adx_threshold == 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  detect_regime – die drei Regime
# ─────────────────────────────────────────────────────────────────────────────

class TestDetectRegime:
    def test_trending(self):
        rd = RegimeDetector(adx_trending_threshold=25.0, atr_volatility_multiplier=1.5)
        assert rd.detect_regime(_trending_df(adx=30.0)) == "TRENDING"

    def test_ranging(self):
        rd = RegimeDetector(adx_trending_threshold=25.0, atr_volatility_multiplier=1.5)
        assert rd.detect_regime(_ranging_df(adx=10.0)) == "RANGING"

    def test_high_volatility(self):
        rd = RegimeDetector(adx_trending_threshold=25.0, atr_volatility_multiplier=1.5)
        # ATR 3x höher als Durchschnitt -> HIGH_VOLATILITY
        features = _high_vol_df(adx=10.0, atr_last=0.003, atr_normal=0.001)
        assert rd.detect_regime(features) == "HIGH_VOLATILITY"

    def test_high_volatility_overrides_trending(self):
        """HIGH_VOLATILITY hat Prioritaet auch wenn ADX > Schwellwert."""
        rd = RegimeDetector(adx_trending_threshold=25.0, atr_volatility_multiplier=1.5)
        # Hoher ADX UND hoher ATR -> HIGH_VOLATILITY gewinnt
        features = _high_vol_df(adx=40.0, atr_last=0.003, atr_normal=0.001)
        assert rd.detect_regime(features) == "HIGH_VOLATILITY"

    def test_adx_exactly_at_threshold_is_trending(self):
        """ADX genau am Schwellwert gilt als TRENDING (>=)."""
        rd = RegimeDetector(adx_trending_threshold=25.0)
        features = _trending_df(adx=25.0)
        assert rd.detect_regime(features) == "TRENDING"

    def test_adx_just_below_threshold_is_ranging(self):
        """ADX knapp unter Schwellwert -> RANGING."""
        rd = RegimeDetector(adx_trending_threshold=25.0)
        features = _ranging_df(adx=24.99)
        assert rd.detect_regime(features) == "RANGING"

    def test_returns_string(self):
        rd = RegimeDetector()
        result = rd.detect_regime(_trending_df())
        assert isinstance(result, str)

    def test_result_is_valid_regime(self):
        rd = RegimeDetector()
        valid = {r.value for r in Regime}
        assert rd.detect_regime(_trending_df()) in valid
        assert rd.detect_regime(_ranging_df()) in valid
        assert rd.detect_regime(_high_vol_df()) in valid


# ─────────────────────────────────────────────────────────────────────────────
#  NaN-Behandlung
# ─────────────────────────────────────────────────────────────────────────────

class TestNanHandling:
    def test_nan_adx_treated_as_zero_gives_ranging(self):
        """NaN in ADX wird als 0.0 behandelt -> kein Trend."""
        df = pd.DataFrame({"adx": [float("nan")], "atr_14": [0.001]})
        rd = RegimeDetector(adx_trending_threshold=25.0)
        assert rd.detect_regime(df) == "RANGING"

    def test_nan_atr_no_high_volatility(self):
        """Alle ATR-Werte NaN -> kein HIGH_VOLATILITY, Entscheidung via ADX."""
        df = pd.DataFrame({"adx": [30.0], "atr_14": [float("nan")]})
        rd = RegimeDetector(adx_trending_threshold=25.0)
        result = rd.detect_regime(df)
        assert result in ("TRENDING", "RANGING")
        assert result != "HIGH_VOLATILITY"

    def test_nan_atr_with_high_adx_gives_trending(self):
        """Kein ATR -> kein HIGH_VOL, aber hoher ADX -> TRENDING."""
        df = pd.DataFrame({"adx": [40.0], "atr_14": [float("nan")]})
        rd = RegimeDetector(adx_trending_threshold=25.0)
        assert rd.detect_regime(df) == "TRENDING"

    def test_mixed_nan_atr_uses_valid_rows(self):
        """Historische NaN-ATR-Werte werden ignoriert, gueltige Werte werden verwendet."""
        df = pd.DataFrame({
            "adx":    [10.0, 10.0, 10.0, 10.0],
            "atr_14": [float("nan"), 0.001, 0.001, 0.003],
        })
        rd = RegimeDetector(adx_trending_threshold=25.0, atr_volatility_multiplier=1.5)
        # mean(0.001, 0.001, 0.003) = 0.001667, current=0.003
        # 0.003 > 1.5 * 0.001667 = 0.0025 -> HIGH_VOLATILITY
        result = rd.detect_regime(df)
        assert result == "HIGH_VOLATILITY"


# ─────────────────────────────────────────────────────────────────────────────
#  Fehlerbehandlung
# ─────────────────────────────────────────────────────────────────────────────

class TestValidation:
    def test_empty_df_raises(self):
        rd = RegimeDetector()
        with pytest.raises(ValueError, match="leer"):
            rd.detect_regime(pd.DataFrame())

    def test_none_raises(self):
        rd = RegimeDetector()
        with pytest.raises((ValueError, AttributeError)):
            rd.detect_regime(None)

    def test_missing_adx_col_raises(self):
        df = pd.DataFrame({"atr_14": [0.001]})
        rd = RegimeDetector()
        with pytest.raises(ValueError, match="adx"):
            rd.detect_regime(df)

    def test_missing_atr_col_raises(self):
        df = pd.DataFrame({"adx": [20.0]})
        rd = RegimeDetector()
        with pytest.raises(ValueError, match="atr_14"):
            rd.detect_regime(df)

    def test_missing_both_cols_raises(self):
        df = pd.DataFrame({"close": [1.1], "rsi": [50.0]})
        rd = RegimeDetector()
        with pytest.raises(ValueError):
            rd.detect_regime(df)


# ─────────────────────────────────────────────────────────────────────────────
#  Eigene Spaltennamen
# ─────────────────────────────────────────────────────────────────────────────

class TestCustomColumnNames:
    def test_custom_adx_col(self):
        rd = RegimeDetector(adx_col="my_adx", atr_col="atr_14")
        df = pd.DataFrame({"my_adx": [30.0], "atr_14": [0.001]})
        assert rd.detect_regime(df) == "TRENDING"

    def test_custom_atr_col(self):
        rd = RegimeDetector(adx_col="adx", atr_col="my_atr")
        df = pd.DataFrame({"adx": [10.0], "my_atr": [0.001]})
        assert rd.detect_regime(df) == "RANGING"

    def test_custom_atr_col_high_vol(self):
        rd = RegimeDetector(adx_col="adx", atr_col="custom_atr",
                            atr_volatility_multiplier=1.5)
        rows = [{"adx": 10.0, "custom_atr": 0.001}] * 49
        rows.append({"adx": 10.0, "custom_atr": 0.003})
        df = pd.DataFrame(rows)
        assert rd.detect_regime(df) == "HIGH_VOLATILITY"

    def test_wrong_default_col_raises_on_custom_df(self):
        """Wenn features_df andere Spaltennamen hat, ohne Konfiguration -> ValueError."""
        rd = RegimeDetector()  # erwartet 'adx' und 'atr_14'
        df = pd.DataFrame({"adx_indicator": [30.0], "avg_true_range": [0.001]})
        with pytest.raises(ValueError):
            rd.detect_regime(df)


# ─────────────────────────────────────────────────────────────────────────────
#  ATR-Fenster-Logik
# ─────────────────────────────────────────────────────────────────────────────

class TestAtrWindow:
    def test_window_limits_lookback(self):
        """Nur die letzten atr_window Zeilen beeinflussen den ATR-Durchschnitt."""
        # 100 Zeilen mit ATR=0.001, dann 1 Zeile mit ATR=0.002
        # window=10: Durchschnitt nur aus letzten 10 Zeilen = 0.001
        # 0.002 > 1.5 * 0.001 = 0.0015 -> HIGH_VOLATILITY
        history = [0.001] * 100
        df = _df(adx=10.0, atr_last=0.002, atr_history=history)
        rd = RegimeDetector(adx_trending_threshold=25.0,
                            atr_volatility_multiplier=1.5,
                            atr_window=10)
        assert rd.detect_regime(df) == "HIGH_VOLATILITY"

    def test_window_larger_than_df_no_error(self):
        """atr_window groesser als verfuegbare Zeilen -> kein Fehler, nutzt alle."""
        df = pd.DataFrame({"adx": [10.0, 10.0], "atr_14": [0.001, 0.001]})
        rd = RegimeDetector(atr_window=1000)
        result = rd.detect_regime(df)
        assert result in {r.value for r in Regime}

    def test_single_row_df(self):
        """Einzel-Zeile DataFrame: ATR-Durchschnitt = ATR-Wert selbst -> nie HIGH_VOL."""
        df = pd.DataFrame({"adx": [10.0], "atr_14": [0.001]})
        rd = RegimeDetector(atr_volatility_multiplier=1.5)
        # mean == current -> current / mean == 1.0 < 1.5 -> kein HIGH_VOL
        result = rd.detect_regime(df)
        assert result != "HIGH_VOLATILITY"

    def test_large_history_no_influence_outside_window(self):
        """Sehr alte hohe ATR-Werte ausserhalb des Fensters beeinflussen Ergebnis nicht."""
        # Erste 100 Zeilen: extrem hoher ATR
        # Letzte 10 Zeilen: normaler ATR (window=10)
        # Aktueller ATR: normal -> kein HIGH_VOL
        history_old  = [0.1] * 100   # weit ausserhalb des window
        history_new  = [0.001] * 9   # im window (total window=10)
        history = history_old + history_new
        df = _df(adx=10.0, atr_last=0.0015, atr_history=history)
        rd = RegimeDetector(atr_volatility_multiplier=1.5, atr_window=10)
        # mean der letzten 10 = mean([0.001]*9 + [0.0015]) = 0.001050
        # 0.0015 > 1.5 * 0.001050 = 0.001575? 0.0015 < 0.001575 -> kein HIGH_VOL
        result = rd.detect_regime(df)
        assert result != "HIGH_VOLATILITY"


# ─────────────────────────────────────────────────────────────────────────────
#  Regimewechsel-Logging / last_regime
# ─────────────────────────────────────────────────────────────────────────────

class TestRegimeChangeLogging:
    def test_last_regime_none_initially(self):
        rd = RegimeDetector()
        assert rd.last_regime is None

    def test_last_regime_set_after_first_call(self):
        rd = RegimeDetector()
        rd.detect_regime(_trending_df())
        assert rd.last_regime == "TRENDING"

    def test_last_regime_updates_on_change(self):
        rd = RegimeDetector()
        rd.detect_regime(_trending_df(adx=30.0))
        assert rd.last_regime == "TRENDING"
        rd.detect_regime(_ranging_df(adx=10.0))
        assert rd.last_regime == "RANGING"

    def test_last_regime_stays_same_without_change(self):
        rd = RegimeDetector()
        rd.detect_regime(_trending_df())
        rd.detect_regime(_trending_df())
        assert rd.last_regime == "TRENDING"

    def test_sequence_trending_ranging_highvol(self):
        rd = RegimeDetector(adx_trending_threshold=25.0, atr_volatility_multiplier=1.5)

        rd.detect_regime(_trending_df(adx=30.0))
        assert rd.last_regime == "TRENDING"

        rd.detect_regime(_ranging_df(adx=10.0))
        assert rd.last_regime == "RANGING"

        rd.detect_regime(_high_vol_df(adx=10.0, atr_last=0.003, atr_normal=0.001))
        assert rd.last_regime == "HIGH_VOLATILITY"

    def test_regime_change_logged(self):
        """Regimewechsel erzeugt einen Log-Eintrag (via loguru-Sink)."""
        import io
        from loguru import logger

        output = io.StringIO()
        handler_id = logger.add(output, format="{message}", level="INFO")
        try:
            rd = RegimeDetector()
            rd.detect_regime(_ranging_df())
            rd.detect_regime(_trending_df())
        finally:
            logger.remove(handler_id)
        assert "Regimewechsel" in output.getvalue()

    def test_initial_regime_logged(self):
        """Erstes Regime wird als 'Initiales Regime' geloggt (via loguru-Sink)."""
        import io
        from loguru import logger

        output = io.StringIO()
        handler_id = logger.add(output, format="{message}", level="INFO")
        try:
            rd = RegimeDetector()
            rd.detect_regime(_ranging_df())
        finally:
            logger.remove(handler_id)
        assert "Initiales" in output.getvalue()

    def test_no_log_when_regime_unchanged(self):
        """Kein 'Regime'-Log wenn sich das Regime nicht aendert."""
        import io
        from loguru import logger

        rd = RegimeDetector()
        rd.detect_regime(_trending_df())  # erstes Mal -> Log (ausserhalb Pruefung)

        output = io.StringIO()
        handler_id = logger.add(output, format="{message}", level="INFO")
        try:
            rd.detect_regime(_trending_df())  # gleiches Regime -> kein Log
        finally:
            logger.remove(handler_id)
        assert "Regime" not in output.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
#  get_position_size_multiplier (Orchestrator-Hook)
# ─────────────────────────────────────────────────────────────────────────────

class TestPositionSizeMultiplier:
    def test_high_vol_returns_half(self):
        rd = RegimeDetector()
        assert rd.get_position_size_multiplier("HIGH_VOLATILITY") == pytest.approx(0.5)

    def test_trending_returns_one(self):
        rd = RegimeDetector()
        assert rd.get_position_size_multiplier("TRENDING") == pytest.approx(1.0)

    def test_ranging_returns_one(self):
        rd = RegimeDetector()
        assert rd.get_position_size_multiplier("RANGING") == pytest.approx(1.0)

    def test_unknown_regime_returns_one(self):
        rd = RegimeDetector()
        assert rd.get_position_size_multiplier("UNKNOWN") == pytest.approx(1.0)

    def test_multiplier_with_detected_regime(self):
        """Kombinierter Test: detect_regime + get_position_size_multiplier."""
        rd = RegimeDetector(adx_trending_threshold=25.0, atr_volatility_multiplier=1.5)
        regime = rd.detect_regime(_high_vol_df(atr_last=0.003, atr_normal=0.001))
        assert regime == "HIGH_VOLATILITY"
        assert rd.get_position_size_multiplier(regime) == pytest.approx(0.5)

    def test_trending_multiplier_from_detection(self):
        rd = RegimeDetector()
        regime = rd.detect_regime(_trending_df())
        assert rd.get_position_size_multiplier(regime) == pytest.approx(1.0)


# ─────────────────────────────────────────────────────────────────────────────
#  Synthetische Preispfade (realistischere Daten)
# ─────────────────────────────────────────────────────────────────────────────

class TestSyntheticPricePaths:
    """
    Prueft dass der Detektor mit realistisch aussehenden Feature-DataFrames
    (wie vom FeatureBuilder erzeugt) korrekt arbeitet.
    """

    def _build_features(
        self,
        n: int,
        adx_values: list[float],
        atr_values: list[float],
    ) -> pd.DataFrame:
        """Minimal-Abbild eines FeatureBuilder-Outputs."""
        assert len(adx_values) == n and len(atr_values) == n
        return pd.DataFrame({
            "adx":   adx_values,
            "atr_14": atr_values,
            "close": [1.1] * n,   # weitere Spalten duerfen existieren
            "rsi_14": [50.0] * n,
        })

    def test_clear_trending_market(self):
        """Klar trendendes Regime: ADX steigt auf 35."""
        n = 60
        adx = [15.0] * 20 + [25.0] * 20 + [35.0] * 20
        atr = [0.001] * n
        df = self._build_features(n, adx, atr)
        rd = RegimeDetector(adx_trending_threshold=25.0, atr_volatility_multiplier=1.5)
        assert rd.detect_regime(df) == "TRENDING"

    def test_clear_ranging_market(self):
        """Seitwärts-Regime: ADX dauerhaft niedrig."""
        n = 60
        adx = [10.0] * n
        atr = [0.001] * n
        df = self._build_features(n, adx, atr)
        rd = RegimeDetector(adx_trending_threshold=25.0)
        assert rd.detect_regime(df) == "RANGING"

    def test_volatility_spike_market(self):
        """Volatilitaets-Spike: ATR springt auf das 3-Fache."""
        n = 60
        adx = [20.0] * n
        atr = [0.001] * 59 + [0.003]  # Spike in der letzten Kerze
        df = self._build_features(n, adx, atr)
        rd = RegimeDetector(
            adx_trending_threshold=25.0,
            atr_volatility_multiplier=1.5,
            atr_window=50,
        )
        assert rd.detect_regime(df) == "HIGH_VOLATILITY"

    def test_extra_columns_ignored(self):
        """Fremde Spalten beeinflussen die Erkennung nicht."""
        df = pd.DataFrame({
            "adx":    [30.0],
            "atr_14": [0.001],
            "ema_9":  [1.095],
            "rsi_14": [62.0],
            "macd":   [0.0003],
        })
        rd = RegimeDetector()
        assert rd.detect_regime(df) == "TRENDING"

    def test_transition_ranging_to_trending(self):
        """Regime-Transition vom Range- in Trend-Regime."""
        rd = RegimeDetector(adx_trending_threshold=25.0)
        ranging_df  = self._build_features(50, [10.0]*50, [0.001]*50)
        trending_df = self._build_features(50, [35.0]*50, [0.001]*50)

        r1 = rd.detect_regime(ranging_df)
        r2 = rd.detect_regime(trending_df)
        assert r1 == "RANGING"
        assert r2 == "TRENDING"
        assert rd.last_regime == "TRENDING"
