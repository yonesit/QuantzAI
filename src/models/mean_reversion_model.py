"""
src/models/mean_reversion_model.py
MeanReversionModel – ML-basiertes Mean-Reversion-Modell fuer QuantzAI.

Unterschied zum SignalModel (Trendfolge):
  Feature-Set:  Standard-23-Baseline + 3 MR-spezifische Zusatz-Features
  Labels:       Triple-Barrier mit engem TP und weitem SL (MR-Profil)
  Horizont:     Kuerzer (max_candles=10 auf H4 = ~2 Handelstage)

MR-spezifische Features (alle look-ahead-frei):
  bb_pct_b       = (close_T – bb_lower_{T-1}) / (bb_upper_{T-1} – bb_lower_{T-1})
                   Position im Bollinger-Band [0..1]. Extremwerte = MR-Signal.
  dist_ema20_atr = (close_T – ema20_{T-1}) / atr14_{T-1}
                   Vorzeichenbehaftete EMA20-Distanz in ATR-Einheiten.
  dist_sma50_atr = (close_T – sma50_{T-1}) / atr14_{T-1}
                   Vorzeichenbehaftete SMA50-Distanz in ATR-Einheiten.

Label-Parameter (MR vs. TF-Standard):
  tp_atr_mult = 1.0   (TF: 2.0)  MR-Moves sind kleiner
  sl_atr_mult = 2.0   (TF: 1.5)  Trend stoppt MR-Trade erst spaet
  max_candles = 10    (TF: 24)   Keine Reversion in 10 Bars -> neutral
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any

from loguru import logger

from src.data.feature_builder import FeatureBuilder
from src.models.label_builder import LabelBuilder
from src.models.signal_model import SignalModel


# ─────────────────────────────────────────────────────────────────────────────
#  MeanReversionModel
# ─────────────────────────────────────────────────────────────────────────────

class MeanReversionModel:
    """
    ML-basiertes Mean-Reversion-Modell.

    Baut auf SignalModel (LightGBM) auf; ersetzt nur Feature-Engineering
    und Label-Parameter durch MR-spezifische Varianten.

    Parameters
    ----------
    lgbm_params : optionale LightGBM-Hyperparameter (wie SignalModel).
    """

    MR_LABEL_PARAMS: dict[str, float | int] = {
        "tp_atr_mult": 1.0,
        "sl_atr_mult": 2.0,
        "max_candles": 10,
    }
    MR_FEATURE_NAMES: list[str] = ["bb_pct_b", "dist_ema20_atr", "dist_sma50_atr"]

    def __init__(self, lgbm_params: dict[str, Any] | None = None) -> None:
        self._model = SignalModel(lgbm_params=lgbm_params)

    # ── Feature-Engineering ───────────────────────────────────────────────────

    def build_features(
        self,
        df: pd.DataFrame,
        symbol: str = "",
        timeframe: str = "",
    ) -> pd.DataFrame:
        """
        Erzeugt 26-Feature-Matrix (Standard-23 + 3 MR-spezifische).

        Parameters
        ----------
        df        : Validierter OHLCV-DataFrame (timestamp, open, high, low,
                    close, volume). Wird unveraendert an FeatureBuilder weitergegeben.
        symbol    : Symbol-Name (fuer Logging).
        timeframe : Timeframe-String (fuer Logging).

        Returns
        -------
        DataFrame mit 26 Feature-Spalten plus Struktur-Spalten
        (timestamp, close, high, low).

        Look-ahead-Freiheit
        -------------------
        Die 3 zusaetzlichen MR-Features nutzen ausschliesslich T-1-Indikatoren
        (bb_lower_{T-1}, bb_upper_{T-1}, ema_20_{T-1}, sma_50_{T-1},
        atr_14_{T-1}), die FeatureBuilder bereits mit .shift(1) berechnet.
        Verglichen wird jeweils mit dem aktuellen Schlusskurs close_T, der
        fuer den LabelBuilder als Referenzpreis dient. Kein Look-ahead.
        """
        # 1. Standard-23-Features
        builder = FeatureBuilder()
        features = builder.build(df, symbol=symbol, timeframe=timeframe,
                                 df_h4=None, df_d1=None)

        # 2. MR-spezifische Features aus bereits verschobenen Spalten
        features = self._add_mr_features(features)

        feat_cols = [c for c in features.columns
                     if c not in {"timestamp", "close", "high", "low"}]
        logger.info(
            "MeanReversionModel | {sym} {tf} | {n} Features | {r} Zeilen",
            sym=symbol, tf=timeframe, n=len(feat_cols), r=len(features),
        )
        return features

    def _add_mr_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Fuegt bb_pct_b, dist_ema20_atr, dist_sma50_atr hinzu."""
        close        = features["close"]           # T+0 (Referenzpreis)
        bb_upper_tm1 = features["bb_upper"]        # T-1 (bereits geshiftet)
        bb_lower_tm1 = features["bb_lower"]        # T-1
        ema20_tm1    = features["ema_20"]          # T-1
        sma50_tm1    = features["sma_50"]          # T-1
        atr14_tm1    = features["atr_14"]          # T-1

        band_width = (bb_upper_tm1 - bb_lower_tm1).where(
            lambda x: x > 0, other=np.nan
        )
        safe_atr = atr14_tm1.where(atr14_tm1 > 0, other=np.nan)

        # BB %B: Position im Band [0..1]; >1 = oberhalb, <0 = unterhalb
        features["bb_pct_b"] = (close - bb_lower_tm1) / band_width

        # Normierte EMA20-Distanz (positiv = oberhalb EMA -> Short-Signal)
        features["dist_ema20_atr"] = (close - ema20_tm1) / safe_atr

        # Normierte SMA50-Distanz (laengerer Kontext)
        features["dist_sma50_atr"] = (close - sma50_tm1) / safe_atr

        return features

    # ── Label-Builder ─────────────────────────────────────────────────────────

    @classmethod
    def default_label_builder(cls) -> LabelBuilder:
        """Gibt LabelBuilder mit MR-spezifischen Parametern zurueck."""
        return LabelBuilder(**cls.MR_LABEL_PARAMS)

    # ── Training & Inferenz (Delegation an SignalModel) ───────────────────────

    def train(
        self,
        features_df: pd.DataFrame,
        labels: pd.Series,
    ) -> dict[str, Any]:
        """
        Trainiert das interne SignalModel.

        Parameters
        ----------
        features_df : Feature-Matrix OHNE timestamp-Spalte.
        labels      : Series mit Labels {-1, 0, 1}.
        """
        return self._model.train(features_df, labels)

    def predict_proba(
        self, features_row: pd.DataFrame | pd.Series
    ) -> dict[str, float]:
        """Gibt Klassenwahrscheinlichkeiten zurueck (delegiert an SignalModel)."""
        return self._model.predict_proba(features_row)

    def get_signal(
        self,
        features_row: pd.DataFrame | pd.Series,
        confidence_threshold: float = 0.55,
    ) -> str:
        """Gibt Handelssignal zurueck: 'long', 'short' oder 'flat'."""
        return self._model.get_signal(features_row, confidence_threshold)

    # ── Walk-Forward ──────────────────────────────────────────────────────────

    def walk_forward_validate(
        self,
        features_df: pd.DataFrame,
        labels: pd.Series,
        timestamp_col: str = "timestamp",
        train_months: int = 6,
        test_months: int = 1,
    ) -> list[dict[str, Any]]:
        """
        Rollierendes Walk-Forward-Backtesting (delegiert an SignalModel).

        Parameters identisch zu SignalModel.walk_forward_validate().
        """
        return self._model.walk_forward_validate(
            features_df,
            labels,
            timestamp_col=timestamp_col,
            train_months=train_months,
            test_months=test_months,
        )

    # ── Persistenz ────────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Speichert das interne SignalModel."""
        self._model.save(path)

    @classmethod
    def load(cls, path: str | Path) -> "MeanReversionModel":
        """Laedt ein gespeichertes MeanReversionModel."""
        instance = cls()
        instance._model = SignalModel.load(path)
        return instance

    # ── Eigenschaften ─────────────────────────────────────────────────────────

    @property
    def feature_names(self) -> list[str]:
        """Feature-Namen (aus internem SignalModel nach Training)."""
        return self._model._feature_names

    @property
    def n_features(self) -> int:
        """Anzahl Features (Standard-23 + 3 MR = 26)."""
        return 23 + len(self.MR_FEATURE_NAMES)
