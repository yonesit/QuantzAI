"""
src/data/feature_builder.py
FeatureBuilder â€" erzeugt eine saubere Feature-Matrix aus OHLCV-Daten.

Indikatoren (alle mit .shift(1) â€" kein Look-ahead Bias):
  Trend:       EMA (9,20,50,200), SMA (20,50), MACD (12,26,9), ADX (14)
  Momentum:    RSI (14), Stochastic (14,3,3), CCI (20), Williams %R (14)
  Volatilitaet: ATR (14), Bollinger Bands (20,2), Keltner Channel (20,2)
  Volumen:     OBV
  Abgeleitet:  candle_body, candle_direction, high_low_range,
               close_position, hour_of_day, day_of_week

Bibliothek: ta (pip install ta)
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import numpy as np
import yaml
from loguru import logger

import ta
from ta.trend import (
    EMAIndicator, SMAIndicator,
    MACD, ADXIndicator,
)
from ta.momentum import (
    RSIIndicator, StochasticOscillator,
    WilliamsRIndicator,
)
from ta.trend import CCIIndicator
from ta.volatility import (
    AverageTrueRange, BollingerBands, KeltnerChannel,
)
from ta.volume import OnBalanceVolumeIndicator


# â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
#  Exceptions
# â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

class FeatureBuilderError(Exception):
    """Wird ausgeloest wenn Feature-Berechnung fehlschlaegt."""


# â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
#  FeatureBuilder
# â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

class FeatureBuilder:
    """
    Berechnet technische Indikatoren und abgeleitete Features.

    Parameters
    ----------
    ema_periods         : EMA-Perioden (Standard: [9, 20, 50, 200])
    sma_periods         : SMA-Perioden (Standard: [20, 50])
    rsi_periods         : RSI-Perioden (Standard: [14])
    atr_period          : ATR-Periode  (Standard: 14)
    bollinger_period    : Bollinger-Periode (Standard: 20)
    bollinger_std       : Bollinger-Standardabweichung (Standard: 2)
    macd_fast           : MACD Fast  (Standard: 12)
    macd_slow           : MACD Slow  (Standard: 26)
    macd_signal         : MACD Signal (Standard: 9)
    adx_period          : ADX-Periode (Standard: 14)
    stoch_k             : Stochastic %K (Standard: 14)
    stoch_d             : Stochastic %D (Standard: 3)
    stoch_smooth        : Stochastic Smooth (Standard: 3)
    cci_period          : CCI-Periode (Standard: 20)
    williams_period     : Williams %R Periode (Standard: 14)
    keltner_period      : Keltner-Periode (Standard: 20)
    keltner_atr_mult    : Keltner ATR-Multiplikator (Standard: 2)
    warmup_candles      : Anzahl Warmup-Candles die entfernt werden (Standard: 200)
    include_time_features : Zeitfeatures einschliessen (Standard: True)
    feature_dir         : Ausgabeverzeichnis fuer Parquet-Dateien (optional)
    """

    def __init__(
        self,
        ema_periods:          list[int] = None,
        sma_periods:          list[int] = None,
        rsi_periods:          list[int] = None,
        atr_period:           int   = 14,
        bollinger_period:     int   = 20,
        bollinger_std:        float = 2.0,
        macd_fast:            int   = 12,
        macd_slow:            int   = 26,
        macd_signal:          int   = 9,
        adx_period:           int   = 14,
        stoch_k:              int   = 14,
        stoch_d:              int   = 3,
        stoch_smooth:         int   = 3,
        cci_period:           int   = 20,
        williams_period:      int   = 14,
        keltner_period:       int   = 20,
        keltner_atr_mult:     float = 2.0,
        warmup_candles:       int   = 200,
        include_time_features: bool = True,
        include_sentiment:    bool  = False,
        sentiment_feature:    Optional[Any] = None,
        feature_dir:          Optional[str | Path] = None,
    ) -> None:
        self.ema_periods          = ema_periods   or [9, 20, 50, 200]
        self.sma_periods          = sma_periods   or [20, 50]
        self.rsi_periods          = rsi_periods   or [14]
        self.atr_period           = atr_period
        self.bollinger_period     = bollinger_period
        self.bollinger_std        = bollinger_std
        self.macd_fast            = macd_fast
        self.macd_slow            = macd_slow
        self.macd_signal          = macd_signal
        self.adx_period           = adx_period
        self.stoch_k              = stoch_k
        self.stoch_d              = stoch_d
        self.stoch_smooth         = stoch_smooth
        self.cci_period           = cci_period
        self.williams_period      = williams_period
        self.keltner_period       = keltner_period
        self.keltner_atr_mult     = keltner_atr_mult
        self.warmup_candles       = warmup_candles
        self.include_time_features = include_time_features
        self.include_sentiment    = include_sentiment
        self._sentiment_feature   = sentiment_feature
        self.feature_dir          = Path(feature_dir) if feature_dir else None

    # â"€â"€ Factory â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

    @classmethod
    def from_config(cls, config_path: str | Path) -> "FeatureBuilder":
        """Erstellt eine Instanz aus config/config.yaml."""
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        ft = cfg.get("features", {})
        return cls(
            ema_periods           = ft.get("ema_periods",   [9, 20, 50, 200]),
            sma_periods           = ft.get("sma_periods",   [20, 50]),
            rsi_periods           = ft.get("rsi_periods",   [14]),
            atr_period            = ft.get("atr_period",    14),
            bollinger_period      = ft.get("bollinger_period", 20),
            bollinger_std         = ft.get("bollinger_std", 2),
            warmup_candles        = ft.get("warmup_candles", 200),
            include_time_features = ft.get("include_time_features", True),
            include_sentiment     = ft.get("include_sentiment", False),
        )

    # â"€â"€ Oeffentliche Methode â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

    def build(
        self,
        df:         pd.DataFrame,
        symbol:     str = "",
        timeframe:  str = "",
        save:       bool = False,
    ) -> pd.DataFrame:
        """
        Berechnet alle Features und gibt einen DataFrame zurueck.

        Parameters
        ----------
        df        : Validierter OHLCV-DataFrame (timestamp, open, high, low, close, volume)
        symbol    : Symbol-Name fuer Dateinamen (z.B. "EURUSD")
        timeframe : Timeframe fuer Dateinamen (z.B. "H1")
        save      : Parquet-Datei speichern wenn True

        Returns
        -------
        pd.DataFrame mit allen Features, Warmup-Periode entfernt.
        """
        if len(df) < self.warmup_candles + 10:
            raise FeatureBuilderError(
                f"Zu wenige Candles: {len(df)} < {self.warmup_candles + 10} "
                f"(warmup={self.warmup_candles})"
            )

        df = df.copy().reset_index(drop=True)

        close  = df["close"]
        high   = df["high"]
        low    = df["low"]
        open_  = df["open"]
        volume = df["volume"] if "volume" in df.columns else pd.Series(0, index=df.index)

        features = pd.DataFrame(index=df.index)
        features["timestamp"] = df["timestamp"]

        # â"€â"€ Trend-Indikatoren â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

        for p in self.ema_periods:
            features[f"ema_{p}"] = (
                EMAIndicator(close=close, window=p, fillna=False).ema_indicator().shift(1)
            )

        for p in self.sma_periods:
            features[f"sma_{p}"] = (
                SMAIndicator(close=close, window=p, fillna=False).sma_indicator().shift(1)
            )

        macd = MACD(
            close=close,
            window_fast=self.macd_fast,
            window_slow=self.macd_slow,
            window_sign=self.macd_signal,
            fillna=False,
        )
        features["macd"]        = macd.macd().shift(1)
        features["macd_signal"] = macd.macd_signal().shift(1)
        features["macd_diff"]   = macd.macd_diff().shift(1)

        adx = ADXIndicator(high=high, low=low, close=close, window=self.adx_period, fillna=False)
        features["adx"]    = adx.adx().shift(1)
        features["adx_pos"] = adx.adx_pos().shift(1)
        features["adx_neg"] = adx.adx_neg().shift(1)

        # â"€â"€ Momentum-Indikatoren â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

        for p in self.rsi_periods:
            features[f"rsi_{p}"] = (
                RSIIndicator(close=close, window=p, fillna=False).rsi().shift(1)
            )

        stoch = StochasticOscillator(
            high=high, low=low, close=close,
            window=self.stoch_k,
            smooth_window=self.stoch_d,
            fillna=False,
        )
        features["stoch_k"] = stoch.stoch().shift(1)
        features["stoch_d"] = stoch.stoch_signal().shift(1)

        features["cci_20"] = (
            CCIIndicator(high=high, low=low, close=close,
                         window=self.cci_period, fillna=False).cci().shift(1)
        )

        features["williams_r"] = (
            WilliamsRIndicator(high=high, low=low, close=close,
                               lbp=self.williams_period, fillna=False).williams_r().shift(1)
        )

        # â"€â"€ Volatilitaets-Indikatoren â"€â"€â"€â"€â"€â"€â"€â"€â"€

        features[f"atr_{self.atr_period}"] = (
            AverageTrueRange(high=high, low=low, close=close,
                             window=self.atr_period, fillna=False).average_true_range().shift(1)
        )

        bb = BollingerBands(
            close=close,
            window=self.bollinger_period,
            window_dev=self.bollinger_std,
            fillna=False,
        )
        features["bb_upper"]  = bb.bollinger_hband().shift(1)
        features["bb_mid"]    = bb.bollinger_mavg().shift(1)
        features["bb_lower"]  = bb.bollinger_lband().shift(1)
        features["bb_width"]  = bb.bollinger_wband().shift(1)
        features["bb_pct"]    = bb.bollinger_pband().shift(1)

        kc = KeltnerChannel(
            high=high, low=low, close=close,
            window=self.keltner_period,
            window_atr=self.atr_period,
            multiplier=self.keltner_atr_mult,
            fillna=False,
        )
        features["kc_upper"] = kc.keltner_channel_hband().shift(1)
        features["kc_mid"]   = kc.keltner_channel_mband().shift(1)
        features["kc_lower"] = kc.keltner_channel_lband().shift(1)

        # â"€â"€ Volumen-Indikatoren â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

        features["obv"] = (
            OnBalanceVolumeIndicator(close=close, volume=volume, fillna=False)
            .on_balance_volume().shift(1)
        )

        # â"€â"€ Abgeleitete Features â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
        # KEIN shift() â€" werden aus aktueller Candle berechnet (kein Zukunftswissen)

        features["candle_body"]      = (close - open_).abs()
        features["candle_direction"] = np.sign(close - open_).astype(int)
        features["high_low_range"]   = high - low
        # close_position: wo schliesst die Kerze innerhalb der Range [0,1]
        hl_range = high - low
        features["close_position"]   = np.where(
            hl_range > 0,
            (close - low) / hl_range,
            0.5,
        )

        # â"€â"€ Zeit-Features â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

        if self.include_time_features:
            ts = pd.DatetimeIndex(df["timestamp"])
            features["hour_of_day"] = ts.hour
            features["day_of_week"] = ts.dayofweek   # 0=Mo, 4=Fr

        # ── Sentiment-Feature (optional, via features.include_sentiment: true) ─

        if self.include_sentiment:
            if self._sentiment_feature is not None:
                try:
                    sentiment_dict = self._sentiment_feature.build_feature(symbol)
                    score = float(sentiment_dict.get("sentiment_score", 0.0))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("SentimentFeature fehlgeschlagen: {exc}", exc=exc)
                    score = 0.0
            else:
                score = 0.0
            features["sentiment_score"] = score

        # â"€â"€ Warmup entfernen â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

        features = features.iloc[self.warmup_candles:].reset_index(drop=True)

        n_features = len(features.columns) - 1   # ohne timestamp
        logger.info(
            "FeatureBuilder | {symbol} {tf} | {rows} Zeilen, {cols} Features",
            symbol=symbol, tf=timeframe,
            rows=len(features), cols=n_features,
        )

        # â"€â"€ Parquet speichern â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

        if save and self.feature_dir:
            self._save_parquet(features, symbol, timeframe)

        return features

    # â"€â"€ Private Methoden â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

    def _save_parquet(
        self, df: pd.DataFrame, symbol: str, timeframe: str
    ) -> Path:
        """Speichert Features als Parquet-Datei."""
        self.feature_dir.mkdir(parents=True, exist_ok=True)
        date_str  = datetime.now(tz=timezone.utc).strftime("%Y%m%d")
        filename  = f"{symbol}_{timeframe}_{date_str}.parquet"
        path      = self.feature_dir / filename
        df.to_parquet(path, index=False, compression="snappy")
        logger.info("Features saved | {path}", path=path)
        return path

    @property
    def feature_names(self) -> list[str]:
        """Gibt alle Feature-Namen zurueck (ohne timestamp)."""
        names = []
        for p in self.ema_periods:
            names.append(f"ema_{p}")
        for p in self.sma_periods:
            names.append(f"sma_{p}")
        names += ["macd", "macd_signal", "macd_diff",
                  "adx", "adx_pos", "adx_neg"]
        for p in self.rsi_periods:
            names.append(f"rsi_{p}")
        names += ["stoch_k", "stoch_d", "cci_20", "williams_r",
                  f"atr_{self.atr_period}",
                  "bb_upper", "bb_mid", "bb_lower", "bb_width", "bb_pct",
                  "kc_upper", "kc_mid", "kc_lower",
                  "obv",
                  "candle_body", "candle_direction",
                  "high_low_range", "close_position"]
        if self.include_time_features:
            names += ["hour_of_day", "day_of_week"]
        if self.include_sentiment:
            names.append("sentiment_score")
        return names

