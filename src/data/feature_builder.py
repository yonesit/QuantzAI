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
        mtf_adx_threshold:    float = 25.0,
        mtf_flip_lookback:    int   = 3,
    ) -> None:
        self.ema_periods          = ema_periods   or [9, 20, 50, 200]
        self.sma_periods          = sma_periods   or [50]
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
        self.mtf_adx_threshold    = mtf_adx_threshold
        self.mtf_flip_lookback    = mtf_flip_lookback

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
            mtf_adx_threshold     = ft.get("mtf_adx_threshold", 25.0),
            mtf_flip_lookback     = ft.get("mtf_flip_lookback", 3),
        )

    # â"€â"€ Oeffentliche Methode â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

    def build(
        self,
        df:         pd.DataFrame,
        symbol:     str = "",
        timeframe:  str = "",
        save:       bool = False,
        df_h4:      Optional[pd.DataFrame] = None,
        df_d1:      Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Berechnet alle Features und gibt einen DataFrame zurueck.

        Parameters
        ----------
        df        : Validierter OHLCV-DataFrame (timestamp, open, high, low, close, volume)
        symbol    : Symbol-Name fuer Dateinamen (z.B. "EURUSD")
        timeframe : Timeframe fuer Dateinamen (z.B. "H1")
        save      : Parquet-Datei speichern wenn True
        df_h4     : Optionaler H4-OHLCV-DataFrame fuer Multi-Timeframe-Kontext.
                    Erzeugt Feature 'h4_trend' = (EMA50-EMA200)/ATR14 auf H4-Basis.
        df_d1     : Optionaler D1-OHLCV-DataFrame. Erzeugt Feature 'd1_trend'.

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
        features["close"]     = df["close"]
        features["high"]      = df["high"]
        features["low"]       = df["low"]

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

        #â"€â"€ Volatilitaets-Indikatoren â"€â"€â"€â"€â"€â"€â"€â"€â"€

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
        features["bb_lower"]  = bb.bollinger_lband().shift(1)
        features["bb_width"]  = bb.bollinger_wband().shift(1)

        kc = KeltnerChannel(
            high=high, low=low, close=close,
            window=self.keltner_period,
            window_atr=self.atr_period,
            multiplier=self.keltner_atr_mult,
            fillna=False,
        )
        features["kc_lower"] = kc.keltner_channel_lband().shift(1)

        # â"€â"€ Volumen-Indikatoren â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

        features["obv"] = (
            OnBalanceVolumeIndicator(close=close, volume=volume, fillna=False)
            .on_balance_volume().shift(1)
        )

        # ── Marktregime-Feature ─────────────────────────────────────────────────
        # Basiert auf bereits-geshifteten adx/atr_14 — kein Look-ahead.
        # RANGING=0  TRENDING=1  HIGH_VOLATILITY=2
        # Prioritaet: HIGH_VOLATILITY > TRENDING > RANGING (wie RegimeDetector)
        _atr_col = f"atr_{self.atr_period}"
        _atr_roll = features[_atr_col].rolling(50, min_periods=1).mean()
        _hv = features[_atr_col] > _atr_roll * 1.5
        _tr = features["adx"] >= 25.0
        features["regime"] = np.where(_hv, 2, np.where(_tr, 1, 0)).astype(int)

        # ── Zeit-Features ────────────────────────────────────────────────────────

        if self.include_time_features:
            ts = pd.DatetimeIndex(df["timestamp"])
            features["hour_of_day"] = ts.hour

        # ── Multi-Timeframe-Features (optional) ─────────────────────────────────
        # Kontinuierliche Encoding-Wahl: (EMA50-EMA200)/ATR14
        #   + gibt Richtung UND Staerke (vs. binaeres +1/-1)
        #   + skalierungsinvariant durch ATR-Normierung
        #   + LightGBM findet optimale Schwellwerte selbst
        # Kein Look-ahead: _merge_mtf_trend nutzt close_time = open_time + tf_hours,
        # d.h. fuer H1-Bar T wird nur der letzte HTF-Bar verwendet, dessen
        # Schlusskurs-Zeitpunkt <= T liegt (abgeschlossene Bar, nicht laufende).

        if df_h4 is not None:
            _h4_mtf = FeatureBuilder._compute_mtf_trend(df_h4, tf_hours=4)
            _h4_mtf = FeatureBuilder._gate_mtf_trend(
                _h4_mtf,
                adx_threshold=self.mtf_adx_threshold,
                flip_lookback=self.mtf_flip_lookback,
            )
            features["h4_trend"] = FeatureBuilder._merge_mtf_trend(
                features["timestamp"], _h4_mtf
            )

        if df_d1 is not None:
            _d1_mtf = FeatureBuilder._compute_mtf_trend(df_d1, tf_hours=24)
            _d1_mtf = FeatureBuilder._gate_mtf_trend(
                _d1_mtf,
                adx_threshold=self.mtf_adx_threshold,
                flip_lookback=self.mtf_flip_lookback,
            )
            features["d1_trend"] = FeatureBuilder._merge_mtf_trend(
                features["timestamp"], _d1_mtf
            )

        # ── Sentiment-Feature (optional, via features.include_sentiment: true) ─
        # Historischer Modus (SentimentHistory): per-Row via get_sentiment_series()
        #   → merge_asof backward, nur Nachrichten mit bucket_time < T (kein Look-ahead)
        # Live-Modus (SentimentFeature): einmaliger build_feature()-Aufruf wie bisher

        if self.include_sentiment:
            sf = self._sentiment_feature
            if sf is not None and hasattr(sf, "get_sentiment_series"):
                try:
                    features["sentiment_score"] = sf.get_sentiment_series(
                        features["timestamp"]
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("SentimentHistory fehlgeschlagen: {exc}", exc=exc)
                    features["sentiment_score"] = 0.0
            elif sf is not None:
                try:
                    score = float(sf.build_feature(symbol).get("sentiment_score", 0.0))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("SentimentFeature fehlgeschlagen: {exc}", exc=exc)
                    score = 0.0
                features["sentiment_score"] = score
            else:
                features["sentiment_score"] = 0.0

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

    @staticmethod
    def _compute_mtf_trend(df_ohlcv: pd.DataFrame, tf_hours: int) -> pd.DataFrame:
        """
        Berechnet den normalisierten Trend-Feature fuer einen hoeheren Timeframe.

        Trendstaerke = (EMA50 - EMA200) / ATR14
          Positiv  -> Bullentrend  (schnelle EMA ueber langsamer EMA)
          Negativ  -> Baerentrend
          ~0       -> kein klarer Trend

        Parameters
        ----------
        df_ohlcv  : OHLCV-DataFrame mit Spalten timestamp, open, high, low, close.
        tf_hours  : Nominale Bar-Laenge in Stunden (H4=4, D1=24).
                    Wird zur Berechnung der close_time = open_time + tf_hours genutzt.

        Returns
        -------
        DataFrame mit 'close_time' (Zeitpunkt des Bar-Schlusses) und 'trend'.
        """
        close  = df_ohlcv["close"].reset_index(drop=True)
        high   = df_ohlcv["high"].reset_index(drop=True)
        low    = df_ohlcv["low"].reset_index(drop=True)

        ema50  = EMAIndicator(close=close, window=50,  fillna=False).ema_indicator()
        ema200 = EMAIndicator(close=close, window=200, fillna=False).ema_indicator()
        atr14  = AverageTrueRange(
            high=high, low=low, close=close, window=14, fillna=False
        ).average_true_range()
        adx14  = ADXIndicator(
            high=high, low=low, close=close, window=14, fillna=False
        ).adx().fillna(0.0)

        safe_atr = atr14.where(atr14 > 0, other=np.nan)
        trend = ((ema50 - ema200) / safe_atr).fillna(0.0)

        close_time = pd.to_datetime(df_ohlcv["timestamp"].values) + pd.Timedelta(hours=tf_hours)
        return pd.DataFrame({
            "close_time": close_time,
            "trend":      trend.values,
            "adx":        adx14.values,
        })

    @staticmethod
    def _gate_mtf_trend(
        mtf_df:        pd.DataFrame,
        adx_threshold: float = 25.0,
        flip_lookback: int   = 3,
    ) -> pd.DataFrame:
        """
        Regime-Filter fuer MTF-Trendwerte.

        Setzt trend = 0 (neutral) wenn der Trend als instabil gilt:
          - ADX des HTF-Bars liegt unter adx_threshold  (kein klarer Trend)
          - ODER es gab einen Vorzeichenwechsel von trend innerhalb der
            letzten flip_lookback HTF-Bars  (frischer Trendwechsel)

        Look-ahead-Freiheit: der Filter operiert ausschliesslich auf
        abgeschlossenen HTF-Bars (close_time bereits gesetzt). Die
        Zuordnung zu H1-Bars erfolgt weiterhin via merge_asof.

        Parameters
        ----------
        mtf_df        : Ausgabe von _compute_mtf_trend (close_time, trend, adx).
        adx_threshold : Mindestwert fuer ADX, damit Trend als stabil gilt.
        flip_lookback : Anzahl zurueckliegender HTF-Bars, in denen ein
                        Vorzeichenwechsel als 'frischer Trendwechsel' gilt.

        Returns
        -------
        DataFrame mit 'close_time' und gefiltertem 'trend'.
        """
        mtf = mtf_df.copy().reset_index(drop=True)

        sign = np.sign(mtf["trend"])
        # True an Bar T wenn Vorzeichen gegenueber T-1 gewechselt hat
        sign_flipped = (sign != sign.shift(1)).astype(int)
        # True wenn irgendwo in den letzten flip_lookback Bars ein Wechsel war
        recently_flipped = sign_flipped.rolling(flip_lookback, min_periods=1).max().astype(bool)

        # Trend gilt als instabil wenn ADX zu niedrig ODER frischer Flip
        unstable = (mtf["adx"] < adx_threshold) | recently_flipped
        mtf["trend"] = mtf["trend"].where(~unstable, other=0.0)

        return mtf[["close_time", "trend"]]

    @staticmethod
    def _merge_mtf_trend(h1_timestamps: pd.Series, mtf_df: pd.DataFrame) -> np.ndarray:
        """
        Weist jedem H1-Bar den Trendwert des zuletzt ABGESCHLOSSENEN HTF-Bars zu.

        Verwendet pd.merge_asof (direction='backward'): sucht den letzten
        HTF-Bar mit close_time <= H1-Bar-Zeitstempel.
        Laufende (noch nicht geschlossene) Bars werden dadurch nie verwendet.
        H1-Bars ohne vorangehenden HTF-Bar erhalten 0.0 (neutral).

        Parameters
        ----------
        h1_timestamps : pd.Series mit H1-Bar-Zeitstempeln (Reihenfolge beliebig).
        mtf_df        : Ausgabe von _compute_mtf_trend (close_time, trend).

        Returns
        -------
        np.ndarray mit Trend-Werten, Reihenfolge entspricht h1_timestamps.
        """
        original_order = np.arange(len(h1_timestamps))
        h1_df = pd.DataFrame({
            "timestamp": pd.to_datetime(h1_timestamps.values),
            "_order":    original_order,
        }).sort_values("timestamp")

        merged = pd.merge_asof(
            h1_df,
            mtf_df.sort_values("close_time")[["close_time", "trend"]],
            left_on="timestamp",
            right_on="close_time",
            direction="backward",
        )
        merged["trend"] = merged["trend"].fillna(0.0)
        return merged.sort_values("_order")["trend"].to_numpy(dtype=float)

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
        names += ["stoch_k", "stoch_d", "cci_20",
                  f"atr_{self.atr_period}",
                  "bb_upper", "bb_lower", "bb_width",
                  "kc_lower",
                  "obv",
                  "regime"]
        if self.include_time_features:
            names += ["hour_of_day"]
        if self.include_sentiment:
            names.append("sentiment_score")
        return names

