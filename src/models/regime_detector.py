"""
src/models/regime_detector.py
RegimeDetector – erkennt das aktuelle Marktregime aus FeatureBuilder-Daten.

Drei Regime (Prioritaet: HIGH_VOLATILITY > TRENDING > RANGING):
  TRENDING        – ADX >= adx_trending_threshold (klare Richtung)
  RANGING         – ADX <  adx_trending_threshold und ATR normal (Seitwärts)
  HIGH_VOLATILITY – aktueller ATR > atr_volatility_multiplier * rolliernder
                    ATR-Durchschnitt (volatile Marktphase, unabhaengig vom ADX)

Nutzt ausschliesslich Spalten, die der FeatureBuilder bereits erzeugt:
  'adx'   (ADXIndicator.adx, .shift(1))
  'atr_14' (AverageTrueRange, .shift(1), Periode 14 per Default)

Regimewechsel werden geloggt (Zeitpunkt, altes -> neues Regime).

Hook fuer TradingOrchestrator:
  get_position_size_multiplier(regime) -> float
    HIGH_VOLATILITY: 0.5 (halbe Positionsgroesse)
    sonst:           1.0
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

import pandas as pd
from loguru import logger


# ── Regime-Konstanten ─────────────────────────────────────────────────────────

class Regime(str, Enum):
    """Marktregime-Konstanten."""
    TRENDING        = "TRENDING"
    RANGING         = "RANGING"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"


# ── RegimeDetector ────────────────────────────────────────────────────────────

class RegimeDetector:
    """
    Erkennt das aktuelle Marktregime aus Feature-Daten des FeatureBuilder.

    Regime-Prioritaet: HIGH_VOLATILITY > TRENDING > RANGING

    Parameters
    ----------
    adx_trending_threshold    : ADX-Wert ab dem ein Trend erkannt wird (Standard: 25.0).
                                Klassischer Schwellwert laut Wilder: 25.
    atr_volatility_multiplier : Faktor, um den der aktuelle ATR den rollierenden
                                Durchschnitt ueberschreiten muss, damit
                                HIGH_VOLATILITY ausgeloest wird (Standard: 1.5).
    atr_window                : Fenster fuer den rollierenden ATR-Durchschnitt
                                (Standard: 50 Kerzen).
    adx_col                   : Spaltenname des ADX in features_df
                                (Standard: 'adx', FeatureBuilder-Konvention).
    atr_col                   : Spaltenname des ATR in features_df
                                (Standard: 'atr_14', FeatureBuilder-Konvention
                                 bei atr_period=14).
    """

    def __init__(
        self,
        adx_trending_threshold:    float = 25.0,
        atr_volatility_multiplier: float = 1.5,
        atr_window:                int   = 50,
        adx_col:                   str   = "adx",
        atr_col:                   str   = "atr_14",
    ) -> None:
        if adx_trending_threshold < 0:
            raise ValueError("adx_trending_threshold muss >= 0 sein.")
        if atr_volatility_multiplier <= 0:
            raise ValueError("atr_volatility_multiplier muss > 0 sein.")
        if atr_window < 1:
            raise ValueError("atr_window muss >= 1 sein.")

        self._adx_threshold  = adx_trending_threshold
        self._atr_multiplier = atr_volatility_multiplier
        self._atr_window     = atr_window
        self._adx_col        = adx_col
        self._atr_col        = atr_col
        self._last_regime: Optional[str] = None

    # ── Oeffentliche Schnittstelle ────────────────────────────────────────────

    def detect_regime(self, features_df: pd.DataFrame) -> str:
        """
        Ermittelt das aktuelle Marktregime aus dem letzten Eintrag von features_df.

        Liest ADX und ATR aus der letzten Zeile; der ATR-Durchschnitt wird
        ueber die letzten atr_window Zeilen berechnet (oder weniger, falls
        features_df kuerzer ist).

        Regime-Prioritaet: HIGH_VOLATILITY > TRENDING > RANGING

        Parameters
        ----------
        features_df : pd.DataFrame mit den Spalten adx_col und atr_col,
                      wie sie FeatureBuilder erzeugt. Muss mindestens eine
                      nicht-NaN-Zeile enthalten.

        Returns
        -------
        str: Regime-Name ('TRENDING', 'RANGING', 'HIGH_VOLATILITY').

        Raises
        ------
        ValueError wenn features_df leer ist oder Pflicht-Spalten fehlen.
        """
        self._validate(features_df)

        adx_val            = self._read_adx(features_df)
        atr_val, atr_mean  = self._read_atr(features_df)

        if atr_mean > 0.0 and atr_val > atr_mean * self._atr_multiplier:
            regime = Regime.HIGH_VOLATILITY.value
        elif adx_val >= self._adx_threshold:
            regime = Regime.TRENDING.value
        else:
            regime = Regime.RANGING.value

        self._log_change(regime, adx_val, atr_val, atr_mean)
        return regime

    def get_position_size_multiplier(self, regime: str) -> float:
        """
        Hook fuer TradingOrchestrator: Positionsgroessen-Faktor je Regime.

        HIGH_VOLATILITY -> 0.5 (halbe Positionsgroesse zur Risikoreduzierung)
        TRENDING        -> 1.0
        RANGING         -> 1.0

        Parameters
        ----------
        regime : Regime-String aus detect_regime().

        Returns
        -------
        float: Multiplikator (0.5 oder 1.0).
        """
        return 0.5 if regime == Regime.HIGH_VOLATILITY.value else 1.0

    @property
    def last_regime(self) -> Optional[str]:
        """Zuletzt erkanntes Regime; None wenn detect_regime() noch nie aufgerufen wurde."""
        return self._last_regime

    # ── Private Methoden ──────────────────────────────────────────────────────

    def _validate(self, features_df: pd.DataFrame) -> None:
        """Prueft ob features_df brauchbar ist und die Pflicht-Spalten enthaelt."""
        if features_df is None or features_df.empty:
            raise ValueError("features_df ist leer oder None.")
        missing = [
            col for col in (self._adx_col, self._atr_col)
            if col not in features_df.columns
        ]
        if missing:
            raise ValueError(
                f"Pflicht-Spalten fehlen in features_df: {missing}. "
                f"Vorhandene Spalten: {list(features_df.columns)}"
            )

    def _read_adx(self, features_df: pd.DataFrame) -> float:
        """ADX-Wert der letzten Zeile; NaN wird als 0.0 behandelt."""
        val = features_df[self._adx_col].iloc[-1]
        return 0.0 if pd.isna(val) else float(val)

    def _read_atr(self, features_df: pd.DataFrame) -> tuple[float, float]:
        """
        Aktueller ATR (letzte Zeile) und rollierender ATR-Durchschnitt
        ueber atr_window Perioden.

        Returns
        -------
        (current_atr, mean_atr) – beide 0.0 wenn keine gueltigen ATR-Werte vorhanden.
        """
        atr_series = features_df[self._atr_col].dropna()
        if atr_series.empty:
            return 0.0, 0.0
        current_atr = float(atr_series.iloc[-1])
        window      = min(self._atr_window, len(atr_series))
        mean_atr    = float(atr_series.iloc[-window:].mean())
        return current_atr, mean_atr

    def _log_change(
        self,
        regime:   str,
        adx_val:  float,
        atr_val:  float,
        atr_mean: float,
    ) -> None:
        """Protokolliert Regimewechsel oder initiales Regime."""
        if regime == self._last_regime:
            return
        if self._last_regime is None:
            logger.info(
                "RegimeDetector: Initiales Regime={r} "
                "(ADX={adx:.1f}, ATR={atr:.6f}, ATR-Ø={mean:.6f})",
                r=regime, adx=adx_val, atr=atr_val, mean=atr_mean,
            )
        else:
            logger.info(
                "RegimeDetector: Regimewechsel {old} -> {new} "
                "(ADX={adx:.1f}, ATR={atr:.6f}, ATR-Ø={mean:.6f})",
                old=self._last_regime, new=regime,
                adx=adx_val, atr=atr_val, mean=atr_mean,
            )
        self._last_regime = regime
