"""
src/data/validator.py
DataValidator – prueft eingehende OHLCV-DataFrames auf 6 Kriterien.

Pruefungen:
  1. Zeitluecken        – fehlende Candles erkennen (Wochenenden ignorieren)
  2. Duplikate          – doppelte Timestamps entfernen (neuester gewinnt)
  3. OHLC-Konsistenz    – high >= open/close/low, low <= open/close/high
  4. Ausreisser         – Kerzen > N * ATR(20) werden geflaggt
  5. NaN-Werte          – vollstaendige Zeilen entfernen
  6. Zero-Range         – high - low > 0

Aufruf:
    validator = DataValidator.from_config("config/config.yaml")
    report    = validator.validate(df, symbol="EURUSD", timeframe="H1")
    if not report.is_usable:
        raise DataQualityError(report.errors)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import yaml
from loguru import logger


# ─────────────────────────────────────────────
#  Exceptions
# ─────────────────────────────────────────────

class DataQualityError(Exception):
    """Wird ausgeloest wenn Datenqualitaet unter den Mindestschwellwert faellt."""


# ─────────────────────────────────────────────
#  Timeframe → Timedelta Mapping
# ─────────────────────────────────────────────

_TF_DELTA: dict[str, timedelta] = {
    "M1":  timedelta(minutes=1),
    "M5":  timedelta(minutes=5),
    "M15": timedelta(minutes=15),
    "M30": timedelta(minutes=30),
    "H1":  timedelta(hours=1),
    "H4":  timedelta(hours=4),
    "D1":  timedelta(days=1),
    "W1":  timedelta(weeks=1),
}


# ─────────────────────────────────────────────
#  DataQualityReport
# ─────────────────────────────────────────────

@dataclass
class DataQualityReport:
    symbol:             str
    timeframe:          str
    total_candles:      int
    missing_candles:    int
    missing_pct:        float
    duplicates_removed: int
    ohlc_violations:    int
    outliers_flagged:   int
    nan_rows_removed:   int
    quality_score:      float        # 0.0 – 1.0
    is_usable:          bool         # False wenn quality_score < min_quality_score
    warnings:           list[str] = field(default_factory=list)
    errors:             list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)

    def save(self, directory: str | Path) -> Path:
        """Speichert den Bericht als JSON in das angegebene Verzeichnis."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        filename = f"{self.symbol}_{self.timeframe}_quality_report.json"
        path = directory / filename
        path.write_text(self.to_json(), encoding="utf-8")
        logger.info("Quality report saved | {path}", path=path)
        return path


# ─────────────────────────────────────────────
#  DataValidator
# ─────────────────────────────────────────────

class DataValidator:
    """
    Prueft OHLCV-DataFrames auf Qualitaetskriterien.

    Parameters
    ----------
    max_missing_pct        : Maximaler Anteil fehlender Candles (Standard: 5.0 %)
    outlier_atr_multiplier : Ausreisser-Schwellwert in ATR-Vielfachen (Standard: 5.0)
    min_quality_score      : Minimaler Qualitaetsscore (Standard: 0.95)
    interpolation_method   : Interpolationsmethode fuer kleine Luecken (Standard: "linear")
    max_gap_candles        : Max. Lueckengroesse fuer Interpolation (Standard: 3)
    report_dir             : Verzeichnis fuer Qualitaetsberichte (optional)
    """

    def __init__(
        self,
        max_missing_pct:        float = 5.0,
        outlier_atr_multiplier: float = 5.0,
        min_quality_score:      float = 0.95,
        interpolation_method:   str   = "linear",
        max_gap_candles:        int   = 3,
        report_dir:             Optional[str | Path] = None,
    ) -> None:
        self.max_missing_pct        = max_missing_pct
        self.outlier_atr_multiplier = outlier_atr_multiplier
        self.min_quality_score      = min_quality_score
        self.interpolation_method   = interpolation_method
        self.max_gap_candles        = max_gap_candles
        self.report_dir             = Path(report_dir) if report_dir else None

    @classmethod
    def from_config(cls, config_path: str | Path) -> "DataValidator":
        """Erstellt eine Instanz aus config/config.yaml."""
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        dv = cfg.get("data_validation", {})
        return cls(
            max_missing_pct        = dv.get("max_missing_pct",        5.0),
            outlier_atr_multiplier = dv.get("outlier_atr_multiplier", 5.0),
            min_quality_score      = dv.get("min_quality_score",      0.95),
            interpolation_method   = dv.get("interpolation_method",   "linear"),
            max_gap_candles        = dv.get("max_gap_candles",        3),
        )

    # ── Oeffentliche Methode ──────────────────

    def validate(
        self,
        df: pd.DataFrame,
        symbol:    str,
        timeframe: str,
    ) -> tuple[DataQualityReport, pd.DataFrame]:
        """
        Fuehrt alle 6 Pruefungen durch.

        Returns
        -------
        (DataQualityReport, pd.DataFrame)
            report   – Qualitaetsbericht mit allen Metriken
            clean_df – Bereinigter DataFrame (Duplikate, NaN, OHLC-Violations entfernt,
                       kleine Luecken interpoliert)

        Bei kritischen Fehlern (zu viele Luecken, leerer DF) wird DataQualityError ausgeloest.
        """
        warnings: list[str] = []
        errors:   list[str] = []

        df = df.copy()

        # Sicherstellen dass timestamp-Spalte vorhanden und sortiert
        if "timestamp" not in df.columns:
            raise DataQualityError("DataFrame hat keine 'timestamp'-Spalte.")
        df = df.sort_values("timestamp").reset_index(drop=True)

        if len(df) == 0:
            raise DataQualityError("DataFrame ist leer – keine Daten zum Validieren.")

        total_before = len(df)

        # ── Pruefung 5: NaN-Werte (vor allem anderen) ──
        nan_rows_removed, df = self._check_nan(df, warnings)

        # ── Pruefung 2: Duplikate ──
        duplicates_removed, df = self._check_duplicates(df, warnings)

        # ── Pruefung 3: OHLC-Konsistenz ──
        ohlc_violations, df = self._check_ohlc_consistency(df, warnings)

        # ── Pruefung 6: Zero-Range ──
        df = self._check_zero_range(df, warnings)

        # ── Pruefung 4: Ausreisser ──
        outliers_flagged = self._check_outliers(df, warnings)

        # ── Pruefung 1: Zeitluecken ──
        missing_candles, df = self._check_gaps(df, timeframe, warnings, errors)

        # ── Abschluss: NaN die durch Interpolation entstanden sind entfernen ──
        ohlcv_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        remaining_nan = df[ohlcv_cols].isna().any(axis=1).sum()
        if remaining_nan:
            df = df[~df[ohlcv_cols].isna().any(axis=1)].reset_index(drop=True)

        total_candles = len(df)
        expected      = total_candles + missing_candles
        missing_pct   = (missing_candles / expected * 100) if expected > 0 else 0.0

        # ── Qualitaetsscore berechnen ──
        quality_score = self._compute_quality_score(
            total_candles, missing_candles, ohlc_violations,
            outliers_flagged, nan_rows_removed,
        )

        is_usable = quality_score >= self.min_quality_score

        if not is_usable:
            msg = (
                f"Qualitaetsscore {quality_score:.3f} unter Mindestwert "
                f"{self.min_quality_score} | {symbol} {timeframe}"
            )
            errors.append(msg)
            logger.error(msg)

        report = DataQualityReport(
            symbol             = symbol,
            timeframe          = timeframe,
            total_candles      = total_candles,
            missing_candles    = missing_candles,
            missing_pct        = round(missing_pct, 4),
            duplicates_removed = duplicates_removed,
            ohlc_violations    = ohlc_violations,
            outliers_flagged   = outliers_flagged,
            nan_rows_removed   = nan_rows_removed,
            quality_score      = round(quality_score, 6),
            is_usable          = is_usable,
            warnings           = warnings,
            errors             = errors,
        )

        if self.report_dir:
            report.save(self.report_dir)

        logger.info(
            "Validation | {symbol} {tf} | score={score:.3f} usable={usable} "
            "missing={missing} dupes={dupes} ohlc={ohlc} outliers={out} nan={nan}",
            symbol=symbol, tf=timeframe,
            score=quality_score, usable=is_usable,
            missing=missing_candles, dupes=duplicates_removed,
            ohlc=ohlc_violations, out=outliers_flagged, nan=nan_rows_removed,
        )

        if missing_pct > self.max_missing_pct:
            raise DataQualityError(
                f"Zu viele fehlende Candles: {missing_pct:.2f}% > {self.max_missing_pct}% | "
                f"{symbol} {timeframe}"
            )

        return report, df

    # ── Private Pruefmethoden ─────────────────

    def _check_nan(
        self, df: pd.DataFrame, warnings: list[str]
    ) -> tuple[int, pd.DataFrame]:
        """Pruefung 5: Zeilen mit NaN in OHLCV-Spalten entfernen."""
        ohlcv = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        mask  = df[ohlcv].isna().any(axis=1)
        count = int(mask.sum())
        if count:
            warnings.append(f"NaN-Werte: {count} Zeilen entfernt.")
            logger.warning("NaN rows removed: {n}", n=count)
        return count, df[~mask].reset_index(drop=True)

    def _check_duplicates(
        self, df: pd.DataFrame, warnings: list[str]
    ) -> tuple[int, pd.DataFrame]:
        """Pruefung 2: Doppelte Timestamps entfernen – neuester Eintrag gewinnt."""
        before = len(df)
        df = df.drop_duplicates(subset=["timestamp"], keep="last")
        count = before - len(df)
        if count:
            warnings.append(f"Duplikate: {count} Eintraege entfernt.")
            logger.warning("Duplicates removed: {n}", n=count)
        return count, df.reset_index(drop=True)

    def _check_ohlc_consistency(
        self, df: pd.DataFrame, warnings: list[str]
    ) -> tuple[int, pd.DataFrame]:
        """Pruefung 3: OHLC-Konsistenz – ungueltige Zeilen entfernen."""
        mask_high = (
            (df["high"] >= df["open"]) &
            (df["high"] >= df["close"]) &
            (df["high"] >= df["low"])
        )
        mask_low = (
            (df["low"] <= df["open"]) &
            (df["low"] <= df["close"]) &
            (df["low"] <= df["high"])
        )
        invalid = ~(mask_high & mask_low)
        count   = int(invalid.sum())
        if count:
            warnings.append(f"OHLC-Verletzungen: {count} Zeilen entfernt.")
            logger.warning("OHLC violations removed: {n}", n=count)
        return count, df[~invalid].reset_index(drop=True)

    def _check_zero_range(
        self, df: pd.DataFrame, warnings: list[str]
    ) -> pd.DataFrame:
        """Pruefung 6: Zeilen mit high == low entfernen."""
        mask  = df["high"] - df["low"] <= 0
        count = int(mask.sum())
        if count:
            warnings.append(f"Zero-Range (high==low): {count} Zeilen entfernt.")
            logger.warning("Zero-range candles removed: {n}", n=count)
        return df[~mask].reset_index(drop=True)

    def _check_outliers(
        self, df: pd.DataFrame, warnings: list[str]
    ) -> int:
        """Pruefung 4: Ausreisser-Flagging via ATR(20). Gibt Anzahl zurueck."""
        if len(df) < 21:
            return 0

        high_low = df["high"] - df["low"]
        high_pc  = (df["high"] - df["close"].shift(1)).abs()
        low_pc   = (df["low"]  - df["close"].shift(1)).abs()
        tr       = pd.concat([high_low, high_pc, low_pc], axis=1).max(axis=1)
        atr20    = tr.rolling(20).mean()
        threshold = atr20 * self.outlier_atr_multiplier
        outliers  = (high_low > threshold).sum()
        count     = int(outliers)
        if count:
            warnings.append(f"Ausreisser geflaggt: {count} Kerzen > {self.outlier_atr_multiplier}x ATR(20).")
            logger.warning("Outliers flagged: {n}", n=count)
        return count

    def _check_gaps(
        self,
        df: pd.DataFrame,
        timeframe: str,
        warnings: list[str],
        errors:   list[str],
    ) -> tuple[int, pd.DataFrame]:
        """
        Pruefung 1: Zeitluecken erkennen.
        - Wochenenden (Sa/So) werden ignoriert
        - Luecken <= max_gap_candles werden interpoliert
        - Groessere Luecken werden geloggt
        """
        delta = _TF_DELTA.get(timeframe.upper())
        if delta is None or len(df) < 2:
            return 0, df

        timestamps = pd.DatetimeIndex(df["timestamp"])
        expected   = pd.date_range(
            start=timestamps[0],
            end=timestamps[-1],
            freq=delta,
            tz=timestamps.tzinfo,
        )

        # Wochenenden herausfiltern (Samstag=5, Sonntag=6) – gilt fuer alle Intraday- und Daily-TF
        if timeframe.upper() in ("M1", "M5", "M15", "M30", "H1", "H4", "D1"):
            expected = expected[expected.dayofweek < 5]

        missing_ts = expected.difference(timestamps)
        count      = len(missing_ts)

        if count == 0:
            return 0, df

        missing_pct = count / len(expected) * 100

        if missing_pct <= self.max_missing_pct:
            # Kleine Luecken: interpolieren
            df = df.set_index("timestamp")
            df = df.reindex(expected)
            # Nur Luecken <= max_gap_candles interpolieren
            mask = df.isna().any(axis=1)
            gap_sizes = mask.astype(int).groupby((~mask).cumsum()).transform("sum")
            interpolate_mask = mask & (gap_sizes <= self.max_gap_candles)
            df[interpolate_mask] = df[interpolate_mask].interpolate(method=self.interpolation_method)
            df = df.reset_index().rename(columns={"index": "timestamp"})
            warnings.append(
                f"Zeitluecken: {count} fehlende Candles ({missing_pct:.2f}%) interpoliert."
            )
            logger.warning("Gaps interpolated: {n} ({pct:.2f}%)", n=count, pct=missing_pct)
        else:
            errors.append(
                f"Kritische Zeitluecken: {count} fehlende Candles ({missing_pct:.2f}%)."
            )
            logger.error("Critical gaps: {n} ({pct:.2f}%)", n=count, pct=missing_pct)

        return count, df

    def _compute_quality_score(
        self,
        total:      int,
        missing:    int,
        ohlc_viol:  int,
        outliers:   int,
        nan_rows:   int,
    ) -> float:
        """Berechnet einen Qualitaetsscore zwischen 0.0 und 1.0."""
        if total == 0:
            return 0.0
        base     = total + missing + ohlc_viol + nan_rows
        penalty  = missing + ohlc_viol + nan_rows + (outliers * 0.1)
        score    = max(0.0, 1.0 - penalty / base) if base > 0 else 0.0
        return round(min(1.0, score), 6)
