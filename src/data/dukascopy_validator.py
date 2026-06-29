"""
src/data/dukascopy_validator.py
Validierung fuer aggregierte Dukascopy-M15-Bars inkl. Spread.

Anders als der generische DataValidator ist diese Pruefung speziell auf lange
Tick-abgeleitete M15-Historie zugeschnitten:
  - Wochenend-/Marktschluss-Luecken werden ERWARTET und sauber markiert,
    NICHT interpoliert (Forex schliesst ~Fr 21:00 UTC, oeffnet ~So 21:00 UTC).
  - Intra-Session-Luecken (fehlende Bars waehrend offener Zeiten) werden separat
    gezaehlt und die groessten gelistet.
  - Duplikate, OHLC-Konsistenz, Ausreisser/Spikes (v.a. XAUUSD).
  - Spread-Plausibilitaet: negative/Null-Spreads, Spread-Statistik in Pips,
    getrennt nach Handelssession (Asia / London / New York).

Reine Funktionen, kein Netzwerk, kein Datei-IO im Kern -> gut testbar.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.data.dukascopy_downloader import SYMBOL_PIP_SIZE

M15 = timedelta(minutes=15)

# Handelssessions nach UTC-Stunde (grobe, robuste Einteilung).
#   Asia:   22:00–07:00   London: 07:00–12:00   NewYork: 12:00–21:00
def _session_for_hour(hour: int) -> str:
    if 7 <= hour < 12:
        return "London"
    if 12 <= hour < 21:
        return "NewYork"
    return "Asia"


@dataclass
class M15QualityReport:
    symbol:              str
    timeframe:           str
    start:               str
    end:                 str
    total_bars:          int
    span_years:          float
    duplicates:          int
    ohlc_violations:     int
    intra_session_missing: int
    weekend_gaps:        int
    outliers:            int
    negative_spread_bars: int
    zero_spread_bars:    int
    spread_pips_min:     float
    spread_pips_median:  float
    spread_pips_mean:    float
    spread_pips_max:     float
    session_spread_pips_median: dict = field(default_factory=dict)
    largest_gaps:        list = field(default_factory=list)
    warnings:            list = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)

    def to_markdown(self) -> str:
        sess = " | ".join(f"{k}: {v:.2f}" for k, v in self.session_spread_pips_median.items())
        gaps = "\n".join(
            f"    - {g['start']} .. {g['end']}  ({g['hours']:.1f} h, {g['missing_bars']} Bars)"
            for g in self.largest_gaps
        ) or "    - keine"
        return (
            f"### {self.symbol} {self.timeframe}\n"
            f"- Zeitraum: {self.start} .. {self.end}  (~{self.span_years:.1f} Jahre)\n"
            f"- M15-Bars gesamt: {self.total_bars:,}\n"
            f"- Duplikate: {self.duplicates} | OHLC-Verletzungen: {self.ohlc_violations} | "
            f"Ausreisser: {self.outliers}\n"
            f"- Intra-Session fehlende Bars: {self.intra_session_missing:,} | "
            f"Wochenend-/Schluss-Luecken: {self.weekend_gaps:,}\n"
            f"- Spread (Pips): min {self.spread_pips_min:.2f} | median {self.spread_pips_median:.2f} | "
            f"mean {self.spread_pips_mean:.2f} | max {self.spread_pips_max:.2f}\n"
            f"- Spread-Median je Session (Pips): {sess}\n"
            f"- Negative Spreads: {self.negative_spread_bars} | Null-Spreads: {self.zero_spread_bars}\n"
            f"- Groesste Luecken:\n{gaps}\n"
        )

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.symbol}_{self.timeframe}_dukascopy_quality.json"
        path.write_text(self.to_json(), encoding="utf-8")
        return path


def _is_weekend_gap(prev_ts: pd.Timestamp, next_ts: pd.Timestamp) -> bool:
    """True wenn die Luecke zwischen prev_ts und next_ts ein Wochenendschluss ist."""
    # Forex schliesst Freitagabend, oeffnet Sonntagabend. Eine Luecke gilt als
    # Wochenende, wenn sie einen Samstag ueberspannt oder Freitag spaet beginnt.
    if prev_ts.weekday() == 4 and prev_ts.hour >= 20:   # Freitag ab 20:00
        return True
    # Luecke ueberspannt einen Samstag?
    days = pd.date_range(prev_ts.normalize(), next_ts.normalize(), freq="D")
    return any(d.weekday() == 5 for d in days)


def validate_dukascopy_m15(
    df: pd.DataFrame,
    symbol: str,
    outlier_sigma: float = 12.0,
    max_largest_gaps: int = 10,
) -> M15QualityReport:
    """
    Validiert einen aggregierten Dukascopy-M15-DataFrame mit Spread-Spalten.

    Erwartete Spalten: timestamp, open, high, low, close, volume,
                       spread_mean, spread_median, tick_count.
    """
    required = {"timestamp", "open", "high", "low", "close", "spread_mean"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"Fehlende Spalten fuer Validierung: {sorted(missing_cols)}")

    warnings: list[str] = []
    pip = SYMBOL_PIP_SIZE.get(symbol, 0.0001)

    df = df.sort_values("timestamp").reset_index(drop=True)
    ts = pd.DatetimeIndex(df["timestamp"])

    # ── Duplikate ──
    duplicates = int(df["timestamp"].duplicated().sum())

    # ── OHLC-Konsistenz ──
    ohlc_bad = ~(
        (df["high"] >= df[["open", "close", "low"]].max(axis=1)) &
        (df["low"]  <= df[["open", "close", "high"]].min(axis=1))
    )
    ohlc_violations = int(ohlc_bad.sum())

    # ── Ausreisser/Spikes (log-return-basiert, robust via MAD) ──
    outliers = 0
    if len(df) > 50:
        ret = np.log(df["close"]).diff()
        med = ret.median()
        mad = (ret - med).abs().median()
        if mad and mad > 0:
            robust_z = (ret - med).abs() / (1.4826 * mad)
            outliers = int((robust_z > outlier_sigma).sum())

    # ── Luecken: Wochenende vs Intra-Session ──
    intra_missing = 0
    weekend_gaps = 0
    gap_list: list[dict] = []
    if len(df) > 1:
        tvals = ts.values.astype("datetime64[ns]")
        diffs = np.diff(tvals)                       # timedelta64[ns]
        m15_td = np.timedelta64(15, "m")
        gap_idx = np.where(diffs > m15_td)[0]        # i -> Luecke zwischen Bar i und i+1
        for i in gap_idx:
            prev_ts = ts[i]
            next_ts = ts[i + 1]
            missing_bars = int((next_ts - prev_ts) / M15) - 1
            if missing_bars <= 0:
                continue
            if _is_weekend_gap(prev_ts, next_ts):
                weekend_gaps += missing_bars
            else:
                intra_missing += missing_bars
                gap_list.append({
                    "start": str(prev_ts),
                    "end": str(next_ts),
                    "hours": (next_ts - prev_ts).total_seconds() / 3600.0,
                    "missing_bars": missing_bars,
                })

    gap_list.sort(key=lambda g: g["missing_bars"], reverse=True)
    largest_gaps = gap_list[:max_largest_gaps]

    # ── Spread-Plausibilitaet ──
    spread = df["spread_mean"].astype(float)
    negative_spread = int((spread < 0).sum())
    zero_spread = int((spread == 0).sum())
    spread_pips = spread[spread >= 0] / pip
    if len(spread_pips) == 0:
        spread_pips = pd.Series([0.0])
        warnings.append("Keine validen Spreads gefunden.")

    # Spread je Session
    sess_med: dict[str, float] = {}
    if "spread_mean" in df.columns and len(df):
        sess = ts.hour.map(_session_for_hour)
        tmp = pd.DataFrame({"session": sess, "spread_pips": spread.values / pip})
        tmp = tmp[tmp["spread_pips"] >= 0]
        for name, grp in tmp.groupby("session"):
            sess_med[str(name)] = round(float(grp["spread_pips"].median()), 3)

    if negative_spread:
        warnings.append(f"{negative_spread} Bars mit negativem Spread (verdaechtig).")
    if ohlc_violations:
        warnings.append(f"{ohlc_violations} OHLC-Verletzungen.")
    if duplicates:
        warnings.append(f"{duplicates} doppelte Timestamps.")

    span_years = ((ts[-1] - ts[0]).days / 365.25) if len(df) > 1 else 0.0

    report = M15QualityReport(
        symbol=symbol,
        timeframe="M15",
        start=str(ts[0]) if len(df) else "N/A",
        end=str(ts[-1]) if len(df) else "N/A",
        total_bars=len(df),
        span_years=round(span_years, 2),
        duplicates=duplicates,
        ohlc_violations=ohlc_violations,
        intra_session_missing=intra_missing,
        weekend_gaps=weekend_gaps,
        outliers=outliers,
        negative_spread_bars=negative_spread,
        zero_spread_bars=zero_spread,
        spread_pips_min=round(float(spread_pips.min()), 3),
        spread_pips_median=round(float(spread_pips.median()), 3),
        spread_pips_mean=round(float(spread_pips.mean()), 3),
        spread_pips_max=round(float(spread_pips.max()), 3),
        session_spread_pips_median=sess_med,
        largest_gaps=largest_gaps,
        warnings=warnings,
    )
    logger.info(
        "Dukascopy-Validierung {sym} | bars={n} dupes={d} ohlc={o} intra_gap={g} "
        "spread_med={s:.2f}pips",
        sym=symbol, n=len(df), d=duplicates, o=ohlc_violations,
        g=intra_missing, s=report.spread_pips_median,
    )
    return report
