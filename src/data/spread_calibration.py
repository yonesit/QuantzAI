"""
src/data/spread_calibration.py
Wiederverwendbare Logik fuer die Spread-Kalibrierung Dukascopy -> Fusion.

Hintergrund: Die langen Dukascopy-M15-Datensaetze tragen ECN-Roh-Spreads, die
deutlich tighter sind als die realen Fusion-Markets-Spreads. Fuer ein
kostenbewusstes Labeling muessen die Dukascopy-Spreads auf Fusion-Niveau
hochskaliert werden. Dieses Modul misst je Handels-Session (UTC) sowohl einen
multiplikativen Faktor (Fusion/Dukascopy) als auch einen additiven Aufschlag
(Fusion - Dukascopy).

WICHTIG zur Einheit:
  * Dukascopy-Spread liegt in PREIS-Einheiten vor (z.B. 0.00003 = 0.3 Pips).
  * MT5/Fusion meldet den Spread in POINTS je Bar (Ganzzahl).
  * Fuer 5-stellige FX-Paare (EURUSD) gilt: 10 Points = 1 Pip, 1 Point = 0.1 Pip.
  * Beide Quellen werden hier auf die gemeinsame Einheit PIPS gebracht, bevor
    verglichen wird – so entsteht kein Faktor-10-Fehler.

Dieses Modul enthaelt KEINE Labeling-Logik und keinen I/O gegen echte
Handelsdaten – nur reine, testbare Funktionen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import pandas as pd

# ── Einheiten-Umrechnung ─────────────────────────────────────────────────────

# Points pro Pip je Symbol (abhaengig von den Nachkommastellen des Brokers).
# EURUSD ist 5-stellig -> 1 Pip = 0.0001 = 10 Points; 1 Point = 0.00001.
POINTS_PER_PIP: dict[str, int] = {
    "EURUSD": 10,
    "GBPUSD": 10,
    "USDJPY": 10,   # 3-stellig: 1 Pip = 0.01 = 10 Points
    "XAUUSD": 10,   # 2-stellig: 1 Pip = 0.10 = 10 Points
}

# Preis-Wert eines Pips je Symbol (fuer Dukascopy-Preis-Spreads -> Pips).
PIP_SIZE: dict[str, float] = {
    "EURUSD": 0.0001,
    "GBPUSD": 0.0001,
    "USDJPY": 0.01,
    "XAUUSD": 0.10,
}


def mt5_points_to_pips(points, symbol: str):
    """Rechnet MT5-Spread (POINTS je Bar) in Pips um. 10 Points = 1 Pip (EURUSD)."""
    ppp = POINTS_PER_PIP.get(symbol)
    if ppp is None:
        raise KeyError(f"Kein POINTS_PER_PIP fuer Symbol {symbol!r} hinterlegt.")
    return np.asarray(points, dtype=float) / ppp


def price_spread_to_pips(price_spread, symbol: str):
    """Rechnet einen Dukascopy-Spread in PREIS-Einheiten in Pips um."""
    pip = PIP_SIZE.get(symbol)
    if pip is None:
        raise KeyError(f"Kein PIP_SIZE fuer Symbol {symbol!r} hinterlegt.")
    return np.asarray(price_spread, dtype=float) / pip


# ── Session-Klassifikation (UTC) ─────────────────────────────────────────────
# Identische Fenster wie im EURUSD-Qualitaetsbericht.

SESSION_BOUNDS: list[tuple[str, int, int]] = [
    ("Asien",    0,  7),   # 00:00–06:59
    ("Europa",   7,  13),  # 07:00–12:59
    ("Overlap",  13, 16),  # 13:00–15:59
    ("US",       16, 21),  # 16:00–20:59
    ("Rollover", 21, 24),  # 21:00–23:59
]

SESSION_NAMES: list[str] = [name for name, _, _ in SESSION_BOUNDS]


def session_of_hour(hour: int) -> str:
    """Ordnet einer UTC-Stunde (0–23) die Handels-Session zu."""
    for name, lo, hi in SESSION_BOUNDS:
        if lo <= hour < hi:
            return name
    raise ValueError(f"Stunde ausserhalb 0–23: {hour}")


def assign_sessions(timestamps: pd.Series) -> pd.Series:
    """Vektorisierte Session-Zuordnung fuer eine Serie von UTC-Timestamps."""
    ts = pd.to_datetime(timestamps, utc=True)
    hours = ts.dt.hour
    out = pd.Series(index=ts.index, dtype="object")
    for name, lo, hi in SESSION_BOUNDS:
        out[(hours >= lo) & (hours < hi)] = name
    return out


# ── Kalibrierungs-Ergebnis ───────────────────────────────────────────────────

@dataclass
class SessionCalibration:
    session: str
    n_overlap: int
    duka_median_pips: float
    fusion_median_pips: float
    factor: float          # fusion_median / duka_median  (multiplikativ)
    additive_pips: float   # fusion_median - duka_median   (additiv)


@dataclass
class CalibrationResult:
    symbol: str
    overlap_start: pd.Timestamp
    overlap_end: pd.Timestamp
    overall: SessionCalibration
    sessions: dict[str, SessionCalibration] = field(default_factory=dict)


def _median(x: pd.Series) -> float:
    return float(np.median(x.to_numpy())) if len(x) else float("nan")


def _one_calibration(session: str, duka: pd.Series, fusion: pd.Series) -> SessionCalibration:
    dm = _median(duka)
    fm = _median(fusion)
    factor = fm / dm if dm and not np.isnan(dm) and dm != 0 else float("nan")
    return SessionCalibration(
        session=session,
        n_overlap=int(min(len(duka), len(fusion))),
        duka_median_pips=round(dm, 4),
        fusion_median_pips=round(fm, 4),
        factor=round(factor, 4),
        additive_pips=round(fm - dm, 4),
    )


def compute_calibration(
    duka: pd.DataFrame,
    fusion: pd.DataFrame,
    symbol: str,
    duka_spread_col: str = "spread_pips",
    fusion_spread_col: str = "spread_pips",
    ts_col: str = "timestamp",
) -> CalibrationResult:
    """Vergleicht Dukascopy- und Fusion-Spreads (beide in PIPS) im Overlap.

    Erwartet in beiden DataFrames eine Timestamp-Spalte (UTC) und eine
    Spread-Spalte in Pips. Bildet je Session (und gesamt) den Median beider
    Quellen und leitet Faktor + additiven Aufschlag ab. Der Vergleich laeuft
    ueber den Median je Session der jeweiligen Quelle (nicht bar-genau
    gematcht) – robust gegen kleine Zeitversaetze/Luecken.
    """
    d = duka[[ts_col, duka_spread_col]].dropna().copy()
    f = fusion[[ts_col, fusion_spread_col]].dropna().copy()
    d[ts_col] = pd.to_datetime(d[ts_col], utc=True)
    f[ts_col] = pd.to_datetime(f[ts_col], utc=True)

    start = max(d[ts_col].min(), f[ts_col].min())
    end = min(d[ts_col].max(), f[ts_col].max())
    d = d[(d[ts_col] >= start) & (d[ts_col] <= end)]
    f = f[(f[ts_col] >= start) & (f[ts_col] <= end)]

    d["session"] = assign_sessions(d[ts_col])
    f["session"] = assign_sessions(f[ts_col])

    overall = _one_calibration("ALL", d[duka_spread_col], f[fusion_spread_col])

    sessions: dict[str, SessionCalibration] = {}
    for name in SESSION_NAMES:
        ds = d.loc[d["session"] == name, duka_spread_col]
        fs = f.loc[f["session"] == name, fusion_spread_col]
        sessions[name] = _one_calibration(name, ds, fs)

    return CalibrationResult(
        symbol=symbol,
        overlap_start=start,
        overlap_end=end,
        overall=overall,
        sessions=sessions,
    )


def build_cost_model(
    *,
    symbol: str,
    commission_per_side_pips: float,
    duka_session_spread: Mapping[str, float],
    fusion_session_spread: Mapping[str, Mapping[str, float]],
    overlap: Mapping[str, str],
    sources: Mapping[str, str],
    measured_vs_assumed: Mapping[str, str],
    notes: str = "",
) -> dict:
    """Baut das kombinierte Kostenmodell (Spread + Kommission GETRENNT).

    Fuehrt die beiden Kostenkomponenten strikt getrennt:
      * ``commission_per_side_pips`` – fix, pro Seite (aus MT5-Deal-History).
      * ``effective_spread_by_session_pips`` – variabel, je Session (aus Ticks).

    Zusaetzlich das Mapping Dukascopy-Session-Spread -> Fusion-Effektiv-Spread
    (Faktor + additiver Aufschlag), damit das Labeling der langen Dukascopy-
    Historie (2016-2026), fuer die es KEINE Fusion-Ticks gibt, den Dukascopy-
    Spread auf Fusion-Niveau bringen und die fixe Kommission addieren kann.

    Kostenkonvention: Der Spread wird pro Round-Turn EINMAL voll gekreuzt
    (halb bei Entry, halb bei Exit); die Kommission faellt auf BEIDEN Seiten an
    -> round_turn = spread + 2 * commission_per_side.
    """
    rt_commission = round(2.0 * commission_per_side_pips, 4)

    sessions: dict[str, dict] = {}
    for name in SESSION_NAMES:
        dm = float(duka_session_spread.get(name, float("nan")))
        fj = fusion_session_spread.get(name, {})
        fm = float(fj.get("median", float("nan")))
        fp90 = float(fj.get("p90", float("nan")))
        n = int(fj.get("n", 0))
        factor = round(fm / dm, 4) if dm else float("nan")
        additive = round(fm - dm, 4)
        total_rt = round(fm + rt_commission, 4)
        sessions[name] = {
            "duka_spread_median_pips": round(dm, 4),
            "fusion_effective_spread_median_pips": round(fm, 4),
            "fusion_effective_spread_p90_pips": round(fp90, 4),
            "fusion_tick_n": n,
            "duka_to_fusion_factor": factor,
            "duka_to_fusion_additive_pips": additive,
            "total_roundturn_cost_pips": total_rt,
        }

    # Robuster Gesamt-Faktor (Median ueber die Sessions) als empfohlener
    # Default, da einzelne Session-Faktoren auf duenner Tick-Basis rauschen.
    factors = [s["duka_to_fusion_factor"] for s in sessions.values()
               if s["duka_to_fusion_factor"] == s["duka_to_fusion_factor"]]  # not NaN
    robust_factor = round(float(np.median(factors)), 4) if factors else float("nan")

    return {
        "symbol": symbol,
        "unit": "pips",
        "cost_convention": (
            "round_turn = effective_spread (1x voll gekreuzt) "
            "+ 2 * commission_per_side"
        ),
        "commission": {
            "per_side_pips": round(commission_per_side_pips, 4),
            "round_turn_pips": rt_commission,
        },
        "effective_spread_by_session_pips": {
            name: {
                "median": sessions[name]["fusion_effective_spread_median_pips"],
                "p90": sessions[name]["fusion_effective_spread_p90_pips"],
                "n": sessions[name]["fusion_tick_n"],
            }
            for name in SESSION_NAMES
        },
        "duka_to_fusion_spread_mapping": {
            name: {
                "factor": sessions[name]["duka_to_fusion_factor"],
                "additive_pips": sessions[name]["duka_to_fusion_additive_pips"],
            }
            for name in SESSION_NAMES
        },
        "recommended_robust_factor": robust_factor,
        "total_roundturn_cost_by_session_pips": {
            name: sessions[name]["total_roundturn_cost_pips"] for name in SESSION_NAMES
        },
        "sessions_detail": sessions,
        "session_bounds_utc": {name: [lo, hi] for name, lo, hi in SESSION_BOUNDS},
        "overlap": dict(overlap),
        "sources": dict(sources),
        "measured_vs_assumed": dict(measured_vs_assumed),
        "points_to_pips_note": (
            "MT5-Spread/Kommission in Konto-Ccy EUR; EURUSD 5-stellig, "
            "10 Points = 1 Pip; Pip-Wert per order_calc_profit gemessen."
        ),
        "notes": notes,
    }


def calibration_to_dict(res: CalibrationResult, *, method: str, sources: Mapping[str, str]) -> dict:
    """Serialisiert das Kalibrierungs-Ergebnis in ein YAML/JSON-taugliches dict."""
    def _sc(sc: SessionCalibration) -> dict:
        return {
            "n_overlap": sc.n_overlap,
            "duka_median_pips": sc.duka_median_pips,
            "fusion_median_pips": sc.fusion_median_pips,
            "factor": sc.factor,
            "additive_pips": sc.additive_pips,
        }

    return {
        "symbol": res.symbol,
        "method": method,
        "unit": "pips",
        "points_to_pips": {
            "note": "MT5-Spread in Points; EURUSD 5-stellig: 10 Points = 1 Pip",
            "points_per_pip": POINTS_PER_PIP.get(res.symbol),
        },
        "overlap": {
            "start": res.overlap_start.isoformat(),
            "end": res.overlap_end.isoformat(),
        },
        "sources": dict(sources),
        "overall": _sc(res.overall),
        "sessions": {name: _sc(sc) for name, sc in res.sessions.items()},
        "session_bounds_utc": {name: [lo, hi] for name, lo, hi in SESSION_BOUNDS},
    }
