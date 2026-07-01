"""
src/data/triple_barrier_labeler.py
Kostenbewusstes Triple-Barrier-Labeling (Long-Perspektive) fuer M15-Forex.

Idee: Fuer jeden Entry-Kandidaten (Signal auf Bar t, Ausfuehrung zum OPEN von
Bar t+1 – kein Look-Ahead) werden drei Barrieren gesetzt:
  * TP (obere Barriere) = entry + tp_mult * ATR[t]
  * SL (untere Barriere) = entry - sl_mult * ATR[t]
  * vertikale Barriere   = max. Haltehorizont in Bars

Die zuerst beruehrte Barriere bestimmt den Exit; bei gleichzeitigem TP/SL in
derselben Kerze gewinnt pessimistisch der SL.

DAS ENTSCHEIDENDE: Ein Trade ist nur dann Label=1 ("gut"), wenn der erzielte
Kurs-Move NACH der vollstaendigen, gemessenen Kostenkette noch im Plus liegt.
Die Kosten stecken also IM Label:
  * Effektiver Spread : Dukascopy-Bar-Spread (Entry- und Exit-Bar, je halb)
                        x recommended_robust_factor. Weil der Dukascopy-Spread
                        pro Bar vorliegt, traegt ein Exit im Rollover-Fenster
                        automatisch den erhoehten Rollover-Spread (Teil C.6).
  * Kommission        : fixe Round-Turn-Kommission, IMMER abgezogen.
  * Slippage          : fixer konservativer Wert pro Seite (Round-Turn = 2x).
  * Swap              : nur wenn der Trade einen Rollover (22:00 UTC) ueberspannt.

Label-Konvention (je Design):
  * -1 : SL zuerst getroffen (Verlust)
  *  1 : netto profitabel (Exit-Move schlaegt Gesamtkosten)
  *  0 : Timeout ODER Brutto-Gewinn, den die Kosten aufgefressen haben
  * NA : kein Label (No-Trade-Zone / Datenluecke / Spike / Randbereich) -> siehe
         Spalte ``status``

Zwei Barrieren-Designs werden parallel ausgegeben:
  * symmetric : TP = SL = k*ATR
  * asymmetric: TP = 2k*ATR, SL = k*ATR (Chance-Risiko 2:1)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger

from src.data.spread_calibration import assign_sessions

PIP_EURUSD = 0.0001


# ── Konfiguration ────────────────────────────────────────────────────────────

@dataclass
class CostConfig:
    """Kostenparameter in PIPS (ausser pip = Preiswert eines Pips)."""
    pip: float = PIP_EURUSD
    spread_factor: float = 0.667          # recommended_robust_factor (Duka->Fusion)
    commission_roundturn_pips: float = 0.464
    slippage_per_side_pips: float = 0.2
    swap_long_pips: float = 0.0           # pro Overnight, Long
    swap_short_pips: float = 0.0          # pro Overnight, Short (hier ungenutzt: Long-Only)


@dataclass
class BarrierConfig:
    horizon: int = 16                     # max. Haltebars (16 * M15 = 4 h)
    atr_period: int = 14
    k: float = 1.5                        # Basis-ATR-Multiplikator
    no_trade_hours: tuple[int, ...] = (21, 22)   # 21:00-22:59 UTC No-Trade
    spike_z: float = 15.0                 # MAD-robust-z Schwelle fuer Extrem-Spikes
    gap_max_minutes: float = 60.0         # groesserer Bar-Gap im Fenster -> skip


DESIGNS: dict[str, tuple[float, float]] = {
    # name -> (tp_mult_faktor, sl_mult_faktor) relativ zu k
    "symmetric": (1.0, 1.0),
    "asymmetric": (2.0, 1.0),
}

_STATUS_LABELED = "labeled"


# ── Hilfsfunktionen (rein, testbar) ──────────────────────────────────────────

def compute_atr(df: pd.DataFrame, period: int) -> np.ndarray:
    """Wilder-ATR ueber high/low/close. Erste `period` Werte sind NaN."""
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    prev_close = np.concatenate(([np.nan], close[:-1]))
    tr = np.maximum.reduce([
        high - low,
        np.abs(high - prev_close),
        np.abs(low - prev_close),
    ])
    # Wilder-Glaettung via ewm(alpha=1/period)
    atr = pd.Series(tr).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    return atr.to_numpy()


def spike_mask(close: np.ndarray, z_thresh: float) -> np.ndarray:
    """MAD-robust-z auf Log-Returns; True wo |z| > z_thresh (Extrem-Spike)."""
    ret = np.diff(np.log(close), prepend=np.nan)
    med = np.nanmedian(ret)
    mad = np.nanmedian(np.abs(ret - med))
    if mad == 0 or np.isnan(mad):
        return np.zeros_like(close, dtype=bool)
    z = 0.6745 * (ret - med) / mad
    return np.abs(z) > z_thresh


def crosses_rollover(entry_ns: int, exit_ns: int) -> bool:
    """True wenn ein 22:00-UTC-Rollover im Intervall (entry, exit] liegt."""
    entry_ts = pd.Timestamp(entry_ns, tz="UTC")
    exit_ts = pd.Timestamp(exit_ns, tz="UTC")
    day = pd.Timedelta(days=1)
    bases = [entry_ts.normalize() - day, entry_ts.normalize(),
             entry_ts.normalize() + day]
    for base in bases:
        roll = base + pd.Timedelta(hours=22)
        if entry_ts < roll <= exit_ts:
            return True
    return False


def net_pips_long(
    gross_price: float,
    sp_entry_pips: float,
    sp_exit_pips: float,
    overnight: bool,
    cost: CostConfig,
) -> tuple[float, float]:
    """Netto-PnL (Pips) und Gesamtkosten (Pips) fuer einen Long-Exit.

    Spread wird pro Seite halb angerechnet (Entry-Bar + Exit-Bar), Kommission
    und Slippage als Round-Turn, Swap nur bei Overnight.
    """
    spread_pips = cost.spread_factor * 0.5 * (sp_entry_pips + sp_exit_pips)
    total = spread_pips + cost.commission_roundturn_pips + 2.0 * cost.slippage_per_side_pips
    if overnight:
        total += cost.swap_long_pips
    net = gross_price / cost.pip - total
    return net, total


# ── Kern-Barrieren-Scan ──────────────────────────────────────────────────────

def _scan_barrier(
    highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
    entry_price: float, tp: float, sl: float, first: int, last: int,
) -> tuple[str, int, float]:
    """Scannt Bars [first, last]; gibt (outcome, exit_idx, exit_price).

    outcome in {"TP","SL","TIMEOUT"}. Bei TP&SL in derselben Kerze -> SL.
    """
    for k in range(first, last + 1):
        tp_hit = highs[k] >= tp
        sl_hit = lows[k] <= sl
        if tp_hit and sl_hit:
            return "SL", k, sl
        if sl_hit:
            return "SL", k, sl
        if tp_hit:
            return "TP", k, tp
    return "TIMEOUT", last, closes[last]


# ── Oeffentliche Hauptfunktion ───────────────────────────────────────────────

def label_dataframe(
    df: pd.DataFrame,
    *,
    cost: CostConfig,
    barrier: BarrierConfig | None = None,
    spread_col: str = "spread_median",
    ts_col: str = "timestamp",
) -> pd.DataFrame:
    """Erzeugt kostenbewusste Triple-Barrier-Labels fuer beide Designs.

    Erwartet OHLC + `spread_col` (Dukascopy-Spread in PREIS-Einheiten) +
    `ts_col` (UTC). Gibt eine Kopie mit Zusatzspalten zurueck:
      status, session, atr,
      outcome_/gross_pips_/net_pips_/cost_pips_/label_  je Design.
    """
    barrier = barrier or BarrierConfig()
    out = df.copy().reset_index(drop=True)
    n = len(out)

    ts = pd.to_datetime(out[ts_col], utc=True)
    hours = ts.dt.hour.to_numpy()
    times_ns = pd.DatetimeIndex(ts).asi8        # UTC-ns als int64
    opens = out["open"].to_numpy(dtype=float)
    highs = out["high"].to_numpy(dtype=float)
    lows = out["low"].to_numpy(dtype=float)
    closes = out["close"].to_numpy(dtype=float)
    sp_pips = out[spread_col].to_numpy(dtype=float) / cost.pip
    atr = compute_atr(out, barrier.atr_period)
    spikes = spike_mask(closes, barrier.spike_z)
    sessions = assign_sessions(ts).to_numpy()

    status = np.array([_STATUS_LABELED] * n, dtype=object)
    # Ergebnis-Container je Design
    res = {d: {
        "outcome": np.array([None] * n, dtype=object),
        "gross": np.full(n, np.nan),
        "net": np.full(n, np.nan),
        "costp": np.full(n, np.nan),
        "label": np.full(n, np.nan),
    } for d in DESIGNS}

    gap_max_ns = barrier.gap_max_minutes * 60 * 1_000_000_000
    no_trade = set(barrier.no_trade_hours)

    for t in range(n):
        # Randbereich / ATR
        if np.isnan(atr[t]) or atr[t] <= 0.0:
            status[t] = "insufficient_atr"
            continue
        if hours[t] in no_trade:
            status[t] = "no_trade"
            continue
        if t + 1 >= n:
            status[t] = "insufficient_future"
            continue
        first = t + 1
        last = min(t + barrier.horizon, n - 1)
        if last < first:
            status[t] = "insufficient_future"
            continue
        # Datenluecke (Wochenende / fehlender Tag 2023-01-12): Fenster ueberspannt
        # eine Marktschliessung -> nicht labeln.
        window = times_ns[t:last + 1]
        if np.any(np.diff(window) > gap_max_ns):
            status[t] = "gap_skip"
            continue
        # Extrem-Spike im Entry- oder Haltefenster -> fake Wick moeglich -> skip.
        if spikes[t] or np.any(spikes[first:last + 1]):
            status[t] = "spike_skip"
            continue

        entry_price = opens[first]
        for d, (tp_f, sl_f) in DESIGNS.items():
            tp = entry_price + tp_f * barrier.k * atr[t]
            sl = entry_price - sl_f * barrier.k * atr[t]
            outcome, xidx, xprice = _scan_barrier(
                highs, lows, closes, entry_price, tp, sl, first, last)
            gross = xprice - entry_price
            res[d]["outcome"][t] = outcome
            res[d]["gross"][t] = gross / cost.pip
            if outcome == "SL":
                res[d]["label"][t] = -1
                # Netto/Kosten fuer SL informativ (Label bleibt -1)
                overnight = crosses_rollover(times_ns[first], times_ns[xidx])
                net, cp = net_pips_long(gross, sp_pips[t], sp_pips[xidx], overnight, cost)
                res[d]["net"][t] = net
                res[d]["costp"][t] = cp
            else:
                overnight = crosses_rollover(times_ns[first], times_ns[xidx])
                net, cp = net_pips_long(gross, sp_pips[t], sp_pips[xidx], overnight, cost)
                res[d]["net"][t] = net
                res[d]["costp"][t] = cp
                res[d]["label"][t] = 1 if net > 0 else 0

    out["status"] = status
    out["session"] = sessions
    out["atr"] = atr
    for d in DESIGNS:
        out[f"outcome_{d}"] = res[d]["outcome"]
        out[f"gross_pips_{d}"] = res[d]["gross"]
        out[f"net_pips_{d}"] = res[d]["net"]
        out[f"cost_pips_{d}"] = res[d]["costp"]
        out[f"label_{d}"] = res[d]["label"]

    _log_summary(out)
    return out


def _log_summary(out: pd.DataFrame) -> None:
    total = len(out)
    labeled = int((out["status"] == _STATUS_LABELED).sum())
    logger.info("Triple-Barrier-Labeling | {tot} Bars | {lab} gelabelt | Status: {st}",
                tot=total, lab=labeled,
                st={k: int(v) for k, v in out["status"].value_counts().items()})
