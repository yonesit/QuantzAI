"""
src/models/pnl_backtest.py
Reine, testbare P&L-/Sharpe-Bausteine fuer die kostenbewusste Baseline.

Kernidee: Die realisierte Netto-P&L eines Long-Entries steckt bereits
(kostenbereinigt) in der Spalte net_pips_* des gelabelten Datensatzes. Fixes
Risiko pro Trade (z.B. 1 %) am SL-Abstand normiert diese Pips auf eine
Equity-Rendite – KEINE All-in-Position, damit der Sharpe realistisch ist.

Trade-Rendite (in Equity-Anteilen):
    return_frac = risk_frac * net_pips / sl_distance_pips
wobei sl_distance_pips = sl_mult * k * ATR / pip (das je-Trade riskierte
Vol-normierte Delta). Ein Voll-Verlust am SL entspricht damit exakt risk_frac.
"""

from __future__ import annotations

import numpy as np


def sl_distance_pips(atr_price: np.ndarray, k: float, sl_mult: float, pip: float) -> np.ndarray:
    """SL-Abstand je Bar in Pips = sl_mult * k * ATR / pip."""
    return sl_mult * k * np.asarray(atr_price, dtype=float) / pip


def trade_returns(
    net_pips: np.ndarray,
    atr_price: np.ndarray,
    *,
    k: float,
    sl_mult: float,
    pip: float,
    risk_frac: float = 0.01,
) -> np.ndarray:
    """Vol-normierte Equity-Renditen je Trade bei fixem Risiko pro Trade."""
    sl_pips = sl_distance_pips(atr_price, k, sl_mult, pip)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = risk_frac * np.asarray(net_pips, dtype=float) / sl_pips
    r[~np.isfinite(r)] = 0.0
    return r


def sharpe(returns: np.ndarray, periods_per_year: float) -> float:
    """Annualisiertes Sharpe der Renditereihe. <2 Werte oder std=0 -> 0.0."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return 0.0
    sd = r.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(r.mean() / sd * np.sqrt(periods_per_year))


def max_drawdown(returns: np.ndarray) -> float:
    """Maximaler Drawdown der kumulierten (additiven) Equity-Kurve, als Anteil."""
    r = np.asarray(returns, dtype=float)
    if len(r) == 0:
        return 0.0
    equity = 1.0 + np.cumsum(r)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak)
    return float(dd.min())


def profit_factor(returns: np.ndarray) -> float:
    """Summe Gewinne / |Summe Verluste|. Keine Verluste -> inf."""
    r = np.asarray(returns, dtype=float)
    gains = r[r > 0].sum()
    losses = -r[r < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def win_rate(returns: np.ndarray) -> float:
    r = np.asarray(returns, dtype=float)
    if len(r) == 0:
        return 0.0
    return float((r > 0).mean())


def periods_per_year_from_trades(n_trades: int, n_years: float) -> float:
    """Annualisierungsfaktor = durchschnittliche Trades pro Jahr (>=1)."""
    if n_years <= 0:
        return 1.0
    return max(1.0, n_trades / n_years)


def non_overlapping_mask(positions: np.ndarray, horizon: int) -> np.ndarray:
    """Waehlt aus aufsteigenden Bar-Positionen ueberschneidungsfreie Entries.

    Modelliert einen Bot mit EINER Position zur Zeit: nach einem Entry bei Bar p
    wird erst wieder ein Entry >= p + horizon zugelassen. Verhindert die
    Sharpe-Inflation durch massiv ueberlappende 4h-Trades im 15-min-Raster.
    """
    pos = np.asarray(positions, dtype=int)
    keep = np.zeros(len(pos), dtype=bool)
    last_exit = -1
    for i, p in enumerate(pos):
        if p >= last_exit:
            keep[i] = True
            last_exit = p + horizon
    return keep


def daily_sharpe(returns: np.ndarray, entry_ts: np.ndarray,
                 day_start, day_end) -> float:
    """Interpretierbares Sharpe auf TAEGLICH aggregierten Renditen (sqrt(252)).

    Trades werden nach Kalendertag (UTC) summiert; handelsfreie Tage im
    Testfenster zaehlen als 0-Rendite. So ist der Sharpe unabhaengig von der
    Trade-Frequenz vergleichbar.
    """
    import pandas as pd
    if len(returns) == 0:
        return 0.0
    s = pd.Series(np.asarray(returns, dtype=float),
                  index=pd.DatetimeIndex(entry_ts).floor("D"))
    daily = s.groupby(level=0).sum()
    idx = pd.date_range(pd.Timestamp(day_start).floor("D"),
                        pd.Timestamp(day_end).floor("D"), freq="D")
    daily = daily.reindex(idx, fill_value=0.0)
    return sharpe(daily.to_numpy(), periods_per_year=252)
