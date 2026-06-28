"""
src/backtesting/vectorbt_runner.py
BacktestRunner – vectorbt-basiertes Backtesting mit realistischen Kosten.

Realistische Kosten:
  - Spread: als prozentualer Anteil des Handelsvolumens (fees-Parameter)
  - Slippage: konfigurierbar in Pips, in Preisanteil umgerechnet
  - Swap/Overnight-Kosten: pro gehaltener Nacht, post-Simulation berechnet

Kennzahlen:
  Gesamtertrag, Sharpe, Sortino, Max-Drawdown, Gewinnfaktor,
  Win-Rate, Avg-Gewinn/-Verlust pro Trade, Equity-Curve

In-Sample / Out-of-Sample:
  Trennbar per is_mask (Boolean-Series) oder is_end (Datum-String).
  Overfitting-Warnung wenn IS-Sharpe >> OOS-Sharpe (konfigurierbarer Schwellwert).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import vectorbt as vbt
from loguru import logger


# ── Annualisierungsfaktoren fuer Sharpe/Sortino ───────────────────────────────
_ANN_FACTOR: dict[str, int] = {
    "1min": 525_600, "5min": 105_120, "15min": 35_040,
    "30min": 17_520, "1h": 8_760, "4h": 2_190,
    "1d": 252, "1w": 52,
}


@dataclass
class BacktestConfig:
    """
    Konfiguration fuer den BacktestRunner.

    Parameters
    ----------
    init_cash                   : Startkapital in Kontowaehrung.
    spread_pct                  : Spread als Anteil des Handelspreises pro Seite
                                  (z.B. 0.0001 = 1 Pip bei EUR/USD um 1.0).
    slippage_pips               : Zusaetzlicher Slippage-Aufschlag in Pips.
    pip_size                    : Pip-Groesse des Instruments (Standard: 0.0001).
    swap_long_per_night         : Swap-Kosten pro gehaltener Nacht fuer Long-Positionen
                                  (in Kontowaehrung, positiv = Kosten).
    swap_short_per_night        : Swap-Kosten pro gehaltener Nacht fuer Short-Positionen.
    freq                        : Pandas-Frequenz-String fuer Annualisierung
                                  ('1h', '4h', '1d' usw.).
    overfitting_sharpe_threshold: Absolute Differenz IS-Sharpe minus OOS-Sharpe,
                                  ab der eine Overfitting-Warnung ausgegeben wird.
    """
    init_cash: float = 10_000.0
    spread_pct: float = 0.0001
    slippage_pips: float = 1.0
    pip_size: float = 0.0001
    swap_long_per_night: float = 0.0
    swap_short_per_night: float = 0.0
    freq: str = "1h"
    overfitting_sharpe_threshold: float = 0.5


@dataclass
class BacktestResult:
    """
    Ergebnisse eines Backtests.

    Attributes
    ----------
    total_return        : Gesamtertrag als Dezimalzahl (0.1 = +10 %).
    sharpe_ratio        : Annualisierter Sharpe Ratio.
    sortino_ratio       : Annualisierter Sortino Ratio.
    max_drawdown        : Maximaler Drawdown (negativ, z.B. -0.15 = -15 %).
    profit_factor       : Brutto-Gewinn / Brutto-Verlust; inf wenn kein Verlust.
    win_rate            : Anteil gewinnender Trades (0.0–1.0).
    avg_win             : Durchschnittlicher Gewinn je gewinnendem Trade.
    avg_loss            : Durchschnittlicher Verlust je verlierendem Trade (negativ).
    n_trades            : Gesamtanzahl abgeschlossener Trades.
    equity_curve        : Zeitreihe des Portfolio-Werts (DatetimeIndex).
    is_sharpe           : Sharpe Ratio im In-Sample-Zeitraum (None wenn nicht berechnet).
    oos_sharpe          : Sharpe Ratio im Out-of-Sample-Zeitraum.
    overfitting_warning : True wenn IS-Sharpe deutlich besser als OOS-Sharpe.
    """
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    profit_factor: float
    win_rate: float
    avg_win: float
    avg_loss: float
    n_trades: int
    equity_curve: pd.Series
    is_sharpe: Optional[float] = None
    oos_sharpe: Optional[float] = None
    overfitting_warning: bool = False


class BacktestRunner:
    """
    Fuehrt Backtests mit vectorbt durch.

    Nimmt historische Close-Preise und Signale ('long'/'short'/'flat')
    entgegen, simuliert Trades mit realistischen Kosten und berechnet
    gaengige Performance-Kennzahlen.

    Parameters
    ----------
    config : BacktestConfig – alle Parameter zum Backtest (optional,
             wird mit Standardwerten benutzt wenn None).
    """

    def __init__(self, config: Optional[BacktestConfig] = None) -> None:
        self._cfg = config or BacktestConfig()

    # ── Oeffentliche Schnittstelle ────────────────────────────────────────────

    def run(
        self,
        close: pd.Series,
        signals: pd.Series,
        is_mask: Optional[pd.Series] = None,
    ) -> BacktestResult:
        """
        Fuehrt Backtest auf gegebener Close-Preis-Serie mit Signalen aus.

        Parameters
        ----------
        close   : pd.Series mit DatetimeIndex und Close-Preisen.
        signals : pd.Series mit gleichem Index, Werte 'long'/'short'/'flat'.
        is_mask : Optionale Boolean-Series (gleiches Index),
                  True = In-Sample, False = Out-of-Sample.

        Returns
        -------
        BacktestResult mit allen Kennzahlen und Equity-Curve.
        """
        cfg = self._cfg

        entries, exits, short_entries, short_exits = self._signals_to_entries(signals)

        avg_price = float(close.mean()) if len(close) > 0 else 1.0
        fees_per_side = cfg.spread_pct / 2.0
        slippage_per_side = (cfg.slippage_pips * cfg.pip_size) / avg_price / 2.0

        # Look-Ahead-Fix (Schritt D): Signale entstehen aus den Daten der Kerze i,
        # ausgefuehrt wird aber zum Preis der FOLGEkerze (i+1) – nicht zum Close
        # derselben Kerze, der das Signal erst erzeugt hat.
        exec_price = self._execution_price(close)

        pf = vbt.Portfolio.from_signals(
            close=close,
            entries=entries,
            exits=exits,
            short_entries=short_entries,
            short_exits=short_exits,
            price=exec_price,
            fees=fees_per_side,
            slippage=slippage_per_side,
            init_cash=cfg.init_cash,
            freq=cfg.freq,
        )

        equity_curve = pf.value()
        trade_records = pf.trades.records_readable
        pnl_adjusted = self._apply_swap_costs(trade_records, cfg)
        # Swap-Kosten fliessen in die Equity-Serie ein, BEVOR der Sharpe darauf
        # berechnet wird – sonst bleibt der Sharpe swap-blind (Bug-Fix Schritt B).
        equity_for_sharpe = self._swap_adjusted_equity(equity_curve, trade_records, cfg)
        result = self._compute_metrics(
            pf, pnl_adjusted, equity_curve, equity_for_sharpe
        )

        if is_mask is not None:
            is_sharpe, oos_sharpe = self._compute_is_oos_sharpe(
                equity_for_sharpe, is_mask, cfg
            )
            result.is_sharpe = is_sharpe
            result.oos_sharpe = oos_sharpe

            if (
                is_sharpe is not None
                and oos_sharpe is not None
                and (is_sharpe - oos_sharpe) > cfg.overfitting_sharpe_threshold
            ):
                result.overfitting_warning = True
                logger.warning(
                    "Overfitting-Warnung: IS-Sharpe={is_s:.3f} >> OOS-Sharpe={oos_s:.3f} "
                    "(Differenz {diff:.3f} > Schwellwert {thr:.3f})",
                    is_s=is_sharpe,
                    oos_s=oos_sharpe,
                    diff=is_sharpe - oos_sharpe,
                    thr=cfg.overfitting_sharpe_threshold,
                )

        logger.info(
            "Backtest abgeschlossen | Trades={n} | Return={r:.2%} | "
            "Sharpe={s:.3f} | MaxDD={dd:.2%}",
            n=result.n_trades,
            r=result.total_return,
            s=result.sharpe_ratio,
            dd=result.max_drawdown,
        )
        return result

    def run_with_model(
        self,
        features_df: pd.DataFrame,
        signal_func: Callable[[pd.DataFrame], str],
        close_col: str = "close",
        is_end: Optional[str] = None,
    ) -> BacktestResult:
        """
        Backtest mit Signal-Funktion (z.B. SignalModel.get_signal) auf Feature-DataFrame.

        Parameters
        ----------
        features_df : DataFrame mit DatetimeIndex, Feature-Spalten und close_col.
        signal_func : Callable (row_df: pd.DataFrame) -> 'long'/'short'/'flat'.
                      Wird zeilenweise auf features_df angewendet.
        close_col   : Spaltenname fuer den Close-Preis in features_df.
        is_end      : ISO-Datum-String fuer IS/OOS-Trennung (z.B. '2023-12-31').
                      Alles <= is_end gilt als In-Sample.

        Returns
        -------
        BacktestResult mit allen Kennzahlen inklusive IS/OOS-Sharpe wenn is_end gesetzt.
        """
        if close_col not in features_df.columns:
            raise ValueError(
                f"Spalte '{close_col}' nicht in features_df. "
                f"Vorhandene Spalten: {list(features_df.columns)}"
            )

        close = features_df[close_col]

        logger.info(
            "Signalerzeugung fuer {n} Kerzen ...", n=len(features_df)
        )
        signals_raw = [signal_func(features_df.iloc[[i]]) for i in range(len(features_df))]
        signals = pd.Series(signals_raw, index=features_df.index, dtype=object)

        is_mask: Optional[pd.Series] = None
        if is_end is not None:
            is_end_ts = pd.Timestamp(is_end)
            if features_df.index.tz is not None and is_end_ts.tzinfo is None:
                is_end_ts = is_end_ts.tz_localize(features_df.index.tz)
            is_mask = pd.Series(
                features_df.index <= is_end_ts, index=features_df.index
            )

        return self.run(close=close, signals=signals, is_mask=is_mask)

    # ── Signal-Konvertierung ──────────────────────────────────────────────────

    @staticmethod
    def _signals_to_entries(
        signals: pd.Series,
    ) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Konvertiert 'long'/'short'/'flat'-Signale in vectorbt Entry/Exit-Boolean-Serien.

        Entry faeuert wenn das Signal auf die Richtung wechselt.
        Exit faeuert wenn das Signal von der Richtung wegwechselt.
        """
        sig = signals.values
        n = len(sig)

        entries       = np.zeros(n, dtype=bool)
        exits         = np.zeros(n, dtype=bool)
        short_entries = np.zeros(n, dtype=bool)
        short_exits   = np.zeros(n, dtype=bool)

        prev = "flat"
        for i, cur in enumerate(sig):
            cur = str(cur).lower()
            if cur == "long" and prev != "long":
                entries[i] = True
            if cur != "long" and prev == "long":
                exits[i] = True
            if cur == "short" and prev != "short":
                short_entries[i] = True
            if cur != "short" and prev == "short":
                short_exits[i] = True
            prev = cur

        idx = signals.index
        return (
            pd.Series(entries,       index=idx),
            pd.Series(exits,         index=idx),
            pd.Series(short_entries, index=idx),
            pd.Series(short_exits,   index=idx),
        )

    # ── Ausfuehrungspreis (Look-Ahead-Fix) ────────────────────────────────────

    @staticmethod
    def _execution_price(close: pd.Series) -> pd.Series:
        """
        Ausfuehrungspreis = Preis der FOLGEkerze (Look-Ahead-Fix, Schritt D).

        Ein Signal aus Kerze i darf nicht zum Close von Kerze i ausgefuehrt
        werden (genau der Preis, der das Signal erzeugt hat), sondern fruehestens
        zum naechsten verfuegbaren Preis. Hier: Close der Folgekerze
        (``close.shift(-1)``). Die letzte Kerze hat keinen Nachfolger und behaelt
        ihren eigenen Close (kein Trade nach dem Datenende moeglich).
        """
        return close.shift(-1).fillna(close)

    # ── Swap-Kosten ───────────────────────────────────────────────────────────

    @staticmethod
    def _apply_swap_costs(
        trade_records: pd.DataFrame,
        cfg: BacktestConfig,
    ) -> np.ndarray:
        """
        Berechnet swap-bereinigtes PnL pro Trade.

        Swap-Kosten werden anhand der Haltedauer in Naechten berechnet
        und vom rohen PnL abgezogen.

        Returns
        -------
        np.ndarray mit bereinigtem PnL je Trade (leeres Array bei keinen Trades).
        """
        if trade_records.empty:
            return np.array([], dtype=float)

        pnl = trade_records["PnL"].values.copy().astype(float)

        if cfg.swap_long_per_night == 0.0 and cfg.swap_short_per_night == 0.0:
            return pnl

        for idx in range(len(trade_records)):
            row = trade_records.iloc[idx]
            try:
                entry_ts = pd.Timestamp(row["Entry Timestamp"])
                exit_ts  = pd.Timestamp(row["Exit Timestamp"])
                nights   = max(0, (exit_ts - entry_ts).days)
                direction = str(row.get("Direction", "")).lower()
                if "short" in direction:
                    swap_cost = nights * cfg.swap_short_per_night
                else:
                    swap_cost = nights * cfg.swap_long_per_night
                pnl[idx] -= swap_cost
            except Exception as exc:  # noqa: BLE001
                logger.debug("Swap-Berechnung Fehler fuer Trade {i}: {exc}", i=idx, exc=exc)

        return pnl

    @staticmethod
    def _swap_adjusted_equity(
        equity_curve: pd.Series,
        trade_records: pd.DataFrame,
        cfg: BacktestConfig,
    ) -> pd.Series:
        """
        Zieht Swap-Kosten von der Equity-Serie ab, BEVOR daraus der Sharpe
        berechnet wird.

        Pro Trade werden die Swap-Kosten (Naechte * Swap/Nacht) ab dem
        Exit-Zeitpunkt kumulativ von der Equity abgezogen. So schlaegt sich die
        Overnight-Finanzierung in der Rendite-Serie nieder – und damit im
        Sharpe –, nicht nur im Pro-Trade-PnL.

        Returns die unveraenderte Equity-Serie wenn keine Swap-Kosten
        konfiguriert sind oder keine Trades vorliegen.
        """
        if cfg.swap_long_per_night == 0.0 and cfg.swap_short_per_night == 0.0:
            return equity_curve
        if trade_records is None or trade_records.empty:
            return equity_curve

        deductions = pd.Series(0.0, index=equity_curve.index)
        for idx in range(len(trade_records)):
            row = trade_records.iloc[idx]
            try:
                entry_ts = pd.Timestamp(row["Entry Timestamp"])
                exit_ts  = pd.Timestamp(row["Exit Timestamp"])
                nights   = max(0, (exit_ts - entry_ts).days)
                if nights == 0:
                    continue
                direction = str(row.get("Direction", "")).lower()
                per_night = (
                    cfg.swap_short_per_night if "short" in direction
                    else cfg.swap_long_per_night
                )
                deductions.loc[deductions.index >= exit_ts] += nights * per_night
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "Swap-Equity-Anpassung Fehler fuer Trade {i}: {exc}", i=idx, exc=exc
                )

        return equity_curve - deductions

    # ── Kennzahlen ────────────────────────────────────────────────────────────

    def _compute_metrics(
        self,
        pf: vbt.Portfolio,
        pnl_adjusted: np.ndarray,
        equity_curve: pd.Series,
        equity_for_sharpe: Optional[pd.Series] = None,
    ) -> BacktestResult:
        """Berechnet alle Kennzahlen aus Portfolio und bereinigtem PnL."""
        total_return = float(pf.total_return())
        max_drawdown = float(pf.max_drawdown())

        # Sharpe aus der (swap-bereinigten) Equity-Serie statt aus
        # pf.sharpe_ratio(), damit konfigurierte Swap-Kosten tatsaechlich in die
        # Kennzahl einfliessen. equity_for_sharpe == equity_curve wenn Swap=0.
        eq_for_sharpe = equity_for_sharpe if equity_for_sharpe is not None else equity_curve
        sharpe_val  = _safe_float(pnl_sharpe(eq_for_sharpe.pct_change(), self._cfg.freq))
        sortino_val = _safe_float(pf.sortino_ratio())

        n_trades = len(pnl_adjusted)

        if n_trades > 0:
            wins   = pnl_adjusted[pnl_adjusted > 0]
            losses = pnl_adjusted[pnl_adjusted < 0]

            win_rate      = float(len(wins) / n_trades)
            gross_profit  = float(wins.sum())
            gross_loss    = float(abs(losses.sum()))
            profit_factor = (
                gross_profit / gross_loss if gross_loss > 0
                else (float("inf") if gross_profit > 0 else 0.0)
            )
            avg_win  = float(wins.mean())   if len(wins)   > 0 else 0.0
            avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0
        else:
            win_rate = profit_factor = avg_win = avg_loss = 0.0

        return BacktestResult(
            total_return=total_return,
            sharpe_ratio=sharpe_val,
            sortino_ratio=sortino_val,
            max_drawdown=max_drawdown,
            profit_factor=profit_factor,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            n_trades=n_trades,
            equity_curve=equity_curve,
        )

    @staticmethod
    def _compute_is_oos_sharpe(
        equity_curve: pd.Series,
        is_mask: pd.Series,
        cfg: BacktestConfig,
    ) -> tuple[Optional[float], Optional[float]]:
        """Berechnet Sharpe Ratio fuer In-Sample- und Out-of-Sample-Zeitraum."""

        ann = _ANN_FACTOR.get(cfg.freq.lower(), 252)

        def _sharpe(eq: pd.Series) -> Optional[float]:
            ret = eq.pct_change().dropna()
            if len(ret) < 2:
                return None
            std = float(ret.std())
            if std == 0.0:
                return None
            return float(ret.mean() / std * np.sqrt(ann))

        is_idx  = is_mask[is_mask].index
        oos_idx = is_mask[~is_mask].index

        is_eq  = equity_curve[equity_curve.index.isin(is_idx)]
        oos_eq = equity_curve[equity_curve.index.isin(oos_idx)]

        return _sharpe(is_eq), _sharpe(oos_eq)


# ── Hilfsfunktionen ───────────────────────────────────────────────────────────

def _safe_float(value: Any) -> float:
    """Gibt 0.0 zurueck wenn value None oder NaN ist."""
    try:
        v = float(value)
        return 0.0 if np.isnan(v) or np.isinf(v) else v
    except (TypeError, ValueError):
        return 0.0


def pnl_sharpe(returns: "pd.Series | np.ndarray", freq: str = "1h") -> Optional[float]:
    """
    Annualisierter Sharpe Ratio aus einer ECHTEN Rendite-/P&L-Serie.

    Abgrenzung zum Klassifikations-Proxy in SignalModel._compute_sharpe():
    Dort werden kuenstliche +1/-1-Werte aus Treffer/Fehltreffer der
    Klassenvorhersage gebildet (ohne Preise, ohne Lotgroesse, ohne Kosten).
    Diese Funktion erwartet dagegen echte Renditen, z.B. aus der
    vectorbt-Equity-Curve (``equity.pct_change()``) – inklusive der via
    BacktestConfig simulierten Kosten (Spread, Slippage, Swap, Kommission).

    Parameters
    ----------
    returns : Rendite-Serie (prozentuale Equity-Veraenderung je Bar).
    freq    : Pandas-Frequenz-String fuer die Annualisierung ('4h', '1d' ...).

    Returns
    -------
    Annualisierter Sharpe als float, oder None wenn < 2 Werte oder Std == 0.
    """
    arr = np.asarray(pd.Series(returns).dropna(), dtype=float)
    if arr.size < 2:
        return None
    std = float(arr.std())
    if std == 0.0:
        return None
    ann = _ANN_FACTOR.get(freq.lower(), 252)
    return float(arr.mean() / std * np.sqrt(ann))


# Symbolspezifische Pip-Groessen (sonst Forex-Default 0.0001).
# XAUUSD: 2 Dezimalstellen -> 0.01 (sonst ist die Slippage fuer Gold faktisch
# null, weil (slippage_pips * 0.0001) bei Goldpreis ~1800 verschwindet).
# JPY-Paare: 3 Dezimalstellen -> 0.01.
_SYMBOL_PIP_SIZE: dict[str, float] = {
    "XAUUSD": 0.01,
    "XAGUSD": 0.001,
    "USDJPY": 0.01,
    "EURJPY": 0.01,
    "GBPJPY": 0.01,
}


def pip_size_for_symbol(symbol: str, default: float = 0.0001) -> float:
    """
    Gibt die korrekte Pip-Groesse fuer ein Symbol zurueck.

    Forex-Majors: 0.0001 (Default). XAUUSD: 0.01. JPY-Paare: 0.01.
    Verhindert den Bug, dass die Backtest-Slippage fuer Gold mit dem
    Forex-Default 0.0001 faktisch null wird.
    """
    return _SYMBOL_PIP_SIZE.get(symbol.upper(), default)


def timeframe_to_freq(timeframe: str) -> str:
    """
    Konvertiert MT5-Zeitrahmen-Notation in pandas-Frequenz-String.

    Beispiele: 'H1' -> '1h', 'H4' -> '4h', 'D1' -> '1d', 'M5' -> '5min'.
    """
    mapping = {
        "M1": "1min", "M5": "5min", "M15": "15min", "M30": "30min",
        "H1": "1h",   "H4": "4h",
        "D1": "1d",   "W1": "1w",
    }
    freq = mapping.get(timeframe.upper())
    if freq is None:
        logger.warning(
            "Unbekannter Zeitrahmen '{tf}' – verwende '1h' als Fallback.", tf=timeframe
        )
        return "1h"
    return freq
