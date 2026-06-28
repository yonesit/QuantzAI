"""
src/backtesting/wf_pnl.py
Walk-Forward-P&L-Sharpe – echte vectorbt-P&L statt Klassifikations-Proxy.

Hintergrund
-----------
SignalModel.walk_forward_validate() / _compute_sharpe() berechnet den
OOS-Sharpe aus kuenstlichen +1/-1-Renditen (Treffer/Fehltreffer der
Klassenvorhersage) – ohne Preise, ohne Lotgroesse, ohne Kosten.

Dieses Modul nutzt DIESELBE rollierende Fensterlogik und DIESELBEN
argmax-Signale, laesst die Trades aber von BacktestRunner mit echter
P&L und den in BacktestConfig konfigurierten Kosten (Spread, Slippage,
Swap, Kommission) simulieren. Ergebnis: ein direkt vergleichbarer,
realistischer P&L-Sharpe je Fenster.

Das Modell wird per ``model_factory`` injiziert (z.B.
``lambda: lightgbm.LGBMClassifier(**params)``), damit dieses Modul keine
harte Abhaengigkeit auf die Modell-Schicht hat und unabhaengig testbar bleibt.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.backtesting.vectorbt_runner import BacktestConfig, BacktestRunner

# Identisch zu SignalModel: Original-Label {-1, 0, 1} -> LightGBM-Klasse {0, 1, 2}
_LABEL_TO_CLASS: dict[int, int] = {-1: 0, 0: 1, 1: 2}
# argmax-Klasse -> Handelssignal (Klasse 1 = neutral = kein Trade)
_CLASS_TO_SIGNAL: dict[int, str] = {0: "short", 1: "flat", 2: "long"}


def compute_wf_pnl_sharpe(
    features_df: pd.DataFrame,
    labels: pd.Series,
    feat_cols: list[str],
    model_factory: Callable[[], Any],
    config: Optional[BacktestConfig] = None,
    timestamp_col: str = "timestamp",
    close_col: str = "close",
    train_months: int = 6,
    test_months: int = 1,
) -> list[dict[str, Any]]:
    """
    Rollierendes Walk-Forward-Backtesting mit echter vectorbt-P&L.

    Fensterlogik identisch zu SignalModel.walk_forward_validate(): 6M Training /
    1M Test, rollierend; ein Fenster wird uebersprungen wenn < 10 Train- oder
    < 2 Test-Zeilen vorliegen.

    Pro Fenster:
      1. model = model_factory(); model.fit(X_train, y_train)   (y in {0, 1, 2})
      2. argmax(predict_proba(X_test)) -> Signale long/short/flat
         (identisch zum Proxy – KEIN Confidence-Gate)
      3. BacktestRunner(config).run(close_test, signals) -> echter P&L-Sharpe

    Parameters
    ----------
    features_df   : DataFrame mit timestamp_col, close_col und den Feature-Spalten.
    labels        : Series mit Labels {-1, 0, 1}, gleicher Index wie features_df.
    feat_cols     : Liste der Feature-Spalten fuer das Modell.
    model_factory : Callable, das ein frisches Modell mit fit()/predict_proba()
                    zurueckgibt (predict_proba -> ndarray Form (n, 3)).
    config        : BacktestConfig (Kosten/freq); Default = BacktestConfig().
    timestamp_col : Name der Zeitstempel-Spalte.
    close_col     : Name der Schlusskurs-Spalte.
    train_months  : Laenge des Trainingsfensters in Monaten.
    test_months   : Laenge des Testfensters in Monaten.

    Returns
    -------
    Liste von dicts je Fenster mit: window, test_start, test_end, n_test,
    n_trades, oos_pnl_sharpe (float; 0.0 wenn vectorbt keinen Sharpe liefert).
    """
    if timestamp_col not in features_df.columns:
        raise ValueError(f"Spalte '{timestamp_col}' fehlt in features_df.")
    if close_col not in features_df.columns:
        raise ValueError(f"Spalte '{close_col}' fehlt in features_df.")

    cfg = config or BacktestConfig()
    ts = pd.to_datetime(features_df[timestamp_col])
    X_all = features_df[feat_cols]
    close_all = features_df[close_col]

    min_date = ts.min()
    max_date = ts.max()

    results: list[dict[str, Any]] = []
    window_idx = 0
    current = min_date

    while True:
        train_end = current + pd.DateOffset(months=train_months)
        test_end = train_end + pd.DateOffset(months=test_months)
        if test_end > max_date:
            break

        train_mask = (ts >= current) & (ts < train_end)
        test_mask = (ts >= train_end) & (ts < test_end)

        if train_mask.sum() < 10 or test_mask.sum() < 2:
            current += pd.DateOffset(months=test_months)
            continue

        X_train = X_all[train_mask].values.astype(float)
        y_train = np.array([_LABEL_TO_CLASS[int(v)] for v in labels[train_mask].values])
        X_test = X_all[test_mask].values.astype(float)

        model = model_factory()
        model.fit(X_train, y_train)
        proba = np.asarray(model.predict_proba(X_test))
        sig_classes = np.argmax(proba, axis=1)

        idx_test = pd.DatetimeIndex(ts[test_mask].values)
        close_test = pd.Series(
            close_all[test_mask].values.astype(float), index=idx_test, name="close"
        )
        signals = pd.Series(
            [_CLASS_TO_SIGNAL.get(int(c), "flat") for c in sig_classes],
            index=idx_test, dtype=object,
        )

        result = BacktestRunner(cfg).run(close_test, signals)

        results.append({
            "window": window_idx,
            "test_start": str(train_end.date()),
            "test_end": str(test_end.date()),
            "n_test": int(test_mask.sum()),
            "n_trades": int(result.n_trades),
            "oos_pnl_sharpe": float(result.sharpe_ratio),
        })
        logger.info(
            "WF-P&L Fenster {w} | Trades={n} | P&L-Sharpe={s:.3f}",
            w=window_idx, n=result.n_trades, s=result.sharpe_ratio,
        )

        window_idx += 1
        current += pd.DateOffset(months=test_months)

    return results


def aggregate_pnl_sharpe(results: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Aggregiert die fensterweisen P&L-Sharpes zu Mean/Median/Statistik.

    Mean und Median werden – wie beim Klassifikations-Proxy im Research-Log –
    ueber ALLE Fenster gebildet (inkl. Null-Sharpe-Fenster ohne Trades),
    damit der Vergleich fair bleibt.
    """
    if not results:
        return {
            "n_windows": 0, "mean_pnl_sharpe": None, "median_pnl_sharpe": None,
            "std_pnl_sharpe": None, "profitable_windows": 0, "total_trades": 0,
        }
    sharpes = np.array([r["oos_pnl_sharpe"] for r in results], dtype=float)
    profitable = int((sharpes > 0).sum())
    return {
        "n_windows": len(results),
        "mean_pnl_sharpe": float(sharpes.mean()),
        "median_pnl_sharpe": float(np.median(sharpes)),
        "std_pnl_sharpe": float(sharpes.std()),
        "profitable_windows": profitable,
        "total_trades": int(sum(r["n_trades"] for r in results)),
    }
