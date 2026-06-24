"""
scripts/train_model.py
Trainiert SignalModel auf gecachten Feature-Parquet-Dateien.

Ablauf:
  1. Feature-Parquet aus data/features/<symbol>_<tf>_*.parquet laden
  2. LabelBuilder – Triple-Barrier-Labels erzeugen
  3. Walk-Forward-Validierung (6 Monate Train / 1 Monat Test)
  4. Monte-Carlo-Randomisierungstest (p < 0.05)
  5. Finales Modell auf allen Daten trainieren und speichern
  6. JSON-Ergebnisdatei fuer Forschungslog speichern

Verwendung:
  python scripts/train_model.py --symbol EURUSD --tf M15 [--max-candles 48]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.label_builder import LabelBuilder
from src.models.signal_model import SignalModel, build_save_path


def _load_features(symbol: str, timeframe: str = "H1") -> pd.DataFrame:
    features_dir = Path("data") / "features"
    pattern = f"{symbol.upper()}_{timeframe.upper()}_*.parquet"
    files = sorted(features_dir.glob(pattern), reverse=True)
    if not files:
        raise FileNotFoundError(
            f"Feature-Datei nicht gefunden: {features_dir / pattern}\n"
            f"Fuehre zuerst aus:\n"
            f"  python scripts/fetch_data.py --symbol {symbol} --tf {timeframe} --start 2020-01-01 --end 2024-01-01"
        )
    path = files[0]
    df = pd.read_parquet(path)
    logger.info("Features geladen: {path} | {n} Zeilen", path=path, n=len(df))
    return df


def _build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Waehlt Feature-Spalten aus (ohne Preis-Lecks)."""
    exclude = {"label", "timestamp", "open", "volume", "close", "high", "low"}
    feat_cols = [c for c in df.columns if c not in exclude]
    return df[feat_cols + (["timestamp"] if "timestamp" in df.columns else [])], feat_cols


def _wf_stats(sharpes: list[float]) -> dict:
    """Berechnet vollstaendige Walk-Forward-Statistiken."""
    n = len(sharpes)
    mean   = statistics.mean(sharpes)
    median = statistics.median(sharpes)
    std    = statistics.stdev(sharpes) if n > 1 else 0.0
    n_pos  = sum(1 for s in sharpes if s > 0)

    # Ausreisser: > 2 Std vom Median entfernt
    if n > 4:
        lower = median - 2 * std
        upper = median + 2 * std
        outliers = [
            {"window": i, "sharpe": round(s, 3)}
            for i, s in enumerate(sharpes)
            if s < lower or s > upper
        ]
    else:
        outliers = []

    return {
        "n_windows":     n,
        "mean":          round(mean,   3),
        "median":        round(median, 3),
        "std":           round(std,    3),
        "n_profitable":  n_pos,
        "pct_profitable": round(n_pos / n * 100, 1) if n > 0 else 0.0,
        "outliers":      outliers,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="QuantzAI SignalModel Training")
    parser.add_argument("--symbol",   required=True, help="Handelssymbol, z.B. EURUSD")
    parser.add_argument("--tf",       default="H1",  help="Timeframe (Standard: H1)")
    parser.add_argument("--version",  type=int, default=1)
    parser.add_argument("--max-candles",  type=int,   default=24,
                        help="Triple-Barrier-Zeitlimit in Kerzen (Standard: 24)")
    parser.add_argument("--tp-atr-mult",  type=float, default=2.0,
                        help="TP-ATR-Multiplikator (Standard: 2.0)")
    parser.add_argument("--sl-atr-mult",  type=float, default=1.5,
                        help="SL-ATR-Multiplikator (Standard: 1.5)")
    parser.add_argument("--confidence-threshold", type=float, default=0.55)
    parser.add_argument("--n-permutations",       type=int,   default=50,
                        help="Monte-Carlo-Permutationen (Standard: 50)")
    parser.add_argument("--save-results", action="store_true",
                        help="JSON-Ergebnisdatei neben dem Modell speichern")
    args = parser.parse_args()

    logger.info(
        "=== QuantzAI SignalModel Training | {sym} {tf} | "
        "TP={tp}x SL={sl}x max_candles={mc} ===",
        sym=args.symbol, tf=args.tf,
        tp=args.tp_atr_mult, sl=args.sl_atr_mult, mc=args.max_candles,
    )

    # 1. Features laden
    df = _load_features(args.symbol, args.tf)

    # 2. Labels erzeugen
    label_builder = LabelBuilder(
        tp_atr_mult=args.tp_atr_mult,
        sl_atr_mult=args.sl_atr_mult,
        max_candles=args.max_candles,
    )
    labels = label_builder.build_labels(df)
    n_trades = int((labels != 0).sum())
    label_dist = labels.value_counts().to_dict()
    logger.info("Labels erzeugt | Verteilung: {v} | Trades: {t}", v=label_dist, t=n_trades)

    # 3. Feature-Matrix aufbauen
    features_with_ts, feat_cols = _build_feature_matrix(df)
    logger.info("Feature-Matrix: {n} Spalten", n=len(feat_cols))

    # 4. Walk-Forward-Validierung
    model = SignalModel()
    logger.info("Starte Walk-Forward-Validierung (6M Train / 1M Test)...")
    wf_results = model.walk_forward_validate(
        features_with_ts,
        labels,
        timestamp_col="timestamp",
        train_months=6,
        test_months=1,
    )

    stats: dict = {}
    all_sharpes: list[float] = []
    if wf_results:
        all_sharpes = [r["oos_sharpe"] for r in wf_results]
        stats = _wf_stats(all_sharpes)
        logger.info(
            "Walk-Forward | {w} Fenster | "
            "Ø Sharpe={m:.3f} | Median={med:.3f} | Std={s:.3f} | "
            "Profitabel={p}/{w} ({pp:.1f}%)",
            w=stats["n_windows"], m=stats["mean"], med=stats["median"],
            s=stats["std"], p=stats["n_profitable"], pp=stats["pct_profitable"],
        )
        if stats["outliers"]:
            logger.warning(
                "Ausreisser (>2 Std): {o}",
                o=[(f"Fenster {x['window']}", x["sharpe"]) for x in stats["outliers"]],
            )
    else:
        logger.warning("Keine Walk-Forward-Fenster – zu wenig Daten?")

    # 5. Finales Training auf allen Daten
    logger.info("Trainiere finales Modell auf {n} Datenpunkten...", n=len(df))
    feat_only = features_with_ts[[c for c in features_with_ts.columns if c != "timestamp"]]
    train_metrics = model.train(feat_only, labels)
    logger.info("Training abgeschlossen | {m}", m=train_metrics)

    # Top-3-Features aus Feature-Importance
    top_features: list[str] = []
    try:
        importances = model._clf.feature_importances_
        ranked = sorted(zip(feat_cols, importances), key=lambda x: x[1], reverse=True)
        top_features = [f for f, _ in ranked[:3]]
        logger.info("Top-3 Features (Importance): {f}", f=top_features)
    except Exception:
        pass

    # 6. Monte-Carlo-Randomisierungstest
    logger.info("Starte Monte-Carlo-Test ({p} Permutationen)...", p=args.n_permutations)
    mc_result = model.monte_carlo_test(feat_only, labels, n_permutations=args.n_permutations)
    if mc_result["significant"]:
        logger.info(
            "Monte-Carlo BESTANDEN: p={p:.4f} < 0.05 | real={r:.4f} vs perm_mean={m:.4f}",
            p=mc_result["p_value"], r=mc_result["real_score"], m=mc_result["permutation_mean"],
        )
    else:
        logger.warning(
            "Monte-Carlo NICHT BESTANDEN: p={p:.4f} >= 0.05",
            p=mc_result["p_value"],
        )

    # 7. Modell speichern (Symbol + TF im Dateinamen)
    save_path = build_save_path(
        args.version, date.today(), timeframe=args.tf, symbol=args.symbol
    )
    model.save(save_path)
    logger.info("Modell gespeichert: {path}", path=save_path)

    # 8. JSON-Ergebnisdatei
    tf_part = f"_{args.tf.upper()}" if args.tf.upper() != "H4" else ""
    results_path = Path("models") / f"{args.symbol.upper()}{tf_part}_wf_results_{date.today().strftime('%Y%m%d')}.json"
    results = {
        "symbol":            args.symbol.upper(),
        "timeframe":         args.tf.upper(),
        "date":              str(date.today()),
        "label_tp":          args.tp_atr_mult,
        "label_sl":          args.sl_atr_mult,
        "label_max_candles": args.max_candles,
        "n_trades_total":    n_trades,
        "n_features":        len(feat_cols),
        "wf":                stats,
        "all_sharpes":       [round(s, 3) for s in all_sharpes],
        "top_features":      top_features,
        "mc_significant":    mc_result.get("significant", False),
        "mc_pvalue":         round(mc_result.get("p_value", 1.0), 4),
        "model_path":        str(save_path),
    }
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Ergebnisse gespeichert: {p}", p=results_path)

    # Zusammenfassung
    sep = "=" * 60
    logger.info(
        "\n{sep}\nERGEBNIS: {sym} {tf}\n"
        "  Oe OOS-Sharpe : {m:.3f}\n"
        "  Median Sharpe: {med:.3f}\n"
        "  Std Sharpe   : {s:.3f}\n"
        "  Profitabel   : {p}/{w} ({pp:.1f}%)\n"
        "  Trades gesamt: {t}\n"
        "  Monte Carlo  : p={mc:.4f} ({sig})\n"
        "  Top Features : {feat}\n"
        "{sep}",
        sep=sep, sym=args.symbol, tf=args.tf,
        m=stats.get("mean", 0), med=stats.get("median", 0), s=stats.get("std", 0),
        p=stats.get("n_profitable", 0), w=stats.get("n_windows", 0),
        pp=stats.get("pct_profitable", 0), t=n_trades,
        mc=mc_result.get("p_value", 1.0),
        sig="signifikant" if mc_result.get("significant") else "NICHT signifikant",
        feat=top_features,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
