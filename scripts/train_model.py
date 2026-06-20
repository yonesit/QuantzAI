"""
scripts/train_model.py
Trainiert SignalModel auf gecachten Feature-Parquet-Dateien.

Ablauf:
  1. Feature-Parquet aus data/features/<symbol>_features.parquet laden
  2. LabelBuilder – Triple-Barrier-Labels erzeugen
  3. Walk-Forward-Validierung (6 Monate Train / 1 Monat Test)
  4. Monte-Carlo-Randomisierungstest (p < 0.05)
  5. Finales Modell auf allen Daten trainieren und speichern

Verwendung:
  python scripts/train_model.py --symbol EURUSD [--version 1]
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd
from loguru import logger

# Projekt-Root zum PYTHONPATH hinzufuegen damit src/* importierbar ist
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
            f"Fuehre zuerst 'python scripts/fetch_data.py --symbol {symbol} --tf {timeframe}' aus."
        )
    path = files[0]
    df = pd.read_parquet(path)
    logger.info("Features geladen: {path} | {n} Zeilen", path=path, n=len(df))
    return df


def _build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Waehlt Feature-Spalten aus (keine label/timestamp-aehnlichen Spalten)."""
    exclude = {"label", "timestamp", "open", "volume"}
    feat_cols = [c for c in df.columns if c not in exclude]
    return df[feat_cols + (["timestamp"] if "timestamp" in df.columns else [])], feat_cols


def main() -> int:
    parser = argparse.ArgumentParser(description="QuantzAI SignalModel Training")
    parser.add_argument("--symbol", required=True, help="Handelssymbol, z.B. EURUSD")
    parser.add_argument("--tf", default="H1", help="Timeframe, z.B. H1, M15, D1 (Standard: H1)")
    parser.add_argument("--version", type=int, default=1, help="Modellversion (Standard: 1)")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.55,
        help="Konfidenz-Schwellwert fuer get_signal (Standard: 0.55)",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=100,
        help="Anzahl Monte-Carlo-Permutationen (Standard: 100)",
    )
    args = parser.parse_args()

    logger.info("=== QuantzAI SignalModel Training | Symbol: {sym} TF: {tf} ===", sym=args.symbol, tf=args.tf)

    # 1. Features laden
    df = _load_features(args.symbol, args.tf)

    # 2. Labels erzeugen
    label_builder = LabelBuilder()
    labels = label_builder.build_labels(df)
    logger.info("Labels erzeugt | Verteilung: {v}", v=labels.value_counts().to_dict())

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
    if wf_results:
        sharpes = [r["oos_sharpe"] for r in wf_results]
        accs = [r["accuracy"] for r in wf_results]
        logger.info(
            "Walk-Forward abgeschlossen | Fenster: {w} | "
            "OOS-Sharpe mean={sm:.3f} | Accuracy mean={am:.3f}",
            w=len(wf_results),
            sm=sum(sharpes) / len(sharpes),
            am=sum(accs) / len(accs),
        )
    else:
        logger.warning("Keine Walk-Forward-Fenster – zu wenig Daten?")

    # 5. Finales Training auf allen Daten
    logger.info("Trainiere finales Modell auf allen {n} Datenpunkten...", n=len(df))
    feat_only = features_with_ts[[c for c in features_with_ts.columns if c != "timestamp"]]
    train_metrics = model.train(feat_only, labels)
    logger.info("Training abgeschlossen | {m}", m=train_metrics)

    # 6. Monte-Carlo-Randomisierungstest
    logger.info("Starte Monte-Carlo-Test ({p} Permutationen)...", p=args.n_permutations)
    mc_result = model.monte_carlo_test(feat_only, labels, n_permutations=args.n_permutations)
    if mc_result["significant"]:
        logger.info(
            "Monte-Carlo BESTANDEN: p={p:.4f} < 0.05 | real={r:.4f} vs perm_mean={m:.4f}",
            p=mc_result["p_value"],
            r=mc_result["real_score"],
            m=mc_result["permutation_mean"],
        )
    else:
        logger.warning(
            "Monte-Carlo NICHT BESTANDEN: p={p:.4f} >= 0.05 | "
            "Modell nicht signifikant besser als Zufall!",
            p=mc_result["p_value"],
        )

    # 7. Modell speichern
    save_path = build_save_path(args.version, date.today())
    model.save(save_path)
    logger.info("Fertig. Modell gespeichert: {path}", path=save_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
