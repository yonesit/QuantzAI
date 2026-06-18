"""
scripts/shap_analysis.py
SHAP-Analyse fuer QuantzAI SignalModel – exportiert Plots als PNG.

Warum PNG statt Jupyter-Notebook:
  Notebooks mischen Code, Output und State in einer JSON-Datei, was
  saubere Git-Diffs, reproduzierbare Laeufe in CI und automatisiertes
  Debugging erschwert. Ein Skript ist direkt ausfuehrbar (cron, pipeline),
  versionierbar und produziert deterministisch benannte Artefakte.

Erzeugte Plots (Ausgabeordner: reports/shap/):
  - shap_summary_long.png    : Bee-Swarm-Plot fuer Klasse 'long'
  - shap_summary_short.png   : Bee-Swarm-Plot fuer Klasse 'short'
  - shap_bar_importance.png  : Mittlere absolute SHAP-Werte (alle Klassen)
  - shap_single_<idx>.png    : Waterfall-Plot fuer einen einzelnen Sample

Verwendung:
  python scripts/shap_analysis.py --model models/signal_model_v1_YYYYMMDD.joblib
                                  --features data/features/EURUSD_features.parquet
                                  [--output reports/shap]
                                  [--sample-idx 0]

WARNUNG: Nicht im Live-Pfad verwenden.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # kein Display noetig (headless / CI)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.signal_model import SignalModel
from src.models.interpretability import explain_batch, get_expected_values, _normalize_shap


# ── Plot-Funktionen ───────────────────────────────────────────────────────────

def plot_summary(
    shap_arr: np.ndarray,
    X: np.ndarray,
    feature_names: list[str],
    cls_idx: int,
    cls_name: str,
    out_path: Path,
) -> None:
    """Bee-Swarm-Summary-Plot fuer eine Klasse."""
    fig, ax = plt.subplots(figsize=(10, max(4, len(feature_names) * 0.4)))
    shap.summary_plot(
        shap_arr[:, :, cls_idx],
        X,
        feature_names=feature_names,
        plot_type="dot",
        show=False,
        ax=ax,
    )
    ax.set_title(f"SHAP Summary – Klasse '{cls_name}'", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Gespeichert: {p}", p=out_path)


def plot_bar_importance(
    shap_arr: np.ndarray,
    feature_names: list[str],
    out_path: Path,
) -> None:
    """Balkendiagramm mittlerer absoluter SHAP-Werte (alle Klassen kombiniert)."""
    # mean |shap| ueber alle Samples und alle Klassen
    mean_abs = np.abs(shap_arr).mean(axis=(0, 2))  # shape (n_features,)
    sorted_idx = np.argsort(mean_abs)[::-1]
    top_n = min(20, len(feature_names))
    idx = sorted_idx[:top_n]

    fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.45)))
    ax.barh(
        [feature_names[i] for i in reversed(idx)],
        mean_abs[list(reversed(idx))],
        color="#1f77b4",
    )
    ax.set_xlabel("Mittlerer absoluter SHAP-Wert")
    ax.set_title("Feature Importance (SHAP, alle Klassen)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Gespeichert: {p}", p=out_path)


def plot_waterfall_single(
    shap_vals_row: np.ndarray,
    expected_value: float,
    feature_names: list[str],
    sample_idx: int,
    cls_name: str,
    out_path: Path,
) -> None:
    """Waterfall-Plot fuer einen einzelnen Sample."""
    top_n = min(15, len(feature_names))
    sorted_idx = np.argsort(np.abs(shap_vals_row))[::-1][:top_n]

    vals = shap_vals_row[sorted_idx]
    names = [feature_names[i] for i in sorted_idx]
    cumsum = expected_value + np.cumsum(vals)

    fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.45)))
    colors = ["#d73027" if v > 0 else "#4575b4" for v in vals]
    ax.barh(names[::-1], vals[::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP-Wert")
    ax.set_title(
        f"SHAP Waterfall – Sample #{sample_idx} | Klasse '{cls_name}'\n"
        f"Erwartungswert: {expected_value:.4f}  |  "
        f"Modell-Output: {expected_value + shap_vals_row.sum():.4f}"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Gespeichert: {p}", p=out_path)


# ── Hauptlogik ────────────────────────────────────────────────────────────────

def run_analysis(
    model_path: str | Path,
    features_path: str | Path,
    output_dir: str | Path,
    sample_idx: int = 0,
) -> dict[str, Path]:
    """
    Fuehrt SHAP-Analyse durch und speichert alle Plots als PNG.

    Returns
    -------
    dict mit Plot-Name -> Dateipfad.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Modell laden
    logger.info("Lade Modell: {p}", p=model_path)
    model = SignalModel.load(model_path)

    # Features laden
    logger.info("Lade Features: {p}", p=features_path)
    df = pd.read_parquet(features_path)
    feature_names = model._feature_names
    feat_cols = [c for c in feature_names if c in df.columns]
    X = df[feat_cols].values.astype(float)

    # SHAP berechnen
    logger.info("Berechne SHAP-Werte fuer {n} Samples...", n=len(X))
    lgbm_model = model._model
    explainer = shap.TreeExplainer(lgbm_model)
    shap_vals = explainer.shap_values(X)
    shap_arr = _normalize_shap(shap_vals)  # (n, f, 3)
    ev_arr = np.asarray(explainer.expected_value).flatten()

    class_names = ["short", "neutral", "long"]
    saved: dict[str, Path] = {}

    # Summary-Plots
    for cls_idx, cls_name in enumerate(class_names):
        if cls_name == "neutral":
            continue  # neutral-Summary meist weniger informativ
        p = out_dir / f"shap_summary_{cls_name}.png"
        plot_summary(shap_arr, X, feat_cols, cls_idx, cls_name, p)
        saved[f"summary_{cls_name}"] = p

    # Bar-Importance
    p = out_dir / "shap_bar_importance.png"
    plot_bar_importance(shap_arr, feat_cols, p)
    saved["bar_importance"] = p

    # Waterfall fuer einen einzelnen Sample
    idx = min(sample_idx, len(X) - 1)
    cls_idx_max = int(np.argmax(np.abs(shap_arr[idx]).mean(axis=0)))
    cls_name_max = class_names[cls_idx_max]
    p = out_dir / f"shap_single_{idx}.png"
    plot_waterfall_single(
        shap_arr[idx, :, cls_idx_max],
        float(ev_arr[cls_idx_max]),
        feat_cols,
        idx,
        cls_name_max,
        p,
    )
    saved[f"waterfall_{idx}"] = p

    logger.info("SHAP-Analyse abgeschlossen. {n} Plots gespeichert.", n=len(saved))
    return saved


def main() -> int:
    parser = argparse.ArgumentParser(description="QuantzAI SHAP-Analyse")
    parser.add_argument("--model", required=True, help="Pfad zur .joblib-Modelldatei")
    parser.add_argument(
        "--features", required=True, help="Pfad zur Feature-Parquet-Datei"
    )
    parser.add_argument(
        "--output", default="reports/shap", help="Ausgabeordner fuer PNGs"
    )
    parser.add_argument(
        "--sample-idx", type=int, default=0, help="Sample-Index fuer Waterfall-Plot"
    )
    args = parser.parse_args()

    run_analysis(args.model, args.features, args.output, args.sample_idx)
    return 0


if __name__ == "__main__":
    sys.exit(main())
