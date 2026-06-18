"""
src/models/interpretability.py
SHAP-basierte Modell-Interpretierbarkeit fuer QuantzAI.

WARNUNG: Nicht im Live-Pfad verwenden. Nur fuer Backtesting und manuelles
Debugging geeignet. SHAP-Berechnung ist rechenintensiv (O(n * features)).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from loguru import logger
from typing import Any

from src.models.signal_model import SignalModel


def explain_prediction(
    model: SignalModel,
    features_row: pd.DataFrame | pd.Series,
) -> dict[str, dict[str, float]]:
    """
    Berechnet SHAP-Werte fuer eine einzelne Vorhersage.

    WARNUNG: Nicht im Live-Pfad verwenden.

    Parameters
    ----------
    model        : Trainiertes SignalModel.
    features_row : Einzelne Zeile als DataFrame (1, n_features) oder Series.

    Returns
    -------
    dict mit Klassenname -> {feature_name: shap_value}.
    Klassen: 'short', 'neutral', 'long'.

    Hinweis
    -------
    SHAP-Werte addieren sich zum log-odds-Beitrag jeder Klasse
    (Abweichung vom Erwartungswert), nicht direkt zur Wahrscheinlichkeit.
    Konsistenz: sum(shap_values) + expected_value == raw model output (log-odds).
    """
    lgbm_model = _get_lgbm(model)
    feature_names = model._feature_names

    X = _to_2d(features_row, feature_names)
    explainer = shap.TreeExplainer(lgbm_model)
    shap_vals = explainer.shap_values(X)  # list[ndarray] oder ndarray shape (1, n_feat, n_classes)

    shap_arr = _normalize_shap(shap_vals)  # shape (1, n_features, 3)
    class_names = ["short", "neutral", "long"]

    result: dict[str, dict[str, float]] = {}
    for cls_idx, cls_name in enumerate(class_names):
        row_vals = shap_arr[0, :, cls_idx]
        result[cls_name] = {
            fn: float(v) for fn, v in zip(feature_names, row_vals)
        }
    return result


def explain_batch(
    model: SignalModel,
    features_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Berechnet SHAP-Werte fuer einen ganzen Backtest-Zeitraum.

    WARNUNG: Nicht im Live-Pfad verwenden.

    Parameters
    ----------
    model       : Trainiertes SignalModel.
    features_df : DataFrame mit Feature-Spalten (n_samples, n_features).
                  Darf timestamp-Spalte enthalten; wird ignoriert.

    Returns
    -------
    dict mit Klassenname -> DataFrame (n_samples, n_features) der SHAP-Werte.
    """
    lgbm_model = _get_lgbm(model)
    feature_names = model._feature_names

    feat_cols = [c for c in features_df.columns if c in feature_names]
    X = features_df[feat_cols].values.astype(float)

    logger.info(
        "SHAP Batch-Erklaerung | {n} Samples | {f} Features",
        n=len(X),
        f=len(feat_cols),
    )

    explainer = shap.TreeExplainer(lgbm_model)
    shap_vals = explainer.shap_values(X)
    shap_arr = _normalize_shap(shap_vals)  # shape (n, n_features, 3)

    class_names = ["short", "neutral", "long"]
    result: dict[str, pd.DataFrame] = {}
    for cls_idx, cls_name in enumerate(class_names):
        df_shap = pd.DataFrame(
            shap_arr[:, :, cls_idx],
            columns=feat_cols,
            index=features_df.index,
        )
        result[cls_name] = df_shap

    return result


def get_expected_values(model: SignalModel) -> dict[str, float]:
    """
    Gibt die SHAP-Erwartungswerte (base values) pro Klasse zurueck.

    Der Erwartungswert ist der mittlere Modell-Output (log-odds) im Trainingsdatensatz.
    SHAP-Werte + Erwartungswert = roher Modell-Output fuer diese Klasse.
    """
    lgbm_model = _get_lgbm(model)
    explainer = shap.TreeExplainer(lgbm_model)
    ev = explainer.expected_value  # array[3] oder liste
    ev_arr = np.asarray(ev).flatten()
    class_names = ["short", "neutral", "long"]
    return {cls: float(ev_arr[i]) for i, cls in enumerate(class_names)}


# ── Hilfsmethoden ─────────────────────────────────────────────────────────────

def _get_lgbm(model: SignalModel):
    if model._model is None:
        raise RuntimeError("SignalModel nicht trainiert. Rufe train() zuerst auf.")
    return model._model


def _to_2d(row: pd.DataFrame | pd.Series, feature_names: list[str]) -> np.ndarray:
    if isinstance(row, pd.Series):
        return row[feature_names].values.reshape(1, -1).astype(float)
    return row[feature_names].values.astype(float)


def _normalize_shap(shap_vals) -> np.ndarray:
    """
    Normalisiert SHAP-Output auf shape (n_samples, n_features, n_classes).

    LightGBM TreeExplainer gibt je nach Version entweder:
      - list[ndarray(n,f)] der Laenge n_classes
      - ndarray(n, f, n_classes)
    """
    if isinstance(shap_vals, list):
        # list[ndarray(n, f)] -> stack zu (n, f, n_classes)
        return np.stack(shap_vals, axis=-1)
    arr = np.asarray(shap_vals)
    if arr.ndim == 3:
        return arr  # schon (n, f, n_classes)
    raise ValueError(f"Unerwartete SHAP-Shape: {arr.shape}")
