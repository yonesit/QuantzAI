"""
src/models/signal_model.py
SignalModel – LightGBM-Wrapper fuer Triple-Barrier-Klassifikation.

Label-Mapping (LightGBM braucht 0-basierte Klassen):
  -1 (Short)   -> 0
   0 (Neutral) -> 1
   1 (Long)    -> 2

KI-Abstinenzregel: get_signal gibt 'flat' zurueck wenn
  max(probabilities) < confidence_threshold (Standard: 0.55)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from loguru import logger
from pathlib import Path
from datetime import date
from typing import Any


# Label-Mapping: original {-1, 0, 1} -> LightGBM {0, 1, 2}
_LABEL_TO_CLASS: dict[int, int] = {-1: 0, 0: 1, 1: 2}
_CLASS_TO_NAME: dict[int, str] = {0: "short", 1: "neutral", 2: "long"}
_NAME_TO_CLASS: dict[str, int] = {v: k for k, v in _CLASS_TO_NAME.items()}


class SignalModel:
    """
    LightGBM-Wrapper fuer Triple-Barrier-Signalklassifikation.

    Parameters
    ----------
    lgbm_params : dict mit LightGBM-Hyperparametern (optional).
    """

    def __init__(self, lgbm_params: dict[str, Any] | None = None) -> None:
        default_params: dict[str, Any] = {
            "objective": "multiclass",
            "num_class": 3,
            "num_leaves": 31,
            "learning_rate": 0.05,
            "n_estimators": 100,
            "random_state": 42,
            "verbose": -1,
        }
        if lgbm_params:
            default_params.update(lgbm_params)
        self._params = default_params
        self._model: lgb.LGBMClassifier | None = None
        self._feature_names: list[str] = []

    # ── Oeffentliche Schnittstelle ────────────────────────────────────────────

    def train(self, features_df: pd.DataFrame, labels: pd.Series) -> dict[str, Any]:
        """
        Trainiert das Modell auf den uebergebenen Features und Labels.

        Parameters
        ----------
        features_df : DataFrame mit Feature-Spalten (keine label/timestamp-Spalte).
        labels      : Series mit Labels {-1, 0, 1}.

        Returns
        -------
        dict mit Trainings-Metriken: n_samples, n_features, class_distribution.
        """
        X = features_df.values.astype(float)
        y_raw = labels.values
        y = np.array([_LABEL_TO_CLASS[int(v)] for v in y_raw])

        self._feature_names = list(features_df.columns)
        self._model = lgb.LGBMClassifier(**self._params)
        self._model.fit(X, y)

        unique, counts = np.unique(y_raw, return_counts=True)
        class_dist = {int(k): int(v) for k, v in zip(unique, counts)}
        metrics = {
            "n_samples": len(X),
            "n_features": X.shape[1],
            "class_distribution": class_dist,
        }
        logger.info("SignalModel trainiert | {metrics}", metrics=metrics)
        return metrics

    def predict_proba(self, features_row: pd.DataFrame | pd.Series) -> dict[str, float]:
        """
        Gibt Klassenwahrscheinlichkeiten zurueck.

        Parameters
        ----------
        features_row : Einzelne Zeile als DataFrame (1, n_features) oder Series.

        Returns
        -------
        dict mit Schluesseln 'long', 'short', 'neutral' und float-Wahrscheinlichkeiten.
        """
        self._require_trained()
        X = self._to_2d(features_row)
        proba = self._model.predict_proba(X)[0]  # shape (3,)
        return {
            "short":   float(proba[0]),
            "neutral": float(proba[1]),
            "long":    float(proba[2]),
        }

    def get_signal(
        self,
        features_row: pd.DataFrame | pd.Series,
        confidence_threshold: float = 0.55,
    ) -> str:
        """
        Gibt Handelssignal zurueck: 'long', 'short' oder 'flat'.

        KI-Abstinenzregel: wenn max(probabilities) < confidence_threshold -> 'flat'.
        """
        proba = self.predict_proba(features_row)
        max_prob = max(proba.values())
        if max_prob < confidence_threshold:
            return "flat"
        return max(proba, key=lambda k: proba[k])

    def save(self, path: str | Path) -> None:
        """Speichert Modell + Feature-Namen als .joblib-Datei."""
        self._require_trained()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self._model,
            "feature_names": self._feature_names,
            "params": self._params,
        }
        joblib.dump(payload, path)
        logger.info("SignalModel gespeichert -> {path}", path=path)

    @classmethod
    def load(cls, path: str | Path) -> "SignalModel":
        """Laedt ein gespeichertes Modell aus einer .joblib-Datei."""
        path = Path(path)
        payload = joblib.load(path)
        instance = cls(lgbm_params=payload["params"])
        instance._model = payload["model"]
        instance._feature_names = payload["feature_names"]
        logger.info("SignalModel geladen <- {path}", path=path)
        return instance

    # ── Walk-Forward-Validierung ──────────────────────────────────────────────

    def walk_forward_validate(
        self,
        features_df: pd.DataFrame,
        labels: pd.Series,
        timestamp_col: str = "timestamp",
        train_months: int = 6,
        test_months: int = 1,
    ) -> list[dict[str, Any]]:
        """
        Rollierendes Walk-Forward-Backtesting.

        Parameters
        ----------
        features_df   : DataFrame mit Feature-Spalten UND timestamp_col.
        labels        : Series mit Labels {-1, 0, 1}, gleicher Index wie features_df.
        timestamp_col : Name der Zeitstempel-Spalte in features_df.
        train_months  : Laenge des Trainingsfensters in Monaten.
        test_months   : Laenge des Testfensters in Monaten.

        Returns
        -------
        Liste von dicts je Fenster: window, train_size, test_size, oos_sharpe, accuracy.
        """
        if timestamp_col not in features_df.columns:
            raise ValueError(f"Spalte '{timestamp_col}' nicht in features_df gefunden.")

        ts = pd.to_datetime(features_df[timestamp_col])
        feat_cols = [c for c in features_df.columns if c != timestamp_col]
        X_all = features_df[feat_cols]

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
            y_train_raw = labels[train_mask].values
            y_train = np.array([_LABEL_TO_CLASS[int(v)] for v in y_train_raw])

            X_test = X_all[test_mask].values.astype(float)
            y_test_raw = labels[test_mask].values
            y_test = np.array([_LABEL_TO_CLASS[int(v)] for v in y_test_raw])

            model = lgb.LGBMClassifier(**self._params)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            accuracy = float((y_pred == y_test).mean())

            oos_sharpe = self._compute_sharpe(model, X_test, y_test_raw)

            window_result = {
                "window": window_idx,
                "train_start": str(current.date()),
                "train_end": str(train_end.date()),
                "test_start": str(train_end.date()),
                "test_end": str(test_end.date()),
                "train_size": int(train_mask.sum()),
                "test_size": int(test_mask.sum()),
                "oos_sharpe": oos_sharpe,
                "accuracy": accuracy,
            }
            results.append(window_result)
            logger.info(
                "Walk-Forward Fenster {w} | OOS-Sharpe: {s:.3f} | Accuracy: {a:.3f}",
                w=window_idx,
                s=oos_sharpe,
                a=accuracy,
            )

            window_idx += 1
            current += pd.DateOffset(months=test_months)

        return results

    def monte_carlo_test(
        self,
        features_df: pd.DataFrame,
        labels: pd.Series,
        n_permutations: int = 100,
        metric: str = "accuracy",
    ) -> dict[str, Any]:
        """
        Monte-Carlo-Randomisierungstest: prueft ob echtes Modell signifikant
        besser ist als auf zufaellig gemischten Labels (p < 0.05 erwuenscht).

        Returns
        -------
        dict mit: real_score, permutation_scores, p_value, significant.
        """
        self._require_trained()
        X = features_df.values.astype(float)
        y_raw = labels.values
        y = np.array([_LABEL_TO_CLASS[int(v)] for v in y_raw])

        real_score = self._score_model(self._model, X, y, metric)

        rng = np.random.default_rng(42)
        perm_scores: list[float] = []
        for _ in range(n_permutations):
            y_shuffled = rng.permutation(y)
            perm_model = lgb.LGBMClassifier(**self._params)
            perm_model.fit(X, y_shuffled)
            # Bewertung auf ORIGINALEN Labels y, nicht auf y_shuffled –
            # misst ob echte Label-Struktur besser vorhersagbar ist als zufaellige.
            perm_scores.append(self._score_model(perm_model, X, y, metric))

        perm_arr = np.array(perm_scores)
        p_value = float((perm_arr >= real_score).mean())
        significant = p_value < 0.05

        result = {
            "real_score": real_score,
            "permutation_mean": float(perm_arr.mean()),
            "permutation_std": float(perm_arr.std()),
            "p_value": p_value,
            "significant": significant,
            "n_permutations": n_permutations,
        }
        logger.info(
            "Monte-Carlo | real={r:.4f} perm_mean={m:.4f} p={p:.4f} signifikant={s}",
            r=real_score,
            m=perm_arr.mean(),
            p=p_value,
            s=significant,
        )
        return result

    # ── Hilfsmethoden ─────────────────────────────────────────────────────────

    def _require_trained(self) -> None:
        if self._model is None:
            raise RuntimeError("SignalModel wurde noch nicht trainiert. Rufe train() zuerst auf.")

    def _to_2d(self, row: pd.DataFrame | pd.Series) -> np.ndarray:
        if isinstance(row, pd.DataFrame) and self._feature_names:
            # Nur die beim Training verwendeten Spalten in korrekter Reihenfolge –
            # verhindert Fehler wenn der uebergebene DataFrame zusaetzliche Spalten
            # (z.B. close, high, low aus dem Parquet) enthaelt.
            return row[self._feature_names].values.astype(float)
        if isinstance(row, pd.Series):
            return row.values.reshape(1, -1).astype(float)
        return row.values.astype(float)

    def _score_model(
        self,
        model: lgb.LGBMClassifier,
        X: np.ndarray,
        y: np.ndarray,
        metric: str,
    ) -> float:
        if metric == "accuracy":
            preds = model.predict(X)
            return float((preds == y).mean())
        raise ValueError(f"Unbekannte Metrik: {metric}")

    def _compute_sharpe(
        self,
        model: lgb.LGBMClassifier,
        X_test: np.ndarray,
        y_raw: np.ndarray,
    ) -> float:
        """Berechnet vereinfachten OOS Sharpe Ratio basierend auf Signalen."""
        proba = model.predict_proba(X_test)
        signals = np.argmax(proba, axis=1)  # 0=short, 1=neutral, 2=long

        # Rendite: +1 bei korrektem Long, -1 bei falschem Long, etc.
        y_mapped = np.array([_LABEL_TO_CLASS[int(v)] for v in y_raw])
        returns: list[float] = []
        for sig, true_cls in zip(signals, y_mapped):
            if sig == 1:  # neutral -> kein Trade
                continue
            r = 1.0 if sig == true_cls else -1.0
            returns.append(r)

        if len(returns) < 2:
            return 0.0
        arr = np.array(returns)
        std = arr.std()
        if std == 0:
            return 0.0
        return float(arr.mean() / std * np.sqrt(252))


def build_save_path(version: int, model_date: date | None = None) -> Path:
    """Erstellt den Speicherpfad gemaess Namenskonvention."""
    d = model_date or date.today()
    fname = f"signal_model_v{version}_{d.strftime('%Y%m%d')}.joblib"
    return Path("models") / fname
