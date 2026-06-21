"""
src/models/performance_attribution.py
Performance-Attribution: rollierende SHAP-Aggregation und Feature-Drift-Erkennung.

NICHT im kritischen Live-Pfad. Nur asynchron nutzen (run_async_update oder
manuell einmal taeglich aufrufen).

Ablauf:
  1. Pro Trade: SHAP-Werte in Rolling-Window (Groesse N, Standard 100) speichern
  2. Aggregation: mittlerer absoluter SHAP-Wert pro Feature -> Feature-Importance
  3. Drift-Erkennung: Vergleich mit Trainings-Baseline per relativem Aenderungsmass
  4. Warnung wenn wichtiges Feature seinen Edge verliert (Re-Training-Empfehlung)

Asynchroner Workflow (empfohlen):
  attr = PerformanceAttribution()
  thread = attr.run_async_update(model, features_df)   # startet Daemon-Thread
  thread.join()                                         # optional: warten
  summary = attr.compute_attribution()                  # Ergebnis abrufen
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from loguru import logger


# ─────────────────────────────────────────────────────────────────────────────
#  Konfiguration
# ──────────────���──────────────────────────────────────────────────────────────

@dataclass
class AttributionConfig:
    """Konfigurationsparameter fuer PerformanceAttribution."""
    window_size:     int   = 100   # max Trades im Rolling-Window
    top_n:           int   = 5     # Anzahl Top-Features in Reports
    drift_threshold: float = 0.5   # relativer Schwellwert fuer Drift-Erkennung (50%)
    min_records:     int   = 10    # Mindest-Eintraege fuer sinnvolle Drift-Analyse


# ───────���─────────────────────���───────────────────────────���───────────────────
#  Ergebnis-Datenklassen
# ─────────────────────────���──────────────────────────────���────────────────────

@dataclass
class FeatureImportanceSummary:
    """
    Aggregierte Feature-Importance aus dem Rolling-Window.

    feature_names und mean_abs_shap haben gleiche Laenge, absteigend sortiert.
    top_features: die top_n Eintraege als (name, wert)-Paare.
    """
    feature_names: list[str]
    mean_abs_shap: list[float]
    n_records:     int
    top_features:  list[tuple[str, float]]
    computed_at:   datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


@dataclass
class DriftReport:
    """
    Ergebnis der Drift-Analyse: Vergleich Live vs. Trainings-Baseline.

    drift_scores: relativer Aenderungswert pro Feature (0.0 = kein Drift).
    drifted_features: Features die den Schwellwert ueberschreiten.
    retrain_recommended: True wenn Drift signifikant und genuegend Daten vorhanden.
    """
    drift_scores:          dict[str, float]
    drifted_features:      list[str]
    max_drift:             float
    has_significant_drift: bool
    retrain_recommended:   bool
    reason:                str
    computed_at:           datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


# ───────────────────────────────────────��─────────────────────────���───────────
#  Haupt-Klasse
# ─��────────────────────────────��──────────────────────���───────────────────────

class PerformanceAttribution:
    """
    Rollierende SHAP-Aggregation und Feature-Drift-Erkennung.

    Parameters
    ----------
    config   : AttributionConfig (Standard-Werte wenn None).
    _shap_fn : Injizierbare Funktion (model, features_df) -> list[dict[str, float]].
               Wird von run_async_update() aufgerufen. Erwartet fuer jede Zeile
               in features_df ein dict {feature_name: shap_value}.
               Standard: _default_shap_fn (nutzt SHAP TreeExplainer).
    """

    def __init__(
        self,
        config:   Optional[AttributionConfig] = None,
        _shap_fn: Optional[Callable] = None,
    ) -> None:
        self._cfg          = config or AttributionConfig()
        self._window:  deque[dict[str, float]] = deque(maxlen=self._cfg.window_size)
        self._lock         = threading.Lock()
        self._last_summary: Optional[FeatureImportanceSummary] = None
        self._shap_fn      = _shap_fn or _default_shap_fn

    # ── Schreib-Methoden ──────────────────────────────────────────────────────

    def record_shap(self, shap_values: dict[str, float]) -> None:
        """
        Fuegt SHAP-Werte einer einzelnen Vorhersage zum Rolling-Window hinzu.

        Aufgerufen nach jeder Handelsentscheidung (long/short, nicht flat).
        Thread-sicher.
        """
        with self._lock:
            self._window.append(dict(shap_values))

    def record_shap_batch(self, shap_rows: list[dict[str, float]]) -> None:
        """
        Fuegt mehrere SHAP-Dicts auf einmal hinzu (z.B. nach async Batch-Berechnung).
        Thread-sicher.
        """
        with self._lock:
            for row in shap_rows:
                self._window.append(dict(row))

    # ── Abfrage-Methoden ───────────���────────────────────────────��─────────────

    def window_size(self) -> int:
        """Aktuelle Anzahl der Eintraege im Rolling-Window."""
        with self._lock:
            return len(self._window)

    def get_top_features(self, n: Optional[int] = None) -> list[tuple[str, float]]:
        """
        Gibt die Top-N-Features nach mittlerem absolutem SHAP zurueck.

        Returns
        -------
        list von (feature_name, mean_abs_shap), absteigend sortiert.
        Leere Liste wenn kein Eintrag im Window.
        """
        top_n   = n if n is not None else self._cfg.top_n
        summary = self.compute_attribution()
        return summary.top_features[:top_n]

    def compute_attribution(self) -> FeatureImportanceSummary:
        """
        Aggregiert das Rolling-Window zu Feature-Importance-Werten.

        Berechnet den mittleren absoluten SHAP-Wert pro Feature ueber alle
        Eintraege im Fenster. Ergebnis wird als _last_summary gecacht.

        Returns
        -------
        FeatureImportanceSummary – leer wenn Window leer.
        """
        with self._lock:
            records = list(self._window)

        if not records:
            empty = FeatureImportanceSummary(
                feature_names=[],
                mean_abs_shap=[],
                n_records=0,
                top_features=[],
            )
            with self._lock:
                self._last_summary = empty
            return empty

        # Alle Feature-Namen (Union aller Records)
        all_features = sorted({k for row in records for k in row})

        # Mittlerer absoluter SHAP-Wert pro Feature
        mean_abs: dict[str, float] = {}
        for feat in all_features:
            vals = [abs(row[feat]) for row in records if feat in row]
            mean_abs[feat] = float(np.mean(vals)) if vals else 0.0

        # Absteigend sortiert
        sorted_pairs = sorted(mean_abs.items(), key=lambda x: x[1], reverse=True)
        feature_names = [p[0] for p in sorted_pairs]
        mean_abs_shap = [p[1] for p in sorted_pairs]
        top_features  = sorted_pairs[: self._cfg.top_n]

        summary = FeatureImportanceSummary(
            feature_names=feature_names,
            mean_abs_shap=mean_abs_shap,
            n_records=len(records),
            top_features=top_features,
        )
        with self._lock:
            self._last_summary = summary

        logger.debug(
            "PerformanceAttribution: Attribution berechnet | {n} Records | "
            "Top-Feature: {tf}",
            n=len(records),
            tf=top_features[0][0] if top_features else "–",
        )
        return summary

    def get_last_summary(self) -> Optional[FeatureImportanceSummary]:
        """Gibt die zuletzt gecachte Summary zurueck (thread-sicher, kein Recalculate)."""
        with self._lock:
            return self._last_summary

    # ── Drift-Erkennung ─────────────────────────────────────────���─────────────

    def detect_drift(
        self,
        training_importance: dict[str, float],
        threshold: Optional[float] = None,
    ) -> DriftReport:
        """
        Vergleicht aktuelle Feature-Importance mit der Trainings-Baseline.

        Drift-Mass pro Feature:
          relative_change = |live_importance - train_importance| / train_importance
          (bei train_importance == 0: |live_importance|)

        Parameters
        ----------
        training_importance : {feature_name: mean_abs_shap} aus dem Training.
        threshold           : Relativer Schwellwert (Standard: config.drift_threshold).

        Returns
        -------
        DriftReport mit Drift-Scores, drifted_features und retrain_recommended.
        """
        thr     = threshold if threshold is not None else self._cfg.drift_threshold
        summary = self.compute_attribution()
        live_map = dict(zip(summary.feature_names, summary.mean_abs_shap))

        drift_scores: dict[str, float] = {}
        for feat, train_val in training_importance.items():
            live_val  = live_map.get(feat, 0.0)
            if abs(train_val) > 0:
                rel = abs(live_val - train_val) / abs(train_val)
            else:
                rel = abs(live_val)
            drift_scores[feat] = float(rel)

        drifted   = [f for f, s in drift_scores.items() if s > thr]
        max_drift = max(drift_scores.values()) if drift_scores else 0.0
        has_drift = len(drifted) > 0

        enough_data = summary.n_records >= self._cfg.min_records
        retrain     = has_drift and enough_data

        if not enough_data:
            reason = (
                f"Zu wenig Daten ({summary.n_records}/{self._cfg.min_records}) "
                "fuer zuverlaessige Drift-Analyse."
            )
        elif has_drift:
            reason = (
                f"Feature-Drift erkannt: {len(drifted)} Feature(s) ueber "
                f"Schwelle {thr:.0%} ({', '.join(drifted[:3])}). "
                f"Max. Drift: {max_drift:.0%}. Re-Training empfohlen."
            )
        else:
            reason = "Kein signifikanter Feature-Drift erkannt."

        return DriftReport(
            drift_scores=drift_scores,
            drifted_features=drifted,
            max_drift=max_drift,
            has_significant_drift=has_drift,
            retrain_recommended=retrain,
            reason=reason,
        )

    def get_retrain_warning(
        self,
        training_importance: dict[str, float],
        threshold: Optional[float] = None,
    ) -> Optional[str]:
        """
        Gibt eine Warnmeldung zurueck wenn Re-Training empfohlen wird, sonst None.

        Warnung wird ausgegeben wenn:
          - >= min_records Eintraege im Window
          - Mind. ein Feature ueberschreitet den Drift-Schwellwert
        """
        report = self.detect_drift(training_importance, threshold)
        if report.retrain_recommended:
            logger.warning(
                "PerformanceAttribution: {r}", r=report.reason
            )
            return report.reason
        return None

    # ── Asynchrone Berechnung ───────────────────────────────────��─────────────

    def run_async_update(
        self,
        model: Any,
        features_df: pd.DataFrame,
    ) -> threading.Thread:
        """
        Startet asynchrone SHAP-Berechnung in einem Daemon-Thread.

        NICHT im Live-Pfad verwenden. Empfohlen: einmal taeglich aufrufen.

        Der Thread:
          1. Nimmt die letzten window_size Zeilen aus features_df
          2. Ruft _shap_fn(model, subset) auf -> list[dict[str, float]]
          3. Fuegt Ergebnisse per record_shap_batch() ein
          4. Berechnet und cacht compute_attribution()

        Parameters
        ----------
        model       : Trainiertes SignalModel (oder kompatibles Objekt).
        features_df : Feature-DataFrame; letzte window_size Zeilen werden genutzt.

        Returns
        -------
        threading.Thread – bereits gestartet (daemon=True).
        """
        n      = min(len(features_df), self._cfg.window_size)
        subset = features_df.iloc[-n:].copy()

        thread = threading.Thread(
            target=self._async_worker,
            args=(model, subset),
            daemon=True,
            name="PerformanceAttribution-async",
        )
        thread.start()
        logger.info(
            "PerformanceAttribution: Async-Update gestartet | {n} Zeilen | "
            "Thread={t}",
            n=n,
            t=thread.name,
        )
        return thread

    def _async_worker(self, model: Any, features_df: pd.DataFrame) -> None:
        """Interner Worker: SHAP berechnen und ins Window eintraegen."""
        try:
            shap_rows = self._shap_fn(model, features_df)
            self.record_shap_batch(shap_rows)
            self.compute_attribution()
            logger.info(
                "PerformanceAttribution: Async-Update abgeschlossen | "
                "{n} Zeilen verarbeitet",
                n=len(shap_rows),
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "PerformanceAttribution: Async-Update Fehler: {e}", e=exc
            )


# ────��─────────────────────���──────────────────────────────────────────────────
#  Standard-SHAP-Funktion
# ───────��─────────────────────��───────────────────────────────────────────────

def _default_shap_fn(model: Any, features_df: pd.DataFrame) -> list[dict[str, float]]:
    """
    Standard-SHAP-Funktion fuer TreeExplainer-basierte Modelle.

    Berechnet den mittleren absoluten SHAP-Wert ueber Long- und Short-Klasse
    pro Feature und Row.

    Wird von run_async_update() als Standard verwendet.
    """
    from src.models.interpretability import explain_batch  # late import

    batch    = explain_batch(model, features_df)
    long_df  = batch.get("long",  pd.DataFrame())
    short_df = batch.get("short", pd.DataFrame())

    n_rows = len(features_df)
    rows: list[dict[str, float]] = []

    for i in range(n_rows):
        row_shap: dict[str, float] = {}

        if not long_df.empty and i < len(long_df):
            for col in long_df.columns:
                row_shap[col] = abs(float(long_df.iloc[i][col]))

        if not short_df.empty and i < len(short_df):
            for col in short_df.columns:
                short_val = abs(float(short_df.iloc[i][col]))
                if col in row_shap:
                    row_shap[col] = (row_shap[col] + short_val) / 2.0
                else:
                    row_shap[col] = short_val

        rows.append(row_shap)

    return rows


# ───────��──────────────────────────���───────────────────────────���──────────────
#  Hilfsfunktion: Trainings-Importance extrahieren
# ─────────────────��──────────────────────────��────────────────────────────────

def extract_training_importance(
    model: Any,
    features_df: pd.DataFrame,
    shap_fn: Optional[Callable] = None,
) -> dict[str, float]:
    """
    Extrahiert die Feature-Importance aus Trainings-Daten via SHAP.

    Ergebnis kann als ``training_importance``-Baseline fuer detect_drift()
    verwendet werden.

    Parameters
    ----------
    model       : Trainiertes SignalModel.
    features_df : Trainings-Feature-DataFrame.
    shap_fn     : Optional – alternative SHAP-Funktion (Standard: _default_shap_fn).

    Returns
    -------
    dict {feature_name: mean_abs_shap} – absteigend sortiert.
    """
    fn        = shap_fn or _default_shap_fn
    shap_rows = fn(model, features_df)

    attr = PerformanceAttribution()
    attr.record_shap_batch(shap_rows)
    summary = attr.compute_attribution()

    return dict(zip(summary.feature_names, summary.mean_abs_shap))
