"""
src/models/retraining_scheduler.py
RetrainingScheduler – automatisches, versioniertes Re-Training des SignalModells.

Ablauf je run()-Aufruf:
  1. IS/OOS-Split der uebergebenen Features/Labels (konfigurierbares Verhaeltnis, Standard 70/30)
  2. Neues SignalModel auf IS-Daten trainieren
  3. Versionierte Speicherung (signal_model_v{n}_{datum}.joblib)
  4. OOS-Metriken (Sharpe, Win-Rate) fuer neues UND aktives Modell berechnen
  5. Vergleich: Neues Modell deployen wenn Sharpe-Delta >= min_sharpe_delta
  6. model_registry.json aktualisieren (active + fallback)
  7. Telegram-Alert mit Vergleichszahlen senden
  8. Audit-Log-Eintrag schreiben

Konfiguration:
  RetrainingConfig – alle Parameter (Intervall, Schwellwert, Pfade usw.)

Testbarkeit:
  alert_fn, audit_fn und _now_fn sind injizierbar.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.models.signal_model import SignalModel, build_save_path, _LABEL_TO_CLASS


# ─────────────────────────────────────────────────────────────────────────────
#  Exclude-Set fuer Feature-Selektion (konsistent mit train_model.py)
# ─────────────────────────────────────────────────────────────────────────────

_STRUCT_COLS: frozenset[str] = frozenset(
    {"label", "timestamp", "open", "volume", "close", "high", "low"}
)

_REGISTRY_FILENAME = "model_registry.json"


# ─────────────────────────────────────────────────────────────────────────────
#  Datenklassen
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RetrainingConfig:
    """
    Konfiguration fuer den RetrainingScheduler.

    Parameters
    ----------
    models_dir         : Verzeichnis fuer Modell-Dateien.
    interval_days      : Mindestabstand zwischen Re-Training-Laeufen (Standard: 7 Tage).
    preferred_hour_utc : Stunde (UTC) fuer is_due-Pruefung – ausserhalb Handelszeiten
                         (Standard: 2 Uhr UTC = ausserhalb EUR/USD-Haupthandel).
    min_sharpe_delta   : Minimaler OOS-Sharpe-Unterschied (new - old) fuer Deployment.
                         Negativer Wert erlaubt leichte Verschlechterung (Standard: -0.1).
    is_ratio           : Anteil der Daten fuer In-Sample-Training (Standard: 0.70).
    confidence_threshold: Konfidenz-Schwellwert fuer get_signal (Standard: 0.55).
    """
    models_dir:            Path  = field(default_factory=lambda: Path("models"))
    interval_days:         int   = 7
    preferred_hour_utc:    int   = 2
    min_sharpe_delta:      float = -0.1
    is_ratio:              float = 0.70
    confidence_threshold:  float = 0.55


@dataclass
class ModelMetrics:
    """OOS-Performance-Kennzahlen eines Modells."""
    sharpe:     float
    win_rate:   float
    n_oos_rows: int


@dataclass
class RetrainingResult:
    """Ergebnis eines Re-Training-Laufs."""
    timestamp:       datetime
    symbol:          str
    timeframe:       str
    new_model_path:  Path
    old_model_path:  Optional[Path]
    new_metrics:     ModelMetrics
    old_metrics:     Optional[ModelMetrics]
    promoted:        bool
    reason:          str


# ─────────────────────────────────────────────────────────────────────────────
#  RetrainingScheduler
# ─────────────────────────────────────────────────────────────────────────────

class RetrainingScheduler:
    """
    Orchestriert das automatische, versionierte Re-Training des SignalModells.

    Parameters
    ----------
    config    : RetrainingConfig mit allen Parametern.
    alert_fn  : Optionaler Callable(message: str) fuer Telegram-/Slack-Alerts.
    audit_fn  : Optionaler Callable(event_type: str, details: dict) fuer Audit-Log.
    _now_fn   : Ersetzt datetime.now() – injizierbar fuer deterministische Tests.
    """

    def __init__(
        self,
        config:    RetrainingConfig,
        alert_fn:  Optional[Callable[[str], None]] = None,
        audit_fn:  Optional[Callable[[str, dict], None]] = None,
        _now_fn:   Optional[Callable[[], datetime]] = None,
    ) -> None:
        self._cfg      = config
        self._alert_fn = alert_fn
        self._audit_fn = audit_fn
        self._now_fn   = _now_fn or (lambda: datetime.now(timezone.utc))

    # ── Oeffentliche Schnittstelle ────────────────────────────────────────────

    def is_due(self, last_run: Optional[datetime] = None) -> bool:
        """
        True wenn Re-Training faellig ist.

        Faellig wenn:
        - Noch nie gelaufen (last_run=None), ODER
        - Mindestintervall seit letztem Run abgelaufen UND
          aktuelle Stunde == preferred_hour_utc.
        """
        now = self._now_fn()
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        if last_run is None:
            return True

        if last_run.tzinfo is None:
            last_run = last_run.replace(tzinfo=timezone.utc)

        elapsed_days = (now - last_run).total_seconds() / 86400.0
        if elapsed_days < self._cfg.interval_days:
            return False

        return now.hour == self._cfg.preferred_hour_utc

    def run(
        self,
        symbol:      str,
        timeframe:   str,
        features_df: pd.DataFrame,
        labels:      pd.Series,
        version:     int = 1,
    ) -> RetrainingResult:
        """
        Fuehrt einen vollstaendigen Re-Training-Lauf durch.

        Parameters
        ----------
        symbol      : Handelssymbol (z.B. 'EURUSD').
        timeframe   : Zeitrahmen (z.B. 'H1').
        features_df : Vollstaendiger Features-DataFrame (inkl. close/high/low).
        labels      : Triple-Barrier-Labels, gleicher Index wie features_df.
        version     : Modellversionsnummer fuer Dateinamen.

        Returns
        -------
        RetrainingResult mit Metriken, Pfaden und Deployment-Entscheidung.
        """
        now = self._now_fn()
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        logger.info(
            "RetrainingScheduler: Start | {sym}/{tf} | {n} Zeilen",
            sym=symbol, tf=timeframe, n=len(features_df),
        )

        feat_cols = self._get_feature_cols(features_df)
        split     = int(len(features_df) * self._cfg.is_ratio)

        X_is  = features_df[feat_cols].iloc[:split].copy()
        y_is  = labels.iloc[:split]
        X_oos = features_df[feat_cols].iloc[split:].values.astype(float)
        y_oos = labels.iloc[split:].values

        # Neues Modell trainieren und speichern
        new_model = SignalModel(
            lgbm_params={"verbose": -1, "random_state": 42}
        )
        new_model.train(X_is, y_is)

        self._cfg.models_dir.mkdir(parents=True, exist_ok=True)
        new_path = build_save_path(version, now.date())
        # Pfad relativ zu CWD konstruieren wie build_save_path es macht,
        # aber models_dir-Override respektieren
        new_path = self._cfg.models_dir / new_path.name
        new_model.save(new_path)

        # Neues Modell evaluieren
        new_metrics = self._evaluate(new_model, X_oos, y_oos)

        # Aktuell aktives Modell laden und evaluieren
        old_path    = self._find_active_model(exclude=new_path)
        old_metrics: Optional[ModelMetrics] = None
        if old_path is not None:
            try:
                old_model   = SignalModel.load(old_path)
                old_metrics = self._evaluate(old_model, X_oos, y_oos)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Altes Modell konnte nicht evaluiert werden: {e}", e=exc
                )

        # Deployment-Entscheidung
        promoted, reason = self._should_promote(new_metrics, old_metrics)
        if promoted:
            self._write_registry(new_path, old_path, now)

        result = RetrainingResult(
            timestamp=now,
            symbol=symbol,
            timeframe=timeframe,
            new_model_path=new_path,
            old_model_path=old_path,
            new_metrics=new_metrics,
            old_metrics=old_metrics,
            promoted=promoted,
            reason=reason,
        )

        self._send_alert(result)
        self._write_audit(result)

        logger.info(
            "RetrainingScheduler: {status} | {reason}",
            status="DEPLOYED" if promoted else "REJECTED",
            reason=reason,
        )
        return result

    def get_active_model_path(self) -> Optional[Path]:
        """Gibt den Pfad des aktuell aktiven Modells zurueck (aus Registry)."""
        registry = self._load_registry()
        active   = registry.get("active")
        if active:
            p = self._cfg.models_dir / active
            if p.exists():
                return p
        return self._find_active_model()

    # ── Interna ───────────────────────────────────────────────────────────────

    @staticmethod
    def _get_feature_cols(df: pd.DataFrame) -> list[str]:
        return [c for c in df.columns if c not in _STRUCT_COLS]

    def _evaluate(
        self,
        model:  SignalModel,
        X_oos:  np.ndarray,
        y_oos:  np.ndarray,
    ) -> ModelMetrics:
        """
        Berechnet vereinfachten OOS-Sharpe und Win-Rate.

        Sharpe: mean(returns) / std(returns) * sqrt(252), wobei
        return[i] = +1 bei korrektem Direktionssignal, -1 bei falschem.
        Neutrale Vorhersagen werden uebersprungen.

        Win-Rate: Anteil korrekter direktionaler Vorhersagen (ohne Neutral).
        """
        if model._model is None:
            return ModelMetrics(sharpe=0.0, win_rate=0.0, n_oos_rows=int(len(y_oos)))

        proba   = model._model.predict_proba(X_oos)
        signals = np.argmax(proba, axis=1)   # 0=short, 1=neutral, 2=long
        y_cls   = np.array([_LABEL_TO_CLASS[int(v)] for v in y_oos])

        returns:    list[float] = []
        correct:    int = 0
        directional: int = 0

        for sig, true_cls in zip(signals, y_cls):
            if sig == 1:   # neutral – kein Trade
                continue
            directional += 1
            hit = sig == true_cls
            returns.append(1.0 if hit else -1.0)
            if hit:
                correct += 1

        win_rate = correct / directional if directional > 0 else 0.0

        if len(returns) < 2:
            sharpe = 0.0
        else:
            arr = np.array(returns, dtype=float)
            std = float(arr.std())
            sharpe = float(arr.mean() / std * np.sqrt(252)) if std > 0 else 0.0

        return ModelMetrics(
            sharpe=sharpe,
            win_rate=win_rate,
            n_oos_rows=int(len(y_oos)),
        )

    def _find_active_model(
        self, exclude: Optional[Path] = None
    ) -> Optional[Path]:
        """Aktuell aktives Modell aus Registry; Fallback: neueste Datei."""
        registry = self._load_registry()
        active   = registry.get("active")
        if active:
            p = self._cfg.models_dir / active
            if p.exists() and p != exclude:
                return p

        files = sorted(
            (
                f for f in self._cfg.models_dir.glob("signal_model_v*.joblib")
                if f != exclude
            ),
            reverse=True,
        )
        return files[0] if files else None

    def _should_promote(
        self,
        new_m: ModelMetrics,
        old_m: Optional[ModelMetrics],
    ) -> tuple[bool, str]:
        if old_m is None:
            return True, "Kein aktives Modell vorhanden – direkt deployed."

        delta = new_m.sharpe - old_m.sharpe
        thr   = self._cfg.min_sharpe_delta

        if delta >= thr:
            return (
                True,
                f"Deployed (OOS-Sharpe-Delta: {delta:+.3f} >= Schwellwert {thr:+.3f})",
            )
        return (
            False,
            f"Abgelehnt (OOS-Sharpe-Delta: {delta:+.3f} < Schwellwert {thr:+.3f})",
        )

    def _write_registry(
        self,
        new_path: Path,
        old_path: Optional[Path],
        now:      datetime,
    ) -> None:
        registry: dict[str, Any] = {
            "active":           new_path.name,
            "fallback":         old_path.name if old_path else None,
            "last_retrain_utc": now.isoformat(),
        }
        dest = self._cfg.models_dir / _REGISTRY_FILENAME
        dest.write_text(json.dumps(registry, indent=2), encoding="utf-8")
        logger.info("Registry aktualisiert -> {p}", p=dest)

    def _load_registry(self) -> dict[str, Any]:
        path = self._cfg.models_dir / _REGISTRY_FILENAME
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Registry konnte nicht gelesen werden: {e}", e=exc)
            return {}

    def _send_alert(self, result: RetrainingResult) -> None:
        if not self._alert_fn:
            return

        status_icon = "✅" if result.promoted else "⚠️"
        new_m  = result.new_metrics
        old_m  = result.old_metrics

        old_sharpe_str   = f"{old_m.sharpe:.3f}"   if old_m else "–"
        old_winrate_str  = f"{old_m.win_rate:.1%}"  if old_m else "–"

        message = (
            f"{status_icon} *Re-Training {result.symbol}/{result.timeframe}*\n"
            f"Zeit: `{result.timestamp.strftime('%Y-%m-%d %H:%M UTC')}`\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"*Neues Modell*\n"
            f"  OOS-Sharpe:  `{new_m.sharpe:.3f}`\n"
            f"  Win-Rate:    `{new_m.win_rate:.1%}`\n"
            f"  OOS-Zeilen:  `{new_m.n_oos_rows}`\n"
            f"*Aktives Modell*\n"
            f"  OOS-Sharpe:  `{old_sharpe_str}`\n"
            f"  Win-Rate:    `{old_winrate_str}`\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"Entscheidung: `{result.reason}`"
        )
        try:
            self._alert_fn(message)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Alert konnte nicht gesendet werden: {e}", e=exc)

    def _write_audit(self, result: RetrainingResult) -> None:
        if not self._audit_fn:
            return

        event_type = (
            "MODEL_RETRAIN_DEPLOYED" if result.promoted
            else "MODEL_RETRAIN_REJECTED"
        )
        details: dict[str, Any] = {
            "symbol":           result.symbol,
            "timeframe":        result.timeframe,
            "new_model":        str(result.new_model_path),
            "old_model":        str(result.old_model_path) if result.old_model_path else None,
            "new_sharpe":       result.new_metrics.sharpe,
            "new_win_rate":     result.new_metrics.win_rate,
            "old_sharpe":       result.old_metrics.sharpe   if result.old_metrics else None,
            "old_win_rate":     result.old_metrics.win_rate if result.old_metrics else None,
            "promoted":         result.promoted,
            "reason":           result.reason,
            "timestamp":        result.timestamp.isoformat(),
        }
        try:
            self._audit_fn(event_type, details)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Audit-Log-Eintrag fehlgeschlagen: {e}", e=exc)
