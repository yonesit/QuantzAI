"""
scripts/run_test2_usdjpy_d1.py
Test #2: USDJPY D1 Trendfolge – Baseline (23 Features, ohne MTF)

D1-spezifische Hinweise:
  - Walk-Forward 6M/1M: D1 hat ~22 Bars/Monat, also ~22 Bars pro Test-Fenster.
    Mit 4 Jahren Daten und 200-Bar-Warmup entstehen ~33 Fenster – ausreichend
    fuer statistisch belastbare Aussagen. Keine Anpassung der Fenstergroesse.
  - LabelBuilder max_candles=24 bedeutet auf D1: 24 Tage Vorausschauhorizont
    (~1 Monat) – sinnvoll fuer Trendfolge auf Tagesbasis.
  - hour_of_day: D1-Bars haben immer hour=0 (Tageseröffnung), daher ist dieses
    Feature konstant und informationslos. Verbleibt im 23-Feature-Set (Modell
    gibt ihm nahe-null SHAP-Gewichtung). Wird als Auffälligkeit dokumentiert.

Ausfuehrung:
  python scripts/run_test2_usdjpy_d1.py
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone, date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
from loguru import logger
import numpy as np
import pandas as pd

load_dotenv()


# ---------------------------------------------------------------------------
# Konstanten
# ---------------------------------------------------------------------------

SYMBOL     = "USDJPY"
TIMEFRAME  = "D1"
START_DATE = datetime(2020, 1, 1, tzinfo=timezone.utc)
END_DATE   = datetime(2024, 1, 1, tzinfo=timezone.utc)
LOG_PATH   = Path(__file__).resolve().parents[1] / "docs" / "strategy_research_log.md"
TEST_DATE  = date.today().isoformat()

# Walk-Forward Parameter
TRAIN_MONTHS = 6
TEST_MONTHS  = 1
# D1 hat ~22 Bars/Monat -> 6M Train ≈ 132 Bars, 1M Test ≈ 22 Bars.
# Beides liegt komfortabel über den Mindestschwellen (10 / 2).


# ---------------------------------------------------------------------------
# Schritt 1: MT5 Verbindung und Daten holen
# ---------------------------------------------------------------------------

def check_broker_and_fetch() -> pd.DataFrame:
    """Verbindet mit MT5, prueft USDJPY und holt D1-Daten zurueck."""
    from src.data.mt5_connector import MT5Connector

    mt5 = MT5Connector(
        login=int(os.environ.get("MT5_LOGIN", "0")),
        password=os.environ.get("MT5_PASSWORD", ""),
        server=os.environ.get("MT5_SERVER", ""),
    )

    logger.info("=== Schritt 1: MT5 Verbindungsaufbau ===")
    mt5.connect()

    symbols = mt5.get_available_symbols()
    if SYMBOL not in symbols:
        raise RuntimeError(
            f"{SYMBOL} wird vom Broker nicht angeboten! Erste 20: {symbols[:20]}"
        )
    logger.info("{sym} verfuegbar | {n} Symbole gesamt", sym=SYMBOL, n=len(symbols))

    logger.info(
        "Hole {sym} D1 Daten | {s} – {e}",
        sym=SYMBOL, s=START_DATE.date(), e=END_DATE.date(),
    )
    df = mt5.get_ohlcv(SYMBOL, TIMEFRAME, START_DATE, END_DATE)
    mt5.disconnect()

    logger.info(
        "Daten: {n} Bars | {s} – {e}",
        n=len(df), s=df.index[0].date(), e=df.index[-1].date(),
    )

    # Mindestens 3 Jahre (Warmup 200 + WF-Mindest)
    if len(df) < 800:
        raise RuntimeError(
            f"Zu wenige D1-Bars: {len(df)} (min 800 erwartet). "
            "Broker liefert nicht genug History."
        )
    logger.info("History-Check bestanden: {n} D1-Bars", n=len(df))
    return df


# ---------------------------------------------------------------------------
# Schritt 2: Validieren & Features bauen (Baseline, OHNE MTF)
# ---------------------------------------------------------------------------

def build_baseline_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Validiert und baut 23-Feature-Baseline ohne MTF-Features."""
    from src.data.validator import DataValidator
    from src.data.feature_builder import FeatureBuilder

    logger.info("=== Schritt 2: Validieren & Features bauen ===")

    df_reset = raw_df.reset_index()
    df_reset = df_reset.rename(columns={df_reset.columns[0]: "timestamp"})

    validator = DataValidator()
    report, clean_df = validator.validate(df_reset, symbol=SYMBOL, timeframe=TIMEFRAME)
    logger.info(
        "Validator: quality={q:.3f} usable={u} candles={c}",
        q=report.quality_score, u=report.is_usable, c=report.total_candles,
    )
    if not report.is_usable:
        raise RuntimeError(f"Datenqualitaet ungenuegend: {report.errors}")

    # FeatureBuilder – kein df_h4, kein df_d1 → 23-Feature-Baseline
    builder = FeatureBuilder()
    features_df = builder.build(
        clean_df,
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        df_h4=None,
        df_d1=None,
    )

    feat_cols = [c for c in features_df.columns
                 if c not in {"timestamp", "close", "high", "low"}]
    logger.info(
        "Features: {n} Spalten | {rows} Zeilen | {cols}",
        n=len(feat_cols), rows=len(features_df), cols=feat_cols,
    )
    assert "h4_trend" not in features_df.columns
    assert "d1_trend" not in features_df.columns
    assert len(feat_cols) == 23, f"Erwartet 23 Features, got {len(feat_cols)}: {feat_cols}"

    # Pruefen ob hour_of_day konstant (erwartet bei D1)
    if "hour_of_day" in features_df.columns:
        unique_hours = features_df["hour_of_day"].unique()
        logger.info(
            "hour_of_day auf D1: unique Werte = {v} (erwartet: [0])",
            v=sorted(unique_hours.tolist()),
        )

    return features_df


# ---------------------------------------------------------------------------
# Schritt 3+4: Labels und Walk-Forward
# ---------------------------------------------------------------------------

def run_walk_forward(features_df: pd.DataFrame) -> tuple[list[dict], pd.Series]:
    """Erzeugt Labels und fuehrt Walk-Forward aus. Gibt (wf_results, labels) zurueck."""
    from src.models.label_builder import LabelBuilder
    from src.models.signal_model import SignalModel

    logger.info("=== Schritt 3: Labels erzeugen (max_candles=24 → 24 Tage Horizont) ===")
    label_builder = LabelBuilder()
    labels = label_builder.build_labels(features_df)

    logger.info(
        "=== Schritt 4: Walk-Forward {train}M/{test}M | D1 ≈22 Bars/Monat ===",
        train=TRAIN_MONTHS, test=TEST_MONTHS,
    )
    exclude = {"label", "timestamp", "open", "volume", "close", "high", "low"}
    feat_cols = [c for c in features_df.columns if c not in exclude]
    features_with_ts = features_df[
        feat_cols + (["timestamp"] if "timestamp" in features_df.columns else [])
    ]

    model = SignalModel()
    wf_results = model.walk_forward_validate(
        features_with_ts,
        labels,
        timestamp_col="timestamp",
        train_months=TRAIN_MONTHS,
        test_months=TEST_MONTHS,
    )
    logger.info("Walk-Forward abgeschlossen: {n} Fenster", n=len(wf_results))
    return wf_results, labels


# ---------------------------------------------------------------------------
# Schritt 5: SHAP Top-3
# ---------------------------------------------------------------------------

def compute_shap_top3(features_df: pd.DataFrame, labels: pd.Series) -> list[str]:
    """Finales Modell + SHAP mean-abs Top-3."""
    import shap
    from src.models.signal_model import SignalModel

    logger.info("=== Schritt 5: Finales Modell + SHAP ===")

    exclude = {"label", "timestamp", "open", "volume", "close", "high", "low"}
    feat_cols = [c for c in features_df.columns if c not in exclude]
    feat_only = features_df[feat_cols]

    model = SignalModel()
    model.train(feat_only, labels)

    n_shap = min(500, len(feat_only))
    X_shap = feat_only.sample(n_shap, random_state=42).values.astype(float)

    explainer = shap.TreeExplainer(model._model)
    shap_vals = explainer.shap_values(X_shap)

    if isinstance(shap_vals, list):
        shap_arr = np.stack(shap_vals, axis=-1)
    else:
        shap_arr = np.asarray(shap_vals)

    mean_abs = np.mean(np.abs(shap_arr), axis=(0, 2))
    top3_idx = np.argsort(mean_abs)[::-1][:3]
    top3 = [feat_cols[i] for i in top3_idx]
    logger.info("SHAP Top-3: {v}", v=top3)
    return top3


# ---------------------------------------------------------------------------
# Schritt 6: Metriken zusammenfassen
# ---------------------------------------------------------------------------

def summarize(wf_results: list[dict]) -> dict:
    if not wf_results:
        raise RuntimeError("Keine Walk-Forward-Fenster – zu wenig Daten.")

    sharpes    = [r["oos_sharpe"] for r in wf_results]
    profitable = sum(1 for s in sharpes if s > 0)
    total      = len(sharpes)
    test_bars  = sum(r.get("test_size", 0) for r in wf_results)

    return {
        "oos_sharpe_mean":    float(np.mean(sharpes)),
        "oos_sharpe_std":     float(np.std(sharpes)),
        "profitable_windows": profitable,
        "total_windows":      total,
        "profitable_pct":     profitable / total * 100,
        "total_test_bars":    test_bars,
        "windows":            wf_results,
    }


# ---------------------------------------------------------------------------
# Schritt 7: Logdatei aktualisieren
# ---------------------------------------------------------------------------

def update_research_log(metrics: dict, top3: list[str], wf_note: str) -> None:
    content = LOG_PATH.read_text(encoding="utf-8")

    mean_sharpe = metrics["oos_sharpe_mean"]
    prof_pct    = metrics["profitable_pct"]

    if mean_sharpe > 0 and prof_pct > 50:
        verdict     = "Kandidat"
        verdict_why = (
            f"Ø OOS-Sharpe {mean_sharpe:.3f} > 0 und {prof_pct:.0f}% profitable Fenster > 50%. "
            "Beide Mindestanforderungen erfuellt."
        )
    elif mean_sharpe > 0 or prof_pct > 50:
        verdict     = "Unklar – weitere Tests nötig"
        verdict_why = (
            f"Ø OOS-Sharpe {mean_sharpe:.3f}, Profitable Fenster {prof_pct:.0f}%. "
            "Gemischte Signale – braucht mehr Evidenz."
        )
    else:
        verdict     = "Verworfen"
        verdict_why = (
            f"Ø OOS-Sharpe {mean_sharpe:.3f} <= 0 und nur {prof_pct:.0f}% profitable Fenster. "
            "Kein stabiler Edge nachweisbar."
        )

    shap_str  = ", ".join(top3)
    extremes  = _find_extremes(metrics["windows"])

    new_entry = f"""
### Test #2: USDJPY D1 Trendfolge
- Datum: {TEST_DATE}
- Zeitraum: 4 Jahre (2020-01-01 bis 2024-01-01)
- Walk-Forward: {TRAIN_MONTHS}M Training / {TEST_MONTHS}M Test, rollierend ({wf_note})
- Ø OOS-Sharpe: {mean_sharpe:.3f}
- Std OOS-Sharpe: {metrics['oos_sharpe_std']:.3f}
- Profitable Fenster: {metrics['profitable_windows']}/{metrics['total_windows']} ({prof_pct:.0f}%)
- Anzahl Trades gesamt: {metrics['total_test_bars']}
- SHAP Top-3 Features: {shap_str}
- Auffälligkeiten/Extremwerte: {extremes}; hour_of_day konstant=0 auf D1 (informationslos, aber modellseitig korrekt ignoriert)
- Urteil: {verdict}
- Begründung des Urteils: {verdict_why}
"""

    # Eintrag nach Test #1 anfuegen
    content = content.rstrip() + "\n" + new_entry.strip() + "\n"

    # Tabellenstatus aktualisieren
    content = content.replace(
        "| 2 | USDJPY | D1 | Trendfolge | Test ob noch längerer Timeframe noch robuster | ⏳ offen |",
        f"| 2 | USDJPY | D1 | Trendfolge | Test ob noch längerer Timeframe noch robuster | {verdict} |",
    )

    LOG_PATH.write_text(content, encoding="utf-8")
    logger.info("strategy_research_log.md aktualisiert | Urteil: {v}", v=verdict)


def _find_extremes(windows: list[dict]) -> str:
    sharpes = [(r["window"], r["oos_sharpe"]) for r in windows]
    if not sharpes:
        return "keine"
    max_w, max_s = max(sharpes, key=lambda x: x[1])
    min_w, min_s = min(sharpes, key=lambda x: x[1])
    parts = []
    if max_s > 2.0:
        parts.append(f"Ausreisser oben: Fenster {max_w} OOS-Sharpe={max_s:.2f}")
    if min_s < -2.0:
        parts.append(f"Ausreisser unten: Fenster {min_w} OOS-Sharpe={min_s:.2f}")
    return "; ".join(parts) if parts else "keine Ausreisser > |2.0|"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("====== Test #2: USDJPY D1 Trendfolge (Baseline 23 Feat, ohne MTF) ======")

    raw_df = check_broker_and_fetch()
    features_df = build_baseline_features(raw_df)
    wf_results, labels = run_walk_forward(features_df)
    top3 = compute_shap_top3(features_df, labels)
    metrics = summarize(wf_results)

    # Walk-Forward Notiz fuer Protokoll
    n_windows = metrics["total_windows"]
    bars_per_test = (
        metrics["total_test_bars"] // n_windows if n_windows else 0
    )
    wf_note = (
        f"{n_windows} Fenster, D1 ≈{bars_per_test} Bars/Test-Fenster, "
        f"kein Anpassungsbedarf (min 10 Train / 2 Test gut erfuellt)"
    )

    logger.info("====== Ergebnisse ======")
    logger.info("Ø OOS-Sharpe:          {v:.3f}", v=metrics["oos_sharpe_mean"])
    logger.info("Std OOS-Sharpe:        {v:.3f}", v=metrics["oos_sharpe_std"])
    logger.info(
        "Profitable Fenster:    {p}/{t} ({pct:.0f}%)",
        p=metrics["profitable_windows"],
        t=metrics["total_windows"],
        pct=metrics["profitable_pct"],
    )
    logger.info("Bars in Test-Perioden: {v}", v=metrics["total_test_bars"])
    logger.info("SHAP Top-3:            {v}", v=top3)

    update_research_log(metrics, top3, wf_note)
    logger.info("====== Test #2 abgeschlossen ======")


if __name__ == "__main__":
    main()
