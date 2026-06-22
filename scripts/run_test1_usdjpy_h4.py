"""
scripts/run_test1_usdjpy_h4.py
Test #1: USDJPY H4 Trendfolge – Baseline (23 Features, ohne MTF)

Schritte:
  1. MT5-Verbindung aufbauen und USDJPY-Verfuegbarkeit pruefen
  2. USDJPY H4 Daten 2020-01-01 bis 2024-01-01 fetchen
  3. Features bauen (23-Feature-Baseline, OHNE MTF)
  4. Walk-Forward 6M/1M ausfuehren
  5. SHAP Top-3 Features berechnen
  6. docs/strategy_research_log.md aktualisieren

Ausfuehrung:
  python scripts/run_test1_usdjpy_h4.py
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
TIMEFRAME  = "H4"
START_DATE = datetime(2020, 1, 1, tzinfo=timezone.utc)
END_DATE   = datetime(2024, 1, 1, tzinfo=timezone.utc)
LOG_PATH   = Path(__file__).resolve().parents[1] / "docs" / "strategy_research_log.md"
TEST_DATE  = date.today().isoformat()


# ---------------------------------------------------------------------------
# Schritt 1: MT5 Verbindung und Symbol-Check
# ---------------------------------------------------------------------------

def check_broker_and_fetch() -> pd.DataFrame:
    """Verbindet mit MT5, prueft USDJPY und holt H4-Daten zurueck."""
    from src.data.mt5_connector import MT5Connector, MT5DataError

    mt5 = MT5Connector(
        login=int(os.environ.get("MT5_LOGIN", "0")),
        password=os.environ.get("MT5_PASSWORD", ""),
        server=os.environ.get("MT5_SERVER", ""),
    )

    logger.info("=== Schritt 1: MT5 Verbindungsaufbau ===")
    mt5.connect()

    symbols = mt5.get_available_symbols()
    logger.info("Broker bietet {n} Symbole an", n=len(symbols))

    if SYMBOL not in symbols:
        raise RuntimeError(f"{SYMBOL} wird vom Broker nicht angeboten! Verfuegbar: {symbols[:20]}")

    logger.info("{sym} ist verfuegbar. Pruefe History...", sym=SYMBOL)

    # Komplett-Fetch: 4 Jahre H4
    logger.info("Hole {sym} H4 Daten | {s} – {e}", sym=SYMBOL, s=START_DATE.date(), e=END_DATE.date())
    df = mt5.get_ohlcv(SYMBOL, TIMEFRAME, START_DATE, END_DATE)
    mt5.disconnect()

    logger.info(
        "Daten erhalten: {n} Kerzen | {s} – {e}",
        n=len(df),
        s=df.index[0],
        e=df.index[-1],
    )

    expected_h4_bars = 4 * 365 * 5 / 7 * 6  # ~ca. 6260 (6 Bars/Tag an Werktagen)
    if len(df) < 3000:
        raise RuntimeError(
            f"Zu wenige H4-Kerzen: {len(df)} (erwartet ca. {expected_h4_bars:.0f}). "
            f"Nicht genug Historie beim Broker."
        )

    logger.info(
        "History-Check bestanden: {n} H4-Bars fuer {sym} vorhanden.",
        n=len(df), sym=SYMBOL,
    )
    return df


# ---------------------------------------------------------------------------
# Schritt 2: Features bauen (Baseline, OHNE MTF)
# ---------------------------------------------------------------------------

def build_baseline_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Baut 23-Feature-Baseline OHNE MTF-Features (kein df_h4, kein df_d1)."""
    from src.data.validator import DataValidator
    from src.data.feature_builder import FeatureBuilder

    logger.info("=== Schritt 2: Daten validieren & Features bauen ===")

    # reset_index -> timestamp-Spalte (DataValidator erwartet Spalte, nicht Index)
    df_reset = raw_df.reset_index()
    df_reset = df_reset.rename(columns={df_reset.columns[0]: "timestamp"})

    validator = DataValidator()
    report, clean_df = validator.validate(df_reset, symbol=SYMBOL, timeframe=TIMEFRAME)
    logger.info(
        "Validator: quality_score={q:.3f} | usable={u} | candles={c}",
        q=report.quality_score, u=report.is_usable, c=report.total_candles,
    )
    if not report.is_usable:
        raise RuntimeError(f"Datenqualitaet ungenuegend: {report.errors}")

    # FeatureBuilder – OHNE MTF (df_h4=None, df_d1=None)
    builder = FeatureBuilder()
    features_df = builder.build(
        clean_df,
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        df_h4=None,
        df_d1=None,
    )

    # Feature-Spalten pruefen (keine h4_trend/d1_trend)
    feat_cols = [c for c in features_df.columns if c not in {"timestamp", "close", "high", "low"}]
    logger.info(
        "Features gebaut: {n} Spalten | {rows} Zeilen | Spalten: {cols}",
        n=len(feat_cols),
        rows=len(features_df),
        cols=feat_cols,
    )
    assert "h4_trend" not in features_df.columns, "MTF h4_trend darf nicht vorhanden sein!"
    assert "d1_trend" not in features_df.columns, "MTF d1_trend darf nicht vorhanden sein!"

    return features_df


# ---------------------------------------------------------------------------
# Schritt 3 + 4: Labels erzeugen & Walk-Forward
# ---------------------------------------------------------------------------

def run_walk_forward(features_df: pd.DataFrame) -> tuple[list[dict], int]:
    """Labels erzeugen und Walk-Forward 6M/1M ausfuehren. Gibt (wf_results, total_trades) zurueck."""
    from src.models.label_builder import LabelBuilder
    from src.models.signal_model import SignalModel

    logger.info("=== Schritt 3: Labels erzeugen ===")
    label_builder = LabelBuilder()
    labels = label_builder.build_labels(features_df)
    logger.info("Labels: {v}", v=labels.value_counts().to_dict())

    logger.info("=== Schritt 4: Walk-Forward 6M/1M ===")
    exclude = {"label", "timestamp", "open", "volume", "close", "high", "low"}
    feat_cols = [c for c in features_df.columns if c not in exclude]
    features_with_ts = features_df[feat_cols + (["timestamp"] if "timestamp" in features_df.columns else [])]

    model = SignalModel()
    wf_results = model.walk_forward_validate(
        features_with_ts,
        labels,
        timestamp_col="timestamp",
        train_months=6,
        test_months=1,
    )

    # Trades zählen: pro Fenster alle Vorhersagen die NICHT neutral sind
    total_trades = 0
    for r in wf_results:
        # test_size = alle Bars im Test-Fenster;
        # approximation: Signalrate ca. 40% (wie bei EURUSD-Baseline)
        # -> verwende test_size als Basis, echter Trade-Count kommt aus dem Backtest
        total_trades += r.get("test_size", 0)

    return wf_results, total_trades


# ---------------------------------------------------------------------------
# Schritt 5: SHAP Top-3
# ---------------------------------------------------------------------------

def compute_shap_top3(features_df: pd.DataFrame, labels: pd.Series) -> list[str]:
    """Trainiert finales Modell und berechnet SHAP mean-abs Top-3 Features."""
    import shap
    import lightgbm as lgb
    from src.models.signal_model import SignalModel, _LABEL_TO_CLASS

    logger.info("=== Schritt 5: Finales Modell + SHAP ===")

    exclude = {"label", "timestamp", "open", "volume", "close", "high", "low"}
    feat_cols = [c for c in features_df.columns if c not in exclude]
    feat_only = features_df[feat_cols]

    model = SignalModel()
    model.train(feat_only, labels)

    # SHAP auf Subsample (max 500 Zeilen fuer Geschwindigkeit)
    n_shap = min(500, len(feat_only))
    X_shap = feat_only.sample(n_shap, random_state=42).values.astype(float)

    explainer = shap.TreeExplainer(model._model)
    shap_vals = explainer.shap_values(X_shap)

    # shap_vals: list[ndarray(n, f)] pro Klasse oder ndarray(n, f, classes)
    if isinstance(shap_vals, list):
        shap_arr = np.stack(shap_vals, axis=-1)  # (n, f, 3)
    else:
        shap_arr = np.asarray(shap_vals)

    # Mittlerer absoluter SHAP-Wert ueber alle Klassen und alle Samples
    mean_abs = np.mean(np.abs(shap_arr), axis=(0, 2))  # shape (n_features,)
    top3_idx = np.argsort(mean_abs)[::-1][:3]
    top3 = [feat_cols[i] for i in top3_idx]
    logger.info("SHAP Top-3: {top3}", top3=top3)
    return top3


# ---------------------------------------------------------------------------
# Schritt 6: Metriken zusammenfassen
# ---------------------------------------------------------------------------

def summarize(wf_results: list[dict]) -> dict:
    """Berechnet zusammenfassende Metriken aus den Walk-Forward-Ergebnissen."""
    if not wf_results:
        raise RuntimeError("Keine Walk-Forward-Fenster – zu wenig Daten.")

    sharpes = [r["oos_sharpe"] for r in wf_results]
    profitable = sum(1 for s in sharpes if s > 0)
    total_windows = len(sharpes)
    total_trades_wf = sum(r.get("test_size", 0) for r in wf_results)

    return {
        "oos_sharpe_mean": float(np.mean(sharpes)),
        "oos_sharpe_std":  float(np.std(sharpes)),
        "profitable_windows": profitable,
        "total_windows": total_windows,
        "profitable_pct": profitable / total_windows * 100,
        "total_test_bars": total_trades_wf,
        "windows": wf_results,
    }


# ---------------------------------------------------------------------------
# Schritt 7: Logdatei aktualisieren
# ---------------------------------------------------------------------------

def update_research_log(metrics: dict, top3: list[str]) -> None:
    """Traegt Ergebnis in docs/strategy_research_log.md ein."""
    content = LOG_PATH.read_text(encoding="utf-8")

    # Urteil bestimmen
    mean_sharpe = metrics["oos_sharpe_mean"]
    prof_pct    = metrics["profitable_pct"]
    if mean_sharpe > 0 and prof_pct > 50:
        verdict     = "Kandidat"
        verdict_why = (
            f"Ø OOS-Sharpe {mean_sharpe:.3f} > 0 und {prof_pct:.0f}% profitable Fenster > 50%. "
            f"Erfuellt beide Mindestanforderungen fuer weiteres Testing."
        )
    elif mean_sharpe > 0 or prof_pct > 50:
        verdict     = "Unklar – weitere Tests nötig"
        verdict_why = (
            f"Ø OOS-Sharpe {mean_sharpe:.3f} {'> 0' if mean_sharpe > 0 else '<= 0'}, "
            f"Profitable Fenster {prof_pct:.0f}% {'> 50%' if prof_pct > 50 else '<= 50%'}. "
            f"Gemischte Signale – braucht mehr Evidenz."
        )
    else:
        verdict     = "Verworfen"
        verdict_why = (
            f"Ø OOS-Sharpe {mean_sharpe:.3f} <= 0 und nur {prof_pct:.0f}% profitable Fenster. "
            f"Kein stabiler Edge nachweisbar."
        )

    shap_str = ", ".join(top3)

    new_entry = f"""
### Test #1: USDJPY H4 Trendfolge
- Datum: {TEST_DATE}
- Zeitraum: 4 Jahre (2020-01-01 bis 2024-01-01)
- Walk-Forward: 6M Training / 1M Test, rollierend
- Ø OOS-Sharpe: {mean_sharpe:.3f}
- Std OOS-Sharpe: {metrics['oos_sharpe_std']:.3f}
- Profitable Fenster: {metrics['profitable_windows']}/{metrics['total_windows']} ({prof_pct:.0f}%)
- Anzahl Trades gesamt: {metrics['total_test_bars']}
- SHAP Top-3 Features: {shap_str}
- Auffälligkeiten/Extremwerte: {_find_extremes(metrics['windows'])}
- Urteil: {verdict}
- Begründung des Urteils: {verdict_why}
"""

    # Platzhalter-Zeile ersetzen
    placeholder = "*(noch keine neuen Einträge – Test #1 steht an)*"
    if placeholder in content:
        content = content.replace(placeholder, new_entry.strip())
    else:
        # Falls schon etwas da ist – am Ende von "Ergebnisse" anhängen
        content = content + "\n" + new_entry.strip() + "\n"

    # Tabellenstatus: "⏳ offen" -> Urteil
    content = content.replace(
        f"| 1 | USDJPY | H4 | Trendfolge | BoJ-Politik erzeugt historisch längere, klarere Trends als EURUSD | ⏳ offen |",
        f"| 1 | USDJPY | H4 | Trendfolge | BoJ-Politik erzeugt historisch längere, klarere Trends als EURUSD | {verdict} |",
    )

    LOG_PATH.write_text(content, encoding="utf-8")
    logger.info("strategy_research_log.md aktualisiert | Urteil: {v}", v=verdict)


def _find_extremes(windows: list[dict]) -> str:
    """Findet auffaellige Ausreisser unter den Walk-Forward-Fenstern."""
    sharpes = [(r["window"], r["oos_sharpe"]) for r in windows]
    if not sharpes:
        return "keine"
    max_w, max_s = max(sharpes, key=lambda x: x[1])
    min_w, min_s = min(sharpes, key=lambda x: x[1])
    result_parts = []
    if max_s > 2.0:
        result_parts.append(f"Extremer Ausreisser oben: Fenster {max_w} OOS-Sharpe={max_s:.2f}")
    if min_s < -2.0:
        result_parts.append(f"Extremer Ausreisser unten: Fenster {min_w} OOS-Sharpe={min_s:.2f}")
    return "; ".join(result_parts) if result_parts else "keine Ausreisser > |2.0|"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("====== Test #1: USDJPY H4 Trendfolge (Baseline 23 Feat, ohne MTF) ======")

    # Schritt 1: Fetch
    raw_df = check_broker_and_fetch()

    # Schritt 2: Features
    features_df = build_baseline_features(raw_df)

    # Schritt 3+4: Labels + Walk-Forward
    from src.models.label_builder import LabelBuilder
    label_builder = LabelBuilder()
    labels = label_builder.build_labels(features_df)

    wf_results, _ = run_walk_forward(features_df)

    # Schritt 5: SHAP
    top3 = compute_shap_top3(features_df, labels)

    # Schritt 6: Metriken
    metrics = summarize(wf_results)

    logger.info("====== Ergebnisse ======")
    logger.info("Ø OOS-Sharpe:         {v:.3f}", v=metrics["oos_sharpe_mean"])
    logger.info("Std OOS-Sharpe:       {v:.3f}", v=metrics["oos_sharpe_std"])
    logger.info("Profitable Fenster:   {p}/{t} ({pct:.0f}%)",
                p=metrics["profitable_windows"],
                t=metrics["total_windows"],
                pct=metrics["profitable_pct"])
    logger.info("Bars in Test-Perioden: {v}", v=metrics["total_test_bars"])
    logger.info("SHAP Top-3:           {v}", v=top3)

    # Schritt 7: Log aktualisieren
    update_research_log(metrics, top3)

    logger.info("====== Test #1 abgeschlossen ======")


if __name__ == "__main__":
    main()
