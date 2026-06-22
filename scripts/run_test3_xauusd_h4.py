"""
scripts/run_test3_xauusd_h4.py
Test #3: XAUUSD H4 Trendfolge – Baseline (23 Features, ohne MTF)

Ablauf:
  1. Broker-Check: XAUUSD vorhanden? Genueg Historie (4 Jahre)?
     -> Falls nein: sauber abbrechen mit Meldung.
  2. H4-Daten 2020-01-01 bis 2024-01-01 fetchen.
  3. Features bauen (23-Feature-Baseline, OHNE MTF – wie Test #1/#2).
  4. Walk-Forward 6M/1M (rollierend).
  5. Metriken inkl. Median OOS-Sharpe und Robustheits-Check.
  6. Falls Ausreisser dominieren: Vergleich mit/ohne automatisch einbauen.
  7. docs/strategy_research_log.md aktualisieren.

Symbol-Hinweis:
  Fusion Markets listet Gold als "XAUUSD". Falls der exakte Name abweicht,
  wird automatisch nach Alternativen gesucht (XAU*, *GOLD*, *Gold*).
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

SYMBOL_CANDIDATES = ["XAUUSD", "XAUUSD.", "XAU/USD", "GOLD", "Gold"]
TIMEFRAME  = "H4"
START_DATE = datetime(2020, 1, 1, tzinfo=timezone.utc)
END_DATE   = datetime(2024, 1, 1, tzinfo=timezone.utc)
LOG_PATH   = Path(__file__).resolve().parents[1] / "docs" / "strategy_research_log.md"
TEST_DATE  = date.today().isoformat()

MIN_BARS   = 3000   # Mindest-H4-Bars fuer 4 Jahre (exkl. Wochenenden)
WARMUP     = 200    # FeatureBuilder warmup_candles


# ---------------------------------------------------------------------------
# Schritt 1: Broker-Check
# ---------------------------------------------------------------------------

def check_broker() -> tuple[str | None, "pd.DataFrame | None"]:
    """
    Verbindet mit MT5, sucht XAUUSD und prueft Historie.

    Returns
    -------
    (symbol, df)  – beides None wenn Symbol/Historie nicht verfuegbar.
    """
    from src.data.mt5_connector import MT5Connector, MT5DataError

    mt5 = MT5Connector(
        login=int(os.environ.get("MT5_LOGIN", "0")),
        password=os.environ.get("MT5_PASSWORD", ""),
        server=os.environ.get("MT5_SERVER", ""),
    )
    logger.info("=== Schritt 1: Broker-Check ===")
    mt5.connect()
    available = set(mt5.get_available_symbols())
    logger.info("Broker bietet {n} Symbole an", n=len(available))

    # Symbol-Suche
    symbol = None
    for cand in SYMBOL_CANDIDATES:
        if cand in available:
            symbol = cand
            logger.info("Symbol gefunden: '{s}'", s=symbol)
            break

    if symbol is None:
        # Breite Suche nach XAU / Gold im Symbollisting
        gold_like = [s for s in available
                     if "XAU" in s.upper() or "GOLD" in s.upper()]
        if gold_like:
            symbol = gold_like[0]
            logger.warning(
                "Kein exakter XAUUSD-Match – verwende '{s}'. "
                "Alternativsymbole: {a}",
                s=symbol, a=gold_like[:5],
            )
        else:
            mt5.disconnect()
            logger.error(
                "XAUUSD / Gold nicht beim Broker verfuegbar. "
                "Test #3 wird uebersprungen."
            )
            return None, None

    # Daten holen
    logger.info(
        "Hole {sym} H4 | {s} – {e}",
        sym=symbol, s=START_DATE.date(), e=END_DATE.date(),
    )
    try:
        df = mt5.get_ohlcv(symbol, TIMEFRAME, START_DATE, END_DATE)
    except MT5DataError as exc:
        mt5.disconnect()
        logger.error(
            "Datenabruf fehlgeschlagen fuer {sym}: {exc}. "
            "Test #3 wird uebersprungen.",
            sym=symbol, exc=exc,
        )
        return None, None
    finally:
        mt5.disconnect()

    logger.info(
        "Daten: {n} Bars | {s} – {e}",
        n=len(df), s=df.index[0].date(), e=df.index[-1].date(),
    )

    if len(df) < MIN_BARS:
        logger.error(
            "Zu wenige Bars: {n} < {m}. Nicht genug Historie beim Broker. "
            "Test #3 wird uebersprungen.",
            n=len(df), m=MIN_BARS,
        )
        return None, None

    logger.info("History-Check bestanden: {n} H4-Bars fuer {s}", n=len(df), s=symbol)
    return symbol, df


# ---------------------------------------------------------------------------
# Schritt 2: Validieren + Features
# ---------------------------------------------------------------------------

def build_features(raw_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    from src.data.validator import DataValidator
    from src.data.feature_builder import FeatureBuilder

    logger.info("=== Schritt 2: Validieren & Features bauen ===")

    df_reset = raw_df.reset_index()
    df_reset = df_reset.rename(columns={df_reset.columns[0]: "timestamp"})

    validator = DataValidator()
    report, clean_df = validator.validate(df_reset, symbol=symbol, timeframe=TIMEFRAME)
    logger.info(
        "Validator: quality={q:.3f} usable={u} candles={c}",
        q=report.quality_score, u=report.is_usable, c=report.total_candles,
    )
    if not report.is_usable:
        raise RuntimeError(f"Datenqualitaet ungenuegend: {report.errors}")

    # Baseline: kein df_h4, kein df_d1 -> kein MTF (H4-Pipeline haengt ohnehin keins an)
    builder = FeatureBuilder()
    features_df = builder.build(
        clean_df, symbol=symbol, timeframe=TIMEFRAME, df_h4=None, df_d1=None,
    )

    feat_cols = [c for c in features_df.columns
                 if c not in {"timestamp", "close", "high", "low"}]
    assert "h4_trend" not in features_df.columns
    assert "d1_trend" not in features_df.columns
    assert len(feat_cols) == 23, f"Erwartet 23 Features, got {len(feat_cols)}"
    logger.info("Features: {n} Spalten | {r} Zeilen", n=len(feat_cols), r=len(features_df))
    return features_df


# ---------------------------------------------------------------------------
# Schritt 3+4: Labels + Walk-Forward
# ---------------------------------------------------------------------------

def run_walk_forward(features_df: pd.DataFrame) -> tuple[list[dict], pd.Series]:
    from src.models.label_builder import LabelBuilder
    from src.models.signal_model import SignalModel

    logger.info("=== Schritt 3+4: Labels + Walk-Forward 6M/1M ===")
    lb = LabelBuilder()
    labels = lb.build_labels(features_df)

    exclude = {"label", "timestamp", "open", "volume", "close", "high", "low"}
    feat_cols = [c for c in features_df.columns if c not in exclude]
    ts_cols = feat_cols + (["timestamp"] if "timestamp" in features_df.columns else [])

    model = SignalModel()
    wf_results = model.walk_forward_validate(
        features_df[ts_cols], labels,
        timestamp_col="timestamp",
        train_months=6, test_months=1,
    )
    logger.info("Walk-Forward: {n} Fenster", n=len(wf_results))
    return wf_results, labels


# ---------------------------------------------------------------------------
# Schritt 5: SHAP Top-3
# ---------------------------------------------------------------------------

def compute_shap_top3(features_df: pd.DataFrame, labels: pd.Series) -> list[str]:
    import shap
    from src.models.signal_model import SignalModel

    logger.info("=== Schritt 5: SHAP Top-3 ===")
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
# Schritt 6: Metriken + Robustheits-Analyse
# ---------------------------------------------------------------------------

def _stats(sharpes: list[float], label: str) -> dict:
    arr = np.array(sharpes)
    profitable = int((arr > 0).sum())
    n = len(arr)
    return {
        "label": label,
        "n": n,
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "profitable": profitable,
        "profitable_pct": profitable / n * 100,
    }


def analyse_robustness(wf_results: list[dict]) -> dict:
    """
    Berechnet Metriken gesamt + prueft ob einzelne Fenster dominieren.

    Ausreisser-Definition: |OOS-Sharpe| > 2 * IQR-bereinigter Schwelle.
    Einfachere Heuristik: Fenster deren |Sharpe| > mean + 2*std gilt als Ausreisser.
    """
    sharpes = [r["oos_sharpe"] for r in wf_results]
    full = _stats(sharpes, "Alle Fenster")

    # Ausreisser-Erkennung: |sharpe - mean| > 2 * std
    threshold = 2.0 * full["std"]
    outlier_indices = [
        i for i, s in enumerate(sharpes)
        if abs(s - full["mean"]) > threshold
    ]

    result = {"full": full, "outlier_indices": outlier_indices}

    if outlier_indices:
        filtered_sharpes = [s for i, s in enumerate(sharpes) if i not in set(outlier_indices)]
        filtered = _stats(filtered_sharpes, f"Ohne Ausreisser ({outlier_indices})")
        result["filtered"] = filtered

        logger.info(
            "Ausreisser gefunden (|s - mean| > 2*std = {t:.2f}): Fenster {idx} "
            "| Sharpes: {vals}",
            t=threshold,
            idx=outlier_indices,
            vals=[round(sharpes[i], 3) for i in outlier_indices],
        )
    else:
        result["filtered"] = None
        logger.info("Keine Ausreisser (alle |s - mean| <= 2*std = {t:.2f})", t=threshold)

    _print_comparison(result)
    return result


def _print_comparison(result: dict) -> None:
    full = result["full"]
    filt = result.get("filtered")

    if filt:
        logger.info("=" * 70)
        logger.info("{:40s} {:>14s} {:>14s}", "Metrik", full["label"], filt["label"])
        logger.info("-" * 70)
        for key, fmt in [
            ("n",              "{:>14d} {:>14d}"),
            ("mean",           "{:>14.3f} {:>14.3f}"),
            ("std",            "{:>14.3f} {:>14.3f}"),
            ("median",         "{:>14.3f} {:>14.3f}"),
            ("min",            "{:>14.3f} {:>14.3f}"),
            ("max",            "{:>14.3f} {:>14.3f}"),
        ]:
            logger.info("{:40s} " + fmt, key, full[key], filt[key])
        logger.info(
            "{:40s} {:>14s} {:>14s}",
            "Profitable Fenster",
            f"{full['profitable']}/{full['n']} ({full['profitable_pct']:.0f}%)",
            f"{filt['profitable']}/{filt['n']} ({filt['profitable_pct']:.0f}%)",
        )
        logger.info("=" * 70)
    else:
        logger.info(
            "Ergebnis: Ø Sharpe={m:.3f} Std={s:.3f} Median={md:.3f} "
            "Profitable={p}/{n} ({pct:.0f}%)",
            m=full["mean"], s=full["std"], md=full["median"],
            p=full["profitable"], n=full["n"], pct=full["profitable_pct"],
        )


# ---------------------------------------------------------------------------
# Schritt 7: Urteil
# ---------------------------------------------------------------------------

def determine_verdict(stats: dict) -> tuple[str, str]:
    """
    Bestimmt Urteil basierend auf Metriken.
    Nutzt bevorzugt gefilterte Stats (ohne Ausreisser) wenn vorhanden.
    """
    primary = stats.get("filtered") or stats["full"]
    m   = primary["mean"]
    pct = primary["profitable_pct"]
    outlier_note = (
        f" (ohne Ausreisser-Fenster {stats['outlier_indices']})"
        if stats.get("outlier_indices") else ""
    )

    if m > 0 and pct > 50:
        verdict = "Kandidat"
        why = (
            f"Ø OOS-Sharpe {m:.3f} > 0 und {pct:.0f}% profitable Fenster > 50%"
            f"{outlier_note}. Beide Mindestanforderungen erfuellt."
        )
    elif m > 0 or pct > 50:
        verdict = "Unklar – weitere Tests nötig"
        why = (
            f"Ø OOS-Sharpe {m:.3f} ({'> 0' if m > 0 else '<= 0'}), "
            f"Profitable Fenster {pct:.0f}% ({'> 50%' if pct > 50 else '<= 50%'})"
            f"{outlier_note}. Gemischte Signale."
        )
    else:
        verdict = "Verworfen"
        why = (
            f"Ø OOS-Sharpe {m:.3f} <= 0 und nur {pct:.0f}% profitable Fenster"
            f"{outlier_note}. Kein stabiler Edge."
        )
    return verdict, why


# ---------------------------------------------------------------------------
# Schritt 8: Log aktualisieren
# ---------------------------------------------------------------------------

def _find_outlier_window_dates(
    wf_results: list[dict], outlier_indices: list[int]
) -> str:
    """Gibt Zeitraum-Strings fuer Ausreisser-Fenster zurueck."""
    parts = []
    for i in outlier_indices:
        r = wf_results[i]
        parts.append(
            f"Fenster {i} ({r['test_start']} – {r['test_end']}): "
            f"OOS-Sharpe={r['oos_sharpe']:.2f}"
        )
    return "; ".join(parts)


def _extremes_note(wf_results: list[dict]) -> str:
    sharpes = [(r["window"], r["oos_sharpe"]) for r in wf_results]
    max_w, max_s = max(sharpes, key=lambda x: x[1])
    min_w, min_s = min(sharpes, key=lambda x: x[1])
    parts = []
    if max_s > 2.0:
        r = wf_results[max_w]
        parts.append(
            f"Ausreisser oben: Fenster {max_w} ({r['test_start']}–{r['test_end']}) "
            f"OOS-Sharpe={max_s:.2f}"
        )
    if min_s < -2.0:
        r = wf_results[min_w]
        parts.append(
            f"Ausreisser unten: Fenster {min_w} ({r['test_start']}–{r['test_end']}) "
            f"OOS-Sharpe={min_s:.2f}"
        )
    return "; ".join(parts) if parts else "keine Ausreisser > |2.0|"


def update_research_log(
    symbol: str,
    wf_results: list[dict],
    top3: list[str],
    robustness: dict,
    verdict: str,
    verdict_why: str,
) -> None:
    content = LOG_PATH.read_text(encoding="utf-8")

    full = robustness["full"]
    filt = robustness.get("filtered")
    outlier_indices = robustness.get("outlier_indices", [])
    total_test_bars = sum(r.get("test_size", 0) for r in wf_results)
    extremes = _extremes_note(wf_results)

    # Robustheits-Block (nur wenn Ausreisser gefunden)
    robustness_block = ""
    if filt and outlier_indices:
        outlier_detail = _find_outlier_window_dates(wf_results, outlier_indices)
        robustness_block = (
            f"\n**Robustheits-Analyse:**\n"
            f"\n"
            f"| Metrik | Alle Fenster | Ohne Ausreisser |\n"
            f"|--------|-------------|----------------|\n"
            f"| Ø OOS-Sharpe | {full['mean']:.3f} | {filt['mean']:.3f} |\n"
            f"| Std OOS-Sharpe | {full['std']:.3f} | {filt['std']:.3f} |\n"
            f"| Median OOS-Sharpe | {full['median']:.3f} | {filt['median']:.3f} |\n"
            f"| Profitable Fenster | {full['profitable']}/{full['n']} "
            f"({full['profitable_pct']:.0f}%) | "
            f"{filt['profitable']}/{filt['n']} ({filt['profitable_pct']:.0f}%) |\n"
            f"\n"
            f"**Ausreisser-Fenster:** {outlier_detail}"
        )

    new_entry = f"""
### Test #3: {symbol} H4 Trendfolge
- Datum: {TEST_DATE}
- Zeitraum: 4 Jahre (2020-01-01 bis 2024-01-01)
- Walk-Forward: 6M Training / 1M Test, rollierend ({full['n']} Fenster)
- Ø OOS-Sharpe: {full['mean']:.3f}
- Std OOS-Sharpe: {full['std']:.3f}
- Median OOS-Sharpe: {full['median']:.3f}
- Profitable Fenster: {full['profitable']}/{full['n']} ({full['profitable_pct']:.0f}%)
- Anzahl Trades gesamt: {total_test_bars}
- SHAP Top-3 Features: {', '.join(top3)}
- Auffälligkeiten/Extremwerte: {extremes}
- Urteil: {verdict}
- Begründung des Urteils: {verdict_why}{robustness_block}
"""

    content = content.rstrip() + "\n" + new_entry.strip() + "\n"

    # Tabellenstatus aktualisieren
    content = content.replace(
        "| 3 | XAUUSD | H4 | Trendfolge | Andere Asset-Klasse, andere Treiber (Inflation/Risk-Off), oft trendstärker | ⏳ offen (Datenverfügbarkeit prüfen) |",
        f"| 3 | XAUUSD | H4 | Trendfolge | Andere Asset-Klasse, andere Treiber (Inflation/Risk-Off), oft trendstärker | {verdict} |",
    )

    LOG_PATH.write_text(content, encoding="utf-8")
    logger.info("strategy_research_log.md aktualisiert | Urteil: {v}", v=verdict)


def skip_test3(reason: str) -> None:
    """Traegt 'Uebersprungen' in die Testmatrix ein falls Broker-Check schlaegt fehl."""
    content = LOG_PATH.read_text(encoding="utf-8")
    content = content.replace(
        "| 3 | XAUUSD | H4 | Trendfolge | Andere Asset-Klasse, andere Treiber (Inflation/Risk-Off), oft trendstärker | ⏳ offen (Datenverfügbarkeit prüfen) |",
        f"| 3 | XAUUSD | H4 | Trendfolge | Andere Asset-Klasse, andere Treiber (Inflation/Risk-Off), oft trendstärker | Übersprungen: {reason} |",
    )
    LOG_PATH.write_text(content, encoding="utf-8")
    logger.warning("Test #3 uebersprungen: {r}", r=reason)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("====== Test #3: XAUUSD H4 Trendfolge (Baseline 23 Feat, ohne MTF) ======")

    # Schritt 1: Broker-Check
    symbol, raw_df = check_broker()
    if symbol is None or raw_df is None:
        skip_test3("XAUUSD nicht verfuegbar oder zu wenig Historie beim Broker")
        return

    # Schritt 2: Features
    features_df = build_features(raw_df, symbol)

    # Schritt 3+4: WF
    wf_results, labels = run_walk_forward(features_df)

    # Schritt 5: SHAP
    top3 = compute_shap_top3(features_df, labels)

    # Schritt 6: Robustheit
    robustness = analyse_robustness(wf_results)

    # Schritt 7: Urteil
    verdict, verdict_why = determine_verdict(robustness)

    # Zusammenfassung
    full = robustness["full"]
    logger.info("====== Ergebnisse ======")
    logger.info("Ø OOS-Sharpe:         {v:.3f}", v=full["mean"])
    logger.info("Std OOS-Sharpe:       {v:.3f}", v=full["std"])
    logger.info("Median OOS-Sharpe:    {v:.3f}", v=full["median"])
    logger.info(
        "Profitable Fenster:   {p}/{t} ({pct:.0f}%)",
        p=full["profitable"], t=full["n"], pct=full["profitable_pct"],
    )
    logger.info("Ausreisser-Fenster:   {v}", v=robustness["outlier_indices"])
    logger.info("SHAP Top-3:           {v}", v=top3)
    logger.info("Urteil:               {v}", v=verdict)

    # Schritt 8: Log
    update_research_log(symbol, wf_results, top3, robustness, verdict, verdict_why)
    logger.info("====== Test #3 abgeschlossen ======")


if __name__ == "__main__":
    main()
