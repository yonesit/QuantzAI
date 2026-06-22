"""
scripts/run_test4_eurusd_h4_mr.py
Test #4: EURUSD H4 Mean-Reversion (MeanReversionModel)

Ablauf:
  1. MT5 Verbindung + Verfuegbarkeitspruefung EURUSD H4
  2. Daten holen (2020-01-01 bis 2024-01-01)
  3. MR-Features bauen (26: Standard-23 + bb_pct_b, dist_ema20_atr, dist_sma50_atr)
  4. Labels erzeugen (MR-Parameter: tp=1.0, sl=2.0, max_candles=10)
  5. Walk-Forward 6M/1M
  6. SHAP Top-3 (TreeExplainer, 500 Samples)
  7. Robustheits-Analyse (Ausreisser |s - mean| > 2*std)
  8. Historische Einordnung der Ausreisser
  9. Research-Log aktualisieren + Vergleich mit H1-Baseline
 10. pytest 100% gruen (nur relevante Unit-Tests)

Ausfuehrung:
  python scripts/run_test4_eurusd_h4_mr.py
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

SYMBOL     = "EURUSD"
TIMEFRAME  = "H4"
START_DATE = datetime(2020, 1, 1, tzinfo=timezone.utc)
END_DATE   = datetime(2024, 1, 1, tzinfo=timezone.utc)
LOG_PATH   = Path(__file__).resolve().parents[1] / "docs" / "strategy_research_log.md"
TEST_DATE  = date.today().isoformat()

TRAIN_MONTHS = 6
TEST_MONTHS  = 1

# Ausreisser-Schwelle
OUTLIER_SIGMA = 2.0


# ---------------------------------------------------------------------------
# Schritt 1: MT5 Verbindung und Daten holen
# ---------------------------------------------------------------------------

def check_broker_and_fetch() -> pd.DataFrame:
    """Verbindet mit MT5, prueft EURUSD H4 und holt Daten."""
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
        "Hole {sym} {tf} Daten | {s} – {e}",
        sym=SYMBOL, tf=TIMEFRAME, s=START_DATE.date(), e=END_DATE.date(),
    )
    df = mt5.get_ohlcv(SYMBOL, TIMEFRAME, START_DATE, END_DATE)
    mt5.disconnect()

    logger.info(
        "Daten: {n} Bars | {s} – {e}",
        n=len(df), s=df.index[0].date(), e=df.index[-1].date(),
    )

    # Mindestens 3000 Bars (4 Jahre H4 = ~8766 Stunden / 4 ≈ 2191 Bars Trading-Zeit)
    if len(df) < 3000:
        raise RuntimeError(
            f"Zu wenige H4-Bars: {len(df)} (min 3000 erwartet). "
            "Broker liefert nicht genug History."
        )
    logger.info("History-Check bestanden: {n} H4-Bars", n=len(df))
    return df


# ---------------------------------------------------------------------------
# Schritt 2: Validieren & MR-Features bauen
# ---------------------------------------------------------------------------

def build_mr_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Validiert und baut 26-Feature-MR-Set."""
    from src.data.validator import DataValidator
    from src.models.mean_reversion_model import MeanReversionModel

    logger.info("=== Schritt 2: Validieren & MR-Features bauen ===")

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

    mr_model = MeanReversionModel()
    features_df = mr_model.build_features(clean_df, symbol=SYMBOL, timeframe=TIMEFRAME)

    feat_cols = [c for c in features_df.columns
                 if c not in {"timestamp", "close", "high", "low"}]
    logger.info(
        "MR-Features: {n} Spalten | {rows} Zeilen",
        n=len(feat_cols), rows=len(features_df),
    )
    assert len(feat_cols) == 26, (
        f"Erwartet 26 MR-Features, got {len(feat_cols)}: {feat_cols}"
    )
    for mr_feat in ("bb_pct_b", "dist_ema20_atr", "dist_sma50_atr"):
        assert mr_feat in features_df.columns, f"MR-Feature '{mr_feat}' fehlt"

    return features_df


# ---------------------------------------------------------------------------
# Schritt 3+4: Labels und Walk-Forward
# ---------------------------------------------------------------------------

def run_walk_forward(features_df: pd.DataFrame) -> tuple[list[dict], pd.Series]:
    """Erzeugt MR-Labels und fuehrt Walk-Forward aus."""
    from src.models.mean_reversion_model import MeanReversionModel
    from src.models.signal_model import SignalModel

    logger.info("=== Schritt 3: MR-Labels (tp=1.0, sl=2.0, max_candles=10) ===")
    label_builder = MeanReversionModel.default_label_builder()
    labels = label_builder.build_labels(features_df)

    dist = {v: int((labels == v).sum()) for v in [-1, 0, 1]}
    total = len(labels)
    logger.info(
        "Label-Verteilung: Short={s} ({sp:.1f}%) Neutral={n} ({np:.1f}%) Long={l} ({lp:.1f}%)",
        s=dist[-1], sp=dist[-1]/total*100,
        n=dist[0],  np=dist[0]/total*100,
        l=dist[1],  lp=dist[1]/total*100,
    )

    logger.info(
        "=== Schritt 4: Walk-Forward {train}M/{test}M ===",
        train=TRAIN_MONTHS, test=TEST_MONTHS,
    )
    exclude = {"label", "timestamp", "open", "volume", "close", "high", "low"}
    feat_cols = [c for c in features_df.columns if c not in exclude]
    features_with_ts = features_df[
        feat_cols + (["timestamp"] if "timestamp" in features_df.columns else [])
    ]

    # SignalModel intern (MeanReversionModel.walk_forward_validate delegiert)
    model = SignalModel()
    wf_results = model.walk_forward_validate(
        features_with_ts,
        labels,
        timestamp_col="timestamp",
        train_months=TRAIN_MONTHS,
        test_months=TEST_MONTHS,
    )
    logger.info("Walk-Forward abgeschlossen: {n} Fenster", n=len(wf_results))

    for r in wf_results:
        logger.debug(
            "  Fenster {w:2d} | {ts} – {te} | Sharpe={s:+.3f} | Trades={t}",
            w=r["window"], ts=r.get("test_start", "?"), te=r.get("test_end", "?"),
            s=r["oos_sharpe"], t=r.get("test_size", "?"),
        )

    return wf_results, labels


# ---------------------------------------------------------------------------
# Schritt 5: SHAP Top-3
# ---------------------------------------------------------------------------

def compute_shap_top3(features_df: pd.DataFrame, labels: pd.Series) -> list[str]:
    """Finales Modell + SHAP mean-abs Top-3."""
    import shap
    from src.models.signal_model import SignalModel
    from src.models.mean_reversion_model import MeanReversionModel

    logger.info("=== Schritt 5: Finales Modell + SHAP Top-3 ===")

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
        "oos_sharpe_median":  float(np.median(sharpes)),
        "profitable_windows": profitable,
        "total_windows":      total,
        "profitable_pct":     profitable / total * 100,
        "total_test_bars":    test_bars,
        "sharpes":            sharpes,
        "windows":            wf_results,
    }


# ---------------------------------------------------------------------------
# Schritt 7: Robustheits-Analyse
# ---------------------------------------------------------------------------

def analyse_robustness(metrics: dict) -> dict:
    """
    Erkennt Ausreisser (|s - mean| > OUTLIER_SIGMA * std) und berechnet
    Statistiken mit und ohne Ausreisser.

    Gibt dict mit:
      full        : Vollstaendige Statistiken
      filtered    : Statistiken ohne Ausreisser (None wenn keine Ausreisser)
      outlier_idxs: Indizes der Ausreisser-Fenster
    """
    sharpes = np.array(metrics["sharpes"])
    mean_s  = sharpes.mean()
    std_s   = sharpes.std()
    threshold = OUTLIER_SIGMA * std_s

    outlier_idxs = [
        i for i, s in enumerate(sharpes)
        if abs(s - mean_s) > threshold
    ]

    def _stats(arr: np.ndarray, label: str) -> dict:
        n = len(arr)
        pct = float((arr > 0).sum() / n * 100)
        return {
            "label":          label,
            "n":              n,
            "mean":           float(arr.mean()),
            "std":            float(arr.std()),
            "median":         float(np.median(arr)),
            "profitable":     int((arr > 0).sum()),
            "profitable_pct": pct,
        }

    full = _stats(sharpes, "Alle Fenster")

    if outlier_idxs:
        mask = [i for i in range(len(sharpes)) if i not in outlier_idxs]
        filtered_arr = sharpes[mask]
        filtered = _stats(filtered_arr, f"Ohne {len(outlier_idxs)} Ausreisser")
    else:
        filtered = None

    if outlier_idxs:
        logger.info(
            "Ausreisser-Fenster ({n}): {idxs}",
            n=len(outlier_idxs), idxs=outlier_idxs,
        )
        for i in outlier_idxs:
            w = metrics["windows"][i]
            logger.info(
                "  Fenster {i} | {ts} – {te} | OOS-Sharpe={s:+.3f}",
                i=i, ts=w.get("test_start", "?"), te=w.get("test_end", "?"),
                s=w["oos_sharpe"],
            )
    else:
        logger.info("Keine Ausreisser gefunden (Schwelle: |s - mean| > {s:.1f}*std)", s=OUTLIER_SIGMA)

    return {
        "full":         full,
        "filtered":     filtered,
        "outlier_idxs": outlier_idxs,
    }


# ---------------------------------------------------------------------------
# Schritt 8: Historische Einordnung der Ausreisser
# ---------------------------------------------------------------------------

def classify_outliers(outlier_idxs: list[int], wf_results: list[dict]) -> list[dict]:
    """
    Ordnet jedes Ausreisser-Fenster historisch ein.
    Gibt Liste von Dicts mit: window_idx, test_start, test_end, sharpe, note, recurring.
    """
    # Bekannte Markt-Events auf EURUSD H4 (2020-2024)
    # recurring=True → normales, wiederkehrendes Risiko
    # recurring=False → singulaeres Event
    EVENT_MAP = [
        # Covid-Crash / Risk-Off Schock
        (("2020-02-01", "2020-05-01"), "Covid-19-Crash: Risk-Off, EURUSD-Volatilitaetsspike. Normal-wiederkehrendes Risiko (Liquiditaets-Spread).", True),
        # US-Wahl 2020
        (("2020-10-01", "2020-12-01"), "US-Praesidentschaftswahl 2020: kurzfristige EURUSD-Volatilitaet. Normal-wiederkehrend (Wahlrisiko).", True),
        # Fed Taper Tantrum 2021
        (("2021-09-01", "2021-12-01"), "Fed Tapering-Ankuendigung 2021: USD-Staerkung, EURUSD-Druck. Normal-wiederkehrend (Fed-Zyklus).", True),
        # Ukraine-Krieg
        (("2022-01-15", "2022-04-15"), "Ukraine-Krieg Feb 2022: EUR-Abwertung, Risk-Off. Singulaeres Ereignis.", False),
        # Aggressive Fed-Zinszyklen 2022
        (("2022-04-01", "2022-10-01"), "Aggressivster Fed-Zyklus seit 1980: USD-Staerkung, EURUSD auf 20-Jahres-Tief. Normal-wiederkehrendes Risiko (Zins-Zyklus).", True),
        # UK-Gilts-Krise / LDI
        (("2022-09-01", "2022-11-01"), "UK-Gilts-Krise (Truss/Kwarteng): EUR-Kollateralschaden, Liquiditaets-Stress. Weitgehend singulaer.", False),
        # SVB-Bankenkrise
        (("2023-03-01", "2023-05-01"), "SVB-Bankenkrise Maerz 2023: Risk-Off, USD-Safe-Haven. Normal-wiederkehrendes Risiko (Bankenstress).", True),
    ]

    results = []
    for idx in outlier_idxs:
        w = wf_results[idx]
        ts = str(w.get("test_start", ""))
        te = str(w.get("test_end", ""))
        sharpe = w["oos_sharpe"]

        note = "Kein spezifisches Event identifiziert – moegliche normal-wiederkehrende Volatilitaet."
        recurring = True

        for (ev_start, ev_end), ev_note, ev_recurring in EVENT_MAP:
            if ts[:7] >= ev_start[:7] and ts[:7] <= ev_end[:7]:
                note = ev_note
                recurring = ev_recurring
                break

        results.append({
            "window_idx": idx,
            "test_start": ts,
            "test_end":   te,
            "sharpe":     sharpe,
            "note":       note,
            "recurring":  recurring,
        })
        recurring_label = "Normal-wiederkehrend" if recurring else "Singulaer"
        logger.info(
            "Fenster {i} ({ts}–{te}) | Sharpe={s:+.3f} | {rl}: {n}",
            i=idx, ts=ts, te=te, s=sharpe, rl=recurring_label, n=note,
        )

    return results


# ---------------------------------------------------------------------------
# Schritt 9: Research-Log aktualisieren
# ---------------------------------------------------------------------------

def _determine_verdict(rb: dict) -> tuple[str, str]:
    """Urteil aus Robustheits-Analyse. Primaer: filtered wenn vorhanden."""
    primary = rb["filtered"] if rb["filtered"] else rb["full"]
    m = primary["mean"]
    p = primary["profitable_pct"]

    if m > 0 and p > 50:
        verdict = "Kandidat"
        why = (
            f"Ø OOS-Sharpe {m:.3f} > 0 und {p:.0f}% profitable Fenster > 50%. "
            "Beide Mindestanforderungen erfuellt."
        )
    elif m > 0 or p > 50:
        verdict = "Unklar – weitere Tests nötig"
        why = (
            f"Gemischte Signale: Ø OOS-Sharpe {m:.3f} "
            f"({'> 0' if m > 0 else '<= 0'}), "
            f"Profitable Fenster {p:.0f}% "
            f"({'> 50%' if p > 50 else '<= 50%'})."
        )
    else:
        verdict = "Verworfen"
        why = (
            f"Ø OOS-Sharpe {m:.3f} <= 0 und nur {p:.0f}% profitable Fenster <= 50%. "
            "Kein stabiler Edge nachweisbar."
        )

    return verdict, why


def _extremes_note(wf_results: list[dict], outlier_idxs: list[int]) -> str:
    sharpes = [(r["window"], r["oos_sharpe"]) for r in wf_results]
    max_w, max_s = max(sharpes, key=lambda x: x[1])
    min_w, min_s = min(sharpes, key=lambda x: x[1])
    parts = []
    if max_s > 2.0:
        max_r = wf_results[max_w]
        ts = str(max_r.get("test_start", "?"))[:10]
        te = str(max_r.get("test_end", "?"))[:10]
        parts.append(f"Ausreisser oben: Fenster {max_w} ({ts}–{te}) OOS-Sharpe={max_s:.2f}")
    if min_s < -2.0:
        min_r = wf_results[min_w]
        ts = str(min_r.get("test_start", "?"))[:10]
        te = str(min_r.get("test_end", "?"))[:10]
        parts.append(f"Ausreisser unten: Fenster {min_w} ({ts}–{te}) OOS-Sharpe={min_s:.2f}")
    return "; ".join(parts) if parts else "keine Ausreisser > |2.0|"


def update_research_log(
    metrics: dict,
    top3: list[str],
    rb: dict,
    outlier_details: list[dict],
    verdict: str,
    verdict_why: str,
) -> None:
    content = LOG_PATH.read_text(encoding="utf-8")

    full = rb["full"]
    filtered = rb["filtered"]
    outlier_idxs = rb["outlier_idxs"]

    shap_str  = ", ".join(top3)
    extremes  = _extremes_note(metrics["windows"], outlier_idxs)

    # Aufbau Basis-Eintrag
    new_entry_lines = [
        f"",
        f"### Test #4: EURUSD H4 Mean-Reversion",
        f"- Datum: {TEST_DATE}",
        f"- Zeitraum: 4 Jahre (2020-01-01 bis 2024-01-01)",
        f"- Modell: MeanReversionModel (26 Features: Standard-23 + bb_pct_b, dist_ema20_atr, dist_sma50_atr)",
        f"- Label-Parameter: tp_atr_mult=1.0, sl_atr_mult=2.0, max_candles=10 (H4 = ~2 Handelstage)",
        f"- Walk-Forward: {TRAIN_MONTHS}M Training / {TEST_MONTHS}M Test, rollierend ({full['n']} Fenster)",
        f"- Ø OOS-Sharpe: {full['mean']:.3f}",
        f"- Std OOS-Sharpe: {full['std']:.3f}",
        f"- Median OOS-Sharpe: {full['median']:.3f}",
        f"- Profitable Fenster: {full['profitable']}/{full['n']} ({full['profitable_pct']:.0f}%)",
        f"- Anzahl Trades gesamt: {metrics['total_test_bars']}",
        f"- SHAP Top-3 Features: {shap_str}",
        f"- Auffälligkeiten/Extremwerte: {extremes}",
        f"- Urteil: {verdict}",
        f"- Begründung des Urteils: {verdict_why}",
    ]

    # Robustheits-Block (nur wenn Ausreisser vorhanden)
    if outlier_idxs and filtered:
        new_entry_lines += [
            "",
            "**Robustheits-Analyse:**",
            "",
            "| Metrik | Alle Fenster | Ohne Ausreisser |",
            "|--------|-------------|----------------|",
            f"| Ø OOS-Sharpe | {full['mean']:.3f} | {filtered['mean']:.3f} |",
            f"| Std OOS-Sharpe | {full['std']:.3f} | {filtered['std']:.3f} |",
            f"| Median OOS-Sharpe | {full['median']:.3f} | {filtered['median']:.3f} |",
            f"| Profitable Fenster | {full['profitable']}/{full['n']} ({full['profitable_pct']:.0f}%) | {filtered['profitable']}/{filtered['n']} ({filtered['profitable_pct']:.0f}%) |",
        ]

        # Ausreisser-Details
        for d in outlier_details:
            recurring_str = "normal-wiederkehrend" if d["recurring"] else "singulaer"
            new_entry_lines.append(
                f"\n**Fenster {d['window_idx']} ({d['test_start'][:10]} – {d['test_end'][:10]}):** "
                f"OOS-Sharpe={d['sharpe']:+.2f}. Einordnung ({recurring_str}): {d['note']}"
            )

    # Vergleich mit H1-Baseline
    new_entry_lines += [
        "",
        "**Vergleich mit EURUSD H1-Baseline:**",
        "EURUSD H1 Mean-Reversion wurde nicht separat getestet (kein Eintrag im Log, "
        "kein entsprechender Commit in der Git-History). "
        "Vergleich gegen EURUSD H1 Trendfolge-Baseline (Test 0a, Ø OOS-Sharpe -0.484, 39% profitable Fenster): "
        f"MR H4 {'übertrifft' if full['mean'] > -0.484 else 'unterbietet'} die H1-TF-Baseline "
        f"im Ø OOS-Sharpe ({full['mean']:.3f} vs. -0.484) und "
        f"{'übertrifft' if full['profitable_pct'] > 39 else 'unterbietet'} sie in der Profitablen-Quote "
        f"({full['profitable_pct']:.0f}% vs. 39%). "
        "Ein direkter H1-MR ↔ H4-MR Vergleich ist nicht moeglich – das Experiment #H1-MR fehlt.",
    ]

    new_entry = "\n".join(new_entry_lines) + "\n"

    # An den Log anhaengen
    content = content.rstrip() + "\n" + new_entry.strip() + "\n"

    # Tabellenstatus aktualisieren
    content = content.replace(
        "| 4 | EURUSD | H4 | Mean-Reversion | Testet ob MR auf höherem Timeframe als H1 besser performt | ⏳ offen |",
        f"| 4 | EURUSD | H4 | Mean-Reversion | Testet ob MR auf höherem Timeframe als H1 besser performt | {verdict} |",
    )

    LOG_PATH.write_text(content, encoding="utf-8")
    logger.info("strategy_research_log.md aktualisiert | Urteil: {v}", v=verdict)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("====== Test #4: EURUSD H4 Mean-Reversion (MeanReversionModel) ======")

    # 1. Daten holen
    raw_df = check_broker_and_fetch()

    # 2. MR-Features
    features_df = build_mr_features(raw_df)

    # 3-4. Walk-Forward
    wf_results, labels = run_walk_forward(features_df)

    # 5. SHAP
    top3 = compute_shap_top3(features_df, labels)

    # 6. Metriken
    metrics = summarize(wf_results)

    logger.info("====== Ergebnisse (alle Fenster) ======")
    logger.info("Ø OOS-Sharpe:          {v:.3f}", v=metrics["oos_sharpe_mean"])
    logger.info("Median OOS-Sharpe:     {v:.3f}", v=metrics["oos_sharpe_median"])
    logger.info("Std OOS-Sharpe:        {v:.3f}", v=metrics["oos_sharpe_std"])
    logger.info(
        "Profitable Fenster:    {p}/{t} ({pct:.0f}%)",
        p=metrics["profitable_windows"],
        t=metrics["total_windows"],
        pct=metrics["profitable_pct"],
    )
    logger.info("Bars in Test-Perioden: {v}", v=metrics["total_test_bars"])
    logger.info("SHAP Top-3:            {v}", v=top3)

    # 7. Robustheit
    rb = analyse_robustness(metrics)

    # 8. Historische Einordnung
    outlier_details = classify_outliers(rb["outlier_idxs"], wf_results)

    # Urteil (primaer: gefiltert wenn Ausreisser vorhanden)
    verdict, verdict_why = _determine_verdict(rb)
    logger.info("Urteil: {v}", v=verdict)

    if rb["filtered"]:
        f = rb["filtered"]
        logger.info("====== Ergebnisse (ohne Ausreisser) ======")
        logger.info("Ø OOS-Sharpe:          {v:.3f}", v=f["mean"])
        logger.info("Median OOS-Sharpe:     {v:.3f}", v=f["median"])
        logger.info(
            "Profitable Fenster:    {p}/{t} ({pct:.0f}%)",
            p=f["profitable"], t=f["n"], pct=f["profitable_pct"],
        )

    # 9. Log
    update_research_log(metrics, top3, rb, outlier_details, verdict, verdict_why)

    logger.info("====== Test #4 abgeschlossen ======")


if __name__ == "__main__":
    main()
