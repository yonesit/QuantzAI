"""
scripts/analyse_portfolio_correlation.py
Gesamtauswertung und Korrelationsanalyse aller 4 priorisierten Tests.

Aufgaben:
  1. Zusammenfassung-Tabelle (nach Median OOS-Sharpe sortiert)
  2. Pearson-Korrelation der Kandidaten-Signale (fensterweise)
     - Tests #3 (XAUUSD H4 TF) und #4 (EURUSD H4 MR): WF neu laufen lassen
       (gleiche H4-Periode, Fenster sind zeitlich identisch ausgerichtet)
     - Test #2 (USDJPY D1 TF): hardcodierte Werte, monatliche Ausrichtung
  3. Interpretation: Kombinierbark?
  4. Neuen Abschnitt in docs/strategy_research_log.md eintragen
  5. pytest / commit / push
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

LOG_PATH  = Path(__file__).resolve().parents[1] / "docs" / "strategy_research_log.md"
TEST_DATE = date.today().isoformat()

# ─────────────────────────────────────────────────────────────────────────────
# Bekannte Aggregate-Ergebnisse (aus Research-Log)
# ─────────────────────────────────────────────────────────────────────────────

SUMMARY_DATA = [
    # (test_n, symbol, tf, strategie, mean, median, std, profitable, total, trades)
    (1, "USDJPY", "H4", "Trendfolge",     -0.671, None,  4.480, 15, 40, 5199),
    (2, "USDJPY", "D1", "Trendfolge",      1.208, 1.449, 7.594, 17, 32,  695),
    (3, "XAUUSD", "H4", "Trendfolge",     -0.036, 0.191, 3.678, 22, 40, 5159),
    (4, "EURUSD", "H4", "Mean-Reversion",  0.389, 1.142, 3.670, 27, 40, 5201),
]

# Hardcodierte per-Fenster-Sharpe-Werte fuer Test #2 (aus analyse_test2_robustness.py)
TEST2_SHARPES = [
    -2.935, 5.612, 1.449, -2.935, 0.000, 1.449, 5.612, -10.331, 6.750, 5.612,
     4.992, -8.645, 3.892, 22.590, -8.101, -9.762, 13.657, -10.331, -5.612,
    -8.101, 9.762, -4.099, 0.000, 0.000, 6.928, -8.645, 4.500, 5.612, -0.691,
     3.892, 6.197, 10.331,
]


# ─────────────────────────────────────────────────────────────────────────────
# Schritt 1: Tests #3 und #4 neu laufen lassen (gleiche Parameter wie original)
# ─────────────────────────────────────────────────────────────────────────────

def _fetch(symbol: str, timeframe: str) -> pd.DataFrame:
    from src.data.mt5_connector import MT5Connector
    mt5 = MT5Connector(
        login=int(os.environ.get("MT5_LOGIN", "0")),
        password=os.environ.get("MT5_PASSWORD", ""),
        server=os.environ.get("MT5_SERVER", ""),
    )
    mt5.connect()
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    end   = datetime(2024, 1, 1, tzinfo=timezone.utc)
    df = mt5.get_ohlcv(symbol, timeframe, start, end)
    mt5.disconnect()
    df_reset = df.reset_index()
    df_reset = df_reset.rename(columns={df_reset.columns[0]: "timestamp"})
    return df_reset


def _validate(df: pd.DataFrame, symbol: str, tf: str) -> pd.DataFrame:
    from src.data.validator import DataValidator
    validator = DataValidator()
    _, clean = validator.validate(df, symbol=symbol, timeframe=tf)
    return clean


def run_wf_test3() -> list[dict]:
    """XAUUSD H4 Trendfolge: Feature + Label + WF."""
    from src.data.feature_builder import FeatureBuilder
    from src.models.label_builder import LabelBuilder
    from src.models.signal_model import SignalModel

    logger.info("=== Lade XAUUSD H4 (Test #3) ===")
    df = _validate(_fetch("XAUUSD", "H4"), "XAUUSD", "H4")

    builder = FeatureBuilder()
    features = builder.build(df, symbol="XAUUSD", timeframe="H4", df_h4=None, df_d1=None)

    labels = LabelBuilder().build_labels(features)  # tp=2.0, sl=1.5, max=24

    exclude = {"label", "timestamp", "open", "volume", "close", "high", "low"}
    feat_cols = [c for c in features.columns if c not in exclude]
    X = features[feat_cols + ["timestamp"]]

    results = SignalModel().walk_forward_validate(X, labels, timestamp_col="timestamp",
                                                  train_months=6, test_months=1)
    logger.info("Test #3: {n} Fenster berechnet", n=len(results))
    return results


def run_wf_test4() -> list[dict]:
    """EURUSD H4 Mean-Reversion: Feature + Label + WF."""
    from src.models.mean_reversion_model import MeanReversionModel
    from src.models.signal_model import SignalModel

    logger.info("=== Lade EURUSD H4 (Test #4) ===")
    df = _validate(_fetch("EURUSD", "H4"), "EURUSD", "H4")

    mr = MeanReversionModel()
    features = mr.build_features(df, symbol="EURUSD", timeframe="H4")

    labels = MeanReversionModel.default_label_builder().build_labels(features)

    exclude = {"label", "timestamp", "open", "volume", "close", "high", "low"}
    feat_cols = [c for c in features.columns if c not in exclude]
    X = features[feat_cols + ["timestamp"]]

    results = SignalModel().walk_forward_validate(X, labels, timestamp_col="timestamp",
                                                  train_months=6, test_months=1)
    logger.info("Test #4: {n} Fenster berechnet", n=len(results))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Schritt 2: Korrelationsanalyse
# ─────────────────────────────────────────────────────────────────────────────

def _wf_to_series(results: list[dict]) -> pd.Series:
    """Konvertiert WF-Ergebnisse in eine benannte pd.Series mit Monats-Index."""
    records = {}
    for r in results:
        ts = r.get("test_start")
        if ts is not None:
            month_key = str(ts)[:7]  # "YYYY-MM"
            records[month_key] = r["oos_sharpe"]
    return pd.Series(records, name="sharpe")


def _test2_to_series(sharpes: list[float], first_window_date: str = "2020-09") -> pd.Series:
    """
    USDJPY D1 Test #2: Schreibt Sharpe-Werte auf monatliche Keys.

    Das erste Test-Fenster beginnt ca. 2020-09 (min_date ≈ 2020-03 + 6M).
    Jeder Monat wird sequenziell hochgezaehlt.
    """
    start = pd.Period(first_window_date, freq="M")
    records = {}
    for i, s in enumerate(sharpes):
        key = str((start + i).to_timestamp())[:7]
        records[key] = s
    return pd.Series(records, name="sharpe")


def correlate(wf3: list[dict], wf4: list[dict]) -> dict:
    """
    Berechnet Pearson-Korrelationen zwischen den drei Kandidaten.

    Tests #3 und #4 haben gleiche Fenster (H4, gleiche Periode) → direkte Korrelation.
    Test #2 (D1, 32 Fenster) → monatlich ausrichten.
    """
    s3 = _wf_to_series(wf3)   # XAUUSD H4 TF (40 Fenster)
    s4 = _wf_to_series(wf4)   # EURUSD H4 MR (40 Fenster)
    s2 = _test2_to_series(TEST2_SHARPES)  # USDJPY D1 TF (32 Fenster)

    # Fenster-für-Fenster für Tests #3 und #4 (identisch ausgerichtet)
    sharpes3_arr = np.array([r["oos_sharpe"] for r in wf3])
    sharpes4_arr = np.array([r["oos_sharpe"] for r in wf4])
    assert len(sharpes3_arr) == len(sharpes4_arr), \
        f"Fenster-Mismatch: #3={len(sharpes3_arr)}, #4={len(sharpes4_arr)}"

    r34 = float(np.corrcoef(sharpes3_arr, sharpes4_arr)[0, 1])

    # Überlappende Monate für Test #2 vs. #3/#4
    common_23 = sorted(set(s2.index) & set(s3.index))
    common_24 = sorted(set(s2.index) & set(s4.index))
    common_23_all = sorted(set(s2.index) & set(s3.index) & set(s4.index))

    r23 = float(np.corrcoef(s2[common_23].values, s3[common_23].values)[0, 1]) \
          if len(common_23) >= 3 else float("nan")
    r24 = float(np.corrcoef(s2[common_24].values, s4[common_24].values)[0, 1]) \
          if len(common_24) >= 3 else float("nan")

    # Alle drei: auf gemeinsame Monate einschraenken
    if len(common_23_all) >= 3:
        arr2 = s2[common_23_all].values
        arr3 = s3[common_23_all].values
        arr4 = s4[common_23_all].values
        corr_matrix = np.corrcoef([arr2, arr3, arr4])
        r23_all, r24_all, r34_all = corr_matrix[0,1], corr_matrix[0,2], corr_matrix[1,2]
    else:
        r23_all, r24_all, r34_all = r23, r24, r34

    logger.info("Korrelation #2 D1 ↔ #3 H4 TF ({n} Monate): r={r:.3f}", n=len(common_23), r=r23)
    logger.info("Korrelation #2 D1 ↔ #4 H4 MR ({n} Monate): r={r:.3f}", n=len(common_24), r=r24)
    logger.info("Korrelation #3 H4 TF ↔ #4 H4 MR (alle 40 Fenster): r={r:.3f}", n=40, r=r34)

    # Simultane Gewinner/Verlierer-Analyse (beide positiv, beide negativ)
    pos_both_34 = int(((sharpes3_arr > 0) & (sharpes4_arr > 0)).sum())
    neg_both_34 = int(((sharpes3_arr < 0) & (sharpes4_arr < 0)).sum())
    diverge_34  = int(((sharpes3_arr > 0) != (sharpes4_arr > 0)).sum())

    logger.info(
        "Gleichzeitig positiv (#3 + #4): {a}/40 | Gleichzeitig negativ: {b}/40 | Divergent: {c}/40",
        a=pos_both_34, b=neg_both_34, c=diverge_34,
    )

    return {
        "r23": r23, "r24": r24, "r34": r34,
        "n23": len(common_23), "n24": len(common_24),
        "pos_both_34": pos_both_34,
        "neg_both_34": neg_both_34,
        "diverge_34":  diverge_34,
        "sharpes3": sharpes3_arr,
        "sharpes4": sharpes4_arr,
        "s2": s2, "s3": s3, "s4": s4,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Schritt 3: Portfolio-Kombinations-Simulation
# ─────────────────────────────────────────────────────────────────────────────

def simulate_equal_weight_portfolio(corr: dict) -> dict:
    """
    Simuliert ein gleichgewichtetes Portfolio aus Tests #3 und #4 (H4, gleiche Fenster).
    Zeigt ob Kombination besser/stabiler als jedes Einzelsystem.
    """
    s3 = corr["sharpes3"]
    s4 = corr["sharpes4"]
    combined = (s3 + s4) / 2.0

    def stats(arr: np.ndarray, label: str) -> dict:
        return {
            "label": label,
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "std": float(arr.std()),
            "profitable_pct": float((arr > 0).sum() / len(arr) * 100),
            "sharpe_of_means": float(arr.mean() / arr.std() * np.sqrt(12)) if arr.std() > 0 else 0.0,
        }

    st3 = stats(s3, "XAUUSD H4 TF")
    st4 = stats(s4, "EURUSD H4 MR")
    stc = stats(combined, "50/50 Portfolio (#3+#4)")

    logger.info("Portfolio-Simulation:")
    for st in [st3, st4, stc]:
        logger.info(
            "  {l}: Ø={m:.3f} | Median={med:.3f} | Std={s:.3f} | Prof={p:.0f}%",
            l=st["label"], m=st["mean"], med=st["median"],
            s=st["std"], p=st["profitable_pct"],
        )

    return {"test3": st3, "test4": st4, "combined": stc}


# ─────────────────────────────────────────────────────────────────────────────
# Schritt 4: Research-Log aktualisieren
# ─────────────────────────────────────────────────────────────────────────────

def _interpret_correlation(r: float) -> str:
    if abs(r) < 0.2:
        return "nahezu unkorreliert"
    elif abs(r) < 0.4:
        return "schwach korreliert"
    elif abs(r) < 0.6:
        return "moderat korreliert"
    else:
        return "stark korreliert"


def update_research_log(corr: dict, portfolio: dict) -> None:
    content = LOG_PATH.read_text(encoding="utf-8")

    # ── Zusammenfassung-Tabelle ──────────────────────────────────────────────
    rows_by_median = sorted(
        SUMMARY_DATA,
        key=lambda x: x[4] if x[5] is None else x[5],  # sort by median (fallback: mean)
        reverse=True,
    )

    table_lines = [
        "",
        "---",
        "",
        "## Gesamtauswertung: Alle 4 priorisierten Tests",
        "",
        f"*Stand: {TEST_DATE} – sortiert nach Median OOS-Sharpe (robust gegenüber Einzelausreißern)*",
        "",
        "| Rang | Test | Symbol | TF | Strategie | Ø OOS-Sharpe | Median OOS-Sharpe | Std OOS-Sharpe | Profitable Fenster | Trades | Urteil |",
        "|------|------|--------|----|-----------|-------------|-------------------|----------------|---------------------|--------|--------|",
    ]

    urteile = {1: "Verworfen", 2: "Kandidat", 3: "Kandidat", 4: "Kandidat"}
    for rang, (n, sym, tf, strat, mean, median, std, prof, total, trades) in enumerate(rows_by_median, 1):
        median_str = f"{median:.3f}" if median is not None else "n/a¹"
        table_lines.append(
            f"| {rang} | #{n} | {sym} | {tf} | {strat} | {mean:.3f} | {median_str} | "
            f"{std:.3f} | {prof}/{total} ({prof/total*100:.0f}%) | {trades:,} | {urteile[n]} |"
        )

    table_lines += [
        "",
        "*¹ Median wurde für Test #1 nicht separat erfasst (Einführung ab Test #2).*",
        "",
        "**Lesart:** Median OOS-Sharpe > 0 bedeutet: in mehr als 50% aller Monate war das Modell profitabel",
        "(der Median ist der 50%-Quantilswert des Sharpe-Verteilung). Er ist robuster als der Mittelwert,",
        "da er von extremen Einzelfenstern nicht verzerrt wird.",
    ]

    # ── Korrelationsanalyse ──────────────────────────────────────────────────
    r34_str = _interpret_correlation(corr["r34"])
    r23_str = _interpret_correlation(corr["r23"]) if not np.isnan(corr["r23"]) else "nicht berechnet"
    r24_str = _interpret_correlation(corr["r24"]) if not np.isnan(corr["r24"]) else "nicht berechnet"

    pc3 = portfolio["test3"]
    pc4 = portfolio["test4"]
    pcc = portfolio["combined"]

    corr_lines = [
        "",
        "---",
        "",
        "## Korrelationsanalyse der drei Kandidaten",
        "",
        "**Methode:** Pearson-Korrelation der fensterweisen OOS-Sharpe-Werte.",
        "Tests #3 und #4 (beide H4, gleiche Periode) sind fenstergenau ausgerichtet (40 Fenster identisch).",
        "Test #2 (D1, 32 Fenster) wird monatlich ausgerichtet auf den Überschneidungszeitraum.",
        "",
        "### Paarweise Korrelationen",
        "",
        "| Paar | Pearson r | Überlappende Fenster | Interpretation |",
        "|------|-----------|---------------------|----------------|",
        f"| USDJPY D1 TF ↔ XAUUSD H4 TF (#2 vs. #3) | {corr['r23']:.3f} | {corr['n23']} Monate | {r23_str} |",
        f"| USDJPY D1 TF ↔ EURUSD H4 MR (#2 vs. #4) | {corr['r24']:.3f} | {corr['n24']} Monate | {r24_str} |",
        f"| XAUUSD H4 TF ↔ EURUSD H4 MR (#3 vs. #4) | {corr['r34']:.3f} | 40 Fenster (exakt) | {r34_str} |",
        "",
        "### Gleichgerichtete Fenster (#3 und #4, n=40)",
        "",
        f"- Beide positiv (profitabler Monat fuer beide): **{corr['pos_both_34']}/40** ({corr['pos_both_34']/40*100:.0f}%)",
        f"- Beide negativ (Verlusmonat fuer beide): **{corr['neg_both_34']}/40** ({corr['neg_both_34']/40*100:.0f}%)",
        f"- Divergent (einer positiv, einer negativ): **{corr['diverge_34']}/40** ({corr['diverge_34']/40*100:.0f}%)",
        "",
        "### Portfolio-Simulation: 50/50 Kombination von #3 und #4",
        "",
        "| Metrik | XAUUSD H4 TF (#3) | EURUSD H4 MR (#4) | 50/50 Portfolio |",
        "|--------|------------------|------------------|-----------------|",
        f"| Ø OOS-Sharpe | {pc3['mean']:.3f} | {pc4['mean']:.3f} | {pcc['mean']:.3f} |",
        f"| Median OOS-Sharpe | {pc3['median']:.3f} | {pc4['median']:.3f} | {pcc['median']:.3f} |",
        f"| Std OOS-Sharpe | {pc3['std']:.3f} | {pc4['std']:.3f} | **{pcc['std']:.3f}** |",
        f"| Profitable Fenster | {pc3['profitable_pct']:.0f}% | {pc4['profitable_pct']:.0f}% | {pcc['profitable_pct']:.0f}% |",
        "",
    ]

    # Interpretation
    std_reduction = (1 - pcc["std"] / max(pc3["std"], pc4["std"])) * 100
    diversification_benefit = pcc["std"] < min(pc3["std"], pc4["std"])

    if corr["r34"] < 0.2:
        corr_text = (
            f"Die Korrelation zwischen XAUUSD H4 TF und EURUSD H4 MR beträgt r={corr['r34']:.3f} "
            f"({r34_str}). Das ist der theoretisch erwartete Effekt: Trendfolge und Mean-Reversion "
            f"arbeiten nach entgegengesetzten Marktregime-Annahmen. "
            f"In Trend-Monaten profitiert die TF-Strategie, während MR kämpft – und umgekehrt "
            f"in Seitwärtsphasen. Diese niedrige Korrelation macht eine Kombination prinzipiell attraktiv."
        )
    elif 0.2 <= corr["r34"] < 0.5:
        corr_text = (
            f"Die Korrelation zwischen XAUUSD H4 TF und EURUSD H4 MR beträgt r={corr['r34']:.3f} "
            f"({r34_str}). Trotz unterschiedlicher Strategie-Typen (TF vs. MR) teilen beide Systeme "
            f"gemeinsame Risikofaktoren (USD-Stärke, Volatilitätsregime), was die moderate Korrelation erklärt. "
            f"Eine Kombination bringt noch Diversifikationsnutzen, aber weniger als im unkorrellierten Fall."
        )
    else:
        corr_text = (
            f"Die Korrelation zwischen XAUUSD H4 TF und EURUSD H4 MR beträgt r={corr['r34']:.3f} "
            f"({r34_str}). Das ist überraschend hoch für unterschiedliche Strategie-Typen. "
            f"Mögliche Ursache: beide Assets reagieren stark auf USD-Regime-Wechsel, die dominante "
            f"Risikoquelle im Beobachtungszeitraum war. Eine Kombination bietet begrenzten Diversifikationsnutzen."
        )

    if diversification_benefit:
        portfolio_text = (
            f"Das 50/50 Portfolio hat eine Std von {pcc['std']:.3f} – deutlich unter dem Minimum "
            f"der Einzelsysteme ({min(pc3['std'], pc4['std']):.3f}). Echter Diversifikationsnutzen: "
            f"Risiko sinkt ohne entsprechenden Renditeverlust."
        )
    else:
        portfolio_text = (
            f"Das 50/50 Portfolio reduziert die Std auf {pcc['std']:.3f} (Einzelsysteme: "
            f"{pc3['std']:.3f} / {pc4['std']:.3f}). Die Std des Portfolios liegt "
            f"zwischen den Einzelwerten – teilweiser, aber kein vollständiger Diversifikationsnutzen."
        )

    corr_lines += [
        "### Interpretation",
        "",
        corr_text,
        "",
        portfolio_text,
        "",
        "**USDJPY D1 TF (Test #2) als dritte Komponente:** "
        f"Die Korrelation mit XAUUSD H4 TF beträgt r={corr['r23']:.3f} ({r23_str}), "
        f"mit EURUSD H4 MR r={corr['r24']:.3f} ({r24_str}). "
        f"Hinweis: Die monatliche Ausrichtung ist eine Näherung (D1- vs. H4-Fenster sind nicht identisch), "
        f"die Werte sind daher weniger präzise als die #3/#4-Korrelation.",
        "",
        "**Kombinations-Empfehlung:**",
        "Vor einer echten Portfolio-Kombination müssen zwei weitere Fragen geklärt werden:",
        "1. Transaktionskosten und Spread auf allen drei Instrumenten (besonders USDJPY D1: nur 695 Trades,",
        "   wenig Signal-Frequenz; XAUUSD und EURUSD H4 mit ~5200 Trades deutlich aktiver).",
        "2. Kapital-Effizienz: USDJPY D1 bindet Overnight-Margin-Risiko; H4-Systeme sind kürzer exponiert.",
        "Wenn beide Punkte akzeptabel: Kombination von #3 (XAUUSD TF) + #4 (EURUSD MR) als erster Schritt empfohlen,",
        "da deren Korrelation direkt messbar und die Signal-Frequenz vergleichbar ist.",
    ]

    new_section = "\n".join(table_lines + corr_lines) + "\n"
    content = content.rstrip() + "\n" + new_section.strip() + "\n"
    LOG_PATH.write_text(content, encoding="utf-8")
    logger.info("strategy_research_log.md – Gesamtauswertung eingetragen")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("====== Gesamtauswertung + Korrelationsanalyse ======")

    # WF fuer Tests #3 und #4
    wf3 = run_wf_test3()
    wf4 = run_wf_test4()

    # Korrelation
    corr = correlate(wf3, wf4)

    # Portfolio-Simulation
    portfolio = simulate_equal_weight_portfolio(corr)

    # Log-Update
    update_research_log(corr, portfolio)

    logger.info("====== Gesamtauswertung abgeschlossen ======")


if __name__ == "__main__":
    main()
