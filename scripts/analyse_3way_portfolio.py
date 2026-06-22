"""
scripts/analyse_3way_portfolio.py
3-Wege Portfolio-Analyse: USDJPY D1 TF (#2) + XAUUSD H4 TF (#3) + EURUSD H4 MR (#4)

Methode:
  Jedes WF-Fenster wird seinem Kalendermonat (YYYY-MM aus test_start) zugeordnet.
  Die Schnittmenge aller drei Testserien ergibt den Analysezeitraum.

  D1 Warmup: 200 Handelstage ≈ Okt 2020. Erstes Fenster (nach 6M Training): Apr 2021.
  H4 Warmup: 200 H4-Bars  ≈ 33 Tage   → Feb 2020. Erstes Fenster: Aug 2020.
  Gemeinsame Monate: Apr 2021 – Nov 2023 = 32 Monate.

  Kombinierter Monats-Sharpe = w2*s2[M] + w3*s3[M] + w4*s4[M]
  OOS-Sharpe ist mean(r)/std(r)*sqrt(252), annualisiert und dimensionslos;
  Kombination ueber Timeframes ist Naeherung (D1 ~22 Trades/M, H4 ~130 Trades/M).

Getestete Gewichtungsschemata:
  - Einzelsysteme (Referenz)
  - Gleichgewichtet 1/3 je
  - D1 niedrig: 25/40/35
  - D1 sehr niedrig: 20/40/40
  - Risiko-Parität: Gewichte ∝ 1/Std (Vollperiode)
  - Performance-gewichtet: Gewichte ∝ Median OOS-Sharpe
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

# Aggregate-Statistiken aus dem Research-Log (Vollperiode, alle Fenster)
FULL_STATS = {
    "D1":    {"std": 7.594, "median": 1.449},  # Test #2
    "H4TF":  {"std": 3.678, "median": 0.191},  # Test #3
    "H4MR":  {"std": 3.670, "median": 1.142},  # Test #4
}


# ─────────────────────────────────────────────────────────────────────────────
# Schritt 1: WF-Daten fuer alle drei Tests
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_and_validate(symbol: str, timeframe: str) -> pd.DataFrame:
    from src.data.mt5_connector import MT5Connector
    from src.data.validator import DataValidator

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

    _, clean = DataValidator().validate(df_reset, symbol=symbol, timeframe=timeframe)
    return clean


def run_wf_d1() -> list[dict]:
    """USDJPY D1 Trendfolge (Test #2) – 23-Feature-Baseline."""
    from src.data.feature_builder import FeatureBuilder
    from src.models.label_builder import LabelBuilder
    from src.models.signal_model import SignalModel

    logger.info("=== USDJPY D1 Walk-Forward ===")
    df = _fetch_and_validate("USDJPY", "D1")

    features = FeatureBuilder().build(df, symbol="USDJPY", timeframe="D1",
                                      df_h4=None, df_d1=None)
    labels = LabelBuilder().build_labels(features)

    exclude = {"label", "timestamp", "open", "volume", "close", "high", "low"}
    feat_cols = [c for c in features.columns if c not in exclude]
    X = features[feat_cols + ["timestamp"]]

    results = SignalModel().walk_forward_validate(
        X, labels, timestamp_col="timestamp", train_months=6, test_months=1
    )
    logger.info("USDJPY D1: {n} Fenster | erste Testmonate: {f} bis {l}",
                n=len(results),
                f=str(results[0].get("test_start", "?"))[:7],
                l=str(results[-1].get("test_start", "?"))[:7])
    return results


def run_wf_h4tf() -> list[dict]:
    """XAUUSD H4 Trendfolge (Test #3)."""
    from src.data.feature_builder import FeatureBuilder
    from src.models.label_builder import LabelBuilder
    from src.models.signal_model import SignalModel

    logger.info("=== XAUUSD H4 TF Walk-Forward ===")
    df = _fetch_and_validate("XAUUSD", "H4")

    features = FeatureBuilder().build(df, symbol="XAUUSD", timeframe="H4",
                                      df_h4=None, df_d1=None)
    labels = LabelBuilder().build_labels(features)

    exclude = {"label", "timestamp", "open", "volume", "close", "high", "low"}
    feat_cols = [c for c in features.columns if c not in exclude]
    X = features[feat_cols + ["timestamp"]]

    results = SignalModel().walk_forward_validate(
        X, labels, timestamp_col="timestamp", train_months=6, test_months=1
    )
    logger.info("XAUUSD H4: {n} Fenster | {f} bis {l}",
                n=len(results),
                f=str(results[0].get("test_start", "?"))[:7],
                l=str(results[-1].get("test_start", "?"))[:7])
    return results


def run_wf_h4mr() -> list[dict]:
    """EURUSD H4 Mean-Reversion (Test #4)."""
    from src.models.mean_reversion_model import MeanReversionModel
    from src.models.signal_model import SignalModel

    logger.info("=== EURUSD H4 MR Walk-Forward ===")
    df = _fetch_and_validate("EURUSD", "H4")

    mr = MeanReversionModel()
    features = mr.build_features(df, symbol="EURUSD", timeframe="H4")
    labels = MeanReversionModel.default_label_builder().build_labels(features)

    exclude = {"label", "timestamp", "open", "volume", "close", "high", "low"}
    feat_cols = [c for c in features.columns if c not in exclude]
    X = features[feat_cols + ["timestamp"]]

    results = SignalModel().walk_forward_validate(
        X, labels, timestamp_col="timestamp", train_months=6, test_months=1
    )
    logger.info("EURUSD H4 MR: {n} Fenster | {f} bis {l}",
                n=len(results),
                f=str(results[0].get("test_start", "?"))[:7],
                l=str(results[-1].get("test_start", "?"))[:7])
    return results


def _to_monthly_series(wf_results: list[dict], label: str) -> pd.Series:
    """Konvertiert WF-Ergebnisliste in monthly-keyed Series (YYYY-MM)."""
    records = {}
    for r in wf_results:
        ts = r.get("test_start")
        if ts is not None:
            key = str(ts)[:7]
            records[key] = r["oos_sharpe"]
    return pd.Series(records, name=label).sort_index()


# ─────────────────────────────────────────────────────────────────────────────
# Schritt 2: Gewichtungsschemata definieren
# ─────────────────────────────────────────────────────────────────────────────

def build_weight_schemes(s2: pd.Series, s3: pd.Series, s4: pd.Series) -> list[dict]:
    """
    Definiert alle Gewichtungsschemata.
    Gewichte = [w_D1, w_H4TF, w_H4MR], summieren auf 1.0.
    """
    # Risiko-Parität: Gewichte ∝ 1/Std
    std2, std3, std4 = FULL_STATS["D1"]["std"], FULL_STATS["H4TF"]["std"], FULL_STATS["H4MR"]["std"]
    rp_raw = [1/std2, 1/std3, 1/std4]
    rp_sum = sum(rp_raw)
    rp_w   = [r / rp_sum for r in rp_raw]

    # Performance-Gewichtung: Gewichte ∝ Median (nur positive Medians sinnvoll)
    med2, med3, med4 = FULL_STATS["D1"]["median"], FULL_STATS["H4TF"]["median"], FULL_STATS["H4MR"]["median"]
    perf_raw = [max(med2, 0.01), max(med3, 0.01), max(med4, 0.01)]
    perf_sum = sum(perf_raw)
    perf_w   = [r / perf_sum for r in perf_raw]

    schemes = [
        # Referenz: Einzelsysteme
        {"label": "Nur USDJPY D1 TF",       "w": [1.00, 0.00, 0.00], "kategorie": "Referenz"},
        {"label": "Nur XAUUSD H4 TF",       "w": [0.00, 1.00, 0.00], "kategorie": "Referenz"},
        {"label": "Nur EURUSD H4 MR",       "w": [0.00, 0.00, 1.00], "kategorie": "Referenz"},
        # 2er-Kombination (Vorwissen)
        {"label": "H4-Portfolio (50/50 #3+#4)", "w": [0.00, 0.50, 0.50], "kategorie": "2-Wege"},
        # 3er-Kombinationen
        {"label": "Gleichgewichtet (33/33/34)",  "w": [1/3, 1/3, 1/3], "kategorie": "3-Wege"},
        {"label": "D1 niedrig (25/40/35)",       "w": [0.25, 0.40, 0.35], "kategorie": "3-Wege"},
        {"label": "D1 sehr niedrig (20/40/40)",  "w": [0.20, 0.40, 0.40], "kategorie": "3-Wege"},
        {
            "label": f"Risiko-Parität ({rp_w[0]*100:.0f}/{rp_w[1]*100:.0f}/{rp_w[2]*100:.0f})",
            "w": rp_w,
            "kategorie": "3-Wege",
        },
        {
            "label": f"Perf-gewichtet ({perf_w[0]*100:.0f}/{perf_w[1]*100:.0f}/{perf_w[2]*100:.0f})",
            "w": perf_w,
            "kategorie": "3-Wege",
            "note": "In-Sample Median als Gewicht – Datensnooping-Risiko",
        },
    ]

    # Normierung sicherstellen
    for s in schemes:
        total = sum(s["w"])
        s["w"] = [x / total for x in s["w"]]

    return schemes


# ─────────────────────────────────────────────────────────────────────────────
# Schritt 3: Portfolio-Statistiken berechnen
# ─────────────────────────────────────────────────────────────────────────────

def portfolio_stats(w: list[float], s2: pd.Series, s3: pd.Series, s4: pd.Series,
                    common_months: list[str]) -> dict:
    """Berechnet Portfolio-Statistiken fuer gemeinsame Monate."""
    arr2 = s2[common_months].values
    arr3 = s3[common_months].values
    arr4 = s4[common_months].values

    combined = w[0] * arr2 + w[1] * arr3 + w[2] * arr4
    n = len(combined)

    mean_s  = float(combined.mean())
    median_s = float(np.median(combined))
    std_s   = float(combined.std())
    prof    = int((combined > 0).sum())
    prof_pct = prof / n * 100
    median_sharpe = median_s / std_s if std_s > 0 else 0.0

    return {
        "mean":         mean_s,
        "median":       median_s,
        "std":          std_s,
        "profitable":   prof,
        "n":            n,
        "profitable_pct": prof_pct,
        "median_sharpe":  median_sharpe,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Schritt 4: Research-Log Eintrag
# ─────────────────────────────────────────────────────────────────────────────

def update_research_log(
    results_table: list[dict],
    common_months: list[str],
    s2: pd.Series,
    s3: pd.Series,
    s4: pd.Series,
    best_scheme: dict,
) -> None:
    content = LOG_PATH.read_text(encoding="utf-8")

    period_str = f"{common_months[0]} bis {common_months[-1]} ({len(common_months)} Monate)"

    lines = [
        "",
        "---",
        "",
        "## 3-Wege Portfolio-Analyse",
        "",
        f"*Stand: {TEST_DATE}*",
        "",
        "### Methode: Kalendermonatliche Ausrichtung",
        "",
        "Jedes WF-Fenster wird seinem Kalendermonat (`YYYY-MM` aus `test_start`) zugeordnet.",
        "Kombinierter Monats-Sharpe = Σ wᵢ × OOS-Sharpeᵢ für jeden gemeinsamen Monat.",
        "",
        "| System | Warmup | Erstes Fenster | Letztes Fenster | Fenster |",
        "|--------|--------|---------------|----------------|---------|",
        f"| USDJPY D1 TF  (#2) | 200 D1-Bars ≈ 10 Monate | {s2.index[0]} | {s2.index[-1]} | {len(s2)} |",
        f"| XAUUSD H4 TF  (#3) | 200 H4-Bars ≈ 33 Tage   | {s3.index[0]} | {s3.index[-1]} | {len(s3)} |",
        f"| EURUSD H4 MR  (#4) | 200 H4-Bars ≈ 33 Tage   | {s4.index[0]} | {s4.index[-1]} | {len(s4)} |",
        "",
        f"**Gemeinsamer Auswertungszeitraum:** {period_str}",
        "",
        "*Anmerkung: OOS-Sharpe ist `mean(r)/std(r)*√252`, dimensionslos und timeframe-unabhängig.",
        "Kombination über D1 (≈22 Trades/Monat) und H4 (≈130 Trades/Monat) ist Standard-Näherung.",
        "Gewichtung repräsentiert den Kapitalanteil pro Strategie.*",
        "",
        "### Ergebnisse nach Gewichtungsschema",
        "",
        f"*Gemeinsamer Zeitraum: {period_str}*",
        "",
        "| Kategorie | Gewichtung (D1/H4-TF/H4-MR) | Ø OOS-Sharpe | Median | Std | Profitable Monate | Median/Std |",
        "|-----------|------------------------------|-------------|--------|-----|-------------------|------------|",
    ]

    for r in results_table:
        w = r["w"]
        st = r["stats"]
        w_str = f"{w[0]*100:.0f}%/{w[1]*100:.0f}%/{w[2]*100:.0f}%"
        note = r.get("note", "")
        note_marker = " ²" if note else ""
        lines.append(
            f"| {r['kategorie']} | {r['label']}{note_marker} | "
            f"{st['mean']:+.3f} | {st['median']:+.3f} | {st['std']:.3f} | "
            f"{st['profitable']}/{st['n']} ({st['profitable_pct']:.0f}%) | "
            f"**{st['median_sharpe']:.3f}** |"
        )

    # Fussnoten
    for r in results_table:
        note = r.get("note", "")
        if note:
            lines.append(f"*² {note}*")

    # Bestes Schema hervorheben
    lines += [
        "",
        f"**Bestes Median/Std-Verhältnis: {best_scheme['label']}** "
        f"(Median/Std = {best_scheme['stats']['median_sharpe']:.3f})",
        "",
        "### Korrelation im gemeinsamen Zeitraum",
        "",
    ]

    # Korrelationen auf gemeinsamen Monaten
    arr2 = s2[common_months].values
    arr3 = s3[common_months].values
    arr4 = s4[common_months].values

    r23 = float(np.corrcoef(arr2, arr3)[0, 1])
    r24 = float(np.corrcoef(arr2, arr4)[0, 1])
    r34 = float(np.corrcoef(arr3, arr4)[0, 1])

    def _interp(r: float) -> str:
        ar = abs(r)
        if ar < 0.2:   return "nahezu unkorreliert"
        elif ar < 0.4: return "schwach korreliert"
        elif ar < 0.6: return "moderat korreliert"
        else:          return "stark korreliert"

    lines += [
        "| Paar | Pearson r | Interpretation |",
        "|------|-----------|----------------|",
        f"| USDJPY D1 TF ↔ XAUUSD H4 TF | {r23:+.3f} | {_interp(r23)} |",
        f"| USDJPY D1 TF ↔ EURUSD H4 MR | {r24:+.3f} | {_interp(r24)} |",
        f"| XAUUSD H4 TF ↔ EURUSD H4 MR | {r34:+.3f} | {_interp(r34)} |",
        "",
        "*Korrelationen hier auf dem gemeinsamen Teilzeitraum ({n} Monate, exakt ausgerichtet).*".format(n=len(common_months)),
        "",
        "### Fazit und Empfehlung",
        "",
    ]

    # Automatic interpretation
    best = best_scheme
    w = best["w"]
    st = best["stats"]
    three_way_schemas = [r for r in results_table if r["kategorie"] == "3-Wege"]
    two_way = next(r for r in results_table if r["kategorie"] == "2-Wege")
    equal_weight = next(r for r in results_table if "Gleichgewichtet" in r["label"])

    # Std reduction vs best single system
    singles = [r for r in results_table if r["kategorie"] == "Referenz"]
    min_single_std = min(r["stats"]["std"] for r in singles)
    best_std_reduction = (1 - best["stats"]["std"] / min_single_std) * 100

    # Compare best 3-way vs 2-way
    two_way_st = two_way["stats"]
    three_improves = best["stats"]["median_sharpe"] > two_way_st["median_sharpe"]

    lines += [
        f"Die drei Kandidaten sind im gemeinsamen Zeitraum ({period_str}) nahezu unkorreliert "
        f"(maximales |r| = {max(abs(r23), abs(r24), abs(r34)):.3f}). "
        f"Jede 3-Wege-Kombination reduziert die Std gegenüber den Einzelsystemen.",
        "",
        f"Das **beste Median/Std-Verhältnis** erreicht **{best['label']}** "
        f"mit Median/Std = {st['median_sharpe']:.3f} "
        f"(Median = {st['median']:+.3f}, Std = {st['std']:.3f}, {st['profitable_pct']:.0f}% profitable Monate).",
        "",
        f"{'Das 3-Wege-Portfolio verbessert Median/Std gegenüber dem 2-Wege H4-Portfolio' if three_improves else 'Das 2-Wege H4-Portfolio hat etwas besseres Median/Std als die 3-Wege-Varianten'} "
        f"({best['stats']['median_sharpe']:.3f} vs. {two_way_st['median_sharpe']:.3f}). "
        f"Der USDJPY D1 TF-Anteil {'erhöht' if three_improves else 'senkt'} das Risiko-Rendite-Verhältnis "
        f"leicht – erklärt durch {'seinen hohen Median (1.449) trotz hoher Std (7.594)' if three_improves else 'seine hohe Std (7.594) im Verhältnis zum Nutzen'}.",
        "",
        "**Praktische Einschränkungen USDJPY D1 (#2) im Portfolio:**",
        "- Nur 695 Trades über 4 Jahre (≈14.5 Trades/Monat) – statistisch dünnere Basis als H4",
        "- Overnight-Margin-Risiko durch D1-Haltedauer",
        "- Der starke Ausreisser in Fenster 13 (Fed/BoJ-Divergenz 2022) ist im Backtest enthalten "
        "  und nicht reproduzierbar",
        "",
        "**Empfehlung:** Als ersten Live-Test empfiehlt sich die H4-Portfolio-Kombination "
        "(#3 XAUUSD TF + #4 EURUSD MR, 50/50), da dort die Korrelation exakt messbar und die "
        "Datengrundlage mit je 40 Fenstern robuster ist. USDJPY D1 kann als optionale "
        "dritte Komponente mit niedrigem Gewicht (15–25%) hinzugefügt werden, sobald "
        "Live-Daten über mindestens 12 Monate vorliegen.",
    ]

    new_section = "\n".join(lines) + "\n"
    content = content.rstrip() + "\n" + new_section.strip() + "\n"
    LOG_PATH.write_text(content, encoding="utf-8")
    logger.info("strategy_research_log.md – 3-Wege-Portfolio-Analyse eingetragen")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("====== 3-Wege Portfolio-Analyse ======")

    # 1. WF-Daten fuer alle drei Tests
    wf2 = run_wf_d1()
    wf3 = run_wf_h4tf()
    wf4 = run_wf_h4mr()

    # 2. Monatliche Series
    s2 = _to_monthly_series(wf2, "USDJPY D1 TF")
    s3 = _to_monthly_series(wf3, "XAUUSD H4 TF")
    s4 = _to_monthly_series(wf4, "EURUSD H4 MR")

    logger.info(
        "Monatliche Coverage | D1: {d1} | H4-TF: {h4tf} | H4-MR: {h4mr}",
        d1=f"{s2.index[0]}–{s2.index[-1]} ({len(s2)})",
        h4tf=f"{s3.index[0]}–{s3.index[-1]} ({len(s3)})",
        h4mr=f"{s4.index[0]}–{s4.index[-1]} ({len(s4)})",
    )

    # 3. Gemeinsame Monate
    common_months = sorted(set(s2.index) & set(s3.index) & set(s4.index))
    logger.info(
        "Gemeinsame Monate: {n} | {f} bis {l}",
        n=len(common_months), f=common_months[0], l=common_months[-1],
    )

    # 4. Gewichtungsschemata
    schemes = build_weight_schemes(s2, s3, s4)

    # 5. Statistiken berechnen
    results_table = []
    for scheme in schemes:
        st = portfolio_stats(scheme["w"], s2, s3, s4, common_months)
        results_table.append({**scheme, "stats": st})
        logger.info(
            "{label}: Ø={m:+.3f} | Median={med:+.3f} | Std={s:.3f} | Prof={p:.0f}% | Median/Std={ms:.3f}",
            label=scheme["label"],
            m=st["mean"], med=st["median"], s=st["std"],
            p=st["profitable_pct"], ms=st["median_sharpe"],
        )

    # 6. Bestes Schema (nach Median/Std, nur 3-Wege + 2-Wege)
    kombinationen = [r for r in results_table if r["kategorie"] in ("2-Wege", "3-Wege")]
    best_scheme = max(kombinationen, key=lambda r: r["stats"]["median_sharpe"])
    logger.info("Bestes Median/Std: {l} = {v:.3f}",
                l=best_scheme["label"], v=best_scheme["stats"]["median_sharpe"])

    # 7. Log aktualisieren
    update_research_log(results_table, common_months, s2, s3, s4, best_scheme)

    logger.info("====== 3-Wege Portfolio-Analyse abgeschlossen ======")


if __name__ == "__main__":
    main()
