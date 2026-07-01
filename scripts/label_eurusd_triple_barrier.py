"""
scripts/label_eurusd_triple_barrier.py
Wendet das kostenbewusste Triple-Barrier-Labeling auf den EURUSD-M15-Dukascopy-
Datensatz (2016-2026) an, speichert den gelabelten Datensatz und gibt den
Diagnose-Report (Teil D) aus.

KEIN Modelltraining, KEINE Sharpe-Berechnung – nur Labeling + Diagnose.

Aufruf:
    python scripts/label_eurusd_triple_barrier.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.triple_barrier_labeler import (  # noqa: E402
    CostConfig, BarrierConfig, label_dataframe, DESIGNS,
)

DATA = Path("data/processed/EURUSD_M15_2016-2026.parquet")
COST_MODEL = Path("config/cost_model_EURUSD.yaml")
CONFIG = Path("config/config.yaml")
OUT = Path("data/processed/EURUSD_M15_2016-2026_labeled.parquet")

# Gemessener Pip-Wert EURUSD (order_calc_profit): 8.78 EUR je Pip / 1.0 Lot.
PIP_VALUE_EUR_PER_LOT = 8.78
SLIPPAGE_PER_SIDE_PIPS = 0.2   # konservativ, konfigurierbar


def load_costs() -> CostConfig:
    cm = yaml.safe_load(COST_MODEL.read_text(encoding="utf-8"))
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    factor = float(cm["recommended_robust_factor"])
    commission_rt = float(cm["commission"]["round_turn_pips"])

    swap = cfg["backtest"]["swap"]["EURUSD"]   # EUR/Nacht auf ~0.1-Lot-Notional
    pip_value_0p1 = PIP_VALUE_EUR_PER_LOT * 0.1   # EUR je Pip fuer 0.1 Lot
    swap_long_pips = float(swap["long"]) / pip_value_0p1
    swap_short_pips = float(swap["short"]) / pip_value_0p1

    logger.info("Kosten | Spread-Faktor {f} | Komm-RT {c} Pips | Swap long {sl:.3f} / short {ss:.3f} Pips",
                f=factor, c=commission_rt, sl=swap_long_pips, ss=swap_short_pips)
    return CostConfig(
        pip=0.0001,
        spread_factor=factor,
        commission_roundturn_pips=commission_rt,
        slippage_per_side_pips=SLIPPAGE_PER_SIDE_PIPS,
        swap_long_pips=round(swap_long_pips, 4),
        swap_short_pips=round(swap_short_pips, 4),
    )


def _dist(labeled: pd.DataFrame, design: str) -> dict:
    col = f"label_{design}"
    v = labeled[col].dropna()
    n = len(v)
    return {
        "n": n,
        "win_1": int((v == 1).sum()),
        "sl_-1": int((v == -1).sum()),
        "zero_0": int((v == 0).sum()),
        "win_pct": 100 * (v == 1).mean() if n else 0.0,
        "sl_pct": 100 * (v == -1).mean() if n else 0.0,
        "zero_pct": 100 * (v == 0).mean() if n else 0.0,
    }


def report(out: pd.DataFrame) -> None:
    total = len(out)
    st = out["status"].value_counts().to_dict()
    labeled = out[out["status"] == "labeled"]

    print("\n" + "=" * 72)
    print("TRIPLE-BARRIER DIAGNOSE-REPORT — EURUSD M15 (2016-2026)")
    print("=" * 72)
    print(f"Bars gesamt              : {total:,}")
    print(f"  no_trade (Rollover)    : {st.get('no_trade', 0):,}")
    print(f"  insufficient_atr       : {st.get('insufficient_atr', 0):,}")
    print(f"  insufficient_future    : {st.get('insufficient_future', 0):,}")
    print(f"  gap_skip (Luecke/WE)   : {st.get('gap_skip', 0):,}")
    print(f"  spike_skip (z>15)      : {st.get('spike_skip', 0):,}")
    print(f"  GELABELT               : {st.get('labeled', 0):,}")

    for design in DESIGNS:
        d = _dist(out, design)
        col = f"label_{design}"
        lab = labeled[labeled[col].notna()]
        # Schluesselzahl: brutto profitabel, aber netto NICHT (kein SL)
        non_sl = lab[lab[col] != -1]
        gross_pos = non_sl[f"gross_pips_{design}"] > 0
        net_nonpos = non_sl[f"net_pips_{design}"] <= 0
        eaten = int((gross_pos & net_nonpos).sum())
        gross_pos_n = int(gross_pos.sum())
        wins = lab[lab[col] == 1]
        avg_net_win = float(wins[f"net_pips_{design}"].mean()) if len(wins) else float("nan")

        print("\n" + "-" * 72)
        print(f"DESIGN: {design}")
        print("-" * 72)
        print(f"  Label=1  (netto profitabel): {d['win_1']:>7,}  ({d['win_pct']:5.1f} %)")
        print(f"  Label=-1 (SL zuerst)       : {d['sl_-1']:>7,}  ({d['sl_pct']:5.1f} %)")
        print(f"  Label=0  (Timeout/gekippt) : {d['zero_0']:>7,}  ({d['zero_pct']:5.1f} %)")
        print(f"  >> BRUTTO profitabel, aber NETTO nicht (Kosten fressen Edge):")
        print(f"     {eaten:,} von {gross_pos_n:,} brutto-positiven "
              f"({100*eaten/gross_pos_n:.1f} % der Brutto-Gewinner gekippt)")
        print(f"  Ø Netto-Move der Gewinner  : {avg_net_win:.2f} Pips")
        print(f"  Profitable Labels je Session:")
        sess = wins["session"].value_counts()
        for s in ["Asien", "Europa", "Overlap", "US", "Rollover"]:
            c = int(sess.get(s, 0))
            pct = 100 * c / len(wins) if len(wins) else 0.0
            print(f"     {s:9s}: {c:>6,}  ({pct:4.1f} %)")
    print("=" * 72)


def main() -> int:
    if not DATA.exists():
        logger.error("Fehlt: {p}", p=DATA)
        return 1
    df = pd.read_parquet(DATA)
    cost = load_costs()
    logger.info("Labeling startet | {n} Bars ...", n=len(df))
    out = label_dataframe(df, cost=cost, barrier=BarrierConfig())

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False, compression="snappy")
    logger.info("Gelabelter Datensatz gespeichert: {p}", p=OUT)

    report(out)
    print(f"\nGelabelter Datensatz: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
