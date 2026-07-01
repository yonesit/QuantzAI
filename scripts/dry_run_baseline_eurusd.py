"""
scripts/dry_run_baseline_eurusd.py
DRY-RUN (vorlaeufig!) der Baseline-Trainings-Pipeline auf EURUSD zur
Verifikation der Infrastruktur. NICHT die finale Baseline – die kommt mit
XAUUSD (beide Symbole, gleiches Fenster, Portfolio-Sharpe).

Erfolgsmass: echtes, kostenbereinigtes P&L-OOS-Sharpe ueber Purged-WF-Folds.

Aufruf:
    python scripts/dry_run_baseline_eurusd.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.baseline_trainer import build_features, run_design
from src.models.purged_walk_forward import PurgedWalkForward

LABELED = Path("data/processed/EURUSD_M15_2016-2026_labeled.parquet")


def _print_design(res, imp_top: int = 8) -> None:
    s = res.summary()
    print("\n" + "-" * 72)
    print(f"DESIGN: {res.design}")
    print("-" * 72)
    print(f"  P&L-OOS-Sharpe (Modell) : {s['sharpe_mean']:+.3f}  "
          f"(Streuung ueber Folds ±{s['sharpe_std']:.3f})")
    print(f"  Sharpe je Fold          : {s['sharpe_folds']}")
    print(f"  Benchmark 'immer long'  : {s['bench_long_sharpe_mean']:+.3f}")
    bench_rand = res.bench_random_sharpes
    if bench_rand:
        import numpy as np
        print(f"  Benchmark 'zufaellig'   : {np.mean(bench_rand):+.3f}")
    print(f"  Basisrate Label=1       : {s['base_rate_label1']:.3f}")
    print(f"  Trades gesamt           : {s['n_trades']:,}")
    print(f"  Profit-Factor           : {s['profit_factor']}")
    print(f"  Win-Rate                : {s['win_rate']:.3f}")
    print(f"  Max-Drawdown            : {s['max_drawdown']:.3f}")
    print(f"  Top-Features (gain):")
    for i, (name, val) in enumerate(res.importances.items()):
        if i >= imp_top:
            break
        print(f"     {name:16s} {val:12.1f}")


def main() -> int:
    if not LABELED.exists():
        logger.error("Fehlt: {p}", p=LABELED)
        return 1
    df = pd.read_parquet(LABELED)
    logger.info("Baue Features | {n} Bars ...", n=len(df))
    frame, names = build_features(df)
    logger.info("Features: {k} | {n} Bars", k=len(names), n=len(frame))

    wf = PurgedWalkForward(n_splits=5, label_horizon=16, embargo=16, mode="expanding")
    results = {}
    for design in ("symmetric", "asymmetric"):
        logger.info("Trainiere Design {d} ueber Purged-WF ...", d=design)
        results[design] = run_design(frame, names, design, k=1.5, sl_mult=1.0, wf=wf)

    print("\n" + "=" * 72)
    print("DRY-RUN BASELINE — EURUSD M15 (VORLAEUFIG, finale Baseline folgt mit XAUUSD)")
    print("=" * 72)
    print(f"Purged Walk-Forward: {wf.n_splits} Folds | Horizont {wf.label_horizon} "
          f"Bars | Embargo {wf.embargo} Bars | Modus {wf.mode}")
    print(f"Sizing: fixes Risiko 1% pro Trade am SL-Abstand (kein All-in)")
    for design in ("symmetric", "asymmetric"):
        _print_design(results[design])
    print("=" * 72)

    # Ehrliche Einordnung
    print("\nEINORDNUNG:")
    for design in ("symmetric", "asymmetric"):
        s = results[design].summary()
        edge = s["sharpe_mean"] - s["bench_long_sharpe_mean"]
        verdict = ("Edge ueber Benchmark" if s["sharpe_mean"] > 0 and edge > 0.1
                   else "KEIN klarer Edge")
        print(f"  {design:10s}: Sharpe {s['sharpe_mean']:+.3f} vs Bench "
              f"{s['bench_long_sharpe_mean']:+.3f} -> {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
