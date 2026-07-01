"""
scripts/fetch_fusion_ticks_spread.py
Misst den REALEN effektiven Bid/Ask-Spread von Fusion Markets aus TICKS
(copy_ticks_range) fuer EURUSD. Das Bar-``spread``-Feld ist auf dem Raw-/Zero-
Konto zu 95 % Null und untauglich; die Ticks zeigen den echten momentanen
Spread inkl. der Nicht-Null-Phasen.

WICHTIGE DEMO-LIMITS (gemessen an Konto #383619):
  1. Die Tick-History reicht nur ~3 Monate zurueck (ab ~2026-04); aeltere
     Zeitraeume liefern 0 Ticks. Der Overlap 2022-2026 ist per Tick NICHT
     abdeckbar – gemessen wird das juengste verfuegbare Fenster.
  2. ~98 % der Ticks tragen bid==ask (flags 134) – ein Feed-Artefakt des
     Raw-Feeds, KEIN handelbarer Nullspread. Der reale, handelbare Spread
     liegt in den zweiseitigen Quotes (ask>bid). Deshalb wird der effektive
     Spread aus den zweiseitigen Quotes gemessen (Nullspread-Anteil wird
     separat berichtet).

Es wird das juengste zusammenhaengende Fenster wochenweise gezogen (Speicher
schonen). Ergebnis: per-Session Median/P90 des effektiven Spreads in Pips.

Aufruf:
    python scripts/fetch_fusion_ticks_spread.py --symbol EURUSD
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.spread_calibration import (  # noqa: E402
    PIP_SIZE, assign_sessions, SESSION_NAMES,
)

OUT_DIR = Path("data/processed/fusion_ref")

# Juengstes verfuegbares Tick-Fenster (Demo haelt nur ~3 Monate vor).
DEFAULT_WINDOW_START = "2026-04-06"
DEFAULT_WINDOW_END = "2026-06-29"


def _week_starts(start: str, end: str) -> list[datetime]:
    s = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    e = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    out = []
    while s < e:
        out.append(s)
        s = s + timedelta(days=7)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fusion Tick-Spread Sampler")
    p.add_argument("--symbol", default="EURUSD")
    p.add_argument("--start", default=DEFAULT_WINDOW_START)
    p.add_argument("--end", default=DEFAULT_WINDOW_END)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    symbol = args.symbol
    pip = PIP_SIZE[symbol]

    load_dotenv()
    import MetaTrader5 as mt5  # lokal, damit Tests ohne MT5 laufen
    ok = mt5.initialize(
        login=int(os.environ.get("MT5_LOGIN", "0")),
        password=os.environ.get("MT5_PASSWORD", ""),
        server=os.environ.get("MT5_SERVER", ""),
    )
    if not ok:
        logger.error("MT5 initialize fehlgeschlagen: {e}", e=mt5.last_error())
        return 1

    frames: list[pd.DataFrame] = []   # nur zweiseitige Quotes (ask>bid)
    n_total = 0
    n_zero = 0
    try:
        for start in _week_starts(args.start, args.end):
            end = start + timedelta(days=7)
            ticks = mt5.copy_ticks_range(symbol, start, end, mt5.COPY_TICKS_ALL)
            if ticks is None or len(ticks) == 0:
                logger.warning("Keine Ticks fuer Woche ab {w}", w=start.date())
                continue
            t = pd.DataFrame(ticks)
            t["timestamp"] = pd.to_datetime(t["time_msc"], unit="ms", utc=True)
            t = t[(t["bid"] > 0) & (t["ask"] > 0)]
            t["spread_pips"] = (t["ask"] - t["bid"]) / pip
            n_total += len(t)
            n_zero += int((t["spread_pips"] <= 0).sum())
            # Effektiver Spread = nur zweiseitige Quotes (ask>bid). bid==ask ist
            # Feed-Artefakt (kein handelbarer Nullspread), wird verworfen.
            valid = t[t["spread_pips"] > 0]
            frames.append(valid[["timestamp", "spread_pips"]])
            logger.info("Woche ab {w}: {n} Ticks, davon {v} zweiseitig | Median {m:.3f} Pips",
                        w=start.date(), n=len(t), v=len(valid),
                        m=float(valid["spread_pips"].median()) if len(valid) else float("nan"))
    finally:
        mt5.shutdown()

    if not frames or sum(len(f) for f in frames) == 0:
        logger.error("Keine gueltigen zweiseitigen Ticks geladen. Nichts gespeichert.")
        return 2

    allt = pd.concat(frames, ignore_index=True)
    allt["session"] = assign_sessions(allt["timestamp"])
    zero_share = n_zero / n_total if n_total else float("nan")

    print("\n" + "=" * 66)
    print(f"Effektiver Fusion-Spread aus Ticks | {symbol}")
    print(f"Fenster {args.start} .. {args.end} (juengstes verfuegbares)")
    print(f"Ticks gesamt {n_total:,} | Nullspread-Artefakt-Anteil {zero_share:.1%} "
          f"| zweiseitig gewertet {len(allt):,}")
    print("=" * 66)
    print(f"GESAMT   Median {allt['spread_pips'].median():.3f}  "
          f"P90 {allt['spread_pips'].quantile(0.90):.3f} Pips")
    print("-" * 66)
    for name in SESSION_NAMES:
        s = allt.loc[allt["session"] == name, "spread_pips"]
        if len(s):
            print(f"{name:9s} n={len(s):>9,}  Median {float(s.median()):6.3f}  "
                  f"P90 {float(s.quantile(0.90)):7.3f} Pips")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sample = allt.iloc[::20].reset_index(drop=True)  # jeder 20. Tick
    path = OUT_DIR / f"{symbol}_tick_spread_sample.parquet"
    sample.to_parquet(path, index=False, compression="snappy")
    print("-" * 66)
    print(f"Sample gespeichert ({len(sample):,} Ticks): {path}")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
