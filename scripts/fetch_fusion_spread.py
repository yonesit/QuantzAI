"""
scripts/fetch_fusion_spread.py
Zieht EURUSD M15 mit der MT5-``spread``-Spalte (Spread in POINTS je Bar) von
Fusion Markets und speichert den Datensatz MIT Spread separat – der alte
OHLC-only fusion_ref bleibt als Backup erhalten (wird NICHT ueberschrieben).

Aufruf:
    python scripts/fetch_fusion_spread.py --symbol EURUSD

WICHTIG (Einheiten): MT5 liefert Spread in Points. EURUSD ist 5-stellig,
also 10 Points = 1 Pip. Die Umrechnung erfolgt hier explizit dokumentiert.
Plausibilitaets-Stop: liegt der Median-Spread nicht grob in [0.1, 1.5] Pips,
wird abgebrochen (dann stimmt Points/Pip-Annahme oder Kontoart nicht).
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.mt5_connector import MT5Connector  # noqa: E402
from src.data.spread_calibration import mt5_points_to_pips  # noqa: E402

OUT_DIR = Path("data/processed/fusion_ref")

# Overlap-Fenster mit dem Dukascopy-Datensatz; Fusion haelt real ~4 Jahre vor.
DEFAULT_START = datetime(2022, 1, 1, tzinfo=timezone.utc)
DEFAULT_END = datetime.now(timezone.utc)

PLAUSIBLE_PIPS = (0.1, 1.5)  # erwarteter Median-Bereich fuer Fusion EURUSD


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fusion M15 Pull inkl. Spread-Spalte")
    p.add_argument("--symbol", default="EURUSD")
    p.add_argument("--start", default=None, help="YYYY-MM-DD (UTC)")
    p.add_argument("--end", default=None, help="YYYY-MM-DD (UTC)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    symbol = args.symbol
    start = (datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
             if args.start else DEFAULT_START)
    end = (datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
           if args.end else DEFAULT_END)

    load_dotenv()
    conn = MT5Connector(
        login=int(os.environ.get("MT5_LOGIN", "0")),
        password=os.environ.get("MT5_PASSWORD", ""),
        server=os.environ.get("MT5_SERVER", ""),
    )
    conn.connect()
    try:
        df = conn.get_ohlcv(symbol, "M15", start, end, include_spread=True)
    finally:
        conn.disconnect()

    if "spread" not in df.columns:
        logger.error("MT5 lieferte keine spread-Spalte fuer {sym}. Abbruch.", sym=symbol)
        return 1

    df = df.reset_index().rename(columns={"index": "timestamp"})
    if "timestamp" not in df.columns:
        df = df.rename(columns={df.columns[0]: "timestamp"})

    # Points -> Pips (dokumentiert: 10 Points = 1 Pip fuer EURUSD 5-stellig)
    df["spread_pips"] = mt5_points_to_pips(df["spread"].to_numpy(), symbol)

    med = float(df["spread_pips"].median())
    logger.info("Fusion {sym} | {n} Bars | {s} .. {e} | Median-Spread {m:.3f} Pips",
                sym=symbol, n=len(df),
                s=df['timestamp'].min().date(), e=df['timestamp'].max().date(), m=med)

    lo, hi = PLAUSIBLE_PIPS
    if not (lo <= med <= hi):
        logger.error(
            "PLAUSIBILITAETS-STOP: Median-Spread {m:.3f} Pips liegt ausserhalb "
            "[{lo}, {hi}] – Points/Pip-Umrechnung oder Kontoart pruefen. Nichts gespeichert.",
            m=med, lo=lo, hi=hi)
        return 2

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    y0, y1 = df["timestamp"].min().year, df["timestamp"].max().year
    path = OUT_DIR / f"{symbol}_M15_{y0}-{y1}_fusion_ref_spread.parquet"
    df.to_parquet(path, index=False, compression="snappy")
    logger.info("Gespeichert (mit Spread) | {p} | {n} Bars", p=path, n=len(df))

    print("\n" + "=" * 60)
    print(f"Symbol            : {symbol}")
    print(f"Bars              : {len(df):,}")
    print(f"Zeitraum          : {df['timestamp'].min()} .. {df['timestamp'].max()}")
    print(f"Median-Spread     : {med:.3f} Pips")
    print(f"Mittel-Spread     : {df['spread_pips'].mean():.3f} Pips")
    print(f"Min / Max         : {df['spread_pips'].min():.3f} / {df['spread_pips'].max():.3f} Pips")
    print(f"Datei             : {path}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
