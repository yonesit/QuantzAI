"""
scripts/fetch_dukascopy.py
CLI: laedt maximale Dukascopy-M15-Historie (mit Spread) fuer ein Symbol,
validiert sie und speichert sie als Parquet unter data/processed/.

Resume-faehig: bricht der Lauf ab, einfach erneut starten – bereits geladene
Tage liegen im Cache und werden uebersprungen.

Beispiele
---------
  python scripts/fetch_dukascopy.py --symbol EURUSD
  python scripts/fetch_dukascopy.py --symbol XAUUSD --start 2010-01-04
  python scripts/fetch_dukascopy.py --symbol EURUSD --report-only
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from loguru import logger

# Projekt-Root in den Pfad
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dukascopy_downloader import (  # noqa: E402
    DukascopyDownloader,
    SYMBOL_HISTORY_START,
    SYMBOL_POINT_DIVISOR,
)
from src.data.dukascopy_validator import validate_dukascopy_m15  # noqa: E402

PROCESSED_DIR = Path("data/processed")
CACHE_DIR     = PROCESSED_DIR / "dukascopy_cache"
REPORT_DIR    = PROCESSED_DIR / "quality_reports"


def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dukascopy M15 Downloader (mit Spread)")
    p.add_argument("--symbol", required=True, choices=sorted(SYMBOL_POINT_DIVISOR.keys()))
    p.add_argument("--start", type=_parse_date, default=None,
                   help="Startdatum YYYY-MM-DD (Standard: frueheste Historie des Symbols)")
    p.add_argument("--end", type=_parse_date, default=None,
                   help="Enddatum YYYY-MM-DD (Standard: heute)")
    p.add_argument("--rate-limit", type=float, default=0.25,
                   help="Pause zwischen HTTP-Requests in Sekunden")
    p.add_argument("--report-only", action="store_true",
                   help="Nur aus vorhandenem Cache zusammensetzen + validieren")
    return p.parse_args()


def save_parquet(df, symbol: str) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    y0 = df["timestamp"].min().year
    y1 = df["timestamp"].max().year
    path = PROCESSED_DIR / f"{symbol}_M15_{y0}-{y1}.parquet"
    df.to_parquet(path, index=False, compression="snappy")
    logger.info("Parquet gespeichert | {path} | {n} Bars", path=path, n=len(df))
    return path


def main() -> int:
    args = parse_args()
    symbol = args.symbol
    start = args.start or SYMBOL_HISTORY_START.get(symbol, datetime(2010, 1, 1, tzinfo=timezone.utc))
    end = args.end or datetime.now(timezone.utc)

    dl = DukascopyDownloader(cache_dir=CACHE_DIR, rate_limit_s=args.rate_limit)

    if args.report_only:
        logger.info("Report-only: setze {sym} aus Cache zusammen", sym=symbol)
        df = dl.assemble_from_cache(symbol)
    else:
        logger.info("Starte Dukascopy-Download {sym} | {s} .. {e}",
                    sym=symbol, s=start.date(), e=end.date())
        df, stats = dl.download_range(symbol, start, end)
        logger.info("Download-Stats {sym}: {st}", sym=symbol, st=stats)

    if df.empty:
        logger.error("Keine Daten fuer {sym} – Abbruch.", sym=symbol)
        return 1

    # Output exakt auf das angeforderte Fenster [start, end] klemmen. Der Tages-Cache
    # behaelt bewusst ALLE bereits geladenen Tage (z.B. Vollhistorie ab 2003/2010),
    # damit Phase 5 die laengere Historie spaeter per Resume nachziehen kann.
    before = len(df)
    df = df[(df["timestamp"] >= pd.Timestamp(start)) &
            (df["timestamp"] <= pd.Timestamp(end))].reset_index(drop=True)
    if len(df) != before:
        logger.info("Output auf Fenster geklemmt | {sym} | {a} -> {b} Bars | {s} .. {e}",
                    sym=symbol, a=before, b=len(df), s=start.date(), e=end.date())
    if df.empty:
        logger.error("Nach Klemmung keine Daten im Fenster fuer {sym}.", sym=symbol)
        return 1

    parquet_path = save_parquet(df, symbol)
    report = validate_dukascopy_m15(df, symbol)
    report.save(REPORT_DIR)

    print("\n" + "=" * 70)
    print(report.to_markdown())
    print(f"Parquet: {parquet_path}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
