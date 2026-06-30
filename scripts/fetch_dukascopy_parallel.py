"""
scripts/fetch_dukascopy_parallel.py
Parallel-Variante von fetch_dukascopy.py: teilt das Zeitfenster in N disjunkte
Datums-Slices auf und fuellt den Tages-Cache mit mehreren gleichzeitigen Workern
(je eigene HTTP-Session). Anschliessend EINMAL assemblieren -> Parquet + Report.

Sinn: Dukascopy drosselt teils heftig pro Verbindung. Mehrere gleichzeitige
Verbindungen koennen den Gesamtdurchsatz vervielfachen (sofern nicht hart pro IP
gedrosselt wird). Der Tages-Cache verhindert Doppel-Downloads – Slices sind
disjunkt, jeder Tag wird von genau einem Worker bearbeitet.

Beispiel:
  python scripts/fetch_dukascopy_parallel.py --symbol EURUSD \
      --start 2016-01-01 --end 2026-06-29 --workers 8
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from loguru import logger

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
    p = argparse.ArgumentParser(description="Dukascopy M15 Parallel-Downloader")
    p.add_argument("--symbol", required=True, choices=sorted(SYMBOL_POINT_DIVISOR.keys()))
    p.add_argument("--start", type=_parse_date, default=None)
    p.add_argument("--end", type=_parse_date, default=None)
    p.add_argument("--workers", type=int, default=8, help="Anzahl paralleler Worker")
    p.add_argument("--rate-limit", type=float, default=0.3,
                   help="Pause zwischen Requests je Worker (Sekunden)")
    return p.parse_args()


def split_range(start: datetime, end: datetime, n: int) -> list[tuple[datetime, datetime]]:
    """Teilt [start, end] in n moeglichst gleich grosse, disjunkte Tages-Slices."""
    total_days = (end.date() - start.date()).days + 1
    n = max(1, min(n, total_days))
    chunk = -(-total_days // n)  # ceil
    slices: list[tuple[datetime, datetime]] = []
    s = start
    while s <= end:
        e = min(s + timedelta(days=chunk - 1), end)
        slices.append((s, e))
        s = e + timedelta(days=1)
    return slices


def _worker(symbol: str, s: datetime, e: datetime, rate_limit: float):
    """Ein Worker mit EIGENER Session fuellt den Cache fuer seinen Slice."""
    dl = DukascopyDownloader(cache_dir=CACHE_DIR, rate_limit_s=rate_limit)
    stats = dl.fill_cache_range(symbol, s, e, progress=True)
    logger.info("Worker fertig | {sym} | {a} .. {b} | geladen={f} cached={c} leer={mt} failed={x}",
                sym=symbol, a=s.date(), b=e.date(),
                f=stats.days_fetched, c=stats.days_cached, mt=stats.days_empty, x=stats.days_failed)
    return stats


def save_parquet(df: pd.DataFrame, symbol: str) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    y0, y1 = df["timestamp"].min().year, df["timestamp"].max().year
    path = PROCESSED_DIR / f"{symbol}_M15_{y0}-{y1}.parquet"
    df.to_parquet(path, index=False, compression="snappy")
    logger.info("Parquet gespeichert | {p} | {n} Bars", p=path, n=len(df))
    return path


def main() -> int:
    args = parse_args()
    symbol = args.symbol
    start = args.start or SYMBOL_HISTORY_START.get(symbol, datetime(2010, 1, 1, tzinfo=timezone.utc))
    end = args.end or datetime.now(timezone.utc)

    slices = split_range(start, end, args.workers)
    logger.info("Parallel-Download {sym} | {s} .. {e} | {n} Worker / {k} Slices",
                sym=symbol, s=start.date(), e=end.date(), n=args.workers, k=len(slices))

    failed_total = 0
    with ThreadPoolExecutor(max_workers=len(slices)) as pool:
        futures = {pool.submit(_worker, symbol, s, e, args.rate_limit): (s, e) for s, e in slices}
        for fut in as_completed(futures):
            s, e = futures[fut]
            try:
                st = fut.result()
                failed_total += st.days_failed
            except Exception as exc:  # noqa: BLE001
                logger.error("Slice {s}..{e} abgebrochen: {x}", s=s.date(), e=e.date(), x=exc)

    if failed_total:
        logger.warning("{n} Tage fehlgeschlagen (Drosselung) – erneut starten zieht sie per Resume nach.",
                       n=failed_total)

    # Einmal assemblieren + auf Fenster klemmen
    dl = DukascopyDownloader(cache_dir=CACHE_DIR)
    df = dl.assemble_from_cache(symbol)
    df = df[(df["timestamp"] >= pd.Timestamp(start)) &
            (df["timestamp"] <= pd.Timestamp(end))].reset_index(drop=True)
    if df.empty:
        logger.error("Keine Daten im Fenster fuer {sym}.", sym=symbol)
        return 1

    parquet_path = save_parquet(df, symbol)
    report = validate_dukascopy_m15(df, symbol)
    report.save(REPORT_DIR)

    print("\n" + "=" * 70)
    print(report.to_markdown())
    print(f"Parquet: {parquet_path}")
    print(f"Fehlgeschlagene Tage (Resume noetig): {failed_total}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
