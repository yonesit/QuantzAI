"""
scripts/download_gdelt_news.py
Laedt GDELT 2.0 GKG-Daten fuer einen Zeitraum und speichert sie als
lokale Parquet-Datenbank unter data/news/gdelt_EURUSD.parquet.

Verwendung:
  python scripts/download_gdelt_news.py --start 2020-01-01 --end 2024-01-01
  python scripts/download_gdelt_news.py --start 2024-01-01 --end 2024-02-01 --step 30

Parameter:
  --start    Startdatum  (YYYY-MM-DD, UTC)
  --end      Enddatum    (YYYY-MM-DD, UTC, exklusiv)
  --symbol   Waehrungspaar (Standard: EURUSD)
  --step     Download-Intervall in Minuten – muss Vielfaches von 15 sein.
             60 (Standard) = 1 Datei/Stunde  →  ~7 GB fuer 4 Jahre
             120            = 1 Datei/2h     →  ~3.5 GB fuer 4 Jahre
  --data-dir Ausgabeverzeichnis (Standard: data/news)

Hinweis: GDELT 2.0 GKG ist seit 2015-02 verfuegbar.
Fuer 4 Jahre (2020-2024) mit step=60: ~35.000 Dateien, ca. 2-5 Stunden Laufzeit.
Die Dateien werden gefiltert (nur EUR/USD-relevante Artikel), die resultierende
Parquet-DB ist typischerweise < 100 MB.
"""
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

# Projekt-Root zum Python-Pfad hinzufuegen
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.gdelt_sentiment import GDELTDownloader


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GDELT GKG EUR/USD Sentiment Downloader",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--start",    required=True, help="Startdatum YYYY-MM-DD (UTC)")
    parser.add_argument("--end",      required=True, help="Enddatum YYYY-MM-DD (UTC, exklusiv)")
    parser.add_argument("--symbol",   default="EURUSD", help="Waehrungspaar")
    parser.add_argument("--step",     type=int, default=60,
                        help="Download-Intervall in Minuten (Vielfaches von 15)")
    parser.add_argument("--data-dir", default="data/news", dest="data_dir",
                        help="Ausgabeverzeichnis fuer die Parquet-DB")
    args = parser.parse_args()

    try:
        start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
        end   = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)
    except ValueError as exc:
        parser.error(f"Ungaeltiges Datum: {exc}")

    if start >= end:
        parser.error("--start muss vor --end liegen")

    days = (end - start).days
    step = max(15, (args.step // 15) * 15)
    files_per_day = 24 * 60 // step
    total_files = days * files_per_day
    print(f"GDELT Download")
    print(f"  Zeitraum:  {start.date()} bis {end.date()} ({days} Tage)")
    print(f"  Symbol:    {args.symbol}")
    print(f"  Intervall: alle {step} Minuten → ~{files_per_day} Dateien/Tag")
    print(f"  Geschaetzt: ~{total_files:,} Dateien")
    print(f"  Ausgabe:   {Path(args.data_dir).resolve()}/gdelt_{args.symbol}.parquet")
    print()

    downloader = GDELTDownloader(
        data_dir=args.data_dir,
        step_minutes=step,
    )

    df = downloader.download_range(start=start, end=end, symbol=args.symbol)

    print(f"\nFertig: {len(df):,} Buckets mit EUR/USD-relevantem Sentiment gespeichert.")
    if not df.empty:
        print(f"  Zeitraum: {df['bucket_time'].min()} bis {df['bucket_time'].max()}")
        print(f"  Tone-Mittel: {df['avg_tone'].mean():.3f}")
        print(f"  Tone-Bereich: [{df['avg_tone'].min():.2f}, {df['avg_tone'].max():.2f}]")


if __name__ == "__main__":
    main()
