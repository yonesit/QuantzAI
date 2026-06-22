"""
scripts/download_gdelt_news.py
Laedt GDELT 2.0 GKG-Daten fuer einen Zeitraum und speichert sie als
lokale Parquet-Datenbank unter data/news/gdelt_EURUSD.parquet.

Verwendung:
  python scripts/download_gdelt_news.py
  python scripts/download_gdelt_news.py --start 2024-06-01 --end 2025-06-01
  python scripts/download_gdelt_news.py --start 2024-06-01 --end 2025-06-01 --step 60

Parameter:
  --start    Startdatum  (YYYY-MM-DD, UTC) Standard: 2024-06-01
  --end      Enddatum    (YYYY-MM-DD, UTC, exklusiv) Standard: 2025-06-01
  --symbol   Waehrungspaar (Standard: EURUSD)
  --step     Download-Intervall in Minuten (Vielfaches von 15).
             60 (Standard) = 1 Datei/Stunde = ~8.760 Dateien/Jahr
  --data-dir Ausgabeverzeichnis (Standard: data/news)

Resume: Bereits geladene Zeitstempel werden automatisch uebersprungen.
Einfach dasselbe Kommando erneut ausfuehren um einen abgebrochenen
Download fortzusetzen.

Hinweis: GDELT 2.0 GKG ist seit 2015-02 verfuegbar.
"""
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.gdelt_sentiment import GDELTDownloader


_DEFAULT_START = "2024-06-01"
_DEFAULT_END   = "2025-06-01"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GDELT GKG EUR/USD Sentiment Downloader (resume-faehig)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--start",    default=_DEFAULT_START, help="Startdatum YYYY-MM-DD (UTC)")
    parser.add_argument("--end",      default=_DEFAULT_END,   help="Enddatum YYYY-MM-DD (UTC, exklusiv)")
    parser.add_argument("--symbol",   default="EURUSD",       help="Waehrungspaar")
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
    est_hours_low  = total_files * 0.8 / 3600
    est_hours_high = total_files * 2.0 / 3600

    parquet_path = Path(args.data_dir).resolve() / f"gdelt_{args.symbol}.parquet"

    print("GDELT 2.0 Download")
    print(f"  Zeitraum:   {start.date()} bis {end.date()} ({days} Tage)")
    print(f"  Symbol:     {args.symbol}")
    print(f"  Intervall:  alle {step} Minuten → {files_per_day} Dateien/Tag")
    print(f"  Gesamt:     ~{total_files:,} Dateien")
    print(f"  Laufzeit:   ca. {est_hours_low:.1f}–{est_hours_high:.1f} Stunden")
    print(f"  Ausgabe:    {parquet_path}")
    if parquet_path.exists():
        print(f"  Resume:     bestehende DB gefunden – bereits geladene Buckets werden uebersprungen")
    print()

    downloader = GDELTDownloader(
        data_dir=args.data_dir,
        step_minutes=step,
    )

    df = downloader.download_range(start=start, end=end, symbol=args.symbol)

    print(f"\nFertig: {len(df):,} Buckets mit EUR/USD-relevantem Sentiment in der DB.")
    if not df.empty:
        print(f"  Zeitraum: {df['bucket_time'].min()} bis {df['bucket_time'].max()}")
        print(f"  Tone-Mittel: {df['avg_tone'].mean():.3f}")
        print(f"  Tone-Bereich: [{df['avg_tone'].min():.2f}, {df['avg_tone'].max():.2f}]")


if __name__ == "__main__":
    main()
