"""
scripts/build_mtf_features.py
Holt H1-, H4- und D1-OHLCV via MT5Connector und baut das Feature-Parquet
mit Multi-Timeframe-Kontext (h4_trend, d1_trend) fuer das SignalModel.

Beispiel:
    python scripts/build_mtf_features.py --symbol EURUSD --start 2020-01-01 --end 2024-01-01
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
from loguru import logger

from src.data.mt5_connector import MT5Connector
from src.data.validator import DataValidator
from src.data.feature_builder import FeatureBuilder
from src.data.pipeline import _ensure_timestamp_column


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def _connect_mt5() -> MT5Connector:
    load_dotenv()
    conn = MT5Connector(
        login=int(os.environ.get("MT5_LOGIN", "0")),
        password=os.environ.get("MT5_PASSWORD", ""),
        server=os.environ.get("MT5_SERVER", ""),
    )
    conn.connect()
    return conn


def _fetch_and_clean(
    conn: MT5Connector,
    validator: DataValidator,
    symbol: str,
    tf: str,
    start: datetime,
    end: datetime,
) -> "pd.DataFrame":
    import pandas as pd

    logger.info("Hole {symbol} {tf} {start} - {end}", symbol=symbol, tf=tf,
                start=start.date(), end=end.date())
    raw = conn.get_ohlcv(symbol, tf, start, end)
    raw = _ensure_timestamp_column(raw)
    logger.info("  -> {n} rohe Bars", n=len(raw))

    report, clean = validator.validate(raw, symbol=symbol, timeframe=tf)
    logger.info("  -> {n} saubere Bars, Qualitaet {q:.3f}",
                n=len(clean), q=report.quality_score)
    if report.quality_score < 0.90:
        logger.warning("  Qualitaet unter 0.90 – bitte pruefen!")
    return clean


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Baut Feature-Parquet mit Multi-Timeframe-Kontext (h4_trend, d1_trend)."
    )
    parser.add_argument("--symbol", default="EURUSD", help="Trading-Symbol")
    parser.add_argument("--start",  default="2020-01-01", help="Startdatum YYYY-MM-DD")
    parser.add_argument("--end",    default="2024-01-01", help="Enddatum YYYY-MM-DD")
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end   = datetime.strptime(args.end,   "%Y-%m-%d").replace(tzinfo=timezone.utc)

    conn      = _connect_mt5()
    validator = DataValidator()

    df_h1 = _fetch_and_clean(conn, validator, args.symbol, "H1", start, end)
    df_h4 = _fetch_and_clean(conn, validator, args.symbol, "H4", start, end)
    df_d1 = _fetch_and_clean(conn, validator, args.symbol, "D1", start, end)

    logger.info("Baue Features: H1={h1} Bars, H4={h4} Bars, D1={d1} Bars",
                h1=len(df_h1), h4=len(df_h4), d1=len(df_d1))

    feature_dir = Path(__file__).resolve().parents[1] / "data" / "features"
    builder = FeatureBuilder(feature_dir=feature_dir)
    features = builder.build(
        df_h1,
        symbol=args.symbol,
        timeframe="H1",
        save=True,
        df_h4=df_h4,
        df_d1=df_d1,
    )

    cols = [c for c in features.columns if c not in ("timestamp", "open")]
    logger.info("Feature-Matrix: {n} Zeilen, {f} Features: {cols}",
                n=len(features), f=len(cols), cols=cols)

    mtf_cols = [c for c in features.columns if c.endswith("_trend")]
    if mtf_cols:
        logger.info("MTF-Features vorhanden: {cols}", cols=mtf_cols)
        for col in mtf_cols:
            logger.info("  {col}: min={mn:.4f}  max={mx:.4f}  mean={mu:.4f}",
                        col=col,
                        mn=features[col].min(),
                        mx=features[col].max(),
                        mu=features[col].mean())
    else:
        logger.error("Keine MTF-Features im Output!")


if __name__ == "__main__":
    main()
