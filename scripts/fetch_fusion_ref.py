"""
scripts/fetch_fusion_ref.py
Speichert die maximale bei Fusion Markets / MT5 verfuegbare M15-Historie
(~4 Jahre, durch das Terminal-Limit von 100.000 Bars/Chart begrenzt) als
separaten BROKER-ABGLEICHSDATENSATZ.

WICHTIG: Dieser Datensatz ist NICHT fuer Training/Labeling gedacht (zu kurz),
sondern als Referenz fuer Phase 6 (Demo-Validierung): reale Fusion-Fills /
Bar-Preise gegen Backtest-Annahmen abgleichen. Er wird bewusst getrennt von
den langen Dukascopy-Datensaetzen unter data/processed/fusion_ref/ abgelegt
und nie ueberschrieben.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.mt5_connector import MT5Connector  # noqa: E402

OUT_DIR = Path("data/processed/fusion_ref")
SYMBOLS = ["EURUSD", "XAUUSD"]
TIMEFRAME = "M15"
MAX_BARS = 99_000   # unter dem Terminal-Limit von 100.000


def main() -> int:
    load_dotenv()
    conn = MT5Connector(
        login=int(os.environ.get("MT5_LOGIN", "0")),
        password=os.environ.get("MT5_PASSWORD", ""),
        server=os.environ.get("MT5_SERVER", ""),
    )
    conn.connect()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for sym in SYMBOLS:
        df = conn.get_ohlcv_count(sym, TIMEFRAME, count=MAX_BARS)
        df = df.reset_index().rename(columns={"index": "timestamp"})
        if "timestamp" not in df.columns:
            df = df.rename(columns={df.columns[0]: "timestamp"})
        y0, y1 = df["timestamp"].min().year, df["timestamp"].max().year
        path = OUT_DIR / f"{sym}_M15_{y0}-{y1}_fusion_ref.parquet"
        df.to_parquet(path, index=False, compression="snappy")
        logger.info("Fusion-Ref gespeichert | {p} | {n} Bars | {s} .. {e}",
                    p=path, n=len(df),
                    s=df['timestamp'].min().date(), e=df['timestamp'].max().date())

    conn.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
