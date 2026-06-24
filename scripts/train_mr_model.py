"""
scripts/train_mr_model.py
MeanReversionModel-Training fuer beliebiges Symbol und Timeframe.

Standard-Parameter (Test #4):
  --symbol EURUSD --tf H4 --start 2020-01-01 --end 2024-01-01

Fuer M15-Training:
  python scripts/train_mr_model.py --symbol EURUSD --tf M15 --start 2022-01-01

Ausgabe: models/mean_reversion_model_v1_<TF>_<DATUM>.joblib
         (H4 behält altes Format ohne TF-Suffix fuer Rueckwaertskompatibilitaet)
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
from loguru import logger

load_dotenv()


def _build_out_path(symbol: str, timeframe: str) -> Path:
    tf_part = f"_{timeframe}" if timeframe.upper() != "H4" else ""
    fname = f"mean_reversion_model_v1{tf_part}_{date.today().strftime('%Y%m%d')}.joblib"
    return Path("models") / fname


def main() -> None:
    parser = argparse.ArgumentParser(description="MeanReversionModel Training")
    parser.add_argument("--symbol",  default="EURUSD", help="Symbol (Standard: EURUSD)")
    parser.add_argument("--tf",      default="H4",     help="Timeframe (Standard: H4)")
    parser.add_argument("--start",   default="2020-01-01", help="Startdatum YYYY-MM-DD")
    parser.add_argument("--end",     default=None,     help="Enddatum YYYY-MM-DD (Standard: heute)")
    args = parser.parse_args()

    symbol    = args.symbol.upper()
    timeframe = args.tf.upper()
    start     = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end       = (
        datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        if args.end else datetime.now(timezone.utc)
    )
    out_path  = _build_out_path(symbol, timeframe)

    logger.info("=== MR-Modell Training: {s} {tf} | {st} – {e} ===",
                s=symbol, tf=timeframe, st=start.date(), e=end.date())

    # 1. Daten holen
    from src.data.mt5_connector import MT5Connector
    mt5 = MT5Connector(
        login=int(os.environ["MT5_LOGIN"]),
        password=os.environ["MT5_PASSWORD"],
        server=os.environ["MT5_SERVER"],
    )
    mt5.connect()
    raw_df = mt5.get_ohlcv(symbol, timeframe, start, end)
    mt5.disconnect()
    logger.info("Daten: {n} Bars | {s} – {e}",
                n=len(raw_df),
                s=raw_df.index[0].date(),
                e=raw_df.index[-1].date())

    # 2. Validieren
    from src.data.validator import DataValidator
    df_reset = raw_df.reset_index().rename(columns={raw_df.index.name or "index": "timestamp"})
    report, clean_df = DataValidator().validate(df_reset, symbol=symbol, timeframe=timeframe)
    if not report.is_usable:
        raise RuntimeError(f"Datenqualitaet ungenuegend: {report.errors}")
    logger.info("Validator OK | quality={q:.3f} | {c} Bars", q=report.quality_score, c=report.total_candles)

    # 3. MR-Features (26 = Standard-23 + bb_pct_b + dist_ema20_atr + dist_sma50_atr)
    from src.models.mean_reversion_model import MeanReversionModel
    mr_model = MeanReversionModel()
    features_df = mr_model.build_features(clean_df, symbol=symbol, timeframe=timeframe)

    feat_cols = [c for c in features_df.columns if c not in {"timestamp", "close", "high", "low"}]
    logger.info("Features: {n} Spalten | {r} Zeilen", n=len(feat_cols), r=len(features_df))

    # 4. Labels
    label_builder = MeanReversionModel.default_label_builder()
    labels = label_builder.build_labels(features_df)

    dist  = {v: int((labels == v).sum()) for v in [-1, 0, 1]}
    total = len(labels)
    logger.info(
        "Labels: Short={s} ({sp:.1f}%) Neutral={n} ({np:.1f}%) Long={l} ({lp:.1f}%)",
        s=dist[-1], sp=dist[-1]/total*100,
        n=dist[0],  np=dist[0]/total*100,
        l=dist[1],  lp=dist[1]/total*100,
    )

    # 5. Training auf Gesamtdatensatz
    logger.info("Training auf {n} Samples ...", n=total)
    metrics = mr_model.train(features_df[feat_cols], labels)
    logger.info("Training abgeschlossen | {m}", m=metrics)

    # 6. Speichern
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mr_model.save(out_path)
    logger.info("Modell gespeichert: {p}", p=out_path.resolve())


if __name__ == "__main__":
    main()
