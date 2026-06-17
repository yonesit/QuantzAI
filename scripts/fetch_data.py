"""
scripts/fetch_data.py
CLI fuer die DataPipeline.

Beispiele
---------
python scripts/fetch_data.py --symbol EURUSD --tf H1 --start 2020-01-01 --end 2024-12-31
python scripts/fetch_data.py --symbols EURUSD,GBPUSD,USDJPY --tf H1 --start 2022-01-01
python scripts/fetch_data.py --symbol EURUSD --tf H1 --report-only
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

# Projekt-Root in den Pfad aufnehmen, damit "src.*"-Importe funktionieren
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loguru import logger

from src.data.mt5_connector import MT5Connector
from src.data.oanda_connector import OANDAConnector
from src.data.data_router import DataRouter, PriceValidator
from src.data.validator import DataValidator
from src.data.feature_builder import FeatureBuilder
from src.data.pipeline import DataPipeline


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QuantzAI Datenpipeline – Fetch, Validate, Build Features."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--symbol", type=str, help="Einzelnes Symbol, z.B. EURUSD")
    group.add_argument("--symbols", type=str, help="Mehrere Symbole, kommagetrennt")

    parser.add_argument("--tf", type=str, default="H1", help="Timeframe, z.B. H1, M15, D1")
    parser.add_argument("--start", type=str, help="Startdatum YYYY-MM-DD")
    parser.add_argument("--end", type=str, help="Enddatum YYYY-MM-DD (Standard: heute)")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Nur Qualitaetsbericht fuer bereits vorhandene Daten erzeugen",
    )
    parser.add_argument(
        "--force-refetch",
        action="store_true",
        help="Ignoriert Hash-Check und holt Daten neu",
    )
    return parser.parse_args()


def _build_pipeline() -> DataPipeline:
    """Baut alle Komponenten anhand der .env / config.yaml-Werte."""
    import os
    from dotenv import load_dotenv
    load_dotenv()

    mt5 = MT5Connector(
        login=int(os.environ.get("MT5_LOGIN", "0")),
        password=os.environ.get("MT5_PASSWORD", ""),
        server=os.environ.get("MT5_SERVER", ""),
    )
    oanda = OANDAConnector(
        api_key=os.environ.get("OANDA_API_KEY", ""),
        account_id=os.environ.get("OANDA_ACCOUNT_ID", ""),
        demo=os.environ.get("OANDA_DEMO", "true").lower() == "true",
    )

    try:
        mt5.connect()
    except Exception as exc:  # noqa: BLE001
        logger.warning("MT5 nicht erreichbar beim Start: {exc}", exc=exc)

    try:
        oanda.connect()
    except Exception as exc:  # noqa: BLE001
        logger.warning("OANDA nicht erreichbar beim Start: {exc}", exc=exc)

    router    = DataRouter(mt5, oanda, validator=PriceValidator())
    validator = DataValidator()
    builder   = FeatureBuilder()

    return DataPipeline(router, validator, builder)


def main() -> None:
    args = _parse_args()

    symbols = [args.symbol] if args.symbol else [s.strip() for s in args.symbols.split(",")]

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc) if args.start else None
    end   = (
        datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        if args.end else datetime.now(timezone.utc)
    )

    pipeline = _build_pipeline()

    if args.report_only:
        logger.info("Report-Only Modus – es werden keine neuen Daten geholt.")
        # In diesem Modus koennte z.B. der letzte Report pro Symbol angezeigt werden.
        return

    if start is None:
        logger.error("--start ist erforderlich (ausser bei --report-only).")
        sys.exit(1)

    if len(symbols) == 1:
        result = pipeline.run_batch(symbols[0], args.tf, start, end, force_refetch=args.force_refetch)
        logger.info("Ergebnis: {result}", result=result)
    else:
        results = pipeline.run_batch_multi(symbols, args.tf, start, end, force_refetch=args.force_refetch)
        for symbol, result in results.items():
            logger.info("{symbol}: {result}", symbol=symbol, result=result)


if __name__ == "__main__":
    main()
