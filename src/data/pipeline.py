"""
src/data/pipeline.py
DataPipeline – orchestriert Fetch -> Validate -> Build Features -> Speichern.

Einziger Einstiegspunkt fuer alle Datenprozesse (Batch + Live).
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

from src.data.data_router import DataRouter
from src.data.validator import DataValidator, DataQualityError


# ─────────────────────────────────────────────
#  Exceptions
# ─────────────────────────────────────────────

class PipelineError(Exception):
    """Allgemeiner Pipeline-Fehler."""


# ─────────────────────────────────────────────
#  DataPipeline
# ─────────────────────────────────────────────

class DataPipeline:
    """
    Orchestriert den vollstaendigen Datenfluss:
    DataRouter -> DataValidator -> FeatureBuilder -> Speichern.

    Parameters
    ----------
    router          : DataRouter-Instanz (waehlt MT5/OANDA automatisch)
    validator       : DataValidator-Instanz
    feature_builder : FeatureBuilder-Instanz
    features_dir    : Zielordner fuer Parquet-Dateien
    reports_dir     : Zielordner fuer Qualitaetsberichte
    live_interval   : Sekunden zwischen Live-Updates (Standard: 300 = 5 Min)
    """

    def __init__(
        self,
        router: DataRouter,
        validator: DataValidator,
        feature_builder,
        features_dir: str = "data/features",
        reports_dir: str = "data/processed/quality_reports",
        live_interval: int = 300,
    ) -> None:
        self._router          = router
        self._validator        = validator
        self._feature_builder  = feature_builder
        self._features_dir     = Path(features_dir)
        self._reports_dir      = Path(reports_dir)
        self._live_interval    = live_interval

        self._features_dir.mkdir(parents=True, exist_ok=True)
        self._reports_dir.mkdir(parents=True, exist_ok=True)

        self._live_thread: Optional[threading.Thread] = None
        self._stop_live = threading.Event()

    # ── Batch-Modus ──────────────────────────────────

    def run_batch(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        force_refetch: bool = False,
        progress_callback=None,
    ) -> dict:
        """
        Fuehrt einen vollstaendigen Batch-Lauf durch:
        Fetch -> Validate -> Build Features -> Speichern.

        Parameters
        ----------
        symbol         : z.B. "EURUSD"
        timeframe      : z.B. "H1"
        start, end     : Zeitraum
        force_refetch  : True = ignoriert Hash-Check und holt Daten neu
        progress_callback : optionale Funktion fuer Fortschrittsanzeige (z.B. tqdm.update)

        Returns
        -------
        dict mit: output_path, report_path, quality_report, feature_count
        """
        output_path = self._feature_path(symbol, timeframe, end)

        if not force_refetch and self._is_already_processed(output_path, start, end):
            logger.info(
                "Pipeline: Daten bereits vorhanden (Hash match) | {symbol} {tf} -> skip",
                symbol=symbol, tf=timeframe,
            )
            return {
                "output_path": str(output_path),
                "skipped": True,
            }

        logger.info(
            "Pipeline START | symbol={symbol} tf={tf} | {start} - {end}",
            symbol=symbol, tf=timeframe, start=start, end=end,
        )

        # 1. Fetch
        raw_df = self._router.get_ohlcv(symbol, timeframe, start, end)
        if progress_callback:
            progress_callback(1)

        # 2. Validate
        try:
            clean_df, report = self._validator.validate(raw_df, symbol=symbol, timeframe=timeframe)
        except DataQualityError as exc:
            logger.error("Pipeline gestoppt | DataQualityError: {exc}", exc=exc)
            raise PipelineError(f"Datenqualitaet ungenuegend fuer {symbol} {timeframe}: {exc}") from exc

        if progress_callback:
            progress_callback(1)

        report_path = self._save_quality_report(report, symbol, timeframe)

        # 3. Build Features
        features_df = self._feature_builder.build(clean_df)
        if progress_callback:
            progress_callback(1)

        # 4. Speichern
        features_df.to_parquet(output_path, engine="pyarrow")
        self._write_hash(output_path, start, end)

        if progress_callback:
            progress_callback(1)

        logger.info(
            "Pipeline FERTIG | {symbol} {tf} | {n} Features-Zeilen -> {path}",
            symbol=symbol, tf=timeframe, n=len(features_df), path=output_path,
        )

        return {
            "output_path":   str(output_path),
            "report_path":   str(report_path),
            "quality_report": report,
            "feature_count": len(features_df),
            "skipped": False,
        }

    def run_batch_multi(
        self,
        symbols: list[str],
        timeframe: str,
        start: datetime,
        end: datetime,
        force_refetch: bool = False,
        show_progress: bool = True,
    ) -> dict[str, dict]:
        """Fuehrt run_batch() fuer mehrere Symbole aus."""
        results = {}

        iterator = symbols
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(symbols, desc="Symbole")
            except ImportError:
                logger.warning("tqdm nicht installiert – Fortschrittsanzeige deaktiviert.")

        for symbol in iterator:
            try:
                results[symbol] = self.run_batch(
                    symbol, timeframe, start, end, force_refetch=force_refetch
                )
            except PipelineError as exc:
                logger.error("Pipeline-Fehler fuer {symbol}: {exc}", symbol=symbol, exc=exc)
                results[symbol] = {"error": str(exc)}

        return results

    # ── Live-Modus ───────────────────────────────────

    def start_live(self, symbol: str, timeframe: str, lookback_candles: int = 300) -> None:
        """
        Startet den Live-Modus als Hintergrund-Thread.
        Holt alle `live_interval` Sekunden neue Candles und aktualisiert Features.
        """
        self._stop_live.clear()
        self._live_thread = threading.Thread(
            target=self._live_loop,
            args=(symbol, timeframe, lookback_candles),
            daemon=True,
            name="data-pipeline-live",
        )
        self._live_thread.start()
        logger.info(
            "Live-Modus gestartet | {symbol} {tf} | interval={interval}s",
            symbol=symbol, tf=timeframe, interval=self._live_interval,
        )

    def stop_live(self) -> None:
        """Stoppt den Live-Modus."""
        self._stop_live.set()
        if self._live_thread:
            self._live_thread.join(timeout=5)
        logger.info("Live-Modus gestoppt.")

    def _live_loop(self, symbol: str, timeframe: str, lookback_candles: int) -> None:
        while not self._stop_live.wait(timeout=self._live_interval):
            try:
                self._live_update(symbol, timeframe, lookback_candles)
            except Exception as exc:  # noqa: BLE001
                logger.error("Live-Update Fehler | {symbol} | {exc}", symbol=symbol, exc=exc)

    def _live_update(self, symbol: str, timeframe: str, lookback_candles: int) -> dict:
        """Ein einzelner Live-Update-Zyklus. Oeffentlich testbar."""
        raw_df = self._router.get_ohlcv_count(symbol, timeframe, count=lookback_candles)

        clean_df, report = self._validator.validate(raw_df, symbol=symbol, timeframe=timeframe)
        self._save_quality_report(report, symbol, timeframe)

        features_df = self._feature_builder.build(clean_df)

        output_path = self._feature_path(symbol, timeframe, datetime.now(timezone.utc))
        features_df.to_parquet(output_path, engine="pyarrow")

        logger.info(
            "Live-Update | {symbol} {tf} | {n} Zeilen",
            symbol=symbol, tf=timeframe, n=len(features_df),
        )
        return {"output_path": str(output_path), "feature_count": len(features_df)}

    # ── Intern: Pfade & Idempotenz ───────────────────

    def _feature_path(self, symbol: str, timeframe: str, ref_date: datetime) -> Path:
        date_str = ref_date.strftime("%Y%m%d")
        filename = f"{symbol}_{timeframe}_{date_str}.parquet"
        return self._features_dir / filename

    def _hash_path(self, output_path: Path) -> Path:
        return output_path.with_suffix(".hash.json")

    def _compute_hash(self, start: datetime, end: datetime) -> str:
        raw = f"{start.isoformat()}|{end.isoformat()}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _is_already_processed(self, output_path: Path, start: datetime, end: datetime) -> bool:
        hash_path = self._hash_path(output_path)
        if not output_path.exists() or not hash_path.exists():
            return False
        try:
            with open(hash_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("hash") == self._compute_hash(start, end)
        except (OSError, json.JSONDecodeError):
            return False

    def _write_hash(self, output_path: Path, start: datetime, end: datetime) -> None:
        hash_path = self._hash_path(output_path)
        with open(hash_path, "w", encoding="utf-8") as f:
            json.dump({
                "hash": self._compute_hash(start, end),
                "start": start.isoformat(),
                "end": end.isoformat(),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }, f, indent=2)

    def _save_quality_report(self, report, symbol: str, timeframe: str) -> Path:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"{symbol}_{timeframe}_{timestamp}.json"
        path = self._reports_dir / filename

        report_dict = asdict(report) if hasattr(report, "__dataclass_fields__") else dict(report)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=2, default=str)

        return path
