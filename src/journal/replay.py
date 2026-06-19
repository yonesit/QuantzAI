"""
src/journal/replay.py
TradeReplay – rekonstruiert den Chart und Kontext zum Entry-Zeitpunkt eines Trades.

Alle Daten stammen aus:
  - TradeJournal-DB:   Trade-Metadaten (Entry/Exit, Symbol, Signal-Konfidenz, ...)
  - Parquet-Dateien:   historische OHLCV + Features der DataPipeline

Kein Live-Abruf von MT5/OANDA → historisch exakt reproduzierbar,
auch wenn der Broker keine Daten mehr fuer den Zeitraum liefert.

No-Lookahead-Garantie:
  get_replay_data() gibt ausschliesslich Candles und Features zurueck,
  deren Zeitstempel <= Entry-Zeitpunkt des Trades liegt.
  Das Flag meta['no_lookahead'] ist immer True.

Rueckgabeformat (GUI-freundlich):
  {
    "trade":        dict             – vollstaendiger Trade-Eintrag aus dem Journal
    "candles":      list[dict]       – OHLCV-Punkte mit 'time', 'open', ... 'volume'
    "entry_marker": dict             – {time, price, direction}
    "exit_marker":  dict | None      – {time, price} oder None wenn Trade noch offen
    "indicators":   dict[str, list]  – {Spaltenname: [Werte parallel zu candles]}
    "news_events":  list             – geparste News aus trade['news_context']
    "meta":         dict             – symbol, timeframe, lookback_candles, no_lookahead, ...
  }

Parquet-Namenskonvention (DataPipeline):
  {features_dir}/{symbol}_{timeframe}_{YYYYMMDD}.parquet
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

from src.journal.trade_journal import TradeJournal


_OHLCV_COLS = frozenset({"open", "high", "low", "close", "volume"})


# ─────────────────────────────────────────────────────────────────────────────
#  Exceptions
# ─────────────────────────────────────────────────────────────────────────────

class TradeNotFoundError(Exception):
    """Trade-ID existiert nicht im Journal."""


class ReplayDataNotFoundError(Exception):
    """Keine Parquet-Daten fuer den angegebenen Trade / Zeitraum gefunden."""


# ─────────────────────────────────────────────────────────────────────────────
#  TradeReplay
# ─────────────────────────────────────────────────────────────────────────────

class TradeReplay:
    """
    Rekonstruiert vergangene Trades fuer Chart-Replay und Post-Trade-Analyse.

    Parameters
    ----------
    journal           : TradeJournal-Instanz (Quelle fuer Trade-Metadaten).
    features_dir      : Verzeichnis mit Parquet-Dateien der DataPipeline.
                        Standard: 'data/features'.
    default_timeframe : Timeframe-Kuerzel fuer Parquet-Dateinamen wenn in
                        get_replay_data() kein `timeframe` angegeben wird.
                        Standard: 'H1'.
    """

    def __init__(
        self,
        journal: TradeJournal,
        features_dir: str | Path = "data/features",
        default_timeframe: str = "H1",
    ) -> None:
        self._journal      = journal
        self._features_dir = Path(features_dir)
        self._default_tf   = default_timeframe

    # ── Oeffentliche Schnittstelle ────────────────────────────────────────────

    def get_replay_data(
        self,
        trade_id: int,
        lookback_candles: int = 100,
        timeframe: Optional[str] = None,
    ) -> dict:
        """
        Rekonstruiert alle Daten fuer den angegebenen Trade.

        No-Lookahead-Garantie:
            Alle zurückgegebenen Candles und Indikatoren liegen zeitlich
            <= Entry-Zeitpunkt des Trades. meta['no_lookahead'] ist True.

        Parameters
        ----------
        trade_id         : Trade-ID aus TradeJournal.log_trade_open().
        lookback_candles : Anzahl Kerzen vor dem Entry (Standard: 100).
        timeframe        : Parquet-Timeframe, z.B. 'H1', 'H4', 'D1'.
                           Wird default_timeframe verwendet wenn None.

        Returns
        -------
        dict  – GUI-freundliches Replay-Paket (Struktur siehe Modul-Docstring).

        Raises
        ------
        TradeNotFoundError      : trade_id nicht im Journal.
        ReplayDataNotFoundError : Keine Parquet-Daten fuer Symbol / Zeitraum.
        """
        trade = self._journal.get_trade(trade_id)
        if trade is None:
            raise TradeNotFoundError(
                f"Trade-ID {trade_id} nicht im Journal gefunden."
            )

        tf         = timeframe or self._default_tf
        symbol     = trade.get("symbol") or ""
        entry_time = _parse_iso(trade.get("entry_time"))
        exit_time  = _parse_iso(trade.get("exit_time"))

        if entry_time is None:
            raise ReplayDataNotFoundError(
                f"Trade {trade_id} hat keinen gueltigen entry_time."
            )

        logger.debug(
            "TradeReplay: lade Replay fuer Trade {id} | {sym} {tf} | entry={et}",
            id=trade_id, sym=symbol, tf=tf, et=entry_time.isoformat(),
        )

        features_df = self._load_features(symbol, tf, entry_time, lookback_candles)
        candles     = self._df_to_candles(features_df)
        indicators  = self._df_to_indicators(features_df)

        entry_marker: dict = {
            "time":      entry_time.isoformat(),
            "price":     trade.get("entry_price"),
            "direction": trade.get("direction"),
        }
        exit_marker: Optional[dict] = None
        if exit_time is not None:
            exit_marker = {
                "time":  exit_time.isoformat(),
                "price": trade.get("exit_price"),
            }

        return {
            "trade":        trade,
            "candles":      candles,
            "entry_marker": entry_marker,
            "exit_marker":  exit_marker,
            "indicators":   indicators,
            "news_events":  _parse_news(trade.get("news_context")),
            "meta": {
                "symbol":           symbol,
                "timeframe":        tf,
                "lookback_candles": lookback_candles,
                "candles_found":    len(candles),
                "no_lookahead":     True,
                "entry_time":       entry_time.isoformat(),
            },
        }

    # ── Private Methoden ──────────────────────────────────────────────────────

    def _load_features(
        self,
        symbol: str,
        timeframe: str,
        entry_time: datetime,
        lookback_candles: int,
    ) -> pd.DataFrame:
        """
        Laedt alle Parquet-Dateien fuer das Symbol, konkateniert, filtert
        auf Zeitstempel <= entry_time und gibt die letzten lookback_candles
        Zeilen zurueck.

        No-Lookahead wird durch den `<= entry_time`-Filter erzwungen.
        """
        pattern = f"{symbol}_{timeframe}_*.parquet"
        files   = sorted(self._features_dir.glob(pattern))

        if not files:
            raise ReplayDataNotFoundError(
                f"Keine Parquet-Dateien fuer '{symbol}_{timeframe}' "
                f"in '{self._features_dir}'. "
                "DataPipeline.run_batch() fuer diesen Zeitraum ausfuehren."
            )

        frames: list[pd.DataFrame] = []
        for f in files:
            try:
                frames.append(pd.read_parquet(f))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Parquet lesen fehlgeschlagen | {f} | {e}", f=f, e=exc)

        if not frames:
            raise ReplayDataNotFoundError(
                f"Alle Parquet-Dateien fuer '{symbol}_{timeframe}' "
                "konnten nicht gelesen werden."
            )

        combined = pd.concat(frames)
        combined = self._normalize_index(combined)
        # Duplikate entfernen (ueberlappende Batch-Laeufe)
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()

        # ── No-Lookahead: ausschliesslich Daten bis entry_time ───────────────
        filtered = combined[combined.index <= entry_time]

        if filtered.empty:
            raise ReplayDataNotFoundError(
                f"Keine Daten fuer '{symbol}' vor Entry-Zeitpunkt "
                f"{entry_time.isoformat()}."
            )

        return filtered.iloc[-lookback_candles:]

    def _normalize_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Stellt sicher dass der DataFrame einen timezone-aware DatetimeIndex hat.

        Unterstuetzt:
          - DatetimeIndex (timezone-aware oder naive → wird als UTC angenommen)
          - 'timestamp'-Spalte (str oder datetime → wird zu DatetimeIndex)
        """
        if isinstance(df.index, pd.DatetimeIndex):
            if df.index.tz is None:
                df = df.copy()
                df.index = df.index.tz_localize("UTC")
            return df

        if "timestamp" in df.columns:
            df = df.copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            return df.set_index("timestamp")

        raise ReplayDataNotFoundError(
            "Parquet-Dateien haben kein verwertbares Zeitstempel-Format "
            "(kein DatetimeIndex und keine 'timestamp'-Spalte)."
        )

    def _df_to_candles(self, df: pd.DataFrame) -> list[dict]:
        """Konvertiert DataFrame-Zeilen in GUI-freundliche Candle-Dicts."""
        ohlcv_present = [c for c in ("open", "high", "low", "close", "volume")
                         if c in df.columns]
        result: list[dict] = []
        for ts, row in df.iterrows():
            candle: dict = {"time": ts.isoformat()}
            for col in ohlcv_present:
                v = row[col]
                candle[col] = float(v) if pd.notna(v) else None
            result.append(candle)
        return result

    def _df_to_indicators(self, df: pd.DataFrame) -> dict[str, list]:
        """Extrahiert alle Nicht-OHLCV-Spalten als parallele Indikator-Listen."""
        cols = [c for c in df.columns if c.lower() not in _OHLCV_COLS]
        return {
            col: [float(v) if pd.notna(v) else None for v in df[col]]
            for col in cols
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Hilfsfunktionen
# ─────────────────────────────────────────────────────────────────────────────

def _parse_iso(ts_str: Optional[str]) -> Optional[datetime]:
    """Konvertiert ISO-Timestamp-String in timezone-aware datetime; None bei Fehler."""
    if not ts_str:
        return None
    try:
        dt = datetime.fromisoformat(str(ts_str))
        return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def _parse_news(news_str: Optional[str]) -> list:
    """
    Parst das news_context-Feld aus dem Journal.

    Unterstuetzt: JSON-Array, JSON-Objekt, einfacher String, None/leer.
    """
    if not news_str:
        return []
    try:
        result = json.loads(news_str)
        return result if isinstance(result, list) else [result]
    except (json.JSONDecodeError, TypeError):
        return [news_str]
