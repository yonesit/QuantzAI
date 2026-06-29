"""
src/data/dukascopy_downloader.py
DukascopyDownloader – laedt historische Bid/Ask-Tickdaten vom kostenlosen
Dukascopy-Datafeed und aggregiert sie auf M15-Bars inkl. echtem Bid-Ask-Spread.

Warum Dukascopy?
  Fusion-Markets/MT5 haelt nur ~4 Jahre M15-Historie vor (siehe Phase-1-Issue).
  Dukascopy bietet kostenlos Tick-Historie (EURUSD ab ~2003, XAUUSD ab ~2010).
  Aus den Ticks bauen wir M15-Bars UND speichern den realen historischen Spread
  pro Bar – Grundlage fuer das kostenbewusste Triple-Barrier-Labeling (Phase 1).

Datenformat (Dukascopy .bi5):
  URL:  https://datafeed.dukascopy.com/datafeed/{SYM}/{YYYY}/{MM0}/{DD}/{HH}h_ticks.bi5
        MM0 ist NULL-basiert (Januar = 00, Dezember = 11).
  Inhalt: LZMA-komprimiert. Dekomprimiert = Folge von 20-Byte-Records (big-endian):
        int32  ms_offset   (Millisekunden ab Stundenbeginn)
        int32  ask         (Preis * 10^digits)
        int32  bid         (Preis * 10^digits)
        float32 ask_volume
        float32 bid_volume

Robustheit:
  - Rate-Limiting (Pause zwischen Requests) – Dukascopy drosselt.
  - Retries mit exponentiellem Backoff.
  - 404 = keine Daten fuer diese Stunde (Markt geschlossen) -> leer, kein Fehler.
  - Resume: pro Tag wird ein Cache-Parquet geschrieben; vorhandene Tage werden
    uebersprungen. Ein Abbruch verwirft also nur den laufenden Tag.
  - Fortschritts-Logging pro Tag.

Keine Business-Logik – reine Datenbeschaffung + Aggregation.
"""

from __future__ import annotations

import lzma
import random
import struct
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
from loguru import logger

try:  # requests ist Pflicht-Dependency, Import aber defensiv halten
    import requests
except ImportError:  # pragma: no cover
    requests = None  # type: ignore


# ─────────────────────────────────────────────
#  Konstanten
# ─────────────────────────────────────────────

DUKASCOPY_BASE_URL = "https://datafeed.dukascopy.com/datafeed"

# Preis-Skalierung: Dukascopy speichert Integer = Preis * 10^digits.
# EURUSD: 5 Nachkommastellen -> /1e5 ; XAUUSD: 3 -> /1e3.
SYMBOL_POINT_DIVISOR: dict[str, float] = {
    "EURUSD": 1e5,
    "XAUUSD": 1e3,
}

# Pip-Groesse je Symbol (fuer Spread-Reporting in Pips).
SYMBOL_PIP_SIZE: dict[str, float] = {
    "EURUSD": 0.0001,
    "XAUUSD": 0.01,
}

# Frueheste plausibel verfuegbare Historie je Symbol bei Dukascopy.
SYMBOL_HISTORY_START: dict[str, datetime] = {
    "EURUSD": datetime(2003, 5, 5, tzinfo=timezone.utc),
    "XAUUSD": datetime(2010, 1, 4, tzinfo=timezone.utc),
}

_TICK_STRUCT = struct.Struct(">3i2f")  # ms, ask, bid, ask_vol, bid_vol
_TICK_SIZE = _TICK_STRUCT.size          # 20 Bytes

# HTTP-Status, die auf voruebergehende Drosselung deuten -> mit Geduld neu versuchen.
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class DukascopyError(Exception):
    """Download- oder Dekodierfehler bei Dukascopy-Daten."""


# ─────────────────────────────────────────────
#  Reines Dekodieren (netzwerkfrei, gut testbar)
# ─────────────────────────────────────────────

def decode_bi5(raw: bytes, hour_start: datetime, point_divisor: float) -> pd.DataFrame:
    """
    Dekodiert den Inhalt einer .bi5-Datei (LZMA-komprimiert) in einen Tick-DataFrame.

    Parameters
    ----------
    raw           : Roh-Bytes der .bi5-Datei (LZMA-komprimiert). Leer -> leerer DF.
    hour_start    : UTC-Beginn der Stunde, zu der die Datei gehoert.
    point_divisor : Skalierungsfaktor (z.B. 1e5 fuer EURUSD).

    Returns
    -------
    DataFrame mit Spalten: timestamp (UTC), bid, ask, bid_volume, ask_volume.
    """
    if not raw:
        return _empty_tick_frame()

    try:
        data = lzma.decompress(raw)
    except lzma.LZMAError as exc:
        raise DukascopyError(f"LZMA-Dekompression fehlgeschlagen: {exc}") from exc

    if len(data) == 0:
        return _empty_tick_frame()
    if len(data) % _TICK_SIZE != 0:
        raise DukascopyError(
            f"Unerwartete .bi5-Groesse {len(data)} (kein Vielfaches von {_TICK_SIZE})."
        )

    n = len(data) // _TICK_SIZE
    ms = [0] * n
    ask = [0.0] * n
    bid = [0.0] * n
    askv = [0.0] * n
    bidv = [0.0] * n
    for i, (m, a, b, av, bv) in enumerate(_TICK_STRUCT.iter_unpack(data)):
        ms[i] = m
        ask[i] = a / point_divisor
        bid[i] = b / point_divisor
        askv[i] = av
        bidv[i] = bv

    base = pd.Timestamp(hour_start).tz_convert("UTC") if hour_start.tzinfo else \
        pd.Timestamp(hour_start, tz="UTC")
    ts = base + pd.to_timedelta(ms, unit="ms")

    return pd.DataFrame({
        "timestamp":  ts,
        "bid":        bid,
        "ask":        ask,
        "bid_volume": bidv,
        "ask_volume": askv,
    })


def _empty_tick_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp":  pd.Series([], dtype="datetime64[ns, UTC]"),
        "bid":        pd.Series([], dtype="float64"),
        "ask":        pd.Series([], dtype="float64"),
        "bid_volume": pd.Series([], dtype="float64"),
        "ask_volume": pd.Series([], dtype="float64"),
    })


def ticks_to_m15(ticks: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregiert Bid/Ask-Ticks auf M15-Bars inkl. Spread.

    OHLC wird aus dem Mid-Preis (bid+ask)/2 gebildet. Zusaetzlich wird pro Bar
    der durchschnittliche und mediane Bid-Ask-Spread gespeichert – die Grundlage
    fuer kostenbewusstes Labeling mit echten historischen Spreads.

    Returns
    -------
    DataFrame mit Spalten:
        timestamp (Bar-Open, UTC, 15-min-Raster), open, high, low, close,
        volume, spread_mean, spread_median, tick_count.
        Bars ohne Ticks werden nicht erzeugt (Wochenend-Gaps bleiben Luecken).
    """
    if ticks.empty:
        return _empty_m15_frame()

    df = ticks.copy()
    df = df.set_index("timestamp").sort_index()
    mid = (df["bid"] + df["ask"]) / 2.0
    spread = df["ask"] - df["bid"]
    vol = df["bid_volume"] + df["ask_volume"]

    work = pd.DataFrame({"mid": mid, "spread": spread, "vol": vol})
    res = work.resample("15min", label="left", closed="left")

    out = pd.DataFrame({
        "open":          res["mid"].first(),
        "high":          res["mid"].max(),
        "low":           res["mid"].min(),
        "close":         res["mid"].last(),
        "volume":        res["vol"].sum(),
        "spread_mean":   res["spread"].mean(),
        "spread_median": res["spread"].median(),
        "tick_count":    res["mid"].size(),
    })
    # Nur Bars behalten, die tatsaechlich Ticks hatten
    out = out[out["tick_count"] > 0].copy()
    out.index.name = "timestamp"
    out = out.reset_index()
    return out


def _empty_m15_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp":     pd.Series([], dtype="datetime64[ns, UTC]"),
        "open":          pd.Series([], dtype="float64"),
        "high":          pd.Series([], dtype="float64"),
        "low":           pd.Series([], dtype="float64"),
        "close":         pd.Series([], dtype="float64"),
        "volume":        pd.Series([], dtype="float64"),
        "spread_mean":   pd.Series([], dtype="float64"),
        "spread_median": pd.Series([], dtype="float64"),
        "tick_count":    pd.Series([], dtype="int64"),
    })


# ─────────────────────────────────────────────
#  Downloader
# ─────────────────────────────────────────────

@dataclass
class DownloadStats:
    symbol:        str
    days_total:    int = 0
    days_fetched:  int = 0
    days_cached:   int = 0
    days_empty:    int = 0
    days_failed:   int = 0
    hours_fetched: int = 0
    hours_404:     int = 0
    bars_total:    int = 0


class DukascopyDownloader:
    """
    Laedt Dukascopy-Tickdaten und aggregiert sie auf M15-Bars mit Spread.

    Parameters
    ----------
    cache_dir      : Verzeichnis fuer Tages-Cache-Parquets (Resume).
    rate_limit_s   : Pause zwischen HTTP-Requests (Sekunden).
    max_retries    : Versuche pro Stunde bei Netzwerkfehlern.
    timeout_s      : HTTP-Timeout pro Request.
    session        : optionale requests.Session (Tests injizieren Fakes).
    sleep_func     : Pausen-Funktion (Tests injizieren No-Op).
    """

    def __init__(
        self,
        cache_dir:    str | Path,
        rate_limit_s: float = 0.3,
        max_retries:  int = 8,
        timeout_s:    float = 30.0,
        retry_base_s: float = 2.0,
        max_backoff_s: float = 60.0,
        day_cooldown_s: float = 10.0,
        session:      Optional["requests.Session"] = None,
        sleep_func:   Callable[[float], None] = time.sleep,
    ) -> None:
        self.cache_dir      = Path(cache_dir)
        self.rate_limit_s   = rate_limit_s
        self.max_retries    = max_retries
        self.timeout_s      = timeout_s
        self.retry_base_s   = retry_base_s
        self.max_backoff_s  = max_backoff_s
        self.day_cooldown_s = day_cooldown_s
        self._sleep         = sleep_func
        if session is not None:
            self._session = session
        elif requests is not None:
            self._session = requests.Session()
            # Dukascopy antwortet ohne User-Agent haeufig mit 503.
            self._session.headers.update({
                "User-Agent": "Mozilla/5.0 (compatible; QuantzAI-DataFetcher/1.0)",
                "Accept": "*/*",
            })
        else:  # pragma: no cover
            self._session = None

    # ── URL / Fetch ──────────────────────────────

    @staticmethod
    def hour_url(symbol: str, dt: datetime) -> str:
        """Baut die Dukascopy-URL fuer eine bestimmte Stunde (MM ist 0-basiert)."""
        return (
            f"{DUKASCOPY_BASE_URL}/{symbol}/{dt.year:04d}/{dt.month - 1:02d}/"
            f"{dt.day:02d}/{dt.hour:02d}h_ticks.bi5"
        )

    def fetch_hour_raw(self, symbol: str, dt: datetime) -> Optional[bytes]:
        """
        Laedt die Roh-.bi5-Bytes fuer eine Stunde.

        Returns None bei HTTP 404 (Markt geschlossen / keine Daten). Wirft
        DukascopyError erst nach erschoepften Retries bei echten Fehlern.
        """
        if self._session is None:  # pragma: no cover
            raise DukascopyError("requests nicht verfuegbar – Session fehlt.")
        url = self.hour_url(symbol, dt)
        last_err: str = "unbekannt"
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._session.get(url, timeout=self.timeout_s)
                if resp.status_code == 404:
                    return None                       # keine Daten = Markt geschlossen
                if resp.status_code == 200:
                    return resp.content
                if resp.status_code not in _RETRYABLE_STATUS:
                    raise DukascopyError(f"HTTP {resp.status_code} fuer {url}")
                last_err = f"HTTP {resp.status_code} (Drosselung)"
            except DukascopyError:
                raise                                 # nicht-retrybare HTTP-Fehler
            except Exception as exc:  # noqa: BLE001  (Timeout/Connection)
                last_err = str(exc)

            if attempt < self.max_retries:
                # Exponentielles Backoff mit Jitter, gedeckelt – fair gegenueber Dukascopy.
                backoff = min(self.retry_base_s * (2 ** (attempt - 1)), self.max_backoff_s)
                backoff += random.uniform(0, self.retry_base_s)
                logger.warning(
                    "Dukascopy fetch (Versuch {a}/{m}): {e} – backoff {b:.1f}s | {u}",
                    a=attempt, m=self.max_retries, e=last_err, b=backoff, u=url,
                )
                self._sleep(backoff)
        raise DukascopyError(f"Fetch endgueltig fehlgeschlagen nach {self.max_retries} Versuchen: {url} | {last_err}")

    # ── Tages-Download (mit Cache/Resume) ─────────

    def fetch_day(self, symbol: str, day: datetime, stats: DownloadStats) -> pd.DataFrame:
        """Laedt alle 24 Stunden eines Tages und aggregiert sie zu M15-Bars."""
        divisor = SYMBOL_POINT_DIVISOR.get(symbol)
        if divisor is None:
            raise DukascopyError(f"Kein Point-Divisor fuer Symbol '{symbol}' definiert.")

        day0 = day.replace(hour=0, minute=0, second=0, microsecond=0)
        frames: list[pd.DataFrame] = []
        for h in range(24):
            hour_dt = day0 + timedelta(hours=h)
            raw = self.fetch_hour_raw(symbol, hour_dt)
            if self.rate_limit_s:
                self._sleep(self.rate_limit_s)
            if raw is None:
                stats.hours_404 += 1
                continue
            stats.hours_fetched += 1
            ticks = decode_bi5(raw, hour_dt, divisor)
            if not ticks.empty:
                frames.append(ticks)

        if not frames:
            return _empty_m15_frame()
        all_ticks = pd.concat(frames, ignore_index=True)
        return ticks_to_m15(all_ticks)

    def download_range(
        self,
        symbol: str,
        start:  datetime,
        end:    datetime,
        skip_saturday: bool = True,
        progress: bool = True,
    ) -> tuple[pd.DataFrame, DownloadStats]:
        """
        Laedt M15-Bars fuer [start, end] (tageweise, resumebar).

        Pro Tag wird ein Cache-Parquet geschrieben (auch leer, als Resume-Marker).
        Vorhandene Tage werden uebersprungen. Gibt den zusammengesetzten
        M15-DataFrame und Download-Statistiken zurueck.
        """
        start = _as_utc(start)
        end   = _as_utc(end)
        sym_cache = self.cache_dir / symbol
        sym_cache.mkdir(parents=True, exist_ok=True)

        stats = DownloadStats(symbol=symbol)
        day = start.replace(hour=0, minute=0, second=0, microsecond=0)
        end_day = end.replace(hour=0, minute=0, second=0, microsecond=0)

        while day <= end_day:
            stats.days_total += 1
            # Samstag (5) ist durchgehend geschlossen – optional ueberspringen.
            if skip_saturday and day.weekday() == 5:
                day += timedelta(days=1)
                stats.days_total -= 1
                continue

            cache_file = sym_cache / f"{day:%Y-%m-%d}.parquet"
            if cache_file.exists():
                stats.days_cached += 1
                day += timedelta(days=1)
                continue

            # Fehlertoleranz: ein einzelner Tag (Drosselung/Netz) darf den
            # mehrjaehrigen Lauf NICHT abbrechen. Fehlgeschlagene Tage werden
            # NICHT gecacht -> beim naechsten Lauf automatisch erneut versucht.
            try:
                bars = self.fetch_day(symbol, day, stats)
            except DukascopyError as exc:
                stats.days_failed += 1
                logger.error(
                    "Tag {d} fehlgeschlagen ({e}) – nicht gecacht, Retry beim naechsten Lauf. "
                    "Cooldown {c:.0f}s.",
                    d=f"{day:%Y-%m-%d}", e=exc, c=self.day_cooldown_s,
                )
                if self.day_cooldown_s:
                    self._sleep(self.day_cooldown_s)
                day += timedelta(days=1)
                continue

            # Auch leere Tage als Marker schreiben (Resume), aber Stats sauber halten
            bars.to_parquet(cache_file, index=False, compression="snappy")
            if bars.empty:
                stats.days_empty += 1
            else:
                stats.days_fetched += 1
                stats.bars_total += len(bars)

            if progress and stats.days_total % 50 == 0:
                logger.info(
                    "Dukascopy {sym} | Tag {d} | geladen={f} cached={c} leer={e} bars={b}",
                    sym=symbol, d=f"{day:%Y-%m-%d}", f=stats.days_fetched,
                    c=stats.days_cached, e=stats.days_empty, b=stats.bars_total,
                )
            day += timedelta(days=1)

        full = self.assemble_from_cache(symbol)
        logger.info(
            "Dukascopy {sym} fertig | bars={n} | {s} .. {e}",
            sym=symbol, n=len(full),
            s=(full['timestamp'].min() if not full.empty else 'N/A'),
            e=(full['timestamp'].max() if not full.empty else 'N/A'),
        )
        return full, stats

    def assemble_from_cache(self, symbol: str) -> pd.DataFrame:
        """Liest alle Tages-Cache-Parquets eines Symbols und konkateniert sie."""
        sym_cache = self.cache_dir / symbol
        files = sorted(sym_cache.glob("*.parquet"))
        frames = []
        for f in files:
            try:
                d = pd.read_parquet(f)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Cache-Datei unlesbar, ueberspringe: {f} ({e})", f=f, e=exc)
                continue
            if not d.empty:
                frames.append(d)
        if not frames:
            return _empty_m15_frame()
        out = pd.concat(frames, ignore_index=True)
        out = out.drop_duplicates(subset=["timestamp"], keep="last")
        out = out.sort_values("timestamp").reset_index(drop=True)
        return out


def _as_utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
