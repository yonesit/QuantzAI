"""
src/data/gdelt_sentiment.py
GDELT-basiertes historisches Sentiment fuer EUR/USD Backtesting.

GDELTDownloader:
  Laedt GDELT 2.0 GKG-Dateien (15-Minuten-Intervall), filtert auf
  EUR/USD-relevante Themen, speichert als lokale Parquet-DB.
  Resume-faehig: bereits geladene bucket_times werden uebersprungen.

SentimentHistory:
  Liest die Parquet-DB und liefert eine look-ahead-sichere
  get_sentiment_series() analog zu FeatureBuilder._merge_mtf_trend().
  Fuer jede H1-Bar bei Zeitpunkt T wird ausschliesslich Sentiment
  aus Nachrichten mit bucket_time < T verwendet (kein Look-ahead).
"""
from __future__ import annotations

import io
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from urllib.request import urlopen, Request

import numpy as np
import pandas as pd
from loguru import logger


# ── EUR/USD-relevante GDELT GKG v2 Theme-Prefixes ───────────────────────────

_EURUSD_THEME_PREFIXES: tuple[str, ...] = (
    "ECON_",
    "WB_613_",           # Financial Sector
    "WB_673_",           # Monetary Institutions
    "FINSOURCE_INC_ECON",
    "CENTRAL_BANK",
    "MONETARY_POLICY",
)

_FINANCIAL_DOMAINS: frozenset[str] = frozenset([
    "reuters.com", "bloomberg.com", "ft.com", "wsj.com",
    "cnbc.com", "marketwatch.com", "economist.com",
    "forexlive.com", "dailyfx.com", "fxstreet.com",
    "investing.com", "seekingalpha.com",
])

# V2Tone liegt in der Praxis bei Finanznachrichten im Bereich [-30, +30]
_TONE_SCALE = 30.0

_GDELT_GKG_URL = "http://data.gdeltproject.org/gdeltv2/{ts}.gkg.csv.zip"

# Wie viele neue Buckets zwischen zwei Parquet-Schreibvorgaengen (Checkpoint)
_CHECKPOINT_EVERY = 200


# ── Hilfsfunktionen ──────────────────────────────────────────────────────────

def _to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _is_eurusd_relevant(themes: str, source: str) -> bool:
    """True wenn der Artikel EUR/USD-relevant ist."""
    for prefix in _EURUSD_THEME_PREFIXES:
        if prefix in themes:
            return True
    src_lower = source.lower()
    for domain in _FINANCIAL_DOMAINS:
        if domain in src_lower:
            return True
    return False


def _parse_v2tone(tone_str: str) -> Optional[float]:
    """Extrahiert den durchschnittlichen Ton aus dem V2Tone-Feld (erster Wert)."""
    if not tone_str:
        return None
    try:
        return float(tone_str.split(",")[0])
    except (ValueError, IndexError):
        return None


def _iter_gdelt_timestamps(
    start: datetime, end: datetime, step_minutes: int
) -> list[datetime]:
    """Erzeugt GDELT-Datei-Zeitstempel in [start, end) mit step_minutes Abstand.
    GDELT hat Dateien alle 15 Minuten; step muss Vielfaches von 15 sein."""
    step = max(15, (step_minutes // 15) * 15)
    ts = start.replace(second=0, microsecond=0)
    ts = ts.replace(minute=(ts.minute // 15) * 15)
    result: list[datetime] = []
    while ts < end:
        if ts.minute % step == 0:
            result.append(ts)
        ts += timedelta(minutes=15)
    return result


# ── GDELTDownloader ──────────────────────────────────────────────────────────

class GDELTDownloader:
    """
    Laedt GDELT 2.0 GKG-Dateien und extrahiert EUR/USD-relevante
    (bucket_time, avg_tone, n_articles)-Zeilen.

    Resume-faehig: Beim Neustart werden bereits in der Parquet-DB
    vorhandene bucket_times uebersprungen. Alle _CHECKPOINT_EVERY
    neuen Buckets wird ein Zwischenspeicher geschrieben.

    Parameters
    ----------
    data_dir      : Verzeichnis fuer die Parquet-DB (data/news/)
    step_minutes  : Nur jede N-te 15-Minuten-Datei laden (60 = stuendlich).
                    Muss Vielfaches von 15 sein.
    timeout       : HTTP-Timeout pro Datei in Sekunden
    """

    def __init__(
        self,
        data_dir: str | Path = "data/news",
        step_minutes: int = 60,
        timeout: int = 20,
    ) -> None:
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._step_minutes = max(15, (step_minutes // 15) * 15)
        self._timeout = timeout

    # ── Oeffentliche Methode ──────────────────────────────────────

    def download_range(
        self,
        start: datetime,
        end: datetime,
        symbol: str = "EURUSD",
    ) -> pd.DataFrame:
        """
        Laedt GDELT-Dateien fuer [start, end), filtert auf EUR/USD-Themen und
        gibt einen DataFrame (bucket_time, avg_tone, n_articles) zurueck.
        Speichert/aktualisiert gleichzeitig die lokale Parquet-DB.

        Look-ahead-Sicherheit: bucket_time ist der GDELT-Datei-Zeitstempel,
        also der Zeitpunkt bis zu dem die Nachrichten veroeffentlicht wurden.
        Nachrichten aus bucket_time=T werden NICHT fuer H1-Bar bei T verwendet.

        Resume: bereits vorhandene bucket_times in der Parquet-DB werden
        uebersprungen; alle _CHECKPOINT_EVERY Buckets wird ein Checkpoint
        geschrieben.
        """
        start = _to_utc(start)
        end   = _to_utc(end)

        timestamps = _iter_gdelt_timestamps(start, end, self._step_minutes)
        existing   = self._load_existing_timestamps(symbol)
        todo       = [ts for ts in timestamps if ts not in existing]

        n_skipped = len(timestamps) - len(todo)
        if n_skipped:
            logger.info(
                "GDELT resume | {skip} bereits geladen, {todo} verbleiben",
                skip=n_skipped, todo=len(todo),
            )

        pending_rows: list[dict] = []
        n_ok = n_empty = n_err = 0
        t0 = time.monotonic()

        for i, ts in enumerate(todo, 1):
            ts_str = ts.strftime("%Y%m%d%H%M%S")
            url = _GDELT_GKG_URL.format(ts=ts_str)
            try:
                batch = self._fetch_and_filter(url, ts)
                if batch:
                    pending_rows.extend(batch)
                    n_ok += 1
                else:
                    n_empty += 1
            except Exception as exc:  # noqa: BLE001
                logger.debug("GDELT skip | {ts} | {exc}", ts=ts_str, exc=exc)
                n_err += 1

            # Fortschrittslog alle 100 Dateien
            if i % 100 == 0 or i == len(todo):
                elapsed = time.monotonic() - t0
                rate = i / elapsed if elapsed > 0 else 0
                eta_s = (len(todo) - i) / rate if rate > 0 else 0
                eta_h = eta_s / 3600
                logger.info(
                    "GDELT progress | {i}/{n} ({pct:.0f}%) | ok={ok} err={err} | ETA {eta:.1f}h",
                    i=i, n=len(todo), pct=100*i/len(todo),
                    ok=n_ok, err=n_err, eta=eta_h,
                )

            # Zwischenspeicher (Checkpoint)
            if pending_rows and (i % _CHECKPOINT_EVERY == 0 or i == len(todo)):
                self._flush(pending_rows, symbol)
                pending_rows = []

        logger.info(
            "GDELT download done | ok={ok} empty={empty} err={err}",
            ok=n_ok, empty=n_empty, err=n_err,
        )

        return self._read_parquet(symbol)

    # ── Interne Methoden ──────────────────────────────────────────

    def _load_existing_timestamps(self, symbol: str) -> set[datetime]:
        """Gibt die Menge aller bucket_times aus der lokalen Parquet-DB zurueck."""
        path = self._data_dir / f"gdelt_{symbol}.parquet"
        if not path.exists():
            return set()
        df = pd.read_parquet(path, columns=["bucket_time"])
        df["bucket_time"] = pd.to_datetime(df["bucket_time"], utc=True)
        return set(df["bucket_time"].dt.to_pydatetime())

    def _fetch_and_filter(
        self, url: str, bucket_time: datetime
    ) -> list[dict]:
        """Laedt eine GDELT-Datei und gibt gefilterte Zeilen zurueck."""
        req = Request(url, headers={"User-Agent": "QuantzAI/1.0"})
        with urlopen(req, timeout=self._timeout) as resp:
            data = resp.read()

        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            name = zf.namelist()[0]
            with zf.open(name) as f:
                content = f.read().decode("utf-8", errors="replace")

        rows: list[dict] = []
        for line in content.splitlines():
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 16:
                continue
            # GDELT GKG v2 Spalten (Tab-getrennt, 0-indiziert):
            # 1=DATE, 3=SourceCommonName, 8=V2Themes, 15=V2Tone
            themes   = parts[8]  if len(parts) > 8  else ""
            source   = parts[3]  if len(parts) > 3  else ""
            tone_raw = parts[15] if len(parts) > 15 else ""

            if not _is_eurusd_relevant(themes, source):
                continue

            tone = _parse_v2tone(tone_raw)
            if tone is None:
                continue

            rows.append({"bucket_time": bucket_time, "tone": tone})

        return rows

    def _flush(self, rows: list[dict], symbol: str) -> None:
        """Aggregiert rows und schreibt sie in die Parquet-DB (Upsert)."""
        raw = pd.DataFrame(rows)
        agg = (
            raw.groupby("bucket_time")
            .agg(avg_tone=("tone", "mean"), n_articles=("tone", "count"))
            .reset_index()
        )
        agg["bucket_time"] = pd.to_datetime(agg["bucket_time"], utc=True)
        self._upsert_parquet(agg, symbol)

    def _upsert_parquet(self, new_df: pd.DataFrame, symbol: str) -> None:
        """Fuegt new_df in die bestehende Parquet-DB ein (keine Duplikate)."""
        path = self._data_dir / f"gdelt_{symbol}.parquet"
        if path.exists():
            existing = pd.read_parquet(path)
            existing["bucket_time"] = pd.to_datetime(existing["bucket_time"], utc=True)
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["bucket_time"], keep="last")
        else:
            combined = new_df
        combined = combined.sort_values("bucket_time").reset_index(drop=True)
        combined.to_parquet(path, index=False)
        logger.info("GDELT checkpoint | {path} | {n} rows total", path=path, n=len(combined))

    def _read_parquet(self, symbol: str) -> pd.DataFrame:
        path = self._data_dir / f"gdelt_{symbol}.parquet"
        if not path.exists():
            return pd.DataFrame(columns=["bucket_time", "avg_tone", "n_articles"])
        df = pd.read_parquet(path)
        df["bucket_time"] = pd.to_datetime(df["bucket_time"], utc=True)
        return df.sort_values("bucket_time").reset_index(drop=True)

    def _upsert_parquet(self, new_df: pd.DataFrame, symbol: str) -> None:
        """Fuegt new_df in die bestehende Parquet-DB ein (keine Duplikate)."""
        path = self._data_dir / f"gdelt_{symbol}.parquet"
        if path.exists():
            existing = pd.read_parquet(path)
            existing["bucket_time"] = pd.to_datetime(existing["bucket_time"], utc=True)
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["bucket_time"], keep="last")
        else:
            combined = new_df
        combined = combined.sort_values("bucket_time").reset_index(drop=True)
        combined.to_parquet(path, index=False)
        logger.info("GDELT saved | {path} | {n} rows", path=path, n=len(combined))


# ── SentimentHistory ─────────────────────────────────────────────────────────

class SentimentHistory:
    """
    Look-ahead-sichere historische Sentiment-Datenbank aus GDELT.

    Fuer jede H1-Bar bei Zeitpunkt T wird ausschliesslich Sentiment aus
    Nachrichten mit bucket_time < T verwendet (kein Look-ahead).
    Das Fenster (T - window_hours, T) wird per binaerer Suche bestimmt.

    Analog zu FeatureBuilder._merge_mtf_trend().

    Parameters
    ----------
    path         : Pfad zur GDELT Parquet-Datei (gdelt_EURUSD.parquet)
    window_hours : Lookback-Fenster in Stunden (Standard: 2h)
    """

    def __init__(
        self,
        path: str | Path,
        window_hours: float = 2.0,
    ) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"GDELT Sentiment-DB nicht gefunden: {path}\n"
                "Erstelle sie mit: python scripts/download_gdelt_news.py"
            )
        raw = pd.read_parquet(path)
        raw["bucket_time"] = pd.to_datetime(raw["bucket_time"], utc=True)
        raw = raw.sort_values("bucket_time").reset_index(drop=True)

        # In nanosekunden seit Epoch (int64) fuer schnelle binaere Suche
        self._bt_ns: np.ndarray = raw["bucket_time"].values.astype("int64")
        self._weighted: np.ndarray = (raw["avg_tone"] * raw["n_articles"]).values.astype(float)
        self._counts: np.ndarray   = raw["n_articles"].astype(float).values
        self._window_hours = window_hours

    @classmethod
    def from_parquet(
        cls, path: str | Path, window_hours: float = 2.0
    ) -> "SentimentHistory":
        return cls(path=path, window_hours=window_hours)

    # ── Kern-Methode ──────────────────────────────────────────────

    def get_sentiment_series(
        self,
        timestamps: pd.Series,
        window_hours: Optional[float] = None,
    ) -> np.ndarray:
        """
        Fuer jede H1-Bar bei T: durchschnittlicher Ton aller Nachrichten
        mit  T - window_hours <= bucket_time < T  (kein Look-ahead).

        Implementierung: np.searchsorted (binaere Suche) → O(N log M)
        wobei N = Anzahl H1-Bars, M = Anzahl GDELT-Buckets.

        Parameters
        ----------
        timestamps   : pd.Series mit H1-Bar-Zeitstempeln (UTC oder naiv=UTC)
        window_hours : ueberschreibt den Instanz-Standard

        Returns
        -------
        np.ndarray float64 in [-1, +1], gleiche Laenge wie timestamps
        """
        win_h = window_hours if window_hours is not None else self._window_hours
        window_ns = int(win_h * 3600 * 1_000_000_000)

        # Zeitstempel in nanosekunden (UTC)
        ts_ser = pd.Series(pd.to_datetime(timestamps))
        if ts_ser.dt.tz is None:
            ts_ser = ts_ser.dt.tz_localize("UTC")
        ts_ns: np.ndarray = ts_ser.values.astype("int64")

        result = np.zeros(len(ts_ns), dtype=np.float64)

        for i, t_ns in enumerate(ts_ns):
            # Binaere Suche: Buckets in [t - window_ns, t)
            # side='left'  → erstes Element >= Grenze
            # hi: erstes bucket >= t       → bt[0:hi] hat bucket_time < t ✓
            # lo: erstes bucket >= t-window → bt[lo:hi] ist das Fenster
            hi = int(np.searchsorted(self._bt_ns, t_ns, side="left"))
            lo = int(np.searchsorted(self._bt_ns, t_ns - window_ns, side="left"))

            if hi > lo:
                n = self._counts[lo:hi].sum()
                if n > 0:
                    raw_score = self._weighted[lo:hi].sum() / n
                    result[i] = float(np.clip(raw_score / _TONE_SCALE, -1.0, 1.0))

        return result

    def get_sentiment_at(
        self,
        symbol: str,  # reserviert fuer Multi-Symbol-Unterstuetzung
        timestamp: datetime,
    ) -> float:
        """Einzelner Zeitpunkt (live oder Debug)."""
        arr = self.get_sentiment_series(pd.Series([timestamp]))
        return float(arr[0])

    def build_feature(self, symbol: str) -> dict:
        """Protocol-kompatibel mit SentimentFeature: liefert sentiment_score."""
        score = self.get_sentiment_at(symbol, datetime.now(timezone.utc))
        return {"sentiment_score": score}
