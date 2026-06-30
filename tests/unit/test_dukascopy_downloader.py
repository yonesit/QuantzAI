"""
tests/unit/test_dukascopy_downloader.py
Unit-Tests fuer den Dukascopy-Downloader und die M15/Spread-Validierung.

Kein Netzwerk: HTTP wird ueber eine Fake-Session injiziert. Alle Dateien
landen in tmp_path. Es werden KEINE echten paper_trades.json o.ae. beruehrt.
"""

from __future__ import annotations

import lzma
import struct
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src.data.dukascopy_downloader import (
    DukascopyDownloader,
    DukascopyError,
    DownloadStats,
    decode_bi5,
    ticks_to_m15,
    SYMBOL_POINT_DIVISOR,
)
from src.data.dukascopy_validator import validate_dukascopy_m15

_TICK = struct.Struct(">3i2f")


# ─────────────────────────────────────────────
#  Hilfsfunktionen: synthetische .bi5-Bytes
# ─────────────────────────────────────────────

def _make_bi5(ticks: list[tuple[int, int, int, float, float]]) -> bytes:
    """Baut LZMA-komprimierte .bi5-Bytes aus (ms, ask_int, bid_int, askvol, bidvol)."""
    raw = b"".join(_TICK.pack(*t) for t in ticks)
    return lzma.compress(raw)


def _eurusd_ticks() -> list[tuple[int, int, int, float, float]]:
    # divisor 1e5: ask 1.08513 -> 108513 ; bid 1.08512 -> 108512
    return [
        (0,           108513, 108512, 1.0, 1.0),   # 10:00:00.000
        (60_000,      108520, 108518, 2.0, 1.0),   # 10:01:00
        (15 * 60_000, 108540, 108538, 1.0, 3.0),   # 10:15:00 -> naechster Bar
    ]


# ─────────────────────────────────────────────
#  decode_bi5
# ─────────────────────────────────────────────

def test_decode_bi5_scaling_and_timestamps():
    hour = datetime(2024, 3, 5, 10, tzinfo=timezone.utc)
    raw = _make_bi5(_eurusd_ticks())
    df = decode_bi5(raw, hour, SYMBOL_POINT_DIVISOR["EURUSD"])

    assert list(df.columns) == ["timestamp", "bid", "ask", "bid_volume", "ask_volume"]
    assert len(df) == 3
    # Skalierung korrekt
    assert df["ask"].iloc[0] == pytest.approx(1.08513, abs=1e-9)
    assert df["bid"].iloc[0] == pytest.approx(1.08512, abs=1e-9)
    # ask >= bid
    assert (df["ask"] >= df["bid"]).all()
    # Timestamps relativ zur Stunde
    assert df["timestamp"].iloc[0] == pd.Timestamp("2024-03-05 10:00:00", tz="UTC")
    assert df["timestamp"].iloc[2] == pd.Timestamp("2024-03-05 10:15:00", tz="UTC")


def test_decode_bi5_empty_returns_empty_frame():
    hour = datetime(2024, 3, 5, 10, tzinfo=timezone.utc)
    df = decode_bi5(b"", hour, 1e5)
    assert df.empty
    assert "bid" in df.columns


def test_decode_bi5_corrupt_length_raises():
    # Gueltiges LZMA, aber Laenge kein Vielfaches von 20
    bad = lzma.compress(b"12345")
    with pytest.raises(DukascopyError):
        decode_bi5(bad, datetime(2024, 1, 1, tzinfo=timezone.utc), 1e5)


# ─────────────────────────────────────────────
#  ticks_to_m15
# ─────────────────────────────────────────────

def test_ticks_to_m15_aggregation_and_spread():
    hour = datetime(2024, 3, 5, 10, tzinfo=timezone.utc)
    ticks = decode_bi5(_make_bi5(_eurusd_ticks()), hour, SYMBOL_POINT_DIVISOR["EURUSD"])
    bars = ticks_to_m15(ticks)

    # Zwei Bars: 10:00 (zwei Ticks) und 10:15 (ein Tick)
    assert len(bars) == 2
    bar0 = bars.iloc[0]
    # mid open des ersten Ticks = (1.08513+1.08512)/2
    assert bar0["open"] == pytest.approx((1.08513 + 1.08512) / 2, abs=1e-9)
    assert bar0["tick_count"] == 2
    # Spread des ersten Bars: mean von (0.00001, 0.00002)
    assert bar0["spread_mean"] == pytest.approx((0.00001 + 0.00002) / 2, abs=1e-7)
    assert {"spread_mean", "spread_median", "tick_count"}.issubset(bars.columns)


def test_ticks_to_m15_empty():
    bars = ticks_to_m15(pd.DataFrame(
        {"timestamp": pd.Series([], dtype="datetime64[ns, UTC]"),
         "bid": [], "ask": [], "bid_volume": [], "ask_volume": []}))
    assert bars.empty


# ─────────────────────────────────────────────
#  Fake-Session fuer Netzwerk-Tests
# ─────────────────────────────────────────────

class _Resp:
    def __init__(self, status_code, content=b""):
        self.status_code = status_code
        self.content = content


class _FakeSession:
    """Liefert vordefinierte Antworten je URL; zaehlt Aufrufe."""
    def __init__(self, responses: dict[str, list]):
        self.responses = responses        # url -> list of _Resp (nacheinander)
        self.headers = {}
        self.calls: list[str] = []

    def get(self, url, timeout=None):
        self.calls.append(url)
        queue = self.responses.get(url)
        if not queue:
            return _Resp(404)
        resp = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(resp, Exception):
            raise resp
        return resp


def _noop_sleep(_):
    pass


# ─────────────────────────────────────────────
#  fetch_hour_raw: 404 / Retry / endgueltiger Fehler
# ─────────────────────────────────────────────

def test_hour_url_month_is_zero_based():
    url = DukascopyDownloader.hour_url("EURUSD", datetime(2024, 1, 5, 9, tzinfo=timezone.utc))
    assert "/2024/00/05/09h_ticks.bi5" in url   # Januar -> 00


def test_fetch_hour_404_returns_none(tmp_path):
    sess = _FakeSession({})
    dl = DukascopyDownloader(tmp_path, session=sess, sleep_func=_noop_sleep)
    out = dl.fetch_hour_raw("EURUSD", datetime(2024, 3, 5, 3, tzinfo=timezone.utc))
    assert out is None


def test_fetch_hour_retries_then_succeeds(tmp_path):
    hour = datetime(2024, 3, 5, 10, tzinfo=timezone.utc)
    url = DukascopyDownloader.hour_url("EURUSD", hour)
    blob = _make_bi5(_eurusd_ticks())
    sess = _FakeSession({url: [_Resp(503), _Resp(503), _Resp(200, blob)]})
    dl = DukascopyDownloader(tmp_path, session=sess, sleep_func=_noop_sleep, max_retries=4)
    out = dl.fetch_hour_raw("EURUSD", hour)
    assert out == blob
    assert len(sess.calls) == 3


def test_fetch_hour_exhausts_retries_raises(tmp_path):
    hour = datetime(2024, 3, 5, 10, tzinfo=timezone.utc)
    url = DukascopyDownloader.hour_url("EURUSD", hour)
    sess = _FakeSession({url: [_Resp(503)]})
    dl = DukascopyDownloader(tmp_path, session=sess, sleep_func=_noop_sleep, max_retries=3)
    with pytest.raises(DukascopyError):
        dl.fetch_hour_raw("EURUSD", hour)


# ─────────────────────────────────────────────
#  download_range: Cache, Resume, Assemble
# ─────────────────────────────────────────────

def _single_day_session(symbol, day, hour_with_data=10):
    """Fake-Session, die nur fuer eine Stunde des Tages Daten liefert."""
    hour = day.replace(hour=hour_with_data)
    url = DukascopyDownloader.hour_url(symbol, hour)
    blob = _make_bi5(_eurusd_ticks())
    return _FakeSession({url: [_Resp(200, blob)]}), url


def test_download_range_writes_cache_and_assembles(tmp_path):
    day = datetime(2024, 3, 5, tzinfo=timezone.utc)   # Dienstag
    sess, url = _single_day_session("EURUSD", day)
    dl = DukascopyDownloader(tmp_path, session=sess, sleep_func=_noop_sleep, rate_limit_s=0.0)

    df, stats = dl.download_range("EURUSD", day, day)
    assert not df.empty
    assert stats.days_fetched == 1
    assert stats.bars_total == len(df)
    # Cache-Datei existiert
    cache = tmp_path / "EURUSD" / "2024-03-05.parquet"
    assert cache.exists()


def test_download_range_resume_skips_existing(tmp_path):
    day = datetime(2024, 3, 5, tzinfo=timezone.utc)
    sess, url = _single_day_session("EURUSD", day)
    dl = DukascopyDownloader(tmp_path, session=sess, sleep_func=_noop_sleep, rate_limit_s=0.0)

    dl.download_range("EURUSD", day, day)
    calls_first = len(sess.calls)
    assert calls_first > 0

    # Zweiter Lauf: Cache existiert -> keine weiteren HTTP-Calls
    df2, stats2 = dl.download_range("EURUSD", day, day)
    assert len(sess.calls) == calls_first      # keine neuen Requests
    assert stats2.days_cached == 1
    assert not df2.empty


class _AlwaysThrottleSession:
    """Liefert fuer JEDE URL dauerhaft 503 (Drosselung)."""
    def __init__(self):
        self.headers = {}
        self.calls = []

    def get(self, url, timeout=None):
        self.calls.append(url)
        return _Resp(503)


def test_download_range_failed_day_not_cached_and_continues(tmp_path):
    day = datetime(2024, 3, 5, tzinfo=timezone.utc)   # Dienstag
    sess = _AlwaysThrottleSession()
    dl = DukascopyDownloader(
        tmp_path, session=sess, sleep_func=_noop_sleep,
        rate_limit_s=0.0, max_retries=2, day_cooldown_s=0.0,
    )
    df, stats = dl.download_range("EURUSD", day, day)
    # Lauf bricht NICHT ab, aber Tag ist fehlgeschlagen
    assert stats.days_failed == 1
    assert df.empty
    # Fehlgeschlagener Tag wurde NICHT gecacht -> naechster Lauf versucht erneut
    assert not (tmp_path / "EURUSD" / "2024-03-05.parquet").exists()


def test_fill_cache_range_writes_cache_without_assembling(tmp_path):
    day = datetime(2024, 3, 5, tzinfo=timezone.utc)
    sess, url = _single_day_session("EURUSD", day)
    dl = DukascopyDownloader(tmp_path, session=sess, sleep_func=_noop_sleep, rate_limit_s=0.0)

    stats = dl.fill_cache_range("EURUSD", day, day)
    # Rueckgabe sind nur Stats (kein DataFrame); Cache-Datei ist geschrieben
    assert isinstance(stats, DownloadStats)
    assert stats.days_fetched == 1
    assert (tmp_path / "EURUSD" / "2024-03-05.parquet").exists()
    # assemble_from_cache liefert dieselben Bars wie download_range
    df = dl.assemble_from_cache("EURUSD")
    assert not df.empty


def test_download_range_skips_saturday(tmp_path):
    sat = datetime(2024, 3, 9, tzinfo=timezone.utc)   # Samstag
    sess = _FakeSession({})
    dl = DukascopyDownloader(tmp_path, session=sess, sleep_func=_noop_sleep, rate_limit_s=0.0)
    df, stats = dl.download_range("EURUSD", sat, sat, skip_saturday=True)
    # Samstag uebersprungen -> keine HTTP-Calls
    assert sess.calls == []
    assert stats.days_total == 0


# ─────────────────────────────────────────────
#  Validierung
# ─────────────────────────────────────────────

def _clean_m15(n=200, start="2024-03-05 00:00", spread=0.00002):
    idx = pd.date_range(start, periods=n, freq="15min", tz="UTC")
    rng = np.random.default_rng(0)
    close = 1.10 + np.cumsum(rng.normal(0, 0.0002, n))
    return pd.DataFrame({
        "timestamp": idx,
        "open": close, "high": close + 0.0003, "low": close - 0.0003, "close": close,
        "volume": 100.0, "spread_mean": spread, "spread_median": spread, "tick_count": 50,
    })


def test_validate_detects_duplicates_and_ohlc():
    df = _clean_m15(50)
    # Duplikat einfuegen
    df = pd.concat([df, df.iloc[[10]]], ignore_index=True)
    # OHLC-Verletzung: high < close
    df.loc[5, "high"] = df.loc[5, "low"] - 1.0
    rep = validate_dukascopy_m15(df, "EURUSD")
    assert rep.duplicates >= 1
    assert rep.ohlc_violations >= 1


def test_validate_weekend_gap_marked_not_intra(tmp_path):
    # Bar bis Freitag 20:45, naechster Bar Sonntag 21:00 -> Wochenende
    fri = pd.date_range("2024-03-08 20:00", periods=4, freq="15min", tz="UTC")  # Fr
    sun = pd.date_range("2024-03-10 21:00", periods=4, freq="15min", tz="UTC")  # So
    idx = fri.append(sun)
    close = np.linspace(1.10, 1.11, len(idx))
    df = pd.DataFrame({
        "timestamp": idx, "open": close, "high": close + 0.0002,
        "low": close - 0.0002, "close": close, "volume": 10.0,
        "spread_mean": 0.00002, "spread_median": 0.00002, "tick_count": 10,
    })
    rep = validate_dukascopy_m15(df, "EURUSD")
    assert rep.weekend_gaps > 0
    assert rep.intra_session_missing == 0


def test_validate_intra_session_gap_listed():
    df = _clean_m15(20, start="2024-03-05 10:00")
    # Mitten am Dienstag eine Luecke: Zeilen 5..8 entfernen
    df = df.drop(index=[5, 6, 7, 8]).reset_index(drop=True)
    rep = validate_dukascopy_m15(df, "EURUSD")
    assert rep.intra_session_missing >= 4
    assert len(rep.largest_gaps) >= 1


def test_validate_spread_stats_and_negative_flag():
    df = _clean_m15(60, spread=0.00003)
    df.loc[3, "spread_mean"] = -0.00001   # negativer Spread
    rep = validate_dukascopy_m15(df, "EURUSD")
    assert rep.negative_spread_bars == 1
    # 0.00003 / pip(0.0001) = 0.3 pips
    assert rep.spread_pips_median == pytest.approx(0.3, abs=0.05)
    assert "London" in rep.session_spread_pips_median or "NewYork" in rep.session_spread_pips_median


def test_validate_report_roundtrip_json(tmp_path):
    rep = validate_dukascopy_m15(_clean_m15(40), "XAUUSD")
    path = rep.save(tmp_path)
    assert path.exists()
    import json
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["symbol"] == "XAUUSD"
    assert "spread_pips_median" in data
    # Markdown rendert ohne Fehler
    assert "XAUUSD" in rep.to_markdown()
