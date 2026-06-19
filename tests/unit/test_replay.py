"""
tests/unit/test_replay.py
Unit-Tests fuer TradeReplay (Chart-Rekonstruktion).

Abgedeckt:
  TradeJournal.get_trade
    - Gibt dict fuer bekannte ID
    - Gibt None fuer unbekannte ID
    - Alle gespeicherten Felder vorhanden

  TradeReplay._parse_iso (via Hilfsfunktion)
  TradeReplay._parse_news (via Hilfsfunktion)

  get_replay_data – Grundstruktur
    - Gibt alle Pflicht-Keys zurueck
    - meta['no_lookahead'] ist immer True
    - meta['symbol'] und meta['timeframe'] korrekt
    - meta['candles_found'] == len(candles)

  get_replay_data – No-Lookahead-Garantie (Kern-Sicherheitstest)
    - Letzter Candle-Zeitstempel <= Entry-Zeitpunkt
    - Kein Candle nach Entry-Zeitpunkt in der Liste
    - Gilt auch wenn Parquet-Daten bis weit nach Entry reichen

  get_replay_data – lookback_candles
    - Liefert hoechstens lookback_candles Candles
    - Bei weniger vorhandenen Daten: gibt alle verfuegbaren zurueck

  get_replay_data – Entry/Exit-Marker
    - entry_marker enthaelt time, price, direction
    - exit_marker ist None fuer offene Trades
    - exit_marker enthaelt time und price fuer geschlossene Trades
    - entry_time in entry_marker entspricht trade.entry_time

  get_replay_data – Indikatoren
    - indicators-Dict enthaelt Nicht-OHLCV-Spalten
    - indicators-Listen haben gleiche Laenge wie candles
    - OHLCV-Spalten erscheinen NICHT in indicators

  get_replay_data – News-Events
    - news_events = [] wenn news_context leer/None
    - news_events parst JSON-Array korrekt
    - news_events parst Plain-String als Liste

  get_replay_data – OHLCV in Candles
    - Vorhandene OHLCV-Spalten erscheinen in candle-Dicts
    - Fehlende OHLCV-Spalten werden uebersprungen (kein KeyError)
    - 'time'-Key immer vorhanden

  get_replay_data – Fehlerbehandlung
    - TradeNotFoundError fuer unbekannte trade_id
    - ReplayDataNotFoundError wenn keine Parquet-Dateien existieren
    - ReplayDataNotFoundError wenn keine Daten vor Entry-Zeitpunkt
    - ReplayDataNotFoundError wenn trade keinen entry_time hat

  get_replay_data – Trade-ID-Mapping
    - Zwei verschiedene Trades mit verschiedenen Entry-Zeiten geben
      verschiedene Candle-Saetze zurueck
    - Gleiche Parquet-Datei, verschiedene Schnittmengen durch Entry-Filter

  get_replay_data – Parquet-Varianten
    - Parquet mit DatetimeIndex (timezone-aware)
    - Parquet mit DatetimeIndex (timezone-naive → wird als UTC behandelt)
    - Parquet mit 'timestamp'-Spalte statt Index
    - Mehrere Parquet-Dateien werden konkateniert und dedupliziert
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from src.journal.replay import (
    TradeReplay,
    TradeNotFoundError,
    ReplayDataNotFoundError,
    _parse_iso,
    _parse_news,
)
from src.journal.trade_journal import TradeJournal


# ─── Hilfsfunktionen ─────────────────────────────────────────────────────────

def _journal(tmp_path: Path) -> TradeJournal:
    return TradeJournal(db_path=tmp_path / "journal.db")


def _replay(tmp_path: Path, journal: TradeJournal, tf: str = "H1") -> TradeReplay:
    return TradeReplay(
        journal=journal,
        features_dir=tmp_path / "features",
        default_timeframe=tf,
    )


def _dt(year=2026, month=6, day=18, hour=12, minute=0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)


def _make_parquet(
    features_dir: Path,
    symbol: str,
    timeframe: str,
    start: datetime,
    n_rows: int,
    freq_minutes: int = 60,
    with_ohlcv: bool = True,
    use_timestamp_col: bool = False,
) -> Path:
    """
    Erstellt eine Parquet-Datei mit synthetischen Candle + Feature-Daten.

    Parameters
    ----------
    use_timestamp_col : wenn True, wird 'timestamp' als Spalte gespeichert
                        (nicht als Index) – testet den Fallback-Pfad.
    """
    features_dir.mkdir(parents=True, exist_ok=True)
    times = [start + timedelta(minutes=i * freq_minutes) for i in range(n_rows)]
    idx   = pd.DatetimeIndex(times, tz=timezone.utc)

    data: dict = {}
    if with_ohlcv:
        base = 1.0800 + 0.0001 * pd.RangeIndex(n_rows)
        data["open"]   = base.values
        data["high"]   = (base + 0.0005).values
        data["low"]    = (base - 0.0005).values
        data["close"]  = (base + 0.0002).values
        data["volume"] = [1000 + i for i in range(n_rows)]
    data["ema_20"] = [1.0800 + 0.00005 * i for i in range(n_rows)]
    data["rsi_14"] = [50.0 + 0.1 * i for i in range(n_rows)]

    df = pd.DataFrame(data, index=idx)

    if use_timestamp_col:
        df = df.reset_index().rename(columns={"index": "timestamp"})
        df.index = pd.RangeIndex(len(df))

    date_str  = start.strftime("%Y%m%d")
    file_path = features_dir / f"{symbol}_{timeframe}_{date_str}.parquet"
    df.to_parquet(file_path)
    return file_path


def _open_trade(journal: TradeJournal, entry_time: datetime, **kwargs) -> int:
    defaults = {
        "symbol":    "EURUSD",
        "direction": "buy",
        "lot_size":  1.0,
        "entry_price": 1.0850,
        "entry_time": entry_time,
    }
    defaults.update(kwargs)
    return journal.log_trade_open(defaults)


# ─────────────────────────────────────────────────────────────────────────────
#  TradeJournal.get_trade
# ─────────────────────────────────────────────────────────────────────────────

class TestGetTrade:
    def test_returns_dict_for_known_id(self, tmp_path):
        jnl = _journal(tmp_path)
        tid = _open_trade(jnl, _dt())
        trade = jnl.get_trade(tid)
        jnl.close()
        assert isinstance(trade, dict)

    def test_returns_none_for_unknown_id(self, tmp_path):
        jnl = _journal(tmp_path)
        result = jnl.get_trade(9999)
        jnl.close()
        assert result is None

    def test_id_in_result(self, tmp_path):
        jnl = _journal(tmp_path)
        tid = _open_trade(jnl, _dt())
        trade = jnl.get_trade(tid)
        jnl.close()
        assert trade["id"] == tid

    def test_symbol_in_result(self, tmp_path):
        jnl = _journal(tmp_path)
        tid = jnl.log_trade_open({"symbol": "GBPUSD", "direction": "sell", "entry_time": _dt()})
        trade = jnl.get_trade(tid)
        jnl.close()
        assert trade["symbol"] == "GBPUSD"

    def test_status_open_after_open(self, tmp_path):
        jnl = _journal(tmp_path)
        tid = _open_trade(jnl, _dt())
        assert jnl.get_trade(tid)["status"] == "open"
        jnl.close()

    def test_status_closed_after_close(self, tmp_path):
        jnl = _journal(tmp_path)
        tid = _open_trade(jnl, _dt())
        jnl.log_trade_close(tid, {"pnl": 100.0})
        assert jnl.get_trade(tid)["status"] == "closed"
        jnl.close()


# ─────────────────────────────────────────────────────────────────────────────
#  _parse_iso / _parse_news
# ─────────────────────────────────────────────────────────────────────────────

class TestHelpers:
    def test_parse_iso_utc_string(self):
        dt = _parse_iso("2026-06-18T12:00:00+00:00")
        assert dt is not None
        assert dt.tzinfo is not None
        assert dt.year == 2026

    def test_parse_iso_naive_string_gets_utc(self):
        dt = _parse_iso("2026-06-18T12:00:00")
        assert dt is not None
        assert dt.tzinfo is not None

    def test_parse_iso_none(self):
        assert _parse_iso(None) is None

    def test_parse_iso_empty_string(self):
        assert _parse_iso("") is None

    def test_parse_iso_invalid(self):
        assert _parse_iso("not-a-date") is None

    def test_parse_news_none(self):
        assert _parse_news(None) == []

    def test_parse_news_empty(self):
        assert _parse_news("") == []

    def test_parse_news_json_array(self):
        result = _parse_news('["NFP", "CPI"]')
        assert result == ["NFP", "CPI"]

    def test_parse_news_json_object(self):
        result = _parse_news('{"event": "NFP"}')
        assert isinstance(result, list) and len(result) == 1

    def test_parse_news_plain_string(self):
        result = _parse_news("NFP Heute")
        assert result == ["NFP Heute"]


# ─────────────────────────────────────────────────────────────────────────────
#  get_replay_data – Grundstruktur
# ─────────────────────────────────────────────────────────────────────────────

class TestReplayStructure:
    def setup_method(self, tmp_path):
        pass

    def _setup(self, tmp_path):
        jnl = _journal(tmp_path)
        start = _dt(hour=0)
        _make_parquet(tmp_path / "features", "EURUSD", "H1", start, n_rows=200)
        entry = _dt(hour=10)
        tid = _open_trade(jnl, entry)
        return jnl, tid, tmp_path

    def test_required_keys_present(self, tmp_path):
        jnl, tid, tp = self._setup(tmp_path)
        rp = _replay(tp, jnl)
        result = rp.get_replay_data(tid)
        jnl.close()
        for key in ("trade", "candles", "entry_marker", "exit_marker",
                    "indicators", "news_events", "meta"):
            assert key in result, f"Key '{key}' fehlt im Ergebnis"

    def test_meta_no_lookahead_always_true(self, tmp_path):
        jnl, tid, tp = self._setup(tmp_path)
        rp = _replay(tp, jnl)
        result = rp.get_replay_data(tid)
        jnl.close()
        assert result["meta"]["no_lookahead"] is True

    def test_meta_symbol(self, tmp_path):
        jnl, tid, tp = self._setup(tmp_path)
        rp = _replay(tp, jnl)
        result = rp.get_replay_data(tid)
        jnl.close()
        assert result["meta"]["symbol"] == "EURUSD"

    def test_meta_timeframe(self, tmp_path):
        jnl, tid, tp = self._setup(tmp_path)
        rp = _replay(tp, jnl)
        result = rp.get_replay_data(tid)
        jnl.close()
        assert result["meta"]["timeframe"] == "H1"

    def test_meta_candles_found_matches_candles(self, tmp_path):
        jnl, tid, tp = self._setup(tmp_path)
        rp = _replay(tp, jnl)
        result = rp.get_replay_data(tid)
        jnl.close()
        assert result["meta"]["candles_found"] == len(result["candles"])

    def test_trade_dict_in_result(self, tmp_path):
        jnl, tid, tp = self._setup(tmp_path)
        rp = _replay(tp, jnl)
        result = rp.get_replay_data(tid)
        jnl.close()
        assert result["trade"]["id"] == tid
        assert result["trade"]["symbol"] == "EURUSD"

    def test_news_events_is_list(self, tmp_path):
        jnl, tid, tp = self._setup(tmp_path)
        rp = _replay(tp, jnl)
        result = rp.get_replay_data(tid)
        jnl.close()
        assert isinstance(result["news_events"], list)


# ─────────────────────────────────────────────────────────────────────────────
#  No-Lookahead-Garantie (Kern-Sicherheitstest)
# ─────────────────────────────────────────────────────────────────────────────

class TestNoLookahead:
    def test_last_candle_at_or_before_entry(self, tmp_path):
        """
        Kein Candle darf nach dem Entry-Zeitpunkt liegen.
        Das ist der zentrale Korrektheitstest fuer die Replay-Funktion.
        """
        jnl = _journal(tmp_path)
        # Parquet: 200 stündliche Candles ab Mitternacht
        start = _dt(hour=0)
        _make_parquet(tmp_path / "features", "EURUSD", "H1", start, n_rows=200)
        # Entry bei 10:00 Uhr
        entry = _dt(hour=10)
        tid = _open_trade(jnl, entry)

        rp = _replay(tmp_path, jnl)
        result = rp.get_replay_data(tid)
        jnl.close()

        candles = result["candles"]
        assert len(candles) > 0
        for c in candles:
            candle_time = datetime.fromisoformat(c["time"])
            if candle_time.tzinfo is None:
                candle_time = candle_time.replace(tzinfo=timezone.utc)
            assert candle_time <= entry, (
                f"Candle nach Entry! {candle_time.isoformat()} > {entry.isoformat()}"
            )

    def test_no_candle_after_entry_even_with_future_data(self, tmp_path):
        """
        Wenn Parquet-Daten weit nach Entry reichen (z.B. bis heute),
        darf nichts nach Entry in die Candles.
        """
        jnl = _journal(tmp_path)
        # 500 stündliche Candles: viele davon nach Entry
        start = _dt(hour=0)
        _make_parquet(tmp_path / "features", "EURUSD", "H1", start, n_rows=500)
        entry = _dt(hour=5)   # Entry nach 5h -> danach liegen ~495 Candles
        tid = _open_trade(jnl, entry)

        rp = _replay(tmp_path, jnl)
        result = rp.get_replay_data(tid, lookback_candles=200)
        jnl.close()

        for c in result["candles"]:
            ct = datetime.fromisoformat(c["time"])
            if ct.tzinfo is None:
                ct = ct.replace(tzinfo=timezone.utc)
            assert ct <= entry

    def test_meta_entry_time_matches_trade(self, tmp_path):
        jnl = _journal(tmp_path)
        _make_parquet(tmp_path / "features", "EURUSD", "H1", _dt(hour=0), n_rows=50)
        entry = _dt(hour=8)
        tid = _open_trade(jnl, entry)
        rp = _replay(tmp_path, jnl)
        result = rp.get_replay_data(tid)
        jnl.close()
        meta_entry = datetime.fromisoformat(result["meta"]["entry_time"])
        if meta_entry.tzinfo is None:
            meta_entry = meta_entry.replace(tzinfo=timezone.utc)
        assert meta_entry == entry


# ─────────────────────────────────────────────────────────────────────────────
#  lookback_candles
# ─────────────────────────────────────────────────────────────────────────────

class TestLookbackCandles:
    def test_at_most_lookback_candles_returned(self, tmp_path):
        jnl = _journal(tmp_path)
        _make_parquet(tmp_path / "features", "EURUSD", "H1", _dt(hour=0), n_rows=200)
        tid = _open_trade(jnl, _dt(hour=23))
        rp = _replay(tmp_path, jnl)
        result = rp.get_replay_data(tid, lookback_candles=10)
        jnl.close()
        assert len(result["candles"]) <= 10

    def test_exactly_lookback_when_enough_data(self, tmp_path):
        jnl = _journal(tmp_path)
        _make_parquet(tmp_path / "features", "EURUSD", "H1", _dt(hour=0), n_rows=200)
        tid = _open_trade(jnl, _dt(hour=23))
        rp = _replay(tmp_path, jnl)
        result = rp.get_replay_data(tid, lookback_candles=5)
        jnl.close()
        assert len(result["candles"]) == 5

    def test_fewer_candles_when_not_enough_data(self, tmp_path):
        """Wenn weniger Daten als lookback_candles vorhanden: gibt alle zurueck."""
        jnl = _journal(tmp_path)
        # Parquet: 5 stündliche Candles ab 10:00 → 10:00, 11:00, 12:00, 13:00, 14:00
        start = _dt(hour=10)
        _make_parquet(tmp_path / "features", "EURUSD", "H1", start, n_rows=5)
        # Entry bei 12:30 → Candles 10:00, 11:00, 12:00 liegen <= 12:30 (3 Stück)
        entry = _dt(hour=12, minute=30)
        tid = _open_trade(jnl, entry)
        rp = _replay(tmp_path, jnl)
        result = rp.get_replay_data(tid, lookback_candles=100)
        jnl.close()
        assert len(result["candles"]) == 3

    def test_default_lookback_is_100(self, tmp_path):
        jnl = _journal(tmp_path)
        _make_parquet(tmp_path / "features", "EURUSD", "H1", _dt(hour=0), n_rows=200)
        tid = _open_trade(jnl, _dt(hour=23))
        rp = _replay(tmp_path, jnl)
        result = rp.get_replay_data(tid)
        jnl.close()
        assert result["meta"]["lookback_candles"] == 100


# ─────────────────────────────────────────────────────────────────────────────
#  Entry / Exit Marker
# ─────────────────────────────────────────────────────────────────────────────

class TestMarkers:
    def test_entry_marker_keys(self, tmp_path):
        jnl = _journal(tmp_path)
        _make_parquet(tmp_path / "features", "EURUSD", "H1", _dt(hour=0), n_rows=50)
        tid = _open_trade(jnl, _dt(hour=10))
        rp = _replay(tmp_path, jnl)
        result = rp.get_replay_data(tid)
        jnl.close()
        marker = result["entry_marker"]
        assert "time" in marker
        assert "price" in marker
        assert "direction" in marker

    def test_entry_marker_direction(self, tmp_path):
        jnl = _journal(tmp_path)
        _make_parquet(tmp_path / "features", "EURUSD", "H1", _dt(hour=0), n_rows=50)
        tid = jnl.log_trade_open({
            "symbol": "EURUSD", "direction": "sell",
            "entry_time": _dt(hour=10),
        })
        rp = _replay(tmp_path, jnl)
        result = rp.get_replay_data(tid)
        jnl.close()
        assert result["entry_marker"]["direction"] == "sell"

    def test_entry_marker_price(self, tmp_path):
        jnl = _journal(tmp_path)
        _make_parquet(tmp_path / "features", "EURUSD", "H1", _dt(hour=0), n_rows=50)
        tid = jnl.log_trade_open({
            "symbol": "EURUSD", "direction": "buy",
            "entry_price": 1.08750,
            "entry_time": _dt(hour=10),
        })
        rp = _replay(tmp_path, jnl)
        result = rp.get_replay_data(tid)
        jnl.close()
        assert result["entry_marker"]["price"] == pytest.approx(1.08750)

    def test_exit_marker_none_for_open_trade(self, tmp_path):
        jnl = _journal(tmp_path)
        _make_parquet(tmp_path / "features", "EURUSD", "H1", _dt(hour=0), n_rows=50)
        tid = _open_trade(jnl, _dt(hour=10))
        rp = _replay(tmp_path, jnl)
        result = rp.get_replay_data(tid)
        jnl.close()
        assert result["exit_marker"] is None

    def test_exit_marker_present_for_closed_trade(self, tmp_path):
        jnl = _journal(tmp_path)
        _make_parquet(tmp_path / "features", "EURUSD", "H1", _dt(hour=0), n_rows=50)
        tid = _open_trade(jnl, _dt(hour=10))
        jnl.log_trade_close(tid, {
            "exit_price": 1.09000,
            "exit_time": _dt(hour=14),
            "pnl": 150.0,
        })
        rp = _replay(tmp_path, jnl)
        result = rp.get_replay_data(tid)
        jnl.close()
        marker = result["exit_marker"]
        assert marker is not None
        assert "time" in marker
        assert marker["price"] == pytest.approx(1.09000)


# ─────────────────────────────────────────────────────────────────────────────
#  Indikatoren
# ─────────────────────────────────────────────────────────────────────────────

class TestIndicators:
    def _setup(self, tmp_path):
        jnl = _journal(tmp_path)
        _make_parquet(tmp_path / "features", "EURUSD", "H1", _dt(hour=0), n_rows=50)
        tid = _open_trade(jnl, _dt(hour=10))
        rp = _replay(tmp_path, jnl)
        result = rp.get_replay_data(tid, lookback_candles=5)
        jnl.close()
        return result

    def test_indicators_is_dict(self, tmp_path):
        result = self._setup(tmp_path)
        assert isinstance(result["indicators"], dict)

    def test_ema_20_in_indicators(self, tmp_path):
        result = self._setup(tmp_path)
        assert "ema_20" in result["indicators"]

    def test_rsi_14_in_indicators(self, tmp_path):
        result = self._setup(tmp_path)
        assert "rsi_14" in result["indicators"]

    def test_ohlcv_not_in_indicators(self, tmp_path):
        result = self._setup(tmp_path)
        for col in ("open", "high", "low", "close", "volume"):
            assert col not in result["indicators"]

    def test_indicator_length_matches_candles(self, tmp_path):
        result = self._setup(tmp_path)
        n_candles = len(result["candles"])
        for col, vals in result["indicators"].items():
            assert len(vals) == n_candles, (
                f"Indikator '{col}' hat {len(vals)} Werte, erwartet {n_candles}"
            )

    def test_indicator_values_are_floats_or_none(self, tmp_path):
        result = self._setup(tmp_path)
        for vals in result["indicators"].values():
            for v in vals:
                assert v is None or isinstance(v, float)


# ─────────────────────────────────────────────────────────────────────────────
#  OHLCV in Candles
# ─────────────────────────────────────────────────────────────────────────────

class TestCandleFormat:
    def test_time_key_always_present(self, tmp_path):
        jnl = _journal(tmp_path)
        _make_parquet(tmp_path / "features", "EURUSD", "H1", _dt(hour=0), n_rows=20)
        tid = _open_trade(jnl, _dt(hour=10))
        rp = _replay(tmp_path, jnl)
        result = rp.get_replay_data(tid)
        jnl.close()
        for c in result["candles"]:
            assert "time" in c

    def test_ohlcv_present_when_in_parquet(self, tmp_path):
        jnl = _journal(tmp_path)
        _make_parquet(tmp_path / "features", "EURUSD", "H1", _dt(hour=0),
                      n_rows=20, with_ohlcv=True)
        tid = _open_trade(jnl, _dt(hour=10))
        rp = _replay(tmp_path, jnl)
        result = rp.get_replay_data(tid, lookback_candles=5)
        jnl.close()
        for c in result["candles"]:
            assert "open" in c
            assert "close" in c

    def test_no_ohlcv_when_not_in_parquet(self, tmp_path):
        """Wenn Parquet nur Features hat (kein OHLCV), kein KeyError."""
        jnl = _journal(tmp_path)
        _make_parquet(tmp_path / "features", "EURUSD", "H1", _dt(hour=0),
                      n_rows=20, with_ohlcv=False)
        tid = _open_trade(jnl, _dt(hour=10))
        rp = _replay(tmp_path, jnl)
        result = rp.get_replay_data(tid, lookback_candles=5)
        jnl.close()
        for c in result["candles"]:
            assert "time" in c
            assert "open" not in c  # nicht im Parquet, also nicht im Candle

    def test_candle_values_are_floats_or_none(self, tmp_path):
        jnl = _journal(tmp_path)
        _make_parquet(tmp_path / "features", "EURUSD", "H1", _dt(hour=0), n_rows=20)
        tid = _open_trade(jnl, _dt(hour=10))
        rp = _replay(tmp_path, jnl)
        result = rp.get_replay_data(tid, lookback_candles=5)
        jnl.close()
        for c in result["candles"]:
            for k, v in c.items():
                if k != "time":
                    assert v is None or isinstance(v, float)


# ─────────────────────────────────────────────────────────────────────────────
#  News-Events
# ─────────────────────────────────────────────────────────────────────────────

class TestNewsEvents:
    def _trade_with_news(self, jnl, news_context):
        return jnl.log_trade_open({
            "symbol": "EURUSD", "direction": "buy",
            "entry_time": _dt(hour=10),
            "news_context": news_context,
        })

    def test_empty_when_no_news(self, tmp_path):
        jnl = _journal(tmp_path)
        _make_parquet(tmp_path / "features", "EURUSD", "H1", _dt(hour=0), n_rows=50)
        tid = _open_trade(jnl, _dt(hour=10))
        rp = _replay(tmp_path, jnl)
        result = rp.get_replay_data(tid)
        jnl.close()
        assert result["news_events"] == []

    def test_json_array_parsed(self, tmp_path):
        jnl = _journal(tmp_path)
        _make_parquet(tmp_path / "features", "EURUSD", "H1", _dt(hour=0), n_rows=50)
        tid = self._trade_with_news(jnl, '["NFP", "CPI", "FOMC"]')
        rp = _replay(tmp_path, jnl)
        result = rp.get_replay_data(tid)
        jnl.close()
        assert result["news_events"] == ["NFP", "CPI", "FOMC"]

    def test_plain_string_wrapped_in_list(self, tmp_path):
        jnl = _journal(tmp_path)
        _make_parquet(tmp_path / "features", "EURUSD", "H1", _dt(hour=0), n_rows=50)
        tid = self._trade_with_news(jnl, "NFP Heute 14:30 UTC")
        rp = _replay(tmp_path, jnl)
        result = rp.get_replay_data(tid)
        jnl.close()
        assert result["news_events"] == ["NFP Heute 14:30 UTC"]


# ─────────────────────────────────────────────────────────────────────────────
#  Fehlerbehandlung
# ─────────────────────────────────────────────────────────────────────────────

class TestErrorHandling:
    def test_trade_not_found_error(self, tmp_path):
        jnl = _journal(tmp_path)
        rp = _replay(tmp_path, jnl)
        with pytest.raises(TradeNotFoundError):
            rp.get_replay_data(9999)
        jnl.close()

    def test_no_parquet_files_raises(self, tmp_path):
        jnl = _journal(tmp_path)
        tid = _open_trade(jnl, _dt(hour=10))
        # kein Parquet erstellt
        rp = _replay(tmp_path, jnl)
        with pytest.raises(ReplayDataNotFoundError, match="Parquet"):
            rp.get_replay_data(tid)
        jnl.close()

    def test_no_data_before_entry_raises(self, tmp_path):
        jnl = _journal(tmp_path)
        # Parquet beginnt NACH dem Entry-Zeitpunkt
        late_start = _dt(hour=15)
        _make_parquet(tmp_path / "features", "EURUSD", "H1", late_start, n_rows=20)
        early_entry = _dt(hour=10)  # vor allen Daten
        tid = _open_trade(jnl, early_entry)
        rp = _replay(tmp_path, jnl)
        with pytest.raises(ReplayDataNotFoundError):
            rp.get_replay_data(tid)
        jnl.close()

    def test_trade_without_entry_time_raises(self, tmp_path):
        jnl = _journal(tmp_path)
        # Trade ohne entry_time speichern (wird automatisch gesetzt, daher forcieren)
        import sqlite3
        db = tmp_path / "journal.db"
        jnl.close()
        conn = sqlite3.connect(str(db))
        conn.execute(
            "INSERT INTO trades (symbol, direction, status, entry_time) "
            "VALUES ('EURUSD', 'buy', 'open', NULL)"
        )
        conn.commit()
        trade_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.close()
        jnl2 = TradeJournal(db_path=db)
        rp = _replay(tmp_path, jnl2)
        with pytest.raises(ReplayDataNotFoundError):
            rp.get_replay_data(trade_id)
        jnl2.close()


# ─────────────────────────────────────────────────────────────────────────────
#  Trade-ID-Mapping: verschiedene Trades → verschiedene Candle-Saetze
# ─────────────────────────────────────────────────────────────────────────────

class TestTradeIdMapping:
    def test_different_trades_different_candle_sets(self, tmp_path):
        """
        Zwei Trades mit verschiedenen Entry-Zeiten sehen unterschiedliche
        Candle-Mengen in denselben Parquet-Daten.
        """
        jnl = _journal(tmp_path)
        _make_parquet(tmp_path / "features", "EURUSD", "H1", _dt(hour=0), n_rows=200)

        entry_early = _dt(hour=5)
        entry_late  = _dt(hour=20)
        tid1 = _open_trade(jnl, entry_early)
        tid2 = _open_trade(jnl, entry_late)

        rp = _replay(tmp_path, jnl)
        r1 = rp.get_replay_data(tid1, lookback_candles=200)
        r2 = rp.get_replay_data(tid2, lookback_candles=200)
        jnl.close()

        # Trade 2 hat mehr Candles (späterer Entry)
        assert len(r2["candles"]) > len(r1["candles"])

    def test_last_candle_time_differs_between_trades(self, tmp_path):
        """Letzter Candle-Zeitstempel ist verschieden fuer verschiedene Entries."""
        jnl = _journal(tmp_path)
        _make_parquet(tmp_path / "features", "EURUSD", "H1", _dt(hour=0), n_rows=200)

        tid1 = _open_trade(jnl, _dt(hour=5))
        tid2 = _open_trade(jnl, _dt(hour=10))

        rp = _replay(tmp_path, jnl)
        r1 = rp.get_replay_data(tid1, lookback_candles=200)
        r2 = rp.get_replay_data(tid2, lookback_candles=200)
        jnl.close()

        t1_last = r1["candles"][-1]["time"]
        t2_last = r2["candles"][-1]["time"]
        assert t1_last != t2_last


# ─────────────────────────────────────────────────────────────────────────────
#  Parquet-Varianten
# ─────────────────────────────────────────────────────────────────────────────

class TestParquetVariants:
    def test_datetime_index_tz_aware(self, tmp_path):
        """Standard-Pfad: timezone-aware DatetimeIndex."""
        jnl = _journal(tmp_path)
        _make_parquet(tmp_path / "features", "EURUSD", "H1", _dt(hour=0),
                      n_rows=50, use_timestamp_col=False)
        tid = _open_trade(jnl, _dt(hour=10))
        rp = _replay(tmp_path, jnl)
        result = rp.get_replay_data(tid, lookback_candles=5)
        jnl.close()
        assert len(result["candles"]) > 0

    def test_timestamp_column_fallback(self, tmp_path):
        """Fallback: 'timestamp'-Spalte statt DatetimeIndex."""
        jnl = _journal(tmp_path)
        _make_parquet(tmp_path / "features", "EURUSD", "H1", _dt(hour=0),
                      n_rows=50, use_timestamp_col=True)
        tid = _open_trade(jnl, _dt(hour=10))
        rp = _replay(tmp_path, jnl)
        result = rp.get_replay_data(tid, lookback_candles=5)
        jnl.close()
        assert len(result["candles"]) > 0

    def test_multiple_parquet_files_combined(self, tmp_path):
        """Mehrere Parquet-Dateien werden konkateniert."""
        jnl = _journal(tmp_path)
        feat_dir = tmp_path / "features"
        # Datei 1: June 15 00:00, 100 Stunden → endet June 19 03:00
        _make_parquet(feat_dir, "EURUSD", "H1", _dt(year=2026, month=6, day=15, hour=0), n_rows=100)
        # Datei 2: June 19 04:00, 100 Stunden → kein Ueberlapp mit Datei 1
        _make_parquet(feat_dir, "EURUSD", "H1", _dt(year=2026, month=6, day=19, hour=4), n_rows=100)
        # Entry June 21 12:00 → Datei1: 100 Candles, Datei2: 57 Candles → 157 > 100
        entry = _dt(year=2026, month=6, day=21, hour=12)
        tid = _open_trade(jnl, entry)
        rp = TradeReplay(journal=jnl, features_dir=feat_dir)
        result = rp.get_replay_data(tid, lookback_candles=200)
        jnl.close()
        # Daten aus beiden Dateien: 100 + 57 = 157 Candles vor Entry
        assert len(result["candles"]) > 100

    def test_custom_timeframe(self, tmp_path):
        """Eigener Timeframe-Parameter wird fuer Dateisuche genutzt."""
        jnl = _journal(tmp_path)
        _make_parquet(tmp_path / "features", "EURUSD", "H4", _dt(hour=0), n_rows=50)
        tid = _open_trade(jnl, _dt(hour=10))
        rp = TradeReplay(journal=jnl, features_dir=tmp_path / "features",
                         default_timeframe="H4")
        result = rp.get_replay_data(tid, timeframe="H4")
        jnl.close()
        assert result["meta"]["timeframe"] == "H4"

    def test_wrong_timeframe_raises(self, tmp_path):
        """Parquet fuer H4 vorhanden, Anfrage nach H1 → Fehler."""
        jnl = _journal(tmp_path)
        _make_parquet(tmp_path / "features", "EURUSD", "H4", _dt(hour=0), n_rows=50)
        tid = _open_trade(jnl, _dt(hour=10))
        rp = _replay(tmp_path, jnl)   # default H1
        with pytest.raises(ReplayDataNotFoundError):
            rp.get_replay_data(tid, timeframe="H1")
        jnl.close()
