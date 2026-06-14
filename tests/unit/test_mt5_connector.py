"""
Unit-Tests fuer MT5Connector – Python 3.14 kompatibel.

Strategie:
  - Jede Mock-Funktion wird als frisches MagicMock() neu erstellt (kein reset_mock)
  - _mod._MT5_MODULE wird vor jedem Test neu gesetzt
  - connected_connector setzt _connected direkt, ruft nie connect() auf
"""

from __future__ import annotations

import sys
import types
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Stub-Fabrik – erzeugt ein frisches MetaTrader5-Modul
# ---------------------------------------------------------------------------

def _fresh_stub(initialize_rv=True) -> types.ModuleType:
    mt5 = types.ModuleType("MetaTrader5")
    mt5.TIMEFRAME_M1  = 1
    mt5.TIMEFRAME_M5  = 5
    mt5.TIMEFRAME_M15 = 15
    mt5.TIMEFRAME_M30 = 30
    mt5.TIMEFRAME_H1  = 16385
    mt5.TIMEFRAME_H4  = 16388
    mt5.TIMEFRAME_D1  = 16408
    mt5.TIMEFRAME_W1  = 32769
    mt5.initialize        = MagicMock(return_value=initialize_rv)
    mt5.shutdown          = MagicMock()
    mt5.copy_rates_from_pos = MagicMock(return_value=None)
    mt5.copy_rates_range  = MagicMock(return_value=None)
    mt5.symbol_info       = MagicMock(return_value=None)
    mt5.symbols_get       = MagicMock(return_value=None)
    mt5.last_error        = MagicMock(return_value=(0, ""))
    mt5.terminal_info     = MagicMock(return_value=None)
    return mt5


# Einmalig in sys.modules eintragen damit der Import klappt
_INITIAL_STUB = _fresh_stub()
sys.modules["MetaTrader5"] = _INITIAL_STUB

import src.data.mt5_connector as _mod  # noqa: E402
from src.data.mt5_connector import (   # noqa: E402
    MT5Connector,
    MT5ConnectionError,
    MT5DataError,
)

TIMEFRAME_MAP = {
    "M1": 1, "M5": 5, "M15": 15, "M30": 30,
    "H1": 16385, "H4": 16388, "D1": 16408, "W1": 32769,
}

# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def _make_rates(n: int = 5) -> np.ndarray:
    dtype = np.dtype([
        ("time",        np.int64),
        ("open",        np.float64),
        ("high",        np.float64),
        ("low",         np.float64),
        ("close",       np.float64),
        ("tick_volume", np.int64),
        ("spread",      np.int32),
        ("real_volume", np.int64),
    ])
    now = int(datetime.now(timezone.utc).timestamp())
    arr = np.zeros(n, dtype=dtype)
    for i in range(n):
        arr[i]["time"]        = now + i * 60
        arr[i]["open"]        = 1.1000 + i * 0.0001
        arr[i]["high"]        = 1.1010 + i * 0.0001
        arr[i]["low"]         = 1.0990 + i * 0.0001
        arr[i]["close"]       = 1.1005 + i * 0.0001
        arr[i]["tick_volume"] = 100 + i
    return arr


def _make_symbol_info(**kw) -> MagicMock:
    info = MagicMock()
    info.point      = kw.get("point",      0.00001)
    info.digits     = kw.get("digits",     5)
    info.spread     = kw.get("spread",     10)
    info.swap_long  = kw.get("swap_long", -0.5)
    info.swap_short = kw.get("swap_short", 0.3)
    return info


def _dt(offset_days: int = 0) -> datetime:
    return datetime.now(timezone.utc) - timedelta(days=offset_days)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def stub(monkeypatch):
    """
    Vor jedem Test: frischen Stub erzeugen und in den Connector-Cache einhaengen.
    So gibt es nie alte side_effects oder erschoepfte Iteratoren.
    """
    s = _fresh_stub(initialize_rv=True)
    sys.modules["MetaTrader5"] = s
    monkeypatch.setattr(_mod, "_MT5_MODULE", s)
    monkeypatch.setattr(_mod, "_TIMEFRAME_MAP", TIMEFRAME_MAP)
    return s


@pytest.fixture
def connector(stub) -> MT5Connector:
    return MT5Connector(
        login=12345,
        password="secret",
        server="FusionMarkets-Demo",
        max_retries=3,
        health_interval=99999,
    )


@pytest.fixture
def connected_connector(connector, stub) -> MT5Connector:
    """Connector direkt als verbunden markieren – kein connect()-Aufruf."""
    connector._connected = True
    return connector


# ---------------------------------------------------------------------------
# Tests: Verbindung
# ---------------------------------------------------------------------------

class TestConnection:

    def test_connect_success(self, connector, stub):
        connector.connect()
        assert connector.is_connected is True
        stub.initialize.assert_called_once()

    def test_connect_initialize_fails(self, connector, stub):
        stub.initialize.return_value = False
        stub.last_error.return_value = (5, "Init failed")
        with pytest.raises(MT5ConnectionError):
            connector.connect()

    def test_connect_retry_then_success(self, connector, stub):
        stub.initialize.side_effect = [False, False, True]
        stub.last_error.side_effect = [(5, "Temp"), (5, "Temp"), (0, "")]
        connector.connect()
        assert stub.initialize.call_count == 3
        assert connector.is_connected is True

    def test_connect_all_retries_exhausted(self, connector, stub):
        stub.initialize.return_value = False
        stub.last_error.return_value = (5, "Persistent")
        with pytest.raises(MT5ConnectionError):
            connector.connect()

    def test_disconnect(self, connected_connector, stub):
        connected_connector.disconnect()
        assert connected_connector.is_connected is False
        stub.shutdown.assert_called_once()

    def test_is_connected_false_by_default(self, connector):
        assert connector.is_connected is False


# ---------------------------------------------------------------------------
# Tests: OHLCV mit start/end
# ---------------------------------------------------------------------------

class TestGetOHLCV:

    def test_returns_dataframe(self, connected_connector, stub):
        stub.copy_rates_range.return_value = _make_rates(10)
        df = connected_connector.get_ohlcv("EURUSD", "M1", _dt(7), _dt(0))
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10

    def test_columns_present(self, connected_connector, stub):
        stub.copy_rates_range.return_value = _make_rates(3)
        df = connected_connector.get_ohlcv("EURUSD", "M1", _dt(7), _dt(0))
        for col in ("open", "high", "low", "close", "volume"):
            assert col in df.columns, f"Spalte '{col}' fehlt"

    def test_index_is_utc_datetime(self, connected_connector, stub):
        stub.copy_rates_range.return_value = _make_rates(3)
        df = connected_connector.get_ohlcv("EURUSD", "M1", _dt(7), _dt(0))
        assert hasattr(df.index, "tz")
        assert str(df.index.tz) == "UTC"

    def test_invalid_timeframe_raises(self, connected_connector):
        with pytest.raises((MT5DataError, ValueError, KeyError)):
            connected_connector.get_ohlcv("EURUSD", "INVALID", _dt(1), _dt(0))

    def test_no_data_raises(self, connected_connector, stub):
        stub.copy_rates_range.return_value = None
        stub.last_error.return_value = (4806, "No data")
        with pytest.raises(MT5DataError):
            connected_connector.get_ohlcv("EURUSD", "M1", _dt(7), _dt(0))

    def test_not_connected_raises(self, connector):
        with pytest.raises(MT5ConnectionError):
            connector.get_ohlcv("EURUSD", "M1", _dt(7), _dt(0))

    def test_all_timeframes_accepted(self, connected_connector, stub):
        stub.copy_rates_range.return_value = _make_rates(5)
        for tf in ("M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1"):
            df = connected_connector.get_ohlcv("EURUSD", tf, _dt(30), _dt(0))
            assert len(df) == 5, f"Timeframe {tf} fehlgeschlagen"


# ---------------------------------------------------------------------------
# Tests: OHLCV mit count
# ---------------------------------------------------------------------------

class TestGetOHLCVCount:

    def test_returns_dataframe(self, connected_connector, stub):
        stub.copy_rates_from_pos.return_value = _make_rates(10)
        df = connected_connector.get_ohlcv_count("EURUSD", "M1", count=10)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10

    def test_not_connected_raises(self, connector):
        with pytest.raises(MT5ConnectionError):
            connector.get_ohlcv_count("EURUSD", "M1", count=10)

    def test_no_data_raises(self, connected_connector, stub):
        stub.copy_rates_from_pos.return_value = None
        stub.last_error.return_value = (4806, "No data")
        with pytest.raises(MT5DataError):
            connected_connector.get_ohlcv_count("EURUSD", "M1", count=10)


# ---------------------------------------------------------------------------
# Tests: Symbol-Info
# ---------------------------------------------------------------------------

class TestSymbolInfo:

    def test_returns_dict(self, connected_connector, stub):
        stub.symbol_info.return_value = _make_symbol_info()
        info = connected_connector.get_symbol_info("EURUSD")
        assert isinstance(info, dict)

    def test_expected_keys(self, connected_connector, stub):
        stub.symbol_info.return_value = _make_symbol_info()
        info = connected_connector.get_symbol_info("EURUSD")
        for key in ("point", "digits", "spread", "swap_long", "swap_short"):
            assert key in info, f"Key '{key}' fehlt"

    def test_unknown_symbol_raises(self, connected_connector, stub):
        stub.symbol_info.return_value = None
        stub.last_error.return_value = (4301, "Symbol not found")
        with pytest.raises(MT5DataError):
            connected_connector.get_symbol_info("INVALID_XXX")

    def test_not_connected_raises(self, connector):
        with pytest.raises(MT5ConnectionError):
            connector.get_symbol_info("EURUSD")


# ---------------------------------------------------------------------------
# Tests: Context Manager
# ---------------------------------------------------------------------------

class TestContextManager:

    def test_enter_connects(self, connector, stub):
        with connector as c:
            assert c.is_connected is True

    def test_exit_disconnects(self, connector, stub):
        with connector as c:
            pass
        assert c.is_connected is False

    def test_exit_on_exception(self, connector, stub):
        try:
            with connector as c:
                raise RuntimeError("Test-Fehler")
        except RuntimeError:
            pass
        assert c.is_connected is False


# ---------------------------------------------------------------------------
# Tests: Health-Check
# ---------------------------------------------------------------------------

class TestHealthCheck:

    def test_health_check_reconnects_on_disconnect(self, connector, stub):
        connector._connected = True
        terminal = MagicMock()
        terminal.connected = False
        stub.terminal_info.return_value = terminal

        connector._health_check()

        stub.initialize.assert_called()

    def test_health_check_no_action_when_connected(self, connector, stub):
        connector._connected = True
        terminal = MagicMock()
        terminal.connected = True
        stub.terminal_info.return_value = terminal

        connector._health_check()

        stub.initialize.assert_not_called()
