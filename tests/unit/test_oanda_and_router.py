"""
Unit-Tests fuer OANDAConnector und DataRouter.
OANDA-API wird vollstaendig gemockt (requests.Session).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch, PropertyMock

import pandas as pd
import pytest

from src.data.oanda_connector import (
    OANDAConnector,
    OANDAConnectionError,
    OANDADataError,
)
from src.data.data_router import (
    DataRouter,
    PriceValidator,
    PriceDiscrepancyError,
    EmergencyModeError,
)

# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def _candles_response(n: int = 3) -> dict:
    """Erzeugt eine OANDA-API-Antwort mit n Kerzen."""
    now = datetime.now(timezone.utc)
    candles = []
    for i in range(n):
        t = (now + timedelta(minutes=i)).strftime("%Y-%m-%dT%H:%M:%S.000000000Z")
        candles.append({
            "time":     t,
            "complete": True,
            "volume":   100 + i,
            "mid": {
                "o": f"{1.1000 + i * 0.0001:.5f}",
                "h": f"{1.1010 + i * 0.0001:.5f}",
                "l": f"{1.0990 + i * 0.0001:.5f}",
                "c": f"{1.1005 + i * 0.0001:.5f}",
            },
        })
    return {"candles": candles}


def _mock_response(status: int = 200, body: dict = None) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = body or {}
    resp.text = json.dumps(body or {})
    return resp


def _make_connector(session_mock=None) -> OANDAConnector:
    c = OANDAConnector(
        api_key="test-key",
        account_id="123-456",
        demo=True,
        max_retries=1,
    )
    c._connected = True
    if session_mock:
        c._session = session_mock
    else:
        c._session = MagicMock()
    return c


def _dt(offset_days: int = 0) -> datetime:
    return datetime.now(timezone.utc) - timedelta(days=offset_days)


# ---------------------------------------------------------------------------
# Tests: OANDAConnector – Verbindung
# ---------------------------------------------------------------------------

class TestOANDAConnection:

    def test_connect_success(self):
        c = OANDAConnector(api_key="key", account_id="acc", max_retries=1)
        with patch("requests.Session") as MockSession:
            sess = MockSession.return_value
            sess.get.return_value = _mock_response(200, {"account": {}})
            c.connect()
        assert c.is_connected is True

    def test_connect_fails_on_401(self):
        c = OANDAConnector(api_key="bad-key", account_id="acc", max_retries=1)
        with patch("requests.Session") as MockSession:
            sess = MockSession.return_value
            sess.get.return_value = _mock_response(401, {"errorMessage": "Unauthorized"})
            with pytest.raises(OANDAConnectionError):
                c.connect()

    def test_disconnect(self):
        c = _make_connector()
        c.disconnect()
        assert c.is_connected is False

    def test_is_connected_false_by_default(self):
        c = OANDAConnector(api_key="k", account_id="a")
        assert c.is_connected is False

    def test_context_manager(self):
        c = OANDAConnector(api_key="key", account_id="acc", max_retries=1)
        with patch("requests.Session") as MockSession:
            sess = MockSession.return_value
            sess.get.return_value = _mock_response(200, {"account": {}})
            with c as conn:
                assert conn.is_connected is True
        assert c.is_connected is False


# ---------------------------------------------------------------------------
# Tests: OANDAConnector – Symbol-Mapping
# ---------------------------------------------------------------------------

class TestSymbolMapping:

    def test_standard_mapping(self):
        c = _make_connector()
        assert c.map_symbol("EURUSD") == "EUR_USD"
        assert c.map_symbol("GBPUSD") == "GBP_USD"
        assert c.map_symbol("XAUUSD") == "XAU_USD"

    def test_custom_mapping(self):
        c = OANDAConnector(
            api_key="k", account_id="a",
            symbol_map={"CUSTOM": "CUS_TOM"}
        )
        c._connected = True
        c._session = MagicMock()
        assert c.map_symbol("CUSTOM") == "CUS_TOM"

    def test_already_oanda_format(self):
        c = _make_connector()
        assert c.map_symbol("EUR_USD") == "EUR_USD"

    def test_unknown_symbol_raises(self):
        c = _make_connector()
        with pytest.raises(OANDADataError):
            c.map_symbol("UNKNOWNSYMBOL")


# ---------------------------------------------------------------------------
# Tests: OANDAConnector – OHLCV
# ---------------------------------------------------------------------------

class TestOANDAGetOHLCV:

    def test_returns_dataframe(self):
        c = _make_connector()
        c._session.get.return_value = _mock_response(200, _candles_response(5))
        df = c.get_ohlcv("EURUSD", "H1", _dt(7), _dt(0))
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5

    def test_columns_present(self):
        c = _make_connector()
        c._session.get.return_value = _mock_response(200, _candles_response(3))
        df = c.get_ohlcv("EURUSD", "M1", _dt(1), _dt(0))
        for col in ("open", "high", "low", "close", "volume"):
            assert col in df.columns

    def test_index_is_utc(self):
        c = _make_connector()
        c._session.get.return_value = _mock_response(200, _candles_response(3))
        df = c.get_ohlcv("EURUSD", "M1", _dt(1), _dt(0))
        assert str(df.index.tz) == "UTC"

    def test_invalid_timeframe_raises(self):
        c = _make_connector()
        with pytest.raises(OANDADataError):
            c.get_ohlcv("EURUSD", "INVALID", _dt(1), _dt(0))

    def test_api_error_raises(self):
        c = _make_connector()
        c._session.get.return_value = _mock_response(400, {"errorMessage": "Bad Request"})
        with pytest.raises(OANDADataError):
            c.get_ohlcv("EURUSD", "H1", _dt(7), _dt(0))

    def test_not_connected_raises(self):
        c = OANDAConnector(api_key="k", account_id="a")
        with pytest.raises(OANDAConnectionError):
            c.get_ohlcv("EURUSD", "H1", _dt(1), _dt(0))

    def test_all_timeframes_accepted(self):
        c = _make_connector()
        c._session.get.return_value = _mock_response(200, _candles_response(2))
        for tf in ("M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1"):
            df = c.get_ohlcv("EURUSD", tf, _dt(30), _dt(0))
            assert len(df) == 2


class TestOANDAGetOHLCVCount:

    def test_returns_dataframe(self):
        c = _make_connector()
        c._session.get.return_value = _mock_response(200, _candles_response(10))
        df = c.get_ohlcv_count("EURUSD", "M1", count=10)
        assert len(df) == 10

    def test_not_connected_raises(self):
        c = OANDAConnector(api_key="k", account_id="a")
        with pytest.raises(OANDAConnectionError):
            c.get_ohlcv_count("EURUSD", "M1", count=10)


# ---------------------------------------------------------------------------
# Tests: PriceValidator
# ---------------------------------------------------------------------------

class TestPriceValidator:

    def test_within_tolerance(self):
        v = PriceValidator(max_pips=5)
        assert v.check("EURUSD", 1.10000, 1.10003) is True

    def test_exceeds_tolerance(self):
        v = PriceValidator(max_pips=5)
        assert v.check("EURUSD", 1.10000, 1.10060) is False

    def test_exact_boundary(self):
        v = PriceValidator(max_pips=5)
        # Genau 5 Pips = noch OK
        assert v.check("EURUSD", 1.10000, 1.10050) is True

    def test_custom_pip_size(self):
        # JPY-Paare: pip = 0.01
        v = PriceValidator(max_pips=5, pip_size=0.01)
        assert v.check("USDJPY", 150.000, 150.040) is True
        assert v.check("USDJPY", 150.000, 150.060) is False


# ---------------------------------------------------------------------------
# Tests: DataRouter
# ---------------------------------------------------------------------------

def _make_mt5(connected: bool = True) -> MagicMock:
    m = MagicMock()
    type(m).is_connected = PropertyMock(return_value=connected)
    return m


def _make_oanda(connected: bool = True) -> MagicMock:
    m = MagicMock()
    type(m).is_connected = PropertyMock(return_value=connected)
    return m


class TestDataRouter:

    def test_returns_mt5_when_connected(self):
        mt5   = _make_mt5(connected=True)
        oanda = _make_oanda(connected=True)
        router = DataRouter(mt5, oanda)
        assert router.get_connector() is mt5

    def test_returns_oanda_when_mt5_down(self):
        mt5   = _make_mt5(connected=False)
        oanda = _make_oanda(connected=True)
        router = DataRouter(mt5, oanda)
        result = router.get_connector()   # kein Symbol -> kein Preischeck
        assert result is oanda

    def test_raises_when_both_down(self):
        mt5   = _make_mt5(connected=False)
        oanda = _make_oanda(connected=False)
        router = DataRouter(mt5, oanda)
        with pytest.raises(EmergencyModeError):
            router.get_connector()

    def test_raises_on_price_discrepancy(self):
        mt5   = _make_mt5(connected=False)
        oanda = _make_oanda(connected=True)

        validator = MagicMock()
        validator.check.return_value = False

        router = DataRouter(mt5, oanda, validator=validator)

        # _price_check_passes mocken damit er False zurueckgibt
        router._price_check_passes = MagicMock(return_value=False)

        with pytest.raises(PriceDiscrepancyError):
            router.get_connector(symbol="EURUSD")

    def test_fallback_ok_when_price_matches(self):
        mt5   = _make_mt5(connected=False)
        oanda = _make_oanda(connected=True)
        router = DataRouter(mt5, oanda)
        router._price_check_passes = MagicMock(return_value=True)

        result = router.get_connector(symbol="EURUSD")
        assert result is oanda

    def test_active_connector_name_mt5(self):
        mt5   = _make_mt5(connected=True)
        oanda = _make_oanda(connected=True)
        router = DataRouter(mt5, oanda)
        router.get_connector()
        assert router.active_connector_name == "mt5"

    def test_active_connector_name_oanda(self):
        mt5   = _make_mt5(connected=False)
        oanda = _make_oanda(connected=True)
        router = DataRouter(mt5, oanda)
        router.get_connector()
        assert router.active_connector_name == "oanda"

    def test_audit_log_written(self, tmp_path):
        log_file = str(tmp_path / "audit.log")
        mt5   = _make_mt5(connected=True)
        oanda = _make_oanda(connected=True)
        router = DataRouter(mt5, oanda, audit_log_path=log_file)
        router.get_connector()

        with open(log_file) as f:
            content = f.read()
        assert "MT5" in content
        assert "CONNECTOR_SWITCH" in content

    def test_get_ohlcv_delegates(self):
        mt5   = _make_mt5(connected=True)
        oanda = _make_oanda(connected=True)
        router = DataRouter(mt5, oanda)
        start, end = _dt(7), _dt(0)
        router.get_ohlcv("EURUSD", "H1", start, end)
        mt5.get_ohlcv.assert_called_once_with("EURUSD", "H1", start, end)

    def test_get_ohlcv_count_delegates(self):
        mt5   = _make_mt5(connected=True)
        oanda = _make_oanda(connected=True)
        router = DataRouter(mt5, oanda)
        router.get_ohlcv_count("EURUSD", "H1", count=50)
        mt5.get_ohlcv_count.assert_called_once_with("EURUSD", "H1", 50)
