"""
Unit-Tests fuer EconomicCalendar.
HTTP-Requests werden vollstaendig gemockt.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

import pytest

from src.data.calendar import EconomicCalendar, EconomicEvent


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def _event_dict(title: str, country: str, dt: datetime, impact: str = "High") -> dict:
    return {
        "title":   title,
        "country": country,
        "date":    dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "impact":  impact,
    }


def _mock_response(status: int = 200, body: list = None) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = body or []
    resp.raise_for_status = MagicMock()
    if status >= 400:
        import requests
        resp.raise_for_status.side_effect = requests.HTTPError(f"{status} Error")
    return resp


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Tests: refresh() / Caching
# ---------------------------------------------------------------------------

class TestRefresh:

    def test_refresh_loads_events(self, tmp_path):
        cal = EconomicCalendar(cache_dir=str(tmp_path))
        events = [_event_dict("NFP", "USD", _now() + timedelta(hours=2))]

        with patch("requests.get", return_value=_mock_response(200, events)):
            result = cal.refresh(force=True)

        assert result is True
        assert len(cal._events) == 1
        assert cal._events[0].title == "NFP"

    def test_refresh_saves_cache_file(self, tmp_path):
        cal = EconomicCalendar(cache_dir=str(tmp_path))
        events = [_event_dict("CPI", "EUR", _now() + timedelta(hours=1))]

        with patch("requests.get", return_value=_mock_response(200, events)):
            cal.refresh(force=True)

        assert cal._cache_file.exists()

    def test_refresh_fallback_on_network_error(self, tmp_path):
        cal = EconomicCalendar(cache_dir=str(tmp_path))
        import requests
        with patch("requests.get", side_effect=requests.ConnectionError("down")):
            result = cal.refresh(force=True)

        assert result is False

    def test_loads_from_cache_after_failed_refresh(self, tmp_path):
        cal1 = EconomicCalendar(cache_dir=str(tmp_path))
        events = [_event_dict("FOMC", "USD", _now() + timedelta(hours=3))]
        with patch("requests.get", return_value=_mock_response(200, events)):
            cal1.refresh(force=True)

        # Neue Instanz, simuliert Neustart der App
        cal2 = EconomicCalendar(cache_dir=str(tmp_path))
        import requests
        with patch("requests.get", side_effect=requests.ConnectionError("down")):
            cal2.refresh(force=True)

        assert len(cal2._events) == 1
        assert cal2._events[0].title == "FOMC"

    def test_no_refresh_without_force_when_recent(self, tmp_path):
        cal = EconomicCalendar(cache_dir=str(tmp_path))
        cal._last_update = datetime.now(timezone.utc)

        with patch("requests.get") as mock_get:
            cal.refresh(force=False)
            mock_get.assert_not_called()

    def test_invalid_impact_defaults_to_low(self, tmp_path):
        cal = EconomicCalendar(cache_dir=str(tmp_path))
        events = [_event_dict("Minor", "JPY", _now(), impact="Unknown")]
        with patch("requests.get", return_value=_mock_response(200, events)):
            cal.refresh(force=True)
        assert cal._events[0].impact == "Low"


# ---------------------------------------------------------------------------
# Tests: is_no_trade_zone()
# ---------------------------------------------------------------------------

class TestIsNoTradeZone:

    def _calendar_with_event(self, tmp_path, country="USD", impact="High", minutes_from_now=0):
        cal = EconomicCalendar(cache_dir=str(tmp_path), before_minutes=30, after_minutes=15)
        event_time = _now() + timedelta(minutes=minutes_from_now)
        events = [_event_dict("Test Event", country, event_time, impact=impact)]
        with patch("requests.get", return_value=_mock_response(200, events)):
            cal.refresh(force=True)
        return cal, event_time

    def test_blocks_within_before_window(self, tmp_path):
        cal, event_time = self._calendar_with_event(tmp_path, country="USD", minutes_from_now=20)
        check_time = event_time - timedelta(minutes=10)  # 10 min vor Event, < 30 min Fenster
        assert cal.is_no_trade_zone("EURUSD", timestamp=check_time) is True

    def test_blocks_within_after_window(self, tmp_path):
        cal, event_time = self._calendar_with_event(tmp_path, country="USD", minutes_from_now=-5)
        check_time = event_time + timedelta(minutes=10)  # 10 min nach Event, < 15 min Fenster
        assert cal.is_no_trade_zone("EURUSD", timestamp=check_time) is True

    def test_allows_outside_window(self, tmp_path):
        cal, event_time = self._calendar_with_event(tmp_path, country="USD", minutes_from_now=120)
        check_time = _now()  # weit ausserhalb des Fensters
        assert cal.is_no_trade_zone("EURUSD", timestamp=check_time) is False

    def test_medium_impact_does_not_block(self, tmp_path):
        cal, event_time = self._calendar_with_event(
            tmp_path, country="USD", impact="Medium", minutes_from_now=10
        )
        assert cal.is_no_trade_zone("EURUSD", timestamp=_now()) is False

    def test_low_impact_does_not_block(self, tmp_path):
        cal, event_time = self._calendar_with_event(
            tmp_path, country="USD", impact="Low", minutes_from_now=10
        )
        assert cal.is_no_trade_zone("EURUSD", timestamp=_now()) is False

    def test_unrelated_currency_does_not_block(self, tmp_path):
        cal, event_time = self._calendar_with_event(tmp_path, country="JPY", minutes_from_now=10)
        assert cal.is_no_trade_zone("EURUSD", timestamp=_now()) is False

    def test_eur_event_blocks_eurusd(self, tmp_path):
        cal, event_time = self._calendar_with_event(tmp_path, country="EUR", minutes_from_now=10)
        assert cal.is_no_trade_zone("EURUSD", timestamp=_now()) is True

    def test_fallback_true_when_no_data(self, tmp_path):
        cal = EconomicCalendar(cache_dir=str(tmp_path))
        # Kein refresh() aufgerufen, kein Cache vorhanden
        assert cal.is_no_trade_zone("EURUSD") is True

    def test_uses_now_when_timestamp_not_given(self, tmp_path):
        cal, event_time = self._calendar_with_event(tmp_path, country="USD", minutes_from_now=10)
        # Kein timestamp uebergeben -> sollte jetzt verwenden -> innerhalb 30min Fenster
        assert cal.is_no_trade_zone("EURUSD") is True


# ---------------------------------------------------------------------------
# Tests: Waehrungs-Extraktion
# ---------------------------------------------------------------------------

class TestCurrencyExtraction:

    def test_standard_pair(self):
        assert EconomicCalendar._extract_currencies("EURUSD") == {"EUR", "USD"}

    def test_oanda_format(self):
        assert EconomicCalendar._extract_currencies("EUR_USD") == {"EUR", "USD"}

    def test_case_insensitive(self):
        assert EconomicCalendar._extract_currencies("eurusd") == {"EUR", "USD"}


# ---------------------------------------------------------------------------
# Tests: get_upcoming_events()
# ---------------------------------------------------------------------------

class TestUpcomingEvents:

    def test_returns_events_within_window(self, tmp_path):
        cal = EconomicCalendar(cache_dir=str(tmp_path))
        events = [
            _event_dict("Soon", "USD", _now() + timedelta(hours=2)),
            _event_dict("TooFar", "USD", _now() + timedelta(hours=48)),
        ]
        with patch("requests.get", return_value=_mock_response(200, events)):
            cal.refresh(force=True)

        upcoming = cal.get_upcoming_events(hours_ahead=24)
        titles = [e.title for e in upcoming]
        assert "Soon" in titles
        assert "TooFar" not in titles

    def test_filters_by_symbol(self, tmp_path):
        cal = EconomicCalendar(cache_dir=str(tmp_path))
        events = [
            _event_dict("USD Event", "USD", _now() + timedelta(hours=2)),
            _event_dict("JPY Event", "JPY", _now() + timedelta(hours=2)),
        ]
        with patch("requests.get", return_value=_mock_response(200, events)):
            cal.refresh(force=True)

        upcoming = cal.get_upcoming_events(symbol="EURUSD", hours_ahead=24)
        titles = [e.title for e in upcoming]
        assert "USD Event" in titles
        assert "JPY Event" not in titles

    def test_filters_by_min_impact(self, tmp_path):
        cal = EconomicCalendar(cache_dir=str(tmp_path))
        events = [
            _event_dict("Big", "USD", _now() + timedelta(hours=2), impact="High"),
            _event_dict("Small", "USD", _now() + timedelta(hours=2), impact="Low"),
        ]
        with patch("requests.get", return_value=_mock_response(200, events)):
            cal.refresh(force=True)

        upcoming = cal.get_upcoming_events(min_impact="High", hours_ahead=24)
        titles = [e.title for e in upcoming]
        assert "Big" in titles
        assert "Small" not in titles

    def test_sorted_by_time(self, tmp_path):
        cal = EconomicCalendar(cache_dir=str(tmp_path))
        events = [
            _event_dict("Later", "USD", _now() + timedelta(hours=10)),
            _event_dict("Sooner", "USD", _now() + timedelta(hours=1)),
        ]
        with patch("requests.get", return_value=_mock_response(200, events)):
            cal.refresh(force=True)

        upcoming = cal.get_upcoming_events(hours_ahead=24, min_impact="Low")
        assert [e.title for e in upcoming] == ["Sooner", "Later"]
