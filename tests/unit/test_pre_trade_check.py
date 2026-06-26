"""
Unit-Tests fuer PreTradeCheck.

EconomicCalendar und MT5Connector werden vollstaendig gemockt –
kein Netzwerk, kein MT5-Terminal noetig.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.risk.pre_trade_check import PreTradeCheck


# ─────────────────────────────────────────────────────────────────────────────
#  Hilfsfunktionen
# ─────────────────────────────────────────────────────────────────────────────

def _connector(spread_points: int = 10, point: float = 0.00001) -> MagicMock:
    """Mock fuer MT5Connector mit konfigurierbarem Spread."""
    conn = MagicMock()
    conn.get_symbol_info.return_value = {
        "spread":     spread_points,
        "point":      point,
        "digits":     5,
        "swap_long":  -7.0,
        "swap_short": 2.0,
    }
    return conn


def _calendar(is_no_trade: bool = False) -> MagicMock:
    """Mock fuer EconomicCalendar."""
    cal = MagicMock()
    cal.is_no_trade_zone.return_value = is_no_trade
    return cal


def _check(
    max_spread_pips: float = 3.0,
    spread_points: int = 10,
    point: float = 0.00001,
    pip_size: float = 0.0001,
    is_no_trade: bool = False,
) -> PreTradeCheck:
    """Erstellt PreTradeCheck mit den gewuenschten Parametern."""
    return PreTradeCheck(
        calendar=_calendar(is_no_trade),
        connector=_connector(spread_points, point),
        max_spread_pips=max_spread_pips,
        pip_size=pip_size,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: Normalbetrieb (Trade erlaubt)
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalConditions:

    def test_normal_conditions_allowed(self):
        """Niedriger Spread + kein News-Fenster -> Trade erlaubt."""
        # spread=10 points * 0.00001 / 0.0001 = 1.0 pip < 3.0 Limit
        checker = _check(spread_points=10, is_no_trade=False)
        ok, reason = checker.is_safe_to_trade("EURUSD")
        assert ok is True

    def test_reason_contains_spread_info(self):
        """Begruendung im Erfolgsfall enthaelt Spread-Angabe."""
        checker = _check(spread_points=10)
        ok, reason = checker.is_safe_to_trade("EURUSD")
        assert ok is True
        assert "Spread" in reason
        assert "EURUSD" in reason

    def test_returns_tuple(self):
        """Rueckgabewert ist ein Tupel aus bool und str."""
        checker = _check()
        result = checker.is_safe_to_trade("GBPUSD")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], str)

    def test_spread_at_exact_limit_allowed(self):
        """Spread genau gleich dem Schwellwert -> erlaubt (strikt >)."""
        # spread=30 * 0.00001 / 0.0001 = 3.0 Pips == Limit -> erlaubt
        checker = _check(max_spread_pips=3.0, spread_points=30)
        ok, reason = checker.is_safe_to_trade("EURUSD")
        assert ok is True

    def test_spread_zero_allowed(self):
        """Spread von 0 Pips (z.B. ECN-Konto ohne Spread) -> erlaubt."""
        checker = _check(spread_points=0)
        ok, _ = checker.is_safe_to_trade("EURUSD")
        assert ok is True


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: Hoher Spread blockiert
# ─────────────────────────────────────────────────────────────────────────────

class TestHighSpread:

    def test_high_spread_blocked(self):
        """Spread > Limit -> Trade abgelehnt."""
        # spread=50 * 0.00001 / 0.0001 = 5.0 Pips > 3.0 Limit
        checker = _check(spread_points=50)
        ok, reason = checker.is_safe_to_trade("EURUSD")
        assert ok is False

    def test_high_spread_reason_contains_spread(self):
        """Begruendung bei hohem Spread nennt den Spread-Wert."""
        checker = _check(spread_points=50)
        ok, reason = checker.is_safe_to_trade("EURUSD")
        assert ok is False
        assert "Spread" in reason

    def test_high_spread_reason_contains_symbol(self):
        checker = _check(spread_points=50)
        ok, reason = checker.is_safe_to_trade("GBPUSD")
        assert "GBPUSD" in reason

    def test_spread_just_above_limit_blocked(self):
        """Spread knapp ueber Schwellwert -> blockiert."""
        # spread=31 * 0.00001 / 0.0001 = 3.1 Pips > 3.0 Limit
        checker = _check(max_spread_pips=3.0, spread_points=31)
        ok, _ = checker.is_safe_to_trade("EURUSD")
        assert ok is False

    def test_custom_max_spread_respected(self):
        """Konfigurierbarer Schwellwert wird eingehalten."""
        # Schwellwert=1.0 Pip, spread=15 points = 1.5 Pips -> blockiert
        checker = _check(max_spread_pips=1.0, spread_points=15)
        ok, reason = checker.is_safe_to_trade("EURUSD")
        assert ok is False
        assert "1.5" in reason or "Spread" in reason

    def test_tight_threshold_blocks_normal_spread(self):
        """Sehr enges Limit blockiert auch normalen Spread."""
        checker = _check(max_spread_pips=0.5, spread_points=10)
        ok, _ = checker.is_safe_to_trade("EURUSD")
        assert ok is False

    def test_high_spread_does_not_call_connector_when_no_trade_first(self):
        """Wenn Kalender blockiert, wird get_symbol_info NICHT aufgerufen."""
        cal = _calendar(is_no_trade=True)
        conn = _connector(spread_points=100)
        checker = PreTradeCheck(calendar=cal, connector=conn, max_spread_pips=3.0)
        checker.is_safe_to_trade("EURUSD")
        conn.get_symbol_info.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: News-Zeitfenster (No-Trade-Zone)
# ─────────────────────────────────────────────────────────────────────────────

class TestNoTradeZone:

    def test_no_trade_zone_blocks_trade(self):
        """Kalender meldet No-Trade-Zone -> Trade abgelehnt."""
        checker = _check(is_no_trade=True)
        ok, reason = checker.is_safe_to_trade("EURUSD")
        assert ok is False

    def test_no_trade_zone_reason_contains_keyword(self):
        """Begruendung bei No-Trade-Zone erklaert den Grund."""
        checker = _check(is_no_trade=True)
        ok, reason = checker.is_safe_to_trade("EURUSD")
        assert ok is False
        assert "No-Trade" in reason or "News" in reason or "Event" in reason

    def test_no_trade_zone_reason_contains_symbol(self):
        checker = _check(is_no_trade=True)
        ok, reason = checker.is_safe_to_trade("USDJPY")
        assert "USDJPY" in reason

    def test_no_trade_zone_blocks_even_with_low_spread(self):
        """No-Trade-Zone blockiert auch wenn Spread minimal ist."""
        checker = _check(spread_points=1, is_no_trade=True)
        ok, _ = checker.is_safe_to_trade("EURUSD")
        assert ok is False

    def test_no_trade_zone_checked_before_spread(self):
        """Kalender wird VOR Spread-Pruefung ausgewertet (Reihenfolge)."""
        cal = _calendar(is_no_trade=True)
        conn = _connector(spread_points=5)  # niedriger Spread
        checker = PreTradeCheck(calendar=cal, connector=conn, max_spread_pips=3.0)
        ok, reason = checker.is_safe_to_trade("EURUSD")
        assert ok is False
        # Spread-Pruefung nie erreicht -> get_symbol_info nie aufgerufen
        conn.get_symbol_info.assert_not_called()

    def test_no_trade_false_does_not_block(self):
        """is_no_trade_zone=False blockiert NICHT."""
        checker = _check(is_no_trade=False, spread_points=5)
        ok, _ = checker.is_safe_to_trade("EURUSD")
        assert ok is True


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: Mock-Verifizierung (korrekte Weiterleitung)
# ─────────────────────────────────────────────────────────────────────────────

class TestMockDelegation:

    def test_calendar_called_with_correct_symbol(self):
        cal = _calendar()
        conn = _connector()
        checker = PreTradeCheck(calendar=cal, connector=conn)
        checker.is_safe_to_trade("GBPUSD")
        cal.is_no_trade_zone.assert_called_once_with("GBPUSD")

    def test_connector_called_with_correct_symbol(self):
        cal = _calendar(is_no_trade=False)
        conn = _connector()
        checker = PreTradeCheck(calendar=cal, connector=conn)
        checker.is_safe_to_trade("USDJPY")
        conn.get_symbol_info.assert_called_once_with("USDJPY")

    def test_connector_not_called_when_calendar_blocks(self):
        """Spread-Check entfaellt komplett wenn Kalender blockiert."""
        cal = _calendar(is_no_trade=True)
        conn = _connector()
        checker = PreTradeCheck(calendar=cal, connector=conn)
        checker.is_safe_to_trade("EURUSD")
        conn.get_symbol_info.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: Spread-Umrechnung (verschiedene Symbole / Pip-Groessen)
# ─────────────────────────────────────────────────────────────────────────────

class TestSpreadConversion:

    def test_eurusd_standard_pip_conversion(self):
        """EURUSD: spread=10 points * 0.00001 / 0.0001 = 1.0 Pip -> erlaubt."""
        checker = _check(max_spread_pips=3.0, spread_points=10, point=0.00001, pip_size=0.0001)
        ok, _ = checker.is_safe_to_trade("EURUSD")
        assert ok is True

    def test_jpy_pair_pip_conversion(self):
        """USDJPY: spread=10 points * 0.001 / 0.01 = 1.0 Pip -> erlaubt bei Limit 3.0."""
        checker = PreTradeCheck(
            calendar=_calendar(),
            connector=_connector(spread_points=10, point=0.001),
            max_spread_pips=3.0,
            pip_size=0.01,
        )
        ok, _ = checker.is_safe_to_trade("USDJPY")
        assert ok is True

    def test_jpy_high_spread_blocked(self):
        """USDJPY: spread=50 points * 0.001 / 0.01 = 5.0 Pips > 3.0 -> blockiert."""
        checker = PreTradeCheck(
            calendar=_calendar(),
            connector=_connector(spread_points=50, point=0.001),
            max_spread_pips=3.0,
            pip_size=0.01,
        )
        ok, _ = checker.is_safe_to_trade("USDJPY")
        assert ok is False

    def test_spread_calculation_is_correct(self):
        """Numerische Pruefung der Umrechnung: 20 points * 0.00001 / 0.0001 = 2.0 Pips."""
        checker = _check(max_spread_pips=3.0, spread_points=20, point=0.00001, pip_size=0.0001)
        ok, reason = checker.is_safe_to_trade("EURUSD")
        assert ok is True
        assert "2.0" in reason


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: Konfiguration
# ─────────────────────────────────────────────────────────────────────────────

class TestConfiguration:

    def test_default_max_spread_is_3_pips(self):
        """Standard-Schwellwert ist 3.0 Pips: spread=31 points = 3.1 Pips -> blocked."""
        checker = PreTradeCheck(
            calendar=_calendar(),
            connector=_connector(spread_points=31),
        )
        ok, _ = checker.is_safe_to_trade("EURUSD")
        assert ok is False

    def test_relaxed_max_spread_allows_wider_spread(self):
        """Erhoehtes Limit erlaubt breiteren Spread."""
        checker = PreTradeCheck(
            calendar=_calendar(),
            connector=_connector(spread_points=80),  # 8.0 Pips
            max_spread_pips=10.0,
        )
        ok, _ = checker.is_safe_to_trade("EURUSD")
        assert ok is True

    def test_different_symbols_handled_independently(self):
        """Verschiedene Symbole werden unabhaengig geprueft."""
        checker = _check(spread_points=10, is_no_trade=False)
        ok_eur, _ = checker.is_safe_to_trade("EURUSD")
        ok_gbp, _ = checker.is_safe_to_trade("GBPUSD")
        assert ok_eur is True
        assert ok_gbp is True


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: XAUUSD Pip-Konvention + symbol_overrides
#
#  Regression-Test fuer den Bug bei dem XAUUSD mit der Forex-Pip-Groesse
#  (0.0001) berechnet wurde, obwohl Gold 2 Dezimalstellen hat (point=0.01,
#  pip_size=0.01). Folge: 5 points * 0.01 / 0.0001 = 500 statt 5 Pips.
# ─────────────────────────────────────────────────────────────────────────────

def _xauusd_connector(spread_points: int = 5) -> MagicMock:
    """MT5Connector-Mock mit XAUUSD-typischen Werten (point=0.01, 2 Dezimalstellen)."""
    conn = MagicMock()
    conn.get_symbol_info.return_value = {
        "spread": spread_points,
        "point":  0.01,
        "digits": 2,
    }
    return conn


class TestXAUUSDSpreadConvention:

    def test_xauusd_wrong_pip_size_gives_false_block(self):
        """Reproduziert den urspruenglichen Bug: Forex-pip_size blockiert Gold faelschlich.
        5 points * 0.01 / 0.0001 = 500 Pips >> 3.0 Limit."""
        checker = PreTradeCheck(
            calendar=_calendar(),
            connector=_xauusd_connector(spread_points=5),
            max_spread_pips=3.0,
            pip_size=0.0001,  # FALSCH fuer Gold
        )
        ok, reason = checker.is_safe_to_trade("XAUUSD")
        assert ok is False
        assert "500" in reason  # beweist dass die falsche Konvention 500 Pips liefert

    def test_xauusd_correct_pip_size_via_symbol_overrides(self):
        """Mit korrekter Gold-Pip-Groesse (0.01) und realistischem Limit wird Trade erlaubt.
        5 points * 0.01 / 0.01 = 5 Pips < 100 Pips Limit -> OK."""
        checker = PreTradeCheck(
            calendar=_calendar(),
            connector=_xauusd_connector(spread_points=5),
            max_spread_pips=3.0,
            symbol_overrides={"XAUUSD": {"pip_size": 0.01, "max_spread_pips": 100.0}},
        )
        ok, reason = checker.is_safe_to_trade("XAUUSD")
        assert ok is True
        assert "5.0" in reason

    def test_xauusd_spread_calculation_correct(self):
        """Numerische Pruefung: 30 points * 0.01 / 0.01 = 30.0 Pips."""
        checker = PreTradeCheck(
            calendar=_calendar(),
            connector=_xauusd_connector(spread_points=30),
            max_spread_pips=3.0,
            symbol_overrides={"XAUUSD": {"pip_size": 0.01, "max_spread_pips": 100.0}},
        )
        ok, reason = checker.is_safe_to_trade("XAUUSD")
        assert ok is True
        assert "30.0" in reason

    def test_xauusd_wide_spread_blocked_at_correct_threshold(self):
        """150 Pips Gold-Spread (1.50 USD) > 100 Pips Limit -> blockiert."""
        checker = PreTradeCheck(
            calendar=_calendar(),
            connector=_xauusd_connector(spread_points=150),
            max_spread_pips=3.0,
            symbol_overrides={"XAUUSD": {"pip_size": 0.01, "max_spread_pips": 100.0}},
        )
        ok, reason = checker.is_safe_to_trade("XAUUSD")
        assert ok is False
        assert "150.0" in reason

    def test_xauusd_override_does_not_affect_eurusd(self):
        """symbol_overrides fuer XAUUSD aendert nichts an EURUSD-Berechnung."""
        conn = MagicMock()
        conn.get_symbol_info.return_value = {"spread": 10, "point": 0.00001, "digits": 5}
        checker = PreTradeCheck(
            calendar=_calendar(),
            connector=conn,
            max_spread_pips=3.0,
            symbol_overrides={"XAUUSD": {"pip_size": 0.01, "max_spread_pips": 100.0}},
        )
        # EURUSD: 10 * 0.00001 / 0.0001 = 1.0 Pip -> erlaubt
        ok, _ = checker.is_safe_to_trade("EURUSD")
        assert ok is True

    def test_symbol_overrides_only_pip_size(self):
        """Nur pip_size in override: max_spread_pips faellt auf globalen Default zurueck."""
        checker = PreTradeCheck(
            calendar=_calendar(),
            connector=_xauusd_connector(spread_points=5),
            max_spread_pips=10.0,
            symbol_overrides={"XAUUSD": {"pip_size": 0.01}},  # kein max_spread_pips
        )
        # 5 Pips < 10.0 (globaler Default) -> erlaubt
        ok, _ = checker.is_safe_to_trade("XAUUSD")
        assert ok is True

    def test_symbol_overrides_only_max_spread(self):
        """Nur max_spread_pips in override: pip_size faellt auf globalen Default zurueck."""
        checker = PreTradeCheck(
            calendar=_calendar(),
            connector=_xauusd_connector(spread_points=5),
            pip_size=0.0001,
            max_spread_pips=3.0,
            symbol_overrides={"XAUUSD": {"max_spread_pips": 1000.0}},  # kein pip_size
        )
        # 5 * 0.01 / 0.0001 = 500 Pips < 1000.0 -> erlaubt (aber Rechnung bleibt falsch)
        ok, _ = checker.is_safe_to_trade("XAUUSD")
        assert ok is True

    def test_config_style_xauusd_and_usdjpy_overrides(self):
        """Kombinierte Overrides wie in config.yaml: XAUUSD + USDJPY."""
        symbol_overrides = {
            "XAUUSD": {"pip_size": 0.01, "max_spread_pips": 100.0},
            "USDJPY": {"pip_size": 0.01, "max_spread_pips": 3.0},
        }
        # XAUUSD: 5 points * 0.01 / 0.01 = 5 Pips < 100 -> OK
        xau_conn = _xauusd_connector(spread_points=5)
        xau_checker = PreTradeCheck(
            calendar=_calendar(), connector=xau_conn,
            max_spread_pips=3.0, symbol_overrides=symbol_overrides,
        )
        ok_xau, _ = xau_checker.is_safe_to_trade("XAUUSD")
        assert ok_xau is True

        # USDJPY: 10 points * 0.001 / 0.01 = 1 Pip < 3 -> OK
        jpy_conn = MagicMock()
        jpy_conn.get_symbol_info.return_value = {"spread": 10, "point": 0.001, "digits": 3}
        jpy_checker = PreTradeCheck(
            calendar=_calendar(), connector=jpy_conn,
            max_spread_pips=3.0, symbol_overrides=symbol_overrides,
        )
        ok_jpy, _ = jpy_checker.is_safe_to_trade("USDJPY")
        assert ok_jpy is True
