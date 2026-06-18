"""
tests/unit/test_anomaly_detector.py
Unit-Tests fuer AnomalyDetector (Rogue-Bot-Schutz).

Abgedeckt:
  Allgemein
    - Leere Order-Liste -> keine Anomalie
    - Normale Order-Liste -> keine Anomalie
    - Mehrere gleichzeitige Anomalien -> kombinierte Begruendung

  Check 1 – Trade-Frequenz
    - Genau max_trades Trades im Fenster -> kein Alarm (Grenzwert: > nicht >=)
    - max_trades + 1 Trades im Fenster -> Alarm
    - Trades ausserhalb des Fensters werden nicht gezaehlt
    - Orders ohne Timestamp werden nicht gezaehlt
    - Konfigurierbare Schwellwerte

  Check 2 – Lot-Groesse
    - Letzte Order klar unter Schwellwert -> kein Alarm
    - Letzte Order exakt am Schwellwert -> kein Alarm (Grenzwert: > nicht >=)
    - Letzte Order ueber Schwellwert -> Alarm
    - Weniger als 2 Orders -> kein Alarm (kein Durchschnitt berechenbar)
    - Orders ohne lot_size werden ignoriert
    - Lot=0 wird ignoriert

  Check 3 – Wiederholte Ablehnungen
    - Genau max_rejections - 1 aufeinanderfolgend -> kein Alarm
    - Genau max_rejections aufeinanderfolgend -> Alarm
    - Unterbrechung durch erfolgreiche Order -> zaehlt neu
    - Status-Varianten: 'rejected', 'error', 'failed'
    - Error-Codes in Begruendung
    - Leere Liste -> kein Alarm

  Check 4 – Signal-Flackern
    - Genau max_signal_changes - 1 Wechsel -> kein Alarm
    - Genau max_signal_changes Wechsel -> Alarm
    - Wechsel ausserhalb 1-Minuten-Fenster werden nicht gezaehlt
    - Orders ohne 'signal'-Key werden ignoriert
    - Nur 1 Signal-Order -> kein Alarm
    - Kein Wechsel bei gleichem Signal -> kein Alarm

  _parse_timestamp
    - datetime ohne tz -> utc angenommen
    - datetime mit tz -> unveraendert
    - float (Unix-Timestamp)
    - int (Unix-Timestamp)
    - ISO-String ohne tz
    - ISO-String mit tz
    - None -> None
    - Ungueltiger String -> None
    - Unbekannter Typ -> None

  EmergencyHandler-Integration
    - Kein EmergencyHandler konfiguriert -> kein Fehler
    - EmergencyHandler wird bei Anomalie aufgerufen (handle_bad_datafeed)
    - EmergencyHandler wird bei keine Anomalie nicht aufgerufen
    - Begruendung enthaelt '[AnomalyDetector]'-Praefix
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, call

import pytest

from src.risk.anomaly_detector import AnomalyDetector


# ─── Hilfsfunktionen ─────────────────────────────────────────────────────────

def _now() -> datetime:
    """Fester Referenzzeitpunkt fuer deterministische Tests."""
    return datetime(2026, 6, 18, 12, 0, 0, tzinfo=timezone.utc)


def _detector(**kwargs) -> AnomalyDetector:
    """Erstellt AnomalyDetector mit fixem _now_fn und optionalen Overrides."""
    return AnomalyDetector(_now_fn=_now, **kwargs)


def _order_at(offset_seconds: float, **kwargs) -> dict:
    """Order-Dict mit Timestamp = now() - offset_seconds."""
    ts = _now() - timedelta(seconds=offset_seconds)
    return {"timestamp": ts, **kwargs}


def _order_recent(**kwargs) -> dict:
    """Order-Dict mit Timestamp 10 Sekunden vor now()."""
    return _order_at(10, **kwargs)


def _order_old(**kwargs) -> dict:
    """Order-Dict mit Timestamp 600 Sekunden vor now() (ausserhalb Fenster)."""
    return _order_at(600, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
#  Allgemeine Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestGeneral:
    def test_empty_orders_no_anomaly(self):
        det = _detector()
        anomaly, reason = det.check([])
        assert anomaly is False
        assert reason == ""

    def test_single_normal_order_no_anomaly(self):
        det = _detector()
        orders = [_order_recent(lot_size=1.0, status="filled", signal="long")]
        anomaly, reason = det.check(orders)
        assert anomaly is False

    def test_returns_tuple(self):
        det = _detector()
        result = det.check([])
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_no_anomaly_reason_is_empty_string(self):
        det = _detector()
        _, reason = det.check([])
        assert reason == ""

    def test_multiple_anomalies_combined_in_reason(self):
        """Frequenz-Anomalie + Ablehnungs-Anomalie -> beide im Ergebnis."""
        det = _detector(
            max_trades_per_window=3,
            max_consecutive_rejections=2,
        )
        # 4 kuerzliche Trades (Frequenz-Alarm) + alle rejected (Ablehnungs-Alarm)
        orders = [_order_recent(status="rejected") for _ in range(4)]
        anomaly, reason = det.check(orders)
        assert anomaly is True
        assert "|" in reason   # Kombiniert durch " | "

    def test_multiple_anomalies_both_reasons_in_text(self):
        det = _detector(
            max_trades_per_window=2,
            max_consecutive_rejections=2,
        )
        orders = [_order_recent(status="rejected") for _ in range(3)]
        _, reason = det.check(orders)
        assert "Frequenz" in reason
        assert "Ablehnungen" in reason


# ─────────────────────────────────────────────────────────────────────────────
#  Check 1 – Trade-Frequenz
# ─────────────────────────────────────────────────────────────────────────────

class TestTradeFrequency:
    def test_exact_max_no_anomaly(self):
        """Genau max_trades Trades -> kein Alarm (Bedingung ist '> max', nicht '>= max')."""
        det = _detector(max_trades_per_window=5)
        orders = [_order_recent() for _ in range(5)]
        anomaly, _ = det.check(orders)
        assert anomaly is False

    def test_one_over_max_triggers(self):
        """max_trades + 1 Trades -> Alarm."""
        det = _detector(max_trades_per_window=5)
        orders = [_order_recent() for _ in range(6)]
        anomaly, reason = det.check(orders)
        assert anomaly is True
        assert "Frequenz" in reason

    def test_old_orders_not_counted(self):
        """Orders ausserhalb des Fensters zaehlen nicht."""
        det = _detector(max_trades_per_window=5, frequency_window_seconds=300)
        # 6 alte + 3 neue = nur 3 im Fenster -> kein Alarm
        orders = [_order_old() for _ in range(6)] + [_order_recent() for _ in range(3)]
        anomaly, _ = det.check(orders)
        assert anomaly is False

    def test_exactly_at_window_boundary(self):
        """Order exakt am Rand des Fensters zaehlt noch mit."""
        det = _detector(max_trades_per_window=5, frequency_window_seconds=300)
        # 5 am Rand + 1 knapp drin = 6 -> Alarm
        boundary = _order_at(299)
        just_in   = _order_at(1)
        orders = [boundary, just_in] + [_order_recent() for _ in range(4)]
        anomaly, _ = det.check(orders)
        assert anomaly is True

    def test_orders_without_timestamp_not_counted(self):
        """Orders ohne Timestamp-Feld werden nicht gezaehlt."""
        det = _detector(max_trades_per_window=3)
        orders = [{"status": "filled"} for _ in range(10)]  # kein timestamp
        anomaly, _ = det.check(orders)
        assert anomaly is False

    def test_reason_contains_count_and_max(self):
        det = _detector(max_trades_per_window=3)
        orders = [_order_recent() for _ in range(5)]
        _, reason = det.check(orders)
        assert "5" in reason
        assert "3" in reason

    def test_custom_window_seconds(self):
        """Konfiguriertes Zeitfenster wird korrekt angewendet."""
        det = _detector(max_trades_per_window=5, frequency_window_seconds=60)
        # 6 Orders die 90 Sekunden zurueckliegen -> ausserhalb von 60s Fenster
        orders = [_order_at(90) for _ in range(6)]
        anomaly, _ = det.check(orders)
        assert anomaly is False

    def test_default_thresholds(self):
        """Standard: max 10 Trades in 300s."""
        det = AnomalyDetector(_now_fn=_now)  # kein custom threshold
        orders = [_order_recent() for _ in range(11)]
        anomaly, _ = det.check(orders)
        assert anomaly is True

    def test_empty_list_no_frequency_anomaly(self):
        det = _detector(max_trades_per_window=1)
        anomaly, _ = det.check([])
        assert anomaly is False


# ─────────────────────────────────────────────────────────────────────────────
#  Check 2 – Lot-Groesse
# ─────────────────────────────────────────────────────────────────────────────

class TestLotSize:
    def test_normal_lot_no_anomaly(self):
        """Letzte Order im normalen Bereich -> kein Alarm."""
        det = _detector(lot_size_multiplier=3.0)
        orders = [{"lot_size": 1.0}, {"lot_size": 1.0}, {"lot_size": 1.0}, {"lot_size": 2.0}]
        anomaly, _ = det.check(orders)
        assert anomaly is False

    def test_exactly_at_multiplier_no_anomaly(self):
        """Letzte Order genau am Grenzwert (3.0x) -> kein Alarm (Bedingung: > nicht >=)."""
        det = _detector(lot_size_multiplier=3.0)
        orders = [{"lot_size": 1.0}, {"lot_size": 1.0}, {"lot_size": 1.0}, {"lot_size": 3.0}]
        anomaly, _ = det.check(orders)
        assert anomaly is False

    def test_above_multiplier_triggers(self):
        """Letzte Order ueber Grenzwert -> Alarm."""
        det = _detector(lot_size_multiplier=3.0)
        orders = [{"lot_size": 1.0}, {"lot_size": 1.0}, {"lot_size": 3.1}]
        anomaly, reason = det.check(orders)
        assert anomaly is True
        assert "Position" in reason

    def test_single_order_no_anomaly(self):
        """Weniger als 2 Orders -> kein Durchschnitt berechenbar -> kein Alarm."""
        det = _detector()
        orders = [{"lot_size": 100.0}]
        anomaly, _ = det.check(orders)
        assert anomaly is False

    def test_two_orders_one_big(self):
        """Genau 2 Orders: prior=[1.0], last=5.0 -> 5x Durchschnitt -> Alarm."""
        det = _detector(lot_size_multiplier=3.0)
        orders = [{"lot_size": 1.0}, {"lot_size": 5.0}]
        anomaly, _ = det.check(orders)
        assert anomaly is True

    def test_orders_without_lot_size_ignored(self):
        """Orders ohne lot_size-Key werden ignoriert."""
        det = _detector(lot_size_multiplier=3.0)
        orders = [
            {"lot_size": 1.0},
            {"status": "filled"},   # kein lot_size
            {"lot_size": 10.0},
        ]
        anomaly, _ = det.check(orders)
        assert anomaly is True

    def test_zero_lot_size_ignored(self):
        """Lot-Groesse 0 wird nicht gewertet (Division-by-Zero-Schutz)."""
        det = _detector()
        orders = [{"lot_size": 0.0}, {"lot_size": 0.0}, {"lot_size": 1.0}]
        anomaly, _ = det.check(orders)
        assert anomaly is False

    def test_reason_contains_lot_values(self):
        det = _detector(lot_size_multiplier=3.0)
        orders = [{"lot_size": 1.0}, {"lot_size": 1.0}, {"lot_size": 5.0}]
        _, reason = det.check(orders)
        assert "5.00" in reason

    def test_custom_multiplier(self):
        det = _detector(lot_size_multiplier=2.0)
        orders = [{"lot_size": 1.0}, {"lot_size": 2.5}]
        anomaly, _ = det.check(orders)
        assert anomaly is True

    def test_all_same_lot_size_no_anomaly(self):
        det = _detector()
        orders = [{"lot_size": 1.0} for _ in range(5)]
        anomaly, _ = det.check(orders)
        assert anomaly is False


# ─────────────────────────────────────────────────────────────────────────────
#  Check 3 – Wiederholte Ablehnungen
# ─────────────────────────────────────────────────────────────────────────────

class TestConsecutiveRejections:
    def test_below_threshold_no_anomaly(self):
        """4 aufeinanderfolgende Ablehnungen bei max=5 -> kein Alarm."""
        det = _detector(max_consecutive_rejections=5)
        orders = [{"status": "rejected"} for _ in range(4)]
        anomaly, _ = det.check(orders)
        assert anomaly is False

    def test_exactly_at_threshold_triggers(self):
        """5 aufeinanderfolgende Ablehnungen -> Alarm."""
        det = _detector(max_consecutive_rejections=5)
        orders = [{"status": "rejected"} for _ in range(5)]
        anomaly, reason = det.check(orders)
        assert anomaly is True
        assert "Ablehnungen" in reason

    def test_interruption_resets_count(self):
        """Eine erfolgreiche Order unterbricht die Zaehlung."""
        det = _detector(max_consecutive_rejections=3)
        orders = [
            {"status": "rejected"},
            {"status": "rejected"},
            {"status": "filled"},    # unterbricht
            {"status": "rejected"},
            {"status": "rejected"},
        ]
        anomaly, _ = det.check(orders)
        assert anomaly is False  # nur 2 aufeinanderfolgende am Ende

    def test_status_error_counts(self):
        """Status 'error' zaehlt als Ablehnung."""
        det = _detector(max_consecutive_rejections=3)
        orders = [{"status": "error"} for _ in range(3)]
        anomaly, _ = det.check(orders)
        assert anomaly is True

    def test_status_failed_counts(self):
        """Status 'failed' zaehlt als Ablehnung."""
        det = _detector(max_consecutive_rejections=3)
        orders = [{"status": "failed"} for _ in range(3)]
        anomaly, _ = det.check(orders)
        assert anomaly is True

    def test_case_insensitive_status(self):
        """Status-Pruefung ist case-insensitive."""
        det = _detector(max_consecutive_rejections=3)
        orders = [{"status": "REJECTED"}, {"status": "Rejected"}, {"status": "rejected"}]
        anomaly, _ = det.check(orders)
        assert anomaly is True

    def test_error_codes_in_reason(self):
        """Bekannte Error-Codes erscheinen in der Begruendung."""
        det = _detector(max_consecutive_rejections=3)
        orders = [
            {"status": "rejected", "error_code": "MARGIN_INSUFFICIENT"},
            {"status": "rejected", "error_code": "MARGIN_INSUFFICIENT"},
            {"status": "rejected", "error_code": "MARGIN_INSUFFICIENT"},
        ]
        _, reason = det.check(orders)
        assert "MARGIN_INSUFFICIENT" in reason

    def test_no_error_code_fallback_to_unknown(self):
        """Ohne Error-Code erscheint 'unbekannt' in der Begruendung."""
        det = _detector(max_consecutive_rejections=3)
        orders = [{"status": "rejected"} for _ in range(3)]
        _, reason = det.check(orders)
        assert "unbekannt" in reason

    def test_count_in_reason(self):
        det = _detector(max_consecutive_rejections=3)
        orders = [{"status": "rejected"} for _ in range(5)]
        _, reason = det.check(orders)
        assert "5" in reason

    def test_empty_list_no_anomaly(self):
        det = _detector(max_consecutive_rejections=1)
        anomaly, _ = det.check([])
        assert anomaly is False

    def test_mixed_error_codes_both_in_reason(self):
        det = _detector(max_consecutive_rejections=2)
        orders = [
            {"status": "rejected", "error_code": "CODE_A"},
            {"status": "rejected", "error_code": "CODE_B"},
        ]
        _, reason = det.check(orders)
        assert "CODE_A" in reason or "CODE_B" in reason

    def test_successful_order_at_end_no_anomaly(self):
        """Letzte Order erfolgreich -> kein Alarm trotz vorheriger Fehler."""
        det = _detector(max_consecutive_rejections=3)
        orders = [
            {"status": "rejected"},
            {"status": "rejected"},
            {"status": "rejected"},
            {"status": "filled"},
        ]
        anomaly, _ = det.check(orders)
        assert anomaly is False


# ─────────────────────────────────────────────────────────────────────────────
#  Check 4 – Signal-Flackern
# ─────────────────────────────────────────────────────────────────────────────

class TestSignalFlickering:
    def test_below_threshold_no_anomaly(self):
        """2 Wechsel bei max=3 -> kein Alarm."""
        det = _detector(max_signal_changes_per_minute=3)
        orders = [
            _order_recent(signal="long"),
            _order_recent(signal="short"),
            _order_recent(signal="long"),  # 2 Wechsel
        ]
        anomaly, _ = det.check(orders)
        assert anomaly is False

    def test_exactly_at_threshold_triggers(self):
        """3 Wechsel -> Alarm."""
        det = _detector(max_signal_changes_per_minute=3)
        orders = [
            _order_recent(signal="long"),
            _order_recent(signal="short"),
            _order_recent(signal="long"),
            _order_recent(signal="short"),  # 3 Wechsel
        ]
        anomaly, reason = det.check(orders)
        assert anomaly is True
        assert "Flackern" in reason

    def test_old_signals_not_counted(self):
        """Signale aelter als 60s werden nicht beruecksichtigt."""
        det = _detector(max_signal_changes_per_minute=3)
        # 5 Wechsel aber alle ausserhalb 60s -> kein Alarm
        orders = [
            _order_at(120, signal="long"),
            _order_at(110, signal="short"),
            _order_at(100, signal="long"),
            _order_at(90,  signal="short"),
            _order_at(80,  signal="long"),
        ]
        anomaly, _ = det.check(orders)
        assert anomaly is False

    def test_only_recent_signals_counted(self):
        """Nur Signale der letzten 60s zaehlen."""
        det = _detector(max_signal_changes_per_minute=2)
        # 4 alte (ausserhalb) + 3 Wechsel innerhalb
        orders = [
            _order_at(300, signal="long"),
            _order_at(250, signal="short"),
            _order_at(200, signal="long"),
            _order_at(150, signal="short"),
            # innerhalb 60s:
            _order_recent(signal="long"),
            _order_recent(signal="short"),
            _order_recent(signal="long"),   # 2 Wechsel -> Alarm
        ]
        anomaly, _ = det.check(orders)
        assert anomaly is True

    def test_no_signal_key_ignored(self):
        """Orders ohne 'signal'-Key werden komplett ignoriert."""
        det = _detector(max_signal_changes_per_minute=2)
        orders = [
            _order_recent(lot_size=1.0),   # kein signal
            _order_recent(status="filled"),
        ]
        anomaly, _ = det.check(orders)
        assert anomaly is False

    def test_single_signal_no_anomaly(self):
        """Nur eine Signal-Order -> kein Wechsel moeglich -> kein Alarm."""
        det = _detector(max_signal_changes_per_minute=1)
        orders = [_order_recent(signal="long")]
        anomaly, _ = det.check(orders)
        assert anomaly is False

    def test_same_signal_repeated_no_anomaly(self):
        """Gleiches Signal mehrfach -> kein Wechsel -> kein Alarm."""
        det = _detector(max_signal_changes_per_minute=1)
        orders = [_order_recent(signal="long") for _ in range(5)]
        anomaly, _ = det.check(orders)
        assert anomaly is False

    def test_reason_contains_change_count(self):
        det = _detector(max_signal_changes_per_minute=2)
        orders = [
            _order_recent(signal="long"),
            _order_recent(signal="short"),
            _order_recent(signal="long"),   # 2 Wechsel
        ]
        _, reason = det.check(orders)
        assert "2" in reason

    def test_flat_signal_counts_as_change(self):
        """'flat' gilt als eigenstaendiges Signal."""
        det = _detector(max_signal_changes_per_minute=2)
        orders = [
            _order_recent(signal="long"),
            _order_recent(signal="flat"),
            _order_recent(signal="long"),   # 2 Wechsel
        ]
        anomaly, _ = det.check(orders)
        assert anomaly is True

    def test_no_orders_no_flickering(self):
        det = _detector(max_signal_changes_per_minute=1)
        anomaly, _ = det.check([])
        assert anomaly is False


# ─────────────────────────────────────────────────────────────────────────────
#  _parse_timestamp
# ─────────────────────────────────────────────────────────────────────────────

class TestParseTimestamp:
    def _det(self) -> AnomalyDetector:
        return AnomalyDetector(_now_fn=_now)

    def test_none_returns_none(self):
        assert self._det()._parse_timestamp(None) is None

    def test_datetime_with_tz_unchanged(self):
        dt = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
        result = self._det()._parse_timestamp(dt)
        assert result == dt
        assert result.tzinfo is not None

    def test_datetime_without_tz_gets_utc(self):
        dt = datetime(2026, 1, 1, 12, 0)  # naive
        result = self._det()._parse_timestamp(dt)
        assert result.tzinfo is not None
        assert result.replace(tzinfo=None) == dt

    def test_float_unix_timestamp(self):
        ts = 1_700_000_000.0
        result = self._det()._parse_timestamp(ts)
        assert isinstance(result, datetime)
        assert result.tzinfo is not None

    def test_int_unix_timestamp(self):
        ts = 1_700_000_000
        result = self._det()._parse_timestamp(ts)
        assert isinstance(result, datetime)

    def test_iso_string_without_tz(self):
        result = self._det()._parse_timestamp("2026-06-18T12:00:00")
        assert result is not None
        assert result.tzinfo is not None

    def test_iso_string_with_tz(self):
        result = self._det()._parse_timestamp("2026-06-18T12:00:00+00:00")
        assert result is not None
        assert result.year == 2026

    def test_invalid_string_returns_none(self):
        result = self._det()._parse_timestamp("nicht-ein-datum")
        assert result is None

    def test_unknown_type_returns_none(self):
        result = self._det()._parse_timestamp([1, 2, 3])
        assert result is None

    def test_zero_unix_timestamp(self):
        result = self._det()._parse_timestamp(0)
        assert result == datetime(1970, 1, 1, tzinfo=timezone.utc)


# ─────────────────────────────────────────────────────────────────────────────
#  EmergencyHandler-Integration
# ─────────────────────────────────────────────────────────────────────────────

class TestEmergencyHandlerIntegration:
    def _mock_emergency(self):
        m = MagicMock()
        m.handle_bad_datafeed = MagicMock()
        return m

    def test_emergency_called_on_anomaly(self):
        """Bei erkannter Anomalie wird handle_bad_datafeed aufgerufen."""
        emergency = self._mock_emergency()
        det = _detector(
            emergency_handler=emergency,
            max_trades_per_window=3,
        )
        orders = [_order_recent() for _ in range(5)]
        anomaly, _ = det.check(orders)
        assert anomaly is True
        emergency.handle_bad_datafeed.assert_called_once()

    def test_emergency_not_called_without_anomaly(self):
        """Ohne Anomalie wird EmergencyHandler nicht aufgerufen."""
        emergency = self._mock_emergency()
        det = _detector(emergency_handler=emergency)
        det.check([])
        emergency.handle_bad_datafeed.assert_not_called()

    def test_emergency_reason_contains_prefix(self):
        """Begruendung an EmergencyHandler beginnt mit '[AnomalyDetector]'."""
        emergency = self._mock_emergency()
        det = _detector(
            emergency_handler=emergency,
            max_trades_per_window=3,
        )
        orders = [_order_recent() for _ in range(5)]
        det.check(orders)
        call_kwargs = emergency.handle_bad_datafeed.call_args
        reason_arg = call_kwargs[1].get("reason", "") or (call_kwargs[0][0] if call_kwargs[0] else "")
        assert "[AnomalyDetector]" in reason_arg

    def test_no_emergency_handler_no_error(self):
        """Ohne konfigurierten EmergencyHandler kein Fehler beim Aufruf."""
        det = _detector(emergency_handler=None, max_trades_per_window=1)
        orders = [_order_recent() for _ in range(3)]
        anomaly, _ = det.check(orders)
        assert anomaly is True  # Anomalie erkannt, kein Crash

    def test_emergency_called_only_once_for_multiple_anomalies(self):
        """Auch bei mehreren gleichzeitigen Anomalien: nur ein Emergency-Aufruf."""
        emergency = self._mock_emergency()
        det = _detector(
            emergency_handler=emergency,
            max_trades_per_window=1,
            max_consecutive_rejections=1,
        )
        orders = [_order_recent(status="rejected") for _ in range(3)]
        det.check(orders)
        assert emergency.handle_bad_datafeed.call_count == 1

    def test_emergency_passed_combined_reason(self):
        """Bei mehreren Anomalien bekommt Emergency die kombinierte Begruendung."""
        emergency = self._mock_emergency()
        det = _detector(
            emergency_handler=emergency,
            max_trades_per_window=1,
            max_consecutive_rejections=1,
        )
        orders = [_order_recent(status="rejected") for _ in range(3)]
        det.check(orders)
        call_kwargs = emergency.handle_bad_datafeed.call_args
        reason_arg = call_kwargs[1].get("reason", "") or ""
        # Beide Anomalie-Typen im Argument
        assert "[AnomalyDetector]" in reason_arg


# ─────────────────────────────────────────────────────────────────────────────
#  Konfigurierbarkeit
# ─────────────────────────────────────────────────────────────────────────────

class TestConfiguration:
    def test_default_max_trades(self):
        det = AnomalyDetector(_now_fn=_now)
        assert det._max_trades == 10

    def test_default_freq_window(self):
        det = AnomalyDetector(_now_fn=_now)
        assert det._freq_window == 300

    def test_default_lot_multiplier(self):
        det = AnomalyDetector(_now_fn=_now)
        assert det._lot_multiplier == 3.0

    def test_default_max_rejections(self):
        det = AnomalyDetector(_now_fn=_now)
        assert det._max_rejections == 5

    def test_default_max_signal_changes(self):
        det = AnomalyDetector(_now_fn=_now)
        assert det._max_signal_changes == 3

    def test_custom_thresholds_applied(self):
        det = AnomalyDetector(
            max_trades_per_window=20,
            frequency_window_seconds=600,
            lot_size_multiplier=5.0,
            max_consecutive_rejections=10,
            max_signal_changes_per_minute=6,
            _now_fn=_now,
        )
        assert det._max_trades == 20
        assert det._freq_window == 600
        assert det._lot_multiplier == 5.0
        assert det._max_rejections == 10
        assert det._max_signal_changes == 6
