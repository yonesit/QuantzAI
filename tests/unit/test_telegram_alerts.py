"""
tests/unit/test_telegram_alerts.py
Unit-Tests fuer TelegramAlertSender.

Alle HTTP-Calls werden mit Mocks ersetzt – kein echter Netzwerkzugriff.

Abgedeckte Bereiche:
  Initialisierung
    - Gueltige Parameter akzeptiert
    - Leere Token/Chat-ID wirft ValueError

  AlertSender-Protocol
    - isinstance-Check gegen AlertSender

  send_alert
    - Sendet bei Erfolg (HTTP 200)
    - Leere Nachricht wird uebersprungen
    - Netzwerkfehler blockiert NICHT (kein raise)
    - HTTP-Fehler (non-200) blockiert NICHT
    - Unbekannte Exception blockiert NICHT

  Retry-Logik
    - Wiederholt bei Netzwerkfehler bis max_retries
    - Sendet bei Erfolg im 2. Versuch
    - Gibt False zurueck wenn alle Versuche fehlschlagen
    - Timeout wird an requests.post weitergegeben

  Eskalations-Logik
    - Kein Praefix unter Schwellwert
    - Eskalations-Praefix ab Schwellwert
    - Fenster-Bereinigung nach Ablauf

  send_conditional_alert
    - Sendet wenn current_value >= threshold
    - Kein Alert wenn current_value < threshold
    - Benutzerdefinierter Text moeglich

  send_position_opened
    - Enthaelt Symbol, Richtung, Lots, Preis
    - Optionales Ticket enthalten wenn uebergeben

  send_position_closed
    - Enthaelt Symbol, P&L
    - Unterschiedliche Icons fuer Gewinn/Verlust
    - Optionaler Grund im Text

  send_daily_report
    - Enthaelt n_trades, win_rate, total_pnl
    - Infinity-Gewinnfaktor wird als Symbol dargestellt
    - Kein Crash bei leeren Stats

  from_env
    - Liest BOT_TOKEN und CHAT_ID aus Environment
    - Wirft RuntimeError bei fehlendem Token
    - Wirft RuntimeError bei fehlender Chat-ID
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch, call

import pytest
import requests

from src.execution.emergency import AlertSender
from src.monitoring.telegram_alerts import TelegramAlertSender


# ─────────────────────────────────────────────────────────────────────────────
#  Hilfsobjekte
# ─────────────────────────────────────────────────────────────────────────────

class _Resp:
    """Minimaler Mock fuer requests.Response."""
    def __init__(self, status_code: int = 200, text: str = '{"ok":true}') -> None:
        self.status_code = status_code
        self.text        = text

    def json(self) -> dict:
        import json
        return json.loads(self.text)


def _ok_post(*args, **kwargs) -> _Resp:
    return _Resp(200)


def _fail_post(*args, **kwargs) -> _Resp:
    return _Resp(400, '{"ok":false,"description":"Bad Request"}')


def _raising_post(*args, **kwargs):
    raise requests.RequestException("connection refused")


def _sender(
    *,
    post_fn=_ok_post,
    max_retries: int = 3,
    retry_delay: float = 0.0,
    escalation_threshold: int = 5,
    escalation_window_seconds: int = 300,
    now_fn=None,
    token: str = "TEST_TOKEN",
    chat_id: str = "TEST_CHAT",
) -> TelegramAlertSender:
    """Erstellt eine vorkonfigurierte Test-Instanz."""
    return TelegramAlertSender(
        bot_token=token,
        chat_id=chat_id,
        max_retries=max_retries,
        retry_delay=retry_delay,
        escalation_threshold=escalation_threshold,
        escalation_window_seconds=escalation_window_seconds,
        _http_post=post_fn,
        _now_fn=now_fn,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Initialisierung
# ─────────────────────────────────────────────────────────────────────────────

class TestInit:
    def test_creates_with_valid_params(self):
        s = _sender()
        assert s is not None

    def test_empty_token_raises_value_error(self):
        with pytest.raises(ValueError, match="bot_token"):
            TelegramAlertSender(bot_token="", chat_id="123")

    def test_empty_chat_id_raises_value_error(self):
        with pytest.raises(ValueError, match="chat_id"):
            TelegramAlertSender(bot_token="tok", chat_id="")

    def test_whitespace_token_raises_value_error(self):
        with pytest.raises(ValueError):
            TelegramAlertSender(bot_token="   ", chat_id="123")

    def test_implements_alert_sender_protocol(self):
        s = _sender()
        assert isinstance(s, AlertSender)

    def test_has_send_alert_method(self):
        s = _sender()
        assert callable(s.send_alert)

    def test_max_retries_at_least_one(self):
        s = _sender(max_retries=0)
        assert s._max_retries >= 1

    def test_retry_delay_non_negative(self):
        s = _sender(retry_delay=-5.0)
        assert s._retry_delay >= 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  send_alert – Basisverhalten
# ─────────────────────────────────────────────────────────────────────────────

class TestSendAlert:
    def test_success_calls_http_post(self):
        calls = []
        def post_fn(url, **kw):
            calls.append((url, kw))
            return _Resp(200)

        s = _sender(post_fn=post_fn)
        s.send_alert("Test-Nachricht")
        assert len(calls) == 1

    def test_url_contains_token(self):
        calls = []
        def post_fn(url, **kw):
            calls.append(url)
            return _Resp(200)

        s = _sender(post_fn=post_fn, token="MY_SECRET_TOKEN")
        s.send_alert("x")
        assert "MY_SECRET_TOKEN" in calls[0]

    def test_payload_contains_chat_id(self):
        calls = []
        def post_fn(url, **kw):
            calls.append(kw)
            return _Resp(200)

        s = _sender(post_fn=post_fn, chat_id="CHAT_123")
        s.send_alert("x")
        assert calls[0]["json"]["chat_id"] == "CHAT_123"

    def test_payload_contains_message_text(self):
        calls = []
        def post_fn(url, **kw):
            calls.append(kw)
            return _Resp(200)

        s = _sender(post_fn=post_fn)
        s.send_alert("Kritischer Fehler!")
        assert "Kritischer Fehler!" in calls[0]["json"]["text"]

    def test_empty_message_skipped(self):
        calls = []
        s = _sender(post_fn=lambda *a, **kw: (calls.append(1), _Resp(200))[1])
        s.send_alert("")
        assert len(calls) == 0

    def test_network_error_does_not_raise(self):
        s = _sender(post_fn=_raising_post, max_retries=1)
        s.send_alert("Test")  # darf nicht werfen

    def test_http_400_does_not_raise(self):
        s = _sender(post_fn=_fail_post, max_retries=1)
        s.send_alert("Test")  # darf nicht werfen

    def test_unexpected_exception_does_not_raise(self):
        def exploding_post(*a, **kw):
            raise RuntimeError("unerwarteter Fehler")

        s = _sender(post_fn=exploding_post, max_retries=1)
        s.send_alert("Test")  # darf nicht werfen

    def test_timeout_passed_to_post(self):
        calls = []
        def post_fn(url, **kw):
            calls.append(kw.get("timeout"))
            return _Resp(200)

        s = TelegramAlertSender(
            bot_token="tok", chat_id="cid",
            timeout=42, _http_post=post_fn,
        )
        s.send_alert("x")
        assert calls[0] == 42

    def test_returns_true_on_success(self):
        s = _sender()
        result = s._send_message("Test")
        assert result is True

    def test_returns_false_after_all_retries_fail(self):
        s = _sender(post_fn=_raising_post, max_retries=2, retry_delay=0.0)
        result = s._send_message("Test")
        assert result is False


# ─────────────────────────────────────────────────────────────────────────────
#  Retry-Logik
# ─────────────────────────────────────────────────────────────────────────────

class TestRetryLogic:
    def test_retries_on_network_error(self):
        attempts = []
        def post_fn(url, **kw):
            attempts.append(1)
            raise requests.RequestException("timeout")

        s = _sender(post_fn=post_fn, max_retries=3, retry_delay=0.0)
        s._send_message("Test")
        assert len(attempts) == 3

    def test_retries_on_http_error(self):
        attempts = []
        def post_fn(url, **kw):
            attempts.append(1)
            return _Resp(500)

        s = _sender(post_fn=post_fn, max_retries=3, retry_delay=0.0)
        s._send_message("Test")
        assert len(attempts) == 3

    def test_success_on_second_attempt(self):
        attempts = []
        def post_fn(url, **kw):
            attempts.append(1)
            if len(attempts) < 2:
                raise requests.RequestException("first fail")
            return _Resp(200)

        s = _sender(post_fn=post_fn, max_retries=3, retry_delay=0.0)
        result = s._send_message("Test")
        assert result is True
        assert len(attempts) == 2

    def test_single_attempt_when_max_retries_one(self):
        attempts = []
        def post_fn(url, **kw):
            attempts.append(1)
            raise requests.RequestException("fail")

        s = _sender(post_fn=post_fn, max_retries=1, retry_delay=0.0)
        s._send_message("Test")
        assert len(attempts) == 1


# ─────────────────────────────────────────────────────────────────────────────
#  Eskalations-Logik
# ─────────────────────────────────────────────────────────────────────────────

class TestEscalation:
    def _make_clock(self, start: float = 0.0):
        """Gibt eine steuerbare Uhr-Funktion zurueck."""
        t = [start]
        def now():
            return t[0]
        def advance(seconds: float):
            t[0] += seconds
        return now, advance

    def test_no_escalation_prefix_below_threshold(self):
        sent_texts = []
        def post_fn(url, **kw):
            sent_texts.append(kw["json"]["text"])
            return _Resp(200)

        now_fn, _ = self._make_clock()
        # threshold=5, send only 4 alerts
        s = _sender(post_fn=post_fn, escalation_threshold=5, now_fn=now_fn)
        for _ in range(4):
            s.send_alert("Fehler")

        assert all("ESKALATION" not in t for t in sent_texts)

    def test_escalation_prefix_at_threshold(self):
        sent_texts = []
        def post_fn(url, **kw):
            sent_texts.append(kw["json"]["text"])
            return _Resp(200)

        now_fn, _ = self._make_clock()
        # threshold=3 -> 4. Alert wird eskaliert
        s = _sender(post_fn=post_fn, escalation_threshold=3, now_fn=now_fn)
        for _ in range(4):
            s.send_alert("Fehler")

        assert "ESKALATION" in sent_texts[-1]

    def test_escalation_count_increases(self):
        now_fn, _ = self._make_clock()
        s = _sender(escalation_threshold=10, now_fn=now_fn)
        assert s.escalation_count == 0
        s.send_alert("x")
        s.send_alert("y")
        assert s.escalation_count == 2

    def test_old_alerts_removed_from_window(self):
        """Alerts ausserhalb des Fensters werden nicht gezaehlt."""
        now_fn, advance = self._make_clock()
        s = _sender(
            escalation_threshold=10,
            escalation_window_seconds=60,
            now_fn=now_fn,
        )
        for _ in range(5):
            s.send_alert("alt")

        advance(120)  # Fenster abgelaufen

        for _ in range(2):
            s.send_alert("neu")

        # nur 2 Alerts im aktuellen Fenster
        assert s.escalation_count == 2

    def test_alert_icon_in_non_escalated_message(self):
        sent_texts = []
        def post_fn(url, **kw):
            sent_texts.append(kw["json"]["text"])
            return _Resp(200)

        now_fn, _ = self._make_clock()
        s = _sender(post_fn=post_fn, escalation_threshold=10, now_fn=now_fn)
        s.send_alert("Normaler Fehler")
        assert "⚠" in sent_texts[0]


# ─────────────────────────────────────────────────────────────────────────────
#  send_conditional_alert
# ─────────────────────────────────────────────────────────────────────────────

class TestConditionalAlert:
    def test_sends_when_condition_met(self):
        calls = []
        s = _sender(post_fn=lambda *a, **kw: (calls.append(1), _Resp(200))[1])
        s.send_conditional_alert("Drawdown", current_value=12.0, threshold=10.0)
        assert len(calls) == 1

    def test_no_send_when_below_threshold(self):
        calls = []
        s = _sender(post_fn=lambda *a, **kw: (calls.append(1), _Resp(200))[1])
        s.send_conditional_alert("Drawdown", current_value=5.0, threshold=10.0)
        assert len(calls) == 0

    def test_sends_when_exactly_at_threshold(self):
        calls = []
        s = _sender(post_fn=lambda *a, **kw: (calls.append(1), _Resp(200))[1])
        s.send_conditional_alert("X", current_value=10.0, threshold=10.0)
        assert len(calls) == 1

    def test_condition_name_in_auto_message(self):
        sent_texts = []
        def post_fn(url, **kw):
            sent_texts.append(kw["json"]["text"])
            return _Resp(200)

        s = _sender(post_fn=post_fn)
        s.send_conditional_alert("Spread", current_value=5.0, threshold=3.0)
        assert "Spread" in sent_texts[0]

    def test_custom_message_used_when_provided(self):
        sent_texts = []
        def post_fn(url, **kw):
            sent_texts.append(kw["json"]["text"])
            return _Resp(200)

        s = _sender(post_fn=post_fn)
        s.send_conditional_alert("X", 5.0, 3.0, message="Mein Alert-Text")
        # custom message wird durch send_alert weitergeleitet
        assert "Mein Alert-Text" in sent_texts[0]

    def test_current_value_in_auto_message(self):
        sent_texts = []
        def post_fn(url, **kw):
            sent_texts.append(kw["json"]["text"])
            return _Resp(200)

        s = _sender(post_fn=post_fn)
        s.send_conditional_alert("Drawdown", current_value=13.5, threshold=10.0)
        assert "13.5" in sent_texts[0]


# ─────────────────────────────────────────────────────────────────────────────
#  send_position_opened
# ─────────────────────────────────────────────────────────────────────────────

class TestPositionOpened:
    def test_sends_message(self):
        calls = []
        s = _sender(post_fn=lambda *a, **kw: (calls.append(kw["json"]["text"]), _Resp(200))[1])
        s.send_position_opened("EURUSD", "long", 0.10, 1.0850)
        assert len(calls) == 1

    def test_contains_symbol(self):
        sent = []
        s = _sender(post_fn=lambda *a, **kw: (sent.append(kw["json"]["text"]), _Resp(200))[1])
        s.send_position_opened("GBPUSD", "short", 0.05, 1.2700)
        assert "GBPUSD" in sent[0]

    def test_contains_direction(self):
        sent = []
        s = _sender(post_fn=lambda *a, **kw: (sent.append(kw["json"]["text"]), _Resp(200))[1])
        s.send_position_opened("EURUSD", "long", 0.10, 1.0850)
        assert "LONG" in sent[0]

    def test_contains_lot_size(self):
        sent = []
        s = _sender(post_fn=lambda *a, **kw: (sent.append(kw["json"]["text"]), _Resp(200))[1])
        s.send_position_opened("EURUSD", "buy", 0.25, 1.0850)
        assert "0.25" in sent[0]

    def test_contains_price(self):
        sent = []
        s = _sender(post_fn=lambda *a, **kw: (sent.append(kw["json"]["text"]), _Resp(200))[1])
        s.send_position_opened("EURUSD", "buy", 0.10, 1.08500)
        assert "1.08500" in sent[0]

    def test_ticket_included_when_provided(self):
        sent = []
        s = _sender(post_fn=lambda *a, **kw: (sent.append(kw["json"]["text"]), _Resp(200))[1])
        s.send_position_opened("EURUSD", "buy", 0.10, 1.0850, ticket=42)
        assert "42" in sent[0]

    def test_no_crash_without_ticket(self):
        s = _sender()
        s.send_position_opened("EURUSD", "buy", 0.10, 1.0850)  # kein Crash


# ─────────────────────────────────────────────────────────────────────────────
#  send_position_closed
# ─────────────────────────────────────────────────────────────────────────────

class TestPositionClosed:
    def test_sends_message(self):
        calls = []
        s = _sender(post_fn=lambda *a, **kw: (calls.append(1), _Resp(200))[1])
        s.send_position_closed("EURUSD", "long", pnl=15.50)
        assert len(calls) == 1

    def test_contains_symbol(self):
        sent = []
        s = _sender(post_fn=lambda *a, **kw: (sent.append(kw["json"]["text"]), _Resp(200))[1])
        s.send_position_closed("USDJPY", "short", pnl=-8.0)
        assert "USDJPY" in sent[0]

    def test_positive_pnl_has_profit_icon(self):
        sent = []
        s = _sender(post_fn=lambda *a, **kw: (sent.append(kw["json"]["text"]), _Resp(200))[1])
        s.send_position_closed("EURUSD", "long", pnl=20.0)
        assert "✅" in sent[0]

    def test_negative_pnl_has_loss_icon(self):
        sent = []
        s = _sender(post_fn=lambda *a, **kw: (sent.append(kw["json"]["text"]), _Resp(200))[1])
        s.send_position_closed("EURUSD", "long", pnl=-10.0)
        assert "❌" in sent[0]

    def test_reason_in_message_when_provided(self):
        sent = []
        s = _sender(post_fn=lambda *a, **kw: (sent.append(kw["json"]["text"]), _Resp(200))[1])
        s.send_position_closed("EURUSD", "long", pnl=5.0, reason="Stop-Loss")
        assert "Stop-Loss" in sent[0]

    def test_no_crash_without_optional_params(self):
        s = _sender()
        s.send_position_closed("EURUSD", "long", pnl=0.0)


# ─────────────────────────────────────────────────────────────────────────────
#  send_daily_report
# ─────────────────────────────────────────────────────────────────────────────

class TestDailyReport:
    def _stats(self, **overrides) -> dict:
        base = {
            "n_trades":      10,
            "win_rate":      0.60,
            "profit_factor": 1.8,
            "avg_win":       25.0,
            "avg_loss":      15.0,
            "total_pnl":     120.0,
            "best_trade":    50.0,
            "worst_trade":  -30.0,
        }
        base.update(overrides)
        return base

    def test_sends_message(self):
        calls = []
        s = _sender(post_fn=lambda *a, **kw: (calls.append(1), _Resp(200))[1])
        s.send_daily_report(self._stats())
        assert len(calls) == 1

    def test_contains_trade_count(self):
        sent = []
        s = _sender(post_fn=lambda *a, **kw: (sent.append(kw["json"]["text"]), _Resp(200))[1])
        s.send_daily_report(self._stats(n_trades=7))
        assert "7" in sent[0]

    def test_contains_win_rate(self):
        sent = []
        s = _sender(post_fn=lambda *a, **kw: (sent.append(kw["json"]["text"]), _Resp(200))[1])
        s.send_daily_report(self._stats(win_rate=0.65))
        assert "65" in sent[0]

    def test_contains_total_pnl(self):
        sent = []
        s = _sender(post_fn=lambda *a, **kw: (sent.append(kw["json"]["text"]), _Resp(200))[1])
        s.send_daily_report(self._stats(total_pnl=250.50))
        assert "250.50" in sent[0]

    def test_infinity_profit_factor_shown_as_symbol(self):
        sent = []
        s = _sender(post_fn=lambda *a, **kw: (sent.append(kw["json"]["text"]), _Resp(200))[1])
        s.send_daily_report(self._stats(profit_factor=float("inf")))
        assert "∞" in sent[0]

    def test_empty_stats_no_crash(self):
        s = _sender()
        s.send_daily_report({})  # kein Crash bei leeren Stats

    def test_none_best_worst_shown_as_placeholder(self):
        sent = []
        s = _sender(post_fn=lambda *a, **kw: (sent.append(kw["json"]["text"]), _Resp(200))[1])
        s.send_daily_report(self._stats(best_trade=None, worst_trade=None))
        assert "--" in sent[0]


# ─────────────────────────────────────────────────────────────────────────────
#  from_env
# ─────────────────────────────────────────────────────────────────────────────

class TestFromEnv:
    def test_reads_token_and_chat_from_env(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "my_token_123")
        monkeypatch.setenv("TELEGRAM_CHAT_ID",   "my_chat_456")
        s = TelegramAlertSender.from_env(_http_post=_ok_post)
        assert s._token    == "my_token_123"
        assert s._chat_id  == "my_chat_456"

    def test_missing_token_raises_runtime_error(self, monkeypatch):
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "cid")
        with pytest.raises(RuntimeError, match="TELEGRAM_BOT_TOKEN"):
            TelegramAlertSender.from_env()

    def test_missing_chat_id_raises_runtime_error(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
        monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
        with pytest.raises(RuntimeError, match="TELEGRAM_CHAT_ID"):
            TelegramAlertSender.from_env()

    def test_custom_env_var_names(self, monkeypatch):
        monkeypatch.setenv("MY_BOT", "custom_token")
        monkeypatch.setenv("MY_CHAT", "custom_chat")
        s = TelegramAlertSender.from_env(
            token_var="MY_BOT",
            chat_id_var="MY_CHAT",
            _http_post=_ok_post,
        )
        assert s._token   == "custom_token"
        assert s._chat_id == "custom_chat"

    def test_kwargs_passed_to_init(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
        monkeypatch.setenv("TELEGRAM_CHAT_ID",   "cid")
        s = TelegramAlertSender.from_env(max_retries=7, _http_post=_ok_post)
        assert s._max_retries == 7
