"""
src/monitoring/telegram_alerts.py
TelegramAlertSender – echte Implementierung des AlertSender-Protocols.

Sendet Benachrichtigungen ueber den Telegram Bot API wenn:
  - Ein Notfall-Eingriff ausgeloest wird (EmergencyHandler)
  - Eine Position eroeffnet oder geschlossen wird
  - Der taeglich Performance-Report versendet werden soll
  - Ein bedingungsbasierter Schwellwert ueberschritten wird

Konfiguration via .env:
  TELEGRAM_BOT_TOKEN  : Bot-Token aus BotFather
  TELEGRAM_CHAT_ID    : Ziel-Chat, Kanal oder Gruppe

Fehler-Philosophie:
  Telegram-Fehler duerfen den Bot-Betrieb NICHT unterbrechen.
  Alle Netzwerk-/HTTP-Fehler werden geloggt und still geschluckt.

Eskalations-Logik:
  Sobald innerhalb von `escalation_window_seconds` mehr als
  `escalation_threshold` Alerts gesendet werden, wird jeder
  weitere Alert mit einem Eskalations-Praefix markiert.

Testbarkeit:
  `_http_post` und `_now_fn` sind injizierbar um
  HTTP-Aufrufe und Zeitstempel-Abhaengigkeiten zu mocken.
"""

from __future__ import annotations

import os
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import requests
from loguru import logger


# ─────────────────────────────────────────────────────────────────────────────
#  Konstanten
# ─────────────────────────────────────────────────────────────────────────────

_API_BASE     = "https://api.telegram.org/bot{token}/sendMessage"
_ICON_ALERT   = "⚠️"
_ICON_ESCALAT = "🚨"
_ICON_OPEN    = "📈"
_ICON_CLOSE   = "📉"
_ICON_REPORT  = "📊"
_ICON_COND    = "🔔"


# ─────────────────────────────────────────────────────────────────────────────
#  TelegramAlertSender
# ─────────────────────────────────────────────────────────────────────────────

class TelegramAlertSender:
    """
    Implementiert das AlertSender-Protocol via Telegram Bot API.

    Parameters
    ----------
    bot_token                  : Telegram Bot Token (BotFather).
    chat_id                    : Ziel-Chat, Kanal oder Gruppen-ID.
    parse_mode                 : 'Markdown' oder 'HTML' (Standard: 'Markdown').
    timeout                    : HTTP-Timeout pro Versuch in Sekunden.
    max_retries                : Maximale Anzahl Wiederholungsversuche.
    retry_delay                : Wartezeit zwischen Versuchen in Sekunden.
    escalation_threshold       : Alerts in escalation_window_seconds bis zur Eskalation.
    escalation_window_seconds  : Beobachtungsfenster fuer Eskalation.
    _http_post                 : Injectable fuer Tests (ersetzt requests.post).
    _now_fn                    : Injectable fuer Tests (ersetzt time.monotonic).
    """

    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        parse_mode: str = "Markdown",
        timeout: int = 10,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        escalation_threshold: int = 5,
        escalation_window_seconds: int = 300,
        _http_post: Optional[Callable[..., Any]] = None,
        _now_fn: Optional[Callable[[], float]] = None,
    ) -> None:
        bot_token = bot_token.strip()
        chat_id   = chat_id.strip()

        if not bot_token:
            raise ValueError(
                "bot_token darf nicht leer sein. "
                "Wert aus TELEGRAM_BOT_TOKEN pruefen."
            )
        if not chat_id:
            raise ValueError(
                "chat_id darf nicht leer sein. "
                "Wert aus TELEGRAM_CHAT_ID pruefen."
            )

        self._token        = bot_token
        self._chat_id      = chat_id
        self._parse_mode   = parse_mode
        self._timeout      = timeout
        self._max_retries  = max(1, max_retries)
        self._retry_delay  = max(0.0, retry_delay)

        self._escalation_threshold = escalation_threshold
        self._escalation_window_s  = escalation_window_seconds
        self._alert_times: deque[float] = deque()

        self._http_post = _http_post or requests.post
        self._now_fn    = _now_fn or time.monotonic

    # ─────────────────────────────────────────────────────────────────────────
    #  Oeffentliche Schnittstelle
    # ─────────────────────────────────────────────────────────────────────────

    def send_alert(self, message: str) -> None:
        """
        Sendet einen Notfall-Alert. Implementiert AlertSender-Protocol.

        Automatische Eskalationsmarkierung wenn mehr als
        `escalation_threshold` Alerts im Beobachtungsfenster gesendet wurden.
        """
        if not message:
            logger.debug("TelegramAlertSender: leere Nachricht uebersprungen.")
            return

        escalated = self._track_and_check_escalation()

        if escalated:
            text = (
                f"{_ICON_ESCALAT} *ESKALATION* – wiederholt kritische Fehler\n\n"
                f"{message}"
            )
        else:
            text = f"{_ICON_ALERT} {message}"

        self._send_message(text)

    def send_position_opened(
        self,
        symbol: str,
        direction: str,
        lot_size: float,
        price: float,
        ticket: int | str | None = None,
        **kwargs: Any,
    ) -> None:
        """Sendet Benachrichtigung ueber eine neu eroeffnete Position."""
        ticket_str = f" | Ticket: `{ticket}`" if ticket is not None else ""
        text = (
            f"{_ICON_OPEN} *Position eroeffnet*{ticket_str}\n"
            f"Symbol:    `{symbol}`\n"
            f"Richtung:  `{direction.upper()}`\n"
            f"Lot-Groesse: `{lot_size:.2f}`\n"
            f"Preis:     `{price:.5f}`"
        )
        self._send_message(text)

    def send_position_closed(
        self,
        symbol: str,
        direction: str,
        pnl: float,
        ticket: int | str | None = None,
        reason: str = "",
        **kwargs: Any,
    ) -> None:
        """Sendet Benachrichtigung ueber eine geschlossene Position."""
        pnl_icon   = "✅" if pnl >= 0 else "❌"
        sign       = "+" if pnl >= 0 else ""
        ticket_str = f" | Ticket: `{ticket}`" if ticket is not None else ""
        reason_str = f"\nGrund: `{reason}`" if reason else ""
        text = (
            f"{_ICON_CLOSE} *Position geschlossen*{ticket_str}\n"
            f"Symbol:   `{symbol}`\n"
            f"Richtung: `{direction.upper()}`\n"
            f"P&L:      {pnl_icon} `{sign}{pnl:.2f}`{reason_str}"
        )
        self._send_message(text)

    def send_daily_report(self, stats: dict) -> None:
        """
        Sendet den taeglichen Performance-Report.

        Erwartet das Rueckgabe-Dict von TradeJournal.calculate_stats():
          n_trades, win_rate, profit_factor, avg_win, avg_loss,
          total_pnl, best_trade, worst_trade.
        """
        n        = stats.get("n_trades", 0)
        wr       = stats.get("win_rate", 0.0)
        pf       = stats.get("profit_factor", 0.0)
        total    = stats.get("total_pnl", 0.0)
        best     = stats.get("best_trade")
        worst    = stats.get("worst_trade")
        avg_win  = stats.get("avg_win", 0.0)
        avg_loss = stats.get("avg_loss", 0.0)

        pf_str    = "∞" if pf == float("inf") else f"{pf:.2f}"
        best_str  = f"+{best:.2f}" if best is not None else "--"
        worst_str = f"{worst:.2f}" if worst is not None else "--"
        sign      = "+" if total >= 0 else ""
        total_icon = "📈" if total >= 0 else "📉"

        text = (
            f"{_ICON_REPORT} *Taeglicherlicher Performance-Report*\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"Trades:          `{n}`\n"
            f"Win-Rate:        `{wr * 100:.1f}%`\n"
            f"Gewinnfaktor:    `{pf_str}`\n"
            f"Ø Gewinn:        `+{avg_win:.2f}`\n"
            f"Ø Verlust:       `-{avg_loss:.2f}`\n"
            f"Bester Trade:    `{best_str}`\n"
            f"Schlechtester:   `{worst_str}`\n"
            f"Gesamt-P&L:      {total_icon} `{sign}{total:.2f}`"
        )
        self._send_message(text)

    def send_conditional_alert(
        self,
        condition_name: str,
        current_value: float,
        threshold: float,
        message: Optional[str] = None,
    ) -> None:
        """
        Sendet Alert nur wenn `current_value >= threshold`.

        Parameters
        ----------
        condition_name : Bezeichner der Bedingung (z.B. 'Drawdown', 'Spread').
        current_value  : Aktuell gemessener Wert.
        threshold      : Schwellwert – Alert wird gesendet wenn ueberschritten.
        message        : Optionaler benutzerdefinierter Text; wird auto-generiert wenn None.
        """
        if current_value < threshold:
            return

        text = message or (
            f"{_ICON_COND} *Schwellwert ueberschritten*\n"
            f"Bedingung:      `{condition_name}`\n"
            f"Aktuell:        `{current_value:.4f}`\n"
            f"Schwellwert:    `{threshold:.4f}`"
        )
        self.send_alert(text)

    # ─────────────────────────────────────────────────────────────────────────
    #  Konstruktor-Alternativen
    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def from_env(
        cls,
        token_var: str = "TELEGRAM_BOT_TOKEN",
        chat_id_var: str = "TELEGRAM_CHAT_ID",
        **kwargs: Any,
    ) -> "TelegramAlertSender":
        """
        Erstellt Instanz aus Umgebungsvariablen (.env / System-Environment).

        Laedt .env automatisch mit python-dotenv wenn vorhanden.

        Parameters
        ----------
        token_var   : Name der Umgebungsvariable fuer das Bot-Token.
        chat_id_var : Name der Umgebungsvariable fuer die Chat-ID.
        **kwargs    : Weitere Parameter fuer __init__ (z.B. max_retries).

        Raises
        ------
        RuntimeError : Wenn Token oder Chat-ID nicht gesetzt sind.
        """
        try:
            from dotenv import load_dotenv
            load_dotenv(override=False)
        except ImportError:
            pass

        token   = os.environ.get(token_var, "").strip()
        chat_id = os.environ.get(chat_id_var, "").strip()

        if not token:
            raise RuntimeError(
                f"Umgebungsvariable {token_var!r} nicht gesetzt. "
                "Wert in .env oder System-Umgebung eintragen."
            )
        if not chat_id:
            raise RuntimeError(
                f"Umgebungsvariable {chat_id_var!r} nicht gesetzt. "
                "Wert in .env oder System-Umgebung eintragen."
            )

        logger.info(
            "TelegramAlertSender: konfiguriert via Umgebung | chat_id={cid}",
            cid=chat_id,
        )
        return cls(bot_token=token, chat_id=chat_id, **kwargs)

    # ─────────────────────────────────────────────────────────────────────────
    #  Interna
    # ─────────────────────────────────────────────────────────────────────────

    def _send_message(self, text: str) -> bool:
        """
        Sendet einen Text via Telegram Bot API.

        Versucht es bis zu `max_retries`-mal.
        Fehler werden geloggt, niemals nach aussen propagiert.

        Returns
        -------
        True bei Erfolg, False wenn alle Versuche fehlgeschlagen sind.
        """
        url = _API_BASE.format(token=self._token)
        payload = {
            "chat_id":    self._chat_id,
            "text":       text,
            "parse_mode": self._parse_mode,
        }

        for attempt in range(self._max_retries):
            try:
                resp = self._http_post(url, json=payload, timeout=self._timeout)
                if resp.status_code == 200:
                    logger.debug(
                        "Telegram: Nachricht gesendet | Versuch {n}/{m}",
                        n=attempt + 1, m=self._max_retries,
                    )
                    return True

                logger.warning(
                    "Telegram: HTTP {code} | Versuch {n}/{m} | {body}",
                    code=resp.status_code,
                    n=attempt + 1,
                    m=self._max_retries,
                    body=getattr(resp, "text", "")[:200],
                )

            except requests.RequestException as exc:
                logger.warning(
                    "Telegram: Netzwerkfehler | {exc} | Versuch {n}/{m}",
                    exc=exc, n=attempt + 1, m=self._max_retries,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Telegram: Unerwarteter Fehler | {exc}", exc=exc
                )
                return False

            if attempt < self._max_retries - 1 and self._retry_delay > 0:
                time.sleep(self._retry_delay)

        logger.error(
            "Telegram: Nachricht konnte nach {n} Versuchen nicht gesendet werden.",
            n=self._max_retries,
        )
        return False

    def _track_and_check_escalation(self) -> bool:
        """
        Verfolgt Alert-Zeitstempel und prueft ob Eskalationsschwelle erreicht.

        Returns True wenn mehr als `escalation_threshold` Alerts im
        Beobachtungsfenster registriert wurden.
        """
        now     = self._now_fn()
        cutoff  = now - self._escalation_window_s

        # Alte Zeitstempel ausserhalb des Fensters entfernen
        while self._alert_times and self._alert_times[0] < cutoff:
            self._alert_times.popleft()

        self._alert_times.append(now)

        return len(self._alert_times) > self._escalation_threshold

    @property
    def escalation_count(self) -> int:
        """Aktuelle Anzahl Alerts im Eskalations-Beobachtungsfenster (fuer Tests)."""
        now    = self._now_fn()
        cutoff = now - self._escalation_window_s
        return sum(1 for t in self._alert_times if t >= cutoff)
