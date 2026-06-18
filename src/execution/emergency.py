"""
src/execution/emergency.py
EmergencyHandler – definierte Reaktionen fuer jeden Fehlerfall.

Notfall-Szenarien:
  1. MT5_UNREACHABLE  – MT5 nach 3 Retries nicht erreichbar.
                        Offene Positionen pruefen, OANDA-Fallback via DataRouter/PriceValidator.
  2. BAD_DATAFEED     – Datenfeed liefert leere/fehlerhafte Daten.
                        Handel pausieren, kein Blind-Trading.
  3. CRITICAL_DRAWDOWN– Maximaler Drawdown erreicht (RiskGuard).
                        Alle Positionen schliessen, Bot pausieren bis manuelle Freigabe.
  4. UNHANDLED_EXCEPTION – Unbehandelte Exception im Hauptloop.
                        Alle Positionen pruefen + schliessen, Alert senden, Prozess beenden.

Jede Notfall-Aktion wird im Audit-Log mit Zeitstempel und Grund festgehalten.
In Issue #19 wird das durch eine echte Audit-DB ersetzt/ergaenzt.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from loguru import logger

from src.data.data_router import DataRouter, PriceDiscrepancyError, EmergencyModeError
from src.execution.order_executor import OrderExecutor, OrderError
from src.risk.risk_guard import RiskGuard


# ─────────────────────────────────────────────────────────────────────────────
#  Alert-Interface
# ─────────────────────────────────────────────────────────────────────────────

@runtime_checkable
class AlertSender(Protocol):
    """
    Interface fuer Notfall-Benachrichtigungen.

    Wird spaeter mit echtem Telegram-Bot, E-Mail oder anderen
    Notification-Services implementiert. Der EmergencyHandler
    haengt nur von diesem Protocol ab, nicht von einer konkreten Klasse.
    """

    def send_alert(self, message: str) -> None: ...


class LogOnlyAlertSender:
    """
    Fallback-AlertSender: loggt nur, sendet nichts extern.
    Geeignet fuer Entwicklungsumgebungen ohne Telegram-Zugang.
    """

    def send_alert(self, message: str) -> None:
        logger.warning("ALERT (kein Sender konfiguriert): {msg}", msg=message)


# ─────────────────────────────────────────────────────────────────────────────
#  EmergencyHandler
# ─────────────────────────────────────────────────────────────────────────────

class EmergencyHandler:
    """
    Reagiert auf Fehlerszenarien mit klar definierten Aktionen.

    Parameters
    ----------
    executor        : OrderExecutor zum Schliessen offener Positionen.
    data_router     : DataRouter mit PriceValidator fuer OANDA-Fallback-Pruefung.
    risk_guard      : RiskGuard zum Pruefen des Drawdown-Status.
    alert_sender    : AlertSender-Implementierung (Telegram, E-Mail, etc.).
                      Faellt auf LogOnlyAlertSender zurueck wenn None.
    audit_log_path  : Optionaler Pfad fuer strukturiertes Audit-Log (Textdatei).
                      Wird in Issue #19 durch echte Audit-DB ersetzt.
    _exit_fn        : Injizierbar fuer Tests statt sys.exit().
    """

    def __init__(
        self,
        executor: OrderExecutor,
        data_router: DataRouter,
        risk_guard: RiskGuard,
        alert_sender: AlertSender | None = None,
        audit_log_path: str | Path | None = None,
        _exit_fn=None,
    ) -> None:
        self._executor    = executor
        self._data_router = data_router
        self._risk_guard  = risk_guard
        self._alert       = alert_sender or LogOnlyAlertSender()
        self._audit_path  = Path(audit_log_path) if audit_log_path else None
        self._exit_fn     = _exit_fn or sys.exit
        self._trading_paused = False

    # ── Status ────────────────────────────────────────────────────────────────

    @property
    def is_trading_paused(self) -> bool:
        """True wenn der Handel durch einen Notfall-Handler pausiert wurde."""
        return self._trading_paused

    def resume_trading(self) -> None:
        """
        Manuelle Freigabe nach Drawdown oder Datenfeed-Pause.
        Muss explizit durch den Operator aufgerufen werden.
        """
        self._trading_paused = False
        self._audit("TRADING_RESUMED", "Manuell durch Operator freigegeben")
        logger.info("EmergencyHandler: Trading manuell freigegeben.")

    # ── Fehler-Reaktion 1: MT5 nicht erreichbar ───────────────────────────────

    def handle_mt5_unreachable(self, symbols: list[str] | None = None) -> dict[str, Any]:
        """
        Reaktion: MT5 nach 3 Retries nicht erreichbar.

        Offene Positionen werden abgerufen, dann wird ein OANDA-Fallback
        ueber den DataRouter (mit integriertem PriceValidator) versucht.
        Bei Preisdiskrepanz oder fehlendem OANDA wird der Handel pausiert.

        Parameters
        ----------
        symbols : Symbole fuer den Preisabgleich (erstes Symbol wird genutzt).

        Returns
        -------
        dict mit open_positions, oanda_fallback (bool), trading_paused (bool).
        """
        self._audit("MT5_UNREACHABLE", "MT5 nach 3 Retries nicht erreichbar")
        logger.error("EMERGENCY: MT5 nicht erreichbar – pruefe Positionen + OANDA-Fallback.")

        open_positions = self._safe_get_positions()
        symbol = (symbols or [None])[0]

        oanda_fallback = False
        try:
            self._data_router.get_connector(symbol=symbol)
            oanda_fallback = True
            self._audit(
                "OANDA_FALLBACK_ACTIVATED",
                f"Preisabgleich OK | offene Positionen: {len(open_positions)} | symbol={symbol}",
            )
            logger.info(
                "EmergencyHandler: OANDA-Fallback aktiv | {n} offene Positionen.",
                n=len(open_positions),
            )
        except (PriceDiscrepancyError, EmergencyModeError) as exc:
            self._trading_paused = True
            self._audit(
                "TRADING_PAUSED",
                f"MT5 nicht erreichbar + kein sicherer OANDA-Fallback | {exc}",
            )
            logger.error(
                "EmergencyHandler: Kein sicherer Fallback -> Trading pausiert | {exc}",
                exc=exc,
            )

        return {
            "open_positions":  len(open_positions),
            "oanda_fallback":  oanda_fallback,
            "trading_paused":  self._trading_paused,
        }

    # ── Fehler-Reaktion 2: Fehlerhafter Datenfeed ────────────────────────────

    def handle_bad_datafeed(self, symbol: str = "", reason: str = "") -> None:
        """
        Reaktion: Datenfeed liefert leere oder fehlerhafte Daten.

        Trading wird sofort pausiert – kein Blind-Trading mit unbekannten Preisen.

        Parameters
        ----------
        symbol : Betroffenes Symbol (optional, fuer Logging).
        reason : Beschreibung des Fehlers.
        """
        detail = f"symbol={symbol} | {reason}".strip(" |")
        self._audit("BAD_DATAFEED", f"Fehlerhafter Datenfeed | {detail}")
        self._trading_paused = True
        logger.error(
            "EMERGENCY: Fehlerhafter Datenfeed -> Trading pausiert | {detail}",
            detail=detail,
        )

    # ── Fehler-Reaktion 3: Kritischer Drawdown ────────────────────────────────

    def handle_critical_drawdown(self) -> dict[str, Any]:
        """
        Reaktion: Kritischer Drawdown erreicht (RiskGuard.is_max_drawdown_hit()).

        Alle offenen Positionen werden ueber OrderExecutor.close_position()
        geschlossen. Trading wird pausiert bis zur manuellen Freigabe
        via resume_trading().

        Returns
        -------
        dict mit closed_tickets (list), errors (list), trading_paused (True).
        """
        drawdown_status = self._risk_guard.is_max_drawdown_hit()
        self._audit(
            "CRITICAL_DRAWDOWN",
            f"Maximaler Drawdown erreicht (RiskGuard={drawdown_status}) – schliesse alle Positionen",
        )
        logger.error(
            "EMERGENCY: Kritischer Drawdown (is_max_drawdown_hit={s}) – "
            "alle Positionen werden geschlossen.",
            s=drawdown_status,
        )

        closed_tickets, errors = self._close_all_positions()
        self._trading_paused = True

        self._audit(
            "ALL_POSITIONS_CLOSED",
            f"Geschlossen: {closed_tickets} | Fehler: {errors} | Handel pausiert",
        )
        logger.error(
            "EmergencyHandler: Drawdown-Abschluss | geschlossen={n} Fehler={e} | "
            "Handel pausiert bis manuelle Freigabe.",
            n=len(closed_tickets),
            e=len(errors),
        )
        return {
            "closed_tickets": closed_tickets,
            "errors":         errors,
            "trading_paused": True,
        }

    # ── Fehler-Reaktion 4: Unbehandelte Exception ─────────────────────────────

    def handle_unhandled_exception(self, exc: Exception) -> None:
        """
        Reaktion: Unbehandelte Exception im Hauptloop.

        Alle offenen Positionen werden geschlossen, ein Alert wird ueber
        AlertSender (z.B. Telegram) gesendet, dann wird der Prozess beendet.

        Parameters
        ----------
        exc : Die unbehandelte Exception.
        """
        exc_info = f"{type(exc).__name__}: {exc}"
        self._audit("UNHANDLED_EXCEPTION", exc_info)
        logger.critical(
            "EMERGENCY: Unbehandelte Exception -> Notfall-Shutdown | {exc}",
            exc=exc_info,
        )

        closed_tickets, errors = self._close_all_positions()

        alert_msg = (
            f"QuantzAI NOTFALL-SHUTDOWN\n"
            f"Grund: {exc_info}\n"
            f"Positionen geschlossen: {closed_tickets}\n"
            f"Fehler beim Schliessen: {errors}"
        )
        self._alert.send_alert(alert_msg)

        self._audit(
            "PROCESS_EXIT",
            f"Prozess-Shutdown nach unbehandelter Exception | {exc_info}",
        )
        logger.critical("EmergencyHandler: Prozess wird beendet (exit 1).")
        self._exit_fn(1)

    # ── Hilfsmethoden ─────────────────────────────────────────────────────────

    def _safe_get_positions(self) -> list[dict]:
        """Gibt offene Positionen zurueck; bei Fehler leere Liste."""
        try:
            return self._executor.get_open_positions()
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "EmergencyHandler: get_open_positions() fehlgeschlagen | {exc}", exc=exc
            )
            return []

    def _close_all_positions(self) -> tuple[list[int], list[str]]:
        """
        Schliesst alle offenen Positionen.

        Returns
        -------
        (closed_tickets, error_messages)
        """
        positions = self._safe_get_positions()
        closed: list[int] = []
        errors: list[str] = []

        for pos in positions:
            ticket = pos.get("ticket")
            try:
                self._executor.close_position(ticket)
                closed.append(ticket)
                logger.info(
                    "EmergencyHandler: Position {t} erfolgreich geschlossen.", t=ticket
                )
            except (OrderError, Exception) as exc:  # noqa: BLE001
                msg = f"ticket={ticket}: {exc}"
                errors.append(msg)
                logger.error(
                    "EmergencyHandler: Position {t} konnte nicht geschlossen werden | {e}",
                    t=ticket,
                    e=exc,
                )

        return closed, errors

    def _audit(self, event_type: str, reason: str) -> None:
        """
        Schreibt einen strukturierten Eintrag ins Audit-Log.

        Format: AUDIT | ts=<ISO> | event=<TYPE> | reason=<text>
        In Issue #19 wird dies durch einen echten Audit-DB-Schreiber ersetzt.
        """
        ts = datetime.now(timezone.utc).isoformat()
        entry = f"AUDIT | ts={ts} | event={event_type} | reason={reason}"
        logger.info(entry)

        if self._audit_path is not None:
            try:
                self._audit_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self._audit_path, "a", encoding="utf-8") as f:
                    f.write(entry + "\n")
            except OSError as exc:
                logger.error("Audit-Log schreiben fehlgeschlagen: {exc}", exc=exc)
