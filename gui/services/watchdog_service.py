"""
gui/services/watchdog_service.py
WatchdogService – ueberwacht den TradingOrchestrator-Thread in der GUI.

Erkennt Crashes (BotControlsWidget.error_occurred) und startet den Bot
automatisch neu – bis zu max_restarts Mal innerhalb eines Zeitfensters.

Zeitfenster-Logik wiederverwendet aus scripts/watchdog.py (Watchdog._can_restart).

Thread-Sicherheit:
  error_occurred wird via QueuedConnection im Hauptthread empfangen,
  sodass _on_crash() immer im Qt-Hauptthread laeuft.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional, Protocol, runtime_checkable

from loguru import logger

from PySide6.QtCore import QObject, QTimer, Signal, Slot

from gui.widgets.bot_controls_widget import BotControlsWidget, BotState


# ─────────────────────────────────────────────────────────────────────────────
#  Alert-Interface (identisch zu scripts/watchdog.py)
# ─────────────────────────────────────────────────────────────────────────────

@runtime_checkable
class AlertSender(Protocol):
    """Interface fuer Watchdog-Benachrichtigungen (Telegram, E-Mail, ...)."""
    def send_alert(self, message: str) -> None: ...


class LogOnlyAlertSender:
    """Fallback: loggt nur, sendet nichts extern."""
    def send_alert(self, message: str) -> None:
        logger.warning(
            "WATCHDOG ALERT (kein Sender konfiguriert): {msg}", msg=message
        )


# ─────────────────────────────────────────────────────────────────────────────
#  WatchdogService
# ─────────────────────────────────────────────────────────────────────────────

STATUS_RUNNING       = "running"
STATUS_DISABLED      = "disabled"
STATUS_LIMIT_REACHED = "limit_reached"


class WatchdogService(QObject):
    """
    GUI-Watchdog: ueberwacht den BotControlsWidget und startet den Bot
    bei einem Crash automatisch neu.

    Logik identisch zu scripts/watchdog.py:
      - Max. max_restarts Neustarts innerhalb von window_seconds Sekunden.
      - Nach Erreichen des Limits: Alert + Stopp (kein weiterer Neustart).

    Signale
    -------
    status_changed(str)     – "running" | "disabled" | "limit_reached"
    restart_triggered(int)  – Restart-Nummer (1, 2, ...)

    Parameters
    ----------
    bot_controls        : BotControlsWidget – Crash-Signal wird hier abonniert.
    alert_sender        : AlertSender fuer Telegram-/Log-Benachrichtigungen.
    max_restarts        : Maximale Neustarts im Zeitfenster (Standard: 3).
    window_seconds      : Zeitfenster in Sekunden (Standard: 3600 = 1 Stunde).
    restart_delay_ms    : Wartezeit vor Neustart in ms (Standard: 2000).
                          Auf 0 setzen fuer Tests.
    """

    status_changed    = Signal(str)   # STATUS_*-Konstante
    restart_triggered = Signal(int)   # Restart-Nummer

    def __init__(
        self,
        bot_controls:     BotControlsWidget,
        alert_sender:     Optional[AlertSender] = None,
        max_restarts:     int = 3,
        window_seconds:   int = 3600,
        restart_delay_ms: int = 2000,
    ) -> None:
        super().__init__()
        self._bot_controls   = bot_controls
        self._alert          = alert_sender or LogOnlyAlertSender()
        self._max_restarts   = max_restarts
        self._window         = window_seconds
        self._restart_delay  = restart_delay_ms
        self._restart_times: list[datetime] = []
        self._enabled:       bool           = True

        bot_controls.error_occurred.connect(self._on_crash)

    # ─── Oeffentliche API ─────────────────────────────────────────────────────

    def enable(self) -> None:
        """Aktiviert den Watchdog (Standard-Zustand)."""
        if not self._enabled:
            self._enabled = True
            logger.info("WatchdogService: aktiviert")
            self.status_changed.emit(STATUS_RUNNING)

    def disable(self) -> None:
        """Deaktiviert den Watchdog – keine automatischen Neustarts."""
        if self._enabled:
            self._enabled = False
            logger.info("WatchdogService: deaktiviert")
            self.status_changed.emit(STATUS_DISABLED)

    def shutdown(self) -> None:
        """Trennt alle Qt-Verbindungen – muss im closeEvent aufgerufen werden."""
        import warnings
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                self._bot_controls.error_occurred.disconnect(self._on_crash)
        except RuntimeError:
            pass  # bereits getrennt

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    @property
    def restart_count(self) -> int:
        """Gesamtzahl bisher durchgefuehrter Neustarts."""
        return len(self._restart_times)

    @property
    def can_restart(self) -> bool:
        """True wenn im aktuellen Zeitfenster noch Neustarts erlaubt sind."""
        return self._can_restart()

    @property
    def status(self) -> str:
        """Aktueller Status-String (STATUS_*-Konstante)."""
        return STATUS_RUNNING if self._enabled else STATUS_DISABLED

    # ─── Interna ──────────────────────────────────────────────────────────────

    @Slot(str)
    def _on_crash(self, message: str) -> None:
        if not self._enabled:
            logger.debug(
                "WatchdogService: Crash ignoriert (deaktiviert): {m}", m=message
            )
            return

        if not self._can_restart():
            window_min = self._window // 60
            msg = (
                f"QuantzAI Bot nach {self._max_restarts} Crashes "
                f"in {window_min} Minuten gestoppt. "
                f"Manuelle Pruefung erforderlich!"
            )
            logger.error("WatchdogService: Restart-Limit erreicht -> Stopp.")
            self._alert.send_alert(msg)
            self.status_changed.emit(STATUS_LIMIT_REACHED)
            return

        self._restart_times.append(datetime.now(timezone.utc))
        n = len(self._restart_times)
        logger.warning(
            "WatchdogService: Auto-Restart {n}/{max} nach Crash: {err}",
            n=n, max=self._max_restarts, err=message,
        )
        self._alert.send_alert(f"QuantzAI Auto-Restart #{n}: {message}")
        self.restart_triggered.emit(n)
        QTimer.singleShot(self._restart_delay, self._do_restart)

    def _can_restart(self) -> bool:
        now          = datetime.now(timezone.utc)
        window_start = now - timedelta(seconds=self._window)
        recent       = [t for t in self._restart_times if t > window_start]
        return len(recent) < self._max_restarts

    @Slot()
    def _do_restart(self) -> None:
        if not self._enabled:
            return
        if self._bot_controls.bot_state == BotState.STOPPED:
            logger.info("WatchdogService: Starte Bot neu...")
            self._bot_controls.start()
