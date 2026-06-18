"""
scripts/watchdog.py
Externer Watchdog – ueberwacht den QuantzAI-Hauptprozess.

Verhalten:
  - Startet den angegebenen Prozess und wartet auf sein Ende.
  - Bei einem Crash (Exit-Code != 0) wird der Prozess neu gestartet.
  - Maximale Neustarts: 3 innerhalb von 1 Stunde (konfigurierbar).
  - Nach Erreichen des Limits: AlertSender-Benachrichtigung + Stopp.

Dasselbe AlertSender-Protocol wie in src/execution/emergency.py.

Verwendung:
  python scripts/watchdog.py python src/main.py --symbol EURUSD
  python scripts/watchdog.py --max-restarts 2 --window 1800 python src/main.py
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone, timedelta
from typing import Protocol, runtime_checkable

from loguru import logger


# ─────────────────────────────────────────────────────────────────────────────
#  Alert-Interface
# ─────────────────────────────────────────────────────────────────────────────

@runtime_checkable
class AlertSender(Protocol):
    """
    Interface fuer Watchdog-Benachrichtigungen (identisch zu emergency.py).
    Kann spaeter mit Telegram, E-Mail o.ae. implementiert werden.
    """

    def send_alert(self, message: str) -> None: ...


class LogOnlyAlertSender:
    """Fallback: loggt nur, sendet nichts extern."""

    def send_alert(self, message: str) -> None:
        logger.warning("WATCHDOG ALERT (kein Sender konfiguriert): {msg}", msg=message)


# ─────────────────────────────────────────────────────────────────────────────
#  Watchdog
# ─────────────────────────────────────────────────────────────────────────────

class Watchdog:
    """
    Ueberwacht einen Prozess und startet ihn bei Crash neu.

    Parameters
    ----------
    command                 : Kommando-Liste fuer subprocess.Popen.
    alert_sender            : AlertSender-Implementierung (Telegram etc.).
                              Faellt auf LogOnlyAlertSender zurueck wenn None.
    max_restarts            : Maximale Neustarts im Zeitfenster (Standard: 3).
    restart_window_seconds  : Laenge des Zeitfensters in Sekunden (Standard: 3600).
    """

    def __init__(
        self,
        command: list[str],
        alert_sender: AlertSender | None = None,
        max_restarts: int = 3,
        restart_window_seconds: int = 3600,
    ) -> None:
        self._command  = command
        self._alert    = alert_sender or LogOnlyAlertSender()
        self._max_restarts = max_restarts
        self._window   = restart_window_seconds
        self._restart_times: list[datetime] = []

    # ── Oeffentliche Schnittstelle ────────────────────────────────────────────

    def run(self) -> int:
        """
        Startet den Prozess und ueberwacht ihn.

        Gibt 0 zurueck wenn der Prozess normal beendet wurde (Exit 0).
        Gibt 1 zurueck wenn das Restart-Limit erreicht wurde.
        """
        logger.info("Watchdog: Starte Prozess | {cmd}", cmd=" ".join(self._command))
        process = self._start_process()

        while True:
            exit_code = process.wait()

            if exit_code == 0:
                logger.info("Watchdog: Prozess normal beendet (Exit 0).")
                return 0

            logger.warning(
                "Watchdog: Prozess mit Exit-Code {code} beendet.", code=exit_code
            )

            if not self._can_restart():
                window_min = self._window // 60
                msg = (
                    f"QuantzAI Bot nach {self._max_restarts} Crashes "
                    f"in {window_min} Minuten gestoppt. "
                    f"Manuelle Pruefung erforderlich! | Prozess: {' '.join(self._command)}"
                )
                logger.error("Watchdog: Restart-Limit erreicht -> Stopp. | {msg}", msg=msg)
                self._alert.send_alert(msg)
                return 1

            self._restart_times.append(datetime.now(timezone.utc))
            n = len(self._restart_times)
            logger.warning(
                "Watchdog: Neustart {n}/{max} ...",
                n=n,
                max=self._max_restarts,
            )
            process = self._start_process()

    @property
    def restart_count(self) -> int:
        """Anzahl bisher durchgefuehrter Neustarts."""
        return len(self._restart_times)

    # ── Interna ───────────────────────────────────────────────────────────────

    def _start_process(self) -> subprocess.Popen:
        proc = subprocess.Popen(self._command)
        logger.info("Watchdog: Prozess gestartet | PID={pid}", pid=proc.pid)
        return proc

    def _can_restart(self) -> bool:
        """True wenn im aktuellen Zeitfenster noch Neustarts erlaubt sind."""
        now = datetime.now(timezone.utc)
        window_start = now - timedelta(seconds=self._window)
        recent = [t for t in self._restart_times if t > window_start]
        return len(recent) < self._max_restarts


# ─────────────────────────────────────────────────────────────────────────────
#  CLI-Einstiegspunkt
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="QuantzAI Watchdog – ueberwacht und startet den Hauptprozess neu."
    )
    parser.add_argument(
        "command",
        nargs="+",
        help="Zu ueberwachender Prozess inkl. Argumente (z.B. 'python src/main.py')",
    )
    parser.add_argument(
        "--max-restarts",
        type=int,
        default=3,
        help="Maximale Neustarts im Zeitfenster (Standard: 3)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=3600,
        dest="window_seconds",
        help="Zeitfenster in Sekunden (Standard: 3600 = 1 Stunde)",
    )

    args = parser.parse_args()

    watchdog = Watchdog(
        command=args.command,
        max_restarts=args.max_restarts,
        restart_window_seconds=args.window_seconds,
    )
    return watchdog.run()


if __name__ == "__main__":
    sys.exit(main())
