"""
src/journal/report_scheduler.py
ReportScheduler – automatischer Daily/Weekly Performance-Digest.

Laeuft als Hintergrund-Thread und erzeugt:
  - Daily-Report: jeden Tag zur konfigurierbaren Lokalzeit (Standard: 23:00)
  - Weekly-Report: jeden Sonntag zur konfigurierbaren Lokalzeit

Verwendet TradeJournal.generate_report() als Basis und ergaenzt:
  - Sharpe Ratio (aus PnL-Sequenz)
  - Max. Drawdown (aus kumulativer Equity-Kurve)
  - Anzahl offener Positionen

Ausgabe:
  - Markdown-Datei in reports/<period>_<YYYYMMDD_HHMM>.md
  - Versand via AlertSender (z. B. TelegramAlertSender)

Bei keinen abgeschlossenen Trades: kurze Statusmeldung statt leerem Report.

Testbarkeit:
  _now_fn : Injectable fuer Lokalzeit (Standard: datetime.now).
             Steuert Scheduling-Entscheidungen; Perioden-Berechnung
             erfolgt immer in UTC (konsistent mit generate_report()).
"""

from __future__ import annotations

import math
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional, Protocol, Union, runtime_checkable

from loguru import logger

from src.journal.trade_journal import TradeJournal


# ─────────────────────────────────────────────────────────────────────────────
#  Alert-Interface
# ─────────────────────────────────────────────────────────────────────────────

@runtime_checkable
class AlertSender(Protocol):
    """Interface fuer Report-Versand (Telegram, E-Mail, Log, ...)."""
    def send_alert(self, message: str) -> None: ...


class LogOnlyAlertSender:
    """Fallback: loggt den Report als INFO, sendet nichts extern."""
    def send_alert(self, message: str) -> None:
        logger.info("REPORT (kein Sender konfiguriert):\n{msg}", msg=message)


# ─────────────────────────────────────────────────────────────────────────────
#  Hilfsfunktionen (rein, testbar ohne TradeJournal)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_time(time_str: str) -> tuple[int, int]:
    """Parst 'HH:MM' -> (hour, minute). Wirft ValueError bei ungueltigem Format."""
    parts = time_str.split(":")
    if len(parts) != 2:
        raise ValueError(
            f"Ungültiges Zeitformat: {time_str!r}. Erwartet 'HH:MM'."
        )
    try:
        h, m = int(parts[0]), int(parts[1])
    except ValueError:
        raise ValueError(
            f"Ungültiges Zeitformat: {time_str!r}. Stunden/Minuten müssen Zahlen sein."
        )
    if not (0 <= h <= 23 and 0 <= m <= 59):
        raise ValueError(
            f"Ungültige Uhrzeit: {time_str!r}. Stunden 0–23, Minuten 0–59."
        )
    return h, m


def _calc_sharpe(pnls: list[float]) -> Optional[float]:
    """
    Berechnet die einfache Sharpe Ratio einer PnL-Sequenz.

    Verwendet Stichproben-Standardabweichung (n-1).
    Gibt None zurueck wenn < 2 Werte oder Std == 0.
    """
    n = len(pnls)
    if n < 2:
        return None
    mean = sum(pnls) / n
    variance = sum((p - mean) ** 2 for p in pnls) / (n - 1)
    std = math.sqrt(variance)
    if std == 0.0:
        return None
    return mean / std


def _calc_max_drawdown(pnls: list[float]) -> float:
    """
    Berechnet den maximalen Peak-to-Trough Drawdown aus einer PnL-Sequenz.

    Returns 0.0 fuer leere oder aufwaerts-monotone Sequenzen.
    """
    if not pnls:
        return 0.0
    equity  = 0.0
    peak    = 0.0
    max_dd  = 0.0
    for p in pnls:
        equity += p
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd
    return max_dd


# ─────────────────────────────────────────────────────────────────────────────
#  ReportScheduler
# ─────────────────────────────────────────────────────────────────────────────

class ReportScheduler:
    """
    Erzeugt und versendet taeglich/woechentlich Performance-Reports.

    Laeuft als Daemon-Thread (start()/stop()). Gibt es nichts zu berichten,
    wird eine kurze Statusmeldung erzeugt statt eines leeren Reports.

    Signale / Callbacks:
      on_report_generated : Optional[Callable[[str, str], None]]
        Wird nach jedem erzeugten Report aufgerufen: (period, content).

    Parameters
    ----------
    journal             : TradeJournal-Instanz.
    reports_dir         : Verzeichnis fuer Markdown-Dateien (Standard: 'reports').
    daily_time          : Lokalzeit fuer Daily-Report, Format 'HH:MM' (Standard: '23:00').
    weekly_time         : Lokalzeit fuer Weekly-Report an Sonntagen (Standard: '23:00').
    alert_sender        : AlertSender fuer Telegram/Log (Standard: LogOnlyAlertSender).
    check_interval_s    : Polling-Intervall in Sekunden (Standard: 60).
    on_report_generated : Optionaler Callback nach jedem Report.
    _now_fn             : Injectable fuer Lokalzeit (fuer Tests).
    """

    def __init__(
        self,
        journal:             TradeJournal,
        reports_dir:         Union[str, Path]                        = "reports",
        daily_time:          str                                      = "23:00",
        weekly_time:         str                                      = "23:00",
        alert_sender:        Optional[AlertSender]                   = None,
        check_interval_s:    int                                      = 60,
        on_report_generated: Optional[Callable[[str, str], None]]    = None,
        _now_fn:             Optional[Callable[[], datetime]]        = None,
    ) -> None:
        self._journal         = journal
        self._reports_dir     = Path(reports_dir)
        self._alert           = alert_sender or LogOnlyAlertSender()
        self._check_interval  = check_interval_s
        self._on_report       = on_report_generated
        self._now_fn          = _now_fn or datetime.now

        self._daily_h, self._daily_m   = _parse_time(daily_time)
        self._weekly_h, self._weekly_m = _parse_time(weekly_time)

        self._last_daily:  Optional[object] = None  # date object
        self._last_weekly: Optional[object] = None  # date object

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ─── Oeffentliche API ─────────────────────────────────────────────────────

    def start(self) -> None:
        """Startet den Scheduler-Thread (idempotent)."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="ReportScheduler",
        )
        self._thread.start()
        logger.info(
            "ReportScheduler: gestartet | daily={dh}:{dm:02d} weekly={wh}:{wm:02d}",
            dh=self._daily_h, dm=self._daily_m,
            wh=self._weekly_h, wm=self._weekly_m,
        )

    def stop(self, timeout: float = 3.0) -> None:
        """Beendet den Scheduler-Thread sauber (wartet max. timeout Sekunden)."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    @property
    def is_running(self) -> bool:
        """True wenn der Hintergrund-Thread laeuft."""
        return self._thread is not None and self._thread.is_alive()

    def trigger_daily(self) -> str:
        """Erzeugt und versendet sofort einen Daily-Report. Gibt Report-Text zurueck."""
        return self._trigger_report("daily")

    def trigger_weekly(self) -> str:
        """Erzeugt und versendet sofort einen Weekly-Report. Gibt Report-Text zurueck."""
        return self._trigger_report("weekly")

    # ─── Scheduling-Logik ─────────────────────────────────────────────────────

    def _should_run_daily(self, now: datetime) -> bool:
        today = now.date()
        if self._last_daily == today:
            return False
        return (now.hour, now.minute) >= (self._daily_h, self._daily_m)

    def _should_run_weekly(self, now: datetime) -> bool:
        if now.weekday() != 6:          # 0=Mo ... 6=So
            return False
        today = now.date()
        if self._last_weekly == today:
            return False
        return (now.hour, now.minute) >= (self._weekly_h, self._weekly_m)

    # ─── Report-Erstellung ────────────────────────────────────────────────────

    def _trigger_report(self, period: str) -> str:
        local_now = self._now_fn()
        content   = self._build_report(period)
        path      = self._save_report(content, period, local_now)
        self._send_report(content)
        if self._on_report is not None:
            try:
                self._on_report(period, content)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "ReportScheduler: on_report_generated-Callback fehlgeschlagen: {e}", e=exc
                )
        logger.info(
            "ReportScheduler: {p}-Report erstellt | {path}",
            p=period, path=path,
        )
        return content

    def _build_report(self, period: str) -> str:
        """
        Erstellt den vollstaendigen Report-Text.

        Basis:  TradeJournal.generate_report(period)
        Zusatz: Sharpe Ratio, Max. Drawdown, offene Positionen.
        Leer:   Kurze Statusmeldung wenn keine Trades.
        """
        utc_now = datetime.now(timezone.utc)

        if period.lower() == "daily":
            start = datetime(
                utc_now.year, utc_now.month, utc_now.day,
                0, 0, 0, tzinfo=timezone.utc,
            )
            period_lbl = "Täglich"
        elif period.lower() == "weekly":
            start      = utc_now - timedelta(days=7)
            period_lbl = "Wöchentlich"
        else:
            raise ValueError(f"period muss 'daily' oder 'weekly' sein, nicht {period!r}")

        stats = self._journal.calculate_stats(start, utc_now)

        if stats["n_trades"] == 0:
            return (
                f"# QuantzAI Trade-Report ({period_lbl})\n\n"
                f"*{start.strftime('%Y-%m-%d')} – {utc_now.strftime('%Y-%m-%d')} UTC*\n\n"
                "Keine abgeschlossenen Trades in diesem Zeitraum.\n"
            )

        # Basis-Report via TradeJournal.generate_report()
        base = self._journal.generate_report(period)

        # Erweiterte Kennzahlen
        pnls   = self._journal.get_pnl_sequence(start, utc_now)
        sharpe = _calc_sharpe(pnls)
        max_dd = _calc_max_drawdown(pnls)

        sharpe_str = f"{sharpe:.3f}" if sharpe is not None else "n/a"
        extended = (
            f"\n## Erweiterte Kennzahlen\n\n"
            f"| Kennzahl              | Wert           |\n"
            f"|-----------------------|----------------|\n"
            f"| Sharpe Ratio          | {sharpe_str:<15}|\n"
            f"| Max. Drawdown         | {max_dd:.2f}           |\n"
        )

        # Offene Positionen
        open_pos  = self._journal.get_open_positions()
        open_sect = f"\n## Offene Positionen: {len(open_pos)}\n"
        if open_pos:
            open_sect += (
                "\n| Symbol | Richtung | Lot   |\n"
                "|--------|----------|-------|\n"
            )
            for pos in open_pos[:10]:
                sym  = pos.get("symbol")    or "?"
                dire = pos.get("direction") or "?"
                lot  = pos.get("lot_size")
                lot_str = f"{lot:.2f}" if lot is not None else "?"
                open_sect += f"| {sym:<6} | {dire:<8} | {lot_str:<5} |\n"

        return base + extended + open_sect

    def _save_report(self, content: str, period: str, now: datetime) -> Path:
        """Schreibt Report als Markdown-Datei und gibt den Pfad zurueck."""
        self._reports_dir.mkdir(parents=True, exist_ok=True)
        filename = f"report_{period}_{now.strftime('%Y%m%d_%H%M')}.md"
        path     = self._reports_dir / filename
        path.write_text(content, encoding="utf-8")
        return path

    def _send_report(self, content: str) -> None:
        try:
            self._alert.send_alert(content)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "ReportScheduler: Alert-Versand fehlgeschlagen: {e}", e=exc
            )

    # ─── Hintergrund-Thread ───────────────────────────────────────────────────

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                now = self._now_fn()
                if self._should_run_daily(now):
                    self._last_daily = now.date()
                    self._trigger_report("daily")
                if self._should_run_weekly(now):
                    self._last_weekly = now.date()
                    self._trigger_report("weekly")
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "ReportScheduler: unbehandelte Exception im Loop: {e}", e=exc
                )
            self._stop_event.wait(timeout=self._check_interval)
