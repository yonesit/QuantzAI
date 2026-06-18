"""
src/risk/anomaly_detector.py
AnomalyDetector – Rogue-Bot-Schutz durch Erkennung anomalen Handelsverhaltens.

Erkannte Anomalien (4 Typen):
  1. Hohe Trade-Frequenz      – zu viele Trades im konfigurierten Zeitfenster
  2. Grosse Positionsgroesse  – letzte Lot-Groesse >> Vielfaches des Durchschnitts
  3. Wiederholte Ablehnungen  – N aufeinanderfolgende abgelehnte Orders
  4. Signal-Flackern          – Signal wechselt >= M-mal innerhalb einer Minute

Integration mit EmergencyHandler:
  Bei erkannter Anomalie wird emergency_handler.handle_bad_datafeed() aufgerufen,
  welche den Trading-Stop-Mechanismus aktiviert (is_trading_paused = True).

Konfiguration in config.yaml unter 'anomaly_detection'.
Alle Schwellwerte koennen auch direkt per Konstruktor-Parameter gesetzt werden.

Order-Dict Felder (alle optional – fehlende Felder werden uebersprungen):
  timestamp  : datetime | float (Unix) | str (ISO 8601) – Zeitstempel der Order
  lot_size   : float – Positionsgroesse in Lots
  status     : str   – z.B. 'rejected', 'error', 'failed', 'filled'
  error_code : str   – Fehlertyp-Bezeichner fuer Ablehnungs-Check
  signal     : str   – 'long', 'short', 'flat' – Modell-Signal
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional

from loguru import logger


class AnomalyDetector:
    """
    Erkennt anomales Handelsverhalten und loest bei Bedarf den EmergencyHandler aus.

    Parameters
    ----------
    emergency_handler             : Optionaler EmergencyHandler; wird bei erkannter
                                    Anomalie via handle_bad_datafeed() ausgeloest.
    max_trades_per_window         : Max. Anzahl Trades im Zeitfenster (Standard: 10).
    frequency_window_seconds      : Dauer des Zeitfensters in Sekunden (Standard: 300).
    lot_size_multiplier           : Faktor fuer Positionsgroessen-Anomalie (Standard: 3.0).
    max_consecutive_rejections    : Max. aufeinanderfolgende Ablehnungen (Standard: 5).
    max_signal_changes_per_minute : Max. Signalwechsel pro Minute (Standard: 3).
    _now_fn                       : Injizierbar fuer Tests statt datetime.now(utc).
    """

    def __init__(
        self,
        emergency_handler=None,
        max_trades_per_window: int = 10,
        frequency_window_seconds: int = 300,
        lot_size_multiplier: float = 3.0,
        max_consecutive_rejections: int = 5,
        max_signal_changes_per_minute: int = 3,
        _now_fn: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self._emergency          = emergency_handler
        self._max_trades         = max_trades_per_window
        self._freq_window        = frequency_window_seconds
        self._lot_multiplier     = lot_size_multiplier
        self._max_rejections     = max_consecutive_rejections
        self._max_signal_changes = max_signal_changes_per_minute
        self._now                = _now_fn or (lambda: datetime.now(timezone.utc))

    # ── Oeffentliche Schnittstelle ────────────────────────────────────────────

    def check(self, recent_orders: list[dict]) -> tuple[bool, str]:
        """
        Prueft alle 4 Anomalie-Kriterien gegen die uebergebene Order-Liste.

        Alle 4 Pruefungen laufen immer durch; mehrere gleichzeitige Anomalien
        werden kombiniert und als eine Nachricht zurueckgegeben.

        Parameters
        ----------
        recent_orders : Liste von Order-Dicts (z.B. aus AuditLog.get_recent_orders()).

        Returns
        -------
        (True,  "Begruendung | weiterer Grund")  wenn >= 1 Anomalie erkannt.
        (False, "")                               wenn alles normal.
        """
        checks = [
            self._check_trade_frequency,
            self._check_lot_size,
            self._check_consecutive_rejections,
            self._check_signal_flickering,
        ]
        reasons: list[str] = []
        for fn in checks:
            anomaly, reason = fn(recent_orders)
            if anomaly:
                reasons.append(reason)

        if reasons:
            combined = " | ".join(reasons)
            logger.error(
                "AnomalyDetector: {n} Anomalie(n) erkannt -> Emergency | {r}",
                n=len(reasons), r=combined,
            )
            self._trigger_emergency(combined)
            return True, combined

        logger.debug("AnomalyDetector: Keine Anomalie in {n} Orders.", n=len(recent_orders))
        return False, ""

    # ── 4 Anomalie-Pruefungen ─────────────────────────────────────────────────

    def _check_trade_frequency(self, orders: list[dict]) -> tuple[bool, str]:
        """Anomalie wenn zu viele Trades im konfigurierten Zeitfenster."""
        if not orders:
            return False, ""

        cutoff = self._now() - timedelta(seconds=self._freq_window)
        count = 0
        for o in orders:
            ts = self._parse_timestamp(o.get("timestamp"))
            if ts is not None and ts >= cutoff:
                count += 1

        if count > self._max_trades:
            return True, (
                f"Hohe Trade-Frequenz: {count} Trades in {self._freq_window}s "
                f"(Max: {self._max_trades})"
            )
        return False, ""

    def _check_lot_size(self, orders: list[dict]) -> tuple[bool, str]:
        """Anomalie wenn letzte Lot-Groesse ein Vielfaches des bisherigen Durchschnitts."""
        lot_sizes = [
            float(o["lot_size"]) for o in orders
            if "lot_size" in o and float(o.get("lot_size", 0)) > 0
        ]
        if len(lot_sizes) < 2:
            return False, ""

        prior_mean = sum(lot_sizes[:-1]) / len(lot_sizes[:-1])
        last_lot   = lot_sizes[-1]

        if prior_mean > 0 and last_lot > prior_mean * self._lot_multiplier:
            ratio = last_lot / prior_mean
            return True, (
                f"Ungewoehnlich grosse Position: {last_lot:.2f} Lots "
                f"({ratio:.1f}x Durchschnitt {prior_mean:.2f})"
            )
        return False, ""

    def _check_consecutive_rejections(self, orders: list[dict]) -> tuple[bool, str]:
        """Anomalie wenn N aufeinanderfolgende Orders abgelehnt wurden."""
        if not orders:
            return False, ""

        count = 0
        for o in reversed(orders):
            if str(o.get("status", "")).lower() in ("rejected", "error", "failed"):
                count += 1
            else:
                break

        if count >= self._max_rejections:
            tail  = orders[-count:]
            codes = {o.get("error_code", "") for o in tail if o.get("error_code")}
            code_str = ", ".join(sorted(codes)) if codes else "unbekannt"
            return True, (
                f"Wiederholte Ablehnungen: {count} hintereinander "
                f"(Fehlercodes: {code_str})"
            )
        return False, ""

    def _check_signal_flickering(self, orders: list[dict]) -> tuple[bool, str]:
        """Anomalie wenn das Modell-Signal mehrfach pro Minute wechselt."""
        cutoff = self._now() - timedelta(seconds=60)
        recent: list[dict] = []
        for o in orders:
            if "signal" not in o:
                continue
            ts = self._parse_timestamp(o.get("timestamp"))
            if ts is not None and ts >= cutoff:
                recent.append(o)

        if len(recent) < 2:
            return False, ""

        changes = sum(
            1 for i in range(1, len(recent))
            if recent[i]["signal"] != recent[i - 1]["signal"]
        )
        if changes >= self._max_signal_changes:
            return True, (
                f"Signal-Flackern: {changes} Signalwechsel in letzter Minute "
                f"(Max: {self._max_signal_changes})"
            )
        return False, ""

    # ── Hilfsmethoden ─────────────────────────────────────────────────────────

    def _trigger_emergency(self, reason: str) -> None:
        """Loest EmergencyHandler aus wenn konfiguriert."""
        if self._emergency is not None:
            self._emergency.handle_bad_datafeed(
                reason=f"[AnomalyDetector] {reason}",
            )

    def _parse_timestamp(self, ts: Any) -> Optional[datetime]:
        """
        Konvertiert verschiedene Timestamp-Formate in timezone-aware datetime.

        Unterstuetzt: datetime, float/int (Unix-Timestamp), str (ISO 8601).
        Gibt None zurueck fuer unbekannte oder ungueltige Werte.
        """
        if ts is None:
            return None
        if isinstance(ts, datetime):
            return ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)
        if isinstance(ts, (int, float)):
            return datetime.fromtimestamp(float(ts), tz=timezone.utc)
        if isinstance(ts, str):
            try:
                dt = datetime.fromisoformat(ts)
                return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
            except ValueError:
                return None
        return None
