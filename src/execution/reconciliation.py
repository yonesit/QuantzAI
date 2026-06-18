"""
src/execution/reconciliation.py
PositionReconciler – Abgleich interner Positionsliste mit MT5-Realdaten.

Szenario:
  Nach einem Absturz, Verbindungsabbruch oder einer abgelehnten Order
  kann die interne Sicht des Systems von der tatsaechlichen MT5-Position
  abweichen. Der Reconciler erkennt solche Diskrepanzen und fuehrt den
  internen Zustand nach (MT5 ist immer die Wahrheitsquelle).

Erkannte Diskrepanzen:
  - Position lokal offen, bei MT5 nicht mehr vorhanden
    (SL/TP getroffen, manuell geschlossen, Broker-Intervention)
  - Position bei MT5 offen, lokal nicht bekannt
    (System nach Absturz neu gestartet, Order extern platziert)

Trigger:
  1. Automatisch nach jedem MT5-Reconnect (via Callback-Hook).
  2. Periodisch im Hintergrund (Standard: alle 5 Minuten).
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from src.data.mt5_connector import _load_mt5


class PositionReconciler:
    """
    Vergleicht die interne Positionsliste des OrderExecutors mit den
    tatsaechlichen MT5-Positionen und gleicht Abweichungen ab.

    Parameters
    ----------
    connector             : MT5Connector-Instanz.
    executor              : OrderExecutor-Instanz.
    sync_interval_seconds : Intervall fuer periodischen Hintergrund-Sync (Standard: 300 s).
    """

    def __init__(
        self,
        connector,
        executor,
        sync_interval_seconds: int = 300,
    ) -> None:
        self._connector = connector
        self._executor  = executor
        self._interval  = sync_interval_seconds
        self._timer: threading.Timer | None = None
        self._incidents: list[dict[str, Any]] = []

        # Hook in MT5Connector registrieren falls unterstuetzt
        if hasattr(connector, "register_reconnect_callback"):
            connector.register_reconnect_callback(self._on_reconnect)
            logger.debug("PositionReconciler als Reconnect-Hook registriert.")

    # ── Oeffentliche Schnittstelle ────────────────────────────────────────────

    def sync(self) -> dict[str, Any]:
        """
        Vergleicht interne Positionsliste mit MT5-Realdaten.

        MT5 ist die Wahrheitsquelle: Bei Diskrepanz wird der interne
        Zustand angepasst und ein Vorfall protokolliert.

        Returns
        -------
        dict mit:
          missing_locally  : list[int]  – Tickets die MT5 kennt, lokal nicht.
          missing_at_mt5   : list[int]  – Tickets die lokal offen sind, MT5 nicht.
          incidents        : int        – Gesamtzahl erkannter Diskrepanzen.
          in_sync          : bool       – True wenn beide Seiten uebereinstimmen.
          synced_at        : str        – ISO-Zeitstempel des Sync-Zeitpunkts.
        """
        logger.info("PositionReconciler.sync() gestartet.")

        local_positions  = {p["ticket"]: p for p in self._executor.get_open_positions()}
        mt5_positions    = self._fetch_mt5_positions()
        mt5_by_ticket    = {p["ticket"]: p for p in mt5_positions}

        missing_locally = [p for p in mt5_positions  if p["ticket"] not in local_positions]
        missing_at_mt5  = [p for p in local_positions.values() if p["ticket"] not in mt5_by_ticket]

        # ── Reconcile ─────────────────────────────────────────────────────────
        for pos in missing_locally:
            ticket = pos["ticket"]
            logger.warning(
                "Diskrepanz: Position {t} ({sym}) ist in MT5 offen, "
                "lokal nicht bekannt -> wird lokal registriert.",
                t=ticket, sym=pos.get("symbol", "?"),
            )
            self._executor.reconcile_add_position(ticket, pos)
            self._incidents.append({
                "type":   "missing_locally",
                "ticket": ticket,
                "ts":     _now_iso(),
            })

        for pos in missing_at_mt5:
            ticket = pos["ticket"]
            logger.warning(
                "Diskrepanz: Position {t} ({sym}) lokal als offen markiert, "
                "MT5 kennt sie nicht mehr -> wird lokal geschlossen.",
                t=ticket, sym=pos.get("symbol", "?"),
            )
            self._executor.reconcile_close_position(ticket)
            self._incidents.append({
                "type":   "missing_at_mt5",
                "ticket": ticket,
                "ts":     _now_iso(),
            })

        total_incidents = len(missing_locally) + len(missing_at_mt5)
        result = {
            "missing_locally": [p["ticket"] for p in missing_locally],
            "missing_at_mt5":  [p["ticket"] for p in missing_at_mt5],
            "incidents":       total_incidents,
            "in_sync":         total_incidents == 0,
            "synced_at":       _now_iso(),
        }
        logger.info(
            "PositionReconciler.sync() abgeschlossen | in_sync={ok} | "
            "fehlend_lokal={ml} fehlend_mt5={mm}",
            ok=result["in_sync"],
            ml=len(missing_locally),
            mm=len(missing_at_mt5),
        )
        return result

    def start_periodic_sync(self) -> None:
        """Startet den periodischen Hintergrund-Sync."""
        if self._timer is not None:
            logger.debug("PositionReconciler: periodischer Sync laeuft bereits.")
            return
        self._schedule_next()
        logger.info(
            "PositionReconciler: periodischer Sync gestartet (Intervall: {s}s).",
            s=self._interval,
        )

    def stop_periodic_sync(self) -> None:
        """Stoppt den periodischen Hintergrund-Sync."""
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
            logger.info("PositionReconciler: periodischer Sync gestoppt.")

    @property
    def incidents(self) -> list[dict[str, Any]]:
        """Alle bisher protokollierten Vorfaelle (unveraenderliche Kopie)."""
        return list(self._incidents)

    # ── Interna ───────────────────────────────────────────────────────────────

    def _on_reconnect(self) -> None:
        """Wird automatisch nach jedem MT5-Reconnect aufgerufen."""
        logger.info("PositionReconciler: automatischer Sync nach MT5-Reconnect.")
        try:
            self.sync()
        except Exception as exc:  # noqa: BLE001
            logger.error("PositionReconciler Reconnect-Sync Fehler: {exc}", exc=exc)

    def _periodic_tick(self) -> None:
        """Ein Tick des periodischen Syncs: sync() + naechsten Timer planen."""
        try:
            self.sync()
        except Exception as exc:  # noqa: BLE001
            logger.error("PositionReconciler periodischer Sync Fehler: {exc}", exc=exc)
        self._schedule_next()

    def _schedule_next(self) -> None:
        self._timer = threading.Timer(self._interval, self._periodic_tick)
        self._timer.daemon = True
        self._timer.start()

    def _fetch_mt5_positions(self) -> list[dict[str, Any]]:
        """Holt offene Positionen direkt aus MT5."""
        mt5 = _load_mt5()
        raw = mt5.positions_get() or []
        result = []
        for p in raw:
            result.append({
                "ticket":     int(p.ticket),
                "symbol":     str(p.symbol),
                "direction":  "buy" if p.type == mt5.ORDER_TYPE_BUY else "sell",
                "lot_size":   float(p.volume),
                "sl_price":   float(p.sl),
                "tp_price":   float(p.tp),
                "open_price": float(p.price_open),
                "status":     "open",
            })
        return result


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
