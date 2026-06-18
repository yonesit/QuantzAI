"""
src/monitoring/audit_log.py
AuditLog – zentrale SQLite-Datenbank fuer Orders, Fehler und Notfall-Eingriffe.

Tabellen:
  orders      : Alle Order-Events (Eroeffnung, Schliessung, Aenderung)
  errors      : Fehler und Exceptions
  emergencies : Notfall-Eingriffe des EmergencyHandlers und Reconcilers

Ersetzt/ergaenzt den reinen Loguru-Logger aus den frueheren Issues.
Thread-sicher durch Lock.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, date, timezone, timedelta
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from loguru import logger


DateLike = Union[datetime, date, str]


class AuditLog:
    """
    Zentrale SQLite-Datenbank fuer alle Audit-Eintraege.

    Parameters
    ----------
    db_path : Pfad zur SQLite-Datei (wird erstellt falls nicht vorhanden).
    """

    def __init__(self, db_path: Union[str, Path] = "data/processed/audit.db") -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._lock = threading.Lock()
        self._init_tables()
        logger.info("AuditLog: Datenbank geoeffnet | {path}", path=self._path)

    def close(self) -> None:
        """Schliesst die Datenbankverbindung."""
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "AuditLog":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    # ── Schreib-Methoden ──────────────────────────────────────────────────────

    def log_order(self, order_data: dict) -> None:
        """
        Speichert ein Order-Event in der orders-Tabelle.

        Parameters
        ----------
        order_data : dict mit optionalen Feldern symbol, direction, lot_size,
                     sl_price, tp_price, ticket, status.
                     Weitere Felder werden als extra_json gespeichert.
        """
        ts = _now_iso()
        known = {"symbol", "direction", "lot_size", "sl_price", "tp_price", "ticket", "status"}
        extra = {k: v for k, v in order_data.items() if k not in known}

        with self._lock:
            self._conn.execute(
                """
                INSERT INTO orders
                    (ts, symbol, direction, lot_size, sl_price, tp_price, ticket, status, extra_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts,
                    order_data.get("symbol"),
                    order_data.get("direction"),
                    order_data.get("lot_size"),
                    order_data.get("sl_price"),
                    order_data.get("tp_price"),
                    order_data.get("ticket"),
                    order_data.get("status"),
                    json.dumps(extra, default=str) if extra else None,
                ),
            )
            self._conn.commit()
        logger.debug("AuditLog: Order gespeichert | ticket={t}", t=order_data.get("ticket"))

    def log_error(self, error_type: str, details: dict) -> None:
        """
        Speichert einen Fehler oder eine Exception in der errors-Tabelle.

        Parameters
        ----------
        error_type : Kurzbezeichnung, z.B. 'MT5_CONNECTION_ERROR'.
        details    : Fehler-Details als Dictionary.
        """
        ts = _now_iso()
        with self._lock:
            self._conn.execute(
                "INSERT INTO errors (ts, error_type, details_json) VALUES (?, ?, ?)",
                (ts, error_type, json.dumps(details, default=str)),
            )
            self._conn.commit()
        logger.debug("AuditLog: Fehler gespeichert | type={t}", t=error_type)

    def log_emergency(self, event_type: str, details: dict) -> None:
        """
        Speichert einen Notfall-Eingriff in der emergencies-Tabelle.

        Parameters
        ----------
        event_type : Art des Notfalls, z.B. 'CRITICAL_DRAWDOWN', 'MT5_UNREACHABLE'.
        details    : Details als Dictionary, optionales Feld 'reason'.
        """
        ts = _now_iso()
        reason = details.get("reason", "")
        extra = {k: v for k, v in details.items() if k != "reason"}
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO emergencies (ts, event_type, reason, details_json)
                VALUES (?, ?, ?, ?)
                """,
                (
                    ts,
                    event_type,
                    reason,
                    json.dumps(extra, default=str) if extra else None,
                ),
            )
            self._conn.commit()
        logger.debug("AuditLog: Notfall gespeichert | event={e}", e=event_type)

    # ── Lese-Methoden ─────────────────────────────────────────────────────────

    def query_orders(
        self,
        start_date: DateLike,
        end_date: DateLike,
        symbol: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Liest Orders aus der Datenbank.

        Beide Grenzen sind inklusiv:
          - date-Objekt als start_date: Beginn des Tages (00:00:00 UTC).
          - date-Objekt als end_date:   Ende des Tages (23:59:59.999999 UTC).
          - datetime-Objekt:            wird direkt als Grenze verwendet.

        Parameters
        ----------
        start_date : Startdatum/-Zeit (inklusiv).
        end_date   : Enddatum/-Zeit (inklusiv).
        symbol     : Optional: nur Orders fuer dieses Symbol zurueckgeben.

        Returns
        -------
        pd.DataFrame mit allen passenden Orders, sortiert nach ts.
        """
        start_iso = _to_iso_start(start_date)
        end_iso   = _to_iso_end(end_date)

        if symbol is not None:
            sql    = (
                "SELECT * FROM orders "
                "WHERE ts >= ? AND ts <= ? AND symbol = ? "
                "ORDER BY ts ASC"
            )
            params: tuple = (start_iso, end_iso, symbol)
        else:
            sql    = "SELECT * FROM orders WHERE ts >= ? AND ts <= ? ORDER BY ts ASC"
            params = (start_iso, end_iso)

        with self._lock:
            cursor = self._conn.execute(sql, params)
            rows   = cursor.fetchall()
            cols   = [d[0] for d in cursor.description]

        return pd.DataFrame(rows, columns=cols)

    # ── Interna ───────────────────────────────────────────────────────────────

    def _init_tables(self) -> None:
        with self._lock:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS orders (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts          TEXT    NOT NULL,
                    symbol      TEXT,
                    direction   TEXT,
                    lot_size    REAL,
                    sl_price    REAL,
                    tp_price    REAL,
                    ticket      INTEGER,
                    status      TEXT,
                    extra_json  TEXT
                );

                CREATE TABLE IF NOT EXISTS errors (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts           TEXT NOT NULL,
                    error_type   TEXT NOT NULL,
                    details_json TEXT
                );

                CREATE TABLE IF NOT EXISTS emergencies (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts           TEXT NOT NULL,
                    event_type   TEXT NOT NULL,
                    reason       TEXT,
                    details_json TEXT
                );
            """)
            self._conn.commit()


# ─────────────────────────────────────────────────────────────────────────────
#  Hilfsfunktionen
# ─────────────────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_iso_start(d: DateLike) -> str:
    if isinstance(d, datetime):
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        return d.isoformat()
    if isinstance(d, date):
        return datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=timezone.utc).isoformat()
    return str(d)


def _to_iso_end(d: DateLike) -> str:
    """Konvertiert Enddatum: date -> Ende des Tages (23:59:59.999999 UTC)."""
    if isinstance(d, datetime):
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        return d.isoformat()
    if isinstance(d, date):
        return datetime(
            d.year, d.month, d.day, 23, 59, 59, 999999, tzinfo=timezone.utc
        ).isoformat()
    return str(d)
