"""
src/journal/trade_journal.py
TradeJournal – automatische, vollstaendige Erfassung jedes Trades.

Jeder Trade wird mit Entry/Exit, Symbol, Richtung, Lot-Groesse,
P&L, Marktregime, News-Kontext und Signal-Konfidenz gespeichert.

Datenbankdatei: data/processed/journal.db (eigene SQLite-Datei,
gleiche Infrastruktur wie AuditLog: sqlite3 + threading.Lock).

Schnittstelle:
  log_trade_open(trade_data)            -> int   (Trade-ID)
  log_trade_close(trade_id, exit_data)  -> None
  calculate_stats(start, end, symbol)   -> dict
  generate_report(period)               -> str   (Markdown)

Integration:
  TradeJournal wird als optionaler Parameter dem OrderExecutor
  uebergeben (kein bestehender Test bricht).

trade_data-Felder (alle optional ausser symbol und direction):
  symbol            str   – Handelssymbol
  direction         str   – 'buy' oder 'sell'
  lot_size          float – Lot-Groesse
  entry_price       float – Eroeffnungspreis
  entry_time        str | datetime – ISO-Zeitstempel (Default: jetzt)
  regime            str   – Marktregime ('TRENDING', 'RANGING', ...)
  news_context      str   – aktive News/Kalender-Eintraege
  signal_confidence float – Konfidenz des SignalModels (0.0 – 1.0)
  setup             str   – Setup-/Strategie-Bezeichner

exit_data-Felder:
  exit_price  float – Schlusskurs
  exit_time   str | datetime – ISO-Zeitstempel (Default: jetzt)
  pnl         float – realisierter Gewinn/Verlust (positiv = Gewinn)
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
from typing import Optional, Union

from loguru import logger


DateLike = Union[datetime, date, str]


class TradeJournal:
    """
    Persistentes Trade-Journal auf SQLite-Basis.

    Parameters
    ----------
    db_path : Pfad zur SQLite-Datei.
              Default: 'data/processed/journal.db'.
    """

    def __init__(
        self,
        db_path: Union[str, Path] = "data/processed/journal.db",
    ) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._lock = threading.Lock()
        self._init_tables()
        logger.info("TradeJournal: Datenbank geoeffnet | {path}", path=self._path)

    def close(self) -> None:
        """Schliesst die Datenbankverbindung."""
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "TradeJournal":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    # ── Schreib-Methoden ──────────────────────────────────────────────────────

    def log_trade_open(self, trade_data: dict) -> int:
        """
        Legt einen neuen Trade-Eintrag an (Status 'open').

        Parameters
        ----------
        trade_data : dict mit Trade-Kontextdaten (siehe Modul-Docstring).

        Returns
        -------
        int: Eindeutige Trade-ID (Primaerschluessel in der DB).
        """
        ts = _coerce_timestamp(trade_data.get("entry_time")) or _now_iso()
        known = {
            "symbol", "direction", "lot_size", "entry_price", "entry_time",
            "regime", "news_context", "signal_confidence", "setup",
        }
        extra = {k: v for k, v in trade_data.items() if k not in known}

        with self._lock:
            cursor = self._conn.execute(
                """
                INSERT INTO trades
                    (symbol, direction, lot_size, entry_price, entry_time,
                     regime, news_context, signal_confidence, setup, status, extra_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', ?)
                """,
                (
                    trade_data.get("symbol", ""),
                    trade_data.get("direction", ""),
                    trade_data.get("lot_size"),
                    trade_data.get("entry_price"),
                    ts,
                    trade_data.get("regime"),
                    trade_data.get("news_context"),
                    trade_data.get("signal_confidence"),
                    trade_data.get("setup"),
                    json.dumps(extra, default=str) if extra else None,
                ),
            )
            self._conn.commit()
            trade_id: int = cursor.lastrowid  # type: ignore[assignment]

        logger.debug(
            "TradeJournal: Trade geoeffnet | id={id} {sym} {dir}",
            id=trade_id, sym=trade_data.get("symbol"), dir=trade_data.get("direction"),
        )
        return trade_id

    def log_trade_close(self, trade_id: int, exit_data: dict) -> None:
        """
        Schliesst einen bestehenden Trade (Status 'closed') und speichert Exit-Daten.

        Parameters
        ----------
        trade_id  : Trade-ID aus log_trade_open().
        exit_data : dict mit exit_price, exit_time, pnl (alle optional).
        """
        exit_ts = _coerce_timestamp(exit_data.get("exit_time")) or _now_iso()

        with self._lock:
            self._conn.execute(
                """
                UPDATE trades
                SET exit_price = ?, exit_time = ?, pnl = ?, status = 'closed'
                WHERE id = ?
                """,
                (
                    exit_data.get("exit_price"),
                    exit_ts,
                    exit_data.get("pnl"),
                    trade_id,
                ),
            )
            self._conn.commit()

        logger.debug(
            "TradeJournal: Trade geschlossen | id={id} pnl={pnl}",
            id=trade_id, pnl=exit_data.get("pnl"),
        )

    # ── Lese- und Statistik-Methoden ─────────────────────────────────────────

    def get_trade(self, trade_id: int) -> Optional[dict]:
        """
        Liest einen einzelnen Trade aus der DB.

        Parameters
        ----------
        trade_id : Trade-ID aus log_trade_open().

        Returns
        -------
        dict mit allen Spalten, oder None wenn trade_id nicht gefunden.
        """
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM trades WHERE id = ?", (trade_id,)
            )
            row = cursor.fetchone()
            if row is None:
                return None
            cols = [d[0] for d in cursor.description]
            return dict(zip(cols, row))

    def calculate_stats(
        self,
        start_date: DateLike,
        end_date: DateLike,
        symbol: Optional[str] = None,
    ) -> dict:
        """
        Berechnet Performance-Kennzahlen fuer den angegebenen Zeitraum.

        Beruecksichtigt nur abgeschlossene Trades (status='closed') mit
        gesetztem PnL-Wert. Trades werden nach entry_time gefiltert.

        Parameters
        ----------
        start_date : Startdatum/-zeit (inklusiv).
        end_date   : Enddatum/-zeit (inklusiv).
        symbol     : Optional – nur Trades fuer dieses Symbol.

        Returns
        -------
        dict mit:
          n_trades      – Anzahl Trades
          win_rate      – Anteil Gewinn-Trades (0.0 – 1.0)
          profit_factor – Summe Gewinne / Summe Verluste (inf wenn kein Verlust)
          avg_win       – Durchschnitt der Gewinn-Trades
          avg_loss      – Betrag des Durchschnitts der Verlust-Trades
          total_pnl     – Summe aller PnL-Werte
          best_trade    – Hoechster Einzel-PnL (None wenn keine Trades)
          worst_trade   – Niedrigster Einzel-PnL (None wenn keine Trades)
        """
        start_iso = _to_iso_start(start_date)
        end_iso   = _to_iso_end(end_date)

        if symbol is not None:
            sql = (
                "SELECT pnl FROM trades "
                "WHERE status = 'closed' AND pnl IS NOT NULL "
                "AND entry_time >= ? AND entry_time <= ? AND symbol = ? "
                "ORDER BY entry_time ASC"
            )
            params: tuple = (start_iso, end_iso, symbol)
        else:
            sql = (
                "SELECT pnl FROM trades "
                "WHERE status = 'closed' AND pnl IS NOT NULL "
                "AND entry_time >= ? AND entry_time <= ? "
                "ORDER BY entry_time ASC"
            )
            params = (start_iso, end_iso)

        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()

        pnls = [float(row[0]) for row in rows]

        if not pnls:
            return {
                "n_trades":      0,
                "win_rate":      0.0,
                "profit_factor": 0.0,
                "avg_win":       0.0,
                "avg_loss":      0.0,
                "total_pnl":     0.0,
                "best_trade":    None,
                "worst_trade":   None,
            }

        wins   = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        n_trades   = len(pnls)
        win_rate   = len(wins) / n_trades
        sum_wins   = sum(wins)
        sum_losses = abs(sum(losses)) if losses else 0.0

        if sum_losses > 0:
            profit_factor = sum_wins / sum_losses
        elif sum_wins > 0:
            profit_factor = float("inf")
        else:
            profit_factor = 0.0

        avg_win  = sum_wins / len(wins) if wins else 0.0
        avg_loss = abs(sum(losses) / len(losses)) if losses else 0.0

        return {
            "n_trades":      n_trades,
            "win_rate":      win_rate,
            "profit_factor": profit_factor,
            "avg_win":       avg_win,
            "avg_loss":      avg_loss,
            "total_pnl":     sum(pnls),
            "best_trade":    max(pnls),
            "worst_trade":   min(pnls),
        }

    def generate_report(self, period: str = "daily") -> str:
        """
        Generiert einen Markdown-formatierten Performance-Report.

        Parameters
        ----------
        period : 'daily'  – heutiger Tag (00:00 UTC – jetzt)
                 'weekly' – letzte 7 Tage

        Returns
        -------
        str: Markdown-Text fuer direkten Export oder Telegram-Ausgabe.

        Raises
        ------
        ValueError: Unbekanntes period-Argument.
        """
        period_lower = period.lower()
        now = datetime.now(timezone.utc)

        if period_lower == "daily":
            start      = datetime(now.year, now.month, now.day, 0, 0, 0, tzinfo=timezone.utc)
            period_lbl = f"{now.date().isoformat()} (Taeglich)"
        elif period_lower == "weekly":
            start      = now - timedelta(days=7)
            period_lbl = (
                f"{start.date().isoformat()} – {now.date().isoformat()} (Woechentlich)"
            )
        else:
            raise ValueError(
                f"period muss 'daily' oder 'weekly' sein, nicht '{period!r}'"
            )

        stats = self.calculate_stats(start, now)

        pf_str = (
            "∞"
            if stats["profit_factor"] == float("inf")
            else f"{stats['profit_factor']:.2f}"
        )
        best_str  = (
            f"+{stats['best_trade']:.2f}" if stats["best_trade"] is not None else "–"
        )
        worst_str = (
            f"{stats['worst_trade']:.2f}" if stats["worst_trade"] is not None else "–"
        )
        total_sign = "+" if stats["total_pnl"] >= 0 else ""

        lines = [
            f"# QuantzAI Trade-Journal – {period_lbl}",
            "",
            f"*Zeitraum: {start.strftime('%Y-%m-%d %H:%M')} – "
            f"{now.strftime('%Y-%m-%d %H:%M')} UTC*",
            "",
            "| Kennzahl              | Wert           |",
            "|-----------------------|----------------|",
            f"| Trades gesamt         | {stats['n_trades']}              |",
            f"| Win-Rate              | {stats['win_rate']:.1%}         |",
            f"| Gewinnfaktor          | {pf_str}        |",
            f"| Durchschn. Gewinn     | {stats['avg_win']:.2f}         |",
            f"| Durchschn. Verlust    | {stats['avg_loss']:.2f}        |",
            f"| Bester Trade          | {best_str}         |",
            f"| Schlechtester Trade   | {worst_str}       |",
            f"| Gesamt-P&L            | {total_sign}{stats['total_pnl']:.2f}       |",
        ]
        return "\n".join(lines) + "\n"

    def get_pnl_sequence(
        self,
        start_date: DateLike,
        end_date: DateLike,
        symbol: Optional[str] = None,
    ) -> list[float]:
        """
        Gibt die geordnete PnL-Sequenz aller abgeschlossenen Trades im Zeitraum zurueck.

        Nuetzlich fuer Sharpe-Ratio- und Max-Drawdown-Berechnungen.

        Parameters
        ----------
        start_date : Startdatum/-zeit (inklusiv).
        end_date   : Enddatum/-zeit (inklusiv).
        symbol     : Optional – nur Trades fuer dieses Symbol.

        Returns
        -------
        list[float]: PnL-Werte chronologisch sortiert.
        """
        start_iso = _to_iso_start(start_date)
        end_iso   = _to_iso_end(end_date)

        if symbol is not None:
            sql = (
                "SELECT pnl FROM trades "
                "WHERE status = 'closed' AND pnl IS NOT NULL "
                "AND entry_time >= ? AND entry_time <= ? AND symbol = ? "
                "ORDER BY entry_time ASC"
            )
            params: tuple = (start_iso, end_iso, symbol)
        else:
            sql = (
                "SELECT pnl FROM trades "
                "WHERE status = 'closed' AND pnl IS NOT NULL "
                "AND entry_time >= ? AND entry_time <= ? "
                "ORDER BY entry_time ASC"
            )
            params = (start_iso, end_iso)

        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [float(row[0]) for row in rows]

    def get_open_positions(self) -> list[dict]:
        """
        Gibt alle aktuell offenen Trades (status='open') zurueck.

        Returns
        -------
        list[dict]: Offene Trades, juengste zuerst.
        """
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM trades WHERE status = 'open' ORDER BY entry_time DESC"
            )
            cols = [d[0] for d in cursor.description]
            rows = cursor.fetchall()
        return [dict(zip(cols, row)) for row in rows]

    # ── Interna ───────────────────────────────────────────────────────────────

    def _init_tables(self) -> None:
        with self._lock:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS trades (
                    id                INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol            TEXT    NOT NULL DEFAULT '',
                    direction         TEXT    NOT NULL DEFAULT '',
                    lot_size          REAL,
                    entry_price       REAL,
                    entry_time        TEXT,
                    exit_price        REAL,
                    exit_time         TEXT,
                    pnl               REAL,
                    regime            TEXT,
                    news_context      TEXT,
                    signal_confidence REAL,
                    setup             TEXT,
                    status            TEXT    NOT NULL DEFAULT 'open',
                    extra_json        TEXT
                );
            """)
            self._conn.commit()


# ─────────────────────────────────────────────────────────────────────────────
#  Hilfsfunktionen
# ─────────────────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _coerce_timestamp(ts) -> Optional[str]:
    """Konvertiert verschiedene Timestamp-Formate in ISO-String; None bei Fehler."""
    if ts is None:
        return None
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts.isoformat()
    if isinstance(ts, str) and ts:
        return ts
    return None


def _to_iso_start(d: DateLike) -> str:
    if isinstance(d, datetime):
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        return d.isoformat()
    if isinstance(d, date):
        return datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=timezone.utc).isoformat()
    return str(d)


def _to_iso_end(d: DateLike) -> str:
    if isinstance(d, datetime):
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        return d.isoformat()
    if isinstance(d, date):
        return datetime(
            d.year, d.month, d.day, 23, 59, 59, 999999, tzinfo=timezone.utc
        ).isoformat()
    return str(d)
