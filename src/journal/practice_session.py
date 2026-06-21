"""
src/journal/practice_session.py
PracticeSession – Manueller Uebungsmodus auf historischen Daten.

Kein Look-ahead: waehrend der Uebung sind nur Daten bis zum aktuellen
Wiedergabe-Zeitpunkt sichtbar. Vollstaendig getrennt vom echten TradeJournal
und echten Order-Systemen.

Baut auf der Parquet-Lade-Infrastruktur aus src/journal/replay.py auf –
keine doppelte Chart-Rekonstruktions-Logik.

Persistenz:
  PracticeSessionStore: eigene SQLite-Datenbank (standard: data/processed/practice_trades.db)
  Voellig getrennt von TradeJournal.
"""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from loguru import logger

from src.journal.replay import ReplayDataNotFoundError, load_candles_for_range


# ─────────────────────────────────────────────────────────────────────────────
#  Daten-Typen
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PracticePosition:
    """Eine offene oder geschlossene Uebungs-Position."""
    position_id: int
    direction: str          # "buy" | "sell"
    entry_price: float
    lot_size: float
    sl: float
    tp: float
    entry_candle_idx: int
    is_open: bool = True
    exit_price: Optional[float] = None
    exit_candle_idx: Optional[int] = None
    pnl: Optional[float] = None
    closed_by: str = ""     # "manual" | "sl" | "tp"


@dataclass
class PracticeResult:
    """Ergebnis einer geschlossenen Uebungs-Position inkl. Counterfactual."""
    position: PracticePosition
    counterfactual_pnl: float   # Was waere mit entgegengesetzter Entscheidung passiert?


# ─────────────────────────────────────────────────────────────────────────────
#  PracticeSession
# ─────────────────────────────────────────────────────────────────────────────

class PracticeSession:
    """
    Verwaltet eine Uebungs-Session ueber einen historischen Zeitraum.

    No-Lookahead-Garantie:
        current_candles gibt ausschliesslich Kerzen zurueck, deren Index
        <= self.cursor liegt. Zukuenftige Kerzen sind nie zugreifbar.

    Parameters
    ----------
    symbol     : Handelssymbol (z. B. 'EURUSD').
    timeframe  : Timeframe-Kuerzel (z. B. 'H1').
    all_candles: Alle Kerzen des Zeitraums (intern gespeichert, nie vollstaendig
                 nach aussen gegeben – kein Look-ahead).
    """

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        all_candles: list[dict],
    ) -> None:
        self._symbol = symbol
        self._timeframe = timeframe
        self._all_candles: list[dict] = list(all_candles)
        self._cursor: int = 0
        self._positions: dict[int, PracticePosition] = {}
        self._closed: list[PracticePosition] = []
        self._results: list[PracticeResult] = []
        self._next_id: int = 1

    # ── Klassenmethode ────────────────────────────────────────────────────────

    @classmethod
    def from_range(
        cls,
        symbol: str,
        timeframe: str,
        start_dt: datetime,
        end_dt: datetime,
        features_dir: str | Path = "data/features",
    ) -> "PracticeSession":
        """
        Erstellt eine PracticeSession fuer den angegebenen Zeitraum.

        Laedt Kerzen aus Parquet via load_candles_for_range (geteilte
        Replay-Infrastruktur, keine doppelte Chart-Rekonstruktion).

        Raises
        ------
        ReplayDataNotFoundError : Keine Parquet-Daten fuer Symbol/Zeitraum.
        """
        candles = load_candles_for_range(
            features_dir=features_dir,
            symbol=symbol,
            timeframe=timeframe,
            start_dt=start_dt,
            end_dt=end_dt,
        )
        logger.debug(
            "PracticeSession: {n} Kerzen geladen | {sym} {tf} | {s} – {e}",
            n=len(candles), sym=symbol, tf=timeframe,
            s=start_dt.isoformat(), e=end_dt.isoformat(),
        )
        return cls(symbol=symbol, timeframe=timeframe, all_candles=candles)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def symbol(self) -> str:
        return self._symbol

    @property
    def timeframe(self) -> str:
        return self._timeframe

    @property
    def cursor(self) -> int:
        return self._cursor

    @property
    def total_candles(self) -> int:
        return len(self._all_candles)

    @property
    def is_at_end(self) -> bool:
        return self._cursor >= len(self._all_candles) - 1

    @property
    def current_candle(self) -> Optional[dict]:
        """Aktuelle Kerze (bei cursor)."""
        if not self._all_candles:
            return None
        return self._all_candles[self._cursor]

    @property
    def current_candles(self) -> list[dict]:
        """
        Alle Kerzen bis zum aktuellen Zeitpunkt (No-Lookahead-Garantie).

        Gibt ausschliesslich self._all_candles[:cursor + 1] zurueck.
        Zukuenftige Kerzen (Index > cursor) werden NIEMALS zurueckgegeben.
        """
        return self._all_candles[: self._cursor + 1]

    # ── Wiedergabe-Steuerung ──────────────────────────────────────────────────

    def advance(self, steps: int = 1) -> int:
        """
        Rueckt den Cursor um `steps` Kerzen vor.

        Prueft nach dem Vorrucken automatisch SL/TP fuer offene Positionen.

        Returns
        -------
        int : Tatsaechlich vorgerueckte Schritte (0 wenn bereits am Ende).
        """
        if self.is_at_end:
            return 0
        old = self._cursor
        self._cursor = min(self._cursor + steps, len(self._all_candles) - 1)
        self._check_sl_tp()
        return self._cursor - old

    # ── Positions-Management ──────────────────────────────────────────────────

    def open_position(
        self,
        direction: str,
        lot_size: float,
        sl: float,
        tp: float,
    ) -> int:
        """
        Oeffnet eine simulierte Uebungs-Position.

        Parameters
        ----------
        direction : 'buy' oder 'sell'.
        lot_size  : Lot-Groesse (positiv).
        sl        : Stop-Loss-Preis.
        tp        : Take-Profit-Preis.

        Returns
        -------
        int : Eindeutige Position-ID.

        Raises
        ------
        ValueError : Ungueltiges direction oder keine Kerzen geladen.
        """
        if direction not in ("buy", "sell"):
            raise ValueError(f"Ungueltiges direction: {direction!r}. Erwartet 'buy' oder 'sell'.")
        if not self._all_candles:
            raise ValueError("Keine Kerzen geladen. PracticeSession hat leere Candle-Liste.")

        entry_price = self.current_candle["close"]  # type: ignore[index]
        pid = self._next_id
        self._next_id += 1
        pos = PracticePosition(
            position_id=pid,
            direction=direction,
            entry_price=entry_price,
            lot_size=lot_size,
            sl=sl,
            tp=tp,
            entry_candle_idx=self._cursor,
        )
        self._positions[pid] = pos
        logger.debug(
            "PracticeSession: Position {id} geoeffnet | {dir} @ {price} | SL={sl} TP={tp}",
            id=pid, dir=direction, price=entry_price, sl=sl, tp=tp,
        )
        return pid

    def close_position(
        self,
        position_id: int,
        exit_price: Optional[float] = None,
    ) -> PracticeResult:
        """
        Schliesst eine Uebungs-Position und berechnet P&L + Counterfactual.

        Parameters
        ----------
        position_id : ID aus open_position().
        exit_price  : Schlusskurs (None = close der aktuellen Kerze).

        Returns
        -------
        PracticeResult : Enthaelt geschlossene Position und Counterfactual-P&L.

        Raises
        ------
        KeyError : position_id nicht gefunden.
        """
        if position_id not in self._positions:
            raise KeyError(f"Position {position_id} nicht in offenen Positionen gefunden.")

        pos = self._positions.pop(position_id)
        ep = exit_price if exit_price is not None else self.current_candle["close"]  # type: ignore[index]
        pos.exit_price = ep
        pos.exit_candle_idx = self._cursor
        pos.is_open = False
        if not pos.closed_by:
            pos.closed_by = "manual"

        direction_mult = 1.0 if pos.direction == "buy" else -1.0
        pos.pnl = (ep - pos.entry_price) * pos.lot_size * 100_000 * direction_mult
        counterfactual_pnl = (ep - pos.entry_price) * pos.lot_size * 100_000 * (-direction_mult)

        self._closed.append(pos)
        result = PracticeResult(position=pos, counterfactual_pnl=counterfactual_pnl)
        self._results.append(result)
        logger.debug(
            "PracticeSession: Position {id} geschlossen | pnl={pnl:.2f} | via={by}",
            id=position_id, pnl=pos.pnl, by=pos.closed_by,
        )
        return result

    # ── Statistiken ──────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """
        Berechnet Statistiken fuer die aktuelle Session.

        Getrennt vom echten TradeJournal – enthaelt ausschliesslich
        Uebungs-Trades dieser Session.
        """
        if not self._closed:
            return {
                "trade_count": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
            }
        wins = sum(1 for p in self._closed if (p.pnl or 0.0) > 0)
        total = len(self._closed)
        return {
            "trade_count": total,
            "wins": wins,
            "losses": total - wins,
            "win_rate": wins / total,
            "total_pnl": sum(p.pnl or 0.0 for p in self._closed),
        }

    def get_open_positions(self) -> list[PracticePosition]:
        """Gibt Kopie der Liste offener Positionen zurueck."""
        return list(self._positions.values())

    def get_closed_positions(self) -> list[PracticePosition]:
        """Gibt Kopie der Liste geschlossener Positionen zurueck."""
        return list(self._closed)

    def get_results(self) -> list[PracticeResult]:
        """Gibt alle Ergebnisse (PracticeResult) dieser Session zurueck."""
        return list(self._results)

    # ── Private ───────────────────────────────────────────────────────────────

    def _check_sl_tp(self) -> None:
        """Schliesst Positionen automatisch bei SL- oder TP-Erreichung."""
        current = self.current_candle
        if not current:
            return
        high = current.get("high") or current.get("close", 0.0)
        low = current.get("low") or current.get("close", 0.0)

        to_close: list[tuple[int, float, str]] = []
        for pid, pos in list(self._positions.items()):
            if pos.direction == "buy":
                if low <= pos.sl:
                    to_close.append((pid, pos.sl, "sl"))
                elif high >= pos.tp:
                    to_close.append((pid, pos.tp, "tp"))
            else:
                if high >= pos.sl:
                    to_close.append((pid, pos.sl, "sl"))
                elif low <= pos.tp:
                    to_close.append((pid, pos.tp, "tp"))

        for pid, ep, reason in to_close:
            if pid in self._positions:
                self._positions[pid].closed_by = reason
                self.close_position(pid, ep)


# ─────────────────────────────────────────────────────────────────────────────
#  PracticeSessionStore – SQLite-Persistenz (getrennt von TradeJournal)
# ─────────────────────────────────────────────────────────────────────────────

class PracticeSessionStore:
    """
    SQLite-Speicher fuer Uebungs-Trades.

    Voellig getrennt vom echten TradeJournal – schreibt in eine eigene
    Datenbankdatei und teilt keine Tabellen oder Verbindungen.

    Parameters
    ----------
    db_path : Pfad zur SQLite-Datei.
              Standard: 'data/processed/practice_trades.db'.
    """

    def __init__(
        self,
        db_path: str | Path = "data/processed/practice_trades.db",
    ) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._lock = threading.Lock()
        self._init_tables()
        logger.info(
            "PracticeSessionStore: Datenbank geoeffnet | {path}", path=self._path
        )

    def _init_tables(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS practice_trades (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol           TEXT    NOT NULL,
                    timeframe        TEXT    NOT NULL,
                    direction        TEXT    NOT NULL,
                    entry_price      REAL,
                    exit_price       REAL,
                    lot_size         REAL,
                    sl               REAL,
                    tp               REAL,
                    pnl              REAL,
                    closed_by        TEXT,
                    entry_candle_idx INTEGER,
                    exit_candle_idx  INTEGER,
                    session_date     TEXT,
                    created_at       TEXT DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            self._conn.commit()

    def save_result(
        self,
        result: PracticeResult,
        symbol: str,
        timeframe: str,
    ) -> int:
        """
        Speichert das Ergebnis einer geschlossenen Uebungs-Position.

        Returns
        -------
        int : Primaerschluessel des neuen Eintrags.
        """
        pos = result.position
        session_date = datetime.now(timezone.utc).date().isoformat()
        with self._lock:
            cursor = self._conn.execute(
                """
                INSERT INTO practice_trades
                    (symbol, timeframe, direction, entry_price, exit_price,
                     lot_size, sl, tp, pnl, closed_by,
                     entry_candle_idx, exit_candle_idx, session_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    symbol, timeframe, pos.direction,
                    pos.entry_price, pos.exit_price,
                    pos.lot_size, pos.sl, pos.tp, pos.pnl, pos.closed_by,
                    pos.entry_candle_idx, pos.exit_candle_idx, session_date,
                ),
            )
            self._conn.commit()
            return cursor.lastrowid  # type: ignore[return-value]

    def get_global_stats(self) -> dict:
        """
        Globale Statistiken ueber alle gespeicherten Uebungs-Trades.

        Trefferquote und Trade-Anzahl ueber alle Sessions hinweg.
        """
        with self._lock:
            row = self._conn.execute(
                """
                SELECT
                    COUNT(*),
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END),
                    SUM(pnl)
                FROM practice_trades
                WHERE pnl IS NOT NULL
                """
            ).fetchone()

        total: int = row[0] or 0
        wins: int  = int(row[1] or 0)
        total_pnl: float = float(row[2] or 0.0)
        return {
            "trade_count": total,
            "wins": wins,
            "losses": total - wins,
            "win_rate": wins / total if total else 0.0,
            "total_pnl": total_pnl,
        }

    def close(self) -> None:
        """Schliesst die Datenbankverbindung."""
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "PracticeSessionStore":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
