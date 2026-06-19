"""
src/journal/psychology_tracker.py
PsychologyTracker – Psychologie-Tracking fuer manuell bestaetigte Trades.

Erfasst Stimmung und Eroeffnungsgrund bei Trade-Eroeffnung,
Plan-Einhaltung und Schliessungsgrund bei Trade-Schliessung.
Verknuepft Eintraege mit der Trade-ID aus dem TradeJournal.

Tilt-Erkennung (check_tilt_state):
  1. N Verlusttrades in Folge (Standard: 5)
  2. Erhoehte Positionsgroesse nach Verlust (>= tilt_lot_increase_ratio, Standard: 1.2)
  3. Revenge-Mood: ANGRY oder FOMO direkt nach Verlust

Bei erkanntem Tilt: ruft optional TradingOrchestrator.pause() auf.

Mustererkennung (analyze_mood_patterns):
  Benoetigt >= mood_pattern_min_trades (Standard: 30) abgeschlossene Trades.
  Gibt pro MoodState Win-Rate und Trade-Anzahl zurueck.

Testbarkeit:
  _now_fn ist injizierbar um Zeitstempel-Abhaengigkeiten zu mocken.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional

from loguru import logger


# ─────────────────────────────────────────────────────────────────────────────
#  Enums & Dataclasses
# ─────────────────────────────────────────────────────────────────────────────

class MoodState(Enum):
    CALM          = "calm"
    FOCUSED       = "focused"
    NERVOUS       = "nervous"
    FOMO          = "fomo"
    ANGRY         = "angry"
    OVERCONFIDENT = "overconfident"


@dataclass
class TradeRecord:
    trade_id:       str | int
    symbol:         str
    lot_size:       float
    mood_open:      MoodState
    opening_reason: str
    opened_at:      datetime
    pnl:            float | None       = None
    mood_close:     MoodState | None   = None
    plan_followed:  bool | None        = None
    close_reason:   str                = ""
    closed_at:      datetime | None    = None


_REVENGE_MOODS = frozenset({MoodState.ANGRY, MoodState.FOMO})


# ─────────────────────────────────────────────────────────────────────────────
#  PsychologyTracker
# ─────────────────────────────────────────────────────────────────────────────

class PsychologyTracker:
    """
    Verfolgt die psychologische Dimension jedes Trades und erkennt Tilt-Muster.

    Parameters
    ----------
    orchestrator              : Objekt mit .pause()-Methode (z.B. TradingOrchestrator).
                                Wird bei erkanntem Tilt aufgerufen.
    tilt_consecutive_losses   : Anzahl Verluste in Folge bis Tilt ausgeloest wird.
    tilt_lot_increase_ratio   : Faktor fuer Lot-Erhoehung nach Verlust (z.B. 1.2 = 20%).
    mood_pattern_min_trades   : Min. abgeschlossene Trades fuer Muster-Analyse.
    _now_fn                   : Injectable fuer Tests (ersetzt datetime.now).
    """

    def __init__(
        self,
        orchestrator: Any                            = None,
        tilt_consecutive_losses: int                 = 5,
        tilt_lot_increase_ratio: float               = 1.2,
        mood_pattern_min_trades: int                 = 30,
        _now_fn: Optional[Callable[[], datetime]]    = None,
    ) -> None:
        self._orchestrator              = orchestrator
        self._tilt_consecutive_losses   = tilt_consecutive_losses
        self._tilt_lot_increase_ratio   = tilt_lot_increase_ratio
        self._mood_pattern_min_trades   = mood_pattern_min_trades
        self._now_fn                    = _now_fn or (lambda: datetime.now(timezone.utc))

        self._trades:     list[TradeRecord]       = []
        self._open_index: dict[str | int, int]    = {}  # trade_id -> index in _trades

    # ── Oeffentliche Schnittstelle ─────────────────────────────────────────────

    def record_open(
        self,
        trade_id:       str | int,
        symbol:         str,
        mood:           MoodState,
        opening_reason: str,
        lot_size:       float,
        timestamp:      datetime | None = None,
    ) -> None:
        """
        Erfasst den Eroeffnungszustand eines Trades.

        Wird im CONFIRM_REQUIRED-Modus nach manueller Bestaetigung aufgerufen.
        trade_id entspricht der ID aus TradeJournal.log_trade_open().
        """
        opened_at = timestamp or self._now_fn()
        record = TradeRecord(
            trade_id=trade_id,
            symbol=symbol,
            lot_size=lot_size,
            mood_open=mood,
            opening_reason=opening_reason,
            opened_at=opened_at,
        )
        idx = len(self._trades)
        self._trades.append(record)
        self._open_index[trade_id] = idx
        logger.debug(
            "PsychologyTracker: Trade eroeffnet | id={id} | symbol={sym} | mood={m}",
            id=trade_id, sym=symbol, m=mood.value,
        )

    def record_close(
        self,
        trade_id:      str | int,
        mood:          MoodState,
        plan_followed: bool,
        close_reason:  str,
        pnl:           float,
        timestamp:     datetime | None = None,
    ) -> None:
        """
        Erfasst den Abschlusszustand und prueft automatisch auf Tilt.

        Wenn Tilt erkannt wird und ein Orchestrator gesetzt ist, wird
        orchestrator.pause() aufgerufen.

        trade_id muss vorher via record_open() registriert worden sein.
        """
        idx = self._open_index.get(trade_id)
        if idx is None:
            logger.warning(
                "PsychologyTracker: Unbekannte Trade-ID {id} bei record_close – ignoriert.",
                id=trade_id,
            )
            return

        record = self._trades[idx]
        record.pnl           = pnl
        record.mood_close    = mood
        record.plan_followed = plan_followed
        record.close_reason  = close_reason
        record.closed_at     = timestamp or self._now_fn()
        del self._open_index[trade_id]

        logger.debug(
            "PsychologyTracker: Trade geschlossen | id={id} | pnl={p:.2f} | plan={pl}",
            id=trade_id, p=pnl, pl=plan_followed,
        )

        if self.check_tilt_state() and self._orchestrator is not None:
            logger.warning(
                "PsychologyTracker: Tilt erkannt nach Trade {id}! "
                "TradingOrchestrator.pause() wird aufgerufen.",
                id=trade_id,
            )
            self._orchestrator.pause()

    def check_tilt_state(
        self,
        recent_trades: list[TradeRecord] | None = None,
    ) -> bool:
        """
        Prueft ob Tilt-Verhalten vorliegt.

        Prueft drei Muster auf abgeschlossenen Trades:
          1. N Verluste in Folge (tilt_consecutive_losses, Standard 5)
          2. Lot-Erhoehung >= tilt_lot_increase_ratio nach Verlust
          3. ANGRY oder FOMO Stimmung direkt nach Verlust (Revenge-Trading)

        Parameters
        ----------
        recent_trades : Optionale Liste von TradeRecord. Wenn None, werden
                        die intern gespeicherten Trades verwendet.

        Returns
        -------
        True wenn mindestens ein Tilt-Muster erkannt wurde.
        """
        source = recent_trades if recent_trades is not None else self._trades
        closed = [t for t in source if t.pnl is not None]
        if not closed:
            return False

        # Muster 1: N Verluste in Folge
        loss_streak = 0
        for t in reversed(closed):
            if t.pnl < 0:
                loss_streak += 1
            else:
                break
        if loss_streak >= self._tilt_consecutive_losses:
            return True

        # Muster 2 & 3: paarweise Pruefung (vorige vs. aktuelle Trade)
        for i in range(1, len(closed)):
            prev, curr = closed[i - 1], closed[i]
            if prev.pnl < 0:
                # Lot-Eskalation
                if curr.lot_size >= prev.lot_size * self._tilt_lot_increase_ratio:
                    return True
                # Revenge-Mood
                if curr.mood_open in _REVENGE_MOODS:
                    return True

        return False

    def analyze_mood_patterns(self) -> dict[MoodState, dict[str, Any]]:
        """
        Analysiert Zusammenhang zwischen Stimmung und Trefferquote.

        Benoetigt mindestens mood_pattern_min_trades (Standard: 30) abgeschlossene Trades.
        Gibt ein leeres Dict zurueck wenn nicht genuegend Daten vorhanden.

        Returns
        -------
        dict[MoodState -> {"n_trades": int, "win_rate": float}]
        """
        closed = [t for t in self._trades if t.pnl is not None]
        if len(closed) < self._mood_pattern_min_trades:
            return {}

        stats: dict[MoodState, dict[str, int]] = defaultdict(
            lambda: {"n_trades": 0, "wins": 0}
        )
        for t in closed:
            stats[t.mood_open]["n_trades"] += 1
            if t.pnl > 0:
                stats[t.mood_open]["wins"] += 1

        return {
            mood: {
                "n_trades": v["n_trades"],
                "win_rate": v["wins"] / v["n_trades"] if v["n_trades"] > 0 else 0.0,
            }
            for mood, v in stats.items()
        }

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def all_trades(self) -> list[TradeRecord]:
        """Gibt eine Kopie aller erfassten Trades zurueck."""
        return list(self._trades)

    @property
    def open_trade_ids(self) -> list[str | int]:
        """Gibt IDs aller noch offenen (nicht geschlossenen) Trades zurueck."""
        return list(self._open_index.keys())
