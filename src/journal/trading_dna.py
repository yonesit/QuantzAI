"""
src/journal/trading_dna.py
TradingDNA – persoenliches Handelsprofil aus historischen Trade-Daten.

Analysiert nach ausreichend Trades (Standard: 500) persoenliche
Staerken und Schwaechen:
  - Beste/schlechteste Handelszeiten (Stunde des Tages, Wochentag)
  - Beste/schlechteste Symbole und Setups (nach Gesamt-PnL)
  - Optimale historische Positionsgroesse (Lot-Buckets, nach Win-Rate)
  - Wiederkehrende psychologische Schwaechen (via PsychologyTracker)

Sicherheitsprinzip:
  Unter der konfigurierbaren Mindestanzahl an Trades (Standard: 500)
  wird KEIN Profil erstellt. Stattdessen explizite Rueckmeldung, damit
  keine Entscheidungen auf statistisch unzuverlaessiger Basis getroffen
  werden.

Konfidenz-Stufen pro Bucket:
  'low'    : < 30 Trades  (statistisch unzuverlaessig)
  'medium' : 30–99 Trades (moderate Aussagekraft)
  'high'   : >= 100 Trades (hohe statistische Sicherheit)

Testbarkeit:
  _trade_loader injizierbar, um ohne echte TradeJournal-DB zu testen.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from loguru import logger


# ─────────────────────────────────────────────────────────────────────────────
#  Konstanten
# ─────────────────────────────────────────────────────────────────────────────

_WEEKDAYS = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday",
]

# Lot-Groessen-Buckets: (untere Grenze inkl., obere Grenze exkl., Label)
_LOT_BUCKETS: list[tuple[float, float, str]] = [
    (0.00,         0.05,         "< 0.05"),
    (0.05,         0.10,         "0.05–0.10"),
    (0.10,         0.20,         "0.10–0.20"),
    (0.20,         0.50,         "0.20–0.50"),
    (0.50,         float("inf"), ">= 0.50"),
]

_WEAKNESS_WIN_RATE_THRESHOLD = 0.40  # Moods unter dieser Win-Rate gelten als Schwaeche
_WEAKNESS_MIN_SAMPLES        = 10    # Mindest-Trades je Mood fuer Schwaeche-Erkennung


# ─────────────────────────────────────────────────────────────────────────────
#  Hilfsfunktionen
# ─────────────────────────────────────────────────────────────────────────────

def _confidence(n: int) -> str:
    if n >= 100:
        return "high"
    if n >= 30:
        return "medium"
    return "low"


def _parse_entry_time(trade: dict) -> Optional[datetime]:
    ts = trade.get("entry_time")
    if ts is None:
        return None
    try:
        if isinstance(ts, datetime):
            dt = ts
        else:
            dt = datetime.fromisoformat(str(ts))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def _best_worst(
    ranked_by_pnl: list[dict],
    n: int = 3,
) -> tuple[list[dict], list[dict]]:
    """Gibt (best[:n], worst[:n]) aus einer bereits absteigend sortierten Liste."""
    best  = ranked_by_pnl[:n]
    worst = list(reversed(ranked_by_pnl[-n:])) if len(ranked_by_pnl) > 1 else list(ranked_by_pnl)
    return best, worst


def _lot_bucket_index(lot: float) -> int:
    for i, (lo, hi, _) in enumerate(_LOT_BUCKETS):
        if lo <= lot < hi:
            return i
    return len(_LOT_BUCKETS) - 1


# ─────────────────────────────────────────────────────────────────────────────
#  TradingDNA
# ─────────────────────────────────────────────────────────────────────────────

class TradingDNA:
    """
    Erstellt ein persoenliches Handelsprofil aus historischen Trade-Daten.

    Parameters
    ----------
    journal              : TradeJournal-Instanz fuer Datenbankzugriff.
                           Kann None sein wenn _trade_loader gesetzt ist.
    min_trades           : Mindestanzahl abgeschlossener Trades (Standard: 500).
                           Unter diesem Wert wird kein Profil erstellt.
    psychology_tracker   : PsychologyTracker-Instanz fuer Schwaechen-Analyse.
                           Optional – wenn None, entfaellt Schwaechen-Analyse.
    _trade_loader        : Injectable fuer Tests: () -> list[dict].
                           Wenn gesetzt, wird journal ignoriert.
    """

    def __init__(
        self,
        journal: Any                                = None,
        min_trades: int                             = 500,
        psychology_tracker: Any                     = None,
        _trade_loader: Optional[Callable[[], list[dict]]] = None,
    ) -> None:
        self._journal     = journal
        self._min_trades  = min_trades
        self._psychology  = psychology_tracker
        self._trade_loader = _trade_loader

    # ── Oeffentliche Schnittstelle ────────────────────────────────────────────

    def generate_profile(self) -> dict:
        """
        Generiert das vollstaendige Handelsprofil.

        Returns
        -------
        dict mit status='ready' und allen Analyse-Ergebnissen, oder
        dict mit status='insufficient_data' und erklaerenden Feldern.
        """
        trades = self._load_trades()
        n = len(trades)

        if n < self._min_trades:
            msg = (
                f"{n} Trades vorhanden – mindestens {self._min_trades} "
                "benoetigt fuer ein zuverlaessiges Profil ohne Scheinsicherheit."
            )
            logger.info("TradingDNA: {msg}", msg=msg)
            return {
                "status":               "insufficient_data",
                "n_trades":             n,
                "min_trades_required":  self._min_trades,
                "message":              msg,
            }

        logger.info("TradingDNA: Profil wird erstellt | {n} Trades", n=n)

        return {
            "status":                   "ready",
            "n_trades":                 n,
            "min_trades_required":      self._min_trades,
            "trading_hours":            self._analyze_hours(trades),
            "trading_weekdays":         self._analyze_weekdays(trades),
            "symbols":                  self._analyze_dimension(trades, "symbol"),
            "setups":                   self._analyze_dimension(trades, "setup"),
            "position_sizing":          self._analyze_position_sizing(trades),
            "psychological_weaknesses": self._detect_psychological_weaknesses(),
        }

    # ── Interne Analyse-Methoden ──────────────────────────────────────────────

    def _analyze_hours(self, trades: list[dict]) -> dict:
        groups: dict[int, list[float]] = defaultdict(list)
        for t in trades:
            dt  = _parse_entry_time(t)
            pnl = t.get("pnl")
            if dt is None or pnl is None:
                continue
            groups[dt.hour].append(float(pnl))

        ranked: list[dict] = []
        for hour in range(24):
            pnls = groups.get(hour)
            if not pnls:
                continue
            n_h  = len(pnls)
            wins = sum(1 for p in pnls if p > 0)
            ranked.append({
                "hour":       hour,
                "win_rate":   wins / n_h,
                "total_pnl":  sum(pnls),
                "n_trades":   n_h,
                "confidence": _confidence(n_h),
            })

        ranked_by_pnl = sorted(ranked, key=lambda x: x["total_pnl"], reverse=True)
        best, worst   = _best_worst(ranked_by_pnl)
        return {"ranked": ranked_by_pnl, "best": best, "worst": worst}

    def _analyze_weekdays(self, trades: list[dict]) -> dict:
        groups: dict[int, list[float]] = defaultdict(list)
        for t in trades:
            dt  = _parse_entry_time(t)
            pnl = t.get("pnl")
            if dt is None or pnl is None:
                continue
            groups[dt.weekday()].append(float(pnl))

        ranked: list[dict] = []
        for wd_idx in range(7):
            pnls = groups.get(wd_idx)
            if not pnls:
                continue
            n_d  = len(pnls)
            wins = sum(1 for p in pnls if p > 0)
            ranked.append({
                "weekday":     _WEEKDAYS[wd_idx],
                "weekday_idx": wd_idx,
                "win_rate":    wins / n_d,
                "total_pnl":   sum(pnls),
                "n_trades":    n_d,
                "confidence":  _confidence(n_d),
            })

        ranked_by_pnl = sorted(ranked, key=lambda x: x["total_pnl"], reverse=True)
        best, worst   = _best_worst(ranked_by_pnl)
        return {"ranked": ranked_by_pnl, "best": best, "worst": worst}

    def _analyze_dimension(self, trades: list[dict], key: str) -> dict:
        """Analysiert eine beliebige kategorische Dimension (symbol, setup)."""
        groups: dict[str, list[float]] = defaultdict(list)
        for t in trades:
            val = t.get(key)
            pnl = t.get("pnl")
            if val is None or pnl is None:
                continue
            groups[str(val)].append(float(pnl))

        ranked: list[dict] = []
        for val, pnls in groups.items():
            n_v  = len(pnls)
            wins = sum(1 for p in pnls if p > 0)
            ranked.append({
                key:          val,
                "win_rate":   wins / n_v,
                "total_pnl":  sum(pnls),
                "n_trades":   n_v,
                "confidence": _confidence(n_v),
            })

        ranked_by_pnl = sorted(ranked, key=lambda x: x["total_pnl"], reverse=True)
        best, worst   = _best_worst(ranked_by_pnl)
        return {"ranked": ranked_by_pnl, "best": best, "worst": worst}

    def _analyze_position_sizing(self, trades: list[dict]) -> dict:
        bucket_pnls: dict[int, list[float]] = defaultdict(list)
        for t in trades:
            lot = t.get("lot_size")
            pnl = t.get("pnl")
            if lot is None or pnl is None:
                continue
            bucket_pnls[_lot_bucket_index(float(lot))].append(float(pnl))

        lot_buckets: list[dict] = []
        best_wr      = -1.0
        optimal_range: list | None = None
        optimal_label: str | None  = None

        for i, (lo, hi, label) in enumerate(_LOT_BUCKETS):
            pnls = bucket_pnls.get(i)
            if not pnls:
                continue
            n_b  = len(pnls)
            wins = sum(1 for p in pnls if p > 0)
            wr   = wins / n_b
            lot_max = hi if hi != float("inf") else None

            lot_buckets.append({
                "label":      label,
                "lot_min":    lo,
                "lot_max":    lot_max,
                "win_rate":   wr,
                "total_pnl":  sum(pnls),
                "n_trades":   n_b,
                "confidence": _confidence(n_b),
            })

            if wr > best_wr:
                best_wr       = wr
                optimal_range = [lo, lot_max]
                optimal_label = label

        return {
            "lot_buckets":       lot_buckets,
            "optimal_lot_range": optimal_range,
            "optimal_label":     optimal_label,
            "optimal_win_rate":  best_wr if lot_buckets else None,
        }

    def _detect_psychological_weaknesses(self) -> list[str]:
        if self._psychology is None:
            return []

        try:
            patterns = self._psychology.analyze_mood_patterns()
        except Exception as exc:  # noqa: BLE001
            logger.warning("TradingDNA: analyze_mood_patterns() Fehler: {exc}", exc=exc)
            return []

        weaknesses: list[str] = []
        for mood, stats in patterns.items():
            if (
                stats["win_rate"] < _WEAKNESS_WIN_RATE_THRESHOLD
                and stats["n_trades"] >= _WEAKNESS_MIN_SAMPLES
            ):
                weaknesses.append(
                    f"{mood.value.upper()}: win_rate={stats['win_rate']:.0%} "
                    f"({stats['n_trades']} Trades)"
                )

        return sorted(weaknesses)

    # ── Datenladen ────────────────────────────────────────────────────────────

    def _load_trades(self) -> list[dict]:
        if self._trade_loader is not None:
            return self._trade_loader()
        if self._journal is not None:
            return self._fetch_from_journal()
        return []

    def _fetch_from_journal(self) -> list[dict]:
        with self._journal._lock:
            cur  = self._journal._conn.execute(
                "SELECT * FROM trades WHERE status='closed' AND pnl IS NOT NULL"
            )
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
