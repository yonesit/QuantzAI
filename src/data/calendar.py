"""
src/data/calendar.py
EconomicCalendar – Wirtschaftskalender-Integration fuer No-Trade-Zone-Erkennung.

Datenquelle: ForexFactory JSON-Feed (kostenlos, kein API-Key noetig).
Cached lokal, taeglich aktualisiert um 06:00 UTC.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import requests
from loguru import logger


# ─────────────────────────────────────────────
#  Konstanten
# ─────────────────────────────────────────────

_FOREXFACTORY_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

# Waehrung -> Symbole die davon betroffen sind (Mapping wird zur Laufzeit
# aus dem Symbol selbst abgeleitet: EURUSD enthaelt EUR und USD)
_VALID_IMPACTS = {"High", "Medium", "Low"}


# ─────────────────────────────────────────────
#  Event-Datenklasse
# ─────────────────────────────────────────────

@dataclass
class EconomicEvent:
    title: str
    country: str       # Waehrungscode, z.B. "EUR", "USD"
    date: str           # ISO-Timestamp (UTC)
    impact: str          # "High", "Medium", "Low"

    @property
    def timestamp(self) -> datetime:
        dt = datetime.fromisoformat(self.date.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt


# ─────────────────────────────────────────────
#  EconomicCalendar
# ─────────────────────────────────────────────

class EconomicCalendar:
    """
    Verwaltet Wirtschaftsereignisse und liefert No-Trade-Zone-Flags.

    Parameters
    ----------
    cache_dir         : Ordner fuer den lokalen JSON-Cache
    before_minutes    : Sperrzeit vor einem High-Impact-Event (Standard: 30)
    after_minutes     : Sperrzeit nach einem High-Impact-Event (Standard: 15)
    update_hour_utc   : Stunde (UTC) an der der Cache taeglich erneuert wird
    source_url        : URL des Kalender-Feeds
    timeout           : HTTP-Timeout in Sekunden
    """

    def __init__(
        self,
        cache_dir: str = "data/processed/calendar",
        before_minutes: int = 30,
        after_minutes: int = 15,
        update_hour_utc: int = 6,
        source_url: str = _FOREXFACTORY_URL,
        timeout: int = 10,
    ) -> None:
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        self._before = timedelta(minutes=before_minutes)
        self._after  = timedelta(minutes=after_minutes)
        self._update_hour = update_hour_utc
        self._source_url  = source_url
        self._timeout      = timeout

        self._events: list[EconomicEvent] = []
        self._last_update: Optional[datetime] = None

    # ── Cache-Datei ──────────────────────────────────

    @property
    def _cache_file(self) -> Path:
        return self._cache_dir / "calendar_cache.json"

    # ── Aktualisierung ───────────────────────────────

    def refresh(self, force: bool = False) -> bool:
        """
        Holt frische Kalenderdaten falls noetig (taeglich um update_hour_utc).

        Parameters
        ----------
        force : True = ignoriert das Update-Intervall und holt immer neu

        Returns
        -------
        bool: True wenn frische Daten geholt wurden, False wenn Cache verwendet wurde
              oder der Abruf fehlgeschlagen ist.
        """
        if not force and not self._needs_update():
            self._load_cache_if_empty()
            return False

        try:
            resp = requests.get(self._source_url, timeout=self._timeout)
            resp.raise_for_status()
            raw_events = resp.json()

            self._events = self._parse_events(raw_events)
            self._last_update = datetime.now(timezone.utc)
            self._save_cache()

            logger.info(
                "EconomicCalendar aktualisiert | {n} Events geladen",
                n=len(self._events),
            )
            return True

        except (requests.RequestException, ValueError) as exc:
            logger.error(
                "EconomicCalendar Update fehlgeschlagen | {exc} | Fallback auf Cache",
                exc=exc,
            )
            self._load_cache_if_empty()
            return False

    def _needs_update(self) -> bool:
        """Prueft ob seit dem letzten Update ein neuer update_hour_utc vergangen ist."""
        if self._last_update is None:
            return True

        now = datetime.now(timezone.utc)
        last_update_day = self._last_update.date()
        today = now.date()

        if today > last_update_day and now.hour >= self._update_hour:
            return True
        return False

    def _load_cache_if_empty(self) -> None:
        """Laedt den lokalen Cache wenn noch keine Events im Speicher sind."""
        if self._events:
            return
        if not self._cache_file.exists():
            return
        try:
            with open(self._cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._events = [EconomicEvent(**e) for e in data.get("events", [])]
            if data.get("last_update"):
                self._last_update = datetime.fromisoformat(data["last_update"])
            logger.info("EconomicCalendar aus Cache geladen | {n} Events", n=len(self._events))
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            logger.error("Cache laden fehlgeschlagen: {exc}", exc=exc)

    def _save_cache(self) -> None:
        data = {
            "last_update": self._last_update.isoformat() if self._last_update else None,
            "events": [asdict(e) for e in self._events],
        }
        with open(self._cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _parse_events(self, raw_events: list[dict]) -> list[EconomicEvent]:
        events = []
        for item in raw_events:
            try:
                impact = item.get("impact", "Low")
                if impact not in _VALID_IMPACTS:
                    impact = "Low"
                events.append(EconomicEvent(
                    title=item.get("title", "Unknown"),
                    country=item.get("country", "").upper(),
                    date=item.get("date", ""),
                    impact=impact,
                ))
            except (KeyError, TypeError) as exc:
                logger.warning("Event konnte nicht geparst werden: {exc}", exc=exc)
        return events

    # ── No-Trade-Zone ────────────────────────────────

    def is_no_trade_zone(self, symbol: str, timestamp: Optional[datetime] = None) -> bool:
        """
        Prueft ob fuer ein Symbol zu einem Zeitpunkt Handel blockiert ist.

        Parameters
        ----------
        symbol    : z.B. "EURUSD" (wird in Waehrungscodes EUR/USD aufgeteilt)
        timestamp : Zeitpunkt der Pruefung (Standard: jetzt, UTC)

        Returns
        -------
        bool: True = kein Handel (No-Trade-Zone oder Kalender nicht verfuegbar)
              False = Handel erlaubt
        """
        ts = timestamp or datetime.now(timezone.utc)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        # Konservativ: Kein Kalender geladen -> kein Trade
        if not self._events and self._last_update is None:
            self._load_cache_if_empty()
            if not self._events:
                logger.warning(
                    "EconomicCalendar: keine Daten verfuegbar – konservativ No-Trade-Zone"
                )
                return True

        currencies = self._extract_currencies(symbol)

        for event in self._events:
            if event.impact != "High":
                continue
            if event.country not in currencies:
                continue

            window_start = event.timestamp - self._before
            window_end   = event.timestamp + self._after

            if window_start <= ts <= window_end:
                logger.info(
                    "No-Trade-Zone aktiv | {symbol} | Event: {title} ({country}) @ {time}",
                    symbol=symbol, title=event.title, country=event.country,
                    time=event.timestamp,
                )
                return True

        return False

    def get_upcoming_events(
        self,
        symbol: Optional[str] = None,
        hours_ahead: int = 24,
        min_impact: str = "High",
    ) -> list[EconomicEvent]:
        """Gibt kommende Events zurueck, optional gefiltert nach Symbol und Impact."""
        now = datetime.now(timezone.utc)
        cutoff = now + timedelta(hours=hours_ahead)
        impact_order = {"Low": 0, "Medium": 1, "High": 2}
        min_level = impact_order.get(min_impact, 2)

        currencies = self._extract_currencies(symbol) if symbol else None

        result = []
        for event in self._events:
            if impact_order.get(event.impact, 0) < min_level:
                continue
            if currencies is not None and event.country not in currencies:
                continue
            if now <= event.timestamp <= cutoff:
                result.append(event)

        return sorted(result, key=lambda e: e.timestamp)

    @staticmethod
    def _extract_currencies(symbol: str) -> set[str]:
        """Zerlegt EURUSD in {EUR, USD}. Faellt auf Heuristik zurueck bei OANDA-Format."""
        s = symbol.upper().replace("_", "")
        if len(s) == 6:
            return {s[:3], s[3:]}
        # Fallback fuer Symbole wie XAUUSD (6 Zeichen, passt oben schon)
        return {s}
