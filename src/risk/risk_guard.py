"""
src/risk/risk_guard.py
RiskGuard – ueberwacht den Kontostand und stoppt den Handel automatisch
bei Erreichen definierter Verlustgrenzen.

Eigenstaendiges Modul, keine Abhaengigkeiten zu anderen Risk-Modulen.
State wird persistiert und ueberlebt einen Neustart.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, date, timedelta
from pathlib import Path
from typing import Optional

from loguru import logger


# ─────────────────────────────────────────────
#  State-Datenklasse
# ─────────────────────────────────────────────

@dataclass
class RiskState:
    day_start_balance: float
    day_start_date: str              # ISO-Datum, z.B. "2026-06-18"
    all_time_high: float
    daily_limit_hit_at: Optional[str] = None      # ISO-Timestamp oder None
    max_drawdown_hit: bool = False
    post_loss_days_remaining: int = 0
    current_balance: float = 0.0


# ─────────────────────────────────────────────
#  RiskGuard
# ─────────────────────────────────────────────

class RiskGuard:
    """
    Ueberwacht den Kontostand laufend und entscheidet ob Handel erlaubt ist.

    Parameters
    ----------
    daily_loss_limit_pct       : Tägliches Verlustlimit in % vom Tagesstart-Kontostand (Standard: 5.0)
    max_drawdown_pct           : Maximaler Drawdown in % vom Allzeithoch (Standard: 15.0)
    post_loss_days             : Tage mit reduzierter Positionsgroesse nach Tageslimit-Treffer (Standard: 3)
    post_loss_size_multiplier  : Multiplikator fuer Positionsgroesse waehrend Post-Loss-Phase (Standard: 0.5)
    state_path                 : Pfad zur persistierten State-Datei
    """

    def __init__(
        self,
        daily_loss_limit_pct: float = 5.0,
        max_drawdown_pct: float = 15.0,
        post_loss_days: int = 3,
        post_loss_size_multiplier: float = 0.5,
        state_path: str = "data/processed/risk_state.json",
    ) -> None:
        self._daily_limit_pct = daily_loss_limit_pct
        self._max_drawdown_pct = max_drawdown_pct
        self._post_loss_days = post_loss_days
        self._post_loss_multiplier = post_loss_size_multiplier
        self._state_path = Path(state_path)
        self._state_path.parent.mkdir(parents=True, exist_ok=True)

        self._state: Optional[RiskState] = None
        self._load_state()

    # ── Persistenz ───────────────────────────────────

    def _load_state(self) -> None:
        if self._state_path.exists():
            try:
                with open(self._state_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._state = RiskState(**data)
                logger.info("RiskGuard: State aus Datei geladen.")
                return
            except (OSError, json.JSONDecodeError, TypeError) as exc:
                logger.warning("RiskGuard: State laden fehlgeschlagen: {exc}", exc=exc)

        self._state = None  # wird beim ersten update_balance() initialisiert

    def _save_state(self) -> None:
        if self._state is None:
            return
        with open(self._state_path, "w", encoding="utf-8") as f:
            json.dump(asdict(self._state), f, indent=2)

    # ── Kernlogik ────────────────────────────────────

    def update_balance(self, current_balance: float) -> None:
        """
        Aktualisiert den ueberwachten Kontostand. Muss regelmaessig
        (z.B. nach jedem Trade oder periodisch) aufgerufen werden.

        Initialisiert den State beim ersten Aufruf. Setzt den
        Tagesstart-Kontostand automatisch um 00:00 UTC zurueck.
        """
        today = self._today_str()

        if self._state is None:
            self._state = RiskState(
                day_start_balance=current_balance,
                day_start_date=today,
                all_time_high=current_balance,
                current_balance=current_balance,
            )
        else:
            # Tageswechsel erkannt -> Reset des Tagesstart-Kontostands
            if self._state.day_start_date != today:
                self._on_new_day(current_balance, today)

            self._state.current_balance = current_balance
            if current_balance > self._state.all_time_high:
                self._state.all_time_high = current_balance

            # Tageslimit pruefen
            if self._compute_daily_loss_pct(current_balance) >= self._daily_limit_pct:
                if self._state.daily_limit_hit_at is None:
                    self._state.daily_limit_hit_at = datetime.now(timezone.utc).isoformat()
                    self._state.post_loss_days_remaining = self._post_loss_days
                    logger.error(
                        "RiskGuard: TAEGLICHES VERLUSTLIMIT ERREICHT | balance={bal:.2f}",
                        bal=current_balance,
                    )

            # Max Drawdown pruefen
            if self._compute_drawdown_pct(current_balance) >= self._max_drawdown_pct:
                if not self._state.max_drawdown_hit:
                    self._state.max_drawdown_hit = True
                    logger.error(
                        "RiskGuard: MAXIMALER DRAWDOWN ERREICHT | balance={bal:.2f} ath={ath:.2f}",
                        bal=current_balance, ath=self._state.all_time_high,
                    )

        self._save_state()

    def _on_new_day(self, current_balance: float, today: str) -> None:
        """Wird bei Tageswechsel (00:00 UTC) ausgefuehrt."""
        logger.info("RiskGuard: Neuer Handelstag | Tagesstart-Kontostand zurueckgesetzt.")
        self._state.day_start_balance = current_balance
        self._state.day_start_date = today
        self._state.daily_limit_hit_at = None  # Tageslimit-Block wird aufgehoben

        if self._state.post_loss_days_remaining > 0:
            self._state.post_loss_days_remaining -= 1

    def _compute_daily_loss_pct(self, current_balance: float) -> float:
        if self._state.day_start_balance <= 0:
            return 0.0
        loss = self._state.day_start_balance - current_balance
        return max(0.0, (loss / self._state.day_start_balance) * 100.0)

    def _compute_drawdown_pct(self, current_balance: float) -> float:
        if self._state.all_time_high <= 0:
            return 0.0
        drawdown = self._state.all_time_high - current_balance
        return max(0.0, (drawdown / self._state.all_time_high) * 100.0)

    @staticmethod
    def _today_str() -> str:
        return datetime.now(timezone.utc).date().isoformat()

    # ── Oeffentliche Abfrage-Methoden ────────────────

    def is_daily_limit_hit(self) -> bool:
        """True wenn das taegliche Verlustlimit heute bereits erreicht wurde."""
        if self._state is None:
            return False
        return self._state.daily_limit_hit_at is not None

    def is_max_drawdown_hit(self) -> bool:
        """True wenn der maximale Drawdown erreicht wurde (manuelle Freigabe noetig)."""
        if self._state is None:
            return False
        return self._state.max_drawdown_hit

    def is_trading_allowed(self) -> bool:
        """
        True wenn neue Trades erlaubt sind.
        False wenn: Tageslimit erreicht (blockiert fuer den Rest des Tages)
                    ODER maximaler Drawdown erreicht (globaler Stop).
        """
        if self.is_max_drawdown_hit():
            return False
        if self.is_daily_limit_hit():
            return False
        return True

    def get_position_size_multiplier(self) -> float:
        """
        Gibt den Multiplikator fuer die Positionsgroesse zurueck.
        1.0 = normal, post_loss_size_multiplier (Standard 0.5) waehrend
        der Post-Loss-Phase nach einem Tageslimit-Treffer.
        """
        if self._state is None:
            return 1.0
        if self._state.post_loss_days_remaining > 0:
            return self._post_loss_multiplier
        return 1.0

    def reset_max_drawdown(self) -> None:
        """Manuelle Freigabe nach erreichtem maximalen Drawdown."""
        if self._state is not None:
            self._state.max_drawdown_hit = False
            self._save_state()
            logger.info("RiskGuard: Max-Drawdown-Block manuell aufgehoben.")

    @property
    def state(self) -> Optional[RiskState]:
        """Read-only Zugriff auf den aktuellen State (fuer GUI/Monitoring)."""
        return self._state
