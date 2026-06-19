"""
src/modes.py
Drei-Modi-System fuer den TradingOrchestrator.

SUGGEST_ONLY     – Bot erkennt Signale, zeigt sie an, eroeffnet NIEMALS selbst Orders.
CONFIRM_REQUIRED – Bot fragt aktiv nach Bestaetigung; Order nur bei explizitem Ja.
AUTONOMOUS       – Bot agiert vollstaendig selbststaendig; erfordert
                   Umgebungsvariable CONFIRM_AUTONOMOUS=yes als Schutzschranke.

Verwendung:
  from src.modes import TradingMode, ConfirmationCallback, is_autonomous_allowed

  orch = TradingOrchestrator(..., mode=TradingMode.SUGGEST_ONLY)
  orch.set_mode(TradingMode.CONFIRM_REQUIRED)

Sicherheitsprinzip:
  Der Wechsel in den AUTONOMOUS-Modus erfordert immer eine explizite
  Bestätigung durch die Umgebungsvariable CONFIRM_AUTONOMOUS=yes.
  Das verhindert, dass der Bot versehentlich in den autonomen Modus wechselt.
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Protocol, runtime_checkable

# Umgebungsvariable fuer AUTONOMOUS-Freigabe
AUTONOMOUS_ENV_VAR = "CONFIRM_AUTONOMOUS"
AUTONOMOUS_ENV_VAL = "yes"


class TradingMode(Enum):
    """Betriebsmodi des TradingOrchestrators."""

    SUGGEST_ONLY     = "suggest_only"
    CONFIRM_REQUIRED = "confirm_required"
    AUTONOMOUS       = "autonomous"


@runtime_checkable
class ConfirmationCallback(Protocol):
    """
    Callback-Protocol fuer den CONFIRM_REQUIRED-Modus.

    Wird aufgerufen nachdem Signal und Lot-Groesse berechnet wurden,
    bevor die Order platziert wird. Gibt True zurueck wenn die Order
    ausgefuehrt werden soll, False um sie zu verwerfen.
    """

    def confirm_order(
        self,
        symbol:    str,
        direction: str,
        lot_size:  float,
        sl:        float,
        tp:        float,
    ) -> bool: ...


def is_autonomous_allowed() -> bool:
    """
    Prueft ob der AUTONOMOUS-Modus durch die Umgebungsvariable freigegeben ist.

    Returns True wenn CONFIRM_AUTONOMOUS=yes gesetzt ist.
    """
    return os.environ.get(AUTONOMOUS_ENV_VAR, "").strip().lower() == AUTONOMOUS_ENV_VAL
