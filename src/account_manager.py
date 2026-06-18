"""
src/account_manager.py
AccountManager – verwaltet mehrere MT5/OANDA-Konten parallel.

Jedes Konto hat:
  - eine eindeutige ID
  - eigene Komponenten (Connector, OrderExecutor, RiskGuard, PositionSizer)
  - eigenen isolierten Risiko-State
  - optional einen TradingOrchestrator

Konten werden per register_account() hinzugefuegt.
Konfiguration in config.yaml unter accounts: (Login-Daten in .env per Konto-ID).
"""

from __future__ import annotations

import concurrent.futures
import threading
from dataclasses import dataclass, field
from typing import Any, Optional

from loguru import logger


@dataclass
class AccountConfig:
    """Statische Konfiguration eines einzelnen Kontos."""
    account_id: str
    symbols: list[str]
    risk_config: dict[str, Any] = field(default_factory=dict)


@dataclass
class AccountContext:
    """
    Alle laufzeitrelevanten Komponenten fuer ein Konto.

    Parameters
    ----------
    account_id      : Eindeutige Konto-ID (z.B. "demo", "live_1").
    config          : AccountConfig mit Symbolen und Risiko-Parametern.
    connector       : MT5Connector oder OANDAConnector fuer dieses Konto.
    order_executor  : OrderExecutor – Paper- oder Live-Trading.
    risk_guard      : RiskGuard – isolierter Risiko-State pro Konto.
    position_sizer  : PositionSizer – konto-spezifische Lot-Berechnung.
    orchestrator    : TradingOrchestrator (optional, fuer run_cycle).
    """
    account_id: str
    config: AccountConfig
    connector: Any
    order_executor: Any
    risk_guard: Any
    position_sizer: Any
    orchestrator: Optional[Any] = None


class AccountManager:
    """
    Verwaltet mehrere Trading-Konten parallel.

    Jedes Konto hat vollstaendig isolierte Komponenten – kein gemeinsamer
    State zwischen Konten. Das ermoeglicht z.B. ein Demo- und ein Live-Konto
    oder mehrere Strategien auf getrennten Konten.

    Konten werden ueber register_account() hinzugefuegt.
    """

    def __init__(self) -> None:
        self._accounts: dict[str, AccountContext] = {}
        self._lock = threading.Lock()

    # ── Konten-Verwaltung ─────────────────────────────────────────────────────

    def register_account(self, context: AccountContext) -> None:
        """
        Registriert ein Konto.

        Raises
        ------
        ValueError wenn die account_id bereits registriert ist.
        """
        with self._lock:
            if context.account_id in self._accounts:
                raise ValueError(
                    f"Konto '{context.account_id}' ist bereits registriert."
                )
            self._accounts[context.account_id] = context
        logger.info("AccountManager: Konto '{id}' registriert.", id=context.account_id)

    def get_account(self, account_id: str) -> AccountContext:
        """
        Gibt den AccountContext fuer ein Konto zurueck.

        Raises
        ------
        KeyError wenn account_id nicht gefunden.
        """
        with self._lock:
            if account_id not in self._accounts:
                raise KeyError(
                    f"Konto '{account_id}' nicht gefunden. "
                    f"Registriert: {list(self._accounts.keys())}"
                )
            return self._accounts[account_id]

    def list_account_ids(self) -> list[str]:
        """Gibt alle registrierten Konto-IDs zurueck."""
        with self._lock:
            return list(self._accounts.keys())

    def remove_account(self, account_id: str) -> None:
        """
        Entfernt ein Konto.

        Raises
        ------
        KeyError wenn account_id nicht gefunden.
        """
        with self._lock:
            if account_id not in self._accounts:
                raise KeyError(f"Konto '{account_id}' nicht gefunden.")
            del self._accounts[account_id]
        logger.info("AccountManager: Konto '{id}' entfernt.", id=account_id)

    # ── Aggregierte Sicht ─────────────────────────────────────────────────────

    def get_total_exposure(self) -> dict[str, Any]:
        """
        Aggregiert offene Positionen ueber alle Konten hinweg.

        Wichtig fuer Korrelationskontrolle ueber Kontogrenzen hinweg –
        damit wird verhindert, dass dasselbe Symbol auf mehreren Konten
        gleichzeitig uebergewichtet wird.

        Returns
        -------
        dict mit:
          positions       – Liste aller offenen Positionen (je mit account_id)
          by_symbol       – Positionen gruppiert nach Symbol
          total_lots      – Gesamt-Lot-Groesse aller offenen Positionen
          accounts_active – Anzahl Konten mit mind. einer offenen Position
        """
        with self._lock:
            accounts = list(self._accounts.values())

        all_positions: list[dict] = []

        for ctx in accounts:
            try:
                positions = ctx.order_executor.get_open_positions()
                for pos in positions:
                    enriched = dict(pos)
                    enriched["account_id"] = ctx.account_id
                    all_positions.append(enriched)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "AccountManager: get_open_positions() fuer Konto '{id}' "
                    "fehlgeschlagen: {exc}",
                    id=ctx.account_id, exc=exc,
                )

        by_symbol: dict[str, list[dict]] = {}
        for pos in all_positions:
            sym = pos.get("symbol", "unknown")
            by_symbol.setdefault(sym, []).append(pos)

        total_lots = sum(p.get("lot_size", 0.0) for p in all_positions)
        accounts_active = len({p["account_id"] for p in all_positions})

        return {
            "positions": all_positions,
            "by_symbol": by_symbol,
            "total_lots": total_lots,
            "accounts_active": accounts_active,
        }

    # ── Orchestrator-Steuerung ────────────────────────────────────────────────

    def run_cycle_for(self, account_id: str, symbol: str) -> dict[str, Any]:
        """
        Fuehrt einen TradingOrchestrator-Zyklus fuer ein bestimmtes Konto aus.

        Raises
        ------
        KeyError   wenn account_id nicht gefunden.
        RuntimeError wenn das Konto keinen Orchestrator hat.
        """
        ctx = self.get_account(account_id)
        if ctx.orchestrator is None:
            raise RuntimeError(
                f"Konto '{account_id}' hat keinen Orchestrator. "
                "Setze AccountContext.orchestrator bevor du run_cycle_for aufrufst."
            )
        return ctx.orchestrator.run_cycle(symbol)

    def run_all_cycles(
        self,
        symbol: str,
        parallel: bool = True,
    ) -> dict[str, Any]:
        """
        Fuehrt einen Orchestrator-Zyklus fuer alle registrierten Konten aus.

        Konten ohne Orchestrator werden uebersprungen.

        Parameters
        ----------
        symbol   : Handelssymbol (z.B. "EURUSD").
        parallel : True = Konten in parallelen Threads (Default),
                   False = sequenziell.

        Returns
        -------
        dict {account_id -> Zyklus-Ergebnis-dict | {"error": str}}
        """
        with self._lock:
            accounts = [ctx for ctx in self._accounts.values() if ctx.orchestrator is not None]

        results: dict[str, Any] = {}

        if not accounts:
            logger.warning("AccountManager: Keine Konten mit Orchestrator gefunden.")
            return results

        if parallel:
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(accounts)) as pool:
                futures = {
                    pool.submit(self.run_cycle_for, ctx.account_id, symbol): ctx.account_id
                    for ctx in accounts
                }
                for future in concurrent.futures.as_completed(futures):
                    acc_id = futures[future]
                    try:
                        results[acc_id] = future.result()
                    except Exception as exc:  # noqa: BLE001
                        logger.error(
                            "AccountManager: Zyklus fuer Konto '{id}' fehlgeschlagen: {exc}",
                            id=acc_id, exc=exc,
                        )
                        results[acc_id] = {"error": str(exc)}
        else:
            for ctx in accounts:
                try:
                    results[ctx.account_id] = self.run_cycle_for(ctx.account_id, symbol)
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "AccountManager: Zyklus fuer Konto '{id}' fehlgeschlagen: {exc}",
                        id=ctx.account_id, exc=exc,
                    )
                    results[ctx.account_id] = {"error": str(exc)}

        return results
