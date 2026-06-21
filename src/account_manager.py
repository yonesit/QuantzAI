"""
src/account_manager.py
AccountManager – verwaltet mehrere MT5/OANDA-Konten parallel.

Jedes Konto hat:
  - eine eindeutige ID
  - eigene Komponenten (Connector, OrderExecutor, RiskGuard, PositionSizer)
  - eigenen isolierten Risiko-State
  - optional einen TradingOrchestrator

Konten werden per register_account() hinzugefuegt.
Konfiguration in config.yaml unter accounts: (Login-Daten sicher im keyring).
"""

from __future__ import annotations

import concurrent.futures
import json
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from loguru import logger

try:
    import keyring as _keyring_lib
except ImportError:  # pragma: no cover
    _keyring_lib = None  # type: ignore[assignment]


# ─────────────────────────────────────────────────────────────────────────────
#  Keyring-Konstanten
# ─────────────────────────────────────────────────────────────────────────────

KEYRING_SERVICE_PASS = "QuantzAI"
KEYRING_SERVICE_META = "QuantzAI_meta"
_KEYRING_INDEX_KEY   = "__index__"


# ─────────────────────────────────────────────────────────────────────────────
#  Default-Keyring-Funktionen (injizierbar)
# ─────────────────────────────────────────────────────────────────────────────

def _default_keyring_set(service: str, username: str, password: str) -> None:
    if _keyring_lib is None:  # pragma: no cover
        raise RuntimeError("keyring nicht installiert. pip install keyring")
    _keyring_lib.set_password(service, username, password)


def _default_keyring_get(service: str, username: str) -> Optional[str]:
    if _keyring_lib is None:  # pragma: no cover
        return None
    return _keyring_lib.get_password(service, username)


def _default_keyring_delete(service: str, username: str) -> None:
    if _keyring_lib is None:  # pragma: no cover
        return
    try:
        _keyring_lib.delete_password(service, username)
    except Exception:  # noqa: BLE001
        pass


# ─────────────────────────────────────────────────────────────────────────────
#  AccountCredentials
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AccountCredentials:
    """
    Login-Daten fuer ein Konto (Passwort NICHT enthalten – nur via keyring).

    Parameters
    ----------
    account_id : Eindeutige Konto-ID.
    login      : MT5-Login-Nummer oder Benutzername.
    server     : MT5-Server-Adresse.
    broker     : Broker-Name (z.B. "MT5", "OANDA").
    is_live    : True = echtes Kapital, False = Demo-Konto.
    """
    account_id: str
    login:      str
    server:     str
    broker:     str
    is_live:    bool = False


# ─────────────────────────────────────────────────────────────────────────────
#  CredentialStore
# ─────────────────────────────────────────────────────────────────────────────

class CredentialStore:
    """
    Keyring-basierter Credential-Store fuer Trading-Konten.

    Passwoerter werden AUSSCHLIESSLICH via keyring gespeichert (Windows
    Credential Manager unter Windows). Konto-Metadaten (Login, Server,
    Broker, is_live) werden ebenfalls via keyring als JSON gesichert –
    niemals in Klartext-Dateien.

    .env-Datei wird nur als Fallback fuer Entwicklungsumgebungen unterstuetzt
    (import_from_env).

    Parameters
    ----------
    _set_fn    : (service, username, password) -> None
    _get_fn    : (service, username) -> Optional[str]
    _delete_fn : (service, username) -> None
    """

    def __init__(
        self,
        _set_fn:    Optional[Callable[[str, str, str], None]]     = None,
        _get_fn:    Optional[Callable[[str, str], Optional[str]]] = None,
        _delete_fn: Optional[Callable[[str, str], None]]          = None,
    ) -> None:
        self._set    = _set_fn    or _default_keyring_set
        self._get    = _get_fn    or _default_keyring_get
        self._delete = _delete_fn or _default_keyring_delete

    # ── Passwort-Operationen ──────────────────────────────────────────────────

    def store_password(self, account_id: str, password: str) -> None:
        """Speichert Passwort sicher via keyring. Ueberschreibt vorhandenes."""
        self._set(KEYRING_SERVICE_PASS, account_id, password)
        logger.info("CredentialStore: Passwort fuer '{id}' gesetzt.", id=account_id)

    def load_password(self, account_id: str) -> Optional[str]:
        """Laedt Passwort aus keyring. Gibt None zurueck wenn nicht vorhanden."""
        return self._get(KEYRING_SERVICE_PASS, account_id)

    def delete_password(self, account_id: str) -> None:
        """Entfernt Passwort aus keyring (kein Fehler wenn nicht vorhanden)."""
        self._delete(KEYRING_SERVICE_PASS, account_id)

    def has_password(self, account_id: str) -> bool:
        """Gibt True zurueck wenn ein Passwort fuer account_id gespeichert ist."""
        return self.load_password(account_id) is not None

    # ── Konto-Metadaten ───────────────────────────────────────────────────────

    def store_credentials(self, creds: AccountCredentials) -> None:
        """
        Speichert Konto-Metadaten (OHNE Passwort) via keyring als JSON.
        Aktualisiert den Konto-Index automatisch.
        """
        meta = {
            "login":   creds.login,
            "server":  creds.server,
            "broker":  creds.broker,
            "is_live": creds.is_live,
        }
        self._set(KEYRING_SERVICE_META, creds.account_id, json.dumps(meta))

        index = self._load_index()
        if creds.account_id not in index:
            index.append(creds.account_id)
            self._set(KEYRING_SERVICE_META, _KEYRING_INDEX_KEY, json.dumps(index))

        logger.info(
            "CredentialStore: Konto '{id}' gespeichert (is_live={live}).",
            id=creds.account_id, live=creds.is_live,
        )

    def load_credentials(self, account_id: str) -> Optional[AccountCredentials]:
        """Laedt Konto-Metadaten aus keyring. Gibt None zurueck wenn nicht vorhanden."""
        raw = self._get(KEYRING_SERVICE_META, account_id)
        if raw is None:
            return None
        try:
            meta = json.loads(raw)
        except Exception:  # noqa: BLE001
            return None
        return AccountCredentials(
            account_id=account_id,
            login=meta.get("login", ""),
            server=meta.get("server", ""),
            broker=meta.get("broker", ""),
            is_live=bool(meta.get("is_live", False)),
        )

    def delete_credentials(self, account_id: str) -> None:
        """Entfernt Konto-Metadaten und Passwort aus keyring."""
        self._delete(KEYRING_SERVICE_META, account_id)
        self.delete_password(account_id)

        index = self._load_index()
        if account_id in index:
            index.remove(account_id)
            self._set(KEYRING_SERVICE_META, _KEYRING_INDEX_KEY, json.dumps(index))

        logger.info("CredentialStore: Konto '{id}' entfernt.", id=account_id)

    def list_account_ids(self) -> list[str]:
        """Gibt alle gespeicherten Konto-IDs zurueck."""
        return self._load_index()

    def _load_index(self) -> list[str]:
        raw = self._get(KEYRING_SERVICE_META, _KEYRING_INDEX_KEY)
        if raw is None:
            return []
        try:
            return list(json.loads(raw))
        except Exception:  # noqa: BLE001
            return []

    # ── .env-Migration ────────────────────────────────────────────────────────

    def import_from_env(self, env_path: Optional[str] = None) -> list[str]:
        """
        Importiert bestehende .env-Zugangsdaten einmalig in keyring.

        Liest:
          MT5_LOGIN / MT5_ACCOUNT – Login-Nummer
          MT5_PASSWORD            – Passwort (nur via keyring gespeichert)
          MT5_SERVER              – Server-Adresse
          MT5_BROKER              – Broker-Name (Standard: "MT5")

        Angelegtes Konto-ID: "env_import"

        Returns
        -------
        Liste der importierten Konto-IDs (leer wenn keine Daten gefunden).
        """
        from dotenv import dotenv_values  # late import

        path   = env_path or ".env"
        values = dotenv_values(path)

        login    = str(values.get("MT5_LOGIN", "") or values.get("MT5_ACCOUNT", "")).strip()
        password = str(values.get("MT5_PASSWORD", "")).strip()
        server   = str(values.get("MT5_SERVER", "")).strip()
        broker   = str(values.get("MT5_BROKER", "MT5")).strip() or "MT5"

        if not login:
            logger.info("CredentialStore: Keine MT5_LOGIN in '{p}' gefunden.", p=path)
            return []

        account_id = "env_import"
        creds = AccountCredentials(
            account_id=account_id,
            login=login,
            server=server,
            broker=broker,
            is_live=False,
        )
        self.store_credentials(creds)
        if password:
            self.store_password(account_id, password)

        logger.info(
            "CredentialStore: .env-Daten fuer Konto '{id}' importiert.",
            id=account_id,
        )
        return [account_id]


# ─────────────────────────────────────────────────────────────────────────────
#  AccountConfig / AccountContext
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
#  AccountManager
# ─────────────────────────────────────────────────────────────────────────────

class AccountManager:
    """
    Verwaltet mehrere Trading-Konten parallel.

    Jedes Konto hat vollstaendig isolierte Komponenten – kein gemeinsamer
    State zwischen Konten. Das ermoeglicht z.B. ein Demo- und ein Live-Konto
    oder mehrere Strategien auf getrennten Konten.

    Konten werden ueber register_account() hinzugefuegt.
    Das aktive Konto wird ueber set_active_account() / switch_account() gesetzt.
    """

    def __init__(self) -> None:
        self._accounts: dict[str, AccountContext] = {}
        self._lock = threading.Lock()
        self._active_account_id: Optional[str] = None

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
            if self._active_account_id == account_id:
                self._active_account_id = None
        logger.info("AccountManager: Konto '{id}' entfernt.", id=account_id)

    # ── Aktives Konto ─────────────────────────────────────────────────────────

    @property
    def active_account_id(self) -> Optional[str]:
        """Gibt die ID des aktuell aktiven Kontos zurueck (oder None)."""
        with self._lock:
            return self._active_account_id

    @property
    def active_account(self) -> Optional[AccountContext]:
        """Gibt den AccountContext des aktiven Kontos zurueck (oder None)."""
        with self._lock:
            if self._active_account_id is None:
                return None
            return self._accounts.get(self._active_account_id)

    def set_active_account(self, account_id: str) -> None:
        """
        Setzt das aktive Konto.

        Raises
        ------
        KeyError wenn account_id nicht registriert ist.
        """
        with self._lock:
            if account_id not in self._accounts:
                raise KeyError(
                    f"Konto '{account_id}' nicht gefunden. "
                    f"Registriert: {list(self._accounts.keys())}"
                )
            self._active_account_id = account_id
        logger.info("AccountManager: Aktives Konto -> '{id}'.", id=account_id)

    def switch_account(
        self,
        account_id: str,
        disconnect_fn: Optional[Callable[[AccountContext], None]] = None,
        connect_fn:    Optional[Callable[[AccountContext], None]] = None,
    ) -> None:
        """
        Wechselt das aktive Konto sauber.

        1. Trennt das aktuell aktive Konto via disconnect_fn(old_ctx).
        2. Setzt das neue aktive Konto.
        3. Verbindet das neue Konto via connect_fn(new_ctx).

        Fehler in disconnect/connect werden geloggt aber nicht weitergegeben.

        Parameters
        ----------
        account_id    : Ziel-Konto-ID (muss registriert sein).
        disconnect_fn : Optional – aufgerufen mit altem AccountContext.
        connect_fn    : Optional – aufgerufen mit neuem AccountContext.
        """
        old_ctx = self.active_account
        if old_ctx is not None and disconnect_fn is not None:
            try:
                disconnect_fn(old_ctx)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "AccountManager: Disconnect fuer '{id}' fehlgeschlagen: {e}",
                    id=old_ctx.account_id, e=exc,
                )

        self.set_active_account(account_id)

        new_ctx = self.active_account
        if new_ctx is not None and connect_fn is not None:
            try:
                connect_fn(new_ctx)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "AccountManager: Connect fuer '{id}' fehlgeschlagen: {e}",
                    id=account_id, e=exc,
                )

        logger.info(
            "AccountManager: Kontowechsel -> '{id}' abgeschlossen.", id=account_id
        )

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
