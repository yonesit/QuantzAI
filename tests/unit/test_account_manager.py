"""
tests/unit/test_account_manager.py
Unit-Tests fuer AccountManager.

Abgedeckt:
  - register_account: Konto anlegen, Duplikat-Schutz
  - get_account: Erfolg und KeyError
  - list_account_ids / remove_account
  - Isolierte Risiko-States: mehrere Konten teilen sich keinen State
  - get_total_exposure: aggregiert Positionen korrekt, leerer Fall, Fehler-Resilienz
  - run_cycle_for: delegiert an den richtigen Orchestrator, RuntimeError ohne Orchestrator
  - run_all_cycles parallel und sequenziell: alle Konten werden aufgerufen
  - run_all_cycles: Exception pro Konto fuehrt zu {"error": ...} Eintrag
  - run_all_cycles: Konten ohne Orchestrator werden uebersprungen
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, call, patch

import pytest

from src.account_manager import (
    AccountConfig,
    AccountContext,
    AccountCredentials,
    AccountManager,
    CredentialStore,
    KEYRING_SERVICE_PASS,
    KEYRING_SERVICE_META,
    _KEYRING_INDEX_KEY,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Hilfsfunktionen
# ─────────────────────────────────────────────────────────────────────────────

def _make_config(account_id: str = "demo", symbols=None) -> AccountConfig:
    return AccountConfig(
        account_id=account_id,
        symbols=symbols or ["EURUSD"],
        risk_config={"max_risk_per_trade_pct": 1.0},
    )


def _make_executor(positions=None) -> MagicMock:
    ex = MagicMock()
    ex.get_open_positions.return_value = positions or []
    return ex


def _make_orchestrator(result=None) -> MagicMock:
    orch = MagicMock()
    orch.run_cycle.return_value = result or {
        "symbol": "EURUSD",
        "action": "flat",
        "reason": "signal_flat",
    }
    return orch


def _make_context(
    account_id: str = "demo",
    positions=None,
    orchestrator=None,
    symbols=None,
) -> AccountContext:
    return AccountContext(
        account_id=account_id,
        config=_make_config(account_id, symbols),
        connector=MagicMock(),
        order_executor=_make_executor(positions),
        risk_guard=MagicMock(),
        position_sizer=MagicMock(),
        orchestrator=orchestrator,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  register_account / get_account / list / remove
# ─────────────────────────────────────────────────────────────────────────────

class TestRegistration:
    def test_register_and_retrieve(self):
        mgr = AccountManager()
        ctx = _make_context("demo")
        mgr.register_account(ctx)
        assert mgr.get_account("demo") is ctx

    def test_duplicate_raises(self):
        mgr = AccountManager()
        mgr.register_account(_make_context("demo"))
        with pytest.raises(ValueError, match="bereits registriert"):
            mgr.register_account(_make_context("demo"))

    def test_get_unknown_raises(self):
        mgr = AccountManager()
        with pytest.raises(KeyError, match="nicht gefunden"):
            mgr.get_account("nonexistent")

    def test_list_account_ids(self):
        mgr = AccountManager()
        mgr.register_account(_make_context("demo"))
        mgr.register_account(_make_context("live_1"))
        ids = mgr.list_account_ids()
        assert set(ids) == {"demo", "live_1"}

    def test_remove_account(self):
        mgr = AccountManager()
        mgr.register_account(_make_context("demo"))
        mgr.remove_account("demo")
        assert "demo" not in mgr.list_account_ids()

    def test_remove_nonexistent_raises(self):
        mgr = AccountManager()
        with pytest.raises(KeyError):
            mgr.remove_account("ghost")

    def test_two_accounts_are_independent(self):
        """Konto-Kontexte teilen sich keine Objekte."""
        mgr = AccountManager()
        ctx_a = _make_context("A")
        ctx_b = _make_context("B")
        mgr.register_account(ctx_a)
        mgr.register_account(ctx_b)
        assert mgr.get_account("A").risk_guard is not mgr.get_account("B").risk_guard
        assert mgr.get_account("A").order_executor is not mgr.get_account("B").order_executor


# ─────────────────────────────────────────────────────────────────────────────
#  Isolierte Risiko-States
# ─────────────────────────────────────────────────────────────────────────────

class TestIsolatedRiskStates:
    def test_risk_guard_calls_are_per_account(self):
        """RiskGuard jedes Kontos wird unabhaengig aufgerufen."""
        mgr = AccountManager()
        ctx_a = _make_context("A")
        ctx_b = _make_context("B")
        mgr.register_account(ctx_a)
        mgr.register_account(ctx_b)

        mgr.get_account("A").risk_guard.is_trading_allowed.return_value = True
        mgr.get_account("B").risk_guard.is_trading_allowed.return_value = False

        assert mgr.get_account("A").risk_guard.is_trading_allowed() is True
        assert mgr.get_account("B").risk_guard.is_trading_allowed() is False

    def test_position_sizer_independent(self):
        """PositionSizer jedes Kontos ist eine eigene Instanz."""
        mgr = AccountManager()
        for acc_id in ("alpha", "beta", "gamma"):
            mgr.register_account(_make_context(acc_id))

        sizers = [mgr.get_account(i).position_sizer for i in ("alpha", "beta", "gamma")]
        assert len(set(id(s) for s in sizers)) == 3


# ─────────────────────────────────────────────────────────────────────────────
#  get_total_exposure
# ─────────────────────────────────────────────────────────────────────────────

class TestGetTotalExposure:
    def test_empty_manager_returns_zeros(self):
        mgr = AccountManager()
        result = mgr.get_total_exposure()
        assert result["positions"] == []
        assert result["by_symbol"] == {}
        assert result["total_lots"] == 0.0
        assert result["accounts_active"] == 0

    def test_single_account_single_position(self):
        pos = {"symbol": "EURUSD", "lot_size": 0.1, "direction": "buy", "status": "open"}
        mgr = AccountManager()
        mgr.register_account(_make_context("demo", positions=[pos]))

        result = mgr.get_total_exposure()
        assert len(result["positions"]) == 1
        assert result["positions"][0]["account_id"] == "demo"
        assert result["positions"][0]["symbol"] == "EURUSD"
        assert result["total_lots"] == pytest.approx(0.1)
        assert result["accounts_active"] == 1
        assert "EURUSD" in result["by_symbol"]

    def test_two_accounts_aggregated_correctly(self):
        pos_a = {"symbol": "EURUSD", "lot_size": 0.2, "direction": "buy", "status": "open"}
        pos_b = {"symbol": "GBPUSD", "lot_size": 0.5, "direction": "sell", "status": "open"}
        mgr = AccountManager()
        mgr.register_account(_make_context("A", positions=[pos_a]))
        mgr.register_account(_make_context("B", positions=[pos_b]))

        result = mgr.get_total_exposure()
        assert len(result["positions"]) == 2
        assert result["total_lots"] == pytest.approx(0.7)
        assert result["accounts_active"] == 2
        assert set(result["by_symbol"].keys()) == {"EURUSD", "GBPUSD"}

    def test_same_symbol_on_two_accounts_grouped(self):
        pos_a = {"symbol": "EURUSD", "lot_size": 0.1, "direction": "buy", "status": "open"}
        pos_b = {"symbol": "EURUSD", "lot_size": 0.2, "direction": "sell", "status": "open"}
        mgr = AccountManager()
        mgr.register_account(_make_context("A", positions=[pos_a]))
        mgr.register_account(_make_context("B", positions=[pos_b]))

        result = mgr.get_total_exposure()
        assert len(result["by_symbol"]["EURUSD"]) == 2
        assert result["total_lots"] == pytest.approx(0.3)

    def test_account_id_injected_into_each_position(self):
        pos = {"symbol": "USDJPY", "lot_size": 0.3, "status": "open"}
        mgr = AccountManager()
        mgr.register_account(_make_context("live_1", positions=[pos]))

        result = mgr.get_total_exposure()
        assert result["positions"][0]["account_id"] == "live_1"

    def test_executor_error_does_not_break_aggregation(self):
        """Wenn ein Konto get_open_positions() wirft, werden andere Konten trotzdem aggregiert."""
        broken_executor = MagicMock()
        broken_executor.get_open_positions.side_effect = RuntimeError("MT5 down")

        good_pos = {"symbol": "EURUSD", "lot_size": 0.1, "status": "open"}
        good_executor = _make_executor([good_pos])

        mgr = AccountManager()
        ctx_broken = AccountContext(
            account_id="broken",
            config=_make_config("broken"),
            connector=MagicMock(),
            order_executor=broken_executor,
            risk_guard=MagicMock(),
            position_sizer=MagicMock(),
        )
        ctx_good = AccountContext(
            account_id="good",
            config=_make_config("good"),
            connector=MagicMock(),
            order_executor=good_executor,
            risk_guard=MagicMock(),
            position_sizer=MagicMock(),
        )
        mgr.register_account(ctx_broken)
        mgr.register_account(ctx_good)

        result = mgr.get_total_exposure()
        assert len(result["positions"]) == 1
        assert result["positions"][0]["account_id"] == "good"


# ─────────────────────────────────────────────────────────────────────────────
#  run_cycle_for
# ─────────────────────────────────────────────────────────────────────────────

class TestRunCycleFor:
    def test_delegates_to_correct_orchestrator(self):
        orch_a = _make_orchestrator({"action": "open_buy"})
        orch_b = _make_orchestrator({"action": "flat"})
        mgr = AccountManager()
        mgr.register_account(_make_context("A", orchestrator=orch_a))
        mgr.register_account(_make_context("B", orchestrator=orch_b))

        result = mgr.run_cycle_for("A", "EURUSD")
        assert result["action"] == "open_buy"
        orch_a.run_cycle.assert_called_once_with("EURUSD")
        orch_b.run_cycle.assert_not_called()

    def test_raises_key_error_for_unknown_account(self):
        mgr = AccountManager()
        with pytest.raises(KeyError):
            mgr.run_cycle_for("ghost", "EURUSD")

    def test_raises_runtime_error_without_orchestrator(self):
        mgr = AccountManager()
        mgr.register_account(_make_context("demo", orchestrator=None))
        with pytest.raises(RuntimeError, match="keinen Orchestrator"):
            mgr.run_cycle_for("demo", "EURUSD")


# ─────────────────────────────────────────────────────────────────────────────
#  run_all_cycles
# ─────────────────────────────────────────────────────────────────────────────

class TestRunAllCycles:
    def test_parallel_calls_all_orchestrators(self):
        orch_a = _make_orchestrator({"action": "open_buy"})
        orch_b = _make_orchestrator({"action": "flat"})
        mgr = AccountManager()
        mgr.register_account(_make_context("A", orchestrator=orch_a))
        mgr.register_account(_make_context("B", orchestrator=orch_b))

        results = mgr.run_all_cycles("EURUSD", parallel=True)
        assert set(results.keys()) == {"A", "B"}
        assert results["A"]["action"] == "open_buy"
        assert results["B"]["action"] == "flat"

    def test_sequential_calls_all_orchestrators(self):
        orch_a = _make_orchestrator({"action": "skipped"})
        orch_b = _make_orchestrator({"action": "open_sell"})
        mgr = AccountManager()
        mgr.register_account(_make_context("X", orchestrator=orch_a))
        mgr.register_account(_make_context("Y", orchestrator=orch_b))

        results = mgr.run_all_cycles("GBPUSD", parallel=False)
        assert set(results.keys()) == {"X", "Y"}
        orch_a.run_cycle.assert_called_once_with("GBPUSD")
        orch_b.run_cycle.assert_called_once_with("GBPUSD")

    def test_exception_captured_per_account(self):
        orch_ok = _make_orchestrator({"action": "flat"})
        orch_bad = MagicMock()
        orch_bad.run_cycle.side_effect = RuntimeError("broker error")

        mgr = AccountManager()
        mgr.register_account(_make_context("ok", orchestrator=orch_ok))
        mgr.register_account(_make_context("bad", orchestrator=orch_bad))

        results = mgr.run_all_cycles("EURUSD", parallel=False)
        assert results["ok"]["action"] == "flat"
        assert "error" in results["bad"]
        assert "broker error" in results["bad"]["error"]

    def test_parallel_exception_captured(self):
        orch_bad = MagicMock()
        orch_bad.run_cycle.side_effect = ValueError("timeout")

        mgr = AccountManager()
        mgr.register_account(_make_context("err_acct", orchestrator=orch_bad))

        results = mgr.run_all_cycles("EURUSD", parallel=True)
        assert "error" in results["err_acct"]
        assert "timeout" in results["err_acct"]["error"]

    def test_accounts_without_orchestrator_skipped(self):
        orch = _make_orchestrator({"action": "flat"})
        mgr = AccountManager()
        mgr.register_account(_make_context("with_orch", orchestrator=orch))
        mgr.register_account(_make_context("no_orch", orchestrator=None))

        results = mgr.run_all_cycles("EURUSD", parallel=False)
        assert "with_orch" in results
        assert "no_orch" not in results

    def test_empty_manager_returns_empty_dict(self):
        mgr = AccountManager()
        results = mgr.run_all_cycles("EURUSD")
        assert results == {}

    def test_three_accounts_parallel_independent(self):
        """Drei Konten parallel: jeder Orchestrator bekommt genau einen Aufruf."""
        orchs = {i: _make_orchestrator({"action": f"result_{i}"}) for i in range(3)}
        mgr = AccountManager()
        for i, orch in orchs.items():
            mgr.register_account(_make_context(f"acc_{i}", orchestrator=orch))

        results = mgr.run_all_cycles("USDJPY", parallel=True)
        assert len(results) == 3
        for i, orch in orchs.items():
            orch.run_cycle.assert_called_once_with("USDJPY")
            assert results[f"acc_{i}"]["action"] == f"result_{i}"

    def test_run_all_cycles_parallel_is_thread_safe(self):
        """run_all_cycles parallel: kein Race-Condition bei viele gleichzeitigen Laeufen."""
        orch = _make_orchestrator({"action": "flat"})
        mgr = AccountManager()
        for i in range(5):
            mgr.register_account(_make_context(f"acc_{i}", orchestrator=orch))

        errors = []

        def _run():
            try:
                mgr.run_all_cycles("EURUSD", parallel=True)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_run) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread-Fehler: {errors}"


# ─────────────────────────────────────────────────────────────────────────────
#  CredentialStore – Hilfsfunktion fuer Injectable Keyring
# ─────────────────────────────────────────────────────────────────────────────

def _make_store() -> tuple[CredentialStore, dict]:
    """Erstellt einen CredentialStore mit In-Memory-Keyring-Mock."""
    storage: dict[tuple[str, str], str] = {}

    def _set(service: str, username: str, password: str) -> None:
        storage[(service, username)] = password

    def _get(service: str, username: str) -> str | None:
        return storage.get((service, username))

    def _delete(service: str, username: str) -> None:
        storage.pop((service, username), None)

    store = CredentialStore(_set_fn=_set, _get_fn=_get, _delete_fn=_delete)
    return store, storage


# ─────────────────────────────────────────────────────────────────────────────
#  TestCredentialStore
# ─────────────────────────────────────────────────────────────────────────────

class TestCredentialStore:

    def test_store_and_load_password(self):
        store, _ = _make_store()
        store.store_password("demo", "secret")
        assert store.load_password("demo") == "secret"

    def test_load_password_returns_none_when_not_stored(self):
        store, _ = _make_store()
        assert store.load_password("nonexistent") is None

    def test_delete_password_removes_entry(self):
        store, _ = _make_store()
        store.store_password("demo", "secret")
        store.delete_password("demo")
        assert store.load_password("demo") is None

    def test_delete_password_nonexistent_no_error(self):
        store, _ = _make_store()
        store.delete_password("ghost")  # should not raise

    def test_has_password_true_when_stored(self):
        store, _ = _make_store()
        store.store_password("demo", "pw")
        assert store.has_password("demo") is True

    def test_has_password_false_when_not_stored(self):
        store, _ = _make_store()
        assert store.has_password("demo") is False

    def test_store_credentials_saves_meta(self):
        store, storage = _make_store()
        creds = AccountCredentials("acc1", login="12345", server="Demo", broker="MT5")
        store.store_credentials(creds)
        assert (KEYRING_SERVICE_META, "acc1") in storage

    def test_load_credentials_returns_none_when_not_stored(self):
        store, _ = _make_store()
        assert store.load_credentials("nonexistent") is None

    def test_load_credentials_returns_correct_data(self):
        store, _ = _make_store()
        creds = AccountCredentials("acc1", login="12345", server="Demo-Server", broker="MT5")
        store.store_credentials(creds)
        loaded = store.load_credentials("acc1")
        assert loaded is not None
        assert loaded.account_id == "acc1"
        assert loaded.login == "12345"
        assert loaded.server == "Demo-Server"
        assert loaded.broker == "MT5"

    def test_credentials_is_live_false_by_default(self):
        store, _ = _make_store()
        creds = AccountCredentials("acc1", login="", server="", broker="MT5")
        store.store_credentials(creds)
        loaded = store.load_credentials("acc1")
        assert loaded.is_live is False

    def test_credentials_is_live_true(self):
        store, _ = _make_store()
        creds = AccountCredentials("live1", login="", server="", broker="MT5", is_live=True)
        store.store_credentials(creds)
        loaded = store.load_credentials("live1")
        assert loaded.is_live is True

    def test_delete_credentials_removes_meta_and_password(self):
        store, storage = _make_store()
        creds = AccountCredentials("acc1", login="1", server="s", broker="MT5")
        store.store_credentials(creds)
        store.store_password("acc1", "pw")
        store.delete_credentials("acc1")
        assert store.load_credentials("acc1") is None
        assert store.load_password("acc1") is None

    def test_list_account_ids_empty_initially(self):
        store, _ = _make_store()
        assert store.list_account_ids() == []

    def test_list_account_ids_after_store(self):
        store, _ = _make_store()
        store.store_credentials(AccountCredentials("a1", "", "", "MT5"))
        store.store_credentials(AccountCredentials("a2", "", "", "MT5"))
        ids = store.list_account_ids()
        assert "a1" in ids
        assert "a2" in ids

    def test_list_account_ids_after_delete(self):
        store, _ = _make_store()
        store.store_credentials(AccountCredentials("a1", "", "", "MT5"))
        store.store_credentials(AccountCredentials("a2", "", "", "MT5"))
        store.delete_credentials("a1")
        ids = store.list_account_ids()
        assert "a1" not in ids
        assert "a2" in ids

    def test_store_credentials_updates_index_once(self):
        store, _ = _make_store()
        store.store_credentials(AccountCredentials("a1", "", "", "MT5"))
        store.store_credentials(AccountCredentials("a1", "", "", "MT5"))  # duplicate
        ids = store.list_account_ids()
        assert ids.count("a1") == 1

    def test_import_from_env_success(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("MT5_LOGIN=99999\nMT5_PASSWORD=mypass\nMT5_SERVER=Demo-Server\n")
        store, storage = _make_store()
        imported = store.import_from_env(str(env_file))
        assert imported == ["env_import"]
        loaded = store.load_credentials("env_import")
        assert loaded is not None
        assert loaded.login == "99999"
        assert loaded.server == "Demo-Server"
        assert loaded.is_live is False

    def test_import_from_env_stores_password_in_keyring(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("MT5_LOGIN=99999\nMT5_PASSWORD=mypass\n")
        store, _ = _make_store()
        store.import_from_env(str(env_file))
        assert store.load_password("env_import") == "mypass"

    def test_import_from_env_no_login_returns_empty(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("MT5_SERVER=Demo\n")
        store, _ = _make_store()
        imported = store.import_from_env(str(env_file))
        assert imported == []

    def test_import_from_env_no_password_no_keyring_entry(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("MT5_LOGIN=12345\n")
        store, _ = _make_store()
        store.import_from_env(str(env_file))
        assert store.load_password("env_import") is None

    def test_import_from_env_mt5_account_fallback(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("MT5_ACCOUNT=54321\n")
        store, _ = _make_store()
        imported = store.import_from_env(str(env_file))
        assert imported == ["env_import"]
        assert store.load_credentials("env_import").login == "54321"

    def test_load_credentials_handles_corrupt_json(self):
        storage: dict = {}

        def _set(s, u, p):
            storage[(s, u)] = p

        def _get(s, u):
            val = storage.get((s, u))
            if u == "bad_acc":
                return "NOT_JSON{{"
            return val

        def _delete(s, u):
            storage.pop((s, u), None)

        store = CredentialStore(_set_fn=_set, _get_fn=_get, _delete_fn=_delete)
        result = store.load_credentials("bad_acc")
        assert result is None


# ─────────────────────────────────────────────────────────────────────────────
#  AccountManager – Aktives Konto
# ─────────────────────────────────────────────────────────────────────────────

class TestAccountManagerActive:

    def test_active_account_id_initially_none(self):
        mgr = AccountManager()
        assert mgr.active_account_id is None

    def test_active_account_initially_none(self):
        mgr = AccountManager()
        assert mgr.active_account is None

    def test_set_active_account_success(self):
        mgr = AccountManager()
        mgr.register_account(_make_context("demo"))
        mgr.set_active_account("demo")
        assert mgr.active_account_id == "demo"

    def test_active_account_returns_context(self):
        mgr = AccountManager()
        ctx = _make_context("demo")
        mgr.register_account(ctx)
        mgr.set_active_account("demo")
        assert mgr.active_account is ctx

    def test_set_active_account_unknown_raises_key_error(self):
        mgr = AccountManager()
        with pytest.raises(KeyError):
            mgr.set_active_account("nonexistent")

    def test_active_account_clears_on_remove(self):
        mgr = AccountManager()
        mgr.register_account(_make_context("demo"))
        mgr.set_active_account("demo")
        mgr.remove_account("demo")
        assert mgr.active_account_id is None

    def test_set_active_account_can_switch(self):
        mgr = AccountManager()
        mgr.register_account(_make_context("a"))
        mgr.register_account(_make_context("b"))
        mgr.set_active_account("a")
        mgr.set_active_account("b")
        assert mgr.active_account_id == "b"


# ─────────────────────────────────────────────────────────────────────────────
#  AccountManager – switch_account
# ─────────────────────────────────────────────────────────────────────────────

class TestAccountManagerSwitch:

    def test_switch_changes_active_account(self):
        mgr = AccountManager()
        mgr.register_account(_make_context("a"))
        mgr.register_account(_make_context("b"))
        mgr.set_active_account("a")
        mgr.switch_account("b")
        assert mgr.active_account_id == "b"

    def test_switch_calls_disconnect_on_old(self):
        mgr = AccountManager()
        ctx_a = _make_context("a")
        mgr.register_account(ctx_a)
        mgr.register_account(_make_context("b"))
        mgr.set_active_account("a")

        disconnected: list[str] = []
        mgr.switch_account("b", disconnect_fn=lambda c: disconnected.append(c.account_id))
        assert "a" in disconnected

    def test_switch_calls_connect_on_new(self):
        mgr = AccountManager()
        mgr.register_account(_make_context("a"))
        ctx_b = _make_context("b")
        mgr.register_account(ctx_b)
        mgr.set_active_account("a")

        connected: list[str] = []
        mgr.switch_account("b", connect_fn=lambda c: connected.append(c.account_id))
        assert "b" in connected

    def test_switch_disconnect_error_does_not_block(self):
        mgr = AccountManager()
        mgr.register_account(_make_context("a"))
        mgr.register_account(_make_context("b"))
        mgr.set_active_account("a")

        def bad_disconnect(ctx):
            raise RuntimeError("network error")

        mgr.switch_account("b", disconnect_fn=bad_disconnect)
        assert mgr.active_account_id == "b"

    def test_switch_connect_error_does_not_block(self):
        mgr = AccountManager()
        mgr.register_account(_make_context("a"))
        mgr.register_account(_make_context("b"))
        mgr.set_active_account("a")

        def bad_connect(ctx):
            raise RuntimeError("connection timeout")

        mgr.switch_account("b", connect_fn=bad_connect)
        assert mgr.active_account_id == "b"

    def test_switch_no_callbacks_works(self):
        mgr = AccountManager()
        mgr.register_account(_make_context("a"))
        mgr.register_account(_make_context("b"))
        mgr.set_active_account("a")
        mgr.switch_account("b")
        assert mgr.active_account_id == "b"

    def test_switch_from_none_active_skips_disconnect(self):
        mgr = AccountManager()
        mgr.register_account(_make_context("b"))

        disconnected: list[str] = []
        mgr.switch_account("b", disconnect_fn=lambda c: disconnected.append(c.account_id))
        assert disconnected == []
        assert mgr.active_account_id == "b"

    def test_switch_unknown_account_raises(self):
        mgr = AccountManager()
        with pytest.raises(KeyError):
            mgr.switch_account("ghost")
