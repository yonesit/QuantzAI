"""
tests/unit/test_run_gui_bot.py
Unit-Tests fuer scripts/run_gui_bot.py (Startup-Logik, keine echte MT5-Verbindung).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# Projekt-Root in sys.path
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from scripts.run_gui_bot import (
    StartupError,
    _load_config,
    _load_env,
    find_newest_model,
    find_newest_mr_model,
    build_mt5_connector,
    _OandaStub,
    build_trading_stack,
    build_portfolio_stack,
    MultiSymbolOrchestrator,
    _LiveDashboardBackend,
    calc_unrealized_pnl,
)
from src.models.regime_detector import RegimeDetector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_config(tmp_path) -> Path:
    """Minimale config.yaml fuer Tests."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "risk:\n"
        "  daily_loss_limit_pct: 5.0\n"
        "  max_drawdown_pct: 15.0\n"
        "  max_risk_per_trade_pct: 1.0\n"
        "  spread_filter_pips: 3.0\n"
        "model:\n"
        "  confidence_threshold: 0.55\n"
        "features:\n"
        "  ema_periods: [9, 20, 50, 200]\n"
        "  sma_periods: [50]\n"
        "  rsi_periods: [14]\n"
        "  atr_period: 14\n"
        "  bollinger_period: 20\n"
        "  bollinger_std: 2\n"
        "  include_time_features: true\n"
        "  warmup_candles: 200\n"
        "  include_sentiment: false\n"
        "  mtf_adx_threshold: 25.0\n"
        "  mtf_flip_lookback: 3\n"
        "broker:\n"
        "  max_price_discrepancy_pips: 5\n",
        encoding="utf-8",
    )
    return cfg


@pytest.fixture
def tmp_model(tmp_path) -> Path:
    """Erzeugt eine echte (aber minimale) joblib-Modelldatei."""
    import joblib
    import lightgbm as lgb
    import numpy as np

    mdl = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=2,
        random_state=42,
        verbose=-1,
    )
    X = np.random.rand(30, 5)
    y = np.tile([0, 1, 2], 10)
    mdl.fit(X, y)

    payload = {
        "model": mdl,
        "feature_names": [f"f{i}" for i in range(5)],
        "params": {"objective": "multiclass", "num_class": 3},
    }
    p = tmp_path / "signal_model_v1_20260101.joblib"
    import joblib
    joblib.dump(payload, p)
    return p


@pytest.fixture
def tmp_mr_model(tmp_path, tmp_model) -> Path:
    """MR-Modell – gleiche joblib-Struktur wie SignalModel, andere Dateiname."""
    import shutil
    mr_path = tmp_path / "mean_reversion_model_20260101.joblib"
    shutil.copy(tmp_model, mr_path)
    return mr_path


@pytest.fixture
def minimal_connector():
    """Gefakter MT5Connector der schon 'verbunden' ist."""
    c = MagicMock()
    c.is_connected = True
    c.get_account_info.return_value = {
        "login": 12345678,
        "balance": 10_000.0,
        "currency": "EUR",
        "is_demo": True,
    }
    c.get_symbol_info.return_value = {
        "spread": 3,
        "point": 0.00001,
    }
    return c


# ---------------------------------------------------------------------------
# _load_config
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def test_loads_valid_yaml(self, tmp_config):
        cfg = _load_config(tmp_config)
        assert cfg["risk"]["daily_loss_limit_pct"] == 5.0
        assert cfg["model"]["confidence_threshold"] == 0.55

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(StartupError, match="nicht gefunden"):
            _load_config(tmp_path / "does_not_exist.yaml")


# ---------------------------------------------------------------------------
# _load_env
# ---------------------------------------------------------------------------

class TestLoadEnv:
    def test_loads_key_value(self, tmp_path, monkeypatch):
        env_file = tmp_path / ".env"
        env_file.write_text("TEST_KEY_XYZ=hello_world\n", encoding="utf-8")
        monkeypatch.delenv("TEST_KEY_XYZ", raising=False)

        _load_env(str(env_file))
        assert os.environ.get("TEST_KEY_XYZ") == "hello_world"

    def test_does_not_overwrite_existing(self, tmp_path, monkeypatch):
        env_file = tmp_path / ".env"
        env_file.write_text("TEST_KEY_XYZ=from_file\n", encoding="utf-8")
        monkeypatch.setenv("TEST_KEY_XYZ", "already_set")

        _load_env(str(env_file))
        assert os.environ["TEST_KEY_XYZ"] == "already_set"

    def test_missing_file_is_silently_ignored(self, tmp_path):
        # kein Fehler wenn .env nicht existiert
        _load_env(str(tmp_path / ".env"))


class TestLiveDashboardBackend:
    def test_fetch_snapshot_preserves_none_for_missing_totals(self, tmp_path):
        connector = MagicMock()
        connector.get_account_info.return_value = {
            "balance": 10_000.0,
            "currency": "EUR",
            "equity": 10_000.0,
            "login": 123,
            "server": "Demo",
            "leverage": 100,
            "is_demo": True,
        }

        executor = MagicMock()
        executor.get_open_positions.return_value = []
        executor._live = True
        executor._paper_path = tmp_path / "paper_trades.json"

        backend = _LiveDashboardBackend(connector, executor)
        snap = backend.fetch_snapshot()

        assert snap.total_gross_profit is None
        assert snap.total_gross_loss is None

    def test_fetch_snapshot_returns_real_totals_from_closed_paper_trades(self, tmp_path):
        connector = MagicMock()
        connector.get_account_info.return_value = {
            "balance": 10_000.0,
            "currency": "EUR",
            "equity": 10_000.0,
            "login": 123,
            "server": "Demo",
            "leverage": 100,
            "is_demo": True,
        }

        executor = MagicMock()
        executor.get_open_positions.return_value = []
        executor._live = True
        paper_path = tmp_path / "paper_trades.json"
        paper_path.write_text(
            json.dumps([
                {"status": "closed", "pnl": 200.0},
                {"status": "closed", "pnl": -80.0},
            ]),
            encoding="utf-8",
        )
        executor._paper_path = paper_path

        backend = _LiveDashboardBackend(connector, executor)
        snap = backend.fetch_snapshot()

        assert snap.total_gross_profit == 200.0
        assert snap.total_gross_loss == -80.0


# ---------------------------------------------------------------------------
# find_newest_model
# ---------------------------------------------------------------------------

class TestFindNewestModel:
    def test_returns_newest_by_mtime(self, tmp_path):
        import time
        m1 = tmp_path / "signal_model_v1_20260101.joblib"
        m1.write_text("x")
        time.sleep(0.01)
        m2 = tmp_path / "signal_model_v1_20260201.joblib"
        m2.write_text("x")

        result = find_newest_model(tmp_path)
        assert result == m2

    def test_ignores_IS_models(self, tmp_path):
        is_model = tmp_path / "signal_model_v1_IS_20260101.joblib"
        is_model.write_text("x")
        # Kein non-IS-Modell -> StartupError
        with pytest.raises(StartupError, match="Kein trainiertes Modell"):
            find_newest_model(tmp_path)

    def test_raises_when_empty(self, tmp_path):
        with pytest.raises(StartupError, match="Kein trainiertes Modell"):
            find_newest_model(tmp_path)

    def test_raises_contains_hint(self, tmp_path):
        """Fehlermeldung soll Hinweis auf train_model.py enthalten."""
        with pytest.raises(StartupError, match="train_model.py"):
            find_newest_model(tmp_path)


# ---------------------------------------------------------------------------
# find_newest_mr_model
# ---------------------------------------------------------------------------

class TestFindNewestMrModel:
    def test_returns_newest_mr_model(self, tmp_path):
        import time
        m1 = tmp_path / "mean_reversion_model_20260101.joblib"
        m1.write_text("x")
        time.sleep(0.01)
        m2 = tmp_path / "mean_reversion_model_20260201.joblib"
        m2.write_text("x")

        result = find_newest_mr_model(tmp_path)
        assert result == m2

    def test_raises_when_no_mr_model(self, tmp_path):
        with pytest.raises(StartupError, match="MeanReversion"):
            find_newest_mr_model(tmp_path)

    def test_does_not_return_signal_model(self, tmp_path):
        """signal_model_v*.joblib darf NICHT als MR-Modell gelten."""
        (tmp_path / "signal_model_v1_20260101.joblib").write_text("x")
        with pytest.raises(StartupError, match="MeanReversion"):
            find_newest_mr_model(tmp_path)


# ---------------------------------------------------------------------------
# build_mt5_connector
# ---------------------------------------------------------------------------

class TestBuildMt5Connector:
    def test_raises_when_credentials_missing(self, monkeypatch):
        monkeypatch.delenv("MT5_LOGIN",    raising=False)
        monkeypatch.delenv("MT5_PASSWORD", raising=False)
        monkeypatch.delenv("MT5_SERVER",   raising=False)

        with pytest.raises(StartupError, match="Zugangsdaten"):
            build_mt5_connector()

    def test_raises_when_only_login_set(self, monkeypatch):
        monkeypatch.setenv("MT5_LOGIN", "123")
        monkeypatch.delenv("MT5_PASSWORD", raising=False)
        monkeypatch.delenv("MT5_SERVER",   raising=False)

        with pytest.raises(StartupError, match="Zugangsdaten"):
            build_mt5_connector()

    def test_raises_when_login_not_numeric(self, monkeypatch):
        monkeypatch.setenv("MT5_LOGIN",    "not_a_number")
        monkeypatch.setenv("MT5_PASSWORD", "pw")
        monkeypatch.setenv("MT5_SERVER",   "srv")

        with pytest.raises(StartupError, match="gueltige Zahl"):
            build_mt5_connector()

    def test_raises_on_connection_failure(self, monkeypatch):
        monkeypatch.setenv("MT5_LOGIN",    "12345")
        monkeypatch.setenv("MT5_PASSWORD", "pw")
        monkeypatch.setenv("MT5_SERVER",   "srv")

        # MT5Connector wird lazy importiert → Patch am Quellmodul
        with patch("src.data.mt5_connector.MT5Connector") as MockCls:
            instance = MockCls.return_value
            instance.connect.side_effect = ConnectionError("MT5 nicht gefunden")
            with pytest.raises(StartupError, match="Verbindung fehlgeschlagen"):
                build_mt5_connector()

    def test_raises_contains_server_info(self, monkeypatch):
        monkeypatch.setenv("MT5_LOGIN",    "12345")
        monkeypatch.setenv("MT5_PASSWORD", "pw")
        monkeypatch.setenv("MT5_SERVER",   "DemoServer-99")

        with patch("src.data.mt5_connector.MT5Connector") as MockCls:
            instance = MockCls.return_value
            instance.connect.side_effect = ConnectionError("timeout")
            with pytest.raises(StartupError, match="DemoServer-99"):
                build_mt5_connector()


# ---------------------------------------------------------------------------
# _OandaStub
# ---------------------------------------------------------------------------

class TestOandaStub:
    def test_is_not_connected(self):
        assert _OandaStub.is_connected is False

    def test_get_ohlcv_raises(self):
        with pytest.raises(RuntimeError, match="OANDA-Stub"):
            _OandaStub().get_ohlcv("EURUSD", "H1", None, None)


# ---------------------------------------------------------------------------
# build_trading_stack (Single-Symbol)
# ---------------------------------------------------------------------------

class TestBuildTradingStack:
    """
    Testet den vollstaendigen Stack-Aufbau mit einem echten (minimalen) Modell
    und gemockten IO-Abhaengigkeiten (Calendar-HTTP, MT5-Daten).
    """

    @pytest.fixture(autouse=True)
    def _set_confirm_live(self, monkeypatch):
        """build_trading_stack erfordert CONFIRM_LIVE=yes (live_trading_enabled=True)."""
        monkeypatch.setenv("CONFIRM_LIVE", "yes")

    def test_demo_live_mode_enforced(
        self, tmp_config, tmp_model, tmp_path, minimal_connector
    ):
        """OrderExecutor muss live_trading_enabled=True haben (Demo-Live-Modus)."""
        with (
            patch("src.data.calendar.EconomicCalendar.refresh"),
            patch("src.data.calendar.EconomicCalendar.is_no_trade_zone", return_value=False),
        ):
            stack = build_trading_stack(
                config=_load_config(tmp_config),
                connector=minimal_connector,
                model_path=tmp_model,
                symbol="EURUSD",
                timeframe="H1",
            )

        executor = stack["order_executor"]
        # Demo-Live-Modus: live_trading_enabled=True (echte Demo-Positionen via MT5)
        assert executor._live is True, (
            "live_trading_enabled muss True sein (Demo-Live-Modus)!"
        )

    def test_mode_is_confirm_required(
        self, tmp_config, tmp_model, minimal_connector
    ):
        """Orchestrator muss im CONFIRM_REQUIRED-Modus gestartet werden."""
        from src.modes import TradingMode

        with (
            patch("src.data.calendar.EconomicCalendar.refresh"),
            patch("src.data.calendar.EconomicCalendar.is_no_trade_zone", return_value=False),
        ):
            stack = build_trading_stack(
                config=_load_config(tmp_config),
                connector=minimal_connector,
                model_path=tmp_model,
            )

        assert stack["orchestrator"].mode == TradingMode.CONFIRM_REQUIRED

    def test_returns_all_expected_keys(
        self, tmp_config, tmp_model, minimal_connector
    ):
        with (
            patch("src.data.calendar.EconomicCalendar.refresh"),
            patch("src.data.calendar.EconomicCalendar.is_no_trade_zone", return_value=False),
        ):
            stack = build_trading_stack(
                config=_load_config(tmp_config),
                connector=minimal_connector,
                model_path=tmp_model,
            )

        for key in ("orchestrator", "order_executor", "order_relay", "symbols",
                    "pipeline", "audit_log", "connector"):
            assert key in stack, f"Schuessel '{key}' fehlt im Stack"

    def test_symbols_list(self, tmp_config, tmp_model, minimal_connector):
        with (
            patch("src.data.calendar.EconomicCalendar.refresh"),
            patch("src.data.calendar.EconomicCalendar.is_no_trade_zone", return_value=False),
        ):
            stack = build_trading_stack(
                config=_load_config(tmp_config),
                connector=minimal_connector,
                model_path=tmp_model,
                symbol="GBPUSD",
            )

        assert stack["symbols"] == ["GBPUSD"]

    def test_order_relay_attached(self, tmp_config, tmp_model, minimal_connector):
        """OrderEventRelay muss am OrderExecutor haengen (Callbacks gesetzt)."""
        with (
            patch("src.data.calendar.EconomicCalendar.refresh"),
            patch("src.data.calendar.EconomicCalendar.is_no_trade_zone", return_value=False),
        ):
            stack = build_trading_stack(
                config=_load_config(tmp_config),
                connector=minimal_connector,
                model_path=tmp_model,
            )

        executor = stack["order_executor"]
        assert executor._on_open_cb is not None, "on_open_cb nicht gesetzt"
        assert executor._on_close_cb is not None, "on_close_cb nicht gesetzt"

    def test_confirmation_callback_forwarded(
        self, tmp_config, tmp_model, minimal_connector
    ):
        """Der confirmation_callback muss an den Orchestrator weitergegeben werden."""
        cb = MagicMock()
        cb.confirm_order = MagicMock(return_value=True)

        with (
            patch("src.data.calendar.EconomicCalendar.refresh"),
            patch("src.data.calendar.EconomicCalendar.is_no_trade_zone", return_value=False),
        ):
            stack = build_trading_stack(
                config=_load_config(tmp_config),
                connector=minimal_connector,
                model_path=tmp_model,
                confirmation_callback=cb,
            )

        orch = stack["orchestrator"]
        assert orch._confirmation_callback is cb


# ---------------------------------------------------------------------------
# build_portfolio_stack
# ---------------------------------------------------------------------------

class TestBuildPortfolioStack:
    """Testet den Portfolio-Stack-Aufbau (XAUUSD TF + EURUSD MR)."""

    @pytest.fixture(autouse=True)
    def _set_confirm_live(self, monkeypatch):
        """build_portfolio_stack erfordert CONFIRM_LIVE=yes (live_trading_enabled=True)."""
        monkeypatch.setenv("CONFIRM_LIVE", "yes")

    def test_returns_all_expected_keys(
        self, tmp_config, tmp_model, tmp_mr_model, minimal_connector
    ):
        with (
            patch("src.data.calendar.EconomicCalendar.refresh"),
            patch("src.data.calendar.EconomicCalendar.is_no_trade_zone", return_value=False),
        ):
            stack = build_portfolio_stack(
                config=_load_config(tmp_config),
                connector=minimal_connector,
                xauusd_model_path=tmp_model,
                eurusd_mr_model_path=tmp_mr_model,
            )

        for key in ("orchestrator", "order_executor", "order_relay", "symbols",
                    "pipeline", "audit_log", "connector"):
            assert key in stack, f"Schuessel '{key}' fehlt im Portfolio-Stack"

    def test_symbols_are_xauusd_and_eurusd(
        self, tmp_config, tmp_model, tmp_mr_model, minimal_connector
    ):
        with (
            patch("src.data.calendar.EconomicCalendar.refresh"),
            patch("src.data.calendar.EconomicCalendar.is_no_trade_zone", return_value=False),
        ):
            stack = build_portfolio_stack(
                config=_load_config(tmp_config),
                connector=minimal_connector,
                xauusd_model_path=tmp_model,
                eurusd_mr_model_path=tmp_mr_model,
            )

        assert stack["symbols"] == ["XAUUSD", "EURUSD"]

    def test_orchestrator_is_multi_symbol(
        self, tmp_config, tmp_model, tmp_mr_model, minimal_connector
    ):
        with (
            patch("src.data.calendar.EconomicCalendar.refresh"),
            patch("src.data.calendar.EconomicCalendar.is_no_trade_zone", return_value=False),
        ):
            stack = build_portfolio_stack(
                config=_load_config(tmp_config),
                connector=minimal_connector,
                xauusd_model_path=tmp_model,
                eurusd_mr_model_path=tmp_mr_model,
            )

        assert isinstance(stack["orchestrator"], MultiSymbolOrchestrator)

    def test_mode_is_autonomous(
        self, tmp_config, tmp_model, tmp_mr_model, minimal_connector
    ):
        from src.modes import TradingMode

        with (
            patch("src.data.calendar.EconomicCalendar.refresh"),
            patch("src.data.calendar.EconomicCalendar.is_no_trade_zone", return_value=False),
        ):
            stack = build_portfolio_stack(
                config=_load_config(tmp_config),
                connector=minimal_connector,
                xauusd_model_path=tmp_model,
                eurusd_mr_model_path=tmp_mr_model,
            )

        assert stack["orchestrator"].mode == TradingMode.AUTONOMOUS

    def test_confirmation_callback_forwarded_to_both(
        self, tmp_config, tmp_model, tmp_mr_model, minimal_connector
    ):
        """confirmation_callback muss an BEIDE Orchestratoren weitergegeben werden."""
        cb = MagicMock()
        cb.confirm_order = MagicMock(return_value=True)

        with (
            patch("src.data.calendar.EconomicCalendar.refresh"),
            patch("src.data.calendar.EconomicCalendar.is_no_trade_zone", return_value=False),
        ):
            stack = build_portfolio_stack(
                config=_load_config(tmp_config),
                connector=minimal_connector,
                xauusd_model_path=tmp_model,
                eurusd_mr_model_path=tmp_mr_model,
                confirmation_callback=cb,
            )

        multi_orch = stack["orchestrator"]
        for symbol, orch in multi_orch._pairs:
            assert orch._confirmation_callback is cb, (
                f"Orchestrator fuer {symbol}: confirmation_callback nicht gesetzt"
            )

    def test_portfolio_order_executor_is_live(self, tmp_config, tmp_model, tmp_mr_model, minimal_connector):
        with (
            patch("src.data.calendar.EconomicCalendar.refresh"),
            patch("src.data.calendar.EconomicCalendar.is_no_trade_zone", return_value=False),
        ):
            stack = build_portfolio_stack(
                config=_load_config(tmp_config),
                connector=minimal_connector,
                xauusd_model_path=tmp_model,
                eurusd_mr_model_path=tmp_mr_model,
            )

        assert getattr(stack["order_executor"], "_live", False) is True

    def test_build_trading_stack_injects_regime_detector(self, tmp_config, tmp_model, minimal_connector):
        with (
            patch("src.data.calendar.EconomicCalendar.refresh"),
            patch("src.data.calendar.EconomicCalendar.is_no_trade_zone", return_value=False),
        ):
            stack = build_trading_stack(
                config=_load_config(tmp_config),
                connector=minimal_connector,
                model_path=tmp_model,
            )

        assert isinstance(stack["orchestrator"]._regime_detector, RegimeDetector)

    def test_build_portfolio_stack_injects_regime_detectors(self, tmp_config, tmp_model, tmp_mr_model, minimal_connector):
        with (
            patch("src.data.calendar.EconomicCalendar.refresh"),
            patch("src.data.calendar.EconomicCalendar.is_no_trade_zone", return_value=False),
        ):
            stack = build_portfolio_stack(
                config=_load_config(tmp_config),
                connector=minimal_connector,
                xauusd_model_path=tmp_model,
                eurusd_mr_model_path=tmp_mr_model,
            )

        multi_orch = stack["orchestrator"]
        for _, orch in multi_orch._pairs:
            assert isinstance(orch._regime_detector, RegimeDetector)


# ---------------------------------------------------------------------------
# MultiSymbolOrchestrator
# ---------------------------------------------------------------------------

class TestMultiSymbolOrchestrator:
    """Testet das MultiSymbolOrchestrator-Wrapper-Verhalten."""

    def _make_mock_orch(self, signal="flat"):
        """Erstellt einen gemockten TradingOrchestrator."""
        from src.modes import TradingMode
        orch = MagicMock()
        orch.mode = TradingMode.CONFIRM_REQUIRED
        orch.is_paused = False
        orch.run_cycle.return_value = {
            "symbol": "SYM",
            "signal": signal,
            "action": "flat",
            "reason": "signal_flat",
            "ticket": None,
            "lot_size": None,
            "step_stopped_at": "flat_signal",
            "checks": [],
            "timestamp": None,
        }
        return orch

    def test_mode_returns_first_orch_mode(self):
        from src.modes import TradingMode
        orch1 = self._make_mock_orch()
        orch2 = self._make_mock_orch()
        multi = MultiSymbolOrchestrator([("A", orch1), ("B", orch2)])
        assert multi.mode == TradingMode.CONFIRM_REQUIRED

    def test_stop_sets_stop_event(self):
        orch = self._make_mock_orch()
        multi = MultiSymbolOrchestrator([("A", orch)])
        multi.stop()
        assert multi._stop_event.is_set()

    def test_pause_delegates_to_all(self):
        orch1 = self._make_mock_orch()
        orch2 = self._make_mock_orch()
        multi = MultiSymbolOrchestrator([("A", orch1), ("B", orch2)])
        multi.pause("test")
        orch1.pause.assert_called_once_with("test")
        orch2.pause.assert_called_once_with("test")

    def test_resume_delegates_to_all(self):
        orch1 = self._make_mock_orch()
        orch2 = self._make_mock_orch()
        multi = MultiSymbolOrchestrator([("A", orch1), ("B", orch2)])
        multi.resume()
        orch1.resume.assert_called_once()
        orch2.resume.assert_called_once()

    def test_set_mode_delegates_to_all(self):
        from src.modes import TradingMode
        orch1 = self._make_mock_orch()
        orch2 = self._make_mock_orch()
        multi = MultiSymbolOrchestrator([("A", orch1), ("B", orch2)])
        multi.set_mode(TradingMode.SUGGEST_ONLY)
        orch1.set_mode.assert_called_once_with(TradingMode.SUGGEST_ONLY)
        orch2.set_mode.assert_called_once_with(TradingMode.SUGGEST_ONLY)

    def test_set_confirmation_callback_delegates_to_all(self):
        cb = MagicMock()
        orch1 = self._make_mock_orch()
        orch2 = self._make_mock_orch()
        multi = MultiSymbolOrchestrator([("A", orch1), ("B", orch2)])
        multi.set_confirmation_callback(cb)
        orch1.set_confirmation_callback.assert_called_once_with(cb)
        orch2.set_confirmation_callback.assert_called_once_with(cb)

    def test_run_loop_calls_run_cycle_for_each_symbol(self):
        """run_loop muss run_cycle fuer jedes Symbol aufrufen und dann stoppen."""
        import threading
        orch1 = self._make_mock_orch()
        orch2 = self._make_mock_orch()

        cycle_count = [0]
        target = 2  # ein kompletter Durchlauf durch beide Symbole

        original_result = orch1.run_cycle.return_value.copy()
        original_result2 = orch2.run_cycle.return_value.copy()

        multi = MultiSymbolOrchestrator([("A", orch1), ("B", orch2)])

        def _on_activity(result):
            cycle_count[0] += 1
            if cycle_count[0] >= target:
                multi.stop()

        multi.set_activity_callback(_on_activity)

        t = threading.Thread(
            target=multi.run_loop,
            args=(["A", "B"],),
            kwargs={"interval_seconds": 0},
        )
        t.start()
        t.join(timeout=3.0)

        assert not t.is_alive(), "run_loop hat sich nicht beendet!"
        assert orch1.run_cycle.call_count >= 1
        assert orch2.run_cycle.call_count >= 1


# ---------------------------------------------------------------------------
# Regressionstest: Bestaetigung darf Bot nicht inaktiv machen
# ---------------------------------------------------------------------------

class TestConfirmationContinuityRegression:
    """
    Regression: Nach einer Bestaetigung darf der Bot-Loop NICHT inaktiv werden.

    Reproduziert den frueheren Bug wo der Bot nach der ersten Bestaetigung
    keine weiteren Zyklen mehr ausgefuehrt hat.
    """

    @pytest.fixture
    def long_signal_orchestrator(self):
        """TradingOrchestrator im CONFIRM_REQUIRED-Modus der immer LONG signalisiert."""
        import pandas as pd
        from src.orchestrator import TradingOrchestrator
        from src.modes import TradingMode

        features = pd.DataFrame({
            "close": [1.2000],
            "atr":   [0.0010],
            "f0":    [0.5],
        })

        pipeline     = MagicMock()
        risk_guard   = MagicMock()
        risk_guard.is_trading_allowed.return_value           = True
        risk_guard.get_position_size_multiplier.return_value = 1.0

        pre_trade = MagicMock()
        pre_trade.is_safe_to_trade.return_value = (True, "")

        signal_model = MagicMock()
        signal_model.get_signal.return_value    = "long"
        signal_model.predict_proba.return_value = {
            "long": 0.70, "short": 0.15, "neutral": 0.15
        }

        corr_guard = MagicMock()
        corr_guard.can_open_position.return_value = True

        size_result = MagicMock()
        size_result.is_valid           = True
        size_result.lot_size           = 0.01
        size_result.stop_loss_distance = 0.001
        size_result.rejection_reason   = ""

        pos_sizer = MagicMock()
        pos_sizer.calculate_lot_size.return_value = size_result

        executor = MagicMock()
        executor.get_open_positions.return_value = []
        executor.open_position.return_value      = {"ticket": 99999}

        orch = TradingOrchestrator(
            data_pipeline     = pipeline,
            risk_guard        = risk_guard,
            pre_trade_check   = pre_trade,
            signal_model      = signal_model,
            correlation_guard = corr_guard,
            position_sizer    = pos_sizer,
            order_executor    = executor,
            features_loader   = lambda sym: features,
            balance_getter    = lambda: 10_000.0,
            timeframe         = "H4",
            mode              = TradingMode.CONFIRM_REQUIRED,
        )
        return orch, executor

    # ── Einzelne Bestaetigung ─────────────────────────────────────────────

    def test_single_confirmation_executes_order(self, long_signal_orchestrator):
        """Einzelne Bestaetigung muss in ausgefuehrter Order resultieren."""
        orch, executor = long_signal_orchestrator

        cb = MagicMock()
        cb.confirm_order.return_value = True
        cb.last_confirmed_lot_size    = None
        orch.set_confirmation_callback(cb)

        result = orch.run_cycle("XAUUSD")

        assert result["action"] == "open_buy", (
            f"Erwartet 'open_buy', erhalten '{result['action']}'"
        )
        assert cb.confirm_order.call_count == 1
        assert executor.open_position.call_count == 1

    # ── Haupt-Regression: N aufeinanderfolgende Bestaetigungen ────────────

    def test_bot_continues_after_multiple_confirmations(self, long_signal_orchestrator):
        """
        Kern-Regressionstest: N Bestaetigungen hintereinander muessen alle
        durchlaufen – der Bot darf nach der ersten Bestaetigung NICHT einfrieren.
        """
        orch, executor = long_signal_orchestrator

        confirm_count = [0]

        class _CountingCallback:
            last_confirmed_lot_size = None

            def confirm_order(self, symbol, direction, lot_size, sl, tp):
                confirm_count[0] += 1
                self.last_confirmed_lot_size = lot_size
                return True

        orch.set_confirmation_callback(_CountingCallback())

        N = 5
        for i in range(N):
            result = orch.run_cycle("XAUUSD")
            assert result["action"] == "open_buy", (
                f"Zyklus {i + 1}: Erwartet 'open_buy', erhalten '{result['action']}'. "
                "Bot ist nach Bestaetigung inaktiv geworden! (Regression)"
            )

        assert confirm_count[0] == N, (
            f"Erwartet {N} Bestaetigungen, erhalten {confirm_count[0]}"
        )
        assert executor.open_position.call_count == N

    # ── Ablehnung gefolgt von Bestaetigung ────────────────────────────────

    def test_rejection_does_not_break_subsequent_cycles(self, long_signal_orchestrator):
        """
        Abgelehnte Bestaetigung darf naechsten Zyklus nicht blockieren.
        Zyklus 1: ablehnen  →  action='skipped'
        Zyklus 2: bestaetigen →  action='open_buy'
        """
        orch, executor = long_signal_orchestrator

        call_count = [0]

        class _RejectThenConfirm:
            last_confirmed_lot_size = None

            def confirm_order(self, symbol, direction, lot_size, sl, tp):
                call_count[0] += 1
                self.last_confirmed_lot_size = lot_size
                return call_count[0] > 1

        orch.set_confirmation_callback(_RejectThenConfirm())

        r1 = orch.run_cycle("XAUUSD")
        r2 = orch.run_cycle("XAUUSD")

        assert r1["action"] == "skipped", (
            f"Zyklus 1 (Ablehnung): Erwartet 'skipped', erhalten '{r1['action']}'"
        )
        assert r1["reason"] == "order_not_confirmed"

        assert r2["action"] == "open_buy", (
            f"Zyklus 2 nach Ablehnung: Erwartet 'open_buy', erhalten '{r2['action']}'. "
            "Bot ist nach Ablehnung inaktiv geworden! (Regression)"
        )

    # ── Threading: run_loop laeuft nach N Bestaetigungen weiter ──────────

    def test_run_loop_continues_after_n_confirmations(self, long_signal_orchestrator):
        """
        run_loop() muss nach N Bestaetigungen weiter iterieren.
        Testet die threading.Event-Korrektheit im vollstaendigen Loop.
        """
        import threading

        orch, executor = long_signal_orchestrator

        cycle_actions = []
        target        = 3

        class _AutoConfirm:
            last_confirmed_lot_size = None

            def confirm_order(self, symbol, direction, lot_size, sl, tp):
                self.last_confirmed_lot_size = lot_size
                return True

        orch.set_confirmation_callback(_AutoConfirm())

        def _on_activity(result):
            cycle_actions.append(result["action"])
            if len(cycle_actions) >= target:
                orch.stop()

        orch.set_activity_callback(_on_activity)

        t = threading.Thread(
            target=orch.run_loop,
            args=(["XAUUSD"],),
            kwargs={"interval_seconds": 0},
        )
        t.start()
        t.join(timeout=5.0)

        assert not t.is_alive(), (
            "run_loop() hat sich nicht beendet – moeglicher Deadlock!"
        )
        assert len(cycle_actions) >= target, (
            f"Erwartet mindestens {target} Zyklen, erhalten {len(cycle_actions)}. "
            "Bot-Loop ist nach Bestaetigung eingefroren! (Regression)"
        )
        assert all(a == "open_buy" for a in cycle_actions[:target]), (
            f"Nicht alle Zyklen resultierten in 'open_buy': {cycle_actions[:target]}"
        )


# ---------------------------------------------------------------------------
# Integrations-Smoke: main() mit fehlenden Creds bricht fruehzeitig ab
# ---------------------------------------------------------------------------

class TestMainEarlyExit:
    def test_main_exits_1_when_model_missing(self, tmp_path, monkeypatch, tmp_config):
        monkeypatch.chdir(tmp_path)
        # kein models/-Verzeichnis → find_newest_model schlaegt fehl
        (tmp_path / "models").mkdir()
        (tmp_path / "config").mkdir()
        import shutil
        shutil.copy(tmp_config, tmp_path / "config" / "config.yaml")

        from scripts.run_gui_bot import main
        rc = main(["--config", "config/config.yaml"])
        assert rc == 1

    def test_main_exits_1_when_mt5_creds_missing(
        self, tmp_path, monkeypatch, tmp_config, tmp_model, tmp_mr_model
    ):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("MT5_LOGIN",    raising=False)
        monkeypatch.delenv("MT5_PASSWORD", raising=False)
        monkeypatch.delenv("MT5_SERVER",   raising=False)

        (tmp_path / "models").mkdir()
        import shutil
        shutil.copy(tmp_model, tmp_path / "models" / tmp_model.name)
        shutil.copy(tmp_mr_model, tmp_path / "models" / tmp_mr_model.name)
        (tmp_path / "config").mkdir()
        shutil.copy(tmp_config, tmp_path / "config" / "config.yaml")

        from scripts.run_gui_bot import main
        rc = main(["--config", "config/config.yaml"])
        assert rc == 1


# ---------------------------------------------------------------------------
# calc_unrealized_pnl – pure Funktion, keine externe Abhaengigkeit
# ---------------------------------------------------------------------------

class TestCalcUnrealizedPnl:
    """Testet die P&L-Berechnung fuer offene Positionen (BUY/SELL, Gewinn/Verlust)."""

    # ── BUY-Positionen ───────────────────────────────────────────────────────

    def test_buy_profit(self):
        """BUY: Kurs steigt -> positiver P&L."""
        pnl = calc_unrealized_pnl(
            direction="buy",
            open_price=1.0800,
            current_bid=1.0900,
            current_ask=1.0902,
            lot_size=0.1,
            contract_size=100_000,
        )
        assert abs(pnl - 100.0) < 0.001  # (1.0900 - 1.0800) * 0.1 * 100_000

    def test_buy_loss(self):
        """BUY: Kurs faellt -> negativer P&L."""
        pnl = calc_unrealized_pnl(
            direction="buy",
            open_price=1.0900,
            current_bid=1.0800,
            current_ask=1.0802,
            lot_size=0.1,
            contract_size=100_000,
        )
        assert abs(pnl - (-100.0)) < 0.001  # (1.0800 - 1.0900) * 0.1 * 100_000

    def test_buy_breakeven(self):
        """BUY: Kurs unveraendert -> P&L nahe Null."""
        pnl = calc_unrealized_pnl(
            direction="buy",
            open_price=1.0850,
            current_bid=1.0850,
            current_ask=1.0852,
            lot_size=1.0,
            contract_size=100_000,
        )
        assert abs(pnl) < 0.001

    # ── SELL-Positionen ──────────────────────────────────────────────────────

    def test_sell_profit(self):
        """SELL: Kurs faellt -> positiver P&L."""
        pnl = calc_unrealized_pnl(
            direction="sell",
            open_price=1.0900,
            current_bid=1.0798,
            current_ask=1.0800,
            lot_size=0.1,
            contract_size=100_000,
        )
        assert abs(pnl - 100.0) < 0.001  # (1.0900 - 1.0800) * 0.1 * 100_000

    def test_sell_loss(self):
        """SELL: Kurs steigt -> negativer P&L."""
        pnl = calc_unrealized_pnl(
            direction="sell",
            open_price=1.0800,
            current_bid=1.0898,
            current_ask=1.0900,
            lot_size=0.1,
            contract_size=100_000,
        )
        assert abs(pnl - (-100.0)) < 0.001  # (1.0800 - 1.0900) * 0.1 * 100_000

    def test_sell_breakeven(self):
        """SELL: Kurs unveraendert -> P&L nahe Null."""
        pnl = calc_unrealized_pnl(
            direction="sell",
            open_price=1.0850,
            current_bid=1.0848,
            current_ask=1.0850,
            lot_size=1.0,
            contract_size=100_000,
        )
        assert abs(pnl) < 0.001

    # ── XAUUSD (Gold, contract_size=100) ────────────────────────────────────

    def test_xauusd_buy_profit(self):
        """XAUUSD BUY: Gold steigt 1 USD pro Oz -> 1 USD P&L bei 0.01 Lots."""
        pnl = calc_unrealized_pnl(
            direction="buy",
            open_price=1900.00,
            current_bid=1901.00,
            current_ask=1901.10,
            lot_size=0.01,
            contract_size=100,
        )
        assert abs(pnl - 1.0) < 0.001  # (1901.00 - 1900.00) * 0.01 * 100

    def test_xauusd_sell_profit(self):
        """XAUUSD SELL: Gold faellt 1 USD pro Oz -> 1 USD P&L bei 0.01 Lots."""
        pnl = calc_unrealized_pnl(
            direction="sell",
            open_price=1901.00,
            current_bid=1899.90,
            current_ask=1900.00,
            lot_size=0.01,
            contract_size=100,
        )
        assert abs(pnl - 1.0) < 0.001  # (1901.00 - 1900.00) * 0.01 * 100

    # ── Vorzeichen-Sicherheit ────────────────────────────────────────────────

    def test_buy_pnl_positive_when_bid_above_open(self):
        pnl = calc_unrealized_pnl("buy", 1.0, 1.1, 1.102, 1.0, 100_000)
        assert pnl > 0

    def test_buy_pnl_negative_when_bid_below_open(self):
        pnl = calc_unrealized_pnl("buy", 1.1, 1.0, 1.002, 1.0, 100_000)
        assert pnl < 0

    def test_sell_pnl_positive_when_ask_below_open(self):
        pnl = calc_unrealized_pnl("sell", 1.1, 0.998, 1.0, 1.0, 100_000)
        assert pnl > 0

    def test_sell_pnl_negative_when_ask_above_open(self):
        pnl = calc_unrealized_pnl("sell", 1.0, 1.098, 1.1, 1.0, 100_000)
        assert pnl < 0
