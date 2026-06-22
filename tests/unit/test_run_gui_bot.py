"""
tests/unit/test_run_gui_bot.py
Unit-Tests fuer scripts/run_gui_bot.py (Startup-Logik, keine echte MT5-Verbindung).
"""

from __future__ import annotations

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
    build_mt5_connector,
    _OandaStub,
    build_trading_stack,
)


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
    joblib.dump(payload, p)
    return p


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
# build_trading_stack
# ---------------------------------------------------------------------------

class TestBuildTradingStack:
    """
    Testet den vollstaendigen Stack-Aufbau mit einem echten (minimalen) Modell
    und gemockten IO-Abhaengigkeiten (Calendar-HTTP, MT5-Daten).
    """

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
        rc = main(["--symbol", "EURUSD", "--config", "config/config.yaml"])
        assert rc == 1

    def test_main_exits_1_when_mt5_creds_missing(
        self, tmp_path, monkeypatch, tmp_config, tmp_model
    ):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("MT5_LOGIN",    raising=False)
        monkeypatch.delenv("MT5_PASSWORD", raising=False)
        monkeypatch.delenv("MT5_SERVER",   raising=False)

        (tmp_path / "models").mkdir()
        import shutil
        shutil.copy(tmp_model, tmp_path / "models" / tmp_model.name)
        (tmp_path / "config").mkdir()
        shutil.copy(tmp_config, tmp_path / "config" / "config.yaml")

        from scripts.run_gui_bot import main
        rc = main(["--symbol", "EURUSD", "--config", "config/config.yaml"])
        assert rc == 1
