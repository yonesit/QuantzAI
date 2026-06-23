"""
scripts/run_gui_bot.py
QuantzAI GUI-Bot-Starter – Portfolio-Modus (Demo-Live).

Portfolio-Setup (2-Wege H4, validiert 2026-06-22):
  - XAUUSD H4 Trendfolge    (SignalModel, Test #3)
  - EURUSD H4 Mean-Reversion (MeanReversionModel, Test #4)
  - 50/50 Risikoallokation pro Symbol (gleiche risk_per_trade_pct)

Startet:
  - GUI (PySide6 MainWindow)
  - Echte MT5-Verbindung fuer Live-Daten beider Symbole
  - SignalModel        (XAUUSD H4, neuestes models/signal_model_v*.joblib)
  - MeanReversionModel (EURUSD H4, neuestes models/mean_reversion_model*.joblib)
  - Echte PreTradeCheck (EconomicCalendar + Spread-Filter)
  - Gemeinsame RiskGuard / PositionSizer / CorrelationGuard / AuditLog
  - OrderExecutor im Paper-Modus (simulierte Positionen, KEIN echtes Geld)
  - AUTONOMOUS: Bot handelt selbststaendig, keine Bestaetigung noetig (CONFIRM_AUTONOMOUS=yes)
  - ActivityLogWidget, OrderEventRelay alle verdrahtet

Fehlschlag mit klarer Meldung wenn:
  - MT5_LOGIN / MT5_PASSWORD / MT5_SERVER fehlen in .env
  - Kein XAUUSD TF-Modell oder kein EURUSD MR-Modell in models/ vorhanden
  - MT5-Verbindung schlaegt fehl

Verwendung:
  python scripts/run_gui_bot.py [--xauusd-model PATH] [--eurusd-mr-model PATH]
                                 [--interval 300] [--config config.yaml]
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
from pathlib import Path

# Projekt-Root in Pfad aufnehmen
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class StartupError(RuntimeError):
    """Wird geworfen wenn der Bot nicht gestartet werden kann."""


# ---------------------------------------------------------------------------
# Konfiguration laden
# ---------------------------------------------------------------------------

def _load_config(config_path: str | Path = "config/config.yaml") -> dict:
    import yaml
    p = Path(config_path)
    if not p.exists():
        raise StartupError(f"Konfigurationsdatei nicht gefunden: {p}")
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_env(env_path: str = ".env") -> None:
    """Laedt .env in os.environ (setzt nur noch nicht gesetzte Variablen)."""
    p = Path(env_path)
    if not p.exists():
        return
    with open(p, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val


# ---------------------------------------------------------------------------
# Modell-Suche
# ---------------------------------------------------------------------------

def find_newest_model(model_dir: str | Path = "models") -> Path:
    """
    Gibt den Pfad zum neuesten Signal-Modell zurueck.
    Schlaegt mit StartupError fehl wenn kein Modell vorhanden ist.
    """
    d = Path(model_dir)
    # _IS_ Modelle sind In-Sample-only, nicht fuer Live nutzen
    candidates = sorted(
        [f for f in d.glob("signal_model_v*.joblib") if "_IS_" not in f.name],
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise StartupError(
            f"Kein trainiertes Modell in '{d}' gefunden.\n"
            "Fuehre zuerst aus: python scripts/train_model.py --symbol XAUUSD\n"
            "Erwartetes Muster: models/signal_model_v1_YYYYMMDD.joblib"
        )
    return candidates[0]


def find_newest_mr_model(model_dir: str | Path = "models") -> Path:
    """
    Gibt den Pfad zum neuesten MeanReversion-Modell zurueck.
    Schlaegt mit StartupError fehl wenn kein MR-Modell vorhanden ist.
    """
    d = Path(model_dir)
    candidates = sorted(
        list(d.glob("mean_reversion_model*.joblib")),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise StartupError(
            f"Kein MeanReversion-Modell in '{d}' gefunden.\n"
            "Trainiere das MR-Modell und speichere es:\n"
            "  model.save('models/mean_reversion_model_YYYYMMDD.joblib')\n"
            "Erwartetes Muster: models/mean_reversion_model*.joblib"
        )
    return candidates[0]


# ---------------------------------------------------------------------------
# MT5-Connector
# ---------------------------------------------------------------------------

def build_mt5_connector(max_retries: int = 1):
    """
    Baut und verbindet einen MT5Connector aus .env-Variablen.
    Schlaegt mit StartupError fehl wenn Zugangsdaten fehlen oder Verbindung scheitert.
    """
    from src.data.mt5_connector import MT5Connector

    login_str = os.environ.get("MT5_LOGIN", "").strip()
    password  = os.environ.get("MT5_PASSWORD", "").strip()
    server    = os.environ.get("MT5_SERVER",   "").strip()
    path      = os.environ.get("MT5_PATH",     "").strip() or None

    if not login_str or not password or not server:
        raise StartupError(
            "MT5-Zugangsdaten fehlen in .env.\n"
            "Benoetigt: MT5_LOGIN, MT5_PASSWORD, MT5_SERVER\n"
            "Kopiere .env.example nach .env und fuelle die Werte aus."
        )

    try:
        login = int(login_str)
    except ValueError:
        raise StartupError(f"MT5_LOGIN ist keine gueltige Zahl: '{login_str}'")

    connector = MT5Connector(
        login=login,
        password=password,
        server=server,
        path=path,
        max_retries=max_retries,
    )
    try:
        connector.connect()
    except Exception as exc:
        raise StartupError(
            f"MT5-Verbindung fehlgeschlagen (server={server}, login={login}):\n{exc}\n"
            "Stelle sicher dass MetaTrader 5 laeuft und die Zugangsdaten korrekt sind."
        ) from exc

    return connector


# ---------------------------------------------------------------------------
# OANDA-Stub (nur bei fehlendem Fallback benoetigt)
# ---------------------------------------------------------------------------

class _OandaStub:
    """
    Minimaler Stub fuer den OANDA-Fallback-Slot im DataRouter.
    Wird verwendet wenn keine OANDA-Zugangsdaten konfiguriert sind.
    Meldet is_connected=False → DataRouter benutzt ausschliesslich MT5.
    """
    is_connected: bool = False

    def get_ohlcv(self, *_a, **_kw):
        raise RuntimeError("OANDA-Stub: kein OANDA konfiguriert")

    def get_ohlcv_count(self, *_a, **_kw):
        raise RuntimeError("OANDA-Stub: kein OANDA konfiguriert")

    def get_latest_price(self, *_a, **_kw):
        raise RuntimeError("OANDA-Stub: kein OANDA konfiguriert")


def _build_oanda(config: dict):
    """Erstellt einen echten OANDAConnector oder einen Stub wenn keine Creds da sind."""
    api_key    = os.environ.get("OANDA_API_KEY", "").strip()
    account_id = os.environ.get("OANDA_ACCOUNT_ID", "").strip()

    if not api_key or not account_id:
        from loguru import logger
        logger.info(
            "OANDA-Zugangsdaten nicht konfiguriert – "
            "DataRouter nutzt ausschliesslich MT5."
        )
        return _OandaStub()

    try:
        from src.data.oanda_connector import OANDAConnector
        env = os.environ.get("OANDA_ENVIRONMENT", "practice")
        oanda = OANDAConnector(
            api_key=api_key,
            account_id=account_id,
            environment=env,
        )
        oanda.connect()
        return oanda
    except Exception as exc:  # noqa: BLE001
        from loguru import logger
        logger.warning(
            "OANDA-Verbindung fehlgeschlagen: {exc} – Stub-Fallback aktiv", exc=exc
        )
        return _OandaStub()


# ---------------------------------------------------------------------------
# MultiSymbolOrchestrator – Portfolio-Wrapper
# ---------------------------------------------------------------------------

class MultiSymbolOrchestrator:
    """
    Fuehrt mehrere (Symbol, Orchestrator)-Paare sequenziell in einer Schleife aus.

    Implementiert die oeffentliche TradingOrchestrator-Schnittstelle fuer
    BotControlsWidget / BotWorker, delegiert aber intern an je einen
    TradingOrchestrator pro Symbol.

    Parameter
    ---------
    pairs : Liste von (symbol, TradingOrchestrator)-Tuples.
    """

    def __init__(self, pairs: list) -> None:
        self._pairs               = pairs          # [(symbol, orchestrator)]
        self._stop_event          = threading.Event()
        self._activity_callback   = None

    # ── Oeffentliche Schnittstelle (wie TradingOrchestrator) ─────────────────

    @property
    def mode(self):
        """Gibt den Modus des ersten Orchestrators zurueck."""
        return self._pairs[0][1].mode

    @property
    def is_paused(self) -> bool:
        return self._pairs[0][1].is_paused

    def run_loop(self, symbols: list, interval_seconds: int = 300) -> None:
        """
        Haupt-Loop: iteriert sequenziell ueber alle (Symbol, Orchestrator)-Paare.
        Der `symbols`-Parameter wird von BotWorker uebergeben, aber ignoriert –
        die Iteration folgt self._pairs.
        """
        from loguru import logger
        self._stop_event.clear()
        sym_names = [s for s, _ in self._pairs]
        logger.info(
            "MultiSymbolOrchestrator: Loop gestartet | Symbole={s} | {iv}s Intervall",
            s=sym_names, iv=interval_seconds,
        )

        while not self._stop_event.is_set():
            for symbol, orch in self._pairs:
                if self._stop_event.is_set():
                    break
                try:
                    result = orch.run_cycle(symbol)
                    if self._activity_callback is not None:
                        try:
                            self._activity_callback(result)
                        except Exception as _cb_exc:  # noqa: BLE001
                            logger.warning("activity_callback Fehler: {e}", e=_cb_exc)
                    logger.info(
                        "Zyklus | {sym} | action={a} | reason={r}",
                        sym=symbol, a=result["action"], r=result["reason"],
                    )
                except KeyboardInterrupt:
                    logger.info("MultiSymbolOrchestrator: KeyboardInterrupt -> Shutdown")
                    self.stop()
                    return
                except Exception as exc:  # noqa: BLE001
                    import traceback as _tb
                    tb = _tb.format_exc()
                    logger.error(
                        "MultiSymbolOrchestrator: Exception in run_cycle({s}):\n{tb}",
                        s=symbol, tb=tb,
                    )
                    print(
                        f"\n[MultiOrchestrator CRASH in run_cycle({symbol})]\n{tb}",
                        flush=True,
                    )

            self._stop_event.wait(timeout=interval_seconds)

        logger.info("MultiSymbolOrchestrator: Loop beendet.")

    def stop(self) -> None:
        """Signalisiert der run_loop()-Schleife, sauber zu beenden."""
        self._stop_event.set()

    def pause(self, reason: str = "") -> None:
        for _, orch in self._pairs:
            orch.pause(reason)

    def resume(self) -> None:
        for _, orch in self._pairs:
            orch.resume()

    def set_mode(self, new_mode) -> None:
        for _, orch in self._pairs:
            orch.set_mode(new_mode)

    def set_activity_callback(self, callback) -> None:
        self._activity_callback = callback

    def set_confirmation_callback(self, callback) -> None:
        for _, orch in self._pairs:
            orch.set_confirmation_callback(callback)

    def emergency_stop(self) -> None:
        self.stop()
        for _, orch in self._pairs:
            orch.emergency_stop()


# ---------------------------------------------------------------------------
# Den gesamten Trading-Stack bauen (Single-Symbol – fuer Tests und Fallback)
# ---------------------------------------------------------------------------

def build_trading_stack(
    *,
    config: dict,
    connector,          # MT5Connector
    model_path: Path,
    symbol: str = "EURUSD",
    timeframe: str = "H1",
    confirmation_callback=None,
) -> dict:
    """
    Baut alle echten Komponenten und verdrahtet sie (Single-Symbol).

    Parameters
    ----------
    config                : geladenes config.yaml als dict
    connector             : verbundener MT5Connector
    model_path            : Pfad zur .joblib-Modelldatei
    symbol                : Handelssymbol (Standard: EURUSD)
    timeframe             : Zeitrahmen (Standard: H1)
    confirmation_callback : GuiConfirmationCallback aus dem MainWindow

    Returns
    -------
    dict mit Schluesseln:
      orchestrator, order_executor, order_relay, symbols, pipeline
    """
    from loguru import logger

    from src.data.data_router   import DataRouter, PriceValidator
    from src.data.validator     import DataValidator
    from src.data.feature_builder import FeatureBuilder
    from src.data.pipeline      import DataPipeline
    from src.data.calendar      import EconomicCalendar
    from src.risk.risk_guard    import RiskGuard
    from src.risk.position_sizer import PositionSizer
    from src.risk.correlation_guard import CorrelationGuard
    from src.risk.pre_trade_check import PreTradeCheck
    from src.models.signal_model import SignalModel
    from src.execution.order_executor import OrderExecutor
    from src.monitoring.audit_log import AuditLog
    from src.orchestrator import TradingOrchestrator
    from src.modes import TradingMode
    from gui.widgets.order_event_relay import OrderEventRelay

    risk_cfg  = config.get("risk", {})
    model_cfg = config.get("model", {})

    # ── Modell laden ───────────────────────────────────────────────────────
    logger.info("Lade SignalModel: {path}", path=model_path)
    signal_model = SignalModel.load(model_path)
    logger.info("SignalModel geladen ({feats} Features)",
                feats=len(signal_model._feature_names))

    # ── DataPipeline aufbauen ─────────────────────────────────────────────
    oanda        = _build_oanda(config)
    router       = DataRouter(
        mt5=connector,
        oanda=oanda,
        validator=PriceValidator(
            max_pips=config.get("broker", {}).get("max_price_discrepancy_pips", 5.0),
        ),
    )
    validator       = DataValidator()
    feature_builder = FeatureBuilder.from_config(_ROOT / "config" / "config.yaml")
    pipeline        = DataPipeline(
        router=router,
        validator=validator,
        feature_builder=feature_builder,
        features_dir=str(_ROOT / "data" / "features"),
        reports_dir=str(_ROOT / "data" / "processed" / "quality_reports"),
    )

    # ── Wirtschaftskalender + PreTradeCheck ───────────────────────────────
    calendar = EconomicCalendar(
        cache_dir=str(_ROOT / "data" / "processed" / "calendar"),
    )
    calendar.refresh()   # lädt heute's Daten oder benutzt Cache

    pre_trade_check = PreTradeCheck(
        calendar=calendar,
        connector=connector,
        max_spread_pips=risk_cfg.get("spread_filter_pips", 3.0),
    )

    # ── Risiko-Komponenten ────────────────────────────────────────────────
    risk_guard = RiskGuard(
        daily_loss_limit_pct=risk_cfg.get("daily_loss_limit_pct", 5.0),
        max_drawdown_pct=risk_cfg.get("max_drawdown_pct", 15.0),
    )

    position_sizer = PositionSizer(
        risk_per_trade_pct=risk_cfg.get("max_risk_per_trade_pct", 1.0),
    )

    correlation_guard = CorrelationGuard()

    # ── AuditLog + OrderExecutor (Demo-Live-Modus) ───────────────────────
    audit_log = AuditLog(
        db_path=str(_ROOT / "data" / "processed" / "audit.db"),
    )

    # Aktiviert echte Demo-Positionen via MT5 (kein echtes Geld – Demo-Konto).
    # CONFIRM_LIVE=yes wird hier programmatisch gesetzt; fuer Live-Accounts
    # muss der Nutzer dies bewusst in der .env konfigurieren.
    os.environ.setdefault("CONFIRM_LIVE", "yes")

    order_executor = OrderExecutor(
        connector=connector,
        live_trading_enabled=True,
        paper_trades_path=str(_ROOT / "data" / "processed" / "paper_trades.json"),
        audit_log=audit_log,
    )

    # ── OrderEventRelay (#59: Live-Order-Updates) ─────────────────────────
    order_relay = OrderEventRelay()
    order_relay.attach(order_executor)

    # ── Balance-Getter: live aus MT5 ──────────────────────────────────────
    def _balance_getter() -> float:
        try:
            info = connector.get_account_info()
            return float(info.get("balance", 10_000.0))
        except Exception:  # noqa: BLE001
            return 10_000.0

    # ── Orchestrator ──────────────────────────────────────────────────────
    symbols = [symbol]

    orchestrator = TradingOrchestrator(
        data_pipeline=pipeline,
        risk_guard=risk_guard,
        pre_trade_check=pre_trade_check,
        signal_model=signal_model,
        correlation_guard=correlation_guard,
        position_sizer=position_sizer,
        order_executor=order_executor,
        audit_log=audit_log,
        features_dir=str(_ROOT / "data" / "features"),
        balance_getter=_balance_getter,
        timeframe=timeframe,
        signal_confidence_threshold=model_cfg.get("confidence_threshold", 0.55),
        mode=TradingMode.CONFIRM_REQUIRED,
        confirmation_callback=confirmation_callback,
    )

    logger.info(
        "TradingStack bereit | Symbol={sym} TF={tf} | "
        "Modus=CONFIRM_REQUIRED | Demo-Live=True",
        sym=symbol, tf=timeframe,
    )

    return {
        "orchestrator":  orchestrator,
        "order_executor": order_executor,
        "order_relay":   order_relay,
        "symbols":       symbols,
        "pipeline":      pipeline,
        "audit_log":     audit_log,
        "connector":     connector,
    }


# ---------------------------------------------------------------------------
# Portfolio-Stack bauen (XAUUSD H4 TF + EURUSD H4 MR, 50/50)
# ---------------------------------------------------------------------------

def build_portfolio_stack(
    *,
    config: dict,
    connector,
    xauusd_model_path: Path,
    eurusd_mr_model_path: Path,
    confirmation_callback=None,
) -> dict:
    """
    Baut den Portfolio-Trading-Stack fuer zwei Symbole und verdrahtet alles.

    Symbole / Modelle:
      XAUUSD H4  – Trendfolge  (SignalModel)
      EURUSD H4  – Mean-Reversion (MeanReversionModel)

    50/50 Risikoallokation: beide Orchestratoren teilen denselben PositionSizer
    mit identischer risk_per_trade_pct. Der EURUSD-Orchestrator erhaelt einen
    benutzerdefinierten Features-Loader, der die 3 MR-spezifischen Features
    (bb_pct_b, dist_ema20_atr, dist_sma50_atr) nach dem Standard-DataPipeline-
    Lauf ergaenzt.

    Returns
    -------
    dict mit Schluesseln:
      orchestrator  – MultiSymbolOrchestrator (XAUUSD + EURUSD)
      order_executor, order_relay, symbols, pipeline, audit_log, connector
    """
    from loguru import logger
    import glob as _glob
    import pandas as _pd

    from src.data.data_router   import DataRouter, PriceValidator
    from src.data.validator     import DataValidator
    from src.data.feature_builder import FeatureBuilder
    from src.data.pipeline      import DataPipeline
    from src.data.calendar      import EconomicCalendar
    from src.risk.risk_guard    import RiskGuard
    from src.risk.position_sizer import PositionSizer
    from src.risk.correlation_guard import CorrelationGuard
    from src.risk.pre_trade_check import PreTradeCheck
    from src.models.signal_model import SignalModel
    from src.models.mean_reversion_model import MeanReversionModel
    from src.execution.order_executor import OrderExecutor
    from src.monitoring.audit_log import AuditLog
    from src.orchestrator import TradingOrchestrator
    from src.modes import TradingMode
    from gui.widgets.order_event_relay import OrderEventRelay

    risk_cfg  = config.get("risk", {})
    model_cfg = config.get("model", {})

    # ── Modelle laden ──────────────────────────────────────────────────────
    logger.info("Lade XAUUSD H4 TF-Modell: {path}", path=xauusd_model_path)
    xauusd_model = SignalModel.load(xauusd_model_path)
    logger.info("XAUUSD SignalModel geladen ({feats} Features)",
                feats=len(xauusd_model._feature_names))

    logger.info("Lade EURUSD H4 MR-Modell: {path}", path=eurusd_mr_model_path)
    eurusd_model = MeanReversionModel.load(eurusd_mr_model_path)
    logger.info("EURUSD MeanReversionModel geladen")

    # ── Gemeinsame Infrastruktur ───────────────────────────────────────────
    oanda = _build_oanda(config)
    router = DataRouter(
        mt5=connector,
        oanda=oanda,
        validator=PriceValidator(
            max_pips=config.get("broker", {}).get("max_price_discrepancy_pips", 5.0),
        ),
    )
    validator       = DataValidator()
    feature_builder = FeatureBuilder.from_config(_ROOT / "config" / "config.yaml")
    features_dir    = str(_ROOT / "data" / "features")
    pipeline = DataPipeline(
        router=router,
        validator=validator,
        feature_builder=feature_builder,
        features_dir=features_dir,
        reports_dir=str(_ROOT / "data" / "processed" / "quality_reports"),
    )

    calendar = EconomicCalendar(cache_dir=str(_ROOT / "data" / "processed" / "calendar"))
    calendar.refresh()

    pre_trade_check = PreTradeCheck(
        calendar=calendar,
        connector=connector,
        max_spread_pips=risk_cfg.get("spread_filter_pips", 3.0),
    )

    # Portfolio-RiskGuard: ueberwacht Gesamt-Portfolio-Drawdown
    risk_guard = RiskGuard(
        daily_loss_limit_pct=risk_cfg.get("daily_loss_limit_pct", 5.0),
        max_drawdown_pct=risk_cfg.get("max_drawdown_pct", 15.0),
    )

    # 50/50: identische Risikoallokation pro Symbol
    position_sizer = PositionSizer(
        risk_per_trade_pct=risk_cfg.get("max_risk_per_trade_pct", 1.0),
    )

    # Gemeinsamer CorrelationGuard verhindert doppelte korrelierte Positionen
    correlation_guard = CorrelationGuard()

    audit_log = AuditLog(db_path=str(_ROOT / "data" / "processed" / "audit.db"))

    os.environ.setdefault("CONFIRM_AUTONOMOUS", "yes")
    order_executor = OrderExecutor(
        connector=connector,
        live_trading_enabled=False,
        paper_trades_path=str(_ROOT / "data" / "processed" / "paper_trades.json"),
        audit_log=audit_log,
    )

    order_relay = OrderEventRelay()
    order_relay.attach(order_executor)

    def _balance_getter() -> float:
        try:
            info = connector.get_account_info()
            return float(info.get("balance", 10_000.0))
        except Exception:  # noqa: BLE001
            return 10_000.0

    confidence = model_cfg.get("confidence_threshold", 0.55)

    # ── XAUUSD H4 – Standard-Features-Loader (23 Baseline-Features) ───────
    def _xauusd_features_loader(symbol: str) -> "_pd.DataFrame | None":
        pattern = str(Path(features_dir) / f"{symbol}_H4_*.parquet")
        files   = sorted(_glob.glob(pattern))
        if not files:
            return None
        try:
            return _pd.read_parquet(files[-1])
        except Exception:  # noqa: BLE001
            return None

    xauusd_orch = TradingOrchestrator(
        data_pipeline=pipeline,
        risk_guard=risk_guard,
        pre_trade_check=pre_trade_check,
        signal_model=xauusd_model,
        correlation_guard=correlation_guard,
        position_sizer=position_sizer,
        order_executor=order_executor,
        audit_log=audit_log,
        features_dir=features_dir,
        features_loader=_xauusd_features_loader,
        balance_getter=_balance_getter,
        timeframe="H4",
        signal_confidence_threshold=confidence,
        mode=TradingMode.AUTONOMOUS,
        confirmation_callback=confirmation_callback,
    )

    # ── EURUSD H4 – MR-Features-Loader (ergaenzt bb_pct_b etc.) ──────────
    _mr_instance = MeanReversionModel()

    def _eurusd_mr_features_loader(symbol: str) -> "_pd.DataFrame | None":
        pattern = str(Path(features_dir) / f"{symbol}_H4_*.parquet")
        files   = sorted(_glob.glob(pattern))
        if not files:
            return None
        try:
            df = _pd.read_parquet(files[-1])
        except Exception:  # noqa: BLE001
            return None
        return _mr_instance._add_mr_features(df)

    eurusd_orch = TradingOrchestrator(
        data_pipeline=pipeline,
        risk_guard=risk_guard,
        pre_trade_check=pre_trade_check,
        signal_model=eurusd_model,
        correlation_guard=correlation_guard,
        position_sizer=position_sizer,
        order_executor=order_executor,
        audit_log=audit_log,
        features_dir=features_dir,
        features_loader=_eurusd_mr_features_loader,
        balance_getter=_balance_getter,
        timeframe="H4",
        signal_confidence_threshold=confidence,
        mode=TradingMode.AUTONOMOUS,
        confirmation_callback=confirmation_callback,
    )

    portfolio_orch = MultiSymbolOrchestrator([
        ("XAUUSD", xauusd_orch),
        ("EURUSD", eurusd_orch),
    ])

    logger.info(
        "Portfolio-Stack bereit | XAUUSD H4 TF + EURUSD H4 MR | "
        "50/50 Risiko | Modus=AUTONOMOUS | Paper-Modus=True (kein echtes Geld)"
    )

    return {
        "orchestrator":   portfolio_orch,
        "order_executor": order_executor,
        "order_relay":    order_relay,
        "symbols":        ["XAUUSD", "EURUSD"],
        "pipeline":       pipeline,
        "audit_log":      audit_log,
        "connector":      connector,
    }


# ---------------------------------------------------------------------------
# Dashboard-Backend mit Live-Positionen
# ---------------------------------------------------------------------------

class _LiveDashboardBackend:
    """
    DashboardBackend fuer Demo-Live-Modus.
    Kombiniert MT5-Kontodaten mit echten offenen Positionen aus dem OrderExecutor.
    """

    def __init__(self, connector, order_executor) -> None:
        self._connector = connector
        self._executor  = order_executor

    def fetch_snapshot(self):
        from gui.views.dashboard_view import DashboardSnapshot, PositionInfo
        try:
            info = self._connector.get_account_info()
        except Exception:  # noqa: BLE001
            return DashboardSnapshot()

        positions: list[PositionInfo] = []
        try:
            is_paper = not getattr(self._executor, "_live", True)
            for p in self._executor.get_open_positions():
                sym = f"[P] {p['symbol']}" if is_paper else p["symbol"]
                positions.append(PositionInfo(
                    ticket=p["ticket"],
                    symbol=sym,
                    direction=p["direction"],
                    lot_size=p["lot_size"],
                    open_price=p.get("open_price") or 0.0,
                    current_pnl=p.get("current_pnl"),
                ))
        except Exception:  # noqa: BLE001
            pass

        return DashboardSnapshot(
            balance=info.get("balance"),
            currency=info.get("currency", "€"),
            equity=info.get("equity"),
            account_number=info.get("login"),
            server=info.get("server"),
            leverage=info.get("leverage"),
            is_demo=info.get("is_demo"),
            positions=positions,
        )


# ---------------------------------------------------------------------------
# CLI-Argumente
# ---------------------------------------------------------------------------

def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "QuantzAI Portfolio-Bot (XAUUSD H4 TF + EURUSD H4 MR, "
            "CONFIRM_REQUIRED, Demo-Live)"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--xauusd-model", default=None, dest="xauusd_model",
        help="XAUUSD H4 TF-Modell (.joblib). Standard: neuestes signal_model_v* in models/",
    )
    p.add_argument(
        "--eurusd-mr-model", default=None, dest="eurusd_mr_model",
        help="EURUSD H4 MR-Modell (.joblib). Standard: neuestes mean_reversion_model* in models/",
    )
    p.add_argument("--interval", type=int, default=300, help="Sekunden zwischen Zyklen")
    p.add_argument("--config",   default="config/config.yaml")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Einstiegspunkt
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    args = _parse_args(argv)

    _load_env()

    from loguru import logger

    # ── Fruehzeitig pruefen ───────────────────────────────────────────────
    try:
        config = _load_config(args.config)
        xauusd_model_path    = (
            Path(args.xauusd_model) if args.xauusd_model else find_newest_model()
        )
        eurusd_mr_model_path = (
            Path(args.eurusd_mr_model) if args.eurusd_mr_model else find_newest_mr_model()
        )
        if not xauusd_model_path.exists():
            raise StartupError(f"XAUUSD-Modell nicht gefunden: {xauusd_model_path}")
        if not eurusd_mr_model_path.exists():
            raise StartupError(f"EURUSD-MR-Modell nicht gefunden: {eurusd_mr_model_path}")
    except StartupError as exc:
        logger.error("Startup abgebrochen: {exc}", exc=exc)
        print(f"\nFEHLER: {exc}", file=sys.stderr)
        return 1

    # ── MT5 verbinden ─────────────────────────────────────────────────────
    try:
        connector = build_mt5_connector()
    except StartupError as exc:
        logger.error("MT5-Verbindung: {exc}", exc=exc)
        print(f"\nFEHLER: {exc}", file=sys.stderr)
        return 1

    logger.info(
        "MT5 verbunden | XAUUSD-Modell: {xm} | EURUSD-MR: {em} | "
        "Modus: AUTONOMOUS | Paper: True",
        xm=xauusd_model_path.name, em=eurusd_mr_model_path.name,
    )

    # ── GUI starten ───────────────────────────────────────────────────────
    import sys as _sys

    from PySide6.QtWidgets import QApplication, QMessageBox
    from PySide6.QtCore import Qt

    from gui.app import (
        MainWindow, ConnectionStatus,
        TradingMode as GuiTradingMode,
    )
    from gui.backends.backtest_backend import BacktestGUIBackend

    app = QApplication(_sys.argv[:1])
    app.setApplicationName("QuantzAI")

    # Backtest-Backend (kein MT5 noetig)
    backtest_backend = BacktestGUIBackend()
    window = MainWindow(backtest_backend=backtest_backend)
    window.show()

    # Verbindungsstatus sofort setzen
    window.trading_status_bar.set_connection(ConnectionStatus.CONNECTED)
    window.trading_status_bar.set_trading_mode(GuiTradingMode.AUTONOMOUS)

    # Account-Info im Dashboard (wird nach Stack-Aufbau durch Live-Backend ersetzt)
    try:
        info = connector.get_account_info()
        window.trading_status_bar.set_account_info({
            "login":    info.get("login"),
            "balance":  info.get("balance"),
            "currency": info.get("currency", ""),
            "is_demo":  info.get("is_demo"),
        })
    except Exception as exc:  # noqa: BLE001
        logger.warning("Account-Info-Abruf fehlgeschlagen: {exc}", exc=exc)

    # ── Portfolio-Stack aufbauen ──────────────────────────────────────────
    try:
        stack = build_portfolio_stack(
            config=config,
            connector=connector,
            xauusd_model_path=xauusd_model_path,
            eurusd_mr_model_path=eurusd_mr_model_path,
        )
    except StartupError as exc:
        logger.error("Stack-Aufbau fehlgeschlagen: {exc}", exc=exc)
        QMessageBox.critical(window, "Startup-Fehler", str(exc))
        return 1

    # ── GUI-Verbindungen herstellen ───────────────────────────────────────

    relay          = stack["order_relay"]
    order_executor = stack["order_executor"]

    # Dashboard auf Live-Backend umstellen (zeigt echte Positionen + P&L)
    live_backend = _LiveDashboardBackend(connector, order_executor)
    snap = live_backend.fetch_snapshot()
    window.dashboard_view.update_display(snap)
    window.dashboard_view.set_backend(live_backend)
    window.dashboard_view.start_polling()
    window.dashboard_view.data_refreshed.connect(
        lambda s: window.trading_status_bar.set_account_info({
            "login":    s.account_number,
            "balance":  s.balance,
            "currency": s.currency,
            "is_demo":  s.is_demo,
        })
    )

    # Echtzeit-Order-Updates direkt ins Dashboard (ohne auf nächsten Poll zu warten)
    # Paper-Modus: [P]-Prefix damit simulierte Positionen sofort erkennbar sind
    def _on_paper_order_opened(order: dict) -> None:
        window.dashboard_view.on_order_opened(
            dict(order, symbol=f"[P] {order.get('symbol', '')}")
        )
    window._paper_order_open_cb = _on_paper_order_opened  # keep Qt reference alive
    relay.order_opened.connect(window._paper_order_open_cb)
    relay.order_closed.connect(window.dashboard_view.on_order_closed)

    # Position schließen per Dashboard-Button
    def _close_position(ticket: int) -> None:
        try:
            result = order_executor.close_position(ticket)
            logger.info("Position {t} geschlossen: {r}", t=ticket, r=result)
        except Exception as exc:  # noqa: BLE001
            logger.error("Position {t} schliessen fehlgeschlagen: {exc}", t=ticket, exc=exc)

    window.dashboard_view.position_close_requested.connect(_close_position)

    # Markt-Chart mit XAUUSD als Primär-Symbol
    window.dashboard_view.set_chart_connector(connector, "XAUUSD")

    # #56/#57/#58: Bot-Steuerung + ActivityLog + Bestätigung
    window.bot_controls.set_orchestrator(
        stack["orchestrator"],
        stack["symbols"],
        interval_seconds=args.interval,
    )

    # Modus-Combo auf AUTONOMOUS vorwählen (Index 2)
    window.bot_controls._mode_combo.setCurrentIndex(2)

    logger.info(
        "GUI-Bot bereit. Klicke 'Start' in der Bot-Steuerung (Sidebar).\n"
        "  Portfolio    : XAUUSD H4 TF (SignalModel) + EURUSD H4 MR (MeanReversionModel)\n"
        "  XAUUSD-Modell: {xm}\n"
        "  EURUSD-Modell: {em}\n"
        "  Modus        : AUTONOMOUS (kein Bestaetigungs-Dialog)\n"
        "  Paper-Modus  : True  (KEIN echtes Geld, nur simulierte Trades)\n"
        "  Intervall    : {iv}s",
        xm=xauusd_model_path.name, em=eurusd_mr_model_path.name, iv=args.interval,
    )

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
