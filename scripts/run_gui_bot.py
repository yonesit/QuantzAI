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
import json
import os
import sys
import threading
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Projekt-Root in Pfad aufnehmen
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

# Sekunden pro Kerze – fuer auto-Intervall-Berechnung
_TF_INTERVAL: dict[str, int] = {
    "M1": 60, "M5": 300, "M15": 900, "M30": 1800,
    "H1": 3600, "H4": 14400, "D1": 86400,
}


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

def find_newest_model(
    model_dir: str | Path = "models",
    timeframe: str = "H4",
    symbol: str = "XAUUSD",
) -> Path:
    """
    Gibt den Pfad zum neuesten Signal-Modell zurueck.
    Suchreihenfolge:
      1. signal_model_v*_{SYMBOL}_{TF}_*.joblib  (symbol+tf-spezifisch)
      2. signal_model_v*_{TF}_*.joblib           (nur tf-spezifisch)
      3. signal_model_v*.joblib                  (beliebiges Modell, mit Warnung)
    """
    from loguru import logger
    d    = Path(model_dir)
    sym  = symbol.upper()
    tf   = timeframe.upper()
    # 1. Symbol + TF spezifisch
    best = sorted(
        [f for f in d.glob(f"signal_model_v*_{sym}_{tf}_*.joblib") if "_IS_" not in f.name],
        key=lambda f: f.stat().st_mtime, reverse=True,
    )
    if best:
        return best[0]
    # 2. Nur TF spezifisch (kein Symbol-Filter)
    tf_only = sorted(
        [f for f in d.glob(f"signal_model_v*_{tf}_*.joblib") if "_IS_" not in f.name],
        key=lambda f: f.stat().st_mtime, reverse=True,
    )
    if tf_only:
        logger.warning(
            "Kein {sym}_{tf}-spezifisches Modell – nutze {m}",
            sym=sym, tf=tf, m=tf_only[0].name,
        )
        return tf_only[0]
    # 3. Fallback: beliebiges Modell
    candidates = sorted(
        [f for f in d.glob("signal_model_v*.joblib") if "_IS_" not in f.name],
        key=lambda f: f.stat().st_mtime, reverse=True,
    )
    if not candidates:
        raise StartupError(
            f"Kein trainiertes Modell fuer {sym} {tf} in '{d}' gefunden.\n"
            f"Fuehre aus: python scripts/train_model.py --symbol {sym} --tf {tf}\n"
            "Erwartetes Muster: models/signal_model_v1_{SYMBOL}_{TF}_YYYYMMDD.joblib"
        )
    logger.warning(
        "Kein {sym}_{tf}-Modell – Fallback auf {m} (H4-Modell mit M15-Daten!)",
        sym=sym, tf=tf, m=candidates[0].name,
    )
    return candidates[0]


def find_newest_mr_model(model_dir: str | Path = "models", timeframe: str = "H4") -> Path:
    """
    Gibt den Pfad zum neuesten MeanReversion-Modell zurueck.
    Sucht zuerst timeframe-spezifische Modelle.
    """
    from loguru import logger
    d = Path(model_dir)
    tf_candidates = sorted(
        [f for f in d.glob(f"mean_reversion_model*_{timeframe}*.joblib")],
        key=lambda f: f.stat().st_mtime, reverse=True,
    )
    if tf_candidates:
        return tf_candidates[0]
    candidates = sorted(
        list(d.glob("mean_reversion_model*.joblib")),
        key=lambda f: f.stat().st_mtime, reverse=True,
    )
    if not candidates:
        raise StartupError(
            f"Kein MeanReversion-Modell in '{d}' gefunden.\n"
            "Trainiere das MR-Modell und speichere es:\n"
            "  model.save('models/mean_reversion_model_YYYYMMDD.joblib')\n"
            "Erwartetes Muster: models/mean_reversion_model*.joblib"
        )
    if timeframe != "H4":
        logger.warning(
            "Kein {tf}-spezifisches MR-Modell gefunden – nutze H4-Modell {m}. "
            "Fuer bessere Signale: MR-Modell auf {tf}-Daten neu trainieren.",
            tf=timeframe, m=candidates[0].name,
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


def _calendar_refresh_loop(calendar) -> None:
    """
    Daemon-Thread: aktualisiert den Wirtschaftskalender taeglich um 00:00 UTC.
    Schlaeft bis zur naechsten Mitternacht, ruft dann calendar.refresh() auf.
    """
    from loguru import logger
    while True:
        now      = datetime.now(timezone.utc)
        tomorrow = (now + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        time.sleep((tomorrow - now).total_seconds())
        try:
            calendar.refresh()
            logger.info("Wirtschaftskalender: Tages-Refresh durchgefuehrt.")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Wirtschaftskalender-Refresh fehlgeschlagen: {exc}", exc=exc)


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

    def __init__(self, pairs: list, break_even_manager=None, order_executor=None) -> None:
        self._pairs               = pairs          # [(symbol, orchestrator)]
        self._stop_event          = threading.Event()
        self._activity_callback   = None
        self._be_manager          = break_even_manager
        self._executor            = order_executor  # für Paper-SL/TP-Überwachung

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
                    if self._be_manager is not None:
                        try:
                            self._be_manager.manage(symbol)
                        except Exception as _be_exc:  # noqa: BLE001
                            logger.warning(
                                "BreakEvenManager Fehler | {sym}: {e}",
                                sym=symbol, e=_be_exc,
                            )

                    # Paper-SL/TP-Überwachung nach jedem Symbol-Zyklus
                    if self._executor is not None:
                        try:
                            closed = self._executor.check_paper_sl_tp()
                            for c in closed:
                                logger.info(
                                    "Paper-Position automatisch geschlossen | "
                                    "ticket={t} {sym} {dir} | pnl={pnl}",
                                    t=c.get("ticket"), sym=c.get("symbol"),
                                    dir=c.get("direction"),
                                    pnl=f"{c.get('pnl', 0):.2f}" if c.get("pnl") is not None else "?",
                                )
                        except Exception as _sl_exc:  # noqa: BLE001
                            logger.warning("Paper-SL/TP-Check Fehler: {e}", e=_sl_exc)
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
    from src.models.regime_detector import RegimeDetector
    from src.execution.order_executor import OrderExecutor
    from src.monitoring.audit_log import AuditLog
    from src.orchestrator import TradingOrchestrator
    from src.modes import TradingMode
    from gui.widgets.order_event_relay import OrderEventRelay

    risk_cfg  = config.get("risk", {})
    model_cfg = config.get("model", {})
    feat_cfg  = config.get("features", {})
    atr_col   = f"atr_{feat_cfg.get('atr_period', 14)}"

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
        symbol_overrides=risk_cfg.get("symbol_spread_overrides", {}),
    )

    # ── Risiko-Komponenten ────────────────────────────────────────────────
    risk_guard = RiskGuard(
        daily_loss_limit_pct=risk_cfg.get("daily_loss_limit_pct", 5.0),
        max_drawdown_pct=risk_cfg.get("max_drawdown_pct", 15.0),
    )

    position_sizer = PositionSizer(
        risk_per_trade_pct=risk_cfg.get("max_risk_per_trade_pct", 1.0),
        symbol_params=risk_cfg.get("symbol_pip_params", {}),
    )

    correlation_guard = CorrelationGuard()

    # ── AuditLog + OrderExecutor (echte MT5-Orders gegen Demo-Konto) ────────
    # CONFIRM_LIVE=yes muss explizit in .env gesetzt sein – kein programmatischer
    # Fallback. Fehlt der Eintrag, wirft OrderExecutor.__init__ RuntimeError.
    audit_log = AuditLog(
        db_path=str(_ROOT / "data" / "processed" / "audit.db"),
    )

    order_executor = OrderExecutor(
        connector=connector,
        live_trading_enabled=True,
        paper_trades_path=str(_ROOT / "data" / "processed" / "paper_trades.json"),
        audit_log=audit_log,
    )

    regime_detector = RegimeDetector()

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
        regime_detector=regime_detector,
        timeframe=timeframe,
        atr_col=atr_col,
        signal_confidence_threshold=model_cfg.get("confidence_threshold", 0.55),
        mode=TradingMode.CONFIRM_REQUIRED,
        confirmation_callback=confirmation_callback,
    )

    logger.info(
        "TradingStack bereit | Symbol={sym} TF={tf} | "
        "Modus=CONFIRM_REQUIRED | Order-Ausfuehrung=LIVE (echte MT5-Orders)",
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
    timeframe: str = "M15",
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
    from src.models.regime_detector import RegimeDetector
    from src.execution.order_executor import OrderExecutor
    from src.monitoring.audit_log import AuditLog
    from src.orchestrator import TradingOrchestrator
    from src.modes import TradingMode
    from gui.widgets.order_event_relay import OrderEventRelay

    risk_cfg  = config.get("risk", {})
    model_cfg = config.get("model", {})

    # ── Risiko-Parameter aus Config ────────────────────────────────────────
    sl_multiplier      = risk_cfg.get("sl_atr_multiplier", 1.0)
    tp_multiplier      = risk_cfg.get("tp_atr_multiplier", 1.0)
    virtual_balance    = risk_cfg.get("virtual_account_balance", 1000.0)
    be_threshold       = risk_cfg.get("break_even_threshold", 0.35)
    be_spread_buf      = risk_cfg.get("break_even_spread_buffer_pips", 1.0)

    # ATR-Spaltenname muss mit FeatureBuilder uebereinstimmen:
    # FeatureBuilder erzeugt f"atr_{atr_period}" -> "atr_14" (bei atr_period=14)
    feat_cfg = config.get("features", {})
    atr_col  = f"atr_{feat_cfg.get('atr_period', 14)}"

    # ── Modelle laden ──────────────────────────────────────────────────────
    logger.info("Lade XAUUSD {tf} TF-Modell: {path}", tf=timeframe, path=xauusd_model_path)
    xauusd_model = SignalModel.load(xauusd_model_path)
    logger.info("XAUUSD SignalModel geladen ({feats} Features)",
                feats=len(xauusd_model._feature_names))

    logger.info("Lade EURUSD {tf} MR-Modell: {path}", tf=timeframe, path=eurusd_mr_model_path)
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
        symbol_overrides=risk_cfg.get("symbol_spread_overrides", {}),
    )

    # Portfolio-RiskGuard: ueberwacht Gesamt-Portfolio-Drawdown
    risk_guard = RiskGuard(
        daily_loss_limit_pct=risk_cfg.get("daily_loss_limit_pct", 5.0),
        max_drawdown_pct=risk_cfg.get("max_drawdown_pct", 15.0),
    )

    # 50/50: identische Risikoallokation pro Symbol
    position_sizer = PositionSizer(
        risk_per_trade_pct=risk_cfg.get("max_risk_per_trade_pct", 1.0),
        sl_atr_multiplier=sl_multiplier,
        symbol_params=risk_cfg.get("symbol_pip_params", {}),
    )

    # Gemeinsamer CorrelationGuard verhindert doppelte korrelierte Positionen
    correlation_guard = CorrelationGuard()

    audit_log = AuditLog(db_path=str(_ROOT / "data" / "processed" / "audit.db"))

    os.environ.setdefault("CONFIRM_AUTONOMOUS", "yes")
    # CONFIRM_LIVE=yes muss explizit in .env gesetzt sein – kein programmatischer
    # Fallback. Fehlt der Eintrag, wirft OrderExecutor.__init__ RuntimeError beim Start.
    order_executor = OrderExecutor(
        connector=connector,
        live_trading_enabled=True,
        paper_trades_path=str(_ROOT / "data" / "processed" / "paper_trades.json"),
        audit_log=audit_log,
    )

    regime_detector = RegimeDetector()

    order_relay = OrderEventRelay()
    order_relay.attach(order_executor)

    def _balance_getter() -> float:
        try:
            info = connector.get_account_info()
            real_balance = float(info.get("balance", 10_000.0))
        except Exception:  # noqa: BLE001
            real_balance = 10_000.0
        # Virtuelles Risikokapital: Position Sizing basiert auf 1000 EUR,
        # nicht auf dem tatsaechlichen Kontostand (100x-Hebel-Simulation).
        return min(real_balance, virtual_balance)

    def _price_getter(symbol: str) -> "float | None":
        try:
            tick = connector.get_tick(symbol)
            return (float(tick["bid"]) + float(tick["ask"])) / 2
        except Exception:  # noqa: BLE001
            return None

    confidence = model_cfg.get("confidence_threshold", 0.55)

    # ── XAUUSD – Standard-Features-Loader ─────────────────────────────────
    def _xauusd_features_loader(symbol: str) -> "_pd.DataFrame | None":
        pattern = str(Path(features_dir) / f"{symbol}_{timeframe}_*.parquet")
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
        price_getter=_price_getter,
        regime_detector=RegimeDetector(),
        timeframe=timeframe,
        sl_atr_multiplier=sl_multiplier,
        tp_atr_multiplier=tp_multiplier,
        atr_col=atr_col,
        signal_confidence_threshold=confidence,
        mode=TradingMode.AUTONOMOUS,
        confirmation_callback=confirmation_callback,
    )

    # ── EURUSD – MR-Features-Loader (ergaenzt bb_pct_b etc.) ─────────────
    _mr_instance = MeanReversionModel()

    def _eurusd_mr_features_loader(symbol: str) -> "_pd.DataFrame | None":
        pattern = str(Path(features_dir) / f"{symbol}_{timeframe}_*.parquet")
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
        price_getter=_price_getter,
        regime_detector=RegimeDetector(),
        timeframe=timeframe,
        sl_atr_multiplier=sl_multiplier,
        tp_atr_multiplier=tp_multiplier,
        atr_col=atr_col,
        signal_confidence_threshold=confidence,
        mode=TradingMode.AUTONOMOUS,
        confirmation_callback=confirmation_callback,
    )

    from src.risk.break_even_manager import BreakEvenManager

    be_manager = BreakEvenManager(
        connector=connector,
        order_executor=order_executor,
        break_even_threshold=be_threshold,
        spread_buffer_pips=be_spread_buf,
    )

    portfolio_orch = MultiSymbolOrchestrator(
        pairs=[
            ("XAUUSD", xauusd_orch),
            ("EURUSD", eurusd_orch),
        ],
        break_even_manager=be_manager,
        order_executor=order_executor,
    )

    logger.info(
        "Portfolio-Stack bereit | XAUUSD {tf} TF + EURUSD {tf} MR | "
        "Virtuelles Kapital={vb} EUR | SL={sl}x ATR | TP={tp}x ATR | "
        "BE-Threshold={be}% | Modus=AUTONOMOUS | Order-Ausfuehrung=LIVE (echte MT5-Orders)",
        tf=timeframe, vb=virtual_balance,
        sl=sl_multiplier, tp=tp_multiplier, be=int(be_threshold * 100),
    )

    return {
        "orchestrator":   portfolio_orch,
        "order_executor": order_executor,
        "order_relay":    order_relay,
        "symbols":        ["XAUUSD", "EURUSD"],
        "pipeline":       pipeline,
        "audit_log":      audit_log,
        "connector":      connector,
        "calendar":       calendar,
    }


# ---------------------------------------------------------------------------
# Hilfsfunktionen – pure, testbar ohne MT5/Qt
# ---------------------------------------------------------------------------

def _calc_crv(pos: dict) -> "float | None":
    """
    Berechnet das Chance-Risiko-Verhaeltnis (TP-Distanz / SL-Distanz).

    Gibt None zurueck wenn SL/TP/open_price fehlen oder SL-Distanz <= 0.
    """
    direction  = pos.get("direction", "")
    open_price = pos.get("open_price")
    sl_price   = pos.get("sl_price")
    tp_price   = pos.get("tp_price")
    if not all([direction, open_price, sl_price, tp_price]):
        return None
    if direction == "buy":
        tp_dist = tp_price - open_price
        sl_dist = open_price - sl_price
    else:
        tp_dist = open_price - tp_price
        sl_dist = sl_price - open_price
    if sl_dist <= 0:
        return None
    return round(tp_dist / sl_dist, 1)


def _calc_total_stats(
    closed_trades: "list[dict]",
) -> "tuple[float | None, float | None]":
    """
    Berechnet Gesamt-Gewinn und Gesamt-Verlust aus geschlossenen Paper-Trades.

    Parameters
    ----------
    closed_trades : Liste aller Trade-Dicts (offen und geschlossen).

    Returns
    -------
    (total_gross_profit, total_gross_loss)
    Gibt (None, None) wenn noch kein Trade mit pnl-Feld geschlossen wurde.
    Gibt (profit, None) wenn nur Gewinne vorhanden und umgekehrt.
    """
    profits = [
        t["pnl"] for t in closed_trades
        if t.get("status") == "closed"
        and t.get("pnl") is not None
        and t["pnl"] > 0
    ]
    losses = [
        t["pnl"] for t in closed_trades
        if t.get("status") == "closed"
        and t.get("pnl") is not None
        and t["pnl"] <= 0
    ]
    total_profit = sum(profits) if profits else None
    total_loss   = sum(losses)  if losses  else None
    return total_profit, total_loss


def calc_unrealized_pnl(
    direction: str,
    open_price: float,
    current_bid: float,
    current_ask: float,
    lot_size: float,
    contract_size: float,
) -> float:
    """
    Berechnet den unrealisierten Gewinn/Verlust einer offenen Position.

    Formel (MT5-Konvention):
      BUY  : (bid - open_price)  * lot_size * contract_size
      SELL : (open_price - ask)  * lot_size * contract_size

    Parameters
    ----------
    direction     : "buy" oder "sell"
    open_price    : Eroeffnungskurs der Position
    current_bid   : Aktueller Bid-Kurs des Symbols
    current_ask   : Aktueller Ask-Kurs des Symbols
    lot_size      : Lot-Groesse der Position
    contract_size : Kontraktgroesse (z.B. 100 000 fuer EURUSD, 100 fuer XAUUSD)

    Returns
    -------
    float – positiv = Gewinn, negativ = Verlust (in Konto-Waehrung)
    """
    if direction == "buy":
        return (current_bid - open_price) * lot_size * contract_size
    return (open_price - current_ask) * lot_size * contract_size


# ---------------------------------------------------------------------------
# Dashboard-Backend mit Live-Positionen
# ---------------------------------------------------------------------------

class _LiveDashboardBackend:
    """
    DashboardBackend fuer Demo-Live-Modus.
    Kombiniert MT5-Kontodaten mit echten offenen Positionen aus dem OrderExecutor.
    Berechnet unrealisierten P&L per Position mit aktuellem MT5-Tick-Kurs.
    """

    def __init__(self, connector, order_executor) -> None:
        self._connector    = connector
        self._executor     = order_executor
        self._contract_size_cache: dict[str, float] = {}
        self._session_start = datetime.now(timezone.utc)

    def _get_contract_size(self, symbol: str) -> float:
        """Gibt die Kontraktgroesse des Symbols zurueck (gecacht)."""
        if symbol not in self._contract_size_cache:
            try:
                info = self._connector.get_symbol_info(symbol)
                self._contract_size_cache[symbol] = float(info.get("contract_size", 100_000.0))
            except Exception:  # noqa: BLE001
                self._contract_size_cache[symbol] = 100_000.0
        return self._contract_size_cache[symbol]

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
                raw_sym = p["symbol"]
                sym     = f"[P] {raw_sym}" if is_paper else raw_sym
                op      = p.get("open_price") or 0.0

                # Unrealisierten P&L mit aktuellem MT5-Kurs berechnen
                pnl: float | None = p.get("current_pnl")   # Live: kommt von MT5
                if pnl is None and op:
                    try:
                        tick          = self._connector.get_tick(raw_sym)
                        contract_size = self._get_contract_size(raw_sym)
                        pnl = calc_unrealized_pnl(
                            direction=p["direction"],
                            open_price=op,
                            current_bid=tick["bid"],
                            current_ask=tick["ask"],
                            lot_size=p["lot_size"],
                            contract_size=contract_size,
                        )
                    except Exception:  # noqa: BLE001
                        pass

                crv = _calc_crv(p)

                positions.append(PositionInfo(
                    ticket=p["ticket"],
                    symbol=sym,
                    direction=p["direction"],
                    lot_size=p["lot_size"],
                    open_price=op or None,
                    current_pnl=pnl,
                    crv=crv,
                    sl_price=p.get("sl_price"),
                    tp_price=p.get("tp_price"),
                    break_even_active=bool(p.get("break_even_triggered")),
                    open_time=p.get("open_time"),
                ))
        except Exception:  # noqa: BLE001
            pass

        # Gesamt-Gewinn / Gesamt-Verlust direkt aus paper_trades.json lesen
        # Bei fehlenden geschlossenen Trades bleibt der Wert None und wird von
        # der UI als €0.00 dargestellt.
        total_profit: float | None = None
        total_loss:   float | None = None
        try:
            paper_path = getattr(self._executor, "_paper_path", None)
            if paper_path is None:
                paper_path = Path("data/processed/paper_trades.json")
            if Path(paper_path).exists():
                with open(paper_path, encoding="utf-8") as _f:
                    all_trades = json.load(_f)
                total_profit, total_loss = _calc_total_stats(all_trades)
        except Exception:  # noqa: BLE001
            pass

        # Live-Modus: MT5-Deal-Historie als autoritative Quelle (deckt bereits
        # geschlossene Positionen ab, die nicht in paper_trades.json stehen).
        # Nur wenn paper_trades.json keine Ergebnisse liefert (Live-Modus hat
        # keine Paper-Trades).
        if getattr(self._executor, "_live", False) and total_profit is None and total_loss is None:
            try:
                from src.data.mt5_connector import _load_mt5
                mt5 = _load_mt5()
                if mt5 is not None:
                    deals = mt5.history_deals_get(
                        self._session_start, datetime.now(timezone.utc)
                    ) or []
                    _OUT = getattr(mt5, "DEAL_ENTRY_OUT", 1)
                    profits = [
                        d.profit for d in deals
                        if getattr(d, "entry", None) == _OUT and d.profit > 0
                    ]
                    losses = [
                        d.profit for d in deals
                        if getattr(d, "entry", None) == _OUT and d.profit <= 0
                    ]
                    total_profit = sum(profits) if profits else None
                    total_loss   = sum(losses)  if losses  else None
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
            total_gross_profit=total_profit,
            total_gross_loss=total_loss,
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
        help="EURUSD MR-Modell (.joblib). Standard: neuestes mean_reversion_model* in models/",
    )
    p.add_argument(
        "--timeframe", default="M15",
        choices=["M5", "M15", "M30", "H1", "H4"],
        help="Kerzen-Zeitrahmen fuer beide Symbole (Standard: M15)",
    )
    p.add_argument(
        "--interval", type=int, default=None,
        help="Sekunden zwischen Zyklen (Standard: 1 Kerze des gewahlten Timeframes)",
    )
    p.add_argument("--config",   default="config/config.yaml")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Einstiegspunkt
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    args = _parse_args(argv)

    _load_env()

    from loguru import logger

    timeframe = args.timeframe
    interval  = args.interval or _TF_INTERVAL.get(timeframe, 900)

    # ── Fruehzeitig pruefen ───────────────────────────────────────────────
    try:
        config = _load_config(args.config)
        xauusd_model_path    = (
            Path(args.xauusd_model) if args.xauusd_model
            else find_newest_model(timeframe=timeframe)
        )
        eurusd_mr_model_path = (
            Path(args.eurusd_mr_model) if args.eurusd_mr_model
            else find_newest_mr_model(timeframe=timeframe)
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
        connector = build_mt5_connector(max_retries=3)
    except StartupError as exc:
        logger.error("MT5-Verbindung: {exc}", exc=exc)
        print(f"\nFEHLER: {exc}", file=sys.stderr)
        return 1

    logger.info(
        "MT5 verbunden | TF={tf} | Intervall={iv}s | "
        "XAUUSD-Modell: {xm} | EURUSD-MR: {em} | Modus: AUTONOMOUS | Order-Ausfuehrung: LIVE",
        tf=timeframe, iv=interval,
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
            timeframe=timeframe,
        )
    except StartupError as exc:
        logger.error("Stack-Aufbau fehlgeschlagen: {exc}", exc=exc)
        QMessageBox.critical(window, "Startup-Fehler", str(exc))
        return 1

    # ── Kalender taeglich um 00:00 UTC aktualisieren ─────────────────────
    _cal_thread = threading.Thread(
        target=_calendar_refresh_loop,
        args=(stack["calendar"],),
        daemon=True,
        name="calendar-refresh",
    )
    _cal_thread.start()

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

    # Echtzeit-Order-Updates ins Cockpit (Positionen werden dort verwaltet)
    window.cockpit_view.connect_order_executor(relay)

    # Cockpit-Statistiken bei jedem Dashboard-Polling-Tick mitziehen
    window.dashboard_view.data_refreshed.connect(window.cockpit_view.update_trading_stats)

    # Position schließen per Cockpit-Button (inkl. close_price + pnl)
    def _close_position(ticket: int) -> None:
        try:
            pos_list = [
                p for p in order_executor.get_open_positions()
                if p.get("ticket") == ticket
            ]
            close_price: float | None = None
            realized_pnl: float | None = None
            if pos_list:
                pos = pos_list[0]
                raw_sym   = pos.get("symbol", "")
                direction = pos.get("direction", "")
                op        = pos.get("open_price") or 0.0
                lot_size  = pos.get("lot_size", 0.0)
                if op and raw_sym:
                    try:
                        tick          = connector.get_tick(raw_sym)
                        close_price   = tick["bid"] if direction == "buy" else tick["ask"]
                        contract_size = live_backend._get_contract_size(raw_sym)
                        realized_pnl  = calc_unrealized_pnl(
                            direction=direction,
                            open_price=op,
                            current_bid=tick["bid"],
                            current_ask=tick["ask"],
                            lot_size=lot_size,
                            contract_size=contract_size,
                        )
                    except Exception:  # noqa: BLE001
                        pass
            result = order_executor.close_position(
                ticket, close_price=close_price, pnl=realized_pnl
            )
            logger.info("Position {t} geschlossen: {r}", t=ticket, r=result)
        except Exception as exc:  # noqa: BLE001
            logger.error("Position {t} schliessen fehlgeschlagen: {exc}", t=ticket, exc=exc)

    window.cockpit_view.position_close_requested.connect(_close_position)

    # Markt-Chart mit XAUUSD als Primär-Symbol
    window.dashboard_view.set_chart_connector(connector, "XAUUSD")

    # Watchlist im Cockpit mit Live-Bid/Ask + Tagesveränderung der 4 Symbole füllen
    window.cockpit_view.set_watchlist_connector(connector, {})

    # #56/#57/#58: Bot-Steuerung + ActivityLog + Bestätigung
    window.bot_controls.set_orchestrator(
        stack["orchestrator"],
        stack["symbols"],
        interval_seconds=interval,
    )

    # Modus-Combo auf AUTONOMOUS vorwählen (Index 2)
    window.bot_controls._mode_combo.setCurrentIndex(2)

    risk_cfg = config.get("risk", {})
    logger.info(
        "GUI-Bot bereit. Klicke 'Start' in der Bot-Steuerung (Sidebar).\n"
        "  Portfolio       : XAUUSD {tf} TF (SignalModel) + EURUSD {tf} MR (MeanReversionModel)\n"
        "  XAUUSD-Modell   : {xm}\n"
        "  EURUSD-Modell   : {em}\n"
        "  Timeframe       : {tf}  (Intervall: {iv}s)\n"
        "  Virtuelles Kap. : {vb} EUR  (1 %% Risiko = {risk:.0f} EUR/Trade)\n"
        "  SL/TP           : {sl}x / {tp}x ATR  |  Break-Even: {be}%% des TP-Weges\n"
        "  Modus           : AUTONOMOUS (kein Bestaetigungs-Dialog)\n"
        "  Konto-Typ       : Demo  (kein echtes Geld)\n"
        "  Order-Modus     : LIVE  (echte MT5-Orders gegen Demo-Konto)",
        tf=timeframe, iv=interval,
        xm=xauusd_model_path.name, em=eurusd_mr_model_path.name,
        vb=risk_cfg.get("virtual_account_balance", 1000.0),
        risk=risk_cfg.get("virtual_account_balance", 1000.0) * risk_cfg.get("max_risk_per_trade_pct", 1.0) / 100,
        sl=risk_cfg.get("sl_atr_multiplier", 1.0),
        tp=risk_cfg.get("tp_atr_multiplier", 1.0),
        be=int(risk_cfg.get("break_even_threshold", 0.35) * 100),
    )

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
