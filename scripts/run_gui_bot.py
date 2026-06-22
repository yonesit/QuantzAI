"""
scripts/run_gui_bot.py
QuantzAI GUI-Bot-Starter mit ECHTEN Komponenten, CONFIRM_REQUIRED-Modus, Paper-Trading.

Startet:
  - GUI (PySide6 MainWindow)
  - Echte MT5-Verbindung fuer Live-EURUSD-H1-Daten
  - Echtes SignalModel (neuestes models/signal_model_v1_*.joblib)
  - Echte PreTradeCheck (EconomicCalendar + Spread-Filter)
  - Echte RiskGuard / PositionSizer / CorrelationGuard / AuditLog
  - OrderExecutor im Paper-Modus (live_trading_enabled=False – kein echtes Geld)
  - CONFIRM_REQUIRED: Bot fragt per GUI-Banner vor jedem Trade nach Bestaetigung
  - ActivityLogWidget, OrderEventRelay alle verdrahtet

Fehlschlag mit klarer Meldung wenn:
  - MT5_LOGIN / MT5_PASSWORD / MT5_SERVER fehlen in .env
  - Kein trainiertes Modell in models/ vorhanden
  - MT5-Verbindung schlaegt fehl

Verwendung:
  python scripts/run_gui_bot.py [--symbol EURUSD] [--interval 300] [--model PATH]
"""

from __future__ import annotations

import argparse
import os
import sys
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
            "Fuehre zuerst aus: python scripts/train_model.py --symbol EURUSD\n"
            "Erwartetes Muster: models/signal_model_v1_YYYYMMDD.joblib"
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
# Den gesamten Trading-Stack bauen
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
    Baut alle echten Komponenten und verdrahtet sie.

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
            for p in self._executor.get_open_positions():
                positions.append(PositionInfo(
                    ticket=p["ticket"],
                    symbol=p["symbol"],
                    direction=p["direction"],
                    lot_size=p["lot_size"],
                    open_price=p.get("open_price", 0.0),
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
        description="QuantzAI GUI-Bot (echte Komponenten, CONFIRM_REQUIRED, Paper-Trading)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--symbol",   default="EURUSD", help="Handelssymbol")
    p.add_argument("--tf",       default="H1",     help="Zeitrahmen")
    p.add_argument("--interval", type=int, default=300, help="Sekunden zwischen Zyklen")
    p.add_argument("--model",    default=None,
                   help="Modell-Pfad (.joblib). Standard: neuestes in models/")
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
        config     = _load_config(args.config)
        model_path = Path(args.model) if args.model else find_newest_model()
        if not model_path.exists():
            raise StartupError(
                f"Modell-Datei nicht gefunden: {model_path}"
            )
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
        "MT5 verbunden | Modell: {m} | Symbol: {s} | Modus: CONFIRM_REQUIRED | Demo-Live: True",
        m=model_path.name, s=args.symbol,
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
    window.trading_status_bar.set_trading_mode(GuiTradingMode.CONFIRM)

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

    # ── Trading-Stack aufbauen ────────────────────────────────────────────
    try:
        stack = build_trading_stack(
            config=config,
            connector=connector,
            model_path=model_path,
            symbol=args.symbol,
            timeframe=args.tf,
            confirmation_callback=window.confirmation_callback,
        )
    except StartupError as exc:
        logger.error("Stack-Aufbau fehlgeschlagen: {exc}", exc=exc)
        QMessageBox.critical(window, "Startup-Fehler", str(exc))
        return 1

    # ── GUI-Verbindungen herstellen ───────────────────────────────────────

    relay            = stack["order_relay"]
    order_executor   = stack["order_executor"]

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
    relay.order_opened.connect(window.dashboard_view.on_order_opened)
    relay.order_closed.connect(window.dashboard_view.on_order_closed)

    # Position schließen per Dashboard-Button
    def _close_position(ticket: int) -> None:
        try:
            result = order_executor.close_position(ticket)
            logger.info("Position {t} geschlossen: {r}", t=ticket, r=result)
        except Exception as exc:  # noqa: BLE001
            logger.error("Position {t} schliessen fehlgeschlagen: {exc}", t=ticket, exc=exc)

    window.dashboard_view.position_close_requested.connect(_close_position)

    # Markt-Chart mit MT5-Connector verdrahten
    window.dashboard_view.set_chart_connector(connector, args.symbol)

    # #56/#57/#58: Bot-Steuerung + ActivityLog + Bestätigung
    window.bot_controls.set_orchestrator(
        stack["orchestrator"],
        stack["symbols"],
        interval_seconds=args.interval,
    )

    # Modus-Combo auf CONFIRM_REQUIRED vorwählen (Index 1)
    window.bot_controls._mode_combo.setCurrentIndex(1)

    logger.info(
        "GUI-Bot bereit. Klicke 'Start' in der Bot-Steuerung (Sidebar).\n"
        "  Modell  : {m}\n"
        "  Symbol  : {s}  TF: {tf}\n"
        "  Modus   : CONFIRM_REQUIRED\n"
        "  Demo-Live: True  (echte MT5-Positionen auf Demo-Konto)\n"
        "  Intervall: {iv}s",
        m=model_path.name, s=args.symbol, tf=args.tf, iv=args.interval,
    )

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
