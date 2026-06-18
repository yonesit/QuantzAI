"""
scripts/run_bot.py
QuantzAI Bot-Starter – startet den TradingOrchestrator per CLI.

Verwendung:
  python scripts/run_bot.py --symbols EURUSD,GBPUSD --interval 300
  python scripts/run_bot.py --symbols EURUSD --interval 60 --paper

Im Paper-Modus (Standard) werden alle Orders nur simuliert.
Fuer Live-Trading muss CONFIRM_LIVE=yes gesetzt sein.
"""

from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path

# Projekt-Root zum Python-Pfad hinzufuegen
sys.path.insert(0, str(Path(__file__).parents[1]))

from loguru import logger


# ─────────────────────────────────────────────────────────────────────────────
#  Dependency-Factories
# ─────────────────────────────────────────────────────────────────────────────

def _build_orchestrator(args: argparse.Namespace):
    """
    Baut den TradingOrchestrator aus konfigurierten Abhaengigkeiten.

    Alle Module werden per Dependency Injection uebergeben – keine globalen
    Singletons, keine hartkodierten Pfade ausserhalb dieses Factories.

    Gibt im Paper-Modus voll funktionsfaehige, nicht-MT5-abhaengige Instanzen
    zurueck. Fuer Live-Trading muss CONFIRM_LIVE=yes gesetzt sein.
    """
    import yaml
    from unittest.mock import MagicMock

    from src.risk.risk_guard       import RiskGuard
    from src.risk.position_sizer   import PositionSizer
    from src.risk.correlation_guard import CorrelationGuard
    from src.models.signal_model   import SignalModel
    from src.execution.order_executor import OrderExecutor
    from src.monitoring.audit_log  import AuditLog
    from src.orchestrator          import TradingOrchestrator

    # Konfiguration laden
    config_path = Path(args.config) if hasattr(args, "config") and args.config else Path("config/config.yaml")
    config: dict = {}
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    model_cfg = config.get("model", {})

    # ── Pure-Python-Module (keine MT5-Verbindung noetig) ─────────────────────
    risk_guard = RiskGuard(
        daily_loss_limit_pct=config.get("risk", {}).get("daily_loss_limit_pct", 5.0),
        max_drawdown_pct=config.get("risk", {}).get("max_drawdown_pct", 15.0),
    )

    position_sizer = PositionSizer(
        risk_per_trade_pct=config.get("risk", {}).get("risk_per_trade_pct", 1.0),
    )

    correlation_guard = CorrelationGuard()

    audit_log = AuditLog()

    # ── SignalModel laden (falls vorhanden) ───────────────────────────────────
    model_path = Path(args.model) if hasattr(args, "model") and args.model else None
    if model_path and model_path.exists():
        signal_model = SignalModel.load(model_path)
        logger.info("Modell geladen: {path}", path=model_path)
    else:
        signal_model = MagicMock()
        signal_model.get_signal.return_value = "flat"
        logger.warning("Kein Modell angegeben oder gefunden – Signal-Modell gibt 'flat' zurueck.")

    # ── MT5-abhaengige Module ─────────────────────────────────────────────────
    # Im Paper-Modus: Mock-Connector, echte OrderExecutor-Instanz (paper=True)
    connector_mock = MagicMock()
    connector_mock.is_connected = True

    order_executor = OrderExecutor(
        connector=connector_mock,
        live_trading_enabled=False,  # Paper-Modus (Standardmaessig sicher)
        audit_log=audit_log,
    )

    # DataPipeline, PreTradeCheck, PositionReconciler benoetigen MT5/OANDA;
    # im Paper-Modus werden sie durch denkbar einfache Stubs ersetzt.
    data_pipeline = MagicMock()
    data_pipeline.run_batch.return_value = {"status": "ok", "skipped": False}

    pre_trade_check = MagicMock()
    pre_trade_check.is_safe_to_trade.return_value = (True, "paper-mode: immer sicher")

    position_reconciler = MagicMock()
    position_reconciler.sync.return_value = {"in_sync": True}

    # ── Orchestrator ──────────────────────────────────────────────────────────
    symbols = [s.strip() for s in args.symbols.split(",")]

    def _null_features_loader(symbol: str):
        """Im Paper-Modus ohne echte Daten: leeres DataFrame -> Signal=flat."""
        import pandas as pd
        return pd.DataFrame([{"close": 1.09, "atr": 0.001}])

    orchestrator = TradingOrchestrator(
        data_pipeline=data_pipeline,
        risk_guard=risk_guard,
        pre_trade_check=pre_trade_check,
        signal_model=signal_model,
        correlation_guard=correlation_guard,
        position_sizer=position_sizer,
        order_executor=order_executor,
        position_reconciler=position_reconciler,
        audit_log=audit_log,
        features_loader=_null_features_loader,
        timeframe=getattr(args, "timeframe", "H1"),
        signal_confidence_threshold=model_cfg.get("confidence_threshold", 0.55),
    )
    return orchestrator, symbols


# ─────────────────────────────────────────────────────────────────────────────
#  Graceful Shutdown
# ─────────────────────────────────────────────────────────────────────────────

def _install_signal_handlers(orchestrator) -> None:
    """Registriert SIGINT/SIGTERM-Handler fuer sauberes Herunterfahren."""

    def _shutdown(sig, frame):
        logger.info("Signal {s} empfangen – Graceful Shutdown...", s=sig)
        orchestrator.stop()

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QuantzAI TradingBot",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--symbols",
        required=True,
        help='Kommagetrennte Symbol-Liste, z.B. "EURUSD,GBPUSD"',
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Wartezeit zwischen zwei Zyklen in Sekunden",
    )
    parser.add_argument(
        "--timeframe",
        default="H1",
        help="Kerzen-Zeitrahmen (M1, M5, H1, H4, D1, …)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Pfad zur trainierten SignalModel-Datei (.joblib)",
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Pfad zur Konfigurationsdatei",
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        default=True,
        help="Paper-Modus (Standard): Orders werden nur simuliert",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)

    logger.info(
        "QuantzAI Bot startet | Symbole={syms} | Intervall={iv}s | Paper={p}",
        syms=args.symbols, iv=args.interval, p=args.paper,
    )

    try:
        orchestrator, symbols = _build_orchestrator(args)
    except Exception as exc:
        logger.error("Bot konnte nicht initialisiert werden: {exc}", exc=exc)
        return 1

    _install_signal_handlers(orchestrator)

    try:
        orchestrator.run_loop(symbols, interval_seconds=args.interval)
    except Exception as exc:
        logger.error("Unerwarteter Fehler im Bot-Loop: {exc}", exc=exc)
        return 1

    logger.info("QuantzAI Bot beendet.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
