"""
src/shadow.py
Shadow-Mode – paralleler Betrieb ohne echte MT5-Orders.

Ablauf:
  1. ShadowOrderExecutor  – Fake-Executor: protokolliert Trades im Speicher,
                            sendet KEINE Orders an MT5.
  2. ShadowOrchestrator   – Wraps einen TradingOrchestrator, der mit einem
                            ShadowOrderExecutor konstruiert wurde.
                            Nach jedem Zyklus werden hypothetische Trades
                            in der shadow_trades-Tabelle der AuditLog gespeichert.

Parallel-Sicherheit:
  - ShadowOrderExecutor ruft mt5.order_send NIEMALS auf.
  - Jeder ShadowOrchestrator hat seine eigene Executor-Instanz (kein Shared-State).
  - AuditLog ist intern thread-safe (eigener Lock).
  - Der ShadowOrchestrator teilt keine veraenderlichen Objekte mit dem Live-Orchestrator.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any, Optional

from loguru import logger

from src.monitoring.audit_log import AuditLog
from src.orchestrator import TradingOrchestrator


# ─────────────────────────────────────────────────────────────────────────────
#  ShadowOrderExecutor
# ─────────────────────────────────────────────────────────────────────────────

class ShadowOrderExecutor:
    """
    Fake-OrderExecutor fuer den Shadow-Mode.

    Implementiert die gleiche Schnittstelle wie OrderExecutor, sendet aber
    KEINE Orders an MT5. Positionen werden nur im Arbeitsspeicher verfolgt.

    Wichtig: Dieser Executor darf AUSSCHLIESSLICH innerhalb eines
    ShadowOrchestrators verwendet werden.
    """

    def __init__(self) -> None:
        self._next_ticket: int = 1
        self._positions:   dict[int, dict] = {}

    # ── Oeffentliche Schnittstelle (kompatibel zu OrderExecutor) ──────────────

    def open_position(
        self,
        symbol:    str,
        direction: str,
        lot_size:  float,
        sl_price:  float,
        tp_price:  float,
    ) -> dict:
        """Legt eine hypothetische Position an. Kein MT5-Aufruf."""
        ticket = self._next_ticket
        self._next_ticket += 1

        trade: dict[str, Any] = {
            "ticket":     ticket,
            "symbol":     symbol,
            "direction":  direction,
            "lot_size":   lot_size,
            "sl_price":   sl_price,
            "tp_price":   tp_price,
            "open_price": None,
            "status":     "shadow_open",
        }
        self._positions[ticket] = trade

        logger.info(
            "[SHADOW] open_position (kein MT5) | ticket={t} {sym} {dir} {lot}L "
            "SL={sl} TP={tp}",
            t=ticket, sym=symbol, dir=direction, lot=lot_size,
            sl=sl_price, tp=tp_price,
        )
        return trade

    def close_position(self, ticket: int) -> dict:
        """Schliesst eine hypothetische Position."""
        pos = self._positions.get(ticket)
        if pos is None:
            from src.execution.order_executor import OrderError
            raise OrderError(f"Shadow-Position {ticket} nicht gefunden.")
        pos["status"] = "shadow_closed"
        pos["close_price"] = None
        logger.info("[SHADOW] close_position | ticket={t}", t=ticket)
        return dict(pos)

    def get_open_positions(self) -> list[dict]:
        """Gibt alle offenen Shadow-Positionen zurueck."""
        return [
            p for p in self._positions.values()
            if p.get("status") == "shadow_open"
        ]


# ─────────────────────────────────────────────────────────────────────────────
#  ShadowOrchestrator
# ─────────────────────────────────────────────────────────────────────────────

class ShadowOrchestrator:
    """
    Fuehrt einen TradingOrchestrator im Shadow-Mode aus.

    Der uebergebene ``orchestrator`` MUSS mit ``executor`` als order_executor
    konstruiert worden sein. So wird garantiert, dass kein MT5-Aufruf stattfindet.

    Nach jedem Zyklus in dem eine Order geoeffnet worden waere, wird der Trade
    in der ``shadow_trades``-Tabelle der AuditLog gespeichert.

    Parameters
    ----------
    orchestrator : TradingOrchestrator-Instanz, konstruiert mit ``executor``.
    executor     : ShadowOrderExecutor-Instanz – MUSS derselbe sein,
                   der dem orchestrator uebergeben wurde.
    audit_log    : AuditLog – shadow_trades werden hier gespeichert.
    label        : Bezeichner fuer diese Shadow-Instanz (z.B. "shadow_v2").
    """

    def __init__(
        self,
        orchestrator: TradingOrchestrator,
        executor:     ShadowOrderExecutor,
        audit_log:    AuditLog,
        label:        str = "shadow",
    ) -> None:
        if not isinstance(executor, ShadowOrderExecutor):
            raise TypeError(
                "executor muss ein ShadowOrderExecutor sein, damit keine echten "
                "MT5-Orders gesendet werden."
            )
        self._inner       = orchestrator
        self._executor    = executor
        self._audit_log   = audit_log
        self._label       = label
        self._stop_event  = threading.Event()

        logger.info(
            "ShadowOrchestrator: initialisiert | label={l}", l=label
        )

    # ── Oeffentliche Schnittstelle ────────────────────────────────────────────

    def run_cycle(self, symbol: str) -> dict:
        """
        Fuehrt einen Entscheidungszyklus aus.

        Falls der innere Orchestrator eine Order eroeffnen wuerde, wird die
        hypothetische Position mit Konfidenz in shadow_trades protokolliert.
        Kein echter MT5-Aufruf findet statt.

        Returns
        -------
        dict – identisches Format wie TradingOrchestrator.run_cycle().
        """
        result = self._inner.run_cycle(symbol)

        if result.get("action") in ("open_buy", "open_sell"):
            ticket = result.get("ticket")
            if ticket is not None:
                pos = self._executor._positions.get(ticket)
                if pos is not None:
                    shadow_trade = dict(pos)
                    shadow_trade["confidence"] = result.get("confidence")
                    shadow_trade["signal"]     = result.get("signal")
                    shadow_trade["label"]      = self._label
                    self._audit_log.log_shadow_trade(shadow_trade)
                    logger.info(
                        "ShadowOrchestrator: Trade protokolliert | label={l} "
                        "{sym} {act} conf={c}",
                        l=self._label,
                        sym=symbol,
                        act=result.get("action"),
                        c=result.get("confidence"),
                    )

        return result

    def run_loop(
        self,
        symbols:          list[str],
        interval_seconds: int = 300,
    ) -> None:
        """
        Fuehrt die Shadow-Schleife im aktuellen Thread aus.

        Kann sicher parallel zu einem Live-TradingOrchestrator in einem
        separaten Thread laufen. stop() beendet die Schleife sauber.
        """
        self._stop_event.clear()
        logger.info(
            "ShadowOrchestrator: Loop gestartet | label={l} | Symbole={s} | "
            "Intervall={iv}s",
            l=self._label, s=symbols, iv=interval_seconds,
        )

        while not self._stop_event.is_set():
            for symbol in symbols:
                if self._stop_event.is_set():
                    break
                try:
                    self.run_cycle(symbol)
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "ShadowOrchestrator: Fehler in run_cycle | label={l} "
                        "{sym}: {e}",
                        l=self._label, sym=symbol, e=exc,
                    )

            self._stop_event.wait(timeout=interval_seconds)

        logger.info(
            "ShadowOrchestrator: Loop beendet | label={l}", l=self._label
        )

    def stop(self) -> None:
        """Signalisiert run_loop() sauber zu beenden."""
        self._stop_event.set()
        logger.info(
            "ShadowOrchestrator: Stop-Signal gesetzt | label={l}", l=self._label
        )

    @property
    def label(self) -> str:
        return self._label

    # ── Vergleichs-Logik ──────────────────────────────────────────────────────

    def compare_performance(
        self,
        start_date,
        end_date,
    ) -> dict:
        """
        Vergleicht Shadow-Performance mit Live-Performance fuer denselben Zeitraum.

        Returns
        -------
        dict mit:
          n_shadow_trades       – Anzahl Shadow-Trades
          n_live_trades         – Anzahl echter Live-Trades (aus orders-Tabelle)
          shadow_avg_confidence – Durchschnittliche Signal-Konfidenz (Shadow)
          shadow_sharpe         – Geschaetzter Sharpe (None wenn < 2 Trades)
          period_start          – Periodenbeginn als String
          period_end            – Periodenende als String
        """
        shadow_df = self._audit_log.query_shadow_trades(
            start_date, end_date, label=self._label
        )
        live_df = self._audit_log.query_orders(start_date, end_date)

        n_shadow = len(shadow_df)
        n_live   = len(live_df)

        shadow_avg_conf: Optional[float] = None
        shadow_sharpe:   Optional[float] = None

        if n_shadow > 0 and "confidence" in shadow_df.columns:
            confs = shadow_df["confidence"].dropna()
            if len(confs) > 0:
                shadow_avg_conf = float(confs.mean())
            if len(confs) > 1:
                excess = confs - 0.5  # Ueberschuss-Konfidenz ueber Zufall
                std    = float(excess.std())
                if std > 0:
                    shadow_sharpe = float(
                        excess.mean() / std * (len(confs) ** 0.5)
                    )

        return {
            "n_shadow_trades":       n_shadow,
            "n_live_trades":         n_live,
            "shadow_avg_confidence": shadow_avg_conf,
            "shadow_sharpe":         shadow_sharpe,
            "period_start":          str(start_date),
            "period_end":            str(end_date),
        }

    def should_go_live(
        self,
        start_date,
        end_date,
        min_trades:          int   = 30,
        oos_sharpe_threshold: float = 0.5,
    ) -> tuple[bool, str]:
        """
        Empfehlung ob das Shadow-Modell live geschalten werden sollte.

        Kriterien:
          1. Mindestens ``min_trades`` Shadow-Trades im Zeitraum.
          2. Geschaetzter Sharpe > ``oos_sharpe_threshold``.

        Returns
        -------
        (True, begruendung)  wenn beide Kriterien erfuellt sind.
        (False, begruendung) sonst.
        """
        comparison  = self.compare_performance(start_date, end_date)
        n           = comparison["n_shadow_trades"]
        sharpe      = comparison["shadow_sharpe"]
        avg_conf    = comparison["shadow_avg_confidence"]

        problems: list[str] = []

        if n < min_trades:
            problems.append(f"Trades: {n}/{min_trades}")

        if sharpe is None:
            problems.append(
                "Sharpe nicht berechenbar (zu wenig Daten oder keine Konfidenz)"
            )
        elif sharpe <= oos_sharpe_threshold:
            problems.append(
                f"Sharpe {sharpe:.2f} <= Schwelle {oos_sharpe_threshold:.2f}"
            )

        if problems:
            return False, "Nicht bereit: " + "; ".join(problems)

        return True, (
            f"Bereit fuer Live-Schaltung: {n} Trades, "
            f"Sharpe={sharpe:.2f}, avg_confidence={avg_conf:.3f}"
        )
