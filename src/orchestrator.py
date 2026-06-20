"""
src/orchestrator.py
TradingOrchestrator – verbindet alle bestehenden Module zur Entscheidungsschleife.

Ablauf pro Zyklus:
  1. DataPipeline.run_batch()        – Daten holen, validieren, Features bauen
  2. RiskGuard.is_trading_allowed()  – Globale Handelssperre pruefen
  3. PreTradeCheck.is_safe_to_trade()– Spread + News-Filter
  4. SignalModel.get_signal()        – KI-Signal: 'long' | 'short' | 'flat'
  5. flat -> Zyklus beenden, kein Trade
  6. CorrelationGuard.can_open_position() – Korrelations-Filter
  7. PositionSizer.calculate_lot_size()   – ATR-basierte Lot-Groesse
  8. OrderExecutor.open_position()        – Order platzieren (Paper oder Live)
  9. PositionReconciler.sync()            – Periodischer Abgleich (optional)
 10. AuditLog                             – Jeder Schritt wird protokolliert

Keine eigene Geschaeftslogik – der Orchestrator ruft ausschliesslich
oeffentliche Methoden der bestehenden Module auf.
"""

from __future__ import annotations

import glob
import threading
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd
from loguru import logger

from src.modes import ConfirmationCallback, TradingMode, is_autonomous_allowed

# Timeframe -> Minuten pro Kerze
_TIMEFRAME_MINUTES: dict[str, int] = {
    "M1": 1, "M5": 5, "M15": 15, "M30": 30,
    "H1": 60, "H4": 240, "D1": 1440, "W1": 10080,
}


def _validate_mode_transition(mode: TradingMode) -> None:
    """Wirft EnvironmentError wenn AUTONOMOUS ohne CONFIRM_AUTONOMOUS=yes gesetzt wird."""
    if mode == TradingMode.AUTONOMOUS and not is_autonomous_allowed():
        raise EnvironmentError(
            "AUTONOMOUS-Modus erfordert Umgebungsvariable CONFIRM_AUTONOMOUS=yes."
        )


class TradingOrchestrator:
    """
    Verbindet alle bestehenden Module per Dependency Injection.

    Parameters
    ----------
    data_pipeline       : DataPipeline – holt und verarbeitet Marktdaten.
    risk_guard          : RiskGuard – ueberwacht Drawdown und Tageslimit.
    pre_trade_check     : PreTradeCheck – Spread- und News-Filter.
    signal_model        : SignalModel – KI-Signal-Erzeugung.
    correlation_guard   : CorrelationGuard – verhindert korrelierte Uebergewichtung.
    position_sizer      : PositionSizer – ATR-basierte Lot-Berechnung.
    order_executor      : OrderExecutor – Orderausfuehrung (Paper/Live).
    position_reconciler : PositionReconciler – optional, periodischer MT5-Abgleich.
    audit_log           : AuditLog – optional, SQLite-Protokoll.
    emergency_handler   : EmergencyHandler – optional, Notfallbehandlung.
    features_dir        : Verzeichnis der Feature-Parquet-Dateien.
    features_loader     : Injizierbare Funktion (symbol) -> pd.DataFrame | None.
                          Wenn None: sucht aktuellste Parquet-Datei in features_dir.
    balance_getter      : Injizierbare Funktion () -> float fuer aktuellen Kontostand.
                          Wenn None: wird 10000.0 als Fallback verwendet.
    timeframe           : Kerzen-Zeitrahmen (Standard: "H1").
    lookback_candles    : Anzahl Kerzen fuer den Datenabruf (Standard: 300).
    signal_confidence_threshold : KI-Abstinenzregel-Schwelle (Standard: 0.55).
    sl_atr_multiplier   : SL-Distanz = ATR * sl_atr_multiplier (Standard: 1.5).
    tp_atr_multiplier   : TP-Distanz = ATR * tp_atr_multiplier (Standard: 2.0).
    atr_col             : Spaltenname fuer ATR in den Features (Standard: "atr").
    close_col           : Spaltenname fuer Schlusskurs (Standard: "close").
    """

    def __init__(
        self,
        data_pipeline,
        risk_guard,
        pre_trade_check,
        signal_model,
        correlation_guard,
        position_sizer,
        order_executor,
        position_reconciler=None,
        audit_log=None,
        emergency_handler=None,
        features_dir: str = "data/features",
        features_loader: Optional[Callable[[str], Optional[pd.DataFrame]]] = None,
        balance_getter: Optional[Callable[[], float]] = None,
        timeframe: str = "H1",
        lookback_candles: int = 300,
        signal_confidence_threshold: float = 0.55,
        sl_atr_multiplier: float = 1.5,
        tp_atr_multiplier: float = 2.0,
        atr_col: str = "atr",
        close_col: str = "close",
        mode: TradingMode = TradingMode.SUGGEST_ONLY,
        confirmation_callback: Optional[ConfirmationCallback] = None,
    ) -> None:
        self._pipeline        = data_pipeline
        self._risk_guard      = risk_guard
        self._pre_trade       = pre_trade_check
        self._signal_model    = signal_model
        self._corr_guard      = correlation_guard
        self._position_sizer  = position_sizer
        self._executor        = order_executor
        self._reconciler      = position_reconciler
        self._audit_log       = audit_log
        self._emergency       = emergency_handler

        self._features_dir    = features_dir
        self._timeframe       = timeframe
        self._lookback        = lookback_candles
        self._confidence      = signal_confidence_threshold
        self._sl_multiplier   = sl_atr_multiplier
        self._tp_multiplier   = tp_atr_multiplier
        self._atr_col         = atr_col
        self._close_col       = close_col

        self._features_loader        = features_loader or self._default_features_loader
        self._balance_getter         = balance_getter
        self._confirmation_callback  = confirmation_callback

        self._stop_event         = threading.Event()
        self._paused             = False
        self._activity_callback: Optional[Callable[[dict], None]] = None

        # Mode validation and initialisation
        _validate_mode_transition(mode)
        self._mode = mode
        logger.info(
            "TradingOrchestrator: initialisiert | Modus={m}",
            m=self._mode.value,
        )

    # ── Oeffentliche Schnittstelle ────────────────────────────────────────────

    def run_cycle(self, symbol: str) -> dict[str, Any]:
        """
        Fuehrt genau einen Entscheidungszyklus fuer ein Symbol aus.

        Returns
        -------
        dict mit:
          symbol          – Handelssymbol
          signal          – 'long' | 'short' | 'flat' | None
          action          – 'open_buy' | 'open_sell' | 'flat' | 'skipped'
          reason          – Textueller Grund fuer die Entscheidung
          ticket          – Order-Ticket oder None
          lot_size        – Lot-Groesse oder None
          step_stopped_at – Name des Schrittes, der den Zyklus abgebrochen hat
        """
        result: dict[str, Any] = {
            "symbol":          symbol,
            "signal":          None,
            "confidence":      None,
            "action":          "skipped",
            "reason":          "",
            "ticket":          None,
            "lot_size":        None,
            "step_stopped_at": None,
            "checks":          [],
            "timestamp":       datetime.now(timezone.utc),
        }

        # ── Pause-Check ───────────────────────────────────────────────────────
        if self._paused:
            result["reason"]          = "trading_paused"
            result["step_stopped_at"] = "pause"
            logger.info("Zyklus | {sym} | Handelspause aktiv -> Abbruch", sym=symbol)
            return result

        # ── Schritt 1: DataPipeline ───────────────────────────────────────────
        now   = datetime.now(timezone.utc)
        mins  = _TIMEFRAME_MINUTES.get(self._timeframe, 60)
        start = now - timedelta(minutes=self._lookback * mins)

        logger.info("Zyklus | {sym} | Schritt 1: DataPipeline", sym=symbol)
        self._pipeline.run_batch(symbol, self._timeframe, start, now, force_refetch=True)

        features = self._features_loader(symbol)
        if features is None or features.empty:
            result["reason"]          = "no_features_available"
            result["step_stopped_at"] = "data_pipeline"
            self._log_step("CYCLE_NO_FEATURES", {"symbol": symbol})
            logger.warning("Zyklus | {sym} | Keine Features verfuegbar -> Abbruch", sym=symbol)
            return result

        features_row = features.iloc[[-1]]

        # ── Schritt 2: RiskGuard ──────────────────────────────────────────────
        logger.info("Zyklus | {sym} | Schritt 2: RiskGuard", sym=symbol)
        if not self._risk_guard.is_trading_allowed():
            result["checks"].append({"name": "RiskGuard", "passed": False, "reason": "Handelssperre"})
            result["reason"]          = "risk_guard_blocked"
            result["step_stopped_at"] = "risk_guard"
            self._log_step("CYCLE_RISK_GUARD_BLOCKED", {"symbol": symbol})
            logger.info("Zyklus | {sym} | RiskGuard blockiert -> Abbruch", sym=symbol)
            return result
        result["checks"].append({"name": "RiskGuard", "passed": True, "reason": ""})

        # ── Schritt 3: PreTradeCheck ──────────────────────────────────────────
        logger.info("Zyklus | {sym} | Schritt 3: PreTradeCheck", sym=symbol)
        safe, check_reason = self._pre_trade.is_safe_to_trade(symbol)
        if not safe:
            result["checks"].append({"name": "PreTradeCheck", "passed": False, "reason": check_reason})
            result["reason"]          = f"pre_trade_check_failed: {check_reason}"
            result["step_stopped_at"] = "pre_trade_check"
            self._log_step("CYCLE_PRE_TRADE_BLOCKED", {"symbol": symbol, "reason": check_reason})
            logger.info("Zyklus | {sym} | PreTradeCheck blockiert: {r} -> Abbruch", sym=symbol, r=check_reason)
            return result
        result["checks"].append({"name": "PreTradeCheck", "passed": True, "reason": ""})

        # ── Schritt 4: Signal ─────────────────────────────────────────────────
        logger.info("Zyklus | {sym} | Schritt 4: Signal", sym=symbol)
        signal = self._signal_model.get_signal(features_row, self._confidence)
        result["signal"] = signal
        try:
            proba = self._signal_model.predict_proba(features_row)
            result["confidence"] = max(proba.values())
        except Exception:  # noqa: BLE001
            pass

        # ── Schritt 5: Flat ───────────────────────────────────────────────────
        if signal == "flat":
            result["action"]          = "flat"
            result["reason"]          = "signal_flat"
            result["step_stopped_at"] = "flat_signal"
            logger.info("Zyklus | {sym} | Signal=flat -> kein Trade", sym=symbol)
            return result

        direction = "buy" if signal == "long" else "sell"

        # ── Schritt 6: CorrelationGuard ───────────────────────────────────────
        logger.info("Zyklus | {sym} | Schritt 6: CorrelationGuard", sym=symbol)
        open_positions = self._executor.get_open_positions()
        if not self._corr_guard.can_open_position(symbol, direction, open_positions):
            result["checks"].append({"name": "CorrelationGuard", "passed": False, "reason": f"Korreliert ({direction})"})
            result["action"]          = "skipped"
            result["reason"]          = "correlation_guard_blocked"
            result["step_stopped_at"] = "correlation_guard"
            self._log_step("CYCLE_CORR_BLOCKED", {"symbol": symbol, "direction": direction})
            logger.info("Zyklus | {sym} | CorrelationGuard blockiert ({dir}) -> Abbruch", sym=symbol, dir=direction)
            return result
        result["checks"].append({"name": "CorrelationGuard", "passed": True, "reason": ""})

        # ── Schritt 7: PositionSizer ──────────────────────────────────────────
        logger.info("Zyklus | {sym} | Schritt 7: PositionSizer", sym=symbol)
        balance     = self._balance_getter() if self._balance_getter else 10_000.0
        atr         = self._extract_value(features_row, self._atr_col,   0.001)
        close_price = self._extract_value(features_row, self._close_col, 1.0)

        size_result = self._position_sizer.calculate_lot_size(balance, atr, symbol)
        if not size_result.is_valid:
            result["reason"]          = f"position_sizer_invalid: {size_result.rejection_reason}"
            result["step_stopped_at"] = "position_sizer"
            self._log_step("CYCLE_SIZE_INVALID", {"symbol": symbol, "reason": size_result.rejection_reason})
            logger.warning("Zyklus | {sym} | PositionSizer ungueltig: {r}", sym=symbol, r=size_result.rejection_reason)
            return result

        multiplier = self._risk_guard.get_position_size_multiplier()
        lot_size   = max(round(size_result.lot_size * multiplier, 8), 0.01)

        # ── SL / TP berechnen (benoetigt fuer Modus-Checks und Order) ────────────
        sl_dist = size_result.stop_loss_distance
        tp_dist = atr * self._tp_multiplier
        if direction == "buy":
            sl_price = round(close_price - sl_dist, 5)
            tp_price = round(close_price + tp_dist, 5)
        else:
            sl_price = round(close_price + sl_dist, 5)
            tp_price = round(close_price - tp_dist, 5)

        # ── Modus-Check ───────────────────────────────────────────────────────

        # SUGGEST_ONLY: Signal anzeigen, niemals Order ausfuehren
        if self._mode == TradingMode.SUGGEST_ONLY:
            result["action"]          = "suggested"
            result["lot_size"]        = lot_size
            result["reason"]          = "suggest_only_mode"
            result["step_stopped_at"] = "mode_suggest_only"
            self._log_step("CYCLE_SUGGESTED", {
                "symbol": symbol, "direction": direction, "lot_size": lot_size,
                "sl": sl_price, "tp": tp_price,
            })
            logger.info(
                "Zyklus | {sym} | SUGGEST_ONLY | {sig} {dir} {lot} Lots "
                "SL={sl} TP={tp} (kein Trade)",
                sym=symbol, sig=signal, dir=direction, lot=lot_size,
                sl=sl_price, tp=tp_price,
            )
            return result

        # CONFIRM_REQUIRED: ConfirmationCallback befragen
        if self._mode == TradingMode.CONFIRM_REQUIRED:
            confirmed = False
            if self._confirmation_callback is not None:
                try:
                    confirmed = self._confirmation_callback.confirm_order(
                        symbol, direction, lot_size, sl_price, tp_price
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "ConfirmationCallback Fehler: {exc} -> kein Trade", exc=exc
                    )
            else:
                logger.warning(
                    "Zyklus | {sym} | CONFIRM_REQUIRED ohne ConfirmationCallback -> kein Trade",
                    sym=symbol,
                )

            if not confirmed:
                result["action"]          = "skipped"
                result["lot_size"]        = lot_size
                result["reason"]          = "order_not_confirmed"
                result["step_stopped_at"] = "confirmation"
                self._log_step("CYCLE_NOT_CONFIRMED", {
                    "symbol": symbol, "direction": direction,
                })
                logger.info(
                    "Zyklus | {sym} | CONFIRM_REQUIRED | Bestaetigung verweigert -> kein Trade",
                    sym=symbol,
                )
                return result

            logger.info(
                "Zyklus | {sym} | CONFIRM_REQUIRED | Bestaetigung erhalten -> Order",
                sym=symbol,
            )

        # AUTONOMOUS: Umgebungsvariable als Schutzschranke pruefen
        if self._mode == TradingMode.AUTONOMOUS and not is_autonomous_allowed():
            result["action"]          = "skipped"
            result["lot_size"]        = lot_size
            result["reason"]          = "autonomous_not_confirmed_env"
            result["step_stopped_at"] = "mode_autonomous_env_check"
            self._log_step("CYCLE_AUTONOMOUS_ENV_MISSING", {"symbol": symbol})
            logger.error(
                "Zyklus | {sym} | AUTONOMOUS ohne CONFIRM_AUTONOMOUS=yes -> kein Trade",
                sym=symbol,
            )
            return result

        # ── Schritt 8: OrderExecutor ──────────────────────────────────────────
        logger.info("Zyklus | {sym} | Schritt 8: OrderExecutor ({dir} {lot})", sym=symbol, dir=direction, lot=lot_size)

        order = self._executor.open_position(symbol, direction, lot_size, sl_price, tp_price)

        result["action"]   = f"open_{direction}"
        result["ticket"]   = order.get("ticket")
        result["lot_size"] = lot_size
        result["reason"]   = "signal_executed"

        if self._audit_log is not None:
            self._audit_log.log_order(order)

        logger.info(
            "Zyklus | {sym} | Order geoeffnet | ticket={t} {dir} {lot} lots",
            sym=symbol, t=order.get("ticket"), dir=direction, lot=lot_size,
        )

        # ── Schritt 9: PositionReconciler (Hintergrund) ───────────────────────
        if self._reconciler is not None:
            logger.debug("Zyklus | {sym} | Schritt 9: PositionReconciler.sync()", sym=symbol)
            try:
                self._reconciler.sync()
            except Exception as exc:  # noqa: BLE001
                logger.warning("PositionReconciler.sync() Fehler: {exc}", exc=exc)

        return result

    def run_loop(self, symbols: list[str], interval_seconds: int = 300) -> None:
        """
        Fuehrt die Handelschleife dauerhaft aus.

        Wartet `interval_seconds` zwischen den Zyklen. Bei Ctrl+C oder
        `stop()` wird die Schleife sauber beendet (laufender Zyklus wird
        noch abgeschlossen).

        Parameters
        ----------
        symbols          : Liste der zu handelnden Symbole.
        interval_seconds : Wartezeit zwischen zwei Durchlaeufen.
        """
        self._stop_event.clear()
        logger.info(
            "TradingOrchestrator: Loop gestartet | Symbole={syms} | Intervall={iv}s",
            syms=symbols, iv=interval_seconds,
        )

        while not self._stop_event.is_set():
            for symbol in symbols:
                if self._stop_event.is_set():
                    break
                try:
                    result = self.run_cycle(symbol)
                    if self._activity_callback is not None:
                        try:
                            self._activity_callback(result)
                        except Exception as _cb_exc:  # noqa: BLE001
                            logger.warning("activity_callback Fehler: {e}", e=_cb_exc)
                    logger.info(
                        "Zyklus abgeschlossen | {sym} | action={a} | reason={r}",
                        sym=symbol, a=result["action"], r=result["reason"],
                    )
                except KeyboardInterrupt:
                    logger.info("TradingOrchestrator: KeyboardInterrupt -> Graceful Shutdown")
                    self.stop()
                    return
                except Exception as exc:  # noqa: BLE001
                    logger.error("TradingOrchestrator: Unbehandelte Exception | {exc}", exc=exc)
                    if self._emergency is not None:
                        self._emergency.handle_unhandled_exception(exc)
                    else:
                        raise

            # Wartet auf Stopp-Signal oder timeout
            self._stop_event.wait(timeout=interval_seconds)

        logger.info("TradingOrchestrator: Loop beendet.")

    def stop(self) -> None:
        """
        Signalisiert der run_loop()-Schleife, sauber zu beenden.
        Laufende Zyklen werden noch abgeschlossen.
        """
        self._stop_event.set()
        logger.info("TradingOrchestrator: Stop-Signal gesetzt.")

    def pause(self, reason: str = "") -> None:
        """
        Aktiviert Handelspause. run_cycle() gibt sofort 'trading_paused' zurueck.
        Wird z.B. von PsychologyTracker bei erkanntem Tilt-Verhalten aufgerufen.
        """
        self._paused = True
        suffix = f" Grund: {reason}" if reason else ""
        logger.warning("TradingOrchestrator: Handelspause aktiviert.{s}", s=suffix)

    def resume(self) -> None:
        """Hebt die Handelspause auf und ermoeglicht wieder Zyklen."""
        self._paused = False
        logger.info("TradingOrchestrator: Handelspause beendet.")

    @property
    def is_paused(self) -> bool:
        """True wenn die Handelspause aktiv ist."""
        return self._paused

    @property
    def mode(self) -> TradingMode:
        """Aktueller Betriebsmodus des Orchestrators."""
        return self._mode

    def set_activity_callback(
        self, callback: Optional[Callable[[dict], None]]
    ) -> None:
        """
        Setzt einen Callback der nach jedem run_cycle()-Aufruf aufgerufen wird.

        Wird vom BotWorker (QThread) gesetzt, um Zyklus-Ergebnisse
        thread-sicher an die GUI weiterzuleiten. None entfernt den Callback.
        """
        self._activity_callback = callback

    def set_mode(self, new_mode: TradingMode) -> None:
        """
        Wechselt den Betriebsmodus.

        Fuer den Wechsel nach AUTONOMOUS muss CONFIRM_AUTONOMOUS=yes gesetzt sein.

        Raises
        ------
        EnvironmentError
            Wenn AUTONOMOUS ohne korrekte Umgebungsvariable angefordert wird.
        """
        _validate_mode_transition(new_mode)
        old_mode   = self._mode
        self._mode = new_mode
        self._log_step("MODE_CHANGED", {
            "old_mode": old_mode.value,
            "new_mode": new_mode.value,
        })
        logger.info(
            "TradingOrchestrator: Modus gewechselt | {old} -> {new}",
            old=old_mode.value, new=new_mode.value,
        )

    def emergency_stop(self) -> None:
        """
        Notfall-Stop: Handelspause aktivieren + alle offenen Positionen schliessen.

        Ruft pause() auf und anschliessend – falls verfuegbar –
        EmergencyHandler.handle_drawdown_limit() um alle Positionen zu schliessen.
        """
        self.pause(reason="emergency_stop")
        self._log_step("EMERGENCY_STOP", {})
        logger.warning("TradingOrchestrator: NOTFALL-STOP ausgefuehrt.")

        if self._emergency is not None:
            try:
                self._emergency.handle_drawdown_limit()
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "EmergencyHandler.handle_drawdown_limit() Fehler: {exc}", exc=exc
                )
        elif self._executor is not None:
            try:
                positions = self._executor.get_open_positions()
                for pos in positions:
                    ticket = pos.get("ticket")
                    if ticket is not None:
                        self._executor.close_position(ticket)
                        logger.info(
                            "NOTFALL-STOP: Position geschlossen | ticket={t}", t=ticket
                        )
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "NOTFALL-STOP: Fehler beim Schliessen der Positionen: {exc}", exc=exc
                )

    # ── Hilfsmethoden ─────────────────────────────────────────────────────────

    def _extract_value(self, df: pd.DataFrame, col: str, fallback: float) -> float:
        """Liest einen Float-Wert aus der letzten Zeile des DataFrames."""
        if col in df.columns:
            return float(df[col].iloc[-1])
        return fallback

    def _log_step(self, event_type: str, details: dict) -> None:
        """Schreibt einen Zyklusschritt ins AuditLog falls vorhanden."""
        if self._audit_log is not None:
            try:
                self._audit_log.log_error(event_type, details)
            except Exception as exc:  # noqa: BLE001
                logger.warning("AuditLog Schreibfehler: {exc}", exc=exc)

    def _default_features_loader(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Standard-Features-Loader: sucht die aktuellste Parquet-Datei
        fuer das Symbol im features_dir.
        """
        pattern = str(Path(self._features_dir) / f"{symbol}_{self._timeframe}_*.parquet")
        files   = sorted(glob.glob(pattern))
        if not files:
            logger.warning("Keine Feature-Datei gefunden: {pat}", pat=pattern)
            return None
        try:
            return pd.read_parquet(files[-1])
        except Exception as exc:  # noqa: BLE001
            logger.error("Feature-Datei nicht lesbar: {exc}", exc=exc)
            return None
