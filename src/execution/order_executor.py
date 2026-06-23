"""
src/execution/order_executor.py
OrderExecutor – Market-Orders via MT5 oder Paper-Trading.

SICHERHEITSKONZEPT
------------------
  live_trading_enabled=False (Default)
      Orders werden NUR geloggt und in paper_trades.json geschrieben.
      mt5.order_send wird unter KEINEN UMSTAENDEN aufgerufen.

  live_trading_enabled=True
      Zusaetzlich muss die Umgebungsvariable CONFIRM_LIVE=yes gesetzt sein.
      Fehlt sie, wirft __init__ sofort eine RuntimeError – kein stiller
      Fallback auf Paper-Trading, kein Weiterarbeiten.

Fehlerbehandlung:
  Jede abgelehnte MT5-Order wirft OrderError mit Retcode und Kommentar.
  Kein stiller Fehlschlag, kein None-Rueckgabewert bei Fehler.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from loguru import logger

from src.data.mt5_connector import _load_mt5
from src.execution.execution_tracker import ExecutionTracker


class OrderError(Exception):
    """MT5 hat eine Order abgelehnt oder ein kritischer Ausfuehrungsfehler."""


class OrderExecutor:
    """
    Platziert Market-Orders ueber MT5 oder simuliert sie als Paper-Trades.

    Parameters
    ----------
    connector              : MT5Connector-Instanz (wird fuer is_connected / Symbol-Info genutzt).
    live_trading_enabled   : False = Paper-Trading (Default), True = echte MT5-Orders.
    paper_trades_path      : Pfad zu paper_trades.json.
    trailing_stop_min_pips : Mindestabstand des Trailing-Stops vom Kurs in Pips (Standard: 10).
    trailing_stop_step_pips: Mindest-Nachzug-Schritt in Pips (Standard: 5).
    pip_size               : Pip-Groesse des Instruments (Standard: 0.0001 fuer Majors).
    """

    def __init__(
        self,
        connector,
        live_trading_enabled: bool = False,
        paper_trades_path: str | Path = "data/processed/paper_trades.json",
        trailing_stop_min_pips: float = 10.0,
        trailing_stop_step_pips: float = 5.0,
        pip_size: float = 0.0001,
        audit_log=None,
        trade_journal=None,
        execution_tracker: Optional[ExecutionTracker] = None,
    ) -> None:
        if live_trading_enabled:
            confirm = os.environ.get("CONFIRM_LIVE", "")
            if confirm != "yes":
                raise RuntimeError(
                    "live_trading_enabled=True erfordert die Umgebungsvariable "
                    "CONFIRM_LIVE=yes. Setze sie explizit bevor du Live-Trading startest. "
                    "Kein stiller Fallback auf Paper-Trading."
                )

        self._connector = connector
        self._live = live_trading_enabled
        self._paper_path = Path(paper_trades_path)
        self._ts_min_pips = trailing_stop_min_pips
        self._ts_step_pips = trailing_stop_step_pips
        self._pip_size = pip_size
        self._audit_log = audit_log
        self._trade_journal = trade_journal
        self._execution_tracker = execution_tracker

        # Paper-Trading: In-Memory-Positionen {ticket -> position_dict}
        self._paper_positions: dict[int, dict] = {}
        self._next_ticket: int = 1
        # Live-Trading: Ticket -> Journal-Trade-ID fuer spaeteres Close
        self._journal_ticket_map: dict[int, int] = {}

        # GUI-Callbacks fuer sofortige Order-Updates (kein Qt-Import benoetigt)
        self._on_open_cb:  Optional[Callable[[dict], None]] = None
        self._on_close_cb: Optional[Callable[[dict], None]] = None

        logger.info(
            "OrderExecutor | live={live} | trailing_min={tmin}p step={tstep}p",
            live=self._live,
            tmin=self._ts_min_pips,
            tstep=self._ts_step_pips,
        )

    # ── Oeffentliche Methoden ─────────────────────────────────────────────────

    def set_order_callbacks(
        self,
        on_open:  Optional[Callable[[dict], None]] = None,
        on_close: Optional[Callable[[dict], None]] = None,
    ) -> None:
        """
        Setzt Callbacks die nach open_position() bzw. close_position() aufgerufen werden.

        Ermoeglicht GUI-seitige Sofort-Updates ohne Qt-Abhaengigkeit im Executor.
        Beide Parameter koennen None sein um bestehende Callbacks zu entfernen.
        """
        self._on_open_cb  = on_open
        self._on_close_cb = on_close

    def open_position(
        self,
        symbol: str,
        direction: str,
        lot_size: float,
        sl_price: float,
        tp_price: float,
        open_price: Optional[float] = None,
    ) -> dict:
        """
        Oeffnet eine Market-Order.

        Parameters
        ----------
        symbol     : Handelssymbol, z.B. "EURUSD".
        direction  : "buy" oder "sell".
        lot_size   : Lot-Groesse.
        sl_price   : Stop-Loss-Preis.
        tp_price   : Take-Profit-Preis.
        open_price : Optionaler simulierter Eroeffnungspreis (nur Paper-Modus).
                     Wenn angegeben, wird er als Einstiegspreis im Paper-Trade gespeichert.
                     Live-Modus ignoriert diesen Parameter (MT5 liefert den echten Preis).

        Returns
        -------
        dict mit ticket, symbol, direction, lot_size, sl_price, tp_price,
        open_price, status.

        Raises
        ------
        OrderError wenn MT5 die Order ablehnt.
        ValueError fuer ungueltige Parameter.
        """
        direction = direction.lower()
        if direction not in ("buy", "sell"):
            raise ValueError(f"direction muss 'buy' oder 'sell' sein, nicht '{direction}'.")
        if lot_size <= 0:
            raise ValueError("lot_size muss positiv sein.")

        if not self._live:
            return self._open_paper(symbol, direction, lot_size, sl_price, tp_price, open_price)

        return self._open_live(symbol, direction, lot_size, sl_price, tp_price)

    def close_position(
        self,
        ticket: int,
        close_price: Optional[float] = None,
        pnl: Optional[float] = None,
    ) -> dict:
        """
        Schliesst eine offene Position.

        Parameters
        ----------
        ticket      : MT5-Ticket-Nummer (oder Paper-Trading-Ticket).
        close_price : Optionaler Schlusskurs (wird im Paper-Trade gespeichert).
        pnl         : Optionaler realisierter P&L (wird im Paper-Trade gespeichert).

        Returns
        -------
        dict mit ticket, symbol, close_price, status='closed'.

        Raises
        ------
        OrderError wenn die Position nicht gefunden oder der Abschluss abgelehnt wird.
        """
        if not self._live:
            return self._close_paper(ticket, close_price=close_price, pnl=pnl)
        return self._close_live(ticket)

    def set_stop_loss(self, ticket: int, new_sl_price: float) -> None:
        """
        Setzt den Stop-Loss einer offenen Paper-Position auf einen neuen Preis.

        Nur im Paper-Modus. Live-Positionen werden nicht beruehrt.
        """
        if not self._live:
            pos = self._paper_positions.get(ticket)
            if pos and pos.get("status") == "open":
                pos["sl_price"] = new_sl_price
                self._write_paper_trades()
                logger.info(
                    "[PAPER] set_stop_loss | ticket={t} | SL -> {sl:.5f}",
                    t=ticket, sl=new_sl_price,
                )

    def mark_profit_lock_70(self, ticket: int) -> None:
        """Markiert eine Paper-Position als 'Profit-Lock 70% erreicht'."""
        if not self._live:
            pos = self._paper_positions.get(ticket)
            if pos and pos.get("status") == "open":
                pos["profit_lock_70_triggered"] = True
                self._write_paper_trades()
                logger.info(
                    "[PAPER] mark_profit_lock_70 | ticket={t}",
                    t=ticket,
                )

    def mark_break_even(self, ticket: int) -> None:
        """
        Markiert eine Paper-Position als 'Break-Even erreicht'.

        Setzt break_even_triggered=True im Position-Dict und schreibt paper_trades.json.
        Nur im Paper-Modus.
        """
        if not self._live:
            pos = self._paper_positions.get(ticket)
            if pos and pos.get("status") == "open":
                pos["break_even_triggered"] = True
                self._write_paper_trades()
                logger.info(
                    "[PAPER] mark_break_even | ticket={t} | BE aktiv",
                    t=ticket,
                )

    def update_trailing_stop(self, ticket: int, current_price: float) -> None:
        """
        Aktualisiert den Trailing-Stop einer Position.

        Regeln:
          - LONG: SL bewegt sich nur aufwaerts (nie abwaerts).
          - SHORT: SL bewegt sich nur abwaerts (nie aufwaerts).
          - Update nur wenn neuer SL mindestens trailing_stop_step_pips
            besser als aktueller SL.
          - SL wird stets mindestens trailing_stop_min_pips vom Kurs
            entfernt gesetzt.

        Parameters
        ----------
        ticket        : Positions-Ticket.
        current_price : Aktueller Marktpreis (Bid fuer Buy, Ask fuer Sell).

        Raises
        ------
        OrderError wenn Position nicht gefunden oder MT5 Update ablehnt.
        """
        if not self._live:
            self._update_trailing_paper(ticket, current_price)
        else:
            self._update_trailing_live(ticket, current_price)

    def get_open_positions(self) -> list[dict]:
        """
        Gibt alle offenen Positionen zurueck.

        Returns
        -------
        list[dict] mit Feldern: ticket, symbol, direction, lot_size,
        sl_price, tp_price, open_price, status.
        """
        if not self._live:
            return [
                p for p in self._paper_positions.values()
                if p.get("status") == "open"
            ]
        return self._get_live_positions()

    def place_limit_order(
        self,
        symbol:      str,
        direction:   str,
        lot_size:    float,
        sl_price:    float,
        tp_price:    float,
        limit_price: float,
        timeout_s:   Optional[float] = None,
    ) -> dict:
        """
        Platziert eine Limit-Order (kein sofortiger Marktauftrag).

        Parameters
        ----------
        symbol      : Handelssymbol.
        direction   : "buy" oder "sell".
        lot_size    : Lot-Groesse.
        sl_price    : Stop-Loss-Preis.
        tp_price    : Take-Profit-Preis.
        limit_price : Gewuenschter Ausfuehrungspreis (Limit-Preis).
        timeout_s   : Optionale Verfallszeit in Sekunden ab jetzt.
                      Laeuft die Frist ab ohne Ausfuehrung, wird die Order
                      von check_and_expire_limit_orders() storniert.

        Returns
        -------
        dict mit ticket, status='pending_limit' und allen Order-Parametern.
        """
        direction = direction.lower()
        if direction not in ("buy", "sell"):
            raise ValueError(f"direction muss 'buy' oder 'sell' sein, nicht '{direction}'.")
        if lot_size <= 0:
            raise ValueError("lot_size muss positiv sein.")

        if not self._live:
            return self._place_paper_limit(
                symbol, direction, lot_size, sl_price, tp_price, limit_price, timeout_s
            )
        return self._place_live_limit(
            symbol, direction, lot_size, sl_price, tp_price, limit_price, timeout_s
        )

    def cancel_limit_order(self, ticket: int) -> dict:
        """
        Storniert eine ausstehende Limit-Order.

        Returns
        -------
        dict mit ticket und status='cancelled'.

        Raises
        ------
        OrderError wenn die Order nicht gefunden oder bereits ausgefuehrt/storniert.
        """
        if not self._live:
            return self._cancel_paper_limit(ticket)
        return self._cancel_live_limit(ticket)

    def check_and_expire_limit_orders(
        self,
        current_time: Optional[datetime] = None,
    ) -> list[int]:
        """
        Storniert Limit-Orders deren Timeout abgelaufen ist.

        Parameters
        ----------
        current_time : Optionaler Zeitstempel (Standard: jetzt UTC).
                       Injectable fuer Tests.

        Returns
        -------
        list[int] mit den stornierten Tickets.
        """
        now = current_time or datetime.now(timezone.utc)
        cancelled: list[int] = []

        if not self._live:
            for ticket, pos in list(self._paper_positions.items()):
                if pos.get("status") != "pending_limit":
                    continue
                deadline = pos.get("timeout_deadline")
                if deadline is None:
                    continue
                if isinstance(deadline, str):
                    deadline = datetime.fromisoformat(deadline)
                if now >= deadline:
                    self._cancel_paper_limit(ticket)
                    cancelled.append(ticket)
                    logger.info(
                        "[PAPER] Limit-Order {t} storniert (Timeout)", t=ticket
                    )
        else:
            # Live: Check pending orders via MT5
            cancelled = self._expire_live_limit_orders(now)

        return cancelled

    # ── Reconciliation-Schnittstelle ──────────────────────────────────────────

    def reconcile_add_position(self, ticket: int, pos: dict) -> None:
        """
        Fuegt eine extern bekannte Position in die lokale Verfolgung ein.
        Wird vom PositionReconciler aufgerufen wenn MT5 eine Position hat,
        die lokal nicht bekannt ist.
        """
        entry = dict(pos)
        entry["ticket"] = ticket
        entry.setdefault("status", "open")
        self._paper_positions[ticket] = entry
        logger.info("Reconcile: Position {t} lokal registriert ({sym})", t=ticket, sym=entry.get("symbol"))

    def reconcile_close_position(self, ticket: int) -> None:
        """
        Markiert eine lokal bekannte Position als extern geschlossen.
        Wird vom PositionReconciler aufgerufen wenn MT5 die Position
        nicht mehr kennt (SL/TP getroffen, manuell geschlossen, etc.).
        """
        pos = self._paper_positions.get(ticket)
        if pos is not None:
            pos["status"] = "closed"
            logger.info("Reconcile: Position {t} extern geschlossen markiert", t=ticket)
        else:
            logger.warning("Reconcile: Position {t} fuer Close-Markierung nicht gefunden", t=ticket)

    # ── Paper-Trading ─────────────────────────────────────────────────────────

    def _open_paper(
        self,
        symbol: str,
        direction: str,
        lot_size: float,
        sl_price: float,
        tp_price: float,
        fill_price: Optional[float] = None,
    ) -> dict:
        ticket = self._next_ticket
        self._next_ticket += 1
        now = datetime.now(timezone.utc).isoformat()

        position: dict[str, Any] = {
            "ticket":     ticket,
            "symbol":     symbol,
            "direction":  direction,
            "lot_size":   lot_size,
            "sl_price":   sl_price,
            "tp_price":   tp_price,
            "open_price": fill_price,
            "open_time":  now,
            "close_price": None,
            "close_time":  None,
            "status":     "open",
        }
        self._paper_positions[ticket] = position
        self._write_paper_trades()

        logger.info(
            "[PAPER] open_position | ticket={t} {sym} {dir} {lot} lots | "
            "SL={sl} TP={tp}",
            t=ticket, sym=symbol, dir=direction, lot=lot_size,
            sl=sl_price, tp=tp_price,
        )
        if self._audit_log is not None:
            self._audit_log.log_order(dict(position))
        if self._trade_journal is not None:
            journal_id = self._trade_journal.log_trade_open({
                "symbol":    symbol,
                "direction": direction,
                "lot_size":  lot_size,
                "entry_time": now,
            })
            self._paper_positions[ticket]["journal_id"] = journal_id
        order_result = dict(position)
        if self._on_open_cb is not None:
            try:
                self._on_open_cb(order_result)
            except Exception as exc:  # noqa: BLE001
                logger.warning("OrderExecutor: on_open_cb Fehler: {e}", e=exc)
        return order_result

    def _close_paper(
        self,
        ticket: int,
        close_price: Optional[float] = None,
        pnl: Optional[float] = None,
    ) -> dict:
        pos = self._paper_positions.get(ticket)
        if pos is None:
            raise OrderError(
                f"Paper-Position {ticket} nicht gefunden. "
                f"Verfuegbare Tickets: {list(self._paper_positions.keys())}"
            )
        if pos["status"] != "open":
            raise OrderError(f"Paper-Position {ticket} ist bereits geschlossen.")

        now = datetime.now(timezone.utc).isoformat()
        pos["status"] = "closed"
        pos["close_time"] = now
        pos["close_price"] = close_price
        pos["pnl"] = pnl
        self._write_paper_trades()

        logger.info("[PAPER] close_position | ticket={t}", t=ticket)
        if self._audit_log is not None:
            self._audit_log.log_order(dict(pos))
        if self._trade_journal is not None:
            journal_id = pos.get("journal_id")
            if journal_id is not None:
                self._trade_journal.log_trade_close(journal_id, {"exit_time": now})
        close_result = dict(pos)
        if self._on_close_cb is not None:
            try:
                self._on_close_cb(close_result)
            except Exception as exc:  # noqa: BLE001
                logger.warning("OrderExecutor: on_close_cb Fehler: {e}", e=exc)
        return close_result

    def _update_trailing_paper(self, ticket: int, current_price: float) -> None:
        pos = self._paper_positions.get(ticket)
        if pos is None or pos["status"] != "open":
            raise OrderError(f"Paper-Position {ticket} nicht gefunden oder bereits geschlossen.")

        new_sl = self._compute_new_sl(pos["direction"], pos["sl_price"], current_price)
        if new_sl is not None:
            old_sl = pos["sl_price"]
            pos["sl_price"] = new_sl
            self._write_paper_trades()
            logger.info(
                "[PAPER] trailing_stop | ticket={t} | SL {old:.5f} -> {new:.5f}",
                t=ticket, old=old_sl, new=new_sl,
            )
        else:
            logger.debug(
                "[PAPER] trailing_stop | ticket={t} | kein Update noetig", t=ticket
            )

    def _place_paper_limit(
        self,
        symbol:      str,
        direction:   str,
        lot_size:    float,
        sl_price:    float,
        tp_price:    float,
        limit_price: float,
        timeout_s:   Optional[float],
    ) -> dict:
        ticket = self._next_ticket
        self._next_ticket += 1
        now = datetime.now(timezone.utc)

        deadline_iso: Optional[str] = None
        if timeout_s is not None:
            deadline_iso = (now + timedelta(seconds=timeout_s)).isoformat()

        position: dict[str, Any] = {
            "ticket":           ticket,
            "symbol":           symbol,
            "direction":        direction,
            "lot_size":         lot_size,
            "sl_price":         sl_price,
            "tp_price":         tp_price,
            "limit_price":      limit_price,
            "open_price":       None,
            "open_time":        now.isoformat(),
            "close_price":      None,
            "close_time":       None,
            "timeout_deadline": deadline_iso,
            "status":           "pending_limit",
        }
        self._paper_positions[ticket] = position
        self._write_paper_trades()

        logger.info(
            "[PAPER] place_limit_order | ticket={t} {sym} {dir} {lot} lots "
            "@ limit={lp} SL={sl} TP={tp}",
            t=ticket, sym=symbol, dir=direction, lot=lot_size,
            lp=limit_price, sl=sl_price, tp=tp_price,
        )
        return dict(position)

    def _cancel_paper_limit(self, ticket: int) -> dict:
        pos = self._paper_positions.get(ticket)
        if pos is None:
            raise OrderError(
                f"Paper-Limit-Order {ticket} nicht gefunden. "
                f"Verfuegbare Tickets: {list(self._paper_positions.keys())}"
            )
        if pos["status"] != "pending_limit":
            raise OrderError(
                f"Paper-Order {ticket} hat Status '{pos['status']}', "
                f"nicht 'pending_limit'. Stornierung nicht moeglich."
            )
        pos["status"] = "cancelled"
        pos["close_time"] = datetime.now(timezone.utc).isoformat()
        self._write_paper_trades()
        logger.info("[PAPER] cancel_limit_order | ticket={t}", t=ticket)
        return dict(pos)

    # ── Live-Trading ──────────────────────────────────────────────────────────

    def _open_live(
        self,
        symbol:    str,
        direction: str,
        lot_size:  float,
        sl_price:  float,
        tp_price:  float,
        expected_price: Optional[float] = None,
    ) -> dict:
        if not self._connector.is_connected:
            raise OrderError("MT5Connector ist nicht verbunden.")

        mt5 = _load_mt5()
        order_type = mt5.ORDER_TYPE_BUY if direction == "buy" else mt5.ORDER_TYPE_SELL

        request = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       symbol,
            "volume":       lot_size,
            "type":         order_type,
            "sl":           sl_price,
            "tp":           tp_price,
            "type_filling": mt5.ORDER_FILLING_IOC,
            "comment":      "QuantzAI",
        }

        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            retcode = result.retcode if result else "None"
            comment = result.comment if result else "kein Ergebnis von MT5"
            raise OrderError(
                f"MT5 open_position abgelehnt | symbol={symbol} dir={direction} "
                f"retcode={retcode} | {comment}"
            )

        ticket      = result.order
        actual_price = getattr(result, "price", None)
        filled_volume = getattr(result, "volume", lot_size)

        # Partial-Fill-Erkennung (IOC: verbleibende Menge wird storniert)
        if filled_volume is not None and filled_volume < lot_size:
            logger.warning(
                "[LIVE] Partial-Fill | ticket={t} {sym} {dir} "
                "angefordert={req} ausgefuehrt={fill}",
                t=ticket, sym=symbol, dir=direction,
                req=lot_size, fill=filled_volume,
            )

        trade = {
            "ticket":         ticket,
            "symbol":         symbol,
            "direction":      direction,
            "lot_size":       filled_volume if filled_volume is not None else lot_size,
            "requested_lots": lot_size,
            "sl_price":       sl_price,
            "tp_price":       tp_price,
            "open_price":     actual_price,
            "partial_fill":   filled_volume is not None and filled_volume < lot_size,
            "status":         "open",
        }
        logger.info(
            "[LIVE] open_position | ticket={t} {sym} {dir} {lot} lots",
            t=ticket, sym=symbol, dir=direction, lot=filled_volume,
        )

        # Slippage erfassen
        if self._execution_tracker is not None and actual_price is not None:
            ref_price = expected_price if expected_price is not None else actual_price
            self._execution_tracker.record_slippage(
                ticket=ticket,
                symbol=symbol,
                direction=direction,
                expected_price=ref_price,
                actual_price=actual_price,
            )
            # Gebuehren aus Connector-Symbol-Info ableiten (Spread + Commission)
            try:
                info = self._connector.get_symbol_info(symbol)
                if info:
                    spread_pts  = info.get("spread", 0) * self._pip_size
                    commission  = abs(info.get("commission", 0.0))
                    swap        = info.get("swap_long" if direction == "buy" else "swap_short", 0.0)
                    self._execution_tracker.record_fees(
                        ticket=ticket,
                        symbol=symbol,
                        spread=spread_pts,
                        commission=commission,
                        swap=float(swap) if swap is not None else 0.0,
                    )
            except Exception as exc:  # noqa: BLE001
                logger.debug("ExecutionTracker: Gebuehren-Abruf fehlgeschlagen: {e}", e=exc)

        if self._trade_journal is not None:
            journal_id = self._trade_journal.log_trade_open({
                "symbol":      symbol,
                "direction":   direction,
                "lot_size":    trade["lot_size"],
                "entry_price": actual_price,
            })
            self._journal_ticket_map[ticket] = journal_id
        if self._on_open_cb is not None:
            try:
                self._on_open_cb(dict(trade))
            except Exception as exc:  # noqa: BLE001
                logger.warning("OrderExecutor: on_open_cb Fehler: {e}", e=exc)
        return trade

    def _place_live_limit(
        self,
        symbol:      str,
        direction:   str,
        lot_size:    float,
        sl_price:    float,
        tp_price:    float,
        limit_price: float,
        timeout_s:   Optional[float],
    ) -> dict:
        if not self._connector.is_connected:
            raise OrderError("MT5Connector ist nicht verbunden.")

        mt5 = _load_mt5()
        order_type = (
            mt5.ORDER_TYPE_BUY_LIMIT if direction == "buy"
            else mt5.ORDER_TYPE_SELL_LIMIT
        )

        expiration = 0
        if timeout_s is not None:
            exp_dt = datetime.now(timezone.utc) + timedelta(seconds=timeout_s)
            expiration = int(exp_dt.timestamp())

        request: dict[str, Any] = {
            "action":   mt5.TRADE_ACTION_PENDING,
            "symbol":   symbol,
            "volume":   lot_size,
            "type":     order_type,
            "price":    limit_price,
            "sl":       sl_price,
            "tp":       tp_price,
            "comment":  "QuantzAI limit",
        }
        if expiration:
            request["expiration"] = expiration

        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            retcode = result.retcode if result else "None"
            comment = result.comment if result else "kein Ergebnis von MT5"
            raise OrderError(
                f"MT5 place_limit_order abgelehnt | symbol={symbol} dir={direction} "
                f"retcode={retcode} | {comment}"
            )

        ticket = result.order
        logger.info(
            "[LIVE] place_limit_order | ticket={t} {sym} {dir} {lot} lots @ {lp}",
            t=ticket, sym=symbol, dir=direction, lot=lot_size, lp=limit_price,
        )
        return {
            "ticket":      ticket,
            "symbol":      symbol,
            "direction":   direction,
            "lot_size":    lot_size,
            "sl_price":    sl_price,
            "tp_price":    tp_price,
            "limit_price": limit_price,
            "status":      "pending_limit",
        }

    def _cancel_live_limit(self, ticket: int) -> dict:
        if not self._connector.is_connected:
            raise OrderError("MT5Connector ist nicht verbunden.")

        mt5 = _load_mt5()
        orders = mt5.orders_get(ticket=ticket)
        if not orders:
            raise OrderError(f"MT5-Pending-Order {ticket} nicht gefunden.")

        request = {
            "action": mt5.TRADE_ACTION_REMOVE,
            "order":  ticket,
        }
        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            retcode = result.retcode if result else "None"
            comment = result.comment if result else "kein Ergebnis"
            raise OrderError(
                f"MT5 cancel_limit_order abgelehnt | ticket={ticket} "
                f"retcode={retcode} | {comment}"
            )

        logger.info("[LIVE] cancel_limit_order | ticket={t}", t=ticket)
        return {"ticket": ticket, "status": "cancelled"}

    def _expire_live_limit_orders(self, now: datetime) -> list[int]:
        """Storniert live Pending-Orders die keinen MT5-Verfall haben und timed out sind."""
        # In live trading we rely on MT5's own expiration mechanism.
        # This method is a hook for manual-timeout tracking if needed.
        return []

    def _close_live(self, ticket: int) -> dict:
        if not self._connector.is_connected:
            raise OrderError("MT5Connector ist nicht verbunden.")

        mt5 = _load_mt5()
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            raise OrderError(f"MT5-Position {ticket} nicht gefunden.")

        pos = positions[0]
        close_type = (
            mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY
            else mt5.ORDER_TYPE_BUY
        )
        request = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "position":     ticket,
            "symbol":       pos.symbol,
            "volume":       pos.volume,
            "type":         close_type,
            "type_filling": mt5.ORDER_FILLING_IOC,
            "comment":      "QuantzAI close",
        }

        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            retcode = result.retcode if result else "None"
            comment = result.comment if result else "kein Ergebnis"
            raise OrderError(
                f"MT5 close_position abgelehnt | ticket={ticket} "
                f"retcode={retcode} | {comment}"
            )

        trade = {
            "ticket":      ticket,
            "symbol":      pos.symbol,
            "close_price": getattr(result, "price", None),
            "status":      "closed",
        }
        logger.info("[LIVE] close_position | ticket={t}", t=ticket)
        if self._trade_journal is not None:
            journal_id = self._journal_ticket_map.pop(ticket, None)
            if journal_id is not None:
                self._trade_journal.log_trade_close(
                    journal_id, {"exit_price": trade.get("close_price")}
                )
        if self._on_close_cb is not None:
            try:
                self._on_close_cb(dict(trade))
            except Exception as exc:  # noqa: BLE001
                logger.warning("OrderExecutor: on_close_cb Fehler: {e}", e=exc)
        return trade

    def _update_trailing_live(self, ticket: int, current_price: float) -> None:
        if not self._connector.is_connected:
            raise OrderError("MT5Connector ist nicht verbunden.")

        mt5 = _load_mt5()
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            raise OrderError(f"MT5-Position {ticket} nicht gefunden.")

        pos = positions[0]
        direction = "buy" if pos.type == mt5.ORDER_TYPE_BUY else "sell"
        new_sl = self._compute_new_sl(direction, pos.sl, current_price)

        if new_sl is None:
            logger.debug("[LIVE] trailing_stop | ticket={t} | kein Update noetig", t=ticket)
            return

        request = {
            "action":   mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "sl":       new_sl,
            "tp":       pos.tp,
        }
        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            retcode = result.retcode if result else "None"
            raise OrderError(
                f"Trailing-Stop Update abgelehnt | ticket={ticket} retcode={retcode}"
            )
        logger.info(
            "[LIVE] trailing_stop | ticket={t} | SL -> {sl:.5f}", t=ticket, sl=new_sl
        )

    def _get_live_positions(self) -> list[dict]:
        mt5 = _load_mt5()
        positions = mt5.positions_get()
        if not positions:
            return []
        result = []
        for p in positions:
            result.append({
                "ticket":      p.ticket,
                "symbol":      p.symbol,
                "direction":   "buy" if p.type == mt5.ORDER_TYPE_BUY else "sell",
                "lot_size":    p.volume,
                "sl_price":    p.sl,
                "tp_price":    p.tp,
                "open_price":  p.price_open,
                "current_pnl": p.profit,
                "status":      "open",
            })
        return result

    # ── Hilfsmethoden ─────────────────────────────────────────────────────────

    def _compute_new_sl(
        self,
        direction: str,
        current_sl: float,
        current_price: float,
    ) -> float | None:
        """
        Berechnet neuen Trailing-Stop-Preis.

        Gibt None zurueck wenn kein Update noetig (SL wuerde sich nicht
        in guenstige Richtung bewegen oder Schritt zu klein).
        """
        min_dist = self._ts_min_pips * self._pip_size
        step     = self._ts_step_pips * self._pip_size

        if direction == "buy":
            # SL trail unterhalb des Kurses, bewegt sich nur aufwaerts
            candidate = current_price - min_dist
            if candidate >= current_sl + step:
                return candidate
        else:
            # SL trail oberhalb des Kurses, bewegt sich nur abwaerts
            candidate = current_price + min_dist
            if candidate <= current_sl - step:
                return candidate
        return None

    def check_paper_sl_tp(self) -> list[dict]:
        """
        Prueft alle offenen Paper-Positionen auf SL/TP-Treffer und schliesst sie.

        Im Paper-Modus ueberwacht MT5 keine SL/TP (Positionen sind nicht real).
        Diese Methode muss im Takt des Bots aufgerufen werden.

        Returns
        -------
        Liste der geschlossenen Positions-Dicts (leer wenn nichts getroffen).
        """
        if self._live:
            return []

        closed: list[dict] = []
        # Cache fuer Contract-Sizes um nicht pro Position die Broker-API abzufragen
        _contract_cache: dict[str, float] = {}

        for ticket, pos in list(self._paper_positions.items()):
            if pos.get("status") != "open":
                continue

            symbol     = pos["symbol"]
            direction  = pos["direction"]
            sl_price   = pos.get("sl_price")
            tp_price   = pos.get("tp_price")
            open_price = pos.get("open_price") or 0.0
            lot_size   = pos.get("lot_size", 0.0)

            try:
                tick = self._connector.get_tick(symbol)
                bid  = float(tick["bid"])
                ask  = float(tick["ask"])
            except Exception:
                continue

            hit_sl      = False
            hit_tp      = False
            close_price: float | None = None

            if direction == "buy":
                close_price = bid
                if sl_price and bid <= sl_price:
                    hit_sl = True
                elif tp_price and bid >= tp_price:
                    hit_tp = True
            else:  # sell
                close_price = ask
                if sl_price and ask >= sl_price:
                    hit_sl = True
                elif tp_price and ask <= tp_price:
                    hit_tp = True

            if not (hit_sl or hit_tp):
                continue

            # P&L berechnen
            pnl: float | None = None
            if open_price and close_price:
                try:
                    if symbol not in _contract_cache:
                        info = self._connector.get_symbol_info(symbol) or {}
                        _contract_cache[symbol] = float(info.get("contract_size", 100_000.0))
                    contract_size = _contract_cache[symbol]
                    if direction == "buy":
                        pnl = (close_price - open_price) * lot_size * contract_size
                    else:
                        pnl = (open_price - close_price) * lot_size * contract_size
                except Exception:
                    pnl = None

            reason = "TP" if hit_tp else "SL"
            logger.info(
                "[PAPER] {r} getroffen | {sym} {dir} | ticket={t} | "
                "close={cp:.5f} | pnl={pnl}",
                r=reason, sym=symbol, dir=direction, t=ticket,
                cp=close_price,
                pnl=f"{pnl:.2f}" if pnl is not None else "?",
            )

            try:
                result = self._close_paper(ticket, close_price=close_price, pnl=pnl)
                closed.append(result)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[PAPER] SL/TP-Close fehlgeschlagen | ticket={t}: {e}",
                    t=ticket, e=exc,
                )

        return closed

    def _write_paper_trades(self) -> None:
        """Schreibt alle Paper-Trades atomar in die JSON-Datei."""
        self._paper_path.parent.mkdir(parents=True, exist_ok=True)
        all_trades = list(self._paper_positions.values())
        tmp = self._paper_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(all_trades, f, indent=2, ensure_ascii=False)
        tmp.replace(self._paper_path)
