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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from src.data.mt5_connector import _load_mt5


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
    ) -> dict:
        """
        Oeffnet eine Market-Order.

        Parameters
        ----------
        symbol    : Handelssymbol, z.B. "EURUSD".
        direction : "buy" oder "sell".
        lot_size  : Lot-Groesse.
        sl_price  : Stop-Loss-Preis.
        tp_price  : Take-Profit-Preis.

        Returns
        -------
        dict mit ticket, symbol, direction, lot_size, sl_price, tp_price,
        open_price (oder None im Paper-Modus), status.

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
            return self._open_paper(symbol, direction, lot_size, sl_price, tp_price)

        return self._open_live(symbol, direction, lot_size, sl_price, tp_price)

    def close_position(self, ticket: int) -> dict:
        """
        Schliesst eine offene Position.

        Parameters
        ----------
        ticket : MT5-Ticket-Nummer (oder Paper-Trading-Ticket).

        Returns
        -------
        dict mit ticket, symbol, close_price, status='closed'.

        Raises
        ------
        OrderError wenn die Position nicht gefunden oder der Abschluss abgelehnt wird.
        """
        if not self._live:
            return self._close_paper(ticket)
        return self._close_live(ticket)

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
            "open_price": None,
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

    def _close_paper(self, ticket: int) -> dict:
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

    # ── Live-Trading ──────────────────────────────────────────────────────────

    def _open_live(
        self,
        symbol: str,
        direction: str,
        lot_size: float,
        sl_price: float,
        tp_price: float,
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

        ticket = result.order
        trade = {
            "ticket":     ticket,
            "symbol":     symbol,
            "direction":  direction,
            "lot_size":   lot_size,
            "sl_price":   sl_price,
            "tp_price":   tp_price,
            "open_price": getattr(result, "price", None),
            "status":     "open",
        }
        logger.info(
            "[LIVE] open_position | ticket={t} {sym} {dir} {lot} lots",
            t=ticket, sym=symbol, dir=direction, lot=lot_size,
        )
        if self._trade_journal is not None:
            journal_id = self._trade_journal.log_trade_open({
                "symbol":      symbol,
                "direction":   direction,
                "lot_size":    lot_size,
                "entry_price": trade.get("open_price"),
            })
            self._journal_ticket_map[ticket] = journal_id
        if self._on_open_cb is not None:
            try:
                self._on_open_cb(dict(trade))
            except Exception as exc:  # noqa: BLE001
                logger.warning("OrderExecutor: on_open_cb Fehler: {e}", e=exc)
        return trade

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
                "ticket":     p.ticket,
                "symbol":     p.symbol,
                "direction":  "buy" if p.type == mt5.ORDER_TYPE_BUY else "sell",
                "lot_size":   p.volume,
                "sl_price":   p.sl,
                "tp_price":   p.tp,
                "open_price": p.price_open,
                "status":     "open",
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

    def _write_paper_trades(self) -> None:
        """Schreibt alle Paper-Trades atomar in die JSON-Datei."""
        self._paper_path.parent.mkdir(parents=True, exist_ok=True)
        all_trades = list(self._paper_positions.values())
        tmp = self._paper_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(all_trades, f, indent=2, ensure_ascii=False)
        tmp.replace(self._paper_path)
