"""
src/risk/break_even_manager.py
Break-Even-Stop und Trailing-Stop-Verwaltung fuer offene Paper-Positionen.

Mechanismus pro Zyklus:
  1. Aktuellen Bid/Ask von MT5 holen
  2. SL/TP-Treffer pruefen -> Position automatisch schliessen (Paper-Modus)
  3. Break-Even: SL auf Eroeffnungspreis + Puffer ziehen wenn >= threshold des TP-Weges
  4. Trailing-Stop aktivieren sobald Break-Even aktiv ist

Alle Kern-Entscheidungen sind als pure Funktionen implementiert (testbar ohne Mocks).
"""

from __future__ import annotations

from typing import Any, Optional

from loguru import logger


# ─────────────────────────────────────────────────────────────────────────────
#  Pure Hilfsfunktionen (kein I/O, vollstaendig testbar)
# ─────────────────────────────────────────────────────────────────────────────

def should_trigger_break_even(
    direction: str,
    open_price: float,
    tp_price: float,
    current_price: float,
    threshold: float = 0.5,
) -> bool:
    """
    Gibt True wenn die Position >= threshold des Weges zum TP erreicht hat.

    Parameters
    ----------
    direction     : "buy" | "sell"
    open_price    : Eroeffnungskurs
    tp_price      : Take-Profit-Kurs
    current_price : Aktueller Kurs (Bid fuer BUY, Ask fuer SELL)
    threshold     : Anteil der TP-Distanz (0.5 = 50 %)
    """
    if direction == "buy":
        total_dist = tp_price - open_price
        if total_dist <= 0:
            return False
        progress = (current_price - open_price) / total_dist
    else:
        total_dist = open_price - tp_price
        if total_dist <= 0:
            return False
        progress = (open_price - current_price) / total_dist
    return progress >= threshold


def calc_break_even_sl(
    direction: str,
    open_price: float,
    spread_buffer_pips: float = 2.0,
    pip_size: float = 0.0001,
) -> float:
    """
    Berechnet den neuen SL-Preis fuer den Break-Even-Stop.

    BUY : open_price + buffer  (SL knapp ueber Eroeffnung – Spread-Schutz)
    SELL: open_price - buffer  (SL knapp unter Eroeffnung – Spread-Schutz)
    """
    buffer = spread_buffer_pips * pip_size
    if direction == "buy":
        return round(open_price + buffer, 5)
    return round(open_price - buffer, 5)


def is_sl_hit(direction: str, sl_price: float, bid: float, ask: float) -> bool:
    """
    True wenn der aktuelle Kurs den Stop-Loss ausloest.

    BUY  : SL getriggert wenn BID <= sl_price
    SELL : SL getriggert wenn ASK >= sl_price
    """
    if direction == "buy":
        return bid <= sl_price
    return ask >= sl_price


def is_tp_hit(direction: str, tp_price: float, bid: float, ask: float) -> bool:
    """
    True wenn der aktuelle Kurs den Take-Profit ausloest.

    BUY  : TP getriggert wenn BID >= tp_price
    SELL : TP getriggert wenn ASK <= tp_price
    """
    if direction == "buy":
        return bid >= tp_price
    return ask <= tp_price


def calc_realized_pnl(
    direction: str,
    open_price: float,
    close_price: float,
    lot_size: float,
    contract_size: float,
) -> float:
    """
    Berechnet den realisierten Gewinn/Verlust eines geschlossenen Trades.

    BUY : (close_price - open_price) * lot_size * contract_size
    SELL: (open_price - close_price) * lot_size * contract_size
    """
    if direction == "buy":
        return (close_price - open_price) * lot_size * contract_size
    return (open_price - close_price) * lot_size * contract_size


# ─────────────────────────────────────────────────────────────────────────────
#  BreakEvenManager
# ─────────────────────────────────────────────────────────────────────────────

class BreakEvenManager:
    """
    Verwaltet Break-Even-Stop und Trailing-Stop fuer offene Paper-Positionen.

    Wird einmal pro Zyklus pro Symbol aufgerufen (aus MultiSymbolOrchestrator).

    Parameters
    ----------
    connector             : MT5Connector – fuer get_tick() und get_symbol_info()
    order_executor        : OrderExecutor – fuer get_open_positions(), set_stop_loss(),
                            mark_break_even(), update_trailing_stop(), close_position()
    break_even_threshold  : Anteil der TP-Distanz ab dem BE aktiviert wird (Standard: 0.5)
    spread_buffer_pips    : Puffer fuer BE-SL ueber/unter Eroeffnung (Standard: 2.0 Pips)
    pip_size              : Pip-Groesse (Standard: 0.0001 fuer Forex-Majors)
    """

    def __init__(
        self,
        connector,
        order_executor,
        break_even_threshold: float = 0.5,
        spread_buffer_pips: float = 2.0,
        pip_size: float = 0.0001,
    ) -> None:
        self._connector    = connector
        self._executor     = order_executor
        self._be_threshold = break_even_threshold
        self._buf_pips     = spread_buffer_pips
        self._pip_size     = pip_size
        self._contract_size_cache: dict[str, float] = {}

    # ── Oeffentliche Schnittstelle ────────────────────────────────────────────

    def manage(self, symbol: str) -> list[dict[str, Any]]:
        """
        Fuehrt Position-Management fuer alle offenen Positionen eines Symbols aus.

        Returns
        -------
        list[dict] mit Aktion je betroffener Position:
          {"action": "sl_hit"|"tp_hit"|"break_even"|"trailing",
           "ticket": int, ...}
        """
        positions = [
            p for p in self._executor.get_open_positions()
            if p.get("symbol") == symbol and p.get("status") == "open"
        ]
        if not positions:
            return []

        try:
            tick = self._connector.get_tick(symbol)
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "BreakEvenManager: Tick-Abruf fehlgeschlagen | {s}: {e}",
                s=symbol, e=exc,
            )
            return []

        actions: list[dict[str, Any]] = []
        for pos in positions:
            action = self._manage_one(pos, tick, symbol)
            if action:
                actions.append(action)
                logger.info(
                    "BreakEvenManager | {sym} | ticket={t} | action={a}",
                    sym=symbol, t=pos.get("ticket"), a=action.get("action"),
                )
        return actions

    # ── Interne Logik ─────────────────────────────────────────────────────────

    def _manage_one(
        self, pos: dict, tick: dict, symbol: str
    ) -> dict[str, Any] | None:
        ticket     = pos.get("ticket")
        direction  = pos.get("direction", "")
        open_price = pos.get("open_price")
        sl_price   = pos.get("sl_price")
        tp_price   = pos.get("tp_price")
        lot_size   = pos.get("lot_size", 0.0)

        if not all([open_price, sl_price, tp_price, direction]):
            return None

        bid = tick["bid"]
        ask = tick["ask"]
        current_price = bid if direction == "buy" else ask

        # 1. SL-Treffer -> Position schliessen
        if is_sl_hit(direction, sl_price, bid, ask):
            close_price = bid if direction == "buy" else ask
            pnl = calc_realized_pnl(
                direction, open_price, close_price, lot_size,
                self._get_contract_size(symbol),
            )
            self._executor.close_position(ticket, close_price=close_price, pnl=pnl)
            return {"action": "sl_hit", "ticket": ticket,
                    "close_price": close_price, "pnl": pnl}

        # 2. TP-Treffer -> Position schliessen
        if is_tp_hit(direction, tp_price, bid, ask):
            close_price = bid if direction == "buy" else ask
            pnl = calc_realized_pnl(
                direction, open_price, close_price, lot_size,
                self._get_contract_size(symbol),
            )
            self._executor.close_position(ticket, close_price=close_price, pnl=pnl)
            return {"action": "tp_hit", "ticket": ticket,
                    "close_price": close_price, "pnl": pnl}

        be_already_active = bool(pos.get("break_even_triggered"))

        # 3. Break-Even auslösen
        if not be_already_active:
            if should_trigger_break_even(
                direction, open_price, tp_price, current_price, self._be_threshold
            ):
                new_sl = calc_break_even_sl(
                    direction, open_price, self._buf_pips, self._pip_size
                )
                sl_improves = (
                    (direction == "buy"  and new_sl > sl_price) or
                    (direction == "sell" and new_sl < sl_price)
                )
                if sl_improves:
                    self._executor.set_stop_loss(ticket, new_sl)
                self._executor.mark_break_even(ticket)
                return {"action": "break_even", "ticket": ticket, "new_sl": new_sl}

        # 4. Trailing-Stop (nur wenn Break-Even bereits aktiv)
        if be_already_active:
            try:
                self._executor.update_trailing_stop(ticket, current_price)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Trailing-Stop Fehler ticket={t}: {e}", t=ticket, e=exc)
            return {"action": "trailing", "ticket": ticket}

        return None

    def _get_contract_size(self, symbol: str) -> float:
        """Gibt die Kontraktgroesse des Symbols zurueck (gecacht)."""
        if symbol not in self._contract_size_cache:
            try:
                info = self._connector.get_symbol_info(symbol)
                self._contract_size_cache[symbol] = float(
                    info.get("contract_size", 100_000.0)
                )
            except Exception:  # noqa: BLE001
                self._contract_size_cache[symbol] = 100_000.0
        return self._contract_size_cache[symbol]
