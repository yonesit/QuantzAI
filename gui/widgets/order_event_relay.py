"""
gui/widgets/order_event_relay.py
OrderEventRelay – Qt-Signal-Bruecke fuer OrderExecutor.

OrderExecutor ist kein QObject und hat keine Qt-Abhaengigkeit.
Dieser Relay wandelt die Python-Callable-Callbacks des Executors in
Qt-Signale um, die thread-sicher in den GUI-Hauptthread zugestellt werden.

Verwendung:
    relay = OrderEventRelay(parent=main_window)
    relay.attach(order_executor)
    relay.order_opened.connect(dashboard_view.on_order_opened)
    relay.order_opened.connect(cockpit_view.on_order_opened)
    relay.order_closed.connect(dashboard_view.on_order_closed)
    relay.order_closed.connect(cockpit_view.on_order_closed)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QWidget

if TYPE_CHECKING:
    from src.execution.order_executor import OrderExecutor


class OrderEventRelay(QObject):
    """
    Empfaengt OrderExecutor-Callbacks (aus beliebigem Thread) und
    stellt sie als Qt-Signale zu – thread-sicher via automatischer
    QueuedConnection bei Cross-Thread-Emission.

    Signals
    -------
    order_opened(dict)   – nach jedem erfolgreichen open_position()
    order_closed(dict)   – nach jedem erfolgreichen close_position()
    """

    order_opened: Signal = Signal(dict)
    order_closed: Signal = Signal(dict)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

    def attach(self, executor: "OrderExecutor") -> None:
        """Verbindet diesen Relay mit einem OrderExecutor."""
        executor.set_order_callbacks(
            on_open=self.order_opened.emit,
            on_close=self.order_closed.emit,
        )

    def detach(self, executor: "OrderExecutor") -> None:
        """Trennt die Verbindung zum Executor (setzt Callbacks auf None)."""
        executor.set_order_callbacks(on_open=None, on_close=None)
