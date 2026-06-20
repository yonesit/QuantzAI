"""
tests/gui/test_trade_confirmation_dialog.py
Tests fuer TradeProposal, TradeConfirmationBanner und GuiConfirmationCallback.

Prueft:
  - Widget-Erstellung und UI-Inhalte
  - Alle drei Aktionen: Bestaetigen / Ablehnen / Lot anpassen
  - Countdown-Ablauf und timed_out-Signal
  - Thread-sichere confirm_order() aus Worker-Thread
  - Audit-Log-Aufrufe bei allen Aktionen
  - Orchestrator-Integration: angepasste Lot-Groesse, set_confirmation_callback()
  - MainWindow-Integration: confirmation_callback-Property
"""

from __future__ import annotations

import threading
import time
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

# ── pytest-qt-Fixture ────────────────────────────────────────────────────────
pytest_plugins = ["pytestqt"]

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QWidget

from gui.dialogs.trade_confirmation_dialog import (
    GuiConfirmationCallback,
    TradeConfirmationBanner,
    TradeProposal,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Hilfsfunktionen
# ─────────────────────────────────────────────────────────────────────────────

def _proposal(
    symbol: str = "EURUSD",
    direction: str = "buy",
    lot_size: float = 0.10,
    sl_price: float = 1.0800,
    tp_price: float = 1.1200,
    confidence: Optional[float] = None,
    spread_cost: Optional[float] = None,
) -> TradeProposal:
    return TradeProposal(
        symbol=symbol,
        direction=direction,
        lot_size=lot_size,
        sl_price=sl_price,
        tp_price=tp_price,
        confidence=confidence,
        spread_cost=spread_cost,
    )


def _call_confirm_in_thread(
    callback: GuiConfirmationCallback,
    symbol: str = "EURUSD",
    direction: str = "buy",
    lot_size: float = 0.10,
    sl: float = 1.08,
    tp: float = 1.12,
) -> tuple[threading.Thread, list]:
    """Startet confirm_order() in einem Thread und gibt (thread, result[]) zurueck."""
    result: list[Optional[bool]] = [None]

    def _run() -> None:
        result[0] = callback.confirm_order(symbol, direction, lot_size, sl, tp)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t, result


# ─────────────────────────────────────────────────────────────────────────────
#  1.  TradeProposal
# ─────────────────────────────────────────────────────────────────────────────

class TestTradeProposal:
    def test_required_fields(self):
        p = TradeProposal("EURUSD", "buy", 0.10, 1.08, 1.12)
        assert p.symbol    == "EURUSD"
        assert p.direction == "buy"
        assert p.lot_size  == pytest.approx(0.10)
        assert p.sl_price  == pytest.approx(1.08)
        assert p.tp_price  == pytest.approx(1.12)

    def test_optional_defaults_none(self):
        p = TradeProposal("GBPUSD", "sell", 0.05, 1.25, 1.20)
        assert p.confidence  is None
        assert p.spread_cost is None

    def test_confidence_field(self):
        p = TradeProposal("EURUSD", "buy", 0.1, 1.08, 1.12, confidence=0.72)
        assert p.confidence == pytest.approx(0.72)

    def test_spread_cost_field(self):
        p = TradeProposal("EURUSD", "buy", 0.1, 1.08, 1.12, spread_cost=2.50)
        assert p.spread_cost == pytest.approx(2.50)

    def test_sell_direction(self):
        p = TradeProposal("USDJPY", "sell", 0.20, 145.0, 142.0)
        assert p.direction == "sell"


# ─────────────────────────────────────────────────────────────────────────────
#  2.  TradeConfirmationBanner – Initialisierung
# ─────────────────────────────────────────────────────────────────────────────

class TestTradeConfirmationBannerInit:
    def test_creates_without_crash(self, qtbot):
        banner = TradeConfirmationBanner(_proposal())
        qtbot.addWidget(banner)

    def test_object_name(self, qtbot):
        banner = TradeConfirmationBanner(_proposal())
        qtbot.addWidget(banner)
        assert banner.objectName() == "trade_confirmation_banner"

    def test_shows_symbol(self, qtbot):
        banner = TradeConfirmationBanner(_proposal(symbol="GBPUSD"))
        qtbot.addWidget(banner)
        sym = banner.findChild(QWidget, "confirmation_symbol")
        assert sym is not None
        assert "GBPUSD" in sym.text()

    def test_shows_direction_buy(self, qtbot):
        banner = TradeConfirmationBanner(_proposal(direction="buy"))
        qtbot.addWidget(banner)
        dir_lbl = banner.findChild(QWidget, "confirmation_direction")
        assert "BUY" in dir_lbl.text()

    def test_shows_direction_sell(self, qtbot):
        banner = TradeConfirmationBanner(_proposal(direction="sell"))
        qtbot.addWidget(banner)
        dir_lbl = banner.findChild(QWidget, "confirmation_direction")
        assert "SELL" in dir_lbl.text()

    def test_shows_lot_size(self, qtbot):
        banner = TradeConfirmationBanner(_proposal(lot_size=0.25))
        qtbot.addWidget(banner)
        lot_lbl = banner.findChild(QWidget, "confirmation_lot_display")
        assert "0.25" in lot_lbl.text()

    def test_shows_sl_tp(self, qtbot):
        banner = TradeConfirmationBanner(_proposal(sl_price=1.0800, tp_price=1.1200))
        qtbot.addWidget(banner)
        sl_lbl = banner.findChild(QWidget, "confirmation_sl")
        tp_lbl = banner.findChild(QWidget, "confirmation_tp")
        assert sl_lbl is not None and "1.08" in sl_lbl.text()
        assert tp_lbl is not None and "1.12" in tp_lbl.text()

    def test_shows_countdown_initial(self, qtbot):
        banner = TradeConfirmationBanner(_proposal(), timeout_seconds=30)
        qtbot.addWidget(banner)
        countdown = banner.findChild(QWidget, "confirmation_countdown")
        assert "30" in countdown.text()

    def test_has_confirm_button(self, qtbot):
        banner = TradeConfirmationBanner(_proposal())
        qtbot.addWidget(banner)
        btn = banner.findChild(QWidget, "confirmation_confirm_btn")
        assert btn is not None
        assert btn.isEnabled()

    def test_has_reject_button(self, qtbot):
        banner = TradeConfirmationBanner(_proposal())
        qtbot.addWidget(banner)
        btn = banner.findChild(QWidget, "confirmation_reject_btn")
        assert btn is not None
        assert btn.isEnabled()

    def test_has_lot_spinbox(self, qtbot):
        banner = TradeConfirmationBanner(_proposal(lot_size=0.15))
        qtbot.addWidget(banner)
        spin = banner.findChild(QWidget, "confirmation_lot_spin")
        assert spin is not None
        assert spin.value() == pytest.approx(0.15)

    def test_shows_confidence_when_given(self, qtbot):
        banner = TradeConfirmationBanner(_proposal(confidence=0.78))
        qtbot.addWidget(banner)
        conf_lbl = banner.findChild(QWidget, "confirmation_confidence")
        assert conf_lbl is not None
        assert "78" in conf_lbl.text()

    def test_shows_spread_when_given(self, qtbot):
        banner = TradeConfirmationBanner(_proposal(spread_cost=3.20))
        qtbot.addWidget(banner)
        cost_lbl = banner.findChild(QWidget, "confirmation_spread")
        assert cost_lbl is not None
        assert "3.20" in cost_lbl.text()

    def test_no_confidence_label_when_not_given(self, qtbot):
        banner = TradeConfirmationBanner(_proposal())
        qtbot.addWidget(banner)
        assert banner.findChild(QWidget, "confirmation_confidence") is None

    def test_not_resolved_initially(self, qtbot):
        banner = TradeConfirmationBanner(_proposal())
        qtbot.addWidget(banner)
        assert banner.is_resolved is False

    def test_minimum_width(self, qtbot):
        banner = TradeConfirmationBanner(_proposal())
        qtbot.addWidget(banner)
        assert banner.minimumWidth() >= 580

    def test_proposal_property(self, qtbot):
        p = _proposal(symbol="XAUUSD")
        banner = TradeConfirmationBanner(p)
        qtbot.addWidget(banner)
        assert banner.proposal is p


# ─────────────────────────────────────────────────────────────────────────────
#  3.  TradeConfirmationBanner – Bestaetigen
# ─────────────────────────────────────────────────────────────────────────────

class TestTradeConfirmationBannerConfirm:
    def test_confirm_emits_confirmed_signal(self, qtbot):
        banner = TradeConfirmationBanner(_proposal(lot_size=0.10))
        qtbot.addWidget(banner)
        with qtbot.waitSignal(banner.confirmed, timeout=2000) as blocker:
            qtbot.mouseClick(banner._confirm_btn, Qt.MouseButton.LeftButton)
        assert blocker.args[0] == pytest.approx(0.10)

    def test_confirm_uses_spinbox_lot(self, qtbot):
        banner = TradeConfirmationBanner(_proposal(lot_size=0.10))
        qtbot.addWidget(banner)
        banner._lot_spin.setValue(0.25)
        with qtbot.waitSignal(banner.confirmed, timeout=2000) as blocker:
            qtbot.mouseClick(banner._confirm_btn, Qt.MouseButton.LeftButton)
        assert blocker.args[0] == pytest.approx(0.25)

    def test_confirm_disables_buttons(self, qtbot):
        banner = TradeConfirmationBanner(_proposal())
        qtbot.addWidget(banner)
        qtbot.mouseClick(banner._confirm_btn, Qt.MouseButton.LeftButton)
        qtbot.waitSignal(banner.confirmed, timeout=2000)
        assert not banner._confirm_btn.isEnabled()
        assert not banner._reject_btn.isEnabled()

    def test_is_resolved_after_confirm(self, qtbot):
        banner = TradeConfirmationBanner(_proposal())
        qtbot.addWidget(banner)
        with qtbot.waitSignal(banner.confirmed, timeout=2000):
            qtbot.mouseClick(banner._confirm_btn, Qt.MouseButton.LeftButton)
        assert banner.is_resolved is True

    def test_confirmed_not_emitted_twice(self, qtbot):
        banner = TradeConfirmationBanner(_proposal())
        qtbot.addWidget(banner)
        count = [0]
        banner.confirmed.connect(lambda _: count.__setitem__(0, count[0] + 1))
        qtbot.mouseClick(banner._confirm_btn, Qt.MouseButton.LeftButton)
        qtbot.mouseClick(banner._confirm_btn, Qt.MouseButton.LeftButton)
        qtbot.wait(100)
        assert count[0] == 1

    def test_rejected_not_emitted_on_confirm(self, qtbot):
        banner = TradeConfirmationBanner(_proposal())
        qtbot.addWidget(banner)
        rejected_count = [0]
        banner.rejected.connect(lambda: rejected_count.__setitem__(0, rejected_count[0] + 1))
        with qtbot.waitSignal(banner.confirmed, timeout=2000):
            qtbot.mouseClick(banner._confirm_btn, Qt.MouseButton.LeftButton)
        assert rejected_count[0] == 0


# ─────────────────────────────────────────────────────────────────────────────
#  4.  TradeConfirmationBanner – Ablehnen
# ─────────────────────────────────────────────────────────────────────────────

class TestTradeConfirmationBannerReject:
    def test_reject_emits_rejected_signal(self, qtbot):
        banner = TradeConfirmationBanner(_proposal())
        qtbot.addWidget(banner)
        with qtbot.waitSignal(banner.rejected, timeout=2000):
            qtbot.mouseClick(banner._reject_btn, Qt.MouseButton.LeftButton)

    def test_reject_disables_buttons(self, qtbot):
        banner = TradeConfirmationBanner(_proposal())
        qtbot.addWidget(banner)
        with qtbot.waitSignal(banner.rejected, timeout=2000):
            qtbot.mouseClick(banner._reject_btn, Qt.MouseButton.LeftButton)
        assert not banner._confirm_btn.isEnabled()
        assert not banner._reject_btn.isEnabled()

    def test_is_resolved_after_reject(self, qtbot):
        banner = TradeConfirmationBanner(_proposal())
        qtbot.addWidget(banner)
        with qtbot.waitSignal(banner.rejected, timeout=2000):
            qtbot.mouseClick(banner._reject_btn, Qt.MouseButton.LeftButton)
        assert banner.is_resolved is True

    def test_rejected_not_emitted_twice(self, qtbot):
        banner = TradeConfirmationBanner(_proposal())
        qtbot.addWidget(banner)
        count = [0]
        banner.rejected.connect(lambda: count.__setitem__(0, count[0] + 1))
        qtbot.mouseClick(banner._reject_btn, Qt.MouseButton.LeftButton)
        qtbot.mouseClick(banner._reject_btn, Qt.MouseButton.LeftButton)
        qtbot.wait(100)
        assert count[0] == 1

    def test_confirmed_not_emitted_on_reject(self, qtbot):
        banner = TradeConfirmationBanner(_proposal())
        qtbot.addWidget(banner)
        confirmed_count = [0]
        banner.confirmed.connect(lambda _: confirmed_count.__setitem__(0, confirmed_count[0] + 1))
        with qtbot.waitSignal(banner.rejected, timeout=2000):
            qtbot.mouseClick(banner._reject_btn, Qt.MouseButton.LeftButton)
        assert confirmed_count[0] == 0


# ─────────────────────────────────────────────────────────────────────────────
#  5.  TradeConfirmationBanner – Timeout
# ─────────────────────────────────────────────────────────────────────────────

class TestTradeConfirmationBannerTimeout:
    def test_timed_out_emitted_after_countdown(self, qtbot):
        banner = TradeConfirmationBanner(_proposal(), timeout_seconds=1)
        qtbot.addWidget(banner)
        with qtbot.waitSignal(banner.timed_out, timeout=3000):
            pass  # Signal sollte nach ~1 s eintreffen

    def test_is_resolved_after_timeout(self, qtbot):
        banner = TradeConfirmationBanner(_proposal(), timeout_seconds=1)
        qtbot.addWidget(banner)
        with qtbot.waitSignal(banner.timed_out, timeout=3000):
            pass
        assert banner.is_resolved is True

    def test_countdown_label_updates(self, qtbot):
        banner = TradeConfirmationBanner(_proposal(), timeout_seconds=5)
        qtbot.addWidget(banner)
        initial_text = banner._countdown_label.text()
        assert "5" in initial_text
        qtbot.wait(1200)
        updated_text = banner._countdown_label.text()
        assert updated_text != initial_text

    def test_countdown_color_changes_at_10s(self, qtbot):
        banner = TradeConfirmationBanner(_proposal(), timeout_seconds=1)
        qtbot.addWidget(banner)
        with qtbot.waitSignal(banner.timed_out, timeout=3000):
            pass
        style = banner._countdown_label.styleSheet()
        assert "#ef4444" in style

    def test_timed_out_not_emitted_after_confirm(self, qtbot):
        banner = TradeConfirmationBanner(_proposal(), timeout_seconds=1)
        qtbot.addWidget(banner)
        timeout_count = [0]
        banner.timed_out.connect(lambda: timeout_count.__setitem__(0, timeout_count[0] + 1))
        with qtbot.waitSignal(banner.confirmed, timeout=2000):
            qtbot.mouseClick(banner._confirm_btn, Qt.MouseButton.LeftButton)
        qtbot.wait(1500)  # Countdown-Zeit abwarten
        assert timeout_count[0] == 0

    def test_remaining_seconds_property(self, qtbot):
        banner = TradeConfirmationBanner(_proposal(), timeout_seconds=45)
        qtbot.addWidget(banner)
        assert banner.remaining_seconds == 45


# ─────────────────────────────────────────────────────────────────────────────
#  6.  GuiConfirmationCallback – Initialisierung
# ─────────────────────────────────────────────────────────────────────────────

class TestGuiConfirmationCallbackInit:
    def test_creates_without_crash(self, qtbot):
        cb = GuiConfirmationCallback()
        cb.deleteLater()

    def test_has_confirm_order_method(self, qtbot):
        cb = GuiConfirmationCallback()
        assert callable(cb.confirm_order)
        cb.deleteLater()

    def test_last_confirmed_lot_size_initially_none(self, qtbot):
        cb = GuiConfirmationCallback()
        assert cb.last_confirmed_lot_size is None
        cb.deleteLater()

    def test_implements_confirmation_callback_protocol(self, qtbot):
        from src.modes import ConfirmationCallback
        cb = GuiConfirmationCallback()
        assert isinstance(cb, ConfirmationCallback)
        cb.deleteLater()

    def test_timeout_parameter(self, qtbot):
        cb = GuiConfirmationCallback(timeout_seconds=30)
        assert cb._timeout == 30
        cb.deleteLater()

    def test_audit_fn_parameter(self, qtbot):
        audit = MagicMock()
        cb = GuiConfirmationCallback(audit_fn=audit)
        assert cb._audit_fn is audit
        cb.deleteLater()


# ─────────────────────────────────────────────────────────────────────────────
#  7.  GuiConfirmationCallback – Bestaetigen aus Thread
# ─────────────────────────────────────────────────────────────────────────────

class TestGuiConfirmationCallbackConfirm:
    def test_confirm_order_returns_true_on_confirm(self, qtbot):
        cb = GuiConfirmationCallback(timeout_seconds=5)
        t, result = _call_confirm_in_thread(cb)
        qtbot.waitUntil(lambda: cb._banner is not None, timeout=3000)
        with qtbot.waitSignal(cb._banner.confirmed, timeout=2000):
            qtbot.mouseClick(cb._banner._confirm_btn, Qt.MouseButton.LeftButton)
        t.join(timeout=3)
        assert result[0] is True
        cb.deleteLater()

    def test_banner_shown_after_confirm_order_called(self, qtbot):
        cb = GuiConfirmationCallback(timeout_seconds=5)
        t, _ = _call_confirm_in_thread(cb)
        qtbot.waitUntil(lambda: cb._banner is not None, timeout=3000)
        assert cb._banner is not None
        # Aufraeum
        cb._banner._resolve(False)
        t.join(timeout=3)
        cb.deleteLater()

    def test_last_confirmed_lot_size_reflects_original(self, qtbot):
        cb = GuiConfirmationCallback(timeout_seconds=5)
        t, result = _call_confirm_in_thread(cb, lot_size=0.10)
        qtbot.waitUntil(lambda: cb._banner is not None, timeout=3000)
        with qtbot.waitSignal(cb._banner.confirmed, timeout=2000):
            qtbot.mouseClick(cb._banner._confirm_btn, Qt.MouseButton.LeftButton)
        t.join(timeout=3)
        assert cb.last_confirmed_lot_size == pytest.approx(0.10)
        cb.deleteLater()

    def test_last_confirmed_lot_size_reflects_adjusted(self, qtbot):
        cb = GuiConfirmationCallback(timeout_seconds=5)
        t, result = _call_confirm_in_thread(cb, lot_size=0.10)
        qtbot.waitUntil(lambda: cb._banner is not None, timeout=3000)
        cb._banner._lot_spin.setValue(0.05)
        with qtbot.waitSignal(cb._banner.confirmed, timeout=2000):
            qtbot.mouseClick(cb._banner._confirm_btn, Qt.MouseButton.LeftButton)
        t.join(timeout=3)
        assert result[0] is True
        assert cb.last_confirmed_lot_size == pytest.approx(0.05)
        cb.deleteLater()

    def test_symbol_in_banner(self, qtbot):
        cb = GuiConfirmationCallback(timeout_seconds=5)
        t, _ = _call_confirm_in_thread(cb, symbol="XAUUSD")
        qtbot.waitUntil(lambda: cb._banner is not None, timeout=3000)
        sym_lbl = cb._banner.findChild(QWidget, "confirmation_symbol")
        assert "XAUUSD" in sym_lbl.text()
        cb._banner._resolve(False)
        t.join(timeout=3)
        cb.deleteLater()


# ─────────────────────────────────────────────────────────────────────────────
#  8.  GuiConfirmationCallback – Ablehnen aus Thread
# ─────────────────────────────────────────────────────────────────────────────

class TestGuiConfirmationCallbackReject:
    def test_confirm_order_returns_false_on_reject(self, qtbot):
        cb = GuiConfirmationCallback(timeout_seconds=5)
        t, result = _call_confirm_in_thread(cb)
        qtbot.waitUntil(lambda: cb._banner is not None, timeout=3000)
        with qtbot.waitSignal(cb._banner.rejected, timeout=2000):
            qtbot.mouseClick(cb._banner._reject_btn, Qt.MouseButton.LeftButton)
        t.join(timeout=3)
        assert result[0] is False
        cb.deleteLater()

    def test_last_lot_size_none_after_rejection(self, qtbot):
        cb = GuiConfirmationCallback(timeout_seconds=5)
        t, _ = _call_confirm_in_thread(cb)
        qtbot.waitUntil(lambda: cb._banner is not None, timeout=3000)
        with qtbot.waitSignal(cb._banner.rejected, timeout=2000):
            qtbot.mouseClick(cb._banner._reject_btn, Qt.MouseButton.LeftButton)
        t.join(timeout=3)
        assert cb.last_confirmed_lot_size is None
        cb.deleteLater()

    def test_banner_buttons_disabled_after_reject(self, qtbot):
        cb = GuiConfirmationCallback(timeout_seconds=5)
        t, _ = _call_confirm_in_thread(cb)
        qtbot.waitUntil(lambda: cb._banner is not None, timeout=3000)
        with qtbot.waitSignal(cb._banner.rejected, timeout=2000):
            qtbot.mouseClick(cb._banner._reject_btn, Qt.MouseButton.LeftButton)
        t.join(timeout=3)
        assert not cb._banner._confirm_btn.isEnabled()
        assert not cb._banner._reject_btn.isEnabled()
        cb.deleteLater()


# ─────────────────────────────────────────────────────────────────────────────
#  9.  GuiConfirmationCallback – Timeout
# ─────────────────────────────────────────────────────────────────────────────

class TestGuiConfirmationCallbackTimeout:
    def test_confirm_order_returns_false_on_timeout(self, qtbot):
        cb = GuiConfirmationCallback(timeout_seconds=1)
        t, result = _call_confirm_in_thread(cb)
        t.join(timeout=5)
        assert result[0] is False
        cb.deleteLater()

    def test_timeout_seconds_configurable(self, qtbot):
        cb_fast = GuiConfirmationCallback(timeout_seconds=1)
        t, result = _call_confirm_in_thread(cb_fast)
        start = time.monotonic()
        t.join(timeout=4)
        elapsed = time.monotonic() - start
        assert elapsed < 4.0   # sollte deutlich vor 4 s enden
        assert result[0] is False
        cb_fast.deleteLater()

    def test_last_lot_none_after_timeout(self, qtbot):
        cb = GuiConfirmationCallback(timeout_seconds=1)
        t, _ = _call_confirm_in_thread(cb)
        t.join(timeout=5)
        assert cb.last_confirmed_lot_size is None
        cb.deleteLater()


# ─────────────────────────────────────────────────────────────────────────────
#  10. Audit-Log
# ─────────────────────────────────────────────────────────────────────────────

class TestGuiConfirmationCallbackAudit:
    def test_audit_fn_called_on_confirm(self, qtbot):
        audit = MagicMock()
        cb = GuiConfirmationCallback(timeout_seconds=5, audit_fn=audit)
        t, _ = _call_confirm_in_thread(cb)
        qtbot.waitUntil(lambda: cb._banner is not None, timeout=3000)
        with qtbot.waitSignal(cb._banner.confirmed, timeout=2000):
            qtbot.mouseClick(cb._banner._confirm_btn, Qt.MouseButton.LeftButton)
        t.join(timeout=3)
        audit.assert_called_once()
        action, data = audit.call_args[0]
        assert action == "ORDER_CONFIRMED"
        cb.deleteLater()

    def test_audit_fn_called_on_reject(self, qtbot):
        audit = MagicMock()
        cb = GuiConfirmationCallback(timeout_seconds=5, audit_fn=audit)
        t, _ = _call_confirm_in_thread(cb)
        qtbot.waitUntil(lambda: cb._banner is not None, timeout=3000)
        with qtbot.waitSignal(cb._banner.rejected, timeout=2000):
            qtbot.mouseClick(cb._banner._reject_btn, Qt.MouseButton.LeftButton)
        t.join(timeout=3)
        audit.assert_called_once()
        action, data = audit.call_args[0]
        assert action == "ORDER_REJECTED"
        cb.deleteLater()

    def test_audit_fn_called_on_timeout(self, qtbot):
        audit = MagicMock()
        cb = GuiConfirmationCallback(timeout_seconds=1, audit_fn=audit)
        t, _ = _call_confirm_in_thread(cb)
        t.join(timeout=5)
        audit.assert_called_once()
        action, data = audit.call_args[0]
        assert action == "ORDER_REJECTED"
        assert data["timed_out"] is True
        cb.deleteLater()

    def test_audit_data_contains_symbol(self, qtbot):
        audit = MagicMock()
        cb = GuiConfirmationCallback(timeout_seconds=5, audit_fn=audit)
        t, _ = _call_confirm_in_thread(cb, symbol="GBPJPY")
        qtbot.waitUntil(lambda: cb._banner is not None, timeout=3000)
        with qtbot.waitSignal(cb._banner.confirmed, timeout=2000):
            qtbot.mouseClick(cb._banner._confirm_btn, Qt.MouseButton.LeftButton)
        t.join(timeout=3)
        _, data = audit.call_args[0]
        assert data["symbol"] == "GBPJPY"
        cb.deleteLater()

    def test_audit_data_contains_lot_size(self, qtbot):
        audit = MagicMock()
        cb = GuiConfirmationCallback(timeout_seconds=5, audit_fn=audit)
        t, _ = _call_confirm_in_thread(cb, lot_size=0.33)
        qtbot.waitUntil(lambda: cb._banner is not None, timeout=3000)
        with qtbot.waitSignal(cb._banner.confirmed, timeout=2000):
            qtbot.mouseClick(cb._banner._confirm_btn, Qt.MouseButton.LeftButton)
        t.join(timeout=3)
        _, data = audit.call_args[0]
        assert data["lot_size"] == pytest.approx(0.33)
        cb.deleteLater()

    def test_audit_fn_not_called_if_none(self, qtbot):
        cb = GuiConfirmationCallback(timeout_seconds=5, audit_fn=None)
        t, result = _call_confirm_in_thread(cb)
        qtbot.waitUntil(lambda: cb._banner is not None, timeout=3000)
        with qtbot.waitSignal(cb._banner.confirmed, timeout=2000):
            qtbot.mouseClick(cb._banner._confirm_btn, Qt.MouseButton.LeftButton)
        t.join(timeout=3)
        assert result[0] is True  # kein Absturz
        cb.deleteLater()


# ─────────────────────────────────────────────────────────────────────────────
#  11. Orchestrator-Integration
# ─────────────────────────────────────────────────────────────────────────────

class TestOrchestratorIntegration:
    def _make_orchestrator(self):
        """Minimaler Mock-Orchestrator fuer CONFIRM_REQUIRED-Tests."""
        from src.orchestrator import TradingOrchestrator
        from src.modes import TradingMode

        orc = MagicMock(spec=TradingOrchestrator)
        orc._mode = TradingMode.CONFIRM_REQUIRED
        orc._confirmation_callback = None
        orc._activity_callback = None
        orc.set_confirmation_callback = TradingOrchestrator.set_confirmation_callback.__get__(orc)  # type: ignore[assignment]
        return orc

    def test_set_confirmation_callback_exists(self):
        from src.orchestrator import TradingOrchestrator
        assert hasattr(TradingOrchestrator, "set_confirmation_callback")

    def test_set_confirmation_callback_sets_attribute(self):
        from src.orchestrator import TradingOrchestrator
        orc = MagicMock()
        orc._confirmation_callback = None
        TradingOrchestrator.set_confirmation_callback(orc, None)
        assert orc._confirmation_callback is None

    def test_lot_adjustment_applied_in_run_cycle(self, qtbot):
        """
        GuiConfirmationCallback mit angepasster Lot-Groesse wird vom Orchestrator
        uebernommen: last_confirmed_lot_size fliesst in den Order-Aufruf ein.
        """
        from src.orchestrator import TradingOrchestrator
        from src.modes import TradingMode

        # Echter minimaler Orchestrator mit Mocks
        from src.risk.position_sizer import PositionSizeResult
        dp   = MagicMock()
        rg   = MagicMock()
        rg.is_trading_allowed.return_value          = True
        rg.get_position_size_multiplier.return_value = 1.0
        ptc  = MagicMock(); ptc.is_safe_to_trade.return_value  = (True, "ok")
        sm   = MagicMock(); sm.get_signal.return_value          = "long"
        sm.predict_proba.return_value = {"long": 0.7, "short": 0.3}
        cg   = MagicMock(); cg.can_open_position.return_value  = True
        ps   = MagicMock()
        ps.calculate_lot_size.return_value = PositionSizeResult(
            symbol="EURUSD", lot_size=0.10, risk_amount=100.0,
            stop_loss_distance=0.0015, is_valid=True,
        )
        oe   = MagicMock()
        oe.open_position.return_value  = {"ticket": 42}
        oe.get_open_positions.return_value = []

        features = MagicMock()
        features.__len__ = MagicMock(return_value=10)
        features.empty = False
        features.__getitem__ = MagicMock(return_value=MagicMock())
        features.iloc = MagicMock()
        features.iloc.__getitem__ = MagicMock(return_value={"close": 1.1, "atr": 0.001})

        dp.run_batch.return_value = {"EURUSD": features}
        sm.get_signal.return_value = "long"

        def fake_features_loader(_sym):
            import pandas as pd
            return pd.DataFrame({
                "close": [1.1] * 10,
                "atr":   [0.001] * 10,
            })

        cb = GuiConfirmationCallback(timeout_seconds=5)

        orc = TradingOrchestrator(
            data_pipeline=dp,
            risk_guard=rg,
            pre_trade_check=ptc,
            signal_model=sm,
            correlation_guard=cg,
            position_sizer=ps,
            order_executor=oe,
            mode=TradingMode.CONFIRM_REQUIRED,
            confirmation_callback=cb,
            features_loader=fake_features_loader,
        )

        def _run_cycle():
            return orc.run_cycle("EURUSD")

        # confirm_order aus Thread; Banner erscheint im Hauptthread
        result_holder: list = [None]

        def _worker():
            result_holder[0] = orc.run_cycle("EURUSD")

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

        qtbot.waitUntil(lambda: cb._banner is not None, timeout=4000)
        cb._banner._lot_spin.setValue(0.05)
        with qtbot.waitSignal(cb._banner.confirmed, timeout=2000):
            qtbot.mouseClick(cb._banner._confirm_btn, Qt.MouseButton.LeftButton)

        t.join(timeout=5)
        assert result_holder[0] is not None

        # Der OrderExecutor muss mit der angepassten Lot-Groesse aufgerufen worden sein
        call_args = oe.open_position.call_args
        assert call_args is not None
        _, _, lot_arg, _, _ = call_args[0]
        assert lot_arg == pytest.approx(0.05)
        cb.deleteLater()

    def test_set_confirmation_callback_updates_runtime(self):
        from src.orchestrator import TradingOrchestrator
        from src.modes import TradingMode

        dp  = MagicMock()
        orc = TradingOrchestrator(
            data_pipeline=dp,
            risk_guard=MagicMock(),
            pre_trade_check=MagicMock(),
            signal_model=MagicMock(),
            correlation_guard=MagicMock(),
            position_sizer=MagicMock(),
            order_executor=MagicMock(),
            mode=TradingMode.CONFIRM_REQUIRED,
        )
        assert orc._confirmation_callback is None
        cb = MagicMock()
        orc.set_confirmation_callback(cb)
        assert orc._confirmation_callback is cb


# ─────────────────────────────────────────────────────────────────────────────
#  12. MainWindow-Integration
# ─────────────────────────────────────────────────────────────────────────────

class TestMainWindowIntegration:
    def test_main_window_has_confirmation_callback(self, qtbot):
        from gui.app import MainWindow
        win = MainWindow()
        qtbot.addWidget(win)
        assert hasattr(win, "confirmation_callback")
        assert isinstance(win.confirmation_callback, GuiConfirmationCallback)

    def test_custom_confirmation_callback_injected(self, qtbot):
        from gui.app import MainWindow
        custom_cb = GuiConfirmationCallback(timeout_seconds=10)
        win = MainWindow(confirmation_callback=custom_cb)
        qtbot.addWidget(win)
        assert win.confirmation_callback is custom_cb
        custom_cb.deleteLater()

    def test_default_confirmation_callback_created_if_none(self, qtbot):
        from gui.app import MainWindow
        win = MainWindow()
        qtbot.addWidget(win)
        cb = win.confirmation_callback
        assert cb is not None
        assert cb._timeout == 60

    def test_confirmation_callback_not_none_after_build(self, qtbot):
        from gui.app import MainWindow
        win = MainWindow()
        qtbot.addWidget(win)
        assert win.confirmation_callback is not None
