"""
tests/gui/test_dashboard_view.py
Tests fuer gui/views/dashboard_view.py.

Struktur:
  TestComputeRiskStatus   – pure Logik, kein Qt (alle 3 Ampelzustaende)
  TestFormatHelpers        – Formatierungsfunktionen, kein Qt
  TestDashboardSnapshot    – Datenklasse, kein Qt
  TestAccountCard          – Widget-Refresh mit Mock-Daten (pytest-qt)
  TestDrawdownGauge        – Fortschrittsbalken + Warnfarbe (pytest-qt)
  TestRiskTrafficLight     – Ampelfarbe und -text (pytest-qt)
  TestPositionsTable       – Zeilen und P&L-Farbe (pytest-qt)
  TestDailyStatsCard       – Tages-P&L, Trades, Win-Rate (pytest-qt)
  TestSignalPanel          – Signal-Anzeige + Konfidenz (pytest-qt)
  TestDashboardView        – Integration ohne Backend / mit Mock-Backend (pytest-qt)
  TestMainWindowIntegration – DashboardView in MainWindow integriert (pytest-qt)
"""

from __future__ import annotations

import pytest
from pytestqt.qtbot import QtBot

from gui.views.dashboard_view import (
    DashboardSnapshot,
    DashboardView,
    PositionInfo,
    RiskStatus,
    SignalInfo,
    _AccountCard,
    _DailyStatsCard,
    _DrawdownGauge,
    _PositionsTable,
    _RiskTrafficLight,
    _SignalPanel,
    compute_risk_status,
    _fmt_balance,
    _fmt_delta,
    _fmt_pct,
)
from gui.app import MainWindow, Section
from gui.design.theme import ThemeManager, ThemeMode


# ─────────────────────────────────────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def snap_empty() -> DashboardSnapshot:
    return DashboardSnapshot()


@pytest.fixture
def snap_full() -> DashboardSnapshot:
    return DashboardSnapshot(
        balance=10_500.00,
        day_start_balance=10_000.00,
        all_time_high=11_000.00,
        currency="€",
        drawdown_pct=4.5,
        drawdown_limit_pct=15.0,
        daily_loss_pct=1.2,
        daily_loss_limit_pct=5.0,
        post_loss_days_remaining=0,
        risk_status=RiskStatus.GREEN,
        risk_reasons=["Handel erlaubt"],
        positions=[
            PositionInfo(ticket=1, symbol="EURUSD", direction="long",
                         lot_size=0.10, open_price=1.0850, current_pnl=12.50),
            PositionInfo(ticket=2, symbol="GBPUSD", direction="short",
                         lot_size=0.05, open_price=1.2700, current_pnl=-8.00),
        ],
        today_trades=5,
        today_pnl=150.00,
        today_win_rate=0.6,
        signals=[
            SignalInfo(symbol="EURUSD", signal="long",  confidence=0.73),
            SignalInfo(symbol="GBPUSD", signal="flat",  confidence=0.51),
            SignalInfo(symbol="USDJPY", signal="short", confidence=0.62),
        ],
    )


class _MockBackend:
    def __init__(self, snapshot: DashboardSnapshot) -> None:
        self._snap = snapshot
        self.call_count = 0

    def fetch_snapshot(self) -> DashboardSnapshot:
        self.call_count += 1
        return self._snap


# ─────────────────────────────────────────────────────────────────────────────
#  compute_risk_status – pure Logik, kein Qt
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeRiskStatus:
    def test_green_when_all_ok(self):
        status, reasons = compute_risk_status(
            trading_allowed=True,
            daily_limit_hit=False,
            max_drawdown_hit=False,
        )
        assert status is RiskStatus.GREEN

    def test_green_reasons_contain_allowed(self):
        _, reasons = compute_risk_status(
            trading_allowed=True, daily_limit_hit=False, max_drawdown_hit=False
        )
        assert any("erlaubt" in r.lower() for r in reasons)

    def test_red_when_max_drawdown(self):
        status, reasons = compute_risk_status(
            trading_allowed=False,
            daily_limit_hit=False,
            max_drawdown_hit=True,
        )
        assert status is RiskStatus.RED
        assert any("drawdown" in r.lower() for r in reasons)

    def test_red_when_daily_limit(self):
        status, reasons = compute_risk_status(
            trading_allowed=False,
            daily_limit_hit=True,
            max_drawdown_hit=False,
        )
        assert status is RiskStatus.RED
        assert any("limit" in r.lower() or "verlust" in r.lower() for r in reasons)

    def test_red_when_anomaly(self):
        status, reasons = compute_risk_status(
            trading_allowed=True,
            daily_limit_hit=False,
            max_drawdown_hit=False,
            anomaly_detected=True,
        )
        assert status is RiskStatus.RED
        assert any("anomalie" in r.lower() for r in reasons)

    def test_red_when_trading_not_allowed(self):
        status, _ = compute_risk_status(
            trading_allowed=False,
            daily_limit_hit=False,
            max_drawdown_hit=False,
        )
        assert status is RiskStatus.RED

    def test_yellow_when_post_loss_days(self):
        status, reasons = compute_risk_status(
            trading_allowed=True,
            daily_limit_hit=False,
            max_drawdown_hit=False,
            post_loss_days=2,
        )
        assert status is RiskStatus.YELLOW
        assert any("post-loss" in r.lower() or "2" in r for r in reasons)

    def test_yellow_when_drawdown_near_limit(self):
        status, reasons = compute_risk_status(
            trading_allowed=True,
            daily_limit_hit=False,
            max_drawdown_hit=False,
            drawdown_pct=13.0,    # 13/15 = 86.7% > 80%
            drawdown_limit_pct=15.0,
        )
        assert status is RiskStatus.YELLOW
        assert any("drawdown" in r.lower() or "limit" in r.lower() for r in reasons)

    def test_yellow_not_triggered_below_warning(self):
        status, _ = compute_risk_status(
            trading_allowed=True,
            daily_limit_hit=False,
            max_drawdown_hit=False,
            drawdown_pct=5.0,
            drawdown_limit_pct=15.0,  # 5/15 = 33% < 80%
        )
        assert status is RiskStatus.GREEN

    def test_max_drawdown_takes_priority_over_daily_limit(self):
        """max_drawdown hat hoehere Prioritaet – wird als erstes geprueft."""
        status, reasons = compute_risk_status(
            trading_allowed=False,
            daily_limit_hit=True,
            max_drawdown_hit=True,
        )
        assert status is RiskStatus.RED
        assert any("drawdown" in r.lower() for r in reasons)

    def test_returns_list_of_reasons(self):
        _, reasons = compute_risk_status(
            trading_allowed=True, daily_limit_hit=False, max_drawdown_hit=False
        )
        assert isinstance(reasons, list)
        assert len(reasons) >= 1


# ─────────────────────────────────────────────────────────────────────────────
#  Format-Hilfsfunktionen – kein Qt
# ─────────────────────────────────────────────────────────────────────────────

class TestFormatHelpers:
    def test_fmt_balance_none(self):
        assert _fmt_balance(None) == "--"

    def test_fmt_balance_positive(self):
        result = _fmt_balance(10_500.0)
        assert "10" in result and "500" in result and "€" in result

    def test_fmt_delta_none(self):
        assert _fmt_delta(None) == "--"

    def test_fmt_delta_positive_has_plus(self):
        result = _fmt_delta(150.0)
        assert result.startswith("+")

    def test_fmt_delta_negative_no_plus(self):
        result = _fmt_delta(-50.0)
        assert not result.startswith("+")

    def test_fmt_pct_none(self):
        assert _fmt_pct(None) == "--"

    def test_fmt_pct_positive_has_plus(self):
        assert _fmt_pct(1.5).startswith("+")

    def test_fmt_pct_negative_no_plus(self):
        assert not _fmt_pct(-2.0).startswith("+")

    def test_fmt_pct_contains_symbol(self):
        assert "%" in _fmt_pct(3.7)


# ─────────────────────────────────────────────────────────────────────────────
#  DashboardSnapshot – kein Qt
# ─────────────────────────────────────────────────────────────────────────────

class TestDashboardSnapshot:
    def test_default_risk_status_green(self):
        assert DashboardSnapshot().risk_status is RiskStatus.GREEN

    def test_default_positions_empty(self):
        assert DashboardSnapshot().positions == []

    def test_default_signals_empty(self):
        assert DashboardSnapshot().signals == []

    def test_updated_at_is_set(self):
        snap = DashboardSnapshot()
        assert snap.updated_at != ""
        assert "T" in snap.updated_at or " " in snap.updated_at

    def test_custom_currency(self):
        snap = DashboardSnapshot(currency="$")
        assert snap.currency == "$"

    def test_new_fields_default_none(self):
        snap = DashboardSnapshot()
        assert snap.equity         is None
        assert snap.account_number is None
        assert snap.server         is None
        assert snap.leverage       is None
        assert snap.is_demo        is None

    def test_new_fields_set(self):
        snap = DashboardSnapshot(
            equity=9_800.0,
            account_number=383619,
            server="FusionMarkets-Demo",
            leverage=100,
            is_demo=True,
        )
        assert snap.equity         == 9_800.0
        assert snap.account_number == 383619
        assert snap.server         == "FusionMarkets-Demo"
        assert snap.leverage       == 100
        assert snap.is_demo        is True


# ─────────────────────────────────────────────────────────────────────────────
#  _AccountCard (Qt)
# ─────────────────────────────────────────────────────────────────────────────

class TestAccountCard:
    def test_creates(self, qtbot: QtBot):
        card = _AccountCard()
        qtbot.addWidget(card)
        assert card is not None

    def test_refresh_with_balance_shows_value(self, qtbot: QtBot, snap_full: DashboardSnapshot):
        card = _AccountCard()
        qtbot.addWidget(card)
        card.refresh(snap_full)
        assert "10" in card._balance_lbl.text()

    def test_refresh_no_data_shows_placeholder(self, qtbot: QtBot, snap_empty: DashboardSnapshot):
        card = _AccountCard()
        qtbot.addWidget(card)
        card.refresh(snap_empty)
        assert card._balance_lbl.text() == "--"

    def test_day_change_positive_shown(self, qtbot: QtBot, snap_full: DashboardSnapshot):
        card = _AccountCard()
        qtbot.addWidget(card)
        card.refresh(snap_full)  # balance 10500, start 10000 -> +500
        assert "+" in card._day_lbl.text()

    def test_day_change_negative_shown(self, qtbot: QtBot):
        card = _AccountCard()
        qtbot.addWidget(card)
        snap = DashboardSnapshot(balance=9_500.0, day_start_balance=10_000.0)
        card.refresh(snap)
        text = card._day_lbl.text()
        assert "-" in text or "500" in text

    def test_ath_shown(self, qtbot: QtBot, snap_full: DashboardSnapshot):
        card = _AccountCard()
        qtbot.addWidget(card)
        card.refresh(snap_full)
        assert "11" in card._ath_lbl.text() or "000" in card._ath_lbl.text()

    def test_equity_hidden_when_none(self, qtbot: QtBot):
        card = _AccountCard()
        qtbot.addWidget(card)
        card.refresh(DashboardSnapshot())
        assert card._equity_lbl.isHidden()

    def test_equity_shown_when_set(self, qtbot: QtBot):
        card = _AccountCard()
        qtbot.addWidget(card)
        card.refresh(DashboardSnapshot(balance=10_000.0, equity=9_900.0))
        assert not card._equity_lbl.isHidden()
        assert "9" in card._equity_lbl.text()

    def test_account_details_hidden_when_empty(self, qtbot: QtBot):
        card = _AccountCard()
        qtbot.addWidget(card)
        card.refresh(DashboardSnapshot())
        assert card._account_details_lbl.isHidden()

    def test_account_details_shows_number(self, qtbot: QtBot):
        card = _AccountCard()
        qtbot.addWidget(card)
        card.refresh(DashboardSnapshot(account_number=383619))
        assert not card._account_details_lbl.isHidden()
        assert "383619" in card._account_details_lbl.text()

    def test_account_details_shows_demo(self, qtbot: QtBot):
        card = _AccountCard()
        qtbot.addWidget(card)
        card.refresh(DashboardSnapshot(is_demo=True))
        assert "Demo" in card._account_details_lbl.text()

    def test_account_details_shows_live(self, qtbot: QtBot):
        card = _AccountCard()
        qtbot.addWidget(card)
        card.refresh(DashboardSnapshot(is_demo=False))
        assert "Live" in card._account_details_lbl.text()

    def test_account_details_shows_leverage(self, qtbot: QtBot):
        card = _AccountCard()
        qtbot.addWidget(card)
        card.refresh(DashboardSnapshot(leverage=100))
        assert "100" in card._account_details_lbl.text()

    def test_account_details_shows_server(self, qtbot: QtBot):
        card = _AccountCard()
        qtbot.addWidget(card)
        card.refresh(DashboardSnapshot(server="FusionMarkets-Demo"))
        assert "FusionMarkets" in card._account_details_lbl.text()


# ─────────────────────────────────────────────────────────────────────────────
#  _DrawdownGauge (Qt)
# ─────────────────────────────────────────────────────────────────────────────

class TestDrawdownGauge:
    def test_creates(self, qtbot: QtBot):
        g = _DrawdownGauge()
        qtbot.addWidget(g)
        assert g is not None

    def test_zero_drawdown_bar_value_zero(self, qtbot: QtBot):
        g = _DrawdownGauge()
        qtbot.addWidget(g)
        g.refresh(DashboardSnapshot(drawdown_pct=0.0, drawdown_limit_pct=15.0))
        assert g._bar.value() == 0

    def test_normal_drawdown_bar_correct(self, qtbot: QtBot):
        g = _DrawdownGauge()
        qtbot.addWidget(g)
        g.refresh(DashboardSnapshot(drawdown_pct=7.5, drawdown_limit_pct=15.0))
        # 7.5 / 15 * 100 = 50
        assert g._bar.value() == 50

    def test_at_limit_bar_at_100(self, qtbot: QtBot):
        g = _DrawdownGauge()
        qtbot.addWidget(g)
        g.refresh(DashboardSnapshot(drawdown_pct=15.0, drawdown_limit_pct=15.0))
        assert g._bar.value() == 100

    def test_warning_color_at_80_pct_of_limit(self, qtbot: QtBot):
        g = _DrawdownGauge()
        qtbot.addWidget(g)
        # 12/15 = 80% -> warnung
        g.refresh(DashboardSnapshot(drawdown_pct=12.0, drawdown_limit_pct=15.0))
        qss = g._bar.styleSheet()
        assert "#f59e0b" in qss or "#ef4444" in qss  # amber oder rot

    def test_normal_color_below_warning(self, qtbot: QtBot):
        g = _DrawdownGauge()
        qtbot.addWidget(g)
        g.refresh(DashboardSnapshot(drawdown_pct=2.0, drawdown_limit_pct=15.0))
        qss = g._bar.styleSheet()
        assert "#6366f1" in qss  # akzent/indigo

    def test_label_shows_pct_and_limit(self, qtbot: QtBot):
        g = _DrawdownGauge()
        qtbot.addWidget(g)
        g.refresh(DashboardSnapshot(drawdown_pct=4.5, drawdown_limit_pct=15.0))
        text = g._pct_lbl.text()
        assert "4.5" in text and "15.0" in text


# ─────────────────────────────────────────────────────────────────────────────
#  _RiskTrafficLight (Qt)
# ─────────────────────────────────────────────────────────────────────────────

class TestRiskTrafficLight:
    def test_creates(self, qtbot: QtBot):
        w = _RiskTrafficLight()
        qtbot.addWidget(w)
        assert w is not None

    def test_green_status_shows_green_color(self, qtbot: QtBot):
        w = _RiskTrafficLight()
        qtbot.addWidget(w)
        snap = DashboardSnapshot(risk_status=RiskStatus.GREEN, risk_reasons=["ok"])
        w.refresh(snap)
        assert "#22c55e" in w.dot_label.styleSheet()

    def test_yellow_status_shows_amber(self, qtbot: QtBot):
        w = _RiskTrafficLight()
        qtbot.addWidget(w)
        snap = DashboardSnapshot(risk_status=RiskStatus.YELLOW, risk_reasons=["Warnung"])
        w.refresh(snap)
        assert "#f59e0b" in w.dot_label.styleSheet()

    def test_red_status_shows_red(self, qtbot: QtBot):
        w = _RiskTrafficLight()
        qtbot.addWidget(w)
        snap = DashboardSnapshot(risk_status=RiskStatus.RED, risk_reasons=["Gesperrt"])
        w.refresh(snap)
        assert "#ef4444" in w.dot_label.styleSheet()

    def test_status_label_text_green(self, qtbot: QtBot):
        w = _RiskTrafficLight()
        qtbot.addWidget(w)
        w.refresh(DashboardSnapshot(risk_status=RiskStatus.GREEN))
        assert "erlaubt" in w.status_label.text().lower()

    def test_status_label_text_red(self, qtbot: QtBot):
        w = _RiskTrafficLight()
        qtbot.addWidget(w)
        w.refresh(DashboardSnapshot(risk_status=RiskStatus.RED))
        assert "gesperrt" in w.status_label.text().lower()

    def test_reason_shown_in_label(self, qtbot: QtBot):
        w = _RiskTrafficLight()
        qtbot.addWidget(w)
        w.refresh(DashboardSnapshot(
            risk_status=RiskStatus.YELLOW,
            risk_reasons=["Post-Loss-Phase aktiv"],
        ))
        assert "Post-Loss" in w._reason_lbl.text()


# ─────────────────────────────────────────────────────────────────────────────
#  _PositionsTable (Qt)
# ─────────────────────────────────────────────────────────────────────────────

class TestPositionsTable:
    def test_creates(self, qtbot: QtBot):
        t = _PositionsTable()
        qtbot.addWidget(t)
        assert t is not None

    def test_empty_positions_zero_rows(self, qtbot: QtBot, snap_empty: DashboardSnapshot):
        t = _PositionsTable()
        qtbot.addWidget(t)
        t.refresh(snap_empty)
        assert t.table.rowCount() == 0

    def test_two_positions_two_rows(self, qtbot: QtBot, snap_full: DashboardSnapshot):
        t = _PositionsTable()
        qtbot.addWidget(t)
        t.refresh(snap_full)
        assert t.table.rowCount() == 2

    def test_position_symbol_in_table(self, qtbot: QtBot, snap_full: DashboardSnapshot):
        t = _PositionsTable()
        qtbot.addWidget(t)
        t.refresh(snap_full)
        first_cell = t.table.item(0, 0)
        assert first_cell is not None
        assert "EURUSD" in first_cell.text()

    def test_positive_pnl_shown(self, qtbot: QtBot, snap_full: DashboardSnapshot):
        t = _PositionsTable()
        qtbot.addWidget(t)
        t.refresh(snap_full)
        pnl_item = t.table.item(0, 4)
        assert pnl_item is not None
        assert "+" in pnl_item.text()

    def test_negative_pnl_shown(self, qtbot: QtBot, snap_full: DashboardSnapshot):
        t = _PositionsTable()
        qtbot.addWidget(t)
        t.refresh(snap_full)
        pnl_item = t.table.item(1, 4)
        assert pnl_item is not None
        assert "-" in pnl_item.text()

    def test_six_columns(self, qtbot: QtBot):
        t = _PositionsTable()
        qtbot.addWidget(t)
        assert t.table.columnCount() == 6  # 5 data cols + Schliessen-Button col

    def test_header_labels_full_text(self, qtbot: QtBot):
        from PySide6.QtWidgets import QHeaderView
        t = _PositionsTable()
        qtbot.addWidget(t)
        labels = [
            t.table.horizontalHeaderItem(i).text()
            for i in range(t.table.columnCount())
        ]
        assert labels == ["Symbol", "Richtung", "Lots", "Eröffnung", "P&L", ""]

    def test_header_resize_mode_is_resize_to_contents(self, qtbot: QtBot):
        from PySide6.QtWidgets import QHeaderView
        t = _PositionsTable()
        qtbot.addWidget(t)
        hdr = t.table.horizontalHeader()
        # First 4 sections should be ResizeToContents; last stretches
        for i in range(4):
            assert hdr.sectionResizeMode(i) == QHeaderView.ResizeMode.ResizeToContents

    def test_last_section_stretches(self, qtbot: QtBot):
        from PySide6.QtWidgets import QHeaderView
        t = _PositionsTable()
        qtbot.addWidget(t)
        hdr = t.table.horizontalHeader()
        assert hdr.sectionResizeMode(4) == QHeaderView.ResizeMode.Stretch


# ─────────────────────────────────────────────────────────────────────────────
#  _DailyStatsCard (Qt)
# ─────────────────────────────────────────────────────────────────────────────

class TestDailyStatsCard:
    def test_creates(self, qtbot: QtBot):
        c = _DailyStatsCard()
        qtbot.addWidget(c)
        assert c is not None

    def test_shows_trade_count(self, qtbot: QtBot, snap_full: DashboardSnapshot):
        c = _DailyStatsCard()
        qtbot.addWidget(c)
        c.refresh(snap_full)
        assert "5" in c.trades_label.text()

    def test_shows_day_pnl(self, qtbot: QtBot, snap_full: DashboardSnapshot):
        c = _DailyStatsCard()
        qtbot.addWidget(c)
        c.refresh(snap_full)
        assert "150" in c.pnl_label.text()

    def test_shows_win_rate(self, qtbot: QtBot, snap_full: DashboardSnapshot):
        c = _DailyStatsCard()
        qtbot.addWidget(c)
        c.refresh(snap_full)  # 0.6 -> 60.0%
        assert "60" in c.winrate_label.text()

    def test_no_data_shows_placeholder(self, qtbot: QtBot, snap_empty: DashboardSnapshot):
        c = _DailyStatsCard()
        qtbot.addWidget(c)
        c.refresh(snap_empty)
        assert c.pnl_label.text() == "--"
        assert c.winrate_label.text() == "--"


# ─────────────────────────────────────────────────────────────────────────────
#  _SignalPanel (Qt)
# ─────────────────────────────────────────────────────────────────────────────

class TestSignalPanel:
    def test_creates(self, qtbot: QtBot):
        p = _SignalPanel()
        qtbot.addWidget(p)
        assert p is not None

    def test_empty_signals_shows_empty_label(self, qtbot: QtBot, snap_empty: DashboardSnapshot):
        p = _SignalPanel()
        qtbot.addWidget(p)
        p.refresh(snap_empty)
        # isHidden() prueft den expliziten Hide-Status, unabhaengig vom Parent-Fenster
        assert not p.empty_label.isHidden()

    def test_with_signals_hides_empty_label(self, qtbot: QtBot, snap_full: DashboardSnapshot):
        p = _SignalPanel()
        qtbot.addWidget(p)
        p.refresh(snap_full)
        assert p.empty_label.isHidden()

    def test_three_signals_three_rows(self, qtbot: QtBot, snap_full: DashboardSnapshot):
        p = _SignalPanel()
        qtbot.addWidget(p)
        p.refresh(snap_full)
        assert p._rows_layout.count() == 3

    def test_signal_direction_shown(self, qtbot: QtBot, snap_full: DashboardSnapshot):
        p = _SignalPanel()
        qtbot.addWidget(p)
        p.refresh(snap_full)
        # Erste Zeile = EURUSD long
        first_row = p._rows_layout.itemAt(0).widget()
        assert "LONG" in first_row.signal_label.text()

    def test_confidence_shown(self, qtbot: QtBot, snap_full: DashboardSnapshot):
        p = _SignalPanel()
        qtbot.addWidget(p)
        p.refresh(snap_full)
        first_row = p._rows_layout.itemAt(0).widget()
        assert "73" in first_row.confidence_label.text()


# ─────────────────────────────────────────────────────────────────────────────
#  DashboardView (Qt)
# ─────────────────────────────────────────────────────────────────────────────

class TestDashboardView:
    def test_creates_without_backend(self, qtbot: QtBot):
        v = DashboardView()
        qtbot.addWidget(v)
        assert v is not None

    def test_update_display_no_crash(self, qtbot: QtBot, snap_full: DashboardSnapshot):
        v = DashboardView()
        qtbot.addWidget(v)
        v.update_display(snap_full)
        assert v.last_snapshot is snap_full

    def test_update_display_empty_snap_no_crash(self, qtbot: QtBot, snap_empty: DashboardSnapshot):
        v = DashboardView()
        qtbot.addWidget(v)
        v.update_display(snap_empty)

    def test_polling_not_active_without_backend(self, qtbot: QtBot):
        v = DashboardView()
        qtbot.addWidget(v)
        v.start_polling()  # kein Backend -> kein Start
        assert not v.is_polling

    def test_polling_starts_with_backend(self, qtbot: QtBot, snap_full: DashboardSnapshot):
        backend = _MockBackend(snap_full)
        v = DashboardView(backend=backend)
        qtbot.addWidget(v)
        v.start_polling()
        assert v.is_polling
        v.stop_polling()

    def test_polling_stops(self, qtbot: QtBot, snap_full: DashboardSnapshot):
        backend = _MockBackend(snap_full)
        v = DashboardView(backend=backend)
        qtbot.addWidget(v)
        v.start_polling()
        v.stop_polling()
        assert not v.is_polling

    def test_set_backend_then_poll(self, qtbot: QtBot, snap_full: DashboardSnapshot):
        v = DashboardView()
        qtbot.addWidget(v)
        backend = _MockBackend(snap_full)
        v.set_backend(backend)
        v.start_polling()
        assert v.is_polling
        v.stop_polling()

    def test_data_refreshed_signal_emitted(self, qtbot: QtBot, snap_full: DashboardSnapshot):
        """Nach manuellem update_display wird last_snapshot gesetzt."""
        v = DashboardView()
        qtbot.addWidget(v)
        v.update_display(snap_full)
        assert v.last_snapshot.balance == snap_full.balance

    def test_subwidgets_accessible(self, qtbot: QtBot):
        v = DashboardView()
        qtbot.addWidget(v)
        assert v.account_card is not None
        assert v.drawdown_gauge is not None
        assert v.risk_light is not None
        assert v.daily_stats is not None
        assert v.positions_table is not None
        assert v.signal_panel is not None

    def test_timer_fires_and_calls_backend(self, qtbot: QtBot, snap_full: DashboardSnapshot):
        """QTimer ruft Backend auf wenn aktiv."""
        backend = _MockBackend(snap_full)
        v = DashboardView(backend=backend, interval_ms=50)
        qtbot.addWidget(v)
        v.start_polling()
        qtbot.wait(200)  # warte >= 1 Timer-Tick
        v.stop_polling()
        assert backend.call_count >= 1


# ─────────────────────────────────────────────────────────────────────────────
#  MainWindow-Integration (Qt)
# ─────────────────────────────────────────────────────────────────────────────

class TestMainWindowIntegration:
    def test_main_window_has_dashboard_view(self, qtbot: QtBot):
        tm = ThemeManager()
        w = MainWindow(theme_manager=tm)
        qtbot.addWidget(w)
        assert isinstance(w.dashboard_view, DashboardView)

    def test_dashboard_is_first_view(self, qtbot: QtBot):
        tm = ThemeManager()
        w = MainWindow(theme_manager=tm)
        qtbot.addWidget(w)
        assert w.content.count() == 7
        assert w.content.widget(0) is w.dashboard_view

    def test_navigate_to_dashboard_shows_view(self, qtbot: QtBot):
        tm = ThemeManager()
        w = MainWindow(theme_manager=tm)
        qtbot.addWidget(w)
        w.navigate_to(Section.SETTINGS)
        w.navigate_to(Section.DASHBOARD)
        assert w.current_view() is w.dashboard_view

    def test_dashboard_update_from_main_window(self, qtbot: QtBot, snap_full: DashboardSnapshot):
        tm = ThemeManager()
        w = MainWindow(theme_manager=tm)
        qtbot.addWidget(w)
        w.dashboard_view.update_display(snap_full)
        assert w.dashboard_view.last_snapshot.balance == snap_full.balance

    def test_main_window_with_backend(self, qtbot: QtBot, snap_full: DashboardSnapshot):
        tm = ThemeManager()
        backend = _MockBackend(snap_full)
        w = MainWindow(theme_manager=tm, dashboard_backend=backend)
        qtbot.addWidget(w)
        assert w.dashboard_view is not None
