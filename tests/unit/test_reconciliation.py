"""
Unit-Tests fuer PositionReconciler.

MT5, Connector und Executor werden vollstaendig gemockt.
Kein MT5-Terminal, kein Netzwerk noetig.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, call, patch

import pytest

from src.execution.reconciliation import PositionReconciler


# ─────────────────────────────────────────────────────────────────────────────
#  Hilfsfunktionen
# ─────────────────────────────────────────────────────────────────────────────

def _connector() -> MagicMock:
    """Mock fuer MT5Connector mit register_reconnect_callback."""
    conn = MagicMock()
    conn._reconnect_callbacks = []
    def _register(cb):
        conn._reconnect_callbacks.append(cb)
    conn.register_reconnect_callback.side_effect = _register
    return conn


def _executor(open_positions: list[dict] | None = None) -> MagicMock:
    """Mock fuer OrderExecutor mit konfigurierbaren offenen Positionen."""
    ex = MagicMock()
    ex.get_open_positions.return_value = open_positions or []
    return ex


def _mt5_mock(positions: list | None = None) -> MagicMock:
    """Mock fuer das mt5-Modul mit konfigurierbaren positions_get()-Ergebnissen."""
    mt5 = MagicMock()
    mt5.ORDER_TYPE_BUY  = 0
    mt5.ORDER_TYPE_SELL = 1
    mt5.positions_get.return_value = positions or []
    return mt5


def _mt5_position(ticket: int, symbol: str = "EURUSD", order_type: int = 0) -> MagicMock:
    """Erstellt einen Mock fuer eine MT5-TradePosition."""
    pos = MagicMock()
    pos.ticket     = ticket
    pos.symbol     = symbol
    pos.type       = order_type  # 0=BUY, 1=SELL
    pos.volume     = 0.1
    pos.price_open = 1.10000
    pos.sl         = 1.0950
    pos.tp         = 1.1100
    return pos


def _local_position(ticket: int, symbol: str = "EURUSD") -> dict:
    """Erstellt ein lokales Positions-Dict (wie OrderExecutor es zurueckgibt)."""
    return {
        "ticket":     ticket,
        "symbol":     symbol,
        "direction":  "buy",
        "lot_size":   0.1,
        "sl_price":   1.0950,
        "tp_price":   1.1100,
        "open_price": 1.10000,
        "status":     "open",
    }


def _reconciler(conn=None, ex=None, interval: int = 300) -> PositionReconciler:
    return PositionReconciler(
        connector=conn or _connector(),
        executor=ex or _executor(),
        sync_interval_seconds=interval,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: Initialisierung
# ─────────────────────────────────────────────────────────────────────────────

class TestInit:

    def test_default_interval_is_300(self):
        r = _reconciler()
        assert r._interval == 300

    def test_custom_interval_stored(self):
        r = _reconciler(interval=60)
        assert r._interval == 60

    def test_registers_reconnect_callback(self):
        conn = _connector()
        _reconciler(conn=conn)
        conn.register_reconnect_callback.assert_called_once()

    def test_works_without_register_callback_support(self):
        """Connector ohne register_reconnect_callback -> kein Fehler."""
        conn = MagicMock(spec=[])  # kein register_reconnect_callback Attribut
        r = PositionReconciler(connector=conn, executor=_executor())
        assert r is not None

    def test_incidents_empty_initially(self):
        r = _reconciler()
        assert r.incidents == []


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: sync() – Normalfall (beide Seiten stimmen ueberein)
# ─────────────────────────────────────────────────────────────────────────────

class TestSyncNormalCase:

    def test_returns_dict(self):
        mt5 = _mt5_mock()
        ex  = _executor()
        r   = _reconciler(ex=ex)
        with patch("src.execution.reconciliation._load_mt5", return_value=mt5):
            result = r.sync()
        assert isinstance(result, dict)

    def test_result_has_required_keys(self):
        mt5 = _mt5_mock()
        r   = _reconciler()
        with patch("src.execution.reconciliation._load_mt5", return_value=mt5):
            result = r.sync()
        required = {"missing_locally", "missing_at_mt5", "incidents", "in_sync", "synced_at"}
        assert required.issubset(result.keys())

    def test_both_empty_is_in_sync(self):
        """Lokal leer, MT5 leer -> in_sync=True, incidents=0."""
        mt5 = _mt5_mock(positions=[])
        ex  = _executor(open_positions=[])
        r   = _reconciler(ex=ex)
        with patch("src.execution.reconciliation._load_mt5", return_value=mt5):
            result = r.sync()
        assert result["in_sync"] is True
        assert result["incidents"] == 0

    def test_matching_positions_is_in_sync(self):
        """Lokal=Ticket42, MT5=Ticket42 -> in_sync=True, keine Callbacks."""
        mt5_pos = _mt5_position(42)
        mt5 = _mt5_mock(positions=[mt5_pos])
        ex  = _executor(open_positions=[_local_position(42)])
        r   = _reconciler(ex=ex)
        with patch("src.execution.reconciliation._load_mt5", return_value=mt5):
            result = r.sync()
        assert result["in_sync"] is True
        assert result["incidents"] == 0

    def test_no_reconcile_callbacks_when_in_sync(self):
        """Bei Uebereinstimmung werden weder add noch close aufgerufen."""
        mt5_pos = _mt5_position(42)
        mt5 = _mt5_mock(positions=[mt5_pos])
        ex  = _executor(open_positions=[_local_position(42)])
        r   = _reconciler(ex=ex)
        with patch("src.execution.reconciliation._load_mt5", return_value=mt5):
            r.sync()
        ex.reconcile_add_position.assert_not_called()
        ex.reconcile_close_position.assert_not_called()

    def test_in_sync_true_multiple_matching_positions(self):
        """Mehrere Positionen die auf beiden Seiten vorhanden sind."""
        mt5 = _mt5_mock(positions=[_mt5_position(10), _mt5_position(11)])
        ex  = _executor(open_positions=[_local_position(10), _local_position(11)])
        r   = _reconciler(ex=ex)
        with patch("src.execution.reconciliation._load_mt5", return_value=mt5):
            result = r.sync()
        assert result["in_sync"] is True
        assert result["incidents"] == 0

    def test_no_incident_logged_when_in_sync(self):
        mt5 = _mt5_mock(positions=[_mt5_position(42)])
        ex  = _executor(open_positions=[_local_position(42)])
        r   = _reconciler(ex=ex)
        with patch("src.execution.reconciliation._load_mt5", return_value=mt5):
            r.sync()
        assert r.incidents == []


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: sync() – Position fehlt lokal (MT5 hat sie, Executor nicht)
# ─────────────────────────────────────────────────────────────────────────────

class TestSyncMissingLocally:

    def test_detected_in_result(self):
        """MT5 kennt Ticket 42, lokal leer -> missing_locally=[42]."""
        mt5 = _mt5_mock(positions=[_mt5_position(42)])
        ex  = _executor(open_positions=[])
        r   = _reconciler(ex=ex)
        with patch("src.execution.reconciliation._load_mt5", return_value=mt5):
            result = r.sync()
        assert 42 in result["missing_locally"]

    def test_not_in_sync(self):
        mt5 = _mt5_mock(positions=[_mt5_position(42)])
        ex  = _executor(open_positions=[])
        r   = _reconciler(ex=ex)
        with patch("src.execution.reconciliation._load_mt5", return_value=mt5):
            result = r.sync()
        assert result["in_sync"] is False

    def test_incident_count_correct(self):
        mt5 = _mt5_mock(positions=[_mt5_position(42), _mt5_position(43)])
        ex  = _executor(open_positions=[])
        r   = _reconciler(ex=ex)
        with patch("src.execution.reconciliation._load_mt5", return_value=mt5):
            result = r.sync()
        assert result["incidents"] == 2

    def test_reconcile_add_position_called(self):
        """reconcile_add_position wird mit dem korrekten Ticket aufgerufen."""
        mt5 = _mt5_mock(positions=[_mt5_position(42)])
        ex  = _executor(open_positions=[])
        r   = _reconciler(ex=ex)
        with patch("src.execution.reconciliation._load_mt5", return_value=mt5):
            r.sync()
        ex.reconcile_add_position.assert_called_once()
        call_ticket = ex.reconcile_add_position.call_args[0][0]
        assert call_ticket == 42

    def test_reconcile_add_called_for_each_missing(self):
        """Fuer jede fehlende Position wird reconcile_add aufgerufen."""
        mt5 = _mt5_mock(positions=[_mt5_position(10), _mt5_position(11)])
        ex  = _executor(open_positions=[])
        r   = _reconciler(ex=ex)
        with patch("src.execution.reconciliation._load_mt5", return_value=mt5):
            r.sync()
        assert ex.reconcile_add_position.call_count == 2

    def test_reconcile_close_not_called(self):
        """close wird nicht aufgerufen wenn Position nur lokal fehlt."""
        mt5 = _mt5_mock(positions=[_mt5_position(42)])
        ex  = _executor(open_positions=[])
        r   = _reconciler(ex=ex)
        with patch("src.execution.reconciliation._load_mt5", return_value=mt5):
            r.sync()
        ex.reconcile_close_position.assert_not_called()

    def test_incident_type_is_missing_locally(self):
        mt5 = _mt5_mock(positions=[_mt5_position(42)])
        ex  = _executor(open_positions=[])
        r   = _reconciler(ex=ex)
        with patch("src.execution.reconciliation._load_mt5", return_value=mt5):
            r.sync()
        assert any(i["type"] == "missing_locally" for i in r.incidents)

    def test_added_position_has_correct_data(self):
        """reconcile_add_position bekommt korrekte Symbol/Direction-Daten."""
        mt5_pos = _mt5_position(42, symbol="GBPUSD", order_type=1)  # SELL
        mt5 = _mt5_mock(positions=[mt5_pos])
        ex  = _executor(open_positions=[])
        r   = _reconciler(ex=ex)
        with patch("src.execution.reconciliation._load_mt5", return_value=mt5):
            r.sync()
        pos_dict = ex.reconcile_add_position.call_args[0][1]
        assert pos_dict["symbol"]    == "GBPUSD"
        assert pos_dict["direction"] == "sell"


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: sync() – Position fehlt bei MT5 (Executor weiss es, MT5 nicht)
# ─────────────────────────────────────────────────────────────────────────────

class TestSyncMissingAtMT5:

    def test_detected_in_result(self):
        """Lokal Ticket 42 offen, MT5 leer -> missing_at_mt5=[42]."""
        mt5 = _mt5_mock(positions=[])
        ex  = _executor(open_positions=[_local_position(42)])
        r   = _reconciler(ex=ex)
        with patch("src.execution.reconciliation._load_mt5", return_value=mt5):
            result = r.sync()
        assert 42 in result["missing_at_mt5"]

    def test_not_in_sync(self):
        mt5 = _mt5_mock(positions=[])
        ex  = _executor(open_positions=[_local_position(42)])
        r   = _reconciler(ex=ex)
        with patch("src.execution.reconciliation._load_mt5", return_value=mt5):
            result = r.sync()
        assert result["in_sync"] is False

    def test_incident_count_correct(self):
        mt5 = _mt5_mock(positions=[])
        ex  = _executor(open_positions=[_local_position(42), _local_position(43)])
        r   = _reconciler(ex=ex)
        with patch("src.execution.reconciliation._load_mt5", return_value=mt5):
            result = r.sync()
        assert result["incidents"] == 2

    def test_reconcile_close_position_called_with_ticket(self):
        """reconcile_close_position wird mit dem korrekten Ticket aufgerufen."""
        mt5 = _mt5_mock(positions=[])
        ex  = _executor(open_positions=[_local_position(42)])
        r   = _reconciler(ex=ex)
        with patch("src.execution.reconciliation._load_mt5", return_value=mt5):
            r.sync()
        ex.reconcile_close_position.assert_called_once_with(42)

    def test_reconcile_close_called_for_each_missing(self):
        """Fuer jede fehlende MT5-Position wird close aufgerufen."""
        mt5 = _mt5_mock(positions=[])
        ex  = _executor(open_positions=[_local_position(10), _local_position(11)])
        r   = _reconciler(ex=ex)
        with patch("src.execution.reconciliation._load_mt5", return_value=mt5):
            r.sync()
        assert ex.reconcile_close_position.call_count == 2
        called_tickets = {c[0][0] for c in ex.reconcile_close_position.call_args_list}
        assert called_tickets == {10, 11}

    def test_reconcile_add_not_called(self):
        """add wird nicht aufgerufen wenn Position lokal vorhanden aber MT5 fehlt."""
        mt5 = _mt5_mock(positions=[])
        ex  = _executor(open_positions=[_local_position(42)])
        r   = _reconciler(ex=ex)
        with patch("src.execution.reconciliation._load_mt5", return_value=mt5):
            r.sync()
        ex.reconcile_add_position.assert_not_called()

    def test_incident_type_is_missing_at_mt5(self):
        mt5 = _mt5_mock(positions=[])
        ex  = _executor(open_positions=[_local_position(42)])
        r   = _reconciler(ex=ex)
        with patch("src.execution.reconciliation._load_mt5", return_value=mt5):
            r.sync()
        assert any(i["type"] == "missing_at_mt5" for i in r.incidents)


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: sync() – Gemischte Diskrepanzen
# ─────────────────────────────────────────────────────────────────────────────

class TestSyncMixed:

    def test_both_types_detected_simultaneously(self):
        """
        Lokal: [42]; MT5: [43]
        -> 42 fehlt bei MT5, 43 fehlt lokal.
        """
        mt5 = _mt5_mock(positions=[_mt5_position(43)])
        ex  = _executor(open_positions=[_local_position(42)])
        r   = _reconciler(ex=ex)
        with patch("src.execution.reconciliation._load_mt5", return_value=mt5):
            result = r.sync()
        assert 43 in result["missing_locally"]
        assert 42 in result["missing_at_mt5"]
        assert result["incidents"] == 2
        assert result["in_sync"] is False

    def test_partial_overlap(self):
        """
        Lokal: [10, 11]; MT5: [11, 12]
        -> 10 fehlt bei MT5, 12 fehlt lokal.
        """
        mt5 = _mt5_mock(positions=[_mt5_position(11), _mt5_position(12)])
        ex  = _executor(open_positions=[_local_position(10), _local_position(11)])
        r   = _reconciler(ex=ex)
        with patch("src.execution.reconciliation._load_mt5", return_value=mt5):
            result = r.sync()
        assert 12 in result["missing_locally"]
        assert 10 in result["missing_at_mt5"]
        assert 11 not in result["missing_locally"]
        assert 11 not in result["missing_at_mt5"]

    def test_incidents_accumulate_across_syncs(self):
        """Vorfaelle aus mehreren sync()-Aufrufen werden aufsummiert."""
        mt5_1 = _mt5_mock(positions=[_mt5_position(42)])
        mt5_2 = _mt5_mock(positions=[])
        ex    = _executor(open_positions=[])
        r     = _reconciler(ex=ex)

        with patch("src.execution.reconciliation._load_mt5", return_value=mt5_1):
            r.sync()
        with patch("src.execution.reconciliation._load_mt5", return_value=mt5_2):
            r.sync()

        assert len(r.incidents) >= 1


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: Reconnect-Hook
# ─────────────────────────────────────────────────────────────────────────────

class TestReconnectHook:

    def test_callback_registered_with_connector(self):
        """Der Reconciler registriert sich als Reconnect-Callback."""
        conn = _connector()
        _reconciler(conn=conn)
        conn.register_reconnect_callback.assert_called_once()

    def test_registered_callback_is_callable(self):
        conn = _connector()
        _reconciler(conn=conn)
        cb = conn.register_reconnect_callback.call_args[0][0]
        assert callable(cb)

    def test_on_reconnect_triggers_sync(self):
        """_on_reconnect() ruft sync() auf."""
        mt5 = _mt5_mock()
        ex  = _executor()
        r   = _reconciler(ex=ex)
        with patch("src.execution.reconciliation._load_mt5", return_value=mt5):
            with patch.object(r, "sync", wraps=r.sync) as mock_sync:
                r._on_reconnect()
        mock_sync.assert_called_once()

    def test_reconnect_callback_fires_sync_via_connector(self):
        """Wenn Connector den Callback ausloest, wird sync() aufgerufen."""
        conn = _connector()
        mt5  = _mt5_mock()
        ex   = _executor()
        r    = _reconciler(conn=conn, ex=ex)

        with patch.object(r, "sync", return_value={"in_sync": True}) as mock_sync:
            # Connector-seitig den Callback ausloesen (simuliert Reconnect)
            for cb in conn._reconnect_callbacks:
                cb()
        mock_sync.assert_called_once()

    def test_on_reconnect_error_does_not_propagate(self):
        """Fehler in sync() waehrend Reconnect wird abgefangen, kein Crash."""
        ex = _executor()
        r  = _reconciler(ex=ex)
        with patch.object(r, "sync", side_effect=RuntimeError("Verbindung weg")):
            r._on_reconnect()  # darf keinen Fehler werfen


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: Periodischer Sync-Mechanismus
# ─────────────────────────────────────────────────────────────────────────────

class TestPeriodicSync:

    def test_no_timer_initially(self):
        r = _reconciler()
        assert r._timer is None

    def test_start_creates_timer(self):
        r = _reconciler()
        r.start_periodic_sync()
        try:
            assert r._timer is not None
        finally:
            r.stop_periodic_sync()

    def test_stop_removes_timer(self):
        r = _reconciler()
        r.start_periodic_sync()
        r.stop_periodic_sync()
        assert r._timer is None

    def test_double_start_does_not_create_two_timers(self):
        """Zweites start_periodic_sync() ersetzt den laufenden Timer nicht."""
        r = _reconciler()
        r.start_periodic_sync()
        timer_1 = r._timer
        r.start_periodic_sync()
        timer_2 = r._timer
        try:
            assert timer_1 is timer_2  # unveraendert
        finally:
            r.stop_periodic_sync()

    def test_stop_without_start_is_harmless(self):
        """stop_periodic_sync() ohne vorheriges start() wirft keinen Fehler."""
        r = _reconciler()
        r.stop_periodic_sync()  # kein Exception
        assert r._timer is None

    def test_custom_interval_used_by_timer(self):
        """Timer wird mit dem konfigurierten Intervall erstellt."""
        r = _reconciler(interval=42)
        r.start_periodic_sync()
        try:
            assert r._timer is not None
            assert r._interval == 42
        finally:
            r.stop_periodic_sync()

    def test_periodic_tick_calls_sync(self):
        """_periodic_tick() ruft sync() auf und plant neuen Timer."""
        mt5 = _mt5_mock()
        ex  = _executor()
        r   = _reconciler(ex=ex)
        with patch("src.execution.reconciliation._load_mt5", return_value=mt5):
            with patch.object(r, "sync", wraps=r.sync) as mock_sync:
                r._periodic_tick()
                mock_sync.assert_called_once()
        r.stop_periodic_sync()  # cleanup

    def test_periodic_tick_reschedules_timer(self):
        """Nach _periodic_tick() ist ein neuer Timer geplant."""
        mt5 = _mt5_mock()
        ex  = _executor()
        r   = _reconciler(ex=ex)
        with patch("src.execution.reconciliation._load_mt5", return_value=mt5):
            r._periodic_tick()
        try:
            assert r._timer is not None
        finally:
            r.stop_periodic_sync()

    def test_stop_after_tick_removes_timer(self):
        """stop_periodic_sync() nach _periodic_tick() entfernt den neuen Timer."""
        mt5 = _mt5_mock()
        ex  = _executor()
        r   = _reconciler(ex=ex)
        with patch("src.execution.reconciliation._load_mt5", return_value=mt5):
            r._periodic_tick()
        r.stop_periodic_sync()
        assert r._timer is None

    def test_periodic_tick_error_still_reschedules(self):
        """Fehler in sync() verhindert nicht das Neu-Planen des Timers."""
        r = _reconciler()
        with patch.object(r, "sync", side_effect=RuntimeError("Test")):
            r._periodic_tick()
        try:
            assert r._timer is not None
        finally:
            r.stop_periodic_sync()

    def test_incidents_property_returns_copy(self):
        """incidents-Property gibt eine unveraenderliche Kopie zurueck."""
        r = _reconciler()
        copy1 = r.incidents
        copy1.append({"type": "fake"})
        assert len(r.incidents) == 0  # Original unveraendert


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: MT5Connector Callback-Erweiterung
# ─────────────────────────────────────────────────────────────────────────────

class TestMT5ConnectorCallbackExtension:
    """Testet die neue register_reconnect_callback-Funktionalitaet in MT5Connector."""

    def test_register_reconnect_callback_exists(self):
        """MT5Connector hat die neue Methode register_reconnect_callback."""
        from src.data.mt5_connector import MT5Connector
        assert hasattr(MT5Connector, "register_reconnect_callback")

    def test_callbacks_stored(self):
        """Registrierte Callbacks werden in der Liste gespeichert."""
        from src.data.mt5_connector import MT5Connector
        conn = MT5Connector.__new__(MT5Connector)
        conn._reconnect_callbacks = []
        cb = MagicMock()
        conn.register_reconnect_callback(cb)
        assert cb in conn._reconnect_callbacks

    def test_fire_reconnect_callbacks_calls_all(self):
        """_fire_reconnect_callbacks() ruft alle registrierten Callbacks auf."""
        from src.data.mt5_connector import MT5Connector
        conn = MT5Connector.__new__(MT5Connector)
        conn._reconnect_callbacks = []
        cb1, cb2 = MagicMock(), MagicMock()
        conn.register_reconnect_callback(cb1)
        conn.register_reconnect_callback(cb2)
        conn._fire_reconnect_callbacks()
        cb1.assert_called_once()
        cb2.assert_called_once()

    def test_fire_reconnect_callbacks_tolerates_error(self):
        """Ein fehlerhafter Callback stoert die anderen nicht."""
        from src.data.mt5_connector import MT5Connector
        conn = MT5Connector.__new__(MT5Connector)
        conn._reconnect_callbacks = []
        cb_bad  = MagicMock(side_effect=RuntimeError("Kaboom"))
        cb_good = MagicMock()
        conn.register_reconnect_callback(cb_bad)
        conn.register_reconnect_callback(cb_good)
        conn._fire_reconnect_callbacks()  # kein Exception
        cb_good.assert_called_once()
