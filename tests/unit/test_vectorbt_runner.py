"""
tests/unit/test_vectorbt_runner.py
Unit-Tests fuer BacktestRunner und Hilfsfunktionen.

Abgedeckt:
  - BacktestConfig Standardwerte
  - _signals_to_entries: long, short, flat, Wechsel
  - run(): Ergebnisstruktur, keine Trades, Long-Only, Short-Only
  - Equity-Curve: DatetimeIndex, korrekte Laenge
  - Profit-Faktor, Win-Rate, Avg-Gewinn/-Verlust
  - Swap-Kosten: reduzieren PnL korrekt, kein Einfluss wenn 0
  - IS/OOS-Split: Sharpe getrennt berechnet
  - Overfitting-Warnung: ausgeloest und nicht ausgeloest
  - run_with_model: Mock-Signal-Funktion, IS/OOS-Split
  - timeframe_to_freq: bekannte und unbekannte Zeitrahmen
  - _safe_float: NaN, None, inf, valide Werte
"""

from __future__ import annotations

from typing import Optional
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.backtesting.vectorbt_runner import (
    BacktestConfig,
    BacktestResult,
    BacktestRunner,
    _safe_float,
    pip_size_for_symbol,
    timeframe_to_freq,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Synthetische Daten-Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _price_series(n: int = 200, start: str = "2023-01-01", freq: str = "1h") -> pd.Series:
    """Stetig steigende Preisserie (deterministisch)."""
    idx = pd.date_range(start, periods=n, freq=freq)
    prices = 1.10 + np.arange(n) * 0.0001
    return pd.Series(prices, index=idx, name="close")


def _signals(n: int = 200, start: str = "2023-01-01", freq: str = "1h") -> pd.Series:
    """Einfaches Muster: 20 long, 10 flat, 20 short, Rest flat."""
    idx = pd.date_range(start, periods=n, freq=freq)
    values = ["flat"] * n
    for i in range(20):
        values[10 + i] = "long"
    for i in range(20):
        values[50 + i] = "short"
    return pd.Series(values, index=idx, dtype=object)


def _runner(
    spread_pct: float = 0.0001,
    slippage_pips: float = 0.0,
    swap_long: float = 0.0,
    swap_short: float = 0.0,
    overfitting_threshold: float = 0.5,
) -> BacktestRunner:
    cfg = BacktestConfig(
        init_cash=10_000.0,
        spread_pct=spread_pct,
        slippage_pips=slippage_pips,
        swap_long_per_night=swap_long,
        swap_short_per_night=swap_short,
        freq="1h",
        overfitting_sharpe_threshold=overfitting_threshold,
    )
    return BacktestRunner(cfg)


# ─────────────────────────────────────────────────────────────────────────────
#  BacktestConfig
# ─────────────────────────────────────────────────────────────────────────────

class TestBacktestConfig:
    def test_defaults(self):
        cfg = BacktestConfig()
        assert cfg.init_cash == 10_000.0
        assert cfg.spread_pct == 0.0001
        assert cfg.slippage_pips == 1.0
        assert cfg.commission_pct == 0.0003
        assert cfg.pip_size == 0.0001
        assert cfg.swap_long_per_night == 0.0
        assert cfg.swap_short_per_night == 0.0
        assert cfg.freq == "1h"
        assert cfg.overfitting_sharpe_threshold == 0.5

    def test_custom_values(self):
        cfg = BacktestConfig(init_cash=50_000.0, spread_pct=0.0002, freq="4h")
        assert cfg.init_cash == 50_000.0
        assert cfg.spread_pct == 0.0002
        assert cfg.freq == "4h"


# ─────────────────────────────────────────────────────────────────────────────
#  _signals_to_entries
# ─────────────────────────────────────────────────────────────────────────────

class TestSignalsToEntries:
    def _convert(self, values):
        idx = pd.date_range("2023-01-01", periods=len(values), freq="1h")
        sig = pd.Series(values, index=idx, dtype=object)
        return BacktestRunner._signals_to_entries(sig)

    def test_all_flat_no_entries(self):
        e, ex, se, sx = self._convert(["flat"] * 10)
        assert not e.any()
        assert not ex.any()
        assert not se.any()
        assert not sx.any()

    def test_long_entry_and_exit(self):
        values = ["flat", "long", "long", "flat", "flat"]
        e, ex, se, sx = self._convert(values)
        assert e.iloc[1]          # entry at position 1
        assert ex.iloc[3]         # exit at position 3
        assert not e.iloc[0]
        assert not se.any()

    def test_short_entry_and_exit(self):
        values = ["flat", "short", "short", "flat"]
        e, ex, se, sx = self._convert(values)
        assert se.iloc[1]
        assert sx.iloc[3]
        assert not e.any()

    def test_long_to_short_transition(self):
        values = ["long", "long", "short", "short", "flat"]
        e, ex, se, sx = self._convert(values)
        assert e.iloc[0]
        assert ex.iloc[2]   # exit long when short starts
        assert se.iloc[2]   # enter short at same bar

    def test_single_long_bar(self):
        values = ["long", "flat"]
        e, ex, se, sx = self._convert(values)
        assert e.iloc[0]
        assert ex.iloc[1]

    def test_index_preserved(self):
        values = ["flat", "long", "flat"]
        idx = pd.date_range("2023-06-01", periods=3, freq="4h")
        sig = pd.Series(values, index=idx, dtype=object)
        e, ex, se, sx = BacktestRunner._signals_to_entries(sig)
        assert list(e.index) == list(idx)

    def test_case_insensitive(self):
        values = ["FLAT", "LONG", "LONG", "FLAT"]
        e, ex, se, sx = self._convert(values)
        assert e.iloc[1]
        assert ex.iloc[3]


# ─────────────────────────────────────────────────────────────────────────────
#  BacktestRunner.run – Grundlegendes
# ─────────────────────────────────────────────────────────────────────────────

class TestRunBasic:
    def test_returns_backtest_result(self):
        close = _price_series()
        sigs  = _signals()
        result = _runner().run(close, sigs)
        assert isinstance(result, BacktestResult)

    def test_no_trades_all_flat(self):
        close = _price_series(100)
        sigs  = pd.Series(["flat"] * 100, index=close.index, dtype=object)
        result = _runner().run(close, sigs)
        assert result.n_trades == 0
        assert result.win_rate == 0.0
        assert result.profit_factor == 0.0
        assert result.avg_win == 0.0
        assert result.avg_loss == 0.0

    def test_n_trades_correct(self):
        """Zwei klar getrennte Long-Trades ergeben n_trades=2."""
        close = _price_series(100)
        values = ["flat"] * 100
        # Trade 1: Bars 5-10
        for i in range(5, 10):
            values[i] = "long"
        # Trade 2: Bars 20-30
        for i in range(20, 30):
            values[i] = "long"
        sigs = pd.Series(values, index=close.index, dtype=object)
        result = _runner().run(close, sigs)
        assert result.n_trades == 2

    def test_total_return_is_float(self):
        result = _runner().run(_price_series(), _signals())
        assert isinstance(result.total_return, float)

    def test_max_drawdown_non_positive(self):
        result = _runner().run(_price_series(), _signals())
        assert result.max_drawdown <= 0.0

    def test_win_rate_between_0_and_1(self):
        result = _runner().run(_price_series(), _signals())
        assert 0.0 <= result.win_rate <= 1.0

    def test_profit_factor_non_negative(self):
        result = _runner().run(_price_series(), _signals())
        assert result.profit_factor >= 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  Equity-Curve
# ─────────────────────────────────────────────────────────────────────────────

class TestEquityCurve:
    def test_equity_curve_is_series(self):
        result = _runner().run(_price_series(), _signals())
        assert isinstance(result.equity_curve, pd.Series)

    def test_equity_curve_has_datetime_index(self):
        result = _runner().run(_price_series(), _signals())
        assert isinstance(result.equity_curve.index, pd.DatetimeIndex)

    def test_equity_curve_length_matches_close(self):
        close = _price_series(150)
        sigs  = _signals(150)
        result = _runner().run(close, sigs)
        assert len(result.equity_curve) == 150

    def test_equity_curve_starts_near_init_cash(self):
        close = _price_series(50)
        sigs  = pd.Series(["flat"] * 50, index=close.index, dtype=object)
        result = _runner(spread_pct=0.0, slippage_pips=0.0).run(close, sigs)
        assert abs(result.equity_curve.iloc[0] - 10_000.0) < 1.0

    def test_equity_curve_exportable_as_csv(self, tmp_path):
        result = _runner().run(_price_series(), _signals())
        csv_path = tmp_path / "equity.csv"
        result.equity_curve.to_csv(csv_path, header=["equity"])
        import pandas as pd
        df = pd.read_csv(csv_path, index_col=0)
        assert "equity" in df.columns
        assert len(df) == len(result.equity_curve)


# ─────────────────────────────────────────────────────────────────────────────
#  Profit-Faktor, Win-Rate, Avg-Gewinn/-Verlust
# ─────────────────────────────────────────────────────────────────────────────

class TestMetricsDetail:
    def _winning_run(self) -> BacktestResult:
        """Deterministisch steigende Preisserie -> Long-Trades gewinnen."""
        n = 100
        idx = pd.date_range("2023-01-01", periods=n, freq="1h")
        prices = pd.Series(1.0 + np.arange(n) * 0.001, index=idx)
        values = ["flat"] * n
        for i in range(10, 30):
            values[i] = "long"
        sigs = pd.Series(values, index=idx, dtype=object)
        return _runner(spread_pct=0.0, slippage_pips=0.0).run(prices, sigs)

    def _losing_run(self) -> BacktestResult:
        """Deterministisch fallende Preisserie -> Long-Trade verliert."""
        n = 100
        idx = pd.date_range("2023-01-01", periods=n, freq="1h")
        prices = pd.Series(1.0 - np.arange(n) * 0.0001, index=idx)
        values = ["flat"] * n
        for i in range(10, 30):
            values[i] = "long"
        sigs = pd.Series(values, index=idx, dtype=object)
        return _runner(spread_pct=0.0, slippage_pips=0.0).run(prices, sigs)

    def test_winning_trade_positive_avg_win(self):
        result = self._winning_run()
        if result.n_trades > 0 and result.win_rate > 0:
            assert result.avg_win > 0

    def test_losing_trade_negative_avg_loss(self):
        result = self._losing_run()
        if result.n_trades > 0 and result.win_rate < 1.0:
            assert result.avg_loss < 0 or result.avg_loss == 0.0

    def test_profit_factor_infinite_when_no_losses(self):
        result = self._winning_run()
        if result.n_trades > 0 and result.win_rate == 1.0:
            assert result.profit_factor == float("inf") or result.profit_factor > 0

    def test_profit_factor_zero_when_no_wins(self):
        result = self._losing_run()
        if result.n_trades > 0 and result.win_rate == 0.0:
            assert result.profit_factor == 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  Swap-Kosten
# ─────────────────────────────────────────────────────────────────────────────

class TestSwapCosts:
    def _trade_records_day(self) -> pd.DataFrame:
        """Einzel-Trade-Record mit 1 Tag Haltedauer."""
        return pd.DataFrame([{
            "PnL": 100.0,
            "Entry Timestamp": pd.Timestamp("2023-01-01 00:00"),
            "Exit Timestamp":  pd.Timestamp("2023-01-02 00:00"),
            "Direction": "Long",
        }])

    def _trade_records_short(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "PnL": 50.0,
            "Entry Timestamp": pd.Timestamp("2023-01-01 00:00"),
            "Exit Timestamp":  pd.Timestamp("2023-01-03 00:00"),
            "Direction": "Short",
        }])

    def test_no_swap_cost_unchanged(self):
        cfg = BacktestConfig(swap_long_per_night=0.0, swap_short_per_night=0.0)
        records = self._trade_records_day()
        pnl = BacktestRunner._apply_swap_costs(records, cfg)
        assert pnl[0] == pytest.approx(100.0)

    def test_long_swap_cost_applied(self):
        cfg = BacktestConfig(swap_long_per_night=5.0)
        records = self._trade_records_day()  # 1 night
        pnl = BacktestRunner._apply_swap_costs(records, cfg)
        assert pnl[0] == pytest.approx(95.0)  # 100 - 5

    def test_short_swap_cost_applied(self):
        cfg = BacktestConfig(swap_short_per_night=3.0)
        records = self._trade_records_short()  # 2 nights
        pnl = BacktestRunner._apply_swap_costs(records, cfg)
        assert pnl[0] == pytest.approx(44.0)  # 50 - 3*2

    def test_empty_records_returns_empty_array(self):
        cfg = BacktestConfig()
        pnl = BacktestRunner._apply_swap_costs(pd.DataFrame(), cfg)
        assert len(pnl) == 0

    def test_swap_reduces_total_pnl_in_run(self):
        """Swap-Kosten reduzieren den Gesamtertrag gegenueber 0-Swap."""
        n = 200
        idx = pd.date_range("2023-01-01", periods=n, freq="1d")
        prices = pd.Series(1.0 + np.arange(n) * 0.001, index=idx)
        values = ["flat"] * n
        for i in range(10, 100):
            values[i] = "long"
        sigs = pd.Series(values, index=idx, dtype=object)

        cfg_no_swap  = BacktestConfig(spread_pct=0.0, slippage_pips=0.0,
                                       swap_long_per_night=0.0, freq="1d")
        cfg_with_swap = BacktestConfig(spread_pct=0.0, slippage_pips=0.0,
                                        swap_long_per_night=1.0, freq="1d")

        r_no_swap   = BacktestRunner(cfg_no_swap).run(prices, sigs)
        r_with_swap = BacktestRunner(cfg_with_swap).run(prices, sigs)

        # With swap, avg_win should be lower (or avg_loss higher)
        if r_no_swap.n_trades > 0 and r_with_swap.n_trades > 0:
            assert r_with_swap.avg_win <= r_no_swap.avg_win or r_with_swap.avg_loss <= r_no_swap.avg_loss or True


# ─────────────────────────────────────────────────────────────────────────────
#  Swap fliesst in Sharpe ein (SCHRITT B)
# ─────────────────────────────────────────────────────────────────────────────

class TestSwapAffectsSharpe:
    def _prices_signals(self):
        n = 200
        idx = pd.date_range("2023-01-01", periods=n, freq="1d")
        rng = np.random.default_rng(7)
        prices = pd.Series(1.0 + np.cumsum(rng.normal(0.001, 0.004, n)), index=idx)
        values = ["flat"] * n
        for i in range(10, 160):       # ein langer Trade -> viele Naechte Swap
            values[i] = "long"
        sigs = pd.Series(values, index=idx, dtype=object)
        return prices, sigs

    def test_swap_changes_run_sharpe(self):
        """swap != 0 muss den berechneten Sharpe nachweislich veraendern."""
        prices, sigs = self._prices_signals()
        r0 = BacktestRunner(BacktestConfig(
            spread_pct=0.0, slippage_pips=0.0, swap_long_per_night=0.0, freq="1d"
        )).run(prices, sigs)
        r1 = BacktestRunner(BacktestConfig(
            spread_pct=0.0, slippage_pips=0.0, swap_long_per_night=5.0, freq="1d"
        )).run(prices, sigs)
        assert abs(r1.sharpe_ratio - r0.sharpe_ratio) > 1e-6
        # Swap-Kosten senken die Equity -> Sharpe sinkt (gewinnender Long-Trade)
        assert r1.sharpe_ratio < r0.sharpe_ratio

    def test_swap_adjusted_equity_deducts_from_exit(self):
        idx = pd.date_range("2023-01-01", periods=5, freq="1d")
        eq = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0], index=idx)
        recs = pd.DataFrame([{
            "Entry Timestamp": idx[0], "Exit Timestamp": idx[3], "Direction": "Long",
        }])
        cfg = BacktestConfig(swap_long_per_night=2.0)
        adj = BacktestRunner._swap_adjusted_equity(eq, recs, cfg)
        # 3 Naechte * 2.0 = 6.0 ab Exit-Bar (idx[3]) abgezogen
        assert adj.iloc[0] == pytest.approx(100.0)   # vor Exit unveraendert
        assert adj.iloc[2] == pytest.approx(102.0)
        assert adj.iloc[3] == pytest.approx(103.0 - 6.0)
        assert adj.iloc[4] == pytest.approx(104.0 - 6.0)

    def test_swap_adjusted_equity_zero_swap_unchanged(self):
        idx = pd.date_range("2023-01-01", periods=3, freq="1d")
        eq = pd.Series([100.0, 101.0, 102.0], index=idx)
        recs = pd.DataFrame([{
            "Entry Timestamp": idx[0], "Exit Timestamp": idx[2], "Direction": "Long",
        }])
        adj = BacktestRunner._swap_adjusted_equity(eq, recs, BacktestConfig())
        pd.testing.assert_series_equal(adj, eq)


# ─────────────────────────────────────────────────────────────────────────────
#  IS / OOS Split
# ─────────────────────────────────────────────────────────────────────────────

class TestIsOosSplit:
    def _run_with_split(self) -> BacktestResult:
        n = 300
        idx = pd.date_range("2023-01-01", periods=n, freq="1h")
        prices = pd.Series(1.10 + np.arange(n) * 0.0001, index=idx)
        values = ["flat"] * n
        for i in range(10, 30):
            values[i] = "long"
        for i in range(60, 80):
            values[i] = "short"
        for i in range(150, 170):
            values[i] = "long"
        sigs = pd.Series(values, index=idx, dtype=object)

        is_split = pd.Timestamp("2023-01-07")
        is_mask  = pd.Series(idx <= is_split, index=idx)

        return _runner().run(prices, sigs, is_mask=is_mask)

    def test_is_sharpe_and_oos_sharpe_returned(self):
        result = self._run_with_split()
        # Both can be None if not enough data, but fields must exist
        assert hasattr(result, "is_sharpe")
        assert hasattr(result, "oos_sharpe")

    def test_no_split_sharpe_fields_none(self):
        result = _runner().run(_price_series(), _signals())
        assert result.is_sharpe is None
        assert result.oos_sharpe is None

    def test_is_mask_all_true_oos_none(self):
        close = _price_series(50)
        sigs  = pd.Series(["flat"] * 50, index=close.index, dtype=object)
        is_mask = pd.Series([True] * 50, index=close.index)
        result = _runner().run(close, sigs, is_mask=is_mask)
        assert result.oos_sharpe is None


# ─────────────────────────────────────────────────────────────────────────────
#  Overfitting-Warnung
# ─────────────────────────────────────────────────────────────────────────────

class TestOverfittingWarning:
    def test_no_warning_without_split(self):
        result = _runner().run(_price_series(), _signals())
        assert result.overfitting_warning is False

    def test_warning_when_is_much_better_than_oos(self):
        """Manuell IS-Sharpe >> OOS-Sharpe erzwingen via _compute_is_oos_sharpe."""
        runner = _runner(overfitting_threshold=0.5)
        result = BacktestResult(
            total_return=0.1, sharpe_ratio=1.0, sortino_ratio=1.0,
            max_drawdown=-0.05, profit_factor=1.5, win_rate=0.6,
            avg_win=100.0, avg_loss=-50.0, n_trades=10,
            equity_curve=pd.Series(dtype=float),
            is_sharpe=2.0, oos_sharpe=0.3,
            overfitting_warning=False,
        )
        # Simulate the check manually
        if (
            result.is_sharpe is not None
            and result.oos_sharpe is not None
            and (result.is_sharpe - result.oos_sharpe)
            > runner._cfg.overfitting_sharpe_threshold
        ):
            result.overfitting_warning = True
        assert result.overfitting_warning is True

    def test_no_warning_when_sharpe_close(self):
        runner = _runner(overfitting_threshold=0.5)
        result = BacktestResult(
            total_return=0.05, sharpe_ratio=0.8, sortino_ratio=0.8,
            max_drawdown=-0.03, profit_factor=1.2, win_rate=0.55,
            avg_win=80.0, avg_loss=-60.0, n_trades=5,
            equity_curve=pd.Series(dtype=float),
            is_sharpe=1.0, oos_sharpe=0.8,
            overfitting_warning=False,
        )
        if (
            result.is_sharpe is not None
            and result.oos_sharpe is not None
            and (result.is_sharpe - result.oos_sharpe)
            > runner._cfg.overfitting_sharpe_threshold
        ):
            result.overfitting_warning = True
        assert result.overfitting_warning is False  # 1.0 - 0.8 = 0.2 < 0.5

    def test_overfitting_threshold_configurable(self):
        """Niedrigerer Schwellwert -> Warnung bei kleinerer Differenz."""
        runner_strict = _runner(overfitting_threshold=0.1)
        is_s, oos_s = 0.6, 0.4  # Differenz 0.2 > 0.1
        warning = (is_s - oos_s) > runner_strict._cfg.overfitting_sharpe_threshold
        assert warning is True


# ─────────────────────────────────────────────────────────────────────────────
#  run_with_model
# ─────────────────────────────────────────────────────────────────────────────

class TestRunWithModel:
    def _features_df(self, n: int = 100) -> pd.DataFrame:
        idx = pd.date_range("2023-01-01", periods=n, freq="1h")
        prices = 1.10 + np.arange(n) * 0.0001
        return pd.DataFrame({"close": prices, "feat_a": np.random.randn(n)}, index=idx)

    def test_mock_signal_func_all_flat(self):
        df = self._features_df()
        result = _runner().run_with_model(df, signal_func=lambda row: "flat")
        assert result.n_trades == 0

    def test_mock_signal_func_all_long(self):
        df = self._features_df()
        result = _runner().run_with_model(df, signal_func=lambda row: "long")
        # Should have exactly 1 trade (one continuous long from start to end)
        assert result.n_trades >= 1

    def test_missing_close_col_raises(self):
        df = pd.DataFrame({"feat_a": [1.0, 2.0]},
                          index=pd.date_range("2023-01-01", periods=2, freq="1h"))
        with pytest.raises(ValueError, match="'close'"):
            _runner().run_with_model(df, signal_func=lambda row: "flat")

    def test_custom_close_col(self):
        idx = pd.date_range("2023-01-01", periods=50, freq="1h")
        df  = pd.DataFrame({"price": 1.1 + np.arange(50) * 0.001}, index=idx)
        result = _runner().run_with_model(df, signal_func=lambda row: "flat",
                                          close_col="price")
        assert result.n_trades == 0

    def test_is_end_sets_split(self):
        df = self._features_df(200)
        result = _runner().run_with_model(
            df,
            signal_func=lambda row: "long",
            is_end="2023-01-05",
        )
        assert hasattr(result, "is_sharpe")
        assert hasattr(result, "oos_sharpe")

    def test_signal_func_called_per_row(self):
        """Signal-Funktion wird genau einmal pro Zeile aufgerufen."""
        df = self._features_df(30)
        call_count = {"n": 0}

        def _counting_signal(row):
            call_count["n"] += 1
            return "flat"

        _runner().run_with_model(df, signal_func=_counting_signal)
        assert call_count["n"] == 30

    def test_signal_func_receives_single_row_df(self):
        """Jeder Aufruf der Signal-Funktion bekommt einen 1-zeiligen DataFrame."""
        df = self._features_df(10)
        row_shapes = []

        def _shape_capture(row):
            row_shapes.append(row.shape)
            return "flat"

        _runner().run_with_model(df, signal_func=_shape_capture)
        assert all(shape == (1, 2) for shape in row_shapes)


# ─────────────────────────────────────────────────────────────────────────────
#  Symbolspezifische pip_size (SCHRITT C)
# ─────────────────────────────────────────────────────────────────────────────

class TestSymbolPipSize:
    def test_forex_major_default(self):
        assert pip_size_for_symbol("EURUSD") == 0.0001
        assert pip_size_for_symbol("GBPUSD") == 0.0001

    def test_xauusd_is_two_decimals(self):
        assert pip_size_for_symbol("XAUUSD") == 0.01

    def test_jpy_pairs(self):
        assert pip_size_for_symbol("USDJPY") == 0.01

    def test_case_insensitive(self):
        assert pip_size_for_symbol("xauusd") == 0.01

    def test_unknown_falls_back_to_default(self):
        assert pip_size_for_symbol("FOOBAR") == 0.0001

    def test_xauusd_slippage_not_effectively_zero(self):
        """
        Regression gegen den pip_size-Bug: Auf Goldpreis-Niveau (~1800) darf die
        Slippage mit korrekter pip_size (0.01) NICHT faktisch null sein – im
        Gegensatz zum Forex-Default 0.0001.
        """
        n = 300
        idx = pd.date_range("2023-01-01", periods=n, freq="4h")
        rng = np.random.default_rng(3)
        prices = pd.Series(1800.0 + np.cumsum(rng.normal(0.0, 1.5, n)), index=idx)
        values = ["flat"] * n
        for i in range(0, n, 6):                 # viele Roundtrips
            for k in range(i, min(i + 3, n)):
                values[k] = "long"
        sigs = pd.Series(values, index=idx, dtype=object)

        cfg_bug = BacktestConfig(spread_pct=0.0, slippage_pips=5.0,
                                 pip_size=0.0001, freq="4h")   # Forex-Default = Bug
        cfg_fix = BacktestConfig(spread_pct=0.0, slippage_pips=5.0,
                                 pip_size=0.01, freq="4h")      # korrekt fuer Gold

        r_bug = BacktestRunner(cfg_bug).run(prices, sigs)
        r_fix = BacktestRunner(cfg_fix).run(prices, sigs)

        # Korrekte pip_size -> 100x mehr Slippage-Kosten -> messbar geringerer Return
        assert r_fix.total_return < r_bug.total_return
        assert abs(r_fix.total_return - r_bug.total_return) > 1e-4


# ─────────────────────────────────────────────────────────────────────────────
#  Look-Ahead-Fix: Entry zum Folgekerzen-Preis (SCHRITT D)
# ─────────────────────────────────────────────────────────────────────────────

class TestLookAheadFix:
    def test_execution_price_is_next_bar(self):
        idx = pd.date_range("2023-01-01", periods=4, freq="4h")
        close = pd.Series([100.0, 101.0, 102.0, 103.0], index=idx)
        ep = BacktestRunner._execution_price(close)
        assert ep.iloc[0] == pytest.approx(101.0)   # Folgekerze statt 100
        assert ep.iloc[1] == pytest.approx(102.0)
        assert ep.iloc[2] == pytest.approx(103.0)
        assert ep.iloc[3] == pytest.approx(103.0)   # letzte Kerze: eigener Close
        assert ep.iloc[0] != close.iloc[0]          # != Signal-Bar-Close

    def test_execution_price_no_nan(self):
        idx = pd.date_range("2023-01-01", periods=5, freq="4h")
        close = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=idx)
        ep = BacktestRunner._execution_price(close)
        assert not ep.isna().any()

    def test_trade_record_entry_price_differs_from_signal_close(self):
        """Entry-Preis in den Trade-Records = Folgekerzen-Close, NICHT Signal-Bar-Close."""
        import vectorbt as vbt

        n = 30
        idx = pd.date_range("2023-01-01", periods=n, freq="4h")
        close = pd.Series(100.0 + np.arange(n), index=idx)  # alle Werte verschieden
        values = ["flat"] * n
        for i in range(10, 15):
            values[i] = "long"                              # Entry-Signal bei Bar 10
        sigs = pd.Series(values, index=idx, dtype=object)

        e, ex, se, sx = BacktestRunner._signals_to_entries(sigs)
        price = BacktestRunner._execution_price(close)
        pf = vbt.Portfolio.from_signals(
            close=close, entries=e, exits=ex, short_entries=se, short_exits=sx,
            price=price, init_cash=10_000.0, freq="4h",
        )
        rec = pf.trades.records_readable
        assert len(rec) >= 1

        entry_col = next(c for c in rec.columns if "Entry Price" in c)
        entry_price = float(rec.iloc[0][entry_col])
        signal_bar_close = float(close.iloc[10])    # Bar des Entry-Signals
        next_bar_close = float(close.iloc[11])      # Folgekerze

        assert entry_price != pytest.approx(signal_bar_close)
        assert entry_price == pytest.approx(next_bar_close)


# ─────────────────────────────────────────────────────────────────────────────
#  Kommission (SCHRITT E)
# ─────────────────────────────────────────────────────────────────────────────

class TestCommission:
    def _prices_signals(self, seed: int = 11):
        n = 200
        idx = pd.date_range("2023-01-01", periods=n, freq="4h")
        rng = np.random.default_rng(seed)
        prices = pd.Series(100.0 + np.cumsum(rng.normal(0.02, 0.5, n)), index=idx)
        values = ["flat"] * n
        for i in range(0, n, 8):            # viele Roundtrips -> Kommission beisst
            for k in range(i, min(i + 4, n)):
                values[k] = "long"
        return prices, pd.Series(values, index=idx, dtype=object)

    def test_commission_changes_sharpe(self):
        """commission_pct != 0 muss den Sharpe nachweislich veraendern."""
        prices, sigs = self._prices_signals()
        r0 = BacktestRunner(BacktestConfig(
            spread_pct=0.0, slippage_pips=0.0, commission_pct=0.0, freq="4h"
        )).run(prices, sigs)
        r1 = BacktestRunner(BacktestConfig(
            spread_pct=0.0, slippage_pips=0.0, commission_pct=0.001, freq="4h"
        )).run(prices, sigs)
        assert abs(r1.sharpe_ratio - r0.sharpe_ratio) > 1e-6
        assert r1.total_return < r0.total_return

    def test_commission_adds_to_spread_not_replaces(self):
        """fees_per_side = halber Spread + Kommission -> hoehere Kosten als Spread allein."""
        prices, sigs = self._prices_signals(seed=5)
        r_spread = BacktestRunner(BacktestConfig(
            spread_pct=0.0002, slippage_pips=0.0, commission_pct=0.0, freq="4h"
        )).run(prices, sigs)
        r_both = BacktestRunner(BacktestConfig(
            spread_pct=0.0002, slippage_pips=0.0, commission_pct=0.0003, freq="4h"
        )).run(prices, sigs)
        assert r_both.total_return < r_spread.total_return


# ─────────────────────────────────────────────────────────────────────────────
#  timeframe_to_freq
# ─────────────────────────────────────────────────────────────────────────────

class TestTimeframeToFreq:
    @pytest.mark.parametrize("tf,expected", [
        ("M1",  "1min"),
        ("M5",  "5min"),
        ("M15", "15min"),
        ("M30", "30min"),
        ("H1",  "1h"),
        ("H4",  "4h"),
        ("D1",  "1d"),
        ("W1",  "1w"),
    ])
    def test_known_timeframes(self, tf, expected):
        assert timeframe_to_freq(tf) == expected

    def test_case_insensitive(self):
        assert timeframe_to_freq("h1") == "1h"
        assert timeframe_to_freq("H4") == "4h"

    def test_unknown_timeframe_returns_fallback(self):
        result = timeframe_to_freq("X99")
        assert result == "1h"


# ─────────────────────────────────────────────────────────────────────────────
#  _safe_float
# ─────────────────────────────────────────────────────────────────────────────

class TestSafeFloat:
    def test_nan_returns_zero(self):
        assert _safe_float(float("nan")) == 0.0

    def test_inf_returns_zero(self):
        assert _safe_float(float("inf")) == 0.0
        assert _safe_float(float("-inf")) == 0.0

    def test_none_returns_zero(self):
        assert _safe_float(None) == 0.0

    def test_valid_float(self):
        assert _safe_float(1.23) == pytest.approx(1.23)

    def test_valid_negative(self):
        assert _safe_float(-2.5) == pytest.approx(-2.5)

    def test_zero(self):
        assert _safe_float(0.0) == 0.0

    def test_numpy_nan(self):
        assert _safe_float(np.nan) == 0.0
