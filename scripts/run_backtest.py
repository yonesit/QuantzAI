"""
scripts/run_backtest.py
QuantzAI Backtest-CLI – fuehrt einen vectorbt-Backtest fuer ein Symbol aus.

Verwendung:
  python scripts/run_backtest.py --symbol EURUSD --tf H1 --start 2020-01-01 --end 2024-12-31
  python scripts/run_backtest.py --symbol GBPUSD --tf H4 --start 2022-01-01 --end 2024-12-31 \\
      --model models/signal_model_v1.joblib --is-end 2023-12-31

Feature-Dateien werden aus data/features/{symbol}_{tf}_*.parquet geladen.
Ohne --model wird ein Mock-Modell verwendet (gibt 'flat' zurueck – zum Testen).
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from loguru import logger


# ─────────────────────────────────────────────────────────────────────────────
#  Feature-Laden
# ─────────────────────────────────────────────────────────────────────────────

def _load_features(symbol: str, tf: str, start: str, end: str, features_dir: str):
    """
    Laedt das aktuellste Feature-Parquet fuer das Symbol und filtert auf [start, end].

    Returns
    -------
    pd.DataFrame mit DatetimeIndex oder wirft FileNotFoundError.
    """
    import pandas as pd

    pattern = str(Path(features_dir) / f"{symbol}_{tf}_*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"Keine Feature-Datei gefunden fuer {symbol}/{tf} in '{features_dir}'.\n"
            f"Pattern: {pattern}\n"
            f"Bitte zuerst `python scripts/fetch_data.py` ausfuehren."
        )

    df = pd.read_parquet(files[-1])
    logger.info("Feature-Datei geladen: {f}", f=files[-1])

    # Index normalisieren
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    df.index = pd.to_datetime(df.index, utc=True)

    # Zeitbereich filtern
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts   = pd.Timestamp(end,   tz="UTC")
    df = df[(df.index >= start_ts) & (df.index <= end_ts)]

    if df.empty:
        raise ValueError(
            f"Keine Daten im Zeitraum {start} – {end} fuer {symbol}/{tf}."
        )

    logger.info(
        "Zeitraum: {s} – {e} | {n} Kerzen",
        s=df.index[0].date(), e=df.index[-1].date(), n=len(df),
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  Modell / Signal-Funktion
# ─────────────────────────────────────────────────────────────────────────────

def _build_signal_func(model_path: str | None, confidence: float):
    """
    Gibt eine Signal-Funktion (row_df) -> str zurueck.

    Ohne Modellpfad: Mock-Funktion die immer 'flat' zurueckgibt.
    Mit Modellpfad: laedt SignalModel und bindet confidence_threshold.
    """
    if model_path is None:
        logger.warning(
            "Kein Modell angegeben – Signal-Funktion gibt immer 'flat' zurueck."
        )

        def _mock_signal(row_df):
            return "flat"

        return _mock_signal

    from src.models.signal_model import SignalModel
    model = SignalModel.load(model_path)
    logger.info("Modell geladen: {p}", p=model_path)

    def _model_signal(row_df):
        return model.get_signal(row_df, confidence_threshold=confidence)

    return _model_signal


# ─────────────────────────────────────────────────────────────────────────────
#  Ergebnis-Ausgabe
# ─────────────────────────────────────────────────────────────────────────────

def _print_results(result, symbol: str, tf: str) -> None:
    """Gibt Backtest-Ergebnisse als formatierte Tabelle aus."""
    from src.backtesting.vectorbt_runner import BacktestResult

    sep = "─" * 52
    print(f"\n{'═' * 52}")
    print(f"  QuantzAI Backtest  |  {symbol} / {tf}")
    print(f"{'═' * 52}")
    print(f"  Gesamtertrag          : {result.total_return:>10.2%}")
    print(f"  Sharpe Ratio          : {result.sharpe_ratio:>10.3f}")
    print(f"  Sortino Ratio         : {result.sortino_ratio:>10.3f}")
    print(f"  Max. Drawdown         : {result.max_drawdown:>10.2%}")
    print(sep)
    print(f"  Trades gesamt         : {result.n_trades:>10}")
    print(f"  Win-Rate              : {result.win_rate:>10.1%}")
    print(f"  Gewinnfaktor          : {result.profit_factor:>10.3f}")
    print(f"  Avg. Gewinn / Trade   : {result.avg_win:>10.2f}")
    print(f"  Avg. Verlust / Trade  : {result.avg_loss:>10.2f}")

    if result.is_sharpe is not None or result.oos_sharpe is not None:
        print(sep)
        is_str  = f"{result.is_sharpe:.3f}"  if result.is_sharpe  is not None else "n/a"
        oos_str = f"{result.oos_sharpe:.3f}" if result.oos_sharpe is not None else "n/a"
        print(f"  IS-Sharpe             : {is_str:>10}")
        print(f"  OOS-Sharpe            : {oos_str:>10}")
        if result.overfitting_warning:
            print(f"  ⚠  OVERFITTING-WARNUNG: IS >> OOS")

    print(f"{'═' * 52}\n")


# ─────────────────────────────────────────────────────────────────────────────
#  Equity-Curve-Export
# ─────────────────────────────────────────────────────────────────────────────

def _export_equity(result, output_path: str) -> None:
    """Exportiert die Equity-Curve als CSV."""
    result.equity_curve.to_csv(output_path, header=["equity"])
    logger.info("Equity-Curve exportiert -> {p}", p=output_path)


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QuantzAI Backtest via vectorbt",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--symbol",  required=True,         help="Handelssymbol, z.B. EURUSD")
    parser.add_argument("--tf",      default="H1",          help="Zeitrahmen: M1/M5/H1/H4/D1")
    parser.add_argument("--start",   required=True,         help="Startdatum ISO, z.B. 2020-01-01")
    parser.add_argument("--end",     required=True,         help="Enddatum ISO, z.B. 2024-12-31")
    parser.add_argument("--model",   default=None,          help="Pfad zur SignalModel .joblib-Datei")
    parser.add_argument("--is-end",  default=None,          dest="is_end",
                        help="IS/OOS-Trenndatum ISO (alles <= gilt als In-Sample)")
    parser.add_argument("--confidence", type=float, default=0.55,
                        help="KI-Abstinenz-Schwellwert fuer SignalModel")
    parser.add_argument("--init-cash",  type=float, default=10_000.0, dest="init_cash",
                        help="Startkapital")
    parser.add_argument("--spread-pct", type=float, default=0.0001,   dest="spread_pct",
                        help="Spread als Preisanteil (0.0001 = 1 Pip @ EUR/USD)")
    parser.add_argument("--slippage-pips", type=float, default=1.0,   dest="slippage_pips",
                        help="Slippage in Pips")
    parser.add_argument("--swap-long",  type=float, default=0.0,      dest="swap_long",
                        help="Swap-Kosten Long pro Nacht (Kontowaehrung)")
    parser.add_argument("--swap-short", type=float, default=0.0,      dest="swap_short",
                        help="Swap-Kosten Short pro Nacht (Kontowaehrung)")
    parser.add_argument("--features-dir", default="data/features",    dest="features_dir",
                        help="Verzeichnis der Feature-Parquet-Dateien")
    parser.add_argument("--export-equity", default=None,              dest="export_equity",
                        help="Pfad fuer Equity-Curve CSV-Export (optional)")
    parser.add_argument("--overfitting-threshold", type=float, default=0.5,
                        dest="overfitting_threshold",
                        help="IS-OOS-Sharpe-Differenz fuer Overfitting-Warnung")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)

    from src.backtesting.vectorbt_runner import (
        BacktestConfig,
        BacktestRunner,
        timeframe_to_freq,
    )

    freq = timeframe_to_freq(args.tf)
    config = BacktestConfig(
        init_cash=args.init_cash,
        spread_pct=args.spread_pct,
        slippage_pips=args.slippage_pips,
        swap_long_per_night=args.swap_long,
        swap_short_per_night=args.swap_short,
        freq=freq,
        overfitting_sharpe_threshold=args.overfitting_threshold,
    )

    logger.info(
        "Starte Backtest | Symbol={sym} | TF={tf} | {s} – {e}",
        sym=args.symbol, tf=args.tf, s=args.start, e=args.end,
    )

    try:
        features_df = _load_features(
            args.symbol, args.tf, args.start, args.end, args.features_dir
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Datenfehler: {exc}", exc=exc)
        return 1

    signal_func = _build_signal_func(args.model, args.confidence)

    runner = BacktestRunner(config)
    try:
        result = runner.run_with_model(
            features_df=features_df,
            signal_func=signal_func,
            close_col="close",
            is_end=args.is_end,
        )
    except Exception as exc:
        logger.error("Backtest fehlgeschlagen: {exc}", exc=exc)
        return 1

    _print_results(result, args.symbol, args.tf)

    if args.export_equity:
        _export_equity(result, args.export_equity)

    return 0


if __name__ == "__main__":
    sys.exit(main())
