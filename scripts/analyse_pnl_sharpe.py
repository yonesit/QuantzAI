"""
scripts/analyse_pnl_sharpe.py
Erzeugt erstmals einen ECHTEN P&L-basierten OOS-Sharpe via vectorbt – im
Gegensatz zum bisherigen Klassifikations-Proxy (SignalModel._compute_sharpe,
kuenstliche +1/-1-Renditen ohne Kosten).

Methode:
  Identische rollierende Walk-Forward-Fenster (6M Train / 1M Test) und
  identische argmax-Signale wie SignalModel.walk_forward_validate(), aber pro
  Fenster wird die echte P&L mit BacktestRunner (vectorbt) simuliert – inkl.
  der via BacktestConfig konfigurierten Kosten.

Portfolios (wie im Demo-Live-Betrieb):
  - XAUUSD H4 Trendfolge   (SignalModel-Features + Standard-Labels)
  - EURUSD H4 Mean-Reversion (MeanReversionModel-Features + MR-Labels)

Aufruf:
  python scripts/analyse_pnl_sharpe.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# Identisch zu SignalModel._params (Default-LGBM)
LGBM_PARAMS = {
    "objective": "multiclass",
    "num_class": 3,
    "num_leaves": 31,
    "learning_rate": 0.05,
    "n_estimators": 100,
    "random_state": 42,
    "verbose": -1,
}

_EXCLUDE = {"label", "timestamp", "open", "volume", "close", "high", "low"}


def _fetch_and_validate(symbol: str, timeframe: str):
    from src.data.mt5_connector import MT5Connector
    from src.data.validator import DataValidator

    mt5 = MT5Connector(
        login=int(os.environ.get("MT5_LOGIN", "0")),
        password=os.environ.get("MT5_PASSWORD", ""),
        server=os.environ.get("MT5_SERVER", ""),
    )
    mt5.connect()
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 1, tzinfo=timezone.utc)
    df = mt5.get_ohlcv(symbol, timeframe, start, end)
    mt5.disconnect()

    df_reset = df.reset_index()
    df_reset = df_reset.rename(columns={df_reset.columns[0]: "timestamp"})
    _, clean = DataValidator().validate(df_reset, symbol=symbol, timeframe=timeframe)
    return clean


def _run_xauusd_tf(cfg):
    from src.backtesting.wf_pnl import aggregate_pnl_sharpe, compute_wf_pnl_sharpe
    from src.data.feature_builder import FeatureBuilder
    from src.models.label_builder import LabelBuilder
    import lightgbm as lgb

    logger.info("=== XAUUSD H4 TF: WF-P&L ===")
    df = _fetch_and_validate("XAUUSD", "H4")
    features = FeatureBuilder().build(df, symbol="XAUUSD", timeframe="H4",
                                      df_h4=None, df_d1=None)
    labels = LabelBuilder().build_labels(features)
    feat_cols = [c for c in features.columns if c not in _EXCLUDE]

    results = compute_wf_pnl_sharpe(
        features, labels, feat_cols,
        model_factory=lambda: lgb.LGBMClassifier(**LGBM_PARAMS),
        config=cfg,
    )
    return results, aggregate_pnl_sharpe(results)


def _run_eurusd_mr(cfg):
    from src.backtesting.wf_pnl import aggregate_pnl_sharpe, compute_wf_pnl_sharpe
    from src.models.mean_reversion_model import MeanReversionModel
    import lightgbm as lgb

    logger.info("=== EURUSD H4 MR: WF-P&L ===")
    df = _fetch_and_validate("EURUSD", "H4")
    mr = MeanReversionModel()
    features = mr.build_features(df, symbol="EURUSD", timeframe="H4")
    labels = MeanReversionModel.default_label_builder().build_labels(features)
    feat_cols = [c for c in features.columns if c not in _EXCLUDE]

    results = compute_wf_pnl_sharpe(
        features, labels, feat_cols,
        model_factory=lambda: lgb.LGBMClassifier(**LGBM_PARAMS),
        config=cfg,
    )
    return results, aggregate_pnl_sharpe(results)


def _print(name: str, agg: dict) -> None:
    print(f"\n{'=' * 56}")
    print(f"  {name}  |  P&L-Sharpe (vectorbt)")
    print(f"{'=' * 56}")
    print(f"  Fenster              : {agg['n_windows']}")
    print(f"  Ø P&L-Sharpe         : {agg['mean_pnl_sharpe']}")
    print(f"  Median P&L-Sharpe    : {agg['median_pnl_sharpe']}")
    print(f"  Std P&L-Sharpe       : {agg['std_pnl_sharpe']}")
    print(f"  Profitable Fenster   : {agg['profitable_windows']}/{agg['n_windows']}")
    print(f"  Trades gesamt        : {agg['total_trades']}")
    print(f"{'=' * 56}\n")


def main() -> int:
    from src.backtesting.vectorbt_runner import BacktestConfig, timeframe_to_freq

    # SCHRITT A: aktuelle BacktestConfig-Defaults (Spread 0.0001, Slippage 1 Pip,
    # Swap 0.0, pip_size 0.0001). freq = 4h fuer H4.
    cfg = BacktestConfig(freq=timeframe_to_freq("H4"))
    logger.info("BacktestConfig: spread_pct={s} slippage_pips={sl} swap_long={swl} "
                "swap_short={sws} pip_size={p} freq={f}",
                s=cfg.spread_pct, sl=cfg.slippage_pips, swl=cfg.swap_long_per_night,
                sws=cfg.swap_short_per_night, p=cfg.pip_size, f=cfg.freq)

    res_x, agg_x = _run_xauusd_tf(cfg)
    res_e, agg_e = _run_eurusd_mr(cfg)

    _print("XAUUSD H4 TF (Test #3)", agg_x)
    _print("EURUSD H4 MR (Test #4)", agg_e)

    out = {
        "config": {
            "spread_pct": cfg.spread_pct, "slippage_pips": cfg.slippage_pips,
            "swap_long_per_night": cfg.swap_long_per_night,
            "swap_short_per_night": cfg.swap_short_per_night,
            "pip_size": cfg.pip_size, "freq": cfg.freq,
        },
        "XAUUSD_H4_TF": {"aggregate": agg_x, "windows": res_x},
        "EURUSD_H4_MR": {"aggregate": agg_e, "windows": res_e},
    }
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("pnl_sharpe_result.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    logger.info("Ergebnis geschrieben -> {p}", p=out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
