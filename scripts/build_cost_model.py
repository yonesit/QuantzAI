"""
scripts/build_cost_model.py
Baut das kombinierte Kostenmodell EURUSD (Spread + Kommission GETRENNT) und
schreibt config/cost_model_EURUSD.yaml.

Quellen:
  * Kommission: MT5-Deal-History (history_deals_get) des Fusion-Kontos, Pip-
    Wert per order_calc_profit -> commission_per_side in Pips (GEMESSEN).
  * Effektiver Spread je Session: Fusion-Tick-Sample (zweiseitige Quotes),
    erzeugt von fetch_fusion_ticks_spread.py (GEMESSEN, Demo-Limit ~3 Monate).
  * Dukascopy-Session-Spread: langer 2016-2026-Datensatz (Referenz fuers
    Mapping der Vollhistorie).

Aufruf:
    python scripts/build_cost_model.py --symbol EURUSD
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.spread_calibration import (  # noqa: E402
    assign_sessions, price_spread_to_pips, build_cost_model, SESSION_NAMES,
)

DUKA_PARQUET = Path("data/processed/EURUSD_M15_2016-2026.parquet")
TICK_SAMPLE = Path("data/processed/fusion_ref/EURUSD_tick_spread_sample.parquet")
CONFIG_DIR = Path("config")

# Fallback-Kommission (GEMESSEN 2026-07-01, 68 EURUSD-Deals) fuer den Fall,
# dass MT5 nicht erreichbar ist – wird dann als solches gekennzeichnet.
FALLBACK_COMMISSION_SIDE_PIPS = 0.232


def measure_commission_pips(symbol: str) -> tuple[float, str]:
    """Misst commission_per_side in Pips aus MT5-Deal-History. (wert, quelle)."""
    load_dotenv()
    try:
        import MetaTrader5 as mt5
        ok = mt5.initialize(
            login=int(os.environ.get("MT5_LOGIN", "0")),
            password=os.environ.get("MT5_PASSWORD", ""),
            server=os.environ.get("MT5_SERVER", ""),
        )
        if not ok:
            raise RuntimeError(f"initialize: {mt5.last_error()}")
        frm = datetime(2020, 1, 1, tzinfo=timezone.utc)
        to = datetime.now(timezone.utc)
        deals = mt5.history_deals_get(frm, to)
        rows = [d._asdict() for d in deals] if deals else []
        df = pd.DataFrame(rows)
        eur = df[(df["symbol"] == symbol) & (df["volume"] > 0) & (df["commission"] != 0)]
        if len(eur) < 5:
            mt5.shutdown()
            logger.warning("Nur {n} EURUSD-Deals mit Kommission – Fallback.", n=len(eur))
            return FALLBACK_COMMISSION_SIDE_PIPS, "assumed_fallback"
        comm_per_lot_side = float((eur["commission"].abs() / eur["volume"]).median())
        tick = mt5.symbol_info_tick(symbol)
        p = tick.ask
        pip_val = float(mt5.order_calc_profit(mt5.ORDER_TYPE_BUY, symbol, 1.0, p, p + 0.0001))
        mt5.shutdown()
        pips = comm_per_lot_side / pip_val
        logger.info("Kommission gemessen | {c:.4f} Ccy/Lot/Seite | Pip-Wert {pv:.4f} "
                    "| {p:.4f} Pips/Seite | n={n}",
                    c=comm_per_lot_side, pv=pip_val, p=pips, n=len(eur))
        return round(pips, 4), f"measured_mt5_deals_n={len(eur)}"
    except Exception as exc:  # noqa: BLE001
        logger.warning("Kommissionsmessung fehlgeschlagen ({e}) – Fallback.", e=exc)
        return FALLBACK_COMMISSION_SIDE_PIPS, "assumed_fallback"


def duka_session_spread(symbol: str) -> dict[str, float]:
    d = pd.read_parquet(DUKA_PARQUET)
    d["sp"] = price_spread_to_pips(d["spread_median"].to_numpy(), symbol)
    d["session"] = assign_sessions(d["timestamp"])
    return {s: round(float(d.loc[d["session"] == s, "sp"].median()), 4) for s in SESSION_NAMES}


def fusion_session_spread() -> dict[str, dict]:
    f = pd.read_parquet(TICK_SAMPLE)
    if "session" not in f.columns:
        f["session"] = assign_sessions(f["timestamp"])
    out: dict[str, dict] = {}
    for s in SESSION_NAMES:
        v = f.loc[f["session"] == s, "spread_pips"]
        out[s] = {
            "median": round(float(v.median()), 4) if len(v) else float("nan"),
            "p90": round(float(v.quantile(0.90)), 4) if len(v) else float("nan"),
            "n": int(len(v)),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="EURUSD")
    args = ap.parse_args()
    symbol = args.symbol

    if not DUKA_PARQUET.exists() or not TICK_SAMPLE.exists():
        logger.error("Fehlende Eingabe: {d} bzw. {t}", d=DUKA_PARQUET, t=TICK_SAMPLE)
        return 1

    comm_pips, comm_source = measure_commission_pips(symbol)
    duka = duka_session_spread(symbol)
    fusion = fusion_session_spread()

    model = build_cost_model(
        symbol=symbol,
        commission_per_side_pips=comm_pips,
        duka_session_spread=duka,
        fusion_session_spread=fusion,
        overlap={
            "spread_ticks": "2026-04-06..2026-06-29 (Demo-Tick-Limit ~3 Monate)",
            "commission_deals": "2020-01-01..now (alle EURUSD-Deals des Kontos)",
            "duka_history": "2016-01..2026-06 (Dukascopy M15)",
        },
        sources={
            "dukascopy": str(DUKA_PARQUET),
            "fusion_ticks": str(TICK_SAMPLE),
            "commission": "MT5 history_deals_get + order_calc_profit (Konto 383619)",
        },
        measured_vs_assumed={
            "commission_per_side_pips": comm_source,
            "effective_spread_by_session": "measured_fusion_ticks_two_sided_quotes",
            "duka_spread": "measured_dukascopy",
            "caveat_spread": (
                "Fusion-Tick-History nur ~3 Monate (2026-04+); ~97% bid==ask "
                "Feed-Artefakte verworfen; Session-Medianen rauschen (kleine n). "
                "Overall-Effektivspread ~0.3 Pips ~ Dukascopy -> robuster Faktor ~1.0. "
                "Dukascopy-Rollover-Spike (P90 2.15) bleibt massgeblich fuer 21-24 UTC."
            ),
        },
        notes=(
            "Raw-/Zero-Konto: reale Kosten = effektiver Spread (variabel) + fixe "
            "Kommission. Bar-spread-Feld war zu 95% Null und wurde verworfen. "
            "Fuers Labeling der Dukascopy-Vollhistorie: Dukascopy-Spread je Session "
            "mit recommended_robust_factor skalieren, dann round_turn-Kommission "
            "addieren (Konvention siehe cost_convention)."
        ),
    )

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    out = CONFIG_DIR / f"cost_model_{symbol}.yaml"
    with open(out, "w", encoding="utf-8") as fh:
        yaml.safe_dump(model, fh, sort_keys=False, allow_unicode=True)

    print("\n" + "=" * 66)
    print(f"Kostenmodell geschrieben: {out}")
    print(f"Kommission/Seite: {comm_pips} Pips ({comm_source}) | "
          f"Round-Turn: {model['commission']['round_turn_pips']} Pips")
    print(f"Empf. robuster Duka->Fusion-Faktor: {model['recommended_robust_factor']}")
    print("Total Round-Turn je Session (Pips):")
    for s, v in model["total_roundturn_cost_by_session_pips"].items():
        print(f"  {s:9s} {v}")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
