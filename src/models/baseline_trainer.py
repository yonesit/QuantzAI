"""
src/models/baseline_trainer.py
Baseline-Trainings-Pipeline (Dry-Run) fuer den kostenbewusst gelabelten
M15-Datensatz. Validierung ausschliesslich ueber Purged Walk-Forward, Erfolgs-
mass ist ein ECHTES, kostenbereinigtes P&L-OOS-Sharpe – keine Klassifikations-
metrik.

Ablauf je Label-Design (symmetric / asymmetric):
  1. Kausale, stationaere Features (nur Vergangenheit; explizit leakage-getestet).
  2. LightGBM-Classifier je Purged-WF-Fold auf Label==1 (netto-profitabler Long).
  3. Aus den Test-Vorhersagen die Trades simulieren, die das Modell nehmen wuerde
     (p>Schwelle -> Long), Netto-P&L (Kosten stecken im Label) bei FIXEM Risiko
     pro Trade -> Sharpe je Fold.
  4. Benchmarks (immer long / zufaellig / Basisrate) zum Vergleich.

KEIN Look-Ahead: Features nutzen nur Daten bis Bar t; der Entry liegt auf t+1.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from loguru import logger

import ta
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import ADXIndicator, MACD, EMAIndicator, CCIIndicator
from ta.volatility import AverageTrueRange, BollingerBands

from src.models.purged_walk_forward import PurgedWalkForward, assert_no_leakage
from src.models import pnl_backtest as pnl

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None


PIP = 0.0001
M15_BARS_PER_YEAR = 35_040


# ── Feature-Bau (kausal, stationaer) ─────────────────────────────────────────

def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Kausale, stationaere Feature-Matrix aus OHLCV. Alle Werte nutzen nur
    Daten bis einschliesslich Bar t (Entry liegt auf t+1 -> kein Look-Ahead).

    Rueckgabe: (frame_mit_features, feature_namen). Nicht-berechenbare Warmup-
    Zeilen tragen NaN und werden spaeter je Fold verworfen.
    """
    out = df.copy().reset_index(drop=True)
    close, high, low, open_ = out["close"], out["high"], out["low"], out["open"]
    vol = out["volume"] if "volume" in out.columns else pd.Series(1.0, index=out.index)

    atr = AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()
    safe_atr = atr.replace(0, np.nan)

    feats: dict[str, pd.Series] = {}
    feats["ret_1"] = close.pct_change(1)
    feats["ret_4"] = close.pct_change(4)
    feats["ret_16"] = close.pct_change(16)
    feats["rsi_14"] = RSIIndicator(close=close, window=14).rsi()
    feats["atr_norm"] = atr / close
    for p in (9, 20, 50):
        ema = EMAIndicator(close=close, window=p).ema_indicator()
        feats[f"dist_ema{p}"] = (close - ema) / safe_atr
    macd = MACD(close=close)
    feats["macd_diff_norm"] = macd.macd_diff() / safe_atr
    adx = ADXIndicator(high=high, low=low, close=close, window=14)
    feats["adx"] = adx.adx()
    feats["adx_pos"] = adx.adx_pos()
    feats["adx_neg"] = adx.adx_neg()
    bb = BollingerBands(close=close, window=20, window_dev=2.0)
    mid = bb.bollinger_mavg()
    width = (bb.bollinger_hband() - bb.bollinger_lband())
    feats["bb_pos"] = (close - mid) / width.replace(0, np.nan)
    feats["bb_width_norm"] = width / close
    stoch = StochasticOscillator(high=high, low=low, close=close, window=14, smooth_window=3)
    feats["stoch_k"] = stoch.stoch()
    feats["stoch_d"] = stoch.stoch_signal()
    feats["cci_20"] = CCIIndicator(high=high, low=low, close=close, window=20).cci()
    feats["candle_body"] = (close - open_) / safe_atr
    feats["hl_range"] = (high - low) / safe_atr
    rng = (high - low).replace(0, np.nan)
    feats["close_pos"] = (close - low) / rng
    feats["vol_ratio"] = vol / vol.rolling(50, min_periods=10).mean()

    ts = pd.to_datetime(out["timestamp"], utc=True)
    feats["hour"] = ts.dt.hour.astype(float)
    feats["dow"] = ts.dt.dayofweek.astype(float)

    names = list(feats.keys())
    fdf = pd.DataFrame(feats, index=out.index)
    # WICHTIG: um Look-Ahead voellig auszuschliessen, werden die Indikatorwerte
    # von Bar t verwendet, um den Entry auf t+1 zu entscheiden. Nichts nutzt t+1.
    for name in names:
        out[name] = fdf[name].to_numpy()
    return out, names


# ── Ergebnis-Container ───────────────────────────────────────────────────────

@dataclass
class DesignResult:
    design: str
    fold_sharpes: list[float] = field(default_factory=list)
    fold_trades: list[int] = field(default_factory=list)
    all_returns: list[float] = field(default_factory=list)
    bench_long_sharpes: list[float] = field(default_factory=list)
    bench_random_sharpes: list[float] = field(default_factory=list)
    base_rate: float = 0.0
    importances: dict[str, float] = field(default_factory=dict)

    def summary(self) -> dict:
        s = np.array(self.fold_sharpes, dtype=float)
        r = np.array(self.all_returns, dtype=float)
        bl = np.array(self.bench_long_sharpes, dtype=float)
        return {
            "design": self.design,
            "n_folds": len(s),
            "sharpe_mean": float(s.mean()) if len(s) else 0.0,
            "sharpe_std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
            "sharpe_folds": [round(x, 3) for x in self.fold_sharpes],
            "n_trades": int(np.sum(self.fold_trades)),
            "profit_factor": round(pnl.profit_factor(r), 3),
            "win_rate": round(pnl.win_rate(r), 4),
            "max_drawdown": round(pnl.max_drawdown(r), 4),
            "bench_long_sharpe_mean": float(bl.mean()) if len(bl) else 0.0,
            "base_rate_label1": round(self.base_rate, 4),
        }


# ── LGBM-Fit ─────────────────────────────────────────────────────────────────

def _fit_lgbm(X: np.ndarray, y: np.ndarray, seed: int = 42):
    if lgb is None:  # pragma: no cover
        raise RuntimeError("lightgbm nicht installiert.")
    model = lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.05, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=100,
        random_state=seed, n_jobs=-1, verbose=-1,
    )
    model.fit(X, y)
    return model


# ── Haupt-Dry-Run ────────────────────────────────────────────────────────────

def run_design(
    frame: pd.DataFrame,
    feature_names: list[str],
    design: str,
    *,
    k: float = 1.5,
    sl_mult: float = 1.0,
    threshold: float = 0.5,
    risk_frac: float = 0.01,
    wf: PurgedWalkForward | None = None,
    seed: int = 42,
) -> DesignResult:
    """Trainiert + evaluiert ein Label-Design ueber Purged-WF-Folds (P&L-Sharpe)."""
    wf = wf or PurgedWalkForward(n_splits=5, label_horizon=16, embargo=16)
    n = len(frame)
    label_col = f"label_{design}"
    net_col = f"net_pips_{design}"

    y_bin = (frame[label_col] == 1).astype(float).to_numpy()
    labeled = (frame["status"] == "labeled").to_numpy() & frame[label_col].notna().to_numpy()
    feat_ok = frame[feature_names].notna().all(axis=1).to_numpy()
    usable = labeled & feat_ok

    X_all = frame[feature_names].to_numpy(dtype=float)
    net_all = frame[net_col].to_numpy(dtype=float)
    atr_all = frame["atr"].to_numpy(dtype=float)
    ts = pd.to_datetime(frame["timestamp"], utc=True)

    res = DesignResult(design=design)
    res.base_rate = float(y_bin[usable].mean()) if usable.any() else 0.0
    rng = np.random.default_rng(seed)
    imp_acc = np.zeros(len(feature_names))
    imp_folds = 0

    for train_idx, test_idx in wf.split(n):
        assert_no_leakage(train_idx, test_idx, wf.label_horizon, wf.embargo)
        tr = train_idx[usable[train_idx]]
        te = test_idx[usable[test_idx]]
        if len(tr) < wf.min_train or len(te) < 20:
            continue
        if len(np.unique(y_bin[tr])) < 2:
            continue

        model = _fit_lgbm(X_all[tr], y_bin[tr], seed=seed)
        proba = model.predict_proba(X_all[te])[:, 1]

        day_start, day_end = ts.iloc[te[0]], ts.iloc[te[-1]]

        def _sharpe_for(mask: np.ndarray) -> tuple[float, np.ndarray]:
            """Ein-Position-zur-Zeit (nicht ueberlappend) -> taegliches Sharpe."""
            pos = te[mask]
            keep = pnl.non_overlapping_mask(pos, wf.label_horizon)
            entries = pos[keep]
            if len(entries) == 0:
                return 0.0, np.array([])
            r = pnl.trade_returns(net_all[entries], atr_all[entries],
                                  k=k, sl_mult=sl_mult, pip=PIP, risk_frac=risk_frac)
            sh = pnl.daily_sharpe(r, ts.iloc[entries].to_numpy(), day_start, day_end)
            return sh, r

        # Modell: p > Schwelle -> Long
        sh_model, r_model = _sharpe_for(proba > threshold)
        res.fold_sharpes.append(sh_model)
        res.fold_trades.append(int(len(r_model)))
        res.all_returns.extend(r_model.tolist())

        # Benchmark: immer long
        sh_long, _ = _sharpe_for(np.ones(len(te), dtype=bool))
        res.bench_long_sharpes.append(sh_long)
        # Benchmark: zufaellig (gleiche Trade-Schwelle)
        sh_rand, _ = _sharpe_for(rng.random(len(te)) > threshold)
        res.bench_random_sharpes.append(sh_rand)

        imp_acc += model.booster_.feature_importance(importance_type="gain")
        imp_folds += 1

    if imp_folds:
        imp = imp_acc / imp_folds
        order = np.argsort(imp)[::-1]
        res.importances = {feature_names[i]: float(imp[i]) for i in order}
    return res
