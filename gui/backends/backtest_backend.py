"""
gui/backends/backtest_backend.py
Konkrete BacktestBackend-Implementierung fuer die GUI.

Verbindet:
  - data/features/{SYMBOL}_{TIMEFRAME}_{DATE}.parquet  (von DataPipeline)
  - models/signal_model_v{N}_{DATE}.joblib             (trainiertes SignalModel)
  - BacktestRunner (vectorbt)

Klare Fehlermeldungen wenn Voraussetzungen fehlen, statt kryptischer Exceptions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import pandas as pd
from loguru import logger

from src.backtesting.vectorbt_runner import (
    BacktestConfig,
    BacktestResult,
    BacktestRunner,
    timeframe_to_freq,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Fehlerklasse
# ─────────────────────────────────────────────────────────────────────────────

class BacktestSetupError(Exception):
    """
    Wird ausgeloest wenn Voraussetzungen fuer den Backtest fehlen.
    Die Message ist direkt in der GUI darstellbar (kein Stack-Trace noetig).
    """


# ─────────────────────────────────────────────────────────────────────────────
#  Pfad-Helfer
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_FEATURES_DIR = Path("data/features")
_DEFAULT_MODELS_DIR   = Path("models")


def _find_features_file(
    features_dir: Path, symbol: str, timeframe: str
) -> Optional[Path]:
    """Gibt die aktuellste Parquet-Datei fuer symbol+timeframe zurueck."""
    pattern = f"{symbol.upper()}_{timeframe.upper()}_*.parquet"
    files   = sorted(features_dir.glob(pattern), reverse=True)
    return files[0] if files else None


def _find_model_file(models_dir: Path) -> Optional[Path]:
    """Gibt die aktuellste .joblib-Modelldatei zurueck."""
    files = sorted(models_dir.glob("signal_model_v*.joblib"), reverse=True)
    return files[0] if files else None


# ─────────────────────────────────────────────────────────────────────────────
#  BacktestGUIBackend
# ─────────────────────────────────────────────────────────────────────────────

class BacktestGUIBackend:
    """
    Konkrete Implementierung des BacktestBackend-Protokolls fuer die GUI.

    Laedt Feature-Daten aus data/features/*.parquet, das trainierte
    SignalModel aus models/*.joblib und fuehrt BacktestRunner.run_with_model()
    aus.

    Gibt bei fehlenden Voraussetzungen BacktestSetupError mit verstaendlichen
    Anweisungen aus, die direkt in der GUI angezeigt werden koennen.

    Parameters
    ----------
    features_dir : Ordner mit Parquet-Feature-Dateien (Standard: data/features).
    models_dir   : Ordner mit .joblib-Modelldateien (Standard: models).
    """

    def __init__(
        self,
        features_dir: str | Path = _DEFAULT_FEATURES_DIR,
        models_dir:   str | Path = _DEFAULT_MODELS_DIR,
    ) -> None:
        self._features_dir = Path(features_dir)
        self._models_dir   = Path(models_dir)

    # ── Oeffentliche Schnittstelle (BacktestBackend-Protokoll) ────────────────

    def run_backtest(
        self,
        symbol:    str,
        timeframe: str,
        start:     str,
        end:       str,
        is_split:  Optional[str],
        init_cash: float,
    ) -> BacktestResult:
        """
        Fuehrt einen vollstaendigen Backtest durch.

        Raises BacktestSetupError mit lesbarer Anweisung wenn:
          - Keine Feature-Daten fuer symbol/timeframe vorhanden
          - Kein trainiertes Modell vorhanden
          - Zeitraum hat keine Daten
        """
        features_df = self._load_features(symbol, timeframe, start, end)
        signal_func = self._load_signal_func()

        freq   = timeframe_to_freq(timeframe)
        runner = BacktestRunner(BacktestConfig(init_cash=init_cash, freq=freq))

        logger.info(
            "GUI-Backtest: {sym}/{tf} | {s} – {e} | IS-Ende: {is_end}",
            sym=symbol, tf=timeframe, s=start, e=end, is_end=is_split,
        )
        return runner.run_with_model(
            features_df=features_df,
            signal_func=signal_func,
            is_end=is_split,
        )

    def get_available_symbols(self) -> list[str]:
        """Leitet verfuegbare Symbole aus Dateinamen in features_dir ab."""
        seen: set[str] = set()
        for p in self._features_dir.glob("*.parquet"):
            parts = p.stem.split("_")
            if len(parts) >= 2:
                seen.add(parts[0])
        return sorted(seen)

    def export_markdown(self, result: BacktestResult, path: str) -> None:
        """Exportiert BacktestResult als Markdown-Datei."""
        md = _result_to_markdown(result)
        dest = Path(path)
        dest.write_text(md, encoding="utf-8")
        logger.info("Backtest-Ergebnis exportiert -> {path}", path=dest)

    # ── Interne Lade-Methoden ─────────────────────────────────────────────────

    def _load_features(
        self, symbol: str, timeframe: str, start: str, end: str
    ) -> pd.DataFrame:
        parquet = _find_features_file(self._features_dir, symbol, timeframe)

        if parquet is None:
            raise BacktestSetupError(
                f"Keine Feature-Daten fuer {symbol}/{timeframe} gefunden.\n\n"
                "Bitte zuerst Daten holen:\n"
                "  python scripts/fetch_data.py\n\n"
                f"Gesuchter Ordner:  {self._features_dir.resolve()}\n"
                f"Erwartetes Muster: {symbol.upper()}_{timeframe.upper()}_*.parquet"
            )

        logger.debug("Lade Features: {path}", path=parquet)
        df = pd.read_parquet(parquet)

        # DatetimeIndex sicherstellen
        if not isinstance(df.index, pd.DatetimeIndex):
            col = "timestamp" if "timestamp" in df.columns else df.columns[0]
            df = df.set_index(col)
        df.index = pd.to_datetime(df.index, utc=True)

        if "close" not in df.columns:
            raise BacktestSetupError(
                f"Spalte 'close' fehlt in {parquet.name}.\n"
                "Bitte Daten neu generieren:\n"
                "  python scripts/fetch_data.py"
            )

        # Zeitraum-Filter
        start_ts = pd.Timestamp(start).tz_localize("UTC")
        end_ts   = pd.Timestamp(end).tz_localize("UTC")
        df_slice = df[(df.index >= start_ts) & (df.index <= end_ts)]

        if df_slice.empty:
            avail_min = df.index.min().date()
            avail_max = df.index.max().date()
            raise BacktestSetupError(
                f"Keine Daten im Zeitraum {start} – {end} "
                f"fuer {symbol}/{timeframe}.\n\n"
                f"Verfuegbarer Zeitraum: {avail_min} – {avail_max}\n"
                "Bitte Start/Ende anpassen oder neue Daten holen:\n"
                "  python scripts/fetch_data.py"
            )

        return df_slice

    def _load_signal_func(self) -> Callable:
        model_path = _find_model_file(self._models_dir)

        if model_path is None:
            raise BacktestSetupError(
                "Kein trainiertes Modell vorhanden.\n\n"
                "Bitte zuerst das SignalModel trainieren:\n"
                "  python scripts/train_model.py --symbol EURUSD\n\n"
                f"Gesuchter Ordner:  {self._models_dir.resolve()}\n"
                "Erwartetes Muster: signal_model_v*.joblib"
            )

        from src.models.signal_model import SignalModel
        model = SignalModel.load(model_path)
        logger.info("SignalModel geladen: {path}", path=model_path.name)
        return model.get_signal


# ─────────────────────────────────────────────────────────────────────────────
#  Markdown-Export
# ─────────────────────────────────────────────────────────────────────────────

def _result_to_markdown(result: BacktestResult, name: str = "Backtest") -> str:
    pf    = "∞" if result.profit_factor == float("inf") else f"{result.profit_factor:.2f}"
    ov    = "Ja ⚠" if result.overfitting_warning else "Nein"
    is_s  = f"{result.is_sharpe:.3f}"  if result.is_sharpe  is not None else "–"
    oos_s = f"{result.oos_sharpe:.3f}" if result.oos_sharpe is not None else "–"

    lines = [
        f"# {name}",
        "",
        "| Kennzahl | Wert |",
        "|---|---|",
        f"| Gesamtertrag | {result.total_return:.2%} |",
        f"| Sharpe Ratio | {result.sharpe_ratio:.3f} |",
        f"| Sortino Ratio | {result.sortino_ratio:.3f} |",
        f"| Max. Drawdown | {result.max_drawdown:.2%} |",
        f"| Gewinnfaktor | {pf} |",
        f"| Win-Rate | {result.win_rate:.1%} |",
        f"| Ø Gewinn | {result.avg_win:.2f} |",
        f"| Ø Verlust | {result.avg_loss:.2f} |",
        f"| Trades | {result.n_trades} |",
        f"| IS Sharpe | {is_s} |",
        f"| OOS Sharpe | {oos_s} |",
        f"| Overfitting-Warnung | {ov} |",
    ]
    return "\n".join(lines) + "\n"
