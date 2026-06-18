"""
src/risk/correlation_guard.py
CorrelationGuard – verhindert redundante Positionen in stark korrelierten Instrumenten.

Berechnet rollierende Pearson-Korrelation der Log-Returns (Standard: 60 Tage)
zwischen allen gehandelten Symbolpaaren und blockiert neue Positionen, wenn
eine bestehende Position in einem hoch korrelierten Symbol in gleicher Richtung
bereits offen ist.

Korrelationsmatrix wird einmal taeglich neu berechnet und gecacht.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone, date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


class CorrelationGuard:
    """
    Ueberwacht Korrelationen zwischen Symbolpaaren und prueft ob eine neue
    Position das Korrelationsrisiko des Portfolios erhoeht.

    Regel: Korrelation > Schwellwert UND gleiche Richtung -> Position abgelehnt.
    Negative Korrelation (z.B. EURUSD vs USDCHF) bleibt erkannt und erlaubt
    Hedging-Logik (negative Korrelation < Schwellwert -> kein Block).

    Parameters
    ----------
    max_correlation          : Schwellwert fuer positive Korrelation (Standard: 0.8)
    correlation_window_days  : Rollfenster fuer Korrelationsberechnung in Tagen (Standard: 60)
    cache_path               : Optionaler Pfad zur Persistenz der Korrelationsmatrix
    """

    def __init__(
        self,
        max_correlation: float = 0.8,
        correlation_window_days: int = 60,
        cache_path: Optional[str] = None,
    ) -> None:
        self._threshold = max_correlation
        self._window = correlation_window_days
        self._cache_path = Path(cache_path) if cache_path else None

        self._correlation_matrix: dict[tuple[str, str], float] = {}
        self._last_update: Optional[date] = None

        if self._cache_path and self._cache_path.exists():
            self._load_cache()

    # ── Korrelationsberechnung ────────────────────────────────────────────────

    def update_correlations(self, price_data: dict[str, pd.DataFrame]) -> None:
        """
        Berechnet Korrelationsmatrix aus Preisreihen (rollendes Fenster).
        Innerhalb eines Kalendertages (UTC) wird die bestehende Matrix beibehalten.

        Parameters
        ----------
        price_data : dict {symbol -> DataFrame mit 'close'-Spalte}
                     Muss mindestens correlation_window_days + 1 Zeilen enthalten.
        """
        today = datetime.now(timezone.utc).date()
        if self._last_update == today and self._correlation_matrix:
            logger.debug("CorrelationGuard: Korrelationsmatrix bereits aktuell (Cache).")
            return

        symbols = list(price_data.keys())
        if len(symbols) < 2:
            logger.warning("CorrelationGuard: Weniger als 2 Symbole – Korrelation nicht berechenbar.")
            return

        returns: dict[str, pd.Series] = {}
        for sym, df in price_data.items():
            if "close" not in df.columns:
                logger.warning("CorrelationGuard: Symbol {sym} ohne 'close'-Spalte.", sym=sym)
                continue
            close = df["close"].tail(self._window + 1).dropna()
            if len(close) < 2:
                logger.warning("CorrelationGuard: Symbol {sym} – zu wenige Datenpunkte.", sym=sym)
                continue
            log_ret = np.log(close / close.shift(1)).dropna()
            returns[sym] = log_ret

        valid_symbols = list(returns.keys())
        new_matrix: dict[tuple[str, str], float] = {}

        for i, sym_a in enumerate(valid_symbols):
            for sym_b in valid_symbols[i + 1:]:
                aligned = pd.concat([returns[sym_a], returns[sym_b]], axis=1).dropna()
                if len(aligned) < 2:
                    corr = 0.0
                else:
                    corr = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
                    if np.isnan(corr):
                        corr = 0.0
                new_matrix[(sym_a, sym_b)] = corr
                new_matrix[(sym_b, sym_a)] = corr

        self._correlation_matrix = new_matrix
        self._last_update = today

        logger.info(
            "CorrelationGuard: Korrelationsmatrix aktualisiert | {n} Paare | Datum={d}",
            n=len(new_matrix) // 2,
            d=today,
        )

        if self._cache_path:
            self._save_cache()

    def get_correlation(self, symbol_a: str, symbol_b: str) -> float:
        """
        Gibt die berechnete Korrelation zwischen zwei Symbolen zurueck.

        Returns 1.0 fuer gleiche Symbole, 0.0 wenn kein Wert gecacht ist.
        """
        if symbol_a == symbol_b:
            return 1.0
        return self._correlation_matrix.get((symbol_a, symbol_b), 0.0)

    # ── Positionspruefung ─────────────────────────────────────────────────────

    def can_open_position(
        self,
        symbol: str,
        direction: str,
        open_positions: list[dict],
    ) -> bool:
        """
        Prueft ob eine neue Position eroeffnet werden darf.

        Eine Position wird abgelehnt wenn:
          Korrelation(symbol, existing_symbol) > max_correlation
          UND direction == existing_direction

        Negative Korrelation (natuerliche Hedges, z.B. EURUSD/USDCHF bei
        gleicher Richtung) hat corr < 0 und liegt immer unter dem Schwellwert
        -> wird nicht blockiert.

        Parameters
        ----------
        symbol         : Symbol der neuen Position (z.B. "EURUSD")
        direction      : "long" oder "short"
        open_positions : Liste von dicts mit {"symbol": str, "direction": str}

        Returns
        -------
        True wenn die Position eroeffnet werden darf, False wenn abgelehnt.
        """
        direction = direction.lower()

        for pos in open_positions:
            pos_sym = pos.get("symbol", "")
            pos_dir = pos.get("direction", "").lower()

            if pos_sym == symbol:
                continue  # gleiches Symbol – Korrelationsregel nicht anwendbar

            corr = self.get_correlation(symbol, pos_sym)

            if corr > self._threshold and direction == pos_dir:
                logger.warning(
                    "CorrelationGuard: Position abgelehnt | {sym} {dir} "
                    "korreliert mit {psym} {pdir} (corr={c:.3f} > {t:.2f})",
                    sym=symbol,
                    dir=direction,
                    psym=pos_sym,
                    pdir=pos_dir,
                    c=corr,
                    t=self._threshold,
                )
                return False

        return True

    # ── Persistenz ────────────────────────────────────────────────────────────

    def _save_cache(self) -> None:
        if self._cache_path is None:
            return
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        serializable = {
            "last_update": self._last_update.isoformat() if self._last_update else None,
            "matrix": {f"{a}|{b}": v for (a, b), v in self._correlation_matrix.items()},
        }
        with open(self._cache_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)

    def _load_cache(self) -> None:
        try:
            with open(self._cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            raw_date = data.get("last_update")
            self._last_update = date.fromisoformat(raw_date) if raw_date else None
            self._correlation_matrix = {}
            for key, val in data.get("matrix", {}).items():
                parts = key.split("|", 1)
                if len(parts) == 2:
                    self._correlation_matrix[(parts[0], parts[1])] = float(val)
            logger.info("CorrelationGuard: Korrelationsmatrix aus Cache geladen.")
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            logger.warning("CorrelationGuard: Cache laden fehlgeschlagen: {exc}", exc=exc)
            self._correlation_matrix = {}
            self._last_update = None
