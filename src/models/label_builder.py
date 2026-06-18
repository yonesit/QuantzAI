"""
src/models/label_builder.py
LabelBuilder – Triple-Barrier-Label-Erzeugung fuer ueberwachtes Lernen.

Fuer jede Kerze t werden drei Barrieren gesetzt:
  TP = close[t] + tp_atr_mult * ATR[t]   (obere Barriere)
  SL = close[t] - sl_atr_mult * ATR[t]   (untere Barriere)
  Zeitlimit = max_candles Kerzen in die Zukunft

Die naechsten Kerzen werden nacheinander geprueft:
  high[t+k] >= TP  zuerst  -> Label  1  (Long gewinnt)
  low[t+k]  <= SL  zuerst  -> Label -1  (Short gewinnt)
  Kein Treffer in N Kerzen -> Label  0  (neutrales Zeitlimit)

Kein Look-ahead-Bias: Labels basieren ausschliesslich auf *zukuenftigen*
OHLCV-Daten, nie auf Features, die t selbst oder Zukuenftiges vorwegnehmen.

Bei gleichzeitigem TP- und SL-Treffer innerhalb derselben Kerze
(Aufwaerts- und Abwaertsdurchbruch) gewinnt pessimistisch SL (Label -1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


class LabelBuilder:
    """
    Erzeugt Triple-Barrier-Labels aus einem OHLCV-DataFrame mit ATR-Spalte.

    Parameters
    ----------
    tp_atr_mult  : ATR-Multiplikator fuer Take-Profit-Barriere (Standard: 2.0)
    sl_atr_mult  : ATR-Multiplikator fuer Stop-Loss-Barriere   (Standard: 1.5)
    max_candles  : Maximale Anzahl zukuenftiger Kerzen (Zeitlimit)           (Standard: 24)
    atr_col      : Spaltenname der ATR-Spalte im DataFrame                   (Standard: "atr_14")
    """

    def __init__(
        self,
        tp_atr_mult: float = 2.0,
        sl_atr_mult: float = 1.5,
        max_candles: int = 24,
        atr_col: str = "atr_14",
    ) -> None:
        if tp_atr_mult <= 0:
            raise ValueError("tp_atr_mult muss positiv sein.")
        if sl_atr_mult <= 0:
            raise ValueError("sl_atr_mult muss positiv sein.")
        if max_candles < 1:
            raise ValueError("max_candles muss mindestens 1 sein.")

        self._tp_mult = tp_atr_mult
        self._sl_mult = sl_atr_mult
        self._max_candles = max_candles
        self._atr_col = atr_col

    # ── Oeffentliche Schnittstelle ────────────────────────────────────────────

    def build_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        Berechnet Triple-Barrier-Labels fuer alle Kerzen.

        Parameters
        ----------
        df : DataFrame mit Spalten close, high, low sowie atr_col.
             Index kann beliebig sein (DatetimeIndex oder Integer).

        Returns
        -------
        pd.Series mit Labels (1, -1, 0), gleichem Index wie df, Name="label".

        Hinweise
        --------
        - Letzte max_candles Zeilen erhalten Label 0 sofern keine Barriere
          mehr in den verbliebenen Kerzen erreicht wird.
        - Zeilen mit NaN-ATR oder ATR <= 0 erhalten Label 0.
        """
        self._validate(df)

        closes = df["close"].to_numpy(dtype=float)
        highs  = df["high"].to_numpy(dtype=float)
        lows   = df["low"].to_numpy(dtype=float)
        atrs   = df[self._atr_col].to_numpy(dtype=float)
        n = len(df)

        labels = np.zeros(n, dtype=np.int8)

        for t in range(n):
            atr = atrs[t]
            if np.isnan(atr) or atr <= 0.0:
                continue  # Label bleibt 0

            tp = closes[t] + self._tp_mult * atr
            sl = closes[t] - self._sl_mult * atr

            end = min(t + 1 + self._max_candles, n)
            for k in range(t + 1, end):
                tp_hit = highs[k] >= tp
                sl_hit = lows[k]  <= sl

                if tp_hit and sl_hit:
                    labels[t] = -1   # pessimistisch: SL gewinnt bei Gleichstand
                    break
                if tp_hit:
                    labels[t] = 1
                    break
                if sl_hit:
                    labels[t] = -1
                    break
            # kein Break -> Label bleibt 0 (Zeitlimit)

        result = pd.Series(labels.astype(int), index=df.index, name="label")
        self._log_distribution(result)
        return result

    # ── Interna ───────────────────────────────────────────────────────────────

    def _validate(self, df: pd.DataFrame) -> None:
        required = {"close", "high", "low", self._atr_col}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"LabelBuilder: Fehlende Spalten im DataFrame: {sorted(missing)}"
            )
        if df.empty:
            raise ValueError("LabelBuilder: DataFrame ist leer.")

    def _log_distribution(self, labels: pd.Series) -> None:
        total = len(labels)
        if total == 0:
            return
        counts = labels.value_counts()
        parts = []
        for val, name in [(-1, "Short"), (0, "Neutral"), (1, "Long")]:
            cnt = int(counts.get(val, 0))
            pct = cnt / total * 100.0
            parts.append(f"{name}: {cnt} ({pct:.1f}%)")

        logger.info(
            "LabelBuilder Klassenverteilung | {dist} | Gesamt: {total}",
            dist=" | ".join(parts),
            total=total,
        )
