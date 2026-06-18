"""
src/risk/risk_metrics.py
Erweiterte Risikometriken: VaR, CVaR, Kelly-Kriterium, KellyPositionSizer.

Funktionen (unabhaengig nutzbar):
  calculate_var(returns, confidence_level)   -> float  (historischer VaR)
  calculate_cvar(returns, confidence_level)  -> float  (Conditional VaR / ES)
  calculate_kelly_fraction(win_rate, avg_win, avg_loss) -> float
  portfolio_var(open_positions, returns_history, confidence_level) -> float

KellyPositionSizer:
  Gleiche Schnittstelle wie PositionSizer.calculate_lot_size().
  Nutzt Kelly-Fraction statt fester Risiko-Prozentzahl.
  Standard: Half-Kelly (kelly_multiplier=0.5) um Ueberheblichkeit zu begrenzen.

Alle VaR/CVaR-Werte sind als positive Zahlen angegeben (Verlust-Betrag).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from loguru import logger

from src.risk.position_sizer import PositionSizeResult


# ── VaR / CVaR ───────────────────────────────────────────────────────────────

def calculate_var(
    returns: np.ndarray | list,
    confidence_level: float = 0.95,
) -> float:
    """
    Berechnet den historischen Value at Risk (VaR).

    VaR ist der Verlust, der mit Wahrscheinlichkeit (1 - confidence_level)
    ueberschritten wird. Konvention: Ergebnis ist eine positive Zahl
    (Verlust-Betrag), d.h. VaR = 0.05 bedeutet maximal 5 % Verlust
    werden zu 95 % nicht ueberschritten.

    Parameters
    ----------
    returns          : Array historischer Renditen (negativ = Verlust).
    confidence_level : Konfidenzniveau, z.B. 0.95 fuer 95%-VaR.

    Returns
    -------
    float: VaR als positive Zahl. Bei rein positiven Renditen: 0.0.

    Raises
    ------
    ValueError : returns leer, oder confidence_level nicht in (0, 1).
    """
    arr = np.asarray(returns, dtype=float)
    _validate_returns(arr, "calculate_var")
    _validate_confidence(confidence_level, "calculate_var")

    percentile = (1.0 - confidence_level) * 100.0
    var_return = float(np.percentile(arr, percentile))
    return max(0.0, -var_return)


def calculate_cvar(
    returns: np.ndarray | list,
    confidence_level: float = 0.95,
) -> float:
    """
    Berechnet den Conditional Value at Risk (CVaR / Expected Shortfall).

    CVaR ist der erwartete Verlust in den schlimmsten (1 - confidence_level)
    Prozent der Faelle – also der Durchschnitt aller Renditen unterhalb
    des VaR-Schwellwerts.

    Parameters
    ----------
    returns          : Array historischer Renditen.
    confidence_level : Konfidenzniveau, z.B. 0.95 fuer 95%-CVaR.

    Returns
    -------
    float: CVaR als positive Zahl. Immer >= VaR.

    Raises
    ------
    ValueError : returns leer, oder confidence_level nicht in (0, 1).
    """
    arr = np.asarray(returns, dtype=float)
    _validate_returns(arr, "calculate_cvar")
    _validate_confidence(confidence_level, "calculate_cvar")

    percentile = (1.0 - confidence_level) * 100.0
    threshold  = float(np.percentile(arr, percentile))
    tail       = arr[arr <= threshold]

    if tail.size == 0:
        return 0.0
    return max(0.0, -float(tail.mean()))


def portfolio_var(
    open_positions: list[dict],
    returns_history: np.ndarray | list,
    confidence_level: float = 0.95,
) -> float:
    """
    Berechnet den VaR fuer das gesamte offene Portfolio.

    Vereinfachter Ansatz: VaR pro Lot-Einheit skaliert mit der
    Gesamtexposition (Summe aller Lot-Groessen). Konservativ, da
    vollstaendige Korrelation angenommen wird.

    Parameters
    ----------
    open_positions   : Liste von Positions-Dicts (Schluesseln: 'lot_size').
                       Wie von OrderExecutor.get_open_positions() zurueckgegeben.
    returns_history  : Historische Renditen fuer die VaR-Berechnung.
    confidence_level : Konfidenzniveau (Standard: 0.95).

    Returns
    -------
    float: Portfolio-VaR als positive Zahl. 0.0 wenn keine Positionen offen.

    Raises
    ------
    ValueError : returns_history leer oder confidence_level ungueltig.
    """
    if not open_positions:
        return 0.0

    unit_var    = calculate_var(returns_history, confidence_level)
    total_lots  = sum(float(p.get("lot_size", 0.0)) for p in open_positions)

    if total_lots <= 0.0:
        return 0.0

    port_var = unit_var * total_lots
    logger.debug(
        "portfolio_var | Positionen={n} | Lots={lots:.2f} | "
        "Unit-VaR={uv:.4f} | Portfolio-VaR={pv:.4f}",
        n=len(open_positions), lots=total_lots, uv=unit_var, pv=port_var,
    )
    return port_var


# ── Kelly-Kriterium ───────────────────────────────────────────────────────────

def calculate_kelly_fraction(
    win_rate: float,
    avg_win:  float,
    avg_loss: float,
) -> float:
    """
    Berechnet die klassische Kelly-Fraktion.

    Formel: f* = (p * b - q) / b = p - q / b
      p = win_rate
      q = 1 - win_rate
      b = avg_win / avg_loss  (Win/Loss-Verhaeltnis)

    Parameters
    ----------
    win_rate : Gewinn-Wahrscheinlichkeit pro Trade (0.0 – 1.0).
    avg_win  : Durchschnittlicher Gewinn pro gewinnendem Trade (positiv).
    avg_loss : Durchschnittlicher Verlust pro verlierendem Trade (positiv).

    Returns
    -------
    float: Kelly-Fraktion. Kann negativ sein (unfavorable Spiel -> kein Trade).
           Kein internes Clamping – das ist Aufgabe des KellyPositionSizer.

    Raises
    ------
    ValueError : Ungueltige Eingaben (win_rate ausserhalb [0,1],
                 avg_win oder avg_loss nicht positiv).
    """
    if not 0.0 <= win_rate <= 1.0:
        raise ValueError(
            f"win_rate muss in [0.0, 1.0] liegen, erhalten: {win_rate}"
        )
    if avg_win <= 0.0:
        raise ValueError(
            f"avg_win muss positiv sein, erhalten: {avg_win}"
        )
    if avg_loss <= 0.0:
        raise ValueError(
            f"avg_loss muss positiv sein, erhalten: {avg_loss}"
        )

    loss_rate   = 1.0 - win_rate
    win_loss_b  = avg_win / avg_loss
    kelly       = win_rate - loss_rate / win_loss_b
    return float(kelly)


# ── KellyPositionSizer ────────────────────────────────────────────────────────

class KellyPositionSizer:
    """
    Alternativer PositionSizer auf Basis des Kelly-Kriteriums.

    Gleiche Schnittstelle wie PositionSizer (calculate_lot_size),
    damit beide austauschbar per Konfiguration nutzbar sind.

    Risikobetrag = kelly_fraction * kelly_multiplier * account_balance
    Lot-Groesse  = risk_amount / (stop_loss_pips * pip_value)

    Parameters
    ----------
    win_rate            : Historische Gewinn-Wahrscheinlichkeit (0.0 – 1.0).
    avg_win             : Historischer Durchschnittsgewinn pro Trade (positiv).
    avg_loss            : Historischer Durchschnittsverlust pro Trade (positiv).
    kelly_multiplier    : Sicherheits-Cap. Standard: 0.5 (Half-Kelly).
                          Half-Kelly halbiert die aggressiven Kelly-Empfehlungen
                          und reduziert Drawdowns bei Parameterunsicherheit.
    sl_atr_multiplier   : Multiplikator fuer ATR-basierte Stop-Loss-Distanz
                          (Standard: 1.5, wie in PositionSizer).
    min_lot_size        : Mindest-Lot-Groesse (Standard: 0.01).
    max_kelly_fraction  : Maximale Kelly-Fraktion nach Multiplier-Anwendung
                          (Standard: 0.25 = 25 % des Kontostands pro Trade).
    """

    def __init__(
        self,
        win_rate:           float,
        avg_win:            float,
        avg_loss:           float,
        kelly_multiplier:   float = 0.5,
        sl_atr_multiplier:  float = 1.5,
        min_lot_size:       float = 0.01,
        max_kelly_fraction: float = 0.25,
    ) -> None:
        self._win_rate          = win_rate
        self._avg_win           = avg_win
        self._avg_loss          = avg_loss
        self._kelly_multiplier  = kelly_multiplier
        self._sl_multiplier     = sl_atr_multiplier
        self._min_lot_size      = min_lot_size
        self._max_kelly         = max_kelly_fraction

        raw_kelly = calculate_kelly_fraction(win_rate, avg_win, avg_loss)
        self._kelly_fraction = max(0.0, min(raw_kelly * kelly_multiplier, max_kelly_fraction))

        logger.info(
            "KellyPositionSizer | win_rate={wr:.1%} | avg_win={aw:.2f} | "
            "avg_loss={al:.2f} | raw_kelly={rk:.4f} | "
            "applied_fraction={af:.4f} (multiplier={m})",
            wr=win_rate, aw=avg_win, al=avg_loss,
            rk=raw_kelly, af=self._kelly_fraction, m=kelly_multiplier,
        )

    # ── Oeffentliche Schnittstelle ─────────────────────────────────────────

    @property
    def kelly_fraction(self) -> float:
        """Effektiv angewandte Kelly-Fraktion (nach Multiplier und Cap)."""
        return self._kelly_fraction

    def calculate_lot_size(
        self,
        account_balance: float,
        atr:             float,
        symbol:          str,
        risk_pct:        Optional[float] = None,   # ignoriert, nur fuer Schnittstellenkompatibilitaet
        pip_value:       float = 10.0,
        pip_size:        float = 0.0001,
        lot_step:        float = 0.01,
        contract_size:   float = 100_000,
    ) -> PositionSizeResult:
        """
        Berechnet Lot-Groesse auf Basis des Kelly-Kriteriums.

        Gleiche Signatur wie PositionSizer.calculate_lot_size().
        Der risk_pct-Parameter wird ignoriert – die Risikoquote bestimmt
        ausschliesslich die berechnete Kelly-Fraktion.

        Parameters
        ----------
        account_balance : Aktueller Kontostand.
        atr             : ATR des Symbols (fuer Stop-Loss-Berechnung).
        symbol          : Symbol-Name.
        risk_pct        : Ignoriert (Schnittstellenkompatibilitaet).
        pip_value       : Pip-Wert pro Standard-Lot (Standard: 10.0).
        pip_size        : Pip-Groesse (Standard: 0.0001).
        lot_step        : Minimaler Lot-Schritt (Standard: 0.01).
        contract_size   : Kontraktgroesse (Standard: 100 000).

        Returns
        -------
        PositionSizeResult – identische Struktur wie PositionSizer.
        """
        if account_balance <= 0:
            return _rejected(symbol, "Kontostand muss positiv sein.")

        if atr <= 0:
            return _rejected(symbol, "ATR muss positiv sein.")

        if self._kelly_fraction <= 0.0:
            return _rejected(
                symbol,
                f"Kelly-Fraktion ist {self._kelly_fraction:.4f} (<= 0) – "
                "unfavorable Spielparameter, kein Trade.",
            )

        risk_amount        = account_balance * self._kelly_fraction
        stop_loss_distance = atr * self._sl_multiplier
        sl_pips            = stop_loss_distance / pip_size

        if sl_pips <= 0:
            return _rejected(symbol, "Stop-Loss-Distanz in Pips ist 0 oder negativ.")

        raw_lot_size     = risk_amount / (sl_pips * pip_value)
        rounded_lot_size = math.floor(raw_lot_size / lot_step) * lot_step
        rounded_lot_size = round(rounded_lot_size, 8)

        if rounded_lot_size < self._min_lot_size:
            return _rejected(
                symbol,
                f"Kelly-Lot-Groesse ({rounded_lot_size:.5f}) unter Mindestgroesse "
                f"({self._min_lot_size}).",
                risk_amount=risk_amount,
                stop_loss_distance=stop_loss_distance,
            )

        logger.info(
            "KellyPositionSizer: {symbol} | lot={lot} | "
            "risk={risk:.2f} ({frac:.2%} Kelly) | sl_dist={sl:.5f}",
            symbol=symbol, lot=rounded_lot_size,
            risk=risk_amount, frac=self._kelly_fraction,
            sl=stop_loss_distance,
        )

        return PositionSizeResult(
            symbol=symbol,
            lot_size=rounded_lot_size,
            risk_amount=risk_amount,
            stop_loss_distance=stop_loss_distance,
            is_valid=True,
        )

    def update_stats(
        self,
        win_rate: float,
        avg_win:  float,
        avg_loss: float,
    ) -> None:
        """
        Aktualisiert die statistischen Parameter und berechnet die
        Kelly-Fraktion neu (z.B. nach Abschluss weiterer Trades).
        """
        self._win_rate  = win_rate
        self._avg_win   = avg_win
        self._avg_loss  = avg_loss

        raw_kelly            = calculate_kelly_fraction(win_rate, avg_win, avg_loss)
        self._kelly_fraction = max(0.0, min(raw_kelly * self._kelly_multiplier, self._max_kelly))
        logger.info(
            "KellyPositionSizer.update_stats | neues raw_kelly={rk:.4f} | "
            "applied={af:.4f}",
            rk=raw_kelly, af=self._kelly_fraction,
        )


# ── Hilfsfunktionen ───────────────────────────────────────────────────────────

def _validate_returns(arr: np.ndarray, caller: str) -> None:
    if arr.size == 0:
        raise ValueError(f"{caller}: returns darf nicht leer sein.")


def _validate_confidence(confidence_level: float, caller: str) -> None:
    if not 0.0 < confidence_level < 1.0:
        raise ValueError(
            f"{caller}: confidence_level muss in (0, 1) liegen, "
            f"erhalten: {confidence_level}"
        )


def _rejected(
    symbol: str,
    reason: str,
    risk_amount: float = 0.0,
    stop_loss_distance: float = 0.0,
) -> PositionSizeResult:
    return PositionSizeResult(
        symbol=symbol,
        lot_size=0.0,
        risk_amount=risk_amount,
        stop_loss_distance=stop_loss_distance,
        is_valid=False,
        rejection_reason=reason,
    )
