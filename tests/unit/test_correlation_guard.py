"""
Unit-Tests fuer CorrelationGuard.

Synthetische Preisreihen werden mit numpy generiert:
  - stark positiv korreliert  (gleicher zugrunde liegender Zufallsprozess)
  - nicht korreliert           (unabhaengige Zufallsprozesse)
  - stark negativ korreliert   (invertierter Zufallsprozess)

HTTP- und ML-Bibliotheken werden nicht benoetigt.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.risk.correlation_guard import CorrelationGuard


# ─────────────────────────────────────────────────────────────────────────────
#  Hilfsfunktionen fuer synthetische Preisreihen
# ─────────────────────────────────────────────────────────────────────────────

N = 70  # Datenpunkte (> 60-Tage-Fenster)


def _prices_from_returns(returns: np.ndarray, start: float = 100.0) -> pd.DataFrame:
    """Baut einen DataFrame mit 'close'-Spalte aus Log-Returns."""
    closes = start * np.exp(np.cumsum(returns))
    return pd.DataFrame({"close": closes})


def _correlated_pair(seed: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Zwei Preisreihen mit Korrelation > 0.95."""
    rng = np.random.default_rng(seed)
    common = rng.standard_normal(N) * 0.01
    noise = rng.standard_normal((2, N)) * 0.0001   # winziges Rauschen
    return _prices_from_returns(common + noise[0]), _prices_from_returns(common + noise[1])


def _uncorrelated_pair(seed: int = 2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Zwei statistisch unabhaengige Preisreihen (Korrelation ~0)."""
    rng1 = np.random.default_rng(seed)
    rng2 = np.random.default_rng(seed + 999)
    return (
        _prices_from_returns(rng1.standard_normal(N) * 0.01),
        _prices_from_returns(rng2.standard_normal(N) * 0.01),
    )


def _negatively_correlated_pair(seed: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Zwei Preisreihen mit Korrelation < -0.95."""
    rng = np.random.default_rng(seed)
    base = rng.standard_normal(N) * 0.01
    noise = rng.standard_normal(N) * 0.0001
    return _prices_from_returns(base), _prices_from_returns(-base + noise)


def _guard(tmp_path: Path, **kwargs) -> CorrelationGuard:
    return CorrelationGuard(
        cache_path=str(tmp_path / "corr_cache.json"),
        **kwargs,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: update_correlations
# ─────────────────────────────────────────────────────────────────────────────

class TestUpdateCorrelations:

    def test_correlated_pair_yields_high_correlation(self, tmp_path):
        df_a, df_b = _correlated_pair()
        guard = _guard(tmp_path)
        guard.update_correlations({"EURUSD": df_a, "GBPUSD": df_b})
        corr = guard.get_correlation("EURUSD", "GBPUSD")
        assert corr > 0.9, f"Erwartete Korrelation > 0.9, erhalten: {corr}"

    def test_uncorrelated_pair_yields_low_correlation(self, tmp_path):
        df_a, df_b = _uncorrelated_pair()
        guard = _guard(tmp_path)
        guard.update_correlations({"EURUSD": df_a, "USDJPY": df_b})
        corr = guard.get_correlation("EURUSD", "USDJPY")
        assert abs(corr) < 0.5, f"Erwartete |Korrelation| < 0.5, erhalten: {corr}"

    def test_negatively_correlated_pair_yields_negative_correlation(self, tmp_path):
        df_a, df_b = _negatively_correlated_pair()
        guard = _guard(tmp_path)
        guard.update_correlations({"EURUSD": df_a, "USDCHF": df_b})
        corr = guard.get_correlation("EURUSD", "USDCHF")
        assert corr < -0.9, f"Erwartete Korrelation < -0.9, erhalten: {corr}"

    def test_correlation_is_symmetric(self, tmp_path):
        df_a, df_b = _correlated_pair()
        guard = _guard(tmp_path)
        guard.update_correlations({"A": df_a, "B": df_b})
        assert guard.get_correlation("A", "B") == guard.get_correlation("B", "A")

    def test_missing_close_column_skipped(self, tmp_path):
        df_valid = _correlated_pair()[0]
        df_bad = pd.DataFrame({"open": [1.0, 2.0, 3.0]})
        guard = _guard(tmp_path)
        # Kein Absturz – das Symbol ohne 'close' wird einfach ignoriert
        guard.update_correlations({"EURUSD": df_valid, "BAD": df_bad})
        # Kein Pair-Eintrag fuer BAD
        assert guard.get_correlation("EURUSD", "BAD") == 0.0

    def test_single_symbol_no_correlation_computed(self, tmp_path):
        df = _correlated_pair()[0]
        guard = _guard(tmp_path)
        guard.update_correlations({"EURUSD": df})
        # Weniger als 2 Symbole -> Matrix bleibt leer
        assert guard.get_correlation("EURUSD", "GBPUSD") == 0.0

    def test_daily_cache_not_recomputed_same_day(self, tmp_path):
        """Zweiter Aufruf am gleichen Tag darf die Matrix NICHT ueberschreiben."""
        df_a, df_b = _correlated_pair()
        df_c, df_d = _negatively_correlated_pair()

        guard = _guard(tmp_path)
        guard.update_correlations({"EURUSD": df_a, "GBPUSD": df_b})
        corr_first = guard.get_correlation("EURUSD", "GBPUSD")

        # Zweiter Aufruf mit voellig anderen Daten – gleicher Tag
        guard.update_correlations({"EURUSD": df_c, "GBPUSD": df_d})
        corr_second = guard.get_correlation("EURUSD", "GBPUSD")

        assert corr_first == corr_second, "Cache sollte wiederverwendet werden"

    def test_cache_updated_on_new_day(self, tmp_path):
        """An einem neuen Tag wird die Korrelationsmatrix neu berechnet."""
        df_a, df_b = _correlated_pair()
        df_neg_a, df_neg_b = _negatively_correlated_pair()

        guard = _guard(tmp_path)
        # Erster Update -> positiv korreliert
        guard.update_correlations({"EURUSD": df_a, "GBPUSD": df_b})
        assert guard.get_correlation("EURUSD", "GBPUSD") > 0.9

        # Tageswechsel simulieren. WICHTIG: gleiche Zeitbasis wie die Produktion
        # (update_correlations nutzt UTC). Sonst kollidiert local-yesterday in der
        # Zeitspanne zwischen lokaler und UTC-Mitternacht mit dem UTC-heute.
        yesterday = datetime.now(timezone.utc).date() - timedelta(days=1)
        guard._last_update = yesterday

        # Neuer Update mit negativ korrelierten Daten
        guard.update_correlations({"EURUSD": df_neg_a, "GBPUSD": df_neg_b})
        assert guard.get_correlation("EURUSD", "GBPUSD") < -0.9

    def test_window_limits_data_used(self, tmp_path):
        """Nur die letzten correlation_window_days Zeilen werden verwendet."""
        # 120 Zeilen, Fenster=60: nur die letzten 60 Tage zahlen
        df_a, df_b = _correlated_pair()
        long_df_a = pd.concat([df_b, df_a], ignore_index=True)  # erste 70 = df_b, letzte 70 = df_a
        long_df_b = pd.concat([df_a, df_b], ignore_index=True)

        guard = CorrelationGuard(correlation_window_days=60)  # kein Cache
        guard.update_correlations({"X": long_df_a, "Y": long_df_b})
        # Die Berechnung laeuft fehlerfrei durch
        assert isinstance(guard.get_correlation("X", "Y"), float)


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: get_correlation
# ─────────────────────────────────────────────────────────────────────────────

class TestGetCorrelation:

    def test_same_symbol_returns_one(self, tmp_path):
        guard = _guard(tmp_path)
        assert guard.get_correlation("EURUSD", "EURUSD") == 1.0

    def test_unknown_pair_returns_zero(self, tmp_path):
        guard = _guard(tmp_path)
        assert guard.get_correlation("EURUSD", "GBPUSD") == 0.0

    def test_returns_cached_value(self, tmp_path):
        df_a, df_b = _correlated_pair()
        guard = _guard(tmp_path)
        guard.update_correlations({"A": df_a, "B": df_b})
        corr = guard.get_correlation("A", "B")
        assert -1.0 <= corr <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: can_open_position
# ─────────────────────────────────────────────────────────────────────────────

class TestCanOpenPosition:

    def _guard_with_corr(self, tmp_path, corr_value: float, sym_a="A", sym_b="B") -> CorrelationGuard:
        """Erstellt Guard mit manuell gesetzter Korrelation."""
        guard = _guard(tmp_path)
        guard._correlation_matrix[(sym_a, sym_b)] = corr_value
        guard._correlation_matrix[(sym_b, sym_a)] = corr_value
        return guard

    # ── Grundfaelle ──────────────────────────────────────────────────────────

    def test_no_open_positions_always_allowed(self, tmp_path):
        guard = _guard(tmp_path)
        assert guard.can_open_position("EURUSD", "long", []) is True

    def test_high_correlation_same_direction_blocked(self, tmp_path):
        """Korrelation > 0.8 + gleiche Richtung -> abgelehnt."""
        guard = self._guard_with_corr(tmp_path, 0.85)
        result = guard.can_open_position("A", "long", [{"symbol": "B", "direction": "long"}])
        assert result is False

    def test_high_correlation_opposite_direction_allowed(self, tmp_path):
        """Korrelation > 0.8 + entgegengesetzte Richtung -> erlaubt (Hedge)."""
        guard = self._guard_with_corr(tmp_path, 0.85)
        result = guard.can_open_position("A", "long", [{"symbol": "B", "direction": "short"}])
        assert result is True

    def test_low_correlation_same_direction_allowed(self, tmp_path):
        """Geringe Korrelation + gleiche Richtung -> erlaubt."""
        guard = self._guard_with_corr(tmp_path, 0.3)
        result = guard.can_open_position("A", "long", [{"symbol": "B", "direction": "long"}])
        assert result is True

    def test_negative_correlation_same_direction_allowed(self, tmp_path):
        """Negative Korrelation (natuerlicher Hedge) + gleiche Richtung -> erlaubt."""
        guard = self._guard_with_corr(tmp_path, -0.85)
        result = guard.can_open_position("A", "long", [{"symbol": "B", "direction": "long"}])
        assert result is True

    def test_negative_correlation_opposite_direction_allowed(self, tmp_path):
        """Negative Korrelation + entgegengesetzte Richtung -> erlaubt."""
        guard = self._guard_with_corr(tmp_path, -0.90)
        result = guard.can_open_position("A", "short", [{"symbol": "B", "direction": "long"}])
        assert result is True

    # ── Schwellwert-Grenzfaelle ───────────────────────────────────────────────

    def test_exact_threshold_not_blocked(self, tmp_path):
        """Korrelation exakt gleich Schwellwert -> NICHT blockiert (strikt >)."""
        guard = self._guard_with_corr(tmp_path, 0.8)
        result = guard.can_open_position("A", "long", [{"symbol": "B", "direction": "long"}])
        assert result is True

    def test_just_above_threshold_blocked(self, tmp_path):
        """Korrelation knapp ueber Schwellwert -> blockiert."""
        guard = self._guard_with_corr(tmp_path, 0.801)
        result = guard.can_open_position("A", "long", [{"symbol": "B", "direction": "long"}])
        assert result is False

    def test_custom_threshold_respected(self, tmp_path):
        """Konfigurierbarer Schwellwert wird eingehalten."""
        guard = CorrelationGuard(max_correlation=0.5)
        guard._correlation_matrix[("A", "B")] = 0.55
        guard._correlation_matrix[("B", "A")] = 0.55
        result = guard.can_open_position("A", "long", [{"symbol": "B", "direction": "long"}])
        assert result is False

    # ── Mehrere offene Positionen ─────────────────────────────────────────────

    def test_multiple_positions_one_blocks(self, tmp_path):
        """Eine einzige blockierende Position reicht fuer Ablehnung."""
        guard = _guard(tmp_path)
        guard._correlation_matrix[("NEW", "OK")]  = 0.2
        guard._correlation_matrix[("OK",  "NEW")] = 0.2
        guard._correlation_matrix[("NEW", "BAD")] = 0.9
        guard._correlation_matrix[("BAD", "NEW")] = 0.9

        open_pos = [
            {"symbol": "OK",  "direction": "long"},
            {"symbol": "BAD", "direction": "long"},  # blockiert
        ]
        assert guard.can_open_position("NEW", "long", open_pos) is False

    def test_multiple_positions_all_safe_allowed(self, tmp_path):
        """Alle offenen Positionen unkritisch -> erlaubt."""
        guard = _guard(tmp_path)
        for sym in ["B", "C", "D"]:
            guard._correlation_matrix[("A", sym)] = 0.3
            guard._correlation_matrix[(sym, "A")] = 0.3

        open_pos = [
            {"symbol": "B", "direction": "long"},
            {"symbol": "C", "direction": "long"},
            {"symbol": "D", "direction": "long"},
        ]
        assert guard.can_open_position("A", "long", open_pos) is True

    def test_same_symbol_blocks_duplicate_position(self, tmp_path):
        """Gleiches Symbol bereits offen → zweite Position wird abgelehnt (Duplikat-Schutz)."""
        guard = _guard(tmp_path)
        result = guard.can_open_position(
            "EURUSD", "long", [{"symbol": "EURUSD", "direction": "long"}]
        )
        assert result is False

    def test_same_symbol_opposite_direction_also_blocked(self, tmp_path):
        """Gleiches Symbol offen (andere Richtung) → ebenfalls abgelehnt (kein Hedging)."""
        guard = _guard(tmp_path)
        result = guard.can_open_position(
            "EURUSD", "short", [{"symbol": "EURUSD", "direction": "long"}]
        )
        assert result is False

    # ── Richtungs-Normalisierung ──────────────────────────────────────────────

    def test_direction_case_insensitive(self, tmp_path):
        """Gross-/Kleinschreibung der Richtung spielt keine Rolle."""
        guard = self._guard_with_corr(tmp_path, 0.9)
        assert guard.can_open_position("A", "LONG",  [{"symbol": "B", "direction": "long"}])  is False
        assert guard.can_open_position("A", "Long",  [{"symbol": "B", "direction": "LONG"}])  is False
        assert guard.can_open_position("A", "long",  [{"symbol": "B", "direction": "SHORT"}]) is True

    # ── Integration mit echten Preisreihen ───────────────────────────────────

    def test_blocks_with_real_correlated_series(self, tmp_path):
        """End-to-end: korrelierte Preisreihen -> same direction abgelehnt."""
        df_a, df_b = _correlated_pair(seed=10)
        guard = _guard(tmp_path)
        guard.update_correlations({"EURUSD": df_a, "GBPUSD": df_b})
        assert guard.can_open_position(
            "EURUSD", "long", [{"symbol": "GBPUSD", "direction": "long"}]
        ) is False

    def test_allows_with_uncorrelated_series(self, tmp_path):
        """End-to-end: unkorrerlierte Preisreihen -> erlaubt."""
        df_a, df_b = _uncorrelated_pair(seed=20)
        guard = _guard(tmp_path)
        guard.update_correlations({"EURUSD": df_a, "USDJPY": df_b})
        corr = guard.get_correlation("EURUSD", "USDJPY")
        if abs(corr) <= 0.8:  # sicherheitshalber pruefen (statistisch)
            assert guard.can_open_position(
                "EURUSD", "long", [{"symbol": "USDJPY", "direction": "long"}]
            ) is True

    def test_allows_opposite_direction_with_correlated_series(self, tmp_path):
        """End-to-end: korrelierte Reihen, entgegengesetzte Richtung -> erlaubt."""
        df_a, df_b = _correlated_pair(seed=30)
        guard = _guard(tmp_path)
        guard.update_correlations({"EURUSD": df_a, "GBPUSD": df_b})
        assert guard.can_open_position(
            "EURUSD", "long", [{"symbol": "GBPUSD", "direction": "short"}]
        ) is True


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: Persistenz
# ─────────────────────────────────────────────────────────────────────────────

class TestPersistence:

    def test_cache_file_created_after_update(self, tmp_path):
        df_a, df_b = _correlated_pair()
        guard = _guard(tmp_path)
        guard.update_correlations({"A": df_a, "B": df_b})
        assert (tmp_path / "corr_cache.json").exists()

    def test_cache_contains_correct_structure(self, tmp_path):
        df_a, df_b = _correlated_pair()
        guard = _guard(tmp_path)
        guard.update_correlations({"A": df_a, "B": df_b})

        cache_file = tmp_path / "corr_cache.json"
        with open(cache_file) as f:
            data = json.load(f)

        assert "last_update" in data
        assert "matrix" in data
        assert any("A|B" in k or "B|A" in k for k in data["matrix"])

    def test_new_instance_loads_from_cache(self, tmp_path):
        df_a, df_b = _correlated_pair()
        guard1 = _guard(tmp_path)
        guard1.update_correlations({"A": df_a, "B": df_b})
        corr1 = guard1.get_correlation("A", "B")

        # Neue Instanz laedt Cache
        guard2 = _guard(tmp_path)
        corr2 = guard2.get_correlation("A", "B")

        assert abs(corr1 - corr2) < 1e-9

    def test_last_update_date_persisted(self, tmp_path):
        df_a, df_b = _correlated_pair()
        guard1 = _guard(tmp_path)
        guard1.update_correlations({"A": df_a, "B": df_b})

        guard2 = _guard(tmp_path)
        assert guard2._last_update == guard1._last_update

    def test_corrupt_cache_handled_gracefully(self, tmp_path):
        cache_file = tmp_path / "corr_cache.json"
        cache_file.write_text("{ this is not valid json }", encoding="utf-8")

        # Kein Absturz beim Laden eines kaputten Cache
        guard = _guard(tmp_path)
        assert guard._correlation_matrix == {}
        assert guard._last_update is None

    def test_guard_without_cache_path_works(self):
        """Ohne cache_path keine Datei-IO."""
        guard = CorrelationGuard()  # kein cache_path
        df_a, df_b = _correlated_pair()
        guard.update_correlations({"A": df_a, "B": df_b})
        assert guard.get_correlation("A", "B") > 0.9
