"""
Unit-Tests fuer src/data/spread_calibration.py – reine, MT5-freie Logik.
Keine echten Handelsdaten; YAML-Roundtrip laeuft ueber tmp_path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

from src.data.spread_calibration import (
    mt5_points_to_pips,
    price_spread_to_pips,
    session_of_hour,
    assign_sessions,
    compute_calibration,
    build_cost_model,
    SESSION_NAMES,
)


# ── Einheiten-Umrechnung ─────────────────────────────────────────────────────

class TestUnitConversion:

    def test_points_to_pips_eurusd(self):
        # 10 Points = 1 Pip fuer EURUSD (5-stellig)
        assert mt5_points_to_pips(10, "EURUSD") == pytest.approx(1.0)
        assert mt5_points_to_pips(3, "EURUSD") == pytest.approx(0.3)
        assert mt5_points_to_pips(0, "EURUSD") == pytest.approx(0.0)

    def test_points_to_pips_array(self):
        out = mt5_points_to_pips(np.array([0, 10, 20]), "EURUSD")
        assert list(out) == pytest.approx([0.0, 1.0, 2.0])

    def test_price_spread_to_pips_eurusd(self):
        # 0.00003 Preis = 0.3 Pips
        assert price_spread_to_pips(0.00003, "EURUSD") == pytest.approx(0.3)
        assert price_spread_to_pips(0.0001, "EURUSD") == pytest.approx(1.0)

    def test_unknown_symbol_raises(self):
        with pytest.raises(KeyError):
            mt5_points_to_pips(10, "NOPE")
        with pytest.raises(KeyError):
            price_spread_to_pips(0.0001, "NOPE")


# ── Session-Klassifikation ───────────────────────────────────────────────────

class TestSessions:

    @pytest.mark.parametrize("hour,expected", [
        (0, "Asien"), (6, "Asien"),
        (7, "Europa"), (12, "Europa"),
        (13, "Overlap"), (15, "Overlap"),
        (16, "US"), (20, "US"),
        (21, "Rollover"), (23, "Rollover"),
    ])
    def test_session_of_hour(self, hour, expected):
        assert session_of_hour(hour) == expected

    def test_session_out_of_range(self):
        with pytest.raises(ValueError):
            session_of_hour(24)

    def test_assign_sessions_vectorized(self):
        ts = pd.Series(pd.to_datetime([
            "2024-01-01 03:00", "2024-01-01 10:00",
            "2024-01-01 14:00", "2024-01-01 18:00", "2024-01-01 22:00",
        ], utc=True))
        out = assign_sessions(ts)
        assert list(out) == ["Asien", "Europa", "Overlap", "US", "Rollover"]


# ── compute_calibration ──────────────────────────────────────────────────────

def _mk(ts_spreads):
    return pd.DataFrame({
        "timestamp": pd.to_datetime([t for t, _ in ts_spreads], utc=True),
        "spread_pips": [s for _, s in ts_spreads],
    })


class TestComputeCalibration:

    def test_factor_and_additive(self):
        # Dukascopy 0.3, Fusion 0.6 in derselben Stunde -> Faktor 2, additiv +0.3
        duka = _mk([("2024-01-01 03:00", 0.3), ("2024-01-02 03:00", 0.3)])
        fusion = _mk([("2024-01-01 03:00", 0.6), ("2024-01-02 03:00", 0.6)])
        res = compute_calibration(duka, fusion, "EURUSD")
        asien = res.sessions["Asien"]
        assert asien.duka_median_pips == pytest.approx(0.3)
        assert asien.fusion_median_pips == pytest.approx(0.6)
        assert asien.factor == pytest.approx(2.0)
        assert asien.additive_pips == pytest.approx(0.3)

    def test_overlap_window_clamped(self):
        # Fusion beginnt spaeter -> Overlap-Start = spaeterer Start
        duka = _mk([("2020-01-01 03:00", 0.3), ("2024-01-01 03:00", 0.3)])
        fusion = _mk([("2024-01-01 03:00", 0.5)])
        res = compute_calibration(duka, fusion, "EURUSD")
        assert res.overlap_start == pd.Timestamp("2024-01-01 03:00", tz="UTC")


# ── build_cost_model ─────────────────────────────────────────────────────────

class TestBuildCostModel:

    def _model(self):
        duka = {"Asien": 0.3, "Europa": 0.3, "Overlap": 0.3, "US": 0.3, "Rollover": 0.6}
        fusion = {
            "Asien": {"median": 0.6, "p90": 0.6, "n": 100},
            "Europa": {"median": 0.2, "p90": 0.5, "n": 50},
            "Overlap": {"median": 0.2, "p90": 0.4, "n": 40},
            "US": {"median": 0.2, "p90": 0.4, "n": 40},
            "Rollover": {"median": 0.2, "p90": 0.6, "n": 30},
        }
        return build_cost_model(
            symbol="EURUSD",
            commission_per_side_pips=0.232,
            duka_session_spread=duka,
            fusion_session_spread=fusion,
            overlap={"x": "y"},
            sources={"a": "b"},
            measured_vs_assumed={"commission_per_side_pips": "measured"},
            notes="test",
        )

    def test_commission_separated_and_roundturn(self):
        m = self._model()
        assert m["commission"]["per_side_pips"] == pytest.approx(0.232)
        assert m["commission"]["round_turn_pips"] == pytest.approx(0.464)

    def test_components_kept_separate(self):
        # Spread und Kommission duerfen NICHT zusammengeworfen sein
        m = self._model()
        assert "effective_spread_by_session_pips" in m
        assert "commission" in m
        # total = spread + round-turn-commission (Asien: 0.6 + 0.464)
        assert m["total_roundturn_cost_by_session_pips"]["Asien"] == pytest.approx(1.064)
        assert m["total_roundturn_cost_by_session_pips"]["Europa"] == pytest.approx(0.664)

    def test_mapping_factor(self):
        m = self._model()
        mp = m["duka_to_fusion_spread_mapping"]
        assert mp["Asien"]["factor"] == pytest.approx(2.0)
        assert mp["Europa"]["factor"] == pytest.approx(0.6667, abs=1e-3)

    def test_robust_factor_is_median(self):
        m = self._model()
        # Faktoren: 2.0, 0.667, 0.667, 0.667, 0.333 -> Median 0.667
        assert m["recommended_robust_factor"] == pytest.approx(0.6667, abs=1e-3)

    def test_plausibility_range(self):
        # Round-Turn-Gesamtkosten im realistischen Raw-Konto-Bereich (~0.6–1.2 Pips)
        m = self._model()
        for s in SESSION_NAMES:
            tot = m["total_roundturn_cost_by_session_pips"][s]
            assert 0.5 <= tot <= 1.3, f"{s}: {tot} ausserhalb Plausibilitaet"

    def test_yaml_roundtrip(self, tmp_path):
        m = self._model()
        p = tmp_path / "cost_model_EURUSD.yaml"
        with open(p, "w", encoding="utf-8") as fh:
            yaml.safe_dump(m, fh, sort_keys=False, allow_unicode=True)
        loaded = yaml.safe_load(p.read_text(encoding="utf-8"))
        assert loaded["symbol"] == "EURUSD"
        assert loaded["commission"]["round_turn_pips"] == pytest.approx(0.464)
        assert loaded["total_roundturn_cost_by_session_pips"]["Europa"] == pytest.approx(0.664)
