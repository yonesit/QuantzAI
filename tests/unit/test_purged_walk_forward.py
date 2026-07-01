"""Tests fuer Purged Walk-Forward mit Embargo – Fokus: kein Label-Leakage."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.purged_walk_forward import PurgedWalkForward, assert_no_leakage


class TestSplitStructure:

    def test_train_always_before_test(self):
        wf = PurgedWalkForward(n_splits=4, label_horizon=16, embargo=16, min_train=10)
        for tr, te in wf.split(10_000):
            assert tr.max() < te.min()

    def test_purge_embargo_gap_enforced(self):
        wf = PurgedWalkForward(n_splits=5, label_horizon=16, embargo=16, min_train=10)
        for tr, te in wf.split(20_000):
            # Kein Trainings-Label-Fenster darf den Test beruehren
            assert tr.max() + wf.label_horizon + wf.embargo < te.min()

    def test_assert_no_leakage_passes_for_all_folds(self):
        wf = PurgedWalkForward(n_splits=6, label_horizon=16, embargo=8, min_train=10)
        for tr, te in wf.split(50_000):
            assert_no_leakage(tr, te, wf.label_horizon, wf.embargo)  # darf nicht werfen

    def test_expanding_train_grows(self):
        wf = PurgedWalkForward(n_splits=4, label_horizon=8, embargo=4,
                               mode="expanding", min_train=10)
        sizes = [len(tr) for tr, _ in wf.split(10_000)]
        assert sizes == sorted(sizes)          # monoton wachsend
        assert all(tr.min() == 0 for tr, _ in wf.split(10_000))

    def test_rolling_train_starts_late(self):
        wf = PurgedWalkForward(n_splits=4, label_horizon=8, embargo=4,
                               mode="rolling", min_train=10)
        folds = wf.folds(10_000)
        # spaetere Folds starten nicht bei 0 (rollierendes Fenster)
        assert folds[-1][0].min() > 0

    def test_test_blocks_are_disjoint_and_ordered(self):
        wf = PurgedWalkForward(n_splits=5, label_horizon=8, embargo=8, min_train=10)
        folds = wf.folds(30_000)
        prev_end = -1
        for _, te in folds:
            assert te.min() > prev_end
            prev_end = te.max()


class TestAssertNoLeakage:

    def test_raises_on_overlap(self):
        tr = np.arange(0, 100)
        te = np.arange(105, 200)   # nur 5 Bars Abstand, Horizont 16 -> Leak
        with pytest.raises(AssertionError):
            assert_no_leakage(tr, te, label_horizon=16, embargo=0)

    def test_ok_with_sufficient_gap(self):
        tr = np.arange(0, 100)
        te = np.arange(150, 200)
        assert_no_leakage(tr, te, label_horizon=16, embargo=16)

    def test_empty_is_noop(self):
        assert_no_leakage(np.array([], dtype=int), np.array([1, 2]), 16, 16)


class TestValidation:

    def test_bad_mode_raises(self):
        with pytest.raises(ValueError):
            PurgedWalkForward(mode="nope")

    def test_too_few_samples_raises(self):
        wf = PurgedWalkForward(n_splits=5)
        with pytest.raises(ValueError):
            list(wf.split(3))
