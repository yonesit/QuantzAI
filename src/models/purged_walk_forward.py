"""
src/models/purged_walk_forward.py
Purged Walk-Forward Cross-Validation mit Embargo (Lopez de Prado, adaptiert).

Zeitbasierte Folds, Train immer VOR Test. Zwei Leakage-Schutzmechanismen:

  * PURGING : Ein Trainings-Sample bei Bar i traegt ein Label, dessen Fenster
              [i, i + label_horizon] in die Zukunft reicht. Reicht dieses
              Fenster in den Testzeitraum, wuerde das Label leaken. Solche
              Trainings-Bars werden entfernt -> es bleibt eine Luecke von
              mindestens `label_horizon` Bars vor dem Testblock.

  * EMBARGO : Zusaetzliche Sperrzone von `embargo` Bars unmittelbar vor dem
              Testblock, damit auch autokorrelierte Features nicht leaken.

Ergebnis pro Fold: Train endet spaetestens bei
    test_start - label_horizon - embargo
Damit kann KEIN Trainings-Label-Fenster den Testblock beruehren.

Arbeitet in Bar-Index-Raum (die Labels wurden ebenfalls per Bar-Position mit
festem Horizont erzeugt) – das macht das Purging exakt statt approximativ.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass
class PurgedWalkForward:
    n_splits: int = 5
    label_horizon: int = 16
    embargo: int = 16
    mode: str = "expanding"        # "expanding" | "rolling"
    min_train: int = 500           # Mindest-Trainingsgroesse, sonst Fold verworfen

    def __post_init__(self) -> None:
        if self.n_splits < 1:
            raise ValueError("n_splits muss >= 1 sein.")
        if self.label_horizon < 0 or self.embargo < 0:
            raise ValueError("label_horizon/embargo muessen >= 0 sein.")
        if self.mode not in ("expanding", "rolling"):
            raise ValueError("mode muss 'expanding' oder 'rolling' sein.")

    def split(self, n_samples: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Liefert (train_idx, test_idx) je Fold als int-Arrays (Bar-Positionen)."""
        if n_samples < (self.n_splits + 1):
            raise ValueError("Zu wenige Samples fuer die gewuenschten Folds.")

        fold = n_samples // (self.n_splits + 1)
        gap = self.label_horizon + self.embargo

        for k in range(1, self.n_splits + 1):
            test_start = k * fold
            test_end = (k + 1) * fold if k < self.n_splits else n_samples
            train_end = test_start - gap           # Purge + Embargo als harte Luecke
            train_start = 0 if self.mode == "expanding" else max(0, test_start - fold)

            if train_end - train_start < self.min_train:
                continue  # zu kleiner Train -> Fold ueberspringen

            train_idx = np.arange(train_start, train_end, dtype=int)
            test_idx = np.arange(test_start, test_end, dtype=int)
            yield train_idx, test_idx

    def folds(self, n_samples: int) -> list[tuple[np.ndarray, np.ndarray]]:
        return list(self.split(n_samples))


def assert_no_leakage(
    train_idx: np.ndarray, test_idx: np.ndarray, label_horizon: int, embargo: int = 0
) -> None:
    """Wirft AssertionError, wenn ein Trainings-Label-Fenster den Test beruehrt.

    Pruefkriterium: max(train) + label_horizon + embargo < min(test).
    """
    if len(train_idx) == 0 or len(test_idx) == 0:
        return
    last_train = int(train_idx.max())
    first_test = int(test_idx.min())
    reach = last_train + label_horizon + embargo
    assert reach < first_test, (
        f"Leakage: Train-Bar {last_train} + Horizont {label_horizon} + Embargo "
        f"{embargo} = {reach} >= erster Test-Bar {first_test}"
    )
