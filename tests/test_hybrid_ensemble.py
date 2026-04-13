"""Unit tests for delegated draw-specialist ensemble behavior."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClassifierMixin

from src.modeling.hybrid_ensemble import HybridDrawOverrideEnsemble


class FixedProbabilityEstimator(BaseEstimator, ClassifierMixin):
    """Small deterministic estimator for override-path tests."""

    def __init__(self, probabilities: list[list[float]]) -> None:
        self.probabilities = probabilities
        self._fitted_probabilities = np.asarray(probabilities, dtype=np.float64)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: NDArray[np.float64] | None = None,
    ) -> FixedProbabilityEstimator:
        self.classes_ = np.array([0, 1, 2], dtype=np.int64)
        self._fitted_probabilities = np.asarray(self.probabilities, dtype=np.float64)
        return self

    def predict_proba(self, X: pd.DataFrame) -> NDArray[np.float64]:
        row_count = len(X)
        probabilities = self._fitted_probabilities
        if probabilities.shape[0] == 1:
            probabilities = np.repeat(probabilities, row_count, axis=0)
        return probabilities[:row_count]

    def predict(self, X: pd.DataFrame) -> NDArray[np.int64]:
        return self.predict_proba(X).argmax(axis=1).astype(np.int64)


def test_hybrid_ensemble_overrides_uncertain_generalist_with_draw_specialist():
    X = pd.DataFrame({"feature": [1.0, 2.0]})
    y = pd.Series([0, 1], dtype="int64")
    ensemble = HybridDrawOverrideEnsemble(
        generalist_estimator=FixedProbabilityEstimator(
            [[0.38, 0.32, 0.30], [0.70, 0.10, 0.20]]
        ),
        specialist_estimator=FixedProbabilityEstimator(
            [[0.10, 0.75, 0.15], [0.05, 0.80, 0.15]]
        ),
        uncertainty_threshold=0.45,
        draw_conviction_threshold=0.60,
    )

    ensemble.fit(X, y, sample_weight=np.ones(len(X), dtype=np.float64))
    probabilities = ensemble.predict_proba(X)
    predictions = ensemble.predict(X)

    assert np.allclose(probabilities[0], np.array([0.10, 0.75, 0.15]))
    assert predictions.tolist() == [1, 0]


def test_hybrid_ensemble_keeps_generalist_when_confident():
    X = pd.DataFrame({"feature": [1.0]})
    y = pd.Series([0], dtype="int64")
    ensemble = HybridDrawOverrideEnsemble(
        generalist_estimator=FixedProbabilityEstimator([[0.62, 0.21, 0.17]]),
        specialist_estimator=FixedProbabilityEstimator([[0.10, 0.82, 0.08]]),
        uncertainty_threshold=0.45,
        draw_conviction_threshold=0.60,
    )

    ensemble.fit(X, y, sample_weight=np.ones(len(X), dtype=np.float64))
    probabilities = ensemble.predict_proba(X)

    assert np.allclose(probabilities[0], np.array([0.62, 0.21, 0.17]))
    assert ensemble.predict(X).tolist() == [0]


def test_hybrid_ensemble_outputs_normalized_multiclass_probabilities():
    X = pd.DataFrame({"feature": [1.0, 2.0, 3.0]})
    y = pd.Series([0, 1, 2], dtype="int64")
    ensemble = HybridDrawOverrideEnsemble(
        generalist_estimator=FixedProbabilityEstimator([[0.45, 0.28, 0.27]]),
        specialist_estimator=FixedProbabilityEstimator([[0.20, 0.61, 0.19]]),
        uncertainty_threshold=0.50,
        draw_conviction_threshold=0.60,
    )

    ensemble.fit(X, y, sample_weight=np.ones(len(X), dtype=np.float64))
    probabilities = ensemble.predict_proba(X)

    assert probabilities.shape == (3, 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert set(ensemble.predict(X).tolist()).issubset({0, 1, 2})
