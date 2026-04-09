"""Delegated draw-specialist ensemble for football match outcome prediction.

⚠️  LEGACY VERSION - Use SegmentAwareHybridDrawOverrideEnsemble instead
════════════════════════════════════════════════════════════════════════

This module contains the original single-threshold hybrid ensemble.
It is KEPT for backward compatibility only.

✓ RECOMMENDED: Use src.modeling.hybrid_ensemble_segment_aware.SegmentAwareHybridDrawOverrideEnsemble

  Key reason: The original 0.45 global threshold was TOO BROAD.
  It overrides the generalist even on strong, certain predictions (high equity matches),
  leading to net performance loss. The new version is SELECTIVE by segment:

  • Friendlies: Narrower threshold (specialist can help in uncertain context)
  • World Cup: Wider threshold (generalist is strong, avoid interference)
  • Etc.

For migration examples, see tests/test_hybrid_ensemble.py and
the PHASE_3B_SENIOR_FEATURES_SUMMARY.md closure analysis.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClassifierMixin, clone

from src.modeling.evaluation import (
    ProbabilisticEstimator,
    fit_estimator_with_sample_weight,
    predict_proba_aligned,
)

DRAW_CLASS = 1
ENCODED_CLASSES = np.array([0, 1, 2], dtype=np.int64)


class HybridDrawOverrideEnsemble(BaseEstimator, ClassifierMixin):
    """
    Delegate uncertain fixtures to a draw specialist.

    The ensemble keeps the generalist as the primary decision-maker and only
    allows the specialist to override when:
    1. The generalist is uncertain (`max_proba < uncertainty_threshold`)
    2. The specialist predicts draw with sufficient conviction
       (`p_draw >= draw_conviction_threshold`)
    """

    def __init__(
        self,
        generalist_estimator: ProbabilisticEstimator,
        specialist_estimator: ProbabilisticEstimator,
        *,
        uncertainty_threshold: float = 0.45,
        draw_conviction_threshold: float = 0.55,
        specialist_draw_weight_multiplier: float = 1.0,
    ) -> None:
        self.generalist_estimator = generalist_estimator
        self.specialist_estimator = specialist_estimator
        self.uncertainty_threshold = uncertainty_threshold
        self.draw_conviction_threshold = draw_conviction_threshold
        self.specialist_draw_weight_multiplier = specialist_draw_weight_multiplier

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: NDArray[np.float64] | None = None,
    ) -> "HybridDrawOverrideEnsemble":
        if not 0.0 < self.uncertainty_threshold < 1.0:
            raise ValueError("uncertainty_threshold must be between 0 and 1.")
        if not 0.0 < self.draw_conviction_threshold < 1.0:
            raise ValueError("draw_conviction_threshold must be between 0 and 1.")
        if self.specialist_draw_weight_multiplier <= 0:
            raise ValueError("specialist_draw_weight_multiplier must be positive.")

        y_series = pd.Series(y, index=X.index, dtype="int64")
        self.generalist_model_ = clone(self.generalist_estimator)
        self.specialist_model_ = clone(self.specialist_estimator)

        if sample_weight is None:
            self.generalist_model_.fit(X, y_series)
            self.specialist_model_.fit(X, y_series)
        else:
            generalist_weight = np.asarray(sample_weight, dtype=np.float64)
            specialist_weight = self._build_specialist_weights(
                y_series,
                generalist_weight,
            )
            fit_estimator_with_sample_weight(
                self.generalist_model_,
                X,
                y_series,
                generalist_weight,
            )
            fit_estimator_with_sample_weight(
                self.specialist_model_,
                X,
                y_series,
                specialist_weight,
            )

        self.classes_ = ENCODED_CLASSES.copy()
        return self

    def _build_specialist_weights(
        self,
        y: pd.Series,
        sample_weight: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        specialist_weight = np.asarray(sample_weight, dtype=np.float64).copy()
        draw_mask = y.to_numpy(dtype=np.int64, copy=False) == DRAW_CLASS
        specialist_weight[draw_mask] *= self.specialist_draw_weight_multiplier
        specialist_weight *= len(specialist_weight) / specialist_weight.sum()
        return specialist_weight

    def _compute_override_mask(
        self,
        generalist_probabilities: NDArray[np.float64],
        specialist_probabilities: NDArray[np.float64],
    ) -> NDArray[np.bool_]:
        generalist_confidence = generalist_probabilities.max(axis=1)
        specialist_prediction = specialist_probabilities.argmax(axis=1)
        specialist_draw_probability = specialist_probabilities[:, DRAW_CLASS]
        return cast(
            NDArray[np.bool_],
            (generalist_confidence < self.uncertainty_threshold)
            & (specialist_prediction == DRAW_CLASS)
            & (specialist_draw_probability >= self.draw_conviction_threshold),
        )

    def predict_proba(self, X: pd.DataFrame) -> NDArray[np.float64]:
        generalist_probabilities = predict_proba_aligned(self.generalist_model_, X)
        specialist_probabilities = predict_proba_aligned(self.specialist_model_, X)
        probabilities = np.asarray(generalist_probabilities, dtype=np.float64).copy()

        override_mask = self._compute_override_mask(
            generalist_probabilities,
            specialist_probabilities,
        )
        probabilities[override_mask] = specialist_probabilities[override_mask]

        row_sums = probabilities.sum(axis=1, keepdims=True)
        probabilities = probabilities / row_sums
        return cast(NDArray[np.float64], probabilities)

    def predict(self, X: pd.DataFrame) -> NDArray[np.int64]:
        probabilities = self.predict_proba(X)
        predictions = self.classes_[probabilities.argmax(axis=1)]
        return cast(NDArray[np.int64], predictions)
