"""Segment-aware delegated draw-specialist ensemble for football match outcome prediction.

This module implements a strategic refinement to the base hybrid ensemble:
- Overrides are NO LONGER global (0.45 for all data)
- Instead, they are SEGMENT-CONDITIONAL (different per tournament, match type, etc.)
- Narrower uncertainty bands → more conservative specialist activation
- Segment-specific metrics for validation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

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


@dataclass
class SegmentConfig:
    """Configuration for a specific segment (tournament, match type, etc.)."""

    segment_id: str
    uncertainty_threshold: float
    draw_conviction_threshold: float
    min_samples_for_override: int = 10
    description: str = ""

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not 0.0 <= self.uncertainty_threshold <= 1.0:
            raise ValueError(
                f"[{self.segment_id}] uncertainty_threshold must be in [0,1], "
                f"got {self.uncertainty_threshold}"
            )
        if not 0.0 <= self.draw_conviction_threshold <= 1.0:
            raise ValueError(
                f"[{self.segment_id}] draw_conviction_threshold must be in [0,1], "
                f"got {self.draw_conviction_threshold}"
            )
        if self.min_samples_for_override < 1:
            raise ValueError(f"[{self.segment_id}] min_samples_for_override must be ≥1")


class SegmentAwareHybridDrawOverrideEnsemble(ClassifierMixin, BaseEstimator):
    _estimator_type = "classifier"
    """
    Segment-conditional draw override ensemble.

    Strategy:
    ─────────
    Instead of a single uncertainty threshold (0.45), this ensemble allows
    DIFFERENT thresholds per segment (tournament, match type, equity band).

    Activation logic:
    1. Segment detection: Does the fixture belong to a segment with special config?
    2. If YES: Use segment-specific thresholds (narrower, more conservative)
    3. If NO: Fall back to default thresholds
    4. Override only when ALL conditions are met:
       - generalist uncertainty < segment-specific threshold
       - specialist predicts draw
       - specialist draw confidence > threshold
       - (optional) enough prior coverage to trust the segment

    Benefits:
    ─────────
    ✓ Avoids "firing" the specialist on strong data (e.g., World Cup knockout)
    ✓ Activates selectively where generalist struggles (e.g., friendlies)
    ✓ More conservative → less penalty risk
    ✓ Segment metrics enable A/B testing by tournament

    Parameters
    ──────────
    generalist_estimator : ProbabilisticEstimator
        Primary predictor (trained on full data)
    specialist_estimator : ProbabilisticEstimator
        Draw specialist (draw-weighted training)
    default_uncertainty_threshold : float
        Default threshold when no segment match (typically 0.40-0.50)
    default_draw_conviction_threshold : float
        Default specialist draw conviction threshold
    segment_configs : dict[str, SegmentConfig] | None
        Mapping of segment_id → SegmentConfig. If None, only defaults apply.
    segment_detector_fn : callable | None
        Function(X_row: pd.Series) → str | None detecting segment.
        Returns segment_id if match belongs to segment, else None.
        Example: lambda row: "friendlies" if row.get("tournament") == "Friendly" else None
    specialist_draw_weight_multiplier : float
        Weight multiplier for draw samples in specialist training
    """

    def __init__(
        self,
        generalist_estimator: ProbabilisticEstimator,
        specialist_estimator: ProbabilisticEstimator,
        *,
        default_uncertainty_threshold: float = 0.45,
        default_draw_conviction_threshold: float = 0.55,
        segment_configs: dict[str, SegmentConfig] | None = None,
        segment_detector_fn: Any = None,
        specialist_draw_weight_multiplier: float = 1.0,
    ) -> None:
        self.generalist_estimator = generalist_estimator
        self.specialist_estimator = specialist_estimator
        self.default_uncertainty_threshold = default_uncertainty_threshold
        self.default_draw_conviction_threshold = default_draw_conviction_threshold
        self.segment_configs = segment_configs or {}
        self.segment_detector_fn = segment_detector_fn
        self.specialist_draw_weight_multiplier = specialist_draw_weight_multiplier

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: NDArray[np.float64] | None = None,
    ) -> SegmentAwareHybridDrawOverrideEnsemble:
        """Fit both generalist and specialist models."""
        if not 0.0 <= self.default_uncertainty_threshold <= 1.0:
            raise ValueError("default_uncertainty_threshold must be between 0 and 1.")
        if not 0.0 <= self.default_draw_conviction_threshold <= 1.0:
            raise ValueError(
                "default_draw_conviction_threshold must be between 0 and 1."
            )

        # Validate segment configs if provided
        for seg_id, seg_config in self.segment_configs.items():
            if not isinstance(seg_config, SegmentConfig):
                raise TypeError(
                    f"segment_configs['{seg_id}'] must be SegmentConfig, "
                    f"got {type(seg_config)}"
                )

        # Store feature column names for validation in predict
        self.feature_names_in_ = X.columns.tolist()

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
        self.segment_coverage_ = self._compute_segment_coverage(X)
        return self

    def _build_specialist_weights(
        self,
        y: pd.Series,
        sample_weight: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Upweight draw samples for specialist training."""
        specialist_weight = np.asarray(sample_weight, dtype=np.float64).copy()
        draw_mask = y.to_numpy(dtype=np.int64, copy=False) == DRAW_CLASS
        specialist_weight[draw_mask] *= self.specialist_draw_weight_multiplier
        specialist_weight *= len(specialist_weight) / specialist_weight.sum()
        return specialist_weight

    def _compute_segment_coverage(self, X: pd.DataFrame) -> dict[str, int]:
        """Count samples in each segment."""
        coverage: dict[str, int] = {}
        if self.segment_detector_fn is None or len(self.segment_configs) == 0:
            return coverage

        for idx, row in X.iterrows():
            seg_id = self.segment_detector_fn(row)
            if seg_id and seg_id in self.segment_configs:
                coverage[seg_id] = coverage.get(seg_id, 0) + 1

        return coverage

    def _detect_segment(self, row: pd.Series) -> str | None:
        """Detect segment for a single fixture. Returns segment_id or None."""
        if self.segment_detector_fn is None:
            return None
        result = self.segment_detector_fn(row)
        # Type: detector_fn returns Any, but should be str | None
        return cast(str | None, result)

    def _get_thresholds_for_segment(
        self, segment_id: str | None
    ) -> tuple[float, float]:
        """
        Get uncertainty & conviction thresholds for a fixture's segment.

        Returns
        ───────
        (uncertainty_threshold, draw_conviction_threshold)
        """
        if segment_id and segment_id in self.segment_configs:
            config = self.segment_configs[segment_id]
            return config.uncertainty_threshold, config.draw_conviction_threshold

        return (
            self.default_uncertainty_threshold,
            self.default_draw_conviction_threshold,
        )

    def _compute_override_mask(
        self,
        X: pd.DataFrame,
        generalist_probabilities: NDArray[np.float64],
        specialist_probabilities: NDArray[np.float64],
    ) -> NDArray[np.bool_]:
        """
        Compute which rows to override using segment-aware logic.

        Conditions (ALL must be true for override):
        1. Generalist uncertainty < segment-specific threshold
        2. Specialist predicts draw
        3. Specialist draw probability >= segment-specific conviction threshold
        """
        n_rows = len(X)
        override_mask = np.zeros(n_rows, dtype=bool)

        generalist_confidence = generalist_probabilities.max(axis=1)
        specialist_prediction = specialist_probabilities.argmax(axis=1)
        specialist_draw_probability = specialist_probabilities[:, DRAW_CLASS]

        for i, (idx, row) in enumerate(X.iterrows()):
            segment_id = self._detect_segment(row)
            unc_thresh, conv_thresh = self._get_thresholds_for_segment(segment_id)

            override_mask[i] = (
                (generalist_confidence[i] < unc_thresh)
                and (specialist_prediction[i] == DRAW_CLASS)
                and (specialist_draw_probability[i] >= conv_thresh)
            )

        return override_mask

    def predict_proba(self, X: pd.DataFrame) -> NDArray[np.float64]:
        """Predict class probabilities with segment-aware overrides."""
        # Filter X to only include columns used during fit
        X_fit = X[self.feature_names_in_] if hasattr(self, "feature_names_in_") else X

        generalist_probabilities = predict_proba_aligned(self.generalist_model_, X_fit)
        specialist_probabilities = predict_proba_aligned(self.specialist_model_, X_fit)
        probabilities = np.asarray(generalist_probabilities, dtype=np.float64).copy()

        override_mask = self._compute_override_mask(
            X, generalist_probabilities, specialist_probabilities
        )
        probabilities[override_mask] = specialist_probabilities[override_mask]

        row_sums = probabilities.sum(axis=1, keepdims=True)
        probabilities = probabilities / row_sums
        return cast(NDArray[np.float64], probabilities)

    def predict(self, X: pd.DataFrame) -> NDArray[np.int64]:
        """Predict class labels."""
        probabilities = self.predict_proba(X)
        predictions = self.classes_[probabilities.argmax(axis=1)]
        return cast(NDArray[np.int64], predictions)

    def segment_statistics(
        self,
        X: pd.DataFrame,
        y_true: pd.Series,
        y_pred: NDArray[np.int64] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """
        Compute validation metrics per segment.

        Useful for identifying which segments benefit most from specialist activation.

        Returns
        ───────
        dict mapping segment_id → {
            "n_samples": int,
            "override_rate": float,  # % of fixtures where specialist overrode
            "accuracy": float,
            "draw_accuracy": float,
            "draw_precision": float,
        }
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score

        # Filter X to only trained columns for model predictions
        X_fit = X[self.feature_names_in_] if hasattr(self, "feature_names_in_") else X

        if y_pred is None:
            y_pred = self.predict(X)

        y_true_array = np.asarray(y_true, dtype=np.int64)
        stats: dict[str, dict[str, Any]] = {}

        if self.segment_detector_fn is None or len(self.segment_configs) == 0:
            # No segmentation configured
            return stats

        for segment_id in self.segment_configs:
            segment_mask = np.array(
                [self._detect_segment(row) == segment_id for _, row in X.iterrows()]
            )

            if segment_mask.sum() == 0:
                continue

            y_true_seg = y_true_array[segment_mask]
            y_pred_seg = y_pred[segment_mask]

            # Compute override rate for this segment
            X_fit_seg = X_fit[segment_mask]
            X_seg = X[segment_mask]  # Keep X_seg for _compute_override_mask

            gen_proba = predict_proba_aligned(self.generalist_model_, X_fit_seg)
            spec_proba = predict_proba_aligned(self.specialist_model_, X_fit_seg)
            override_mask_seg = self._compute_override_mask(
                X_seg, gen_proba, spec_proba
            )

            stats[segment_id] = {
                "n_samples": segment_mask.sum(),
                "override_rate": override_mask_seg.mean(),
                "accuracy": accuracy_score(y_true_seg, y_pred_seg),
                "draw_accuracy": recall_score(
                    y_true_seg,
                    y_pred_seg,
                    labels=[DRAW_CLASS],
                    average=None,
                    zero_division=0,
                )[0]
                if len(np.unique(y_true_seg)) > 1
                else 0.0,
                "draw_precision": precision_score(
                    y_true_seg,
                    y_pred_seg,
                    labels=[DRAW_CLASS],
                    average=None,
                    zero_division=0,
                )[0]
                if len(np.unique(y_pred_seg)) > 1
                else 0.0,
            }

        return stats
