"""Automated segment-level threshold tuning using out-of-fold predictions."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import clone
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import TimeSeriesSplit

from src.modeling.evaluation import (
    ProbabilisticEstimator,
    fit_estimator_with_sample_weight,
    predict_proba_aligned,
)
from src.modeling.hybrid_ensemble_segment_aware import SegmentConfig

DRAW_CLASS = 1  # -1 -> 0, 0 -> 1, 1 -> 2


def auto_tune_segment_thresholds(
    X: pd.DataFrame,
    y_encoded: pd.Series,
    metadata_df: pd.DataFrame,
    segment_detector_fn: Callable[[pd.Series], str | None],
    generalist_pipeline: ProbabilisticEstimator,
    generalist_sample_weight_fn: Callable[[pd.Series], NDArray[np.float64]],
    specialist_pipeline: ProbabilisticEstimator,
    specialist_sample_weight_fn: Callable[[pd.Series], NDArray[np.float64]],
    n_splits: int = 5,
    max_log_loss_degradation: float = 0.005,
) -> dict[str, SegmentConfig]:
    """
    Tune uncertainty and conviction thresholds per segment using OOF predictions.

    Optimizes for draw F1 score within a log_loss degradation budget.
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting out-of-fold prediction generation for threshold tuning...")

    splitter = TimeSeriesSplit(n_splits=n_splits)

    oof_generalist_probs = []
    oof_specialist_probs = []
    oof_y_true = []
    oof_segments = []

    X_reset = X.reset_index(drop=True)
    y_reset = y_encoded.reset_index(drop=True)
    meta_reset = metadata_df.reset_index(drop=True)

    for fold_idx, (train_idx, valid_idx) in enumerate(splitter.split(X_reset), start=1):
        X_train = X_reset.iloc[train_idx]
        y_train = y_reset.iloc[train_idx]

        X_valid = X_reset.iloc[valid_idx]
        y_valid = y_reset.iloc[valid_idx]

        gen_estimator = cast(ProbabilisticEstimator, clone(generalist_pipeline))
        gen_weights = generalist_sample_weight_fn(y_train)
        fit_estimator_with_sample_weight(gen_estimator, X_train, y_train, gen_weights)

        spec_estimator = cast(ProbabilisticEstimator, clone(specialist_pipeline))
        spec_weights = specialist_sample_weight_fn(y_train)
        fit_estimator_with_sample_weight(spec_estimator, X_train, y_train, spec_weights)

        gen_probs = predict_proba_aligned(gen_estimator, X_valid)
        spec_probs = predict_proba_aligned(spec_estimator, X_valid)

        meta_valid = meta_reset.iloc[valid_idx]
        segments = [segment_detector_fn(row) for _, row in meta_valid.iterrows()]

        oof_generalist_probs.append(gen_probs)
        oof_specialist_probs.append(spec_probs)
        oof_y_true.append(y_valid.to_numpy(dtype=np.int64))
        oof_segments.extend(segments)

    full_gen_probs = np.vstack(oof_generalist_probs)
    full_spec_probs = np.vstack(oof_specialist_probs)
    full_y_true = np.concatenate(oof_y_true)
    full_segments = np.array(oof_segments)

    logger.info(f"Generated OOF probabilities for {len(full_y_true)} rows.")

    # Vectorized Grid Search per Segment
    unc_grid = np.arange(0.30, 0.65, 0.02)
    conv_grid = np.arange(0.40, 0.65, 0.02)

    unique_segments = {s for s in full_segments if s is not None}
    tuned_configs: dict[str, SegmentConfig] = {}

    for segment in sorted(unique_segments):
        mask = full_segments == segment
        if not np.any(mask):
            continue

        seg_y_true = full_y_true[mask]
        seg_gen_probs = full_gen_probs[mask]
        seg_spec_probs = full_spec_probs[mask]

        baseline_pred = seg_gen_probs.argmax(axis=1)
        baseline_draw_f1 = float(
            f1_score(
                seg_y_true,
                baseline_pred,
                labels=[DRAW_CLASS],
                average="macro",
                zero_division=0,
            )
        )
        baseline_log_loss = float(log_loss(seg_y_true, seg_gen_probs, labels=[0, 1, 2]))

        best_unc = 0.0
        best_conv = 1.0
        best_draw_f1 = baseline_draw_f1

        gen_confidence = seg_gen_probs.max(axis=1)
        spec_pred = seg_spec_probs.argmax(axis=1)
        spec_draw_prob = seg_spec_probs[:, DRAW_CLASS]

        for unc in unc_grid:
            for conv in conv_grid:
                override_mask = (
                    (gen_confidence < unc)
                    & (spec_pred == DRAW_CLASS)
                    & (spec_draw_prob >= conv)
                )

                if not np.any(override_mask):
                    continue

                blended_probs = seg_gen_probs.copy()
                blended_probs[override_mask] = seg_spec_probs[override_mask]

                # Safe row normalization in case specialist predicted 0s for a class
                row_sums = blended_probs.sum(axis=1, keepdims=True)
                # Fallback to avoid div zero by ignoring 0 sums
                nonzero_rows = row_sums.squeeze(axis=1) > 0
                if np.any(nonzero_rows):
                    blended_probs[nonzero_rows] = (
                        blended_probs[nonzero_rows] / row_sums[nonzero_rows]
                    )

                blended_pred = blended_probs.argmax(axis=1)

                draw_f1 = float(
                    f1_score(
                        seg_y_true,
                        blended_pred,
                        labels=[DRAW_CLASS],
                        average="macro",
                        zero_division=0,
                    )
                )

                if draw_f1 > best_draw_f1:
                    ll = float(log_loss(seg_y_true, blended_probs, labels=[0, 1, 2]))
                    if ll <= baseline_log_loss + max_log_loss_degradation:
                        best_draw_f1 = draw_f1
                        best_unc = unc
                        best_conv = conv

        if best_draw_f1 > baseline_draw_f1:
            logger.info(
                f"Segment '{segment}': Tuned to unc={best_unc:.2f}, conv={best_conv:.2f} "
                f"(Draw F1: {baseline_draw_f1:.4f} -> {best_draw_f1:.4f})"
            )
            tuned_configs[str(segment)] = SegmentConfig(
                segment_id=str(segment),
                uncertainty_threshold=float(best_unc),
                draw_conviction_threshold=float(best_conv),
            )
        else:
            logger.info(
                f"Segment '{segment}': No threshold improved Draw F1 within log loss budget. Falling back to generalist."
            )
            # Default thresholds where specialist won't trigger easily
            tuned_configs[str(segment)] = SegmentConfig(
                segment_id=str(segment),
                uncertainty_threshold=0.01,
                draw_conviction_threshold=0.99,
            )

    return tuned_configs
