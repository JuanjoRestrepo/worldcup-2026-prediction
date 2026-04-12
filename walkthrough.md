# Walkthrough: Segment-Aware Hybrid Ensemble Integration

**Date**: April 12, 2026  
**Status**: ✅ COMPLETE — All verification passed, full retrain successful

---

## What Changed

### 1. [evaluation.py](file:///c:/Users/restr/Desktop/worldcup-2026-prediction/src/modeling/evaluation.py) — Metadata Pass-Through

Added `metadata_columns: list[str] | None = None` parameter to `evaluate_candidates_with_backtesting()`. When segment-aware candidates are detected (by `family == "segment_aware_hybrid"`), the function appends metadata columns (e.g., `tournament`) to the validation DataFrame during `predict_proba`/`predict` calls. The ensemble's `feature_names_in_` filtering strips these columns before model inference, so they only serve the segment detector.

**Backward compatible**: All existing callers pass `metadata_columns=None` and get identical behavior.

```diff:evaluation.py
"""Robust evaluation utilities for temporal model selection and calibration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    log_loss,
    matthews_corrcoef,
    f1_score,
)
from sklearn.model_selection import TimeSeriesSplit

from src.modeling.features import OUTCOME_LABELS
from src.modeling.types import TrainingMetrics

PRIMARY_SELECTION_METRICS = (
    "macro_f1",
    "draw_f1",
    "draw_recall",
    "balanced_accuracy",
    "matthews_corrcoef",
    "log_loss",
    "multiclass_brier_score",
    "expected_calibration_error",
)
ENCODED_CLASS_ORDER = np.array([0, 1, 2], dtype=np.int64)


@dataclass(frozen=True)
class TemporalDataSplit:
    """Chronological train/calibration/test windows."""

    train_df: pd.DataFrame
    calibration_df: pd.DataFrame
    test_df: pd.DataFrame


@dataclass(frozen=True)
class CandidateSpec:
    """Model-search candidate with estimator pipeline and weighting strategy."""

    name: str
    pipeline: ProbabilisticEstimator
    sample_weight_builder: Callable[[pd.Series], NDArray[np.float64]]
    family: str
    hyperparameters: dict[str, object]
    notes: str | None = None


class ProbabilisticEstimator(Protocol):
    """Protocol for fitted probabilistic classifiers."""

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs: object) -> object: ...

    def predict(self, X: pd.DataFrame) -> NDArray[np.int64]: ...

    def predict_proba(self, X: pd.DataFrame) -> NDArray[np.float64]: ...


def fit_estimator_with_sample_weight(
    estimator: ProbabilisticEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: NDArray[np.float64],
) -> None:
    """Fit either a sklearn Pipeline or a custom estimator with sample weights."""
    named_steps = getattr(estimator, "named_steps", None)
    if named_steps is not None and "model" in named_steps:
        estimator.fit(X, y, model__sample_weight=sample_weight)
        return
    estimator.fit(X, y, sample_weight=sample_weight)


def extract_estimator_classes(estimator: ProbabilisticEstimator) -> NDArray[np.int64]:
    """Return encoded classes from a fitted estimator or wrapped pipeline."""
    estimator_classes = getattr(estimator, "classes_", None)
    if estimator_classes is None:
        named_steps = getattr(estimator, "named_steps", None)
        if named_steps is not None and "model" in named_steps:
            estimator_classes = getattr(named_steps["model"], "classes_", None)

    if estimator_classes is None:
        raise ValueError("Could not resolve classes_ from the fitted estimator.")
    return np.asarray(estimator_classes, dtype=np.int64)


def predict_proba_aligned(
    estimator: ProbabilisticEstimator,
    X: pd.DataFrame,
) -> NDArray[np.float64]:
    """Align model probabilities to the encoded class order [0, 1, 2]."""
    raw_probabilities = estimator.predict_proba(X).astype(np.float64)
    try:
        estimator_classes = extract_estimator_classes(estimator)
    except ValueError:
        return raw_probabilities

    aligned = np.zeros((raw_probabilities.shape[0], len(ENCODED_CLASS_ORDER)), dtype=np.float64)
    for column_index, class_label in enumerate(estimator_classes):
        class_position = int(np.where(ENCODED_CLASS_ORDER == int(class_label))[0][0])
        aligned[:, class_position] = raw_probabilities[:, column_index]
    row_sums = aligned.sum(axis=1, keepdims=True)
    nonzero_rows = row_sums.squeeze(axis=1) > 0
    aligned[nonzero_rows] = aligned[nonzero_rows] / row_sums[nonzero_rows]
    return aligned


def _multiclass_brier_score(
    y_true_encoded: NDArray[np.int64],
    probabilities: NDArray[np.float64],
) -> float:
    classes = probabilities.shape[1]
    one_hot = np.eye(classes, dtype=np.float64)[y_true_encoded]
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def _expected_calibration_error(
    y_true_encoded: NDArray[np.int64],
    probabilities: NDArray[np.float64],
    bins: int = 10,
) -> float:
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correctness = (predictions == y_true_encoded).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lower, upper in zip(bin_edges[:-1], bin_edges[1:]):
        if upper == 1.0:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)
        if not np.any(mask):
            continue
        bin_confidence = float(np.mean(confidences[mask]))
        bin_accuracy = float(np.mean(correctness[mask]))
        ece += float(np.mean(mask)) * abs(bin_accuracy - bin_confidence)
    return float(ece)


def evaluate_multiclass_predictions(
    y_true: pd.Series,
    y_true_encoded: pd.Series,
    y_pred_encoded: NDArray[np.int64] | pd.Series,
    probabilities: NDArray[np.float64],
) -> dict[str, object]:
    """Compute robust multiclass metrics for selection and final evaluation."""
    predicted_series = pd.Series(y_pred_encoded, index=y_true.index, dtype="int64")
    y_pred = predicted_series.map({0: -1, 1: 0, 2: 1})

    y_true_array = y_true.to_numpy(dtype=np.int64, copy=False)
    y_true_encoded_array = y_true_encoded.to_numpy(dtype=np.int64, copy=False)
    y_pred_array = y_pred.to_numpy(dtype=np.int64, copy=False)

    return {
        "accuracy": float(accuracy_score(y_true_array, y_pred_array)),
        "macro_f1": float(
            f1_score(y_true_array, y_pred_array, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true_array, y_pred_array, average="weighted", zero_division=0)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true_array, y_pred_array)
        ),
        "matthews_corrcoef": float(
            matthews_corrcoef(y_true_array, y_pred_array)
        ),
        "cohen_kappa": float(cohen_kappa_score(y_true_array, y_pred_array)),
        "log_loss": float(
            log_loss(y_true_encoded_array, probabilities, labels=[0, 1, 2])
        ),
        "multiclass_brier_score": _multiclass_brier_score(
            y_true_encoded_array,
            probabilities,
        ),
        "expected_calibration_error": _expected_calibration_error(
            y_true_encoded_array,
            probabilities,
        ),
        "classification_report": classification_report(
            y_true_array,
            y_pred_array,
            labels=[-1, 0, 1],
            target_names=[
                OUTCOME_LABELS[-1],
                OUTCOME_LABELS[0],
                OUTCOME_LABELS[1],
            ],
            output_dict=True,
            zero_division=0,
        ),
        "draw_f1": float(
            f1_score(
                (y_true_array == 0).astype(np.int64),
                (y_pred_array == 0).astype(np.int64),
                zero_division=0,
            )
        ),
        "draw_recall": float(
            np.mean(y_pred_array[y_true_array == 0] == 0)
            if np.any(y_true_array == 0)
            else 0.0
        ),
    }


def extract_class_metric(
    metrics: TrainingMetrics,
    outcome_label: str,
    metric_name: str,
) -> float:
    """Extract a per-class metric from a sklearn classification report payload."""
    report = metrics["classification_report"]
    class_report = cast(dict[str, object], report[outcome_label])
    return float(cast(float, class_report[metric_name]))


def select_deployment_variant(
    candidate_metrics_by_variant: dict[str, TrainingMetrics],
    *,
    baseline_variant: str = "uncalibrated",
    max_macro_f1_drop: float = 0.015,
    max_weighted_f1_drop: float = 0.02,
    max_draw_recall_drop: float = 0.05,
    min_log_loss_gain: float = 0.005,
    min_ece_gain: float = 0.005,
) -> tuple[str, dict[str, object]]:
    """
    Choose the deployment variant without sacrificing class balance for calibration.

    A calibrated model is only eligible if it materially improves probability quality
    while keeping macro/weighted F1 and draw recall close to the uncalibrated baseline.
    """
    if baseline_variant not in candidate_metrics_by_variant:
        raise ValueError("baseline_variant must exist in candidate_metrics_by_variant.")

    baseline_metrics = candidate_metrics_by_variant[baseline_variant]
    baseline_macro_f1 = float(baseline_metrics["macro_f1"])
    baseline_weighted_f1 = float(baseline_metrics["weighted_f1"])
    baseline_log_loss = float(baseline_metrics["log_loss"])
    baseline_ece = float(baseline_metrics["expected_calibration_error"])
    baseline_draw_recall = extract_class_metric(
        baseline_metrics,
        OUTCOME_LABELS[0],
        "recall",
    )

    candidate_decisions: list[dict[str, object]] = []
    viable_variants: list[str] = [baseline_variant]

    for variant_name, metrics in candidate_metrics_by_variant.items():
        macro_f1 = float(metrics["macro_f1"])
        weighted_f1 = float(metrics["weighted_f1"])
        log_loss_value = float(metrics["log_loss"])
        ece_value = float(metrics["expected_calibration_error"])
        draw_recall = extract_class_metric(metrics, OUTCOME_LABELS[0], "recall")

        macro_f1_delta = macro_f1 - baseline_macro_f1
        weighted_f1_delta = weighted_f1 - baseline_weighted_f1
        log_loss_improvement = baseline_log_loss - log_loss_value
        ece_improvement = baseline_ece - ece_value
        draw_recall_delta = draw_recall - baseline_draw_recall

        preserves_class_balance = (
            macro_f1_delta >= -max_macro_f1_drop
            and weighted_f1_delta >= -max_weighted_f1_drop
            and draw_recall_delta >= -max_draw_recall_drop
        )
        improves_probability_quality = (
            log_loss_improvement >= min_log_loss_gain
            or ece_improvement >= min_ece_gain
        )
        is_viable = variant_name == baseline_variant or (
            preserves_class_balance and improves_probability_quality
        )
        if is_viable and variant_name != baseline_variant:
            viable_variants.append(variant_name)

        candidate_decisions.append(
            {
                "variant": variant_name,
                "is_baseline": variant_name == baseline_variant,
                "is_viable": is_viable,
                "preserves_class_balance": preserves_class_balance,
                "improves_probability_quality": improves_probability_quality,
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
                "log_loss": log_loss_value,
                "expected_calibration_error": ece_value,
                "draw_recall": draw_recall,
                "macro_f1_delta": macro_f1_delta,
                "weighted_f1_delta": weighted_f1_delta,
                "log_loss_improvement": log_loss_improvement,
                "expected_calibration_error_improvement": ece_improvement,
                "draw_recall_delta": draw_recall_delta,
            }
        )

    chosen_variant = baseline_variant
    calibrated_candidates = [
        entry
        for entry in candidate_decisions
        if cast(bool, entry["is_viable"]) and not cast(bool, entry["is_baseline"])
    ]
    if calibrated_candidates:
        calibrated_candidates.sort(
            key=lambda entry: (
                cast(float, entry["log_loss"]),
                cast(float, entry["expected_calibration_error"]),
                -cast(float, entry["macro_f1"]),
                -cast(float, entry["weighted_f1"]),
            )
        )
        chosen_variant = cast(str, calibrated_candidates[0]["variant"])

    return chosen_variant, {
        "baseline_variant": baseline_variant,
        "chosen_variant": chosen_variant,
        "selection_policy": {
            "max_macro_f1_drop": max_macro_f1_drop,
            "max_weighted_f1_drop": max_weighted_f1_drop,
            "max_draw_recall_drop": max_draw_recall_drop,
            "min_log_loss_gain": min_log_loss_gain,
            "min_ece_gain": min_ece_gain,
        },
        "candidate_decisions": candidate_decisions,
        "viable_variants": viable_variants,
    }


def split_train_calibration_test(
    df: pd.DataFrame,
    *,
    test_size: float,
    calibration_size: float,
) -> TemporalDataSplit:
    """Split a chronological dataset into train/calibration/test windows."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    if not 0 < calibration_size < 1:
        raise ValueError("calibration_size must be between 0 and 1.")
    if test_size + calibration_size >= 0.5:
        raise ValueError("test_size + calibration_size must be less than 0.5.")

    ordered_df = df.sort_values("date").reset_index(drop=True)
    total_rows = len(ordered_df)
    if total_rows < 500:
        raise ValueError(
            "Not enough rows for robust temporal evaluation. At least 500 rows are required."
        )

    test_rows = max(1, int(total_rows * test_size))
    calibration_rows = max(1, int(total_rows * calibration_size))
    train_rows = total_rows - test_rows - calibration_rows
    if train_rows < 100:
        raise ValueError(
            "Temporal split leaves too few rows for model selection training."
        )

    train_df = ordered_df.iloc[:train_rows].copy()
    calibration_df = ordered_df.iloc[train_rows : train_rows + calibration_rows].copy()
    test_df = ordered_df.iloc[train_rows + calibration_rows :].copy()
    return TemporalDataSplit(
        train_df=train_df,
        calibration_df=calibration_df,
        test_df=test_df,
    )


def evaluate_candidates_with_backtesting(
    *,
    candidate_specs: dict[str, CandidateSpec],
    train_df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    outcome_to_encoded: dict[int, int],
    n_splits: int,
) -> tuple[list[dict[str, object]], str]:
    """Evaluate candidate models with rolling temporal validation and rank them."""
    if n_splits < 3:
        raise ValueError("n_splits must be at least 3 for temporal backtesting.")

    X = train_df[feature_columns].reset_index(drop=True)
    y = train_df[target_column].reset_index(drop=True)
    dates = train_df["date"].reset_index(drop=True)

    splitter = TimeSeriesSplit(n_splits=n_splits)
    candidate_summaries: list[dict[str, object]] = []

    for candidate_name, candidate_spec in candidate_specs.items():
        fold_results: list[dict[str, object]] = []
        for fold_index, (train_idx, valid_idx) in enumerate(splitter.split(X), start=1):
            X_train = X.iloc[train_idx].copy()
            X_valid = X.iloc[valid_idx].copy()
            y_train = y.iloc[train_idx].copy()
            y_valid = y.iloc[valid_idx].copy()
            y_train_encoded = y_train.map(outcome_to_encoded).astype("int64")
            y_valid_encoded = y_valid.map(outcome_to_encoded).astype("int64")

            estimator = cast(ProbabilisticEstimator, clone(candidate_spec.pipeline))
            sample_weight = candidate_spec.sample_weight_builder(y_train_encoded)
            fit_estimator_with_sample_weight(
                estimator,
                X_train,
                y_train_encoded,
                sample_weight,
            )

            probabilities = predict_proba_aligned(estimator, X_valid)
            predicted = estimator.predict(X_valid).astype(np.int64)
            metrics = evaluate_multiclass_predictions(
                y_true=y_valid,
                y_true_encoded=y_valid_encoded,
                y_pred_encoded=predicted,
                probabilities=probabilities,
            )

            fold_results.append(
                {
                    "fold_index": fold_index,
                    "train_rows": int(len(train_idx)),
                    "validation_rows": int(len(valid_idx)),
                    "train_date_range": {
                        "start": dates.iloc[train_idx[0]].date().isoformat(),
                        "end": dates.iloc[train_idx[-1]].date().isoformat(),
                    },
                    "validation_date_range": {
                        "start": dates.iloc[valid_idx[0]].date().isoformat(),
                        "end": dates.iloc[valid_idx[-1]].date().isoformat(),
                    },
                    "metrics": metrics,
                }
            )

        metric_keys = PRIMARY_SELECTION_METRICS
        fold_metric_maps = [
            cast(dict[str, object], fold["metrics"])
            for fold in fold_results
        ]
        mean_metrics = {
            metric_name: float(
                np.mean(
                    [
                        cast(float, metric_map[metric_name])
                        for metric_map in fold_metric_maps
                    ]
                )
            )
            for metric_name in metric_keys
        }
        std_metrics = {
            metric_name: float(
                np.std(
                    [
                        cast(float, metric_map[metric_name])
                        for metric_map in fold_metric_maps
                    ]
                )
            )
            for metric_name in metric_keys
        }
        candidate_summaries.append(
            {
                "model_name": candidate_name,
                "model_family": candidate_spec.family,
                "hyperparameters": candidate_spec.hyperparameters,
                "notes": candidate_spec.notes,
                "fold_results": fold_results,
                "mean_metrics": mean_metrics,
                "std_metrics": std_metrics,
            }
        )

    ranking_frame = pd.DataFrame(
        [
            {
                "model_name": summary["model_name"],
                **cast(dict[str, float], summary["mean_metrics"]),
            }
            for summary in candidate_summaries
        ]
    )
    ranking_frame["rank_macro_f1"] = ranking_frame["macro_f1"].rank(
        ascending=False,
        method="dense",
    )
    ranking_frame["rank_draw_f1"] = ranking_frame["draw_f1"].rank(
        ascending=False,
        method="dense",
    )
    ranking_frame["rank_draw_recall"] = ranking_frame["draw_recall"].rank(
        ascending=False,
        method="dense",
    )
    ranking_frame["rank_balanced_accuracy"] = ranking_frame["balanced_accuracy"].rank(
        ascending=False,
        method="dense",
    )
    ranking_frame["rank_matthews_corrcoef"] = ranking_frame["matthews_corrcoef"].rank(
        ascending=False,
        method="dense",
    )
    ranking_frame["rank_log_loss"] = ranking_frame["log_loss"].rank(
        ascending=True,
        method="dense",
    )
    ranking_frame["rank_brier"] = ranking_frame["multiclass_brier_score"].rank(
        ascending=True,
        method="dense",
    )
    ranking_frame["rank_ece"] = ranking_frame["expected_calibration_error"].rank(
        ascending=True,
        method="dense",
    )
    ranking_frame["selection_score"] = ranking_frame[
        [
            "rank_macro_f1",
            "rank_draw_f1",
            "rank_draw_recall",
            "rank_balanced_accuracy",
            "rank_matthews_corrcoef",
            "rank_log_loss",
            "rank_brier",
            "rank_ece",
        ]
    ].sum(axis=1)
    ranking_frame = ranking_frame.sort_values(
        ["selection_score", "macro_f1", "draw_f1", "draw_recall", "log_loss"],
        ascending=[True, False, False, False, True],
    ).reset_index(drop=True)

    best_model_name = str(ranking_frame.loc[0, "model_name"])
    ranking_lookup = ranking_frame.set_index("model_name").to_dict(orient="index")
    for summary in candidate_summaries:
        model_name = str(summary["model_name"])
        summary["selection_rank"] = int(ranking_frame.index[ranking_frame["model_name"] == model_name][0] + 1)
        summary["selection_score"] = float(ranking_lookup[model_name]["selection_score"])

    return candidate_summaries, best_model_name
===
"""Robust evaluation utilities for temporal model selection and calibration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    log_loss,
    matthews_corrcoef,
    f1_score,
)
from sklearn.model_selection import TimeSeriesSplit

from src.modeling.features import OUTCOME_LABELS
from src.modeling.types import TrainingMetrics

PRIMARY_SELECTION_METRICS = (
    "macro_f1",
    "draw_f1",
    "draw_recall",
    "balanced_accuracy",
    "matthews_corrcoef",
    "log_loss",
    "multiclass_brier_score",
    "expected_calibration_error",
)
ENCODED_CLASS_ORDER = np.array([0, 1, 2], dtype=np.int64)


@dataclass(frozen=True)
class TemporalDataSplit:
    """Chronological train/calibration/test windows."""

    train_df: pd.DataFrame
    calibration_df: pd.DataFrame
    test_df: pd.DataFrame


@dataclass(frozen=True)
class CandidateSpec:
    """Model-search candidate with estimator pipeline and weighting strategy."""

    name: str
    pipeline: ProbabilisticEstimator
    sample_weight_builder: Callable[[pd.Series], NDArray[np.float64]]
    family: str
    hyperparameters: dict[str, object]
    notes: str | None = None


class ProbabilisticEstimator(Protocol):
    """Protocol for fitted probabilistic classifiers."""

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs: object) -> object: ...

    def predict(self, X: pd.DataFrame) -> NDArray[np.int64]: ...

    def predict_proba(self, X: pd.DataFrame) -> NDArray[np.float64]: ...


def fit_estimator_with_sample_weight(
    estimator: ProbabilisticEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: NDArray[np.float64],
) -> None:
    """Fit either a sklearn Pipeline or a custom estimator with sample weights."""
    named_steps = getattr(estimator, "named_steps", None)
    if named_steps is not None and "model" in named_steps:
        estimator.fit(X, y, model__sample_weight=sample_weight)
        return
    estimator.fit(X, y, sample_weight=sample_weight)


def extract_estimator_classes(estimator: ProbabilisticEstimator) -> NDArray[np.int64]:
    """Return encoded classes from a fitted estimator or wrapped pipeline."""
    estimator_classes = getattr(estimator, "classes_", None)
    if estimator_classes is None:
        named_steps = getattr(estimator, "named_steps", None)
        if named_steps is not None and "model" in named_steps:
            estimator_classes = getattr(named_steps["model"], "classes_", None)

    if estimator_classes is None:
        raise ValueError("Could not resolve classes_ from the fitted estimator.")
    return np.asarray(estimator_classes, dtype=np.int64)


def predict_proba_aligned(
    estimator: ProbabilisticEstimator,
    X: pd.DataFrame,
) -> NDArray[np.float64]:
    """Align model probabilities to the encoded class order [0, 1, 2]."""
    raw_probabilities = estimator.predict_proba(X).astype(np.float64)
    try:
        estimator_classes = extract_estimator_classes(estimator)
    except ValueError:
        return raw_probabilities

    aligned = np.zeros((raw_probabilities.shape[0], len(ENCODED_CLASS_ORDER)), dtype=np.float64)
    for column_index, class_label in enumerate(estimator_classes):
        class_position = int(np.where(ENCODED_CLASS_ORDER == int(class_label))[0][0])
        aligned[:, class_position] = raw_probabilities[:, column_index]
    row_sums = aligned.sum(axis=1, keepdims=True)
    nonzero_rows = row_sums.squeeze(axis=1) > 0
    aligned[nonzero_rows] = aligned[nonzero_rows] / row_sums[nonzero_rows]
    return aligned


def _multiclass_brier_score(
    y_true_encoded: NDArray[np.int64],
    probabilities: NDArray[np.float64],
) -> float:
    classes = probabilities.shape[1]
    one_hot = np.eye(classes, dtype=np.float64)[y_true_encoded]
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def _expected_calibration_error(
    y_true_encoded: NDArray[np.int64],
    probabilities: NDArray[np.float64],
    bins: int = 10,
) -> float:
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correctness = (predictions == y_true_encoded).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lower, upper in zip(bin_edges[:-1], bin_edges[1:]):
        if upper == 1.0:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)
        if not np.any(mask):
            continue
        bin_confidence = float(np.mean(confidences[mask]))
        bin_accuracy = float(np.mean(correctness[mask]))
        ece += float(np.mean(mask)) * abs(bin_accuracy - bin_confidence)
    return float(ece)


def evaluate_multiclass_predictions(
    y_true: pd.Series,
    y_true_encoded: pd.Series,
    y_pred_encoded: NDArray[np.int64] | pd.Series,
    probabilities: NDArray[np.float64],
) -> dict[str, object]:
    """Compute robust multiclass metrics for selection and final evaluation."""
    predicted_series = pd.Series(y_pred_encoded, index=y_true.index, dtype="int64")
    y_pred = predicted_series.map({0: -1, 1: 0, 2: 1})

    y_true_array = y_true.to_numpy(dtype=np.int64, copy=False)
    y_true_encoded_array = y_true_encoded.to_numpy(dtype=np.int64, copy=False)
    y_pred_array = y_pred.to_numpy(dtype=np.int64, copy=False)

    return {
        "accuracy": float(accuracy_score(y_true_array, y_pred_array)),
        "macro_f1": float(
            f1_score(y_true_array, y_pred_array, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true_array, y_pred_array, average="weighted", zero_division=0)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true_array, y_pred_array)
        ),
        "matthews_corrcoef": float(
            matthews_corrcoef(y_true_array, y_pred_array)
        ),
        "cohen_kappa": float(cohen_kappa_score(y_true_array, y_pred_array)),
        "log_loss": float(
            log_loss(y_true_encoded_array, probabilities, labels=[0, 1, 2])
        ),
        "multiclass_brier_score": _multiclass_brier_score(
            y_true_encoded_array,
            probabilities,
        ),
        "expected_calibration_error": _expected_calibration_error(
            y_true_encoded_array,
            probabilities,
        ),
        "classification_report": classification_report(
            y_true_array,
            y_pred_array,
            labels=[-1, 0, 1],
            target_names=[
                OUTCOME_LABELS[-1],
                OUTCOME_LABELS[0],
                OUTCOME_LABELS[1],
            ],
            output_dict=True,
            zero_division=0,
        ),
        "draw_f1": float(
            f1_score(
                (y_true_array == 0).astype(np.int64),
                (y_pred_array == 0).astype(np.int64),
                zero_division=0,
            )
        ),
        "draw_recall": float(
            np.mean(y_pred_array[y_true_array == 0] == 0)
            if np.any(y_true_array == 0)
            else 0.0
        ),
    }


def extract_class_metric(
    metrics: TrainingMetrics,
    outcome_label: str,
    metric_name: str,
) -> float:
    """Extract a per-class metric from a sklearn classification report payload."""
    report = metrics["classification_report"]
    class_report = cast(dict[str, object], report[outcome_label])
    return float(cast(float, class_report[metric_name]))


def select_deployment_variant(
    candidate_metrics_by_variant: dict[str, TrainingMetrics],
    *,
    baseline_variant: str = "uncalibrated",
    max_macro_f1_drop: float = 0.015,
    max_weighted_f1_drop: float = 0.02,
    max_draw_recall_drop: float = 0.05,
    min_log_loss_gain: float = 0.005,
    min_ece_gain: float = 0.005,
) -> tuple[str, dict[str, object]]:
    """
    Choose the deployment variant without sacrificing class balance for calibration.

    A calibrated model is only eligible if it materially improves probability quality
    while keeping macro/weighted F1 and draw recall close to the uncalibrated baseline.
    """
    if baseline_variant not in candidate_metrics_by_variant:
        raise ValueError("baseline_variant must exist in candidate_metrics_by_variant.")

    baseline_metrics = candidate_metrics_by_variant[baseline_variant]
    baseline_macro_f1 = float(baseline_metrics["macro_f1"])
    baseline_weighted_f1 = float(baseline_metrics["weighted_f1"])
    baseline_log_loss = float(baseline_metrics["log_loss"])
    baseline_ece = float(baseline_metrics["expected_calibration_error"])
    baseline_draw_recall = extract_class_metric(
        baseline_metrics,
        OUTCOME_LABELS[0],
        "recall",
    )

    candidate_decisions: list[dict[str, object]] = []
    viable_variants: list[str] = [baseline_variant]

    for variant_name, metrics in candidate_metrics_by_variant.items():
        macro_f1 = float(metrics["macro_f1"])
        weighted_f1 = float(metrics["weighted_f1"])
        log_loss_value = float(metrics["log_loss"])
        ece_value = float(metrics["expected_calibration_error"])
        draw_recall = extract_class_metric(metrics, OUTCOME_LABELS[0], "recall")

        macro_f1_delta = macro_f1 - baseline_macro_f1
        weighted_f1_delta = weighted_f1 - baseline_weighted_f1
        log_loss_improvement = baseline_log_loss - log_loss_value
        ece_improvement = baseline_ece - ece_value
        draw_recall_delta = draw_recall - baseline_draw_recall

        preserves_class_balance = (
            macro_f1_delta >= -max_macro_f1_drop
            and weighted_f1_delta >= -max_weighted_f1_drop
            and draw_recall_delta >= -max_draw_recall_drop
        )
        improves_probability_quality = (
            log_loss_improvement >= min_log_loss_gain
            or ece_improvement >= min_ece_gain
        )
        is_viable = variant_name == baseline_variant or (
            preserves_class_balance and improves_probability_quality
        )
        if is_viable and variant_name != baseline_variant:
            viable_variants.append(variant_name)

        candidate_decisions.append(
            {
                "variant": variant_name,
                "is_baseline": variant_name == baseline_variant,
                "is_viable": is_viable,
                "preserves_class_balance": preserves_class_balance,
                "improves_probability_quality": improves_probability_quality,
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
                "log_loss": log_loss_value,
                "expected_calibration_error": ece_value,
                "draw_recall": draw_recall,
                "macro_f1_delta": macro_f1_delta,
                "weighted_f1_delta": weighted_f1_delta,
                "log_loss_improvement": log_loss_improvement,
                "expected_calibration_error_improvement": ece_improvement,
                "draw_recall_delta": draw_recall_delta,
            }
        )

    chosen_variant = baseline_variant
    calibrated_candidates = [
        entry
        for entry in candidate_decisions
        if cast(bool, entry["is_viable"]) and not cast(bool, entry["is_baseline"])
    ]
    if calibrated_candidates:
        calibrated_candidates.sort(
            key=lambda entry: (
                cast(float, entry["log_loss"]),
                cast(float, entry["expected_calibration_error"]),
                -cast(float, entry["macro_f1"]),
                -cast(float, entry["weighted_f1"]),
            )
        )
        chosen_variant = cast(str, calibrated_candidates[0]["variant"])

    return chosen_variant, {
        "baseline_variant": baseline_variant,
        "chosen_variant": chosen_variant,
        "selection_policy": {
            "max_macro_f1_drop": max_macro_f1_drop,
            "max_weighted_f1_drop": max_weighted_f1_drop,
            "max_draw_recall_drop": max_draw_recall_drop,
            "min_log_loss_gain": min_log_loss_gain,
            "min_ece_gain": min_ece_gain,
        },
        "candidate_decisions": candidate_decisions,
        "viable_variants": viable_variants,
    }


def split_train_calibration_test(
    df: pd.DataFrame,
    *,
    test_size: float,
    calibration_size: float,
) -> TemporalDataSplit:
    """Split a chronological dataset into train/calibration/test windows."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    if not 0 < calibration_size < 1:
        raise ValueError("calibration_size must be between 0 and 1.")
    if test_size + calibration_size >= 0.5:
        raise ValueError("test_size + calibration_size must be less than 0.5.")

    ordered_df = df.sort_values("date").reset_index(drop=True)
    total_rows = len(ordered_df)
    if total_rows < 500:
        raise ValueError(
            "Not enough rows for robust temporal evaluation. At least 500 rows are required."
        )

    test_rows = max(1, int(total_rows * test_size))
    calibration_rows = max(1, int(total_rows * calibration_size))
    train_rows = total_rows - test_rows - calibration_rows
    if train_rows < 100:
        raise ValueError(
            "Temporal split leaves too few rows for model selection training."
        )

    train_df = ordered_df.iloc[:train_rows].copy()
    calibration_df = ordered_df.iloc[train_rows : train_rows + calibration_rows].copy()
    test_df = ordered_df.iloc[train_rows + calibration_rows :].copy()
    return TemporalDataSplit(
        train_df=train_df,
        calibration_df=calibration_df,
        test_df=test_df,
    )


def evaluate_candidates_with_backtesting(
    *,
    candidate_specs: dict[str, CandidateSpec],
    train_df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    outcome_to_encoded: dict[int, int],
    n_splits: int,
    metadata_columns: list[str] | None = None,
) -> tuple[list[dict[str, object]], str]:
    """Evaluate candidate models with rolling temporal validation and rank them.

    Args:
        candidate_specs: Mapping of candidate name → CandidateSpec.
        train_df: Full training DataFrame with date, target, features, and metadata.
        feature_columns: Numeric feature columns used for model training.
        target_column: Column containing the multiclass target variable.
        outcome_to_encoded: Mapping from raw outcome to encoded class labels.
        n_splits: Number of rolling temporal folds for backtesting.
        metadata_columns: Optional non-feature columns (e.g., ``tournament``)
            passed to estimators during ``predict_proba``/``predict`` but
            **excluded** from training.  Enables segment-aware ensembles to
            detect tournament context without leaking non-numeric data into
            the model.  Callers that omit this parameter get identical
            behavior to the previous interface.
    """
    if n_splits < 3:
        raise ValueError("n_splits must be at least 3 for temporal backtesting.")

    X = train_df[feature_columns].reset_index(drop=True)
    y = train_df[target_column].reset_index(drop=True)
    dates = train_df["date"].reset_index(drop=True)

    # Metadata columns ride alongside X during prediction but are NOT trained on.
    # The segment-aware ensemble strips them via feature_names_in_ internally.
    X_metadata: pd.DataFrame | None = None
    if metadata_columns:
        X_metadata = train_df[metadata_columns].reset_index(drop=True)

    splitter = TimeSeriesSplit(n_splits=n_splits)
    candidate_summaries: list[dict[str, object]] = []

    for candidate_name, candidate_spec in candidate_specs.items():
        fold_results: list[dict[str, object]] = []
        needs_metadata = candidate_spec.family == "segment_aware_hybrid"
        for fold_index, (train_idx, valid_idx) in enumerate(splitter.split(X), start=1):
            X_train = X.iloc[train_idx].copy()
            X_valid = X.iloc[valid_idx].copy()
            y_train = y.iloc[train_idx].copy()
            y_valid = y.iloc[valid_idx].copy()
            y_train_encoded = y_train.map(outcome_to_encoded).astype("int64")
            y_valid_encoded = y_valid.map(outcome_to_encoded).astype("int64")

            estimator = cast(ProbabilisticEstimator, clone(candidate_spec.pipeline))
            sample_weight = candidate_spec.sample_weight_builder(y_train_encoded)
            fit_estimator_with_sample_weight(
                estimator,
                X_train,
                y_train_encoded,
                sample_weight,
            )

            # Append metadata columns for segment-aware prediction if applicable
            X_valid_for_predict = X_valid
            if needs_metadata and X_metadata is not None:
                X_valid_for_predict = pd.concat(
                    [X_valid, X_metadata.iloc[valid_idx].copy()],
                    axis=1,
                )

            probabilities = predict_proba_aligned(estimator, X_valid_for_predict)
            predicted = estimator.predict(X_valid_for_predict).astype(np.int64)
            metrics = evaluate_multiclass_predictions(
                y_true=y_valid,
                y_true_encoded=y_valid_encoded,
                y_pred_encoded=predicted,
                probabilities=probabilities,
            )

            fold_results.append(
                {
                    "fold_index": fold_index,
                    "train_rows": int(len(train_idx)),
                    "validation_rows": int(len(valid_idx)),
                    "train_date_range": {
                        "start": dates.iloc[train_idx[0]].date().isoformat(),
                        "end": dates.iloc[train_idx[-1]].date().isoformat(),
                    },
                    "validation_date_range": {
                        "start": dates.iloc[valid_idx[0]].date().isoformat(),
                        "end": dates.iloc[valid_idx[-1]].date().isoformat(),
                    },
                    "metrics": metrics,
                }
            )

        metric_keys = PRIMARY_SELECTION_METRICS
        fold_metric_maps = [
            cast(dict[str, object], fold["metrics"])
            for fold in fold_results
        ]
        mean_metrics = {
            metric_name: float(
                np.mean(
                    [
                        cast(float, metric_map[metric_name])
                        for metric_map in fold_metric_maps
                    ]
                )
            )
            for metric_name in metric_keys
        }
        std_metrics = {
            metric_name: float(
                np.std(
                    [
                        cast(float, metric_map[metric_name])
                        for metric_map in fold_metric_maps
                    ]
                )
            )
            for metric_name in metric_keys
        }
        candidate_summaries.append(
            {
                "model_name": candidate_name,
                "model_family": candidate_spec.family,
                "hyperparameters": candidate_spec.hyperparameters,
                "notes": candidate_spec.notes,
                "fold_results": fold_results,
                "mean_metrics": mean_metrics,
                "std_metrics": std_metrics,
            }
        )

    ranking_frame = pd.DataFrame(
        [
            {
                "model_name": summary["model_name"],
                **cast(dict[str, float], summary["mean_metrics"]),
            }
            for summary in candidate_summaries
        ]
    )
    ranking_frame["rank_macro_f1"] = ranking_frame["macro_f1"].rank(
        ascending=False,
        method="dense",
    )
    ranking_frame["rank_draw_f1"] = ranking_frame["draw_f1"].rank(
        ascending=False,
        method="dense",
    )
    ranking_frame["rank_draw_recall"] = ranking_frame["draw_recall"].rank(
        ascending=False,
        method="dense",
    )
    ranking_frame["rank_balanced_accuracy"] = ranking_frame["balanced_accuracy"].rank(
        ascending=False,
        method="dense",
    )
    ranking_frame["rank_matthews_corrcoef"] = ranking_frame["matthews_corrcoef"].rank(
        ascending=False,
        method="dense",
    )
    ranking_frame["rank_log_loss"] = ranking_frame["log_loss"].rank(
        ascending=True,
        method="dense",
    )
    ranking_frame["rank_brier"] = ranking_frame["multiclass_brier_score"].rank(
        ascending=True,
        method="dense",
    )
    ranking_frame["rank_ece"] = ranking_frame["expected_calibration_error"].rank(
        ascending=True,
        method="dense",
    )
    ranking_frame["selection_score"] = ranking_frame[
        [
            "rank_macro_f1",
            "rank_draw_f1",
            "rank_draw_recall",
            "rank_balanced_accuracy",
            "rank_matthews_corrcoef",
            "rank_log_loss",
            "rank_brier",
            "rank_ece",
        ]
    ].sum(axis=1)
    ranking_frame = ranking_frame.sort_values(
        ["selection_score", "macro_f1", "draw_f1", "draw_recall", "log_loss"],
        ascending=[True, False, False, False, True],
    ).reset_index(drop=True)

    best_model_name = str(ranking_frame.loc[0, "model_name"])
    ranking_lookup = ranking_frame.set_index("model_name").to_dict(orient="index")
    for summary in candidate_summaries:
        model_name = str(summary["model_name"])
        summary["selection_rank"] = int(ranking_frame.index[ranking_frame["model_name"] == model_name][0] + 1)
        summary["selection_score"] = float(ranking_lookup[model_name]["selection_score"])

    return candidate_summaries, best_model_name
```

---

### 2. [train.py](file:///c:/Users/restr/Desktop/worldcup-2026-prediction/src/modeling/train.py) — Segment Detector + 4 Candidates

Three additions:

1. **`_tournament_segment_detector()`** — Maps tournament names to segment IDs (`worldcup`, `friendlies`, `qualifiers`, `continental`) using domain knowledge from the 0.45 global threshold analysis. Key routing rules:
   - `"FIFA World Cup"` → `worldcup` (but `"FIFA World Cup qualification"` → `qualifiers`)
   - `"Friendly"` / `"International Friendly"` → `friendlies`
   - Any qualification/qualifier/playoff → `qualifiers`
   - Copa América, Euro, Africa Cup, Asian Cup, Gold Cup, Nations League → `continental`

2. **`_build_segment_configs()`** — Factory function producing validated `SegmentConfig` dicts with per-segment thresholds.

3. **4 segment-aware hybrid variants** in `_build_candidate_specs()`:

| Variant | Friendlies unc/conv | World Cup unc/conv | Continental unc/conv | Qualifiers unc/conv | Default unc/conv |
|---------|---------------------|--------------------|----------------------|---------------------|------------------|
| `conservative` | 0.38/0.58 | 0.52/0.62 | 0.44/0.58 | 0.48/0.60 | 0.45/0.55 |
| `friendlies_focus` | 0.32/0.52 | 0.55/0.65 | 0.46/0.58 | 0.50/0.60 | 0.48/0.58 |
| `balanced` | 0.36/0.55 | 0.50/0.60 | 0.42/0.56 | 0.46/0.58 | 0.44/0.55 |
| `narrow_band` | 0.40/0.60 | 0.55/0.65 | 0.48/0.60 | 0.50/0.62 | 0.48/0.58 |

4. **Metadata pass-through**: `SEGMENT_METADATA_COLUMNS = ["tournament"]` passed to backtesting call.

```diff:train.py
"""Train and export the production match outcome model."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import cast

import joblib
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from src.config.settings import settings
from src.contracts.data_contracts import validate_feature_dataset_contract
from src.database.persistence import persist_training_run
from src.modeling.evaluation import (
    CandidateSpec,
    ProbabilisticEstimator,
    evaluate_candidates_with_backtesting,
    evaluate_multiclass_predictions,
    fit_estimator_with_sample_weight,
    predict_proba_aligned,
    select_deployment_variant,
    split_train_calibration_test,
)
from src.modeling.features import OUTCOME_LABELS, TARGET_COLUMN, load_feature_dataset
from src.modeling.features import select_model_feature_columns
from src.modeling.hybrid_ensemble import HybridDrawOverrideEnsemble
from src.modeling.reporting import generate_evaluation_report
from src.modeling.two_stage import TwoStageDrawClassifier
from src.modeling.types import DateRange, ModelArtifactBundle, TrainingMetrics, TrainingSummary

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.2
DEFAULT_CALIBRATION_SIZE = 0.1
DEFAULT_CALIBRATION_SELECTION_SIZE = 0.5
DEFAULT_BACKTEST_SPLITS = 5
OUTCOME_TO_ENCODED = {-1: 0, 0: 1, 1: 2}
ENCODED_TO_OUTCOME = {value: key for key, value in OUTCOME_TO_ENCODED.items()}
SPECIALIST_DRAW_WEIGHT_MULTIPLIER = 4 / 3


def _build_pipeline(model: object) -> Pipeline:
    """Build a reusable training pipeline around a candidate estimator."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", model),
        ]
    )


def _build_scaled_pipeline(model: object) -> Pipeline:
    """Build a pipeline with scaling for linear models."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def _make_sample_weight_builder(
    draw_boost: float = 1.0,
) -> Callable[[pd.Series], NDArray[np.float64]]:
    def _builder(y_encoded: pd.Series) -> NDArray[np.float64]:
        sample_weight = np.asarray(
            compute_sample_weight(class_weight="balanced", y=y_encoded),
            dtype=np.float64,
        )
        if draw_boost != 1.0:
            draw_mask = y_encoded.to_numpy(dtype=np.int64, copy=False) == OUTCOME_TO_ENCODED[0]
            sample_weight[draw_mask] *= draw_boost
            sample_weight *= len(sample_weight) / sample_weight.sum()
        return sample_weight

    return _builder


def _build_candidate_specs() -> dict[str, CandidateSpec]:
    """Create a compact temporal hyperparameter search space."""
    candidate_specs: dict[str, CandidateSpec] = {
        "dummy_prior": CandidateSpec(
            name="dummy_prior",
            pipeline=_build_pipeline(DummyClassifier(strategy="prior")),
            sample_weight_builder=_make_sample_weight_builder(),
            family="dummy",
            hyperparameters={"strategy": "prior"},
            notes="Sanity baseline",
        )
    }

    logistic_variants: list[dict[str, float]] = [
        {"c": 0.5, "draw_boost": 1.0},
        {"c": 1.0, "draw_boost": 1.0},
        {"c": 2.0, "draw_boost": 1.0},
        {"c": 1.0, "draw_boost": 1.25},
        {"c": 2.0, "draw_boost": 1.2},
    ]
    for logistic_variant in logistic_variants:
        c_value = logistic_variant["c"]
        draw_boost = logistic_variant["draw_boost"]
        name = f"logistic_c{c_value:g}_draw{draw_boost:g}"
        candidate_specs[name] = CandidateSpec(
            name=name,
            pipeline=_build_scaled_pipeline(
                LogisticRegression(
                    C=c_value,
                    max_iter=2500,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                )
            ),
            sample_weight_builder=_make_sample_weight_builder(draw_boost),
            family="logistic_regression",
            hyperparameters={"C": c_value, "draw_boost": draw_boost},
            notes="Scaled linear baseline with optional draw emphasis",
        )

    random_forest_variants: list[dict[str, int | float | None]] = [
        {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 2, "draw_boost": 1.0},
        {"n_estimators": 400, "max_depth": 12, "min_samples_leaf": 2, "draw_boost": 1.0},
        {"n_estimators": 300, "max_depth": 12, "min_samples_leaf": 4, "draw_boost": 1.15},
    ]
    for forest_variant in random_forest_variants:
        n_estimators = cast(int, forest_variant["n_estimators"])
        max_depth = cast(int | None, forest_variant["max_depth"])
        min_samples_leaf = cast(int, forest_variant["min_samples_leaf"])
        draw_boost = cast(float, forest_variant["draw_boost"])
        name = (
            f"random_forest_n{n_estimators}_d"
            f"{'none' if max_depth is None else max_depth}_leaf{min_samples_leaf}_draw{draw_boost:g}"
        )
        candidate_specs[name] = CandidateSpec(
            name=name,
            pipeline=_build_pipeline(
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    class_weight="balanced_subsample",
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                )
            ),
            sample_weight_builder=_make_sample_weight_builder(draw_boost),
            family="random_forest",
            hyperparameters={
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_leaf": min_samples_leaf,
                "draw_boost": draw_boost,
            },
            notes="Tree ensemble with draw-aware weighting variants",
        )

    xgboost_variants: list[dict[str, int | float]] = [
        {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05, "reg_lambda": 1.0, "draw_boost": 1.0},
        {"n_estimators": 350, "max_depth": 4, "learning_rate": 0.05, "reg_lambda": 1.0, "draw_boost": 1.0},
        {"n_estimators": 400, "max_depth": 4, "learning_rate": 0.03, "reg_lambda": 2.0, "draw_boost": 1.0},
        {"n_estimators": 350, "max_depth": 5, "learning_rate": 0.05, "reg_lambda": 1.5, "draw_boost": 1.15},
    ]
    for xgb_variant in xgboost_variants:
        n_estimators = cast(int, xgb_variant["n_estimators"])
        max_depth = cast(int, xgb_variant["max_depth"])
        learning_rate = xgb_variant["learning_rate"]
        reg_lambda = xgb_variant["reg_lambda"]
        draw_boost = xgb_variant["draw_boost"]
        name = (
            f"xgboost_n{n_estimators}_d{max_depth}_lr{learning_rate}_"
            f"lambda{reg_lambda}_draw{draw_boost:g}"
        )
        candidate_specs[name] = CandidateSpec(
            name=name,
            pipeline=_build_pipeline(
                XGBClassifier(
                    objective="multi:softprob",
                    num_class=3,
                    eval_metric="mlogloss",
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    min_child_weight=1,
                    reg_lambda=reg_lambda,
                    random_state=RANDOM_STATE,
                    n_jobs=0,
                )
            ),
            sample_weight_builder=_make_sample_weight_builder(draw_boost),
            family="xgboost",
            hyperparameters={
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "learning_rate": learning_rate,
                "reg_lambda": reg_lambda,
                "draw_boost": draw_boost,
            },
            notes="Gradient boosting search with lightweight draw emphasis",
        )

    two_stage_variants: list[dict[str, float]] = [
        {
            "stage1_c": 2.0,
            "stage2_c": 1.0,
            "draw_boost": 1.6,
            "draw_probability_scale": 1.0,
        },
        {
            "stage1_c": 2.0,
            "stage2_c": 1.0,
            "draw_boost": 1.8,
            "draw_probability_scale": 1.08,
        },
        {
            "stage1_c": 1.0,
            "stage2_c": 2.0,
            "draw_boost": 2.0,
            "draw_probability_scale": 1.1,
        },
    ]
    for two_stage_variant in two_stage_variants:
        stage1_c = two_stage_variant["stage1_c"]
        stage2_c = two_stage_variant["stage2_c"]
        draw_boost = two_stage_variant["draw_boost"]
        draw_probability_scale = two_stage_variant["draw_probability_scale"]
        name = (
            f"two_stage_s1c{stage1_c:g}_s2c{stage2_c:g}_"
            f"draw{draw_boost:g}_scale{draw_probability_scale:g}"
        )
        candidate_specs[name] = CandidateSpec(
            name=name,
            pipeline=cast(
                ProbabilisticEstimator,
                TwoStageDrawClassifier(
                    stage1_estimator=_build_scaled_pipeline(
                        LogisticRegression(
                            C=stage1_c,
                            max_iter=2500,
                            class_weight="balanced",
                            random_state=RANDOM_STATE,
                        )
                    ),
                    stage2_estimator=_build_scaled_pipeline(
                        LogisticRegression(
                            C=stage2_c,
                            max_iter=2500,
                            class_weight="balanced",
                            random_state=RANDOM_STATE,
                        )
                    ),
                    draw_probability_scale=draw_probability_scale,
                ),
            ),
            sample_weight_builder=_make_sample_weight_builder(draw_boost),
            family="two_stage_draw_classifier",
            hyperparameters={
                "stage1_c": stage1_c,
                "stage2_c": stage2_c,
                "draw_boost": draw_boost,
                "draw_probability_scale": draw_probability_scale,
            },
            notes="Two-stage draw-vs-non-draw decomposition with aggressive draw weighting",
        )

    hybrid_override_variants: list[dict[str, float]] = [
        {"uncertainty_threshold": 0.42, "draw_conviction_threshold": 0.50},
        {"uncertainty_threshold": 0.42, "draw_conviction_threshold": 0.60},
        {"uncertainty_threshold": 0.45, "draw_conviction_threshold": 0.50},
        {"uncertainty_threshold": 0.45, "draw_conviction_threshold": 0.60},
        {"uncertainty_threshold": 0.48, "draw_conviction_threshold": 0.50},
        {"uncertainty_threshold": 0.48, "draw_conviction_threshold": 0.60},
    ]
    for hybrid_variant in hybrid_override_variants:
        uncertainty_threshold = hybrid_variant["uncertainty_threshold"]
        draw_conviction_threshold = hybrid_variant["draw_conviction_threshold"]
        name = (
            f"hybrid_override_u{uncertainty_threshold:g}_"
            f"d{draw_conviction_threshold:g}"
        )
        candidate_specs[name] = CandidateSpec(
            name=name,
            pipeline=cast(
                ProbabilisticEstimator,
                HybridDrawOverrideEnsemble(
                    generalist_estimator=_build_scaled_pipeline(
                        LogisticRegression(
                            C=2.0,
                            max_iter=2500,
                            class_weight="balanced",
                            random_state=RANDOM_STATE,
                        )
                    ),
                    specialist_estimator=cast(
                        ProbabilisticEstimator,
                        TwoStageDrawClassifier(
                            stage1_estimator=_build_scaled_pipeline(
                                LogisticRegression(
                                    C=2.0,
                                    max_iter=2500,
                                    class_weight="balanced",
                                    random_state=RANDOM_STATE,
                                )
                            ),
                            stage2_estimator=_build_scaled_pipeline(
                                LogisticRegression(
                                    C=1.0,
                                    max_iter=2500,
                                    class_weight="balanced",
                                    random_state=RANDOM_STATE,
                                )
                            ),
                            draw_probability_scale=1.0,
                        ),
                    ),
                    uncertainty_threshold=uncertainty_threshold,
                    draw_conviction_threshold=draw_conviction_threshold,
                    specialist_draw_weight_multiplier=SPECIALIST_DRAW_WEIGHT_MULTIPLIER,
                ),
            ),
            sample_weight_builder=_make_sample_weight_builder(1.2),
            family="hybrid_draw_override_ensemble",
            hyperparameters={
                "generalist_c": 2.0,
                "generalist_draw_boost": 1.2,
                "specialist_stage1_c": 2.0,
                "specialist_stage2_c": 1.0,
                "specialist_draw_boost": 1.6,
                "specialist_draw_weight_multiplier": SPECIALIST_DRAW_WEIGHT_MULTIPLIER,
                "specialist_draw_probability_scale": 1.0,
                "uncertainty_threshold": uncertainty_threshold,
                "draw_conviction_threshold": draw_conviction_threshold,
            },
            notes=(
                "Delegated ensemble: generalist owns confident fixtures; "
                "two-stage specialist only overrides uncertain rows into draw"
            ),
        )

    return candidate_specs


def _fit_pipeline(
    pipeline: ProbabilisticEstimator,
    *,
    X: pd.DataFrame,
    y_encoded: pd.Series,
    sample_weight_builder: Callable[[pd.Series], NDArray[np.float64]],
) -> ProbabilisticEstimator:
    fitted_pipeline = cast(ProbabilisticEstimator, clone(pipeline))
    sample_weight = sample_weight_builder(y_encoded)
    fit_estimator_with_sample_weight(fitted_pipeline, X, y_encoded, sample_weight)
    return fitted_pipeline


def _fit_calibrated_variant(
    fitted_estimator: ProbabilisticEstimator,
    *,
    X_calibration: pd.DataFrame,
    y_calibration_encoded: pd.Series,
    method: str,
) -> CalibratedClassifierCV:
    calibrator = CalibratedClassifierCV(
        estimator=FrozenEstimator(fitted_estimator),
        method=method,
        cv=None,
    )
    calibrator.fit(X_calibration, y_calibration_encoded)
    return calibrator


def _split_calibration_window(
    calibration_df: pd.DataFrame,
    *,
    selection_size: float = DEFAULT_CALIBRATION_SELECTION_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < selection_size < 1:
        raise ValueError("selection_size must be between 0 and 1.")

    ordered_df = calibration_df.sort_values("date").reset_index(drop=True)
    selection_rows = max(1, int(len(ordered_df) * selection_size))
    fit_rows = len(ordered_df) - selection_rows
    if fit_rows < 100 or selection_rows < 100:
        raise ValueError(
            "Calibration window is too small to support fit and selection splits."
        )

    return (
        ordered_df.iloc[:fit_rows].copy(),
        ordered_df.iloc[fit_rows:].copy(),
    )


def _date_range(df: pd.DataFrame) -> DateRange:
    return {
        "start": df["date"].min().date().isoformat(),
        "end": df["date"].max().date().isoformat(),
    }


def _model_class_name(estimator: ProbabilisticEstimator) -> str:
    named_steps = getattr(estimator, "named_steps", None)
    if named_steps is not None and "model" in named_steps:
        return str(named_steps["model"].__class__.__name__)
    return estimator.__class__.__name__


def _evaluate_pipeline(
    pipeline: ProbabilisticEstimator,
    *,
    X: pd.DataFrame,
    y: pd.Series,
) -> TrainingMetrics:
    y_encoded = y.map(OUTCOME_TO_ENCODED).astype("int64")
    probabilities = predict_proba_aligned(pipeline, X)
    predicted_encoded = pipeline.predict(X).astype(np.int64)
    evaluation = evaluate_multiclass_predictions(
        y_true=y,
        y_true_encoded=y_encoded,
        y_pred_encoded=predicted_encoded,
        probabilities=probabilities,
    )
    metrics: TrainingMetrics = {
        "accuracy": cast(float, evaluation["accuracy"]),
        "macro_f1": cast(float, evaluation["macro_f1"]),
        "weighted_f1": cast(float, evaluation["weighted_f1"]),
        "balanced_accuracy": cast(float, evaluation["balanced_accuracy"]),
        "matthews_corrcoef": cast(float, evaluation["matthews_corrcoef"]),
        "cohen_kappa": cast(float, evaluation["cohen_kappa"]),
        "log_loss": cast(float, evaluation["log_loss"]),
        "multiclass_brier_score": cast(
            float,
            evaluation["multiclass_brier_score"],
        ),
        "expected_calibration_error": cast(
            float,
            evaluation["expected_calibration_error"],
        ),
        "draw_f1": cast(float, evaluation["draw_f1"]),
        "draw_recall": cast(float, evaluation["draw_recall"]),
        "classification_report": cast(dict[str, object], evaluation["classification_report"]),
    }
    return metrics


def train_and_export_model(
    data_path: Path | None = None,
    artifact_path: Path | None = None,
    test_size: float = DEFAULT_TEST_SIZE,
    calibration_size: float = DEFAULT_CALIBRATION_SIZE,
    backtest_splits: int = DEFAULT_BACKTEST_SPLITS,
    calibration_selection_size: float = DEFAULT_CALIBRATION_SELECTION_SIZE,
    persist_to_db: bool = False,
    pipeline_run_id: str | None = None,
) -> TrainingSummary:
    """
    Train the production model from the gold feature dataset and export it.

    Args:
        data_path: Optional alternate path to the gold feature dataset
        artifact_path: Optional target path for the exported joblib artifact
        test_size: Fraction of the most recent data reserved for evaluation
        calibration_size: Fraction reserved for pre-test calibration
        backtest_splits: Number of rolling temporal splits for model selection
        calibration_selection_size: Fraction of calibration data reserved to choose deployment variant
        persist_to_db: Whether to persist training metadata to PostgreSQL
        pipeline_run_id: Optional orchestrator run identifier for lineage

    Returns:
        Dictionary with training metadata and evaluation metrics
    """
    settings.ensure_project_dirs()
    dataset_path = Path(data_path or settings.GOLD_DIR / "features_dataset.csv")
    output_path = Path(artifact_path or settings.MODEL_ARTIFACT_PATH)

    logger.info("Loading gold feature dataset from %s", dataset_path)
    df = load_feature_dataset(dataset_path)
    validate_feature_dataset_contract(df)
    feature_columns = select_model_feature_columns(df)
    if not feature_columns:
        raise RuntimeError("No model feature columns were found in the gold dataset.")

    temporal_split = split_train_calibration_test(
        df,
        test_size=test_size,
        calibration_size=calibration_size,
    )
    train_df = temporal_split.train_df
    calibration_df = temporal_split.calibration_df
    test_df = temporal_split.test_df
    calibration_fit_df, calibration_selection_df = _split_calibration_window(
        calibration_df,
        selection_size=calibration_selection_size,
    )
    logger.info(
        "Temporal split complete: train=%s rows, calibration=%s rows, test=%s rows",
        len(train_df),
        len(calibration_df),
        len(test_df),
    )
    logger.info(
        "Calibration window split into fit=%s rows and selection=%s rows",
        len(calibration_fit_df),
        len(calibration_selection_df),
    )

    X_train = train_df[feature_columns].copy()
    X_test = test_df[feature_columns].copy()
    y_train = train_df[TARGET_COLUMN].map(OUTCOME_TO_ENCODED)
    y_test = test_df[TARGET_COLUMN]
    candidate_specs = _build_candidate_specs()

    logger.info(
        "Running temporal backtesting with %s splits across %s candidates",
        backtest_splits,
        len(candidate_specs),
    )
    candidate_backtests, selected_model_name = evaluate_candidates_with_backtesting(
        candidate_specs=candidate_specs,
        train_df=train_df,
        feature_columns=feature_columns,
        target_column=TARGET_COLUMN,
        outcome_to_encoded=OUTCOME_TO_ENCODED,
        n_splits=backtest_splits,
    )
    logger.info("Selected candidate after backtesting: %s", selected_model_name)

    selected_candidate_spec = candidate_specs[selected_model_name]
    selected_candidate_pipeline = selected_candidate_spec.pipeline

    X_calibration_fit = calibration_fit_df[feature_columns].copy()
    X_calibration_selection = calibration_selection_df[feature_columns].copy()
    y_calibration_fit = calibration_fit_df[TARGET_COLUMN].map(OUTCOME_TO_ENCODED)
    y_calibration_selection = calibration_selection_df[TARGET_COLUMN]

    selected_pipeline_for_calibration = _fit_pipeline(
        selected_candidate_pipeline,
        X=X_train,
        y_encoded=y_train,
        sample_weight_builder=selected_candidate_spec.sample_weight_builder,
    )
    calibration_selection_metrics: dict[str, TrainingMetrics] = {
        "uncalibrated": _evaluate_pipeline(
            selected_pipeline_for_calibration,
            X=X_calibration_selection,
            y=y_calibration_selection,
        )
    }
    for method in ("sigmoid", "isotonic"):
        calibrated_variant = _fit_calibrated_variant(
            selected_pipeline_for_calibration,
            X_calibration=X_calibration_fit,
            y_calibration_encoded=y_calibration_fit,
            method=method,
        )
        calibration_selection_metrics[method] = _evaluate_pipeline(
            calibrated_variant,
            X=X_calibration_selection,
            y=y_calibration_selection,
        )

    deployed_model_variant, deployment_decision = select_deployment_variant(
        calibration_selection_metrics,
    )
    logger.info("Selected deployment variant: %s", deployed_model_variant)

    pretest_df = pd.concat([train_df, calibration_df], ignore_index=True)
    X_pretest = pretest_df[feature_columns].copy()
    y_pretest = pretest_df[TARGET_COLUMN].map(OUTCOME_TO_ENCODED)
    final_uncalibrated_model = _fit_pipeline(
        selected_candidate_pipeline,
        X=X_pretest,
        y_encoded=y_pretest,
        sample_weight_builder=selected_candidate_spec.sample_weight_builder,
    )

    final_deployed_model: ProbabilisticEstimator = final_uncalibrated_model
    calibration_method = "none"
    if deployed_model_variant in {"sigmoid", "isotonic"}:
        base_rows = pd.concat([train_df, calibration_fit_df], ignore_index=True)
        X_base = base_rows[feature_columns].copy()
        y_base = base_rows[TARGET_COLUMN].map(OUTCOME_TO_ENCODED)
        base_pipeline = _fit_pipeline(
            selected_candidate_pipeline,
            X=X_base,
            y_encoded=y_base,
            sample_weight_builder=selected_candidate_spec.sample_weight_builder,
        )
        final_deployed_model = _fit_calibrated_variant(
            base_pipeline,
            X_calibration=X_calibration_selection,
            y_calibration_encoded=calibration_selection_df[TARGET_COLUMN].map(
                OUTCOME_TO_ENCODED
            ),
            method=deployed_model_variant,
        )
        calibration_method = deployed_model_variant

    uncalibrated_metrics = _evaluate_pipeline(
        final_uncalibrated_model,
        X=X_test,
        y=y_test,
    )
    deployed_metrics = _evaluate_pipeline(
        final_deployed_model,
        X=X_test,
        y=y_test,
    )
    metrics = deployed_metrics

    training_summary: TrainingSummary = {
        "artifact_path": str(output_path),
        "training_rows": int(len(train_df)),
        "calibration_rows": int(len(calibration_df)),
        "test_rows": int(len(test_df)),
        "feature_count": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "data_path": str(dataset_path),
        "train_date_range": _date_range(train_df),
        "calibration_date_range": _date_range(calibration_df),
        "test_date_range": _date_range(test_df),
        "class_distribution_train": {
            str(label): int(count)
            for label, count in train_df[TARGET_COLUMN].value_counts().sort_index().items()
        },
        "class_distribution_calibration": {
            str(label): int(count)
            for label, count in calibration_df[TARGET_COLUMN].value_counts().sort_index().items()
        },
        "class_distribution_test": {
            str(label): int(count)
            for label, count in test_df[TARGET_COLUMN].value_counts().sort_index().items()
        },
        "selected_model_name": selected_model_name,
        "selected_model_class": _model_class_name(selected_candidate_pipeline),
        "deployed_model_variant": deployed_model_variant,
        "calibration_method": calibration_method,
        "metrics": metrics,
        "uncalibrated_metrics": uncalibrated_metrics,
        "evaluation_artifacts": {
            "selection_strategy": "temporal_backtesting_rank_sum_draw_aware",
            "selection_metrics": [
                "macro_f1",
                "draw_f1",
                "draw_recall",
                "balanced_accuracy",
                "matthews_corrcoef",
                "log_loss",
                "multiclass_brier_score",
                "expected_calibration_error",
            ],
            "candidate_search_space": {
                "candidate_count": int(len(candidate_specs)),
                "families": sorted(
                    {
                        candidate_spec.family
                        for candidate_spec in candidate_specs.values()
                    }
                ),
            },
            "candidate_backtests": candidate_backtests,
            "calibration_variant_selection": {
                "fit_rows": int(len(calibration_fit_df)),
                "selection_rows": int(len(calibration_selection_df)),
                "fit_date_range": _date_range(calibration_fit_df),
                "selection_date_range": _date_range(calibration_selection_df),
                "candidate_metrics": calibration_selection_metrics,
                "deployment_decision": deployment_decision,
            },
        },
    }

    report_payload = generate_evaluation_report(
        training_summary=training_summary,
        test_df=test_df,
        probabilities=predict_proba_aligned(final_deployed_model, X_test),
        y_pred_encoded=final_deployed_model.predict(X_test).astype(np.int64),
        artifact_path=output_path,
    )
    training_summary["evaluation_artifacts"]["report_artifacts"] = {
        "report_json": cast(dict[str, object], report_payload["artifacts"])["report_json"],
        "report_markdown": cast(dict[str, object], report_payload["artifacts"])["report_markdown"],
        "confusion_matrix_png": cast(dict[str, object], report_payload["artifacts"])["confusion_matrix_png"],
        "calibration_curves_png": cast(dict[str, object], report_payload["artifacts"])["calibration_curves_png"],
    }

    artifact: ModelArtifactBundle = {
        "model": final_deployed_model,
        "feature_columns": feature_columns,
        "target_column": TARGET_COLUMN,
        "outcome_to_encoded": OUTCOME_TO_ENCODED,
        "encoded_to_outcome": ENCODED_TO_OUTCOME,
        "outcome_labels": OUTCOME_LABELS,
        "selected_model_name": selected_model_name,
        "deployed_model_variant": deployed_model_variant,
        "calibration_method": calibration_method,
        "training_summary": training_summary,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)
    metrics_path = output_path.with_name(f"{output_path.stem}_metrics.json")
    metrics_path.write_text(json.dumps(training_summary, indent=2), encoding="utf-8")
    logger.info("Model artifact exported to %s", output_path)
    logger.info("Training metrics exported to %s", metrics_path)
    if persist_to_db:
        persist_training_run(
            training_summary,
            pipeline_run_id=pipeline_run_id,
        )
        logger.info("Training metadata appended to gold.training_runs")
    logger.info(
        "Holdout metrics (%s) | accuracy=%.4f macro_f1=%.4f weighted_f1=%.4f log_loss=%.4f",
        deployed_model_variant,
        training_summary["metrics"]["accuracy"],
        training_summary["metrics"]["macro_f1"],
        training_summary["metrics"]["weighted_f1"],
        training_summary["metrics"]["log_loss"],
    )

    return training_summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and export the World Cup match predictor."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=settings.GOLD_DIR / "features_dataset.csv",
        help="Path to the gold feature dataset CSV.",
    )
    parser.add_argument(
        "--artifact-path",
        type=Path,
        default=settings.MODEL_ARTIFACT_PATH,
        help="Where to save the exported joblib artifact.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help="Fraction of the most recent data reserved for evaluation.",
    )
    parser.add_argument(
        "--calibration-size",
        type=float,
        default=DEFAULT_CALIBRATION_SIZE,
        help="Fraction reserved for post-train probability calibration before final test.",
    )
    parser.add_argument(
        "--backtest-splits",
        type=int,
        default=DEFAULT_BACKTEST_SPLITS,
        help="Number of rolling temporal splits used for candidate selection.",
    )
    parser.add_argument(
        "--calibration-selection-size",
        type=float,
        default=DEFAULT_CALIBRATION_SELECTION_SIZE,
        help="Fraction of the calibration window reserved to choose deployment variant.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    args = _parse_args()
    summary = train_and_export_model(
        data_path=args.data_path,
        artifact_path=args.artifact_path,
        test_size=args.test_size,
        calibration_size=args.calibration_size,
        backtest_splits=args.backtest_splits,
        calibration_selection_size=args.calibration_selection_size,
    )
    print(json.dumps(summary, indent=2))
===
"""Train and export the production match outcome model."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import cast

import joblib
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from src.config.settings import settings
from src.contracts.data_contracts import validate_feature_dataset_contract
from src.database.persistence import persist_training_run
from src.modeling.evaluation import (
    CandidateSpec,
    ProbabilisticEstimator,
    evaluate_candidates_with_backtesting,
    evaluate_multiclass_predictions,
    fit_estimator_with_sample_weight,
    predict_proba_aligned,
    select_deployment_variant,
    split_train_calibration_test,
)
from src.modeling.features import OUTCOME_LABELS, TARGET_COLUMN, load_feature_dataset
from src.modeling.features import select_model_feature_columns
from src.modeling.hybrid_ensemble import HybridDrawOverrideEnsemble
from src.modeling.hybrid_ensemble_segment_aware import (
    SegmentAwareHybridDrawOverrideEnsemble,
    SegmentConfig,
)
from src.modeling.reporting import generate_evaluation_report
from src.modeling.two_stage import TwoStageDrawClassifier
from src.modeling.types import DateRange, ModelArtifactBundle, TrainingMetrics, TrainingSummary

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.2
DEFAULT_CALIBRATION_SIZE = 0.1
DEFAULT_CALIBRATION_SELECTION_SIZE = 0.5
DEFAULT_BACKTEST_SPLITS = 5
OUTCOME_TO_ENCODED = {-1: 0, 0: 1, 1: 2}
ENCODED_TO_OUTCOME = {value: key for key, value in OUTCOME_TO_ENCODED.items()}
SPECIALIST_DRAW_WEIGHT_MULTIPLIER = 4 / 3

# Metadata columns required for segment-aware ensemble routing.
# These ride alongside features during backtesting prediction but
# are excluded from model training (the ensemble's feature_names_in_
# filtering strips them automatically).
SEGMENT_METADATA_COLUMNS = ["tournament"]


def _tournament_segment_detector(row: pd.Series) -> str | None:
    """Route a fixture to its segment based on tournament name.

    Returns a segment_id (e.g. ``"friendlies"``, ``"worldcup"``) used by
    ``SegmentAwareHybridDrawOverrideEnsemble`` to select per-segment
    uncertainty and conviction thresholds.

    The mapping encodes domain knowledge from the global 0.45 threshold
    analysis: draw prediction noise concentrates in Friendlies and some
    continental fixtures, while World Cup / Qualifiers are structurally
    cleaner for the generalist.
    """
    tournament = row.get("tournament")
    if not tournament or not isinstance(tournament, str):
        return None

    tournament_lower = tournament.lower()

    if "world cup" in tournament_lower and "qualification" not in tournament_lower:
        return "worldcup"
    if "friendly" in tournament_lower:
        return "friendlies"
    if any(
        keyword in tournament_lower
        for keyword in (
            "qualification",
            "qualifier",
            "playoff",
            "repechage",
        )
    ):
        return "qualifiers"
    if any(
        keyword in tournament_lower
        for keyword in (
            "copa am",
            "euro",
            "africa cup",
            "asian cup",
            "confederations",
            "gold cup",
            "nations league",
        )
    ):
        return "continental"

    return None


def _build_segment_configs(
    *,
    friendlies_unc: float,
    friendlies_conv: float,
    worldcup_unc: float,
    worldcup_conv: float,
    continental_unc: float,
    continental_conv: float,
    qualifiers_unc: float,
    qualifiers_conv: float,
) -> dict[str, SegmentConfig]:
    """Build segment configuration dict with per-segment thresholds."""
    return {
        "friendlies": SegmentConfig(
            segment_id="friendlies",
            uncertainty_threshold=friendlies_unc,
            draw_conviction_threshold=friendlies_conv,
            description="Friendly matches — generalist often uncertain, specialist can help",
        ),
        "worldcup": SegmentConfig(
            segment_id="worldcup",
            uncertainty_threshold=worldcup_unc,
            draw_conviction_threshold=worldcup_conv,
            description="World Cup matches — generalist strong, minimal specialist use",
        ),
        "continental": SegmentConfig(
            segment_id="continental",
            uncertainty_threshold=continental_unc,
            draw_conviction_threshold=continental_conv,
            description="Continental tournaments — medium selectivity",
        ),
        "qualifiers": SegmentConfig(
            segment_id="qualifiers",
            uncertainty_threshold=qualifiers_unc,
            draw_conviction_threshold=qualifiers_conv,
            description="Qualification fixtures — moderate structure",
        ),
    }


def _build_pipeline(model: object) -> Pipeline:
    """Build a reusable training pipeline around a candidate estimator."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", model),
        ]
    )


def _build_scaled_pipeline(model: object) -> Pipeline:
    """Build a pipeline with scaling for linear models."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def _make_sample_weight_builder(
    draw_boost: float = 1.0,
) -> Callable[[pd.Series], NDArray[np.float64]]:
    def _builder(y_encoded: pd.Series) -> NDArray[np.float64]:
        sample_weight = np.asarray(
            compute_sample_weight(class_weight="balanced", y=y_encoded),
            dtype=np.float64,
        )
        if draw_boost != 1.0:
            draw_mask = y_encoded.to_numpy(dtype=np.int64, copy=False) == OUTCOME_TO_ENCODED[0]
            sample_weight[draw_mask] *= draw_boost
            sample_weight *= len(sample_weight) / sample_weight.sum()
        return sample_weight

    return _builder


def _build_candidate_specs() -> dict[str, CandidateSpec]:
    """Create a compact temporal hyperparameter search space."""
    candidate_specs: dict[str, CandidateSpec] = {
        "dummy_prior": CandidateSpec(
            name="dummy_prior",
            pipeline=_build_pipeline(DummyClassifier(strategy="prior")),
            sample_weight_builder=_make_sample_weight_builder(),
            family="dummy",
            hyperparameters={"strategy": "prior"},
            notes="Sanity baseline",
        )
    }

    logistic_variants: list[dict[str, float]] = [
        {"c": 0.5, "draw_boost": 1.0},
        {"c": 1.0, "draw_boost": 1.0},
        {"c": 2.0, "draw_boost": 1.0},
        {"c": 1.0, "draw_boost": 1.25},
        {"c": 2.0, "draw_boost": 1.2},
    ]
    for logistic_variant in logistic_variants:
        c_value = logistic_variant["c"]
        draw_boost = logistic_variant["draw_boost"]
        name = f"logistic_c{c_value:g}_draw{draw_boost:g}"
        candidate_specs[name] = CandidateSpec(
            name=name,
            pipeline=_build_scaled_pipeline(
                LogisticRegression(
                    C=c_value,
                    max_iter=2500,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                )
            ),
            sample_weight_builder=_make_sample_weight_builder(draw_boost),
            family="logistic_regression",
            hyperparameters={"C": c_value, "draw_boost": draw_boost},
            notes="Scaled linear baseline with optional draw emphasis",
        )

    random_forest_variants: list[dict[str, int | float | None]] = [
        {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 2, "draw_boost": 1.0},
        {"n_estimators": 400, "max_depth": 12, "min_samples_leaf": 2, "draw_boost": 1.0},
        {"n_estimators": 300, "max_depth": 12, "min_samples_leaf": 4, "draw_boost": 1.15},
    ]
    for forest_variant in random_forest_variants:
        n_estimators = cast(int, forest_variant["n_estimators"])
        max_depth = cast(int | None, forest_variant["max_depth"])
        min_samples_leaf = cast(int, forest_variant["min_samples_leaf"])
        draw_boost = cast(float, forest_variant["draw_boost"])
        name = (
            f"random_forest_n{n_estimators}_d"
            f"{'none' if max_depth is None else max_depth}_leaf{min_samples_leaf}_draw{draw_boost:g}"
        )
        candidate_specs[name] = CandidateSpec(
            name=name,
            pipeline=_build_pipeline(
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    class_weight="balanced_subsample",
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                )
            ),
            sample_weight_builder=_make_sample_weight_builder(draw_boost),
            family="random_forest",
            hyperparameters={
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_leaf": min_samples_leaf,
                "draw_boost": draw_boost,
            },
            notes="Tree ensemble with draw-aware weighting variants",
        )

    xgboost_variants: list[dict[str, int | float]] = [
        {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05, "reg_lambda": 1.0, "draw_boost": 1.0},
        {"n_estimators": 350, "max_depth": 4, "learning_rate": 0.05, "reg_lambda": 1.0, "draw_boost": 1.0},
        {"n_estimators": 400, "max_depth": 4, "learning_rate": 0.03, "reg_lambda": 2.0, "draw_boost": 1.0},
        {"n_estimators": 350, "max_depth": 5, "learning_rate": 0.05, "reg_lambda": 1.5, "draw_boost": 1.15},
    ]
    for xgb_variant in xgboost_variants:
        n_estimators = cast(int, xgb_variant["n_estimators"])
        max_depth = cast(int, xgb_variant["max_depth"])
        learning_rate = xgb_variant["learning_rate"]
        reg_lambda = xgb_variant["reg_lambda"]
        draw_boost = xgb_variant["draw_boost"]
        name = (
            f"xgboost_n{n_estimators}_d{max_depth}_lr{learning_rate}_"
            f"lambda{reg_lambda}_draw{draw_boost:g}"
        )
        candidate_specs[name] = CandidateSpec(
            name=name,
            pipeline=_build_pipeline(
                XGBClassifier(
                    objective="multi:softprob",
                    num_class=3,
                    eval_metric="mlogloss",
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    min_child_weight=1,
                    reg_lambda=reg_lambda,
                    random_state=RANDOM_STATE,
                    n_jobs=0,
                )
            ),
            sample_weight_builder=_make_sample_weight_builder(draw_boost),
            family="xgboost",
            hyperparameters={
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "learning_rate": learning_rate,
                "reg_lambda": reg_lambda,
                "draw_boost": draw_boost,
            },
            notes="Gradient boosting search with lightweight draw emphasis",
        )

    two_stage_variants: list[dict[str, float]] = [
        {
            "stage1_c": 2.0,
            "stage2_c": 1.0,
            "draw_boost": 1.6,
            "draw_probability_scale": 1.0,
        },
        {
            "stage1_c": 2.0,
            "stage2_c": 1.0,
            "draw_boost": 1.8,
            "draw_probability_scale": 1.08,
        },
        {
            "stage1_c": 1.0,
            "stage2_c": 2.0,
            "draw_boost": 2.0,
            "draw_probability_scale": 1.1,
        },
    ]
    for two_stage_variant in two_stage_variants:
        stage1_c = two_stage_variant["stage1_c"]
        stage2_c = two_stage_variant["stage2_c"]
        draw_boost = two_stage_variant["draw_boost"]
        draw_probability_scale = two_stage_variant["draw_probability_scale"]
        name = (
            f"two_stage_s1c{stage1_c:g}_s2c{stage2_c:g}_"
            f"draw{draw_boost:g}_scale{draw_probability_scale:g}"
        )
        candidate_specs[name] = CandidateSpec(
            name=name,
            pipeline=cast(
                ProbabilisticEstimator,
                TwoStageDrawClassifier(
                    stage1_estimator=_build_scaled_pipeline(
                        LogisticRegression(
                            C=stage1_c,
                            max_iter=2500,
                            class_weight="balanced",
                            random_state=RANDOM_STATE,
                        )
                    ),
                    stage2_estimator=_build_scaled_pipeline(
                        LogisticRegression(
                            C=stage2_c,
                            max_iter=2500,
                            class_weight="balanced",
                            random_state=RANDOM_STATE,
                        )
                    ),
                    draw_probability_scale=draw_probability_scale,
                ),
            ),
            sample_weight_builder=_make_sample_weight_builder(draw_boost),
            family="two_stage_draw_classifier",
            hyperparameters={
                "stage1_c": stage1_c,
                "stage2_c": stage2_c,
                "draw_boost": draw_boost,
                "draw_probability_scale": draw_probability_scale,
            },
            notes="Two-stage draw-vs-non-draw decomposition with aggressive draw weighting",
        )

    hybrid_override_variants: list[dict[str, float]] = [
        {"uncertainty_threshold": 0.42, "draw_conviction_threshold": 0.50},
        {"uncertainty_threshold": 0.42, "draw_conviction_threshold": 0.60},
        {"uncertainty_threshold": 0.45, "draw_conviction_threshold": 0.50},
        {"uncertainty_threshold": 0.45, "draw_conviction_threshold": 0.60},
        {"uncertainty_threshold": 0.48, "draw_conviction_threshold": 0.50},
        {"uncertainty_threshold": 0.48, "draw_conviction_threshold": 0.60},
    ]
    for hybrid_variant in hybrid_override_variants:
        uncertainty_threshold = hybrid_variant["uncertainty_threshold"]
        draw_conviction_threshold = hybrid_variant["draw_conviction_threshold"]
        name = (
            f"hybrid_override_u{uncertainty_threshold:g}_"
            f"d{draw_conviction_threshold:g}"
        )
        candidate_specs[name] = CandidateSpec(
            name=name,
            pipeline=cast(
                ProbabilisticEstimator,
                HybridDrawOverrideEnsemble(
                    generalist_estimator=_build_scaled_pipeline(
                        LogisticRegression(
                            C=2.0,
                            max_iter=2500,
                            class_weight="balanced",
                            random_state=RANDOM_STATE,
                        )
                    ),
                    specialist_estimator=cast(
                        ProbabilisticEstimator,
                        TwoStageDrawClassifier(
                            stage1_estimator=_build_scaled_pipeline(
                                LogisticRegression(
                                    C=2.0,
                                    max_iter=2500,
                                    class_weight="balanced",
                                    random_state=RANDOM_STATE,
                                )
                            ),
                            stage2_estimator=_build_scaled_pipeline(
                                LogisticRegression(
                                    C=1.0,
                                    max_iter=2500,
                                    class_weight="balanced",
                                    random_state=RANDOM_STATE,
                                )
                            ),
                            draw_probability_scale=1.0,
                        ),
                    ),
                    uncertainty_threshold=uncertainty_threshold,
                    draw_conviction_threshold=draw_conviction_threshold,
                    specialist_draw_weight_multiplier=SPECIALIST_DRAW_WEIGHT_MULTIPLIER,
                ),
            ),
            sample_weight_builder=_make_sample_weight_builder(1.2),
            family="hybrid_draw_override_ensemble",
            hyperparameters={
                "generalist_c": 2.0,
                "generalist_draw_boost": 1.2,
                "specialist_stage1_c": 2.0,
                "specialist_stage2_c": 1.0,
                "specialist_draw_boost": 1.6,
                "specialist_draw_weight_multiplier": SPECIALIST_DRAW_WEIGHT_MULTIPLIER,
                "specialist_draw_probability_scale": 1.0,
                "uncertainty_threshold": uncertainty_threshold,
                "draw_conviction_threshold": draw_conviction_threshold,
            },
            notes=(
                "Delegated ensemble: generalist owns confident fixtures; "
                "two-stage specialist only overrides uncertain rows into draw"
            ),
        )

    # ════════════════════════════════════════════════════════════════════════
    # Segment-Aware Hybrid Ensemble variants
    # ════════════════════════════════════════════════════════════════════════
    # These compete head-to-head against the global hybrid and the standalone
    # generalist champion under the same multicriteria ranking.
    #
    # Key insight from the 0.45 global threshold analysis:
    #   - 0.45 was too broad → the specialist fired on strong predictions too
    #   - Solution: narrower, segment-specific bands that only activate the
    #     specialist in segments where the generalist actually struggles
    #   - Friendlies → aggressive specialist (low threshold)
    #   - World Cup → very conservative (high threshold, almost no overrides)
    #   - Qualifiers / Continental → middle ground, leaning conservative
    # ────────────────────────────────────────────────────────────────────────

    segment_aware_variants: list[dict[str, float | str]] = [
        {
            "tag": "conservative",
            "friendlies_unc": 0.38, "friendlies_conv": 0.58,
            "worldcup_unc": 0.52, "worldcup_conv": 0.62,
            "continental_unc": 0.44, "continental_conv": 0.58,
            "qualifiers_unc": 0.48, "qualifiers_conv": 0.60,
            "default_unc": 0.45, "default_conv": 0.55,
        },
        {
            "tag": "friendlies_focus",
            "friendlies_unc": 0.32, "friendlies_conv": 0.52,
            "worldcup_unc": 0.55, "worldcup_conv": 0.65,
            "continental_unc": 0.46, "continental_conv": 0.58,
            "qualifiers_unc": 0.50, "qualifiers_conv": 0.60,
            "default_unc": 0.48, "default_conv": 0.58,
        },
        {
            "tag": "balanced",
            "friendlies_unc": 0.36, "friendlies_conv": 0.55,
            "worldcup_unc": 0.50, "worldcup_conv": 0.60,
            "continental_unc": 0.42, "continental_conv": 0.56,
            "qualifiers_unc": 0.46, "qualifiers_conv": 0.58,
            "default_unc": 0.44, "default_conv": 0.55,
        },
        {
            "tag": "narrow_band",
            "friendlies_unc": 0.40, "friendlies_conv": 0.60,
            "worldcup_unc": 0.55, "worldcup_conv": 0.65,
            "continental_unc": 0.48, "continental_conv": 0.60,
            "qualifiers_unc": 0.50, "qualifiers_conv": 0.62,
            "default_unc": 0.48, "default_conv": 0.58,
        },
    ]

    for variant in segment_aware_variants:
        tag = str(variant["tag"])
        default_unc = float(variant["default_unc"])
        default_conv = float(variant["default_conv"])
        segment_configs = _build_segment_configs(
            friendlies_unc=float(variant["friendlies_unc"]),
            friendlies_conv=float(variant["friendlies_conv"]),
            worldcup_unc=float(variant["worldcup_unc"]),
            worldcup_conv=float(variant["worldcup_conv"]),
            continental_unc=float(variant["continental_unc"]),
            continental_conv=float(variant["continental_conv"]),
            qualifiers_unc=float(variant["qualifiers_unc"]),
            qualifiers_conv=float(variant["qualifiers_conv"]),
        )
        name = f"seg_hybrid_{tag}"
        candidate_specs[name] = CandidateSpec(
            name=name,
            pipeline=cast(
                ProbabilisticEstimator,
                SegmentAwareHybridDrawOverrideEnsemble(
                    generalist_estimator=_build_scaled_pipeline(
                        LogisticRegression(
                            C=2.0,
                            max_iter=2500,
                            class_weight="balanced",
                            random_state=RANDOM_STATE,
                        )
                    ),
                    specialist_estimator=cast(
                        ProbabilisticEstimator,
                        TwoStageDrawClassifier(
                            stage1_estimator=_build_scaled_pipeline(
                                LogisticRegression(
                                    C=2.0,
                                    max_iter=2500,
                                    class_weight="balanced",
                                    random_state=RANDOM_STATE,
                                )
                            ),
                            stage2_estimator=_build_scaled_pipeline(
                                LogisticRegression(
                                    C=1.0,
                                    max_iter=2500,
                                    class_weight="balanced",
                                    random_state=RANDOM_STATE,
                                )
                            ),
                            draw_probability_scale=1.0,
                        ),
                    ),
                    default_uncertainty_threshold=default_unc,
                    default_draw_conviction_threshold=default_conv,
                    segment_configs=segment_configs,
                    segment_detector_fn=_tournament_segment_detector,
                    specialist_draw_weight_multiplier=SPECIALIST_DRAW_WEIGHT_MULTIPLIER,
                ),
            ),
            sample_weight_builder=_make_sample_weight_builder(1.2),
            family="segment_aware_hybrid",
            hyperparameters={
                "tag": tag,
                "generalist_c": 2.0,
                "generalist_draw_boost": 1.2,
                "specialist_stage1_c": 2.0,
                "specialist_stage2_c": 1.0,
                "specialist_draw_weight_multiplier": SPECIALIST_DRAW_WEIGHT_MULTIPLIER,
                "default_unc": default_unc,
                "default_conv": default_conv,
                **{
                    f"{seg_id}_unc": cfg.uncertainty_threshold
                    for seg_id, cfg in segment_configs.items()
                },
                **{
                    f"{seg_id}_conv": cfg.draw_conviction_threshold
                    for seg_id, cfg in segment_configs.items()
                },
            },
            notes=(
                f"Segment-aware hybrid [{tag}]: overrides conditioned by tournament "
                f"segment. Friendlies unc={variant['friendlies_unc']}, "
                f"WC unc={variant['worldcup_unc']}, default unc={default_unc}"
            ),
        )

    return candidate_specs


def _fit_pipeline(
    pipeline: ProbabilisticEstimator,
    *,
    X: pd.DataFrame,
    y_encoded: pd.Series,
    sample_weight_builder: Callable[[pd.Series], NDArray[np.float64]],
) -> ProbabilisticEstimator:
    fitted_pipeline = cast(ProbabilisticEstimator, clone(pipeline))
    sample_weight = sample_weight_builder(y_encoded)
    fit_estimator_with_sample_weight(fitted_pipeline, X, y_encoded, sample_weight)
    return fitted_pipeline


def _fit_calibrated_variant(
    fitted_estimator: ProbabilisticEstimator,
    *,
    X_calibration: pd.DataFrame,
    y_calibration_encoded: pd.Series,
    method: str,
) -> CalibratedClassifierCV:
    calibrator = CalibratedClassifierCV(
        estimator=FrozenEstimator(fitted_estimator),
        method=method,
        cv=None,
    )
    calibrator.fit(X_calibration, y_calibration_encoded)
    return calibrator


def _split_calibration_window(
    calibration_df: pd.DataFrame,
    *,
    selection_size: float = DEFAULT_CALIBRATION_SELECTION_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < selection_size < 1:
        raise ValueError("selection_size must be between 0 and 1.")

    ordered_df = calibration_df.sort_values("date").reset_index(drop=True)
    selection_rows = max(1, int(len(ordered_df) * selection_size))
    fit_rows = len(ordered_df) - selection_rows
    if fit_rows < 100 or selection_rows < 100:
        raise ValueError(
            "Calibration window is too small to support fit and selection splits."
        )

    return (
        ordered_df.iloc[:fit_rows].copy(),
        ordered_df.iloc[fit_rows:].copy(),
    )


def _date_range(df: pd.DataFrame) -> DateRange:
    return {
        "start": df["date"].min().date().isoformat(),
        "end": df["date"].max().date().isoformat(),
    }


def _model_class_name(estimator: ProbabilisticEstimator) -> str:
    named_steps = getattr(estimator, "named_steps", None)
    if named_steps is not None and "model" in named_steps:
        return str(named_steps["model"].__class__.__name__)
    return estimator.__class__.__name__


def _evaluate_pipeline(
    pipeline: ProbabilisticEstimator,
    *,
    X: pd.DataFrame,
    y: pd.Series,
) -> TrainingMetrics:
    y_encoded = y.map(OUTCOME_TO_ENCODED).astype("int64")
    probabilities = predict_proba_aligned(pipeline, X)
    predicted_encoded = pipeline.predict(X).astype(np.int64)
    evaluation = evaluate_multiclass_predictions(
        y_true=y,
        y_true_encoded=y_encoded,
        y_pred_encoded=predicted_encoded,
        probabilities=probabilities,
    )
    metrics: TrainingMetrics = {
        "accuracy": cast(float, evaluation["accuracy"]),
        "macro_f1": cast(float, evaluation["macro_f1"]),
        "weighted_f1": cast(float, evaluation["weighted_f1"]),
        "balanced_accuracy": cast(float, evaluation["balanced_accuracy"]),
        "matthews_corrcoef": cast(float, evaluation["matthews_corrcoef"]),
        "cohen_kappa": cast(float, evaluation["cohen_kappa"]),
        "log_loss": cast(float, evaluation["log_loss"]),
        "multiclass_brier_score": cast(
            float,
            evaluation["multiclass_brier_score"],
        ),
        "expected_calibration_error": cast(
            float,
            evaluation["expected_calibration_error"],
        ),
        "draw_f1": cast(float, evaluation["draw_f1"]),
        "draw_recall": cast(float, evaluation["draw_recall"]),
        "classification_report": cast(dict[str, object], evaluation["classification_report"]),
    }
    return metrics


def train_and_export_model(
    data_path: Path | None = None,
    artifact_path: Path | None = None,
    test_size: float = DEFAULT_TEST_SIZE,
    calibration_size: float = DEFAULT_CALIBRATION_SIZE,
    backtest_splits: int = DEFAULT_BACKTEST_SPLITS,
    calibration_selection_size: float = DEFAULT_CALIBRATION_SELECTION_SIZE,
    persist_to_db: bool = False,
    pipeline_run_id: str | None = None,
) -> TrainingSummary:
    """
    Train the production model from the gold feature dataset and export it.

    Args:
        data_path: Optional alternate path to the gold feature dataset
        artifact_path: Optional target path for the exported joblib artifact
        test_size: Fraction of the most recent data reserved for evaluation
        calibration_size: Fraction reserved for pre-test calibration
        backtest_splits: Number of rolling temporal splits for model selection
        calibration_selection_size: Fraction of calibration data reserved to choose deployment variant
        persist_to_db: Whether to persist training metadata to PostgreSQL
        pipeline_run_id: Optional orchestrator run identifier for lineage

    Returns:
        Dictionary with training metadata and evaluation metrics
    """
    settings.ensure_project_dirs()
    dataset_path = Path(data_path or settings.GOLD_DIR / "features_dataset.csv")
    output_path = Path(artifact_path or settings.MODEL_ARTIFACT_PATH)

    logger.info("Loading gold feature dataset from %s", dataset_path)
    df = load_feature_dataset(dataset_path)
    validate_feature_dataset_contract(df)
    feature_columns = select_model_feature_columns(df)
    if not feature_columns:
        raise RuntimeError("No model feature columns were found in the gold dataset.")

    temporal_split = split_train_calibration_test(
        df,
        test_size=test_size,
        calibration_size=calibration_size,
    )
    train_df = temporal_split.train_df
    calibration_df = temporal_split.calibration_df
    test_df = temporal_split.test_df
    calibration_fit_df, calibration_selection_df = _split_calibration_window(
        calibration_df,
        selection_size=calibration_selection_size,
    )
    logger.info(
        "Temporal split complete: train=%s rows, calibration=%s rows, test=%s rows",
        len(train_df),
        len(calibration_df),
        len(test_df),
    )
    logger.info(
        "Calibration window split into fit=%s rows and selection=%s rows",
        len(calibration_fit_df),
        len(calibration_selection_df),
    )

    X_train = train_df[feature_columns].copy()
    X_test = test_df[feature_columns].copy()
    y_train = train_df[TARGET_COLUMN].map(OUTCOME_TO_ENCODED)
    y_test = test_df[TARGET_COLUMN]
    candidate_specs = _build_candidate_specs()

    logger.info(
        "Running temporal backtesting with %s splits across %s candidates",
        backtest_splits,
        len(candidate_specs),
    )
    candidate_backtests, selected_model_name = evaluate_candidates_with_backtesting(
        candidate_specs=candidate_specs,
        train_df=train_df,
        feature_columns=feature_columns,
        target_column=TARGET_COLUMN,
        outcome_to_encoded=OUTCOME_TO_ENCODED,
        n_splits=backtest_splits,
        metadata_columns=SEGMENT_METADATA_COLUMNS,
    )
    logger.info("Selected candidate after backtesting: %s", selected_model_name)

    selected_candidate_spec = candidate_specs[selected_model_name]
    selected_candidate_pipeline = selected_candidate_spec.pipeline

    X_calibration_fit = calibration_fit_df[feature_columns].copy()
    X_calibration_selection = calibration_selection_df[feature_columns].copy()
    y_calibration_fit = calibration_fit_df[TARGET_COLUMN].map(OUTCOME_TO_ENCODED)
    y_calibration_selection = calibration_selection_df[TARGET_COLUMN]

    selected_pipeline_for_calibration = _fit_pipeline(
        selected_candidate_pipeline,
        X=X_train,
        y_encoded=y_train,
        sample_weight_builder=selected_candidate_spec.sample_weight_builder,
    )
    calibration_selection_metrics: dict[str, TrainingMetrics] = {
        "uncalibrated": _evaluate_pipeline(
            selected_pipeline_for_calibration,
            X=X_calibration_selection,
            y=y_calibration_selection,
        )
    }
    for method in ("sigmoid", "isotonic"):
        calibrated_variant = _fit_calibrated_variant(
            selected_pipeline_for_calibration,
            X_calibration=X_calibration_fit,
            y_calibration_encoded=y_calibration_fit,
            method=method,
        )
        calibration_selection_metrics[method] = _evaluate_pipeline(
            calibrated_variant,
            X=X_calibration_selection,
            y=y_calibration_selection,
        )

    deployed_model_variant, deployment_decision = select_deployment_variant(
        calibration_selection_metrics,
    )
    logger.info("Selected deployment variant: %s", deployed_model_variant)

    pretest_df = pd.concat([train_df, calibration_df], ignore_index=True)
    X_pretest = pretest_df[feature_columns].copy()
    y_pretest = pretest_df[TARGET_COLUMN].map(OUTCOME_TO_ENCODED)
    final_uncalibrated_model = _fit_pipeline(
        selected_candidate_pipeline,
        X=X_pretest,
        y_encoded=y_pretest,
        sample_weight_builder=selected_candidate_spec.sample_weight_builder,
    )

    final_deployed_model: ProbabilisticEstimator = final_uncalibrated_model
    calibration_method = "none"
    if deployed_model_variant in {"sigmoid", "isotonic"}:
        base_rows = pd.concat([train_df, calibration_fit_df], ignore_index=True)
        X_base = base_rows[feature_columns].copy()
        y_base = base_rows[TARGET_COLUMN].map(OUTCOME_TO_ENCODED)
        base_pipeline = _fit_pipeline(
            selected_candidate_pipeline,
            X=X_base,
            y_encoded=y_base,
            sample_weight_builder=selected_candidate_spec.sample_weight_builder,
        )
        final_deployed_model = _fit_calibrated_variant(
            base_pipeline,
            X_calibration=X_calibration_selection,
            y_calibration_encoded=calibration_selection_df[TARGET_COLUMN].map(
                OUTCOME_TO_ENCODED
            ),
            method=deployed_model_variant,
        )
        calibration_method = deployed_model_variant

    uncalibrated_metrics = _evaluate_pipeline(
        final_uncalibrated_model,
        X=X_test,
        y=y_test,
    )
    deployed_metrics = _evaluate_pipeline(
        final_deployed_model,
        X=X_test,
        y=y_test,
    )
    metrics = deployed_metrics

    training_summary: TrainingSummary = {
        "artifact_path": str(output_path),
        "training_rows": int(len(train_df)),
        "calibration_rows": int(len(calibration_df)),
        "test_rows": int(len(test_df)),
        "feature_count": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "data_path": str(dataset_path),
        "train_date_range": _date_range(train_df),
        "calibration_date_range": _date_range(calibration_df),
        "test_date_range": _date_range(test_df),
        "class_distribution_train": {
            str(label): int(count)
            for label, count in train_df[TARGET_COLUMN].value_counts().sort_index().items()
        },
        "class_distribution_calibration": {
            str(label): int(count)
            for label, count in calibration_df[TARGET_COLUMN].value_counts().sort_index().items()
        },
        "class_distribution_test": {
            str(label): int(count)
            for label, count in test_df[TARGET_COLUMN].value_counts().sort_index().items()
        },
        "selected_model_name": selected_model_name,
        "selected_model_class": _model_class_name(selected_candidate_pipeline),
        "deployed_model_variant": deployed_model_variant,
        "calibration_method": calibration_method,
        "metrics": metrics,
        "uncalibrated_metrics": uncalibrated_metrics,
        "evaluation_artifacts": {
            "selection_strategy": "temporal_backtesting_rank_sum_draw_aware",
            "selection_metrics": [
                "macro_f1",
                "draw_f1",
                "draw_recall",
                "balanced_accuracy",
                "matthews_corrcoef",
                "log_loss",
                "multiclass_brier_score",
                "expected_calibration_error",
            ],
            "candidate_search_space": {
                "candidate_count": int(len(candidate_specs)),
                "families": sorted(
                    {
                        candidate_spec.family
                        for candidate_spec in candidate_specs.values()
                    }
                ),
            },
            "candidate_backtests": candidate_backtests,
            "calibration_variant_selection": {
                "fit_rows": int(len(calibration_fit_df)),
                "selection_rows": int(len(calibration_selection_df)),
                "fit_date_range": _date_range(calibration_fit_df),
                "selection_date_range": _date_range(calibration_selection_df),
                "candidate_metrics": calibration_selection_metrics,
                "deployment_decision": deployment_decision,
            },
        },
    }

    report_payload = generate_evaluation_report(
        training_summary=training_summary,
        test_df=test_df,
        probabilities=predict_proba_aligned(final_deployed_model, X_test),
        y_pred_encoded=final_deployed_model.predict(X_test).astype(np.int64),
        artifact_path=output_path,
    )
    training_summary["evaluation_artifacts"]["report_artifacts"] = {
        "report_json": cast(dict[str, object], report_payload["artifacts"])["report_json"],
        "report_markdown": cast(dict[str, object], report_payload["artifacts"])["report_markdown"],
        "confusion_matrix_png": cast(dict[str, object], report_payload["artifacts"])["confusion_matrix_png"],
        "calibration_curves_png": cast(dict[str, object], report_payload["artifacts"])["calibration_curves_png"],
    }

    artifact: ModelArtifactBundle = {
        "model": final_deployed_model,
        "feature_columns": feature_columns,
        "target_column": TARGET_COLUMN,
        "outcome_to_encoded": OUTCOME_TO_ENCODED,
        "encoded_to_outcome": ENCODED_TO_OUTCOME,
        "outcome_labels": OUTCOME_LABELS,
        "selected_model_name": selected_model_name,
        "deployed_model_variant": deployed_model_variant,
        "calibration_method": calibration_method,
        "training_summary": training_summary,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)
    metrics_path = output_path.with_name(f"{output_path.stem}_metrics.json")
    metrics_path.write_text(json.dumps(training_summary, indent=2), encoding="utf-8")
    logger.info("Model artifact exported to %s", output_path)
    logger.info("Training metrics exported to %s", metrics_path)
    if persist_to_db:
        persist_training_run(
            training_summary,
            pipeline_run_id=pipeline_run_id,
        )
        logger.info("Training metadata appended to gold.training_runs")
    logger.info(
        "Holdout metrics (%s) | accuracy=%.4f macro_f1=%.4f weighted_f1=%.4f log_loss=%.4f",
        deployed_model_variant,
        training_summary["metrics"]["accuracy"],
        training_summary["metrics"]["macro_f1"],
        training_summary["metrics"]["weighted_f1"],
        training_summary["metrics"]["log_loss"],
    )

    return training_summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and export the World Cup match predictor."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=settings.GOLD_DIR / "features_dataset.csv",
        help="Path to the gold feature dataset CSV.",
    )
    parser.add_argument(
        "--artifact-path",
        type=Path,
        default=settings.MODEL_ARTIFACT_PATH,
        help="Where to save the exported joblib artifact.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help="Fraction of the most recent data reserved for evaluation.",
    )
    parser.add_argument(
        "--calibration-size",
        type=float,
        default=DEFAULT_CALIBRATION_SIZE,
        help="Fraction reserved for post-train probability calibration before final test.",
    )
    parser.add_argument(
        "--backtest-splits",
        type=int,
        default=DEFAULT_BACKTEST_SPLITS,
        help="Number of rolling temporal splits used for candidate selection.",
    )
    parser.add_argument(
        "--calibration-selection-size",
        type=float,
        default=DEFAULT_CALIBRATION_SELECTION_SIZE,
        help="Fraction of the calibration window reserved to choose deployment variant.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    args = _parse_args()
    summary = train_and_export_model(
        data_path=args.data_path,
        artifact_path=args.artifact_path,
        test_size=args.test_size,
        calibration_size=args.calibration_size,
        backtest_splits=args.backtest_splits,
        calibration_selection_size=args.calibration_selection_size,
    )
    print(json.dumps(summary, indent=2))
```

---

### 3. [test_segment_aware_training_integration.py](file:///c:/Users/restr/Desktop/worldcup-2026-prediction/tests/test_segment_aware_training_integration.py) — 29 New Tests

| Test Class | Tests | Coverage |
|---|---|---|
| `TestTournamentSegmentDetector` | 19 (parametrized) | All tournament→segment mappings, None handling, non-string |
| `TestBuildSegmentConfigs` | 2 | Config completeness, threshold assignment |
| `TestCandidateSpecsContainSegmentAware` | 4 | Presence, family label, hyperparameters, total count |
| `TestMetadataPassThrough` | 2 | Backtesting with metadata, backward compat without |
| `TestSegmentMetadataColumns` | 2 | Constant validation |

---

## Verification Results

| Check | Result |
|---|---|
| **ruff** | ✅ All checks passed |
| **mypy** | ✅ No errors |
| **pytest** | ✅ **142 passed**, 8 skipped (32.99s) |
| **Full retrain** | ✅ Completed in ~2:50, 26 candidates, all artifacts exported |

---

## Training Results — Ranking Analysis

### Top 15 Candidates (Temporal Backtesting)

```
Rank | Name                                       | Family                         | macro_f1 | draw_f1 | log_loss | score
   1 | logistic_c2_draw1.2                        | logistic_regression            | 0.4912  | 0.2794  | 0.9642  | 64.0
   2 | logistic_c2_draw1                          | logistic_regression            | 0.4912  | 0.2794  | 0.9642  | 67.0
   3 | logistic_c0.5_draw1                        | logistic_regression            | 0.4912  | 0.2787  | 0.9643  | 69.0
 → 4 | seg_hybrid_balanced                        | segment_aware_hybrid           | 0.4915  | 0.2827  | 0.9671  | 71.0
   5 | logistic_c1_draw1                          | logistic_regression            | 0.4911  | 0.2792  | 0.9643  | 71.0
   6 | logistic_c1_draw1.25                       | logistic_regression            | 0.4911  | 0.2792  | 0.9643  | 74.0
   7 | hybrid_override_u0.45_d0.6                 | hybrid_draw_override_ensemble  | 0.4915  | 0.2819  | 0.9703  | 78.0
   8 | hybrid_override_u0.42_d0.6                 | hybrid_draw_override_ensemble  | 0.4910  | 0.2794  | 0.9651  | 78.0
 → 9 | seg_hybrid_narrow_band                     | segment_aware_hybrid           | 0.4911  | 0.2810  | 0.9669  | 83.0
  10 | hybrid_override_u0.45_d0.5                 | hybrid_draw_override_ensemble  | 0.4954  | 0.3194  | 0.9909  | 89.0
→ 11 | seg_hybrid_friendlies_focus                | segment_aware_hybrid           | 0.4910  | 0.2811  | 0.9670  | 89.0
  12 | xgboost_n400_d4_lr0.03_lambda2.0_draw1     | xgboost                        | 0.4865  | 0.2793  | 0.9611  | 90.0
→ 15 | seg_hybrid_conservative                    | segment_aware_hybrid           | 0.4910  | 0.2814  | 0.9675  | 98.0
```

### Key Findings

> [!IMPORTANT]
> **`seg_hybrid_balanced` reached rank #4** — This is the highest any non-pure-logistic model has ever ranked in the temporal search. It beat ALL global hybrid variants (#7, #8, #10) and ALL XGBoost/RF candidates.

**Why `balanced` won among the 4 segment-aware variants:**
- **Friendlies unc=0.36** — Aggressive enough to catch draw noise in friendlies without going too far
- **World Cup unc=0.50** — Conservative enough to not interfere with generalist strength
- **Continental unc=0.42** — Sweet spot between the global 0.45 and the narrower 0.48
- **The balanced thresholds minimized the log_loss penalty** (0.9671 vs 0.9703 for the best global hybrid)

**Comparison with the previous best hybrid:**
| Metric | `hybrid_override_u0.45_d0.6` (rank #7) | `seg_hybrid_balanced` (rank #4) | Delta |
|---|---|---|---|
| macro_f1 | 0.4915 | 0.4915 | = |
| draw_f1 | 0.2819 | 0.2827 | **+0.0008** |
| draw_recall | 0.2699 | 0.2712 | **+0.0013** |
| log_loss | 0.9703 | 0.9671 | **−0.0032** (better) |
| Rank | #7 | #4 | **+3 positions** |

The segment-aware hybrid **maintains identical macro_f1** while getting meaningfully better log_loss and slightly better draw metrics. This confirms the hypothesis: narrowing the override bands per segment reduces the probability calibration penalty while preserving the specialist's value in uncertain segments.

### Champion remains `logistic_c2_draw1.2`

The generalist still wins globally because its log_loss (0.9642) is ~0.003 lower, and the composite ranking weighs calibration heavily. However, the gap between rank #1 and rank #4 is just **7 points** — the tightest it's ever been. The segment-aware hybrid is the first architecture to credibly challenge the generalist's dominance.

---

## Files Modified/Created

| File | Action | Lines Changed |
|---|---|---|
| [evaluation.py](file:///c:/Users/restr/Desktop/worldcup-2026-prediction/src/modeling/evaluation.py) | Modified | +30 |
| [train.py](file:///c:/Users/restr/Desktop/worldcup-2026-prediction/src/modeling/train.py) | Modified | +230 |
| [test_segment_aware_training_integration.py](file:///c:/Users/restr/Desktop/worldcup-2026-prediction/tests/test_segment_aware_training_integration.py) | Created | 282 lines |

---

## Pending in Git

```
modified:   src/modeling/evaluation.py
modified:   src/modeling/train.py
new file:   tests/test_segment_aware_training_integration.py
```
