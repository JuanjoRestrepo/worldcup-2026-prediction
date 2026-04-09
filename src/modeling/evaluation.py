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


class ProbabilisticEstimator(Protocol):
    """Protocol for fitted probabilistic classifiers."""

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs: object) -> object: ...

    def predict(self, X: pd.DataFrame) -> NDArray[np.int64]: ...

    def predict_proba(self, X: pd.DataFrame) -> NDArray[np.float64]: ...


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
    candidate_pipelines: dict[str, ProbabilisticEstimator],
    train_df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    outcome_to_encoded: dict[int, int],
    sample_weight_builder: Callable[[pd.Series], NDArray[np.float64]],
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

    for candidate_name, candidate_pipeline in candidate_pipelines.items():
        fold_results: list[dict[str, object]] = []
        for fold_index, (train_idx, valid_idx) in enumerate(splitter.split(X), start=1):
            X_train = X.iloc[train_idx].copy()
            X_valid = X.iloc[valid_idx].copy()
            y_train = y.iloc[train_idx].copy()
            y_valid = y.iloc[valid_idx].copy()
            y_train_encoded = y_train.map(outcome_to_encoded).astype("int64")
            y_valid_encoded = y_valid.map(outcome_to_encoded).astype("int64")

            estimator = cast(ProbabilisticEstimator, clone(candidate_pipeline))
            sample_weight = sample_weight_builder(y_train_encoded)
            estimator.fit(X_train, y_train_encoded, model__sample_weight=sample_weight)

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
            "rank_balanced_accuracy",
            "rank_matthews_corrcoef",
            "rank_log_loss",
            "rank_brier",
            "rank_ece",
        ]
    ].sum(axis=1)
    ranking_frame = ranking_frame.sort_values(
        ["selection_score", "macro_f1", "log_loss"],
        ascending=[True, False, True],
    ).reset_index(drop=True)

    best_model_name = str(ranking_frame.loc[0, "model_name"])
    ranking_lookup = ranking_frame.set_index("model_name").to_dict(orient="index")
    for summary in candidate_summaries:
        model_name = str(summary["model_name"])
        summary["selection_rank"] = int(ranking_frame.index[ranking_frame["model_name"] == model_name][0] + 1)
        summary["selection_score"] = float(ranking_lookup[model_name]["selection_score"])

    return candidate_summaries, best_model_name
