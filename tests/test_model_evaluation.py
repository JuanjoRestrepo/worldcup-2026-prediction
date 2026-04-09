"""Unit tests for temporal evaluation and calibration helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.modeling.evaluation import (
    evaluate_multiclass_predictions,
    select_deployment_variant,
    split_train_calibration_test,
)


def test_split_train_calibration_test_respects_temporal_order():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=1000, freq="D"),
            "target_multiclass": np.random.default_rng(42).choice([-1, 0, 1], 1000),
        }
    )

    split = split_train_calibration_test(
        df,
        test_size=0.2,
        calibration_size=0.1,
    )

    assert len(split.train_df) == 700
    assert len(split.calibration_df) == 100
    assert len(split.test_df) == 200
    assert split.train_df["date"].max() < split.calibration_df["date"].min()
    assert split.calibration_df["date"].max() < split.test_df["date"].min()


def test_evaluate_multiclass_predictions_returns_calibration_metrics():
    y_true = pd.Series([-1, 0, 1, 1], dtype="int64")
    y_true_encoded = pd.Series([0, 1, 2, 2], dtype="int64")
    y_pred_encoded = np.array([0, 1, 2, 2], dtype=np.int64)
    probabilities = np.array(
        [
            [0.80, 0.10, 0.10],
            [0.10, 0.75, 0.15],
            [0.15, 0.10, 0.75],
            [0.20, 0.15, 0.65],
        ],
        dtype=np.float64,
    )

    metrics = evaluate_multiclass_predictions(
        y_true=y_true,
        y_true_encoded=y_true_encoded,
        y_pred_encoded=y_pred_encoded,
        probabilities=probabilities,
    )

    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert "multiclass_brier_score" in metrics
    assert "expected_calibration_error" in metrics
    assert "classification_report" in metrics


def test_select_deployment_variant_rejects_calibration_that_collapses_draw_recall():
    baseline_metrics = {
        "macro_f1": 0.52,
        "weighted_f1": 0.57,
        "log_loss": 0.91,
        "expected_calibration_error": 0.05,
        "classification_report": {
            "away_win": {"recall": 0.66},
            "draw": {"recall": 0.24},
            "home_win": {"recall": 0.68},
        },
    }
    overcalibrated_metrics = {
        "macro_f1": 0.44,
        "weighted_f1": 0.52,
        "log_loss": 0.87,
        "expected_calibration_error": 0.03,
        "classification_report": {
            "away_win": {"recall": 0.64},
            "draw": {"recall": 0.01},
            "home_win": {"recall": 0.87},
        },
    }

    selected_variant, decision = select_deployment_variant(
        {
            "uncalibrated": baseline_metrics,
            "sigmoid": overcalibrated_metrics,
        }
    )

    assert selected_variant == "uncalibrated"
    assert decision["chosen_variant"] == "uncalibrated"


def test_select_deployment_variant_accepts_material_probability_gain_with_small_f1_drop():
    baseline_metrics = {
        "macro_f1": 0.52,
        "weighted_f1": 0.57,
        "log_loss": 0.91,
        "expected_calibration_error": 0.05,
        "classification_report": {
            "away_win": {"recall": 0.66},
            "draw": {"recall": 0.24},
            "home_win": {"recall": 0.68},
        },
    }
    calibrated_metrics = {
        "macro_f1": 0.511,
        "weighted_f1": 0.562,
        "log_loss": 0.89,
        "expected_calibration_error": 0.038,
        "classification_report": {
            "away_win": {"recall": 0.65},
            "draw": {"recall": 0.21},
            "home_win": {"recall": 0.69},
        },
    }

    selected_variant, decision = select_deployment_variant(
        {
            "uncalibrated": baseline_metrics,
            "isotonic": calibrated_metrics,
        }
    )

    assert selected_variant == "isotonic"
    assert "isotonic" in decision["viable_variants"]
