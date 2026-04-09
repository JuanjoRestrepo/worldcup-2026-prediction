"""Train and export the production match outcome model."""

from __future__ import annotations

import argparse
import json
import logging
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
    ProbabilisticEstimator,
    evaluate_candidates_with_backtesting,
    evaluate_multiclass_predictions,
    predict_proba_aligned,
    select_deployment_variant,
    split_train_calibration_test,
)
from src.modeling.features import OUTCOME_LABELS, TARGET_COLUMN, load_feature_dataset
from src.modeling.features import select_model_feature_columns
from src.modeling.types import DateRange, ModelArtifactBundle, TrainingMetrics, TrainingSummary

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.2
DEFAULT_CALIBRATION_SIZE = 0.1
DEFAULT_CALIBRATION_SELECTION_SIZE = 0.5
DEFAULT_BACKTEST_SPLITS = 5
OUTCOME_TO_ENCODED = {-1: 0, 0: 1, 1: 2}
ENCODED_TO_OUTCOME = {value: key for key, value in OUTCOME_TO_ENCODED.items()}


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


def _build_candidate_pipelines() -> dict[str, Pipeline]:
    """Create the candidate model set for temporal selection."""
    return {
        "dummy_prior": _build_pipeline(
            DummyClassifier(strategy="prior")
        ),
        "logistic_regression": _build_scaled_pipeline(
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )
        ),
        "random_forest": _build_pipeline(
            RandomForestClassifier(
                n_estimators=250,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=1,
            )
        ),
        "xgboost": _build_pipeline(
            XGBClassifier(
                objective="multi:softprob",
                num_class=3,
                eval_metric="mlogloss",
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                min_child_weight=1,
                reg_lambda=1.0,
                random_state=RANDOM_STATE,
                n_jobs=0,
            )
        ),
    }


def _build_sample_weight(y_encoded: pd.Series) -> NDArray[np.float64]:
    return np.asarray(
        compute_sample_weight(class_weight="balanced", y=y_encoded),
        dtype=np.float64,
    )


def _fit_pipeline(
    pipeline: Pipeline,
    *,
    X: pd.DataFrame,
    y_encoded: pd.Series,
) -> Pipeline:
    fitted_pipeline = cast(Pipeline, clone(pipeline))
    sample_weight = _build_sample_weight(y_encoded)
    fitted_pipeline.fit(X, y_encoded, model__sample_weight=sample_weight)
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
    candidate_pipelines = _build_candidate_pipelines()

    logger.info(
        "Running temporal backtesting with %s splits across %s candidates",
        backtest_splits,
        len(candidate_pipelines),
    )
    candidate_backtests, selected_model_name = evaluate_candidates_with_backtesting(
        candidate_pipelines=candidate_pipelines,
        train_df=train_df,
        feature_columns=feature_columns,
        target_column=TARGET_COLUMN,
        outcome_to_encoded=OUTCOME_TO_ENCODED,
        sample_weight_builder=_build_sample_weight,
        n_splits=backtest_splits,
    )
    logger.info("Selected candidate after backtesting: %s", selected_model_name)

    selected_candidate_pipeline = candidate_pipelines[selected_model_name]

    X_calibration_fit = calibration_fit_df[feature_columns].copy()
    X_calibration_selection = calibration_selection_df[feature_columns].copy()
    y_calibration_fit = calibration_fit_df[TARGET_COLUMN].map(OUTCOME_TO_ENCODED)
    y_calibration_selection = calibration_selection_df[TARGET_COLUMN]

    selected_pipeline_for_calibration = _fit_pipeline(
        selected_candidate_pipeline,
        X=X_train,
        y_encoded=y_train,
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
        "selected_model_class": selected_candidate_pipeline.named_steps["model"].__class__.__name__,
        "deployed_model_variant": deployed_model_variant,
        "calibration_method": calibration_method,
        "metrics": metrics,
        "uncalibrated_metrics": uncalibrated_metrics,
        "evaluation_artifacts": {
            "selection_strategy": "temporal_backtesting_rank_sum",
            "selection_metrics": [
                "macro_f1",
                "balanced_accuracy",
                "matthews_corrcoef",
                "log_loss",
                "multiclass_brier_score",
                "expected_calibration_error",
            ],
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
