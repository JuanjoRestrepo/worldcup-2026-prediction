"""Train and export the production match outcome model."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from src.config.settings import settings
from src.database.persistence import persist_training_run
from src.modeling.features import OUTCOME_LABELS, TARGET_COLUMN, load_feature_dataset
from src.modeling.features import select_model_feature_columns

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.2
OUTCOME_TO_ENCODED = {-1: 0, 0: 1, 1: 2}
ENCODED_TO_OUTCOME = {value: key for key, value in OUTCOME_TO_ENCODED.items()}


def _build_training_pipeline() -> Pipeline:
    """Build the training pipeline for the production model."""
    model = XGBClassifier(
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
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", model),
        ]
    )


def _split_temporal_holdout(
    df: pd.DataFrame, test_size: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the feature dataset into chronological train and test windows."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    ordered_df = df.sort_values("date").reset_index(drop=True)
    split_idx = int(len(ordered_df) * (1 - test_size))
    split_idx = max(1, min(split_idx, len(ordered_df) - 1))

    train_df = ordered_df.iloc[:split_idx].copy()
    test_df = ordered_df.iloc[split_idx:].copy()
    return train_df, test_df


def _summarize_metrics(
    y_true: pd.Series,
    y_true_encoded: pd.Series,
    y_pred_encoded: pd.Series,
    probabilities,
) -> dict[str, object]:
    y_pred = y_pred_encoded.map(ENCODED_TO_OUTCOME)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "log_loss": float(
            log_loss(y_true_encoded, probabilities, labels=[0, 1, 2])
        ),
        "classification_report": classification_report(
            y_true,
            y_pred,
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


def train_and_export_model(
    data_path: Path | None = None,
    artifact_path: Path | None = None,
    test_size: float = DEFAULT_TEST_SIZE,
    persist_to_db: bool = False,
    pipeline_run_id: str | None = None,
) -> dict[str, object]:
    """
    Train the production model from the gold feature dataset and export it.

    Args:
        data_path: Optional alternate path to the gold feature dataset
        artifact_path: Optional target path for the exported joblib artifact
        test_size: Fraction of the most recent data reserved for evaluation
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
    feature_columns = select_model_feature_columns(df)
    if not feature_columns:
        raise RuntimeError("No model feature columns were found in the gold dataset.")

    train_df, test_df = _split_temporal_holdout(df, test_size=test_size)
    logger.info(
        "Temporal split complete: train=%s rows, test=%s rows",
        len(train_df),
        len(test_df),
    )

    X_train = train_df[feature_columns].copy()
    X_test = test_df[feature_columns].copy()
    y_train = train_df[TARGET_COLUMN].map(OUTCOME_TO_ENCODED)
    y_test = test_df[TARGET_COLUMN]
    y_test_encoded = y_test.map(OUTCOME_TO_ENCODED)

    pipeline = _build_training_pipeline()
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    logger.info("Training XGBoost pipeline with %s features", len(feature_columns))
    pipeline.fit(X_train, y_train, model__sample_weight=sample_weight)

    predicted_encoded = pd.Series(
        pipeline.predict(X_test), index=X_test.index, dtype="int64"
    )
    probabilities = pipeline.predict_proba(X_test)
    metrics = _summarize_metrics(
        y_true=y_test,
        y_true_encoded=y_test_encoded,
        y_pred_encoded=predicted_encoded,
        probabilities=probabilities,
    )

    training_summary = {
        "artifact_path": str(output_path),
        "training_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "feature_count": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "data_path": str(dataset_path),
        "train_date_range": {
            "start": train_df["date"].min().date().isoformat(),
            "end": train_df["date"].max().date().isoformat(),
        },
        "test_date_range": {
            "start": test_df["date"].min().date().isoformat(),
            "end": test_df["date"].max().date().isoformat(),
        },
        "class_distribution_train": {
            str(label): int(count)
            for label, count in train_df[TARGET_COLUMN].value_counts().sort_index().items()
        },
        "class_distribution_test": {
            str(label): int(count)
            for label, count in test_df[TARGET_COLUMN].value_counts().sort_index().items()
        },
        "metrics": metrics,
    }

    artifact = {
        "model": pipeline,
        "feature_columns": feature_columns,
        "target_column": TARGET_COLUMN,
        "outcome_to_encoded": OUTCOME_TO_ENCODED,
        "encoded_to_outcome": ENCODED_TO_OUTCOME,
        "outcome_labels": OUTCOME_LABELS,
        "training_summary": training_summary,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)
    logger.info("Model artifact exported to %s", output_path)
    if persist_to_db:
        persist_training_run(
            training_summary,
            pipeline_run_id=pipeline_run_id,
        )
        logger.info("Training metadata appended to gold.training_runs")
    logger.info(
        "Evaluation metrics | accuracy=%.4f macro_f1=%.4f weighted_f1=%.4f",
        metrics["accuracy"],
        metrics["macro_f1"],
        metrics["weighted_f1"],
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
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    args = _parse_args()
    summary = train_and_export_model(
        data_path=args.data_path,
        artifact_path=args.artifact_path,
        test_size=args.test_size,
    )
    print(json.dumps(summary, indent=2))
