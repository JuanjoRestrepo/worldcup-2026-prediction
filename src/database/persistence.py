"""Utilities for persisting pipeline outputs into PostgreSQL."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import Literal

import pandas as pd
from sqlalchemy.engine import Engine

from src.contracts.data_contracts import (
    validate_persisted_dataframe_contract,
    validate_training_summary_contract,
)
from src.database.connection import get_sqlalchemy_engine
from src.modeling.types import TrainingSummary

IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_identifier(identifier: str) -> str:
    if not IDENTIFIER_PATTERN.match(identifier):
        raise ValueError(f"Invalid SQL identifier: '{identifier}'")
    return identifier


def ensure_schema(engine: Engine, schema_name: str) -> None:
    """Create a schema if it does not exist yet."""
    validated_schema = _validate_identifier(schema_name)
    with engine.begin() as connection:
        connection.exec_driver_sql(f'CREATE SCHEMA IF NOT EXISTS "{validated_schema}"')


def persist_dataframe(
    df: pd.DataFrame,
    *,
    schema_name: str,
    table_name: str,
    if_exists: Literal["fail", "replace", "append"] = "replace",
    engine: Engine | None = None,
    pipeline_run_id: str | None = None,
) -> None:
    """
    Persist a DataFrame into PostgreSQL.

    The persisted copy includes `pipeline_run_id` and `persisted_at_utc` so each
    table snapshot keeps lightweight lineage metadata.
    """
    validated_schema = _validate_identifier(schema_name)
    validated_table = _validate_identifier(table_name)
    sql_engine = engine or get_sqlalchemy_engine()

    persisted_df = df.copy()
    persisted_df["pipeline_run_id"] = pipeline_run_id
    persisted_df["persisted_at_utc"] = datetime.now(UTC).isoformat()
    validate_persisted_dataframe_contract(
        persisted_df,
        schema_name=validated_schema,
        table_name=validated_table,
    )

    ensure_schema(sql_engine, validated_schema)
    persisted_df.to_sql(
        name=validated_table,
        con=sql_engine,
        schema=validated_schema,
        if_exists=if_exists,
        index=False,
        method="multi",
        chunksize=1000,
    )


def build_training_run_frame(
    training_summary: TrainingSummary,
    *,
    pipeline_run_id: str | None = None,
) -> pd.DataFrame:
    """Convert nested training metadata into a one-row DataFrame for DB storage."""
    validate_training_summary_contract(training_summary)
    metrics = training_summary["metrics"]
    row = {
        "pipeline_run_id": pipeline_run_id,
        "artifact_path": training_summary["artifact_path"],
        "data_path": training_summary["data_path"],
        "training_rows": training_summary["training_rows"],
        "test_rows": training_summary["test_rows"],
        "feature_count": training_summary["feature_count"],
        "train_date_start": training_summary["train_date_range"]["start"],
        "train_date_end": training_summary["train_date_range"]["end"],
        "test_date_start": training_summary["test_date_range"]["start"],
        "test_date_end": training_summary["test_date_range"]["end"],
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "log_loss": metrics["log_loss"],
        "feature_columns_json": json.dumps(training_summary["feature_columns"]),
        "class_distribution_train_json": json.dumps(
            training_summary["class_distribution_train"]
        ),
        "class_distribution_test_json": json.dumps(
            training_summary["class_distribution_test"]
        ),
        "classification_report_json": json.dumps(metrics["classification_report"]),
        "trained_at_utc": datetime.now(UTC).isoformat(),
    }
    return pd.DataFrame([row])


def persist_training_run(
    training_summary: TrainingSummary,
    *,
    engine: Engine | None = None,
    pipeline_run_id: str | None = None,
) -> None:
    """Append a training run summary to `gold.training_runs`."""
    training_frame = build_training_run_frame(
        training_summary,
        pipeline_run_id=pipeline_run_id,
    )
    persist_dataframe(
        training_frame,
        schema_name="gold",
        table_name="training_runs",
        if_exists="append",
        engine=engine,
        pipeline_run_id=pipeline_run_id,
    )
