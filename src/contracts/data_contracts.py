"""Formal dataset contracts for the World Cup prediction pipeline."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd


class DataContractError(ValueError):
    """Raised when a DataFrame or training artifact violates a formal contract."""


RAW_HISTORICAL_REQUIRED_COLUMNS = (
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "tournament",
)

STANDARDIZED_MATCH_REQUIRED_COLUMNS = (
    "date",
    "homeTeam",
    "awayTeam",
    "homeGoals",
    "awayGoals",
    "tournament",
)

FEATURE_DATASET_REQUIRED_COLUMNS = (
    "date",
    "homeTeam",
    "awayTeam",
    "homeGoals",
    "awayGoals",
    "elo_home",
    "elo_away",
    "elo_diff",
    "target_multiclass",
    "target",
)

TRAINING_RUN_FRAME_REQUIRED_COLUMNS = (
    "artifact_path",
    "data_path",
    "training_rows",
    "test_rows",
    "feature_count",
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "log_loss",
    "trained_at_utc",
    "pipeline_run_id",
    "persisted_at_utc",
)

ALLOWED_MULTICLASS_TARGETS = {-1, 0, 1}
ALLOWED_BINARY_TARGETS = {0, 1}


def _require_columns(
    df: pd.DataFrame,
    required_columns: Iterable[str],
    *,
    contract_name: str,
) -> None:
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise DataContractError(
            f"{contract_name} contract failed: missing columns {missing_columns}."
        )


def _require_non_empty(df: pd.DataFrame, *, contract_name: str) -> None:
    if df.empty:
        raise DataContractError(f"{contract_name} contract failed: dataframe is empty.")


def _require_non_null(
    df: pd.DataFrame,
    columns: Iterable[str],
    *,
    contract_name: str,
) -> None:
    null_counts = df[list(columns)].isna().sum()
    failing_columns = {
        column: int(count)
        for column, count in null_counts.items()
        if int(count) > 0
    }
    if failing_columns:
        raise DataContractError(
            f"{contract_name} contract failed: null values found in required columns {failing_columns}."
        )


def _require_parseable_dates(
    df: pd.DataFrame,
    column: str,
    *,
    contract_name: str,
) -> None:
    parsed = pd.to_datetime(df[column], errors="coerce")
    if parsed.isna().any():
        invalid_count = int(parsed.isna().sum())
        raise DataContractError(
            f"{contract_name} contract failed: column '{column}' contains {invalid_count} invalid date values."
        )


def _require_numeric_minimum(
    df: pd.DataFrame,
    columns: Iterable[str],
    minimum: float,
    *,
    contract_name: str,
) -> None:
    for column in columns:
        numeric_values = pd.to_numeric(df[column], errors="coerce")
        if numeric_values.isna().any():
            raise DataContractError(
                f"{contract_name} contract failed: column '{column}' contains non-numeric values."
            )
        if (numeric_values < minimum).any():
            raise DataContractError(
                f"{contract_name} contract failed: column '{column}' contains values below {minimum}."
            )


def _require_allowed_values(
    df: pd.DataFrame,
    column: str,
    allowed_values: set[Any],
    *,
    contract_name: str,
) -> None:
    invalid_values = sorted(
        value for value in df[column].dropna().unique() if value not in allowed_values
    )
    if invalid_values:
        raise DataContractError(
            f"{contract_name} contract failed: column '{column}' contains invalid values {invalid_values}."
        )


def validate_historical_raw_contract(df: pd.DataFrame) -> None:
    """Validate the raw historical CSV contract before standardization."""
    contract_name = "historical_raw"
    _require_non_empty(df, contract_name=contract_name)
    _require_columns(df, RAW_HISTORICAL_REQUIRED_COLUMNS, contract_name=contract_name)
    _require_non_null(
        df,
        ("date", "home_team", "away_team", "tournament"),
        contract_name=contract_name,
    )
    _require_parseable_dates(df, "date", contract_name=contract_name)
    _require_numeric_minimum(
        df,
        ("home_score", "away_score"),
        0,
        contract_name=contract_name,
    )


def validate_standardized_matches_contract(
    df: pd.DataFrame,
    *,
    contract_name: str = "standardized_matches",
) -> None:
    """Validate a standardized bronze or silver match table."""
    _require_non_empty(df, contract_name=contract_name)
    _require_columns(
        df,
        STANDARDIZED_MATCH_REQUIRED_COLUMNS,
        contract_name=contract_name,
    )
    _require_non_null(
        df,
        ("date", "homeTeam", "awayTeam", "tournament"),
        contract_name=contract_name,
    )
    _require_parseable_dates(df, "date", contract_name=contract_name)
    _require_numeric_minimum(
        df,
        ("homeGoals", "awayGoals"),
        0,
        contract_name=contract_name,
    )


def validate_feature_dataset_contract(df: pd.DataFrame) -> None:
    """Validate the gold feature dataset before persistence or training."""
    contract_name = "feature_dataset"
    _require_non_empty(df, contract_name=contract_name)
    _require_columns(df, FEATURE_DATASET_REQUIRED_COLUMNS, contract_name=contract_name)
    _require_non_null(
        df,
        (
            "date",
            "homeTeam",
            "awayTeam",
            "homeGoals",
            "awayGoals",
            "elo_home",
            "elo_away",
            "elo_diff",
            "target_multiclass",
            "target",
        ),
        contract_name=contract_name,
    )
    _require_parseable_dates(df, "date", contract_name=contract_name)
    _require_numeric_minimum(
        df,
        ("homeGoals", "awayGoals"),
        0,
        contract_name=contract_name,
    )
    _require_allowed_values(
        df,
        "target_multiclass",
        ALLOWED_MULTICLASS_TARGETS,
        contract_name=contract_name,
    )
    _require_allowed_values(
        df,
        "target",
        ALLOWED_BINARY_TARGETS,
        contract_name=contract_name,
    )


def validate_training_summary_contract(training_summary: dict[str, Any]) -> None:
    """Validate the training summary payload before persistence."""
    required_keys = {
        "artifact_path",
        "data_path",
        "training_rows",
        "test_rows",
        "feature_count",
        "feature_columns",
        "train_date_range",
        "test_date_range",
        "class_distribution_train",
        "class_distribution_test",
        "metrics",
    }
    missing_keys = sorted(required_keys.difference(training_summary))
    if missing_keys:
        raise DataContractError(
            f"training_summary contract failed: missing keys {missing_keys}."
        )

    metrics = training_summary["metrics"]
    for metric_name in ("accuracy", "macro_f1", "weighted_f1"):
        metric_value = metrics.get(metric_name)
        if metric_value is None or not 0 <= float(metric_value) <= 1:
            raise DataContractError(
                f"training_summary contract failed: metric '{metric_name}' must be between 0 and 1."
            )
    if float(metrics.get("log_loss", -1)) < 0:
        raise DataContractError(
            "training_summary contract failed: metric 'log_loss' must be non-negative."
        )
    if int(training_summary["training_rows"]) <= 0 or int(training_summary["test_rows"]) <= 0:
        raise DataContractError(
            "training_summary contract failed: training_rows and test_rows must be positive."
        )
    if int(training_summary["feature_count"]) <= 0:
        raise DataContractError(
            "training_summary contract failed: feature_count must be positive."
        )
    if not training_summary["feature_columns"]:
        raise DataContractError(
            "training_summary contract failed: feature_columns cannot be empty."
        )


def validate_training_run_frame_contract(df: pd.DataFrame) -> None:
    """Validate the persisted training run frame before database storage."""
    contract_name = "training_run_frame"
    _require_non_empty(df, contract_name=contract_name)
    _require_columns(
        df,
        TRAINING_RUN_FRAME_REQUIRED_COLUMNS,
        contract_name=contract_name,
    )
    _require_non_null(
        df,
        (
            "artifact_path",
            "data_path",
            "training_rows",
            "test_rows",
            "feature_count",
            "accuracy",
            "macro_f1",
            "weighted_f1",
            "log_loss",
            "trained_at_utc",
            "persisted_at_utc",
        ),
        contract_name=contract_name,
    )
    _require_parseable_dates(df, "trained_at_utc", contract_name=contract_name)
    _require_parseable_dates(df, "persisted_at_utc", contract_name=contract_name)


def validate_persisted_dataframe_contract(
    df: pd.DataFrame,
    *,
    schema_name: str,
    table_name: str,
) -> None:
    """Validate persisted datasets using schema/table-aware contracts."""
    contract_map = {
        ("bronze", "historical_matches"): validate_standardized_matches_contract,
        ("bronze", "api_matches"): validate_standardized_matches_contract,
        ("silver", "matches_cleaned"): validate_standardized_matches_contract,
        ("gold", "features_dataset"): validate_feature_dataset_contract,
        ("gold", "training_runs"): validate_training_run_frame_contract,
    }
    validator = contract_map.get((schema_name, table_name))
    if validator is None:
        return
    validator(df)
