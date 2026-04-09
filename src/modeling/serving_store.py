"""Serving and monitoring data loaders backed by dbt models and PostgreSQL."""

from __future__ import annotations

from datetime import date
import logging
from functools import lru_cache
from typing import Literal, cast

import pandas as pd
from sqlalchemy.exc import SQLAlchemyError

from src.config.settings import settings
from src.database.connection import get_sqlalchemy_engine
from src.modeling.types import LatestTrainingRunSummary

logger = logging.getLogger(__name__)

DBT_GOLD_TEAM_SNAPSHOTS_TABLE = "gold_latest_team_snapshots"
DBT_GOLD_TEAM_HISTORY_TABLE = "gold_team_feature_snapshots"
DBT_GOLD_TRAINING_RUN_TABLE = "gold_latest_training_run"
RAW_GOLD_TRAINING_RUNS_TABLE = "training_runs"
DBT_TEAM_SNAPSHOTS_SOURCE = "dbt_latest_team_snapshots"
DBT_TEAM_SNAPSHOTS_AT_DATE_SOURCE = "dbt_team_snapshots_as_of_date"
DBT_TRAINING_RUN_SOURCE = "dbt_latest_training_run"
RAW_TRAINING_RUN_SOURCE = "postgres_training_runs"
MONITORING_SOURCE_OPTIONS = {"auto", "dbt", "postgres"}

MonitoringSource = Literal["auto", "dbt", "postgres"]


def _dbt_schema(layer_name: str) -> str:
    return f"{settings.DBT_BASE_SCHEMA}_{layer_name}"


def _normalize_monitoring_source(source: str) -> MonitoringSource:
    normalized_source = source.strip().lower()
    if normalized_source not in MONITORING_SOURCE_OPTIONS:
        raise ValueError("monitoring source must be one of: auto, dbt, postgres.")
    return cast(MonitoringSource, normalized_source)


def _read_relation(query: str) -> pd.DataFrame:
    engine = get_sqlalchemy_engine()
    try:
        return pd.read_sql_query(query, con=engine)
    finally:
        engine.dispose()


@lru_cache(maxsize=1)
def _load_latest_team_snapshots_from_dbt_cached(
    db_host: str,
    db_port: int,
    db_name: str,
    db_user: str,
    dbt_base_schema: str,
) -> pd.DataFrame:
    df = _read_relation(
        f'SELECT * FROM "{dbt_base_schema}_gold"."{DBT_GOLD_TEAM_SNAPSHOTS_TABLE}"'
    )
    if df.empty:
        raise RuntimeError(
            f"dbt model {dbt_base_schema}_gold.{DBT_GOLD_TEAM_SNAPSHOTS_TABLE} is empty."
        )

    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    df["persisted_at_utc"] = pd.to_datetime(df["persisted_at_utc"], utc=True)
    return df.sort_values(["team", "team_role", "snapshot_date"]).reset_index(drop=True)


def load_latest_team_snapshots_from_dbt() -> pd.DataFrame:
    """Load the dbt-curated latest team snapshot model used for serving."""
    return _load_latest_team_snapshots_from_dbt_cached(
        settings.DB_HOST,
        settings.DB_PORT,
        settings.DB_NAME,
        settings.DB_USER,
        settings.DBT_BASE_SCHEMA,
    )


@lru_cache(maxsize=32)
def _load_team_snapshots_as_of_date_from_dbt_cached(
    db_host: str,
    db_port: int,
    db_name: str,
    db_user: str,
    dbt_base_schema: str,
    match_date_iso: str,
) -> pd.DataFrame:
    df = _read_relation(
        f"""
        WITH filtered_snapshots AS (
            SELECT *
            FROM "{dbt_base_schema}_gold"."{DBT_GOLD_TEAM_HISTORY_TABLE}"
            WHERE snapshot_date <= DATE '{match_date_iso}'
        ),
        latest_role_snapshots AS (
            SELECT
                snapshot_date,
                team,
                opponent,
                team_role,
                elo,
                opponent_elo,
                avg_goals_last5,
                avg_goals_conceded_last5,
                global_avg_goals_last5,
                global_avg_conceded_last5,
                win_rate_last5,
                global_win_rate_last5,
                avg_opponent_elo_last5,
                weighted_win_rate_last5,
                opponent_elo_form,
                elo_form,
                home_advantage_effect,
                is_friendly,
                is_world_cup,
                is_qualifier,
                is_continental,
                pipeline_run_id,
                persisted_at_utc
            FROM (
                SELECT
                    *,
                    ROW_NUMBER() OVER (
                        PARTITION BY team, team_role
                        ORDER BY snapshot_date DESC, persisted_at_utc DESC
                    ) AS row_num
                FROM filtered_snapshots
            ) ranked
            WHERE row_num = 1
        ),
        latest_overall_snapshots AS (
            SELECT
                snapshot_date,
                team,
                opponent,
                'overall' AS team_role,
                elo,
                opponent_elo,
                avg_goals_last5,
                avg_goals_conceded_last5,
                global_avg_goals_last5,
                global_avg_conceded_last5,
                win_rate_last5,
                global_win_rate_last5,
                avg_opponent_elo_last5,
                weighted_win_rate_last5,
                opponent_elo_form,
                elo_form,
                home_advantage_effect,
                is_friendly,
                is_world_cup,
                is_qualifier,
                is_continental,
                pipeline_run_id,
                persisted_at_utc
            FROM (
                SELECT
                    *,
                    ROW_NUMBER() OVER (
                        PARTITION BY team
                        ORDER BY snapshot_date DESC, persisted_at_utc DESC
                    ) AS row_num
                FROM filtered_snapshots
            ) ranked
            WHERE row_num = 1
        )
        SELECT * FROM latest_role_snapshots
        UNION ALL
        SELECT * FROM latest_overall_snapshots
        """
    )
    if df.empty:
        raise RuntimeError(
            "dbt team snapshot history is empty for the requested match_date."
        )

    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    df["persisted_at_utc"] = pd.to_datetime(df["persisted_at_utc"], utc=True)
    return df.sort_values(["team", "team_role", "snapshot_date"]).reset_index(drop=True)


def load_team_snapshots_as_of_date_from_dbt(match_date: date) -> tuple[pd.DataFrame, str]:
    """Load team snapshots as of a historical match date from the dbt serving model."""
    snapshots = _load_team_snapshots_as_of_date_from_dbt_cached(
        settings.DB_HOST,
        settings.DB_PORT,
        settings.DB_NAME,
        settings.DB_USER,
        settings.DBT_BASE_SCHEMA,
        match_date.isoformat(),
    )
    return snapshots, DBT_TEAM_SNAPSHOTS_AT_DATE_SOURCE


def _serialize_training_run_value(value: object) -> str | int | float | None:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "isoformat") and not isinstance(value, str):
        iso_value = getattr(value, "isoformat")()
        return str(iso_value)
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, (str, int, float)):
        return value
    return str(value)


def _serialize_optional_string(value: object) -> str | None:
    serialized_value = _serialize_training_run_value(value)
    if serialized_value is None:
        return None
    return str(serialized_value)


def _serialize_training_run_row(row: pd.Series) -> LatestTrainingRunSummary:
    return {
        "pipeline_run_id": _serialize_optional_string(row["pipeline_run_id"]),
        "artifact_path": str(_serialize_training_run_value(row["artifact_path"])),
        "data_path": str(_serialize_training_run_value(row["data_path"])),
        "training_rows": int(row["training_rows"]),
        "test_rows": int(row["test_rows"]),
        "feature_count": int(row["feature_count"]),
        "train_date_start": str(_serialize_training_run_value(row["train_date_start"])),
        "train_date_end": str(_serialize_training_run_value(row["train_date_end"])),
        "test_date_start": str(_serialize_training_run_value(row["test_date_start"])),
        "test_date_end": str(_serialize_training_run_value(row["test_date_end"])),
        "accuracy": float(row["accuracy"]),
        "macro_f1": float(row["macro_f1"]),
        "weighted_f1": float(row["weighted_f1"]),
        "log_loss": float(row["log_loss"]),
        "trained_at_utc": str(_serialize_training_run_value(row["trained_at_utc"])),
        "persisted_at_utc": _serialize_optional_string(row["persisted_at_utc"]),
    }


@lru_cache(maxsize=1)
def _load_latest_training_run_from_dbt_cached(
    db_host: str,
    db_port: int,
    db_name: str,
    db_user: str,
    dbt_base_schema: str,
) -> LatestTrainingRunSummary:
    df = _read_relation(
        f"""
        SELECT
            pipeline_run_id,
            artifact_path,
            data_path,
            training_rows,
            test_rows,
            feature_count,
            train_date_start,
            train_date_end,
            test_date_start,
            test_date_end,
            accuracy,
            macro_f1,
            weighted_f1,
            log_loss,
            trained_at_utc,
            persisted_at_utc
        FROM "{dbt_base_schema}_gold"."{DBT_GOLD_TRAINING_RUN_TABLE}"
        """
    )
    if df.empty:
        raise RuntimeError(
            f"dbt model {dbt_base_schema}_gold.{DBT_GOLD_TRAINING_RUN_TABLE} is empty."
        )
    return _serialize_training_run_row(df.iloc[0])


def _load_latest_training_run_from_dbt() -> LatestTrainingRunSummary:
    return _load_latest_training_run_from_dbt_cached(
        settings.DB_HOST,
        settings.DB_PORT,
        settings.DB_NAME,
        settings.DB_USER,
        settings.DBT_BASE_SCHEMA,
    )


@lru_cache(maxsize=1)
def _load_latest_training_run_from_postgres_cached(
    db_host: str,
    db_port: int,
    db_name: str,
    db_user: str,
) -> LatestTrainingRunSummary:
    df = _read_relation(
        f"""
        SELECT
            pipeline_run_id,
            artifact_path,
            data_path,
            training_rows,
            test_rows,
            feature_count,
            train_date_start,
            train_date_end,
            test_date_start,
            test_date_end,
            accuracy,
            macro_f1,
            weighted_f1,
            log_loss,
            trained_at_utc,
            persisted_at_utc
        FROM "gold"."{RAW_GOLD_TRAINING_RUNS_TABLE}"
        ORDER BY trained_at_utc DESC NULLS LAST, persisted_at_utc DESC NULLS LAST
        LIMIT 1
        """
    )
    if df.empty:
        raise RuntimeError(
            f"PostgreSQL table gold.{RAW_GOLD_TRAINING_RUNS_TABLE} is empty."
        )
    return _serialize_training_run_row(df.iloc[0])


def _load_latest_training_run_from_postgres() -> LatestTrainingRunSummary:
    return _load_latest_training_run_from_postgres_cached(
        settings.DB_HOST,
        settings.DB_PORT,
        settings.DB_NAME,
        settings.DB_USER,
    )


def load_latest_training_run_summary_with_source(
    source: str = "auto",
) -> tuple[LatestTrainingRunSummary, str]:
    """Load the latest training run summary for monitoring."""
    normalized_source = _normalize_monitoring_source(source)

    if normalized_source == "dbt":
        try:
            return _load_latest_training_run_from_dbt(), DBT_TRAINING_RUN_SOURCE
        except (RuntimeError, SQLAlchemyError, ValueError) as exc:
            raise RuntimeError(
                "Failed to load dbt latest training run for monitoring."
            ) from exc

    if normalized_source == "postgres":
        try:
            return _load_latest_training_run_from_postgres(), RAW_TRAINING_RUN_SOURCE
        except (RuntimeError, SQLAlchemyError, ValueError) as exc:
            raise RuntimeError(
                "Failed to load latest training run from PostgreSQL."
            ) from exc

    try:
        return _load_latest_training_run_from_dbt(), DBT_TRAINING_RUN_SOURCE
    except (RuntimeError, SQLAlchemyError, ValueError) as exc:
        logger.warning(
            "Falling back to raw PostgreSQL training_runs after dbt monitoring load failed: %s",
            exc,
        )
        return _load_latest_training_run_from_postgres(), RAW_TRAINING_RUN_SOURCE


def clear_serving_store_cache() -> None:
    """Clear cached dbt/raw serving and monitoring snapshots."""
    _load_latest_team_snapshots_from_dbt_cached.cache_clear()
    _load_team_snapshots_as_of_date_from_dbt_cached.cache_clear()
    _load_latest_training_run_from_dbt_cached.cache_clear()
    _load_latest_training_run_from_postgres_cached.cache_clear()
