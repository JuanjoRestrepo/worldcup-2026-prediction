"""Serving and monitoring data loaders backed by dbt models and PostgreSQL."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Literal, cast

import pandas as pd
from sqlalchemy.exc import SQLAlchemyError

from src.config.settings import settings
from src.database.connection import get_sqlalchemy_engine

logger = logging.getLogger(__name__)

DBT_GOLD_TEAM_SNAPSHOTS_TABLE = "gold_latest_team_snapshots"
DBT_GOLD_TRAINING_RUN_TABLE = "gold_latest_training_run"
RAW_GOLD_TRAINING_RUNS_TABLE = "training_runs"
DBT_TEAM_SNAPSHOTS_SOURCE = "dbt_latest_team_snapshots"
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


def _serialize_training_run_row(row: pd.Series) -> dict[str, object]:
    serialized: dict[str, object] = {}
    for key, value in row.items():
        if isinstance(value, pd.Timestamp):
            serialized[key] = value.isoformat()
        elif hasattr(value, "isoformat") and not isinstance(value, str):
            serialized[key] = value.isoformat()
        elif value is None:
            serialized[key] = None
        elif isinstance(value, float) and pd.isna(value):
            serialized[key] = None
        else:
            serialized[key] = value
    return serialized


@lru_cache(maxsize=1)
def _load_latest_training_run_from_dbt_cached(
    db_host: str,
    db_port: int,
    db_name: str,
    db_user: str,
    dbt_base_schema: str,
) -> dict[str, object]:
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


def _load_latest_training_run_from_dbt() -> dict[str, object]:
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
) -> dict[str, object]:
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


def _load_latest_training_run_from_postgres() -> dict[str, object]:
    return _load_latest_training_run_from_postgres_cached(
        settings.DB_HOST,
        settings.DB_PORT,
        settings.DB_NAME,
        settings.DB_USER,
    )


def load_latest_training_run_summary_with_source(
    source: str = "auto",
) -> tuple[dict[str, object], str]:
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
    _load_latest_training_run_from_dbt_cached.cache_clear()
    _load_latest_training_run_from_postgres_cached.cache_clear()
