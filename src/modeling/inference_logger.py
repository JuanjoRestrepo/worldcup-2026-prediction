"""Inference logging and monitoring for prediction serving."""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone
from typing import Any, cast

import pandas as pd
from sqlalchemy.engine import Engine

from src.database.connection import get_sqlalchemy_engine
from src.database.persistence import ensure_schema
from src.modeling.types import FeatureSnapshotDates

logger = logging.getLogger(__name__)

INFERENCE_LOG_SCHEMA = "monitoring"
INFERENCE_LOG_TABLE = "inference_logs"


def validate_feature_freshness(
    feature_dates: FeatureSnapshotDates | dict[str, str],
    max_age_days: int = 30,
) -> dict[str, Any]:
    """
    Validate that features are fresh (not older than max_age_days).

    Args:
        feature_dates: Dict mapping feature names/teams to ISO date strings
        max_age_days: Maximum acceptable age in days (default: 30)

    Returns:
        Dict with 'is_fresh' (bool), 'warning' (str or None), 'age_days' (dict)
    """
    now = datetime.now(timezone.utc)
    ages = {}
    warnings = []
    is_fresh = True

    for feature_name, raw_date_value in feature_dates.items():
        date_str = str(raw_date_value)
        try:
            # Parse ISO format date (handles both date and datetime)
            if "T" in date_str:
                feature_date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            else:
                feature_date = datetime.fromisoformat(date_str + "T00:00:00+00:00")

            age_days = (now - feature_date).days
            ages[feature_name] = age_days

            if age_days > max_age_days:
                is_fresh = False
                warnings.append(f"{feature_name}: {age_days} days old")
        except (ValueError, TypeError) as exc:
            logger.warning(
                f"Could not parse feature date for {feature_name}: {date_str} ({exc})"
            )

    return {
        "is_fresh": is_fresh,
        "warning": " | ".join(warnings) if warnings else None,
        "age_days": ages,
    }


class InferenceLogger:
    """Centralized logging for model inference requests and predictions."""

    def __init__(self, engine: Engine | None = None) -> None:
        """Initialize with optional SQLAlchemy engine."""
        self.engine = engine or get_sqlalchemy_engine()

    def _ensure_inference_log_table(self) -> None:
        """Create or align the inference logging table for current monitoring fields."""
        ensure_schema(self.engine, INFERENCE_LOG_SCHEMA)
        with self.engine.begin() as connection:
            connection.exec_driver_sql(
                f"""
                CREATE TABLE IF NOT EXISTS "{INFERENCE_LOG_SCHEMA}"."{INFERENCE_LOG_TABLE}" (
                    id SERIAL PRIMARY KEY,
                    request_id VARCHAR(255) NOT NULL,
                    timestamp_utc TIMESTAMP WITH TIME ZONE NOT NULL,
                    requested_match_date DATE,
                    home_team VARCHAR(255) NOT NULL,
                    away_team VARCHAR(255) NOT NULL,
                    neutral BOOLEAN NOT NULL,
                    tournament VARCHAR(255),
                    predicted_class INTEGER NOT NULL,
                    predicted_outcome VARCHAR(50) NOT NULL,
                    class_probabilities_json JSONB NOT NULL,
                    feature_snapshot_dates_json JSONB NOT NULL,
                    feature_source VARCHAR(100) NOT NULL,
                    model_artifact_path TEXT NOT NULL,
                    model_version VARCHAR(100),
                    match_segment VARCHAR(100),
                    is_override_triggered BOOLEAN DEFAULT FALSE,
                    persisted_at_utc TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            # Add columns if they don't exist (for backward compatibility with existing tables)
            connection.exec_driver_sql(
                f"""
                ALTER TABLE "{INFERENCE_LOG_SCHEMA}"."{INFERENCE_LOG_TABLE}"
                ADD COLUMN IF NOT EXISTS match_segment VARCHAR(100)
                """
            )
            connection.exec_driver_sql(
                f"""
                ALTER TABLE "{INFERENCE_LOG_SCHEMA}"."{INFERENCE_LOG_TABLE}"
                ADD COLUMN IF NOT EXISTS is_override_triggered BOOLEAN DEFAULT FALSE
                """
            )
            # Add shadow deployment columns
            shadow_columns = [
                "shadow_predicted_outcome VARCHAR(50)",
                "shadow_class_probabilities_json JSONB",
                "shadow_is_override_triggered BOOLEAN DEFAULT FALSE",
                "shadow_model_name VARCHAR(255)"
            ]
            for col_def in shadow_columns:
                connection.exec_driver_sql(
                    f"""
                    ALTER TABLE "{INFERENCE_LOG_SCHEMA}"."{INFERENCE_LOG_TABLE}"
                    ADD COLUMN IF NOT EXISTS {col_def}
                    """
                )
            connection.exec_driver_sql(
                f"""
                ALTER TABLE "{INFERENCE_LOG_SCHEMA}"."{INFERENCE_LOG_TABLE}"
                ADD COLUMN IF NOT EXISTS requested_match_date DATE
                """
            )

    def log_prediction(
        self,
        home_team: str,
        away_team: str,
        predicted_class: int,
        predicted_outcome: str,
        class_probabilities: dict[str, float],
        neutral: bool,
        tournament: str | None,
        feature_snapshot_dates: FeatureSnapshotDates | dict[str, str],
        feature_source: str,
        model_artifact_path: str,
        model_version: str | None = None,
        requested_match_date: date | None = None,
        request_timestamp_utc: datetime | None = None,
        match_segment: str | None = None,
        is_override_triggered: bool | None = None,
        shadow_predicted_outcome: str | None = None,
        shadow_class_probabilities: dict[str, float] | None = None,
        shadow_is_override_triggered: bool | None = None,
        shadow_model_name: str | None = None,
    ) -> None:
        """
        Log a single prediction to PostgreSQL.

        Args:
            home_team: Home team name
            away_team: Away team name
            predicted_class: Encoded prediction (0, 1, 2)
            predicted_outcome: Human-readable outcome (home_win, away_win, draw)
            class_probabilities: Dict mapping outcomes to probabilities
            neutral: Whether match is on neutral ground
            tournament: Optional tournament name
            feature_snapshot_dates: Dict of snapshot dates for features
            feature_source: Source of features (dbt, postgres, csv)
            model_artifact_path: Path to model artifact used
            model_version: Optional model version identifier
            request_timestamp_utc: Optional request timestamp (defaults to now)
            match_segment: Optional segment detected by ensemble (friendlies, worldcup, etc.)
            is_override_triggered: Optional flag indicating specialist override
        """
        timestamp = request_timestamp_utc or datetime.now(timezone.utc)

        log_row = {
            "request_id": f"{home_team}_{away_team}_{timestamp.timestamp()}",
            "timestamp_utc": timestamp.isoformat(),
            "requested_match_date": (
                requested_match_date.isoformat()
                if requested_match_date is not None
                else None
            ),
            "home_team": home_team,
            "away_team": away_team,
            "neutral": neutral,
            "tournament": tournament,
            "predicted_class": predicted_class,
            "predicted_outcome": predicted_outcome,
            "class_probabilities_json": json.dumps(class_probabilities),
            "feature_snapshot_dates_json": json.dumps(dict(feature_snapshot_dates)),
            "feature_source": feature_source,
            "model_artifact_path": str(model_artifact_path),
            "model_version": model_version or "unknown",
            "match_segment": match_segment,
            "is_override_triggered": is_override_triggered or False,
            "shadow_predicted_outcome": shadow_predicted_outcome,
            "shadow_class_probabilities_json": json.dumps(shadow_class_probabilities) if shadow_class_probabilities else None,
            "shadow_is_override_triggered": shadow_is_override_triggered,
            "shadow_model_name": shadow_model_name,
            "persisted_at_utc": datetime.now(timezone.utc).isoformat(),
        }

        df = pd.DataFrame([log_row])
        self._persist_log(df)

    def _persist_log(self, df: pd.DataFrame) -> None:
        """Persist a log DataFrame to PostgreSQL."""
        try:
            self._ensure_inference_log_table()
            df.to_sql(
                name=INFERENCE_LOG_TABLE,
                con=self.engine,
                schema=INFERENCE_LOG_SCHEMA,
                if_exists="append",
                index=False,
                method="multi",
                chunksize=1000,
            )
            logger.debug(
                f"Logged {len(df)} inference(s) to {INFERENCE_LOG_SCHEMA}.{INFERENCE_LOG_TABLE}"
            )
        except Exception as exc:
            logger.error(f"Failed to persist inference log: {exc}")
            # Don't raise - inference should not fail due to logging error
            # but we should track this in observability

    def get_inference_statistics(
        self,
        hours: int = 24,
        limit: int = 1000,
    ) -> dict[str, Any]:
        """
        Retrieve inference statistics for monitoring dashboard.

        Args:
            hours: Number of hours to look back
            limit: Maximum number of recent inferences to retrieve

        Returns:
            Dictionary with aggregated statistics
        """
        query = f"""
        SELECT 
            COUNT(*) as total_inferences,
            COUNT(DISTINCT home_team || '_' || away_team) as unique_matchups,
            COUNT(DISTINCT feature_source) as feature_sources_used,
            AVG(CAST(class_probabilities_json->>'home_win' AS FLOAT)) as avg_home_win_prob,
            AVG(CAST(class_probabilities_json->>'away_win' AS FLOAT)) as avg_away_win_prob,
            AVG(CAST(class_probabilities_json->>'draw' AS FLOAT)) as avg_draw_prob,
            SUM(CASE WHEN requested_match_date IS NOT NULL THEN 1 ELSE 0 END) as historical_requests,
            SUM(CASE WHEN requested_match_date IS NULL THEN 1 ELSE 0 END) as latest_requests,
            SUM(CASE WHEN predicted_outcome = 'home_win' THEN 1 ELSE 0 END) as home_wins_predicted,
            SUM(CASE WHEN predicted_outcome = 'away_win' THEN 1 ELSE 0 END) as away_wins_predicted,
            SUM(CASE WHEN predicted_outcome = 'draw' THEN 1 ELSE 0 END) as draws_predicted,
            SUM(CASE WHEN shadow_predicted_outcome IS NOT NULL AND predicted_outcome = shadow_predicted_outcome THEN 1 ELSE 0 END) as shadow_agreement_count,
            SUM(CASE WHEN shadow_predicted_outcome = 'draw' THEN 1 ELSE 0 END) as shadow_draws_predicted,
            SUM(CASE WHEN shadow_is_override_triggered = TRUE THEN 1 ELSE 0 END) as shadow_overrides_triggered,
            COUNT(DISTINCT tournament) as tournaments_predicted,
            MIN(timestamp_utc) as earliest_request,
            MAX(timestamp_utc) as latest_request
        FROM "{INFERENCE_LOG_SCHEMA}"."{INFERENCE_LOG_TABLE}"
        WHERE timestamp_utc > NOW() - INTERVAL '{hours} hours'
        """

        try:
            df = pd.read_sql_query(query, con=self.engine)
            if df.empty:
                return {
                    "status": "no_data",
                    "message": f"No inferences in last {hours} hours",
                }

            stats = df.iloc[0].to_dict()
            # Convert numeric columns properly
            for key in stats:
                if pd.isna(stats[key]):
                    stats[key] = None
                elif key.startswith("avg_"):
                    stats[key] = (
                        round(float(stats[key]), 4) if stats[key] is not None else None
                    )
                    
            # Segment-level shadow performance
            segment_query = f"""
            SELECT 
                match_segment,
                COUNT(*) as segment_inferences,
                SUM(CASE WHEN shadow_predicted_outcome IS NOT NULL AND predicted_outcome = shadow_predicted_outcome THEN 1 ELSE 0 END) as shadow_agreement_count,
                SUM(CASE WHEN shadow_predicted_outcome = 'draw' THEN 1 ELSE 0 END) as shadow_draws_predicted,
                SUM(CASE WHEN shadow_is_override_triggered = TRUE THEN 1 ELSE 0 END) as shadow_overrides_triggered
            FROM "{INFERENCE_LOG_SCHEMA}"."{INFERENCE_LOG_TABLE}"
            WHERE timestamp_utc > NOW() - INTERVAL '{hours} hours'
                AND match_segment IS NOT NULL
            GROUP BY match_segment
            ORDER BY segment_inferences DESC
            """
            segment_df = pd.read_sql_query(segment_query, con=self.engine)
            stats["segment_performance"] = segment_df.to_dict(orient="records") if not segment_df.empty else []

            return {
                "status": "ok",
                "period_hours": hours,
                "statistics": stats,
            }
        except Exception as exc:
            logger.error(f"Failed to retrieve inference statistics: {exc}")
            return {
                "status": "error",
                "message": str(exc),
            }

    def get_recent_inferences(self, limit: int = 50) -> list[dict[str, Any]]:
        """
        Retrieve recent inference logs for debugging/auditing.

        Args:
            limit: Maximum number of recent inferences to return

        Returns:
            List of recent inference records
        """
        query = f"""
        SELECT 
            request_id,
            timestamp_utc,
            requested_match_date,
            home_team,
            away_team,
            neutral,
            tournament,
            predicted_outcome,
            class_probabilities_json,
            feature_source,
            model_version,
            shadow_predicted_outcome,
            shadow_is_override_triggered
        FROM "{INFERENCE_LOG_SCHEMA}"."{INFERENCE_LOG_TABLE}"
        ORDER BY timestamp_utc DESC
        LIMIT {limit}
        """

        try:
            df = pd.read_sql_query(query, con=self.engine)
            records = df.to_dict(orient="records")
            return [cast(dict[str, Any], record) for record in records]
        except Exception as exc:
            logger.error(f"Failed to retrieve recent inferences: {exc}")
            return []


# Module-level singleton for convenience
_inference_logger: InferenceLogger | None = None


def get_inference_logger() -> InferenceLogger:
    """Get or create the module-level inference logger singleton."""
    global _inference_logger
    if _inference_logger is None:
        _inference_logger = InferenceLogger()
    return _inference_logger
