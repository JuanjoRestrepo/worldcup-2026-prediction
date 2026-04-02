"""Inference logging and monitoring for prediction serving."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from sqlalchemy.engine import Engine

from src.config.settings import settings
from src.database.connection import get_sqlalchemy_engine
from src.database.persistence import ensure_schema

logger = logging.getLogger(__name__)

INFERENCE_LOG_SCHEMA = "monitoring"
INFERENCE_LOG_TABLE = "inference_logs"


class InferenceLogger:
    """Centralized logging for model inference requests and predictions."""

    def __init__(self, engine: Engine | None = None) -> None:
        """Initialize with optional SQLAlchemy engine."""
        self.engine = engine or get_sqlalchemy_engine()

    def log_prediction(
        self,
        home_team: str,
        away_team: str,
        predicted_class: int,
        predicted_outcome: str,
        class_probabilities: dict[str, float],
        neutral: bool,
        tournament: str | None,
        feature_snapshot_dates: dict[str, str],
        feature_source: str,
        model_artifact_path: str,
        model_version: str | None = None,
        request_timestamp_utc: datetime | None = None,
    ) -> None:
        """
        Log a single prediction to PostgreSQL.

        Args:
            home_team: Home team name
            away_team: Away team name
            predicted_class: Encoded prediction (0, 1, 2)
            predicted_outcome: Human-readable outcome (win, loss, draw)
            class_probabilities: Dict mapping outcomes to probabilities
            neutral: Whether match is on neutral ground
            tournament: Optional tournament name
            feature_snapshot_dates: Dict of snapshot dates for features
            feature_source: Source of features (dbt, postgres, csv)
            model_artifact_path: Path to model artifact used
            model_version: Optional model version identifier
            request_timestamp_utc: Optional request timestamp (defaults to now)
        """
        timestamp = request_timestamp_utc or datetime.now(timezone.utc)

        log_row = {
            "request_id": f"{home_team}_{away_team}_{timestamp.timestamp()}",
            "timestamp_utc": timestamp.isoformat(),
            "home_team": home_team,
            "away_team": away_team,
            "neutral": neutral,
            "tournament": tournament,
            "predicted_class": predicted_class,
            "predicted_outcome": predicted_outcome,
            "class_probabilities_json": json.dumps(class_probabilities),
            "feature_snapshot_dates_json": json.dumps(feature_snapshot_dates),
            "feature_source": feature_source,
            "model_artifact_path": str(model_artifact_path),
            "model_version": model_version or "unknown",
            "persisted_at_utc": datetime.now(timezone.utc).isoformat(),
        }

        df = pd.DataFrame([log_row])
        self._persist_log(df)

    def _persist_log(self, df: pd.DataFrame) -> None:
        """Persist a log DataFrame to PostgreSQL."""
        try:
            ensure_schema(self.engine, INFERENCE_LOG_SCHEMA)
            df.to_sql(
                name=INFERENCE_LOG_TABLE,
                con=self.engine,
                schema=INFERENCE_LOG_SCHEMA,
                if_exists="append",
                index=False,
                method="multi",
                chunksize=1000,
            )
            logger.debug(f"Logged {len(df)} inference(s) to {INFERENCE_LOG_SCHEMA}.{INFERENCE_LOG_TABLE}")
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
            SUM(CASE WHEN predicted_outcome = 'win' THEN 1 ELSE 0 END) as home_wins_predicted,
            SUM(CASE WHEN predicted_outcome = 'loss' THEN 1 ELSE 0 END) as home_losses_predicted,
            SUM(CASE WHEN predicted_outcome = 'draw' THEN 1 ELSE 0 END) as draws_predicted,
            COUNT(DISTINCT tournament) as tournaments_predicted,
            MIN(timestamp_utc) as earliest_request,
            MAX(timestamp_utc) as latest_request
        FROM "{INFERENCE_LOG_SCHEMA}"."{INFERENCE_LOG_TABLE}"
        WHERE timestamp_utc > NOW() - INTERVAL '{hours} hours'
        """

        try:
            df = pd.read_sql_query(query, con=self.engine)
            if df.empty:
                return {"status": "no_data", "message": f"No inferences in last {hours} hours"}

            stats = df.iloc[0].to_dict()
            # Convert numeric columns properly
            for key in stats:
                if pd.isna(stats[key]):
                    stats[key] = None
                elif key.startswith("avg_"):
                    stats[key] = round(float(stats[key]), 4) if stats[key] is not None else None
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
            home_team,
            away_team,
            neutral,
            tournament,
            predicted_outcome,
            class_probabilities_json,
            feature_source,
            model_version
        FROM "{INFERENCE_LOG_SCHEMA}"."{INFERENCE_LOG_TABLE}"
        ORDER BY timestamp_utc DESC
        LIMIT {limit}
        """

        try:
            df = pd.read_sql_query(query, con=self.engine)
            return df.to_dict(orient="records")
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
