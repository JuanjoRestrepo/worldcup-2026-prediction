"""FastAPI serving layer for World Cup 2026 match prediction."""

from __future__ import annotations

import logging
import os
import traceback
from datetime import date
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from src.config.settings import settings
from src.config.team_aliases import normalize_team_name
from src.modeling.inference_logger import (
    get_inference_logger,
    validate_feature_freshness,
)
from src.modeling.predict import predict_match_outcome, toggle_shadow_mode
from src.modeling.serving_store import load_latest_training_run_summary_with_source
from src.modeling.types import LatestTrainingRunSummary, PredictionResult

logger = logging.getLogger(__name__)

app = FastAPI(
    title="World Cup 2026 Prediction API",
    version="0.1.0",
    description=(
        "Production-grade MLOps serving layer for football match outcome prediction. "
        "Features: segment-aware hybrid ensemble, shadow deployment, inference telemetry, "
        "team name aliases, and feature freshness monitoring."
    ),
)

# ────────────────────────────────────────────────────────────────────────────
# Admin authentication
# ────────────────────────────────────────────────────────────────────────────

API_KEY_NAME = "X-Admin-Key"
_api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


def _get_admin_key(api_key: str = Depends(_api_key_header)) -> str:
    """Validate the admin API key from the request header."""
    expected_key = os.getenv("ADMIN_API_KEY")
    if not expected_key:
        raise HTTPException(
            status_code=503,
            detail="Admin authentication is not configured on this server.",
        )
    if api_key != expected_key:
        raise HTTPException(
            status_code=403, detail="Could not validate admin credentials."
        )
    return api_key


# ────────────────────────────────────────────────────────────────────────────
# Request / Response models
# ────────────────────────────────────────────────────────────────────────────


class PredictionRequest(BaseModel):
    """Request body for match outcome prediction."""

    home_team: str = Field(
        ..., min_length=1, description="Home team name (aliases supported, e.g. 'USA')"
    )
    away_team: str = Field(
        ..., min_length=1, description="Away team name (aliases supported)"
    )
    tournament: str | None = Field(None, description="Tournament name (optional)")
    neutral: bool = Field(False, description="Whether match is on neutral ground")
    match_date: date | None = Field(
        None,
        description=(
            "Date for historical predictions (YYYY-MM-DD). "
            "Defaults to the latest available feature snapshot."
        ),
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    service: str
    shadow_as_primary: bool


class PredictionResponse(BaseModel):
    """Match outcome prediction with ensemble telemetry."""

    home_team: str
    away_team: str
    predicted_class: int
    predicted_outcome: str
    class_probabilities: dict[str, float]
    neutral: bool
    tournament: str | None = None
    match_date: date | None = None
    feature_snapshot_dates: dict[str, str]
    feature_source: str
    model_artifact_path: str
    match_segment: str | None = Field(
        None,
        description="Tournament segment detected by ensemble (worldcup/continental/friendlies/qualifiers)",
    )
    is_override_triggered: bool = Field(
        False,
        description="Whether specialist ensemble override was triggered for this prediction",
    )
    shadow_predicted_outcome: str | None = Field(
        None, description="Shadow model's predicted outcome"
    )
    shadow_class_probabilities: dict[str, float] | None = Field(
        None, description="Shadow model's probability distribution"
    )
    shadow_is_override_triggered: bool | None = Field(
        None, description="Whether shadow model triggered an override"
    )
    shadow_model_name: str | None = Field(
        None, description="Name of the shadow model candidate"
    )
    feature_freshness: dict[str, Any] = Field(
        default_factory=dict, description="Feature age alerts (empty if fresh)"
    )


class LatestTrainingRunResponse(BaseModel):
    """Latest training run metadata for monitoring."""

    pipeline_run_id: str | None = None
    artifact_path: str
    data_path: str
    training_rows: int
    test_rows: int
    feature_count: int
    train_date_start: str
    train_date_end: str
    test_date_start: str
    test_date_end: str
    accuracy: float
    macro_f1: float
    weighted_f1: float
    log_loss: float
    trained_at_utc: str
    persisted_at_utc: str | None = None
    monitoring_source: str


class InferenceStatisticsResponse(BaseModel):
    """Aggregated inference statistics for monitoring dashboard."""

    status: str
    period_hours: int | None = None
    statistics: dict[str, Any] | None = None
    message: str | None = None


class RecentInferenceRecord(BaseModel):
    """Single inference log record for auditing."""

    request_id: str
    timestamp_utc: str
    requested_match_date: str | None = None
    home_team: str
    away_team: str
    neutral: bool
    tournament: str | None
    predicted_outcome: str
    class_probabilities_json: str
    feature_source: str
    model_version: str


class RecentInferencesResponse(BaseModel):
    """List of recent prediction records for debugging."""

    status: str
    count: int
    inferences: list[RecentInferenceRecord]


class ToggleShadowRequest(BaseModel):
    """Request body to hot-swap the shadow model."""

    enable: bool


class ToggleShadowResponse(BaseModel):
    """Response after toggling shadow deployment mode."""

    status: str
    message: str
    shadow_as_primary: bool


# ────────────────────────────────────────────────────────────────────────────
# Endpoints
# ────────────────────────────────────────────────────────────────────────────


@app.get("/")
def root() -> dict[str, str]:
    """Root endpoint with API navigation links."""
    return {
        "message": "World Cup 2026 Prediction API",
        "docs": "/docs",
        "health": "/health",
        "config": "/config",
        "predict": "/predict (POST)",
        "live_api": "https://worldcup-2026-prediction.onrender.com",
    }


@app.get("/health", response_model=HealthResponse)
def healthcheck() -> HealthResponse:
    """Readiness probe for container orchestration and uptime monitoring."""
    import src.modeling.predict as predict_module  # noqa: PLC0415 (local import for runtime state)

    return HealthResponse(
        status="ok",
        service="worldcup-api",
        shadow_as_primary=bool(
            getattr(predict_module, "_USE_SHADOW_AS_PRIMARY", False)
        ),
    )


@app.get("/config")
def runtime_config() -> dict[str, str]:
    """Expose non-sensitive runtime configuration for debugging."""
    return {
        "data_dir": str(settings.DATA_DIR),
        "gold_dir": str(settings.GOLD_DIR),
        "dbt_base_schema": settings.DBT_BASE_SCHEMA,
        "prediction_feature_source": settings.PREDICTION_FEATURE_SOURCE,
        "monitoring_source": settings.MONITORING_SOURCE,
        "model_artifact_path": str(settings.MODEL_ARTIFACT_PATH),
        "model_artifact_exists": str(settings.MODEL_ARTIFACT_PATH.exists()).lower(),
    }


@app.post("/admin/toggle-shadow", response_model=ToggleShadowResponse)
def toggle_shadow(
    request: ToggleShadowRequest,
    api_key: str = Depends(_get_admin_key),
) -> ToggleShadowResponse:
    """Admin endpoint to hot-swap the primary model with the shadow model."""
    toggle_shadow_mode(request.enable)
    state_str = "ENABLED" if request.enable else "DISABLED"
    logger.info("Admin action: Shadow model as primary has been %s", state_str)
    return ToggleShadowResponse(
        status="success",
        message=f"Shadow model as primary is now {state_str}",
        shadow_as_primary=request.enable,
    )


@app.get("/monitoring/latest-training-run", response_model=LatestTrainingRunResponse)
def latest_training_run() -> LatestTrainingRunResponse:
    """Expose the latest curated training summary for lightweight monitoring."""
    try:
        training_run, monitoring_source = load_latest_training_run_summary_with_source(
            source=settings.MONITORING_SOURCE,
        )
        typed_training_run: LatestTrainingRunSummary = training_run
        return LatestTrainingRunResponse(
            monitoring_source=monitoring_source,
            **typed_training_run,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Predict the outcome of a fixture from the exported production artifact.

    Supports:
    - Team name aliases (USA → United States)
    - Historical predictions via match_date
    - Stale feature warnings
    - Segment-aware ensemble telemetry
    - Shadow deployment comparison

    Args:
        request: PredictionRequest with home_team, away_team, and optional fields.

    Returns:
        PredictionResponse with prediction, probabilities, and feature freshness info.

    Raises:
        HTTPException 422: Invalid request (team not found, bad date).
        HTTPException 503: Service unavailable (model/features unavailable).
        HTTPException 500: Unexpected internal error.
    """
    normalized_home = normalize_team_name(request.home_team)
    normalized_away = normalize_team_name(request.away_team)

    try:
        logger.info(
            "Prediction request | home=%s away=%s tournament=%s match_date=%s",
            normalized_home,
            normalized_away,
            request.tournament,
            request.match_date,
        )

        prediction: PredictionResult = predict_match_outcome(
            home_team=normalized_home,
            away_team=normalized_away,
            tournament=request.tournament,
            neutral=request.neutral,
            match_date=request.match_date,
        )
        logger.info(
            "Prediction successful | outcome=%s segment=%s override=%s",
            prediction["predicted_outcome"],
            prediction.get("match_segment"),
            prediction.get("is_override_triggered", False),
        )

        freshness = validate_feature_freshness(
            prediction["feature_snapshot_dates"], max_age_days=30
        )
        return PredictionResponse(
            home_team=prediction["home_team"],
            away_team=prediction["away_team"],
            predicted_class=prediction["predicted_class"],
            predicted_outcome=prediction["predicted_outcome"],
            class_probabilities=prediction["class_probabilities"],
            neutral=prediction["neutral"],
            tournament=prediction["tournament"],
            match_date=request.match_date,
            feature_snapshot_dates={
                "home_team": prediction["feature_snapshot_dates"]["home_team"],
                "away_team": prediction["feature_snapshot_dates"]["away_team"],
            },
            feature_source=prediction["feature_source"],
            model_artifact_path=prediction["model_artifact_path"],
            match_segment=prediction.get("match_segment"),
            is_override_triggered=prediction.get("is_override_triggered", False),
            shadow_predicted_outcome=prediction.get("shadow_predicted_outcome"),
            shadow_class_probabilities=prediction.get("shadow_class_probabilities"),
            shadow_is_override_triggered=prediction.get("shadow_is_override_triggered"),
            shadow_model_name=prediction.get("shadow_model_name"),
            feature_freshness=freshness,
        )

    except ValueError as exc:
        error_msg = str(exc).lower()
        if "not found" in error_msg or "no data" in error_msg:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Team not found in feature snapshots: '{request.home_team}' or '{request.away_team}'. "
                    "Please check team names (aliases like USA → United States are supported). "
                    "Available teams must have recent ELO and form data."
                ),
            ) from exc
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Model artifact not found. Train the model first: {exc}",
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503, detail=f"Service unavailable: {exc}"
        ) from exc
    except Exception as exc:
        logger.error(
            "Unexpected error in /predict for %s vs %s: %s\n%s",
            normalized_home,
            normalized_away,
            exc,
            traceback.format_exc(),
        )
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {type(exc).__name__}: {str(exc)[:200]}",
        ) from exc


@app.get("/monitoring/inference-stats", response_model=InferenceStatisticsResponse)
def inference_statistics(hours: int = 24) -> InferenceStatisticsResponse:
    """
    Get inference statistics for the last N hours.

    Provides aggregated metrics: total inferences, prediction distribution,
    feature sources used, probability statistics, and per-segment shadow performance.
    """
    if hours < 1 or hours > 720:
        raise HTTPException(status_code=422, detail="hours must be between 1 and 720")

    inference_logger = get_inference_logger()
    stats_dict = inference_logger.get_inference_statistics(hours=hours)
    return InferenceStatisticsResponse(**stats_dict)


@app.get("/monitoring/recent-inferences", response_model=RecentInferencesResponse)
def recent_inferences(limit: int = 50) -> RecentInferencesResponse:
    """
    Retrieve recent inference logs for auditing and debugging.

    Returns the latest N predictions with full request and prediction details.
    Useful for troubleshooting, feature source tracking, and model behavior inspection.
    """
    if limit < 1 or limit > 500:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 500")

    inference_logger = get_inference_logger()
    records = inference_logger.get_recent_inferences(limit=limit)
    return RecentInferencesResponse(
        status="ok" if records else "no_data",
        count=len(records),
        inferences=[RecentInferenceRecord(**r) for r in records],
    )
