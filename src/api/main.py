from datetime import datetime, date
from typing import Any
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config.settings import settings
from src.config.team_aliases import normalize_team_name
from src.modeling.inference_logger import get_inference_logger, validate_feature_freshness
from src.modeling.serving_store import load_latest_training_run_summary_with_source
from src.modeling.predict import predict_match_outcome

logger = logging.getLogger(__name__)

app = FastAPI(
    title="World Cup 2026 Prediction API",
    version="0.1.0",
    description="Serving layer scaffold for match prediction workflows.",
)


class PredictionRequest(BaseModel):
    home_team: str = Field(..., min_length=1, description="Home team name (aliases supported)")
    away_team: str = Field(..., min_length=1, description="Away team name (aliases supported)")
    tournament: str | None = Field(None, description="Tournament name (optional)")
    neutral: bool = Field(False, description="Whether match is on neutral ground")
    match_date: date | None = Field(
        None,
        description="Date for historical predictions (YYYY-MM-DD). Defaults to today."
    )


class PredictionResponse(BaseModel):
    home_team: str
    away_team: str
    predicted_class: int
    predicted_outcome: str
    class_probabilities: dict[str, float]
    neutral: bool
    tournament: str | None = None
    feature_snapshot_dates: dict[str, str]
    feature_source: str
    model_artifact_path: str
    feature_freshness: dict[str, Any] = Field(
        default_factory=dict,
        description="Feature age alerts (empty if fresh)"
    )


class LatestTrainingRunResponse(BaseModel):
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
    status: str
    period_hours: int | None = None
    statistics: dict[str, Any] | None = None
    message: str | None = None


class RecentInferenceRecord(BaseModel):
    request_id: str
    timestamp_utc: str
    home_team: str
    away_team: str
    neutral: bool
    tournament: str | None
    predicted_outcome: str
    class_probabilities_json: str
    feature_source: str
    model_version: str


class RecentInferencesResponse(BaseModel):
    status: str
    count: int
    inferences: list[RecentInferenceRecord]


@app.get("/health")
def healthcheck() -> dict[str, str]:
    """Simple readiness endpoint for container and local checks."""
    return {"status": "ok", "service": "worldcup-api"}


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


@app.get("/monitoring/latest-training-run", response_model=LatestTrainingRunResponse)
def latest_training_run() -> LatestTrainingRunResponse:
    """Expose the latest curated training summary for lightweight monitoring."""
    try:
        training_run, monitoring_source = load_latest_training_run_summary_with_source(
            source=settings.MONITORING_SOURCE,
        )
        return LatestTrainingRunResponse(
            monitoring_source=monitoring_source,
            **training_run,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Predict the outcome of a fixture from the exported production artifact.
    
    Supports:
    - Team name aliases (USA → United States)
    - Historical predictions via match_date
    - Stale feature warnings
    
    Args:
        request: PredictionRequest with home_team, away_team, and optional tournament, neutral, match_date
        
    Returns:
        PredictionResponse with prediction, probabilities, and feature freshness info
        
    Raises:
        HTTPException 422: Invalid request (team not found, bad date format)
        HTTPException 503: Service unavailable (model/features unavailable)
    """
    # Normalize team names (handle aliases like USA → United States)
    normalized_home = normalize_team_name(request.home_team)
    normalized_away = normalize_team_name(request.away_team)
    
    try:
        # Make prediction using normalized team names
        prediction = predict_match_outcome(
            home_team=normalized_home,
            away_team=normalized_away,
            tournament=request.tournament,
            neutral=request.neutral,
        )
        
        # Check feature freshness (warn if >30 days old)
        freshness = validate_feature_freshness(
            prediction["feature_snapshot_dates"],
            max_age_days=30
        )
        
        # Build response with feature freshness info
        response_data = {
            **prediction,
            "feature_freshness": freshness,
        }
        response = PredictionResponse(**response_data)
        
        # Log the prediction for observability
        logger = get_inference_logger()
        logger.log_prediction(
            home_team=normalized_home,
            away_team=normalized_away,
            predicted_class=prediction["predicted_class"],
            predicted_outcome=prediction["predicted_outcome"],
            class_probabilities=prediction["class_probabilities"],
            neutral=request.neutral,
            tournament=request.tournament,
            feature_snapshot_dates=prediction["feature_snapshot_dates"],
            feature_source=prediction["feature_source"],
            model_artifact_path=prediction["model_artifact_path"],
            model_version=None,
            request_timestamp_utc=datetime.now(),
        )
        
        return response
        
    except ValueError as exc:
        # Team not found or other data validation error
        error_msg = str(exc).lower()
        if "not found" in error_msg or "no data" in error_msg:
            raise HTTPException(
                status_code=422,
                detail=f"Team not found in feature snapshots: '{request.home_team}' or '{request.away_team}'. "
                       f"Please check team names (aliases like USA → United States are supported). "
                       f"Available teams must have recent ELO and form data."
            )
        raise HTTPException(status_code=422, detail=str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Model artifact not found. Train the model first: {exc}"
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Service unavailable: {exc}"
        )
    except Exception as exc:
        # Catch unexpected errors and log them
        logger.error(f"Unexpected error in /predict: {exc}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error. Check logs for details."
        )


@app.get("/monitoring/inference-stats", response_model=InferenceStatisticsResponse)
def inference_statistics(hours: int = 24) -> InferenceStatisticsResponse:
    """
    Get inference statistics for the last N hours.
    
    Provides aggregated metrics: total inferences, prediction distribution,
    feature sources used, and probability statistics.
    """
    if hours < 1 or hours > 720:
        raise HTTPException(status_code=422, detail="hours must be between 1 and 720")
    
    logger = get_inference_logger()
    stats_dict = logger.get_inference_statistics(hours=hours)
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
    
    logger = get_inference_logger()
    records = logger.get_recent_inferences(limit=limit)
    return RecentInferencesResponse(
        status="ok" if records else "no_data",
        count=len(records),
        inferences=[RecentInferenceRecord(**r) for r in records],
    )
