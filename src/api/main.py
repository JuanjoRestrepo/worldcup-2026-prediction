from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config.settings import settings
from src.modeling.serving_store import load_latest_training_run_summary_with_source
from src.modeling.predict import predict_match_outcome

app = FastAPI(
    title="World Cup 2026 Prediction API",
    version="0.1.0",
    description="Serving layer scaffold for match prediction workflows.",
)


class PredictionRequest(BaseModel):
    home_team: str = Field(..., min_length=1)
    away_team: str = Field(..., min_length=1)
    tournament: str | None = None
    neutral: bool = False


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
    """Predict the outcome of a fixture from the exported production artifact."""
    try:
        prediction = predict_match_outcome(
            home_team=request.home_team,
            away_team=request.away_team,
            tournament=request.tournament,
            neutral=request.neutral,
        )
        return PredictionResponse(**prediction)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
