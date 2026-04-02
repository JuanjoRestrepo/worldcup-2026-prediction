from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config.settings import settings
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
    model_artifact_path: str


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
        "model_artifact_path": str(settings.MODEL_ARTIFACT_PATH),
        "model_artifact_exists": str(settings.MODEL_ARTIFACT_PATH.exists()).lower(),
    }


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
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
