from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config.settings import settings

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
    status: str
    detail: str
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
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Placeholder serving endpoint.

    The API contract is ready, but prediction execution stays disabled until
    the training/export flow persists a production model artifact.
    """
    model_artifact = Path(settings.MODEL_ARTIFACT_PATH)
    if not model_artifact.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                "Model artifact not found. Train/export a model to "
                f"{model_artifact} before enabling predictions."
            ),
        )

    raise HTTPException(
        status_code=501,
        detail=(
            "Prediction serving scaffold is ready, but request-to-feature "
            "assembly is not implemented yet."
        ),
    )
