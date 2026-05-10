"""Integration tests for the FastAPI prediction engine using TestClient.

These tests run without requiring a separate running server process,
using FastAPI's TestClient to simulate HTTP requests.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from backend.api.main import app


@pytest.fixture(autouse=True)
def mock_db_dependencies():
    """Mock database dependencies to prevent CI failures on empty DB."""
    with (
        patch("backend.api.main.predict_match_outcome") as mock_predict,
        patch(
            "backend.api.main.load_latest_training_run_summary_with_source"
        ) as mock_latest_run,
    ):
        # Mock prediction response using TypedDict structure
        mock_predict.return_value = {
            "home_team": "Argentina",
            "away_team": "Brazil",
            "predicted_class": 1,
            "predicted_outcome": "home_win",
            "class_probabilities": {"home_win": 0.6, "draw": 0.3, "away_win": 0.1},
            "neutral": True,
            "tournament": "World Cup Qualifiers",
            "match_date": "2026-05-10",
            "feature_snapshot_dates": {
                "home_team": "Argentina",
                "away_team": "Brazil",
                "home_snapshot_date": "2026-05-01",
                "away_snapshot_date": "2026-05-01",
            },
            "feature_source": "mock_postgres",
            "model_artifact_path": "models/match_predictor.joblib",
            "match_segment": "qualifiers",
            "is_override_triggered": False,
        }

        # Mock invalid team failure gracefully if needed
        def side_effect(**kwargs):
            if kwargs.get("home_team") == "Atlantis FC":
                raise ValueError(
                    "Team 'Atlantis FC' was not found in the gold feature dataset."
                )
            return mock_predict.return_value

        mock_predict.side_effect = side_effect

        # Mock monitoring response
        mock_latest_run.return_value = (
            {
                "pipeline_run_id": "mock_run_id",
                "artifact_path": "models/mock.joblib",
                "data_path": "data/mock.csv",
                "training_rows": 1000,
                "test_rows": 200,
                "feature_count": 50,
                "train_date_start": "2010-01-01",
                "train_date_end": "2026-01-01",
                "test_date_start": "2026-01-02",
                "test_date_end": "2026-05-01",
                "accuracy": 0.8,
                "macro_f1": 0.8,
                "weighted_f1": 0.8,
                "log_loss": 0.5,
                "trained_at_utc": "2026-05-10T00:00:00Z",
                "persisted_at_utc": "2026-05-10T00:00:00Z",
            },
            "mock_source",
        )
        yield


@pytest.fixture
def client() -> TestClient:
    """Create a TestClient for the FastAPI app."""
    return TestClient(app)


def test_api_root_endpoint(client: TestClient) -> None:
    """Test the root endpoint (discovery)."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "predict" in data


def test_api_health_check(client: TestClient) -> None:
    """Test health check / config endpoint."""
    response = client.get("/config")
    assert response.status_code == 200
    data = response.json()
    assert "prediction_feature_source" in data
    assert "monitoring_source" in data


def test_api_prediction_success(client: TestClient) -> None:
    """Test prediction endpoint with valid data."""
    fixture = {
        "home_team": "Argentina",
        "away_team": "Brazil",
        "tournament": "World Cup Qualifiers",
        "neutral": True,
    }
    response = client.post(
        "/predict",
        json=fixture,
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 200
    data = response.json()

    # Verify core prediction fields
    assert "home_team" in data
    assert "away_team" in data
    assert "predicted_outcome" in data
    assert "class_probabilities" in data

    # Verify segment-aware fields (added in Phase 6 hardening)
    assert "match_segment" in data
    assert "is_override_triggered" in data
    assert data["home_team"] == "Argentina"
    assert data["away_team"] == "Brazil"


def test_api_prediction_invalid_team(client: TestClient) -> None:
    """Test prediction endpoint with non-existent teams (should still normalize or fail gracefully)."""
    fixture = {
        "home_team": "Atlantis FC",
        "away_team": "Mars United",
        "tournament": "Galactic Cup",
    }
    # The API might normalize these or use defaults if not found in ELO/Features.
    # We just want to ensure it doesn't crash (500).
    response = client.post("/predict", json=fixture)
    assert response.status_code in (200, 422)


def test_api_latest_training_run(client: TestClient) -> None:
    """Test the monitoring endpoint for training metadata."""
    response = client.get("/monitoring/latest-training-run")
    assert response.status_code == 200
    data = response.json()
    assert "pipeline_run_id" in data
    assert "accuracy" in data
    assert "macro_f1" in data
