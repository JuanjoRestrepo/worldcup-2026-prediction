"""Integration tests for the FastAPI prediction engine using TestClient.

These tests run without requiring a separate running server process, 
using FastAPI's TestClient to simulate HTTP requests.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


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
