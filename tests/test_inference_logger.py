"""Tests for inference logging and monitoring."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.modeling.inference_logger import InferenceLogger, INFERENCE_LOG_SCHEMA


@pytest.fixture
def inference_logger(engine_fixture):
    """Create an InferenceLogger with test database engine."""
    return InferenceLogger(engine=engine_fixture)


def test_log_prediction(inference_logger):
    """Test que una predicción se loguea correctamente."""
    timestamp = datetime.now(timezone.utc)
    
    inference_logger.log_prediction(
        home_team="Brazil",
        away_team="Argentina",
        predicted_class=0,
        predicted_outcome="win",
        class_probabilities={"win": 0.6, "loss": 0.2, "draw": 0.2},
        neutral=False,
        tournament="2026 FIFA World Cup",
        feature_snapshot_dates={"Brazil": "2026-04-01", "Argentina": "2026-04-01"},
        feature_source="dbt_latest_team_snapshots",
        model_artifact_path="/models/match_predictor.joblib",
        model_version="v1.0",
        requested_match_date=datetime(2026, 6, 12, tzinfo=timezone.utc).date(),
        request_timestamp_utc=timestamp,
    )
    
    # Verify log was persisted
    stats = inference_logger.get_inference_statistics(hours=1)
    assert stats["status"] == "ok"
    assert stats["statistics"]["total_inferences"] >= 1


def test_get_inference_statistics_empty(inference_logger):
    """Test statisticas cuando no hay inferences."""
    stats = inference_logger.get_inference_statistics(hours=1)
    # First time might be empty, that's ok
    assert stats["status"] in ["ok", "no_data"]


def test_get_recent_inferences(inference_logger):
    """Test retrieval of recent inferences."""
    # Log a test prediction
    inference_logger.log_prediction(
        home_team="France",
        away_team="England",
        predicted_class=1,
        predicted_outcome="loss",
        class_probabilities={"win": 0.3, "loss": 0.5, "draw": 0.2},
        neutral=True,
        tournament="2026 FIFA World Cup",
        feature_snapshot_dates={"France": "2026-04-01", "England": "2026-04-01"},
        feature_source="dbt_latest_team_snapshots",
        model_artifact_path="/models/match_predictor.joblib",
        request_timestamp_utc=datetime.now(timezone.utc),
    )
    
    # Retrieve recent inferences
    recent = inference_logger.get_recent_inferences(limit=10)
    assert isinstance(recent, list)


def test_log_prediction_includes_requested_match_date(monkeypatch):
    captured = {}
    inference_logger = InferenceLogger(engine=object())

    monkeypatch.setattr(
        inference_logger,
        "_persist_log",
        lambda df: captured.setdefault("frame", df.copy()),
    )

    inference_logger.log_prediction(
        home_team="Colombia",
        away_team="Argentina",
        predicted_class=1,
        predicted_outcome="home_win",
        class_probabilities={"home_win": 0.55, "draw": 0.25, "away_win": 0.20},
        neutral=False,
        tournament="FIFA World Cup Qualifiers",
        feature_snapshot_dates={"home_team": "2025-11-15", "away_team": "2025-11-16"},
        feature_source="dbt_team_snapshots_as_of_date",
        model_artifact_path="/models/match_predictor.joblib",
        requested_match_date=datetime(2025, 11, 18, tzinfo=timezone.utc).date(),
        request_timestamp_utc=datetime.now(timezone.utc),
    )

    assert captured["frame"].loc[0, "requested_match_date"] == "2025-11-18"


def test_log_prediction_with_none_tournament(inference_logger):
    """Test logging predication with optional tournament."""
    inference_logger.log_prediction(
        home_team="Germany",
        away_team="Italy",
        predicted_class=2,
        predicted_outcome="draw",
        class_probabilities={"win": 0.33, "loss": 0.33, "draw": 0.34},
        neutral=False,
        tournament=None,  # No tournament specified
        feature_snapshot_dates={"Germany": "2026-04-01", "Italy": "2026-04-01"},
        feature_source="postgres_training_runs",
        model_artifact_path="/models/match_predictor.joblib",
        request_timestamp_utc=datetime.now(timezone.utc),
    )
    
    stats = inference_logger.get_inference_statistics(hours=1)
    assert stats["status"] in ["ok", "no_data"]
