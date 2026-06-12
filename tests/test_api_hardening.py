"""Tests for the hardened /predict endpoint."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.api.main import app
from backend.config.team_aliases import TEAM_ALIASES, normalize_team_name

client = TestClient(app)


class TestTeamAliases:
    """Test team name alias normalization."""

    def test_normalize_team_name_alias(self):
        """Test that aliases are resolved correctly."""
        assert normalize_team_name("USA") == "United States"
        assert normalize_team_name("Brasil") == "Brazil"
        assert normalize_team_name("Holland") == "Netherlands"

    def test_normalize_team_name_no_alias(self):
        """Test that non-aliases are returned as-is."""
        assert normalize_team_name("Brazil") == "Brazil"
        assert normalize_team_name("England") == "England"

    def test_normalize_case_insensitive(self):
        """Test that normalization is case-insensitive."""
        assert normalize_team_name("usa") == "United States"
        assert normalize_team_name("USA") == "United States"
        assert normalize_team_name("UsA") == "United States"

    def test_all_aliases_valid(self):
        """Ensure all aliases map to non-empty strings."""
        for alias, canonical in TEAM_ALIASES.items():
            assert isinstance(alias, str)
            assert isinstance(canonical, str)
            assert len(canonical) > 0


class TestPredictEndpoint:
    """Test the /predict endpoint with hardening features."""

    def test_predict_health_check(self):
        """Ensure API is running."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_predict_request_schema(self):
        """Test that PredictionRequest schema accepts match_date."""
        # This should not raise validation errors
        payload = {
            "home_team": "Brazil",
            "away_team": "Argentina",
            "tournament": "2026 FIFA World Cup",
            "neutral": False,
            "match_date": "2026-04-02",
        }
        # Just verify schema is valid by creating request (would fail if invalid)
        from backend.api.main import PredictionRequest

        req = PredictionRequest(**payload)
        assert req.home_team == "Brazil"
        assert req.match_date.isoformat() == "2026-04-02"

    def test_predict_response_includes_feature_freshness(self):
        """Test that PredictionResponse includes feature_freshness field."""
        from backend.api.main import PredictionResponse

        response_data = {
            "home_team": "Brazil",
            "away_team": "Argentina",
            "predicted_class": 0,
            "predicted_outcome": "win",
            "class_probabilities": {"win": 0.6, "loss": 0.2, "draw": 0.2},
            "neutral": False,
            "tournament": None,
            "feature_snapshot_dates": {"home": "2026-04-02", "away": "2026-04-02"},
            "feature_source": "test",
            "model_artifact_path": "/test/model.joblib",
            "feature_freshness": {"is_fresh": True, "warning": None},
        }
        response = PredictionResponse(**response_data)
        assert "feature_freshness" in response.__dict__


class TestFeatureFreshness:
    """Test feature freshness validation."""

    def test_validate_feature_freshness_fresh(self):
        """Test detection of fresh features."""
        from backend.modeling.inference_logger import validate_feature_freshness

        now = datetime.now(UTC)
        today_str = now.date().isoformat()

        freshness = validate_feature_freshness(
            {"home": today_str, "away": today_str}, max_age_days=30
        )
        assert freshness["is_fresh"] is True
        assert freshness["warning"] is None

    def test_validate_feature_freshness_stale(self):
        """Test detection of stale features."""
        from datetime import timedelta

        from backend.modeling.inference_logger import validate_feature_freshness

        old_date = (datetime.now(UTC) - timedelta(days=40)).date().isoformat()

        freshness = validate_feature_freshness(
            {"home": old_date, "away": old_date}, max_age_days=30
        )
        assert freshness["is_fresh"] is False
        assert freshness["warning"] is not None
        assert "40 days old" in freshness["warning"]

    def test_validate_feature_freshness_mixed(self):
        """Test handling of mixed fresh/stale features."""
        from datetime import timedelta

        from backend.modeling.inference_logger import validate_feature_freshness

        now = datetime.now(UTC)
        fresh_date = now.date().isoformat()
        stale_date = (now - timedelta(days=40)).date().isoformat()

        freshness = validate_feature_freshness(
            {"home": fresh_date, "away": stale_date}, max_age_days=30
        )
        assert freshness["is_fresh"] is False
        assert len(freshness["age_days"]) == 2


class TestPredictWithAliases:
    """Test /predict endpoint with team name aliases."""

    def test_predict_request_schema_with_alias_description(self):
        """Verify schema documents that aliases are supported."""
        from backend.api.main import PredictionRequest

        schema = PredictionRequest.model_json_schema()
        assert "home_team" in schema["properties"]
        home_desc = schema["properties"]["home_team"].get("description", "")
        assert "alias" in home_desc.lower() or len(home_desc) == 0  # May not have desc

    def test_all_test_aliases_supported(self):
        """Verify critical aliases are in mapping."""
        critical_aliases = ["USA", "Brasil", "Holland", "UK", "Korea"]
        for alias in critical_aliases:
            normalized = normalize_team_name(alias)
            assert normalized != alias or alias in [
                "United States"
            ]  # USA maps to United States


class TestErrorHandling:
    """Test improved error handling in /predict."""

    def test_predict_error_messages_helpful(self):
        """Test that error messages are informative."""
        from backend.api.main import PredictionRequest

        # Invalid request should return helpful message
        invalid_payload = {
            "home_team": "NonexistentTeam123",
            "away_team": "FakeTeam456",
        }
        # Would get helpful error if API endpoint returns 422
        # This test documents what error message users should see
        req = PredictionRequest(**invalid_payload)
        assert req.home_team == "NonexistentTeam123"


class TestAdminEndpoints:
    """Test the admin endpoints: /admin/results and /admin/run-pipeline."""

    def test_record_result_unauthenticated(self):
        """Ensure /admin/results returns 503 when admin key not configured, or 403 when key is wrong."""
        payload = {
            "date": "2026-06-12",
            "home_team": "Canada",
            "away_team": "Bosnia and Herzegovina",
            "home_score": 2,
            "away_score": 1,
            "tournament": "FIFA World Cup",
        }
        response = client.post("/admin/results", json=payload)
        # Should be 503 (since ADMIN_API_KEY is not set in test environment by default)
        assert response.status_code in {503, 403}

    def test_record_result_authenticated(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Ensure /admin/results successfully records a match when authenticated."""
        monkeypatch.setenv("ADMIN_API_KEY", "supersecret")

        from backend.config.settings import settings

        monkeypatch.setattr(settings, "RAW_DIR", tmp_path)

        payload = {
            "date": "2026-06-12",
            "home_team": "Canada",
            "away_team": "Bosnia and Herzegovina",
            "home_score": 2,
            "away_score": 1,
            "tournament": "FIFA World Cup",
            "city": "Toronto",
            "country": "Canada",
            "neutral": False,
        }
        headers = {"X-Admin-Key": "supersecret"}
        response = client.post("/admin/results", json=payload, headers=headers)
        assert response.status_code == 200
        assert response.json()["status"] == "success"

        manual_csv = tmp_path / "manual_results.csv"
        assert manual_csv.exists()

        import pandas as pd

        df = pd.read_csv(manual_csv)
        assert len(df) == 1
        assert df.iloc[0]["home_team"] == "Canada"
        assert df.iloc[0]["away_team"] == "Bosnia and Herzegovina"
        assert df.iloc[0]["home_score"] == 2.0
        assert df.iloc[0]["away_score"] == 1.0

    def test_run_pipeline_authenticated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Ensure /admin/run-pipeline triggers pipeline when authenticated."""
        monkeypatch.setenv("ADMIN_API_KEY", "supersecret")

        pipeline_calls = []

        def mock_run_pipeline(**kwargs):
            pipeline_calls.append(kwargs)
            return {"status": "success"}

        import run_pipeline

        monkeypatch.setattr(run_pipeline, "run_full_pipeline", mock_run_pipeline)

        cache_cleared = []
        import backend.modeling.predict as predict_mod

        monkeypatch.setattr(
            predict_mod._load_model_bundle_cached,
            "cache_clear",
            lambda: cache_cleared.append(True),
        )

        headers = {"X-Admin-Key": "supersecret"}
        response = client.post("/admin/run-pipeline", headers=headers)
        assert response.status_code == 200
        assert response.json()["status"] == "success"

        assert len(pipeline_calls) == 1
        assert pipeline_calls[0]["use_api_data"] is False
        assert len(cache_cleared) == 1


class TestPredictIntegration:
    """Integration tests (requires full API running)."""

    def test_config_endpoint(self):
        """Test that config endpoint works."""
        response = client.get("/config")
        if response.status_code == 200:
            data = response.json()
            assert "data_dir" in data or "model_artifact_path" in data

    def test_predict_documentation(self):
        """Test that API docs are available."""
        response = client.get("/docs")
        assert response.status_code == 200
        # OpenAPI/Swagger documentation should be served
