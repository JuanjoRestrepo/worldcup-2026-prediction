"""Unit tests for FastAPI serving helpers."""

from datetime import date

import src.api.main as api_main


def test_runtime_config_exposes_prediction_feature_source(monkeypatch):
    monkeypatch.setattr(api_main.settings, "PREDICTION_FEATURE_SOURCE", "auto")
    monkeypatch.setattr(api_main.settings, "MONITORING_SOURCE", "auto")
    monkeypatch.setattr(api_main.settings, "DBT_BASE_SCHEMA", "analytics")

    config = api_main.runtime_config()

    assert config["prediction_feature_source"] == "auto"
    assert config["monitoring_source"] == "auto"
    assert config["dbt_base_schema"] == "analytics"


def test_predict_endpoint_returns_active_feature_source(monkeypatch):
    captured: dict[str, object] = {}

    class _LoggerStub:
        def log_prediction(self, **kwargs):
            captured["logged_request"] = kwargs

    def _predict_stub(**kwargs):
        captured["predict_kwargs"] = kwargs
        return {
            "home_team": kwargs["home_team"],
            "away_team": kwargs["away_team"],
            "predicted_class": 1,
            "predicted_outcome": "home_win",
            "class_probabilities": {
                "home_win": 0.62,
                "draw": 0.21,
                "away_win": 0.17,
            },
            "neutral": kwargs["neutral"],
            "tournament": kwargs["tournament"],
            "match_date": kwargs["match_date"].isoformat() if kwargs["match_date"] else None,
            "feature_snapshot_dates": {
                "home_team": "2026-03-25",
                "away_team": "2026-03-26",
            },
            "feature_source": "dbt_team_snapshots_as_of_date",
            "model_artifact_path": "models/match_predictor.joblib",
        }

    monkeypatch.setattr(
        api_main,
        "predict_match_outcome",
        _predict_stub,
    )
    monkeypatch.setattr(api_main, "get_inference_logger", lambda: _LoggerStub())
    monkeypatch.setattr(
        api_main,
        "validate_feature_freshness",
        lambda feature_dates, max_age_days: {"is_fresh": True, "warning": None, "age_days": {}},
    )

    response = api_main.predict(
        api_main.PredictionRequest(
            home_team="USA",
            away_team="Argentina",
            tournament="FIFA World Cup Qualifiers",
            neutral=False,
            match_date=date(2025, 11, 18),
        )
    )

    assert response.feature_source == "dbt_team_snapshots_as_of_date"
    assert response.home_team == "United States"
    assert response.match_date == date(2025, 11, 18)
    assert captured["predict_kwargs"]["match_date"] == date(2025, 11, 18)
    assert captured["logged_request"]["requested_match_date"] == date(2025, 11, 18)


def test_latest_training_run_endpoint_returns_monitoring_source(monkeypatch):
    monkeypatch.setattr(
        api_main,
        "load_latest_training_run_summary_with_source",
        lambda source: (
            {
                "pipeline_run_id": "run-123",
                "artifact_path": "models/match_predictor.joblib",
                "data_path": "data/gold/features_dataset.csv",
                "training_rows": 100,
                "test_rows": 20,
                "feature_count": 16,
                "train_date_start": "2020-01-01",
                "train_date_end": "2024-12-31",
                "test_date_start": "2025-01-01",
                "test_date_end": "2025-12-31",
                "accuracy": 0.61,
                "macro_f1": 0.55,
                "weighted_f1": 0.6,
                "log_loss": 0.92,
                "trained_at_utc": "2026-04-02T05:00:00+00:00",
                "persisted_at_utc": "2026-04-02T05:01:00+00:00",
            },
            "dbt_latest_training_run",
        ),
    )

    response = api_main.latest_training_run()

    assert response.monitoring_source == "dbt_latest_training_run"
