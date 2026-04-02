"""Unit tests for FastAPI serving helpers."""

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
    monkeypatch.setattr(
        api_main,
        "predict_match_outcome",
        lambda **kwargs: {
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
            "feature_snapshot_dates": {
                "home_team": "2026-03-25",
                "away_team": "2026-03-26",
            },
            "feature_source": "dbt_latest_team_snapshots",
            "model_artifact_path": "models/match_predictor.joblib",
        },
    )

    response = api_main.predict(
        api_main.PredictionRequest(
            home_team="Colombia",
            away_team="Argentina",
            tournament="FIFA World Cup Qualifiers",
            neutral=False,
        )
    )

    assert response.feature_source == "dbt_latest_team_snapshots"


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
