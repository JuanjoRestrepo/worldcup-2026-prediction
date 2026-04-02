"""Unit tests for FastAPI serving helpers."""

import src.api.main as api_main


def test_runtime_config_exposes_prediction_feature_source(monkeypatch):
    monkeypatch.setattr(api_main.settings, "PREDICTION_FEATURE_SOURCE", "auto")

    config = api_main.runtime_config()

    assert config["prediction_feature_source"] == "auto"


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
            "feature_source": "postgres",
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

    assert response.feature_source == "postgres"
