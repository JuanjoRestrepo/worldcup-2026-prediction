"""Unit tests for prediction helpers and model bundle compatibility."""

from __future__ import annotations

import numpy as np
import pandas as pd

import src.modeling.predict as predict_module


class _WrappedModel:
    classes_ = np.array([0, 1, 2], dtype=np.int64)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.array([2], dtype=np.int64)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.array([[0.2, 0.25, 0.55]], dtype=np.float64)


def test_predict_match_outcome_supports_models_without_named_steps(monkeypatch):
    monkeypatch.setenv("PREDICTION_FEATURE_SOURCE", "postgres")
    monkeypatch.setattr(
        predict_module,
        "load_model_bundle",
        lambda artifact_path=None: {
            "model": _WrappedModel(),
            "feature_columns": ["elo_home", "elo_away"],
            "target_column": "target_multiclass",
            "outcome_to_encoded": {-1: 0, 0: 1, 1: 2},
            "encoded_to_outcome": {0: -1, 1: 0, 2: 1},
            "outcome_labels": {-1: "away_win", 0: "draw", 1: "home_win"},
            "selected_model_name": "logistic_regression",
            "deployed_model_variant": "sigmoid",
            "calibration_method": "sigmoid",
            "training_summary": {},
        },
    )
    monkeypatch.setattr(
        predict_module,
        "load_latest_team_snapshots_from_dbt",
        lambda: pd.DataFrame(
            {
                "team": ["Brazil", "Brazil", "Argentina", "Argentina"],
                "team_role": ["overall", "home", "overall", "away"],
                "snapshot_date": pd.to_datetime(
                    ["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-02"]
                ),
                "elo": [1800.0, 1800.0, 1825.0, 1825.0],
                "global_avg_goals_last5": [1.8, 1.8, 1.7, 1.7],
                "global_avg_conceded_last5": [0.8, 0.8, 0.7, 0.7],
                "global_win_rate_last5": [0.7, 0.7, 0.72, 0.72],
                "avg_goals_last5": [1.8, 2.0, 1.7, 1.6],
                "avg_goals_conceded_last5": [0.8, 0.7, 0.7, 0.8],
                "win_rate_last5": [0.7, 0.75, 0.72, 0.68],
                "avg_opponent_elo_last5": [1750.0, 1755.0, 1760.0, 1765.0],
                "weighted_win_rate_last5": [0.69, 0.73, 0.7, 0.67],
                "opponent_elo_form": [15.0, 18.0, 12.0, 14.0],
                "elo_form": [20.0, 22.0, 18.0, 19.0],
                "home_advantage_effect": [0.0, 0.05, 0.0, -0.02],
                "persisted_at_utc": pd.to_datetime(
                    [
                        "2026-01-03T00:00:00Z",
                        "2026-01-03T00:00:00Z",
                        "2026-01-03T00:00:00Z",
                        "2026-01-03T00:00:00Z",
                    ],
                    utc=True,
                ),
            }
        ),
    )

    result = predict_module.predict_match_outcome(
        home_team="Brazil",
        away_team="Argentina",
        tournament="FIFA World Cup",
        neutral=False,
    )

    assert result["predicted_class"] == 1
    assert result["predicted_outcome"] == "home_win"
    assert result["class_probabilities"]["home_win"] == 0.55
    assert result["feature_source"] == "dbt_latest_team_snapshots"


class _WrappedShadowModel:
    classes_ = np.array([0, 1, 2], dtype=np.int64)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.array([1], dtype=np.int64)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.array([[0.2, 0.6, 0.2]], dtype=np.float64)

    def _compute_override_mask(
        self, X: pd.DataFrame, gen_probs: np.ndarray, spec_probs: np.ndarray
    ) -> np.ndarray:
        return np.array([True], dtype=bool)

    @property
    def generalist_model_(self):
        return self

    @property
    def specialist_model_(self):
        return self


def test_predict_match_outcome_shadow_deployment(monkeypatch):
    monkeypatch.setenv("PREDICTION_FEATURE_SOURCE", "postgres")
    def mock_load_model_bundle(artifact_path=None):
        path_str = str(artifact_path)
        if "shadow" in path_str:
            return {
                "model": _WrappedShadowModel(),
                "feature_columns": ["elo_home", "elo_away"],
                "target_column": "target_multiclass",
                "outcome_to_encoded": {-1: 0, 0: 1, 1: 2},
                "encoded_to_outcome": {0: -1, 1: 0, 2: 1},
                "outcome_labels": {-1: "away_win", 0: "draw", 1: "home_win"},
                "selected_model_name": "seg_hybrid_balanced",
                "deployed_model_variant": "uncalibrated",
                "calibration_method": "none",
                "training_summary": {},
            }
        return {
            "model": _WrappedModel(),
            "feature_columns": ["elo_home", "elo_away"],
            "target_column": "target_multiclass",
            "outcome_to_encoded": {-1: 0, 0: 1, 1: 2},
            "encoded_to_outcome": {0: -1, 1: 0, 2: 1},
            "outcome_labels": {-1: "away_win", 0: "draw", 1: "home_win"},
            "selected_model_name": "logistic_regression",
            "deployed_model_variant": "sigmoid",
            "calibration_method": "sigmoid",
            "training_summary": {},
        }

    monkeypatch.setattr(predict_module, "load_model_bundle", mock_load_model_bundle)

    import pathlib

    original_exists = pathlib.Path.exists

    def mock_exists(self):
        if "shadow.joblib" in str(self):
            return True
        return original_exists(self)

    monkeypatch.setattr(pathlib.Path, "exists", mock_exists)

    monkeypatch.setattr(
        predict_module,
        "load_latest_team_snapshots_from_dbt",
        lambda: pd.DataFrame(
            {
                "team": ["Brazil", "Brazil", "Argentina", "Argentina"],
                "team_role": ["overall", "home", "overall", "away"],
                "snapshot_date": pd.to_datetime(["2026-01-01"] * 4),
                "elo": [1800.0, 1800.0, 1825.0, 1825.0],
                "global_avg_goals_last5": [1.8] * 4,
                "global_avg_conceded_last5": [0.8] * 4,
                "global_win_rate_last5": [0.7] * 4,
                "avg_goals_last5": [1.8] * 4,
                "avg_goals_conceded_last5": [0.8] * 4,
                "win_rate_last5": [0.7] * 4,
                "avg_opponent_elo_last5": [1750.0] * 4,
                "weighted_win_rate_last5": [0.69] * 4,
                "opponent_elo_form": [15.0] * 4,
                "elo_form": [20.0] * 4,
                "home_advantage_effect": [0.0] * 4,
                "persisted_at_utc": pd.to_datetime(
                    ["2026-01-03T00:00:00Z"] * 4, utc=True
                ),
            }
        ),
    )

    result = predict_module.predict_match_outcome(
        home_team="Brazil",
        away_team="Argentina",
        tournament="FIFA World Cup",
        neutral=False,
    )

    assert result["predicted_class"] == 1
    assert result["shadow_predicted_outcome"] == "draw"
    assert result["shadow_class_probabilities"]["draw"] == 0.6
    assert result["shadow_is_override_triggered"] is True
    assert result["shadow_model_name"] == "seg_hybrid_balanced"
