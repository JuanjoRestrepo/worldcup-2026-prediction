"""Tests for train.py."""

from unittest.mock import patch

import pandas as pd

from backend.modeling.features import TARGET_COLUMN
from backend.modeling.train import train_and_export_model


@patch("backend.modeling.train.persist_training_run")
def test_train_and_export_model(mock_persist, tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Test the end-to-end model training, evaluation, and exporting logic."""
    artifact_path = tmp_path / "model.joblib"

    # Mock dataset with 1600 rows. Must contain all 3 classes to avoid XGBoost errors.
    dates = pd.date_range("2010-01-01", periods=1600, freq="D")
    df = pd.DataFrame(
        {
            "date": dates,
            "homeTeam": ["A", "B", "C", "D"] * 400,
            "awayTeam": ["B", "A", "D", "C"] * 400,
            "homeGoals": [2, 1, 0, 1] * 400,
            "awayGoals": [1, 1, 2, 0] * 400,
            TARGET_COLUMN: [1, 0, -1, 1] * 400,
            "elo_home": [1500] * 1600,
            "elo_away": [1400] * 1600,
            "home_avg_goals_last5": [1.5] * 1600,
            "away_avg_goals_last5": [1.0] * 1600,
            "tournament": [
                "Friendly",
                "FIFA World Cup",
                "Friendly",
                "UEFA Euro qualification",
            ]
            * 400,
            "is_friendly": [1, 0, 1, 0] * 400,
            "is_world_cup": [0, 1, 0, 0] * 400,
            "is_qualifier": [0, 0, 0, 1] * 400,
            "is_continental": [0, 0, 0, 0] * 400,
        }
    )

    with (
        patch("backend.modeling.train.load_feature_dataset", return_value=df),
        patch("backend.modeling.train.validate_feature_dataset_contract"),
    ):
        # Reduce backtest_splits to 3 to make it fast while satisfying minimum requirements
        res = train_and_export_model(
            data_path=tmp_path / "dummy.csv",
            artifact_path=artifact_path,
            test_size=0.2,
            calibration_size=0.2,
            backtest_splits=3,
            persist_to_db=True,
            version_tag="test_v1",
        )

    # Artifact generation checks
    assert artifact_path.exists()

    # Versioned artifact check
    versioned_path = tmp_path / "model_test_v1.joblib"
    assert versioned_path.exists()

    # DB persistence check
    assert mock_persist.call_count == 1

    # Output metrics validation
    assert res["selected_model_name"] is not None
    assert "accuracy" in res["metrics"]
    assert "log_loss" in res["metrics"]
