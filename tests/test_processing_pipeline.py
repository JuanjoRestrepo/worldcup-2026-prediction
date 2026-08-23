"""Tests for processing_pipeline.py."""

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd
from pytest import MonkeyPatch

from backend.processing.pipelines.processing_pipeline import (
    _create_elo_form_features,
    _create_multiclass_target,
    _create_tournament_features,
    run_processing_pipeline,
)


def test_create_tournament_features() -> None:
    """Verify tournament dummy features are correctly flagged."""
    df = pd.DataFrame(
        {"tournament": ["FIFA World Cup", "Friendly", "UEFA Euro qualification"]}
    )
    res = _create_tournament_features(df)
    assert res.iloc[0]["is_world_cup"] == 1
    assert res.iloc[0]["is_friendly"] == 0
    assert res.iloc[1]["is_friendly"] == 1
    assert res.iloc[2]["is_qualifier"] == 1


def test_create_elo_form_features() -> None:
    """Verify ELO rolling form logic, including the shift(1) to avoid leakage."""
    df = pd.DataFrame(
        {
            "homeTeam": ["A", "A", "A"],
            "awayTeam": ["B", "B", "B"],
            "elo_home": [1000, 1100, 1200],
            "elo_away": [1000, 900, 800],
        }
    )
    res = _create_elo_form_features(df, window=2)
    # The first row is NaN because shift(1) pushes previous state.
    assert pd.isna(res.iloc[0]["home_elo_form"])
    assert res.iloc[1]["home_elo_form"] == 1000.0
    assert res.iloc[2]["home_elo_form"] == 1050.0  # mean of 1000 and 1100


def test_create_multiclass_target() -> None:
    """Verify target columns are generated correctly."""
    df = pd.DataFrame(
        {
            "homeGoals": [2, 1, 0],
            "awayGoals": [1, 1, 2],
        }
    )
    res = _create_multiclass_target(df)
    assert res.iloc[0]["target_multiclass"] == 1
    assert res.iloc[1]["target_multiclass"] == 0
    assert res.iloc[2]["target_multiclass"] == -1

    assert res.iloc[0]["target"] == 1
    assert res.iloc[1]["target"] == 0
    assert res.iloc[2]["target"] == 0


@patch("backend.processing.pipelines.processing_pipeline.persist_dataframe")
@patch(
    "backend.processing.pipelines.processing_pipeline.validate_feature_dataset_contract"
)
@patch(
    "backend.processing.pipelines.processing_pipeline.validate_standardized_matches_contract"
)
@patch("backend.processing.pipelines.processing_pipeline.load_api_data")
@patch("backend.processing.pipelines.processing_pipeline.load_historical_data")
def test_run_processing_pipeline(
    mock_load_historical: Any,
    mock_load_api: Any,
    mock_validate_standard: Any,
    mock_validate_feature: Any,
    mock_persist: Any,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """Verify end-to-end execution of the processing pipeline with mocks."""
    from backend.config.settings import settings

    monkeypatch.setattr(settings, "SILVER_DIR", tmp_path)
    monkeypatch.setattr(settings, "GOLD_DIR", tmp_path)

    # Historical data includes a mix of dates to test drift filter.
    mock_load_historical.return_value = pd.DataFrame(
        {
            "date": ["1980-01-01", "2024-01-01", "2024-01-02", "2024-01-03"],
            "home_team": ["A", "A", "C", "A"],
            "away_team": ["B", "B", "D", "C"],
            "home_score": [0, 2, 1, 3],
            "away_score": [0, 1, 1, 0],
            "tournament": ["Friendly", "Friendly", "FIFA World Cup", "Friendly"],
            "city": ["CityA", "CityA", "CityC", "CityA"],
            "country": ["CountryA", "CountryA", "CountryC", "CountryA"],
            "neutral": [False, False, True, False],
        }
    )

    # Return some duplicate API data to test deduplication
    mock_load_api.return_value = pd.DataFrame(
        {
            "date": ["2024-01-02"],
            "homeTeam": ["C"],
            "awayTeam": ["D"],
            "homeGoals": [1],
            "awayGoals": [1],
            "tournament": ["FIFA World Cup"],
            "city": ["CityC"],
            "country": ["CountryC"],
            "neutral": [True],
        }
    )

    # Patch Path.exists for transfermarkt static file used in advanced features
    with patch("pathlib.Path.exists", return_value=False):
        res = run_processing_pipeline(use_api_data=True, persist_to_db=True)

    assert not res.empty

    # 1980 match dropped
    assert len(res) == 3

    # Check key columns from each phase are present
    assert "elo_home" in res.columns
    assert "is_world_cup" in res.columns
    assert "target_multiclass" in res.columns
    assert "home_ewma_goals" in res.columns

    # Verify silver and gold artifacts were persisted
    assert (tmp_path / "matches_cleaned.csv").exists()
    assert (tmp_path / "features_dataset.csv").exists()

    # DB persistence should have been called twice (Silver and Gold)
    assert mock_persist.call_count == 2
