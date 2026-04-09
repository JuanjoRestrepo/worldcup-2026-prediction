"""Unit tests for formal dataset contracts."""

from __future__ import annotations

import pandas as pd
import pytest

from src.contracts.data_contracts import (
    DataContractError,
    validate_feature_dataset_contract,
    validate_historical_raw_contract,
    validate_standardized_matches_contract,
    validate_training_summary_contract,
)


def test_validate_historical_raw_contract_accepts_valid_frame():
    df = pd.DataFrame(
        {
            "date": ["2024-01-01"],
            "home_team": ["Colombia"],
            "away_team": ["Argentina"],
            "home_score": [1],
            "away_score": [0],
            "tournament": ["Friendly"],
        }
    )

    validate_historical_raw_contract(df)


def test_validate_historical_raw_contract_rejects_missing_columns():
    df = pd.DataFrame(
        {
            "date": ["2024-01-01"],
            "home_team": ["Colombia"],
        }
    )

    with pytest.raises(DataContractError, match="missing columns"):
        validate_historical_raw_contract(df)


def test_validate_standardized_matches_contract_rejects_negative_scores():
    df = pd.DataFrame(
        {
            "date": ["2024-01-01"],
            "homeTeam": ["Colombia"],
            "awayTeam": ["Argentina"],
            "homeGoals": [-1],
            "awayGoals": [0],
            "tournament": ["Friendly"],
        }
    )

    with pytest.raises(DataContractError, match="below 0"):
        validate_standardized_matches_contract(df)


def test_validate_feature_dataset_contract_rejects_invalid_targets():
    df = pd.DataFrame(
        {
            "date": ["2024-01-01"],
            "homeTeam": ["Colombia"],
            "awayTeam": ["Argentina"],
            "homeGoals": [1],
            "awayGoals": [0],
            "elo_home": [1800.0],
            "elo_away": [1900.0],
            "elo_diff": [-100.0],
            "target_multiclass": [2],
            "target": [1],
        }
    )

    with pytest.raises(DataContractError, match="target_multiclass"):
        validate_feature_dataset_contract(df)


def test_validate_training_summary_contract_rejects_out_of_range_metrics():
    training_summary = {
        "artifact_path": "models/model.joblib",
        "data_path": "data/gold/features_dataset.csv",
        "training_rows": 100,
        "test_rows": 20,
        "feature_count": 10,
        "feature_columns": ["elo_home"],
        "train_date_range": {"start": "2020-01-01", "end": "2024-01-01"},
        "test_date_range": {"start": "2024-01-02", "end": "2025-01-01"},
        "class_distribution_train": {"-1": 30, "0": 20, "1": 50},
        "class_distribution_test": {"-1": 5, "0": 5, "1": 10},
        "metrics": {
            "accuracy": 1.2,
            "macro_f1": 0.5,
            "weighted_f1": 0.6,
            "log_loss": 0.9,
        },
    }

    with pytest.raises(DataContractError, match="accuracy"):
        validate_training_summary_contract(training_summary)
