"""Tests for advanced_features.py."""

from unittest.mock import patch

import numpy as np
import pandas as pd

from backend.processing.transformers.advanced_features import (
    _tournament_pressure_score,
    compute_advanced_features,
)


def test_tournament_pressure_score() -> None:
    """Verify tournament names map to correct pressure scores."""
    assert _tournament_pressure_score("FIFA World Cup") == 1.0
    assert _tournament_pressure_score("Friendly") == 0.20
    assert _tournament_pressure_score("Unknown Tournament") == 0.55


@patch("pathlib.Path.exists", return_value=False)
def test_compute_advanced_features_basic(mock_exists) -> None:  # type: ignore[no-untyped-def]
    """Verify compute_advanced_features returns expected columns and values."""
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    df = pd.DataFrame(
        {
            "date": dates,
            "homeTeam": ["A", "B", "A", "C", "A"],
            "awayTeam": ["B", "C", "C", "A", "B"],
            "homeGoals": [2, 1, 0, 1, 3],
            "awayGoals": [1, 1, 0, 2, 1],
            "tournament": ["Friendly"] * 5,
            "elo_home": [1500, 1400, 1550, 1300, 1600],
            "elo_away": [1400, 1300, 1300, 1500, 1400],
        }
    )

    res = compute_advanced_features(df)

    expected_cols = [
        "tournament_pressure_score",
        "home_days_since_last_match",
        "away_days_since_last_match",
        "home_ewma_goals",
        "home_ewma_conceded",
        "away_ewma_goals",
        "away_ewma_conceded",
        "home_clean_sheet_rate_last10",
        "away_clean_sheet_rate_last10",
        "home_goals_variance_last10",
        "away_goals_variance_last10",
        "h2h_home_win_rate",
        "h2h_draw_rate",
        "h2h_avg_goals",
        "home_confederation_avg_elo",
        "away_confederation_avg_elo",
        "home_squad_value",
        "away_squad_value",
        "talent_differential",
    ]

    for col in expected_cols:
        assert col in res.columns

    # Check tournament pressure
    assert res.iloc[0]["tournament_pressure_score"] == 0.20

    # Rest days (first match has nan which fills to 30.0)
    assert res.iloc[0]["home_days_since_last_match"] == 30.0
    assert res.iloc[0]["away_days_since_last_match"] == 30.0

    # A's next match is row 2 (date: 2024-01-03).
    # A was home in row 0 (2024-01-01). 2 days difference.
    assert res.iloc[2]["home_days_since_last_match"] == 2.0


@patch("pathlib.Path.exists", return_value=True)
def test_talent_differential_with_tm_data(mock_exists, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Verify talent differential computation works when TM file exists."""
    df = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-01")],
            "homeTeam": ["Argentina"],
            "awayTeam": ["France"],
            "homeGoals": [0],
            "awayGoals": [0],
            "tournament": ["Friendly"],
            "elo_home": [1500],
            "elo_away": [1500],
        }
    )

    tm_df = pd.DataFrame(
        {
            "Nation": ["Argentina", "France", "Unknown"],
            "Total_Value_Num": [10_000_000.0, 20_000_000.0, 5_000.0],
        }
    )

    with patch("pandas.read_csv", return_value=tm_df):
        res = compute_advanced_features(df)

    # Fuzzy matching should link Argentina and France
    assert res["home_squad_value"].iloc[0] == 10_000_000.0
    assert res["away_squad_value"].iloc[0] == 20_000_000.0

    # differential should be log(10M + 1) - log(20M + 1)
    diff = np.log1p(10_000_000.0) - np.log1p(20_000_000.0)
    assert np.isclose(res["talent_differential"].iloc[0], diff)
