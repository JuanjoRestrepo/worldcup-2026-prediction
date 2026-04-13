"""Unit tests for rolling features module with leakage prevention."""

import pandas as pd

from src.processing.transformers.rolling_features import compute_rolling_features


def test_compute_rolling_features_columns():
    """Test that rolling features adds expected columns."""
    df = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=10),
            "homeTeam": ["Brazil"] * 5 + ["Argentina"] * 5,
            "awayTeam": ["Argentina"] * 5 + ["Brazil"] * 5,
            "homeGoals": [2, 1, 1, 0, 2, 2, 1, 1, 0, 2],
            "awayGoals": [1, 1, 1, 2, 0, 1, 1, 1, 2, 0],
        }
    )

    result = compute_rolling_features(df, window=5)

    expected_columns = [
        "goal_diff",
        "home_avg_goals_last5",
        "away_avg_goals_conceded_last5",
        "away_avg_goals_last5",
        "home_avg_goals_conceded_last5",
        "home_win_rate_last5",
        "away_win_rate_last5",
    ]

    for col in expected_columns:
        assert col in result.columns


def test_compute_rolling_features_leakage_prevention():
    """Test that shift(1) prevents data leakage (first game should be NaN or min_periods=1)."""
    df = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=5),
            "homeTeam": ["A", "A", "A", "A", "A"],
            "awayTeam": ["B", "B", "B", "B", "B"],
            "homeGoals": [5, 5, 5, 5, 5],
            "awayGoals": [0, 0, 0, 0, 0],
        }
    )

    result = compute_rolling_features(df, window=5)

    # First row: shift(1) means no past data, min_periods=1 means NaN
    assert pd.isna(result.loc[0, "home_avg_goals_last5"])

    # Second row: should average just first match (5 goals)
    assert result.loc[1, "home_avg_goals_last5"] == 5.0

    # Third row: should average first two matches (5 + 5) / 2 = 5.0
    assert result.loc[2, "home_avg_goals_last5"] == 5.0


def test_compute_rolling_features_calculation():
    """Test that rolling features are calculated correctly."""
    df = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=6),
            "homeTeam": ["A", "A", "A", "A", "A", "A"],
            "awayTeam": ["B", "B", "B", "B", "B", "B"],
            "homeGoals": [1, 2, 3, 2, 1, 2],
            "awayGoals": [0, 0, 0, 0, 0, 0],
        }
    )

    result = compute_rolling_features(df, window=3)

    # Row 0: no past data
    assert pd.isna(result.loc[0, "home_avg_goals_last5"])

    # Row 1: average of row 0 only (1 goal)
    assert result.loc[1, "home_avg_goals_last5"] == 1.0

    # Row 2: average of rows 0-1 (1+2)/2 = 1.5
    assert result.loc[2, "home_avg_goals_last5"] == 1.5

    # Row 3: average of rows 0-2 (1+2+3)/3 = 2.0
    assert result.loc[3, "home_avg_goals_last5"] == 2.0


def test_compute_rolling_features_goal_diff():
    """Test that goal_diff is calculated correctly."""
    df = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=3),
            "homeTeam": ["A", "B", "A"],
            "awayTeam": ["B", "A", "B"],
            "homeGoals": [2, 1, 3],
            "awayGoals": [1, 0, 2],
        }
    )

    result = compute_rolling_features(df)

    assert result.loc[0, "goal_diff"] == 1
    assert result.loc[1, "goal_diff"] == 1
    assert result.loc[2, "goal_diff"] == 1


def test_compute_rolling_features_win_rate():
    """Test that win rate rolling features are calculated correctly."""
    df = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=5),
            "homeTeam": ["A", "A", "A", "A", "A"],
            "awayTeam": ["B", "B", "B", "B", "B"],
            "homeGoals": [2, 1, 2, 1, 2],  # A wins 4 out of 5
            "awayGoals": [0, 0, 0, 0, 0],
        }
    )

    result = compute_rolling_features(df, window=3)

    # Row 0: no past data
    assert pd.isna(result.loc[0, "home_win_rate_last5"])

    # Row 1: A won 1 out of 1 past match = 1.0
    assert result.loc[1, "home_win_rate_last5"] == 1.0

    # Row 2: A won 2 out of 2 past matches = 1.0
    assert result.loc[2, "home_win_rate_last5"] == 1.0

    # Row 3: A won 3 out of 3 past matches = 1.0
    assert result.loc[3, "home_win_rate_last5"] == 1.0


def test_compute_rolling_features_window():
    """Test that rolling window parameter works correctly."""
    df = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=10),
            "homeTeam": ["A"] * 10,
            "awayTeam": ["B"] * 10,
            "homeGoals": [1] * 10,
            "awayGoals": [0] * 10,
        }
    )

    result = compute_rolling_features(df, window=5)

    # After 5+ games, rolling average should be 1.0
    assert result.loc[5:, "home_avg_goals_last5"].mean() == 1.0


def test_compute_rolling_features_null_handling():
    """Test that rolling features handle initial NaN values correctly."""
    df = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=5),
            "homeTeam": ["A", "A", "A", "A", "A"],
            "awayTeam": ["B", "B", "B", "B", "B"],
            "homeGoals": [1, 2, 3, 2, 1],
            "awayGoals": [0, 0, 0, 0, 0],
        }
    )

    result = compute_rolling_features(df, window=5)

    # First row should be NaN due to shift(1)
    assert pd.isna(result.loc[0, "home_avg_goals_last5"])

    # Rest should not be NaN (min_periods=1)
    assert result.loc[1:, "home_avg_goals_last5"].isna().sum() == 0
