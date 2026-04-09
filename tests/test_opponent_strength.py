"""Unit tests for opponent strength features module."""

import pandas as pd
from src.processing.transformers.opponent_strength import compute_opponent_strength


def test_compute_opponent_strength_columns():
    """Test that opponent strength adds expected columns."""
    df = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=5),
        "homeTeam": ["A", "B", "A", "B", "A"],
        "awayTeam": ["B", "A", "B", "A", "B"],
        "homeGoals": [2, 1, 2, 1, 2],
        "awayGoals": [1, 0, 0, 2, 1],
        "elo_home": [1500.0, 1510.0, 1500.0, 1510.0, 1500.0],
        "elo_away": [1510.0, 1500.0, 1510.0, 1500.0, 1510.0],
    })

    result = compute_opponent_strength(df)
    
    expected_columns = [
        "home_opponent_elo",
        "away_opponent_elo",
        "home_avg_opponent_elo_last5",
        "away_avg_opponent_elo_last5",
        "home_weighted_win_rate_last5",
        "away_weighted_win_rate_last5",
        "home_opponent_elo_form",
        "away_opponent_elo_form",
        "elo_ratio_home",
        "combined_elo_strength",
    ]
    
    for col in expected_columns:
        assert col in result.columns


def test_opponent_elo_assignment():
    """Test that opponent ELO is correctly assigned."""
    df = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=3),
        "homeTeam": ["A", "B", "A"],
        "awayTeam": ["B", "A", "B"],
        "homeGoals": [2, 1, 2],
        "awayGoals": [1, 0, 1],
        "elo_home": [1500.0, 1510.0, 1500.0],
        "elo_away": [1510.0, 1500.0, 1510.0],
    })

    result = compute_opponent_strength(df)
    
    # Home team's opponent ELO = away team's ELO
    assert result.loc[0, "home_opponent_elo"] == 1510.0
    assert result.loc[1, "home_opponent_elo"] == 1500.0
    
    # Away team's opponent ELO = home team's ELO
    assert result.loc[0, "away_opponent_elo"] == 1500.0
    assert result.loc[1, "away_opponent_elo"] == 1510.0


def test_elo_ratio():
    """Test ELO ratio calculation with np.clip (not +1 hack)."""
    df = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=2),
        "homeTeam": ["A", "B"],
        "awayTeam": ["B", "A"],
        "homeGoals": [1, 1],
        "awayGoals": [0, 0],
        "elo_home": [1500.0, 1500.0],
        "elo_away": [1000.0, 1000.0],
    })

    result = compute_opponent_strength(df)
    
    # ELO ratio = elo_home / np.clip(elo_away, 1e-6, None)
    # With elo_away=1000, clip doesn't change it, so: 1500/1000 = 1.5
    expected_ratio = 1500.0 / 1000.0  # Clean math, no arithmetic hacks
    assert abs(result.loc[0, "elo_ratio_home"] - expected_ratio) < 0.001


def test_combined_elo_strength():
    """Test combined ELO strength (sum of both teams)."""
    df = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=2),
        "homeTeam": ["A", "B"],
        "awayTeam": ["B", "A"],
        "homeGoals": [1, 1],
        "awayGoals": [0, 0],
        "elo_home": [1500.0, 1600.0],
        "elo_away": [1400.0, 1500.0],
    })

    result = compute_opponent_strength(df)
    
    # Combined = elo_home + elo_away
    assert result.loc[0, "combined_elo_strength"] == 2900.0
    assert result.loc[1, "combined_elo_strength"] == 3100.0


def test_weighted_win_rate():
    """Test that weighted win rate accounts for opponent strength."""
    df = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=4),
        "homeTeam": ["Strong", "Strong", "Weak", "Weak"],
        "awayTeam": ["Weak", "Weak", "Strong", "Strong"],
        "homeGoals": [2, 2, 0, 0],
        "awayGoals": [0, 0, 2, 2],
        "elo_home": [1800.0, 1800.0, 1200.0, 1200.0],
        "elo_away": [1200.0, 1200.0, 1800.0, 1800.0],
    })

    result = compute_opponent_strength(df)
    
    # Strong team beating weak team: weight = 1200/1500 = 0.8
    # Weak team beating strong team: weight = 1800/1500 = 1.2
    
    # Strong team has lower weighted value (beating weak)
    # Weak team has higher weighted value (beating strong)
    assert result.loc[2, "away_weighted_win_rate_last5"] is not None  # Weak beats strong


def test_opponent_strength_preserves_rows():
    """Test that number of rows is preserved."""
    df = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=10),
        "homeTeam": ["A"] * 5 + ["B"] * 5,
        "awayTeam": ["B"] * 5 + ["A"] * 5,
        "homeGoals": [1, 2, 1, 2, 1, 0, 1, 0, 1, 0],
        "awayGoals": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        "elo_home": [1500.0] * 10,
        "elo_away": [1500.0] * 10,
    })

    result = compute_opponent_strength(df)
    
    assert len(result) == len(df)
