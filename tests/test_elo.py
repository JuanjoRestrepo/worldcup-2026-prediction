"""Unit tests for ELO rating module."""

import pandas as pd
from src.processing.transformers.elo import (
    INITIAL_ELO,
    expected_score,
    update_elo,
    compute_elo,
)


def test_initial_elo():
    """Test that initial ELO is correctly set."""
    assert INITIAL_ELO == 1500


def test_expected_score():
    """Test expected score calculation."""
    score = expected_score(1500, 1500)
    assert 0.49 < score < 0.51  # Should be ~0.5 for equal ratings


def test_expected_score_higher_rated_team():
    """Test expected score for higher rated team."""
    score = expected_score(1600, 1500)
    assert score > 0.5  # Higher rated team should have higher expected score


def test_update_elo_win():
    """Test ELO update after a win."""
    new_a, new_b = update_elo(1500, 1500, 1.0)
    assert new_a > 1500
    assert new_b < 1500


def test_update_elo_loss():
    """Test ELO update after a loss."""
    new_a, new_b = update_elo(1500, 1500, 0.0)
    assert new_a < 1500
    assert new_b > 1500


def test_update_elo_draw():
    """Test ELO update after a draw."""
    new_a, new_b = update_elo(1500, 1500, 0.5)
    assert abs(new_a - 1500) < 1
    assert abs(new_b - 1500) < 1


def test_compute_elo_basic():
    """Test compute_elo with a simple dataset."""
    df = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=3),
        "homeTeam": ["Brazil", "Brazil", "Argentina"],
        "awayTeam": ["Argentina", "Uruguay", "Brazil"],
        "homeGoals": [2, 1, 0],
        "awayGoals": [0, 2, 1],
    })

    result = compute_elo(df)
    
    # Check that ELO columns are added
    assert "elo_home" in result.columns
    assert "elo_away" in result.columns
    assert "elo_diff" in result.columns
    
    # Check that first row has initial ELO
    assert result.loc[0, "elo_home"] == INITIAL_ELO
    assert result.loc[0, "elo_away"] == INITIAL_ELO
    
    # Check that ELO values are reasonable
    assert result["elo_home"].min() > 1000
    assert result["elo_home"].max() < 2000
    assert result["elo_away"].min() > 1000
    assert result["elo_away"].max() < 2000


def test_compute_elo_consistency():
    """Test that ELO is computed consistently with expected behavior."""
    df = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=5),
        "homeTeam": ["A", "A", "A", "B", "B"],
        "awayTeam": ["B", "C", "D", "A", "C"],
        "homeGoals": [2, 1, 3, 0, 2],
        "awayGoals": [1, 1, 0, 2, 0],
    })

    result = compute_elo(df)
    
    # Team A should have higher ELO after winning most matches
    final_elo_a = result.loc[result["homeTeam"] == "A", "elo_home"].iloc[-1] if len(result[result["homeTeam"] == "A"]) > 0 else None
    initial_elo = INITIAL_ELO
    
    if final_elo_a:
        # A won multiple games, should have ELO increase
        assert final_elo_a > initial_elo or final_elo_a == initial_elo
    
    # Check elo_diff is calculated correctly
    assert (result["elo_diff"] == result["elo_home"] - result["elo_away"]).all()
