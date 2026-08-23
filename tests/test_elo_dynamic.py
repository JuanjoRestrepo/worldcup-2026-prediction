"""Unit tests for dynamic ELO K-factors and goal margin multipliers."""

from __future__ import annotations

import pandas as pd

from backend.processing.transformers.elo import (
    compute_elo,
    get_goal_margin_multiplier,
    get_tournament_k_factor,
    update_elo,
)


def test_get_tournament_k_factor() -> None:
    """Verify tournament importance scaling for ELO."""
    assert get_tournament_k_factor("FIFA World Cup", is_knockout=False) == 50.0
    assert get_tournament_k_factor("FIFA World Cup", is_knockout=True) == 60.0
    assert get_tournament_k_factor("UEFA Euro") == 40.0
    assert get_tournament_k_factor("FIFA World Cup Qualification") == 30.0
    assert get_tournament_k_factor("Friendly") == 15.0


def test_get_goal_margin_multiplier() -> None:
    """Verify goal margin multipliers."""
    assert get_goal_margin_multiplier(1, 0) == 1.0
    assert get_goal_margin_multiplier(2, 2) == 1.0
    assert get_goal_margin_multiplier(3, 1) == 1.5
    assert get_goal_margin_multiplier(4, 0) == 1.875


def test_update_elo_dynamic() -> None:
    """Verify dynamic ELO update delta scales with K-factor."""
    new_a_low, new_b_low = update_elo(1500.0, 1500.0, score_a=1.0, k_factor=15.0)
    new_a_high, new_b_high = update_elo(1500.0, 1500.0, score_a=1.0, k_factor=60.0)

    delta_low = new_a_low - 1500.0
    delta_high = new_a_high - 1500.0

    assert delta_low == 7.5
    assert delta_high == 30.0


def test_compute_elo_with_dynamic_k() -> None:
    """Verify compute_elo processes history without errors and generates expected columns."""
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-06-01", "2026-06-15", "2026-07-04"]),
            "homeTeam": ["Team A", "Team B", "Team A"],
            "awayTeam": ["Team B", "Team C", "Team C"],
            "homeGoals": [2, 0, 3],
            "awayGoals": [0, 1, 0],
            "tournament": ["Friendly", "FIFA World Cup", "FIFA World Cup"],
            "is_knockout": [0, 0, 1],
        }
    )

    res = compute_elo(df)
    assert "elo_home" in res.columns
    assert "elo_away" in res.columns
    assert "elo_diff" in res.columns
    assert len(res) == 3
