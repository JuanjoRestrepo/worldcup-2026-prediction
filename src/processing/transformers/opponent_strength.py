"""Opponent strength features for modeling match context.

CRITICAL DESIGN NOTES:
- Weighted metrics include wins, draws, and losses (not just wins)
- Index alignment: NO .values assignments (prevents silent errors)  
- ELO ratio: Uses np.clip instead of arithmetic hacks
- All rolling features use .shift(1) to prevent leakage
"""

import numpy as np
import pandas as pd


def compute_opponent_strength(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Compute opponent strength features capturing match difficulty context.
    
    Key insight: Beating Brazil ≠ Beating Bolivia (same win, different opponent quality)
    
    Args:
        df: DataFrame with elo_home, elo_away, teamcolumns and date sorting
        window: Rolling window size for opponent stats (default 5 matches)
        
    Returns:
        DataFrame with added opponent strength columns
    """
    df = df.sort_values("date").reset_index(drop=True)

    # ============================================================================
    # OPPONENT ELO: Current strength of opposing team
    # ============================================================================
    # For home team: opponent = away team
    df["home_opponent_elo"] = df["elo_away"]
    df["away_opponent_elo"] = df["elo_home"]
    
    # Rolling average opponent ELO (recent opponent difficulty)
    df["home_avg_opponent_elo_last5"] = (
        df.groupby("homeTeam")["elo_away"]
        .shift(1)
        .rolling(window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    
    df["away_avg_opponent_elo_last5"] = (
        df.groupby("awayTeam")["elo_home"]
        .shift(1)
        .rolling(window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    
    # ============================================================================
    # WEIGHTED WIN RATE: Result adjusted for opponent difficulty
    #
    # CRITICAL: Include ALL results (wins, draws, losses) weighted by opponent ELO
    #
    # Old approach (WRONG):
    #   weighted_score = 1 * (elo_opponent / 1500)  [only for wins]
    #   Problem: Losing to Brazil = 0 [same as losing to Bolivia]
    #
    # New approach (CORRECT):
    #   score = 1 (win) | 0.5 (draw) | 0 (loss)
    #   weighted_score = score * (elo_opponent / 1500)
    #   Now losing to Brazil is better information than losing to Bolivia
    # ============================================================================
    
    # Match outcomes: 1=win, 0.5=draw, 0=loss (better info than binary)
    df["home_match_score"] = np.where(
        df["homeGoals"] > df["awayGoals"],
        1.0,
        np.where(df["homeGoals"] == df["awayGoals"], 0.5, 0.0)
    )
    
    df["away_match_score"] = np.where(
        df["awayGoals"] > df["homeGoals"],
        1.0,
        np.where(df["awayGoals"] == df["homeGoals"], 0.5, 0.0)
    )
    
    # Normalize opponent ELO to 1500 baseline
    df["home_opponent_elo_norm"] = df["elo_away"] / 1500.0
    df["away_opponent_elo_norm"] = df["elo_home"] / 1500.0
    
    # Weighted score: result × opponent_difficulty
    df["home_weighted_score"] = df["home_match_score"] * df["home_opponent_elo_norm"]
    df["away_weighted_score"] = df["away_match_score"] * df["away_opponent_elo_norm"]
    
    # Rolling weighted win rate (average quality of outcomes, not just count)
    df["home_weighted_win_rate_last5"] = (
        df.groupby("homeTeam")["home_weighted_score"]
        .shift(1)
        .rolling(window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    
    df["away_weighted_win_rate_last5"] = (
        df.groupby("awayTeam")["away_weighted_score"]
        .shift(1)
        .rolling(window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    
    # Clean up temporary columns
    df = df.drop(
        columns=["home_match_score", "away_match_score", 
                 "home_opponent_elo_norm", "away_opponent_elo_norm",
                 "home_weighted_score", "away_weighted_score"],
        errors="ignore"
    )
    
    # ============================================================================
    # ELO RATIO: Multiplicative strength comparison (not additive dif)
    #
    # CRITICAL: Use np.clip to handle division by zero mathematically,
    # NOT arithmetic hacks like elo / (elo + 1)
    # ============================================================================
    
    # Clip away team ELO to avoid division by near-zero (unlikely but safe)
    elo_away_clipped = np.clip(df["elo_away"], 1e-6, None)
    
    # ELO ratio: home_team_power / away_team_power
    # 1.5 = home team 50% stronger
    # 0.67 = away team ~50% stronger  
    df["elo_ratio_home"] = df["elo_home"].astype(float) / elo_away_clipped
    
    # Combined ELO: sum of both teams (match intensity indicator)
    df["combined_elo_strength"] = df["elo_home"].astype(float) + df["elo_away"].astype(float)
    
    # ============================================================================
    # OPPONENT FORM: Opponent's recent ELO trend
    #
    # CRITICAL: Maintain index alignment—NO .values!
    # Using .values can cause silent index misalignment errors
    # ============================================================================
    
    # Home team's opponent form: recent avg ELO of away teams faced
    # Use direct rolling groupby (maintains index alignment automatically)
    df["home_opponent_elo_form"] = (
        df.groupby("homeTeam")["elo_away"]
        .shift(1)
        .rolling(window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    
    # Away team's opponent form: recent avg ELO of home teams faced
    df["away_opponent_elo_form"] = (
        df.groupby("awayTeam")["elo_home"]
        .shift(1)
        .rolling(window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    
    return df
