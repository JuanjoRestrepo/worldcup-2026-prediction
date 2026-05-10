"""Rolling features for team performance metrics with true global features.

ARCHITECTURE:
- Position-specific: Captures role-dependent performance (home/away)
- Global (true): Built from UNION of all matches, not split by position
- Leakage: Prevented via .shift(1) on all rolling computations
- Vectorization: Pure pandas groupby().rolling() (no loops, O(n log n))
"""

import numpy as np
import pandas as pd


def compute_rolling_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Compute rolling features: position-specific AND true global.

    LEAKAGE PREVENTION: All rolling aggregates use .shift(1) to prevent
    current-match information from leaking into historical aggregates.

    Args:
        df: DataFrame with match data, sorted by date
        window: Rolling window size (default 5 games)

    Returns:
        DataFrame with added rolling feature columns
    """
    df = df.sort_values("date").reset_index(drop=True)

    # ============================================================================
    # GOAL DIFFERENTIAL
    # ============================================================================
    df["goal_diff"] = df["homeGoals"] - df["awayGoals"]

    # ============================================================================
    # POSITION-SPECIFIC FEATURES (condition on home/away role)
    # ============================================================================
    df["home_avg_goals_last5"] = (
        df.groupby("homeTeam")["homeGoals"]
        .shift(1)
        .rolling(window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["home_avg_goals_conceded_last5"] = (
        df.groupby("homeTeam")["awayGoals"]
        .shift(1)
        .rolling(window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["away_avg_goals_last5"] = (
        df.groupby("awayTeam")["awayGoals"]
        .shift(1)
        .rolling(window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["away_avg_goals_conceded_last5"] = (
        df.groupby("awayTeam")["homeGoals"]
        .shift(1)
        .rolling(window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # ============================================================================
    # TRUE GLOBAL FEATURES (aggregate ALL matches via long-format union)
    # ============================================================================

    # Build long-format: each match contributes TWO rows (home & away perspectives)
    home_perspective = (
        df[["date", "homeTeam", "homeGoals", "awayGoals"]]
        .rename(
            columns={
                "homeTeam": "team",
                "homeGoals": "goals_for",
                "awayGoals": "goals_against",
            }
        )
        .copy()
    )
    home_perspective["match_idx"] = range(len(df))  # Track original match

    away_perspective = (
        df[["date", "awayTeam", "awayGoals", "homeGoals"]]
        .rename(
            columns={
                "awayTeam": "team",
                "awayGoals": "goals_for",
                "homeGoals": "goals_against",
            }
        )
        .copy()
    )
    away_perspective["match_idx"] = range(len(df))  # Same frame indices

    long_df = pd.concat([home_perspective, away_perspective], ignore_index=True)
    long_df = long_df.sort_values(["team", "date"]).reset_index(drop=True)

    # Compute global rolling stats (all matches for each team)
    long_df["global_avg_goals_for"] = (
        long_df.groupby("team")["goals_for"]
        .shift(1)
        .rolling(window, min_periods=1)
        .mean()
        .values  # Convert to array for clean assignment
    )

    long_df["global_avg_goals_against"] = (
        long_df.groupby("team")["goals_against"]
        .shift(1)
        .rolling(window, min_periods=1)
        .mean()
        .values
    )

    # Split back: home rows have indices 0..len(df)-1, away rows have indices len(df)..2*len(df)-1
    home_global = long_df.iloc[: len(df)][
        ["global_avg_goals_for", "global_avg_goals_against"]
    ].reset_index(drop=True)
    away_global = long_df.iloc[len(df) :][
        ["global_avg_goals_for", "global_avg_goals_against"]
    ].reset_index(drop=True)

    df["home_global_avg_goals_last5"] = home_global["global_avg_goals_for"].values
    df["home_global_avg_conceded_last5"] = home_global[
        "global_avg_goals_against"
    ].values
    df["away_global_avg_goals_last5"] = away_global["global_avg_goals_for"].values
    df["away_global_avg_conceded_last5"] = away_global[
        "global_avg_goals_against"
    ].values

    # ============================================================================
    # WIN RATES: POSITION-SPECIFIC
    # (encode draws as 0.5, better information than binary 0/1)
    # ============================================================================
    df["home_result"] = np.where(
        df["homeGoals"] > df["awayGoals"],
        1.0,  # Win
        np.where(
            df["homeGoals"] == df["awayGoals"],
            0.5,  # Draw (partial credit)
            0.0,  # Loss
        ),
    )

    df["away_result"] = np.where(
        df["awayGoals"] > df["homeGoals"],
        1.0,  # Win
        np.where(
            df["awayGoals"] == df["homeGoals"],
            0.5,  # Draw (partial credit)
            0.0,  # Loss
        ),
    )

    df["home_win_rate_last5"] = (
        df.groupby("homeTeam")["home_result"]
        .shift(1)
        .rolling(window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["away_win_rate_last5"] = (
        df.groupby("awayTeam")["away_result"]
        .shift(1)
        .rolling(window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # ============================================================================
    # WIN RATES: TRUE GLOBAL
    # (computed from long format, all matches aggregated)
    # ============================================================================
    long_df["result"] = np.where(
        long_df["goals_for"] > long_df["goals_against"],
        1.0,
        np.where(long_df["goals_for"] == long_df["goals_against"], 0.5, 0.0),
    )

    long_df["global_win_rate"] = (
        long_df.groupby("team")["result"]
        .shift(1)
        .rolling(window, min_periods=1)
        .mean()
        .values
    )

    # Split back by original match indices
    home_global_wr = long_df.iloc[: len(df)][["global_win_rate"]].reset_index(drop=True)
    away_global_wr = long_df.iloc[len(df) :][["global_win_rate"]].reset_index(drop=True)

    df["home_global_win_rate_last5"] = home_global_wr["global_win_rate"].values
    df["away_global_win_rate_last5"] = away_global_wr["global_win_rate"].values

    # ============================================================================
    # HOME ADVANTAGE EFFECT
    # ============================================================================
    df["home_advantage_effect"] = df["home_win_rate_last5"] - df["away_win_rate_last5"]

    # ============================================================================
    # DRAW RATE: POSITION-SPECIFIC (explicit signal for draw specialist)
    # Binary (1.0 = draw, 0.0 = decisive) to measure each team's propensity
    # to draw when playing at home vs. away — distinct from global win_rate.
    # ============================================================================
    df["is_draw"] = (df["homeGoals"] == df["awayGoals"]).astype(float)

    df["home_draw_rate_last5"] = (
        df.groupby("homeTeam")["is_draw"]
        .shift(1)
        .rolling(window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["away_draw_rate_last5"] = (
        df.groupby("awayTeam")["is_draw"]
        .shift(1)
        .rolling(window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # Clean up temporary columns
    df = df.drop(columns=["home_result", "away_result", "is_draw"], errors="ignore")

    return df
