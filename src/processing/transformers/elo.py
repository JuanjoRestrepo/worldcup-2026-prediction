"""ELO rating calculation for team strength estimation."""

import pandas as pd

INITIAL_ELO = 1500
K = 20


def expected_score(ratingA: float, ratingB: float) -> float:
    """Calculate expected score for team A against team B."""
    return 1 / (1 + 10 ** ((ratingB - ratingA) / 400))


def update_elo(ratingA: float, ratingB: float, scoreA: float) -> tuple:
    """
    Update ELO ratings after a match.
    
    Args:
        ratingA: Team A's current ELO
        ratingB: Team B's current ELO
        scoreA: Match result for team A (1=win, 0.5=draw, 0=loss)
        
    Returns:
        Tuple of (new_ratingA, new_ratingB)
    """
    expA = expected_score(ratingA, ratingB)
    newA = ratingA + K * (scoreA - expA)
    newB = ratingB + K * ((1 - scoreA) - (1 - expA))
    return newA, newB


def compute_elo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute ELO ratings for all teams across match history.
    
    Args:
        df: DataFrame with columns: date, homeTeam, awayTeam, homeGoals, awayGoals
        
    Returns:
        DataFrame with added columns: elo_home, elo_away, elo_diff
    """
    ratings = {}
    elo_home = []
    elo_away = []

    df = df.sort_values("date").reset_index(drop=True)

    for _, row in df.iterrows():
        home = row["homeTeam"]
        away = row["awayTeam"]

        # Get current ELO (or initialize)
        rating_home = ratings.get(home, INITIAL_ELO)
        rating_away = ratings.get(away, INITIAL_ELO)

        elo_home.append(rating_home)
        elo_away.append(rating_away)

        # Calculate result (1=home win, 0.5=draw, 0=home loss)
        if row["homeGoals"] > row["awayGoals"]:
            score_home = 1
        elif row["homeGoals"] < row["awayGoals"]:
            score_home = 0
        else:
            score_home = 0.5

        # Update ratings
        new_home, new_away = update_elo(rating_home, rating_away, score_home)
        ratings[home] = new_home
        ratings[away] = new_away

    df["elo_home"] = elo_home
    df["elo_away"] = elo_away
    df["elo_diff"] = df["elo_home"] - df["elo_away"]

    return df
