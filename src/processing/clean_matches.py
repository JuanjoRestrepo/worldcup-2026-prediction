import pandas as pd


def clean_matches(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and normalize match dataset.
    """

    df = df.copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df["home_team"] = df["home_team"].str.strip()
    df["away_team"] = df["away_team"].str.strip()

    df = df.dropna(subset=["home_score", "away_score"])

    df["goal_diff"] = df["home_score"] - df["away_score"]

    return df