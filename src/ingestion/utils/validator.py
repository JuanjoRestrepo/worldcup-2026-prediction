import pandas as pd

REQUIRED_COLUMNS = [
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "tournament"
]


def validate_schema(df: pd.DataFrame) -> None:
    """
    Validate dataset schema.

    Raises:
        ValueError if schema is invalid
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")