"""Standardize match data from different sources to common format."""

from typing import Any

import pandas as pd


def standardize_api(matches: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Convert API match data to standardized DataFrame format.

    Args:
        matches: List of match dictionaries from API

    Returns:
        DataFrame with standardized columns
    """
    standardized_rows = []

    for match in matches:
        try:
            row = {
                "date": match.get("utcDate", ""),
                "homeTeam": match.get("homeTeam", {}).get("name", "Unknown"),
                "awayTeam": match.get("awayTeam", {}).get("name", "Unknown"),
                "homeGoals": match.get("score", {}).get("fullTime", {}).get("home"),
                "awayGoals": match.get("score", {}).get("fullTime", {}).get("away"),
                "competition": match.get("competition", {}).get("name", "Unknown"),
                "tournament": match.get("area", {}).get("name", "Unknown"),
                "status": match.get("status", "UNKNOWN"),
            }
            standardized_rows.append(row)
        except Exception as e:
            print(f"Error standardizing match: {e}")
            continue

    df = pd.DataFrame(standardized_rows)
    return df


def standardize_csv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure CSV format matches standardized schema.

    Args:
        df: DataFrame from CSV with columns like 'date', 'home_team', 'away_team', etc.

    Returns:
        Standardized DataFrame
    """
    # Rename columns to standard format (if needed)
    rename_map = {
        "home_team": "homeTeam",
        "away_team": "awayTeam",
        "home_score": "homeGoals",
        "away_score": "awayGoals",
    }

    # Only rename columns that exist
    df_renamed = df.rename(
        columns={k: v for k, v in rename_map.items() if k in df.columns}
    )

    return df_renamed
