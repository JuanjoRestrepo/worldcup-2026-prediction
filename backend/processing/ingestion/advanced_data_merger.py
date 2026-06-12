"""Silver transformation layer: Merges Bronze scraped data into our main dataset using fuzzy matching."""

import logging
from pathlib import Path

import pandas as pd
from thefuzz import process

logger = logging.getLogger(__name__)

BRONZE_DIR = Path("data/bronze")
SILVER_DIR = Path("data/silver")
SILVER_DIR.mkdir(parents=True, exist_ok=True)


def fuzzy_match_teams(
    target_teams: list[str], source_teams: list[str], threshold: int = 85
) -> dict[str, str]:
    """Matches team names from an external source to our Kaggle dataset.

    Args:
        target_teams: The canonical team names (from matches_cleaned.csv).
        source_teams: The scraped team names (from FBref/SoFIFA).
        threshold: The Levenshtein distance threshold for a confident match.

    Returns:
        dict: Mapping of source_team -> target_team.
    """
    mapping = {}
    for src in source_teams:
        match, score = process.extractOne(src, target_teams)
        if score >= threshold:
            mapping[src] = match
        else:
            logger.warning(
                f"Could not confidently match '{src}' (Best match: '{match}' at {score}%)"
            )
    return mapping


def integrate_fbref_features(
    base_df: pd.DataFrame, fbref_df: pd.DataFrame, team_mapping: dict[str, str]
) -> pd.DataFrame:
    """Merges FBref match-level features (xG, Possession) into the historical dataset."""
    # This assumes fbref_df has 'team', 'date', 'xg', 'possession', etc.
    # Implementation depends on the exact shape of the FBref dataframe.
    # Currently a placeholder for the merging logic once FBref data is confirmed.
    logger.info("FBref integration placeholder triggered.")
    return base_df


def run_silver_transformation() -> None:
    """Execute the full Silver transformation pipeline."""
    logger.info("Starting Silver transformation...")

    try:
        base_matches = pd.read_csv(SILVER_DIR / "matches_cleaned.csv")
    except FileNotFoundError:
        logger.error("matches_cleaned.csv not found in Silver directory.")
        return

    canonical_teams = list(
        set(base_matches["team_home"].unique())
        | set(base_matches["team_away"].unique())
    )

    fbref_path = BRONZE_DIR / "fbref_raw.parquet"
    if fbref_path.exists():
        logger.info("Found FBref Bronze data. Beginning integration...")
        fbref_df = pd.read_parquet(fbref_path)

        # Typical FBref team column is 'team'
        if "team" in fbref_df.columns:
            source_teams = fbref_df["team"].dropna().unique().tolist()
            mapping = fuzzy_match_teams(canonical_teams, source_teams)
            logger.info(f"Mapped {len(mapping)} FBref teams successfully.")
            # We would integrate here once columns are mapped out
        else:
            logger.warning("FBref data missing 'team' column. Cannot match.")
    else:
        logger.warning("FBref Bronze data not found. Skipping FBref integration.")

    logger.info("Silver transformation completed.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_silver_transformation()
