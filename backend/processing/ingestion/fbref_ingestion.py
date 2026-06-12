"""Ingestion module for pulling Opta event data via FBref using soccerdata."""

import logging
from pathlib import Path

import pandas as pd
import soccerdata as sd

logger = logging.getLogger(__name__)

# Base output path matching Medallion architecture
BRONZE_DIR = Path("data/bronze")
BRONZE_DIR.mkdir(parents=True, exist_ok=True)


def extract_fbref_match_stats(
    leagues: list[str] = ["INT-World Cup", "INT-European Championship"],
    seasons: int | list[int] = [2022, 2024],
    export_path: Path = BRONZE_DIR / "fbref_raw.parquet",
) -> pd.DataFrame:
    """Extract historical match stats (possession, xG) from FBref.

    Args:
        leagues: List of soccerdata league codes.
        seasons: Target seasons/years.
        export_path: Destination for the bronze output.

    Returns:
        pd.DataFrame: The scraped DataFrame containing possession and match details.
    """
    logger.info(f"Initializing FBref scraper for {leagues} (Seasons: {seasons})...")

    try:
        fbref = sd.FBref(leagues=leagues, seasons=seasons)

        logger.info("Fetching possession match stats...")
        df_possession = fbref.read_team_match_stats(stat_type="schedule")

        logger.info("Fetching summary match stats (for xG)...")
        df_summary = fbref.read_team_match_stats(stat_type="shooting")

        # Merge both outputs to have a comprehensive event-level view
        df_merged = pd.merge(
            df_possession.reset_index(),
            df_summary.reset_index(),
            how="outer",
            suffixes=("_poss", "_summ"),
        )

        logger.info(f"Successfully scraped {len(df_merged)} event rows from FBref.")

        df_merged.to_parquet(export_path, index=False)
        logger.info(f"Exported FBref raw data to {export_path}")

        return df_merged
    except Exception as e:
        logger.error(f"Failed to extract FBref data: {e}")
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    extract_fbref_match_stats()
