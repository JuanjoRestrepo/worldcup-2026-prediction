"""Ingestion module for pulling EA FC / SoFIFA squad depths."""

import json
import logging
from pathlib import Path

import pandas as pd
import soccerdata as sd

logger = logging.getLogger(__name__)

BRONZE_DIR = Path("data/bronze")
BRONZE_DIR.mkdir(parents=True, exist_ok=True)


# We must inject a custom mapping into soccerdata's config to force SoFIFA
# to scrape National Teams, since it restricts to domestic leagues by default.
def inject_international_sofifa_config() -> None:
    """Injects 'INT-National' into soccerdata's custom league_dict.json."""
    base_dir = Path.home() / "soccerdata" / "config"
    base_dir.mkdir(parents=True, exist_ok=True)

    league_file = base_dir / "league_dict.json"

    custom_leagues = {}
    if league_file.exists():
        with open(league_file) as f:
            try:
                custom_leagues = json.load(f)
            except json.JSONDecodeError:
                custom_leagues = {}

    # "[World] International" is SoFIFA's typical internal structure for national teams.
    # We will also add a fallback just in case.
    if "INT-National" not in custom_leagues:
        custom_leagues["INT-National"] = {"SoFIFA": "[World] International"}
        with open(league_file, "w") as f:
            json.dump(custom_leagues, f, indent=4)
        logger.info("Injected INT-National mapping into soccerdata league_dict.json")


def extract_sofifa_squad_depth(
    leagues: list[str] = ["INT-National"],
    export_path: Path = BRONZE_DIR / "sofifa_raw.parquet",
) -> pd.DataFrame:
    """Extract official EA FC team rosters and overall ratings from SoFIFA.

    Args:
        leagues: Target leagues (custom mapped to National Teams).
        export_path: Destination for bronze output.
    """
    logger.info("Injecting custom SoFIFA configuration...")
    inject_international_sofifa_config()

    try:
        # Re-import to ensure config picks up the new JSON on initialization
        import importlib

        importlib.reload(sd._config)

        logger.info(f"Initializing SoFIFA scraper for {leagues}...")
        sofifa = sd.SoFIFA(leagues=leagues)

        logger.info("Fetching team rosters and player Overall Ratings (OVR)...")
        df_rosters = sofifa.read_team_players()

        logger.info(f"Successfully scraped {len(df_rosters)} players from SoFIFA.")

        df_rosters.reset_index().to_parquet(export_path, index=False)
        logger.info(f"Exported SoFIFA raw data to {export_path}")

        return df_rosters.reset_index()  # type: ignore[no-any-return]
    except Exception as e:
        logger.error(f"Failed to extract SoFIFA data: {e}")
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    extract_sofifa_squad_depth()
