"""Load and process API data for integration into processing pipeline."""

import json
import logging
import pandas as pd

from src.config.settings import settings

logger = logging.getLogger(__name__)

API_DATA_DIR = settings.RAW_DIR


def load_api_data() -> pd.DataFrame:
    """
    Load API JSON data files and convert to standardized DataFrame.
    
    Returns:
        DataFrame with API match data (empty DataFrame if no files found)
    """
    api_files = list(API_DATA_DIR.glob("api_international_matches_*.json"))
    
    if not api_files:
        logger.warning("⚠️  No API data files found in data/raw/")
        return pd.DataFrame()
    
    all_matches = []
    
    for api_file in api_files:
        try:
            with open(api_file, "r") as f:
                data = json.load(f)
                matches = data.get("matches", [])
                all_matches.extend(matches)
            logger.info(f"✅ Loaded {len(matches)} matches from {api_file.name}")
        except Exception as e:
            logger.error(f"❌ Error loading {api_file.name}: {str(e)}")
            continue
    
    if not all_matches:
        logger.warning("⚠️  No matches found in API files")
        return pd.DataFrame()
    
    # Convert API format to standardized DataFrame
    rows = []
    for match in all_matches:
        row = {
            "date": match.get("utcDate", "").split("T")[0],  # Extract date only
            "homeTeam": match.get("homeTeam", {}).get("name", "Unknown"),
            "awayTeam": match.get("awayTeam", {}).get("name", "Unknown"),
            "homeGoals": match.get("score", {}).get("fullTime", {}).get("home", 0),
            "awayGoals": match.get("score", {}).get("fullTime", {}).get("away", 0),
            "tournament": match.get("competition", {}).get("name", "Unknown"),
            "city": "Unknown",  # API doesn't provide city
            "country": "Unknown",  # API doesn't provide country
            "neutral": False,  # API doesn't explicitly provide neutral
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    
    logger.info(f"✅ Converted API data: {len(df)} matches")
    return df
