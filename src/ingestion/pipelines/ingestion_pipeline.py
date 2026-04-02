import logging
from datetime import datetime, timedelta
from pathlib import Path
import json
import os

from dotenv import load_dotenv

from src.ingestion.clients.api_client import FootballAPIClient
from src.ingestion.clients.csv_client import load_historical_data
from src.ingestion.utils.validator import validate_schema

logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")

# Load environment variables from .env file
load_dotenv()


def run_ingestion_pipeline():
    logger.info("Starting ingestion pipeline")

    # 1. CSV (histórico)
    df = load_historical_data()
    validate_schema(df)
    logger.info(f"Loaded {len(df)} historical records from CSV")

    # 2. API - Only fetch recent data to save free tier requests
    api_key = os.getenv("FOOTBALL_API_KEY")
    
    # Get the last 30 days of matches (free tier limit is low, so fetch minimal data)
    today = datetime.now()
    thirty_days_ago = (today - timedelta(days=30)).strftime("%Y-%m-%d")
    today_str = today.strftime("%Y-%m-%d")
    
    api_data = None
    if api_key:
        try:
            logger.info(f"Fetching recent matches from API (last 30 days: {thirty_days_ago} to {today_str})")
            client = FootballAPIClient(api_key=api_key)
            api_data = client.get_matches(
                date_from=thirty_days_ago,
                date_to=today_str,
            )
            logger.info(f"Successfully fetched {len(api_data.get('matches', []))} recent matches from API")
        except Exception as e:
            logger.warning(f"Could not fetch data from API (free tier has request limits): {e}")
            logger.info("Continuing with historical data only")
            api_data = None
    else:
        logger.warning("FOOTBALL_API_KEY not set. Using historical data only")

    # 3. Save raw API data if available
    if api_data:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = RAW_DIR / f"api_matches_{timestamp}.json"
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(api_data, f)

        logger.info(f"Saved API data → {output_path}")
    
    logger.info("Pipeline completed")