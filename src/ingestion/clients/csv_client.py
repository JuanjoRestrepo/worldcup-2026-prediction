import pandas as pd
import logging

from src.config.settings import settings

logger = logging.getLogger(__name__)

RAW_PATH = settings.RAW_DIR / "international_results.csv"


def load_historical_data() -> pd.DataFrame:
    """
    Load historical football matches dataset.

    Returns:
        pd.DataFrame: Raw historical matches
    """
    try:
        df = pd.read_csv(RAW_PATH)
        logger.info(f"Loaded historical dataset: {df.shape}")
        return df

    except Exception as e:
        logger.error(f"Error loading CSV: {e}")
        raise
