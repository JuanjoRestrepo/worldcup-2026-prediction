import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

RAW_PATH = Path("data/raw/international_results.csv")


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