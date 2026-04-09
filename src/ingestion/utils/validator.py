import pandas as pd

from src.contracts.data_contracts import validate_historical_raw_contract


def validate_schema(df: pd.DataFrame) -> None:
    """
    Validate dataset schema.

    Raises:
        ValueError if schema is invalid
    """
    validate_historical_raw_contract(df)
