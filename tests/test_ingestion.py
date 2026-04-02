from src.ingestion.clients.csv_client import load_historical_data


def test_load_csv():
    df = load_historical_data()

    assert df is not None
    assert len(df) > 1000