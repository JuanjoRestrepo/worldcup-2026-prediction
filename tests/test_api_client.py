import os
from src.ingestion.clients.api_client import FootballAPIClient


def test_api_connection():
    api_key = os.getenv("FOOTBALL_API_KEY")

    if not api_key:
        return  # skip if not configured

    client = FootballAPIClient(api_key=api_key)

    data = client.get_matches()

    assert "matches" in data