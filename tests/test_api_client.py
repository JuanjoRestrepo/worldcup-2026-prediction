import os

import pytest

from src.ingestion.clients.api_client import FootballAPIClient


def test_api_connection():
    api_key = os.getenv("FOOTBALL_API_KEY")

    if not api_key:
        pytest.skip("FOOTBALL_API_KEY is not configured.")

    client = FootballAPIClient(api_key=api_key)

    try:
        data = client.get_matches()
    except Exception as exc:
        pytest.skip(f"External football API is unreachable in this environment: {exc}")

    assert "matches" in data
