import json
import socket

import pytest
import requests

BASE_URL = "http://localhost:8000"


def is_server_running(host="localhost", port=8000):
    """Check if the API server is running."""
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except (TimeoutError, ConnectionRefusedError, OSError):
        return False


@pytest.mark.skipif(
    not is_server_running(), reason="API server not running on localhost:8000"
)
def test_api_health_check():
    """Test health check endpoint."""
    print("🏥 HEALTH CHECK:")
    response = requests.get(f"{BASE_URL}/config")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")
    assert response.status_code == 200


@pytest.mark.skipif(
    not is_server_running(), reason="API server not running on localhost:8000"
)
def test_api_prediction():
    """Test prediction endpoint."""
    print("🎯 PREDICTION TEST:")
    fixture = {
        "home_team": "Argentina",
        "away_team": "Brazil",
        "tournament": "World Cup Qualifiers",
    }
    response = requests.post(
        f"{BASE_URL}/predict",
        json=fixture,
        headers={"Content-Type": "application/json"},
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    assert response.status_code == 200

    # Check what was returned
    if response.status_code == 200:
        data = response.json()
        print("\n✅ Prediction Fields Returned:")
        for key in data.keys():
            print(f"  - {key}: {data[key]}")

        # Verify segment awareness
        assert "match_segment" in data, "match_segment field missing"
        assert "is_override_triggered" in data, "is_override_triggered field missing"
        print("\n✅ Segment-aware fields verified:")
        print(f"  - match_segment: {data['match_segment']}")
        print(f"  - is_override_triggered: {data['is_override_triggered']}")
