import json

import requests

BASE_URL = "http://localhost:8000"

# Test health check
print("🏥 HEALTH CHECK:")
response = requests.get(f"{BASE_URL}/config")
print(f"Status: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}\n")

# Test prediction
print("🎯 PREDICTION TEST:")
fixture = {
    "home_team": "Argentina",
    "away_team": "Brazil",
    "tournament": "World Cup Qualifiers",
}
response = requests.post(
    f"{BASE_URL}/predict", json=fixture, headers={"Content-Type": "application/json"}
)
print(f"Status: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}")

# Check what was returned
if response.status_code == 200:
    data = response.json()
    print("\n✅ Prediction Fields Returned:")
    for key in data.keys():
        print(f"  - {key}: {data[key]}")
