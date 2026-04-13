"""Quick smoke-test for the /predict endpoint."""
import os

os.environ["PREDICTION_FEATURE_SOURCE"] = "csv"
os.environ["MONITORING_SOURCE"] = "auto"
os.environ["ADMIN_API_KEY"] = "test_key"

from fastapi.testclient import TestClient  # noqa: E402
from src.api.main import app  # noqa: E402

client = TestClient(app)

health = client.get("/health")
print("Health:", health.status_code, health.json())

r = client.post(
    "/predict",
    json={
        "home_team": "Argentina",
        "away_team": "France",
        "tournament": "FIFA World Cup",
        "neutral": True,
    },
)
print("Predict status:", r.status_code)
if r.status_code == 200:
    data = r.json()
    probs = {k: round(v * 100, 1) for k, v in data["class_probabilities"].items()}
    print("Outcome:", data["predicted_outcome"])
    print("Probs:", probs)
else:
    print("Error:", r.text[:500])
