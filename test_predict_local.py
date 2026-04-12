#!/usr/bin/env python
"""Quick test of prediction logic locally."""

from src.modeling.predict import predict_match_outcome

try:
    print("🔍 Testing prediction locally...")
    result = predict_match_outcome(
        home_team="Argentina",
        away_team="Brazil",
        tournament="World Cup",
        neutral=False,
        match_date=None,
    )
    print("✅ SUCCESS: Prediction works locally!")
    print(f"   Outcome: {result['predicted_outcome']}")
    print(f"   Probabilities: {result['class_probabilities']}")
except Exception as e:
    print(f"❌ ERROR: {type(e).__name__}: {e}")
    import traceback

    traceback.print_exc()
