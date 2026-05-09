import joblib

try:
    model = joblib.load("models/match_predictor.joblib")
    print(f"Active model name: {model.get('selected_model_name', 'Unknown')}")
    print(f"Algorithm: {type(model.get('model'))}")
except Exception as e:
    print(f"Error loading model: {e}")
