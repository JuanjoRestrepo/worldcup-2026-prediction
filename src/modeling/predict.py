"""Prediction helpers for loading the exported model and scoring fixtures."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib

from src.config.settings import settings
from src.modeling.features import (
    build_match_feature_frame,
    load_feature_dataset_with_source,
)


@lru_cache(maxsize=2)
def _load_model_bundle_cached(artifact_path: str) -> dict[str, object]:
    return joblib.load(artifact_path)


def load_model_bundle(artifact_path: Path | None = None) -> dict[str, object]:
    """Load the exported model artifact bundle."""
    resolved_path = Path(artifact_path or settings.MODEL_ARTIFACT_PATH)
    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at '{resolved_path}'. Train the model first."
        )
    return _load_model_bundle_cached(str(resolved_path))


def predict_match_outcome(
    home_team: str,
    away_team: str,
    tournament: str | None = None,
    neutral: bool = False,
    artifact_path: Path | None = None,
    feature_data_path: Path | None = None,
    feature_source: str | None = None,
) -> dict[str, object]:
    """
    Predict the outcome of a fixture using the exported model artifact.

    Args:
        home_team: Home team name
        away_team: Away team name
        tournament: Optional tournament label for tournament flags
        neutral: Whether the fixture is on neutral ground
        artifact_path: Optional alternate model artifact path
        feature_data_path: Optional alternate gold dataset path for snapshots
        feature_source: Optional feature source override: auto, postgres, or csv

    Returns:
        Dictionary with the predicted class, probabilities, and snapshot metadata
    """
    bundle = load_model_bundle(artifact_path=artifact_path)
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]
    encoded_to_outcome = bundle["encoded_to_outcome"]
    outcome_labels = bundle["outcome_labels"]

    resolved_feature_source = feature_source or settings.PREDICTION_FEATURE_SOURCE
    feature_history, active_feature_source = load_feature_dataset_with_source(
        dataset_path=feature_data_path,
        source=resolved_feature_source,
    )
    feature_frame, snapshot_dates = build_match_feature_frame(
        home_team=home_team,
        away_team=away_team,
        tournament=tournament,
        neutral=neutral,
        feature_columns=feature_columns,
        feature_history_df=feature_history,
    )

    predicted_encoded = int(model.predict(feature_frame)[0])
    encoded_classes = [int(value) for value in model.named_steps["model"].classes_]
    probabilities = model.predict_proba(feature_frame)[0]

    class_probabilities = {}
    for encoded_class, probability in zip(encoded_classes, probabilities):
        outcome = int(encoded_to_outcome[encoded_class])
        class_probabilities[outcome_labels[outcome]] = float(probability)

    predicted_outcome = int(encoded_to_outcome[predicted_encoded])
    return {
        "home_team": snapshot_dates["home_team"],
        "away_team": snapshot_dates["away_team"],
        "predicted_class": predicted_outcome,
        "predicted_outcome": outcome_labels[predicted_outcome],
        "class_probabilities": class_probabilities,
        "neutral": bool(neutral),
        "tournament": tournament,
        "feature_snapshot_dates": {
            "home_team": snapshot_dates["home_snapshot_date"],
            "away_team": snapshot_dates["away_snapshot_date"],
        },
        "feature_source": active_feature_source,
        "model_artifact_path": str(Path(artifact_path or settings.MODEL_ARTIFACT_PATH)),
    }
