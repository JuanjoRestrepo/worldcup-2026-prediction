"""Prediction helpers for loading the exported model and scoring fixtures."""

from __future__ import annotations

from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import cast

import joblib

from src.config.settings import settings
from src.modeling.evaluation import extract_estimator_classes, predict_proba_aligned
from src.modeling.features import (
    build_match_feature_frame,
    build_match_feature_frame_from_team_snapshots,
    load_feature_dataset_with_source,
)
from src.modeling.inference_logger import InferenceLogger
from src.modeling.serving_store import (
    load_latest_team_snapshots_from_dbt,
    load_team_snapshots_as_of_date_from_dbt,
)
from src.modeling.types import ModelArtifactBundle, PredictionResult


@lru_cache(maxsize=2)
def _load_model_bundle_cached(artifact_path: str) -> ModelArtifactBundle:
    return cast(ModelArtifactBundle, joblib.load(artifact_path))


def load_model_bundle(artifact_path: Path | None = None) -> ModelArtifactBundle:
    """Load the exported model artifact bundle."""
    resolved_path = Path(artifact_path or settings.MODEL_ARTIFACT_PATH)
    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at '{resolved_path}'. Train the model first."
        )
    return _load_model_bundle_cached(str(resolved_path))


def _detect_match_segment(tournament: str | None) -> str | None:
    """
    Detect which segment a match belongs to based on tournament.

    Used for Segment-Aware Hybrid Ensemble routing.

    Args:
        tournament: Tournament name from match data

    Returns:
        Segment ID (e.g., "worldcup", "friendlies") or None if unmatched
    """
    if not tournament:
        return None

    tournament_lower = tournament.lower()

    if "world cup" in tournament_lower or "fifa world cup" in tournament_lower:
        return "worldcup"
    elif "friendly" in tournament_lower or "international friendly" in tournament_lower:
        return "friendlies"
    elif any(
        keyword in tournament_lower
        for keyword in [
            "copa america",
            "euro",
            "africa cup",
            "asian cup",
            "confederations",
        ]
    ):
        return "continental"

    # Other official tournaments
    if any(
        keyword in tournament_lower
        for keyword in ["qualifier", "qualification", "playoff", "repechage"]
    ):
        return "qualifiers"

    return None


def predict_match_outcome(
    home_team: str,
    away_team: str,
    tournament: str | None = None,
    neutral: bool = False,
    match_date: date | None = None,
    artifact_path: Path | None = None,
    feature_data_path: Path | None = None,
    feature_source: str | None = None,
) -> PredictionResult:
    """
    Predict the outcome of a fixture using the exported model artifact.

    Args:
        home_team: Home team name
        away_team: Away team name
        tournament: Optional tournament label for tournament flags
        neutral: Whether the fixture is on neutral ground
        match_date: Optional historical fixture date for as-of snapshot serving
        artifact_path: Optional alternate model artifact path
        feature_data_path: Optional alternate gold dataset path for snapshots
        feature_source: Optional feature source override: auto, dbt, postgres, or csv

    Returns:
        Dictionary with the predicted class, probabilities, and snapshot metadata
    """
    bundle = load_model_bundle(artifact_path=artifact_path)
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]
    encoded_to_outcome = bundle["encoded_to_outcome"]
    outcome_labels = bundle["outcome_labels"]

    resolved_feature_source = feature_source or settings.PREDICTION_FEATURE_SOURCE
    if resolved_feature_source in {"auto", "dbt"}:
        try:
            if match_date is None:
                team_snapshots = load_latest_team_snapshots_from_dbt()
                active_feature_source = "dbt_latest_team_snapshots"
            else:
                team_snapshots, active_feature_source = (
                    load_team_snapshots_as_of_date_from_dbt(match_date)
                )
            feature_frame, snapshot_dates = (
                build_match_feature_frame_from_team_snapshots(
                    home_team=home_team,
                    away_team=away_team,
                    tournament=tournament,
                    neutral=neutral,
                    feature_columns=feature_columns,
                    team_snapshots_df=team_snapshots,
                )
            )
        except RuntimeError as exc:
            if resolved_feature_source == "dbt":
                raise RuntimeError(
                    "Failed to build serving features from dbt team snapshots."
                ) from exc

            feature_history, active_feature_source = load_feature_dataset_with_source(
                dataset_path=feature_data_path,
                source="auto",
            )
            feature_frame, snapshot_dates = build_match_feature_frame(
                home_team=home_team,
                away_team=away_team,
                tournament=tournament,
                neutral=neutral,
                feature_columns=feature_columns,
                feature_history_df=feature_history,
                match_date=match_date,
            )
    else:
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
            match_date=match_date,
        )

    predicted_encoded = int(model.predict(feature_frame)[0])
    encoded_classes = [int(value) for value in extract_estimator_classes(model)]
    probabilities = predict_proba_aligned(model, feature_frame)[0]

    class_probabilities = {}
    for encoded_class, probability in zip(encoded_classes, probabilities):
        outcome = int(encoded_to_outcome[encoded_class])
        class_probabilities[outcome_labels[outcome]] = float(probability)

    predicted_outcome = int(encoded_to_outcome[predicted_encoded])

    # ============================================================================
    # Segment-Aware Telemetry: Detect tournament segment for monitoring
    # ============================================================================
    # Note: Ensemble-based override logic is reserved for future phases when
    # separate generalist/specialist estimators are available in the artifact.
    # For now, we capture segment for observability and logging.

    match_segment = _detect_match_segment(tournament)
    is_override_triggered: bool = (
        False  # Default: no specialist override in current version
    )

    # ============================================================================
    # Log prediction with segment-aware telemetry
    # ============================================================================
    try:
        logger_instance = InferenceLogger()
        logger_instance.log_prediction(
            home_team=snapshot_dates["home_team"],
            away_team=snapshot_dates["away_team"],
            predicted_class=predicted_outcome,
            predicted_outcome=outcome_labels[predicted_outcome],
            class_probabilities=class_probabilities,
            neutral=bool(neutral),
            tournament=tournament,
            feature_snapshot_dates={
                "home_team": snapshot_dates["home_snapshot_date"],
                "away_team": snapshot_dates["away_snapshot_date"],
            },
            feature_source=active_feature_source,
            model_artifact_path=str(
                Path(artifact_path or settings.MODEL_ARTIFACT_PATH)
            ),
            requested_match_date=match_date,
            match_segment=match_segment,
            is_override_triggered=is_override_triggered,
        )
    except Exception as e:
        import logging

        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to log prediction to inference table: {e}")

    # ============================================================================
    # Return enriched prediction result with segment-aware telemetry
    # ============================================================================
    return {
        "home_team": snapshot_dates["home_team"],
        "away_team": snapshot_dates["away_team"],
        "predicted_class": predicted_outcome,
        "predicted_outcome": outcome_labels[predicted_outcome],
        "class_probabilities": class_probabilities,
        "neutral": bool(neutral),
        "tournament": tournament,
        "match_date": match_date.isoformat() if match_date is not None else None,
        "feature_snapshot_dates": {
            "home_team": snapshot_dates["home_snapshot_date"],
            "away_team": snapshot_dates["away_snapshot_date"],
        },
        "feature_source": active_feature_source,
        "model_artifact_path": str(Path(artifact_path or settings.MODEL_ARTIFACT_PATH)),
        "match_segment": match_segment,
        "is_override_triggered": is_override_triggered,
    }
