"""Prediction helpers for loading the exported model and scoring fixtures."""

from __future__ import annotations

import logging
import threading
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import cast

import joblib
import numpy as np
from numpy.typing import NDArray

from src.config.settings import settings
from src.modeling.evaluation import extract_estimator_classes, predict_proba_aligned
from src.modeling.features import (
    build_match_feature_frame,
    build_match_feature_frame_from_team_snapshots,
    load_feature_dataset_with_source,
)
from src.modeling.inference_logger import InferenceLogger
from src.modeling.segment_routing import detect_match_segment
from src.modeling.serving_store import (
    load_latest_team_snapshots_from_dbt,
    load_team_snapshots_as_of_date_from_dbt,
)
from src.modeling.types import ModelArtifactBundle, PredictionResult

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────────
# Model loading — cached, path-keyed
# ────────────────────────────────────────────────────────────────────────────


@lru_cache(maxsize=4)
def _load_model_bundle_cached(artifact_path: str) -> ModelArtifactBundle:
    return cast(ModelArtifactBundle, joblib.load(artifact_path))


def load_model_bundle(artifact_path: Path | None = None) -> ModelArtifactBundle:
    """Load the exported model artifact bundle from a cached or resolved path."""
    resolved_path = Path(artifact_path or settings.MODEL_ARTIFACT_PATH)
    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at '{resolved_path}'. Train the model first."
        )
    return _load_model_bundle_cached(str(resolved_path))


# ────────────────────────────────────────────────────────────────────────────
# Shadow deployment toggle — thread-safe via RLock
# ────────────────────────────────────────────────────────────────────────────

_shadow_lock = threading.RLock()
_USE_SHADOW_AS_PRIMARY: bool = False


def toggle_shadow_mode(enable: bool) -> None:
    """
    Thread-safely toggle whether the shadow artifact should be used as primary.

    Invalidates the model bundle cache to ensure the new primary is loaded fresh.
    """
    global _USE_SHADOW_AS_PRIMARY  # noqa: PLW0603
    with _shadow_lock:
        _USE_SHADOW_AS_PRIMARY = enable
        _load_model_bundle_cached.cache_clear()
        logger.info("Shadow mode toggled — shadow_as_primary=%s, cache cleared", enable)


def _is_shadow_primary() -> bool:
    """Read the shadow flag in a thread-safe manner."""
    with _shadow_lock:
        return _USE_SHADOW_AS_PRIMARY


# ────────────────────────────────────────────────────────────────────────────
# Probability decoding — shared by primary and shadow to avoid duplication
# ────────────────────────────────────────────────────────────────────────────


def _decode_probabilities(
    model: object,
    feature_frame: object,
    encoded_to_outcome: dict[int, int],
    outcome_labels: dict[int, str],
) -> tuple[dict[str, float], str]:
    """
    Decode raw model probabilities into a labeled outcome dict.

    Args:
        model: Fitted probabilistic estimator (sklearn-compatible).
        feature_frame: pd.DataFrame with model-ready features.
        encoded_to_outcome: Mapping from encoded int → original outcome int.
        outcome_labels: Mapping from original outcome int → label string.

    Returns:
        Tuple of (class_probabilities dict, predicted_outcome label string).
    """
    from pandas import DataFrame  # noqa: PLC0415

    assert isinstance(feature_frame, DataFrame)  # noqa: S101

    encoded_classes: list[int] = [
        int(v)
        for v in extract_estimator_classes(cast(object, model))  # type: ignore[arg-type]
    ]
    probabilities: NDArray[np.float64] = predict_proba_aligned(
        cast(object, model),
        feature_frame,  # type: ignore[arg-type]
    )[0]

    class_probabilities: dict[str, float] = {}
    for encoded_class, probability in zip(encoded_classes, probabilities, strict=False):
        outcome = int(encoded_to_outcome[encoded_class])
        class_probabilities[outcome_labels[outcome]] = float(probability)

    predicted_encoded = int(cast(object, model).predict(feature_frame)[0])  # type: ignore[union-attr]
    predicted_outcome_label = outcome_labels[int(encoded_to_outcome[predicted_encoded])]

    return class_probabilities, predicted_outcome_label


# ────────────────────────────────────────────────────────────────────────────
# Main prediction entry point
# ────────────────────────────────────────────────────────────────────────────


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
        home_team: Home team name (already normalized via alias map).
        away_team: Away team name (already normalized via alias map).
        tournament: Optional tournament label for tournament flags.
        neutral: Whether the fixture is on neutral ground.
        match_date: Optional historical fixture date for as-of snapshot serving.
        artifact_path: Optional alternate model artifact path.
        feature_data_path: Optional alternate gold dataset path for snapshots.
        feature_source: Optional feature source override: auto, dbt, postgres, or csv.

    Returns:
        PredictionResult dict with prediction, probabilities, and snapshot metadata.
    """
    default_artifact = Path(artifact_path or settings.MODEL_ARTIFACT_PATH)
    shadow_artifact = default_artifact.with_name(
        f"{default_artifact.stem}_shadow.joblib"
    )

    if _is_shadow_primary() and shadow_artifact.exists():
        primary_path = shadow_artifact
        alt_path = default_artifact
    else:
        primary_path = default_artifact
        alt_path = shadow_artifact

    bundle = load_model_bundle(artifact_path=primary_path)
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]
    encoded_to_outcome = bundle["encoded_to_outcome"]
    outcome_labels = bundle["outcome_labels"]

    # ── Feature loading ──────────────────────────────────────────────────────
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
        except Exception as exc:
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

    # ── Primary model inference ───────────────────────────────────────────────
    class_probabilities, predicted_outcome_label = _decode_probabilities(
        model, feature_frame, encoded_to_outcome, outcome_labels
    )
    predicted_encoded_raw = int(model.predict(feature_frame)[0])  # type: ignore[union-attr]
    predicted_outcome_int = int(encoded_to_outcome[predicted_encoded_raw])

    # ── Segment-aware telemetry ───────────────────────────────────────────────
    match_segment = detect_match_segment(tournament)
    is_override_triggered: bool = (
        False  # specialist override reserved for future phases
    )

    # ── Shadow deployment inference ───────────────────────────────────────────
    shadow_predicted_outcome: str | None = None
    shadow_class_probabilities: dict[str, float] | None = None
    shadow_model_name: str | None = None
    shadow_is_override_triggered: bool = False

    try:
        if alt_path.exists():
            shadow_bundle = load_model_bundle(artifact_path=alt_path)
            shadow_model = shadow_bundle["model"]

            shadow_probs, shadow_outcome_label = _decode_probabilities(
                shadow_model, feature_frame, encoded_to_outcome, outcome_labels
            )
            shadow_predicted_outcome = shadow_outcome_label
            shadow_class_probabilities = shadow_probs
            shadow_model_name = shadow_bundle["selected_model_name"]

            if hasattr(shadow_model, "_compute_override_mask"):
                override_frame = feature_frame.copy()
                override_frame["tournament"] = tournament
                gen_probs = predict_proba_aligned(
                    shadow_model.generalist_model_, feature_frame
                )  # type: ignore[union-attr]
                spec_probs = predict_proba_aligned(
                    shadow_model.specialist_model_, feature_frame
                )  # type: ignore[union-attr]
                shadow_is_override_triggered = bool(
                    shadow_model._compute_override_mask(
                        override_frame, gen_probs, spec_probs
                    )[0]  # type: ignore[union-attr]
                )
    except Exception as exc:
        logger.warning("Shadow inference failed — continuing without shadow: %s", exc)

    # ── Inference logging ─────────────────────────────────────────────────────
    try:
        inference_logger = InferenceLogger()
        inference_logger.log_prediction(
            home_team=snapshot_dates["home_team"],
            away_team=snapshot_dates["away_team"],
            predicted_class=predicted_outcome_int,
            predicted_outcome=predicted_outcome_label,
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
            shadow_predicted_outcome=shadow_predicted_outcome,
            shadow_class_probabilities=shadow_class_probabilities,
            shadow_model_name=shadow_model_name,
            shadow_is_override_triggered=shadow_is_override_triggered,
        )
    except Exception as exc:
        logger.warning("Failed to log prediction to inference table: %s", exc)

    # ── Return enriched prediction result ─────────────────────────────────────
    return {
        "home_team": snapshot_dates["home_team"],
        "away_team": snapshot_dates["away_team"],
        "predicted_class": predicted_outcome_int,
        "predicted_outcome": predicted_outcome_label,
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
        "shadow_predicted_outcome": shadow_predicted_outcome,
        "shadow_class_probabilities": shadow_class_probabilities,
        "shadow_is_override_triggered": shadow_is_override_triggered,
        "shadow_model_name": shadow_model_name,
    }
