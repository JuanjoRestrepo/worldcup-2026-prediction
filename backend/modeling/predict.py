"""Prediction helpers for loading the exported model and scoring fixtures."""

from __future__ import annotations

import logging
import threading
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
from numpy.typing import NDArray
from scipy.stats import poisson

from backend.config.settings import settings
from backend.modeling.evaluation import extract_estimator_classes, predict_proba_aligned
from backend.modeling.features import (
    build_match_feature_frame,
    build_match_feature_frame_from_team_snapshots,
    load_feature_dataset_with_source,
)
from backend.modeling.inference_logger import InferenceLogger
from backend.modeling.segment_routing import detect_match_segment
from backend.modeling.serving_store import (
    load_latest_team_snapshots_from_dbt,
    load_team_snapshots_as_of_date_from_dbt,
)
from backend.modeling.types import ModelArtifactBundle, PredictionResult

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


def _calculate_most_probable_score(
    expected_home_goals: float,
    expected_away_goals: float,
    predicted_outcome_label: str,
    max_goals: int = 10,
) -> str:
    """
    Computes the most probable exact score using a bivariate Poisson distribution,
    strictly conditioned on the primary model's predicted outcome.
    """
    best_score = (0, 0)
    max_prob = -1.0

    for h in range(max_goals):
        for a in range(max_goals):
            # Calculate independent Poisson joint probability
            prob = float(
                poisson.pmf(h, expected_home_goals)
                * poisson.pmf(a, expected_away_goals)
            )

            # Filter mathematically impossible scores for the given outcome
            if predicted_outcome_label == "draw" and h != a:
                continue
            if predicted_outcome_label == "home_win" and h <= a:
                continue
            if predicted_outcome_label == "away_win" and a <= h:
                continue

            # Find the argmax of the valid subset
            if prob > max_prob:
                max_prob = prob
                best_score = (h, a)

    return f"{best_score[0]}-{best_score[1]}"


def _decode_probabilities(
    model: Any,
    feature_frame: Any,
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

    encoded_classes: list[int] = [int(v) for v in extract_estimator_classes(model)]
    probabilities: NDArray[np.float64] = predict_proba_aligned(
        model,
        feature_frame,
    )[0]

    class_probabilities: dict[str, float] = {}
    for encoded_class, probability in zip(encoded_classes, probabilities, strict=False):
        outcome = int(encoded_to_outcome[encoded_class])
        class_probabilities[outcome_labels[outcome]] = float(probability)

    predicted_encoded = int(model.predict(feature_frame)[0])
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
    log_inference: bool = True,
    is_knockout: bool = False,
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
        is_knockout: Set to True for elimination-round matches. Signals to the
            model that a draw is impossible (games decided by extra time /
            penalties), which structurally suppresses the draw class probability.

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
    home_goals_model = bundle.get("home_goals_model")
    away_goals_model = bundle.get("away_goals_model")

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
                    is_knockout=is_knockout,
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
                is_knockout=is_knockout,
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
            is_knockout=is_knockout,
        )

    # ── Primary model inference ───────────────────────────────────────────────
    class_probabilities, predicted_outcome_label = _decode_probabilities(
        model, feature_frame, encoded_to_outcome, outcome_labels
    )
    predicted_encoded_raw = int(model.predict(feature_frame)[0])
    predicted_outcome_int = int(encoded_to_outcome[predicted_encoded_raw])

    # ── Expected Goals & Dixon-Coles inference ───────────────────────────────
    expected_home_goals: float | None = None
    expected_away_goals: float | None = None
    predicted_score: str | None = None
    if home_goals_model is not None and away_goals_model is not None:
        try:
            expected_home_goals = float(home_goals_model.predict(feature_frame)[0])
            expected_away_goals = float(away_goals_model.predict(feature_frame)[0])

            # Reconcile predicted score using bivariate Poisson matrix
            predicted_score = _calculate_most_probable_score(
                expected_home_goals, expected_away_goals, predicted_outcome_label
            )

            # Compute Dixon-Coles score matrix probabilities
            from backend.modeling.dixon_coles import (  # noqa: PLC0415
                calculate_outcome_probs_from_score_matrix,
                calculate_score_matrix,
            )

            dc_matrix = calculate_score_matrix(
                expected_home_goals, expected_away_goals, rho=-0.05, max_goals=10
            )
            dc_away_p, dc_draw_p, dc_home_p = calculate_outcome_probs_from_score_matrix(
                dc_matrix, is_knockout=is_knockout
            )

            if neutral:
                # Perform double-inversion query for neutral ground balance
                try:
                    inv_frame, _ = build_match_feature_frame(
                        home_team=away_team,
                        away_team=home_team,
                        tournament=tournament,
                        neutral=neutral,
                        feature_columns=feature_columns,
                        feature_history_df=feature_history,
                        match_date=match_date,
                        is_knockout=is_knockout,
                    )
                    inv_probs, _ = _decode_probabilities(
                        model, inv_frame, encoded_to_outcome, outcome_labels
                    )
                    inv_hg = float(home_goals_model.predict(inv_frame)[0])
                    inv_ag = float(away_goals_model.predict(inv_frame)[0])

                    inv_dc_matrix = calculate_score_matrix(
                        inv_hg, inv_ag, rho=-0.05, max_goals=10
                    )
                    inv_away_p, inv_draw_p, inv_home_p = (
                        calculate_outcome_probs_from_score_matrix(
                            inv_dc_matrix, is_knockout=is_knockout
                        )
                    )

                    # Symmetrize Dixon-Coles probabilities
                    dc_home_p = (dc_home_p + inv_away_p) / 2.0
                    dc_away_p = (dc_away_p + inv_home_p) / 2.0
                    dc_draw_p = (dc_draw_p + inv_draw_p) / 2.0

                    # Symmetrize Tree model probabilities
                    class_probabilities["home_win"] = (
                        class_probabilities["home_win"] + inv_probs["away_win"]
                    ) / 2.0
                    class_probabilities["away_win"] = (
                        class_probabilities["away_win"] + inv_probs["home_win"]
                    ) / 2.0
                    class_probabilities["draw"] = (
                        class_probabilities["draw"] + inv_probs["draw"]
                    ) / 2.0
                except Exception as exc:
                    logger.warning("Neutral symmetrization fallback: %s", exc)

            # Blend Dixon-Coles goals probabilities (70%) with Base model (30%)
            w_dc = 0.70
            final_p_home = (
                w_dc * dc_home_p + (1.0 - w_dc) * class_probabilities["home_win"]
            )
            final_p_away = (
                w_dc * dc_away_p + (1.0 - w_dc) * class_probabilities["away_win"]
            )
            final_p_draw = w_dc * dc_draw_p + (1.0 - w_dc) * class_probabilities["draw"]

            tot_p = final_p_home + final_p_away + final_p_draw
            if tot_p > 0:
                final_p_home /= tot_p
                final_p_away /= tot_p
                final_p_draw /= tot_p

            class_probabilities = {
                "home_win": final_p_home,
                "draw": final_p_draw,
                "away_win": final_p_away,
            }

            if final_p_home >= final_p_draw and final_p_home >= final_p_away:
                predicted_outcome_label = "home_win"
                predicted_outcome_int = 1
            elif final_p_away >= final_p_draw and final_p_away >= final_p_home:
                predicted_outcome_label = "away_win"
                predicted_outcome_int = -1
            else:
                predicted_outcome_label = "draw"
                predicted_outcome_int = 0

        except Exception as exc:
            logger.warning("Expected goals inference failed: %s", exc)

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
                )
                spec_probs = predict_proba_aligned(
                    shadow_model.specialist_model_, feature_frame
                )
                shadow_is_override_triggered = bool(
                    shadow_model._compute_override_mask(
                        override_frame, gen_probs, spec_probs
                    )[0]
                )
    except Exception as exc:
        logger.warning("Shadow inference failed — continuing without shadow: %s", exc)

    # ── Inference logging ─────────────────────────────────────────────────────
    if log_inference:
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
        "expected_home_goals": expected_home_goals,
        "expected_away_goals": expected_away_goals,
        "predicted_score": predicted_score,
    }
