"""Train and export the production match outcome model."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import cast

import joblib
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from src.config.settings import settings
from src.contracts.data_contracts import validate_feature_dataset_contract
from src.database.persistence import persist_training_run
from src.modeling.evaluation import (
    CandidateSpec,
    ProbabilisticEstimator,
    evaluate_candidates_with_backtesting,
    evaluate_multiclass_predictions,
    fit_estimator_with_sample_weight,
    predict_proba_aligned,
    select_deployment_variant,
    split_train_calibration_test,
)
from src.modeling.features import OUTCOME_LABELS, TARGET_COLUMN, load_feature_dataset
from src.modeling.features import select_model_feature_columns
from src.modeling.hybrid_ensemble import HybridDrawOverrideEnsemble
from src.modeling.hybrid_ensemble_segment_aware import (
    SegmentAwareHybridDrawOverrideEnsemble,
    SegmentConfig,
)
from src.modeling.reporting import generate_evaluation_report
from src.modeling.segment_routing import (
    SEGMENT_METADATA_COLUMNS,
    tournament_segment_detector,
)
from src.modeling.two_stage import TwoStageDrawClassifier
from src.modeling.types import DateRange, ModelArtifactBundle, TrainingMetrics, TrainingSummary

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.2
DEFAULT_CALIBRATION_SIZE = 0.1
DEFAULT_CALIBRATION_SELECTION_SIZE = 0.5
DEFAULT_BACKTEST_SPLITS = 5
OUTCOME_TO_ENCODED = {-1: 0, 0: 1, 1: 2}
ENCODED_TO_OUTCOME = {value: key for key, value in OUTCOME_TO_ENCODED.items()}
SPECIALIST_DRAW_WEIGHT_MULTIPLIER = 4 / 3


def _build_segment_configs(
    *,
    friendlies_unc: float,
    friendlies_conv: float,
    worldcup_unc: float,
    worldcup_conv: float,
    continental_unc: float,
    continental_conv: float,
    qualifiers_unc: float,
    qualifiers_conv: float,
) -> dict[str, SegmentConfig]:
    """Build segment configuration dict with per-segment thresholds."""
    return {
        "friendlies": SegmentConfig(
            segment_id="friendlies",
            uncertainty_threshold=friendlies_unc,
            draw_conviction_threshold=friendlies_conv,
            description="Friendly matches — generalist often uncertain, specialist can help",
        ),
        "worldcup": SegmentConfig(
            segment_id="worldcup",
            uncertainty_threshold=worldcup_unc,
            draw_conviction_threshold=worldcup_conv,
            description="World Cup matches — generalist strong, minimal specialist use",
        ),
        "continental": SegmentConfig(
            segment_id="continental",
            uncertainty_threshold=continental_unc,
            draw_conviction_threshold=continental_conv,
            description="Continental tournaments — medium selectivity",
        ),
        "qualifiers": SegmentConfig(
            segment_id="qualifiers",
            uncertainty_threshold=qualifiers_unc,
            draw_conviction_threshold=qualifiers_conv,
            description="Qualification fixtures — moderate structure",
        ),
    }


def _build_pipeline(model: object) -> Pipeline:
    """Build a reusable training pipeline around a candidate estimator."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", model),
        ]
    )


def _build_scaled_pipeline(model: object) -> Pipeline:
    """Build a pipeline with scaling for linear models."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def _make_sample_weight_builder(
    draw_boost: float = 1.0,
) -> Callable[[pd.Series], NDArray[np.float64]]:
    def _builder(y_encoded: pd.Series) -> NDArray[np.float64]:
        sample_weight = np.asarray(
            compute_sample_weight(class_weight="balanced", y=y_encoded),
            dtype=np.float64,
        )
        if draw_boost != 1.0:
            draw_mask = y_encoded.to_numpy(dtype=np.int64, copy=False) == OUTCOME_TO_ENCODED[0]
            sample_weight[draw_mask] *= draw_boost
            sample_weight *= len(sample_weight) / sample_weight.sum()
        return sample_weight

    return _builder


def _build_candidate_specs(
    tuned_segment_configs: dict[str, SegmentConfig] | None = None,
) -> dict[str, CandidateSpec]:
    """Create a compact temporal hyperparameter search space."""
    candidate_specs: dict[str, CandidateSpec] = {
        "dummy_prior": CandidateSpec(
            name="dummy_prior",
            pipeline=_build_pipeline(DummyClassifier(strategy="prior")),
            sample_weight_builder=_make_sample_weight_builder(),
            family="dummy",
            hyperparameters={"strategy": "prior"},
            notes="Sanity baseline",
        )
    }

    logistic_variants: list[dict[str, float]] = [
        {"c": 0.5, "draw_boost": 1.0},
        {"c": 1.0, "draw_boost": 1.0},
        {"c": 2.0, "draw_boost": 1.0},
        {"c": 1.0, "draw_boost": 1.25},
        {"c": 2.0, "draw_boost": 1.2},
    ]
    for logistic_variant in logistic_variants:
        c_value = logistic_variant["c"]
        draw_boost = logistic_variant["draw_boost"]
        name = f"logistic_c{c_value:g}_draw{draw_boost:g}"
        candidate_specs[name] = CandidateSpec(
            name=name,
            pipeline=_build_scaled_pipeline(
                LogisticRegression(
                    C=c_value,
                    max_iter=2500,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                )
            ),
            sample_weight_builder=_make_sample_weight_builder(draw_boost),
            family="logistic_regression",
            hyperparameters={"C": c_value, "draw_boost": draw_boost},
            notes="Scaled linear baseline with optional draw emphasis",
        )

    random_forest_variants: list[dict[str, int | float | None]] = [
        {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 2, "draw_boost": 1.0},
        {"n_estimators": 400, "max_depth": 12, "min_samples_leaf": 2, "draw_boost": 1.0},
        {"n_estimators": 300, "max_depth": 12, "min_samples_leaf": 4, "draw_boost": 1.15},
    ]
    for forest_variant in random_forest_variants:
        n_estimators = cast(int, forest_variant["n_estimators"])
        max_depth = cast(int | None, forest_variant["max_depth"])
        min_samples_leaf = cast(int, forest_variant["min_samples_leaf"])
        draw_boost = cast(float, forest_variant["draw_boost"])
        name = (
            f"random_forest_n{n_estimators}_d"
            f"{'none' if max_depth is None else max_depth}_leaf{min_samples_leaf}_draw{draw_boost:g}"
        )
        candidate_specs[name] = CandidateSpec(
            name=name,
            pipeline=_build_pipeline(
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    class_weight="balanced_subsample",
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                )
            ),
            sample_weight_builder=_make_sample_weight_builder(draw_boost),
            family="random_forest",
            hyperparameters={
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_leaf": min_samples_leaf,
                "draw_boost": draw_boost,
            },
            notes="Tree ensemble with draw-aware weighting variants",
        )

    xgboost_variants: list[dict[str, int | float]] = [
        {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05, "reg_lambda": 1.0, "draw_boost": 1.0},
        {"n_estimators": 350, "max_depth": 4, "learning_rate": 0.05, "reg_lambda": 1.0, "draw_boost": 1.0},
        {"n_estimators": 400, "max_depth": 4, "learning_rate": 0.03, "reg_lambda": 2.0, "draw_boost": 1.0},
        {"n_estimators": 350, "max_depth": 5, "learning_rate": 0.05, "reg_lambda": 1.5, "draw_boost": 1.15},
    ]
    for xgb_variant in xgboost_variants:
        n_estimators = cast(int, xgb_variant["n_estimators"])
        max_depth = cast(int, xgb_variant["max_depth"])
        learning_rate = xgb_variant["learning_rate"]
        reg_lambda = xgb_variant["reg_lambda"]
        draw_boost = xgb_variant["draw_boost"]
        name = (
            f"xgboost_n{n_estimators}_d{max_depth}_lr{learning_rate}_"
            f"lambda{reg_lambda}_draw{draw_boost:g}"
        )
        candidate_specs[name] = CandidateSpec(
            name=name,
            pipeline=_build_pipeline(
                XGBClassifier(
                    objective="multi:softprob",
                    num_class=3,
                    eval_metric="mlogloss",
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    min_child_weight=1,
                    reg_lambda=reg_lambda,
                    random_state=RANDOM_STATE,
                    n_jobs=0,
                )
            ),
            sample_weight_builder=_make_sample_weight_builder(draw_boost),
            family="xgboost",
            hyperparameters={
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "learning_rate": learning_rate,
                "reg_lambda": reg_lambda,
                "draw_boost": draw_boost,
            },
            notes="Gradient boosting search with lightweight draw emphasis",
        )

    two_stage_variants: list[dict[str, float]] = [
        {
            "stage1_c": 2.0,
            "stage2_c": 1.0,
            "draw_boost": 1.6,
            "draw_probability_scale": 1.0,
        },
        {
            "stage1_c": 2.0,
            "stage2_c": 1.0,
            "draw_boost": 1.8,
            "draw_probability_scale": 1.08,
        },
        {
            "stage1_c": 1.0,
            "stage2_c": 2.0,
            "draw_boost": 2.0,
            "draw_probability_scale": 1.1,
        },
    ]
    for two_stage_variant in two_stage_variants:
        stage1_c = two_stage_variant["stage1_c"]
        stage2_c = two_stage_variant["stage2_c"]
        draw_boost = two_stage_variant["draw_boost"]
        draw_probability_scale = two_stage_variant["draw_probability_scale"]
        name = (
            f"two_stage_s1c{stage1_c:g}_s2c{stage2_c:g}_"
            f"draw{draw_boost:g}_scale{draw_probability_scale:g}"
        )
        candidate_specs[name] = CandidateSpec(
            name=name,
            pipeline=cast(
                ProbabilisticEstimator,
                TwoStageDrawClassifier(
                    stage1_estimator=_build_scaled_pipeline(
                        LogisticRegression(
                            C=stage1_c,
                            max_iter=2500,
                            class_weight="balanced",
                            random_state=RANDOM_STATE,
                        )
                    ),
                    stage2_estimator=_build_scaled_pipeline(
                        LogisticRegression(
                            C=stage2_c,
                            max_iter=2500,
                            class_weight="balanced",
                            random_state=RANDOM_STATE,
                        )
                    ),
                    draw_probability_scale=draw_probability_scale,
                ),
            ),
            sample_weight_builder=_make_sample_weight_builder(draw_boost),
            family="two_stage_draw_classifier",
            hyperparameters={
                "stage1_c": stage1_c,
                "stage2_c": stage2_c,
                "draw_boost": draw_boost,
                "draw_probability_scale": draw_probability_scale,
            },
            notes="Two-stage draw-vs-non-draw decomposition with aggressive draw weighting",
        )

    hybrid_override_variants: list[dict[str, float]] = [
        {"uncertainty_threshold": 0.42, "draw_conviction_threshold": 0.50},
        {"uncertainty_threshold": 0.42, "draw_conviction_threshold": 0.60},
        {"uncertainty_threshold": 0.45, "draw_conviction_threshold": 0.50},
        {"uncertainty_threshold": 0.45, "draw_conviction_threshold": 0.60},
        {"uncertainty_threshold": 0.48, "draw_conviction_threshold": 0.50},
        {"uncertainty_threshold": 0.48, "draw_conviction_threshold": 0.60},
    ]
    for hybrid_variant in hybrid_override_variants:
        uncertainty_threshold = hybrid_variant["uncertainty_threshold"]
        draw_conviction_threshold = hybrid_variant["draw_conviction_threshold"]
        name = (
            f"hybrid_override_u{uncertainty_threshold:g}_"
            f"d{draw_conviction_threshold:g}"
        )
        candidate_specs[name] = CandidateSpec(
            name=name,
            pipeline=cast(
                ProbabilisticEstimator,
                HybridDrawOverrideEnsemble(
                    generalist_estimator=_build_scaled_pipeline(
                        LogisticRegression(
                            C=2.0,
                            max_iter=2500,
                            class_weight="balanced",
                            random_state=RANDOM_STATE,
                        )
                    ),
                    specialist_estimator=cast(
                        ProbabilisticEstimator,
                        TwoStageDrawClassifier(
                            stage1_estimator=_build_scaled_pipeline(
                                LogisticRegression(
                                    C=2.0,
                                    max_iter=2500,
                                    class_weight="balanced",
                                    random_state=RANDOM_STATE,
                                )
                            ),
                            stage2_estimator=_build_scaled_pipeline(
                                LogisticRegression(
                                    C=1.0,
                                    max_iter=2500,
                                    class_weight="balanced",
                                    random_state=RANDOM_STATE,
                                )
                            ),
                            draw_probability_scale=1.0,
                        ),
                    ),
                    uncertainty_threshold=uncertainty_threshold,
                    draw_conviction_threshold=draw_conviction_threshold,
                    specialist_draw_weight_multiplier=SPECIALIST_DRAW_WEIGHT_MULTIPLIER,
                ),
            ),
            sample_weight_builder=_make_sample_weight_builder(1.2),
            family="hybrid_draw_override_ensemble",
            hyperparameters={
                "generalist_c": 2.0,
                "generalist_draw_boost": 1.2,
                "specialist_stage1_c": 2.0,
                "specialist_stage2_c": 1.0,
                "specialist_draw_boost": 1.6,
                "specialist_draw_weight_multiplier": SPECIALIST_DRAW_WEIGHT_MULTIPLIER,
                "specialist_draw_probability_scale": 1.0,
                "uncertainty_threshold": uncertainty_threshold,
                "draw_conviction_threshold": draw_conviction_threshold,
            },
            notes=(
                "Delegated ensemble: generalist owns confident fixtures; "
                "two-stage specialist only overrides uncertain rows into draw"
            ),
        )

    # ════════════════════════════════════════════════════════════════════════
    # Segment-Aware Hybrid Ensemble variants
    # ════════════════════════════════════════════════════════════════════════
    # These compete head-to-head against the global hybrid and the standalone
    # generalist champion under the same multicriteria ranking.
    #
    # Key insight from the 0.45 global threshold analysis:
    #   - 0.45 was too broad → the specialist fired on strong predictions too
    #   - Solution: narrower, segment-specific bands that only activate the
    #     specialist in segments where the generalist actually struggles
    #   - Friendlies → aggressive specialist (low threshold)
    #   - World Cup → very conservative (high threshold, almost no overrides)
    #   - Qualifiers / Continental → middle ground, leaning conservative
    # ────────────────────────────────────────────────────────────────────────

    if tuned_segment_configs is not None:
        name = "seg_hybrid_auto_tuned"
        candidate_specs[name] = CandidateSpec(
            name=name,
            pipeline=cast(
                ProbabilisticEstimator,
                SegmentAwareHybridDrawOverrideEnsemble(
                    generalist_estimator=_build_scaled_pipeline(
                        LogisticRegression(
                            C=2.0, max_iter=2500, class_weight="balanced", random_state=RANDOM_STATE
                        )
                    ),
                    specialist_estimator=cast(
                        ProbabilisticEstimator,
                        TwoStageDrawClassifier(
                            stage1_estimator=_build_scaled_pipeline(
                                LogisticRegression(
                                    C=2.0, max_iter=2500, class_weight="balanced", random_state=RANDOM_STATE
                                )
                            ),
                            stage2_estimator=_build_scaled_pipeline(
                                LogisticRegression(
                                    C=1.0, max_iter=2500, class_weight="balanced", random_state=RANDOM_STATE
                                )
                            ),
                            draw_probability_scale=1.0,
                        ),
                    ),
                    default_uncertainty_threshold=0.0,
                    default_draw_conviction_threshold=1.0,
                    segment_configs=tuned_segment_configs,
                    segment_detector_fn=tournament_segment_detector,
                    specialist_draw_weight_multiplier=SPECIALIST_DRAW_WEIGHT_MULTIPLIER,
                ),
            ),
            sample_weight_builder=_make_sample_weight_builder(1.2),
            family="segment_aware_hybrid",
            hyperparameters={
                "tag": "auto_tuned",
                "generalist_c": 2.0,
                "generalist_draw_boost": 1.2,
                "specialist_stage1_c": 2.0,
                "specialist_stage2_c": 1.0,
                "specialist_draw_weight_multiplier": SPECIALIST_DRAW_WEIGHT_MULTIPLIER,
                **{f"{seg}_unc": cfg.uncertainty_threshold for seg, cfg in tuned_segment_configs.items()},
                **{f"{seg}_conv": cfg.draw_conviction_threshold for seg, cfg in tuned_segment_configs.items()},
            },
            notes="Segment-aware hybrid [auto_tuned]: generated via post-datos OOF temporal validation search.",
        )
    else:
        # Fallback to hand-crafted variants if no auto-tuning dictionary is passed.
        segment_aware_variants: list[dict[str, float | str]] = [
            {
                "tag": "conservative",
                "friendlies_unc": 0.38, "friendlies_conv": 0.58,
                "worldcup_unc": 0.52, "worldcup_conv": 0.62,
                "continental_unc": 0.44, "continental_conv": 0.58,
                "qualifiers_unc": 0.48, "qualifiers_conv": 0.60,
                "default_unc": 0.45, "default_conv": 0.55,
            },
            {
                "tag": "friendlies_focus",
                "friendlies_unc": 0.32, "friendlies_conv": 0.52,
                "worldcup_unc": 0.55, "worldcup_conv": 0.65,
                "continental_unc": 0.46, "continental_conv": 0.58,
                "qualifiers_unc": 0.50, "qualifiers_conv": 0.60,
                "default_unc": 0.48, "default_conv": 0.58,
            },
            {
                "tag": "balanced",
                "friendlies_unc": 0.36, "friendlies_conv": 0.55,
                "worldcup_unc": 0.50, "worldcup_conv": 0.60,
                "continental_unc": 0.42, "continental_conv": 0.56,
                "qualifiers_unc": 0.46, "qualifiers_conv": 0.58,
                "default_unc": 0.44, "default_conv": 0.55,
            },
            {
                "tag": "narrow_band",
                "friendlies_unc": 0.40, "friendlies_conv": 0.60,
                "worldcup_unc": 0.55, "worldcup_conv": 0.65,
                "continental_unc": 0.48, "continental_conv": 0.60,
                "qualifiers_unc": 0.50, "qualifiers_conv": 0.62,
                "default_unc": 0.48, "default_conv": 0.58,
            },
        ]
    
        for variant in segment_aware_variants:
            tag = str(variant["tag"])
            default_unc = float(variant["default_unc"])
            default_conv = float(variant["default_conv"])
            segment_configs = _build_segment_configs(
                friendlies_unc=float(variant["friendlies_unc"]),
                friendlies_conv=float(variant["friendlies_conv"]),
                worldcup_unc=float(variant["worldcup_unc"]),
                worldcup_conv=float(variant["worldcup_conv"]),
                continental_unc=float(variant["continental_unc"]),
                continental_conv=float(variant["continental_conv"]),
                qualifiers_unc=float(variant["qualifiers_unc"]),
                qualifiers_conv=float(variant["qualifiers_conv"]),
            )
            name = f"seg_hybrid_{tag}"
            candidate_specs[name] = CandidateSpec(
            name=name,
            pipeline=cast(
                ProbabilisticEstimator,
                SegmentAwareHybridDrawOverrideEnsemble(
                    generalist_estimator=_build_scaled_pipeline(
                        LogisticRegression(
                            C=2.0,
                            max_iter=2500,
                            class_weight="balanced",
                            random_state=RANDOM_STATE,
                        )
                    ),
                    specialist_estimator=cast(
                        ProbabilisticEstimator,
                        TwoStageDrawClassifier(
                            stage1_estimator=_build_scaled_pipeline(
                                LogisticRegression(
                                    C=2.0,
                                    max_iter=2500,
                                    class_weight="balanced",
                                    random_state=RANDOM_STATE,
                                )
                            ),
                            stage2_estimator=_build_scaled_pipeline(
                                LogisticRegression(
                                    C=1.0,
                                    max_iter=2500,
                                    class_weight="balanced",
                                    random_state=RANDOM_STATE,
                                )
                            ),
                            draw_probability_scale=1.0,
                        ),
                    ),
                    default_uncertainty_threshold=default_unc,
                    default_draw_conviction_threshold=default_conv,
                    segment_configs=segment_configs,
                    segment_detector_fn=tournament_segment_detector,
                    specialist_draw_weight_multiplier=SPECIALIST_DRAW_WEIGHT_MULTIPLIER,
                ),
            ),
            sample_weight_builder=_make_sample_weight_builder(1.2),
            family="segment_aware_hybrid",
            hyperparameters={
                "tag": tag,
                "generalist_c": 2.0,
                "generalist_draw_boost": 1.2,
                "specialist_stage1_c": 2.0,
                "specialist_stage2_c": 1.0,
                "specialist_draw_weight_multiplier": SPECIALIST_DRAW_WEIGHT_MULTIPLIER,
                "default_unc": default_unc,
                "default_conv": default_conv,
                **{
                    f"{seg_id}_unc": cfg.uncertainty_threshold
                    for seg_id, cfg in segment_configs.items()
                },
                **{
                    f"{seg_id}_conv": cfg.draw_conviction_threshold
                    for seg_id, cfg in segment_configs.items()
                },
            },
            notes=(
                f"Segment-aware hybrid [{tag}]: overrides conditioned by tournament "
                f"segment. Friendlies unc={variant['friendlies_unc']}, "
                f"WC unc={variant['worldcup_unc']}, default unc={default_unc}"
            ),
        )

    return candidate_specs


def _fit_pipeline(
    pipeline: ProbabilisticEstimator,
    *,
    X: pd.DataFrame,
    y_encoded: pd.Series,
    sample_weight_builder: Callable[[pd.Series], NDArray[np.float64]],
) -> ProbabilisticEstimator:
    fitted_pipeline = cast(ProbabilisticEstimator, clone(pipeline))
    sample_weight = sample_weight_builder(y_encoded)
    fit_estimator_with_sample_weight(fitted_pipeline, X, y_encoded, sample_weight)
    return fitted_pipeline


def _fit_calibrated_variant(
    fitted_estimator: ProbabilisticEstimator,
    *,
    X_calibration: pd.DataFrame,
    y_calibration_encoded: pd.Series,
    method: str,
) -> CalibratedClassifierCV:
    calibrator = CalibratedClassifierCV(
        estimator=FrozenEstimator(fitted_estimator),
        method=method,
        cv=None,
    )
    calibrator.fit(X_calibration, y_calibration_encoded)
    return calibrator


def _split_calibration_window(
    calibration_df: pd.DataFrame,
    *,
    selection_size: float = DEFAULT_CALIBRATION_SELECTION_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < selection_size < 1:
        raise ValueError("selection_size must be between 0 and 1.")

    ordered_df = calibration_df.sort_values("date").reset_index(drop=True)
    selection_rows = max(1, int(len(ordered_df) * selection_size))
    fit_rows = len(ordered_df) - selection_rows
    if fit_rows < 100 or selection_rows < 100:
        raise ValueError(
            "Calibration window is too small to support fit and selection splits."
        )

    return (
        ordered_df.iloc[:fit_rows].copy(),
        ordered_df.iloc[fit_rows:].copy(),
    )


def _date_range(df: pd.DataFrame) -> DateRange:
    return {
        "start": df["date"].min().date().isoformat(),
        "end": df["date"].max().date().isoformat(),
    }


def _model_class_name(estimator: ProbabilisticEstimator) -> str:
    named_steps = getattr(estimator, "named_steps", None)
    if named_steps is not None and "model" in named_steps:
        return str(named_steps["model"].__class__.__name__)
    return estimator.__class__.__name__


def _evaluate_pipeline(
    pipeline: ProbabilisticEstimator,
    *,
    X: pd.DataFrame,
    y: pd.Series,
) -> TrainingMetrics:
    y_encoded = y.map(OUTCOME_TO_ENCODED).astype("int64")
    probabilities = predict_proba_aligned(pipeline, X)
    predicted_encoded = pipeline.predict(X).astype(np.int64)
    evaluation = evaluate_multiclass_predictions(
        y_true=y,
        y_true_encoded=y_encoded,
        y_pred_encoded=predicted_encoded,
        probabilities=probabilities,
    )
    metrics: TrainingMetrics = {
        "accuracy": cast(float, evaluation["accuracy"]),
        "macro_f1": cast(float, evaluation["macro_f1"]),
        "weighted_f1": cast(float, evaluation["weighted_f1"]),
        "balanced_accuracy": cast(float, evaluation["balanced_accuracy"]),
        "matthews_corrcoef": cast(float, evaluation["matthews_corrcoef"]),
        "cohen_kappa": cast(float, evaluation["cohen_kappa"]),
        "log_loss": cast(float, evaluation["log_loss"]),
        "multiclass_brier_score": cast(
            float,
            evaluation["multiclass_brier_score"],
        ),
        "expected_calibration_error": cast(
            float,
            evaluation["expected_calibration_error"],
        ),
        "draw_f1": cast(float, evaluation["draw_f1"]),
        "draw_recall": cast(float, evaluation["draw_recall"]),
        "classification_report": cast(dict[str, object], evaluation["classification_report"]),
    }
    return metrics


def train_and_export_model(
    data_path: Path | None = None,
    artifact_path: Path | None = None,
    test_size: float = DEFAULT_TEST_SIZE,
    calibration_size: float = DEFAULT_CALIBRATION_SIZE,
    backtest_splits: int = DEFAULT_BACKTEST_SPLITS,
    calibration_selection_size: float = DEFAULT_CALIBRATION_SELECTION_SIZE,
    persist_to_db: bool = False,
    pipeline_run_id: str | None = None,
) -> TrainingSummary:
    """
    Train the production model from the gold feature dataset and export it.

    Args:
        data_path: Optional alternate path to the gold feature dataset
        artifact_path: Optional target path for the exported joblib artifact
        test_size: Fraction of the most recent data reserved for evaluation
        calibration_size: Fraction reserved for pre-test calibration
        backtest_splits: Number of rolling temporal splits for model selection
        calibration_selection_size: Fraction of calibration data reserved to choose deployment variant
        persist_to_db: Whether to persist training metadata to PostgreSQL
        pipeline_run_id: Optional orchestrator run identifier for lineage

    Returns:
        Dictionary with training metadata and evaluation metrics
    """
    settings.ensure_project_dirs()
    dataset_path = Path(data_path or settings.GOLD_DIR / "features_dataset.csv")
    output_path = Path(artifact_path or settings.MODEL_ARTIFACT_PATH)

    logger.info("Loading gold feature dataset from %s", dataset_path)
    df = load_feature_dataset(dataset_path)
    validate_feature_dataset_contract(df)
    feature_columns = select_model_feature_columns(df)
    if not feature_columns:
        raise RuntimeError("No model feature columns were found in the gold dataset.")

    temporal_split = split_train_calibration_test(
        df,
        test_size=test_size,
        calibration_size=calibration_size,
    )
    train_df = temporal_split.train_df
    calibration_df = temporal_split.calibration_df
    test_df = temporal_split.test_df
    calibration_fit_df, calibration_selection_df = _split_calibration_window(
        calibration_df,
        selection_size=calibration_selection_size,
    )
    logger.info(
        "Temporal split complete: train=%s rows, calibration=%s rows, test=%s rows",
        len(train_df),
        len(calibration_df),
        len(test_df),
    )
    logger.info(
        "Calibration window split into fit=%s rows and selection=%s rows",
        len(calibration_fit_df),
        len(calibration_selection_df),
    )

    X_train = train_df[feature_columns].copy()
    X_test = test_df[feature_columns].copy()
    y_train = train_df[TARGET_COLUMN].map(OUTCOME_TO_ENCODED)
    y_test = test_df[TARGET_COLUMN]
    from src.modeling.tuning import auto_tune_segment_thresholds
    
    logger.info("Auto-tuning segment thresholds using OOF prior to evaluating candidates...")
    
    auto_tune_gen_pipe = _build_scaled_pipeline(
        LogisticRegression(
            C=2.0, max_iter=2500, class_weight="balanced", random_state=RANDOM_STATE
        )
    )
    auto_tune_spec_pipe = TwoStageDrawClassifier(
        stage1_estimator=_build_scaled_pipeline(
            LogisticRegression(
                C=2.0, max_iter=2500, class_weight="balanced", random_state=RANDOM_STATE
            )
        ),
        stage2_estimator=_build_scaled_pipeline(
            LogisticRegression(
                C=1.0, max_iter=2500, class_weight="balanced", random_state=RANDOM_STATE
            )
        ),
        draw_probability_scale=1.0,
    )
    
    def specialist_weight_fn(y_encoded: pd.Series) -> NDArray[np.float64]:
        base_w = _make_sample_weight_builder(1.2)(y_encoded)
        draw_mask = (y_encoded.to_numpy() == 1)
        base_w[draw_mask] *= SPECIALIST_DRAW_WEIGHT_MULTIPLIER
        base_w *= len(base_w) / base_w.sum()
        return base_w

    tuned_configs = auto_tune_segment_thresholds(
        X=train_df[feature_columns].copy(),
        y_encoded=train_df[TARGET_COLUMN].map(OUTCOME_TO_ENCODED),
        metadata_df=train_df[["tournament"]],
        segment_detector_fn=tournament_segment_detector,
        generalist_pipeline=cast(ProbabilisticEstimator, auto_tune_gen_pipe),
        generalist_sample_weight_fn=_make_sample_weight_builder(1.2),
        specialist_pipeline=cast(ProbabilisticEstimator, auto_tune_spec_pipe),
        specialist_sample_weight_fn=specialist_weight_fn,
        n_splits=backtest_splits,
    )
    
    candidate_specs = _build_candidate_specs(tuned_configs)
    logger.info(
        "Running temporal backtesting with %s splits across %s candidates",
        backtest_splits,
        len(candidate_specs),
    )
    candidate_backtests, selected_model_name = evaluate_candidates_with_backtesting(
        candidate_specs=candidate_specs,
        train_df=train_df,
        feature_columns=feature_columns,
        target_column=TARGET_COLUMN,
        outcome_to_encoded=OUTCOME_TO_ENCODED,
        n_splits=backtest_splits,
        metadata_columns=SEGMENT_METADATA_COLUMNS,
    )
    logger.info("Selected candidate after backtesting: %s", selected_model_name)

    selected_candidate_spec = candidate_specs[selected_model_name]
    selected_candidate_pipeline = selected_candidate_spec.pipeline

    X_calibration_fit = calibration_fit_df[feature_columns].copy()
    X_calibration_selection = calibration_selection_df[feature_columns].copy()
    y_calibration_fit = calibration_fit_df[TARGET_COLUMN].map(OUTCOME_TO_ENCODED)
    y_calibration_selection = calibration_selection_df[TARGET_COLUMN]

    selected_pipeline_for_calibration = _fit_pipeline(
        selected_candidate_pipeline,
        X=X_train,
        y_encoded=y_train,
        sample_weight_builder=selected_candidate_spec.sample_weight_builder,
    )
    calibration_selection_metrics: dict[str, TrainingMetrics] = {
        "uncalibrated": _evaluate_pipeline(
            selected_pipeline_for_calibration,
            X=X_calibration_selection,
            y=y_calibration_selection,
        )
    }
    for method in ("sigmoid", "isotonic"):
        calibrated_variant = _fit_calibrated_variant(
            selected_pipeline_for_calibration,
            X_calibration=X_calibration_fit,
            y_calibration_encoded=y_calibration_fit,
            method=method,
        )
        calibration_selection_metrics[method] = _evaluate_pipeline(
            calibrated_variant,
            X=X_calibration_selection,
            y=y_calibration_selection,
        )

    deployed_model_variant, deployment_decision = select_deployment_variant(
        calibration_selection_metrics,
    )
    logger.info("Selected deployment variant: %s", deployed_model_variant)

    pretest_df = pd.concat([train_df, calibration_df], ignore_index=True)
    X_pretest = pretest_df[feature_columns].copy()
    y_pretest = pretest_df[TARGET_COLUMN].map(OUTCOME_TO_ENCODED)
    final_uncalibrated_model = _fit_pipeline(
        selected_candidate_pipeline,
        X=X_pretest,
        y_encoded=y_pretest,
        sample_weight_builder=selected_candidate_spec.sample_weight_builder,
    )

    final_deployed_model: ProbabilisticEstimator = final_uncalibrated_model
    calibration_method = "none"
    if deployed_model_variant in {"sigmoid", "isotonic"}:
        base_rows = pd.concat([train_df, calibration_fit_df], ignore_index=True)
        X_base = base_rows[feature_columns].copy()
        y_base = base_rows[TARGET_COLUMN].map(OUTCOME_TO_ENCODED)
        base_pipeline = _fit_pipeline(
            selected_candidate_pipeline,
            X=X_base,
            y_encoded=y_base,
            sample_weight_builder=selected_candidate_spec.sample_weight_builder,
        )
        final_deployed_model = _fit_calibrated_variant(
            base_pipeline,
            X_calibration=X_calibration_selection,
            y_calibration_encoded=calibration_selection_df[TARGET_COLUMN].map(
                OUTCOME_TO_ENCODED
            ),
            method=deployed_model_variant,
        )
        calibration_method = deployed_model_variant

    uncalibrated_metrics = _evaluate_pipeline(
        final_uncalibrated_model,
        X=X_test,
        y=y_test,
    )
    deployed_metrics = _evaluate_pipeline(
        final_deployed_model,
        X=X_test,
        y=y_test,
    )
    metrics = deployed_metrics

    training_summary: TrainingSummary = {
        "artifact_path": str(output_path),
        "training_rows": int(len(train_df)),
        "calibration_rows": int(len(calibration_df)),
        "test_rows": int(len(test_df)),
        "feature_count": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "data_path": str(dataset_path),
        "train_date_range": _date_range(train_df),
        "calibration_date_range": _date_range(calibration_df),
        "test_date_range": _date_range(test_df),
        "class_distribution_train": {
            str(label): int(count)
            for label, count in train_df[TARGET_COLUMN].value_counts().sort_index().items()
        },
        "class_distribution_calibration": {
            str(label): int(count)
            for label, count in calibration_df[TARGET_COLUMN].value_counts().sort_index().items()
        },
        "class_distribution_test": {
            str(label): int(count)
            for label, count in test_df[TARGET_COLUMN].value_counts().sort_index().items()
        },
        "selected_model_name": selected_model_name,
        "selected_model_class": _model_class_name(selected_candidate_pipeline),
        "deployed_model_variant": deployed_model_variant,
        "calibration_method": calibration_method,
        "metrics": metrics,
        "uncalibrated_metrics": uncalibrated_metrics,
        "evaluation_artifacts": {
            "selection_strategy": "temporal_backtesting_rank_sum_draw_aware",
            "selection_metrics": [
                "macro_f1",
                "draw_f1",
                "draw_recall",
                "balanced_accuracy",
                "matthews_corrcoef",
                "log_loss",
                "multiclass_brier_score",
                "expected_calibration_error",
            ],
            "candidate_search_space": {
                "candidate_count": int(len(candidate_specs)),
                "families": sorted(
                    {
                        candidate_spec.family
                        for candidate_spec in candidate_specs.values()
                    }
                ),
            },
            "candidate_backtests": candidate_backtests,
            "calibration_variant_selection": {
                "fit_rows": int(len(calibration_fit_df)),
                "selection_rows": int(len(calibration_selection_df)),
                "fit_date_range": _date_range(calibration_fit_df),
                "selection_date_range": _date_range(calibration_selection_df),
                "candidate_metrics": calibration_selection_metrics,
                "deployment_decision": deployment_decision,
            },
        },
    }

    report_payload = generate_evaluation_report(
        training_summary=training_summary,
        test_df=test_df,
        probabilities=predict_proba_aligned(final_deployed_model, X_test),
        y_pred_encoded=final_deployed_model.predict(X_test).astype(np.int64),
        artifact_path=output_path,
    )
    training_summary["evaluation_artifacts"]["report_artifacts"] = {
        "report_json": cast(dict[str, object], report_payload["artifacts"])["report_json"],
        "report_markdown": cast(dict[str, object], report_payload["artifacts"])["report_markdown"],
        "confusion_matrix_png": cast(dict[str, object], report_payload["artifacts"])["confusion_matrix_png"],
        "calibration_curves_png": cast(dict[str, object], report_payload["artifacts"])["calibration_curves_png"],
    }

    artifact: ModelArtifactBundle = {
        "model": final_deployed_model,
        "feature_columns": feature_columns,
        "target_column": TARGET_COLUMN,
        "outcome_to_encoded": OUTCOME_TO_ENCODED,
        "encoded_to_outcome": ENCODED_TO_OUTCOME,
        "outcome_labels": OUTCOME_LABELS,
        "selected_model_name": selected_model_name,
        "deployed_model_variant": deployed_model_variant,
        "calibration_method": calibration_method,
        "training_summary": training_summary,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)
    metrics_path = output_path.with_name(f"{output_path.stem}_metrics.json")
    metrics_path.write_text(json.dumps(training_summary, indent=2), encoding="utf-8")
    logger.info("Model artifact exported to %s", output_path)
    logger.info("Training metrics exported to %s", metrics_path)
    if persist_to_db:
        persist_training_run(
            training_summary,
            pipeline_run_id=pipeline_run_id,
        )
        logger.info("Training metadata appended to gold.training_runs")
    logger.info(
        "Holdout metrics (%s) | accuracy=%.4f macro_f1=%.4f weighted_f1=%.4f log_loss=%.4f",
        deployed_model_variant,
        training_summary["metrics"]["accuracy"],
        training_summary["metrics"]["macro_f1"],
        training_summary["metrics"]["weighted_f1"],
        training_summary["metrics"]["log_loss"],
    )

    # ============================================================================
    # Shadow Deployment Export
    # ============================================================================
    shadow_candidates = [
        c for c in candidate_backtests if c["family"] == "segment_aware_hybrid"
    ]
    if shadow_candidates:
        shadow_candidates.sort(key=lambda c: cast(float, c["overall_rank"]))
        shadow_model_name = cast(str, shadow_candidates[0]["name"])
        logger.info("Training and exporting shadow candidate: %s", shadow_model_name)

        shadow_spec = candidate_specs[shadow_model_name]
        final_shadow_model = _fit_pipeline(
            shadow_spec.pipeline,
            X=X_pretest,
            y_encoded=y_pretest,
            sample_weight_builder=shadow_spec.sample_weight_builder,
        )

        shadow_artifact_path = output_path.with_name(f"{output_path.stem}_shadow.joblib")
        shadow_artifact: ModelArtifactBundle = {
            "model": final_shadow_model,
            "feature_columns": feature_columns,
            "target_column": TARGET_COLUMN,
            "outcome_to_encoded": OUTCOME_TO_ENCODED,
            "encoded_to_outcome": ENCODED_TO_OUTCOME,
            "outcome_labels": OUTCOME_LABELS,
            "selected_model_name": shadow_model_name,
            "deployed_model_variant": "uncalibrated",
            "calibration_method": "none",
            "training_summary": training_summary,
        }
        joblib.dump(shadow_artifact, shadow_artifact_path)
        logger.info("Shadow model artifact exported to %s", shadow_artifact_path)

    return training_summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and export the World Cup match predictor."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=settings.GOLD_DIR / "features_dataset.csv",
        help="Path to the gold feature dataset CSV.",
    )
    parser.add_argument(
        "--artifact-path",
        type=Path,
        default=settings.MODEL_ARTIFACT_PATH,
        help="Where to save the exported joblib artifact.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help="Fraction of the most recent data reserved for evaluation.",
    )
    parser.add_argument(
        "--calibration-size",
        type=float,
        default=DEFAULT_CALIBRATION_SIZE,
        help="Fraction reserved for post-train probability calibration before final test.",
    )
    parser.add_argument(
        "--backtest-splits",
        type=int,
        default=DEFAULT_BACKTEST_SPLITS,
        help="Number of rolling temporal splits used for candidate selection.",
    )
    parser.add_argument(
        "--calibration-selection-size",
        type=float,
        default=DEFAULT_CALIBRATION_SELECTION_SIZE,
        help="Fraction of the calibration window reserved to choose deployment variant.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    args = _parse_args()
    summary = train_and_export_model(
        data_path=args.data_path,
        artifact_path=args.artifact_path,
        test_size=args.test_size,
        calibration_size=args.calibration_size,
        backtest_splits=args.backtest_splits,
        calibration_selection_size=args.calibration_selection_size,
    )
    print(json.dumps(summary, indent=2))
