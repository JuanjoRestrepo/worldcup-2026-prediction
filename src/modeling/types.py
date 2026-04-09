"""Shared typed payloads for training, serving, and monitoring."""

from __future__ import annotations

from typing import Any, NotRequired, TypedDict

import pandas as pd


class DateRange(TypedDict):
    start: str
    end: str


class TrainingMetrics(TypedDict):
    accuracy: float
    macro_f1: float
    weighted_f1: float
    balanced_accuracy: float
    matthews_corrcoef: float
    cohen_kappa: float
    log_loss: float
    multiclass_brier_score: float
    expected_calibration_error: float
    draw_f1: NotRequired[float]
    draw_recall: NotRequired[float]
    classification_report: dict[str, object]


class TrainingSummary(TypedDict):
    artifact_path: str
    data_path: str
    training_rows: int
    test_rows: int
    calibration_rows: int
    feature_count: int
    feature_columns: list[str]
    train_date_range: DateRange
    calibration_date_range: DateRange
    test_date_range: DateRange
    class_distribution_train: dict[str, int]
    class_distribution_calibration: dict[str, int]
    class_distribution_test: dict[str, int]
    selected_model_name: str
    selected_model_class: str
    deployed_model_variant: str
    calibration_method: str
    metrics: TrainingMetrics
    uncalibrated_metrics: NotRequired[TrainingMetrics]
    evaluation_artifacts: NotRequired[dict[str, object]]


class ModelArtifactBundle(TypedDict):
    model: Any
    feature_columns: list[str]
    target_column: str
    outcome_to_encoded: dict[int, int]
    encoded_to_outcome: dict[int, int]
    outcome_labels: dict[int, str]
    selected_model_name: str
    deployed_model_variant: str
    calibration_method: str
    training_summary: TrainingSummary


class TeamSnapshotMetadata(TypedDict):
    home_team: str
    away_team: str
    home_snapshot_date: str
    away_snapshot_date: str


class FeatureSnapshotDates(TypedDict):
    home_team: str
    away_team: str


class OverallSnapshot(TypedDict):
    elo: float
    global_avg_goals_last5: float
    global_avg_conceded_last5: float
    global_win_rate_last5: float
    date: pd.Timestamp


class TeamContext(TypedDict):
    overall: pd.Series
    home: pd.Series | None
    away: pd.Series | None
    elo: float
    global_avg_goals_last5: float
    global_avg_conceded_last5: float
    global_win_rate_last5: float
    snapshot_date: pd.Timestamp


class PredictionResult(TypedDict):
    home_team: str
    away_team: str
    predicted_class: int
    predicted_outcome: str
    class_probabilities: dict[str, float]
    neutral: bool
    tournament: str | None
    match_date: str | None
    feature_snapshot_dates: FeatureSnapshotDates
    feature_source: str
    model_artifact_path: str


class LatestTrainingRunSummary(TypedDict):
    pipeline_run_id: str | None
    artifact_path: str
    data_path: str
    training_rows: int
    test_rows: int
    feature_count: int
    train_date_start: str
    train_date_end: str
    test_date_start: str
    test_date_end: str
    accuracy: float
    macro_f1: float
    weighted_f1: float
    log_loss: float
    trained_at_utc: str
    persisted_at_utc: str | None
