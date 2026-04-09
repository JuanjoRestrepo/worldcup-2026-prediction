"""Shared typed payloads for training, serving, and monitoring."""

from __future__ import annotations

from typing import TypedDict

import pandas as pd
from sklearn.pipeline import Pipeline


class DateRange(TypedDict):
    start: str
    end: str


class TrainingMetrics(TypedDict):
    accuracy: float
    macro_f1: float
    weighted_f1: float
    log_loss: float
    classification_report: dict[str, object]


class TrainingSummary(TypedDict):
    artifact_path: str
    data_path: str
    training_rows: int
    test_rows: int
    feature_count: int
    feature_columns: list[str]
    train_date_range: DateRange
    test_date_range: DateRange
    class_distribution_train: dict[str, int]
    class_distribution_test: dict[str, int]
    metrics: TrainingMetrics


class ModelArtifactBundle(TypedDict):
    model: Pipeline
    feature_columns: list[str]
    target_column: str
    outcome_to_encoded: dict[int, int]
    encoded_to_outcome: dict[int, int]
    outcome_labels: dict[int, str]
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
