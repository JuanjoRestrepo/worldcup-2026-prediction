"""Unit tests for PostgreSQL persistence helpers."""

import pandas as pd

from src.database.persistence import build_training_run_frame, persist_dataframe


class _DummyConnection:
    def __init__(self) -> None:
        self.statements: list[str] = []

    def exec_driver_sql(self, statement: str) -> None:
        self.statements.append(statement)


class _DummyBeginContext:
    def __init__(self, connection: _DummyConnection) -> None:
        self.connection = connection

    def __enter__(self) -> _DummyConnection:
        return self.connection

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class _DummyEngine:
    def __init__(self) -> None:
        self.connection = _DummyConnection()

    def begin(self) -> _DummyBeginContext:
        return _DummyBeginContext(self.connection)


def test_persist_dataframe_creates_schema_and_adds_lineage(monkeypatch):
    df = pd.DataFrame({"match_id": [1, 2], "value": [10, 20]})
    engine = _DummyEngine()
    captured: dict[str, object] = {}

    def fake_to_sql(
        self,
        name,
        con,
        schema,
        if_exists,
        index,
        method,
        chunksize,
    ):
        captured["df"] = self.copy()
        captured["name"] = name
        captured["con"] = con
        captured["schema"] = schema
        captured["if_exists"] = if_exists
        captured["index"] = index
        captured["method"] = method
        captured["chunksize"] = chunksize

    monkeypatch.setattr(pd.DataFrame, "to_sql", fake_to_sql)

    persist_dataframe(
        df,
        schema_name="bronze",
        table_name="historical_matches",
        engine=engine,
        pipeline_run_id="run-123",
    )

    persisted_df = captured["df"]
    assert 'CREATE SCHEMA IF NOT EXISTS "bronze"' in engine.connection.statements
    assert captured["name"] == "historical_matches"
    assert captured["schema"] == "bronze"
    assert captured["if_exists"] == "replace"
    assert captured["method"] == "multi"
    assert "pipeline_run_id" in persisted_df.columns
    assert "persisted_at_utc" in persisted_df.columns
    assert persisted_df["pipeline_run_id"].tolist() == ["run-123", "run-123"]


def test_build_training_run_frame_serializes_nested_metadata():
    training_summary = {
        "artifact_path": "models/match_predictor.joblib",
        "data_path": "data/gold/features_dataset.csv",
        "training_rows": 100,
        "test_rows": 20,
        "feature_count": 5,
        "feature_columns": ["elo_home", "elo_away"],
        "train_date_range": {"start": "2020-01-01", "end": "2024-01-01"},
        "test_date_range": {"start": "2024-01-02", "end": "2025-01-01"},
        "class_distribution_train": {"-1": 30, "0": 20, "1": 50},
        "class_distribution_test": {"-1": 5, "0": 5, "1": 10},
        "metrics": {
            "accuracy": 0.6,
            "macro_f1": 0.5,
            "weighted_f1": 0.58,
            "log_loss": 0.91,
            "classification_report": {"home_win": {"precision": 0.7}},
        },
    }

    frame = build_training_run_frame(training_summary, pipeline_run_id="run-456")

    assert len(frame) == 1
    assert frame.loc[0, "pipeline_run_id"] == "run-456"
    assert frame.loc[0, "artifact_path"] == "models/match_predictor.joblib"
    assert '"elo_home"' in frame.loc[0, "feature_columns_json"]
    assert '"accuracy": 0.6' not in frame.loc[0, "classification_report_json"]
    assert '"home_win"' in frame.loc[0, "classification_report_json"]
