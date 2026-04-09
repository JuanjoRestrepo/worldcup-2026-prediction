"""Unit tests for dbt-backed serving and monitoring loaders."""

from datetime import date

import pandas as pd
import pytest

from src.modeling import serving_store


def test_load_latest_training_run_summary_with_source_prefers_dbt(monkeypatch):
    monkeypatch.setattr(
        serving_store,
        "_load_latest_training_run_from_dbt",
        lambda: {"artifact_path": "models/match_predictor.joblib"},
    )
    monkeypatch.setattr(
        serving_store,
        "_load_latest_training_run_from_postgres",
        lambda: pytest.fail("raw postgres fallback should not be used"),
    )

    training_run, active_source = serving_store.load_latest_training_run_summary_with_source(
        source="auto"
    )

    assert active_source == "dbt_latest_training_run"
    assert training_run["artifact_path"] == "models/match_predictor.joblib"


def test_load_latest_training_run_summary_with_source_falls_back_to_postgres(
    monkeypatch,
):
    monkeypatch.setattr(
        serving_store,
        "_load_latest_training_run_from_dbt",
        lambda: (_ for _ in ()).throw(RuntimeError("dbt relation missing")),
    )
    monkeypatch.setattr(
        serving_store,
        "_load_latest_training_run_from_postgres",
        lambda: {"artifact_path": "models/raw.joblib"},
    )

    training_run, active_source = serving_store.load_latest_training_run_summary_with_source(
        source="auto"
    )

    assert active_source == "postgres_training_runs"
    assert training_run["artifact_path"] == "models/raw.joblib"


def test_load_team_snapshots_as_of_date_from_dbt_returns_historical_source(monkeypatch):
    sample_df = pd.DataFrame(
        {
            "snapshot_date": ["2025-11-15", "2025-11-15"],
            "team": ["United States", "Argentina"],
            "team_role": ["overall", "overall"],
            "persisted_at_utc": ["2026-04-02T00:00:00+00:00", "2026-04-02T00:00:00+00:00"],
        }
    )
    monkeypatch.setattr(serving_store, "_read_relation", lambda query: sample_df.copy())

    snapshots, active_source = serving_store.load_team_snapshots_as_of_date_from_dbt(
        date(2025, 11, 18)
    )

    assert active_source == "dbt_team_snapshots_as_of_date"
    assert pd.api.types.is_datetime64_any_dtype(snapshots["snapshot_date"])
