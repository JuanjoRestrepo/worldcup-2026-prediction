"""Unit tests for gold feature dataset loading across serving sources."""

from pathlib import Path

import pandas as pd
import pytest

from src.modeling import features


def _sample_feature_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-03-25"]),
            "homeTeam": ["Colombia"],
            "awayTeam": ["Argentina"],
            "elo_home": [1805.0],
            "elo_away": [1902.0],
            "target_multiclass": [1],
            "target": [1],
            "pipeline_run_id": ["run-123"],
            "persisted_at_utc": ["2026-04-02T00:00:00+00:00"],
        }
    )


@pytest.fixture(autouse=True)
def clear_feature_dataset_cache() -> None:
    features.clear_feature_dataset_cache()
    yield
    features.clear_feature_dataset_cache()


def test_select_model_feature_columns_excludes_lineage_columns():
    df = _sample_feature_df()

    assert features.select_model_feature_columns(df) == ["elo_home", "elo_away"]


def test_load_feature_dataset_with_source_prefers_postgres(monkeypatch):
    sample_df = _sample_feature_df()

    monkeypatch.setattr(
        features,
        "_load_feature_dataset_from_postgres",
        lambda: sample_df,
    )
    monkeypatch.setattr(
        features,
        "_load_feature_dataset_from_csv",
        lambda dataset_path: pytest.fail(
            f"CSV fallback should not be used when PostgreSQL succeeds: {dataset_path}"
        ),
    )

    loaded_df, active_source = features.load_feature_dataset_with_source(source="auto")

    assert active_source == "postgres"
    pd.testing.assert_frame_equal(loaded_df, sample_df)


def test_load_feature_dataset_with_source_falls_back_to_csv(monkeypatch):
    sample_df = _sample_feature_df()

    def fail_postgres() -> pd.DataFrame:
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(features, "_load_feature_dataset_from_postgres", fail_postgres)
    monkeypatch.setattr(
        features,
        "_load_feature_dataset_from_csv",
        lambda dataset_path: sample_df.assign(source_path=str(dataset_path)),
    )

    loaded_df, active_source = features.load_feature_dataset_with_source(
        dataset_path=Path("data/gold/features_dataset.csv"),
        source="auto",
    )

    assert active_source == "csv"
    assert loaded_df.loc[0, "source_path"] == "data\\gold\\features_dataset.csv"


def test_load_feature_dataset_with_source_requires_postgres_when_requested(
    monkeypatch,
):
    monkeypatch.setattr(
        features,
        "_load_feature_dataset_from_postgres",
        lambda: (_ for _ in ()).throw(RuntimeError("missing table")),
    )

    with pytest.raises(
        RuntimeError,
        match="Failed to load gold.features_dataset from PostgreSQL.",
    ):
        features.load_feature_dataset_with_source(source="postgres")
