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


def _sample_latest_team_snapshots_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "snapshot_date": pd.to_datetime(
                [
                    "2026-03-25",
                    "2026-03-25",
                    "2026-03-25",
                    "2026-03-26",
                    "2026-03-26",
                    "2026-03-26",
                ]
            ),
            "team": [
                "Colombia",
                "Colombia",
                "Colombia",
                "Argentina",
                "Argentina",
                "Argentina",
            ],
            "opponent": [
                "Peru",
                "Peru",
                "Peru",
                "Brazil",
                "Brazil",
                "Brazil",
            ],
            "team_role": ["home", "away", "overall", "home", "away", "overall"],
            "elo": [1805.0, 1798.0, 1805.0, 1910.0, 1902.0, 1902.0],
            "opponent_elo": [1720.0, 1740.0, 1720.0, 1888.0, 1862.0, 1888.0],
            "avg_goals_last5": [1.8, 1.2, 1.6, 2.0, 1.4, 1.5],
            "avg_goals_conceded_last5": [0.6, 0.8, 0.7, 0.4, 0.7, 0.6],
            "global_avg_goals_last5": [1.6, 1.6, 1.6, 1.5, 1.5, 1.5],
            "global_avg_conceded_last5": [0.7, 0.7, 0.7, 0.6, 0.6, 0.6],
            "win_rate_last5": [0.8, 0.4, 0.6, 0.8, 0.5, 0.6],
            "global_win_rate_last5": [0.6, 0.6, 0.6, 0.6, 0.6, 0.6],
            "avg_opponent_elo_last5": [1765.0, 1745.0, 1755.0, 1850.0, 1840.0, 1845.0],
            "weighted_win_rate_last5": [0.78, 0.42, 0.6, 0.81, 0.48, 0.63],
            "opponent_elo_form": [1710.0, 1730.0, 1720.0, 1875.0, 1855.0, 1865.0],
            "elo_form": [1802.0, 1794.0, 1799.0, 1908.0, 1898.0, 1900.0],
            "home_advantage_effect": [0.2, 0.2, 0.2, 0.15, 0.15, 0.15],
            "is_friendly": [0, 0, 0, 0, 0, 0],
            "is_world_cup": [0, 0, 0, 0, 0, 0],
            "is_qualifier": [1, 1, 1, 1, 1, 1],
            "is_continental": [0, 0, 0, 0, 0, 0],
            "pipeline_run_id": ["run-123"] * 6,
            "persisted_at_utc": pd.to_datetime(["2026-04-02T00:00:00+00:00"] * 6),
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


def test_build_match_feature_frame_from_latest_snapshots():
    latest_snapshots_df = _sample_latest_team_snapshots_df()
    feature_columns = [
        "elo_home",
        "elo_away",
        "elo_diff",
        "home_avg_goals_last5",
        "away_avg_goals_last5",
        "home_global_win_rate_last5",
        "away_global_win_rate_last5",
        "home_advantage_effect",
        "is_qualifier",
    ]

    feature_frame, snapshot_dates = features.build_match_feature_frame_from_latest_snapshots(
        home_team="Colombia",
        away_team="Argentina",
        tournament="FIFA World Cup Qualifiers",
        neutral=False,
        feature_columns=feature_columns,
        latest_team_snapshots_df=latest_snapshots_df,
    )

    assert feature_frame.loc[0, "elo_home"] == 1805.0
    assert feature_frame.loc[0, "elo_away"] == 1902.0
    assert feature_frame.loc[0, "elo_diff"] == -97.0
    assert feature_frame.loc[0, "home_avg_goals_last5"] == 1.8
    assert feature_frame.loc[0, "away_avg_goals_last5"] == 1.4
    assert feature_frame.loc[0, "is_qualifier"] == 1
    assert snapshot_dates == {
        "home_team": "Colombia",
        "away_team": "Argentina",
        "home_snapshot_date": "2026-03-25",
        "away_snapshot_date": "2026-03-26",
    }
