"""Shared feature utilities for model training and inference."""

from __future__ import annotations

from datetime import date
import logging
from functools import lru_cache
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pandas as pd
from sqlalchemy.exc import SQLAlchemyError

from src.config.settings import settings
from src.database.connection import get_sqlalchemy_engine
from src.modeling.serving_store import clear_serving_store_cache

TARGET_COLUMN = "target_multiclass"
BINARY_TARGET_COLUMN = "target"
LEAKAGE_COLUMNS = {"goal_diff", TARGET_COLUMN, BINARY_TARGET_COLUMN}
LINEAGE_COLUMNS = {"pipeline_run_id", "persisted_at_utc"}
NON_FEATURE_COLUMNS = {
    "date",
    "homeTeam",
    "awayTeam",
    "homeGoals",
    "awayGoals",
    "tournament",
    "city",
    "country",
    *LINEAGE_COLUMNS,
    *LEAKAGE_COLUMNS,
}
OUTCOME_LABELS = {
    -1: "away_win",
    0: "draw",
    1: "home_win",
}
FEATURE_SOURCE_OPTIONS = {"csv", "postgres", "auto"}
FEATURE_DB_SCHEMA = "gold"
FEATURE_DB_TABLE = "features_dataset"

logger = logging.getLogger(__name__)

FeatureDatasetSource = Literal["csv", "postgres", "auto"]


@lru_cache(maxsize=2)
def _load_feature_dataset_from_csv_cached(dataset_path: str) -> pd.DataFrame:
    df = pd.read_csv(dataset_path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


@lru_cache(maxsize=1)
def _load_feature_dataset_from_postgres_cached(
    db_host: str,
    db_port: int,
    db_name: str,
    db_user: str,
) -> pd.DataFrame:
    engine = get_sqlalchemy_engine()
    try:
        df = pd.read_sql_query(
            f'SELECT * FROM "{FEATURE_DB_SCHEMA}"."{FEATURE_DB_TABLE}"',
            con=engine,
        )
    finally:
        engine.dispose()

    if df.empty:
        raise RuntimeError(
            f"PostgreSQL table {FEATURE_DB_SCHEMA}.{FEATURE_DB_TABLE} is empty."
        )
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def clear_feature_dataset_cache() -> None:
    """Clear cached CSV and PostgreSQL feature snapshots."""
    _load_feature_dataset_from_csv_cached.cache_clear()
    _load_feature_dataset_from_postgres_cached.cache_clear()
    clear_serving_store_cache()


def _normalize_feature_source(source: str) -> FeatureDatasetSource:
    normalized_source = source.strip().lower()
    if normalized_source not in FEATURE_SOURCE_OPTIONS:
        raise ValueError("feature source must be one of: auto, postgres, csv.")
    return cast(FeatureDatasetSource, normalized_source)


def _load_feature_dataset_from_csv(dataset_path: Path) -> pd.DataFrame:
    resolved_path = dataset_path.resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Gold feature dataset not found at '{resolved_path}'."
        )
    return _load_feature_dataset_from_csv_cached(str(resolved_path))


def _load_feature_dataset_from_postgres() -> pd.DataFrame:
    return _load_feature_dataset_from_postgres_cached(
        settings.DB_HOST,
        settings.DB_PORT,
        settings.DB_NAME,
        settings.DB_USER,
    )


def load_feature_dataset_with_source(
    dataset_path: Path | None = None,
    source: str = "csv",
) -> tuple[pd.DataFrame, str]:
    """Load the gold feature dataset and return the resolved source."""
    resolved_path = Path(dataset_path or settings.GOLD_DIR / "features_dataset.csv")
    normalized_source = _normalize_feature_source(source)

    if normalized_source == "csv":
        return _load_feature_dataset_from_csv(resolved_path), "csv"

    if normalized_source == "postgres":
        try:
            return _load_feature_dataset_from_postgres(), "postgres"
        except (RuntimeError, SQLAlchemyError, ValueError) as exc:
            raise RuntimeError(
                "Failed to load gold.features_dataset from PostgreSQL."
            ) from exc

    try:
        return _load_feature_dataset_from_postgres(), "postgres"
    except (RuntimeError, SQLAlchemyError, ValueError) as exc:
        logger.warning(
            "Falling back to CSV gold dataset after PostgreSQL feature load failed: %s",
            exc,
        )
        return _load_feature_dataset_from_csv(resolved_path), "csv"


def load_feature_dataset(
    dataset_path: Path | None = None,
    source: str = "csv",
) -> pd.DataFrame:
    """Load the model-ready gold dataset."""
    df, _ = load_feature_dataset_with_source(dataset_path=dataset_path, source=source)
    return df


def select_model_feature_columns(df: pd.DataFrame) -> list[str]:
    """Select numeric model features while excluding leakage and identifiers."""
    return [
        column
        for column in df.columns
        if column not in NON_FEATURE_COLUMNS
        and pd.api.types.is_numeric_dtype(df[column])
    ]


def build_tournament_flags(tournament: str | None) -> dict[str, int]:
    """Create the same tournament flags used during feature engineering."""
    tournament_name = (tournament or "").strip().lower()
    return {
        "is_friendly": int("friendly" in tournament_name),
        "is_world_cup": int(
            any(token in tournament_name for token in ("world cup", "wc", "fifa"))
        ),
        "is_qualifier": int(
            any(
                token in tournament_name
                for token in (
                    "qualification",
                    "qualifier",
                    "wcq",
                    "ecq",
                    "copaaq",
                    "acnq",
                    "afcq",
                    "ofcq",
                    "uefaq",
                    "conmebolq",
                )
            )
        ),
        "is_continental": int(
            any(
                token in tournament_name
                for token in (
                    "championship",
                    "euro",
                    "copa",
                    "africa",
                    "asian",
                    "confederation",
                )
            )
        ),
    }


def _safe_value(row: pd.Series | None, column: str) -> float:
    if row is None:
        return float("nan")
    value = row.get(column, np.nan)
    return float(value) if pd.notna(value) else float("nan")


def _coalesce(*values: float) -> float:
    for value in values:
        if pd.notna(value):
            return float(value)
    return float("nan")


def _resolve_team_name(df: pd.DataFrame, team_name: str) -> str:
    normalized = team_name.strip().casefold()
    team_map = {}

    for column in ("homeTeam", "awayTeam"):
        for current_team in df[column].dropna().unique():
            team_map[current_team.casefold()] = current_team

    if normalized not in team_map:
        raise ValueError(f"Team '{team_name}' was not found in the gold feature dataset.")

    return team_map[normalized]


def _latest_row(df: pd.DataFrame, mask: pd.Series) -> pd.Series | None:
    if not mask.any():
        return None
    return df.loc[mask].sort_values("date").iloc[-1]


def _extract_overall_snapshot(row: pd.Series, team_name: str) -> dict[str, float | pd.Timestamp]:
    if row["homeTeam"] == team_name:
        return {
            "elo": _safe_value(row, "elo_home"),
            "global_avg_goals_last5": _safe_value(row, "home_global_avg_goals_last5"),
            "global_avg_conceded_last5": _safe_value(
                row, "home_global_avg_conceded_last5"
            ),
            "global_win_rate_last5": _safe_value(row, "home_global_win_rate_last5"),
            "date": row["date"],
        }

    return {
        "elo": _safe_value(row, "elo_away"),
        "global_avg_goals_last5": _safe_value(row, "away_global_avg_goals_last5"),
        "global_avg_conceded_last5": _safe_value(
            row, "away_global_avg_conceded_last5"
        ),
        "global_win_rate_last5": _safe_value(row, "away_global_win_rate_last5"),
        "date": row["date"],
    }


def _build_team_context(df: pd.DataFrame, team_name: str) -> dict[str, object]:
    overall_row = _latest_row(
        df, (df["homeTeam"] == team_name) | (df["awayTeam"] == team_name)
    )
    if overall_row is None:
        raise ValueError(f"Team '{team_name}' was not found in the gold feature dataset.")

    snapshot = _extract_overall_snapshot(overall_row, team_name)
    return {
        "overall": overall_row,
        "home": _latest_row(df, df["homeTeam"] == team_name),
        "away": _latest_row(df, df["awayTeam"] == team_name),
        "elo": snapshot["elo"],
        "global_avg_goals_last5": snapshot["global_avg_goals_last5"],
        "global_avg_conceded_last5": snapshot["global_avg_conceded_last5"],
        "global_win_rate_last5": snapshot["global_win_rate_last5"],
        "snapshot_date": snapshot["date"],
    }


def build_match_feature_frame(
    home_team: str,
    away_team: str,
    tournament: str | None,
    neutral: bool,
    feature_columns: list[str],
    feature_history_df: pd.DataFrame | None = None,
    match_date: date | None = None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Build a model-ready feature row for an upcoming match.

    The assembly uses the latest available team snapshots in the gold dataset:
    - role-specific features from the team's latest match in that role
    - global features from the team's latest match overall
    - match-level interaction features recomputed for the requested fixture
    """
    if home_team.strip().casefold() == away_team.strip().casefold():
        raise ValueError("Home team and away team must be different teams.")

    df = feature_history_df if feature_history_df is not None else load_feature_dataset()
    if match_date is not None:
        requested_timestamp = pd.Timestamp(match_date)
        df = df.loc[df["date"] <= requested_timestamp].copy()
        if df.empty:
            raise ValueError(
                f"No historical feature data is available on or before '{match_date.isoformat()}'."
            )

    resolved_home_team = _resolve_team_name(df, home_team)
    resolved_away_team = _resolve_team_name(df, away_team)

    home_context = _build_team_context(df, resolved_home_team)
    away_context = _build_team_context(df, resolved_away_team)

    home_home_row = home_context["home"]
    away_away_row = away_context["away"]
    home_elo = float(home_context["elo"])
    away_elo = float(away_context["elo"])
    flags = build_tournament_flags(tournament)

    home_win_rate = _coalesce(
        _safe_value(home_home_row, "home_win_rate_last5"),
        float(home_context["global_win_rate_last5"]),
    )
    away_win_rate = _coalesce(
        _safe_value(away_away_row, "away_win_rate_last5"),
        float(away_context["global_win_rate_last5"]),
    )

    row = {
        "neutral": bool(neutral),
        "elo_home": home_elo,
        "elo_away": away_elo,
        "elo_diff": home_elo - away_elo,
        "home_avg_goals_last5": _safe_value(home_home_row, "home_avg_goals_last5"),
        "home_avg_goals_conceded_last5": _safe_value(
            home_home_row, "home_avg_goals_conceded_last5"
        ),
        "away_avg_goals_last5": _safe_value(away_away_row, "away_avg_goals_last5"),
        "away_avg_goals_conceded_last5": _safe_value(
            away_away_row, "away_avg_goals_conceded_last5"
        ),
        "home_global_avg_goals_last5": float(home_context["global_avg_goals_last5"]),
        "home_global_avg_conceded_last5": float(
            home_context["global_avg_conceded_last5"]
        ),
        "away_global_avg_goals_last5": float(away_context["global_avg_goals_last5"]),
        "away_global_avg_conceded_last5": float(
            away_context["global_avg_conceded_last5"]
        ),
        "home_win_rate_last5": _safe_value(home_home_row, "home_win_rate_last5"),
        "away_win_rate_last5": _safe_value(away_away_row, "away_win_rate_last5"),
        "home_global_win_rate_last5": float(home_context["global_win_rate_last5"]),
        "away_global_win_rate_last5": float(away_context["global_win_rate_last5"]),
        "home_advantage_effect": 0.0 if neutral else home_win_rate - away_win_rate,
        "home_opponent_elo": away_elo,
        "away_opponent_elo": home_elo,
        "home_avg_opponent_elo_last5": _safe_value(
            home_home_row, "home_avg_opponent_elo_last5"
        ),
        "away_avg_opponent_elo_last5": _safe_value(
            away_away_row, "away_avg_opponent_elo_last5"
        ),
        "home_weighted_win_rate_last5": _safe_value(
            home_home_row, "home_weighted_win_rate_last5"
        ),
        "away_weighted_win_rate_last5": _safe_value(
            away_away_row, "away_weighted_win_rate_last5"
        ),
        "elo_ratio_home": home_elo / max(away_elo, 1e-6),
        "combined_elo_strength": home_elo + away_elo,
        "home_opponent_elo_form": _safe_value(home_home_row, "home_opponent_elo_form"),
        "away_opponent_elo_form": _safe_value(away_away_row, "away_opponent_elo_form"),
        "home_elo_form": _safe_value(home_home_row, "home_elo_form"),
        "away_elo_form": _safe_value(away_away_row, "away_elo_form"),
        **flags,
    }

    feature_row = {column: row.get(column, float("nan")) for column in feature_columns}
    feature_frame = pd.DataFrame([feature_row])
    snapshot_dates = {
        "home_team": resolved_home_team,
        "away_team": resolved_away_team,
        "home_snapshot_date": pd.Timestamp(home_context["snapshot_date"]).date().isoformat(),
        "away_snapshot_date": pd.Timestamp(away_context["snapshot_date"]).date().isoformat(),
    }
    return feature_frame, snapshot_dates


def _resolve_snapshot_team_name(df: pd.DataFrame, team_name: str) -> str:
    normalized = team_name.strip().casefold()
    team_map = {
        current_team.casefold(): current_team
        for current_team in df["team"].dropna().unique()
    }
    if normalized not in team_map:
        raise ValueError(f"Team '{team_name}' was not found in the dbt serving model.")
    return team_map[normalized]


def _latest_snapshot_row(
    df: pd.DataFrame,
    team_name: str,
    team_role: str,
) -> pd.Series | None:
    mask = (df["team"] == team_name) & (df["team_role"] == team_role)
    if not mask.any():
        return None
    return df.loc[mask].sort_values("snapshot_date").iloc[-1]


def build_match_feature_frame_from_team_snapshots(
    home_team: str,
    away_team: str,
    tournament: str | None,
    neutral: bool,
    feature_columns: list[str],
    team_snapshots_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Build a feature row from a dbt-curated team snapshot serving model."""
    if home_team.strip().casefold() == away_team.strip().casefold():
        raise ValueError("Home team and away team must be different teams.")

    resolved_home_team = _resolve_snapshot_team_name(team_snapshots_df, home_team)
    resolved_away_team = _resolve_snapshot_team_name(team_snapshots_df, away_team)

    home_home_row = _latest_snapshot_row(
        team_snapshots_df,
        resolved_home_team,
        "home",
    )
    away_away_row = _latest_snapshot_row(
        team_snapshots_df,
        resolved_away_team,
        "away",
    )
    home_overall_row = _latest_snapshot_row(
        team_snapshots_df,
        resolved_home_team,
        "overall",
    )
    away_overall_row = _latest_snapshot_row(
        team_snapshots_df,
        resolved_away_team,
        "overall",
    )

    if home_overall_row is None or away_overall_row is None:
        raise ValueError(
            "The dbt latest team snapshot model is missing an overall snapshot for one of the teams."
        )

    home_elo = _safe_value(home_overall_row, "elo")
    away_elo = _safe_value(away_overall_row, "elo")
    flags = build_tournament_flags(tournament)

    home_win_rate = _coalesce(
        _safe_value(home_home_row, "win_rate_last5"),
        _safe_value(home_overall_row, "global_win_rate_last5"),
    )
    away_win_rate = _coalesce(
        _safe_value(away_away_row, "win_rate_last5"),
        _safe_value(away_overall_row, "global_win_rate_last5"),
    )

    row = {
        "neutral": bool(neutral),
        "elo_home": home_elo,
        "elo_away": away_elo,
        "elo_diff": home_elo - away_elo,
        "home_avg_goals_last5": _safe_value(home_home_row, "avg_goals_last5"),
        "home_avg_goals_conceded_last5": _safe_value(
            home_home_row,
            "avg_goals_conceded_last5",
        ),
        "away_avg_goals_last5": _safe_value(away_away_row, "avg_goals_last5"),
        "away_avg_goals_conceded_last5": _safe_value(
            away_away_row,
            "avg_goals_conceded_last5",
        ),
        "home_global_avg_goals_last5": _safe_value(
            home_overall_row,
            "global_avg_goals_last5",
        ),
        "home_global_avg_conceded_last5": _safe_value(
            home_overall_row,
            "global_avg_conceded_last5",
        ),
        "away_global_avg_goals_last5": _safe_value(
            away_overall_row,
            "global_avg_goals_last5",
        ),
        "away_global_avg_conceded_last5": _safe_value(
            away_overall_row,
            "global_avg_conceded_last5",
        ),
        "home_win_rate_last5": _safe_value(home_home_row, "win_rate_last5"),
        "away_win_rate_last5": _safe_value(away_away_row, "win_rate_last5"),
        "home_global_win_rate_last5": _safe_value(
            home_overall_row,
            "global_win_rate_last5",
        ),
        "away_global_win_rate_last5": _safe_value(
            away_overall_row,
            "global_win_rate_last5",
        ),
        "home_advantage_effect": 0.0 if neutral else home_win_rate - away_win_rate,
        "home_opponent_elo": away_elo,
        "away_opponent_elo": home_elo,
        "home_avg_opponent_elo_last5": _safe_value(
            home_home_row,
            "avg_opponent_elo_last5",
        ),
        "away_avg_opponent_elo_last5": _safe_value(
            away_away_row,
            "avg_opponent_elo_last5",
        ),
        "home_weighted_win_rate_last5": _safe_value(
            home_home_row,
            "weighted_win_rate_last5",
        ),
        "away_weighted_win_rate_last5": _safe_value(
            away_away_row,
            "weighted_win_rate_last5",
        ),
        "elo_ratio_home": home_elo / max(away_elo, 1e-6),
        "combined_elo_strength": home_elo + away_elo,
        "home_opponent_elo_form": _safe_value(home_home_row, "opponent_elo_form"),
        "away_opponent_elo_form": _safe_value(away_away_row, "opponent_elo_form"),
        "home_elo_form": _safe_value(home_home_row, "elo_form"),
        "away_elo_form": _safe_value(away_away_row, "elo_form"),
        **flags,
    }

    feature_row = {column: row.get(column, float("nan")) for column in feature_columns}
    feature_frame = pd.DataFrame([feature_row])
    snapshot_dates = {
        "home_team": resolved_home_team,
        "away_team": resolved_away_team,
        "home_snapshot_date": pd.Timestamp(
            home_overall_row["snapshot_date"]
        ).date().isoformat(),
        "away_snapshot_date": pd.Timestamp(
            away_overall_row["snapshot_date"]
        ).date().isoformat(),
    }
    return feature_frame, snapshot_dates


def build_match_feature_frame_from_latest_snapshots(
    home_team: str,
    away_team: str,
    tournament: str | None,
    neutral: bool,
    feature_columns: list[str],
    latest_team_snapshots_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Backward-compatible wrapper for latest-snapshot serving."""
    return build_match_feature_frame_from_team_snapshots(
        home_team=home_team,
        away_team=away_team,
        tournament=tournament,
        neutral=neutral,
        feature_columns=feature_columns,
        team_snapshots_df=latest_team_snapshots_df,
    )
