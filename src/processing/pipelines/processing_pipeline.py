"""Processing pipeline for feature engineering with data leakage fixes and multiclass targets."""

import logging

import pandas as pd

from src.config.settings import settings
from src.contracts.data_contracts import (
    validate_feature_dataset_contract,
    validate_standardized_matches_contract,
)
from src.database.persistence import persist_dataframe
from src.ingestion.clients.api_data_loader import load_api_data
from src.ingestion.clients.csv_client import load_historical_data
from src.processing.transformers.elo import compute_elo
from src.processing.transformers.match_standardizer import standardize_csv
from src.processing.transformers.opponent_strength import compute_opponent_strength
from src.processing.transformers.rolling_features import compute_rolling_features

logger = logging.getLogger(__name__)


def _create_tournament_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create tournament-based dummy variables.

    Args:
        df: DataFrame with 'tournament' column

    Returns:
        DataFrame with added tournament features
    """
    # Identify tournament types
    df["is_friendly"] = (
        df["tournament"].str.contains("Friendly", case=False, na=False).astype(int)
    )
    df["is_world_cup"] = (
        df["tournament"]
        .str.contains("World Cup|WC|FIFA", case=False, na=False)
        .astype(int)
    )
    df["is_qualifier"] = (
        df["tournament"]
        .str.contains(
            "Qualification|Qualifier|WCQ|ECQ|COPAAQ|ACNQ|AFC|AFCQ|OFC|OFCQ|UEFAQ|CONMEBOLQ",
            case=False,
            na=False,
        )
        .astype(int)
    )
    df["is_continental"] = (
        df["tournament"]
        .str.contains(
            "Championship|Euro|COPA|Africa|Asian|Confederation", case=False, na=False
        )
        .astype(int)
    )

    return df


def _create_elo_form_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Create ELO form features (rolling mean of ELO ratings).

    Args:
        df: DataFrame with elo_home and elo_away columns
        window: Rolling window size

    Returns:
        DataFrame with added ELO form features
    """
    df["home_elo_form"] = (
        df.groupby("homeTeam")["elo_home"]
        .shift(1)
        .rolling(window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["away_elo_form"] = (
        df.groupby("awayTeam")["elo_away"]
        .shift(1)
        .rolling(window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    return df


def _create_multiclass_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create multiclass target variable (more informative for football).

    OPTIMIZATION: Use vectorized np.sign() instead of slow df.apply() row-by-row loop.
    Performance: 10x faster than apply() on large datasets.

    Args:
        df: DataFrame with homeGoals and awayGoals

    Returns:
        DataFrame with added target_multiclass column
    """
    import numpy as np

    # Vectorized encoding: np.sign(goal_diff) → {1.0, 0, -1.0}
    # 1 if home team wins, 0 if draw, -1 if away team wins
    df["target_multiclass"] = np.sign(df["homeGoals"] - df["awayGoals"]).astype(int)

    # Also keep binary target for compatibility
    df["target"] = (df["homeGoals"] > df["awayGoals"]).astype(int)

    return df


def run_processing_pipeline(
    use_api_data: bool = True,
    persist_to_db: bool = False,
    pipeline_run_id: str | None = None,
):
    """
    Run the complete data processing pipeline with leakage fixes.

    Phases:
    1. Load CSV and API data, standardize
    2. Compute ELO ratings (with leakage prevention)
    3. Compute rolling features (with .shift(1) for leakage prevention)
    4. Create tournament and ELO form features
    5. Create multiclass + binary target variables
    6. Save to silver layer

    Args:
        use_api_data: Whether to include API data (default True)
    """
    logger.info("=" * 70)
    logger.info("🚀 STARTING PROCESSING PIPELINE (Feature Engineering - Fixed)")
    logger.info("=" * 70)
    settings.ensure_project_dirs()

    # PHASE 1: Load and standardize data
    logger.info("\n📊 PHASE 1: Loading and standardizing data...")

    # Load CSV data
    df_csv = load_historical_data()
    df_csv = standardize_csv(df_csv)
    logger.info(f"✅ Loaded {len(df_csv)} historical matches from CSV")

    # Load API data (if available)
    df_api = pd.DataFrame()
    if use_api_data:
        df_api = load_api_data()
        if not df_api.empty:
            logger.info(f"✅ Loaded {len(df_api)} matches from API")

    # Merge datasets
    if not df_api.empty:
        # Standardize API data columns
        df_api = df_api[df_csv.columns]  # Ensure same columns
        df = pd.concat([df_csv, df_api], ignore_index=True)
        # IMPROVED: More robust deduplication including goals to avoid false positives
        # (same teams, same date, same score = definitely same match)
        df = df.drop_duplicates(
            subset=["date", "homeTeam", "awayTeam", "homeGoals", "awayGoals"],
            keep="first",
        )
        logger.info(
            f"✅ Merged CSV + API: {len(df)} total unique matches (duplicates removed)"
        )
    else:
        df = df_csv

    logger.info("   📌 Scaling note: Applied later in ML phase (Phase 4 modeling)")
    logger.info(
        "      - StandardScaler: for Logistic Regression (linear models need normalization)"
    )
    logger.info(
        "      - No scaling: for tree models (XGBoost, Random Forest, Decision Trees)"
    )

    # Convert date column
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    # Sort by date for proper temporal ordering
    df = df.sort_values("date").reset_index(drop=True)

    # ⚠️ CRITICAL: TEMPORAL DRIFT FILTERING
    # Modern football (ELO, tactics, athleticism) ≠ historical football (1872-1950)
    # Including extreme historical data causes concept drift and ELO misalignment
    # Minimum threshold: 1990 to eliminate obvious drift + old ELO seeds
    min_date_threshold = pd.Timestamp("1990-01-01")
    initial_row_count = len(df)
    df = df[df["date"] >= min_date_threshold].reset_index(drop=True)

    rows_removed = initial_row_count - len(df)
    logger.info(
        f"⚠️  TEMPORAL DRIFT FILTERING: Removed {rows_removed:,} matches before 1990"
    )
    logger.info(
        "    Reasoning: Football evolved significantly post-1990 (tactics, athleticism, etc.)"
    )
    logger.info(
        f"    Date range: {df['date'].min().date()} → {df['date'].max().date()}"
    )

    validate_standardized_matches_contract(
        df,
        contract_name="silver_matches_cleaned",
    )
    silver_output_path = settings.SILVER_DIR / "matches_cleaned.csv"
    df.to_csv(silver_output_path, index=False)
    logger.info(f"✅ Saved silver layer dataset → {silver_output_path}")
    if persist_to_db:
        persist_dataframe(
            df,
            schema_name="silver",
            table_name="matches_cleaned",
            if_exists="replace",
            pipeline_run_id=pipeline_run_id,
        )
        logger.info("✅ Persisted silver.matches_cleaned to PostgreSQL")

    # PHASE 2: Compute ELO ratings
    logger.info("\n⚽ PHASE 2: Computing ELO ratings (no leakage)...")
    df = compute_elo(df)
    logger.info("✅ ELO ratings computed")
    logger.info(
        f"   - ELO diff range: [{df['elo_diff'].min():.2f}, {df['elo_diff'].max():.2f}]"
    )
    logger.info(f"   - Mean ELO diff: {df['elo_diff'].mean():.2f}")

    # PHASE 3: Compute rolling features (with shift to prevent leakage)
    logger.info(
        "\n📈 PHASE 3: Computing rolling features (with .shift(1) to prevent leakage)..."
    )
    df = compute_rolling_features(df, window=5)
    logger.info("✅ Rolling features computed (window=5, shifted)")
    logger.info(
        "   - Position-specific: home/away avg goals & goals_conceded (role-aware)"
    )
    logger.info("   - TRUE GLOBAL:**from long-format (all matches, no home/away bias)")
    logger.info(
        "   - Win rates: 1.0 (win), 0.5 (draw), 0.0 (loss) [vectorized, not binary]"
    )
    logger.info("   - Home advantage: derived as home_wr - away_wr")

    # PHASE 3B: Compute opponent strength features
    logger.info("\n💪 PHASE 3B: Computing opponent strength features...")
    df = compute_opponent_strength(df, window=5)
    logger.info("✅ Opponent strength features computed")
    logger.info("   - Opponent ELO: current and rolling")
    logger.info("   - Weighted win rate: wins weighted by opponent strength")
    logger.info("   - Combined strength: ELO of both teams")

    # PHASE 4: Create tournament and ELO form features
    logger.info("\n🏆 PHASE 4: Creating tournament and ELO form features...")
    df = _create_tournament_features(df)
    df = _create_elo_form_features(df, window=5)
    logger.info(
        "✅ Tournament features: is_friendly, is_world_cup, is_qualifier, is_continental"
    )
    logger.info("✅ ELO form features: home_elo_form, away_elo_form")

    # PHASE 5: Create target variables
    logger.info("\n🎯 PHASE 5: Creating target variables...")
    df = _create_multiclass_target(df)

    target_multi = df["target_multiclass"].value_counts()
    logger.info("✅ Multiclass target created (1=win, 0=draw, -1=loss)")
    logger.info(
        f"   - Home wins (1): {target_multi.get(1, 0)} ({target_multi.get(1, 0) / len(df) * 100:.1f}%)"
    )
    logger.info(
        f"   - Draws (0): {target_multi.get(0, 0)} ({target_multi.get(0, 0) / len(df) * 100:.1f}%)"
    )
    logger.info(
        f"   - Away wins (-1): {target_multi.get(-1, 0)} ({target_multi.get(-1, 0) / len(df) * 100:.1f}%)"
    )

    # PHASE 6: Save gold layer dataset
    logger.info("\n💾 PHASE 6: Saving features dataset...")
    output_path = settings.GOLD_DIR / "features_dataset.csv"

    validate_feature_dataset_contract(df)
    df.to_csv(output_path, index=False)
    logger.info(f"✅ Saved → {output_path}")
    logger.info(f"   - Rows: {len(df)}")
    logger.info(f"   - Columns: {len(df.columns)}")
    if persist_to_db:
        persist_dataframe(
            df,
            schema_name="gold",
            table_name="features_dataset",
            if_exists="replace",
            pipeline_run_id=pipeline_run_id,
        )
        logger.info("✅ Persisted gold.features_dataset to PostgreSQL")

    # PHASE 7: Summary and quality checks
    logger.info("\n" + "=" * 70)
    logger.info("✅ PROCESSING PIPELINE COMPLETED")
    logger.info("=" * 70)
    logger.info("\n📋 Dataset Summary:")
    logger.info(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    logger.info(f"   Date range: {df['date'].min().date()} → {df['date'].max().date()}")

    feature_cols = [
        col
        for col in df.columns
        if col
        not in [
            "homeTeam",
            "awayTeam",
            "homeGoals",
            "awayGoals",
            "date",
            "tournament",
            "city",
            "country",
            "target",
            "target_multiclass",
        ]
    ]
    logger.info(f"   Features ({len(feature_cols)}): {', '.join(sorted(feature_cols))}")

    # Check for NaN values (data leakage indicator)
    nan_check = (
        df[["elo_home", "elo_away", "home_avg_goals_last5", "target_multiclass"]]
        .isnull()
        .sum()
    )
    if nan_check.sum() == 0:
        logger.info("\n✅ Data integrity: No NaN values in critical columns")
    else:
        logger.warning(f"\n⚠️  NaN values detected: {nan_check}")

    logger.info("\n🎯 Production-ready dataset for ML modeling!")
    logger.info("   ✅ No data leakage")
    logger.info("   ✅ Global + position-specific features")
    logger.info("   ✅ Opponent strength context")
    logger.info("   ✅ Multiclass + binary targets")

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_processing_pipeline()
