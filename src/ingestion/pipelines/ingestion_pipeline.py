import json
import logging
from datetime import datetime, timedelta

import pandas as pd

from src.config.settings import settings
from src.contracts.data_contracts import validate_standardized_matches_contract
from src.database.persistence import persist_dataframe
from src.ingestion.clients.api_client import FootballAPIClient
from src.ingestion.clients.csv_client import load_historical_data
from src.ingestion.utils.international_validator import filter_international_matches
from src.ingestion.utils.validator import validate_schema
from src.processing.transformers.match_standardizer import (
    standardize_api,
    standardize_csv,
)

logger = logging.getLogger(__name__)


def run_ingestion_pipeline(
    persist_to_db: bool = False,
    pipeline_run_id: str | None = None,
) -> None:
    logger.info("Starting ingestion pipeline")
    logger.info(
        "NOTE: This pipeline ONLY ingests international/national team data (NO club leagues)"
    )
    settings.ensure_project_dirs()

    # 1. CSV (histórico)
    logger.info("\n📊 PHASE 1: Loading historical international results from CSV...")
    df = load_historical_data()
    validate_schema(df)

    df_csv_standardized = standardize_csv(df)
    validate_standardized_matches_contract(
        df_csv_standardized,
        contract_name="bronze_historical_matches",
    )
    csv_output_path = settings.BRONZE_DIR / "historical_standardized.csv"
    df_csv_standardized.to_csv(csv_output_path, index=False)
    logger.info(f"✅ Saved standardized historical data → {csv_output_path}")
    if persist_to_db:
        persist_dataframe(
            df_csv_standardized,
            schema_name="bronze",
            table_name="historical_matches",
            if_exists="replace",
            pipeline_run_id=pipeline_run_id,
        )
        logger.info("✅ Persisted bronze.historical_matches to PostgreSQL")

    # 2. API - Fetch recent international data
    logger.info("\n🌐 PHASE 2: Fetching recent international matches from API...")
    api_key = settings.FOOTBALL_API_KEY

    # Get the last 90 days of matches (wider window to find international matches)
    # International windows are sparse outside tournaments
    today = datetime.now()
    ninety_days_ago = (today - timedelta(days=90)).strftime("%Y-%m-%d")
    today_str = today.strftime("%Y-%m-%d")

    api_data = None
    df_api_standardized = pd.DataFrame(columns=df_csv_standardized.columns)
    if api_key:
        try:
            logger.info(
                f"Fetching recent matches from API (last 90 days: {ninety_days_ago} to {today_str})"
            )
            client = FootballAPIClient(api_key=api_key)
            api_data = client.get_matches(
                date_from=ninety_days_ago,
                date_to=today_str,
            )

            # 🚨 FILTER: Keep ONLY international competitions
            logger.info(
                "\n🔍 FILTERING: Removing club league data, keeping ONLY national teams..."
            )
            if api_data and "matches" in api_data:
                total_before = len(api_data["matches"])
                international_matches = filter_international_matches(
                    api_data["matches"]
                )
                api_data["matches"] = international_matches
                df_api_standardized = standardize_api(international_matches)
                if not df_api_standardized.empty:
                    initial_api_rows = len(df_api_standardized)
                    df_api_standardized = df_api_standardized.dropna(
                        subset=["homeGoals", "awayGoals"]
                    ).reset_index(drop=True)
                    skipped_unscored_rows = initial_api_rows - len(df_api_standardized)
                    if skipped_unscored_rows > 0:
                        logger.info(
                            "Skipped %s API matches without final scores before bronze validation",
                            skipped_unscored_rows,
                        )
                    validate_standardized_matches_contract(
                        df_api_standardized,
                        contract_name="bronze_api_matches",
                    )
                logger.info(
                    f"✅ API Data filtered: {total_before} → {len(international_matches)} matches (removed {total_before - len(international_matches)} club league matches)"
                )

            if api_data and api_data.get("matches"):
                api_output_path = settings.BRONZE_DIR / "api_standardized.csv"
                df_api_standardized.to_csv(api_output_path, index=False)
                logger.info(f"✅ Saved standardized API data → {api_output_path}")
            else:
                logger.warning("⚠️  No international matches found in API response")
                api_data = None
        except Exception as e:
            logger.warning(
                f"⚠️  Could not fetch data from API (free tier has request limits): {e}"
            )
            logger.info("Continuing with historical data only")
            api_data = None
    else:
        logger.warning("⚠️  FOOTBALL_API_KEY not set. Using historical data only")

    if persist_to_db:
        persist_dataframe(
            df_api_standardized,
            schema_name="bronze",
            table_name="api_matches",
            if_exists="replace",
            pipeline_run_id=pipeline_run_id,
        )
        logger.info("✅ Persisted bronze.api_matches to PostgreSQL")

    # 3. Save cleaned API data
    logger.info("\n💾 PHASE 3: Saving processed data...")
    if api_data:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = settings.RAW_DIR / f"api_international_matches_{timestamp}.json"

        with open(output_path, "w") as f:
            json.dump(api_data, f, indent=2)

        logger.info(f"✅ Saved cleaned API data (international only) → {output_path}")
        logger.info(
            f"   📈 Total international matches saved: {len(api_data.get('matches', []))}"
        )

    logger.info("\n" + "=" * 60)
    logger.info("✅ INGESTION PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)
    logger.info("\n📋 SUMMARY:")
    logger.info(f"  ✓ Historical data: {len(df)} international records")
    if api_data:
        logger.info(
            f"  ✓ API data: {len(api_data.get('matches', []))} recent international matches"
        )
        logger.info(f"  ✓ Data coverage: {ninety_days_ago} to {today_str}")
    logger.info("  ✓ Format: INTERNATIONAL MATCHES ONLY (no club leagues)")
    logger.info(
        "\n🎯 Next step: Data processing & feature engineering for WC 2026 prediction"
    )
