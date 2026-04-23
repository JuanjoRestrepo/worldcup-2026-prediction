#!/usr/bin/env python
"""Run the FastAPI server with the correct port from environment variable."""

import logging
import os
import sys

import uvicorn

from src.api.main import app
from src.database.connection import get_sqlalchemy_engine
from sqlalchemy import text

logger = logging.getLogger(__name__)


def ensure_dbt_tables_exist():
    """Check if dbt analytics tables exist. If not, run dbt and data pipelines."""
    try:
        logger.info("🔍 Checking if dbt analytics tables exist...")

        engine = get_sqlalchemy_engine()
        with engine.connect() as connection:
            result = connection.execute(
                text(
                    "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema='analytics_gold' AND table_name='gold_latest_team_snapshots')"
                )
            )
            table_exists = result.scalar()

        if table_exists:
            logger.info("✅ Analytics tables found. Skipping initialization.")
            return

        logger.warning(
            "⚠️  Analytics tables not found. Running initialization pipeline..."
        )

        logger.info("📂 Loading data...")
        try:
            import subprocess
            from pathlib import Path

            # Try load_data script if available
            if Path("load_data.py").exists():
                logger.info("Running load_data.py...")
                sub_result = subprocess.run(
                    [sys.executable, "load_data.py"],
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                if sub_result.returncode != 0:
                    logger.error(f"load_data.py failed: {sub_result.stderr}")
                else:
                    logger.info("✅ Data loaded successfully")

            # Run dbt
            if Path("run_dbt.py").exists():
                logger.info("Running dbt pipeline...")
                sub_result = subprocess.run(
                    [sys.executable, "run_dbt.py", "run"],
                    capture_output=True,
                    text=True,
                    timeout=600,
                )
                if sub_result.returncode != 0:
                    logger.error(f"dbt run failed: {sub_result.stderr}")
                else:
                    logger.info("✅ dbt pipeline completed")
        except Exception as e:
            logger.error(f"Initialization pipeline failed: {e}")
            # Don't crash the server - it can still serve predictions from CSV

    except Exception as e:
        logger.warning(f"Could not check dbt tables: {e}. Continuing anyway...")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    logger = logging.getLogger(__name__)

    # Ensure dbt tables exist before starting the server
    ensure_dbt_tables_exist()

    # Render uses PORT environment variable
    port = int(os.getenv("PORT", 10000))
    host = os.getenv("HOST", "0.0.0.0")

    logger.info(f"🚀 Starting FastAPI server on {host}:{port}")
    uvicorn.run(
        app,
        host=host,
        port=port,
        workers=1,  # Render free tier only supports 1 worker
    )
