#!/usr/bin/env python
"""Orchestrate the end-to-end World Cup prediction pipeline."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from time import perf_counter

from src.config.settings import settings
from src.ingestion.pipelines.ingestion_pipeline import run_ingestion_pipeline
from src.modeling.train import train_and_export_model
from src.processing.pipelines.processing_pipeline import run_processing_pipeline

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full World Cup prediction pipeline: ingestion, processing, "
            "and model training."
        )
    )
    parser.add_argument(
        "--skip-ingestion",
        action="store_true",
        help="Skip the ingestion stage and reuse existing raw/bronze data.",
    )
    parser.add_argument(
        "--skip-processing",
        action="store_true",
        help="Skip the processing stage and reuse the existing gold dataset.",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip the training stage and do not refresh the model artifact.",
    )
    parser.add_argument(
        "--no-api-data",
        action="store_true",
        help="Process historical CSV data only, without loading API snapshots.",
    )
    parser.add_argument(
        "--persist-to-db",
        action="store_true",
        default=settings.PERSIST_TO_DB,
        help="Persist bronze/silver/gold outputs and training metadata to PostgreSQL.",
    )
    parser.add_argument(
        "--artifact-path",
        type=Path,
        default=settings.MODEL_ARTIFACT_PATH,
        help="Override the output path for the exported model artifact.",
    )
    parser.add_argument(
        "--gold-data-path",
        type=Path,
        default=settings.GOLD_DIR / "features_dataset.csv",
        help="Path to the gold feature dataset used for training.",
    )
    parser.add_argument(
        "--version-tag",
        type=str,
        default=None,
        help=(
            "Semantic version tag for a named model snapshot written alongside "
            "the production artifact (e.g. 'v2_apr2026' produces "
            "'match_predictor_v2_apr2026.joblib'). Optional."
        ),
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of the most recent gold dataset reserved for evaluation.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity for the orchestration script.",
    )
    return parser.parse_args()


def run_full_pipeline(
    *,
    run_ingestion: bool = True,
    run_processing: bool = True,
    run_training: bool = True,
    use_api_data: bool = True,
    persist_to_db: bool = False,
    artifact_path: Path | None = None,
    gold_data_path: Path | None = None,
    test_size: float = 0.2,
    version_tag: str | None = None,
) -> dict[str, Any]:
    """
    Execute the full pipeline with optional stage skipping.

    Returns:
        Dictionary summarizing executed stages, output paths, and timings.
    """
    settings.ensure_project_dirs()
    pipeline_run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    summary: dict[str, Any] = {
        "pipeline_run_id": pipeline_run_id,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "stages": {},
        "artifacts": {
            "raw_dir": str(settings.RAW_DIR),
            "bronze_dir": str(settings.BRONZE_DIR),
            "silver_path": str(settings.SILVER_DIR / "matches_cleaned.csv"),
            "gold_path": str(
                gold_data_path or settings.GOLD_DIR / "features_dataset.csv"
            ),
            "model_artifact_path": str(artifact_path or settings.MODEL_ARTIFACT_PATH),
        },
    }

    logger.info("=" * 72)
    logger.info("STARTING END-TO-END WORLD CUP PIPELINE")
    logger.info("=" * 72)
    logger.info(
        "Configuration | ingestion=%s processing=%s training=%s api_data=%s",
        run_ingestion,
        run_processing,
        run_training,
        use_api_data,
    )
    logger.info("Database persistence enabled: %s", persist_to_db)

    if run_ingestion:
        start = perf_counter()
        logger.info("[Stage 1/3] Running ingestion pipeline...")
        run_ingestion_pipeline(
            persist_to_db=persist_to_db,
            pipeline_run_id=pipeline_run_id,
        )
        summary["stages"]["ingestion"] = {
            "status": "completed",
            "duration_seconds": round(perf_counter() - start, 2),
            "persisted_to_db": persist_to_db,
        }
    else:
        logger.info("[Stage 1/3] Skipping ingestion pipeline")
        summary["stages"]["ingestion"] = {"status": "skipped"}

    if run_processing:
        start = perf_counter()
        logger.info("[Stage 2/3] Running processing pipeline...")
        processed_df = run_processing_pipeline(
            use_api_data=use_api_data,
            persist_to_db=persist_to_db,
            pipeline_run_id=pipeline_run_id,
        )
        summary["stages"]["processing"] = {
            "status": "completed",
            "duration_seconds": round(perf_counter() - start, 2),
            "rows": int(len(processed_df)),
            "columns": int(len(processed_df.columns)),
            "persisted_to_db": persist_to_db,
        }
    else:
        logger.info("[Stage 2/3] Skipping processing pipeline")
        summary["stages"]["processing"] = {"status": "skipped"}

    if run_training:
        start = perf_counter()
        logger.info("[Stage 3/3] Running training pipeline...")
        training_summary = train_and_export_model(
            data_path=gold_data_path,
            artifact_path=artifact_path,
            test_size=test_size,
            persist_to_db=persist_to_db,
            pipeline_run_id=pipeline_run_id,
            version_tag=version_tag,
        )
        summary["stages"]["training"] = {
            "status": "completed",
            "duration_seconds": round(perf_counter() - start, 2),
            "metrics": {
                "accuracy": training_summary["metrics"]["accuracy"],
                "macro_f1": training_summary["metrics"]["macro_f1"],
                "weighted_f1": training_summary["metrics"]["weighted_f1"],
                "log_loss": training_summary["metrics"]["log_loss"],
            },
            "training_rows": training_summary["training_rows"],
            "test_rows": training_summary["test_rows"],
            "feature_count": training_summary["feature_count"],
            "persisted_to_db": persist_to_db,
        }
    else:
        logger.info("[Stage 3/3] Skipping training pipeline")
        summary["stages"]["training"] = {"status": "skipped"}

    summary["finished_at_utc"] = datetime.now(UTC).isoformat()
    logger.info("=" * 72)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 72)
    return summary


if __name__ == "__main__":
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    pipeline_summary = run_full_pipeline(
        run_ingestion=not args.skip_ingestion,
        run_processing=not args.skip_processing,
        run_training=not args.skip_training,
        use_api_data=not args.no_api_data,
        persist_to_db=args.persist_to_db,
        artifact_path=args.artifact_path,
        gold_data_path=args.gold_data_path,
        test_size=args.test_size,
        version_tag=args.version_tag,
    )
    print(json.dumps(pipeline_summary, indent=2))
