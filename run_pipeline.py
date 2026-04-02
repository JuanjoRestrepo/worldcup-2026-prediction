#!/usr/bin/env python
"""Run the processing pipeline."""

from src.processing.pipelines.processing_pipeline import run_processing_pipeline
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)

if __name__ == "__main__":
    run_processing_pipeline()
    print("\n✅ Pipeline execution complete!")
