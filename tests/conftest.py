"""Pytest configuration and fixtures."""

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from backend.database.connection import get_sqlalchemy_engine

# Add the project root to sys.path so that 'backend' imports work
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def engine_fixture() -> Iterator[Engine]:
    """Provide a live SQLAlchemy engine or skip integration tests when DB is unavailable."""
    try:
        engine = get_sqlalchemy_engine()
        with engine.connect() as connection:
            connection.exec_driver_sql("SELECT 1")
    except (RuntimeError, SQLAlchemyError) as exc:
        if 'engine' in locals():
            engine.dispose()
        pytest.skip(f"PostgreSQL not available for integration test: {exc}")

    try:
        yield engine
    finally:
        engine.dispose()
