"""Pytest configuration and fixtures."""
import sys
from pathlib import Path

import pytest
from sqlalchemy.engine import Engine

from src.database.connection import get_sqlalchemy_engine

# Add the project root to sys.path so that 'src' imports work
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def engine_fixture() -> Engine:
    """Provide a SQLAlchemy engine for database tests."""
    return get_sqlalchemy_engine()

