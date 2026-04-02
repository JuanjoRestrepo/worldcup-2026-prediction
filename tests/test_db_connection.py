from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging
from src.database.connection import get_connection


def test_connection():
    """
    Automated test (pytest) — validates DB connectivity.
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT 1;")
        result = cursor.fetchone()

        assert result is not None, "Query returned no results"
        assert result[0] == 1
    finally:
        conn.close()


def debug_connection():
    """
    Manual debug function — prints DB version.
    Useful for local troubleshooting.
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()

        print("Connected to:", version)
    finally:
        conn.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    debug_connection()