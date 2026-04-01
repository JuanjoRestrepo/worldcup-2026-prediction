from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database.connection import get_connection


def test_connection():
    """Basic test to verify database connectivity."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT version();")
    version = cursor.fetchone()

    print("Connected to:", version)

    cursor.close()
    conn.close()


if __name__ == "__main__":
    test_connection()
