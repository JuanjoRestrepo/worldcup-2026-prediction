import psycopg2
from psycopg2.extensions import connection as _connection
from src.config.settings import settings


def get_connection() -> _connection:
    """
    Establish a connection to PostgreSQL database.

    Returns:
        psycopg2 connection object

    Raises:
        Exception: if connection fails
    """
    try:
        conn = psycopg2.connect(
            host=settings.DB_HOST,
            port=settings.DB_PORT,
            database=settings.DB_NAME,
            user=settings.DB_USER,
            password=settings.DB_PASSWORD,
        )
        return conn

    except Exception as e:
        raise Exception(f"Database connection failed: {e}")