import psycopg2
from psycopg2.extensions import connection as _connection
from sqlalchemy import create_engine
from sqlalchemy.engine import URL, Engine

from src.config.settings import settings


def get_connection() -> _connection:
    """
    Establish a psycopg2 connection to PostgreSQL.

    Returns:
        psycopg2 connection object

    Raises:
        RuntimeError: if connection fails
    """
    try:
        return psycopg2.connect(
            host=settings.DB_HOST,
            port=settings.DB_PORT,
            database=settings.DB_NAME,
            user=settings.DB_USER,
            password=settings.DB_PASSWORD,
            connect_timeout=5,
        )
    except psycopg2.OperationalError as exc:
        raise RuntimeError(f"[DB ERROR] Connection failed: {exc}")


def get_sqlalchemy_engine() -> Engine:
    """
    Build a SQLAlchemy engine for pandas-based persistence.

    Returns:
        SQLAlchemy Engine targeting the configured PostgreSQL database
    """
    database_url = URL.create(
        "postgresql+psycopg2",
        username=settings.DB_USER,
        password=settings.DB_PASSWORD,
        host=settings.DB_HOST,
        port=settings.DB_PORT,
        database=settings.DB_NAME,
    )
    return create_engine(database_url, future=True, pool_pre_ping=True)
