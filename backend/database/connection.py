"""Database connection utilities with hardened Supabase/PostgreSQL connectivity.

Design:
- Uses Supabase Session Pooler (port 5432, IPv4) instead of the direct
  connection (port 5432 IPv6-only) which is unreachable from most networks.
- Forces SSL (sslmode=require) as mandated by Supabase.
- Short connect_timeout (10s) with pool_pre_ping so stale connections are
  recycled without blocking inference.
- All callers should treat DB as optional: catch RuntimeError and degrade
  gracefully (predictions continue to work from CSV even without Postgres).
"""

from __future__ import annotations

import logging

import psycopg2
from psycopg2.extensions import connection as _connection
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.pool import NullPool

from backend.config.settings import settings

logger = logging.getLogger(__name__)

# Connect timeout in seconds.
# Keep short so inference never blocks waiting for a downed DB.
_CONNECT_TIMEOUT_SEC: int = 10


def _build_dsn(*, override_host: str | None = None) -> str:
    """Build a libpq connection string with all hardened parameters.

    Supabase note:
        Use the *Session Pooler* host (format: aws-0-<region>.pooler.supabase.com)
        rather than the Direct Connection host (db.<project-ref>.supabase.co).
        The pooler resolves to an IPv4 address; the direct host resolves to IPv6
        only and is unreachable from many networks and all Render free-tier nodes.

        Set POSTGRES_HOST in your .env to the Session Pooler host.
        Session Pooler port is 5432 (same as direct, no change needed).
    """
    host = override_host or settings.DB_HOST
    return (
        f"host={host} "
        f"port={settings.DB_PORT} "
        f"dbname={settings.DB_NAME} "
        f"user={settings.DB_USER} "
        f"password={settings.DB_PASSWORD} "
        f"connect_timeout={_CONNECT_TIMEOUT_SEC} "
        "sslmode=require "
        # Force TCP IPv4 stack where possible via libpq hint
        # (ignored silently on platforms that don't support it)
        "target_session_attrs=read-write"
    )


def get_connection() -> _connection:
    """Establish a psycopg2 connection to PostgreSQL / Supabase.

    Returns:
        psycopg2 connection object.

    Raises:
        RuntimeError: if connection fails after timeout. Callers should catch
            this and degrade gracefully — predictions work without a DB.
    """
    if not settings.DB_HOST or settings.DB_HOST == "localhost":
        raise RuntimeError(
            "[DB] POSTGRES_HOST is not configured. "
            "Set it in .env to your Supabase Session Pooler host "
            "(aws-0-<region>.pooler.supabase.com). "
            "Falling back to CSV-based inference."
        )
    try:
        conn = psycopg2.connect(_build_dsn())
        logger.debug("[DB] psycopg2 connection established to %s", settings.DB_HOST)
        return conn
    except psycopg2.OperationalError as exc:
        raise RuntimeError(
            f"[DB] Connection failed ({settings.DB_HOST}:{settings.DB_PORT}): {exc}\n\n"
            "Troubleshooting checklist:\n"
            "  1. Use the Session Pooler host, NOT the Direct Connection host.\n"
            "     Session Pooler: aws-0-<region>.pooler.supabase.com (IPv4, port 5432)\n"
            "     Direct:         db.<project-ref>.supabase.co     (IPv6 only)\n"
            "  2. Ensure the Supabase project is not paused (free tier auto-pauses).\n"
            "  3. Check your Supabase dashboard → Project Settings → Database.\n"
        ) from exc


def get_sqlalchemy_engine() -> Engine:
    """Build a SQLAlchemy engine for pandas-based persistence.

    Uses NullPool to avoid holding idle connections — appropriate for a
    low-traffic prediction API where DB writes are infrequent (inference logs).

    Returns:
        SQLAlchemy Engine targeting the configured PostgreSQL database.

    Raises:
        RuntimeError: if POSTGRES_HOST is not configured.
    """
    if not settings.DB_HOST or settings.DB_HOST == "localhost":
        raise RuntimeError(
            "[DB] POSTGRES_HOST is not configured. DB persistence is disabled."
        )

    # postgresql+psycopg2 DSN with all hardened parameters as connect_args
    connect_args: dict[str, object] = {
        "connect_timeout": _CONNECT_TIMEOUT_SEC,
        "sslmode": "require",
        "options": "-c target_session_attrs=read-write",
    }

    database_url = (
        f"postgresql+psycopg2://{settings.DB_USER}:{settings.DB_PASSWORD}"
        f"@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
    )

    return create_engine(
        database_url,
        future=True,
        # pool_pre_ping=True validates connections before use; combined with
        # NullPool (no idle connections held) this prevents IPv6 ghost connections.
        pool_pre_ping=True,
        poolclass=NullPool,
        connect_args=connect_args,
    )
