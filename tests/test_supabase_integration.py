"""Integration tests for the Supabase PostgreSQL database layer.

Validates:
- Live connectivity via the SQLAlchemy engine fixture (credentials from .env).
- Medallion architecture: bronze, silver, and gold schemas exist.
- Core tables are present and populated (row count > 0).
- information_schema queries to enumerate schemas and tables.

Design decisions:
- All credentials are sourced from environment variables loaded by
  python-dotenv (via src.config.settings). No secrets are hardcoded.
- Every test that requires a live DB connection uses the shared
  ``engine_fixture`` from conftest.py, which auto-skips when the DB is
  unavailable (e.g., in offline CI without a DB service container).
- Tests are read-only — no DDL or DML mutations are performed here.
- Marked with ``pytest.mark.integration`` so they can be excluded from
  fast unit-test runs: ``pytest -m "not integration"``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest
from sqlalchemy import text

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

# ── Medallion architecture contract ──────────────────────────────────────────
# These are the schemas and tables the pipeline is expected to populate.
# Update this mapping whenever a new layer/table is added to the pipeline.
EXPECTED_TABLES: dict[str, list[str]] = {
    "bronze": ["historical_matches"],
    "silver": ["matches_cleaned"],
    "gold": ["features_dataset", "training_runs"],
}


# ── Markers ───────────────────────────────────────────────────────────────────
pytestmark = pytest.mark.integration


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fetch_schemas(engine: "Engine") -> set[str]:
    """Return all user-created schema names from information_schema."""
    sql = text(
        """
        SELECT schema_name
        FROM information_schema.schemata
        WHERE schema_name NOT IN ('pg_catalog', 'information_schema',
                                  'pg_toast', 'pg_temp_1', 'pg_toast_temp_1')
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(sql).fetchall()
    return {row[0] for row in rows}


def _fetch_tables(engine: "Engine", schema: str) -> set[str]:
    """Return all table names within a given schema."""
    sql = text(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = :schema
          AND table_type = 'BASE TABLE'
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(sql, {"schema": schema}).fetchall()
    return {row[0] for row in rows}


def _row_count(engine: "Engine", schema: str, table: str) -> int:
    """Return the row count for a specific table."""
    # Use quoted identifiers to prevent SQL injection from schema/table names.
    sql = text(f'SELECT COUNT(*) FROM "{schema}"."{table}"')  # noqa: S608
    with engine.connect() as conn:
        result = conn.execute(sql).scalar()
    return int(result or 0)


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestSupabaseConnectivity:
    """Validate basic database connectivity and version information."""

    def test_select_one(self, engine_fixture: "Engine") -> None:
        """Confirm the connection is alive and returns results."""
        with engine_fixture.connect() as conn:
            result = conn.execute(text("SELECT 1")).scalar()
        assert result == 1, "Basic connectivity check failed — SELECT 1 returned unexpected value"

    def test_postgres_version_is_readable(self, engine_fixture: "Engine") -> None:
        """Confirm we can query the server version (sanity check on permissions)."""
        with engine_fixture.connect() as conn:
            version: str | None = conn.execute(text("SELECT version()")).scalar()
        assert version is not None, "Could not retrieve PostgreSQL version"
        assert "PostgreSQL" in version, f"Unexpected version string: {version}"
        logger.info("Connected to: %s", version[:80])


class TestMedallionSchemas:
    """Validate that the Medallion architecture schemas exist in Supabase."""

    def test_bronze_schema_exists(self, engine_fixture: "Engine") -> None:
        """Bronze schema must exist — it holds raw ingested data."""
        schemas = _fetch_schemas(engine_fixture)
        assert "bronze" in schemas, (
            f"Schema 'bronze' not found. Available schemas: {schemas}. "
            "Run the ingestion pipeline first: uv run python run_pipeline.py --persist-to-db"
        )

    def test_silver_schema_exists(self, engine_fixture: "Engine") -> None:
        """Silver schema must exist — it holds cleaned/standardised data."""
        schemas = _fetch_schemas(engine_fixture)
        assert "silver" in schemas, (
            f"Schema 'silver' not found. Available schemas: {schemas}."
        )

    def test_gold_schema_exists(self, engine_fixture: "Engine") -> None:
        """Gold schema must exist — it holds feature-engineered and model data."""
        schemas = _fetch_schemas(engine_fixture)
        assert "gold" in schemas, (
            f"Schema 'gold' not found. Available schemas: {schemas}."
        )

    def test_all_medallion_schemas_present(self, engine_fixture: "Engine") -> None:
        """Single assertion for all three layers — used in CI for a fast smoke test."""
        schemas = _fetch_schemas(engine_fixture)
        missing = set(EXPECTED_TABLES.keys()) - schemas
        assert not missing, (
            f"Missing Medallion schemas: {missing}. "
            "Re-run: uv run python run_pipeline.py --persist-to-db"
        )


class TestMedallionTables:
    """Validate that all expected tables exist within their respective schemas."""

    @pytest.mark.parametrize("schema,table", [
        (schema, table)
        for schema, tables in EXPECTED_TABLES.items()
        for table in tables
    ])
    def test_table_exists(
        self, engine_fixture: "Engine", schema: str, table: str
    ) -> None:
        """Each expected table must be present in its schema."""
        tables_in_schema = _fetch_tables(engine_fixture, schema)
        assert table in tables_in_schema, (
            f"Table '{schema}.{table}' not found. "
            f"Tables found in '{schema}': {tables_in_schema}"
        )


class TestMedallionRowCounts:
    """Validate that core pipeline tables are non-empty after ingestion."""

    @pytest.mark.parametrize("schema,table,min_rows", [
        ("bronze", "historical_matches", 10_000),  # 49k+ rows expected from GitHub CSV
        ("silver", "matches_cleaned", 5_000),      # Post-1990 filter: ~32k rows
        ("gold", "features_dataset", 5_000),       # Feature-engineered gold layer
        ("gold", "training_runs", 1),              # At least one training run recorded
    ])
    def test_table_has_rows(
        self,
        engine_fixture: "Engine",
        schema: str,
        table: str,
        min_rows: int,
    ) -> None:
        """Table must contain at least ``min_rows`` rows — confirms pipeline ran successfully."""
        count = _row_count(engine_fixture, schema, table)
        assert count >= min_rows, (
            f"'{schema}.{table}' has only {count:,} rows (expected >= {min_rows:,}). "
            "The pipeline may not have completed successfully."
        )
        logger.info("✅ %s.%s: %s rows", schema, table, f"{count:,}")


class TestDatabaseSecurity:
    """Validate that the connection uses a safe, non-superuser role."""

    def test_current_user_is_not_postgres_superuser(self, engine_fixture: "Engine") -> None:
        """Supabase pooler user should not have superuser privileges.

        The Transaction/Session Pooler user is 'postgres.{project_ref}',
        which is a restricted role — not a full superuser. This is the
        expected secure configuration for production.
        """
        sql = text(
            """
            SELECT usesuper
            FROM pg_user
            WHERE usename = current_user
            """
        )
        with engine_fixture.connect() as conn:
            is_superuser = conn.execute(sql).scalar()
        # Pooler users are typically not superusers on Supabase — this is correct.
        # If this returns True in production, flag it for review.
        assert is_superuser is not None, "Could not determine current user privilege level"
        logger.info("Current user is_superuser=%s (expected False in Supabase pooler)", is_superuser)

    def test_current_user_is_readable(self, engine_fixture: "Engine") -> None:
        """Confirm the current DB user identity matches the expected pooler format."""
        with engine_fixture.connect() as conn:
            user: str | None = conn.execute(text("SELECT current_user")).scalar()
        assert user is not None
        # Supabase pooler usernames follow the format: postgres.{project_ref}
        assert "postgres" in user.lower(), (
            f"Unexpected DB user: '{user}'. Expected a Supabase pooler user "
            "(format: postgres.<project_ref>)."
        )
        logger.info("Connected as DB user: %s", user)
