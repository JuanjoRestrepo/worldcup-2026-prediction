"""Load raw CSV data into PostgreSQL bronze layer for dbt transformation."""

import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from src.database.connection import get_sqlalchemy_engine

DATA_DIR = Path(__file__).parent / "data"
RAW_DIR = DATA_DIR / "raw"
GOLD_DIR = DATA_DIR / "gold"


def load_csv_to_postgres(
    csv_path: Path, table_name: str, schema: str = "bronze"
) -> int:
    """Load CSV file into PostgreSQL table."""
    if not csv_path.exists():
        print(f"❌ File not found: {csv_path}")
        return 0

    print(f"\n📂 Loading {csv_path.name} → {schema}.{table_name}...")
    df = pd.read_csv(csv_path)
    print(f"   Records: {len(df)}")
    print(f"   Columns: {', '.join(df.columns.tolist())}")

    engine = get_sqlalchemy_engine()
    try:
        # Drop existing table
        with engine.connect() as conn:
            conn.execute(
                text(f'DROP TABLE IF EXISTS "{schema}"."{table_name}" CASCADE')
            )
            conn.commit()

        # Load data
        df.to_sql(
            table_name,
            con=engine,
            schema=schema,
            if_exists="replace",
            index=False,
            method="multi",
            chunksize=1000,
        )
        print(f"   ✅ Loaded {len(df)} rows")
        return len(df)
    finally:
        engine.dispose()


def main() -> int:
    """Load all source data into PostgreSQL."""
    print("\n" + "=" * 70)
    print(" 📊 LOADING SOURCE DATA INTO POSTGRE SQL".center(70))
    print("=" * 70)

    total_rows = 0

    # Load international results
    rows = load_csv_to_postgres(
        RAW_DIR / "international_results.csv", "historical_matches"
    )
    total_rows += rows

    # Load features dataset
    rows = load_csv_to_postgres(
        GOLD_DIR / "features_dataset.csv", "features_dataset", schema="gold"
    )
    total_rows += rows

    print("\n" + "=" * 70)
    print(f" ✅ LOADED {total_rows} TOTAL ROWS".center(70))
    print("=" * 70)
    print("\n📋 Next steps:")
    print("   1. Run dbt to transform data: uv run python run_dbt.py run")
    print("   2. Test API: uv run python tests/test_api_local.py")
    print("   3. Deploy to production")

    return 0


if __name__ == "__main__":
    sys.exit(main())
