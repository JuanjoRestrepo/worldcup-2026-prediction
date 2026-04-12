# ETL Design Patterns & Pipeline Templates

## Table of Contents

0. [Environment Setup (uv + pyproject.toml)](#environment)
1. [Pandas / Polars Pipeline](#pandas-polars)
2. [PySpark Pipeline](#pyspark)
3. [dbt Model Pattern](#dbt)
4. [Airflow DAG Template](#airflow)
5. [Data Validation with Pandera / Great Expectations](#validation)

---

## 0. Environment Setup (uv + pyproject.toml) {#environment}

All data engineering projects use `uv` for environment and dependency management.
Never use `pip`, `conda`, or `requirements.txt`.

```bash
# Initialize project
uv init etl_project && cd etl_project
uv python pin 3.12

# Create .venv
uv venv .venv --python 3.12
source .venv/bin/activate

# Add ETL-specific dependencies
uv add pandas polars sqlalchemy pandera great-expectations \
       apache-airflow pyspark dbt-core pydantic pyyaml httpx

# Add dev tools (Ruff + mypy mandatory)
uv add --dev ruff mypy pytest pytest-cov ipykernel

# Sync all dependencies
uv sync
```

**Standard `pyproject.toml` for ETL projects** — include `[tool.ruff]` and `[tool.mypy]`
sections as defined in the main SKILL.md Code Quality & Environment Standards section.

---

## 1. Pandas / Polars Pipeline {#pandas-polars}

```python
import pandas as pd
import polars as pl
import logging
import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import pandera as pa
from pandera import Column, DataFrameSchema

logger = logging.getLogger(__name__)

# --- Configuration ---
@dataclass
class PipelineConfig:
    """Externalized pipeline configuration."""
    input_path: str
    output_path: str
    target_column: str
    missing_threshold: float = 0.5
    categorical_encoding: str = "onehot"  # or 'label', 'target'

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        with open(path) as f:
            return cls(**yaml.safe_load(f))


# --- Schema Validation ---
INPUT_SCHEMA = DataFrameSchema({
    "id": Column(int, nullable=False),
    "date": Column(str, nullable=False),
    "value": Column(float, nullable=True),
})


# --- Pipeline Steps (modular, idempotent) ---
def ingest(config: PipelineConfig) -> pd.DataFrame:
    """Load raw data and validate against input schema."""
    path = Path(config.input_path)
    df = pd.read_csv(path) if path.suffix == ".csv" else pd.read_excel(path)
    INPUT_SCHEMA.validate(df)
    logger.info("Ingested %d rows from %s", len(df), path)
    return df


def clean(df: pd.DataFrame, missing_threshold: float) -> pd.DataFrame:
    """Drop high-null columns, fill remaining nulls, deduplicate."""
    null_rate = df.isnull().mean()
    cols_to_drop = null_rate[null_rate > missing_threshold].index.tolist()
    df = df.drop(columns=cols_to_drop)
    logger.info("Dropped columns (>%.0f%% missing): %s", missing_threshold * 100, cols_to_drop)

    df = df.drop_duplicates()
    df = df.fillna(df.median(numeric_only=True))
    return df


def transform(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Apply feature engineering transformations."""
    # Encode categoricals
    categorical_cols = df.select_dtypes(include="object").columns.difference([config.target_column])
    if config.categorical_encoding == "onehot":
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Parse dates if present
    date_cols = df.filter(like="date").columns
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")
        df[f"{col}_year"] = df[col].dt.year
        df[f"{col}_month"] = df[col].dt.month
        df[f"{col}_dayofweek"] = df[col].dt.dayofweek

    return df


def load(df: pd.DataFrame, output_path: str) -> None:
    """Persist transformed data to disk."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info("Saved %d rows to %s", len(df), output_path)


def run_pipeline(config_path: str) -> pd.DataFrame:
    """Orchestrate the full ETL pipeline."""
    config = PipelineConfig.from_yaml(config_path)
    df = ingest(config)
    df = clean(df, config.missing_threshold)
    df = transform(df, config)
    load(df, config.output_path)
    return df
```

---

## 2. PySpark Pipeline {#pyspark}

```python
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StructType
import logging

logger = logging.getLogger(__name__)

SPARK_APP_NAME: str = "etl_pipeline"
INPUT_FORMAT: str = "parquet"
OUTPUT_FORMAT: str = "delta"  # or 'parquet'


def create_spark_session(app_name: str = SPARK_APP_NAME) -> SparkSession:
    """Initialize SparkSession with sensible defaults."""
    return (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.shuffle.partitions", "200")
        .getOrCreate()
    )


def ingest_spark(spark: SparkSession, path: str, schema: StructType = None) -> DataFrame:
    """Read data with optional schema enforcement."""
    reader = spark.read.format(INPUT_FORMAT)
    if schema:
        reader = reader.schema(schema)
    df = reader.load(path)
    logger.info("Loaded %d rows from %s", df.count(), path)
    return df


def clean_spark(df: DataFrame, null_threshold: float = 0.5) -> DataFrame:
    """Remove high-null columns and deduplicate."""
    total = df.count()
    null_rates = {c: df.filter(F.col(c).isNull()).count() / total for c in df.columns}
    cols_to_drop = [c for c, r in null_rates.items() if r > null_threshold]
    df = df.drop(*cols_to_drop)
    df = df.dropDuplicates()
    return df


def write_spark(df: DataFrame, output_path: str, partition_cols: list[str] = None) -> None:
    """Write to output with optional partitioning."""
    writer = df.write.format(OUTPUT_FORMAT).mode("overwrite")
    if partition_cols:
        writer = writer.partitionBy(*partition_cols)
    writer.save(output_path)
    logger.info("Written to %s", output_path)
```

---

## 3. dbt Model Pattern {#dbt}

```sql
-- models/marts/core/fact_orders.sql
-- Materialization: table (use 'incremental' for large datasets)
{{ config(
    materialized='incremental',
    unique_key='order_id',
    on_schema_change='fail'
) }}

WITH source AS (
    SELECT * FROM {{ source('raw', 'orders') }}
),

cleaned AS (
    SELECT
        order_id,
        customer_id,
        order_date::DATE                            AS order_date,
        COALESCE(total_amount, 0)                   AS total_amount,
        status
    FROM source
    WHERE order_id IS NOT NULL
),

enriched AS (
    SELECT
        c.*,
        DATE_TRUNC('month', c.order_date)           AS order_month,
        EXTRACT(DOW FROM c.order_date)              AS day_of_week
    FROM cleaned c
)

SELECT * FROM enriched
{% if is_incremental() %}
WHERE order_date > (SELECT MAX(order_date) FROM {{ this }})
{% endif %}
```

---

## 4. Airflow DAG Template {#airflow}

```python
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
import logging

logger = logging.getLogger(__name__)

DEFAULT_ARGS: dict = {
    "owner": "data-engineering",
    "depends_on_past": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": True,
}

with DAG(
    dag_id="etl_pipeline",
    default_args=DEFAULT_ARGS,
    schedule_interval="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["etl", "data-engineering"],
) as dag:

    start = EmptyOperator(task_id="start")

    ingest_task = PythonOperator(
        task_id="ingest",
        python_callable=lambda: logger.info("Ingesting data..."),
    )

    transform_task = PythonOperator(
        task_id="transform",
        python_callable=lambda: logger.info("Transforming data..."),
    )

    load_task = PythonOperator(
        task_id="load",
        python_callable=lambda: logger.info("Loading data..."),
    )

    end = EmptyOperator(task_id="end")

    start >> ingest_task >> transform_task >> load_task >> end
```

---

## 5. Data Validation {#validation}

```python
import pandera as pa
from pandera import Column, DataFrameSchema, Check
import pandas as pd


# Define schema with business rules
ORDERS_SCHEMA = DataFrameSchema(
    columns={
        "order_id": Column(int, Check.greater_than(0), nullable=False),
        "total_amount": Column(float, Check.in_range(0, 1_000_000), nullable=False),
        "status": Column(str, Check.isin(["pending", "completed", "cancelled"])),
        "order_date": Column("datetime64[ns]", nullable=False),
    },
    checks=Check(lambda df: df["order_date"].max() <= pd.Timestamp.now()),
    coerce=True,
    strict=False,
)


def validate_dataframe(df: pd.DataFrame, schema: pa.DataFrameSchema) -> pd.DataFrame:
    """Validate DataFrame against schema; raise on violation."""
    try:
        validated = schema.validate(df, lazy=True)
        return validated
    except pa.errors.SchemaErrors as e:
        raise ValueError(f"Schema validation failed:\n{e.failure_cases}") from e
```
