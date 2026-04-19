# dbt Layer

This `dbt` project sits on top of the PostgreSQL tables persisted by the Python
pipeline:

- `bronze.historical_matches`
- `bronze.api_matches`
- `silver.matches_cleaned`
- `gold.features_dataset`
- `gold.training_runs`

The first `dbt` increment does not replace the Python transformations. Instead,
it adds:

- documented sources for the persisted medallion tables
- curated `snake_case` SQL views for analytics
- freshness checks on persisted bronze, silver, and gold sources
- lineage-oriented tests between serving snapshots and the latest training run
- serving-oriented team snapshot models
- monitoring-oriented training run models

## Recommended Local Flow

1. Run the Python pipeline with DB persistence:

```bash
uv sync
uv run python run_pipeline.py --persist-to-db
```

2. Validate the connection:

```bash
uv run python run_dbt.py debug
uv run python run_dbt.py source freshness
```

3. Build the dbt models:

```bash
uv run python run_dbt.py run
uv run python run_dbt.py test
```

`run_dbt.py` loads the same `.env` file used by the Python pipeline and will
auto-create `dbt/profiles.yml` from `dbt/profiles.yml.example` if it is missing.
It also defaults to `DBT_THREADS=1`, which keeps dbt compatible with this local
Windows workflow and avoids multiprocessing issues in constrained environments.
Use `uv run` rather than `.venv\Scripts\activate`; this repo standardizes local
Python execution on `uv`.

## Initial Models

- `bronze_all_matches`
- `silver_matches_cleaned_curated`
- `gold_feature_dataset_curated`
- `gold_training_runs_curated`
- `gold_team_feature_snapshots`
- `gold_latest_team_snapshots`
- `gold_latest_training_run`

## Downstream Usage

- FastAPI serving prefers `analytics_gold.gold_latest_team_snapshots`
- FastAPI monitoring prefers `analytics_gold.gold_latest_training_run`
- dbt exposures document those downstream dependencies for lineage
