# World Cup 2026 Prediction Pipeline

End-to-end pipeline for ingesting, processing, and serving international football match predictions with production-grade observability.

## Current Stack

- Python for ingestion, processing, and modeling
- PostgreSQL in Docker for infrastructure
- dbt for SQL curation and data quality over persisted medallion tables
- FastAPI scaffold for the serving layer + inference logging
- SQLAlchemy for ORM and connection pooling

## Project Documentation

- 📊 [PROJECT_STATUS.md](PROJECT_STATUS.md) — Executive overview & architecture
- 📝 [INFERENCE_LOGGING_GUIDE.md](INFERENCE_LOGGING_GUIDE.md) — Complete logging system
- 🧪 [TESTING_INFERENCE_LOGGING.md](TESTING_INFERENCE_LOGGING.md) — Local testing guide
- 🔧 [INGESTION_ARCHITECTURE.md](INGESTION_ARCHITECTURE.md) — Data pipeline design
- 📈 [PHASE_3B_SENIOR_FEATURES_SUMMARY.md](PHASE_3B_SENIOR_FEATURES_SUMMARY.md) — Feature engineering

## Project Setup

1. Copy `.env.example` to `.env`
2. Start PostgreSQL:

```bash
docker compose up -d
```

3. Create or refresh the local virtual environment:

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

## Data Layers

- `data/raw`: source CSV and raw API snapshots
- `data/bronze`: standardized source extracts
- `data/silver`: cleaned canonical match tables
- `data/gold`: feature datasets ready for ML

## Pipeline Flow

1. Ingestion writes raw snapshots and standardized bronze data
2. Processing writes `data/silver/matches_cleaned.csv`
3. Processing writes `data/gold/features_dataset.csv`
4. Training exports `models/match_predictor.joblib`
5. FastAPI serves predictions from `src.api.main:app`
6. **NEW:** Every prediction is logged to `monitoring.inference_logs` table
7. FastAPI prefers `dbt`-curated latest team snapshots, then falls back to PostgreSQL and CSV

## Run End-to-End Pipeline

```bash
.venv\Scripts\python run_pipeline.py
```

Useful variants:

```bash
.venv\Scripts\python run_pipeline.py --skip-ingestion
.venv\Scripts\python run_pipeline.py --skip-ingestion --skip-processing
.venv\Scripts\python run_pipeline.py --no-api-data
.venv\Scripts\python run_pipeline.py --persist-to-db
```

This orchestrates ingestion, processing, and training, and prints a JSON
summary with stage timings, artifact paths, and evaluation metrics.

When `--persist-to-db` is enabled, the pipeline also writes snapshots to
PostgreSQL:

- `bronze.historical_matches`
- `bronze.api_matches`
- `silver.matches_cleaned`
- `gold.features_dataset`
- `gold.training_runs`
- **NEW:** `monitoring.inference_logs` (created automatically in `docker/postgres/init.sql`)

These persisted tables are the source layer for the `dbt` project documented in
[`dbt/README.md`](dbt/README.md).

## Run Training

```bash
.venv\Scripts\python -m src.modeling.train
```

This trains the production XGBoost pipeline on `data/gold/features_dataset.csv`

## Serve Predictions with Observability

```bash
uvicorn src.api.main:app --reload
```

Visit `http://localhost:8000/docs` for interactive API documentation.

**Key endpoints:**

- `POST /predict` — Predict match outcome (auto-logs to PostgreSQL)
- `GET /monitoring/inference-stats` — Last 24h aggregated statistics
- `GET /monitoring/recent-inferences` — Recent predictions for auditing
- `GET /monitoring/latest-training-run` — Current model metrics

**Example prediction:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"home_team":"Brazil","away_team":"Argentina"}'
```

See [INFERENCE_LOGGING_GUIDE.md](INFERENCE_LOGGING_GUIDE.md) for full API documentation and usage patterns.
using a temporal holdout split and exports `models/match_predictor.joblib`.

## Run API

```bash
.venv\Scripts\uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Prediction serving uses `PREDICTION_FEATURE_SOURCE=auto` by default, which:

- loads team feature snapshots from `analytics_gold.gold_latest_team_snapshots` when available
- falls back to `gold.features_dataset` in PostgreSQL if the dbt model is unavailable
- falls back to `data/gold/features_dataset.csv` if PostgreSQL is unavailable

Use `PREDICTION_FEATURE_SOURCE=dbt` to require dbt-backed serving,
`PREDICTION_FEATURE_SOURCE=postgres` to require raw DB-backed serving, or
`PREDICTION_FEATURE_SOURCE=csv` to force local CSV inference.

Training monitoring uses `MONITORING_SOURCE=auto` by default, which prefers
`analytics_gold.gold_latest_training_run` and falls back to `gold.training_runs`.

Example prediction request:

```bash
curl -X POST "http://localhost:8000/predict" ^
  -H "Content-Type: application/json" ^
  -d "{\"home_team\":\"Colombia\",\"away_team\":\"Argentina\",\"tournament\":\"FIFA World Cup Qualifiers\",\"neutral\":false}"
```

Monitoring request:

```bash
curl "http://localhost:8000/monitoring/latest-training-run"
```

## Docker

- `docker-compose.yml`: PostgreSQL + FastAPI service
- `docker/api/Dockerfile`: API container image

Run both services:

```bash
docker compose up --build
```

Run only the API after generating `data/gold/features_dataset.csv` and
`models/match_predictor.joblib` locally:

```bash
docker compose up --build api
```

## dbt

The repo now includes a `dbt` project under [`dbt/README.md`](dbt/README.md)
that starts on top of the persisted PostgreSQL layers instead of replacing the
Python pipeline.

Recommended flow:

```bash
.venv\Scripts\python run_pipeline.py --persist-to-db
.venv\Scripts\python run_dbt.py debug
.venv\Scripts\python run_dbt.py source freshness
.venv\Scripts\python run_dbt.py run
.venv\Scripts\python run_dbt.py test
```
