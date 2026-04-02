# World Cup 2026 Prediction Pipeline

End-to-end pipeline for ingesting, processing, and serving international football match predictions.

## Current Stack

- Python for ingestion, processing, and modeling
- PostgreSQL in Docker for infrastructure
- FastAPI scaffold for the serving layer

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

## Run Training

```bash
.venv\Scripts\python -m src.modeling.train
```

This trains the production XGBoost pipeline on `data/gold/features_dataset.csv`
using a temporal holdout split and exports `models/match_predictor.joblib`.

## Run API

```bash
.venv\Scripts\uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Example prediction request:

```bash
curl -X POST "http://localhost:8000/predict" ^
  -H "Content-Type: application/json" ^
  -d "{\"home_team\":\"Colombia\",\"away_team\":\"Argentina\",\"tournament\":\"FIFA World Cup Qualifiers\",\"neutral\":false}"
```

## Docker

- `docker-compose.yml`: PostgreSQL
- `docker/api/Dockerfile`: API container scaffold
