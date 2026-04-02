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

## Next Runtime Targets

- Processing pipeline output in `data/gold/features_dataset.csv`
- Future model artifact in `models/match_predictor.joblib`
- FastAPI app entrypoint at `src.api.main:app`

## Docker

- `docker-compose.yml`: PostgreSQL
- `docker/api/Dockerfile`: API container scaffold
