# World Cup 2026 Prediction Engine

[![CI Pipeline](https://github.com/JuanjoRestrepo/worldcup-2026-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/JuanjoRestrepo/worldcup-2026-prediction/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Live API](https://img.shields.io/badge/API-live-success)](https://worldcup-2026-prediction.onrender.com/docs)
[![Tests: 157](https://img.shields.io/badge/tests-157%20passing-success)](tests/)

Industrial-grade MLOps system for football match outcome prediction. Uses a **segment-aware hybrid ensemble** where a Logistic Regression champion handles confident predictions, while a TwoStage draw specialist takes over in high-uncertainty zones (especially friendlies). Every prediction is logged to PostgreSQL with full telemetry.

🚀 **[Live API](https://worldcup-2026-prediction.onrender.com)** · [Architecture](docs/ARCHITECTURE.md) · [API Docs](https://worldcup-2026-prediction.onrender.com/docs)

---

## Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.12 |
| Package manager | `uv` (exclusive — no pip/conda) |
| Data engineering | PostgreSQL + dbt (Medallion: Bronze→Silver→Gold) |
| Feature serving | dbt team snapshot models |
| Model selection | Rolling temporal backtesting (5 folds, 26 candidates) |
| Serving | FastAPI + uvicorn |
| Inference telemetry | PostgreSQL (`monitoring.inference_logs`, JSONB) |
| Code quality | ruff (12 rule sets) + mypy strict |
| CI/CD | GitHub Actions → Render |
| Deployment | Render.com (free tier) |

---

## Architecture Overview

```
CSV Data → Bronze → Silver → Gold (dbt) → Training (26 candidates)
                                               ↓
                              Champion: logistic_c2_draw1 (LR)
                              Shadow:   seg_hybrid_balanced (Segment-Aware Ensemble)
                                               ↓
                                          FastAPI /predict
                                        ↙       ↓       ↘
                              Alias map   Feature    Segment
                              (USA→US)    freshness  routing
                                                ↓
                                    monitoring.inference_logs
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full system diagram.

---

## Quick Start

```bash
# 1. Clone & install
uv sync

# 2. Start PostgreSQL
docker compose up -d postgres

# 3. Run full pipeline (ingest → process → train → dbt)
uv run python run_pipeline.py --persist-to-db

# 4. Run dbt curated models
uv run python run_dbt.py run && uv run python run_dbt.py test

# 5. Start the API
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## Run Tests

```bash
uv run python -m pytest tests/ -v --tb=short
```

**157 tests** covering: rolling features, ELO decay, draw rate, hybrid ensemble, segment-aware routing, inference logging, API hardening, team aliases, calibration, backtesting integration, and more.

---

## Make a Prediction

```bash
# Local
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"home_team": "USA", "away_team": "Mexico", "tournament": "CONCACAF Qualifiers", "neutral": false}'

# Production
curl -X POST "https://worldcup-2026-prediction.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{"home_team": "Argentina", "away_team": "Brazil", "tournament": "World Cup"}'
```

Response includes:
- `predicted_outcome` (home_win / draw / away_win) + `class_probabilities`
- `match_segment` (worldcup / continental / qualifiers / friendlies)
- `is_override_triggered` (segment-aware specialist override flag)
- `shadow_predicted_outcome` + `shadow_class_probabilities` (shadow ensemble result)
- `feature_freshness` (staleness warning when snapshots > 30 days old)

**Team name aliases are supported**: USA, USMNT, United States → all resolve to "United States".

---

## Pipeline Details

```bash
# Run specific stages
uv run python run_pipeline.py --skip-ingestion            # skip data download
uv run python run_pipeline.py --skip-ingestion --skip-processing  # retrain only
uv run python run_pipeline.py --persist-to-db             # write all layers to PostgreSQL
```

Writes when `--persist-to-db`:
- `bronze.historical_matches`, `bronze.api_matches`
- `silver.matches_cleaned`
- `gold.features_dataset`, `gold.training_runs`

---

## Training Details

```bash
uv run python -m src.modeling.train
```

Performs:
1. **Auto-tunes** segment thresholds via OOF temporal validation
2. **26 candidate models** compete across 5 rolling time-series folds
3. **Multicriteria ranking**: accuracy + ECE + draw recall + Brier score
4. **Calibration selection**: isotonic vs sigmoid vs uncalibrated (OOF holdout)
5. **Shadow deployment**: rank-4 segment-aware hybrid exported as `_shadow.joblib`

Outputs:
- `models/match_predictor.joblib` (champion)
- `models/match_predictor_shadow.joblib` (shadow)
- `models/match_predictor_metrics.json`
- `models/match_predictor_evaluation_report.{json,md}`

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/` | Root with navigation links |
| `GET`  | `/health` | Readiness probe (shadow_as_primary flag) |
| `GET`  | `/config` | Non-sensitive runtime configuration |
| `POST` | `/predict` | Match outcome prediction |
| `GET`  | `/monitoring/latest-training-run` | Latest training metadata |
| `GET`  | `/monitoring/inference-stats?hours=24` | Aggregated inference statistics |
| `GET`  | `/monitoring/recent-inferences?limit=50` | Recent prediction audit trail |
| `POST` | `/admin/toggle-shadow` | Hot-swap champion ↔ shadow (API key required) |

Interactive docs: `http://localhost:8000/docs`

---

## Docker

```bash
# Full stack (PostgreSQL + API)
docker compose up --build

# API only (requires pre-built data/model artifacts)
docker compose up --build api
```

The API container runs as **non-root** (`appuser`) and uses the `/health` endpoint for healthchecks.

---

## CI/CD

Every push/PR to `main` or `develop` triggers:

```
ruff check   → ruff format --check   → mypy --strict   → pytest (157 tests)   → dbt parse
```

PRs that fail any gate receive an automatic comment and **cannot be merged**.

See: [docs/CI_CD_GUIDE.md](docs/CI_CD_GUIDE.md)

---

## Production Deployment

**Live API:** 🚀 [https://worldcup-2026-prediction.onrender.com](https://worldcup-2026-prediction.onrender.com)

- Hosting: Render.com (Web Service + PostgreSQL)
- Auto-deploy on `git push main`
- Release task: retrains model on each deploy with fresh data

```
git push main
  → GitHub Actions: ruff + mypy + pytest + dbt parse ✅
  → Render: uv sync --no-dev → run_pipeline.py (retrain)
  → uvicorn FastAPI starts → API live
```

---

## Documentation

| Topic | File |
|-------|------|
| Architecture | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| Project status | [PROJECT_STATUS.md](PROJECT_STATUS.md) |
| History of changes | [walkthrough.md](walkthrough.md) |
| Segment-aware hybrid guide | [docs/SEGMENT_AWARE_HYBRID_GUIDE.md](docs/SEGMENT_AWARE_HYBRID_GUIDE.md) |
| Inference logging guide | [docs/INFERENCE_LOGGING_GUIDE.md](docs/INFERENCE_LOGGING_GUIDE.md) |
| CI/CD guide | [docs/CI_CD_GUIDE.md](docs/CI_CD_GUIDE.md) |
| Quick start | [docs/QUICK_START.md](docs/QUICK_START.md) |
