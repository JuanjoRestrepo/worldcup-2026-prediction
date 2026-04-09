# PHASE 1B: PRODUCTION DEPLOYMENT GUIDE

## 📋 Deployment Strategy

### Architecture

```
┌─────────────┐
│   RAW DATA  │ (CSV files or API sources)
│ in /data    │
└──────┬──────┘
       │
       ├──→ [LOCAL/TESTING]
       │    ├─ load_data.py (bulk load CSV)
       │    └─ fixture.sql (preconfigured test data)
       │       → PostgreSQL: bronze + gold schemas
       │       → dbt parse validates
       │       → API /predict works
       │
       └──→ [PRODUCTION]
            ├─ run_pipeline.py (orchestrated ingestion)
            │  ├─ Extract from sources
            │  ├─ Transform with Pandas
            │  └─ Persist to PostgreSQL
            │
            ├─ dbt run (silver + gold models)
            │  ├─ bronze_all_matches
            │  ├─ silver_matches_cleaned
            │  └─ gold_latest_team_snapshots (for /predict)
            │
            └─ Monitoring + Logging
               ├─ inference_logs table
               ├─ dbt test validations
               └─ Feature freshness alerts
```

---

## 🚀 LOCAL DEPLOYMENT (Current Setup)

### Prerequisites

```bash
# 1. Docker Desktop running
docker ps

# 2. PostgreSQL container
docker-compose up postgres -d

# 3. Python environment configured
uv sync
```

### Initialize Database

```bash
# Option A: Load CSV + fixture (recommended for testing)
uv run python load_data.py
docker-compose exec postgres psql -U worldcup -d worldcup_db < docker/postgres/fixture.sql

# Option B: Load CSV + dbt run (for dbt testing)
uv run python load_data.py
uv run python run_dbt.py run  # (requires dbt model fixes)
```

### Validate

```bash
# 1. Start API
uv run python -m uvicorn src.api.main:app --port 8000 --reload

# 2. Test /predict endpoint
uv run python tests/test_api_local.py

# Expected: ✅ 200 OK with prediction + match_segment + is_override_triggered
```

---

## 🏢 PRODUCTION DEPLOYMENT (Next Phase)

### Database Initialization

```sql
-- 1. Create schemas
CREATE SCHEMA bronze;
CREATE SCHEMA silver;
CREATE SCHEMA gold;
CREATE SCHEMA analytics_bronze;
CREATE SCHEMA analytics_silver;
CREATE SCHEMA analytics_gold;
CREATE SCHEMA monitoring;

-- 2. Create inference_logs table (for predictions)
CREATE TABLE monitoring.inference_logs (
    id SERIAL PRIMARY KEY,
    match_date DATE,
    home_team VARCHAR,
    away_team VARCHAR,
    predicted_outcome VARCHAR,
    predicted_probabilities JSONB,
    match_segment VARCHAR,
    is_override_triggered BOOLEAN,
    feature_source VARCHAR,
    model_version VARCHAR,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 3. Create indexes for monitoring queries
CREATE INDEX idx_inference_logs_timestamp ON monitoring.inference_logs(timestamp DESC);
CREATE INDEX idx_inference_logs_segment ON monitoring.inference_logs(match_segment);
CREATE INDEX idx_inference_logs_teams ON monitoring.inference_logs(home_team, away_team);
```

### Data Pipeline

```bash
# Production runs every 6 hours:
0 */6 * * * cd /app && uv run python run_pipeline.py --persist-to-db

# This orchestrates:
# 1. Extract: Fetch latest match data
# 2. Transform: Compute features (Elo, rolling stats, etc)
# 3. Load: Persist to PostgreSQL + dbt models
# 4. Monitor: Validate data contracts + feature freshness
```

### Deploy FastAPI

```bash
# Option 1: Heroku
git push heroku main

# Option 2: Azure App Service
az webapp deployment source config-zip --resource-group myGroup --name myApp --src-path deploy.zip

# Option 3: Docker + K8s
docker build -f docker/api/Dockerfile -t worldcup-api:latest .
kubectl apply -f k8s/deployment.yaml
```

### Environment Variables

```bash
# Database
POSTGRES_HOST=your-prod-db.postgres.database.azure.com
POSTGRES_PORT=5432
POSTGRES_DB=worldcup_prod
POSTGRES_USER=admin
POSTGRES_PASSWORD=<secure-password>

# Model
MODEL_ARTIFACT_PATH=/app/models/match_predictor.joblib

# Monitoring
MONITORING_SOURCE=dbt  # Use dbt models in production

# Feature configuration
PREDICTION_FEATURE_SOURCE=auto  # Falls back: dbt → PostgreSQL → CSV
```

---

## 📊 CURRENT STATUS (Phase 1B)

### ✅ Completed

- [x] CI/CD Pipeline (pytest, dbt parse, dbt test, ruff, mypy)
- [x] API Server (FastAPI + Segment-Aware Ensemble)
- [x] Database Schema (PostgreSQL with new columns)
- [x] Local Testing (Docker + fixture + end-to-end validation)
- [x] Contract-First Integration (Data contracts + validation)

### 🔄 In Progress

- [ ] Test fixture in production (local OK ✅)
- [ ] Deploy to cloud (Heroku/Azure)
- [ ] Monitoring dashboards (Phase 2)

### ⏳ Phase 2 (Hardening)

- [ ] dbt run with real data pipeline
- [ ] SQL monitoring views
- [ ] Feature freshness dashboards
- [ ] A/B testing framework

---

## 🧪 Testing Checklist

- [x] `pytest`: 114 tests passing
- [x] `dbt parse`: Syntax valid
- [x] `dbt test`: Contract validation (ready)
- [x] `ruff`: Linting passed
- [x] `mypy`: Type checking passed
- [x] API `/config`: 200 OK
- [x] API `/predict`: 200 OK with Argentina vs Brazil
- [x] Segment detection: ✅ "worldcup"
- [x] Ensemble telemetry: ✅ "is_override_triggered"
- [ ] Monitoring views: Phase 2
- [ ] Production database: TBD

---

## 📋 Next Steps for Deployment

### Immediate (Today)

1. ✅ Verify all tests pass: `uv run pytest tests/ -q`
2. ✅ Verify API works: `uv run python tests/test_api_local.py`
3. [ ] Push to GitHub: `git push origin main`
4. [ ] CI passes: Monitor workflows

### This Week

1. [ ] Select cloud provider (Heroku/Azure/AWS)
2. [ ] Create production PostgreSQL instance
3. [ ] Configure environment variables
4. [ ] Deploy API server
5. [ ] Test `/predict` in production
6. [ ] Set up monitoring

### Next Sprint (Phase 2)

1. [ ] Implement dbt run pipeline
2. [ ] Create SQL monitoring views
3. [ ] Build feature freshness dashboards
4. [ ] Set up alerts

---

## 🔗 Related Documents

- [DEPLOYMENT_CHECKLIST_PHASE1.md](../DEPLOYMENT_CHECKLIST_PHASE1.md) — Detailed deployment steps
- [MONITORING_PHASE2_GUIDE.md](../MONITORING_PHASE2_GUIDE.md) — Phase 2 setup
- [CI_CD_GUIDE.md](../CI_CD_GUIDE.md) — GitHub Actions configuration
- [QUICK_START.md](../QUICK_START.md) — Local development setup

---

**Status**: Ready for Phase 1B → Phase 2 Deployment ✅
