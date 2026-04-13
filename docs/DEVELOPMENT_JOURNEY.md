# Development Journey: World Cup 2026 Prediction Engine

This document chronologically synthesizes the technical iterations, pivots, and learnings from building a scalable, MLOps-driven machine learning system for international football prediction.

---

## Phase 1: Ingestion & Pipeline Foundations

### Objective
Establish a reproducible, idempotent, and type-safe data pipeline to feed the ML models, utilizing modern Data Engineering best practices.

### The Bronze-Silver-Gold Architecture
We implemented a medallion architecture using `dbt` (Data Build Tool) backed by PostgreSQL.
- **Bronze (Ingestion)**: Raw ingestion of historical international results from CSVs.
- **Silver (Cleansing & Standardization)**: Type coercion, missing value handling, and standardization of team names across different eras.
- **Gold (Feature Engineering & Rollups)**: The creation of the core feature store. This layer calculates ELO ratings, time-decayed rolling averages of team form (win rates, goals scored/conceded), and home-field advantages.

### Milestone Achievement
By using `SQLAlchemy` and `dbt`, the entire ingestion pipeline guarantees idempotency—a fresh schema can be rebuilt from scratch entirely automatically.

---

## Phase 2: Modeling Pivot — From Generalist to Segment-Aware Ensemble

### The Target Leakage Discovery
During initial modeling with Gradient Boosted Trees (XGBoost/LightGBM), we identified severe data leakage caused by naive train/test splits that randomly mixed future matches into the training set. 

**The Fix:**
- Enforced strict **temporal cross-validation** (TimeSeriesSplit).
- Added lag features and ensured rolling windows strictly excluded the target match's data.

### The Draw-Specialist Architecture
Traditional multiclass classification (Home Win, Draw, Away Win) severely underestimates the likelihood of a draw because "Draw" is the minority class in football (~25%).
We pivoted to a **Segment-Aware Hybrid Ensemble**:
1. **The Generalist Model**: Predicts the baseline probabilities (Home/Away/Draw).
2. **The Draw Specialist**: A secondary binary classifier tuned specifically for high-uncertainty regions (e.g., when the Generalist outputs 35% vs 35% probabilities).
3. **The Router**: If the absolute difference between `p_home` and `p_away` is beneath a strict threshold, the Draw Specialist activates and overrides the Generalist.

---

## Phase 3: MLOps Evaluation & Shadow Deployment

### Calibration over Accuracy
In football prediction, calibration is more important than accuracy. A model outputting 60% probability must see that outcome occur exactly 60% of the time.
We implemented **Platt Scaling (CalibratedClassifierCV)** over our ensemble, resulting in near-perfect diagonal calibration curves on the Brier Score metric.

### Shadow Model Deployment
To safely iterate on the algorithm without breaking production:
1. The **Primary Model** (`match_predictor.joblib`) handles the core API requests and dictates the final prediction.
2. The **Shadow Model** (`match_predictor_shadow.joblib`) runs in parallel within the same `/predict` request.
3. Both sets of probabilities are returned in the telemetry JSON to measure drift and compare generalist vs ensemble performance in real-time.

---

## Phase 4: Production API & Web DevOps

### Resilient Inference Architecture (Offline-First)
Local development and free-tier cloud deployments (like Render) often lack a dedicated, heavy PostgreSQL instance.
- **The Solution:** The `prediction_feature_source` is configurable via `.env`. By setting it to `csv`, the API bypasses the PostgreSQL database entirely and loads a frozen snapshot of the `features_dataset.csv` directly into RAM. This guarantees 100% uptime for predictions even if the cloud database goes offline.

### Strict API Security
The `FastAPI` instance distinguishes between client-facing inferences and private admin routes (`/admin/*`). Admin routes are strictly shielded by `ADMIN_API_KEY` verification.

### End-to-End Orchestration
- **Backend:** containerized via a highly optimized Dockerfile using `UvicornWorker` behind `gunicorn`, restricted to `WEB_CONCURRENCY=1` to fit within 512MB RAM constraints on the Render free tier.
- **Frontend:** Streamlit Community Cloud isolates the heavy ML backend from the UI layer, fetching predictions via secure HTTP requests.

---

## Key Learnings & Future Roadmap

- **ELO Time Decay:** Applying an exponential time decay to ELO difference calculations was the single biggest feature contribution. A win 3 years ago matters significantly less than a win last month.
- **Neutral Grounds:** The World Cup is played predominantly on neutral ground. Standardizing the target label required explicitly zeroing out home-field advantage features when `neutral=True`.
- **Future Work:** Deploying the continuous training loop using Airflow or GitHub Actions cron jobs to daily-refresh the Gold layer via external APIs.
