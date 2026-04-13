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

## Phase 5: MLOps Automation & Promotion Gating

### Continuous Data Ingestion
To ensure the prediction model never becomes stale, we modernized the ingestion pipeline (`csv_client.py`) to natively pull and validate real-time updates directly from the canonical GitHub remote source. It dynamically handles structural quirks in upstream datasets—such as future World Cup fixtures containing `NA` scores—by intercepting and cleaning them before they trigger strict `DataContract` failures.

### The Champion vs. Challenger Gate
We completely eliminated the risk of blind, degraded deployments. Retraining the dataset executes an automated evaluation pipeline (`reporting_comparison.py`) that pits the newly trained model (the **Challenger**) against the deployed production model (the **Champion**). 
The pipeline computes deltas across Log-Loss, Macro F1, and Draw-Class metrics. Only if the Challenger strictly outperforms the Champion are the artifacts overwritten. If a performance regression is detected (such as via unexpected temporal drift), the script enforces a `KEEP_V1` decision and intelligently archives the snapshot.

### Production CI/CD
This entire pipeline runs offline via a scheduled GitHub Actions workflow (`retrain.yml`). It securely tests, formats, runs static analysis, trains models, evaluates them, and forces peer-review if promotion happens—acting as a completely automated, resilient MLOps backend.

---

## Key Learnings & Future Roadmap

- **ELO Time Decay:** Applying an exponential time decay to ELO difference calculations was the single biggest feature contribution. A win 3 years ago matters significantly less than a win last month.
- **Neutral Grounds:** The World Cup is played predominantly on neutral ground. Standardizing the target label required explicitly zeroing out home-field advantage features when `neutral=True`.
- **Model Degradation from Noise:** In Phase 5, we learned firsthand that feeding *more* data (recent friendlies) doesn't strictly guarantee a *better* aggregate model. The automated gating system successfully detected a Log-Loss degradation (+0.029) generated by the noise of pre-tournament friendlies and safely blocked deployment!
