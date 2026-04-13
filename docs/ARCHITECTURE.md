# System Architecture — World Cup 2026 Prediction Engine

> **Stack**: Python 3.12 · FastAPI · PostgreSQL · dbt · scikit-learn · XGBoost · GitHub Actions · Render

---

## High-Level Data Flow

```
                          ┌──────────────────────────────────────────────────────────┐
                          │                  DATA INGESTION                          │
                          │  CSV (Kaggle international football results dataset)     │
                          │  → src/processing/pipelines/run_ingestion.py            │
                          └──────────────────────┬───────────────────────────────────┘
                                                 │
                                                 ▼
                          ┌──────────────────────────────────────────────────────────┐
                          │                BRONZE LAYER (PostgreSQL)                 │
                          │  schema: bronze                                          │
                          │  Tables: raw_matches  (standardized raw data)           │
                          └──────────────────────┬───────────────────────────────────┘
                                                 │  dbt
                                                 ▼
                          ┌──────────────────────────────────────────────────────────┐
                          │                SILVER LAYER (dbt models)                 │
                          │  ELO ratings (time-decay aware, K=20, mean-reversion)   │
                          │  Rolling features (window=5): goals, win_rate,          │
                          │    draw_rate_last5 (NEW), opponent_elo_form             │
                          │  Tournament flags: is_world_cup, is_friendly, etc.      │
                          └──────────────────────┬───────────────────────────────────┘
                                                 │  dbt
                                                 ▼
                          ┌──────────────────────────────────────────────────────────┐
                          │                GOLD LAYER (dbt models)                   │
                          │  schema: gold                                            │
                          │  Tables:                                                 │
                          │    features_dataset    (ML-ready, leakage-free)         │
                          │    team_snapshot_v2    (latest per-team state)          │
                          │    latest_team_snapshot (serving model for inference)   │
                          └──────────────────────┬───────────────────────────────────┘
                                                 │
                                                 ▼
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                           TRAINING PIPELINE (src/modeling/train.py)                  │
│                                                                                      │
│  26 candidate models compete under rolling temporal backtesting (5 folds):           │
│    - LogisticRegression variants (C ∈ {0.5, 1, 2}, draw_boost)                     │
│    - RandomForest variants (n_estimators, max_depth, draw_boost)                    │
│    - XGBoost variants (lr, reg_lambda, draw_boost)                                  │
│    - TwoStageDrawClassifier (stage1 binary + stage2 draw specialist)                │
│    - HybridDrawOverrideEnsemble (generalist + specialist, global threshold)         │
│    - SegmentAwareHybridEnsemble × 4 variants (per-tournament thresholds)            │
│                                                                                      │
│  Model selection: multicriteria ranking (accuracy, calibration, draw recall)        │
│  Calibration: CalibratedClassifierCV with isotonic/sigmoid, OOF selection           │
│  Auto-tuning: OOF segment threshold search (auto_tune_segment_thresholds)           │
│                                                                                      │
│  ✅ Champion: logistic_c2_draw1   (Logistic Regression, isotonic calibrated)        │
│  🥈 Shadow:   seg_hybrid_balanced (Segment-Aware Ensemble, Rank #4)                 │
│                                                                                      │
│  Output: models/match_predictor.joblib (ModelArtifactBundle TypedDict)             │
│          models/match_predictor_shadow.joblib                                        │
└──────────────────────────────────────────────────────────────────────────────────────┘
                                                 │
                                                 ▼
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                            SERVING LAYER (FastAPI)                                   │
│                                                                                      │
│  POST /predict                                                                       │
│    ┌─────────────────────────────────────────────────────────────────────┐          │
│    │  1. Normalize team name  →  team_aliases.py (40+ mappings)         │          │
│    │  2. Load model bundle    →  lru_cache(maxsize=4)                   │          │
│    │  3. Build feature frame  →  dbt serving model (gold.latest_team)   │          │
│    │                              with CSV fallback (auto source)        │          │
│    │  4. Detect segment       →  segment_routing.py (tournament classifier)│        │
│    │  5. Primary inference    →  Champion model                          │          │
│    │  6. Shadow inference     →  Shadow model (parallel, non-blocking)  │          │
│    │  7. Log to PostgreSQL    →  monitoring.inference_logs (async)      │          │
│    │  8. Feature freshness    →  validate_feature_freshness() warning   │          │
│    └─────────────────────────────────────────────────────────────────────┘          │
│                                                                                      │
│  GET  /health           Readiness probe (shadow_as_primary flag)                    │
│  GET  /monitoring/inference-stats    Aggregated statistics (24h window)             │
│  GET  /monitoring/recent-inferences  Audit trail (last N predictions)               │
│  POST /admin/toggle-shadow           Hot-swap champion ↔ shadow (auth required)     │
└──────────────────────────────────────────────────────────────────────────────────────┘
                                                 │
                                                 ▼
                          ┌──────────────────────────────────────────────────────────┐
                          │              TELEMETRY (PostgreSQL)                       │
                          │  schema: monitoring                                       │
                          │  Table: inference_logs (JSONB probabilities)             │
                          │    Fields: team, tournament, segment, probabilities,     │
                          │            shadow_outcome, shadow_override_triggered,    │
                          │            feature_source, model_artifact_path           │
                          └──────────────────────────────────────────────────────────┘
```

---

## Ensemble Architecture (Segment-Aware Hybrid)

```
Prediction request
       │
       ▼
┌─────────────┐
│ Tournament  │
│ Segment     │ → worldcup / qualifiers / continental / friendlies / None
│ Detector    │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────────────────┐
│           GENERALIST (Champion — LR C=2)            │
│   Predicts all outcome classes                      │
│   Calibrated via isotonic regression (OOF)          │
└──────┬──────────────────────────────────────────────┘
       │
       │  Is max_prob < segment_uncertainty_threshold?
       │  Is draw conviction ≥ segment_draw_threshold?
       │
       ├─ YES (uncertain + draw-favorable) ──→ ┌─────────────────────────────────┐
       │                                        │  SPECIALIST (Shadow - Segment)  │
       │                                        │  TwoStageDrawClassifier         │
       │                                        │  Stage1: Win vs Non-Win         │
       │                                        │  Stage2: Draw vs Loss           │
       │                                        └──────────────┬──────────────────┘
       │                                                       │
       └─ NO  (confident prediction) ──────────┐              │
                                               ▼              ▼
                                         Final prediction (routed)
```

| Segment     | Uncertainty Threshold | Draw Conviction | Rationale |
|-------------|-----------------------|-----------------|-----------|
| friendlies  | 0.36                  | 0.55            | Specialist helps most here |
| worldcup    | 0.50                  | 0.60            | Generalist dominates |
| continental | 0.42                  | 0.56            | Medium selectivity |
| qualifiers  | 0.46                  | 0.58            | Moderate structure |
| default     | 0.44                  | 0.55            | Safe fallback |

---

## CI/CD Pipeline

```
git push / pull_request
       │
       ▼
┌──────────────────────────────────────────────┐
│         GitHub Actions (.github/workflows)   │
│                                              │
│  PostgreSQL service container (postgres:15)  │
│  Python 3.12 + uv sync (cached by uv.lock)  │
│                                              │
│  1. ruff check     — lint (12 rule sets)    │
│  2. ruff format    — format check           │
│  3. mypy --strict  — type checking          │
│  4. pytest         — 150+ tests            │
│  5. dbt parse      — dbt syntax validation  │
└────────────────────┬─────────────────────────┘
                     │  All pass?
                     ├─ YES → PR comment: "✅ CI passed"
                     └─ NO  → PR comment: "❌ CI failed" + block merge
                              
                     │  Merge to main
                     ▼
              ┌─────────────┐
              │   Render    │
              │  Release    │
              │  Task:      │
              │  retrain    │
              │  model      │
              └─────────────┘
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Contract-First** | `ModelArtifactBundle` TypedDict enforces schema between train → serve |
| **Logistic champion** | Best calibration (lowest ECE) + highest balanced accuracy across folds |
| **Shadow deployment** | Risk-free A/B: shadow runs in parallel, no production impact |
| **Async inference logging** | Failures in DB write don't break prediction serving |
| **ELO time-decay** | Teams inactive for months shouldn't carry stale extreme ratings |
| **draw_rate_last5** | Direct draw-propensity signal for the two-stage specialist |
| **dbt medallion** | Separation between raw ingestion and ML-ready features for auditability |
| **uv exclusive** | Deterministic, fast, pyproject.toml-native dependency management |

---

## File Structure Summary

```
worldcup-2026-prediction/
├── src/
│   ├── api/          main.py (FastAPI)
│   ├── config/       settings.py, team_aliases.py
│   ├── contracts/    data_contracts.py (schema validation)
│   ├── database/     connection.py, persistence.py
│   ├── modeling/     train.py, predict.py, evaluate.py, features.py,
│   │                 hybrid_ensemble*.py, segment_routing.py, tuning.py,
│   │                 inference_logger.py, serving_store.py, types.py
│   └── processing/
│       └── transformers/ elo.py, rolling_features.py, opponent_strength.py
├── tests/            ~157 tests (pytest)
├── dbt_worldcup/     dbt project (bronze/silver/gold models)
├── docker/           Dockerfile (multi-stage, non-root), postgres/init.sql
├── .github/workflows/ ci.yml (lint+format+mypy+pytest+dbt)
├── docs/             session summaries, guides, roadmaps
├── pyproject.toml    uv project (ruff 12 rules, mypy strict)
└── README.md
```
