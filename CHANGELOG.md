# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.3.0] - 2026-06-11

### Added
- **Probability Calibration via Temperature Scaling**: Custom ensemble models now receive post-hoc temperature scaling via `scipy.optimize.minimize_scalar` to minimize Expected Calibration Error (ECE) while preserving explicit draw-override thresholds.
- **Advanced Feature Engineering**: Added 14 new variables mapping industry-standard analytics context:
  - Exponential Recency (EWMA) Form (Goals scored/conceded, α=0.3)
  - Defensive Solidity: Clean sheet rates over the last 10 matches
  - Scoring Volatility: Goals scoring variance to proxy attack consistency
  - Fatigue Tracking: Rest days since last match
  - Confederation Strength: Average ELO of a team's home confederation
  - Tournament Context/Pressure: FIFA WC mapped to 1.0, Friendlies mapped to 0.2
  - Head-to-Head (H2H) Records: Historic win rates, draw rates, and goals in head-to-head match-ups
- **CV-Tuned Poisson Regressors**: Expected goals (xG) models are now backed by 3-fold cross-validated GridSearch to automatically select the optimal XGBRegressor hyperparameters.

### Fixed
- Re-enabled proper probability calibration for custom rule-based ensembles which previously bypassed calibration, drastically improving ECE and Log Loss metric evaluations.

---

## [1.2.0] - 2026-05-09

### Added
- **Supabase Integration**: Permanent PostgreSQL backend with Medallion architecture (Bronze/Silver/Gold).
- **dbt Transformation Layer**: Full dbt project for analytics and data quality validation.
- **Dynamic Theme Engine**: Streamlit theme orchestration via `config.toml` rewriting for high-contrast accessibility.
- **Integration Test Suite**: `tests/test_supabase_integration.py` for validating database health and security.
- **MyPy Strict Mode**: Achieved 100% strict type-checking across the entire codebase.

### Fixed
- Resolved all Ruff linting violations (including `pyupgrade` and `isort`).
- Fixed contrast issues in the Streamlit UI for the "Neutral Ground" label and sidebar widgets.
- Corrected database connection pooling configuration for Supabase compatibility.

### Changed
- Refactored `backend/database` to use `.env` settings via a centralized `settings.py`.
- Modernized type hints in all test suites and orchestration scripts.

---

## [1.1.0] - 2026-04-20

### Added
- **Segment-Aware Hybrid Ensemble**: Model that routes predictions to a draw specialist based on tournament type.
- **Champion vs Challenger Gate**: Automated model promotion gating in `reporting_comparison.py`.
- **Shadow Deployment**: Parallel inference of experimental models.

---

## [1.0.0] - 2026-03-15

### Added
- Initial release of the World Cup 2026 Prediction Engine.
- FastAPI inference backend.
- Streamlit dashboard.
- Base XGBoost model with temporal cross-validation.
