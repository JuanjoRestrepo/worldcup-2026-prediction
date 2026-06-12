# World Cup 2026 Prediction Engine - Roadmap

This document outlines planned improvements and future features for the prediction engine, focusing on advanced analytics, real-time data integration, and infrastructure hardening.

## 1. Advanced Feature Engineering
- **Expected Goals (xG) Integration:** Integrate historical xG, xA, and non-penalty xG from robust sources (e.g., Opta via API-Football).
- **Player-Level Metrics:** Incorporate player availability, injury severity, fatigue metrics (minutes played in top 5 leagues), and individual form tracking.
- **In-Game States:** Build conditional probabilities based on game state (e.g., probability of winning when conceding first).
- **Live Transfermarkt API:** Build a stable backend connector for automated weekly squad value ingestion instead of a static lookup table, utilizing headless residential proxies to bypass Cloudflare.

## 2. Modeling & Calibration Improvements
- **Bayesian Modeling:** Implement a Bayesian hierarchical model using `PyMC` to naturally capture uncertainty and the shrinkage of extreme predictions.
- **Dynamic Bivariate Poisson:** Extend the Poisson regression into a Bivariate Poisson model (using the `skellam` distribution or copulas) to explicitly capture the correlation between home and away goals.
- **Live Updating Odds:** Introduce a real-time calibration mechanism against betting exchange odds (e.g., Betfair) to capture late market movements (injuries, lineup drops).

## 3. System Architecture & DevOps
- **Automated Data Pipelines:** Transition ETL workflows to Apache Airflow or Dagster for fully automated, scheduled data ingestion and training cycles.
- **Feature Store:** Migrate the current CSV/Parquet-based Bronze/Silver/Gold architecture into a formal Feature Store (e.g., Feast) for point-in-time correctness during backtesting.
- **Cloud Deployment:** Containerize the API with Docker and deploy to AWS Elastic Beanstalk, Google Cloud Run, or Azure App Service.
- **Monitoring & Observability:** Implement Prometheus and Grafana to track prediction drift and accuracy across tournaments.

## 4. Frontend & User Experience
- **Interactive Visualisations:** Add SHAP feature importance charts dynamically to the UI so users can see exactly *why* a team is favored.
- **Simulation Mode:** Allow users to simulate an entire tournament bracket (Monte Carlo simulation of group stages and knockouts) using the predictive model.
- **Head-to-Head Deep Dives:** Show historical H2H records, form trends, and clean-sheet probabilities within the prediction screen.

## 5. Security & Code Quality
- **100% Type Coverage:** Enforce strict MyPy coverage across all modules and scripts.
- **Extended Test Suite:** Increase unit and integration testing coverage for the modeling module to 95%+, adding property-based testing (`hypothesis`) for the mathematical transformations.
