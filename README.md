<h1 align="center">🏆 World Cup 2026 Prediction Engine</h1>

<p align="center">
  <em>A Segment-Aware Hybrid Ensemble for International Football Prediction</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi" alt="FastAPI">
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker">
  <img src="https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/dbt-FF694B?style=for-the-badge&logo=dbt&logoColor=white" alt="dbt">
</p>

---

## 🧩 Overview

The **World Cup 2026 Prediction Engine** is an end-to-end Machine Learning ecosystem designed to accurately forecast international football fixtures. Moving beyond naive generalist classifiers, this project implements a **Segment-Aware Hybrid Ensemble**, integrating specialized sub-models (such as a *Draw-Specialist*) specifically tuned to predict high-uncertainty matchups.

Built with professional Data Engineering and MLOps practices, the project features a `dbt`-powered medallion data architecture and a completely containerized deployment pipeline ready for cloud hosting.

---

## 🏗 Architecture

### 1. Data Engineering (Medallion Pipeline)
- **Bronze (Raw)**: Historical match data dating back decades.
- **Silver (Cleaned)**: Type-coercion, missing value imputation, and entity resolution (standardizing country names across historical eras).
- **Gold (Feature Store)**: Computes dynamic time-decayed **ELO ratings**, rolling form statistics (Win/Loss/Draw ratios, Goals Scored/Conceded), and explicit neutral-ground flags.

### 2. Modeling Strategy
- **Generalist Predictor**: A rigorously calibrated XGBoost baseline.
- **Draw Specialist**: A secondary binary classifier explicitly trained to correct the habitual under-prediction of ties in classic multinomial logistic environments.
- **Dynamic Routing**: An inference router that activates the Draw Specialist exclusively when the prediction uncertainty (difference between Home vs Away win probabilities) crosses a calculated threshold.

### 3. MLOps & Observability
- **Shadow Modeling**: The API runs an experimental model alongside the production model in real-time, capturing telemetry for A/B offline comparison without impacting end-user predictions.
- **Inference Logging**: Predictions and feature snapshots are persisted to a PostgreSQL database for continuous drift monitoring.

---

## 🚀 Quick Start (Local Development)

### Prerequisites
- Install `uv` (the lightning-fast Python package manager written in Rust).
- No Docker or PostgreSQL is strictly required for the prediction engine to run locally (offline-first architecture).

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/JuanjoRestrepo/worldcup-2026-prediction.git
cd worldcup-2026-prediction

# 2. Configure Environment
cp .env.example .env

# 3. Start the Inference API
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# 4. Start the Dashboard UI (in a separate terminal)
uv run streamlit run src/frontend/app.py
```

Open `http://localhost:8501` to view your frontend dashboard.

---

## 📚 Technical Journey

Interested in the engineering decisions, leakage fixes, and the evolution from Jupyter Notebooks to a production API? Read the full [Development Journey](docs/DEVELOPMENT_JOURNEY.md).

---

## ☁️ Deployment

This project is fully automated for zero-cost cloud deployments:
- **API Backend**: Dockerized FastAPI deployed to [Render](https://render.com) using the included `render.yaml` blueprint.
- **Dashboard UI**: Deployed to [Streamlit Community Cloud](https://share.streamlit.io).
- **Telemetry DB**: [Supabase](https://supabase.com) PostgreSQL footprint for storing `inference_logs`.

*For detailed deployment instructions, see the provided `DEPLOYMENT.md` available in your artifact history.*
