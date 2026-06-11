# Roadmap: World Cup 2026 Prediction Engine

This roadmap outlines strategic directions and future improvements for the ML engine, feature engineering, and the overall application architecture to achieve top-tier performance for predicting the 2026 World Cup.

---

## 1. Advanced Data Sources & Integrations

The current model relies exclusively on historical match results (goals, tournaments, dates). To leap forward in predictive power, integrating more granular data is the top priority:

* **Event-Level Data Integration:** Integrate Opta or StatsBomb event data for granular features (e.g., non-penalty xG, deep completions, passes into the penalty area). This will capture tactical dominance that result-only data misses.
* **Lineups and Roster Quality:** Scrape Transfermarkt values or FIFA ratings to calculate the exact market value / talent rating of the starting XI, adjusting for injured key players.
* **Travel & Climate Context:** For WC 2026, travel distances between North American cities, altitude (e.g., Mexico City), and humidity can significantly affect performance. Build features mapping these geographical factors to team resilience.

## 2. Model & Algorithm Enhancements

* **Deep Learning for Tabular Data (TabNet / FT-Transformer):** Although tree-based models (XGBoost/LightGBM) currently dominate tabular data, experimenting with modern tabular deep learning architectures might capture complex non-linear feature interactions (e.g., how specific formations counter others).
* **Multi-Task Learning:** Train a neural network to simultaneously predict the match outcome, the exact scoreline, and the number of corners/cards. The shared representations often improve the primary classification task.
* **Dynamic Time Warping (DTW) on Form:** Instead of basic EWMA, use DTW to find historical periods where a team had a similar trajectory to their current form, to better predict their next result.
* **Bivariate Poisson / Dixon-Coles Improvements:** The current Poisson regressors assume home and away goals are independent. Upgrading to a Bivariate Poisson or a full Dixon-Coles model with a low-scoring correlation parameter ($\rho$) will better model the 0-0, 1-0, 0-1, and 1-1 clusters.

## 3. Operations & MLOps

* **Automated Data Quality & Drift Detection:** Implement `Great Expectations` or `Pandera` in the pipeline to catch schema changes in API responses and detect data drift (e.g., if goal-scoring rates suddenly jump globally).
* **Continuous Training (CT) Pipeline:** Fully automate the training loop via GitHub Actions or Airflow so that when new match results arrive, the model retrains, evaluates via the Champion vs Challenger gate, and automatically redeploys if superior.
* **Feature Store Integration:** Transition `processing_pipeline.py` into an organized Feature Store (e.g., Feast or Hopsworks) to serve pre-calculated ELO and form metrics to the real-time inference API at single-digit millisecond latency.

## 4. UI / UX & Frontend

* **Interactive "What-If" Scenario Builder:** Allow the user to manually override features in the UI. For example, "What if Messi doesn't play?" (simulated by dropping Argentina's ELO by X points).
* **xG vs Actual Goals Visualization:** Add charts showing teams that are over-performing or under-performing their Expected Goals, providing narrative context for the predictions.
* **Explainable AI (XAI) Dashboard:** Surface SHAP values directly in the Streamlit app. For every match prediction, show a waterfall chart explaining *why* the model chose the outcome (e.g., "+15% due to Home Advantage, -5% due to poor H2H record").
