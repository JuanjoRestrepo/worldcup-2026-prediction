# World Cup 2026 Model Retraining Walkthrough

This walkthrough details the execution and results of retraining the World Cup 2026 prediction model using the newly automated data ingestion and promotion gating pipeline.

## 1. Automated Execution Pipeline

The pipeline successfully executed the following steps end-to-end:

1. **Automated Ingestion**: `csv_client.py` gracefully downloaded the newest raw data from the `martj42` GitHub repository (~49k rows, extending to recent 2026 matches).
2. **Missing/Future Data Handling**: We discovered that the upstream CSV contains future `World Cup 2026` schedule fixtures with `NA` scores. The ingestion pipeline was dynamically fixed to proactively drop these future fixtures before strict data-contract schema validation occurs.
3. **Training & Exporting**: The training pipeline digested the updated `features_dataset.csv`. Instead of blindly overwriting production, it emitted `match_predictor_v2_apr2026.joblib` and updated the `match_predictor_evaluation_report.json` with the new v2 metrics.

## 2. Model Comparison & Promotion Gate

Following the model retraining, the `reporting_comparison.py` tool generated a side-by-side evaluation against the historic Jan 2026 checkpoint (`v1`).

The automated Promotion Gate correctly identified strict model regression and triggered a **`KEEP_V1`** decision.

### Comparison Results (v1 vs v2)

| Metric       | v1 (Jan 2026) | v2 (Apr 2026) | Delta   | Status            |
| :----------- | :------------ | :------------ | :------ | :---------------- |
| **Accuracy** | 0.5778        | 0.5615        | -0.0163 | 📉 Regression     |
| **Macro F1** | 0.5165        | 0.5014        | -0.0151 | 📉 Regression     |
| **Log-Loss** | 0.8952        | 0.9245        | +0.0294 | 📈 Higher (Worse) |

- The model suffered widespread accuracy degradation from the new sample inclusion, failing the `EQUIVALENCE_TOLERANCE` threshold (±0.002) completely.
- The degradation was pervasive, manifesting across all competition tiers (World Cup, Friendlies) and all confederations (UEFA, CONMEBOL, etc.).

## 3. Final Architecture Status

- The previous Champion Model (`v1`) safely remains exactly where it was and continues to serve the prediction API without interference.
- The new `v2_apr2026` joblib artifact has been cleanly archived.
- The Git CI/CD system (`retrain.yml`) is set up to automate this offline comparison check monthly, protecting the deployed app from silent degradation exactly like the one caught above.

All unit tests pass and code quality checks (Ruff, MyPy) are clean. The prediction API deployment is strongly hardened.
