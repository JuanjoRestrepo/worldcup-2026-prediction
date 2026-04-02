# Data Layers

This project uses a simple medallion-style layout:

- `raw/`: immutable source files and raw API payloads
- `bronze/`: standardized source extracts
- `silver/`: cleaned, deduplicated, canonical match-level tables
- `gold/`: model-ready feature datasets and prediction-ready artifacts

Tracked data:

- `raw/international_results.csv`

Ignored generated artifacts:

- API JSON snapshots in `raw/`
- generated CSV outputs in `bronze/`, `silver/`, and `gold/`
