# ⚡ Quick Start: Test Inference Logging (5 minutes)

**Just want to see it working? Follow this.**

---

## Step 1: Start PostgreSQL

```bash
# From project root
docker-compose up -d postgres

# Verify it's running
docker-compose ps
# Should show: postgres | postgres:latest | Up
```

## Step 2: Initialize Database Schema

```bash
# Connect and apply schema
psql -U worldcup -h localhost -d worldcup_db -f docker/postgres/init.sql

# Verify monitoring schema created
psql -U worldcup -h localhost -d worldcup_db -c "\dn"
# Should see: monitoring | worldcup
```

## Step 3: Run Unit Tests

```bash
# Activate venv
.venv\Scripts\activate

# Run inference logger tests
pytest tests/test_inference_logger.py -v

# Expected output:
# test_log_prediction PASSED
# test_get_inference_statistics_empty PASSED
# test_get_recent_inferences PASSED
```

## Step 4: Start API Server

```bash
# Terminal 1
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Expected output:
# Uvicorn running on http://0.0.0.0:8000
# Press CTRL+C to quit
```

## Step 5: Test /predict Endpoint (Terminal 2)

```bash
# Make a prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "home_team": "Brazil",
    "away_team": "Argentina",
    "tournament": "2026 FIFA World Cup",
    "neutral": false
  }'

# Response should include:
# {
#   "home_team": "Brazil",
#   "away_team": "Argentina",
#   "predicted_outcome": "win",
#   "class_probabilities": {...},
#   "feature_source": "dbt_latest_team_snapshots"
# }

# ✅ This prediction is now in PostgreSQL!
```

## Step 6: Check Statistics Endpoint

```bash
# Aggregated stats
curl "http://localhost:8000/monitoring/inference-stats?hours=24"

# Should return:
# {
#   "status": "ok",
#   "period_hours": 24,
#   "statistics": {
#     "total_inferences": 1,
#     "unique_matchups": 1,
#     "avg_home_win_prob": 0.XX,
#     ...
#   }
# }
```

## Step 7: Check Recent Inferences Endpoint

```bash
# Audit log
curl "http://localhost:8000/monitoring/recent-inferences?limit=5"

# Should return your prediction with:
# - request_id
# - timestamp_utc
# - home_team, away_team
# - predicted_outcome
# - class_probabilities_json
# - feature_source
# - model_version
```

## Step 8: Direct Database Query

```bash
# Count total predictions
psql -U worldcup -h localhost -d worldcup_db -c \
  "SELECT COUNT(*) FROM monitoring.inference_logs;"
# Should show: 1

# See the record
psql -U worldcup -h localhost -d worldcup_db -c \
  "SELECT home_team, away_team, predicted_outcome FROM monitoring.inference_logs;"
# Should show: Brazil | Argentina | win
```

---

## 🎯 What You Just Verified

✅ PostgreSQL running with monitoring schema  
✅ `inference_logs` table created with data  
✅ Unit tests passing  
✅ API logging predictions automatically  
✅ Statistics aggregation working  
✅ Audit trail endpoint functional  
✅ Production observability layer complete

---

## 🚀 Make 5 More Predictions

```bash
# Try these different matchups:
for team_pair in "Brazil_Argentina" "France_England" "Germany_Spain" "USA_Mexico" "Netherlands_Belgium"; do
  IFS='_' read -r home away <<< "$team_pair"
  curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d "{\"home_team\":\"$home\",\"away_team\":\"$away\"}" -s > /dev/null
  echo "✓ Logged: $home vs $away"
done

# Check aggregated stats again
curl "http://localhost:8000/monitoring/inference-stats?hours=24" | jq '.statistics.total_inferences'
# Should now show: 6+
```

---

## Troubleshooting

| Issue                          | Fix                                        |
| ------------------------------ | ------------------------------------------ |
| `psycopg2: connection refused` | Run `docker-compose up -d postgres`        |
| `monitoring schema not found`  | Run `psql ... -f docker/postgres/init.sql` |
| Port 8000 already in use       | Change: `--port 8001`                      |
| Tests fail with DB error       | Ensure PostgreSQL is running               |
| `/predict` returns 503         | Model artifact may be missing              |

---

## 📖 Next

- **Full testing guide:** [TESTING_INFERENCE_LOGGING.md](TESTING_INFERENCE_LOGGING.md)
- **API documentation:** [INFERENCE_LOGGING_GUIDE.md](INFERENCE_LOGGING_GUIDE.md)
- **Next feature steps:** [ROADMAP_HARDEN_PREDICT.md](ROADMAP_HARDEN_PREDICT.md)
- **Project overview:** [PROJECT_STATUS.md](PROJECT_STATUS.md)

---

**Estimated time:** 5 minutes  
**Result:** 🟢 Production observability working locally
