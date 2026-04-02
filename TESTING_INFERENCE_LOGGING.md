# Local Testing Guide: Inference Logging System

## Prerequisites

```bash
# Activate venv
.venv\Scripts\activate

# Ensure dependencies installed
pip install -r requirements.txt

# Start PostgreSQL (from project root)
docker-compose up -d postgres
```

## Test 1: Database Schema Setup

```bash
# Connect to PostgreSQL
psql -U worldcup -h localhost -d worldcup_db

# Verify schemas created
\dn

# Expected output:
# Schema      | Owner
# monitoring  | worldcup
# bronze      | worldcup
# silver      | worldcup
# gold        | worldcup

# Check inference_logs table
SELECT table_name FROM information_schema.tables
WHERE table_schema = 'monitoring';

# Exit psql
\q
```

## Test 2: Unit Tests

```bash
# Run inference logger tests
pytest tests/test_inference_logger.py -v

# Run all API tests
pytest tests/test_api.py -v

# Check coverage
pytest tests/test_inference_logger.py --cov=src.modeling.inference_logger
```

## Test 3: Manual API Testing

### Start API Server

```bash
# Terminal 1: Start uvicorn
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Test Endpoints (Terminal 2)

#### Health Check

```bash
curl http://localhost:8000/health
# Response: {"status":"ok","service":"worldcup-api"}
```

#### Configuration

```bash
curl http://localhost:8000/config
# Shows model paths, data dirs, feature source configuration
```

#### Prediction with Auto-Logging

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "home_team": "Brazil",
    "away_team": "Argentina",
    "tournament": "2026 FIFA World Cup",
    "neutral": false
  }'

# Response includes prediction + probabilities
```

**🔍 Check logs were persisted:**

```bash
psql -U worldcup -h localhost -d worldcup_db -c \
  "SELECT COUNT(*) as total_logs FROM monitoring.inference_logs;"
# Expected: 1+ rows
```

#### Inference Statistics (Last 24h)

```bash
curl "http://localhost:8000/monitoring/inference-stats?hours=24"

# Response:
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

#### Recent Inferences (Debug Log)

```bash
curl "http://localhost:8000/monitoring/recent-inferences?limit=5"

# Response:
# {
#   "status": "ok",
#   "count": 1,
#   "inferences": [
#     {
#       "request_id": "Brazil_Argentina_...",
#       "timestamp_utc": "2026-04-02T14:45:00+00:00",
#       "home_team": "Brazil",
#       ...
#     }
#   ]
# }
```

## Test 4: Load Testing

**Simulate multiple predictions quickly:**

```bash
python3 << 'EOF'
import requests
import json
from datetime import datetime

teams = ["Brazil", "Argentina", "France", "England", "Germany"]
api_url = "http://localhost:8000/predict"

for home in teams[:3]:
    for away in teams[1:4]:
        if home != away:
            response = requests.post(
                api_url,
                json={
                    "home_team": home,
                    "away_team": away,
                    "neutral": False,
                    "tournament": "2026 FIFA World Cup"
                }
            )
            if response.status_code == 200:
                print(f"✓ {home} vs {away}")
            else:
                print(f"✗ {home} vs {away}: {response.status_code}")

print("\nLoad test complete!")
EOF
```

**Verify aggregate statistics updated:**

```bash
curl "http://localhost:8000/monitoring/inference-stats?hours=24" | jq '.statistics.total_inferences'
# Should show > 5
```

## Test 5: SQL Verification

**Direct database queries:**

```bash
# Total predictions
psql -U worldcup -h localhost -d worldcup_db -c \
  "SELECT COUNT(*) FROM monitoring.inference_logs;"

# Distribution by outcome
psql -U worldcup -h localhost -d worldcup_db -c \
  "SELECT predicted_outcome, COUNT(*) FROM monitoring.inference_logs
   GROUP BY predicted_outcome;"

# By team
psql -U worldcup -h localhost -d worldcup_db -c \
  "SELECT home_team, COUNT(*) FROM monitoring.inference_logs
   GROUP BY home_team;"

# Feature sources used
psql -U worldcup -h localhost -d worldcup_db -c \
  "SELECT DISTINCT feature_source FROM monitoring.inference_logs;"
```

## Test 6: Error Scenarios

### Missing Model Artifact

```bash
# This should gracefully fail with 503
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"home_team": "Unknown", "away_team": "Team"}'

# Response: 503 Service Unavailable
```

### Invalid Tournament (won't fail, stored as-is)

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "home_team": "Brazil",
    "away_team": "Argentina",
    "tournament": "Custom Tournament 2030"
  }'

# Should succeed and store custom tournament name
```

### Statistics Query with Invalid Hours

```bash
curl "http://localhost:8000/monitoring/inference-stats?hours=1000"
# Response: 422 Unprocessable Entity (hours must be 1-720)
```

## Troubleshooting

| Issue                                            | Solution                                                    |
| ------------------------------------------------ | ----------------------------------------------------------- |
| `psycopg2.OperationalError: connection refused`  | Verify PostgreSQL running: `docker-compose ps`              |
| `monitoring.inference_logs table does not exist` | Apply schema: `psql ... < docker/postgres/init.sql`         |
| `/predict` returns 503                           | Check model exists at path in config, or run training first |
| No logs appearing in DB                          | Check inference_logger logs in uvicorn output for errors    |
| Statistics show "no_data"                        | Make predictions first, then query                          |

## Summary Checklist

- [ ] PostgreSQL running with monitoring schema
- [ ] inference_logs table created with indexes
- [ ] Unit tests passing (test_inference_logger.py)
- [ ] API server running (uvicorn)
- [ ] POST /predict succeeds and returns prediction
- [ ] Logs appear in monitoring.inference_logs
- [ ] GET /monitoring/inference-stats returns stats
- [ ] GET /monitoring/recent-inferences returns logs
- [ ] Load test (5+ predictions) aggregates correctly
