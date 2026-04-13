# Next Priority: Hardening the /predict Endpoint

**Estimated Effort:** 2-3 hours  
**Impact:** High (production-readiness)  
**Portfolio Value:** Medium (expected feature)

---

## Current State

```python
@app.post("/predict")
def predict(request: PredictionRequest) -> PredictionResponse:
    prediction = predict_match_outcome(
        home_team=request.home_team,
        away_team=request.away_team,
        tournament=request.tournament,
        neutral=request.neutral,
    )
    # ✅ Auto-logs to monitoring.inference_logs
    return PredictionResponse(**prediction)
```

**What works:**

- ✅ Basic prediction
- ✅ Automatic logging
- ✅ Error handling for missing model/teams

**What's missing:**

- ❌ Team name aliases (USA vs "United States")
- ❌ Historical predictions (specific match_date)
- ❌ Stale feature detection (warn if snapshots >30d old)
- ❌ Better error messages for data quality issues

---

## Feature 1: Team Name Aliases

**Problem:**

```bash
# This fails:
curl -X POST http://localhost:8000/predict \
  -d '{"home_team":"USA","away_team":"Mexico"}'
# → "team USA not found in snapshots"

# But this works:
curl -X POST http://localhost:8000/predict \
  -d '{"home_team":"United States","away_team":"Mexico"}'
```

**Solution:**

### 1. Create alias mapping file

File: `src/config/team_aliases.py`

```python
"""Team name aliases for prediction requests."""

TEAM_ALIASES = {
    # Common abbreviations
    "USA": "United States",
    "England": "England",
    "Korea": "South Korea",
    "South_Korea": "South Korea",
    "Czech": "Czechia",
    "Czechia": "Czechia",
    "Ivory_Coast": "Côte d'Ivoire",
    "Cote_d_Ivoire": "Côte d'Ivoire",
    # Add more as needed
}

def normalize_team_name(team_name: str) -> str:
    """
    Normalize team name using alias mapping.

    Args:
        team_name: Raw team name from request

    Returns:
        Normalized team name
    """
    return ALIASES.get(team_name, team_name)
```

### 2. Update prediction request handling

File: `src/api/main.py`

```python
from src.config.team_aliases import normalize_team_name

@app.post("/predict")
def predict(request: PredictionRequest) -> PredictionResponse:
    # Normalize team names
    home_team = normalize_team_name(request.home_team)
    away_team = normalize_team_name(request.away_team)

    try:
        prediction = predict_match_outcome(
            home_team=home_team,
            away_team=away_team,
            tournament=request.tournament,
            neutral=request.neutral,
        )
        # ... logging code
        return PredictionResponse(**prediction)
    except ValueError as exc:
        # More helpful error messages
        if "not found" in str(exc).lower():
            raise HTTPException(
                status_code=422,
                detail=f"Team not found: {request.home_team} or {request.away_team}. "
                       f"Check spelling or use full team name (e.g., 'United States' not 'USA')"
            )
        raise
```

### 3. Test

```bash
pytest tests/test_api.py -k "test_predict_team_aliases"
```

---

## Feature 2: Historical Predictions (match_date parameter)

**Problem:**
Currently, `/predict` always uses TODAY's features. But what if you want to predict Brazil vs Argentina **as of April 1, 2026** (using features from that date)?

**Solution:**

### 1. Update PredictionRequest

```python
from datetime import date

class PredictionRequest(BaseModel):
    home_team: str = Field(..., min_length=1)
    away_team: str = Field(..., min_length=1)
    tournament: str | None = None
    neutral: bool = False
    match_date: date | None = None  # NEW: optional historical prediction
```

### 2. Create history loading function

File: `src/modeling/serving_store.py`

```python
def load_team_snapshots_at_date(match_date: date) -> pd.DataFrame:
    """
    Load team feature snapshots as they were on a specific date.

    Queries the dbt historical snapshots table to get features
    from a specific point in time.
    """
    query = f'''
    SELECT * FROM "{settings.DBT_BASE_SCHEMA}_gold"."{DBT_GOLD_TEAM_SNAPSHOTS_TABLE}"
    WHERE snapshot_date = %s
    '''
    engine = get_sqlalchemy_engine()
    return pd.read_sql_query(query, con=engine, params=[match_date])
```

### 3. Update predict endpoint

```python
@app.post("/predict")
def predict(request: PredictionRequest) -> PredictionResponse:
    home_team = normalize_team_name(request.home_team)
    away_team = normalize_team_name(request.away_team)

    # Determine feature source date
    feature_date = request.match_date or date.today()

    try:
        prediction = predict_match_outcome(
            home_team=home_team,
            away_team=away_team,
            tournament=request.tournament,
            neutral=request.neutral,
            feature_date=feature_date,  # NEW parameter
        )
        # ... rest of logic
```

### 4. Test

```bash
# Historical prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "home_team": "Brazil",
    "away_team": "Argentina",
    "match_date": "2026-03-15"
  }'
```

---

## Feature 3: Stale Feature Detection

**Problem:**
If features haven't been updated in 30 days, prediction quality suffers (teams' form, injuries, etc. change).

**Solution:**

### 1. Add validation to InferenceLogger

```python
# In src/modeling/inference_logger.py

from datetime import datetime, timedelta, timezone

def validate_feature_freshness(
    feature_dates: dict[str, str],
    max_age_days: int = 30,
) -> tuple[bool, str]:
    """
    Check if features are fresh.

    Returns:
        (is_fresh, warning_message)
    """
    now = datetime.now(timezone.utc)

    for team, date_str in feature_dates.items():
        feature_date = datetime.fromisoformat(date_str)
        age_days = (now - feature_date).days

        if age_days > max_age_days:
            return False, f"Features for {team} are {age_days} days old"

    return True, ""
```

### 2. Add to response

```python
class PredictionResponse(BaseModel):
    # ... existing fields
    feature_freshness: dict[str, str | bool] = Field(
        default={},
        description="Feature age warnings"
    )
```

### 3. Use in predict endpoint

```python
@app.post("/predict")
def predict(request: PredictionRequest) -> PredictionResponse:
    # ... existing code

    prediction = predict_match_outcome(...)

    # Check feature freshness
    is_fresh, warning = validate_feature_freshness(
        prediction["feature_snapshot_dates"]
    )

    response = PredictionResponse(**prediction)
    if not is_fresh:
        response.feature_freshness = {
            "is_fresh": False,
            "warning": warning
        }

    return response
```

---

## Testing Strategy

### 1. Unit Tests

File: `tests/test_api.py`

```python
def test_predict_with_team_alias():
    response = client.post("/predict", json={
        "home_team": "USA",
        "away_team": "Mexico"
    })
    assert response.status_code == 200
    assert "United States" in response.json()["home_team"]

def test_predict_historical_date():
    response = client.post("/predict", json={
        "home_team": "Brazil",
        "away_team": "Argentina",
        "match_date": "2026-03-15"
    })
    assert response.status_code == 200

def test_predict_stale_features_warning():
    response = client.post("/predict", json={
        "home_team": "Brazil",
        "away_team": "Argentina"
    })
    if response.status_code == 200:
        assert "feature_freshness" in response.json()

def test_predict_invalid_team_name():
    response = client.post("/predict", json={
        "home_team": "Fakelandia",
        "away_team": "Nonexistia"
    })
    assert response.status_code == 422
    assert "not found" in response.json()["detail"].lower()
```

### 2. Integration Test

```bash
# Run all with live API
uvicorn src.api.main:app --reload &
pytest tests/test_api.py -v
```

---

## Implementation Checklist

- [ ] Create `src/config/team_aliases.py`
- [ ] Update `PredictionRequest` model
- [ ] Implement `normalize_team_name()` in predict endpoint
- [ ] Add `match_date` optional parameter
- [ ] Implement `load_team_snapshots_at_date()` in serving_store
- [ ] Add stale feature detection function
- [ ] Update `PredictionResponse` with `feature_freshness`
- [ ] Write unit tests for all new features
- [ ] Test with real API locally
- [ ] Update [INFERENCE_LOGGING_GUIDE.md](INFERENCE_LOGGING_GUIDE.md) with examples
- [ ] Update API docs docstrings

---

## Expected Result

**Before:**

```bash
curl -X POST http://localhost:8000/predict \
  -d '{"home_team":"USA"}'
# → 422 Team USA not found
```

**After:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "home_team": "USA",
    "away_team": "Mexico",
    "match_date": "2026-03-15"
  }'

# Response:
{
  "home_team": "United States",
  "away_team": "Mexico",
  "predicted_outcome": "win",
  "class_probabilities": {...},
  "feature_freshness": {
    "is_fresh": true,
    "warning": null
  }
}
```

---

## Why This Matters

✅ **Production-ready (handles real-world usage)**

- Users don't always know exact team names
- Historical predictions enable backtesting
- Stale data warnings prevent bad surprises

✅ **Portfolio value**

- Shows you think about edge cases
- Demonstrates data quality mindset
- Better than 10 endpoints that don't work well

**Next after this:** CI/CD (automate testing)
