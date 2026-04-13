# Inference Logging & Monitoring System

## Overview

El sistema de logging de inferencias proporciona **observabilidad completa** para el modelo en producción. Captura cada predicción con timestamp, probabilidades, features usadas, y fuentes, permitiendo auditoría real, debugging, y calibración del modelo.

## Architecture

### Database Schema: `monitoring.inference_logs`

```sql
CREATE TABLE monitoring.inference_logs (
    id SERIAL PRIMARY KEY,
    request_id VARCHAR(255),              -- Unique request identifier
    timestamp_utc TIMESTAMP,               -- When request was made
    requested_match_date DATE,             -- Optional historical as-of serving date
    home_team VARCHAR(255),
    away_team VARCHAR(255),
    neutral BOOLEAN,
    tournament VARCHAR(255),               -- Optional tournament label
    predicted_class INTEGER,               -- -1=away_win, 0=draw, 1=home_win
    predicted_outcome VARCHAR(50),         -- Human-readable outcome
    class_probabilities_json JSONB,        -- Full probability distribution
    feature_snapshot_dates_json JSONB,     -- When features were captured
    feature_source VARCHAR(100),           -- 'dbt', 'postgres', or 'csv'
    model_artifact_path TEXT,              -- Path to model used
    model_version VARCHAR(100),            -- Model version identifier
    persisted_at_utc TIMESTAMP,            -- When log was persisted
    PRIMARY KEY (id)
);
```

**Indexes para performance:**
- `timestamp_utc DESC` → Queries históricas rápidas
- `(home_team, away_team)` → Lookup rápido por matchup
- `predicted_outcome` → Filtrado por clase predicha
- `model_version` → Comparación entre versiones de modelos

### Module: `src/modeling/inference_logger.py`

**`InferenceLogger` class:**
- `log_prediction()` → Loguea una predicción individual
- `get_inference_statistics(hours=24)` → Agregaciones y distribuciones
- `get_recent_inferences(limit=50)` → Debug logging

**Características:**
- ✅ Error resilience: si logging falla, prediction sigue adelante
- ✅ Singleton pattern: `get_inference_logger()` para acceso global
- ✅ Async-safe: usa SQLAlchemy con connection pooling

## API Endpoints

### POST `/predict`
**Enhanced con logging automático:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "home_team": "Brazil",
    "away_team": "Argentina",
    "neutral": false,
    "tournament": "2026 FIFA World Cup"
  }'
```

**Response:**
```json
{
  "home_team": "Brazil",
  "away_team": "Argentina",
  "predicted_class": 1,
  "predicted_outcome": "home_win",
  "class_probabilities": {
    "home_win": 0.72,
    "away_win": 0.15,
    "draw": 0.13
  },
  "neutral": false,
  "tournament": "2026 FIFA World Cup",
  "feature_snapshot_dates": {
    "home": "2026-04-02",
    "away": "2026-04-02"
  },
  "feature_source": "dbt_latest_team_snapshots",
  "model_artifact_path": "/models/match_predictor.joblib"
}
```

→ **Cada predicción se loguea automáticamente en `monitoring.inference_logs`**

### GET `/monitoring/inference-stats?hours=24`
**Agregaciones en tiempo real:**
```bash
curl http://localhost:8000/monitoring/inference-stats?hours=24
```

**Response:**
```json
{
  "status": "ok",
  "period_hours": 24,
  "statistics": {
    "total_inferences": 156,
    "unique_matchups": 48,
    "feature_sources_used": 2,
    "avg_home_win_prob": 0.51,
    "avg_away_win_prob": 0.24,
    "avg_draw_prob": 0.25,
    "home_wins_predicted": 89,
    "away_wins_predicted": 41,
    "draws_predicted": 26,
    "tournaments_predicted": 3,
    "earliest_request": "2026-04-01T10:30:00+00:00",
    "latest_request": "2026-04-02T14:45:00+00:00"
  }
}
```

**Interpretación:**
- `avg_home_win_prob: 0.51` → Modelo no está sesgado
- `home_wins_predicted: 89` vs `away_wins_predicted: 41` → Distribution balanceada
- `unique_matchups` → Cobertura de equipos

### GET `/monitoring/recent-inferences?limit=50`
**Debug & auditing logs:**
```bash
curl http://localhost:8000/monitoring/recent-inferences?limit=10
```

**Response:**
```json
{
  "status": "ok",
  "count": 10,
  "inferences": [
    {
      "request_id": "Brazil_Argentina_1743667500.5",
      "timestamp_utc": "2026-04-02T14:45:00.5+00:00",
      "requested_match_date": "2025-11-18",
      "home_team": "Brazil",
      "away_team": "Argentina",
      "neutral": false,
      "tournament": "2026 FIFA World Cup",
      "predicted_outcome": "home_win",
      "class_probabilities_json": "{\"home_win\": 0.72, \"away_win\": 0.15, \"draw\": 0.13}",
      "feature_source": "dbt_latest_team_snapshots",
      "model_version": "v1.0.0"
    },
    ...
  ]
}
```

**Casos de uso:**
- ✅ **Auditar predicciones específicas** por equipo/fecha
- ✅ **Comparar** probabilidades vs desempeño real post-partido
- ✅ **Detectar anomalías** (ej: team siempre predicho como favorito)
- ✅ **Calibração** (¿El 72% win prob → 72% actual wins?)

## Usage Patterns

### 1. **Monitoreo Básico (Dashboard-Ready)**
```python
# Desde tu dashboard o cron job:
response = requests.get("http://api:8000/monitoring/inference-stats?hours=24")
stats = response.json()

# Alert si hay cambios radicales:
if stats["statistics"]["avg_home_win_prob"] > 0.60:
    log_warning("HOME TEAM BIAS DETECTED")
```

### 2. **Calibración Post-Partido**
```python
# Después de un partido real, comparar:
recent = requests.get("http://api:8000/monitoring/recent-inferences?limit=100").json()

for inf in recent["inferences"]:
    if inf["home_team"] == "Brazil" and inf["away_team"] == "Argentina":
        probs = json.loads(inf["class_probabilities_json"])
        actual_outcome = lookup_actual_score(inf["home_team"], inf["away_team"])
        print(f"Predicted: {probs}, Actual: {actual_outcome}")
        # → Measure calibration error
```

### 3. **Debugging Feature Quality**
```python
# ¿Por qué una predicción fue extraña?
logs = requests.get("http://api:8000/monitoring/recent-inferences?limit=5").json()

for inference in logs["inferences"]:
    print(f"Feature Source: {inference['feature_source']}")
    print(f"Snapshot Dates: {inference['feature_snapshot_dates_json']}")
    # → Verify if features were stale/fresh
```

### 4. **Model Versioning**
```python
# Comparar desempeño entre versiones:
SELECT 
    model_version,
    AVG(CAST(class_probabilities_json->>'win' AS FLOAT)) as avg_win_prob,
    COUNT(*) as prediction_count
FROM monitoring.inference_logs
WHERE timestamp_utc > NOW() - INTERVAL '7 days'
GROUP BY model_version;
```

## Integration with CI/CD

**En tu workflow (ej: `.github/workflows/deploy.yml`):**
```yaml
- name: Initialize monitoring schema
  run: |
    psql -U $DB_USER -h $DB_HOST -d $DB_NAME \
      -f docker/postgres/init.sql
```

## Next Steps: Calibration & Evaluation

1. **Collect inference logs** (3-7 days de predicciones)
2. **Match con resultados reales** (actualizar `actual_outcome` en table)
3. **Calcular:**
   - Calibration curve (predicted prob vs empirical freq)
   - ECE (Expected Calibration Error)
   - Brier Score por tournament/región
4. **Ajustar** threshold de confianza si es necesario

## Testing

```bash
pytest tests/test_inference_logger.py -v
pytest tests/test_api.py -v  # Tests de endpoints de monitoring
```

Fixtures:
- `engine_fixture` → SQLAlchemy engine para test DB
- `inference_logger` → InferenceLogger con test engine

## Troubleshooting

| Problema | Solución |
|----------|----------|
| `monitoring.inference_logs` table not found | Ejecutar `init.sql` o `ensemble schema` |
| Logs no aparecen en DB | Check logs para errores de conexión (non-blocking) |
| `/monitoring/inference-stats` dice "no_data" | Esperar a que se acumulen predicciones (>1 en periodo) |
| Lento en queries grandes | Usar índices: `CREATE INDEX idx_logs_ts ON monitoring.inference_logs(timestamp_utc DESC)` |

## Summary

✅ **Implementado:**
- [x] Schema de `monitoring.inference_logs` con indexes
- [x] `InferenceLogger` class para logging sin bloqueos
- [x] Auto-logging en `/predict` endpoint
- [x] `/monitoring/inference-stats` para agregaciones
- [x] `/monitoring/recent-inferences` para auditing
- [x] Tests para inference logger
- [x] SQL schema en `docker/postgres/init.sql`

**Esto es el foundation para:**
- Model monitoring en producción
- Calibration & evaluation continua
- Feature quality tracking
- Compliance & auditing
