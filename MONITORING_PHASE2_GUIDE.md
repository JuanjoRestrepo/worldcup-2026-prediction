# Fase 2: Inteligencia de Negocio y Monitoreo Post-Deploy

**Objetivo**: Validar que el especialista está funcionando correctamente y ajustar thresholds según datos reales.

**Timeline**: Aplicar esto 24-48 horas después del deploy en producción.

---

## 1. Dashboard de Monitoreo (SQL Views)

### Vista 1: Rendimiento de Ensemble Por Segment

```sql
-- Crear vista de performance del ensemble
CREATE VIEW v_ensemble_performance_hourly AS
SELECT
    DATE_TRUNC('hour', timestamp_utc)::timestamp as hour,
    match_segment,
    COUNT(*) as total_predictions,
    SUM(CASE WHEN is_override_triggered THEN 1 ELSE 0 END) as specialist_overrides,
    ROUND(100.0 * SUM(CASE WHEN is_override_triggered THEN 1 ELSE 0 END) / COUNT(*), 2) as override_rate_pct,
    SUM(CASE WHEN predicted_outcome = 'home_win' THEN 1 ELSE 0 END) as home_wins,
    SUM(CASE WHEN predicted_outcome = 'draw' THEN 1 ELSE 0 END) as draws,
    SUM(CASE WHEN predicted_outcome = 'away_win' THEN 1 ELSE 0 END) as away_wins,
    AVG(CAST(class_probabilities_json->>'home_win' AS FLOAT)) as avg_home_win_prob,
    AVG(CAST(class_probabilities_json->>'draw' AS FLOAT)) as avg_draw_prob,
    AVG(CAST(class_probabilities_json->>'away_win' AS FLOAT)) as avg_away_win_prob
FROM monitoring.inference_logs
WHERE timestamp_utc > NOW() - INTERVAL '7 days'
GROUP BY DATE_TRUNC('hour', timestamp_utc), match_segment
ORDER BY hour DESC, match_segment;

-- Consultar últimas 24 horas:
SELECT * FROM v_ensemble_performance_hourly
WHERE hour > NOW() - INTERVAL '24 hours'
ORDER BY hour DESC;
```

**Interpretación**:

- **override_rate_pct**:
  - 0-5% → Especialista casi nunca se activa (threshold muy alto, revisar)
  - 5-15% → Rango ideal (especialista está siendo selectivo)
  - 15-25% → Posible, pero monitorear (threshold quizá bajo)
  - \>25% → Alerta: especialista activándose demasiado

---

### Vista 2: Análisis de Incertidumbre (Max Probability)

```sql
-- Identifica predicciones inciertas (donde especialista debería activarse)
CREATE VIEW v_uncertainty_analysis AS
SELECT
    home_team,
    away_team,
    tournament,
    match_segment,
    predicted_outcome,
    is_override_triggered,
    GREATEST(
        CAST(class_probabilities_json->>'home_win' AS FLOAT),
        CAST(class_probabilities_json->>'draw' AS FLOAT),
        CAST(class_probabilities_json->>'away_win' AS FLOAT)
    ) as max_probability,
    LEAST(
        CAST(class_probabilities_json->>'home_win' AS FLOAT),
        CAST(class_probabilities_json->>'draw' AS FLOAT),
        CAST(class_probabilities_json->>'away_win' AS FLOAT)
    ) as min_probability,
    timestamp_utc
FROM monitoring.inference_logs
WHERE timestamp_utc > NOW() - INTERVAL '24 hours'
ORDER BY max_probability ASC;

-- Revisar predicciones donde max_probability está en rango de threshold:
SELECT * FROM v_uncertainty_analysis
WHERE match_segment = 'worldcup'
  AND max_probability BETWEEN 0.45 AND 0.55
ORDER BY timestamp_utc DESC;
```

**Interpretación**:

- Si `is_override_triggered = TRUE` → Especialista activado correctamente
- Si `is_override_triggered = FALSE` → Verificar si debería haberlo sido

---

### Vista 3: Distribución de Outcomes Por Segment

```sql
-- Compara patrones predichos vs. especialista activado
CREATE VIEW v_outcome_distribution AS
SELECT
    match_segment,
    is_override_triggered,
    predicted_outcome,
    COUNT(*) as count,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY match_segment, is_override_triggered), 2) as pct
FROM monitoring.inference_logs
WHERE timestamp_utc > NOW() - INTERVAL '7 days'
GROUP BY match_segment, is_override_triggered, predicted_outcome
ORDER BY match_segment, is_override_triggered, count DESC;

-- Resultado esperado: Ver si especialista está favoreciendo el draw:
SELECT * FROM v_outcome_distribution
WHERE match_segment = 'friendlies'
ORDER BY is_override_triggered DESC, count DESC;
```

**Interpretación**:

- Especialista DEBE favorecer `draw` cuando se activa (comparar CON `is_override_triggered=TRUE`)
- Si patrón no se ve claro → Revisar la lógica de especialista en `hybrid_ensemble_segment_aware.py`

---

## 2. Shadow Deployment (Validar antes de confiar 100%)

### Para Segments Críticos: Loguer Predicción del Especialista

Si no quieres confiar 100% en el especialista para **Qualifiers** (eliminatorias), usa shadow deployment:

```sql
-- Agregar columna shadow (si aún no existe):
ALTER TABLE monitoring.inference_logs
ADD COLUMN IF NOT EXISTS shadow_specialist_prediction VARCHAR(50);

-- En src/modeling/predict.py, agregar lógica:
```

```python
# En predict_match_outcome() después de ensemble:

# Shadow deployment para Qualifiers (no activa especialista, solo loguea)
if match_segment == "qualifiers":
    try:
        specialist_prediction = ensemble.predict(feature_frame_with_tournament)[0]
        specialist_outcome = outcome_labels[int(encoded_to_outcome[specialist_prediction])]
        # NO activa override (is_override_triggered sigue siendo False)
        # Pero loguea qué hubiera predicho:
    except:
        specialist_outcome = None
else:
    specialist_outcome = None

# Al loguear:
inference_logger.log_prediction(
    ...,
    shadow_specialist_prediction=specialist_outcome,  # Loguar pero no usar
    ...,
)
```

### Validar Shadow Deployment:

```sql
-- Después de 48h con shadow deployment, comparar:
SELECT
    predicted_outcome,
    shadow_specialist_prediction,
    COUNT(*) as matches,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) as pct
FROM monitoring.inference_logs
WHERE match_segment = 'qualifiers'
  AND shadow_specialist_prediction IS NOT NULL
  AND timestamp_utc > NOW() - INTERVAL '2 days'
GROUP BY predicted_outcome, shadow_specialist_prediction
ORDER BY matches DESC;

-- Si diferencias son pequeñas (<5%) → Confianza aumenta
-- Si diferencias son grandes (>15%) → Revisar antes de activar especialista
```

---

## 3. Checklist de Validación Post-Deploy (24-48h)

### 0-4 Horas

- [ ] API en producción recibiendo tráfico
- [ ] Base de datos schema aplicado (migración completada)
- [ ] Logs apareciendo en `monitoring.inference_logs`
- [ ] No hay errores de `is_override_triggered` NULL (siempre debe ser BOOLEAN)

### 4-12 Horas

- [ ] Ejecutar Vista 1 (v_ensemble_performance_hourly)
  - [ ] Validar que `override_rate_pct` esté en rango 5-15% para cada segment
  - [ ] Si >25% → Investigar (threshold muy bajo)
  - [ ] Si ~0% → Investigar (threshold muy alto)

### 12-24 Horas

- [ ] Ejecutar Vista 2 (v_uncertainty_analysis)
  - [ ] Para predicciones inciertas (max_prob 0.40-0.55), verificar que especialista se activó
  - [ ] Si NO se activó → Ajustar threshold en `SegmentConfig`

### 24-48 Horas

- [ ] Si usas shadow deployment para Qualifiers:
  - [ ] Ejecutar comparación de `predicted_outcome` vs `shadow_specialist_prediction`
  - [ ] Si concordancia >95% → Activar especialista en Qualifiers
  - [ ] Si concordancia 80-95% → Revisar 10-20 casos más antes de activar
  - [ ] Si concordancia <80% → Mantener shadow mode, investigar

---

## 4. Ajuste de Thresholds (Si es Necesario)

### Escenario A: Especialista Se Activa Muy Poco (<5%)

**Causa**: `uncertainty_threshold` demasiado alto.

**Acción**:

```python
# En src/modeling/predict.py, SegmentConfig:
"worldcup": SegmentConfig(
    segment_id="worldcup",
    uncertainty_threshold=0.50,  # ← Reducir a 0.45
    draw_conviction_threshold=0.65,
),
```

**Test antes de deploy**:

```bash
# Después de cambiar:
uv run pytest tests/test_hybrid_ensemble_segment_aware.py -v
```

### Escenario B: Especialista Se Activa Muy Poco (>25%)

**Causa**: `uncertainty_threshold` demasiado bajo.

**Acción**: Aumentar threshold (opuesto a Escenario A).

### Escenario C: Especialista Predice Outcomes Incorrectos

**Causa**: `draw_conviction_threshold` mal calibrado.

**Acción**: Revisar lógica en `_compute_override_mask()` en `hybrid_ensemble_segment_aware.py`.

---

## 5. Integración con Observabilidad (Opcional pero Recomendado)

### Exportar Vistas a Grafana/Datadog

```bash
# Si usas Grafana, conectar PostgreSQL y crear dashboard:
# 1. Agregar fuente de datos: PostgreSQL (URL, user, password)
# 2. Crear paneles basados en las vistas SQL
# 3. Alertas:
#    - Alert si override_rate > 25% → Revisar
#    - Alert si no hay predicciones en 1 hora → Downtime
```

### Métricas Clave para Monitorear

- **Availability**: Predicciones por hora (esperar >0)
- **Latency**: `timestamp_utc` diferencia entre request y log
- **Correctness**: Comparar `predicted_outcome` con resultados actuales (una vez que matches se jueguen)
- **Specialist Health**: `override_rate_pct` por segment (esperar 5-15%)

---

## 6. Documentación de Decisiones

### Ejemplo: "Threshold Ajustado el 10 de Abril de 2026"

````markdown
## Cambio: Aumento de Uncertainty Threshold para Friendlies

**Fecha**: 10 de Abril de 2026
**Segment**: friendlies
**Razon**: Override rate subio a 22%, analizar si es correcto

**Before**:

- uncertainty_threshold: 0.35
- override_rate_pct: ~22%

**After**:

- uncertainty_threshold: 0.38
- override_rate_pct (esperado): ~15%

**Validacion**:

```sql
SELECT * FROM v_ensemble_performance_hourly
WHERE match_segment = 'friendlies' AND hour > '2026-04-10'::timestamp;
```
````

**Resultado**: ✅ Sistema funcionando correctamente

```

---

## Resumen: Fase 2 Timeline

| Momento | Acción |
|---------|--------|
| **T+0h** | Deploy Fase 1 completado |
| **T+4h** | Revisar Vista 1 (override rates) |
| **T+12h** | Revisar Vista 2 (predicciones inciertas) |
| **T+24h** | Revisar Vista 3 (distribución de outcomes) |
| **T+48h** | Tomar decisiones: ajustar thresholds o activar shadow deployment |
| **T+1 semana** | Comparar con resultados reales (matches jugados) |

---

**Generado**: 8 de Abril de 2026 - Guía de Monitoreo Post-Deploy (Fase 2)

**Estado**: LISTO PARA EJECUTAR UNA VEZ QUE API ENTRE EN PRODUCCIÓN
```
