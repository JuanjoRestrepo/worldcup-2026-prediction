# 🎯 Ingestion Architecture: International Data Only

## Problema Identificado

El pipeline original mezclaba:

- ❌ **Clubes**: Barcelona, Milan, Bayern, PSG (Bundesliga, La Liga, Serie A, Premier League, Ligue 1)
- ✅ **Selecciones nacionales**: England, Germany, Spain (World Cup, Qualifiers, Friendlies)

**Impacto crítico:**

- Características incompatibles (club season ≠ international windows)
- Modelo entrenado con datos incoherentes
- No-alineación con objetivo: **Predecir World Cup 2026**

## 🚨 Decisión Arquitectónica (REQUERIDA)

### ✅ SOLO COMPETICIONES INTERNACIONALES

El pipeline filtra y mantiene ÚNICAMENTE:

| Tipo              | Códigos              | Ejemplos                    |
| ----------------- | -------------------- | --------------------------- |
| **World Cup**     | `WC`, `WCQ`          | 2026 matches, qualifiers    |
| **Euro**          | `EC`, `ECQ`          | European Championship       |
| **Copa América**  | `COPA`, `COPAAQ`     | South American championship |
| **African Cup**   | `ACN`, `ACNQ`        | African championship        |
| **AFC Asian Cup** | `AFC`, `AFCQ`        | Asian championship          |
| **CONCACAF**      | `CNL`                | Nations League (3 Americas) |
| **Friendlies**    | `FR`                 | International friendlies    |
| **Qualifiers**    | `UEFAQ`, `CONMEBOLQ` | Continental qualifiers      |

### ❌ EXCLUYE COMPLETAMENTE

Club leagues:

- `PL` (Premier League)
- `BL1` (Bundesliga)
- `PD` (La Liga)
- `SA` (Serie A)
- `FL1` (Ligue 1)
- `PPL` (Primeira Liga)
- `DED` (Eredivisie)
- `BSA` (Campeonato Brasileiro)
- `ECL`/`CL`/`EL` (European club competitions)
- Y todos los demás...

## 📊 Pipeline Flow

```
1. Load Historical CSV
   ↓
2. Load Recent API Data (30 days)
   ↓
3. 🔍 FILTER: Only International (Selecciones Nacionales)
   ├─ Remove all club leagues
   └─ Remove all club competitions
   ↓
4. Save Cleaned Data
   ↓
5. Ready for Feature Engineering → ML Model → WC 2026 Prediction
```

## 🔧 Implementación

### Validador (`international_validator.py`)

```python
def validate_international_match(match: Dict) -> bool:
    """Returns True only for international competitions"""
    competition_code = match.get("competition", {}).get("code", "")
    return competition_code in INTERNATIONAL_COMPETITIONS
```

### Pipeline Actualizado

1. Carga datos históricos
2. Obtiene datos recientes de API
3. **🚨 FILTRA automáticamente** con `filter_international_matches()`
4. Guarda solo datos limpios → `data/raw/api_international_matches_*.json`
5. Log detallado de qué se removió vs. qué se guardó

## ✅ Validación

Tests incluídos (`test_international_validator.py`):

- ✓ World Cup matches KONservados
- ✓ Euro matches conservados
- ✓ Premier League matches removidos
- ✓ Bundesliga matches removidos
- ✓ La Liga matches removidos
- ✓ Friendlies conservados
- ✓ No quedan códigos de clubes en resultado final

## 🎯 Beneficio

Ahora el modelo entrena SOLO con:

- Equipos nacionales
- Dinámicas internacionales
- Información relevante para **World Cup 2026**
- **100% coherencia** en las características

## 📌 Próximos Pasos

1. ✅ Ejecutar pipeline con filtro
2. ✅ Verificar logs (match count before/after)
3. ✅ Validar JSON resultante
4. → Data processing & feature engineering
5. → Model training con datos limpios
