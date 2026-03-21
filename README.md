# 1. Definición del proyecto

## Sistema de predicción probabilística y simulación del Mundial 2026

### Objetivo general:
Construir un sistema que estime probabilidades de resultado de partidos internacionales y, a partir de ellas, simule múltiples veces el torneo completo para estimar el desempeño de cada selección.

### Enfoque:
No predecir “ganador único”, sino modelar la incertidumbre del fútbol de forma profesional.

---

# 2. Arquitectura propuesta

## Capa 1. Ingesta de datos
Aquí se obtienen los datos desde varias fuentes.

### Fuentes posibles:

1. Histórico de partidos internacionales
2. Ranking/FIFA-like ratings
3. Resultados recientes de eliminatorias y amistosos
4. Información adicional de selecciones, torneos, sedes y fechas

### Objetivo de esta capa:
traer datos crudos, sin transformar, y conservar trazabilidad.

Salida esperada:

- archivos RAW
- tablas temporales
- logs de carga


## Capa 2. Validación y limpieza
Aquí se corrigen problemas de calidad.

Procesos:

- eliminación de duplicados
- control de nulos
- normalización de nombres de equipos
- validación de fechas
- validación de marcadores
- detección de inconsistencias entre fuentes

### Objetivo de esta capa:
garantizar datos confiables antes del modelado.


## Capa 3. Feature engineering
Aquí se convierte la información bruta en variables útiles.

Features sugeridas:

- diferencia de ranking entre selecciones
- forma reciente
- goles a favor y en contra recientes
- promedio móvil de rendimiento
- localía / neutralidad
- fuerza del rival
- experiencia en torneos
- racha de victorias/empates/derrotas
- rating dinámico tipo Elo

### Objetivo de esta capa:
construir variables explicativas que alimenten el modelo.


## Capa 4. Modelado
Esta es la parte analítica principal.

### Nivel 1: predicción de partidos
Modelos posibles:

- baseline: regresión logística
- modelo de goles: Poisson
- modelo más fuerte: XGBoost / LightGBM
- calibración de probabilidades

### Nivel 2: simulación del torneo
Con las probabilidades del modelo:

- se simulan fase de grupos
- se simulan llaves eliminatorias
- se repiten miles de veces/casos
- se obtienen probabilidades agregadas por selección

Salida:

- probabilidad de avanzar
- probabilidad de llegar a cuartos
- semifinales
- final
- campeón


## Capa 5. Presentación de resultados
Aquí se convierte todo en algo visible y profesional.

Entregables:

- dashboard
- notebook de análisis
- reporte técnico
- README del repositorio
- visualizaciones de probabilidades
- ranking de selecciones por chance de título


# 3. Requisitos del proyecto
Requisitos funcionales

El sistema debe:

1. Obtener datos históricos de partidos internacionales.
2. Integrar datos recientes de selecciones y rankings.
3. Limpiar y estandarizar la información.
4. Construir variables predictoras.
5. Entrenar modelos para estimar resultados.
6. Simular el torneo completo.
7. Generar métricas y visualizaciones.
8. Exportar resultados reproducibles.

Requisitos no funcionales

El sistema debe ser:

1. **Reproducible:** Que cualquiera pueda ejecutar el pipeline.

2. **Modular:** Separado por etapas: ingestión, preparación, modelado, simulación.

3. **Trazable:** Con logs, versiones y documentación.

4. **Escalable:** Capaz de incorporar nuevas fuentes de datos.

5. **Mantenible:** Código limpio, organizado y con estructura profesional.

6. **Interpretativo:** Que no solo prediga, sino que explique por qué.


---


Estructura técnica sugerida del repositorio

```
worldcup-2026-prediction/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── curated/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_simulation.ipynb
│
├── src/
│   ├── ingestion/
│   ├── cleaning/
│   ├── features/
│   ├── modeling/
│   ├── simulation/
│   └── utils/
│
├── reports/
├── tests/
├── README.md
└── requirements.txt
```