# Predicción de Riesgo Perinatal — DANE Colombia 2018

Juanita Maria Arango Mejia
Carolina Uribe Villa

Sistema de Machine Learning para predecir dos outcomes clínicos críticos en nacimientos
a partir de los microdatos oficiales del DANE (Departamento Administrativo Nacional de Estadística,
Colombia, año 2018).

---

## Problema

Colombia registra anualmente más de 600 000 nacimientos. Dos condiciones neonatales
concentran la mayor carga de morbimortalidad perinatal:

| Condición | Prevalencia (2018) | Impacto clínico |
|---|---|---|
| **Cesárea** | ~45 % de los partos | Recuperación materna más larga, costo hospitalario mayor |
| **Bajo peso al nacer** | ~9 % de los nacimientos | Principal predictor de mortalidad neonatal y morbilidad infantil |

Ambas condiciones son parcialmente predecibles a partir de variables disponibles
**antes** del parto (características maternas, geográficas y de acceso al sistema de salud).
Un modelo que las anticipe puede orientar el cuidado prenatal diferencial.

---

## Arquitectura del Proyecto

```
ju-ca-nacimientos2018/
│
├── data/
│   ├── raw/
│   │   └── nac2018.csv                  # Dataset original DANE (≈649 000 filas, 62 MB)
│   └── processed/
│       └── nac2018_cleaned.csv          # Dataset limpio post-pipeline (~648 000 filas)
│
├── notebooks/
│   ├── 01_limpieza_datos.ipynb          # Limpieza, validación y encoding DANE
│   ├── 02_eda_processor.ipynb           # Análisis Exploratorio de Datos (EDA)
│   └── 03_modelado_ml.ipynb             # Entrenamiento, evaluación y selección de modelos
│
├── src/                                 # Código reutilizable (importable como módulo)
│   ├── __init__.py
│   ├── preprocessing.py                 # Pipeline de limpieza y feature engineering
│   └── train.py                         # Entrenamiento, CV, comparación y persistencia
│
├── app/
│   └── main.py                          # Interfaz gráfica Streamlit (identidad DANE)
│
├── models/                              # Modelos entrenados (generados al ejecutar)
│   ├── cesarea_pipeline.pkl             # Pipeline sklearn completo para cesárea
│   └── bajo_peso_pipeline.pkl           # Pipeline sklearn completo para bajo peso
│
├── scripts/
│   ├── quick_train.py                   # Entrenamiento rápido sin CV (validación UI)
│   └── generate_notebooks.py            # Generador de notebooks desde Python
│
├── main.py                              # CLI: limpieza + feature engineering + entrenamiento
├── pyproject.toml                       # Configuración del proyecto (UV / PEP 518)
├── uv.lock                              # Lock file de dependencias (reproducible)
└── diccionario_datos.md                 # Diccionario de variables DANE
```

### Flujo de datos end-to-end

```
nac2018.csv (crudo, 62 MB, encoding latin-1)
        │
        ▼
  [src/preprocessing.py — clean_data()]
  • Elimina 26 columnas irrelevantes (post-parto, admin, granulares)
  • Imputa nulos administrativos con −1 (código "desconocido")
  • Coerce tipos numéricos (pd.to_numeric con errors='coerce')
  • Filtra registros inválidos (códigos DANE fuera de rango)
  • 648 154 filas retenidas de 649 115 (99.85 %)
        │
        ▼
  [src/preprocessing.py — engineer_features()]
  • CESAREA    = 1 si TIPO_PARTO == 2
  • BAJO_PESO  = 1 si PESO_NAC ∈ {1, 2, 3, 4}   (< 2 500 g)
  • EDAD_RIESGO= 1 si EDAD_MADRE ∈ {1, 7, 8, 9}  (< 15 o ≥ 40 años)
  • PREMATURO  = 1 si T_GES ∈ {1, 2, 3}           (< 37 semanas)
  • PRIMIGESTA = 1 si N_EMB == 1
        │
        ▼
  nac2018_cleaned.csv
        │
        ▼
  [src/train.py — train_model()]
  • Preprocesador ColumnTransformer (numérico + categórico)
  • Validación cruzada estratificada 5-fold sobre 3 candidatos
  • Selección del mejor modelo por ROC-AUC
  • Refit sobre datos completos + persistencia joblib
        │
        ├──► models/cesarea_pipeline.pkl
        └──► models/bajo_peso_pipeline.pkl
                    │
                    ▼
           [app/main.py — Streamlit]
           Formulario → predicción → gauge + factores de riesgo
```

---

## Dataset: Microdatos DANE Nacimientos 2018

**Fuente:** Departamento Administrativo Nacional de Estadística (DANE)
**URL:** https://microdatos.dane.gov.co/index.php/catalog/652
**Registros:** 649 115 nacimientos ocurridos en Colombia durante el año 2018
**Formato:** CSV, encoding `latin-1`, ~62 MB

### ⚠️ Nota crítica sobre la codificación DANE

El DANE usa **códigos enteros categóricos** en columnas que aparentan ser numéricas.
Todos los filtros y la lógica de negocio deben operar sobre estos códigos, no sobre valores reales.

| Variable | Código | Interpretación real |
|---|---|---|
| `PESO_NAC` | 1 | < 1 000 g (extremo bajo peso) |
| `PESO_NAC` | 2 | 1 000 – 1 499 g |
| `PESO_NAC` | 3 | 1 500 – 1 999 g |
| `PESO_NAC` | 4 | 2 000 – 2 499 g |
| `PESO_NAC` | 5 | 2 500 – 2 999 g (normal) |
| `PESO_NAC` | 6 | 3 000 – 3 499 g (normal) |
| `PESO_NAC` | 7 | 3 500 – 3 999 g (normal) |
| `PESO_NAC` | 8 | 4 000 – 4 499 g |
| `PESO_NAC` | 9 | ≥ 4 500 g |
| `EDAD_MADRE` | 1 | < 15 años |
| `EDAD_MADRE` | 2 | 15 – 19 años |
| `EDAD_MADRE` | 3 | 20 – 24 años |
| `EDAD_MADRE` | 4 | 25 – 29 años |
| `EDAD_MADRE` | 5 | 30 – 34 años |
| `EDAD_MADRE` | 6 | 35 – 39 años |
| `EDAD_MADRE` | 7 | 40 – 44 años |
| `EDAD_MADRE` | 8 | 45 – 49 años |
| `EDAD_MADRE` | 9 | ≥ 50 años |
| `T_GES` | 2 | 22 – 27 semanas (muy prematuro) |
| `T_GES` | 3 | 28 – 36 semanas (prematuro) |
| `T_GES` | 4 | 37 – 41 semanas (término, ~80 % de los casos) |
| `T_GES` | 5 | ≥ 42 semanas (postérmino) |
| `T_GES` | 9 | Sin información |
| `TIPO_PARTO` | 1 | Espontáneo / vaginal |
| `TIPO_PARTO` | 2 | Cesárea |
| `TIPO_PARTO` | 3 | Instrumental (fórceps / vacío) |

---

## Análisis Exploratorio de Datos (EDA)

### Variables objetivo

| Target | Positivos | Negativos | Proporción positiva |
|---|---|---|---|
| `CESAREA` | ~290 000 | ~358 000 | ≈ 44.7 % |
| `BAJO_PESO` | ~59 000 | ~589 000 | ≈ 9.1 % |

El dataset presenta **desbalance moderado en BAJO_PESO** (~10:1).
Ambos modelos usan `class_weight='balanced'` para compensarlo.

### Hallazgos clave del EDA

**Edad materna y resultado del parto**
- Madres `< 15 años` (código 1) tienen mayor riesgo de cesárea por desproporción cefalopélvica.
- Madres `≥ 40 años` (códigos 7-9) presentan tasas de cesárea 15-20 puntos porcentuales
  más altas que el grupo de referencia (25-29 años, código 4).

**Semanas de gestación y bajo peso**
- El 80 % de los nacimientos ocurre entre semanas 37-41 (código T_GES = 4).
- Los nacimientos prematuros (T_GES ∈ {2, 3}) concentran el 65-70 % de los casos de bajo peso,
  siendo la prematuridad el principal predictor de esta condición.

**Control prenatal**
- Un número de consultas prenatales `< 4` se asocia con mayor riesgo en ambos outcomes.
- La variable `NUMCONSUL` actúa como proxy del acceso efectivo al sistema de salud.

**Área de residencia y régimen de salud**
- Residencia rural (código 3) correlaciona con mayor bajo peso, posiblemente por menor
  acceso a nutrición y atención obstétrica especializada.
- Sin cobertura en salud (código 5) o régimen desconocido (−1) presentan mayor riesgo.

**Gestación múltiple**
- `MUL_PARTO > 1` (embarazos gemelares o múltiples) es un predictor fuerte de cesárea
  y de bajo peso simultáneamente, dado el menor tiempo de gestación promedio.

**Distribución geográfica**
- Bogotá (código 11) y Antioquia (código 5) concentran el 38 % de los nacimientos.
- La tasa de cesárea varía entre departamentos: San Andrés presenta las tasas más altas;
  departamentos amazónicos tienen mayor proporción de partos vaginales.

---

## Variables del Modelo

### Features de entrada (10 variables)

| Variable | Descripción | Tipo | Rango / Categorías |
|---|---|---|---|
| `EDAD_MADRE` | Grupo de edad de la madre | Numérico (código 1-9) | 1=<15y … 9=50+y |
| `T_GES` | Semanas de gestación al nacer | Numérico (código) | 2-5, 9 |
| `NUMCONSUL` | Consultas prenatales realizadas | Numérico | 0 – 99 |
| `N_HIJOSV` | Número de hijos vivos actualmente | Numérico | 0 – 99 |
| `N_EMB` | Total de embarazos (incluido el actual) | Numérico | 1 – 99 |
| `AREA_RES` | Área de residencia de la madre | Categórico | 1=Urbana, 2=Centro poblado, 3=Rural |
| `IDCLASADMI` | Régimen de salud | Categórico | 1=Contributivo … 5=No asegurado |
| `SEXO` | Sexo del neonato | Categórico | 1=Masculino, 2=Femenino |
| `MUL_PARTO` | Tipo de gestación | Categórico | 1=Simple, 2=Gemelar, 3=Múltiple |
| `CODPTORE` | Código departamento de residencia | Categórico | 5-99 (32 departamentos) |

### Variables objetivo (targets)

| Variable | Derivación | Evento positivo |
|---|---|---|
| `CESAREA` | `TIPO_PARTO == 2` | Parto por cesárea |
| `BAJO_PESO` | `PESO_NAC ∈ {1, 2, 3, 4}` | Peso al nacer < 2 500 g |

### Variables eliminadas (26 columnas)

Las siguientes columnas fueron excluidas por las razones indicadas:

| Columna | Razón de exclusión |
|---|---|
| `APGAR1`, `APGAR2` | Medidas después del nacimiento (data leakage) |
| `TALLA_NAC` | Correlacionada con `PESO_NAC`; disponible solo post-parto |
| `FECHA_NACM` | 47.4 % nulos; no predictiva |
| `OTRO_SIT` | 99.7 % nulos |
| `COD_MUNIC`, `CODMUNRE` | Demasiado granulares; se usa `CODPTORE` |
| `IDPERTET` | Grupo étnico — consideración de sesgo |
| `ANO`, `MES` | Constantes (2018) o sin valor predictivo |
| `ATEN_PAR`, `SIT_PARTO` | Administrativas, no disponibles pre-parto |
| `NIV_EDUM`, `NIV_EDUP` | Educación padre/madre — predictor secundario |
| `T_GES_AGRU_CIE` | Versión agrupada de `T_GES` — redundante |
| `IDHEMOCLAS`, `IDFACTORRH` | Tipo sanguíneo — valor predictivo marginal |

---

## Modelos de Machine Learning

### Estrategia de comparación

Para cada target se evalúan tres clasificadores usando **validación cruzada estratificada de 5 folds**.
La estratificación garantiza que cada fold mantiene la proporción de positivos del conjunto completo.

```
X (10 features) ──► ColumnTransformer ──► Clasificador ──► predicción
                    │ numérico:              │
                    │ SimpleImputer(median)  │ LogisticRegression
                    │ StandardScaler         │ RandomForestClassifier
                    │ categórico:            │ HistGradientBoostingClassifier
                    │ SimpleImputer(mode)    │
                    │ OneHotEncoder          │
                    └────────────────────────┘
```

### Modelos evaluados

#### 1. Regresión Logística (baseline)

- **Hiperparámetros:** `C=0.5`, `max_iter=500`, `class_weight='balanced'`
- **Ventajas:** Coeficientes interpretables, entrenamiento muy rápido, funciona bien
  cuando las relaciones son aproximadamente lineales.
- **Limitaciones:** No captura interacciones no lineales entre variables (por ejemplo,
  la interacción entre edad materna y semanas de gestación).
- **Rol en el proyecto:** Baseline de referencia. Si el modelo lineal ya logra ROC-AUC
  alto, la complejidad adicional no se justifica.

#### 2. Random Forest

- **Hiperparámetros:** `n_estimators=200`, `max_depth=10`, `min_samples_leaf=100`,
  `class_weight='balanced'`, `n_jobs=-1`
- **Ventajas:** Captura no linealidades e interacciones automáticamente.
  Robusto a outliers. Proporciona importancia de variables.
- **Limitaciones:** Más lento de entrenar que Logistic Regression en datasets grandes.
  Requiere más memoria. Puede sobreajustar si `max_depth` es muy alto.
- **Rol en el proyecto:** Modelo intermedio que valida si la no-linealidad aporta.

#### 3. HistGradientBoostingClassifier (modelo final)

- **Hiperparámetros:** `max_iter=300`, `learning_rate=0.05`, `max_depth=6`,
  `min_samples_leaf=100`, `class_weight='balanced'`
- **Ventajas:** Implementación eficiente de boosting tipo LightGBM nativa en sklearn.
  Manejo nativo de valores faltantes. Excelente rendimiento en datasets grandes (>100k filas).
  Regularización implícita por `min_samples_leaf` y `max_depth`.
- **Mecanismo:** Construye árboles secuencialmente, donde cada árbol corrige los errores
  del anterior (gradiente del loss). Histogramas discretizan las features continuas para
  acelerar la búsqueda de splits.
- **Rol en el proyecto:** Modelo de producción — obtiene el mayor ROC-AUC en ambos targets.

### Métricas de evaluación

| Métrica | Descripción | Relevancia en este problema |
|---|---|---|
| **ROC-AUC** | Área bajo la curva ROC | Métrica principal — mide separabilidad sin depender del umbral |
| **Average Precision (AP)** | Área bajo curva Precision-Recall | Crítica para BAJO_PESO por el desbalance de clases |

El **ROC-AUC** va de 0.5 (aleatorio) a 1.0 (perfecto). Un modelo con ROC-AUC > 0.70
en un problema médico real con variables administrativas se considera aceptable.

### Preprocesamiento dentro del pipeline

Todo el preprocesamiento está encapsulado dentro del **sklearn Pipeline**, lo que garantiza:
- **Sin data leakage:** los parámetros del escalador y el encoder se calculan solo sobre
  los datos de entrenamiento de cada fold.
- **Portabilidad:** el archivo `.pkl` contiene el pipeline completo — no se necesita
  preprocesar manualmente para predecir.
- **Reproducibilidad:** `random_state=42` en todos los modelos y splits.

---

## Interfaz Gráfica — Streamlit

La aplicación web (`app/main.py`) implementa la identidad visual institucional del DANE Colombia:

- **Paleta de colores:** magenta `#C0134E`, azul navy `#233979`, dorado `#FEC800`
- **Logo:** ícono de tetero (neonatos) + logotipo DANE
- **Sidebar:** formulario de ingreso de 10 variables clínicas con selectboxes y sliders
- **Panel principal:**
  - Gauge de probabilidad de cesárea (verde < 30 %, ámbar 30-60 %, rojo > 60 %)
  - Gauge de probabilidad de bajo peso al nacer
  - Badge de nivel de riesgo (BAJO / MODERADO / ALTO)
  - Lista de factores de riesgo identificados
  - Tabla de resumen clínico

---

## Instalación

### Prerrequisitos

- **Python 3.12+**
- **UV** — gestor moderno de entornos virtuales para Python

```bash
# Instalar UV en Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Instalar UV en Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Pasos

```bash
# 1. Clonar el repositorio
git clone https://github.com/jmarangom/ju-ca-nacimientos2018.git
cd ju-ca-nacimientos2018

# 2. Crear entorno virtual e instalar dependencias
uv sync

# 3. Colocar el dataset crudo
# Descargar desde: https://microdatos.dane.gov.co/index.php/catalog/652
# Guardar como: data/raw/nac2018.csv
```

### Dependencias principales

| Paquete | Versión mínima | Uso |
|---|---|---|
| `pandas` | 2.2.0 | Manipulación de datos |
| `scikit-learn` | 1.5.0 | Modelos ML, pipelines, métricas |
| `numpy` | 1.26.0 | Álgebra lineal y operaciones vectoriales |
| `matplotlib` | 3.9.0 | Visualizaciones estáticas (notebooks) |
| `seaborn` | 0.13.2 | Visualizaciones estadísticas |
| `plotly` | 5.24.0 | Gráficos interactivos (gauges en app) |
| `streamlit` | 1.40.0 | Interfaz web |
| `joblib` | 1.4.0 | Serialización de modelos |
| `jupyter` | — | Ejecución de notebooks |

---

## Uso

### Opción A — Pipeline completo (recomendado para producción)

Limpieza + feature engineering + entrenamiento con validación cruzada 5-fold:

```bash
uv run python main.py
```

Tiempo estimado: 5-15 minutos (según hardware).

### Opción B — Entrenamiento rápido (validación de interfaz)

Usa HistGBM con 100 iteraciones, sin CV, para obtener modelos en ~1 minuto:

```bash
uv run python -m scripts.quick_train
```

### Opción C — Notebooks paso a paso

```bash
uv run jupyter notebook
```

Ejecutar en orden:
1. `notebooks/01_limpieza_datos.ipynb` — Limpieza y validación
2. `notebooks/02_eda_processor.ipynb` — EDA completo
3. `notebooks/03_modelado_ml.ipynb` — Entrenamiento y evaluación

### Iniciar la interfaz gráfica

```bash
uv run streamlit run app/main.py
```

Abrir en el navegador: **http://localhost:8501**

---

## Estructura del Código

### `src/preprocessing.py`

Módulo central del pipeline de datos. Expone:

```python
build_dataset(raw_path)  # Carga + limpieza + feature engineering en un paso
clean_data(df)           # Solo limpieza
engineer_features(df)    # Solo feature engineering
load_raw_data(path)      # Carga CSV con encoding latin-1
```

Constantes clave:
```python
LOW_WEIGHT_CODES = {1, 2, 3, 4}           # Códigos DANE para peso < 2500g
FEATURES = [                                # 10 variables de entrada
    "EDAD_MADRE", "T_GES", "NUMCONSUL",
    "N_HIJOSV", "N_EMB",                   # numéricas
    "AREA_RES", "IDCLASADMI", "SEXO",
    "MUL_PARTO", "CODPTORE",               # categóricas
]
```

### `src/train.py`

Módulo de entrenamiento. Expone:

```python
train_model(df, target, model_filename, cv=5)  # Entrena y guarda el mejor modelo
build_preprocessor()                            # ColumnTransformer reutilizable
get_feature_importance(pipeline)               # Importancia de variables
```

---

## Limitaciones y Consideraciones Éticas

1. **Datos históricos:** El modelo fue entrenado sobre datos de 2018. Cambios en protocolos
   médicos o demografía pueden afectar la validez prospectiva.

2. **Causalidad vs. correlación:** Las predicciones reflejan asociaciones estadísticas,
   no relaciones causales. No deben usarse como criterio único de decisión clínica.

3. **Variables administrativas:** Varias features (régimen de salud, área de residencia)
   son proxies socioeconómicos. El modelo puede capturar inequidades estructurales
   del sistema de salud colombiano.

4. **Sin validación externa:** Los modelos no han sido validados en datos de otros años
   ni en otras poblaciones. Se recomienda validación prospectiva antes de cualquier
   despliegue clínico.

5. **Grupo étnico excluido:** La variable `IDPERTET` fue excluida deliberadamente para
   evitar que el modelo discrimine por pertenencia étnica.

---

## Fuentes y Referencias

- **Dataset:** DANE — Estadísticas Vitales, Microdatos Nacimientos 2018
  https://microdatos.dane.gov.co/index.php/catalog/652
- **Diccionario de variables DANE:** https://microdatos.dane.gov.co/index.php/catalog/652/data-dictionary
- **OMS — Bajo peso al nacer:** https://www.who.int/topics/low_birthweight
- **scikit-learn HistGradientBoosting:** https://scikit-learn.org/stable/modules/ensemble.html#histogram-based-gradient-boosting
- **UV package manager:** https://docs.astral.sh/uv/

---

## Autores

- **Juanita Arango** — Concepción del proyecto y arquitectura
- **Carolina Uribe** — Desarrollo del pipeline y modelos

Proyecto académico — Especialización en Ciencias de Datos e Inteligencia Artificial.
