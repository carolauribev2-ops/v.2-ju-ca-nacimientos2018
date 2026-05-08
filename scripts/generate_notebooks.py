"""Generate notebook JSON files for EDA and ML modeling."""

import json
from pathlib import Path

NB_DIR = Path(__file__).parent.parent / "notebooks"


def md_cell(cell_id: str, text: str) -> dict:
    return {"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": [text]}


def code_cell(cell_id: str, src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": [src],
    }


def make_metadata() -> dict:
    return {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "pygments_lexer": "ipython3",
            "version": "3.12.0",
        },
    }


# ── Notebook 02 — EDA ──────────────────────────────────────────────────────────

EDA_CELLS = [
    md_cell("e-md-01", (
        "# 02 — Analisis Exploratorio de Datos (EDA)\n\n"
        "**Objetivo:** Explorar el dataset de nacimientos limpio para comprender "
        "la distribucion de las variables objetivo y generar hipotesis para el modelado.\n\n"
        "Ejecute primero `01_limpieza_datos.ipynb`."
    )),
    code_cell("e-cd-01", (
        "import sys\n"
        "from pathlib import Path\n"
        "import matplotlib.pyplot as plt\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import seaborn as sns\n\n"
        "PROJECT_ROOT = Path.cwd().parent\n"
        "if str(PROJECT_ROOT) not in sys.path:\n"
        "    sys.path.insert(0, str(PROJECT_ROOT))\n\n"
        "PROCESSED_PATH = PROJECT_ROOT / 'data' / 'processed' / 'nac2018_cleaned.csv'\n"
        "sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)\n"
        "plt.rcParams['figure.dpi'] = 120"
    )),
    code_cell("e-cd-02", (
        "df = pd.read_csv(PROCESSED_PATH)\n"
        "print(f'Dataset: {df.shape[0]:,} filas x {df.shape[1]} columnas')\n"
        "df.head(3)"
    )),
    md_cell("e-md-02", "## 1. Distribucion de variables objetivo"),
    code_cell("e-cd-03", (
        "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n\n"
        "for ax, target, title, labels in zip(\n"
        "    axes,\n"
        "    ['CESAREA', 'BAJO_PESO'],\n"
        "    ['Tipo de Parto', 'Peso al Nacer'],\n"
        "    [['Espontaneo', 'Cesarea'], ['Normal (>=2500g)', 'Bajo peso (<2500g)']],\n"
        "):\n"
        "    counts = df[target].value_counts().sort_index()\n"
        "    bars = ax.bar(labels, counts.values, color=['#4CAF50', '#F44336'],\n"
        "                  alpha=0.85, edgecolor='white')\n"
        "    ax.set_title(title, fontweight='bold', pad=12)\n"
        "    ax.set_ylabel('Numero de registros')\n"
        "    for bar, count in zip(bars, counts.values):\n"
        "        pct = count / len(df) * 100\n"
        "        ax.text(bar.get_x() + bar.get_width() / 2,\n"
        "                bar.get_height() + 500,\n"
        "                f'{count:,}\\n({pct:.1f}%)',\n"
        "                ha='center', va='bottom', fontsize=10)\n\n"
        "plt.suptitle('Distribucion de Variables Objetivo',\n"
        "             fontsize=14, fontweight='bold', y=1.02)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    )),
    md_cell("e-md-03", "## 2. Edad materna y resultados"),
    code_cell("e-cd-04", (
        "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n\n"
        "for ax, target, title in zip(\n"
        "    axes,\n"
        "    ['CESAREA', 'BAJO_PESO'],\n"
        "    ['Edad materna vs Cesarea', 'Edad materna vs Bajo Peso'],\n"
        "):\n"
        "    for val, color, label in [(0, '#4CAF50', 'No'), (1, '#F44336', 'Si')]:\n"
        "        df[df[target] == val]['EDAD_MADRE'].dropna().plot.kde(\n"
        "            ax=ax, color=color, label=label, alpha=0.7, linewidth=2)\n"
        "    ax.axvline(18, color='orange', linestyle='--', alpha=0.7, label='18 anos')\n"
        "    ax.axvline(35, color='purple', linestyle='--', alpha=0.7, label='35 anos')\n"
        "    ax.set_title(title, fontweight='bold')\n"
        "    ax.set_xlabel('Edad de la madre (anos)')\n"
        "    ax.legend(title='Resultado')\n\n"
        "plt.suptitle('Distribucion de Edad Materna por Resultado',\n"
        "             fontsize=14, fontweight='bold')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    )),
    code_cell("e-cd-05", (
        "df['GRUPO_EDAD'] = pd.cut(\n"
        "    df['EDAD_MADRE'],\n"
        "    bins=[0, 17, 24, 34, 45, 100],\n"
        "    labels=['<18', '18-24', '25-34', '35-45', '>45'],\n"
        ")\n"
        "edad_stats = (df.groupby('GRUPO_EDAD', observed=True)\n"
        "               [['CESAREA', 'BAJO_PESO']].mean() * 100)\n"
        "print('Tasa (%) por grupo de edad materna:')\n"
        "edad_stats.round(1)"
    )),
    md_cell("e-md-04", "## 3. Semanas de gestacion (T_GES)"),
    code_cell("e-cd-06", (
        "print('Distribucion de T_GES:')\n"
        "print(df['T_GES'].describe())\n"
        "print('\\nValue counts ordenados:')\n"
        "print(df['T_GES'].value_counts().sort_index())"
    )),
    code_cell("e-cd-07", (
        "fig, ax = plt.subplots(figsize=(12, 4))\n"
        "tasa_ces = df.groupby('T_GES')['CESAREA'].mean() * 100\n"
        "tasa_bp = df.groupby('T_GES')['BAJO_PESO'].mean() * 100\n"
        "ax.plot(tasa_ces.index, tasa_ces.values, 'o-', color='#F44336', label='Cesarea %')\n"
        "ax.plot(tasa_bp.index, tasa_bp.values, 's-', color='#FF9800', label='Bajo Peso %')\n"
        "ax.set_xlabel('T_GES')\n"
        "ax.set_ylabel('Tasa (%)')\n"
        "ax.set_title('Tasas por Semanas de Gestacion', fontweight='bold')\n"
        "ax.legend()\n"
        "plt.tight_layout()\n"
        "plt.show()"
    )),
    md_cell("e-md-05", "## 4. Acceso a salud y area de residencia"),
    code_cell("e-cd-08", (
        "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n\n"
        "area_map = {1: 'Urbana', 2: 'Centro Poblado', 3: 'Rural', -1: 'Desconocido'}\n"
        "regimen_map = {1: 'Contributivo', 2: 'Subsidiado', -1: 'No asegurado'}\n\n"
        "for ax, col, mapping, title in zip(\n"
        "    axes,\n"
        "    ['AREA_RES', 'IDCLASADMI'],\n"
        "    [area_map, regimen_map],\n"
        "    ['Area de Residencia', 'Regimen de Salud'],\n"
        "):\n"
        "    stats = df.groupby(col)[['CESAREA', 'BAJO_PESO']].mean() * 100\n"
        "    stats.index = [mapping.get(i, str(i)) for i in stats.index]\n"
        "    stats.plot(kind='bar', ax=ax,\n"
        "               color=['#F44336', '#FF9800'], alpha=0.85, edgecolor='white')\n"
        "    ax.set_title(f'Tasas por {title}', fontweight='bold')\n"
        "    ax.set_ylabel('Tasa (%)')\n"
        "    ax.legend(['Cesarea', 'Bajo Peso'])\n"
        "    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')\n\n"
        "plt.suptitle('Tasas segun Acceso a Salud', fontsize=14, fontweight='bold')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    )),
    md_cell("e-md-06", "## 5. Control prenatal y paridad"),
    code_cell("e-cd-09", (
        "fig, axes = plt.subplots(1, 2, figsize=(14, 4))\n\n"
        "cons_valid = df[df['NUMCONSUL'].between(0, 20)]\n"
        "tasa_ces = cons_valid.groupby('NUMCONSUL')['CESAREA'].mean() * 100\n"
        "tasa_bp = cons_valid.groupby('NUMCONSUL')['BAJO_PESO'].mean() * 100\n"
        "axes[0].plot(tasa_ces.index, tasa_ces.values, 'o-', color='#F44336', label='Cesarea %')\n"
        "axes[0].plot(tasa_bp.index, tasa_bp.values, 's-', color='#FF9800', label='Bajo Peso %')\n"
        "axes[0].axvline(4, color='gray', linestyle='--', alpha=0.6, label='Min recomendado')\n"
        "axes[0].set_xlabel('Numero de consultas prenatales')\n"
        "axes[0].set_ylabel('Tasa (%)')\n"
        "axes[0].set_title('Control Prenatal vs Resultado', fontweight='bold')\n"
        "axes[0].legend()\n\n"
        "mul_map = {1: 'Unico', 2: 'Gemelar', 3: 'Triple+'}\n"
        "mul_df = df[df['MUL_PARTO'].isin([1, 2, 3])]\n"
        "mul_stats = mul_df.groupby('MUL_PARTO')[['CESAREA', 'BAJO_PESO']].mean() * 100\n"
        "mul_stats.index = [mul_map[i] for i in mul_stats.index]\n"
        "mul_stats.plot(kind='bar', ax=axes[1],\n"
        "               color=['#F44336', '#FF9800'], alpha=0.85, edgecolor='white')\n"
        "axes[1].set_title('Tipo de Gestacion vs Resultado', fontweight='bold')\n"
        "axes[1].set_ylabel('Tasa (%)')\n"
        "axes[1].legend(['Cesarea', 'Bajo Peso'])\n"
        "plt.setp(axes[1].get_xticklabels(), rotation=0)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    )),
    md_cell("e-md-07", "## 6. Top departamentos"),
    code_cell("e-cd-10", (
        "dept_names = {\n"
        "    11: 'Bogota', 5: 'Antioquia', 76: 'Valle', 8: 'Atlantico',\n"
        "    13: 'Bolivar', 25: 'Cundinamarca', 68: 'Santander', 52: 'Narino',\n"
        "    23: 'Cordoba', 19: 'Cauca', 15: 'Boyaca', 73: 'Tolima',\n"
        "    47: 'Magdalena', 41: 'Huila', 20: 'Cesar'\n"
        "}\n"
        "top_depts = df['CODPTORE'].value_counts().nlargest(15).index.tolist()\n"
        "dept_df = (\n"
        "    df[df['CODPTORE'].isin(top_depts)]\n"
        "    .groupby('CODPTORE')[['CESAREA', 'BAJO_PESO']]\n"
        "    .mean() * 100\n"
        ")\n"
        "dept_df.index = [dept_names.get(i, str(i)) for i in dept_df.index]\n"
        "dept_df = dept_df.sort_values('CESAREA', ascending=True)\n\n"
        "fig, ax = plt.subplots(figsize=(10, 7))\n"
        "dept_df.plot(kind='barh', ax=ax,\n"
        "             color=['#F44336', '#FF9800'], alpha=0.85, edgecolor='white')\n"
        "ax.set_title('Tasas por Departamento de Residencia (Top 15)', fontweight='bold')\n"
        "ax.set_xlabel('Tasa (%)')\n"
        "ax.legend(['Cesarea', 'Bajo Peso'])\n"
        "plt.tight_layout()\n"
        "plt.show()"
    )),
    md_cell("e-md-08", "## 7. Matriz de correlacion"),
    code_cell("e-cd-11", (
        "from src.preprocessing import NUMERIC_FEATURES\n\n"
        "corr_cols = NUMERIC_FEATURES + ['CESAREA', 'BAJO_PESO']\n"
        "corr = df[corr_cols].corr()\n\n"
        "fig, ax = plt.subplots(figsize=(9, 7))\n"
        "mask = np.triu(np.ones_like(corr, dtype=bool))\n"
        "sns.heatmap(\n"
        "    corr, annot=True, fmt='.2f', cmap='RdYlGn',\n"
        "    center=0, mask=mask, ax=ax,\n"
        "    linewidths=0.5, square=True, cbar_kws={'shrink': 0.8},\n"
        ")\n"
        "ax.set_title('Matriz de Correlacion', fontweight='bold', pad=15)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    )),
    md_cell("e-md-09", "## 8. Resumen"),
    code_cell("e-cd-12", (
        "from src.preprocessing import FEATURES, NUMERIC_FEATURES\n\n"
        "print('=' * 60)\n"
        "print('RESUMEN EDA')\n"
        "print('=' * 60)\n"
        "print(f'Total registros validos: {len(df):,}')\n"
        "print(f\"Tasa de cesarea:        {df['CESAREA'].mean():.1%}\")\n"
        "print(f\"Tasa de bajo peso:      {df['BAJO_PESO'].mean():.1%}\")\n"
        "print(f\"Edad materna media:     {df['EDAD_MADRE'].mean():.1f} anos\")\n"
        "print(f'Nulos en features:      {df[FEATURES].isnull().sum().sum()}')\n"
        "print('\\nDescriptivas numericas:')\n"
        "df[NUMERIC_FEATURES].describe().round(2)"
    )),
]

# ── Notebook 03 — ML modeling ──────────────────────────────────────────────────

ML_CELLS = [
    md_cell("m-md-01", (
        "# 03 — Modelado Predictivo\n\n"
        "**Objetivo:** Entrenar, evaluar y guardar los modelos de clasificacion para:\n"
        "- Prediccion de cesarea (CESAREA)\n"
        "- Prediccion de bajo peso al nacer (BAJO_PESO)\n\n"
        "Ejecute `01_limpieza_datos.ipynb` primero para generar el dataset procesado."
    )),
    code_cell("m-cd-01", (
        "import sys\n"
        "from pathlib import Path\n"
        "import matplotlib.pyplot as plt\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import seaborn as sns\n"
        "from sklearn.metrics import (\n"
        "    ConfusionMatrixDisplay, RocCurveDisplay,\n"
        "    classification_report, roc_auc_score,\n"
        ")\n"
        "from sklearn.model_selection import train_test_split\n\n"
        "PROJECT_ROOT = Path.cwd().parent\n"
        "if str(PROJECT_ROOT) not in sys.path:\n"
        "    sys.path.insert(0, str(PROJECT_ROOT))\n\n"
        "from src.preprocessing import FEATURES, TARGET_CESAREA, TARGET_BAJO_PESO\n"
        "from src.train import train_model, get_feature_importance\n\n"
        "PROCESSED_PATH = PROJECT_ROOT / 'data' / 'processed' / 'nac2018_cleaned.csv'\n"
        "plt.rcParams['figure.dpi'] = 120"
    )),
    code_cell("m-cd-02", (
        "df = pd.read_csv(PROCESSED_PATH)\n"
        "print(f'Dataset cargado: {df.shape[0]:,} filas')\n"
        "print(f\"Tasa CESAREA:    {df['CESAREA'].mean():.1%}\")\n"
        "print(f\"Tasa BAJO_PESO:  {df['BAJO_PESO'].mean():.1%}\")"
    )),
    md_cell("m-md-02", (
        "## 1. Modelo — Cesarea\n\n"
        "Se evaluan tres clasificadores con validacion cruzada estratificada (5 folds).\n"
        "El mejor segun ROC-AUC se guarda en `models/cesarea_pipeline.pkl`."
    )),
    code_cell("m-cd-03", (
        "pipeline_ces, results_ces = train_model(\n"
        "    df, TARGET_CESAREA, 'cesarea_pipeline.pkl', cv=5\n"
        ")\n"
        "print('\\nComparacion de modelos (Cesarea):')\n"
        "pd.DataFrame(results_ces).T.round(4)"
    )),
    md_cell("m-md-03", "## 2. Evaluacion del modelo de Cesarea"),
    code_cell("m-cd-04", (
        "X = df[FEATURES].dropna()\n"
        "y_ces = df.loc[X.index, TARGET_CESAREA]\n\n"
        "X_train, X_test, y_train, y_test = train_test_split(\n"
        "    X, y_ces, test_size=0.20, stratify=y_ces, random_state=42\n"
        ")\n\n"
        "y_pred = pipeline_ces.predict(X_test)\n"
        "y_prob = pipeline_ces.predict_proba(X_test)[:, 1]\n\n"
        "print(f'ROC-AUC (test): {roc_auc_score(y_test, y_prob):.4f}')\n"
        "print()\n"
        "print(classification_report(y_test, y_pred,\n"
        "      target_names=['Espontaneo', 'Cesarea']))"
    )),
    code_cell("m-cd-05", (
        "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n\n"
        "RocCurveDisplay.from_predictions(\n"
        "    y_test, y_prob, ax=axes[0],\n"
        "    name='Modelo Cesarea', color='#F44336'\n"
        ")\n"
        "axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5)\n"
        "axes[0].set_title('Curva ROC — Cesarea', fontweight='bold')\n\n"
        "ConfusionMatrixDisplay.from_predictions(\n"
        "    y_test, y_pred, ax=axes[1],\n"
        "    display_labels=['Espontaneo', 'Cesarea'],\n"
        "    colorbar=False, cmap='Reds'\n"
        ")\n"
        "axes[1].set_title('Matriz de Confusion — Cesarea', fontweight='bold')\n\n"
        "plt.tight_layout()\n"
        "plt.show()"
    )),
    md_cell("m-md-04", "## 3. Importancia de variables — Cesarea"),
    code_cell("m-cd-06", (
        "fi_ces = get_feature_importance(pipeline_ces)\n"
        "if fi_ces is not None:\n"
        "    top = fi_ces.head(15)\n"
        "    fig, ax = plt.subplots(figsize=(9, 5))\n"
        "    ax.barh(top['feature'][::-1], top['importance'][::-1],\n"
        "            color='#F44336', alpha=0.85, edgecolor='white')\n"
        "    ax.set_title('Top 15 Variables — Modelo Cesarea', fontweight='bold')\n"
        "    ax.set_xlabel('Importancia')\n"
        "    plt.tight_layout()\n"
        "    plt.show()\n"
        "    print(top.to_string(index=False))\n"
        "else:\n"
        "    print('Importancia no disponible para este tipo de modelo.')"
    )),
    md_cell("m-md-05", (
        "## 4. Modelo — Bajo Peso al Nacer\n\n"
        "Mismo proceso de seleccion. El modelo se guarda en `models/bajo_peso_pipeline.pkl`."
    )),
    code_cell("m-cd-07", (
        "pipeline_bp, results_bp = train_model(\n"
        "    df, TARGET_BAJO_PESO, 'bajo_peso_pipeline.pkl', cv=5\n"
        ")\n"
        "print('\\nComparacion de modelos (Bajo Peso):')\n"
        "pd.DataFrame(results_bp).T.round(4)"
    )),
    md_cell("m-md-06", "## 5. Evaluacion del modelo de Bajo Peso"),
    code_cell("m-cd-08", (
        "y_bp = df.loc[X.index, TARGET_BAJO_PESO]\n"
        "_, X_test2, _, y_test2 = train_test_split(\n"
        "    X, y_bp, test_size=0.20, stratify=y_bp, random_state=42\n"
        ")\n\n"
        "y_pred2 = pipeline_bp.predict(X_test2)\n"
        "y_prob2 = pipeline_bp.predict_proba(X_test2)[:, 1]\n\n"
        "print(f'ROC-AUC (test): {roc_auc_score(y_test2, y_prob2):.4f}')\n"
        "print()\n"
        "print(classification_report(y_test2, y_pred2,\n"
        "      target_names=['Normal', 'Bajo Peso']))"
    )),
    code_cell("m-cd-09", (
        "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n\n"
        "RocCurveDisplay.from_predictions(\n"
        "    y_test2, y_prob2, ax=axes[0],\n"
        "    name='Modelo Bajo Peso', color='#FF9800'\n"
        ")\n"
        "axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5)\n"
        "axes[0].set_title('Curva ROC — Bajo Peso', fontweight='bold')\n\n"
        "ConfusionMatrixDisplay.from_predictions(\n"
        "    y_test2, y_pred2, ax=axes[1],\n"
        "    display_labels=['Normal', 'Bajo Peso'],\n"
        "    colorbar=False, cmap='Oranges'\n"
        ")\n"
        "axes[1].set_title('Matriz de Confusion — Bajo Peso', fontweight='bold')\n\n"
        "plt.tight_layout()\n"
        "plt.show()"
    )),
    md_cell("m-md-07", "## 6. Importancia de variables — Bajo Peso"),
    code_cell("m-cd-10", (
        "fi_bp = get_feature_importance(pipeline_bp)\n"
        "if fi_bp is not None:\n"
        "    top = fi_bp.head(15)\n"
        "    fig, ax = plt.subplots(figsize=(9, 5))\n"
        "    ax.barh(top['feature'][::-1], top['importance'][::-1],\n"
        "            color='#FF9800', alpha=0.85, edgecolor='white')\n"
        "    ax.set_title('Top 15 Variables — Modelo Bajo Peso', fontweight='bold')\n"
        "    ax.set_xlabel('Importancia')\n"
        "    plt.tight_layout()\n"
        "    plt.show()\n"
        "    print(top.to_string(index=False))\n"
        "else:\n"
        "    print('Importancia no disponible para este tipo de modelo.')"
    )),
    md_cell("m-md-08", "## 7. Resumen y siguientes pasos"),
    code_cell("m-cd-11", (
        "from pathlib import Path\n\n"
        "models_dir = PROJECT_ROOT / 'models'\n"
        "saved = list(models_dir.glob('*.pkl'))\n"
        "print('Modelos guardados:')\n"
        "for p in saved:\n"
        "    size_mb = p.stat().st_size / 1024 / 1024\n"
        "    print(f'  {p.name}  ({size_mb:.1f} MB)')\n\n"
        "print('\\nPara iniciar la interfaz grafica:')\n"
        "print('  uv run streamlit run app/main.py')"
    )),
]


def build_notebook(cells: list) -> dict:
    return {"cells": cells, "metadata": make_metadata(), "nbformat": 4, "nbformat_minor": 5}


def main() -> None:
    NB_DIR.mkdir(exist_ok=True)
    eda_path = NB_DIR / "02_eda_processor.ipynb"
    ml_path = NB_DIR / "03_modelado_ml.ipynb"

    eda_path.write_text(
        json.dumps(build_notebook(EDA_CELLS), ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"Written: {eda_path}")

    ml_path.write_text(
        json.dumps(build_notebook(ML_CELLS), ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"Written: {ml_path}")


if __name__ == "__main__":
    main()
