"""
Entrenamiento rapido para validacion de la interfaz grafica.
Usa LogisticRegression + 2 folds para obtener modelos en ~1 minuto.
Para produccion, usa: uv run python main.py
"""

from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.preprocessing import (
    build_dataset,
    FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES,
    TARGET_CESAREA, TARGET_BAJO_PESO,
)

RAW_PATH = Path("data/raw/nac2018.csv")
MODELS_DIR = Path("models")


def build_quick_pipeline():
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    prep = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    clf = HistGradientBoostingClassifier(
        max_iter=100, learning_rate=0.1, max_depth=5,
        class_weight="balanced", random_state=42,
    )
    return Pipeline([("prep", prep), ("clf", clf)])


def main():
    print("Cargando y limpiando datos...")
    df = build_dataset(RAW_PATH)

    MODELS_DIR.mkdir(exist_ok=True)

    for target, filename in [
        (TARGET_CESAREA, "cesarea_pipeline.pkl"),
        (TARGET_BAJO_PESO, "bajo_peso_pipeline.pkl"),
    ]:
        print(f"\nEntrenando modelo: {target}...")
        X = df[FEATURES].copy()
        y = df[target].copy()
        valid = X.notna().all(axis=1) & y.notna()
        X, y = X[valid], y[valid]
        print(f"  Muestras: {len(X):,}  |  Positivos: {y.mean():.1%}")

        pipeline = build_quick_pipeline()
        pipeline.fit(X, y)

        out = MODELS_DIR / filename
        joblib.dump(pipeline, out)
        print(f"  Guardado: {out}")

    print("\nModelos listos. Ejecuta la app:")
    print("  uv run streamlit run app/main.py")


if __name__ == "__main__":
    main()
