"""
CLI entry point: run the full data pipeline and train both models.

Usage (from project root):
    uv run python main.py
"""

import json
import time
from datetime import datetime
from pathlib import Path

from src.preprocessing import FEATURES, build_dataset
from src.train import MODELS_DIR, prepare_xy, train_model

RAW_PATH = Path("data/raw/nac2018.csv")
PROCESSED_PATH = Path("data/processed/nac2018_cleaned.csv")
METRICS_PATH = MODELS_DIR / "metricas_modelos.json"

TARGETS = {
    "CESAREA": "cesarea_pipeline.pkl",
    "BAJO_PESO": "bajo_peso_pipeline.pkl",
}


def main() -> None:
    print("=" * 60)
    print("  Proyecto Nacimientos DANE 2018 — Pipeline completo")
    print("=" * 60)

    df = build_dataset(RAW_PATH)

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False, encoding="utf-8")
    print(f"\nDataset limpio guardado en {PROCESSED_PATH}")

    metrics = {
        "generado": datetime.now().isoformat(timespec="seconds"),
        "dataset": PROCESSED_PATH.as_posix(),
        "registros_dataset": int(len(df)),
        "variables_modelo": list(FEATURES),
        "validacion": "Validación cruzada estratificada de 5 particiones",
        "modelos": {},
    }
    for target, filename in TARGETS.items():
        start = time.perf_counter()
        _, comparison = train_model(df, target, filename)
        _, y = prepare_xy(df, target)
        best = max(comparison, key=lambda name: comparison[name]["roc_auc"])
        metrics["modelos"][target] = {
            "archivo": f"models/{filename}",
            "modelo_seleccionado": best,
            "muestras_entrenamiento": int(len(y)),
            "tasa_positivos": float(y.mean()),
            "comparacion_cv": {
                name: {key: float(value) for key, value in values.items()}
                for name, values in comparison.items()
            },
            "segundos_entrenamiento": round(time.perf_counter() - start, 1),
        }

    METRICS_PATH.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nMétricas guardadas en {METRICS_PATH}")

    print("\nPipeline completado. Ejecuta la interfaz con:")
    print("  uv run streamlit run app/main.py")
    print("O conversa con el agente de consultas:")
    print("  uv run python -m agent.run")


if __name__ == "__main__":
    main()
