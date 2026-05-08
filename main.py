"""
CLI entry point: run the full data pipeline and train both models.

Usage (from project root):
    uv run python main.py
"""

from pathlib import Path

from src.preprocessing import build_dataset
from src.train import train_model

RAW_PATH = Path("data/raw/nac2018.csv")
PROCESSED_PATH = Path("data/processed/nac2018_cleaned.csv")


def main() -> None:
    print("=" * 60)
    print("  Proyecto Nacimientos DANE 2018 — Pipeline completo")
    print("=" * 60)

    df = build_dataset(RAW_PATH)

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False, encoding="utf-8")
    print(f"\nDataset limpio guardado en {PROCESSED_PATH}")

    train_model(df, "CESAREA", "cesarea_pipeline.pkl")
    train_model(df, "BAJO_PESO", "bajo_peso_pipeline.pkl")

    print("\nPipeline completado. Ejecuta la interfaz con:")
    print("  uv run streamlit run app/main.py")


if __name__ == "__main__":
    main()
