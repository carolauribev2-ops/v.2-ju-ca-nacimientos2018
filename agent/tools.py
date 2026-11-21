"""Tools backed by the project's processed data and trained pipelines."""

from collections.abc import Callable, Mapping
from math import isfinite
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.preprocessing import (
    FEATURES,
    TARGET_BAJO_PESO,
    TARGET_CESAREA,
    build_dataset,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "nac2018_cleaned.csv"
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "nac2018.csv"
MODELS_DIR = PROJECT_ROOT / "models"

OUTCOMES = {TARGET_CESAREA, TARGET_BAJO_PESO}
GROUP_COLUMNS = {
    "CODPTORE",
    "EDAD_MADRE",
    "T_GES",
    "AREA_RES",
    "IDCLASADMI",
    "SEXO",
    "MUL_PARTO",
}


def load_birth_data() -> tuple[pd.DataFrame, str]:
    """Load processed data, or build it from the raw CSV using project code."""
    if PROCESSED_DATA_PATH.is_file():
        data = pd.read_csv(PROCESSED_DATA_PATH, low_memory=False)
        source = PROCESSED_DATA_PATH.relative_to(PROJECT_ROOT).as_posix()
    elif RAW_DATA_PATH.is_file():
        data = build_dataset(RAW_DATA_PATH)
        source = "data/raw/nac2018.csv procesado con src.preprocessing.build_dataset"
    else:
        raise FileNotFoundError(
            "No se encontró data/processed/nac2018_cleaned.csv ni "
            "data/raw/nac2018.csv."
        )

    required_columns = set(FEATURES) | OUTCOMES | GROUP_COLUMNS
    missing_columns = sorted(required_columns - set(data.columns))
    if missing_columns:
        raise ValueError(
            f"El dataset {source} no contiene las columnas requeridas: "
            f"{', '.join(missing_columns)}"
        )
    if data.empty:
        raise ValueError(f"El dataset {source} no contiene registros.")
    return data, source


def _validate_outcome(outcome: str) -> None:
    if outcome not in OUTCOMES:
        raise ValueError(f"Resultado no soportado: {outcome!r}.")


def _validate_group_column(group_by: str) -> None:
    if group_by not in GROUP_COLUMNS:
        allowed = ", ".join(sorted(GROUP_COLUMNS))
        raise ValueError(f"Grupo no soportado: {group_by!r}. Opciones: {allowed}.")


def _outcome_summary(data: pd.DataFrame, outcome: str) -> dict[str, Any]:
    _validate_outcome(outcome)
    values = pd.to_numeric(data[outcome], errors="coerce")
    valid = values.isin([0, 1])
    total_valid = int(valid.sum())
    positive = int((values[valid] == 1).sum())
    non_positive = int((values[valid] == 0).sum())
    return {
        "outcome": outcome,
        "positive_count": positive,
        "non_positive_count": non_positive,
        "records_with_valid_outcome": total_valid,
        "unknown_or_missing_count": int(len(data) - total_valid),
        "positive_rate": positive / total_valid if total_valid else None,
    }


def dataset_overview(data: pd.DataFrame, source: str) -> dict[str, Any]:
    """Return observed dataset size and both observed outcome summaries."""
    return {
        "source": source,
        "record_count": int(len(data)),
        "column_count": int(len(data.columns)),
        "outcomes": {
            outcome: _outcome_summary(data, outcome)
            for outcome in sorted(OUTCOMES)
        },
    }


def outcome_summary(
    data: pd.DataFrame, outcome: str, source: str
) -> dict[str, Any]:
    """Return observed counts and prevalence for one binary outcome."""
    return {"source": source, **_outcome_summary(data, outcome)}


def _json_value(value: Any) -> Any:
    if pd.isna(value):
        return "sin_dato"
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def grouped_outcome(
    data: pd.DataFrame, outcome: str, group_by: str, source: str
) -> dict[str, Any]:
    """Return observed outcome counts/rates by a supported DANE code column."""
    _validate_outcome(outcome)
    _validate_group_column(group_by)

    values = pd.to_numeric(data[outcome], errors="coerce")
    work = pd.DataFrame({group_by: data[group_by], outcome: values})
    summary = (
        work.groupby(group_by, dropna=False, sort=True, observed=True)
        .agg(
            record_count=(outcome, "size"),
            valid_outcome_count=(
                outcome,
                lambda group: int(group.isin([0, 1]).sum()),
            ),
            positive_count=(outcome, lambda group: int((group == 1).sum())),
        )
        .reset_index()
    )

    groups = []
    for row in summary.to_dict(orient="records"):
        valid_count = int(row["valid_outcome_count"])
        positive_count = int(row["positive_count"])
        groups.append(
            {
                "group_value": _json_value(row[group_by]),
                "record_count": int(row["record_count"]),
                "valid_outcome_count": valid_count,
                "positive_count": positive_count,
                "unknown_or_missing_count": int(row["record_count"]) - valid_count,
                "positive_rate": (
                    positive_count / valid_count if valid_count else None
                ),
            }
        )

    return {
        "source": source,
        "outcome": outcome,
        "group_by": group_by,
        "groups": groups,
    }


def group_distribution(
    data: pd.DataFrame, group_by: str, source: str
) -> dict[str, Any]:
    """Return record counts for each value of a supported DANE code column."""
    _validate_group_column(group_by)
    counts = (
        data.groupby(group_by, dropna=False, sort=True, observed=True)
        .size()
        .reset_index(name="record_count")
    )
    return {
        "source": source,
        "group_by": group_by,
        "groups": [
            {
                "group_value": _json_value(row[group_by]),
                "record_count": int(row["record_count"]),
            }
            for row in counts.to_dict(orient="records")
        ],
    }


def predict_risks(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Run both saved sklearn pipelines for a complete model feature profile."""
    if not isinstance(profile, Mapping):
        raise TypeError("El perfil debe ser un objeto con las 10 variables del modelo.")

    missing = [feature for feature in FEATURES if feature not in profile]
    extra = sorted(set(profile) - set(FEATURES))
    if missing or extra:
        details = []
        if missing:
            details.append(f"faltan: {', '.join(missing)}")
        if extra:
            details.append(f"no reconocidas: {', '.join(extra)}")
        raise ValueError("Perfil inválido; " + "; ".join(details) + ".")

    numeric_profile: dict[str, float] = {}
    for feature in FEATURES:
        try:
            value = float(profile[feature])
        except (TypeError, ValueError) as error:
            raise ValueError(f"{feature} debe ser numérico.") from error
        if not isfinite(value):
            raise ValueError(f"{feature} debe ser un número finito.")
        numeric_profile[feature] = value

    input_data = pd.DataFrame([numeric_profile], columns=FEATURES)
    model_paths = {
        TARGET_CESAREA: MODELS_DIR / "cesarea_pipeline.pkl",
        TARGET_BAJO_PESO: MODELS_DIR / "bajo_peso_pipeline.pkl",
    }
    probabilities = {}
    for outcome, model_path in model_paths.items():
        if not model_path.is_file():
            raise FileNotFoundError(f"No se encontró el modelo requerido: {model_path}")
        model = joblib.load(model_path)
        classes = list(model.classes_)
        if 1 not in classes:
            raise ValueError(f"El modelo {model_path.name} no tiene clase positiva 1.")
        positive_index = classes.index(1)
        probabilities[outcome] = float(
            model.predict_proba(input_data)[0][positive_index]
        )

    return {
        "model_files": {
            outcome: path.relative_to(PROJECT_ROOT).as_posix()
            for outcome, path in model_paths.items()
        },
        "probabilities": probabilities,
        "feature_names": list(FEATURES),
    }


def build_tools(
    data: pd.DataFrame, source: str
) -> dict[str, Callable[..., dict[str, Any]]]:
    """Create the explicit tool registry consumed by the local agent."""
    return {
        "dataset_overview": lambda: dataset_overview(data, source),
        "outcome_summary": lambda outcome: outcome_summary(data, outcome, source),
        "grouped_outcome": lambda outcome, group_by: grouped_outcome(
            data, outcome, group_by, source
        ),
        "group_distribution": lambda group_by: group_distribution(
            data, group_by, source
        ),
        "predict_risks": predict_risks,
    }
