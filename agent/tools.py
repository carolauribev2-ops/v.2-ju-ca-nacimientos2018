"""Tools backed by the project's processed data and trained pipelines.

Every number the agent reports must come from one of these functions. They are
deterministic: the same data and arguments always return the same evidence.
"""

import json
from collections.abc import Callable, Mapping
from functools import lru_cache
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

from .catalog import DEPARTAMENTOS, VARIABLES, label_for

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "nac2018_cleaned.csv"
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "nac2018.csv"
MODELS_DIR = PROJECT_ROOT / "models"
METRICS_PATH = MODELS_DIR / "metricas_modelos.json"
MODEL_LABELS = {
    "logistic_regression": "Regresión logística",
    "random_forest": "Random Forest",
    "hist_gradient_boosting": "HistGradientBoosting",
}

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
# Present in datasets built with the current preprocessing; optional for older files.
OPTIONAL_GROUP_COLUMNS = {"T_GES_AGRU_CIE"}
ALL_GROUP_COLUMNS = GROUP_COLUMNS | OPTIONAL_GROUP_COLUMNS
FILTER_COLUMNS = ALL_GROUP_COLUMNS | OUTCOMES

# Allowed values for each model feature (codes seen in the training data).
PROFILE_RULES: dict[str, tuple[str, Any]] = {
    "EDAD_MADRE": ("set", set(range(1, 10))),
    "T_GES": ("set", {1, 2, 3, 4, 5, 6, 9}),
    "NUMCONSUL": ("range", (0, 25)),
    "N_HIJOSV": ("range", (1, 20)),
    "N_EMB": ("range", (1, 18)),
    "AREA_RES": ("set", {1, 2, 3, 9, -1}),
    "IDCLASADMI": ("set", {1, 2, 3, 4, 5, 9, -1}),
    "SEXO": ("set", {1, 2}),
    "MUL_PARTO": ("set", {1, 2, 3, 4, 9}),
    "CODPTORE": ("set", set(DEPARTAMENTOS)),
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


# ── Validation helpers ─────────────────────────────────────────────────────────


def _validate_outcome(outcome: str) -> None:
    if outcome not in OUTCOMES:
        raise ValueError(
            f"Resultado no soportado: {outcome!r}. Opciones: CESAREA, BAJO_PESO."
        )


def _validate_group_column(data: pd.DataFrame, group_by: str) -> None:
    if group_by not in ALL_GROUP_COLUMNS:
        allowed = ", ".join(sorted(ALL_GROUP_COLUMNS))
        raise ValueError(f"Grupo no soportado: {group_by!r}. Opciones: {allowed}.")
    if group_by not in data.columns:
        raise ValueError(
            f"La columna {group_by} no está en el dataset cargado. Regenera "
            "data/processed con `uv run python main.py`."
        )


def _as_int_code(value: Any, variable: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{variable}: el código debe ser un entero, no {value!r}.")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{variable}: el código {value!r} no es numérico.") from error
    if not isfinite(number) or not number.is_integer():
        raise ValueError(f"{variable}: el código {value!r} debe ser un entero.")
    return int(number)


def _normalize_filters(filters: Any) -> dict[str, list[int]]:
    """Accept {"VAR": [codes]} or [{"variable": "VAR", "codes": [codes]}]."""
    if filters is None:
        return {}
    if isinstance(filters, Mapping):
        items = list(filters.items())
    elif isinstance(filters, list):
        items = []
        for entry in filters:
            if not isinstance(entry, Mapping) or "variable" not in entry:
                raise ValueError(
                    "Cada filtro debe tener la forma "
                    '{"variable": "CODPTORE", "codes": [5]}.'
                )
            items.append((entry["variable"], entry.get("codes", [])))
    else:
        raise ValueError("Los filtros deben ser un objeto o una lista de filtros.")

    normalized: dict[str, list[int]] = {}
    for variable, codes in items:
        if variable not in FILTER_COLUMNS:
            allowed = ", ".join(sorted(FILTER_COLUMNS))
            raise ValueError(
                f"No se puede filtrar por {variable!r}. Opciones: {allowed}."
            )
        if not isinstance(codes, (list, tuple, set)):
            codes = [codes]
        if not codes:
            raise ValueError(f"El filtro de {variable} no tiene códigos.")
        values = [_as_int_code(code, variable) for code in codes]
        existing = normalized.setdefault(variable, [])
        existing.extend(code for code in values if code not in existing)
    return normalized


def _outcome_values(data: pd.DataFrame, outcome: str) -> pd.Series:
    """Binary outcome as numbers; unknown labels become NaN."""
    values = pd.to_numeric(data[outcome], errors="coerce")
    if outcome == TARGET_BAJO_PESO and "PESO_NAC" in data.columns:
        # PESO_NAC = 9 is "Sin información" in the DANE dictionary.
        unknown_weight = pd.to_numeric(data["PESO_NAC"], errors="coerce") == 9
        values = values.mask(unknown_weight)
    return values.where(values.isin([0, 1]))


def _apply_filters(
    data: pd.DataFrame, filters: Any
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    normalized = _normalize_filters(filters)
    mask = pd.Series(True, index=data.index)
    applied = []
    for variable, codes in normalized.items():
        if variable not in data.columns:
            raise ValueError(
                f"La columna {variable} no está en el dataset cargado; no se puede "
                "filtrar por ella."
            )
        if variable in OUTCOMES:
            column = _outcome_values(data, variable)
        else:
            column = pd.to_numeric(data[variable], errors="coerce")
        mask &= column.isin(codes)
        applied.append(
            {
                "variable": variable,
                "variable_label": VARIABLES.get(variable, {}).get("nombre", variable),
                "codes": codes,
                "labels": [label_for(variable, code) for code in codes],
            }
        )
    return data[mask], applied


# ── Tools ──────────────────────────────────────────────────────────────────────


def _outcome_summary(data: pd.DataFrame, outcome: str) -> dict[str, Any]:
    _validate_outcome(outcome)
    values = _outcome_values(data, outcome)
    valid = values.notna()
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


def dataset_overview(
    data: pd.DataFrame, source: str, filters: Any = None
) -> dict[str, Any]:
    """Return the observed record count and both outcome summaries (optionally filtered)."""
    subset, applied = _apply_filters(data, filters)
    return {
        "source": source,
        "filters": applied,
        "record_count": int(len(subset)),
        "total_records_in_dataset": int(len(data)),
        "column_count": int(len(data.columns)),
        "outcomes": {
            outcome: _outcome_summary(subset, outcome) for outcome in sorted(OUTCOMES)
        },
    }


def outcome_summary(
    data: pd.DataFrame, outcome: str, source: str, filters: Any = None
) -> dict[str, Any]:
    """Return observed counts and prevalence for one binary outcome."""
    subset, applied = _apply_filters(data, filters)
    return {
        "source": source,
        "filters": applied,
        "record_count": int(len(subset)),
        **_outcome_summary(subset, outcome),
    }


def _json_value(value: Any) -> Any:
    if pd.isna(value):
        return "sin_dato"
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def grouped_outcome(
    data: pd.DataFrame,
    outcome: str,
    group_by: str,
    source: str,
    filters: Any = None,
) -> dict[str, Any]:
    """Return observed outcome counts/rates by a supported DANE code column."""
    _validate_outcome(outcome)
    _validate_group_column(data, group_by)
    subset, applied = _apply_filters(data, filters)

    work = pd.DataFrame(
        {group_by: subset[group_by], outcome: _outcome_values(subset, outcome)}
    )
    summary = (
        work.groupby(group_by, dropna=False, sort=True)
        .agg(
            record_count=(outcome, "size"),
            valid_outcome_count=(outcome, "count"),
            positive_count=(outcome, lambda group: int((group == 1).sum())),
        )
        .reset_index()
    )

    groups = []
    for row in summary.to_dict(orient="records"):
        valid_count = int(row["valid_outcome_count"])
        positive_count = int(row["positive_count"])
        group_value = _json_value(row[group_by])
        groups.append(
            {
                "group_value": group_value,
                "group_label": label_for(group_by, group_value),
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
        "filters": applied,
        "outcome": outcome,
        "group_by": group_by,
        "group_by_label": VARIABLES.get(group_by, {}).get("nombre", group_by),
        "groups": groups,
    }


def group_distribution(
    data: pd.DataFrame, group_by: str, source: str, filters: Any = None
) -> dict[str, Any]:
    """Return record counts and shares for each value of a DANE code column."""
    _validate_group_column(data, group_by)
    subset, applied = _apply_filters(data, filters)
    counts = (
        subset.groupby(group_by, dropna=False, sort=True)
        .size()
        .reset_index(name="record_count")
    )
    total = int(len(subset))
    groups = []
    for row in counts.to_dict(orient="records"):
        group_value = _json_value(row[group_by])
        record_count = int(row["record_count"])
        groups.append(
            {
                "group_value": group_value,
                "group_label": label_for(group_by, group_value),
                "record_count": record_count,
                "share": record_count / total if total else None,
            }
        )
    return {
        "source": source,
        "filters": applied,
        "group_by": group_by,
        "group_by_label": VARIABLES.get(group_by, {}).get("nombre", group_by),
        "record_count": total,
        "groups": groups,
    }


@lru_cache(maxsize=4)
def _load_model(model_path: Path) -> Any:
    return joblib.load(model_path)


def _validate_profile_value(feature: str, value: float) -> None:
    kind, rule = PROFILE_RULES[feature]
    if kind == "set" and int(value) not in rule:
        allowed = ", ".join(str(code) for code in sorted(rule))
        raise ValueError(f"{feature}={int(value)} no es un código válido ({allowed}).")
    if kind == "range":
        low, high = rule
        if not low <= value <= high:
            hint = " (99 = sin información en el DANE)" if value == 99 else ""
            raise ValueError(
                f"{feature}={int(value)} está fuera del rango del modelo "
                f"({low}-{high}){hint}."
            )


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
        value = float(_as_int_code(profile[feature], feature))
        _validate_profile_value(feature, value)
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
        model = _load_model(model_path)
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
        "profile": {feature: int(value) for feature, value in numeric_profile.items()},
        "profile_labels": {
            feature: label_for(feature, value)
            for feature, value in numeric_profile.items()
            if feature in VARIABLES
        },
        "feature_names": list(FEATURES),
        "warning": (
            "Estimación estadística del modelo, no frecuencia observada ni "
            "consejo clínico."
        ),
    }


def model_metrics(metrics_path: Path = METRICS_PATH) -> dict[str, Any]:
    """Return the cross-validated performance saved by `main.py` for both models."""
    if not metrics_path.is_file():
        raise FileNotFoundError(
            "No hay métricas guardadas; ejecuta `uv run python main.py` para "
            "reentrenar y generarlas."
        )
    saved = json.loads(metrics_path.read_text(encoding="utf-8"))
    models = {}
    for outcome, info in saved["modelos"].items():
        comparison = [
            {
                "model": name,
                "model_label": MODEL_LABELS.get(name, name),
                "roc_auc": values["roc_auc"],
                "roc_auc_std": values["roc_auc_std"],
                "average_precision": values["avg_precision"],
                "average_precision_std": values["avg_precision_std"],
            }
            for name, values in info["comparacion_cv"].items()
        ]
        models[outcome] = {
            "selected_model": info["modelo_seleccionado"],
            "selected_model_label": MODEL_LABELS.get(
                info["modelo_seleccionado"], info["modelo_seleccionado"]
            ),
            "training_samples": info["muestras_entrenamiento"],
            "positive_rate": info["tasa_positivos"],
            "average_precision_baseline": info["tasa_positivos"],
            "cross_validation": comparison,
            "model_file": info["archivo"],
        }
    return {
        "source": metrics_path.relative_to(PROJECT_ROOT).as_posix()
        if metrics_path.is_relative_to(PROJECT_ROOT)
        else str(metrics_path),
        "generated_at": saved["generado"],
        "validation": saved["validacion"],
        "features": saved["variables_modelo"],
        "models": models,
        "notes": (
            "ROC-AUC: 0,5 = azar, 1 = perfecto. La línea base de la precisión "
            "promedio (AP) es la tasa de positivos."
        ),
    }


def build_tools(
    data: pd.DataFrame, source: str
) -> dict[str, Callable[..., dict[str, Any]]]:
    """Create the explicit tool registry consumed by the agents."""
    return {
        "dataset_overview": lambda filters=None: dataset_overview(
            data, source, filters
        ),
        "outcome_summary": lambda outcome, filters=None: outcome_summary(
            data, outcome, source, filters
        ),
        "grouped_outcome": lambda outcome, group_by, filters=None: grouped_outcome(
            data, outcome, group_by, source, filters
        ),
        "group_distribution": lambda group_by, filters=None: group_distribution(
            data, group_by, source, filters
        ),
        "predict_risks": lambda profile: predict_risks(profile),
        "model_metrics": lambda: model_metrics(),
    }
