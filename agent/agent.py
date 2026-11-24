"""Modo básico: enrutador determinista en español, sin LLM.

Se usa cuando no hay credenciales de Claude. Reconoce un conjunto acotado de
preguntas (totales, tasas, desgloses y filtros por departamento, área, edad,
régimen, prematuridad o multiplicidad) y las responde solo con las
herramientas de ``agent.tools``. Si no entiende la pregunta lo dice, en vez de
devolver una cifra que responde otra cosa.
"""

from dataclasses import dataclass, field
import re
from typing import Any

import pandas as pd

from .catalog import find_departments, normalize_text
from .tools import build_tools, load_birth_data

OUTCOME_NAMES = {
    "CESAREA": "cesárea",
    "BAJO_PESO": "bajo peso al nacer",
}
GROUP_NAMES = {
    "CODPTORE": "departamento de residencia",
    "EDAD_MADRE": "grupo de edad materna",
    "T_GES": "semanas de gestación (agrupación DANE)",
    "T_GES_AGRU_CIE": "semanas de gestación (agrupación CIE)",
    "AREA_RES": "área de residencia",
    "IDCLASADMI": "entidad administradora en salud",
    "SEXO": "sexo",
    "MUL_PARTO": "multiplicidad del embarazo",
}
HELP_TEXT = (
    "Puedo responder con los datos de nacimientos DANE 2018: totales, tasas de "
    "cesárea o bajo peso, desgloses (por departamento, edad, semanas de gestación, "
    "área, régimen, sexo o multiplicidad) y filtros como «en Antioquia», «rural», "
    "«adolescentes», «prematuros», «subsidiado» o «embarazo múltiple». También "
    "explico el desempeño de los modelos («¿qué tan bueno es el modelo?»). Para "
    "predicciones pásame un perfil completo con --profile. En modo básico entiendo "
    "frases simples; con ANTHROPIC_API_KEY configurada el agente usa Claude y "
    "entiende preguntas libres."
)


@dataclass(frozen=True)
class AgentResponse:
    answer: str
    tool_name: str | None
    evidence: dict[str, Any] | None
    mode: str = field(default="basico")


def _normalize(text: str) -> str:
    return normalize_text(text)


def _has_term(question: str, *terms: str) -> bool:
    return any(re.search(rf"\b{re.escape(term)}\b", question) for term in terms)


def _requested_outcomes(question: str) -> list[str]:
    outcomes = []
    if re.search(r"\bcesareas?\b", question):
        outcomes.append("CESAREA")
    if "bajo peso" in question or "peso bajo" in question:
        outcomes.append("BAJO_PESO")
    return outcomes


# Term -> group column, checked in order. Only used after "por/según/cada".
_GROUP_TERMS: list[tuple[tuple[str, ...], str]] = [
    (("departamento", "departamentos", "depto", "deptos"), "CODPTORE"),
    (("edad", "edades", "grupo de edad"), "EDAD_MADRE"),
    (("sexo",), "SEXO"),
    (("area", "zona", "residencia"), "AREA_RES"),
    (("regimen", "afiliacion", "entidad", "eps", "aseguramiento"), "IDCLASADMI"),
    (("multiplicidad", "tipo de gestacion", "tipo de embarazo"), "MUL_PARTO"),
    (("semana", "semanas", "gestacion", "tiempo de gestacion"), "T_GES"),
]


def _requested_group(question: str) -> str | None:
    for terms, column in _GROUP_TERMS:
        for term in terms:
            if re.search(
                rf"\b(por|segun|cada|desglos\w*|distribucion)\s+(de\s+|del\s+|el\s+|la\s+|los\s+|las\s+)*{re.escape(term)}\b",
                question,
            ):
                return column
    return None


def _age_code(years: int) -> int | None:
    if 10 <= years <= 54:
        return min((years - 10) // 5 + 1, 9)
    return None


def _requested_filters(
    question: str, available_columns: set[str]
) -> dict[str, list[int]]:
    filters: dict[str, list[int]] = {}

    departments = find_departments(question)
    if departments:
        filters["CODPTORE"] = departments

    if _has_term(question, "rural", "rurales", "campo"):
        filters["AREA_RES"] = [3]
    elif _has_term(question, "centro poblado", "centros poblados"):
        filters["AREA_RES"] = [2]
    elif _has_term(question, "urbana", "urbanas", "urbano", "cabecera", "cabeceras"):
        filters["AREA_RES"] = [1]

    age_match = re.search(r"\b(\d{2})\s*anos\b", question)
    if _has_term(question, "adolescente", "adolescentes"):
        filters["EDAD_MADRE"] = [1, 2]
    elif re.search(r"\b(menores de 15|menor de 15|ninas)\b", question):
        filters["EDAD_MADRE"] = [1]
    elif re.search(r"\b(mayores de 40|mayor de 40|40 anos o mas)\b", question):
        filters["EDAD_MADRE"] = [7, 8, 9]
    elif age_match and (code := _age_code(int(age_match.group(1)))) is not None:
        filters["EDAD_MADRE"] = [code]

    if _has_term(question, "prematuro", "prematuros", "prematura", "prematuras"):
        if "T_GES_AGRU_CIE" in available_columns:
            filters["T_GES_AGRU_CIE"] = [1, 2, 3]
        else:
            filters["T_GES"] = [1, 2]

    if _has_term(
        question,
        "embarazo multiple",
        "embarazos multiples",
        "parto multiple",
        "gemelar",
        "gemelares",
        "gemelos",
        "mellizos",
    ):
        filters["MUL_PARTO"] = [2, 3, 4]

    if _has_term(question, "subsidiado", "subsidiada", "subsidiadas"):
        filters["IDCLASADMI"] = [2]
    elif _has_term(question, "contributivo", "contributiva"):
        filters["IDCLASADMI"] = [1, 3]
    elif _has_term(question, "no asegurada", "no aseguradas", "sin aseguramiento"):
        filters["IDCLASADMI"] = [-1]

    return filters


def _format_percentage(value: float | None) -> str:
    return f"{value:.1%}" if value is not None else "sin datos válidos"


def _describe_filters(applied: list[dict[str, Any]]) -> str:
    if not applied:
        return ""
    parts = [
        f"{GROUP_NAMES.get(item['variable'], item['variable'])} = "
        + ", ".join(item["labels"])
        for item in applied
    ]
    return " [Filtro: " + "; ".join(parts) + "]"


class BirthsAgent:
    """Answer only requests that can be fulfilled by the registered tools."""

    mode = "basico"

    def __init__(
        self, data: pd.DataFrame | None = None, source: str | None = None
    ) -> None:
        if data is None:
            data, source = load_birth_data()
        elif source is None:
            raise ValueError("Se requiere indicar la fuente del dataset proporcionado.")
        self.data = data
        self.source = source
        self.tools = build_tools(self.data, self.source)

    def reset(self) -> None:
        """El modo básico no guarda memoria entre preguntas."""

    def answer(
        self, question: str, profile: dict[str, Any] | None = None
    ) -> AgentResponse:
        normalized = _normalize(question)

        other_years = [
            year for year in re.findall(r"\b(19\d{2}|20\d{2})\b", normalized)
            if year != "2018"
        ]
        if other_years:
            return AgentResponse(
                answer=(
                    f"Solo tengo datos de nacimientos ocurridos en Colombia en 2018; "
                    f"no puedo responder sobre {', '.join(other_years)}."
                ),
                tool_name=None,
                evidence=None,
            )

        outcomes = _requested_outcomes(normalized)
        group_by = _requested_group(normalized)
        filters = _requested_filters(normalized, set(self.data.columns))
        prediction_requested = any(
            term in normalized
            for term in ("predecir", "predice", "prediccion", "segun el modelo")
        )

        metrics_requested = _has_term(
            normalized,
            "metrica",
            "metricas",
            "auc",
            "roc",
            "desempeno",
            "rendimiento",
            "que tan bueno",
            "que tan confiable",
            "que tan preciso",
            "precision del modelo",
            "evaluacion del modelo",
        )
        if metrics_requested and profile is None:
            try:
                evidence = self.tools["model_metrics"]()
            except FileNotFoundError as error:
                return AgentResponse(str(error), None, None)
            parts = []
            for outcome, info in evidence["models"].items():
                best = next(
                    row
                    for row in info["cross_validation"]
                    if row["model"] == info["selected_model"]
                )
                parts.append(
                    f"{OUTCOME_NAMES[outcome]}: {info['selected_model_label']} con "
                    f"ROC-AUC {best['roc_auc']:.3f} ± {best['roc_auc_std']:.3f} y "
                    f"precisión promedio {best['average_precision']:.3f} (línea base "
                    f"{info['average_precision_baseline']:.3f}), "
                    f"{info['training_samples']:,} muestras"
                )
            return AgentResponse(
                f"Desempeño con {evidence['validation'].lower()}: "
                + "; ".join(parts)
                + ".",
                "model_metrics",
                evidence,
            )

        if prediction_requested or profile is not None:
            if profile is None:
                return AgentResponse(
                    answer=(
                        "Para estimar probabilidades necesito un perfil con las 10 "
                        "variables del modelo. Pásalo con --profile o --profile-file; "
                        "las estimaciones no son resultados observados ni consejo "
                        "clínico."
                    ),
                    tool_name=None,
                    evidence=None,
                )
            try:
                evidence = self.tools["predict_risks"](profile)
            except (TypeError, ValueError, FileNotFoundError) as error:
                return AgentResponse(
                    answer=f"No pude estimar el riesgo: {error}",
                    tool_name=None,
                    evidence=None,
                )
            requested = outcomes or list(OUTCOME_NAMES)
            answer = "Estimaciones del modelo (no son frecuencias observadas): "
            answer += "; ".join(
                f"{OUTCOME_NAMES[outcome]} "
                f"{_format_percentage(evidence['probabilities'][outcome])}"
                for outcome in requested
            )
            return AgentResponse(answer, "predict_risks", evidence)

        if group_by and outcomes:
            results = [
                self.tools["grouped_outcome"](outcome, group_by, filters)
                for outcome in outcomes
            ]
            descriptions = []
            for result in results:
                groups = "; ".join(
                    f"{group['group_label']} ({group['group_value']}): "
                    f"{group['positive_count']:,}/{group['valid_outcome_count']:,} "
                    f"({_format_percentage(group['positive_rate'])})"
                    for group in result["groups"]
                )
                descriptions.append(
                    f"{OUTCOME_NAMES[result['outcome']]} por "
                    f"{GROUP_NAMES[group_by]}: {groups}"
                )
            return AgentResponse(
                "Resultados observados"
                + _describe_filters(results[0]["filters"])
                + ": "
                + " | ".join(descriptions),
                "grouped_outcome",
                {"results": results},
            )

        if group_by:
            evidence = self.tools["group_distribution"](group_by, filters)
            groups = "; ".join(
                f"{group['group_label']} ({group['group_value']}): "
                f"{group['record_count']:,} ({_format_percentage(group['share'])})"
                for group in evidence["groups"]
            )
            return AgentResponse(
                f"Registros por {GROUP_NAMES[group_by]}"
                f"{_describe_filters(evidence['filters'])}: {groups}",
                "group_distribution",
                evidence,
            )

        count_requested = any(
            term in normalized
            for term in (
                "nacimiento",
                "nacimientos",
                "nacidos",
                "nacieron",
                "registro",
                "registros",
                "resumen",
                "total",
                "cuantos",
                "cuantas",
                "bebes",
                "madres",
                "partos",
            )
        )

        if len(outcomes) == 1 and not count_requested:
            evidence = self.tools["outcome_summary"](outcomes[0], filters)
            answer = (
                f"{OUTCOME_NAMES[outcomes[0]].capitalize()}"
                f"{_describe_filters(evidence['filters'])}: "
                f"{evidence['positive_count']:,} casos de "
                f"{evidence['records_with_valid_outcome']:,} registros con "
                f"resultado válido ({_format_percentage(evidence['positive_rate'])})."
            )
            return AgentResponse(answer, "outcome_summary", evidence)

        if count_requested or outcomes or filters:
            evidence = self.tools["dataset_overview"](filters)
            outcomes_text = "; ".join(
                f"{OUTCOME_NAMES[outcome]}: "
                f"{summary['positive_count']:,}/"
                f"{summary['records_with_valid_outcome']:,} "
                f"({_format_percentage(summary['positive_rate'])})"
                for outcome, summary in evidence["outcomes"].items()
            )
            scope = "El dataset contiene" if not evidence["filters"] else "Hay"
            return AgentResponse(
                f"{scope} {evidence['record_count']:,} registros"
                f"{_describe_filters(evidence['filters'])}. "
                f"Resultados observados: {outcomes_text}.",
                "dataset_overview",
                evidence,
            )

        if _has_term(normalized, "hola", "ayuda", "que puedes hacer", "que sabes"):
            return AgentResponse(answer=HELP_TEXT, tool_name=None, evidence=None)

        return AgentResponse(
            answer="No puedo responder esa pregunta en modo básico. " + HELP_TEXT,
            tool_name=None,
            evidence=None,
        )
