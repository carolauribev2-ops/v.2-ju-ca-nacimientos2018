"""Deterministic Spanish-language router for the project's data tools."""

from dataclasses import dataclass
import re
import unicodedata
from typing import Any

import pandas as pd

from .tools import build_tools, load_birth_data

OUTCOME_NAMES = {
    "CESAREA": "cesárea",
    "BAJO_PESO": "bajo peso al nacer",
}
GROUP_NAMES = {
    "CODPTORE": "departamento de residencia",
    "EDAD_MADRE": "grupo de edad materna",
    "T_GES": "grupo de semanas de gestación",
    "AREA_RES": "área de residencia",
    "IDCLASADMI": "régimen de salud",
    "SEXO": "sexo",
    "MUL_PARTO": "tipo de gestación",
}


@dataclass(frozen=True)
class AgentResponse:
    answer: str
    tool_name: str | None
    evidence: dict[str, Any] | None


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    without_accents = "".join(
        character for character in normalized if not unicodedata.combining(character)
    )
    return " ".join(without_accents.casefold().split())


def _requested_outcomes(question: str) -> list[str]:
    outcomes = []
    if "cesarea" in question:
        outcomes.append("CESAREA")
    if "bajo peso" in question or "peso bajo" in question:
        outcomes.append("BAJO_PESO")
    return outcomes


def _requested_group(question: str) -> str | None:
    if _has_term(question, "departamento", "depto"):
        return "CODPTORE"
    if _has_term(question, "edad"):
        return "EDAD_MADRE"
    if _has_term(question, "sexo"):
        return "SEXO"
    if _has_term(question, "area", "residencia"):
        return "AREA_RES"
    if _has_term(question, "regimen", "afiliacion"):
        return "IDCLASADMI"
    if _has_term(question, "tipo de gestacion", "embarazo multiple"):
        return "MUL_PARTO"
    if _has_term(question, "semana", "semanas", "gestacion", "prematuro", "prematura"):
        return "T_GES"
    return None


def _has_term(question: str, *terms: str) -> bool:
    return any(re.search(rf"\b{re.escape(term)}\b", question) for term in terms)


def _format_percentage(value: float | None) -> str:
    return f"{value:.1%}" if value is not None else "sin datos válidos"


class BirthsAgent:
    """Answer only requests that can be fulfilled by the registered tools."""

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

    def answer(
        self, question: str, profile: dict[str, Any] | None = None
    ) -> AgentResponse:
        normalized = _normalize(question)
        outcomes = _requested_outcomes(normalized)
        group_by = _requested_group(normalized)
        prediction_requested = any(
            term in normalized
            for term in ("predecir", "predice", "prediccion", "segun el modelo")
        )

        if prediction_requested:
            if profile is None:
                return AgentResponse(
                    answer=(
                        "Para estimar probabilidades necesito un perfil con las 10 "
                        "variables del modelo. Pásalo con --profile; las estimaciones "
                        "no son resultados observados ni consejo clínico."
                    ),
                    tool_name=None,
                    evidence=None,
                )
            evidence = self.tools["predict_risks"](profile)
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
                self.tools["grouped_outcome"](outcome, group_by)
                for outcome in outcomes
            ]
            evidence = {"results": results}
            descriptions = []
            for result in results:
                groups = "; ".join(
                    f"{group['group_value']}: {group['positive_count']}/"
                    f"{group['valid_outcome_count']} "
                    f"({_format_percentage(group['positive_rate'])})"
                    for group in result["groups"]
                )
                descriptions.append(
                    f"{OUTCOME_NAMES[result['outcome']]} por "
                    f"{GROUP_NAMES[group_by]}: {groups}"
                )
            return AgentResponse(
                "Resultados observados por código/categoría: "
                + " | ".join(descriptions),
                "grouped_outcome",
                evidence,
            )

        overview_requested = any(
            term in normalized
            for term in ("nacimiento", "nacimientos", "registro", "resumen", "total")
        )
        if overview_requested and not group_by:
            evidence = self.tools["dataset_overview"]()
            outcomes_text = "; ".join(
                f"{OUTCOME_NAMES[outcome]}: "
                f"{summary['positive_count']:,}/"
                f"{summary['records_with_valid_outcome']:,} "
                f"({_format_percentage(summary['positive_rate'])})"
                for outcome, summary in evidence["outcomes"].items()
            )
            return AgentResponse(
                f"El dataset contiene {evidence['record_count']:,} registros. "
                f"Resultados observados: {outcomes_text}.",
                "dataset_overview",
                evidence,
            )

        if len(outcomes) == 1:
            evidence = self.tools["outcome_summary"](outcomes[0])
            answer = (
                f"{OUTCOME_NAMES[outcomes[0]].capitalize()}: "
                f"{evidence['positive_count']:,} casos positivos de "
                f"{evidence['records_with_valid_outcome']:,} registros con "
                f"resultado válido ({_format_percentage(evidence['positive_rate'])}); "
                f"{evidence['non_positive_count']:,} no positivos."
            )
            return AgentResponse(answer, "outcome_summary", evidence)

        if group_by:
            evidence = self.tools["group_distribution"](group_by)
            groups = "; ".join(
                f"{group['group_value']}: {group['record_count']:,}"
                for group in evidence["groups"]
            )
            return AgentResponse(
                f"Registros por {GROUP_NAMES[group_by]} (códigos/categorías DANE): "
                f"{groups}",
                "group_distribution",
                evidence,
            )

        return AgentResponse(
            answer=(
                "No puedo responder con las herramientas disponibles. Prueba con "
                "el total de nacimientos, una tasa de cesárea/bajo peso, un desglose "
                "por departamento/edad/gestación o una predicción con perfil completo."
            ),
            tool_name=None,
            evidence=None,
        )
