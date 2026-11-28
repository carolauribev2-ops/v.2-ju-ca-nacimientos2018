"""Agente conversacional con Claude (Anthropic) que usa las herramientas del proyecto.

Claude decide qué herramientas llamar y con qué filtros; las cifras siempre salen
de ``agent.tools`` (datos DANE procesados y modelos entrenados). El historial se
guarda en memoria para permitir preguntas de seguimiento.
"""

from __future__ import annotations

import json
import os
from typing import Any

import anthropic
import pandas as pd

from src.preprocessing import FEATURES

from .agent import AgentResponse
from .catalog import catalog_as_text
from .tools import (
    ALL_GROUP_COLUMNS,
    FILTER_COLUMNS,
    OUTCOMES,
    build_tools,
    load_birth_data,
)

DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_EFFORT = "medium"
MAX_TOKENS = 16000
MAX_STEPS = 10
# Server-side refusal fallback (beta) is available on these models.
FALLBACK_MODELS = {
    "claude-opus-5-5",
    "claude-opus-5",
    "claude-sonnet-5-5",
    "claude-fable-5-2",
    "claude-fable-5-1",
}
FALLBACK_BETA = "server-side-fallback-2026-07-01"

_FILTERS_SCHEMA: dict[str, Any] = {
    "type": "array",
    "description": (
        "Filtros que definen el subconjunto a analizar; lista vacía = todo el "
        "dataset. Cada filtro restringe una variable a uno o más códigos DANE del "
        "catálogo y los filtros se combinan con Y."
    ),
    "items": {
        "type": "object",
        "properties": {
            "variable": {"type": "string", "enum": sorted(FILTER_COLUMNS)},
            "codes": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "Códigos DANE aceptados para la variable.",
            },
        },
        "required": ["variable", "codes"],
        "additionalProperties": False,
    },
}

_PROFILE_DESCRIPTIONS = {
    "EDAD_MADRE": "Grupo de edad de la madre: 1=10-14 … 9=50-54 años (no años).",
    "T_GES": "Tiempo de gestación DANE: 1=<22, 2=22-27, 3=28-37, 4=38-41, 5=42+, 6=ignorado, 9=sin información.",
    "NUMCONSUL": "Número de consultas prenatales (0-25).",
    "N_HIJOSV": "Hijos nacidos vivos incluido el presente (1-20).",
    "N_EMB": "Número de embarazos incluido el actual (1-18).",
    "AREA_RES": "Área de residencia: 1=cabecera, 2=centro poblado, 3=rural disperso, 9=sin información, -1=sin dato.",
    "IDCLASADMI": "Entidad en salud: 1=EPS contributivo, 2=EPS subsidiado, 3=adaptada, 4=especial, 5=exceptuada, 9=sin información, -1=no asegurada.",
    "SEXO": "Sexo del bebé: 1=masculino, 2=femenino.",
    "MUL_PARTO": "Multiplicidad: 1=simple, 2=doble, 3=triple, 4=cuádruple o más, 9=sin información.",
    "CODPTORE": "Código DANE del departamento de residencia (ver catálogo) o -1.",
}

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "dataset_overview",
        "description": (
            "Cuenta registros y calcula las tasas observadas de cesárea y de bajo "
            "peso al nacer (con numerador y denominador) para todo el dataset o "
            "para un subconjunto filtrado. Úsala para preguntas de «¿cuántos…?», "
            "«¿qué porcentaje…?» o la tasa de un subgrupo concreto (p. ej. "
            "cesárea en Antioquia o bajo peso en prematuros)."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"filters": _FILTERS_SCHEMA},
            "required": ["filters"],
            "additionalProperties": False,
        },
    },
    {
        "name": "grouped_outcome",
        "description": (
            "Desglosa la tasa observada de un resultado (CESAREA o BAJO_PESO) por "
            "cada categoría de una variable, opcionalmente dentro de un subconjunto "
            "filtrado. Úsala para comparar grupos o encontrar dónde la tasa es "
            "mayor o menor (p. ej. cesárea por departamento, bajo peso por edad "
            "materna en zona rural)."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "outcome": {"type": "string", "enum": sorted(OUTCOMES)},
                "group_by": {"type": "string", "enum": sorted(ALL_GROUP_COLUMNS)},
                "filters": _FILTERS_SCHEMA,
            },
            "required": ["outcome", "group_by", "filters"],
            "additionalProperties": False,
        },
    },
    {
        "name": "group_distribution",
        "description": (
            "Cuenta cuántos registros hay en cada categoría de una variable y qué "
            "proporción representan, opcionalmente dentro de un subconjunto "
            "filtrado. Úsala para distribuciones (p. ej. nacimientos por "
            "departamento o por grupo de edad de la madre)."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "group_by": {"type": "string", "enum": sorted(ALL_GROUP_COLUMNS)},
                "filters": _FILTERS_SCHEMA,
            },
            "required": ["group_by", "filters"],
            "additionalProperties": False,
        },
    },
    {
        "name": "predict_risks",
        "description": (
            "Estima con los modelos entrenados la probabilidad de cesárea y de bajo "
            "peso al nacer para un perfil con las 10 variables del modelo. Úsala "
            "solo cuando el usuario pida una predicción o el riesgo estimado de un "
            "caso concreto y conozcas las 10 variables; no inventes valores."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "profile": {
                    "type": "object",
                    "properties": {
                        feature: {
                            "type": "integer",
                            "description": _PROFILE_DESCRIPTIONS[feature],
                        }
                        for feature in FEATURES
                    },
                    "required": list(FEATURES),
                    "additionalProperties": False,
                }
            },
            "required": ["profile"],
            "additionalProperties": False,
        },
    },
    {
        "name": "model_metrics",
        "description": (
            "Devuelve el desempeño de los modelos predictivos medido con validación "
            "cruzada: ROC-AUC y precisión promedio de cada algoritmo candidato, el "
            "modelo seleccionado, el número de muestras y la calibración de las "
            "probabilidades. Úsala cuando pregunten qué tan bueno o confiable es el "
            "modelo, qué algoritmo se usó, cómo se evaluó o si sus porcentajes son "
            "probabilidades reales."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
]

_TOOL_NAMES = {tool["name"] for tool in TOOL_DEFINITIONS}

SYSTEM_PROMPT = f"""Eres el agente de análisis del proyecto «Predicción de riesgo perinatal — DANE Colombia 2018». Respondes en español preguntas sobre los registros de nacidos vivos en Colombia durante 2018 (microdatos DANE ya depurados) y sobre las estimaciones de dos modelos de scikit-learn (probabilidad de cesárea y de bajo peso al nacer).

Cómo trabajar:
1. Toda cifra de tu respuesta debe salir de un resultado de herramienta de esta conversación. No uses cifras de memoria ni las aproximes. Si ninguna herramienta puede calcular lo pedido, dilo y explica qué sí puedes calcular.
2. Distingue siempre frecuencias observadas (dataset_overview, grouped_outcome, group_distribution) de estimaciones del modelo (predict_risks). Las estimaciones no son resultados observados ni consejo clínico. Para hablar de la calidad de los modelos usa model_metrics (validación cruzada) y explica las métricas en lenguaje sencillo.
3. Las variables son códigos DANE, no medidas: EDAD_MADRE es un grupo de edad, no años. Traduce lo que diga el usuario a códigos con el catálogo («madre de 35 años» → EDAD_MADRE 6; «Antioquia» → CODPTORE 5; «rural» → AREA_RES 3) y en la respuesta usa los nombres, no solo los códigos.
4. Para prematuridad (<37 semanas) filtra T_GES_AGRU_CIE en [1, 2, 3]: el código 3 de T_GES (28-37 semanas) incluye la semana 37, que ya es a término.
5. Aplica los filtros que la pregunta implique y reporta las tasas con numerador y denominador. Para comparar un subgrupo con el total, consulta ambos.
6. predict_risks necesita las 10 variables del modelo. Si falta alguna, pregunta por las que faltan en lugar de suponerlas.
7. Solo hay datos de 2018 y de las variables del catálogo. Las columnas eliminadas en el preprocesamiento (APGAR, talla, educación, estado civil, pertenencia étnica, municipio, edad del padre) no están disponibles.
8. Sé breve y directo. Con varios grupos usa una tabla corta; en desgloses largos (como los 33 departamentos) muestra los más relevantes para la pregunta e indica que hay más.

Catálogo de códigos DANE:
{catalog_as_text()}

Variables del modelo predictivo: {", ".join(FEATURES)}."""


class AgentError(RuntimeError):
    """Error de la API de Anthropic traducido a un mensaje en español."""


def _friendly_error(error: Exception, model: str) -> AgentError:
    if isinstance(error, anthropic.AuthenticationError):
        return AgentError(
            "La credencial de Anthropic no es válida. Revisa ANTHROPIC_API_KEY."
        )
    if isinstance(error, anthropic.PermissionDeniedError):
        return AgentError(f"La credencial no tiene permiso para usar {model}.")
    if isinstance(error, anthropic.NotFoundError):
        return AgentError(f"El modelo {model} no existe o no está disponible.")
    if isinstance(error, anthropic.RateLimitError):
        return AgentError("Se alcanzó el límite de uso de la API; intenta más tarde.")
    if isinstance(error, anthropic.BadRequestError):
        return AgentError(f"La API rechazó la solicitud: {error.message}")
    if isinstance(error, anthropic.APIStatusError):
        return AgentError(
            f"Error del servicio de Anthropic ({error.status_code}); intenta más tarde."
        )
    if isinstance(error, anthropic.APIConnectionError):
        return AgentError("No hay conexión con la API de Anthropic.")
    return AgentError(str(error))


class ClaudeBirthsAgent:
    """Agente con Claude: elige herramientas, aplica filtros y conversa en varios turnos."""

    mode = "llm"

    def __init__(
        self,
        data: pd.DataFrame | None = None,
        source: str | None = None,
        *,
        client: Any = None,
        model: str | None = None,
        effort: str | None = None,
        max_steps: int = MAX_STEPS,
    ) -> None:
        if data is None:
            data, source = load_birth_data()
        elif source is None:
            raise ValueError("Se requiere indicar la fuente del dataset proporcionado.")
        self.data = data
        self.source = source
        self.tools = build_tools(data, source)
        if client is None:
            missing_credentials = AgentError(
                "No hay credencial de Anthropic. Define ANTHROPIC_API_KEY en el "
                "entorno o en el archivo .env (ver .env.example)."
            )
            try:
                client = anthropic.Anthropic()
            except anthropic.AnthropicError as error:
                raise missing_credentials from error
            # The SDK resolves env vars and `ant auth login` profiles at construction.
            if not (client.api_key or client.auth_token or client.credentials):
                raise missing_credentials
        self.client = client
        self.model = model or os.environ.get("AGENTE_MODELO") or DEFAULT_MODEL
        self.effort = effort or os.environ.get("AGENTE_ESFUERZO") or DEFAULT_EFFORT
        self.max_steps = max_steps
        self.messages: list[dict[str, Any]] = []

    def reset(self) -> None:
        """Olvida la conversación anterior."""
        self.messages = []

    def _create(self) -> Any:
        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": MAX_TOKENS,
            "system": SYSTEM_PROMPT,
            "tools": TOOL_DEFINITIONS,
            "messages": self.messages,
            "output_config": {"effort": self.effort},
            "cache_control": {"type": "ephemeral"},
        }
        if self.model in FALLBACK_MODELS:
            request["betas"] = [FALLBACK_BETA]
            request["fallbacks"] = "default"
        return self.client.beta.messages.create(**request)

    def _run_tool(self, block: Any, log: list[dict[str, Any]]) -> dict[str, Any]:
        arguments = dict(block.input or {})
        try:
            if block.name not in _TOOL_NAMES:
                raise ValueError(f"Herramienta desconocida: {block.name}")
            result = self.tools[block.name](**arguments)
        except (TypeError, ValueError, KeyError, FileNotFoundError) as error:
            log.append({"tool": block.name, "input": arguments, "error": str(error)})
            return {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": f"Error: {error}",
                "is_error": True,
            }
        log.append({"tool": block.name, "input": arguments, "result": result})
        return {
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": json.dumps(result, ensure_ascii=False),
        }

    def answer(
        self, question: str, profile: dict[str, Any] | None = None
    ) -> AgentResponse:
        content = question
        if profile is not None:
            content += (
                "\n\nPerfil proporcionado para predict_risks (JSON): "
                + json.dumps(profile, ensure_ascii=False)
            )
        turn_start = len(self.messages)
        self.messages.append({"role": "user", "content": content})
        tool_log: list[dict[str, Any]] = []

        try:
            for _ in range(self.max_steps):
                response = self._create()
                if response.stop_reason == "refusal":
                    del self.messages[turn_start:]
                    return self._response(
                        "El modelo declinó responder esta solicitud.", tool_log
                    )
                self.messages.append({"role": "assistant", "content": response.content})

                if response.stop_reason == "tool_use":
                    results = [
                        self._run_tool(block, tool_log)
                        for block in response.content
                        if block.type == "tool_use"
                    ]
                    self.messages.append({"role": "user", "content": results})
                    continue
                if response.stop_reason == "pause_turn":
                    continue

                text = "\n".join(
                    block.text for block in response.content if block.type == "text"
                ).strip()
                if response.stop_reason == "max_tokens":
                    text += "\n\n[Respuesta truncada por longitud.]"
                return self._response(text or "(sin respuesta de texto)", tool_log)
        except anthropic.AnthropicError as error:
            del self.messages[turn_start:]
            raise _friendly_error(error, self.model) from error

        del self.messages[turn_start:]
        return self._response(
            f"No terminé en {self.max_steps} pasos; reformula la pregunta de forma "
            "más concreta.",
            tool_log,
        )

    def _response(self, text: str, tool_log: list[dict[str, Any]]) -> AgentResponse:
        names = list(dict.fromkeys(call["tool"] for call in tool_log))
        return AgentResponse(
            answer=text,
            tool_name=", ".join(names) if names else None,
            evidence={"tool_calls": tool_log} if tool_log else None,
            mode=self.mode,
        )
