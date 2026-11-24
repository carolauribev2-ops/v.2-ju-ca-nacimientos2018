"""Línea de comandos del agente de nacimientos DANE 2018.

Ejemplos (desde la raíz del proyecto):
    uv run python -m agent.run                        # chat interactivo
    uv run python -m agent.run -q "¿Tasa de cesárea en Antioquia?"
    uv run python -m agent.run -q "Predice el riesgo" --profile-file agent/sample_profile.json
    uv run python -m agent.run --modo basico          # sin LLM
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .agent import AgentResponse, BirthsAgent
from .factory import create_agent, llm_credentials_available
from .llm_agent import AgentError

EXIT_WORDS = {"salir", "exit", "quit", ":q"}


def _parse_profile(raw_profile: str | None, parser: argparse.ArgumentParser) -> (
    dict[str, Any] | None
):
    if raw_profile is None:
        return None
    try:
        profile = json.loads(raw_profile)
    except json.JSONDecodeError as error:
        parser.error(f"--profile debe ser JSON válido: {error}")
    if not isinstance(profile, dict):
        parser.error("--profile debe ser un objeto JSON.")
    return profile


def _print_response(response: AgentResponse, show_evidence: bool) -> None:
    print(response.answer)
    if show_evidence and response.evidence is not None:
        print(
            json.dumps(
                {"tool": response.tool_name, "evidence": response.evidence},
                ensure_ascii=False,
                indent=2,
            )
        )
    elif response.tool_name:
        print(f"  (herramientas: {response.tool_name})")


def _ask(agent, fallback, question, profile, mode) -> AgentResponse:
    """Pregunta al agente; en modo auto, si falla la API usa el modo básico."""
    try:
        return agent.answer(question, profile)
    except AgentError as error:
        if mode != "auto":
            raise
        print(f"[Aviso] {error} Respondo en modo básico.", file=sys.stderr)
        return fallback().answer(question, profile)


def _chat(agent, fallback, mode: str, show_evidence: bool) -> int:
    label = "Claude" if agent.mode == "llm" else "modo básico (sin LLM)"
    print(f"Agente de nacimientos DANE 2018 — {label}.")
    print("Escribe tu pregunta; «nuevo» reinicia la conversación y «salir» termina.\n")
    while True:
        try:
            question = input("Tú: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not question:
            continue
        if question.casefold() in EXIT_WORDS:
            return 0
        if question.casefold() == "nuevo":
            agent.reset()
            print("Conversación reiniciada.\n")
            continue
        try:
            response = _ask(agent, fallback, question, None, mode)
        except Exception as error:  # noqa: BLE001 - mostrar y seguir el chat
            print(f"Error: {error}\n", file=sys.stderr)
            continue
        print("Agente: ", end="")
        _print_response(response, show_evidence)
        print()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Agente de consulta de nacimientos DANE 2018 (datos y modelos)."
    )
    parser.add_argument(
        "--question",
        "-q",
        help="Pregunta única. Si se omite, abre un chat interactivo.",
    )
    parser.add_argument(
        "--profile",
        help="Objeto JSON con las 10 variables requeridas para solicitar predicciones.",
    )
    parser.add_argument(
        "--profile-file",
        type=Path,
        help="Archivo JSON con las 10 variables del perfil de predicción.",
    )
    parser.add_argument(
        "--modo",
        choices=["auto", "llm", "basico"],
        default="auto",
        help="auto (Claude si hay ANTHROPIC_API_KEY), llm o basico. Por defecto: auto.",
    )
    parser.add_argument(
        "--sin-evidencia",
        action="store_true",
        help="No imprimir el JSON de evidencia de las herramientas.",
    )
    args = parser.parse_args()
    if args.profile is not None and args.profile_file is not None:
        parser.error("Usa --profile o --profile-file, no ambos.")
    profile_text = args.profile
    if args.profile_file is not None:
        try:
            profile_text = args.profile_file.read_text(encoding="utf-8")
        except OSError as error:
            parser.error(f"No se pudo leer --profile-file: {error}")
    profile = _parse_profile(profile_text, parser)

    if args.modo == "llm" and not llm_credentials_available():
        print(
            "[Aviso] No se encontró ANTHROPIC_API_KEY en el entorno ni en .env; "
            "se intentará con otras credenciales de Anthropic.",
            file=sys.stderr,
        )

    try:
        agent = create_agent(args.modo)
    except (FileNotFoundError, ValueError) as error:
        print(f"Error al cargar los datos del agente: {error}", file=sys.stderr)
        return 2
    except RuntimeError as error:  # AgentError: sin credenciales
        print(f"Error al iniciar el agente: {error}", file=sys.stderr)
        return 3

    def fallback() -> BirthsAgent:
        return BirthsAgent(agent.data, agent.source)

    if args.question is None:
        return _chat(agent, fallback, args.modo, show_evidence=False)

    try:
        response = _ask(agent, fallback, args.question, profile, args.modo)
    except RuntimeError as error:
        print(f"Error del agente: {error}", file=sys.stderr)
        return 3
    _print_response(response, show_evidence=not args.sin_evidencia)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
