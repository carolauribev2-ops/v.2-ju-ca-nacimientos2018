"""Command-line entry point for the local births-data agent."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .agent import BirthsAgent


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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Consulta local de nacimientos DANE 2018 basada en datos y modelos."
    )
    parser.add_argument(
        "--question",
        "-q",
        required=True,
        help="Pregunta sobre conteos, tasas, grupos o predicciones.",
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

    try:
        agent = BirthsAgent()
    except (FileNotFoundError, ValueError) as error:
        print(f"Error al cargar los datos del agente: {error}", file=sys.stderr)
        return 2

    response = agent.answer(args.question, profile)
    print(response.answer)
    if response.evidence is not None:
        print(
            json.dumps(
                {"tool": response.tool_name, "evidence": response.evidence},
                ensure_ascii=False,
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
