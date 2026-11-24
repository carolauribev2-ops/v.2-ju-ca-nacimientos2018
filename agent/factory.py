"""Crea el agente adecuado: Claude si hay credenciales, modo básico si no."""

from __future__ import annotations

import os
from typing import Literal

import pandas as pd
from dotenv import load_dotenv

from .agent import BirthsAgent
from .tools import PROJECT_ROOT, load_birth_data

Mode = Literal["auto", "llm", "basico"]

load_dotenv(PROJECT_ROOT / ".env", override=False)


def llm_credentials_available() -> bool:
    """True si hay una credencial de Anthropic en el entorno o en .env."""
    return bool(
        os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
    )


def create_agent(
    mode: Mode = "auto",
    data: pd.DataFrame | None = None,
    source: str | None = None,
):
    """Devuelve ClaudeBirthsAgent (modo llm) o BirthsAgent (modo básico)."""
    if mode not in ("auto", "llm", "basico"):
        raise ValueError(f"Modo no soportado: {mode!r}. Usa auto, llm o basico.")
    if data is None:
        data, source = load_birth_data()
    use_llm = mode == "llm" or (mode == "auto" and llm_credentials_available())
    if not use_llm:
        return BirthsAgent(data, source)

    from .llm_agent import ClaudeBirthsAgent

    return ClaudeBirthsAgent(data, source)
