"""Catálogo de códigos DANE (Nacimientos 2018) usado por las herramientas del agente.

Las etiquetas provienen del diccionario oficial de microdatos del DANE
(https://microdatos.dane.gov.co/index.php/catalog/652/data-dictionary/F41?file_name=nac2018).
El código -1 no es del DANE: lo asigna ``src.preprocessing`` al imputar nulos.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

DEPARTAMENTOS: dict[int, str] = {
    5: "Antioquia",
    8: "Atlántico",
    11: "Bogotá, D.C.",
    13: "Bolívar",
    15: "Boyacá",
    17: "Caldas",
    18: "Caquetá",
    19: "Cauca",
    20: "Cesar",
    23: "Córdoba",
    25: "Cundinamarca",
    27: "Chocó",
    41: "Huila",
    44: "La Guajira",
    47: "Magdalena",
    50: "Meta",
    52: "Nariño",
    54: "Norte de Santander",
    63: "Quindío",
    66: "Risaralda",
    68: "Santander",
    70: "Sucre",
    73: "Tolima",
    76: "Valle del Cauca",
    81: "Arauca",
    85: "Casanare",
    86: "Putumayo",
    88: "San Andrés, Providencia y Santa Catalina",
    91: "Amazonas",
    94: "Guainía",
    95: "Guaviare",
    97: "Vaupés",
    99: "Vichada",
    -1: "Sin dato (nulo imputado como -1)",
}

VARIABLES: dict[str, dict[str, Any]] = {
    "CODPTORE": {
        "nombre": "Departamento de residencia habitual de la madre",
        "codigos": DEPARTAMENTOS,
    },
    "EDAD_MADRE": {
        "nombre": "Edad de la madre (grupo quinquenal, no años)",
        "codigos": {
            1: "10-14 años",
            2: "15-19 años",
            3: "20-24 años",
            4: "25-29 años",
            5: "30-34 años",
            6: "35-39 años",
            7: "40-44 años",
            8: "45-49 años",
            9: "50-54 años",
        },
        "nota": "El código 99 (sin información) se excluye en el preprocesamiento.",
    },
    "T_GES": {
        "nombre": "Tiempo de gestación, agrupación DANE (semanas)",
        "codigos": {
            1: "Menos de 22 semanas",
            2: "22-27 semanas",
            3: "28-37 semanas",
            4: "38-41 semanas",
            5: "42 semanas y más",
            6: "Ignorado",
            9: "Sin información",
        },
        "nota": (
            "El código 3 incluye la semana 37, que ya es a término. Para "
            "prematuridad (<37 semanas) usa T_GES_AGRU_CIE en {1, 2, 3}."
        ),
    },
    "T_GES_AGRU_CIE": {
        "nombre": "Tiempo de gestación, agrupación CIE (semanas)",
        "codigos": {
            1: "Menos de 22 semanas",
            2: "22-27 semanas",
            3: "28-36 semanas",
            4: "37-41 semanas",
            5: "42 semanas y más",
            6: "Ignorado",
            9: "Sin información",
        },
        "nota": "Prematuro (<37 semanas) = códigos 1, 2 y 3. No es variable del modelo.",
    },
    "AREA_RES": {
        "nombre": "Área de residencia habitual de la madre",
        "codigos": {
            1: "Cabecera municipal",
            2: "Centro poblado (inspección, corregimiento o caserío)",
            3: "Rural disperso",
            9: "Sin información",
            -1: "Sin dato (nulo imputado como -1)",
        },
    },
    "IDCLASADMI": {
        "nombre": "Entidad administradora en salud de la madre",
        "codigos": {
            1: "Entidad promotora de salud (EPS) - contributivo",
            2: "Entidad promotora de salud (EPS) - subsidiado",
            3: "Entidad adaptada de salud",
            4: "Entidad especial de salud",
            5: "Entidad exceptuada de salud",
            9: "Sin información",
            -1: "Sin entidad registrada (madre no asegurada)",
        },
        "nota": (
            "En el dataset crudo, -1 (nulo) coincide uno a uno con SEG_SOCIAL = 5 "
            "(no asegurado); los códigos 1 y 3 corresponden al régimen contributivo."
        ),
    },
    "SEXO": {
        "nombre": "Sexo del nacido vivo",
        "codigos": {1: "Masculino", 2: "Femenino"},
        "nota": "El código 3 (indeterminado) se excluye en el preprocesamiento.",
    },
    "MUL_PARTO": {
        "nombre": "Multiplicidad del embarazo",
        "codigos": {
            1: "Simple",
            2: "Doble",
            3: "Triple",
            4: "Cuádruple o más",
            9: "Sin información",
        },
    },
    "CESAREA": {
        "nombre": "Parto por cesárea (TIPO_PARTO = 2)",
        "codigos": {1: "Sí (cesárea)", 0: "No (espontáneo o instrumentado)"},
    },
    "BAJO_PESO": {
        "nombre": "Bajo peso al nacer (< 2.500 g; PESO_NAC 1-4)",
        "codigos": {1: "Sí (< 2.500 g)", 0: "No (≥ 2.500 g)"},
        "nota": "PESO_NAC = 9 (sin información) se trata como dato faltante.",
    },
}

NUMERIC_VARIABLES: dict[str, str] = {
    "NUMCONSUL": "Número de consultas prenatales (0 = ninguna; 99 = sin información)",
    "N_HIJOSV": "Hijos nacidos vivos que ha tenido la madre, incluido el presente",
    "N_EMB": "Número de embarazos, incluido el presente",
}


def label_for(variable: str, code: Any) -> str:
    """Return the DANE label for a code, or a neutral description if unknown."""
    if code is None or code == "sin_dato":
        return "Sin dato"
    codes = VARIABLES.get(variable, {}).get("codigos", {})
    try:
        numeric = float(code)
    except (TypeError, ValueError):
        return str(code)
    if numeric.is_integer() and int(numeric) in codes:
        return codes[int(numeric)]
    shown = int(numeric) if numeric.is_integer() else numeric
    return f"Código {shown} (sin etiqueta en el diccionario DANE)"


def catalog_as_text() -> str:
    """Render the catalog as stable plain text for the LLM system prompt."""
    lines = []
    for variable, info in VARIABLES.items():
        codes = "; ".join(f"{code} = {label}" for code, label in info["codigos"].items())
        lines.append(f"- {variable} ({info['nombre']}): {codes}.")
        if "nota" in info:
            lines.append(f"  Nota: {info['nota']}")
    for variable, description in NUMERIC_VARIABLES.items():
        lines.append(f"- {variable}: {description}.")
    return "\n".join(lines)


def normalize_text(text: str) -> str:
    """Lowercase, strip accents and collapse whitespace."""
    normalized = unicodedata.normalize("NFKD", text)
    without_accents = "".join(c for c in normalized if not unicodedata.combining(c))
    return " ".join(without_accents.casefold().split())


_DEPARTMENT_ALIASES: dict[str, int] = {
    normalize_text(name.split(",")[0]): code
    for code, name in DEPARTAMENTOS.items()
    if code > 0
}
_DEPARTMENT_ALIASES.update(
    {
        "bogota d.c.": 11,
        "bogota dc": 11,
        "san andres": 88,
        "providencia": 88,
        "guajira": 44,
        "valle": 76,
    }
)


def find_departments(text: str) -> list[int]:
    """Find department codes mentioned by name in free text (longest match first)."""
    remaining = normalize_text(text)
    found: list[int] = []
    for alias in sorted(_DEPARTMENT_ALIASES, key=len, reverse=True):
        pattern = rf"\b{re.escape(alias)}\b"
        if re.search(pattern, remaining):
            code = _DEPARTMENT_ALIASES[alias]
            if code not in found:
                found.append(code)
            remaining = re.sub(pattern, " ", remaining)
    return found
