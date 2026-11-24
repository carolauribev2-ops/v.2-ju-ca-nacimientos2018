"""Página de chat con el agente de nacimientos DANE 2018.

Run (desde la raíz del proyecto):
    uv run streamlit run app/main.py   → menú lateral «Asistente»
"""

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.agent import BirthsAgent  # noqa: E402
from agent.factory import llm_credentials_available  # noqa: E402
from agent.tools import load_birth_data  # noqa: E402

st.set_page_config(page_title="DANE · Asistente", page_icon="💬", layout="wide")

# Guion de demostración: funciona en modo básico y con Claude.
EXAMPLES = [
    "¿Cuántos nacimientos hubo en 2018 y cuál fue la tasa de cesárea?",
    "¿Cuál es la tasa de cesárea en Antioquia?",
    "Compara la cesárea entre Antioquia, Bogotá y Chocó",
    "¿Cuáles son los 5 departamentos con mayor tasa de cesárea?",
    "Bajo peso por edad de la madre en zona rural",
    "¿Cuál es la tasa de bajo peso en prematuros?",
    "¿Qué tan bueno es el modelo?",
    "¿Cuántos nacimientos hubo en 2020?",
]
MODE_LLM = "Claude (LLM)"
MODE_BASIC = "Básico (sin LLM)"


@st.cache_resource(show_spinner="Cargando microdatos DANE 2018…")
def _load_data():
    return load_birth_data()


def _new_agent(mode: str):
    data, source = _load_data()
    if mode == MODE_LLM:
        from agent.llm_agent import ClaudeBirthsAgent

        return ClaudeBirthsAgent(data, source)
    return BirthsAgent(data, source)


def _render_evidence(tool_name: str | None, evidence: dict | None) -> None:
    if evidence is None:
        return
    with st.expander(f"Evidencia · herramientas: {tool_name}"):
        st.json(evidence, expanded=False)


has_credentials = llm_credentials_available()
with st.sidebar:
    st.markdown("### Asistente de datos")
    modes = [MODE_LLM, MODE_BASIC] if has_credentials else [MODE_BASIC]
    mode = st.radio("Modo del agente", modes)
    if not has_credentials:
        st.caption(
            "Modo básico: responde con reglas, sin LLM. Para activar Claude, define "
            "`ANTHROPIC_API_KEY` en `.env` y reinicia la app."
        )
    if st.button("Nueva conversación", width="stretch"):
        st.session_state.pop("agent", None)
        st.session_state.history = []

    st.markdown("### Preguntas de ejemplo")
    for index, example in enumerate(EXAMPLES):
        if st.button(example, key=f"example_{index}", width="stretch"):
            st.session_state.pending_question = example

if st.session_state.get("agent_mode") != mode or "agent" not in st.session_state:
    try:
        st.session_state.agent = _new_agent(mode)
    except RuntimeError as error:  # AgentError: credencial inválida o ausente
        st.error(str(error))
        st.stop()
    st.session_state.agent_mode = mode
    st.session_state.history = []

st.title("Asistente de nacimientos DANE 2018")
st.caption(
    "Las cifras se calculan con los microdatos DANE 2018 procesados y los modelos del "
    "proyecto; cada respuesta muestra la evidencia de las herramientas usadas. Las "
    "predicciones no son consejo clínico."
)

if not st.session_state.history and "pending_question" not in st.session_state:
    st.info("Escribe una pregunta abajo o elige una de las preguntas de ejemplo del menú lateral.")

for message in st.session_state.history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        _render_evidence(message.get("tool_name"), message.get("evidence"))

typed = st.chat_input("Pregunta sobre los nacimientos de 2018…")
if question := typed or st.session_state.pop("pending_question", None):
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Consultando los datos…"):
            try:
                response = st.session_state.agent.answer(question)
            except RuntimeError as error:  # AgentError de la API
                st.error(str(error))
                st.stop()
        st.markdown(response.answer)
        _render_evidence(response.tool_name, response.evidence)

    st.session_state.history.append(
        {
            "role": "assistant",
            "content": response.answer,
            "tool_name": response.tool_name,
            "evidence": response.evidence,
        }
    )
