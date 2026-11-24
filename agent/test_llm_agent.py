"""Tests del agente con Claude usando el SDK real y un transporte HTTP simulado.

No hacen llamadas de red ni consumen créditos: un MockTransport responde como la
API de Mensajes y registra cada solicitud para verificar el ciclo de herramientas.
"""

import json
import unittest

import anthropic
import httpx2
import pandas as pd

from agent.llm_agent import AgentError, ClaudeBirthsAgent


def _message(content, stop_reason):
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


class FakeApi:
    """Devuelve respuestas en orden y guarda los cuerpos de las solicitudes."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(json.loads(request.content))
        status, body = self.responses.pop(0)
        return httpx2.Response(status, json=body)

    def client(self) -> anthropic.Anthropic:
        return anthropic.Anthropic(
            api_key="sk-ant-test",
            max_retries=0,
            http_client=anthropic.DefaultHttpxClient(
                transport=httpx2.MockTransport(self.handler)
            ),
        )


class ClaudeAgentLoopTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data = pd.DataFrame(
            {
                "CESAREA": [1, 0, 1, 1],
                "BAJO_PESO": [0, 1, 0, 0],
                "CODPTORE": [5, 5, 11, 11],
            }
        )

    def _agent(self, api: FakeApi) -> ClaudeBirthsAgent:
        return ClaudeBirthsAgent(self.data, "fixture.csv", client=api.client())

    def test_tool_call_is_executed_and_answer_returned(self) -> None:
        api = FakeApi(
            [
                (200, _message(
                    [
                        {"type": "text", "text": "Consulto los datos."},
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "dataset_overview",
                            "input": {"filters": [{"variable": "CODPTORE", "codes": [5]}]},
                        },
                    ],
                    "tool_use",
                )),
                (200, _message(
                    [{"type": "text", "text": "En Antioquia la tasa de cesárea es 1/2 (50%)."}],
                    "end_turn",
                )),
            ]
        )
        agent = self._agent(api)

        response = agent.answer("¿Tasa de cesárea en Antioquia?")

        self.assertEqual(response.mode, "llm")
        self.assertIn("50%", response.answer)
        self.assertEqual(response.tool_name, "dataset_overview")
        call = response.evidence["tool_calls"][0]
        self.assertEqual(call["result"]["record_count"], 2)

        first, second = api.requests
        self.assertEqual(first["model"], "claude-opus-5-5")
        self.assertEqual(first["output_config"], {"effort": "medium"})
        self.assertEqual(first["fallbacks"], "default")
        self.assertEqual(first["cache_control"], {"type": "ephemeral"})
        self.assertEqual(
            {tool["name"] for tool in first["tools"]},
            {
                "dataset_overview",
                "grouped_outcome",
                "group_distribution",
                "predict_risks",
                "model_metrics",
            },
        )
        self.assertTrue(all(tool["strict"] for tool in first["tools"]))

        tool_result_message = second["messages"][-1]
        self.assertEqual(tool_result_message["role"], "user")
        [tool_result] = tool_result_message["content"]
        self.assertEqual(tool_result["tool_use_id"], "toolu_1")
        self.assertEqual(json.loads(tool_result["content"])["record_count"], 2)

    def test_follow_up_question_keeps_history(self) -> None:
        api = FakeApi(
            [
                (200, _message([{"type": "text", "text": "Hola."}], "end_turn")),
                (200, _message([{"type": "text", "text": "Sigo aquí."}], "end_turn")),
            ]
        )
        agent = self._agent(api)

        agent.answer("Hola")
        agent.answer("¿Y ahora?")

        roles = [message["role"] for message in api.requests[1]["messages"]]
        self.assertEqual(roles, ["user", "assistant", "user"])

    def test_invalid_tool_input_is_returned_as_error_result(self) -> None:
        api = FakeApi(
            [
                (200, _message(
                    [{
                        "type": "tool_use",
                        "id": "toolu_bad",
                        "name": "predict_risks",
                        "input": {"profile": {"EDAD_MADRE": 35}},
                    }],
                    "tool_use",
                )),
                (200, _message([{"type": "text", "text": "Faltan datos."}], "end_turn")),
            ]
        )
        agent = self._agent(api)

        response = agent.answer("Predice el riesgo de una madre de 35 años")

        [tool_result] = api.requests[1]["messages"][-1]["content"]
        self.assertTrue(tool_result["is_error"])
        self.assertIn("faltan", tool_result["content"])
        self.assertIn("error", response.evidence["tool_calls"][0])

    def test_api_error_is_translated_and_history_rolled_back(self) -> None:
        api = FakeApi(
            [
                (401, {
                    "type": "error",
                    "error": {"type": "authentication_error", "message": "invalid x-api-key"},
                }),
            ]
        )
        agent = self._agent(api)

        with self.assertRaisesRegex(AgentError, "ANTHROPIC_API_KEY"):
            agent.answer("Hola")
        self.assertEqual(agent.messages, [])


if __name__ == "__main__":
    unittest.main()
