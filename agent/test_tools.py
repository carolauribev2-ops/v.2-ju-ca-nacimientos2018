"""Focused tests for tool counts, denominators, and profile validation."""

import unittest

import pandas as pd

from agent.agent import BirthsAgent
from agent.tools import grouped_outcome, outcome_summary, predict_risks


class BirthToolsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data = pd.DataFrame(
            {
                "CESAREA": [1, 0, 1, None],
                "BAJO_PESO": [0, 1, 1, None],
                "CODPTORE": [5, 5, 11, None],
            }
        )

    def test_outcome_summary_uses_only_valid_binary_labels(self) -> None:
        result = outcome_summary(self.data, "CESAREA", "fixture.csv")

        self.assertEqual(result["positive_count"], 2)
        self.assertEqual(result["non_positive_count"], 1)
        self.assertEqual(result["records_with_valid_outcome"], 3)
        self.assertEqual(result["unknown_or_missing_count"], 1)
        self.assertAlmostEqual(result["positive_rate"], 2 / 3)

    def test_grouped_outcome_keeps_missing_groups_and_excludes_unknown_labels(self) -> None:
        result = grouped_outcome(
            self.data, "CESAREA", "CODPTORE", "fixture.csv"
        )

        self.assertEqual(
            result["groups"],
            [
                {
                    "group_value": 5,
                    "record_count": 2,
                    "valid_outcome_count": 2,
                    "positive_count": 1,
                    "unknown_or_missing_count": 0,
                    "positive_rate": 0.5,
                },
                {
                    "group_value": 11,
                    "record_count": 1,
                    "valid_outcome_count": 1,
                    "positive_count": 1,
                    "unknown_or_missing_count": 0,
                    "positive_rate": 1.0,
                },
                {
                    "group_value": "sin_dato",
                    "record_count": 1,
                    "valid_outcome_count": 0,
                    "positive_count": 0,
                    "unknown_or_missing_count": 1,
                    "positive_rate": None,
                },
            ],
        )

    def test_prediction_rejects_incomplete_profile_before_loading_models(self) -> None:
        with self.assertRaisesRegex(ValueError, "faltan"):
            predict_risks({"EDAD_MADRE": 4})

    def test_question_for_total_and_one_rate_uses_overview_tool(self) -> None:
        data = pd.DataFrame(
            {
                "CESAREA": [1, 0],
                "BAJO_PESO": [0, 1],
            }
        )
        agent = BirthsAgent(data, "fixture.csv")

        response = agent.answer(
            "¿Cuántos nacimientos hay y cuál es la tasa de cesárea?"
        )

        self.assertEqual(response.tool_name, "dataset_overview")
        self.assertEqual(response.evidence["record_count"], 2)

    def test_unsupported_question_does_not_claim_data(self) -> None:
        data = pd.DataFrame({"CESAREA": [1], "BAJO_PESO": [0]})
        agent = BirthsAgent(data, "fixture.csv")

        response = agent.answer("¿Qué ocurrió en otro año?")

        self.assertIsNone(response.tool_name)
        self.assertIsNone(response.evidence)


if __name__ == "__main__":
    unittest.main()
