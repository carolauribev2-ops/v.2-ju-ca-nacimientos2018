"""Focused tests for tool counts, denominators, filters and profile validation."""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from agent.agent import BirthsAgent
from agent.catalog import find_departments, label_for
from agent.tools import (
    dataset_overview,
    grouped_outcome,
    model_metrics,
    outcome_summary,
    predict_risks,
)

METRICS_FIXTURE = {
    "generado": "2026-11-23T20:00:00",
    "validacion": "Validación cruzada estratificada de 5 particiones",
    "variables_modelo": ["EDAD_MADRE"],
    "modelos": {
        "CESAREA": {
            "archivo": "models/cesarea_pipeline.pkl",
            "modelo_seleccionado": "hist_gradient_boosting",
            "muestras_entrenamiento": 100,
            "tasa_positivos": 0.45,
            "comparacion_cv": {
                "logistic_regression": {
                    "roc_auc": 0.61, "roc_auc_std": 0.01,
                    "avg_precision": 0.55, "avg_precision_std": 0.01,
                },
                "hist_gradient_boosting": {
                    "roc_auc": 0.66, "roc_auc_std": 0.002,
                    "avg_precision": 0.60, "avg_precision_std": 0.003,
                },
            },
            "calibracion": {
                "particion_prueba": 0.2,
                "tasa_real_prueba": 0.45,
                "sin_calibrar": {"probabilidad_media": 0.50, "brier": 0.23, "roc_auc": 0.66},
                "calibrado": {"probabilidad_media": 0.45, "brier": 0.22, "roc_auc": 0.66},
            },
        }
    },
}


def _write_metrics_fixture(directory: str) -> Path:
    path = Path(directory) / "metricas.json"
    path.write_text(json.dumps(METRICS_FIXTURE), encoding="utf-8")
    return path

SAMPLE_PROFILE = json.loads(
    (Path(__file__).parent / "sample_profile.json").read_text(encoding="utf-8")
)


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
                    "group_label": "Antioquia",
                    "record_count": 2,
                    "valid_outcome_count": 2,
                    "positive_count": 1,
                    "unknown_or_missing_count": 0,
                    "positive_rate": 0.5,
                },
                {
                    "group_value": 11,
                    "group_label": "Bogotá, D.C.",
                    "record_count": 1,
                    "valid_outcome_count": 1,
                    "positive_count": 1,
                    "unknown_or_missing_count": 0,
                    "positive_rate": 1.0,
                },
                {
                    "group_value": "sin_dato",
                    "group_label": "Sin dato",
                    "record_count": 1,
                    "valid_outcome_count": 0,
                    "positive_count": 0,
                    "unknown_or_missing_count": 1,
                    "positive_rate": None,
                },
            ],
        )

    def test_filters_restrict_the_records_and_are_reported(self) -> None:
        result = dataset_overview(
            self.data, "fixture.csv", [{"variable": "CODPTORE", "codes": [5]}]
        )

        self.assertEqual(result["record_count"], 2)
        self.assertEqual(result["outcomes"]["CESAREA"]["positive_count"], 1)
        self.assertEqual(result["filters"][0]["labels"], ["Antioquia"])

    def test_filter_on_unknown_variable_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "No se puede filtrar"):
            dataset_overview(self.data, "fixture.csv", {"APGAR1": [1]})

    def test_unknown_birth_weight_is_not_counted_as_normal(self) -> None:
        data = pd.DataFrame({"BAJO_PESO": [1, 0, 0], "PESO_NAC": [3, 6, 9]})

        result = outcome_summary(data, "BAJO_PESO", "fixture.csv")

        self.assertEqual(result["records_with_valid_outcome"], 2)
        self.assertEqual(result["unknown_or_missing_count"], 1)
        self.assertAlmostEqual(result["positive_rate"], 0.5)

    def test_prediction_rejects_incomplete_profile_before_loading_models(self) -> None:
        with self.assertRaisesRegex(ValueError, "faltan"):
            predict_risks({"EDAD_MADRE": 4})

    def test_prediction_rejects_age_in_years_instead_of_code(self) -> None:
        with self.assertRaisesRegex(ValueError, "EDAD_MADRE=35"):
            predict_risks({**SAMPLE_PROFILE, "EDAD_MADRE": 35})

    def test_prediction_returns_probabilities_for_sample_profile(self) -> None:
        result = predict_risks(SAMPLE_PROFILE)

        for probability in result["probabilities"].values():
            self.assertGreaterEqual(probability, 0.0)
            self.assertLessEqual(probability, 1.0)
        self.assertEqual(result["profile_labels"]["CODPTORE"], "Bogotá, D.C.")

    def test_model_metrics_reads_saved_cross_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = model_metrics(_write_metrics_fixture(directory))

        cesarea = result["models"]["CESAREA"]
        self.assertEqual(cesarea["selected_model_label"], "HistGradientBoosting")
        self.assertEqual(cesarea["average_precision_baseline"], 0.45)
        self.assertEqual(len(cesarea["cross_validation"]), 2)
        self.assertEqual(cesarea["calibration"]["mean_probability_after"], 0.45)

    def test_model_metrics_without_file_explains_how_to_generate(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "main.py"):
            model_metrics(Path(tempfile.gettempdir()) / "no_existe_metricas.json")


class CatalogTests(unittest.TestCase):
    def test_find_departments_prefers_longest_names(self) -> None:
        self.assertEqual(
            find_departments("Compara Norte de Santander con Valle del Cauca"),
            [54, 76],
        )

    def test_label_for_unknown_code_is_explicit(self) -> None:
        self.assertIn("sin etiqueta", label_for("CODPTORE", 777))
        self.assertEqual(label_for("T_GES", 3), "28-37 semanas")


class BasicAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data = pd.DataFrame(
            {
                "CESAREA": [1, 0, 1, 1],
                "BAJO_PESO": [0, 1, 0, 0],
                "CODPTORE": [5, 5, 11, 11],
                "AREA_RES": [1, 3, 1, 1],
            }
        )
        self.agent = BirthsAgent(self.data, "fixture.csv")

    def test_question_for_total_and_one_rate_uses_overview_tool(self) -> None:
        response = self.agent.answer(
            "¿Cuántos nacimientos hay y cuál es la tasa de cesárea?"
        )

        self.assertEqual(response.tool_name, "dataset_overview")
        self.assertEqual(response.evidence["record_count"], 4)

    def test_department_question_applies_filter_instead_of_national_rate(self) -> None:
        response = self.agent.answer("¿Cuál es la tasa de cesárea en Antioquia?")

        self.assertEqual(response.tool_name, "outcome_summary")
        self.assertEqual(response.evidence["record_count"], 2)
        self.assertAlmostEqual(response.evidence["positive_rate"], 0.5)
        self.assertIn("Antioquia", response.answer)

    def test_grouped_question_with_filter(self) -> None:
        response = self.agent.answer("cesárea por departamento en zona rural")

        self.assertEqual(response.tool_name, "grouped_outcome")
        groups = response.evidence["results"][0]["groups"]
        self.assertEqual([group["group_value"] for group in groups], [5])

    def test_model_quality_question_uses_metrics_tool(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = _write_metrics_fixture(directory)
            self.agent.tools["model_metrics"] = lambda: model_metrics(path)
            response = self.agent.answer("¿Qué tan bueno es el modelo?")

        self.assertEqual(response.tool_name, "model_metrics")
        self.assertIn("ROC-AUC 0,660", response.answer)
        self.assertIn("calibradas: promedio 45,0 %", response.answer)

    def _ranking_agent(self) -> BirthsAgent:
        rows = []
        # (department, records, C-sections): -1 is unknown and 13 is too small.
        for code, total, positives in [(5, 100, 20), (11, 100, 50), (27, 100, 80),
                                       (-1, 100, 90), (13, 10, 10)]:
            rows += [{"CODPTORE": code, "CESAREA": int(i < positives), "BAJO_PESO": 0}
                     for i in range(total)]
        return BirthsAgent(pd.DataFrame(rows), "fixture.csv")

    def test_ranking_skips_unknown_and_small_groups(self) -> None:
        response = self._ranking_agent().answer(
            "¿Cuáles son los 2 departamentos con mayor tasa de cesárea?"
        )

        self.assertEqual(response.tool_name, "grouped_outcome")
        self.assertIn("1. Chocó (27)", response.answer)
        self.assertIn("2. Bogotá, D.C. (11)", response.answer)
        self.assertNotIn("Antioquia", response.answer)
        self.assertIn("Total nacional", response.answer)

    def test_named_departments_are_compared_side_by_side(self) -> None:
        response = self._ranking_agent().answer(
            "Compara la cesárea entre Antioquia y Chocó"
        )

        groups = response.evidence["results"][0]["groups"]
        self.assertEqual([group["group_value"] for group in groups], [5, 27])
        self.assertIn("20,0 %", response.answer)
        self.assertIn("80,0 %", response.answer)

    def test_other_year_is_not_answered_with_2018_data(self) -> None:
        response = self.agent.answer("¿Cuántos nacimientos hubo en 2020?")

        self.assertIsNone(response.tool_name)
        self.assertIn("2018", response.answer)

    def test_unsupported_question_does_not_claim_data(self) -> None:
        response = self.agent.answer("¿Qué ocurrió en otro año?")

        self.assertIsNone(response.tool_name)
        self.assertIsNone(response.evidence)


if __name__ == "__main__":
    unittest.main()
