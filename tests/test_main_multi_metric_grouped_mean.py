import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from main import (
    build_multi_metric_grouped_mean_result,
    generate_multi_metric_grouped_mean_insight,
    validate_multi_metric_grouped_mean_insight,
    create_evidence,
    build_multi_metric_evidence_mapping,
)


def make_result(metric, no_mean, yes_mean, no_count=3, yes_count=3):
    return {
        "status": "PASS",
        "analysis_type": "grouped_mean_analysis",
        "metric_column": metric,
        "group_column": "y",
        "group_results": {
            "no": {
                "row_count": no_count,
                "valid_value_count": no_count,
                "mean_value": no_mean,
            },
            "yes": {
                "row_count": yes_count,
                "valid_value_count": yes_count,
                "mean_value": yes_mean,
            },
        },
        "mean_difference_yes_minus_no": yes_mean - no_mean,
    }


class MultiMetricPipelineIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.task = {
            "task_type": "comparison",
            "group_column": "y",
            "requested_metric_columns": ["age", "balance", "campaign"],
        }
        values = {
            "age": (31.6666666667, 41.6666666667),
            "balance": (216.6666666667, 316.6666666667),
            "campaign": (2.0, 2.3333333333),
        }
        self.execution = {
            "steps": [
                {
                    "step_number": i,
                    "step": f"Compare the average '{metric}' grouped by y",
                    "status": "PASS",
                    "execution_status": "PASS",
                    "validation_status": "PASS",
                    "final_result": make_result(metric, *means),
                }
                for i, (metric, means) in enumerate(values.items(), start=1)
            ]
        }

    def test_combined_result_contains_every_requested_metric(self):
        combined = build_multi_metric_grouped_mean_result(self.task, self.execution)
        self.assertEqual(combined["status"], "PASS")
        self.assertEqual(
            set(combined["metric_results"]),
            {"age", "balance", "campaign"},
        )
        self.assertEqual(combined["requested_metric_count"], 3)
        self.assertEqual(combined["validated_metric_count"], 3)

    def test_missing_metric_forces_failure(self):
        incomplete = {
            "steps": self.execution["steps"][:2],
        }
        combined = build_multi_metric_grouped_mean_result(self.task, incomplete)
        self.assertEqual(combined["status"], "FAIL")
        self.assertIn("campaign", combined["missing_metrics"])

    def test_insight_and_metric_validation_cover_all_metrics(self):
        combined = build_multi_metric_grouped_mean_result(self.task, self.execution)
        insight = generate_multi_metric_grouped_mean_insight(combined)
        validation = validate_multi_metric_grouped_mean_insight(
            insight,
            combined,
            base_validation={"overall": "PASS", "pass": True},
        )
        self.assertEqual(validation["overall"], "PASS")
        for metric in self.task["requested_metric_columns"]:
            self.assertIn(metric, insight.lower())
            self.assertTrue(validation["checks"]["multi_metric_claim_validation"]["metrics"][metric]["pass"])

    def test_evidence_and_mapping_keep_all_metrics(self):
        combined = build_multi_metric_grouped_mean_result(self.task, self.execution)
        insight = generate_multi_metric_grouped_mean_insight(combined)
        validation = validate_multi_metric_grouped_mean_insight(
            insight,
            combined,
            base_validation={"overall": "PASS", "pass": True},
        )
        evidence = create_evidence(
            analysis_type="multi_metric_grouped_mean_analysis",
            primary_result=combined,
            grouped_series=None,
            execution_results=self.execution,
            data_profile={},
            dataset_path=None,
        )
        mapping = build_multi_metric_evidence_mapping(insight, evidence, validation)
        self.assertEqual(evidence["status"], "PASS")
        self.assertEqual(mapping["status"], "PASS")
        self.assertEqual(set(evidence["metric_results"]), {"age", "balance", "campaign"})


if __name__ == "__main__":
    unittest.main()
