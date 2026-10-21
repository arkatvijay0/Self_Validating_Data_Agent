import os
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from task_analyzer import analyze_task_mock
from eda_planner import generate_plan_mock
from code_generator import generate_code_mock
from insight_generator import generate_insight
from evidence_mapping_grouped_mean_fixed import map_insight_to_evidence


class DynamicGroupedMeanMetricTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            "age": [20, 30, 40, 50],
            "balance": [100, 200, 300, 400],
            "y": ["no", "yes", "no", "yes"],
        })
        self.temp = tempfile.TemporaryDirectory()
        self.csv_path = Path(self.temp.name) / "bank.csv"
        self.frame.to_csv(self.csv_path, index=False)
        self.profile = {
            "columns": {
                "age": {"role": "continuous"},
                "balance": {"role": "continuous"},
                "y": {"role": "categorical"},
            }
        }

    def tearDown(self):
        self.temp.cleanup()

    def test_task_analyzer_selects_age_not_balance(self):
        task = analyze_task_mock(
            "Compare average age between customers who subscribed and those who did not.",
            self.profile,
        )
        self.assertEqual(task["metric_column"], "age")
        self.assertFalse(task.get("unsupported_metric", False))

    def test_planner_preserves_age_metric(self):
        task = analyze_task_mock(
            "Compare average age between customers who subscribed and those who did not.",
            self.profile,
        )
        plan = generate_plan_mock(task["task_type"], self.profile,
            "Compare average age between customers who subscribed and those who did not.")
        self.assertTrue(any("'age'" in step for step in plan))
        self.assertFalse(any("average balance" in step.lower() for step in plan))

    def test_generated_code_calculates_requested_metric(self):
        step = "Compare the average 'age' grouped by subscription outcome using 'y' (yes/no)."
        code = generate_code_mock(step, str(self.csv_path), data_profile=self.profile)
        namespace = {}
        exec(code, namespace)
        result = namespace["result"]
        self.assertEqual(result["metric_column"], "age")
        self.assertEqual(result["group_results"]["no"]["mean_value"], 30.0)
        self.assertEqual(result["group_results"]["yes"]["mean_value"], 40.0)
        self.assertEqual(result["mean_difference_yes_minus_no"], 10.0)

    def test_age_insight_mentions_age_and_correct_means(self):
        result = {
            "status": "PASS",
            "analysis_type": "grouped_mean_analysis",
            "metric_column": "age",
            "group_column": "y",
            "group_results": {
                "no": {"row_count": 2, "valid_value_count": 2, "mean_value": 30.0, "mean": 30.0},
                "yes": {"row_count": 2, "valid_value_count": 2, "mean_value": 40.0, "mean": 40.0},
            },
            "mean_difference_yes_minus_no": 10.0,
        }
        insight = (
            "The average age was 30.00 for non-subscribers and 40.00 for subscribers. "
            "The subscriber group average was 10.00 higher."
        )
        validation = map_insight_to_evidence(
            insight,
            {**result, "status": "PASS"},
        )
        self.assertEqual(validation["status"], "PASS")

    def test_wrong_metric_insight_fails_validation(self):
        result = {
            "status": "PASS",
            "analysis_type": "grouped_mean_analysis",
            "metric_column": "age",
            "group_column": "y",
            "group_results": {
                "no": {"row_count": 2, "valid_value_count": 2, "mean_value": 30.0},
                "yes": {"row_count": 2, "valid_value_count": 2, "mean_value": 40.0},
            },
            "mean_difference_yes_minus_no": 10.0,
        }
        insight = (
            "The average balance was 30.00 for non-subscribers and 40.00 for subscribers. "
            "The subscriber group average was 10.00 higher."
        )
        validation = map_insight_to_evidence(
            insight,
            {**result, "status": "PASS"},
        )
        self.assertEqual(validation["status"], "FAIL")
        self.assertTrue(any(
            item["check"] == "requested_metric_mentioned" and item["status"] == "FAIL"
            for item in validation["checks"]
        ))


if __name__ == "__main__":
    unittest.main()
