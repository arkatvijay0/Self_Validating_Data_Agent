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


class MultiMetricGroupedMeanTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            "age": [20, 30, 40, 50, 35, 45],
            "balance": [100, 200, 300, 400, 250, 350],
            "campaign": [1, 2, 3, 4, 2, 1],
            "y": ["no", "yes", "no", "yes", "no", "yes"],
        })

        self.temp = tempfile.TemporaryDirectory()
        self.csv_path = Path(self.temp.name) / "bank.csv"
        self.frame.to_csv(self.csv_path, index=False)

        self.profile = {
            "columns": {
                "age": {"role": "continuous"},
                "balance": {"role": "continuous"},
                "campaign": {"role": "continuous"},
                "y": {"role": "categorical"},
            }
        }

        self.question = (
            "Compare the average age, balance, and campaign "
            "between customers who subscribed and those who did not."
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_analyzer_preserves_all_requested_metrics(self):
        task = analyze_task_mock(self.question, self.profile)

        self.assertEqual(task["task_type"], "comparison")
        self.assertEqual(
            set(task.get("requested_metric_columns", [])),
            {"age", "balance", "campaign"},
            "Task analysis must retain every requested numeric metric.",
        )

    def test_planner_creates_steps_for_every_requested_metric(self):
        task = analyze_task_mock(self.question, self.profile)
        plan = generate_plan_mock(
            task["task_type"],
            self.profile,
            self.question,
        )

        plan_text = "\n".join(plan).lower()

        for metric in ("age", "balance", "campaign"):
            with self.subTest(metric=metric):
                self.assertIn(
                    f"'{metric}'",
                    plan_text,
                    f"The plan must include the requested metric: {metric}",
                )

    def test_each_metric_can_be_calculated_independently(self):
        expected = {
            "age": {"no": 31.6666666667, "yes": 41.6666666667},
            "balance": {"no": 216.6666666667, "yes": 316.6666666667},
            "campaign": {"no": 2.0, "yes": 2.3333333333},
        }

        for metric, means in expected.items():
            with self.subTest(metric=metric):
                step = (
                    f"Compare the average '{metric}' grouped by "
                    "subscription outcome using 'y' (yes/no)."
                )

                code = generate_code_mock(
                    step,
                    str(self.csv_path),
                    data_profile=self.profile,
                )

                namespace = {}
                exec(code, namespace)
                result = namespace["result"]

                self.assertEqual(result["metric_column"], metric)

                self.assertAlmostEqual(
                    result["group_results"]["no"]["mean_value"],
                    means["no"],
                    places=6,
                )
                self.assertAlmostEqual(
                    result["group_results"]["yes"]["mean_value"],
                    means["yes"],
                    places=6,
                )


if __name__ == "__main__":
    unittest.main()


