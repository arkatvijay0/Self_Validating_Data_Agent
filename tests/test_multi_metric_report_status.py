import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from final_report import create_final_report


class MultiMetricReportStatusTests(unittest.TestCase):
    def make_validation(self, *, pass_all=True):
        metrics = {
            "age": {
                "result_valid": True,
                "metric_named": True,
                "non_subscriber_mean_present": True,
                "subscriber_mean_present": True,
                "difference_present": True,
                "pass": True,
            },
            "balance": {
                "result_valid": True,
                "metric_named": True,
                "non_subscriber_mean_present": True,
                "subscriber_mean_present": True,
                "difference_present": True,
                "pass": True,
            },
            "campaign": {
                "result_valid": True,
                "metric_named": pass_all,
                "non_subscriber_mean_present": True,
                "subscriber_mean_present": True,
                "difference_present": True,
                "pass": pass_all,
            },
        }
        return {
            "overall": "PASS" if pass_all else "FAIL",
            "pass": pass_all,
            "checks": {
                "multi_metric_claim_validation": {
                    "pass": pass_all,
                    "requested_metric_count": 3,
                    "validated_metric_count": 3,
                    "metrics": metrics,
                },
                "base_insight_validation": {
                    "pass": True,
                    "result": {"overall": "PASS", "pass": True},
                },
            },
        }

    def make_report(self, validation):
        result = {
            "status": "PASS",
            "analysis_type": "multi_metric_grouped_mean_analysis",
            "requested_metric_columns": ["age", "balance", "campaign"],
            "metric_results": {},
        }
        evidence = {
            "status": "PASS",
            "analysis_type": "multi_metric_grouped_mean_analysis",
        }
        mapping = {
            "status": "PASS" if validation["pass"] else "FAIL",
            "traceability_status": "PASS",
            "analysis_type": "multi_metric_grouped_mean_analysis",
            "claim_validation": {
                "status": "PASS" if validation["pass"] else "FAIL",
                "total_claims": 3,
                "passed_claims": 3 if validation["pass"] else 2,
                "failed_claims": 0 if validation["pass"] else 1,
            },
        }
        return create_final_report(
            user_question=(
                "Compare the average age, balance, and campaign between "
                "customers who subscribed and those who did not."
            ),
            analysis_type="multi_metric_grouped_mean_analysis",
            validated_result=result,
            insight=(
                "age: non-subscriber average = 40.84, subscriber average = 41.67, "
                "difference = +0.83. balance: non-subscriber average = 1,303.71, "
                "subscriber average = 1,804.27, difference = +500.55. campaign: "
                "non-subscriber average = 2.85, subscriber average = 2.14, "
                "difference = -0.71."
            ),
            insight_validation=validation,
            evidence=evidence,
            recommendation={
                "status": "PASS",
                "recommendation": "Review these descriptive differences alongside other factors.",
            },
            insight_evidence_mapping=mapping,
        )

    def test_multi_metric_checks_do_not_default_to_legacy_failures(self):
        report = self.make_report(self.make_validation(pass_all=True))
        self.assertIn("Overall Validation: PASS", report)
        self.assertIn("Presence Check: PASS", report)
        self.assertIn("Result Check: PASS", report)
        self.assertIn("Claim-Level Validation: PASS", report)
        self.assertIn("\nPASS\n", report)

    def test_failed_metric_presence_remains_a_failure(self):
        report = self.make_report(self.make_validation(pass_all=False))
        self.assertIn("Overall Validation: FAIL", report)
        self.assertIn("Presence Check: FAIL", report)
        self.assertIn("Result Check: FAIL", report)
        self.assertIn("Claim-Level Validation: FAIL", report)
        self.assertIn("\nFAIL\n", report)


if __name__ == "__main__":
    unittest.main()
