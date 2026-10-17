"""Focused tests for claim-to-evidence outcome classification.

Run from the repository root with:
    & "$HOME\\anaconda3\\envs\\data_agent\\python.exe" .\\src\\test_claim_evidence_verification.py
"""

import unittest

from claim_evidence_verification import (
    CONTRADICTED,
    INSUFFICIENT_EVIDENCE,
    SUPPORTED,
    verify_claim_against_evidence,
    verify_claims,
)


class ClaimEvidenceVerificationTests(unittest.TestCase):
    def setUp(self):
        self.evidence = {
            "status": "PASS",
            "analysis_type": "mean_analysis",
            "supporting_result": {
                "analysis_type": "mean_analysis",
                "metric_column": "Total Profit",
                "mean": 10.0,
                "valid_value_count": 3,
            },
        }

    def test_matching_mean_is_supported(self):
        result = verify_claim_against_evidence(
            "The average Total Profit is 10.00.", self.evidence
        )
        self.assertEqual(result["outcome"], SUPPORTED)
        self.assertTrue(result["pass"])

    def test_mismatched_mean_is_contradicted(self):
        result = verify_claim_against_evidence(
            "The average Total Profit is 99.00.", self.evidence
        )
        self.assertEqual(result["outcome"], CONTRADICTED)
        self.assertFalse(result["pass"])

    def test_unvalidated_evidence_is_insufficient(self):
        evidence = dict(self.evidence, status="FAIL")
        result = verify_claim_against_evidence(
            "The average Total Profit is 10.00.", evidence
        )
        self.assertEqual(result["outcome"], INSUFFICIENT_EVIDENCE)

    def test_empty_claim_is_insufficient(self):
        result = verify_claim_against_evidence("", self.evidence)
        self.assertEqual(result["outcome"], INSUFFICIENT_EVIDENCE)

    def test_batch_counts(self):
        result = verify_claims(
            [
                "The average Total Profit is 10.00.",
                "The average Total Profit is 99.00.",
            ],
            self.evidence,
        )
        self.assertEqual(result["claim_count"], 2)
        self.assertEqual(result["supported_count"], 1)
        self.assertEqual(result["contradicted_count"], 1)
        self.assertEqual(result["overall_outcome"], CONTRADICTED)


if __name__ == "__main__":
    unittest.main(verbosity=2)
