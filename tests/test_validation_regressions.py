import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from validators import validate_result, self_validate
from insight_validator import (
    validate_result_presence,
    validate_grouped_mean_direction_claims,
)
from final_report import create_final_report


def grouped_balance_result():
    return {
        "status": "PASS",
        "analysis_type": "grouped_mean_analysis",
        "metric_column": "balance",
        "group_column": "y",
        "group_results": {
            "no": {"mean_value": 1303.714969, "mean": 1303.714969},
            "yes": {"mean_value": 1804.267915, "mean": 1804.267915},
        },
        "mean_difference_yes_minus_no": 500.552946,
    }


def test_unsupported_result_fails_result_validation():
    result = {
        "status": "UNSUPPORTED",
        "reason": "No code-generation rule supports this request yet.",
    }
    assert validate_result(result)["pass"] is False
    assert validate_result_presence(result)["pass"] is False


def test_incorrect_subscriber_lower_claim_fails():
    result = grouped_balance_result()
    validation = validate_grouped_mean_direction_claims(
        insight="Subscribers have a higher average balance than non-subscribers.",
        result=result,
        user_question=(
            "Validate the exact claim: subscribers have a lower average "
            "balance than non-subscribers."
        ),
    )
    assert validation["pass"] is False
    assert any(
        check.get("source") == "user_question" and check.get("pass") is False
        for check in validation["checks"]
    )


def test_correct_subscriber_higher_claim_passes():
    result = grouped_balance_result()
    validation = validate_grouped_mean_direction_claims(
        insight="Subscribers have a higher average balance than non-subscribers.",
        result=result,
        user_question="Compare average balance for subscribers and non-subscribers.",
    )
    assert validation["pass"] is True


def test_unsupported_primary_result_forces_report_reliability_fail():
    result = {
        "status": "UNSUPPORTED",
        "reason": "No code-generation rule supports this request yet.",
    }
    report = create_final_report(
        user_question="Test unsupported request",
        analysis_type="mean_analysis",
        validated_result=result,
        insight="An apparently plausible insight.",
        insight_validation={"overall": "PASS", "pass": True, "checks": {}},
        evidence={"status": "PASS", "evidence": []},
        recommendation={"status": "PASS", "recommendation": "Test recommendation"},
        insight_evidence_mapping={"status": "PASS"},
    )
    section_start = report.find("8. OVERALL RELIABILITY")
    section_end = report.find("9. VALIDATION AND SELF-CORRECTION PIPELINE")
    assert section_start >= 0 and section_end > section_start
    assert "FAIL" in report[section_start:section_end]


def test_valid_nonempty_numeric_result_still_passes():
    assert validate_result({"mean": 12.5, "status": "PASS"})["pass"] is True
