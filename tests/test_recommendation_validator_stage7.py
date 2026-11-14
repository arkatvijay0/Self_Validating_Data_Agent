import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recommendation_validator import validate_recommendation


def test_recommendation_with_evidence_numbers_passes():
    result = validate_recommendation(
        recommendation="The validated average balance is 1,250.50.",
        evidence={"status": "PASS", "analysis_type": "mean_analysis",
                  "mean": 1250.50, "count": 12},
        analysis_type="mean_analysis",
    )
    assert result["status"] == "PASS"
    assert result["verified_numeric_claims"] == 1


def test_unmatched_numeric_claim_is_review():
    result = validate_recommendation(
        recommendation="The validated average is 99,999.",
        evidence={"status": "PASS", "mean": 1250.50},
    )
    assert result["status"] == "REVIEW"
    assert result["unverified_numeric_claims"] == 1


def test_failed_evidence_fails_recommendation():
    result = validate_recommendation(
        recommendation="Investigate further.",
        evidence={"status": "FAIL"},
    )
    assert result["status"] == "FAIL"


def test_empty_recommendation_fails():
    result = validate_recommendation(
        recommendation=" ",
        evidence={"status": "PASS", "mean": 2},
    )
    assert result["status"] == "FAIL"


def test_structured_basis_contradiction_fails():
    result = validate_recommendation(
        recommendation={
            "status": "PASS",
            "recommendation": "Use metric X with value 10.",
            "basis": {"metric": "Wrong Metric", "value": 10},
        },
        evidence={"status": "PASS", "metric": "Correct Metric", "value": 10},
    )
    assert result["status"] == "FAIL"


def test_derived_group_difference_can_be_grounded():
    result = validate_recommendation(
        recommendation="The leading group is ahead by 20.",
        evidence={"status": "PASS", "supporting_result": {"A": 100, "B": 80}},
    )
    assert result["status"] == "PASS"


def test_non_numeric_generic_recommendation_passes():
    result = validate_recommendation(
        recommendation="Investigate further before making business decisions.",
        evidence={"status": "PASS", "analysis_type": "generic"},
    )
    assert result["status"] == "PASS"


def test_final_report_marks_unverified_recommendation_review():
    from final_report import create_final_report

    report = create_final_report(
        user_question="What is the average balance?",
        analysis_type="mean_analysis",
        validated_result={"status": "PASS", "mean": 100.0},
        insight="The average balance is 100.",
        insight_validation={"overall": "PASS", "pass": True, "checks": {
            "presence": {"pass": True}, "result": {"pass": True}
        }},
        evidence={"status": "PASS", "mean": 100.0},
        recommendation="The average balance is 999999.",
        insight_evidence_mapping={"status": "PASS", "traceability_status": "PASS"},
        recommendation_validation={"status": "REVIEW", "overall": "REVIEW",
                                   "pass": False, "checks": []},
    )
    assert "Recommendation Status: REVIEW" in report
    assert "OVERALL RELIABILITY" in report
    assert "\nREVIEW\n" in report
