import ast
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from claim_evidence_validator import (
    extract_numeric_claims,
    validate_text_against_evidence,
    validate_insight_and_recommendation,
)


def test_extracts_integer_decimal_and_percentage_tokens():
    claims = extract_numeric_claims("Average is 1250.50; increase is 12%.")
    assert [item["value"] for item in claims] == [1250.50, 12.0]
    assert claims[1]["is_percentage"] is True


def test_all_numeric_claims_match_evidence():
    result = validate_text_against_evidence(
        "The mean is 1250.5 across 100 rows.",
        {"mean": 1250.5, "row_count": 100},
    )
    assert result["status"] == "PASS"
    assert result["matched_claim_count"] == 2


def test_unmatched_numeric_claim_returns_review_not_pass():
    result = validate_text_against_evidence(
        "The mean is 999.",
        {"mean": 1250.5},
    )
    assert result["status"] == "REVIEW"
    assert result["unmatched_claim_count"] == 1


def test_text_without_numbers_is_not_applicable():
    result = validate_text_against_evidence(
        "The management group has the highest average.",
        {"management": {"mean": 120.0}},
    )
    # claim_evidence_validator documents NOT_APPLICABLE for text without numbers
    # (semantic claims need separate validation); it must not downgrade others.
    assert result["status"] == "NOT_APPLICABLE"
    assert result["numeric_claim_count"] == 0


def test_nested_evidence_and_numpy_like_scalars_are_supported():
    class Scalar:
        def item(self):
            return 42

    result = validate_text_against_evidence(
        "The count is 42.",
        {"details": [{"count": Scalar()}]},
    )
    assert result["status"] == "PASS"


def test_booleans_are_not_treated_as_numeric_evidence():
    result = validate_text_against_evidence(
        "The value is 1.",
        {"passed": True},
    )
    assert result["status"] == "REVIEW"


def test_insight_and_recommendation_are_validated_separately():
    result = validate_insight_and_recommendation(
        "Average balance is 250.",
        "Review the segment with 250 accounts.",
        {"average": 250, "accounts": 250},
    )
    assert result["insight"]["status"] == "PASS"
    assert result["recommendation"]["status"] == "PASS"
    assert result["overall"] == "PASS"
