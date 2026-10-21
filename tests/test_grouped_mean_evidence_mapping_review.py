
import json
import sys
from pathlib import Path

# Import the separate review module, not the original mapper.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

import evidence_mapping_grouped_mean_fixed as mapper


NO_MEAN = 1303.7149691899203
YES_MEAN = 1804.2679145396105
DIFFERENCE = 500.55294534969016

INSIGHT = (
    "Average customer balance was 1,303.71 for non-subscribers "
    "and 1,804.27 for subscribers. The subscriber group average "
    "was 500.55 higher (yes minus no). The comparison is "
    "descriptive and does not establish causation."
)


def build_full_pipeline_evidence():
    """Reproduce the important evidence structure from the pipeline log."""
    primary_result = {
        "status": "PASS",
        "analysis_type": "grouped_mean_analysis",
        "metric_column": "balance",
        "group_column": "y",
        "group_results": {
            "no": {
                "row_count": 39922,
                "valid_value_count": 39922,
                "mean_balance": NO_MEAN,
            },
            "yes": {
                "row_count": 5289,
                "valid_value_count": 5289,
                "mean_balance": YES_MEAN,
            },
        },
        "mean_difference_yes_minus_no": DIFFERENCE,
        "dataset_row_count": 45211,
        "analyzed_row_count": 45211,
    }

    # Mirror the nested evidence structure printed by src/main.py.
    return {
        "status": "PASS",
        "analysis_type": "grouped_mean_analysis",
        "metric_column": "balance",
        "group_column": "y",
        "group_results": primary_result["group_results"],
        "mean_difference_yes_minus_no": DIFFERENCE,
        "supporting_result": primary_result,
        "result": primary_result,
        "evidence_summary": json.dumps(primary_result),
        "evidence_sources": [
            {
                "source_id": "primary_result.group_results",
                "source_type": "validated_grouped_mean_result",
                "group_column": "y",
                "metric_column": "balance",
                "aggregation": "mean",
                "result": primary_result,
            }
        ],
        "reason": None,
    }


def test_full_pipeline_evidence_accepts_correct_insight():
    evidence = build_full_pipeline_evidence()

    result = mapper.map_insight_to_evidence(INSIGHT, evidence)

    print("\nCORRECT INSIGHT RESULT:")
    print(json.dumps(result, indent=2))

    assert result["status"] == "PASS", (
        "Correct grouped-mean insight should pass. "
        f"Returned status: {result['status']}; "
        f"reason: {result.get('reason')}"
    )

    assert result["analysis_type"] == "grouped_mean_analysis"

    checks = {item["check"]: item["status"] for item in result["checks"]}
    assert checks["both_group_means_available"] == "PASS"
    assert checks["non_subscriber_mean_mentioned"] == "PASS"
    assert checks["subscriber_mean_mentioned"] == "PASS"
    assert checks["mean_difference_mentioned"] == "PASS"

    assert abs(result["group_results"]["no"]["mean_balance"] - NO_MEAN) < 0.01
    assert abs(result["group_results"]["yes"]["mean_balance"] - YES_MEAN) < 0.01
    assert abs(result["mean_difference_yes_minus_no"] - DIFFERENCE) < 0.01


def test_full_pipeline_evidence_rejects_wrong_mean():
    evidence = build_full_pipeline_evidence()
    incorrect_insight = INSIGHT.replace("1,303.71", "999.99")

    result = mapper.map_insight_to_evidence(incorrect_insight, evidence)

    print("\nINCORRECT INSIGHT RESULT:")
    print(json.dumps(result, indent=2))

    assert result["status"] == "FAIL", (
        "Incorrect non-subscriber mean should be rejected."
    )

    checks = {item["check"]: item["status"] for item in result["checks"]}
    assert checks["non_subscriber_mean_mentioned"] == "FAIL"


if __name__ == "__main__":
    test_full_pipeline_evidence_accepts_correct_insight()
    test_full_pipeline_evidence_rejects_wrong_mean()
    print("\nPASS: Both full-structure grouped-mean review tests passed.")
