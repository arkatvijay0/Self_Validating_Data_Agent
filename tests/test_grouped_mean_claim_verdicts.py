
import pandas as pd

from src.insight_validator import validate_insight


def grouped_balance_result():
    """Create deterministic grouped means for claim-validation tests."""
    return {
        "status": "PASS",
        "analysis_type": "grouped_mean_analysis",
        "metric_column": "balance",
        "group_column": "y",
        "group_results": {
            "no": {
                "row_count": 4,
                "valid_value_count": 4,
                "mean_value": 100.0,
                "mean": 100.0,
            },
            "yes": {
                "row_count": 4,
                "valid_value_count": 4,
                "mean_value": 200.0,
                "mean": 200.0,
            },
        },
        "mean_difference_yes_minus_no": 100.0,
        "dataset_row_count": 8,
        "analyzed_row_count": 8,
    }


def test_higher_average_claim_is_supported():
    result = grouped_balance_result()
    insight = (
        "Subscribers have a higher average balance than non-subscribers. "
        "The subscriber mean is 200 and the non-subscriber mean is 100."
    )

    validation = validate_insight(
        insight=insight,
        result=result,
        user_question=(
            "Validate this claim: subscribers have a higher average "
            "balance than non-subscribers."
        ),
    )

    assert validation["overall"] == "PASS"
    assert (
        validation["checks"]["grouped_mean_claim_validation"]
        ["user_claim_validation"]["status"]
        == "SUPPORTED"
    )


def test_lower_average_claim_is_not_supported_but_insight_is_accurate():
    result = grouped_balance_result()
    insight = (
        "Subscribers have a higher average balance than non-subscribers. "
        "The subscriber mean is 200 and the non-subscriber mean is 100."
    )

    validation = validate_insight(
        insight=insight,
        result=result,
        user_question=(
            "Validate this claim: subscribers have a lower average "
            "balance than non-subscribers."
        ),
    )

    assert validation["overall"] == "PASS"
    assert (
        validation["checks"]["grouped_mean_claim_validation"]
        ["user_claim_validation"]["status"]
        == "NOT_SUPPORTED"
    )
