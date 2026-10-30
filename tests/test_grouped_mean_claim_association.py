import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from insight_validator import validate_grouped_mean_direction_claims

from main import (
    validate_multi_metric_grouped_mean_insight,
    build_multi_metric_evidence_mapping,
)


def grouped_result(metric, no_mean, yes_mean):
    return {
        "status": "PASS",
        "analysis_type": "grouped_mean_analysis",
        "metric_column": metric,
        "group_column": "y",
        "group_results": {
            "no": {
                "mean_value": no_mean,
                "mean": no_mean,
            },
            "yes": {
                "mean_value": yes_mean,
                "mean": yes_mean,
            },
        },
        "mean_difference_yes_minus_no": yes_mean - no_mean,
    }


def multi_metric_result():
    values = {
        "age": (30.0, 40.0),
        "balance": (100.0, 200.0),
        "campaign": (2.0, 3.0),
    }

    metric_results = {
        metric: grouped_result(metric, no_mean, yes_mean)
        for metric, (no_mean, yes_mean) in values.items()
    }

    return {
        "status": "PASS",
        "analysis_type": "multi_metric_grouped_mean_analysis",
        "requested_metric_columns": list(values),
        "metric_results": metric_results,
        "requested_metric_count": len(values),
        "validated_metric_count": len(values),
        "missing_metrics": [],
    }


def correct_multi_metric_insight():
    return (
        "Average comparison between customers who subscribed and those who did not: "
        "age: non-subscriber average = 30.00, subscriber average = 40.00, "
        "subscriber-minus-non-subscriber difference = +10.00 "
        "(subscriber average is higher). "
        "balance: non-subscriber average = 100.00, subscriber average = 200.00, "
        "subscriber-minus-non-subscriber difference = +100.00 "
        "(subscriber average is higher). "
        "campaign: non-subscriber average = 2.00, subscriber average = 3.00, "
        "subscriber-minus-non-subscriber difference = +1.00 "
        "(subscriber average is higher)."
    )


def test_single_metric_reversed_group_means_fail():
    result = grouped_result("balance", 100.0, 200.0)

    validation = validate_grouped_mean_direction_claims(
        insight=(
            "The balance no group mean was 200.00, while the balance yes "
            "group mean was 100.00."
        ),
        result=result,
    )

    assert validation["insight_pass"] is False
    assert validation["insight_overall"] == "FAIL"


def test_single_metric_correct_group_means_pass():
    result = grouped_result("balance", 100.0, 200.0)

    validation = validate_grouped_mean_direction_claims(
        insight=(
            "The average balance for non-subscribers was 100.00, "
            "while the average balance for subscribers was 200.00."
        ),
        result=result,
    )

    assert validation["insight_pass"] is True


def test_arbitrary_category_means_must_be_attached_to_correct_labels():
    result = {
        "status": "PASS",
        "analysis_type": "grouped_mean_analysis",
        "metric_column": "sales",
        "group_results": {
            "North": {"mean_value": 100.0},
            "South": {"mean_value": 200.0},
        },
    }

    validation = validate_grouped_mean_direction_claims(
        insight="Sales for North averaged 200.00; South averaged 100.00.",
        result=result,
    )

    assert validation["insight_pass"] is False


def test_multi_metric_correct_metric_scoped_means_pass():
    result = multi_metric_result()

    validation = validate_multi_metric_grouped_mean_insight(
        correct_multi_metric_insight(),
        result
    )

    assert validation["overall"] == "PASS"
    assert validation["pass"] is True

    assert all(
        item["pass"]
        for item in validation["checks"]["multi_metric_claim_validation"]
        ["metrics"].values()
    )


def test_multi_metric_values_from_another_metric_do_not_pass():
    result = multi_metric_result()

    incorrect = (
        "Average comparison between customers who subscribed and those who did not: "
        "age: non-subscriber average = 100.00, subscriber average = 200.00, "
        "subscriber-minus-non-subscriber difference = +100.00 "
        "(subscriber average is higher). "
        "balance: non-subscriber average = 100.00, subscriber average = 200.00, "
        "subscriber-minus-non-subscriber difference = +100.00 "
        "(subscriber average is higher). "
        "campaign: non-subscriber average = 2.00, subscriber average = 3.00, "
        "subscriber-minus-non-subscriber difference = +1.00 "
        "(subscriber average is higher)."
    )

    validation = validate_multi_metric_grouped_mean_insight(
        incorrect,
        result
    )

    assert validation["overall"] == "FAIL"

    assert (
        validation["checks"]["multi_metric_claim_validation"]
        ["metrics"]["age"]["pass"] is False
    )


def test_multi_metric_swapped_group_values_fail():
    result = multi_metric_result()

    incorrect = correct_multi_metric_insight().replace(
        "age: non-subscriber average = 30.00, subscriber average = 40.00",
        "age: non-subscriber average = 40.00, subscriber average = 30.00",
    )

    validation = validate_multi_metric_grouped_mean_insight(
        incorrect,
        result
    )

    assert validation["overall"] == "FAIL"

    age_check = (
        validation["checks"]["multi_metric_claim_validation"]
        ["metrics"]["age"]
    )

    assert age_check["non_subscriber_mean_present"] is False
    assert age_check["subscriber_mean_present"] is False


def test_multi_metric_evidence_mapping_rejects_wrong_metric_values():
    result = multi_metric_result()

    evidence = {
        "status": "PASS",
        "analysis_type": result["analysis_type"],
        "requested_metric_columns": result["requested_metric_columns"],
        "metric_results": result["metric_results"],
    }

    incorrect = correct_multi_metric_insight().replace(
        "age: non-subscriber average = 30.00, subscriber average = 40.00",
        "age: non-subscriber average = 100.00, subscriber average = 200.00",
    )

    base_validation = validate_multi_metric_grouped_mean_insight(
        incorrect,
        result
    )

    mapping = build_multi_metric_evidence_mapping(
        incorrect,
        evidence,
        base_validation
    )

    assert mapping["status"] == "FAIL"


def test_grouped_mean_validator_scopes_multi_metric_claims():
    result = multi_metric_result()

    incorrect = correct_multi_metric_insight().replace(
        "age: non-subscriber average = 30.00, subscriber average = 40.00",
        "age: non-subscriber average = 100.00, subscriber average = 200.00",
    )

    validation = validate_grouped_mean_direction_claims(
        insight=incorrect,
        result=result,
    )

    assert validation["insight_pass"] is False
    assert validation["insight_overall"] == "FAIL"