import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from main import _is_valid_grouped_mean_result


def make_grouped_mean_result(no_mean=40.0, yes_mean=60.0, difference=20.0):
    return {
        "status": "PASS",
        "analysis_type": "grouped_mean_analysis",
        "metric_column": "balance",
        "group_column": "subscribed",
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
        "mean_difference_yes_minus_no": difference,
    }


def test_correct_grouped_mean_difference_passes():
    result = make_grouped_mean_result(
        no_mean=40.0,
        yes_mean=60.0,
        difference=20.0,
    )

    assert _is_valid_grouped_mean_result(result) is True


def test_incorrect_grouped_mean_difference_fails():
    result = make_grouped_mean_result(
        no_mean=40.0,
        yes_mean=60.0,
        difference=500.0,
    )

    assert _is_valid_grouped_mean_result(result) is False


def test_missing_grouped_mean_difference_fails():
    result = make_grouped_mean_result()
    result["mean_difference_yes_minus_no"] = None

    assert _is_valid_grouped_mean_result(result) is False


def test_non_finite_grouped_mean_difference_fails():
    result = make_grouped_mean_result()
    result["mean_difference_yes_minus_no"] = float("nan")

    assert _is_valid_grouped_mean_result(result) is False


def test_invalid_group_mean_fails():
    result = make_grouped_mean_result(
        no_mean=float("inf"),
        yes_mean=60.0,
        difference=20.0,
    )

    assert _is_valid_grouped_mean_result(result) is False


def test_small_floating_point_difference_is_accepted():
    result = make_grouped_mean_result(
        no_mean=40.0,
        yes_mean=60.0,
        difference=20.0000000001,
    )

    assert _is_valid_grouped_mean_result(result) is True