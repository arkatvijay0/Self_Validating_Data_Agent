"""Integration tests for task_analyzer.py and analytical_spec.py."""
from task_analyzer import analyze_task_mock


PROFILE = {
    "columns": {
        "salary": {"role": "continuous"},
        "age": {"role": "continuous"},
        "department": {"role": "categorical"},
        "location": {"role": "categorical"},
        "hire_date": {"role": "time_index"},
        "balance": {"role": "continuous"},
        "job": {"role": "categorical"},
        "housing": {"role": "categorical"},
        "duration": {"role": "continuous"},
        "month": {"role": "categorical"},
        "y": {"role": "categorical"},
    }
}


def test_legacy_fields_are_preserved_for_mean():
    result = analyze_task_mock("What is the average salary?", PROFILE)
    assert result["task_type"] == "mean_analysis"
    assert result["metric_column"] == "salary"
    assert result["target_columns"] == ["salary"]
    assert result["analytical_spec"]["metrics"] == ["salary"]
    assert result["analytical_spec"]["operations"] == ["mean"]


def test_legacy_fields_are_preserved_for_grouped_mean():
    result = analyze_task_mock(
        "Compare average salary by department and location", PROFILE
    )
    assert result["task_type"] == "comparison"
    spec = result["analytical_spec"]
    assert spec["metrics"] == ["salary"]
    assert spec["group_by"] == ["department", "location"]


def test_legacy_filtered_month_fields_are_preserved():
    result = analyze_task_mock(
        "What is the average call duration in August?", PROFILE
    )
    assert result["task_type"] == "mean_analysis"
    assert result["filter_column"] == "month"
    assert result["filter_month"] == "aug"
    assert result["analytical_spec"]["metrics"] == ["duration"]


def test_existing_ambiguous_question_flag_is_preserved():
    result = analyze_task_mock(
        "Do subscribers have a higher lower balance than non-subscribers?",
        PROFILE,
    )
    assert result["ambiguous_direction_claim"] is True
    assert result["analytical_spec"]["original_question"].startswith("Do subscribers")


def test_every_result_contains_specification():
    result = analyze_task_mock("Give me a descriptive summary", PROFILE)
    assert isinstance(result["analytical_spec"], dict)
    assert result["analytical_spec"]["schema_version"] == "1.0"
