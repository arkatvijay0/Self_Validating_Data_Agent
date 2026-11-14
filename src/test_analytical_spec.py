"""Focused tests for analytical_spec.py; run with pytest -q."""
from analytical_spec import build_analytical_spec


PROFILE = {
    "columns": {
        "salary": {"role": "continuous"},
        "age": {"role": "continuous"},
        "department": {"role": "categorical"},
        "location": {"role": "categorical"},
        "hire_date": {"role": "time_index"},
    }
}


def test_multiple_operations_and_group_columns():
    spec = build_analytical_spec(
        "Calculate the average and maximum salary by department and location",
        PROFILE,
    )
    assert spec["operations"] == ["mean", "max"]
    assert spec["metrics"] == ["salary"]
    assert spec["group_by"] == ["department", "location"]
    assert spec["status"] == "READY"


def test_top_n_grouped_average():
    spec = build_analytical_spec(
        "Find the top 5 departments by average salary, descending",
        PROFILE,
    )
    assert "mean" in spec["operations"]
    assert "rank" in spec["operations"]
    assert spec["metrics"] == ["salary"]
    assert spec["group_by"] == ["department"]
    assert spec["sorting"] == {"direction": "descending", "limit": 5}


def test_numeric_filter_is_not_a_metric():
    spec = build_analytical_spec(
        "Calculate average salary for age greater than 30",
        PROFILE,
    )
    assert spec["metrics"] == ["salary"]
    assert spec["filters"] == [{
        "column": "age",
        "operator": ">",
        "value": 30.0,
        "value_type": "numeric",
        "source_text": "age greater than 30",
    }]


def test_time_trend():
    spec = build_analytical_spec("Show monthly salary trend over time", PROFILE)
    assert "time_trend" in spec["operations"]
    assert spec["time"]["column"] == "hire_date"
    assert spec["time"]["grain"] == "M"


def test_multiple_statistics():
    spec = build_analytical_spec(
        "Find the median salary and standard deviation",
        PROFILE,
    )
    assert spec["operations"] == ["median", "std"]
    assert spec["metrics"] == ["salary"]


def test_original_question_is_preserved():
    question = "Calculate average salary by department"
    spec = build_analytical_spec(question, PROFILE)
    assert spec["original_question"] == question
    assert spec["schema_version"] == "1.0"
    assert "requested_operations_match" in spec["validation_requirements"]
