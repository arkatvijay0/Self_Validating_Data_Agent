import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from eda_planner import generate_plan_mock


def make_profile():
    return {
        "columns": {
            "balance": {"role": "continuous"},
            "duration": {"role": "continuous"},
            "job": {"role": "categorical"},
            "housing": {"role": "categorical"},
            "age": {"role": "continuous"},
            "month": {"role": "categorical"},
            "date": {"role": "time_index"},
        }
    }


def test_planner_preserves_multiple_metrics_and_group_columns():
    spec = {
        "status": "READY",
        "operations": ["mean", "max"],
        "metrics": ["balance", "duration"],
        "group_by": ["job", "housing"],
        "filters": [],
        "comparisons": [{"type": "group_comparison"}],
        "statistics": [],
        "time": {"column": None, "grain": None},
        "sorting": {"direction": None, "limit": None},
        "requested_outputs": ["table"],
        "mentioned_columns": ["balance", "duration", "job", "housing"],
    }
    plan = generate_plan_mock("comparison", make_profile(), "compare metrics", analytical_spec=spec)
    joined = " ".join(plan)
    for value in ("balance", "duration", "job", "housing", "mean", "maximum"):
        assert value in joined


def test_planner_preserves_explicit_numeric_filter():
    spec = {
        "status": "READY",
        "operations": ["mean"],
        "metrics": ["balance"],
        "group_by": [],
        "filters": [{"column": "age", "operator": ">", "value": 30.0}],
        "comparisons": [],
        "statistics": [],
        "time": {"column": None, "grain": None},
        "sorting": {"direction": None, "limit": None},
        "requested_outputs": ["result"],
    }
    plan = generate_plan_mock("mean_analysis", make_profile(), "average balance where age > 30", analytical_spec=spec)
    assert any("age` > 30" in step for step in plan)
    assert any("balance" in step and "mean" in step for step in plan)


def test_planner_does_not_guess_missing_metric():
    spec = {
        "status": "READY",
        "operations": ["mean"],
        "metrics": [],
        "group_by": [],
        "filters": [],
        "comparisons": [],
        "statistics": [],
        "time": {"column": None, "grain": None},
        "sorting": {"direction": None, "limit": None},
        "requested_outputs": ["result"],
    }
    plan = generate_plan_mock("mean_analysis", make_profile(), "average something", analytical_spec=spec)
    assert any(step.startswith("STOP:") for step in plan)
    assert not any("balance" in step for step in plan)


def test_planner_preserves_correlation_pair_and_method():
    spec = {
        "status": "READY",
        "operations": ["correlation"],
        "metrics": ["balance", "duration"],
        "group_by": [],
        "filters": [],
        "comparisons": [],
        "statistics": ["spearman"],
        "time": {"column": None, "grain": None},
        "sorting": {"direction": None, "limit": None},
        "requested_outputs": ["result"],
    }
    plan = generate_plan_mock("relationship_analysis", make_profile(), "correlation", analytical_spec=spec)
    assert any("spearman" in step and "balance" in step and "duration" in step for step in plan)


def test_legacy_call_without_spec_remains_compatible():
    plan = generate_plan_mock("mean_analysis", make_profile(), "average balance")
    assert isinstance(plan, list)
    assert plan
    assert any("balance" in step for step in plan)
