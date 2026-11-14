"""Focused Stage 8 regression tests for specification parsing and planning."""
import importlib.util
from pathlib import Path

BASE = Path(__file__).parent


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, BASE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

analytical = load_module("analytical_spec_stage8", "analytical_spec_stage8.py")
planner = load_module("eda_planner_stage8", "eda_planner_stage8.py")
generator = load_module("code_generator_stage8", "code_generator_stage8.py")

PROFILE = {
    "columns": {
        "age": {"role": "continuous"},
        "job": {"role": "categorical"},
        "balance": {"role": "continuous"},
        "contact": {"role": "categorical"},
        "month": {"role": "categorical"},
        "campaign": {"role": "continuous"},
        "conversion": {"role": "continuous"},
    }
}


def build(question):
    return analytical.build_analytical_spec(question, PROFILE)


def test_average_account_balance_is_scalar_mean():
    spec = build("Calculate the average account balance.")
    assert spec["operations"] == ["mean"]
    assert spec["metrics"] == ["balance"]
    assert spec["group_by"] == []


def test_highest_monthly_conversions_is_grouped_sum_and_rank():
    spec = build("Find the month with the highest conversions.")
    assert spec["operations"] == ["sum", "rank"]
    assert spec["metrics"] == ["conversion"]
    assert spec["group_by"] == ["month"]
    assert spec["sorting"]["direction"] == "descending"
    plan = planner._build_plan_from_analytical_spec(spec)
    assert plan
    rank_steps = [step for step in plan if "rank" in step.casefold()]
    assert rank_steps
    assert any("sum" in step.casefold() and "month" in step.casefold() for step in rank_steps)
    assert not any("prepare the requested output" in step.casefold() for step in plan)
    rank_spec = generator._select_spec_for_step(rank_steps[0], spec)
    assert rank_spec["operations"] == ["sum", "rank"]


def test_count_customers_in_each_job_category_groups_by_job():
    spec = build("Count the number of customers in each job category.")
    assert spec["operations"] == ["count"]
    assert spec["metrics"] == []
    assert spec["group_by"] == ["job"]
    plan = planner._build_plan_from_analytical_spec(spec)
    assert any("count source rows" in step.casefold() and "job" in step.casefold() for step in plan)
    assert not any("prepare the requested output" in step.casefold() for step in plan)


def test_average_campaign_contacts_means_campaign_without_contact_grouping():
    spec = build("Calculate the average number of campaign contacts per customer.")
    assert spec["operations"] == ["mean"]
    assert spec["metrics"] == ["campaign"]
    assert spec["group_by"] == []
    assert "contact" not in spec["mentioned_columns"]


def test_output_metadata_is_not_an_executable_step():
    spec = build("Calculate the average account balance.")
    plan = planner._build_plan_from_analytical_spec(spec)
    assert all("output form" not in step.casefold() for step in plan)
    assert all("requested output" not in step.casefold() for step in plan)



def _execute_generated(step_text, spec, frame):
    import tempfile
    import pandas as pd
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as handle:
        csv_path = handle.name
    try:
        frame.to_csv(csv_path, index=False)
        code = generator.generate_code_mock(
            step_text=step_text,
            filepath=csv_path,
            analytical_spec=spec,
        )
        namespace = {}
        exec(code, namespace)
        return namespace.get("result")
    finally:
        Path(csv_path).unlink(missing_ok=True)


def test_generated_grouped_conversion_rank_executes():
    import pandas as pd
    spec = build("Find the month with the highest conversions.")
    plan = planner._build_plan_from_analytical_spec(spec)
    rank_step = next(step for step in plan if "rank" in step.casefold())
    frame = pd.DataFrame({
        "month": ["jan", "jan", "feb", "feb", "feb", "mar"],
        "conversion": [1, 0, 1, 1, 0, 1],
    })
    result = _execute_generated(rank_step, spec, frame)
    assert result["status"] == "PASS", result
    ranking = result["operation_results"]["ranking"]
    assert ranking[0]["month"] == "feb"
    assert ranking[0]["conversion__sum"] == 2.0


def test_generated_grouped_customer_count_executes():
    import pandas as pd
    spec = build("Count the number of customers in each job category.")
    plan = planner._build_plan_from_analytical_spec(spec)
    count_step = next(step for step in plan if "count source rows" in step.casefold())
    frame = pd.DataFrame({"job": ["admin", "admin", "technician", "student", "student", "student"]})
    result = _execute_generated(count_step, spec, frame)
    assert result["status"] == "PASS", result
    counts = {row["job"]: row["row_count"] for row in result["operation_results"]["aggregations"]}
    assert counts == {"admin": 2, "technician": 1, "student": 3}


def test_generated_campaign_mean_executes_without_grouping():
    import pandas as pd
    spec = build("Calculate the average number of campaign contacts per customer.")
    plan = planner._build_plan_from_analytical_spec(spec)
    mean_step = next(step for step in plan if "calculate the arithmetic mean" in step.casefold())
    frame = pd.DataFrame({"campaign": [1, 2, 3, 4], "contact": ["cellular", "telephone", "cellular", "telephone"]})
    result = _execute_generated(mean_step, spec, frame)
    assert result["status"] == "PASS", result
    assert result["group_by"] == []
    assert result["operation_results"]["aggregations"]["campaign__mean"] == 2.5

if __name__ == "__main__":
    tests = [
        test_average_account_balance_is_scalar_mean,
        test_highest_monthly_conversions_is_grouped_sum_and_rank,
        test_count_customers_in_each_job_category_groups_by_job,
        test_average_campaign_contacts_means_campaign_without_contact_grouping,
        test_output_metadata_is_not_an_executable_step,
        test_generated_grouped_conversion_rank_executes,
        test_generated_grouped_customer_count_executes,
        test_generated_campaign_mean_executes_without_grouping,
    ]
    failed = []
    for test in tests:
        try:
            test()
            print(f"PASS: {test.__name__}")
        except Exception as exc:
            failed.append((test.__name__, exc))
            print(f"FAIL: {test.__name__}: {exc}")
    print(f"\nStage 8 focused tests: {len(tests)-len(failed)}/{len(tests)} passed")
    if failed:
        raise SystemExit(1)
