"""Execution tests for specification-driven Stage 2 code generation."""
import importlib.util
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = PROJECT_ROOT / "src" / "code_generator.py"


def _load_generator():
    spec = importlib.util.spec_from_file_location("stage2_code_generator", GENERATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(tmp_path, analytical_spec, step_text):
    generator = _load_generator()
    csv_path = tmp_path / "sample.csv"
    pd.DataFrame({
        "salary": [10, 20, 30, 40, 100],
        "age": [20, 30, 40, 50, 60],
        "department": ["A", "A", "B", "B", "B"],
        "date": ["2026-01-01", "2026-01-02", "2026-02-01", "2026-02-02", "2026-03-01"],
    }).to_csv(csv_path, index=False)
    generated = generator.generate_code_mock(
        step_text=step_text,
        filepath=str(csv_path),
        analytical_spec=analytical_spec,
    )
    compile(generated, "<generated_eda_code>", "exec")
    namespace = {}
    exec(generated, namespace)
    return namespace["result"]


def _base_spec(**overrides):
    spec = {
        "status": "READY",
        "operations": ["mean"],
        "metrics": ["salary"],
        "group_by": [],
        "filters": [],
        "statistics": [],
        "time": {"column": None, "grain": None},
        "sorting": {"direction": None, "limit": None},
    }
    spec.update(overrides)
    return spec


def test_spec_driven_overall_mean(tmp_path):
    result = _run(tmp_path, _base_spec(), "Calculate average salary")
    assert result["status"] == "PASS"
    assert result["operation_results"]["aggregations"]["salary__mean"] == 40.0


def test_spec_driven_grouped_multiple_operations(tmp_path):
    spec = _base_spec(
        operations=["mean", "max"],
        group_by=["department"],
    )
    result = _run(tmp_path, spec, "Calculate average and maximum salary by department")
    assert result["status"] == "PASS"
    rows = result["operation_results"]["aggregations"]
    assert rows[0]["salary__mean"] == 15.0
    assert rows[0]["salary__max"] == 20.0
    assert rows[1]["salary__mean"] == 56.666666666666664
    assert rows[1]["salary__max"] == 100.0


def test_spec_driven_numeric_filter(tmp_path):
    spec = _base_spec(filters=[{
        "column": "age", "operator": ">", "value": 30.0,
        "value_type": "numeric", "source_text": "age greater than 30",
    }])
    result = _run(tmp_path, spec, "Calculate average salary")
    assert result["status"] == "PASS"
    assert result["analyzed_rows"] == 3
    assert result["operation_results"]["aggregations"]["salary__mean"] == 56.666666666666664


def test_spec_driven_correlation(tmp_path):
    spec = _base_spec(
        operations=["correlation"], metrics=["salary", "age"],
        statistics=["pearson"],
    )
    result = _run(tmp_path, spec, "Calculate correlation between salary and age")
    assert result["status"] == "PASS"
    assert result["operation_results"]["correlation"]["pearson"]["salary"]["age"] > 0.89


def test_spec_driven_outlier_detection(tmp_path):
    spec = _base_spec(operations=["outlier_detection"])
    result = _run(tmp_path, spec, "Detect outliers in salary")
    assert result["status"] == "PASS"
    assert result["operation_results"]["outlier_detection"]["salary"]["outlier_count"] == 1


def test_missing_metric_is_rejected(tmp_path):
    spec = _base_spec(metrics=["not_a_real_column"])
    result = _run(tmp_path, spec, "Calculate average not_a_real_column")
    assert result["status"] == "UNSUPPORTED"
    assert "not_a_real_column" in result["reason"]


def test_review_spec_does_not_generate_analysis(tmp_path):
    spec = _base_spec(status="REVIEW", issues=["Metric is ambiguous."])
    result = _run(tmp_path, spec, "Calculate average salary")
    assert result["status"] == "REVIEW"
    assert "ambiguous" in result["reason"]


def test_legacy_generation_remains_available(tmp_path):
    generator = _load_generator()
    csv_path = tmp_path / "legacy.csv"
    pd.DataFrame({"salary": [10, 20]}).to_csv(csv_path, index=False)
    code = generator.generate_code_mock("Calculate the average salary", str(csv_path))
    compile(code, "<legacy_generated_code>", "exec")
    namespace = {}
    exec(code, namespace)
    assert namespace["result"]["status"] in {"PASS", "FAIL", "UNSUPPORTED"}


def test_spec_driven_row_count_without_metric(tmp_path):
    spec = _base_spec(operations=["count"], metrics=[])
    result = _run(tmp_path, spec, "Count source rows")
    assert result["status"] == "PASS"
    assert result["operation_results"]["aggregations"]["row_count"] == 5


def test_planner_filter_step_and_recompute_step_are_supported(tmp_path):
    generator = _load_generator()
    csv_path = tmp_path / "sample.csv"
    pd.DataFrame({"salary": [10, 20, 30], "age": [20, 40, 60]}).to_csv(csv_path, index=False)
    spec = _base_spec(
        operations=["mean"],
        filters=[{"column": "age", "operator": ">", "value": 25.0}],
    )
    filter_code = generator.generate_code_mock(
        "Apply all explicitly requested filters before calculating results: age > 25.",
        str(csv_path), analytical_spec=spec,
    )
    filter_ns = {}
    exec(filter_code, filter_ns)
    assert filter_ns["result"]["status"] == "PASS"
    assert filter_ns["result"]["operation_results"]["filter_summary"]["filtered_rows"] == 2

    validation_code = generator.generate_code_mock(
        "Independently recompute the requested operations using the same source rows.",
        str(csv_path), analytical_spec=spec,
    )
    validation_ns = {}
    exec(validation_code, validation_ns)
    assert validation_ns["result"]["status"] == "PASS"
    assert validation_ns["result"]["operation_results"]["aggregations"]["salary__mean"] == 25.0


def test_real_analytical_spec_and_planner_steps_drive_generation(tmp_path):
    from analytical_spec import build_analytical_spec
    from eda_planner import generate_plan_mock

    generator = _load_generator()
    csv_path = tmp_path / "sample.csv"
    pd.DataFrame({
        "salary": [10, 20, 30, 40],
        "department": ["A", "A", "B", "B"],
    }).to_csv(csv_path, index=False)
    profile = {
        "columns": {
            "salary": {"role": "continuous"},
            "department": {"role": "categorical"},
        }
    }
    question = "Calculate the average and maximum salary by department"
    analytical_spec = build_analytical_spec(question, profile)
    plan = generate_plan_mock(
        task_type="statistical_analysis",
        data_profile=profile,
        user_query=question,
        analytical_spec=analytical_spec,
    )
    assert analytical_spec["status"] == "READY"
    assert any("arithmetic mean" in step for step in plan)
    assert any("maximum" in step for step in plan)

    results = []
    for step in plan:
        generated = generator.generate_code_mock(
            step, str(csv_path), data_profile=profile,
            analytical_spec=analytical_spec,
        )
        namespace = {}
        exec(generated, namespace)
        results.append(namespace["result"])

    operation_results = [
        result["operation_results"]
        for result in results
        if result.get("status") == "PASS"
    ]
    mean_rows = [
        row
        for item in operation_results
        for row in item.get("aggregations", [])
        if isinstance(item.get("aggregations"), list)
    ]
    max_rows = mean_rows
    assert any(row.get("salary__mean") == 15.0 for row in mean_rows)
    assert any(row.get("salary__max") == 40.0 for row in max_rows)
