import ast
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from result_validator import (
    merge_step_validation,
    validate_single_step_analytical_result,
)


def _dataset(tmp_path):
    frame = pd.DataFrame({
        "amount": [10.0, 20.0, 30.0, 40.0],
        "units": [1, 2, 3, 4],
        "group": ["a", "a", "b", "b"],
    })
    path = tmp_path / "stage6_sample.csv"
    frame.to_csv(path, index=False)
    return path


def _mean_spec():
    return {
        "status": "PASS",
        "operations": ["mean", "max"],
        "metrics": ["amount"],
        "group_by": [],
        "filters": [],
        "statistics": [],
        "sorting": {"direction": "descending", "limit": None},
        "time": {},
    }


def _mean_result(value=25.0):
    return {
        "status": "PASS",
        "analysis_type": "specification_driven",
        "operations": ["mean"],
        "metrics": ["amount"],
        "group_by": [],
        "filters_applied": [],
        "input_rows": 4,
        "analyzed_rows": 4,
        "operation_results": {
            "aggregations": {"amount__mean": value}
        },
    }


def test_single_step_mean_independent_validation_passes(tmp_path):
    path = _dataset(tmp_path)
    result = validate_single_step_analytical_result(
        str(path), _mean_spec(), "Calculate the average amount", 1, _mean_result()
    )
    assert result["status"] == "PASS"
    assert result["validated_operations"] == ["mean"]


def test_single_step_mean_mismatch_fails(tmp_path):
    path = _dataset(tmp_path)
    result = validate_single_step_analytical_result(
        str(path), _mean_spec(), "Calculate the average amount", 1, _mean_result(26.0)
    )
    assert result["status"] == "FAIL"
    assert any("mismatch" in error.lower() for error in result["errors"])


def test_step_scoping_does_not_require_other_requested_operations(tmp_path):
    path = _dataset(tmp_path)
    result = validate_single_step_analytical_result(
        str(path), _mean_spec(), "Calculate the average amount", 1, _mean_result()
    )
    assert result["status"] == "PASS"
    assert result["validated_operations"] == ["mean"]


def test_non_independently_supported_operation_is_skipped(tmp_path):
    path = _dataset(tmp_path)
    spec = {
        "status": "PASS",
        "operations": ["missing_values"],
        "metrics": [],
        "group_by": [],
        "filters": [],
    }
    result = validate_single_step_analytical_result(
        str(path), spec, "Check missing values", 2,
        {"status": "PASS", "operations": ["missing_values"]},
    )
    assert result["status"] == "SKIPPED"


def test_filter_row_count_is_checked_independently(tmp_path):
    path = _dataset(tmp_path)
    spec = {
        "status": "PASS",
        "operations": ["mean"],
        "metrics": ["amount"],
        "group_by": [],
        "filters": [{"column": "amount", "operator": ">", "value": 10}],
    }
    result_payload = {
        "status": "PASS",
        "operations": ["mean"],
        "metrics": ["amount"],
        "group_by": [],
        "filters_applied": spec["filters"],
        "input_rows": 4,
        "analyzed_rows": 2,  # Should be 3.
        "operation_results": {"aggregations": {"amount__mean": 30.0}},
    }
    result = validate_single_step_analytical_result(
        str(path), spec, "Calculate the average amount", 1, result_payload
    )
    assert result["status"] == "FAIL"
    assert any("filtered row count" in error.lower() for error in result["errors"])


def test_merge_independent_fail_forces_retry_status():
    merged = merge_step_validation(
        {"overall": "PASS", "checks": {"existing": {"pass": True}}},
        {"status": "FAIL", "errors": ["Mean does not match."], "checks": []},
    )
    assert merged["overall"] == "FAIL"
    assert merged["checks"]["independent_analytical_result"]["pass"] is False
    assert "Mean does not match." in merged["checks"]["independent_analytical_result"]["reason"]


def test_merge_review_is_not_silently_marked_pass():
    merged = merge_step_validation(
        {"overall": "PASS", "checks": {}},
        {"status": "REVIEW", "errors": ["Metric could not be resolved."], "checks": []},
    )
    assert merged["overall"] == "REVIEW"


def test_skipped_independent_check_preserves_existing_self_validation():
    original = {"overall": "PASS", "checks": {"existing": {"pass": True}}}
    merged = merge_step_validation(
        original,
        {"status": "SKIPPED", "reason": "No supported independent operation."},
    )
    assert merged["overall"] == "PASS"
    assert "independent_analytical_result" not in merged["checks"]


def test_self_corrector_invokes_independent_validation_before_acceptance():
    source = (SRC_DIR / "self_correcting_step_executor.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "execute_self_correcting_step"
    )
    call_lines = {
        node.func.id: node.lineno
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "validate_single_step_analytical_result" in call_lines
    assert "merge_step_validation" in call_lines
    assert call_lines["validate_single_step_analytical_result"] < call_lines["merge_step_validation"]
