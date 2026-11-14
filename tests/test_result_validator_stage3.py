import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from result_validator import validate_analytical_results


def _spec():
    return {
        "operations": ["mean", "sum"],
        "metrics": ["amount"],
        "group_by": [],
        "filters": [],
    }


def _execution(result):
    return {"steps": [{"step_number": 1, "step": "Calculate the average and sum of amount", "final_result": result}]}


def test_valid_scalar_aggregations_pass(tmp_path):
    path = tmp_path / "data.csv"
    pd.DataFrame({"amount": [2, 4, 6]}).to_csv(path, index=False)
    result = {
        "status": "PASS", "operations": ["mean", "sum"],
        "input_rows": 3, "analyzed_rows": 3, "filters_applied": [],
        "operation_results": {"aggregations": {"amount__mean": 4.0, "amount__sum": 12.0}},
    }
    report = validate_analytical_results(str(path), _spec(), _execution(result))
    assert report["status"] == "PASS", report


def test_incorrect_aggregation_fails(tmp_path):
    path = tmp_path / "data.csv"
    pd.DataFrame({"amount": [2, 4, 6]}).to_csv(path, index=False)
    result = {
        "status": "PASS", "operations": ["mean", "sum"],
        "input_rows": 3, "analyzed_rows": 3, "filters_applied": [],
        "operation_results": {"aggregations": {"amount__mean": 99.0, "amount__sum": 12.0}},
    }
    report = validate_analytical_results(str(path), _spec(), _execution(result))
    assert report["status"] == "FAIL"
    assert any("mismatch" in error.lower() for error in report["errors"])


def test_wrong_row_count_fails(tmp_path):
    path = tmp_path / "data.csv"
    pd.DataFrame({"amount": [2, 4, 6]}).to_csv(path, index=False)
    result = {
        "status": "PASS", "operations": ["mean", "sum"],
        "input_rows": 2, "analyzed_rows": 3, "filters_applied": [],
        "operation_results": {"aggregations": {"amount__mean": 4.0, "amount__sum": 12.0}},
    }
    report = validate_analytical_results(str(path), _spec(), _execution(result))
    assert report["status"] == "FAIL"


def test_missing_requested_operation_fails(tmp_path):
    path = tmp_path / "data.csv"
    pd.DataFrame({"amount": [2, 4, 6]}).to_csv(path, index=False)
    result = {
        "status": "PASS", "operations": ["mean"],
        "input_rows": 3, "analyzed_rows": 3, "filters_applied": [],
        "operation_results": {"aggregations": {"amount__mean": 4.0}},
    }
    report = validate_analytical_results(str(path), _spec(), _execution(result))
    assert report["status"] == "FAIL"
