"""Regression tests for independent validation of advanced EDA operations."""
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from result_validator import validate_analytical_results


def _run(path, spec, step, result):
    execution = {"steps": [{"step_number": 1, "step": step, "final_result": result}]}
    return validate_analytical_results(str(path), spec, execution)


def _base_result(ops, rows, outputs):
    return {
        "status": "PASS", "operations": ops,
        "input_rows": rows, "analyzed_rows": rows,
        "filters_applied": [], "operation_results": outputs,
    }


def test_correlation_is_independently_validated(tmp_path):
    path = tmp_path / "corr.csv"
    df = pd.DataFrame({"x": [1, 2, 3, 4], "y": [2, 4, 6, 8]})
    df.to_csv(path, index=False)
    spec = {"operations": ["correlation"], "metrics": ["x", "y"], "group_by": [], "filters": [], "statistics": ["pearson"]}
    corr = df[["x", "y"]].corr(method="pearson").to_dict()
    report = _run(path, spec, "Calculate correlation between x and y", _base_result(["correlation"], 4, {"correlation": {"pearson": corr}}))
    assert report["status"] == "PASS", report


def test_incorrect_correlation_fails(tmp_path):
    path = tmp_path / "corr_bad.csv"
    pd.DataFrame({"x": [1, 2, 3], "y": [2, 4, 6]}).to_csv(path, index=False)
    spec = {"operations": ["correlation"], "metrics": ["x", "y"], "group_by": [], "filters": [], "statistics": ["pearson"]}
    bad = {"pearson": {"x": {"x": 1.0, "y": 0.0}, "y": {"x": 0.0, "y": 1.0}}}
    report = _run(path, spec, "Calculate correlation between x and y", _base_result(["correlation"], 3, {"correlation": bad}))
    assert report["status"] == "FAIL"


def test_iqr_outliers_are_independently_validated(tmp_path):
    path = tmp_path / "outliers.csv"
    df = pd.DataFrame({"amount": [1, 2, 3, 4, 100]})
    df.to_csv(path, index=False)
    s = df["amount"]
    q1, q3 = s.quantile([.25, .75])
    iqr = q3 - q1
    expected = {"amount": {"method": "IQR", "lower_bound": float(q1 - 1.5 * iqr),
                           "upper_bound": float(q3 + 1.5 * iqr),
                           "outlier_count": int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()),
                           "valid_count": 5}}
    spec = {"operations": ["outlier_detection"], "metrics": ["amount"], "group_by": [], "filters": []}
    report = _run(path, spec, "Detect outliers in amount", _base_result(["outlier_detection"], 5, {"outlier_detection": expected}))
    assert report["status"] == "PASS", report


def test_wrong_outlier_count_fails(tmp_path):
    path = tmp_path / "outliers_bad.csv"
    pd.DataFrame({"amount": [1, 2, 3, 4, 100]}).to_csv(path, index=False)
    spec = {"operations": ["outlier_detection"], "metrics": ["amount"], "group_by": [], "filters": []}
    bad = {"amount": {"method": "IQR", "lower_bound": -2.0, "upper_bound": 6.0, "outlier_count": 0, "valid_count": 5}}
    report = _run(path, spec, "Detect outliers in amount", _base_result(["outlier_detection"], 5, {"outlier_detection": bad}))
    assert report["status"] == "FAIL"


def test_monthly_trend_is_independently_validated(tmp_path):
    path = tmp_path / "trend.csv"
    df = pd.DataFrame({"date": ["2026-01-02", "2026-01-20", "2026-02-01"], "sales": [10, 20, 5]})
    df.to_csv(path, index=False)
    expected = [{"period": "2026-01", "sales__sum": 30.0}, {"period": "2026-02", "sales__sum": 5.0}]
    spec = {"operations": ["time_trend"], "metrics": ["sales"], "group_by": [], "filters": [], "time": {"column": "date", "grain": "M"}}
    report = _run(path, spec, "Calculate monthly trend for sales", _base_result(["time_trend"], 3, {"time_trend": expected}))
    assert report["status"] == "PASS", report


def test_grouped_ranking_is_independently_validated(tmp_path):
    path = tmp_path / "rank.csv"
    df = pd.DataFrame({"region": ["A", "A", "B", "B"], "amount": [2, 4, 10, 20]})
    df.to_csv(path, index=False)
    ranking = [
        {"region": "B", "amount__mean": 15.0},
        {"region": "A", "amount__mean": 3.0},
    ]
    spec = {"operations": ["mean", "rank"], "metrics": ["amount"], "group_by": ["region"],
            "filters": [], "sorting": {"direction": "descending", "limit": None}}
    result = _base_result(["mean", "rank"], 4, {"aggregations": ranking[::-1], "ranking": ranking})
    report = _run(path, spec, "Rank regions by average amount", result)
    assert report["status"] == "PASS", report


def test_non_grouped_ranking_wrong_order_fails(tmp_path):
    path = tmp_path / "rank_bad.csv"
    df = pd.DataFrame({"amount": [3, 9, 1]})
    df.to_csv(path, index=False)
    spec = {"operations": ["rank"], "metrics": ["amount"], "group_by": [], "filters": [],
            "sorting": {"direction": "descending", "limit": 2}}
    result = _base_result(["rank"], 3, {"ranking": [{"amount": 3}, {"amount": 9}]})
    report = _run(path, spec, "Show top 2 amount", result)
    assert report["status"] == "FAIL"
