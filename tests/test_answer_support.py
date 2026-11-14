"""Deterministic insight, validation, evidence and recommendation for counts and aggregates."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import answer_support as A
from main import downgrade_if_no_answer, build_clarification_request
from anomaly_detector import select_anomaly_features
import pandas as pd

COUNT = {"status": "PASS", "analysis_type": "specification_driven", "operations": ["count"],
         "metrics": [], "group_by": ["job"], "filters_applied": [], "input_rows": 100, "analyzed_rows": 100,
         "operation_results": {"aggregations": [
             {"job": "a", "row_count": 10}, {"job": "b", "row_count": 50},
             {"job": "c", "row_count": 30}, {"job": "d", "row_count": 10}]}}
TOP3 = {"analytical_spec": {"answer_focus": {"type": "group_top_n", "aggregate": "count",
                                             "direction": "descending", "limit": 3}}}


def test_top_n_count_is_ranked_and_truncated():
    tagged = A.extract_special_result(COUNT)
    insight = A.special_insight(tagged, TOP3)
    assert "b: 50; c: 30" in insight and "d:" not in insight.split("Counts")[0]
    assert A.special_validate(insight, tagged, TOP3)["pass"] is True


def test_wrong_count_or_order_fails_validation():
    tagged = A.extract_special_result(COUNT)
    good = A.special_insight(tagged, TOP3)
    assert A.special_validate(good.replace("b: 50", "b: 51"), tagged, TOP3)["pass"] is False
    swapped = good.replace("b: 50; c: 30", "c: 30; b: 50")
    assert A.special_validate(swapped, tagged, TOP3)["pass"] is False


def test_ungrouped_and_grouped_aggregates():
    ungrouped = {"status": "PASS", "analysis_type": "specification_driven", "operations": ["mean"],
                 "metrics": ["balance"], "group_by": [], "filters_applied": [], "input_rows": 10,
                 "analyzed_rows": 10, "operation_results": {"aggregations": {"balance__mean": 864.2999}}}
    tagged = A.extract_special_result(ungrouped)
    text = A.special_insight(tagged, {})
    assert "864.30" in text and A.special_validate(text, tagged, {})["pass"]
    assert not A.special_validate(text.replace("864.30", "864.31"), tagged, {})["pass"]

    grouped = dict(ungrouped, operations=["median"], group_by=["loan"],
                   operation_results={"aggregations": [
                       {"loan": "no", "balance__median": 100.0}, {"loan": "yes", "balance__median": 50.5}]})
    tagged = A.extract_special_result(grouped)
    text = A.special_insight(tagged, {})
    assert "no: 100.00; yes: 50.50" in text and A.special_validate(text, tagged, {})["pass"]


def test_single_metric_grouped_mean_keeps_its_existing_path():
    grouped_mean = {"status": "PASS", "analysis_type": "specification_driven", "operations": ["mean"],
                    "metrics": ["balance"], "group_by": ["loan"], "input_rows": 2, "analyzed_rows": 2,
                    "operation_results": {"aggregations": [{"loan": "no", "balance__mean": 1.0}]}}
    assert A.extract_special_result(grouped_mean) is None


def test_correlation_value_strength_and_validation():
    result = {"status": "PASS", "analysis_type": "specification_driven", "operations": ["correlation"],
              "metrics": ["age", "balance"], "group_by": [], "input_rows": 5, "analyzed_rows": 5,
              "operation_results": {"correlation": {"pearson": {
                  "age": {"age": 1.0, "balance": 0.42}, "balance": {"age": 0.42, "balance": 1.0}}}}}
    tagged = A.extract_special_result(result)
    text = A.special_insight(tagged, {})
    assert "0.420" in text and "moderate positive" in text and A.special_validate(text, tagged, {})["pass"]


def test_no_answer_insight_is_never_pass():
    out = downgrade_if_no_answer("A result was produced, but it could not be safely converted into a numerical insight.",
                                 {"overall": "PASS", "pass": True})
    assert out["overall"] == "REVIEW" and out["pass"] is False
    assert downgrade_if_no_answer("The average balance is 1.00.", {"overall": "PASS"})["overall"] == "PASS"


def test_clarification_for_unsupported_and_open_ended_questions():
    profile = {"columns": {"balance": {"role": "continuous"}, "job": {"role": "categorical"},
                           "age": {"role": "continuous"}, "conversion": {"role": "continuous"}}}
    unsupported = {"analytical_spec": {"status": "REVIEW", "original_question": "avg of salary",
                                       "issues": ["No numeric column in the dataset matches the measure named in the question."]}}
    assert "No numeric column" in build_clarification_request(unsupported, profile)
    open_ended = {"analytical_spec": {"status": "READY", "issues": [],
                                      "original_question": "What are the key factors affecting conversion?"}}
    assert "open-ended" in build_clarification_request(open_ended, profile)
    fine = {"analytical_spec": {"status": "READY", "issues": [], "original_question": "average balance by job"}}
    assert build_clarification_request(fine, profile) is None


def test_isolation_forest_features_follow_the_question_and_skip_flags():
    df = pd.DataFrame({"balance": range(10), "age": range(10, 20), "conversion": [0, 1] * 5,
                       "month_number": range(10)})
    assert select_anomaly_features(df, {"metrics": ["balance"]}) == ["balance"]
    assert select_anomaly_features(df, {}) == ["balance", "age"]


def test_iqr_mapping_checks_exact_values():
    iqr = {"outlier_count": 3551, "valid_count": 45211, "lower_bound": -1877.25, "upper_bound": 2792.75}
    ok = "3,551 outliers from 45,211 valid values below -1,877.25 or above 2,792.75"
    assert A.build_iqr_evidence_mapping(ok, iqr)["status"] == "PASS"
    assert A.build_iqr_evidence_mapping(ok.replace("3,551", "3,552"), iqr)["status"] == "FAIL"
