"""Distributions, rates, rankings, data quality, correlation answers, numeric-looking labels."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import answer_support as A
from analytical_spec import build_analytical_spec
from claim_evidence_validator import validate_text_against_evidence
from recommendation_validator import validate_recommendation

CAT = ["job", "education", "housing", "loan", "contact", "month", "age_group", "campaign_contact_group", "y"]
NUM = ["age", "balance", "duration", "campaign", "conversion"]
PROFILE = {"columns": {**{c: {"role": "categorical"} for c in CAT}, **{c: {"role": "continuous"} for c in NUM}}}


def spec(q):
    return build_analytical_spec(q, PROFILE)


def result(ops, metrics, groups, aggs, rows=100, extra=None):
    out = {"aggregations": aggs}
    out.update(extra or {})
    return {"status": "PASS", "analysis_type": "specification_driven", "operations": ops, "metrics": metrics,
            "group_by": groups, "input_rows": rows, "analyzed_rows": rows, "operation_results": out}


@pytest.fixture(autouse=True)
def reset_focus():
    A._STATE["focus"] = None
    yield
    A._STATE["focus"] = None


# ---- question understanding ----------------------------------------------------------------
def test_distribution_questions_become_grouped_counts_with_shares():
    for q in ("Calculate the distribution of customers across education categories.",
              "Show the breakdown of customers by education.", "What percentage of customers have a housing loan?"):
        s = spec(q)
        assert s["operations"] == ["count"] and s["status"] == "READY"
        assert s["answer_focus"]["type"] == "group_distribution"
    assert spec("What percentage of customers have a housing loan?")["group_by"] == ["housing"]   # not housing+loan


def test_numeric_distribution_is_a_set_of_statistics():
    s = spec("Show the distribution of balance.")
    assert s["metrics"] == ["balance"] and s["operations"] == ["mean", "median", "std", "min", "max"]


def test_rate_and_conversion_count_questions():
    assert (spec("What is the overall conversion rate?")["operations"], spec("What is the overall conversion rate?")["metrics"]) == (["mean"], ["conversion"])
    assert spec("What is the conversion rate by education?")["group_by"] == ["education"]
    s = spec("How many customers subscribed?")
    assert s["operations"] == ["sum"] and s["metrics"] == ["conversion"] and s["group_by"] == []
    assert spec("Which month has the highest conversion rate?")["group_by"] == ["month"]


def test_longest_and_shortest_are_rankings():
    s = spec("Which contact type has the longest average call duration?")
    assert s["group_by"] == ["contact"] and s["metrics"] == ["duration"] and s["answer_focus"]["direction"] == "descending"
    assert spec("Which job has the shortest average duration?")["answer_focus"]["direction"] == "ascending"


def test_top_n_means_keep_rank_and_limit():
    s = spec("List the top 5 jobs by average balance.")
    assert s["group_by"] == ["job"] and s["operations"] == ["mean", "rank"] and s["sorting"]["limit"] == 5


def test_data_quality_questions_use_the_right_operations():
    assert spec("How many missing values are there?")["operations"] == ["missing_values"]
    assert spec("Are there duplicate rows?")["operations"] == ["duplicate_count"]


def test_percentiles_are_refused_not_answered_with_something_else():
    s = spec("What is the 90th percentile of balance?")
    assert s["status"] == "REVIEW" and any("Percentiles" in i for i in s["issues"])


def test_age_group_is_one_column_not_a_metric():
    s = spec("Show the average balance by age group.")
    assert s["metrics"] == ["balance"] and s["group_by"] == ["age_group"]


def test_campaign_contacts_is_not_a_grouping_in_distributions():
    assert spec("Show the distribution of campaign contacts by campaign contact group.")["group_by"] == ["campaign_contact_group"]


# ---- answers ---------------------------------------------------------------------------------
def test_distribution_insight_shows_shares_and_validates():
    res = result(["count"], [], ["education"], [{"education": "a", "row_count": 60}, {"education": "b", "row_count": 40}])
    A._STATE["focus"] = {"type": "group_distribution", "aggregate": "count", "direction": "descending", "limit": None}
    ta = {"analytical_spec": {"answer_focus": A._STATE["focus"]}}
    tagged = A.extract_special_result(res)
    text = A.special_insight(tagged, ta)
    assert "a: 60 (60.00%); b: 40 (40.00%)" in text and A.special_validate(text, tagged, ta)["pass"]
    assert not A.special_validate(text.replace("(60.00%)", "(61.00%)"), tagged, ta)["pass"]


def test_rate_is_shown_as_a_percentage_and_grouped_rates_take_over():
    ung = result(["mean"], ["conversion"], [], {"conversion__mean": 0.1170})
    tagged = A.extract_special_result(ung)
    text = A.special_insight(tagged, {})
    assert "conversion rate is 11.70%" in text and A.special_validate(text, tagged, {})["pass"]
    grouped = result(["mean"], ["conversion"], ["job"], [{"job": "a", "conversion__mean": 0.2}, {"job": "b", "conversion__mean": 0.1}])
    tagged = A.extract_special_result(grouped)
    assert tagged is not None and "a: 20.00%; b: 10.00%" in A.special_insight(tagged, {})
    summed = result(["sum"], ["conversion"], [], {"conversion__sum": 5289.0}, rows=45211)
    assert "number of conversions is 5,289" in A.special_insight(A.extract_special_result(summed), {})


def test_top_n_mean_shows_only_the_requested_groups_in_order():
    aggs = [{"job": j, "balance__mean": v} for j, v in (("a", 10.0), ("b", 30.0), ("c", 20.0), ("d", 5.0))]
    res = result(["mean", "rank"], ["balance"], ["job"], aggs)
    focus = {"type": "group_ranking", "direction": "descending", "limit": 2}
    A._STATE["focus"] = focus
    ta = {"analytical_spec": {"answer_focus": focus}}
    tagged = A.extract_special_result(res)
    text = A.special_insight(tagged, ta)
    assert "b: 30.00; c: 20.00" in text and "a: 10.00" not in text
    assert A.special_validate(text, tagged, ta)["pass"]
    assert not A.special_validate(text.replace("b: 30.00; c: 20.00", "c: 20.00; b: 30.00"), tagged, ta)["pass"]
    leaked = text.replace("c: 20.00.", "c: 20.00; a: 10.00.")
    assert not A.special_validate(leaked, tagged, ta)["pass"]


def test_missing_values_and_duplicates():
    miss = result(["missing_values"], [], [], {}, rows=10, extra={"missing_values": {"total_missing_cells": 0, "by_column": {"a": 0, "b": 0}}})
    tagged = A.extract_special_result(miss)
    text = A.special_insight(tagged, {})
    assert "no missing values" in text and A.special_validate(text, tagged, {})["pass"]
    miss2 = result(["missing_values"], [], [], {}, rows=10, extra={"missing_values": {"total_missing_cells": 7, "by_column": {"a": 7, "b": 0}}})
    tagged = A.extract_special_result(miss2)
    text = A.special_insight(tagged, {})
    assert "7 missing cells" in text and "a: 7" in text and A.special_validate(text, tagged, {})["pass"]
    dup = result(["duplicate_count"], [], [], {}, rows=200, extra={"duplicate_count": 5})
    tagged = A.extract_special_result(dup)
    text = A.special_insight(tagged, {})
    assert "5 duplicate rows (2.50% of 200 rows)" in text and A.special_validate(text, tagged, {})["pass"]


def test_correlation_answers_the_yes_no_question_and_reports_shared_variance():
    res = {"status": "PASS", "analysis_type": "specification_driven", "operations": ["correlation"],
           "metrics": ["age", "balance"], "group_by": [], "input_rows": 9, "analyzed_rows": 9,
           "operation_results": {"correlation": {"pearson": {"age": {"age": 1.0, "balance": 0.098}, "balance": {"age": 0.098, "balance": 1.0}}}}}
    ta = {"analytical_spec": {"original_question": "Determine whether age and balance have a strong positive relationship."}}
    tagged = A.extract_special_result(res)
    text = A.special_insight(tagged, ta)
    assert "Answer: no." in text and "r² = 0.010" in text and "0.96% shared variance" in text
    assert A.special_validate(text, tagged, ta)["pass"]
    assert not A.special_validate(text.replace("Answer: no.", "Answer: yes."), tagged, ta)["pass"]
    plain = A.special_insight(tagged, {"analytical_spec": {"original_question": "Is there a correlation between age and balance?"}})
    assert "Answer:" not in plain or "no. The data do not show" not in plain


# ---- numeric-looking category labels -----------------------------------------------------------
def test_digits_inside_group_labels_are_not_numeric_claims():
    evidence = {"group_results": {"26-35": {"mean_value": 870.82, "group_values": {"age_group": "26-35"}},
                                  "65+": {"mean_value": 902.54, "group_values": {"age_group": "65+"}}}}
    text = "Average balance by age_group: 26-35: 870.82; 65+: 902.54."
    assert validate_text_against_evidence(text, evidence)["status"] == "PASS"
    assert validate_text_against_evidence(text.replace("870.82", "871.99"), evidence)["status"] == "REVIEW"
    rec = "The highest average is 902.54 for age_group=65+, and the lowest is 870.82 for age_group=26-35."
    assert validate_recommendation(rec, {"status": "PASS", **evidence}, None, None)["unverified_numeric_claims"] == 0
