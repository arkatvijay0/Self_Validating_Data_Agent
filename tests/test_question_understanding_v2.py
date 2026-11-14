"""Question understanding: grouped ranking, number words, unique counts, comparisons, filters."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analytical_spec import build_analytical_spec

CAT = ["job", "education", "marital", "housing", "loan", "contact", "month", "poutcome", "y", "age_group"]
NUM = ["age", "balance", "duration", "campaign", "pdays", "previous"]
PROFILE = {"columns": {**{c: {"role": "categorical"} for c in CAT},
                       **{c: {"role": "continuous"} for c in NUM}}}


def spec(q):
    return build_analytical_spec(q, PROFILE)


def test_which_group_has_highest_average_is_grouped_not_scalar_max():
    s = spec("Which job category has the highest average balance?")
    assert s["group_by"] == ["job"] and s["metrics"] == ["balance"]
    assert s["operations"] == ["mean"]            # 'highest' must not become max
    assert s["answer_focus"]["direction"] == "descending"


def test_lowest_is_ascending():
    s = spec("Which education category has the lowest average balance?")
    assert s["group_by"] == ["education"] and s["answer_focus"]["direction"] == "ascending"


def test_adjacent_column_words_form_one_phrase():
    s = spec("Which month has the highest average campaign duration?")
    assert s["metrics"] == ["duration"] and s["group_by"] == ["month"]
    assert spec("Identify unusual values in campaign duration")["metrics"] == ["duration"]


def test_scalar_extrema_stay_scalar():
    s = spec("What is the maximum balance?")
    assert s["group_by"] == [] and s["operations"] == ["max"] and s["metrics"] == ["balance"]


def test_top_n_by_count_with_words_and_digits():
    for q in ("Identify the top three job categories by customer count.",
              "Identify the top 3 job categories by customer count."):
        s = spec(q)
        assert s["group_by"] == ["job"] and s["operations"] == ["count"]
        assert s["sorting"] == {"direction": "descending", "limit": 3}


def test_most_customers_and_most_common():
    assert spec("Which contact type has the most customers?")["answer_focus"]["limit"] == 1
    assert spec("What is the most common job?")["group_by"] == ["job"]


def test_unique_count_targets_the_named_column():
    s = spec("How many unique jobs are there?")
    assert s["operations"] == ["nunique"] and s["metrics"] == ["job"]


def test_higher_or_lower_for_customers_with_attribute_is_a_comparison():
    s = spec("Is the average balance higher or lower for customers with a loan?")
    assert s["group_by"] == ["loan"] and s["metrics"] == ["balance"]
    from task_analyzer import analyze_task_mock
    s = analyze_task_mock(
        "Compare average balance between customers with and without housing loans.", PROFILE
    )["analytical_spec"]
    assert s["group_by"] == ["housing"] and s["status"] == "READY"


def test_categorical_filters_are_reported_not_dropped():
    s = spec("What is the average balance for customers with a loan?")
    assert s["status"] == "REVIEW" and any("not supported" in i for i in s["issues"])
    s = spec("Average campaign by loan where contact is cellular")
    assert s["status"] == "REVIEW" and s["group_by"] == ["loan"]


def test_numeric_filter_with_is_is_parsed():
    s = spec("Average balance grouped by housing where age is greater than 40")
    assert s["metrics"] == ["balance"]
    assert [(f["column"], f["operator"], f["value"]) for f in s["filters"]] == [("age", ">", 40.0)]


def test_unmatched_measure_is_an_issue_not_a_guess():
    s = spec("What is the average salary by job?")
    assert s["status"] == "REVIEW" and s["metrics"] == []
    assert spec("What is the maximum revenue?")["status"] == "REVIEW"


def test_correlated_is_recognised():
    assert "correlation" in spec("Is balance correlated with duration?")["operations"]
