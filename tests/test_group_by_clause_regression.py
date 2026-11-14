"""Regression tests: only columns in the grouping clause are grouping columns."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analytical_spec import build_analytical_spec

CATEGORICAL = ["job", "marital", "education", "default", "housing", "loan",
               "contact", "month", "poutcome", "y"]
NUMERIC = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
PROFILE = {"columns": {
    **{c: {"role": "categorical"} for c in CATEGORICAL},
    **{c: {"role": "continuous"} for c in NUMERIC},
}}


def group_by(question):
    return build_analytical_spec(question, PROFILE)["group_by"]


def test_plural_metric_word_is_not_a_grouping_column():
    # "contacts" matches the `contact` column via plural matching, but it is
    # part of the metric phrase, not the grouping.
    spec = build_analytical_spec(
        "Calculate the average campaign contacts grouped by loan status.", PROFILE)
    assert spec["group_by"] == ["loan"]
    assert spec["metrics"] == ["campaign"]


def test_filter_clause_column_is_not_a_grouping_column():
    assert group_by("Average campaign by loan where contact is cellular") == ["loan"]


def test_multiple_grouping_columns_keep_written_order():
    assert group_by("Average balance grouped by contact and loan") == ["contact", "loan"]
    assert group_by("Average balance grouped by loan and contact") == ["loan", "contact"]


def test_simple_by_and_per_still_work():
    assert group_by("Average balance by job") == ["job"]
    assert group_by("Average duration per month") == ["month"]


def test_ranking_phrase_keeps_group_before_by():
    assert group_by("Find the top 5 jobs by average balance, descending") == ["job"]


def test_no_grouping_language_means_no_grouping():
    assert group_by("What is the average campaign for contacts?") == []
