"""Correctly rounded numbers must be grounded; wrong numbers must not be."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from claim_evidence_validator import validate_text_against_evidence
from recommendation_generator import generate_recommendation
from recommendation_validator import validate_recommendation

ROWS = [("cellular", "no", 2.669675311415152), ("cellular", "yes", 2.794375),
        ("telephone", "no", 3.28869286287089), ("telephone", "yes", 4.080097087378641),
        ("unknown", "no", 2.8078813250819077), ("unknown", "yes", 2.677165354330709)]
EVIDENCE = {
    "analysis_type": "grouped_mean_analysis", "status": "PASS",
    "metric_column": "campaign", "group_column": "contact and loan",
    "group_columns": ["contact", "loan"],
    "group_results": {f"contact={c}, loan={l}": {"mean_value": m,
                      "group_values": {"contact": c, "loan": l}} for c, l, m in ROWS},
}
PRIMARY = {"status": "PASS", "analysis_type": "specification_driven",
           "operations": ["mean"], "metrics": ["campaign"], "group_by": ["contact", "loan"],
           "operation_results": {"aggregations": [
               {"contact": c, "loan": l, "campaign__mean": m} for c, l, m in ROWS]}}
TEXT = ("For campaign grouped by contact and loan, the highest average is 4.08 for "
        "contact=telephone, loan=yes, and the lowest average is 2.67 for "
        "contact=cellular, loan=no.")


def test_rounded_value_is_grounded_by_claim_validator():
    assert validate_text_against_evidence("The mean is 2.67.", {"m": 2.669675})["status"] == "PASS"


def test_wrong_rounding_is_not_grounded():
    assert validate_text_against_evidence("The mean is 2.69.", {"m": 2.669675})["status"] == "REVIEW"


def test_integer_claims_stay_strict():
    assert validate_text_against_evidence("There are 3 rows.", {"m": 2.7})["status"] == "REVIEW"


def test_group_count_claim_is_grounded():
    assert validate_text_against_evidence("There are 6 groups.", EVIDENCE)["status"] == "PASS"


def test_generated_text_passes_independent_validation():
    rec = generate_recommendation("comparison", EVIDENCE)
    assert rec["status"] == "PASS"
    result = validate_recommendation(rec, EVIDENCE, "grouped_mean_analysis", PRIMARY,
                                     {"recommendation": {"overall": "PASS"}})
    assert result["status"] == "PASS", result["reasons"]


def test_pipeline_text_passes():
    assert validate_recommendation(TEXT, EVIDENCE, None, PRIMARY)["status"] == "PASS"


def test_wrong_value_is_review():
    assert validate_recommendation(TEXT.replace("4.08", "4.09"), EVIDENCE, None, PRIMARY)["status"] == "REVIEW"


def test_swapped_values_are_review():
    swapped = TEXT.replace("4.08", "@").replace("2.67", "4.08").replace("@", "2.67")
    assert validate_recommendation(swapped, EVIDENCE, None, PRIMARY)["status"] == "REVIEW"


def test_review_always_has_a_reason():
    result = validate_recommendation(TEXT, EVIDENCE, None, PRIMARY,
                                     {"recommendation": {"overall": "REVIEW"}})
    assert result["status"] == "REVIEW" and result["reasons"]


def test_single_group_column_accepts_bare_values():
    rows = [("no", 2.6949959241631385), ("yes", 2.659844054580897)]
    primary = {"status": "PASS", "analysis_type": "specification_driven", "operations": ["mean"],
               "metrics": ["campaign"], "group_by": ["loan"],
               "operation_results": {"aggregations": [{"loan": v, "campaign__mean": m} for v, m in rows]}}
    text = ("For campaign grouped by loan, the highest average is 2.69 for no, and the "
            "lowest average is 2.66 for yes.")
    assert validate_recommendation(text, {"status": "PASS", "means": dict(rows)}, None, primary)["status"] == "PASS"
    swapped = text.replace("2.69", "@").replace("2.66", "2.69").replace("@", "2.66")
    assert validate_recommendation(swapped, {"status": "PASS", "means": dict(rows)}, None, primary)["status"] == "REVIEW"


def test_generator_handles_bad_inputs_without_crashing():
    assert generate_recommendation("mean", None)["status"] == "FAIL"
    bad = {"status": "PASS", "analysis_type": "correlation_analysis", "x_variable": "a",
           "y_variable": "b", "correlation": "n/a", "direction": "positive", "strength": "weak"}
    assert generate_recommendation("relationship_analysis", bad)["status"] == "FAIL"


def test_mean_from_list_style_aggregations():
    rec = generate_recommendation("mean", {"status": "PASS", "operation_results": {
        "aggregations": [{"campaign__mean": 2.76}]}})
    assert rec["status"] == "PASS" and "2.76" in rec["recommendation"]
