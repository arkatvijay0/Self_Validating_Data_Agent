"""
Facade used by main.py for results that have dedicated, fully validated
handling: grouped counts (grouped_count_support) and ungrouped / non-mean
grouped aggregates (aggregate_support). Grouped means keep their existing path.
"""
import grouped_count_support as _count
import aggregate_support as _agg

SPECIAL_TYPES = {_count.ANALYSIS_TYPE, _agg.ANALYSIS_TYPE}

# The question's answer focus (e.g. "top 5", "rank", "conversion rate") decides which
# validated path takes over a grouped result. main() sets it once per run, right
# after task analysis; without it only the default routing applies.
_STATE = {"focus": None}


def set_answer_focus(task_analysis):
    spec = task_analysis.get("analytical_spec") if isinstance(task_analysis, dict) else None
    focus = spec.get("answer_focus") if isinstance(spec, dict) else None
    _STATE["focus"] = focus if isinstance(focus, dict) else None


def extract_special_result(primary_result):
    normalized = _count.extract_specification_grouped_count(primary_result)
    if normalized is not None:
        return {"analysis_type": _count.ANALYSIS_TYPE, "data": normalized}
    normalized = _agg.extract_specification_aggregate(primary_result, _STATE["focus"])
    if normalized is not None:
        return {"analysis_type": _agg.ANALYSIS_TYPE, "data": normalized}
    return None


def special_insight(tagged, task_analysis):
    if tagged["analysis_type"] == _count.ANALYSIS_TYPE:
        return _count.generate_grouped_count_insight(tagged["data"], task_analysis)
    return _agg.generate_aggregate_insight(tagged["data"], task_analysis)


def special_validate(insight, tagged, task_analysis):
    if tagged["analysis_type"] == _count.ANALYSIS_TYPE:
        return _count.validate_grouped_count_insight(insight, tagged["data"], task_analysis)
    return _agg.validate_aggregate_insight(insight, tagged["data"], task_analysis)


def special_evidence(tagged, primary_result, task_analysis):
    if tagged["analysis_type"] == _count.ANALYSIS_TYPE:
        return _count.build_grouped_count_evidence(tagged["data"], primary_result, task_analysis)
    return _agg.build_aggregate_evidence(tagged["data"], primary_result, task_analysis)


def special_mapping(insight, evidence, task_analysis):
    if (evidence or {}).get("analysis_type") == _count.ANALYSIS_TYPE:
        return _count.build_grouped_count_evidence_mapping(insight, evidence, task_analysis)
    return _agg.build_aggregate_evidence_mapping(insight, evidence, task_analysis)


def special_recommendation(evidence, task_analysis):
    if (evidence or {}).get("analysis_type") == _count.ANALYSIS_TYPE:
        return _count.generate_grouped_count_recommendation(evidence, task_analysis)
    return _agg.generate_aggregate_recommendation(evidence, task_analysis)


def build_iqr_evidence_mapping(insight, iqr_result):
    """Check an IQR outlier insight against the validated IQR result."""
    text = str(insight or "")
    needed = {}
    try:
        needed["outlier_count"] = f"{int(iqr_result['outlier_count']):,}"
        needed["valid_count"] = f"{int(iqr_result['valid_count']):,}"
        needed["lower_bound"] = f"{float(iqr_result['lower_bound']):,.2f}"
        needed["upper_bound"] = f"{float(iqr_result['upper_bound']):,.2f}"
    except (KeyError, TypeError, ValueError):
        return {"status": "PARTIAL", "analysis_type": "outlier_analysis",
                "reason": "The IQR result does not contain the expected fields."}
    import re as _re
    checks = [{"check": name, "status": "PASS" if _re.search(r"(?<![\d.,])" + _re.escape(v) + r"(?![\d,]|\.\d)", text) else "FAIL"}
              for name, v in needed.items()]
    passed = all(c["status"] == "PASS" for c in checks)
    return {
        "status": "PASS" if passed else "FAIL", "analysis_type": "outlier_analysis",
        "claim_validation": {"status": "PASS" if passed else "FAIL", "total_claims": len(checks),
                             "passed_claims": sum(c["status"] == "PASS" for c in checks),
                             "failed_claims": sum(c["status"] == "FAIL" for c in checks), "checks": checks},
        "evidence_validation": {"status": "PASS"},
        "reason": None if passed else "The insight must report the outlier count, valid count and both IQR bounds exactly.",
        "evidence_sources": [{"source_id": "primary_result.operation_results.outlier_detection",
                              "source_type": "independently_validated_iqr_result"}],
        "traceability_status": "PASS" if passed else "REVIEW",
    }


def build_isolation_forest_mapping(insight, primary_result):
    """Check an Isolation Forest insight against the validated detector result."""
    import re as _re
    text = str(insight or "")
    r = primary_result if isinstance(primary_result, dict) else {}
    try:
        needed = {
            "row_count": f"{int(r['row_count']):,}",
            "anomaly_count": f"{int(r['anomaly_count']):,}",
            "normal_count": f"{int(r['normal_count']):,}",
            "anomaly_percentage": f"{float(r['anomaly_percentage']):.2f}%",
        }
        features = [str(f) for f in r["feature_columns"]]
    except (KeyError, TypeError, ValueError):
        return {"status": "PARTIAL", "analysis_type": "anomaly_detection",
                "reason": "The Isolation Forest result does not contain the expected fields."}
    checks = [{"check": k, "status": "PASS" if _re.search(r"(?<![\d.,])" + _re.escape(v), text) else "FAIL"}
              for k, v in needed.items()]
    checks.append({"check": "feature_columns",
                   "status": "PASS" if all(_re.search(r"(?<!\w)" + _re.escape(f) + r"(?!\w)", text) for f in features) else "FAIL"})
    passed = all(c["status"] == "PASS" for c in checks)
    return {
        "status": "PASS" if passed else "FAIL", "analysis_type": "anomaly_detection",
        "claim_validation": {"status": "PASS" if passed else "FAIL", "total_claims": len(checks),
                             "passed_claims": sum(c["status"] == "PASS" for c in checks),
                             "failed_claims": sum(c["status"] == "FAIL" for c in checks), "checks": checks},
        "evidence_validation": {"status": "PASS"},
        "reason": None if passed else "The insight must report the validated counts, percentage and feature list exactly.",
        "evidence_sources": [{"source_id": "primary_result", "source_type": "independently_validated_isolation_forest_result"}],
        "traceability_status": "PASS" if passed else "REVIEW",
    }
