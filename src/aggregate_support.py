"""
Aggregate support for specification-driven results that the grouped-mean and
grouped-count paths do not cover:

  * ungrouped aggregates    "What is the average balance?", "maximum balance",
                            "How many customers are there?", "total balance",
                            "average age for customers with duration above 300"
  * grouped non-mean        "median duration by contact", "total balance for
                            each education", "standard deviation of balance by
                            loan", "which housing group has the highest median
                            balance?"

The executor's validated result is the only source of numbers. This module
writes the insight, validates it deterministically, builds the evidence and
the insight-to-evidence mapping, and writes a recommendation whose numbers all
come from the evidence.
"""
import math
import re

ANALYSIS_TYPE = "specification_aggregate_analysis"

SUPPORTED_OPS = ("mean", "median", "sum", "min", "max", "std", "variance", "count", "nunique")
_OP_PHRASE = {
    "mean": "average", "median": "median", "sum": "total", "min": "minimum",
    "max": "maximum", "std": "standard deviation", "variance": "variance",
    "count": "count of non-missing values", "nunique": "number of distinct values",
}
_FILTER_WORDS = {">": "above", ">=": "at least", "<": "below", "<=": "at most",
                 "==": "equal to", "!=": "not equal to"}


def _finite(value):
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _is_count_op(op):
    return op in {"count", "nunique", "row_count"}


def fmt(op, value):
    """Display format: whole numbers with separators for counts, 2 decimals otherwise."""
    return f"{int(round(value)):,}" if _is_count_op(op) else f"{value:,.2f}"


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def extract_specification_aggregate(primary_result):
    """Normalise a validated ungrouped or grouped-non-mean aggregate, else None."""
    if not isinstance(primary_result, dict):
        return None
    if str(primary_result.get("analysis_type", "")).lower() != "specification_driven":
        return None
    if str(primary_result.get("status", "")).upper() != "PASS":
        return None

    operations = [str(op).strip().lower() for op in (primary_result.get("operations") or [])]
    if operations == ["correlation"]:
        return _extract_correlation(primary_result)
    if not operations or any(op not in SUPPORTED_OPS for op in operations):
        return None                           # rank/outliers/time trend/... use other paths
    metrics = [str(m).strip() for m in (primary_result.get("metrics") or []) if str(m).strip()]
    groups = [str(g).strip() for g in (primary_result.get("group_by") or []) if str(g).strip()]

    # Paths that already exist and are validated end to end.
    if groups and operations == ["mean"] and len(metrics) < 2:
        return None                           # single-metric grouped mean (proven path)
    if groups and not metrics and operations == ["count"]:
        return None                           # grouped count -> grouped_count_support

    results = primary_result.get("operation_results")
    aggregations = results.get("aggregations") if isinstance(results, dict) else None
    if aggregations is None:
        return None

    # (key, metric, op) measures to report
    measures = []
    if metrics:
        for metric in metrics:
            for op in operations:
                measures.append({"key": f"{metric}__{op}", "metric": metric, "op": op})
    elif operations == ["count"]:
        measures.append({"key": "row_count", "metric": None, "op": "count"})
    else:
        return None

    rows = []
    if groups:
        if not isinstance(aggregations, list) or not aggregations:
            return None
        for row in aggregations:
            if not isinstance(row, dict) or any(g not in row or row[g] is None for g in groups):
                return None
            values = {}
            for m in measures:
                number = _finite(row.get(m["key"]))
                if number is None:
                    return None
                values[m["key"]] = number
            gv = {g: (row[g] if isinstance(row[g], (int, float, str)) else str(row[g])) for g in groups}
            label = (str(gv[groups[0]]) if len(groups) == 1
                     else ", ".join(f"{g}={gv[g]}" for g in groups))
            rows.append({"label": label, "group_values": gv, "values": values})
        if len({r["label"] for r in rows}) != len(rows):
            return None
    else:
        if not isinstance(aggregations, dict):
            return None
        values = {}
        for m in measures:
            number = _finite(aggregations.get(m["key"]))
            if number is None:
                return None
            values[m["key"]] = number
        rows.append({"label": None, "group_values": {}, "values": values})

    analyzed = primary_result.get("analyzed_rows", primary_result.get("input_rows"))
    total_rows = int(analyzed) if _finite(analyzed) is not None else None
    filters = [f for f in (primary_result.get("filters_applied") or []) if isinstance(f, dict)]
    return {
        "mode": "grouped" if groups else "ungrouped",
        "group_columns": groups, "group_column": " and ".join(groups),
        "measures": measures, "rows": rows,
        "total_rows": total_rows,
        "input_rows": int(primary_result["input_rows"]) if _finite(primary_result.get("input_rows")) is not None else None,
        "filters": filters,
    }



def _strength(r):
    a = abs(r)
    if a < 0.1:
        return "negligible"
    if a < 0.3:
        return "weak"
    if a < 0.5:
        return "moderate"
    if a < 0.7:
        return "strong"
    return "very strong"


def _direction(r):
    return "positive" if r > 0 else "negative" if r < 0 else "no"


def _extract_correlation(primary_result):
    """Pairwise correlations from a validated correlation matrix."""
    metrics = [str(m).strip() for m in (primary_result.get("metrics") or []) if str(m).strip()]
    results = primary_result.get("operation_results")
    matrices = results.get("correlation") if isinstance(results, dict) else None
    if len(metrics) < 2 or not isinstance(matrices, dict) or not matrices:
        return None
    pairs = []
    for method, matrix in matrices.items():
        if not isinstance(matrix, dict):
            return None
        for i, x in enumerate(metrics):
            for y in metrics[i + 1:]:
                value = _finite((matrix.get(x) or {}).get(y)) if isinstance(matrix.get(x), dict) else None
                if value is None or not -1.0 <= value <= 1.0:
                    return None
                pairs.append({"x": x, "y": y, "method": str(method).capitalize(), "value": value})
    analyzed = primary_result.get("analyzed_rows", primary_result.get("input_rows"))
    return {
        "mode": "correlation", "pairs": pairs, "metrics": metrics,
        "total_rows": int(analyzed) if _finite(analyzed) is not None else None,
        "filters": [f for f in (primary_result.get("filters_applied") or []) if isinstance(f, dict)],
        "measures": [], "rows": [], "group_columns": [], "group_column": "", "input_rows": None,
    }


def _pair_sentence(p):
    r = p["value"]
    strength = _strength(r)
    if strength == "negligible":
        return (f"The {p['method']} correlation between {p['x']} and {p['y']} is {r:.3f}, "
                "a negligible linear relationship.")
    return (f"The {p['method']} correlation between {p['x']} and {p['y']} is {r:.3f}, "
            f"a {strength} {_direction(r)} linear relationship.")


def _focus(task_analysis):
    spec = task_analysis.get("analytical_spec") if isinstance(task_analysis, dict) else None
    focus = spec.get("answer_focus") if isinstance(spec, dict) else None
    return focus if isinstance(focus, dict) and focus.get("type") == "group_extrema" else None


def _filter_text(filters):
    parts = []
    for f in filters:
        column, operator, value = f.get("column"), f.get("operator"), f.get("value")
        if column is None or value is None:
            continue
        parts.append(f"{column} {_FILTER_WORDS.get(operator, operator)} {value}")
    return " and ".join(parts)


def _measure_phrase(m):
    if m["metric"] is None:
        return "number of records"
    if m["op"] == "nunique":
        return f"number of distinct {m['metric']} values"
    return f"{_OP_PHRASE[m['op']]} {m['metric']}"


def _scope_text(n):
    """Rows the figure was computed from, with any filter."""
    text = ""
    ft = _filter_text(n["filters"])
    if ft:
        text += f" for rows where {ft}"
    if n["total_rows"] is not None:
        text += f" ({n['total_rows']:,} rows analysed"
        if ft and n["input_rows"] is not None and n["input_rows"] != n["total_rows"]:
            text += f" of {n['input_rows']:,}"
        text += ")"
    return text


# ---------------------------------------------------------------------------
# Insight
# ---------------------------------------------------------------------------

def generate_aggregate_insight(n, task_analysis=None):
    if n is None:
        return "No valid aggregate result was available for insight generation."
    if n["mode"] == "correlation":
        ft = _filter_text(n["filters"])
        basis = (f" Based on {n['total_rows']:,} rows" if n["total_rows"] is not None else " Based on the analysed rows")
        basis += f" where {ft}." if ft else "."
        return (" ".join(_pair_sentence(p) for p in n["pairs"]) + basis +
                " Correlation measures linear association only and does not establish causation.")
    scope = _scope_text(n)

    if n["mode"] == "ungrouped":
        values = n["rows"][0]["values"]
        sentences = []
        for m in n["measures"]:
            v = values[m["key"]]
            if m["metric"] is None:
                sentences.append(f"There are {fmt('count', v)} records in the dataset{scope}.")
            else:
                sentences.append(f"The {_measure_phrase(m)} is {fmt(m['op'], v)}{scope}.")
        return " ".join(sentences) + " This is a descriptive figure and does not establish causation."

    lines = []
    focus = _focus(task_analysis)
    for m in n["measures"]:
        ordered = sorted(n["rows"], key=lambda r: (-r["values"][m["key"]], r["label"].casefold()))
        listing = "; ".join(f"{r['label']}: {fmt(m['op'], r['values'][m['key']])}" for r in ordered)
        context = [f"{len(ordered)} groups"]
        if n["total_rows"] is not None:
            rows_text = f"{n['total_rows']:,} rows analysed"
            if _filter_text(n["filters"]) and n["input_rows"] not in (None, n["total_rows"]):
                rows_text += f" of {n['input_rows']:,}"
            context.append(rows_text)
        if _filter_text(n["filters"]):
            context.append(f"rows where {_filter_text(n['filters'])}")
        line = (f"{_measure_phrase(m).capitalize()} by {n['group_column']} "
                f"({', '.join(context)}): {listing}.")
        if focus and len(ordered) >= 2:
            hi, lo = ordered[0], ordered[-1]
            hi_t = f"{hi['label']} ({fmt(m['op'], hi['values'][m['key']])})"
            lo_t = f"{lo['label']} ({fmt(m['op'], lo['values'][m['key']])})"
            if focus.get("direction") == "ascending":
                line += f" Lowest {_measure_phrase(m)}: {lo_t}; highest: {hi_t}."
            else:
                line += f" Highest {_measure_phrase(m)}: {hi_t}; lowest: {lo_t}."
        lines.append(line)
    return " ".join(lines) + " These are descriptive values and do not establish causation."


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_aggregate_insight(insight, n, task_analysis=None):
    text = str(insight or "")
    lower = " ".join(text.casefold().replace("_", " ").split())
    checks = {"result_status": n is not None, "metric_mentioned": False,
              "operation_mentioned": False, "every_value_matches_validated_result": False}
    if n is None:
        return {"overall": "FAIL", "pass": False, "checks": checks,
                "reason": "The aggregate result is invalid or incomplete."}

    if n["mode"] == "correlation":
        checks = {"result_status": True, "variables_mentioned": True,
                  "every_correlation_matches_validated_result": True,
                  "strength_and_direction_match": True}
        for p in n["pairs"]:
            checks["variables_mentioned"] &= p["x"].casefold() in lower and p["y"].casefold() in lower
            sentence = re.escape(f"{p['value']:.3f}")
            checks["every_correlation_matches_validated_result"] &= re.search(
                r"(?<![\d.])" + sentence + r"(?!\d)", text) is not None
            expected = _strength(p["value"])
            checks["strength_and_direction_match"] &= (
                expected in lower and (expected == "negligible" or _direction(p["value"]) in lower))
        passed = all(checks.values())
        return {"overall": "PASS" if passed else "FAIL", "pass": passed,
                "analysis_type": ANALYSIS_TYPE, "checks": checks,
                "reason": None if passed else
                "The insight must name both variables and report each validated coefficient, strength and direction."}

    metrics = [m["metric"] for m in n["measures"] if m["metric"]]
    checks["metric_mentioned"] = all(
        " ".join(m.casefold().replace("_", " ").split()) in lower for m in metrics
    ) if metrics else True
    checks["operation_mentioned"] = all(
        (re.search(r"\bdistinct\b", lower) if m["op"] == "nunique"
         else re.search(rf"\b{re.escape(_OP_PHRASE[m['op']])}\b", lower))
        if m["metric"] else ("record" in lower)
        for m in n["measures"]
    )

    if n["mode"] == "ungrouped":
        values = n["rows"][0]["values"]
        ok = []
        for m in n["measures"]:
            shown = fmt(m["op"], values[m["key"]])
            ok.append(re.search(r"(?<![\d.,])" + re.escape(shown) + r"(?![\d,]|\.\d)", text) is not None)
        checks["every_value_matches_validated_result"] = bool(ok) and all(ok)
    else:
        checks["all_group_columns_mentioned"] = all(
            " ".join(g.casefold().replace("_", " ").split()) in lower for g in n["group_columns"]
        )
        pair_ok = []
        for m in n["measures"]:
            for r in n["rows"]:
                shown = fmt(m["op"], r["values"][m["key"]])
                pattern = re.escape(r["label"]) + r"\s*:\s*" + re.escape(shown) + r"(?![\d,]|\.\d)"
                pair_ok.append(re.search(pattern, text, flags=re.IGNORECASE) is not None)
        checks["every_value_matches_validated_result"] = bool(pair_ok) and all(pair_ok)

        focus = _focus(task_analysis)
        if focus:
            for m in n["measures"]:
                ordered = sorted(n["rows"], key=lambda r: (-r["values"][m["key"]], r["label"].casefold()))
                winner = ordered[-1] if focus.get("direction") == "ascending" else ordered[0]
                key = "winner_" + m["key"] + "_reported"
                checks[key] = re.search(
                    re.escape(winner["label"]) + r"\s*\(" + re.escape(fmt(m["op"], winner["values"][m["key"]])) + r"\)",
                    text, flags=re.IGNORECASE) is not None
    passed = all(checks.values())
    return {
        "overall": "PASS" if passed else "FAIL", "pass": passed,
        "analysis_type": ANALYSIS_TYPE, "checks": checks,
        "reason": None if passed else
        "The insight must name the metric and operation and report every validated value exactly.",
    }


# ---------------------------------------------------------------------------
# Evidence / mapping
# ---------------------------------------------------------------------------

def build_aggregate_evidence(n, primary_result, task_analysis=None):
    if n is None:
        return {"status": "FAIL", "analysis_type": ANALYSIS_TYPE,
                "reason": "No validated aggregate result was available."}
    if n["mode"] == "correlation":
        return {"status": "PASS", "analysis_type": ANALYSIS_TYPE, "mode": "correlation",
                "pairs": n["pairs"], "total_rows": n["total_rows"],
                "source_analysis_type": "specification_driven", "source_result": primary_result,
                "reason": None}
    evidence = {
        "status": "PASS", "analysis_type": ANALYSIS_TYPE, "mode": n["mode"],
        "measures": n["measures"], "total_rows": n["total_rows"],
        "source_analysis_type": "specification_driven", "source_result": primary_result,
        "reason": None,
    }
    if n["mode"] == "grouped":
        evidence["group_column"] = n["group_column"]
        evidence["group_columns"] = n["group_columns"]
        evidence["group_count"] = len(n["rows"])
        evidence["group_results"] = {r["label"]: {"values": r["values"], "group_values": r["group_values"]}
                                     for r in n["rows"]}
    else:
        evidence["values"] = n["rows"][0]["values"]
    return evidence


def build_aggregate_evidence_mapping(insight, evidence, task_analysis=None):
    n = extract_specification_aggregate(
        (evidence or {}).get("source_result") if isinstance(evidence, dict) else None)
    validation = validate_aggregate_insight(insight, n, task_analysis)
    passed = bool(validation.get("pass"))
    checks = [{"check": k, "status": "PASS" if v else "FAIL"} for k, v in validation["checks"].items()]
    return {
        "status": "PASS" if passed else "FAIL", "analysis_type": ANALYSIS_TYPE,
        "claim_validation": {
            "status": "PASS" if passed else "FAIL", "total_claims": len(checks),
            "passed_claims": sum(c["status"] == "PASS" for c in checks),
            "failed_claims": sum(c["status"] == "FAIL" for c in checks), "checks": checks},
        "evidence_validation": {"status": (evidence or {}).get("status", "FAIL")},
        "reason": None if passed else validation.get("reason"),
        "evidence_sources": [{"source_id": "primary_result.operation_results.aggregations",
                              "source_type": "independently_validated_specification_aggregate"}],
        "traceability_status": "PASS" if passed else "REVIEW",
    }


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------

def generate_aggregate_recommendation(evidence, task_analysis=None):
    if not isinstance(evidence, dict) or str(evidence.get("status", "")).upper() != "PASS":
        return {"status": "FAIL", "reason": "Evidence validation did not pass."}
    n = extract_specification_aggregate(evidence.get("source_result"))
    if n is None:
        return {"status": "FAIL", "reason": "Aggregate evidence is incomplete."}

    if n["mode"] == "correlation":
        p = max(n["pairs"], key=lambda q: abs(q["value"]))
        strength = _strength(p["value"])
        if strength == "negligible":
            text = (f"{p['x']} and {p['y']} show no meaningful linear relationship "
                    f"({p['method']} r = {p['value']:.3f}). Do not use one to predict the other "
                    "on the basis of this result; check for non-linear patterns or relationships "
                    "within segments before concluding there is no association.")
        else:
            text = (f"{p['x']} and {p['y']} show a {strength} {_direction(p['value'])} linear "
                    f"relationship ({p['method']} r = {p['value']:.3f}). Check whether it holds "
                    "within relevant segments and is not driven by outliers before acting. "
                    "Correlation does not establish causation.")
        return {"status": "PASS", "recommendation": text, "basis": {
            "analysis_type": ANALYSIS_TYPE, "value": float(p["value"])}}

    m = n["measures"][0]
    if n["mode"] == "ungrouped":
        v = n["rows"][0]["values"][m["key"]]
        if m["metric"] is None:
            text = (f"The dataset has {fmt('count', v)} records. Use this as the base for rates and "
                    "shares, and check that the rows represent the population you intend to analyse.")
        elif m["op"] in {"min", "max"}:
            text = (f"The {_measure_phrase(m)} is {fmt(m['op'], v)}. An extreme value can be a data-entry "
                    "error or a legitimate rare case, so verify it against the source record before "
                    "acting on it, and compare it with the typical (median) value.")
        else:
            text = (f"Use the {_measure_phrase(m)} of {fmt(m['op'], v)} as a descriptive baseline. "
                    "A single figure can hide skew and outliers, so compare it with the median and "
                    "relevant subgroups before drawing conclusions.")
        return {"status": "PASS", "recommendation": text, "basis": {
            "analysis_type": ANALYSIS_TYPE, "metric": m["metric"], "value": float(v)}}

    sentences = []
    for measure in n["measures"]:
        ranked = sorted(n["rows"], key=lambda r: (-r["values"][measure["key"]], r["label"].casefold()))
        top, bottom = ranked[0], ranked[-1]
        tv, bv = top["values"][measure["key"]], bottom["values"][measure["key"]]
        if len(ranked) == 1 or tv == bv:
            sentences.append(f"All groups share the same {_measure_phrase(measure)}, {fmt(measure['op'], tv)}.")
        else:
            sentences.append(
                f"For {_measure_phrase(measure)} by {n['group_column']}, the highest value is "
                f"{fmt(measure['op'], tv)} for {top['label']}, and the lowest is "
                f"{fmt(measure['op'], bv)} for {bottom['label']}.")
    ordered = sorted(n["rows"], key=lambda r: (-r["values"][m["key"]], r["label"].casefold()))
    hi, lo = ordered[0], ordered[-1]
    hv, lv = hi["values"][m["key"]], lo["values"][m["key"]]
    body = " ".join(sentences)
    text = (f"{body} Review the number of records in each group and investigate whether the "
            "differences are meaningful before making decisions. These descriptive values do not "
            "establish causation.")
    return {"status": "PASS", "recommendation": text, "basis": {
        "analysis_type": ANALYSIS_TYPE, "group_column": n["group_column"],
        "top_category": hi["label"], "value": float(hv),
        "bottom_category": lo["label"], "bottom_value": float(lv)}}
