"""
Recommendation generator.

Builds deterministic, evidence-grounded recommendations from validated EDA
evidence. Every number that appears in a recommendation is read from the
evidence (never recomputed from a narrative or invented), so that
recommendation_validator.py can ground each claim independently.
"""
import math


# ============================================================
# SHARED HELPERS
# ============================================================

# Containers that different pipeline versions use to wrap validated results.
_CONTAINER_KEYS = (
    "primary_result",
    "result",
    "supporting_result",
    "evidence",
    "validated_result",
    "source_result",
)

_GROUPED_ANALYSIS_TYPES = {
    "grouped_analysis",
    "grouped_mean_analysis",
    "specification_driven_grouped_mean",
}

# Wording used for each aggregation in the recommendation text.
_OPERATION_WORDS = {
    "mean": "average",
    "average": "average",
    "avg": "average",
    "sum": "total",
    "total": "total",
    "count": "count",
    "median": "median",
    "min": "minimum",
    "max": "maximum",
}


def format_number(value):
    try:
        return f"{float(value):,.2f}"
    except (ValueError, TypeError):
        return str(value)


def _fail(reason, **extra):
    return {"status": "FAIL", "reason": reason, **extra}


def _is_pass(evidence):
    """Case-insensitive PASS check, used consistently by every generator."""
    return (
        isinstance(evidence, dict)
        and str(evidence.get("status", "")).strip().upper() == "PASS"
    )


def _finite_number(value):
    """Return a finite float, or None. Booleans are not numbers here."""
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _collect_sources(evidence, max_depth=5):
    """Collect the evidence dict and its known nested result containers."""
    sources = []

    def walk(obj, depth):
        if depth > max_depth or not isinstance(obj, dict):
            return
        if any(obj is seen for seen in sources):   # identity, not equality
            return
        sources.append(obj)
        for key in _CONTAINER_KEYS:
            child = obj.get(key)
            if isinstance(child, dict):
                walk(child, depth + 1)

    walk(evidence, 0)
    return sources


def _first_present(sources, *keys):
    for source in sources:
        for key in keys:
            value = source.get(key)
            if value is not None and value != "":
                return value
    return None


def _resolve_operation(sources, default="mean"):
    """Find the aggregation name: 'operation', 'aggregation' or a single 'operations' entry."""
    operation = _first_present(sources, "operation", "aggregation")
    if operation is None:
        operations = _first_present(sources, "operations")
        if isinstance(operations, (list, tuple)) and len(operations) == 1:
            operation = operations[0]
    return str(operation or default).strip().lower()


# ============================================================
# 1. GROUPED ANALYSIS RECOMMENDATION
# ============================================================

def generate_grouped_recommendation(evidence):
    """
    Generate a recommendation from validated grouped evidence.

    Supports the current group_results schema (single or composite groups)
    and the legacy top_category/top_value schema.

    For group_results evidence the text names BOTH the highest and the lowest
    group together with their values, in the form
    "the highest average is 4.08 for contact=telephone, loan=yes", which is
    exactly what recommendation_validator.py checks for.
    """
    if not isinstance(evidence, dict):
        return _fail("Evidence must be a dictionary.")
    if not _is_pass(evidence):
        return _fail("Evidence validation did not pass.")

    sources = _collect_sources(evidence)

    metric = _first_present(sources, "metric", "metric_column", "measure_column") or "requested metric"
    metric_label = str(metric).strip()

    group_columns = _first_present(sources, "group_columns")
    group_column = _first_present(sources, "group_column", "category")
    if isinstance(group_columns, (list, tuple)) and group_columns:
        group_description = " and ".join(str(col) for col in group_columns)
    elif group_column:
        group_description = str(group_column)
    else:
        group_description = "the requested groups"

    operation = _resolve_operation(sources)
    measure = _OPERATION_WORDS.get(operation, operation)

    groups = _first_present(sources, "group_results")

    # --- Current schema: rank directly from validated group values ----------
    ranked_groups = []          # (label, value)
    group_counts = []           # record counts, only if present for every group
    if isinstance(groups, dict) and groups:
        for label, details in groups.items():
            count = None
            if isinstance(details, dict):
                raw_value = details.get(
                    "mean_value",
                    details.get("mean", details.get("average", details.get("value"))),
                )
                count = _finite_number(
                    details.get("count", details.get("n", details.get("row_count")))
                )
            else:
                raw_value = details
            numeric_value = _finite_number(raw_value)
            if numeric_value is None:
                return _fail(f"Group '{label}' has no finite validated value.")
            ranked_groups.append((str(label), numeric_value))
            if count is not None:
                group_counts.append(int(count))

        if len(group_counts) != len(ranked_groups):
            group_counts = []   # partial counts would be misleading

        ranked_groups.sort(key=lambda item: item[1], reverse=True)
        top_category, top_value = ranked_groups[0]
        bottom_category, bottom_value = ranked_groups[-1]
        second_category, second_value = (
            ranked_groups[1] if len(ranked_groups) > 1 else (None, None)
        )
    else:
        # --- Legacy schema ---------------------------------------------------
        top_category = _first_present(sources, "top_category")
        top_value = _first_present(sources, "top_value")
        second_category = _first_present(sources, "second_category")
        second_value = _first_present(sources, "second_value")
        bottom_category, bottom_value = None, None

    if top_category is None:
        return _fail("Top category is missing from validated grouped evidence.")

    top_numeric = _finite_number(top_value)
    if top_numeric is None:
        return _fail("Top value is missing or non-numeric in validated grouped evidence.")

    # --- Build the text ------------------------------------------------------
    lead = f"For {metric_label} grouped by {group_description}, "

    if ranked_groups:
        bottom_numeric = float(bottom_value)
        if len(ranked_groups) == 1:
            body = (
                f"the only group is {top_category}, with an {measure} of "
                f"{format_number(top_numeric)}."
                if measure == "average"
                else f"the only group is {top_category}, with a {measure} of "
                     f"{format_number(top_numeric)}."
            )
        elif top_numeric == bottom_numeric:
            body = f"all groups have the same {measure}, {format_number(top_numeric)}."
        else:
            body = (
                f"the highest {measure} is {format_number(top_numeric)} for "
                f"{top_category}, and the lowest {measure} is "
                f"{format_number(bottom_numeric)} for {bottom_category}."
            )
    else:
        body = f"the highest {measure} is {format_number(top_numeric)} for {top_category}."
        second_numeric = _finite_number(second_value)
        if second_category is not None and second_numeric is not None:
            body += (
                f" {second_category} is next-highest at "
                f"{format_number(second_numeric)}."
            )

    recommendation = lead + body

    if group_counts:
        recommendation += (
            f" Group sizes range from {min(group_counts):,} to "
            f"{max(group_counts):,} records. Investigate whether the "
            f"differences are meaningful given these group sizes before "
            f"making decisions."
        )
    else:
        recommendation += (
            " Review the number of records in each group and investigate "
            "whether the differences are meaningful before making decisions."
        )

    noun = "averages" if measure == "average" else "values"
    recommendation += f" These descriptive {noun} do not establish causation."

    basis = {
        "analysis_type": str(
            evidence.get("analysis_type")
            or ("grouped_mean_analysis" if ranked_groups else "grouped_analysis")
        ),
        "group_column": group_description,
        "metric": metric_label,
        "operation": operation,
        "top_category": top_category,
        "value": float(top_numeric),
        "group_count": len(ranked_groups) if ranked_groups else None,
        "ranking_source": (
            "validated_group_results" if ranked_groups
            else "legacy_validated_top_category"
        ),
    }
    if ranked_groups:
        basis["bottom_category"] = bottom_category
        basis["bottom_value"] = float(bottom_value)

    return {"status": "PASS", "recommendation": recommendation, "basis": basis}


# ============================================================
# 2. MEAN ANALYSIS RECOMMENDATION
# ============================================================

def _mean_from_aggregations(sources):
    """
    Look for '<metric>__mean' values in operation_results.aggregations.

    aggregations may be a list of row dicts (current schema) or a single dict
    (older schema). Returns (metric, value, hit_count).
    """
    for source in sources:
        operation_results = source.get("operation_results")
        aggregations = (
            operation_results.get("aggregations")
            if isinstance(operation_results, dict) else None
        )
        if isinstance(aggregations, dict):
            rows = [aggregations]
        elif isinstance(aggregations, list):
            rows = aggregations
        else:
            continue

        hits = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            for key, val in row.items():
                if str(key).lower().endswith("__mean"):
                    hits.append((str(key).rsplit("__", 1)[0], val))
        if hits:
            metric, value = hits[0]
            return metric, value, len(hits)
    return None, None, 0


def generate_mean_recommendation(evidence):
    """
    Generate a recommendation from validated mean evidence.

    Supports the original metric/value schema, filtered results such as
    analysis_type=filtered_mean, filter_column=month, filter_value=aug,
    metric_column=duration, mean_duration=232.40, and specification-driven
    aggregations of the form {"campaign__mean": 2.76}.
    """
    if not isinstance(evidence, dict):
        return _fail("Evidence must be a dictionary.")
    if not _is_pass(evidence):
        return _fail("Evidence validation did not pass.")

    sources = _collect_sources(evidence)

    metric = _first_present(sources, "metric", "metric_column", "column")
    value = _first_present(sources, "value", "mean", "mean_duration", "average", "average_value")

    if value is None:
        agg_metric, agg_value, hit_count = _mean_from_aggregations(sources)
        if hit_count > 1:
            return _fail(
                "Evidence contains several mean values (grouped result); "
                "use the grouped recommendation instead of a single mean."
            )
        if hit_count == 1:
            value = agg_value
            if metric is None:
                metric = agg_metric

    filter_column = _first_present(sources, "filter_column")
    filter_value = _first_present(sources, "filter_value")
    filtered_rows = _first_present(sources, "filtered_rows", "row_count")
    valid_count = _first_present(sources, "valid_duration_count", "valid_count", "count")

    if value is None:
        return _fail(
            "Mean value is missing. Expected one of: value, mean, mean_duration, "
            "average, average_value, or a '<metric>__mean' aggregation.",
            available_keys=sorted({key for source in sources for key in source}),
        )

    numeric_value = _finite_number(value)
    if numeric_value is None:
        return _fail(f"Mean value is not a finite number: {value!r}.")

    if metric is None:
        metric = "measure"

    scope = ""
    if filter_column is not None and filter_value is not None:
        scope = f" for {filter_column}={filter_value}"

    sample_detail = ""
    if filtered_rows is not None:
        sample_detail += f" The filtered dataset contains {filtered_rows} rows."
    if valid_count is not None:
        sample_detail += f" The mean uses {valid_count} valid observations."

    # Do not assume a domain (e.g. "months"): only mention comparisons that
    # the evidence actually supports.
    if filter_column is not None:
        follow_up = f"compare other values of {filter_column}"
    else:
        follow_up = "compare relevant subgroups"

    recommendation = (
        f"Use the validated average {str(metric).lower()}{scope} "
        f"of approximately {format_number(numeric_value)} as a descriptive "
        f"baseline for further analysis.{sample_detail} Investigate unusually "
        f"low or high observations and {follow_up} before drawing "
        f"conclusions. This average alone does not establish causation."
    )

    basis = {
        "analysis_type": evidence.get("analysis_type", "mean_analysis"),
        "metric": metric,
        "value": numeric_value,
    }
    if filter_column is not None:
        basis["filter_column"] = filter_column
    if filter_value is not None:
        basis["filter_value"] = filter_value
    if filtered_rows is not None:
        basis["filtered_rows"] = filtered_rows
    if valid_count is not None:
        basis["valid_count"] = valid_count

    return {"status": "PASS", "recommendation": recommendation, "basis": basis}


# ============================================================
# 3. CORRELATION / RELATIONSHIP RECOMMENDATION
# ============================================================

def generate_correlation_recommendation(evidence):
    """
    Generate a recommendation from correlation evidence.

    Expected evidence structure:
        {
            "status": "PASS",
            "analysis_type": "correlation_analysis",
            "x_variable": "Units Sold",
            "y_variable": "Total Profit",
            "method": "Pearson",
            "correlation": 0.5645,
            "direction": "positive",
            "strength": "moderate"
        }
    """
    if not isinstance(evidence, dict):
        return _fail("Evidence must be a dictionary.")
    if not _is_pass(evidence):
        return _fail("Evidence validation did not pass.")

    x_variable = evidence.get("x_variable")
    y_variable = evidence.get("y_variable")
    method = evidence.get("method", "Pearson")
    correlation = evidence.get("correlation")
    direction = evidence.get("direction")
    strength = evidence.get("strength")

    if x_variable is None:
        return _fail("X variable is missing from correlation evidence.")
    if y_variable is None:
        return _fail("Y variable is missing from correlation evidence.")
    if correlation is None:
        return _fail("Correlation value is missing from evidence.")
    if direction is None:
        return _fail("Correlation direction is missing from evidence.")
    if strength is None:
        return _fail("Correlation strength is missing from evidence.")

    # Previously float(correlation) could raise ValueError for bad input.
    correlation_number = _finite_number(correlation)
    if correlation_number is None:
        return _fail(f"Correlation value is not a finite number: {correlation!r}.")
    if not -1.0 <= correlation_number <= 1.0:
        return _fail(f"Correlation value {correlation_number} is outside [-1, 1].")

    correlation_text = format_number(correlation_number)

    if direction == "positive":
        relationship_text = (
            f"{x_variable} and {y_variable} show a {strength} positive relationship"
        )
    elif direction == "negative":
        relationship_text = (
            f"{x_variable} and {y_variable} show a {strength} negative relationship"
        )
    else:
        relationship_text = (
            f"{x_variable} and {y_variable} show little or no linear relationship"
        )

    # No dataset-specific wording (the old text mentioned regions, product
    # categories and sales channels regardless of the data).
    recommendation = (
        f"{relationship_text}, with a {method} correlation of approximately "
        f"{correlation_text}. Investigate whether this relationship remains "
        f"consistent across relevant segments before making business "
        f"decisions. Correlation indicates association and should not be "
        f"interpreted as proof of causation."
    )

    return {
        "status": "PASS",
        "recommendation": recommendation,
        "basis": {
            "analysis_type": evidence.get("analysis_type", "correlation_analysis"),
            "x_variable": x_variable,
            "y_variable": y_variable,
            "method": method,
            "correlation": correlation_number,
            "direction": direction,
            "strength": strength,
        },
    }


# ============================================================
# 4. MAIN RECOMMENDATION GENERATOR
# ============================================================

def generate_recommendation(task_type, evidence):
    """
    Generate a recommendation based on the validated evidence.

    Parameters
    ----------
    task_type : str
        Type of analysis performed.
    evidence : dict
        Evidence produced by the evidence mapping stage.

    Returns
    -------
    dict
        Structured recommendation.
    """
    task_type = "generic" if task_type is None else str(task_type).lower()

    # Guard first: the routing below calls evidence.get(...).
    if not isinstance(evidence, dict):
        return _fail("Evidence must be a dictionary.")

    analysis_type = str(evidence.get("analysis_type", "")).lower()

    if task_type == "comparison" or analysis_type in _GROUPED_ANALYSIS_TYPES:
        return generate_grouped_recommendation(evidence)

    if task_type == "relationship_analysis" or analysis_type == "correlation_analysis":
        return generate_correlation_recommendation(evidence)

    if "mean" in task_type or "average" in task_type:
        return generate_mean_recommendation(evidence)

    if not _is_pass(evidence):
        return _fail("Evidence validation did not pass.")

    return {
        "status": "PASS",
        "recommendation": (
            "Use the validated EDA evidence as the basis for further "
            "investigation and decision-making. Additional analysis should be "
            "performed before taking business-critical action."
        ),
        "basis": {"analysis_type": task_type},
    }


# ============================================================
# MODULE TEST
# ============================================================

def _bank_marketing_evidence():
    rows = [
        ("cellular", "no", 2.669675311415152),
        ("cellular", "yes", 2.794375),
        ("telephone", "no", 3.28869286287089),
        ("telephone", "yes", 4.080097087378641),
        ("unknown", "no", 2.8078813250819077),
        ("unknown", "yes", 2.677165354330709),
    ]
    return {
        "analysis_type": "grouped_mean_analysis",
        "status": "PASS",
        "metric_column": "campaign",
        "group_column": "contact and loan",
        "group_columns": ["contact", "loan"],
        "group_results": {
            f"contact={c}, loan={l}": {
                "mean_value": m,
                "group_values": {"contact": c, "loan": l},
            }
            for c, l, m in rows
        },
    }


if __name__ == "__main__":
    line = "=" * 70
    print(line)
    print("RECOMMENDATION GENERATOR TEST")
    print(line)

    def show(title, result):
        print(f"\n{title}")
        print(result)

    show("TEST 1 - GROUPED (legacy schema)", generate_recommendation(
        "comparison",
        {
            "status": "PASS",
            "analysis_type": "grouped_analysis",
            "category": "Region",
            "metric": "Total Profit",
            "operation": "SUM",
            "evidence": {
                "top_category": "Sub-Saharan Africa",
                "top_value": 12183211.40,
                "second_category": "Europe",
                "second_value": 11082938.63,
            },
        },
    ))

    show("TEST 2 - INVALID EVIDENCE", generate_recommendation(
        "comparison", {"status": "FAIL", "reason": "Invalid analysis result."}
    ))

    show("TEST 3 - MEAN", generate_recommendation(
        "mean",
        {"status": "PASS", "analysis_type": "mean_analysis",
         "metric": "Total Profit", "value": 7500000.50},
    ))

    show("TEST 4 - CORRELATION", generate_recommendation(
        "relationship_analysis",
        {"status": "PASS", "analysis_type": "correlation_analysis",
         "x_variable": "Units Sold", "y_variable": "Total Profit",
         "method": "Pearson", "correlation": 0.5645504620845976,
         "direction": "positive", "strength": "moderate"},
    ))

    show("TEST 5 - INCOMPLETE CORRELATION EVIDENCE", generate_recommendation(
        "relationship_analysis",
        {"status": "PASS", "analysis_type": "correlation_analysis",
         "x_variable": "Units Sold", "y_variable": "Total Profit"},
    ))

    show("TEST 6 - GROUPED (group_results, composite groups)",
         generate_recommendation("comparison", _bank_marketing_evidence()))

    show("TEST 7 - MEAN from list-style aggregations", generate_recommendation(
        "mean",
        {"status": "PASS", "operation_results": {"aggregations": [{"campaign__mean": 2.76}]}},
    ))

    show("TEST 8 - NON-DICT EVIDENCE", generate_recommendation("mean", None))

    show("TEST 9 - NON-NUMERIC CORRELATION", generate_recommendation(
        "relationship_analysis",
        {"status": "PASS", "analysis_type": "correlation_analysis",
         "x_variable": "a", "y_variable": "b", "correlation": "n/a",
         "direction": "positive", "strength": "weak"},
    ))

    print("\n" + line)
    print("RECOMMENDATION GENERATOR TEST COMPLETED")
    print(line)
