"""
Grouped-count support ("top three job categories by customer count",
"which contact type has the most customers?", "customers per marital status").

The independently validated executor returns one row per group:
    {"job": "admin.", "row_count": 5209}
This module turns that validated result into

  * an insight that answers the question (top / bottom N, or the single winner),
  * a deterministic validator for that insight,
  * evidence and an insight-to-evidence mapping,
  * a recommendation whose numbers all come from the evidence.

Every number in every sentence is read from the validated result; nothing is
recomputed from the narrative. Ranking and truncation (top N) are applied here,
on the full validated counts, so the independently verified aggregation stays
unchanged.
"""
import math
import re

ANALYSIS_TYPE = "grouped_count_analysis"


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def _is_int_like(value):
    if isinstance(value, bool):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return math.isfinite(number) and number == int(number) and number >= 0


def extract_specification_grouped_count(primary_result):
    """
    Normalise a validated specification-driven COUNT-by-group result, or None.

    Accepted: status PASS, operations == ["count"] (a rank step is optional),
    no metric columns, one or more group columns, and rows carrying row_count.
    """
    if not isinstance(primary_result, dict):
        return None
    if str(primary_result.get("analysis_type", "")).lower() != "specification_driven":
        return None
    if str(primary_result.get("status", "")).upper() != "PASS":
        return None

    operations = [str(op).strip().lower() for op in (primary_result.get("operations") or [])]
    if "count" not in operations or any(op not in {"count", "rank"} for op in operations):
        return None
    if primary_result.get("metrics"):
        return None

    group_columns = [str(c).strip() for c in (primary_result.get("group_by") or []) if str(c).strip()]
    if not group_columns:
        return None

    results = primary_result.get("operation_results")
    rows = results.get("aggregations") if isinstance(results, dict) else None
    if not isinstance(rows, list) or not rows:
        return None

    group_results = {}
    for row in rows:
        if not isinstance(row, dict) or "row_count" not in row:
            return None
        if any(column not in row or row[column] is None for column in group_columns):
            return None
        if not _is_int_like(row["row_count"]):
            return None
        values = {column: row[column] for column in group_columns}
        label = (
            str(values[group_columns[0]]) if len(group_columns) == 1
            else ", ".join(f"{column}={values[column]}" for column in group_columns)
        )
        if label in group_results:
            return None
        group_results[label] = {
            "count_value": int(float(row["row_count"])),
            "group_values": {k: (v if isinstance(v, (int, float, str)) else str(v)) for k, v in values.items()},
        }

    total = sum(item["count_value"] for item in group_results.values())
    input_rows = primary_result.get("analyzed_rows", primary_result.get("input_rows"))
    total_rows = int(input_rows) if _is_int_like(input_rows) else total
    return {
        "group_columns": group_columns,
        "group_column": " and ".join(group_columns),
        "group_results": group_results,
        "group_count": len(group_results),
        "total_rows": total_rows,
        "counted_rows": total,
    }


def _focus(task_analysis):
    spec = task_analysis.get("analytical_spec") if isinstance(task_analysis, dict) else None
    focus = spec.get("answer_focus") if isinstance(spec, dict) else None
    if isinstance(focus, dict) and focus.get("aggregate") == "count":
        return focus
    return None


def _ranked(normalized, direction):
    items = [(label, d["count_value"]) for label, d in normalized["group_results"].items()]
    ascending = direction == "ascending"
    items.sort(key=lambda item: (item[1] if ascending else -item[1], item[0].casefold()))
    return items


def _share(count, total):
    return (count / total * 100.0) if total else 0.0


def displayed_groups(normalized, focus):
    """(direction, limit, list of (label, count)) that the insight must show."""
    direction = (focus or {}).get("direction") or "descending"
    limit = (focus or {}).get("limit")
    ranked = _ranked(normalized, direction)
    if isinstance(limit, int) and limit > 0:
        ranked = ranked[:limit]
    return direction, limit, ranked


# ---------------------------------------------------------------------------
# Insight
# ---------------------------------------------------------------------------

def generate_grouped_count_insight(normalized, task_analysis=None):
    if normalized is None:
        return "No valid grouped count result was available for insight generation."
    focus = _focus(task_analysis)
    direction, limit, shown = displayed_groups(normalized, focus)
    column = normalized["group_column"]
    groups = normalized["group_count"]
    total = normalized["total_rows"]
    listing = "; ".join(f"{label}: {count:,}" for label, count in shown)
    context = f"({groups} groups, {total:,} records in total)"

    if focus and focus.get("type") == "group_distribution":
        with_share = "; ".join(
            f"{label}: {count:,} ({_share(count, total):.2f}%)" for label, count in shown)
        return (
            f"Distribution of customers by {column} {context}: {with_share}. "
            "Counts and shares are descriptive and do not establish causation."
        )

    if focus and limit == 1 and shown:
        label, count = shown[0]
        word = "fewest" if direction == "ascending" else "most"
        return (
            f"The {column} with the {word} customers is {label}: {count:,} "
            f"of {total:,} records across {groups} groups. "
            "Counts are descriptive and do not establish causation."
        )
    if focus and isinstance(limit, int) and limit > 1:
        head = "Bottom" if direction == "ascending" else "Top"
        return (
            f"{head} {len(shown)} {column} groups by customer count {context}: "
            f"{listing}. Counts are descriptive and do not establish causation."
        )
    return (
        f"Customer count by {column} {context}: {listing}. "
        "Counts are descriptive and do not establish causation."
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_grouped_count_insight(insight, normalized, task_analysis=None):
    text = str(insight or "")
    checks = {
        "result_status": normalized is not None,
        "group_column_mentioned": False,
        "all_displayed_groups_paired_with_correct_count": False,
        "displayed_group_order_matches_ranking": False,
        "group_count_and_total_match_result": False,
    }
    if normalized is None:
        return {"overall": "FAIL", "pass": False, "checks": checks,
                "reason": "The grouped count result is invalid or incomplete."}

    focus = _focus(task_analysis)
    _, _, shown = displayed_groups(normalized, focus)
    lower = " ".join(text.casefold().replace("_", " ").split())
    checks["group_column_mentioned"] = all(
        " ".join(c.casefold().replace("_", " ").split()) in lower
        for c in normalized["group_columns"]
    )

    positions, pairs = [], []
    single_winner = bool(focus and focus.get("limit") == 1)
    is_distribution = bool(focus and focus.get("type") == "group_distribution")
    for label, count in shown:
        pattern = re.escape(str(label)) + r"\s*:\s*" + re.escape(f"{count:,}")
        if is_distribution:
            pattern += r"\s*\(" + re.escape(f"{_share(count, normalized['total_rows']):.2f}%") + r"\)"
        else:
            pattern += r"(?![\d,])"
        match = re.search(pattern, text, flags=re.IGNORECASE)
        pairs.append(match is not None)
        positions.append(match.start() if match else -1)
    checks["all_displayed_groups_paired_with_correct_count"] = bool(pairs) and all(pairs)
    checks["displayed_group_order_matches_ranking"] = (
        bool(positions) and all(p >= 0 for p in positions) and positions == sorted(positions)
    )
    checks["group_count_and_total_match_result"] = (
        f"{normalized['total_rows']:,}" in text
        and re.search(rf"(?<![\d,]){normalized['group_count']}\s+groups", text) is not None
    )
    # a single-winner sentence has the winner pair but must not name other groups' counts
    if single_winner:
        others = [l for l in normalized["group_results"] if l not in {s[0] for s in shown}]
        checks["no_non_winner_counts_claimed"] = not any(
            re.search(re.escape(l) + r"\s*:\s*\d", text, flags=re.IGNORECASE) for l in others
        )
    passed = all(checks.values())
    return {
        "overall": "PASS" if passed else "FAIL",
        "pass": passed,
        "analysis_type": ANALYSIS_TYPE,
        "checks": checks,
        "group_count": normalized["group_count"],
        "group_columns": normalized["group_columns"],
        "reason": None if passed else (
            "The insight must show the requested groups, in ranked order, each with "
            "its validated count, plus the group count and total records."
        ),
    }


# ---------------------------------------------------------------------------
# Evidence and mapping
# ---------------------------------------------------------------------------

def build_grouped_count_evidence(normalized, primary_result, task_analysis=None):
    if normalized is None:
        return {"status": "FAIL", "analysis_type": ANALYSIS_TYPE,
                "reason": "No validated grouped count result was available."}
    focus = _focus(task_analysis)
    direction, limit, shown = displayed_groups(normalized, focus)
    group_results = {
        label: dict(details, percent_value=_share(details["count_value"], normalized["total_rows"]))
        for label, details in normalized["group_results"].items()
    }
    evidence = {
        "status": "PASS",
        "analysis_type": ANALYSIS_TYPE,
        "group_column": normalized["group_column"],
        "group_columns": normalized["group_columns"],
        "group_results": group_results,
        "group_count": normalized["group_count"],
        "total_rows": normalized["total_rows"],
        "ranking_direction": direction,
        "displayed_groups": [label for label, _ in shown],
        "source_analysis_type": "specification_driven",
        "source_result": primary_result,
        "reason": None,
    }
    if isinstance(limit, int):
        evidence["requested_limit"] = limit
    # "result" is what the report prints under Supporting Evidence and what the
    # traceability step links to; without it both were empty.
    evidence["result"] = {
        "group_column": normalized["group_column"],
        "group_count": normalized["group_count"],
        "total_rows": normalized["total_rows"],
        "ranking_direction": direction,
        "displayed_groups": evidence["displayed_groups"],
        "group_results": group_results,
    }
    return evidence


def build_grouped_count_evidence_mapping(insight, evidence, task_analysis=None):
    normalized = extract_specification_grouped_count(
        (evidence or {}).get("source_result") if isinstance(evidence, dict) else None
    )
    validation = validate_grouped_count_insight(insight, normalized, task_analysis)
    passed = bool(validation.get("pass"))
    checks = [
        {"check": name, "status": "PASS" if ok else "FAIL"}
        for name, ok in validation["checks"].items()
    ]
    return {
        "status": "PASS" if passed else "FAIL",
        "analysis_type": ANALYSIS_TYPE,
        "group_column": (evidence or {}).get("group_column"),
        "group_columns": (evidence or {}).get("group_columns"),
        "claim_validation": {
            "status": "PASS" if passed else "FAIL",
            "total_claims": len(checks),
            "passed_claims": sum(c["status"] == "PASS" for c in checks),
            "failed_claims": sum(c["status"] == "FAIL" for c in checks),
            "checks": checks,
        },
        "evidence_validation": {"status": (evidence or {}).get("status", "FAIL")},
        "reason": None if passed else validation.get("reason"),
        "evidence_sources": [{
            "source_id": "primary_result.operation_results.aggregations",
            "source_type": "independently_validated_specification_grouped_count",
            "group_column": (evidence or {}).get("group_column"),
            "aggregation": "count",
        }],
        "traceability_status": "PASS" if passed else "REVIEW",
    }


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------

def generate_grouped_count_recommendation(evidence, task_analysis=None):
    if not isinstance(evidence, dict) or str(evidence.get("status", "")).upper() != "PASS":
        return {"status": "FAIL", "reason": "Evidence validation did not pass."}
    normalized = extract_specification_grouped_count(evidence.get("source_result"))
    if normalized is None:
        return {"status": "FAIL", "reason": "Grouped count evidence is incomplete."}

    column = normalized["group_column"]
    ranked = _ranked(normalized, "descending")
    (top_label, top_count), (low_label, low_count) = ranked[0], ranked[-1]
    total = normalized["total_rows"]
    shares = bool((_focus(task_analysis) or {}).get("type") == "group_distribution")
    top_share = f" ({_share(top_count, total):.2f}%)" if shares else ""
    low_share = f" ({_share(low_count, total):.2f}%)" if shares else ""
    if len(ranked) == 1:
        body = (f"{top_label} is the only {column} group, with {top_count:,} customers.")
    elif top_count == low_count:
        body = f"All {column} groups have the same customer count, {top_count:,}."
    else:
        body = (
            f"The largest {column} group is {top_label} with {top_count:,} customers{top_share}, "
            f"and the smallest is {low_label} with {low_count:,} customers{low_share}."
        )
    text = (
        f"{body} Group size alone does not show performance: compare an outcome such as "
        "conversion within each group before prioritising segments, and treat very small "
        "groups cautiously. These counts are descriptive and do not establish causation."
    )
    return {
        "status": "PASS",
        "recommendation": text,
        "basis": {
            "analysis_type": ANALYSIS_TYPE,
            "group_column": column,
            "top_category": top_label,
            "value": float(top_count),
            "bottom_category": low_label,
            "bottom_value": float(low_count),
        },
    }
