"""
Aggregate support for specification-driven results that the grouped-mean and
grouped-count paths do not cover:

  * ungrouped aggregates   "average balance", "maximum balance", "how many customers",
                           "standard deviation of duration", distributions of a numeric column
  * grouped non-mean       "median duration by contact", "total balance per education"
  * rates                  "conversion rate", "conversion rate by education",
                           "which month has the highest conversion rate", "how many subscribed"
  * ranked groups          "top 5 jobs by average balance", "rank education by average balance"
  * data quality           "how many missing values", "are there duplicate rows"
  * correlations           "is there a correlation between age and balance",
                           including yes/no questions ("is the relationship strong and positive?")

The executor's validated result is the only source of numbers. This module writes
the insight, validates it deterministically, builds the evidence and the
insight-to-evidence mapping, and writes a recommendation whose numbers all come
from the evidence.
"""
import math
import re

ANALYSIS_TYPE = "specification_aggregate_analysis"

SUPPORTED_OPS = ("mean", "median", "sum", "min", "max", "std", "variance", "count", "nunique")
RATE_METRICS = {"conversion"}          # 0/1 outcome columns: their mean is a rate
_OP_PHRASE = {
    "mean": "average", "median": "median", "sum": "total", "min": "minimum",
    "max": "maximum", "std": "standard deviation", "variance": "variance",
    "count": "count of non-missing values", "nunique": "number of distinct values",
}
_FILTER_WORDS = {">": "above", ">=": "at least", "<": "below", "<=": "at most",
                 "==": "equal to", "!=": "not equal to"}
_STRENGTH_RANK = {"negligible": 0, "weak": 1, "moderate": 2, "strong": 3, "very strong": 4}


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


def _is_rate(metric):
    return bool(metric) and (metric in RATE_METRICS or str(metric).endswith("_rate"))


def fmt(op, value):
    """Display format: whole numbers with separators for counts, 2 decimals otherwise."""
    return f"{int(round(value)):,}" if _is_count_op(op) else f"{value:,.2f}"


def fm(m, value):
    """Format a value of measure `m`: a mean of a 0/1 outcome is shown as a percentage."""
    if m["metric"] and _is_rate(m["metric"]) and m["op"] == "mean":
        return f"{value * 100:,.2f}%"
    if m["metric"] and _is_rate(m["metric"]) and m["op"] == "sum":
        return f"{int(round(value)):,}"
    return fmt(m["op"], value)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def _rows_info(primary_result):
    analyzed = primary_result.get("analyzed_rows", primary_result.get("input_rows"))
    total_rows = int(analyzed) if _finite(analyzed) is not None else None
    input_rows = primary_result.get("input_rows")
    return total_rows, int(input_rows) if _finite(input_rows) is not None else None


def _extract_data_quality(primary_result, operation):
    results = primary_result.get("operation_results")
    if not isinstance(results, dict):
        return None
    total_rows, input_rows = _rows_info(primary_result)
    if total_rows is None:
        return None
    if operation == "missing_values":
        block = results.get("missing_values")
        if not isinstance(block, dict) or _finite(block.get("total_missing_cells")) is None:
            return None
        by_column = {str(k): int(v) for k, v in (block.get("by_column") or {}).items() if _finite(v) is not None}
        return {"mode": "missing", "total_missing": int(block["total_missing_cells"]),
                "by_column": by_column, "column_count": len(by_column), "total_rows": total_rows,
                "input_rows": input_rows, "filters": [], "measures": [], "rows": [],
                "group_columns": [], "group_column": ""}
    value = _finite(results.get("duplicate_count"))
    if value is None:
        return None
    return {"mode": "duplicates", "duplicates": int(value), "total_rows": total_rows,
            "duplicate_percent": (int(value) / total_rows * 100.0) if total_rows else 0.0,
            "input_rows": input_rows, "filters": [], "measures": [], "rows": [],
            "group_columns": [], "group_column": ""}


def extract_specification_aggregate(primary_result, focus=None):
    """Normalise a validated ungrouped / grouped-non-mean / rate / ranked aggregate, else None."""
    if not isinstance(primary_result, dict):
        return None
    if str(primary_result.get("analysis_type", "")).lower() != "specification_driven":
        return None
    if str(primary_result.get("status", "")).upper() != "PASS":
        return None

    operations = [str(op).strip().lower() for op in (primary_result.get("operations") or [])]
    if operations == ["correlation"]:
        return _extract_correlation(primary_result)
    if operations in (["missing_values"], ["duplicate_count"]):
        return _extract_data_quality(primary_result, operations[0])
    focus_type = (focus or {}).get("type") if isinstance(focus, dict) else None
    if focus_type == "group_ranking":
        operations = [op for op in operations if op != "rank"]
    if not operations or any(op not in SUPPORTED_OPS for op in operations):
        return None                           # rank/outliers/time trend/... use other paths
    metrics = [str(m).strip() for m in (primary_result.get("metrics") or []) if str(m).strip()]
    groups = [str(g).strip() for g in (primary_result.get("group_by") or []) if str(g).strip()]

    # Paths that already exist and are validated end to end.
    if groups and operations == ["mean"] and len(metrics) < 2:
        takes_over = focus_type == "group_ranking" or any(_is_rate(m) for m in metrics)
        if not takes_over:
            return None                       # single-metric grouped mean (proven path)
    if groups and not metrics and operations == ["count"]:
        return None                           # grouped count -> grouped_count_support

    results = primary_result.get("operation_results")
    aggregations = results.get("aggregations") if isinstance(results, dict) else None
    if aggregations is None:
        return None

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

    total_rows, input_rows = _rows_info(primary_result)
    filters = [f for f in (primary_result.get("filters_applied") or []) if isinstance(f, dict)]
    return {
        "mode": "grouped" if groups else "ungrouped",
        "group_columns": groups, "group_column": " and ".join(groups),
        "measures": measures, "rows": rows,
        "total_rows": total_rows, "input_rows": input_rows, "filters": filters,
    }


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------

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
                pairs.append({"x": x, "y": y, "method": str(method).capitalize(), "value": value,
                              "r_squared": value * value, "shared_variance_percent": value * value * 100.0})
    total_rows, _ = _rows_info(primary_result)
    return {
        "mode": "correlation", "pairs": pairs, "metrics": metrics, "total_rows": total_rows,
        "filters": [f for f in (primary_result.get("filters_applied") or []) if isinstance(f, dict)],
        "measures": [], "rows": [], "group_columns": [], "group_column": "", "input_rows": None,
    }


def _pair_sentence(p):
    r = p["value"]
    strength = _strength(r)
    shared = f"r\u00b2 = {p['r_squared']:.3f}, about {p['shared_variance_percent']:.2f}% shared variance"
    if strength == "negligible":
        return (f"The {p['method']} correlation between {p['x']} and {p['y']} is {r:.3f}, "
                f"a negligible linear relationship ({shared}).")
    return (f"The {p['method']} correlation between {p['x']} and {p['y']} is {r:.3f}, "
            f"a {strength} {_direction(r)} linear relationship ({shared}).")


def _asked_relationship(question):
    """('strong'|'moderate'|None, 'positive'|'negative'|None) from a yes/no style question."""
    q = str(question or "").casefold()
    if not re.search(r"\b(?:whether|is there|are there|do|does|is|are|determine|check)\b", q):
        return None, None
    strength = "strong" if re.search(r"\bstrong\b", q) else "moderate" if re.search(r"\bmoderate\b", q) else None
    direction = "positive" if re.search(r"\bpositive\b", q) else "negative" if re.search(r"\bnegative\b", q) else None
    return strength, direction


def _assessment(question, pair):
    """Direct answer to "do X and Y have a strong positive relationship?" or None."""
    asked_strength, asked_direction = _asked_relationship(question)
    if asked_strength is None and asked_direction is None:
        return None
    r = pair["value"]
    actual_strength, actual_direction = _strength(r), _direction(r)
    strength_ok = asked_strength is None or _STRENGTH_RANK[actual_strength] >= _STRENGTH_RANK[asked_strength]
    direction_ok = asked_direction is None or actual_direction == asked_direction
    asked = " ".join(w for w in (asked_strength, asked_direction) if w)
    if strength_ok and direction_ok:
        return f"Answer: yes. The data show a {asked} relationship."
    return (f"Answer: no. The data do not show a {asked} relationship; the observed relationship "
            f"is {actual_strength}{'' if actual_strength == 'negligible' else ' ' + actual_direction}.")


def _question(task_analysis):
    spec = task_analysis.get("analytical_spec") if isinstance(task_analysis, dict) else None
    return (spec or {}).get("original_question") if isinstance(spec, dict) else None


# ---------------------------------------------------------------------------
# Wording helpers
# ---------------------------------------------------------------------------

def _focus(task_analysis):
    spec = task_analysis.get("analytical_spec") if isinstance(task_analysis, dict) else None
    focus = spec.get("answer_focus") if isinstance(spec, dict) else None
    if isinstance(focus, dict) and focus.get("type") in {"group_extrema", "group_ranking"}:
        return focus
    return None


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
    if _is_rate(m["metric"]) and m["op"] == "mean":
        return f"{m['metric']} rate"
    if _is_rate(m["metric"]) and m["op"] == "sum":
        return f"number of {m['metric']}s"
    if m["op"] == "nunique":
        return f"number of distinct {m['metric']} values"
    return f"{_OP_PHRASE[m['op']]} {m['metric']}"


def _scope_text(n):
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


def _ordered(n, m, focus):
    descending = not (focus and focus.get("direction") == "ascending")
    return sorted(n["rows"], key=lambda r: ((-1 if descending else 1) * r["values"][m["key"]], r["label"].casefold()))


def _shown(n, m, focus):
    """Rows the insight must show for measure m, in order."""
    ordered = _ordered(n, m, focus)
    if focus and focus.get("type") == "group_ranking":
        limit = focus.get("limit")
        return ordered[:limit] if isinstance(limit, int) and limit > 0 else ordered
    return sorted(n["rows"], key=lambda r: (-r["values"][m["key"]], r["label"].casefold()))


# ---------------------------------------------------------------------------
# Insight
# ---------------------------------------------------------------------------

def generate_aggregate_insight(n, task_analysis=None):
    if n is None:
        return "No valid aggregate result was available for insight generation."

    if n["mode"] == "missing":
        total, rows, cols = n["total_missing"], n["total_rows"], n["column_count"]
        if total == 0:
            return (f"There are no missing values: all {cols} columns are complete across {rows:,} rows. "
                    "Placeholder text such as 'unknown' is not counted as missing.")
        affected = sorted(((c, v) for c, v in n["by_column"].items() if v > 0), key=lambda x: (-x[1], x[0]))
        listing = "; ".join(f"{c}: {v:,}" for c, v in affected)
        return (f"There are {total:,} missing cells across {rows:,} rows; {len(affected)} of {cols} columns "
                f"have missing values: {listing}.")
    if n["mode"] == "duplicates":
        d, rows = n["duplicates"], n["total_rows"]
        if d == 0:
            return f"There are no duplicate rows among the {rows:,} rows."
        return f"There are {d:,} duplicate rows ({n['duplicate_percent']:.2f}% of {rows:,} rows)."

    if n["mode"] == "correlation":
        ft = _filter_text(n["filters"])
        basis = (f" Based on {n['total_rows']:,} rows" if n["total_rows"] is not None else " Based on the analysed rows")
        basis += f" where {ft}." if ft else "."
        text = " ".join(_pair_sentence(p) for p in n["pairs"]) + basis
        answer = _assessment(_question(task_analysis), max(n["pairs"], key=lambda q: abs(q["value"])))
        if answer:
            text += " " + answer
        return text + " Correlation measures linear association only and does not establish causation."

    scope = _scope_text(n)
    if n["mode"] == "ungrouped":
        values = n["rows"][0]["values"]
        metrics_in = {m["metric"] for m in n["measures"]}
        if len(n["measures"]) > 1 and len(metrics_in) == 1 and None not in metrics_in:
            # Several statistics of one column: one sentence, one scope.
            parts = ", ".join(f"{_OP_PHRASE[m['op']]} {fm(m, values[m['key']])}" for m in n["measures"])
            return (f"Distribution of {next(iter(metrics_in))}: {parts}{scope}. "
                    "These are descriptive figures and do not establish causation.")
        sentences = []
        for m in n["measures"]:
            v = values[m["key"]]
            if m["metric"] is None:
                sentences.append(f"There are {fmt('count', v)} records in the dataset{scope}.")
            else:
                sentences.append(f"The {_measure_phrase(m)} is {fm(m, v)}{scope}.")
        return " ".join(sentences) + " This is a descriptive figure and does not establish causation."

    lines = []
    focus_any = _focus(task_analysis)
    for m in n["measures"]:
        shown = _shown(n, m, focus_any)
        listing = "; ".join(f"{r['label']}: {fm(m, r['values'][m['key']])}" for r in shown)
        context = [f"{len(n['rows'])} groups"]
        if n["total_rows"] is not None:
            rows_text = f"{n['total_rows']:,} rows analysed"
            if _filter_text(n["filters"]) and n["input_rows"] not in (None, n["total_rows"]):
                rows_text += f" of {n['input_rows']:,}"
            context.append(rows_text)
        if _filter_text(n["filters"]):
            context.append(f"rows where {_filter_text(n['filters'])}")
        if focus_any and focus_any.get("type") == "group_ranking":
            limit = focus_any.get("limit")
            if isinstance(limit, int) and limit > 0:
                head = f"{'Bottom' if focus_any.get('direction') == 'ascending' else 'Top'} {len(shown)} {n['group_column']} groups by {_measure_phrase(m)}"
            else:
                head = (f"{_measure_phrase(m).capitalize()} by {n['group_column']}, ranked "
                        f"{'lowest' if focus_any.get('direction') == 'ascending' else 'highest'} first")
            line = f"{head} ({', '.join(context)}): {listing}."
        else:
            line = f"{_measure_phrase(m).capitalize()} by {n['group_column']} ({', '.join(context)}): {listing}."
        if focus_any and focus_any.get("type") == "group_extrema" and len(n["rows"]) >= 2:
            ordered = _shown(n, m, None)
            hi, lo = ordered[0], ordered[-1]
            hi_t = f"{hi['label']} ({fm(m, hi['values'][m['key']])})"
            lo_t = f"{lo['label']} ({fm(m, lo['values'][m['key']])})"
            if focus_any.get("direction") == "ascending":
                line += f" Lowest {_measure_phrase(m)}: {lo_t}; highest: {hi_t}."
            else:
                line += f" Highest {_measure_phrase(m)}: {hi_t}; lowest: {lo_t}."
        lines.append(line)
    return " ".join(lines) + " These are descriptive values and do not establish causation."


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _has_value(text, shown):
    return re.search(r"(?<![\d.,])" + re.escape(shown) + r"(?!\d|,\d|\.\d)", text) is not None


def validate_aggregate_insight(insight, n, task_analysis=None):
    text = str(insight or "")
    lower = " ".join(text.casefold().replace("_", " ").split())
    if n is None:
        return {"overall": "FAIL", "pass": False,
                "checks": {"result_status": False},
                "reason": "The aggregate result is invalid or incomplete."}

    def finish(checks, reason):
        passed = all(checks.values())
        return {"overall": "PASS" if passed else "FAIL", "pass": passed,
                "analysis_type": ANALYSIS_TYPE, "checks": checks, "reason": None if passed else reason}

    if n["mode"] == "missing":
        ok = [_has_value(text, f"{n['total_rows']:,}")]
        if n["total_missing"] == 0:
            ok += [f"all {n['column_count']} columns" in lower, "no missing values" in lower]
        else:
            ok += [_has_value(text, f"{n['total_missing']:,}")]
            ok += [re.search(re.escape(c) + r"\s*:\s*" + re.escape(f"{v:,}"), text) is not None
                   for c, v in n["by_column"].items() if v > 0]
        return finish({"result_status": True, "missing_value_counts_match_validated_result": all(ok)},
                      "The insight must report the validated missing-value counts exactly.")
    if n["mode"] == "duplicates":
        ok = [_has_value(text, f"{n['total_rows']:,}")]
        ok += [("no duplicate rows" in lower)] if n["duplicates"] == 0 else [
            _has_value(text, f"{n['duplicates']:,}"), _has_value(text, f"{n['duplicate_percent']:.2f}%")]
        return finish({"result_status": True, "duplicate_counts_match_validated_result": all(ok)},
                      "The insight must report the validated duplicate count exactly.")

    if n["mode"] == "correlation":
        checks = {"result_status": True, "variables_mentioned": True,
                  "every_correlation_matches_validated_result": True,
                  "strength_and_direction_match": True, "shared_variance_matches": True,
                  "question_answered_consistently": True}
        for p in n["pairs"]:
            checks["variables_mentioned"] &= p["x"].casefold() in lower and p["y"].casefold() in lower
            checks["every_correlation_matches_validated_result"] &= _has_value(text, f"{p['value']:.3f}")
            expected = _strength(p["value"])
            checks["strength_and_direction_match"] &= (
                expected in lower and (expected == "negligible" or _direction(p["value"]) in lower))
            checks["shared_variance_matches"] &= (
                f"r\u00b2 = {p['r_squared']:.3f}" in text and _has_value(text, f"{p['shared_variance_percent']:.2f}%"))
        answer = _assessment(_question(task_analysis), max(n["pairs"], key=lambda q: abs(q["value"])))
        checks["question_answered_consistently"] = (answer in text) if answer else ("Answer:" not in text)
        return finish(checks, "The insight must name both variables and report each validated coefficient, "
                              "strength, direction, shared variance and a consistent yes/no answer.")

    checks = {"result_status": True, "metric_mentioned": False, "operation_mentioned": False,
              "every_value_matches_validated_result": False}
    metrics = [m["metric"] for m in n["measures"] if m["metric"]]
    checks["metric_mentioned"] = all(
        " ".join(m.casefold().replace("_", " ").split()) in lower for m in metrics) if metrics else True

    def operation_ok(m):
        if not m["metric"]:
            return "record" in lower
        if _is_rate(m["metric"]) and m["op"] == "mean":
            return f"{m['metric']} rate" in lower
        if _is_rate(m["metric"]) and m["op"] == "sum":
            return f"number of {m['metric']}s" in lower
        if m["op"] == "nunique":
            return re.search(r"\bdistinct\b", lower) is not None
        return re.search(rf"\b{re.escape(_OP_PHRASE[m['op']])}\b", lower) is not None

    checks["operation_mentioned"] = all(operation_ok(m) for m in n["measures"])

    if n["mode"] == "ungrouped":
        values = n["rows"][0]["values"]
        ok = [_has_value(text, fm(m, values[m["key"]])) for m in n["measures"]]
        checks["every_value_matches_validated_result"] = bool(ok) and all(ok)
        return finish(checks, "The insight must name the metric and operation and report every validated value exactly.")

    checks["all_group_columns_mentioned"] = all(
        " ".join(g.casefold().replace("_", " ").split()) in lower for g in n["group_columns"])
    focus = _focus(task_analysis)
    pair_ok, order_ok, hidden_ok = [], [], []
    for m in n["measures"]:
        shown = _shown(n, m, focus)
        positions = []
        for r in shown:
            pattern = re.escape(r["label"]) + r"\s*:\s*" + re.escape(fm(m, r["values"][m["key"]])) + r"(?!\d|,\d|\.\d)"
            found = re.search(pattern, text, flags=re.IGNORECASE)
            pair_ok.append(found is not None)
            positions.append(found.start() if found else -1)
        if focus and focus.get("type") == "group_ranking":
            order_ok.append(all(p >= 0 for p in positions) and positions == sorted(positions))
            shown_labels = {r["label"] for r in shown}
            hidden_ok.append(not any(
                re.search(re.escape(r["label"]) + r"\s*:\s*[\d-]", text, flags=re.IGNORECASE)
                for r in n["rows"] if r["label"] not in shown_labels))
    checks["every_value_matches_validated_result"] = bool(pair_ok) and all(pair_ok)
    if order_ok:
        checks["ranking_order_matches_validated_values"] = all(order_ok)
        checks["only_requested_groups_are_shown"] = all(hidden_ok)
    if focus and focus.get("type") == "group_extrema":
        for m in n["measures"]:
            ordered = _shown(n, m, None)
            winner = ordered[-1] if focus.get("direction") == "ascending" else ordered[0]
            checks["winner_" + m["key"] + "_reported"] = re.search(
                re.escape(winner["label"]) + r"\s*\(" + re.escape(fm(m, winner["values"][m["key"]])) + r"\)",
                text, flags=re.IGNORECASE) is not None
    return finish(checks, "The insight must name the metric and operation and report every validated value exactly.")


# ---------------------------------------------------------------------------
# Evidence / mapping
# ---------------------------------------------------------------------------

def _with_percent(n, values):
    """Add percentage forms of rate means so displayed percentages are grounded in evidence."""
    out = dict(values)
    for m in n["measures"]:
        if m["metric"] and _is_rate(m["metric"]) and m["op"] == "mean" and m["key"] in values:
            out[m["key"] + "__percent"] = values[m["key"]] * 100.0
    return out


def build_aggregate_evidence(n, primary_result, task_analysis=None):
    if n is None:
        return {"status": "FAIL", "analysis_type": ANALYSIS_TYPE,
                "reason": "No validated aggregate result was available."}
    base = {"status": "PASS", "analysis_type": ANALYSIS_TYPE, "mode": n["mode"],
            "total_rows": n["total_rows"], "source_analysis_type": "specification_driven",
            "source_result": primary_result, "reason": None}
    if n["mode"] == "correlation":
        base.update({"pairs": n["pairs"], "result": {"pairs": n["pairs"], "total_rows": n["total_rows"]}})
        return base
    if n["mode"] == "missing":
        base["missing_cells"] = n["total_missing"]
        base["column_count"] = n["column_count"]
        base["by_column"] = n["by_column"]
        base["result"] = {k: base[k] for k in ("missing_cells", "column_count", "by_column", "total_rows")}
        return base
    if n["mode"] == "duplicates":
        base["duplicate_rows"] = n["duplicates"]
        base["duplicate_percent"] = n["duplicate_percent"]
        base["result"] = {k: base[k] for k in ("duplicate_rows", "duplicate_percent", "total_rows")}
        return base
    base["measures"] = n["measures"]
    if n["mode"] == "grouped":
        base["group_column"] = n["group_column"]
        base["group_columns"] = n["group_columns"]
        base["group_count"] = len(n["rows"])
        base["group_results"] = {r["label"]: {"values": _with_percent(n, r["values"]), "group_values": r["group_values"]}
                                 for r in n["rows"]}
    else:
        base["values"] = _with_percent(n, n["rows"][0]["values"])
    base["result"] = {k: base[k] for k in ("mode", "measures", "total_rows", "group_column", "group_count",
                                           "group_results", "values") if k in base}
    return base


def build_aggregate_evidence_mapping(insight, evidence, task_analysis=None):
    n = extract_specification_aggregate(
        (evidence or {}).get("source_result") if isinstance(evidence, dict) else None, _focus_dict(task_analysis))
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
        "evidence_sources": [{"source_id": "primary_result.operation_results",
                              "source_type": "independently_validated_specification_aggregate"}],
        "traceability_status": "PASS" if passed else "REVIEW",
    }


def _focus_dict(task_analysis):
    spec = task_analysis.get("analytical_spec") if isinstance(task_analysis, dict) else None
    focus = spec.get("answer_focus") if isinstance(spec, dict) else None
    return focus if isinstance(focus, dict) else None


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------

def generate_aggregate_recommendation(evidence, task_analysis=None):
    if not isinstance(evidence, dict) or str(evidence.get("status", "")).upper() != "PASS":
        return {"status": "FAIL", "reason": "Evidence validation did not pass."}
    n = extract_specification_aggregate(evidence.get("source_result"), _focus_dict(task_analysis))
    if n is None:
        return {"status": "FAIL", "reason": "Aggregate evidence is incomplete."}
    ok = lambda text, basis=None: {"status": "PASS", "recommendation": text,  # noqa: E731
                                   "basis": basis or {"analysis_type": ANALYSIS_TYPE}}

    if n["mode"] == "missing":
        if n["total_missing"] == 0:
            return ok("No imputation or row removal is needed for missing values. Placeholder values "
                      "(for example the text 'unknown' or a negative sentinel code) are not counted as missing, "
                      "so check categorical and sentinel values separately before modelling.")
        return ok(f"Decide how to treat the {n['total_missing']:,} missing cells before modelling: impute, "
                  "exclude, or model them explicitly, and check whether the gaps are concentrated in particular "
                  "columns or groups.")
    if n["mode"] == "duplicates":
        if n["duplicates"] == 0:
            return ok("No duplicate rows were found, so no de-duplication is needed. Records that differ only in "
                      "an identifier can still be the same customer; check any key columns if that matters.")
        return ok(f"Review the {n['duplicates']:,} duplicate rows before removing them: confirm they are true "
                  "repeats and not legitimate repeated events, and keep the first occurrence if they are.")

    if n["mode"] == "correlation":
        p = max(n["pairs"], key=lambda q: abs(q["value"]))
        strength = _strength(p["value"])
        if strength == "negligible":
            text = (f"{p['x']} and {p['y']} show no meaningful linear relationship "
                    f"({p['method']} r = {p['value']:.3f}). With many rows even a tiny correlation can be "
                    "statistically significant but is not practically useful, so do not use one to predict "
                    "the other; check for non-linear patterns or relationships within segments before "
                    "concluding there is no association.")
        else:
            text = (f"{p['x']} and {p['y']} show a {strength} {_direction(p['value'])} linear "
                    f"relationship ({p['method']} r = {p['value']:.3f}). Check whether it holds "
                    "within relevant segments and is not driven by outliers before acting. "
                    "Correlation does not establish causation.")
        return ok(text, {"analysis_type": ANALYSIS_TYPE, "value": float(p["value"])})

    m = n["measures"][0]
    if n["mode"] == "ungrouped":
        v = n["rows"][0]["values"][m["key"]]
        if m["metric"] is None:
            text = (f"The dataset has {fmt('count', v)} records. Use this as the base for rates and "
                    "shares, and check that the rows represent the population you intend to analyse.")
        elif _is_rate(m["metric"]) and m["op"] == "mean":
            text = (f"The overall {_measure_phrase(m)} is {fm(m, v)}. Use it as the baseline when comparing segments: "
                    "a segment is only interesting if its rate differs from this baseline by more than its sample "
                    "size can explain.")
        elif _is_rate(m["metric"]) and m["op"] == "sum":
            text = (f"{fm(m, v)} customers converted. Pair this count with the total number of customers to "
                    "judge the rate, and compare segments before drawing conclusions.")
        elif m["op"] in {"min", "max"}:
            text = (f"The {_measure_phrase(m)} is {fm(m, v)}. An extreme value can be a data-entry "
                    "error or a legitimate rare case, so verify it against the source record before "
                    "acting on it, and compare it with the typical (median) value.")
        elif len(n["measures"]) > 1:
            text = ("Read these figures together: a mean far from the median, or a large standard deviation, "
                    "signals a skewed distribution with a long tail. Use the median for a typical value and "
                    "check outliers before relying on the mean.")
        else:
            text = (f"Use the {_measure_phrase(m)} of {fm(m, v)} as a descriptive baseline. "
                    "A single figure can hide skew and outliers, so compare it with the median and "
                    "relevant subgroups before drawing conclusions.")
        return ok(text, {"analysis_type": ANALYSIS_TYPE, "metric": m["metric"], "value": float(v)})

    focus = _focus(task_analysis)
    sentences = []
    for measure in n["measures"]:
        ranked = _shown(n, measure, None)
        top, bottom = ranked[0], ranked[-1]
        tv, bv = top["values"][measure["key"]], bottom["values"][measure["key"]]
        if len(ranked) == 1 or tv == bv:
            sentences.append(f"All groups share the same {_measure_phrase(measure)}, {fm(measure, tv)}.")
        else:
            sentences.append(
                f"For {_measure_phrase(measure)} by {n['group_column']}, the highest value is "
                f"{fm(measure, tv)} for {top['label']}, and the lowest is "
                f"{fm(measure, bv)} for {bottom['label']}.")
    ordered = _shown(n, m, None)
    hi, lo = ordered[0], ordered[-1]
    hv, lv = hi["values"][m["key"]], lo["values"][m["key"]]
    caution = ("Rates from small groups are unstable: check the number of records behind each group before acting."
               if _is_rate(m["metric"]) else
               "Review the number of records in each group and investigate whether the differences are meaningful before making decisions.")
    text = " ".join(sentences) + f" {caution} These descriptive values do not establish causation."
    return ok(text, {"analysis_type": ANALYSIS_TYPE, "group_column": n["group_column"],
                     "top_category": hi["label"], "value": float(hv),
                     "bottom_category": lo["label"], "bottom_value": float(lv)})
