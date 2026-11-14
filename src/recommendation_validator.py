"""
Independent validation of generated recommendations against validated evidence.

This module is deliberately conservative:
- FAIL: the recommendation generator/evidence explicitly reports failure, or the
  recommendation contradicts the evidence.
- PASS: every numeric claim can be matched to evidence or a deterministic
  arithmetic derivation from the evidence, and (for grouped means) the highest
  and lowest groups are named together with the correct values.
- REVIEW: a claim cannot be independently grounded with the supplied evidence.
  Every REVIEW result carries at least one human-readable reason.

It does not use an LLM and never invents replacement values.
"""
import math
import re


# Numbers such as 4.08, 12,183,211.40, 45211, -3.2, 5%. A number glued to a
# word/underscore/dot (e.g. "q4", "v2.1") is not a claim.
_NUMBER_RE = re.compile(
    r"(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)%?"
)

_STATUS_SYNONYMS = {
    "PASSED": "PASS", "OK": "PASS", "SUCCESS": "PASS",
    "FAILED": "FAIL", "ERROR": "FAIL",
    "N/A": "NOT_APPLICABLE", "NA": "NOT_APPLICABLE",
}

# Distance (in characters) within which the other components of a composite
# group label (e.g. "loan=yes" next to "contact=telephone") must appear.
_LABEL_WINDOW = 60


# ============================================================
# NUMERIC HELPERS
# ============================================================

def _finite_float(value):
    if isinstance(value, bool):
        return None
    try:
        number = float(str(value).replace(",", "").replace("%", "").strip())
        return number if math.isfinite(number) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _walk_numbers(value, path="evidence", depth=0):
    """Collect finite numeric values and their paths from nested evidence."""
    if depth > 12:
        return []
    found = []
    if isinstance(value, bool) or value is None:
        return found
    if isinstance(value, dict):
        for key, child in value.items():
            found.extend(_walk_numbers(child, f"{path}.{key}", depth + 1))
    elif isinstance(value, (list, tuple, set)):
        for index, child in enumerate(value):
            found.extend(_walk_numbers(child, f"{path}[{index}]", depth + 1))
    else:
        number = _finite_float(value)
        if number is not None:
            found.append((number, path))
    return found


def _decimals(raw):
    """Number of decimal places the claim was written with ('4.08' -> 2)."""
    cleaned = str(raw).replace(",", "").replace("%", "").lstrip("+-")
    return len(cleaned.split(".")[1]) if "." in cleaned else 0


def _claim_matches(raw, value, actual):
    """
    Rounding-aware comparison.

    A claim written with d decimals matches `actual` only if it is within half
    a unit of its last written digit. The previous fixed 0.02 tolerance let
    wrong claims such as 2.69 or 4.09 be "verified" against 2.677 / 4.080.
    """
    tolerance = 0.5 * 10 ** -_decimals(raw) + 1e-9
    if abs(value - actual) <= tolerance:
        return True
    if str(raw).endswith("%"):            # "5%" may be evidence 0.05
        return abs(value / 100.0 - actual) <= tolerance / 100.0
    return False


def _close(left, right):
    """Tolerance for comparing a structured basis value to evidence."""
    return abs(left - right) <= max(0.005, abs(right) * 1e-6)


def _recommendation_text(recommendation):
    if isinstance(recommendation, dict):
        return str(recommendation.get("recommendation", "") or "")
    return str(recommendation or "")


def _derived_differences(numbers):
    """Return pairwise differences for numeric evidence, bounded for safety."""
    values = sorted({round(item[0], 10) for item in numbers})
    if len(values) > 250:
        return []
    return [abs(values[i] - values[j])
            for i in range(len(values))
            for j in range(i + 1, len(values))]


def _normalize_status(value):
    status = str(value or "").strip().upper()
    return _STATUS_SYNONYMS.get(status, status)


# ============================================================
# GROUPED-MEAN BASIS (derived from the validated primary result)
# ============================================================

def _derive_grouped_mean_basis(primary_result):
    """Build a deterministic recommendation basis from a PASS grouped-mean result."""
    if not isinstance(primary_result, dict):
        return None
    if str(primary_result.get("status", "")).upper() != "PASS":
        return None
    if str(primary_result.get("analysis_type", "")).lower() != "specification_driven":
        return None

    operations = primary_result.get("operations", [])
    if not isinstance(operations, (list, tuple)):
        return None
    if "mean" not in [str(item).lower() for item in operations]:
        return None

    metrics = primary_result.get("metrics", [])
    group_columns = primary_result.get("group_by", [])
    operation_results = primary_result.get("operation_results", {})
    rows = operation_results.get("aggregations") if isinstance(operation_results, dict) else None

    if not isinstance(metrics, (list, tuple)) or len(metrics) != 1:
        return None
    if not isinstance(group_columns, (list, tuple)) or not group_columns:
        return None
    if not isinstance(rows, list) or not rows:
        return None

    metric = str(metrics[0])
    value_key = f"{metric}__mean"
    normalized_rows = []

    for row in rows:
        if not isinstance(row, dict) or value_key not in row:
            return None
        value = _finite_float(row[value_key])
        if value is None:
            return None

        group_values = {}
        for column in group_columns:
            if column not in row:
                return None
            group_values[str(column)] = str(row[column])

        label = ", ".join(f"{column}={group_values[str(column)]}" for column in group_columns)
        normalized_rows.append({"label": label, "group_values": group_values, "mean": value})

    highest = max(normalized_rows, key=lambda item: item["mean"])
    lowest = min(normalized_rows, key=lambda item: item["mean"])

    return {
        "metric": metric,
        "analysis_type": "specification_driven",
        "operation": "mean",
        "group_columns": [str(column) for column in group_columns],
        "top_group": highest["label"],
        "top_group_values": highest["group_values"],
        "top_mean": highest["mean"],
        "bottom_group": lowest["label"],
        "bottom_group_values": lowest["group_values"],
        "bottom_mean": lowest["mean"],
        "group_count": len(normalized_rows),
        "groups": normalized_rows,
    }


def _label_spans(text, group_values):
    """
    Find every place where ALL 'column=value' parts of a composite group occur
    together (within _LABEL_WINDOW characters). Returns (start, end) spans.

    Matching 'column=value' with word boundaries replaces the old bare
    substring test, under which a value like 'no' was "found" inside 'not'.
    """
    if not group_values:
        return []
    if len(group_values) == 1:
        # Single grouping column: the text may say "loan=no" or just "no"
        # (e.g. "... is 2.69 for no"). A bare value is only accepted here, and
        # the value-pairing check still ties it to its own number.
        (col, val), = group_values.items()
        patterns = [re.compile(
            rf"(?<!\w)(?:{re.escape(str(col))}\s*=\s*)?{re.escape(str(val))}(?!\w)",
            re.IGNORECASE)]
    else:
        patterns = [
            re.compile(rf"(?<!\w){re.escape(str(col))}\s*=\s*{re.escape(str(val))}(?!\w)",
                       re.IGNORECASE)
            for col, val in group_values.items()
        ]
    spans = []
    for first in patterns[0].finditer(text):
        start, end = first.start(), first.end()
        ok = True
        for pattern in patterns[1:]:
            near = [m for m in pattern.finditer(text)
                    if m.start() >= first.start() - _LABEL_WINDOW
                    and m.end() <= first.end() + _LABEL_WINDOW]
            if not near:
                ok = False
                break
            best = min(near, key=lambda m: abs(m.start() - first.start()))
            start, end = min(start, best.start()), max(end, best.end())
        if ok:
            spans.append((start, end))
    return spans


def _text_mentions_group(text, group_values):
    """Require every 'column=value' component of a composite group in the text."""
    return bool(_label_spans(str(text or ""), group_values))


def _text_pairs_group_with_mean(text, group_values, mean):
    """
    True if, for some mention of the group, the NEAREST number in the text
    (outside the label itself) equals the group's validated mean. This catches
    swapped values, e.g. 'highest is 2.67 for <the 4.08 group>'.
    """
    text = str(text or "")
    numbers = [(m.start(), m.end(), m.group(0)) for m in _NUMBER_RE.finditer(text)]
    for span_start, span_end in _label_spans(text, group_values):
        candidates = []
        for n_start, n_end, raw in numbers:
            if n_start < span_end and n_end > span_start:      # inside the label
                continue
            distance = max(n_start - span_end, span_start - n_end, 0)
            candidates.append((distance, raw))
        if not candidates:
            continue
        _, nearest_raw = min(candidates, key=lambda item: item[0])
        nearest_value = _finite_float(nearest_raw)
        if nearest_value is not None and _claim_matches(nearest_raw, nearest_value, mean):
            return True
    return False


def _text_mentions_value(text, mean):
    for match in _NUMBER_RE.finditer(str(text or "")):
        raw = match.group(0)
        value = _finite_float(raw)
        if value is not None and _claim_matches(raw, value, mean):
            return True
    return False


def _check(name, ok, reason_if_not, **extra):
    item = {"name": name, "status": "PASS" if ok else "REVIEW", **extra}
    if not ok:
        item["reason"] = reason_if_not
    return item


def _validate_derived_grouped_mean_basis(text, basis):
    """Check the recommendation states the metric and the validated extrema."""
    checks = []
    metric = str(basis["metric"])
    checks.append(_check(
        "basis_metric_mentioned",
        metric.casefold() in text.casefold(),
        f"Metric '{metric}' is not mentioned.",
    ))

    top, bottom = basis["top_mean"], basis["bottom_mean"]

    # All groups tie: the generator states one shared value without naming groups.
    if basis["group_count"] > 1 and top == bottom:
        checks.append(_check(
            "tied_group_value_mentioned",
            _text_mentions_value(text, top),
            "All groups share one mean but the recommendation does not state it.",
        ))
        return checks

    for role, group, values, mean in (
        ("highest", basis["top_group"], basis["top_group_values"], top),
        ("lowest", basis["bottom_group"], basis["bottom_group_values"], bottom),
    ):
        checks.append(_check(
            f"{role}_average_group_mentioned",
            _text_mentions_group(text, values),
            f"The {role}-average group '{group}' is not named as column=value pairs.",
            expected_group=group,
        ))
        checks.append(_check(
            f"{role}_average_value_matches_group",
            _text_pairs_group_with_mean(text, values, mean),
            f"The number stated next to '{group}' does not match its validated mean.",
            expected_group=group,
        ))
    return checks


# ============================================================
# STRUCTURED BASIS (generator-supplied) CHECKS
# ============================================================

def _check_generator_basis(basis, evidence, derived_basis):
    """Compare a generator-supplied basis with the evidence. Returns mismatches."""
    mismatches = []
    containers = [c for c in (evidence, evidence.get("evidence"),
                              evidence.get("supporting_result")) if isinstance(c, dict)]

    for key in ("metric", "analysis_type", "filter_column", "filter_value",
                "top_category", "group_column"):
        claimed = basis.get(key)
        if claimed is None:
            continue
        actual_values = [c[key] for c in containers if key in c]
        if actual_values and not any(
            str(claimed).strip().casefold() == str(actual).strip().casefold()
            for actual in actual_values
        ):
            mismatches.append({"field": key, "claimed": claimed, "evidence_values": actual_values})

    # Numeric basis values must match SOME number in the evidence (the old
    # check silently skipped when no specially-named key existed).
    evidence_values = [number for number, _ in _walk_numbers(evidence)]
    for key in ("value", "mean", "correlation", "bottom_value"):
        claimed = _finite_float(basis.get(key))
        if claimed is None:
            continue
        if evidence_values and not any(_close(claimed, actual) for actual in evidence_values):
            mismatches.append({"field": key, "claimed": claimed,
                               "evidence_values": evidence_values[:20]})

    # Cross-check the generator's extrema against the validated primary result.
    if derived_basis is not None:
        for key, expected in (("top_category", derived_basis["top_group"]),
                              ("bottom_category", derived_basis["bottom_group"])):
            claimed = basis.get(key)
            if claimed is not None and str(claimed).strip().casefold() != expected.casefold():
                mismatches.append({"field": key, "claimed": claimed,
                                   "evidence_values": [expected]})
        for key, expected in (("value", derived_basis["top_mean"]),
                              ("bottom_value", derived_basis["bottom_mean"])):
            claimed = _finite_float(basis.get(key))
            if claimed is not None and not _close(claimed, expected):
                mismatches.append({"field": key, "claimed": claimed,
                                   "evidence_values": [expected]})
    return mismatches


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def _early_fail(check_name, reason):
    return {
        "status": "FAIL", "overall": "FAIL", "pass": False,
        "total_numeric_claims": 0, "verified_numeric_claims": 0,
        "unverified_numeric_claims": 0,
        "checks": [{"name": check_name, "status": "FAIL", "reason": reason}],
        "reasons": [reason],
    }


def validate_recommendation(
    recommendation,
    evidence,
    analysis_type=None,
    primary_result=None,
    claim_evidence_validation=None,
):
    """
    Independently check recommendation presence, evidence status, structured
    basis fields, and numeric claims in recommendation text.
    """
    text = _recommendation_text(recommendation)
    checks = []
    reasons = []

    if not text.strip():
        result = _early_fail("recommendation_presence", "Recommendation is empty.")
        result["analysis_type"] = analysis_type
        return result

    if isinstance(recommendation, dict):
        generator_status = str(recommendation.get("status", "")).upper()
        if generator_status in {"FAIL", "FAILED", "ERROR", "UNSUPPORTED"}:
            result = _early_fail("generator_status",
                                 str(recommendation.get("reason", generator_status)))
            result["analysis_type"] = analysis_type
            return result

    if not isinstance(evidence, dict) or str(evidence.get("status", "")).upper() != "PASS":
        result = _early_fail("evidence_status",
                             "Recommendation evidence is missing or not PASS.")
        result["analysis_type"] = analysis_type
        return result

    # ---- numeric claim grounding -------------------------------------------
    evidence_numbers = _walk_numbers(evidence)
    derived_differences = _derived_differences(evidence_numbers)

    numeric_claims = []
    for match in _NUMBER_RE.finditer(text):
        raw = match.group(0)
        number = _finite_float(raw)
        if number is not None:
            numeric_claims.append({"text": raw, "value": number})

    verified, unverified = [], []
    for claim in numeric_claims:
        direct = next(
            (path for actual, path in evidence_numbers
             if _claim_matches(claim["text"], claim["value"], actual)),
            None,
        )
        derived = any(_claim_matches(claim["text"], claim["value"], actual)
                      for actual in derived_differences)
        if direct:
            verified.append({**claim, "matched_evidence_path": direct})
        elif derived:
            verified.append({**claim, "matched_evidence_path": "deterministic_pairwise_difference"})
        else:
            unverified.append(claim)

    checks.append({
        "name": "numeric_claim_grounding",
        "status": ("NOT_APPLICABLE" if not numeric_claims
                   else ("PASS" if not unverified else "REVIEW")),
        "total": len(numeric_claims),
        "verified": len(verified),
        "unverified": len(unverified),
        "verified_claims": verified,
        "unverified_claims": unverified,
    })
    if unverified:
        reasons.append(
            "Numerical statements could not be independently grounded: "
            + ", ".join(item["text"] for item in unverified) + "."
        )

    # ---- structured basis ----------------------------------------------------
    # The generator's basis (when supplied) is checked against the evidence AND
    # against the primary result; for grouped means the text itself is also
    # checked against the primary result, whether or not a basis was supplied.
    generator_basis = recommendation.get("basis") if isinstance(recommendation, dict) else None
    if not isinstance(generator_basis, dict) or not generator_basis:
        generator_basis = None

    derived_basis = _derive_grouped_mean_basis(primary_result)

    basis_mismatches = []
    basis_checks = []
    basis_source = None

    if generator_basis is not None:
        basis_source = "recommendation"
        basis_mismatches = _check_generator_basis(generator_basis, evidence, derived_basis)

    if derived_basis is not None:
        basis_source = "validated_primary_result"
        basis_checks = _validate_derived_grouped_mean_basis(text, derived_basis)

    if basis_mismatches:
        basis_status = "FAIL"
        basis_reason = "Structured recommendation basis contradicts available evidence."
        reasons.append(basis_reason)
    elif basis_checks:
        failing = [item for item in basis_checks if item["status"] != "PASS"]
        basis_status = "REVIEW" if failing else "PASS"
        basis_reason = (
            "Recommendation basis was derived from the PASS primary result "
            "and its metric and composite group references were checked."
        )
        for item in failing:
            reasons.append(item.get("reason", f"Basis check '{item['name']}' needs review."))
    elif generator_basis is not None:
        basis_status = "PASS"
        basis_reason = "Structured recommendation basis is consistent with available evidence."
    else:
        # A plain-text recommendation without a structured basis is not
        # downgraded: its numeric claims are already grounded above, and a
        # missing basis is not evidence of a problem.
        basis_status = "NOT_APPLICABLE"
        basis_reason = (
            "No structured recommendation basis was supplied and none could be "
            "derived from a validated primary result; numeric grounding applies."
        )

    checks.append({
        "name": "structured_basis_consistency",
        "status": basis_status,
        "mismatches": basis_mismatches,
        "reason": basis_reason,
        "basis_source": basis_source,
        "basis_checks": basis_checks,
    })

    # ---- upstream claim/evidence validation -----------------------------------
    # Consume the nested recommendation result rather than the combined
    # insight+recommendation status, so an insight REVIEW is not misreported
    # as a recommendation problem.
    if isinstance(claim_evidence_validation, dict):
        nested = claim_evidence_validation.get("recommendation")
        claim_result = nested if isinstance(nested, dict) else claim_evidence_validation

        claim_status = _normalize_status(claim_result.get("overall") or claim_result.get("status"))
        try:
            failed_claim_count = int(claim_result.get("failed_claims", 0) or 0)
        except (TypeError, ValueError, OverflowError):
            failed_claim_count = 1

        soft = claim_status in {"REVIEW", "NOT_APPLICABLE"}
        failed = (
            claim_status == "FAIL"
            or (claim_result.get("pass") is False and not soft)
            or (failed_claim_count > 0 and not soft)
        )

        item = {"name": "existing_claim_evidence_validation"}
        if failed:
            item["status"] = "FAIL"
            item["reason"] = "Existing claim/evidence grounding reported a failure."
            reasons.append(item["reason"])
        elif claim_status == "PASS":
            item["status"] = "PASS"
        elif claim_status == "NOT_APPLICABLE":
            item["status"] = "NOT_APPLICABLE"
        else:
            # Previously this produced REVIEW with an empty reasons list.
            upstream = claim_result.get("reason") or claim_result.get("reasons")
            if isinstance(upstream, (list, tuple)):
                upstream = "; ".join(str(r) for r in upstream)
            if claim_status == "REVIEW":
                detail = "Existing claim/evidence validation reported REVIEW"
            else:
                detail = ("Existing claim/evidence validation returned no "
                          f"recognizable status ({claim_status or 'missing'})")
            item["status"] = "REVIEW"
            item["reason"] = f"{detail}: {upstream}." if upstream else f"{detail}."
            reasons.append(item["reason"])
        checks.append(item)

    if not numeric_claims:
        checks.append({
            "name": "numeric_claim_presence",
            "status": "NOT_APPLICABLE",
            "reason": "No numerical claims were made; recommendation remains subject to evidence review.",
        })

    hard_fail = bool(basis_mismatches) or any(c["status"] == "FAIL" for c in checks)
    needs_review = any(c["status"] == "REVIEW" for c in checks)
    status = "FAIL" if hard_fail else ("REVIEW" if needs_review else "PASS")

    # Guarantee: a non-PASS status is never reported without an explanation.
    if status != "PASS" and not reasons:
        reasons = [c.get("reason") or f"Check '{c['name']}' reported {c['status']}."
                   for c in checks if c["status"] in {"FAIL", "REVIEW"}]

    return {
        "status": status,
        "overall": status,
        "pass": status == "PASS",
        "analysis_type": analysis_type,
        "total_numeric_claims": len(numeric_claims),
        "verified_numeric_claims": len(verified),
        "unverified_numeric_claims": len(unverified),
        "checks": checks,
        "reasons": reasons,
    }
