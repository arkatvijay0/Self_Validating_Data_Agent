
"""
Conservative numeric-claim grounding for the Self-Validating EDA Agent.

This module complements (does not replace) insight_validator.py and the
existing evidence mapper. It checks whether numeric tokens in generated
insights/recommendations can be found in the supplied evidence.

A PASS means only that all detected numeric tokens have evidence matches.
It does NOT prove that non-numeric causal, comparative, or semantic claims
are true. An unmatched number returns REVIEW rather than FAIL because a
legitimate derived statistic may not be stored verbatim in the evidence.
"""

from __future__ import annotations

import math
import re
from typing import Any


_NUMBER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)?(?:\.\d+)?"
    r"(?:[eE][-+]?\d+)?"
    r"%?"
    r"(?![A-Za-z0-9_])"
)


def _to_number(token: str):
    """Convert a matched numeric token to a float; percentages stay in points."""
    cleaned = token.replace(",", "").rstrip("%")
    if not cleaned or cleaned in {"+", "-"}:
        return None
    try:
        value = float(cleaned)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def extract_numeric_claims(text: Any) -> list[dict]:
    """Extract numeric tokens from text while preserving their original form."""
    if text is None:
        return []

    source = str(text)
    claims = []
    for match in _NUMBER_PATTERN.finditer(source):
        token = match.group(0)
        value = _to_number(token)
        if value is None:
            continue
        claims.append({
            "text": token,
            "value": value,
            "start": match.start(),
            "end": match.end(),
            "is_percentage": token.endswith("%"),
        })
    return claims


def _collect_evidence_numbers(value: Any, path: str = "evidence") -> list[dict]:
    """Collect finite numeric evidence values recursively, excluding booleans."""
    found = []

    if isinstance(value, bool) or value is None:
        return found

    if isinstance(value, (int, float)):
        numeric = float(value)
        if math.isfinite(numeric):
            found.append({"value": numeric, "path": path})
        return found

    # NumPy scalar support without requiring NumPy as a dependency.
    if hasattr(value, "item") and callable(value.item):
        try:
            scalar = value.item()
        except Exception:
            scalar = value
        if scalar is not value:
            return _collect_evidence_numbers(scalar, path)

    if isinstance(value, dict):
        for key, item in value.items():
            found.extend(_collect_evidence_numbers(item, f"{path}.{key}"))
            # The NUMBER of groups in a grouped result is a deterministic fact
            # about the evidence, so statements such as "6 groups" can be grounded.
            if key == "group_results" and isinstance(item, dict) and item:
                found.append({"value": float(len(item)),
                              "path": f"{path}.{key}.__group_count__"})
    elif isinstance(value, (list, tuple, set)):
        for index, item in enumerate(value):
            found.extend(_collect_evidence_numbers(item, f"{path}[{index}]"))

    return found


def _decimal_places(token: str) -> int:
    """Decimal places a number was written with: '2.67' -> 2, '6' -> 0, '1.5e3' -> 1."""
    mantissa = token.replace(",", "").rstrip("%").lstrip("+-")
    mantissa = re.split(r"[eE]", mantissa)[0]
    return len(mantissa.split(".")[1]) if "." in mantissa else 0


def _numbers_match(
    left: float,
    right: float,
    rel_tol: float,
    abs_tol: float,
    claim_decimals: int = 0,
) -> bool:
    """
    Match a claimed number to an evidence value.

    Besides the tight relative tolerance, a claim written with d >= 1 decimals
    matches if it equals the evidence value rounded (or truncated) to d places,
    i.e. it is within half a unit of its last digit. Without this, correctly
    rounded displays such as 2.67 for 2.66968 were rejected (the 1e-4 relative
    tolerance only allows 0.00027 there) and every grouped-mean report ended
    in REVIEW. Integer claims stay strict: '3' must not match 2.7.
    """
    if math.isclose(left, right, rel_tol=rel_tol, abs_tol=abs_tol):
        return True
    if claim_decimals >= 1:
        return abs(left - right) <= 0.5 * 10 ** (-claim_decimals) + 1e-9
    return False


def validate_text_against_evidence(
    text: Any,
    evidence: Any,
    *,
    rel_tol: float = 1e-4,
    abs_tol: float = 1e-6,
) -> dict:
    """
    Validate numeric tokens in one generated text artifact against evidence.

    Status:
      PASS   - at least one numeric claim and every claim matched evidence.
      REVIEW - one or more claims had no direct match.
      NOT_APPLICABLE - no numeric claims were found.

    REVIEW is deliberately non-failing: derived values, rounded formatting,
    percentages, and semantic claims may need domain-specific validation.
    """
    claims = extract_numeric_claims(text)
    evidence_numbers = _collect_evidence_numbers(evidence)

    matched = []
    unmatched = []

    for claim in claims:
        candidates = [
            item for item in evidence_numbers
            if _numbers_match(
                claim["value"],
                item["value"],
                rel_tol=rel_tol,
                abs_tol=abs_tol,
                claim_decimals=_decimal_places(claim["text"]),
            )
        ]
        if candidates:
            matched.append({
                **claim,
                "matched_evidence_paths": [item["path"] for item in candidates[:5]],
            })
        else:
            unmatched.append(claim)

    if not claims:
        status = "NOT_APPLICABLE"
        reason = "No numeric claims were found; semantic claims require separate validation."
    elif unmatched:
        status = "REVIEW"
        reason = (
            "One or more numeric claims did not match a numeric value in the supplied "
            "evidence. Review derived values, rounding, percentages, and claim context."
        )
    else:
        status = "PASS"
        reason = (
            "Every detected numeric token matched at least one numeric evidence value. "
            "This does not validate non-numeric or causal claims."
        )

    return {
        "status": status,
        "pass": True if status == "PASS" else (False if status == "REVIEW" else None),
        "validation_scope": "numeric_token_to_evidence_match_only",
        "numeric_claim_count": len(claims),
        "matched_claim_count": len(matched),
        "unmatched_claim_count": len(unmatched),
        "matched_claims": matched,
        "unmatched_claims": unmatched,
        "evidence_numeric_value_count": len(evidence_numbers),
        "reason": reason,
        "limitations": [
            "A numeric match does not prove that the number is used in the correct sentence context.",
            "Non-numeric claims, causality, and recommendation logic are not validated here.",
            "Percentages are compared as displayed numeric points, not converted to proportions.",
            "Derived values not stored directly in evidence may receive REVIEW.",
        ],
    }


def validate_insight_and_recommendation(
    insight: Any,
    recommendation: Any,
    evidence: Any,
) -> dict:
    """Validate insight and recommendation numeric claims independently."""
    insight_result = validate_text_against_evidence(insight, evidence)

    # Support both plain-text recommendations and structured generator output.
    # Only the human-readable recommendation is checked for numeric claims.
    # The structured basis and generator status are checked independently by
    # recommendation_validator.py.
    if isinstance(recommendation, dict):
        recommendation_text = recommendation.get("recommendation", "")
    else:
        recommendation_text = recommendation

    recommendation_result = validate_text_against_evidence(
        recommendation_text,
        evidence,
    )

    component_statuses = (
        insight_result["status"],
        recommendation_result["status"],
    )

    # A component with no numeric claims is NOT_APPLICABLE, not a failure.
    # It must not downgrade another component whose numeric claims passed.
    if "FAIL" in component_statuses:
        overall = "FAIL"
    elif "REVIEW" in component_statuses:
        overall = "REVIEW"
    elif "PASS" in component_statuses:
        overall = "PASS"
    else:
        overall = "NOT_APPLICABLE"

    return {
        "insight": insight_result,
        "recommendation": recommendation_result,
        "status": overall,
        "overall": overall,
        "pass": True if overall == "PASS" else (False if overall in {"FAIL", "REVIEW"} else None),
        "scope": "numeric_claim_grounding_only",
    }
