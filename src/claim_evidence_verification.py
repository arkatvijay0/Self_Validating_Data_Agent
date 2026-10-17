"""Claim-to-evidence verification layer for the Self-Validating Data Agent.

This module standardizes the outcome of existing evidence mapping without
replacing analysis-specific validators. It distinguishes a supported claim,
an explicit contradiction, and a claim that cannot be verified from available
evidence.
"""

from __future__ import annotations

from typing import Any, Dict


SUPPORTED = "SUPPORTED"
CONTRADICTED = "CONTRADICTED"
INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


def _load_evidence_mapper():
    """Load the existing mapper in either script or package execution mode."""
    try:
        from .evidence_mapping import map_insight_to_evidence
    except ImportError:
        from evidence_mapping import map_insight_to_evidence
    return map_insight_to_evidence


def _classify_mapping(mapping: Dict[str, Any]) -> str:
    """Classify a legacy mapping result conservatively."""
    status = str(mapping.get("status", "")).upper()
    reason = str(mapping.get("reason", "")).lower()

    if status == "PASS":
        return SUPPORTED

    # Only classify as contradicted when the validator explicitly reports
    # disagreement, not merely a missing field or an unavailable validator.
    contradiction_markers = (
        "contradict",
        "does not match",
        "do not match",
        "does not agree",
        "does not match the evidence",
        "expected top category",
        "direction claim does not match",
        "reported mean",
    )
    if status == "FAIL" and any(marker in reason for marker in contradiction_markers):
        return CONTRADICTED

    return INSUFFICIENT_EVIDENCE


def verify_claim_against_evidence(
    insight: Any,
    evidence: Any,
    analysis_type: str | None = None,
) -> Dict[str, Any]:
    """Verify an insight against evidence using the project's existing mapper.

    Returns a stable outcome label and preserves the full legacy validation
    object under ``details``. ``PASS``/``FAIL`` status remains available for
    callers that use the existing convention.
    """
    if not isinstance(evidence, dict):
        details = {
            "status": "FAIL",
            "reason": "Evidence must be a dictionary.",
            "checks": [],
        }
        outcome = INSUFFICIENT_EVIDENCE
    elif str(evidence.get("status", "")).upper() != "PASS":
        details = {
            "status": "FAIL",
            "reason": "Evidence has not passed its own validation.",
            "checks": [],
        }
        outcome = INSUFFICIENT_EVIDENCE
    elif insight is None or not str(insight).strip():
        details = {
            "status": "FAIL",
            "reason": "Insight is empty; no claim can be verified.",
            "checks": [],
        }
        outcome = INSUFFICIENT_EVIDENCE
    else:
        mapper = _load_evidence_mapper()
        # The current mapper accepts exactly (insight, evidence); the
        # analysis type is read from the evidence payload itself.
        details = mapper(
            insight=str(insight),
            evidence=evidence,
        )
        if not isinstance(details, dict):
            details = {
                "status": "FAIL",
                "reason": "Evidence mapper returned an unsupported result.",
                "checks": [],
            }
        outcome = _classify_mapping(details)

    return {
        "outcome": outcome,
        "pass": outcome == SUPPORTED,
        "status": "PASS" if outcome == SUPPORTED else "FAIL",
        "analysis_type": analysis_type or (
            evidence.get("analysis_type") if isinstance(evidence, dict) else None
        ),
        "claim": str(insight) if insight is not None else None,
        "reason": details.get("reason"),
        "checks": details.get("checks", []),
        "details": details,
    }


def verify_claims(claims, evidence, analysis_type=None):
    """Verify a list of claims and return per-claim and aggregate outcomes."""
    if isinstance(claims, str) or not isinstance(claims, (list, tuple)):
        claims = [claims]

    results = [
        verify_claim_against_evidence(claim, evidence, analysis_type)
        for claim in claims
    ]
    counts = {
        SUPPORTED: sum(item["outcome"] == SUPPORTED for item in results),
        CONTRADICTED: sum(item["outcome"] == CONTRADICTED for item in results),
        INSUFFICIENT_EVIDENCE: sum(
            item["outcome"] == INSUFFICIENT_EVIDENCE for item in results
        ),
    }
    if counts[CONTRADICTED]:
        overall = CONTRADICTED
    elif counts[INSUFFICIENT_EVIDENCE]:
        overall = INSUFFICIENT_EVIDENCE
    else:
        overall = SUPPORTED

    return {
        "overall_outcome": overall,
        "claim_count": len(results),
        "supported_count": counts[SUPPORTED],
        "contradicted_count": counts[CONTRADICTED],
        "insufficient_evidence_count": counts[INSUFFICIENT_EVIDENCE],
        "claims": results,
    }
