import os
import json
from datetime import datetime


# ==========================================================
# HELPER FUNCTION
# ==========================================================

def format_value(value):
    """
    Format values for readable report output.
    """

    if isinstance(value, float):

        return f"{value:,.2f}"

    if isinstance(value, int):

        return f"{value:,}"

    if isinstance(value, list):
        # Render labeled statistical rows in a readable, auditable format.
        if value and all(isinstance(item, dict) and item.get("column_name") for item in value):
            lines = []
            preferred = ("count", "mean", "median", "std", "min", "max", "skewness", "missing_count", "missing_percentage")
            for item in value:
                lines.append(f"- {item.get('column_name')}:")
                for key in preferred:
                    if key in item:
                        lines.append(f"    {key}: {format_value(item[key])}")
            return "\n".join(lines)
        return json.dumps(value, indent=2, ensure_ascii=False, default=str)

    if isinstance(value, dict):

        return json.dumps(
            value,
            indent=2,
            default=str
        )

    return str(value)


# ==========================================================
# VALIDATION STATUS HELPER
# ==========================================================

def get_validation_status(validation):
    """
    Safely extract PASS/FAIL status from validation result.
    """

    if not isinstance(validation, dict):

        return "FAIL"

    if validation.get("overall") == "PASS":

        return "PASS"

    if validation.get("pass") is True:

        return "PASS"

    return "FAIL"


# ==========================================================
# CREATE FINAL REPORT
# ==========================================================

def create_final_report(
    user_question,
    analysis_type,
    validated_result,
    insight,
    insight_validation,
    evidence,
    recommendation,
    insight_evidence_mapping=None
):
    """
    Create the final Self-Validating EDA Agent report.
    """

    report_lines = []

    # ======================================================
    # HEADER
    # ======================================================

    report_lines.append(
        "=" * 60
    )

    report_lines.append(
        "SELF-VALIDATING EDA AGENT"
    )

    report_lines.append(
        "FINAL ANALYSIS REPORT"
    )

    report_lines.append(
        "=" * 60
    )

    report_lines.append("")

    report_lines.append(
        "Generated On:"
    )

    report_lines.append(
        datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    )

    report_lines.append("")

    # ======================================================
    # 1. USER QUESTION
    # ======================================================

    report_lines.append(
        "1. USER QUESTION"
    )

    report_lines.append(
        "-" * 60
    )

    report_lines.append(
        str(user_question)
    )

    report_lines.append("")

    # ======================================================
    # 2. ANALYSIS TYPE
    # ======================================================

    report_lines.append(
        "2. ANALYSIS TYPE"
    )

    report_lines.append(
        "-" * 60
    )

    report_lines.append(
        str(analysis_type)
    )

    report_lines.append("")

    # ======================================================
    # 3. VALIDATED EDA RESULT
    # ======================================================

    report_lines.append(
        "3. VALIDATED EDA RESULT"
    )

    report_lines.append(
        "-" * 60
    )

    if isinstance(
        validated_result,
        dict
    ):

        for key, value in validated_result.items():

            report_lines.append(
                f"{key}: {format_value(value)}"
            )

    else:

        report_lines.append(
            format_value(validated_result)
        )

    report_lines.append("")

    # ======================================================
    # 4. GENERATED INSIGHT
    # ======================================================

    report_lines.append(
        "4. GENERATED INSIGHT"
    )

    report_lines.append(
        "-" * 60
    )

    report_lines.append(
        str(insight)
    )

    report_lines.append("")

    # ======================================================
    # 5. INSIGHT VALIDATION
    # ======================================================

    report_lines.append(
        "5. INSIGHT VALIDATION"
    )

    report_lines.append(
        "-" * 60
    )

    insight_status = get_validation_status(
        insight_validation
    )

    # Reflect critical claim/mapping failures in the displayed insight status too.
    early_mapping = insight_evidence_mapping if isinstance(insight_evidence_mapping, dict) else {}
    early_claims = early_mapping.get("claim_validation")
    early_claim_failure = str(early_mapping.get("status", "")).upper() == "FAIL"
    if isinstance(early_claims, dict):
        try:
            early_claim_failure = early_claim_failure or int(early_claims.get("failed_claims", 0) or 0) > 0
        except (TypeError, ValueError):
            early_claim_failure = True
        early_claim_failure = early_claim_failure or early_claims.get("status") == "FAIL"
    if early_claim_failure:
        insight_status = "FAIL"

    report_lines.append(
        f"Overall Validation: {insight_status}"
    )

    # ------------------------------------------------------
    # Presence check
    # ------------------------------------------------------

    if isinstance(
        insight_validation,
        dict
    ):

        checks = insight_validation.get(
            "checks",
            {}
        )

        presence = checks.get(
            "presence",
            {}
        )

        result_check = checks.get(
            "result",
            {}
        )

        report_lines.append(
            "Presence Check: "
            + (
                "PASS"
                if presence.get("pass") is True
                else "FAIL"
            )
        )

        report_lines.append(
            "Result Check: "
            + (
                "PASS"
                if result_check.get("pass") is True
                else "FAIL"
            )
        )

        # --------------------------------------------------
        # Claim-level validation
        # --------------------------------------------------

        claim_validation = checks.get(
            "claim_level_validation"
        )

        if isinstance(
            claim_validation,
            dict
        ):

            claim_status = (
                "PASS"
                if (
                    claim_validation.get(
                        "overall"
                    ) == "PASS"
                    or
                    claim_validation.get(
                        "pass"
                    ) is True
                )
                else "FAIL"
            )

            report_lines.append(
                f"Claim-Level Validation: "
                f"{claim_status}"
            )

            total_claims = claim_validation.get(
                "total_claims"
            )

            passed_claims = claim_validation.get(
                "passed_claims"
            )

            failed_claims = claim_validation.get(
                "failed_claims"
            )

            if total_claims is not None:

                report_lines.append(
                    f"Total Claims: "
                    f"{total_claims}"
                )

            if passed_claims is not None:

                report_lines.append(
                    f"Passed Claims: "
                    f"{passed_claims}"
                )

            if failed_claims is not None:

                report_lines.append(
                    f"Failed Claims: "
                    f"{failed_claims}"
                )

    report_lines.append("")

    report_lines.append(
        "The insight validation stage checks whether "
        "the generated natural-language statement is "
        "supported by the validated numerical result."
    )

    report_lines.append("")

    # ======================================================
    # 6. SUPPORTING EVIDENCE
    # ======================================================

    report_lines.append(
        "6. SUPPORTING EVIDENCE"
    )

    report_lines.append(
        "-" * 60
    )

    evidence_status = "FAIL"

    if isinstance(
        evidence,
        dict
    ):

        if evidence.get(
            "status"
        ) == "PASS":

            evidence_status = "PASS"

    report_lines.append(
        f"Evidence Status: {evidence_status}"
    )

    report_lines.append("")

    if isinstance(
        evidence,
        dict
    ):

        evidence_result = evidence.get(
            "result"
        )

        if evidence_result is not None:

            report_lines.append(
                format_value(
                    evidence_result
                )
            )

        evidence_summary = evidence.get(
            "evidence_summary"
        )

        if evidence_summary:

            report_lines.append("")

            report_lines.append(
                "Evidence Summary:"
            )

            report_lines.append(
                str(evidence_summary)
            )

    else:

        report_lines.append(
            format_value(evidence)
        )

    report_lines.append("")

    # ------------------------------------------------------
    # Insight-to-evidence traceability
    # ------------------------------------------------------

    report_lines.append("INSIGHT-TO-EVIDENCE TRACEABILITY")
    report_lines.append("-" * 60)

    mapping = insight_evidence_mapping if isinstance(insight_evidence_mapping, dict) else {}
    mapping_status = mapping.get("status", "NOT PROVIDED")
    traceability_status = mapping.get("traceability_status", "REVIEW")
    report_lines.append(f"Insight mapping status: {mapping_status}")
    report_lines.append(f"Traceability status: {traceability_status}")
    report_lines.append(f"Mapped analysis type: {mapping.get('analysis_type', evidence.get('analysis_type', 'unknown') if isinstance(evidence, dict) else 'unknown')}")

    sources = mapping.get("evidence_sources", [])
    if isinstance(sources, list) and sources:
        report_lines.append("Evidence source references:")
        for source in sources:
            if not isinstance(source, dict):
                continue
            source_id = source.get("source_id", "unnamed_source")
            source_type = source.get("source_type", "unspecified")
            details = []
            for key in ("category_column", "measure_column", "aggregation", "x_variable", "y_variable", "method", "time_column", "step_number", "step_name", "execution_status", "validation_status"):
                if source.get(key) is not None:
                    details.append(f"{key}={source[key]}")
            suffix = f" ({'; '.join(details)})" if details else ""
            report_lines.append(f"- {source_id} [{source_type}]{suffix}")
            result = source.get("result")
            if result is not None and source_id in {"primary_result", "dataset_facts.numeric_summary", "dataset_facts.profit_by_region", "grouped_series", "correlation_result", "trend_result"}:
                report_lines.append("  Linked result:")
                rendered_result = format_value(result)
                for result_line in str(rendered_result).splitlines():
                    report_lines.append(f"    {result_line}")
    else:
        report_lines.append("No explicit evidence-source references were attached to this insight.")

    claim_validation = mapping.get("claim_validation")
    if isinstance(claim_validation, dict):
        report_lines.append("Claim validation details:")
        report_lines.append(format_value(claim_validation))

    report_lines.append("")

    # ======================================================
    # 7. RECOMMENDATION
    # ======================================================

    report_lines.append(
        "7. RECOMMENDATION"
    )

    report_lines.append(
        "-" * 60
    )

    recommendation_status = "FAIL"

    recommendation_text = recommendation

    if isinstance(
        recommendation,
        dict
    ):

        if recommendation.get(
            "status"
        ) == "PASS":

            recommendation_status = "PASS"

        recommendation_text = recommendation.get(
            "recommendation",
            recommendation
        )

    elif isinstance(recommendation, str) and recommendation.strip():
        # A non-empty plain-text recommendation was generated successfully.
        recommendation_status = "PASS"

    report_lines.append(
        f"Recommendation Status: "
        f"{recommendation_status}"
    )

    report_lines.append("")

    report_lines.append(
        str(recommendation_text)
    )

    report_lines.append("")

    # ======================================================
    # 8. OVERALL RELIABILITY
    # ======================================================

    report_lines.append(
        "8. OVERALL RELIABILITY"
    )

    report_lines.append(
        "-" * 60
    )

    # Critical claim/mapping failures override a passing top-level stage.
    mapping_for_reliability = insight_evidence_mapping if isinstance(insight_evidence_mapping, dict) else {}

    claim_validation_for_reliability = None
    if isinstance(insight_validation, dict):
        validation_checks = insight_validation.get("checks", {})
        if isinstance(validation_checks, dict):
            claim_validation_for_reliability = validation_checks.get("claim_level_validation")
    if claim_validation_for_reliability is None:
        claim_validation_for_reliability = mapping_for_reliability.get("claim_validation")

    claim_failed = False
    if isinstance(claim_validation_for_reliability, dict):
        try:
            failed_claim_count = int(claim_validation_for_reliability.get("failed_claims", 0) or 0)
        except (TypeError, ValueError):
            failed_claim_count = 0
        claim_failed = (
            claim_validation_for_reliability.get("overall") == "FAIL"
            or claim_validation_for_reliability.get("pass") is False
            or failed_claim_count > 0
        )
    mapping_failed = str(mapping_for_reliability.get("status", "")).upper() == "FAIL"
    # REVIEW means source references are incomplete, not that the numerical
    # result or recommendation is invalid. Keep it visible in the report, but
    # do not turn otherwise-passing core validation into a hard FAIL.

    overall_reliability = "PASS"

    if insight_status != "PASS":

        overall_reliability = "FAIL"

    if evidence_status != "PASS":

        overall_reliability = "FAIL"

    if recommendation_status != "PASS":

        overall_reliability = "FAIL"

    if claim_failed or mapping_failed:
        overall_reliability = "FAIL"

    report_lines.append(
        overall_reliability
    )

    report_lines.append("")

    report_lines.append(
        "Overall reliability is marked PASS when the insight, evidence, "
        "recommendation, claim-level checks, and insight mapping pass. "
        "Traceability REVIEW indicates missing explicit source references; "
        "it remains visible as a limitation but does not by itself invalidate "
        "a numerically validated result."
    )

    report_lines.append("")

    # ======================================================
    # 9. VALIDATION AND SELF-CORRECTION PIPELINE
    # ======================================================

    report_lines.append(
        "9. VALIDATION AND SELF-CORRECTION PIPELINE"
    )

    report_lines.append(
        "-" * 60
    )

    pipeline = [
        "User Question",
        "      ↓",
        "Data Profiling",
        "      ↓",
        "Task Analysis",
        "      ↓",
        "EDA Planning",
        "      ↓",
        "Code Generation",
        "      ↓",
        "Code Execution",
        "      ↓",
        "Code Validation",
        "      ↓",
        "Data Usage Validation",
        "      ↓",
        "Result Validation",
        "      ↓",
        "PASS?",
        "   ↙       ↘",
        " YES       NO",
        " ↓          ↓",
        "Continue   Correction Feedback",
        "             ↓",
        "        Regenerate Code",
        "             ↓",
        "           Retry",
        "             ↓",
        "Validated EDA Result",
        "      ↓",
        "Insight Generation",
        "      ↓",
        "Insight Validation",
        "      ↓",
        "Evidence Mapping",
        "      ↓",
        "Recommendation",
        "      ↓",
        "Final Report"
    ]

    report_lines.extend(
        pipeline
    )

    report_lines.append("")

    # ======================================================
    # 10. VALIDATION STATUS SUMMARY
    # ======================================================

    report_lines.append(
        "10. VALIDATION STATUS SUMMARY"
    )

    report_lines.append(
        "-" * 60
    )

    report_lines.append(
        f"Insight Validation: "
        f"{insight_status}"
    )

    report_lines.append(
        f"Evidence Validation: "
        f"{evidence_status}"
    )

    report_lines.append(
        f"Recommendation Validation: "
        f"{recommendation_status}"
    )

    if isinstance(claim_validation_for_reliability, dict):
        claim_summary_status = "FAIL" if claim_failed else (
            "PASS" if claim_validation_for_reliability.get("overall") == "PASS"
            or claim_validation_for_reliability.get("pass") is True else "REVIEW"
        )
        report_lines.append(f"Claim-Level Validation: {claim_summary_status}")

    if mapping_for_reliability:
        report_lines.append(f"Insight Mapping: {mapping_for_reliability.get('status', 'NOT PROVIDED')}")
        report_lines.append(f"Traceability: {mapping_for_reliability.get('traceability_status', 'REVIEW')}")

    report_lines.append(
        f"Overall Reliability: "
        f"{overall_reliability}"
    )

    # ======================================================
    # END OF REPORT
    # ======================================================

    report_lines.append("")

    report_lines.append(
        "=" * 60
    )

    report_lines.append(
        "END OF REPORT"
    )

    report_lines.append(
        "=" * 60
    )

    # ======================================================
    # RETURN REPORT
    # ======================================================

    return "\n".join(
        report_lines
    )


# ==========================================================
# SAVE REPORT
# ==========================================================

def save_report(
    report,
    filepath
):
    """
    Save the final report to a text file
    and return the saved file path.
    """

    directory = os.path.dirname(
        filepath
    )

    if directory:

        os.makedirs(
            directory,
            exist_ok=True
        )

    with open(
        filepath,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            report
        )

    print(
        f"\nReport saved successfully to:\n"
        f"{filepath}"
    )

    return filepath


# ==========================================================
# MODULE TEST
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)

    print(
        "FINAL REPORT MODULE TEST"
    )

    print("=" * 60)

    sample_result = {
        "count": 100,
        "mean": 441681.98399999994,
        "median": 290767.995,
        "std": 438537.90705963754,
        "min": 1258.02,
        "max": 1719922.04,
        "Q1": 121443.58499999999,
        "Q3": 635828.8,
        "IQR": 514385.2150000001,
        "lower_bound": -650134.2375000002,
        "upper_bound": 1407406.6225,
        "outlier_count": 5,
        "outlier_values": [
            1487261.02,
            1505888.07,
            1541705.29,
            1719922.04,
            1678540.98
        ]
    }

    sample_insight = (
        "The analysis evaluated 100 Total Profit "
        "observations. The mean Total Profit was "
        "441,681.98, while the median was "
        "290,767.99."
    )

    sample_validation = {
        "overall": "PASS",
        "pass": True,
        "checks": {
            "presence": {
                "pass": True
            },
            "result": {
                "pass": True
            },
            "claim_level_validation": {
                "overall": "PASS",
                "pass": True,
                "total_claims": 14,
                "passed_claims": 14,
                "failed_claims": 0
            }
        }
    }

    sample_evidence = {
        "status": "PASS",
        "result": sample_result,
        "evidence_summary": str(
            sample_result
        )
    }

    sample_recommendation = {
        "status": "PASS",
        "recommendation":
            "Use the validated EDA evidence "
            "as the basis for further "
            "investigation."
    }

    report = create_final_report(
        "Are there any unusual total profit values?",
        "anomaly_detection",
        sample_result,
        sample_insight,
        sample_validation,
        sample_evidence,
        sample_recommendation
    )

    print(report)

    test_path = "outputs/test_final_report.txt"

    save_report(
        report,
        test_path
    )

    print(
        "\nFINAL REPORT MODULE TEST COMPLETED"
    )