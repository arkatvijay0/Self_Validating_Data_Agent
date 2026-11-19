import os
import json
from datetime import datetime


# ==========================================================
# HELPER FUNCTION
# ==========================================================

def _compact_samples(value, features=None, depth=0):
    """
    Display only: Isolation Forest sample rows carry every dataset column and were
    printed several times (result, evidence, traceability). Reduce each sample row
    to the analysed features and the score wherever they appear.
    """
    if depth > 8:
        return value
    if isinstance(value, dict):
        own = value.get("feature_columns")
        features = [str(f) for f in own] if isinstance(own, list) and own else features
        out = {}
        for key, item in value.items():
            if key == "sample_anomalies" and isinstance(item, list) and item and all(isinstance(r, dict) for r in item):
                keep = list(features or []) + ["anomaly_score"]
                out[key] = [{k: r[k] for k in keep if k in r} or r for r in item]
            else:
                out[key] = _compact_samples(item, features, depth + 1)
        return out
    if isinstance(value, list):
        return [_compact_samples(item, features, depth + 1) for item in value]
    return value


def format_value(value):
    """
    Format values for readable report output.
    """

    value = _compact_samples(value)

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


def _group_label(row, group_columns):
    values = [row.get(column) for column in group_columns]
    if len(group_columns) == 1:
        return f"{group_columns[0]}={values[0]}"
    return ", ".join(f"{column}={value}" for column, value in zip(group_columns, values))


def render_result_table(validated_result, max_rows=30):
    """
    Readable, sorted table for grouped results (highest first) with group sizes.
    Presentation only: it reads the validated result and changes nothing in it.
    """
    if not isinstance(validated_result, dict):
        return []
    groups = [str(g) for g in (validated_result.get("group_by") or [])]
    results = validated_result.get("operation_results")
    rows = results.get("aggregations") if isinstance(results, dict) else None
    if not groups or not isinstance(rows, list) or not rows:
        return []

    value_keys = [
        key for key in rows[0]
        if key not in groups and (key.endswith(("__mean", "__median", "__sum", "__min", "__max",
                                                "__std", "__variance", "__count", "__nunique"))
                                  or key == "row_count")
    ]
    if not value_keys:
        return []
    value_key = value_keys[0]
    sizes = {}
    for size_row in (results.get("group_sizes") or []):
        if isinstance(size_row, dict) and all(g in size_row for g in groups):
            sizes[tuple(str(size_row[g]) for g in groups)] = size_row.get("row_count")

    table = []
    for row in rows:
        value = row.get(value_key)
        if not isinstance(value, (int, float)):
            return []
        table.append((_group_label(row, groups), float(value), sizes.get(tuple(str(row[g]) for g in groups))))
    table.sort(key=lambda item: -item[1])

    is_count = value_key.endswith(("__count", "__nunique")) or value_key == "row_count"
    fmt = (lambda v: f"{int(round(v)):,}") if is_count else (lambda v: f"{v:,.2f}")
    has_sizes = any(item[2] is not None for item in table)
    shown = table
    note = ""
    if len(table) > max_rows:
        shown = table[:15] + [None] + table[-5:]
        note = f"{len(table)} groups in total; showing the 15 highest and 5 lowest."
    width = max(len(item[0]) for item in table if item is not None)
    lines = [f"RESULT TABLE ({value_key.replace('__', ' ')}, highest first)", "-" * 60]
    header = f"{'rank':>4}  {'group':<{width}}  {'value':>14}"
    header += f"  {'records':>9}" if has_sizes else ""
    lines.append(header)
    rank = 0
    for item in shown:
        if item is None:
            lines.append("  ...")
            rank = len(table) - 5
            continue
        rank += 1
        line = f"{rank:>4}  {item[0]:<{width}}  {fmt(item[1]):>14}"
        if has_sizes:
            line += f"  {(f'{int(item[2]):,}' if item[2] is not None else '-'):>9}"
        lines.append(line)
    if note:
        lines.append(note)
    return lines


def compact_result_for_display(result, max_rows=12):
    """
    Large grouped results are shown as a sorted RESULT TABLE; the raw row dump in
    section 3 then keeps only the first rows. Display only: the validated result
    itself is untouched.
    """
    if isinstance(result, dict) and isinstance(result.get("sample_anomalies"), list):
        # Isolation Forest: each sample row carries every dataset column. Keep the
        # analysed features and the score; the full rows stay in the result.
        keep = [str(f) for f in (result.get("feature_columns") or [])] + ["anomaly_score"]
        rows = [{k: row[k] for k in keep if k in row} for row in result["sample_anomalies"] if isinstance(row, dict)]
        shown = dict(result)
        shown["sample_anomalies"] = rows
        shown["sample_note"] = "Only the analysed features and the anomaly score are shown; full rows are in the execution output."
        return shown
    if not isinstance(result, dict) or not render_result_table(result):
        return result
    operation_results = result.get("operation_results")
    rows = operation_results.get("aggregations") if isinstance(operation_results, dict) else None
    if not isinstance(rows, list) or len(rows) <= max_rows:
        return result
    compact = {k: v for k, v in operation_results.items() if k != "group_sizes"}
    compact["aggregations"] = rows[:5] + [f"... {len(rows) - 5} more groups: see the RESULT TABLE below"]
    shown = dict(result)
    shown["operation_results"] = compact
    return shown


def render_dataset_overview(overview):
    if not isinstance(overview, dict) or not overview:
        return []
    lines = ["DATASET OVERVIEW", "-" * 60]
    for label, key in (("File", "file"), ("Rows", "rows"), ("Columns", "columns"),
                       ("Missing cells", "missing_cells"), ("Column types", "column_types"),
                       ("Identifier columns", "identifier_columns")):
        if overview.get(key) not in (None, "", []):
            value = overview[key]
            lines.append(f"{label}: {value:,}" if isinstance(value, int) else f"{label}: {value}")
    lines.append("")
    return lines


def render_pipeline_trace(trace):
    """What actually happened in THIS run, stage by stage."""
    lines = []
    for index, item in enumerate(trace, start=1):
        stage, status, detail = item
        lines.append(f"{index:>2}. {stage}")
        lines.append(f"      Status : {status}")
        if detail:
            lines.append(f"      Detail : {detail}")
    return lines


def create_final_report(
    user_question,
    analysis_type,
    validated_result,
    insight,
    insight_validation,
    evidence,
    recommendation,
    insight_evidence_mapping=None,
    recommendation_validation=None,
    report_extras=None
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

    report_extras = report_extras if isinstance(report_extras, dict) else {}
    overview_lines = render_dataset_overview(report_extras.get("dataset_overview"))
    if overview_lines:
        report_lines.append("")
        report_lines.extend(overview_lines)

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

        for key, value in compact_result_for_display(validated_result).items():

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

    table_lines = render_result_table(validated_result)
    if table_lines:
        report_lines.extend(table_lines)
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
    grouped_user_claim_status_for_insight = ""
    if isinstance(insight_validation, dict):
        checks_for_insight_status = insight_validation.get("checks", {})
        if isinstance(checks_for_insight_status, dict):
            grouped_check_for_insight_status = checks_for_insight_status.get(
                "grouped_mean_claim_validation", {}
            )
            if isinstance(grouped_check_for_insight_status, dict):
                user_check_for_insight_status = grouped_check_for_insight_status.get(
                    "user_claim_validation", {}
                )
                if isinstance(user_check_for_insight_status, dict):
                    grouped_user_claim_status_for_insight = str(
                        user_check_for_insight_status.get("status", "")
                    ).upper()

    # When grouped validation explicitly separates the user's unsupported claim,
    # do not mislabel an otherwise accurate generated insight as invalid.
    if grouped_user_claim_status_for_insight == "NOT_SUPPORTED":
        early_claim_failure = False

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

        # Support the legacy, multi-metric, and specification-driven
        # grouped-mean validation schemas.
        multi_metric_check = checks.get("multi_metric_claim_validation")
        specification_grouped_mean_keys = {
            "result_status",
            "metric_mentioned",
            "all_grouping_columns_mentioned",
            "all_composite_group_labels_mentioned",
            "all_group_means_mentioned",
            "each_group_mean_paired_with_correct_label",
        }

        if isinstance(multi_metric_check, dict):
            metric_checks = multi_metric_check.get("metrics", {})
            if not isinstance(metric_checks, dict):
                metric_checks = {}

            # Presence passes only when every requested metric is represented
            # by name. An empty metrics dictionary must not pass by default.
            presence_pass = bool(metric_checks) and all(
                isinstance(metric_check, dict)
                and metric_check.get("metric_named") is True
                for metric_check in metric_checks.values()
            )

            # The multi-metric validator's pass flag reflects the full set of
            # metric claims, including whether the values/differences validate.
            result_pass = multi_metric_check.get("pass") is True

        elif specification_grouped_mean_keys.issubset(checks.keys()):
            # The specification-driven grouped-mean validator reports its
            # checks directly under checks, rather than under legacy
            # presence/result dictionaries.
            presence_pass = all(
                checks.get(key) is True
                for key in (
                    "metric_mentioned",
                    "all_grouping_columns_mentioned",
                    "all_composite_group_labels_mentioned",
                )
            )

            # A result passes only when the source result is valid, all
            # expected means are stated, and every mean is paired with its
            # correct composite group label.
            result_pass = all(
                checks.get(key) is True
                for key in (
                    "result_status",
                    "all_group_means_mentioned",
                    "each_group_mean_paired_with_correct_label",
                )
            )

        elif str(insight_validation.get("analysis_type", "")) in {
            "grouped_count_analysis", "specification_aggregate_analysis"
        }:
            # Dedicated count / aggregate validators report plain boolean checks.
            flags = {k: v for k, v in checks.items() if isinstance(v, bool)}
            presence_keys = [k for k in flags if "mentioned" in k]
            presence_pass = bool(flags) and all(flags[k] for k in presence_keys)
            result_pass = bool(flags) and all(v for k, v in flags.items() if k not in presence_keys)

        else:
            presence = checks.get("presence", {})
            result_check = checks.get("result", {})
            presence_pass = (
                isinstance(presence, dict) and presence.get("pass") is True
            )
            result_pass = (
                isinstance(result_check, dict) and result_check.get("pass") is True
            )

        report_lines.append(
            "Presence Check: " + ("PASS" if presence_pass else "FAIL")
        )

        report_lines.append(
            "Result Check: " + ("PASS" if result_pass else "FAIL")
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

    # Report the user's directional claim separately from insight accuracy.
    validation_checks_for_claim = (
        insight_validation.get("checks", {})
        if isinstance(insight_validation, dict) else {}
    )
    grouped_check_for_claim = (
        validation_checks_for_claim.get("grouped_mean_claim_validation", {})
        if isinstance(validation_checks_for_claim, dict) else {}
    )
    user_claim_check_for_report = (
        grouped_check_for_claim.get("user_claim_validation", {})
        if isinstance(grouped_check_for_claim, dict) else {}
    )
    if isinstance(user_claim_check_for_report, dict):
        user_claim_status_for_report = str(user_claim_check_for_report.get("status", "")).upper()
        if user_claim_status_for_report in {"SUPPORTED", "NOT_SUPPORTED", "NOT_ASSESSED", "REVIEW"}:
            report_lines.append(f"User Claim Verdict: {user_claim_status_for_report}")
            if user_claim_check_for_report.get("reason"):
                report_lines.append(f"User Claim Explanation: {user_claim_check_for_report['reason']}")

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

        # The summary repeats the evidence printed just above; skip exact duplicates.
        if evidence_summary and evidence_result is not None:
            try:
                import json as _json
                if _json.loads(str(evidence_summary)) == _json.loads(
                    _json.dumps(evidence_result, default=str)
                ):
                    evidence_summary = None
            except (ValueError, TypeError):
                pass

        if evidence_summary:

            report_lines.append("")

            report_lines.append(
                "Evidence Summary:"
            )

            summary_text = str(evidence_summary)
            try:
                import json as _json
                summary_text = _json.dumps(
                    _compact_samples(_json.loads(summary_text)), indent=2, default=str
                )
            except (ValueError, TypeError):
                pass

            report_lines.append(
                summary_text
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

    independent_recommendation_validation = (
        recommendation_validation
        if isinstance(recommendation_validation, dict)
        else (
            insight_evidence_mapping.get("recommendation_validation", {})
            if isinstance(insight_evidence_mapping, dict) else {}
        )
    )
    independent_recommendation_status = str(
        independent_recommendation_validation.get("status", "")
    ).upper()

    if independent_recommendation_status in {"FAIL", "REVIEW"}:
        recommendation_status = independent_recommendation_status

    report_lines.append(
        f"Recommendation Status: "
        f"{recommendation_status}"
    )

    if independent_recommendation_validation:
        report_lines.append(
            "Independent Recommendation Validation: "
            + independent_recommendation_status
        )
        report_lines.append(
            format_value(independent_recommendation_validation)
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
            str(claim_validation_for_reliability.get("overall", "")).upper() == "FAIL"
            or str(claim_validation_for_reliability.get("status", "")).upper() == "FAIL"
            or claim_validation_for_reliability.get("pass") is False
            or failed_claim_count > 0
        )
    grouped_user_claim_for_reliability = {}
    if isinstance(insight_validation, dict):
        validation_checks_for_reliability = insight_validation.get("checks", {})
        if isinstance(validation_checks_for_reliability, dict):
            grouped_validation_for_reliability = validation_checks_for_reliability.get(
                "grouped_mean_claim_validation", {}
            )
            if isinstance(grouped_validation_for_reliability, dict):
                grouped_user_claim_for_reliability = grouped_validation_for_reliability.get(
                    "user_claim_validation", {}
                )
    user_claim_not_supported = (
        isinstance(grouped_user_claim_for_reliability, dict)
        and str(grouped_user_claim_for_reliability.get("status", "")).upper() == "NOT_SUPPORTED"
    )

    mapping_failed = str(mapping_for_reliability.get("status", "")).upper() == "FAIL"
    # REVIEW means source references are incomplete, not that the numerical
    # result or recommendation is invalid. Keep it visible in the report, but
    # do not turn otherwise-passing core validation into a hard FAIL.

    overall_reliability = "PASS"

    # A successful-looking insight/evidence wrapper must never override an
    # unsupported or explicitly failed primary analysis result.
    primary_result_status = ""
    if isinstance(validated_result, dict):
        primary_result_status = str(validated_result.get("status", "")).strip().upper()

    if primary_result_status in {"UNSUPPORTED", "NOT_SUPPORTED", "FAIL", "FAILED", "ERROR"}:
        overall_reliability = "FAIL"

    if insight_status != "PASS":

        overall_reliability = "FAIL"

    if evidence_status != "PASS":

        overall_reliability = "FAIL"

    if recommendation_status == "FAIL":

        overall_reliability = "FAIL"
    elif recommendation_status == "REVIEW" and overall_reliability == "PASS":
        overall_reliability = "REVIEW"

    # An unsupported original user claim is reported separately. It should not
    # invalidate an otherwise correct analysis and evidence-backed generated insight.
    if claim_failed or mapping_failed:
        overall_reliability = "FAIL"

    report_lines.append(
        overall_reliability
    )

    report_lines.append("")

    traceability_status_for_report = str(
        mapping_for_reliability.get("traceability_status", "REVIEW")
    ).strip().upper()

    if traceability_status_for_report == "PASS":
        traceability_explanation = (
          "Traceability PASS indicates that explicit source references "
          "are available for the mapped evidence."
        )
    elif traceability_status_for_report == "REVIEW":
        traceability_explanation = (
         "Traceability REVIEW indicates that explicit source references "
         "may be incomplete; review the evidence mapping before relying "
         "on full source-level traceability."
        )
    else:
        traceability_explanation = (
            f"Traceability status is {traceability_status_for_report or 'UNAVAILABLE'}; "
            "inspect the evidence mapping to determine whether source references "
            "are available."
        )

    report_lines.append(
         "Overall reliability is marked PASS when the generated insight, evidence, "
         "recommendation, applicable claim-level checks, and insight mapping pass. "
         "The user's original directional claim is reported separately and does not "
         "by itself invalidate an otherwise correct analysis. "
          + traceability_explanation
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
        "      |",
        "      v",
        "Data Profiling",
        "      |",
        "      v",
        "Task Analysis",
        "      |",
        "      v",
        "EDA Planning",
        "      |",
        "      v",
        "Code Generation",
        "      |",
        "      v",
        "Code Execution",
        "      |",
        "      v",
        "Code Validation",
        "      |",
        "      v",
        "Data Usage Validation",
        "      |",
        "      v",
        "Result Validation",
        "      |",
        "      v",
        "PASS?",
        "   /       \\",
        " YES       NO",
        "  |         |",
        "  v         v",
        "Continue   Correction Feedback",
        "             |",
        "             v",
        "        Regenerate Code",
        "             |",
        "             v",
        "           Retry",
        "             |",
        "             v",
        "Validated EDA Result",
        "      |",
        "      v",
        "Insight Generation",
        "      |",
        "      v",
        "Insight Validation",
        "      |",
        "      v",
        "Evidence Mapping",
        "      |",
        "      v",
        "Recommendation",
        "      |",
        "      v",
        "Independent Recommendation Validation",
        "      |",
        "      v",
        "Final Report"
    ]

    trace = report_extras.get("pipeline_trace")
    if trace:
        report_lines.extend(render_pipeline_trace(trace))
    else:
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
            "PASS"
            if str(claim_validation_for_reliability.get("overall", "")).upper() == "PASS"
            or str(claim_validation_for_reliability.get("status", "")).upper() == "PASS"
            or claim_validation_for_reliability.get("pass") is True
            else "REVIEW"
        )
        report_lines.append(f"Claim-Level Validation: {claim_summary_status}")

    if isinstance(grouped_user_claim_for_reliability, dict) and grouped_user_claim_for_reliability.get("status"):
        report_lines.append(
            f"User Claim Verdict: {str(grouped_user_claim_for_reliability.get('status')).upper()}"
        )

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
