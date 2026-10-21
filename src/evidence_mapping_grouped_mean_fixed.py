import pandas as pd
import numpy as np
import re


# ============================================================
# HELPER - CONVERT TO SERIES
# ============================================================

def _to_series(result):

    if isinstance(
        result,
        pd.Series
    ):

        return result

    if isinstance(
        result,
        dict
    ):

        if (
            "Region" in result
            and
            "Total Profit" in result
        ):

            return pd.Series(
                {
                    result["Region"]:
                        result["Total Profit"]
                }
            )

        try:

            return pd.Series(
                result
            )

        except Exception:

            return None

    if isinstance(
        result,
        pd.DataFrame
    ):

        if (
            result.shape[1] >= 2
        ):

            return pd.Series(
                result.iloc[:, 1].values,
                index=result.iloc[:, 0].values
            )

    return None


# ============================================================
# HELPER - SAFE NUMBER
# ============================================================

def _safe_number(value):

    try:

        return float(value)

    except Exception:

        return None


# ============================================================
# GROUPED EVIDENCE
# ============================================================

def create_grouped_evidence(
    result,
    category,
    metric,
    operation
):

    series = _to_series(
        result
    )

    if series is None:

        return {
            "status": "FAIL",
            "analysis_type":
                "grouped_analysis",
            "category":
                category,
            "metric":
                metric,
            "operation":
                operation,
            "reason":
                "Result could not be converted to a pandas Series."
        }

    if len(series) == 0:

        return {
            "status": "FAIL",
            "analysis_type":
                "grouped_analysis",
            "category":
                category,
            "metric":
                metric,
            "operation":
                operation,
            "reason":
                "Grouped result is empty."
        }

    series = series.sort_values(
        ascending=False
    )

    top_category = str(
        series.index[0]
    )

    top_value = _safe_number(
        series.iloc[0]
    )

    second_category = None
    second_value = None

    if len(series) > 1:

        second_category = str(
            series.index[1]
        )

        second_value = _safe_number(
            series.iloc[1]
        )

    supporting_result = {
        str(key):
            _safe_number(value)
        for key, value in series.items()
    }

    if second_category is not None:

        evidence_summary = (
            f"The {operation} of {metric} "
            f"was calculated for each {category}. "
            f"{top_category} had the highest value "
            f"at {top_value:,.2f}. "
            f"{second_category} ranked second "
            f"with {second_value:,.2f}."
        )

    else:

        evidence_summary = (
            f"The {operation} of {metric} "
            f"was calculated for each {category}. "
            f"{top_category} had the highest value "
            f"at {top_value:,.2f}."
        )

    return {

        "status":
            "PASS",

        "analysis_type":
            "grouped_analysis",

        "category":
            category,

        "metric":
            metric,

        "operation":
            operation,

        "evidence": {

            "top_category":
                top_category,

            "top_value":
                top_value,

            "second_category":
                second_category,

            "second_value":
                second_value
        },

        "supporting_result":
            supporting_result,

        "evidence_summary":
            evidence_summary
    }


# ============================================================
# CORRELATION EVIDENCE
# ============================================================

def create_correlation_evidence(
    result,
    x_column,
    y_column,
    method="Pearson"
):

    try:

        correlation = float(
            result
        )

    except Exception:

        return {
            "status": "FAIL",
            "analysis_type":
                "correlation_analysis",
            "reason":
                "Correlation result is not numeric."
        }

    if correlation > 0:

        direction = "positive"

    elif correlation < 0:

        direction = "negative"

    else:

        direction = "neutral"

    absolute_value = abs(
        correlation
    )

    if absolute_value >= 0.7:

        strength = "strong"

    elif absolute_value >= 0.3:

        strength = "moderate"

    else:

        strength = "weak"

    return {

        "status":
            "PASS",

        "analysis_type":
            "correlation_analysis",

        "x_variable":
            x_column,

        "y_variable":
            y_column,

        "method":
            method,

        "correlation":
            correlation,

        "direction":
            direction,

        "strength":
            strength,

        "evidence_summary":
            (
                f"{method} correlation between "
                f"{x_column} and {y_column} "
                f"is {correlation:.4f}, indicating "
                f"a {strength} {direction} relationship."
            )
    }




# ============================================================
# TREND EVIDENCE
# ============================================================

def create_trend_evidence(
    result,
    metric="Total Revenue",
    time_label="Order Date"
):
    """
    Build evidence for a time-based result.

    Accepts a pandas Series/DataFrame, a dictionary, or a list of
    dictionaries. It preserves the supplied result and only reports
    a direction when numeric observations can be extracted reliably.
    """
    try:
        if isinstance(result, pd.DataFrame):
            frame = result.copy()
        elif isinstance(result, pd.Series):
            frame = result.rename(metric).reset_index()
            frame.columns = [time_label, metric]
        elif isinstance(result, dict):
            # A mapping of date/time labels to metric values.
            frame = pd.DataFrame({
                time_label: list(result.keys()),
                metric: list(result.values())
            })
        elif isinstance(result, list):
            frame = pd.DataFrame(result)
        else:
            frame = pd.DataFrame()

        if frame.empty:
            return {
                "status": "FAIL",
                "analysis_type": "trend_analysis",
                "metric": metric,
                "time_label": time_label,
                "reason": "Trend result is empty or has an unsupported format."
            }

        # Find time and metric columns using the requested names first.
        time_col = time_label if time_label in frame.columns else None
        metric_col = metric if metric in frame.columns else None

        if metric_col is None:
            numeric_cols = [
                col for col in frame.columns
                if pd.api.types.is_numeric_dtype(frame[col])
            ]
            if len(numeric_cols) == 1:
                metric_col = numeric_cols[0]

        if time_col is None:
            non_metric_cols = [
                col for col in frame.columns if col != metric_col
            ]
            if non_metric_cols:
                time_col = non_metric_cols[0]

        if metric_col is None:
            return {
                "status": "FAIL",
                "analysis_type": "trend_analysis",
                "metric": metric,
                "time_label": time_label,
                "reason": "No numeric metric column was found in the trend result."
            }

        frame[metric_col] = pd.to_numeric(frame[metric_col], errors="coerce")
        frame = frame.dropna(subset=[metric_col])

        if frame.empty:
            return {
                "status": "FAIL",
                "analysis_type": "trend_analysis",
                "metric": metric,
                "time_label": time_label,
                "reason": "Trend result contains no valid numeric metric values."
            }

        if time_col is not None:
            try:
                frame["_parsed_time"] = pd.to_datetime(frame[time_col], errors="coerce")
                if frame["_parsed_time"].notna().any():
                    frame = frame.sort_values("_parsed_time")
            except Exception:
                pass

        values = frame[metric_col].astype(float).tolist()
        first_value = float(values[0])
        last_value = float(values[-1])

        if last_value > first_value:
            direction = "increasing"
        elif last_value < first_value:
            direction = "decreasing"
        else:
            direction = "unchanged"

        supporting_result = {
            "time_label": str(time_col or time_label),
            "metric": str(metric_col),
            "observation_count": int(len(values)),
            "first_value": first_value,
            "last_value": last_value,
            "minimum": float(min(values)),
            "maximum": float(max(values)),
            "direction_first_to_last": direction,
            "records": [
                {
                    "time": str(row[time_col]) if time_col is not None else str(i),
                    "value": float(row[metric_col])
                }
                for i, (_, row) in enumerate(frame.iterrows())
            ]
        }

        return {
            "status": "PASS",
            "analysis_type": "trend_analysis",
            "metric": str(metric_col),
            "time_label": str(time_col or time_label),
            "evidence": supporting_result,
            "supporting_result": supporting_result,
            "evidence_summary": (
                f"{metric_col} has {len(values)} valid observations. "
                f"The first observed value is {first_value:,.2f}, the last is "
                f"{last_value:,.2f}, and the first-to-last direction is {direction}."
            )
        }

    except Exception as exc:
        return {
            "status": "FAIL",
            "analysis_type": "trend_analysis",
            "metric": metric,
            "time_label": time_label,
            "reason": f"Trend evidence could not be created: {exc}"
        }



# ============================================================
# CORRELATION EVIDENCE SUMMARY
# ============================================================

def create_evidence_summary(
    evidence
):

    if not isinstance(
        evidence,
        dict
    ):

        return (
            "Evidence validation failed."
        )

    if (
        evidence.get("status")
        !=
        "PASS"
    ):

        return (
            "Evidence validation failed."
        )

    if (
        evidence.get("analysis_type")
        ==
        "grouped_analysis"
    ):

        return evidence.get(
            "evidence_summary",
            "Evidence summary unavailable."
        )

    if (
        evidence.get("analysis_type")
        ==
        "correlation_analysis"
    ):

        return evidence.get(
            "evidence_summary",
            "Evidence summary unavailable."
        )

    return (
        evidence.get(
            "evidence_summary",
            "Evidence summary unavailable."
        )
    )


# ============================================================
# EXTRACT CLAIMED CATEGORY
# ============================================================

def _extract_claimed_category(
    insight,
    evidence
):

    if not isinstance(
        insight,
        str
    ):

        return None

    category = evidence.get(
        "category"
    )

    evidence_details = evidence.get(
        "evidence",
        {}
    )

    top_category = evidence_details.get(
        "top_category"
    )

    second_category = evidence_details.get(
        "second_category"
    )

    candidates = []

    if top_category is not None:

        candidates.append(
            str(top_category)
        )

    if second_category is not None:

        candidates.append(
            str(second_category)
        )

    supporting_result = evidence.get(
        "supporting_result"
    )

    if isinstance(
        supporting_result,
        dict
    ):

        candidates.extend(
            [
                str(key)
                for key in supporting_result.keys()
            ]
        )

    # --------------------------------------------------------
    # Match the longest category names first.
    # This prevents partial matches.
    # --------------------------------------------------------

    candidates = sorted(
        set(candidates),
        key=len,
        reverse=True
    )

    normalized_insight = (
        insight.lower()
    )

    for candidate in candidates:

        if (
            candidate.lower()
            in
            normalized_insight
        ):

            return candidate

    return None


# ============================================================
# VALIDATE GROUPED INSIGHT AGAINST EVIDENCE
# ============================================================

def _validate_grouped_insight(
    insight,
    evidence
):

    evidence_details = evidence.get(
        "evidence",
        {}
    )

    expected_category = (
        evidence_details.get(
            "top_category"
        )
    )

    expected_value = (
        evidence_details.get(
            "top_value"
        )
    )

    claimed_category = (
        _extract_claimed_category(
            insight,
            evidence
        )
    )

    if expected_category is None:

        return {
            "status":
                "FAIL",

            "reason":
                "Evidence does not contain a top category."
        }

    if claimed_category is None:

        return {
            "status":
                "FAIL",

            "reason":
                (
                    "The insight does not contain "
                    "a category supported by the evidence."
                )
        }

    # --------------------------------------------------------
    # Core semantic contradiction check
    # --------------------------------------------------------

    if (
        claimed_category.lower()
        !=
        str(
            expected_category
        ).lower()
    ):

        return {

            "status":
                "FAIL",

            "reason":
                (
                    "Insight contradicts the evidence. "
                    f"Expected top category: "
                    f"{expected_category}. "
                    f"Insight claims: "
                    f"{claimed_category}."
                ),

            "expected_category":
                expected_category,

            "claimed_category":
                claimed_category,

            "expected_value":
                expected_value
        }

    # --------------------------------------------------------
    # Category agrees with evidence
    # --------------------------------------------------------

    return {

        "status":
            "PASS",

        "reason":
            None,

        "expected_category":
            expected_category,

        "claimed_category":
            claimed_category,

        "expected_value":
            expected_value
    }


# ============================================================
# VALIDATE CORRELATION INSIGHT
# ============================================================

def _validate_correlation_insight(
    insight,
    evidence
):

    if not isinstance(
        insight,
        str
    ):

        return {
            "status":
                "FAIL",
            "reason":
                "Insight is not text."
        }

    correlation = evidence.get(
        "correlation"
    )

    if correlation is None:

        return {
            "status":
                "FAIL",
            "reason":
                "Correlation evidence is missing."
        }

    method = evidence.get(
        "method",
        "Pearson"
    )

    x_variable = evidence.get(
        "x_variable"
    )

    y_variable = evidence.get(
        "y_variable"
    )

    normalized = (
        insight.lower()
    )

    if (
        x_variable
        and
        x_variable.lower()
        not in
        normalized
    ):

        return {
            "status":
                "FAIL",
            "reason":
                (
                    f"Insight does not mention "
                    f"{x_variable}."
                )
        }

    if (
        y_variable
        and
        y_variable.lower()
        not in
        normalized
    ):

        return {
            "status":
                "FAIL",
            "reason":
                (
                    f"Insight does not mention "
                    f"{y_variable}."
                )
        }

    expected_direction = evidence.get(
        "direction"
    )

    expected_strength = evidence.get(
        "strength"
    )

    if (
        expected_direction
        and
        expected_direction.lower()
        not in
        normalized
    ):

        return {
            "status":
                "FAIL",
            "reason":
                (
                    "Insight direction does not "
                    "match the evidence."
                )
        }

    if (
        expected_strength
        and
        expected_strength.lower()
        not in
        normalized
    ):

        return {
            "status":
                "FAIL",
            "reason":
                (
                    "Insight strength does not "
                    "match the evidence."
                )
        }

    return {

        "status":
            "PASS",

        "reason":
            None,

        "method":
            method,

        "correlation":
            correlation,

        "direction":
            expected_direction,

        "strength":
            expected_strength
    }


# ============================================================
# DESCRIPTIVE-SUMMARY CLAIM VALIDATION
# ============================================================

def _parse_claim_number(value):
    try:
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _close_enough(claimed, actual, tolerance=0.03):
    claimed_value = _parse_claim_number(claimed)
    actual_value = _parse_claim_number(actual)
    return claimed_value is not None and actual_value is not None and abs(claimed_value - actual_value) <= max(tolerance, abs(actual_value) * 0.00001)


def _expected_skew_description(skewness):
    value = _parse_claim_number(skewness)
    if value is None:
        return None
    if abs(value) < 0.5:
        return "approximately symmetric"
    return "positively skewed" if value > 0 else "negatively skewed"


def _validate_descriptive_summary_claims(insight, evidence):
    """Validate descriptive-summary claims against independently computed dataset facts."""
    supporting = evidence.get("supporting_result", {})
    facts = supporting.get("dataset_facts") if isinstance(supporting, dict) else None
    if not isinstance(facts, dict) or facts.get("validation_error"):
        return {"status": "FAIL", "reason": "Independent dataset facts are missing; claims cannot be verified.", "checks": [], "verified_claims": 0, "failed_claims": 1}

    text = str(insight or "")
    lines = text.splitlines()
    checks = []

    def record(name, passed, reason):
        checks.append({"claim": name, "status": "PASS" if passed else "FAIL", "reason": None if passed else reason})

    match = re.search(r"Dataset contains\s+([\d,]+)\s+rows and\s+([\d,]+)\s+columns", text, re.I)
    claimed_rows = int(match.group(1).replace(",", "")) if match else None
    claimed_columns = int(match.group(2).replace(",", "")) if match else None
    record("dataset_dimensions", bool(match and claimed_rows == facts.get("row_count") and claimed_columns == facts.get("column_count")), "Reported row/column counts do not match the dataset.")

    columns_line = next((line for line in lines if line.strip().lower().startswith("- columns:")), "")
    missing_columns = [column for column in facts.get("columns", []) if str(column) not in columns_line]
    record("dataset_columns", bool(columns_line and not missing_columns), f"Column list is missing: {missing_columns}" if missing_columns else "Column list is missing.")

    missing_match = re.search(r"Total missing cells:\s*(\d+)", text, re.I)
    record("missing_cells", bool(missing_match and int(missing_match.group(1)) == facts.get("missing_cells")), "Reported missing-cell count does not match the dataset.")
    duplicate_match = re.search(r"Exact duplicate rows:\s*(\d+)", text, re.I)
    record("duplicate_rows", bool(duplicate_match and int(duplicate_match.group(1)) == facts.get("duplicate_rows")), "Reported duplicate-row count does not match the dataset.")

    numeric_summary = facts.get("numeric_summary", {})
    stat_pattern = re.compile(r"valid observations;\s*mean\s+([\d,.-]+);\s*median\s+([\d,.-]+);\s*minimum\s+([\d,.-]+);\s*maximum\s+([\d,.-]+)\.\s*Distribution is\s+(.+?)\s*\(skewness\s+([\d,.-]+)\)", re.I)
    for column, expected in numeric_summary.items():
        line = next((item for item in lines if re.search(r"^\s*-\s*" + re.escape(str(column)) + r":", item)), "")
        found = stat_pattern.search(line)
        passed = False
        reason = f"No complete numerical summary was found for {column}."
        if found:
            mean, median, minimum, maximum, description, skewness = found.groups()
            expected_skew = _expected_skew_description(expected.get("skewness"))
            passed = (
                _close_enough(mean, expected.get("mean"))
                and _close_enough(median, expected.get("median"))
                and _close_enough(minimum, expected.get("min"))
                and _close_enough(maximum, expected.get("max"))
                and _close_enough(skewness, expected.get("skewness"))
                and expected_skew is not None
                and expected_skew in description.lower()
            )
            reason = f"Statistics or skew interpretation for {column} do not match independently computed values."
        record(f"numeric_summary:{column}", passed, reason)

    categorical_summary = facts.get("categorical_summary", {})
    expected_categorical = [column for column in ("Region", "Item Type", "Sales Channel", "Order Priority") if column in categorical_summary]
    for column in expected_categorical:
        line = next((item for item in lines if re.search(r"^\s*-\s*" + re.escape(column) + r":", item)), "")
        expected_values = categorical_summary.get(column, [])
        passed = bool(line)
        missing_values = []
        for item in expected_values:
            value = str(item.get("value", ""))
            count = int(item.get("count", -1))
            if not re.search(re.escape(value) + r"\s*\(\s*" + str(count) + r"\s*\)", line):
                passed = False
                missing_values.append(f"{value} ({count})")
        reason = f"Reported frequencies do not match: {missing_values}" if missing_values else f"Frequency summary for {column} is missing."
        record(f"categorical_frequencies:{column}", passed, reason)

    for column, expected in facts.get("date_ranges", {}).items():
        found = re.search(r"-\s*" + re.escape(str(column)) + r" range:\s*(\d{4}-\d{2}-\d{2})\s+to\s+(\d{4}-\d{2}-\d{2})\s*\((\d+) valid dates\)", text, re.I)
        passed = bool(found and found.group(1) == expected.get("min") and found.group(2) == expected.get("max") and int(found.group(3)) == expected.get("count"))
        record(f"date_range:{column}", passed, f"Reported date range for {column} does not match the dataset.")

    profit_by_region = facts.get("profit_by_region", {})
    if profit_by_region:
        highest_region = max(profit_by_region, key=profit_by_region.get)
        lowest_region = min(profit_by_region, key=profit_by_region.get)
        high_match = re.search(r"Highest total profit by region:\s*(.+?),\s*with\s*([\d,]+(?:\.\d+)?)", text, re.I)
        low_match = re.search(r"Lowest total profit by region:\s*(.+?),\s*with\s*([\d,]+(?:\.\d+)?)", text, re.I)
        high_pass = bool(high_match and high_match.group(1).strip() == highest_region and _close_enough(high_match.group(2), profit_by_region[highest_region]))
        low_pass = bool(low_match and low_match.group(1).strip() == lowest_region and _close_enough(low_match.group(2), profit_by_region[lowest_region]))
        record("highest_regional_profit", high_pass, "Highest-profit region/value does not match the dataset.")
        record("lowest_regional_profit", low_pass, "Lowest-profit region/value does not match the dataset.")

    failures = [item for item in checks if item["status"] != "PASS"]
    return {"status": "PASS" if checks and not failures else "FAIL", "reason": None if not failures else "One or more insight claims failed independent evidence verification.", "checks": checks, "verified_claims": len(checks) - len(failures), "failed_claims": len(failures)}




# ============================================================
# CONVERSION-SPECIFIC INSIGHT VALIDATION
# ============================================================

def _find_conversion_metrics(obj):
    """Find the dictionary containing validated conversion metrics in nested evidence."""
    metric_keys = {
        "conversion_rate_percent", "converted_count_conversion_1",
        "non_converted_count_conversion_0", "total_records",
        "row_by_row_matches", "row_by_row_mismatches"
    }
    if isinstance(obj, dict):
        if metric_keys.intersection(obj.keys()):
            return obj
        for key in ("primary_result", "supporting_result", "evidence", "result"):
            found = _find_conversion_metrics(obj.get(key))
            if found is not None:
                return found
    return None


def _validate_conversion_insight(insight, evidence):
    """Validate only the conversion claims relevant to a conversion-specific insight."""
    metrics = _find_conversion_metrics(evidence)
    if not isinstance(metrics, dict):
        return {
            "status": "FAIL",
            "reason": "Validated conversion metrics are missing from evidence.",
            "checks": [],
            "verified_claims": 0,
            "failed_claims": 1,
        }

    text = str(insight or "")
    lower = text.lower()
    checks = []

    def record(name, passed, reason):
        checks.append({
            "claim": name,
            "status": "PASS" if passed else "FAIL",
            "reason": None if passed else reason,
        })

    def contains_number(expected, tolerance=0.02):
        if expected is None:
            return False
        # Match complete numeric tokens, allowing comma formatting and rounding.
        tokens = re.findall(r"(?<![A-Za-z])[-+]?\d[\d,]*(?:\.\d+)?", text)
        values = []
        for token in tokens:
            try:
                values.append(float(token.replace(",", "")))
            except ValueError:
                pass
        target = float(expected)
        return any(abs(value - target) <= max(tolerance, abs(target) * 0.0001)
                   for value in values)

    # Only require a value if the corresponding claim/topic appears in the insight.
    if re.search(r"\b(total records|records|customers)\b", lower):
        record("total_records",
               contains_number(metrics.get("total_records")),
               "Total-record count does not match validated evidence.")

    if re.search(r"\b(converted customers|converted count|conversion = 1|conversion rate)\b", lower):
        record("converted_count",
               contains_number(metrics.get("converted_count_conversion_1")),
               "Converted-customer count does not match validated evidence.")

    if re.search(r"\b(non.converted customers|non.converted count|conversion = 0|non.conversion rate)\b", lower):
        record("non_converted_count",
               contains_number(metrics.get("non_converted_count_conversion_0")),
               "Non-converted-customer count does not match validated evidence.")

    if "conversion rate" in lower:
        record("conversion_rate",
               contains_number(metrics.get("conversion_rate_percent")),
               "Conversion rate does not match validated evidence.")
    if "non-conversion rate" in lower or "non conversion rate" in lower:
        record("non_conversion_rate",
               contains_number(metrics.get("non_conversion_rate_percent")),
               "Non-conversion rate does not match validated evidence.")

    if re.search(r"\b(row.by.row|mismatch|matches|consistency validation)\b", lower):
        consistency_pass = (
            metrics.get("compared_records") is not None
            and metrics.get("row_by_row_matches") == metrics.get("compared_records")
            and metrics.get("row_by_row_mismatches") == 0
            and metrics.get("unmatched_conversion_values", 0) == 0
            and metrics.get("unmapped_y_values", 0) == 0
        )
        record("conversion_column_consistency", consistency_pass,
               "The conversion and y columns do not fully match after yes/no mapping.")

    # Conversion-specific insights should mention the main metric or counts.
    if not checks:
        record("conversion_topic", "conversion" in lower,
               "Insight does not mention the conversion analysis.")

    failures = [item for item in checks if item["status"] != "PASS"]
    return {
        "status": "PASS" if checks and not failures else "FAIL",
        "reason": None if not failures else "One or more conversion claims failed evidence verification.",
        "checks": checks,
        "verified_claims": len(checks) - len(failures),
        "failed_claims": len(failures),
    }


# ============================================================
# MAP INSIGHT TO EVIDENCE
# ============================================================

def map_insight_to_evidence(
    insight,
    evidence
):

    if not isinstance(
        evidence,
        dict
    ):

        return {

            "status":
                "FAIL",

            "analysis_type":
                "unknown",

            "insight":
                insight,

            "reason":
                "Evidence must be a dictionary."
        }

    if (
        evidence.get("status")
        !=
        "PASS"
    ):

        return {

            "status":
                "FAIL",

            "analysis_type":
                evidence.get(
                    "analysis_type",
                    "unknown"
                ),

            "insight":
                insight,

            "reason":
                evidence.get(
                    "reason",
                    "Evidence is invalid."
                )
        }

    analysis_type = evidence.get(
        "analysis_type"
    )
    # --------------------------------------------------------
    # FILTERED MEAN EVIDENCE
    # --------------------------------------------------------
    # The main pipeline can wrap the validated result as:
    # evidence["result"] -> primary_result, or
    # evidence["supporting_result"]["primary_result"].
    # Search only known result containers and use the validated
    # filtered_mean payload itself as the source of its values.
    def find_filtered_mean(obj, depth=0):
        if depth > 6 or not isinstance(obj, dict):
            return None
        if (
            obj.get("analysis_type") == "filtered_mean"
            and obj.get("mean_duration") is not None
        ):
            return obj
        for key in ("primary_result", "result", "supporting_result", "evidence", "validated_result"):
            child = obj.get(key)
            found = find_filtered_mean(child, depth + 1)
            if found is not None:
                return found
        return None

    filtered_mean_result = find_filtered_mean(evidence)
    if analysis_type == "filtered_mean" or filtered_mean_result is not None:
        source = filtered_mean_result if filtered_mean_result is not None else evidence
        month_value = source.get("filter_value", evidence.get("filter_value"))
        mean_value = source.get("mean_duration", evidence.get("mean_duration"))
        rows = source.get("filtered_rows", evidence.get("filtered_rows"))
        valid_count = source.get("valid_duration_count", evidence.get("valid_duration_count"))
        text = str(insight or "").lower()
        checks = []
        if month_value:
            month_names = {
                "jan": "january", "feb": "february", "mar": "march",
                "apr": "april", "may": "may", "jun": "june",
                "jul": "july", "aug": "august", "sep": "september",
                "oct": "october", "nov": "november", "dec": "december",
            }
            expected_month = month_names.get(str(month_value).lower(), str(month_value).lower())
            checks.append({
                "check": "requested_month_mentioned",
                "status": "PASS" if expected_month in text else "FAIL",
            })
        try:
            mean_mentioned = (
                mean_value is not None
                and "duration" in text
                and f"{float(mean_value):.2f}" in text
            )
        except (TypeError, ValueError):
            mean_mentioned = False
        checks.append({
            "check": "mean_duration_mentioned",
            "status": "PASS" if mean_mentioned else "FAIL",
        })
        failed = [item for item in checks if item["status"] != "PASS"]
        return {
            "status": "PASS" if checks and not failed else "FAIL",
            "analysis_type": "filtered_mean",
            "insight": insight,
            "filter_value": month_value,
            "mean_duration": mean_value,
            "filtered_rows": rows,
            "valid_duration_count": valid_count,
            "checks": checks,
            "reason": None if not failed else "Filtered mean insight did not match its evidence.",
        }


    # --------------------------------------------------------
    # GROUPED MEAN ANALYSIS
    # --------------------------------------------------------
    # Validates comparison insights using group means and the observed
    # difference. This is deliberately separate from grouped_analysis,
    # which validates category rankings rather than numeric means.
    def find_grouped_mean_result(obj, depth=0):
        if depth > 7 or not isinstance(obj, dict):
            return None
        if (
            obj.get("analysis_type") == "grouped_mean_analysis"
            and isinstance(obj.get("group_results"), dict)
        ):
            return obj
        for key in (
            "primary_result", "result", "supporting_result",
            "evidence", "validated_result"
        ):
            found = find_grouped_mean_result(obj.get(key), depth + 1)
            if found is not None:
                return found
        return None

    grouped_mean_result = find_grouped_mean_result(evidence)
    if analysis_type == "grouped_mean_analysis" or grouped_mean_result is not None:
        source = grouped_mean_result if grouped_mean_result is not None else evidence
        groups = source.get("group_results", {})
        # Bank-marketing subscription comparison: no = non-subscribers,
        # yes = subscribers. Accept common casing in group labels.
        no_group = next((v for k, v in groups.items() if str(k).strip().lower() in ("no", "not subscribed", "non-subscribers", "non_subscribers")), None)
        yes_group = next((v for k, v in groups.items() if str(k).strip().lower() in ("yes", "subscribed", "subscribers")), None)

        def get_group_mean(group):
            if not isinstance(group, dict):
                return None
            for key in ("mean_balance", "mean", "average", "mean_value"):
                if group.get(key) is not None:
                    return _parse_claim_number(group.get(key))
            return None

        no_mean = get_group_mean(no_group)
        yes_mean = get_group_mean(yes_group)
        difference = source.get("mean_difference_yes_minus_no")
        if difference is None and no_mean is not None and yes_mean is not None:
            difference = yes_mean - no_mean
        difference = _parse_claim_number(difference)
        insight_text = str(insight or "")
        insight_lower = insight_text.lower()
        checks = []
        groups_available = no_mean is not None and yes_mean is not None
        checks.append({
            "check": "both_group_means_available",
            "status": "PASS" if groups_available else "FAIL",
        })
        no_mean_mentioned = groups_available and _close_enough(
            next((n for n in (_parse_claim_number(m.group(0)) for m in re.finditer(r"(?<![A-Za-z])[-+]?\d[\d,]*(?:\.\d+)?", insight_text)) if _close_enough(n, no_mean)), None), no_mean
        )
        yes_mean_mentioned = groups_available and _close_enough(
            next((n for n in (_parse_claim_number(m.group(0)) for m in re.finditer(r"(?<![A-Za-z])[-+]?\d[\d,]*(?:\.\d+)?", insight_text)) if _close_enough(n, yes_mean)), None), yes_mean
        )
        # Require both numeric values and recognizable group descriptions.
        checks.append({
            "check": "non_subscriber_mean_mentioned",
            "status": "PASS" if no_mean_mentioned and ("non-subscrib" in insight_lower or "not subscribed" in insight_lower) else "FAIL",
        })
        checks.append({
            "check": "subscriber_mean_mentioned",
            "status": "PASS" if yes_mean_mentioned and "subscrib" in insight_lower else "FAIL",
        })
        if difference is not None:
            numbers = [
                _parse_claim_number(m.group(0))
                for m in re.finditer(r"(?<![A-Za-z])[-+]?\d[\d,]*(?:\.\d+)?", insight_text)
            ]
            diff_mentioned = any(_close_enough(n, difference) for n in numbers if n is not None)
            checks.append({
                "check": "mean_difference_mentioned",
                "status": "PASS" if diff_mentioned else "FAIL",
            })
        failed_checks = [item for item in checks if item["status"] != "PASS"]
        return {
            "status": "PASS" if checks and not failed_checks else "FAIL",
            "analysis_type": "grouped_mean_analysis",
            "insight": insight,
            "group_results": groups,
            "mean_difference_yes_minus_no": difference,
            "checks": checks,
            "evidence_summary": create_evidence_summary(evidence),
            "reason": None if not failed_checks else "The grouped-mean insight does not match the validated group means and difference.",
        }

    # --------------------------------------------------------
    # MEAN ANALYSIS
    # --------------------------------------------------------
    if analysis_type == "mean_analysis":

        def find_mean_result(obj, depth=0):
            """Find the validated mean payload inside known result containers."""
            if depth > 6 or not isinstance(obj, dict):
                return None

            if (
                obj.get("analysis_type") == "mean_analysis"
                and obj.get("mean") is not None
            ):
                return obj

            for key in (
                "primary_result",
                "result",
                "supporting_result",
                "evidence",
                "validated_result",
            ):
                found = find_mean_result(obj.get(key), depth + 1)
                if found is not None:
                    return found
            return None

        mean_result = find_mean_result(evidence)
        text = str(insight or "")
        normalized_text = " ".join(
            text.lower().replace("_", " ").split()
        )

        if mean_result is None:
            return {
                "status": "FAIL",
                "analysis_type": "mean_analysis",
                "insight": insight,
                "evidence_summary": create_evidence_summary(evidence),
                "reason": "No validated mean result was found in the evidence.",
                "checks": [{
                    "check": "validated_mean_available",
                    "status": "FAIL",
                }],
            }

        metric = mean_result.get("metric_column")
        expected_mean = _parse_claim_number(mean_result.get("mean"))
        checks = []

        metric_text = " ".join(
            str(metric).replace("_", " ").lower().split()
        ) if metric is not None else ""
        metric_pattern = (
            r"(?<![\w])" + re.escape(metric_text) + r"(?![\w])"
            if metric_text else None
        )
        metric_ok = bool(
            metric_pattern and re.search(metric_pattern, normalized_text)
        )
        checks.append({
            "check": "metric_column_mentioned",
            "status": "PASS" if metric_ok else "FAIL",
        })

        # Parse complete numeric tokens, including comma-formatted values.
        numeric_tokens = re.findall(
            r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?",
            text,
        )
        claimed_values = []
        for token in numeric_tokens:
            parsed = _parse_claim_number(token)
            if parsed is not None:
                claimed_values.append(parsed)

        mean_words_present = bool(
            re.search(r"\b(mean|average)\b", normalized_text)
        )
        mean_ok = (
            expected_mean is not None
            and mean_words_present
            and any(round(value, 2) == round(expected_mean, 2)
                    for value in claimed_values)
        )
        checks.append({
            "check": "reported_mean_matches_validated_result",
            "status": "PASS" if mean_ok else "FAIL",
        })

        # If the insight states an observation count, validate it as well.
        valid_count = mean_result.get("valid_value_count")
        if re.search(r"\b(observations?|valid values?|numeric values?)\b", normalized_text):
            count_tokens = re.findall(
                r"\b(\d[\d,]*)\s+(?:valid\s+)?(?:numeric\s+)?(?:observations?|values?)\b",
                normalized_text,
            )
            count_values = []
            for token in count_tokens:
                try:
                    count_values.append(int(token.replace(",", "")))
                except ValueError:
                    pass
            count_ok = valid_count is not None and int(valid_count) in count_values
            checks.append({
                "check": "reported_valid_count_matches_evidence",
                "status": "PASS" if count_ok else "FAIL",
            })

        failed_checks = [item for item in checks if item["status"] != "PASS"]
        return {
            "status": "PASS" if not failed_checks else "FAIL",
            "analysis_type": "mean_analysis",
            "insight": insight,
            "metric_column": metric,
            "validated_mean": expected_mean,
            "valid_value_count": valid_count,
            "evidence_summary": create_evidence_summary(evidence),
            "supporting_result": mean_result,
            "checks": checks,
            "reason": (
                None if not failed_checks
                else "The mean insight does not match the validated evidence."
            ),
        }


    # --------------------------------------------------------
    # GROUPED ANALYSIS
    # --------------------------------------------------------

    if (
        analysis_type
        ==
        "grouped_analysis"
    ):

        validation = (
            _validate_grouped_insight(
                insight,
                evidence
            )
        )

        result = {

            "status":
                validation.get(
                    "status"
                ),

            "analysis_type":
                analysis_type,

            "insight":
                insight,

            "evidence_summary":
                create_evidence_summary(
                    evidence
                ),

            "top_category":
                evidence.get(
                    "evidence",
                    {}
                ).get(
                    "top_category"
                ),

            "top_value":
                evidence.get(
                    "evidence",
                    {}
                ).get(
                    "top_value"
                ),

            "supporting_result":
                evidence.get(
                    "supporting_result"
                )
        }

        if validation.get(
            "status"
        ) == "FAIL":

            result["reason"] = (
                validation.get(
                    "reason"
                )
            )

            result[
                "expected_category"
            ] = validation.get(
                "expected_category"
            )

            result[
                "claimed_category"
            ] = validation.get(
                "claimed_category"
            )

        return result

    # --------------------------------------------------------
    # CONVERSION-FACTOR ANALYSIS
    # --------------------------------------------------------
    if analysis_type == "conversion_factor_analysis":
        source = evidence.get("supporting_result", evidence.get("result"))
        if not isinstance(source, dict):
            return {
                "status": "FAIL",
                "analysis_type": analysis_type,
                "insight": insight,
                "evidence_summary": create_evidence_summary(evidence),
                "reason": "Structured conversion-factor evidence is missing.",
            }

        # Support the normal evidence shape and a wrapped primary-result shape.
        if source.get("analysis_type") != "conversion_factor_analysis":
            for key in ("primary_result", "result", "supporting_result"):
                nested = source.get(key)
                if isinstance(nested, dict) and nested.get("analysis_type") == "conversion_factor_analysis":
                    source = nested
                    break

        categorical = source.get("categorical_factor_results")
        numeric = source.get("numeric_factor_results")
        valid = (
            source.get("analysis_type") == "conversion_factor_analysis"
            and isinstance(categorical, dict)
            and isinstance(numeric, dict)
            and isinstance(source.get("overall_conversion_rate_percent"), (int, float))
        )
        text_lower = str(insight or "").lower()
        legacy_error = "no valid correlation result" in text_lower
        status = "PASS" if valid and not legacy_error and str(insight or "").strip() else "FAIL"
        reason = None
        if not valid:
            reason = "Conversion-factor evidence does not contain the required validated result structure."
        elif legacy_error:
            reason = "The insight still contains the legacy correlation-analysis fallback message."
        elif not str(insight or "").strip():
            reason = "Insight text is empty."

        return {
            "status": status,
            "analysis_type": analysis_type,
            "insight": insight,
            "evidence_summary": create_evidence_summary(evidence),
            "supporting_result": source,
            "claim_validation": {
                "status": status,
                "checks": {
                    "structured_conversion_factor_evidence": valid,
                    "legacy_correlation_fallback_absent": not legacy_error,
                    "insight_present": bool(str(insight or "").strip()),
                },
            },
            "reason": reason,
        }

    # --------------------------------------------------------
    # CORRELATION ANALYSIS
    # --------------------------------------------------------

    if (
        analysis_type
        ==
        "correlation_analysis"
    ):

        validation = (
            _validate_correlation_insight(
                insight,
                evidence
            )
        )

        result = {

            "status":
                validation.get(
                    "status"
                ),

            "analysis_type":
                analysis_type,

            "insight":
                insight,

            "evidence_summary":
                create_evidence_summary(
                    evidence
                ),

            "x_variable":
                evidence.get(
                    "x_variable"
                ),

            "y_variable":
                evidence.get(
                    "y_variable"
                ),

            "method":
                evidence.get(
                    "method"
                ),

            "correlation":
                evidence.get(
                    "correlation"
                ),

            "direction":
                evidence.get(
                    "direction"
                ),

            "strength":
                evidence.get(
                    "strength"
                )
        }

        if validation.get(
            "status"
        ) == "FAIL":

            result["reason"] = (
                validation.get(
                    "reason"
                )
            )

        return result

    # --------------------------------------------------------
    # GENERIC / DESCRIPTIVE SUMMARY
    # --------------------------------------------------------

    if analysis_type in ("trend_analysis", "time_series_analysis", "timeseries_analysis"):
        trend_evidence = evidence.get("evidence", evidence.get("supporting_result", {}))
        if not isinstance(trend_evidence, dict) or not trend_evidence:
            return {
                "status": "FAIL",
                "analysis_type": analysis_type,
                "insight": insight,
                "evidence_summary": create_evidence_summary(evidence),
                "reason": "No structured trend evidence is available to validate the insight.",
            }

        # Check explicit first/last direction claims when present.
        text_lower = str(insight).lower()
        expected_direction = trend_evidence.get("direction_first_to_last")
        direction_words = {
            "increasing": ("increasing", "increased", "rising", "grew", "growth"),
            "decreasing": ("decreasing", "decreased", "falling", "declined", "decline"),
            "unchanged": ("unchanged", "remained stable", "stayed the same", "constant"),
        }
        mentioned = [
            direction for direction, words in direction_words.items()
            if any(word in text_lower for word in words)
        ]
        if mentioned and expected_direction not in mentioned:
            return {
                "status": "FAIL",
                "analysis_type": analysis_type,
                "insight": insight,
                "evidence_summary": create_evidence_summary(evidence),
                "supporting_result": trend_evidence,
                "reason": (
                    f"Trend direction claim does not match the evidence; "
                    f"expected '{expected_direction}'."
                ),
            }
        return {
            "status": "PASS",
            "analysis_type": analysis_type,
            "insight": insight,
            "evidence_summary": create_evidence_summary(evidence),
            "supporting_result": trend_evidence,
            "claim_validation": {
                "status": "PASS",
                "checked": "trend direction when explicitly stated",
                "evidence_direction": expected_direction,
            },
        }

    if analysis_type == "descriptive_summary":
        conversion_metrics = _find_conversion_metrics(evidence)
        if conversion_metrics is not None:
            validation = _validate_conversion_insight(insight, evidence)
        else:
            validation = _validate_descriptive_summary_claims(insight, evidence)
        return {
            "status": validation.get("status", "FAIL"),
            "analysis_type": analysis_type,
            "insight": insight,
            "evidence_summary": create_evidence_summary(evidence),
            "supporting_result": evidence.get("supporting_result"),
            "claim_validation": validation,
            "reason": validation.get("reason"),
        }

    # Avoid false PASS results when no dedicated claim validator exists.
    return {
        "status": "PARTIAL",
        "analysis_type": analysis_type or "generic",
        "insight": insight,
        "evidence_summary": create_evidence_summary(evidence),
        "supporting_result": evidence.get("supporting_result"),
        "reason": "Evidence was collected, but no dedicated validator exists for this analysis type.",
    }
