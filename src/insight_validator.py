import re


# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

def _safe_float(value, default=None):
    """
    Safely convert a value to float.
    """

    try:
        return float(value)

    except (TypeError, ValueError):
        return default


def _approximately_equal(
    claimed,
    actual,
    absolute_tolerance=0.01,
    relative_tolerance=0.0001
):
    """
    Check whether two numerical values are approximately equal.
    """

    claimed = _safe_float(claimed)
    actual = _safe_float(actual)

    if claimed is None or actual is None:
        return False

    difference = abs(claimed - actual)

    allowed_difference = max(
        absolute_tolerance,
        abs(actual) * relative_tolerance
    )

    return difference <= allowed_difference


def _contains_phrase(text, phrase):
    """
    Case-insensitive phrase check.
    """

    return str(phrase).lower() in str(text).lower()


# ==========================================================
# INSIGHT PRESENCE
# ==========================================================

def validate_insight_presence(insight):
    """
    Check whether an insight exists and contains text.
    """

    if insight is None:
        return {
            "pass": False,
            "reason": "Insight is None."
        }

    text = str(insight).strip()

    if not text:
        return {
            "pass": False,
            "reason": "Insight is empty."
        }

    return {
        "pass": True,
        "reason": None
    }


# ==========================================================
# RESULT PRESENCE
# ==========================================================

def validate_result_presence(result):
    """
    Check whether a result exists.
    """

    if result is None:
        return {
            "pass": False,
            "reason": "Result is None."
        }

    if isinstance(result, dict):
        status = str(result.get("status", "")).strip().upper()
        if status in {"UNSUPPORTED", "NOT_SUPPORTED", "FAIL", "FAILED", "ERROR"}:
            return {
                "pass": False,
                "reason": (
                    "Result is not a successful supported analysis: "
                    + str(result.get("reason") or status)
                )[:500],
            }

    return {
        "pass": True,
        "reason": None
    }


# ==========================================================
# ANOMALY CLAIM EXTRACTION
# ==========================================================

def extract_anomaly_claims(insight):
    """
    Extract numerical claims from anomaly insight.
    """

    text = str(insight)

    claims = {}

    # ------------------------------------------------------
    # Observation count
    # ------------------------------------------------------

    match = re.search(
        r"evaluated\s+([\d,]+)\s+Total Profit observations",
        text,
        re.IGNORECASE
    )

    if match:
        claims["count"] = int(
            match.group(1).replace(",", "")
        )

    # ------------------------------------------------------
    # Mean
    # ------------------------------------------------------

    match = re.search(
        r"mean Total Profit was\s+"
        r"([-+]?[\d,]+(?:\.\d+)?)",
        text,
        re.IGNORECASE
    )

    if match:
        claims["mean"] = float(
            match.group(1).replace(",", "")
        )

    # ------------------------------------------------------
    # Median
    # ------------------------------------------------------

    match = re.search(
        r"median was\s+"
        r"([-+]?[\d,]+(?:\.\d+)?)",
        text,
        re.IGNORECASE
    )

    if match:
        claims["median"] = float(
            match.group(1).replace(",", "")
        )

    # ------------------------------------------------------
    # Minimum / Maximum
    # ------------------------------------------------------

    match = re.search(
        r"ranged from\s+"
        r"([-+]?[\d,]+(?:\.\d+)?)"
        r"\s+to\s+"
        r"([-+]?[\d,]+(?:\.\d+)?)",
        text,
        re.IGNORECASE
    )

    if match:
        claims["min"] = float(
            match.group(1).replace(",", "")
        )

        claims["max"] = float(
            match.group(2).replace(",", "")
        )

    # ------------------------------------------------------
    # Q1 / Q3 / IQR
    # ------------------------------------------------------

    match = re.search(
        r"Q1 was\s+"
        r"([-+]?[\d,]+(?:\.\d+)?)"
        r",\s*Q3 was\s+"
        r"([-+]?[\d,]+(?:\.\d+)?)"
        r",\s*and the IQR was\s+"
        r"([-+]?[\d,]+(?:\.\d+)?)",
        text,
        re.IGNORECASE
    )

    if match:
        claims["Q1"] = float(
            match.group(1).replace(",", "")
        )

        claims["Q3"] = float(
            match.group(2).replace(",", "")
        )

        claims["IQR"] = float(
            match.group(3).replace(",", "")
        )

    # ------------------------------------------------------
    # Lower / Upper boundaries
    # ------------------------------------------------------

    match = re.search(
        r"lower and upper boundaries were\s+"
        r"([-+]?[\d,]+(?:\.\d+)?)"
        r"\s+and\s+"
        r"([-+]?[\d,]+(?:\.\d+)?)",
        text,
        re.IGNORECASE
    )

    if match:
        claims["lower_bound"] = float(
            match.group(1).replace(",", "")
        )

        claims["upper_bound"] = float(
            match.group(2).replace(",", "")
        )

    # ------------------------------------------------------
    # Outlier count
    # ------------------------------------------------------

    match = re.search(
        r"identified\s+([\d,]+)"
        r"\s+potential Total Profit outliers",
        text,
        re.IGNORECASE
    )

    if match:
        claims["outlier_count"] = int(
            match.group(1).replace(",", "")
        )

    # ------------------------------------------------------
    # Outlier range
    # ------------------------------------------------------

    match = re.search(
        r"outlier values ranged from\s+"
        r"([-+]?[\d,]+(?:\.\d+)?)"
        r"\s+to\s+"
        r"([-+]?[\d,]+(?:\.\d+)?)",
        text,
        re.IGNORECASE
    )

    if match:
        claims["outlier_min"] = float(
            match.group(1).replace(",", "")
        )

        claims["outlier_max"] = float(
            match.group(2).replace(",", "")
        )

    # ------------------------------------------------------
    # Above upper boundary
    # ------------------------------------------------------

    match = re.search(
        r"All\s+([\d,]+)"
        r"\s+identified potential outliers were above "
        r"the upper IQR boundary",
        text,
        re.IGNORECASE
    )

    if match:
        claims["above_upper_count"] = int(
            match.group(1).replace(",", "")
        )

    claims["all_outliers_above_upper_bound"] = (
        "above the upper IQR boundary"
        in text.lower()
    )

    return claims


# ==========================================================
# ANOMALY CLAIM VALIDATION
# ==========================================================

def validate_anomaly_claims(
    insight,
    result
):
    """
    Validate anomaly claims against the actual result.
    """

    if not isinstance(result, dict):
        return {
            "overall": "FAIL",
            "pass": False,
            "total_claims": 0,
            "passed_claims": 0,
            "failed_claims": 0,
            "reason": "Anomaly result is not a dictionary."
        }

    claims = extract_anomaly_claims(
        insight
    )

    checks = []

    # ------------------------------------------------------
    # Count
    # ------------------------------------------------------

    if "count" in claims:

        actual_count = result.get(
            "count"
        )

        checks.append(
            (
                "count",
                claims["count"] == int(
                    actual_count
                )
            )
        )

    # ------------------------------------------------------
    # Mean
    # ------------------------------------------------------

    if "mean" in claims:

        checks.append(
            (
                "mean",
                _approximately_equal(
                    claims["mean"],
                    result.get("mean")
                )
            )
        )

    # ------------------------------------------------------
    # Median
    # ------------------------------------------------------

    if "median" in claims:

        checks.append(
            (
                "median",
                _approximately_equal(
                    claims["median"],
                    result.get("median")
                )
            )
        )

    # ------------------------------------------------------
    # Minimum
    # ------------------------------------------------------

    if "min" in claims:

        checks.append(
            (
                "min",
                _approximately_equal(
                    claims["min"],
                    result.get("min")
                )
            )
        )

    # ------------------------------------------------------
    # Maximum
    # ------------------------------------------------------

    if "max" in claims:

        checks.append(
            (
                "max",
                _approximately_equal(
                    claims["max"],
                    result.get("max")
                )
            )
        )

    # ------------------------------------------------------
    # Q1 / Q3 / IQR
    # ------------------------------------------------------

    for key in [
        "Q1",
        "Q3",
        "IQR"
    ]:

        if key in claims:

            checks.append(
                (
                    key,
                    _approximately_equal(
                        claims[key],
                        result.get(key)
                    )
                )
            )

    # ------------------------------------------------------
    # Bounds
    # ------------------------------------------------------

    for key in [
        "lower_bound",
        "upper_bound"
    ]:

        if key in claims:

            checks.append(
                (
                    key,
                    _approximately_equal(
                        claims[key],
                        result.get(key)
                    )
                )
            )

    # ------------------------------------------------------
    # Outlier count
    # ------------------------------------------------------

    if "outlier_count" in claims:

        checks.append(
            (
                "outlier_count",
                claims["outlier_count"]
                ==
                int(
                    result.get(
                        "outlier_count",
                        -1
                    )
                )
            )
        )

    # ------------------------------------------------------
    # Outlier range
    # ------------------------------------------------------

    actual_outliers = result.get(
        "outlier_values",
        []
    )

    numeric_outliers = []

    for value in actual_outliers:

        numeric_value = _safe_float(
            value
        )

        if numeric_value is not None:
            numeric_outliers.append(
                numeric_value
            )

    if numeric_outliers:

        actual_min = min(
            numeric_outliers
        )

        actual_max = max(
            numeric_outliers
        )

        if (
            "outlier_min" in claims
            and
            "outlier_max" in claims
        ):

            checks.append(
                (
                    "outlier_range",
                    _approximately_equal(
                        claims["outlier_min"],
                        actual_min
                    )
                    and
                    _approximately_equal(
                        claims["outlier_max"],
                        actual_max
                    )
                )
            )

    # ------------------------------------------------------
    # Above upper boundary
    # ------------------------------------------------------

    actual_upper_bound = result.get(
        "upper_bound"
    )

    actual_above_upper = 0

    if actual_upper_bound is not None:

        for value in numeric_outliers:

            if value > float(
                actual_upper_bound
            ):

                actual_above_upper += 1

    if "above_upper_count" in claims:

        checks.append(
            (
                "above_upper_count",
                claims["above_upper_count"]
                ==
                actual_above_upper
            )
        )

    # ------------------------------------------------------
    # All outliers above upper boundary
    # ------------------------------------------------------

    if claims.get(
        "all_outliers_above_upper_bound"
    ):

        checks.append(
            (
                "all_outliers_above_upper_bound",
                actual_above_upper
                ==
                len(numeric_outliers)
            )
        )

    # ------------------------------------------------------
    # Summary
    # ------------------------------------------------------

    total_claims = len(
        checks
    )

    passed_claims = sum(
        1
        for _, status in checks
        if status
    )

    failed_claims = (
        total_claims
        -
        passed_claims
    )

    overall_pass = (
        total_claims > 0
        and
        failed_claims == 0
    )

    return {
        "overall": (
            "PASS"
            if overall_pass
            else "FAIL"
        ),

        "pass": overall_pass,

        "total_claims": total_claims,

        "passed_claims": passed_claims,

        "failed_claims": failed_claims,

        "checks": {
            name: status
            for name, status in checks
        },

        "claims": claims
    }


# ==========================================================
# CORRELATION VALIDATION
# ==========================================================

def validate_correlation_insight(
    insight,
    result
):
    """
    Validate correlation insight.

    The relationship-analysis primary result
    is a scalar Pearson correlation value.
    """

    text = str(
        insight
    )

    # ------------------------------------------------------
    # Accept scalar OR dictionary
    # ------------------------------------------------------

    if isinstance(
        result,
        dict
    ):

        actual = result.get(
            "pearson_correlation",
            result.get(
                "pearson"
            )
        )

    else:

        actual = result

    actual = _safe_float(
        actual
    )

    if actual is None:

        return {
            "pass": False,
            "reason": (
                "Correlation result is not numeric."
            )
        }

    # ------------------------------------------------------
    # Required variables
    # ------------------------------------------------------

    variables_pass = (
        _contains_phrase(
            text,
            "Units Sold"
        )
        and
        _contains_phrase(
            text,
            "Total Profit"
        )
    )

    if not variables_pass:

        return {
            "pass": False,
            "reason": (
                "Insight does not identify both "
                "Units Sold and Total Profit."
            )
        }

    # ------------------------------------------------------
    # Extract reported correlation
    # ------------------------------------------------------

    match = re.search(
        r"correlation.*?"
        r"(?:approximately|about|of|was|is)"
        r"\s+([-+]?(?:\d+(?:\.\d+)?))",
        text,
        re.IGNORECASE
    )

    claimed = None

    if match:

        claimed = _safe_float(
            match.group(1)
        )

    if claimed is None:

        return {
            "pass": False,
            "reason": (
                "No correlation value was found "
                "in the insight."
            )
        }

    # ------------------------------------------------------
    # Numerical validation
    # ------------------------------------------------------

    numerical_pass = _approximately_equal(
        claimed,
        actual,
        absolute_tolerance=0.01,
        relative_tolerance=0.001
    )

    if not numerical_pass:

        return {
            "pass": False,
            "reason": (
                "Reported correlation does not "
                "match the validated result."
            ),
            "claimed": claimed,
            "actual": actual
        }

    # ------------------------------------------------------
    # Direction validation
    # ------------------------------------------------------

    if actual > 0:

        direction_pass = (
            "positive"
            in text.lower()
        )

    elif actual < 0:

        direction_pass = (
            "negative"
            in text.lower()
        )

    else:

        direction_pass = (
            "no"
            in text.lower()
        )

    # ------------------------------------------------------
    # Strength validation
    # ------------------------------------------------------

    absolute_value = abs(
        actual
    )

    if (
        0.40
        <= absolute_value
        < 0.60
    ):

        strength_pass = (
            "moderate"
            in text.lower()
        )

        strength = "moderate"

    elif (
        0.60
        <= absolute_value
        < 0.80
    ):

        strength_pass = (
            "strong"
            in text.lower()
        )

        strength = "strong"

    elif absolute_value < 0.40:

        strength_pass = (
            "weak"
            in text.lower()
        )

        strength = "weak"

    else:

        strength_pass = (
            "very strong"
            in text.lower()
            or
            "strong"
            in text.lower()
        )

        strength = "very strong"

    overall_pass = (
        variables_pass
        and
        numerical_pass
        and
        direction_pass
        and
        strength_pass
    )

    return {
        "pass": overall_pass,

        "reason": (
            None
            if overall_pass
            else
            "Correlation insight failed "
            "one or more validation checks."
        ),

        "claimed": claimed,

        "actual": actual,

        "direction": (
            "positive"
            if actual > 0
            else
            "negative"
            if actual < 0
            else
            "none"
        ),

        "strength": strength
    }


# ==========================================================
# GROUPED NUMERIC COMPARISON CLAIM VALIDATION
# ==========================================================

def validate_grouped_mean_direction_claims(insight, result, user_question=None):
    """
    Validate generated insight direction separately from the user's claim.
    A correct insight passes even when it disproves the original user claim.
    """
    if not isinstance(result, dict):
        return {
            "pass": False, "overall": "FAIL",
            "reason": "Grouped result is not a dictionary.", "checks": [],
            "user_claim_validation": {
                "status": "REVIEW", "pass": None,
                "reason": "Grouped result is not a dictionary.", "checks": [],
            },
        }

    def numeric(value):
        try:
            value = float(value)
            return value if value == value and abs(value) != float("inf") else None
        except (TypeError, ValueError, OverflowError):
            return None

    metric_results = result.get("metric_results")
    candidates = (
        [(str(metric), item) for metric, item in metric_results.items()]
        if isinstance(metric_results, dict) and metric_results
        else [(str(result.get("metric_column", "")), result)]
    )
    insight_checks = []
    user_checks = []

    for metric, item in candidates:
        if not isinstance(item, dict):
            continue
        groups = item.get("group_results")
        if not isinstance(groups, dict):
            continue
        no_group, yes_group = groups.get("no", {}), groups.get("yes", {})
        if not isinstance(no_group, dict) or not isinstance(yes_group, dict):
            continue
        no_mean = numeric(no_group.get("mean_value", no_group.get("mean")))
        yes_mean = numeric(yes_group.get("mean_value", yes_group.get("mean")))
        if no_mean is None or yes_mean is None:
            insight_checks.append({
                "metric": metric, "pass": False,
                "reason": "Both subscriber and non-subscriber means must be finite numbers.",
            })
            continue

        actual_direction = (
            "lower" if yes_mean < no_mean
            else "higher" if yes_mean > no_mean
            else "equal"
        )
        sources = (
            ("insight", str(insight or "").lower(), insight_checks),
            ("user_question", str(user_question or "").lower(), user_checks),
        )
        for source_name, source_text, target_checks in sources:
            mentions_subscribers = any(
                term in source_text for term in ("subscriber", "subscribers", "subscription")
            )
            mentions_non_subscribers = any(
                term in source_text for term in (
                    "non-subscriber", "non subscriber", "non-subscribers",
                    "non subscribers", "non subscriber group",
                )
            )
            has_comparison = any(
                term in source_text for term in (
                    "lower", "higher", "less than", "greater than", "more than",
                    "less", "greater", "exceed", "exceeds", "below", "above",
                )
            )
            if not (mentions_subscribers and mentions_non_subscribers and has_comparison):
                continue

            lower_claim = any(
                term in source_text for term in ("lower", "less than", "less", "below")
            )
            higher_claim = any(
                term in source_text for term in (
                    "higher", "greater than", "greater", "more than", "exceed", "exceeds", "above"
                )
            )
            if lower_claim == higher_claim:
                continue

            claimed_direction = "lower" if lower_claim else "higher"
            claim_pass = (
                (lower_claim and yes_mean < no_mean)
                or (higher_claim and yes_mean > no_mean)
            )
            target_checks.append({
                "source": source_name,
                "metric": metric,
                "claimed_direction": claimed_direction,
                "actual_direction": actual_direction,
                "subscriber_mean": yes_mean,
                "non_subscriber_mean": no_mean,
                "pass": bool(claim_pass),
                "reason": None if claim_pass else (
                    f"The {source_name.replace('_', ' ')} says subscribers are "
                    f"{claimed_direction}, but the calculated means are "
                    f"subscribers={yes_mean:.2f}, non-subscribers={no_mean:.2f}."
                ),
            })

    insight_failures = [item for item in insight_checks if not item.get("pass", False)]
    user_failures = [item for item in user_checks if not item.get("pass", False)]
    user_status = (
        "NOT_SUPPORTED" if user_failures
        else "SUPPORTED" if user_checks
        else "NOT_ASSESSED"
    )
    return {
        # The overall verdict reflects whether the generated insight is accurate.
        # The user's claim is reported separately and may be NOT_SUPPORTED
        # without making an accurate, evidence-grounded insight fail.
        "pass": not insight_failures,
        "overall": "PASS" if not insight_failures else "FAIL",
        "checks": insight_checks + user_checks,
        "insight_checks": insight_checks,
        "insight_pass": not insight_failures,
        "insight_overall": "PASS" if not insight_failures else "FAIL",
        "reason": "; ".join(
            item.get("reason", "") for item in insight_failures
        ) or None,
        "user_claim_validation": {
            "status": user_status,
            "pass": not user_failures if user_checks else None,
            "checks": user_checks,
            "reason": "; ".join(
                item.get("reason", "") for item in user_failures
            ) or None,
        },
    }


# ==========================================================
# GROUPED VALIDATION
# ==========================================================

def validate_grouped_insight(
    insight,
    result
):
    """
    Basic validation for grouped analysis.
    """

    if result is None:

        return {
            "pass": False,
            "reason": "Grouped result is None."
        }

    if not str(
        insight
    ).strip():

        return {
            "pass": False,
            "reason": "Grouped insight is empty."
        }

    return {
        "pass": True,
        "reason": None
    }


# ==========================================================
# TREND VALIDATION
# ==========================================================

def validate_trend_insight(
    insight,
    result
):
    """
    Basic validation for trend analysis.
    """

    if result is None:

        return {
            "pass": False,
            "reason": "Trend result is None."
        }

    if not str(
        insight
    ).strip():

        return {
            "pass": False,
            "reason": "Trend insight is empty."
        }

    return {
        "pass": True,
        "reason": None
    }


# ==========================================================
# GENERIC VALIDATION
# ==========================================================

def validate_generic_insight(
    insight,
    result
):
    """
    Generic validation fallback.
    """

    if result is None:

        return {
            "pass": False,
            "reason": "Result is None."
        }

    if not str(
        insight
    ).strip():

        return {
            "pass": False,
            "reason": "Insight is empty."
        }

    return {
        "pass": True,
        "reason": None
    }




# ==========================================================
# QUESTION-TO-ANSWER RELEVANCE
# ==========================================================

def validate_question_relevance(
    user_question,
    insight,
    result
):
    """
    Check whether the answer addresses key constraints explicitly
    stated in the user's question.

    This is a deterministic relevance guard, not a substitute for
    validating the underlying calculation against the raw dataset.
    """
    question = str(user_question or "").strip().lower()
    insight_text = str(insight or "").strip().lower()
    # Avoid ambiguous truth-value errors and noisy reprs for pandas objects.
    if result is None:
        result_text = ""
    elif isinstance(result, dict):
        result_text = str(result).strip().lower()
    elif hasattr(result, "to_dict"):
        try:
            result_text = str(result.to_dict()).strip().lower()
        except Exception:
            result_text = ""
    else:
        result_text = str(result).strip().lower()
    answer_text = insight_text + " " + result_text

    if not question:
        return {
            "pass": True,
            "reason": "No user question was supplied; relevance check skipped.",
            "skipped": True
        }

    checks = {}
    reasons = []

    # If the user specifies a month, require the answer to address that month.
    months = [
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december"
    ]
    requested_months = [
        month for month in months
        if re.search(r"\b" + re.escape(month) + r"\b", question)
    ]

    if requested_months:
        month_found = any(
            re.search(r"\b" + re.escape(month) + r"\b", insight_text)
            for month in requested_months
        )
        checks["requested_month_addressed"] = month_found
        if not month_found:
            reasons.append(
                "The insight does not explicitly address the requested month: "
                + ", ".join(requested_months)
                + "."
            )

    # Duration questions must return a duration, not unrelated dataset statistics.
    duration_requested = bool(re.search(
        r"\b(duration|how long|length of time|time span|elapsed time|"
        r"number of days|days in|how many days|period length)\b",
        question
    ))

    if duration_requested:
        duration_terms_found = bool(re.search(
            r"\b(duration|days?|weeks?|months?|years?|elapsed|time span|"
            r"calendar|31|30|28|29)\b",
            insight_text
        ))
        checks["duration_answer_present"] = duration_terms_found
        if not duration_terms_found:
            reasons.append(
                "The question asks for a duration, but the insight does not state "
                "a duration or time unit."
            )

    # Ensure explicitly named business metrics are mentioned in the answer.
    metric_groups = {
        "profit": r"\b(profit|earnings?)\b",
        "revenue": r"\b(revenue|sales value|turnover)\b",
        "cost": r"\b(cost|expense|expenditure)\b",
        "units": r"\b(units?|quantity|volume)\b",
        "orders": r"\b(orders?|order count)\b",
        "customers": r"\b(customers?|clients?|subscribers?|non[- ]subscribers?)\b",
    }

    for metric, pattern in metric_groups.items():
        if re.search(pattern, question):
            found = bool(re.search(pattern, insight_text))
            checks[f"requested_metric_{metric}_addressed"] = found
            if not found:
                reasons.append(
                    f"The question mentions {metric}, but the insight does not address it."
                )

    passed = not reasons
    return {
        "pass": passed,
        "reason": None if passed else " ".join(reasons),
        "checks": checks,
        "requested_months": requested_months,
        "duration_requested": duration_requested
    }




# ==========================================================
# BANK MARKETING CONVERSION VALIDATION
# ==========================================================

def _find_conversion_metrics(obj):
    """Find conversion metrics inside nested result structures."""
    keys = {
        "conversion_rate_percent", "converted_count_conversion_1",
        "non_converted_count_conversion_0", "total_records",
        "row_by_row_matches", "row_by_row_mismatches"
    }
    if isinstance(obj, dict):
        if keys.intersection(obj.keys()):
            return obj
        for key in ("primary_result", "supporting_result", "evidence", "result"):
            found = _find_conversion_metrics(obj.get(key))
            if found is not None:
                return found
    return None


def validate_conversion_insight(insight, result):
    """Check conversion-rate figures and conversion/y label consistency against validated metrics."""
    metrics = _find_conversion_metrics(result)
    if not isinstance(metrics, dict):
        return {"pass": False, "reason": "Validated conversion metrics are missing."}

    text = str(insight or "")
    lower = text.lower()
    checks = {}

    def number_present(expected, tolerance=0.02):
        if expected is None:
            return False
        tokens = re.findall(r"(?<![A-Za-z])[-+]?\d[\d,]*(?:\.\d+)?", text)
        for token in tokens:
            try:
                value = float(token.replace(",", ""))
            except ValueError:
                continue
            if abs(value - float(expected)) <= max(tolerance, abs(float(expected)) * 0.0001):
                return True
        return False

    if "total records" in lower or "total records:" in lower:
        checks["total_records"] = number_present(metrics.get("total_records"))
    if "converted customers" in lower or "converted count" in lower:
        checks["converted_count"] = number_present(metrics.get("converted_count_conversion_1"))
    if "non-converted customers" in lower or "non-converted count" in lower:
        checks["non_converted_count"] = number_present(metrics.get("non_converted_count_conversion_0"))
    if "conversion rate" in lower:
        checks["conversion_rate"] = number_present(metrics.get("conversion_rate_percent"))
    if "non-conversion rate" in lower or "non conversion rate" in lower:
        checks["non_conversion_rate"] = number_present(metrics.get("non_conversion_rate_percent"))
    if "consistency validation" in lower or "row-by-row" in lower or "mismatch" in lower:
        checks["conversion_y_consistency"] = (
            metrics.get("compared_records") is not None
            and metrics.get("row_by_row_matches") == metrics.get("compared_records")
            and metrics.get("row_by_row_mismatches") == 0
            and metrics.get("unmatched_conversion_values", 0) == 0
            and metrics.get("unmapped_y_values", 0) == 0
        )

    passed = bool(checks) and all(checks.values())
    return {
        "pass": passed,
        "overall": "PASS" if passed else "FAIL",
        "checks": checks,
        "reason": None if passed else "One or more conversion claims do not match validated evidence."
    }




# ==========================================================
# DESCRIPTIVE SUMMARY VALIDATION
# ==========================================================

def validate_descriptive_summary_claims(insight, result):
    """
    Validate dataset-size and data-quality claims against independent evidence.
    This is deliberately conservative: it validates claims we can verify from
    dataset_facts and does not treat generic presence checks as factual proof.
    """
    facts = None
    if isinstance(result, dict):
        supporting = result.get("supporting_result")
        if isinstance(supporting, dict):
            facts = supporting.get("dataset_facts")
        if facts is None:
            facts = result.get("dataset_facts")

    if not isinstance(facts, dict):
        return {
            "pass": False,
            "overall": "FAIL",
            "checks": {},
            "reason": "Independent dataset facts are missing; descriptive claims cannot be verified."
        }

    text = str(insight or "")
    checks = {}
    patterns = [
        (
            "row_count",
            r"(?:dataset contains|contains|has|with)\s+([\d,]+)\s+rows",
            "row_count",
        ),
        (
            "column_count",
            r"(?:and\s+)?([\d,]+)\s+columns",
            "column_count",
        ),
        (
            "missing_cells",
            r"([\d,]+)\s+missing cells",
            "missing_cells",
        ),
        (
            "duplicate_rows",
            r"([\d,]+)\s+(?:exact\s+)?duplicate rows",
            "duplicate_rows",
        ),
    ]

    for check_name, pattern, fact_key in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                claimed = int(match.group(1).replace(",", ""))
                actual = int(facts.get(fact_key))
                checks[check_name] = {
                    "pass": claimed == actual,
                    "claimed": claimed,
                    "actual": actual,
                }
            except (TypeError, ValueError):
                checks[check_name] = {
                    "pass": False,
                    "claimed": match.group(1),
                    "actual": facts.get(fact_key),
                    "reason": "Independent fact is missing or not numeric.",
                }

    # Validate top categorical values when the insight uses the exact
    # "most common ... is VALUE" / "largest ... group is VALUE" wording.
    categorical = facts.get("categorical_summary", {})
    if isinstance(categorical, dict):
        for column, entries in categorical.items():
            if not isinstance(entries, list) or not entries:
                continue
            top_value = str(entries[0].get("value", "")) if isinstance(entries[0], dict) else ""
            if not top_value:
                continue
            value_pattern = re.compile(
                r"(?:most common|most frequent|largest)\s+"
                + re.escape(str(column))
                + r"\s+(?:category|group|value)?\s*(?:is|was|:)\s*"
                + re.escape(top_value),
                re.IGNORECASE,
            )
            if value_pattern.search(text):
                checks[f"top_category_{column}"] = {
                    "pass": True,
                    "claimed": top_value,
                    "actual": top_value,
                }

    failed = [name for name, item in checks.items() if not item.get("pass", False)]
    passed = bool(checks) and not failed
    return {
        "pass": passed,
        "overall": "PASS" if passed else "FAIL",
        "checks": checks,
        "verified_claims": len(checks) - len(failed),
        "failed_claims": len(failed),
        "reason": None if passed else (
            "No verifiable descriptive claims were found, or one or more claims "
            "did not match independently computed dataset facts."
        ),
    }



def validate_conversion_factor_insight(insight, result):
    """Validate that factor-analysis insights use metrics present in the result."""
    def find_result(obj, depth=0):
        if depth > 7 or not isinstance(obj, dict):
            return None
        if (
            obj.get("analysis_type") == "conversion_factor_analysis"
            and isinstance(obj.get("categorical_factor_results"), dict)
            and isinstance(obj.get("numeric_factor_results"), dict)
        ):
            return obj
        for key in (
            "primary_result", "result", "supporting_result", "evidence",
            "validated_analysis_result"
        ):
            found = find_result(obj.get(key), depth + 1)
            if found is not None:
                return found
        return None

    data = find_result(result)
    text = str(insight or "")
    lower = text.lower()
    if data is None:
        return {
            "pass": False,
            "status": "FAIL",
            "reason": "Validated conversion-factor result structure is missing."
        }

    total = data.get("valid_target_records", data.get("total_dataset_records"))
    converted = data.get("converted_count")
    rate = data.get("overall_conversion_rate_percent")
    checks = {
        "factor_results_present": bool(
            data["categorical_factor_results"] or data["numeric_factor_results"]
        ),
        "overall_metrics_valid": (
            isinstance(total, (int, float)) and total > 0
            and isinstance(converted, (int, float))
            and isinstance(rate, (int, float))
            and 0 <= converted <= total and 0 <= rate <= 100
        ),
        "overall_rate_claim_present": (
            isinstance(rate, (int, float))
            and re.search(r"(?<![\w.])" + re.escape(f"{rate:.2f}") + r"\s*%", text) is not None
        ),
        "factor_patterns_described": any(
            term in lower for term in
            ("factor", "group", "conversion rate", "conversion rates", "associated")
        ),
        "causal_language_qualified": (
            "observational" in lower or "not proof of causation" in lower
        ),
    }

    # Verify every percentage claim that exactly matches a known factor rate
    # is sourced from the calculated result. Require at least one group rate
    # to be mentioned in addition to the overall conversion rate.
    known_rates = set()
    for groups in list(data["categorical_factor_results"].values()) + \
                  list(data["numeric_factor_results"].values()):
        if isinstance(groups, list):
            for group in groups:
                if isinstance(group, dict):
                    value = group.get("conversion_rate_percent")
                    if isinstance(value, (int, float)):
                        known_rates.add(round(float(value), 2))

    mentioned_rates = {
        float(match)
        for match in re.findall(r"(?<![\w.])(\d{1,3}(?:\.\d+)?)\s*%", text)
    }
    checks["factor_rates_supported"] = bool(mentioned_rates.intersection(known_rates))
    checks["no_unsupported_percentage_claims"] = all(
        value == round(float(rate), 2) or value in known_rates
        for value in mentioned_rates
    ) if isinstance(rate, (int, float)) else False

    passed = all(checks.values())
    return {
        "pass": passed,
        "status": "PASS" if passed else "FAIL",
        "checks": checks,
        "reason": None if passed else
            "One or more conversion-factor insight claims failed deterministic validation.",
        "validated_target": data.get("target_column"),
        "overall_conversion_rate_percent": rate,
        "known_factor_rate_count": len(known_rates),
        "mentioned_percentage_values": sorted(mentioned_rates),
    }

# ==========================================================
# MAIN VALIDATOR
# ==========================================================

def validate_insight(
    insight,
    result,
    analysis_type=None,
    task_analysis=None,
    user_question=None,
    data_profile=None
):
    """
    Main insight validation function.
    """

    presence = validate_insight_presence(
        insight
    )

    result_check = validate_result_presence(
        result
    )

    if not presence["pass"]:

        return {
            "overall": "FAIL",
            "pass": False,
            "checks": {
                "presence": presence,
                "result": result_check
            }
        }

    if not result_check["pass"]:

        return {
            "overall": "FAIL",
            "pass": False,
            "checks": {
                "presence": presence,
                "result": result_check
            }
        }

    analysis = str(
        analysis_type
        or ""
    ).lower()

    task_type = ""

    if isinstance(
        task_analysis,
        dict
    ):

        task_type = str(
            task_analysis.get(
                "task_type",
                ""
            )
        ).lower()

    question = str(
        user_question
        or ""
    ).lower()

    relevance_validation = validate_question_relevance(
        user_question=user_question,
        insight=insight,
        result=result
    )

    # Descriptive summaries need independent dataset facts. Do this before
    # auto-detecting nested conversion metrics, otherwise an unrelated nested
    # metric dictionary can force a conversion-rate validator onto a general
    # descriptive insight.
    if "descriptive_summary" in analysis or "descriptive_summary" in task_type:
        task_validation = validate_descriptive_summary_claims(insight, result)
        overall_pass = (
            presence["pass"]
            and result_check["pass"]
            and task_validation.get(
                "insight_pass", task_validation.get("pass", False)
            )
            and relevance_validation.get("pass", False)
        )
        return {
            "overall": "PASS" if overall_pass else "FAIL",
            "pass": overall_pass,
            "checks": {
                "presence": presence,
                "result": result_check,
                "descriptive_claim_validation": task_validation,
                "question_relevance": relevance_validation,
            },
        }

    if (
        "conversion_factor_analysis" in analysis
        or "conversion_factor_analysis" in task_type
        or (
            isinstance(result, dict)
            and result.get("analysis_type") == "conversion_factor_analysis"
        )
    ):
        task_validation = validate_conversion_factor_insight(insight, result)
        overall_pass = (
            presence["pass"]
            and result_check["pass"]
            and task_validation.get("pass", False)
            and relevance_validation.get("pass", False)
        )
        return {
            "overall": "PASS" if overall_pass else "FAIL",
            "pass": overall_pass,
            "checks": {
                "presence": presence,
                "result": result_check,
                "conversion_factor_claim_validation": task_validation,
                "question_relevance": relevance_validation,
            },
        }

    conversion_metrics = _find_conversion_metrics(result)
    conversion_requested = (
        "conversion_factor_analysis" in analysis
        or "conversion_factor_analysis" in task_type
        or "conversion rate" in question
        or "converted customers" in question
        or "non-converted" in question
        or (
            "conversion" in question
            and any(
                term in question
                for term in ("factor", "associated", "patterns", "influence")
            )
        )
    )
    if conversion_requested and conversion_metrics is not None:
        task_validation = validate_conversion_insight(insight, result)
        overall_pass = (
            presence["pass"]
            and result_check["pass"]
            and task_validation.get("pass", False)
            and relevance_validation.get("pass", False)
        )
        return {
            "overall": "PASS" if overall_pass else "FAIL",
            "pass": overall_pass,
            "checks": {
                "presence": presence,
                "result": result_check,
                "conversion_claim_validation": task_validation,
                "question_relevance": relevance_validation,
            },
        }

    # ------------------------------------------------------
    # ANOMALY
    # ------------------------------------------------------

    if (
        "anomaly" in analysis
        or
        "outlier" in analysis
        or
        "anomaly" in task_type
        or
        "outlier" in task_type
        or
        "unusual" in question
    ):

        task_validation = validate_anomaly_claims(
            insight,
            result
        )

        check_name = (
            "claim_level_validation"
        )

    # ------------------------------------------------------
    # GROUPED MEAN NUMERICAL DIRECTION CLAIMS
    # ------------------------------------------------------

    if (
        "grouped_mean_analysis" in analysis
        or "grouped_mean_analysis" in task_type
        or (
            isinstance(result, dict)
            and result.get("analysis_type") in {
                "grouped_mean_analysis",
                "multi_metric_grouped_mean_analysis",
            }
        )
    ):
        task_validation = validate_grouped_mean_direction_claims(
            insight=insight,
            result=result,
            user_question=user_question,
        )
        overall_pass = (
            presence["pass"]
            and result_check["pass"]
            and task_validation.get("pass", False)
            and relevance_validation.get("pass", False)
        )
        return {
            "overall": "PASS" if overall_pass else "FAIL",
            "pass": overall_pass,
            "checks": {
                "presence": presence,
                "result": result_check,
                "grouped_mean_claim_validation": task_validation,
                "question_relevance": relevance_validation,
            },
        }

    # ------------------------------------------------------
    # GROUPED BALANCE CONVERSION RATE
    # Check the validated result type before the broad
    # relationship-analysis routing below.
    # ------------------------------------------------------

    if (
        isinstance(result, dict)
        and result.get("analysis_type") == "grouped_conversion_rate_by_balance"
    ):
        task_validation = validate_grouped_insight(
            insight,
            result
        )
        check_name = "grouped"

    # ------------------------------------------------------
    # CORRELATION / RELATIONSHIP
    # ------------------------------------------------------

    elif (
        "correlation" in analysis
        or
        "relationship" in analysis
        or
        "correlation" in task_type
        or
        "relationship" in task_type
        or
        "relationship" in question
        or
        "correlation" in question
    ):

        task_validation = validate_correlation_insight(
            insight,
            result
        )

        check_name = "correlation"

    # ------------------------------------------------------
    # GROUPED / COMPARISON
    # ------------------------------------------------------

    elif (
        "group" in analysis
        or
        "comparison" in analysis
        or
        "group" in task_type
        or
        "comparison" in task_type
        or
        "highest" in question
    ):

        task_validation = validate_grouped_insight(
            insight,
            result
        )

        check_name = "grouped"

    # ------------------------------------------------------
    # TREND
    # ------------------------------------------------------

    elif (
        "trend" in analysis
        or
        "time" in analysis
        or
        "trend" in task_type
        or
        "time" in task_type
        or
        "over time" in question
    ):

        task_validation = validate_trend_insight(
            insight,
            result
        )

        check_name = "trend"

    # ------------------------------------------------------
    # GENERIC
    # ------------------------------------------------------

    else:

        task_validation = validate_generic_insight(
            insight,
            result
        )

        check_name = "generic"

    overall_pass = (
        presence["pass"]
        and
        result_check["pass"]
        and
        task_validation.get(
            "pass",
            False
        )
        and
        relevance_validation.get(
            "pass",
            False
        )
    )

    return {
        "overall": (
            "PASS"
            if overall_pass
            else "FAIL"
        ),

        "pass": overall_pass,

        "checks": {
            "presence": presence,
            "result": result_check,
            check_name: task_validation,
            "question_relevance": relevance_validation
        }
    }


# ==========================================================
# MODULE TEST
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)

    print(
        "INSIGHT VALIDATOR TEST"
    )

    print("=" * 60)

    # ------------------------------------------------------
    # Correlation test
    # ------------------------------------------------------

    test_result = (
        0.5645504620845976
    )

    test_insight = (
        "The Pearson correlation between Units Sold "
        "and Total Profit is approximately 0.56, "
        "indicating a moderate relationship. This "
        "means that higher Units Sold values tend "
        "to be associated with positive changes "
        "in Total Profit."
    )

    validation = validate_insight(
        test_insight,
        test_result,
        analysis_type="relationship_analysis",
        task_analysis={
            "task_type":
                "relationship_analysis"
        },
        user_question=(
            "Is there a relationship between "
            "units sold and total profit?"
        )
    )

    print()

    print(
        "Correlation Validation:"
    )

    print(
        validation
    )

    print()

    if validation.get(
        "overall"
    ) == "PASS":

        print(
            "VALIDATOR TEST: PASS"
        )

    else:

        print(
            "VALIDATOR TEST: FAIL"
        )