import re
import math


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
            match.group(1).replace(",", "").replace("$", "")
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
            match.group(2).replace(",", "").replace("$", "")
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
        r"(?:lower and upper boundaries were|values below)\s+"
        r"([-+]?\$?[\d,]+(?:\.\d+)?)"
        r"(?:\s+and\s+|\s+or\s+above\s+)"
        r"([-+]?\$?[\d,]+(?:\.\d+)?)",
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
        r"(?:identified\s+([\d,]+)\s+potential\s+Total Profit outliers|"
        r"([\d,]+)\s+outliers?\s+(?:were\s+)?identified)",
        text,
        re.IGNORECASE
    )

    if match:
        count_text = match.group(1) or match.group(2)
        claims["outlier_count"] = int(count_text.replace(",", ""))

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
            "count", result.get("valid_value_count", result.get("row_count"))
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
                        "outlier_count", result.get("anomaly_count", -1)
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
# ISOLATION FOREST CLAIM EXTRACTION AND VALIDATION
# ==========================================================

def extract_isolation_forest_claims(insight):
    """
    Extract only count and percentage claims that can be checked against
    the compact Isolation Forest result returned by the step executor.
    """
    text = str(insight or "")
    claims = {}

    number = r"([\d,]+(?:\.\d+)?)"

    patterns = {
        "row_count": [
            rf"(?:evaluated|processed|analysed|analyzed|examined)\s+{number}\s+(?:rows|records|observations)\b",
            rf"{number}\s+(?:total\s+)?(?:rows|records|observations)\b",
        ],
        "anomaly_count": [
            rf"(?:detected|identified|found|flagged)\s+{number}\s+(?:potential\s+)?(?:anomalies|anomalous\s+rows|outliers)\b",
            rf"{number}\s+(?:potential\s+)?(?:anomalies|anomalous\s+rows|outliers)\b",
        ],
        "normal_count": [
            rf"{number}\s+(?:normal\s+)?(?:rows|records|observations)\s+(?:were\s+)?(?:classified|labelled|labeled)\s+as\s+normal\b",
            rf"{number}\s+normal\s+(?:rows|records|observations)\b",
        ],
    }

    for key, candidates in patterns.items():
        for pattern in candidates:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                raw = match.group(1).replace(",", "")
                value = _safe_float(raw)
                if value is not None and math.isfinite(value):
                    claims[key] = value
                break

    # A percentage is captured only when its nearby wording identifies it
    # as an anomaly/outlier percentage or rate.
    for match in re.finditer(
        rf"{number}\s*%",
        text,
        re.IGNORECASE,
    ):
        start, end = match.span()
        context = text[max(0, start - 55):min(len(text), end + 35)].lower()
        if any(term in context for term in ("anomal", "outlier")):
            value = _safe_float(match.group(1).replace(",", ""))
            if value is not None and math.isfinite(value):
                claims["anomaly_percentage"] = value
                break

    return claims


def validate_isolation_forest_claims(insight, result):
    """
    Validate Isolation Forest result integrity and any numeric claims made
    by the insight. This intentionally does not apply IQR statistics or
    infer that an anomaly is an error, fraud, or another specific cause.
    """
    failure = {
        "overall": "FAIL",
        "pass": False,
        "total_claims": 0,
        "passed_claims": 0,
        "failed_claims": 0,
    }

    if not isinstance(result, dict):
        return {
            **failure,
            "reason": "Isolation Forest result is not a dictionary.",
            "checks": {},
        }

    method = str(result.get("method", "")).strip().lower()
    normalized_method = re.sub(r"[\s_-]+", "", method)
    if normalized_method != "isolationforest":
        return {
            **failure,
            "reason": "Result method is not Isolation Forest.",
            "checks": {"method": False},
        }

    def valid_count(value):
        number_value = _safe_float(value)
        if (
            number_value is None
            or not math.isfinite(number_value)
            or number_value < 0
            or not number_value.is_integer()
        ):
            return None
        return int(number_value)

    row_count = valid_count(result.get("row_count"))
    anomaly_count = valid_count(result.get("anomaly_count"))
    normal_count = valid_count(result.get("normal_count"))
    anomaly_percentage = _safe_float(result.get("anomaly_percentage"))
    feature_columns = result.get("feature_columns")
    summary = result.get("summary")

    checks = {
        "method": True,
        "result_status": str(result.get("status", "")).upper() == "PASS",
        "row_count_valid": row_count is not None and row_count > 0,
        "anomaly_count_valid": anomaly_count is not None,
        "normal_count_valid": normal_count is not None,
        "feature_columns_valid": (
            isinstance(feature_columns, (list, tuple))
            and len(feature_columns) > 0
        ),
        "anomaly_percentage_valid": (
            anomaly_percentage is not None
            and math.isfinite(anomaly_percentage)
            and 0 <= anomaly_percentage <= 100
        ),
    }

    counts_available = (
        row_count is not None
        and anomaly_count is not None
        and normal_count is not None
    )
    checks["row_count_reconciles"] = (
        counts_available
        and row_count == anomaly_count + normal_count
    )

    if (
        counts_available
        and row_count is not None
        and row_count > 0
        and anomaly_percentage is not None
        and math.isfinite(anomaly_percentage)
    ):
        expected_percentage = 100.0 * anomaly_count / row_count
        checks["anomaly_percentage_reconciles"] = _approximately_equal(
            anomaly_percentage,
            expected_percentage,
            absolute_tolerance=0.01,
            relative_tolerance=0.0001,
        )
    else:
        checks["anomaly_percentage_reconciles"] = False

    # When the detector summary is present, verify that it agrees with the
    # compact result rather than trusting two potentially conflicting copies.
    if isinstance(summary, dict):
        summary_total = summary.get("total_rows")
        summary_anomalies = summary.get("anomaly_count")
        summary_normals = summary.get("normal_count")
        summary_percentage = summary.get("detected_anomaly_percentage")

        if summary_total is not None:
            checks["summary_total_matches"] = valid_count(summary_total) == row_count
        if summary_anomalies is not None:
            checks["summary_anomaly_count_matches"] = (
                valid_count(summary_anomalies) == anomaly_count
            )
        if summary_normals is not None:
            checks["summary_normal_count_matches"] = (
                valid_count(summary_normals) == normal_count
            )
        if summary_percentage is not None:
            parsed_summary_percentage = _safe_float(summary_percentage)
            checks["summary_percentage_matches"] = (
                parsed_summary_percentage is not None
                and math.isfinite(parsed_summary_percentage)
                and anomaly_percentage is not None
                and _approximately_equal(
                    parsed_summary_percentage,
                    anomaly_percentage,
                )
            )

    claims = extract_isolation_forest_claims(insight)
    claim_checks = {}

    expected_values = {
        "row_count": row_count,
        "anomaly_count": anomaly_count,
        "normal_count": normal_count,
        "anomaly_percentage": anomaly_percentage,
    }
    for key, claimed_value in claims.items():
        actual_value = expected_values.get(key)
        if actual_value is None:
            claim_checks[key] = False
        elif key in {"row_count", "anomaly_count", "normal_count"}:
            claim_checks[key] = (
                float(claimed_value).is_integer()
                and int(claimed_value) == actual_value
            )
        else:
            claim_checks[key] = _approximately_equal(
                claimed_value,
                actual_value,
                absolute_tolerance=0.01,
                relative_tolerance=0.0001,
            )

    # Require at least one verifiable numeric claim; otherwise a generic
    # sentence could pass without communicating validated anomaly results.
    checks["numeric_claim_present"] = bool(claim_checks)
    checks.update({
        f"claim_{key}": status
        for key, status in claim_checks.items()
    })

    failed_checks = [name for name, passed in checks.items() if not passed]
    passed_checks = len(checks) - len(failed_checks)
    overall_pass = bool(checks) and not failed_checks

    return {
        "overall": "PASS" if overall_pass else "FAIL",
        "pass": overall_pass,
        "total_claims": len(checks),
        "passed_claims": passed_checks,
        "failed_claims": len(failed_checks),
        "checks": checks,
        "claims": claims,
        "validated_values": {
            "row_count": row_count,
            "anomaly_count": anomaly_count,
            "normal_count": normal_count,
            "anomaly_percentage": anomaly_percentage,
            "feature_columns": list(feature_columns)
            if isinstance(feature_columns, (list, tuple))
            else [],
        },
        "reason": None if overall_pass else (
            "Isolation Forest result integrity or one or more numeric "
            "insight claims failed deterministic validation."
        ),
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
    Validate grouped-mean insights against computed group means.

    Supports arbitrary category groups (for example job or region) and retains
    the subscriber/non-subscriber directional check for yes/no comparisons.
    A user assumption that is disproved by the data is reported separately from
    the validity of the generated insight.
    """
    try:
        from .grouped_mean_claim_validation import (
            extract_metric_statement,
            group_label_is_present,
            number_is_associated_with_group,
        )
    except ImportError:
        from grouped_mean_claim_validation import (
            extract_metric_statement,
            group_label_is_present,
            number_is_associated_with_group,
        )

    if not isinstance(result, dict):
        return {
            "pass": False,
            "overall": "FAIL",
            "reason": "Grouped result is not a dictionary.",
            "checks": [],
            "insight_checks": [],
            "insight_pass": False,
            "insight_overall": "FAIL",
            "user_claim_validation": {
                "status": "REVIEW",
                "pass": None,
                "reason": "Grouped result is not a dictionary.",
                "checks": [],
            },
        }

    def numeric(value):
        try:
            number = float(value)
            return number if number == number and abs(number) != float("inf") else None
        except (TypeError, ValueError, OverflowError):
            return None

    def number_appears(value, text):
        """Check a displayed number, allowing comma formatting and rounding."""
        if value is None:
            return False
        # Use a numeric-token scan to avoid matching 12.3 inside 112.30.
        tokens = re.findall(r"(?<![A-Za-z0-9_])[-+]?\d[\d,]*(?:\.\d+)?", text)
        for token in tokens:
            try:
                shown = float(token.replace(",", ""))
            except ValueError:
                continue
            if abs(shown - value) <= max(0.011, abs(value) * 0.00001):
                return True
        return False

    metric_results = result.get("metric_results")
    is_multi_metric = isinstance(metric_results, dict) and bool(metric_results)
    if is_multi_metric:
        candidates = [(str(metric), item) for metric, item in metric_results.items()]
        metric_names = [metric for metric, _ in candidates]
    else:
        candidates = [(str(result.get("metric_column", "")), result)]
        metric_names = [candidates[0][0]]

    insight_text = str(insight or "").strip()
    normalized_insight = insight_text.casefold()
    question_text = str(user_question or "").casefold()
    insight_checks = []
    user_checks = []

    for metric, item in candidates:
        if not isinstance(item, dict):
            insight_checks.append({
                "metric": metric,
                "pass": False,
                "reason": "Metric result is not a dictionary.",
            })
            continue

        groups = item.get("group_results")
        if not isinstance(groups, dict) or not groups:
            insight_checks.append({
                "metric": metric,
                "pass": False,
                "reason": "No grouped means were available to validate.",
            })
            continue

        if is_multi_metric:
            metric_text = extract_metric_statement(
                insight_text, metric, metric_names
            )
            if metric_text is None:
                insight_checks.append({
                    "metric": metric,
                    "check": "metric_statement_present",
                    "pass": False,
                    "reason": f"No metric-specific statement was found for '{metric}'.",
                })
                continue
        else:
            metric_text = insight_text

        normalized_metric_text = metric_text.casefold()
        metric_is_mentioned = (
            bool(metric.strip())
            and metric.casefold() in normalized_insight
            and (not is_multi_metric or metric_text is not None)
        )
        if not metric_is_mentioned:
            insight_checks.append({
                "metric": metric,
                "pass": False,
                "reason": f"The insight does not mention metric '{metric}'.",
            })
            continue

        # Validate all arbitrary category groups by checking both the label
        # and the numeric mean reported in the generated insight.
        generic_group_checks = []
        for group_name, group_data in groups.items():
            if not isinstance(group_data, dict):
                generic_group_checks.append({
                    "group": str(group_name),
                    "pass": False,
                    "reason": "Group statistics are not a dictionary.",
                })
                continue
            mean_value = numeric(group_data.get("mean_value", group_data.get("mean")))
            group_label_found = group_label_is_present(
                metric_text, group_name
            )
            mean_found = number_is_associated_with_group(
                metric_text, group_name, mean_value
            )
            group_pass = group_label_found and mean_found
            generic_group_checks.append({
                "group": str(group_name),
                "mean": mean_value,
                "group_label_mentioned": group_label_found,
                "mean_mentioned": mean_found,
                "pass": group_pass,
                "reason": None if group_pass else (
                    f"The insight does not clearly report group '{group_name}' "
                    "and its validated mean."
                ),
            })

        is_binary = "no" in groups and "yes" in groups
        if not is_binary:
            insight_checks.extend(generic_group_checks)
            continue

        no_group = groups.get("no", {})
        yes_group = groups.get("yes", {})
        no_mean = numeric(no_group.get("mean_value", no_group.get("mean"))) if isinstance(no_group, dict) else None
        yes_mean = numeric(yes_group.get("mean_value", yes_group.get("mean"))) if isinstance(yes_group, dict) else None

        # If this result is not actually a valid binary numeric comparison,
        # use generic per-group validation instead.
        if no_mean is None or yes_mean is None:
            insight_checks.extend(generic_group_checks)
            continue

        # The insight must mention both group labels and their actual means.
        binary_checks = [
            {
                "metric": metric,
                "check": "metric_mentioned",
                "pass": metric_is_mentioned,
                "reason": None if metric_is_mentioned else f"Metric '{metric}' is not mentioned.",
            },
            {
                "metric": metric,
                "check": "no_group_mean_reported",
                "pass": (
                    group_label_is_present(metric_text, "no")
                    and number_is_associated_with_group(
                        metric_text, "no", no_mean
                    )
                ),
                "reason": "The insight does not report the validated no-group mean.",
            },
            {
                "metric": metric,
                "check": "yes_group_mean_reported",
                "pass": (
                    group_label_is_present(metric_text, "yes")
                    and number_is_associated_with_group(
                        metric_text, "yes", yes_mean
                    )
                ),
                "reason": "The insight does not report the validated yes-group mean.",
            },
        ]
        insight_checks.extend(binary_checks)

        actual_direction = (
            "lower" if yes_mean < no_mean
            else "higher" if yes_mean > no_mean
            else "equal"
        )
        for source_name, source_text, target_checks in (
            ("insight", normalized_metric_text, insight_checks),
            ("user_question", question_text, user_checks),
        ):
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

    insight_failures = [check for check in insight_checks if not check.get("pass", False)]
    user_failures = [check for check in user_checks if not check.get("pass", False)]
    user_status = (
        "NOT_SUPPORTED" if user_failures
        else "SUPPORTED" if user_checks
        else "NOT_ASSESSED"
    )
    combined_failures = insight_failures + user_failures
    return {
        "pass": not combined_failures,
        "overall": "PASS" if not combined_failures else "FAIL",
        "checks": insight_checks + user_checks,
        "insight_checks": insight_checks,
        "insight_pass": not insight_failures,
        "insight_overall": "PASS" if not insight_failures else "FAIL",
        "reason": "; ".join(
            check.get("reason", "") for check in combined_failures if check.get("reason")
        ) or None,
        "user_claim_validation": {
            "status": user_status,
            "pass": not user_failures if user_checks else None,
            "checks": user_checks,
            "reason": "; ".join(
                check.get("reason", "") for check in user_failures if check.get("reason")
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
        if metric == "customers":
            # "customers" often identifies the population, not the requested metric.
            # Require it in the answer only when the question asks for a customer count.
            asks_customer_count = bool(re.search(
                r"\b(count|number|how many|total number|customer count|"
                r"number of customers|count of customers)\b",
                question,
            ))
            if not asks_customer_count:
                continue
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
    Validate dataset-level descriptive claims against independent evidence.

    Statistical claims such as mean, median, and standard deviation are
    handled by dedicated result and evidence validators. They should not
    fail this dataset-facts validation when no dataset-level claims exist.
    """
    text = str(insight or "")

    # Identify claims that require independent dataset-level facts.
    dataset_claim_patterns = [
        r"(?:dataset contains|contains|has|with)\s+([\d,]+)\s+rows",
        r"(?:and\s+)?([\d,]+)\s+columns",
        r"([\d,]+)\s+missing cells",
        r"([\d,]+)\s+(?:exact\s+)?duplicate rows",
        r"(?:most common|most frequent|largest)\s+.+?\s+(?:category|group|value)?\s*(?:is|was|:)",
    ]

    has_dataset_claim = any(
        re.search(pattern, text, re.IGNORECASE)
        for pattern in dataset_claim_patterns
    )

    if not has_dataset_claim:
        return {
            "pass": True,
            "overall": "NOT_APPLICABLE",
            "status": "NOT_APPLICABLE",
            "skipped": True,
            "checks": {},
            "verified_claims": 0,
            "failed_claims": 0,
            "reason": (
                "No dataset-level descriptive claims were found. "
                "Statistical claims are handled by dedicated result "
                "and evidence validators."
            ),
        }

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
            "reason": (
                "Independent dataset facts are missing; "
                "dataset-level descriptive claims cannot be verified."
            ),
        }

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

    categorical = facts.get("categorical_summary", {})

    if isinstance(categorical, dict):
        for column, entries in categorical.items():
            if not isinstance(entries, list) or not entries:
                continue

            top_value = (
                str(entries[0].get("value", ""))
                if isinstance(entries[0], dict)
                else ""
            )

            if not top_value:
                continue

            value_pattern = re.compile(
                r"(?:most common|most frequent|largest)\s+"
                + re.escape(str(column))
                + r"\s+(?:category|group|value)?\s*(?:is|was|:)\s*"
                + r"(.+?)(?:[.;\n]|$)",
                re.IGNORECASE,
            )

            match = value_pattern.search(text)

            if match:
                claimed_value = match.group(1).strip().strip("'\"")

                checks[f"top_category_{column}"] = {
                    "pass": claimed_value.casefold() == top_value.casefold(),
                    "claimed": claimed_value,
                    "actual": top_value,
                }

    failed = [
        name
        for name, item in checks.items()
        if not item.get("pass", False)
    ]

    passed = bool(checks) and not failed

    return {
        "pass": passed,
        "overall": "PASS" if passed else "FAIL",
        "checks": checks,
        "verified_claims": len(checks) - len(failed),
        "failed_claims": len(failed),
        "reason": None if passed else (
            "No verifiable dataset-level claims were found, or one or "
            "more claims did not match independently computed facts."
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

def validate_extrema_insight(insight, result, user_question=None):
    """Validate a scalar maximum/minimum claim against the generated result."""
    if not isinstance(result, dict):
        return {"pass": False, "reason": "Extrema result is not a dictionary.", "checks": []}

    def find_payload(obj, depth=0):
        if depth > 6 or not isinstance(obj, dict):
            return None
        if (obj.get("analysis_type") == "extrema_analysis"
                and obj.get("value") is not None
                and obj.get("operation") in {"maximum", "minimum"}):
            return obj
        for key in ("primary_result", "result", "supporting_result", "evidence", "validated_result"):
            found = find_payload(obj.get(key), depth + 1)
            if found is not None:
                return found
        return None

    payload = find_payload(result)
    if payload is None:
        return {"pass": False, "reason": "No validated maximum/minimum payload was found.", "checks": []}
    operation = payload.get("operation")
    metric = str(payload.get("metric_column") or "").strip()
    expected = _safe_float(payload.get("value"))
    text = " ".join(str(insight or "").lower().replace("_", " ").split())
    metric_text = " ".join(metric.lower().replace("_", " ").split())
    metric_ok = bool(metric_text and re.search(
        r"(?<![\w])" + re.escape(metric_text) + r"(?![\w])", text
    ))
    operation_pattern = (r"\b(maximum|max|highest|largest|greatest)\b"
                         if operation == "maximum"
                         else r"\b(minimum|min|lowest|smallest|least)\b")
    operation_ok = bool(re.search(operation_pattern, text))
    claimed_numbers = []
    for token in re.findall(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?", str(insight or "")):
        parsed = _safe_float(token.replace(",", ""))
        if parsed is not None:
            claimed_numbers.append(parsed)
    value_ok = expected is not None and any(
        round(number, 2) == round(expected, 2) for number in claimed_numbers
    )
    checks = [
        {"check": "metric_column_mentioned", "pass": metric_ok},
        {"check": "extreme_operation_mentioned", "pass": operation_ok},
        {"check": "extreme_value_matches_result", "pass": value_ok},
    ]
    failed = [check for check in checks if not check["pass"]]
    return {
        "pass": not failed,
        "reason": None if not failed else "The insight does not state the requested maximum/minimum metric and validated value.",
        "operation": operation, "metric_column": metric,
        "validated_value": expected, "checks": checks,
    }



# ==========================================================
# NESTED IQR RESULT DISCOVERY
# ==========================================================

def _find_iqr_result(result, depth=0, metric_hint=None):
    """Find a validated IQR result inside direct or specification-driven output."""
    if depth > 10:
        return None
    if isinstance(result, list):
        for item in result:
            found = _find_iqr_result(item, depth + 1, metric_hint)
            if found is not None:
                return found
        return None
    if not isinstance(result, dict):
        return None

    method = re.sub(r"[\s_-]+", "", str(result.get("method", "")).strip().lower())
    if method in {"iqr", "interquartilerange"}:
        if str(result.get("status", "PASS")).upper() == "PASS":
            normalized = dict(result)
            if metric_hint and not any(normalized.get(key) for key in ("metric_column", "column", "metric")):
                normalized["column"] = str(metric_hint)
            return normalized

    operation_results = result.get("operation_results")
    if isinstance(operation_results, dict):
        for operation_name, operation_result in operation_results.items():
            if str(operation_name).lower() in {"outlier_detection", "outlier_analysis", "anomaly_detection"} and isinstance(operation_result, dict):
                for metric_name, metric_result in operation_result.items():
                    found = _find_iqr_result(metric_result, depth + 1, metric_name)
                    if found is not None:
                        return found
            found = _find_iqr_result(operation_result, depth + 1, metric_hint)
            if found is not None:
                return found

    for key in ("primary_result", "result", "validated_result", "selected_result", "supporting_result", "summary", "outlier_detection", "results"):
        if key in result:
            found = _find_iqr_result(result.get(key), depth + 1, metric_hint)
            if found is not None:
                return found

    for key, value in result.items():
        if key in {"operation_results", "primary_result", "result", "validated_result", "selected_result", "supporting_result", "summary", "outlier_detection", "results", "dataset_facts", "numeric_summary", "categorical_summary"}:
            continue
        if isinstance(value, (dict, list)):
            found = _find_iqr_result(value, depth + 1, str(key))
            if found is not None:
                return found
    return None


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

    # Scalar maximum/minimum must be validated against the calculated value.
    extrema_requested = (
        "extrema_analysis" in analysis
        or "extrema_analysis" in task_type
        or (isinstance(result, dict) and result.get("analysis_type") == "extrema_analysis")
        or (
            re.search(r"\b(maximum|max|minimum|min)\b", question)
            and not any(term in question for term in ("by category", "for each", "rank", "ranking"))
        )
    )
    if extrema_requested:
        task_validation = validate_extrema_insight(
            insight=insight, result=result, user_question=user_question
        )
        overall_pass = (
            presence["pass"] and result_check["pass"]
            and task_validation.get("pass", False)
            and relevance_validation.get("pass", False)
        )
        return {
            "overall": "PASS" if overall_pass else "FAIL",
            "pass": overall_pass,
            "checks": {
                "presence": presence, "result": result_check,
                "extrema_claim_validation": task_validation,
                "question_relevance": relevance_validation,
            },
        }

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
    # IQR OUTLIER VALIDATION
    # Specification-driven results keep method/bounds/count nested under
    # operation_results[outlier_detection][metric]. Normalize that result
    # before invoking the existing IQR claim validator.
    # ------------------------------------------------------
    iqr_result = _find_iqr_result(result)
    if iqr_result is not None:
        normalized_iqr = dict(iqr_result)
        normalized_iqr.setdefault("outlier_count", normalized_iqr.get("anomaly_count"))
        normalized_iqr.setdefault("column", normalized_iqr.get("metric_column", normalized_iqr.get("metric")))
        normalized_iqr.setdefault("valid_value_count", normalized_iqr.get("valid_count"))
        task_validation = validate_anomaly_claims(insight, normalized_iqr)
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
                "iqr_claim_validation": task_validation,
                "question_relevance": relevance_validation,
            },
        }

    # ------------------------------------------------------
    # ISOLATION FOREST
    # Keep this route separate from the legacy IQR validator.
    # Return immediately so later generic routing cannot overwrite it.
    # ------------------------------------------------------

    result_method = (
        str(result.get("method", "")).strip().lower()
        if isinstance(result, dict)
        else ""
    )
    normalized_result_method = re.sub(
        r"[\s_-]+",
        "",
        result_method,
    )
    is_isolation_forest = normalized_result_method == "isolationforest"

    if is_isolation_forest:
        task_validation = validate_isolation_forest_claims(
            insight,
            result,
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
                "isolation_forest_claim_validation": task_validation,
                "question_relevance": relevance_validation,
            },
        }

    # ------------------------------------------------------
    # LEGACY IQR ANOMALY / OUTLIER VALIDATION
    # ------------------------------------------------------

    if (
        "anomaly" in analysis
        or "outlier" in analysis
        or "anomaly" in task_type
        or "outlier" in task_type
        or "unusual" in question
    ):
        task_validation = validate_anomaly_claims(
            insight,
            result
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
                "claim_level_validation": task_validation,
                "question_relevance": relevance_validation,
            },
        }

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
            and task_validation.get(
                "insight_pass",
                task_validation.get("pass", False),
            )
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
        and str(result.get("analysis_type", "")).startswith("grouped_conversion_rate_by_")
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