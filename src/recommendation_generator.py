# ============================================================
# RECOMMENDATION GENERATOR
# ============================================================


# ============================================================
# HELPER - FORMAT NUMBER
# ============================================================

def format_number(value):

    try:
        return f"{float(value):,.2f}"

    except (ValueError, TypeError):

        return str(value)


# ============================================================
# 1. GROUPED ANALYSIS RECOMMENDATION
# ============================================================

def generate_grouped_recommendation(
    evidence
):
    """
    Generate a recommendation from grouped-analysis
    evidence.

    Example:
        Region -> Total Profit
    """

    # --------------------------------------------------------
    # Validate evidence
    # --------------------------------------------------------

    if not isinstance(evidence, dict):

        return {
            "status": "FAIL",
            "reason": (
                "Evidence must be a dictionary."
            )
        }

    if evidence.get("status") != "PASS":

        return {
            "status": "FAIL",
            "reason": (
                "Evidence validation did not pass."
            )
        }

    # --------------------------------------------------------
    # Extract evidence
    # --------------------------------------------------------

    evidence_details = evidence.get(
        "evidence",
        {}
    )

    top_category = evidence_details.get(
        "top_category"
    )

    top_value = evidence_details.get(
        "top_value"
    )

    second_category = evidence_details.get(
        "second_category"
    )

    second_value = evidence_details.get(
        "second_value"
    )

    metric = evidence.get(
        "metric",
        "metric"
    )

    # --------------------------------------------------------
    # Validate required information
    # --------------------------------------------------------

    if top_category is None:

        return {
            "status": "FAIL",
            "reason": (
                "Top category is missing "
                "from evidence."
            )
        }

    if top_value is None:

        return {
            "status": "FAIL",
            "reason": (
                "Top value is missing "
                "from evidence."
            )
        }

    # --------------------------------------------------------
    # Format values
    # --------------------------------------------------------

    top_value_text = format_number(
        top_value
    )

    # --------------------------------------------------------
    # Calculate difference from second category
    # --------------------------------------------------------

    difference_text = ""

    if (
        second_category is not None
        and second_value is not None
    ):

        try:

            difference = (
                float(top_value)
                - float(second_value)
            )

            difference_text = (
                f" It exceeds "
                f"{second_category}, the second-highest "
                f"region, by approximately "
                f"{format_number(difference)}."
            )

        except (ValueError, TypeError):

            difference_text = ""

    # --------------------------------------------------------
    # Generate recommendation
    # --------------------------------------------------------

    recommendation = (
        f"Consider prioritizing further analysis "
        f"of {top_category}, as it has the highest "
        f"{metric.lower()} at approximately "
        f"{top_value_text}."
        f"{difference_text} "
        f"Investigate the factors contributing to "
        f"this performance before making "
        f"resource-allocation decisions."
    )

    # --------------------------------------------------------
    # Return structured result
    # --------------------------------------------------------

    return {
        "status": "PASS",
        "recommendation": recommendation,
        "basis": {
            "category": top_category,
            "metric": metric,
            "value": float(top_value)
        }
    }


# ============================================================
# 2. MEAN ANALYSIS RECOMMENDATION
# ============================================================

def generate_mean_recommendation(evidence):
    """
    Generate a recommendation from validated mean evidence.

    Supports both the original mean evidence schema (metric/value)
    and filtered analysis results such as:
      analysis_type=filtered_mean, filter_column=month,
      filter_value=aug, metric_column=duration, mean_duration=232.40.
    """

    if not isinstance(evidence, dict):
        return {
            "status": "FAIL",
            "reason": "Evidence must be a dictionary."
        }

    if evidence.get("status") != "PASS":
        return {
            "status": "FAIL",
            "reason": "Evidence validation did not pass."
        }

    # Some pipeline versions wrap the numerical result in "evidence".
    nested = evidence.get("evidence")
    details = nested if isinstance(nested, dict) else {}
    sources = [evidence, details]

    def first_present(*keys):
        for source in sources:
            for key in keys:
                value = source.get(key)
                if value is not None and value != "":
                    return value
        return None

    metric = first_present("metric", "metric_column", "column")
    value = first_present(
        "value", "mean", "mean_duration", "average", "average_value"
    )

    # Filtered analysis metadata may also be nested.
    filter_column = first_present("filter_column")
    filter_value = first_present("filter_value")
    filtered_rows = first_present("filtered_rows", "row_count")
    valid_count = first_present("valid_duration_count", "valid_count", "count")

    if value is None:
        return {
            "status": "FAIL",
            "reason": (
                "Mean value is missing. Expected one of: value, mean, "
                "mean_duration, average, or average_value."
            ),
            "available_keys": sorted(
                set(evidence.keys()).union(details.keys())
            )
        }

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return {
            "status": "FAIL",
            "reason": f"Mean value is not numeric: {value!r}."
        }

    if metric is None:
        metric = "duration" if "duration" in str(evidence).lower() else "measure"

    scope = ""
    if filter_column is not None and filter_value is not None:
        scope = f" for {filter_column}={filter_value}"

    sample_detail = ""
    if filtered_rows is not None:
        sample_detail += f" The filtered dataset contains {filtered_rows} rows."
    if valid_count is not None:
        sample_detail += f" The mean uses {valid_count} valid observations."

    recommendation = (
        f"Use the validated average {str(metric).lower()}{scope} "
        f"of approximately {format_number(numeric_value)} as a descriptive "
        f"baseline for further analysis.{sample_detail} Investigate unusually "
        f"short or long observations and compare other months before drawing "
        f"conclusions. This average alone does not establish causation."
    )

    basis = {
        "analysis_type": evidence.get("analysis_type", "mean_analysis"),
        "metric": metric,
        "value": numeric_value
    }
    if filter_column is not None:
        basis["filter_column"] = filter_column
    if filter_value is not None:
        basis["filter_value"] = filter_value
    if filtered_rows is not None:
        basis["filtered_rows"] = filtered_rows
    if valid_count is not None:
        basis["valid_count"] = valid_count

    return {
        "status": "PASS",
        "recommendation": recommendation,
        "basis": basis
    }


# ============================================================
# 3. CORRELATION / RELATIONSHIP RECOMMENDATION
# ============================================================

def generate_correlation_recommendation(
    evidence
):
    """
    Generate a recommendation from correlation evidence.

    Expected evidence structure:

        {
            "status": "PASS",
            "analysis_type": "correlation_analysis",
            "x_variable": "Units Sold",
            "y_variable": "Total Profit",
            "method": "Pearson",
            "correlation": 0.5645,
            "direction": "positive",
            "strength": "moderate"
        }
    """

    # --------------------------------------------------------
    # Validate evidence
    # --------------------------------------------------------

    if not isinstance(evidence, dict):

        return {
            "status": "FAIL",
            "reason": (
                "Evidence must be a dictionary."
            )
        }

    if evidence.get("status") != "PASS":

        return {
            "status": "FAIL",
            "reason": (
                "Evidence validation did not pass."
            )
        }

    # --------------------------------------------------------
    # Extract correlation evidence
    # --------------------------------------------------------

    x_variable = evidence.get(
        "x_variable"
    )

    y_variable = evidence.get(
        "y_variable"
    )

    method = evidence.get(
        "method",
        "Pearson"
    )

    correlation = evidence.get(
        "correlation"
    )

    direction = evidence.get(
        "direction"
    )

    strength = evidence.get(
        "strength"
    )

    # --------------------------------------------------------
    # Validate required information
    # --------------------------------------------------------

    if x_variable is None:

        return {
            "status": "FAIL",
            "reason": (
                "X variable is missing from "
                "correlation evidence."
            )
        }

    if y_variable is None:

        return {
            "status": "FAIL",
            "reason": (
                "Y variable is missing from "
                "correlation evidence."
            )
        }

    if correlation is None:

        return {
            "status": "FAIL",
            "reason": (
                "Correlation value is missing "
                "from evidence."
            )
        }

    if direction is None:

        return {
            "status": "FAIL",
            "reason": (
                "Correlation direction is missing "
                "from evidence."
            )
        }

    if strength is None:

        return {
            "status": "FAIL",
            "reason": (
                "Correlation strength is missing "
                "from evidence."
            )
        }

    # --------------------------------------------------------
    # Format correlation
    # --------------------------------------------------------

    correlation_text = format_number(
        correlation
    )

    # --------------------------------------------------------
    # Generate recommendation
    # --------------------------------------------------------

    if direction == "positive":

        relationship_text = (
            f"{x_variable} and {y_variable} show a "
            f"{strength} positive relationship"
        )

    elif direction == "negative":

        relationship_text = (
            f"{x_variable} and {y_variable} show a "
            f"{strength} negative relationship"
        )

    else:

        relationship_text = (
            f"{x_variable} and {y_variable} show "
            f"little or no linear relationship"
        )

    recommendation = (
        f"{relationship_text}, with a {method} "
        f"correlation of approximately "
        f"{correlation_text}. "
        f"Investigate whether this relationship "
        f"remains consistent across regions, product "
        f"categories, and sales channels before making "
        f"business decisions. Correlation indicates "
        f"association and should not be interpreted "
        f"as proof of causation."
    )

    # --------------------------------------------------------
    # Return structured result
    # --------------------------------------------------------

    return {
        "status": "PASS",
        "recommendation": recommendation,
        "basis": {
            "analysis_type": "relationship_analysis",
            "x_variable": x_variable,
            "y_variable": y_variable,
            "method": method,
            "correlation": float(correlation),
            "direction": direction,
            "strength": strength
        }
    }


# ============================================================
# 4. MAIN RECOMMENDATION GENERATOR
# ============================================================

def generate_recommendation(
    task_type,
    evidence
):
    """
    Generate a recommendation based on the validated
    evidence.

    Parameters
    ----------
    task_type : str
        Type of analysis performed.

    evidence : dict
        Evidence produced by the evidence mapping stage.

    Returns
    -------
    dict
        Structured recommendation.
    """

    # --------------------------------------------------------
    # Validate task type
    # --------------------------------------------------------

    if task_type is None:

        task_type = "generic"

    task_type = str(
        task_type
    ).lower()

    # --------------------------------------------------------
    # Grouped comparison
    # --------------------------------------------------------

    if (
        task_type == "comparison"
        or evidence.get(
            "analysis_type"
        ) == "grouped_analysis"
    ):

        return generate_grouped_recommendation(
            evidence
        )

    # --------------------------------------------------------
    # Relationship / correlation analysis
    # --------------------------------------------------------

    if (
        task_type == "relationship_analysis"
        or evidence.get(
            "analysis_type"
        ) == "correlation_analysis"
    ):

        return generate_correlation_recommendation(
            evidence
        )

    # --------------------------------------------------------
    # Mean analysis
    # --------------------------------------------------------

    if (
        "mean" in task_type
        or "average" in task_type
    ):

        return generate_mean_recommendation(
            evidence
        )

    # --------------------------------------------------------
    # Generic recommendation
    # --------------------------------------------------------

    if not isinstance(evidence, dict):

        return {
            "status": "FAIL",
            "reason": (
                "Evidence must be a dictionary."
            )
        }

    if evidence.get("status") != "PASS":

        return {
            "status": "FAIL",
            "reason": (
                "Evidence validation did not pass."
            )
        }

    recommendation = (
        "Use the validated EDA evidence as the "
        "basis for further investigation and "
        "decision-making. Additional analysis "
        "should be performed before taking "
        "business-critical action."
    )

    return {
        "status": "PASS",
        "recommendation": recommendation,
        "basis": {
            "analysis_type": task_type
        }
    }


# ============================================================
# MODULE TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("RECOMMENDATION GENERATOR TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Test evidence - GROUPED
    # --------------------------------------------------------

    test_evidence = {

        "status": "PASS",

        "analysis_type": "grouped_analysis",

        "category": "Region",

        "metric": "Total Profit",

        "operation": "SUM",

        "evidence": {

            "top_category":
                "Sub-Saharan Africa",

            "top_value":
                12183211.40,

            "second_category":
                "Europe",

            "second_value":
                11082938.63
        },

        "supporting_result": {

            "Sub-Saharan Africa":
                12183211.40,

            "Europe":
                11082938.63,

            "Asia":
                6113845.87,

            "Middle East and North Africa":
                5761191.86,

            "Australia and Oceania":
                4722160.03,

            "Central America and the Caribbean":
                2846907.85,

            "North America":
                1457942.76
        }
    }

    # --------------------------------------------------------
    # TEST 1 - GROUPED RECOMMENDATION
    # --------------------------------------------------------

    print("\nTEST 1 - GROUPED RECOMMENDATION")

    recommendation = generate_recommendation(
        task_type="comparison",
        evidence=test_evidence
    )

    print("\nGenerated Recommendation:")
    print(recommendation)

    # --------------------------------------------------------
    # TEST 2 - INVALID EVIDENCE
    # --------------------------------------------------------

    print("\n" + "-" * 70)

    print("TEST 2 - INVALID EVIDENCE")

    invalid_evidence = {

        "status": "FAIL",

        "reason":
            "Invalid analysis result."
    }

    recommendation = generate_recommendation(
        task_type="comparison",
        evidence=invalid_evidence
    )

    print("\nGenerated Recommendation:")
    print(recommendation)

    # --------------------------------------------------------
    # TEST 3 - MEAN RECOMMENDATION
    # --------------------------------------------------------

    print("\n" + "-" * 70)

    print("TEST 3 - MEAN RECOMMENDATION")

    mean_evidence = {

        "status": "PASS",

        "analysis_type":
            "mean_analysis",

        "metric":
            "Total Profit",

        "value":
            7500000.50
    }

    recommendation = generate_recommendation(
        task_type="mean",
        evidence=mean_evidence
    )

    print("\nGenerated Recommendation:")
    print(recommendation)

    # --------------------------------------------------------
    # TEST 4 - CORRELATION RECOMMENDATION
    # --------------------------------------------------------

    print("\n" + "-" * 70)

    print("TEST 4 - CORRELATION RECOMMENDATION")

    correlation_evidence = {

        "status": "PASS",

        "analysis_type":
            "correlation_analysis",

        "x_variable":
            "Units Sold",

        "y_variable":
            "Total Profit",

        "method":
            "Pearson",

        "correlation":
            0.5645504620845976,

        "direction":
            "positive",

        "strength":
            "moderate"
    }

    recommendation = generate_recommendation(
        task_type="relationship_analysis",
        evidence=correlation_evidence
    )

    print("\nGenerated Recommendation:")
    print(recommendation)

    # --------------------------------------------------------
    # TEST 5 - INVALID CORRELATION EVIDENCE
    # --------------------------------------------------------

    print("\n" + "-" * 70)

    print("TEST 5 - INVALID CORRELATION EVIDENCE")

    invalid_correlation_evidence = {

        "status": "PASS",

        "analysis_type":
            "correlation_analysis",

        "x_variable":
            "Units Sold",

        "y_variable":
            "Total Profit"
    }

    recommendation = generate_recommendation(
        task_type="relationship_analysis",
        evidence=invalid_correlation_evidence
    )

    print("\nGenerated Recommendation:")
    print(recommendation)

    print("\n" + "=" * 70)
    print("RECOMMENDATION GENERATOR TEST COMPLETED")
    print("=" * 70)