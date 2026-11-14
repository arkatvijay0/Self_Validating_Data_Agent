import math
import re
import pandas as pd


# ============================================================
# NUMBER FORMATTING
# ============================================================

def format_number(value):
    """Format a numerical value safely."""
    try:
        number = float(value)
        if not math.isfinite(number):
            return "not available"
        return f"{number:,.2f}"
    except (TypeError, ValueError):
        return str(value)


def safe_float(value):
    """Convert a value to a finite float, or return None."""
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


# ============================================================
# CORRELATION INTERPRETATION & EXTRACTION
# ============================================================

def _extract_correlation_data(result):
    """Recursively extract correlation coefficient and variables from nested structures."""
    if isinstance(result, (int, float)):
        return safe_float(result), None, None
    
    val = None
    x_var = None
    y_var = None
    
    if isinstance(result, dict):
        for k in ("pearson", "correlation", "value", "pearson_correlation", "coefficient", "r"):
            if k in result:
                sub = result[k]
                if isinstance(sub, (int, float)):
                    val = safe_float(sub)
                    break
                elif isinstance(sub, dict):
                    v, x, y = _extract_correlation_data(sub)
                    if v is not None:
                        val = v
                        if x: x_var = x
                        if y: y_var = y
                        break
        
        if val is None and "operation_results" in result:
            v, x, y = _extract_correlation_data(result["operation_results"])
            if v is not None:
                val = v
                if x: x_var = x
                if y: y_var = y

        # If it's a correlation matrix (dict of dicts), prioritize off-diagonal elements
        if val is None or val == 1.0:
            matrix_vals = []
            off_diagonal_val = None
            off_diag_x = None
            off_diag_y = None
            for k1, subdict in result.items():
                if isinstance(subdict, dict):
                    for k2, v2 in subdict.items():
                        num = safe_float(v2)
                        if num is not None:
                            if k1 != k2:
                                off_diagonal_val = num
                                off_diag_x = k1
                                off_diag_y = k2
                                break
                            matrix_vals.append((num, k1, k2))
                    if off_diagonal_val is not None:
                        break
            if off_diagonal_val is not None:
                return off_diagonal_val, off_diag_x, off_diag_y
            elif matrix_vals:
                return matrix_vals[0][0], matrix_vals[0][1], matrix_vals[0][2]

        if val is None:
            for k, v in result.items():
                if isinstance(v, (int, float)) and -1 <= float(v) <= 1 and k not in ("analyzed_rows", "input_rows", "status"):
                    val = safe_float(v)
                    break
                elif isinstance(v, dict):
                    sub_v, sub_x, sub_y = _extract_correlation_data(v)
                    if sub_v is not None:
                        val = sub_v
                        if sub_x: x_var = sub_x
                        if sub_y: y_var = sub_y
                        break
                        
        if "x_variable" in result:
            x_var = result["x_variable"]
        if "y_variable" in result:
            y_var = result["y_variable"]
            
    return val, x_var, y_var


def interpret_correlation(value):
    """Describe the strength and direction of a correlation using standardized thresholds."""
    value = safe_float(value)

    if value is None:
        return "unable to determine the relationship"

    absolute_value = abs(value)

    if absolute_value < 0.20:
        strength = "very weak"
    elif absolute_value < 0.40:
        strength = "weak"
    elif absolute_value < 0.60:
        strength = "moderate"
    elif absolute_value < 0.80:
        strength = "strong"
    else:
        strength = "very strong"

    if value == 0:
        return "no linear relationship"

    direction = "positive" if value > 0 else "negative"
    return f"{strength} {direction} relationship"


# ============================================================
# GROUPED COMPARISON INSIGHT
# ============================================================

def generate_grouped_insight(result):
    """Generate an insight from grouped numerical results."""

    if isinstance(result, dict):
        result = pd.Series(result)

    if not isinstance(result, pd.Series) or result.empty:
        return "No valid grouped result was available for insight generation."

    numeric_result = pd.to_numeric(
        result, errors="coerce"
    ).dropna()

    if numeric_result.empty:
        return "No valid numerical grouped result was available."

    numeric_result = numeric_result.sort_values(ascending=False)

    top_category = numeric_result.index[0]
    top_value = numeric_result.iloc[0]

    if len(numeric_result) > 1:
        second_category = numeric_result.index[1]
        second_value = numeric_result.iloc[1]

        return (
            f"{top_category} generated the highest grouped value, "
            f"at {format_number(top_value)}. "
            f"{second_category} ranked second, at "
            f"{format_number(second_value)}."
        )

    return (
        f"{top_category} had the highest grouped value, "
        f"at {format_number(top_value)}."
    )


# ============================================================
# DESCRIPTIVE SUMMARY INSIGHT
# ============================================================

def generate_descriptive_insight(result, user_question=""):
    """Summarize a descriptive-statistics DataFrame."""

    if not isinstance(result, pd.DataFrame) or result.empty:
        return (
            "A descriptive summary could not be generated because "
            "no valid statistics table was supplied."
        )

    stats = result.copy()

    identifier_names = {
        "id", "order id", "customer id", "transaction id",
        "record id", "invoice id", "account id",
    }

    metric_rows = [
        metric
        for metric in stats.index
        if str(metric).strip().lower() not in identifier_names
    ]

    stats = stats.loc[metric_rows].copy()

    if stats.empty:
        return "No suitable business measures were available after excluding identifier fields."

    lines = ["Descriptive Summary:"]

    lines.append(
        "- Numerical business measures analyzed: "
        + ", ".join(str(metric) for metric in stats.index)
        + "."
    )

    for metric in stats.index:
        row = stats.loc[metric]
        mean = safe_float(row.get("mean"))
        median = safe_float(row.get("median"))
        if median is None:
            median = safe_float(row.get("50%"))

        detail = f"- {metric}:"
        if mean is not None:
            detail += f" mean {format_number(mean)}"
        if median is not None:
            detail += f"; median {format_number(median)}"
        lines.append(detail + ".")

    return "\n".join(lines)


# ============================================================
# COMPREHENSIVE DATASET SUMMARY
# ============================================================

def generate_dataset_summary_insight(dataset_path, stats_result=None):
    """Create a detailed descriptive summary from the selected dataset."""
    try:
        df = pd.read_csv(dataset_path)
    except Exception as error:
        fallback = generate_descriptive_insight(stats_result)
        return fallback + f"\n- Additional dataset summary could not be loaded: {error}."

    lines = ["Comprehensive Dataset Summary:"]
    row_count, column_count = df.shape

    lines.append(f"- Dataset contains {row_count:,} rows and {column_count:,} columns.")
    lines.append("- Columns: " + ", ".join(str(c) for c in df.columns) + ".")
    lines.append(f"- Total missing cells: {int(df.isna().sum().sum()):,}.")
    lines.append(f"- Exact duplicate rows: {int(df.duplicated().sum()):,}.")

    return "\n".join(lines)


# ============================================================
# CORRELATION INSIGHT
# ============================================================

def generate_correlation_insight(result, user_question):
    """Generate an insight from a correlation coefficient."""
    correlation_value, x_var, y_var = _extract_correlation_data(result)
    correlation_value = safe_float(correlation_value)

    if correlation_value is None or not -1 <= correlation_value <= 1:
        return "No valid correlation result was available for insight generation."

    interpretation = interpret_correlation(correlation_value)
    question_lower = str(user_question).lower()

    if not x_var or not y_var:
        match = re.search(r"between\s+([a-zA-Z0-9_\s]+)\s+and\s+([a-zA-Z0-9_\s?]+)", question_lower)
        if match:
            x_var = match.group(1).strip()
            y_var = match.group(2).strip().rstrip("?")
        else:
            x_var = "the first variable"
            y_var = "the second variable"

    if correlation_value > 0:
        association = "higher values tend to occur together"
    elif correlation_value < 0:
        association = "higher values of one tend to accompany lower values of the other"
    else:
        association = "no linear association was observed"

    return (
        f"The correlation between {x_var} and {y_var} is "
        f"{correlation_value:.2f}, indicating a {interpretation}. "
        f"Within this dataset, {association}. "
        f"Correlation alone does not establish causation."
    )


# ============================================================
# MEAN INSIGHT
# ============================================================

def generate_mean_insight(result, user_question):
    """Generate an insight from a single mean value."""
    value = safe_float(result)
    if value is None:
        return "No valid mean result was available for insight generation."
    return f"The calculated mean is {format_number(value)}."


def generate_generic_insight(result):
    """Generate a safe fallback insight."""
    if result is None:
        return "No valid result was available for insight generation."
    value = safe_float(result)
    if value is not None:
        return f"The calculated result was {format_number(value)}."
    return "A result was produced, but it could not be safely converted into a numerical insight."


# ============================================================
# MAIN INSIGHT GENERATOR
# ============================================================

def generate_insight(
    user_question,
    result,
    task_analysis,
    dataset_path=None,
):
    """Generate an evidence-based insight."""
    question = str(user_question or "")
    question_lower = question.lower()

    if isinstance(task_analysis, dict):
        task_type = str(task_analysis.get("task_type", "")).lower()
    else:
        task_type = str(task_analysis or "").lower()

    if (
        task_type in {"relationship_analysis", "correlation_analysis"}
        or "correlation" in question_lower
        or "relationship" in question_lower
    ):
        return generate_correlation_insight(result, question)

    if task_type in {"mean", "average", "mean_analysis"}:
        return generate_mean_insight(result, question)

    return generate_generic_insight(result)