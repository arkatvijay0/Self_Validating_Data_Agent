
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
# CORRELATION INTERPRETATION
# ============================================================

def interpret_correlation(value):
    """Describe the strength and direction of a correlation."""
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
    """
    Summarize a descriptive-statistics DataFrame.

    Expected layout:
        Rows: metrics such as Total Revenue and Total Profit
        Columns: mean, median, min, max, skewness, and so on.
    """

    if not isinstance(result, pd.DataFrame) or result.empty:
        return (
            "A descriptive summary could not be generated because "
            "no valid statistics table was supplied."
        )

    stats = result.copy()

    # Correctly select metric rows, not DataFrame columns.
    identifier_names = {
        "id",
        "order id",
        "customer id",
        "transaction id",
        "record id",
        "invoice id",
        "account id",
    }

    metric_rows = [
        metric
        for metric in stats.index
        if str(metric).strip().lower() not in identifier_names
    ]

    stats = stats.loc[metric_rows].copy()

    if stats.empty:
        return (
            "No suitable business measures were available after "
            "excluding identifier fields."
        )

    lines = ["Descriptive Summary:"]

    lines.append(
        "- Numerical business measures analyzed: "
        + ", ".join(str(metric) for metric in stats.index)
        + "."
    )

    priority_metrics = [
        "Total Revenue",
        "Total Profit",
        "Units Sold",
    ]

    reported_metrics = set()

    for metric in priority_metrics:
        if metric not in stats.index:
            continue

        row = stats.loc[metric]
        reported_metrics.add(metric)

        count = safe_float(row.get("count"))
        mean = safe_float(row.get("mean"))
        median = safe_float(row.get("median"))

        if median is None:
            median = safe_float(row.get("50%"))

        minimum = safe_float(row.get("min"))
        maximum = safe_float(row.get("max"))
        skewness = safe_float(row.get("skewness"))
        missing_count = safe_float(row.get("missing_count"))

        detail = f"- {metric}:"

        if count is not None:
            detail += f" {int(count)} valid observations"

        if mean is not None:
            detail += f"; mean {format_number(mean)}"

        if median is not None:
            detail += f"; median {format_number(median)}"

        if minimum is not None and maximum is not None:
            detail += (
                f"; range {format_number(minimum)} to "
                f"{format_number(maximum)}"
            )

        lines.append(detail + ".")

        if skewness is not None:
            if skewness > 0.5:
                distribution = "positively skewed"
            elif skewness < -0.5:
                distribution = "negatively skewed"
            else:
                distribution = "approximately symmetric"

            lines.append(
                f"  - Distribution: {distribution} "
                f"(skewness {skewness:.2f})."
            )

        if missing_count is not None:
            lines.append(
                f"  - Missing values in supplied statistics: "
                f"{int(missing_count)}."
            )

    # Include other numeric measures without duplicating priority metrics.
    for metric in stats.index:
        if metric in reported_metrics:
            continue

        row = stats.loc[metric]
        mean = safe_float(row.get("mean"))
        median = safe_float(row.get("median"))

        if median is None:
            median = safe_float(row.get("50%"))

        if mean is not None:
            detail = f"- {metric}: mean {format_number(mean)}"

            if median is not None:
                detail += f"; median {format_number(median)}"

            lines.append(detail + ".")

    lines.append(
        "- Identifier fields were excluded from business-measure statistics."
    )

    lines.append(
        "- Dataset structure, categorical frequencies, and missing values "
        "across all columns require their respective validated step results; "
        "they cannot be inferred from this statistics table alone."
    )

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

    total_missing = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    lines.append(f"- Total missing cells: {total_missing:,}.")
    lines.append(f"- Exact duplicate rows: {duplicate_rows:,}.")

    missing_by_column = df.isna().sum()
    columns_with_missing = missing_by_column[missing_by_column > 0]
    if columns_with_missing.empty:
        lines.append("- Missing-value check: no missing values were found.")
    else:
        lines.append("- Columns with missing values:")
        for column, count in columns_with_missing.items():
            percentage = float(count) / max(len(df), 1) * 100
            lines.append(f"  - {column}: {int(count):,} ({percentage:.2f}%).")

    numeric_df = df.select_dtypes(include="number")
    identifier_tokens = (" id", "_id", "identifier", "account number", "record number")
    business_numeric_columns = [
        column for column in numeric_df.columns
        if not any(token in str(column).lower() for token in identifier_tokens)
        and str(column).strip().lower() != "id"
    ]

    if business_numeric_columns:
        lines.append("- Numerical business-measure summary:")
        for column in business_numeric_columns:
            values = pd.to_numeric(numeric_df[column], errors="coerce").dropna()
            if values.empty:
                lines.append(f"  - {column}: no valid numeric observations.")
                continue
            line = (
                f"  - {column}: {len(values):,} valid observations; "
                f"mean {format_number(values.mean())}; "
                f"median {format_number(values.median())}; "
                f"minimum {format_number(values.min())}; "
                f"maximum {format_number(values.max())}."
            )
            skewness_value = safe_float(values.skew())
            if skewness_value is not None:
                if skewness_value > 0.5:
                    shape = "positively skewed"
                elif skewness_value < -0.5:
                    shape = "negatively skewed"
                else:
                    shape = "approximately symmetric"
                line += f" Distribution is {shape} (skewness {skewness_value:.2f})."
            lines.append(line)

    categorical_df = df.select_dtypes(include=["object", "category", "bool"])
    identifier_name_tokens = (" id", "identifier", "transaction", "invoice")
    useful_categorical_columns = [
        column for column in categorical_df.columns
        if not any(token in str(column).lower() for token in identifier_name_tokens)
        and "date" not in str(column).lower()
        and categorical_df[column].nunique(dropna=False) <= max(30, int(len(df) * 0.5))
    ]
    if useful_categorical_columns:
        lines.append("- Most frequent categorical values (up to three per column):")
        for column in useful_categorical_columns:
            counts = df[column].fillna("<MISSING>").astype(str).value_counts().head(3)
            formatted_counts = ", ".join(
                f"{category} ({int(count):,})" for category, count in counts.items()
            )
            lines.append(f"  - {column}: {formatted_counts}.")

    for column in df.columns:
        if "date" not in str(column).lower():
            continue
        parsed_dates = pd.to_datetime(df[column], errors="coerce").dropna()
        if not parsed_dates.empty:
            lines.append(
                f"- {column} range: {parsed_dates.min().date()} to "
                f"{parsed_dates.max().date()} ({len(parsed_dates):,} valid dates)."
            )

    if "Region" in df.columns and "Total Profit" in df.columns:
        profit_values = pd.to_numeric(df["Total Profit"], errors="coerce")
        grouped_profit = (
            pd.DataFrame({"Region": df["Region"], "Total Profit": profit_values})
            .dropna(subset=["Region", "Total Profit"])
            .groupby("Region")["Total Profit"]
            .sum()
            .sort_values(ascending=False)
        )
        if not grouped_profit.empty:
            lines.append(
                f"- Highest total profit by region: {grouped_profit.index[0]}, "
                f"with {format_number(grouped_profit.iloc[0])} across the available records."
            )
            if len(grouped_profit) > 1:
                lines.append(
                    f"- Lowest total profit by region: {grouped_profit.index[-1]}, "
                    f"with {format_number(grouped_profit.iloc[-1])}."
                )

    lines.append(
        "- These are descriptive findings from this dataset; they do not "
        "by themselves establish causation or guarantee future performance."
    )
    return "\n".join(lines)


# ============================================================
# CORRELATION INSIGHT
# ============================================================

def generate_correlation_insight(result, user_question):
    """Generate an insight from a correlation coefficient."""

    if isinstance(result, dict):
        for key in ("pearson", "correlation", "value"):
            if key in result:
                result = result[key]
                break

    correlation_value = safe_float(result)

    if (
        correlation_value is None
        or not -1 <= correlation_value <= 1
    ):
        return "No valid correlation result was available for insight generation."

    interpretation = interpret_correlation(correlation_value)
    question_lower = str(user_question).lower()

    if "units sold" in question_lower and "total profit" in question_lower:
        x_name = "Units Sold"
        y_name = "Total Profit"
    else:
        x_name = "the first variable"
        y_name = "the second variable"

    if correlation_value > 0:
        association = "higher values tend to occur together"
    elif correlation_value < 0:
        association = (
            "higher values of one tend to accompany lower values of the other"
        )
    else:
        association = "no linear association was observed"

    return (
        f"The correlation between {x_name} and {y_name} is "
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

    question_lower = str(user_question).lower()

    if "total profit" in question_lower:
        metric = "Total Profit"
    elif "total revenue" in question_lower:
        metric = "Total Revenue"
    elif "units sold" in question_lower:
        metric = "Units Sold"
    else:
        metric = "the selected metric"

    return f"The average {metric} is {format_number(value)}."


# ============================================================
# GENERIC FALLBACK
# ============================================================

def generate_generic_insight(result):
    """Generate a safe fallback insight."""

    if result is None:
        return "No valid result was available for insight generation."

    if isinstance(result, pd.DataFrame):
        return generate_descriptive_insight(result)

    if isinstance(result, pd.Series):
        return generate_grouped_insight(result)

    if isinstance(result, dict):
        if any(key in result for key in ("pearson", "correlation")):
            return generate_correlation_insight(
                result,
                "correlation analysis",
            )

        return (
            "A structured result was generated. Review the validated "
            "result and its supporting evidence for specific findings."
        )

    value = safe_float(result)

    if value is not None:
        return f"The calculated result was {format_number(value)}."

    return (
        "A result was produced, but it could not be safely converted "
        "into a numerical insight."
    )


# ============================================================
# BANK CONVERSION INSIGHT
# ============================================================

def generate_conversion_insight(dataset_path):
    """Calculate conversion counts, rates, and label consistency from the dataset."""
    try:
        df = pd.read_csv(dataset_path)
    except Exception as error:
        return f"Conversion analysis could not load the dataset: {error}"

    required = {"conversion", "y"}
    missing = sorted(required - set(df.columns))
    if missing:
        return "Conversion analysis cannot run; missing required columns: " + ", ".join(missing)

    conversion = pd.to_numeric(df["conversion"], errors="coerce")
    y_mapped = df["y"].astype("string").str.strip().str.lower().map({"yes": 1, "no": 0})
    total = len(df)
    converted = int((conversion == 1).sum())
    not_converted = int((conversion == 0).sum())
    valid_conversion = int(conversion.isin([0, 1]).sum())
    valid_y = int(y_mapped.isin([0, 1]).sum())
    comparable = conversion.notna() & y_mapped.notna()
    matches = int((conversion[comparable] == y_mapped[comparable]).sum())
    mismatches = int(comparable.sum() - matches)
    rate = converted / total * 100 if total else 0.0
    non_rate = not_converted / total * 100 if total else 0.0
    validation = "PASS" if valid_conversion == total and valid_y == total and mismatches == 0 else "FAIL"

    return (
        "Bank Marketing Conversion Analysis:\n"
        f"- Total records: {total:,}.\n"
        f"- Converted customers (conversion = 1): {converted:,}.\n"
        f"- Non-converted customers (conversion = 0): {not_converted:,}.\n"
        f"- Conversion rate = {converted:,} / {total:,} × 100 = {rate:.4f}%.\n"
        f"- Non-conversion rate = {not_converted:,} / {total:,} × 100 = {non_rate:.4f}%.\n"
        f"- Valid conversion values (0 or 1): {valid_conversion:,} of {total:,}.\n"
        f"- Valid y labels after yes→1/no→0 mapping: {valid_y:,} of {total:,}.\n"
        f"- Row-by-row matches: {matches:,}; mismatches: {mismatches:,}.\n"
        f"- Conversion consistency validation: {validation}.\n"
        "- These are descriptive results for the supplied dataset; they do not establish causation."
    )


# ============================================================
# DIRECT DATA QUALITY QUESTION INSIGHTS
# ============================================================

def generate_missing_values_insight(dataset_path):
    """Answer missing-value questions directly from the selected dataset."""
    if not dataset_path:
        return (
            "Missing-value analysis could not be independently verified "
            "because the dataset path was not supplied."
        )
    try:
        df = pd.read_csv(dataset_path)
        total_missing = int(df.isna().sum().sum())
        columns_with_missing = df.isna().sum()
        columns_with_missing = columns_with_missing[columns_with_missing > 0]
        if columns_with_missing.empty:
            return (
                f"The dataset contains {total_missing:,} missing cells across "
                f"{len(df):,} rows and {len(df.columns):,} columns. "
                "No missing values were found in any column."
            )
        details = ", ".join(
            f"{column}: {int(count):,}"
            for column, count in columns_with_missing.items()
        )
        return (
            f"The dataset contains {total_missing:,} missing cells across "
            f"{len(df):,} rows and {len(df.columns):,} columns. "
            f"Columns with missing values: {details}."
        )
    except Exception as error:
        return f"Missing-value analysis could not be completed: {error}"


# ============================================================
# MAIN INSIGHT GENERATOR
# ============================================================


def generate_conversion_factor_insight(result):
    """Summarize validated conversion-factor results without causal claims."""
    def find_result(obj, depth=0):
        if depth > 6 or not isinstance(obj, dict):
            return None
        if (
            obj.get("analysis_type") == "conversion_factor_analysis"
            and isinstance(obj.get("categorical_factor_results"), dict)
            and isinstance(obj.get("numeric_factor_results"), dict)
        ):
            return obj
        for key in (
            "primary_result", "result", "supporting_result",
            "evidence", "validated_analysis_result"
        ):
            found = find_result(obj.get(key), depth + 1)
            if found is not None:
                return found
        return None

    data = find_result(result)
    if data is None:
        return (
            "Conversion-factor insights could not be generated because "
            "the validated categorical and numeric factor results were unavailable."
        )

    total = data.get("valid_target_records", data.get("total_dataset_records"))
    converted = data.get("converted_count")
    rate = data.get("overall_conversion_rate_percent")
    if not isinstance(total, (int, float)) or not isinstance(converted, (int, float)) \
            or not isinstance(rate, (int, float)) or total <= 0:
        return "Conversion-factor insights could not be generated because the overall conversion metrics are invalid."

    sentences = [
        f"The dataset contains {int(total):,} valid customer records, of which "
        f"{int(converted):,} converted, giving an overall conversion rate of {rate:.2f}%."
    ]

    candidates = []
    for column, groups in data["categorical_factor_results"].items():
        if not isinstance(groups, list):
            continue
        eligible = [
            g for g in groups
            if isinstance(g, dict)
            and g.get("eligible_for_comparison", True)
            and isinstance(g.get("conversion_rate_percent"), (int, float))
            and isinstance(g.get("records"), (int, float))
            and g.get("records", 0) > 0
        ]
        if len(eligible) >= 2:
            eligible.sort(key=lambda g: g["conversion_rate_percent"], reverse=True)
            high, low = eligible[0], eligible[-1]
            candidates.append((
                high["conversion_rate_percent"] - low["conversion_rate_percent"],
                column, high, low
            ))

    candidates.sort(key=lambda x: x[0], reverse=True)
    for _, column, high, low in candidates[:4]:
        sentences.append(
            f"For {column}, the highest observed group was '{high['group']}' "
            f"at {high['conversion_rate_percent']:.2f}% "
            f"(n={int(high['records']):,}), while '{low['group']}' was "
            f"{low['conversion_rate_percent']:.2f}% "
            f"(n={int(low['records']):,})."
        )

    numeric_candidates = []
    for column, groups in data["numeric_factor_results"].items():
        if not isinstance(groups, list):
            continue
        eligible = [
            g for g in groups
            if isinstance(g, dict)
            and g.get("eligible_for_comparison", True)
            and isinstance(g.get("conversion_rate_percent"), (int, float))
            and isinstance(g.get("records"), (int, float))
            and g.get("records", 0) > 0
        ]
        if len(eligible) >= 2:
            eligible.sort(key=lambda g: g["conversion_rate_percent"], reverse=True)
            numeric_candidates.append((eligible[0]["conversion_rate_percent"], column, eligible[0]))

    numeric_candidates.sort(reverse=True)
    for rate_value, column, group in numeric_candidates[:2]:
        sentences.append(
            f"For numeric factor {column}, group '{group['group']}' had an observed "
            f"conversion rate of {rate_value:.2f}% (n={int(group['records']):,})."
        )

    sentences.append(
        "These are observational associations, not proof of causation. "
        "Call duration is measured after contact and should not be treated as "
        "a pre-contact targeting feature."
    )
    return " ".join(sentences)

def generate_insight(user_question, result, task_analysis, dataset_path=None):
    """
    Generate a natural-language insight.

    Keeps the existing three-argument interface used by main.py.
    """

    question = str(user_question or "")
    question_lower = question.lower()

    if isinstance(task_analysis, dict):
        task_type = str(
            task_analysis.get("task_type", "")
        ).lower()
    else:
        task_type = str(task_analysis or "").lower()

    # Dedicated route for validated conversion-factor analysis.
    result_type = (
        str(result.get("analysis_type", "")).lower()
        if isinstance(result, dict) else ""
    )
    if (
        result_type == "conversion_factor_analysis"
        or task_type == "conversion_factor_analysis"
        or (
            "conversion" in question_lower
            and any(term in question_lower for term in
                    ("factor", "associated", "patterns", "influence"))
            and isinstance(result, dict)
            and isinstance(result.get("categorical_factor_results"), dict)
        )
    ):
        return generate_conversion_factor_insight(result)

    # Answer data-quality questions directly from the source dataset.
    missing_value_terms = (
        "missing value", "missing values", "missing data", "null value",
        "null values", "nulls", "nan values", "how many missing",
    )
    if any(term in question_lower for term in missing_value_terms):
        return generate_missing_values_insight(dataset_path)

    # Filtered monthly mean: independently recompute from the source dataset.
    month_codes = {
        "january": "jan", "february": "feb", "march": "mar", "april": "apr",
        "may": "may", "june": "jun", "july": "jul", "august": "aug",
        "september": "sep", "october": "oct", "november": "nov", "december": "dec",
    }
    selected_month = next((name for name in month_codes if name in question_lower), None)
    if "duration" in question_lower and any(word in question_lower for word in ("average", "mean", "avg")) and selected_month:
        if not dataset_path:
            return "Filtered average call duration cannot be verified because the dataset path was not supplied."
        try:
            source_df = pd.read_csv(dataset_path)
            if not {"month", "duration"}.issubset(source_df.columns):
                return "Filtered average call duration failed: required columns 'month' and 'duration' are missing."
            month_values = source_df["month"].astype("string").str.strip().str.lower()
            filtered_df = source_df.loc[month_values.isin([month_codes[selected_month], selected_month])]
            durations = pd.to_numeric(filtered_df["duration"], errors="coerce").dropna()
            if filtered_df.empty or durations.empty:
                return f"No valid duration observations were found for {selected_month.title()}."
            return (
                f"Filtered average call duration for {selected_month.title()}: "
                f"{durations.mean():.2f}. "
                f"Rows matching the month: {len(filtered_df):,}; "
                f"valid duration observations: {len(durations):,}. "
                f"Calculation: sum of valid {selected_month.title()} duration values divided by "
                f"the number of valid duration observations. This is descriptive, not causal."
            )
        except Exception as error:
            return f"Filtered average call duration could not be verified: {error}"

    # Route specific conversion questions before generic descriptive summaries.
    if (
        "conversion" in question_lower
        and any(token in question_lower for token in [
            "rate", "percentage", "converted", "conversion = 1",
            "conversion=1", "matches the y column", "row-by-row"
        ])
    ):
        if dataset_path:
            return generate_conversion_insight(dataset_path)
        return "Conversion analysis requires the dataset path to independently verify conversion and y values."

    # Handle specific mean/average questions before the generic dataset summary.
    mean_terms = ("average", "avg", "mean")
    is_mean_question = any(re.search(r"\b" + re.escape(term) + r"\b", question_lower) for term in mean_terms)
    if is_mean_question:
        # The dataset is the source of truth for both column existence and calculation.
        if dataset_path:
            try:
                source_df = pd.read_csv(dataset_path)
            except Exception as error:
                return f"The requested average could not be calculated because the dataset could not be read: {error}"

            # Resolve only whole-word column-name matches, avoiding false matches such as
            # 'age' in 'average' and 'y' in 'salary'.
            requested_column = None
            normalized_question = re.sub(r"[_\-]+", " ", question_lower)
            for column in source_df.columns:
                normalized_column = re.sub(r"[_\-]+", " ", str(column).lower()).strip()
                if normalized_column and re.search(r"(?<![a-z0-9])" + re.escape(normalized_column) + r"(?![a-z0-9])", normalized_question):
                    requested_column = column
                    break

            # Small, explicit aliases for common customer-data wording.
            if requested_column is None:
                aliases = {
                    "customer age": "age",
                    "age of customers": "age",
                    "customers age": "age",
                    "call duration": "duration",
                    "duration of calls": "duration",
                }
                for phrase, column_name in aliases.items():
                    if phrase in normalized_question and column_name in source_df.columns:
                        requested_column = column_name
                        break

            if requested_column is None:
                return (
                    "I cannot calculate the requested average because the dataset does not contain "
                    "a clearly identified column for that metric. Please ask for a column that exists "
                    "in the dataset."
                )

            values = pd.to_numeric(source_df[requested_column], errors="coerce").dropna()
            if values.empty:
                return f"I cannot calculate the average of '{requested_column}' because it has no valid numeric values."

            # Apply a month filter when the question names one and the dataset has a month column.
            month_codes = {
                "january": "jan", "february": "feb", "march": "mar", "april": "apr",
                "may": "may", "june": "jun", "july": "jul", "august": "aug",
                "september": "sep", "october": "oct", "november": "nov", "december": "dec",
            }
            selected_month = next((month for month in month_codes if re.search(r"\b" + month + r"\b", question_lower)), None)
            filtered_rows = len(source_df)
            if selected_month and "month" in source_df.columns:
                month_values = source_df["month"].astype("string").str.strip().str.lower()
                filtered_df = source_df.loc[month_values.isin([month_codes[selected_month], selected_month])]
                values = pd.to_numeric(filtered_df[requested_column], errors="coerce").dropna()
                filtered_rows = len(filtered_df)
                if values.empty:
                    return f"No valid values for '{requested_column}' were found for {selected_month.title()}."
                return (
                    f"The average {str(requested_column).replace('_', ' ')} for {selected_month.title()} is "
                    f"{values.mean():,.2f}, based on {len(values):,} valid observations "
                    f"across {filtered_rows:,} rows matching that month."
                )

            return (
                f"The average {str(requested_column).replace('_', ' ')} is {values.mean():,.2f}, "
                f"based on {len(values):,} valid numeric observations."
            )

        # Without a dataset path, use a numeric result only when one was actually supplied.
        raw_value = result
        if isinstance(result, dict):
            for key in ("mean", "average", "mean_value", "average_value", "mean_duration", "value"):
                if key in result:
                    raw_value = result[key]
                    break
        value = safe_float(raw_value)
        if value is not None:
            return f"The calculated average is {format_number(value)}."

        return "The requested average could not be verified because the dataset path or a numeric mean result was not supplied."

    # Descriptive analysis.
    if (
        task_type in {
            "descriptive_summary",
            "descriptive_analysis",
            "descriptive",
            "descriptive_statistics",
        }
        or "complete summary" in question_lower
        or "descriptive summary" in question_lower
        or "summarize the dataset" in question_lower
        or "summary of the dataset" in question_lower
    ):
        if dataset_path:
            return generate_dataset_summary_insight(dataset_path, result)
        return generate_descriptive_insight(result, question)

    # Relationship analysis.
    if (
        task_type in {"relationship_analysis", "correlation_analysis"}
        or "correlation" in question_lower
        or "relationship" in question_lower
    ):
        return generate_correlation_insight(result, question)

    # Grouped comparison.
    if task_type in {"comparison", "grouped_analysis", "grouped"}:
        return generate_grouped_insight(result)

    # Mean calculation.
    if task_type in {"mean", "average"}:
        return generate_mean_insight(result, question)

    return generate_generic_insight(result)


# ============================================================
# MODULE TESTS
# ============================================================

if __name__ == "__main__":

    print("=" * 65)
    print("INSIGHT GENERATOR TESTS")
    print("=" * 65)

    # Test 1: Grouped comparison.
    grouped_result = pd.Series({
        "Sub-Saharan Africa": 12183211.40,
        "Europe": 11082938.63,
        "Asia": 6113845.87,
    })

    print("\nTEST 1 - GROUPED COMPARISON")
    print(generate_insight(
        "Which region generated the highest total profit?",
        grouped_result,
        {"task_type": "comparison"},
    ))

    # Test 2: Correlation.
    print("\nTEST 2 - CORRELATION")
    print(generate_insight(
        "Is there a relationship between Units Sold and Total Profit?",
        0.5645504620845976,
        {"task_type": "relationship_analysis"},
    ))

    # Test 3: Mean.
    print("\nTEST 3 - MEAN")
    print(generate_insight(
        "What is the average Total Profit?",
        441681.984,
        {"task_type": "mean"},
    ))

    # Test 4: Descriptive summary.
    print("\nTEST 4 - DESCRIPTIVE SUMMARY")

    sample_stats = pd.DataFrame({
        "count": [100, 100, 100],
        "mean": [555020412.36, 1373487.6831, 441681.984],
        "std": [260615257.13, 1460028.71, 438537.91],
        "min": [114606559.0, 4870.26, 1258.02],
        "25%": [338922488.0, 268721.2125, 121443.585],
        "50%": [557708561.0, 752314.36, 290767.995],
        "75%": [790755080.75, 2212044.6825, 635828.8],
        "max": [994022214.0, 5997054.98, 1719922.04],
        "median": [557708561.0, 752314.36, 290767.995],
        "skewness": [-0.098, 1.278, 1.239],
        "missing_count": [0, 0, 0],
    }, index=["Order ID", "Total Revenue", "Total Profit"])

    summary = generate_insight(
        "Give me a complete summary of the sales dataset",
        sample_stats,
        {"task_type": "descriptive_summary"},
    )

    print(summary)

    assert "Order ID:" not in summary
    assert "Total Revenue" in summary
    assert "Total Profit" in summary

    print("\nAll insight-generator tests completed successfully.")
