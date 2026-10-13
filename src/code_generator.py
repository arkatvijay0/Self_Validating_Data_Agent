# ============================================================
# CODE GENERATOR
# SELF-VALIDATING EDA AGENT
# Rule-based local implementation
# ============================================================

import os
import pandas as pd


# ============================================================
# COLUMN NORMALIZATION
# ============================================================

def normalize_column_name(column_name):
    """Normalize common aliases to actual dataset column names."""

    if column_name is None:
        return None

    normalized = str(column_name).strip()

    aliases = {
        "unitssold": "Units Sold",
        "totalprofit": "Total Profit",
        "totalrevenue": "Total Revenue",
        "totalcost": "Total Cost",
        "orderid": "Order ID",
    }

    return aliases.get(
        normalized.lower(),
        normalized
    )


# ============================================================
# COMMON CODE HEADER
# ============================================================

def build_code_header(filepath):
    """
    Build executable code that loads the requested dataset.

    The filepath is inserted into the generated code as an
    absolute path so execution does not depend on the
    terminal's current working directory.
    """

    filepath = os.path.abspath(filepath)

    return f'''import pandas as pd
import numpy as np

df = pd.read_csv(r"{filepath}")

'''


# ============================================================
# RULE-BASED CODE GENERATOR
# ============================================================

def generate_code_mock(
    step_text,
    filepath,
    data_profile=None,
    feedback=None
):
    """
    Generate executable Python code for an EDA plan step.

    Parameters
    ----------
    step_text : str
        The EDA step that needs code.

    filepath : str
        Path to the CSV dataset.

    data_profile : dict, optional
        Dataset profile supplied by the pipeline.
        This is used only by the generator itself when needed.
        Generated execution code does not depend on this
        variable being present in its execution namespace.

    feedback : str, optional
        Feedback from a previous failed execution.

    Returns
    -------
    str
        Complete executable Python code.
    """

    step_lower = str(step_text).strip().lower()

    code = build_code_header(filepath)


    if step_lower.startswith("stop:"):
        code += '''
result = {
    "status": "UNSUPPORTED",
    "reason": "The requested metric is not available.",
    "available_columns": df.columns.tolist(),
}
'''
        return code


    # Filtered average call duration for a named month.
    month_codes = {
        "january": "jan", "february": "feb", "march": "mar", "april": "apr",
        "may": "may", "june": "jun", "july": "jul", "august": "aug",
        "september": "sep", "october": "oct", "november": "nov", "december": "dec",
    }
    month_tokens = {**month_codes, **{code: code for code in month_codes.values()}}
    requested_month = next((name for name in month_tokens if name in step_lower), None)
    if "duration" in step_lower and any(word in step_lower for word in ("average", "mean", "avg")) and requested_month:
        month_code = month_codes.get(requested_month, requested_month)
        code += f"""
required_columns = ["month", "duration"]
missing_columns = [column for column in required_columns if column not in df.columns]
if missing_columns:
    result = {{"status": "FAIL", "reason": "Required columns missing: " + ", ".join(missing_columns)}}
else:
    month_values = df["month"].astype("string").str.strip().str.lower()
    filtered = df.loc[month_values.isin([{month_code!r}, {requested_month!r}])].copy()
    duration_values = pd.to_numeric(filtered["duration"], errors="coerce")
    valid_duration = duration_values.dropna()
    result = {{
        "status": "PASS" if not filtered.empty and not valid_duration.empty else "FAIL",
        "analysis_type": "filtered_mean",
        "filter_column": "month",
        "filter_value": {month_code!r},
        "metric_column": "duration",
        "filtered_rows": int(len(filtered)),
        "valid_duration_count": int(valid_duration.count()),
        "mean_duration": float(valid_duration.mean()) if not valid_duration.empty else None,
        "median_duration": float(valid_duration.median()) if not valid_duration.empty else None,
        "reason": None if not filtered.empty and not valid_duration.empty else "No valid rows found for the requested month."
    }}
"""
        return code

    # ========================================================
    # CUSTOMER CONVERSION FACTOR ANALYSIS
    # ========================================================
    # This rule must run before the overall conversion-validation
    # rule so questions about conversion factors are not reduced
    # to a row-by-row target consistency check.

    conversion_factor_request = (
        "conversion" in step_lower
        and any(
            phrase in step_lower
            for phrase in (
                "factor",
                "associated with",
                "affecting",
                "influencing",
                "important patterns",
                "patterns in",
                "customer conversion",
                "conversion by",
                "drivers of",
                "predictors of",
            )
        )
    )

    if conversion_factor_request:
        code += '''
# Customer conversion-factor analysis.
# Use y (yes/no) as the authoritative target when it is available.
if "y" in df.columns:
    target_text = df["y"].astype("string").str.strip().str.lower()
    target = target_text.map({"yes": 1, "no": 0})
    target_column = "y"
elif "conversion" in df.columns:
    target = pd.to_numeric(df["conversion"], errors="coerce")
    target_column = "conversion"
else:
    target = None
    target_column = None

if target is None:
    result = {
        "status": "FAIL",
        "analysis_type": "conversion_factor_analysis",
        "reason": "No supported conversion target column ('y' or 'conversion') was found.",
        "available_columns": df.columns.tolist(),
    }
else:
    valid_target_mask = target.isin([0, 1])
    analysis_df = df.loc[valid_target_mask].copy()
    analysis_df["_converted_target"] = target.loc[valid_target_mask].astype(int)

    total_records = int(len(analysis_df))
    converted_count = int(analysis_df["_converted_target"].sum())
    overall_rate = (
        float(analysis_df["_converted_target"].mean() * 100)
        if total_records else None
    )

    # Categorical predictors are compared directly. The target,
    # derived target, and obvious identifiers are excluded.
    categorical_columns = analysis_df.select_dtypes(
        include=["object", "category", "bool", "string"]
    ).columns.tolist()

    excluded_names = {
        "y", "conversion", "_converted_target",
        "id", "customer_id", "customer id",
        "record_id", "record id",
    }

    factor_results = {}
    minimum_group_size = 10

    for column in categorical_columns:
        if column.strip().lower() in excluded_names:
            continue

        working = analysis_df[[column, "_converted_target"]].copy()
        working["_factor_group"] = (
            working[column].astype("string").fillna("Missing").str.strip()
        )
        working.loc[working["_factor_group"] == "", "_factor_group"] = "Missing"

        grouped = working.groupby(
            "_factor_group", dropna=False
        )["_converted_target"].agg(
            records="count",
            converted_count="sum",
            conversion_rate="mean",
        )

        grouped["conversion_rate_percent"] = (
            grouped["conversion_rate"] * 100
        ).round(2)
        grouped = grouped.drop(columns=["conversion_rate"]).reset_index()
        grouped = grouped.rename(columns={"_factor_group": "group"})
        grouped["records"] = grouped["records"].astype(int)
        grouped["converted_count"] = grouped["converted_count"].astype(int)
        grouped["non_converted_count"] = (
            grouped["records"] - grouped["converted_count"]
        ).astype(int)
        grouped["eligible_for_comparison"] = (
            grouped["records"] >= minimum_group_size
        )
        grouped = grouped.sort_values(
            ["conversion_rate_percent", "records"],
            ascending=[False, False]
        )

        factor_results[column] = grouped.to_dict(orient="records")

    # Numeric predictors are binned into quantile groups where possible.
    # This produces readable group comparisons rather than treating
    # every numeric value as a separate category.
    numeric_candidates = [
        column for column in analysis_df.select_dtypes(include="number").columns
        if column not in {"_converted_target"}
        and column.strip().lower() not in excluded_names
    ]

    numeric_factor_results = {}
    for column in numeric_candidates:
        values = pd.to_numeric(analysis_df[column], errors="coerce")
        non_missing = values.dropna()

        if non_missing.nunique() < 2 or len(non_missing) < minimum_group_size:
            continue

        try:
            bins = pd.qcut(
                values,
                q=4,
                duplicates="drop"
            )
            if bins.nunique(dropna=True) < 2:
                continue

            working = pd.DataFrame({
                "_factor_group": bins.astype("string").fillna("Missing"),
                "_converted_target": analysis_df["_converted_target"],
            })
            grouped = working.groupby(
                "_factor_group", dropna=False
            )["_converted_target"].agg(
                records="count",
                converted_count="sum",
                conversion_rate="mean",
            )
            grouped["conversion_rate_percent"] = (
                grouped["conversion_rate"] * 100
            ).round(2)
            grouped = grouped.drop(columns=["conversion_rate"]).reset_index()
            grouped = grouped.rename(columns={"_factor_group": "group"})
            grouped["records"] = grouped["records"].astype(int)
            grouped["converted_count"] = grouped["converted_count"].astype(int)
            grouped["non_converted_count"] = (
                grouped["records"] - grouped["converted_count"]
            ).astype(int)
            grouped["eligible_for_comparison"] = (
                grouped["records"] >= minimum_group_size
            )
            numeric_factor_results[column] = grouped.to_dict(orient="records")
        except (ValueError, TypeError):
            # A numeric column that cannot be sensibly binned is omitted.
            continue

    usable_factor_count = len(factor_results) + len(numeric_factor_results)

    result = {
        "status": (
            "PASS" if total_records > 0 and usable_factor_count > 0 else "FAIL"
        ),
        "analysis_type": "conversion_factor_analysis",
        "target_column": target_column,
        "target_mapping": (
            {"yes": 1, "no": 0}
            if target_column == "y"
            else {"numeric_values": "0 = no conversion; 1 = conversion"}
        ),
        "total_dataset_records": int(len(df)),
        "valid_target_records": total_records,
        "excluded_invalid_target_records": int(len(df) - total_records),
        "converted_count": converted_count,
        "non_converted_count": int(total_records - converted_count),
        "overall_conversion_rate_percent": (
            round(overall_rate, 2) if overall_rate is not None else None
        ),
        "overall_conversion_rate_formula": (
            f"{converted_count} / {total_records} * 100"
        ),
        "categorical_factor_results": factor_results,
        "numeric_factor_results": numeric_factor_results,
        "minimum_group_size_for_comparison": minimum_group_size,
        "interpretation_notes": [
            "Group conversion rate = converted records in group / valid records in group * 100.",
            "Groups below the minimum group size are shown but marked ineligible for comparison.",
            "These are observational associations and do not establish causation.",
            "Call duration is measured after contact and may not be available for pre-contact targeting.",
        ],
    }
'''
        return code

    # Question-specific bank conversion validation.
    conversion_step = (
        "conversion" in step_lower
        and any(token in step_lower for token in [
            "rate", "percentage", "converted", "conversion = 1",
            "conversion=1", "row-by-row", "map y"
        ])
    )
    if conversion_step:
        code += """
required_columns = ["conversion", "y"]
missing_columns = [column for column in required_columns if column not in df.columns]
if missing_columns:
    result = {
        "status": "FAIL",
        "reason": "Required columns are missing: " + ", ".join(missing_columns),
        "available_columns": df.columns.tolist(),
    }
else:
    conversion_numeric = pd.to_numeric(df["conversion"], errors="coerce")
    y_normalized = df["y"].astype("string").str.strip().str.lower()
    y_mapped = y_normalized.map({"yes": 1, "no": 0})
    total_records = int(len(df))
    converted_count = int((conversion_numeric == 1).sum())
    non_converted_count = int((conversion_numeric == 0).sum())
    conversion_valid = int(conversion_numeric.isin([0, 1]).sum())
    y_valid = int(y_mapped.isin([0, 1]).sum())
    compared = conversion_numeric.notna() & y_mapped.notna()
    matches = int((conversion_numeric[compared] == y_mapped[compared]).sum())
    mismatches = int(compared.sum() - matches)
    conversion_rate = (converted_count / total_records * 100) if total_records else 0.0
    non_conversion_rate = (non_converted_count / total_records * 100) if total_records else 0.0
    result = {
        "status": "PASS" if (
            conversion_valid == total_records and y_valid == total_records and mismatches == 0
        ) else "FAIL",
        "total_records": total_records,
        "converted_count_conversion_1": converted_count,
        "non_converted_count_conversion_0": non_converted_count,
        "conversion_rate_percent": round(conversion_rate, 4),
        "non_conversion_rate_percent": round(non_conversion_rate, 4),
        "valid_conversion_values": conversion_valid,
        "valid_y_values_after_mapping": y_valid,
        "compared_records": int(compared.sum()),
        "row_by_row_matches": matches,
        "row_by_row_mismatches": mismatches,
        "unmatched_conversion_values": int(total_records - conversion_valid),
        "unmapped_y_values": int(total_records - y_valid),
        "conversion_rate_formula": f"{converted_count} / {total_records} * 100",
        "non_conversion_rate_formula": f"{non_converted_count} / {total_records} * 100",
    }
"""
        return code


    # ========================================================
    # 1. DETECT ANALYSIS CONTEXT
    # ========================================================

    explicit_relationship_target = (
        "units sold" in step_lower
        and "total profit" in step_lower
    )

    relationship_context = (
        explicit_relationship_target
        or (
            "relationship" in step_lower
            and any(
                word in step_lower
                for word in [
                    "outlier",
                    "strength",
                    "direction",
                    "interpret",
                    "influence",
                ]
            )
        )
    )

    trend_context = (
        "revenue" in step_lower
        and any(
            word in step_lower
            for word in [
                "over time",
                "trend",
                "month",
                "change",
                "increase",
                "decrease",
            ]
        )
    )

    # ========================================================
    # 2. CATEGORICAL FREQUENCY ANALYSIS
    # ========================================================
    #
    # IMPORTANT FIX:
    # The generated code detects categorical columns directly
    # from df. It does not reference data_profile because
    # data_profile is not automatically available inside the
    # generated code execution namespace.
    # ========================================================

    categorical_frequency_request = (
        "categor" in step_lower
        and any(
            word in step_lower
            for word in [
                "frequent",
                "frequency",
                "most common",
                "value counts",
                "distribution",
                "frequency table",
            ]
        )
    )

    if categorical_frequency_request:

        code += '''
# Detect categorical columns from the loaded dataset.
categorical_columns = df.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

# Exclude identifiers because their unique values generally
# do not provide useful categorical frequency insights.
identifier_tokens = [
    "order id",
    "customer id",
    "transaction id",
    "record id",
    "invoice id",
]

categorical_columns = [
    column
    for column in categorical_columns
    if not any(
        token in column.lower()
        for token in identifier_tokens
    )
]

frequency_results = {}

for column in categorical_columns:

    counts = df[column].value_counts(
        dropna=False
    ).head(10)

    frequency_results[column] = pd.DataFrame({
        "value": counts.index.astype(str),
        "count": counts.values,
        "percentage": (
            counts.values / max(len(df), 1) * 100
        ).round(2),
    }).reset_index(drop=True)

if frequency_results:

    result = frequency_results

else:

    result = {
        "message": "No categorical columns were found.",
        "available_columns": df.columns.tolist(),
    }
'''

        return code

    # ========================================================
    # 3. EXPLICIT IQR OUTLIER STEP
    # ========================================================

    if (
        "identify potential outliers" in step_lower
        and "iqr" in step_lower
        and "total profit" in step_lower
    ):

        code += '''
profit = pd.to_numeric(
    df["Total Profit"],
    errors="coerce"
)

Q1 = profit.quantile(0.25)
Q3 = profit.quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outlier_mask = (
    (profit < lower_bound)
    | (profit > upper_bound)
)

outliers = df.loc[outlier_mask].copy()

result = outliers[
    [
        "Region",
        "Country",
        "Order ID",
        "Total Profit",
    ]
]
'''

        return code

    # ========================================================
    # 4. DESCRIPTIVE STATISTICS FOR TOTAL PROFIT
    # ========================================================

    if (
        "calculate descriptive statistics for" in step_lower
        and "total profit" in step_lower
    ):

        code += '''
profit = pd.to_numeric(
    df["Total Profit"],
    errors="coerce"
).dropna()

result = profit.describe()
'''

        return code

    # ========================================================
    # 5. QUARTILES AND IQR
    # ========================================================

    if (
        "total profit" in step_lower
        and (
            "calculate the first quartile" in step_lower
            or "calculate the interquartile range" in step_lower
            or "calculate q1" in step_lower
            or "calculate q3" in step_lower
        )
    ):

        code += '''
profit = pd.to_numeric(
    df["Total Profit"],
    errors="coerce"
).dropna()

Q1 = profit.quantile(0.25)
Q3 = profit.quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

result = {
    "Q1": float(Q1),
    "Q3": float(Q3),
    "IQR": float(IQR),
    "lower_bound": float(lower_bound),
    "upper_bound": float(upper_bound),
}
'''

        return code

    # ========================================================
    # 6. RELATIONSHIP ANALYSIS
    # Units Sold vs Total Profit
    # ========================================================

    if explicit_relationship_target:

        # ----------------------------------------------------
        # Pearson correlation
        # ----------------------------------------------------

        if "pearson" in step_lower:

            code += '''
x = pd.to_numeric(
    df["Units Sold"],
    errors="coerce"
)

y = pd.to_numeric(
    df["Total Profit"],
    errors="coerce"
)

valid = pd.DataFrame({
    "Units Sold": x,
    "Total Profit": y,
}).dropna()

correlation = valid["Units Sold"].corr(
    valid["Total Profit"],
    method="pearson"
)

result = {
    "method": "Pearson",
    "correlation": float(correlation),
    "observations": int(len(valid)),
}
'''

            return code

        # ----------------------------------------------------
        # Spearman correlation
        # ----------------------------------------------------

        if "spearman" in step_lower:

            code += '''
x = pd.to_numeric(
    df["Units Sold"],
    errors="coerce"
)

y = pd.to_numeric(
    df["Total Profit"],
    errors="coerce"
)

valid = pd.DataFrame({
    "Units Sold": x,
    "Total Profit": y,
}).dropna()

correlation = valid["Units Sold"].corr(
    valid["Total Profit"],
    method="spearman"
)

result = {
    "method": "Spearman",
    "correlation": float(correlation),
    "observations": int(len(valid)),
}
'''

            return code

        # ----------------------------------------------------
        # Relationship interpretation
        # ----------------------------------------------------

        if any(
            word in step_lower
            for word in [
                "interpret",
                "strength",
                "direction",
            ]
        ):

            code += '''
x = pd.to_numeric(
    df["Units Sold"],
    errors="coerce"
)

y = pd.to_numeric(
    df["Total Profit"],
    errors="coerce"
)

valid = pd.DataFrame({
    "Units Sold": x,
    "Total Profit": y,
}).dropna()

pearson = valid["Units Sold"].corr(
    valid["Total Profit"],
    method="pearson"
)

spearman = valid["Units Sold"].corr(
    valid["Total Profit"],
    method="spearman"
)

if pd.isna(pearson):

    strength = "undefined"
    direction = "undefined"

else:

    absolute_correlation = abs(pearson)

    if absolute_correlation < 0.30:
        strength = "weak"
    elif absolute_correlation < 0.70:
        strength = "moderate"
    else:
        strength = "strong"

    if pearson > 0:
        direction = "positive"
    elif pearson < 0:
        direction = "negative"
    else:
        direction = "no linear relationship"

result = {
    "pearson": float(pearson),
    "spearman": float(spearman),
    "strength": strength,
    "direction": direction,
    "observations": int(len(valid)),
}
'''

            return code

        # ----------------------------------------------------
        # Relationship outliers
        # ----------------------------------------------------

        if (
            "outlier" in step_lower
            or "unusual" in step_lower
        ):

            code += '''
result = {}

for column in ["Units Sold", "Total Profit"]:

    values = pd.to_numeric(
        df[column],
        errors="coerce"
    ).dropna()

    Q1 = values.quantile(0.25)
    Q3 = values.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = values[
        (values < lower_bound)
        | (values > upper_bound)
    ]

    result[column] = {
        "Q1": float(Q1),
        "Q3": float(Q3),
        "IQR": float(IQR),
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound),
        "outlier_count": int(len(outliers)),
        "outlier_values": [
            float(value)
            for value in outliers.tolist()
        ],
    }
'''

            return code

        # ----------------------------------------------------
        # Relationship distribution
        # ----------------------------------------------------

        if any(
            word in step_lower
            for word in [
                "distribution",
                "distributions",
                "inspect",
            ]
        ):

            code += '''
result = df[
    ["Units Sold", "Total Profit"]
].describe()
'''

            return code

        # ----------------------------------------------------
        # General relationship calculation
        # ----------------------------------------------------

        code += '''
valid = df[
    ["Units Sold", "Total Profit"]
].apply(
    pd.to_numeric,
    errors="coerce"
).dropna()

result = {
    "pearson": float(
        valid["Units Sold"].corr(
            valid["Total Profit"],
            method="pearson"
        )
    ),
    "spearman": float(
        valid["Units Sold"].corr(
            valid["Total Profit"],
            method="spearman"
        )
    ),
    "observations": int(len(valid)),
}
'''

        return code

    # ========================================================
    # 7. REVENUE TREND ANALYSIS
    # ========================================================

    if trend_context:

        code += '''
df["Order Date"] = pd.to_datetime(
    df["Order Date"],
    errors="coerce"
)

valid = df.dropna(
    subset=["Order Date"]
).copy()

monthly_revenue = (
    valid.groupby(
        valid["Order Date"].dt.to_period("M")
    )["Total Revenue"]
    .sum()
    .sort_index()
)
'''

        if (
            "largest increase" in step_lower
            or "largest decrease" in step_lower
            or "increase and decrease" in step_lower
        ):

            code += '''
monthly_change = monthly_revenue.diff().dropna()

if not monthly_change.empty:

    increase_period = monthly_change.idxmax()
    decrease_period = monthly_change.idxmin()

    result = {
        "monthly_revenue": {
            str(period): float(value)
            for period, value
            in monthly_revenue.items()
        },
        "largest_increase_period": str(increase_period),
        "largest_increase": float(
            monthly_change.loc[increase_period]
        ),
        "largest_decrease_period": str(decrease_period),
        "largest_decrease": float(
            monthly_change.loc[decrease_period]
        ),
    }

else:

    result = {
        "monthly_revenue": {
            str(period): float(value)
            for period, value
            in monthly_revenue.items()
        },
        "message": "Not enough monthly observations to calculate changes.",
    }
'''

        elif (
            "monthly change" in step_lower
            or "month-to-month" in step_lower
        ):

            code += '''
result = monthly_revenue.diff()
'''

        else:

            code += '''
result = monthly_revenue
'''

        return code

    # ========================================================
    # 8. REGION PROFIT COMPARISON
    # ========================================================

    if (
        "region" in step_lower
        and "total profit" in step_lower
        and (
            "group" in step_lower
            or "rank" in step_lower
            or "highest" in step_lower
            or "compare" in step_lower
        )
    ):

        code += '''
region_profit = (
    df.groupby("Region")["Total Profit"]
    .sum()
    .sort_values(ascending=False)
)

if not region_profit.empty:

    highest_region = region_profit.index[0]
    highest_profit = region_profit.iloc[0]

    result = {
        "region_profit": region_profit,
        "highest_profit_region": str(highest_region),
        "highest_total_profit": float(highest_profit),
    }

else:

    result = {
        "message": "No region-level profit data is available."
    }
'''

        return code

    # ========================================================
    # 9. MISSING VALUES
    # ========================================================

    if (
        (
            "missing" in step_lower
            or "null" in step_lower
            or "incomplete" in step_lower
        )
        and "mean" not in step_lower
        and "average" not in step_lower
    ):

        code += '''
missing_count = df.isnull().sum()

missing_percentage = (
    df.isnull().mean() * 100
).round(2)

result = pd.DataFrame({
    "missing_count": missing_count,
    "missing_percentage": missing_percentage,
})
'''

        return code

    # ========================================================
    # 10. NUMERICAL DESCRIPTIVE STATISTICS
    # ========================================================

    if (
        "descriptive statistics" in step_lower
        or "descriptive analysis" in step_lower
        or (
            "numerical statistics" in step_lower
            and "distribution" not in step_lower
        )
    ):

        code += '''
numeric_df = df.select_dtypes(
    include="number"
)

result = numeric_df.describe().T

result["median"] = numeric_df.median()

result["skewness"] = numeric_df.skew()

result["missing_count"] = numeric_df.isnull().sum()

result["missing_percentage"] = (
    numeric_df.isnull().mean() * 100
).round(2)
'''

        return code

    # ========================================================
    # 11. DISTRIBUTION AND SKEWNESS
    # ========================================================

    if (
        "skewness" in step_lower
        or "distribution" in step_lower
        or "distributions" in step_lower
    ):

        code += '''
numeric_df = df.select_dtypes(
    include="number"
)

result = pd.DataFrame({
    "count": numeric_df.count(),
    "mean": numeric_df.mean(),
    "median": numeric_df.median(),
    "std": numeric_df.std(),
    "min": numeric_df.min(),
    "max": numeric_df.max(),
    "skewness": numeric_df.skew(),
    "missing_count": numeric_df.isnull().sum(),
})
'''

        return code



    # ========================================================
    # 12. GENERAL MEAN / AVERAGE
    # ========================================================

    if any(
        word in step_lower.split()
        for word in ("mean", "average", "avg")
    ):

        import re

        # Read column names from the actual dataset.
        try:
            available_columns = pd.read_csv(
                filepath,
                nrows=0
            ).columns.tolist()
        except Exception:
            available_columns = []

        # First support explicitly quoted column names.
        match = re.search(
            r"""(?:in|of)\s+['"]([^'"]+)['"]""",
            str(step_text),
            re.IGNORECASE
        )

        requested_column = (
            match.group(1).strip()
            if match
            else None
        )

        actual_column = None

        if requested_column:
            for column in available_columns:
                if column.strip().lower() == requested_column.lower():
                    actual_column = column
                    break

        # Otherwise, match an actual column name mentioned
        # in the prompt. Match longer column names first.
        if actual_column is None and not requested_column:

            def words(value):
                return re.findall(
                    r"[a-z0-9]+",
                    str(value).lower()
                )

            prompt_words = words(step_text)

            for column in sorted(
                available_columns,
                key=lambda item: len(words(item)),
                reverse=True
            ):
                column_words = words(column)
                width = len(column_words)

                if not width:
                    continue

                for index in range(
                    len(prompt_words) - width + 1
                ):
                    if prompt_words[index:index + width] == column_words:
                        actual_column = column
                        break

                if actual_column is not None:
                    break

        if actual_column is None:
            code += '''
result = {
    "status": "UNSUPPORTED",
    "reason": "Could not identify a matching dataset column in the request.",
    "available_columns": df.columns.tolist(),
}
'''
            return code

        # Generate safe numeric mean calculation.
        code += f'''
requested_column = {actual_column!r}

values = pd.to_numeric(
    df[requested_column],
    errors="coerce"
)

valid_values = values.dropna()

result = {{
    "status": "PASS" if not valid_values.empty else "FAIL",
    "analysis_type": "mean_analysis",
    "metric_column": requested_column,
    "mean": (
        float(valid_values.mean())
        if not valid_values.empty
        else None
    ),
    "valid_value_count": int(valid_values.count()),
    "dataset_row_count": int(len(df)),
}}
'''

        return code

# ============================================================
# LOCAL TEST
# ============================================================

if __name__ == "__main__":

    project_root = os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )

    filepath = os.path.join(
        project_root,
        "data",
        "raw",
        "sales.csv"
    )

    test_steps = [
        "Calculate descriptive statistics for numerical columns.",
        "Check the dataset for missing values.",
        "Identify most frequent values in categorical columns.",
        "Analyze distribution and skewness of numerical variables.",
        "Summarize overall structure and characteristics of dataset.",
        "Group the data by Region and calculate the total of Total Profit.",
        "Calculate the Pearson correlation between Units Sold and Total Profit.",
        "Identify potential outliers using the IQR method for Total Profit.",
    ]

    print("=" * 70)
    print("CODE GENERATOR TEST")
    print("=" * 70)

    for index, step in enumerate(test_steps, start=1):

        print(f"\nTEST STEP {index}:")
        print(step)

        generated_code = generate_code_mock(
            step_text=step,
            filepath=filepath,
        )

        print("\nGenerated code preview:")
        print(generated_code[:700])

        # Confirm the generated code is syntactically valid.
        compile(
            generated_code,
            filename=f"<generated_step_{index}>",
            mode="exec",
        )

        print("Syntax check: PASS")

    print("\nAll generated-code syntax checks passed.")
