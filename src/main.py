# =========================================================
# MAIN PIPELINE
# SELF-VALIDATING EDA AGENT
# =========================================================

import os
import sys
import json
import json
import math
import re
import os
import inspect
import pandas as pd
import numpy as np

from answer_support import (
    SPECIAL_TYPES,
    extract_special_result,
    special_insight,
    special_validate,
    special_evidence,
    special_mapping,
    special_recommendation,
    set_answer_focus,
    build_iqr_evidence_mapping,
    build_isolation_forest_mapping,
)


# =========================================================
# PROJECT PATH
# =========================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

SRC_DIR = os.path.join(
    PROJECT_ROOT,
    "src"
)

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "customer.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

if SRC_DIR not in sys.path:
    sys.path.insert(
        0,
        SRC_DIR
    )


# =========================================================
# IMPORT PROJECT MODULES
# =========================================================

from data_profiler import profile_dataset

from task_analyzer import analyze_task_mock

from eda_planner import generate_plan_mock

from multi_step_executor import (
    execute_plan_steps,
    create_self_correcting_executor
)

from result_validator import validate_analytical_results

from claim_evidence_validator import (
    validate_insight_and_recommendation,
)

from result_aggregator import (
    aggregate_results,
    print_aggregation_summary
)

from insight_generator import (
    generate_insight
)

from insight_validator import (
    validate_insight
)

from evidence_mapping_grouped_mean_fixed import(
    create_grouped_evidence,
    create_correlation_evidence,
    create_trend_evidence,
    create_evidence_summary,
    map_insight_to_evidence,
)

from recommendation_generator import (
    generate_recommendation
)

from recommendation_validator import validate_recommendation

from final_report import (
    create_final_report,
    save_report
)


# =========================================================
# DISPLAY HELPER
# =========================================================

def print_section(title):

    print("\n")
    print("=" * 80)
    print(title)
    print("=" * 80)


# =========================================================
# JSON-SAFE CONVERSION
# IMPORTANT FIX FOR PANDAS PERIOD
# =========================================================

def safe_json_value(value):

    # None
    if value is None:
        return None

    # Pandas Period
    if isinstance(value, pd.Period):
        return str(value)

    # Pandas Timestamp
    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    # NumPy integer
    if isinstance(value, np.integer):
        return int(value)

    # NumPy floating point
    if isinstance(value, np.floating):
        return float(value)

    # NumPy boolean
    if isinstance(value, np.bool_):
        return bool(value)

    # DataFrame
    if isinstance(value, pd.DataFrame):
        # Keep the original column name when serializing describe().T-style
        # statistical summaries. Without this, each row loses its identity.
        statistic_keys = {"count", "mean", "std", "min", "max", "median", "skewness", "25%", "50%", "75%", "missing_count", "missing_percentage"}
        is_stat_summary = bool(statistic_keys.intersection(map(str, value.columns)))
        if is_stat_summary and not isinstance(value.index, pd.RangeIndex):
            rows = []
            for label, record in value.iterrows():
                row = {"column_name": str(label)}
                row.update(safe_json_value(record.to_dict()))
                rows.append(row)
            return rows

        return [
            safe_json_value(record)
            for record in value.to_dict(orient="records")
        ]

    # Series
    if isinstance(value, pd.Series):

        return {
            str(safe_json_value(key)):
            safe_json_value(item)
            for key, item in value.items()
        }

    # Dictionary
    if isinstance(value, dict):

        return {
            str(safe_json_value(key)):
            safe_json_value(item)
            for key, item in value.items()
        }

    # List
    if isinstance(value, list):

        return [
            safe_json_value(item)
            for item in value
        ]

    # Tuple
    if isinstance(value, tuple):

        return [
            safe_json_value(item)
            for item in value
        ]

    # Set
    if isinstance(value, set):

        return [
            safe_json_value(item)
            for item in value
        ]

    # Python primitive
    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool
        )
    ):
        return value

    # Final fallback
    return str(value)


# =========================================================
# JSON PRINT HELPER
# =========================================================

def print_json(value, max_chars=12000):
    """Print useful JSON diagnostics without flooding the console with row-level data."""
    rendered = json.dumps(
        safe_json_value(value),
        indent=2,
        ensure_ascii=False,
        default=str,
    )
    if len(rendered) > max_chars:
        rendered = (
            rendered[:max_chars]
            + f"\\n... output shortened ({len(rendered):,} characters total). "
              "Full results remain available in the saved report/results."
        )
    print(rendered)


# =========================================================
# COLUMN NORMALIZATION
# =========================================================

def normalize_column_name(column):

    if column is None:
        return column

    column = str(column).strip()

    aliases = {

        "UnitsSold": "Units Sold",
        "Units_Sold": "Units Sold",

        "TotalProfit": "Total Profit",
        "Total_Profit": "Total Profit",

        "TotalRevenue": "Total Revenue",
        "Total_Revenue": "Total Revenue",

        "TotalCost": "Total Cost",
        "Total_Cost": "Total Cost",

        "UnitPrice": "Unit Price",
        "Unit_Price": "Unit Price",

        "UnitCost": "Unit Cost",
        "Unit_Cost": "Unit Cost",

        "OrderDate": "Order Date",
        "Order_Date": "Order Date",

        "ShipDate": "Ship Date",
        "Ship_Date": "Ship Date",

        "OrderID": "Order ID",
        "Order_ID": "Order ID",

        "ItemType": "Item Type",
        "Item_Type": "Item Type",

        "SalesChannel": "Sales Channel",
        "Sales_Channel": "Sales Channel",

        "OrderPriority": "Order Priority",
        "Order_Priority": "Order Priority"
    }

    return aliases.get(
        column,
        column
    )


# =========================================================
# FINAL DATA QUALITY HELPERS
# =========================================================

def is_identifier_column(column_name):
    """Return True for numeric identifiers that should not be summarized as measures."""
    name = str(column_name).strip().lower()
    normalized = name.replace("_", " ").replace("-", " ")
    tokens = normalized.split()
    return (
        name in {"id", "orderid", "order_id", "customerid", "customer_id"}
        or any(token in {"id", "identifier", "key", "code"} for token in tokens)
        or normalized.endswith(" number")
        or normalized.endswith(" no")
    )


def _numeric_stat_rows(result, numeric_columns):
    """Remove identifier rows from numerical summaries, preserving unrelated results."""
    stat_keys = {"mean", "median", "skewness", "std", "25%", "75%"}

    # pandas describe().T commonly returns a DataFrame indexed by column name.
    if isinstance(result, pd.DataFrame):
        if not stat_keys.intersection(set(map(str, result.columns))):
            return result
        identifier_labels = [
            label for label in result.index
            if is_identifier_column(label)
        ]
        if identifier_labels:
            return result.drop(index=identifier_labels, errors="ignore")
        return result

    # Some executors convert the same summary to one dictionary per numeric column.
    if isinstance(result, list) and result:
        if not all(isinstance(item, dict) for item in result):
            return result
        if not any(stat_keys.intersection(item.keys()) for item in result):
            return result

        business_columns = [
            column for column in numeric_columns
            if not is_identifier_column(column)
        ]
        if len(result) == len(numeric_columns):
            column_result_pairs = zip(numeric_columns, result)
        elif len(result) == len(business_columns):
            # Also handle results that were already filtered once by an upstream step.
            column_result_pairs = zip(business_columns, result)
        else:
            return result

        labeled_rows = []
        for column, item in column_result_pairs:
            if is_identifier_column(column):
                continue
            labeled_item = dict(item)
            labeled_item.setdefault("column_name", str(column))
            labeled_rows.append(labeled_item)
        return labeled_rows

    return result


def sanitize_identifier_statistics(primary_result, execution_results, dataset_path):
    """Keep numeric IDs available as identifiers, but exclude them from statistical summaries."""
    try:
        frame = pd.read_csv(dataset_path)
        numeric_columns = list(frame.select_dtypes(include="number").columns)
    except Exception:
        return primary_result, execution_results

    cleaned_primary = _numeric_stat_rows(primary_result, numeric_columns)

    if isinstance(execution_results, dict):
        steps = execution_results.get("steps", [])
        if isinstance(steps, list):
            for step in steps:
                if not isinstance(step, dict):
                    continue
                for key in ("final_result", "result"):
                    if key in step:
                        step[key] = _numeric_stat_rows(step[key], numeric_columns)

    return cleaned_primary, execution_results


def build_dataset_facts(dataset_path):
    """Build independent, deterministic facts used by evidence-to-insight validation."""
    frame = pd.read_csv(dataset_path)
    identifier_columns = [
        column for column in frame.columns if is_identifier_column(column)
    ]
    numeric_columns = [
        column for column in frame.select_dtypes(include="number").columns
        if column not in identifier_columns
    ]
    date_columns = []
    parsed_dates = {}
    for column in frame.columns:
        if "date" in str(column).lower():
            parsed = pd.to_datetime(frame[column], errors="coerce")
            if parsed.notna().any():
                date_columns.append(column)
                parsed_dates[column] = parsed

    categorical_columns = [
        column for column in frame.columns
        if column not in numeric_columns
        and column not in identifier_columns
        and column not in date_columns
    ]

    numeric_summary = {}
    for column in numeric_columns:
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if values.empty:
            continue
        numeric_summary[column] = {
            "count": int(values.count()),
            "mean": round(float(values.mean()), 2),
            "median": round(float(values.median()), 2),
            "min": round(float(values.min()), 2),
            "max": round(float(values.max()), 2),
            "skewness": round(float(values.skew()), 2) if values.count() > 2 else 0.0,
        }

    categorical_summary = {}
    for column in categorical_columns:
        counts = frame[column].astype(str).value_counts(dropna=False).head(3)
        categorical_summary[column] = [
            {"value": str(value), "count": int(count)}
            for value, count in counts.items()
        ]

    date_ranges = {}
    for column, values in parsed_dates.items():
        valid = values.dropna()
        date_ranges[column] = {
            "min": valid.min().strftime("%Y-%m-%d"),
            "max": valid.max().strftime("%Y-%m-%d"),
            "count": int(valid.count()),
        }

    grouped_profit = {}
    if "Region" in frame.columns and "Total Profit" in frame.columns:
        grouped = frame.groupby("Region", dropna=False)["Total Profit"].sum().sort_values(ascending=False)
        grouped_profit = {str(key): round(float(value), 2) for key, value in grouped.items()}

    return {
        "row_count": int(frame.shape[0]),
        "column_count": int(frame.shape[1]),
        "columns": [str(column) for column in frame.columns],
        "missing_cells": int(frame.isna().sum().sum()),
        "duplicate_rows": int(frame.duplicated().sum()),
        "identifier_columns": [str(column) for column in identifier_columns],
        "numeric_summary": numeric_summary,
        "categorical_summary": categorical_summary,
        "date_ranges": date_ranges,
        "profit_by_region": grouped_profit,
    }


# =========================================================
# TARGET COLUMN NORMALIZATION
# =========================================================

def normalize_target_columns(target_columns):

    if not target_columns:
        return []

    normalized = []

    for column in target_columns:

        actual_column = normalize_column_name(
            column
        )

        if actual_column not in normalized:

            normalized.append(
                actual_column
            )

    return normalized


# =========================================================
# DATASET COLUMN EXTRACTION
# =========================================================

def extract_dataset_columns(data_profile):

    if not isinstance(
        data_profile,
        dict
    ):
        return []

    columns = data_profile.get(
        "columns",
        []
    )

    if isinstance(
        columns,
        list
    ):
        return columns

    if isinstance(
        columns,
        dict
    ):
        return list(
            columns.keys()
        )

    return []


# =========================================================
# RESULT DISPLAY
# =========================================================

def convert_result_for_display(result):

    return safe_json_value(result)


# =========================================================
# GET STEP RESULT
# =========================================================

def get_step_result(
    execution_results,
    step_number
):

    if not isinstance(
        execution_results,
        dict
    ):
        return None

    steps = execution_results.get(
        "steps",
        []
    )

    for step in steps:

        if step.get(
            "step_number"
        ) == step_number:

            return step.get(
                "final_result"
            )

    return None


# =========================================================
# FIND RESULT CONTAINER
# =========================================================

def unwrap_result(result):

    if result is None:
        return None

    if isinstance(
        result,
        str
    ):

        try:

            decoded = json.loads(
                result
            )

            return unwrap_result(
                decoded
            )

        except Exception:

            return result

    if isinstance(
        result,
        dict
    ):

        for key in [
            "result",
            "data",
            "records",
            "value"
        ]:

            if key in result:

                nested = result.get(
                    key
                )

                if nested is not None:

                    return unwrap_result(
                        nested
                    )

    return result


# =========================================================
# BUILD GROUPED SERIES
# =========================================================

def build_grouped_series(
    execution_results,
    filepath
):

    # -----------------------------------------------------
    # SOURCE 1 - STEP 2 RESULT
    # -----------------------------------------------------

    step_2_result = get_step_result(
        execution_results,
        2
    )

    step_2_result = unwrap_result(
        step_2_result
    )

    try:

        if isinstance(
            step_2_result,
            list
        ):

            df = pd.DataFrame(
                step_2_result
            )

            df.columns = [
                normalize_column_name(
                    column
                )
                for column in df.columns
            ]

            if (
                "Region" in df.columns
                and
                "Total Profit" in df.columns
            ):

                df["Total Profit"] = pd.to_numeric(
                    df["Total Profit"],
                    errors="coerce"
                )

                df = df.dropna(
                    subset=[
                        "Region",
                        "Total Profit"
                    ]
                )

                if not df.empty:

                    series = pd.Series(
                        df["Total Profit"].astype(float).values,
                        index=df["Region"].astype(str).values,
                        name="Total Profit"
                    )

                    return series

    except Exception as error:

        print(
            "Step 2 grouped-result conversion failed:",
            error
        )

    # -----------------------------------------------------
    # SOURCE 2 - DATASET FALLBACK
    # -----------------------------------------------------

    print(
        "\nUsing validated dataset fallback to reconstruct "
        "Region -> Total Profit."
    )

    try:

        df = pd.read_csv(
            filepath
        )

        required_columns = [
            "Region",
            "Total Profit"
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing_columns:

            print(
                "Missing columns:",
                missing_columns
            )

            return None

        df["Total Profit"] = pd.to_numeric(
            df["Total Profit"],
            errors="coerce"
        )

        df = df.dropna(
            subset=[
                "Region",
                "Total Profit"
            ]
        )

        grouped = (
            df.groupby(
                "Region"
            )["Total Profit"]
            .sum()
            .sort_values(
                ascending=False
            )
        )

        grouped.name = "Total Profit"

        return grouped

    except Exception as error:

        print(
            "Dataset fallback failed:",
            error
        )

        return None


# =========================================================
# EVIDENCE TRACEABILITY
# =========================================================


def build_mean_evidence_mapping(insight, evidence):
    """Validate a mean insight against its computed metric, mean and count."""
    evidence = evidence if isinstance(evidence, dict) else {}

    result = evidence.get("result")
    if not isinstance(result, dict):
        supporting = evidence.get("supporting_result", {})
        result = (
            supporting.get("primary_result")
            if isinstance(supporting, dict)
            else None
        )
    if not isinstance(result, dict):
        result = {}

    metric = result.get("metric_column") or result.get("metric")
    expected_mean = result.get(
        "mean", result.get("mean_value", result.get("average"))
    )
    expected_count = result.get(
        "valid_value_count",
        result.get("valid_count", result.get("count"))
    )

    text = str(insight or "")
    normalized = text.lower()

    metric_pass = bool(metric) and str(metric).strip().lower() in normalized

    mean_pass = False
    if (
        isinstance(expected_mean, (int, float, np.integer, np.floating))
        and not isinstance(expected_mean, (bool, np.bool_))
        and np.isfinite(float(expected_mean))
    ):
        tokens = re.findall(
            r"(?<![A-Za-z0-9_])-?\d[\d,]*(?:\.\d+)?", text
        )
        for token in tokens:
            try:
                if abs(
                    float(token.replace(",", "")) - float(expected_mean)
                ) <= 0.005:
                    mean_pass = True
                    break
            except ValueError:
                continue

    count_pass = True
    count_is_valid = (
        isinstance(expected_count, (int, np.integer))
        and not isinstance(expected_count, (bool, np.bool_))
    )

    if count_is_valid:
        count_tokens = re.findall(
            r"(?<![A-Za-z0-9_])\d[\d,]*(?![A-Za-z0-9_])", text
        )
        count_values = set()
        for token in count_tokens:
            try:
                count_values.add(int(token.replace(",", "")))
            except ValueError:
                continue
        count_pass = int(expected_count) in count_values

    checks = [
        {
            "check": "metric_column_mentioned",
            "status": "PASS" if metric_pass else "FAIL",
        },
        {
            "check": "reported_mean_matches_validated_result",
            "status": "PASS" if mean_pass else "FAIL",
        },
    ]

    if count_is_valid:
        checks.append({
            "check": "reported_valid_count_matches_evidence",
            "status": "PASS" if count_pass else "FAIL",
        })

    passed = all(item["status"] == "PASS" for item in checks)

    return {
        "status": "PASS" if passed else "FAIL",
        "analysis_type": "mean_analysis",
        "metric_column": metric,
        "validated_mean": (
            float(expected_mean)
            if isinstance(
                expected_mean, (int, float, np.integer, np.floating)
            )
            else None
        ),
        "validated_count": int(expected_count) if count_is_valid else None,
        "claim_validation": {
            "status": "PASS" if passed else "FAIL",
            "total_claims": len(checks),
            "passed_claims": sum(
                item["status"] == "PASS" for item in checks
            ),
            "failed_claims": sum(
                item["status"] == "FAIL" for item in checks
            ),
            "checks": checks,
        },
        "reason": (
            None if passed
            else "Mean insight does not match the validated metric, "
                 "mean or valid observation count."
        ),
    }


def build_extrema_evidence_mapping(insight, evidence):
    """Validate the metric, operation, value and count against the validated extrema result."""
    evidence = evidence if isinstance(evidence, dict) else {}
    result = evidence.get("result")
    if not isinstance(result, dict):
        supporting = evidence.get("supporting_result", {})
        result = (
            supporting.get("primary_result")
            if isinstance(supporting, dict)
            else None
        )
    if not isinstance(result, dict):
        result = evidence.get("primary_result")
    if not isinstance(result, dict) or str(result.get("analysis_type", "")).lower() != "extrema_analysis":
        return {
            "status": "FAIL",
            "analysis_type": "extrema_analysis",
            "reason": "No validated extrema result was available for evidence mapping.",
            "checks": [],
        }

    operation = str(result.get("operation", "")).strip().lower()
    metric = str(result.get("metric_column") or result.get("metric") or "").strip()
    expected = result.get("value")
    count = result.get("valid_value_count", result.get("valid_count", result.get("count")))
    text = str(insight or "").lower()
    metric_pass = bool(metric) and metric.lower() in text
    operation_terms = (
        ("maximum", "max", "highest", "largest", "greatest")
        if operation == "maximum"
        else ("minimum", "min", "lowest", "smallest", "least")
    )
    operation_pass = operation in {"maximum", "minimum"} and any(
        re.search(r"\b" + re.escape(term) + r"\b", text)
        for term in operation_terms
    )
    numbers = []
    for token in re.findall(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?", str(insight or "")):
        try:
            numbers.append(float(token.replace(",", "")))
        except ValueError:
            pass
    value_pass = (
        isinstance(expected, (int, float, np.integer, np.floating))
        and not isinstance(expected, (bool, np.bool_))
        and np.isfinite(float(expected))
        and any(abs(number - float(expected)) <= 0.005 for number in numbers)
    )
    count_is_valid = (
        isinstance(count, (int, np.integer))
        and not isinstance(count, (bool, np.bool_))
    )
    count_pass = not count_is_valid or any(
        token.replace(",", "") == str(int(count))
        for token in re.findall(r"(?<![\w.])\d[\d,]*(?![\w.])", str(insight or ""))
    )
    checks = [
        {"check": "metric_column_mentioned", "status": "PASS" if metric_pass else "FAIL"},
        {"check": "extreme_operation_mentioned", "status": "PASS" if operation_pass else "FAIL"},
        {"check": "extreme_value_matches_validated_result", "status": "PASS" if value_pass else "FAIL"},
    ]
    if count_is_valid:
        checks.append({
            "check": "valid_observation_count_matches_result",
            "status": "PASS" if count_pass else "FAIL",
        })
    passed = all(item["status"] == "PASS" for item in checks)
    return {
        "status": "PASS" if passed else "FAIL",
        "analysis_type": "extrema_analysis",
        "operation": operation,
        "metric_column": metric,
        "validated_value": float(expected) if isinstance(expected, (int, float, np.integer, np.floating)) else None,
        "validated_count": int(count) if count_is_valid else None,
        "claim_validation": {
            "status": "PASS" if passed else "FAIL",
            "total_claims": len(checks),
            "passed_claims": sum(item["status"] == "PASS" for item in checks),
            "failed_claims": sum(item["status"] == "FAIL" for item in checks),
            "checks": checks,
        },
        "reason": None if passed else "Extrema insight does not match the validated metric, operation, value or count.",
    }



def extract_specification_grouped_mean(primary_result):
    """Normalize validated specification-driven means with one or more group columns."""
    if not isinstance(primary_result, dict):
        return None
    if str(primary_result.get("analysis_type", "")).lower() != "specification_driven":
        return None
    if str(primary_result.get("status", "")).upper() != "PASS":
        return None

    operations = primary_result.get("operations", [])
    metrics = primary_result.get("metrics", [])
    group_columns = primary_result.get("group_by", [])
    operation_results = primary_result.get("operation_results", {})
    rows = operation_results.get("aggregations") if isinstance(operation_results, dict) else None

    if (
        not isinstance(operations, (list, tuple))
        or "mean" not in [str(item).strip().lower() for item in operations]
        or not isinstance(metrics, (list, tuple))
        or len(metrics) != 1
        or not isinstance(group_columns, (list, tuple))
        or not group_columns
        or not isinstance(rows, list)
        or not rows
    ):
        return None

    metric = str(metrics[0]).strip()
    normalized_group_columns = [str(item).strip() for item in group_columns]
    if not metric or any(not item for item in normalized_group_columns):
        return None

    mean_key = f"{metric}__mean"
    group_results = {}
    # Group sizes reported by the executor (separate from the validated means).
    sizes_by_key = {}
    size_rows = operation_results.get("group_sizes") if isinstance(operation_results, dict) else None
    if isinstance(size_rows, list):
        for size_row in size_rows:
            if isinstance(size_row, dict) and all(c in size_row for c in normalized_group_columns):
                sizes_by_key[tuple(str(size_row[c]) for c in normalized_group_columns)] = size_row
    for row in rows:
        if not isinstance(row, dict) or any(column not in row for column in normalized_group_columns):
            return None
        if mean_key not in row:
            return None

        group_values = {
            column: row.get(column)
            for column in normalized_group_columns
        }
        if any(value is None for value in group_values.values()):
            return None

        try:
            mean_number = float(row.get(mean_key))
        except (TypeError, ValueError, OverflowError):
            return None
        if not np.isfinite(mean_number):
            return None

        group_label = (
            str(group_values[normalized_group_columns[0]])
            if len(normalized_group_columns) == 1
            else ", ".join(
                f"{column}={group_values[column]}"
                for column in normalized_group_columns
            )
        )
        if group_label in group_results:
            return None

        group_results[group_label] = {
            "mean_value": mean_number,
            "group_values": safe_json_value(group_values),
        }
        size_row = sizes_by_key.get(tuple(str(group_values[c]) for c in normalized_group_columns))
        if isinstance(size_row, dict):
            try:
                group_results[group_label]["count"] = int(size_row["row_count"])
                valid_key = f"{metric}__valid_count"
                if valid_key in size_row:
                    group_results[group_label]["valid_count"] = int(size_row[valid_key])
            except (KeyError, TypeError, ValueError):
                pass

    if not group_results:
        return None

    return {
        "analysis_type": "grouped_mean_analysis",
        "status": "PASS",
        "metric_column": metric,
        "group_column": " and ".join(normalized_group_columns),
        "group_columns": normalized_group_columns,
        "group_results": group_results,
        "source_analysis_type": "specification_driven",
        "source_result": safe_json_value(primary_result),
    }


def validate_specification_grouped_mean_insight(insight, primary_result):
    """Independently verify every composite group label and its validated mean."""
    normalized = extract_specification_grouped_mean(primary_result)
    text = str(insight or "")
    lower_text = " ".join(text.casefold().replace("_", " ").split())

    checks = {
        "result_status": normalized is not None,
        "metric_mentioned": False,
        "all_grouping_columns_mentioned": False,
        "all_composite_group_labels_mentioned": False,
        "all_group_means_mentioned": False,
    }

    if normalized is None:
        return {
            "overall": "FAIL",
            "pass": False,
            "checks": checks,
            "reason": "The specification-driven grouped-mean result is invalid or incomplete.",
        }

    metric = normalized["metric_column"]
    checks["metric_mentioned"] = (
        " ".join(metric.casefold().replace("_", " ").split()) in lower_text
    )
    checks["all_grouping_columns_mentioned"] = all(
        " ".join(column.casefold().replace("_", " ").split()) in lower_text
        for column in normalized["group_columns"]
    )

    group_label_checks = []
    mean_checks = []
    pair_checks = []
    for label, details in normalized["group_results"].items():
        normalized_label = " ".join(label.casefold().replace("_", " ").split())
        group_label_checks.append(normalized_label in lower_text)

        expected = float(details["mean_value"])
        candidates = {f"{expected:,.2f}", f"{expected:.2f}"}
        mean_checks.append(any(
            re.search(r"(?<!\d)" + re.escape(candidate) + r"(?!\d)", text)
            for candidate in candidates
        ))
        pair_checks.append(any(
            re.search(
                re.escape(str(label)) + r"\s*:\s*" + re.escape(candidate),
                text,
                flags=re.IGNORECASE,
            )
            for candidate in candidates
        ))

    checks["all_composite_group_labels_mentioned"] = (
        bool(group_label_checks) and all(group_label_checks)
    )
    checks["all_group_means_mentioned"] = (
        bool(mean_checks) and all(mean_checks)
    )
    checks["each_group_mean_paired_with_correct_label"] = (
        bool(pair_checks) and all(pair_checks)
    )
    passed = all(checks.values())

    return {
        "overall": "PASS" if passed else "FAIL",
        "pass": passed,
        "analysis_type": "specification_driven_grouped_mean",
        "checks": checks,
        "group_count": len(normalized["group_results"]),
        "metric_column": metric,
        "group_columns": normalized["group_columns"],
        "reason": None if passed else (
            "The insight must mention the requested metric, every grouping column, "
            "every composite group label, and every validated group mean."
        ),
    }


def build_grouped_mean_evidence_mapping(insight, evidence):
    """Validate every group label and mean, including composite group keys."""
    evidence = evidence if isinstance(evidence, dict) else {}
    result = evidence.get("result")
    if not isinstance(result, dict):
        result = evidence.get("supporting_result")
    if not isinstance(result, dict):
        result = evidence.get("primary_result")
    if not isinstance(result, dict):
        result = {}

    metric = str(result.get("metric_column") or result.get("metric") or "").strip()
    group_columns = result.get("group_columns")
    if not isinstance(group_columns, (list, tuple)) or not group_columns:
        group_columns = [result.get("group_column")] if result.get("group_column") else []
    group_columns = [str(item).strip() for item in group_columns if str(item).strip()]
    groups = result.get("group_results")
    text = str(insight or "")
    lower_text = " ".join(text.casefold().replace("_", " ").split())

    checks = []
    metric_pass = bool(metric) and (
        " ".join(metric.casefold().replace("_", " ").split()) in lower_text
    )
    checks.append({
        "check": "metric_column_mentioned",
        "status": "PASS" if metric_pass else "FAIL",
        "metric_column": metric,
    })

    grouping_pass = bool(group_columns) and all(
        " ".join(column.casefold().replace("_", " ").split()) in lower_text
        for column in group_columns
    )
    checks.append({
        "check": "all_grouping_columns_mentioned",
        "status": "PASS" if grouping_pass else "FAIL",
        "group_columns": group_columns,
    })

    if not isinstance(groups, dict) or not groups:
        checks.append({
            "check": "group_results_present",
            "status": "FAIL",
            "reason": "No group-level means are available in the evidence.",
        })
        group_checks = []
    else:
        checks.append({
            "check": "group_results_present",
            "status": "PASS",
            "group_count": len(groups),
        })

        def finite_number(value):
            try:
                number = float(value)
                return number if np.isfinite(number) else None
            except (TypeError, ValueError, OverflowError):
                return None

        group_checks = []
        for name, details in groups.items():
            mean = None
            if isinstance(details, dict):
                mean = finite_number(details.get("mean_value", details.get("mean")))
            normalized_label = " ".join(
                str(name).casefold().replace("_", " ").split()
            )
            label_found = bool(normalized_label) and normalized_label in lower_text
            mean_found = False
            pair_found = False
            if mean is not None:
                candidates = {f"{mean:,.2f}", f"{mean:.2f}"}
                mean_found = any(
                    re.search(r"(?<!\d)" + re.escape(candidate) + r"(?!\d)", text)
                    for candidate in candidates
                )
                pair_found = any(
                    re.search(
                        re.escape(str(name)) + r"\s*:\s*" + re.escape(candidate),
                        text,
                        flags=re.IGNORECASE,
                    )
                    for candidate in candidates
                )
            passed = label_found and mean_found and pair_found
            group_checks.append({
                "group": str(name),
                "validated_mean": mean,
                "group_label_mentioned": label_found,
                "mean_mentioned": mean_found,
                "correct_label_mean_pair": pair_found,
                "status": "PASS" if passed else "FAIL",
                "reason": None if passed else
                    "The insight does not pair this composite group with its own validated mean.",
            })

    all_groups_pass = bool(group_checks) and all(
        item["status"] == "PASS" for item in group_checks
    )
    checks.append({
        "check": "all_group_means_match_insight",
        "status": "PASS" if all_groups_pass else "FAIL",
        "total_groups": len(group_checks),
        "passed_groups": sum(item["status"] == "PASS" for item in group_checks),
    })

    evidence_pass = str(evidence.get("status", "PASS")).upper() == "PASS"
    checks.append({
        "check": "validated_evidence_status",
        "status": "PASS" if evidence_pass else "FAIL",
    })

    passed = all(item["status"] == "PASS" for item in checks)
    return {
        "status": "PASS" if passed else "FAIL",
        "analysis_type": "grouped_mean_analysis",
        "metric_column": metric or None,
        "group_column": " and ".join(group_columns) if group_columns else None,
        "group_columns": group_columns,
        "claim_validation": {
            "status": "PASS" if passed else "FAIL",
            "total_claims": len(checks) + len(group_checks),
            "passed_claims": (
                sum(item["status"] == "PASS" for item in checks)
                + sum(item["status"] == "PASS" for item in group_checks)
            ),
            "failed_claims": (
                sum(item["status"] == "FAIL" for item in checks)
                + sum(item["status"] == "FAIL" for item in group_checks)
            ),
            "checks": checks,
            "group_checks": group_checks,
        },
        "evidence_validation": {
            "status": "PASS" if evidence_pass else "FAIL",
        },
        "reason": None if passed else
            "The grouped-mean insight does not fully match the validated composite group-level evidence.",
    }


def attach_evidence_traceability(mapped_evidence, evidence):
    """Attach explicit source-result references to the insight mapping."""
    if not isinstance(mapped_evidence, dict):
        mapped_evidence = {"status": "FAIL", "reason": "Mapping output was not a dictionary."}

    sources = []
    if not isinstance(evidence, dict):
        mapped_evidence["evidence_sources"] = sources
        return mapped_evidence

    analysis_type = str(evidence.get("analysis_type", "unknown"))
    if analysis_type == "multi_metric_grouped_mean_analysis":
        metric_results = evidence.get("metric_results", {})
        if isinstance(metric_results, dict):
            for metric, result in metric_results.items():
                sources.append({
                    "source_id": f"primary_result.metric_results.{metric}",
                    "source_type": "validated_grouped_mean_result",
                    "group_column": evidence.get("group_column"),
                    "metric_column": metric,
                    "aggregation": "mean",
                    "result": safe_json_value(result),
                })
    elif analysis_type == "specification_driven_grouped_mean":
        sources.append({
            "source_id": "primary_result.operation_results.aggregations",
            "source_type": "independently_validated_specification_grouped_mean",
            "group_column": evidence.get("group_column"),
            "metric_column": evidence.get("metric_column"),
            "aggregation": "mean",
            "result": safe_json_value(evidence.get("supporting_result")),
        })
    elif analysis_type == "grouped_mean_analysis":
        sources.append({
            "source_id": "primary_result.group_results",
            "source_type": "validated_grouped_mean_result",
            "group_column": evidence.get("group_column"),
            "metric_column": evidence.get("metric_column"),
            "aggregation": "mean",
            "result": safe_json_value(evidence.get("supporting_result", evidence.get("result"))),
        })
    elif analysis_type == "grouped_analysis":
        sources.append({
            "source_id": "grouped_series",
            "source_type": "grouped_aggregation",
            "category_column": evidence.get("category"),
            "measure_column": evidence.get("metric"),
            "aggregation": evidence.get("operation"),
            "result": safe_json_value(evidence.get("supporting_result")),
        })
    elif analysis_type == "correlation_analysis":
        sources.append({
            "source_id": "correlation_result",
            "source_type": "correlation_analysis",
            "x_variable": evidence.get("x_variable"),
            "y_variable": evidence.get("y_variable"),
            "method": evidence.get("method"),
            "correlation": evidence.get("correlation"),
            "result": safe_json_value(evidence.get("supporting_result", evidence.get("evidence"))),
        })
    elif analysis_type in ("trend_analysis", "time_series_analysis", "timeseries_analysis"):
        sources.append({
            "source_id": "trend_result",
            "source_type": "time_series_analysis",
            "time_column": evidence.get("time_label"),
            "measure_column": evidence.get("metric"),
            "result": safe_json_value(evidence.get("supporting_result", evidence.get("evidence"))),
        })
    else:
        primary = evidence.get("result", evidence.get("primary_result"))
        if primary is not None:
            sources.append({
                "source_id": "primary_result",
                "source_type": "validated_analysis_result",
                "result": safe_json_value(primary),
            })
        facts = evidence.get("supporting_result", {}).get("dataset_facts") if isinstance(evidence.get("supporting_result"), dict) else None
        if isinstance(facts, dict):
            if facts.get("numeric_summary"):
                sources.append({
                    "source_id": "dataset_facts.numeric_summary",
                    "source_type": "independently_computed_dataset_summary",
                    "result": safe_json_value(facts.get("numeric_summary")),
                })
            if facts.get("profit_by_region"):
                sources.append({
                    "source_id": "dataset_facts.profit_by_region",
                    "source_type": "independently_computed_grouped_result",
                    "group_column": "Region",
                    "measure_column": "Total Profit",
                    "aggregation": "sum",
                    "result": safe_json_value(facts.get("profit_by_region")),
                })
        for step in evidence.get("supporting_result", {}).get("execution_steps", []) if isinstance(evidence.get("supporting_result"), dict) else []:
            if isinstance(step, dict):
                sources.append({
                    "source_id": f"execution_steps[{step.get('step_number')}]",
                    "source_type": "pipeline_execution_step",
                    "step_number": step.get("step_number"),
                    "step_name": step.get("step_name"),
                    "execution_status": step.get("execution_status"),
                    "validation_status": step.get("validation_status"),
                })

    mapped_evidence["evidence_sources"] = sources
    mapped_evidence["traceability_status"] = "PASS" if sources and mapped_evidence.get("status") == "PASS" else "REVIEW"
    return mapped_evidence


# =========================================================
# PRIMARY RESULT
# =========================================================

def get_primary_result(
    aggregated_result,
    execution_results
):

    if isinstance(
        aggregated_result,
        dict
    ):

        possible_keys = [
            "primary_result",
            "result",
            "validated_result",
            "selected_result"
        ]

        for key in possible_keys:

            if key in aggregated_result:

                value = aggregated_result.get(
                    key
                )

                if value is not None:

                    return value

    return get_step_result(
        execution_results,
        5
    )




# =========================================================
# MULTI-METRIC GROUPED-MEAN COMPARISON
# =========================================================

def _is_valid_grouped_mean_result(result):
    """
    Return True only when a grouped-mean result contains valid group means
    and a mathematically consistent yes-minus-no difference.
    """
    if not isinstance(result, dict):
        return False

    if str(result.get("analysis_type", "")).lower() != "grouped_mean_analysis":
        return False

    groups = result.get("group_results")

    if not isinstance(groups, dict):
        return False

    no_group = groups.get("no")
    yes_group = groups.get("yes")

    if not isinstance(no_group, dict) or not isinstance(yes_group, dict):
        return False

    def finite_number(value):
        return (
            isinstance(value, (int, float, np.integer, np.floating))
            and not isinstance(value, (bool, np.bool_))
            and np.isfinite(float(value))
        )

    no_mean = no_group.get("mean_value", no_group.get("mean"))
    yes_mean = yes_group.get("mean_value", yes_group.get("mean"))
    difference = result.get("mean_difference_yes_minus_no")

    if not all(
        finite_number(value)
        for value in (no_mean, yes_mean, difference)
    ):
        return False

    expected_difference = float(yes_mean) - float(no_mean)

    if not np.isfinite(expected_difference):
        return False

    return bool(
        np.isclose(
            float(difference),
            expected_difference,
            rtol=1e-9,
            atol=1e-9,
        )
    )


def build_multi_metric_grouped_mean_result(task_analysis, execution_results):
    """
    Assemble all requested grouped-mean metrics from validated executor steps.

    A metric is included only when a corresponding successful, structurally valid
    grouped-mean result exists. Missing requested metrics force status=FAIL.
    """
    if not isinstance(task_analysis, dict) or not isinstance(execution_results, dict):
        return None

    requested = task_analysis.get("requested_metric_columns", [])
    if not isinstance(requested, list):
        return None

    requested = list(dict.fromkeys(
        str(column).strip() for column in requested
        if isinstance(column, str) and str(column).strip()
    ))
    if len(requested) < 2:
        return None

    steps = execution_results.get("steps", [])
    if not isinstance(steps, list):
        steps = []

    metric_results = {}
    for metric in requested:
        candidates = []
        for step in steps:
            if not isinstance(step, dict):
                continue

            # Do not promote failed execution/validation attempts into the report.
            status = str(step.get("status", "")).upper()
            execution_status = str(step.get("execution_status", "")).upper()
            validation_status = str(step.get("validation_status", "")).upper()
            success_labels = {"PASS", "SUCCESS", "OK"}
            if status not in success_labels:
                continue
            if execution_status and execution_status not in success_labels:
                continue
            if validation_status and validation_status not in success_labels:
                continue

            result = step.get("final_result")
            if not isinstance(result, dict):
                result = step.get("result")
            if (
                isinstance(result, dict)
                and str(result.get("metric_column", "")).strip() == metric
                and _is_valid_grouped_mean_result(result)
            ):
                candidates.append((step, result))

        if candidates:
            # Prefer the first actual comparison result over a later independent
            # validation/recomputation step when both have the same metric.
            preferred = next(
                (
                    item for item in candidates
                    if "compare" in str(item[0].get("step", "")).lower()
                ),
                candidates[0],
            )
            metric_results[metric] = safe_json_value(preferred[1])

    missing = [metric for metric in requested if metric not in metric_results]
    all_valid = not missing and all(
        _is_valid_grouped_mean_result(result)
        for result in metric_results.values()
    )

    first_result = next(iter(metric_results.values()), {})
    return {
        "status": "PASS" if all_valid else "FAIL",
        "analysis_type": "multi_metric_grouped_mean_analysis",
        "group_column": first_result.get("group_column", task_analysis.get("group_column", "y")),
        "requested_metric_columns": requested,
        "metric_results": metric_results,
        "validated_metric_count": len(metric_results),
        "requested_metric_count": len(requested),
        "missing_metrics": missing,
        "reason": None if all_valid else (
            "One or more requested metrics are missing or failed grouped-mean validation: "
            + ", ".join(missing or requested)
        ),
    }


def generate_multi_metric_grouped_mean_insight(combined_result):
    """Create one deterministic insight that explicitly reports every requested metric."""
    if not isinstance(combined_result, dict):
        return "The multi-metric comparison could not be generated because the combined result is missing."

    requested = combined_result.get("requested_metric_columns", [])
    results = combined_result.get("metric_results", {})
    missing = combined_result.get("missing_metrics", [])
    if combined_result.get("status") != "PASS" or missing:
        missing_text = ", ".join(missing) if missing else "one or more requested metrics"
        return (
            "The multi-metric comparison could not be fully validated. "
            f"Missing or invalid results: {missing_text}. No incomplete comparison is reported as complete."
        )

    statements = []
    for metric in requested:
        result = results.get(metric)
        if not _is_valid_grouped_mean_result(result):
            return f"The comparison for {metric} failed structural validation."
        groups = result["group_results"]
        no_group = groups["no"]
        yes_group = groups["yes"]
        no_mean = float(no_group.get("mean_value", no_group.get("mean")))
        yes_mean = float(yes_group.get("mean_value", yes_group.get("mean")))
        difference = float(result["mean_difference_yes_minus_no"])
        direction = "higher" if difference > 0 else "lower" if difference < 0 else "the same"
        statements.append(
            f"{metric}: non-subscriber average = {no_mean:,.2f}, "
            f"subscriber average = {yes_mean:,.2f}, "
            f"subscriber-minus-non-subscriber difference = {difference:+,.2f} "
            f"(subscriber average is {direction})."
        )

    return (
        "Average comparison between customers who subscribed and those who did not: "
        + " ".join(statements)
        + " These are descriptive differences and do not establish causation."
    )


def build_multi_metric_evidence_mapping(insight, evidence, insight_validation):
    """Create deterministic evidence mapping for every requested grouped-mean metric."""
    evidence = evidence if isinstance(evidence, dict) else {}
    results = evidence.get("metric_results", {})
    requested = evidence.get("requested_metric_columns", [])
    text = str(insight or "").lower()

    metric_checks = {}
    for metric in requested if isinstance(requested, list) else []:
        metric_result = results.get(metric) if isinstance(results, dict) else None
        metric_checks[metric] = {
            "result_present": _is_valid_grouped_mean_result(metric_result),
            "metric_named_in_insight": str(metric).lower() in text,
        }
        metric_checks[metric]["pass"] = all(metric_checks[metric].values())

    detailed_validation = validate_multi_metric_grouped_mean_insight(
        insight=insight,
        combined_result=evidence,
        base_validation=insight_validation,
    )
    insight_pass = (
        isinstance(insight_validation, dict)
        and insight_validation.get("pass") is True
        and str(insight_validation.get("overall", "")).upper() == "PASS"
        and detailed_validation.get("pass") is True
    )
    evidence_pass = (
        evidence.get("status") == "PASS"
        and isinstance(results, dict)
        and len(results) == len(requested)
        and bool(requested)
    )
    metrics_pass = bool(metric_checks) and all(
        check.get("pass") is True for check in metric_checks.values()
    )
    passed = evidence_pass and metrics_pass and insight_pass

    return {
        "status": "PASS" if passed else "FAIL",
        "analysis_type": "multi_metric_grouped_mean_analysis",
        "claim_validation": {
            "status": "PASS" if metrics_pass else "FAIL",
            "total_claims": len(metric_checks),
            "passed_claims": sum(1 for check in metric_checks.values() if check.get("pass")),
            "failed_claims": sum(1 for check in metric_checks.values() if not check.get("pass")),
            "checks": metric_checks,
        },
        "evidence_validation": {
            "status": "PASS" if evidence_pass else "FAIL",
            "requested_metric_count": len(requested) if isinstance(requested, list) else 0,
            "evidence_metric_count": len(results) if isinstance(results, dict) else 0,
        },
        "insight_validation": {
            "status": "PASS" if insight_pass else "FAIL",
        },
        "reason": None if passed else (
            "Multi-metric evidence mapping requires valid evidence and insight claims "
            "for every requested metric."
        ),
    }


def validate_multi_metric_grouped_mean_insight(insight, combined_result, base_validation=None):
    """Require every requested metric and its means/difference to be present in the insight."""
    try:
        from .grouped_mean_claim_validation import metric_group_mean_checks
    except ImportError:
        from grouped_mean_claim_validation import metric_group_mean_checks

    requested = (
        combined_result.get("requested_metric_columns", [])
        if isinstance(combined_result, dict) else []
    )
    results = (
        combined_result.get("metric_results", {})
        if isinstance(combined_result, dict) else {}
    )
    text = str(insight or "").lower()
    checks = {}

    def number_present(expected):
        expected = float(expected)
        # Match the formatted number with or without thousands separators.
        token = f"{expected:,.2f}".lower()
        plain = f"{expected:.2f}".lower()
        return token in text or plain in text

    for metric in requested:
        result = results.get(metric)
        metric_ok = _is_valid_grouped_mean_result(result)
        groups = result.get("group_results", {}) if metric_ok else {}
        no_group = groups.get("no", {}) if isinstance(groups, dict) else {}
        yes_group = groups.get("yes", {}) if isinstance(groups, dict) else {}
        no_mean = no_group.get("mean_value", no_group.get("mean"))
        yes_mean = yes_group.get("mean_value", yes_group.get("mean"))
        difference = result.get("mean_difference_yes_minus_no") if metric_ok else None

        association = (
            metric_group_mean_checks(insight, metric, result, requested)
            if metric_ok else {}
        )
        checks[metric] = {
            "result_valid": metric_ok,
            "metric_named": bool(association.get("statement_present")),
            "non_subscriber_mean_present": bool(
                association.get("non_subscriber_mean_present")
            ),
            "subscriber_mean_present": bool(
                association.get("subscriber_mean_present")
            ),
            "difference_present": bool(
                association.get("difference_present")
            ),
            "reason": association.get("reason"),
        }
        checks[metric]["pass"] = all(
            checks[metric][key]
            for key in (
                "result_valid",
                "metric_named",
                "non_subscriber_mean_present",
                "subscriber_mean_present",
                "difference_present",
            )
        )

    all_metrics_pass = (
        bool(requested)
        and len(requested) == len(results)
        and all(item.get("pass") for item in checks.values())
        and isinstance(combined_result, dict)
        and combined_result.get("status") == "PASS"
    )
    base_pass = True
    if isinstance(base_validation, dict):
        if "pass" in base_validation:
            base_pass = base_validation.get("pass") is True
        elif "overall" in base_validation:
            base_pass = str(base_validation.get("overall")).upper() == "PASS"

    passed = all_metrics_pass and base_pass
    return {
        "overall": "PASS" if passed else "FAIL",
        "pass": passed,
        "checks": {
            "multi_metric_claim_validation": {
                "pass": all_metrics_pass,
                "requested_metric_count": len(requested),
                "validated_metric_count": len(results),
                "metrics": checks,
            },
            "base_insight_validation": {
                "pass": base_pass,
                "result": base_validation,
            },
        },
        "reason": None if passed else (
            "Every requested metric must have a valid result and its metric name, "
            "both group means, and difference must appear in the insight."
        ),
    }
def validate_specification_multi_metric_grouped_mean_insight(
    insight,
    primary_result,
    base_validation=None,
):
    """
    Validate specification-driven grouped means for multiple metrics
    and arbitrary categorical groups, such as City.
    """
    text = str(insight or "").lower()

    checks = {
        "result_status": False,
        "analysis_type": False,
        "mean_operation": False,
        "metric_count": False,
        "single_grouping_column": False,
        "aggregation_rows": False,
        "metric_claims": {},
    }

    if not isinstance(primary_result, dict):
        base_pass = (
            base_validation.get("pass") is True
            if isinstance(base_validation, dict)
            else False
        )
        return {
            "overall": "FAIL",
            "pass": False,
            "checks": checks,
            "reason": "Primary result must be a dictionary.",
        }

    checks["result_status"] = primary_result.get("status") == "PASS"
    checks["analysis_type"] = (
        str(primary_result.get("analysis_type", "")).lower()
        == "specification_driven"
    )

    operations = primary_result.get("operations", [])
    operations = (
        [str(item).lower() for item in operations]
        if isinstance(operations, (list, tuple, set))
        else []
    )
    checks["mean_operation"] = "mean" in operations

    metrics = primary_result.get("metrics", [])
    metrics = metrics if isinstance(metrics, list) else []
    checks["metric_count"] = len(metrics) >= 2

    group_by = primary_result.get("group_by", [])
    group_by = group_by if isinstance(group_by, list) else []
    checks["single_grouping_column"] = len(group_by) == 1

    operation_results = primary_result.get("operation_results", {})
    aggregations = (
        operation_results.get("aggregations", [])
        if isinstance(operation_results, dict)
        else []
    )

    checks["aggregation_rows"] = (
        isinstance(aggregations, list) and len(aggregations) > 0
    )

    def number_is_present(expected):
        try:
            value = float(expected)
            if not math.isfinite(value):
                return False

            formatted = f"{value:,.2f}"
            plain = f"{value:.2f}"

            for token in {formatted, plain}:
                pattern = (
                    r"(?<!\d)"
                    + re.escape(token)
                    + r"(?!\d)"
                )
                if re.search(pattern, text):
                    return True

            return False
        except (TypeError, ValueError, OverflowError):
            return False

    metric_claims = {}
    valid_structure = (
        checks["result_status"]
        and checks["analysis_type"]
        and checks["mean_operation"]
        and checks["metric_count"]
        and checks["single_grouping_column"]
        and checks["aggregation_rows"]
    )

    if valid_structure:
        grouping_column = str(group_by[0])
        grouping_column_present = grouping_column.lower() in text

        seen_groups = set()
        row_checks = []

        for row in aggregations:
            if not isinstance(row, dict):
                row_checks.append(False)
                continue

            group_value = row.get(grouping_column)

            if group_value is None:
                row_checks.append(False)
                continue

            group_label = str(group_value).strip()
            group_key = group_label.lower()

            if not group_label or group_key in seen_groups:
                row_checks.append(False)
                continue

            seen_groups.add(group_key)
            group_label_present = group_key in text

            for metric in metrics:
                metric = str(metric)
                result_key = f"{metric}__mean"
                expected_value = row.get(result_key)

                try:
                    numeric_value = float(expected_value)
                    value_valid = math.isfinite(numeric_value)
                except (TypeError, ValueError, OverflowError):
                    value_valid = False
                    numeric_value = None

                claim = metric_claims.setdefault(
                    metric,
                    {
                        "metric_named": metric.lower() in text,
                        "grouping_column_named": grouping_column_present,
                        "all_group_labels_present": True,
                        "all_group_means_present": True,
                        "groups_checked": 0,
                    },
                )

                claim["groups_checked"] += 1

                if not group_label_present:
                    claim["all_group_labels_present"] = False

                if (
                    not value_valid
                    or not number_is_present(numeric_value)
                ):
                    claim["all_group_means_present"] = False

            row_checks.append(group_label_present)

        all_rows_valid = (
            len(row_checks) == len(aggregations)
            and all(row_checks)
        )

        for claim in metric_claims.values():
            claim["pass"] = all(
                [
                    claim["metric_named"],
                    claim["grouping_column_named"],
                    claim["all_group_labels_present"],
                    claim["all_group_means_present"],
                    claim["groups_checked"] == len(aggregations),
                ]
            )

        checks["all_group_labels_present"] = all_rows_valid
        checks["metric_claims"] = metric_claims

    else:
        checks["all_group_labels_present"] = False

    all_metric_claims_pass = (
        len(metric_claims) == len(metrics)
        and len(metric_claims) >= 2
        and all(
            claim.get("pass", False)
            for claim in metric_claims.values()
        )
    )

    checks["all_metric_claims_pass"] = all_metric_claims_pass

    base_pass = True
    if isinstance(base_validation, dict):
        if "pass" in base_validation:
            base_pass = base_validation.get("pass") is True
        elif "overall" in base_validation:
            base_pass = (
                str(base_validation.get("overall")).upper() == "PASS"
            )

    checks["base_insight_validation"] = {
        "pass": base_pass,
        "result": base_validation,
    }

    structural_pass = all(
        [
            checks["result_status"],
            checks["analysis_type"],
            checks["mean_operation"],
            checks["metric_count"],
            checks["single_grouping_column"],
            checks["aggregation_rows"],
            checks["all_group_labels_present"],
            all_metric_claims_pass,
        ]
    )

    passed = structural_pass and base_pass

    return {
        "overall": "PASS" if passed else "FAIL",
        "pass": passed,
        "checks": checks,
        "reason": None if passed else (
            "Specification-driven multi-metric grouped-mean validation "
            "requires every requested metric, grouping column, group label, "
            "and independently calculated group mean to be present and valid."
        ),
    }

# =========================================================
# CREATE EVIDENCE
# =========================================================

def build_job_conversion_rate_series(filepath):
    """Compute conversion rates per job category directly from the uploaded dataset."""
    frame = pd.read_csv(filepath)
    if "job" not in frame.columns:
        return None

    if "conversion" in frame.columns:
        outcome = pd.to_numeric(frame["conversion"], errors="coerce")
        converted = outcome.eq(1)
        valid = outcome.isin([0, 1])
    elif "y" in frame.columns:
        labels = frame["y"].astype(str).str.strip().str.lower()
        converted = labels.eq("yes")
        valid = labels.isin(["yes", "no"])
    else:
        return None

    work = frame.loc[valid & frame["job"].notna(), ["job"]].copy()
    work["_converted"] = converted.loc[work.index].astype(int)
    grouped = work.groupby("job", dropna=False)["_converted"].agg(["sum", "count"])
    grouped["conversion_rate_percent"] = grouped["sum"] / grouped["count"] * 100
    grouped = grouped.sort_values("conversion_rate_percent", ascending=False)
    rates = grouped["conversion_rate_percent"].round(4)
    rates.name = "Conversion Rate (%)"
    # Preserve counts for the evidence payload without changing the Series API.
    rates.attrs["grouped_counts"] = {
        str(category): {
            "converted_count": int(row["sum"]),
            "total_count": int(row["count"]),
            "conversion_rate_percent": round(float(row["conversion_rate_percent"]), 4)
        }
        for category, row in grouped.iterrows()
    }
    return rates


def _normalized_detection_method(result):
    """Return the explicit outlier method from a result without guessing."""
    if not isinstance(result, dict):
        return ""
    summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
    method = result.get("method") or summary.get("algorithm") or summary.get("method") or ""
    return re.sub(r"[\s_-]+", "", str(method).strip().lower())


def _find_iqr_result(result, depth=0, metric_hint=None):
    """Find an IQR result, including specification-driven operation_results."""
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

    method = _normalized_detection_method(result)
    if method in {"iqr", "interquartilerange"}:
        status = str(result.get("status", "PASS")).upper()
        if status == "PASS":
            normalized = dict(result)
            if metric_hint and not any(normalized.get(key) for key in ("metric_column", "column", "metric")):
                normalized["metric_column"] = str(metric_hint)
            normalized.setdefault("analysis_type", "outlier_analysis")
            return normalized

    # Specification-driven EDA stores results as:
    # operation_results -> outlier_detection -> <metric> -> {method, bounds, counts}.
    operation_results = result.get("operation_results")
    if isinstance(operation_results, dict):
        for operation_name, operation_result in operation_results.items():
            if str(operation_name).lower() in {"outlier_detection", "outlier_analysis", "anomaly_detection"}:
                if isinstance(operation_result, dict):
                    for metric_name, metric_result in operation_result.items():
                        found = _find_iqr_result(metric_result, depth + 1, metric_name)
                        if found is not None:
                            return found
            found = _find_iqr_result(operation_result, depth + 1, metric_hint)
            if found is not None:
                return found

    # Also inspect common result containers and any nested dict/list values.
    preferred_keys = (
        "primary_result", "result", "validated_result", "selected_result",
        "supporting_result", "summary", "outlier_detection", "results",
    )
    visited = set()
    for key in preferred_keys:
        if key in result:
            visited.add(key)
            found = _find_iqr_result(result.get(key), depth + 1, metric_hint)
            if found is not None:
                return found

    for key, value in result.items():
        if key in visited or key in {"dataset_facts", "numeric_summary", "categorical_summary"}:
            continue
        if isinstance(value, (dict, list)):
            child_metric = str(key) if key not in {"operation_results", "outlier_detection", "results"} else metric_hint
            found = _find_iqr_result(value, depth + 1, child_metric)
            if found is not None:
                return found

    return None


def create_evidence(
    analysis_type,
    primary_result,
    grouped_series=None,
    execution_results=None,
    data_profile=None,
    dataset_path=None
):

    analysis_type = str(
        analysis_type
    ).lower()

    # Isolation Forest evidence must preserve its validated counts and scores.
    # Do not route anomaly results through generic descriptive evidence.
    isolation_summary = (
        primary_result.get("summary", {})
        if isinstance(primary_result, dict)
        and isinstance(primary_result.get("summary"), dict)
        else {}
    )
    isolation_method = (
        primary_result.get("method")
        if isinstance(primary_result, dict)
        else None
    )
    if not isolation_method:
        isolation_method = isolation_summary.get(
            "algorithm", isolation_summary.get("method", "")
        )
    normalized_isolation_method = re.sub(
        r"[\s_-]+", "", str(isolation_method).lower()
    )
    iqr_result = _find_iqr_result(primary_result)
    if iqr_result is not None:
        metric = iqr_result.get("metric_column") or iqr_result.get("column") or iqr_result.get("metric")
        lower = iqr_result.get("lower_bound")
        upper = iqr_result.get("upper_bound")
        outlier_count = iqr_result.get("outlier_count", iqr_result.get("anomaly_count"))
        valid_count = iqr_result.get("valid_count", iqr_result.get("valid_value_count"))
        row_count = iqr_result.get(
            "row_count",
            primary_result.get("analyzed_rows", primary_result.get("input_rows"))
            if isinstance(primary_result, dict) else None,
        )
        def _finite_number(value):
            try:
                number = float(value)
                return number if np.isfinite(number) else None
            except (TypeError, ValueError, OverflowError):
                return None
        lower_value, upper_value = _finite_number(lower), _finite_number(upper)
        count_valid = (isinstance(outlier_count, (int, np.integer)) and not isinstance(outlier_count, (bool, np.bool_)) and int(outlier_count) >= 0)
        valid_count_valid = (isinstance(valid_count, (int, np.integer)) and not isinstance(valid_count, (bool, np.bool_)) and int(valid_count) >= 0)
        evidence_valid = (str(iqr_result.get("status", "PASS")).upper() == "PASS" and isinstance(metric, str) and bool(metric.strip()) and lower_value is not None and upper_value is not None and lower_value <= upper_value and count_valid and valid_count_valid and int(outlier_count) <= int(valid_count))
        payload = {
            "analysis_type": "outlier_analysis", "method": "IQR", "status": str(iqr_result.get("status", "PASS")).upper(),
            "metric_column": str(metric).strip() if metric else None, "row_count": row_count,
            "valid_value_count": int(valid_count) if valid_count_valid else None,
            "outlier_count": int(outlier_count) if count_valid else None,
            "lower_bound": lower_value, "upper_bound": upper_value,
            "result": safe_json_value(iqr_result), "supporting_result": safe_json_value(iqr_result),
        }
        return {**payload, "status": "PASS" if evidence_valid else "FAIL",
                "evidence_summary": json.dumps(safe_json_value(payload), indent=2, ensure_ascii=False, default=str),
                "reason": None if evidence_valid else "IQR evidence failed method, metric, bounds, or count validation."}

    is_isolation_forest = normalized_isolation_method == "isolationforest"
    if is_isolation_forest:
        result_status = (
            str(primary_result.get("status", "")).upper()
            if isinstance(primary_result, dict)
            else ""
        )
        row_count = (
            primary_result.get("row_count", isolation_summary.get("total_rows"))
            if isinstance(primary_result, dict) else None
        )
        anomaly_count = (
            primary_result.get("anomaly_count", isolation_summary.get("anomaly_count"))
            if isinstance(primary_result, dict) else None
        )
        normal_count = (
            primary_result.get("normal_count", isolation_summary.get("normal_count"))
            if isinstance(primary_result, dict) else None
        )
        anomaly_percentage = (
            primary_result.get(
                "anomaly_percentage",
                isolation_summary.get("detected_anomaly_percentage"),
            )
            if isinstance(primary_result, dict) else None
        )
        feature_columns = (
            primary_result.get(
                "feature_columns", isolation_summary.get("feature_columns", [])
            )
            if isinstance(primary_result, dict) else []
        )
        contamination = (
            primary_result.get(
                "contamination", isolation_summary.get("contamination_setting")
            )
            if isinstance(primary_result, dict) else None
        )

        def _valid_nonnegative_integer(value):
            return (
                isinstance(value, (int, np.integer))
                and not isinstance(value, (bool, np.bool_))
                and int(value) >= 0
            )

        counts_valid = all((
            _valid_nonnegative_integer(row_count),
            _valid_nonnegative_integer(anomaly_count),
            _valid_nonnegative_integer(normal_count),
        ))
        try:
            percentage_value = float(anomaly_percentage)
            percentage_valid = (
                np.isfinite(percentage_value)
                and 0 <= percentage_value <= 100
            )
        except (TypeError, ValueError, OverflowError):
            percentage_value = None
            percentage_valid = False

        count_reconciliation = (
            counts_valid
            and int(row_count) > 0
            and int(row_count) == int(anomaly_count) + int(normal_count)
        )
        percentage_reconciliation = (
            count_reconciliation
            and percentage_valid
            and abs(
                percentage_value - (int(anomaly_count) / int(row_count) * 100)
            ) <= 0.02
        )
        method_valid = normalized_isolation_method == "isolationforest"
        features_valid = (
            isinstance(feature_columns, (list, tuple))
            and len(feature_columns) > 0
        )
        evidence_valid = all((
            result_status == "PASS",
            method_valid,
            counts_valid,
            count_reconciliation,
            percentage_valid,
            percentage_reconciliation,
            features_valid,
        ))

        result_payload = safe_json_value(primary_result)
        evidence_payload = {
            "analysis_type": "anomaly_detection",
            "method": "Isolation Forest",
            "status": result_status,
            "row_count": int(row_count) if _valid_nonnegative_integer(row_count) else None,
            "anomaly_count": int(anomaly_count) if _valid_nonnegative_integer(anomaly_count) else None,
            "normal_count": int(normal_count) if _valid_nonnegative_integer(normal_count) else None,
            "anomaly_percentage": percentage_value,
            "feature_columns": list(feature_columns) if features_valid else [],
            "contamination": contamination,
            "summary": safe_json_value(isolation_summary),
            "result": result_payload,
            "supporting_result": result_payload,
        }
        return {
            **evidence_payload,
            "status": "PASS" if evidence_valid else "FAIL",
            "evidence_summary": json.dumps(
                safe_json_value(evidence_payload),
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            "reason": None if evidence_valid else (
                "Isolation Forest evidence failed method, status, feature, count, "
                "or anomaly-percentage validation."
            ),
        }

    # Specification-driven grouped means carry their own validated aggregation rows.
    # Normalize those rows directly; never reconstruct a legacy Region -> Total Profit series.
    specification_grouped = extract_specification_grouped_mean(primary_result)
    if specification_grouped is not None:
        raw_result = safe_json_value(primary_result)
        evidence_payload = {
            "status": "PASS",
            "analysis_type": "specification_driven_grouped_mean",
            "metric_column": specification_grouped["metric_column"],
            "group_column": specification_grouped["group_column"],
            "group_columns": specification_grouped["group_columns"],
            "group_results": safe_json_value(specification_grouped["group_results"]),
            "result": safe_json_value(specification_grouped),
            "supporting_result": raw_result,
            "evidence_summary": json.dumps(
                safe_json_value(specification_grouped),
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            "reason": None,
        }
        return evidence_payload

    # Multi-metric grouped means must be preserved as a single evidence bundle.
    if (
        isinstance(primary_result, dict)
        and str(primary_result.get("analysis_type", "")).lower()
        == "multi_metric_grouped_mean_analysis"
    ):
        valid = (
            primary_result.get("status") == "PASS"
            and isinstance(primary_result.get("metric_results"), dict)
            and primary_result.get("requested_metric_count")
                == primary_result.get("validated_metric_count")
            and not primary_result.get("missing_metrics")
            and all(
                _is_valid_grouped_mean_result(value)
                for value in primary_result["metric_results"].values()
            )
        )
        payload = safe_json_value(primary_result)
        return {
            "status": "PASS" if valid else "FAIL",
            "analysis_type": "multi_metric_grouped_mean_analysis",
            "group_column": primary_result.get("group_column"),
            "requested_metric_columns": primary_result.get("requested_metric_columns", []),
            "metric_results": safe_json_value(primary_result.get("metric_results", {})),
            "supporting_result": payload,
            "result": payload,
            "evidence_summary": json.dumps(payload, indent=2, ensure_ascii=False),
            "reason": None if valid else primary_result.get(
                "reason", "One or more requested metric results failed validation."
            ),
        }

    # Preserve the real conversion-factor result; do not send it to
    # the legacy Pearson correlation or sales/profit evidence branches.
    if (
        analysis_type == "conversion_factor_analysis"
        or (
            isinstance(primary_result, dict)
            and primary_result.get("analysis_type") == "conversion_factor_analysis"
        )
    ):
        valid_result = (
            isinstance(primary_result, dict)
            and isinstance(primary_result.get("categorical_factor_results"), dict)
            and isinstance(primary_result.get("numeric_factor_results"), dict)
            and isinstance(primary_result.get("overall_conversion_rate_percent"), (int, float))
        )
        evidence_payload = safe_json_value(primary_result)
        return {
            "status": "PASS" if valid_result else "FAIL",
            "analysis_type": "conversion_factor_analysis",
            "target_column": primary_result.get("target_column") if isinstance(primary_result, dict) else None,
            "supporting_result": evidence_payload,
            "result": evidence_payload,
            "evidence_summary": json.dumps(
                evidence_payload, indent=2, ensure_ascii=False
            ),
            "reason": None if valid_result else
                "Validated conversion-factor result is missing required fields.",
        }

    # -----------------------------------------------------
    # GROUPED CONVERSION RATE
    # -----------------------------------------------------
    if (
        dataset_path
        and grouped_series is not None
        and str(getattr(grouped_series, "name", "")).lower() == "conversion rate (%)"
    ):
        grouped_evidence = create_grouped_evidence(
            grouped_series,
            "job",
            "Conversion Rate (%)",
            "conversion_rate"
        )
        supporting = {
            "grouped_result": safe_json_value(grouped_series),
            "grouped_counts": safe_json_value(grouped_series.attrs.get("grouped_counts", {})),
            "top_category": str(grouped_series.index[0]) if len(grouped_series) else None,
            "top_conversion_rate_percent": float(grouped_series.iloc[0]) if len(grouped_series) else None,
        }
        if isinstance(grouped_evidence, dict):
            grouped_evidence["analysis_type"] = "grouped_analysis"
            grouped_evidence["category"] = str(primary_result.get("group_column", "job"))
            grouped_evidence["metric"] = "Conversion Rate (%)"
            grouped_evidence["operation"] = "conversion_rate"
            grouped_evidence["supporting_result"] = supporting
            grouped_evidence["result"] = safe_json_value(supporting)
            grouped_evidence["evidence_summary"] = json.dumps(
                safe_json_value(supporting), indent=2, ensure_ascii=False
            )
            grouped_evidence.setdefault("reason", None)
            return grouped_evidence
        return {
            "status": "PASS" if len(grouped_series) > 0 else "FAIL",
            "analysis_type": "grouped_analysis",
            "category": str(primary_result.get("group_column", "job")),
            "metric": "Conversion Rate (%)",
            "operation": "conversion_rate",
            "supporting_result": supporting,
            "result": safe_json_value(supporting),
            "evidence_summary": json.dumps(safe_json_value(supporting), indent=2, ensure_ascii=False),
            "reason": None if len(grouped_series) > 0 else "No valid job categories were available."
        }

    # -----------------------------------------------------
    # GROUPED MEAN EVIDENCE (binary or arbitrary categories)
    # -----------------------------------------------------
    if (
        isinstance(primary_result, dict)
        and str(primary_result.get("analysis_type", "")).lower()
        == "grouped_mean_analysis"
    ):
        group_results = primary_result.get("group_results", {})
        numeric_types = (int, float, np.integer, np.floating)
        valid_groups = {}
        if isinstance(group_results, dict):
            for group_name, group_data in group_results.items():
                if not isinstance(group_data, dict):
                    continue
                mean_value = group_data.get("mean_value", group_data.get("mean"))
                row_count = group_data.get("row_count")
                valid_count = group_data.get(
                    "valid_value_count",
                    group_data.get("valid_count", row_count),
                )
                mean_is_valid = (
                    isinstance(mean_value, numeric_types)
                    and not isinstance(mean_value, (bool, np.bool_))
                    and np.isfinite(float(mean_value))
                )
                counts_are_valid = (
                    isinstance(row_count, (int, np.integer))
                    and not isinstance(row_count, (bool, np.bool_))
                    and int(row_count) >= 0
                    and isinstance(valid_count, (int, np.integer))
                    and not isinstance(valid_count, (bool, np.bool_))
                    and int(valid_count) >= 0
                )
                if mean_is_valid and counts_are_valid:
                    valid_groups[str(group_name)] = group_data

        metric = primary_result.get("metric_column")
        group_column = primary_result.get("group_column")
        is_binary_comparison = (
            isinstance(group_results, dict)
            and "no" in group_results
            and "yes" in group_results
        )
        difference = primary_result.get("mean_difference_yes_minus_no")
        if is_binary_comparison and not (
            isinstance(difference, numeric_types)
            and not isinstance(difference, (bool, np.bool_))
            and np.isfinite(float(difference))
        ):
            no_group = group_results.get("no", {})
            yes_group = group_results.get("yes", {})
            try:
                difference = (
                    float(yes_group.get("mean_value", yes_group.get("mean")))
                    - float(no_group.get("mean_value", no_group.get("mean")))
                )
            except (TypeError, ValueError, OverflowError, AttributeError):
                difference = None

        valid_result = (
            isinstance(metric, str)
            and bool(metric.strip())
            and isinstance(group_column, str)
            and bool(group_column.strip())
            and bool(valid_groups)
            and len(valid_groups) == len(group_results)
            and (
                not is_binary_comparison
                or (
                    "no" in valid_groups
                    and "yes" in valid_groups
                    and difference is not None
                )
            )
        )
        evidence_payload = safe_json_value(primary_result)
        return {
            "status": "PASS" if valid_result else "FAIL",
            "analysis_type": "grouped_mean_analysis",
            "metric_column": metric,
            "group_column": group_column,
            "group_results": safe_json_value(group_results),
            "mean_difference_yes_minus_no": (
                float(difference)
                if isinstance(difference, numeric_types)
                and not isinstance(difference, (bool, np.bool_))
                and np.isfinite(float(difference))
                else None
            ),
            "supporting_result": evidence_payload,
            "result": evidence_payload,
            "evidence_summary": json.dumps(
                evidence_payload, indent=2, ensure_ascii=False
            ),
            "reason": None if valid_result else (
                "Grouped mean result is missing a valid metric, group column, "
                "finite group means, or required group counts."
            ),
        }

    # -----------------------------------------------------
    # COMPARISON
    # -----------------------------------------------------

    if (
        "comparison" in analysis_type
        or
        "ranking" in analysis_type
    ):

        if grouped_series is None:

            return {
                "status": "FAIL",
                "analysis_type": "grouped_analysis",
                "reason": (
                    "Grouped result could not be reconstructed."
                )
            }

        return create_grouped_evidence(
            grouped_series,
            "Region",
            "Total Profit",
            "sum"
        )

    # -----------------------------------------------------
    # RELATIONSHIP
    # -----------------------------------------------------

    if (
        "relationship" in analysis_type
        or
        "correlation" in analysis_type
    ):

        return create_correlation_evidence(
            result=primary_result,
            x_column="Units Sold",
            y_column="Total Profit",
            method="Pearson"
        )

    # -----------------------------------------------------
    # TREND
    # -----------------------------------------------------

    if (
        "trend" in analysis_type
        or
        "time_series" in analysis_type
        or
        "timeseries" in analysis_type
    ):

        return create_trend_evidence(
            result=primary_result,
            metric="Total Revenue",
            time_label="Order Date"
        )

    # -----------------------------------------------------
    # GENERAL / DESCRIPTIVE SUMMARY
    # -----------------------------------------------------
    # This means evidence collection completed successfully.
    # The mapper separately checks whether the insight is
    # supported by these collected results.

    execution_steps = []

    if isinstance(execution_results, dict):
        raw_steps = execution_results.get("steps", [])

        if isinstance(raw_steps, list):
            for step in raw_steps:
                if not isinstance(step, dict):
                    continue

                step_result = step.get("final_result")

                if step_result is None:
                    step_result = step.get("result")

                execution_steps.append({
                    "step_number": step.get("step_number"),
                    "step_name": step.get("step_name", step.get("description")),
                    "status": step.get("status"),
                    "execution_status": step.get("execution_status"),
                    "validation_status": step.get("validation_status"),
                    "result": safe_json_value(step_result)
                })

    try:
        dataset_facts = build_dataset_facts(dataset_path) if dataset_path else None
    except Exception as error:
        dataset_facts = {"validation_error": str(error)}

    supporting_result = {
        "primary_result": safe_json_value(primary_result),
        "grouped_series": safe_json_value(grouped_series),
        "execution_steps": execution_steps,
        "dataset_facts": safe_json_value(dataset_facts),
        "data_profile": safe_json_value(data_profile),
    }

    evidence_payload = {
        "analysis_type": analysis_type or "general_analysis",
        "primary_result": safe_json_value(primary_result),
        "grouped_series": safe_json_value(grouped_series),
        "execution_steps": execution_steps,
        "dataset_facts": safe_json_value(dataset_facts),
    }

    return {
        "status": "PASS" if (primary_result is not None or execution_steps) else "FAIL",
        "analysis_type": analysis_type or "general_analysis",
        "result": safe_json_value(primary_result),
        "supporting_result": supporting_result,
        "evidence_summary": json.dumps(
            evidence_payload,
            indent=2,
            ensure_ascii=False,
            default=str
        ),
        "reason": None if (primary_result is not None or execution_steps) else (
            "No primary result or execution-step results were available."
        )
    }


def _describe_binary_group(group_column, group_value):
    """Describe yes/no groups using the actual grouping column, not unrelated labels."""
    column = str(group_column or "").strip().lower().replace("_", " ")
    value = str(group_value or "").strip().lower()

    labels = {
        "housing": {
            "yes": "customers with housing loans",
            "no": "customers without housing loans",
        },
        "loan": {
            "yes": "customers with personal loans",
            "no": "customers without personal loans",
        },
        "y": {
            "yes": "customers who subscribed",
            "no": "customers who did not subscribe",
        },
        "conversion": {
            "yes": "converted customers",
            "no": "non-converted customers",
            "1": "converted customers",
            "0": "non-converted customers",
        },
    }
    if value in labels.get(column, {}):
        return labels[column][value]
    return f"customers in the {column or 'requested'} = {value or 'unknown'} group"


# =========================================================
# INSIGHT GENERATION
# =========================================================


def build_clarification_request(task_analysis, data_profile):
    """
    Return a user-facing explanation when the question cannot be run faithfully, else None.

    Cases: (1) the analytical specification has unresolved issues, e.g. a
    categorical filter that cannot be executed; (2) the question names a metric
    that is not a column of the dataset. Questions routed to a dedicated
    analysis (analysis_intent) are left to that analysis.
    """
    if not isinstance(task_analysis, dict):
        return None
    columns = data_profile.get("columns", {}) if isinstance(data_profile, dict) else {}
    numeric = [c for c, i in columns.items() if i.get("role") in ("continuous", "numeric", "integer", "float")]
    categorical = [c for c, i in columns.items() if i.get("role") in ("categorical", "category")]

    problems = []
    spec = task_analysis.get("analytical_spec")

    # Open-ended outcome questions ("key factors affecting conversion", "recommendations
    # from conversion patterns") need one specific attribute to compare. They used to
    # run into a failing, unvalidated path; ask which attribute instead.
    original_question = str((spec or {}).get("original_question") or "").casefold() if isinstance(spec, dict) else ""
    open_ended = (
        re.search(r"\b(factors?|drivers?|patterns?|influenc\w*|affect\w*|predict\w*|recommend\w*|insights?|explain\w*)\b", original_question)
        and re.search(r"\b(conversion|convert\w*|subscri\w*)\b", original_question)
        and not re.search(r"\bby\b|\bper\b|\bgrouped\b", original_question)
    )
    if open_ended:
        base_categorical = [c for c in categorical if not any(n in c for n in numeric) and c not in ("y", "conversion")][:3]
        lines = ["This is an open-ended question, so I did not guess which attribute to analyse:",
                 "  - 'Factors/patterns/recommendations' for conversion needs a specific attribute to compare."]
        suggestions = [f"What is the conversion rate by {c}?" for c in base_categorical]
        if numeric:
            suggestions.append(f"Compare average {'balance' if 'balance' in numeric else numeric[0]} "
                               "between customers who subscribed and those who did not.")
        if suggestions:
            lines.append("Questions I can answer with your columns, for example:")
            lines += [f"  - {s}" for s in suggestions[:4]]
        return "\n".join(lines)

    spec_issues = []
    if isinstance(spec, dict) and str(spec.get("status", "")).upper() == "REVIEW":
        spec_issues = [str(i) for i in (spec.get("issues") or []) if str(i).strip()]
    categorical_filter_issue = any("not supported yet" in issue for issue in spec_issues)
    if spec_issues and (categorical_filter_issue or not task_analysis.get("analysis_intent")):
        problems.extend(spec_issues)
        if any("No numeric column" in issue for issue in spec_issues):
            problems.append(
                f"Numeric columns available: {', '.join(numeric) if numeric else 'none'}."
            )
    elif task_analysis.get("unsupported_metric") and not task_analysis.get("analysis_intent"):
        phrase = task_analysis.get("requested_metric")
        problems.append(
            (f"'{phrase}' does not match any column in the dataset. " if phrase
             else "The measure in your question does not match any column in the dataset. ")
            + f"Numeric columns available: {', '.join(numeric) if numeric else 'none'}."
        )
    if not problems:
        return None

    # Example questions built from real, base (non-derived) columns, using the
    # measure the user actually mentioned when there is one.
    base_categorical = [c for c in categorical if not any(n in c for n in numeric) and c != "y"] or categorical
    mentioned = [c for c in ((spec or {}).get("mentioned_columns") or []) if c in numeric] if isinstance(spec, dict) else []
    focus_metric = (mentioned or numeric or [None])[0]
    other_numeric = [c for c in numeric if c != focus_metric]
    examples = []
    named_categorical = [c for c in ((spec or {}).get("mentioned_columns") or []) if c in categorical] if isinstance(spec, dict) else []
    if named_categorical:
        examples.append(f"How many customers are there in each {named_categorical[0]} category?")
        examples.append(f"What is the distribution of customers by {named_categorical[0]}?")
    if focus_metric and base_categorical:
        examples.append(f"What is the average {focus_metric} by {base_categorical[0]}?")
        examples.append(f"Which {base_categorical[0]} has the highest average {focus_metric}?")
    if focus_metric and other_numeric:
        examples.append(f"Is there a correlation between {focus_metric} and {other_numeric[0]}?")
    lines = ["I could not run this question as written, so I did not guess:"]
    lines += [f"  - {problem}" for problem in dict.fromkeys(problems)]
    if examples:
        lines.append("Questions I can answer with your columns, for example:")
        lines += [f"  - {example}" for example in examples[:3]]
    return "\n".join(lines)


_NO_ANSWER_MARKERS = (
    "could not be safely converted",
    "a structured result was generated",
    "could not be summarized",
    "cannot be summarized",
    "no valid ",
)


def insight_states_no_answer(insight):
    """True when the insight is a fallback sentence that answers nothing."""
    text = str(insight or "").strip().casefold()
    return not text or any(marker in text[:160] for marker in _NO_ANSWER_MARKERS)


def downgrade_if_no_answer(insight, insight_validation):
    """
    A validator that PASSes a sentence saying "no insight could be produced" is
    vacuous: the result must be REVIEW, not PASS, so reports never present an
    unanswered question as reliable.
    """
    if not insight_states_no_answer(insight):
        return insight_validation
    base = dict(insight_validation) if isinstance(insight_validation, dict) else {}
    status = str(base.get("overall", base.get("status", ""))).upper()
    if status in {"FAIL", "REVIEW"}:
        return insight_validation
    base.update({
        "overall": "REVIEW", "status": "REVIEW", "pass": False,
        "reason": "The generated insight does not state an answer to the question; "
                  "the underlying result could not be converted into a validated statement.",
    })
    return base


def _trace_status(value):
    """PASS/FAIL/REVIEW/... for a validation dict, string or bool."""
    if isinstance(value, dict):
        for key in ("overall", "status", "overall_status", "validation_status"):
            if value.get(key) is not None:
                return str(value[key]).strip().upper()
        if value.get("pass") is True:
            return "PASS"
        if value.get("pass") is False:
            return "FAIL"
        return "REVIEW"
    if isinstance(value, str):
        return value.strip().upper() or "REVIEW"
    return "PASS" if value is True else "FAIL" if value is False else "REVIEW"


def build_dataset_overview(data_profile, data_path, display_name=None):
    """Dataset facts for the report header (stages: upload, loading, profiling)."""
    profile = data_profile if isinstance(data_profile, dict) else {}
    columns = profile.get("columns", {}) if isinstance(profile.get("columns"), dict) else {}
    roles = {}
    for info in columns.values():
        role = str((info or {}).get("role") or "unknown")
        roles[role] = roles.get(role, 0) + 1
    missing = 0
    for info in columns.values():
        try:
            missing += int((info or {}).get("null_count") or 0)
        except (TypeError, ValueError):
            pass
    return {
        "file": display_name or os.path.basename(str(data_path)),
        "rows": profile.get("n_rows"),
        "columns": profile.get("n_columns", len(columns)),
        "missing_cells": missing,
        "column_types": ", ".join(f"{role} {count}" for role, count in sorted(roles.items())),
        "identifier_columns": ", ".join(c for c, i in columns.items() if (i or {}).get("role") == "identifier"),
    }


def build_pipeline_trace(
    data_profile, data_path, user_question, task_analysis, execution_results,
    aggregated_result, primary_result, analysis_type, insight_validation,
    mapped_evidence, recommendation_validation, ollama_rewrite_status,
    display_name=None,
):
    """
    One entry per workflow stage with what happened in THIS run, replacing the
    static diagram that looked identical for every question.
    """
    overview = build_dataset_overview(data_profile, data_path, display_name)
    spec = task_analysis.get("analytical_spec", {}) if isinstance(task_analysis, dict) else {}
    steps = execution_results.get("steps", []) if isinstance(execution_results, dict) else []
    reliability = (aggregated_result or {}).get("reliability", {}) if isinstance(aggregated_result, dict) else {}
    total_attempts = reliability.get("total_attempts")
    corrections = reliability.get("correction_steps")
    method = str((primary_result or {}).get("method", "")) if isinstance(primary_result, dict) else ""
    isolation = "isolation" in method.lower()

    def csv(value):
        return ", ".join(str(v) for v in (value or [])) or "none"

    if total_attempts is None:
        self_correction = ("NOT REPORTED", "")
    elif corrections:
        self_correction = ("USED", f"{corrections} step(s) needed a corrected re-execution ({total_attempts} attempts in total)")
    else:
        self_correction = ("NOT NEEDED", f"every step passed on its first attempt ({total_attempts} attempt(s) for {len(steps)} step(s))")

    mapping_detail = ""
    if isinstance(mapped_evidence, dict) and mapped_evidence.get("traceability_status"):
        mapping_detail = f"traceability {mapped_evidence.get('traceability_status')}"

    return [
        ("Dataset Upload", "PASS", f"{overview['file']} ({overview['rows']:,} rows x {overview['columns']} columns)"
         if isinstance(overview.get("rows"), int) else overview["file"]),
        ("Data Loading and Preprocessing", "PASS",
         f"loaded without row removal; missing cells: {overview['missing_cells']:,}"),
        ("Data Profiling", "PASS", f"column types: {overview['column_types'] or 'n/a'}"),
        ("Natural Language Query", "RECEIVED", str(user_question)),
        ("Task Analysis and EDA Planning", _trace_status(spec.get("status", "READY")),
         f"task={task_analysis.get('task_type') if isinstance(task_analysis, dict) else None}; "
         f"operations=[{csv(spec.get('operations'))}]; metrics=[{csv(spec.get('metrics'))}]; "
         f"group_by=[{csv(spec.get('group_by'))}]; filters={len(spec.get('filters') or [])}; "
         f"plan steps={len(steps)}"),
        ("Python Code Generation and Execution", _trace_status(
            execution_results.get("overall_status") if isinstance(execution_results, dict) else None),
         f"{sum(1 for s in steps if str(s.get('status', '')).upper() == 'PASS')} of {len(steps)} step(s) succeeded"),
        ("Result Aggregation", _trace_status(
            (aggregated_result or {}).get("overall_status", (aggregated_result or {}).get("status"))
            if isinstance(aggregated_result, dict) else None), f"analysis type: {analysis_type}"),
        ("Independent Validation", _trace_status(
            (execution_results or {}).get("analytical_result_validation")
            if isinstance(execution_results, dict) else None),
         "results recomputed independently from the source data and compared"),
        ("Self-Correction and Re-execution", self_correction[0], self_correction[1]),
        ("Insight Generation and Validation", _trace_status(insight_validation), ""),
        ("Evidence Mapping", _trace_status(mapped_evidence), mapping_detail),
        ("Ollama-Based Insight Rewriting (optional)", ollama_rewrite_status.partition(" (")[0],
         ollama_rewrite_status.partition(" (")[2].rstrip(")")),
        ("Isolation Forest Anomaly Detection", "EXECUTED" if isolation else "NOT REQUESTED",
         (f"features: {csv(primary_result.get('feature_columns'))}; contamination setting: "
          f"{primary_result.get('contamination')}") if isolation and isinstance(primary_result, dict)
         else "the question did not ask for Isolation Forest"),
        ("Recommendation Generation and Validation", _trace_status(recommendation_validation), ""),
        ("Final EDA Report", "GENERATED", ""),
    ]


def generate_pipeline_insight(
    user_question,
    primary_result,
    grouped_series,
    task_analysis,
    dataset_path=None
):

    task_type = str(
        task_analysis.get(
            "task_type",
            ""
        )
    ).lower()

    # Grouped counts / top-N by customer count: answer from the validated counts.
    special_result = extract_special_result(primary_result)
    if special_result is not None:
        return special_insight(special_result, task_analysis)

    # IQR insight is generated from the actual univariate bounds/count result.
    iqr_result = _find_iqr_result(primary_result)
    if iqr_result is not None:
        metric = iqr_result.get("metric_column") or iqr_result.get("column") or iqr_result.get("metric")
        lower = iqr_result.get("lower_bound")
        upper = iqr_result.get("upper_bound")
        count = iqr_result.get("outlier_count", iqr_result.get("anomaly_count"))
        valid_count = iqr_result.get("valid_count", iqr_result.get("valid_value_count"))
        try:
            lower, upper = float(lower), float(upper)
            valid_bounds = np.isfinite(lower) and np.isfinite(upper) and lower <= upper
        except (TypeError, ValueError, OverflowError):
            valid_bounds = False
        if (str(iqr_result.get("status", "PASS")).upper() == "PASS" and isinstance(metric, str) and metric.strip()
                and valid_bounds and isinstance(count, (int, np.integer)) and not isinstance(count, (bool, np.bool_))
                and isinstance(valid_count, (int, np.integer)) and not isinstance(valid_count, (bool, np.bool_))):
            return (f"Using the IQR method, {int(count):,} outliers were identified in the {metric.strip()} column "
                    f"from {int(valid_count):,} valid numeric observations. Values below {lower:,.2f} "
                    f"or above {upper:,.2f} are flagged as outliers. These observations should be reviewed "
                    "in context; an IQR flag does not by itself indicate an error or explain its cause.")
        return "The IQR result cannot be summarized because its metric, bounds, or counts failed validation."

    # Isolation Forest insight is generated deterministically from its validated
    # result so that the numeric claims can be independently checked downstream.
    if (
        isinstance(primary_result, dict)
        and _normalized_detection_method(primary_result) == "isolationforest"
    ):
        summary = primary_result.get("summary", {})
        summary = summary if isinstance(summary, dict) else {}
        method = re.sub(
            r"[\s_-]+", "",
            str(primary_result.get("method", summary.get("algorithm", ""))).lower(),
        )
        row_count = primary_result.get("row_count", summary.get("total_rows"))
        anomaly_count = primary_result.get("anomaly_count", summary.get("anomaly_count"))
        normal_count = primary_result.get("normal_count", summary.get("normal_count"))
        anomaly_percentage = primary_result.get(
            "anomaly_percentage", summary.get("detected_anomaly_percentage")
        )
        feature_columns = primary_result.get(
            "feature_columns", summary.get("feature_columns", [])
        )
        try:
            percentage_value = float(anomaly_percentage)
        except (TypeError, ValueError, OverflowError):
            percentage_value = float("nan")
        valid = (
            method == "isolationforest"
            and str(primary_result.get("status", "")).upper() == "PASS"
            and isinstance(row_count, (int, np.integer))
            and not isinstance(row_count, (bool, np.bool_))
            and int(row_count) > 0
            and isinstance(anomaly_count, (int, np.integer))
            and not isinstance(anomaly_count, (bool, np.bool_))
            and int(anomaly_count) >= 0
            and isinstance(normal_count, (int, np.integer))
            and not isinstance(normal_count, (bool, np.bool_))
            and int(normal_count) >= 0
            and int(row_count) == int(anomaly_count) + int(normal_count)
            and np.isfinite(percentage_value)
            and 0 <= percentage_value <= 100
            and abs(
                percentage_value - (int(anomaly_count) / int(row_count) * 100)
            ) <= 0.02
            and isinstance(feature_columns, (list, tuple))
            and len(feature_columns) > 0
        )
        if not valid:
            return (
                "The Isolation Forest result cannot be summarized because its "
                "method, status, counts, percentage, or feature list failed validation."
            )
        feature_text = ", ".join(str(item) for item in feature_columns)
        return (
            f"Isolation Forest evaluated {int(row_count):,} records and detected "
            f"{int(anomaly_count):,} anomalies ({percentage_value:.2f}%). "
            f"{int(normal_count):,} records were classified as normal. "
            f"The model analyzed {len(feature_columns)} numeric "
            f"{'feature' if len(feature_columns) == 1 else 'features'}: {feature_text}. "
            + (
                f"The contamination setting ({float(primary_result['contamination']) * 100:.0f}%) "
                "fixes the share of records that is flagged, so the percentage reflects this "
                "setting, not a finding about the data. "
                if isinstance(primary_result, dict)
                and isinstance(primary_result.get("contamination"), (int, float))
                else ""
            )
            + "Flagged records should be reviewed in context; the anomaly score indicates "
            "relative unusualness, not the cause of the observation."
        )

    # Specification-driven grouped means: report each validated category mean.
    specification_grouped = extract_specification_grouped_mean(primary_result)
    if specification_grouped is not None:
        metric = specification_grouped["metric_column"]
        group_columns = specification_grouped["group_columns"]
        group_column_text = " and ".join(group_columns)
        groups = specification_grouped["group_results"]
        summary_text = "; ".join(
            f"{name}: {details['mean_value']:,.2f}"
            for name, details in sorted(
                groups.items(), key=lambda item: item[0].casefold()
            )
        )

        # "Which <group> has the highest/lowest average <metric>?" must state
        # the winner explicitly, not only list every group.
        answer_sentence = ""
        analytical_spec_for_answer = (
            task_analysis.get("analytical_spec")
            if isinstance(task_analysis, dict) else None
        )
        answer_focus = (
            analytical_spec_for_answer.get("answer_focus")
            if isinstance(analytical_spec_for_answer, dict) else None
        )
        if (
            isinstance(answer_focus, dict)
            and answer_focus.get("type") == "group_extrema"
            and len(groups) >= 2
        ):
            ranked_groups = sorted(
                groups.items(),
                key=lambda item: item[1]["mean_value"],
                reverse=True,
            )
            highest_name, highest_details = ranked_groups[0]
            lowest_name, lowest_details = ranked_groups[-1]
            highest_text = f"{highest_name} ({highest_details['mean_value']:,.2f})"
            lowest_text = f"{lowest_name} ({lowest_details['mean_value']:,.2f})"
            if answer_focus.get("direction") == "ascending":
                answer_sentence = (
                    f"Lowest average {metric}: {lowest_text}; "
                    f"highest average {metric}: {highest_text}. "
                )
            else:
                answer_sentence = (
                    f"Highest average {metric}: {highest_text}; "
                    f"lowest average {metric}: {lowest_text}. "
                )

        return (
            f"Average {metric} by {group_column_text} ({len(groups)} groups): "
            f"{summary_text}. {answer_sentence}"
            "These are descriptive group averages and do not establish causation."
        )

    # Scalar maximum/minimum: answer directly from the validated result.
    # Do not send a supported extrema query to the generic insight generator.
    if (
        isinstance(primary_result, dict)
        and str(primary_result.get("analysis_type", "")).lower()
        == "extrema_analysis"
    ):
        operation = str(primary_result.get("operation", "")).strip().lower()
        metric = primary_result.get("metric_column") or primary_result.get("metric")
        value = primary_result.get("value")
        count = primary_result.get(
            "valid_value_count",
            primary_result.get("valid_count", primary_result.get("count"))
        )
        valid_value = (
            isinstance(value, (int, float, np.integer, np.floating))
            and not isinstance(value, (bool, np.bool_))
            and np.isfinite(float(value))
        )
        if operation not in {"maximum", "minimum"}:
            return "The requested extreme cannot be summarized because the validated operation is missing or unsupported."
        if not isinstance(metric, str) or not metric.strip() or not valid_value:
            return (
                f"The {operation} cannot be summarized because the validated result "
                "does not contain a valid metric name and finite value."
            )
        extreme_word = "maximum" if operation == "maximum" else "minimum"
        count_text = (
            f", calculated from {int(count):,} valid numeric observations"
            if isinstance(count, (int, np.integer))
            and not isinstance(count, (bool, np.bool_))
            else ""
        )
        return (
            f"The {extreme_word} of the {metric.strip()} column is "
            f"{float(value):,.2f}{count_text}."
        )

    # Simple mean: use only the validated numeric result.
    if (
        isinstance(primary_result, dict)
        and str(primary_result.get("analysis_type", "")).lower()
        == "mean_analysis"
    ):
        metric = primary_result.get("metric_column") or primary_result.get("metric")
        value = primary_result.get(
            "mean", primary_result.get("mean_value", primary_result.get("average"))
        )
        count = primary_result.get(
            "valid_value_count",
            primary_result.get("valid_count", primary_result.get("count"))
        )

        if (
            isinstance(metric, str)
            and metric.strip()
            and isinstance(value, (int, float, np.integer, np.floating))
            and not isinstance(value, (bool, np.bool_))
            and np.isfinite(float(value))
        ):
            count_text = (
                f", calculated from {int(count):,} valid numeric observations"
                if isinstance(count, (int, np.integer))
                and not isinstance(count, (bool, np.bool_))
                else ""
            )
            return (
                f"The average of the {metric.strip()} column is "
                f"{float(value):,.2f}{count_text}."
            )

        return (
            "The average cannot be summarized because the validated result "
            "does not contain a valid metric name and finite mean."
        )

    if (
        isinstance(primary_result, dict)
        and str(primary_result.get("analysis_type", "")).lower()
        == "multi_metric_grouped_mean_analysis"
    ):
        return generate_multi_metric_grouped_mean_insight(primary_result)

    # Grouped means: support both binary comparisons (yes/no) and
    # arbitrary categories such as job, education, or region.
    if (
        isinstance(primary_result, dict)
        and str(primary_result.get("analysis_type", "")).lower()
        == "grouped_mean_analysis"
    ):
        groups = primary_result.get("group_results", {})
        metric_value = primary_result.get("metric_column")
        group_column_value = primary_result.get("group_column")
        metric = str(metric_value).strip() if metric_value is not None else ""
        group_column = (
            str(group_column_value).strip()
            if group_column_value is not None else "group"
        )

        if not metric:
            return (
                "The grouped averages cannot be summarized because the "
                "validated result does not identify the metric column."
            )
        if not isinstance(groups, dict) or not groups:
            return (
                f"The grouped averages for {metric} cannot be summarized "
                "because no group results were provided."
            )

        def _finite_number(value):
            try:
                number = float(value)
                return number if np.isfinite(number) else None
            except (TypeError, ValueError, OverflowError):
                return None

        # Preserve the dedicated yes/no comparison format when its validated
        # means and difference are present.
        if "no" in groups and "yes" in groups:
            no_group = groups.get("no", {})
            yes_group = groups.get("yes", {})
            no_mean = _finite_number(
                no_group.get("mean_value", no_group.get("mean"))
                if isinstance(no_group, dict) else None
            )
            yes_mean = _finite_number(
                yes_group.get("mean_value", yes_group.get("mean"))
                if isinstance(yes_group, dict) else None
            )
            difference = _finite_number(
                primary_result.get("mean_difference_yes_minus_no")
            )
            if difference is None and no_mean is not None and yes_mean is not None:
                difference = yes_mean - no_mean

            if no_mean is not None and yes_mean is not None and difference is not None:
                no_count = no_group.get("row_count", 0)
                yes_count = yes_group.get("row_count", 0)
                no_label = _describe_binary_group(group_column, "no")
                yes_label = _describe_binary_group(group_column, "yes")
                direction = "higher" if difference > 0 else (
                    "lower" if difference < 0 else "the same"
                )
                if difference == 0:
                    comparison = "The averages for the two groups are equal."
                else:
                    comparison = (
                        f"The average for {yes_label} is {abs(difference):,.2f} "
                        f"{direction} than the average for {no_label}."
                    )
                return (
                    f"Average {metric} was {no_mean:,.2f} for {no_label} "
                    f"and {yes_mean:,.2f} for {yes_label}. {comparison} "
                    f"(yes minus no = {difference:,.2f}; counts: "
                    f"{no_label}: {no_count:,}, {yes_label}: {yes_count:,}). "
                    "This is a descriptive comparison and does not establish causation."
                )

        # Generic grouped mean: report every valid category and its mean.
        summaries = []
        invalid_groups = []
        for group_name in sorted(groups, key=lambda item: str(item).casefold()):
            group_data = groups[group_name]
            if not isinstance(group_data, dict):
                invalid_groups.append(str(group_name))
                continue
            mean_value = _finite_number(
                group_data.get("mean_value", group_data.get("mean"))
            )
            if mean_value is None:
                invalid_groups.append(str(group_name))
                continue
            summaries.append((str(group_name), mean_value))

        if not summaries:
            return (
                f"The grouped averages for {metric} could not be summarized "
                "because no group contained a valid numeric mean."
            )

        summary_text = "; ".join(
            f"{name}: {mean:,.2f}" for name, mean in summaries
        )
        warning = (
            f" Some groups had missing or invalid means: {', '.join(invalid_groups)}."
            if invalid_groups else ""
        )
        return (
            f"Average {metric} by {group_column} "
            f"({len(summaries)} groups): {summary_text}.{warning} "
            "These are descriptive group averages and do not establish causation."
        )

    # A specific grouped conversion-rate question must not be routed to the
    # generic conversion insight, which reports only the overall dataset rate.
    if (
        grouped_series is not None
        and str(getattr(grouped_series, "name", "")).strip().lower() == "conversion rate (%)"
    ):
        counts = getattr(grouped_series, "attrs", {}).get("grouped_counts", {})
        if len(grouped_series) == 0:
            return "No valid job categories were available to compare conversion rates."

        top_category = str(grouped_series.index[0])
        top_rate = float(grouped_series.iloc[0])
        top_counts = counts.get(top_category, {}) if isinstance(counts, dict) else {}
        converted = top_counts.get("converted_count")
        total = top_counts.get("total_count")

        if len(grouped_series) > 1:
            second_category = str(grouped_series.index[1])
            second_rate = float(grouped_series.iloc[1])
            second_counts = counts.get(second_category, {}) if isinstance(counts, dict) else {}
            second_converted = second_counts.get("converted_count")
            second_total = second_counts.get("total_count")
            second_detail = (
                f" The second-highest category is {second_category} at {second_rate:.2f}%"
                + (f" ({second_converted:,} converted out of {second_total:,} customers)."
                   if second_converted is not None and second_total is not None else ".")
            )
        else:
            second_detail = ""

        top_detail = (
            f" ({converted:,} converted out of {total:,} customers)"
            if converted is not None and total is not None else ""
        )
        return (
            f"The {primary_result.get('group_column', 'requested')} group with the highest conversion rate is {top_category} at {top_rate:.2f}%{top_detail}."
            f"{second_detail} These are descriptive rates in the uploaded dataset and do not establish causation."
        )

    if (
        task_type == "comparison"
        and
        grouped_series is not None
    ):

        return generate_insight(
            user_question,
            grouped_series,
            task_analysis,
            dataset_path=dataset_path
        )

    return generate_insight(
        user_question,
        primary_result,
        task_analysis,
        dataset_path=dataset_path
    )


# =========================================================
# RECOMMENDATION ADAPTER
# =========================================================

def generate_pipeline_recommendation(
    user_question,
    analysis_type,
    primary_result,
    grouped_series,
    insight,
    task_analysis,
    evidence
):

    if str(analysis_type).lower() in SPECIAL_TYPES:
        special_rec = special_recommendation(evidence, task_analysis)
        if special_rec.get("status") == "PASS":
            return special_rec["recommendation"]
        return special_rec

    iqr_result = _find_iqr_result(primary_result)
    if iqr_result is not None:
        if not isinstance(evidence, dict) or str(evidence.get("status", "")).upper() != "PASS":
            return "Do not act on the IQR flags yet. Resolve the IQR evidence validation issues and rerun the analysis."
        metric = iqr_result.get("metric_column") or iqr_result.get("column") or iqr_result.get("metric") or "requested metric"
        count = iqr_result.get("outlier_count", iqr_result.get("anomaly_count", 0))
        lower = iqr_result.get("lower_bound")
        upper = iqr_result.get("upper_bound")
        return (f"Review the {int(count):,} records flagged outside the IQR bounds for {metric}. "
                f"Check values below {float(lower):,.2f} or above {float(upper):,.2f} against source records, "
                "data-entry rules, and business context. Do not automatically delete or relabel these observations; "
                "some may represent legitimate rare cases.")

    if (
        isinstance(primary_result, dict)
        and _normalized_detection_method(primary_result) == "isolationforest"
    ):
        if (
            str(primary_result.get("status", "")).upper() != "PASS"
            or not isinstance(evidence, dict)
            or str(evidence.get("status", "")).upper() != "PASS"
        ):
            return (
                "Do not act on the anomaly results yet. Resolve the Isolation Forest "
                "validation issues and rerun the analysis before reviewing flagged records."
            )
        anomaly_count = primary_result.get("anomaly_count", 0)
        sample_rows = primary_result.get("sample_anomalies")
        sample_count = len(sample_rows) if isinstance(sample_rows, pd.DataFrame) else 0
        sample_note = (
            f" The execution output includes a sample of {sample_count} flagged records; "
            "review the available sample and retrieve the full flagged set when needed."
            if sample_count else
            " Retrieve the flagged records from the execution output for detailed review."
        )
        return (
            f"Review the {int(anomaly_count):,} records flagged by Isolation Forest "
            "against their original values, data-entry rules, and business context. "
            "Prioritize records with lower decision-function scores because lower scores "
            "indicate more unusual observations. Check whether each flag reflects a "
            "data-quality issue, a legitimate rare event, or a pattern requiring further "
            "investigation; do not automatically delete or relabel flagged records."
            + (
                " With a single feature, extreme values often receive identical scores, "
                "so rank the flagged records by the feature value itself."
                if len(primary_result.get("feature_columns") or []) == 1 else ""
            )
            + sample_note
        )

    # Avoid sales-specific fallback wording (for example, "second-highest region")
    # for job-category conversion analysis.
    if (
        grouped_series is not None
        and str(getattr(grouped_series, "name", "")).strip().lower() == "conversion rate (%)"
        and len(grouped_series) > 0
    ):
        top_category = str(grouped_series.index[0])
        top_rate = float(grouped_series.iloc[0])
        counts = getattr(grouped_series, "attrs", {}).get("grouped_counts", {})
        top_counts = counts.get(top_category, {}) if isinstance(counts, dict) else {}
        converted = top_counts.get("converted_count")
        total = top_counts.get("total_count")
        count_detail = (
            f" ({converted:,} converted out of {total:,} customers)"
            if converted is not None and total is not None else ""
        )
        return (
            f"Consider further analysis of the {top_category} group for {primary_result.get('group_column', 'the requested factor')}, which has the highest observed "
            f"conversion rate at {top_rate:.2f}%{count_detail}. Compare sample sizes and customer characteristics "
            "before allocating resources; this descriptive association does not establish causation."
        )

    if (
        isinstance(primary_result, dict)
        and str(primary_result.get("analysis_type", "")).lower()
        == "multi_metric_grouped_mean_analysis"
    ):
        if primary_result.get("status") != "PASS":
            return (
                "Do not make decisions from this comparison yet. Validate all requested "
                "metric-specific group means and rerun the analysis."
            )
        return (
            "Review age, balance, and campaign differences together with group sizes, "
            "data quality, and other customer characteristics before deciding on "
            "targeting strategies. These descriptive comparisons do not establish causation."
        )

    if (
        isinstance(primary_result, dict)
        and str(primary_result.get("analysis_type", "")).lower()
        == "grouped_mean_analysis"
    ):
        groups = primary_result.get("group_results", {})
        metric = str(
            primary_result.get("metric_column")
            or primary_result.get("metric")
            or "requested metric"
        ).strip()
        if not isinstance(groups, dict) or not groups:
            return (
                f"Validate the grouped averages for {metric} before using them "
                "to guide decisions; no group-level results were available."
            )

        def _recommendation_mean(group):
            if not isinstance(group, dict):
                return None
            for key in ("mean_value", "mean", f"mean_{metric}", "average"):
                value = group.get(key)
                try:
                    number = float(value)
                    if np.isfinite(number):
                        return number
                except (TypeError, ValueError, OverflowError):
                    continue
            return None

        # Keep the established binary recommendation for genuine yes/no groups.
        if "no" in groups and "yes" in groups:
            no_mean = _recommendation_mean(groups.get("no"))
            yes_mean = _recommendation_mean(groups.get("yes"))
            if no_mean is not None and yes_mean is not None:
                difference = yes_mean - no_mean
                group_column = str(primary_result.get("group_column", "group"))
                no_label = _describe_binary_group(group_column, "no")
                yes_label = _describe_binary_group(group_column, "yes")
                direction = (
                    "higher" if difference > 0
                    else "lower" if difference < 0
                    else "the same"
                )
                if difference == 0:
                    comparison = f"both groups have the same average {metric} ({yes_mean:.2f})"
                else:
                    comparison = (
                        f"the average for {yes_label} is {direction} by "
                        f"{abs(difference):.2f} compared with {no_label}"
                    )
                return (
                    f"The average {metric} for {no_label} is {no_mean:.2f}, "
                    f"and for {yes_label} it is {yes_mean:.2f}; {comparison}. "
                    "Review group sizes and other customer characteristics before "
                    "making decisions. This descriptive comparison does not establish causation."
                )

        valid_means = [
            (str(name), _recommendation_mean(group), group)
            for name, group in groups.items()
        ]
        valid_means = [item for item in valid_means if item[1] is not None]
        if len(valid_means) != len(groups) or not valid_means:
            return (
                f"Some grouped averages for {metric} are missing or non-numeric. "
                "Validate the group-level results before using them for decisions."
            )

        highest = max(valid_means, key=lambda item: item[1])
        lowest = min(valid_means, key=lambda item: item[1])
        group_column = str(primary_result.get("group_column") or "category")
        return (
            f"For {metric} grouped by {group_column}, the highest average is "
            f"{highest[1]:,.2f} for '{highest[0]}', and the lowest average is "
            f"{lowest[1]:,.2f} for '{lowest[0]}'. Review each group's sample size "
            "and relevant customer characteristics before making decisions. "
            "These descriptive averages do not establish causation."
        )

    # --------------------------------------------------------
    # Specification-driven grouped-mean recommendation.
    # Supports one or more grouping columns and one metric.
    # --------------------------------------------------------

    specification_grouped = extract_specification_grouped_mean(primary_result)
    if specification_grouped is not None:
        if not isinstance(evidence, dict) or str(evidence.get("status", "")).upper() != "PASS":
            return (
                "Do not make decisions from these grouped averages yet. "
                "Resolve the evidence validation issues and rerun the analysis."
            )

        metric = specification_grouped["metric_column"]
        group_columns = specification_grouped["group_columns"]
        groups = specification_grouped["group_results"]
        valid_means = [
            (label, float(details["mean_value"]))
            for label, details in groups.items()
            if isinstance(details, dict)
            and isinstance(details.get("mean_value"), (int, float, np.integer, np.floating))
            and np.isfinite(float(details["mean_value"]))
        ]
        if len(valid_means) != len(groups) or not valid_means:
            return (
                f"Do not make decisions from this analysis yet. "
                f"One or more grouped means for {metric} are invalid."
            )

        highest = max(valid_means, key=lambda item: item[1])
        lowest = min(valid_means, key=lambda item: item[1])
        grouping_text = " and ".join(group_columns)

        def _display_label(label):
            # "no" -> "housing=no": a bare yes/no says nothing about the column.
            return f"{group_columns[0]}={label}" if len(group_columns) == 1 and "=" not in str(label) else str(label)

        counts = {
            label: details.get("count")
            for label, details in groups.items()
            if isinstance(details, dict) and isinstance(details.get("count"), int)
        }
        if len(counts) == len(groups) and counts and len(groups) <= 2:
            # Two groups: the range sentence and the small-group caution add nothing.
            size_text = (
                f" The highest-average group has {counts[highest[0]]:,} records and the "
                f"lowest-average group has {counts[lowest[0]]:,} records."
            )
            follow_up = " Investigate whether the differences are meaningful before making decisions."
        elif len(counts) == len(groups) and counts:
            size_text = (
                f" The highest-average group has {counts[highest[0]]:,} records and the "
                f"lowest-average group has {counts[lowest[0]]:,} records; group sizes range "
                f"from {min(counts.values()):,} to {max(counts.values()):,} records. "
                "Treat averages from groups with few records cautiously."
            )
            follow_up = " Investigate whether the differences are meaningful before making decisions."
        else:
            size_text = ""
            follow_up = (
                " Review the number of records in each group and investigate whether "
                "the differences are meaningful before making decisions."
            )
        return (
            f"For {metric} grouped by {grouping_text}, the highest average is "
            f"{highest[1]:,.2f} for {_display_label(highest[0])}, and the lowest average is "
            f"{lowest[1]:,.2f} for {_display_label(lowest[0])}."
            + size_text + follow_up
            + " These descriptive averages do not establish causation."
        )

    # --------------------------------------------------------
    # Specification-driven multi-metric grouped-mean recommendation.
    # --------------------------------------------------------

    if (
        isinstance(primary_result, dict)
        and str(primary_result.get("analysis_type", "")).lower()
        == "specification_driven"
        and str(primary_result.get("status", "")).upper() == "PASS"
        and isinstance(primary_result.get("operations"), (list, tuple))
        and "mean" in [
            str(operation).strip().lower()
            for operation in primary_result.get("operations", [])
        ]
        and isinstance(primary_result.get("metrics"), (list, tuple))
        and len(primary_result.get("metrics", [])) >= 2
        and isinstance(primary_result.get("group_by"), (list, tuple))
        and len(primary_result.get("group_by", [])) == 1
    ):
        operation_results = primary_result.get("operation_results", {})
        rows = (
            operation_results.get("aggregations")
            if isinstance(operation_results, dict)
            else None
        )

        metrics = primary_result.get("metrics", [])
        group_column = str(primary_result["group_by"][0]).strip()

        if not isinstance(rows, list) or not rows or not group_column:
            return (
                "Do not make decisions from this analysis yet. "
                "The multi-metric grouped aggregation rows are unavailable."
            )

        metric_recommendations = []

        for metric in metrics:
            metric = str(metric).strip()
            mean_key = f"{metric}__mean"
            valid_values = []

            for row in rows:
                if not isinstance(row, dict):
                    valid_values = []
                    break

                group_name = str(row.get(group_column, "")).strip()
                value = row.get(mean_key)

                try:
                    numeric_value = float(value)
                except (TypeError, ValueError, OverflowError):
                    valid_values = []
                    break

                if (
                    not group_name
                    or not np.isfinite(numeric_value)
                ):
                    valid_values = []
                    break

                valid_values.append((group_name, numeric_value))

            if len(valid_values) != len(rows) or not valid_values:
                return (
                    f"Do not make decisions from this analysis yet. "
                    f"The grouped means for {metric} are incomplete or invalid."
                )

            highest = max(valid_values, key=lambda item: item[1])
            lowest = min(valid_values, key=lambda item: item[1])

            metric_recommendations.append(
                f"{metric}: highest average in {highest[0]} "
                f"({highest[1]:,.2f}); lowest average in {lowest[0]} "
                f"({lowest[1]:,.2f})"
            )

        return (
            "Review the differences across the requested metrics and "
            "investigate the groups with the highest and lowest averages: "
            + "; ".join(metric_recommendations)
            + ". Compare group sizes, data quality, and relevant customer "
            "characteristics before making business decisions. "
            "These descriptive averages do not establish causation."
        )
    try:

        signature = inspect.signature(
            generate_recommendation
        )

        parameters = signature.parameters

        comparison_result = (
            grouped_series
            if grouped_series is not None
            else primary_result
        )

        values = {

            "user_question": user_question,

            "question": user_question,

            "analysis_type": analysis_type,

            "task_type": analysis_type,

            "result": comparison_result,

            "primary_result": primary_result,

            "grouped_result": grouped_series,

            "insight": insight,

            "task_analysis": task_analysis,

            "evidence": evidence
        }

        kwargs = {}

        for parameter_name in parameters:

            if parameter_name in values:

                kwargs[
                    parameter_name
                ] = values[
                    parameter_name
                ]

        return generate_recommendation(
            **kwargs
        )

    except Exception as error:

        print(
            "\nRecommendation adapter error:"
        )

        print(
            error
        )

        return (
            "Based on the validated analysis, "
            "the identified result should be investigated "
            "further before making business decisions."
        )


# =========================================================
# MAIN
# =========================================================

def main(dataset_path=None, user_question=None, test_fault=False, dataset_name=None):

    # Use the dataset supplied by Streamlit when present; retain the default
    # sales dataset for standalone command-line runs.
    if dataset_path:
        DATA_PATH = os.path.abspath(str(dataset_path))
    else:
        DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "customer.csv")

    print_section(
        "SELF-VALIDATING EDA AGENT"
    )

    print(
        "Dataset:",
        DATA_PATH
    )

    # =====================================================
    # STEP 1 - DATA PROFILING
    # =====================================================

    print_section(
        "STEP 1 - DATA PROFILING"
    )

    data_profile = profile_dataset(
        DATA_PATH
    )

    dataset_columns = extract_dataset_columns(
        data_profile
    )

    print(
        "Data profiling completed."
    )

    print(
        "\nDataset Columns:"
    )

    for column in dataset_columns:

        print(
            "-",
            column
        )

    # =====================================================
    # STEP 2 - TASK ANALYSIS
    # =====================================================

    print_section(
        "STEP 2 - TASK ANALYSIS"
    )

    if user_question is None:
        user_question = input("\nEnter your analysis question: ").strip()
    else:
        user_question = str(user_question).strip()

    if not user_question:
        raise ValueError("Please enter a non-empty analysis question.")

    print(
        "User Question:"
    )

    print(
        user_question
    )

    task_analysis = analyze_task_mock(
        user_question,
        data_profile
    )

    print(
        "\nTask Analysis:"
    )

    print_json(
        task_analysis
    )

    # Stop before planning or execution when the user's claim is contradictory.
    # The task analyzer has already identified that clarification is required.
    if task_analysis.get("ambiguous_direction_claim"):
        clarification = (
            "Your question contains contradictory direction terms: 'higher' and 'lower'. "
            "Please clarify whether you want to check if subscribers have a higher or "
            "lower average balance than non-subscribers."
        )
        print_section("CLARIFICATION REQUIRED")
        print(clarification)
        print("No EDA plan was generated and no analysis was executed.")
        return None

    # Stop when the question cannot be executed faithfully. Previously such
    # questions ran anyway, silently dropped the part they could not handle
    # (e.g. a categorical filter) and returned a confident but wrong answer,
    # or cascaded into FAIL at every later stage with an empty recommendation.
    clarification_request = build_clarification_request(task_analysis, data_profile)
    if clarification_request:
        print_section("CLARIFICATION REQUIRED")
        print(clarification_request)
        print("No EDA plan was generated and no analysis was executed.")
        try:
            report_text = (
                "============================================================\n"
                "SELF-VALIDATING EDA AGENT\nFINAL ANALYSIS REPORT\n"
                "============================================================\n\n"
                f"Generated On:\n{__import__('datetime').datetime.now():%Y-%m-%d %H:%M:%S}\n\n"
                f"1. USER QUESTION\n------------------------------------------------------------\n{user_question}\n\n"
                "2. STATUS\n------------------------------------------------------------\n"
                "NEEDS CLARIFICATION (no analysis was run)\n"
                "Pipeline stopped at: Task Analysis and EDA Planning. Dataset upload, loading and "
                "profiling completed; no code was generated or executed.\n\n"
                f"3. WHAT IS NEEDED\n------------------------------------------------------------\n{clarification_request}\n\n"
                "============================================================\nEND OF REPORT\n"
                "============================================================\n"
            )
            save_report(report_text, os.path.join(OUTPUT_DIR, "final_eda_report.txt"))
        except Exception as exc:  # reporting must never hide the clarification
            print(f"(Could not save clarification report: {exc})")
        print_section("FINAL PIPELINE STATUS")
        print("OVERALL RESULT: NEEDS CLARIFICATION")
        return None

    # The question's answer focus (top N, ranking, rate ...) selects the validated
    # answer path for grouped results.
    set_answer_focus(task_analysis)

    # =====================================================
    # STEP 3 - EDA PLAN
    # =====================================================

    print_section(
        "STEP 3 - EDA PLAN"
    )

    # IMPORTANT: generate_plan_mock expects (task_type, data_profile, user_query).
    # Passing these arguments in the wrong order can produce an invalid/empty plan.
    analytical_spec = task_analysis.get("analytical_spec")

    plan = generate_plan_mock(
        task_analysis.get("task_type", "descriptive_summary"),
        data_profile,
        user_question,
        analytical_spec=analytical_spec
    )

    # Defensive fallback: never let a missing plan crash the pipeline.
    if not isinstance(plan, (list, tuple)) or not plan:
        print("WARNING: EDA planner returned no steps; using a safe descriptive plan.")
        plan = [
            "Report the dataset row count, column count, and column names.",
            "Calculate descriptive statistics for numeric business measures, excluding identifiers.",
            "Check missing values and duplicate rows.",
            "Summarize important categorical values and date ranges.",
        ]

    print(
        "Generated EDA Plan:"
    )

    for index, step in enumerate(
        plan,
        start=1
    ):

        print(
            f"{index}.",
            step
        )

    target_columns = normalize_target_columns(
        task_analysis.get(
            "target_columns",
            []
        )
    )

    print(
        "\nTarget Columns:"
    )

    print(
        target_columns
    )

    # =====================================================
    # STEP 4 - MULTI-STEP EXECUTION
    # =====================================================

    print_section(
        "STEP 4 - MULTI-STEP SELF-CORRECTING EDA EXECUTION"
    )

    step_executor = create_self_correcting_executor(
        filepath=DATA_PATH,
        data_profile=data_profile,
        dataset_columns=dataset_columns,
        target_columns=target_columns,
        test_fault=bool(test_fault),
        analytical_spec=analytical_spec
    )

    execution_results = execute_plan_steps(
        plan=plan,
        step_executor=step_executor
    )

    # Stage 3: independently verify specification-driven results against the source CSV.
    analytical_result_validation = validate_analytical_results(
        filepath=DATA_PATH,
        analytical_spec=analytical_spec,
        execution_results=execution_results,
    )
    execution_results["analytical_result_validation"] = analytical_result_validation

    print_section("STAGE 3 - INDEPENDENT ANALYTICAL RESULT VALIDATION")
    print("Validation Status:", analytical_result_validation.get("status"))
    print_json(analytical_result_validation)

    # =====================================================
    # EXECUTION SUMMARY
    # =====================================================

    print_section(
        "MULTI-STEP EXECUTION SUMMARY"
    )

    print(
        "Overall Status:",
        execution_results.get(
            "overall_status"
        )
    )

    print(
        "Total Steps:",
        execution_results.get(
            "total_steps"
        )
    )

    print(
        "Successful Steps:",
        execution_results.get(
            "successful_steps"
        )
    )

    print(
        "Failed Steps:",
        execution_results.get(
            "failed_steps"
        )
    )

    print(
        "Total Execution Attempts:",
        execution_results.get(
            "total_attempts"
        )
    )

    print(
        "\nIndividual Step Results:"
    )

    for step in execution_results.get(
        "steps",
        []
    ):

        print(
            f"Step {step.get('step_number')}: "
            f"{step.get('status')} | "
            f"Attempts: {step.get('total_attempts')} | "
            f"Execution: {step.get('execution_status')} | "
            f"Validation: {step.get('validation_status')}"
        )

    # =====================================================
    # STEP 5 - RESULT AGGREGATION
    # =====================================================

    print_section(
        "STEP 5 - RESULT AGGREGATION"
    )

    aggregated_result = aggregate_results(
        user_question=user_question,
        all_steps=execution_results.get(
            "steps",
            []
        ),
        task_analysis=task_analysis
    )

    print_aggregation_summary(
        aggregated_result
    )

    primary_result = get_primary_result(
        aggregated_result,
        execution_results
    )

    primary_result, execution_results = sanitize_identifier_statistics(
        primary_result,
        execution_results,
        DATA_PATH
    )

    # For explicit multi-metric grouped-mean questions, promote a combined result
    # so downstream insight, validation, evidence, and reporting see every metric.
    combined_multi_metric_result = build_multi_metric_grouped_mean_result(
        task_analysis,
        execution_results
    )
    if combined_multi_metric_result is not None:
        # The legacy combination only understands per-metric legacy steps. When
        # the executor already returned a valid specification-driven multi-metric
        # grouped mean, keep it instead of replacing it with an empty FAIL result.
        already_valid_spec_multi_metric = (
            isinstance(primary_result, dict)
            and str(primary_result.get("analysis_type", "")).lower() == "specification_driven"
            and str(primary_result.get("status", "")).upper() == "PASS"
            and "mean" in [str(op).lower() for op in primary_result.get("operations", [])]
            and len(primary_result.get("metrics", [])) >= 2
            and len(primary_result.get("group_by", [])) == 1
        )
        if (
            str(combined_multi_metric_result.get("status", "")).upper() == "PASS"
            or not already_valid_spec_multi_metric
        ):
            primary_result = combined_multi_metric_result

    print(
        "\nPrimary Validated Result:"
    )

    # IMPORTANT:
    # safe_json_value converts Pandas Period
    # into normal strings such as "2013-07".
    print_json(
        primary_result
    )

    # =====================================================
    # GROUPED SERIES
    # =====================================================

    # A validated grouped-mean comparison already carries its own group-level
    # evidence. Do not run the legacy Region -> Total Profit reconstruction for it.
    primary_analysis_type = (
        str(primary_result.get("analysis_type", "")).lower()
        if isinstance(primary_result, dict)
        else ""
    )
    analytical_spec_for_grouping = (
        task_analysis.get("analytical_spec")
        if isinstance(task_analysis, dict)
        else None
    )
    requested_group_by = (
        analytical_spec_for_grouping.get("group_by", [])
        if isinstance(analytical_spec_for_grouping, dict)
        else []
    )
    is_ungrouped_descriptive_summary = (
        str(task_analysis.get("task_type", "")).lower() == "descriptive_summary"
        and not requested_group_by
    )

    if (
        primary_analysis_type in {
            "grouped_mean_analysis",
            "multi_metric_grouped_mean_analysis",
            "anomaly_detection",
            "outlier_analysis",
        }
        or extract_specification_grouped_mean(primary_result) is not None
        or is_ungrouped_descriptive_summary
    ):
        grouped_series = None
    else:
        grouped_series = build_grouped_series(
            execution_results,
            DATA_PATH
        )

    # Calculate grouped conversion rates for explicitly requested factors.
    question_lower = str(user_question or "").lower()
    requested_group = None

    columns = set(pd.read_csv(DATA_PATH, nrows=0).columns)

    group_aliases = {
        "account balance": "balance",
        "balance": "balance",
        "job": "job",
        "education": "education",
        "marital": "marital",
        "housing": "housing",
        "loan": "loan",
        "month": "month",
        "contact": "contact",
        "poutcome": "poutcome",
    }

    for phrase in sorted(group_aliases, key=len, reverse=True):
        column = group_aliases[phrase]
        if phrase in question_lower and column in columns:
            requested_group = column
            break

    if (
        requested_group
        and ("conversion rate" in question_lower
             or "conversion rates" in question_lower)
    ):
        if requested_group == "balance":
            from src.bank_conversion_analysis import (
                calculate_conversion_rate_by_group,
            )

            grouped_series = calculate_conversion_rate_by_group(
                DATA_PATH,
                "balance",
                bins=[
                    -float("inf"), 0, 1000, 5000, 10000, float("inf")
                ],
                labels=[
                    "Negative", "0-1,000", "1,001-5,000",
                    "5,001-10,000", "Above 10,000"
                ],
            )["conversion_rate_percent"]

            # Preserve exact counts for report evidence.
            grouped_result = calculate_conversion_rate_by_group(
                DATA_PATH,
                "balance",
                bins=[
                    -float("inf"), 0, 1000, 5000, 10000, float("inf")
                ],
                labels=[
                    "Negative", "0-1,000", "1,001-5,000",
                    "5,001-10,000", "Above 10,000"
                ],
            )
            grouped_series.attrs["grouped_counts"] = {
                str(index): {
                    "converted_count": int(row["converted"]),
                    "total_count": int(row["customers"]),
                }
                for index, row in grouped_result.iterrows()
            }
            grouped_series = grouped_series.sort_values(ascending=False)
            grouped_series.name = "conversion rate (%)"

            top_category = str(grouped_series.index[0])
            top_counts = grouped_series.attrs["grouped_counts"][top_category]

            primary_result = {
                "status": "PASS",
                "analysis_type": "grouped_conversion_rate_by_balance",
                "group_column": "balance",
                "metric": "conversion_rate_percent",
                "top_category": top_category,
                "top_conversion_rate_percent": float(grouped_series.iloc[0]),
                "converted_count": top_counts["converted_count"],
                "total_count": top_counts["total_count"],
                "grouped_result": safe_json_value(grouped_series),
                "grouped_counts": safe_json_value(
                    grouped_series.attrs["grouped_counts"]
                ),
            }

        elif requested_group == "job":
            grouped_series = build_job_conversion_rate_series(DATA_PATH)

            if grouped_series is not None and len(grouped_series) > 0:
                counts = getattr(grouped_series, "attrs", {}).get(
                    "grouped_counts", {}
                )
                top_category = str(grouped_series.index[0])
                top_counts = counts.get(top_category, {})

                primary_result = {
                    "status": "PASS",
                    "analysis_type": "grouped_conversion_rate_by_job",
                    "group_column": "job",
                    "metric": "conversion_rate_percent",
                    "top_category": top_category,
                    "top_conversion_rate_percent": round(
                        float(grouped_series.iloc[0]), 4
                    ),
                    "converted_count": top_counts.get("converted_count"),
                    "total_count": top_counts.get("total_count"),
                    "grouped_result": safe_json_value(grouped_series),
                    "grouped_counts": safe_json_value(counts),
                }

    print("\nGrouped Series Used for Insight:")

    if grouped_series is not None:
        print(grouped_series)

    elif (
        (
            isinstance(primary_result, dict)
            and str(primary_result.get("analysis_type", "")).lower()
            in {
                "grouped_mean_analysis",
                "multi_metric_grouped_mean_analysis",
            }
            and primary_result.get("status") == "PASS"
        )
        or extract_specification_grouped_mean(primary_result) is not None
    ):
        print(
            "Not required: validated grouped results are available "
            "directly in primary_result."
        )

    else:
        print(
            "Grouped series is unavailable for this analysis."
        )

    # =====================================================
    # ANALYSIS TYPE
    # =====================================================

    result_analysis_type = (
        primary_result.get("analysis_type", "")
        if isinstance(primary_result, dict)
        else ""
    )

    special_result = extract_special_result(primary_result)
    if special_result is not None:
        analysis_type = special_result["analysis_type"]
    elif str(result_analysis_type).lower() in (
        "conversion_factor_analysis",
        "grouped_mean_analysis",
        "multi_metric_grouped_mean_analysis",
        "anomaly_detection",
        "outlier_analysis",
        "anomaly_summary",
    ):
        analysis_type = str(result_analysis_type).lower()
    else:
        analysis_type = (
            task_analysis.get(
                "task_type",
                "general_analysis"
            )
            if isinstance(task_analysis, dict)
            else "general_analysis"
        )

    # =====================================================
    # STEP 6 - INSIGHT GENERATION
    # =====================================================

    print_section(
        "STEP 6 - INSIGHT GENERATION"
    )

    insight = generate_pipeline_insight(
        user_question=user_question,
        primary_result=primary_result,
        grouped_series=grouped_series,
        task_analysis=task_analysis,
        dataset_path=DATA_PATH
    )

    print(
        "Generated Insight:"
    )

    print(
        insight
    )

    # =====================================================
    # STEP 7 - INSIGHT VALIDATION
    # =====================================================

    print_section(
        "STEP 7 - INSIGHT VALIDATION"
    )

    validation_result_data = (
        grouped_series
        if (
            str(analysis_type).lower() == "comparison"
            and grouped_series is not None
        )
        else primary_result
    )

    insight_validation = validate_insight(
    insight=insight,
    result=validation_result_data,
    analysis_type=analysis_type,
    task_analysis=task_analysis,
    user_question=user_question,
    data_profile=data_profile
    )

    if (
        isinstance(primary_result, dict)
        and str(primary_result.get("analysis_type", "")).lower()
        == "multi_metric_grouped_mean_analysis"
    ):
        base_insight_validation = insight_validation
        insight_validation = validate_multi_metric_grouped_mean_insight(
            insight=insight,
            combined_result=primary_result,
            base_validation=base_insight_validation,
        )
    elif (
        isinstance(primary_result, dict)
        and str(primary_result.get("analysis_type", "")).lower()
        == "specification_driven"
        and "mean" in [
            str(item).lower()
            for item in primary_result.get("operations", [])
        ]
        and len(primary_result.get("metrics", [])) >= 2
        and len(primary_result.get("group_by", [])) == 1
    ):
        base_insight_validation = insight_validation

        insight_validation = (
            validate_specification_multi_metric_grouped_mean_insight(
                insight=insight,
                primary_result=primary_result,
                base_validation=base_insight_validation,
            )
        )
    elif extract_specification_grouped_mean(primary_result) is not None:
        insight_validation = validate_specification_grouped_mean_insight(
            insight=insight,
            primary_result=primary_result,
        )
    # Dedicated aggregate/count insights are validated by their own validator,
    # whichever legacy validator ran above.
    if extract_special_result(primary_result) is not None:
        insight_validation = special_validate(
            insight, extract_special_result(primary_result), task_analysis
        )
    insight_validation = downgrade_if_no_answer(insight, insight_validation)
    print(
        "Insight Validation Result:"
    )

    if isinstance(
        insight_validation,
        dict
    ):

        print_json(
            insight_validation
        )

    else:

        print(
            insight_validation
        )

    # =====================================================
    # STEP 8 - EVIDENCE MAPPING
    # =====================================================

    print_section(
        "STEP 8 - EVIDENCE MAPPING"
    )

    evidence = create_evidence(
        analysis_type=analysis_type,
        primary_result=primary_result,
        grouped_series=grouped_series,
        execution_results=execution_results,
        data_profile=data_profile,
        dataset_path=DATA_PATH
    )

    if extract_special_result(primary_result) is not None:
        evidence = special_evidence(
            extract_special_result(primary_result), primary_result, task_analysis
        )

    # Store the number of validated groups as explicit numeric evidence.
    # This allows the numeric-claim validator to ground phrases such as "(4 groups)"
    # without weakening its general numeric matching rules.
    if (
        isinstance(evidence, dict)
        and str(evidence.get("analysis_type", "")).lower()
        in {"specification_driven_grouped_mean", "grouped_mean_analysis"}
    ):
        validated_groups = evidence.get("group_results")
        if isinstance(validated_groups, dict) and validated_groups:
            evidence["group_count"] = len(validated_groups)

    # Descriptive-summary claim validation requires independently computed
    # dataset facts, which are supplied by create_evidence(). The initial
    # Step 7 call only has primary_result, so rerun this validator now that
    # the evidence object is available.
    if str(analysis_type).lower() == "descriptive_summary":
        insight_validation = validate_insight(
            insight=insight,
            result=evidence,
            analysis_type=analysis_type,
            task_analysis=task_analysis,
            user_question=user_question,
            data_profile=data_profile
        )

        print("\nInsight Validation Result (using independent evidence):")
        print_json(insight_validation)

    # Deterministic answer for a single-column mean question.
    # This keeps the primary answer tied to the validated numeric result rather
    # than relying on a generic narrative generated earlier in the pipeline.
    if (
        str(analysis_type).lower() == "mean_analysis"
        and isinstance(primary_result, dict)
        and str(primary_result.get("status", "")).upper() == "PASS"
        and isinstance(primary_result.get("mean"), (int, float, np.integer, np.floating))
        and np.isfinite(float(primary_result.get("mean")))
    ):
        mean_column = str(primary_result.get("metric_column", "requested numeric column"))
        mean_value = float(primary_result["mean"])
        valid_count = primary_result.get("valid_value_count")
        deterministic_insight = (
            f"The average of the {mean_column} column is {mean_value:,.2f}"
            + (f", calculated from {int(valid_count):,} valid numeric observations." 
               if isinstance(valid_count, (int, np.integer)) else ".")
        )
        deterministic_validation = validate_insight(
            insight=deterministic_insight,
            result=validation_result_data,
            analysis_type=analysis_type,
            task_analysis=task_analysis,
            user_question=user_question,
            data_profile=data_profile,
        )
        if (
            isinstance(deterministic_validation, dict)
            and deterministic_validation.get("overall") == "PASS"
            and deterministic_validation.get("pass") is True
        ):
            insight = deterministic_insight
            insight_validation = deterministic_validation
            print("Deterministic mean insight: ACCEPTED (validated numeric result).")
        else:
            print("Deterministic mean insight: validation did not pass; keeping the existing insight.")
            print_json(deterministic_validation)

    # Deterministic answer for a single-column maximum/minimum question.
    # Validate against the actual extrema payload before replacing the generic insight.
    if (
        str(analysis_type).lower() == "extrema_analysis"
        and isinstance(primary_result, dict)
        and str(primary_result.get("status", "")).upper() == "PASS"
        and str(primary_result.get("operation", "")).lower() in {"maximum", "minimum"}
    ):
        deterministic_insight = generate_pipeline_insight(
            user_question=user_question,
            primary_result=primary_result,
            grouped_series=grouped_series,
            task_analysis=task_analysis,
            dataset_path=DATA_PATH,
        )
        deterministic_validation = validate_insight(
            insight=deterministic_insight,
            result=validation_result_data,
            analysis_type=analysis_type,
            task_analysis=task_analysis,
            user_question=user_question,
            data_profile=data_profile,
        )
        if (
            isinstance(deterministic_validation, dict)
            and deterministic_validation.get("overall") == "PASS"
            and deterministic_validation.get("pass") is True
        ):
            insight = deterministic_insight
            insight_validation = deterministic_validation
            print("Deterministic extrema insight: ACCEPTED (validated numeric result).")
        else:
            print("Deterministic extrema insight: validation did not pass.")
            print_json(deterministic_validation)

    # OPTIONAL OLLAMA INSIGHT REWRITING (validated candidate only)
    # Preserve the deterministic insight and its validation as fallbacks.
    original_insight = insight
    original_insight_validation = insight_validation
    ollama_rewrite_status = "NOT RUN"
    try:
        from ollama_insight_rewriter import rewrite_insight_with_ollama

        candidate_insight = rewrite_insight_with_ollama(
            user_question=user_question,
            original_insight=original_insight,
            verified_results=primary_result,
            evidence=evidence,
        )

        if candidate_insight:
            candidate_validation_result = (
                evidence
                if str(analysis_type).lower() == "descriptive_summary"
                else validation_result_data
            )

            candidate_validation = validate_insight(
                insight=candidate_insight,
                result=candidate_validation_result,
                analysis_type=analysis_type,
                task_analysis=task_analysis,
                user_question=user_question,
                data_profile=data_profile,
            )

            if (
                isinstance(primary_result, dict)
                and str(primary_result.get("analysis_type", "")).lower()
                == "multi_metric_grouped_mean_analysis"
            ):
                candidate_validation = (
                    validate_multi_metric_grouped_mean_insight(
                        insight=candidate_insight,
                        combined_result=primary_result,
                        base_validation=candidate_validation,
                    )
                )

            # A mean rewrite is accepted only if its metric, mean and
            # valid observation count match the computed evidence.
            candidate_mapping_passed = True

            if (
                isinstance(primary_result, dict)
                and str(primary_result.get("analysis_type", "")).lower()
                == "mean_analysis"
            ):
                candidate_mapping = build_mean_evidence_mapping(
                    candidate_insight, evidence
                )
                candidate_mapping_passed = (
                    candidate_mapping.get("status") == "PASS"
                )
                print(
                    "Ollama mean evidence mapping: "
                    + ("PASS" if candidate_mapping_passed else "FAIL")
                )

            elif str(analysis_type).lower() == "grouped_mean_analysis":
                candidate_mapping = map_insight_to_evidence(
                    candidate_insight, evidence
                )
                candidate_mapping_passed = (
                    isinstance(candidate_mapping, dict)
                    and candidate_mapping.get("status") == "PASS"
                )

            elif str(analysis_type).lower() in {
                "anomaly_detection", "outlier_analysis"
            }:
                candidate_mapping = map_insight_to_evidence(
                    candidate_insight, evidence
                )
                candidate_mapping_passed = (
                    isinstance(candidate_mapping, dict)
                    and candidate_mapping.get("status") == "PASS"
                )

            # Count / aggregate / correlation answers are validated by their own
            # deterministic validators; the generic validator and the default
            # mapping flag above do not understand them.
            if extract_special_result(primary_result) is not None:
                candidate_validation = special_validate(
                    candidate_insight, extract_special_result(primary_result), task_analysis
                )
                _special_mapping = special_mapping(candidate_insight, evidence, task_analysis)
                candidate_mapping_passed = (
                    isinstance(_special_mapping, dict) and _special_mapping.get("status") == "PASS"
                )

            candidate_passed = (
                isinstance(candidate_validation, dict)
                and candidate_validation.get("overall") == "PASS"
                and candidate_validation.get("pass") is True
                and candidate_mapping_passed
            )

            if candidate_passed:
                insight = candidate_insight
                insight_validation = candidate_validation
                ollama_rewrite_status = "ACCEPTED (rewrite passed the existing validators)"
                print(
                    "Ollama insight rewrite: ACCEPTED "
                    "(existing validators passed)."
                )
            else:
                insight = original_insight
                insight_validation = original_insight_validation
                ollama_rewrite_status = "REJECTED (validation failed; original insight kept)"
                print(
                    "Ollama insight rewrite: REJECTED "
                    "(validation failed); original insight retained."
                )
        else:
            try:
                import ollama_insight_rewriter as _rewriter_module
                _why = dict(_rewriter_module.last_status)
            except Exception:  # status is diagnostic only
                _why = {"state": "SKIPPED", "detail": "no rewrite produced"}
            _detail = _why.get("detail") or "no rewrite produced"
            ollama_rewrite_status = f"SKIPPED ({_detail}; original insight kept)"
            print(
                "Ollama insight rewrite: SKIPPED - "
                f"{_detail}; original insight retained."
            )

    except Exception as exc:
        insight = original_insight
        insight_validation = original_insight_validation
        ollama_rewrite_status = "SKIPPED (Ollama unavailable or errored; original insight kept)"
        print(
            "Ollama insight rewrite: SKIPPED due to an error; "
            "original insight retained. "
            f"Details: {exc}"
        )

    print(
        "Evidence:"
    )

    print_json(
        evidence
    )

    evidence_summary = create_evidence_summary(
        evidence
    )

    print(
        "\nEvidence Summary:"
    )

    print(
        evidence_summary
    )

    mapped_evidence = map_insight_to_evidence(
        insight,
        evidence
    )

    if (
        isinstance(primary_result, dict)
        and str(primary_result.get("analysis_type", "")).lower()
        == "mean_analysis"
    ):
        mapped_evidence = build_mean_evidence_mapping(insight, evidence)

    if (
        isinstance(primary_result, dict)
        and str(primary_result.get("analysis_type", "")).lower()
        == "extrema_analysis"
    ):
        mapped_evidence = build_extrema_evidence_mapping(insight, evidence)

    # Specification-driven grouped means use the same per-group claim checks
    # after create_evidence has normalized their validated aggregation rows.
    if extract_specification_grouped_mean(primary_result) is not None:
        mapped_evidence = build_grouped_mean_evidence_mapping(insight, evidence)

    # Generic grouped means need per-category evidence mapping; the legacy
    # mapper assumes a binary yes/no comparison and can incorrectly fail here.
    if (
        isinstance(primary_result, dict)
        and str(primary_result.get("analysis_type", "")).lower()
        == "grouped_mean_analysis"
    ):
        mapped_evidence = build_grouped_mean_evidence_mapping(insight, evidence)

    # Use a deterministic per-metric mapping for multi-metric comparisons.
    # The generic mapper may be designed around a single primary result.
    if (
        isinstance(primary_result, dict)
        and str(primary_result.get("analysis_type", "")).lower()
        == "multi_metric_grouped_mean_analysis"
    ):
        mapped_evidence = build_multi_metric_evidence_mapping(
            insight=insight,
            evidence=evidence,
            insight_validation=insight_validation,
        )

    if extract_special_result(primary_result) is not None:
        mapped_evidence = special_mapping(insight, evidence, task_analysis)
    elif (
        isinstance(mapped_evidence, dict)
        and str(mapped_evidence.get("status", "")).upper() == "PARTIAL"
        and _find_iqr_result(primary_result) is not None
    ):
        mapped_evidence = build_iqr_evidence_mapping(insight, _find_iqr_result(primary_result))
    elif (
        isinstance(mapped_evidence, dict)
        and str(mapped_evidence.get("status", "")).upper() == "PARTIAL"
        and isinstance(primary_result, dict)
        and str(primary_result.get("analysis_type", "")).lower() == "anomaly_detection"
        and "isolation" in str(primary_result.get("method", "")).lower()
    ):
        mapped_evidence = build_isolation_forest_mapping(insight, primary_result)

    mapped_evidence = attach_evidence_traceability(
        mapped_evidence,
        evidence
    )

    print(
        "\nInsight-Evidence Mapping:"
    )

    print_json(
        mapped_evidence
    )

    # =====================================================
    # STEP 9 - RECOMMENDATION
    # =====================================================

    print_section(
        "STEP 9 - RECOMMENDATION GENERATION"
    )

    recommendation = generate_pipeline_recommendation(
        user_question=user_question,
        analysis_type=analysis_type,
        primary_result=primary_result,
        grouped_series=grouped_series,
        insight=insight,
        task_analysis=task_analysis,
        evidence=evidence
    )

    print(
        "Recommendation:"
    )

    print(
        recommendation
    )

    # =====================================================
    # STEP 9A - NUMERIC CLAIM / EVIDENCE GROUNDING
    # =====================================================

    print_section(
        "STEP 9A - NUMERIC CLAIM / EVIDENCE GROUNDING"
    )

    claim_evidence_validation = validate_insight_and_recommendation(
        insight=insight,
        recommendation=recommendation,
        evidence=evidence,
    )

    print_json(claim_evidence_validation)

    # Preserve the claim-validation result with the evidence mapping so it
    # remains available to the final report without changing report APIs.
    if isinstance(mapped_evidence, dict):
        mapped_evidence["claim_evidence_validation"] = claim_evidence_validation

    # =====================================================
    # STEP 9B - INDEPENDENT RECOMMENDATION VALIDATION
    # =====================================================

    print_section(
        "STEP 9B - INDEPENDENT RECOMMENDATION VALIDATION"
    )

    recommendation_validation = validate_recommendation(
        recommendation=recommendation,
        evidence=evidence,
        analysis_type=analysis_type,
        primary_result=primary_result,
        claim_evidence_validation=claim_evidence_validation,
    )

    print_json(recommendation_validation)

    if isinstance(mapped_evidence, dict):
        mapped_evidence["recommendation_validation"] = recommendation_validation

    # =====================================================
    # STEP 10 - FINAL REPORT
    # =====================================================

    print_section(
        "STEP 10 - FINAL REPORT GENERATION"
    )

    try:
        report_extras = {
            "dataset_overview": build_dataset_overview(data_profile, DATA_PATH, dataset_name),
            "pipeline_trace": build_pipeline_trace(
                data_profile, DATA_PATH, user_question, task_analysis, execution_results,
                aggregated_result, primary_result, analysis_type, insight_validation,
                mapped_evidence, recommendation_validation, ollama_rewrite_status,
                dataset_name,
            ),
        }
    except Exception as exc:  # the trace must never prevent the report
        print(f"(Pipeline trace unavailable: {exc})")
        report_extras = {}

    final_report = create_final_report(
        user_question=user_question,
        analysis_type=analysis_type,
        validated_result=safe_json_value(primary_result),
        insight=insight,
        insight_validation=insight_validation,
        evidence=evidence,
        recommendation=recommendation,
        insight_evidence_mapping=mapped_evidence,
        recommendation_validation=recommendation_validation,
        report_extras=report_extras,
    )

    report_path = os.path.join(
        OUTPUT_DIR,
        "final_eda_report.txt"
    )

    saved_report_path = save_report(
        final_report,
        report_path
    )

    print("\\nFinal Report:")
    print(
        "Report saved successfully. "
        f"Report size: {len(str(final_report)):,} characters. "
        "The Streamlit interface displays a concise answer; use the report "
        "download for the complete detailed report."
    )

    print(
        "\\nReport saved to:"
    )

    print(
        saved_report_path
    )

    # =====================================================
    # FINAL PIPELINE STATUS
    # =====================================================

    print_section(
        "FINAL PIPELINE STATUS"
    )

    def validation_status(value):
        """Return PASS only when a validation explicitly reports PASS."""
        if isinstance(value, dict):
            for key in ("overall", "status", "overall_status", "validation_status"):
                candidate = value.get(key)
                if candidate is not None:
                    return str(candidate).strip().upper()
            if value.get("pass") is True:
                return "PASS"
            if value.get("pass") is False:
                return "FAIL"
            return "REVIEW"
        if isinstance(value, str):
            return value.strip().upper()
        if value is True:
            return "PASS"
        if value is False:
            return "FAIL"
        return "REVIEW"

    execution_status = validation_status(execution_results.get("overall_status"))
    insight_status = validation_status(insight_validation)
    evidence_status = validation_status(evidence)
    mapping_status = validation_status(mapped_evidence)
    claim_status = validation_status(claim_evidence_validation)
    # Recommendation generation returns either a structured validation
    # dictionary or plain recommendation text. Do not interpret the text
    # itself as a validation status.
    if isinstance(recommendation, dict):
        recommendation_status = validation_status(recommendation)
    elif isinstance(recommendation, str) and recommendation.strip():
        recommendation_status = "PASS"
    else:
        recommendation_status = "FAIL"
    recommendation_validation_status = validation_status(recommendation_validation)

    stage_statuses = {
        "Multi-Step Execution": execution_status,
        "Insight Validation": insight_status,
        "Evidence Validation": evidence_status,
        "Insight-Evidence Mapping": mapping_status,
        "Numeric Claim Grounding": claim_status,
        "Recommendation Generation": recommendation_status,
        "Independent Recommendation Validation": recommendation_validation_status,
    }

    for stage_name, stage_status in stage_statuses.items():
        print(f"{stage_name}: {stage_status}")

    if all(status == "PASS" for status in stage_statuses.values()):
        print("\nOVERALL RESULT: PASS")
    else:
        print("\nOVERALL RESULT: REVIEW REQUIRED")
        non_pass = {
            name: status
            for name, status in stage_statuses.items()
            if status != "PASS"
        }
        print("Stages requiring attention:")
        print_json(non_pass)


# =========================================================
# PROGRAM ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()