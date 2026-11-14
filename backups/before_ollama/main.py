# =========================================================
# MAIN PIPELINE
# SELF-VALIDATING EDA AGENT
# =========================================================

import os
import sys
import json
import inspect
import pandas as pd
import numpy as np


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
    "sales.csv"
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

from evidence_mapping import (
    create_grouped_evidence,
    create_correlation_evidence,
    create_trend_evidence,
    create_evidence_summary,
)

from evidence_mapping_grouped_mean_fixed import (
    map_insight_to_evidence,
)

from recommendation_generator import (
    generate_recommendation
)

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
    """Return True only when a grouped-mean result has both groups and finite means."""
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
    return finite_number(no_mean) and finite_number(yes_mean) and finite_number(difference)


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

    insight_pass = (
        isinstance(insight_validation, dict)
        and insight_validation.get("pass") is True
        and str(insight_validation.get("overall", "")).upper() == "PASS"
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

        checks[metric] = {
            "result_valid": metric_ok,
            "metric_named": str(metric).lower() in text,
            "non_subscriber_mean_present": metric_ok and number_present(no_mean),
            "subscriber_mean_present": metric_ok and number_present(yes_mean),
            "difference_present": metric_ok and number_present(difference),
        }
        checks[metric]["pass"] = all(
            value for key, value in checks[metric].items() if key != "pass"
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
    # GROUPED MEAN COMPARISON
    # -----------------------------------------------------
    # Keep group-specific means and counts as the evidence source instead
    # of routing this result to the legacy Region / Total Profit branch.
    if (
        isinstance(primary_result, dict)
        and str(primary_result.get("analysis_type", "")).lower()
        == "grouped_mean_analysis"
    ):
        group_results = primary_result.get("group_results", {})
        no_group = group_results.get("no", {}) if isinstance(group_results, dict) else {}
        yes_group = group_results.get("yes", {}) if isinstance(group_results, dict) else {}
        difference = primary_result.get("mean_difference_yes_minus_no")

        numeric_types = (int, float, np.integer, np.floating)
        valid_result = (
            isinstance(no_group, dict)
            and isinstance(yes_group, dict)
            and isinstance(no_group.get("mean_value", no_group.get("mean")), numeric_types)
            and isinstance(yes_group.get("mean_value", yes_group.get("mean")), numeric_types)
            and isinstance(no_group.get("row_count"), (int, np.integer))
            and isinstance(yes_group.get("row_count"), (int, np.integer))
            and isinstance(no_group.get("valid_value_count"), (int, np.integer))
            and isinstance(yes_group.get("valid_value_count"), (int, np.integer))
            and isinstance(difference, numeric_types)
        )

        evidence_payload = safe_json_value(primary_result)
        return {
            "status": "PASS" if valid_result else "FAIL",
            "analysis_type": "grouped_mean_analysis",
            "metric_column": primary_result.get("metric_column", "balance"),
            "group_column": primary_result.get("group_column", "y"),
            "group_results": safe_json_value(group_results),
            "mean_difference_yes_minus_no": (
                float(difference) if isinstance(difference, numeric_types) else None
            ),
            "supporting_result": evidence_payload,
            "result": evidence_payload,
            "evidence_summary": json.dumps(
                evidence_payload, indent=2, ensure_ascii=False
            ),
            "reason": None if valid_result else (
                "Grouped mean result is missing required group statistics."
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


# =========================================================
# INSIGHT GENERATION
# =========================================================

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

    if (
        isinstance(primary_result, dict)
        and str(primary_result.get("analysis_type", "")).lower()
        == "multi_metric_grouped_mean_analysis"
    ):
        return generate_multi_metric_grouped_mean_insight(primary_result)

    # A grouped mean comparison must use the validated group means, not a
    # generic overall mean or a sales-specific grouped-series fallback.
    if (
        isinstance(primary_result, dict)
        and str(primary_result.get("analysis_type", "")).lower()
        == "grouped_mean_analysis"
    ):
        groups = primary_result.get("group_results", {})
        no_group = groups.get("no", {}) if isinstance(groups, dict) else {}
        yes_group = groups.get("yes", {}) if isinstance(groups, dict) else {}
        no_mean = no_group.get("mean_value", no_group.get("mean"))
        yes_mean = yes_group.get("mean_value", yes_group.get("mean"))
        difference = primary_result.get("mean_difference_yes_minus_no")
        metric_value = primary_result.get("metric_column")
        metric = str(metric_value).strip() if metric_value is not None else ""

        if not metric:
            return (
                "The grouped mean comparison cannot be summarized because the "
                "requested metric column is missing from the validated result. "
                "No substitute metric was used."
            )

        numeric_values = (no_mean, yes_mean, difference)
        if all(
            isinstance(value, (int, float, np.integer, np.floating))
            and not isinstance(value, bool)
            and np.isfinite(float(value))
            for value in numeric_values
        ):
            direction = "higher" if float(difference) > 0 else (
                "lower" if float(difference) < 0 else "the same as"
            )
            difference_magnitude = abs(float(difference))
            return (
                f"Average {metric} was {float(no_mean):,.2f} for non-subscribers "
                f"and {float(yes_mean):,.2f} for subscribers. "
                f"The subscriber group average was {difference_magnitude:,.2f} "
                f"{direction} than the non-subscriber average "
                f"(yes minus no = {float(difference):,.2f}). "
                "The comparison is descriptive and does not establish causation."
            )
        return (
            f"The grouped average comparison for {metric} could not be summarized "
            "because one or more required group means were missing or non-numeric."
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
        if not isinstance(groups, dict):
            groups = {}

        metric = str(
            primary_result.get("metric_column")
            or primary_result.get("metric")
            or "requested metric"
        ).strip()
        metric_label = metric.replace("_", " ")

        def extract_mean(group):
            if not isinstance(group, dict):
                return None

            for key in ("mean_value", "mean", f"mean_{metric}", "average"):
                value = group.get(key)
                if isinstance(
                    value, (int, float, np.integer, np.floating)
                ) and np.isfinite(float(value)):
                    return float(value)

            return None

        no_group = groups.get("no", {})
        yes_group = groups.get("yes", {})

        no_mean = extract_mean(no_group)
        yes_mean = extract_mean(yes_group)

        if no_mean is None or yes_mean is None:
            return (
                f"The grouped comparison for {metric_label} could not be "
                "fully verified because one or both group means are missing. "
                "Review the metric-specific result and evidence before using "
                "this comparison for customer targeting."
            )

        difference = yes_mean - no_mean

        if difference > 0:
            direction = "higher"
        elif difference < 0:
            direction = "lower"
        else:
            direction = "the same"

        if difference == 0:
            comparison = (
                f"both groups have the same average {metric_label} "
                f"({yes_mean:.2f})"
            )
        else:
            comparison = (
                f"the subscriber group's average {metric_label} is "
                f"{direction} by approximately {abs(difference):.2f}"
            )

        return (
            f"The non-subscriber group's average {metric_label} is "
            f"{no_mean:.2f}, and the subscriber group's average is "
            f"{yes_mean:.2f}; {comparison}. Investigate whether this "
            "difference is associated with other customer characteristics. "
            "Check group sizes and data quality before making targeting "
            "decisions. This descriptive comparison does not establish "
            "causation."
        )

        groups = primary_result.get("group_results", {})
        no_group = groups.get("no", {}) if isinstance(groups, dict) else {}
        yes_group = groups.get("yes", {}) if isinstance(groups, dict) else {}
        no_mean = no_group.get("mean_balance")
        yes_mean = yes_group.get("mean_balance")
        if isinstance(no_mean, (int, float, np.integer, np.floating)) and isinstance(
            yes_mean, (int, float, np.integer, np.floating)
        ):
            return (
                "Investigate whether the higher average balance among subscribers "
                "is associated with other customer characteristics, such as job, "
                "age, or campaign contact history. Compare group sizes and avoid "
                "treating this descriptive difference as evidence that balance causes subscription."
            )
        return (
            "Validate the group-level balance statistics before using this comparison "
            "to guide customer targeting."
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

def main(dataset_path=None, user_question=None, test_fault=False):

    # Use the dataset supplied by Streamlit when present; retain the default
    # sales dataset for standalone command-line runs.
    if dataset_path:
        DATA_PATH = os.path.abspath(str(dataset_path))
    else:
        DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "sales.csv")

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

    # =====================================================
    # STEP 3 - EDA PLAN
    # =====================================================

    print_section(
        "STEP 3 - EDA PLAN"
    )

    # IMPORTANT: generate_plan_mock expects (task_type, data_profile, user_query).
    # Passing these arguments in the wrong order can produce an invalid/empty plan.
    plan = generate_plan_mock(
        task_analysis.get("task_type", "descriptive_summary"),
        data_profile,
        user_question
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
        test_fault=bool(test_fault)
    )

    execution_results = execute_plan_steps(
        plan=plan,
        step_executor=step_executor
    )

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

    print(
        "\nGrouped Series Used for Insight:"
    )

    if grouped_series is not None:

        print(
            grouped_series
        )

    else:

        print(
            "Could not reconstruct grouped series."
        )

    # =====================================================
    # ANALYSIS TYPE
    # =====================================================

    result_analysis_type = (
        primary_result.get("analysis_type", "")
        if isinstance(primary_result, dict)
        else ""
    )

    if str(result_analysis_type).lower() in (
        "conversion_factor_analysis",
        "grouped_mean_analysis",
        "multi_metric_grouped_mean_analysis",
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
    # STEP 10 - FINAL REPORT
    # =====================================================

    print_section(
        "STEP 10 - FINAL REPORT GENERATION"
    )

    final_report = create_final_report(
        user_question=user_question,
        analysis_type=analysis_type,
        validated_result=safe_json_value(primary_result),
        insight=insight,
        insight_validation=insight_validation,
        evidence=evidence,
        recommendation=recommendation,
        insight_evidence_mapping=mapped_evidence
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

    execution_pass = (
        execution_results.get(
            "overall_status"
        )
        == "PASS"
    )

    if isinstance(
        insight_validation,
        dict
    ):

        insight_pass = (
            insight_validation.get(
                "overall"
            )
            == "PASS"
        )

    else:

        insight_pass = (
            str(
                insight_validation
            ).upper()
            == "PASS"
        )

    evidence_pass = (
        isinstance(
            evidence,
            dict
        )
        and
        evidence.get(
            "status"
        )
        == "PASS"
    )

    mapping_pass = (
        isinstance(
            mapped_evidence,
            dict
        )
        and
        mapped_evidence.get(
            "status"
        )
        == "PASS"
    )

    print(
        "Multi-Step Execution:",
        "PASS"
        if execution_pass
        else
        "FAIL"
    )

    print(
        "Insight Validation:",
        "PASS"
        if insight_pass
        else
        "FAIL"
    )

    print(
        "Evidence Mapping:",
        "PASS"
        if evidence_pass
        else
        "FAIL"
    )

    print(
        "Insight-Evidence Mapping:",
        "PASS"
        if mapping_pass
        else
        "FAIL"
    )

    if (
        execution_pass
        and
        insight_pass
        and
        evidence_pass
        and
        mapping_pass
    ):

        print(
            "\nOVERALL RESULT: PASS"
        )

    else:

        print(
            "\nOVERALL RESULT: REVIEW REQUIRED"
        )


# =========================================================
# PROGRAM ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()