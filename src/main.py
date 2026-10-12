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
    map_insight_to_evidence
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

def print_json(value):

    print(
        json.dumps(
            safe_json_value(value),
            indent=4
        )
    )


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
    if analysis_type == "grouped_analysis":
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
# CREATE EVIDENCE
# =========================================================

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

def main(dataset_path=None, user_question=None):

    # Use the dataset and question supplied by app.py when available.
    # Preserve command-line behavior when main() is run directly.
    active_dataset_path = (
        os.path.abspath(os.fspath(dataset_path))
        if dataset_path is not None and str(dataset_path).strip()
        else DATA_PATH
    )

    if not os.path.isfile(active_dataset_path):
        raise FileNotFoundError(
            f"Dataset file was not found: {active_dataset_path}"
        )

    if user_question is None:
        user_question = input("\nEnter your analysis question: ").strip()
    else:
        user_question = str(user_question).strip()

    if not user_question:
        raise ValueError("Please enter a non-empty analysis question.")

    print_section(
        "SELF-VALIDATING EDA AGENT"
    )

    print(
        "Dataset:",
        active_dataset_path
    )

    # =====================================================
    # STEP 1 - DATA PROFILING
    # =====================================================

    print_section(
        "STEP 1 - DATA PROFILING"
    )

    data_profile = profile_dataset(
        active_dataset_path
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

    user_question = input(
        "\nEnter your analysis question: "
    ).strip()

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
        filepath=active_dataset_path,
        data_profile=data_profile,
        dataset_columns=dataset_columns,
        target_columns=target_columns,
        test_fault=False
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
        active_dataset_path
    )

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
        active_dataset_path
    )

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

    analysis_type = (
        task_analysis.get(
            "task_type",
            "general_analysis"
        )
        if isinstance(
            task_analysis,
            dict
        )
        else
        "general_analysis"
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
        dataset_path=active_dataset_path
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
            str(analysis_type).lower()
            == "comparison"
            and
            grouped_series is not None
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
        dataset_path=active_dataset_path
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

    print(
        "\nFinal Report:"
    )

    print(
        final_report
    )

    print(
        "\nReport saved to:"
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