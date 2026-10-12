"""
TEST 3I - CLAIM-LEVEL EVIDENCE VALIDATION

Purpose:
Validate individual analytical claims against independently
computed evidence.

This test builds on the existing TEST 3H architecture.
"""

import os
import sys
import json

import pandas as pd


# ======================================================================
# PROJECT PATH
# ======================================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

SRC_DIR = os.path.join(
    PROJECT_ROOT,
    "src"
)

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


# ======================================================================
# IMPORTS
# ======================================================================

from data_profiler import profile_dataset
from task_analyzer import analyze_task_mock
from eda_planner import generate_plan_mock
from multi_step_executor import execute_plan_steps
from self_correcting_step_executor import (
    execute_self_correcting_step
)
from result_aggregator import aggregate_results
from insight_generator import generate_insight
from evidence_mapping import create_grouped_evidence
from claim_validator import validate_claims


# ======================================================================
# PATHS
# ======================================================================

DATASET_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "sales.csv"
)

OUTPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "test_3i_claim_validation_results.json"
)


# ======================================================================
# JSON SAFE CONVERSION
# ======================================================================

def safe_json_value(value):

    if value is None:
        return None

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

    if isinstance(
        value,
        (
            pd.Timestamp,
            pd.Period
        )
    ):
        return str(value)

    if hasattr(value, "item"):

        try:
            return value.item()

        except Exception:
            pass

    if isinstance(value, pd.Series):

        return {
            str(key): safe_json_value(val)
            for key, val in value.to_dict().items()
        }

    if isinstance(value, pd.DataFrame):

        return [
            {
                str(key): safe_json_value(val)
                for key, val in row.items()
            }
            for row in value.to_dict(
                orient="records"
            )
        ]

    if isinstance(value, dict):

        return {
            str(key): safe_json_value(val)
            for key, val in value.items()
        }

    if isinstance(
        value,
        (
            list,
            tuple,
            set
        )
    ):

        return [
            safe_json_value(item)
            for item in value
        ]

    return str(value)


# ======================================================================
# MAIN
# ======================================================================

def main():

    print("=" * 70)
    print(
        "TEST 3I - CLAIM-LEVEL EVIDENCE VALIDATION"
    )
    print("=" * 70)

    question = (
        "Which region generated "
        "the highest total profit?"
    )

    print("\nDataset:")
    print(DATASET_PATH)

    print("\nQuestion:")
    print(question)


    # ==================================================================
    # STEP 1 - DATA PROFILING
    # ==================================================================

    print("\n" + "=" * 70)
    print("STEP 1 - DATA PROFILING")
    print("=" * 70)

    data_profile = profile_dataset(
        DATASET_PATH
    )

    print("Data profiling: PASS")


    # ==================================================================
    # STEP 2 - TASK ANALYSIS
    # ==================================================================

    print("\n" + "=" * 70)
    print("STEP 2 - TASK ANALYSIS")
    print("=" * 70)

    task_analysis = analyze_task_mock(
        question,
        data_profile
    )

    print(task_analysis)


    # ==================================================================
    # STEP 3 - EDA PLAN
    # ==================================================================

    print("\n" + "=" * 70)
    print("STEP 3 - EDA PLAN")
    print("=" * 70)

    plan = generate_plan_mock(
        question,
        task_analysis,
        data_profile
    )

    for index, step in enumerate(
        plan,
        start=1
    ):

        print(
            f"{index}. {step}"
        )


    # ==================================================================
    # STEP 4 - EDA EXECUTION
    # ==================================================================

    print("\n" + "=" * 70)
    print("STEP 4 - EDA EXECUTION")
    print("=" * 70)

    def step_executor(
        step_number,
        step_text
    ):

        return execute_self_correcting_step(
            step_number=step_number,
            step_text=step_text,
            filepath=DATASET_PATH,
            test_fault=False
        )

    execution_result = execute_plan_steps(
        plan=plan,
        step_executor=step_executor
    )

    print(
        "\nOverall Execution Status:",
        execution_result.get(
            "overall_status"
        )
    )

    print(
        "Successful Steps:",
        execution_result.get(
            "successful_steps"
        )
    )

    print(
        "Failed Steps:",
        execution_result.get(
            "failed_steps"
        )
    )


    # ==================================================================
    # STEP 5 - RESULT AGGREGATION
    # ==================================================================

    print("\n" + "=" * 70)
    print("STEP 5 - RESULT AGGREGATION")
    print("=" * 70)

    all_steps = execution_result.get(
        "steps",
        []
    )

    aggregation = aggregate_results(
        question,
        all_steps,
        task_analysis
    )

    print(
        "Aggregation Status:",
        aggregation.get(
            "overall_status"
        )
    )

    primary_result = aggregation.get(
        "primary_result"
    )

    print("\nPrimary Result:")
    print(primary_result)


    # ==================================================================
    # STEP 6 - INDEPENDENT EVIDENCE
    # ==================================================================

    print("\n" + "=" * 70)
    print(
        "STEP 6 - INDEPENDENT EVIDENCE"
    )
    print("=" * 70)

    df = pd.read_csv(
        DATASET_PATH
    )

    grouped = (
        df.groupby("Region")["Total Profit"]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    print("\nGrouped Evidence:")
    print(grouped)

    evidence = create_grouped_evidence(
        grouped,
        "Region",
        "Total Profit",
        "sum"
    )

    print("\nEvidence:")
    print(evidence)


    # ==================================================================
    # STEP 7 - INSIGHT GENERATION
    # ==================================================================

    print("\n" + "=" * 70)
    print("STEP 7 - INSIGHT GENERATION")
    print("=" * 70)

    # Use the grouped Series because the grouped insight generator
    # expects a grouped analytical result.

    insight_result = grouped

    insight = generate_insight(
        question,
        insight_result,
        task_analysis
    )

    print("\nGenerated Insight:")
    print(insight)


    # ==================================================================
    # STEP 8 - CORRECT CLAIM VALIDATION
    # ==================================================================

    print("\n" + "=" * 70)
    print(
        "STEP 8 - CORRECT CLAIM VALIDATION"
    )
    print("=" * 70)

    correct_claims = [

        {
            "claim": (
                "Sub-Saharan Africa generated "
                "the highest total profit."
            ),
            "claim_type":
                "highest_category",
            "evidence":
                evidence
        },

        {
            "claim": (
                "Sub-Saharan Africa generated "
                "approximately 12,183,211.40."
            ),
            "claim_type":
                "numeric_value",
            "evidence":
                evidence
        },

        {
            "claim":
                "Europe ranked second.",
            "claim_type":
                "ranking",
            "evidence":
                evidence
        }
    ]

    correct_claim_validation = validate_claims(
        correct_claims
    )

    print(
        "\nCorrect Claim Validation:"
    )

    print(
        correct_claim_validation
    )

    correct_claims_pass = (
        correct_claim_validation.get(
            "failed_claims",
            0
        ) == 0
    )


    # ==================================================================
    # STEP 9 - CLAIM-LEVEL FAULT INJECTION
    # ==================================================================

    print("\n" + "=" * 70)
    print(
        "STEP 9 - CLAIM-LEVEL FAULT INJECTION"
    )
    print("=" * 70)

    wrong_claims = [

        {
            "claim": (
                "Sub-Saharan Africa generated "
                "the highest total profit."
            ),
            "claim_type":
                "highest_category",
            "evidence":
                evidence
        },

        {
            "claim": (
                "Sub-Saharan Africa generated "
                "approximately 12,183,211.40."
            ),
            "claim_type":
                "numeric_value",
            "evidence":
                evidence
        },

        {
            "claim":
                "Asia ranked second.",
            "claim_type":
                "ranking",
            "evidence":
                evidence
        }
    ]

    print("\nInjected Wrong Claim:")
    print(
        "Asia ranked second."
    )


    # ==================================================================
    # STEP 10 - WRONG CLAIM DETECTION
    # ==================================================================

    print("\n" + "=" * 70)
    print(
        "STEP 10 - WRONG CLAIM DETECTION"
    )
    print("=" * 70)

    wrong_claim_validation = validate_claims(
        wrong_claims
    )

    print(
        "\nWrong Claim Validation:"
    )

    print(
        wrong_claim_validation
    )

    wrong_claim_detected = (
        wrong_claim_validation.get(
            "failed_claims",
            0
        ) == 1
    )


    # ==================================================================
    # STEP 11 - CLAIM METRICS
    # ==================================================================

    print("\n" + "=" * 70)
    print(
        "STEP 11 - CLAIM-LEVEL METRICS"
    )
    print("=" * 70)

    correct_claim_accuracy = (
        correct_claim_validation.get(
            "claim_accuracy",
            0.0
        )
    )

    wrong_claim_accuracy = (
        wrong_claim_validation.get(
            "claim_accuracy",
            0.0
        )
    )

    print(
        "Total Correct Claims:",
        correct_claim_validation.get(
            "total_claims",
            0
        )
    )

    print(
        "Correct Claims Passed:",
        correct_claim_validation.get(
            "passed_claims",
            0
        )
    )

    print(
        "Correct Claims Failed:",
        correct_claim_validation.get(
            "failed_claims",
            0
        )
    )

    print(
        "Correct Claim Accuracy:",
        f"{correct_claim_accuracy * 100:.2f}%"
    )

    print(
        "Wrong Claim Detected:",
        "PASS"
        if wrong_claim_detected
        else "FAIL"
    )

    print(
        "Wrong Claim Accuracy:",
        f"{wrong_claim_accuracy * 100:.2f}%"
    )


    # ==================================================================
    # STEP 12 - FINAL EVALUATION
    # ==================================================================

    final_pass = (

        execution_result.get(
            "overall_status"
        ) == "PASS"

        and

        aggregation.get(
            "overall_status"
        ) == "PASS"

        and

        correct_claims_pass

        and

        wrong_claim_detected
    )

    print("\n" + "=" * 70)
    print(
        "TEST 3I - FINAL EVALUATION"
    )
    print("=" * 70)

    print(
        "EDA Execution       :",
        "PASS"
        if execution_result.get(
            "overall_status"
        ) == "PASS"
        else "FAIL"
    )

    print(
        "Aggregation         :",
        "PASS"
        if aggregation.get(
            "overall_status"
        ) == "PASS"
        else "FAIL"
    )

    print(
        "Correct Claims      :",
        "PASS"
        if correct_claims_pass
        else "FAIL"
    )

    print(
        "Wrong Claim Detect  :",
        "PASS"
        if wrong_claim_detected
        else "FAIL"
    )

    print("\n" + "=" * 70)

    if final_pass:

        print(
            "TEST 3I RESULT: PASS"
        )

        print(
            "The system successfully validated "
            "individual analytical claims."
        )

        print(
            "The intentionally incorrect claim "
            "was detected."
        )

    else:

        print(
            "TEST 3I RESULT: FAIL"
        )

    print("=" * 70)


    # ==================================================================
    # SAVE RESULTS
    # ==================================================================

    results = {

        "test":
            "3I",

        "question":
            question,

        "execution_status":
            execution_result.get(
                "overall_status"
            ),

        "aggregation_status":
            aggregation.get(
                "overall_status"
            ),

        "primary_result":
            primary_result,

        "insight":
            insight,

        "evidence":
            evidence,

        "correct_claim_validation":
            correct_claim_validation,

        "wrong_claim_validation":
            wrong_claim_validation,

        "correct_claim_accuracy":
            correct_claim_accuracy,

        "wrong_claim_detection":
            wrong_claim_detected,

        "final_status":
            (
                "PASS"
                if final_pass
                else "FAIL"
            )
    }

    os.makedirs(
        os.path.dirname(
            OUTPUT_PATH
        ),
        exist_ok=True
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            safe_json_value(results),
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\nResults saved to:")
    print(OUTPUT_PATH)


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    main()