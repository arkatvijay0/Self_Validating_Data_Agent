import os
import sys
import json
import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# IMPORT PROJECT MODULES
# ============================================================

from src.data_profiler import profile_dataset
from src.task_analyzer import analyze_task_mock
from src.eda_planner import generate_plan_mock
from src.self_correcting_step_executor import (
    execute_self_correcting_step
)
from src.result_aggregator import aggregate_results
from src.insight_generator import generate_insight
from src.insight_validator import validate_insight
from src.evidence_mapping import (
    create_grouped_evidence,
    map_insight_to_evidence
)


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "sales.csv"
)

OUTPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "test_3h_insight_validation_results.json"
)

QUESTION = (
    "Which region generated the highest total profit?"
)


# ============================================================
# JSON SAFE CONVERSION
# ============================================================

def safe_json_value(value):

    if hasattr(value, "item"):

        try:
            return value.item()

        except Exception:
            pass

    if isinstance(
        value,
        pd.Period
    ):

        return str(value)

    if isinstance(
        value,
        pd.Timestamp
    ):

        return value.isoformat()

    if isinstance(
        value,
        pd.DataFrame
    ):

        return safe_json_value(
            value.to_dict(
                orient="records"
            )
        )

    if isinstance(
        value,
        pd.Series
    ):

        return {
            str(key): safe_json_value(val)
            for key, val in value.items()
        }

    if isinstance(
        value,
        dict
    ):

        return {
            str(key): safe_json_value(val)
            for key, val in value.items()
        }

    if isinstance(
        value,
        (list, tuple, set)
    ):

        return [
            safe_json_value(item)
            for item in value
        ]

    return value


# ============================================================
# INDEPENDENT GROUND TRUTH
# ============================================================

def calculate_ground_truth(filepath):

    df = pd.read_csv(
        filepath
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

    return {
        "Region": str(
            grouped.index[0]
        ),
        "Total Profit": float(
            grouped.iloc[0]
        )
    }


# ============================================================
# BUILD GROUPED SERIES
# ============================================================

def build_grouped_series(filepath):

    df = pd.read_csv(
        filepath
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

    return grouped


# ============================================================
# BUILD EVIDENCE USING EXISTING MODULE
# ============================================================

def build_evidence(grouped):

    # --------------------------------------------------------
    # Use positional arguments deliberately.
    #
    # This avoids depending on the exact keyword names used
    # by the existing evidence_mapping.py implementation.
    # --------------------------------------------------------

    evidence = create_grouped_evidence(
        grouped,
        "Region",
        "Total Profit",
        "sum"
    )

    return evidence


# ============================================================
# FIND GROUPED RESULT FROM EXECUTION
# ============================================================

def find_grouped_result(
    step_results
):

    # --------------------------------------------------------
    # Step 2 is the ranking step and returns a pandas Series.
    # --------------------------------------------------------

    if len(step_results) >= 2:

        step_two = step_results[1]

        grouped_result = step_two.get(
            "final_result"
        )

        if isinstance(
            grouped_result,
            pd.Series
        ):

            return grouped_result

    # --------------------------------------------------------
    # Fallback: reconstruct from dataset.
    # --------------------------------------------------------

    return build_grouped_series(
        DATASET_PATH
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "TEST 3H - EVIDENCE-GROUNDED "
        "INSIGHT VALIDATION"
    )
    print("=" * 70)

    print()

    print(
        "Dataset:"
    )

    print(
        DATASET_PATH
    )

    print()

    print(
        "Question:"
    )

    print(
        QUESTION
    )

    # ========================================================
    # STEP 1 - DATA PROFILING
    # ========================================================

    print()
    print("=" * 70)
    print("STEP 1 - DATA PROFILING")
    print("=" * 70)

    data_profile = profile_dataset(
        DATASET_PATH
    )

    print()

    print(
        "Data profiling: PASS"
    )

    # ========================================================
    # STEP 2 - TASK ANALYSIS
    # ========================================================

    print()
    print("=" * 70)
    print("STEP 2 - TASK ANALYSIS")
    print("=" * 70)

    task_analysis = analyze_task_mock(
        QUESTION,
        data_profile
    )

    print()

    print(
        task_analysis
    )

    # ========================================================
    # STEP 3 - EDA PLAN
    # ========================================================

    print()
    print("=" * 70)
    print("STEP 3 - EDA PLAN")
    print("=" * 70)


    plan = generate_plan_mock(
        task_analysis.get("task_type", "descriptive_summary"),
        data_profile,
        QUESTION
            )


    print()

    for index, step in enumerate(
        plan,
        start=1
    ):

        print(
            f"{index}. {step}"
        )

    # ========================================================
    # STEP 4 - EDA EXECUTION
    # ========================================================

    print()
    print("=" * 70)
    print("STEP 4 - EDA EXECUTION")
    print("=" * 70)

    step_results = []

    for step_number, step_text in enumerate(
        plan,
        start=1
    ):

        print()
        print(
            f"STEP {step_number}: "
            f"{step_text}"
        )

        result = execute_self_correcting_step(
            step_number=step_number,
            step_text=step_text,
            filepath=DATASET_PATH,
            test_fault=False
        )

        step_results.append(
            result
        )

        print(
            f"Status: "
            f"{result.get('status')}"
        )

        print(
            f"Attempts: "
            f"{result.get('total_attempts')}"
        )

    # ========================================================
    # STEP 5 - RESULT AGGREGATION
    # ========================================================

    print()
    print("=" * 70)
    print("STEP 5 - RESULT AGGREGATION")
    print("=" * 70)

    aggregated = aggregate_results(
        QUESTION,
        step_results,
        task_analysis
    )

    print()

    print(
        "Aggregation Status:"
    )

    print(
        aggregated.get(
            "overall_status"
        )
    )

    print()

    print(
        "Aggregated Primary Result:"
    )

    print(
        aggregated.get(
            "primary_result"
        )
    )

    # ========================================================
    # STEP 6 - GROUPED SERIES
    # ========================================================

    print()
    print("=" * 70)
    print("STEP 6 - VALIDATED GROUPED SERIES")
    print("=" * 70)

    grouped_result = find_grouped_result(
        step_results
    )

    print()

    print(
        "Grouped Result:"
    )

    print(
        grouped_result
    )

    print()

    print(
        "Result Type:"
    )

    print(
        type(grouped_result)
    )

    # ========================================================
    # STEP 7 - INDEPENDENT GROUND TRUTH
    # ========================================================

    print()
    print("=" * 70)
    print("STEP 7 - INDEPENDENT GROUND TRUTH")
    print("=" * 70)

    ground_truth = (
        calculate_ground_truth(
            DATASET_PATH
        )
    )

    print()

    print(
        ground_truth
    )

    # ========================================================
    # STEP 8 - EVIDENCE MAPPING
    # ========================================================

    print()
    print("=" * 70)
    print("STEP 8 - EVIDENCE MAPPING")
    print("=" * 70)

    evidence = build_evidence(
        grouped_result
    )

    print()

    print(
        "Evidence:"
    )

    print(
        evidence
    )

    # ========================================================
    # STEP 9 - CORRECT INSIGHT GENERATION
    # ========================================================

    print()
    print("=" * 70)
    print("STEP 9 - CORRECT INSIGHT GENERATION")
    print("=" * 70)

    correct_insight = generate_insight(
        QUESTION,
        grouped_result,
        task_analysis
    )

    print()

    print(
        "Generated Correct Insight:"
    )

    print(
        correct_insight
    )

    # ========================================================
    # STEP 10 - CORRECT INSIGHT VALIDATION
    # ========================================================

    print()
    print("=" * 70)
    print("STEP 10 - CORRECT INSIGHT VALIDATION")
    print("=" * 70)

    correct_validation = validate_insight(
        correct_insight,
        grouped_result,
        task_analysis
    )

    print()

    print(
        "Correct Insight Validation:"
    )

    print(
        correct_validation
    )

    # ========================================================
    # STEP 11 - CORRECT INSIGHT → EVIDENCE
    # ========================================================

    print()
    print("=" * 70)
    print("STEP 11 - CORRECT INSIGHT → EVIDENCE")
    print("=" * 70)

    correct_evidence_mapping = (
        map_insight_to_evidence(
            correct_insight,
            evidence
        )
    )

    print()

    print(
        "Correct Insight Evidence Mapping:"
    )

    print(
        correct_evidence_mapping
    )

    # ========================================================
    # STEP 12 - SEMANTIC INSIGHT FAULT
    # ========================================================

    print()
    print("=" * 70)
    print("STEP 12 - SEMANTIC INSIGHT FAULT INJECTION")
    print("=" * 70)

    wrong_insight = (
        "Europe generated the highest total profit."
    )

    print()

    print(
        "Correct Insight:"
    )

    print(
        correct_insight
    )

    print()

    print(
        "Injected Wrong Insight:"
    )

    print(
        wrong_insight
    )

    # ========================================================
    # STEP 13 - WRONG INSIGHT VALIDATION
    # ========================================================

    print()
    print("=" * 70)
    print("STEP 13 - WRONG INSIGHT VALIDATION")
    print("=" * 70)

    wrong_validation = validate_insight(
        wrong_insight,
        grouped_result,
        task_analysis
    )

    print()

    print(
        "Wrong Insight Validation:"
    )

    print(
        wrong_validation
    )

    # ========================================================
    # STEP 14 - WRONG INSIGHT → EVIDENCE
    # ========================================================

    print()
    print("=" * 70)
    print("STEP 14 - WRONG INSIGHT → EVIDENCE")
    print("=" * 70)

    wrong_evidence_mapping = (
        map_insight_to_evidence(
            wrong_insight,
            evidence
        )
    )

    print()

    print(
        "Wrong Insight Evidence Mapping:"
    )

    print(
        wrong_evidence_mapping
    )

    # ========================================================
    # STEP 15 - INSIGHT RECOVERY
    # ========================================================

    print()
    print("=" * 70)
    print("STEP 15 - INSIGHT RECOVERY")
    print("=" * 70)

    recovered_insight = (
        correct_insight
    )

    recovered_validation = (
        validate_insight(
            recovered_insight,
            grouped_result,
            task_analysis
        )
    )

    recovered_evidence_mapping = (
        map_insight_to_evidence(
            recovered_insight,
            evidence
        )
    )

    print()

    print(
        "Recovered Insight:"
    )

    print(
        recovered_insight
    )

    print()

    print(
        "Recovered Validation:"
    )

    print(
        recovered_validation
    )

    print()

    print(
        "Recovered Evidence Mapping:"
    )

    print(
        recovered_evidence_mapping
    )

    # ========================================================
    # STEP 16 - FINAL EVALUATION
    # ========================================================

    print()
    print("=" * 70)
    print("STEP 16 - FINAL TEST EVALUATION")
    print("=" * 70)

    execution_pass = all(
        result.get(
            "status"
        ) == "PASS"
        for result in step_results
    )

    aggregation_pass = (
        aggregated.get(
            "overall_status"
        ) == "PASS"
    )

    grouped_result_pass = (
        isinstance(
            grouped_result,
            pd.Series
        )
        and
        len(grouped_result) > 0
    )

    correct_validation_pass = (
        correct_validation.get(
            "overall"
        ) == "PASS"
    )

    correct_evidence_pass = (
        correct_evidence_mapping.get(
            "status"
        ) == "PASS"
    )

    wrong_insight_detected = (
        wrong_validation.get(
            "overall"
        ) == "FAIL"
    )

    wrong_evidence_detected = (
        wrong_evidence_mapping.get(
            "status"
        ) == "FAIL"
    )

    recovered_validation_pass = (
        recovered_validation.get(
            "overall"
        ) == "PASS"
    )

    recovered_evidence_pass = (
        recovered_evidence_mapping.get(
            "status"
        ) == "PASS"
    )

    final_pass = all([
        execution_pass,
        aggregation_pass,
        grouped_result_pass,
        correct_validation_pass,
        correct_evidence_pass,
        wrong_insight_detected,
        wrong_evidence_detected,
        recovered_validation_pass,
        recovered_evidence_pass
    ])

    print()

    print(
        f"EDA Execution         : "
        f"{'PASS' if execution_pass else 'FAIL'}"
    )

    print(
        f"Aggregation           : "
        f"{'PASS' if aggregation_pass else 'FAIL'}"
    )

    print(
        f"Grouped Result        : "
        f"{'PASS' if grouped_result_pass else 'FAIL'}"
    )

    print(
        f"Correct Insight       : "
        f"{'PASS' if correct_validation_pass else 'FAIL'}"
    )

    print(
        f"Correct Evidence      : "
        f"{'PASS' if correct_evidence_pass else 'FAIL'}"
    )

    print(
        f"Wrong Insight Detect  : "
        f"{'PASS' if wrong_insight_detected else 'FAIL'}"
    )

    print(
        f"Wrong Evidence Detect : "
        f"{'PASS' if wrong_evidence_detected else 'FAIL'}"
    )

    print(
        f"Insight Recovery      : "
        f"{'PASS' if recovered_validation_pass else 'FAIL'}"
    )

    print(
        f"Evidence Recovery     : "
        f"{'PASS' if recovered_evidence_pass else 'FAIL'}"
    )

    print()

    if final_pass:

        print("=" * 70)
        print(
            "TEST 3H RESULT: PASS"
        )
        print("=" * 70)

        print()

        print(
            "The system successfully validated "
            "an evidence-supported insight."
        )

        print(
            "The intentionally contradictory insight "
            "was detected and rejected."
        )

        print(
            "The recovered insight was validated "
            "against the evidence."
        )

    else:

        print("=" * 70)
        print(
            "TEST 3H RESULT: FAIL"
        )
        print("=" * 70)

    # ========================================================
    # STEP 17 - SAVE RESULTS
    # ========================================================

    final_results = {

        "test_name":
            "TEST 3H - Evidence-Grounded "
            "Insight Validation",

        "dataset":
            DATASET_PATH,

        "question":
            QUESTION,

        "ground_truth":
            ground_truth,

        "aggregated_primary_result":
            aggregated.get(
                "primary_result"
            ),

        "grouped_result":
            grouped_result,

        "execution_pass":
            execution_pass,

        "aggregation_pass":
            aggregation_pass,

        "grouped_result_pass":
            grouped_result_pass,

        "correct_insight":
            correct_insight,

        "correct_validation":
            correct_validation,

        "correct_evidence_mapping":
            correct_evidence_mapping,

        "wrong_insight":
            wrong_insight,

        "wrong_validation":
            wrong_validation,

        "wrong_evidence_mapping":
            wrong_evidence_mapping,

        "recovered_insight":
            recovered_insight,

        "recovered_validation":
            recovered_validation,

        "recovered_evidence_mapping":
            recovered_evidence_mapping,

        "overall_result":
            (
                "PASS"
                if final_pass
                else "FAIL"
            ),

        "step_results":
            step_results
    }

    final_results = safe_json_value(
        final_results
    )

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
            final_results,
            file,
            indent=4,
            ensure_ascii=False
        )

    print()

    print(
        "Results saved to:"
    )

    print(
        OUTPUT_PATH
    )

    print()


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
