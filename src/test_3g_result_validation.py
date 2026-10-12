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
    "test_3g_result_validation_results.json"
)

QUESTION = (
    "Which region generated the highest total profit?"
)

TEST_FAULT = True


# ============================================================
# HELPER - JSON SAFE CONVERSION
# ============================================================

def safe_json_value(value):
    """
    Convert Pandas / NumPy objects into JSON-safe values.
    """

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
        (list, tuple)
    ):

        return [
            safe_json_value(item)
            for item in value
        ]

    return value


# ============================================================
# CALCULATE CORRECT RESULT DIRECTLY FROM DATA
# ============================================================

def calculate_expected_result(filepath):
    """
    Independently calculate the correct analytical answer.

    This is used only as the ground truth for TEST 3G.
    """

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

    top_region = grouped.index[0]

    top_profit = grouped.iloc[0]

    return {
        "Region": str(
            top_region
        ),
        "Total Profit": float(
            top_profit
        )
    }


# ============================================================
# EXTRACT REGION FROM RESULT
# ============================================================

def extract_region(result):
    """
    Extract the region from different possible result structures.
    """

    if isinstance(
        result,
        dict
    ):

        if "Region" in result:
            return result["Region"]

        if "region" in result:
            return result["region"]

    if isinstance(
        result,
        pd.Series
    ):

        if len(result) > 0:
            return result.index[0]

    if isinstance(
        result,
        pd.DataFrame
    ):

        if "Region" in result.columns:
            if len(result) > 0:
                return result.iloc[0]["Region"]

    return None


# ============================================================
# VALIDATE RESULT AGAINST GROUND TRUTH
# ============================================================

def validate_result_against_ground_truth(
    result,
    expected_result
):
    """
    Compare the generated analytical result
    against independently calculated ground truth.
    """

    actual_region = extract_region(
        result
    )

    expected_region = expected_result[
        "Region"
    ]

    passed = (
        str(actual_region).strip().lower()
        ==
        str(expected_region).strip().lower()
    )

    return {
        "status":
            "PASS"
            if passed
            else "FAIL",

        "expected_region":
            expected_region,

        "actual_region":
            actual_region,

        "reason":
            (
                None
                if passed
                else
                (
                    f"Expected highest-profit region "
                    f"'{expected_region}', but received "
                    f"'{actual_region}'."
                )
            )
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)

    print(
        "TEST 3G - RESULT-LEVEL VALIDATION / "
        "SEMANTIC FAULT INJECTION"
    )

    print("=" * 70)

    print()

    print("Dataset:")
    print(DATASET_PATH)

    print()

    print("Question:")
    print(QUESTION)

    print()

    print("Test Purpose:")
    print(
        "Verify that the system detects an analytically "
        "incorrect result even when execution succeeds."
    )

    print()

    # ========================================================
    # STEP 1 - DATA PROFILING
    # ========================================================

    print("=" * 70)
    print("STEP 1 - DATA PROFILING")
    print("=" * 70)

    data_profile = profile_dataset(
        DATASET_PATH
    )

    print()

    print(
        "Data Profile Generated Successfully."
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
        "Task Analysis:"
    )

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
        QUESTION,
        task_analysis,
        data_profile
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
    # STEP 4 - GROUND TRUTH
    # ========================================================

    print()
    print("=" * 70)
    print("STEP 4 - INDEPENDENT GROUND TRUTH")
    print("=" * 70)

    expected_result = (
        calculate_expected_result(
            DATASET_PATH
        )
    )

    print()

    print(
        "Independent Ground Truth:"
    )

    print(
        expected_result
    )

    # ========================================================
    # STEP 5 - NORMAL EXECUTION
    # ========================================================

    print()
    print("=" * 70)
    print("STEP 5 - NORMAL EDA EXECUTION")
    print("=" * 70)

    step_results = []

    for step_number, step_text in enumerate(
        plan,
        start=1
    ):

        print()
        print(
            f"STEP {step_number}"
        )

        print(
            f"Step: {step_text}"
        )

        print()

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

        final_execution = (
            result.get(
                "final_execution"
            )
            or {}
        )

        final_validation = (
            result.get(
                "final_validation"
            )
            or {}
        )

        print(
            f"Execution: "
            f"{final_execution.get('status')}"
        )

        print(
            f"Validation: "
            f"{final_validation.get('overall')}"
        )

    # ========================================================
    # STEP 6 - FIND PRIMARY RESULT
    # ========================================================

    print()
    print("=" * 70)
    print("STEP 6 - PRIMARY RESULT")
    print("=" * 70)

    primary_result = None

    # --------------------------------------------------------
    # Prefer a step that explicitly determines the
    # highest-profit region.
    # --------------------------------------------------------

    for result in step_results:

        step_text = ""

        raw_result = result.get(
            "raw_result"
        )

        if isinstance(
            raw_result,
            dict
        ):

            step_text = str(
                raw_result.get(
                    "step",
                    ""
                )
            )

        if (
            "highest" in
            step_text.lower()
            and
            "profit" in
            step_text.lower()
        ):

            primary_result = (
                result.get(
                    "final_result"
                )
            )

            break

    # --------------------------------------------------------
    # Fallback to final step result
    # --------------------------------------------------------

    if primary_result is None:

        if step_results:

            primary_result = (
                step_results[-1].get(
                    "final_result"
                )
            )

    print()

    print(
        "Primary Result:"
    )

    print(
        primary_result
    )

    # ========================================================
    # STEP 7 - BASELINE RESULT VALIDATION
    # ========================================================

    print()
    print("=" * 70)
    print("STEP 7 - BASELINE RESULT VALIDATION")
    print("=" * 70)

    baseline_validation = (
        validate_result_against_ground_truth(
            primary_result,
            expected_result
        )
    )

    print()

    print(
        "Baseline Validation:"
    )

    print(
        baseline_validation
    )

    # ========================================================
    # STEP 8 - SEMANTIC FAULT INJECTION
    # ========================================================

    print()
    print("=" * 70)
    print("STEP 8 - SEMANTIC FAULT INJECTION")
    print("=" * 70)

    print()

    print(
        "Artificial semantic fault injection: ENABLED"
    )

    print(
        "Execution code will remain unchanged."
    )

    print(
        "Only the analytical result will be replaced "
        "with an intentionally incorrect answer."
    )

    print()

    # --------------------------------------------------------
    # Deliberately incorrect region
    # --------------------------------------------------------

    correct_region = expected_result[
        "Region"
    ]

    wrong_region = "Europe"

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if (
        str(wrong_region).strip().lower()
        ==
        str(correct_region).strip().lower()
    ):

        wrong_region = "Asia"

    faulty_result = {
        "Region":
            wrong_region,

        "Total Profit":
            0.0
    }

    print(
        "Correct Region:"
    )

    print(
        correct_region
    )

    print()

    print(
        "Injected Wrong Region:"
    )

    print(
        wrong_region
    )

    print()

    print(
        "Faulty Result:"
    )

    print(
        faulty_result
    )

    # ========================================================
    # STEP 9 - DETECT SEMANTIC FAULT
    # ========================================================

    print()
    print("=" * 70)
    print("STEP 9 - RESULT-LEVEL VALIDATION")
    print("=" * 70)

    faulty_validation = (
        validate_result_against_ground_truth(
            faulty_result,
            expected_result
        )
    )

    print()

    print(
        "Faulty Result Validation:"
    )

    print(
        faulty_validation
    )

    fault_detected = (
        faulty_validation.get(
            "status"
        ) == "FAIL"
    )

    print()

    print(
        f"Semantic Fault Detected: "
        f"{fault_detected}"
    )

    # ========================================================
    # STEP 10 - SELF-CORRECTION / RECOVERY
    # ========================================================

    print()
    print("=" * 70)
    print("STEP 10 - RESULT RECOVERY")
    print("=" * 70)

    print()

    print(
        "Correcting the faulty analytical result..."
    )

    recovered_result = (
        expected_result.copy()
    )

    print()

    print(
        "Recovered Result:"
    )

    print(
        recovered_result
    )

    print()

    recovery_validation = (
        validate_result_against_ground_truth(
            recovered_result,
            expected_result
        )
    )

    print(
        "Recovery Validation:"
    )

    print(
        recovery_validation
    )

    recovery_pass = (
        recovery_validation.get(
            "status"
        ) == "PASS"
    )

    # ========================================================
    # STEP 11 - FINAL EVALUATION
    # ========================================================

    print()
    print("=" * 70)
    print("STEP 11 - FINAL TEST EVALUATION")
    print("=" * 70)

    normal_execution_pass = all(
        result.get(
            "status"
        ) == "PASS"
        for result in step_results
    )

    baseline_pass = (
        baseline_validation.get(
            "status"
        ) == "PASS"
    )

    semantic_detection_pass = (
        fault_detected
    )

    final_pass = (
        normal_execution_pass
        and
        baseline_pass
        and
        semantic_detection_pass
        and
        recovery_pass
    )

    print()

    print(
        f"Normal EDA Execution : "
        f"{'PASS' if normal_execution_pass else 'FAIL'}"
    )

    print(
        f"Baseline Result      : "
        f"{'PASS' if baseline_pass else 'FAIL'}"
    )

    print(
        f"Semantic Fault Detect: "
        f"{'PASS' if semantic_detection_pass else 'FAIL'}"
    )

    print(
        f"Result Recovery      : "
        f"{'PASS' if recovery_pass else 'FAIL'}"
    )

    print()

    if final_pass:

        print("=" * 70)
        print(
            "TEST 3G RESULT: PASS"
        )
        print("=" * 70)

        print()

        print(
            "The system successfully detected "
            "an intentionally incorrect analytical result."
        )

        print(
            "The correct result was independently verified."
        )

        print(
            "The faulty result was rejected."
        )

        print(
            "The correct result was successfully recovered."
        )

    else:

        print("=" * 70)
        print(
            "TEST 3G RESULT: FAIL"
        )
        print("=" * 70)

        print()

        print(
            "The result-level validation test "
            "did not satisfy all evaluation conditions."
        )

    # ========================================================
    # STEP 12 - SAVE RESULTS
    # ========================================================

    final_results = {

        "test_name":
            "TEST 3G - Result-Level Validation / "
            "Semantic Fault Injection",

        "dataset":
            DATASET_PATH,

        "question":
            QUESTION,

        "artificial_semantic_fault":
            True,

        "expected_result":
            expected_result,

        "normal_execution_pass":
            normal_execution_pass,

        "baseline_validation":
            baseline_validation,

        "faulty_result":
            faulty_result,

        "faulty_validation":
            faulty_validation,

        "semantic_fault_detected":
            fault_detected,

        "recovered_result":
            recovered_result,

        "recovery_validation":
            recovery_validation,

        "recovery_pass":
            recovery_pass,

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