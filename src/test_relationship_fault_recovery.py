import os
import sys

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

SRC_PATH = os.path.join(
    PROJECT_ROOT,
    "src"
)

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)


# ============================================================
# PROJECT MODULES
# ============================================================

from data_profiler import profile_dataset
from task_analyzer import analyze_task_mock
from eda_planner import generate_plan_mock

from multi_step_executor import execute_plan_steps
from self_correcting_step_executor import execute_self_correcting_step

import self_correcting_step_executor

from result_aggregator import aggregate_results
from insight_generator import generate_insight
from insight_validator import validate_insight

from evidence_mapping import (
    create_correlation_evidence,
    map_insight_to_evidence
)

from recommendation_generator import generate_recommendation


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "sales.csv"
)

USER_QUESTION = (
    "Is there a relationship between units sold and total profit?"
)


# ============================================================
# DISPLAY HELPER
# ============================================================

def print_section(title):

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# NORMALIZATION HELPERS
# ============================================================

def normalize_profile(profile_result):

    if isinstance(profile_result, dict):
        return profile_result

    if hasattr(profile_result, "to_dict"):
        return profile_result.to_dict()

    return profile_result


def normalize_result(result):

    if isinstance(result, dict):
        return result

    if hasattr(result, "to_dict"):
        return result.to_dict()

    return result


# ============================================================
# TEST-ONLY FAULT INJECTION
# ============================================================
#
# IMPORTANT:
# This function belongs ONLY to this evaluation test.
#
# Production code is NOT modified.
#
# The original executor's fault injector only replaces
# "Total Profit". That does not create a fault for every
# relationship-analysis step.
#
# This test injector guarantees that every first attempt
# contains an actual invalid column reference.
# ============================================================

def inject_test_fault_for_all_steps(code):

    if not code:
        return code

    # --------------------------------------------------------
    # Case 1:
    # If Total Profit exists, use the existing proven
    # Wrong Profit fault.
    # --------------------------------------------------------

    if "Total Profit" in code:

        return code.replace(
            "Total Profit",
            "Wrong Profit",
            1
        )

    # --------------------------------------------------------
    # Case 2:
    # If Units Sold exists, deliberately break that column.
    # --------------------------------------------------------

    if "Units Sold" in code:

        return code.replace(
            "Units Sold",
            "Wrong Units Sold",
            1
        )

    # --------------------------------------------------------
    # Case 3:
    # Generic EDA code.
    #
    # For steps such as:
    #   - IQR analysis
    #   - descriptive statistics
    #
    # the generated code may not explicitly contain a
    # specific column name.
    #
    # Therefore append a guaranteed invalid column access.
    # --------------------------------------------------------

    faulty_code = (
        code
        + "\n"
        + "# TEST-ONLY ARTIFICIAL FAULT\n"
        + 'test_fault_trigger = df["Wrong Profit"]\n'
    )

    return faulty_code


# ============================================================
# PATCH TEST FAULT INJECTION
# ============================================================
#
# execute_self_correcting_step() internally looks up
# inject_test_fault from the self_correcting_step_executor
# module.
#
# We temporarily replace that function with our stronger
# test-only injector.
#
# Production source code remains unchanged.
# ============================================================

self_correcting_step_executor.inject_test_fault = (
    inject_test_fault_for_all_steps
)


# ============================================================
# STEP EXECUTOR FACTORY
# ============================================================

def create_step_executor(
    data_path,
    test_fault=True
):

    def step_executor(
        step_number,
        step_text
    ):

        return execute_self_correcting_step(
            step_number=step_number,
            step_text=step_text,
            filepath=data_path,
            test_fault=test_fault
        )

    return step_executor


# ============================================================
# ATTEMPT HISTORY HELPERS
# ============================================================

def get_attempt_execution_status(attempt):

    if not isinstance(attempt, dict):
        return None

    if "execution_status" in attempt:
        return attempt.get(
            "execution_status"
        )

    execution = attempt.get(
        "execution"
    )

    if isinstance(execution, dict):

        return execution.get(
            "status"
        )

    execution_result = attempt.get(
        "execution_result"
    )

    if isinstance(
        execution_result,
        dict
    ):

        return execution_result.get(
            "status"
        )

    return None


def get_attempt_validation_status(attempt):

    if not isinstance(attempt, dict):
        return None

    if "validation_status" in attempt:
        return attempt.get(
            "validation_status"
        )

    validation = attempt.get(
        "validation"
    )

    if isinstance(validation, dict):

        return (
            validation.get("overall")
            or validation.get("status")
        )

    validation_result = attempt.get(
        "validation_result"
    )

    if isinstance(
        validation_result,
        dict
    ):

        return (
            validation_result.get("overall")
            or validation_result.get("status")
        )

    return None


# ============================================================
# ATTEMPT HISTORY DISPLAY
# ============================================================

def print_attempt_history(
    step_results
):

    print_section(
        "STEP 6 - SELF-CORRECTION ATTEMPT HISTORY"
    )

    for step_record in step_results:

        print(
            f"\nStep {step_record.get('step_number')}: "
            f"{step_record.get('step')}"
        )

        print(
            f"Final Status: "
            f"{step_record.get('status')}"
        )

        print(
            f"Total Attempts: "
            f"{step_record.get('total_attempts')}"
        )

        print(
            f"Execution Status: "
            f"{step_record.get('execution_status')}"
        )

        print(
            f"Validation Status: "
            f"{step_record.get('validation_status')}"
        )

        print("\nAttempt History:")

        attempts = step_record.get(
            "attempt_history",
            step_record.get(
                "attempts",
                []
            )
        )

        for index, attempt in enumerate(
            attempts,
            start=1
        ):

            execution_status = (
                get_attempt_execution_status(
                    attempt
                )
            )

            validation_status = (
                get_attempt_validation_status(
                    attempt
                )
            )

            print(
                f"  Attempt {index}"
            )

            print(
                f"    Execution: "
                f"{execution_status}"
            )

            print(
                f"    Validation: "
                f"{validation_status}"
            )


# ============================================================
# RECOVERY VERIFICATION
# ============================================================

def verify_recovery(
    step_results
):

    recovered_steps = 0

    for step_record in step_results:

        attempts = step_record.get(
            "attempt_history",
            step_record.get(
                "attempts",
                []
            )
        )

        had_failure = False
        had_success_after_failure = False

        for attempt in attempts:

            execution_status = (
                get_attempt_execution_status(
                    attempt
                )
            )

            validation_status = (
                get_attempt_validation_status(
                    attempt
                )
            )

            if (
                execution_status == "FAIL"
                or validation_status == "FAIL"
            ):

                had_failure = True

            elif (
                had_failure
                and execution_status == "SUCCESS"
                and validation_status == "PASS"
            ):

                had_success_after_failure = True

        if had_success_after_failure:

            recovered_steps += 1

    total_steps = len(
        step_results
    )

    recovery_detected = (
        recovered_steps == total_steps
        and total_steps > 0
    )

    return {
        "recovered_steps": recovered_steps,
        "total_steps": total_steps,
        "recovery_detected": recovery_detected
    }


# ============================================================
# EVALUATION METRICS
# ============================================================

def calculate_evaluation_metrics(
    execution_result,
    recovery_result
):

    total_steps = execution_result.get(
        "total_steps",
        0
    )

    successful_steps = execution_result.get(
        "successful_steps",
        0
    )

    failed_steps = execution_result.get(
        "failed_steps",
        0
    )

    total_attempts = execution_result.get(
        "total_attempts",
        0
    )

    recovered_steps = recovery_result.get(
        "recovered_steps",
        0
    )

    correction_rate = (
        (
            recovered_steps
            / total_steps
        ) * 100
        if total_steps > 0
        else 0
    )

    success_rate = (
        (
            successful_steps
            / total_steps
        ) * 100
        if total_steps > 0
        else 0
    )

    average_attempts = (
        total_attempts
        / total_steps
        if total_steps > 0
        else 0
    )

    return {

        "total_steps":
            total_steps,

        "successful_steps":
            successful_steps,

        "failed_steps":
            failed_steps,

        "total_attempts":
            total_attempts,

        "recovered_steps":
            recovered_steps,

        "correction_rate":
            correction_rate,

        "success_rate":
            success_rate,

        "average_attempts":
            average_attempts
    }


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # TEST TITLE
    # ========================================================

    print_section(
        "TEST 3D - FULL SELF-CORRECTION EVALUATION"
    )

    print(
        "Dataset:"
    )

    print(
        DATASET_PATH
    )

    print(
        "\nUser Question:"
    )

    print(
        USER_QUESTION
    )


    # ========================================================
    # STEP 1 - DATA PROFILING
    # ========================================================

    print_section(
        "STEP 1 - DATA PROFILING"
    )

    profile_result = profile_dataset(
        DATASET_PATH
    )

    data_profile = normalize_profile(
        profile_result
    )

    print(
        "Data Profile Generated Successfully."
    )


    # ========================================================
    # STEP 2 - TASK ANALYSIS
    # ========================================================

    print_section(
        "STEP 2 - TASK ANALYSIS"
    )

    task_analysis = analyze_task_mock(
        USER_QUESTION,
        data_profile
    )

    print(
        "Task Analysis:"
    )

    print(
        task_analysis
    )


    # ========================================================
    # STEP 3 - EDA PLAN
    # ========================================================

    print_section(
        "STEP 3 - EDA PLAN"
    )

    plan = generate_plan_mock(
        USER_QUESTION,
        task_analysis,
        data_profile
    )

    print(
        "Generated Plan:"
    )

    for index, step in enumerate(
        plan,
        start=1
    ):

        print(
            f"{index}. {step}"
        )


    # ========================================================
    # STEP 4 - CONTROLLED FAULT INJECTION
    # ========================================================

    print_section(
        "STEP 4 - CONTROLLED FAULT INJECTION"
    )

    print(
        "TEST MODE ENABLED"
    )

    print(
        "\nEach EDA step will receive "
        "an intentional controlled fault."
    )

    print(
        "\nFault injection is guaranteed "
        "for all 5 steps."
    )

    print(
        "\nExpected behavior:"
    )

    print(
        "Failure -> Validation -> "
        "Correction -> Retry -> PASS"
    )


    # ========================================================
    # STEP 5 - FULL SELF-CORRECTING EXECUTION
    # ========================================================

    print_section(
        "STEP 5 - FULL SELF-CORRECTING EXECUTION"
    )

    step_executor = create_step_executor(
        DATASET_PATH,
        test_fault=True
    )

    execution_result = execute_plan_steps(
        plan,
        step_executor
    )

    print(
        "\nOverall Execution Status:",
        execution_result.get(
            "overall_status"
        )
    )

    print(
        "Total Steps:",
        execution_result.get(
            "total_steps"
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

    print(
        "Total Attempts:",
        execution_result.get(
            "total_attempts"
        )
    )

    step_results = execution_result.get(
        "steps",
        []
    )


    # ========================================================
    # STEP 6 - ATTEMPT HISTORY
    # ========================================================

    print_attempt_history(
        step_results
    )


    # ========================================================
    # STEP 7 - RECOVERY VERIFICATION
    # ========================================================

    print_section(
        "STEP 7 - SELF-CORRECTION VERIFICATION"
    )

    recovery_result = verify_recovery(
        step_results
    )

    print(
        "Recovered Steps:",
        recovery_result[
            "recovered_steps"
        ]
    )

    print(
        "Total Steps:",
        recovery_result[
            "total_steps"
        ]
    )

    print(
        "Recovery detected:",
        recovery_result[
            "recovery_detected"
        ]
    )

    all_steps_passed = (
        execution_result.get(
            "successful_steps",
            0
        )
        ==
        execution_result.get(
            "total_steps",
            0
        )
    )

    print(
        "\nAll steps passed after correction:",
        all_steps_passed
    )


    # ========================================================
    # STEP 8 - EVALUATION METRICS
    # ========================================================

    print_section(
        "STEP 8 - EVALUATION METRICS"
    )

    metrics = calculate_evaluation_metrics(
        execution_result,
        recovery_result
    )

    print(
        "Total Steps:",
        metrics[
            "total_steps"
        ]
    )

    print(
        "Successful Steps:",
        metrics[
            "successful_steps"
        ]
    )

    print(
        "Failed Steps:",
        metrics[
            "failed_steps"
        ]
    )

    print(
        "Total Attempts:",
        metrics[
            "total_attempts"
        ]
    )

    print(
        "Recovered Steps:",
        metrics[
            "recovered_steps"
        ]
    )

    print(
        "Correction Rate:",
        f"{metrics['correction_rate']:.2f}%"
    )

    print(
        "Success Rate:",
        f"{metrics['success_rate']:.2f}%"
    )

    print(
        "Average Attempts per Step:",
        f"{metrics['average_attempts']:.2f}"
    )


    # ========================================================
    # STEP 9 - RESULT AGGREGATION
    # ========================================================

    print_section(
        "STEP 9 - RESULT AGGREGATION"
    )

    aggregated_result = aggregate_results(
        USER_QUESTION,
        step_results,
        task_analysis
    )

    print(
        "Aggregated Result:"
    )

    print(
        aggregated_result
    )

    aggregation_status = (
        aggregated_result.get(
            "overall_status"
        )
        if isinstance(
            aggregated_result,
            dict
        )
        else None
    )

    print(
        "\nResult Aggregation:",
        aggregation_status
    )


    # ========================================================
    # STEP 10 - INSIGHT GENERATION
    # ========================================================

    print_section(
        "STEP 10 - INSIGHT GENERATION"
    )

    primary_result = normalize_result(
        aggregated_result.get(
            "primary_result"
        )
    )

    insight = generate_insight(
        USER_QUESTION,
        primary_result,
        task_analysis
    )

    print(
        "Generated Insight:"
    )

    print(
        insight
    )


    # ========================================================
    # STEP 11 - INSIGHT VALIDATION
    # ========================================================

    print_section(
        "STEP 11 - INSIGHT VALIDATION"
    )

    insight_validation = validate_insight(
        insight,
        primary_result,
        task_analysis
    )

    print(
        "Insight Validation:"
    )

    print(
        insight_validation
    )

    insight_validation_status = (
        insight_validation.get(
            "overall"
        )
        if isinstance(
            insight_validation,
            dict
        )
        else None
    )

    print(
        "\nInsight Validation:",
        insight_validation_status
    )


    # ========================================================
    # STEP 12 - EVIDENCE MAPPING
    # ========================================================

    print_section(
        "STEP 12 - EVIDENCE MAPPING"
    )

    evidence = create_correlation_evidence(
        result=primary_result,
        x_column="Units Sold",
        y_column="Total Profit",
        method="Pearson"
    )

    print(
        "Evidence:"
    )

    print(
        evidence
    )

    mapped_evidence = map_insight_to_evidence(
        insight,
        evidence
    )

    print(
        "\nMapped Evidence:"
    )

    print(
        mapped_evidence
    )

    evidence_status = (
        mapped_evidence.get(
            "status"
        )
        if isinstance(
            mapped_evidence,
            dict
        )
        else None
    )

    print(
        "\nEvidence Mapping:",
        evidence_status
    )


    # ========================================================
    # STEP 13 - RECOMMENDATION GENERATION
    # ========================================================

    print_section(
        "STEP 13 - RECOMMENDATION GENERATION"
    )

    recommendation = generate_recommendation(
        task_type="relationship_analysis",
        evidence=evidence
    )

    print(
        "Recommendation:"
    )

    print(
        recommendation
    )

    recommendation_status = (
        recommendation.get(
            "status"
        )
        if isinstance(
            recommendation,
            dict
        )
        else None
    )

    print(
        "\nRecommendation Status:",
        recommendation_status
    )


    # ========================================================
    # STEP 14 - FINAL TEST VERIFICATION
    # ========================================================

    print_section(
        "STEP 14 - FINAL TEST VERIFICATION"
    )

    test_pass = (

        # ----------------------------------------------------
        # Execution
        # ----------------------------------------------------

        execution_result.get(
            "overall_status"
        )
        ==
        "PASS"

        and

        # ----------------------------------------------------
        # Steps
        # ----------------------------------------------------

        metrics[
            "total_steps"
        ] > 0

        and

        metrics[
            "successful_steps"
        ]
        ==
        metrics[
            "total_steps"
        ]

        and

        metrics[
            "failed_steps"
        ]
        == 0

        and

        # ----------------------------------------------------
        # Recovery
        # ----------------------------------------------------

        metrics[
            "recovered_steps"
        ]
        ==
        metrics[
            "total_steps"
        ]

        and

        metrics[
            "correction_rate"
        ]
        == 100.0

        and

        recovery_result[
            "recovery_detected"
        ]

        and

        all_steps_passed

        # ----------------------------------------------------
        # Result Aggregation
        # ----------------------------------------------------

        and

        aggregation_status
        ==
        "PASS"

        # ----------------------------------------------------
        # Insight Validation
        # ----------------------------------------------------

        and

        insight_validation_status
        ==
        "PASS"

        # ----------------------------------------------------
        # Evidence Mapping
        # ----------------------------------------------------

        and

        evidence_status
        ==
        "PASS"

        # ----------------------------------------------------
        # Recommendation
        # ----------------------------------------------------

        and

        recommendation_status
        ==
        "PASS"
    )


    # ========================================================
    # FINAL METRICS DISPLAY
    # ========================================================

    print(
        "Total Steps:",
        metrics[
            "total_steps"
        ]
    )

    print(
        "Successful Steps:",
        metrics[
            "successful_steps"
        ]
    )

    print(
        "Failed Steps:",
        metrics[
            "failed_steps"
        ]
    )

    print(
        "Total Attempts:",
        metrics[
            "total_attempts"
        ]
    )

    print(
        "Recovered Steps:",
        metrics[
            "recovered_steps"
        ]
    )

    print(
        "Correction Rate:",
        f"{metrics['correction_rate']:.2f}%"
    )

    print(
        "Success Rate:",
        f"{metrics['success_rate']:.2f}%"
    )

    print(
        "Average Attempts:",
        f"{metrics['average_attempts']:.2f}"
    )

    print(
        "Recovery Detected:",
        recovery_result[
            "recovery_detected"
        ]
    )

    print(
        "All Steps Passed:",
        all_steps_passed
    )

    print(
        "Result Aggregation:",
        aggregation_status
    )

    print(
        "Insight Validation:",
        insight_validation_status
    )

    print(
        "Evidence Mapping:",
        evidence_status
    )

    print(
        "Recommendation:",
        recommendation_status
    )


    # ========================================================
    # FINAL RESULT
    # ========================================================

    print_section(
        "TEST 3D RESULT"
    )

    if test_pass:

        print(
            "TEST 3D RESULT: PASS"
        )

        print(
            "\nFull self-correction evaluation "
            "completed successfully."
        )

        print(
            "\nThe system demonstrated:"
        )

        print(
            "1. Controlled fault injection"
        )

        print(
            "2. Failure detection"
        )

        print(
            "3. Validation feedback"
        )

        print(
            "4. Automatic correction"
        )

        print(
            "5. Successful retry"
        )

        print(
            "6. Result aggregation"
        )

        print(
            "7. Insight validation"
        )

        print(
            "8. Evidence grounding"
        )

        print(
            "9. Recommendation generation"
        )

    else:

        print(
            "TEST 3D RESULT: FAIL"
        )

        print(
            "\nReview the failed stage before "
            "continuing to the next experiment."
        )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()