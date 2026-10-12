import os
import sys

# ---------------------------------------------------------
# Project path setup
# ---------------------------------------------------------

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)


# ---------------------------------------------------------
# Project imports
# ---------------------------------------------------------

from data_profiler import profile_dataset
from task_analyzer import analyze_task_mock
from eda_planner import generate_plan_mock
from multi_step_executor import (
    create_self_correcting_executor,
    execute_plan_steps
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "sales.csv"
)

USER_QUESTION = "Which region generated the highest total profit?"


# ---------------------------------------------------------
# Helper
# ---------------------------------------------------------

def print_section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ---------------------------------------------------------
# Main controlled fault-recovery test
# ---------------------------------------------------------

def main():

    print_section("FULL MULTI-STEP SELF-CORRECTION TEST")

    print("Dataset:")
    print(DATA_PATH)

    print()
    print("User Question:")
    print(USER_QUESTION)

    # -----------------------------------------------------
    # 1. Generate Data Profile
    # -----------------------------------------------------

    print_section("STEP 1: DATA PROFILING")

    data_profile = profile_dataset(DATA_PATH)

    print("Data profiling completed.")

    dataset_columns = list(
        data_profile.get("columns", {}).keys()
    )

    print("Dataset columns:")
    print(dataset_columns)

    # -----------------------------------------------------
    # 2. Analyze User Task
    # -----------------------------------------------------

    print_section("STEP 2: TASK ANALYSIS")

    task_analysis = analyze_task_mock(
        USER_QUESTION,
        data_profile
    )

    print("Task Analysis:")
    print(task_analysis)

    # -----------------------------------------------------
    # 3. Generate EDA Plan
    # -----------------------------------------------------

    print_section("STEP 3: EDA PLANNING")

    plan = generate_plan_mock(
        USER_QUESTION,
        task_analysis,
        data_profile
    )

    print("Generated Plan:")

    for index, step in enumerate(plan, start=1):
        print(f"{index}. {step}")

    # -----------------------------------------------------
    # 4. Target columns
    # -----------------------------------------------------

    target_columns = task_analysis.get(
        "target_columns",
        []
    )

    print()
    print("Target Columns:")
    print(target_columns)

    # -----------------------------------------------------
    # 5. Create SELF-CORRECTING executor
    #
    # IMPORTANT:
    # test_fault=True deliberately introduces
    # Wrong Profit during the first attempt.
    # -----------------------------------------------------

    print_section(
        "STEP 4: CREATE SELF-CORRECTING EXECUTOR"
    )

    print(
        "Controlled fault injection: ENABLED"
    )

    print(
        "The first attempt of each step will "
        "contain an intentional error."
    )

    step_executor = create_self_correcting_executor(
        filepath=DATA_PATH,
        data_profile=data_profile,
        dataset_columns=dataset_columns,
        target_columns=target_columns,
        test_fault=True
    )

    # -----------------------------------------------------
    # 6. Execute complete plan
    # -----------------------------------------------------

    print_section(
        "STEP 5: EXECUTE FULL PLAN WITH FAULT INJECTION"
    )

    execution_results = execute_plan_steps(
        plan=plan,
        step_executor=step_executor
    )

    # -----------------------------------------------------
    # 7. Display overall result
    # -----------------------------------------------------

    print_section("OVERALL EXECUTION RESULT")

    print(
        "Overall Status:",
        execution_results.get("overall_status")
    )

    print(
        "Total Steps:",
        execution_results.get("total_steps")
    )

    print(
        "Successful Steps:",
        execution_results.get("successful_steps")
    )

    print(
        "Failed Steps:",
        execution_results.get("failed_steps")
    )

    print(
        "Total Execution Attempts:",
        execution_results.get("total_attempts")
    )

    # -----------------------------------------------------
    # 8. Detailed step results
    # -----------------------------------------------------

    print_section("STEP-BY-STEP RECOVERY RESULTS")

    steps = execution_results.get(
        "steps",
        []
    )

    recovered_steps = 0

    for step in steps:

        step_number = step.get(
            "step_number"
        )

        step_text = step.get(
            "step"
        )

        status = step.get(
            "status"
        )

        attempts = step.get(
            "total_attempts"
        )

        execution_status = step.get(
            "execution_status"
        )

        validation_status = step.get(
            "validation_status"
        )

        print()
        print(
            f"Step {step_number}: {step_text}"
        )

        print(
            "Status:",
            status
        )

        print(
            "Attempts:",
            attempts
        )

        print(
            "Execution:",
            execution_status
        )

        print(
            "Validation:",
            validation_status
        )

        # -------------------------------------------------
        # Recovery detection
        # -------------------------------------------------

        if (
            status == "PASS"
            and attempts > 1
        ):
            recovered_steps += 1

            print(
                "Recovery:",
                "SUCCESSFUL"
            )

        elif status == "PASS":

            print(
                "Recovery:",
                "PASS, but no retry detected"
            )

        else:

            print(
                "Recovery:",
                "FAILED"
            )

        # -------------------------------------------------
        # Attempt history
        # -------------------------------------------------

        attempt_history = step.get(
            "attempt_history",
            []
        )

        if attempt_history:

            print()
            print("Attempt History:")

            for attempt in attempt_history:

                attempt_number = attempt.get(
                    "attempt"
                )

                attempt_execution = attempt.get(
                    "execution",
                    {}
                )

                attempt_validation = attempt.get(
                    "validation",
                    {}
                )

                print(
                    f"  Attempt {attempt_number}:"
                )

                print(
                    "    Execution:",
                    attempt_execution.get(
                        "status"
                    )
                )

                print(
                    "    Validation:",
                    attempt_validation.get(
                        "overall"
                    )
                )

                feedback = attempt.get(
                    "feedback",
                    []
                )

                if feedback:

                    print(
                        "    Feedback:"
                    )

                    for message in feedback:

                        print(
                            "      -",
                            message
                        )

    # -----------------------------------------------------
    # 9. Final test decision
    # -----------------------------------------------------

    total_steps = execution_results.get(
        "total_steps",
        0
    )

    successful_steps = execution_results.get(
        "successful_steps",
        0
    )

    failed_steps = execution_results.get(
        "failed_steps",
        0
    )

    total_attempts = execution_results.get(
        "total_attempts",
        0
    )

    # -----------------------------------------------------
    # Expected behavior:
    #
    # All steps should PASS.
    # At least one retry should happen.
    # -----------------------------------------------------

    test_pass = (
        execution_results.get("overall_status")
        == "PASS"
        and total_steps > 0
        and successful_steps == total_steps
        and failed_steps == 0
        and total_attempts > total_steps
        and recovered_steps > 0
    )

    print_section(
        "SELF-CORRECTION TEST SUMMARY"
    )

    print(
        "Total Steps:",
        total_steps
    )

    print(
        "Successful Steps:",
        successful_steps
    )

    print(
        "Failed Steps:",
        failed_steps
    )

    print(
        "Total Attempts:",
        total_attempts
    )

    print(
        "Recovered Steps:",
        recovered_steps
    )

    if test_pass:

        print()
        print(
            "TEST 2 RESULT: PASS"
        )

        print(
            "The multi-step self-correction "
            "mechanism successfully detected "
            "and recovered from injected faults."
        )

        print()
        print(
            "OVERALL RESULT: PASS"
        )

    else:

        print()
        print(
            "TEST 2 RESULT: FAIL"
        )

        print(
            "The complete multi-step "
            "self-correction mechanism requires "
            "further debugging."
        )

        print()
        print(
            "OVERALL RESULT: FAIL"
        )


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------

if __name__ == "__main__":
    main()