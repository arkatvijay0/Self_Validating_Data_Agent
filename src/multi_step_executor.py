"""
multi_step_executor.py

Multi-step EDA controller using the validation-feedback-based
self-correcting step executor.

Workflow:

    EDA Plan
       ↓
    Step 1 → Generate → Execute → Validate → Correct if needed
       ↓
    Step 2 → Generate → Execute → Validate → Correct if needed
       ↓
    Step 3 → Generate → Execute → Validate → Correct if needed
       ↓
    ...
       ↓
    Aggregate all step results
"""

import os

from self_correcting_step_executor import (
    execute_self_correcting_step
)


# =========================================================
# STEP RECORD
# =========================================================

def create_step_record(
    step_number,
    step_text,
    result
):
    """
    Create a standardized record for one EDA step.
    """

    final_execution = result.get(
        "final_execution"
    ) or {}

    final_validation = result.get(
        "final_validation"
    ) or {}

    return {
        "step_number": step_number,
        "step": step_text,
        "status": result.get(
            "status",
            "UNRESOLVED"
        ),
        "total_attempts": result.get(
            "total_attempts",
            0
        ),
        "final_code": result.get(
            "final_code"
        ),
        "final_result": result.get(
            "final_result"
        ),
        "execution_status": final_execution.get(
            "status"
        ),
        "validation_status": final_validation.get(
            "overall"
        ),
        "attempt_history": result.get(
            "attempts",
            []
        ),
        "raw_result": result
    }


# =========================================================
# EXECUTE ALL PLAN STEPS
# =========================================================

def execute_plan_steps(
    plan,
    step_executor
):
    """
    Execute every EDA step in the supplied plan.

    Parameters
    ----------
    plan : list
        EDA plan steps.

    step_executor : callable
        Function used to execute each step.

    Returns
    -------
    dict
        Multi-step execution summary.
    """

    if not plan:

        return {
            "overall_status": "FAIL",
            "total_steps": 0,
            "successful_steps": 0,
            "failed_steps": 0,
            "total_attempts": 0,
            "steps": [],
            "message": "EDA plan is empty."
        }

    step_results = []

    total_attempts = 0

    # =====================================================
    # EXECUTE EACH STEP
    # =====================================================

    for index, step in enumerate(
        plan,
        start=1
    ):

        # -------------------------------------------------
        # Handle both strings and dictionary steps
        # -------------------------------------------------

        if isinstance(step, dict):

            step_text = (
                step.get("step")
                or
                step.get("description")
                or
                step.get("task")
                or
                str(step)
            )

        else:

            step_text = str(step)

        print("\n")
        print("=" * 70)
        print(
            f"EXECUTING EDA PLAN STEP {index}"
        )
        print("=" * 70)

        print("\nStep:")
        print(step_text)

        # -------------------------------------------------
        # Execute the step
        # -------------------------------------------------

        result = step_executor(
            step_number=index,
            step_text=step_text
        )

        # -------------------------------------------------
        # Create standardized record
        # -------------------------------------------------

        step_record = create_step_record(
            step_number=index,
            step_text=step_text,
            result=result
        )

        step_results.append(
            step_record
        )

        total_attempts += (
            step_record["total_attempts"]
        )

        print("\nStep Status:")
        print(
            step_record["status"]
        )

        print("\nExecution Status:")
        print(
            step_record["execution_status"]
        )

        print("\nValidation Status:")
        print(
            step_record["validation_status"]
        )

        print("\nTotal Attempts:")
        print(
            step_record["total_attempts"]
        )

    # =====================================================
    # SUMMARY
    # =====================================================

    successful_steps = sum(
        1
        for step in step_results
        if step["status"] == "PASS"
    )

    failed_steps = (
        len(step_results)
        - successful_steps
    )

    if (
        len(step_results) == len(plan)
        and
        failed_steps == 0
    ):

        overall_status = "PASS"

    else:

        overall_status = "FAIL"

    return {
        "overall_status": overall_status,
        "total_steps": len(
            step_results
        ),
        "successful_steps":
            successful_steps,
        "failed_steps":
            failed_steps,
        "total_attempts":
            total_attempts,
        "steps":
            step_results
    }


# =========================================================
# CREATE SELF-CORRECTING EXECUTOR
# =========================================================

def create_self_correcting_executor(
    filepath,
    data_profile=None,
    dataset_columns=None,
    target_columns=None,
    test_fault=False
):
    """
    Create an executor function that automatically
    provides dataset context to each EDA step.
    """

    if dataset_columns is None:

        import pandas as pd

        if not os.path.exists(filepath):

            raise FileNotFoundError(
                f"Dataset not found: {filepath}"
            )

        df = pd.read_csv(filepath)

        dataset_columns = list(
            df.columns
        )

    if target_columns is None:

        target_columns = []

    def step_executor(
        step_number,
        step_text
    ):
        """
        Execute one EDA step using the
        self-correcting executor.
        """

        return execute_self_correcting_step(
            step_number=step_number,
            step_text=step_text,
            filepath=filepath,
            data_profile=data_profile,
            dataset_columns=dataset_columns,
            target_columns=target_columns,
            test_fault=test_fault
        )

    return step_executor


# =========================================================
# PRINT SUMMARY
# =========================================================

def print_execution_summary(
    execution_summary
):
    """
    Print a concise multi-step execution summary.
    """

    print("\n")
    print("#" * 70)
    print("MULTI-STEP EDA EXECUTION SUMMARY")
    print("#" * 70)

    print("\nOverall Status:")
    print(
        execution_summary[
            "overall_status"
        ]
    )

    print("\nTotal Steps:")
    print(
        execution_summary[
            "total_steps"
        ]
    )

    print("\nSuccessful Steps:")
    print(
        execution_summary[
            "successful_steps"
        ]
    )

    print("\nFailed Steps:")
    print(
        execution_summary[
            "failed_steps"
        ]
    )

    print("\nTotal Execution Attempts:")
    print(
        execution_summary[
            "total_attempts"
        ]
    )

    print("\nStep Summary:")
    print("-" * 70)

    for step in execution_summary[
        "steps"
    ]:

        print(
            f"Step {step['step_number']}: "
            f"{step['status']} | "
            f"Attempts: {step['total_attempts']} | "
            f"Execution: {step['execution_status']} | "
            f"Validation: {step['validation_status']}"
        )


# =========================================================
# TEST 1
# NORMAL MULTI-STEP EXECUTION
# =========================================================

def test_normal_multi_step_execution():

    print("\n")
    print("#" * 70)
    print(
        "TEST 1: NORMAL MULTI-STEP "
        "SELF-CORRECTING EXECUTION"
    )
    print("#" * 70)

    filepath = (
        r"C:\Users\Vijay\Desktop"
        r"\Self_Validating_Data_Agent"
        r"\data\raw\sales.csv"
    )

    dataset_columns = [
        "Region",
        "Country",
        "Item Type",
        "Sales Channel",
        "Order Priority",
        "Order Date",
        "Order ID",
        "Ship Date",
        "Units Sold",
        "Unit Price",
        "Unit Cost",
        "Total Revenue",
        "Total Cost",
        "Total Profit"
    ]

    plan = [
        (
            "Group the data by Region and calculate "
            "the total of Total Profit."
        ),
        (
            "Rank the Region categories according "
            "to total Total Profit."
        ),
        (
            "Compare the distribution of Total Profit "
            "across Region categories."
        ),
        (
            "Check for unusually high or low "
            "Total Profit values within each Region."
        ),
        (
            "Identify the Region category with the "
            "highest total Total Profit."
        )
    ]

    executor = create_self_correcting_executor(
        filepath=filepath,
        data_profile=None,
        dataset_columns=dataset_columns,
        target_columns=[
            "Region",
            "Total Profit"
        ],
        test_fault=False
    )

    execution_summary = execute_plan_steps(
        plan=plan,
        step_executor=executor
    )

    print_execution_summary(
        execution_summary
    )

    if (
        execution_summary[
            "overall_status"
        ] == "PASS"
        and
        execution_summary[
            "successful_steps"
        ] == 5
    ):

        print(
            "\nTEST 1 RESULT: PASS"
        )

        return True

    print(
        "\nTEST 1 RESULT: FAIL"
    )

    return False


# =========================================================
# TEST 2
# CONTROLLED FAULT RECOVERY
# =========================================================

def test_multi_step_fault_recovery():

    print("\n")
    print("#" * 70)
    print(
        "TEST 2: MULTI-STEP FAULT RECOVERY"
    )
    print("#" * 70)

    filepath = (
        r"C:\Users\Vijay\Desktop"
        r"\Self_Validating_Data_Agent"
        r"\data\raw\sales.csv"
    )

    dataset_columns = [
        "Region",
        "Country",
        "Item Type",
        "Sales Channel",
        "Order Priority",
        "Order Date",
        "Order ID",
        "Ship Date",
        "Units Sold",
        "Unit Price",
        "Unit Cost",
        "Total Revenue",
        "Total Cost",
        "Total Profit"
    ]

    plan = [
        (
            "Group the data by Region and calculate "
            "the total of Total Profit."
        ),
        (
            "Rank the Region categories according "
            "to total Total Profit."
        ),
        (
            "Compare the distribution of Total Profit "
            "across Region categories."
        ),
        (
            "Check for unusually high or low "
            "Total Profit values within each Region."
        ),
        (
            "Identify the Region category with the "
            "highest total Total Profit."
        )
    ]

    executor = create_self_correcting_executor(
        filepath=filepath,
        data_profile=None,
        dataset_columns=dataset_columns,
        target_columns=[
            "Region",
            "Total Profit"
        ],
        test_fault=True
    )

    execution_summary = execute_plan_steps(
        plan=plan,
        step_executor=executor
    )

    print_execution_summary(
        execution_summary
    )

    # -----------------------------------------------------
    # Every step should recover from the artificial fault.
    # -----------------------------------------------------

    all_steps_recovered = all(
        step["status"] == "PASS"
        and step["total_attempts"] >= 2
        for step in execution_summary[
            "steps"
        ]
    )

    if (
        execution_summary[
            "overall_status"
        ] == "PASS"
        and
        all_steps_recovered
    ):

        print(
            "\nTEST 2 RESULT: PASS"
        )

        return True

    print(
        "\nTEST 2 RESULT: FAIL"
    )

    return False


# =========================================================
# MAIN TEST RUNNER
# =========================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 70)
    print(
        "MULTI-STEP SELF-CORRECTING EDA CONTROLLER"
    )
    print("=" * 70)

    test_results = []

    # -----------------------------------------------------
    # Production mode
    # -----------------------------------------------------

    test_results.append(
        test_normal_multi_step_execution()
    )

    # -----------------------------------------------------
    # Controlled fault recovery mode
    # -----------------------------------------------------

    test_results.append(
        test_multi_step_fault_recovery()
    )

    # -----------------------------------------------------
    # Final test summary
    # -----------------------------------------------------

    print("\n")
    print("=" * 70)
    print("FINAL TEST SUMMARY")
    print("=" * 70)

    passed = sum(
        test_results
    )

    total = len(
        test_results
    )

    print(
        f"\nTests Passed: {passed}/{total}"
    )

    if passed == total:

        print(
            "\nOVERALL RESULT: PASS"
        )

    else:

        print(
            "\nOVERALL RESULT: FAIL"
        )