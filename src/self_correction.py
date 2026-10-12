import os
import sys
import pandas as pd


# ============================================================
# IMPORT PROJECT MODULES
# ============================================================

SRC_DIR = os.path.dirname(os.path.abspath(__file__))

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from code_generator import generate_code_mock
from validators import self_validate


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(SRC_DIR)

DATA_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "sales.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

MAX_RETRIES = 3


# ============================================================
# CONTROLLED FAULT INJECTION
# ============================================================

def generate_faulty_code(
    step_text,
    filepath,
    data_profile,
    attempt
):
    """
    Generate intentionally faulty code on the first attempt.

    This is used only for testing whether the
    self-correction mechanism can detect and recover
    from an incorrect generated program.
    """

    if attempt == 1:

        print("\n[FAULT INJECTION]")
        print("Generating intentionally incorrect code...")
        print("Incorrect column: Wrong Profit")

        return f'''import pandas as pd
import numpy as np

df = pd.read_csv(r"{filepath}")

result = df.groupby("Region")["Wrong Profit"].sum().sort_values(
    ascending=False
)
'''

    # After the first failed attempt, use the real
    # code generator to produce corrected code.

    print("\n[CORRECTION]")
    print("Calling the real code generator...")

    return generate_code_mock(
        step_text,
        filepath,
        data_profile
    )


# ============================================================
# CODE EXECUTOR
# ============================================================

def execute_generated_code(code):
    """
    Execute generated Python code and return the result.

    The execution result contains:
        status  -> SUCCESS or FAILED
        success -> True or False
        stdout
        stderr
        result
    """

    execution_namespace = {}

    try:

        exec(
            code,
            execution_namespace
        )

        result = execution_namespace.get("result")

        return {
            "status": "SUCCESS",
            "success": True,
            "stdout": "",
            "stderr": "",
            "result": result
        }

    except Exception as error:

        return {
            "status": "FAILED",
            "success": False,
            "stdout": "",
            "stderr": (
                f"{type(error).__name__}: {error}"
            ),
            "result": None
        }


# ============================================================
# COLLECT VALIDATION FEEDBACK
# ============================================================

def collect_validation_feedback(validation_result):
    """
    Extract useful feedback from failed validation checks.
    """

    feedback = []

    checks = validation_result.get(
        "checks",
        {}
    )

    for check_name, check_result in checks.items():

        if not check_result.get("pass", False):

            reason = check_result.get(
                "reason",
                "Unknown validation error"
            )

            feedback.append(
                f"{check_name} validation failed: {reason}"
            )

    return feedback


# ============================================================
# SELF-CORRECTION LOOP
# ============================================================

def run_self_correction(
    step_text,
    filepath,
    data_profile,
    dataset_columns,
    target_columns
):
    """
    Test the complete self-correction process.

    Attempt 1:
        Intentionally incorrect code

    Attempt 2:
        Real code generator creates corrected code

    Validation:
        Code + Data + Result
    """

    attempts = []

    feedback = None

    for attempt in range(1, MAX_RETRIES + 1):

        print("\n" + "=" * 60)
        print(f"SELF-CORRECTION ATTEMPT {attempt}")
        print("=" * 60)

        # ----------------------------------------------------
        # Generate code
        # ----------------------------------------------------

        generated_code = generate_faulty_code(
            step_text,
            filepath,
            data_profile,
            attempt
        )

        print("\nGenerated Code:")
        print(generated_code)

        # ----------------------------------------------------
        # Execute
        # ----------------------------------------------------

        print("\nExecuting generated code...")

        execution_result = execute_generated_code(
            generated_code
        )

        print("\nExecution Result:")
        print(execution_result)

        # ----------------------------------------------------
        # Validate
        # ----------------------------------------------------

        print("\nRunning self-validation...")

        validation_result = self_validate(
            execution_result=execution_result,
            code=generated_code,
            dataset_columns=dataset_columns,
            target_columns=target_columns
        )

        print("\nValidation Result:")
        print(validation_result)

        # ----------------------------------------------------
        # Store attempt
        # ----------------------------------------------------

        attempts.append(
            {
                "attempt": attempt,
                "code": generated_code,
                "execution": execution_result,
                "validation": validation_result
            }
        )

        # ----------------------------------------------------
        # PASS
        # ----------------------------------------------------

        if validation_result["overall"] == "PASS":

            print("\n" + "=" * 60)
            print("SELF-CORRECTION SUCCESSFUL")
            print("=" * 60)

            return {
                "status": "PASS",
                "attempts": attempts,
                "final_code": generated_code,
                "final_execution": execution_result,
                "final_validation": validation_result
            }

        # ----------------------------------------------------
        # FAIL
        # ----------------------------------------------------

        print("\nValidation FAILED.")

        feedback_list = collect_validation_feedback(
            validation_result
        )

        feedback = "; ".join(
            feedback_list
        )

        print("\nValidation Feedback:")
        print(feedback)

        # ----------------------------------------------------
        # Show what will happen next
        # ----------------------------------------------------

        if attempt < MAX_RETRIES:

            print(
                "\nThe validation feedback will be used "
                "to correct the next attempt."
            )

    # ========================================================
    # UNRESOLVED
    # ========================================================

    print("\n" + "=" * 60)
    print("SELF-CORRECTION UNRESOLVED")
    print("=" * 60)

    return {
        "status": "UNRESOLVED",
        "attempts": attempts,
        "final_code": attempts[-1]["code"],
        "final_execution": attempts[-1]["execution"],
        "final_validation": attempts[-1]["validation"]
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("CONTROLLED SELF-CORRECTION TEST")
    print("=" * 60)

    # --------------------------------------------------------
    # Check dataset
    # --------------------------------------------------------

    print("\nDataset:")
    print(DATA_FILE)

    if not os.path.exists(DATA_FILE):

        print("\nERROR: sales.csv was not found.")

        sys.exit(1)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = pd.read_csv(DATA_FILE)

    print("\nDataset loaded successfully.")

    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    print("\nDataset Columns:")
    print(list(df.columns))

    # --------------------------------------------------------
    # Data profile
    # --------------------------------------------------------

    data_profile = {
        "columns": {
            "Region": {
                "role": "categorical"
            },
            "Total Profit": {
                "role": "continuous"
            },
            "Total Revenue": {
                "role": "continuous"
            },
            "Units Sold": {
                "role": "continuous"
            }
        }
    }

    # --------------------------------------------------------
    # EDA task
    # --------------------------------------------------------

    step_text = (
        "Group the data by Region and calculate "
        "the total of Total Profit."
    )

    # --------------------------------------------------------
    # Expected columns
    # --------------------------------------------------------

    target_columns = [
        "Region",
        "Total Profit"
    ]

    # --------------------------------------------------------
    # Run controlled self-correction
    # --------------------------------------------------------

    result = run_self_correction(
        step_text=step_text,
        filepath=DATA_FILE,
        data_profile=data_profile,
        dataset_columns=list(df.columns),
        target_columns=target_columns
    )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("FINAL SELF-CORRECTION RESULT")
    print("=" * 60)

    print("\nStatus:")
    print(result["status"])

    print("\nNumber of Attempts:")
    print(len(result["attempts"]))

    print("\nFinal Validation:")
    print(result["final_validation"])

    print("\nFinal Execution Status:")
    print(
        result["final_execution"].get(
            "status",
            "UNKNOWN"
        )
    )

    print("\nFinal Result:")
    print(result["final_execution"]["result"])

    # --------------------------------------------------------
    # Attempt summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("ATTEMPT SUMMARY")
    print("=" * 60)

    for attempt in result["attempts"]:

        print(
            f"\nAttempt {attempt['attempt']}: "
            f"{attempt['validation']['overall']}"
        )

        print(
            "Execution Status:",
            attempt["execution"].get(
                "status",
                "UNKNOWN"
            )
        )

        print(
            "Validation:",
            attempt["validation"]
        )

    print("\n" + "=" * 60)
    print("CONTROLLED SELF-CORRECTION TEST COMPLETED")
    print("=" * 60)