"""
self_correcting_step_executor.py

Self-correcting execution engine for individual EDA steps.

Workflow:

    Generate Code
          ↓
    Execute Code
          ↓
    Validate
          ↓
        PASS?
       /     \
     YES      NO
      ↓        ↓
   Finish   Collect Feedback
                ↓
          Code Corrector
                ↓
          Corrected Code
                ↓
             Execute
                ↓
             Validate
                ↓
           PASS / Retry

The production workflow does NOT inject artificial errors.

An optional test mode is provided only to demonstrate
the self-correction mechanism.
"""


# =========================================================
# IMPORTS
# =========================================================

import os
import sys
import traceback

import pandas as pd


# =========================================================
# PROJECT MODULES
# =========================================================

from code_generator import generate_code_mock
from validators import self_validate
from code_corrector import correct_generated_code


# =========================================================
# CONFIGURATION
# =========================================================

MAX_RETRIES = 3


# =========================================================
# DATASET LOADER
# =========================================================

def load_dataset(filepath):
    """
    Load the dataset using pandas.

    Parameters
    ----------
    filepath : str
        Path to CSV dataset.

    Returns
    -------
    pandas.DataFrame
        Loaded dataset.
    """

    if not filepath:

        raise ValueError(
            "Dataset filepath was not provided."
        )

    if not os.path.exists(filepath):

        raise FileNotFoundError(
            f"Dataset not found: {filepath}"
        )

    df = pd.read_csv(filepath)

    return df


# =========================================================
# CODE GENERATION
# =========================================================

def generate_step_code(
    step_text,
    filepath,
    data_profile=None
):
    """
    Generate Python code for an EDA step.

    This function performs NORMAL production
    code generation.

    No artificial fault is injected here.
    """

    code = generate_code_mock(
        step_text,
        filepath,
        data_profile
    )

    if not code:

        raise ValueError(
            "Code generator returned empty code."
        )

    return code


# =========================================================
# TEST FAULT INJECTION
# =========================================================


def inject_test_fault(code):
    """
    Inject a deliberate execution error for testing only.

    The generated code is changed so its first execution fails.
    This function is used only when test_fault=True.
    """
    if not code:
        return code

    # Prefer corrupting a dataset column reference when one exists.
    if 'df["' in code:
        return code.replace(
            'df["',
            'df["__TEST_INVALID_COLUMN__',
            1,
        )

    if "df['" in code:
        return code.replace(
            "df['",
            "df['__TEST_INVALID_COLUMN__",
            1,
        )

    # For generic operations such as row counts, append an
    # intentional NameError after the generated result.
    return (
        code
        + "\n# Test-only deliberate execution failure\n"
        + "__self_correction_test_fault__()\n"
    )



# =========================================================
# EXECUTE GENERATED CODE
# =========================================================

def execute_generated_code(
    code,
    filepath
):
    """
    Execute generated Python code.

    The dataset is loaded into the variable `df`.

    Returns a standardized execution result.
    """

    if not code:

        return {
            "status": "FAIL",
            "success": False,
            "stdout": "",
            "stderr": "Generated code is empty.",
            "result": None
        }

    try:

        df = load_dataset(
            filepath
        )

        execution_namespace = {
            "pd": pd,
            "df": df
        }

        exec(
            code,
            execution_namespace
        )

        result = execution_namespace.get(
            "result"
        )

        return {
            "status": "SUCCESS",
            "success": True,
            "stdout": "CODE EXECUTION SUCCESSFUL",
            "stderr": "",
            "result": result
        }

    except Exception as error:

        return {
            "status": "FAIL",
            "success": False,
            "stdout": "",
            "stderr": f"{type(error).__name__}: {error}",
            "result": None
        }


# =========================================================
# COLLECT VALIDATION FEEDBACK
# =========================================================

def collect_validation_feedback(
    validation_result
):
    """
    Convert validation results into structured
    feedback for the code corrector.
    """

    if not validation_result:

        return {
            "overall": "FAIL",
            "feedback": {
                "general": {
                    "pass": False,
                    "reason":
                        "Validation result is empty."
                }
            }
        }

    feedback = {}

    checks = validation_result.get(
        "checks",
        {}
    )

    for check_name, check_result in checks.items():

        if isinstance(
            check_result,
            dict
        ):

            feedback[check_name] = {
                "pass":
                    check_result.get(
                        "pass",
                        False
                    ),

                "reason":
                    check_result.get(
                        "reason"
                    )
            }

        else:

            feedback[check_name] = {
                "pass": bool(check_result),
                "reason": None
            }

    feedback["overall"] = validation_result.get(
        "overall",
        "FAIL"
    )

    return feedback


# =========================================================
# BUILD DATA PROFILE COLUMNS
# =========================================================

def get_dataset_columns(filepath):
    """
    Return dataset column names.
    """

    df = load_dataset(
        filepath
    )

    return list(
        df.columns
    )


# =========================================================
# EXTRACT STATISTICAL DATA
# =========================================================

def extract_statistical_data(
    filepath,
    step_text,
    target_columns=None
):
    """
    Extract the actual dataset series required for
    statistical validation.

    This is separate from the EDA result because
    statistical results such as Pearson correlation
    return a scalar value rather than the original
    input series.

    Parameters
    ----------
    filepath : str
        Dataset path.

    step_text : str
        Current EDA step.

    target_columns : list
        Expected target columns.

    Returns
    -------
    pandas.Series or None
        Series to be used by statistical validation.
    """

    try:

        df = load_dataset(
            filepath
        )

    except Exception:

        return None

    if target_columns is None:

        target_columns = []

    step_lower = step_text.lower()

    # -----------------------------------------------------
    # Pearson correlation
    # -----------------------------------------------------

    if "pearson" in step_lower:

        # Prefer Units Sold because it is the first
        # variable in the relationship question.

        if "Units Sold" in df.columns:

            return df["Units Sold"]

        # Fallback to first numeric target column.

        for column in target_columns:

            if column in df.columns:

                if pd.api.types.is_numeric_dtype(
                    df[column]
                ):

                    return df[column]

        return None

    # -----------------------------------------------------
    # T-test
    # -----------------------------------------------------

    if "t-test" in step_lower:

        for column in target_columns:

            if column in df.columns:

                if pd.api.types.is_numeric_dtype(
                    df[column]
                ):

                    return df[column]

        return None

    return None


# =========================================================
# IDENTIFY STATISTICAL TEST
# =========================================================

def identify_statistical_test(
    step_text
):
    """
    Identify whether the current EDA step uses
    a statistical method supported by the validator.
    """

    step_lower = step_text.lower()

    if "pearson" in step_lower:

        return "pearson_correlation"

    if "t-test" in step_lower:

        return "t-test"

    return None


# =========================================================
# EXECUTE ONE SELF-CORRECTING STEP
# =========================================================

def execute_self_correcting_step(
    step_number,
    step_text,
    filepath,
    data_profile=None,
    dataset_columns=None,
    target_columns=None,
    test_fault=False
):
    """
    Execute one EDA step using a real
    validation-feedback-based self-correction loop.

    Parameters
    ----------
    step_number : int
        EDA step number.

    step_text : str
        EDA instruction.

    filepath : str
        Dataset path.

    data_profile : dict, optional
        Dataset profile.

    dataset_columns : list, optional
        Dataset column names.

    target_columns : list, optional
        Expected target columns.

    test_fault : bool
        If True, inject an artificial error on the
        first attempt for demonstration/testing only.

    Returns
    -------
    dict
        Complete execution history.
    """

    # =====================================================
    # PREPARE DATA INFORMATION
    # =====================================================

    if dataset_columns is None:

        dataset_columns = get_dataset_columns(
            filepath
        )

    if target_columns is None:

        target_columns = []

    attempt_history = []

    current_code = None

    final_execution = None
    final_validation = None
    final_result = None

    # =====================================================
    # ATTEMPT LOOP
    # =====================================================

    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):

        print(
            "\n" + "-" * 70
        )

        print(
            f"STEP {step_number} | "
            f"ATTEMPT {attempt}"
        )

        print(
            "-" * 70
        )

        # =================================================
        # ATTEMPT 1
        # Generate fresh code
        # =================================================

        if attempt == 1:

            try:

                current_code = generate_step_code(
                    step_text=step_text,
                    filepath=filepath,
                    data_profile=data_profile
                )

            except Exception as error:

                attempt_record = {

                    "attempt": attempt,

                    "code": None,

                    "execution": {
                        "status": "FAIL",
                        "success": False,
                        "stdout": "",
                        "stderr": str(error),
                        "result": None
                    },

                    "validation": {
                        "overall": "FAIL",

                        "checks": {
                            "code_generation": {
                                "pass": False,
                                "reason": str(error)
                            }
                        }
                    },

                    "feedback": {
                        "overall": "FAIL",

                        "code_generation": {
                            "pass": False,
                            "reason": str(error)
                        }
                    },

                    "correction": None
                }

                attempt_history.append(
                    attempt_record
                )

                final_execution = (
                    attempt_record["execution"]
                )

                final_validation = (
                    attempt_record["validation"]
                )

                final_result = None

                continue

            # -------------------------------------------------
            # Optional test-only fault injection
            # -------------------------------------------------

            if test_fault:

                current_code = inject_test_fault(
                    current_code
                )

                print(
                    "TEST MODE: Artificial fault "
                    "injected into generated code."
                )

        # =================================================
        # ATTEMPT > 1
        # current_code already contains corrected code
        # =================================================

        print(
            "\nGenerated/current code:"
        )

        print(
            current_code
        )

        # =================================================
        # EXECUTION
        # =================================================

        execution_result = execute_generated_code(
            code=current_code,
            filepath=filepath
        )

        print(
            "\nExecution status:"
        )

        print(
            execution_result["status"]
        )

        if execution_result["stderr"]:

            print(
                "Execution error:"
            )

            print(
                execution_result["stderr"]
            )

        # =================================================
        # VALIDATION
        # =================================================

        # -------------------------------------------------
        # Identify statistical test
        # -------------------------------------------------

        test_used = identify_statistical_test(
            step_text
        )

        # -------------------------------------------------
        # IMPORTANT FIX
        #
        # Statistical validation must use the original
        # dataset values, not the statistical result.
        #
        # Example:
        #
        # Pearson result:
        #     0.5645
        #
        # But normality validation needs:
        #     df["Units Sold"]
        #
        # -------------------------------------------------

        data_series = None

        if test_used is not None:

            data_series = extract_statistical_data(
                filepath=filepath,
                step_text=step_text,
                target_columns=target_columns
            )

        # =================================================
        # SELF VALIDATION
        # =================================================

        validation_result = self_validate(
            execution_result=execution_result,

            code=current_code,

            dataset_columns=dataset_columns,

            target_columns=target_columns,

            test_used=test_used,

            data_series=data_series
        )

        print(
            "\nValidation status:"
        )

        print(
            validation_result.get(
                "overall"
            )
        )

        # =================================================
        # SUCCESS
        # =================================================

        if validation_result.get(
            "overall"
        ) == "PASS":

            final_execution = (
                execution_result
            )

            final_validation = (
                validation_result
            )

            final_result = (
                execution_result.get(
                    "result"
                )
            )

            attempt_record = {

                "attempt": attempt,

                "code": current_code,

                "execution": execution_result,

                "validation": validation_result,

                "feedback": None,

                "correction": None
            }

            attempt_history.append(
                attempt_record
            )

            print(
                f"\nSTEP {step_number}: "
                f"SELF-CORRECTION PASS"
            )

            break

        # =================================================
        # VALIDATION FAILURE
        # =================================================

        feedback = collect_validation_feedback(
            validation_result
        )

        # Preserve execution errors for feedback-based correction.
        if not execution_result.get("success", False):
            feedback["execution_error"] = {
                "pass": False,
                "reason": execution_result.get(
                    "stderr",
                    "Execution failed without an error message."
                )
            }

        print(
            "\nValidation feedback:"
        )

        print(
            feedback
        )

        # =================================================
        # ATTEMPT RECORD
        # =================================================

        attempt_record = {

            "attempt": attempt,

            "code": current_code,

            "execution": execution_result,

            "validation": validation_result,

            "feedback": feedback,

            "correction": None
        }

        # =================================================
        # MAX RETRY CHECK
        # =================================================

        if attempt >= MAX_RETRIES:

            attempt_history.append(
                attempt_record
            )

            final_execution = (
                execution_result
            )

            final_validation = (
                validation_result
            )

            final_result = (
                execution_result.get(
                    "result"
                )
            )

            print(
                f"\nSTEP {step_number}: "
                f"SELF-CORRECTION FAILED "
                f"AFTER {MAX_RETRIES} ATTEMPTS"
            )

            break

        # =================================================
        # CODE CORRECTION
        # =================================================

        print(
            "\nRunning validation-feedback "
            "code correction..."
        )

        correction_result = (
            correct_generated_code(
                step_text=step_text,

                generated_code=current_code,

                feedback=feedback,

                dataset_columns=dataset_columns,

                target_columns=target_columns,

                filepath=filepath,

                data_profile=data_profile,

                attempt_number=attempt + 1
            )
        )

        attempt_record["correction"] = (
            correction_result
        )

        print(
            "\nCorrection status:"
        )

        print(
            correction_result.get(
                "status"
            )
        )

        print(
            "Correction method:"
        )

        print(
            correction_result.get(
                "correction_method"
            )
        )

        if correction_result.get(
            "corrections"
        ):

            print(
                "Corrections:"
            )

            for correction in (
                correction_result[
                    "corrections"
                ]
            ):

                print(
                    correction
                )

        # -------------------------------------------------
        # If no correction possible,
        # stop the loop.
        # -------------------------------------------------

        if correction_result.get(
            "status"
        ) not in ["PASS"]:

            attempt_history.append(
                attempt_record
            )

            final_execution = (
                execution_result
            )

            final_validation = (
                validation_result
            )

            final_result = (
                execution_result.get(
                    "result"
                )
            )

            print(
                "\nNo further correction "
                "could be applied."
            )

            break

        # -------------------------------------------------
        # Use corrected code for next attempt.
        # -------------------------------------------------

        current_code = (
            correction_result[
                "corrected_code"
            ]
        )

        attempt_history.append(
            attempt_record
        )

    # =====================================================
    # FINAL STATUS
    # =====================================================

    if (
        final_validation
        and
        final_validation.get(
            "overall"
        ) == "PASS"
    ):

        final_status = "PASS"

    else:

        final_status = "UNRESOLVED"

    return {

        "status": final_status,

        "step_number": step_number,

        "step": step_text,

        "attempts": attempt_history,

        "total_attempts": len(
            attempt_history
        ),

        "final_code": current_code,

        "final_execution": final_execution,

        "final_validation": final_validation,

        "final_result": final_result
    }


# =========================================================
# TEST 1
# PRODUCTION MODE
# =========================================================

def test_production_mode():

    print("\n")

    print("#" * 70)

    print(
        "TEST 1: PRODUCTION SELF-CORRECTION"
    )

    print("#" * 70)

    filepath = (
        r"C:\Users\Vijay\Desktop"
        r"\Self_Validating_Data_Agent"
        r"\data\raw\sales.csv"
    )

    dataset_columns = get_dataset_columns(
        filepath
    )

    step_text = (
        "Group the data by Region and calculate "
        "the total of Total Profit."
    )

    result = execute_self_correcting_step(

        step_number=1,

        step_text=step_text,

        filepath=filepath,

        data_profile=None,

        dataset_columns=dataset_columns,

        target_columns=[
            "Region",
            "Total Profit"
        ],

        test_fault=False
    )

    print(
        "\nFinal status:"
    )

    print(
        result["status"]
    )

    print(
        "\nTotal attempts:"
    )

    print(
        result["total_attempts"]
    )

    if result["status"] == "PASS":

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
# ARTIFICIAL FAULT DEMONSTRATION
# =========================================================

def test_fault_recovery(filepath=None):
    """Test automatic recovery using a supplied or available dataset."""

    from pathlib import Path

    print("\n" + "#" * 70)
    print("TEST 2: ARTIFICIAL FAULT RECOVERY")
    print("#" * 70)

    if filepath is None:
        candidates = [
            Path("outputs/uploads/active_uploaded_dataset.csv"),
            Path("data/raw/sales.csv"),
        ]
        filepath = next(
            (str(path.resolve()) for path in candidates if path.is_file()),
            None,
        )
    else:
        filepath = str(Path(filepath).resolve())

    if filepath is None or not Path(filepath).is_file():
        print("TEST 2 RESULT: FAIL - No suitable dataset file was found.")
        return False

    dataset_columns = get_dataset_columns(filepath)
    available_columns = set(dataset_columns)

    if {"month", "duration"}.issubset(available_columns):
        step_text = "Calculate the average duration of calls in August."
        target_columns = ["duration"]
    elif {"Region", "Total Profit"}.issubset(available_columns):
        step_text = (
            "Group the data by Region and calculate "
            "the total of Total Profit."
        )
        target_columns = ["Region", "Total Profit"]
    else:
        print("Dataset:", filepath)
        print("Available columns:", dataset_columns)
        print(
            "TEST 2 RESULT: FAIL - No supported test question "
            "matches this dataset's columns."
        )
        return False

    print("Dataset:", filepath)
    print("Test question:", step_text)

    result = execute_self_correcting_step(
        step_number=1,
        step_text=step_text,
        filepath=filepath,
        data_profile=None,
        dataset_columns=dataset_columns,
        target_columns=target_columns,
        test_fault=True,
    )

    status = result.get("status")
    attempts = result.get("total_attempts", 0)

    print("\nFinal status:", status)
    print("Total attempts:", attempts)

    passed = status == "PASS" and attempts >= 2
    print(
        "\nTEST 2 RESULT:",
        "PASS" if passed else "FAIL",
    )
    return passed


# MAIN TEST RUNNER
# =========================================================

if __name__ == "__main__":

    print("\n")

    print("=" * 70)

    print(
        "SELF-CORRECTING EDA STEP EXECUTOR"
    )

    print("=" * 70)

    test_results = []

    # -----------------------------------------------------
    # Production test
    # -----------------------------------------------------

    test_results.append(
        test_production_mode()
    )

    # -----------------------------------------------------
    # Controlled fault-recovery test
    # -----------------------------------------------------

    test_results.append(
        test_fault_recovery()
    )

    # -----------------------------------------------------
    # Final summary
    # -----------------------------------------------------

    print("\n")

    print("=" * 70)

    print(
        "FINAL TEST SUMMARY"
    )

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