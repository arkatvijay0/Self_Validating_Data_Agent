import os
import sys
import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

SRC_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

if SRC_DIR not in sys.path:
    sys.path.insert(
        0,
        SRC_DIR
    )


# ============================================================
# IMPORT PROJECT MODULES
# ============================================================

from code_generator import generate_code_mock
from validators import self_validate


# ============================================================
# REAL EDA STEP EXECUTOR
# ============================================================

def execute_eda_step(
    step_text,
    step_number,
    filepath,
    data_profile,
    dataset_columns,
    target_columns
):
    """
    Execute one real EDA plan step.

    Pipeline:

        EDA Step
            ↓
        Code Generation
            ↓
        Code Execution
            ↓
        Self Validation
            ↓
        PASS / FAIL

    Parameters
    ----------
    step_text : str
        EDA plan step.

    step_number : int
        Position of the step in the EDA plan.

    filepath : str
        Dataset path.

    data_profile : dict
        Dataset profile.

    dataset_columns : list
        Dataset column names.

    target_columns : list
        Columns relevant to the task.

    Returns
    -------
    dict
        Standardized step execution result.
    """

    print("\n" + "-" * 60)
    print(
        f"REAL EDA EXECUTOR - STEP {step_number}"
    )
    print("-" * 60)

    print("\nEDA Step:")
    print(step_text)

    # ========================================================
    # STEP 1 - CODE GENERATION
    # ========================================================

    print("\n[1] Generating Python code...")

    try:

        generated_code = generate_code_mock(
            step_text,
            filepath,
            data_profile
        )

    except Exception as error:

        return {
            "status": "FAIL",
            "step_number": step_number,
            "step": step_text,
            "generated_code": None,
            "execution": None,
            "validation": None,
            "result": None,
            "error": (
                f"{type(error).__name__}: {error}"
            )
        }

    print("\nGenerated Code:")
    print(generated_code)

    # ========================================================
    # STEP 2 - CODE EXECUTION
    # ========================================================

    print("\n[2] Executing generated code...")

    execution_namespace = {}

    try:

        exec(
            generated_code,
            execution_namespace
        )

        result = execution_namespace.get(
            "result"
        )

        execution_result = {
            "status": "SUCCESS",
            "success": True,
            "stdout": "",
            "stderr": "",
            "result": result
        }

    except Exception as error:

        execution_result = {
            "status": "FAILED",
            "success": False,
            "stdout": "",
            "stderr": (
                f"{type(error).__name__}: {error}"
            ),
            "result": None
        }

    print("\nExecution Result:")
    print(execution_result)

    # ========================================================
    # STEP 3 - SELF VALIDATION
    # ========================================================

    print("\n[3] Running self-validation...")

    validation_result = self_validate(
        execution_result=execution_result,
        code=generated_code,
        dataset_columns=dataset_columns,
        target_columns=target_columns
    )

    print("\nValidation Result:")
    print(validation_result)

    # ========================================================
    # STEP 4 - DETERMINE FINAL STATUS
    # ========================================================

    if (
        validation_result.get(
            "overall"
        )
        == "PASS"
    ):

        final_status = "PASS"

        error = None

    else:

        final_status = "FAIL"

        failed_checks = []

        checks = validation_result.get(
            "checks",
            {}
        )

        for check_name, check_result in checks.items():

            if not check_result.get(
                "pass",
                False
            ):

                failed_checks.append(
                    check_name
                )

        if failed_checks:

            error = (
                "Validation failed for: "
                + ", ".join(
                    failed_checks
                )
            )

        else:

            error = (
                "EDA step validation failed."
            )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    return {
        "status": final_status,
        "step_number": step_number,
        "step": step_text,
        "generated_code": generated_code,
        "execution": execution_result,
        "validation": validation_result,
        "result": result,
        "error": error
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("REAL EDA STEP EXECUTOR TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    PROJECT_ROOT = os.path.dirname(
        SRC_DIR
    )

    DATA_FILE = os.path.join(
        PROJECT_ROOT,
        "data",
        "raw",
        "sales.csv"
    )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    if not os.path.exists(
        DATA_FILE
    ):

        print(
            "\nERROR: sales.csv was not found."
        )

        sys.exit(1)

    df = pd.read_csv(
        DATA_FILE
    )

    print(
        "\nDataset loaded successfully."
    )

    print(
        "Rows:",
        len(df)
    )

    print(
        "Columns:",
        len(df.columns)
    )

    # --------------------------------------------------------
    # Dataset profile
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
    # Test step
    # --------------------------------------------------------

    step_text = (
        "Group the data by Region and "
        "calculate the total of Total Profit."
    )

    target_columns = [
        "Region",
        "Total Profit"
    ]

    # --------------------------------------------------------
    # Execute
    # --------------------------------------------------------

    result = execute_eda_step(
        step_text=step_text,
        step_number=1,
        filepath=DATA_FILE,
        data_profile=data_profile,
        dataset_columns=list(
            df.columns
        ),
        target_columns=target_columns
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("REAL EDA STEP EXECUTION RESULT")
    print("=" * 70)

    print(
        "\nStatus:",
        result["status"]
    )

    print(
        "\nResult:"
    )

    print(
        result["result"]
    )

    print(
        "\nValidation:"
    )

    print(
        result["validation"]
    )

    print("\n" + "=" * 70)
    print("REAL EDA STEP EXECUTOR TEST COMPLETED")
    print("=" * 70)