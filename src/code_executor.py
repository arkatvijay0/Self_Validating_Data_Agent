import subprocess
import os
import json
import sys


# ============================================================
# PROJECT PATHS
# ============================================================

CURRENT_FILE = os.path.abspath(__file__)

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(CURRENT_FILE)
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# CODE EXECUTOR
# ============================================================

def execute_code(code_str):
    """
    Execute generated Python code and retrieve
    the variable named 'result'.
    """

    # --------------------------------------------------------
    # Output files
    # --------------------------------------------------------

    generated_file = os.path.join(
        OUTPUT_DIR,
        "generated_step.py"
    )

    result_file = os.path.join(
        OUTPUT_DIR,
        "result.json"
    )

    # --------------------------------------------------------
    # Delete previous result
    # --------------------------------------------------------

    if os.path.exists(result_file):
        os.remove(result_file)

    # --------------------------------------------------------
    # Add result-saving code
    # --------------------------------------------------------

    wrapper = code_str + f"""

# ============================================================
# SAVE RESULT
# ============================================================

import json

try:

    # Preserve pandas structures when saving the result
    if isinstance(result, pd.Series):

        serializable_result = result.to_dict()

    elif isinstance(result, pd.DataFrame):

        serializable_result = result.to_dict(
            orient="records"
        )

    else:

        serializable_result = result

    with open(
        r"{result_file}",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            serializable_result,
            f,
            default=str,
            indent=4
        )

except Exception as e:

    with open(
        r"{result_file}",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            {{"error": str(e)}},
            f,
            indent=4
        )
"""

    # --------------------------------------------------------
    # Save generated Python file
    # --------------------------------------------------------

    with open(
        generated_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(wrapper)

    # --------------------------------------------------------
    # Execute generated code
    # --------------------------------------------------------

    try:

        process = subprocess.run(
            [
                sys.executable,
                generated_file
            ],
            capture_output=True,
            text=True,
            timeout=30
        )

    except subprocess.TimeoutExpired:

        return {
            "success": False,
            "stdout": "",
            "stderr": "Execution timed out after 30 seconds.",
            "result": None,
            "generated_file": generated_file
        }

    except Exception as e:

        return {
            "success": False,
            "stdout": "",
            "stderr": str(e),
            "result": None,
            "generated_file": generated_file
        }

    # --------------------------------------------------------
    # Read result
    # --------------------------------------------------------

    result_value = None

    if os.path.exists(result_file):

        try:

            with open(
                result_file,
                "r",
                encoding="utf-8"
            ) as f:

                result_value = json.load(f)

        except Exception:

            result_value = None

    # --------------------------------------------------------
    # Determine success
    # --------------------------------------------------------

    success = (
        process.returncode == 0
        and result_value is not None
    )

    # --------------------------------------------------------
    # Return execution result
    # --------------------------------------------------------

    return {

        "success": success,

        "stdout": process.stdout,

        "stderr": process.stderr,

        "result": result_value,

        "generated_file": generated_file

    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("CODE EXECUTOR TEST")
    print("=" * 60)

    # --------------------------------------------------------
    # Dataset path
    # --------------------------------------------------------

    DATA_FILE = os.path.join(
        PROJECT_ROOT,
        "data",
        "raw",
        "sales.csv"
    )

    print("\nDataset:")
    print(DATA_FILE)

    # --------------------------------------------------------
    # Check dataset exists
    # --------------------------------------------------------

    if not os.path.exists(DATA_FILE):

        print("\nERROR: Dataset not found.")

        print(
            "Expected dataset at:"
        )

        print(DATA_FILE)

        sys.exit(1)

    # --------------------------------------------------------
    # Test generated code
    # --------------------------------------------------------

    test_code = f"""
import pandas as pd

df = pd.read_csv(
    r"{DATA_FILE}"
)

result = (
    df.groupby("Region")["Total Profit"]
    .sum()
    .sort_values(
        ascending=False
    )
)
"""

    # --------------------------------------------------------
    # Display generated code
    # --------------------------------------------------------

    print("\nGenerated Code:")
    print(test_code)

    # --------------------------------------------------------
    # Execute
    # --------------------------------------------------------

    print("\nExecuting code...")

    execution_result = execute_code(
        test_code
    )

    # --------------------------------------------------------
    # Display result
    # --------------------------------------------------------

    print("\nExecution Result:")

    print(
        json.dumps(
            execution_result,
            indent=4,
            default=str
        )
    )

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    print("\n" + "=" * 60)

    if execution_result["success"]:

        print(
            "CODE EXECUTION SUCCESSFUL"
        )

    else:

        print(
            "CODE EXECUTION FAILED"
        )

    print("=" * 60)