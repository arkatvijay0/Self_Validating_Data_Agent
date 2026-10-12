import numpy as np
import pandas as pd
from scipy import stats


# ============================================================
# 1. CODE VALIDATION
# ============================================================

def validate_code(execution_result):
    """Check whether execution succeeded and returned a result."""

    if execution_result is None:
        return {
            "pass": False,
            "reason": "Execution result is None",
        }

    if execution_result.get("status") == "FAIL":
        error = execution_result.get("error")

        return {
            "pass": False,
            "reason": str(error)[:500] if error else "Code execution failed",
        }

    if execution_result.get("success") is False:
        error = (
            execution_result.get("stderr")
            or execution_result.get("error")
            or "Code execution failed"
        )

        return {
            "pass": False,
            "reason": str(error)[:500],
        }

    if execution_result.get("result") is None:
        return {
            "pass": False,
            "reason": "No result was produced",
        }

    return {
        "pass": True,
        "reason": None,
    }


# ============================================================
# 2. DATA USAGE VALIDATION
# ============================================================

def validate_data_usage(
    code,
    dataset_columns,
    target_columns=None,
    required_code_columns=None,
):
    """
    Validate dataset columns without incorrectly requiring
    every task-level target column in every individual step.

    target_columns:
        Task-level columns that must exist in the dataset.

    required_code_columns:
        Optional columns that this specific code step must
        explicitly reference. Supply these only when the
        caller knows the step requires those columns.
    """

    if dataset_columns is None:
        dataset_columns = []

    if target_columns is None:
        target_columns = []

    if required_code_columns is None:
        required_code_columns = []

    dataset_columns = list(dataset_columns)
    target_columns = list(target_columns)
    required_code_columns = list(required_code_columns)

    # --------------------------------------------------------
    # Check that task-level target columns exist.
    # --------------------------------------------------------

    missing_targets = [
        column
        for column in target_columns
        if column not in dataset_columns
    ]

    if missing_targets:
        return {
            "pass": False,
            "reason": (
                "Target columns not found in dataset: "
                f"{missing_targets}"
            ),
        }

    # --------------------------------------------------------
    # Check only explicitly required columns for this step.
    #
    # Do not enforce all task-level targets here. For example,
    # categorical frequency analysis can be valid without
    # explicitly referencing Total Revenue.
    # --------------------------------------------------------

    missing_required_columns = [
        column
        for column in required_code_columns
        if column not in dataset_columns
    ]

    if missing_required_columns:
        return {
            "pass": False,
            "reason": (
                "Required step columns not found in dataset: "
                f"{missing_required_columns}"
            ),
        }

    code_text = str(code)

    missing_from_code = [
        column
        for column in required_code_columns
        if column not in code_text
    ]

    if missing_from_code:
        return {
            "pass": False,
            "reason": (
                "Required step columns not used in code: "
                f"{missing_from_code}"
            ),
        }

    return {
        "pass": True,
        "reason": None,
        "checked_target_columns": target_columns,
        "checked_step_columns": required_code_columns,
        "note": (
            "Task-level targets were checked for dataset "
            "existence. Code usage is enforced only for "
            "explicitly specified step-level columns."
        ),
    }


# ============================================================
# 3. RESULT VALIDATION
# ============================================================

def validate_result(result_value):
    """Check whether an analysis result is usable and non-empty."""

    if result_value is None:
        return {
            "pass": False,
            "reason": "Result is None",
        }

    if isinstance(result_value, str):
        cleaned = result_value.strip().lower()

        if cleaned in {"", "none", "null", "nan", "na"}:
            return {
                "pass": False,
                "reason": "Result is empty or null",
            }

    if isinstance(result_value, (list, tuple, dict, set)):
        if len(result_value) == 0:
            return {
                "pass": False,
                "reason": "Result is empty",
            }

    if isinstance(result_value, (pd.DataFrame, pd.Series)):
        if result_value.empty:
            return {
                "pass": False,
                "reason": "Result DataFrame or Series is empty",
            }

    if isinstance(result_value, (float, np.floating)):
        if pd.isna(result_value):
            return {
                "pass": False,
                "reason": "Result is NaN",
            }

    return {
        "pass": True,
        "reason": None,
    }


# ============================================================
# 4. NORMALITY CHECK
# ============================================================

def check_normality(data_series):
    """Perform a Shapiro-Wilk normality test."""

    if data_series is None:
        return {
            "pass": False,
            "reason": "No data provided for normality check",
            "p_value": None,
        }

    try:
        if not isinstance(data_series, pd.Series):
            data_series = pd.Series(data_series)

        clean_data = pd.to_numeric(
            data_series,
            errors="coerce",
        ).dropna()

    except Exception as error:
        return {
            "pass": False,
            "reason": f"Unable to prepare data: {error}",
            "p_value": None,
        }

    if len(clean_data) < 3:
        return {
            "pass": False,
            "reason": (
                "Insufficient observations for Shapiro-Wilk test"
            ),
            "p_value": None,
        }

    if len(clean_data) > 5000:
        clean_data = clean_data.sample(
            n=5000,
            random_state=42,
        )

    try:
        statistic, p_value = stats.shapiro(clean_data)

        return {
            "pass": bool(p_value > 0.05),
            "reason": (
                None
                if p_value > 0.05
                else "Normality assumption not supported by the test"
            ),
            "statistic": float(statistic),
            "p_value": float(p_value),
        }

    except Exception as error:
        return {
            "pass": False,
            "reason": f"Normality test failed: {error}",
            "p_value": None,
        }


# ============================================================
# 5. STATISTICAL VALIDATION
# ============================================================

def statistical_check(test_used, data_series=None):
    """Validate assumptions for supported statistical methods."""

    if test_used is None:
        return {
            "pass": True,
            "reason": None,
        }

    if data_series is None:
        return {
            "pass": False,
            "reason": "No data provided for statistical validation",
        }

    try:
        clean_data = pd.to_numeric(
            pd.Series(data_series),
            errors="coerce",
        ).dropna()

    except Exception as error:
        return {
            "pass": False,
            "reason": f"Unable to prepare statistical data: {error}",
        }

    if len(clean_data) < 3:
        return {
            "pass": False,
            "reason": (
                "Insufficient observations for statistical validation"
            ),
        }

    # --------------------------------------------------------
    # T-TEST
    # --------------------------------------------------------

    if test_used == "t-test":
        normality = check_normality(clean_data)

        if not normality["pass"]:
            return {
                "pass": False,
                "reason": (
                    "Normality was not supported by the test. "
                    "Review the assumptions and consider an "
                    "appropriate non-parametric alternative."
                ),
                "warning": normality.get("reason"),
                "alternative": "Mann-Whitney U",
                "p_value": normality.get("p_value"),
            }

        return {
            "pass": True,
            "reason": None,
            "normality": normality,
        }

    # --------------------------------------------------------
    # PEARSON CORRELATION
    # --------------------------------------------------------

    if test_used == "pearson_correlation":
        normality = check_normality(clean_data)

        if not normality["pass"]:
            return {
                "pass": True,
                "reason": None,
                "warning": (
                    "Normality was not supported by Shapiro-Wilk. "
                    "Review Pearson correlation assumptions and "
                    "consider Spearman correlation as a rank-based "
                    "alternative where appropriate."
                ),
                "alternative": "Spearman correlation",
                "normality": normality,
            }

        return {
            "pass": True,
            "reason": None,
            "normality": normality,
        }

    # --------------------------------------------------------
    # SPEARMAN CORRELATION
    # --------------------------------------------------------

    if test_used == "spearman_correlation":
        return {
            "pass": True,
            "reason": None,
            "note": (
                "Spearman correlation is rank-based and does not "
                "require normally distributed variables."
            ),
        }

    # --------------------------------------------------------
    # UNKNOWN TEST
    # --------------------------------------------------------

    return {
        "pass": True,
        "reason": None,
        "warning": (
            "No specialized statistical assumption check is "
            f"implemented for '{test_used}'."
        ),
    }


# ============================================================
# 6. COMPLETE SELF-VALIDATION
# ============================================================

def self_validate(
    execution_result,
    code,
    dataset_columns,
    target_columns,
    test_used=None,
    data_series=None,
    required_code_columns=None,
):
    """
    Run code, data, result, and optional statistical validation.

    required_code_columns should contain only the columns that
    the individual plan step is expected to use.
    """

    result_value = None

    if execution_result is not None:
        result_value = execution_result.get("result")

    checks = {
        "code": validate_code(execution_result),

        "data": validate_data_usage(
            code=code,
            dataset_columns=dataset_columns,
            target_columns=target_columns,
            required_code_columns=required_code_columns,
        ),

        "result": validate_result(result_value),
    }

    if test_used is not None:
        checks["statistical"] = statistical_check(
            test_used=test_used,
            data_series=data_series,
        )

    overall_pass = all(
        check.get("pass", False)
        for check in checks.values()
    )

    return {
        "overall": "PASS" if overall_pass else "FAIL",
        "checks": checks,
    }


# ============================================================
# TESTS
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("VALIDATORS TEST")
    print("=" * 70)

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
        "Total Profit",
    ]

    # Test 1: successful execution
    execution_result = {
        "status": "SUCCESS",
        "success": True,
        "result": pd.DataFrame({"count": [100]}),
    }

    print("\nTEST 1 - Code validation")
    print(validate_code(execution_result))

    # Test 2: categorical analysis should not require
    # Total Revenue to appear explicitly in the generated code.
    categorical_code = """
categorical_columns = df.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

result = {
    column: df[column].value_counts()
    for column in categorical_columns
}
"""

    print("\nTEST 2 - Generic categorical step")
    categorical_validation = validate_data_usage(
        code=categorical_code,
        dataset_columns=dataset_columns,
        target_columns=["Total Revenue"],
    )
    print(categorical_validation)

    assert categorical_validation["pass"] is True

    # Test 3: strict step-level checks still work.
    print("\nTEST 3 - Required step column")
    strict_validation = validate_data_usage(
        code='result = df["Total Revenue"].mean()',
        dataset_columns=dataset_columns,
        target_columns=["Total Revenue"],
        required_code_columns=["Total Revenue"],
    )
    print(strict_validation)

    assert strict_validation["pass"] is True

    # Test 4: reject a required column absent from the code.
    print("\nTEST 4 - Missing required step column")
    failed_validation = validate_data_usage(
        code="result = df.describe()",
        dataset_columns=dataset_columns,
        target_columns=["Total Revenue"],
        required_code_columns=["Total Revenue"],
    )
    print(failed_validation)

    assert failed_validation["pass"] is False

    # Test 5: result validation
    print("\nTEST 5 - Result validation")
    print(validate_result(pd.DataFrame({"count": [100]})))

    # Test 6: statistical validation
    sample_series = pd.Series(
        np.random.RandomState(42).normal(0, 1, 100)
    )

    print("\nTEST 6 - Pearson statistical validation")
    print(
        statistical_check(
            test_used="pearson_correlation",
            data_series=sample_series,
        )
    )

    print("\nTEST 7 - Spearman statistical validation")
    print(
        statistical_check(
            test_used="spearman_correlation",
            data_series=sample_series,
        )
    )

    print("\n" + "=" * 70)
    print("VALIDATORS TEST COMPLETED")
    print("=" * 70)