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

    # Preserve the most specific execution error for the
    # self-correction loop. execute_generated_code() stores
    # runtime errors in "stderr", including when status="FAIL".
    if (
        execution_result.get("status") == "FAIL"
        or execution_result.get("success") is False
    ):
        error = (
            execution_result.get("stderr")
            or execution_result.get("error")
            or execution_result.get("exception")
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
# 3A. RULE-BASED RESULT VALIDATION (ALGORITHM 1)
# ============================================================

def validate_result_rules(
    result_value,
    count_value=None,
    total_value=None,
    percentage_value=None,
    tolerance=0.5,
):
    """
    Apply deterministic business/data rules to a result.

    The function is deliberately separate from validate_result()
    so existing callers remain backward-compatible.

    Parameters
    ----------
    result_value:
        Result produced by the analysis. A dictionary, pandas
        Series, or one-row DataFrame may include fields such as
        converted_count, total_count, conversion_rate_percent.
    count_value:
        Optional explicit numerator/count (e.g. conversions).
    total_value:
        Optional explicit denominator/total (e.g. customers).
    percentage_value:
        Optional reported percentage to check against count_value
        and total_value.
    tolerance:
        Maximum allowed absolute difference between reported and
        calculated percentage, in percentage points.

    Returns
    -------
    dict with "pass", "reason", and a list of checks.
    """

    checks = []

    def fail(reason, check_name):
        checks.append({
            "name": check_name,
            "pass": False,
            "reason": reason,
        })

    def passed(check_name, details=None):
        item = {
            "name": check_name,
            "pass": True,
            "reason": None,
        }
        if details is not None:
            item["details"] = details
        checks.append(item)

    def finite_number(value):
        if isinstance(value, bool):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if not np.isfinite(number):
            return None
        return number

    # Convert simple pandas outputs to a mapping when possible.
    values = {}
    if isinstance(result_value, dict):
        values = result_value
    elif isinstance(result_value, pd.Series):
        values = result_value.to_dict()
    elif isinstance(result_value, pd.DataFrame):
        if len(result_value) == 1:
            values = result_value.iloc[0].to_dict()

    # Normalize field names for alias matching.
    normalized = {
        "".join(character.lower() for character in str(key)
                if character.isalnum()): value
        for key, value in values.items()
    }

    def find_alias(aliases):
        for alias in aliases:
            key = "".join(
                character.lower() for character in alias
                if character.isalnum()
            )
            if key in normalized:
                return normalized[key]
        return None

    if result_value is None:
        fail("Result is None.", "result_presence")
    else:
        passed("result_presence")

    # Infer count and percentage values only when the result has
    # recognizable, unambiguous field names.
    if count_value is None:
        count_value = find_alias((
            "converted_count", "conversion_count", "conversions",
            "converted", "success_count", "numerator",
        ))

    if total_value is None:
        total_value = find_alias((
            "total_count", "customer_count", "total_customers",
            "total", "sample_size", "denominator",
        ))

    if percentage_value is None:
        percentage_value = find_alias((
            "conversion_rate_percent", "conversion_percentage",
            "conversion_percent", "percentage", "percent",
            "rate_percent", "success_rate_percent",
        ))

    # Check explicitly named percentage fields, plus an explicit
    # percentage_value argument. Avoid interpreting arbitrary
    # fields named simply "rate" as percentages.
    percentage_fields = []
    for key, value in values.items():
        key_normalized = "".join(
            character.lower() for character in str(key)
            if character.isalnum()
        )
        if (
            "percent" in key_normalized
            or "percentage" in key_normalized
        ):
            percentage_fields.append((str(key), value))

    if percentage_value is not None and not any(
        value is percentage_value for _, value in percentage_fields
    ):
        percentage_fields.append(("percentage_value", percentage_value))

    for field_name, raw_percentage in percentage_fields:
        percentage = finite_number(raw_percentage)
        if percentage is None:
            fail(
                f"{field_name} must be a finite numeric value.",
                f"percentage:{field_name}",
            )
        elif not 0 <= percentage <= 100:
            fail(
                f"{field_name} must be between 0 and 100; "
                f"received {percentage}.",
                f"percentage:{field_name}",
            )
        else:
            passed(
                f"percentage:{field_name}",
                {"value": percentage, "valid_range": [0, 100]},
            )

    # Validate counts and their relationship.
    count = finite_number(count_value) if count_value is not None else None
    total = finite_number(total_value) if total_value is not None else None

    if count_value is not None:
        if count is None:
            fail("Count must be a finite numeric value.", "count_validity")
        elif count < 0:
            fail("Count cannot be negative.", "count_validity")
        elif not count.is_integer():
            fail("Count must be a whole number.", "count_validity")
        else:
            passed("count_validity", {"count": int(count)})

    if total_value is not None:
        if total is None:
            fail("Total must be a finite numeric value.", "total_validity")
        elif total <= 0:
            fail("Total must be greater than zero.", "total_validity")
        elif not total.is_integer():
            fail("Total must be a whole number.", "total_validity")
        else:
            passed("total_validity", {"total": int(total)})

    if count is not None and total is not None:
        if count < 0 or total <= 0:
            # More specific failures have already been recorded.
            pass
        elif count > total:
            fail(
                f"Count ({count:g}) cannot exceed total ({total:g}).",
                "count_total_consistency",
            )
        else:
            passed(
                "count_total_consistency",
                {"count": count, "total": total},
            )

        if count >= 0 and total > 0:
            calculated_percentage = (count / total) * 100
            passed(
                "calculated_percentage",
                {"calculated_percentage": calculated_percentage},
            )

            if percentage_value is not None:
                reported_percentage = finite_number(percentage_value)
                if reported_percentage is None:
                    fail(
                        "Reported percentage must be finite and numeric.",
                        "reported_percentage_consistency",
                    )
                elif abs(
                    reported_percentage - calculated_percentage
                ) > tolerance:
                    fail(
                        "Reported percentage "
                        f"({reported_percentage:.4f}%) does not match "
                        f"the calculated percentage "
                        f"({calculated_percentage:.4f}%) within "
                        f"{tolerance} percentage points.",
                        "reported_percentage_consistency",
                    )
                else:
                    passed(
                        "reported_percentage_consistency",
                        {
                            "reported_percentage": reported_percentage,
                            "calculated_percentage": calculated_percentage,
                            "tolerance_percentage_points": tolerance,
                        },
                    )

    overall_pass = all(item["pass"] for item in checks)

    failures = [
        item["reason"] for item in checks if not item["pass"]
    ]

    return {
        "pass": overall_pass,
        "reason": "; ".join(failures) if failures else None,
        "checks": checks,
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


    # Test 8: valid rule-based count and percentage
    print("\nTEST 8 - Rule-based validation: valid conversion rate")
    valid_rule_result = validate_result_rules({
        "converted_count": 135,
        "total_count": 829,
        "conversion_rate_percent": (135 / 829) * 100,
    })
    print(valid_rule_result)
    assert valid_rule_result["pass"] is True

    # Test 9: reject a reported percentage that is inconsistent.
    print("\nTEST 9 - Rule-based validation: incorrect conversion rate")
    invalid_rule_result = validate_result_rules({
        "converted_count": 135,
        "total_count": 829,
        "conversion_rate_percent": 19.0,
    })
    print(invalid_rule_result)
    assert invalid_rule_result["pass"] is False

    # Test 10: reject counts greater than the total.
    print("\nTEST 10 - Rule-based validation: count exceeds total")
    invalid_count_result = validate_result_rules({
        "converted_count": 135,
        "total_count": 100,
        "conversion_rate_percent": 135.0,
    })
    print(invalid_count_result)
    assert invalid_count_result["pass"] is False

    # Test 11: reject a percentage outside the valid range.
    print("\nTEST 11 - Rule-based validation: percentage out of range")
    invalid_percentage_result = validate_result_rules({
        "conversion_rate_percent": 125.0,
    })
    print(invalid_percentage_result)
    assert invalid_percentage_result["pass"] is False

    print("\n" + "=" * 70)
    print("VALIDATORS TEST COMPLETED")
    print("=" * 70)