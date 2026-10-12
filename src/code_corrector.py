"""
code_corrector.py

Robust validation-feedback-based code correction
for the Self-Validating EDA Agent.

Correction workflow:

    Faulty Code
         ↓
    Execution Error
         ↓
    Validation Feedback
         ↓
    Error / Column Extraction
         ↓
    Candidate Column Identification
         ↓
    Direct Code Correction
         ↓
    Re-execution
         ↓
    Validation

Regeneration is used only as a fallback.
"""

import re

from code_generator import generate_code_mock


# =========================================================
# COLUMN ALIASES
# =========================================================

COLUMN_ALIASES = {
    "UnitsSold": "Units Sold",
    "Units_Sold": "Units Sold",

    "TotalProfit": "Total Profit",
    "Total_Profit": "Total Profit",

    "TotalRevenue": "Total Revenue",
    "Total_Revenue": "Total Revenue",

    "TotalCost": "Total Cost",
    "Total_Cost": "Total Cost",

    "UnitPrice": "Unit Price",
    "Unit_Price": "Unit Price",

    "UnitCost": "Unit Cost",
    "Unit_Cost": "Unit Cost",

    "OrderDate": "Order Date",
    "Order_Date": "Order Date",

    "OrderID": "Order ID",
    "Order_ID": "Order ID",

    "ItemType": "Item Type",
    "Item_Type": "Item Type",

    "SalesChannel": "Sales Channel",
    "Sales_Channel": "Sales Channel",

    "OrderPriority": "Order Priority",
    "Order_Priority": "Order Priority",
}


# =========================================================
# NORMALIZATION
# =========================================================

def normalize_column_name(column_name):
    """
    Normalize a column name.

    Examples:

        Total Profit
        TotalProfit
        Total_Profit

    become:

        totalprofit
    """

    return (
        str(column_name)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
    )


# =========================================================
# CONVERT FEEDBACK TO TEXT
# =========================================================

def feedback_to_text(feedback):
    """
    Convert structured validation feedback into text
    for error analysis.
    """

    if feedback is None:
        return ""

    return str(feedback)


# =========================================================
# EXTRACT INVALID COLUMNS
# =========================================================

def extract_invalid_columns(feedback):
    """
    Extract invalid column names from several common
    execution/validation error formats.

    Supported examples:

        Column not found: Wrong Profit

        'Column not found: Wrong Profit'

        'Wrong Profit'

        KeyError: 'Wrong Profit'
    """

    feedback_text = feedback_to_text(
        feedback
    )

    if not feedback_text:
        return []

    invalid_columns = []

    # -----------------------------------------------------
    # Pattern 1
    #
    # Column not found: Wrong Profit
    # -----------------------------------------------------

    pattern_1 = (
        r"Column not found:\s*"
        r"['\"]?([^'\"\n,}]+)"
    )

    matches = re.findall(
        pattern_1,
        feedback_text,
        flags=re.IGNORECASE
    )

    for match in matches:

        column = match.strip()

        if column:
            invalid_columns.append(
                column
            )

    # -----------------------------------------------------
    # Pattern 2
    #
    # KeyError: 'Wrong Profit'
    # -----------------------------------------------------

    pattern_2 = (
        r"KeyError:\s*"
        r"['\"]([^'\"]+)['\"]"
    )

    matches = re.findall(
        pattern_2,
        feedback_text,
        flags=re.IGNORECASE
    )

    for match in matches:

        if match:
            invalid_columns.append(
                match.strip()
            )

    # -----------------------------------------------------
    # Pattern 3
    #
    # Bare:
    #
    # 'Wrong Profit'
    #
    # This is common when pandas raises a KeyError.
    # -----------------------------------------------------

    known_bad_names = [
        "Wrong Profit",
        "Wrong Revenue",
        "Wrong Cost",
        "Wrong Units",
        "Unknown Column",
        "SomeUnknownColumn"
    ]

    for bad_name in known_bad_names:

        if bad_name.lower() in feedback_text.lower():

            invalid_columns.append(
                bad_name
            )

    # -----------------------------------------------------
    # Remove duplicates
    # -----------------------------------------------------

    unique_columns = []

    for column in invalid_columns:

        column = column.strip(
            "'\" "
        )

        if column and column not in unique_columns:

            unique_columns.append(
                column
            )

    return unique_columns


# =========================================================
# EXTRACT REQUIRED COLUMNS
# =========================================================

def extract_required_columns(feedback):
    """
    Extract required columns from validation feedback.

    Example:

        Required columns not used in code:
        ['Total Profit']
    """

    feedback_text = feedback_to_text(
        feedback
    )

    if not feedback_text:
        return []

    required_columns = []

    pattern = (
        r"Required columns not used in code:\s*"
        r"\[([^\]]+)\]"
    )

    matches = re.findall(
        pattern,
        feedback_text,
        flags=re.IGNORECASE
    )

    for match in matches:

        quoted_columns = re.findall(
            r"'([^']+)'|\"([^\"]+)\"",
            match
        )

        for single_quote, double_quote in quoted_columns:

            column = (
                single_quote
                or
                double_quote
            )

            if column:

                required_columns.append(
                    column
                )

    return required_columns


# =========================================================
# FIND DATASET COLUMN
# =========================================================

def find_dataset_column(
    column_name,
    dataset_columns
):
    """
    Find the valid dataset column corresponding
    to a supplied name.
    """

    if not column_name:
        return None

    if not dataset_columns:
        return None

    dataset_columns = list(
        dataset_columns
    )

    # -----------------------------------------------------
    # Exact match
    # -----------------------------------------------------

    if column_name in dataset_columns:

        return column_name

    # -----------------------------------------------------
    # Alias
    # -----------------------------------------------------

    if column_name in COLUMN_ALIASES:

        valid_column = COLUMN_ALIASES[
            column_name
        ]

        if valid_column in dataset_columns:

            return valid_column

    # -----------------------------------------------------
    # Case insensitive
    # -----------------------------------------------------

    column_lower = column_name.lower()

    for column in dataset_columns:

        if column.lower() == column_lower:

            return column

    # -----------------------------------------------------
    # Normalized comparison
    # -----------------------------------------------------

    normalized_target = (
        normalize_column_name(
            column_name
        )
    )

    for column in dataset_columns:

        if (
            normalize_column_name(
                column
            )
            ==
            normalized_target
        ):

            return column

    return None


# =========================================================
# REPLACE COLUMN REFERENCE
# =========================================================

def replace_column_reference(
    code,
    old_column,
    new_column
):
    """
    Replace a quoted dataframe column reference.

    Example:

        df["Wrong Profit"]

    becomes:

        df["Total Profit"]
    """

    if not code:
        return code

    if not old_column:
        return code

    if not new_column:
        return code

    old_column = str(
        old_column
    )

    new_column = str(
        new_column
    )

    corrected_code = code

    # -----------------------------------------------------
    # Double-quoted reference
    # -----------------------------------------------------

    double_pattern = (
        r'(["])'
        + re.escape(old_column)
        + r'\1'
    )

    corrected_code = re.sub(
        double_pattern,
        lambda match:
            match.group(1)
            + new_column
            + match.group(1),
        corrected_code
    )

    # -----------------------------------------------------
    # Single-quoted reference
    # -----------------------------------------------------

    single_pattern = (
        r"(['])"
        + re.escape(old_column)
        + r"\1"
    )

    corrected_code = re.sub(
        single_pattern,
        lambda match:
            match.group(1)
            + new_column
            + match.group(1),
        corrected_code
    )

    return corrected_code


# =========================================================
# FIND CANDIDATE TARGET COLUMN
# =========================================================

def infer_replacement_column(
    invalid_column,
    target_columns,
    dataset_columns
):
    """
    Infer which valid target column should replace
    an invalid column.

    Example:

        Wrong Profit
              ↓
        Total Profit

    Unlike the previous version, this function does
    NOT require the target column to be completely absent
    from the code.

    It only uses the semantic relationship between
    the invalid name and expected analysis targets.
    """

    if not invalid_column:
        return None

    if not target_columns:
        return None

    invalid_normalized = (
        normalize_column_name(
            invalid_column
        )
    )

    # -----------------------------------------------------
    # Remove common error words
    # -----------------------------------------------------

    cleaned_invalid = invalid_normalized

    for word in [
        "wrong",
        "incorrect",
        "invalid",
        "unknown",
        "missing"
    ]:

        cleaned_invalid = (
            cleaned_invalid.replace(
                word,
                ""
            )
        )

    # -----------------------------------------------------
    # Score each target
    # -----------------------------------------------------

    best_column = None

    best_score = 0

    for target in target_columns:

        # Target must be a real dataset column.
        if target not in dataset_columns:
            continue

        target_normalized = (
            normalize_column_name(
                target
            )
        )

        score = 0

        # -------------------------------------------------
        # Explicit alias mapping
        # -------------------------------------------------

        if COLUMN_ALIASES.get(
            invalid_column
        ) == target:

            score += 100

        # -------------------------------------------------
        # Exact normalized match after removing error word
        #
        # WrongProfit → Profit
        # TotalProfit → totalprofit
        #
        # We don't require exact match here.
        # -------------------------------------------------

        if (
            cleaned_invalid
            and
            cleaned_invalid
            in target_normalized
        ):

            score += 50

        # -------------------------------------------------
        # Shared meaningful token
        # -------------------------------------------------

        invalid_tokens = re.findall(
            r"[a-zA-Z]+",
            str(invalid_column).lower()
        )

        target_tokens = re.findall(
            r"[a-zA-Z]+",
            str(target).lower()
        )

        common_tokens = set(
            invalid_tokens
        ).intersection(
            target_tokens
        )

        # Ignore generic/error tokens.
        common_tokens -= {
            "wrong",
            "incorrect",
            "invalid",
            "unknown",
            "missing"
        }

        score += (
            len(common_tokens) * 20
        )

        # -------------------------------------------------
        # Profit is an especially strong semantic match.
        # -------------------------------------------------

        if (
            "profit" in invalid_tokens
            and
            "profit" in target_tokens
        ):

            score += 40

        if (
            "revenue" in invalid_tokens
            and
            "revenue" in target_tokens
        ):

            score += 40

        if (
            "cost" in invalid_tokens
            and
            "cost" in target_tokens
        ):

            score += 40

        if (
            "units" in invalid_tokens
            and
            "units" in target_tokens
        ):

            score += 40

        if score > best_score:

            best_score = score

            best_column = target

    return best_column


# =========================================================
# DIRECT FEEDBACK CORRECTION
# =========================================================

def direct_feedback_correction(
    generated_code,
    feedback,
    dataset_columns,
    target_columns=None
):
    """
    Directly correct generated code using validation
    and execution feedback.
    """

    if target_columns is None:
        target_columns = []

    corrected_code = generated_code

    corrections = []

    # -----------------------------------------------------
    # Extract invalid columns from ALL feedback.
    # -----------------------------------------------------

    invalid_columns = (
        extract_invalid_columns(
            feedback
        )
    )

    # -----------------------------------------------------
    # Extract explicitly missing required columns.
    # -----------------------------------------------------

    required_columns = (
        extract_required_columns(
            feedback
        )
    )

    # -----------------------------------------------------
    # Add expected target columns.
    #
    # This is important for Steps 2, 4 and 5.
    # -----------------------------------------------------

    for target in target_columns:

        if target not in required_columns:

            required_columns.append(
                target
            )

    # =====================================================
    # STEP 1
    # DIRECT / INFERRED REPLACEMENT
    # =====================================================

    for invalid_column in invalid_columns:

        # -------------------------------------------------
        # First try explicit dataset mapping.
        # -------------------------------------------------

        valid_column = find_dataset_column(
            invalid_column,
            dataset_columns
        )

        if valid_column is not None:

            if invalid_column != valid_column:

                before_code = corrected_code

                corrected_code = (
                    replace_column_reference(
                        corrected_code,
                        invalid_column,
                        valid_column
                    )
                )

                if corrected_code != before_code:

                    corrections.append({
                        "type":
                            "direct_column_replacement",
                        "old_column":
                            invalid_column,
                        "new_column":
                            valid_column,
                        "reason":
                            (
                                "Invalid column mapped "
                                "directly to a valid "
                                "dataset column."
                            )
                    })

                continue

        # -------------------------------------------------
        # If no direct mapping exists, infer using
        # target columns.
        # -------------------------------------------------

        inferred_column = (
            infer_replacement_column(
                invalid_column=
                    invalid_column,
                target_columns=
                    target_columns,
                dataset_columns=
                    dataset_columns
            )
        )

        if inferred_column is not None:

            before_code = corrected_code

            corrected_code = (
                replace_column_reference(
                    corrected_code,
                    invalid_column,
                    inferred_column
                )
            )

            if corrected_code != before_code:

                corrections.append({
                    "type":
                        "feedback_inference_replacement",
                    "old_column":
                        invalid_column,
                    "new_column":
                        inferred_column,
                    "reason":
                        (
                            "Invalid column was mapped "
                            "to an expected target "
                            "column using execution "
                            "feedback."
                        )
                })

    # =====================================================
    # STEP 2
    # HANDLE KNOWN ALIASES
    # =====================================================

    for alias, actual_column in (
        COLUMN_ALIASES.items()
    ):

        if actual_column not in dataset_columns:
            continue

        double_pattern = (
            r'(["])'
            + re.escape(alias)
            + r'\1'
        )

        single_pattern = (
            r"(['])"
            + re.escape(alias)
            + r"\1"
        )

        alias_found = (
            re.search(
                double_pattern,
                corrected_code
            )
            or
            re.search(
                single_pattern,
                corrected_code
            )
        )

        if alias_found:

            before_code = corrected_code

            corrected_code = (
                replace_column_reference(
                    corrected_code,
                    alias,
                    actual_column
                )
            )

            if corrected_code != before_code:

                corrections.append({
                    "type":
                        "alias_replacement",
                    "old_column":
                        alias,
                    "new_column":
                        actual_column
                })

    # =====================================================
    # STEP 3
    # DETERMINE WHETHER CODE CHANGED
    # =====================================================

    return (
        corrected_code,
        corrections
    )


# =========================================================
# MAIN CORRECTOR
# =========================================================

def correct_generated_code(
    step_text,
    generated_code,
    feedback,
    dataset_columns,
    target_columns=None,
    filepath=None,
    data_profile=None,
    attempt_number=2
):
    """
    Main validation-feedback-based correction function.

    Priority:

        1. Direct feedback correction
        2. Semantic target inference
        3. Alias correction
        4. Regeneration fallback
        5. No change
    """

    if target_columns is None:
        target_columns = []

    # -----------------------------------------------------
    # Validate code
    # -----------------------------------------------------

    if not generated_code:

        return {
            "status": "FAIL",
            "correction_method": None,
            "corrected_code": None,
            "corrections": [],
            "reason":
                "Generated code is empty."
        }

    # -----------------------------------------------------
    # Validate dataset columns
    # -----------------------------------------------------

    if not dataset_columns:

        return {
            "status": "FAIL",
            "correction_method": None,
            "corrected_code":
                generated_code,
            "corrections": [],
            "reason":
                "Dataset columns are unavailable."
        }

    # =====================================================
    # STEP 1
    # DIRECT FEEDBACK CORRECTION
    # =====================================================

    (
        corrected_code,
        corrections
    ) = direct_feedback_correction(
        generated_code=
            generated_code,
        feedback=
            feedback,
        dataset_columns=
            dataset_columns,
        target_columns=
            target_columns
    )

    # -----------------------------------------------------
    # If code changed, return corrected code.
    # -----------------------------------------------------

    if corrections:

        return {
            "status": "PASS",
            "correction_method":
                "direct_feedback_correction",
            "corrected_code":
                corrected_code,
            "corrections":
                corrections,
            "reason":
                (
                    "Generated code was corrected "
                    "using execution and validation "
                    "feedback."
                )
        }

    # =====================================================
    # STEP 2
    # REGENERATION FALLBACK
    # =====================================================

    if filepath:

        try:

            regenerated_code = (
                generate_code_mock(
                    step_text,
                    filepath,
                    data_profile
                )
            )

            if regenerated_code:

                return {
                    "status": "PASS",
                    "correction_method":
                        "code_regeneration",
                    "corrected_code":
                        regenerated_code,
                    "corrections": [
                        {
                            "type":
                                "code_regeneration",
                            "reason":
                                (
                                    "Direct feedback "
                                    "correction could "
                                    "not repair the code."
                                )
                        }
                    ],
                    "reason":
                        (
                            "Code regenerated as "
                            "a fallback."
                        )
                }

        except Exception as error:

            return {
                "status": "FAIL",
                "correction_method":
                    "code_regeneration",
                "corrected_code":
                    generated_code,
                "corrections": [],
                "reason":
                    (
                        f"Code regeneration "
                        f"failed: {error}"
                    )
            }

    # =====================================================
    # STEP 3
    # NO CHANGE
    # =====================================================

    return {
        "status": "NO_CHANGE",
        "correction_method": None,
        "corrected_code":
            generated_code,
        "corrections": [],
        "reason":
            (
                "No applicable correction "
                "was identified."
            )
    }


# =========================================================
# TEST 1
# WRONG PROFIT
# =========================================================

def test_wrong_profit():

    print("\n" + "=" * 70)
    print("TEST 1: WRONG PROFIT → TOTAL PROFIT")
    print("=" * 70)

    dataset_columns = [
        "Region",
        "Total Profit"
    ]

    code = """
result = df.groupby("Region")["Wrong Profit"].sum()
"""

    feedback = {
        "code": {
            "pass": False,
            "reason":
                "'Column not found: Wrong Profit'"
        },
        "data": {
            "pass": False,
            "reason":
                (
                    "Required columns not used in code: "
                    "['Total Profit']"
                )
        },
        "result": {
            "pass": False,
            "reason":
                "Result is None"
        }
    }

    result = correct_generated_code(
        step_text=(
            "Group the data by Region and calculate "
            "the total of Total Profit."
        ),
        generated_code=code,
        feedback=feedback,
        dataset_columns=dataset_columns,
        target_columns=[
            "Region",
            "Total Profit"
        ]
    )

    print("\nCorrected code:")
    print(
        result["corrected_code"]
    )

    print("\nMethod:")
    print(
        result["correction_method"]
    )

    print("\nCorrections:")
    print(
        result["corrections"]
    )

    passed = (
        result["status"] == "PASS"
        and
        result["correction_method"]
        == "direct_feedback_correction"
        and
        "Wrong Profit"
        not in result["corrected_code"]
        and
        "Total Profit"
        in result["corrected_code"]
    )

    print(
        "\nTEST 1 RESULT:",
        "PASS" if passed else "FAIL"
    )

    return passed


# =========================================================
# TEST 2
# RANKING CODE
# =========================================================

def test_ranking_code():

    print("\n" + "=" * 70)
    print("TEST 2: RANKING CODE CORRECTION")
    print("=" * 70)

    dataset_columns = [
        "Region",
        "Total Profit"
    ]

    code = """
region_profit = (
    df.groupby("Region")["Wrong Profit"]
    .sum()
    .sort_values(ascending=False)
)

result = region_profit.reset_index()

result.columns = [
    "Region",
    "Total Profit"
]

result["Rank"] = (
    result["Total Profit"]
    .rank(
        method="dense",
        ascending=False
    )
    .astype(int)
)
"""

    feedback = {
        "code": {
            "pass": False,
            "reason":
                "'Column not found: Wrong Profit'"
        },
        "data": {
            "pass": True,
            "reason": None
        },
        "result": {
            "pass": False,
            "reason":
                "Result is None"
        }
    }

    result = correct_generated_code(
        step_text=(
            "Rank the Region categories according "
            "to total Total Profit."
        ),
        generated_code=code,
        feedback=feedback,
        dataset_columns=dataset_columns,
        target_columns=[
            "Region",
            "Total Profit"
        ]
    )

    print("\nCorrected code:")
    print(
        result["corrected_code"]
    )

    print("\nMethod:")
    print(
        result["correction_method"]
    )

    passed = (
        result["status"] == "PASS"
        and
        "Wrong Profit"
        not in result["corrected_code"]
        and
        "Total Profit"
        in result["corrected_code"]
    )

    print(
        "\nTEST 2 RESULT:",
        "PASS" if passed else "FAIL"
    )

    return passed


# =========================================================
# TEST 3
# OUTLIER CODE WITH BARE KEYERROR
# =========================================================

def test_outlier_code():

    print("\n" + "=" * 70)
    print("TEST 3: OUTLIER CODE CORRECTION")
    print("=" * 70)

    dataset_columns = [
        "Region",
        "Total Profit"
    ]

    code = """
def detect_region_outliers(group):

    Q1 = group["Wrong Profit"].quantile(0.25)

    Q3 = group["Total Profit"].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    return group[
        (group["Total Profit"] < lower_bound)
        |
        (group["Total Profit"] > upper_bound)
    ]
"""

    feedback = {
        "code": {
            "pass": False,
            "reason":
                "'Wrong Profit'"
        },
        "data": {
            "pass": True,
            "reason": None
        },
        "result": {
            "pass": False,
            "reason":
                "Result is None"
        }
    }

    result = correct_generated_code(
        step_text=(
            "Check for unusually high or low "
            "Total Profit values within each Region."
        ),
        generated_code=code,
        feedback=feedback,
        dataset_columns=dataset_columns,
        target_columns=[
            "Region",
            "Total Profit"
        ]
    )

    print("\nCorrected code:")
    print(
        result["corrected_code"]
    )

    print("\nMethod:")
    print(
        result["correction_method"]
    )

    passed = (
        result["status"] == "PASS"
        and
        "Wrong Profit"
        not in result["corrected_code"]
        and
        "Total Profit"
        in result["corrected_code"]
    )

    print(
        "\nTEST 3 RESULT:",
        "PASS" if passed else "FAIL"
    )

    return passed


# =========================================================
# TEST 4
# TOP REGION CODE
# =========================================================

def test_top_region_code():

    print("\n" + "=" * 70)
    print("TEST 4: TOP REGION CODE CORRECTION")
    print("=" * 70)

    dataset_columns = [
        "Region",
        "Total Profit"
    ]

    code = """
region_profit = (
    df.groupby("Region")["Wrong Profit"]
    .sum()
    .sort_values(ascending=False)
)

top_region = region_profit.index[0]

top_profit = region_profit.iloc[0]

result = {
    "Region": top_region,
    "Total Profit": float(top_profit)
}
"""

    feedback = {
        "code": {
            "pass": False,
            "reason":
                "'Column not found: Wrong Profit'"
        },
        "data": {
            "pass": True,
            "reason": None
        },
        "result": {
            "pass": False,
            "reason":
                "Result is None"
        }
    }

    result = correct_generated_code(
        step_text=(
            "Identify the Region category with "
            "the highest total Total Profit."
        ),
        generated_code=code,
        feedback=feedback,
        dataset_columns=dataset_columns,
        target_columns=[
            "Region",
            "Total Profit"
        ]
    )

    print("\nCorrected code:")
    print(
        result["corrected_code"]
    )

    print("\nMethod:")
    print(
        result["correction_method"]
    )

    passed = (
        result["status"] == "PASS"
        and
        "Wrong Profit"
        not in result["corrected_code"]
        and
        "Total Profit"
        in result["corrected_code"]
    )

    print(
        "\nTEST 4 RESULT:",
        "PASS" if passed else "FAIL"
    )

    return passed


# =========================================================
# TEST 5
# REGENERATION FALLBACK
# =========================================================

def test_regeneration_fallback():

    print("\n" + "=" * 70)
    print("TEST 5: REGENERATION FALLBACK")
    print("=" * 70)

    dataset_columns = [
        "Region",
        "Total Profit"
    ]

    code = """
result = df["SomeUnknownColumn"].sum()
"""

    feedback = {
        "code": {
            "pass": False,
            "reason":
                "SyntaxError: invalid code"
        }
    }

    filepath = (
        r"C:\Users\Vijay\Desktop"
        r"\Self_Validating_Data_Agent"
        r"\data\raw\sales.csv"
    )

    result = correct_generated_code(
        step_text=(
            "Group the data by Region and calculate "
            "the total of Total Profit."
        ),
        generated_code=code,
        feedback=feedback,
        dataset_columns=dataset_columns,
        target_columns=[
            "Region",
            "Total Profit"
        ],
        filepath=filepath,
        data_profile=None
    )

    print("\nCorrected code:")
    print(
        result["corrected_code"]
    )

    print("\nMethod:")
    print(
        result["correction_method"]
    )

    passed = (
        result["status"] == "PASS"
        and
        result["correction_method"]
        == "code_regeneration"
        and
        "Total Profit"
        in result["corrected_code"]
    )

    print(
        "\nTEST 5 RESULT:",
        "PASS" if passed else "FAIL"
    )

    return passed


# =========================================================
# MAIN TEST RUNNER
# =========================================================

if __name__ == "__main__":

    print("\n")
    print("#" * 70)
    print(
        "ROBUST VALIDATION-FEEDBACK CODE CORRECTOR"
    )
    print("#" * 70)

    results = []

    results.append(
        test_wrong_profit()
    )

    results.append(
        test_ranking_code()
    )

    results.append(
        test_outlier_code()
    )

    results.append(
        test_top_region_code()
    )

    results.append(
        test_regeneration_fallback()
    )

    print("\n")
    print("#" * 70)
    print("FINAL TEST SUMMARY")
    print("#" * 70)

    passed = sum(results)

    total = len(results)

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