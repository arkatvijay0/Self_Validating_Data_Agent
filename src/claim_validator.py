"""
Claim-Level Evidence Validator

Purpose:
Validate individual analytical claims against independently
computed evidence.

Supported claim types:
1. highest_category
2. numeric_value
3. ranking
"""


import re


# ======================================================================
# TEXT NORMALIZATION
# ======================================================================

def normalize_text(text):
    """
    Normalize text for comparison.
    """

    if text is None:
        return ""

    return " ".join(
        str(text)
        .lower()
        .strip()
        .split()
    )


# ======================================================================
# EVIDENCE FIELD ACCESS
# ======================================================================

def get_evidence_value(
    evidence,
    key
):
    """
    Retrieve an evidence field.

    The project evidence structure stores analytical evidence
    inside evidence["evidence"].

    This function also supports a flat structure for robustness.
    """

    if not isinstance(
        evidence,
        dict
    ):
        return None

    # Current project structure
    nested_evidence = evidence.get(
        "evidence"
    )

    if isinstance(
        nested_evidence,
        dict
    ):

        if key in nested_evidence:
            return nested_evidence.get(
                key
            )

    # Fallback for flat evidence
    return evidence.get(
        key
    )


# ======================================================================
# CATEGORY EXTRACTION
# ======================================================================

def extract_category_claim(
    claim,
    evidence
):
    """
    Identify which category is mentioned in the claim.
    """

    normalized_claim = normalize_text(
        claim
    )

    supporting_result = evidence.get(
        "supporting_result",
        {}
    )

    if not isinstance(
        supporting_result,
        dict
    ):
        return None

    for category in supporting_result.keys():

        category_text = normalize_text(
            category
        )

        if (
            category_text
            and category_text in normalized_claim
        ):
            return category

    return None


# ======================================================================
# HIGHEST CATEGORY VALIDATION
# ======================================================================

def validate_highest_category_claim(
    claim,
    evidence
):
    """
    Validate a claim stating that a category
    has the highest value.
    """

    top_category = get_evidence_value(
        evidence,
        "top_category"
    )

    claimed_category = extract_category_claim(
        claim,
        evidence
    )

    if top_category is None:

        return {
            "status": "FAIL",
            "claim": claim,
            "claim_type": "highest_category",
            "reason": (
                "Evidence does not contain "
                "a top category."
            )
        }

    if claimed_category is None:

        return {
            "status": "FAIL",
            "claim": claim,
            "claim_type": "highest_category",
            "reason": (
                "Could not identify the "
                "claimed category."
            )
        }

    if (
        normalize_text(claimed_category)
        ==
        normalize_text(top_category)
    ):

        return {
            "status": "PASS",
            "claim": claim,
            "claim_type": "highest_category",
            "claimed_category": claimed_category,
            "expected_category": top_category,
            "reason": (
                "Claim matches the evidence."
            )
        }

    return {
        "status": "FAIL",
        "claim": claim,
        "claim_type": "highest_category",
        "claimed_category": claimed_category,
        "expected_category": top_category,
        "reason": (
            "Claim contradicts the evidence. "
            f"Expected highest category: "
            f"{top_category}. "
            f"Claim states: "
            f"{claimed_category}."
        )
    }


# ======================================================================
# NUMERIC VALUE EXTRACTION
# ======================================================================

def extract_numeric_value(
    claim
):
    """
    Extract the first numeric value from a claim.
    """

    if claim is None:
        return None

    matches = re.findall(
        r"\d[\d,]*(?:\.\d+)?",
        str(claim)
    )

    if not matches:
        return None

    value_text = matches[0].replace(
        ",",
        ""
    )

    try:
        return float(
            value_text
        )

    except ValueError:
        return None


# ======================================================================
# NUMERIC VALUE VALIDATION
# ======================================================================

def validate_value_claim(
    claim,
    expected_value,
    tolerance=0.01
):
    """
    Validate a numerical claim against evidence.
    """

    claimed_value = extract_numeric_value(
        claim
    )

    if claimed_value is None:

        return {
            "status": "FAIL",
            "claim": claim,
            "claim_type": "numeric_value",
            "reason": (
                "No numerical value could "
                "be extracted."
            )
        }

    if expected_value is None:

        return {
            "status": "FAIL",
            "claim": claim,
            "claim_type": "numeric_value",
            "reason": (
                "Expected evidence value "
                "is missing."
            )
        }

    expected_value = float(
        expected_value
    )

    difference = abs(
        claimed_value
        -
        expected_value
    )

    allowed_difference = max(
        abs(expected_value)
        *
        tolerance,
        0.01
    )

    if difference <= allowed_difference:

        return {
            "status": "PASS",
            "claim": claim,
            "claim_type": "numeric_value",
            "claimed_value": claimed_value,
            "expected_value": expected_value,
            "difference": difference,
            "reason": (
                "Claimed value matches "
                "the evidence."
            )
        }

    return {
        "status": "FAIL",
        "claim": claim,
        "claim_type": "numeric_value",
        "claimed_value": claimed_value,
        "expected_value": expected_value,
        "difference": difference,
        "reason": (
            f"Claimed value {claimed_value} "
            f"does not match expected value "
            f"{expected_value}."
        )
    }


# ======================================================================
# RANK EXTRACTION
# ======================================================================

def extract_rank(
    claim
):
    """
    Extract ranking from text.
    """

    normalized_claim = normalize_text(
        claim
    )

    rank_patterns = {
        1: ["first", "1st"],
        2: ["second", "2nd"],
        3: ["third", "3rd"],
        4: ["fourth", "4th"],
        5: ["fifth", "5th"]
    }

    for rank, patterns in rank_patterns.items():

        for pattern in patterns:

            if pattern in normalized_claim:
                return rank

    return None


# ======================================================================
# RANK VALIDATION
# ======================================================================

def validate_rank_claim(
    claim,
    expected_category,
    expected_rank
):
    """
    Validate a ranking claim.

    Example:
        Europe ranked second.
    """

    normalized_claim = normalize_text(
        claim
    )

    if expected_category is None:

        return {
            "status": "FAIL",
            "claim": claim,
            "claim_type": "ranking",
            "reason": (
                "Expected ranking category "
                "is missing from evidence."
            )
        }

    normalized_expected_category = (
        normalize_text(
            expected_category
        )
    )

    if (
        normalized_expected_category
        not in normalized_claim
    ):

        return {
            "status": "FAIL",
            "claim": claim,
            "claim_type": "ranking",
            "category": expected_category,
            "reason": (
                f"Expected category "
                f"'{expected_category}' "
                f"was not found in the claim."
            )
        }

    claimed_rank = extract_rank(
        claim
    )

    if claimed_rank is None:

        return {
            "status": "FAIL",
            "claim": claim,
            "claim_type": "ranking",
            "category": expected_category,
            "reason": (
                "Could not identify "
                "the claimed rank."
            )
        }

    if claimed_rank == expected_rank:

        return {
            "status": "PASS",
            "claim": claim,
            "claim_type": "ranking",
            "category": expected_category,
            "claimed_rank": claimed_rank,
            "expected_rank": expected_rank,
            "reason": (
                "Ranking claim matches "
                "the evidence."
            )
        }

    return {
        "status": "FAIL",
        "claim": claim,
        "claim_type": "ranking",
        "category": expected_category,
        "claimed_rank": claimed_rank,
        "expected_rank": expected_rank,
        "reason": (
            f"Expected {expected_category} "
            f"to rank {expected_rank}, "
            f"but claim states rank "
            f"{claimed_rank}."
        )
    }




# ======================================================================
# CONVERSION-SPECIFIC CLAIM VALIDATION
# ======================================================================

def _find_conversion_result(evidence):
    """Find conversion metrics in nested or flat evidence structures."""
    if not isinstance(evidence, dict):
        return None
    metric_keys = {
        "conversion_rate_percent", "converted_count_conversion_1",
        "non_converted_count_conversion_0", "total_records",
        "row_by_row_matches", "row_by_row_mismatches"
    }
    if metric_keys.intersection(evidence.keys()):
        return evidence
    for key in ("evidence", "supporting_result", "primary_result", "result"):
        found = _find_conversion_result(evidence.get(key))
        if found is not None:
            return found
    return None


def validate_conversion_claim(claim, evidence, metric_key, tolerance=0.02):
    """Validate a conversion metric by its named evidence field, not the first number in text."""
    result = _find_conversion_result(evidence)
    expected = result.get(metric_key) if result else None
    if expected is None:
        return {
            "status": "FAIL", "claim": claim, "claim_type": "conversion_metric",
            "metric": metric_key, "reason": f"Evidence field '{metric_key}' is missing."
        }

    text = normalize_text(claim)
    # Find numbers with optional commas, decimals and percent signs.
    matches = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", str(claim))
    claimed_values = []
    for token in matches:
        try:
            claimed_values.append(float(token.replace(",", "")))
        except ValueError:
            continue

    expected_float = float(expected)
    # For percentages, allow the expected percent rounded to 2 decimal places.
    matched = any(abs(value - expected_float) <= max(tolerance, abs(expected_float) * 0.0001)
                  for value in claimed_values)
    if matched:
        return {
            "status": "PASS", "claim": claim, "claim_type": "conversion_metric",
            "metric": metric_key, "claimed_values": claimed_values,
            "expected_value": expected_float, "reason": "Claim matches the named evidence metric."
        }
    return {
        "status": "FAIL", "claim": claim, "claim_type": "conversion_metric",
        "metric": metric_key, "claimed_values": claimed_values,
        "expected_value": expected_float, "reason": "Claim does not contain the expected evidence value."
    }


def validate_conversion_consistency_claim(claim, evidence):
    """Validate a claim that conversion and y are equivalent row by row."""
    result = _find_conversion_result(evidence)
    if result is None:
        return {
            "status": "FAIL", "claim": claim, "claim_type": "conversion_consistency",
            "reason": "Conversion evidence is missing."
        }
    matches = result.get("row_by_row_matches")
    compared = result.get("compared_records")
    mismatches = result.get("row_by_row_mismatches")
    unmapped = result.get("unmapped_y_values", 0)
    unmatched = result.get("unmatched_conversion_values", 0)
    passed = (
        matches is not None and compared is not None and mismatches is not None
        and int(matches) == int(compared) and int(mismatches) == 0
        and int(unmapped or 0) == 0 and int(unmatched or 0) == 0
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "claim": claim, "claim_type": "conversion_consistency",
        "row_by_row_matches": matches, "compared_records": compared,
        "row_by_row_mismatches": mismatches,
        "reason": "All mapped labels match row by row." if passed
                  else "Evidence shows missing, unmapped, or mismatched labels."
    }


# ======================================================================
# DATASET STRUCTURE AND DATA-QUALITY CLAIM VALIDATION
# ======================================================================

def _find_named_value(data, accepted_keys):
    """Recursively find a value stored under one of the accepted keys."""
    if not isinstance(data, dict):
        return None
    for key, value in data.items():
        if str(key).strip().lower() in accepted_keys and not isinstance(value, (dict, list, tuple)):
            return value
    for value in data.values():
        found = _find_named_value(value, accepted_keys)
        if found is not None:
            return found
    return None

def validate_dataset_dimensions_claim(claim, evidence):
    """Validate row and column counts against explicitly named evidence fields."""
    row_keys = {"row_count", "rows", "total_rows", "n_rows", "dataset_rows", "num_rows"}
    column_keys = {"column_count", "columns_count", "total_columns", "n_columns", "dataset_columns_count", "num_columns"}
    expected_rows = _find_named_value(evidence, row_keys)
    expected_columns = _find_named_value(evidence, column_keys)
    text = str(claim or "")
    row_match = re.search(r"([\d,]+)\s+rows?", text, flags=re.IGNORECASE)
    column_match = re.search(r"([\d,]+)\s+columns?", text, flags=re.IGNORECASE)
    claimed_rows = int(row_match.group(1).replace(",", "")) if row_match else None
    claimed_columns = int(column_match.group(1).replace(",", "")) if column_match else None
    if expected_rows is None or expected_columns is None:
        return {"status": "FAIL", "claim": claim, "claim_type": "dataset_dimensions",
                "claimed_rows": claimed_rows, "claimed_columns": claimed_columns,
                "reason": "Evidence must contain explicitly named row_count and column_count values."}
    try:
        expected_rows, expected_columns = int(expected_rows), int(expected_columns)
    except (TypeError, ValueError):
        return {"status": "FAIL", "claim": claim, "claim_type": "dataset_dimensions",
                "reason": "Expected row/column evidence values are not integers."}
    passed = claimed_rows == expected_rows and claimed_columns == expected_columns
    return {"status": "PASS" if passed else "FAIL", "claim": claim,
            "claim_type": "dataset_dimensions", "claimed_rows": claimed_rows,
            "expected_rows": expected_rows, "claimed_columns": claimed_columns,
            "expected_columns": expected_columns,
            "reason": "Dataset dimensions match the evidence." if passed else "Reported row/column counts do not match the dataset."}

def validate_data_quality_count_claim(claim, evidence, claim_type):
    """Validate a missing-cell or duplicate-row count against named evidence."""
    key_map = {
        "missing_cells": {"total_missing", "total_missing_cells", "missing_cells", "missing_value_count"},
        "duplicate_rows": {"duplicate_rows", "duplicate_row_count", "duplicates_count", "exact_duplicate_rows"},
    }
    expected = _find_named_value(evidence, key_map[claim_type])
    if expected is None:
        return {"status": "FAIL", "claim": claim, "claim_type": claim_type,
                "reason": "The corresponding named data-quality evidence value is missing."}
    try:
        expected = int(expected)
    except (TypeError, ValueError):
        return {"status": "FAIL", "claim": claim, "claim_type": claim_type,
                "reason": "The expected data-quality evidence value is not an integer."}
    numbers = re.findall(r"\d[\d,]*", str(claim or ""))
    values = [int(item.replace(",", "")) for item in numbers]
    passed = expected in values
    return {"status": "PASS" if passed else "FAIL", "claim": claim,
            "claim_type": claim_type, "expected_value": expected,
            "claimed_values": values,
            "reason": "Claim matches the named evidence value." if passed else "Claim does not contain the expected evidence value."}


# ======================================================================
# CLAIM VALIDATION
# ======================================================================

def validate_claim(
    claim,
    evidence,
    claim_type
):
    """
    Validate one individual claim.
    """

    if claim_type == "dataset_dimensions":
        return validate_dataset_dimensions_claim(claim, evidence)

    if claim_type in ("missing_cells", "duplicate_rows"):
        return validate_data_quality_count_claim(claim, evidence, claim_type)

    if claim_type == "highest_category":

        return validate_highest_category_claim(
            claim,
            evidence
        )

    if claim_type == "numeric_value":

        expected_value = get_evidence_value(
            evidence,
            "top_value"
        )

        return validate_value_claim(
            claim,
            expected_value
        )

    if claim_type == "ranking":

        expected_category = get_evidence_value(
            evidence,
            "second_category"
        )

        return validate_rank_claim(
            claim,
            expected_category,
            2
        )

    if claim_type in ("conversion_rate", "converted_count", "non_converted_count",
                      "total_records", "non_conversion_rate", "conversion_metric"):
        metric_map = {
            "conversion_rate": "conversion_rate_percent",
            "converted_count": "converted_count_conversion_1",
            "non_converted_count": "non_converted_count_conversion_0",
            "total_records": "total_records",
            "non_conversion_rate": "non_conversion_rate_percent",
        }
        metric_key = metric_map.get(claim_type, claim_type)
        return validate_conversion_claim(claim, evidence, metric_key)

    if claim_type == "conversion_consistency":
        return validate_conversion_consistency_claim(claim, evidence)

    return {
        "status": "FAIL",
        "claim": claim,
        "claim_type": claim_type,
        "reason": (
            f"Unsupported claim type: "
            f"{claim_type}"
        )
    }


# ======================================================================
# MULTIPLE CLAIM VALIDATION
# ======================================================================

def validate_claims(
    claims
):
    """
    Validate multiple claims and calculate
    claim-level metrics.
    """

    results = []

    for item in claims:

        claim = item.get(
            "claim"
        )

        claim_type = item.get(
            "claim_type"
        )

        evidence = item.get(
            "evidence",
            {}
        )

        result = validate_claim(
            claim,
            evidence,
            claim_type
        )

        results.append(
            result
        )

    total_claims = len(
        results
    )

    passed_claims = sum(
        1
        for result in results
        if result.get(
            "status"
        ) == "PASS"
    )

    failed_claims = (
        total_claims
        -
        passed_claims
    )

    accuracy = (
        passed_claims
        /
        total_claims
        if total_claims > 0
        else 0.0
    )

    overall_status = (
        "PASS"
        if total_claims > 0
        and failed_claims == 0
        else "FAIL"
    )

    return {
        "overall_status": overall_status,
        "total_claims": total_claims,
        "passed_claims": passed_claims,
        "failed_claims": failed_claims,
        "claim_accuracy": accuracy,
        "results": results
    }


# ======================================================================
# STANDALONE TEST
# ======================================================================

if __name__ == "__main__":

    print("=" * 70)
    print(
        "CLAIM VALIDATOR - STANDALONE TEST"
    )
    print("=" * 70)

    evidence = {

        "status":
            "PASS",

        "analysis_type":
            "grouped_analysis",

        "evidence": {

            "top_category":
                "Sub-Saharan Africa",

            "top_value":
                12183211.4,

            "second_category":
                "Europe",

            "second_value":
                11082938.63
        },

        "supporting_result": {

            "Sub-Saharan Africa":
                12183211.4,

            "Europe":
                11082938.63,

            "Asia":
                6113845.87
        }
    }

    claims = [

        {
            "claim": (
                "Sub-Saharan Africa generated "
                "the highest total profit."
            ),
            "claim_type":
                "highest_category",
            "evidence":
                evidence
        },

        {
            "claim": (
                "Sub-Saharan Africa generated "
                "approximately 12,183,211.40."
            ),
            "claim_type":
                "numeric_value",
            "evidence":
                evidence
        },

        {
            "claim":
                "Europe ranked second.",
            "claim_type":
                "ranking",
            "evidence":
                evidence
        },

        {
            "claim":
                "Asia ranked second.",
            "claim_type":
                "ranking",
            "evidence":
                evidence
        }
    ]

    result = validate_claims(
        claims
    )

    print("\nValidation Result:")
    print(result)

    print("\n" + "=" * 70)
    print(
        "STANDALONE TEST COMPLETE"
    )
    print("=" * 70)