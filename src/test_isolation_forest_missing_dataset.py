
import os
import sys

import pytest


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SRC_PATH = os.path.join(PROJECT_ROOT, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)


# ============================================================
# PROJECT MODULE
# ============================================================

from self_correcting_step_executor import (
    execute_self_correcting_step,
)


# ============================================================
# TEST CONFIGURATION
# ============================================================

ISOLATION_FOREST_STEP = (
    "Detect multivariate anomalies using Isolation Forest "
    "across suitable numeric features"
)


# ============================================================
# REGRESSION TESTS
# ============================================================

def test_missing_dataset_returns_structured_failure(tmp_path):
    """A missing dataset should return a structured failure."""

    missing_file = tmp_path / "dataset_does_not_exist.csv"

    result = execute_self_correcting_step(
        step_number=1,
        step_text=ISOLATION_FOREST_STEP,
        filepath=str(missing_file),
    )

    assert result["status"] == "UNRESOLVED"
    assert result["total_attempts"] == 1
    assert result["final_validation"]["overall"] == "FAIL"

    assert result["final_execution"]["status"] == "FAIL"
    assert result["final_execution"]["success"] is False
    assert "FileNotFoundError" in result["final_execution"]["stderr"]
    assert str(missing_file) in result["final_execution"]["stderr"]

    assert result["final_result"] is None
    assert len(result["attempts"]) == 1
    assert result["attempts"][0]["validation"]["overall"] == "FAIL"


def test_missing_dataset_error_is_consistent_in_attempt_history(
    tmp_path,
):
    """The recorded attempt should contain the same error."""

    missing_file = tmp_path / "missing_bank_marketing.csv"

    result = execute_self_correcting_step(
        step_number=2,
        step_text=ISOLATION_FOREST_STEP,
        filepath=str(missing_file),
    )

    attempt = result["attempts"][0]
    error = result["final_execution"]["stderr"]

    assert attempt["execution"]["stderr"] == error
    assert attempt["validation"]["overall"] == "FAIL"
    assert attempt["feedback"]["overall"] == "FAIL"
