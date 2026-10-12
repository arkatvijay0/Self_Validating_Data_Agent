import os
import sys
import json
import pandas as pd
import numpy as np


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# IMPORT PROJECT MODULES
# ============================================================

from src.data_profiler import profile_dataset
from src.task_analyzer import analyze_task_mock
from src.eda_planner import generate_plan_mock
from src.self_correcting_step_executor import (
    execute_self_correcting_step
)
from src.result_aggregator import aggregate_results


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "sales.csv"
)

OUTPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "test_3f_multitask_generalization_results.json"
)


# ============================================================
# MULTI-TASK QUESTIONS
# ============================================================

TEST_QUESTIONS = [

    {
        "id": 1,
        "question":
            "Which region generated the highest total profit?",
        "expected_task_type":
            "comparison"
    },

    {
        "id": 2,
        "question":
            "How did total revenue change over time?",
        "expected_task_type":
            "trend_analysis"
    },

    {
        "id": 3,
        "question":
            "Is there a relationship between units sold and total profit?",
        "expected_task_type":
            "relationship_analysis"
    },

    {
        "id": 4,
        "question":
            "Are there any unusual total profit values?",
        "expected_task_type":
            "outlier_detection"
    },

    {
        "id": 5,
        "question":
            "Give me a summary of the sales dataset.",
        "expected_task_type":
            "descriptive_analysis"
    }
]


# ============================================================
# TASK TYPE NORMALIZATION
# ============================================================

TASK_TYPE_ALIASES = {

    "comparison":
        "comparison",

    "trend_analysis":
        "trend_analysis",

    "relationship_analysis":
        "relationship_analysis",

    "correlation_analysis":
        "relationship_analysis",

    "outlier_detection":
        "outlier_detection",

    "anomaly_detection":
        "outlier_detection",

    "descriptive_analysis":
        "descriptive_analysis",

    "descriptive_summary":
        "descriptive_analysis"
}


def normalize_task_type(task_type):
    """
    Normalize equivalent task-type labels so that
    terminology differences do not create false failures.
    """

    if task_type is None:
        return None

    return TASK_TYPE_ALIASES.get(
        task_type,
        task_type
    )


# ============================================================
# EXECUTION STATUS HELPER
# ============================================================

def extract_execution_status(attempt):

    if not isinstance(attempt, dict):
        return None

    direct_status = attempt.get(
        "execution_status"
    )

    if direct_status is not None:
        return direct_status

    execution = attempt.get(
        "execution"
    )

    if isinstance(execution, dict):
        return execution.get(
            "status"
        )

    execution_result = attempt.get(
        "execution_result"
    )

    if isinstance(execution_result, dict):
        return execution_result.get(
            "status"
        )

    return None


# ============================================================
# VALIDATION STATUS HELPER
# ============================================================

def extract_validation_status(attempt):

    if not isinstance(attempt, dict):
        return None

    direct_status = attempt.get(
        "validation_status"
    )

    if direct_status is not None:
        return direct_status

    validation = attempt.get(
        "validation"
    )

    if isinstance(validation, dict):

        return (
            validation.get("overall")
            or validation.get("status")
        )

    validation_result = attempt.get(
        "validation_result"
    )

    if isinstance(validation_result, dict):

        return (
            validation_result.get("overall")
            or validation_result.get("status")
        )

    return None


# ============================================================
# JSON SAFE CONVERSION
# ============================================================

def safe_json_value(value):
    """
    Recursively convert Pandas / NumPy objects into
    JSON-compatible values.

    Important:
    Pandas Period objects are converted to strings,
    including when they appear as dictionary keys.
    """

    # --------------------------------------------------------
    # Pandas Period
    # --------------------------------------------------------

    if isinstance(
        value,
        pd.Period
    ):

        return str(value)


    # --------------------------------------------------------
    # Pandas Timestamp
    # --------------------------------------------------------

    if isinstance(
        value,
        pd.Timestamp
    ):

        return value.isoformat()


    # --------------------------------------------------------
    # NumPy scalar
    # --------------------------------------------------------

    if isinstance(
        value,
        np.generic
    ):

        return value.item()


    # --------------------------------------------------------
    # DataFrame
    # --------------------------------------------------------

    if isinstance(
        value,
        pd.DataFrame
    ):

        records = value.to_dict(
            orient="records"
        )

        return safe_json_value(
            records
        )


    # --------------------------------------------------------
    # Series
    # --------------------------------------------------------

    if isinstance(
        value,
        pd.Series
    ):

        result = {}

        for key, val in value.items():

            result[str(key)] = (
                safe_json_value(val)
            )

        return result


    # --------------------------------------------------------
    # Dictionary
    # --------------------------------------------------------

    if isinstance(
        value,
        dict
    ):

        result = {}

        for key, val in value.items():

            # Every JSON dictionary key must be
            # a supported primitive type.
            safe_key = str(key)

            result[safe_key] = (
                safe_json_value(val)
            )

        return result


    # --------------------------------------------------------
    # List / Tuple
    # --------------------------------------------------------

    if isinstance(
        value,
        (list, tuple)
    ):

        return [
            safe_json_value(item)
            for item in value
        ]


    # --------------------------------------------------------
    # Set
    # --------------------------------------------------------

    if isinstance(
        value,
        set
    ):

        return [
            safe_json_value(item)
            for item in value
        ]


    # --------------------------------------------------------
    # None / string / number / bool
    # --------------------------------------------------------

    return value


# ============================================================
# RUN ONE QUESTION
# ============================================================

def run_single_question(
    question_record,
    data_profile
):

    question_id = question_record[
        "id"
    ]

    question = question_record[
        "question"
    ]

    expected_task_type = question_record[
        "expected_task_type"
    ]


    # ========================================================
    # HEADER
    # ========================================================

    print()
    print("=" * 70)

    print(
        f"TEST QUESTION {question_id}"
    )

    print("=" * 70)

    print()

    print("Question:")
    print(question)

    print()

    print("Expected Task Type:")
    print(expected_task_type)

    print()


    # ========================================================
    # TASK ANALYSIS
    # ========================================================

    print("-" * 70)
    print("TASK ANALYSIS")
    print("-" * 70)

    task_analysis = analyze_task_mock(
        question,
        data_profile
    )

    actual_task_type = task_analysis.get(
        "task_type"
    )

    normalized_expected = (
        normalize_task_type(
            expected_task_type
        )
    )

    normalized_actual = (
        normalize_task_type(
            actual_task_type
        )
    )

    task_type_match = (
        normalized_expected ==
        normalized_actual
    )

    print()

    print("Task Analysis:")
    print(task_analysis)

    print()

    print(
        f"Actual Task Type: "
        f"{actual_task_type}"
    )

    print(
        f"Normalized Expected: "
        f"{normalized_expected}"
    )

    print(
        f"Normalized Actual: "
        f"{normalized_actual}"
    )

    print(
        f"Semantic Task Type Match: "
        f"{task_type_match}"
    )

    print()


    # ========================================================
    # EDA PLAN
    # ========================================================

    print("-" * 70)
    print("EDA PLAN")
    print("-" * 70)

    plan = generate_plan_mock(
        question,
        task_analysis,
        data_profile
    )

    print()

    for index, step in enumerate(
        plan,
        start=1
    ):

        print(
            f"{index}. {step}"
        )

    print()


    # ========================================================
    # EDA EXECUTION
    # ========================================================

    print("-" * 70)
    print("EDA EXECUTION")
    print("-" * 70)

    step_results = []

    for step_number, step_text in enumerate(
        plan,
        start=1
    ):

        print()

        print(
            f"STEP {step_number}"
        )

        print(
            f"Step: {step_text}"
        )

        print()

        result = execute_self_correcting_step(
            step_number=step_number,
            step_text=step_text,
            filepath=DATASET_PATH,
            test_fault=False
        )

        step_results.append(
            result
        )

        final_execution = (
            result.get(
                "final_execution"
            )
            or {}
        )

        final_validation = (
            result.get(
                "final_validation"
            )
            or {}
        )

        print(
            f"Status: "
            f"{result.get('status')}"
        )

        print(
            f"Attempts: "
            f"{result.get('total_attempts')}"
        )

        print(
            f"Execution: "
            f"{final_execution.get('status')}"
        )

        print(
            f"Validation: "
            f"{final_validation.get('overall')}"
        )


    # ========================================================
    # STEP METRICS
    # ========================================================

    total_steps = len(
        step_results
    )

    successful_steps = sum(
        1
        for result in step_results
        if result.get("status") == "PASS"
    )

    failed_steps = (
        total_steps -
        successful_steps
    )

    total_attempts = sum(
        result.get(
            "total_attempts",
            0
        )
        for result in step_results
    )

    first_attempt_successes = 0

    for result in step_results:

        attempts = result.get(
            "attempts",
            []
        )

        if len(attempts) == 1:

            execution_status = (
                extract_execution_status(
                    attempts[0]
                )
            )

            validation_status = (
                extract_validation_status(
                    attempts[0]
                )
            )

            if (
                execution_status == "SUCCESS"
                and
                validation_status == "PASS"
            ):

                first_attempt_successes += 1


    first_attempt_success_rate = (
        first_attempt_successes /
        total_steps *
        100
        if total_steps > 0
        else 0
    )


    # ========================================================
    # RESULT AGGREGATION
    # ========================================================

    print()

    print("-" * 70)
    print("RESULT AGGREGATION")
    print("-" * 70)

    aggregation = aggregate_results(
        question,
        step_results,
        task_analysis
    )

    aggregation_status = (
        aggregation.get(
            "overall_status"
        )
    )

    print()

    print(
        f"Aggregation Status: "
        f"{aggregation_status}"
    )

    print()

    print("Primary Result:")

    print(
        aggregation.get(
            "primary_result"
        )
    )


    # ========================================================
    # QUESTION RESULT
    # ========================================================

    question_pass = (
        task_type_match
        and
        total_steps > 0
        and
        successful_steps == total_steps
        and
        failed_steps == 0
        and
        aggregation_status == "PASS"
    )

    print()

    print("-" * 70)

    if question_pass:

        print(
            f"QUESTION {question_id} RESULT: PASS"
        )

    else:

        print(
            f"QUESTION {question_id} RESULT: FAIL"
        )

    print("-" * 70)


    # ========================================================
    # RETURN
    # ========================================================

    return {

        "question_id":
            question_id,

        "question":
            question,

        "expected_task_type":
            expected_task_type,

        "actual_task_type":
            actual_task_type,

        "normalized_expected_task_type":
            normalized_expected,

        "normalized_actual_task_type":
            normalized_actual,

        "task_type_match":
            task_type_match,

        "task_analysis":
            task_analysis,

        "plan":
            plan,

        "total_steps":
            total_steps,

        "successful_steps":
            successful_steps,

        "failed_steps":
            failed_steps,

        "total_attempts":
            total_attempts,

        "first_attempt_successes":
            first_attempt_successes,

        "first_attempt_success_rate":
            first_attempt_success_rate,

        "step_results":
            step_results,

        "aggregation":
            aggregation,

        "aggregation_status":
            aggregation_status,

        "overall_result":
            (
                "PASS"
                if question_pass
                else "FAIL"
            )
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)

    print(
        "TEST 3F - MULTI-QUESTION / MULTI-TASK "
        "GENERALIZATION EVALUATION"
    )

    print("=" * 70)

    print()

    print("Dataset:")
    print(DATASET_PATH)

    print()

    print(
        "Artificial Fault Injection:"
    )

    print("DISABLED")

    print()

    print(
        "Number of Test Questions:"
    )

    print(
        len(TEST_QUESTIONS)
    )

    print()


    # ========================================================
    # DATA PROFILING
    # ========================================================

    print("=" * 70)
    print("DATA PROFILING")
    print("=" * 70)

    data_profile = profile_dataset(
        DATASET_PATH
    )

    print()

    print(
        "Data Profile Generated Successfully."
    )

    print()


    # ========================================================
    # RUN QUESTIONS
    # ========================================================

    question_results = []

    for question_record in TEST_QUESTIONS:

        result = run_single_question(
            question_record,
            data_profile
        )

        question_results.append(
            result
        )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print()
    print("=" * 70)

    print(
        "TEST 3F - FINAL GENERALIZATION SUMMARY"
    )

    print("=" * 70)


    # ========================================================
    # BASIC COUNTS
    # ========================================================

    total_questions = len(
        question_results
    )

    successful_questions = sum(
        1
        for result in question_results
        if result.get(
            "overall_result"
        ) == "PASS"
    )

    failed_questions = (
        total_questions -
        successful_questions
    )

    total_steps = sum(
        result.get(
            "total_steps",
            0
        )
        for result in question_results
    )

    successful_steps = sum(
        result.get(
            "successful_steps",
            0
        )
        for result in question_results
    )

    failed_steps = sum(
        result.get(
            "failed_steps",
            0
        )
        for result in question_results
    )

    total_attempts = sum(
        result.get(
            "total_attempts",
            0
        )
        for result in question_results
    )

    first_attempt_successes = sum(
        result.get(
            "first_attempt_successes",
            0
        )
        for result in question_results
    )

    task_type_matches = sum(
        1
        for result in question_results
        if result.get(
            "task_type_match"
        )
    )

    aggregation_passes = sum(
        1
        for result in question_results
        if result.get(
            "aggregation_status"
        ) == "PASS"
    )


    # ========================================================
    # METRICS
    # ========================================================

    question_success_rate = (
        successful_questions /
        total_questions *
        100
        if total_questions > 0
        else 0
    )

    step_success_rate = (
        successful_steps /
        total_steps *
        100
        if total_steps > 0
        else 0
    )

    first_attempt_success_rate = (
        first_attempt_successes /
        total_steps *
        100
        if total_steps > 0
        else 0
    )

    task_classification_accuracy = (
        task_type_matches /
        total_questions *
        100
        if total_questions > 0
        else 0
    )

    aggregation_success_rate = (
        aggregation_passes /
        total_questions *
        100
        if total_questions > 0
        else 0
    )

    average_attempts = (
        total_attempts /
        total_steps
        if total_steps > 0
        else 0
    )


    # ========================================================
    # PRINT METRICS
    # ========================================================

    print()

    print(
        f"Total Questions             : "
        f"{total_questions}"
    )

    print(
        f"Successful Questions        : "
        f"{successful_questions}"
    )

    print(
        f"Failed Questions            : "
        f"{failed_questions}"
    )

    print()

    print(
        f"Task Classification Accuracy: "
        f"{task_classification_accuracy:.2f}%"
    )

    print(
        f"Total EDA Steps             : "
        f"{total_steps}"
    )

    print(
        f"Successful EDA Steps        : "
        f"{successful_steps}"
    )

    print(
        f"Failed EDA Steps            : "
        f"{failed_steps}"
    )

    print(
        f"Step Success Rate           : "
        f"{step_success_rate:.2f}%"
    )

    print()

    print(
        f"Total Attempts              : "
        f"{total_attempts}"
    )

    print(
        f"Average Attempts            : "
        f"{average_attempts:.2f}"
    )

    print(
        f"First-Attempt Success Rate  : "
        f"{first_attempt_success_rate:.2f}%"
    )

    print(
        f"Aggregation Success Rate    : "
        f"{aggregation_success_rate:.2f}%"
    )


    # ========================================================
    # QUESTION SUMMARY
    # ========================================================

    print()
    print("=" * 70)

    print(
        "QUESTION-BY-QUESTION RESULTS"
    )

    print("=" * 70)

    for result in question_results:

        print()

        print(
            f"Question {result['question_id']}: "
            f"{result['question']}"
        )

        print(
            f"Expected Task : "
            f"{result['expected_task_type']}"
        )

        print(
            f"Actual Task   : "
            f"{result['actual_task_type']}"
        )

        print(
            f"Normalized    : "
            f"{result['normalized_actual_task_type']}"
        )

        print(
            f"Task Match    : "
            f"{result['task_type_match']}"
        )

        print(
            f"Steps         : "
            f"{result['successful_steps']}/"
            f"{result['total_steps']}"
        )

        print(
            f"Attempts      : "
            f"{result['total_attempts']}"
        )

        print(
            f"Aggregation   : "
            f"{result['aggregation_status']}"
        )

        print(
            f"Result        : "
            f"{result['overall_result']}"
        )


    # ========================================================
    # FINAL DECISION
    # ========================================================

    overall_pass = (
        successful_questions ==
        total_questions
        and
        failed_questions == 0
        and
        failed_steps == 0
        and
        task_type_matches ==
        total_questions
        and
        aggregation_passes ==
        total_questions
    )


    print()
    print("=" * 70)

    if overall_pass:

        print(
            "TEST 3F RESULT: PASS"
        )

    else:

        print(
            "TEST 3F RESULT: FAIL"
        )

    print("=" * 70)

    print()

    if overall_pass:

        print(
            "The Self-Validating EDA Agent successfully "
            "handled multiple analytical questions."
        )

        print(
            "The system demonstrated semantic task "
            "classification across multiple analytical categories."
        )

        print(
            "EDA execution succeeded across all tested questions."
        )

        print(
            "Result aggregation succeeded across all tested questions."
        )

    else:

        print(
            "One or more multi-task generalization "
            "tests failed."
        )


    # ========================================================
    # SAVE RESULTS
    # ========================================================

    final_results = {

        "test_name":
            "TEST 3F - Multi-Question / Multi-Task "
            "Generalization Evaluation",

        "dataset":
            DATASET_PATH,

        "artificial_fault_injection":
            False,

        "total_questions":
            total_questions,

        "successful_questions":
            successful_questions,

        "failed_questions":
            failed_questions,

        "question_success_rate":
            question_success_rate,

        "task_classification_accuracy":
            task_classification_accuracy,

        "total_steps":
            total_steps,

        "successful_steps":
            successful_steps,

        "failed_steps":
            failed_steps,

        "step_success_rate":
            step_success_rate,

        "total_attempts":
            total_attempts,

        "average_attempts":
            average_attempts,

        "first_attempt_success_rate":
            first_attempt_success_rate,

        "aggregation_success_rate":
            aggregation_success_rate,

        "overall_result":
            (
                "PASS"
                if overall_pass
                else "FAIL"
            ),

        "questions":
            question_results
    }


    # ========================================================
    # JSON SAFE CONVERSION
    # ========================================================

    final_results = safe_json_value(
        final_results
    )


    # ========================================================
    # SAVE
    # ========================================================

    os.makedirs(
        os.path.dirname(
            OUTPUT_PATH
        ),
        exist_ok=True
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            final_results,
            file,
            indent=4,
            ensure_ascii=False
        )


    print()

    print(
        "Results saved to:"
    )

    print(
        OUTPUT_PATH
    )

    print()


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()