import os
import sys
import json
import pandas as pd

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
from src.self_correcting_step_executor import execute_self_correcting_step
from src.result_aggregator import aggregate_results
from src.insight_generator import generate_insight
from src.insight_validator import validate_insight
from src.evidence_mapping import (
    create_correlation_evidence,
    map_insight_to_evidence
)
from src.recommendation_generator import generate_recommendation


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "sales.csv"
)

QUESTION = (
    "Is there a relationship between units sold and total profit?"
)

TEST_FAULT = False

OUTPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "test_3e_production_results.json"
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def extract_execution_status(attempt):
    """
    Robustly extract execution status from an attempt record.
    Supports multiple result structures.
    """

    if not isinstance(attempt, dict):
        return None

    direct_status = attempt.get("execution_status")

    if direct_status is not None:
        return direct_status

    execution = attempt.get("execution")

    if isinstance(execution, dict):
        return execution.get("status")

    execution_result = attempt.get("execution_result")

    if isinstance(execution_result, dict):
        return execution_result.get("status")

    return None


def extract_validation_status(attempt):
    """
    Robustly extract validation status from an attempt record.
    Supports multiple result structures.
    """

    if not isinstance(attempt, dict):
        return None

    direct_status = attempt.get("validation_status")

    if direct_status is not None:
        return direct_status

    validation = attempt.get("validation")

    if isinstance(validation, dict):
        return (
            validation.get("overall")
            or validation.get("status")
        )

    validation_result = attempt.get("validation_result")

    if isinstance(validation_result, dict):
        return (
            validation_result.get("overall")
            or validation_result.get("status")
        )

    return None


def safe_json_value(value):
    """
    Convert NumPy/Pandas values into JSON-safe values.
    """

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")

    if isinstance(value, pd.Series):
        return value.to_dict()

    if isinstance(value, dict):
        return {
            str(k): safe_json_value(v)
            for k, v in value.items()
        }

    if isinstance(value, list):
        return [
            safe_json_value(v)
            for v in value
        ]

    return value


# ============================================================
# MAIN TEST
# ============================================================

def main():

    print("=" * 70)
    print("TEST 3E - CLEAN END-TO-END PRODUCTION EVALUATION")
    print("=" * 70)

    print("Dataset:")
    print(DATASET_PATH)

    print()

    print("User Question:")
    print(QUESTION)

    print()

    print("TEST MODE:")
    print("DISABLED")

    print()

    print("No artificial faults will be injected.")

    print()
    print("=" * 70)

    # ========================================================
    # STEP 1 - DATA PROFILING
    # ========================================================

    print("STEP 1 - DATA PROFILING")
    print("=" * 70)

    data_profile = profile_dataset(DATASET_PATH)

    print("Data Profile Generated Successfully.")

    print()

    # ========================================================
    # STEP 2 - TASK ANALYSIS
    # ========================================================

    print("STEP 2 - TASK ANALYSIS")
    print("=" * 70)

    task_analysis = analyze_task_mock(
        QUESTION,
        data_profile
    )

    print()
    print("Task Analysis:")
    print(task_analysis)

    print()

    # ========================================================
    # STEP 3 - EDA PLAN
    # ========================================================

    print("STEP 3 - EDA PLAN")
    print("=" * 70)

    plan = generate_plan_mock(
        QUESTION,
        task_analysis,
        data_profile
    )

    print("Generated Plan:")

    for index, step in enumerate(plan, start=1):
        print(f"{index}. {step}")

    print()

    # ========================================================
    # STEP 4 - CLEAN PRODUCTION EXECUTION
    # ========================================================

    print("STEP 4 - CLEAN PRODUCTION EXECUTION")
    print("=" * 70)

    print("Artificial fault injection: DISABLED")
    print("Each EDA step will execute normally.")

    print()

    step_results = []

    for step_number, step_text in enumerate(plan, start=1):

        print("=" * 70)
        print(
            f"EXECUTING EDA PLAN STEP {step_number}"
        )
        print("=" * 70)

        print()

        print("Step:")
        print(step_text)

        print()

        # ----------------------------------------------------
        # IMPORTANT FIX:
        # filepath=DATASET_PATH
        # ----------------------------------------------------

        result = execute_self_correcting_step(
            step_number=step_number,
            step_text=step_text,
            filepath=DATASET_PATH,
            test_fault=TEST_FAULT
        )

        step_results.append(result)

        print()

        print(
            f"Step {step_number} Status: "
            f"{result.get('status')}"
        )

        print(
            f"Attempts: "
            f"{result.get('total_attempts')}"
        )

        print(
            f"Execution Status: "
            f"{(result.get('final_execution') or {}).get('status')}"
        )

        print(
            f"Validation Status: "
            f"{(result.get('final_validation') or {}).get('overall')}"
        )

        print()

    # ========================================================
    # STEP 5 - EXECUTION SUMMARY
    # ========================================================

    print("=" * 70)
    print("STEP 5 - PRODUCTION EXECUTION SUMMARY")
    print("=" * 70)

    total_steps = len(step_results)

    successful_steps = sum(
        1
        for result in step_results
        if result.get("status") == "PASS"
    )

    failed_steps = total_steps - successful_steps

    total_attempts = sum(
        result.get("total_attempts", 0)
        for result in step_results
    )

    print()

    print(f"Total Steps      : {total_steps}")
    print(f"Successful Steps : {successful_steps}")
    print(f"Failed Steps     : {failed_steps}")
    print(f"Total Attempts   : {total_attempts}")

    print()

    # ========================================================
    # STEP 6 - ATTEMPT HISTORY
    # ========================================================

    print("=" * 70)
    print("STEP 6 - ATTEMPT HISTORY")
    print("=" * 70)

    recovery_detected = False
    recovered_steps = 0

    for step_number, result in enumerate(
        step_results,
        start=1
    ):

        print()
        print(f"STEP {step_number}")
        print("-" * 50)

        attempts = result.get("attempts", [])

        for attempt_number, attempt in enumerate(
            attempts,
            start=1
        ):

            execution_status = extract_execution_status(
                attempt
            )

            validation_status = extract_validation_status(
                attempt
            )

            print(
                f"Attempt {attempt_number}: "
                f"Execution={execution_status}, "
                f"Validation={validation_status}"
            )

        # ----------------------------------------------------
        # Production test should normally succeed on attempt 1
        # ----------------------------------------------------

        if len(attempts) == 1:

            execution_status = extract_execution_status(
                attempts[0]
            )

            validation_status = extract_validation_status(
                attempts[0]
            )

            if (
                execution_status == "SUCCESS"
                and validation_status == "PASS"
            ):
                print(
                    "Result: Successful on first attempt."
                )

        elif len(attempts) > 1:

            first_execution = extract_execution_status(
                attempts[0]
            )

            first_validation = extract_validation_status(
                attempts[0]
            )

            final_execution = extract_execution_status(
                attempts[-1]
            )

            final_validation = extract_validation_status(
                attempts[-1]
            )

            if (
                first_execution != "SUCCESS"
                or first_validation != "PASS"
            ) and (
                final_execution == "SUCCESS"
                and final_validation == "PASS"
            ):
                recovery_detected = True
                recovered_steps += 1

                print(
                    "Result: Step recovered after retry."
                )

    print()

    # ========================================================
    # STEP 7 - PRODUCTION RELIABILITY METRICS
    # ========================================================

    print("=" * 70)
    print("STEP 7 - PRODUCTION RELIABILITY METRICS")
    print("=" * 70)

    average_attempts = (
        total_attempts / total_steps
        if total_steps > 0
        else 0
    )

    production_success_rate = (
        successful_steps / total_steps * 100
        if total_steps > 0
        else 0
    )

    first_attempt_successes = 0

    for result in step_results:

        attempts = result.get("attempts", [])

        if len(attempts) == 1:

            execution_status = extract_execution_status(
                attempts[0]
            )

            validation_status = extract_validation_status(
                attempts[0]
            )

            if (
                execution_status == "SUCCESS"
                and validation_status == "PASS"
            ):
                first_attempt_successes += 1

    first_attempt_success_rate = (
        first_attempt_successes / total_steps * 100
        if total_steps > 0
        else 0
    )

    print()

    print(
        f"Production Success Rate : "
        f"{production_success_rate:.2f}%"
    )

    print(
        f"First-Attempt Success Rate : "
        f"{first_attempt_success_rate:.2f}%"
    )

    print(
        f"Average Attempts : "
        f"{average_attempts:.2f}"
    )

    print(
        f"Recovery Detected : "
        f"{recovery_detected}"
    )

    print(
        f"Recovered Steps : "
        f"{recovered_steps}"
    )

    print()

    # ========================================================
    # STEP 8 - RESULT AGGREGATION
    # ========================================================

    print("=" * 70)
    print("STEP 8 - RESULT AGGREGATION")
    print("=" * 70)

    aggregation = aggregate_results(
        QUESTION,
        step_results,
        task_analysis
    )

    print()

    print("Aggregation Status:")
    print(
        aggregation.get(
            "overall_status"
        )
    )

    print()

    print("Primary Result:")
    print(
        aggregation.get(
            "primary_result"
        )
    )

    print()

    print("Supporting Results:")
    print(
        aggregation.get(
            "supporting_results"
        )
    )

    print()

    # ========================================================
    # STEP 9 - INSIGHT GENERATION
    # ========================================================

    print("=" * 70)
    print("STEP 9 - INSIGHT GENERATION")
    print("=" * 70)

    primary_result = aggregation.get(
        "primary_result"
    )

    insight = generate_insight(
        QUESTION,
        primary_result,
        task_analysis
    )

    print()

    print("Generated Insight:")
    print(insight)

    print()

    # ========================================================
    # STEP 10 - INSIGHT VALIDATION
    # ========================================================

    print("=" * 70)
    print("STEP 10 - INSIGHT VALIDATION")
    print("=" * 70)

    insight_validation = validate_insight(
        insight,
        primary_result,
        task_analysis
    )

    print()

    print("Insight Validation:")
    print(insight_validation)

    print()

    # ========================================================
    # STEP 11 - EVIDENCE MAPPING
    # ========================================================

    print("=" * 70)
    print("STEP 11 - EVIDENCE MAPPING")
    print("=" * 70)

    evidence = create_correlation_evidence(
        result=primary_result,
        x_column="Units Sold",
        y_column="Total Profit",
        method="Pearson"
    )

    print()

    print("Evidence:")
    print(evidence)

    print()

    mapped_evidence = map_insight_to_evidence(
        insight,
        evidence
    )

    print("Mapped Evidence:")
    print(mapped_evidence)

    print()

    # ========================================================
    # STEP 12 - RECOMMENDATION
    # ========================================================

    print("=" * 70)
    print("STEP 12 - RECOMMENDATION GENERATION")
    print("=" * 70)

    recommendation = generate_recommendation(
        task_type="relationship_analysis",
        evidence=evidence
    )

    print()

    print("Recommendation:")
    print(recommendation)

    print()

    # ========================================================
    # STEP 13 - FINAL PRODUCTION EVALUATION
    # ========================================================

    print("=" * 70)
    print("STEP 13 - FINAL PRODUCTION EVALUATION")
    print("=" * 70)

    aggregation_pass = (
        aggregation.get("overall_status") == "PASS"
    )

    insight_pass = (
        insight_validation.get("overall") == "PASS"
    )

    evidence_pass = (
        evidence.get("status") == "PASS"
    )

    recommendation_pass = (
        recommendation.get("status") == "PASS"
    )

    all_steps_pass = (
        successful_steps == total_steps
        and failed_steps == 0
    )

    production_test_pass = (
        all_steps_pass
        and aggregation_pass
        and insight_pass
        and evidence_pass
        and recommendation_pass
    )

    print()

    print(
        f"Total Steps             : {total_steps}"
    )

    print(
        f"Successful Steps        : {successful_steps}"
    )

    print(
        f"Failed Steps            : {failed_steps}"
    )

    print(
        f"Total Attempts          : {total_attempts}"
    )

    print(
        f"Average Attempts        : {average_attempts:.2f}"
    )

    print(
        f"Production Success Rate : "
        f"{production_success_rate:.2f}%"
    )

    print(
        f"First-Attempt Success   : "
        f"{first_attempt_success_rate:.2f}%"
    )

    print(
        f"Aggregation             : "
        f"{'PASS' if aggregation_pass else 'FAIL'}"
    )

    print(
        f"Insight Validation      : "
        f"{'PASS' if insight_pass else 'FAIL'}"
    )

    print(
        f"Evidence Mapping        : "
        f"{'PASS' if evidence_pass else 'FAIL'}"
    )

    print(
        f"Recommendation          : "
        f"{'PASS' if recommendation_pass else 'FAIL'}"
    )

    print()

    # ========================================================
    # FINAL RESULT
    # ========================================================

    if production_test_pass:

        print("=" * 70)
        print("TEST 3E RESULT: PASS")
        print("=" * 70)

        print()
        print(
            "The clean production pipeline successfully completed "
            "without artificial fault injection."
        )

        print(
            "All EDA steps executed successfully."
        )

        print(
            "All validation stages passed."
        )

        print(
            "Result aggregation passed."
        )

        print(
            "Insight validation passed."
        )

        print(
            "Evidence mapping passed."
        )

        print(
            "Recommendation generation passed."
        )

    else:

        print("=" * 70)
        print("TEST 3E RESULT: FAIL")
        print("=" * 70)

        print()
        print(
            "One or more production pipeline stages failed."
        )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    final_results = {
        "test_name": "TEST 3E - Clean End-to-End Production Evaluation",
        "dataset": DATASET_PATH,
        "question": QUESTION,
        "test_fault": TEST_FAULT,
        "total_steps": total_steps,
        "successful_steps": successful_steps,
        "failed_steps": failed_steps,
        "total_attempts": total_attempts,
        "average_attempts": average_attempts,
        "production_success_rate": production_success_rate,
        "first_attempt_success_rate": first_attempt_success_rate,
        "recovery_detected": recovery_detected,
        "recovered_steps": recovered_steps,
        "aggregation_pass": aggregation_pass,
        "insight_validation_pass": insight_pass,
        "evidence_mapping_pass": evidence_pass,
        "recommendation_pass": recommendation_pass,
        "overall_result": (
            "PASS"
            if production_test_pass
            else "FAIL"
        ),
        "task_analysis": task_analysis,
        "plan": plan,
        "steps": step_results,
        "aggregation": aggregation,
        "insight": insight,
        "insight_validation": insight_validation,
        "evidence": evidence,
        "mapped_evidence": mapped_evidence,
        "recommendation": recommendation
    }

    final_results = safe_json_value(
        final_results
    )

    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
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
            default=str
        )

    print()

    print(
        f"Production evaluation results saved to:"
    )

    print(OUTPUT_PATH)

    print()


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()