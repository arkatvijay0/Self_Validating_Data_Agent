import os
import sys


# =========================================================
# PROJECT PATH SETUP
# =========================================================

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_ROOT = os.path.dirname(
    CURRENT_DIR
)

if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)


# =========================================================
# PROJECT IMPORTS
# =========================================================

from data_profiler import profile_dataset

from task_analyzer import analyze_task_mock

from eda_planner import generate_plan_mock

from multi_step_executor import (
    create_self_correcting_executor,
    execute_plan_steps
)

from result_aggregator import aggregate_results

from insight_generator import generate_insight

from insight_validator import validate_insight

from evidence_mapping import (
    create_correlation_evidence,
    create_evidence_summary,
    map_insight_to_evidence
)

from recommendation_generator import (
    generate_recommendation
)

from final_report import (
    create_final_report
)


# =========================================================
# CONFIGURATION
# =========================================================

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "sales.csv"
)

USER_QUESTION = (
    "Is there a relationship between "
    "units sold and total profit?"
)


# =========================================================
# HELPER
# =========================================================

def print_section(title):

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# =========================================================
# MAIN TEST
# =========================================================

def main():

    print_section(
        "TEST 3B - COMPLETE RELATIONSHIP PIPELINE"
    )

    print("Dataset:")
    print(DATA_PATH)

    print()
    print("User Question:")
    print(USER_QUESTION)


    # =====================================================
    # STEP 1 - DATA PROFILING
    # =====================================================

    print_section(
        "STEP 1 - DATA PROFILING"
    )

    data_profile = profile_dataset(
        DATA_PATH
    )

    dataset_columns = list(
        data_profile.get(
            "columns",
            {}
        ).keys()
    )

    print(
        "Data profiling completed."
    )

    print(
        "Number of columns:",
        len(dataset_columns)
    )


    # =====================================================
    # STEP 2 - TASK ANALYSIS
    # =====================================================

    print_section(
        "STEP 2 - TASK ANALYSIS"
    )

    task_analysis = analyze_task_mock(
        USER_QUESTION,
        data_profile
    )

    task_type = task_analysis.get(
        "task_type"
    )

    target_columns = task_analysis.get(
        "target_columns",
        []
    )

    print(
        "Task Analysis:"
    )

    print(
        task_analysis
    )

    print()

    print(
        "Analysis Type:",
        task_type
    )

    print(
        "Target Columns:",
        target_columns
    )


    # =====================================================
    # STEP 3 - EDA PLANNING
    # =====================================================

    print_section(
        "STEP 3 - EDA PLANNING"
    )

    plan = generate_plan_mock(
        USER_QUESTION,
        task_analysis,
        data_profile
    )

    print(
        "Generated EDA Plan:"
    )

    for index, step in enumerate(
        plan,
        start=1
    ):

        print(
            f"{index}. {step}"
        )


    # =====================================================
    # STEP 4 - SELF-CORRECTING EXECUTOR
    # =====================================================

    print_section(
        "STEP 4 - SELF-CORRECTING EDA EXECUTION"
    )

    step_executor = create_self_correcting_executor(
        filepath=DATA_PATH,
        data_profile=data_profile,
        dataset_columns=dataset_columns,
        target_columns=target_columns,
        test_fault=False
    )

    execution_results = execute_plan_steps(
        plan=plan,
        step_executor=step_executor
    )

    print(
        "Overall Execution Status:",
        execution_results.get(
            "overall_status"
        )
    )

    print(
        "Total Steps:",
        execution_results.get(
            "total_steps"
        )
    )

    print(
        "Successful Steps:",
        execution_results.get(
            "successful_steps"
        )
    )

    print(
        "Failed Steps:",
        execution_results.get(
            "failed_steps"
        )
    )


    # =====================================================
    # CHECK EDA EXECUTION
    # =====================================================

    eda_execution_pass = (
        execution_results.get(
            "overall_status"
        ) == "PASS"
        and
        execution_results.get(
            "successful_steps",
            0
        )
        ==
        execution_results.get(
            "total_steps",
            0
        )
        and
        execution_results.get(
            "failed_steps",
            0
        ) == 0
    )

    if not eda_execution_pass:

        print(
            "\nEDA execution failed."
        )

        print(
            "\nTEST 3B RESULT: FAIL"
        )

        return


    print(
        "\nEDA execution verified successfully."
    )


    # =====================================================
    # STEP 5 - RESULT AGGREGATION
    # =====================================================

    print_section(
        "STEP 5 - RESULT AGGREGATION"
    )

    steps = execution_results.get(
        "steps",
        []
    )

    aggregation = aggregate_results(
        user_question=USER_QUESTION,
        all_steps=steps,
        task_analysis=task_analysis
    )

    print(
        "Aggregation Result:"
    )

    print(
        aggregation
    )


    aggregation_status = aggregation.get(
        "overall_status",
        aggregation.get(
            "status",
            "FAIL"
        )
    )

    print()

    print(
        "Aggregation Status:",
        aggregation_status
    )


    # =====================================================
    # EXTRACT VALIDATED CORRELATION RESULT
    # =====================================================

    validated_result = aggregation.get(
        "primary_result"
    )


    # -----------------------------------------------------
    # Fallback: obtain Pearson result directly
    # if aggregator structure differs
    # -----------------------------------------------------

    if validated_result is None:

        for step in steps:

            step_text = str(
                step.get(
                    "step",
                    ""
                )
            ).lower()

            result = step.get(
                "final_result"
            )

            if (
                "pearson" in step_text
                and isinstance(
                    result,
                    (int, float)
                )
            ):

                validated_result = result

                break


    print()

    print(
        "Validated Correlation Result:"
    )

    print(
        validated_result
    )


    aggregation_pass = (
        validated_result is not None
        and
        aggregation_status == "PASS"
    )


    if aggregation_pass:

        print(
            "\nResult aggregation: PASS"
        )

    else:

        print(
            "\nResult aggregation: FAIL"
        )

        print(
            "\nTEST 3B RESULT: FAIL"
        )

        return


    # =====================================================
    # STEP 6 - INSIGHT GENERATION
    # =====================================================

    print_section(
        "STEP 6 - INSIGHT GENERATION"
    )

    insight = generate_insight(
        USER_QUESTION,
        validated_result,
        task_analysis
    )

    print(
        "Generated Insight:"
    )

    print(
        insight
    )


    insight_pass = (
        isinstance(
            insight,
            str
        )
        and
        len(
            insight.strip()
        ) > 0
        and
        "No valid" not in insight
    )


    print()

    print(
        "Insight Generation Status:",
        "PASS"
        if insight_pass
        else "FAIL"
    )


    if not insight_pass:

        print(
            "\nTEST 3B RESULT: FAIL"
        )

        return


    # =====================================================
    # STEP 7 - INSIGHT VALIDATION
    # =====================================================

    print_section(
        "STEP 7 - INSIGHT VALIDATION"
    )

    insight_validation = validate_insight(
        insight,
        validated_result,
        task_analysis
    )

    print(
        "Insight Validation:"
    )

    print(
        insight_validation
    )


    insight_validation_pass = (
        insight_validation.get(
            "overall"
        ) == "PASS"
    )


    print()

    print(
        "Insight Validation Status:",
        "PASS"
        if insight_validation_pass
        else "FAIL"
    )


    if not insight_validation_pass:

        print(
            "\nTEST 3B RESULT: FAIL"
        )

        return


    # =====================================================
    # STEP 8 - EVIDENCE MAPPING
    # =====================================================

    print_section(
        "STEP 8 - EVIDENCE MAPPING"
    )

    x_column = "Units Sold"

    y_column = "Total Profit"

    if len(target_columns) >= 2:

        x_column = target_columns[0]

        y_column = target_columns[1]


    evidence = create_correlation_evidence(
        result=validated_result,
        x_column=x_column,
        y_column=y_column,
        method="Pearson"
    )


    evidence_summary = create_evidence_summary(
        evidence
    )


    mapped_evidence = map_insight_to_evidence(
        insight,
        evidence
    )


    print(
        "Evidence:"
    )

    print(
        evidence
    )

    print()

    print(
        "Evidence Summary:"
    )

    print(
        evidence_summary
    )

    print()

    print(
        "Mapped Evidence:"
    )

    print(
        mapped_evidence
    )


    evidence_pass = (
        isinstance(
            evidence,
            dict
        )
        and
        evidence.get(
            "status"
        ) == "PASS"
        and
        isinstance(
            mapped_evidence,
            dict
        )
        and
        mapped_evidence.get(
            "status"
        ) == "PASS"
    )


    print()

    print(
        "Evidence Mapping Status:",
        "PASS"
        if evidence_pass
        else "FAIL"
    )


    if not evidence_pass:

        print(
            "\nTEST 3B RESULT: FAIL"
        )

        return


    # =====================================================
    # STEP 9 - RECOMMENDATION
    # =====================================================

    print_section(
        "STEP 9 - RECOMMENDATION GENERATION"
    )

    recommendation = generate_recommendation(
        task_type=task_type,
        evidence=evidence
    )


    print(
        "Recommendation:"
    )

    print(
        recommendation
    )


    recommendation_pass = (
        isinstance(
            recommendation,
            dict
        )
        and
        recommendation.get(
            "status"
        ) == "PASS"
        and
        bool(
            recommendation.get(
                "recommendation"
            )
        )
    )


    print()

    print(
        "Recommendation Status:",
        "PASS"
        if recommendation_pass
        else "FAIL"
    )


    if not recommendation_pass:

        print(
            "\nTEST 3B RESULT: FAIL"
        )

        return


    # =====================================================
    # STEP 10 - FINAL REPORT
    # =====================================================

    print_section(
        "STEP 10 - FINAL REPORT GENERATION"
    )

    final_report = create_final_report(
        user_question=USER_QUESTION,
        analysis_type=task_type,
        validated_result=validated_result,
        insight=insight,
        insight_validation=insight_validation,
        evidence=evidence,
        recommendation=recommendation
    )


    print(
        "Final Report:"
    )

    print(
        final_report
    )


    report_pass = (
        isinstance(
            final_report,
            str
        )
        and
        len(
            final_report.strip()
        ) > 0
        and
        "SELF-VALIDATING EDA AGENT"
        in final_report
    )


    print()

    print(
        "Final Report Status:",
        "PASS"
        if report_pass
        else "FAIL"
    )


    if not report_pass:

        print(
            "\nTEST 3B RESULT: FAIL"
        )

        return


    # =====================================================
    # FINAL PIPELINE VERIFICATION
    # =====================================================

    print_section(
        "TEST 3B - COMPLETE PIPELINE RESULT"
    )


    pipeline_checks = {

        "EDA Execution":
            eda_execution_pass,

        "Result Aggregation":
            aggregation_pass,

        "Insight Generation":
            insight_pass,

        "Insight Validation":
            insight_validation_pass,

        "Evidence Mapping":
            evidence_pass,

        "Recommendation":
            recommendation_pass,

        "Final Report":
            report_pass
    }


    for name, status in pipeline_checks.items():

        print(
            f"{name:<25}: "
            f"{'PASS' if status else 'FAIL'}"
        )


    overall_pass = all(
        pipeline_checks.values()
    )


    print()

    print("=" * 70)


    if overall_pass:

        print(
            "TEST 3B RESULT: PASS"
        )

        print()

        print(
            "Complete relationship-analysis "
            "pipeline executed successfully."
        )

        print()

        print(
            "Validated EDA Result:"
        )

        print(
            validated_result
        )

        print()

        print(
            "Insight:"
        )

        print(
            insight
        )

        print()

        print(
            "Evidence Status:",
            evidence.get(
                "status"
            )
        )

        print()

        print(
            "Recommendation Status:",
            recommendation.get(
                "status"
            )
        )

        print()

        print(
            "OVERALL PIPELINE RESULT: PASS"
        )

    else:

        print(
            "TEST 3B RESULT: FAIL"
        )

        print()

        print(
            "One or more downstream "
            "pipeline stages failed."
        )

        print()

        print(
            "OVERALL PIPELINE RESULT: FAIL"
        )


    print("=" * 70)


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()