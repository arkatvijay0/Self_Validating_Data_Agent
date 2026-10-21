"""
Result Aggregator
-----------------
Aggregates validated multi-step EDA results into:

1. Primary result
2. Supporting results
3. Validated step information
4. Execution reliability information

This module does NOT generate insights.
It only organizes validated analytical results for downstream
Insight Generation, Evidence Mapping, Recommendations, and Reporting.
"""

from typing import Any, Dict, List, Optional


# ============================================================
# Helper Functions
# ============================================================

def get_step_status(step: Dict[str, Any]) -> str:
    """
    Safely extract the final status of an executed step.
    """

    return str(
        step.get("status", "")
    ).upper()


def is_successful_step(step: Dict[str, Any]) -> bool:
    """
    Check whether a step passed self-validation.
    """

    return get_step_status(step) == "PASS"


def get_step_result(step: Dict[str, Any]) -> Any:
    """
    Extract the final validated result from a step.
    """

    if "final_result" in step:
        return step["final_result"]

    final_execution = step.get("final_execution", {})

    if isinstance(final_execution, dict):
        return final_execution.get("result")

    return None


def get_step_validation(step: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract final validation information.
    """

    validation = step.get("final_validation")

    if isinstance(validation, dict):
        return validation

    return {}


def get_step_attempts(step: Dict[str, Any]) -> int:
    """
    Extract number of execution attempts.
    """

    attempts = step.get("total_attempts")

    if isinstance(attempts, int):
        return attempts

    attempt_history = step.get("attempts")

    if isinstance(attempt_history, list):
        return len(attempt_history)

    return 0


# ============================================================
# Analysis Type Detection
# ============================================================

def detect_analysis_type(
    user_question: str,
    task_analysis: Optional[Dict[str, Any]] = None
) -> str:
    """
    Determine the broad analysis type.

    Priority:
    1. Task analyzer result
    2. User question keywords
    """

    if isinstance(task_analysis, dict):

        task_type = task_analysis.get("task_type")

        if task_type:
            task_type = str(task_type).lower()

            if "relationship" in task_type:
                return "relationship_analysis"

            if "comparison" in task_type:
                return "comparison"

            if "time" in task_type or "trend" in task_type:
                return "time_series"

            if "outlier" in task_type or "anomal" in task_type:
                return "outlier_analysis"

            if "descriptive" in task_type:
                return "descriptive_analysis"

            if "mean" in task_type:
                return "mean_analysis"

    question = str(user_question).lower()

    if any(
        keyword in question
        for keyword in [
            "relationship",
            "correlation",
            "associated",
            "association"
        ]
    ):
        return "relationship_analysis"

    if any(
        keyword in question
        for keyword in [
            "highest",
            "lowest",
            "rank",
            "ranking",
            "compare",
            "comparison"
        ]
    ):
        return "comparison"

    if any(
        keyword in question
        for keyword in [
            "over time",
            "trend",
            "monthly",
            "yearly",
            "daily"
        ]
    ):
        return "time_series"

    if any(
        keyword in question
        for keyword in [
            "outlier",
            "unusual",
            "unusually",
            "anomal"
        ]
    ):
        return "outlier_analysis"

    if any(
        keyword in question
        for keyword in [
            "average",
            "mean"
        ]
    ):
        return "mean_analysis"

    return "descriptive_analysis"


# ============================================================
# Primary Result Selection
# ============================================================

def select_primary_result(
    successful_steps: List[Dict[str, Any]],
    analysis_type: str
) -> Optional[Dict[str, Any]]:
    """
    Select the most useful validated result for the user's question.

    The selection is rule-based and deterministic.
    """

    if not successful_steps:
        return None

    # --------------------------------------------------------
    # Comparison
    # --------------------------------------------------------

    if analysis_type == "comparison":


        # Prefer a valid grouped-mean result over a generic overall mean.
        for step in reversed(successful_steps):
            result = get_step_result(step)

            if (
                isinstance(result, dict)
                and result.get("analysis_type") == "grouped_mean_analysis"
                and result.get("status") == "PASS"
                and isinstance(result.get("group_results"), dict)
                and len(result["group_results"]) >= 2
                and result.get("mean_difference_yes_minus_no") is not None
            ):
                return step

        # Prefer a step explicitly identifying the highest category.
        for step in reversed(successful_steps):

            step_text = str(
                step.get("step", "")
            ).lower()

            result = get_step_result(step)

            if (
                "highest" in step_text
                and isinstance(result, dict)
            ):
                return step

        # Next preference: ranking result.
        for step in reversed(successful_steps):

            step_text = str(
                step.get("step", "")
            ).lower()

            result = get_step_result(step)

            if (
                "rank" in step_text
                and result is not None
            ):
                return step

        # Final fallback: last successful step.
        return successful_steps[-1]

    # --------------------------------------------------------
    # Relationship Analysis
    # --------------------------------------------------------

    if analysis_type == "relationship_analysis":

        # Prefer Pearson correlation.
        for step in successful_steps:

            step_text = str(
                step.get("step", "")
            ).lower()

            result = get_step_result(step)

            if (
                "pearson" in step_text
                and result is not None
            ):
                return step

        # Then Spearman.
        for step in successful_steps:

            step_text = str(
                step.get("step", "")
            ).lower()

            result = get_step_result(step)

            if (
                "spearman" in step_text
                and result is not None
            ):
                return step

        # Generic correlation fallback.
        for step in successful_steps:

            step_text = str(
                step.get("step", "")
            ).lower()

            if (
                "correlation" in step_text
                or "relationship" in step_text
            ):
                return step

        return successful_steps[0]

    # --------------------------------------------------------
    # Time Series
    # --------------------------------------------------------

    if analysis_type == "time_series":

        for step in reversed(successful_steps):

            step_text = str(
                step.get("step", "")
            ).lower()

            if any(
                keyword in step_text
                for keyword in [
                    "trend",
                    "over time",
                    "monthly",
                    "change"
                ]
            ):
                return step

        return successful_steps[-1]

    # --------------------------------------------------------
    # Outlier Analysis
    # --------------------------------------------------------

    if analysis_type == "outlier_analysis":

        # Prefer Isolation Forest when a successful step identifies it.
        for step in reversed(successful_steps):
            step_text = str(step.get("step", "")).lower()
            result = get_step_result(step)

            result_method = (
                str(result.get("method", "")).lower()
                if isinstance(result, dict)
                else ""
            )

            if (
                "isolation forest" in step_text
                or "isolationforest" in step_text
                or "isolation forest" in result_method
                or "isolationforest" in result_method
            ):
                return step

        # Preserve existing behavior for other outlier requests.
        for step in reversed(successful_steps):
            step_text = str(step.get("step", "")).lower()

            if any(
                keyword in step_text
                for keyword in ["outlier", "unusual", "anomal"]
            ):
                return step

        return successful_steps[-1]

    # --------------------------------------------------------
    # Mean Analysis
    # --------------------------------------------------------

    if analysis_type == "mean_analysis":

        for step in successful_steps:

            step_text = str(
                step.get("step", "")
            ).lower()

            if (
                "mean" in step_text
                or "average" in step_text
            ):
                return step

        return successful_steps[0]

    # --------------------------------------------------------
    # Descriptive Analysis
    # --------------------------------------------------------

    return successful_steps[0]


# ============================================================
# Supporting Result Selection
# ============================================================

def select_supporting_results(
    successful_steps: List[Dict[str, Any]],
    primary_step: Optional[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Select all successful steps other than the primary result.
    """

    if primary_step is None:
        return successful_steps.copy()

    primary_id = id(primary_step)

    supporting = []

    for step in successful_steps:

        if id(step) != primary_id:
            supporting.append(step)

    return supporting


# ============================================================
# Step Summary
# ============================================================

def create_step_summary(
    all_steps: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Create a compact summary for every executed step.
    """

    summary = []

    for index, step in enumerate(all_steps, start=1):

        status = get_step_status(step)

        validation = get_step_validation(step)

        final_validation_status = str(
            validation.get("overall", "")
        ).upper()

        summary.append(
            {
                "step_number": step.get(
                    "step_number",
                    index
                ),
                "step": step.get(
                    "step",
                    ""
                ),
                "status": status,
                "validation": final_validation_status,
                "attempts": get_step_attempts(step),
                "has_result": (
                    get_step_result(step) is not None
                )
            }
        )

    return summary


# ============================================================
# Reliability Metrics
# ============================================================

def calculate_reliability_metrics(
    all_steps: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Calculate execution reliability metrics.
    """

    total_steps = len(all_steps)

    successful_steps = sum(
        1
        for step in all_steps
        if is_successful_step(step)
    )

    failed_steps = total_steps - successful_steps

    total_attempts = sum(
        get_step_attempts(step)
        for step in all_steps
    )

    correction_steps = sum(
        1
        for step in all_steps
        if get_step_attempts(step) > 1
    )

    success_rate = (
        successful_steps / total_steps * 100
        if total_steps > 0
        else 0.0
    )

    correction_rate = (
        correction_steps / total_steps * 100
        if total_steps > 0
        else 0.0
    )

    average_attempts = (
        total_attempts / total_steps
        if total_steps > 0
        else 0.0
    )

    return {
        "total_steps": total_steps,
        "successful_steps": successful_steps,
        "failed_steps": failed_steps,
        "total_attempts": total_attempts,
        "correction_steps": correction_steps,
        "success_rate": round(
            success_rate,
            2
        ),
        "correction_rate": round(
            correction_rate,
            2
        ),
        "average_attempts": round(
            average_attempts,
            2
        )
    }


# ============================================================
# Main Aggregator
# ============================================================

def aggregate_results(
    user_question: str,
    all_steps: List[Dict[str, Any]],
    task_analysis: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Aggregate all validated EDA step results.

    Returns:
        {
            analysis_type,
            primary_result,
            supporting_results,
            validated_steps,
            step_summary,
            reliability,
            overall_status
        }
    """

    analysis_type = detect_analysis_type(
        user_question=user_question,
        task_analysis=task_analysis
    )

    successful_steps = [
        step
        for step in all_steps
        if is_successful_step(step)
    ]

    primary_step = select_primary_result(
        successful_steps=successful_steps,
        analysis_type=analysis_type
    )

    supporting_steps = select_supporting_results(
        successful_steps=successful_steps,
        primary_step=primary_step
    )

    reliability = calculate_reliability_metrics(
        all_steps
    )

    step_summary = create_step_summary(
        all_steps
    )

    overall_status = (
        "PASS"
        if (
            all_steps
            and
            len(successful_steps) == len(all_steps)
            and
            primary_step is not None
        )
        else "FAIL"
    )

    return {
        "analysis_type": analysis_type,

        "primary_result": (
            get_step_result(primary_step)
            if primary_step
            else None
        ),

        "primary_step": (
            {
                "step_number": primary_step.get(
                    "step_number"
                ),
                "step": primary_step.get(
                    "step"
                ),
                "attempts": get_step_attempts(
                    primary_step
                )
            }
            if primary_step
            else None
        ),

        "supporting_results": [
            {
                "step_number": step.get(
                    "step_number"
                ),
                "step": step.get(
                    "step"
                ),
                "result": get_step_result(
                    step
                ),
                "attempts": get_step_attempts(
                    step
                )
            }
            for step in supporting_steps
        ],

        "validated_steps": len(
            successful_steps
        ),

        "step_summary": step_summary,

        "reliability": reliability,

        "overall_status": overall_status
    }


# ============================================================
# Display Helper
# ============================================================

def print_aggregation_summary(
    aggregation: Dict[str, Any]
) -> None:
    """
    Print a readable aggregation summary.
    """

    print("\n")
    print("#" * 70)
    print("MULTI-STEP RESULT AGGREGATION SUMMARY")
    print("#" * 70)

    print(
        "\nAnalysis Type:"
    )
    print(
        aggregation.get(
            "analysis_type"
        )
    )

    print(
        "\nOverall Status:"
    )
    print(
        aggregation.get(
            "overall_status"
        )
    )

    print(
        "\nValidated Steps:"
    )
    print(
        aggregation.get(
            "validated_steps"
        )
    )

    print(
        "\nPrimary Step:"
    )
    print(
        aggregation.get(
            "primary_step"
        )
    )

    print(
        "\nPrimary Result:"
    )
    print(
        aggregation.get(
            "primary_result"
        )
    )

    print(
        "\nSupporting Results:"
    )

    supporting = aggregation.get(
        "supporting_results",
        []
    )

    for item in supporting:

        print(
            f"Step {item.get('step_number')}: "
            f"{item.get('step')}"
        )

    print(
        "\nReliability:"
    )

    reliability = aggregation.get(
        "reliability",
        {}
    )

    for key, value in reliability.items():

        print(
            f"{key}: {value}"
        )

    print(
        "\nStep Summary:"
    )

    for item in aggregation.get(
        "step_summary",
        []
    ):

        print(
            f"Step {item.get('step_number')}: "
            f"{item.get('status')} | "
            f"Validation: {item.get('validation')} | "
            f"Attempts: {item.get('attempts')}"
        )


# ============================================================
# Standalone Tests
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("#" * 70)
    print("RESULT AGGREGATOR TESTS")
    print("#" * 70)

    # --------------------------------------------------------
    # TEST DATA
    # --------------------------------------------------------

    comparison_steps = [
        {
            "step_number": 1,
            "step": (
                "Group the data by Region and "
                "calculate the total of Total Profit."
            ),
            "status": "PASS",
            "total_attempts": 1,
            "final_result": {
                "Sub-Saharan Africa": 12183211.40,
                "Europe": 11082938.63
            },
            "final_validation": {
                "overall": "PASS"
            }
        },

        {
            "step_number": 2,
            "step": (
                "Rank the Region categories "
                "according to total Total Profit."
            ),
            "status": "PASS",
            "total_attempts": 1,
            "final_result": [
                {
                    "Rank": 1,
                    "Region": "Sub-Saharan Africa",
                    "Total Profit": 12183211.40
                },
                {
                    "Rank": 2,
                    "Region": "Europe",
                    "Total Profit": 11082938.63
                }
            ],
            "final_validation": {
                "overall": "PASS"
            }
        },

        {
            "step_number": 3,
            "step": (
                "Compare the distribution of "
                "Total Profit across Region categories."
            ),
            "status": "PASS",
            "total_attempts": 1,
            "final_result": {
                "Sub-Saharan Africa": {
                    "mean": 1218321.14
                },
                "Europe": {
                    "mean": 1108293.86
                }
            },
            "final_validation": {
                "overall": "PASS"
            }
        },

        {
            "step_number": 4,
            "step": (
                "Check for unusually high or low "
                "Total Profit values within each Region."
            ),
            "status": "PASS",
            "total_attempts": 1,
            "final_result": [],
            "final_validation": {
                "overall": "PASS"
            }
        },

        {
            "step_number": 5,
            "step": (
                "Identify the Region category with "
                "the highest total Total Profit."
            ),
            "status": "PASS",
            "total_attempts": 1,
            "final_result": {
                "Region": "Sub-Saharan Africa",
                "Total Profit": 12183211.40
            },
            "final_validation": {
                "overall": "PASS"
            }
        }
    ]

    task_analysis = {
        "task_type": "comparison"
    }

    question = (
        "Which region generated "
        "the highest total profit?"
    )

    aggregation = aggregate_results(
        user_question=question,
        all_steps=comparison_steps,
        task_analysis=task_analysis
    )

    print_aggregation_summary(
        aggregation
    )

    # --------------------------------------------------------
    # TEST 1
    # --------------------------------------------------------

    test_1_pass = (
        aggregation["overall_status"] == "PASS"
        and
        aggregation["analysis_type"] == "comparison"
        and
        aggregation["validated_steps"] == 5
        and
        aggregation["primary_step"]["step_number"] == 5
        and
        aggregation["primary_result"]["Region"]
        == "Sub-Saharan Africa"
    )

    print(
        "\nTEST 1 RESULT:",
        "PASS" if test_1_pass else "FAIL"
    )

    # --------------------------------------------------------
    # TEST 2
    # --------------------------------------------------------

    relationship_steps = [
        {
            "step_number": 1,
            "step": (
                "Calculate Pearson correlation "
                "between Units Sold and Total Profit."
            ),
            "status": "PASS",
            "total_attempts": 1,
            "final_result": {
                "correlation": 0.5645504620845976
            },
            "final_validation": {
                "overall": "PASS"
            }
        },

        {
            "step_number": 2,
            "step": (
                "Calculate Spearman correlation "
                "between Units Sold and Total Profit."
            ),
            "status": "PASS",
            "total_attempts": 1,
            "final_result": {
                "correlation": 0.52
            },
            "final_validation": {
                "overall": "PASS"
            }
        },

        {
            "step_number": 3,
            "step": (
                "Compare the distribution of "
                "Units Sold and Total Profit."
            ),
            "status": "PASS",
            "total_attempts": 1,
            "final_result": {
                "Units Sold": {
                    "mean": 512.2
                }
            },
            "final_validation": {
                "overall": "PASS"
            }
        }
    ]

    relationship_task = {
        "task_type": "relationship_analysis"
    }

    relationship_question = (
        "Is there a relationship between "
        "units sold and total profit?"
    )

    relationship_aggregation = aggregate_results(
        user_question=relationship_question,
        all_steps=relationship_steps,
        task_analysis=relationship_task
    )

    relationship_test_pass = (
        relationship_aggregation["overall_status"]
        == "PASS"
        and
        relationship_aggregation["analysis_type"]
        == "relationship_analysis"
        and
        relationship_aggregation["primary_step"]
        ["step_number"] == 1
        and
        relationship_aggregation
        ["primary_result"]
        ["correlation"]
        == 0.5645504620845976
    )

    print(
        "\nTEST 2 RESULT:",
        "PASS"
        if relationship_test_pass
        else "FAIL"
    )

    # --------------------------------------------------------
    # TEST 3
    # --------------------------------------------------------

    failed_steps = [
        comparison_steps[0],
        {
            "step_number": 2,
            "step": "Faulty step",
            "status": "UNRESOLVED",
            "total_attempts": 3,
            "final_result": None,
            "final_validation": {
                "overall": "FAIL"
            }
        }
    ]

    failed_aggregation = aggregate_results(
        user_question=question,
        all_steps=failed_steps,
        task_analysis=task_analysis
    )

    test_3_pass = (
        failed_aggregation["overall_status"]
        == "FAIL"
        and
        failed_aggregation["reliability"]
        ["failed_steps"]
        == 1
        and
        failed_aggregation["reliability"]
        ["successful_steps"]
        == 1
    )

    print(
        "\nTEST 3 RESULT:",
        "PASS"
        if test_3_pass
        else "FAIL"
    )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    tests = [
        test_1_pass,
        relationship_test_pass,
        test_3_pass
    ]

    passed = sum(
        1
        for test in tests
        if test
    )

    print("\n")
    print("#" * 70)
    print("FINAL TEST SUMMARY")
    print("#" * 70)

    print(
        f"\nTests Passed: {passed}/{len(tests)}"
    )

    print(
        "OVERALL RESULT:",
        "PASS"
        if passed == len(tests)
        else "FAIL"
    )