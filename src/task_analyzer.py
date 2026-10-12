import json


TASK_TYPES = [
    "descriptive_summary",
    "trend_analysis",
    "comparison",
    "anomaly_detection",
    "relationship_analysis",
    "decision_support"
]


def find_column(query, columns):
    """
    Find a dataset column mentioned or implied by the user's question.
    """

    query_lower = query.lower()

    # Exact / partial column name matching
    for column in columns:

        column_lower = column.lower()

        if column_lower in query_lower:
            return column

    # Common aliases
    aliases = {
        "revenue": "Total Revenue",
        "sales": "Total Revenue",
        "profit": "Total Profit",
        "cost": "Total Cost",
        "units": "Units Sold",
        "quantity": "Units Sold",
        "region": "Region",
        "country": "Country",
        "product": "Item Type",
        "item": "Item Type",
        "channel": "Sales Channel",
        "sales channel": "Sales Channel",
        "order date": "Order Date",
        "date": "Order Date"
    }

    for keyword, column in aliases.items():

        if keyword in query_lower and column in columns:
            return column

    return None


def analyze_task_mock(user_query, data_profile):
    """
    Temporary rule-based Task Analyzer.

    Uses the dataset profile to identify relevant columns.
    """

    query = user_query.lower()

    # Specific filtered aggregation: average call duration for a named month.
    if ("duration" in query and any(word in query for word in ("average", "mean", "avg"))
            and any(month in query for month in ("january", "february", "march", "april", "may", "june",
                                                  "july", "august", "september", "october", "november", "december"))):
        month_alias = {
            "january": "jan", "february": "feb", "march": "mar", "april": "apr",
            "may": "may", "june": "jun", "july": "jul", "august": "aug",
            "september": "sep", "october": "oct", "november": "nov", "december": "dec",
        }
        selected_month = next((name for name in month_alias if name in query), None)
        return {
            "task_type": "mean_analysis",
            "target_columns": [c for c in ("duration", "month") if c in data_profile.get("columns", {})],
            "filter_month": month_alias.get(selected_month),
            "notes": "Filtered mean request: filter rows to the named month before calculating the average.",
        }

    columns = list(
        data_profile["columns"].keys()
    )

    # --------------------------------------------------
    # 1. Determine task type
    # --------------------------------------------------

    if any(word in query for word in [
        "trend",
        "over time",
        "change over time",
        "monthly",
        "yearly",
        "quarter",
        "increase",
        "decrease",
        "changed"
    ]):

        task_type = "trend_analysis"

    elif any(word in query for word in [
        "relationship",
        "related",
        "correlation",
        "correlated",
        "association"
    ]):

        task_type = "relationship_analysis"

    elif any(word in query for word in [
        "anomaly",
        "anomalies",
        "outlier",
        "outliers",
        "unusual",
        "abnormal"
    ]):

        task_type = "anomaly_detection"

    elif any(word in query for word in [
        "should we",
        "recommend",
        "recommendation",
        "decision",
        "focus on"
    ]):

        task_type = "decision_support"

    elif any(word in query for word in [
        "which",
        "highest",
        "lowest",
        "compare",
        "comparison",
        "better",
        "best",
        "worst"
    ]):

        task_type = "comparison"

    else:

        task_type = "descriptive_summary"

    # --------------------------------------------------
    # 2. Find relevant columns
    # --------------------------------------------------

    target_columns = []

    # Find all explicitly mentioned columns
    for column in columns:

        if column.lower() in query:
            target_columns.append(column)

    # Add aliases
    alias_matches = {
        "revenue": "Total Revenue",
        "sales": "Total Revenue",
        "profit": "Total Profit",
        "cost": "Total Cost",
        "units": "Units Sold",
        "quantity": "Units Sold",
        "region": "Region",
        "country": "Country",
        "product": "Item Type",
        "item": "Item Type",
        "channel": "Sales Channel",
        "order date": "Order Date",
        "conversion rate": "conversion",
        "converted customers": "conversion",
        "conversion": "conversion",
        "outcome column": "y"
    }

    for keyword, column in alias_matches.items():

        if keyword in query and column in columns:

            if column not in target_columns:
                target_columns.append(column)

    # Trend analysis normally needs a date column
    if task_type == "trend_analysis":

        if "Order Date" in columns:
            if "Order Date" not in target_columns:
                target_columns.insert(
                    0,
                    "Order Date"
                )

    # Relationship analysis needs numeric columns
    if task_type == "relationship_analysis":

        numeric_columns = []

        for column, info in data_profile["columns"].items():

            if info.get("role") == "continuous":
                numeric_columns.append(column)

        for column in numeric_columns:

            if column.lower() in query:

                if column not in target_columns:
                    target_columns.append(column)

    # --------------------------------------------------
    # 3. Create explanation
    # --------------------------------------------------

    notes = (
        "Task classified using temporary rule-based "
        "analyzer and dataset profile."
    )

    return {
        "task_type": task_type,
        "target_columns": target_columns,
        "notes": notes
    }


if __name__ == "__main__":

    from data_profiler import profile_dataset

    # -----------------------------------------------
    # Load your actual dataset profile
    # -----------------------------------------------

    filepath = "data/rawz/sales.csv"

    profile = profile_dataset(filepath)

    # -----------------------------------------------
    # Test questions
    # -----------------------------------------------

    test_questions = [

        "Which region generated the highest total profit?",

        "How did total revenue change over time?",

        "Is there a relationship between units sold and total profit?",

        "Are there any unusual total profit values?",

        "Which sales channel is better?",

        "Give me a summary of the sales data."
    ]

    print("=" * 60)
    print("TASK ANALYZER + DATA PROFILER TEST")
    print("=" * 60)

    for question in test_questions:

        result = analyze_task_mock(
            question,
            profile
        )

        print("\nQuestion:")
        print(question)

        print("\nResult:")

        print(
            json.dumps(
                result,
                indent=4
            )
        )

    print("\n" + "=" * 60)
    print("TASK ANALYZER TEST COMPLETED")
    print("=" * 60)