import json


def generate_plan_mock(task_type, data_profile, user_query):
    """
    Generate an EDA plan using the actual user question
    and dataset profile.
    """

    columns = data_profile["columns"]

    numeric_columns = [
        col
        for col, info in columns.items()
        if info.get("role") == "continuous"
    ]

    categorical_columns = [
        col
        for col, info in columns.items()
        if info.get("role") == "categorical"
    ]

    date_columns = [
        col
        for col, info in columns.items()
        if info.get("role") == "time_index"
    ]

    query = user_query.lower()

    if ("duration" in query and any(word in query for word in ("average", "mean", "avg"))
            and any(month in query for month in ("january", "february", "march", "april", "may", "june",
                                                  "july", "august", "september", "october", "november", "december"))):
        month_names = {
            "january": "jan", "february": "feb", "march": "mar", "april": "apr",
            "may": "may", "june": "jun", "july": "jul", "august": "aug",
            "september": "sep", "october": "oct", "november": "nov", "december": "dec",
        }
        selected_month = next((name for name in month_names if name in query), None)
        month_code = month_names.get(selected_month, "")
        return [
            f"Filter rows where the 'month' column equals '{month_code}', then calculate the mean of numeric 'duration' values and report the filtered row count and valid duration count."
        ]

    # ==================================================
    # Find relevant columns from the user's question
    # ==================================================

    target_columns = []

    # Exact column-name matching
    for column in columns:

        if column.lower() in query:

            target_columns.append(column)

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
        "order date": "Order Date",
        "date": "Order Date"
    }

    for keyword, column in aliases.items():

        if keyword in query and column in columns:

            if column not in target_columns:
                target_columns.append(column)

    # ==================================================
    # Remove identifiers from analytical columns
    # ==================================================

    numeric_columns = [
        col
        for col in numeric_columns
        if columns[col].get("role") != "identifier"
    ]

    # ==================================================
    # DESCRIPTIVE SUMMARY
    # ==================================================

    # Handle specific conversion questions before generic descriptive EDA.
    conversion_question = (
        ("conversion" in query and any(token in query for token in
            ["rate", "converted", "conversion = 1", "conversion=1", "percentage"]))
        or ("row-by-row" in query and "y" in query)
        or ("matches the y column" in query)
    )
    if conversion_question:
        return [
            "Calculate conversion counts and conversion/non-conversion percentages from the conversion column; map y values yes/no to 1/0 and verify row-by-row equality, including unmatched values."
        ]

    if task_type == "descriptive_summary":

        steps = [
            "Calculate descriptive statistics such as mean, median, standard deviation, minimum, and maximum for the numeric variables.",
            "Calculate the number and percentage of missing values for every column.",
            "Identify the most frequent categories in categorical variables.",
            "Check the distribution and skewness of important numeric variables.",
            "Summarize the overall structure and characteristics of the dataset."
        ]

    # ==================================================
    # COMPARISON
    # ==================================================

    elif task_type == "comparison":

        group_column = None
        value_column = None

        # Find categorical column from target columns
        for column in target_columns:

            if column in categorical_columns:
                group_column = column
                break

        # Find numeric column from target columns
        for column in target_columns:

            if column in numeric_columns:
                value_column = column
                break

        # If query contains region
        if "region" in query and "Region" in columns:
            group_column = "Region"

        # If query contains profit
        if "profit" in query and "Total Profit" in columns:
            value_column = "Total Profit"

        # If query contains revenue
        if "revenue" in query and "Total Revenue" in columns:
            value_column = "Total Revenue"

        # If query contains sales channel
        if "channel" in query and "Sales Channel" in columns:
            group_column = "Sales Channel"

        if group_column and value_column:

            steps = [
                f"Group the data by '{group_column}' and calculate the total and mean of '{value_column}'.",
                f"Rank the '{group_column}' categories according to total '{value_column}'.",
                f"Compare the distribution of '{value_column}' across '{group_column}' categories.",
                f"Check for unusually high or low '{value_column}' values within each '{group_column}' category.",
                f"Identify the '{group_column}' category with the highest total '{value_column}'."
            ]

        else:

            steps = [
                "Identify the relevant categorical and numeric variables for comparison.",
                "Calculate summary statistics for the selected variables.",
                "Compare the groups using appropriate aggregation methods.",
                "Identify the highest and lowest performing groups.",
                "Summarize the main differences between the groups."
            ]

    # ==================================================
    # TREND ANALYSIS
    # ==================================================

    elif task_type == "trend_analysis":

        date_column = None
        value_column = None

        if "Order Date" in columns:
            date_column = "Order Date"

        for column in target_columns:

            if column in numeric_columns:
                value_column = column
                break

        if "revenue" in query and "Total Revenue" in columns:
            value_column = "Total Revenue"

        if "profit" in query and "Total Profit" in columns:
            value_column = "Total Profit"

        if date_column and value_column:

            steps = [
                f"Convert '{date_column}' to datetime format and sort the data chronologically.",
                f"Aggregate '{value_column}' by month.",
                f"Calculate the change in '{value_column}' between consecutive months.",
                f"Identify the months with the largest increases and decreases in '{value_column}'.",
                f"Evaluate the overall trend in '{value_column}' over time."
            ]

        else:

            steps = [
                "Identify the available date and numeric variables.",
                "Sort the observations chronologically.",
                "Aggregate the relevant numeric variable over time.",
                "Calculate changes between consecutive periods.",
                "Identify the overall trend and unusual changes."
            ]

    # ==================================================
    # RELATIONSHIP ANALYSIS
    # ==================================================

    elif task_type == "relationship_analysis":

        relationship_columns = [
            col
            for col in target_columns
            if col in numeric_columns
        ]

        # Explicitly handle units + profit
        if "units" in query and "Units Sold" in columns:
            if "Units Sold" not in relationship_columns:
                relationship_columns.append("Units Sold")

        if "profit" in query and "Total Profit" in columns:
            if "Total Profit" not in relationship_columns:
                relationship_columns.append("Total Profit")

        if len(relationship_columns) >= 2:

            col1 = relationship_columns[0]
            col2 = relationship_columns[1]

            steps = [
                f"Calculate the Pearson correlation between '{col1}' and '{col2}'.",
                f"Calculate the Spearman correlation between '{col1}' and '{col2}'.",
                f"Check the distributions of '{col1}' and '{col2}' before selecting the appropriate correlation method.",
                f"Determine the strength and direction of the relationship between '{col1}' and '{col2}'.",
                f"Check for potential outliers that may influence the correlation."
            ]

        else:

            steps = [
                "Identify two suitable numeric variables for relationship analysis.",
                "Check the distributions of the selected variables.",
                "Calculate Pearson and Spearman correlations.",
                "Compare the strength and direction of the relationships.",
                "Check whether outliers affect the observed relationship."
            ]

    # ==================================================
    # ANOMALY DETECTION
    # ==================================================

    elif task_type == "anomaly_detection":

        value_column = None

        if "profit" in query and "Total Profit" in columns:
            value_column = "Total Profit"

        elif "revenue" in query and "Total Revenue" in columns:
            value_column = "Total Revenue"

        elif target_columns:

            for column in target_columns:

                if column in numeric_columns:
                    value_column = column
                    break

        elif numeric_columns:

            value_column = numeric_columns[0]

        if value_column:

            steps = [
                f"Calculate the mean, median, and standard deviation of '{value_column}'.",
                f"Identify potential outliers in '{value_column}' using the IQR method.",
                f"Calculate z-scores for '{value_column}' and identify extreme observations.",
                f"Compare anomalies detected using the IQR and z-score methods.",
                f"Summarize the observations identified as potential anomalies in '{value_column}'."
            ]

        else:

            steps = [
                "Identify suitable numeric variables for anomaly detection.",
                "Calculate descriptive statistics.",
                "Use the IQR method to identify potential outliers.",
                "Use z-scores to identify extreme observations.",
                "Compare and summarize the detected anomalies."
            ]

    # ==================================================
    # DECISION SUPPORT
    # ==================================================

    elif task_type == "decision_support":

        steps = [
            "Calculate descriptive statistics for the key business metrics.",
            "Compare performance across relevant categorical groups.",
            "Identify the highest-performing and lowest-performing groups.",
            "Check for unusual observations that could affect the decision.",
            "Summarize the strongest evidence supporting the decision."
        ]

    # ==================================================
    # FALLBACK
    # ==================================================

    else:

        steps = [
            "Inspect the structure and data types of the dataset.",
            "Calculate descriptive statistics for numeric variables.",
            "Check missing values and data quality.",
            "Analyze important categorical variables.",
            "Identify important patterns and unusual observations."
        ]

    return steps
    # --------------------------------------------------
    # DESCRIPTIVE SUMMARY
    # --------------------------------------------------

    if task_type == "descriptive_summary":

        steps = [
            "Calculate descriptive statistics such as mean, median, standard deviation, minimum, and maximum for numeric columns.",
            "Calculate the number of missing values and missing-value percentage for every column.",
            "Identify the most frequent categories in categorical columns.",
            "Check the distribution and skewness of important numeric variables.",
            "Summarize the overall structure and characteristics of the dataset."
        ]

    # --------------------------------------------------
    # COMPARISON
    # --------------------------------------------------

    elif task_type == "comparison":

        group_column = (
            categorical_columns[0]
            if categorical_columns
            else None
        )

        numeric_column = (
            numeric_columns[0]
            if numeric_columns
            else None
        )

        if group_column and numeric_column:

            steps = [
                f"Group the data by '{group_column}' and calculate the total and mean of '{numeric_column}'.",
                f"Rank the '{group_column}' categories according to total '{numeric_column}'.",
                f"Compare the distribution of '{numeric_column}' across '{group_column}' categories.",
                f"Check for unusually high or low '{numeric_column}' values within each '{group_column}' category.",
                f"Identify the category with the highest total '{numeric_column}' and report the result."
            ]

        else:

            steps = [
                "Identify the main numeric variables available for comparison.",
                "Calculate summary statistics for the selected variables.",
                "Compare minimum, maximum, mean, and median values.",
                "Identify the highest and lowest observed values.",
                "Summarize the main differences found in the dataset."
            ]

    # --------------------------------------------------
    # TREND ANALYSIS
    # --------------------------------------------------

    elif task_type == "trend_analysis":

        if date_columns and numeric_columns:

            date_column = date_columns[0]
            numeric_column = numeric_columns[0]

            steps = [
                f"Convert '{date_column}' to datetime format and sort the data chronologically.",
                f"Aggregate '{numeric_column}' by month or another appropriate time period.",
                f"Calculate the change in '{numeric_column}' between consecutive time periods.",
                f"Identify periods with the largest increases and decreases in '{numeric_column}'.",
                f"Check the overall trend and identify any unusual changes over time."
            ]

        else:

            steps = [
                "Identify any available date or time-related column.",
                "Sort observations chronologically if a time variable is available.",
                "Aggregate important numeric variables over time.",
                "Calculate changes between consecutive periods.",
                "Identify the overall trend and unusual changes."
            ]

    # --------------------------------------------------
    # RELATIONSHIP ANALYSIS
    # --------------------------------------------------

    elif task_type == "relationship_analysis":

        if len(numeric_columns) >= 2:

            col1 = numeric_columns[0]
            col2 = numeric_columns[1]

            steps = [
                f"Calculate the Pearson correlation between '{col1}' and '{col2}'.",
                f"Calculate the Spearman correlation between '{col1}' and '{col2}'.",
                f"Check the distributions of '{col1}' and '{col2}' before selecting the appropriate correlation method.",
                f"Identify whether the relationship is positive, negative, or weak.",
                f"Check for potential outliers that may influence the correlation."
            ]

        else:

            steps = [
                "Identify numeric variables suitable for relationship analysis.",
                "Check the distributions of the available numeric variables.",
                "Calculate appropriate correlation statistics.",
                "Identify the strength and direction of relationships.",
                "Check whether outliers affect the observed relationship."
            ]

    # --------------------------------------------------
    # ANOMALY DETECTION
    # --------------------------------------------------

    elif task_type == "anomaly_detection":

        if numeric_columns:

            numeric_column = numeric_columns[0]

            steps = [
                f"Calculate the mean, median, and standard deviation of '{numeric_column}'.",
                f"Identify potential outliers in '{numeric_column}' using the IQR method.",
                f"Calculate z-scores for '{numeric_column}' and identify extreme observations.",
                f"Compare detected anomalies using both IQR and z-score methods.",
                f"Summarize the observations identified as potential anomalies."
            ]

        else:

            steps = [
                "Identify numeric variables suitable for anomaly detection.",
                "Calculate descriptive statistics for the numeric variables.",
                "Use the IQR method to identify potential outliers.",
                "Use standardized scores to identify extreme observations.",
                "Compare and summarize the detected anomalies."
            ]

    # --------------------------------------------------
    # DECISION SUPPORT
    # --------------------------------------------------

    elif task_type == "decision_support":

        steps = [
            "Calculate descriptive statistics for the key business metrics.",
            "Compare performance across relevant categorical groups.",
            "Identify the highest-performing and lowest-performing groups.",
            "Check for unusual observations that could affect the decision.",
            "Summarize the strongest evidence that can support the decision."
        ]

    # --------------------------------------------------
    # FALLBACK
    # --------------------------------------------------

    else:

        steps = [
            "Inspect the structure and data types of the dataset.",
            "Calculate descriptive statistics for numeric variables.",
            "Check missing values and data quality.",
            "Analyze important categorical variables.",
            "Identify important patterns and unusual observations."
        ]

    return steps


if __name__ == "__main__":

    from data_profiler import profile_dataset

    filepath = "C:/Users/Vijay/Desktop/Self_Validating_Data_Agent/data/raw/sales.csv"

    # -----------------------------------------------
    # Generate dataset profile
    # -----------------------------------------------

    profile = profile_dataset(filepath)

    # -----------------------------------------------
    # Test questions
    # -----------------------------------------------

    test_cases = [

        (
            "comparison",
            "Which region generated the highest total profit?"
        ),

        (
            "trend_analysis",
            "How did total revenue change over time?"
        ),

        (
            "relationship_analysis",
            "Is there a relationship between units sold and total profit?"
        ),

        (
            "anomaly_detection",
            "Are there any unusual total profit values?"
        ),

        (
            "descriptive_summary",
            "Give me a summary of the sales data."
        )

    ]

    print("=" * 60)
    print("EDA PLANNER TEST")
    print("=" * 60)

    for task_type, question in test_cases:

        print("\n" + "-" * 60)

        print("Task Type:")
        print(task_type)

        print("\nUser Question:")
        print(question)

        plan = generate_plan_mock(
            task_type,
            profile,
            question
        )

        print("\nGenerated EDA Plan:")

        for i, step in enumerate(plan, start=1):

            print(f"{i}. {step}")

    print("\n" + "=" * 60)
    print("EDA PLANNER TEST COMPLETED")
    print("=" * 60)