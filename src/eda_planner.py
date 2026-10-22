import json
import re


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

    # BANK_CONVERSION_ROUTING_FIX_V1
    # Match natural-language names to columns actually present in this dataset.
    bank_aliases = {
        "account balance": "balance",
        "contact method": "contact",
        "housing loan": "housing",
        "personal loan": "loan",
        "job category": "job",
        "previous campaign outcome": "poutcome",
        "previous outcome": "poutcome",
        "previously contacted": "previously_contacted",
        "call duration": "duration",
    }
    requested_factors = []
    for alias, column in sorted(bank_aliases.items(), key=lambda item: len(item[0]), reverse=True):
        if column in columns and re.search(r"(?<![\w])" + re.escape(alias) + r"(?![\w])", query):
            if column not in requested_factors:
                requested_factors.append(column)
    for column in columns:
        normalized = str(column).lower().replace("_", " ")
        if re.search(r"(?<![\w])" + re.escape(normalized) + r"(?![\w])", query):
            if column not in ("conversion", "y") and column not in requested_factors:
                requested_factors.append(column)

    # BEGIN CONVERSION-FACTOR PLAN RULE
    conversion_factor_request = (
        "conversion" in query
        and any(
            phrase in query
            for phrase in (
                "factor",
                "associated",
                "affecting",
                "influencing",
                "important patterns",
                "patterns in",
                "customer conversion",
                "drivers of",
                "predictors of",
                "conversion rate",
                "conversion rates",
                "prioritize",
                "recommend",
                "pre-contact",
                "before a marketing call",
            )
        )
    )

    if conversion_factor_request and (
        "conversion" in columns or "y" in columns
    ):
        if requested_factors:
            factors_text = ", ".join(f"'{column}'" for column in requested_factors)
            factor_scope = (
                f"Analyze the explicitly requested factor columns: {factors_text}. "
                "Do not replace them with a default column."
            )
        else:
            factor_scope = (
                "Compare conversion rates across available customer attributes "
                "and analyze numeric factors by meaningful groups."
            )
        return [
            (
                "Identify customer conversion outcomes and calculate the overall "
                "conversion rate. " + factor_scope + " For each requested factor, "
                "report group sizes, converted counts, observed conversion rates, "
                "and the highest/lowest groups where meaningful. Validate all "
                "rates against the source dataset. Treat findings as observational, "
                "not proof of causation. For pre-contact recommendations, exclude "
                "post-contact variables such as call duration."
            )
        ]
    # END CONVERSION-FACTOR PLAN RULE


    # Job-only fallback is permitted only when job is explicitly requested.
    if (("conversion rate" in query or "conversion rates" in query)
            and any(term in query for term in ("highest", "lowest", "which", "rank", "compare"))
            and "job" in columns
            and ("job" in requested_factors)
            and len(requested_factors) == 1):
        outcome_column = "conversion" if "conversion" in columns else ("y" if "y" in columns else None)
        if outcome_column:
            return [
                "Group customers by the explicitly requested 'job' category.",
                f"Within each job category, count converted customers using '{outcome_column}' (1/yes means converted).",
                "Calculate each job category's observed conversion rate as converted customers divided by all customers in that category, multiplied by 100.",
                "Rank job categories by observed conversion rate in descending order and report group sizes.",
                "Validate the grouped counts and rates against the uploaded dataset; do not claim causation."
            ]

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

    # GROUPED MEAN: compare the explicitly requested numeric metric by subscription outcome.
    grouped_mean_comparison = (
        task_type == "comparison"
        and any(term in query.split() for term in ("average", "mean", "avg"))
        and any(
            term in query
            for term in ("compare", "comparison", "across", "between")
        )
        and any(term in query for term in ("subscrib", "subscription"))
        and ("y" in columns or "conversion" in columns)
    )

    if grouped_mean_comparison:
        group_column = "y" if "y" in columns else "conversion"
        matched_metrics = [
            col for col in columns
            if col not in ("y", "conversion")
            and data_profile.get("columns", {}).get(col, {}).get("role") == "continuous"
            and re.search(
                r"(?<![\w])" + re.escape(str(col).replace("_", " ")) + r"(?![\w])",
                query,
            )
        ]
        if not matched_metrics:
            aliases = {
                "account balance": "balance",
                "call duration": "duration",
            }
            matched_metrics = [
                destination for alias, destination in aliases.items()
                if destination in columns
                and re.search(r"(?<![\w])" + re.escape(alias) + r"(?![\w])", query)
            ]
        if not matched_metrics:
            return [
                "STOP: The requested numeric metric could not be matched to a dataset column. Do not substitute another metric."
            ]
        plan = []
        for metric_column in matched_metrics:
            plan.append(
                f"Compare the average '{metric_column}' grouped by "
                f"subscription outcome using '{group_column}' (yes/no). "
                "Report each group's row count, valid numeric count, mean, "
                "and the yes-minus-no difference."
            )
            plan.append(
                f"Independently recompute and validate the average "
                f"'{metric_column}' grouped by subscription outcome using "
                f"'{group_column}' (yes/no), reporting each group's row "
                "count, valid numeric count, mean, and the yes-minus-no difference."
            )
        return plan



    # ==================================================
    # Specific mean/average questions with filter preservation
    # ==================================================

    if task_type == "mean_analysis":
        def _whole_phrase_match(text, phrase):
            phrase = str(phrase).strip().replace("_", " ")
            if not phrase:
                return False
            pattern = r"(?<![\w])" + re.escape(phrase) + r"(?![\w])"
            return re.search(pattern, text, flags=re.IGNORECASE) is not None

        requested_column = next(
            (
                col for col in sorted(
                    columns, key=lambda value: len(str(value)), reverse=True
                )
                if _whole_phrase_match(query, str(col))
            ),
            None,
        )

        if requested_column is None:
            mean_aliases = {
                "revenue": "Total Revenue",
                "sales": "Total Revenue",
                "profit": "Total Profit",
                "cost": "Total Cost",
                "units": "Units Sold",
                "quantity": "Units Sold",
                "account balance": "balance",
                "call duration": "duration",
            }
            for alias, destination in sorted(
                mean_aliases.items(), key=lambda item: len(item[0]), reverse=True
            ):
                if destination in columns and _whole_phrase_match(query, alias):
                    requested_column = destination
                    break

        if requested_column is None:
            return [
                "STOP: The requested mean/average metric could not be matched to a dataset column. Do not substitute another metric. Return an unsupported-metric response and list available columns."
            ]

        # Preserve supported filters explicitly mentioned by the user.
        filter_instructions = []

        month_names = {
            "january": "jan",
            "february": "feb",
            "march": "mar",
            "april": "apr",
            "may": "may",
            "june": "jun",
            "july": "jul",
            "august": "aug",
            "september": "sep",
            "october": "oct",
            "november": "nov",
            "december": "dec",
        }

        selected_month = next(
            (
                name for name in month_names
                if _whole_phrase_match(query, name)
            ),
            None,
        )

        if selected_month and "month" in columns:
            filter_instructions.append(
                f"filter the source rows where 'month' equals "
                f"'{month_names[selected_month]}' (also accept the full month name)"
            )

        if "marital" in columns:
            marital_value = next(
                (
                    value for value in ("single", "married", "divorced")
                    if _whole_phrase_match(query, value)
                ),
                None,
            )
            if marital_value:
                filter_instructions.append(
                    f"filter the source rows where 'marital' equals "
                    f"'{marital_value}'"
                )

        filter_text = (
            "First, " + " and ".join(filter_instructions) + ". Then, "
            if filter_instructions
            else ""
        )

        return [
            f"{filter_text}calculate the arithmetic mean of numeric values in "
            f"'{requested_column}', excluding missing values. Report the metric "
            f"name, mean, filtered row count, and valid numeric observation count. "
            f"Do not calculate the mean on unfiltered rows when filters were requested.",
            f"Independently validate the mean of '{requested_column}' against the "
            f"source dataset after applying exactly the same requested filters. "
            f"Confirm the filtered row count and valid numeric observation count."
        ]

    # ==================================================


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
                "Detect multivariate anomalies using Isolation Forest across suitable numeric features.",
                f"Compare anomalies detected using the IQR and z-score methods.",
                f"Summarize the observations identified as potential anomalies in '{value_column}'."
            ]

        else:

            steps = [
                "Identify suitable numeric variables for anomaly detection.",
                "Calculate descriptive statistics.",
                "Use the IQR method to identify potential outliers.",
                "Detect multivariate anomalies using Isolation Forest across suitable numeric features.",
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
