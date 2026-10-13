"""Rule-based task analyzer for the Self-Validating EDA Agent.

This version uses whole-word column matching, recognizes scalar aggregations,
and explicitly flags requested metrics that are absent from the dataset.
"""

import re

TASK_TYPES = [
    "descriptive_summary",
    "mean_analysis",
    "trend_analysis",
    "comparison",
    "anomaly_detection",
    "relationship_analysis",
    "decision_support",
]

MONTH_ALIASES = {
    "january": "jan", "february": "feb", "march": "mar", "april": "apr",
    "may": "may", "june": "jun", "july": "jul", "august": "aug",
    "september": "sep", "october": "oct", "november": "nov", "december": "dec",
}

# These aliases are only applied when the mapped column actually exists.
COLUMN_ALIASES = {
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
    "date": "Order Date",
    "converted customers": "conversion",
    "conversion rate": "conversion",
    # Bank Marketing dataset aliases
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


def _contains_term(text, term):
    """Return True when term appears as a whole phrase, not a substring."""
    term = str(term).strip().lower()
    if not term:
        return False
    # Treat underscores as word characters, as they are in Python column names.
    pattern = rf"(?<![\w]){re.escape(term)}(?![\w])"
    return re.search(pattern, text.lower()) is not None


def find_column(query, columns):
    """Find a dataset column explicitly mentioned or matched by a safe alias."""
    query_lower = str(query).lower()
    # Prefer the longest matching real column name, avoiding accidental matches
    # such as 'age' inside 'average' or 'y' inside 'salary'.
    matches = [
        column for column in columns
        if _contains_term(query_lower, str(column).lower().replace("_", " "))
    ]
    if matches:
        return max(matches, key=lambda value: len(str(value)))

    for alias, column in sorted(COLUMN_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        if column in columns and _contains_term(query_lower, alias):
            return column
    return None


def _metric_phrase(query):
    """Extract a likely metric named in a mean/average question."""
    patterns = [
        r"\b(?:average|avg|mean)\s+(?:value\s+of\s+)?(?:the\s+)?([a-zA-Z][a-zA-Z_ ]*?)\s+(?:of|for|among|across)\b",
        r"\b(?:average|avg|mean)\s+(?:value\s+of\s+)?(?:the\s+)?([a-zA-Z][a-zA-Z_]*)\b",
        r"\b(?:what is|what's|calculate|find|show|tell me)\s+(?:the\s+)?(?:average|avg|mean)\s+(?:of\s+)?(?:the\s+)?([a-zA-Z][a-zA-Z_ ]*?)(?:\?|$)",
    ]
    cleaned = str(query).strip().lower().rstrip("?.! ")
    for pattern in patterns:
        match = re.search(pattern, cleaned)
        if match:
            phrase = match.group(1).strip()
            phrase = re.sub(r"\b(customers?|clients?|people|records?|observations?|rows?)\b", "", phrase).strip()
            phrase = re.sub(r"\s+", " ", phrase)
            if phrase and phrase not in {"value", "of", "for", "the"}:
                return phrase
    return None


def analyze_task_mock(user_query, data_profile):
    """Classify a question and identify relevant columns from a dataset profile."""
    query = str(user_query).lower()
    profile_columns = data_profile.get("columns", {}) if isinstance(data_profile, dict) else {}
    columns = list(profile_columns.keys())
    query_column_matches = []

    for column in columns:
        searchable_name = str(column).lower().replace("_", " ")
        if _contains_term(query, searchable_name):
            query_column_matches.append(column)

    # Safe, dataset-aware aliases.
    for alias, column in sorted(COLUMN_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        if column in columns and _contains_term(query, alias) and column not in query_column_matches:
            query_column_matches.append(column)

    # BANK_CONVERSION_ROUTING_FIX_V1
    # Do not silently substitute "job" when the user named other factors.
    conversion_rate_comparison = (
        ("conversion rate" in query or "conversion rates" in query)
        and any(term in query for term in ("highest", "lowest", "which", "rank", "compare", "prioritize", "recommend"))
    )
    outcome_column = "conversion" if "conversion" in columns else ("y" if "y" in columns else None)

    if conversion_rate_comparison and outcome_column:
        requested_factors = [
            c for c in query_column_matches
            if c in columns and c not in ("conversion", "y")
        ]
        if requested_factors:
            return {
                "task_type": "relationship_analysis",
                "target_columns": list(dict.fromkeys(requested_factors + [outcome_column])),
                "analysis_intent": "conversion_factor_analysis",
                "metric_column": outcome_column,
                "requested_factors": list(dict.fromkeys(requested_factors)),
                "notes": (
                    "Conversion-factor analysis: use the factors explicitly named "
                    "in the question; do not substitute a default grouping column. "
                    "Report group sizes and observed conversion rates."
                ),
            }
        if "job" in columns and (_contains_term(query, "job") or _contains_term(query, "job category")):
            return {
                "task_type": "comparison",
                "target_columns": ["job", outcome_column],
                "group_column": "job",
                "metric_column": outcome_column,
                "comparison_metric": "conversion_rate",
                "notes": "Grouped conversion-rate request explicitly names job category.",
            }

    # Preserve filtered mean requests, including the August call-duration case.
    has_mean_intent = any(_contains_term(query, term) for term in ("average", "mean", "avg"))
    selected_month_name = next((name for name in MONTH_ALIASES if _contains_term(query, name)), None)
    if has_mean_intent and "duration" in columns and selected_month_name:
        return {
            "task_type": "mean_analysis",
            "target_columns": [c for c in ("duration", "month") if c in columns],
            "metric_column": "duration",
            "filter_column": "month",
            "filter_month": MONTH_ALIASES[selected_month_name],
            "requested_metric": "duration",
            "unsupported_metric": False,
            "notes": "Filtered mean request: filter rows to the named month before calculating the average.",
        }

    # Classify specific numeric aggregation before generic descriptive summary.
    if has_mean_intent:
        metric_phrase = _metric_phrase(query)
        if not query_column_matches and metric_phrase:
            # Only flag the likely metric as unsupported when it does not match a
            # real column or a valid dataset-aware alias.
            return {
                "task_type": "mean_analysis",
                "target_columns": [],
                "metric_column": None,
                "requested_metric": metric_phrase,
                "unsupported_metric": True,
                "notes": f"The requested metric '{metric_phrase}' was not found among the dataset columns.",
            }
        metric_column = next(
            (c for c in query_column_matches if profile_columns.get(c, {}).get("role") == "continuous"),
            query_column_matches[0] if query_column_matches else None,
        )
        return {
            "task_type": "mean_analysis",
            "target_columns": [metric_column] if metric_column else query_column_matches,
            "metric_column": metric_column,
            "requested_metric": metric_column or metric_phrase,
            "unsupported_metric": metric_column is None and bool(metric_phrase),
            "notes": "Calculate the mean of the metric explicitly requested by the user.",
        }

    # BEGIN CONVERSION-FACTOR INTENT RULE
    conversion_factor_intent = (
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
                "pre-contact",
                "before a marketing call",
                "prioritize",
                "recommend",
            )
        )
    )

    if conversion_factor_intent:
        conversion_targets = [
            column for column in ("conversion", "y")
            if column in profile_columns
        ]
        requested_factors = [
            column for column in query_column_matches
            if column in profile_columns and column not in ("conversion", "y")
        ]
        conversion_targets = list(dict.fromkeys(requested_factors + conversion_targets))

        return {
            "task_type": "relationship_analysis",
            "target_columns": conversion_targets,
            "requested_factors": requested_factors,
            "analysis_intent": "conversion_factor_analysis",
            "metric_column": (
                "conversion" if "conversion" in profile_columns
                else "y" if "y" in profile_columns
                else None
            ),
            "notes": (
                "Conversion-factor analysis: compare conversion outcomes "
                "across available customer attributes and numeric factors."
            ),
        }
    # END CONVERSION-FACTOR INTENT RULE

    if any(word in query for word in ("trend", "over time", "change over time", "monthly", "yearly", "quarter", "increase", "decrease", "changed")):
        task_type = "trend_analysis"
    elif any(word in query for word in ("relationship", "related", "correlation", "correlated", "association")):
        task_type = "relationship_analysis"
    elif any(word in query for word in ("anomaly", "anomalies", "outlier", "outliers", "unusual", "abnormal")):
        task_type = "anomaly_detection"
    elif any(word in query for word in ("should we", "recommend", "recommendation", "decision", "focus on")):
        task_type = "decision_support"
    elif any(word in query for word in ("which", "highest", "lowest", "compare", "comparison", "better", "best", "worst", "rank")):
        task_type = "comparison"
    else:
        task_type = "descriptive_summary"

    # Trend analysis may require an available date/time column.
    if task_type == "trend_analysis":
        time_columns = [c for c, info in profile_columns.items() if info.get("role") == "time_index"]
        for column in time_columns:
            if column not in query_column_matches:
                query_column_matches.insert(0, column)

    if task_type == "relationship_analysis":
        for column, info in profile_columns.items():
            if info.get("role") == "continuous" and _contains_term(query, str(column).lower().replace("_", " ")):
                if column not in query_column_matches:
                    query_column_matches.append(column)

    return {
        "task_type": task_type,
        "target_columns": query_column_matches,
        "notes": "Task classified using a rule-based analyzer and the available dataset profile.",
    }


if __name__ == "__main__":
    print("Import analyze_task_mock() and pass a dataset profile to test the analyzer.")
