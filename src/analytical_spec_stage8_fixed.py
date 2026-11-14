"""
analytical_spec.py

Shared, dataset-aware analytical specification for the Self-Validating EDA Agent.

This module converts a user question and dataset profile into a structured,
backward-compatible description of the requested analysis. It is deliberately
conservative: it records only requirements that can be identified with
reasonable confidence and flags ambiguous or unresolved requests instead of
silently guessing.

The specification is an intermediate representation. It does not generate
Python code and does not claim that every natural-language request is supported.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


SCHEMA_VERSION = "1.0"

OPERATION_PATTERNS: Sequence[Tuple[str, Sequence[str]]] = (
    ("mean", (r"\baverage\b", r"\bavg\b", r"\bmean\b")),
    ("median", (r"\bmedian\b", r"\bmidpoint\b")),
    ("sum", (r"\bsum\b", r"\btotal\b")),
    ("count", (r"\bcount\b", r"\bnumber of\b", r"\bhow many\b")),
    ("min", (r"\bminimum\b", r"\bsmallest\b", r"\blowest\b")),
    ("max", (r"\bmaximum\b", r"\blargest\b", r"\bhighest\b")),
    ("std", (r"\bstandard deviation\b", r"\bstd(?:dev)?\b")),
    ("variance", (r"\bvariance\b",)),
    ("quantile", (r"\bquartile\b", r"\bpercentile\b", r"\bquantile\b")),
    ("nunique", (r"\bdistinct\b", r"\bunique\b", r"\bdifferent values\b")),
)

GROUP_MARKERS = (
    "grouped by", "group by", "by", "per", "for each", "within each",
    "across", "among",
)

TIME_GRAINS = {
    "daily": "D",
    "day": "D",
    "weekly": "W",
    "week": "W",
    "monthly": "M",
    "month": "M",
    "quarterly": "Q",
    "quarter": "Q",
    "yearly": "Y",
    "annual": "Y",
    "year": "Y",
}

COLUMN_ALIASES = {
    "revenue": ("revenue", "sales"),
    "profit": ("profit",),
    "cost": ("cost",),
    "units": ("units", "quantity", "volume"),
    "date": ("date", "time"),
    "salary": ("salary", "pay", "compensation"),
    "age": ("age",),
}


def _normalise(value: Any) -> str:
    """Normalize text for matching without changing the original column name."""
    return re.sub(r"[\W_]+", " ", str(value).casefold()).strip()


def _whole_phrase_present(text: str, phrase: str) -> bool:
    text_norm = f" {_normalise(text)} "
    phrase_norm = f" {_normalise(phrase)} "
    return bool(phrase_norm.strip()) and phrase_norm in text_norm


def _profile_columns(data_profile: Any) -> Dict[str, Dict[str, Any]]:
    if not isinstance(data_profile, dict):
        return {}
    columns = data_profile.get("columns", {})
    if not isinstance(columns, dict):
        return {}
    return {
        str(name): info if isinstance(info, dict) else {}
        for name, info in columns.items()
    }


def _match_columns(question: str, columns: Dict[str, Dict[str, Any]]) -> List[str]:
    """Return explicit dataset-column mentions, longest names first."""
    matches = []
    for name in columns:
        searchable = name.replace("_", " ")
        # Handle ordinary English plural mentions (department/departments)
        # while retaining the exact dataset column as the resolved name.
        candidates = [searchable]
        if searchable.casefold().endswith("y") and len(searchable) > 1:
            candidates.append(searchable[:-1] + "ies")
        elif searchable.casefold().endswith(("s", "x", "z", "ch", "sh")):
            candidates.append(searchable + "es")
        else:
            candidates.append(searchable + "s")
        if any(_whole_phrase_present(question, candidate) for candidate in candidates):
            matches.append(name)
    return sorted(set(matches), key=lambda name: (-len(_normalise(name)), name.casefold()))


def _resolve_aliases(
    question: str,
    columns: Dict[str, Dict[str, Any]],
    already_matched: Iterable[str],
) -> List[str]:
    """Resolve conservative semantic aliases only when exactly one column fits."""
    matched = set(already_matched)
    resolved: List[str] = []

    for column in columns:
        norm_column = _normalise(column)
        for concept, aliases in COLUMN_ALIASES.items():
            if concept not in norm_column:
                continue
            if any(_whole_phrase_present(question, alias) for alias in aliases):
                resolved.append(column)
                break

    # Keep the result deterministic and do not add duplicate columns.
    return [name for name in dict.fromkeys(resolved) if name not in matched]


def _extract_operations(question: str) -> List[str]:
    operations: List[str] = []
    for operation, patterns in OPERATION_PATTERNS:
        if any(re.search(pattern, question, flags=re.IGNORECASE) for pattern in patterns):
            operations.append(operation)
    return operations


def _extract_group_columns(
    question: str,
    matched_columns: Sequence[str],
    profile_columns: Dict[str, Dict[str, Any]],
) -> List[str]:
    """Select mentioned categorical columns when grouping language is present."""
    q = question.casefold()
    has_grouping_language = any(
        re.search(rf"\b{re.escape(marker)}\b", q) for marker in GROUP_MARKERS
    )
    if not has_grouping_language:
        return []

    categorical = [
        name for name in matched_columns
        if profile_columns.get(name, {}).get("role") in {"categorical", "category"}
    ]
    return list(dict.fromkeys(categorical))


def _extract_filters(
    question: str,
    profile_columns: Dict[str, Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Parse simple, explicit numeric comparisons.

    Supported forms include:
      age > 30
      age greater than 30
      salary at least 50000
      balance below 1000

    Categorical and compound natural-language filters are intentionally not
    inferred here; they should be handled by a later semantic parser.
    """
    filters: List[Dict[str, Any]] = []
    issues: List[str] = []

    numeric_columns = [
        name for name, info in profile_columns.items()
        if info.get("role") in {"continuous", "numeric", "integer", "float"}
    ]

    operators = [
        (r"(?:>=|at least|greater than or equal to)", ">="),
        (r"(?:<=|at most|less than or equal to)", "<="),
        (r"(?:!=|not equal to)", "!="),
        (r"(?:>|greater than|more than|above|over)", ">"),
        (r"(?:<|less than|below|under)", "<"),
        (r"(?:==|=|equal to|equals)", "=="),
    ]

    for column in numeric_columns:
        escaped = re.escape(column)
        for operator_pattern, operator in operators:
            pattern = (
                rf"(?<![\w]){escaped}(?![\w])\s*"
                rf"{operator_pattern}\s*(-?\d+(?:\.\d+)?)"
            )
            match = re.search(pattern, question, flags=re.IGNORECASE)
            if match:
                filters.append({
                    "column": column,
                    "operator": operator,
                    "value": float(match.group(1)),
                    "value_type": "numeric",
                    "source_text": match.group(0),
                })
                break

    return filters, issues


def _extract_limit(question: str) -> Optional[int]:
    patterns = (
        r"\btop\s+(\d+)\b",
        r"\bfirst\s+(\d+)\b",
        r"\bbottom\s+(\d+)\b",
        r"\blast\s+(\d+)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, question, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def _extract_sort_direction(question: str) -> Optional[str]:
    q = question.casefold()
    if any(term in q for term in ("ascending", "lowest to highest", "smallest to largest")):
        return "ascending"
    if any(term in q for term in ("descending", "highest to lowest", "largest to smallest", "top")):
        return "descending"
    if "bottom" in q:
        return "ascending"
    return None


def _extract_time_grain(question: str) -> Optional[str]:
    q = question.casefold()
    for phrase, grain in TIME_GRAINS.items():
        if re.search(rf"\b{re.escape(phrase)}\b", q):
            return grain
    return None


def build_analytical_spec(
    user_question: str,
    data_profile: Any,
    legacy_analysis: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a versioned analytical specification.

    `legacy_analysis` is optional. When supplied, its task_type and explicitly
    resolved target columns are retained for compatibility with existing
    reporting code. New consumers should use `analytical_spec`.
    """
    question = str(user_question or "").strip()
    profile_columns = _profile_columns(data_profile)
    issues: List[str] = []

    if not question:
        issues.append("The analytical question is empty.")

    explicit_matches = _match_columns(question, profile_columns)
    alias_matches = _resolve_aliases(question, profile_columns, explicit_matches)
    mentioned_columns = list(dict.fromkeys(explicit_matches + alias_matches))

    group_by = _extract_group_columns(question, mentioned_columns, profile_columns)

    # Preserve a categorical grouping column already resolved by the legacy
    # task analyzer. Semantic group descriptions (for example, "customers who
    # subscribed and those who did not") may not contain the physical dataset
    # column name, so question-only matching would otherwise drop that column.
    if isinstance(legacy_analysis, dict):
        legacy_group = legacy_analysis.get("group_column")
        legacy_task = str(legacy_analysis.get("task_type", "")).casefold()
        legacy_comparison_metric = str(
            legacy_analysis.get("comparison_metric", "")
        ).casefold()
        legacy_group_info = profile_columns.get(legacy_group, {})
        legacy_group_role = legacy_group_info.get("role")
        legacy_group_is_categorical = legacy_group_role in {
            "categorical", "category", "boolean", "bool"
        }
        legacy_group_is_relevant = (
            legacy_task == "comparison"
            or legacy_comparison_metric in {"mean", "median", "sum", "count", "conversion_rate"}
            or bool(legacy_analysis.get("analysis_intent") == "conversion_factor_analysis")
        )
        if (
            isinstance(legacy_group, str)
            and legacy_group in profile_columns
            and legacy_group_is_categorical
            and legacy_group_is_relevant
        ):
            if legacy_group not in group_by:
                group_by.append(legacy_group)
            if legacy_group not in mentioned_columns:
                mentioned_columns.append(legacy_group)

    numeric_mentions = [
        name for name in mentioned_columns
        if profile_columns.get(name, {}).get("role") in {
            "continuous", "numeric", "integer", "float"
        }
    ]
    categorical_mentions = [
        name for name in mentioned_columns
        if profile_columns.get(name, {}).get("role") in {"categorical", "category"}
    ]

    operations = _extract_operations(question)
    filters, filter_issues = _extract_filters(question, profile_columns)
    issues.extend(filter_issues)

    # A metric is a mentioned numeric column that is not being used as a group.
    filter_columns = {item["column"] for item in filters}
    metrics = [
        name for name in numeric_mentions
        if name not in group_by and name not in filter_columns
    ]

    # If the legacy analyzer resolved a metric, preserve it when the question
    # does not explicitly expose a numeric column through direct matching.
    if not metrics and isinstance(legacy_analysis, dict):
        legacy_metric = legacy_analysis.get("metric_column")
        if legacy_metric in profile_columns:
            metrics = [legacy_metric]

    if not operations:
        if any(term in question.casefold() for term in ("describe", "summary", "profile")):
            operations = ["describe"]
        elif any(term in question.casefold() for term in ("missing", "null", "empty values")):
            operations = ["missing_values"]
        elif any(term in question.casefold() for term in ("duplicate", "duplicated")):
            operations = ["duplicate_count"]

    if not operations and not any(
        term in question.casefold()
        for term in ("relationship", "correlation", "outlier", "anomaly", "trend")
    ):
        issues.append("No supported analytical operation could be identified confidently.")

    if any(term in question.casefold() for term in ("correlation", "correlated", "relationship")):
        operations = list(dict.fromkeys(operations + ["correlation"]))

    if any(term in question.casefold() for term in ("outlier", "outliers", "anomaly", "anomalies")):
        operations = list(dict.fromkeys(operations + ["outlier_detection"]))

    if any(term in question.casefold() for term in ("trend", "over time", "monthly", "yearly")):
        operations = list(dict.fromkeys(operations + ["time_trend"]))

    if any(term in question.casefold() for term in ("rank", "top ", "bottom ")):
        operations = list(dict.fromkeys(operations + ["rank"]))

    comparisons: List[Dict[str, Any]] = []
    if any(term in question.casefold() for term in ("compare", "comparison", "versus", " vs ")):
        comparisons.append({
            "type": "group_comparison" if group_by else "unspecified_comparison",
            "group_by": list(group_by),
            "metrics": list(metrics),
        })

    requested_outputs: List[str] = []
    q = question.casefold()
    if any(term in q for term in ("chart", "plot", "graph", "visualize", "visualise")):
        requested_outputs.append("visualization")
    if any(term in q for term in ("table", "tabular")):
        requested_outputs.append("table")
    if any(term in q for term in ("explain", "interpret", "insight", "recommend")):
        requested_outputs.append("interpretation")
    if not requested_outputs:
        requested_outputs.append("result")

    statistical_methods: List[str] = []
    if "pearson" in q:
        statistical_methods.append("pearson")
    if "spearman" in q:
        statistical_methods.append("spearman")
    if "t-test" in q or "t test" in q:
        statistical_methods.append("t_test")
    if "chi-square" in q or "chi square" in q:
        statistical_methods.append("chi_square")

    time_columns = [
        name for name, info in profile_columns.items()
        if info.get("role") in {"time_index", "datetime", "date"}
    ]
    time_column = next((name for name in mentioned_columns if name in time_columns), None)
    if time_column is None and "time_trend" in operations and len(time_columns) == 1:
        time_column = time_columns[0]

    sort_direction = _extract_sort_direction(question)
    limit = _extract_limit(question)

    validation_requirements = [
        "execution_succeeded",
        "result_non_empty",
        "requested_columns_exist",
        "requested_operations_match",
    ]
    if group_by:
        validation_requirements.append("grouping_columns_preserved")
    if filters:
        validation_requirements.append("filter_conditions_applied")
    if statistical_methods:
        validation_requirements.append("statistical_assumptions_reviewed")

    legacy_task_type = (
        legacy_analysis.get("task_type")
        if isinstance(legacy_analysis, dict) else None
    )

    spec = {
        "schema_version": SCHEMA_VERSION,
        "original_question": question,
        "task_type": legacy_task_type,
        "operations": operations,
        "metrics": metrics,
        "group_by": group_by,
        "categorical_columns": categorical_mentions,
        "filters": filters,
        "comparisons": comparisons,
        "statistics": statistical_methods,
        "time": {
            "column": time_column,
            "grain": _extract_time_grain(question),
        },
        "sorting": {
            "direction": sort_direction,
            "limit": limit,
        },
        "requested_outputs": requested_outputs,
        "validation_requirements": validation_requirements,
        "mentioned_columns": mentioned_columns,
        "issues": issues,
        "status": "READY" if not issues else "REVIEW",
    }

    return spec
