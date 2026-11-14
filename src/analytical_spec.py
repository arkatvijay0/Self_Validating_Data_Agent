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


# Words that end the "group by" clause. Columns mentioned after them belong to
# a filter or another clause, not to the grouping
# (e.g. "average campaign by loan where contact is cellular").
_GROUP_CLAUSE_TERMINATORS = re.compile(
    r"\b(?:where|when|if|whose|having|excluding|except|only|for|with|who|that|which)\b"
    r"|[;:?.!]",
    flags=re.IGNORECASE,
)


def _extract_group_columns(
    question: str,
    matched_columns: Sequence[str],
    profile_columns: Dict[str, Dict[str, Any]],
) -> List[str]:
    """
    Select the categorical columns named in the grouping clause.

    Only columns mentioned AFTER the grouping marker are grouping columns.
    Previously every categorical column mentioned anywhere in the question was
    used once any grouping word was present, so
      "average campaign contacts grouped by loan status"
    grouped by [contact, loan]: "contacts" (part of the metric phrase) matched
    the `contact` column through plural matching.
    """
    q = question.casefold()

    marker_end = None
    for marker in GROUP_MARKERS:              # ordered most to least specific
        found = re.search(rf"\b{re.escape(marker)}\b", q)
        if found:
            marker_end = found.end()
            break
    if marker_end is None:
        return []

    clause = question[marker_end:]
    terminator = _GROUP_CLAUSE_TERMINATORS.search(clause)
    if terminator:
        clause = clause[:terminator.start()]

    categorical_roles = {"categorical", "category"}
    in_clause = set(_match_columns(clause, profile_columns))
    categorical = [
        name for name in matched_columns
        if name in in_clause
        and profile_columns.get(name, {}).get("role") in categorical_roles
    ]

    # Ranking phrasing: "top 5 departments BY average salary" - here "by"
    # introduces the metric, and the group column precedes the marker. Only
    # used when the clause after the marker names a metric/operation and no
    # grouping column, so "average contacts by <unknown>" is not misread.
    if not categorical:
        clause_names_metric = bool(_extract_operations(clause)) or any(
            profile_columns.get(name, {}).get("role") not in categorical_roles
            for name in in_clause
        )
        if clause_names_metric:
            head = question[: max(marker_end - 1, 0)]
            head_columns = set(_match_columns(head, profile_columns))
            categorical = [
                name for name in matched_columns
                if name in head_columns
                and profile_columns.get(name, {}).get("role") in categorical_roles
            ]
            clause = head

    # Keep the order in which the user wrote the columns.
    normalised_clause = _normalise(clause)

    def position(name: str) -> int:
        found = re.search(re.escape(_normalise(name)), normalised_clause)
        return found.start() if found else len(normalised_clause)

    return list(dict.fromkeys(sorted(categorical, key=position)))


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
                rf"(?<![\w]){escaped}(?![\w])\s*(?:(?:is|are|was|were)\s+)?"
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


_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "fifteen": 15, "twenty": 20,
}
_NUMBER_WORD_PATTERN = "|".join(_NUMBER_WORDS)


def _extract_limit(question: str) -> Optional[int]:
    """Parse 'top 5', 'top five', 'bottom 3', 'first ten' ... (digits or words)."""
    pattern = rf"\b(?:top|first|bottom|last)\s+(\d+|{_NUMBER_WORD_PATTERN})\b"
    match = re.search(pattern, question, flags=re.IGNORECASE)
    if not match:
        return None
    token = match.group(1).casefold()
    return int(token) if token.isdigit() else _NUMBER_WORDS.get(token)


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


# ---------------------------------------------------------------------------
# Grouped ranking questions
#   "Which job category has the highest average balance?"
#   "Which month has the highest average campaign duration?"
#   "Identify the top three job categories by customer count."
# These name the group BEFORE an extremum word and contain no "by/per/for each"
# marker, so marker-based grouping cannot see them. Without this they were
# misread as scalar max/min questions on an ungrouped metric.
# ---------------------------------------------------------------------------
_CATEGORICAL_ROLES = {"categorical", "category", "boolean", "bool"}
_NUMERIC_ROLES = {"continuous", "numeric", "integer", "float"}
_DESC_WORDS = r"highest|largest|greatest|biggest|most|maximum|top|best"
_ASC_WORDS = r"lowest|smallest|least|fewest|minimum|bottom|worst"
_COUNT_NOUNS = r"customers?|clients?|people|persons?|records?|rows?|observations?|entries|members?|individuals?"
_EXPLICIT_GROUP_MARKER = re.compile(
    r"\b(?:grouped by|group by|per|for each|within each|across|among)\b", re.IGNORECASE
)
_AGGREGATE_WORDS = (
    ("mean", r"\baverage\b|\bmean\b|\bavg\b"),
    ("median", r"\bmedian\b"),
    ("sum", r"\btotal\b|\bsum\b"),
)


def _column_variants(name: str) -> List[str]:
    searchable = name.replace("_", " ")
    variants = [searchable]
    low = searchable.casefold()
    if low.endswith("y") and len(searchable) > 1:
        variants.append(searchable[:-1] + "ies")
    elif low.endswith(("s", "x", "z", "ch", "sh")):
        variants.append(searchable + "es")
    else:
        variants.append(searchable + "s")
    return variants


def _column_spans(text: str, name: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    for variant in _column_variants(name):
        for found in re.finditer(rf"(?<!\w){re.escape(variant)}(?!\w)", text, flags=re.IGNORECASE):
            spans.append((found.start(), found.end()))
    return spans


def _columns_in(text: str, profile_columns: Dict[str, Dict[str, Any]], roles: set) -> List[Tuple[str, int, int]]:
    """Columns of the given roles mentioned in text, as (name, start, end), by position."""
    found: List[Tuple[str, int, int]] = []
    for name, info in profile_columns.items():
        if info.get("role") not in roles:
            continue
        spans = _column_spans(text, name)
        if spans:
            start, end = min(spans)
            found.append((name, start, end))
    # a longer column name wins over a shorter one it contains (age_group vs age)
    found.sort(key=lambda item: (item[1], -(item[2] - item[1])))
    kept: List[Tuple[str, int, int]] = []
    for item in found:
        if any(item[1] >= k[1] and item[2] <= k[2] and item[0] != k[0] for k in kept):
            continue
        kept.append(item)
    return kept



_COMPARISON_WORDS = re.compile(
    r"\b(?:higher|lower|greater|larger|bigger|smaller|more|less|better|worse|differ\w*|difference)\b",
    re.IGNORECASE,
)
_WITH_CLAUSE = re.compile(
    r"\b(?:with|without|who\s+(?:have|has|had|subscribed|converted)|that\s+(?:have|has|had))\b",
    re.IGNORECASE,
)
# Words that name an outcome column without using its physical name.
_OUTCOME_WORDS = {
    "subscribed": "y", "subscribers": "y", "subscription": "y", "subscribe": "y",
    "converted": "y", "conversion": "y",
}


def _detect_group_comparison(
    question: str,
    profile_columns: Dict[str, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    "Is the average balance higher or lower for customers with a loan?"
    "Do customers with a housing loan have a higher balance than those without?"
    -> compare the metric's mean across the groups of the named attribute.
    """
    text = str(question or "")
    clause = _WITH_CLAUSE.search(text)
    if not clause or not _COMPARISON_WORDS.search(text):
        return None
    tail = text[clause.start():]
    groups = _columns_in(tail, profile_columns, _CATEGORICAL_ROLES)
    group = groups[0][0] if groups else None
    if group is None:
        for word, column in _OUTCOME_WORDS.items():
            if re.search(rf"\b{word}\b", tail, flags=re.IGNORECASE) and column in profile_columns:
                group = column
                break
    if group is None:
        return None
    metrics = [m for m, _, _ in _columns_in(text, profile_columns, _NUMERIC_ROLES) if m != group]
    if len(metrics) != 1:
        return None
    return {"kind": "extrema", "group": group, "metrics": metrics,
            "aggregate": "mean", "direction": "descending", "limit": None}


def detect_grouped_ranking(
    question: str,
    profile_columns: Dict[str, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Detect a grouped ranking question and return its structure, or None.

    Returns {"group", "metrics", "aggregate", "direction", "limit", "kind"}
    where aggregate is mean/median/sum/count and kind is "extrema" or "top_n".
    Questions that already carry an explicit grouping marker, mention
    conversion (handled by dedicated logic), or have no resolvable group and
    metric are left untouched (None).
    """
    text = str(question or "").strip()
    if not text or not isinstance(profile_columns, dict) or not profile_columns:
        return None
    low = text.casefold()
    if _EXPLICIT_GROUP_MARKER.search(low) or "conversion" in low:
        return None

    # ---- top/bottom N <group> by count -------------------------------------
    count_by = re.search(
        rf"\bby\s+(?:the\s+)?(?:(?:customer|client|record|row|observation|member)\s+)?"
        rf"(?:counts?|number|frequency|size)\b|\bby\s+(?:the\s+)?number\s+of\s+(?:{_COUNT_NOUNS})\b",
        low,
    )
    if count_by and re.search(r"\b(?:top|bottom|first|last)\b", low):
        groups = _columns_in(text[: count_by.start()], profile_columns, _CATEGORICAL_ROLES)
        if groups:
            bottom = bool(re.search(r"\b(?:bottom|last)\b", low))
            return {
                "kind": "top_n", "group": groups[-1][0], "metrics": [],
                "aggregate": "count",
                "direction": "ascending" if bottom else "descending",
                "limit": _extract_limit(text),
            }

    # ---- most/least common <group> -----------------------------------------
    common = re.search(r"\b(most|least)\s+(?:common|frequent|popular)\b", low)
    if common:
        groups = _columns_in(text[common.end():], profile_columns, _CATEGORICAL_ROLES)
        if groups:
            return {
                "kind": "extrema", "group": groups[0][0], "metrics": [],
                "aggregate": "count",
                "direction": "ascending" if common.group(1) == "least" else "descending",
                "limit": 1,
            }

    # ---- which/what <group> has the highest/lowest [aggregate] <metric> ----
    if re.search(r"\bby\b", low):
        return None  # "top 5 jobs by average balance" is handled by the rank path
    extremum = re.search(rf"\b(?P<desc>{_DESC_WORDS})\b|\b(?P<asc>{_ASC_WORDS})\b", low)
    if not extremum:
        return _detect_group_comparison(text, profile_columns)
    head, tail = text[: extremum.start()], text[extremum.end():]
    groups = _columns_in(head, profile_columns, _CATEGORICAL_ROLES)
    if not groups:
        return None
    group = groups[-1][0]                       # nearest to the extremum word
    direction = "descending" if extremum.group("desc") else "ascending"

    aggregate = None
    for name, pattern in _AGGREGATE_WORDS:
        if re.search(pattern, low):
            aggregate = name
            break

    metrics = _columns_in(tail, profile_columns, _NUMERIC_ROLES)
    # "campaign duration": adjacent column words form one phrase; the last word
    # is the measure ("duration"), the first only qualifies it.
    names_in_order = []
    for index, (name, start, end) in enumerate(metrics):
        if index + 1 < len(metrics) and tail[end:metrics[index + 1][1]].strip() == "":
            continue
        names_in_order.append(name)
    metrics_final = list(dict.fromkeys(names_in_order))
    metrics_final = [m for m in metrics_final if m != group]

    if not metrics_final:
        if re.search(rf"\b(?:number of|count of|{_COUNT_NOUNS})\b", tail, flags=re.IGNORECASE):
            return {"kind": "extrema", "group": group, "metrics": [], "aggregate": "count",
                    "direction": direction, "limit": 1}
        return None
    if len(metrics_final) > 1:
        return None  # several metrics: leave to the multi-metric logic
    return {"kind": "extrema", "group": group, "metrics": metrics_final,
            "aggregate": aggregate or "mean", "direction": direction, "limit": 1}



def _unsupported_categorical_filters(
    question: str,
    profile_columns: Dict[str, Dict[str, Any]],
    group_by: List[str],
) -> List[str]:
    """
    Categorical filters ("for customers with a loan", "where contact is cellular")
    are not executable yet. They used to be dropped silently, so the question
    was answered for ALL rows. Report them instead of returning a wrong number.
    """
    issues: List[str] = []
    text = str(question or "")
    categorical = [n for n, i in profile_columns.items() if i.get("role") in _CATEGORICAL_ROLES]

    for name in categorical:
        if name in group_by:
            continue
        equality = re.search(
            rf"(?<!\w){re.escape(name.replace('_', ' '))}(?!\w)\s*(?:is|are|=|==|equals?|equal to|was)\s+['\"]?([\w.\-]+)",
            text, flags=re.IGNORECASE)
        if equality:
            issues.append(
                f"Filtering on a categorical value ('{name} is {equality.group(1)}') is not supported yet; "
                "the question would otherwise be answered for all rows."
            )
    clause = _WITH_CLAUSE.search(text)
    if clause:
        group_spans = [sp for g in group_by for sp in _column_spans(text, g)]
        for name, start, end in _columns_in(text[clause.start():], profile_columns, _CATEGORICAL_ROLES):
            start += clause.start()
            end += clause.start()
            # "housing loans": `loan` only completes the phrase naming the
            # `housing` group; it is not a second, separate filter.
            adjacent = any(
                text[end:g_start].strip() == "" or text[g_end:start].strip() == ""
                for g_start, g_end in group_spans
                if g_start >= end or g_end <= start
            )
            if adjacent:
                continue
            if name not in group_by and not any(name in i for i in issues):
                issues.append(
                    f"Restricting the analysis to customers with/without '{name}' is not supported yet; "
                    f"ask for a comparison by {name} instead."
                )
    return issues


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
    # task analyzer. Semantic group descriptions may not contain the physical
    # dataset column name, so question-only matching can otherwise drop it.
    if isinstance(legacy_analysis, dict):
        legacy_group = legacy_analysis.get("group_column")
        if isinstance(legacy_group, str) and legacy_group in profile_columns:
            legacy_group_info = profile_columns.get(legacy_group, {})
            legacy_group_role = legacy_group_info.get("role")
            legacy_group_is_categorical = legacy_group_role in {
                "categorical", "category", "boolean", "bool"
            }
            legacy_task = str(
                legacy_analysis.get("task_type", "")
            ).casefold()
            legacy_comparison_metric = str(
                legacy_analysis.get("comparison_metric", "")
            ).casefold()
            legacy_group_is_relevant = (
                legacy_task == "comparison"
                or legacy_comparison_metric in {
                    "mean", "median", "sum", "count", "conversion_rate"
                }
                or legacy_analysis.get("analysis_intent")
                == "conversion_factor_analysis"
            )
            if legacy_group_is_categorical and legacy_group_is_relevant:
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
        # Never adopt a grouping column or a non-numeric column as the metric
        # ("average salary by job" once averaged `job`).
        if (
            legacy_metric in profile_columns
            and legacy_metric not in group_by
            and profile_columns[legacy_metric].get("role") in _NUMERIC_ROLES
        ):
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
        for term in ("relationship", "correlation", "correlated", "association", "associated",
                     "outlier", "anomaly", "anomalies", "anomalous", "unusual", "abnormal", "trend")
    ):
        issues.append("No supported analytical operation could be identified confidently.")

    if any(term in question.casefold() for term in ("correlation", "correlated", "relationship", "association", "associated")):
        operations = list(dict.fromkeys(operations + ["correlation"]))

    # Preserve the user's explicitly requested outlier/anomaly method.
    # General anomaly-detection wording defaults to Isolation Forest; ordinary
    # outlier wording defaults to IQR unless another method is explicit.
    question_lower = question.casefold()
    outlier_method = None
    if any(term in question_lower for term in (
        "outlier", "outliers", "anomaly", "anomalies", "anomalous", "unusual", "abnormal",
        "extreme values", "extreme value",
    )):
        operations = list(dict.fromkeys(operations + ["outlier_detection"]))
        if "isolation forest" in question_lower:
            outlier_method = "IsolationForest"
        elif any(term in question_lower for term in ("z-score", "z score", "zscore", "standard score")):
            outlier_method = "ZScore"
        elif any(term in question_lower for term in ("iqr", "interquartile range")):
            outlier_method = "IQR"
        elif any(
            term in question_lower for term in ("detect anomalies", "detect anomaly",
                                                "anomaly detection", "anomalies in",
                                                "find anomalies", "identify anomalies")
        ):
            outlier_method = "IsolationForest"
        else:
            outlier_method = "IQR"

    if any(term in question.casefold() for term in ("trend", "over time", "monthly", "yearly")):
        operations = list(dict.fromkeys(operations + ["time_trend"]))

    if any(term in question.casefold() for term in ("rank", "top ", "bottom ")):
        operations = list(dict.fromkeys(operations + ["rank"]))

    issues.extend(_unsupported_categorical_filters(question, profile_columns, group_by))

    # A numeric aggregate was requested but no numeric column matches the measure.
    _numeric_ops = {"mean", "median", "sum", "min", "max", "std", "variance"}
    _other_ops = {"correlation", "outlier_detection", "time_trend", "describe",
                  "missing_values", "duplicate_count", "count", "nunique", "rank"}
    if (
        set(operations) & _numeric_ops
        and not metrics
        and not (set(operations) & _other_ops)
        and not any(t in question.casefold() for t in ("relationship", "outlier", "anomal", "unusual", "abnormal", "trend"))
    ):
        issues.append(
            "No numeric column in the dataset matches the measure named in the question."
        )

    # "How many unique jobs are there?": the distinct count applies to the NAMED
    # column (categorical columns are never metrics), not to the row count.
    if "nunique" in operations and not metrics:
        named = [
            name for name in mentioned_columns
            if profile_columns.get(name, {}).get("role") != "identifier"
        ]
        if named:
            metrics = named[:3]
            operations = [op for op in operations if op != "count"]
            issues = [i for i in issues if "No supported analytical operation" not in i]

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

    # "campaign duration": adjacent numeric column words form ONE phrase; the last
    # word is the measure and the first only qualifies it.
    if len(metrics) > 1:
        spans = {m: min(_column_spans(question, m) or [(len(question), len(question))]) for m in metrics}
        ordered = sorted(metrics, key=lambda m: spans[m][0])
        keep = []
        for index, name in enumerate(ordered):
            if index + 1 < len(ordered) and question[spans[name][1]:spans[ordered[index + 1]][0]].strip() == "":
                continue
            keep.append(name)
        if keep and len(keep) < len(metrics):
            metrics = keep

    # Grouped ranking questions: the group is named before the extremum word.
    answer_focus = None
    ranking = detect_grouped_ranking(question, profile_columns)
    if ranking is not None:
        group_by = [ranking["group"]]
        metrics = list(ranking["metrics"])
        if ranking["aggregate"] == "count":
            # Plain per-group counts are executed and independently validated;
            # the top/bottom-N selection is applied to those validated counts
            # when the insight is written (see grouped_count_support).
            operations = ["count"]
        else:
            # "highest"/"lowest" describe which GROUP wins; they are not a
            # max/min of the metric, so they must not become operations.
            operations = [ranking["aggregate"]]
        sort_direction = ranking["direction"]
        limit = ranking["limit"]
        issues = [i for i in issues if "No supported analytical operation" not in i]
        mentioned_columns = list(dict.fromkeys(
            mentioned_columns + [ranking["group"]] + list(ranking["metrics"])
        ))
        answer_focus = {
            "type": "group_extrema" if ranking["kind"] == "extrema" else "group_top_n",
            "group": ranking["group"], "metric": (ranking["metrics"] or [None])[0],
            "aggregate": ranking["aggregate"], "direction": ranking["direction"],
            "limit": ranking["limit"],
        }

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
        "outlier_method": outlier_method,
        "identifier_columns": [
            name for name, info in profile_columns.items()
            if info.get("role") == "identifier"
        ],
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
    if answer_focus is not None:
        spec["answer_focus"] = answer_focus

    return spec
