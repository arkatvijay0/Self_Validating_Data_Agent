
from pathlib import Path
import ast
import re

APP_PATH = Path("app.py")

if not APP_PATH.exists():
    raise FileNotFoundError(
        "app.py was not found. Run this script from your project root."
    )

source = APP_PATH.read_text(encoding="utf-8")

if "def answer_simple_numeric_question(" in source:
    print("Fast numeric analysis is already installed. No changes made.")
    raise SystemExit(0)

anchor = "\ndef answer_simple_data_quality_question("
if anchor not in source:
    raise RuntimeError(
        "Could not find the expected data-quality helper. "
        "No changes made; restore the expected app.py version first."
    )

helper = r'''

def answer_simple_numeric_question(text: str, frame: pd.DataFrame) -> str | None:
    """Answer ungrouped scalar numeric questions without running the full agent."""
    q = text.strip().lower()

    # Do not intercept grouped, ranking, or comparison questions.
    grouped_terms = (
        " by ", " per ", " for each ", " group by ", " grouped by ",
        "which job", "which category", "which region", "compare ",
        "correlation", "relationship", "trend", "over time",
        "isolation forest", "anomal", "outlier", "predict ",
    )
    padded_q = f" {q} "
    if any(term in padded_q for term in grouped_terms):
        return None

    # Determine the requested scalar operation.
    operation_patterns = [
        ("maximum", r"\b(maximum|max|highest|largest|greatest)\b"),
        ("minimum", r"\b(minimum|min|lowest|smallest|least)\b"),
        ("average", r"\b(average|mean)\b"),
        ("median", r"\bmedian\b"),
        ("sum", r"\b(sum|total)\b"),
        ("count", r"\b(count|how many)\b"),
        ("range", r"\brange\b"),
        ("standard deviation", r"\b(standard deviation|std dev|std)\b"),
    ]

    operation = next(
        (name for name, pattern in operation_patterns
         if re.search(pattern, q)),
        None,
    )
    if operation is None:
        return None

    # Identify a real column by matching its full name, case-insensitively.
    # Longer names are tested first to avoid partial matches.
    columns = list(frame.columns)
    matching_columns = sorted(
        [
            col for col in columns
            if re.search(
                r"(?<!\w)" + re.escape(str(col).strip().lower()) + r"(?!\w)",
                q,
            )
        ],
        key=lambda col: len(str(col)),
        reverse=True,
    )

    if not matching_columns:
        # Support common phrasing such as "average balance" only when the
        # requested column name itself appears in the question.
        return None

    column = matching_columns[0]

    if operation == "count":
        # A named-column count counts non-missing values in that column.
        valid_count = int(frame[column].notna().sum())
        return (
            f"Non-missing values in '{column}': {valid_count:,}.\n"
            f"Dataset rows: {len(frame):,}.\n"
            "Method: direct pandas count on the uploaded dataset. "
            "The full self-validation pipeline was not run."
        )

    values = pd.to_numeric(frame[column], errors="coerce")
    values = values.dropna()
    valid_count = int(values.size)

    if valid_count == 0:
        return (
            f"No valid numeric observations were found in '{column}'. "
            "No numeric result was calculated."
        )

    if operation == "maximum":
        value = float(values.max())
        label = "Maximum"
    elif operation == "minimum":
        value = float(values.min())
        label = "Minimum"
    elif operation == "average":
        value = float(values.mean())
        label = "Average"
    elif operation == "median":
        value = float(values.median())
        label = "Median"
    elif operation == "sum":
        value = float(values.sum())
        label = "Total"
    elif operation == "range":
        value = float(values.max() - values.min())
        label = "Range (maximum minus minimum)"
    else:
        value = float(values.std(ddof=1)) if valid_count > 1 else 0.0
        label = "Sample standard deviation"

    if not (value == value and abs(value) != float("inf")):
        return f"The calculation for '{column}' did not produce a finite result."

    return (
        f"{label} of '{column}': {value:,.2f}\n"
        f"Valid numeric observations: {valid_count:,} of {len(frame):,} rows.\n"
        "Method: direct pandas calculation using valid numeric values; "
        "non-numeric and missing values were excluded. "
        "The full self-validation pipeline was not run."
    )
'''

source = source.replace(anchor, helper + anchor, 1)

quick_anchor = (
    "    quick_answer = answer_simple_data_quality_question(question, preview_df)\n"
    "    if quick_answer and not test_fault:\n"
)
if quick_anchor not in source:
    raise RuntimeError(
        "Could not find the expected fast-path call site. No changes made."
    )

replacement = (
    "    quick_answer = answer_simple_data_quality_question(question, preview_df)\n"
    "    if quick_answer is None:\n"
    "        quick_answer = answer_simple_numeric_question(question, preview_df)\n"
    "    if quick_answer and not test_fault:\n"
)

source = source.replace(quick_anchor, replacement, 1)

# Verify syntax before replacing the original file.
ast.parse(source, filename=str(APP_PATH))

# Keep a second safety backup if the first one exists.
backup = Path("app.py.fast_analysis_patch.bak")
backup.write_text(APP_PATH.read_text(encoding="utf-8"), encoding="utf-8")

APP_PATH.write_text(source, encoding="utf-8")
print("Updated app.py successfully.")
print("Backup saved as app.py.fast_analysis_patch.bak")
