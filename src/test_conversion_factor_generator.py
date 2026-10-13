
from pathlib import Path
import json
import sys

import pandas as pd


# Resolve the project root from this file's location.
PROJECT_ROOT = Path(__file__).resolve().parent

if PROJECT_ROOT.name.lower() == "src":
    PROJECT_ROOT = PROJECT_ROOT.parent

SRC_DIR = PROJECT_ROOT / "src"

# Ensure the project source directory is importable.
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from code_generator import generate_code_mock


DATASET_CANDIDATES = [
    PROJECT_ROOT / "outputs" / "uploads" / "bank_marketing_cleaned_uploaded.csv",
    PROJECT_ROOT / "outputs" / "uploads" / "active_uploaded_dataset.csv",
]

DATASET_PATH = None

for candidate in DATASET_CANDIDATES:
    if not candidate.exists():
        continue

    columns = set(pd.read_csv(candidate, nrows=0).columns)

    if "y" in columns or "conversion" in columns:
        DATASET_PATH = candidate
        break


if DATASET_PATH is None:
    print("ERROR: Could not find a suitable conversion dataset.")

    for candidate in DATASET_CANDIDATES:
        print(f"Checked: {candidate}")
        print(f"Exists: {candidate.exists()}")

    raise SystemExit(1)


QUESTION = (
    "Summarize the important patterns in this dataset "
    "and identify the factors associated with customer conversion."
)


print("=" * 70)
print("CONVERSION-FACTOR GENERATOR TEST")
print("=" * 70)
print("Dataset:", DATASET_PATH)

dataset = pd.read_csv(DATASET_PATH)

print("Dataset shape:", dataset.shape)
print("Dataset columns:", dataset.columns.tolist())


generated_code = generate_code_mock(
    step_text=QUESTION,
    filepath=str(DATASET_PATH.resolve()),
)

try:
    compile(
        generated_code,
        "<generated_conversion_analysis>",
        "exec",
    )
except SyntaxError as exc:
    print("\nFAIL: Generated code has a syntax error.")
    print(exc)
    raise SystemExit(1)

print("\nGenerated code syntax: PASS")


namespace = {}

try:
    exec(generated_code, namespace)
except Exception as exc:
    print("\nFAIL: Generated code execution failed.")
    print(f"{type(exc).__name__}: {exc}")
    raise


result = namespace.get("result")

if not isinstance(result, dict):
    print("\nFAIL: Expected a dictionary result.")
    print("Actual result type:", type(result).__name__)
    print(repr(result)[:3000])
    raise SystemExit(1)


summary = {
    "status": result.get("status"),
    "analysis_type": result.get("analysis_type"),
    "target_column": result.get("target_column"),
    "total_dataset_records": result.get("total_dataset_records"),
    "valid_target_records": result.get("valid_target_records"),
    "converted_count": result.get("converted_count"),
    "overall_conversion_rate_percent":
        result.get("overall_conversion_rate_percent"),
    "categorical_factor_columns":
        list(result.get("categorical_factor_results", {}).keys()),
    "numeric_factor_columns":
        list(result.get("numeric_factor_results", {}).keys()),
}

print("\nResult summary:")
print(json.dumps(summary, indent=2, default=str))


if result.get("analysis_type") != "conversion_factor_analysis":
    print("\nFAIL: Conversion-factor analysis was not selected.")
    raise SystemExit(1)

if result.get("status") != "PASS":
    print("\nFAIL: Analysis returned a failure status.")
    print(json.dumps(result, indent=2, default=str)[:5000])
    raise SystemExit(1)


valid_rows = result.get("valid_target_records")
converted_count = result.get("converted_count")
reported_rate = result.get("overall_conversion_rate_percent")

if valid_rows != len(dataset):
    print(
        "\nFAIL: Valid target row count differs from dataset row count:",
        valid_rows,
        "versus",
        len(dataset),
    )
    raise SystemExit(1)


expected_rate = (
    round(converted_count / valid_rows * 100, 2)
    if valid_rows
    else 0
)

if reported_rate != expected_rate:
    print(
        "\nFAIL: Conversion rate mismatch.",
        "Expected:",
        expected_rate,
        "Reported:",
        reported_rate,
    )
    raise SystemExit(1)


if not result.get("categorical_factor_results"):
    print("\nFAIL: No categorical factor tables were produced.")
    raise SystemExit(1)


print("\nPASS: Conversion-factor generator test succeeded.")
print("Verified overall conversion rate:", f"{reported_rate}%")


print("\nSample categorical factor results:")

for column, groups in list(
    result["categorical_factor_results"].items()
)[:3]:
    print(f"\n{column}:")
    print(json.dumps(groups[:5], indent=2, default=str))
