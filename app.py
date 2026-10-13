"""Streamlit interface for the Self-Validating EDA Agent.

Supports CSV and Excel uploads. Excel files are converted to CSV for the
existing pipeline, whose modules currently load data using pandas.read_csv.
"""
from __future__ import annotations

import contextlib
import io
import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
UPLOAD_DIR = OUTPUT_DIR / "uploads"
REPORT_PATH = OUTPUT_DIR / "final_eda_report.txt"

SRC_DIR_STR = str(SRC_DIR)
if SRC_DIR_STR not in sys.path:
    sys.path.insert(0, SRC_DIR_STR)

try:
    import main as pipeline
except Exception as exc:
    pipeline = None
    PIPELINE_IMPORT_ERROR = exc
else:
    PIPELINE_IMPORT_ERROR = None

st.set_page_config(
    page_title="Self-Validating EDA Agent",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Self-Validating EDA Agent")
st.caption("Upload a dataset, preview it, ask a question, and inspect the validation report.")

with st.sidebar:
    st.header("How to use")
    st.markdown(
        "1. Upload a CSV or Excel file.\n"
        "2. Review the data preview and columns.\n"
        "3. Enter one question about the data.\n"
        "4. Select **Analyze Dataset** and review the report and pipeline logs."
    )
    st.info("The analysis is descriptive. Correlation or grouped results do not establish causation.")

uploaded_file = st.file_uploader(
    "Upload dataset (CSV or Excel)",
    type=["csv", "xlsx", "xls"],
    help="The uploaded file is used for this run; the default sales.csv is not used when a file is uploaded.",
)

if uploaded_file is None:
    st.info("Upload a CSV or Excel dataset to begin. The pipeline's default sales.csv is retained for command-line runs.")
    st.stop()

try:
    file_bytes = uploaded_file.getvalue()
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".csv":
        try:
            preview_df = pd.read_csv(io.BytesIO(file_bytes), low_memory=False)
        except UnicodeDecodeError:
            preview_df = pd.read_csv(io.BytesIO(file_bytes), encoding="latin-1", low_memory=False)
    else:
        preview_df = pd.read_excel(io.BytesIO(file_bytes))
except Exception as exc:
    st.error(f"Could not read the uploaded file: {exc}")
    st.stop()

if preview_df.empty:
    st.error("The uploaded dataset has no rows. Please upload a non-empty file.")
    st.stop()

st.subheader("Dataset preview")
metric_cols = st.columns(3)
metric_cols[0].metric("Rows", f"{preview_df.shape[0]:,}")
metric_cols[1].metric("Columns", f"{preview_df.shape[1]:,}")
metric_cols[2].metric("Missing cells", f"{int(preview_df.isna().sum().sum()):,}")
st.dataframe(preview_df.head(20), use_container_width=True)
with st.expander("View column names and data types"):
    st.dataframe(
        pd.DataFrame({"Column": preview_df.columns.astype(str), "Data type": preview_df.dtypes.astype(str).values}),
        use_container_width=True,
        hide_index=True,
    )

question = st.text_input(
    "What do you want to know about this dataset?",
    placeholder="e.g., Which campaign had the highest conversion rate?",
)

# General calendar fact questions do not require dataset execution. Handle the
# common May-duration wording without presenting it as a dataset-derived result.
def answer_calendar_question(text: str) -> str | None:
    q = text.lower().strip()
    month_may = bool(re.search(r"\bmay\b", q))
    asks_duration = bool(re.search(r"\b(duration|how many days|number of days|days in|length of the month)\b", q))
    if month_may and asks_duration:
        return (
            "May has 31 days. This is a calendar fact, not a result calculated from the uploaded dataset. "
            "If you meant the duration between dates in May contained in your dataset, ask for that explicitly "
            "and specify a year if needed."
        )
    return None


def answer_simple_data_quality_question(text: str, frame: pd.DataFrame) -> str | None:
    """Answer basic deterministic dataset checks without running the full EDA pipeline."""
    q = text.lower().strip()

    asks_missing = bool(re.search(r"\b(missing|null|na values|nan values)\b", q)) and bool(
        re.search(r"\b(value|values|cell|cells|data|count|present|many|column|columns)\b", q)
    )
    asks_duplicates = bool(re.search(r"\b(duplicate|duplicates|duplicated)\b", q))
    asks_rows = bool(re.search(r"\b(row|rows|records|observations)\b", q)) and bool(
        re.search(r"\b(how many|count|number|total|size)\b", q)
    )
    asks_columns = bool(re.search(r"\b(column|columns|features|variables)\b", q)) and bool(
        re.search(r"\b(how many|count|number|total)\b", q)
    )

    if asks_missing:
        missing_by_column = frame.isna().sum()
        total_missing = int(missing_by_column.sum())
        lines = [f"Total missing cells: {total_missing:,}."]
        if total_missing == 0:
            lines.append(f"Checked {len(frame):,} rows across {len(frame.columns):,} columns; no null/NaN cells were found.")
        else:
            lines.append("Columns with missing values:")
            for column, count in missing_by_column[missing_by_column > 0].items():
                pct = (int(count) / max(len(frame), 1)) * 100
                lines.append(f"- {column}: {int(count):,} ({pct:.2f}% of rows)")
        lines.append("Method: direct check of the uploaded dataset using pandas isna().sum(). This quick check does not run the full self-validation pipeline.")
        return "\n".join(lines)

    if asks_duplicates:
        count = int(frame.duplicated().sum())
        return (
            f"Exact duplicate rows: {count:,}.\n"
            f"Checked all {len(frame):,} rows across {len(frame.columns):,} columns.\n"
            "Method: direct full-row duplicate check on the uploaded dataset; the full EDA pipeline was not run."
        )

    if asks_rows and not asks_columns:
        return f"The uploaded dataset contains {len(frame):,} rows (records). This is a direct dataset check; the full EDA pipeline was not run."

    if asks_columns:
        return f"The uploaded dataset contains {len(frame.columns):,} columns (variables).\nColumns: {', '.join(map(str, frame.columns))}.\nThis is a direct dataset check; the full EDA pipeline was not run."

    return None

if st.button("🚀 Analyze Dataset", type="primary", use_container_width=True):
    if not question.strip():
        st.warning("Please enter an analysis question first.")
        st.stop()

    calendar_answer = answer_calendar_question(question)
    if calendar_answer:
        st.subheader("Answer")
        st.success(calendar_answer)
        st.caption("The EDA pipeline was not run for this general calendar question.")
        st.stop()

    quick_answer = answer_simple_data_quality_question(question, preview_df)
    if quick_answer:
        st.subheader("Quick dataset check")
        st.info(quick_answer)
        st.caption(
            "This fast path avoids the slower multi-step EDA pipeline "
            "for basic data-quality questions."
        )
        st.stop()

    if pipeline is None:
        st.error(f"Could not import the analysis pipeline: {PIPELINE_IMPORT_ERROR}")
        st.stop()

    try:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Save the uploaded dataset as CSV for the existing pipeline.
        dataset_path = UPLOAD_DIR / "active_uploaded_dataset.csv"
        if suffix == ".csv":
            dataset_path.write_bytes(file_bytes)
        else:
            preview_df.to_csv(dataset_path, index=False)

        # Remove any previous report so it cannot be mistaken for this run.
        if REPORT_PATH.exists():
            REPORT_PATH.unlink()

        log_buffer = io.StringIO()
        with st.spinner("Profiling, planning, executing, and validating the analysis..."):
            with contextlib.redirect_stdout(log_buffer), contextlib.redirect_stderr(log_buffer):
                pipeline.main(
                    dataset_path=str(dataset_path),
                    user_question=question,
                )

        # Capture output without displaying a green completion banner.
        logs = log_buffer.getvalue()

        with st.expander("View pipeline output", expanded=False):
            st.text(logs[-30000:] if logs else "No console output was captured.")

        if REPORT_PATH.exists():
            report_text = REPORT_PATH.read_text(
                encoding="utf-8",
                errors="replace",
            )

            st.subheader("Final analysis report")
            st.text(report_text)

            st.download_button(
                "Download report (.txt)",
                data=report_text,
                file_name="final_eda_report.txt",
                mime="text/plain",
            )

            if (
                "Overall reliability\n------------------------------------------------------------\nPASS"
                in report_text
                or "Overall reliability: PASS" in report_text
            ):
                st.success("Overall reliability: PASS")
            elif "Overall reliability" in report_text and "FAIL" in report_text:
                st.warning(
                    "The report indicates a validation/reliability failure. "
                    "Review the insight and question relevance checks."
                )
        else:
            st.warning(
                "The pipeline finished without creating a final report. "
                "Check the pipeline output for details."
            )

    except Exception as exc:
        st.error(f"Analysis failed: {type(exc).__name__}: {exc}")
        st.exception(exc)
