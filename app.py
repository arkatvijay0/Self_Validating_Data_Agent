"""Self-Validating AI-Agent: upload-only, dataset-focused Streamlit application.

No datasets are auto-loaded. The interface shows only the user's uploaded dataset,
analysis controls, visualizations, and validation results. Run: streamlit run app.py
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import re
import sys
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = ROOT / "outputs" / "uploads"
REPORT_PATH = ROOT / "outputs" / "final_eda_report.txt"
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

try:
    import main as pipeline
    PIPELINE_ERROR = None
except Exception as exc:
    pipeline = None
    PIPELINE_ERROR = f"{type(exc).__name__}: {exc}"

st.set_page_config(page_title="Self Validating AI-Agent", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
html,body,[class*="css"] {font-family:'DM Sans',sans-serif;}
[data-testid="stAppViewContainer"] {background:linear-gradient(145deg,#eff7ff 0%,#f9fbff 47%,#f5f1ff 100%);}
.block-container {padding-top:0.15rem!important;max-width:1450px;padding-bottom:2.5rem;}
/* Eliminate the large blank top bar without hiding sidebar controls. */
[data-testid="stHeader"] {height:2.2rem;background:transparent!important;}
[data-testid="stMainBlockContainer"] {padding-top:0.15rem!important;}
[data-testid="stSidebar"] .block-container {padding-top:1.2rem!important;}
/* Keep the title close to the top of the content area. */
h1:first-child {margin-top:0!important;padding-top:0!important;}
[data-testid="stSidebar"] {background:linear-gradient(180deg,#f0faff 0%,#f4f1ff 100%);border-right:1px solid #dce4f7;}
[data-testid="stSidebar"] [data-testid="stFileUploader"] {background:#ffffff;border-radius:14px;padding:6px;}
[data-testid="stMetric"] {background:linear-gradient(145deg,#ffffff 0%,#f7faff 100%);border:1px solid #d8e5f6;border-radius:16px;padding:18px 20px;box-shadow:0 5px 16px rgba(36,67,125,.06);}
[data-testid="stMetricLabel"] {color:#536e8e;font-weight:600;}
[data-testid="stMetricValue"] {color:#153c76;font-weight:750;}
.stButton > button[kind="primary"] {background:linear-gradient(100deg,#007faf,#6459db);border:0;border-radius:10px;color:white;font-weight:700;box-shadow:0 5px 14px rgba(66,81,179,.18);}
.stButton > button[kind="primary"]:hover {background:linear-gradient(100deg,#006f9c,#5146c5);color:white;}
.stButton > button {border:1px solid #cddcf1;border-radius:10px;color:#184a73;font-weight:600;}
[data-testid="stTabs"] button {font-weight:700;color:#4b6886;}
[data-testid="stTabs"] button[aria-selected="true"] {color:#1255b0;border-bottom-color:#4f7cda;}
[data-testid="stAlert"] {border-radius:12px;}
h1 {color:#183e74!important;letter-spacing:-.035em;font-weight:800!important;}
h2,h3 {color:#214b80!important;}
[data-testid="stDataFrame"] {border-radius:12px;overflow:hidden;border:1px solid #dce5f4;}
[data-testid="stTextArea"] textarea {background:#fff;border-radius:12px;border:1px solid #cfdcf0;}

/* Responsive dataset overview cards. Values never use Streamlit's ellipsis layout. */
.dataset-kpi-grid {
    display: grid;
    grid-template-columns: minmax(220px, 1.3fr) repeat(4, minmax(148px, 1fr));
    gap: 14px;
    align-items: stretch;
    margin: 12px 0 20px 0;
}
.dataset-kpi-card {
    background: linear-gradient(145deg, #ffffff 0%, #f7faff 100%);
    border: 1px solid #d8e5f6;
    border-radius: 16px;
    padding: 19px 20px;
    min-width: 0;
    min-height: 124px;
    box-shadow: 0 5px 16px rgba(36,67,125,.06);
}
.dataset-kpi-label {
    color: #536e8e;
    font-size: 0.95rem;
    font-weight: 600;
    margin-bottom: 10px;
    line-height: 1.3;
}
.dataset-kpi-value {
    color: #153c76;
    font-size: clamp(1.6rem, 2.15vw, 2.35rem);
    line-height: 1.2;
    letter-spacing: -0.045em;
    font-weight: 750;
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
}
@media (max-width: 1250px) {
    .dataset-kpi-grid {
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    }
}
@media (max-width: 650px) {
    .dataset-kpi-grid {
        grid-template-columns: repeat(auto-fit, minmax(165px, 1fr));
        gap: 10px;
    }
    .dataset-kpi-card {
        padding: 16px;
        min-height: 108px;
    }
    .dataset-kpi-value {
        font-size: clamp(1.35rem, 5vw, 1.9rem);
    }
}
</style>
""", unsafe_allow_html=True)



def format_indian_number(value: int) -> str:
    """Readable integer formatting: 53211 -> 53,211; 1000000 -> 10,00,000."""
    number = int(value)
    sign = "-" if number < 0 else ""
    digits = str(abs(number))
    if len(digits) <= 3:
        return sign + digits
    trailing = digits[-3:]
    leading = digits[:-3]
    groups = []
    while leading:
        groups.insert(0, leading[-2:])
        leading = leading[:-2]
    return sign + ",".join(groups + [trailing])


def render_dataset_metrics(frame: pd.DataFrame) -> None:
    """Render all five metrics without truncating large row counts."""
    row_count, column_count = frame.shape
    missing_count = int(frame.isna().sum().sum())
    duplicate_count = int(frame.duplicated().sum())
    total_cells = row_count * column_count
    completeness = 100 * (1 - missing_count / total_cells) if total_cells else 0.0

    metrics = [
        ("Rows", format_indian_number(row_count)),
        ("Columns", format_indian_number(column_count)),
        ("Missing cells", format_indian_number(missing_count)),
        ("Duplicate rows", format_indian_number(duplicate_count)),
        ("Completeness", f"{completeness:.1f}%"),
    ]

    cards = "".join(
        '<div class="dataset-kpi-card">'
        f'<div class="dataset-kpi-label">{label}</div>'
        f'<div class="dataset-kpi-value" title="{value}">{value}</div>'
        '</div>'
        for label, value in metrics
    )
    st.markdown(f'<div class="dataset-kpi-grid">{cards}</div>', unsafe_allow_html=True)

def read_upload(file) -> pd.DataFrame:
    content = file.getvalue()
    if not content:
        raise ValueError("Uploaded file is empty")
    name = file.name.lower()
    if name.endswith(".csv"):
        try:
            return pd.read_csv(io.BytesIO(content), low_memory=False)
        except UnicodeDecodeError:
            return pd.read_csv(io.BytesIO(content), encoding="latin-1", low_memory=False)
    return pd.read_excel(io.BytesIO(content))


def report_status(report: str) -> str:
    """Read a declared overall status, not a PASS inside unrelated report text."""
    # The current final_report.py places a standalone decision under section 8.
    match = re.search(
        r"(?ims)^\s*8\.\s+OVERALL RELIABILITY\s*\n\s*-+\s*\n"
        r"\s*(PASS|FAIL|REVIEW REQUIRED|NEEDS CLARIFICATION)\b", report
    )
    if match:
        return match.group(1).upper()
    # Clarification reports use section 2 STATUS instead of section 8.
    match = re.search(
        r"(?ims)^\s*2\.\s+STATUS\s*\n\s*-+\s*\n"
        r"\s*(NEEDS CLARIFICATION|FAIL|PASS|REVIEW REQUIRED)\b", report
    )
    if match:
        return match.group(1).upper()
    # Fallback for older report formats.
    match = re.search(
        r"(?i)OVERALL\s+(?:RESULT|RELIABILITY|STATUS)\s*(?:IS|:|=|-)\s*"
        r"(PASS|FAIL|REVIEW REQUIRED|NEEDS CLARIFICATION)\b", report
    )
    return match.group(1).upper() if match else "NOT CONFIRMED"


def extract_section(report: str, title: str, next_title: str) -> str | None:
    match = re.search(re.escape(title) + r"(.*?)" + re.escape(next_title), report, re.I | re.S)
    return match.group(1).strip("\r\n =-:") if match else None



def get_report_section(report: str, number: int, heading: str) -> str | None:
    """Read one numbered report section without including neighbouring sections."""
    pattern = (
        rf"(?im)^\s*{number}\.\s+{re.escape(heading)}\s*$"
        rf"\s*-{{3,}}\s*\n(.*?)"
        rf"(?=^\s*\d+\.\s+[A-Z][A-Z /&-]+\s*$|\Z)"
    )
    match = re.search(pattern, report, flags=re.MULTILINE | re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else None


def extract_report_results(report: str) -> tuple[list[dict], list[dict]]:
    """Extract only reported analytical values; never invent metrics.

    Handles section-3 scalar results and JSON objects such as
    operation_results -> aggregations. Keeps group labels attached to values.
    Returns (metric/value rows, grouped result rows).
    """
    text = get_report_section(report, 3, "VALIDATED EDA RESULT")
    if not text:
        return [], []

    parsed = {}
    lines = text.splitlines()
    decoder = json.JSONDecoder()
    index = 0
    while index < len(lines):
        line = lines[index]
        match = re.match(r"^([A-Za-z][\w ./%()-]{0,65}):\s*(.*)$", line)
        if not match:
            index += 1
            continue
        key, raw_value = match.group(1).strip(), match.group(2).strip()
        if raw_value.startswith(("{", "[")):
            remaining = "\n".join([raw_value] + lines[index + 1:])
            try:
                value, offset = decoder.raw_decode(remaining)
            except (ValueError, TypeError):
                index += 1
                continue
            index += remaining[:offset].count("\n") + 1
            parsed[key] = value
            continue
        parsed[key] = raw_value
        index += 1

    # The report's compact, JSON-formatted aggregation rows are a display subset
    # when very large. Do not claim that the displayed rows cover every group.
    operation_results = parsed.get("operation_results")
    grouped_rows = []
    if isinstance(operation_results, dict):
        candidate = operation_results.get("aggregations")
        if isinstance(candidate, list):
            grouped_rows = [row for row in candidate if isinstance(row, dict)]
    if not grouped_rows and isinstance(parsed.get("aggregations"), list):
        grouped_rows = [row for row in parsed["aggregations"] if isinstance(row, dict)]

    # Ignore technical metadata, structures and any ungrounded generic prose.
    skip_keys = {
        "status", "analysis_type", "group_by", "operation_results", "aggregations",
        "task_type", "target_column", "reason", "sample_anomalies", "feature_columns",
        "outlier_values", "operation", "group_columns", "columns", "group_sizes",
    }
    scalar = []
    for key, value in parsed.items():
        if key.lower() in skip_keys or isinstance(value, (dict, list)):
            continue
        display = str(value).strip()
        if not display or len(display) > 80:
            continue
        # Only show measured/structured quantities; don't relabel metadata as KPIs.
        if not re.fullmatch(r"[-+]?\d[\d,]*(?:\.\d+)?(?:[Ee][-+]?\d+)?\s*%?", display):
            continue
        scalar.append({"Measure": key.replace("_", " ").strip().capitalize(), "Value": display})
    return scalar[:12], grouped_rows


def present_grouped_results(grouped_rows: list[dict], limit: int = 12) -> tuple[pd.DataFrame | None, int]:
    """Compact table using the group-value pairs from the report itself."""
    if not grouped_rows:
        return None, 0
    columns = []
    for row in grouped_rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    if len(columns) > 7:
        # Retain group labels plus the first analytical measure columns.
        group_cols = [k for k in columns if "__" not in k and k != "row_count"]
        measure_cols = [k for k in columns if k not in group_cols]
        columns = group_cols[:3] + measure_cols[:4]
    view = pd.DataFrame([{k: row.get(k) for k in columns} for row in grouped_rows[:limit]])
    view = view.rename(columns={k: k.replace("__", " · ").replace("_", " ").strip().capitalize() for k in columns})
    return view, len(grouped_rows)


def report_check_status(report: str, section: int, title: str, label: str) -> str:
    section_text = get_report_section(report, section, title)
    if not section_text:
        return "Not reported"
    match = re.search(rf"(?im)^\s*{re.escape(label)}\s*:\s*(PASS|FAIL|REVIEW|SKIPPED|NOT NEEDED)\b", section_text)
    return match.group(1).upper() if match else "Not reported"




# -----------------------------------------------------------------------------
# VISUAL ANSWER SUMMARY
# Charts reflect only the latest report's structured values. Never invent a
# numerator, denominator, category, percentage, or model confidence score.
# -----------------------------------------------------------------------------


def parse_result_object(report: str) -> dict:
    """Parse section 3 of the text report (scalar and multiline JSON fields)."""
    body = get_report_section(report, 3, "VALIDATED EDA RESULT")
    if not body:
        return {}
    parsed = {}
    lines = body.splitlines()
    decoder = json.JSONDecoder()
    index = 0
    while index < len(lines):
        match = re.match(r"^([A-Za-z][\w ./%()-]{0,65}):\s*(.*)$", lines[index])
        if not match:
            index += 1
            continue
        key, raw = match.group(1).strip(), match.group(2).strip()
        if raw.startswith(("{", "[")):
            remaining = "\n".join([raw] + lines[index + 1:])
            try:
                value, offset = decoder.raw_decode(remaining)
            except (ValueError, TypeError):
                index += 1
                continue
            parsed[key] = value
            index += remaining[:offset].count("\n") + 1
        else:
            parsed[key] = raw
            index += 1
    return parsed


def as_number(value) -> float | None:
    """Convert recorded numeric values to finite floats, not textual statements."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip().replace(",", "")
        if cleaned.endswith("%"):
            cleaned = cleaned[:-1].strip()
        if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", cleaned):
            return None
        try:
            value = float(cleaned)
        except ValueError:
            return None
    if not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def numeric_field(result: dict, *names: str) -> float | None:
    for name in names:
        value = as_number(result.get(name))
        if value is not None:
            return value
    return None


def visual_style(fig: go.Figure, *, height: int = 220) -> go.Figure:
    fig.update_layout(
        template="plotly_white", height=height,
        margin=dict(l=15, r=15, t=12, b=24),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial", color="#244466", size=12),
        hoverlabel=dict(bgcolor="#172B4D", font_color="white", font_size=13),
        legend=dict(orientation="h", y=-0.24, x=0, bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def build_split_bar(first_label: str, first: float,
                    second_label: str, second: float) -> go.Figure:
    """Stacked counts (with percentages calculated only from a real denominator)."""
    total = first + second
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[first], y=["Records"], orientation="h", name=first_label,
        marker_color="#FB7185", marker_line_width=0,
        hovertemplate=f"{first_label}: %{{x:,.0f}} records ({first / total * 100:.2f}%)<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=[second], y=["Records"], orientation="h", name=second_label,
        marker_color="#14B8A6", marker_line_width=0,
        hovertemplate=f"{second_label}: %{{x:,.0f}} records ({second / total * 100:.2f}%)<extra></extra>",
    ))
    fig.update_layout(barmode="stack", bargap=0.55, showlegend=True)
    fig.update_xaxes(range=[0, total], showgrid=False, visible=False, fixedrange=True)
    fig.update_yaxes(showgrid=False, visible=False, fixedrange=True)
    return visual_style(fig, height=160)


def reported_correlation(result: dict) -> tuple[str, float] | None:
    """Return one explicitly labelled coefficient; no guessing from arbitrary numbers."""
    matches = []
    def walk(node, context="", depth=0):
        if depth > 6:
            return
        if isinstance(node, dict):
            for key, value in node.items():
                name = str(key).lower()
                new_context = context + "/" + name
                if isinstance(value, (dict, list)):
                    walk(value, new_context, depth + 1)
                else:
                    number = as_number(value)
                    if number is None or not -1 <= number <= 1:
                        continue
                    if any(word in new_context for word in ("pearson", "spearman", "correlation", "corr_coefficient")):
                        matches.append((new_context, number))
        elif isinstance(node, list):
            for element in node:
                walk(element, context, depth + 1)
    for key in ("pearson_r", "spearman_r", "correlation", "correlation_coefficient"):
        value = as_number(result.get(key))
        if value is not None and -1 <= value <= 1:
            return (key.replace("_", " ").title(), value)
    if "operation_results" in result:
        walk(result["operation_results"])
    # Many coefficients need a matrix/table; never select an arbitrary pair.
    unique = {round(value, 9) for _, value in matches}
    if len(unique) == 1 and matches:
        label = "Spearman correlation" if "spearman" in matches[0][0] else "Pearson correlation"
        return label, matches[0][1]
    return None


def group_chart_rows(result: dict, rows: list[dict], question: str):
    """Choose labels and one measured value, preserving genuine group-value pairs."""
    if not rows:
        return None
    groups = result.get("group_by")
    if not isinstance(groups, list):
        groups = []
    observed = [r for r in rows if isinstance(r, dict)]
    if not observed:
        return None
    all_keys = list(dict.fromkeys(k for row in observed for k in row))
    measure_keys = [
        k for k in all_keys
        if k.endswith(("__mean", "__median", "__sum", "__count", "__min", "__max", "__std", "__variance", "__nunique"))
    ]
    if not measure_keys and "row_count" in all_keys:
        measure_keys = ["row_count"]
    if not measure_keys:
        return None
    q = question.lower()
    if "median" in q:
        measure_keys.sort(key=lambda k: "__median" not in k)
    elif "average" in q or "mean" in q:
        measure_keys.sort(key=lambda k: "__mean" not in k)
    elif "count" in q or "how many" in q or "top" in q:
        measure_keys.sort(key=lambda k: not (k.endswith("__count") or k == "row_count"))
    elif "conversion" in q and "rate" in q:
        measure_keys.sort(key=lambda k: not ("conversion" in k and "mean" in k))
    elif "sum" in q or "total" in q:
        measure_keys.sort(key=lambda k: "__sum" not in k)
    key = measure_keys[0]
    if not groups or not all(g in all_keys for g in groups):
        groups = [k for k in all_keys if k not in measure_keys and k != "row_count"][:2]
    if not groups:
        return None
    data = []
    for row in observed:
        value = as_number(row.get(key))
        if value is None or not all(g in row for g in groups):
            continue
        label = " · ".join(f"{g}={row[g]}" for g in groups)
        data.append({"Group": label, "Value": value})
    if not data:
        return None
    data.sort(key=lambda r: r["Value"], reverse=True)
    return (data[:12], key, len(data), len(observed))


def make_answer_visual(report: str, question: str) -> dict | None:
    """Return a chart spec made only from the latest validated-report numbers.

    No artificial percentages and no charts for unsupported/clarification results.
    """
    result = parse_result_object(report)
    if not result:
        return None
    analysis_type = str(result.get("analysis_type", "")).lower()
    # Anomaly/outlier analyses: explicit two-class record counts.
    anomalous = numeric_field(result, "anomaly_count", "outlier_count")
    normal = numeric_field(result, "normal_count")
    total = numeric_field(result, "analyzed_rows", "row_count", "input_rows", "count")
    if anomalous is not None and anomalous >= 0:
        if normal is None and total is not None:
            normal = total - anomalous
        if normal is not None and normal >= 0:
            total_from_counts = anomalous + normal
            if total_from_counts > 0 and (total is None or abs(total-total_from_counts) < 1.1):
                percentage = 100 * anomalous / total_from_counts
                fig = build_split_bar("Flagged", anomalous, "Not flagged", normal)
                return {
                    "title": "Flagged vs unflagged records",
                    "chart": fig,
                    "detail": (f"Flagged: {anomalous:,.0f} ({percentage:.2f}%)  ·  "
                               f"Not flagged: {normal:,.0f} ({100-percentage:.2f}%)"),
                    "note": (
                        "The displayed percentage is the share of records flagged by the method. "
                        "Anomaly flags are not proof of fraud or data errors."
                        + (" Isolation Forest's contamination setting influences this share."
                           if "isolation" in str(result.get("method", "")).lower() else "")
                    ),
                }
    # Class counts that partition the reported total.
    converted = numeric_field(result, "converted_count", "positive_count", "success_count")
    total = numeric_field(result, "total_count", "total_customers", "row_count", "analyzed_rows")
    if converted is not None and total is not None and total > 0 and 0 <= converted <= total:
        return {
            "title": "Outcome share",
            "chart": build_split_bar("Positive", converted, "Other records", total - converted),
            "detail": f"Positive: {converted:,.0f} ({converted / total * 100:.2f}%)  ·  Other: {total-converted:,.0f}",
            "note": "Shares are calculated from the two counts in this report.",
        }
    # Grouped means, counts, and comparisons — actual reported values, no fake percentages.
    groups = []
    ops = result.get("operation_results")
    if isinstance(ops, dict) and isinstance(ops.get("aggregations"), list):
        groups = ops["aggregations"]
    elif isinstance(result.get("aggregations"), list):
        groups = result["aggregations"]
    group_info = group_chart_rows(result, groups, question)
    if group_info:
        entries, measure_key, observed_count, total_rows_in_report = group_info
        view = pd.DataFrame(entries).sort_values("Value", ascending=True)
        fig = px.bar(view, y="Group", x="Value", orientation="h", color="Value",
                     color_continuous_scale=["#06B6D4", "#6366F1", "#EC4899"],
                     custom_data=["Group"])
        fig.update_traces(marker_line_width=0, hovertemplate="%{customdata[0]}<br>Reported value: %{x:,.4g}<extra></extra>")
        fig.update_coloraxes(showscale=False)
        fig.update_yaxes(automargin=True, showgrid=False, title=None)
        fig.update_xaxes(title=measure_key.replace("__", " · ").replace("_", " "),
                         gridcolor="#E1E9F4", zeroline=True, zerolinecolor="#CBD5E1")
        visual_style(fig, height=min(540, max(220, len(entries) * 34 + 105)))
        truncated = any(not isinstance(row, dict) for row in groups)
        note = "Group lengths and values are taken from the reported aggregation; bars are values, not percentages."
        if observed_count > len(entries):
            note += f" Showing {len(entries)} of {observed_count} available numeric groups."
        if truncated:
            note += " The report shortened the original group list; not all groups may appear."
        return {"title": "Group comparison", "chart": fig, "detail": f"Measure: {measure_key.replace('__', ' · ')}", "note": note}
    # Correlation coefficients range from -1 to 1, not 0 to 100%.
    corr = reported_correlation(result)
    if corr is not None:
        label, value = corr
        fig = go.Figure()
        fig.add_trace(go.Bar(x=[value], y=[label], orientation="h",
                             marker_color="#14B8A6" if value >= 0 else "#F97373",
                             width=0.38, text=[f"r = {value:.3f}"], textposition="outside",
                             hovertemplate=f"{label}: {value:.5f}<extra></extra>", showlegend=False))
        fig.update_xaxes(range=[-1.05, 1.05], zeroline=True, zerolinewidth=2,
                         zerolinecolor="#73829A", tickvals=[-1, -.5, 0, .5, 1],
                         gridcolor="#E5EAF4")
        fig.update_yaxes(showgrid=False, title=None)
        visual_style(fig, height=180)
        return {"title": "Correlation strength and direction", "chart": fig,
                "detail": f"{label}: {value:.4f}",
                "note": "The correlation coefficient ranges from −1 to +1; this is not a percentage or proof of causation."}
    # A statistical range may be visualized without inventing a percent denominator.
    lo = numeric_field(result, "min", "minimum")
    hi = numeric_field(result, "max", "maximum")
    mean = numeric_field(result, "mean")
    median = numeric_field(result, "median")
    if lo is not None and hi is not None and hi > lo and (mean is not None or median is not None):
        points = [("Mean", mean, "#4F46E5"), ("Median", median, "#F97316")]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[lo, hi], y=["Observed range", "Observed range"],
                                 mode="lines", line=dict(color="#94A3B8", width=10),
                                 name="Min–max", hovertemplate="Value: %{x:,.4g}<extra></extra>"))
        for name, value, color in points:
            if value is not None and lo <= value <= hi:
                fig.add_trace(go.Scatter(x=[value], y=["Observed range"], mode="markers",
                                         name=name, marker=dict(size=17, color=color, line=dict(color="white", width=2)),
                                         hovertemplate=f"{name}: {value:,.4g}<extra></extra>"))
        fig.update_yaxes(showgrid=False, showticklabels=False, title=None)
        fig.update_xaxes(gridcolor="#EBF0F6", title="Reported value", zeroline=False)
        visual_style(fig, height=170)
        return {"title": "Reported min–max range", "chart": fig,
                "detail": f"Minimum: {lo:,.3g}  ·  Maximum: {hi:,.3g}",
                "note": "The range and markers summarize reported statistics; this is not a full distribution plot."}
    # Explicit percentage/rate, only when the field itself denotes a share.
    for key, raw in result.items():
        lowkey = key.lower()
        if lowkey in ("status", "analysis_type") or not any(x in lowkey for x in ("percentage", "conversion_rate", "success_rate")):
            continue
        rate = as_number(raw)
        if rate is None:
            continue
        if "percentage" in lowkey:
            rate_pct = rate
        else:
            rate_pct = rate * 100 if 0 <= rate <= 1 else rate
        if 0 <= rate_pct <= 100:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=[rate_pct], y=["Rate"], orientation="h", marker_color="#6366F1",
                                 name="Reported share", hovertemplate="Reported share: %{x:.2f}%<extra></extra>"))
            fig.update_xaxes(range=[0, 100], ticksuffix="%", gridcolor="#E5EAF4")
            fig.update_yaxes(showgrid=False, showticklabels=False)
            visual_style(fig, height=140)
            return {"title": lowkey.replace("_", " ").capitalize(), "chart": fig,
                    "detail": f"Reported rate: {rate_pct:.2f}%",
                    "note": "The bar represents an explicitly reported rate, not a fabricated proportion of a count."}
    return None


def get_primary_scalar(result: dict, question: str) -> tuple[str, float] | None:
    """A standalone measured value is still useful without a fabricated bar."""
    keys = [
        ("average", "mean"), ("mean", "mean"), ("median", "median"),
        ("standard deviation", "std"), ("std", "std"),
        ("variance", "variance"), ("minimum", "min"),
        ("maximum", "max"), ("sum", "sum"),
        ("total", "sum"), ("missing", "missing_count"),
        ("duplicate", "duplicate_count"),
        ("count", "count"), ("how many", "count"),
    ]
    q = question.lower()
    preferred = [metric for phrase, metric in keys if phrase in q]
    # No arbitrary metadata (sample_limit, contamination, ID) as analysis KPI.
    preferred.extend(["mean", "median", "sum", "count", "min", "max", "std"])
    for metric in dict.fromkeys(preferred):
        number = numeric_field(result, metric)
        if number is not None:
            return metric.replace("_", " ").title(), number
    return None


def render_answer_visual(report: str, question: str) -> None:
    """Place one optional, relevant visual inside the concise result panel."""
    visual = make_answer_visual(report, question)
    if not visual:
        standalone = get_primary_scalar(parse_result_object(report), question)
        if standalone is not None:
            label, value = standalone
            st.markdown("**Result at a glance**")
            st.metric(label, f"{value:,.4g}" if value != int(value) else f"{int(value):,}")
            st.caption("A single measured value has no meaningful percentage denominator, so no bar is shown.")
        return
    st.markdown("**Visual result summary**")
    st.caption(visual["title"])
    chart_key = "answer_" + hashlib.sha256((question + report).encode("utf-8")).hexdigest()[:16]
    st.plotly_chart(visual["chart"], use_container_width=True, key=chart_key,
                    config={"displaylogo": False, "toImageButtonOptions": {"format": "png", "scale": 2, "filename": "analysis_summary"}})
    st.markdown(visual["detail"])
    st.caption(visual["note"])


def render_result_panel(item: dict) -> None:
    """A brief, readable result snapshot directly below Run analysis."""
    st.markdown("### Analysis results")
    with st.container(border=True):
        header_left, header_right = st.columns([2.5, 1], gap="medium")
        with header_left:
            st.markdown("**Main results from your question**")
            st.caption("Values displayed below come from the latest generated report, not from a separate calculation.")
        with header_right:
            st.metric("Pipeline status", item["status"])

        if item["error"]:
            st.error(f"Execution error: {item['error']}")

        report = item.get("report")
        if not report:
            st.warning("This run did not produce an updated final report. Inspect the execution logs for details.")
            return

        if item["status"] == "NEEDS CLARIFICATION":
            clarification = get_report_section(report, 3, "WHAT IS NEEDED")
            st.warning("The pipeline requested clarification; no analytical results are being presented as validated.")
            if clarification:
                st.write(clarification)
            return

        values, groups = extract_report_results(report)
        group_table, listed_count = present_grouped_results(groups)
        if values:
            # Highlight the question-specific measure before supporting statistics.
            question_lower = str(item.get("query", "")).lower()
            priorities = []
            if any(word in question_lower for word in ("outlier", "anomal", "unusual")):
                priorities = ["outlier count", "lower bound", "upper bound"]
            elif "correlat" in question_lower or "relationship" in question_lower:
                priorities = ["correlation", "pearson", "spearman"]
            elif "conversion" in question_lower or "rate" in question_lower:
                priorities = ["conversion rate", "overall rate", "converted count"]
            elif "average" in question_lower or "mean" in question_lower:
                priorities = ["mean"]
            elif "median" in question_lower:
                priorities = ["median"]
            elif "how many" in question_lower or "count" in question_lower:
                priorities = ["count"]
            def _rank(value_row):
                name = value_row["Measure"].lower()
                return min((i for i, phrase in enumerate(priorities) if phrase in name), default=50)
            values = sorted(values, key=_rank)
            if len(values) > 8:
                st.caption("Showing the eight most relevant reported measurements; additional values remain in the full report.")
            st.dataframe(pd.DataFrame(values[:8]), hide_index=True, use_container_width=True,
                         column_config={"Measure": st.column_config.TextColumn("Measure"),
                                        "Value": st.column_config.TextColumn("Reported value")})
        if group_table is not None:
            st.markdown("**Grouped calculation results**")
            st.dataframe(group_table, hide_index=True, use_container_width=True)
            if listed_count > len(group_table):
                st.caption(f"Showing {len(group_table)} of {listed_count} grouped rows included in the report. Download the report for more detail.")
        if not values and group_table is None:
            st.info("This report contains no compact numeric result table. Review the generated insight and full report below.")

        # A question-aware visualization made from the same validated report.
        # It is omitted when no meaningful ratio or comparison exists.
        if item["status"] == "PASS":
            render_answer_visual(report, str(item.get("query", "")))
        else:
            st.warning("No visual is shown as validated because the pipeline did not report an overall PASS.")

        insight = get_report_section(report, 4, "GENERATED INSIGHT")
        if insight:
            insight = re.split(r"(?m)^RESULT TABLE\s*\(", insight, maxsplit=1)[0].strip("\n -")
            if insight and insight.lower() not in {"none", "n/a"}:
                st.markdown("**Key finding**")
                st.write(insight)

        with st.expander("Validation details", expanded=False):
            st.write("**Insight validation:**", report_check_status(report, 5, "INSIGHT VALIDATION", "Overall Validation"))
            st.write("**Supporting evidence:**", report_check_status(report, 6, "SUPPORTING EVIDENCE", "Evidence Status"))
            st.caption("These are statuses reported by the agent, not an external guarantee of correctness.")

    st.caption("Reported analysis values should be checked against source data when correctness is critical.")


with st.sidebar:
    st.subheader("Upload dataset")
    uploaded = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"],
                                help="Select a dataset to explore and analyze.")
    st.caption("Supported: CSV, XLSX, XLS")
    st.divider()
    with st.expander("Advanced options"):
        show_logs = st.toggle("Show execution logs", value=False)
        inject_fault = st.toggle("Fault-injection test", value=False,
                                 help="Use only to test the pipeline's error recovery behavior.")

frame = None
active_path = None
active_name = None
try:
    if uploaded is not None:
        frame = read_upload(uploaded)
        if frame.empty or not len(frame.columns):
            raise ValueError("Dataset has no usable rows or columns")
        raw_hash = hashlib.sha256(uploaded.getvalue()).hexdigest()[:16]
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        active_path = UPLOAD_DIR / f"uploaded_{raw_hash}.csv"
        if not active_path.is_file():
            frame.to_csv(active_path, index=False)
        active_name = uploaded.name

except Exception as exc:
    st.error(f"Could not load the dataset: {exc}")

st.title("🤖 VeriEDA|AI-Agent")
st.caption("Upload a dataset, explore its columns, and run analytical questions with result validation.")

if frame is None:
    st.info("No dataset selected. Upload a CSV or Excel file in the sidebar to begin.")
else:
    st.write(f"**Current dataset:** {active_name}")
    # Session history must always belong to the dataset currently selected.
    dataset_fingerprint = hashlib.sha256(uploaded.getvalue()).hexdigest() if uploaded is not None else None
    if st.session_state.get("_active_dataset_fingerprint") != dataset_fingerprint:
        st.session_state["_active_dataset_fingerprint"] = dataset_fingerprint
        st.session_state["analysis_runs"] = []
        st.session_state["question"] = ""
    render_dataset_metrics(frame)

    analyze, explore, visualize, history = st.tabs(["Ask a question", "Dataset overview", "Visualize", "Results & validation"])

    with analyze:
        if "question" not in st.session_state:
            st.session_state.question = ""
        st.text_area("Your question", key="question", height=105, placeholder="e.g. Calculate average balance grouped by housing and loan")
        run = st.button("Run analysis", type="primary", use_container_width=True, disabled=pipeline is None)
        if pipeline is None:
            st.error("The analysis pipeline could not be loaded. Confirm that src/main.py and its dependencies are available.")
            with st.expander("Technical details"):
                st.code(PIPELINE_ERROR or "Unknown import error", language="text")
        if run and pipeline is not None:
            query = st.session_state.question.strip()
            if not query:
                st.warning("Enter a question before running the analysis.")
            else:
                before = REPORT_PATH.read_bytes() if REPORT_PATH.exists() else None
                before_mtime = REPORT_PATH.stat().st_mtime_ns if REPORT_PATH.exists() else None
                start = time.perf_counter()
                log = io.StringIO()
                error = None
                with st.spinner("Running analysis and validation..."):
                    try:
                        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                            pipeline.main(dataset_path=str(active_path), user_question=query,
                                          test_fault=inject_fault, dataset_name=active_name)
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {exc}"
                elapsed = time.perf_counter() - start
                after = REPORT_PATH.read_bytes() if REPORT_PATH.exists() else None
                after_mtime = REPORT_PATH.stat().st_mtime_ns if REPORT_PATH.exists() else None
                # Never present a stale report as a newly generated result.
                report_is_new = (after is not None and
                                 (before is None or after != before or after_mtime != before_mtime))
                report = after.decode("utf-8", errors="replace") if report_is_new else None
                # Guard against a report written by a different concurrent request.
                if report:
                    reported_question = get_report_section(report, 1, "USER QUESTION")
                    if reported_question and reported_question.strip() != query:
                        error = "The report did not match this question. Try running it again."
                        report = None
                if error:
                    report = None
                run_data = {"dataset": active_name, "query": query, "elapsed": round(elapsed, 3),
                            "report": report, "status": ("ERROR" if error else report_status(report) if report else "NOT CONFIRMED"),
                            "error": error, "logs": log.getvalue(), "fault_test": inject_fault}
                st.session_state.setdefault("analysis_runs", []).insert(0, run_data)
                st.session_state.analysis_runs = st.session_state.analysis_runs[:20]
        latest = st.session_state.get("analysis_runs", [])
        if latest:
            item = latest[0]
            # The summary sits directly below the Run analysis button.
            render_result_panel(item)
            st.caption(f"Execution time: {item['elapsed']:.2f} seconds")
            if item["report"]:
                rec = get_report_section(item["report"], 7, "RECOMMENDATION")
                if rec and item["status"] != "NEEDS CLARIFICATION":
                    st.markdown("**Recommendation**")
                    # Remove the technical recommendation-status line from the human summary.
                    rec = re.sub(r"^Recommendation Status:.*\n?", "", rec, flags=re.M).strip()
                    if rec:
                        st.write(rec)
                with st.expander("Read complete report (.txt)", expanded=False):
                    st.text(item["report"])
                st.download_button("Download complete report (.txt)", item["report"],
                                   "eda_report.txt", "text/plain", use_container_width=True)
            if item["fault_test"]:
                st.info("A fault-injection test was requested. Inspect the report to confirm whether correction occurred.")
            if show_logs:
                with st.expander("Execution logs", expanded=True):
                    st.code(item["logs"][-15000:] or "No console output", language="text")

    with explore:
        # Show column-level profiling only, not raw record previews.
        # This prevents question text in benchmark CSV files from appearing
        # as if it were the underlying customer dataset.
        st.subheader("Dataset overview")
        schema = pd.DataFrame({
            "Column": list(frame.columns),
            "Data type": [str(frame[c].dtype) for c in frame.columns],
            "Non-null": [int(frame[c].notna().sum()) for c in frame.columns],
            "Missing": [int(frame[c].isna().sum()) for c in frame.columns],
            "Unique values": [int(frame[c].nunique(dropna=True)) for c in frame.columns],
        })
        st.subheader("Column details")
        st.dataframe(schema, use_container_width=True, hide_index=True)
        st.subheader("Numeric summary")
        numbers = frame.select_dtypes(include="number")
        if not numbers.empty:
            st.dataframe(numbers.describe().T, use_container_width=True)
        else:
            st.info("No numeric columns detected.")

    with visualize:
        st.subheader("Interactive visualizations")
        st.caption("Explore the uploaded dataset with customizable colors, chart types, and hover details.")

        numeric = frame.select_dtypes(include="number").columns.tolist()
        categorical = [c for c in frame.columns if c not in numeric]

        palette_names = {
            "Vibrant": ["#6C3CE9", "#06B6D4", "#FB7185", "#F59E0B", "#10B981", "#3B82F6", "#EC4899", "#8B5CF6"],
            "Ocean": ["#0891B2", "#2563EB", "#14B8A6", "#0EA5E9", "#6366F1", "#22D3EE"],
            "Sunset": ["#F97316", "#F43F5E", "#EAB308", "#D946EF", "#FB7185", "#EA580C"],
            "Emerald": ["#059669", "#10B981", "#84CC16", "#14B8A6", "#0D9488", "#22C55E"],
        }
        graph_types = ["Distribution", "Category counts", "Scatter plot", "Correlation heatmap",
                       "Missing values", "Box plot", "Violin plot", "Category share", "Grouped comparison"]
        col_type, col_palette = st.columns([2, 1])
        with col_type:
            chart_type = st.selectbox("Chart type", graph_types)
        with col_palette:
            palette_name = st.selectbox("Color palette", list(palette_names))
        palette = palette_names[palette_name]
        color_scale = [palette[i % len(palette)] for i in range(max(2, len(palette)))]

        def style_chart(fig, title: str, height: int = 470):
            fig.update_layout(
                title={"text": title, "font": {"size": 21, "color": "#173E74", "family": "Arial"}, "x": 0.02},
                template="plotly_white", height=height,
                font={"family": "Arial", "color": "#334155", "size": 12},
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#F8FAFF",
                margin=dict(l=44, r=25, t=78, b=55),
                hoverlabel=dict(bgcolor="#172B4D", font_color="white", font_size=13),
                legend=dict(title="", bgcolor="rgba(255,255,255,.9)", orientation="h", y=-0.23),
                colorway=palette,
            )
            fig.update_xaxes(showline=True, linecolor="#CFD9EE", showgrid=False,
                             ticks="outside", tickcolor="#CBD5E1", zeroline=False)
            fig.update_yaxes(showline=False, gridcolor="#E3EAF7", griddash="dot", zeroline=False)
            return fig

        plot_config = {"displaylogo": False, "responsive": True,
                       "toImageButtonOptions": {"format": "png", "scale": 2, "filename": "eda_visualization"}}
        fig = None
        note = None
        if chart_type == "Distribution":
            if numeric:
                column = st.selectbox("Numeric column", numeric)
                bins = st.slider("Number of bins", min_value=10, max_value=100, value=35, step=5)
                values = frame[[column]].dropna()
                if len(values) > 100000:
                    values = values.sample(n=100000, random_state=42)
                    note = "Histogram preview uses 100,000 sampled records for browser performance."
                fig = px.histogram(values, x=column, nbins=bins, marginal="box",
                                   color_discrete_sequence=[palette[0]], opacity=0.9)
                fig.update_traces(marker_line_color="rgba(255,255,255,.8)", marker_line_width=0.55,
                                  selector=dict(type="histogram"))
                fig = style_chart(fig, f"Distribution of {column}")
            else:
                st.info("No numeric columns found.")

        elif chart_type == "Category counts":
            if categorical:
                column = st.selectbox("Category column", categorical)
                limit = st.slider("Show top categories", min_value=3, max_value=30, value=12)
                counts = frame[column].fillna("(Missing)").astype(str).value_counts().head(limit)
                view = counts.rename_axis("Category").reset_index(name="Count")
                fig = px.bar(view, x="Count", y="Category", orientation="h", color="Category",
                             color_discrete_sequence=palette, text="Count")
                fig.update_traces(textposition="outside", cliponaxis=False, marker_line_width=0)
                fig.update_layout(showlegend=False, yaxis=dict(autorange="reversed"))
                fig = style_chart(fig, f"Most common values · {column}", max(410, 36 * len(view) + 160))
            else:
                st.info("No categorical columns found.")

        elif chart_type == "Scatter plot":
            if len(numeric) >= 2:
                a, b, c = st.columns(3)
                with a:
                    x = st.selectbox("X axis", numeric)
                with b:
                    y = st.selectbox("Y axis", numeric, index=1)
                with c:
                    group = st.selectbox("Color by", ["None"] + categorical)
                fields = list(dict.fromkeys([x, y] + ([] if group == "None" else [group])))
                view = frame[fields].dropna()
                if len(view) > 3500:
                    view = view.sample(n=3500, random_state=42)
                    note = "Scatter plot displays a reproducible 3,500-row sample."
                fig = px.scatter(view, x=x, y=y, color=group if group != "None" else None,
                                 color_discrete_sequence=palette, opacity=0.72,
                                 hover_data=fields)
                fig.update_traces(marker=dict(size=8, line=dict(width=.6, color="rgba(255,255,255,.8)")))
                fig = style_chart(fig, f"{x} vs {y}")
            else:
                st.info("At least two numeric columns are required.")

        elif chart_type == "Correlation heatmap":
            if len(numeric) >= 2:
                selected = st.multiselect("Numeric columns", numeric, default=numeric[:min(12, len(numeric))])
                if len(selected) >= 2:
                    correlation = frame[selected].corr()
                    fig = px.imshow(correlation, text_auto=".2f", aspect="auto", zmin=-1, zmax=1,
                                    color_continuous_scale="RdBu_r", labels=dict(color="Pearson r"))
                    fig = style_chart(fig, "Correlation between numeric features", max(430, len(selected) * 34 + 180))
                    fig.update_xaxes(showgrid=False)
                    fig.update_yaxes(showgrid=False)
                else:
                    st.info("Select at least two columns.")
            else:
                st.info("At least two numeric columns are required.")

        elif chart_type == "Missing values":
            counts = frame.isna().sum().sort_values(ascending=False)
            view = counts.rename_axis("Column").reset_index(name="Missing cells")
            view["Missing %"] = (100 * view["Missing cells"] / len(frame)).round(2)
            fig = px.bar(view, x="Column", y="Missing cells", color="Missing %", text="Missing cells",
                         color_continuous_scale=["#06B6D4", "#6366F1", "#F43F5E"],
                         hover_data={"Missing %": ":.2f"})
            fig.update_traces(textposition="outside")
            fig = style_chart(fig, "Missing data by column")
            if int(counts.sum()) == 0:
                note = "No missing cells were found in this dataset."

        elif chart_type in ("Box plot", "Violin plot"):
            if numeric:
                a, b = st.columns(2)
                with a:
                    measure = st.selectbox("Numeric column", numeric)
                with b:
                    group = st.selectbox("Group by (optional)", ["None"] + categorical)
                fields = [measure] if group == "None" else [measure, group]
                view = frame[fields].dropna()
                if len(view) > 12000:
                    view = view.sample(n=12000, random_state=42)
                    note = "Distribution preview uses a reproducible 12,000-row sample."
                if group != "None" and view[group].nunique() > 20:
                    top = view[group].value_counts().head(15).index
                    view = view[view[group].isin(top)]
                    note = (note or "") + " Displaying the 15 most frequent groups."
                kwargs = dict(x=group if group != "None" else None, y=measure,
                              color=group if group != "None" else None,
                              color_discrete_sequence=palette)
                if chart_type == "Violin plot":
                    fig = px.violin(view, box=True, points=False, **kwargs)
                else:
                    fig = px.box(view, points=False, **kwargs)
                fig = style_chart(fig, f"{chart_type} · {measure}")
                fig.update_layout(showlegend=False)
            else:
                st.info("No numeric columns found.")

        elif chart_type == "Category share":
            if categorical:
                column = st.selectbox("Category column", categorical)
                counts = frame[column].fillna("(Missing)").astype(str).value_counts().head(8)
                fig = px.pie(names=counts.index.tolist(), values=counts.values, hole=.56,
                             color_discrete_sequence=palette)
                fig.update_traces(textinfo="percent+label", textposition="outside",
                                  marker=dict(line=dict(color="white", width=2)))
                fig = style_chart(fig, f"Category share · {column}", height=520)
                if frame[column].nunique(dropna=False) > 8:
                    note = "Only the eight largest categories are plotted; their percentages are relative to the plotted subset."
            else:
                st.info("No categorical columns found.")

        elif chart_type == "Grouped comparison":
            if numeric and categorical:
                a, b, c = st.columns(3)
                with a:
                    group = st.selectbox("Group column", categorical)
                with b:
                    measure = st.selectbox("Value column", numeric)
                with c:
                    agg = st.selectbox("Calculation", ["Mean", "Median", "Sum", "Count"])
                if agg == "Count":
                    result = frame.groupby(group, dropna=False).size().reset_index(name="Value")
                else:
                    method = {"Mean": "mean", "Median": "median", "Sum": "sum"}[agg]
                    result = frame.groupby(group, dropna=False)[measure].agg(method).reset_index(name="Value")
                result[group] = result[group].astype(str)
                result = result.sort_values("Value", ascending=False).head(20)
                fig = px.bar(result, x=group, y="Value", color=group,
                             color_discrete_sequence=palette, text_auto=".3s")
                fig.update_layout(showlegend=False)
                fig = style_chart(fig, f"{agg} of {measure} by {group}")
                if frame[group].nunique(dropna=False) > 20:
                    note = "Showing the 20 groups with the highest values."
            else:
                st.info("A numeric and a categorical column are required.")

        if fig is not None:
            st.plotly_chart(fig, use_container_width=True, config=plot_config)
            if note:
                st.caption(note)
            st.caption("Interactive chart · Hover for values, zoom, select, or use the camera icon to export a PNG.")
        st.caption("Charts visualize the uploaded data; they are separate from the agent's validated analytical conclusions.")

    with history:
        st.subheader("Analysis history")
        runs = st.session_state.get("analysis_runs", [])
        if not runs:
            st.info("No analyses have been run during this session.")
        else:
            rows = [{"Dataset": run["dataset"], "Question": run["query"], "Status": run["status"],
                     "Runtime (s)": run["elapsed"], "Execution error": "Yes" if run["error"] else "No"} for run in runs]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.download_button("Download run history (CSV)", pd.DataFrame(rows).to_csv(index=False).encode(),
                               "analysis_history.csv", "text/csv")
            st.caption("History is stored only in the current Streamlit session.")
        st.subheader("Validation details")
        st.write("Open the full report for the latest question to inspect available code, result, insight, and evidence checks. "
                 "A PASS status indicates only the checks reported by the pipeline, not guaranteed analytical correctness.")
