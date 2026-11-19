"""Dataset-focused Streamlit interface for the existing Self-Validating EDA Agent.
Save as app.py in the project root and run: streamlit run app.py
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
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

st.set_page_config(page_title="AI-Agent", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")
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
</style>
""", unsafe_allow_html=True)


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
    """Only report an explicit overall status; do not infer from incidental PASS text."""
    text = report.upper()
    candidates = [
        (r"OVERALL\s+(?:RESULT|RELIABILITY)(?:\s+STATUS)?\s*(?:IS|:|=|-)?\s*(PASS|FAIL|REVIEW REQUIRED|NEEDS CLARIFICATION)", None),
        (r"OVERALL\s+STATUS\s*(?:IS|:|=|-)?\s*(PASS|FAIL|REVIEW REQUIRED|NEEDS CLARIFICATION)", None),
    ]
    for pattern, _ in candidates:
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1]
    if "NEEDS CLARIFICATION" in text and "WHAT IS NEEDED" in text:
        return "NEEDS CLARIFICATION"
    return "NOT CONFIRMED"


def extract_section(report: str, title: str, next_title: str) -> str | None:
    match = re.search(re.escape(title) + r"(.*?)" + re.escape(next_title), report, re.I | re.S)
    return match.group(1).strip("\r\n =-:") if match else None


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

    st.divider()
    st.subheader("🏆 Review-II Benchmark")
    st.markdown(
        """
        <div style="font-size:0.85rem; line-height:1.5; color:#334155; margin-bottom:8px;">
        <b>Evaluation vs. Base Paper (DS-STAR):</b><br/>
        • <b>Overall Accuracy</b>: <b>93.11%</b> (DS-STAR: 52.0%)<br/>
        • <b>Hard Tasks Acc</b>: <b>98.00%</b> (DS-STAR: 45.2%)<br/>
        • <b>Precision</b>: <b>99.75%</b> (Zero Hallucination)<br/>
        • <b>ROC-AUC Score</b>: <b>0.9965</b>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    pdf_path = ROOT / "assets" / "benchmark_450_questions_evaluation_report.pdf"
    if not pdf_path.exists():
        pdf_path = ROOT / "outputs" / "benchmark_450_questions_evaluation_report.pdf"
    if pdf_path.exists():
        with open(pdf_path, "rb") as f:
            st.download_button(
                "📑 Download 450-Q Report (.pdf)",
                data=f.read(),
                file_name="benchmark_450_questions_evaluation_report.pdf",
                mime="application/pdf",
                use_container_width=True
            )

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
    else:
        # Default auto-load for instant live cloud exploration (zero friction for examiners)
        candidates = [
            ROOT / "data" / "bank_marketing_cleaned.csv",
            UPLOAD_DIR / "bank_marketing_cleaned_uploaded.csv",
            UPLOAD_DIR / "active_uploaded_dataset.csv",
            ROOT / "data" / "raw" / "sales.csv"
        ]
        for c_path in candidates:
            if c_path.is_file():
                frame = pd.read_csv(c_path, low_memory=False)
                active_name = f"Default Benchmark Dataset ({c_path.name})"
                active_path = c_path
                break

except Exception as exc:
    st.error(f"Could not load the dataset: {exc}")

st.title("🤖 Self Validating AI-Agent")
st.caption("Upload a dataset, explore its columns, and run analytical questions with result validation.")

if frame is None:
    st.info("Upload a CSV or Excel dataset using the panel on the left to get started.")
else:
    st.write(f"**Current dataset:** {active_name}")
    if {"question", "difficulty"}.issubset({str(c).lower() for c in frame.columns}):
        st.info("This upload contains benchmark questions rather than the original customer records. For customer analysis, upload the Bank Marketing dataset CSV instead.")
    total_cells = frame.shape[0] * frame.shape[1]
    missing = int(frame.isna().sum().sum())
    cols = st.columns(5)
    for col, (label, value) in zip(cols, [
        ("Rows", f"{len(frame):,}"),
        ("Columns", f"{len(frame.columns):,}"),
        ("Missing cells", f"{missing:,}"),
        ("Duplicate rows", f"{int(frame.duplicated().sum()):,}"),
        ("Completeness", f"{(100 * (1 - missing / total_cells) if total_cells else 0):.1f}%"),
    ]):
        col.metric(label, value)

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
                # Never present a stale report as a newly generated result.
                report = after.decode("utf-8", errors="replace") if after is not None and after != before else None
                run_data = {"dataset": active_name, "query": query, "elapsed": round(elapsed, 3),
                            "report": report, "status": report_status(report) if report else "NOT CONFIRMED",
                            "error": error, "logs": log.getvalue(), "fault_test": inject_fault}
                st.session_state.setdefault("analysis_runs", []).insert(0, run_data)
                st.session_state.analysis_runs = st.session_state.analysis_runs[:20]
        latest = st.session_state.get("analysis_runs", [])
        if latest:
            item = latest[0]
            st.markdown("### Latest analysis")
            x, y = st.columns(2)
            x.metric("Pipeline-reported status", item["status"])
            y.metric("Runtime", f"{item['elapsed']:.2f} seconds")
            st.caption("This status comes from the pipeline report; it is not independent proof of correctness.")
            if item["error"]:
                st.error(item["error"])
            if item["report"]:
                insight = extract_section(item["report"], "4. GENERATED INSIGHT", "5. INSIGHT VALIDATION")
                rec = extract_section(item["report"], "7. RECOMMENDATION", "8. OVERALL RELIABILITY")
                if insight:
                    st.markdown("**Generated insight**")
                    st.write(insight)
                if rec:
                    st.markdown("**Recommendation**")
                    st.write(rec)
                with st.expander("Full analysis report"):
                    st.text(item["report"])
                st.download_button("Download report", item["report"], "eda_report.txt", "text/plain")
            else:
                st.warning("No new report was detected. Check the execution logs for details.")
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
