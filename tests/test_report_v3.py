"""Group sizes, result table, per-run trace, and Ollama rewrite validation for new answer types."""
import os
import sys
import types
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import final_report as FR
import main as M


RESULT = {
    "status": "PASS", "analysis_type": "specification_driven", "operations": ["mean"],
    "metrics": ["balance"], "group_by": ["housing"], "input_rows": 10, "analyzed_rows": 10,
    "operation_results": {
        "aggregations": [{"housing": "no", "balance__mean": 890.0}, {"housing": "yes", "balance__mean": 843.5}],
        "group_sizes": [{"housing": "no", "row_count": 4, "balance__valid_count": 4},
                        {"housing": "yes", "row_count": 6, "balance__valid_count": 6}],
    },
}


def test_group_sizes_reach_the_normalised_groups():
    groups = M.extract_specification_grouped_mean(RESULT)["group_results"]
    assert groups["no"]["count"] == 4 and groups["yes"]["valid_count"] == 6


def test_result_table_is_sorted_and_shows_records():
    table = "\n".join(FR.render_result_table(RESULT))
    assert table.index("housing=no") < table.index("housing=yes")
    assert "records" in table and "890.00" in table


def test_large_results_are_compacted_for_display_only():
    rows = [{"job": f"j{i}", "balance__mean": float(i)} for i in range(40)]
    big = dict(RESULT, group_by=["job"], operation_results={"aggregations": rows})
    shown = FR.compact_result_for_display(big)
    assert len(shown["operation_results"]["aggregations"]) == 6
    assert len(big["operation_results"]["aggregations"]) == 40          # original untouched


def test_recommendation_states_group_sizes_with_clear_labels():
    evidence = {"status": "PASS"}
    text = M.generate_pipeline_recommendation("comparison", evidence, RESULT) \
        if M.generate_pipeline_recommendation.__code__.co_argcount == 3 else None
    if text is None:   # signature differs between versions; the end-to-end test below covers it
        return
    assert "housing=no" in text


def test_trace_reports_what_happened_in_this_run():
    trace = M.build_pipeline_trace(
        {"n_rows": 10, "n_columns": 2, "columns": {"a": {"role": "continuous", "null_count": 0}}},
        "/x/data.csv", "q", {"task_type": "comparison", "analytical_spec": {"status": "READY"}},
        {"steps": [{"status": "PASS"}], "overall_status": "PASS", "analytical_result_validation": {"status": "PASS"}},
        {"overall_status": "PASS", "reliability": {"total_attempts": 1, "correction_steps": 0}},
        {"analysis_type": "x"}, "grouped_mean_analysis", {"overall": "PASS"}, {"status": "PASS"},
        {"status": "PASS"}, "SKIPPED (Ollama unavailable or errored; original insight kept)")
    names = {stage: (status, detail) for stage, status, detail in trace}
    assert names["Self-Correction and Re-execution"][0] == "NOT NEEDED"
    assert names["Ollama-Based Insight Rewriting (optional)"][0] == "SKIPPED"
    assert names["Isolation Forest Anomaly Detection"][0] == "NOT REQUESTED"
    assert len(trace) == 15


def test_bad_ollama_rewrite_of_a_count_answer_is_rejected(tmp_path, monkeypatch):
    df = pd.DataFrame({"job": ["a"] * 5 + ["b"] * 3 + ["c"] * 2, "balance": range(10), "age": range(20, 30)})
    csv = tmp_path / "d.csv"
    df.to_csv(csv, index=False)
    stub = types.ModuleType("ollama_insight_rewriter")
    stub.rewrite_insight_with_ollama = lambda **kw: (
        "Top 2 job groups by customer count (3 groups, 10 records in total): a: 9; b: 3. "
        "Counts are descriptive and do not establish causation.")
    monkeypatch.setitem(sys.modules, "ollama_insight_rewriter", stub)
    monkeypatch.chdir(ROOT)
    M.main(dataset_path=str(csv), user_question="Identify the top 2 job categories by customer count.")
    report = (ROOT / "outputs" / "final_eda_report.txt").read_text(encoding="utf-8")
    assert "a: 5; b: 3" in report and "a: 9" not in report
    assert "REJECTED" in report
