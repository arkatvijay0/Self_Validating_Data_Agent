"""Ollama rewrite status/timeouts, Isolation Forest settings, report compaction."""
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import final_report as FR
import ollama_insight_rewriter as R
from analytical_spec import build_analytical_spec
import answer_support as A

INSIGHT = "Average balance by housing (2 groups): no: 890.09; yes: 843.73. These are descriptive group averages."


def _install_fake_client(monkeypatch, available=True, reply=None):
    fake = types.ModuleType("ollama_client")
    fake.ollama_is_available = lambda model=None, timeout=5: available
    fake.ask_ollama = lambda **kw: reply
    monkeypatch.setitem(sys.modules, "ollama_client", fake)
    monkeypatch.setenv("EDA_ENABLE_OLLAMA_INSIGHT_REWRITE", "1")


def _rewrite():
    return R.rewrite_insight_with_ollama(user_question="q", original_insight=INSIGHT, verified_results={})


def test_disabled_by_default_says_how_to_enable(monkeypatch):
    monkeypatch.delenv("EDA_ENABLE_OLLAMA_INSIGHT_REWRITE", raising=False)
    assert _rewrite() is None
    assert R.last_status["state"] == "DISABLED" and "EDA_ENABLE_OLLAMA_INSIGHT_REWRITE" in R.last_status["detail"]


def test_unreachable_service_is_reported_not_hidden(monkeypatch):
    _install_fake_client(monkeypatch, available=False)
    assert _rewrite() is None
    assert R.last_status["state"] == "UNAVAILABLE" and "ollama pull" in R.last_status["detail"]


def test_rewrite_that_changes_a_number_is_discarded(monkeypatch):
    _install_fake_client(monkeypatch, reply=INSIGHT.replace("890.09", "980.09"))
    assert _rewrite() is None
    assert R.last_status["state"] == "UNSAFE" and "numeric" in R.last_status["detail"]


def test_faithful_rewrite_is_returned_as_a_candidate(monkeypatch):
    _install_fake_client(monkeypatch, reply=INSIGHT.replace("These are", "Note that these are"))
    assert _rewrite().startswith("Average balance")
    assert R.last_status["state"] == "CANDIDATE"


def test_unchanged_reply_is_reported(monkeypatch):
    _install_fake_client(monkeypatch, reply=INSIGHT)
    assert _rewrite() is None and R.last_status["state"] == "UNCHANGED"


def test_output_budget_grows_with_the_insight():
    assert R.output_token_budget("short") == R.REWRITE_MAX_OUTPUT_TOKENS
    assert R.output_token_budget("x" * 4000) > R.REWRITE_MAX_OUTPUT_TOKENS
    assert R.output_token_budget("x" * 100000) <= 1536


def test_availability_check_has_a_timeout(monkeypatch):
    ollama = pytest.importorskip("ollama")
    import importlib
    import ollama_client
    importlib.reload(ollama_client)
    seen = {}

    class FakeClient:
        def __init__(self, timeout=None, **kw):
            seen["timeout"] = timeout
        def list(self):
            return {"models": []}

    monkeypatch.setattr(ollama_client.ollama, "Client", FakeClient)
    assert ollama_client.ollama_is_available("m") is False
    assert seen["timeout"] and seen["timeout"] <= 10      # ollama.list() alone has no timeout


def test_contamination_comes_from_the_question_only_for_isolation_forest():
    profile = {"columns": {"balance": {"role": "continuous"}, "job": {"role": "categorical"}}}
    spec = build_analytical_spec("Flag the top 1% most unusual customers by balance using Isolation Forest.", profile)
    assert spec["outlier_method"] == "IsolationForest" and spec["contamination"] == pytest.approx(0.01)
    assert build_analytical_spec("Detect anomalies in balance using Isolation Forest.", profile)["contamination"] is None
    assert build_analytical_spec("Detect outliers in balance using IQR, 10% of rows", profile)["contamination"] is None
    assert build_analytical_spec("Flag 80% of customers using Isolation Forest", profile)["contamination"] is None


def test_sample_rows_are_compacted_everywhere_they_appear():
    row = {"age": 5, "job": "x", "balance": 100, "anomaly_score": -0.2, "anomaly_label": "anomaly"}
    nested = {"feature_columns": ["balance"], "sample_anomalies": [row],
              "evidence": {"result": {"sample_anomalies": [row]}}}
    text = FR.format_value(nested)
    assert '"age"' not in text and '"job"' not in text and '"balance": 100' in text
    assert nested["sample_anomalies"][0]["job"] == "x"          # original untouched


def test_count_answer_validation_displays_presence_and_result_pass():
    result = {"status": "PASS", "analysis_type": "specification_driven", "operations": ["count"], "metrics": [],
              "group_by": ["job"], "input_rows": 10, "analyzed_rows": 10,
              "operation_results": {"aggregations": [{"job": "a", "row_count": 6}, {"job": "b", "row_count": 4}]}}
    tagged = A.extract_special_result(result)
    insight = A.special_insight(tagged, {})
    validation = A.special_validate(insight, tagged, {})
    evidence = A.special_evidence(tagged, result, {})
    report = FR.create_final_report(
        user_question="q", analysis_type="grouped_count_analysis", validated_result=result, insight=insight,
        insight_validation=validation, evidence=evidence, recommendation="r",
        insight_evidence_mapping=A.special_mapping(insight, evidence, {}), recommendation_validation={"status": "PASS"})
    assert "Presence Check: PASS" in report and "Result Check: PASS" in report
    assert evidence["result"]["group_count"] == 2             # section 6 and traceability need this
