"""
Check the optional / advanced pipeline components on THIS machine and say exactly
what is wrong when one is not working.

    python tools/check_components.py [path/to/dataset.csv]

1. Ollama rewriting  - env flag, python package, service reachable (5 s limit),
                       model pulled, and a real sample rewrite through the same
                       safety checks the pipeline uses.
2. Isolation Forest  - the pipeline's detector versus a separate, direct
                       scikit-learn run on the raw column; named-column selection;
                       flagged share equals the requested share.
3. Self-correction   - forces a code fault and confirms the pipeline recovers and
                       reports it.
"""
import contextlib
import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)

DATASET = sys.argv[1] if len(sys.argv) > 1 else os.path.join("outputs", "uploads", "bank_marketing_cleaned_uploaded.csv")
results = []


def report(name, ok, detail=""):
    results.append(ok)
    print(("OK    " if ok is True else "FAIL  " if ok is False else "INFO  ") + name + (f"  - {detail}" if detail else ""))


def check_ollama():
    print("\n== 1. Ollama insight rewriting (optional) ==")
    enabled = os.getenv("EDA_ENABLE_OLLAMA_INSIGHT_REWRITE", "0").strip().lower() in {"1", "true", "yes"}
    report("EDA_ENABLE_OLLAMA_INSIGHT_REWRITE", None if enabled else None,
           "enabled" if enabled else "NOT set - rewriting is OFF by default. PowerShell: $env:EDA_ENABLE_OLLAMA_INSIGHT_REWRITE='1'")
    try:
        import ollama_client
    except ImportError as exc:
        return report("python package 'ollama'", False, f"{exc}  ->  pip install ollama")
    report("python package 'ollama'", True)
    model = ollama_client.DEFAULT_MODEL
    if not ollama_client.ollama_is_available(model):
        return report(f"Ollama service + model '{model}'", False,
                      f"not reachable (or model missing). Start Ollama, then:  ollama pull {model}")
    report(f"Ollama service + model '{model}'", True)
    if not enabled:
        return report("sample rewrite", None, "skipped because the feature flag is off")
    import ollama_insight_rewriter as rewriter
    sample = ("Average balance by housing (2 groups): no: 1,596.50; yes: 1,175.10. "
              "These are descriptive group averages and do not establish causation.")
    candidate = rewriter.rewrite_insight_with_ollama(
        user_question="Compare average balance between customers with and without housing loans.",
        original_insight=sample, verified_results={"no": 1596.50, "yes": 1175.10})
    state = rewriter.last_status["state"]
    report("sample rewrite through the pipeline's safety checks", state in {"CANDIDATE", "UNCHANGED", "UNSAFE"},
           f"{state}: {rewriter.last_status['detail']}" + (f"\n        candidate: {candidate}" if candidate else ""))


def check_isolation_forest():
    print("\n== 2. Isolation Forest ==")
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import IsolationForest
    from anomaly_detector import detect_anomalies, select_anomaly_features
    df = pd.read_csv(DATASET)
    numeric = df.select_dtypes(include="number").columns.tolist()
    column = "balance" if "balance" in numeric else numeric[0]
    features = select_anomaly_features(df, {"metrics": [column]})
    report("named column is the only feature", features == [column], f"features={features}")
    for share in (0.05, 0.01):
        out = detect_anomalies(df, feature_columns=features, contamination=share, random_state=42, n_estimators=200)
        direct = IsolationForest(n_estimators=200, contamination=share, random_state=42).fit_predict(df[[column]].to_numpy())
        direct_count = int((direct == -1).sum())
        got = int(out["summary"]["anomaly_count"])
        report(f"pipeline detector vs direct scikit-learn ({share:.0%})", abs(got - direct_count) <= max(2, len(df) // 1000),
               f"pipeline={got:,} direct={direct_count:,}")
        report(f"flagged share equals the setting ({share:.0%})", abs(got / len(df) - share) < 0.002, f"{got / len(df):.2%}")
    general = select_anomaly_features(df, {})
    report("general features skip binary flags and identifiers", not any(df[c].nunique() <= 2 for c in general),
           f"features={general}")


def check_self_correction():
    print("\n== 3. Self-correction (forced fault) ==")
    import main
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            main.main(dataset_path=DATASET, user_question="What is the average balance?", test_fault=True)
    except BaseException as exc:  # noqa: BLE001
        return report("pipeline with injected fault", False, f"{type(exc).__name__}: {exc}")
    text = open(os.path.join("outputs", "final_eda_report.txt"), encoding="utf-8").read()
    used = re.search(r"Self-Correction and Re-execution\s+Status : USED", text) is not None
    report("fault was corrected and reported as USED", used and "OVERALL RESULT: PASS" in buffer.getvalue())


if __name__ == "__main__":
    for step in (check_ollama, check_isolation_forest, check_self_correction):
        try:
            step()
        except Exception as exc:  # noqa: BLE001
            report(step.__name__, False, f"{type(exc).__name__}: {exc}")
    failed = sum(1 for r in results if r is False)
    print(f"\n{'ALL CHECKS PASSED' if not failed else str(failed) + ' CHECK(S) FAILED'}")
    sys.exit(1 if failed else 0)
