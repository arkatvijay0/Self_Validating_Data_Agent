"""
Run questions through the real pipeline and check every answer against an
independent pandas calculation.

    python tools/verify_answers.py [path/to/bank_marketing.csv]

Prints OK/FAIL per question and a total. Needs the bank-marketing columns
(job, balance, duration, contact, education, housing, loan, month, age, y,
conversion). Run it from the project root after any change to the pipeline.
"""
import contextlib
import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)

import pandas as pd  # noqa: E402
import main  # noqa: E402

DATASET = sys.argv[1] if len(sys.argv) > 1 else os.path.join("outputs", "uploads", "bank_marketing_cleaned_uploaded.csv")
df = pd.read_csv(DATASET)


LAST_REPORT = ""


def run(question):
    global LAST_REPORT
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            main.main(dataset_path=DATASET, user_question=question)
    except BaseException as exc:  # noqa: BLE001 - report any crash as a failure
        return "CRASH", f"{type(exc).__name__}: {exc}"
    report = open(os.path.join("outputs", "final_eda_report.txt"), encoding="utf-8").read()
    LAST_REPORT = report
    if "NEEDS CLARIFICATION" in report:
        return "NEEDS CLARIFICATION", report
    insight = report.split("4. GENERATED INSIGHT")[1].split("5. INSIGHT VALIDATION")[0]
    overall = report.split("8. OVERALL RELIABILITY")[1].split("Overall reliability is")[0].strip().split()[-1]
    return overall, insight


f2 = lambda v: f"{v:,.2f}"          # noqa: E731
fi = lambda v: f"{int(v):,}"        # noqa: E731
grp = lambda c, m, fn: df.groupby(c)[m].agg(fn)  # noqa: E731


def has_pairs(text, series, fmt=f2):
    return all(re.search(re.escape(str(k)) + r"\s*:\s*" + re.escape(fmt(v)) + r"(?![\d,]|\.\d)", text)
               for k, v in series.items())


def has_winner(text, series, highest=True):
    key = series.idxmax() if highest else series.idxmin()
    return re.search(re.escape(str(key)) + r"\s*\(\s*" + re.escape(f2(series[key])), text) is not None


counts_job = df.job.value_counts()
q1, q3 = df.balance.quantile(.25), df.balance.quantile(.75)
iqr_count = int(((df.balance < q1 - 1.5 * (q3 - q1)) | (df.balance > q3 + 1.5 * (q3 - q1))).sum())
long_calls = df[df.duration > 300]

CHECKS = [
    ("What is the average balance?", lambda t: f2(df.balance.mean()) in t),
    ("What is the median duration?", lambda t: f2(df.duration.median()) in t),
    ("What is the maximum balance?", lambda t: f2(df.balance.max()) in t),
    ("What is the total balance?", lambda t: f2(df.balance.sum()) in t),
    ("How many customers are there?", lambda t: fi(len(df)) in t),
    ("How many unique jobs are there?", lambda t: re.search(rf"\b{df.job.nunique()}\b", t) is not None),
    ("Calculate the average balance grouped by job.", lambda t: has_pairs(t, grp("job", "balance", "mean"))),
    ("Calculate the median duration by contact.", lambda t: has_pairs(t, grp("contact", "duration", "median"))),
    ("What is the standard deviation of balance by loan?", lambda t: has_pairs(t, grp("loan", "balance", "std"))),
    ("Which job category has the highest average balance?", lambda t: has_winner(t, grp("job", "balance", "mean"), True)),
    ("Which education category has the lowest average balance?", lambda t: has_winner(t, grp("education", "balance", "mean"), False)),
    ("Which month has the highest average campaign duration?", lambda t: has_winner(t, grp("month", "duration", "mean"), True)),
    ("Which housing group has the highest median balance?", lambda t: has_winner(t, grp("housing", "balance", "median"), True)),
    ("Identify the top three job categories by customer count.",
     lambda t: has_pairs(t, counts_job.head(3), fi) and str(counts_job.index[3]) + ":" not in t),
    ("Which contact type has the most customers?",
     lambda t: (lambda c: str(c.index[0]) in t and fi(c.iloc[0]) in t)(df.contact.value_counts())),
    ("Is there a correlation between age and balance?", lambda t: f"{df.age.corr(df.balance):.3f}" in t),
    ("What is the average age for customers with duration above 300?",
     lambda t: f2(long_calls.age.mean()) in t and fi(len(long_calls)) in t),
    ("Compare average balance between customers with and without housing loans.",
     lambda t: has_pairs(t, grp("housing", "balance", "mean"))),
    ("Is the average balance higher or lower for customers with a loan?",
     lambda t: has_pairs(t, grp("loan", "balance", "mean"))),
    ("Calculate the average balance and average duration grouped by loan.",
     lambda t: has_pairs(t, grp("loan", "balance", "mean")) and has_pairs(t, grp("loan", "duration", "mean"))),
    ("Detect outliers in balance using IQR.", lambda t: fi(iqr_count) in t),
    ("Calculate the average age grouped by job and marital status.",
     lambda t: all(f"job={j}, marital={m}: {f2(v)}" in t for (j, m), v in df.groupby(["job", "marital"]).age.mean().items())),
    ("Compare average balance between customers with and without housing loans.",
     lambda t: all(f"housing={k}" in LAST_REPORT and fi(n) in LAST_REPORT for k, n in df.housing.value_counts().items())),
    ("What is the conversion rate by job?",
     lambda t: (lambda r: str(r.idxmax()) in t and f"{r.max():.2f}%" in t)(df.groupby("job").conversion.mean() * 100)),
]
CLARIFY = [
    "What is the average salary by job?",
    "What is the average balance for customers with a loan?",
    "Generate actionable recommendations based on customer balance and conversion patterns.",
]

if __name__ == "__main__":
    passed = 0
    for question, check in CHECKS:
        status, text = run(question)
        ok = status == "PASS" and bool(check(text))
        passed += ok
        print(("OK   " if ok else "FAIL ") + f"[{status}] {question}")
    for question in CLARIFY:
        status, _ = run(question)
        ok = status == "NEEDS CLARIFICATION"
        passed += ok
        print(("OK   " if ok else "FAIL ") + f"[{status}] {question}")
    total = len(CHECKS) + len(CLARIFY)
    print(f"\n{passed}/{total} answers verified against pandas")
    sys.exit(0 if passed == total else 1)
