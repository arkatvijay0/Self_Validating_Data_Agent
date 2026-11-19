v5 = improvements on top of v4 (install v2, v3, v4, then v5).
Changed source files: aggregate_support.py analytical_spec.py answer_support.py claim_evidence_validator.py grouped_count_support.py main.py recommendation_validator.py

INSTALL (PowerShell, project root):
  copy fixes_v5\src\*.py   src  copy fixes_v5\tests\*.py tests  copy fixes_v5\tools\*.py tools
VERIFY (43 questions checked against pandas, plus the test suite):
  $env:PYTHONPATH="src"
  python -m pytest -q tests src\test_analytical_spec.py src\test_task_analyzer_spec.py src\test_eda_planner_spec.py
  python tools\verify_answers.py outputs\uploads\bank_marketing_cleaned_uploaded.csv
