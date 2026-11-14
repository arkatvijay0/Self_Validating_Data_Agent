INSTALL (from the project root, PowerShell):
  copy fixes_v2\src\*.py   src\
  copy fixes_v2\tests\*.py tests\
  mkdir tools ; copy fixes_v2\tools\verify_answers.py tools\

VERIFY:
  $env:PYTHONPATH="src"
  python -m pytest -q tests src\test_analytical_spec.py src\test_task_analyzer_spec.py src\test_eda_planner_spec.py
  python tools\verify_answers.py outputs\uploads\bank_marketing_cleaned_uploaded.csv

The 3 NEW files (grouped_count_support.py, aggregate_support.py, answer_support.py) must be copied too.
all_changes.diff shows every change against your original project (line endings normalised).
