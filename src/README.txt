# Grouped Mean Routing Fix

This bundle contains complete replacement versions of:
- `src/task_analyzer.py`
- `src/eda_planner.py`
- `src/code_generator.py`

It expands the grouped-mean intent detection to recognize questions using words such as
“separately”, “group means”, and directional claim language such as “lower than”.
It does not hardcode the expected balance values; values must still be calculated from
the active dataset.

Before replacing files, make backups. After replacement, run the targeted offline tests
and verify the generated report using the bank-marketing dataset.
