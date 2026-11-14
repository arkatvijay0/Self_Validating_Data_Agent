
"""Safely insert optional, validated Ollama insight rewriting into src/main.py."""

from pathlib import Path
import shutil
import sys

MAIN_FILE = Path("src/main.py")
BACKUP_FILE = Path("src/main.py.pre_ollama_integration.bak")

ANCHOR = '''    print(
        "Evidence:"
    )
'''

MARKER = "# OPTIONAL OLLAMA INSIGHT REWRITING (validated candidate only)"

INSERTION = '''    # OPTIONAL OLLAMA INSIGHT REWRITING (validated candidate only)
    # Preserve the deterministic insight and its validation as fallbacks.
    original_insight = insight
    original_insight_validation = insight_validation

    try:
        from ollama_insight_rewriter import rewrite_insight_with_ollama

        candidate_insight = rewrite_insight_with_ollama(
            user_question=user_question,
            original_insight=original_insight,
            verified_results=primary_result,
            evidence=evidence,
        )

        if candidate_insight:
            candidate_validation_result = (
                evidence
                if str(analysis_type).lower() == "descriptive_summary"
                else validation_result_data
            )

            candidate_validation = validate_insight(
                insight=candidate_insight,
                result=candidate_validation_result,
                analysis_type=analysis_type,
                task_analysis=task_analysis,
                user_question=user_question,
                data_profile=data_profile,
            )

            if (
                isinstance(primary_result, dict)
                and str(primary_result.get("analysis_type", "")).lower()
                == "multi_metric_grouped_mean_analysis"
            ):
                candidate_validation = (
                    validate_multi_metric_grouped_mean_insight(
                        insight=candidate_insight,
                        combined_result=primary_result,
                        base_validation=candidate_validation,
                    )
                )

            candidate_passed = (
                isinstance(candidate_validation, dict)
                and candidate_validation.get("overall") == "PASS"
                and candidate_validation.get("pass") is True
            )

            if candidate_passed:
                insight = candidate_insight
                insight_validation = candidate_validation
                print(
                    "Ollama insight rewrite: ACCEPTED "
                    "(existing validators passed)."
                )
            else:
                insight = original_insight
                insight_validation = original_insight_validation
                print(
                    "Ollama insight rewrite: REJECTED "
                    "(validation failed); original insight retained."
                )
        else:
            print(
                "Ollama insight rewrite: SKIPPED; "
                "original insight retained."
            )

    except Exception as exc:
        insight = original_insight
        insight_validation = original_insight_validation
        print(
            "Ollama insight rewrite: SKIPPED due to an error; "
            "original insight retained. "
            f"Details: {exc}"
        )

'''

def main():
    if not MAIN_FILE.is_file():
        print(f"ERROR: Cannot find {MAIN_FILE}. Run from the project root.")
        return 1

    if BACKUP_FILE.exists():
        print(f"ERROR: Backup already exists: {BACKUP_FILE}")
        print("Rename or preserve it manually before retrying.")
        return 1

    source = MAIN_FILE.read_text(encoding="utf-8")

    if MARKER in source:
        print("ERROR: Ollama integration marker already exists.")
        print("No changes made.")
        return 1

    anchor_count = source.count(ANCHOR)
    if anchor_count != 1:
        print(
            f"ERROR: Expected exactly one insertion anchor; "
            f"found {anchor_count}. No changes made."
        )
        return 1

    updated_source = source.replace(ANCHOR, INSERTION + ANCHOR, 1)

    try:
        compile(updated_source, str(MAIN_FILE), "exec")
    except SyntaxError as exc:
        print(f"ERROR: Proposed main.py has a syntax error: {exc}")
        print("No changes made.")
        return 1

    shutil.copy2(MAIN_FILE, BACKUP_FILE)

    try:
        MAIN_FILE.write_text(updated_source, encoding="utf-8")
    except Exception:
        shutil.copy2(BACKUP_FILE, MAIN_FILE)
        print("ERROR: Write failed; main.py was restored from backup.")
        return 1

    print("Integration inserted successfully.")
    print(f"Backup created: {BACKUP_FILE}")
    print("Next, run the syntax checks and tests shown in the instructions.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
