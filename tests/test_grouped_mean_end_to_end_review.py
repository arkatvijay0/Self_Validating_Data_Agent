
import importlib
import sys
from pathlib import Path
from contextlib import redirect_stdout
from io import StringIO


def test_bank_marketing_grouped_mean_end_to_end():
    project_root = Path(__file__).resolve().parents[1]
    src = project_root / "src"
    dataset = (
        project_root
        / "outputs"
        / "uploads"
        / "bank_marketing_cleaned_uploaded.csv"
    )

    assert dataset.is_file(), f"Dataset not found: {dataset}"

    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    # Import the original module and reviewed module separately.
    original = importlib.import_module("evidence_mapping")
    reviewed = importlib.import_module(
        "evidence_mapping_grouped_mean_fixed"
    )



    # Reload main using its current, explicit mapper import.
    import main
    main = importlib.reload(main)

    assert (
        main.map_insight_to_evidence
        is reviewed.map_insight_to_evidence
    )

    captured = StringIO()

    print("DIAGNOSTIC 1: Imports completed.", flush=True)
    print(f"DIAGNOSTIC 2: Dataset exists: {dataset.is_file()}", flush=True)
    print("DIAGNOSTIC 3: Starting the full EDA pipeline.", flush=True)

    with redirect_stdout(captured):
        main.main(
            dataset_path=str(dataset),
            user_question=(
                "Compare the average balance across customers "
                "who subscribed and those who did not."
            ),
        )

    print("DIAGNOSTIC 4: EDA pipeline returned.", flush=True)

    output = captured.getvalue()
    print(output)

    assert "Insight-Evidence Mapping:" in output
    assert "OVERALL RESULT:" in output
    assert "Insight-Evidence Mapping: PASS" in output, (
        "The pipeline did not report PASS for insight-evidence mapping."
    )
    assert "OVERALL RESULT: PASS" in output, (
        "The full pipeline did not report an overall PASS."
    )

    print("PASS: End-to-end grouped-mean review completed.")
