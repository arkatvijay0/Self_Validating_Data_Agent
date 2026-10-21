
import importlib
import sys
from pathlib import Path


def test_main_uses_review_module_without_editing_original():
    project_root = Path(__file__).resolve().parents[1]
    src = project_root / "src"

    sys.path.insert(0, str(src))

    import evidence_mapping as original
    import evidence_mapping_grouped_mean_fixed as reviewed

    assert callable(original.map_insight_to_evidence)
    assert callable(reviewed.map_insight_to_evidence)

    # Redirect only this test process's module lookup.
    sys.modules["evidence_mapping"] = reviewed

    import main
    reloaded_main = importlib.reload(main)

    assert (
        reloaded_main.map_insight_to_evidence
        is reviewed.map_insight_to_evidence
    )

    assert Path(original.__file__).name == "evidence_mapping.py"
    assert (
        Path(reviewed.__file__).name
        == "evidence_mapping_grouped_mean_fixed.py"
    )

    print("PASS: main uses the separate review module in this process.")
    print("PASS: original evidence_mapping.py remains a separate module.")
