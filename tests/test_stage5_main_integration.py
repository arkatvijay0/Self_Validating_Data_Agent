import ast
from pathlib import Path


def test_main_imports_and_runs_claim_evidence_validator():
    main_path = Path(__file__).resolve().parents[1] / "src" / "main.py"
    tree = ast.parse(main_path.read_text(encoding="utf-8"))

    imports_validator = any(
        isinstance(node, ast.ImportFrom)
        and node.module == "claim_evidence_validator"
        and any(alias.name == "validate_insight_and_recommendation" for alias in node.names)
        for node in ast.walk(tree)
    )
    calls_validator = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "validate_insight_and_recommendation"
        for node in ast.walk(tree)
    )
    assert imports_validator
    assert calls_validator
