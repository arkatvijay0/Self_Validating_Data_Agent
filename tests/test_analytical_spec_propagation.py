"""Regression tests for end-to-end analytical_spec propagation."""
import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FILES = {
    "main": PROJECT_ROOT / "src/main.py",
    "multi_step": PROJECT_ROOT / "src/multi_step_executor.py",
    "step_executor": PROJECT_ROOT / "src/self_correcting_step_executor.py",
    "corrector": PROJECT_ROOT / "src/code_corrector.py",
    "generator": PROJECT_ROOT / "src/code_generator.py",
}


def _tree(name):
    return ast.parse(FILES[name].read_text(encoding="utf-8"))


def _function(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"Function not found: {name}")


def _has_keyword_call(function_node, called_name, keyword_name):
    for node in ast.walk(function_node):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else (
            func.attr if isinstance(func, ast.Attribute) else None
        )
        if name == called_name and any(k.arg == keyword_name for k in node.keywords):
            return True
    return False


def test_main_passes_spec_to_planner_and_executor():
    tree = _tree("main")
    main_fn = _function(tree, "main")
    assert _has_keyword_call(main_fn, "generate_plan_mock", "analytical_spec")
    assert _has_keyword_call(main_fn, "create_self_correcting_executor", "analytical_spec")


def test_multi_step_executor_accepts_and_forwards_spec():
    tree = _tree("multi_step")
    factory = _function(tree, "create_self_correcting_executor")
    assert "analytical_spec" in [a.arg for a in factory.args.args]
    assert _has_keyword_call(factory, "execute_self_correcting_step", "analytical_spec")


def test_step_executor_preserves_spec_in_generation_and_correction():
    tree = _tree("step_executor")
    step_fn = _function(tree, "execute_self_correcting_step")
    assert "analytical_spec" in [a.arg for a in step_fn.args.args]
    assert _has_keyword_call(step_fn, "generate_step_code", "analytical_spec")
    assert _has_keyword_call(step_fn, "correct_generated_code", "analytical_spec")
    assert _has_keyword_call(step_fn, "generate_code_mock", "analytical_spec")


def test_corrector_accepts_and_forwards_spec_to_fallback():
    tree = _tree("corrector")
    fn = _function(tree, "correct_generated_code")
    assert "analytical_spec" in [a.arg for a in fn.args.args]
    assert _has_keyword_call(fn, "generate_code_mock", "analytical_spec")


def test_generator_keeps_backward_compatible_optional_spec_parameter():
    tree = _tree("generator")
    fn = _function(tree, "generate_code_mock")
    args = [a.arg for a in fn.args.args]
    assert args[-1] == "analytical_spec"
    assert len(fn.args.defaults) == 4
