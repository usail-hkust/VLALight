import ast
import re
from pathlib import Path


def _mode_constraint_regex(mode: str) -> str:
    """Load the pure helper without importing Ray-dependent VERL modules."""
    source_path = (
        Path(__file__).parents[2]
        / "verl"
        / "experimental"
        / "agent_loop"
        / "single_turn_agent_loop.py"
    )
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    node = next(
        item for item in module.body
        if isinstance(item, ast.FunctionDef) and item.name == "_mode_constraint_regex"
    )
    namespace: dict[str, object] = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source_path), "exec"), namespace)
    return namespace["_mode_constraint_regex"](mode)


def test_forced_fast_mode_must_start_with_exact_mode_tag():
    pattern = _mode_constraint_regex("fast")
    assert re.fullmatch(pattern, "<mode>fast</mode><signal>NTST</signal>")
    assert not re.fullmatch(pattern, "fast\n<mode>fast</mode><signal>NTST</signal>")
    assert not re.fullmatch(pattern, "<mode>slow</mode><signal>NTST</signal>")


def test_forced_slow_mode_requires_one_leading_slow_tag_and_reasoning():
    pattern = _mode_constraint_regex("slow")
    assert re.fullmatch(
        pattern,
        "<mode>slow</mode><reasoning>compare queues</reasoning><signal>NTST</signal>",
    )
    assert not re.fullmatch(pattern, "slow\n<mode>slow</mode><reasoning>x</reasoning>")
    assert not re.fullmatch(pattern, "<mode>slow</mode><signal>NTST</signal>")
