"""Cheap contract check for the V35 mode-token constraint.

This check is deliberately independent of GPU/vLLM startup.  It verifies that
the grammar accepts the original tag order and rejects the opposite mode.
"""

import re

from verl.experimental.agent_loop.single_turn_agent_loop import _mode_constraint_regex


def _response(mode: str, *, with_reasoning: bool = False, perception_closed: bool = True) -> str:
    perception_end = "</perception>" if perception_closed else ""
    reasoning_block = "<reasoning>compare competing phases</reasoning>\n" if with_reasoning else ""
    return (
        f"<perception>queues and persistent demand{perception_end}\n"
        f"<mode>{mode}</mode>\n"
        f"{reasoning_block}"
        "<current_v>{\"ETWT\": 2}</current_v>\n"
        "<signal>ETWT</signal>"
    )


def _response_with_padded_mode(mode: str, *, with_reasoning: bool = False) -> str:
    return _response(mode, with_reasoning=with_reasoning).replace(
        f"<mode>{mode}</mode>", f"<mode>\n{mode}\n</mode>"
    )


def main() -> None:
    results = {}
    for forced_mode in ("fast", "slow"):
        regex = _mode_constraint_regex(forced_mode)
        expected = _response(forced_mode, with_reasoning=forced_mode == "slow")
        accepts_expected = re.fullmatch(regex, expected) is not None
        accepts_missing_perception_close = re.fullmatch(
            regex, _response(forced_mode, with_reasoning=forced_mode == "slow", perception_closed=False)
        ) is not None
        rejects_opposite = re.fullmatch(
            regex, _response("slow" if forced_mode == "fast" else "fast", with_reasoning=forced_mode == "slow")
        ) is None
        rejects_wrong_reasoning_branch = re.fullmatch(
            regex, _response(forced_mode, with_reasoning=forced_mode == "fast")
        ) is None
        rejects_wrong_mode_first = re.fullmatch(
            regex,
            _response("slow" if forced_mode == "fast" else "fast", with_reasoning=False)
            + _response(forced_mode, with_reasoning=forced_mode == "slow"),
        ) is None
        rejects_duplicate_mode = re.fullmatch(regex, expected + f"<mode>{forced_mode}</mode>") is None
        rejects_padded_mode = re.fullmatch(
            regex, _response_with_padded_mode(forced_mode, with_reasoning=forced_mode == "slow")
        ) is None
        results[forced_mode] = {
            "regex": regex,
            "accepts_original_order": accepts_expected,
            "accepts_missing_perception_close": accepts_missing_perception_close,
            "rejects_opposite_mode": rejects_opposite,
            "rejects_wrong_reasoning_branch": rejects_wrong_reasoning_branch,
            "rejects_wrong_mode_first": rejects_wrong_mode_first,
            "rejects_duplicate_mode": rejects_duplicate_mode,
            "rejects_padded_mode": rejects_padded_mode,
        }
        assert accepts_expected
        assert accepts_missing_perception_close
        assert rejects_opposite
        assert rejects_wrong_reasoning_branch
        assert rejects_wrong_mode_first
        assert rejects_duplicate_mode
        assert rejects_padded_mode
    print({"status": "PASS", "results": results})


if __name__ == "__main__":
    main()
