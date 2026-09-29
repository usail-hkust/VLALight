#!/usr/bin/env python3
"""Verify malformed model outputs receive gate rewards instead of exceptions."""

from v35_offline_grpo_reward import _format, compute_score


CASES = (
    "",
    "<perception>null</perception><mode>fast</mode><signal>ETWT</signal>",
    "<perception>{}</perception><mode>fast</mode><signal>ETWT</signal>",
    "<perception>{\"ETWT\":null,\"NTST\":{},\"ELWL\":{},\"NLSL\":{}}</perception>"
    "<mode>fast</mode><signal>ETWT</signal>",
    "<perception>{\"ETWT\":{\"current_v\":7},\"NTST\":{},\"ELWL\":{},\"NLSL\":{}}</perception>"
    "<mode>fast</mode><signal>ETWT</signal>",
    "<perception>{\"ETWT\":{\"current_v\":3}}</perception><mode>fast</mode>"
    "<signal>ETWT</signal>",
    "<perception>{bad json}</perception><mode>slow</mode><reasoning>x</reasoning>"
    "<signal>NTST</signal>",
)


def main() -> None:
    expected_scores = (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    for index, solution in enumerate(CASES):
        parsed, signal_bad, _ = _format(solution)
        result = compute_score(solution, {})
        score = result["score"]
        assert result["score"] == expected_scores[index], (index, signal_bad, result)
        assert not parsed, (index, solution)
    print(f"PASS: {len(CASES)} malformed outputs returned gate rewards without exceptions.")


if __name__ == "__main__":
    main()
