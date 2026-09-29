"""Rule reward for V30 simulator-free, action-feedback GRPO.

The dataset's ground truth is a precomputed four-action feedback table. This
module deliberately sees no SUMO state, heuristic winner, teacher response, or
future video: it only parses the policy response and looks up its selected
physical signal action.
"""

from __future__ import annotations

import json
import re
from typing import Any


PHASES = frozenset({"ETWT", "NTST", "ELWL", "NLSL"})
PERCEPTION_REWARD_WEIGHT = 0.5
MOVEMENTS = {
    "ETWT": frozenset({"ET", "WT"}),
    "NTST": frozenset({"NT", "ST"}),
    "ELWL": frozenset({"EL", "WL"}),
    "NLSL": frozenset({"NL", "SL"}),
}
SIGNAL_RE = re.compile(r"<signal>\s*([A-Z]+)\s*</signal>")
MODE_RE = re.compile(r"<mode>\s*(fast|slow)\s*</mode>")
REASONING_RE = re.compile(r"<reasoning>\s*(.*?)\s*</reasoning>", re.DOTALL)
PERCEPTION_RE = re.compile(r"<perception>\s*(.*?)\s*</perception>", re.DOTALL)
CURRENT_V_RE = re.compile(r"<current_v>\s*(.*?)\s*</current_v>", re.DOTALL)
REQUIRED_BLOCKS = ("<perception>", "</perception>", "<current_v>", "</current_v>")


def _as_mapping(ground_truth: Any) -> dict[str, Any]:
    if isinstance(ground_truth, str):
        return json.loads(ground_truth)
    if isinstance(ground_truth, dict):
        return ground_truth
    raise TypeError(f"Unsupported ground_truth type: {type(ground_truth)!r}")


def _parse_current_v(solution_str: str) -> tuple[dict[str, int] | None, dict[str, int] | None]:
    """Extract final totals and the matching totals inside perception JSON."""
    final_matches = CURRENT_V_RE.findall(solution_str)
    perception_matches = PERCEPTION_RE.findall(solution_str)
    if len(final_matches) != 1 or len(perception_matches) != 1:
        return None, None
    try:
        final = json.loads(final_matches[0])
        perception = json.loads(perception_matches[0])
        if set(final) != PHASES:
            return None, None
        final_totals = {phase: int(final[phase]) for phase in PHASES}
        candidates = perception["candidate_phases"]
        by_phase = {candidate["signal"]: candidate for candidate in candidates}
        if set(by_phase) != PHASES:
            return None, None
        perception_totals = {
            phase: int(by_phase[phase]["current_v"]["total"])
            for phase in PHASES
        }
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None, None
    return final_totals, perception_totals


def _is_nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _valid_structured_perception(solution_str: str) -> bool:
    """Validate the V35 output contract without judging unlabelled features.

    V30 supplies exact labels only for Current V. Queue, trend, coordination,
    and history fields are still mandatory schema elements, but are not given a
    fabricated numerical reward in this offline stage.
    """
    matches = PERCEPTION_RE.findall(solution_str)
    if len(matches) != 1:
        return False
    try:
        perception = json.loads(matches[0])
        if perception.get("current_phase") not in PHASES:
            return False
        candidates = perception["candidate_phases"]
        by_phase = {candidate["signal"]: candidate for candidate in candidates}
        if len(candidates) != 4 or set(by_phase) != PHASES:
            return False
        for phase, candidate in by_phase.items():
            movements = MOVEMENTS[phase]
            current_v = candidate["current_v"]
            current_q = candidate["current_q"]
            arrivals = candidate["coordinated_arrivals"]
            if set(current_v) != {"total", *movements} or set(current_q) != {"total", *movements}:
                return False
            if not all(_is_nonnegative_int(current_v[key]) for key in current_v):
                return False
            if not all(_is_nonnegative_int(current_q[key]) for key in current_q):
                return False
            if current_v["total"] != sum(current_v[key] for key in movements):
                return False
            if current_q["total"] != sum(current_q[key] for key in movements):
                return False
            if any(current_q[key] > current_v[key] for key in current_v):
                return False
            if not isinstance(candidate["demand_trend_v30_minus_v5"], int):
                return False
            if not isinstance(candidate["queue_trend_q30_minus_q5"], int):
                return False
            if not _is_nonnegative_int(candidate["nonzero_v_history_length_since_last_service"]):
                return False
            breakdown = arrivals["breakdown"]
            if not _is_nonnegative_int(arrivals["total"]) or set(breakdown) != movements:
                return False
            for movement, item in breakdown.items():
                if not _is_nonnegative_int(item["count"]) or item["is_boundary"] not in {"yes", "no"}:
                    return False
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return True


def _valid_tag_order(solution_str: str, mode: str, reasoning_matches: list[str]) -> bool:
    """Require the five V35 blocks in their specified order."""
    tags = ["<perception>", "</perception>", "<mode>", "</mode>"]
    if mode == "slow":
        if len(reasoning_matches) != 1:
            return False
        tags.extend(["<reasoning>", "</reasoning>"])
    elif reasoning_matches:
        return False
    tags.extend(["<current_v>", "</current_v>", "<signal>", "</signal>"])
    positions = [solution_str.find(tag) for tag in tags]
    return all(position >= 0 for position in positions) and positions == sorted(positions)


def compute_score(data_source: str, solution_str: str, ground_truth: Any, extra_info: dict[str, Any] | None = None) -> dict[str, float]:
    """Score one generated response.

    Traffic quality is normalized within the current state's four actions.
    A complete structured response is required before either traffic or
    perception feedback is awarded. The two optimized quantities then share
    the same natural [0, 1] scale: traffic quality and Current V accuracy.
    """
    del data_source, extra_info
    invalid_penalty = -1.0
    signal_matches = SIGNAL_RE.findall(solution_str)
    if len(signal_matches) != 1 or signal_matches[0] not in PHASES:
        return {
            "score": invalid_penalty,
            "traffic_reward": 0.0,
            "format_reward": 0.0,
            "reasoning_penalty": 0.0,
            "valid_action": 0.0,
        }

    try:
        feedback = _as_mapping(ground_truth)
        action = signal_matches[0]
        traffic_reward = float(feedback["actions"][action]["traffic_reward"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        # A corrupted feedback record is a data error, not a model error. Do
        # not accidentally reward a completion against an unknown table.
        return {
            "score": invalid_penalty,
            "traffic_reward": 0.0,
            "format_reward": 0.0,
            "reasoning_penalty": 0.0,
            "valid_action": 0.0,
        }

    mode_matches = MODE_RE.findall(solution_str)
    mode = mode_matches[0] if len(mode_matches) == 1 else None
    reasoning_matches = REASONING_RE.findall(solution_str)
    has_required_blocks = all(block in solution_str for block in REQUIRED_BLOCKS)
    valid_mode = mode in {"fast", "slow"}
    valid_reasoning = (mode == "fast" and not reasoning_matches) or (
        mode == "slow" and len(reasoning_matches) == 1 and bool(reasoning_matches[0].strip())
    )
    valid_format = (
        has_required_blocks
        and valid_mode
        and valid_reasoning
        and _valid_tag_order(solution_str, mode, reasoning_matches)
        and _valid_structured_perception(solution_str)
    )

    # V30 labels and the input-window endpoint describe the same decision
    # boundary.  A one-vehicle tolerance absorbs visual/frame discretization.
    final_totals, perception_totals = _parse_current_v(solution_str)
    target = feedback.get("perception_target", {})
    perception_reward = 0.0
    if valid_format and set(target) == PHASES and final_totals is not None and perception_totals is not None:
        # Each phase contributes one equal, interpretable quarter. A phase is
        # correct only when its two required output locations agree and its
        # value lies within the accepted one-vehicle visual tolerance.
        perception_reward = 0.25 * sum(
            abs(final_totals[phase] - int(target[phase])) <= 1
            and abs(final_totals[phase] - perception_totals[phase]) <= 1
            for phase in PHASES
        )

    traffic_reward = traffic_reward if valid_format else 0.0
    # Traffic quality is the primary objective. Current-V perception remains
    # useful as an auxiliary grounding signal, but contributes at half scale
    # so it cannot compensate for a materially worse phase choice.
    score = traffic_reward + PERCEPTION_REWARD_WEIGHT * perception_reward
    return {
        "score": float(score),
        "traffic_reward": traffic_reward,
        "format_reward": 1.0 if valid_format else 0.0,
        "perception_reward": float(perception_reward),
        "reasoning_penalty": 0.0,
        "valid_action": 1.0,
    }
