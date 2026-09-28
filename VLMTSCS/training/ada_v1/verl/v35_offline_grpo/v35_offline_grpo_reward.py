"""Offline V35 GRPO reward: contract gate and counterfactual traffic."""

from __future__ import annotations

import json
import math
import os
import re
import datetime
from contextvars import ContextVar
from functools import lru_cache
from typing import Any

PHASES = ("ETWT", "NTST", "ELWL", "NLSL")
MOVEMENTS = {"ETWT": ("ET", "WT"), "NTST": ("NT", "ST"), "ELWL": ("EL", "WL"), "NLSL": ("NL", "SL")}
# Reasoning is charged only through its configured length penalty. There is no
# separate fixed cost for selecting slow, so the mode selector has no built-in
# fast prior.
REASONING_FREE_TOKENS = 300
REASONING_TAU = 400.0
REASONING_MAX_LENGTH_PENALTY = 0.10
MODE_BONUS_WEIGHT = 0.0
MODE_CONFIDENCE_CENTER = 5.0
MODE_CONFIDENCE_TEMPERATURE = 2.0
SIGNAL_ADVANTAGE_CLIP = float(os.environ.get("V35_SIGNAL_ADVANTAGE_CLIP", "2.0"))
PERCEPTION_SPLIT_WEIGHTING = os.environ.get("V35_PERCEPTION_SPLIT_WEIGHTING", "1") == "1"
# Perception is optimized by a gold-token CE auxiliary loss. Keep all
# perception diagnostics below, but do not place them in the GRPO scalar.
PERCEPTION_REWARD_WEIGHT = 0.0
_REWARD_CONTEXT: ContextVar[dict[str, Any]] = ContextVar("reward_context", default={})


def _write_reward_log(record: dict[str, Any]) -> None:
    """Append one complete reward audit record; safe for concurrent workers."""
    path = os.environ.get("V35_GRPO_REWARD_LOG", "")
    if not path:
        return
    record = dict(record)
    for key, value in _REWARD_CONTEXT.get().items():
        record.setdefault(key, value)
    record["timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    record["pid"] = os.getpid()
    try:
        payload = (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, payload)
        finally:
            os.close(fd)
    except OSError:
        # Reward logging must never interrupt training.
        pass


def _json(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return None
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return None


def _tag(text: str, name: str) -> list[str]:
    return re.findall(rf"<{name}>\s*(.*?)\s*</{name}>", text or "", flags=re.I | re.S)


def _has_exact_tag_pair(text: str, name: str) -> bool:
    return (
        len(re.findall(rf"<{name}>", text or "", flags=re.I)) == 1
        and len(re.findall(rf"</{name}>", text or "", flags=re.I)) == 1
    )


def _phase_map(perception: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(perception, dict):
        return {}
    if all(isinstance(perception.get(p), dict) for p in PHASES):
        return {p: perception[p] for p in PHASES}
    rows = perception.get("candidate_phases")
    if isinstance(rows, list):
        return {r.get("signal"): r for r in rows if isinstance(r, dict) and r.get("signal") in PHASES}
    return {}


def _has_exact_candidate_phases(perception: Any) -> bool:
    if not isinstance(perception, dict):
        return False
    rows = perception.get("candidate_phases")
    if not isinstance(rows, list) or len(rows) != len(PHASES):
        return False
    signals = [row.get("signal") for row in rows if isinstance(row, dict)]
    return len(signals) == len(PHASES) and set(signals) == set(PHASES)


def _valid_phase_row(row: Any, phase: str) -> bool:
    """Validate one perception row without raising on malformed model output."""
    if not isinstance(row, dict):
        return False

    current_v = row.get("current_v")
    current_q = row.get("current_q")
    coordinated = row.get("coordinated_arrivals")
    if not all(isinstance(group, dict) for group in (current_v, current_q, coordinated)):
        return False
    if not all(isinstance(group.get("total"), (int, float)) for group in (current_v, current_q, coordinated)):
        return False
    if not isinstance(row.get("nonzero_v_history_length_since_last_service"), (int, float)):
        return False
    if not isinstance(row.get("demand_trend_v30_minus_v5"), (int, float)):
        return False
    if not isinstance(row.get("queue_trend_q30_minus_q5"), (int, float)):
        return False

    movements = MOVEMENTS[phase]
    if not all(isinstance(current_v.get(movement), (int, float)) for movement in movements):
        return False
    if not all(isinstance(current_q.get(movement), (int, float)) for movement in movements):
        return False
    if current_v["total"] != sum(current_v[movement] for movement in movements):
        return False
    if current_q["total"] != sum(current_q[movement] for movement in movements):
        return False
    if not 0 <= current_q["total"] <= current_v["total"]:
        return False
    breakdown = coordinated.get("breakdown")
    if not isinstance(breakdown, dict) or coordinated["total"] < 0:
        return False
    for movement in movements:
        entry = breakdown.get(movement)
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("count"), (int, float))
            or entry.get("is_boundary") not in ("yes", "no")
        ):
            return False
    return True


def _format(solution: str) -> tuple[bool, bool, dict[str, Any]]:
    names = ("perception", "mode", "reasoning", "signal")
    blocks = {n: _tag(solution, n) for n in names}
    required = ("perception", "mode", "signal")
    signal_bad = (
        len(blocks["signal"]) != 1
        or not _has_exact_tag_pair(solution, "signal")
        or blocks["signal"][0].strip() not in PHASES
    )
    if any(len(blocks[name]) != 1 or not _has_exact_tag_pair(solution, name) for name in required):
        return False, signal_bad, {}
    mode = blocks["mode"][0].strip().lower()
    expected_order = ("perception", "mode", "signal") if mode == "fast" else (
        "perception", "mode", "reasoning", "signal"
    )
    positions = [solution.lower().find(f"<{name}>") for name in expected_order]
    order = all(position >= 0 for position in positions) and positions == sorted(positions)
    reasoning_contract = (
        mode == "fast" and not blocks["reasoning"] and not re.search(r"</?reasoning>", solution, flags=re.I)
    ) or (
        mode == "slow"
        and len(blocks["reasoning"]) == 1
        and _has_exact_tag_pair(solution, "reasoning")
        and bool(blocks["reasoning"][0].strip())
    )
    # The full match rejects old <current_v> blocks, code fences, and any
    # additional top-level text/tags instead of silently accepting them.
    if mode == "fast":
        envelope = r"\s*<perception>.*?</perception>\s*<mode>fast</mode>\s*<signal>.*?</signal>\s*"
    else:
        envelope = r"\s*<perception>.*?</perception>\s*<mode>slow</mode>\s*<reasoning>.*?</reasoning>\s*<signal>.*?</signal>\s*"
    exact_envelope = re.fullmatch(envelope, solution or "", flags=re.I | re.S) is not None
    if not order or not reasoning_contract or not exact_envelope:
        return False, signal_bad, {}
    perception = _json(blocks["perception"][0])
    phases = _phase_map(perception)
    valid = mode in ("fast", "slow") and _has_exact_candidate_phases(perception) and len(phases) == 4
    if valid:
        valid = all(_valid_phase_row(phases.get(phase), phase) for phase in PHASES)
    return bool(valid), signal_bad, {"mode": mode, "signal": blocks["signal"][0].strip(), "perception": perception, "reasoning": blocks["reasoning"][0] if blocks["reasoning"] else ""}


def _decision_format_valid(solution: str) -> bool:
    """Validate only the mode/reasoning/signal suffix.

    Perception structure is trained by the independent gold-CE path and must
    not turn an otherwise valid decision suffix into a GRPO format penalty.
    """
    mode_blocks = _tag(solution, "mode")
    signal_blocks = _tag(solution, "signal")
    reasoning_blocks = _tag(solution, "reasoning")
    if (
        len(mode_blocks) != 1
        or not _has_exact_tag_pair(solution, "mode")
        or len(signal_blocks) != 1
        or not _has_exact_tag_pair(solution, "signal")
        or signal_blocks[0].strip() not in PHASES
    ):
        return False

    mode = mode_blocks[0].strip().lower()
    if mode == "fast":
        if reasoning_blocks or re.search(r"</?reasoning>", solution, flags=re.I):
            return False
        suffix_pattern = r"\s*<mode>fast</mode>\s*<signal>.*?</signal>\s*"
    elif mode == "slow":
        if (
            len(reasoning_blocks) != 1
            or not _has_exact_tag_pair(solution, "reasoning")
            or not reasoning_blocks[0].strip()
        ):
            return False
        suffix_pattern = (
            r"\s*<mode>slow</mode>\s*<reasoning>.*?</reasoning>\s*"
            r"<signal>.*?</signal>\s*"
        )
    else:
        return False

    mode_start = solution.lower().find("<mode>")
    return mode_start >= 0 and re.fullmatch(
        suffix_pattern, solution[mode_start:], flags=re.I | re.S
    ) is not None


def _rank_scores(values: dict[str, float], higher_is_better: bool) -> dict[str, float]:
    if len(set(values.values())) == 1:
        return {p: 0.5 for p in PHASES}
    order = sorted(PHASES, key=lambda p: values[p], reverse=higher_is_better)
    base = (1.0, 2.0 / 3.0, 1.0 / 3.0, 0.0)
    scores: dict[str, float] = {}
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        scores_for_tie = base[i:j]
        for p in order[i:j]:
            scores[p] = sum(scores_for_tie) / len(scores_for_tie)
        i = j
    return scores


def _traffic(ground_truth: dict[str, Any], signal: str) -> float:
    """Return the selected phase's DeepSeek preference score in ``[0, 1]``."""
    if not isinstance(ground_truth, dict):
        return 0.0
    annotation = ground_truth.get("phase_preference_teacher")
    if not isinstance(annotation, dict):
        return 0.0
    scores = annotation.get("scores")
    if not isinstance(scores, dict) or set(scores) != set(PHASES) or signal not in PHASES:
        return 0.0
    if not all(
        isinstance(scores.get(phase), (int, float))
        and math.isfinite(float(scores[phase]))
        and 0.0 <= float(scores[phase]) <= 10.0
        for phase in PHASES
    ):
        return 0.0
    value = scores.get(signal)
    return float(value) / 10.0


def _traffic_state_values(value: Any) -> list[float]:
    """Flatten 11 traffic-state values per phase (44 values total)."""
    phases = _phase_map(value)
    if len(phases) != 4:
        return []
    result: list[float] = []
    for phase in PHASES:
        row = phases.get(phase)
        if not isinstance(row, dict):
            return []
        try:
            movements = MOVEMENTS[phase]
            current_v = row["current_v"]
            current_q = row["current_q"]
            arrivals = row["coordinated_arrivals"]
            result.extend([float(current_v["total"]), *(float(current_v[m]) for m in movements)])
            result.extend([float(current_q["total"]), *(float(current_q[m]) for m in movements)])
            result.extend([float(row["demand_trend_v30_minus_v5"]), float(row["queue_trend_q30_minus_q5"])])
            breakdown = arrivals["breakdown"]
            result.extend([float(arrivals["total"]), *(float(breakdown[m]["count"]) for m in movements)])
        except (KeyError, TypeError, ValueError):
            return []
    return result


def _auxiliary_perception_values(value: Any) -> tuple[list[str], list[float]]:
    """Return the eight boundary labels and four persistent-history values."""
    phases = _phase_map(value)
    if len(phases) != 4:
        return [], []
    boundaries: list[str] = []
    histories: list[float] = []
    for phase in PHASES:
        row = phases.get(phase)
        if not isinstance(row, dict):
            return [], []
        try:
            history = row["nonzero_v_history_length_since_last_service"]
            breakdown = row["coordinated_arrivals"]["breakdown"]
            if not isinstance(history, (int, float)):
                return [], []
            histories.append(float(history))
            for movement in MOVEMENTS[phase]:
                boundary = breakdown[movement]["is_boundary"]
                if boundary not in ("yes", "no"):
                    return [], []
                boundaries.append(boundary)
        except (KeyError, TypeError):
            return [], []
    return boundaries, histories


def _perception_components(prediction: Any, target: Any) -> dict[str, float | int] | None:
    """Score 44 traffic fields plus eight prepared auxiliary target fields.

    Numeric traffic fields use a smooth distance score. This preserves the
    distinction between a near miss and an all-zero prediction, while the
    auxiliary categorical/exact fields remain strict accuracies.
    """
    predicted_traffic = _traffic_state_values(prediction)
    target_traffic = _traffic_state_values(target)
    predicted_boundaries, predicted_history = _auxiliary_perception_values(prediction)
    target_boundaries, target_history = _auxiliary_perception_values(target)
    if (
        len(predicted_traffic) != 44
        or len(target_traffic) != 44
        or len(predicted_boundaries) != 8
        or len(target_boundaries) != 8
        or len(predicted_history) != 4
        or len(target_history) != 4
    ):
        return None

    nonzero_matches = [
        predicted == expected
        for predicted, expected in zip(predicted_traffic, target_traffic)
        if expected != 0
    ]
    zero_matches = [
        predicted == expected
        for predicted, expected in zip(predicted_traffic, target_traffic)
        if expected == 0
    ]
    nonzero_distances = [
        1.0 / (abs(predicted - expected) + 1.0)
        for predicted, expected in zip(predicted_traffic, target_traffic)
        if expected != 0
    ]
    zero_distances = [
        1.0 / (abs(predicted - expected) + 1.0)
        for predicted, expected in zip(predicted_traffic, target_traffic)
        if expected == 0
    ]
    # Keep exact accuracies for diagnostics, but use smooth distance scores for
    # the optimization signal. +1 avoids division by zero; it is not a +/-1
    # tolerance.
    nonzero_accuracy = sum(nonzero_matches) / len(nonzero_matches) if nonzero_matches else None
    zero_accuracy = sum(zero_matches) / len(zero_matches) if zero_matches else None
    nonzero_distance_score = sum(nonzero_distances) / len(nonzero_distances) if nonzero_distances else None
    zero_distance_score = sum(zero_distances) / len(zero_distances) if zero_distances else None
    if not PERCEPTION_SPLIT_WEIGHTING:
        all_distance_scores = nonzero_distances + zero_distances
        traffic_state_score = (
            sum(all_distance_scores) / len(all_distance_scores)
            if all_distance_scores else 0.0
        )
    elif nonzero_distance_score is None:
        traffic_state_score = float(zero_distance_score) if zero_distance_score is not None else 0.0
    elif zero_distance_score is None:
        traffic_state_score = float(nonzero_distance_score)
    else:
        traffic_state_score = 0.8 * nonzero_distance_score + 0.2 * zero_distance_score

    boundary_matches = [predicted == expected for predicted, expected in zip(predicted_boundaries, target_boundaries)]
    history_matches = [predicted == expected for predicted, expected in zip(predicted_history, target_history)]
    boundary_accuracy = sum(boundary_matches) / len(boundary_matches)
    history_accuracy = sum(history_matches) / len(history_matches)
    perception_reward = (
        0.90 * traffic_state_score
        + 0.05 * boundary_accuracy
        + 0.05 * history_accuracy
    )
    return {
        "perception_reward": perception_reward,
        "traffic_state_score": traffic_state_score,
        "perception_nonzero_accuracy": nonzero_accuracy if nonzero_accuracy is not None else zero_accuracy,
        "perception_zero_accuracy": zero_accuracy if zero_accuracy is not None else nonzero_accuracy,
        "perception_nonzero_distance_score": (
            nonzero_distance_score if nonzero_distance_score is not None else zero_distance_score
        ),
        "perception_zero_distance_score": (
            zero_distance_score if zero_distance_score is not None else nonzero_distance_score
        ),
        "perception_nonzero_correct": sum(nonzero_matches),
        "perception_nonzero_total": len(nonzero_matches),
        "perception_zero_correct": sum(zero_matches),
        "perception_zero_total": len(zero_matches),
        "boundary_correct": sum(boundary_matches),
        "boundary_total": len(boundary_matches),
        "boundary_exact_accuracy": boundary_accuracy,
        "history_correct": sum(history_matches),
        "history_total": len(history_matches),
        "history_exact_accuracy": history_accuracy,
        "perception_correct": sum(nonzero_matches) + sum(zero_matches) + sum(boundary_matches) + sum(history_matches),
        "perception_total": 56,
    }


def _perception_bad(solution: str, perception_obj: Any) -> bool:
    traffic_values = _traffic_state_values(perception_obj)
    boundaries, histories = _auxiliary_perception_values(perception_obj)
    if len(traffic_values) != 44 or len(boundaries) != 8 or len(histories) != 4:
        return True
    perception_blocks = _tag(solution, "perception")
    return len(perception_blocks) != 1


@lru_cache(maxsize=1)
def _tokenizer():
    from transformers import AutoTokenizer
    path = os.environ.get(
        "V35_GRPO_MODEL_PATH",
        "/path/to/models/v35_four_video_context_reasoning_512x960_retrain",
    )
    return AutoTokenizer.from_pretrained(path, trust_remote_code=True)


def _reasoning_penalty(mode: str, reasoning: str) -> float:
    """Penalize only slow reasoning that exceeds the 300-token budget."""
    if mode != "slow":
        return 0.0
    try:
        token_count = len(_tokenizer()(reasoning, add_special_tokens=False)["input_ids"])
    except Exception:
        token_count = len(reasoning.split())
    excess = max(0, token_count - REASONING_FREE_TOKENS)
    length_penalty = REASONING_MAX_LENGTH_PENALTY * (
        1.0 - math.exp(-excess / REASONING_TAU)
    )
    return length_penalty


def _mode_bonus(mode: str, perception_target: Any) -> tuple[float, float, float]:
    """Return legacy queue-gap diagnostics without shaping the mode reward."""
    phase_rows = _phase_map(perception_target)
    q_values = []
    for phase in PHASES:
        row = phase_rows.get(phase, {})
        current_q = row.get("current_q", {}) if isinstance(row, dict) else {}
        value = current_q.get("total") if isinstance(current_q, dict) else None
        if not isinstance(value, (int, float)):
            return 0.0, 0.0, 0.0
        q_values.append(float(value))
    ordered = sorted(q_values, reverse=True)
    q_gap = max(0.0, ordered[0] - ordered[1])
    fast_confidence = 1.0 / (
        1.0 + math.exp(-(q_gap - MODE_CONFIDENCE_CENTER) / MODE_CONFIDENCE_TEMPERATURE)
    )
    return 0.0, q_gap, fast_confidence


def _signal_counterfactual_advantage(action_rewards: dict[str, float], signal: str) -> tuple[float, float, float]:
    """Compare the sampled signal with its strongest alternative action."""
    if set(action_rewards) != set(PHASES) or signal not in action_rewards:
        return 0.0, 0.0, 0.0
    values = [float(action_rewards[phase]) for phase in PHASES]
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    std = math.sqrt(variance)
    if std <= 1e-8:
        return 0.0, mean, std
    best_alternative = max(float(action_rewards[phase]) for phase in PHASES if phase != signal)
    advantage = (float(action_rewards[signal]) - best_alternative) / std
    advantage = max(-SIGNAL_ADVANTAGE_CLIP, min(SIGNAL_ADVANTAGE_CLIP, advantage))
    return advantage, mean, std


def _invalid_result(reasoning_penalty: float, perception_target: Any) -> dict[str, float | int]:
    """Return a complete numeric result for malformed model output.

    Validation aggregates every reward-extra key across samples.  Returning
    the same schema on format failures prevents missing values from becoming
    ``None`` during metric aggregation.
    """
    target_values = _traffic_state_values(perception_target)
    nonzero_total = sum(value != 0 for value in target_values)
    zero_total = len(target_values) - nonzero_total
    return {
        "score": -1.0,
        "format_reward": 0.0,
        "format_term": -1.0,
        "base_reward": 0.0,
        "traffic_reward": 0.0,
        "perception_reward": 0.0,
        "perception_reward_weight": PERCEPTION_REWARD_WEIGHT,
        "reasoning_penalty": float(reasoning_penalty),
        "best_traffic_reward": 0.0,
        "action_reward_mean": 0.0,
        "action_reward_std": 0.0,
        "signal_counterfactual_advantage": 0.0,
        "fast_decision_penalty": 0.0,
        "q_gap": 0.0,
        "fast_confidence": 0.0,
        "mode_bonus": 0.0,
        "traffic_state_score": 0.0,
        "perception_nonzero_accuracy": 0.0,
        "perception_zero_accuracy": 0.0,
        "perception_nonzero_distance_score": 0.0,
        "perception_zero_distance_score": 0.0,
        "perception_nonzero_correct": 0,
        "perception_nonzero_total": nonzero_total,
        "perception_zero_correct": 0,
        "perception_zero_total": zero_total,
        "boundary_correct": 0,
        "boundary_total": 8,
        "boundary_exact_accuracy": 0.0,
        "history_correct": 0,
        "history_total": 4,
        "history_exact_accuracy": 0.0,
        "perception_correct": 0,
        "perception_total": 56,
        "perception_valid": 0,
        "signal_valid": 0,
    }


def compute_score(solution_str, ground_truth, **kwargs):
    gt = _json(ground_truth) or {}
    solution = solution_str or ""
    extra_info = kwargs.get("extra_info") or {}
    split = extra_info.get("split", "unknown") if isinstance(extra_info, dict) else "unknown"
    sample_id = extra_info.get("index") if isinstance(extra_info, dict) else None
    reward_context = {
        "forced_mode": extra_info.get("forced_mode") if isinstance(extra_info, dict) else kwargs.get("forced_mode"),
        "rollout_n": extra_info.get("rollout_n") if isinstance(extra_info, dict) else kwargs.get("rollout_n"),
    }
    _REWARD_CONTEXT.set(reward_context)
    parsed, signal_bad, output = _format(solution)
    mode_blocks = _tag(solution, "mode")
    reasoning_blocks = _tag(solution, "reasoning")
    mode = mode_blocks[0].strip().lower() if len(mode_blocks) == 1 else ""
    reward_context["generated_mode"] = mode or None
    reasoning = reasoning_blocks[0] if len(reasoning_blocks) == 1 else ""
    reasoning_penalty = _reasoning_penalty(mode, reasoning)

    if signal_bad:
        result = _invalid_result(reasoning_penalty, gt.get("perception_target"))
        _write_reward_log({"split": split, "sample_id": sample_id, "solution": solution,
                           "ground_truth": gt, "parsed": parsed, "mode": output.get("mode"),
                           "signal": output.get("signal"), "signal_bad": True,
                           "perception_bad": False, **result})
        return result

    if not parsed:
        perception_blocks = _tag(solution, "perception")
        signal_blocks = _tag(solution, "signal")
        output = {
            "mode": mode,
            "signal": signal_blocks[0].strip(),
            "perception": _json(perception_blocks[0]) if len(perception_blocks) == 1 else {},
            "reasoning": reasoning,
        }

    perception_bad = _perception_bad(solution, output.get("perception"))
    if perception_bad:
        result = _invalid_result(reasoning_penalty, gt.get("perception_target"))
        decision_format_valid = _decision_format_valid(solution)
        signal = output.get("signal", "")
        traffic_reward = _traffic(gt, signal)
        action_rewards = {phase: _traffic(gt, phase) for phase in PHASES}
        signal_advantage, action_reward_mean, action_reward_std = _signal_counterfactual_advantage(
            action_rewards, signal
        )
        mode_bonus, q_gap, fast_confidence = _mode_bonus(
            output.get("mode", ""), gt.get("perception_target")
        )
        format_term = 0.0 if decision_format_valid else -0.5
        base_reward = traffic_reward + mode_bonus - reasoning_penalty
        result.update({
            "score": base_reward + format_term,
            "format_reward": 1.0 if decision_format_valid else 0.0,
            "format_term": format_term,
            "base_reward": base_reward,
            "traffic_reward": traffic_reward,
            "reasoning_penalty": reasoning_penalty,
            "best_traffic_reward": max(action_rewards.values(), default=traffic_reward),
            "action_reward_mean": action_reward_mean,
            "action_reward_std": action_reward_std,
            "signal_counterfactual_advantage": signal_advantage,
            "q_gap": q_gap,
            "fast_confidence": fast_confidence,
            "mode_bonus": mode_bonus,
            "perception_valid": 0,
            "signal_valid": 1,
        })
        _write_reward_log({"split": split, "sample_id": sample_id, "solution": solution,
                           "ground_truth": gt, "parsed": parsed, "mode": output.get("mode"),
                           "signal": output.get("signal"), "signal_bad": False,
                           "perception_bad": True, **result})
        return result

    format_reward = 1.0 if parsed else 0.0
    perception_components = _perception_components(
        output.get("perception"), gt.get("perception_target")
    )
    if perception_components is None:
        perception_components = {
            "perception_reward": 0.0,
            "traffic_state_score": 0.0,
            "perception_nonzero_accuracy": 0.0,
            "perception_zero_accuracy": 0.0,
            "perception_nonzero_distance_score": 0.0,
            "perception_zero_distance_score": 0.0,
            "perception_nonzero_correct": 0,
            "perception_nonzero_total": 0,
            "perception_zero_correct": 0,
            "perception_zero_total": 0,
            "boundary_correct": 0,
            "boundary_total": 0,
            "boundary_exact_accuracy": 0.0,
            "history_correct": 0,
            "history_total": 0,
            "history_exact_accuracy": 0.0,
            "perception_correct": 0,
            "perception_total": 56,
        }
    perception_reward = float(perception_components["perception_reward"])
    traffic_reward = _traffic(gt, output["signal"])
    action_rewards = {phase: _traffic(gt, phase) for phase in PHASES}
    best_traffic_reward = max(action_rewards.values(), default=traffic_reward)
    signal_advantage, action_reward_mean, action_reward_std = _signal_counterfactual_advantage(
        action_rewards, output["signal"]
    )
    mode_bonus, q_gap, fast_confidence = _mode_bonus(
        output.get("mode", ""), gt.get("perception_target")
    )
    fast_decision_penalty = 0.0
    base_reward = (
        traffic_reward
        + mode_bonus
        - reasoning_penalty
    )
    format_term = -0.5 if not parsed else 0.0
    score = base_reward + format_term
    result = {
        "score": score,
        "format_reward": format_reward,
        "format_term": format_term,
        "base_reward": base_reward,
        "traffic_reward": traffic_reward,
        "perception_reward": perception_reward,
        "perception_reward_weight": PERCEPTION_REWARD_WEIGHT,
        "reasoning_penalty": reasoning_penalty,
        "best_traffic_reward": best_traffic_reward,
        "action_reward_mean": action_reward_mean,
        "action_reward_std": action_reward_std,
        "signal_counterfactual_advantage": signal_advantage,
        "fast_decision_penalty": fast_decision_penalty,
        "q_gap": q_gap,
        "fast_confidence": fast_confidence,
        "mode_bonus": mode_bonus,
        "perception_valid": 1,
        "signal_valid": 1,
        **perception_components,
    }
    _write_reward_log({"split": split, "sample_id": sample_id, "solution": solution,
                       "ground_truth": gt, "parsed": parsed, "mode": output.get("mode"),
                       "signal": output.get("signal"), "signal_bad": signal_bad,
                       "perception_bad": perception_bad, **result})
    return result
