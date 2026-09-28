#!/usr/bin/env python3
"""Verify the documented V35 reward equation and format gate."""

import math

import v35_offline_grpo_reward as reward


def _phase_row(phase: str, total: int) -> dict:
    first, second = reward.MOVEMENTS[phase]
    return {
        "current_v": {"total": total, first: total, second: 0},
        "current_q": {"total": 0, first: 0, second: 0},
        "demand_trend_v30_minus_v5": 0,
        "queue_trend_q30_minus_q5": 0,
        "coordinated_arrivals": {
            "total": 0,
            "breakdown": {
                first: {"count": 0, "is_boundary": "no"},
                second: {"count": 0, "is_boundary": "no"},
            },
        },
        "nonzero_v_history_length_since_last_service": 0,
    }


def _solution(mode: str, totals: dict[str, int], reasoning: str = "") -> str:
    import json

    perception = {
        "candidate_phases": [
            {"signal": phase, **_phase_row(phase, totals[phase])}
            for phase in reward.PHASES
        ]
    }
    reasoning_block = f"<reasoning>{reasoning}</reasoning>" if mode == "slow" else ""
    return (
        f"<perception>{json.dumps(perception)}</perception>"
        f"<mode>{mode}</mode>{reasoning_block}"
        "<signal>ETWT</signal>"
    )


def main() -> None:
    counterfactual_rewards = {"ETWT": 0.0, "NTST": 0.4, "ELWL": 1.0, "NLSL": 0.4}
    best_advantage, action_mean, action_std = reward._signal_counterfactual_advantage(
        counterfactual_rewards, "ELWL"
    )
    worst_advantage, _, _ = reward._signal_counterfactual_advantage(counterfactual_rewards, "ETWT")
    assert best_advantage > 0.0
    assert worst_advantage < 0.0
    assert math.isclose(action_mean, 0.45)
    assert action_std > 0.0

    target = {"ETWT": 4, "NTST": 3, "ELWL": 2, "NLSL": 1}
    perception_target = {
        "candidate_phases": [
            {"signal": phase, **_phase_row(phase, target[phase])}
            for phase in reward.PHASES
        ]
    }
    ground_truth = {
        "perception_target": perception_target,
        "phase_preference_teacher": {
            "scores": {"ETWT": 10.0, "NTST": 4.0, "ELWL": 8.0, "NLSL": 2.0}
        },
        "actions": {
            "ETWT": {"remaining_v_30s": 1, "remaining_queue_30s": 1, "discharged_30s": 4},
            "NTST": {"remaining_v_30s": 2, "remaining_queue_30s": 2, "discharged_30s": 3},
            "ELWL": {"remaining_v_30s": 3, "remaining_queue_30s": 3, "discharged_30s": 2},
            "NLSL": {"remaining_v_30s": 4, "remaining_queue_30s": 4, "discharged_30s": 1},
        },
    }

    fast_result = reward.compute_score(_solution("fast", target), ground_truth)
    fast_score = fast_result["score"]
    fast_mode_bonus, _, _ = reward._mode_bonus("fast", perception_target)
    expected_teacher_score = 1.0
    assert math.isclose(
        fast_score,
        expected_teacher_score + fast_mode_bonus,
    ), fast_score
    assert fast_result["format_reward"] == 1.0
    assert math.isclose(fast_result["traffic_reward"], expected_teacher_score)
    assert fast_result["mode_bonus"] == 0.0
    assert fast_result["signal_counterfactual_advantage"] > 0.0
    assert fast_result["perception_valid"] == 1
    assert fast_result["signal_valid"] == 1
    assert fast_result["perception_reward"] == 1.0
    assert fast_result["perception_reward_weight"] == 0.0
    assert fast_result["traffic_state_score"] == 1.0
    assert fast_result["boundary_exact_accuracy"] == 1.0
    assert fast_result["history_exact_accuracy"] == 1.0
    assert fast_result["perception_total"] == 56
    assert fast_result["reasoning_penalty"] == 0.0

    zeros = {phase: 0 for phase in reward.PHASES}
    zero_result = reward.compute_score(_solution("fast", zeros), ground_truth)
    assert math.isclose(zero_result["score"], fast_score), (zero_result, fast_result)
    assert math.isclose(
        zero_result["signal_counterfactual_advantage"],
        fast_result["signal_counterfactual_advantage"],
    )
    # No +/-1 tolerance: exact-match diagnostics stay at zero, while the
    # optimization reward gives partial credit according to numeric distance.
    assert zero_result["perception_nonzero_correct"] == 0
    assert math.isclose(zero_result["perception_nonzero_distance_score"], 77.0 / 240.0), zero_result
    assert math.isclose(zero_result["perception_zero_distance_score"], 1.0), zero_result
    if reward.PERCEPTION_SPLIT_WEIGHTING:
        assert math.isclose(zero_result["traffic_state_score"], 137.0 / 300.0), zero_result
        assert math.isclose(zero_result["perception_reward"], 0.511), zero_result
    else:
        expected_unweighted = (
            zero_result["perception_nonzero_distance_score"] * zero_result["perception_nonzero_total"]
            + zero_result["perception_zero_distance_score"] * zero_result["perception_zero_total"]
        ) / 44.0
        assert math.isclose(zero_result["traffic_state_score"], expected_unweighted), zero_result

    near = {phase: target[phase] - 1 for phase in reward.PHASES}
    near_result = reward.compute_score(_solution("fast", near), ground_truth)
    assert near_result["perception_nonzero_correct"] == 0
    assert near_result["perception_nonzero_distance_score"] > zero_result["perception_nonzero_distance_score"]
    assert near_result["perception_reward"] > zero_result["perception_reward"]

    boundary_miss = _solution("fast", target).replace('"is_boundary": "no"', '"is_boundary": "yes"', 1)
    boundary_result = reward.compute_score(boundary_miss, ground_truth)
    assert math.isclose(boundary_result["boundary_exact_accuracy"], 7.0 / 8.0), boundary_result
    assert math.isclose(boundary_result["perception_reward"], 0.99375), boundary_result

    history_miss = _solution("fast", target).replace(
        '"nonzero_v_history_length_since_last_service": 0',
        '"nonzero_v_history_length_since_last_service": 1',
        1,
    )
    history_result = reward.compute_score(history_miss, ground_truth)
    assert math.isclose(history_result["history_exact_accuracy"], 3.0 / 4.0), history_result
    assert math.isclose(history_result["perception_reward"], 0.9875), history_result

    class FixedTokenizer:
        def __call__(self, _text, add_special_tokens=False):
            return {"input_ids": list(range(300))}

    reward._tokenizer = lambda: FixedTokenizer()
    slow_result = reward.compute_score(_solution("slow", target, "concise reasoning"), ground_truth)
    slow_score = slow_result["score"]
    slow_mode_bonus, _, _ = reward._mode_bonus("slow", perception_target)
    expected = (
        expected_teacher_score
        + slow_mode_bonus
    )
    assert math.isclose(slow_score, expected), (slow_score, expected)

    class LongTokenizer:
        def __call__(self, _text, add_special_tokens=False):
            return {"input_ids": list(range(500))}

    reward._tokenizer = lambda: LongTokenizer()
    long_result = reward.compute_score(_solution("slow", target, "long reasoning"), ground_truth)
    expected_long_penalty = reward.REASONING_MAX_LENGTH_PENALTY * (
        1.0 - math.exp(-(500 - reward.REASONING_FREE_TOKENS) / reward.REASONING_TAU)
    )
    expected_long = (
        expected_teacher_score
        + slow_mode_bonus
        - expected_long_penalty
    )
    assert math.isclose(long_result["score"], expected_long), (long_result, expected_long)

    misplaced = _solution("slow", target, "x").replace(
        "<reasoning>x</reasoning>", ""
    ).replace("<signal>ETWT</signal>", "<signal>ETWT</signal><reasoning>x</reasoning>")
    invalid_result = reward.compute_score(misplaced, ground_truth)
    invalid_score = invalid_result["score"]
    expected_invalid = (
        expected_teacher_score
        + slow_mode_bonus
        - expected_long_penalty
        - 0.5
    )
    assert math.isclose(invalid_score, expected_invalid), (invalid_score, expected_invalid)
    assert invalid_result["format_reward"] == 0.0

    extra_current_v = _solution("fast", target).replace(
        "<signal>", "<current_v>{}</current_v><signal>"
    )
    extra_result = reward.compute_score(extra_current_v, ground_truth)
    expected_extra = (
        expected_teacher_score
        + fast_mode_bonus
        - 0.5
    )
    assert math.isclose(extra_result["score"], expected_extra), extra_result
    assert extra_result["format_term"] == -0.5

    missing_perception_close = _solution("fast", target).replace("</perception>", "", 1)
    missing_close_result = reward.compute_score(missing_perception_close, ground_truth)
    assert math.isclose(missing_close_result["score"], expected_teacher_score + fast_mode_bonus), missing_close_result
    assert missing_close_result["format_term"] == 0.0
    assert missing_close_result["format_reward"] == 1.0
    assert missing_close_result["perception_valid"] == 0
    assert missing_close_result["signal_valid"] == 1

    duplicated = _solution("fast", target).replace(
        '"signal": "NLSL"', '"signal": "ELWL"'
    )
    duplicate_result = reward.compute_score(duplicated, ground_truth)
    assert math.isclose(duplicate_result["score"], expected_teacher_score + fast_mode_bonus), duplicate_result
    assert duplicate_result["format_term"] == 0.0
    assert duplicate_result["format_reward"] == 1.0
    assert duplicate_result["perception_valid"] == 0
    assert duplicate_result["signal_valid"] == 1
    print("PASS: contract penalties and base reward formula.")


if __name__ == "__main__":
    main()
