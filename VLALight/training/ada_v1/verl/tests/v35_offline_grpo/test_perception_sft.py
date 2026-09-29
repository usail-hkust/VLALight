import math

import pytest

from v35_offline_grpo.perception_sft import (
    build_grpo_loss_mask,
    build_mode_selector_weights,
    build_perception_sft_row_indices,
    build_tag_content_mask,
)


class _CharTokenizer:
    @staticmethod
    def encode(text, add_special_tokens=False):
        return [ord(char) for char in text]


def test_row_indices_restore_noncontiguous_rollout_groups():
    uids = [uid for _ in range(6) for uid in ("a", "b", "c", "d")]

    gold_indices, interleaved = build_perception_sft_row_indices(uids, rollout_n=6)

    assert gold_indices == [0, 1, 2, 3]
    assert interleaved == [
        0,
        4,
        8,
        12,
        16,
        20,
        24,
        1,
        5,
        9,
        13,
        17,
        21,
        25,
        2,
        6,
        10,
        14,
        18,
        22,
        26,
        3,
        7,
        11,
        15,
        19,
        23,
        27,
    ]


def test_decision_masks_separate_mode_from_grpo_suffix():
    tokenizer = _CharTokenizer()
    response = (
        "<perception>{}</perception>"
        "<mode>slow</mode>"
        "<reasoning>compare</reasoning>"
        "<signal>ELWL</signal>"
    )
    response_ids = tokenizer.encode(response)

    grpo_mask = build_grpo_loss_mask(response_ids, tokenizer)
    mode_mask = build_tag_content_mask(response_ids, tokenizer, "mode")
    signal_mask = build_tag_content_mask(response_ids, tokenizer, "signal")

    mode_start = response.index("slow")
    signal_start = response.index("ELWL")
    suffix_start = response.index("</mode>") + len("</mode>")
    assert grpo_mask[:suffix_start] == [0] * suffix_start
    assert all(grpo_mask[suffix_start:])
    assert [response[index] for index, value in enumerate(mode_mask) if value] == list("slow")
    assert [response[index] for index, value in enumerate(signal_mask) if value] == list("ELWL")
    assert not grpo_mask[mode_start]
    assert grpo_mask[signal_start]


def test_mode_selector_uses_balanced_group_outcomes():
    uids = ["a"] * 6 + ["b"] * 6
    reward_infos = []
    for fast_reward, slow_reward in ((0.4, 1.0), (1.0, 0.4)):
        reward_infos.extend(
            [{"forced_mode": "fast", "traffic_reward": fast_reward, "reasoning_penalty": 0.0}] * 3
        )
        reward_infos.extend(
            [{"forced_mode": "slow", "traffic_reward": slow_reward, "reasoning_penalty": 0.0}] * 3
        )

    prefixes = [(1, 2)] * 6 + [(3, 4)] * 6
    weights, metrics, active, details = build_mode_selector_weights(
        uids,
        reward_infos,
        rollout_n=6,
        temperature=0.2,
        min_probability=0.1,
        mode_token_counts=[1] * 12,
        mode_prefixes=prefixes,
    )

    expected = [0.2] * 3 + [1.8] * 3 + [1.8] * 3 + [0.2] * 3
    assert all(math.isclose(actual, target) for actual, target in zip(weights, expected, strict=True))
    assert all(active)
    assert details["a"]["valid"] is True
    assert math.isclose(metrics["fast_mean_traffic_utility"], 0.7)
    assert math.isclose(metrics["slow_mean_traffic_utility"], 0.7)
    assert math.isclose(metrics["mode_target_slow_probability"], 0.5)
    assert math.isclose(metrics["slow_better_group_ratio"], 0.5)


def test_mode_weights_are_exactly_equivalent_to_same_prefix_soft_ce():
    p_target = 1.0 / (1.0 + math.exp(-1.0))
    fast_nll = 0.7
    slow_nll = 0.2
    reward_infos = [
        *[{"forced_mode": "fast", "traffic_reward": 0.0}] * 3,
        *[{"forced_mode": "slow", "traffic_reward": 1.0}] * 3,
    ]
    weights, _, active, details = build_mode_selector_weights(
        ["scene"] * 6,
        reward_infos,
        rollout_n=6,
        temperature=1.0,
        min_probability=0.2,
        mode_token_counts=[1] * 6,
        mode_prefixes=[(1, 2, 3)] * 6,
    )

    weighted_token_mean = (
        sum(weight * fast_nll for weight in weights[:3])
        + sum(weight * slow_nll for weight in weights[3:])
    ) / 6
    expected_soft_ce = (1.0 - p_target) * fast_nll + p_target * slow_nll

    assert all(active)
    assert math.isclose(details["scene"]["p_target_slow"], p_target)
    assert math.isclose(weighted_token_mean, expected_soft_ce)


def test_mode_selector_keeps_invalid_branch_for_forced_mode_credit():
    reward_infos = [
        *[
            {"forced_mode": "fast", "traffic_reward": 0.8, "format_reward": 1.0, "signal_valid": 1}
            for _ in range(3)
        ],
        *[
            {"forced_mode": "slow", "traffic_reward": 0.0, "format_reward": 0.0, "signal_valid": 0}
            for _ in range(3)
        ],
    ]
    weights, metrics, active, details = build_mode_selector_weights(
        ["a"] * 6,
        reward_infos,
        rollout_n=6,
        temperature=0.2,
        min_probability=0.2,
        mode_token_counts=[1] * 6,
        mode_prefixes=[(1, 2)] * 6,
    )

    # The format-invalid SLOW branch stays in the forced 3:3 comparison. In
    # production its format_penalty is part of the online dimensions.
    assert all(active)
    assert all(weight > 0.0 for weight in weights)
    assert metrics["mode_target_slow_probability"] == pytest.approx(0.2)
    assert metrics["mode_target_neutral_group_ratio"] == 0.0
    assert details["a"]["valid"] is True


def test_mode_selector_requires_same_perception_prefix():
    reward_infos = [
        *[{"forced_mode": "fast", "traffic_reward": 0.4}] * 3,
        *[{"forced_mode": "slow", "traffic_reward": 0.8}] * 3,
    ]
    weights, _, active, _ = build_mode_selector_weights(
        ["a"] * 6,
        reward_infos,
        rollout_n=6,
        temperature=0.2,
        min_probability=0.2,
        mode_token_counts=[1] * 6,
        mode_prefixes=[(1,)] * 3 + [(2,)] * 3,
    )

    assert weights == [0.0] * 6
    assert active == [False] * 6


def test_mode_selector_diagnostics_keep_malformed_rows_visible():
    rewards = [
        *[{"forced_mode": "fast", "network_reward": 1.0, "format_valid": True}] * 3,
        *[{"forced_mode": "slow", "network_reward": 0.0, "format_valid": False}] * 3,
    ]
    _, metrics, active, _ = build_mode_selector_weights(
        ["scene"] * 6,
        rewards,
        rollout_n=6,
        temperature=0.2,
        min_probability=0.2,
        mode_token_counts=[1, 1, 1, 1, 1, 0],
        mode_prefixes=[(1, 2)] * 6,
    )

    assert all(active)
    assert metrics["mode_fast_row_count"] == 3.0
    assert metrics["mode_slow_row_count"] == 3.0
    assert metrics["mode_malformed_row_ratio"] == pytest.approx(0.5)
    assert metrics["mode_zero_token_row_ratio"] == pytest.approx(1.0 / 6.0)
