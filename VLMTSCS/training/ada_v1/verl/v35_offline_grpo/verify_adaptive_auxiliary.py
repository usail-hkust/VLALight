#!/usr/bin/env python3
"""CPU-only contract check for V35 signal and mode auxiliary losses."""

from __future__ import annotations

import math

from perception_sft import build_grpo_loss_mask, build_mode_selector_weights, build_tag_content_mask
from v35_offline_grpo_reward import _signal_counterfactual_advantage


class CharTokenizer:
    @staticmethod
    def encode(text, add_special_tokens=False):
        return [ord(char) for char in text]


def main() -> None:
    tokenizer = CharTokenizer()
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
    assert [response[index] for index, value in enumerate(mode_mask) if value] == list("slow")
    assert [response[index] for index, value in enumerate(signal_mask) if value] == list("ELWL")
    assert not any(grpo_mask[index] for index, value in enumerate(mode_mask) if value)
    assert all(grpo_mask[index] for index, value in enumerate(signal_mask) if value)

    action_rewards = {"ETWT": 0.0, "NTST": 0.4, "ELWL": 1.0, "NLSL": 0.4}
    best_advantage, mean, std = _signal_counterfactual_advantage(action_rewards, "ELWL")
    worst_advantage, _, _ = _signal_counterfactual_advantage(action_rewards, "ETWT")
    assert best_advantage > 0.0 and worst_advantage < 0.0
    assert math.isclose(mean, 0.45) and std > 0.0

    # A non-optimal action must still receive a negative update when its reward
    # happens to equal the four-action mean.
    mean_action_rewards = {"ETWT": 1.0, "NTST": 1.0 / 3.0, "ELWL": 0.0, "NLSL": 0.0}
    mean_action_advantage, _, _ = _signal_counterfactual_advantage(mean_action_rewards, "NTST")
    assert mean_action_advantage < 0.0

    uids = ["sample"] * 6
    reward_infos = [
        *[
            {"forced_mode": "fast", "traffic_reward": 0.4, "reasoning_penalty": 0.0}
            for _ in range(3)
        ],
        *[
            {"forced_mode": "slow", "traffic_reward": 1.0, "reasoning_penalty": 0.0}
            for _ in range(3)
        ],
    ]
    weights, metrics, active, _ = build_mode_selector_weights(
        uids,
        reward_infos,
        rollout_n=6,
        temperature=0.2,
        min_probability=0.1,
        mode_token_counts=[1] * 6,
        mode_prefixes=[(1, 2)] * 6,
    )
    assert all(math.isclose(weight, 0.2) for weight in weights[:3])
    assert all(math.isclose(weight, 1.8) for weight in weights[3:])
    assert all(active)
    assert metrics["slow_better_group_ratio"] == 1.0
    print(
        {
            "status": "PASS",
            "signal_best_advantage": best_advantage,
            "signal_worst_advantage": worst_advantage,
            "signal_mean_action_advantage": mean_action_advantage,
            **metrics,
        }
    )


if __name__ == "__main__":
    main()
