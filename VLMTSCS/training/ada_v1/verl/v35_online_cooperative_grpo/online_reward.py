"""Reward components for online cooperative Stage 2 GRPO."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping
from typing import Any

from .stage2_protocol import DecisionResponse, parse_decision_response


def endpoint_score(before: float, after: float) -> float:
    """Scale-free queue improvement in [0, 1].

    0.5 means unchanged, 1.0 means cleared, and values below 0.5 mean
    congestion increased.  An empty-to-empty intersection is perfect.
    """
    before, after = max(0.0, float(before)), max(0.0, float(after))
    total = before + after
    return 1.0 if total == 0.0 else before / total


def compute_endpoint_scores(
    before: Mapping[str, float] | Any,
    after: Mapping[str, float] | Any,
    target_id: str,
) -> tuple[float, float]:
    """Return ``(global_score, local_score)`` using endpoint queue states."""
    before_map, after_map = _as_queue_map(before), _as_queue_map(after)
    if set(before_map) - set(after_map):
        raise ValueError("after queue metrics miss intersections")
    if target_id not in before_map or target_id not in after_map:
        raise KeyError(f"target intersection {target_id!r} is absent from queue metrics")
    global_score = endpoint_score(sum(before_map.values()), sum(after_map.values()))
    local_score = endpoint_score(before_map[target_id], after_map[target_id])
    return global_score, local_score


def mode_utility(
    local_score: float,
    global_score: float,
    reasoning_penalty_value: float,
    *,
    local_weight: float = 0.8,
    global_weight: float = 0.2,
) -> float:
    """Utility used by the forced 3:3 mode selector."""
    if local_weight < 0 or global_weight < 0 or local_weight + global_weight <= 0:
        raise ValueError("mode reward weights must be non-negative and non-zero")
    norm = local_weight + global_weight
    return (local_weight * float(local_score) + global_weight * float(global_score)) / norm - float(
        reasoning_penalty_value
    )


def count_reasoning_tokens(reasoning: str, tokenizer: Any = None) -> int:
    if not reasoning:
        return 0
    if tokenizer is not None:
        try:
            encoded = tokenizer.encode(reasoning, add_special_tokens=False)
        except TypeError:
            encoded = tokenizer.encode(reasoning)
        return len(encoded)
    return len(re.findall(r"\S+", reasoning))


def reasoning_penalty(
    token_count: int,
    *,
    free_tokens: int = 0,
    tau: float = 400.0,
    max_penalty: float = 0.10,
) -> float:
    """Smooth cost; with ``free_tokens=0`` every reasoning token contributes."""
    if token_count < 0 or free_tokens < 0 or tau <= 0 or max_penalty < 0:
        raise ValueError("invalid reasoning penalty parameters")
    excess = max(0, int(token_count) - int(free_tokens))
    return float(max_penalty * (1.0 - math.exp(-excess / float(tau))))


def _as_queue_map(values: Mapping[str, float] | Any) -> dict[str, float]:
    if isinstance(values, Mapping):
        return {str(key): float(value) for key, value in values.items()}
    if hasattr(values, "by_intersection"):
        return _as_queue_map(values.by_intersection)
    raise TypeError("queue metrics must be a mapping or expose by_intersection")


def compute_queue_rewards(
    before: Mapping[str, float] | Any,
    after: Mapping[str, float] | Any,
    target_id: str,
    *,
    scale: float = 1.0,
    global_scale: float | None = None,
    local_scale: float | None = None,
) -> tuple[float, float]:
    """Return ``(global_queue_reward, local_queue_reward)`` as queue reduction."""
    global_scale = scale if global_scale is None else global_scale
    local_scale = scale if local_scale is None else local_scale
    if global_scale <= 0 or local_scale <= 0:
        raise ValueError("queue reward scales must be positive")
    before_map, after_map = _as_queue_map(before), _as_queue_map(after)
    missing = set(before_map) - set(after_map)
    if missing:
        raise ValueError(f"after queue metrics miss intersections: {sorted(missing)}")
    global_reward = (sum(before_map.values()) - sum(after_map.values())) / global_scale
    if target_id not in before_map or target_id not in after_map:
        raise KeyError(f"target intersection {target_id!r} is absent from queue metrics")
    local_reward = (before_map[target_id] - after_map[target_id]) / local_scale
    return float(global_reward), float(local_reward)


def compute_online_reward(
    response: str | DecisionResponse,
    *,
    before_queues: Mapping[str, float] | Any,
    after_queues: Mapping[str, float] | Any,
    target_id: str,
    tokenizer: Any = None,
    forced_mode: str | None = None,
    global_weight: float = 0.5,
    local_weight: float = 1.0,
    reasoning_weight: float = 0.5,
    queue_scale: float = 1.0,
    global_queue_scale: float | None = None,
    local_queue_scale: float | None = None,
    reasoning_free_tokens: int = 0,
    reasoning_tau: float = 400.0,
    reasoning_max_penalty: float = 0.10,
) -> dict[str, Any]:
    parsed = response if isinstance(response, DecisionResponse) else parse_decision_response(response, forced_mode=forced_mode)
    global_reward, local_reward = compute_queue_rewards(
        before_queues, after_queues, target_id, scale=queue_scale,
        global_scale=global_queue_scale, local_scale=local_queue_scale,
    )
    tokens = count_reasoning_tokens(parsed.reasoning, tokenizer)
    penalty = reasoning_penalty(
        tokens, free_tokens=reasoning_free_tokens, tau=reasoning_tau, max_penalty=reasoning_max_penalty
    )
    cost_reward = -penalty
    global_score, local_score = compute_endpoint_scores(
        before_queues, after_queues, target_id
    )
    score = (
        float(global_weight) * global_reward
        + float(local_weight) * local_reward
        + float(reasoning_weight) * cost_reward
        + parsed.format_penalty
    )
    return {
        "score": float(score),
        "global_queue_reward": global_reward,
        "local_queue_reward": local_reward,
        "reasoning_cost_reward": cost_reward,
        "reasoning_penalty": penalty,
        "format_penalty": parsed.format_penalty,
        "format_reward": parsed.format_penalty,
        "signal_valid": int(parsed.signal_valid),
        "format_valid": int(parsed.format_valid),
        "mode": parsed.mode,
        "forced_mode": forced_mode,
        "signal": parsed.signal,
        "reasoning_tokens": tokens,
        "global_score": global_score,
        "local_score": local_score,
        "mode_utility": mode_utility(local_score, global_score, penalty),
    }
