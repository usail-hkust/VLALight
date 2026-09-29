"""Build V35 perception-only SFT targets and segmented rollout masks."""

from __future__ import annotations

import json
import logging
import math
import re
from copy import deepcopy
from typing import Any


PHASES = ("ETWT", "NTST", "ELWL", "NLSL")
MOVEMENTS = {
    "ETWT": ("ET", "WT"),
    "NTST": ("NT", "ST"),
    "ELWL": ("EL", "WL"),
    "NLSL": ("NL", "SL"),
}
ALLOWED_LANES = {
    "ETWT": "East straight and West straight lanes",
    "NTST": "North straight and South straight lanes",
    "ELWL": "East left and West left lanes",
    "NLSL": "North left and South left lanes",
}
CURRENT_PHASE_PATTERN = re.compile(r"\bCurrent phase:\s*(ETWT|NTST|ELWL|NLSL)\b", re.I)
JSON_CURRENT_PHASE_PATTERN = re.compile(
    r'"current_phase"\s*:\s*"(ETWT|NTST|ELWL|NLSL)"', re.I
)
PERCEPTION_DECISION_TOKEN_WEIGHT = 2.0
_JSON_NUMBER_PATTERN = re.compile(r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?")
_DECISION_OBJECT_PATTERN = re.compile(
    r'"(?:current_v|current_q)"\s*:\s*\{(?P<body>[^{}]*)\}', re.S
)
_DECISION_SCALAR_PATTERN = re.compile(
    r'"(?:demand_trend_v30_minus_v5|queue_trend_q30_minus_q5|'
    r'nonzero_v_history_length_since_last_service|total|count)"\s*:\s*'
    r'(?P<value>-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?)'
)
logger = logging.getLogger(__name__)


def _as_python(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def _message_text(messages: Any) -> str:
    messages = _as_python(messages)
    if not isinstance(messages, (list, tuple)):
        return ""
    chunks: list[str] = []
    for message in messages:
        message = _as_python(message)
        if not isinstance(message, dict):
            continue
        content = _as_python(message.get("content"))
        if isinstance(content, str):
            chunks.append(content)
        elif isinstance(content, (list, tuple)):
            for part in content:
                part = _as_python(part)
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    chunks.append(part["text"])
    return "\n".join(chunks)


def build_perception_sft_row_indices(uids: Any, rollout_n: int) -> tuple[list[int], list[int]]:
    """Return gold source rows and group-preserving rollout/gold row order."""
    if rollout_n <= 0:
        raise ValueError(f"rollout_n must be positive, got {rollout_n}")

    groups: dict[str, list[int]] = {}
    for index, raw_uid in enumerate(uids):
        uid = str(_as_python(raw_uid))
        groups.setdefault(uid, []).append(index)
    if not groups:
        raise ValueError("perception SFT requires at least one uid group")

    bad_groups = {uid: len(indices) for uid, indices in groups.items() if len(indices) != rollout_n}
    if bad_groups:
        preview = dict(list(bad_groups.items())[:5])
        raise ValueError(f"perception SFT expected {rollout_n} rollouts per uid, got {preview}")

    rollout_count = sum(len(indices) for indices in groups.values())
    gold_indices: list[int] = []
    interleaved_indices: list[int] = []
    for gold_offset, indices in enumerate(groups.values()):
        gold_indices.append(indices[0])
        interleaved_indices.extend(indices)
        interleaved_indices.append(rollout_count + gold_offset)
    return gold_indices, interleaved_indices


def _ground_truth(reward_model: Any) -> dict[str, Any]:
    reward_model = _as_python(reward_model)
    if not isinstance(reward_model, dict):
        raise ValueError("perception SFT requires reward_model")
    value = _as_python(reward_model.get("ground_truth"))
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError("perception SFT requires reward_model.ground_truth")
    return value


def build_perception_target(reward_model: Any, raw_prompt: Any) -> str:
    """Return the canonical perception-only assistant continuation."""
    ground_truth = _ground_truth(reward_model)
    stage1_target = ground_truth.get("stage1_target")
    if isinstance(stage1_target, dict):
        if set(stage1_target.get("phases", {})) != set(PHASES):
            raise ValueError("online Stage 1 target must contain exactly four phases")
        return "<perception>\n" + json.dumps(
            stage1_target, ensure_ascii=False, indent=2
        ) + "\n</perception>"
    target = ground_truth.get("perception_target")
    if not isinstance(target, dict) or set(target) != set(PHASES):
        raise ValueError("perception SFT target must contain exactly four phases")

    prompt_text = _message_text(raw_prompt)
    match = CURRENT_PHASE_PATTERN.search(prompt_text)
    if match is None:
        # Online SUMO prompts carry the same value in their structured local
        # observation rather than the legacy prose header.
        match = JSON_CURRENT_PHASE_PATTERN.search(prompt_text)
    if match is None:
        raise ValueError("could not find 'Current phase' in the V35 prompt")

    candidate_phases = []
    for phase in PHASES:
        row = deepcopy(target[phase])
        if not isinstance(row, dict):
            raise ValueError(f"invalid perception target row for {phase}")
        candidate_phases.append(
            {
                "signal": phase,
                "allowed_lanes": ALLOWED_LANES[phase],
                "current_v": row["current_v"],
                "current_q": row["current_q"],
                "demand_trend_v30_minus_v5": row["demand_trend_v30_minus_v5"],
                "queue_trend_q30_minus_q5": row["queue_trend_q30_minus_q5"],
                "coordinated_arrivals": row["coordinated_arrivals"],
                "nonzero_v_history_length_since_last_service": row[
                    "nonzero_v_history_length_since_last_service"
                ],
            }
        )

    payload = {
        "current_phase": match.group(1).upper(),
        "candidate_phases": candidate_phases,
    }
    return "<perception>\n" + json.dumps(payload, ensure_ascii=False, indent=2) + "\n</perception>"


def build_perception_sft_token_weights(target_text: str, tokenizer) -> list[float]:
    """Return token weights for the unchanged perception target.

    Decision-relevant numeric leaves receive weight 2.0. All schema, text,
    and non-decision tokens receive weight 1.0. The coordination breakdown
    counts remain supervised and are treated as decision-relevant numbers;
    ``is_boundary`` is intentionally left at the default weight.
    """
    target_ids = tokenizer.encode(target_text, add_special_tokens=False)
    weights = [1.0] * len(target_ids)
    try:
        encoded = tokenizer(
            target_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        encoded_ids = encoded["input_ids"]
        offsets = encoded["offset_mapping"]
        if encoded_ids and isinstance(encoded_ids[0], (list, tuple)):
            encoded_ids = encoded_ids[0]
            offsets = offsets[0]
        if list(encoded_ids) != list(target_ids) or len(offsets) != len(target_ids):
            raise ValueError("tokenizer encode and offset-mapping tokenization disagree")
    except (KeyError, TypeError, ValueError, NotImplementedError) as exc:
        raise ValueError(
            "Perception token weighting requires a fast tokenizer with aligned offset mapping"
        ) from exc

    decision_spans: list[tuple[int, int]] = []
    for match in _DECISION_OBJECT_PATTERN.finditer(target_text):
        body_start = match.start("body")
        decision_spans.extend(
            (body_start + number.start(), body_start + number.end())
            for number in _JSON_NUMBER_PATTERN.finditer(match.group("body"))
        )
    for match in _DECISION_SCALAR_PATTERN.finditer(target_text):
        decision_spans.append((match.start("value"), match.end("value")))

    # Every decision value must be represented by at least one tokenizer
    # offset.  Failing closed prevents a tokenizer/offset mismatch from
    # silently disabling the requested weighting for a field.
    for span_start, span_end in decision_spans:
        if not any(
            token_start < span_end and token_end > span_start
            for token_start, token_end in offsets
            if token_end > token_start
        ):
            raise ValueError(
                "perception decision span is not covered by tokenizer offsets: "
                f"[{span_start}, {span_end})"
            )

    for index, (start, end) in enumerate(offsets):
        if end <= start:
            continue
        if any(start < span_end and end > span_start for span_start, span_end in decision_spans):
            weights[index] = PERCEPTION_DECISION_TOKEN_WEIGHT
    return weights


def find_subsequence(values: list[int], needle: list[int]) -> int | None:
    if not needle or len(needle) > len(values):
        return None
    limit = len(values) - len(needle) + 1
    for index in range(limit):
        if values[index : index + len(needle)] == needle:
            return index
    return None


def build_tag_content_mask(response_ids: list[int], tokenizer, tag: str) -> list[int]:
    """Return a mask covering only the content inside one XML-style tag."""
    open_tokens = tokenizer.encode(f"<{tag}>", add_special_tokens=False)
    close_tokens = tokenizer.encode(f"</{tag}>", add_special_tokens=False)
    start = find_subsequence(response_ids, open_tokens)
    if start is None:
        return [0] * len(response_ids)
    content_start = start + len(open_tokens)
    relative_end = find_subsequence(response_ids[content_start:], close_tokens)
    if relative_end is None:
        return [0] * len(response_ids)
    content_end = content_start + relative_end
    if content_end <= content_start:
        return [0] * len(response_ids)
    return [0] * content_start + [1] * (content_end - content_start) + [0] * (len(response_ids) - content_end)


def build_grpo_loss_mask(response_ids: list[int], tokenizer) -> list[int]:
    """Mask policy gradients to the decision suffix after ``</mode>``.

    Malformed outputs without a mode tag retain a full mask so the format
    penalty can suppress the sampled sequence. Gold perception CE supplies the
    positive correction for the perception prefix. The mode content is trained
    separately from the FAST/SLOW group comparison, so rollout-level traffic
    advantages must not leak into the mode selector.
    """
    mode_end_tokens = tokenizer.encode("</mode>", add_special_tokens=False)
    end = find_subsequence(response_ids, mode_end_tokens)
    start = None if end is None else end + len(mode_end_tokens)
    if start is None:
        return [1] * len(response_ids)
    return [0] * start + [1] * (len(response_ids) - start)


def _mode_row_is_valid(info: dict[str, Any]) -> bool:
    """A forced route stays eligible even when its decision suffix is invalid.

    The mode target is supplied by the balanced forced assignment, while the
    format penalty is one of its GDPO reward dimensions. Excluding malformed
    rows therefore hides precisely the outcome that the router must learn to
    avoid. Rows without a mode token are filtered separately by the caller.
    """
    return True


def build_mode_selector_weights(
    uids: list[str],
    reward_infos: list[dict[str, Any]],
    *,
    rollout_n: int,
    temperature: float,
    min_probability: float,
    mode_token_counts: list[int] | None = None,
    mode_prefixes: list[tuple[int, ...]] | None = None,
) -> tuple[list[float], dict[str, float], list[bool], dict[str, dict[str, float | bool]]]:
    """Build exact same-prefix soft-CE weights from forced FAST/SLOW outcomes."""
    if len(uids) != len(reward_infos):
        raise ValueError("mode selector requires one reward record per uid")
    if rollout_n < 2 or rollout_n % 2:
        # A single rollout cannot provide a FAST-vs-SLOW counterfactual.  It
        # is still useful for decision/SFT learning, so disable only the mode
        # auxiliary loss instead of aborting the whole V1 step.
        group_count = len({str(uid) for uid in uids})
        metrics = {
            "fast_mean_traffic_utility": 0.0,
            "slow_mean_traffic_utility": 0.0,
            "mode_target_slow_probability": 0.5,
            "slow_better_group_ratio": 0.0,
            "mode_target_neutral_group_ratio": 1.0,
            "mode_active_row_ratio": 0.0,
            "mode_group_count": float(group_count),
            "mode_complete_prefix_group_ratio": 0.0,
            "mode_fast_row_count": 0.0,
            "mode_slow_row_count": 0.0,
            "mode_malformed_row_ratio": 0.0,
            "mode_zero_token_row_ratio": 0.0,
            "mode_disabled_incomplete_rollout_ratio": 1.0,
        }
        return [0.0] * len(uids), metrics, [False] * len(uids), {}
    if temperature <= 0:
        raise ValueError(f"mode selector temperature must be positive, got {temperature}")
    if not 0.0 <= min_probability < 0.5:
        raise ValueError(f"mode selector min_probability must be in [0, 0.5), got {min_probability}")
    if mode_token_counts is not None and len(mode_token_counts) != len(uids):
        raise ValueError("mode selector mode_token_counts must match uids")
    if mode_prefixes is not None and len(mode_prefixes) != len(uids):
        raise ValueError("mode selector mode_prefixes must match uids")

    groups: dict[str, list[int]] = {}
    for index, uid in enumerate(uids):
        groups.setdefault(str(uid), []).append(index)

    weights = [0.0] * len(uids)
    active_rows = [False] * len(uids)
    fast_means: list[float] = []
    slow_means: list[float] = []
    slow_targets: list[float] = []
    slow_better = 0
    neutral_groups = 0
    complete_prefix_groups = 0
    malformed_rows = 0
    zero_mode_token_rows = 0
    fast_rows = 0
    slow_rows = 0
    group_details: dict[str, dict[str, float | bool]] = {}
    for uid, indices in groups.items():
        if len(indices) != rollout_n:
            raise ValueError(f"mode selector expected {rollout_n} rows for {uid}, got {len(indices)}")
        mode_rewards: dict[str, list[float]] = {"fast": [], "slow": []}
        valid_indices: dict[str, list[int]] = {"fast": [], "slow": []}
        # Normalize each reward dimension inside this exact six-candidate
        # group before comparing FAST with SLOW. This prevents network queue
        # scale, local score, reasoning cost, and format penalties from
        # implicitly changing one another's weight.
        # Mode chooses FAST versus SLOW.  It owns traffic utility and the
        # reasoning cost of that choice, but does not own formatting errors
        # in the downstream reasoning/signal span.
        reward_keys = ("local_score", "network_reward", "reasoning_cost_reward")
        reward_weights = (1.0, 0.5, 0.5)
        has_online_dimensions = any(
            key in reward_infos[index] for index in indices for key in reward_keys
        )
        normalized_utility = {index: 0.0 for index in indices}
        if has_online_dimensions:
            for key, weight in zip(reward_keys, reward_weights, strict=True):
                values = []
                for index in indices:
                    value = float(reward_infos[index].get(key, 0.0))
                    values.append(value if math.isfinite(value) else 0.0)
                mean = sum(values) / len(values)
                variance = sum((value - mean) ** 2 for value in values) / len(values)
                std = math.sqrt(variance)
                if std > 1e-6:
                    for index, value in zip(indices, values, strict=True):
                        normalized_utility[index] += weight * (value - mean) / (std + 1e-6)
        else:
            # Compatibility for the offline utility tests and old archived
            # rollouts. New online rows always carry the dimensions above.
            for index in indices:
                traffic = float(reward_infos[index].get("traffic_reward", 0.0))
                cost = float(reward_infos[index].get("reasoning_penalty", 0.0))
                normalized_utility[index] = traffic - cost
        for index in indices:
            info = reward_infos[index]
            mode = str(info.get("forced_mode") or info.get("generated_mode") or "").lower()
            if mode not in mode_rewards:
                raise ValueError(f"mode selector row for {uid} has invalid mode {mode!r}")
            fast_rows += int(mode == "fast")
            slow_rows += int(mode == "slow")
            malformed_rows += int(not bool(info.get("format_valid", info.get("signal_valid", True))))
            if mode_token_counts is not None:
                zero_mode_token_rows += int(int(mode_token_counts[index]) <= 0)
            if _mode_row_is_valid(info):
                mode_rewards[mode].append(normalized_utility[index])
                valid_indices[mode].append(index)

        has_evidence = bool(mode_rewards["fast"] and mode_rewards["slow"])
        if has_evidence:
            fast_mean = sum(mode_rewards["fast"]) / len(mode_rewards["fast"])
            slow_mean = sum(mode_rewards["slow"]) / len(mode_rewards["slow"])
            scaled_delta = max(-60.0, min(60.0, (slow_mean - fast_mean) / temperature))
            p_slow = 1.0 / (1.0 + math.exp(-scaled_delta))
            p_slow = min(1.0 - min_probability, max(min_probability, p_slow))
            slow_better += int(slow_mean > fast_mean)
        else:
            fast_mean = sum(mode_rewards["fast"]) / len(mode_rewards["fast"]) if mode_rewards["fast"] else 0.0
            slow_mean = sum(mode_rewards["slow"]) / len(mode_rewards["slow"]) if mode_rewards["slow"] else 0.0
            p_slow = 0.5
            neutral_groups += 1
        fast_means.append(fast_mean)
        slow_means.append(slow_mean)
        slow_targets.append(p_slow)

        eligible = indices
        if mode_prefixes is not None:
            prefix_groups: dict[tuple[int, ...], dict[str, list[int]]] = {}
            for mode in ("fast", "slow"):
                for index in valid_indices[mode]:
                    prefix = tuple(mode_prefixes[index])
                    prefix_groups.setdefault(prefix, {"fast": [], "slow": []})[mode].append(index)
            shared = [(prefix, rows) for prefix, rows in prefix_groups.items() if rows["fast"] and rows["slow"]]
            if shared:
                _, rows = max(shared, key=lambda item: len(item[1]["fast"]) + len(item[1]["slow"]))
                eligible = rows["fast"] + rows["slow"]
                complete_prefix_groups += 1
            else:
                eligible = []
        if not has_evidence:
            eligible = []
        token_counts = {
            index: max(1, int(mode_token_counts[index])) if mode_token_counts is not None else 1
            for index in eligible
        }
        total_tokens = sum(token_counts.values())
        fast_tokens = sum(
            token_counts[index] for index in eligible
            if str(reward_infos[index].get("forced_mode") or reward_infos[index].get("generated_mode") or "").lower() == "fast"
        )
        slow_tokens = sum(
            token_counts[index] for index in eligible
            if str(reward_infos[index].get("forced_mode") or reward_infos[index].get("generated_mode") or "").lower() == "slow"
        )
        for index in eligible:
            mode = str(reward_infos[index].get("forced_mode") or reward_infos[index].get("generated_mode") or "").lower()
            if mode == "fast" and fast_tokens:
                weights[index] = (1.0 - p_slow) * total_tokens / fast_tokens
            elif mode == "slow" and slow_tokens:
                weights[index] = p_slow * total_tokens / slow_tokens
            active_rows[index] = True
        group_details[uid] = {
            "fast_utility": float(fast_mean),
            "slow_utility": float(slow_mean),
            "p_target_slow": float(p_slow),
            "valid": bool(has_evidence and eligible),
        }

    group_count = len(groups)
    metrics = {
        "fast_mean_traffic_utility": sum(fast_means) / group_count,
        "slow_mean_traffic_utility": sum(slow_means) / group_count,
        "mode_target_slow_probability": sum(slow_targets) / group_count,
        "slow_better_group_ratio": slow_better / group_count,
        "mode_target_neutral_group_ratio": neutral_groups / group_count,
        "mode_active_row_ratio": sum(active_rows) / len(active_rows) if active_rows else 0.0,
        "mode_group_count": float(group_count),
        "mode_complete_prefix_group_ratio": complete_prefix_groups / group_count,
        "mode_fast_row_count": float(fast_rows),
        "mode_slow_row_count": float(slow_rows),
        "mode_malformed_row_ratio": malformed_rows / len(reward_infos) if reward_infos else 0.0,
        "mode_zero_token_row_ratio": zero_mode_token_rows / len(reward_infos) if reward_infos else 0.0,
    }
    return weights, metrics, active_rows, group_details
