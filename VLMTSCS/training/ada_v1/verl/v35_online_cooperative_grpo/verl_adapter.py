"""Bridge the live cooperative runtime to VERL's batch protocol.

The online JSONL rows are intentionally only bootstrap metadata.  This module
turns those rows into live master streams, expands each city candidate into one
policy-training record per intersection, and optionally materializes plain-text
records as a :class:`verl.protocol.DataProto` when a tokenizer is available.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .online_rollout import CityRolloutResult
from .online_runtime import OnlineBatchResult, OnlineCooperativeRuntime
from .online_samples import OnlineSampleSpec
from .stage2_protocol import build_stage2_prompt
from .stage1_protocol import (
    apply_stage1_perceptions,
    build_stage1_messages,
    extract_coordination_frames,
    parse_perception,
)


def _online_mode_advantages(records: Sequence[Mapping[str, Any]]) -> list[float]:
    """Compute a GDPO advantage for the mode token within each intersection."""
    import os

    groups: dict[str, list[int]] = {}
    for i, row in enumerate(records):
        groups.setdefault(str(row.get("mode_group_id", row.get("group_id", ""))), []).append(i)
    advantages = [0.0] * len(records)
    reward_keys = tuple(
        key.strip()
        for key in os.environ.get(
            "V35_MODE_GDPO_REWARD_KEYS",
            "network_reward,local_score,reasoning_cost_reward,format_penalty",
        ).split(",")
        if key.strip()
    )
    if not reward_keys:
        raise ValueError("V35_MODE_GDPO_REWARD_KEYS must contain at least one reward key")
    for _, indices in groups.items():
        # Keep malformed/invalid responses in the comparison. Their
        # format_penalty is deliberately part of the reward dimensions, so
        # removing them here would make the mode selector blind to failures.
        if len(indices) < 2:
            continue
        for key in reward_keys:
            values = [float(records[i].get(key, 0.0)) for i in indices]
            values = [value if np.isfinite(value) else 0.0 for value in values]
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            std = variance**0.5
            if std <= 1e-6:
                continue
            for i, value in zip(indices, values, strict=True):
                advantages[i] += (value - mean) / (std + 1e-6)
    return advantages


def _tag_mask(response_ids: Sequence[int], tokenizer: Any, tag: str, *, include: bool = True) -> list[int]:
    """Create a response-token mask for a tagged section."""
    def find(values: list[int], needle: list[int], start: int = 0) -> int | None:
        if not needle:
            return start
        for i in range(start, len(values) - len(needle) + 1):
            if values[i : i + len(needle)] == needle:
                return i
        return None
    open_ids = _tokenize(tokenizer, f"<{tag}>")
    close_ids = _tokenize(tokenizer, f"</{tag}>")
    start = find(list(response_ids), open_ids)
    if start is None:
        return [0] * len(response_ids)
    end = find(list(response_ids), close_ids, start + len(open_ids))
    if end is None:
        end = len(response_ids)
    left, right = start + len(open_ids), end
    return [1 if left <= i < right else 0 for i in range(len(response_ids))]


def _disjoint_decision_masks(
    response_ids: Sequence[int], tokenizer: Any, *, format_valid: bool
) -> tuple[list[int], list[int], int]:
    """Return mutually exclusive reasoning/signal masks for one response.

    Malformed responses remain in reward/group statistics through their
    format penalty, but must not turn ambiguous free-form spans into policy
    targets.  Signal takes precedence for any tokenizer-boundary overlap in a
    response that the parser accepted.
    """
    width = len(response_ids)
    if not format_valid:
        return [0] * width, [0] * width, 0
    reasoning = _tag_mask(response_ids, tokenizer, "reasoning")
    signal = _tag_mask(response_ids, tokenizer, "signal")
    overlap = sum(int(bool(a and b)) for a, b in zip(reasoning, signal, strict=True))
    if overlap:
        reasoning = [
            int(bool(reasoning_token and not signal_token))
            for reasoning_token, signal_token in zip(reasoning, signal, strict=True)
        ]
    return reasoning, signal, overlap


def _python_value(value: Any) -> Any:
    """Unwrap numpy scalar/object values without touching nested mappings."""
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return _python_value(value.item())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _batch_rows(batch: Any) -> list[dict[str, Any]]:
    """Read DataProto, a row mapping, or a sequence of row mappings."""
    if hasattr(batch, "non_tensor_batch"):
        values = getattr(batch, "non_tensor_batch") or {}
        length = len(batch)
        rows = []
        for index in range(length):
            rows.append({key: _python_value(value[index]) for key, value in values.items()})
        return rows
    if isinstance(batch, Mapping):
        # A mapping of columns is convenient in unit tests and dataloader
        # adapters.  A single row mapping is also accepted.
        if "extra_info" in batch and isinstance(batch["extra_info"], Mapping):
            return [dict(batch)]
        sequence_values = [value for value in batch.values() if isinstance(value, (list, tuple, np.ndarray))]
        if not sequence_values:
            return [dict(batch)]
        length = len(sequence_values[0])
        if any(len(value) != length for value in sequence_values):
            raise ValueError("column-style online batch has inconsistent column lengths")
        return [{key: _python_value(value[index]) for key, value in batch.items()} for index in range(length)]
    if isinstance(batch, Sequence) and not isinstance(batch, (str, bytes, bytearray)):
        rows = [dict(row) for row in batch]
        return rows
    raise TypeError("online batch must be a DataProto, mapping, or sequence of mappings")


def _metadata(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row.get("extra_info", {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("online row extra_info must be a mapping")
    return value


def _field(row: Mapping[str, Any], metadata: Mapping[str, Any], name: str, *, required: bool = True) -> Any:
    value = row.get(name, metadata.get(name))
    if value is None and required:
        raise ValueError(f"online row is missing required field {name!r}")
    return _python_value(value)


def specs_from_batch(batch: Any, *, config: Any, split: str) -> list[OnlineSampleSpec]:
    """Convert VERL fake rows into validated persistent stream specifications."""
    rows = _batch_rows(batch)
    if not rows:
        raise ValueError("online batch cannot be empty")
    specs: list[OnlineSampleSpec] = []
    seen_streams: set[str] = set()
    known_cities = set(config.cities)
    for row in rows:
        metadata = _metadata(row)
        city = str(_field(row, metadata, "city"))
        if city not in known_cities:
            raise ValueError(f"online row city {city!r} is not registered in config")
        stream_id = str(_field(row, metadata, "stream_id"))
        if stream_id in seen_streams:
            raise ValueError(f"online batch contains duplicate stream_id {stream_id!r}")
        seen_streams.add(stream_id)
        specs.append(
            OnlineSampleSpec(
                data_source=str(_field(row, metadata, "data_source", required=False) or "v35_online_sumo"),
                city=city,
                stream_id=stream_id,
                episode_id=int(_field(row, metadata, "episode_id", required=False) or 0),
                seed=int(_field(row, metadata, "seed")),
                ordinal=int(_field(row, metadata, "ordinal")),
            )
        )
    split_config = getattr(config, split)
    if len(specs) > split_config.batch_size:
        raise ValueError(
            f"online batch has {len(specs)} rows but {split} batch_size is {split_config.batch_size}"
        )
    return specs


def _record_extra_info(record: Mapping[str, Any], spec: OnlineSampleSpec, *, selected: bool) -> dict[str, Any]:
    # Keep the complete online credit assignment payload under one stable key.
    # Trainer workers receive ``extra_fields`` rather than the original row.
    reward_extra_info = {
        key: record.get(key, 0.0)
        for key in (
            "local_score", "network_reward", "reasoning_cost_reward", "format_penalty",
            "signal_valid", "format_valid", "forced_mode", "decision_group_id",
            "mode_group_id", "episode_group_id",
        )
    }
    return {
        "online": True,
        "city": spec.city,
        "stream_id": spec.stream_id,
        "sample_id": spec.sample_id,
        "episode_id": spec.episode_id,
        "seed": spec.seed,
        "ordinal": spec.ordinal,
        "step": int(record["step"]),
        "intersection_id": str(record["intersection_id"]),
        "rollout_id": int(record["rollout_id"]),
        "group_id": str(record["group_id"]),
        "mode_group_id": str(record.get("mode_group_id", record["group_id"])),
        "network_group_id": str(record.get("network_group_id", record.get("episode_group_id", ""))),
        "episode_group_id": str(record.get("episode_group_id", "")),
        "network_uid": str(record.get("episode_group_id", "")),
        "episode_reward": float(record.get("episode_reward", record.get("network_reward", 0.0))),
        "train_cycle": int(record.get("train_cycle", 0)),
        "forced_mode": record.get("forced_mode"),
        "decision_group_id": str(record.get("decision_group_id", record.get("group_id", ""))),
        "reward_extra_info": reward_extra_info,
        "selected": bool(selected),
        "before_queues": dict(record.get("before_queues", {})),
        "after_queues": dict(record.get("after_queues", {})),
        "trajectory_signals": list(record.get("trajectory_signals", ())),
        "local_perception": record.get("local_perception"),
        "cooperative_perception": record.get("cooperative_perception"),
        "perception_audit": dict(record.get("perception_audit", {})),
        "candidate_intersections": list(record.get("candidate_intersections", ())),
    }


def records_for_verl(result: OnlineBatchResult) -> list[dict[str, Any]]:
    """Add VERL keys and selection metadata to flattened GDPO records."""
    output: list[dict[str, Any]] = []
    for item_index, (item, candidates) in enumerate(zip(result.items, result.rollouts)):
        selected_id = result.selected_rollout_ids[item_index] if item_index < len(result.selected_rollout_ids) else None
        spec = item.spec
        for candidate in candidates:
            if not candidate.results:
                raise ValueError("network rollout candidate contains no intersection results")
            for record in _records_for_candidate(candidate):
                # A training batch can contain two streams from the same city
                # at the same SUMO step.  Include the persistent stream/sample
                # identity so each target's six forced-mode candidates remain
                # one independent GDPO group instead of being merged together.
                group_id = (
                    f"{spec.sample_id}:{record['city']}:{record['step']}"
                    f":{record['intersection_id']}"
                )
                selected = selected_id is not None and int(record["rollout_id"]) == int(selected_id)
                enriched = dict(record)
                enriched["group_id"] = group_id
                network_group_id = f"{spec.sample_id}:{record['city']}:{record['step']}"
                enriched["uid"] = group_id
                enriched["network_group_id"] = network_group_id
                enriched["episode_group_id"] = network_group_id
                enriched["network_uid"] = network_group_id
                enriched["mode_group_id"] = f"{network_group_id}:{record['intersection_id']}"
                # Decision credit is local to one intersection, while the
                # network reward remains an independent reward dimension.
                enriched["decision_group_id"] = enriched["mode_group_id"]
                enriched["data_source"] = spec.data_source
                enriched["reward_model"] = {"ground_truth": "online"}
                # VERL's transfer queue materializes row metadata under
                # ``extra_fields``. Keep ``extra_info`` for the dataset
                # contract, but expose the identical payload under the key
                # consumed by trainer_base._compute_advantage().
                # Build the trainer payload from the enriched row.  The
                # original row still has the candidate-local group id and
                # does not contain the persistent sample/network identity;
                # using it here silently turns every decision group into
                # singleton/mixed groups in _compute_advantage().
                enriched["extra_info"] = _record_extra_info(enriched, spec, selected=selected)
                enriched["extra_fields"] = enriched["extra_info"]
                output.append(enriched)
    return output


def validate_verl_records(records: Sequence[Mapping[str, Any]], *, expected_rollouts: int | None = None) -> None:
    """Validate GDPO grouping before records enter the trainer."""
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        uid = str(record.get("uid", record.get("group_id", "")))
        if not uid:
            raise ValueError("online record is missing uid/group_id")
        # ``trainer_base._compute_advantage`` consumes this nested payload
        # after AgentLoop/TransferQueue materialization.  Check it here,
        # before the expensive rollout reaches the optimizer, because a
        # missing or stale value otherwise degrades to zero credit silently.
        extra_fields = record.get("extra_fields")
        if not isinstance(extra_fields, Mapping):
            raise ValueError(f"online record {uid!r} is missing extra_fields")
        reward_extra_info = extra_fields.get("reward_extra_info")
        if not isinstance(reward_extra_info, Mapping):
            raise ValueError(f"online record {uid!r} is missing reward_extra_info")
        for key in ("decision_group_id", "mode_group_id", "episode_group_id"):
            if str(extra_fields.get(key, "")) != str(record.get(key, "")):
                raise ValueError(f"online record {uid!r} has stale extra_fields.{key}")
            if str(reward_extra_info.get(key, "")) != str(record.get(key, "")):
                raise ValueError(f"online record {uid!r} has stale reward_extra_info.{key}")
        for key in ("local_score", "network_reward", "reasoning_cost_reward", "format_penalty"):
            try:
                payload_value = float(reward_extra_info[key])
                record_value = float(record[key])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"online record {uid!r} is missing reward {key!r}") from exc
            if not math.isfinite(payload_value) or payload_value != record_value:
                raise ValueError(
                    f"online record {uid!r} has inconsistent rewards: "
                    f"stale reward_extra_info.{key}"
                )
        groups.setdefault(uid, []).append(record)
    for uid, group in groups.items():
        rollout_ids = [int(item["rollout_id"]) for item in group]
        if len(set(rollout_ids)) != len(rollout_ids):
            raise ValueError(f"GDPO group {uid!r} contains duplicate rollout IDs")
        if expected_rollouts is not None and len(group) != int(expected_rollouts):
            raise ValueError(
                f"GDPO group {uid!r} has {len(group)} candidates; expected {int(expected_rollouts)}"
            )
        steps = {int(item["step"]) for item in group}
        cities = {str(item["city"]) for item in group}
        intersections = {str(item["intersection_id"]) for item in group}
        if len(steps) != 1 or len(cities) != 1 or len(intersections) != 1:
            raise ValueError(f"GDPO group {uid!r} mixes city/step/intersection values")

    network_trajectories: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    network_rollouts: dict[str, set[int]] = {}
    for record in records:
        network_group = str(record.get("network_group_id", ""))
        if not network_group:
            raise ValueError("online record is missing network_group_id")
        rollout_id = int(record["rollout_id"])
        network_trajectories.setdefault((network_group, rollout_id), []).append(record)
        network_rollouts.setdefault(network_group, set()).add(rollout_id)
    for (network_group, rollout_id), trajectory in network_trajectories.items():
        rewards = {
            float(row.get("network_reward", row.get("episode_reward", 0.0)))
            for row in trajectory
        }
        if len(rewards) != 1:
            raise ValueError(
                f"network trajectory {network_group!r}/rollout_{rollout_id} has inconsistent rewards"
            )
        intersections = [str(row["intersection_id"]) for row in trajectory]
        if len(intersections) != len(set(intersections)):
            raise ValueError(
                f"network trajectory {network_group!r}/rollout_{rollout_id} contains duplicate intersections"
            )
    if expected_rollouts is not None:
        for network_group, rollout_ids in network_rollouts.items():
            if len(rollout_ids) != int(expected_rollouts):
                raise ValueError(
                    f"network group {network_group!r} has {len(rollout_ids)} trajectories; "
                    f"expected {int(expected_rollouts)}"
                )


def _records_for_candidate(candidate: CityRolloutResult) -> Iterable[dict[str, Any]]:
    """Flatten one candidate without importing the public helper recursively."""
    group_prefix = f"{candidate.city}:{candidate.step}"
    for row in candidate.results:
        yield {
            "uid": f"{group_prefix}:{row.intersection_id}",
            "group_id": f"{group_prefix}:{row.intersection_id}",
            "city": candidate.city,
            "step": candidate.step,
            "rollout_id": candidate.rollout_id,
            "intersection_id": row.intersection_id,
            "prompt": row.prompt,
            "response": row.response,
            "mode": row.forced_mode,
            "forced_mode": row.forced_mode,
            "parsed_mode": row.parsed.mode,
            "parsed_signal": row.parsed.signal,
            "reasoning": row.parsed.reasoning,
            "signal_valid": bool(row.parsed.signal_valid),
            "format_valid": bool(row.parsed.format_valid),
            "before_queues": dict(candidate.before_queues),
            "after_queues": dict(candidate.after_queues),
            "candidate_intersections": tuple(item.intersection_id for item in candidate.results),
            "trajectory_signals": [
                dict(item) for item in candidate.cycle_results
                if str(item.get("intersection_id")) == row.intersection_id
            ],
            "local_perception": row.local_perception,
            "cooperative_perception": row.cooperative_perception,
            "perception_audit": dict(row.perception_audit),
            **row.reward,
        }


def _tokenize(tokenizer: Any, text: str) -> list[int]:
    try:
        encoded = tokenizer(text, add_special_tokens=False)
    except TypeError:
        encoded = tokenizer(text)
    if isinstance(encoded, Mapping):
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    while encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return [int(token) for token in encoded]


def _trim_generated_response_padding(tensor_batch: Any) -> tuple[int, int, Any]:
    """Remove right-padding introduced by fixed-length online generation.

    vLLM correctly stops a response at EOS, but its generated ``DataProto``
    is padded to ``max_response_length``.  Keeping that trailing padding in
    the actor batch makes every forward/backward pass pay for 4096 response
    positions even when the answer is only a few hundred tokens long.
    """
    import torch

    responses = tensor_batch["responses"]
    response_mask = tensor_batch["response_mask"]
    original_width = int(responses.shape[1])
    response_lengths = response_mask.to(dtype=torch.int64).sum(dim=1)
    trimmed_width = int(response_lengths.max().item())
    if trimmed_width <= 0:
        raise ValueError("generated online responses contain no valid tokens")
    if trimmed_width >= original_width:
        return original_width, original_width, response_lengths

    # The agent loop right-pads every response.  Refuse to crop if that
    # invariant is violated: silently dropping a non-padding token would be
    # worse than retaining a slow batch.
    positions = torch.arange(original_width, device=response_mask.device).unsqueeze(0)
    expected_mask = positions < response_lengths.unsqueeze(1)
    if not torch.equal(response_mask.bool(), expected_mask):
        raise ValueError("generated response_mask is not contiguous right-padding")

    prompt_width = int(tensor_batch["prompts"].shape[1])
    for key in list(tensor_batch.keys()):
        value = tensor_batch[key]
        if not hasattr(value, "ndim") or value.ndim < 2:
            continue
        # Per-response tensors: responses, masks, log-probs, etc.
        if int(value.shape[1]) == original_width:
            tensor_batch[key] = value[:, :trimmed_width]
        # Full sequence tensors retain the complete prompt and only lose the
        # unused response suffix.
        elif int(value.shape[1]) == prompt_width + original_width:
            tensor_batch[key] = value[:, : prompt_width + trimmed_width]
    return original_width, trimmed_width, response_lengths


def records_to_data_proto(
    records: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    generated: Any = None,
    jobs: Sequence[Mapping[str, Any]] | None = None,
) -> Any:
    """Materialize online records as a VERL ``DataProto``.

    The online path passes the DataProto emitted by AgentLoopManager.  Its
    target-major order differs from SUMO's rollout-major reward records, so
    rows are joined by ``(uid, intersection_id, rollout_id)`` before indexing
    the already-padded tensors.  The tokenizer path remains for unit tests and
    external callers only.
    """
    if not records:
        raise ValueError("cannot materialize an empty online record list")
    try:
        import torch
        from tensordict import TensorDict
        from verl.protocol import DataProto
    except ImportError as exc:  # pragma: no cover - deployment dependency
        raise ImportError("torch, tensordict, and VERL are required for DataProto materialization") from exc

    if generated is not None:
        if jobs is None:
            raise ValueError("jobs are required when reusing generated tensors")
        if len(generated) != len(jobs):
            raise ValueError(
                f"generated batch has {len(generated)} rows but jobs has {len(jobs)} rows"
            )

        def row_key(value: Mapping[str, Any]) -> tuple[str, str, int]:
            try:
                return (
                    str(value["uid"]),
                    str(value["intersection_id"]),
                    int(value["rollout_id"]),
                )
            except KeyError as exc:
                raise ValueError(f"online row is missing {exc.args[0]!r}") from exc

        generated_indices: dict[tuple[str, str, int], int] = {}
        for index, job in enumerate(jobs):
            key = row_key(job)
            if key in generated_indices:
                raise ValueError(f"duplicate online generation key: {key}")
            generated_indices[key] = index
        order: list[int] = []
        for record in records:
            key = row_key(record)
            if key not in generated_indices:
                raise ValueError(f"online reward row has no generated response: {key}")
            order.append(generated_indices[key])

        source_batch = generated.batch
        required = ("prompts", "responses", "attention_mask", "response_mask")
        if any(key not in source_batch for key in required):
            raise ValueError("generated DataProto is missing prompt/response alignment tensors")
        index_tensors: dict[str, Any] = {}
        tensor_order_cache: dict[Any, Any] = {}
        for key in source_batch.keys():
            if key == "rm_scores":
                continue
            value = source_batch[key]
            if not hasattr(value, "shape") or value.shape[0] != len(jobs):
                raise ValueError(f"generated tensor {key!r} has an invalid batch dimension")
            device = getattr(value, "device", None)
            if device not in tensor_order_cache:
                tensor_order_cache[device] = torch.tensor(order, dtype=torch.long, device=device)
            index_tensors[key] = value.index_select(0, tensor_order_cache[device])
        tensor_batch = TensorDict(index_tensors, batch_size=len(records))
        original_response_width, trimmed_response_width, generated_response_lengths = (
            _trim_generated_response_padding(tensor_batch)
        )
        padded_prompts = tensor_batch["prompts"]
        padded_responses = tensor_batch["responses"]
        attention_mask = tensor_batch["attention_mask"]
        response_mask = tensor_batch["response_mask"]
        prompt_width = padded_prompts.shape[1]
        prompt_lengths = attention_mask[:, :prompt_width].sum(dim=1).to(dtype=torch.int64)
        response_lengths = response_mask.to(dtype=torch.int64).sum(dim=1)
        meta_info = dict(getattr(generated, "meta_info", {}) or {})
        import logging
        logging.getLogger(__name__).info(
            "[ONLINE_RESPONSE_PADDING_TRIM] rows=%d generated_width=%d retained_width=%d "
            "valid_tokens_min=%d valid_tokens_mean=%.1f valid_tokens_max=%d",
            len(records), original_response_width, trimmed_response_width,
            int(generated_response_lengths.min().item()),
            float(generated_response_lengths.to(dtype=torch.float32).mean().item()),
            int(generated_response_lengths.max().item()),
        )
    else:
        prompt_ids = [_tokenize(tokenizer, str(record["prompt"])) for record in records]
        response_ids = [_tokenize(tokenizer, str(record["response"])) for record in records]
        if any(not ids for ids in prompt_ids):
            raise ValueError("tokenizer produced an empty prompt")
        if any(not ids for ids in response_ids):
            raise ValueError("tokenizer produced an empty response")
        pad_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
        pad_right = lambda values: torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(value, dtype=torch.int64) for value in values], batch_first=True, padding_value=pad_id
        )
        padded_responses = pad_right(response_ids)
        prompt_width = max(len(value) for value in prompt_ids)
        padded_prompts = torch.full((len(prompt_ids), prompt_width), pad_id, dtype=torch.int64)
        for index, value in enumerate(prompt_ids):
            padded_prompts[index, prompt_width - len(value) :] = torch.tensor(value, dtype=torch.int64)
        input_ids = torch.cat([padded_prompts, padded_responses], dim=1)
        attention_mask = torch.zeros_like(input_ids, dtype=torch.int64)
        position_ids = torch.zeros_like(input_ids, dtype=torch.int64)
        response_mask = torch.zeros((len(records), padded_responses.shape[1]), dtype=torch.int64)
        for index, (prompt, response) in enumerate(zip(prompt_ids, response_ids)):
            prompt_start = prompt_width - len(prompt)
            attention_mask[index, prompt_start:prompt_width] = 1
            attention_mask[index, prompt_width : prompt_width + len(response)] = 1
            position_ids[index, prompt_start:prompt_width] = torch.arange(len(prompt), dtype=torch.int64)
            position_ids[index, prompt_width : prompt_width + len(response)] = torch.arange(
                len(prompt), len(prompt) + len(response), dtype=torch.int64
            )
            response_mask[index, : len(response)] = 1
        tensor_batch = TensorDict(
            {
                "prompts": padded_prompts,
                "responses": padded_responses,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "response_mask": response_mask,
            },
            batch_size=len(records),
        )
        prompt_lengths = torch.tensor([len(value) for value in prompt_ids], dtype=torch.int64)
        response_lengths = response_mask.sum(dim=1).to(dtype=torch.int64)
        meta_info = {}
    # The generic VERL reward path expects one scalar outcome in ``rm_scores``.
    # The shared GRPO baseline uses only the network-level reward. Local credit
    # and format penalties are added hierarchically by the trainer.
    # response token exactly as the native reward manager does.
    rm_scores = torch.zeros_like(response_mask, dtype=torch.float32)
    positions = torch.arange(response_mask.shape[1], device=response_mask.device).expand_as(response_mask)
    last_indices = torch.where(response_mask > 0, positions, torch.full_like(positions, -1)).max(dim=1).values
    if torch.any(last_indices < 0):
        raise ValueError("online records must contain a non-empty response")
    rm_scores[torch.arange(len(records), device=response_mask.device), last_indices] = torch.tensor(
        [float(record.get("network_reward", record.get("episode_reward", 0.0))) for record in records],
        dtype=torch.float32,
        device=response_mask.device,
    )
    tensor_batch["rm_scores"] = rm_scores
    # Segmented online objective: mode selector is trained only on the mode
    # span, while the network GRPO objective sees the decision suffix.  Rows
    # from t1-t3 are never materialized as records, so they are naturally
    # excluded here.
    mode_weights = _online_mode_advantages(records)
    mode_masks, reasoning_masks, signal_masks, grpo_masks = [], [], [], []
    mode_token_indices: list[int] = []
    mode_token_values: list[int] = []
    mode_labels: list[float] = []
    raw_overlap_tokens = 0
    rows_with_overlap = 0
    reasoning_signal_overlap_tokens = 0
    reasoning_signal_overlap_rows = 0
    for row_index in range(len(records)):
        valid_len = int(response_mask[row_index].sum().item())
        ids = padded_responses[row_index, :valid_len].detach().cpu().tolist()
        mode_mask = _tag_mask(ids, tokenizer, "mode")
        mode_masks.append(mode_mask)
        mode_positions = [index for index, value in enumerate(mode_mask) if value]
        # A true FAST/SLOW classifier requires exactly one token for the mode
        # value.  Multi-token mode strings remain on the legacy token-CE
        # fallback instead of silently treating only part of the label as a
        # binary decision.
        if len(mode_positions) == 1:
            mode_token_indices.append(int(mode_positions[0]))
            mode_token_values.append(int(ids[mode_positions[0]]))
        else:
            mode_token_indices.append(-1)
            mode_token_values.append(-1)
        mode_name = str(
            records[row_index].get("forced_mode")
            or records[row_index].get("generated_mode")
            or ""
        ).lower()
        mode_labels.append(1.0 if mode_name == "slow" else 0.0)
        reasoning_mask, signal_mask, decision_overlap = _disjoint_decision_masks(
            ids,
            tokenizer,
            format_valid=bool(records[row_index].get("format_valid", False)),
        )
        reasoning_signal_overlap_tokens += decision_overlap
        reasoning_signal_overlap_rows += int(decision_overlap > 0)
        reasoning_masks.append(reasoning_mask)
        signal_masks.append(signal_mask)
        tagged_decision = [int(bool(a or b)) for a, b in zip(reasoning_mask, signal_mask, strict=True)]
        # A malformed response has no legal decision span.  Never turn its
        # entire free-form response into a PPO target: that was the direct
        # source of the 4096-token collapse after a formatting failure.
        raw_decision = tagged_decision
        overlap = sum(int(bool(a and b)) for a, b in zip(mode_mask, raw_decision, strict=True))
        raw_overlap_tokens += overlap
        rows_with_overlap += int(overlap > 0)
        # Explicit subtraction makes masks disjoint despite tokenizer tag boundaries.
        grpo_masks.append([
            int(bool(decision and not mode))
            for decision, mode in zip(raw_decision, mode_mask, strict=True)
        ])
    def pad_rows(rows: list[list[int]]) -> Any:
        result = torch.zeros((len(rows), padded_responses.shape[1]), dtype=torch.float32, device=response_mask.device)
        for i, values in enumerate(rows):
            if values:
                result[i, : len(values)] = torch.tensor(values, dtype=torch.float32, device=result.device)
        return result
    tensor_batch["mode_aux_mask"] = pad_rows(mode_masks)
    tensor_batch["reasoning_aux_mask"] = pad_rows(reasoning_masks) * response_mask
    tensor_batch["signal_aux_mask"] = pad_rows(signal_masks) * response_mask
    tensor_batch["grpo_loss_mask"] = pad_rows(grpo_masks) * response_mask
    tensor_batch["kl_loss_mask"] = tensor_batch["grpo_loss_mask"]
    tensor_batch["mode_aux_weight"] = pad_rows([[w] * len(m) for w, m in zip(mode_weights, mode_masks)])
    # Derive the two candidate token IDs from the actual generated mode spans.
    # This avoids tokenizer-version differences around leading whitespace.
    # If either class is multi-token/absent, binary mode loss is disabled for
    # those rows and the loss function uses its compatibility fallback.
    observed_fast = {
        value for value, record in zip(mode_token_values, records, strict=True)
        if value >= 0 and str(record.get("forced_mode") or record.get("generated_mode") or "").lower() == "fast"
    }
    observed_slow = {
        value for value, record in zip(mode_token_values, records, strict=True)
        if value >= 0 and str(record.get("forced_mode") or record.get("generated_mode") or "").lower() == "slow"
    }
    fast_token_id = next(iter(observed_fast)) if len(observed_fast) == 1 else -1
    slow_token_id = next(iter(observed_slow)) if len(observed_slow) == 1 else -1
    binary_valid = [
        index >= 0 and fast_token_id >= 0 and slow_token_id >= 0
        and fast_token_id != slow_token_id
        and value in (fast_token_id, slow_token_id)
        for index, value in zip(mode_token_indices, mode_token_values, strict=True)
    ]
    binary_masks = []
    for index, valid in zip(mode_token_indices, binary_valid, strict=True):
        row = [0] * padded_responses.shape[1]
        if valid:
            row[index] = 1
        binary_masks.append(row)
    tensor_batch["mode_binary_mask"] = pad_rows(binary_masks)
    tensor_batch["mode_token_index"] = torch.tensor(mode_token_indices, dtype=torch.int64)
    tensor_batch["mode_fast_token_id"] = torch.full(
        (len(records),), int(fast_token_id), dtype=torch.int64
    )
    tensor_batch["mode_slow_token_id"] = torch.full(
        (len(records),), int(slow_token_id), dtype=torch.int64
    )
    tensor_batch["mode_binary_target"] = torch.tensor(mode_labels, dtype=torch.float32)
    tensor_batch["mode_binary_valid"] = torch.tensor(binary_valid, dtype=torch.bool)
    tensor_batch["mode_binary_weight"] = torch.tensor(
        [float(weight) if valid else 0.0 for weight, valid in zip(mode_weights, binary_valid, strict=True)],
        dtype=torch.float32,
    )
    tensor_batch["grpo_advantage_scale"] = torch.ones_like(response_mask, dtype=torch.float32)
    # Do not gate mode tokens on signal validity.  Malformed signal rows stay
    # in the mode group and receive their negative format reward through the
    # GDPO weight; dropping their mask would make that feedback ineffective.
    overlap = int((tensor_batch["mode_aux_mask"].bool() & tensor_batch["grpo_loss_mask"].bool()).sum().item())
    if overlap:
        raise ValueError(f"mode/decision loss masks overlap on {overlap} tokens")
    import logging
    logging.getLogger(__name__).info(
        "[ONLINE_MASK_DIAG] rows=%d mode_tokens=%d decision_tokens=%d "
        "raw_overlap_tokens=%d rows_with_overlap=%d "
        "reasoning_signal_overlap_tokens=%d reasoning_signal_overlap_rows=%d "
        "decision_groups=%d network_groups=%d valid_rows=%d malformed_rows=%d",
        len(records), int(tensor_batch["mode_aux_mask"].sum().item()),
        int(tensor_batch["grpo_loss_mask"].sum().item()), raw_overlap_tokens,
        rows_with_overlap,
        reasoning_signal_overlap_tokens, reasoning_signal_overlap_rows,
        len({str(r.get('decision_group_id', '')) for r in records}),
        len({str(r.get('episode_group_id', '')) for r in records}),
        sum(bool(r.get("signal_valid", False)) for r in records),
        sum(not bool(r.get("format_valid", False)) for r in records),
    )
    object_keys = {
        "uid",
        "group_id",
        "data_source",
        "reward_model",
        "extra_info",
        "extra_fields",
        "prompt",
        "response",
        "reasoning",
        "multi_modal_inputs",
        "perception_sft_multi_modal_inputs",
        "stage1_prompt",
        "stage1_response",
    }
    non_tensor: dict[str, np.ndarray] = {}
    keys = set().union(*(record.keys() for record in records))
    for key in sorted(keys):
        values = [record.get(key) for record in records]
        if key in object_keys or any(isinstance(value, (Mapping, list, tuple, str, type(None))) for value in values):
            non_tensor[key] = np.asarray(values, dtype=object)
        else:
            non_tensor[key] = np.asarray(values)
    # These fields are emitted by AgentLoopManager._postprocess and are useful
    # to downstream trainer/logging code even though this Stage 2 path is text-only.
    non_tensor.setdefault("prompt_len", prompt_lengths.cpu().numpy().astype(np.int32))
    non_tensor.setdefault("response_len", response_lengths.cpu().numpy().astype(np.int32))
    # V1 metrics fetches this exact TransferQueue field.  Online SUMO
    # rollouts are single-turn, but the former ``__num_turns__`` spelling was
    # never consumed by the trainer and caused a post-update KeyError.
    non_tensor["num_turns"] = np.ones(len(records), dtype=np.int32)
    for mask_name in (
        "grpo_loss_mask", "kl_loss_mask", "mode_aux_mask", "reasoning_aux_mask",
        "signal_aux_mask", "mode_binary_mask",
    ):
        count = int(tensor_batch[mask_name].sum().item())
        # DataProto requires every non-tensor field to have a leading batch
        # dimension.  A scalar ndarray makes check_consistency() fail at
        # ``val.shape[0]`` for online batches.
        non_tensor[f"{mask_name}_num_tokens"] = np.full(
            len(records), count, dtype=np.int64
        )
        sequence_count = int((tensor_batch[mask_name].sum(dim=1) > 0).sum().item())
        non_tensor[f"{mask_name}_num_sequences"] = np.full(
            len(records), sequence_count, dtype=np.int64
        )
    # Binary mode loss uses the active binary mask after GDPO filtering.  Its
    # own normalizer must travel with the batch; reusing mode_aux_mask's count
    # would over-normalize when malformed/multi-token rows are disabled.
    binary_count = int(tensor_batch["mode_binary_mask"].sum().item())
    binary_sequences = int((tensor_batch["mode_binary_mask"].sum(dim=1) > 0).sum().item())
    non_tensor["mode_binary_mask_num_tokens"] = np.full(len(records), binary_count, dtype=np.int64)
    non_tensor["mode_binary_mask_num_sequences"] = np.full(len(records), binary_sequences, dtype=np.int64)
    non_tensor.setdefault("multi_modal_inputs", np.asarray([{} for _ in records], dtype=object))
    # The generic reward extractor indexes every key listed in meta_info's
    # reward_extra_keys. Online records may not carry legacy keys such as
    # ``acc``; provide a batch-aligned fallback so extraction remains safe.
    for reward_key in meta_info.get("reward_extra_keys", ()):
        if reward_key not in non_tensor:
            non_tensor[reward_key] = np.asarray(
                [record.get(reward_key, record.get("score", 0.0)) for record in records],
                dtype=np.float32,
            )
    # FSDP log-prob recomputation reads temperature from DataProto.meta_info.
    # The in-process online generator does not always propagate it, so use
    # the neutral value for teacher-forced likelihood evaluation.
    meta_info.setdefault("temperature", 1.0)
    meta_info.setdefault("timing", {})
    return DataProto(batch=tensor_batch, non_tensor_batch=non_tensor, meta_info=meta_info)


def build_generation_jobs(items: Sequence[Any], prepared: Sequence[Any]) -> list[dict[str, Any]]:
    """Build one text-only VERL request for every target/mode candidate."""
    if len(items) != len(prepared):
        raise ValueError("items and prepared snapshots must have equal length")
    jobs: list[dict[str, Any]] = []
    # Keep all six candidates for a target adjacent.  This preserves VERL's
    # rollout_n/group ordering and makes per-target diagnostics straightforward.
    for sample_index, (item, candidate) in enumerate(zip(items, prepared)):
        observations = {row.intersection_id: row for row in item.snapshot.observations}
        for intersection_id in observations:
            row = observations[intersection_id]
            group_id = f"{item.spec.sample_id}:{item.snapshot.city}:{item.snapshot.step}:{intersection_id}"
            for rollout_id, modes in enumerate(candidate.assignments):
                mode = str(modes[intersection_id])
                prompt = build_stage2_prompt(
                    row.local_perception,
                    row.cooperative_perception,
                    template=None,
                    forced_mode=mode,
                )
                jobs.append({
                    "prompt": prompt,
                    "raw_prompt": [{"role": "user", "content": prompt}],
                    "uid": group_id,
                    "group_id": group_id,
                    "sample_index": sample_index,
                    "sample_id": item.spec.sample_id,
                    "data_source": item.spec.data_source,
                    "city": item.snapshot.city,
                    "step": int(item.snapshot.step),
                    "intersection_id": str(intersection_id),
                    "rollout_id": int(rollout_id),
                    "forced_mode": mode,
                    "job_kind": "stage2_t0",
                    "deployment": False,
                    # SUMO/router is the online gold source for Stage 1.  The
                    # perception SFT helper uses the canonical v35 field
                    # names, so adapt the compact runtime phase table here.
                    "reward_model": {
                        "ground_truth": {
                            "perception_target": _perception_target_from_local(
                                row.local_perception
                            ),
                            "stage1_target": _stage1_target(row),
                        }
                    },
                })
    return jobs


def build_stage1_generation_jobs(items: Sequence[Any]) -> list[dict[str, Any]]:
    """Build one multimodal perception request per intersection."""
    started = time.perf_counter()
    jobs: list[dict[str, Any]] = []
    for sample_index, item in enumerate(items):
        paths = item.video_details.get("paths") or {}
        for row in item.snapshot.observations:
            media = paths.get(row.intersection_id) or {}
            videos = [media.get(direction) for direction in ("E", "W", "N", "S")]
            if any(not value for value in videos):
                raise RuntimeError(f"missing Stage 1 direction video for {row.intersection_id}")
            # Keep extracted coordination frames beside the source videos so
            # temporal/direction audits can inspect one intersection folder.
            visual_dir = Path(str(videos[0])).parent
            frames = extract_coordination_frames(
                city=item.snapshot.city,
                target_id=row.intersection_id,
                video_paths=paths,
                output_dir=visual_dir,
            )
            phases = row.local_perception.get("phases") or {}
            messages = build_stage1_messages(
                intersection_id=row.intersection_id,
                current_phase=row.current_phase,
                ages={name: (phases.get(name) or {}).get("age", 0) for name in phases},
                videos=[str(value) for value in videos],
                coordination_frames=frames,
            )
            uid = f"{item.spec.sample_id}:{item.snapshot.city}:{item.snapshot.step}:{row.intersection_id}:stage1"
            jobs.append({
                "prompt": "", "raw_prompt": messages, "uid": uid, "group_id": uid,
                "sample_index": sample_index, "sample_id": item.spec.sample_id,
                "data_source": item.spec.data_source, "city": item.snapshot.city,
                "step": int(item.snapshot.step), "intersection_id": row.intersection_id,
                "rollout_id": 0, "forced_mode": None, "job_kind": "stage1",
                # Stage 1 is a structured perception pass, not an exploratory
                # policy rollout. Sampling here causes avoidable schema drift.
                "sampling_temperature": 0.0,
                "deployment": False,
                "reward_model": {"ground_truth": {
                    "perception_target": _perception_target_from_local(row.local_perception),
                    "stage1_target": _stage1_target(row),
                }},
            })
    print(
        f"[STAGE1_TIMING] build_jobs samples={len(items)} jobs={len(jobs)} "
        f"elapsed_s={time.perf_counter() - started:.3f}",
        flush=True,
    )
    return jobs


def stage1_snapshots_from_generation(
    items: Sequence[Any], generated: Any, jobs: Sequence[Mapping[str, Any]], tokenizer: Any
) -> list[Any]:
    if len(generated) != len(jobs):
        raise ValueError("Stage 1 generated rows do not align with jobs")
    by_sample: list[dict[str, Any]] = [dict() for _ in items]
    fallback_count = 0
    for index, job in enumerate(jobs):
        ids = generated.batch["responses"][index]
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        text = tokenizer.decode(ids, skip_special_tokens=True).strip()
        sample_index = int(job["sample_index"])
        intersection_id = str(job["intersection_id"])
        try:
            perception = parse_perception(text)
        except (ValueError, json.JSONDecodeError) as exc:
            # Stage 1's generated text remains attached as the SFT response
            # below.  It must not make a whole live batch unusable: the
            # snapshot already contains same-timestep SUMO/router perception
            # gold, which is the canonical Stage 1 target and is safe input
            # for Stage 2.  ``apply_stage1_perceptions`` then rebuilds all
            # neighbor fields from this one consistent city snapshot.
            source_row = next(
                (
                    row
                    for row in items[sample_index].snapshot.observations
                    if row.intersection_id == intersection_id
                ),
                None,
            )
            if source_row is None:
                raise ValueError(
                    f"Stage 1 job references unknown intersection {intersection_id!r}"
                ) from exc
            diagnostic_dir = Path(items[sample_index].snapshot_dir) / "stage1" / intersection_id
            diagnostic_dir.mkdir(parents=True, exist_ok=True)
            response_path = diagnostic_dir / "invalid_generation.txt"
            response_path.write_text(text, encoding="utf-8")
            metadata = {
                "uid": str(job.get("uid", "")),
                "sample_index": sample_index,
                "sample_id": str(job.get("sample_id", "")),
                "city": str(job.get("city", "")),
                "step": int(job.get("step", -1)),
                "intersection_id": intersection_id,
                "response_token_slots": len(ids),
                "perception_open_tags": text.count("<perception>"),
                "perception_close_tags": text.count("</perception>"),
                "parser_error": f"{type(exc).__name__}: {exc}",
                "stage2_perception_source": "sumo_gold_fallback",
            }
            (diagnostic_dir / "invalid_generation.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            perception = dict(source_row.local_perception)
            fallback_count += 1
            print(
                "[STAGE1_PARSE_FALLBACK] "
                f"sample={sample_index} intersection={intersection_id} "
                f"source=sumo_gold error={type(exc).__name__}",
                flush=True,
            )
        by_sample[sample_index][intersection_id] = perception
    output = []
    for item, perceptions in zip(items, by_sample):
        snapshot = apply_stage1_perceptions(item.snapshot, perceptions)
        output.append(replace(item, snapshot=snapshot))
    print(
        f"[STAGE1_PARSE_DIAG] jobs={len(jobs)} parsed={len(jobs) - fallback_count} "
        f"sumo_gold_fallbacks={fallback_count}",
        flush=True,
    )
    return output


def attach_stage1_supervision(
    generated: Any, jobs: Sequence[Mapping[str, Any]], stage1_generated: Any,
    stage1_jobs: Sequence[Mapping[str, Any]],
) -> None:
    """Use the real multimodal Stage 1 prompt as each group's SFT source."""
    keys = (
        "perception_sft_responses", "perception_sft_input_ids",
        "perception_sft_attention_mask", "perception_sft_position_ids",
        "perception_sft_mask", "perception_sft_token_weight",
    )
    available = [key for key in keys if key in stage1_generated.batch]
    if not available:
        return
    missing = set(keys) - set(available)
    if missing:
        raise ValueError(f"Stage 1 generation has incomplete SFT tensors: {sorted(missing)}")
    lookup = {
        (int(job["sample_index"]), str(job["intersection_id"])): index
        for index, job in enumerate(stage1_jobs)
    }
    import torch
    order = torch.tensor([
        lookup[(int(job["sample_index"]), str(job["intersection_id"]))]
        for job in jobs
    ], dtype=torch.long, device=stage1_generated.batch[available[0]].device)
    for key in keys:
        source = stage1_generated.batch[key]
        generated.batch[key] = source.index_select(0, order.to(source.device))


def build_deployment_generation_jobs(items: Sequence[Any]) -> list[dict[str, Any]]:
    """Build one autonomous validation request per intersection."""
    jobs: list[dict[str, Any]] = []
    for sample_index, item in enumerate(items):
        for row in item.snapshot.observations:
            group_id = f"{item.spec.sample_id}:{item.snapshot.city}:{item.snapshot.step}:{row.intersection_id}"
            prompt = build_stage2_prompt(
                row.local_perception, row.cooperative_perception, template=None, forced_mode=None
            )
            jobs.append({
                "prompt": prompt,
                "raw_prompt": [{"role": "user", "content": prompt}],
                "uid": group_id,
                "group_id": group_id,
                "sample_index": sample_index,
                "sample_id": item.spec.sample_id,
                "data_source": item.spec.data_source,
                "city": item.snapshot.city,
                "step": int(item.snapshot.step),
                "intersection_id": str(row.intersection_id),
                "rollout_id": 0,
                "forced_mode": None,
                "job_kind": "stage2_t0",
                "deployment": True,
                "sampling_temperature": 1.0,
                "reward_model": {"ground_truth": {
                    "perception_target": _perception_target_from_local(row.local_perception),
                    "stage1_target": _stage1_target(row),
                }},
            })
    return jobs


def build_temporal_generation_jobs(
    items: Sequence[Any], prepared: Sequence[Any], snapshots: Sequence[Sequence[Any]], *, cycle: int
) -> list[dict[str, Any]]:
    """Build deterministic Stage 2 jobs for every live candidate at t1..t3."""
    jobs: list[dict[str, Any]] = []
    for sample_index, (item, candidate, sample_snapshots) in enumerate(zip(items, prepared, snapshots)):
        if len(sample_snapshots) != len(candidate.assignments):
            raise ValueError("temporal snapshots and assignments are not aligned")
        for rollout_id, snapshot in enumerate(sample_snapshots):
            for row in snapshot.observations:
                uid = (
                    f"{item.spec.sample_id}:{snapshot.city}:{item.snapshot.step}:"
                    f"{row.intersection_id}:t{cycle}:r{rollout_id}"
                )
                prompt = build_stage2_prompt(
                    row.local_perception, row.cooperative_perception, forced_mode=None
                )
                jobs.append({
                    "prompt": prompt, "raw_prompt": [{"role": "user", "content": prompt}],
                    "uid": uid, "group_id": uid, "sample_index": sample_index,
                    "sample_id": item.spec.sample_id, "data_source": item.spec.data_source,
                    "city": snapshot.city, "step": int(snapshot.step),
                    "intersection_id": row.intersection_id, "rollout_id": rollout_id,
                    "forced_mode": None, "job_kind": f"stage2_t{cycle}",
                    "deployment": True,
                    "reward_model": {"ground_truth": {
                        "perception_target": _perception_target_from_local(row.local_perception),
                        "stage1_target": _stage1_target(row),
                    }},
                })
    return jobs


def _perception_target_from_local(local: Any) -> dict[str, Any]:
    """Convert one SUMO local observation into the online Stage-1 target."""
    phases = local.get("phases", {}) if isinstance(local, Mapping) else {}
    target: dict[str, Any] = {}
    for phase in ("ETWT", "NTST", "ELWL", "NLSL"):
        row = phases.get(phase, {}) if isinstance(phases, Mapping) else {}
        target[phase] = {
            "current_v": list(row.get("v", [0, 0])),
            "current_q": list(row.get("q", [0, 0])),
            "demand_trend_v30_minus_v5": row.get("dv", 0),
            "queue_trend_q30_minus_q5": row.get("dq", 0),
            "coordinated_arrivals": {},
            "nonzero_v_history_length_since_last_service": row.get("age", 0),
        }
    return target


def _stage1_target(observation: Any) -> dict[str, Any]:
    """Build the exact multimodal Stage-1 schema from one SUMO observation."""
    local = observation.local_perception
    cooperative = observation.cooperative_perception
    phases = local.get("phases", {}) if isinstance(local, Mapping) else {}
    coordination = (
        cooperative.get("local_coordination", {})
        if isinstance(cooperative, Mapping)
        else {}
    )
    return {
        "current_phase": local.get("current_phase"),
        "phases": {
            phase: {
                **dict(phases.get(phase, {}) or {}),
                "coord": dict(coordination.get(phase, {}) or {}),
            }
            for phase in ("ETWT", "NTST", "ELWL", "NLSL")
        },
    }


def generation_jobs_to_data_proto(
    jobs: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    global_steps: int = -1,
    validate: bool = False,
) -> Any:
    """Create the minimal text-only DataProto accepted by AgentLoopManager."""
    if not jobs:
        raise ValueError("cannot generate from an empty job list")
    try:
        import torch
        from tensordict import TensorDict
        from verl.protocol import DataProto
    except ImportError as exc:  # pragma: no cover
        raise ImportError("torch, tensordict, and VERL are required") from exc

    # RLHFDataset uses a dummy tensor because AgentLoopManager only needs the
    # batch dimension and the raw chat messages before it tokenizes them.
    tensor_batch = TensorDict(
        {"dummy_tensor": torch.zeros((len(jobs), 1), dtype=torch.uint8)},
        batch_size=len(jobs),
    )
    n = len(jobs)
    def object_array(values: Iterable[Any]) -> np.ndarray:
        values = list(values)
        result = np.empty(n, dtype=object)
        result[:] = values
        return result
    non_tensor = {
        "raw_prompt": object_array(job["raw_prompt"] for job in jobs),
        "uid": object_array(job["uid"] for job in jobs),
        "group_id": object_array(job["group_id"] for job in jobs),
        "index": np.asarray([job["uid"] for job in jobs], dtype=object),
        "sample_index": np.asarray([int(job["sample_index"]) for job in jobs], dtype=np.int64),
        "sample_id": object_array(job["sample_id"] for job in jobs),
        "data_source": object_array(job["data_source"] for job in jobs),
        "city": object_array(job["city"] for job in jobs),
        "step": np.asarray([int(job["step"]) for job in jobs], dtype=np.int64),
        "intersection_id": object_array(job["intersection_id"] for job in jobs),
        "rollout_id": np.asarray([int(job["rollout_id"]) for job in jobs], dtype=np.int64),
        "forced_mode": object_array(job["forced_mode"] for job in jobs),
        "reward_model": object_array(job["reward_model"] for job in jobs),
        "job_kind": object_array(job.get("job_kind", "stage2_t0") for job in jobs),
        "deployment": np.asarray([bool(job.get("deployment", False)) for job in jobs], dtype=np.bool_),
        "sampling_temperature": np.asarray(
            [job.get("sampling_temperature") for job in jobs], dtype=object
        ),
    }
    return DataProto(
        batch=tensor_batch,
        non_tensor_batch=non_tensor,
        meta_info={"global_steps": int(global_steps), "validate": bool(validate)},
    )


def response_groups_from_generation(
    generated: Any,
    jobs: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    num_samples: int,
    num_rollouts: int,
) -> list[list[dict[str, str]]]:
    """Map AgentLoopManager responses back to sample/rollout/intersection."""
    if len(generated) != len(jobs):
        raise ValueError(f"policy generated {len(generated)} rows for {len(jobs)} jobs")
    responses = generated.batch["responses"]
    groups: list[list[dict[str, str]]] = [
        [dict() for _ in range(num_rollouts)] for _ in range(num_samples)
    ]
    for index, job in enumerate(jobs):
        token_ids = responses[index]
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        text = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
        sample_index = int(job["sample_index"])
        rollout_id = int(job["rollout_id"])
        intersection_id = str(job["intersection_id"])
        groups[sample_index][rollout_id][intersection_id] = text
    for sample in groups:
        for rollout in sample:
            if not rollout:
                raise ValueError("VERL generated an empty candidate response")
    return groups


@dataclass
class OnlineVERLBatch:
    result: OnlineBatchResult
    records: list[dict[str, Any]]
    data_proto: Any = None


class OnlineVERLCollector:
    """Trainer-facing collector for live cooperative SUMO samples."""

    def __init__(self, runtime: OnlineCooperativeRuntime, *, config: Any, split: str = "train") -> None:
        if split not in {"train", "val"}:
            raise ValueError("split must be train or val")
        self.runtime = runtime
        self.config = config
        self.split = split

    def _run_temporal(
        self, items: Sequence[Any], prepared: Sequence[Any], t0_groups: Any, *,
        sequence_generator: Any, tokenizer: Any, global_steps: int,
    ) -> list[list[CityRolloutResult]]:
        coordinator = self.runtime.coordinator
        temporal_started = time.perf_counter()
        phase_started = time.perf_counter()
        session = coordinator.start_temporal_batch(prepared, tokenizer=tokenizer)
        print(
            f"[STAGE2_TIMING] split={self.split} phase=start_actors "
            f"samples={len(items)} rollouts={sum(len(group) for group in session.actors)} "
            f"elapsed_s={time.perf_counter() - phase_started:.3f}",
            flush=True,
        )
        try:
            phase_started = time.perf_counter()
            coordinator.apply_temporal_cycle(session, t0_groups, cycle=0, advance=True)
            print(
                f"[STAGE2_TIMING] split={self.split} cycle=t0 phase=sumo_advance "
                f"elapsed_s={time.perf_counter() - phase_started:.3f}",
                flush=True,
            )
            for cycle in range(1, self.runtime.config.evaluation_decision_cycles + 1):
                phase_started = time.perf_counter()
                future_jobs = build_temporal_generation_jobs(
                    items, prepared, session.snapshots, cycle=cycle
                )
                # Future control is evaluation-only and must be deterministic.
                future_batch = generation_jobs_to_data_proto(
                    future_jobs, tokenizer, global_steps=global_steps, validate=True
                )
                print(
                    f"[STAGE2_TIMING] split={self.split} cycle=t{cycle} phase=build_jobs "
                    f"jobs={len(future_jobs)} elapsed_s={time.perf_counter() - phase_started:.3f}",
                    flush=True,
                )
                phase_started = time.perf_counter()
                future_generated = sequence_generator(future_batch)
                print(
                    f"[STAGE2_TIMING] split={self.split} cycle=t{cycle} phase=generation "
                    f"jobs={len(future_jobs)} elapsed_s={time.perf_counter() - phase_started:.3f}",
                    flush=True,
                )
                future_groups = response_groups_from_generation(
                    future_generated, future_jobs, tokenizer,
                    num_samples=len(items), num_rollouts=len(prepared[0].assignments),
                )
                phase_started = time.perf_counter()
                coordinator.apply_temporal_cycle(
                    session, future_groups, cycle=cycle,
                    # t3 is the terminal observation/action.  Execute its
                    # action at the current t3 state, but do not advance to
                    # an unintended t4 state.
                    advance=(cycle < self.runtime.config.evaluation_decision_cycles),
                )
                print(
                    f"[STAGE2_TIMING] split={self.split} cycle=t{cycle} phase=sumo_advance "
                    f"elapsed_s={time.perf_counter() - phase_started:.3f}",
                    flush=True,
                )
            phase_started = time.perf_counter()
            results = coordinator.finish_temporal_batch(session)
            print(
                f"[STAGE2_TIMING] split={self.split} phase=finish_reward_cleanup "
                f"elapsed_s={time.perf_counter() - phase_started:.3f} "
                f"total_elapsed_s={time.perf_counter() - temporal_started:.3f}",
                flush=True,
            )
            return results
        except BaseException:
            coordinator.abort_temporal_batch(session)
            raise

    def collect(
        self,
        batch: Any,
        *,
        tokenizer: Any = None,
        materialize: bool = False,
        sequence_generator: Any = None,
        global_steps: int = -1,
        validate: bool = False,
    ) -> OnlineVERLBatch:
        if materialize and tokenizer is None:
            raise ValueError("tokenizer is required when materialize=True")
        specs = specs_from_batch(batch, config=self.config, split=self.split)
        generated = None
        jobs = None
        stage1_jobs = None
        stage1_generated = None
        if sequence_generator is None:
            # Compatibility path for smoke tests and external policy serving.
            result = self.runtime.rollout_and_commit_specs(specs)
        elif validate:
            items = self.runtime.collect_batch_specs(specs)
            stage1_jobs = build_stage1_generation_jobs(items)
            stage1_batch = generation_jobs_to_data_proto(
                stage1_jobs, tokenizer, global_steps=global_steps, validate=False
            )
            stage1_started = time.perf_counter()
            stage1_generated = sequence_generator(stage1_batch)
            print(
                f"[STAGE1_TIMING] generation validate=true jobs={len(stage1_jobs)} "
                f"elapsed_s={time.perf_counter() - stage1_started:.3f}", flush=True
            )
            items = stage1_snapshots_from_generation(
                items, stage1_generated, stage1_jobs, tokenizer
            )
            prepared = self.runtime.coordinator.prepare_deployment_batch_from_master_actors(
                [(item.snapshot, item.master_actor, item.snapshot_dir) for item in items]
            )
            # Stage 2 receives actor-private copies of the exported SUMO
            # state.  Keep the source masters through this validation call so
            # the next ten-row batch continues the same SUMO timeline.
            self.runtime.release_exported_masters(items)
            jobs = build_deployment_generation_jobs(items)
            prompt_batch = generation_jobs_to_data_proto(
                jobs, tokenizer, global_steps=global_steps, validate=True
            )
            stage2_started = time.perf_counter()
            generated = sequence_generator(prompt_batch)
            print(
                f"[STAGE2_TIMING] split={self.split} cycle=t0 phase=generation "
                f"jobs={len(jobs)} elapsed_s={time.perf_counter() - stage2_started:.3f}",
                flush=True,
            )
            attach_stage1_supervision(generated, jobs, stage1_generated, stage1_jobs)
            response_groups = response_groups_from_generation(
                generated, jobs, tokenizer, num_samples=len(items), num_rollouts=1
            )
            rollouts = self._run_temporal(
                items, prepared, response_groups, sequence_generator=sequence_generator,
                tokenizer=tokenizer, global_steps=global_steps,
            )
            result = self.runtime.commit_deployment_rollouts(items, rollouts)
            # ``ray_trainer._validate_online`` resets the complete master
            # pool after its final dataloader batch.  Do not reset here: that
            # would make every ten-row batch redo warm-up from step zero.
        else:
            items = self.runtime.collect_batch_specs(specs)
            stage1_jobs = build_stage1_generation_jobs(items)
            stage1_batch = generation_jobs_to_data_proto(
                stage1_jobs, tokenizer, global_steps=global_steps, validate=False
            )
            stage1_started = time.perf_counter()
            stage1_generated = sequence_generator(stage1_batch)
            print(
                f"[STAGE1_TIMING] generation validate=false jobs={len(stage1_jobs)} "
                f"elapsed_s={time.perf_counter() - stage1_started:.3f}", flush=True
            )
            items = stage1_snapshots_from_generation(
                items, stage1_generated, stage1_jobs, tokenizer
            )
            prepared = self.runtime.coordinator.prepare_batch_from_master_actors(
                [(item.snapshot, item.master_actor, item.snapshot_dir) for item in items], seed=None
            )
            jobs = build_generation_jobs(items, prepared)
            prompt_batch = generation_jobs_to_data_proto(
                jobs, tokenizer, global_steps=global_steps, validate=validate
            )
            stage2_started = time.perf_counter()
            generated = sequence_generator(prompt_batch)
            print(
                f"[STAGE2_TIMING] split={self.split} cycle=t0 phase=generation "
                f"jobs={len(jobs)} elapsed_s={time.perf_counter() - stage2_started:.3f}",
                flush=True,
            )
            attach_stage1_supervision(generated, jobs, stage1_generated, stage1_jobs)
            response_groups = response_groups_from_generation(
                generated,
                jobs,
                tokenizer,
                num_samples=len(items),
                num_rollouts=self.runtime.config.num_rollouts,
            )
            rollouts = self._run_temporal(
                items, prepared, response_groups, sequence_generator=sequence_generator,
                tokenizer=tokenizer, global_steps=global_steps,
            )
            result = self.runtime.commit_rollouts(items, rollouts)
        records = records_for_verl(result)
        if stage1_jobs is not None and stage1_generated is not None:
            stage1_by_key: dict[tuple[int, str], tuple[Any, str, Any, Any]] = {}
            stage1_mmi = getattr(stage1_generated, "non_tensor_batch", {}).get(
                "multi_modal_inputs"
            )
            for index, job in enumerate(stage1_jobs):
                token_ids = stage1_generated.batch["responses"][index]
                if hasattr(token_ids, "tolist"):
                    token_ids = token_ids.tolist()
                stage1_by_key[(int(job["sample_index"]), str(job["intersection_id"]))] = (
                    job["raw_prompt"], tokenizer.decode(token_ids, skip_special_tokens=True).strip(),
                    stage1_mmi[index] if stage1_mmi is not None else {},
                    job["reward_model"]["ground_truth"]["stage1_target"],
                )
            sample_index = {item.spec.sample_id: index for index, item in enumerate(result.items)}
            for record in records:
                key = (sample_index[str(record["extra_info"]["sample_id"])], str(record["intersection_id"]))
                (
                    record["stage1_prompt"], record["stage1_response"],
                    record["perception_sft_multi_modal_inputs"], record["stage1_target"],
                ) = stage1_by_key[key]
        if generated is not None and jobs is not None:
            generated_extra = getattr(generated, "non_tensor_batch", {}).get("extra_fields")
            if generated_extra is not None:
                diagnostics = {}
                for index, job in enumerate(jobs):
                    extra = generated_extra[index]
                    if isinstance(extra, Mapping):
                        route = extra.get("bernoulli_mode_route")
                        if route is not None:
                            diagnostics[(str(job["uid"]), str(job["intersection_id"]), int(job["rollout_id"]))] = route
                for record in records:
                    key = (str(record["uid"]), str(record["intersection_id"]), int(record["rollout_id"]))
                    if key in diagnostics:
                        record["mode_route"] = diagnostics[key]
        validate_verl_records(
            records, expected_rollouts=1 if validate and sequence_generator is not None else self.runtime.config.num_rollouts
        )
        data_proto = (
            records_to_data_proto(records, tokenizer, generated=generated, jobs=jobs)
            if materialize
            else None
        )
        return OnlineVERLBatch(result=result, records=records, data_proto=data_proto)


__all__ = [
    "build_stage1_generation_jobs",
    "build_generation_jobs",
    "build_deployment_generation_jobs",
    "build_temporal_generation_jobs",
    "attach_stage1_supervision",
    "generation_jobs_to_data_proto",
    "OnlineVERLBatch",
    "OnlineVERLCollector",
    "records_for_verl",
    "response_groups_from_generation",
    "stage1_snapshots_from_generation",
    "records_to_data_proto",
    "specs_from_batch",
    "validate_verl_records",
]
