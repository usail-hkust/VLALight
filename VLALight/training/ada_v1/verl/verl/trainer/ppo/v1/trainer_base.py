# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import math
import os
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pprint import pprint
from typing import Any, Optional

import numpy as np
import ray
import torch
import transfer_queue as tq
from omegaconf import DictConfig, OmegaConf, open_dict
from packaging.version import InvalidVersion, Version
from tensordict import TensorDict
from tensordict.tensorclass import NonTensorData, NonTensorStack
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm
from transfer_queue import KVBatchMeta

from verl.checkpoint_engine import CheckpointEngineManager
from verl.experimental.agent_loop import AgentLoopManager
from verl.experimental.reward_loop import RewardLoopManager
from verl.experimental.teacher_loop import MultiTeacherModelManager
from verl.protocol import DataProto, DataProtoFuture
from verl.single_controller.ray import (
    RayClassWithInitArgs,
    RayWorkerGroup,
    ResourcePoolManager,
    create_colocated_worker_cls,
)
from verl.trainer.distillation import is_distillation_enabled
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    RolloutMoELoadBalanceMetricsAccumulator,
    compute_data_metrics,
    compute_moe_lb_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    compute_variance_proxy_metrics,
    get_metric_data_with_optional_routed_experts,
    process_validation_metrics,
)
from verl.trainer.ppo.padding_utils import upsample_batch_to_divisible_size
from verl.trainer.ppo.ray_trainer import apply_kl_penalty, compute_spec_decode_metrics
from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch
from verl.trainer.ppo.utils import (
    Role,
    create_rl_dataset,
    create_rl_sampler,
    need_critic,
    need_reference_policy,
    need_teacher_policy,
)
from verl.trainer.ppo.v1.replay_buffer import DAPO_FILTERED_REWARD_COUNTS_KEY, ReplayBuffer, ReplayBufferAsync
from verl.trainer.ppo.v1.utils import MetricsAggregator, compute_advantage_for_multi_trajectories
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.debug import marked_timer


def _stable_generation_sort_key(key: str) -> tuple:
    """Sort legacy ``sample_rollout_output`` keys and newer online keys safely.

    Online collector keys can contain labels such as ``online``/``lane`` in
    the underscore-delimited suffix.  They are opaque identifiers, so only
    parse the numeric suffix when both components are actually integers.
    """
    parts = str(key).rsplit("_", 2)
    if len(parts) == 3:
        try:
            return (0, parts[0], int(parts[1]), int(parts[2]))
        except (TypeError, ValueError):
            pass
    return (1, str(key), 0, 0)


def _decode_generation_value(tokenizer, value: Any) -> str:
    """Decode token ids defensively; online metadata must never crash logging."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        try:
            ids = [int(token) for token in value]
        except (TypeError, ValueError):
            return str(value)
        return tokenizer.decode(ids, skip_special_tokens=True)
    return str(value)
from verl.utils.debug.metrics import calculate_debug_metrics
from verl.utils.import_utils import load_extern_type
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import rename_dict
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.skip import SkipManager
from verl.utils.tracking import DapoFilteredRewardTableLogger, Tracking, ValidationGenerationsLogger
from verl.workers.config import CriticConfig, DistillationConfig, HFModelConfig
from verl.workers.engine_workers import ActorRolloutRefWorker, TrainingWorker, TrainingWorkerConfig
from verl.workers.rollout.llm_server import LLMServerClient, LLMServerManager
from verl.workers.utils.losses import value_loss
from verl.workers.utils.padding import response_from_nested, response_to_nested


def apply_greedy_sampling_params(params: dict[str, Any]) -> None:
    params["top_p"] = 1.0
    params["top_k"] = -1
    params["temperature"] = 0


logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


def _emit_online_diag(message: str) -> None:
    """Emit compact online-credit diagnostics reliably from Ray workers."""
    logger.info(message)
    # Ray's logging configuration can filter module INFO records.  Keep one
    # flushed stdout copy so branch skips and metadata loss are observable.
    print(message, flush=True)


def _reset_online_validation_runtime(agent_loop_manager: Any) -> None:
    """Release persistent online Val masters after one complete Val pass.

    The V1 online manager keeps Val masters alive across the five dataloader
    waves so each lane advances continuously.  They must be reset before
    Train starts (and after every periodic Val), otherwise their native
    SUMO/Panda3D/EGL processes remain resident and can block the first Train
    materialization RPC.  Non-online managers simply do not expose
    ``reset_split`` and are left unchanged.
    """
    reset_split = getattr(agent_loop_manager, "reset_split", None)
    if not callable(reset_split):
        return
    try:
        reset_split("val")
    except Exception as exc:
        # Cleanup must not hide the original validation exception, but a
        # failed reset is important evidence for diagnosing resource leaks.
        _emit_online_diag(
            f"[V1_ONLINE_RUNTIME_RESET_ERROR] split=val error={type(exc).__name__}: {exc}"
        )
        raise


def _suspend_online_train_runtime(agent_loop_manager: Any) -> None:
    """Release Train's master pool before validation allocates its own pool."""
    suspend = getattr(agent_loop_manager, "suspend_train_masters_for_validation", None)
    if callable(suspend):
        suspend()


def _resume_online_train_runtime(agent_loop_manager: Any) -> None:
    """Restore Train's master pool after validation releases its native resources."""
    resume = getattr(agent_loop_manager, "resume_train_masters_after_validation", None)
    if callable(resume):
        resume()


_PERCEPTION_SFT_SOURCE_KEYS = (
    "perception_sft_responses",
    "perception_sft_input_ids",
    "perception_sft_attention_mask",
    "perception_sft_position_ids",
)
_PERCEPTION_SFT_REQUIRED_KEYS = (
    *_PERCEPTION_SFT_SOURCE_KEYS,
    "grpo_loss_mask",
    "perception_sft_mask",
    "perception_sft_token_weight",
)


def _tensordict_row(data: TensorDict, index: int) -> dict[str, Any]:
    row = {}
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            row[key] = value[index]
        elif isinstance(value, NonTensorData):
            row[key] = value.data
        else:
            row[key] = value[index].data
    return row


def _select_tensordict_rows(data: TensorDict, indices: list[int]) -> TensorDict:
    """Select rows without advanced-indexing nested/jagged tensors."""
    return tu.list_of_dict_to_tensordict([_tensordict_row(data, index) for index in indices])


def _build_perception_sft_actor_batch(batch: KVBatchMeta, rollout_n: int) -> tuple[KVBatchMeta, list[str]]:
    """Append one actor-only gold perception row per rollout group in TransferQueue."""
    data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id)
    missing = [key for key in _PERCEPTION_SFT_REQUIRED_KEYS if key not in data]
    if missing:
        raise ValueError(f"perception SFT is enabled but rollout output is missing {missing}")
    if "perception_sft_multi_modal_inputs" not in data:
        raise ValueError("perception SFT requires dedicated Stage 1 multimodal inputs")
    if "routed_experts" in data:
        raise ValueError("perception SFT is not compatible with rollout routing replay")
    if "teacher_logprobs" in data or "teacher_ids" in data:
        raise ValueError("perception SFT is not compatible with top-k distillation")

    from v35_offline_grpo.perception_sft import build_perception_sft_row_indices

    prompt_uids = []
    online_group_ids = None
    extra_fields = data.get("extra_fields")
    if extra_fields is not None:
        extracted = []
        extra_values = extra_fields.tolist() if hasattr(extra_fields, "tolist") else list(extra_fields)
        for value in extra_values:
            if hasattr(value, "data"):
                value = value.data
            if isinstance(value, dict) and bool(value.get("online", False)):
                extracted.append(str(value.get("decision_group_id", "")))
            else:
                extracted.append("")
        if extracted and all(extracted):
            online_group_ids = extracted
    padding_counts: dict[str, int] = defaultdict(int)
    for key, tag in zip(batch.keys, batch.tags, strict=True):
        uid = online_group_ids[len(prompt_uids)] if online_group_ids is not None else key.split("_", 1)[0]
        if tag.get("is_padding", False):
            # V1 may create several complete padding groups under one UID.
            # Give each rollout_n-sized chunk its own temporary group label.
            padding_index = padding_counts[uid]
            padding_counts[uid] += 1
            uid = f"{uid}:padding-group-{padding_index // rollout_n}"
        prompt_uids.append(uid)
    prompt_uids = np.asarray(prompt_uids, dtype=object)
    effective_rollout_n = rollout_n
    if online_group_ids is not None:
        counts: dict[str, int] = defaultdict(int)
        for uid in prompt_uids:
            counts[str(uid)] += 1
        sizes = set(counts.values())
        if len(sizes) != 1:
            raise ValueError(f"online perception SFT requires uniform decision groups, got {dict(list(counts.items())[:5])}")
        effective_rollout_n = sizes.pop()
        expected = int(os.environ.get("V35_ONLINE_NUM_ROLLOUTS", "6"))
        if effective_rollout_n != expected:
            raise ValueError(
                f"online perception SFT expected {expected} Stage-2 candidates per decision group, "
                f"got {effective_rollout_n}"
            )
        _emit_online_diag(
            f"[ONLINE_SFT_GROUPING] groups={len(counts)} candidates_per_group={effective_rollout_n}"
        )
    gold_indices_list, interleaved_indices_list = build_perception_sft_row_indices(prompt_uids, effective_rollout_n)
    rollout_data = data.clone()
    gold_data = _select_tensordict_rows(data, gold_indices_list)
    gold_data["responses"] = gold_data["perception_sft_responses"]
    gold_data["input_ids"] = gold_data["perception_sft_input_ids"]
    gold_data["attention_mask"] = gold_data["perception_sft_attention_mask"]
    gold_data["position_ids"] = gold_data["perception_sft_position_ids"]
    gold_data["response_mask"] = gold_data["perception_sft_mask"].to(torch.float32)
    gold_data["perception_sft_mask"] = gold_data["response_mask"].clone()
    gold_data["loss_mask"] = gold_data["response_mask"].clone()
    if "perception_sft_multi_modal_inputs" in data:
        gold_data["multi_modal_inputs"] = gold_data["perception_sft_multi_modal_inputs"]
        rollout_data["multi_modal_inputs"] = NonTensorStack.from_list(
            [NonTensorData({}) for _ in range(len(rollout_data))]
        )
    if "grpo_advantage_scale" in gold_data:
        gold_data["grpo_advantage_scale"] = torch.zeros_like(
            gold_data["response_mask"], dtype=torch.float32
        )

    for mask_key, value_key in (
        ("reasoning_aux_mask", "reasoning_aux_advantage"),
        ("signal_aux_mask", "signal_aux_advantage"),
        ("mode_aux_mask", "mode_aux_weight"),
    ):
        if mask_key in gold_data:
            # Online rollout masks are materialized as float32 by the
            # adapter. TransferQueue requires one dtype per field across all
            # rows, so gold rows must use the same dtype rather than inheriting
            # int64 from token masks.
            gold_data[mask_key] = torch.zeros_like(
                gold_data["response_mask"], dtype=torch.float32
            )
            gold_data[value_key] = torch.zeros_like(gold_data["response_mask"], dtype=torch.float32)

    # Gold perception rows do not contain a Stage-2 mode decision.  Disable
    # every binary-router field explicitly so they can share one TransferQueue
    # actor batch without accidentally contributing a zero-prompt classifier
    # target or inheriting the rollout row's mode token ids.
    if "mode_binary_mask" in gold_data:
        gold_data["mode_binary_mask"] = torch.zeros_like(
            gold_data["response_mask"], dtype=rollout_data["mode_binary_mask"].dtype
        )
    for key in ("mode_token_index", "mode_fast_token_id", "mode_slow_token_id"):
        if key in gold_data:
            gold_data[key] = torch.full_like(gold_data[key], -1)
    if "mode_binary_target" in gold_data:
        gold_data["mode_binary_target"] = torch.zeros_like(gold_data["mode_binary_target"])
    if "mode_binary_valid" in gold_data:
        gold_data["mode_binary_valid"] = torch.zeros_like(gold_data["mode_binary_valid"], dtype=torch.bool)
    if "mode_binary_weight" in gold_data:
        gold_data["mode_binary_weight"] = torch.zeros_like(gold_data["mode_binary_weight"], dtype=torch.float32)

    # Preserve an existing kl_loss_mask written by the online adapter.  It is
    # commonly float32 even when response_mask is int64; overwriting it here
    # silently changed the dtype of an already-registered TQ field.
    if "kl_loss_mask" not in rollout_data:
        rollout_data["kl_loss_mask"] = rollout_data["response_mask"].clone()
    rollout_data["perception_sft_mask"] = torch.zeros_like(rollout_data["response_mask"])
    rollout_data["perception_sft_token_weight"] = torch.zeros_like(
        rollout_data["response_mask"], dtype=torch.float32
    )
    gold_data["grpo_loss_mask"] = torch.zeros_like(
        gold_data["response_mask"], dtype=torch.float32
    )

    # Normalize every token-mask field on both sides of the combined actor
    # batch independently.  The original rollout row for each field already
    # exists in TransferQueue, so that field's dtype is authoritative (the
    # dtypes are not necessarily all the same: kl_loss_mask is often float32
    # while response_mask is int64).  Gold rows are converted per field before
    # the combined put, avoiding both int64->float32 and float32->int64
    # production-status conflicts.
    for mask_key in (
        "response_mask", "loss_mask", "grpo_loss_mask", "kl_loss_mask",
        "perception_sft_mask", "reasoning_aux_mask", "signal_aux_mask",
        "mode_aux_mask", "mode_binary_mask",
    ):
        canonical = rollout_data.get(mask_key)
        if not isinstance(canonical, torch.Tensor):
            canonical = rollout_data["response_mask"]
        canonical_dtype = canonical.dtype
        for subset in (rollout_data, gold_data):
            if mask_key in subset:
                subset[mask_key] = subset[mask_key].to(canonical_dtype)

    # Fail locally with the offending field instead of allowing TransferQueue
    # to emit a delayed, partition-level dtype mismatch.
    shared_keys = set(rollout_data.keys()) & set(gold_data.keys())
    for key in shared_keys:
        rollout_value, gold_value = rollout_data[key], gold_data[key]
        if isinstance(rollout_value, torch.Tensor) and isinstance(gold_value, torch.Tensor):
            if rollout_value.dtype != gold_value.dtype:
                # Existing rollout keys are already registered in
                # TransferQueue. Their dtype is authoritative for a combined
                # write; cast the newly-created gold rows to it instead of
                # rejecting the batch or letting TQ fail asynchronously.
                _emit_online_diag(
                    f"[ONLINE_ACTOR_DTYPE_ALIGN] field={key} "
                    f"rollout={rollout_value.dtype} gold={gold_value.dtype}"
                )
                gold_data[key] = gold_value.to(dtype=rollout_value.dtype)
    # Keep the KL mask aligned with the rollout schema.  Legacy/offline
    # fixtures may omit it, in which case the rollout mask created above is
    # the canonical dtype and the gold row must still contain the field.
    gold_data["kl_loss_mask"] = torch.zeros_like(
        gold_data.get("kl_loss_mask", gold_data["response_mask"]),
        dtype=rollout_data["kl_loss_mask"].dtype,
    )

    for key in (
        "old_log_probs",
        "advantages",
        "ref_log_prob",
        "rollout_log_probs",
        "rollout_is_weights",
        "token_level_scores",
        "token_level_rewards",
        "returns",
        "values",
        "rm_scores",
    ):
        if key in gold_data:
            gold_data[key] = torch.zeros_like(gold_data["response_mask"], dtype=gold_data[key].dtype)

    for key in _PERCEPTION_SFT_SOURCE_KEYS:
        rollout_data.pop(key)
        gold_data.pop(key)
    if "perception_sft_multi_modal_inputs" in rollout_data:
        rollout_data.pop("perception_sft_multi_modal_inputs")
        gold_data.pop("perception_sft_multi_modal_inputs")

    rollout_count = len(rollout_data)
    combined_data = tu.list_of_dict_to_tensordict(
        [
            _tensordict_row(rollout_data, index)
            if index < rollout_count
            else _tensordict_row(gold_data, index - rollout_count)
            for index in interleaved_indices_list
        ]
    )
    # Final write barrier: TransferQueue enforces one dtype per field across
    # the whole partition.  Row-wise construction can promote a field (for
    # example int64 rollout masks plus float32 gold masks) even after the
    # subsets were aligned above.  Re-apply the dtype of the original rollout
    # field to the complete combined tensor immediately before kv_batch_put.
    for key in set(rollout_data.keys()) & set(combined_data.keys()):
        source = rollout_data[key]
        merged = combined_data[key]
        if isinstance(source, torch.Tensor) and isinstance(merged, torch.Tensor):
            if merged.dtype != source.dtype:
                _emit_online_diag(
                    f"[ONLINE_ACTOR_DTYPE_WRITE_BARRIER] field={key} "
                    f"merged={merged.dtype} canonical={source.dtype}"
                )
                combined_data[key] = merged.to(dtype=source.dtype)
    gold_keys = [f"{batch.keys[index]}_perception_sft_{uuid.uuid4().hex}" for index in gold_indices_list]
    combined_keys = np.asarray([*batch.keys, *gold_keys], dtype=object)[interleaved_indices_list].tolist()

    gold_tags = []
    gold_attention_mask = gold_data["attention_mask"]
    gold_lengths = (
        gold_attention_mask.offsets().diff().tolist()
        if gold_attention_mask.is_nested
        else gold_attention_mask.sum(dim=-1).tolist()
    )
    for index, seq_len in zip(gold_indices_list, gold_lengths, strict=True):
        tag = dict(batch.tags[index])
        tag.update(seq_len=int(seq_len), is_perception_sft=True)
        gold_tags.append(tag)
    combined_tags = np.asarray([*batch.tags, *gold_tags], dtype=object)[interleaved_indices_list].tolist()

    actor_batch = tq.kv_batch_put(
        keys=combined_keys,
        partition_id=batch.partition_id,
        fields=combined_data,
        tags=combined_tags,
    )
    logger.info(
        "Prepared perception-SFT actor batch: rollout_rows=%d gold_rows=%d actor_rows=%d",
        len(batch),
        len(gold_keys),
        len(combined_keys),
    )
    return actor_batch, gold_keys


def _split_perception_sft_actor_batch(
    actor_batch: KVBatchMeta,
) -> tuple[KVBatchMeta, KVBatchMeta, list[str]]:
    """Split the expanded actor batch into independent SFT and RL batches.

    The combined batch is retained in TransferQueue for compatibility, but
    each update receives only its own rows. The caller can run the two batches
    as separate forward/backward passes while retaining one optimizer step.
    Row selection is performed through the existing NestedTensor-safe row
    materializer.
    """
    data = tq.kv_batch_get(keys=actor_batch.keys, partition_id=actor_batch.partition_id)
    sft_indices = [i for i, tag in enumerate(actor_batch.tags) if tag.get("is_perception_sft", False)]
    sft_index_set = set(sft_indices)
    rl_indices = [i for i in range(len(actor_batch.keys)) if i not in sft_index_set]
    if not sft_indices or not rl_indices:
        raise ValueError("perception SFT actor batch must contain both SFT and rollout rows")

    def make_subset(indices: list[int], suffix: str) -> KVBatchMeta:
        keys = [f"{actor_batch.keys[i]}_{suffix}" for i in indices]
        fields = _select_tensordict_rows(data, indices)
        tags = [dict(actor_batch.tags[i]) for i in indices]
        return tq.kv_batch_put(
            keys=keys,
            partition_id=actor_batch.partition_id,
            fields=fields,
            tags=tags,
        )

    sft_batch = make_subset(sft_indices, "sft_update")
    rl_batch = make_subset(rl_indices, "rl_update")
    return sft_batch, rl_batch, [*sft_batch.keys, *rl_batch.keys]


def _tq_supports_checkpoint() -> bool:
    """Whether the installed TransferQueue can snapshot/restore its state for checkpoint consistency."""
    try:
        version_supported = Version(getattr(tq, "__version__", "")) >= Version("0.1.9")
    except InvalidVersion:
        return False
    return (
        version_supported
        and callable(getattr(tq, "save_checkpoint", None))
        and callable(getattr(tq, "load_checkpoint", None))
    )


class PPOTrainer(ABC):
    """Base class for PPO trainer.

    Args:
        config: DictConfig from yaml config file.
    """

    def __init__(self, config: DictConfig):
        self.config = config
        self.use_critic = need_critic(self.config)
        self.use_reference_policy = need_reference_policy(self.config)
        self.use_teacher_policy = need_teacher_policy(self.config)
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self.trainer_mode = self.config.trainer.v1.trainer_mode
        self.parameter_sync_step = self.config.trainer.v1.get(self.trainer_mode, {}).get("parameter_sync_step", 1)
        self.replay_buffer = self._build_replay_buffer()
        self._rollout_moe_lb_metrics_accumulator = RolloutMoELoadBalanceMetricsAccumulator(
            model_config=self.config.actor_rollout_ref.model
        )
        # track mini-batch index within a parameter_sync_step cycle for Decoupled PPO
        self.local_trigger_step = 0

    def _build_replay_buffer(self) -> ReplayBuffer:
        """Instantiate the replay buffer (or a user-provided custom sampler).

        Set ``trainer.v1.sampler.custom_sampler.{path,name}`` to plug in a custom
        ``ReplayBuffer`` subclass; otherwise the built-in implementation is used.
        """
        sampler_config = self.config.trainer.v1.sampler
        custom_sampler = sampler_config.get("custom_sampler", None)
        has_custom_sampler = bool(
            custom_sampler is not None and custom_sampler.get("path") and custom_sampler.get("name")
        )
        if has_custom_sampler:
            sampler_cls = load_extern_type(custom_sampler.path, custom_sampler.name)
        else:
            sampler_cls = ReplayBuffer if self.trainer_mode == "sync" else ReplayBufferAsync

        replay_buffer_kwargs = dict(
            trainer_mode=self.trainer_mode,
            trainer_config=self.config.trainer.v1.get(self.trainer_mode, {}),
            max_off_policy_threshold=sampler_config.max_off_policy_threshold,
            max_off_policy_strategy=sampler_config.max_off_policy_strategy,
            sampler_kwargs=sampler_config.sampler_kwargs,
            refill_fn=self._add_prompts_to_generate,
        )
        # Preserve the existing constructor contract for external samplers; custom implementations own
        # their filtering semantics and can consume algorithm.filter_groups through their own config.
        if not has_custom_sampler:
            filter_groups_metric = self._resolve_filter_groups_metric()
            sync_refill_failed_groups = bool(sampler_config.get("sync_refill_failed_groups", False))
            replay_buffer_kwargs.update(
                filter_groups_metric=filter_groups_metric,
                sync_refill_failed_groups=sync_refill_failed_groups,
            )
            if sampler_cls is ReplayBuffer:
                filter_groups = self.config.algorithm.get("filter_groups", None)
                max_inflight_gen_batches = 1
                if filter_groups_metric is not None:
                    max_inflight_gen_batches = filter_groups.get("max_inflight_gen_batches", 1)
                train_batch_size = self.config.data.train_batch_size
                replay_buffer_kwargs.update(
                    train_batch_size=train_batch_size,
                    gen_batch_size=1
                    if filter_groups_metric is not None or sync_refill_failed_groups
                    else (self.config.data.get("gen_batch_size", None) or train_batch_size),
                    max_inflight_gen_batches=max_inflight_gen_batches,
                )
        return sampler_cls(**replay_buffer_kwargs)

    def _resolve_filter_groups_metric(self) -> str | None:
        """Resolve DAPO's group metric and verify that rollout computes it before sampling."""
        filter_groups = self.config.algorithm.get("filter_groups", None)
        filter_enabled = bool(filter_groups is not None and filter_groups.get("enable", False))
        if not filter_enabled:
            return None

        filter_metric = filter_groups.get("metric", None)
        if not filter_metric:
            raise ValueError("algorithm.filter_groups.metric must be set when group filtering is enabled")

        reward_model = self.config.reward.reward_model
        streaming_reward_path = not reward_model.enable or reward_model.enable_resource_pool
        assert streaming_reward_path, (
            "algorithm.filter_groups requires the reward metric at sampling time: use rule-based reward or "
            "reward.reward_model.enable_resource_pool=True. A colocated reward model computes rewards only "
            "after replay-buffer sampling."
        )
        max_num_gen_batches = filter_groups.get("max_num_gen_batches", 0)
        if max_num_gen_batches > 0:
            logger.warning(
                "algorithm.filter_groups.max_num_gen_batches=%s is ignored by the built-in V1 ReplayBuffer; "
                "use max_inflight_gen_batches to bound concurrent Sync DAPO generation.",
                max_num_gen_batches,
            )
        return str(filter_metric)

    def init(self):
        """Initialize all components of the trainer.

        1. WorkerGroup: actor, critic, reference with model engine: FSDP/Megatron/VeOmni/...
        2. LLMServerManager: launch and manage LLM server replicas for generation.
        3. CheckpointEngineManager: sync weights between worker group and LLM server replicas.
        4. RewardLoopManager: reward workers for rule-based reward, optional LLM server for model-based reward.
        5. [Optional] MultiTeacherModelManager: LLM teacher servers for on-policy distillation.
        """
        self._setup()
        self.on_init_end()

    def _setup(self):
        self._init_tokenizer()
        self._init_dataloader()
        self._init_dump_executor()
        self._init_resource_pool_mgr()
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # 1. define actor and rollout class
        actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
        actor_rollout_resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
        actor_rollout_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[actor_role],
            config=self.config.actor_rollout_ref,
            distillation_config=self.config.get("distillation"),
            role=str(actor_role),
        )
        self.resource_pool_to_cls[actor_rollout_resource_pool][str(actor_role)] = actor_rollout_cls

        # 2. define critic class
        if self.use_critic:
            critic_cfg: CriticConfig = omega_conf_to_dataclass(self.config.critic)
            critic_cfg.engine.infer_max_token_len_per_gpu = critic_cfg.ppo_infer_max_token_len_per_gpu
            critic_cfg.engine.max_token_len_per_gpu = critic_cfg.ppo_infer_max_token_len_per_gpu

            # Wire the critic profiler config via the hydra path (real dataclass tool_config), so the
            # standalone critic TrainingWorker gets a working DistProfiler instead of a silent no-op.
            critic_omega_profiler_config = self.config.critic.get("profiler", {})
            critic_profiler_config = (
                omega_conf_to_dataclass(critic_omega_profiler_config) if critic_omega_profiler_config else None
            )

            worker_cfg = TrainingWorkerConfig(
                model_type="value_model",
                model_config=critic_cfg.model,
                engine_config=critic_cfg.engine,
                optimizer_config=critic_cfg.optim,
                checkpoint_config=critic_cfg.checkpoint,
                profiler_config=critic_profiler_config,
            )
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=worker_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # 3. create worker group for actor rollout and critic
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.config.trainer.device
        logger.info(f"worker group kwargs: {wg_kwargs}")

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            if not class_dict:
                continue
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = RayWorkerGroup(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            logger.info(f"create worker group {spawn_wg.keys()}")

        # 5. initialize critic model engine
        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            self.critic_wg.reset()
            value_loss_ = partial(value_loss, config=critic_cfg)
            self.critic_wg.set_loss_fn(value_loss_)
            logger.info("critic model engine initialized")

        # 6. initialize actor and ref model engine
        self.actor_rollout_wg = all_wg[str(actor_role)]
        self.actor_rollout_wg.init_model()
        logger.info("actor and ref model engine initialized")

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        lora_rank = self.config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = self.config.actor_rollout_ref.model.get("lora_rank", 0)
        self.ref_in_actor = lora_rank > 0 or self.config.actor_rollout_ref.model.get("lora_adapter_path") is not None
        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg[str(actor_role)]
        if self.ref_in_actor:
            self.ref_policy_wg = self.actor_rollout_wg

        # 7. initialize reward loop manager
        resource_pool = (
            self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            if self.config.reward.reward_model.enable
            else None
        )
        self.reward_loop_manager = RewardLoopManager(
            config=self.config,
            rm_resource_pool=resource_pool,
        )
        logger.info("reward loop manager initialized")

        # 8. initialize teacher loop manager
        if self.use_teacher_policy:
            teacher_resource_pool = self.resource_pool_manager.get_resource_pool(Role.TeacherModel)
            self.teacher_model_manager = MultiTeacherModelManager(
                config=self.config,
                resource_pool=teacher_resource_pool,
            )
            self.distillation_config: DistillationConfig = omega_conf_to_dataclass(self.config.distillation)
        else:
            self.teacher_model_manager = None
            self.distillation_config = None

        # 9. initialize agent loop manager
        self.llm_server_manager: LLMServerManager = LLMServerManager.create(
            config=self.config, worker_group=self.actor_rollout_wg, rollout_resource_pool=actor_rollout_resource_pool
        )

        # 10. initialize checkpoint engine manager
        checkpoint_engine_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine)
        checkpoint_engine_config.backend = "naive"
        self.checkpoint_manager: CheckpointEngineManager = CheckpointEngineManager(
            config=checkpoint_engine_config,
            actor_wg=self.actor_rollout_wg,
            replicas=self.llm_server_manager.get_replicas(),
        )
        logger.info("checkpoint engine manager initialized")

        # sleep all replicas to load checkpoint
        self.checkpoint_manager.sleep_replicas()
        self._load_checkpoint()

        logger.info("all initialize finished, ready to fit")

    def get_llm_client(self) -> LLMServerClient:
        """Get the LLM server client for rollout generation."""
        return self.llm_server_manager.get_client()

    def get_teacher_client(self) -> Optional[dict[str, LLMServerClient]]:
        """Get the On-Policy Distillation teacher server clients.

        Returns:
            dict[str, LLMServerClient]: The teacher server clients.
        """
        return self.teacher_model_manager.get_client() if self.use_teacher_policy else None

    def get_reward_handles(self) -> list[ray.actor.ActorHandle]:
        """Get the handles of reward loop workers."""
        return self.reward_loop_manager.reward_loop_worker_handles

    def fit(self, agent_loop_manager: AgentLoopManager):
        """Fit the trainer with the agent loop manager.

        Args:
            agent_loop_manager: The agent loop manager to generate sequences.
        """
        self.agent_loop_manager = agent_loop_manager

        # initialize SkipManager for V1 rollout skip support
        SkipManager.init(self.config)

        self.logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )
        self.dapo_filtered_reward_logger = DapoFilteredRewardTableLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # perform validation before training
        if self.config.trainer.get("val_before_train", True):
            self.on_validate_begin()
            _suspend_online_train_runtime(self.agent_loop_manager)
            try:
                val_metrics = self._validate()
            finally:
                # Do not let Val's persistent SUMO/EGL pool overlap the first
                # Train materialization.  The reset is intentionally outside
                # ``_validate`` so all validation dataloader waves can share
                # their ten lanes, while cleanup still runs on exceptions.
                try:
                    _reset_online_validation_runtime(self.agent_loop_manager)
                finally:
                    _resume_online_train_runtime(self.agent_loop_manager)
            self.on_validate_end()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            self.logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                self._shutdown_dump_executor()
                return

        current_epoch = self.global_steps // self.steps_per_epoch
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        # SkipManager skips warmup batches in async trainers, so it doesn't conflict with reissue.
        SkipManager.set_step(self.global_steps)
        self._reissue_inflight_prompts()
        self.prev_step_profile = False
        self.curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        self.next_step_profile = False

        self.on_train_begin()
        last_val_metrics = None
        while current_epoch < self.config.trainer.total_epochs and self.global_steps <= self.total_training_steps:
            is_last_step = self.global_steps >= self.total_training_steps
            metrics = {}
            self.timing_raw = {}

            # 1. perform rollout and actor/critic training
            with marked_timer("step", self.timing_raw):
                self.on_step_begin()

                self._start_profiling()
                batch = self.step(metrics, self.timing_raw)
                self._stop_profiling()

                # 2. save checkpoint
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                ):
                    with marked_timer("save_checkpoint", self.timing_raw, color="green"):
                        self._save_checkpoint()

                self.on_step_end()
                metrics.update(self._consume_sync_metrics())

            # 4. validate
            if self.config.trainer.test_freq > 0 and (
                is_last_step or self.global_steps % self.config.trainer.test_freq == 0
            ):
                with marked_timer("testing", self.timing_raw, color="green"):
                    self.on_validate_begin()
                    _suspend_online_train_runtime(self.agent_loop_manager)
                    try:
                        val_metrics: dict = self._validate()
                    finally:
                        try:
                            _reset_online_validation_runtime(self.agent_loop_manager)
                        finally:
                            _resume_online_train_runtime(self.agent_loop_manager)
                    self.on_validate_end()
                    if is_last_step:
                        last_val_metrics = val_metrics
                metrics.update(val_metrics)

            # 5. record metrics
            self._compute_metrics(batch, metrics, self.timing_raw, global_steps=self.global_steps, epoch=current_epoch)

            # 6. dump rollout generations if enabled
            rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
            if rollout_data_dir:
                self._log_rollout_data(batch, self.timing_raw, rollout_data_dir)

            # 7. cleanup transfer queue
            tq.kv_clear(keys=batch.keys, partition_id=batch.partition_id)

            dapo_filtered_reward_counts = metrics.pop(DAPO_FILTERED_REWARD_COUNTS_KEY, None)
            self.logger.log(data=metrics, step=self.global_steps)
            if dapo_filtered_reward_counts:
                self.dapo_filtered_reward_logger.log(
                    self.config.trainer.logger, dapo_filtered_reward_counts, self.global_steps
                )
            progress_bar.update(1)
            self.global_steps += 1
            SkipManager.set_step(self.global_steps)
            current_epoch = (self.global_steps - 1) // self.steps_per_epoch
            if is_last_step:
                self._shutdown_dump_executor()
                pprint(f"Final validation metrics: {last_val_metrics}")
                progress_bar.close()
                return

        self.on_train_end()
        # Ensure dump executor is shut down when training loop ends without reaching is_last_step
        self._shutdown_dump_executor()

    def step(self, metrics: dict, timing_raw: dict) -> KVBatchMeta:
        train_batch_size = self.config.data.train_batch_size
        assert train_batch_size % self.parameter_sync_step == 0, (
            f"train_batch_size ({train_batch_size}) must be divisible by "
            f"parameter_sync_step ({self.parameter_sync_step})"
        )
        sample_batch_size = train_batch_size // self.parameter_sync_step

        self._add_batch_to_generate()

        metrics_aggregator = MetricsAggregator()
        combined_keys: list = []
        combined_tags: list = []
        combined_partition_id = "train"
        for trigger_idx in range(self.parameter_sync_step):
            self.local_trigger_step = trigger_idx
            iter_metrics: dict = {}
            batch = self._step_once(iter_metrics, timing_raw, sample_batch_size)
            sample_count = sum(not tag.get("is_padding", False) for tag in batch.tags)
            metrics_aggregator.add_step_metrics(iter_metrics, sample_count=sample_count)
            combined_keys.extend(batch.keys)
            combined_tags.extend(batch.tags)
            combined_partition_id = batch.partition_id

        metrics.update(metrics_aggregator.get_aggregated_metrics())
        return KVBatchMeta(partition_id=combined_partition_id, keys=combined_keys, tags=combined_tags)

    def _step_once(self, metrics: dict, timing_raw: dict, sample_batch_size: int) -> KVBatchMeta:
        """Run a single local update: sample one mini-batch and perform the full PPO pipeline once."""
        # 1. sample batch from replay buffer
        with marked_timer("gen", timing_raw, color="red"):
            self.on_sample_begin()
            batch, off_policy_metrics = self.replay_buffer.sample(
                global_steps=self.global_steps,
                partition_id="train",
                batch_size=sample_batch_size,
            )
            metrics.update(off_policy_metrics)
            batch.extra_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
            self.on_sample_end()

        # 2. [OPTIONAL] compute reward score with colocated reward model
        if self.reward_loop_manager.reward_loop_worker_handles is None:
            with marked_timer("reward", timing_raw, color="yellow"):
                batch = self._compute_reward_colocate(batch, metrics=metrics)

        # 3. balance batch across data parallel groups
        batch = self._balance_batch(batch, metrics=metrics)

        # 4. compute old_log_prob
        with marked_timer("old_log_prob", timing_raw, color="blue"):
            batch = self._compute_old_log_prob(batch, metrics=metrics)

        # 5. [OPTIONAL] compute ref_log_prob
        if self.use_reference_policy:
            with marked_timer("ref", timing_raw, color="olive"):
                batch = self._compute_ref_log_prob(batch, metrics=metrics)

        # 6. [OPTIONAL] compute critic values
        if self.use_critic:
            with marked_timer("values", timing_raw, color="cyan"):
                batch = self._compute_values(batch, metrics=metrics)

        # 7. compute advantage and return
        with marked_timer("adv", timing_raw, color="brown"):
            batch = self._compute_advantage(batch, metrics=metrics)

        # 8. [OPTIONAL] update critic
        if self.use_critic:
            with marked_timer("update_critic", timing_raw, color="pink"):
                batch = self._update_critic(batch, metrics=metrics)

        # 9. update actor
        if self.config.trainer.critic_warmup <= self.global_steps:
            with marked_timer("update_actor", timing_raw, color="red"):
                perception_sft_coef = float(os.environ.get("V35_PERCEPTION_SFT_COEF", "0"))
                if perception_sft_coef > 0.0:
                    actor_batch, gold_keys = _build_perception_sft_actor_batch(
                        batch, rollout_n=self.config.actor_rollout_ref.rollout.n
                    )
                    try:
                        sft_rows = sum(tag.get("is_perception_sft", False) for tag in actor_batch.tags)
                        _emit_online_diag(
                            f"[ONLINE_ACTOR_JOINT_UPDATE] sft_rows={sft_rows} "
                            f"rl_rows={len(actor_batch.keys) - sft_rows} "
                            f"actor_rows={len(actor_batch.keys)} updates=1 "
                            "branches=perception_sft+mode+reasoning+signal"
                        )
                        # Keep the rows separate for peak-memory control, but
                        # accumulate both backward passes before one optimizer
                        # step. Each sub-update disables optimizer stepping.
                        sft_batch, rl_batch, _ = _split_perception_sft_actor_batch(actor_batch)
                        self.actor_rollout_wg.actor_optimizer_zero_grad()
                        _emit_online_diag(
                        f"[ONLINE_ACTOR_GRAD_ACCUM_BEGIN] sft_rows={len(sft_batch.keys)} "
                            f"rl_rows={len(rl_batch.keys)} optimizer_step=0 "
                            f"sft_micro={os.environ.get('V35_SFT_MICRO_BATCH_SIZE_PER_GPU', '2')} "
                            f"rl_micro={os.environ.get('V35_RL_MICRO_BATCH_SIZE_PER_GPU', '8')}"
                        )
                        self._update_actor(
                            sft_batch,
                            metrics=metrics,
                            skip_optimizer_zero_grad=True,
                            skip_optimizer_step=True,
                            metric_namespace="grad_accum_sft",
                            micro_batch_size_per_gpu=int(os.environ.get("V35_SFT_MICRO_BATCH_SIZE_PER_GPU", "2")),
                        )
                        self._update_actor(
                            rl_batch, metrics=metrics,
                            skip_optimizer_zero_grad=True,
                            skip_optimizer_step=True,
                            metric_namespace="grad_accum_rl",
                            micro_batch_size_per_gpu=int(os.environ.get("V35_RL_MICRO_BATCH_SIZE_PER_GPU", "8")),
                        )
                        self.actor_rollout_wg.actor_optimizer_step(expect_nonzero_grad=True)
                        _emit_online_diag(
                            f"[ONLINE_ACTOR_GRAD_ACCUM_FINAL] sft_rows={len(sft_batch.keys)} "
                            f"rl_rows={len(rl_batch.keys)} optimizer_step=1"
                        )
                        tq.kv_clear(keys=list(sft_batch.keys) + list(rl_batch.keys), partition_id=actor_batch.partition_id)
                    finally:
                        tq.kv_clear(keys=gold_keys, partition_id=batch.partition_id)
                else:
                    batch = self._update_actor(batch, metrics=metrics)

        return batch

    # ------------------------------ abstract methods ------------------------------

    def on_init_end(self):
        """Called after the initialization ends."""
        return

    def on_train_begin(self):
        """Called before the training loop starts."""
        return

    def on_train_end(self):
        """Called after the training loop ends."""
        return

    def on_validate_begin(self):
        """Called before the validation loop starts."""
        return

    def on_validate_end(self):
        """Called after the validation loop ends."""
        return

    def on_step_begin(self):
        """Called at the beginning of each training step."""
        return

    @abstractmethod
    def on_step_end(self):
        """Called at the end of each training step."""
        return

    def _consume_sync_metrics(self) -> dict:
        """Weight-sync stats stashed by ``on_step_end`` (e.g. the delta engines'
        changed ratio / wire payload), merged into this step's logged metrics."""
        metrics = getattr(self, "_pending_sync_metrics", None) or {}
        self._pending_sync_metrics = {}
        return metrics

    def on_sample_begin(self):
        """Called at the beginning of sampling batch from replay buffer."""
        return

    @abstractmethod
    def on_sample_end(self):
        """Called after sampling a batch from replay buffer."""
        return

    # ------------------------------ common methods ------------------------------

    def _get_n_gpus_for_throughput(self) -> int:
        """Return the total number of GPUs used for throughput normalization.

        By default this is the trainer-side GPU count from the resource pool
        manager.  Modes that use additional dedicated GPUs (e.g. separate-async
        standalone rollout) should override this to include them.
        """
        return self.resource_pool_manager.get_n_gpus()

    def _init_tokenizer(self):
        """Initialize tokenizer and processor from the model config."""
        model_config: HFModelConfig = omega_conf_to_dataclass(self.config.actor_rollout_ref.model)
        self.tokenizer = model_config.tokenizer
        # Used for multimodal LLM, could be None
        self.processor = model_config.processor

    def _init_dataloader(self):
        """Initialize train and validate dataloader."""
        self.train_dataset = create_rl_dataset(
            self.config.data.train_files,
            self.config.data,
            self.tokenizer,
            self.processor,
            is_train=True,
            max_samples=self.config.data.get("train_max_samples", -1),
        )
        self.val_dataset = create_rl_dataset(
            self.config.data.val_files,
            self.config.data,
            self.tokenizer,
            self.processor,
            is_train=False,
            max_samples=self.config.data.get("val_max_samples", -1),
        )

        # Exact refill counts require single-prompt dataloader fetches.
        filter_groups = self.config.algorithm.get("filter_groups", None)
        dapo_enabled = bool(filter_groups is not None and filter_groups.get("enable", False))
        sync_refill_failed_groups = bool(self.config.trainer.v1.sampler.get("sync_refill_failed_groups", False))
        requires_exact_refill = self.trainer_mode != "sync" or dapo_enabled or sync_refill_failed_groups
        if requires_exact_refill:
            user_gen_batch_size = self.config.data.get("gen_batch_size", None)
            if user_gen_batch_size not in (None, 1):
                logger.warning(f"data.gen_batch_size={user_gen_batch_size} is overridden to 1.")
            elif user_gen_batch_size is None:
                logger.info("data.gen_batch_size defaulted to 1.")
            with open_dict(self.config):
                self.config.data.gen_batch_size = 1

        # use gen_batch_size as the batch size for the dataloader if set, otherwise use train_batch_size
        gen_batch_size = self.config.data.get("gen_batch_size", None) or self.config.data.train_batch_size
        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=gen_batch_size,
            num_workers=self.config.data["dataloader_num_workers"],
            drop_last=True,
            collate_fn=collate_fn,
            sampler=create_rl_sampler(self.config.data, self.train_dataset),
        )
        self.train_dataloader_it = None
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=self.config.data.val_batch_size or len(self.val_dataset),
            num_workers=self.config.data["dataloader_num_workers"],
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )
        logger.info(
            f"train and validate dataloader initialized, train dataset size: "
            f"{len(self.train_dataset)}, val dataset size: {len(self.val_dataset)}"
        )

        self.steps_per_epoch = len(self.train_dataset) // self.config.data.train_batch_size

        # adjust total_training_steps
        total_training_steps = self.steps_per_epoch * self.config.trainer.total_epochs
        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps
        self.total_training_steps = total_training_steps
        logger.info(f"Total training steps: {self.total_training_steps}")

        # The LR scheduler steps once per local update, and each global step performs
        # ``parameter_sync_step`` local updates (see ``PPOTrainer.step``). The optimizer's
        # schedule horizon must therefore count optimizer updates.
        optim_total_training_steps = total_training_steps * self.parameter_sync_step
        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = optim_total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = optim_total_training_steps
        except Exception as e:
            logger.warning(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _init_resource_pool_mgr(self):
        config = self.config
        # role => worker class
        self.role_worker_mapping = {}
        # role => resource pool
        self.mapping = {}

        # Add actor rollout worker to mapping
        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None

        role = Role.ActorRolloutRef if need_reference_policy(config) and not ref_in_actor else Role.ActorRollout
        self.role_worker_mapping[role] = ray.remote(ActorRolloutRefWorker)
        self.mapping[role] = "global_pool"

        # Add critic worker to mapping.
        if need_critic(config):
            self.role_worker_mapping[Role.Critic] = ray.remote(TrainingWorker)
            self.mapping[Role.Critic] = "global_pool"

        # Global resource pool is used for actor, rollout, critic, ref
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }

        # Add separate resource pool for reward model if enabled
        if config.reward.reward_model.enable_resource_pool:
            if config.reward.reward_model.n_gpus_per_node <= 0:
                raise ValueError("config.reward.reward_model.n_gpus_per_node must be greater than 0")
            if config.reward.reward_model.nnodes <= 0:
                raise ValueError("config.reward.reward_model.nnodes must be greater than 0")

            reward_pool = [config.reward.reward_model.n_gpus_per_node] * config.reward.reward_model.nnodes
            resource_pool_spec["reward_pool"] = reward_pool
            self.mapping[Role.RewardModel] = "reward_pool"
        else:
            config.reward.reward_model.nnodes = config.trainer.nnodes
            config.reward.reward_model.n_gpus_per_node = config.trainer.n_gpus_per_node
            self.mapping[Role.RewardModel] = "global_pool"

        distillation_config = config.get("distillation")
        if is_distillation_enabled(distillation_config):
            if distillation_config.n_gpus_per_node <= 0:
                raise ValueError("config.distillation.n_gpus_per_node must be greater than 0")
            if distillation_config.nnodes <= 0:
                raise ValueError("config.distillation.nnodes must be greater than 0")

            teacher_pool = [distillation_config.n_gpus_per_node] * distillation_config.nnodes
            resource_pool_spec["teacher_pool"] = teacher_pool
            self.mapping[Role.TeacherModel] = "teacher_pool"

        self.resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=self.mapping)

    def _load_checkpoint(self):
        self.global_steps = 0

        # 1. find latest checkpoint folder
        if self.config.trainer.resume_mode == "disable":
            return
        elif self.config.trainer.resume_mode == "auto":
            checkpoint_folder = self.config.trainer.default_local_dir
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest
            if global_step_folder is None:
                logger.info("Training from scratch")
                return
        elif self.config.trainer.resume_mode == "resume_path":
            assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
            assert "global_step_" in self.config.trainer.resume_from_path, "resume ckpt must specify the global_steps"
            global_step_folder = self.config.trainer.resume_from_path
            if not os.path.isabs(global_step_folder):
                working_dir = os.getcwd()
                global_step_folder = os.path.join(working_dir, global_step_folder)
        else:
            logger.exception(f"Unknown resume mode {self.config.trainer.resume_mode}")

        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])
        logger.info(f"Resuming from {global_step_folder}, setting global step to {self.global_steps}")

        # 2. load actor checkpoint
        self.actor_rollout_wg.load_checkpoint(
            local_path=os.path.join(global_step_folder, "actor"),
            del_local_after_load=self.config.trainer.del_local_ckpt_after_load,
        )

        # 3. load critic checkpoint
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                local_path=os.path.join(global_step_folder, str(Role.Critic)),
                del_local_after_load=self.config.trainer.del_local_ckpt_after_load,
            )

        # 4. load dataloader checkpoint
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            logger.warning(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

        # 5. restore TransferQueue state (async modes). Re-issuing the restored in-flight prompts is
        # deferred to fit() to use the agent_loop_manager.
        if self.trainer_mode != "sync" and _tq_supports_checkpoint():
            tq_ckpt_path = os.path.join(global_step_folder, "transfer_queue")
            if os.path.exists(tq_ckpt_path):
                logger.info(f"Loading TransferQueue state from {tq_ckpt_path}")
                tq.load_checkpoint(tq_ckpt_path)

    def _reissue_inflight_prompts(self, partition_id: str = "train") -> int:
        """Restart checkpointed pending/running prompt groups from their persisted prompt data."""
        if self.trainer_mode == "sync" or not _tq_supports_checkpoint():
            return 0
        data = tq.kv_list(partition_id)
        if not data:
            return 0
        items = data.get(partition_id, {})
        inflight_uids = [
            key
            for key, tag in items.items()
            if tag.get("is_prompt", False) and tag.get("status") in ("pending", "running")
        ]
        if not inflight_uids:
            return 0

        batch = tq.kv_batch_get(keys=inflight_uids, partition_id=partition_id)
        inflight_uid_set = set(inflight_uids)
        old_trajectory_keys = [
            key
            for key, tag in items.items()
            if not tag.get("is_prompt", False) and key.split("_", 1)[0] in inflight_uid_set
        ]
        if old_trajectory_keys:
            tq.kv_clear(keys=old_trajectory_keys, partition_id=partition_id)

        # Treat this as a new dispatch attempt for the resumed training step.
        tu.assign_non_tensor_data(batch, "global_steps", self.global_steps)
        tags = [{"is_prompt": True, "status": "pending", "global_steps": self.global_steps} for _ in inflight_uids]
        tq.kv_batch_put(keys=inflight_uids, partition_id=partition_id, tags=tags)
        self.agent_loop_manager.generate_sequences(batch)

        logger.info(
            f"Re-issued {len(inflight_uids)} in-flight prompts for step {self.global_steps}; "
            f"cleared {len(old_trajectory_keys)} old trajectories from partition {partition_id}"
        )
        return len(inflight_uids)

    def _save_checkpoint(self):
        """Save actor, critic, and dataloader checkpoints to local (and optionally remote) storage."""
        from verl.utils.fs import local_mkdir_safe

        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )
        logger.info(f"Saving checkpoint to {local_global_step_folder}")

        # resolve max checkpoints to keep
        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            logger.warning(
                "remove_previous_ckpt_in_save is deprecated, "
                "set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        # save actor
        actor_local_path = os.path.join(local_global_step_folder, "actor")
        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )
        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        # save critic
        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader state
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        torch.save(self.train_dataloader.state_dict(), dataloader_local_path)

        # save TransferQueue state for async modes so in-flight prompts (already fetched from the
        # dataloader but not yet trained into this checkpoint's weights) survive a restart:
        # finished trajectories are restored as-is, pending/running prompts are re-issued on resume.
        # Requires a TransferQueue release with checkpoint support (see _tq_supports_checkpoint).
        if self.trainer_mode != "sync" and _tq_supports_checkpoint():
            tq.save_checkpoint(
                os.path.join(local_global_step_folder, "transfer_queue"),
                metadata={"global_steps": self.global_steps},
            )

        # write latest checkpointed iteration tracker for atomic resume
        actor_ckpt_cfg = self.config.actor_rollout_ref.actor.get("checkpoint", {})
        if actor_ckpt_cfg.get("async_save", False):
            logger.info("skip write latest_checkpointed_iteration.txt when async_save is True")
            return
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _validate(self) -> dict[str, float]:
        # Lists to collect samples for the table
        sample_uids = []
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        data_sources = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)
        dump_all_inputs: list[str] = []
        dump_all_outputs: list[str] = []
        dump_all_keys: list[str] = []
        session_to_sample_idx: dict[str, int] = {}

        for batch_dict in self.val_dataloader:
            # 1. put batch to agent loop manager
            batch_dict["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(batch_dict["raw_prompt"]))], dtype=object
            )
            batch = tu.get_tensordict(batch_dict)
            tu.assign_non_tensor_data(batch, "global_steps", self.global_steps)
            tu.assign_non_tensor_data(batch, "validate", True)
            # Register each prompt (GRPO group) in TransferQueue as a tag-only status marker.
            # global_steps is required by ReplayBuffer's metadata sync / staleness ordering.
            tags = [
                {"is_prompt": True, "status": "pending", "global_steps": self.global_steps} for _ in range(len(batch))
            ]
            tq.kv_batch_put(keys=list(batch["uid"]), partition_id="val", tags=tags)
            self.agent_loop_manager.generate_sequences(batch)

            # 2. sample batch from replay buffer: one prompt (GRPO group) per submitted row.
            batch, _ = self.replay_buffer.sample(
                global_steps=self.global_steps, partition_id="val", batch_size=len(batch)
            )

            # 3. [OPTIONAL] compute reward score with colocated reward model
            if self.reward_loop_manager.reward_loop_worker_handles is None:
                self.checkpoint_manager.sleep_replicas()
                batch = self._compute_reward_colocate(batch)
                self.checkpoint_manager.update_weights()

            # 4. collect necessary data for logging
            # For multi-output agent loops, only use the final output per session for metrics.
            # Keys have format {uid}_{session_id}_{index}; keep only the highest index per session.
            session_max: dict[str, tuple[int, int]] = {}  # session_key -> (max_index, position)
            for pos, key in enumerate(batch.keys):
                parts = key.rsplit("_", 2)
                if len(parts) == 3:
                    session_key = f"{parts[0]}_{parts[1]}"
                    index = int(parts[2])
                    if session_key not in session_max or index > session_max[session_key][0]:
                        session_max[session_key] = (index, pos)
                else:
                    session_max[key] = (0, pos)
            sorted_sessions = sorted(session_max.items(), key=lambda x: x[1][1])
            final_indices = [pos for _, (_, pos) in sorted_sessions]
            final_keys = [batch.keys[i] for i in final_indices]
            base_offset = len(sample_scores)
            session_to_sample_idx.update(
                {session_key: base_offset + j for j, (session_key, _) in enumerate(sorted_sessions)}
            )

            text_data = tq.kv_batch_get(
                keys=batch.keys, partition_id=batch.partition_id, select_fields=["prompts", "responses"]
            )
            text_data["prompts"] = text_data["prompts"].to_padded_tensor(padding=self.tokenizer.pad_token_id)
            text_data["responses"] = text_data["responses"].to_padded_tensor(padding=self.tokenizer.pad_token_id)
            all_inputs = [_decode_generation_value(self.tokenizer, ids) for ids in text_data["prompts"]]
            all_outputs = [_decode_generation_value(self.tokenizer, ids) for ids in text_data["responses"]]

            fields = ["uid", "rm_scores", "num_turns", "reward_model", "data_source", "extra_fields"]
            data = tq.kv_batch_get(keys=final_keys, partition_id=batch.partition_id, select_fields=fields)

            sample_uids.extend(data.pop("uid").tolist())
            sample_outputs.extend(all_outputs[i] for i in final_indices)
            sample_inputs.extend(all_inputs[i] for i in final_indices)
            scores = data["rm_scores"].sum(dim=1).tolist()
            sample_scores.extend(scores)
            num_turns = data.pop("num_turns", None)
            if num_turns is None:
                # Online SUMO validation is a single-turn trajectory and does
                # not route through AgentLoop's optional `num_turns` field.
                _emit_online_diag(
                    "[ONLINE_VAL_NUM_TURNS_FALLBACK] num_turns missing; "
                    f"using one turn for {len(final_indices)} validation rows"
                )
                sample_turns.extend([1] * len(final_indices))
            else:
                sample_turns.extend(num_turns.tolist())
            reward_extra_infos_dict["reward"].extend(scores)

            extra_fields_list = data.pop("extra_fields", None)
            if extra_fields_list is not None:
                n_prior = len(reward_extra_infos_dict["reward"]) - len(extra_fields_list.tolist())
                for extra_field in extra_fields_list.tolist():
                    reward_extra_info = (
                        extra_field.get("reward_extra_info", {}) if isinstance(extra_field, dict) else {}
                    )
                    for key in reward_extra_infos_dict:
                        if key != "reward" and key not in reward_extra_info:
                            reward_extra_infos_dict[key].append(None)
                    for key, value in reward_extra_info.items():
                        if key not in reward_extra_infos_dict:
                            reward_extra_infos_dict[key] = [None] * n_prior
                        reward_extra_infos_dict[key].append(value)
                    n_prior += 1

            reward_model = data.pop("reward_model", None)
            if reward_model is not None:
                sample_gts.extend([item.get("ground_truth", None) for item in reward_model.tolist()])
            else:
                sample_gts.extend([None] * len(final_indices))

            data_source = data.pop("data_source", None)
            if data_source is not None:
                data_sources.extend(data_source.tolist())
            else:
                data_sources.extend(["unknown"] * len(final_indices))

            dump_all_inputs.extend(all_inputs)
            dump_all_outputs.extend(all_outputs)
            dump_all_keys.extend(batch.keys)

            # 5. cleanup transfer queue
            tq.kv_clear(keys=batch.keys, partition_id=batch.partition_id)

        # logger to wandb
        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump to local dir
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            # Sort according to uid (so that generations in the same rollout are together)
            sort_keys = []
            for key in dump_all_keys:
                sort_keys.append(_stable_generation_sort_key(key))
            sorted_indices = sorted(range(len(dump_all_keys)), key=lambda i: sort_keys[i])
            dump_all_inputs = [dump_all_inputs[i] for i in sorted_indices]
            dump_all_outputs = [dump_all_outputs[i] for i in sorted_indices]
            dump_all_keys = [dump_all_keys[i] for i in sorted_indices]

            # For ground truths, scores and reward extra infos, find the values in the
            # lists for the final samples of each session
            dump_all_sessions = [
                f"{parts[0]}_{parts[1]}" if len(parts) == 3 else key
                for key in dump_all_keys
                for parts in [key.rsplit("_", 2)]
            ]
            session_final_indices = [session_to_sample_idx[session] for session in dump_all_sessions]
            self._dump_generations(
                inputs=dump_all_inputs,
                outputs=dump_all_outputs,
                gts=[sample_gts[i] for i in session_final_indices],
                scores=[sample_scores[i] for i in session_final_indices],
                reward_extra_infos_dict={
                    k: [v[i] for i in session_final_indices] for k, v in reward_extra_infos_dict.items()
                }
                | {"uid": dump_all_keys},
                dump_path=val_data_dir,
            )

        return self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""
        generations_to_log = self.config.trainer.log_val_generations
        if generations_to_log == 0:
            return

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    @staticmethod
    def _write_generations(inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path, global_steps):
        """Write generation samples as JSONL (runs in background thread)."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        def json_encode_default(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif hasattr(obj, "tolist"):
                return obj.tolist()
            raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

        with open(filename, "w") as f:
            for i in range(n):
                entry = {k: v[i] for k, v in base_data.items()}
                f.write(json.dumps(entry, ensure_ascii=False, default=json_encode_default) + "\n")

        print(f"Dumped generations to {filename}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL asynchronously."""
        global_steps = self.global_steps
        future = self._dump_executor.submit(
            self._write_generations,
            inputs,
            outputs,
            gts,
            scores,
            reward_extra_infos_dict,
            dump_path,
            global_steps,
        )
        self._dump_futures.append(future)
        # Clean up completed futures and surface any exceptions early
        still_pending = []
        for f in self._dump_futures:
            if f.done():
                f.result()  # re-raises if the write failed
            else:
                still_pending.append(f)
        self._dump_futures = still_pending

    def _init_dump_executor(self):
        """Create or recreate the dump executor and futures list."""
        self._dump_executor = ThreadPoolExecutor(max_workers=1)
        self._dump_futures = []

    def _shutdown_dump_executor(self):
        """Drain pending dump futures and shut down the executor."""
        for f in self._dump_futures:
            f.result()
        self._dump_futures.clear()
        self._dump_executor.shutdown(wait=True)

    def _log_rollout_data(self, batch: KVBatchMeta, timing_raw: dict, rollout_data_dir: str):
        """Fetch rollout data from TransferQueue and dump sorted by uid."""
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            fields = ["uid", "prompts", "responses", "rm_scores", "reward_model"]
            data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=fields)
            data["prompts"] = data["prompts"].to_padded_tensor(padding=self.tokenizer.pad_token_id)
            data["responses"] = data["responses"].to_padded_tensor(padding=self.tokenizer.pad_token_id)

            uids = data.pop("uid").tolist()
            inputs = [_decode_generation_value(self.tokenizer, ids) for ids in data["prompts"]]
            outputs = [_decode_generation_value(self.tokenizer, ids) for ids in data["responses"]]
            scores = data["rm_scores"].sum(dim=1).tolist()

            reward_model = data.pop("reward_model", None)
            if reward_model is not None:
                gts = [item.get("ground_truth", None) for item in reward_model.tolist()]
            else:
                gts = [None] * len(uids)

            # Sort by uid key ({sample}_{rollout}_{output})
            sort_keys = []
            for key in batch.keys:
                sort_keys.append(_stable_generation_sort_key(key))
            sorted_indices = sorted(range(len(sort_keys)), key=lambda i: sort_keys[i])

            inputs = [inputs[i] for i in sorted_indices]
            outputs = [outputs[i] for i in sorted_indices]
            gts = [gts[i] for i in sorted_indices]
            scores = [scores[i] for i in sorted_indices]

            reward_extra_infos_dict = {"uid": [batch.keys[i] for i in sorted_indices]}

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=rollout_data_dir,
            )

    def _val_metrics_update(self, data_sources, sample_uids, reward_extra_infos_dict, sample_turns) -> dict[str, float]:
        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.array(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def _start_profiling(self) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        do_profile = (
            not self.prev_step_profile and self.curr_step_profile
            if self.config.global_profiler.profile_continuous_steps
            else self.curr_step_profile
        )

        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        self.next_step_profile = (
            self.global_steps + 1 in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        do_profile = (
            self.curr_step_profile and not self.next_step_profile
            if self.config.global_profiler.profile_continuous_steps
            else self.curr_step_profile
        )
        self.prev_step_profile = self.curr_step_profile
        self.curr_step_profile = self.next_step_profile

        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()

    def _fetch_one_gen_batch(self) -> TensorDict:
        """Fetch one ``gen_batch_size`` chunk from the dataloader."""
        try:
            if self.train_dataloader_it is None:
                self.train_dataloader_it = iter(self.train_dataloader)
            batch_dict = next(self.train_dataloader_it)
        except StopIteration:
            self.train_dataloader_it = iter(self.train_dataloader)
            batch_dict = next(self.train_dataloader_it)

        batch_dict["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch_dict["raw_prompt"]))], dtype=object)
        return tu.get_tensordict(batch_dict)

    def _next_train_batch(self, num_prompts: int | None = None) -> TensorDict:
        """Fetch and coalesce the requested number of prompts."""
        train_batch_size = self.config.data.train_batch_size
        if num_prompts is None:
            num_prompts = train_batch_size
        gen_batch_size = self.config.data.get("gen_batch_size", None) or train_batch_size
        if num_prompts <= 0 or num_prompts % gen_batch_size != 0:
            raise ValueError(
                f"num_prompts ({num_prompts}) must be a positive multiple of gen_batch_size "
                f"({gen_batch_size}); it is submitted in whole gen_batch_size dataloader fetches."
            )

        chunks = [self._fetch_one_gen_batch() for _ in range(num_prompts // gen_batch_size)]
        batch = chunks[0] if len(chunks) == 1 else tu.concat_tensordict(chunks)
        tu.assign_non_tensor_data(batch, "global_steps", self.global_steps)
        return batch

    def _submit_batch_to_rollout(self, batch: TensorDict) -> int:
        """Register prompts in TransferQueue and dispatch them for generation."""
        tags = [{"is_prompt": True, "status": "pending", "global_steps": self.global_steps} for _ in range(len(batch))]
        if self.trainer_mode != "sync":
            tq.kv_batch_put(
                keys=list(batch["uid"]),
                partition_id="train",
                tags=tags,
                # Persist prompt data for async checkpoint recovery.
                # TODO: maybe let workers do it?
                fields=batch.select(*[key for key in batch.keys() if not isinstance(batch.get(key), NonTensorData)]),
            )
        else:
            tq.kv_batch_put(keys=list(batch["uid"]), partition_id="train", tags=tags)

        self.agent_loop_manager.generate_sequences(batch)
        return len(batch)

    def _add_prompts_to_generate(self, num_prompts: int) -> int:
        """Add an exact number of prompts to the AgentLoopManager."""
        batch = self._next_train_batch(num_prompts)
        return self._submit_batch_to_rollout(batch)

    @SkipManager.annotate_tq(role="rollout_tq", phase="submit")
    def _add_batch_to_generate(self):
        """Add one training batch to the AgentLoopManager."""
        batch = self._next_train_batch()
        self._submit_batch_to_rollout(batch)

    def _compute_reward_colocate(self, batch: KVBatchMeta, metrics: dict | None = None) -> KVBatchMeta:
        """Compute the reward score with a colocated reward model."""
        assert self.reward_loop_manager is not None, "RewardLoopManager is None"

        # 1. read the fields required by the reward model from TransferQueue.
        fields = ["prompts", "responses", "raw_prompt"]
        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=fields)

        prompt_lengths = data["prompts"].offsets().diff()
        response_lengths = data["responses"].offsets().diff()
        prompts = data["prompts"].to_padded_tensor(padding=self.tokenizer.pad_token_id)
        responses = data["responses"].to_padded_tensor(padding=self.tokenizer.pad_token_id)

        # 2. rebuild the attention mask aligned with the [prompts | responses] layout.
        prompt_mask = self._lengths_to_mask(prompt_lengths, prompts.size(1))
        response_mask = self._lengths_to_mask(response_lengths, responses.size(1))
        attention_mask = torch.cat([prompt_mask, response_mask], dim=1)

        # `raw_prompt` is a non-tensor field; depending on the TransferQueue backend it
        # comes back as a tensordict LinkedList (a `list` subclass), a NonTensorStack or a
        # numpy array. `list(...)` normalizes all of them to a plain list where each element
        # is one sample's chat-message list (whereas `.tolist()` only exists on numpy/tensors).
        raw_prompts = list(data["raw_prompt"])
        raw_prompt_arr = np.empty(len(raw_prompts), dtype=object)
        raw_prompt_arr[:] = raw_prompts

        rm_input = DataProto(
            batch=TensorDict(
                {"prompts": prompts, "responses": responses, "attention_mask": attention_mask},
                batch_size=len(batch),
            ),
            non_tensor_batch={"raw_prompt": raw_prompt_arr},
        )

        # 3. run the reward model (wakes/sleeps the reward model internally).
        rm_output = self.reward_loop_manager.compute_rm_score(rm_input)

        # 4. write rm_scores (and reward extra info) back to TransferQueue.
        padded_rm_scores = rm_output.batch["rm_scores"]
        rm_scores = torch.nested.as_nested_tensor(
            [padded_rm_scores[i, : response_lengths[i]] for i in range(len(batch))],
            layout=torch.jagged,
        )
        write_back = {"rm_scores": rm_scores}
        for key in rm_output.meta_info.get("reward_extra_keys", []):
            write_back[key] = rm_output.non_tensor_batch[key]
        tq.kv_batch_put(
            keys=batch.keys,
            partition_id=batch.partition_id,
            fields=tu.get_tensordict(write_back),
        )

        return batch

    @staticmethod
    def _lengths_to_mask(lengths: torch.Tensor, width: int) -> torch.Tensor:
        """Build a right-padded mask of shape (len(lengths), width) from per-row valid lengths."""
        positions = torch.arange(width, device=lengths.device).unsqueeze(0)
        return (positions < lengths.unsqueeze(1)).to(torch.int64)

    def _get_required_batch_multiple(self, dp_size: int) -> int:
        """Return the global batch multiple required by downstream train steps(e.g. critics, actors)."""
        required_multiple = dp_size

        # If enabled with critic training, the batch should align with critic PPO mini-batches.
        if self.use_critic:
            critic_global_mini_batch_size = self.config.critic.ppo_mini_batch_size
            critic_global_mini_batch_size *= self.config.actor_rollout_ref.rollout.n
            required_multiple = math.lcm(required_multiple, critic_global_mini_batch_size)

        # If there is an actor update, the batch should align with actor PPO mini-batches too.
        if self.config.trainer.critic_warmup <= self.global_steps:
            actor_global_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
            actor_global_mini_batch_size *= self.config.actor_rollout_ref.rollout.n
            required_multiple = math.lcm(required_multiple, actor_global_mini_batch_size)

        # Notice lcm(a, b, c) == lcm(lcm(a, b), c), so it is optimal.
        return required_multiple

    def _balance_batch(self, batch: KVBatchMeta, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        """Reorder the data on single controller such that each dp rank gets similar total tokens."""
        # get actor dp size
        role, worker_group = "actor", self.actor_rollout_wg
        if role not in worker_group._dispatch_info:
            dp_rank_mapping = worker_group._query_dispatch_info(role)
            worker_group._dispatch_info[role] = dp_rank_mapping
        else:
            dp_rank_mapping = worker_group._dispatch_info[role]
        dp_size = max(dp_rank_mapping) + 1

        # Upsampling the batch with padding sequences
        batch_multiple = self._get_required_batch_multiple(dp_size)
        batch = upsample_batch_to_divisible_size(batch, batch_multiple, self.tokenizer.eos_token_id)
        global_seqlen_lst = torch.tensor([tag["seq_len"] for tag in batch.tags], dtype=torch.int64)
        workload_lst = calculate_workload(global_seqlen_lst)

        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_partition_lst = get_seqlen_balanced_partitions(workload_lst, k_partitions=dp_size, equal_size=True)
        batch.reorder([j for partition in global_partition_lst for j in partition])
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst.tolist(), partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)
        return batch

    def _compute_old_log_prob(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
        """Compute the old log prob of the batch."""
        # Operating Mode Selection:
        # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
        # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
        #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
        rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
        bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
        if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
            data = tq.kv_batch_get(
                keys=batch.keys, partition_id=batch.partition_id, select_fields=["rollout_log_probs"]
            )
            data["old_log_probs"] = data.pop("rollout_log_probs")
            tq.kv_batch_put(keys=batch.keys, partition_id=batch.partition_id, fields=data)
            return batch

        # 1. compute log probs
        batch.extra_info.update(
            {
                "calculate_entropy": True,
                "compute_loss": False,
                "temperature": self.config.actor_rollout_ref.rollout.temperature,
            }
        )
        output: KVBatchMeta = self.actor_rollout_wg.compute_log_prob(batch)
        assert len(output) == len(batch)

        fields = ["entropy", "log_probs", "response_mask"]
        if self.config.actor_rollout_ref.rollout.calculate_log_probs:
            fields.extend(["responses", "rollout_log_probs"])
        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=fields)

        # 2. write old_log_probs and entropy back to TransferQueue
        data["old_log_probs"] = response_from_nested(data.pop("log_probs"), data["response_mask"])
        data["entropy"] = response_from_nested(data.pop("entropy"), data["response_mask"])
        batch = tq.kv_batch_put(
            keys=batch.keys, partition_id=batch.partition_id, fields=data.select("old_log_probs", "entropy")
        )

        data = DataProto(batch=data.to_padded_tensor())

        # 3. calculate actor entroy metrics
        actor_config = self.config.actor_rollout_ref.actor
        entropy_agg = agg_loss(
            loss_mat=data.batch["entropy"],
            loss_mask=data.batch["response_mask"],
            loss_agg_mode=actor_config.loss_agg_mode,
            loss_scale_factor=actor_config.loss_scale_factor,
        )
        old_log_prob_metrics = {
            "actor/entropy": entropy_agg.detach().item(),
            # "perf/mfu/actor_infer": old_log_prob_mfu,
        }
        metrics.update(old_log_prob_metrics)

        # 4. calculate rollout vs actor logprobs diff
        if self.config.actor_rollout_ref.rollout.calculate_log_probs:
            metrics.update(calculate_debug_metrics(data))

        return batch

    def _compute_ref_log_prob(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
        """Compute the reference log prob of the batch."""
        # 1. compute log probs
        metadata = {
            "calculate_entropy": False,
            "compute_loss": False,
            "temperature": self.config.actor_rollout_ref.rollout.temperature,
        }
        if self.ref_in_actor:
            metadata["no_lora_adapter"] = True
        batch.extra_info.update(metadata)
        if self.ref_in_actor:
            output = self.actor_rollout_wg.compute_log_prob(batch)
        else:
            output = self.ref_policy_wg.compute_ref_log_prob(batch)
        assert len(output) == len(batch)

        # 2. write ref_log_prob and entropy back to TransferQueue
        data = tq.kv_batch_get(
            keys=batch.keys, partition_id=batch.partition_id, select_fields=["log_probs", "response_mask"]
        )
        data["ref_log_prob"] = response_from_nested(data.pop("log_probs"), data["response_mask"])
        tq.kv_batch_put(keys=batch.keys, partition_id=batch.partition_id, fields=data.select("ref_log_prob"))

        return batch

    def _compute_values(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
        """Compute the values of the batch."""
        # 1. compute value
        batch.extra_info.update(
            {
                "compute_loss": False,
                "temperature": self.config.actor_rollout_ref.rollout.temperature,
            }
        )
        output = self.critic_wg.infer_batch(batch)
        # TODO: DataProtoFuture support KVBatchMeta
        ray.get(output.futures)

        # 2. write value back to TransferQueue
        data = tq.kv_batch_get(
            keys=batch.keys, partition_id=batch.partition_id, select_fields=["values", "response_mask"]
        )
        data["values"] = response_from_nested(data.pop("values"), data["response_mask"])
        tq.kv_batch_put(keys=batch.keys, partition_id=batch.partition_id, fields=data.select("values"))

        return batch

    def _compute_advantage(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
        """Compute the advantage of the batch."""
        def _as_rows(value):
            """Normalize tensor/numpy/non-tensor stack rows from TransferQueue."""
            if hasattr(value, "tolist"):
                return value.tolist()
            return list(value)

        mode_selector_coef = float(os.environ.get("V35_MODE_SELECTOR_COEF", "0"))
        fields = [
            "uid",
            "response_mask",
            "rm_scores",
            "rollout_log_probs",
            "old_log_probs",
            "ref_log_prob",
            "values",
            "grpo_advantage_scale",
        ]
        if mode_selector_coef > 0.0 or self.config.algorithm.adv_estimator in ("gdpo", "grpo"):
            fields.extend([
                "extra_fields", "mode_aux_mask", "reasoning_aux_mask",
                "signal_aux_mask", "grpo_loss_mask", "responses",
            ])
            # Binary mode metadata is emitted by the online collector.  Do
            # not request these optional fields for legacy/offline partitions;
            # older TransferQueue schemas legitimately do not contain them and
            # should continue through the token-CE compatibility path.
            online_enabled = bool(self.config.get("online_cooperative", {}).get("enabled", False))
            if mode_selector_coef > 0.0 and online_enabled:
                fields.extend([
                    "mode_binary_mask", "mode_token_index", "mode_fast_token_id",
                    "mode_slow_token_id", "mode_binary_target", "mode_binary_valid",
                    "mode_binary_weight",
                ])
        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=fields)

        response_mask = data["response_mask"]
        # ``uid`` is still a tensor at this point.  Read it before rebuilding
        # the padded DataProto; online metadata below may use it as fallback.
        uid_values = _as_rows(data["uid"])
        raw_extra_fields = data.get("extra_fields") if "extra_fields" in data else None
        extra_field_rows = 0
        reward_payload_rows = 0
        if raw_extra_fields is not None:
            raw_extra_values = _as_rows(raw_extra_fields)
            extra_field_rows = len(raw_extra_values)
            for raw_extra in raw_extra_values:
                if hasattr(raw_extra, "data"):
                    raw_extra = raw_extra.data
                if isinstance(raw_extra, dict) and isinstance(raw_extra.get("reward_extra_info"), dict):
                    reward_payload_rows += 1
        online_metadata = {}
        reward_infos = None
        preserved_non_tensor = {}
        if "extra_fields" in data:
            reward_infos = []
            episode_groups = []
            mode_groups = []
            for extra_fields in _as_rows(data.pop("extra_fields")):
                if hasattr(extra_fields, "data"):
                    extra_fields = extra_fields.data
                extra_fields = extra_fields if isinstance(extra_fields, dict) else {}
                reward_info = dict(extra_fields.get("reward_extra_info") or {})
                reward_info.setdefault("online", bool(extra_fields.get("online", False)))
                reward_info.setdefault("forced_mode", extra_fields.get("forced_mode"))
                reward_info.setdefault("generated_mode", extra_fields.get("generated_mode"))
                reward_infos.append(reward_info)
                episode_groups.append(extra_fields.get("episode_group_id"))
                mode_groups.append(extra_fields.get("mode_group_id"))
                for key in (
                    "decision_group_id", "local_score", "network_reward",
                    "reasoning_cost_reward", "format_penalty", "signal_valid",
                    "format_valid",
                ):
                    online_metadata.setdefault(key, []).append(
                        reward_info.get(key, extra_fields.get(key))
                    )
            if any(value is not None for value in episode_groups):
                preserved_non_tensor["episode_group_id"] = np.asarray(
                    [value if value is not None else uid for value, uid in zip(episode_groups, uid_values)],
                    dtype=object,
                )
            if any(value is not None for value in mode_groups):
                preserved_non_tensor["mode_group_id"] = np.asarray(
                    [value if value is not None else uid for value, uid in zip(mode_groups, uid_values)],
                    dtype=object,
                )
        # Do not discard online grouping and reward dimensions here.  The old
        # reconstruction silently made decision GDPO unreachable.
        for key, values in online_metadata.items():
            if len(values) == len(uid_values):
                fallback = "" if key == "decision_group_id" else 0.0
                preserved_non_tensor[key] = np.asarray(
                    [fallback if value is None else value for value in values], dtype=object
                )
        data = DataProto(batch=data.to_padded_tensor(), non_tensor_batch=preserved_non_tensor)
        data.batch["token_level_scores"] = data.batch["rm_scores"]
        data.batch.pop("uid")
        data.non_tensor_batch["uid"] = np.array(uid_values, dtype=object)

        decision_group_values = data.non_tensor_batch.get("decision_group_id")
        mode_group_values = data.non_tensor_batch.get("mode_group_id")
        if decision_group_values is None:
            decision_group_values = ()
        if mode_group_values is None:
            mode_group_values = ()
        decision_group_count = len({str(value) for value in decision_group_values if str(value)})
        mode_group_count = len({str(value) for value in mode_group_values if str(value)})
        reward_keys = ("local_score", "network_reward", "reasoning_cost_reward", "format_penalty")
        complete_reward_rows = 0
        nonfinite_reward_values = 0
        if reward_infos is not None:
            for reward_info in reward_infos:
                if all(key in reward_info for key in reward_keys):
                    complete_reward_rows += 1
                for key in reward_keys:
                    try:
                        if not math.isfinite(float(reward_info.get(key, 0.0))):
                            nonfinite_reward_values += 1
                    except (TypeError, ValueError):
                        nonfinite_reward_values += 1
        _emit_online_diag(
            "[ONLINE_CREDIT_INPUT] "
            f"rows={len(uid_values)} extra_fields_rows={extra_field_rows} "
            f"reward_payload_rows={reward_payload_rows} reward_complete_rows={complete_reward_rows} "
            f"decision_groups={decision_group_count} mode_groups={mode_group_count} "
            f"nonfinite_reward_values={nonfinite_reward_values}"
        )

        if mode_selector_coef > 0.0:
            from v35_offline_grpo.perception_sft import build_mode_selector_weights

            # ``rollout.n`` deliberately remains one in the online recipe:
            # it describes the single Stage-1 generation.  Mode GDPO instead
            # compares the inner balanced Stage-2 candidates.
            mode_rollout_n = self.config.actor_rollout_ref.rollout.n
            if reward_infos and any(isinstance(info, dict) and info.get("online") for info in reward_infos):
                mode_rollout_n = int(os.environ.get("V35_ONLINE_NUM_ROLLOUTS", "6"))

            _emit_online_diag(
                "[MODE_GDPO_PATH] "
                f"status=enter rows={len(uid_values)} "
                f"reward_infos={'present' if reward_infos is not None else 'missing'} "
                f"mode_groups={mode_group_count} rollout_n={mode_rollout_n}"
            )

            response_rows = data.batch["responses"]
            mode_masks = data.batch["mode_aux_mask"].to(bool)
            mode_token_counts = mode_masks.sum(dim=-1).tolist()
            mode_prefixes = []
            for response, mask in zip(response_rows, mode_masks, strict=True):
                positions = torch.nonzero(mask, as_tuple=False).flatten()
                mode_start = int(positions[0].item()) if positions.numel() else int(response.numel())
                mode_prefixes.append(tuple(int(value) for value in response[:mode_start].tolist()))
            mode_group_values = data.non_tensor_batch.get("mode_group_id")
            if mode_group_values is None:
                mode_group_values = np.asarray(uid_values, dtype=object)
            mode_weights, mode_metrics, active_rows, mode_group_details = build_mode_selector_weights(
                [str(uid) for uid in mode_group_values],
                reward_infos,
                rollout_n=mode_rollout_n,
                temperature=float(os.environ.get("V35_MODE_SELECTOR_TEMPERATURE", "0.2")),
                min_probability=float(os.environ.get("V35_MODE_SELECTOR_MIN_PROB", "0.1")),
                mode_token_counts=mode_token_counts,
                mode_prefixes=mode_prefixes,
            )
            active_tensor = torch.tensor(
                active_rows, dtype=torch.bool, device=data.batch["mode_aux_mask"].device
            ).unsqueeze(-1)
            data.batch["mode_aux_mask"] = data.batch["mode_aux_mask"] * active_tensor
            row_weights = torch.tensor(
                mode_weights,
                dtype=torch.float32,
                device=data.batch["mode_aux_mask"].device,
            ).unsqueeze(-1)
            data.batch["mode_aux_weight"] = data.batch["mode_aux_mask"].to(torch.float32) * row_weights
            # The binary classifier must consume exactly the same active rows
            # and GDPO weights as the mode token branch.  Keep the p(slow)
            # target soft and group-constant, while disabling rows that have
            # no shared prefix/evidence or no valid one-token mode span.
            if "mode_binary_valid" in data.batch:
                binary_valid = data.batch["mode_binary_valid"].to(torch.bool)
                active_rows_tensor = active_tensor.squeeze(-1)
                data.batch["mode_binary_valid"] = binary_valid & active_rows_tensor
                if "mode_binary_mask" in data.batch:
                    data.batch["mode_binary_mask"] = (
                        data.batch["mode_binary_mask"]
                        * active_rows_tensor.unsqueeze(-1).to(data.batch["mode_binary_mask"].dtype)
                    )
                data.batch["mode_binary_weight"] = (
                    data.batch["mode_binary_valid"].to(torch.float32)
                    * row_weights.squeeze(-1)
                )
                if "mode_binary_target" in data.batch:
                    targets = torch.tensor(
                        [
                            float(mode_group_details.get(str(group), {}).get("p_target_slow", 0.5))
                            for group in mode_group_values
                        ],
                        dtype=torch.float32,
                        device=data.batch["mode_aux_mask"].device,
                    )
                    data.batch["mode_binary_target"] = targets.clamp(0.0, 1.0)
            metrics.update({f"aux/{key}": value for key, value in mode_metrics.items()})
            # One concise line per update makes the router's credit assignment
            # auditable without emitting every one of the 672 rollout rows.
            # Detail values are computed before masks are zeroed, so malformed
            # rows and missing mode tokens cannot disappear from the diagnosis.
            detail_values = list(mode_group_details.values())
            logger.info(
                "[MODE_GDPO_DIAG] groups=%d active_rows=%d/%d fast_rows=%d slow_rows=%d "
                "malformed_ratio=%.4f zero_mode_token_ratio=%.4f same_prefix_ratio=%.4f "
                "target_slow=%.4f fast_utility=%.5g slow_utility=%.5g "
                "utility_range=[%.5g,%.5g]",
                len(mode_group_details), int(sum(active_rows)), len(active_rows),
                int(mode_metrics["mode_fast_row_count"]), int(mode_metrics["mode_slow_row_count"]),
                mode_metrics["mode_malformed_row_ratio"], mode_metrics["mode_zero_token_row_ratio"],
                mode_metrics["mode_complete_prefix_group_ratio"],
                mode_metrics["mode_target_slow_probability"],
                mode_metrics["fast_mean_traffic_utility"], mode_metrics["slow_mean_traffic_utility"],
                min((float(item["fast_utility"]) for item in detail_values), default=0.0),
                max((float(item["slow_utility"]) for item in detail_values), default=0.0),
            )
            _emit_online_diag(
                "[MODE_GDPO_RESULT] "
                f"groups={len(mode_group_details)} active_rows={sum(active_rows)}/{len(active_rows)} "
                f"fast_rows={int(mode_metrics['mode_fast_row_count'])} "
                f"slow_rows={int(mode_metrics['mode_slow_row_count'])} "
                f"malformed_ratio={float(mode_metrics['mode_malformed_row_ratio']):.5g} "
                f"target_slow={float(mode_metrics['mode_target_slow_probability']):.5g}"
            )
        else:
            _emit_online_diag(
                "[MODE_GDPO_PATH] "
                f"status=disabled coef={mode_selector_coef:.5g} rows={len(uid_values)}"
            )

        # 1. apply kl penalty to rewards
        if self.config.algorithm.use_kl_in_reward:
            data, kl_metrics = apply_kl_penalty(
                data, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
            )
            metrics.update(kl_metrics)
        else:
            data.batch["token_level_rewards"] = data.batch["token_level_scores"]

        # 2. Compute rollout correction: IS weights, rejection sampling, and metrics
        # Only runs in decoupled mode (computes once per batch using stable π_old)
        # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
        rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
        bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
        rollout_correction = (
            rollout_corr_config is not None and "rollout_log_probs" in data.batch and not bypass_recomputing_logprobs
        )
        if rollout_correction:
            data, is_metrics = compute_rollout_correction_and_add_to_batch(data, rollout_corr_config)
            metrics.update(is_metrics)

        # 3. compute advantages
        data = compute_advantage_for_multi_trajectories(
            data,
            batch_keys=batch.keys,
            adv_estimator=self.config.algorithm.adv_estimator,
            gamma=self.config.algorithm.gamma,
            lam=self.config.algorithm.lam,
            # Online collection expands each bootstrap stream into the inner
            # Stage-2 candidate suite.  ``rollout.n`` stays 1 because V1 has
            # one outer Stage-1 pass; use the suite size for GRPO grouping.
            num_repeat=int(os.environ.get("V35_ONLINE_NUM_ROLLOUTS", self.config.actor_rollout_ref.rollout.n)),
            norm_adv_by_std_in_grpo=self.config.algorithm.get("norm_adv_by_std_in_grpo", True),
            config=self.config.algorithm,
        )

        # Online cooperative decision credit: normalize each reward dimension
        # within the same snapshot/intersection group, then apply the result
        # only to reasoning+signal tokens.  The existing episode-level GRPO
        # remains the source of network reward; this replaces its duplicated
        # per-intersection broadcast for the decision suffix only.
        if "decision_group_id" in data.non_tensor_batch and "grpo_loss_mask" in data.batch:
            _emit_online_diag(
                "[DECISION_GDPO_PATH] "
                f"status=enter rows={len(uid_values)} groups={decision_group_count}"
            )
            groups = np.asarray(data.non_tensor_batch["decision_group_id"], dtype=object)
            common_keys = ("local_score", "network_reward", "format_penalty")
            common_weights = (
                float(os.environ.get("V35_DECISION_LOCAL_SCORE_WEIGHT", "1.0")),
                float(os.environ.get("V35_DECISION_NETWORK_REWARD_WEIGHT", "0.5")),
                float(os.environ.get("V35_DECISION_FORMAT_PENALTY_WEIGHT", "1.0")),
            )
            reasoning_cost_weight = float(os.environ.get("V35_DECISION_REASONING_COST_REWARD_WEIGHT", "0.5"))
            reasoning_adv = torch.zeros_like(data.batch["advantages"])
            signal_adv = torch.zeros_like(data.batch["advantages"])
            group_map = {}
            for i, group in enumerate(groups.tolist()):
                group_map.setdefault(str(group), []).append(i)
            for indices in group_map.values():
                if len(indices) < 2:
                    continue
                combined = torch.zeros(len(indices), device=data.batch["advantages"].device)
                for key, weight in zip(common_keys, common_weights, strict=True):
                    vals = torch.tensor(
                        [float(data.non_tensor_batch.get(key, np.zeros(len(groups), dtype=np.float32))[i]) for i in indices],
                        dtype=torch.float32, device=combined.device,
                    )
                    if not torch.isfinite(vals).all():
                        vals = torch.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
                    std = vals.std(unbiased=False)
                    if float(std) > 1e-6:
                        combined += weight * (vals - vals.mean()) / (std + 1e-6)
                costs = torch.tensor(
                    [float(data.non_tensor_batch.get("reasoning_cost_reward", np.zeros(len(groups), dtype=np.float32))[i]) for i in indices],
                    dtype=torch.float32, device=combined.device,
                )
                costs = torch.nan_to_num(costs, nan=0.0, posinf=0.0, neginf=0.0)
                cost_std = costs.std(unbiased=False)
                reasoning_combined = combined.clone()
                if float(cost_std) > 1e-6:
                    reasoning_combined += reasoning_cost_weight * (costs - costs.mean()) / (cost_std + 1e-6)
                for row, signal_value, reasoning_value in zip(indices, combined, reasoning_combined, strict=True):
                    # Previous signal credit omitted reasoning-cost reward; retained for reference.
                    # signal_adv[row] = signal_value
                    signal_adv[row] = reasoning_value
                    reasoning_adv[row] = reasoning_value
            decision_mask = data.batch["grpo_loss_mask"].to(torch.bool)
            reasoning_mask = data.batch.get("reasoning_aux_mask", torch.zeros_like(decision_mask)).to(torch.bool)
            signal_mask = data.batch.get("signal_aux_mask", torch.zeros_like(decision_mask)).to(torch.bool)
            if torch.any(reasoning_mask & signal_mask):
                raise ValueError("reasoning/signal loss masks overlap")
            segmented_adv = torch.where(reasoning_mask, reasoning_adv, torch.where(signal_mask, signal_adv, torch.zeros_like(signal_adv)))
            data.batch["advantages"] = torch.where(decision_mask, segmented_adv, data.batch["advantages"])
            data.batch["returns"] = torch.where(decision_mask, segmented_adv, data.batch["returns"])
            data.batch["reasoning_aux_advantage"] = reasoning_adv
            data.batch["signal_aux_advantage"] = signal_adv
            # Batch-level audit trail for diagnosing credit assignment without
            # logging every token.  Keep raw reward statistics alongside the
            # resulting advantage and mask counts so propagation boundaries
            # can be checked from a single training log line.
            reward_diag = []
            for key in (*common_keys, "reasoning_cost_reward"):
                # Archived/partial rows may not carry every online reward
                # dimension.  The advantage path already treats those as
                # zero; diagnostics must follow the same contract instead of
                # raising a secondary KeyError and hiding the real update.
                raw = np.asarray(
                    data.non_tensor_batch.get(
                        key, np.zeros(len(groups), dtype=np.float32)
                    ),
                    dtype=np.float32,
                )
                finite = np.isfinite(raw)
                clean = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
                reward_diag.append(
                    f"{key}=mean:{clean.mean():.5g},std:{clean.std():.5g},"
                    f"min:{clean.min():.5g},max:{clean.max():.5g},nonfinite:{int((~finite).sum())}"
                )
            singleton_groups = sum(len(items) < 2 for items in group_map.values())
            valid_flags = np.asarray(
                data.non_tensor_batch.get("signal_valid", np.ones(len(groups), dtype=bool)),
                dtype=bool,
            )
            logger.info(
                "[DECISION_GDPO_DIAG] groups=%d singleton_groups=%d rows=%d "
                "decision_tokens=%d mode_tokens=%d valid_rows=%d invalid_rows=%d "
                "adv_mean=%.5g adv_std=%.5g adv_absmax=%.5g finite=%s rewards={%s}",
                len(group_map), singleton_groups, len(groups), int(decision_mask.sum().item()),
                int(data.batch.get("mode_aux_mask", torch.zeros_like(decision_mask)).sum().item()),
                int(valid_flags.sum()), int(len(groups) - valid_flags.sum()),
                float(segmented_adv.mean().item()), float(segmented_adv.std(unbiased=False).item()),
                float(segmented_adv.abs().max().item()), bool(torch.isfinite(segmented_adv).all().item()),
                ";".join(reward_diag),
            )
            _emit_online_diag(
                "[DECISION_GDPO_RESULT] "
                f"groups={len(group_map)} singleton_groups={singleton_groups} rows={len(groups)} "
                f"decision_tokens={int(decision_mask.sum().item())} "
                f"reasoning_tokens={int(reasoning_mask.sum().item())} "
                f"signal_tokens={int(signal_mask.sum().item())} "
                f"adv_mean={float(segmented_adv.mean().item()):.5g} "
                f"adv_std={float(segmented_adv.std(unbiased=False).item()):.5g} "
                f"adv_nonzero={int((segmented_adv.abs() > 1e-8).sum().item())} "
                f"finite={bool(torch.isfinite(segmented_adv).all().item())}"
            )
        else:
            missing = []
            if "decision_group_id" not in data.non_tensor_batch:
                missing.append("decision_group_id")
            if "grpo_loss_mask" not in data.batch:
                missing.append("grpo_loss_mask")
            _emit_online_diag(
                "[DECISION_GDPO_PATH] "
                f"status=skip rows={len(uid_values)} missing={','.join(missing) or 'unknown'}"
            )

        # 4. write nested advantages and returns back to TransferQueue
        fields = ["advantages", "returns", "reasoning_aux_advantage", "signal_aux_advantage"]
        if mode_selector_coef > 0.0:
            fields.extend(["mode_aux_mask", "mode_aux_weight"])
            if "mode_binary_mask" in data.batch:
                fields.extend(["mode_binary_mask", "mode_binary_valid", "mode_binary_target", "mode_binary_weight"])
        if self.config.algorithm.use_kl_in_reward:
            fields.append("token_level_rewards")
        if rollout_correction:
            fields.append("response_mask")
            if "rollout_is_weights" in data.batch:
                fields.append("rollout_is_weights")

        output = {}
        for field in fields:
            if field in {"mode_binary_valid", "mode_binary_target", "mode_binary_weight"}:
                # These are row-level classifier metadata, not response-token
                # tensors; keep their leading batch dimension dense.
                output[field] = data.batch[field]
            else:
                output[field] = response_to_nested(data.batch[field], response_mask)
        output = TensorDict(output, batch_size=len(batch))

        batch = tq.kv_batch_put(keys=batch.keys, partition_id=batch.partition_id, fields=output)

        return batch

    def _update_critic(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
        """Update the critic network."""
        ppo_mini_batch_size = self.config.critic.ppo_mini_batch_size
        ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
        extra_info = {
            "global_batch_size": ppo_mini_batch_size,
            "mini_batch_size": ppo_mini_batch_size,
            "epochs": self.config.critic.ppo_epochs,
            "seed": self.config.critic.data_loader_seed,
            "dataloader_kwargs": {"shuffle": self.config.critic.shuffle},
            "temperature": self.config.actor_rollout_ref.rollout.temperature,
        }
        batch.extra_info.update(extra_info)

        output: DataProtoFuture = self.critic_wg.train_mini_batch(batch)
        output: TensorDict = output.get()
        output = rename_dict(output["metrics"], "critic/")
        output["perf/mfu/critic"] = output.pop("critic/mfu")
        critic_metrics = reduce_metrics(output)
        metrics.update(critic_metrics)

        return batch

    def _update_actor(
        self, batch: KVBatchMeta, metrics: dict,
        skip_optimizer_zero_grad: bool = False,
        skip_optimizer_step: bool = False,
        metric_namespace: str | None = None,
        micro_batch_size_per_gpu: int | None = None,
    ) -> KVBatchMeta:
        """Update the actor network."""
        ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
        perception_sft_coef = float(os.environ.get("V35_PERCEPTION_SFT_COEF", "0"))
        online_cooperative_enabled = bool(self.config.get("online_cooperative", {}).get("enabled", False))
        if online_cooperative_enabled:
            # The online collector expands one source snapshot into one Stage-2 row
            # per intersection and per candidate (plus one perception-SFT row per
            # intersection).  `rollout.n` remains one because vLLM itself produces
            # one completion at a time, so the generic calculation below would make
            # an 8-row mini-batch for a 392-row online actor batch.  That caused 49
            # optimizer updates from one four-master collection.  Keep the complete
            # expanded online collection in one PPO mini-batch; the engine still
            # uses `ppo_micro_batch_size_per_gpu` for memory-safe accumulation.
            ppo_mini_batch_size = len(batch)
            _emit_online_diag(
                "[ONLINE_ACTOR_BATCHING] "
                f"actor_rows={len(batch)} configured_source_mini_batch="
                f"{self.config.actor_rollout_ref.actor.ppo_mini_batch_size} "
                f"configured_rollout_n={self.config.actor_rollout_ref.rollout.n} "
                f"effective_global_mini_batch={ppo_mini_batch_size} expected_optimizer_steps="
                f"{self.config.actor_rollout_ref.actor.ppo_epochs}"
            )
        else:
            samples_per_prompt = self.config.actor_rollout_ref.rollout.n + (1 if perception_sft_coef > 0.0 else 0)
            ppo_mini_batch_size = ppo_mini_batch_size * samples_per_prompt
        calculate_entropy = self.config.actor_rollout_ref.actor.calculate_entropy or (
            self.config.actor_rollout_ref.actor.entropy_coeff != 0.0
        )
        distillation_use_topk = (
            self.distillation_config.distillation_loss.loss_settings.use_topk
            if is_distillation_enabled(self.config.get("distillation"))
            else False
        )
        distillation_only = False  # distillation_only flag means we can skip policy loss and reduce mem footprint
        if is_distillation_enabled(self.config.get("distillation")):
            distillation_loss_cfg = self.distillation_config.distillation_loss
            distillation_only = (
                distillation_use_topk
                and not distillation_loss_cfg.use_task_rewards
                and not distillation_loss_cfg.use_policy_gradient
            )
        extra_info = {
            "calculate_entropy": calculate_entropy,
            "distillation_use_topk": distillation_use_topk,
            "distillation_only": distillation_only,
            "global_batch_size": ppo_mini_batch_size,
            "mini_batch_size": ppo_mini_batch_size,
            "epochs": self.config.actor_rollout_ref.actor.ppo_epochs,
            "seed": self.config.actor_rollout_ref.actor.data_loader_seed,
            "dataloader_kwargs": {"shuffle": self.config.actor_rollout_ref.actor.shuffle},
            "temperature": self.config.actor_rollout_ref.rollout.temperature,
            "skip_optimizer_zero_grad": skip_optimizer_zero_grad,
            "skip_optimizer_step": skip_optimizer_step,
            "skip_lr_scheduler_step": skip_optimizer_step,
        }
        if micro_batch_size_per_gpu is not None:
            if micro_batch_size_per_gpu <= 0:
                raise ValueError("micro_batch_size_per_gpu must be positive")
            extra_info["micro_batch_size_per_gpu"] = int(micro_batch_size_per_gpu)
            extra_info["online_update_namespace"] = metric_namespace or "default"
            _emit_online_diag(
                f"[ONLINE_ACTOR_MICROBATCH_CONFIG] namespace={metric_namespace or 'default'} "
                f"rows={len(batch)} micro_batch_size_per_gpu={micro_batch_size_per_gpu}"
            )
        batch.extra_info.update(extra_info)

        output: TensorDict = self.actor_rollout_wg.update_actor(batch)
        output = rename_dict(output["metrics"], "actor/")
        output["perf/mfu/actor"] = output.pop("actor/mfu")
        actor_metrics = reduce_metrics(output)
        if metric_namespace:
            metrics.update({
                f"actor/{metric_namespace}/{key.removeprefix('actor/')}": value
                for key, value in actor_metrics.items()
            })
        metrics.update(actor_metrics)

        return batch

    def _compute_metrics(self, batch: KVBatchMeta, metrics, timing_raw, global_steps, epoch):
        # 1. collect necessary fields from TransferQueue for computing metrics
        non_padding_mask = np.array([not tag.get("is_padding", False) for tag in batch.tags], dtype=bool)
        fields = [
            "prompts",
            "responses",
            "response_mask",
            "values",
            "advantages",
            "returns",
            "rm_scores",
            "token_level_rewards",
            "num_turns",
        ]
        moe_lb_metrics_interval = self.config.actor_rollout_ref.rollout.get("moe_load_balance_metrics_interval", 0)
        data = get_metric_data_with_optional_routed_experts(
            keys=batch.keys,
            partition_id=batch.partition_id,
            fields=fields,
            moe_lb_metrics_interval=moe_lb_metrics_interval,
            global_steps=global_steps,
            accumulator=self._rollout_moe_lb_metrics_accumulator,
            kv_batch_get=tq.kv_batch_get,
        )

        # Text-only online SUMO trajectories are always one turn.  Keep the
        # standard metric when AgentLoop supplied the field, but do not let a
        # missing optional metric field terminate an already-completed update.
        if "num_turns" in data:
            num_turns = np.asarray(data.pop("num_turns").tolist(), dtype=np.int32)
        else:
            _emit_online_diag(
                "[ONLINE_METRICS_NUM_TURNS_FALLBACK] num_turns missing; "
                f"using one turn for {len(batch.keys)} online rows"
            )
            num_turns = np.ones(len(batch.keys), dtype=np.int32)
        prompt_length = data["prompts"].offsets().diff()
        response_length = data["responses"].offsets().diff()
        global_token_num = (prompt_length + response_length).tolist()
        min_global_steps = np.array([tag["min_global_steps"] for tag in batch.tags], dtype=int)[non_padding_mask]
        max_global_steps = np.array([tag["max_global_steps"] for tag in batch.tags], dtype=int)[non_padding_mask]

        # Only fetch speculative decoding stats when rollout writes them.
        spec_drafts = spec_accepts = spec_verifies = None
        mtp_config = getattr(self.config.actor_rollout_ref.model, "mtp", None)
        if mtp_config is not None and mtp_config.enable and mtp_config.enable_rollout:
            spec_data = tq.kv_batch_get(
                keys=batch.keys,
                partition_id=batch.partition_id,
                select_fields=["extra_fields"],
            )
            extra_fields = spec_data["extra_fields"].tolist()
            spec_drafts = [extra_field["spec_num_draft_tokens"] for extra_field in extra_fields]
            spec_accepts = [extra_field["spec_num_accepted_tokens"] for extra_field in extra_fields]
            spec_verifies = [extra_field["spec_num_verify_steps"] for extra_field in extra_fields]

        data = data.to_padded_tensor()
        data["token_level_scores"] = data["rm_scores"]
        if "token_level_rewards" not in data:
            data["token_level_rewards"] = data["rm_scores"]
        data["prompt_length"] = prompt_length.float()
        data["response_length"] = response_length.float()
        batch = DataProto(batch=data, meta_info={"global_token_num": global_token_num})
        metrics_batch = batch.select_idxs(non_padding_mask) if non_padding_mask.any() else batch

        # 2. compute metrics
        metrics.update({"training/global_step": global_steps, "training/epoch": epoch})
        metrics.update(
            compute_moe_lb_metrics(
                metrics_batch=metrics_batch,
                moe_lb_metrics_interval=moe_lb_metrics_interval,
                global_steps=global_steps,
                accumulator=self._rollout_moe_lb_metrics_accumulator,
            )
        )
        metrics.update(compute_data_metrics(batch=metrics_batch, use_critic=self.use_critic))
        # Canonical per-intersection online reward used by TensorBoard for
        # both train and validation comparisons.  This is deliberately a
        # monitoring scalar; segmented GDPO advantages remain unchanged.
        online_fields = metrics_batch.non_tensor_batch
        if "local_score" in online_fields or "local_queue_reward" in online_fields:
            def _online_values(name, fallback=0.0):
                values = np.asarray(online_fields.get(name, fallback), dtype=np.float32)
                return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).reshape(-1)
            local_values = _online_values("local_score", _online_values("local_queue_reward"))
            network_values = _online_values("network_reward", _online_values("global_queue_reward"))
            format_values = _online_values("format_penalty")
            cost_values = _online_values("reasoning_cost_reward")
            total_values = local_values + 0.5 * network_values + format_values + 0.5 * cost_values
            for name, values in (("local", local_values), ("network", network_values),
                                 ("format_penalty", format_values), ("reasoning_cost", cost_values),
                                 ("total", total_values)):
                if values.size:
                    metrics.update({
                        f"reward/{name}": float(values.mean()),
                        f"reward/{name}/std": float(values.std()),
                        f"reward/{name}/min": float(values.min()),
                        f"reward/{name}/max": float(values.max()),
                    })
        metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
        n_gpus = self._get_n_gpus_for_throughput()
        metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
        gradient_norm = metrics.get("actor/grad_norm", None)
        metrics.update(compute_variance_proxy_metrics(batch=metrics_batch, gradient_norm=gradient_norm))

        # 3. other auxiliary metrics
        if non_padding_mask.any():
            num_turns = num_turns[non_padding_mask]
        metrics.update(
            {
                "training/num_turns/mean": num_turns.mean(),
                "training/num_turns/max": num_turns.max(),
                "training/num_turns/min": num_turns.min(),
            }
        )

        # 4. per-request speculative-decoding aggregation (same metrics async PPO logs;
        # see compute_spec_decode_metrics in verl/trainer/ppo/ray_trainer.py).
        metrics.update(compute_spec_decode_metrics(spec_drafts, spec_accepts, spec_verifies, non_padding_mask))

        # 5. off-policy staleness metrics
        #   global_steps is the model weight version (one update_weights per global_step), and
        #   min/max_global_steps are the versions a trajectory was generated across, so all quantities
        #   below are already in model-version units.
        #   - trajectory_spans: how many distinct model versions a single trajectory was
        #     generated across (1 == fully generated on a single version). This captures the
        #     within-trajectory policy inconsistency caused by partial rollout / continuation.
        #   - trajectory_staleness: how many model versions the trajectory lags behind the
        #     *current* policy. A trajectory spans versions [min_global_steps, max_global_steps],
        #     so the lag is a range: the freshest weights used give the lower bound
        #     (global_steps - max_global_steps) and the oldest weights the worst case
        #     (global_steps - min_global_steps). We log the lower bound as the primary metric.
        trajectory_spans = max_global_steps - min_global_steps + 1
        trajectory_staleness = (global_steps - 1) - max_global_steps
        trajectory_staleness_worst = (global_steps - 1) - min_global_steps
        metrics.update(
            {
                "training/off_policy/trajectory_spans/mean": trajectory_spans.mean(),
                "training/off_policy/trajectory_spans/max": trajectory_spans.max(),
                "training/off_policy/trajectory_spans/min": trajectory_spans.min(),
                "training/off_policy/trajectory_staleness/mean": trajectory_staleness.mean(),
                "training/off_policy/trajectory_staleness/max": trajectory_staleness.max(),
                "training/off_policy/trajectory_staleness/min": trajectory_staleness.min(),
                "training/off_policy/trajectory_staleness_worst/mean": trajectory_staleness_worst.mean(),
                "training/off_policy/trajectory_staleness_worst/max": trajectory_staleness_worst.max(),
                "training/off_policy/trajectory_staleness_worst/min": trajectory_staleness_worst.min(),
            }
        )


TRAINER_REGISTRY: dict[str, type[PPOTrainer]] = {}


def register_trainer(name: str):
    """Class decorator that registers a :class:`PPOTrainer` subclass under ``name``."""

    def decorator(cls: type[PPOTrainer]) -> type[PPOTrainer]:
        if not (isinstance(cls, type) and issubclass(cls, PPOTrainer)):
            raise TypeError(f"register_trainer expected a PPOTrainer subclass, got {cls!r}")
        existing = TRAINER_REGISTRY.get(name)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"Trainer name '{name}' is already registered to {existing.__name__}; "
                f"cannot re-register it to {cls.__name__}."
            )
        TRAINER_REGISTRY[name] = cls
        return cls

    return decorator


def get_trainer_cls(name: str) -> type[PPOTrainer]:
    """Return the :class:`PPOTrainer` subclass registered under ``name``."""
    try:
        return TRAINER_REGISTRY[name]
    except KeyError:
        available = ", ".join(sorted(TRAINER_REGISTRY)) or "<none>"
        raise ValueError(f"Unknown trainer '{name}'. Available trainers: {available}.") from None
