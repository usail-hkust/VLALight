"""V1 bridge for the live cooperative SUMO collector.

V1 normally sends bootstrap JSONL rows straight to ``AgentLoopWorkerTQ``.
Those rows are only stream identities, not policy trajectories.  This manager
materializes the complete online pipeline before publishing trajectories to
TransferQueue, while retaining the V1 optimizer/update implementation.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import transfer_queue as tq
from tensordict import NonTensorData, NonTensorStack, TensorDict

from verl.experimental.agent_loop import AgentLoopManager
from verl.protocol import DataProto
from verl.utils.ray_utils import auto_await
from verl.utils.tensordict_utils import list_of_dict_to_tensordict


def _td_value(value: Any, index: int) -> Any:
    if isinstance(value, NonTensorData):
        return value.data
    if isinstance(value, NonTensorStack):
        return value[index].data
    # TransferQueue/TensorDict may collapse a broadcast metadata field to a
    # scalar (for example ``global_steps``).  Scalars are already the value
    # for every row and are not subscriptable.
    if isinstance(value, (str, bytes, int, float, bool)) or value is None:
        return value
    try:
        return value[index]
    except (IndexError, KeyError, TypeError):
        return value


def _bootstrap_proto(batch: TensorDict) -> DataProto:
    """Convert V1's TensorDict bootstrap rows without losing JSON metadata."""
    non_tensor = {}
    tensors = {}
    for key, value in batch.items():
        if isinstance(value, (NonTensorData, NonTensorStack)):
            non_tensor[key] = np.asarray([_td_value(value, i) for i in range(len(batch))], dtype=object)
        else:
            tensors[key] = value
    return DataProto(batch=TensorDict(tensors, batch_size=len(batch)), non_tensor_batch=non_tensor)


class OnlineCooperativeAgentLoopManagerTQ:
    """Collect online Stage 1/2 trajectories, then publish them to TQ."""

    def __init__(self, *, config, llm_client, teacher_client=None, reward_loop_worker_handles=None, tokenizer=None):
        self.config = config
        self.tokenizer = tokenizer
        self._generator_args = dict(
            config=config,
            llm_client=llm_client,
            teacher_client=teacher_client,
            reward_loop_worker_handles=reward_loop_worker_handles,
        )
        self.generator = None
        self.train_collector = None
        self.val_collector = None
        self._runtimes = []
        # Keep an explicit split index so the trainer can tear down only the
        # validation pool at the end of one complete validation pass.  Val
        # masters are intentionally persistent *within* a pass (the ten
        # lanes advance one V25 cycle between batches), but must not survive
        # into the following Train pass and contend for EGL/TraCI resources.
        self._runtime_by_split = {}

    @classmethod
    @auto_await
    async def create(cls, *args, **kwargs):
        instance = cls(*args, **kwargs)
        instance.generator = await AgentLoopManager.create(**instance._generator_args)
        instance._init_collectors()
        return instance

    def _init_collectors(self) -> None:
        online_node = self.config.get("online_cooperative", {})
        if not bool(online_node.get("enabled", False)):
            raise ValueError("OnlineCooperativeAgentLoopManagerTQ requires online_cooperative.enabled=true")
        online_root = next(
            (path for path in Path(__file__).resolve().parents if (path / "v35_online_cooperative_grpo").is_dir()),
            None,
        )
        if online_root is None:
            raise RuntimeError("cannot locate v35_online_cooperative_grpo")
        if str(online_root) not in sys.path:
            sys.path.insert(0, str(online_root))
        repo_root = next(
            (path for path in online_root.parents if (path / "utils").is_dir() and (path / "data").is_dir()),
            online_root.parent,
        )
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from v35_online_cooperative_grpo import (
            OnlineCooperativeRuntime, OnlineVERLCollector, RayCityRolloutCoordinator,
            SUMOEnvFactory, build_city_snapshot, make_policy_factory,
        )
        from v35_online_cooperative_grpo.online_config import hydrate_repository_defaults, load_online_config

        config_path = online_node.get("config_path")
        if not config_path:
            raise ValueError("online_cooperative.config_path is required when enabled")
        online_cfg = hydrate_repository_defaults(load_online_config(config_path))
        # V1's outer rollout.n remains 1 (one Stage-1 pass).  Downstream
        # segmented losses need the actual inner Stage-2 candidate count.
        os.environ["V35_ONLINE_NUM_ROLLOUTS"] = str(online_cfg.num_rollouts)
        env_factory = SUMOEnvFactory(
            {name: city.traffic_env_conf for name, city in online_cfg.cities.items()},
            {name: city.path_conf for name, city in online_cfg.cities.items()},
            work_root=Path(os.environ.get("V35_RUNTIME_ROOT", online_root / "runtime")) / "sumo_work",
            media_root=Path(os.environ.get("V35_RUNTIME_ROOT", online_root / "runtime")) / "media",
            repo_root=repo_root,
            decision_cycle_seconds=online_cfg.decision_cycle_seconds,
        )
        coordinator = RayCityRolloutCoordinator(env_factory, num_rollouts=online_cfg.num_rollouts)

        def master_factory(city, seed, actor_id):
            return coordinator.create_master_actor(city, seed, actor_id=actor_id)

        def observation_builder(master, city, step):
            return build_city_snapshot(master, city, step, routes_dir=online_root / "artifacts")

        policy_factory = make_policy_factory(
            base_url=online_cfg.policy.base_url, model=online_cfg.policy.model,
            temperature=online_cfg.exploration_temperature, max_tokens=online_cfg.policy.max_tokens,
            timeout=online_cfg.policy.timeout, api_key=online_cfg.policy.api_key,
        )
        root = Path(os.environ.get("V35_RUNTIME_ROOT", online_root / "runtime")) / "snapshots"
        for split in ("train", "val"):
            runtime = OnlineCooperativeRuntime(
                online_cfg, split=split, master_factory=master_factory, coordinator=coordinator,
                observation_builder=observation_builder, policy_factory=policy_factory,
                prompt_template=None, snapshot_root=root,
            )
            self._runtimes.append(runtime)
            self._runtime_by_split[split] = runtime
            collector = OnlineVERLCollector(runtime, config=online_cfg, split=split)
            if split == "train":
                self.train_collector = collector
            else:
                self.val_collector = collector
        print("[V1_ONLINE_COLLECTOR_READY] stage1_stage2_sumo_bridge=enabled", flush=True)

    def reset_split(self, split: str) -> None:
        """Reset one online split's mutable SUMO master pool.

        Validation owns ten persistent masters for the duration of one
        ``_validate`` call.  The trainer invokes this method from a
        ``finally`` block, so the pool is destroyed both after a successful
        pass and when validation raises.  Train is deliberately untouched.
        Calling this when no masters were created is a no-op apart from the
        diagnostic line, which makes repeated cleanup safe.
        """
        split = str(split)
        runtime = self._runtime_by_split.get(split)
        if runtime is None:
            return
        runtime.reset()
        print(f"[V1_ONLINE_RUNTIME_RESET] split={split} masters_destroyed=true", flush=True)

    def suspend_train_masters_for_validation(self) -> None:
        """Free Train's native SUMO/EGL pool for a complete validation pass."""
        runtime = self._runtime_by_split.get("train")
        if runtime is not None:
            runtime.suspend_for_validation()

    def resume_train_masters_after_validation(self) -> None:
        """Restore Train's native SUMO/EGL pool after validation cleanup."""
        runtime = self._runtime_by_split.get("train")
        if runtime is not None:
            runtime.resume_after_validation()

    def _publish(self, online_batch, source_uids: list[str], *, validate: bool, global_steps: int) -> None:
        data = online_batch.data_proto
        if data is None or len(data) == 0:
            raise ValueError("online collector returned no materialized trajectories")
        required = {
            "prompts", "responses", "input_ids", "attention_mask", "position_ids", "response_mask",
            "rm_scores", "grpo_loss_mask", "mode_aux_mask", "reasoning_aux_mask", "signal_aux_mask",
        }
        # A current online mode-enabled run must publish the row-level binary
        # classifier metadata as well.  Keep the check conditional so archived
        # data can still be inspected with the manager when the mode branch is
        # disabled.
        if float(os.environ.get("V35_MODE_SELECTOR_COEF", "0")) > 0.0:
            required.update({
                "mode_binary_mask", "mode_token_index", "mode_fast_token_id",
                "mode_slow_token_id", "mode_binary_target", "mode_binary_valid",
                "mode_binary_weight",
            })
        missing = sorted(required - set(data.batch.keys()))
        if missing:
            raise ValueError(f"V1 online collector output is missing tensors: {missing}")
        if not validate:
            sft_required = {
                "perception_sft_responses", "perception_sft_input_ids", "perception_sft_attention_mask",
                "perception_sft_position_ids", "perception_sft_mask", "perception_sft_token_weight",
            }
            missing_sft = sorted(sft_required - set(data.batch.keys()))
            if missing_sft:
                raise ValueError(f"V1 online collector output is missing Stage 1 SFT tensors: {missing_sft}")
        for key in ("decision_group_id", "mode_group_id", "extra_fields"):
            if key not in data.non_tensor_batch:
                raise ValueError(f"V1 online collector output is missing metadata {key!r}")

        # ``records_to_data_proto`` materializes rows in exactly this order.
        # TransferQueue/TQ can drop non-tensor fields, so retain the original
        # collector records as the authoritative identity/group metadata.
        authoritative_records = list(getattr(online_batch, "records", ()) or ())
        if len(authoritative_records) != len(data):
            raise ValueError(
                "online collector metadata length mismatch: "
                f"records={len(authoritative_records)} materialized={len(data)}"
            )
        metadata_keys = (
            "sample_id", "sample_index", "uid", "group_id", "decision_group_id",
            "mode_group_id", "episode_group_id", "network_group_id", "intersection_id",
            "city", "step", "rollout_id",
        )
        metadata_repaired = 0
        metadata_missing = []
        result_sample_ids = [str(item.spec.sample_id) for item in online_batch.result.items]
        sample_to_uid = {
            sample_id: source_uids[index]
            for index, sample_id in enumerate(result_sample_ids)
            if index < len(source_uids) and sample_id
        }
        # Some V1 agent-loop paths do not preserve ``sample_id`` in the
        # materialized non-tensor fields.  The result list is still ordered
        # identically to the bootstrap inputs, so retain a positional mapping
        # instead of aborting the whole update on an empty metadata field.
        positional_uids = source_uids if len(result_sample_ids) == len(source_uids) == len(data) else None
        rows, keys, tags = [], [], []
        fallback_counts = {"sample_index": 0, "positional": 0, "expanded": 0}
        for index in range(len(data)):
            row = {key: _td_value(value, index) for key, value in data.batch.items()}
            row.update({key: values[index] for key, values in data.non_tensor_batch.items()})
            authoritative = authoritative_records[index]
            # Never allow an empty materialized value to override the stable
            # value produced by records_for_verl().  This fixes sample_id=''
            # and also preserves all GDPO grouping metadata.
            for key in metadata_keys:
                expected = authoritative.get(key)
                current = row.get(key)
                if expected is not None and (current is None or str(current) == ""):
                    row[key] = expected
                    metadata_repaired += 1
            sample_id = str(row.get("sample_id", ""))
            source_uid = sample_to_uid.get(sample_id)
            # Materialization may omit sample_id while retaining the stable
            # sample_index copied from the generation job.  This is the
            # correct mapping when one bootstrap expands into many rows.
            if source_uid is None:
                sample_index = row.get("sample_index")
                try:
                    sample_index = int(sample_index)
                except (TypeError, ValueError):
                    sample_index = -1
                if 0 <= sample_index < len(source_uids):
                    source_uid = source_uids[sample_index]
                    row["sample_id"] = result_sample_ids[index] if index < len(result_sample_ids) else sample_id
                    fallback_counts["sample_index"] += 1
            if source_uid is None and positional_uids is not None:
                source_uid = positional_uids[index]
                row["sample_id"] = result_sample_ids[index]
                fallback_counts["positional"] += 1
            # Expanded online batches contain one row per rollout while
            # ``source_uids`` contains one row per bootstrap prompt.  Some TQ
            # versions drop both sample_id and sample_index; recover the
            # owner by the stable grouped order instead of crashing.  This is
            # only accepted when expansion is uniform, so IDs cannot be
            # silently assigned across unequal groups.
            if source_uid is None and source_uids and len(data) % len(source_uids) == 0:
                rows_per_source = len(data) // len(source_uids)
                source_index = min(index // rows_per_source, len(source_uids) - 1)
                source_uid = source_uids[source_index]
                row["sample_id"] = source_uid
                fallback_counts["expanded"] += 1
            if source_uid is None:
                raise ValueError(f"online trajectory has unknown sample_id={sample_id!r}")
            # Validate only after all authoritative/fallback reconstruction has
            # run.  Expanded batches legitimately start without materialized
            # sample_id; checking earlier made a successful fallback fatal.
            for key in ("sample_id", "uid", "decision_group_id", "mode_group_id"):
                if not row.get(key):
                    metadata_missing.append((index, key))
            row["loss_mask"] = row["response_mask"]
            keys.append(f"{source_uid}_online_{index:05d}")
            rows.append(row)
            prompt_len = int(row.get("prompt_len", row["attention_mask"].shape[-1] - row["response_mask"].sum().item()))
            response_len = int(row.get("response_len", row["response_mask"].sum().item()))
            tags.append({
                "status": "success", "prompt_len": prompt_len, "response_len": response_len,
                "seq_len": prompt_len + response_len, "global_steps": global_steps,
                "min_global_steps": global_steps, "max_global_steps": global_steps,
            })
        print(
            f"[V1_ONLINE_PUBLISH_METADATA] split={'val' if validate else 'train'} "
            f"rows={len(data)} repaired={metadata_repaired} "
            f"missing_required={len(metadata_missing)} "
            f"sample_ids={sum(bool(str(r.get('sample_id', ''))) for r in rows)}/{len(rows)} "
            f"fallbacks={fallback_counts}",
            flush=True,
        )
        if metadata_missing:
            raise ValueError(f"authoritative online metadata still missing: {metadata_missing[:8]}")
        partition = "val" if validate else "train"
        tq.kv_batch_put(keys=keys, fields=list_of_dict_to_tensordict(rows), tags=tags, partition_id=partition)
        tq.kv_batch_put(
            keys=source_uids, partition_id=partition,
            tags=[{"is_prompt": True, "status": "finished", "global_steps": global_steps} for _ in source_uids],
        )
        print(
            f"[V1_ONLINE_COLLECTOR_PUBLISH] split={partition} bootstrap_rows={len(source_uids)} "
            f"trajectory_rows={len(rows)} decision_groups={len(set(map(str, data.non_tensor_batch['decision_group_id'])))}",
            flush=True,
        )

    def generate_sequences(self, prompts: TensorDict) -> None:
        validate = bool(_td_value(prompts["validate"], 0)) if "validate" in prompts else False
        source_uids = [str(_td_value(prompts["uid"], i)) for i in range(len(prompts))]
        global_steps = int(_td_value(prompts["global_steps"], 0)) if "global_steps" in prompts else -1
        tq.kv_batch_put(
            keys=source_uids, partition_id="val" if validate else "train",
            tags=[{"is_prompt": True, "status": "running", "global_steps": global_steps} for _ in source_uids],
        )
        try:
            collector = self.val_collector if validate else self.train_collector
            online_batch = collector.collect(
                _bootstrap_proto(prompts), tokenizer=self.tokenizer, materialize=True,
                sequence_generator=self.generator.generate_sequences, global_steps=global_steps, validate=validate,
            )
            self._publish(online_batch, source_uids, validate=validate, global_steps=global_steps)
        except BaseException:
            tq.kv_batch_put(
                keys=source_uids, partition_id="val" if validate else "train",
                tags=[{"is_prompt": True, "status": "failure", "global_steps": global_steps} for _ in source_uids],
            )
            raise

    def close(self) -> None:
        for runtime in reversed(self._runtimes):
            runtime.close()
        self._runtimes.clear()
        self._runtime_by_split.clear()
