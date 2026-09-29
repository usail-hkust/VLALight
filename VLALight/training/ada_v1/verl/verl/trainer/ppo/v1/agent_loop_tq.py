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

# TODO: move this file to verl.experimental.agent_loop after V1 is stable
"""TransferQueue adapter for AgentLoopManager and AgentLoopWorker"""

import asyncio
import json
import logging
import os
from typing import Any

import ray
import torch
import transfer_queue as tq
from tensordict import NonTensorData, NonTensorStack, TensorDict

from verl.experimental.agent_loop import (
    AgentLoopManager,
    AgentLoopOutput,
    AgentLoopWorker,
    get_trajectory_info,
)
from verl.utils.ray_utils import auto_await
from verl.utils.tensordict_utils import list_of_dict_to_tensordict

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


def apply_greedy_sampling_params(params: dict[str, Any]) -> None:
    params["top_p"] = 1.0
    params["top_k"] = -1
    params["temperature"] = 0


async def _settle_session_tasks(tasks: list[asyncio.Task[Any]]) -> list[BaseException]:
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return [result for result in results if isinstance(result, BaseException)]


@ray.remote
class AgentLoopWorkerTQ(AgentLoopWorker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        tq.init()
        self.background_tasks = set()

    async def generate_sequences(self, batch: TensorDict) -> None:
        """Spawn agent loop for each sample in the batch without waiting for the results."""
        validate = batch["validate"] if "validate" in batch else False
        batch.pop("validate", None)
        config = self.config.actor_rollout_ref.rollout
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repetition_penalty=1.0,
            logprobs=config.calculate_log_probs,
        )

        # override sampling params for validation
        if validate:
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["top_k"] = config.val_kwargs.top_k
            sampling_params["temperature"] = config.val_kwargs.temperature

        # by default, we assume it's a single turn agent
        if "agent_name" not in batch:
            default_agent_loop = config.agent.default_agent_loop
            batch["agent_name"] = NonTensorData(default_agent_loop)

        trajectory_info = await get_trajectory_info(batch["global_steps"], batch["index"], validate)

        # create background tasks for each sample in the batch
        for i in range(len(batch)):
            # TODO(wuxibin): add trace support
            trace_this_sample = False
            prompt = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    prompt[k] = v[i]
                elif isinstance(v, NonTensorStack):
                    prompt[k] = v[i].data
                elif isinstance(v, NonTensorData):
                    prompt[k] = v.data
                else:
                    logger.exception(f"Unsupported type {type(v)} for key {k}")

            # “fire-and-forget” background tasks
            task = asyncio.create_task(
                self._run_prompt(prompt, sampling_params, trajectory=trajectory_info[i], trace=trace_this_sample)
            )
            self.background_tasks.add(task)
            task.add_done_callback(self.background_tasks.discard)

    async def _run_prompt(self, prompt: dict, sampling_params: dict, trajectory: dict, trace: bool = False) -> None:
        """Spawn multiple agent loops in parallel according to rollout.n or rollout.val_kwargs.n."""
        uid, partition_id = prompt["uid"], "train" if not trajectory["validate"] else "val"
        await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": "running"})
        tasks = []
        try:
            # NOTE: user can dynamically adjust n for each sample here, e.g according to task difficulty.
            config = self.config.actor_rollout_ref.rollout
            n = prompt.pop("__rollout_n__", config.n if not trajectory["validate"] else config.val_kwargs.n)
            do_sample = prompt.pop("__do_sample__", True)

            run_sampling_params = dict(sampling_params)
            if not trajectory["validate"] and not do_sample:
                apply_greedy_sampling_params(run_sampling_params)

            tasks = []
            for i in range(n):
                task = asyncio.create_task(
                    self._run_agent_loop(
                        run_sampling_params,
                        trajectory=trajectory,
                        trace=trace,
                        is_validation=bool(trajectory["validate"]),
                        session_id=i,
                        rollout_n=i,
                        **prompt,
                    )
                )
                tasks.append(task)

            # Publish a terminal status only after every session settles, so no sibling can write after
            # ReplayBuffer clears a failed group.
            session_errors = await _settle_session_tasks(tasks)
            if session_errors:
                for error in session_errors:
                    logger.error(
                        f"Error in _run_prompt for uid={uid}",
                        exc_info=(type(error), error, error.__traceback__),
                    )
                status = "failure"
            else:
                status = "finished"
            await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": status})
        except Exception as e:
            logger.exception(f"Error in _run_prompt: {e}")
            if tasks:
                await _settle_session_tasks(tasks)
            await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": "failure"})

    async def _agent_loop_postprocess(
        self, output: AgentLoopOutput | list[AgentLoopOutput], validate, **kwargs
    ) -> None:
        """Put agent loop outputs into TransferQueue."""
        uid, session_id = kwargs["uid"], kwargs["session_id"]
        outputs = output if isinstance(output, list) else [output]
        if not outputs:
            logger.warning(f"Empty output for prompt {uid}_{session_id}")
            return

        extra_info = dict(kwargs.get("extra_info") or {})
        extra_info["split"] = "val" if validate else "train"
        # Preserve the logical bootstrap id explicitly.  AgentLoopOutput's
        # generic ``as_dict`` may contain an empty sample_id and otherwise
        # overwrite/drop the non-tensor prompt metadata during postprocess.
        # The generation job carries this field in kwargs; copy it into both
        # locations consumed by TransferQueue and the online materializer.
        # TQ jobs historically carry the stable logical id as
        # ``extra_info["index"]`` (and some legacy paths use ``kwargs["index"]``)
        # rather than a top-level sample_id.  Use those sources before
        # allowing an empty id to reach the online publisher.
        sample_id = (
            kwargs.get("sample_id")
            or extra_info.get("sample_id")
            or extra_info.get("index")
            or kwargs.get("index")
        )
        if sample_id:
            kwargs["sample_id"] = str(sample_id)
            extra_info["sample_id"] = str(sample_id)
        else:
            logger.warning(
                "[V1_SAMPLE_ID_MISSING_AT_POSTPROCESS] uid=%r index=%r",
                kwargs.get("uid"), kwargs.get("index"),
            )
        rollout_n = kwargs.get("rollout_n")
        if rollout_n is not None:
            rollout_n = int(rollout_n)
            extra_info["rollout_n"] = rollout_n
            if not validate and os.environ.get("V35_FORCE_MODE_BALANCE", "0") == "1":
                rollout_count = int(os.environ.get("V35_ROLLOUT_N", "6"))
                if rollout_count >= 2 and rollout_count % 2 == 0:
                    extra_info["forced_mode"] = "fast" if rollout_n < rollout_count // 2 else "slow"
        kwargs["extra_info"] = extra_info
        await self._compute_score(outputs, kwargs=kwargs)

        final_output = outputs[-1]
        # TODO: Support output:list[AgentLoopOutput]
        await self._compute_teacher_logprobs(
            final_output,
            prompt_ids=final_output.prompt_ids,
            response_ids=final_output.response_ids,
            validate=validate,
            sample_kwargs=kwargs,
        )

        if final_output.reward_score is not None:
            for output in outputs[:-1]:
                output.reward_score = final_output.reward_score
                output.extra_fields["reward_extra_info"] = final_output.extra_fields["reward_extra_info"]

        # NOTE: agent loop may has multiple outputs, put each output into TransferQueue.
        # key format: {uid}_{session_id}_{index}
        # - uid: raw prompt uid from dataset
        # - session_id: session id for rollout.n sampling
        # - index: index of agent loop output
        keys, fields, tags = [], [], []
        for i, output in enumerate(outputs):
            prompts = torch.tensor(output.prompt_ids, dtype=torch.int64)
            responses = torch.tensor(output.response_ids, dtype=torch.int64)
            input_ids = torch.cat([prompts, responses], dim=0)
            attention_mask = torch.ones_like(input_ids, dtype=torch.int64)
            multi_modal_inputs = self._compute_multi_modal_inputs(output, input_ids)
            position_ids = self._compute_position_ids(
                input_ids.unsqueeze(0), attention_mask.unsqueeze(0), multi_modal_inputs
            ).squeeze(0)

            actor_aux_fields = {}
            reward_info = output.extra_fields.get("reward_extra_info", {})
            perception_sft_coef = float(os.environ.get("V35_PERCEPTION_SFT_COEF", "0"))
            signal_aux_coef = float(os.environ.get("V35_SIGNAL_AUX_COEF", "0"))
            mode_selector_coef = float(os.environ.get("V35_MODE_SELECTOR_COEF", "0"))
            segmented_actor_loss = any(
                coef > 0.0 for coef in (perception_sft_coef, signal_aux_coef, mode_selector_coef)
            )
            if segmented_actor_loss and not validate:
                from v35_offline_grpo.perception_sft import build_grpo_loss_mask

                rollout_response_mask = torch.tensor(output.response_mask, dtype=torch.int64)
                grpo_loss_mask = torch.tensor(
                    build_grpo_loss_mask(output.response_ids, self.tokenizer), dtype=torch.int64
                )
                grpo_loss_mask *= rollout_response_mask
                decision_valid = float(int(reward_info.get("perception_valid", 1)) != 0)
                actor_aux_fields.update(
                    grpo_loss_mask=grpo_loss_mask,
                    grpo_advantage_scale=torch.full_like(
                        rollout_response_mask, decision_valid, dtype=torch.float32
                    ),
                    kl_loss_mask=rollout_response_mask.clone(),
                )

            # Stage 1 is the only generation pass that owns an SFT target.
            # Stage 2 rows are later joined with the aligned Stage 1 gold
            # tensors by ``attach_stage1_supervision``.  Trying to construct
            # a second SFT target here is both incorrect and fragile: online
            # Stage 2 metadata may carry a non-JSON source marker rather than
            # a perception target.
            job_kind = str(kwargs.get("job_kind", ""))
            is_stage1_sft_job = job_kind == "stage1"
            if perception_sft_coef > 0.0 and not validate and is_stage1_sft_job:
                from v35_offline_grpo.perception_sft import (
                    build_perception_sft_token_weights,
                    build_perception_target,
                )

                try:
                    target_text = build_perception_target(
                        kwargs.get("reward_model"), kwargs.get("raw_prompt")
                    )
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    sample_id = kwargs.get("index", "unknown")
                    reward_model = kwargs.get("reward_model")
                    ground_truth = reward_model.get("ground_truth") if isinstance(reward_model, dict) else None
                    raise ValueError(
                        "Stage 1 perception SFT gold is invalid "
                        f"for uid={sample_id!r}: ground_truth_type={type(ground_truth).__name__} "
                        f"ground_truth_len={len(ground_truth) if isinstance(ground_truth, str) else 'n/a'}"
                    ) from exc
                target_ids = self.tokenizer.encode(target_text, add_special_tokens=False)
                if len(target_ids) > self.config.actor_rollout_ref.rollout.response_length:
                    sample_id = kwargs.get("index", "unknown")
                    raise ValueError(
                        f"perception SFT target for {sample_id} has {len(target_ids)} tokens, "
                        f"exceeding response_length={self.config.actor_rollout_ref.rollout.response_length}"
                    )

                perception_responses = torch.tensor(target_ids, dtype=torch.int64)
                perception_mask = torch.ones_like(perception_responses, dtype=torch.int64)
                perception_token_weight = torch.tensor(
                    build_perception_sft_token_weights(target_text, self.tokenizer),
                    dtype=torch.float32,
                )
                if perception_token_weight.numel() != perception_responses.numel():
                    raise ValueError("perception token weights are not aligned with target tokens")
                perception_input_ids = torch.cat([prompts, perception_responses], dim=0)
                perception_attention_mask = torch.ones_like(perception_input_ids, dtype=torch.int64)
                perception_position_ids = self._compute_position_ids(
                    perception_input_ids.unsqueeze(0),
                    perception_attention_mask.unsqueeze(0),
                    dict(multi_modal_inputs),
                ).squeeze(0)
                actor_aux_fields.update({
                    "perception_sft_responses": perception_responses,
                    "perception_sft_input_ids": perception_input_ids,
                    "perception_sft_attention_mask": perception_attention_mask,
                    "perception_sft_position_ids": perception_position_ids,
                    "perception_sft_mask": perception_mask,
                    "perception_sft_token_weight": perception_token_weight,
                })
            elif perception_sft_coef > 0.0 and not validate and job_kind.startswith("stage2"):
                # Keep this compact because it can be emitted once per
                # rollout.  It proves Stage 2 did not attempt to parse its
                # metadata as an independent perception gold target.
                logger.debug("[STAGE1_SFT_DEFERRED] uid=%s job_kind=%s", kwargs.get("index"), job_kind)

            if signal_aux_coef > 0.0 and not validate:
                from v35_offline_grpo.perception_sft import build_tag_content_mask

                signal_mask = torch.tensor(
                    build_tag_content_mask(output.response_ids, self.tokenizer, "signal"), dtype=torch.int64
                )
                signal_mask *= torch.tensor(output.response_mask, dtype=torch.int64)
                signal_advantage = float(reward_info.get("signal_counterfactual_advantage", 0.0))
                actor_aux_fields.update(
                    signal_aux_mask=signal_mask,
                    signal_aux_advantage=signal_mask.to(torch.float32) * signal_advantage,
                )

            if mode_selector_coef > 0.0 and not validate:
                from v35_offline_grpo.perception_sft import build_tag_content_mask

                mode_mask = torch.tensor(
                    build_tag_content_mask(output.response_ids, self.tokenizer, "mode"), dtype=torch.int64
                )
                mode_mask *= torch.tensor(output.response_mask, dtype=torch.int64)
                actor_aux_fields.update(
                    mode_aux_mask=mode_mask,
                    mode_aux_weight=torch.zeros_like(mode_mask, dtype=torch.float32),
                )

            # Processor-only multimodal metadata is needed for M-RoPE above,
            # but should not be forwarded to the model.
            multi_modal_inputs.pop("mm_token_type_ids", None)

            keys.append(f"{uid}_{session_id}_{i}")
            field = output.as_dict()
            field.update(kwargs)
            # do not store raw image/video
            field.pop("multi_modal_data", None)
            # TODO: uniform response_mask and loss_mask
            field["loss_mask"] = field["response_mask"]
            field["input_ids"] = input_ids
            field["position_ids"] = position_ids
            field["multi_modal_inputs"] = multi_modal_inputs
            field.update(actor_aux_fields)
            fields.append(field)
            prompt_len, response_len = field["prompts"].size(0), field["responses"].size(0)
            tags.append(
                {
                    "status": "success",
                    "prompt_len": prompt_len,
                    "response_len": response_len,
                    "seq_len": prompt_len + response_len,
                    # These tags are used for off-policy staleness control, if a trajectory
                    # spans too many global steps, we need to filter it out.
                    # global_steps: which global steps this sample is from dataloader
                    "global_steps": kwargs["global_steps"],
                    # min_global_steps: start generation model weights version of this trajectory
                    "min_global_steps": field["extra_fields"].get("min_global_steps"),
                    # max_global_steps: end generation model weights version of this trajectory
                    "max_global_steps": field["extra_fields"].get("max_global_steps"),
                }
            )

        await tq.async_kv_batch_put(
            keys=keys,
            fields=list_of_dict_to_tensordict(fields),
            tags=tags,
            partition_id="train" if not validate else "val",
        )


class AgentLoopManagerTQ(AgentLoopManager):
    def __init__(self, *args, **kwargs):
        self.agent_loop_workers_class = AgentLoopWorkerTQ
        super().__init__(*args, **kwargs)

    @classmethod
    @auto_await
    async def create(cls, *args, **kwargs):
        """Create agent loop manager."""
        instance = cls(*args, **kwargs)
        await instance._init_agent_loop_workers()
        return instance

    def generate_sequences(self, prompts: TensorDict) -> None:
        """
        Dispatch input batch to agent loop workers without blocking. Workers should put agent loop outputs
        into TransferQueue once an agent loop finished.

        Args:
            prompts (TensorDict): Input batch from train or validation dataset.
        """
        chunkes = prompts.chunk(len(self.agent_loop_workers))
        ray.get(
            [
                worker.generate_sequences.remote(chunk)
                for worker, chunk in zip(self.agent_loop_workers, chunkes, strict=False)
            ]
        )
