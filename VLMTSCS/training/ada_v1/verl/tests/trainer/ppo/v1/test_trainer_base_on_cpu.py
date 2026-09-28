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

from types import SimpleNamespace
from unittest.mock import patch

import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from tensordict.tensorclass import NonTensorStack

from verl.trainer.ppo.padding_utils import construct_minimal_padding_template
from verl.trainer.ppo.v1.replay_buffer import ReplayBuffer, ReplayBufferAsync
from verl.trainer.ppo.v1.trainer_base import PPOTrainer, _build_perception_sft_actor_batch


class _StubTrainer(PPOTrainer):
    def on_step_end(self):
        pass

    def on_sample_end(self):
        pass


class _CustomSampler:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _trainer_with_filter_groups(filter_groups: dict, trainer_mode: str = "sync") -> _StubTrainer:
    trainer = _StubTrainer.__new__(_StubTrainer)
    trainer.trainer_mode = trainer_mode
    trainer.config = OmegaConf.create(
        {
            "algorithm": {"filter_groups": filter_groups},
            "data": {"train_batch_size": 64, "gen_batch_size": 8},
            "reward": {"reward_model": {"enable": False, "enable_resource_pool": False}},
            "trainer": {
                "v1": {
                    trainer_mode: {},
                    "sampler": {
                        "custom_sampler": None,
                        "max_off_policy_threshold": 1,
                        "max_off_policy_strategy": "drop",
                        "sampler_kwargs": {},
                    },
                }
            },
        }
    )
    return trainer


def test_builtin_sampler_class_follows_trainer_mode():
    sync_sampler = _trainer_with_filter_groups({"enable": False}, trainer_mode="sync")._build_replay_buffer()
    async_samplers = [
        _trainer_with_filter_groups({"enable": True, "metric": "acc"}, trainer_mode=mode)._build_replay_buffer()
        for mode in ("colocate_async", "separate_async")
    ]

    assert type(sync_sampler) is ReplayBuffer
    assert all(type(sampler) is ReplayBufferAsync for sampler in async_samplers)
    assert all(sampler.filter_groups_metric == "acc" for sampler in async_samplers)
    assert all(sampler.train_batch_size is None for sampler in async_samplers)
    assert all(sampler.gen_batch_size is None for sampler in async_samplers)


def test_custom_sampler_skips_builtin_filter_groups_validation():
    trainer = _trainer_with_filter_groups({"enable": True, "metric": "acc"})
    trainer.config.trainer.v1.sampler.custom_sampler = {"path": "custom.py", "name": "CustomSampler"}

    with (
        patch("verl.trainer.ppo.v1.trainer_base.load_extern_type", return_value=_CustomSampler),
        patch.object(trainer, "_resolve_filter_groups_metric") as resolve_filter_groups_metric,
    ):
        sampler = trainer._build_replay_buffer()

    resolve_filter_groups_metric.assert_not_called()
    assert isinstance(sampler, _CustomSampler)
    assert "filter_groups_metric" not in sampler.kwargs
    assert "train_batch_size" not in sampler.kwargs
    assert "gen_batch_size" not in sampler.kwargs
    assert "max_inflight_gen_batches" not in sampler.kwargs
    assert "sync_refill_failed_groups" not in sampler.kwargs


def test_builtin_filter_groups_uses_default_inflight_limit():
    trainer = _trainer_with_filter_groups({"enable": True, "metric": "acc"})

    sampler = trainer._build_replay_buffer()

    assert sampler.filter_groups_metric == "acc"
    assert sampler.train_batch_size == 64
    assert sampler.gen_batch_size == 1
    assert sampler.max_inflight_gen_batches == 1


def test_builtin_filter_groups_forwards_configured_inflight_limit():
    trainer = _trainer_with_filter_groups({"enable": True, "metric": "acc", "max_inflight_gen_batches": 3})

    sampler = trainer._build_replay_buffer()

    assert sampler.max_inflight_gen_batches == 3


def test_builtin_sync_failure_refill_forces_single_prompt_generation():
    trainer = _trainer_with_filter_groups({"enable": False})
    trainer.config.trainer.v1.sampler.sync_refill_failed_groups = True

    sampler = trainer._build_replay_buffer()

    assert sampler.sync_refill_failed_groups is True
    assert sampler.gen_batch_size == 1


def test_sync_failure_refill_overrides_dataloader_generation_batch_size():
    trainer = _trainer_with_filter_groups({"enable": False})
    trainer.config.trainer.v1.sampler.sync_refill_failed_groups = True
    trainer.config.data.update(
        {
            "train_files": [],
            "val_files": [],
            "train_max_samples": -1,
            "val_max_samples": -1,
            "dataloader_num_workers": 0,
            "val_batch_size": 1,
            "validation_shuffle": False,
        }
    )
    trainer.config.trainer.total_epochs = 1
    trainer.config.trainer.total_training_steps = None
    trainer.parameter_sync_step = 1
    trainer.tokenizer = None
    trainer.processor = None

    with (
        patch("verl.trainer.ppo.v1.trainer_base.create_rl_dataset", side_effect=[[{}, {}], [{}]]),
        patch("verl.trainer.ppo.v1.trainer_base.create_rl_sampler", return_value=None),
        patch("verl.trainer.ppo.v1.trainer_base.StatefulDataLoader") as dataloader,
        patch("verl.trainer.ppo.v1.trainer_base.logger.warning") as warning,
    ):
        trainer._init_dataloader()

    assert trainer.config.data.gen_batch_size == 1
    assert dataloader.call_args_list[0].kwargs["batch_size"] == 1
    warning.assert_any_call("data.gen_batch_size=8 is overridden to 1.")


def test_builtin_filter_groups_warns_when_total_generation_limit_is_configured():
    trainer = _trainer_with_filter_groups({"enable": True, "metric": "acc", "max_num_gen_batches": 10})

    with patch("verl.trainer.ppo.v1.trainer_base.logger.warning") as warning:
        trainer._build_replay_buffer()

    warning.assert_called_once_with(
        "algorithm.filter_groups.max_num_gen_batches=%s is ignored by the built-in V1 ReplayBuffer; "
        "use max_inflight_gen_batches to bound concurrent Sync DAPO generation.",
        10,
    )


def _nested(rows: list[list[int] | list[float]], dtype: torch.dtype) -> torch.Tensor:
    return torch.nested.as_nested_tensor([torch.tensor(row, dtype=dtype) for row in rows], layout=torch.jagged)


def test_build_perception_sft_actor_batch_interleaves_gold_rows_and_aligns_masks():
    prompt_uids = ["uid-a", "uid-b", "uid-c", "uid-d"]
    keys = [f"{uid}_{session_id}_0" for session_id in range(6) for uid in prompt_uids]
    row_count = len(keys)

    rollout_responses = [[10 + index, 20 + index] for index in range(row_count)]
    gold_responses = [[100 + index, 200 + index, 300 + index] for index in range(row_count)]
    rollout_inputs = [[1, 2, *response] for response in rollout_responses]
    gold_inputs = [[1, 2, *response] for response in gold_responses]
    data = TensorDict(
        {
            "responses": _nested(rollout_responses, torch.int64),
            "input_ids": _nested(rollout_inputs, torch.int64),
            "attention_mask": _nested([[1] * 4 for _ in keys], torch.int64),
            "position_ids": _nested([list(range(4)) for _ in keys], torch.int64),
            "response_mask": _nested([[1, 1] for _ in keys], torch.int64),
            "loss_mask": _nested([[1, 1] for _ in keys], torch.int64),
            "grpo_loss_mask": _nested([[0, 1] for _ in keys], torch.int64),
            "grpo_advantage_scale": _nested([[1.0, 1.0] for _ in keys], torch.float32),
            "perception_sft_responses": _nested(gold_responses, torch.int64),
            "perception_sft_input_ids": _nested(gold_inputs, torch.int64),
            "perception_sft_attention_mask": _nested([[1] * 5 for _ in keys], torch.int64),
            "perception_sft_position_ids": _nested([list(range(5)) for _ in keys], torch.int64),
            "perception_sft_mask": _nested([[1, 1, 1] for _ in keys], torch.int64),
            "perception_sft_token_weight": _nested([[1.0, 2.0, 1.0] for _ in keys], torch.float32),
            "signal_aux_mask": _nested([[0, 1] for _ in keys], torch.int64),
            "signal_aux_advantage": _nested([[0.0, 0.75] for _ in keys], torch.float32),
            "reasoning_aux_mask": _nested([[1, 0] for _ in keys], torch.int64),
            "reasoning_aux_advantage": _nested([[0.5, 0.0] for _ in keys], torch.float32),
            "mode_aux_mask": _nested([[1, 0] for _ in keys], torch.int64),
            "mode_aux_weight": _nested([[0.8, 0.0] for _ in keys], torch.float32),
            "old_log_probs": _nested([[0.2, 0.3] for _ in keys], torch.float32),
            "advantages": _nested([[0.5, 0.5] for _ in keys], torch.float32),
            "ref_log_prob": _nested([[0.1, 0.1] for _ in keys], torch.float32),
            "metadata": NonTensorStack(*[{"row": index} for index in range(row_count)]),
            "multi_modal_inputs": NonTensorStack(*[{} for _ in keys]),
            "perception_sft_multi_modal_inputs": NonTensorStack(
                *[{"video": f"stage1-{index}"} for index in range(row_count)]
            ),
        },
        batch_size=[row_count],
    )
    tags = [{"seq_len": 4, "status": "success"} for _ in keys]
    batch = SimpleNamespace(keys=list(keys), partition_id="train", tags=tags)
    actor_batch = object()

    with (
        patch("verl.trainer.ppo.v1.trainer_base.tq.kv_batch_get", return_value=data),
        patch("verl.trainer.ppo.v1.trainer_base.tq.kv_batch_put", return_value=actor_batch) as kv_batch_put,
    ):
        result, gold_keys = _build_perception_sft_actor_batch(batch, rollout_n=6)

    assert result is actor_batch
    assert len(gold_keys) == 4
    assert batch.keys == keys
    assert "perception_sft_responses" in data

    put_kwargs = kv_batch_put.call_args.kwargs
    combined_keys = put_kwargs["keys"]
    combined_tags = put_kwargs["tags"]
    combined_data = put_kwargs["fields"]
    assert len(combined_keys) == 28
    assert len(combined_data) == 28

    for group_index, uid in enumerate(prompt_uids):
        start = group_index * 7
        assert combined_keys[start : start + 6] == [f"{uid}_{session_id}_0" for session_id in range(6)]
        assert combined_keys[start + 6].startswith(f"{uid}_0_0_perception_sft_")
        assert combined_tags[start + 6]["seq_len"] == 5
        assert combined_tags[start + 6]["is_perception_sft"] is True

        for rollout_index in range(start, start + 6):
            assert combined_data["grpo_loss_mask"][rollout_index].tolist() == [0, 1]
            assert combined_data["grpo_advantage_scale"][rollout_index].tolist() == [1.0, 1.0]
            assert combined_data["perception_sft_mask"][rollout_index].tolist() == [0, 0]
            assert combined_data["perception_sft_token_weight"][rollout_index].tolist() == [0.0, 0.0]
            assert combined_data["kl_loss_mask"][rollout_index].tolist() == [1, 1]
            assert combined_data["signal_aux_mask"][rollout_index].tolist() == [0, 1]
            assert combined_data["signal_aux_advantage"][rollout_index].tolist() == [0.0, 0.75]
            assert combined_data["reasoning_aux_mask"][rollout_index].tolist() == [1, 0]
            assert combined_data["reasoning_aux_advantage"][rollout_index].tolist() == [0.5, 0.0]
            assert combined_data["mode_aux_mask"][rollout_index].tolist() == [1, 0]
            assert torch.allclose(
                combined_data["mode_aux_weight"][rollout_index], torch.tensor([0.8, 0.0])
            )
            assert combined_data["multi_modal_inputs"][rollout_index].data == {}

        gold_index = start + 6
        assert combined_data["grpo_loss_mask"][gold_index].tolist() == [0, 0, 0]
        assert combined_data["grpo_advantage_scale"][gold_index].tolist() == [0.0, 0.0, 0.0]
        assert combined_data["perception_sft_mask"][gold_index].tolist() == [1, 1, 1]
        assert combined_data["perception_sft_token_weight"][gold_index].tolist() == [1.0, 2.0, 1.0]
        assert combined_data["kl_loss_mask"][gold_index].tolist() == [0, 0, 0]
        assert combined_data["loss_mask"][gold_index].tolist() == [1, 1, 1]
        assert combined_data["old_log_probs"][gold_index].tolist() == [0.0, 0.0, 0.0]
        assert combined_data["signal_aux_mask"][gold_index].tolist() == [0, 0, 0]
        assert combined_data["signal_aux_advantage"][gold_index].tolist() == [0.0, 0.0, 0.0]
        assert combined_data["reasoning_aux_mask"][gold_index].tolist() == [0, 0, 0]
        assert combined_data["reasoning_aux_advantage"][gold_index].tolist() == [0.0, 0.0, 0.0]
        assert combined_data["mode_aux_mask"][gold_index].tolist() == [0, 0, 0]
        assert combined_data["mode_aux_weight"][gold_index].tolist() == [0.0, 0.0, 0.0]
        assert combined_data["metadata"][gold_index].data == {"row": group_index}
        assert combined_data["multi_modal_inputs"][gold_index].data == {
            "video": f"stage1-{group_index}"
        }

    for source_key in (
        "perception_sft_responses",
        "perception_sft_input_ids",
        "perception_sft_attention_mask",
        "perception_sft_position_ids",
        "perception_sft_multi_modal_inputs",
    ):
        assert source_key not in combined_data


def test_padding_template_replaces_perception_gold_with_inert_text_row():
    source = {
        "prompts": torch.tensor([1, 2]),
        "responses": torch.tensor([3, 4]),
        "input_ids": torch.tensor([1, 2, 3, 4]),
        "attention_mask": torch.ones(4, dtype=torch.int64),
        "position_ids": torch.arange(4),
        "response_mask": torch.ones(2, dtype=torch.int64),
        "grpo_loss_mask": torch.ones(2, dtype=torch.int64),
        "grpo_advantage_scale": torch.ones(2, dtype=torch.float32),
        "perception_sft_responses": torch.tensor([5, 6, 7]),
        "perception_sft_input_ids": torch.tensor([1, 2, 5, 6, 7]),
        "perception_sft_attention_mask": torch.ones(5, dtype=torch.int64),
        "perception_sft_position_ids": torch.arange(5),
        "perception_sft_mask": torch.ones(3, dtype=torch.int64),
        "perception_sft_token_weight": torch.tensor([1.0, 2.0, 1.0]),
        "signal_aux_mask": torch.tensor([0, 1]),
        "signal_aux_advantage": torch.tensor([0.0, 0.75]),
        "reasoning_aux_mask": torch.tensor([1, 0]),
        "reasoning_aux_advantage": torch.tensor([0.5, 0.0]),
        "mode_aux_mask": torch.tensor([1, 0]),
        "mode_aux_weight": torch.tensor([0.8, 0.0]),
        "multi_modal_inputs": {"video_grid_thw": torch.tensor([[1, 2, 2]])},
    }

    sample, tag = construct_minimal_padding_template(source, {"seq_len": 4}, eos_token_id=99)

    assert sample["perception_sft_responses"].tolist() == [99]
    assert sample["perception_sft_input_ids"].tolist() == [99, 99]
    assert sample["perception_sft_attention_mask"].tolist() == [1, 1]
    assert sample["perception_sft_position_ids"].tolist() == [0, 1]
    assert sample["perception_sft_mask"].tolist() == [0]
    assert sample["perception_sft_token_weight"].tolist() == [0.0]
    assert sample["grpo_loss_mask"].tolist() == [0]
    assert sample["grpo_advantage_scale"].tolist() == [0.0]
    assert sample["signal_aux_mask"].tolist() == [0]
    assert sample["signal_aux_advantage"].tolist() == [0.0]
    assert sample["reasoning_aux_mask"].tolist() == [0]
    assert sample["reasoning_aux_advantage"].tolist() == [0.0]
    assert sample["mode_aux_mask"].tolist() == [0]
    assert sample["mode_aux_weight"].tolist() == [0.0]
    assert sample["multi_modal_inputs"] == {}
    assert tag["is_padding"] is True
