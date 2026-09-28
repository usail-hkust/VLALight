"""Regression tests for the online FAST/SLOW binary mode objective."""

from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("tensordict")

from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.workers.utils.losses import ppo_loss


def _nested(rows):
    return torch.nested.as_nested_tensor([torch.tensor(row, dtype=torch.float32) for row in rows], layout=torch.jagged)


def _config():
    return SimpleNamespace(
        global_batch_info={},
        loss_scale_factor=None,
        loss_agg_mode="token-mean",
        entropy_coeff=0.0,
        use_kl_loss=False,
        clip_ratio=0.2,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        policy_loss={"loss_mode": "vanilla"},
        get=lambda key, default=None: {"clip_ratio_c": 3.0}.get(key, default),
        kl_loss_type="low_var_kl",
    )


def test_binary_mode_loss_uses_two_class_logits_and_has_finite_gradient(monkeypatch):
    monkeypatch.setenv("V35_MODE_SELECTOR_COEF", "0.02")
    monkeypatch.setenv("V35_PERCEPTION_SFT_COEF", "0")
    monkeypatch.setenv("V35_SIGNAL_AUX_COEF", "0")

    batch_size, prompt_len, response_len = 2, 3, 2
    prompts = torch.tensor([[1, 2, 3], [1, 2, 3]], dtype=torch.long)
    responses = torch.tensor([[10, 11], [10, 11]], dtype=torch.long)
    attention_mask = torch.ones(batch_size, prompt_len + response_len, dtype=torch.long)
    response_mask = torch.ones(batch_size, response_len, dtype=torch.float32)
    td = TensorDict(
        {
            "prompts": prompts,
            "responses": responses,
            "attention_mask": attention_mask,
            "response_mask": response_mask,
            "old_log_probs": torch.zeros(batch_size, response_len),
            "advantages": torch.zeros(batch_size, response_len),
            "grpo_loss_mask": torch.zeros(batch_size, response_len),
            "kl_loss_mask": torch.zeros(batch_size, response_len),
            "reasoning_aux_mask": torch.zeros(batch_size, response_len),
            "signal_aux_mask": torch.zeros(batch_size, response_len),
            "mode_aux_mask": torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
            "mode_binary_mask": torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
            "mode_token_index": torch.zeros(batch_size, dtype=torch.long),
            "mode_fast_token_id": torch.full((batch_size,), 10, dtype=torch.long),
            "mode_slow_token_id": torch.full((batch_size,), 11, dtype=torch.long),
            "mode_binary_target": torch.tensor([0.25, 0.75]),
            "mode_binary_valid": torch.ones(batch_size, dtype=torch.bool),
            "mode_binary_weight": torch.ones(batch_size),
            "grpo_advantage_scale": torch.ones(batch_size, response_len),
        },
        batch_size=[batch_size],
    )
    for key, count in (
        ("grpo_loss_mask", 0),
        ("kl_loss_mask", 0),
        ("reasoning_aux_mask", 0),
        ("signal_aux_mask", 0),
        ("mode_aux_mask", 2),
        ("mode_binary_mask", 2),
    ):
        tu.assign_non_tensor_data(td, f"{key}_num_tokens", count)
        tu.assign_non_tensor_data(td, f"{key}_num_sequences", 0 if count == 0 else batch_size)
    tu.assign_non_tensor_data(td, "dp_size", 1)
    tu.assign_non_tensor_data(td, "batch_num_tokens", 4)
    tu.assign_non_tensor_data(td, "global_batch_size", batch_size)

    log_probs = torch.nested.as_nested_tensor(
        [torch.zeros(prompt_len + response_len, requires_grad=True) for _ in range(batch_size)],
        layout=torch.jagged,
    )
    binary_logits = torch.tensor([[0.0, 1.0], [1.0, 0.0]], requires_grad=True)
    loss, metrics = ppo_loss(_config(), {"log_probs": log_probs, "mode_binary_logits": binary_logits}, td)

    assert torch.isfinite(loss)
    assert metrics["actor/mode_binary_enabled"] == 1.0
    assert metrics["actor/mode_binary_valid_rows"].aggregate() == batch_size
    loss.backward()
    assert torch.isfinite(binary_logits.grad).all()
    assert binary_logits.grad.abs().sum() > 0
