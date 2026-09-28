# Copyright 2025 Bytedance Ltd. and/or its affiliates
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


import os

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from verl.trainer.ppo.core_algos import agg_loss, compute_value_loss, get_policy_loss_fn, kl_penalty
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.metric import AggregationType, Metric
from verl.utils.torch_functional import masked_mean, masked_sum
from verl.workers.config import ActorConfig, CriticConfig
from verl.workers.utils.padding import no_padding_2_padding
def sft_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    pad_mode = tu.get_non_tensor_data(data=data, key="pad_mode", default=DatasetPadMode.NO_PADDING)
    dp_size = data["dp_size"]
    batch_num_tokens = data["batch_num_tokens"]

    log_prob = model_output["log_probs"]

    if pad_mode == DatasetPadMode.NO_PADDING:
        # log_prob and loss mask are nested tensors of shape [bsz, j1]
        # for each sample, loss mask shape is [1, prompt_length + response_length]
        loss_mask = data["loss_mask"]

        log_prob_flatten = log_prob.values()
        loss_mask_flatten = loss_mask.values()

        # left-shift the loss mask by one token to align with log_prob
        loss_mask_flatten = torch.roll(loss_mask_flatten, shifts=-1, dims=0)

        # NOTE: loss is averaged over all tokens in the batch across all data parallel groups,
        # For FSDP backend, the loss is directly used for backward; while for Megatron backend,
        # the loss should be scaled by `num_microbatches` for pp schedule.
        loss = -masked_sum(log_prob_flatten, loss_mask_flatten) / batch_num_tokens * dp_size
    else:
        response_mask = data["response_mask"].to(bool)
        loss = -masked_sum(log_prob, response_mask) / batch_num_tokens * dp_size

    return loss, {}


def ppo_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    """Computes ppo loss from model output (log_prob, entropy, values, etc. ) and old_log_probs from data."""
    log_prob = no_padding_2_padding(model_output["log_probs"], data)
    entropy = model_output.get("entropy", None)
    if entropy is not None:
        entropy = no_padding_2_padding(entropy, data)

    # global batch info for loss aggregation
    default_global_batch_info = {
        "dp_size": data["dp_size"],
        "batch_num_tokens": data["batch_num_tokens"],
        "global_batch_size": data["global_batch_size"],
        "loss_scale_factor": config.loss_scale_factor,
    }
    config.global_batch_info.update(default_global_batch_info)

    # assumes that if any of the global batch info is set, the policy_loss_fn will
    # normalize using dp_size/global_bsz/global_token; in this case, metric aggregation should be SUM
    # to reflect the mean loss over the global batch
    if (
        data["dp_size"] > 1
        or data["batch_num_tokens"] is not None
        or data["global_batch_size"] is not None
        or config.loss_scale_factor is not None
    ):
        metric_aggregation = AggregationType.SUM
    else:
        metric_aggregation = AggregationType.MEAN

    metrics = {}

    # select fields and convert to padded tensor
    perception_sft_coef = float(os.environ.get("V35_PERCEPTION_SFT_COEF", "0"))
    signal_aux_coef = float(os.environ.get("V35_SIGNAL_AUX_COEF", "0"))
    mode_selector_coef = float(os.environ.get("V35_MODE_SELECTOR_COEF", "0"))
    segmented_loss = any(coef > 0.0 for coef in (perception_sft_coef, signal_aux_coef, mode_selector_coef))
    fields = ["response_mask", "old_log_probs", "advantages"]
    if segmented_loss:
        required_masks = ["grpo_loss_mask", "kl_loss_mask"]
        auxiliary_values = ["grpo_advantage_scale"]
        if perception_sft_coef > 0.0:
            required_masks.append("perception_sft_mask")
            auxiliary_values.append("perception_sft_token_weight")
        # Reasoning and signal are separate decision segments. They share
        # the PPO implementation but must consume their own advantages.
        # The online adapter supplies these fields even when a coefficient is
        # zero, allowing the base PPO path to remain correctly segmented.
        required_masks.extend(["reasoning_aux_mask", "signal_aux_mask"])
        auxiliary_values.extend(["reasoning_aux_advantage", "signal_aux_advantage"])
        if mode_selector_coef > 0.0:
            required_masks.append("mode_aux_mask")
            auxiliary_values.append("mode_aux_weight")
            binary_mode_fields = [
                "mode_binary_mask",
                "mode_token_index",
                "mode_fast_token_id",
                "mode_slow_token_id",
                "mode_binary_target",
                "mode_binary_valid",
                "mode_binary_weight",
            ]
            if all(key in data for key in binary_mode_fields):
                required_masks.append("mode_binary_mask")
                auxiliary_values.extend(binary_mode_fields[1:])
        missing_masks = [key for key in required_masks if key not in data]
        missing_values = [key for key in auxiliary_values if key not in data]
        # Auxiliary advantages may be absent on SFT-only or legacy micro
        # batches. Masks remain mandatory, while a missing advantage is a
        # zero-gradient branch rather than a fatal actor update error.
        if missing_masks:
            raise ValueError(
                f"segmented actor loss is missing masks={missing_masks}, values={missing_values}"
            )
        segmented_normalizers = {}
        for mask_key in required_masks:
            for suffix in ("num_tokens", "num_sequences"):
                normalizer_key = f"{mask_key}_{suffix}"
                value = tu.get_non_tensor_data(data, normalizer_key, None)
                if value is None:
                    raise ValueError(f"actor batch is missing segmented normalizer {normalizer_key}")
                segmented_normalizers[normalizer_key] = value
        fields.extend([*required_masks, *[v for v in auxiliary_values if v in data]])
    if "rollout_is_weights" in data:
        fields.append("rollout_is_weights")
    if "ref_log_prob" in data:
        fields.append("ref_log_prob")
    data = data.select(*fields).to_padded_tensor()

    # Row-level binary metadata is dense [B] tensors.  ``to_padded_tensor``
    # may leave them as [B, 1] or (for a one-row jagged round-trip) scalars;
    # normalize to a stable shape before the classifier branch and keep token
    # masks as [B, R].
    for key in ("mode_token_index", "mode_fast_token_id", "mode_slow_token_id", "mode_binary_target", "mode_binary_valid", "mode_binary_weight"):
        if key in data and data[key].ndim > 1:
            data[key] = data[key].reshape(data.shape[0], -1)[:, 0]

    response_mask = data["response_mask"].to(bool)
    grpo_loss_mask = data["grpo_loss_mask"].to(bool) if segmented_loss else response_mask
    perception_sft_mask = (
        data["perception_sft_mask"].to(bool) if perception_sft_coef > 0.0 else None
    )
    signal_aux_mask = data["signal_aux_mask"].to(bool) if segmented_loss else None
    reasoning_aux_mask = data["reasoning_aux_mask"].to(bool) if segmented_loss else None
    reasoning_aux_advantage = data.get("reasoning_aux_advantage", torch.zeros_like(data["advantages"]))
    signal_aux_advantage = data.get("signal_aux_advantage", torch.zeros_like(data["advantages"]))
    mode_aux_mask = data["mode_aux_mask"].to(bool) if mode_selector_coef > 0.0 else None
    mode_binary_available = mode_selector_coef > 0.0 and all(
        key in data
        for key in (
            "mode_binary_mask",
            "mode_token_index",
            "mode_fast_token_id",
            "mode_slow_token_id",
            "mode_binary_target",
            "mode_binary_valid",
            "mode_binary_weight",
        )
    )
    mode_binary_mask = data["mode_binary_mask"].to(bool) if mode_binary_available else None
    mode_binary_target = data.get("mode_binary_target", None)
    mode_binary_valid = data.get("mode_binary_valid", None)
    mode_binary_weight = data.get("mode_binary_weight", None)
    kl_loss_mask = data["kl_loss_mask"].to(bool) if segmented_loss else response_mask
    # compute policy loss
    old_log_prob = data["old_log_probs"]
    advantages = data["advantages"]
    if segmented_loss:
        # Perception-only failures are corrected by gold CE. Do not propagate
        # their scalar format penalty into the decision suffix.
        advantages = advantages * data["grpo_advantage_scale"].detach()
        # The base PPO branch is the sole decision update.  Construct a
        # disjoint per-token advantage so reasoning and signal are not
        # updated twice by an auxiliary branch.
        if reasoning_aux_mask is not None and signal_aux_mask is not None:
            # A stale worker or tokenizer boundary can still produce an
            # overlap after the adapter's row-level disjointing.  Do not
            # abort the whole Ray update for one such micro-batch: signal is
            # the authoritative suffix, so remove those positions from the
            # reasoning branch and keep a finite, auditable update.
            mask_overlap = reasoning_aux_mask & signal_aux_mask
            overlap_count = int(mask_overlap.sum().item())
            if overlap_count:
                reasoning_aux_mask = reasoning_aux_mask & ~signal_aux_mask
                metrics["actor/diag_mask_overlap_repaired"] = Metric(
                    value=overlap_count, aggregation=AggregationType.SUM
                )
            reasoning_advantage = reasoning_aux_advantage
            signal_advantage = signal_aux_advantage
            decision_mask = reasoning_aux_mask | signal_aux_mask
            advantages = torch.where(reasoning_aux_mask, reasoning_advantage.detach(), advantages)
            advantages = torch.where(signal_aux_mask, signal_advantage.detach(), advantages)
            grpo_loss_mask = decision_mask
    # Keep branch-level diagnostics independent of the optional auxiliary
    # coefficients. This makes a zero decision loss distinguishable from a
    # missing mask, an all-zero advantage, or an empty decision span.
    advantage_finite = torch.isfinite(advantages)
    reasoning_adv_finite = torch.isfinite(reasoning_aux_advantage)
    signal_adv_finite = torch.isfinite(signal_aux_advantage)
    rollout_is_weights = data.get("rollout_is_weights", None)

    loss_agg_mode = config.loss_agg_mode

    loss_mode = config.policy_loss.get("loss_mode", "vanilla")

    policy_loss_fn = get_policy_loss_fn(loss_mode)
    if segmented_loss:
        grpo_global_batch_info = {
            **default_global_batch_info,
            "batch_num_tokens": segmented_normalizers["grpo_loss_mask_num_tokens"],
            "global_batch_size": segmented_normalizers["grpo_loss_mask_num_sequences"],
        }
        config.global_batch_info.update(grpo_global_batch_info)
    pg_loss, pg_metrics = policy_loss_fn(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=grpo_loss_mask,
        loss_agg_mode=loss_agg_mode,
        config=config,
        rollout_is_weights=rollout_is_weights,
    )
    config.global_batch_info.update(default_global_batch_info)

    # AggregationType.MEAN for pg metrics: assumes policy_loss_fn normalizes by local_bsz/local_tokens
    # Ex: in compute_policy_loss_vanilla, pg_metrics are pg_clipfrac, ppo_kl, pg_clipfrac_lower
    pg_metrics = Metric.from_dict(pg_metrics, aggregation=AggregationType.MEAN)

    metrics.update(pg_metrics)
    metrics["actor/pg_loss"] = Metric(value=pg_loss, aggregation=metric_aggregation)
    policy_loss = pg_loss

    # add entropy loss
    if entropy is not None:
        entropy_global_batch_info = grpo_global_batch_info if segmented_loss else default_global_batch_info
        entropy_loss = agg_loss(
            loss_mat=entropy,
            loss_mask=grpo_loss_mask,
            loss_agg_mode=loss_agg_mode,
            **entropy_global_batch_info,
        )
        entropy_coeff = config.entropy_coeff
        policy_loss -= entropy_coeff * entropy_loss
        metrics["actor/entropy_loss"] = Metric(value=entropy_loss, aggregation=metric_aggregation)

    # add kl loss
    if config.use_kl_loss:
        ref_log_prob = data["ref_log_prob"]
        # compute kl loss
        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=config.kl_loss_type)
        kl_global_batch_info = default_global_batch_info
        if segmented_loss:
            kl_global_batch_info = {
                **default_global_batch_info,
                "batch_num_tokens": segmented_normalizers["kl_loss_mask_num_tokens"],
                "global_batch_size": segmented_normalizers["kl_loss_mask_num_sequences"],
            }
        kl_loss = agg_loss(
            loss_mat=kld,
            loss_mask=kl_loss_mask,
            loss_agg_mode=config.loss_agg_mode,
            **kl_global_batch_info,
        )

        policy_loss += kl_loss * config.kl_loss_coef
        metrics["kl_loss"] = Metric(value=kl_loss, aggregation=metric_aggregation)
        metrics["kl_coef"] = config.kl_loss_coef

    if segmented_loss:
        if perception_sft_coef > 0.0:
            perception_token_count = segmented_normalizers["perception_sft_mask_num_tokens"]
            # Recompute from the mini-batch tensors.  Non-tensor metadata is
            # not guaranteed to survive FSDP micro-batch slicing.
            perception_weight_sum = float(
                (data["perception_sft_token_weight"].detach() * perception_sft_mask.to(data["perception_sft_token_weight"].dtype)).sum().item()
                * default_global_batch_info["dp_size"]
            )
            if perception_weight_sum > 0.0 and perception_token_count > 0:
                perception_sft_loss = agg_loss(
                    loss_mat=-log_prob * data["perception_sft_token_weight"].detach(),
                    loss_mask=perception_sft_mask,
                    loss_agg_mode="token-mean",
                    dp_size=default_global_batch_info["dp_size"],
                    batch_num_tokens=perception_weight_sum,
                )
            else:
                # FSDP can produce a local micro-batch containing only RL
                # rows; its local supervised contribution is exactly zero.
                perception_sft_loss = torch.zeros_like(pg_loss)
            perception_weighted_contribution = perception_sft_coef * perception_sft_loss
            policy_loss += perception_weighted_contribution
            metrics["actor/perception_sft_loss"] = Metric(
                value=perception_sft_loss,
                aggregation=metric_aggregation,
            )
            metrics["actor/perception_sft_weighted_contribution"] = Metric(
                value=perception_weighted_contribution,
                aggregation=metric_aggregation,
            )
            metrics["actor/perception_sft_coef"] = perception_sft_coef
            metrics["actor/perception_sft_tokens"] = Metric(
                value=perception_sft_mask.sum() * default_global_batch_info["dp_size"],
                aggregation=AggregationType.SUM,
            )
            metrics["actor/perception_sft_weight_sum"] = Metric(
                value=perception_weight_sum,
                aggregation=AggregationType.SUM,
            )
            metrics["actor/perception_sft_mean_token_weight"] = Metric(
                value=(float(perception_weight_sum) / float(perception_token_count))
                if perception_token_count > 0
                else 0.0,
                aggregation=AggregationType.MEAN,
            )

        if signal_aux_coef > 0.0:
            metrics["actor/signal_aux_coef"] = signal_aux_coef
            metrics["actor/signal_aux_advantage"] = Metric(
                value=masked_mean(signal_aux_advantage, signal_aux_mask), aggregation=metric_aggregation)
        if reasoning_aux_mask is not None:
            metrics["actor/reasoning_aux_advantage"] = Metric(
                value=masked_mean(reasoning_aux_advantage, reasoning_aux_mask), aggregation=metric_aggregation)

        if mode_selector_coef > 0.0:
            mode_token_count = segmented_normalizers["mode_aux_mask_num_tokens"]
            binary_logits = model_output.get("mode_binary_logits") if mode_binary_available else None
            use_binary_mode = (
                mode_binary_available
                and torch.is_tensor(binary_logits)
                and binary_logits.ndim == 2
                and binary_logits.shape[-1] == 2
            )
            if mode_binary_available and not use_binary_mode:
                # Current online rows always carry the binary metadata.  A
                # missing/incorrectly shaped two-column output means the
                # selected engine failed to implement the classifier path;
                # silently reverting to the old full-vocabulary CE would make
                # the run look healthy while training the wrong objective.
                raise RuntimeError(
                    "mode binary metadata is present, but the actor engine did not return "
                    "mode_binary_logits with shape [batch, 2]; refusing legacy mode CE fallback"
                )
            if use_binary_mode:
                # This is a genuine FAST/SLOW classifier.  The target is the
                # group-level GDPO p(slow), while the positive weight balances
                # the forced FAST/SLOW candidates in the same decision group.
                binary_logits = binary_logits.to(dtype=log_prob.dtype)
                target = mode_binary_target.reshape(-1).to(device=binary_logits.device, dtype=binary_logits.dtype)
                valid_rows = mode_binary_valid.reshape(-1).to(device=binary_logits.device, dtype=torch.bool)
                weights = mode_binary_weight.reshape(-1).to(device=binary_logits.device, dtype=binary_logits.dtype)
                valid_rows = valid_rows & torch.isfinite(binary_logits).all(dim=-1)
                valid_rows = valid_rows & torch.isfinite(target) & torch.isfinite(weights)
                # Keep diagnostics finite even for invalid rows that are
                # masked out of the loss.  ``nan * 0`` is still NaN in PyTorch,
                # so sanitize before computing probabilities/means.
                binary_logits = torch.nan_to_num(binary_logits, nan=0.0, posinf=0.0, neginf=0.0)
                target = torch.nan_to_num(target, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
                weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
                binary_log_probs = F.log_softmax(binary_logits, dim=-1)
                binary_ce = -((1.0 - target) * binary_log_probs[:, 0] + target * binary_log_probs[:, 1])
                # The classifier emits one loss per row (two class logits),
                # while mode_binary_mask is a response-token mask. Reduce the
                # latter to one row flag before agg_loss so CE is not
                # broadcast across the entire response width.
                binary_row_valid = valid_rows & mode_binary_mask.any(dim=-1)
                binary_loss_mask = binary_row_valid.unsqueeze(-1)
                binary_loss_mat = binary_ce.unsqueeze(-1) * weights.unsqueeze(-1)
                binary_token_count = int(
                    segmented_normalizers.get(
                        "mode_binary_mask_num_tokens", int(binary_loss_mask.sum().item())
                    )
                )
                if binary_token_count <= 0 or not bool(binary_row_valid.any().item()):
                    mode_selector_raw_loss = binary_logits.sum() * 0.0
                    binary_valid_rows_count = binary_logits.new_zeros(())
                    target_slow_mean = binary_logits.new_zeros(())
                    prob_fast_mean = binary_logits.new_zeros(())
                    prob_slow_mean = binary_logits.new_zeros(())
                    mode_binary_kl = binary_logits.sum() * 0.0
                else:
                    mode_selector_raw_loss = agg_loss(
                        loss_mat=binary_loss_mat,
                        loss_mask=binary_loss_mask,
                        loss_agg_mode="token-mean",
                        dp_size=default_global_batch_info["dp_size"],
                        batch_num_tokens=binary_token_count,
                    )
                    valid_float = binary_row_valid.to(binary_logits.dtype)
                    valid_denom = valid_float.sum().clamp_min(1.0)
                    probs = binary_log_probs.exp()
                    binary_valid_rows_count = valid_float.sum()
                    target_slow_mean = (target * valid_float).sum() / valid_denom
                    prob_fast_mean = (probs[:, 0] * valid_float).sum() / valid_denom
                    prob_slow_mean = (probs[:, 1] * valid_float).sum() / valid_denom
                    target_probs = torch.stack((1.0 - target, target), dim=-1).clamp_min(1e-8)
                    mode_binary_kl = (
                        (probs * (binary_log_probs - target_probs.log())).sum(dim=-1) * valid_float
                    ).sum() / valid_denom
                mode_trust_kl = binary_logits.sum() * 0.0
                max_weighted = float(os.environ.get("V35_MODE_MAX_WEIGHTED_LOSS", "0.10"))
                if max_weighted <= 0.0:
                    raise ValueError("V35_MODE_MAX_WEIGHTED_LOSS must be positive")
                raw_limit = max_weighted / mode_selector_coef
                raw_magnitude = float(mode_selector_raw_loss.detach().abs().item())
                mode_loss_scale = min(1.0, raw_limit / max(raw_magnitude, 1e-12))
                mode_total_loss_scale = 1.0
                mode_selector_loss = mode_selector_raw_loss * mode_loss_scale
                metrics["actor/mode_binary_loss"] = Metric(value=mode_selector_loss, aggregation=metric_aggregation)
                metrics["actor/mode_binary_valid_rows"] = Metric(value=binary_valid_rows_count, aggregation=AggregationType.SUM)
                metrics["actor/mode_binary_valid_ratio"] = Metric(
                    value=binary_valid_rows_count / max(float(binary_row_valid.numel()), 1.0), aggregation=AggregationType.MEAN
                )
                metrics["actor/mode_binary_target_slow_mean"] = Metric(value=target_slow_mean, aggregation=AggregationType.MEAN)
                metrics["actor/mode_binary_prob_fast_mean"] = Metric(value=prob_fast_mean, aggregation=AggregationType.MEAN)
                metrics["actor/mode_binary_prob_slow_mean"] = Metric(value=prob_slow_mean, aggregation=AggregationType.MEAN)
                metrics["actor/mode_binary_kl"] = Metric(value=mode_binary_kl, aggregation=metric_aggregation)
            elif mode_token_count <= 0:
                mode_selector_raw_loss = log_prob.sum() * 0.0
                mode_trust_kl = log_prob.sum() * 0.0
                mode_loss_scale = 1.0
                mode_total_loss_scale = 1.0
                mode_selector_loss = mode_selector_raw_loss
            else:
                # Compatibility path for archived batches produced before the
                # binary mode fields were introduced.
                mode_selector_raw_loss = agg_loss(
                    loss_mat=-log_prob * data["mode_aux_weight"].detach(),
                    loss_mask=mode_aux_mask,
                    loss_agg_mode="token-mean",
                    dp_size=default_global_batch_info["dp_size"],
                    batch_num_tokens=mode_token_count,
                )
                mode_trust_kl = log_prob.sum() * 0.0
                max_weighted = float(os.environ.get("V35_MODE_MAX_WEIGHTED_LOSS", "0.10"))
                raw_limit = max_weighted / mode_selector_coef
                raw_magnitude = float(mode_selector_raw_loss.detach().abs().item())
                mode_loss_scale = min(1.0, raw_limit / max(raw_magnitude, 1e-12))
                mode_total_loss_scale = 1.0
                mode_selector_loss = mode_selector_raw_loss * mode_loss_scale

            # Common diagnostics for both the binary and legacy fallback.
            policy_loss += mode_selector_coef * mode_selector_loss
            metrics["actor/mode_selector_loss"] = Metric(value=mode_selector_loss, aggregation=metric_aggregation)
            metrics["actor/mode_selector_raw_loss"] = Metric(value=mode_selector_raw_loss, aggregation=metric_aggregation)
            metrics["actor/mode_selector_loss_scale"] = mode_loss_scale
            metrics["actor/mode_selector_total_loss_scale"] = mode_total_loss_scale
            metrics["actor/mode_selector_trust_kl"] = Metric(value=mode_trust_kl, aggregation=metric_aggregation)
            metrics["actor/mode_selector_weighted_contribution"] = Metric(
                value=mode_selector_coef * mode_selector_loss, aggregation=metric_aggregation
            )
            metrics["actor/mode_selector_coef"] = mode_selector_coef
            metrics["actor/mode_binary_enabled"] = float(use_binary_mode)
            metrics["actor/mode_selector_weight"] = Metric(
                value=(masked_mean(data["mode_aux_weight"], mode_aux_mask) if mode_token_count > 0 else mode_selector_loss.detach()),
                aggregation=metric_aggregation,
            )

    # Lightweight branch diagnostics for postmortem analysis.  Keep these as
    # scalar metrics so distributed workers can aggregate them safely.
    metrics["actor/diag_loss_finite"] = Metric(
        value=torch.isfinite(policy_loss.detach()).to(torch.float32),
        aggregation=AggregationType.MEAN,
    )
    metrics["actor/diag_grpo_tokens"] = Metric(
        value=grpo_loss_mask.sum(), aggregation=AggregationType.SUM
    )
    metrics["actor/diag_advantage_finite"] = Metric(
        value=advantage_finite.to(torch.float32).mean(), aggregation=AggregationType.MEAN
    )
    metrics["actor/diag_advantage_nonzero_tokens"] = Metric(
        value=((advantages.abs() > 1e-8) & grpo_loss_mask).sum(), aggregation=AggregationType.SUM
    )
    if segmented_loss:
        # signal_aux_mask belongs to an optional supervised auxiliary branch.
        # The decision policy loss itself is always masked by grpo_loss_mask.
        decision_diag_mask = (reasoning_aux_mask | signal_aux_mask) if reasoning_aux_mask is not None and signal_aux_mask is not None else grpo_loss_mask
        metrics["actor/diag_decision_tokens"] = Metric(
            value=decision_diag_mask.sum(), aggregation=AggregationType.SUM
        )
        metrics["actor/diag_reasoning_tokens"] = Metric(
            value=reasoning_aux_mask.sum(), aggregation=AggregationType.SUM
        )
        metrics["actor/diag_signal_tokens"] = Metric(
            value=signal_aux_mask.sum(), aggregation=AggregationType.SUM
        )
        metrics["actor/diag_reasoning_advantage_finite"] = Metric(
            value=reasoning_adv_finite.to(torch.float32).mean(), aggregation=AggregationType.MEAN
        )
        metrics["actor/diag_signal_advantage_finite"] = Metric(
            value=signal_adv_finite.to(torch.float32).mean(), aggregation=AggregationType.MEAN
        )
        metrics["actor/diag_reasoning_advantage_nonzero_tokens"] = Metric(
            value=((reasoning_aux_advantage.abs() > 1e-8) & reasoning_aux_mask).sum(),
            aggregation=AggregationType.SUM,
        )
        metrics["actor/diag_signal_advantage_nonzero_tokens"] = Metric(
            value=((signal_aux_advantage.abs() > 1e-8) & signal_aux_mask).sum(),
            aggregation=AggregationType.SUM,
        )
        mode_diag_mask = mode_aux_mask if mode_aux_mask is not None else torch.zeros_like(decision_diag_mask)
        metrics["actor/diag_mode_tokens"] = Metric(
            value=mode_diag_mask.sum(), aggregation=AggregationType.SUM
        )
        metrics["actor/diag_mask_overlap"] = Metric(
            value=((reasoning_aux_mask & signal_aux_mask).sum() if reasoning_aux_mask is not None and signal_aux_mask is not None else (decision_diag_mask.bool() & mode_diag_mask.bool()).sum()),
            aggregation=AggregationType.SUM,
        )
        metrics["actor/diag_signal_loss_finite"] = Metric(
            value=torch.isfinite(signal_aux_advantage.detach()).to(torch.float32).mean()
            if signal_aux_coef > 0.0 and signal_aux_mask is not None
            else torch.ones((), device=policy_loss.device),
            aggregation=AggregationType.MEAN,
        )
        metrics["actor/diag_mode_loss_finite"] = Metric(
            value=torch.isfinite(mode_selector_loss.detach()).to(torch.float32)
            if mode_selector_coef > 0.0 else torch.ones((), device=policy_loss.device),
            aggregation=AggregationType.MEAN,
        )
        mode_weight = data.get("mode_aux_weight", torch.zeros_like(log_prob))
        mode_weight_masked = mode_weight.masked_select(mode_diag_mask)
        metrics["actor/diag_mode_weight_finite"] = Metric(
            value=torch.isfinite(mode_weight).to(torch.float32).mean(), aggregation=AggregationType.MEAN
        )
        metrics["actor/diag_mode_weight_nonzero_tokens"] = Metric(
            value=(mode_weight.abs() > 1e-8).logical_and(mode_diag_mask).sum(), aggregation=AggregationType.SUM
        )
        if mode_weight_masked.numel():
            metrics["actor/diag_mode_weight_min"] = Metric(
                value=mode_weight_masked.min(), aggregation=AggregationType.MEAN
            )
            metrics["actor/diag_mode_weight_max"] = Metric(
                value=mode_weight_masked.max(), aggregation=AggregationType.MEAN
            )
        else:
            zero = torch.zeros((), device=policy_loss.device)
            metrics["actor/diag_mode_weight_min"] = Metric(value=zero, aggregation=AggregationType.MEAN)
            metrics["actor/diag_mode_weight_max"] = Metric(value=zero, aggregation=AggregationType.MEAN)
    return policy_loss, metrics


def value_loss(config: CriticConfig, model_output, data: TensorDict, dp_group=None):
    """value loss

    Args:
        config: CriticConfig
        model_output: model output from the model
        data: the input to the model
        dp_group: data paralle group

    Returns:
        value loss
    """
    vpreds = no_padding_2_padding(model_output["values"], data)  # (bsz, response_length)

    # Normalize the value loss over the global mini-batch (dp_size / batch_num_tokens /
    # global_batch_size) instead of the local micro-batch, so the accumulated critic gradient is
    # invariant to how the mini-batch is split into micro-batches (as the actor's ppo_loss does).
    dp_size = data["dp_size"]
    batch_num_tokens = data["batch_num_tokens"]
    global_batch_size = data["global_batch_size"]

    # When the loss is normalized over the global batch, each micro-batch contributes a partial sum,
    # so the loss metric must be aggregated with SUM to reflect the global-batch mean.
    if (
        dp_size > 1
        or batch_num_tokens is not None
        or global_batch_size is not None
        or config.loss_scale_factor is not None
    ):
        metric_aggregation = AggregationType.SUM
    else:
        metric_aggregation = AggregationType.MEAN

    # select fields and convert to padded tensor
    data = data.select("values", "returns", "response_mask").to_padded_tensor()
    values = data["values"]
    returns = data["returns"]
    response_mask = data["response_mask"].to(bool)

    vf_loss, vf_clipfrac = compute_value_loss(
        vpreds=vpreds,
        values=values,
        returns=returns,
        response_mask=response_mask,
        cliprange_value=config.cliprange_value,
        loss_agg_mode=config.loss_agg_mode,
        dp_size=dp_size,
        batch_num_tokens=batch_num_tokens,
        global_batch_size=global_batch_size,
        loss_scale_factor=config.loss_scale_factor,
    )

    metrics = {
        "critic/vf_loss": Metric(value=vf_loss, aggregation=metric_aggregation),
        "critic/vf_clipfrac": vf_clipfrac.detach().item(),
        "critic/vpred_mean": masked_mean(vpreds, response_mask).detach().item(),
    }

    return vf_loss, metrics
