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
import logging
import os
import asyncio
import hashlib
import random
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.utils.tokenizer.chat_template import apply_chat_template
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_SHARED_PERCEPTION_CACHE: dict[tuple[str, int, str], tuple[list[int], list[float], str]] = {}
_SHARED_PERCEPTION_LOCKS: dict[tuple[str, int, str], asyncio.Lock] = {}


def _shared_perception_lock(key: tuple[str, int, str]) -> asyncio.Lock:
    lock = _SHARED_PERCEPTION_LOCKS.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _SHARED_PERCEPTION_LOCKS[key] = lock
    return lock


def _find_token_subsequence(values: list[int], needle: list[int]) -> int | None:
    if not needle or len(needle) > len(values):
        return None
    for index in range(len(values) - len(needle) + 1):
        if values[index : index + len(needle)] == needle:
            return index
    return None


def _mode_constraint_regex(mode: str) -> str:
    """Constrain the mode/reasoning branch without requiring ``</perception>``.

    Some SFT responses omit the perception closing tag.  The constraint must
    therefore not depend on perception parsing.  It only preserves the part of
    the schema that distinguishes the two policies: FAST never contains a
    reasoning block, while SLOW contains a non-empty one after its mode tag.
    """
    if mode not in {"fast", "slow"}:
        raise ValueError(f"Unsupported V35 mode constraint: {mode!r}")
    # xgrammar does not support lookahead.  This character fragment accepts
    # everything except the two literal control-tag openings, <mode> and
    # <reasoning>.  The mode tag itself must be the first response content.
    # Previously ``no_control_tag`` was allowed before it, which admitted
    # outputs such as ``fast\n<mode>fast</mode>...``.  Those rows had a valid
    # mode-token mask but an invalid decision response, so mode CE could learn
    # from a different textual contract than the decision policy.
    safe_control_char = (
        r"(?:[^<]|<[^mr]|<m[^o]|<mo[^d]|<mod[^e]|<mode[^>]|"
        r"<r[^e]|<re[^a]|<rea[^s]|<reas[^o]|<reaso[^n]|"
        r"<reason[^i]|<reasoni[^n]|<reasonin[^g]|<reasoning[^>])"
    )
    no_control_tag = rf"(?:{safe_control_char})*"
    if mode == "fast":
        # Keep the suffix permissive on purpose: malformed signal/reasoning
        # outcomes must still be retained and receive their format penalty.
        return rf"<mode>fast</mode>{no_control_tag}"
    return (
        rf"<mode>slow</mode>{no_control_tag}"
        rf"<reasoning>\s*[^<\s][^<]*</reasoning>{no_control_tag}"
    )


def _apply_mode_token_constraint(sampling_params: dict[str, Any], mode: str) -> dict[str, Any]:
    """Apply a vLLM structured-output constraint for a single rollout.

    This is intentionally a generation-time constraint rather than a text
    replacement after sampling, so response token ids and their logprobs stay
    consistent with the actual rollout used by GRPO.
    """
    try:
        from vllm.sampling_params import StructuredOutputsParams
    except ImportError as exc:
        raise RuntimeError(
            "V35_MODE_TOKEN_CONSTRAINT=1 requires a vLLM version exposing "
            "StructuredOutputsParams. Disable it or upgrade vLLM."
        ) from exc

    constrained = dict(sampling_params)
    constrained["structured_outputs"] = StructuredOutputsParams(regex=_mode_constraint_regex(mode))
    return constrained


@register("single_turn_agent")
class SingleTurnAgentLoop(AgentLoopBase):
    """Naive agent loop that only do single turn chat completion."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length
        seed = os.environ.get("V35_MODE_SELECTOR_SEED", "").strip()
        self._mode_selector_seed = seed or None

    def _mode_selector_draw(self, priority: int, perception_ids: list[int]) -> float:
        if self._mode_selector_seed is None:
            return random.random()
        payload = f"{self._mode_selector_seed}:{priority}:" + ",".join(map(str, perception_ids))
        value = int.from_bytes(hashlib.sha256(payload.encode("ascii")).digest()[:8], "big")
        return value / float(1 << 64)

    async def _get_shared_perception(
        self,
        *,
        uid: Any,
        global_steps: Any,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        images: Any,
        audios: Any,
        videos: Any,
        mm_processor_kwargs: dict[str, Any],
        priority: int,
    ) -> tuple[list[int], list[float], str, bool]:
        """Generate one deterministic perception prefix per training UID group."""
        if uid is None:
            return [], [], "", False
        step = -1 if global_steps is None else int(global_steps)
        key = (str(uid), step, str(self.tokenizer.__class__.__name__))
        cached = _SHARED_PERCEPTION_CACHE.get(key)
        if cached is not None:
            return cached[0], cached[1], cached[2], True

        async with _shared_perception_lock(key):
            cached = _SHARED_PERCEPTION_CACHE.get(key)
            if cached is not None:
                return cached[0], cached[1], cached[2], True

            first_params = dict(sampling_params)
            first_params.pop("logits_processors", None)
            first_params.pop("structured_outputs", None)
            first_params.pop("prompt_logprobs", None)
            first_params["temperature"] = 0.0
            first_params["top_p"] = 1.0
            first_params["top_k"] = -1
            first_params["stop"] = ["</perception>"]
            first_params["include_stop_str_in_output"] = True
            first_params["max_tokens"] = min(
                int(first_params.get("max_tokens", self.response_length)), self.response_length
            )
            first = await self.server_manager.generate(
                request_id=(
                    f"det-perception-{priority}"
                    if getattr(self.rollout_config, "full_determinism", False)
                    else uuid4().hex
                ),
                prompt_ids=prompt_ids,
                sampling_params=first_params,
                image_data=images,
                audio_data=audios,
                video_data=videos,
                mm_processor_kwargs=mm_processor_kwargs,
                priority=priority,
            )
            perception_ids = list(first.token_ids)
            close_ids = self.tokenizer.encode("</perception>", add_special_tokens=False)
            close_start = _find_token_subsequence(perception_ids, close_ids)
            if close_start is None:
                return [], [], "", False

            perception_ids = perception_ids[: close_start + len(close_ids)]
            perception_text = self.tokenizer.decode(perception_ids, skip_special_tokens=False)
            perception_logprobs = list(first.log_probs or [])[: len(perception_ids)]
            perception_hash = hashlib.sha256(perception_text.encode("utf-8")).hexdigest()
            _SHARED_PERCEPTION_CACHE[key] = (perception_ids, perception_logprobs, perception_hash)

            if len(_SHARED_PERCEPTION_CACHE) > 2048:
                for stale_key in list(_SHARED_PERCEPTION_CACHE)[:512]:
                    _SHARED_PERCEPTION_CACHE.pop(stale_key, None)
                    _SHARED_PERCEPTION_LOCKS.pop(stale_key, None)
            return perception_ids, perception_logprobs, perception_hash, True

    async def _run_bernoulli_validation(
        self,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        images: Any,
        videos: Any,
        audios: Any,
        mm_processor_kwargs: dict[str, Any],
        priority: int,
    ) -> tuple[TokenOutput, list[int], list[int], list[float] | None, dict[str, Any]] | None:
        """Route Val/deployment with vLLM 0.18 prompt logprobs.

        The first request stops at ``</perception>``. Two short prompt-only
        requests then score the same perception prefix followed by the fast
        and slow mode strings. The selected mode is appended to that prefix
        and only the remaining reasoning/signal suffix is generated.
        """
        from v35_offline_grpo.mode_routing import (
            binary_slow_probability,
            extract_prompt_sequence_logprob,
            extract_prompt_sequence_token_logprobs,
            mode_token_layout,
            write_mode_probability_diagnostic,
        )

        def new_request_id() -> str:
            return f"det-{priority}" if getattr(self.rollout_config, "full_determinism", False) else uuid4().hex

        async def generate(ids: list[int], params: dict[str, Any]) -> TokenOutput:
            return await self.server_manager.generate(
                request_id=new_request_id(),
                prompt_ids=ids,
                sampling_params=dict(params),
                image_data=images,
                audio_data=audios,
                video_data=videos,
                mm_processor_kwargs=mm_processor_kwargs,
                priority=priority,
            )

        layout = mode_token_layout(self.tokenizer)
        perception_params = dict(sampling_params)
        perception_params.pop("logits_processors", None)
        perception_params.pop("structured_outputs", None)
        perception_params["stop"] = ["</perception>"]
        perception_params["include_stop_str_in_output"] = True
        perception_params["logprobs"] = 1
        perception_params.pop("prompt_logprobs", None)
        perception_params["max_tokens"] = min(self.response_length, int(perception_params.get("max_tokens", self.response_length)))
        perception_output = await generate(prompt_ids, perception_params)
        perception_ids = list(perception_output.token_ids)
        close_ids = self.tokenizer.encode("</perception>", add_special_tokens=False)
        perception_closed = bool(close_ids) and tuple(perception_ids[-len(close_ids) :]) == tuple(close_ids)
        if not perception_closed:
            # Never fabricate a closing tag: the one-stage validation path is
            # the only honest fallback when the first request is truncated.
            write_mode_probability_diagnostic(
                os.environ.get("V35_MODE_PROB_DIAGNOSTIC_LOG", ""),
                {
                    "global_step": int(getattr(perception_output, "extra_fields", {}).get("global_steps", -1)),
                    "sample_id": str(priority),
                    "decode": "one_stage_fallback",
                    "routing_fallback_reason": "perception_prefix_not_closed",
                    "threshold": None,
                    "selected_mode": None,
                    "p_slow_model": None,
                    "p_fast_model": None,
                    "perception_token_count": len(perception_ids),
                },
            )
            return None

        score_params = {
            "max_tokens": 1,
            "prompt_logprobs": 0,
            "temperature": 0.0,
        }
        candidate_scores: dict[str, float] = {}
        candidate_token_logs: dict[str, list[float]] = {}
        for mode, mode_ids in layout["full"].items():
            score_output = await generate(prompt_ids + perception_ids + mode_ids, score_params)
            # Score only the mode word. The surrounding ``\n<mode>`` and
            # ``</mode>`` tokens are identical candidates and must not create
            # a length-dependent prior.
            score = extract_prompt_sequence_logprob(score_output.extra_fields, layout["words"][mode])
            if score is None:
                raise RuntimeError(f"vLLM did not return prompt_logprobs for {mode} candidate")
            candidate_scores[mode] = score
            token_logs = extract_prompt_sequence_token_logprobs(
                score_output.extra_fields, layout["words"][mode]
            )
            if token_logs is None:
                raise RuntimeError(f"vLLM did not return token prompt_logprobs for {mode} candidate")
            candidate_token_logs[mode] = token_logs

        p_slow = binary_slow_probability(candidate_scores["fast"], candidate_scores["slow"])
        min_probability = float(os.environ.get("V35_MODE_SELECTOR_MIN_PROB", "0.2"))
        min_probability = min(0.5, max(0.0, min_probability))
        p_slow_raw = float(p_slow)
        p_slow_clipped = min(1.0 - min_probability, max(min_probability, p_slow_raw))
        selector_draw = self._mode_selector_draw(priority, perception_ids)
        selected_mode = "slow" if selector_draw < p_slow_clipped else "fast"
        selected_ids = layout["full"][selected_mode]
        suffix_params = dict(sampling_params)
        suffix_params.pop("logits_processors", None)
        suffix_params.pop("structured_outputs", None)
        suffix_params["logprobs"] = 1
        suffix_params["max_tokens"] = max(
            1,
            self.response_length - len(perception_ids) - len(selected_ids),
        )
        suffix_output = await generate(prompt_ids + perception_ids + selected_ids, suffix_params)

        diagnostic = {
            "global_step": int(getattr(perception_output, "extra_fields", {}).get("global_steps", -1)),
            "sample_id": str(priority),
            "decode": "two_stage_prompt_logprobs",
            "p_slow_model": p_slow,
            "p_fast_model": 1.0 - p_slow,
            "p_slow_raw": p_slow_raw,
            "p_slow_clipped": p_slow_clipped,
            "selector_draw": selector_draw,
            "route_method": "bernoulli",
            "threshold": None,
            "threshold_used": False,
            "selected_mode": selected_mode,
            "fast_logprob": candidate_scores["fast"],
            "slow_logprob": candidate_scores["slow"],
            "fast_token_logprobs": candidate_token_logs["fast"],
            "slow_token_logprobs": candidate_token_logs["slow"],
            "perception_token_count": len(perception_ids),
            "candidate_token_count": len(selected_ids),
            "suffix_max_tokens": suffix_params["max_tokens"],
        }
        write_mode_probability_diagnostic(os.environ.get("V35_MODE_PROB_DIAGNOSTIC_LOG", ""), diagnostic)

        response_ids = perception_ids + selected_ids + list(suffix_output.token_ids)
        response_mask = [1] * len(response_ids)
        # This route is validation-only.  The mode candidates were scored by
        # prompt logprobs in separate requests, so they are not a valid single
        # sampled-response logprob stream for PPO or response diagnostics.
        response_logprobs = None
        extra_fields = dict(suffix_output.extra_fields)
        extra_fields.update({"bernoulli_mode_route": diagnostic, "selected_mode": selected_mode})
        merged_output = TokenOutput(
            token_ids=response_ids,
            log_probs=response_logprobs,
            routed_experts=suffix_output.routed_experts,
            stop_reason=suffix_output.stop_reason,
            num_preempted=(perception_output.num_preempted or 0) + (suffix_output.num_preempted or 0),
            extra_fields=extra_fields,
        )
        return merged_output, response_ids, response_mask, response_logprobs, extra_fields

    async def _run_stage2_bernoulli_validation(
        self, *, prompt_ids: list[int], sampling_params: dict[str, Any],
        priority: int, sampling_temperature: float | None,
    ) -> tuple[TokenOutput, list[int], list[int], None, dict[str, Any]]:
        """Score FAST/SLOW on the unchanged Stage 2 prefix, then sample a branch."""
        from v35_offline_grpo.mode_routing import (
            binary_slow_probability,
            extract_prompt_sequence_logprob,
            extract_prompt_sequence_token_logprobs,
            mode_token_layout,
            write_mode_probability_diagnostic,
        )

        layout = mode_token_layout(self.tokenizer)

        async def generate(ids: list[int], params: dict[str, Any]) -> TokenOutput:
            return await self.server_manager.generate(
                request_id=(
                    f"stage2-mode-{priority}"
                    if getattr(self.rollout_config, "full_determinism", False)
                    else uuid4().hex
                ),
                prompt_ids=ids, sampling_params=params, image_data=None,
                audio_data=None, video_data=None, mm_processor_kwargs={}, priority=priority,
            )

        score_params = {"max_tokens": 1, "prompt_logprobs": 0, "temperature": 0.0}
        scores, token_logs = {}, {}
        for mode, mode_ids in layout["full"].items():
            scored = await generate(prompt_ids + mode_ids, dict(score_params))
            score = extract_prompt_sequence_logprob(scored.extra_fields, layout["words"][mode])
            logs = extract_prompt_sequence_token_logprobs(scored.extra_fields, layout["words"][mode])
            if score is None or logs is None:
                raise RuntimeError(f"vLLM did not return prompt_logprobs for {mode} mode")
            scores[mode], token_logs[mode] = float(score), logs

        p_slow_raw = binary_slow_probability(scores["fast"], scores["slow"])
        floor = float(os.environ.get("V35_MODE_SELECTOR_MIN_PROB", "0.2"))
        p_slow = min(1.0 - floor, max(floor, p_slow_raw))
        draw = self._mode_selector_draw(priority, prompt_ids)
        selected_mode = "slow" if draw < p_slow else "fast"
        selected_ids = list(layout["full"][selected_mode])
        suffix_params = dict(sampling_params)
        suffix_params.pop("structured_outputs", None)
        suffix_params.pop("logits_processors", None)
        suffix_params.pop("prompt_logprobs", None)
        if sampling_temperature is not None:
            suffix_params["temperature"] = float(sampling_temperature)
        suffix_params["stop"] = ["</signal>"]
        suffix_params["include_stop_str_in_output"] = True
        suffix_params["max_tokens"] = max(
            1, min(int(suffix_params.get("max_tokens", self.response_length)),
                   self.response_length - len(selected_ids))
        )
        suffix = await generate(prompt_ids + selected_ids, suffix_params)
        response_ids = selected_ids + list(suffix.token_ids)
        diagnostic = {
            "route_method": "bernoulli", "selected_mode": selected_mode,
            "p_slow_model": p_slow_raw, "p_slow_sampled": p_slow,
            "p_fast_model": 1.0 - p_slow_raw, "selector_draw": draw,
            "fast_logprob": scores["fast"], "slow_logprob": scores["slow"],
            "fast_token_logprobs": token_logs["fast"],
            "slow_token_logprobs": token_logs["slow"],
        }
        write_mode_probability_diagnostic(
            os.environ.get("V35_MODE_PROB_DIAGNOSTIC_LOG", ""), diagnostic
        )
        output = TokenOutput(
            token_ids=response_ids, log_probs=None, routed_experts=suffix.routed_experts,
            stop_reason=suffix.stop_reason, num_preempted=suffix.num_preempted,
            extra_fields={**(suffix.extra_fields or {}), "bernoulli_mode_route": diagnostic,
                          "selected_mode": selected_mode},
        )
        return output, response_ids, [1] * len(response_ids), None, output.extra_fields

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], priority: int = 0, **kwargs) -> AgentLoopOutput:
        # priority may arrive as np.int64 from non_tensor_batch; normalize to Python int.
        priority = int(priority)
        messages = list(kwargs["raw_prompt"])
        # Online Stage-2 supplies the per-job mode assignment through the
        # non-tensor batch.  Prefer it over the legacy process-wide toggle so
        # each target can receive its exact balanced fast/slow assignment.
        forced_mode = kwargs.get("forced_mode")
        if forced_mode is not None:
            forced_mode = str(forced_mode).strip().lower()
            if forced_mode not in {"fast", "slow"}:
                raise ValueError(f"forced_mode must be fast or slow, got {forced_mode!r}")
        is_validation = bool(kwargs.get("is_validation", False))
        job_kind = str(kwargs.get("job_kind", "legacy_combined"))
        is_stage2 = job_kind.startswith("stage2_")
        deployment = bool(kwargs.get("deployment", False))
        sampling_temperature = kwargs.get("sampling_temperature")
        if sampling_temperature is not None:
            sampling_temperature = float(sampling_temperature)
            sampling_params = dict(sampling_params)
            sampling_params["temperature"] = sampling_temperature
        mode_token_constraint = (
            is_stage2 and not deployment
            and os.environ.get("V35_MODE_TOKEN_CONSTRAINT", "0") == "1"
        )
        # Optional V35 training-only mode balancing. Deployment leaves this
        # disabled, so the model receives no forced [FAST]/[SLOW] prefix.
        if (
            forced_mode is None
            and is_stage2
            and not deployment
            and not is_validation
            and os.environ.get("V35_FORCE_MODE_BALANCE", "0") == "1"
        ):
            rollout_id = int(kwargs.get("rollout_n", 0))
            rollout_count = int(os.environ.get("V35_ROLLOUT_N", "6"))
            if rollout_count < 2 or rollout_count % 2:
                raise ValueError("V35_ROLLOUT_N must be an even number >= 2")
            forced_mode = "fast" if rollout_id < rollout_count // 2 else "slow"
        if forced_mode is not None and is_stage2 and not deployment and not mode_token_constraint:
            raise RuntimeError(
                "forced mode requires V35_MODE_TOKEN_CONSTRAINT=1; "
                "modifying the formal Stage 2 prompt is forbidden"
            )

        # 1. extract multimodal inputs from messages
        multi_modal_data = await self.process_multi_modal_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")
        audios = multi_modal_data.get("audios")
        mm_processor_kwargs = self._get_mm_processor_kwargs(audios)

        # 2. apply chat template and tokenize
        use_continuous_token = self.enable_continuous_token and not multi_modal_data
        bernoulli_validation = (
            is_validation
            and ((job_kind == "legacy_combined") or (is_stage2 and deployment))
            and os.environ.get("V35_MODE_BERNOULLI_ROUTING", "0") == "1"
            and not use_continuous_token
        )
        if use_continuous_token:
            prompt_ids = await self.ct_build_initial_tokens(messages)
            processor_prompt = None
        else:
            # Preserve the exact pre-processor template. Reward-time multimodal
            # processing must start from these one-per-media placeholders, rather
            # than decoding the patch-expanded prompt IDs returned by the model.
            processor_prompt = await self.loop.run_in_executor(
                None,
                lambda: apply_chat_template(
                    self.processor,
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            prompt_ids = await self.apply_chat_template(
                messages,
                images=images,
                videos=videos,
                audios=audios,
                mm_processor_kwargs=mm_processor_kwargs,
            )

        # 3. generate sequences
        # Training does not enter the adaptive-validation branch, so this must
        # exist before the normal rollout timer below.
        metrics = {}
        bernoulli_extra_fields = None
        shared_perception_ids: list[int] = []
        shared_perception_logprobs: list[float] = []
        shared_perception_hash: str | None = None
        shared_perception_used = False
        shared_perception_fallback_reason: str | None = None
        if (
            not is_validation
            and job_kind == "legacy_combined"
            and os.environ.get("V35_SHARED_PERCEPTION", "0") == "1"
            and not use_continuous_token
        ):
            (
                shared_perception_ids,
                shared_perception_logprobs,
                shared_perception_hash,
                shared_perception_used,
            ) = await self._get_shared_perception(
                uid=kwargs.get("uid"),
                global_steps=kwargs.get("global_steps", -1),
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                images=images,
                audios=audios,
                videos=videos,
                mm_processor_kwargs=mm_processor_kwargs,
                priority=priority,
            )
            if not shared_perception_used:
                shared_perception_fallback_reason = "missing_uid_or_incomplete_perception_prefix"
        if bernoulli_validation:
            if is_stage2 and deployment:
                bernoulli_result = await self._run_stage2_bernoulli_validation(
                    prompt_ids=prompt_ids, sampling_params=sampling_params,
                    priority=priority, sampling_temperature=sampling_temperature,
                )
            else:
                bernoulli_result = await self._run_bernoulli_validation(
                    prompt_ids=prompt_ids,
                    sampling_params=sampling_params,
                    images=images,
                    videos=videos,
                    audios=audios,
                    mm_processor_kwargs=mm_processor_kwargs,
                    priority=priority,
                )
            if bernoulli_result is not None:
                output, response_ids, response_mask, response_logprobs, bernoulli_extra_fields = bernoulli_result
                metrics = {"bernoulli_mode_route": 1.0}
            else:
                bernoulli_validation = False
                metrics = {
                    "bernoulli_mode_route": 0.0,
                    "bernoulli_mode_route_fallback": 1.0,
                }
        if not bernoulli_validation:
            if forced_mode is not None and mode_token_constraint:
                sampling_params = _apply_mode_token_constraint(sampling_params, forced_mode)
            # A formal Stage-2 response is complete as soon as the signal
            # closing tag is emitted.  The generic rollout defaults otherwise
            # leave ``max_tokens=4096`` with only EOS as a stop condition.  In
            # combination with the mode xgrammar regex, a malformed/verbose
            # tail can keep vLLM's ``sample_tokens`` RPC busy until its
            # 300-second executor timeout, killing the EngineCore after an
            # otherwise successful actor update.  Keep the closing tag in the
            # returned response so the protocol parser and response masks are
            # unchanged; this only bounds generation at the protocol boundary.
            if is_stage2:
                sampling_params = dict(sampling_params)
                existing_stop = sampling_params.get("stop")
                if existing_stop is None:
                    stop_values = []
                elif isinstance(existing_stop, str):
                    stop_values = [existing_stop]
                else:
                    stop_values = list(existing_stop)
                if "</signal>" not in stop_values:
                    stop_values.append("</signal>")
                sampling_params["stop"] = stop_values
                sampling_params["include_stop_str_in_output"] = True
            with simple_timer("generate_sequences", metrics):
                if shared_perception_used:
                    suffix_params = dict(sampling_params)
                    suffix_params["stop"] = ["</signal>"]
                    suffix_params["include_stop_str_in_output"] = True
                    suffix_params["max_tokens"] = max(
                        1,
                        min(
                            int(suffix_params.get("max_tokens", self.response_length)),
                            self.response_length - len(shared_perception_ids),
                        ),
                    )
                    suffix = await self.server_manager.generate(
                        request_id=(
                            f"det-suffix-{priority}"
                            if getattr(self.rollout_config, "full_determinism", False)
                            else uuid4().hex
                        ),
                        prompt_ids=prompt_ids + shared_perception_ids,
                        sampling_params=suffix_params,
                        image_data=images,
                        audio_data=audios,
                        video_data=videos,
                        mm_processor_kwargs=mm_processor_kwargs,
                        priority=priority,
                    )
                    output = TokenOutput(
                        token_ids=shared_perception_ids + list(suffix.token_ids),
                        log_probs=shared_perception_logprobs + list(suffix.log_probs or []),
                        routed_experts=suffix.routed_experts,
                        stop_reason=suffix.stop_reason,
                        num_preempted=suffix.num_preempted,
                        extra_fields={
                            **(suffix.extra_fields or {}),
                            "shared_perception": True,
                            "shared_perception_prefix_hash": shared_perception_hash,
                        },
                    )
                else:
                    request_id = (
                        f"det-{priority}"
                        if getattr(self.rollout_config, "full_determinism", False)
                        else uuid4().hex
                    )
                    output = await self.server_manager.generate(
                        request_id=request_id,
                        prompt_ids=prompt_ids,
                        sampling_params=sampling_params,
                        image_data=images,
                        audio_data=audios,
                        video_data=videos,
                        mm_processor_kwargs=mm_processor_kwargs,
                        priority=priority,
                    )
            output.extra_fields = dict(output.extra_fields or {})
            output.extra_fields.setdefault("shared_perception", False)
            if shared_perception_fallback_reason:
                output.extra_fields["shared_perception_fallback_reason"] = shared_perception_fallback_reason
            if shared_perception_hash:
                output.extra_fields["shared_perception_prefix_hash"] = shared_perception_hash
        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1

        if bernoulli_validation:
            pass
        elif use_continuous_token:
            merge_result, response_mask, response_logprobs = await self.ct_merge_assistant_token(
                prompt_ids,
                output.token_ids,
                [],
                [] if output.log_probs else None,
                assistant_logprobs=output.log_probs if output.log_probs else None,
            )
            response_ids = merge_result.token_ids[-len(response_mask) :] if response_mask else []
            prompt_ids = merge_result.token_ids[: len(merge_result.token_ids) - len(response_mask)]
        else:
            response_ids = output.token_ids
            response_mask = [1] * len(output.token_ids)
            response_logprobs = output.log_probs

        output: AgentLoopOutput = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=response_logprobs[: self.response_length] if response_logprobs else None,
            routed_experts=(
                output.routed_experts[: len(prompt_ids) + self.response_length]
                if output.routed_experts is not None
                else None
            ),
            multi_modal_data=multi_modal_data,
            mm_processor_kwargs=mm_processor_kwargs,
            processor_prompt=processor_prompt,
            num_turns=2,
            metrics=metrics,
            extra_fields=bernoulli_extra_fields if bernoulli_extra_fields is not None else output.extra_fields,
        )

        # keeping the schema consistent with tool_agent_loop
        output.extra_fields.update({"turn_scores": [], "tool_rewards": []})
        if forced_mode is not None:
            output.extra_fields["forced_mode"] = forced_mode
            output.extra_fields["rollout_n"] = int(kwargs.get("rollout_n", 0))

        return output
