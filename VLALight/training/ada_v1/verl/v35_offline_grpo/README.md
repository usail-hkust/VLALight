# V35 4B Offline GRPO

The current launcher combines three actor objectives while preserving the
balanced rollout contract:

```text
3 forced FAST + 3 forced SLOW rollouts
1 gold-perception actor row per prompt
traffic GRPO on the decision suffix after </mode>
counterfactual signal loss on <signal> content
same-perception FAST/SLOW soft CE on <mode> content
clipped model probability with Bernoulli mode routing for validation/deployment
```

Perception is trained only by gold-token CE. The scalar reward contains the
traffic utility, the configured reasoning-length penalty (applied only when a
reasoning block is present), and format penalties. There is no fixed
slow-mode prior or extra `slow_cost`; the legacy queue-gap mode bonus is
disabled.

Run a smoke test from the repository root:

```bash
python v35_offline_grpo/verify_reward_formula.py
python v35_offline_grpo/verify_adaptive_auxiliary.py

SMOKE=1 \
WORK_DIR=/path/to/runs/v35_adaptive_aux_smoke \
bash v35_offline_grpo/run_v35_4b_offline_grpo.sh
```

The smoke performs one four-prompt training update followed by four validation
samples. It appends model probabilities, independent random draws, and routed
modes to `mode_probability_diagnostics.jsonl`.

Important defaults:

```text
V35_PERCEPTION_SFT_COEF=0.10
V35_SIGNAL_AUX_COEF=0.03
V35_MODE_SELECTOR_COEF=0.02
V35_MODE_SELECTOR_TEMPERATURE=0.20
V35_MODE_SELECTOR_MIN_PROB=0.20
V35_MODE_BERNOULLI_ROUTING=1
V35_MODE_SELECTOR_SEED=42
```

The selector target has no fixed slow cost. A group contributes mode CE only
when both forced branches contain a valid decision suffix and the selected rows
share the exact same generated perception prefix. Invalid branches are skipped,
not converted into zero traffic utility. Validation clips the model's slow
probability to `[0.2, 0.8]` and samples from that probability. There is no
learned or fixed decision threshold.

Run a full job by explicitly selecting the intended cold-start checkpoint:

```bash
MODEL_PATH=/path/to/models/v35_four_video_context_reasoning_512x960_retrain \
TRAIN_BATCH_SIZE=12 \
PPO_MINI_BATCH_SIZE=12 \
WORK_DIR=/path/to/runs/v35_bernoulli_aux \
V35_MODE_BERNOULLI_ROUTING=1 \
bash v35_offline_grpo/run_v35_4b_offline_grpo.sh
```

Deployment uses the same two-stage route only when the serving endpoint is a
local vLLM 0.18+ Chat Completions server. Set these fields in
`utils/vlm_config.py` (or pass the equivalent runtime config):

```python
VLM_MODE_BERNOULLI_ROUTING = True
VLM_MODE_ROUTING_LOG = "/path/to/deployment/mode_probability_diagnostics.jsonl"
```

The endpoint must return `prompt_logprobs` and `prompt_token_ids`. The agent
first generates the closed perception block, scores `fast` and `slow` under
that exact prefix, clips the slow probability, samples the mode, and continues
the chosen mode. If the endpoint lacks these fields or perception is truncated, it logs
`one_stage_fallback` and preserves the original one-shot behavior. Do not
enable this option for a remote provider that does not implement vLLM's
prompt-logprob extension.
