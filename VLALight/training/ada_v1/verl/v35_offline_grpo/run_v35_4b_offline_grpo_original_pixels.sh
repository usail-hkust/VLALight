#!/usr/bin/env bash
set -euo pipefail

# Original-pixel GRPO comparison run.
# The shared launcher keeps the same GRPO algorithm, reward, batch and rollout
# settings as the reduced-pixel run; only model, data and visual pixel budget
# are overridden here.
MODEL_PATH=/path/to/models/v35_four_video_context_reasoning \
DATA_DIR=/path/to/datasets/grpo_v30_offline_local_video_dataset \
MM_MAX_PIXELS=1105920 \
ACTOR_MAX_TOKEN_LEN_PER_GPU=${ACTOR_MAX_TOKEN_LEN_PER_GPU:-32768} \
ROLLOUT_LOGPROB_MAX_TOKEN_LEN_PER_GPU=${ROLLOUT_LOGPROB_MAX_TOKEN_LEN_PER_GPU:-32768} \
bash "$(dirname -- "${BASH_SOURCE[0]}")/run_v35_4b_offline_grpo.sh" "$@"
