#!/usr/bin/env bash
set -euo pipefail

# Keep GRPO groups balanced while training. Set V35_FORCE_MODE_BALANCE=0 to
# run an unforced/deployment-style rollout instead.
export V35_FORCE_MODE_BALANCE=${V35_FORCE_MODE_BALANCE:-1}
export V35_ROLLOUT_N=${V35_ROLLOUT_N:-6}
export V35_MODE_TOKEN_CONSTRAINT=${V35_MODE_TOKEN_CONSTRAINT:-1}
export V35_SHARED_PERCEPTION=${V35_SHARED_PERCEPTION:-1}
SMOKE=${SMOKE:-0}

MODEL_PATH=${MODEL_PATH:-/path/to/models/v35_four_video_context_reasoning_512x960_retrain}
DATA_DIR=${DATA_DIR:-/path/to/datasets/grpo_v30_offline_local_video_dataset_reduced_pixels}
WORK_DIR=${WORK_DIR:-/path/to/runs/v35_4b_offline_grpo_512x960_4tags}
REWARD_LOG=${REWARD_LOG:-${WORK_DIR}/reward_details.jsonl}
MODE_PROB_DIAGNOSTIC_LOG=${MODE_PROB_DIAGNOSTIC_LOG:-${WORK_DIR}/mode_probability_diagnostics.jsonl}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
N_GPUS=${N_GPUS:-4}
# A 4B model fits comfortably on one H100. TP=1 lets the four trainer ranks
# host four rollout replicas instead of paying TP=4 communication overhead.
ROLLOUT_TP=${ROLLOUT_TP:-1}
FSDP_SIZE=${FSDP_SIZE:-4}
# Qwen3.5's Gated DeltaNet needs the optional `fla` package for Ulysses CP.
# Keep this disabled unless that dependency is installed and explicitly tested.
ULYSSES_SP_SIZE=${ULYSSES_SP_SIZE:-1}
USE_FUSED_KERNELS=${USE_FUSED_KERNELS:-True}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-4}
ROLLOUT_ENFORCE_EAGER=${ROLLOUT_ENFORCE_EAGER:-True}
VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-}
AGENT_LOOP_WORKERS=${AGENT_LOOP_WORKERS:-8}
ROLLOUT_N=${ROLLOUT_N:-6}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16}
V35_PERCEPTION_SFT_COEF=${V35_PERCEPTION_SFT_COEF:-0.1}
V35_SIGNAL_AUX_COEF=${V35_SIGNAL_AUX_COEF:-0.03}
V35_SIGNAL_ADVANTAGE_CLIP=${V35_SIGNAL_ADVANTAGE_CLIP:-2.0}
V35_MODE_SELECTOR_COEF=${V35_MODE_SELECTOR_COEF:-0.02}
V35_MODE_SELECTOR_TEMPERATURE=${V35_MODE_SELECTOR_TEMPERATURE:-0.2}
V35_MODE_SELECTOR_MIN_PROB=${V35_MODE_SELECTOR_MIN_PROB:-0.2}
V35_MODE_BERNOULLI_ROUTING=${V35_MODE_BERNOULLI_ROUTING:-1}
V35_MODE_SELECTOR_SEED=${V35_MODE_SELECTOR_SEED:-42}
V35_MODE_ROUTING_FALLBACK_REASON=${V35_MODE_ROUTING_FALLBACK_REASON:-none}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-2}
TEST_FREQ=${TEST_FREQ:-10}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}
TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:--1}
VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:--1}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-61440}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-65536}
# Dynamic PPO batching uses this as the packed-token budget per GPU. Keeping
# it at 65K can pack several ~9K multimodal samples onto one H100 and exhaust
# activation memory even when ppo_micro_batch_size_per_gpu=1.
ACTOR_MAX_TOKEN_LEN_PER_GPU=${ACTOR_MAX_TOKEN_LEN_PER_GPU:-16384}
ROLLOUT_LOGPROB_MAX_TOKEN_LEN_PER_GPU=${ROLLOUT_LOGPROB_MAX_TOKEN_LEN_PER_GPU:-32768}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.50}
MM_MAX_PIXELS=${MM_MAX_PIXELS:-491520}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REWARD_FILE=${REWARD_FILE:-${SCRIPT_DIR}/v35_offline_grpo_reward.py}

if [[ "${SMOKE}" == "1" ]]; then
    # Four prompts produce 24 GRPO rollouts plus four gold perception rows.
    # This is exactly one actor mini-batch and is divisible across four GPUs.
    TRAIN_MAX_SAMPLES=4
    TRAIN_BATCH_SIZE=4
    PPO_MINI_BATCH_SIZE=4
    TOTAL_EPOCHS=1
    VAL_MAX_SAMPLES=4
    VAL_BEFORE_TRAIN=False
    # Validate once after the update so the smoke covers the calibrated vLLM
    # mode router, not only the training-side threshold update.
    TEST_FREQ=1
fi

TRAIN_FILE=${TRAIN_FILE:-${DATA_DIR}/train_2000_temporal_prompt.jsonl}
VAL_FILE=${VAL_FILE:-${DATA_DIR}/val_100_temporal_prompt.jsonl}
OUTPUT_DIR="${WORK_DIR}/checkpoints"
LOG_DIR="${WORK_DIR}/logs"
export CUDA_VISIBLE_DEVICES MODEL_PATH V35_GRPO_MODEL_PATH=${MODEL_PATH}
export V35_GRPO_REWARD_LOG=${REWARD_LOG}
export V35_PERCEPTION_SFT_COEF
export V35_SIGNAL_AUX_COEF V35_SIGNAL_ADVANTAGE_CLIP
export V35_MODE_SELECTOR_COEF V35_MODE_SELECTOR_TEMPERATURE
export V35_MODE_SELECTOR_MIN_PROB V35_MODE_BERNOULLI_ROUTING V35_MODE_SELECTOR_SEED
export V35_MODE_ROUTING_FALLBACK_REASON
export V35_MODE_PROB_DIAGNOSTIC_LOG=${MODE_PROB_DIAGNOSTIC_LOG}
if [[ -n "${VLLM_ATTENTION_BACKEND}" ]]; then
    export VLLM_ATTENTION_BACKEND
else
    unset VLLM_ATTENTION_BACKEND || true
fi
export V35_GRPO_DATA_DIR=${DATA_DIR}
export RAY_TMPDIR=${RAY_TMPDIR:-/dev/shm/${USER:-user}_v35_grpo_ray}
export TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

for path in "${MODEL_PATH}/config.json" "${TRAIN_FILE}" "${VAL_FILE}" "${REWARD_FILE}"; do
    [[ -e "${path}" ]] || { echo "Missing: ${path}" >&2; exit 1; }
done
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}" "${RAY_TMPDIR}" "$(dirname "${REWARD_LOG}")" "$(dirname "${MODE_PROB_DIAGNOSTIC_LOG}")"

if [[ "${V35_SIGNAL_AUX_COEF}" != "0" || "${V35_MODE_SELECTOR_COEF}" != "0" ]]; then
    [[ "${ROLLOUT_N}" == "${V35_ROLLOUT_N}" ]] || {
        echo "ROLLOUT_N (${ROLLOUT_N}) must equal V35_ROLLOUT_N (${V35_ROLLOUT_N})" >&2
        exit 1
    }
fi
if [[ "${V35_MODE_SELECTOR_COEF}" != "0" ]]; then
    [[ "${V35_FORCE_MODE_BALANCE}" == "1" && "${V35_MODE_TOKEN_CONSTRAINT}" == "1" ]] || {
        echo "Mode selector training requires balanced and constrained FAST/SLOW rollouts" >&2
        exit 1
    }
    (( ROLLOUT_N >= 2 && ROLLOUT_N % 2 == 0 )) || {
        echo "Mode selector training requires an even ROLLOUT_N" >&2
        exit 1
    }
fi
if [[ "${V35_MODE_BERNOULLI_ROUTING}" == "1" ]]; then
    [[ "${V35_MODE_SELECTOR_COEF}" != "0" ]] || {
        echo "Bernoulli mode routing requires V35_MODE_SELECTOR_COEF > 0" >&2
        exit 1
    }
fi
if [[ "${V35_MODE_BERNOULLI_ROUTING}" == "1" ]]; then
    if ! python - <<'PY'
from vllm.sampling_params import SamplingParams

SamplingParams(max_tokens=1, prompt_logprobs=0)
print("Bernoulli mode routing preflight: prompt_logprobs PASS")
PY
    then
        echo "WARNING: this vLLM does not support request-level prompt_logprobs; disabling Bernoulli Val routing." >&2
        export V35_MODE_BERNOULLI_ROUTING=0
        export V35_MODE_ROUTING_FALLBACK_REASON=vllm_sampling_params_missing_prompt_logprobs
    fi
fi
stamp=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/v35_4b_offline_grpo_${stamp}.log"
TENSORBOARD_DIR="${WORK_DIR}/tensorboard/${stamp}"
export TENSORBOARD_DIR
mkdir -p "${TENSORBOARD_DIR}"

python -c 'from torch.utils.tensorboard import SummaryWriter' >/dev/null 2>&1 || {
    echo "Missing TensorBoard dependency. Install it with: pip install tensorboard" >&2
    exit 1
}

echo "model=${MODEL_PATH}"
echo "data=${DATA_DIR} train=$(wc -l < "${TRAIN_FILE}") val=$(wc -l < "${VAL_FILE}")"
echo "smoke=${SMOKE} train_max_samples=${TRAIN_MAX_SAMPLES}"
echo "multimodal_max_pixels=${MM_MAX_PIXELS} (512x960)"
echo "epochs=${TOTAL_EPOCHS} val_before_train=${VAL_BEFORE_TRAIN} validation_every_steps=${TEST_FREQ}"
echo "gpus=${CUDA_VISIBLE_DEVICES} rollout_tp=${ROLLOUT_TP} rollout_n=${ROLLOUT_N} max_response=${MAX_RESPONSE_LENGTH}"
echo "vllm_attention_backend=${VLLM_ATTENTION_BACKEND:-auto} max_num_seqs=${ROLLOUT_MAX_NUM_SEQS} enforce_eager=${ROLLOUT_ENFORCE_EAGER}"
echo "actor_fused_kernels=${USE_FUSED_KERNELS} ulysses_sp_size=${ULYSSES_SP_SIZE}"
echo "actor_token_budget_per_gpu=${ACTOR_MAX_TOKEN_LEN_PER_GPU} rollout_logprob_token_budget=${ROLLOUT_LOGPROB_MAX_TOKEN_LEN_PER_GPU}"
echo "val_batch=16 agent_loop_workers=${AGENT_LOOP_WORKERS} rollout_bypass_ppo_clip=true"
echo "tensorboard=${TENSORBOARD_DIR}"
echo "reward_log=${V35_GRPO_REWARD_LOG}"
echo "perception_sft_coef=${V35_PERCEPTION_SFT_COEF} perception_in_grpo_reward=0"
echo "shared_perception=${V35_SHARED_PERCEPTION} (one deterministic perception prefix per training UID)"
echo "signal_aux_coef=${V35_SIGNAL_AUX_COEF} signal_advantage_clip=${V35_SIGNAL_ADVANTAGE_CLIP}"
echo "mode_selector_coef=${V35_MODE_SELECTOR_COEF} temperature=${V35_MODE_SELECTOR_TEMPERATURE} min_prob=${V35_MODE_SELECTOR_MIN_PROB}"
echo "mode_bernoulli_routing=${V35_MODE_BERNOULLI_ROUTING} selector_seed=${V35_MODE_SELECTOR_SEED}"
echo "mode_routing_fallback_reason=${V35_MODE_ROUTING_FALLBACK_REASON}"

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.rollout_correction.bypass_mode=True \
    algorithm.rollout_correction.loss_type=ppo_clip \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_max_samples="${TRAIN_MAX_SAMPLES}" \
    data.val_max_samples="${VAL_MAX_SAMPLES}" \
    data.video_key=videos \
    data.image_key=images \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    +data.video_nframes=6 \
    +data.image_min_pixels=65536 \
    +data.image_max_pixels="${MM_MAX_PIXELS}" \
    +data.video_min_pixels=65536 \
    +data.video_max_pixels="${MM_MAX_PIXELS}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.val_batch_size=16 \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.filter_overlong_prompts=False \
    data.truncation=error \
    data.shuffle=True \
    data.dataloader_num_workers=2 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.trust_remote_code=True \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels="${USE_FUSED_KERNELS}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.shuffle=False \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${ACTOR_MAX_TOKEN_LEN_PER_GPU}" \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.calculate_entropy=False \
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.actor.entropy_from_logits_chunk_size=256 \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.fsdp_config.fsdp_size="${FSDP_SIZE}" \
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size="${ULYSSES_SP_SIZE}" \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${ACTOR_MAX_TOKEN_LEN_PER_GPU}" \
    actor_rollout_ref.ref.fsdp_config.ulysses_sequence_parallel_size="${ULYSSES_SP_SIZE}" \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.use_torch_compile=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
    actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}" \
    actor_rollout_ref.rollout.max_num_batched_tokens=65536 \
    actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS}" \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.prompt_length="${MAX_PROMPT_LENGTH}" \
    actor_rollout_ref.rollout.response_length="${MAX_RESPONSE_LENGTH}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${ROLLOUT_LOGPROB_MAX_TOKEN_LEN_PER_GPU}" \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager="${ROLLOUT_ENFORCE_EAGER}" \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enable_prefix_caching=False \
    actor_rollout_ref.rollout.agent.num_workers="${AGENT_LOOP_WORKERS}" \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=512 \
    reward.custom_reward_function.path="${REWARD_FILE}" \
    reward.custom_reward_function.name=compute_score \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name=v35_offline_grpo \
    trainer.experiment_name=v35_4b_offline_grpo \
    trainer.n_gpus_per_node="${N_GPUS}" \
    trainer.nnodes=1 \
    trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.save_freq=10 \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.default_local_dir="${OUTPUT_DIR}" \
    2>&1 | tee "${LOG_FILE}"

echo "completed; log=${LOG_FILE}"
