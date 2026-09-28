#!/usr/bin/env bash
# V30 simulator-free GRPO. Run from training/verl after exporting MODEL_PATH
# and SFT_LORA_PATH. The dataset contains only prior-window videos; reward is
# looked up from precomputed same-snapshot counterfactual feedback.
set -xeuo pipefail

MODEL_PATH=${MODEL_PATH:?Set MODEL_PATH to the Qwen3.5 base model directory}
SFT_LORA_PATH=${SFT_LORA_PATH:?Set SFT_LORA_PATH to the completed V35 SFT LoRA directory}
DATASET_DIR=${DATASET_DIR:-$HOME/VLMTSCS/grpo_v30_offline_local_video_dataset}
NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-4}
SMOKE=${SMOKE:-0}

TRAIN_FILE=$DATASET_DIR/train.jsonl
VAL_FILE=$DATASET_DIR/val.jsonl
TRAIN_MAX_SAMPLES=-1
TOTAL_EPOCHS=2
if [[ "$SMOKE" == "1" ]]; then
  TRAIN_MAX_SAMPLES=500
  TOTAL_EPOCHS=1
fi

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  algorithm.kl_ctrl.kl_coef=0.02 \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$VAL_FILE" \
  data.train_max_samples=$TRAIN_MAX_SAMPLES \
  data.train_batch_size=8 \
  data.max_prompt_length=65536 \
  data.max_response_length=768 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.dataloader_num_workers=8 \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.trust_remote_code=True \
  actor_rollout_ref.model.lora_adapter_path="$SFT_LORA_PATH" \
  actor_rollout_ref.model.lora_rank=16 \
  actor_rollout_ref.model.lora_alpha=32 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=65536 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=sync \
  actor_rollout_ref.rollout.tensor_model_parallel_size=$NDEVICES_PER_NODE \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=0.95 \
  actor_rollout_ref.rollout.max_model_len=65536 \
  actor_rollout_ref.rollout.max_num_batched_tokens=65536 \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=65536 \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=65536 \
  reward.custom_reward_function.path="$(pwd)/examples/vlmtscs/v30_offline_reward.py" \
  reward.custom_reward_function.name=compute_score \
  reward.reward_manager.name=naive \
  reward.num_workers=8 \
  trainer.n_gpus_per_node=$NDEVICES_PER_NODE \
  trainer.total_epochs=$TOTAL_EPOCHS \
  trainer.project_name=vlmtscs_v30_offline_grpo \
  trainer.experiment_name="qwen35_v30_offline_$(date +%Y%m%d_%H%M%S)" \
  trainer.save_freq=10 \
  trainer.test_freq=10 \
  trainer.logger='["console"]' \
  "$@"
