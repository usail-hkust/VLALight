#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd -- "${VERL_ROOT}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
export V35_RUN_ID="${V35_RUN_ID:-${V35_RUN_NAME:-v35_online}_${RUN_TIMESTAMP}}"
CHECKPOINT_DIR="${VERL_ROOT}/checkpoints/${V35_RUN_ID}"
export V35_RUNTIME_ROOT="${V35_RUNTIME_ROOT:-${VERL_ROOT}/runtime/${V35_RUN_ID}}"
export TENSORBOARD_DIR="${TENSORBOARD_DIR:-${VERL_ROOT}/tensorboard_log/${V35_RUN_ID}}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export V35_VERBOSE_DEBUG="${V35_VERBOSE_DEBUG:-0}"
export V35_SUMO_DEBUG="${V35_SUMO_DEBUG:-0}"
export V35_RENDER_LOG_LEVEL="${V35_RENDER_LOG_LEVEL:-WARNING}"
# Separate actor micro-batches.  Both branches default to the conservative
# tested value: the binary mode branch still shares the full VLM forward, and
# raising RL to 4/8 has previously caused large activation peaks/OOMs.
export V35_SFT_MICRO_BATCH_SIZE_PER_GPU="${V35_SFT_MICRO_BATCH_SIZE_PER_GPU:-2}"
export V35_RL_MICRO_BATCH_SIZE_PER_GPU="${V35_RL_MICRO_BATCH_SIZE_PER_GPU:-2}"
export V35_MODE_TOKEN_CONSTRAINT=1
export V35_PERCEPTION_SFT_COEF="${V35_PERCEPTION_SFT_COEF:-0.1}"
export V35_MODE_SELECTOR_COEF="${V35_MODE_SELECTOR_COEF:-0.02}"
export V35_MODE_GDPO_REWARD_KEYS="${V35_MODE_GDPO_REWARD_KEYS:-network_reward,local_score,reasoning_cost_reward,format_penalty}"
export V35_MODE_SELECTOR_TEMPERATURE="${V35_MODE_SELECTOR_TEMPERATURE:-0.2}"
export V35_MODE_SELECTOR_MIN_PROB="${V35_MODE_SELECTOR_MIN_PROB:-0.2}"
# The mode router is a two-class soft-target CE over FAST/SLOW logits.  Keep
# the branch contribution bounded; no squared log-prob trust-KL is added to
# the classifier objective.
export V35_MODE_MAX_WEIGHTED_LOSS="${V35_MODE_MAX_WEIGHTED_LOSS:-0.10}"
export V35_MODE_TRUST_KL_COEF="${V35_MODE_TRUST_KL_COEF:-0.0}"
export V35_MODE_BERNOULLI_ROUTING="${V35_MODE_BERNOULLI_ROUTING:-1}"
export V35_MODE_SELECTOR_SEED="${V35_MODE_SELECTOR_SEED:-42}"
export V35_MODE_PROB_DIAGNOSTIC_LOG="${V35_MODE_PROB_DIAGNOSTIC_LOG:-${CHECKPOINT_DIR}/mode_probability_diagnostics.jsonl}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export V35_RENDER_EGL_DEVICES="${V35_RENDER_EGL_DEVICES:-0,1,2,3}"
export V35_RENDER_SLOTS_PER_GPU="${V35_RENDER_SLOTS_PER_GPU:-4}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export PYTHONPATH="${VERL_ROOT}:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

SMOKE=0
FRESH=0
SKIP_PREFLIGHT=0
CHECK_ENDPOINT=0
HYDRA_OVERRIDES=()
while (($#)); do
  case "$1" in
    --smoke)
      SMOKE=1
      ;;
    --fresh)
      FRESH=1
      ;;
    --skip-preflight)
      SKIP_PREFLIGHT=1
      ;;
    --check-policy-endpoint)
      CHECK_ENDPOINT=1
      ;;
    --help|-h)
      cat <<'EOF'
Usage: V35_MODEL_DIR=<merged-model-dir> run_formal_online_cooperative_grpo.sh [--smoke] [--fresh] [--skip-preflight] [--check-policy-endpoint] [hydra overrides]

Examples:
  ./v35_online_cooperative_grpo/run_formal_online_cooperative_grpo.sh --smoke --fresh
  ./v35_online_cooperative_grpo/run_formal_online_cooperative_grpo.sh --fresh
  ./v35_online_cooperative_grpo/run_formal_online_cooperative_grpo.sh --check-policy-endpoint --smoke --fresh
EOF
      exit 0
      ;;
    *)
      HYDRA_OVERRIDES+=("$1")
      ;;
  esac
  shift
done

cd -- "$VERL_ROOT"

if [[ -n "${V35_MODEL_DIR:-}" ]]; then
  if [[ ! -d "$V35_MODEL_DIR" ]]; then
    echo "V35_MODEL_DIR does not exist: $V35_MODEL_DIR" >&2
    exit 1
  fi
  for override in "${HYDRA_OVERRIDES[@]}"; do
    if [[ "${override%%=*}" == "actor_rollout_ref.model.path" ]]; then
      echo "Set the model through either V35_MODEL_DIR or actor_rollout_ref.model.path, not both." >&2
      exit 1
    fi
  done
  V35_MODEL_DIR="$(cd -- "$V35_MODEL_DIR" && pwd)"
  HYDRA_OVERRIDES+=("actor_rollout_ref.model.path=${V35_MODEL_DIR}")
fi

# A V35 run owns every GPU listed in CUDA_VISIBLE_DEVICES.  Starting a second
# Ray/vLLM cluster with the same mask does not provide another independent
# allocation: both clusters execute on the same devices and can leave
# vLLM's multiprocessing RPC waiting indefinitely.  Fail closed before
# creating any workers.  Set V35_ALLOW_CONCURRENT=1 only when the caller has
# deliberately isolated the processes with a different GPU mask/container.
if [[ "${V35_ALLOW_CONCURRENT:-0}" != "1" ]]; then
  existing_trainers="$(ps -eo pid=,args= 2>/dev/null | awk -v root="$VERL_ROOT" -v self="$$" '
    $1 != self && index($0, "python -m verl.trainer.main_ppo") && index($0, root) { print }
  ')"
  if [[ -n "$existing_trainers" ]]; then
    echo "An existing V35 trainer already uses this VERL root/GPU allocation:" >&2
    echo "$existing_trainers" >&2
    echo "Stop that run or set V35_ALLOW_CONCURRENT=1 after isolating GPUs." >&2
    exit 2
  fi
fi

# The process check above catches manually launched trainers.  The flock also
# closes the race between two invocations of this launcher before either
# trainer process becomes visible to ps.  The descriptor remains open for the
# lifetime of this shell and is released automatically on exit.
if command -v flock >/dev/null 2>&1 && [[ "${V35_ALLOW_CONCURRENT:-0}" != "1" ]]; then
  lock_mask="${CUDA_VISIBLE_DEVICES//,/__}"
  V35_RUN_LOCK_FILE="${V35_RUN_LOCK_FILE:-/tmp/v35_online_gpu_${lock_mask}.lock}"
  exec {V35_RUN_LOCK_FD}>"$V35_RUN_LOCK_FILE"
  if ! flock -n "$V35_RUN_LOCK_FD"; then
    echo "Another V35 launcher already owns GPU mask ${CUDA_VISIBLE_DEVICES}." >&2
    echo "lock_file=$V35_RUN_LOCK_FILE" >&2
    exit 2
  fi
  export V35_RUN_LOCK_FILE
  echo "run_lock_file=$V35_RUN_LOCK_FILE"
fi

mkdir -p \
  "$CHECKPOINT_DIR" \
  "$V35_RUNTIME_ROOT" \
  "$TENSORBOARD_DIR" \
  "$(dirname -- "$V35_MODE_PROB_DIAGNOSTIC_LOG")"

EGL_SHIM_SOURCE="${REPO_ROOT}/training/hpc/egl_device_enumeration_compat.c"
export V35_EGL_DEVICE_SHIM="${V35_EGL_DEVICE_SHIM:-${V35_RUNTIME_ROOT}/libegl_device_enumeration_compat.so}"
if [[ ! -f "$EGL_SHIM_SOURCE" ]]; then
  echo "Missing EGL device compatibility source: $EGL_SHIM_SOURCE" >&2
  exit 1
fi
if ! command -v gcc >/dev/null 2>&1; then
  echo "gcc is required to build the per-renderer EGL device compatibility shim" >&2
  exit 1
fi
gcc -shared -fPIC -O2 \
  "$EGL_SHIM_SOURCE" \
  -o "$V35_EGL_DEVICE_SHIM" \
  -ldl -pthread

# Some HPC containers expose CUDA devices but do not mount the NVIDIA
# graphics-capability libraries system-wide.  When the documented user-space
# NVIDIA/GLVND installation is present, pass it only to renderer actors.  Do
# not globally preload these libraries into VERL, PyTorch, or vLLM workers.
NVIDIA_DRIVER_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n 1 | tr -d '[:space:]')"
V35_NVIDIA_EGL_ROOT="${V35_NVIDIA_EGL_ROOT:-${HOME}/nvidia-gl-${NVIDIA_DRIVER_VERSION}}"
V35_NVIDIA_EGL_LIB="${V35_NVIDIA_EGL_ROOT}/usr/lib/x86_64-linux-gnu"
V35_NVIDIA_EGL_JSON="${V35_NVIDIA_EGL_ROOT}/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
V35_GLVND_LIB="${V35_GLVND_ROOT:-${HOME}/glvnd-system}/lib"
if [[ -f "${V35_NVIDIA_EGL_LIB}/libEGL_nvidia.so.${NVIDIA_DRIVER_VERSION}" && \
      -f "${V35_NVIDIA_EGL_JSON}" && \
      -f "${V35_GLVND_LIB}/libGLdispatch.so.0" ]]; then
  export V35_RENDER_LD_LIBRARY_PATH="${V35_GLVND_LIB}:${V35_NVIDIA_EGL_LIB}:/lib/x86_64-linux-gnu"
  export V35_RENDER_LD_PRELOAD="${V35_GLVND_LIB}/libGLdispatch.so.0:${V35_GLVND_LIB}/libEGL.so.1:${V35_GLVND_LIB}/libOpenGL.so.0:${V35_GLVND_LIB}/libGLX.so.0:${V35_GLVND_LIB}/libGL.so.1"
  export V35_RENDER_EGL_VENDOR_LIBRARY_FILENAMES="${V35_NVIDIA_EGL_JSON}"
  export V35_RENDER_GLX_VENDOR_LIBRARY_NAME=nvidia
  export V35_RENDER_EGL_PLATFORM=surfaceless
  echo "renderer_glvnd=user-space driver=${NVIDIA_DRIVER_VERSION} root=${V35_NVIDIA_EGL_ROOT}"
fi
if [[ "$V35_MODE_BERNOULLI_ROUTING" == "1" ]]; then
  "$PYTHON_BIN" - <<'PY'
from vllm.sampling_params import SamplingParams
SamplingParams(max_tokens=1, prompt_logprobs=0)
print("mode_probability_preflight=PASS")
PY
fi

if ((SKIP_PREFLIGHT == 0)); then
  PREFLIGHT_ARGS=(
    --config "$SCRIPT_DIR/online_sumo.yaml" \
    --verl-config "$VERL_ROOT/verl/trainer/config/online_cooperative_grpo.yaml"
  )
  if ((CHECK_ENDPOINT == 1)); then
    PREFLIGHT_ARGS+=(--check-endpoint)
  fi
  if [[ -n "${V35_MODEL_DIR:-}" ]]; then
    PREFLIGHT_ARGS+=(--model "$V35_MODEL_DIR")
  fi
  "$PYTHON_BIN" "$SCRIPT_DIR/preflight_online_cooperative.py" "${PREFLIGHT_ARGS[@]}"
fi

if ((SMOKE == 1)); then
  # Add smoke defaults only when the caller did not provide that key. Hydra
  # uses the last duplicate override, so appending unconditional defaults
  # would silently replace explicit batch-size settings.
  smoke_defaults=(
    "trainer.total_epochs=1"
    "trainer.experiment_name=smoke"
    "trainer.logger=[console,tensorboard]"
    "trainer.val_before_train=false"
    # The distributed trainer requires a global batch divisible by the four
    # data-parallel ranks used by the default launcher.
    "data.train_batch_size=4"
    "data.gen_batch_size=4"
    "data.train_max_samples=4"
    # Make smoke validation a real 10-sample suite, rather than merely using
    # a dataloader batch size of ten while iterating all fifty validation rows.
    "data.val_max_samples=10"
  )
  for default in "${smoke_defaults[@]}"; do
    key="${default%%=*}"
    found=0
    for override in "${HYDRA_OVERRIDES[@]}"; do
      [[ "${override%%=*}" == "$key" ]] && found=1 && break
    done
    if ((found == 0)); then
      HYDRA_OVERRIDES+=("$default")
    fi
  done
fi

if ((FRESH == 1)); then
  HYDRA_OVERRIDES+=("trainer.resume_mode=disable")
fi

# Use BF16 for the online actor weights as well as its autocast compute.  The
# global FSDP default remains FP32 for other jobs; this override is scoped to
# this online actor and can still be replaced explicitly by the caller.
actor_precision_defaults=(
  "actor_rollout_ref.actor.fsdp_config.model_dtype=bf16"
)
for default in "${actor_precision_defaults[@]}"; do
  key="${default%%=*}"
  found=0
  for override in "${HYDRA_OVERRIDES[@]}"; do
    [[ "${override%%=*}" == "$key" ]] && found=1 && break
  done
  if ((found == 0)); then
    HYDRA_OVERRIDES+=("$default")
  fi
done

# Timestamp all run-owned output unless the caller explicitly supplies a
# Hydra path.  V35_RUN_ID may be reused deliberately when resuming a run.
run_path_defaults=(
  "trainer.default_local_dir=${CHECKPOINT_DIR}"
  "trainer.rollout_data_dir=${CHECKPOINT_DIR}/rollouts"
  "trainer.validation_data_dir=${CHECKPOINT_DIR}/validation"
)
for default in "${run_path_defaults[@]}"; do
  key="${default%%=*}"
  found=0
  for override in "${HYDRA_OVERRIDES[@]}"; do
    [[ "${override%%=*}" == "$key" ]] && found=1 && break
  done
  if ((found == 0)); then
    HYDRA_OVERRIDES+=("$default")
  fi
done

echo "online_grpo_root=$VERL_ROOT"
echo "run_id=$V35_RUN_ID"
echo "checkpoint_dir=$CHECKPOINT_DIR"
echo "runtime_dir=$V35_RUNTIME_ROOT"
echo "tensorboard_dir=$TENSORBOARD_DIR"
echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES nproc_per_node=$NPROC_PER_NODE"
echo "render_egl_devices=$V35_RENDER_EGL_DEVICES"
echo "render_slots_per_gpu=$V35_RENDER_SLOTS_PER_GPU"
echo "egl_device_shim=$V35_EGL_DEVICE_SHIM"
echo "objectives=stage1_sft:${V35_PERCEPTION_SFT_COEF} mode_gdpo:${V35_MODE_SELECTOR_COEF} decision_grpo:1"
echo "mode_gdpo_reward_keys=$V35_MODE_GDPO_REWARD_KEYS"
echo "launch=python -m verl.trainer.main_ppo --config-name=online_cooperative_grpo ${HYDRA_OVERRIDES[*]-}"
RUN_LOG="${CHECKPOINT_DIR}/run.log"
RUN_MANIFEST="${CHECKPOINT_DIR}/run_manifest.txt"
{
  echo "run_id=$V35_RUN_ID"
  echo "started_at=$(date --iso-8601=seconds)"
  echo "checkpoint_dir=$CHECKPOINT_DIR"
  echo "runtime_dir=$V35_RUNTIME_ROOT"
  echo "tensorboard_dir=$TENSORBOARD_DIR"
  echo "console_log=$RUN_LOG"
  echo "hydra_overrides=${HYDRA_OVERRIDES[*]-}"
} > "$RUN_MANIFEST"
echo "run_manifest=$RUN_MANIFEST"
echo "console_log=$RUN_LOG"

"$PYTHON_BIN" -m verl.trainer.main_ppo \
  --config-name=online_cooperative_grpo \
  "${HYDRA_OVERRIDES[@]}" \
  2>&1 | tee "$RUN_LOG"
