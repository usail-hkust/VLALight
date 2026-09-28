#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
cd "$(dirname "$0")"
# Keep torch.distributed tied to the currently activated Python environment.
# A bare `torchrun` can otherwise resolve to verl_qwen35 (Python 3.10), while
# this LlamaFactory revision requires Python 3.11+ for enum.StrEnum.
python -m torch.distributed.run --nproc_per_node="$NPROC_PER_NODE" src/train.py examples/train_lora/qwen35_4b_v35_perception_temporal_hpc_lora.yaml "$@"
