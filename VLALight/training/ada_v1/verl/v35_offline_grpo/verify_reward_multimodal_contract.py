#!/usr/bin/env python3
"""Preflight the V35 GRPO reward-time Qwen multimodal reconstruction.

This script runs no rollout and no training.  It verifies that the exact
single-turn agent-loop path reuses the original chat-template text instead of
trying to reconstruct media placeholders from patch-expanded prompt IDs.
"""

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

# Executing this file directly otherwise puts only ``v35_offline_grpo`` on
# sys.path, which can accidentally import an older site-packages ``verl``.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from transformers import AutoProcessor

from verl.experimental.agent_loop.agent_loop import AgentLoopWorker
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.tokenizer import build_multimodal_processor_inputs
from verl.utils.tokenizer.chat_template import apply_chat_template


def build_messages(row: dict, processor, data_dir: Path) -> list[dict]:
    """Use the dataset's production placeholder replacement without loading a dataset."""
    builder = object.__new__(RLHFDataset)
    builder.image_key = "images"
    builder.video_key = "videos"
    builder.audio_key = "audio"
    builder.processor = processor
    builder.config = {
        "image_min_pixels": 65_536,
        "image_max_pixels": 1_105_920,
        "video_min_pixels": 65_536,
        "video_max_pixels": 1_105_920,
        "video_nframes": 6,
    }
    os.environ["V35_GRPO_DATA_DIR"] = str(data_dir)
    return builder._build_messages(copy.deepcopy(row), key="prompt")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", default="train.jsonl")
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    rows = [json.loads(line) for line in (data_dir / args.split).read_text(encoding="utf-8").splitlines() if line]
    row = rows[args.index]
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    config = {
        "image_min_pixels": 65_536,
        "image_max_pixels": 1_105_920,
        "video_min_pixels": 65_536,
        "video_max_pixels": 1_105_920,
        "video_nframes": 6,
    }

    messages = build_messages(row, processor, data_dir)
    image_patch_size = int(processor.image_processor.patch_size)
    images, videos, audios = RLHFDataset._process_multi_modal_info(
        messages, image_patch_size=image_patch_size, config=config
    )
    raw_prompt = apply_chat_template(processor, messages, tokenize=False, add_generation_prompt=True)
    first_inputs = build_multimodal_processor_inputs(
        processor, text=[raw_prompt], images=images, videos=videos, audio=audios, mm_processor_kwargs={}
    )

    expected_videos = len(videos or [])
    actual_grids = int(first_inputs["video_grid_thw"].shape[0])
    assert actual_grids == expected_videos, (actual_grids, expected_videos)

    output = SimpleNamespace(
        processor_prompt=raw_prompt,
        prompt_ids=first_inputs["input_ids"][0].tolist(),
        response_ids=[],
        multi_modal_data={"images": images, "videos": videos, "audios": audios},
        mm_processor_kwargs={},
    )
    worker = SimpleNamespace(processor=processor, tokenizer=processor.tokenizer)
    worker._get_mm_processor_kwargs = lambda _audios: {}
    reconstructed = AgentLoopWorker._compute_multi_modal_inputs(
        worker,
        output,
        torch.tensor([output.prompt_ids + output.response_ids], dtype=torch.long),
    )
    assert int(reconstructed["video_grid_thw"].shape[0]) == expected_videos
    torch.testing.assert_close(reconstructed["video_grid_thw"], first_inputs["video_grid_thw"])

    full_ids = torch.tensor([output.prompt_ids + output.response_ids], dtype=torch.long)
    position_ids = AgentLoopWorker._compute_position_ids(
        worker,
        full_ids,
        torch.ones_like(full_ids),
        reconstructed,
    )
    assert position_ids.shape[-1] == full_ids.shape[-1]

    print(f"sample={row.get('extra_info', {}).get('index', args.index)}")
    print(f"videos={expected_videos} images={len(images or [])} video_grids={actual_grids}")
    print("PASS: reward-time reconstruction and Qwen M-RoPE use per-frame video grids.")


if __name__ == "__main__":
    main()
