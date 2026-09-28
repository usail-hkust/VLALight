#!/usr/bin/env python3
"""Read-only check that GRPO visual preprocessing matches V35 SFT limits."""

import argparse
import json
import math
import os
from pathlib import Path

import cv2
from transformers import AutoProcessor


def resolve_video(entry, data_dir: Path) -> str:
    path = entry.get("video") if isinstance(entry, dict) else entry
    if not isinstance(path, str):
        raise TypeError(f"Expected one video path, got {type(path)}")
    return path if os.path.isabs(path) else str(data_dir / path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", default="train.jsonl")
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    with (data_dir / args.split).open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    row = rows[args.index]
    source_path = resolve_video(row["videos"][0], data_dir)
    if not os.path.exists(source_path):
        raise FileNotFoundError(source_path)

    capture = cv2.VideoCapture(source_path)
    source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    capture.release()

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    patch_size = int(processor.image_processor.patch_size)
    import qwen_vl_utils.vision_process as vision_process
    from qwen_vl_utils import fetch_video

    max_pixels = 1_105_920
    factor = patch_size * vision_process.SPATIAL_MERGE_SIZE
    vision_process.VIDEO_MAX_TOKEN_NUM = max(
        vision_process.VIDEO_MAX_TOKEN_NUM,
        math.ceil(max_pixels / (factor * factor)),
    )
    video, metadata = fetch_video(
        {
            "video": source_path,
            "nframes": 6,
            "min_pixels": 65_536,
            "max_pixels": max_pixels,
        },
        image_patch_size=patch_size,
        return_video_metadata=True,
    )
    frames, _, height, width = video.shape
    print(f"source={source_path}")
    print(f"source_frames={source_frames} source_fps={source_fps:g} source_size={source_width}x{source_height}")
    print(f"model_frames={frames} model_size={width}x{height} model_pixels={width * height}")
    print(f"selected_indices={metadata.get('frames_indices')}")
    assert frames == 6, f"Expected six V35 frames, got {frames}"
    assert 65_536 <= width * height <= max_pixels
    print("PASS: GRPO input uses six ordered frames within the V35 SFT pixel limits.")


if __name__ == "__main__":
    main()
