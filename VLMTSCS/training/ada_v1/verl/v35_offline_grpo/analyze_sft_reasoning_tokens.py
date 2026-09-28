#!/usr/bin/env python3
"""Measure SFT slow-reasoning lengths with the exact training tokenizer."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path


PERCENTILES = (0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100)


def percentile(values: list[int], percent: int) -> int:
    index = round((len(values) - 1) * percent / 100)
    return values[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--files", nargs="+", default=["train.json", "val.json"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    token_lengths: list[int] = []
    file_counts: dict[str, int] = {}
    for name in args.files:
        rows = json.loads((args.dataset_dir / name).read_text(encoding="utf-8"))
        count = 0
        for row in rows:
            response = next(message["content"] for message in row["messages"] if message["role"] == "assistant")
            mode_match = re.search(r"<mode>\s*(.*?)\s*</mode>", response, re.I | re.S)
            if mode_match is None or mode_match.group(1).strip().lower() != "slow":
                continue
            reasoning_match = re.search(r"<reasoning>\s*(.*?)\s*</reasoning>", response, re.I | re.S)
            if reasoning_match is None or not reasoning_match.group(1).strip():
                raise ValueError(f"Slow sample without reasoning in {name}")
            length = len(tokenizer(reasoning_match.group(1).strip(), add_special_tokens=False)["input_ids"])
            token_lengths.append(length)
            count += 1
        file_counts[name] = count

    token_lengths.sort()
    report = {
        "model_path": args.model_path,
        "dataset_dir": str(args.dataset_dir),
        "slow_samples": len(token_lengths),
        "file_counts": file_counts,
        "tokens": {
            **{f"p{p:02d}": percentile(token_lengths, p) for p in PERCENTILES},
            "mean": statistics.mean(token_lengths),
            "std": statistics.pstdev(token_lengths),
        },
    }
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
