#!/usr/bin/env python3
"""Create deterministic city-stratified GRPO train/validation subsets."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def _allocate(counts: dict[str, int], target: int) -> dict[str, int]:
    total = sum(counts.values())
    exact = {city: target * count / total for city, count in counts.items()}
    allocated = {city: int(value) for city, value in exact.items()}
    remaining = target - sum(allocated.values())
    order = sorted(counts, key=lambda city: (exact[city] - allocated[city], counts[city]), reverse=True)
    for city in order[:remaining]:
        allocated[city] += 1
    return allocated


def _make_subset(source: Path, destination: Path, target: int, seed: int) -> dict[str, int]:
    groups: dict[str, list[dict]] = defaultdict(list)
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            city = row.get("extra_info", {}).get("city", "unknown")
            groups[city].append(row)

    allocation = _allocate({city: len(rows) for city, rows in groups.items()}, target)
    rng = random.Random(seed)
    selected = []
    for city in sorted(groups):
        rows = groups[city]
        rng.shuffle(rows)
        selected.extend(rows[: allocation[city]])
    rng.shuffle(selected)

    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return allocation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--train-size", type=int, default=2000)
    parser.add_argument("--val-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_name = f"train_{args.train_size}.jsonl"
    val_name = f"val_{args.val_size}.jsonl"
    train_counts = _make_subset(
        args.data_dir / "train.jsonl", args.data_dir / train_name, args.train_size, args.seed
    )
    val_counts = _make_subset(
        args.data_dir / "val.jsonl", args.data_dir / val_name, args.val_size, args.seed + 1
    )
    print(f"created {train_name}: {train_counts}")
    print(f"created {val_name}: {val_counts}")


if __name__ == "__main__":
    main()
