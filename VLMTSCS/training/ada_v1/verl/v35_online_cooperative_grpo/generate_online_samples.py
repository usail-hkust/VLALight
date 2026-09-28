#!/usr/bin/env python3
"""Generate bootstrap JSONL rows for the live SUMO Stage 2 stream.

These rows only satisfy the trainer's input contract.  The actual prompt and
perception are materialized from each persistent SUMO master by the online
runtime at rollout time.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PACKAGE_PARENT = Path(__file__).resolve().parents[1]
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from v35_online_cooperative_grpo.online_config import load_online_config  # noqa: E402
from v35_online_cooperative_grpo.online_samples import write_fake_jsonl  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("online_sumo.yaml"),
        help="online SUMO YAML configuration",
    )
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.count is not None and args.count < 0:
        parser.error("--count must be non-negative")
    config = load_online_config(args.config)
    count = write_fake_jsonl(config, args.output, split=args.split, count=args.count)
    print(f"wrote={count} split={args.split} output={args.output}")


if __name__ == "__main__":
    main()
