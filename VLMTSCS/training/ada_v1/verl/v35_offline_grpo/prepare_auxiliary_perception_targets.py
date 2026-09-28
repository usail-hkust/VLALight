"""Materialize boundary and persistent-history targets in V35 GRPO JSONL."""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any


PHASE_MOVEMENTS = {
    "ETWT": ("ET", "WT"),
    "NTST": ("NT", "ST"),
    "ELWL": ("EL", "WL"),
    "NLSL": ("NL", "SL"),
}
FRAME_MOVEMENTS = {
    "W": ("ET", "EL"),
    "E": ("WT", "WL"),
    "S": ("NT", "NL"),
    "N": ("ST", "SL"),
}
HISTORY_PATTERN = re.compile(r"- (ETWT|NTST|ELWL|NLSL):\s*(\d+)")
FRAME_PATTERN = re.compile(r"coordination_frame_([EWNS])_")


def _prompt_history(prompt: list[dict[str, Any]]) -> dict[str, int]:
    user_text = "\n".join(
        str(message.get("content", ""))
        for message in prompt
        if message.get("role") == "user"
    )
    history = {phase: int(value) for phase, value in HISTORY_PATTERN.findall(user_text)}
    if set(history) != set(PHASE_MOVEMENTS):
        raise ValueError(f"expected four history values, got {history}")
    return history


def _boundary_targets(images: list[str]) -> dict[str, str]:
    directions = {
        match.group(1)
        for image in images
        if (match := FRAME_PATTERN.search(str(image))) is not None
    }
    return {
        movement: "no" if direction in directions else "yes"
        for direction, movements in FRAME_MOVEMENTS.items()
        for movement in movements
    }


def _augment(row: dict[str, Any]) -> bool:
    reward_model = row.get("reward_model")
    if not isinstance(reward_model, dict):
        raise ValueError("missing reward_model")
    raw_ground_truth = reward_model.get("ground_truth")
    ground_truth = json.loads(raw_ground_truth) if isinstance(raw_ground_truth, str) else raw_ground_truth
    if not isinstance(ground_truth, dict):
        raise ValueError("invalid reward_model.ground_truth")
    target = ground_truth.get("perception_target")
    if not isinstance(target, dict) or set(target) != set(PHASE_MOVEMENTS):
        raise ValueError("invalid perception_target phases")

    history = _prompt_history(row.get("prompt", []))
    boundaries = _boundary_targets(row.get("images", []))
    changed = False
    for phase, movements in PHASE_MOVEMENTS.items():
        phase_target = target[phase]
        if phase_target.get("nonzero_v_history_length_since_last_service") != history[phase]:
            phase_target["nonzero_v_history_length_since_last_service"] = history[phase]
            changed = True
        breakdown = phase_target.get("coordinated_arrivals", {}).get("breakdown")
        if not isinstance(breakdown, dict):
            raise ValueError(f"missing coordinated breakdown for {phase}")
        for movement in movements:
            entry = breakdown.get(movement)
            if not isinstance(entry, dict):
                raise ValueError(f"missing breakdown entry for {phase}.{movement}")
            if entry.get("is_boundary") != boundaries[movement]:
                entry["is_boundary"] = boundaries[movement]
                changed = True

    reward_model["ground_truth"] = (
        json.dumps(ground_truth, ensure_ascii=False, separators=(",", ":"))
        if isinstance(raw_ground_truth, str)
        else ground_truth
    )
    return changed


def _process(path: Path, in_place: bool) -> tuple[int, int]:
    changed = 0
    total = 0
    with path.open(encoding="utf-8") as source:
        if not in_place:
            for line_number, line in enumerate(source, 1):
                if line.strip():
                    _augment(json.loads(line))
                    total += 1
            return total, changed

        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False, newline="\n"
        ) as destination:
            temporary_path = Path(destination.name)
            try:
                for line_number, line in enumerate(source, 1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    changed += _augment(row)
                    destination.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                    total += 1
                destination.flush()
                os.fsync(destination.fileno())
            except Exception:
                temporary_path.unlink(missing_ok=True)
                raise
    os.replace(temporary_path, path)
    return total, changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--in-place", action="store_true")
    args = parser.parse_args()
    for path in args.paths:
        total, changed = _process(path, args.in_place)
        print(json.dumps({"path": str(path), "records": total, "changed": changed, "in_place": args.in_place}))


if __name__ == "__main__":
    main()
