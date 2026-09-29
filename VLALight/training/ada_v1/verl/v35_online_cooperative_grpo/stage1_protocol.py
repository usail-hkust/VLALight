"""Formal multimodal Stage 1 prompt construction and perception parsing."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from .online_rollout import CitySnapshot


_PHASE_MOVEMENTS = {
    "ETWT": ("ET", "WT"), "NTST": ("NT", "ST"),
    "ELWL": ("EL", "WL"), "NLSL": ("NL", "SL"),
}


@lru_cache(maxsize=1)
def _baseline_messages() -> tuple[str, str]:
    source = (
        Path(__file__).resolve().parents[4]
        / "sft_v35_local_perception_dataset_reduced_pixels"
        / "train.json"
    )
    rows = json.loads(source.read_text(encoding="utf-8"))
    messages = rows[0]["messages"]
    return str(messages[0]["content"]), str(messages[1]["content"])


def build_stage1_messages(
    *,
    intersection_id: str,
    current_phase: str,
    ages: Mapping[str, Any],
    videos: Sequence[str],
    coordination_frames: Sequence[tuple[str, str]],
) -> list[dict[str, Any]]:
    """Build the exact SFT prompt with concrete online media paths."""
    if len(videos) != 4:
        raise ValueError("Stage 1 requires E, W, N, S videos in that order")
    system, user = _baseline_messages()
    user = re.sub(r"Intersection: .*", f"Intersection: {intersection_id}", user, count=1)
    user = re.sub(r"Current phase: .*", f"Current phase: {current_phase}", user, count=1)
    for phase in ("ETWT", "NTST", "ELWL", "NLSL"):
        user = re.sub(
            rf"(?m)^- {phase}: .*?$", f"- {phase}: {int(ages.get(phase, 0))}", user, count=1
        )
    labels = [str(direction).upper() for direction, _ in coordination_frames]
    user = re.sub(
        r"<image>(?:<image>)*", "<image>" * len(coordination_frames), user, count=1
    )
    user = re.sub(
        r"Coordination frame inputs, in this exact order: .*?\.",
        "Coordination frame inputs, in this exact order: " + ", ".join(labels) + ".",
        user,
        count=1,
    )

    media = iter(
        [{"type": "video", "video": str(path)} for path in videos]
        + [{"type": "image", "image": str(path)} for _, path in coordination_frames]
    )
    content: list[dict[str, Any]] = []
    parts = re.split(r"(<video>|<image>)", user)
    for part in parts:
        if part in {"<video>", "<image>"}:
            content.append(next(media))
        elif part:
            content.append({"type": "text", "text": part})
    try:
        next(media)
    except StopIteration:
        pass
    else:
        raise RuntimeError("Stage 1 media placeholders do not match supplied paths")
    return [{"role": "system", "content": system}, {"role": "user", "content": content}]


def parse_perception(text: str) -> dict[str, Any]:
    matches = re.findall(r"<perception>\s*(\{.*?\})\s*</perception>", text or "", re.S)
    if len(matches) != 1:
        raise ValueError("Stage 1 output must contain exactly one perception block")
    value = json.loads(matches[0])
    if not isinstance(value, dict) or set(value.get("phases", {})) != {"ETWT", "NTST", "ELWL", "NLSL"}:
        raise ValueError("Stage 1 perception has an invalid phase schema")
    return value


def apply_stage1_perceptions(
    snapshot: CitySnapshot, perceptions: Mapping[str, Mapping[str, Any]]
) -> CitySnapshot:
    """Replace every local observation only after the full city batch exists."""
    expected = {row.intersection_id for row in snapshot.observations}
    actual = {str(key) for key in perceptions}
    if actual != expected:
        raise ValueError(
            f"Stage 1 city barrier mismatch: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )
    def movement_value(local: Mapping[str, Any], movement: str) -> dict[str, Any]:
        for phase, names in _PHASE_MOVEMENTS.items():
            if movement not in names:
                continue
            phase_row = (local.get("phases") or {}).get(phase) or {}
            index = names.index(movement)
            values_v, values_q = phase_row.get("v") or (), phase_row.get("q") or ()
            return {
                "v": values_v[index] if index < len(values_v) else 0,
                "q": values_q[index] if index < len(values_q) else 0,
            }
        raise ValueError(f"unknown movement {movement!r}")

    observations = []
    for row in snapshot.observations:
        local = dict(perceptions[row.intersection_id])
        if local.get("current_phase") != row.current_phase:
            raise ValueError(
                f"Stage 1 changed current phase for {row.intersection_id}: "
                f"{local.get('current_phase')!r} != {row.current_phase!r}"
            )
        skeleton = row.cooperative_perception if isinstance(row.cooperative_perception, Mapping) else {}
        neighbors = {}
        for side, metadata in (skeleton.get("neighbors") or {}).items():
            metadata = dict(metadata)
            source_id = str(metadata.get("source_intersection"))
            if source_id not in perceptions:
                raise ValueError(f"router is missing Stage 1 perception for {source_id!r}")
            movement_names = tuple((metadata.get("upstream_movements") or {}).keys())
            metadata["upstream_movements"] = {
                movement: movement_value(perceptions[source_id], movement)
                for movement in movement_names
            }
            neighbors[str(side)] = metadata
        coordination = {
            phase: dict(((local.get("phases") or {}).get(phase) or {}).get("coord") or {})
            for phase in _PHASE_MOVEMENTS
        }
        cooperative = {"local_coordination": coordination, "neighbors": neighbors}
        # Stage 1 predicts coordination inside each phase, whereas the formal
        # Stage 2 contract carries it exclusively in cooperative_perception.
        # Remove it from local_perception after routing to avoid presenting
        # the same evidence twice and to match SUMO-built temporal snapshots.
        local = {
            "current_phase": local.get("current_phase"),
            "phases": {
                phase: {
                    key: value
                    for key, value in dict((local.get("phases") or {}).get(phase) or {}).items()
                    if key != "coord"
                }
                for phase in _PHASE_MOVEMENTS
            },
        }
        observations.append(replace(
            row, local_perception=local, cooperative_perception=cooperative
        ))
    return replace(snapshot, observations=tuple(observations))


def coordination_sources(city: str, target_id: str) -> list[tuple[str, str, str, float | None]]:
    """Return entry, source, camera and route distance for each upstream feed."""
    city_key = "jinan" if city.lower().startswith("jinan") else "hangzhou"
    path = Path(__file__).resolve().parent / "artifacts" / f"movement_routes_{city_key}.json"
    # Reuse the online route loader so generated artifacts receive the same
    # topology-derived distance/travel-time enrichment as temporal prompts.
    from .observation_builder import _load_routes
    routes = _load_routes(city_key, path.parent)["routes"]
    by_entry: dict[str, tuple[str, str, str, float | None]] = {}
    for source_id, source in routes.items():
        for route in (source.get("movements") or {}).values():
            receiver = route.get("receiver_id")
            entry = str(route.get("receiver_entry_direction") or "").upper()
            camera = str(route.get("exit_direction") or "").upper()
            if str(receiver) == target_id and entry and camera:
                distance = route.get("distance_m")
                try:
                    distance = float(distance) if distance is not None else None
                except (TypeError, ValueError):
                    distance = None
                by_entry.setdefault(entry, (entry, str(source_id), camera, distance))
    return [by_entry[key] for key in ("E", "W", "N", "S") if key in by_entry]


def extract_coordination_frames(
    *, city: str, target_id: str, video_paths: Mapping[str, Mapping[str, str]], output_dir: str | Path
) -> list[tuple[str, str]]:
    """Extract the last synchronized frame from each routed upstream video."""
    import cv2

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    result: list[tuple[str, str]] = []
    started = time.perf_counter()
    for entry, source_id, camera, distance_m in coordination_sources(city, target_id):
        frame_started = time.perf_counter()
        source = (video_paths.get(source_id) or {}).get(camera)
        if not source:
            raise RuntimeError(f"missing upstream video {source_id}/{camera} for {target_id}/{entry}")
        capture = cv2.VideoCapture(str(source))
        frame = None
        selected_index = None
        if distance_m is not None and distance_m > 0:
            from utils.coordination_frame_selector import select_coordination_frame
            selected, _ = select_coordination_frame(distance_m)
            selected_index = max(1, int(selected.frame_index))
        try:
            index = 0
            while True:
                ok, candidate = capture.read()
                if not ok:
                    break
                frame = candidate
                index += 1
                if selected_index is not None and index >= selected_index:
                    break
        finally:
            capture.release()
        if frame is None:
            raise RuntimeError(f"cannot decode upstream video: {source}")
        # Read the source intersection's exit camera, but label the image by
        # the direction from which that flow enters the target intersection.
        # For example, a source W camera feeds the target's E entry and must
        # supervise ET/EL rather than WT/WL.
        target = root / f"coordination_frame_{entry}.jpg"
        if not cv2.imwrite(str(target), frame):
            raise RuntimeError(f"cannot write coordination frame: {target}")
        result.append((entry, str(target)))
        if os.environ.get("V35_VERBOSE_DEBUG", "0") == "1":
            print(
                "[STAGE1_TIMING] coordination_frame "
                f"target={target_id} entry={entry} source={source_id} camera={camera} "
                f"frame_index={selected_index or index} distance_m={distance_m} "
                f"elapsed_s={time.perf_counter() - frame_started:.3f}",
                flush=True,
            )
    if os.environ.get("V35_VERBOSE_DEBUG", "0") == "1":
        print(
            f"[STAGE1_TIMING] coordination_frames target={target_id} "
            f"count={len(result)} elapsed_s={time.perf_counter() - started:.3f}",
            flush=True,
        )
    return result


__all__ = [
    "apply_stage1_perceptions", "build_stage1_messages", "coordination_sources",
    "extract_coordination_frames", "parse_perception",
]
