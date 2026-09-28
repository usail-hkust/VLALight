import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


DIRECTIONS: Tuple[str, ...] = ("N", "E", "W", "S")


def _read_image_chinese_safe(image_path: Path) -> np.ndarray:
    """Read an image from disk with Chinese-path compatibility."""
    data = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to decode image: {image_path}")
    return image


def _write_image_chinese_safe(image_path: Path, image: np.ndarray) -> None:
    """Write an image to disk with Chinese-path compatibility."""
    image_path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(image_path.suffix or ".jpg", image)
    if not success:
        raise ValueError(f"Failed to encode image for: {image_path}")
    encoded.tofile(str(image_path))


@dataclass
class FrameRecord:
    step: int
    sim_time_sec: float
    paths: Dict[str, Path]


def _resolve_images_root(session_dir: Path) -> Path:
    images_dir = session_dir / "images"
    if images_dir.exists():
        return images_dir
    return session_dir


def _discover_step_dirs(images_root: Path) -> List[Tuple[int, Path]]:
    step_dirs: List[Tuple[int, Path]] = []
    for child in images_root.iterdir():
        if not child.is_dir():
            continue
        name = child.name
        if not name.startswith("step_"):
            continue
        try:
            step_num = int(name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        step_dirs.append((step_num, child))
    return sorted(step_dirs, key=lambda item: item[0])


def _discover_intersections(images_root: Path, step_dirs: Sequence[Tuple[int, Path]]) -> List[str]:
    intersections = set()
    for _, step_dir in step_dirs:
        for child in step_dir.iterdir():
            if child.is_dir():
                intersections.add(child.name)
    return sorted(intersections)


def _collect_intersection_frames(
    images_root: Path,
    step_dirs: Sequence[Tuple[int, Path]],
    intersection_id: str,
    sim_step_sec: float,
) -> List[FrameRecord]:
    frames: List[FrameRecord] = []
    for step_num, step_dir in step_dirs:
        inter_dir = step_dir / intersection_id
        if not inter_dir.exists():
            continue
        paths: Dict[str, Path] = {}
        missing = False
        for direction in DIRECTIONS:
            img_path = inter_dir / f"{direction}.jpg"
            if not img_path.exists():
                missing = True
                break
            paths[direction] = img_path
        if missing:
            continue
        frames.append(
            FrameRecord(
                step=step_num,
                sim_time_sec=step_num * sim_step_sec,
                paths=paths,
            )
        )
    return frames


def _compose_2x2_frame(images: Dict[str, np.ndarray], pad: int = 8) -> np.ndarray:
    n = images["N"]
    e = images["E"]
    w = images["W"]
    s = images["S"]

    if n.shape != e.shape or n.shape != w.shape or n.shape != s.shape:
        raise ValueError("All directional images must have the same shape to build a 2x2 composite.")

    height, width = n.shape[:2]
    canvas = np.full((height * 2 + pad * 3, width * 2 + pad * 3, 3), 32, dtype=np.uint8)
    canvas[pad:pad + height, pad:pad + width] = n
    canvas[pad:pad + height, width + pad * 2:width * 2 + pad * 2] = e
    canvas[height + pad * 2:height * 2 + pad * 2, pad:pad + width] = w
    canvas[height + pad * 2:height * 2 + pad * 2, width + pad * 2:width * 2 + pad * 2] = s
    return canvas


def _open_video_writer(video_path: Path, frame_size: Tuple[int, int], fps: float) -> cv2.VideoWriter:
    video_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, frame_size)
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {video_path}")
    return writer


def _build_intersection_videos(
    session_dir: Path,
    intersection_id: str,
    frames: Sequence[FrameRecord],
    output_root: Path,
    fps: float,
    export_direction_videos: bool,
    export_composite_frames: bool,
) -> Dict[str, object]:
    if not frames:
        raise ValueError(f"No frames found for intersection: {intersection_id}")

    first_images = {direction: _read_image_chinese_safe(frames[0].paths[direction]) for direction in DIRECTIONS}
    first_comp = _compose_2x2_frame(first_images)
    comp_h, comp_w = first_comp.shape[:2]

    composite_writer = _open_video_writer(
        output_root / "videos" / f"{intersection_id}_2x2.mp4",
        (comp_w, comp_h),
        fps,
    )

    direction_writers: Dict[str, cv2.VideoWriter] = {}
    if export_direction_videos:
        dir_h, dir_w = first_images["N"].shape[:2]
        for direction in DIRECTIONS:
            direction_writers[direction] = _open_video_writer(
                output_root / "videos" / f"{intersection_id}_{direction}.mp4",
                (dir_w, dir_h),
                fps,
            )

    frame_manifest = []
    try:
        for idx, frame in enumerate(frames):
            images = {direction: _read_image_chinese_safe(frame.paths[direction]) for direction in DIRECTIONS}
            composite = _compose_2x2_frame(images)
            composite_writer.write(composite)

            composite_frame_path: Optional[Path] = None
            if export_composite_frames:
                composite_frame_path = (
                    output_root
                    / "composite_frames"
                    / intersection_id
                    / f"step_{frame.step:04d}.jpg"
                )
                _write_image_chinese_safe(composite_frame_path, composite)

            for direction, writer in direction_writers.items():
                writer.write(images[direction])

            frame_manifest.append(
                {
                    "frame_index": idx,
                    "step_index": frame.step,
                    "sim_time_sec": frame.sim_time_sec,
                    "direction_images": {
                        direction: os.path.relpath(str(frame.paths[direction]), str(session_dir))
                        for direction in DIRECTIONS
                    },
                    "composite_frame": (
                        os.path.relpath(str(composite_frame_path), str(output_root))
                        if composite_frame_path is not None else None
                    ),
                }
            )
    finally:
        composite_writer.release()
        for writer in direction_writers.values():
            writer.release()

    return {
        "intersection_id": intersection_id,
        "fps": fps,
        "frame_count": len(frames),
        "first_step": frames[0].step,
        "last_step": frames[-1].step,
        "sim_start_sec": frames[0].sim_time_sec,
        "sim_end_sec": frames[-1].sim_time_sec,
        "composite_video": os.path.relpath(
            str(output_root / "videos" / f"{intersection_id}_2x2.mp4"),
            str(output_root),
        ),
        "direction_videos": {
            direction: os.path.relpath(
                str(output_root / "videos" / f"{intersection_id}_{direction}.mp4"),
                str(output_root),
            )
            for direction in direction_writers
        },
        "frames": frame_manifest,
    }


def build_video_dataset(
    session_dir: str,
    sim_step_sec: float = 5.0,
    fps: float = 2.0,
    export_direction_videos: bool = True,
    export_composite_frames: bool = True,
    intersections: Optional[Sequence[str]] = None,
) -> Path:
    session_path = Path(session_dir).resolve()
    images_root = _resolve_images_root(session_path)
    step_dirs = _discover_step_dirs(images_root)
    if not step_dirs:
        raise FileNotFoundError(f"No step_* directories found under: {images_root}")

    if intersections is None:
        target_intersections = _discover_intersections(images_root, step_dirs)
    else:
        target_intersections = sorted(intersections)

    if not target_intersections:
        raise ValueError(f"No intersections found under: {images_root}")

    output_root = session_path / "video_export"
    output_root.mkdir(parents=True, exist_ok=True)

    dataset_manifest = {
        "session_dir": str(session_path),
        "images_root": str(images_root),
        "sim_step_sec": sim_step_sec,
        "fps": fps,
        "frame_time_sec": 1.0 / fps,
        "intersections": [],
    }

    for intersection_id in target_intersections:
        frames = _collect_intersection_frames(images_root, step_dirs, intersection_id, sim_step_sec)
        if not frames:
            continue
        manifest = _build_intersection_videos(
            session_dir=session_path,
            intersection_id=intersection_id,
            frames=frames,
            output_root=output_root,
            fps=fps,
            export_direction_videos=export_direction_videos,
            export_composite_frames=export_composite_frames,
        )
        dataset_manifest["intersections"].append(manifest)

    if not dataset_manifest["intersections"]:
        raise ValueError("No usable intersection frame sequences were found.")

    metadata_path = output_root / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(dataset_manifest, f, ensure_ascii=False, indent=2)

    return metadata_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build per-intersection videos and metadata from saved TransSimHub frame folders."
    )
    parser.add_argument("session_dir", help="Session directory containing images/step_xxxx/... frames.")
    parser.add_argument("--sim-step-sec", type=float, default=5.0, help="Simulation time advanced by one saved step.")
    parser.add_argument("--fps", type=float, default=2.0, help="Output video FPS.")
    parser.add_argument(
        "--intersection",
        action="append",
        dest="intersections",
        help="Specific intersection ID to export. Can be used multiple times.",
    )
    parser.add_argument(
        "--no-direction-videos",
        action="store_true",
        help="Disable exporting per-direction videos.",
    )
    parser.add_argument(
        "--no-composite-frames",
        action="store_true",
        help="Disable exporting per-step 2x2 composite frames.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    metadata_path = build_video_dataset(
        session_dir=args.session_dir,
        sim_step_sec=args.sim_step_sec,
        fps=args.fps,
        export_direction_videos=not args.no_direction_videos,
        export_composite_frames=not args.no_composite_frames,
        intersections=args.intersections,
    )
    print(f"Video dataset exported. Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
