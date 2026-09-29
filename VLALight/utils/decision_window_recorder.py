import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .async_video_writer import AsyncVideoWriter, AsyncVideoWritePool
from .video_recorder import add_direction_label, convert_rgb_to_bgr


DIRECTIONS: Tuple[str, ...] = ("N", "E", "W", "S")


@dataclass
class IntervalContext:
    decision_step: int
    sim_start_sec: float


class DecisionWindowRecorder:
    """
    Records one decision-window clip per intersection.

    Each clip corresponds to the simulation interval between two VLM decisions.
    The clip can later be fed to a video-capable VLM, while the current-step
    still images continue to support existing image tools.
    """

    def __init__(
        self,
        session_dir: str,
        direction_mapping: Dict[str, Dict[str, str]],
        fps: int = 5,
        add_labels: bool = True,
        export_direction_videos: bool = False,
        export_composite_video: bool = True,
        export_direction_sequence: bool = False,
        enable_preprocess: bool = True,
        left_crop: float = 0.40,
        right_crop: float = 0.30,
        scale_mode: str = "fit_width",
        frame_view: str = "legacy_crop",
        tile_width: int = 768,
        sensor_type: str = "junction_front_all",
        verbose: bool = False,
        record_mode: str = "sampled",
        sim_interval: float = 1.0,
        sample_interval: float = 1.0,
        preprocess_interpolation: str = "linear",
        video_codec: str = "mp4v",
        video_extension: str = "mp4",
        async_video_write: bool = True,
        async_video_workers: int = 4,
        async_video_max_pending_writes: Optional[int] = None,
    ) -> None:
        self.session_dir = Path(session_dir)
        self.direction_mapping = direction_mapping
        self.fps = fps
        self.add_labels = add_labels
        self.export_composite_video = export_composite_video
        self.export_direction_videos = export_direction_videos
        self.export_direction_sequence = export_direction_sequence
        self.enable_preprocess = enable_preprocess
        self.left_crop = left_crop
        self.right_crop = right_crop
        self.scale_mode = scale_mode
        self.frame_view = frame_view
        self.tile_width = int(tile_width)
        self.sensor_type = sensor_type
        self.verbose = verbose
        self.record_mode = str(record_mode or "sampled").lower()
        self.sim_interval = float(sim_interval or 1.0)
        self.sample_interval = float(sample_interval or 1.0)
        self.video_codec = str(video_codec or "mp4v")
        if len(self.video_codec) != 4:
            raise ValueError(f"video_codec must contain exactly 4 characters: {self.video_codec!r}")
        self.video_extension = str(video_extension or "mp4").strip().lstrip(".") or "mp4"
        interpolation_map = {
            "nearest": cv2.INTER_NEAREST,
            "linear": cv2.INTER_LINEAR,
            "area": cv2.INTER_AREA,
            "cubic": cv2.INTER_CUBIC,
            "lanczos4": cv2.INTER_LANCZOS4,
        }
        self._resize_interpolation = interpolation_map.get(
            str(preprocess_interpolation or "linear").lower(),
            cv2.INTER_LINEAR,
        )
        self.async_video_write = bool(async_video_write)
        self.async_video_workers = max(1, int(async_video_workers or 1))
        pending_limit = int(async_video_max_pending_writes or 0)
        self.async_video_max_pending_writes = pending_limit if pending_limit > 0 else None

        self.output_root = self.session_dir / "images"
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.output_root / "decision_video_metadata.jsonl"

        self._current_ctx: Optional[IntervalContext] = None
        self._composite_writers: Dict[str, cv2.VideoWriter] = {}
        self._direction_writers: Dict[Tuple[str, str], cv2.VideoWriter] = {}
        self._sequence_writers: Dict[str, cv2.VideoWriter] = {}
        self._sequence_shapes: Dict[str, Tuple[int, int]] = {}
        self._frame_counts: Dict[str, int] = {}
        self._frame_times: Dict[str, List[float]] = {}
        self._current_paths: Dict[str, Dict[str, str]] = {}
        self._writer_pool: Optional[AsyncVideoWritePool] = None

    def _get_camera_plan(self, tls_id: str, num_cameras: int = 4):
        return [
            (f"{tls_id}_{cam_idx}", self._get_direction(tls_id, cam_idx))
            for cam_idx in range(num_cameras)
        ]

    def _get_direction(self, tls_id: str, cam_idx: int) -> str:
        if tls_id in self.direction_mapping:
            return self.direction_mapping[tls_id].get(str(cam_idx), str(cam_idx))
        return str(cam_idx)

    def begin_interval(self, decision_step: int, sim_start_sec: float) -> None:
        self.finalize_interval(sim_end_sec=sim_start_sec, force_discard=True)
        self._current_ctx = IntervalContext(decision_step=decision_step, sim_start_sec=sim_start_sec)
        self._composite_writers = {}
        self._direction_writers = {}
        self._sequence_writers = {}
        self._sequence_shapes = {}
        self._frame_counts = {}
        self._frame_times = {}
        self._current_paths = {}
        self._writer_pool = (
            AsyncVideoWritePool(
                num_workers=self.async_video_workers,
                max_queue_size=self.async_video_max_pending_writes,
            )
            if self.async_video_write else None
        )

    def _compose_2x2(self, images: Dict[str, np.ndarray], pad: int = 8) -> np.ndarray:
        n = images["N"]
        e = images["E"]
        w = images["W"]
        s = images["S"]
        h = max(frame.shape[0] for frame in (n, e, w, s))
        wid = max(frame.shape[1] for frame in (n, e, w, s))
        n, e, w, s = (self._pad_to(frame, h, wid) for frame in (n, e, w, s))
        canvas = np.full((h * 2 + pad * 3, wid * 2 + pad * 3, 3), 32, dtype=np.uint8)
        canvas[pad:pad + h, pad:pad + wid] = n
        canvas[pad:pad + h, wid + pad * 2:wid * 2 + pad * 2] = e
        canvas[h + pad * 2:h * 2 + pad * 2, pad:pad + wid] = w
        canvas[h + pad * 2:h * 2 + pad * 2, wid + pad * 2:wid * 2 + pad * 2] = s
        return canvas

    def _direction_sequence_frames(self, images: Dict[str, np.ndarray]) -> List[np.ndarray]:
        target_h = max(images[direction].shape[0] for direction in DIRECTIONS)
        target_w = max(images[direction].shape[1] for direction in DIRECTIONS)
        return [self._pad_to(images[direction], target_h, target_w) for direction in DIRECTIONS]

    def _open_writer(self, path: Path, shape: Tuple[int, int]) -> cv2.VideoWriter:
        path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*self.video_codec)
        writer = cv2.VideoWriter(str(path), fourcc, self.fps, shape)
        if not writer.isOpened():
            raise RuntimeError(f"Failed to create video writer: {path}")
        if self._writer_pool is not None:
            return AsyncVideoWriter(writer, self._writer_pool, copy_frame=False)
        return writer

    def _get_sensor_image(self, image_data):
        if not isinstance(image_data, dict):
            return image_data
        if self.sensor_type in image_data:
            return image_data.get(self.sensor_type)
        for fallback_key in ("junction_front_all", "junction_back_all"):
            if fallback_key in image_data:
                return image_data.get(fallback_key)
        return None

    def _preprocess_frame(self, bgr_image: np.ndarray) -> np.ndarray:
        if not self.enable_preprocess:
            return bgr_image
        if self.frame_view == "legacy_crop":
            bgr_image = self._crop_and_resize(bgr_image)
        return self._resize_to_width(bgr_image, self.tile_width)

    def _preprocess_rgb_frame(self, rgb_image: np.ndarray) -> np.ndarray:
        if not self.enable_preprocess:
            return convert_rgb_to_bgr(rgb_image)
        if self.frame_view == "legacy_crop":
            rgb_image = self._crop_and_resize(rgb_image)
        bgr = convert_rgb_to_bgr(rgb_image)
        return self._resize_to_width(bgr, self.tile_width)

    def _add_direction_label_fast(self, bgr_image: np.ndarray, direction: str) -> np.ndarray:
        return add_direction_label(bgr_image, direction)

    def _prepare_frame(self, image: np.ndarray, direction: str) -> np.ndarray:
        if image.dtype != np.uint8:
            if image.dtype in (np.float32, np.float64) and image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)
        if image.shape[-1] == 4:
            image = image[..., :3]

        rgb = image
        if self.enable_preprocess:
            if self.frame_view == "approach":
                rgb = self._to_approach_view(rgb, direction)
                rgb = self._resize_to_square(rgb, self.tile_width)
            elif self.frame_view == "legacy_crop":
                rgb = self._crop_and_resize(rgb)
        bgr = convert_rgb_to_bgr(rgb)
        if self.frame_view != "approach":
            bgr = self._resize_to_width(bgr, self.tile_width)
        if self.add_labels:
            bgr = add_direction_label(bgr, direction)
        return bgr

    def _crop_and_resize(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        left_crop = max(0.0, min(float(self.left_crop), 0.95))
        right_crop = max(0.0, min(float(self.right_crop), 0.95))
        left_px = int(w * left_crop)
        right_px = int(w * (1.0 - right_crop))
        if right_px <= left_px:
            return image

        cropped = image[:, left_px:right_px]
        crop_h, crop_w = cropped.shape[:2]
        mode = str(self.scale_mode or "fit_width").lower()
        if mode == "fit_width" and crop_w > 0:
            new_h = max(1, int(round(crop_h * (w / float(crop_w)))))
            return cv2.resize(cropped, (w, new_h), interpolation=self._resize_interpolation)
        if mode == "fit_height" and crop_h > 0:
            new_w = max(1, int(round(crop_w * (h / float(crop_h)))))
            return cv2.resize(cropped, (new_w, h), interpolation=self._resize_interpolation)
        return cropped

    def _to_approach_view(self, image: np.ndarray, direction: str) -> np.ndarray:
        h, w = image.shape[:2]
        default = (0.25, 0.72)
        x_windows = {
            "N": (0.18, 0.555),
            "E": (0.50, 0.875),
            "W": (0.30, 0.675),
            "S": (0.27, 0.645),
        }
        x1_ratio, x2_ratio = x_windows.get(direction, default)
        x1 = max(0, min(w - 1, int(round(w * x1_ratio))))
        x2 = max(x1 + 1, min(w, int(round(w * x2_ratio))))
        cropped = image[:h, x1:x2]
        crop_h, crop_w = cropped.shape[:2]
        if crop_w <= 0 or crop_h <= 0:
            return image
        target_w = max(1, int(self.tile_width))
        target_h = max(1, int(round(crop_h * (target_w / float(crop_w)))))
        return cv2.resize(cropped, (target_w, target_h), interpolation=self._resize_interpolation)

    def _resize_to_square(self, image: np.ndarray, target_size: int) -> np.ndarray:
        target = max(1, int(target_size))
        return cv2.resize(image, (target, target), interpolation=self._resize_interpolation)

    def _resize_to_width(self, image: np.ndarray, target_width: int) -> np.ndarray:
        if target_width <= 0:
            return image
        h, w = image.shape[:2]
        if w == target_width:
            return image
        scale = target_width / float(w)
        target_height = max(1, int(round(h * scale)))
        return cv2.resize(image, (target_width, target_height), interpolation=self._resize_interpolation)

    @staticmethod
    def _pad_to(image: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        h, w = image.shape[:2]
        if h == target_h and w == target_w:
            return image
        canvas = np.full((target_h, target_w, 3), 32, dtype=np.uint8)
        y = max(0, (target_h - h) // 2)
        x = max(0, (target_w - w) // 2)
        canvas[y:y + h, x:x + w] = image
        return canvas

    @staticmethod
    def _resize_to_shape(image: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        target_h, target_w = target_shape
        if image.shape[0] == target_h and image.shape[1] == target_w:
            return image
        return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)

    def add_sensor_data(
        self,
        sensor_data: Dict,
        tls_ids: List[str],
        num_cameras: int = 4,
        sim_time: Optional[float] = None,
    ) -> int:
        if self._current_ctx is None:
            raise RuntimeError("DecisionWindowRecorder.begin_interval() must be called before add_sensor_data().")

        written = 0
        for tls_id in tls_ids:
            frames: Dict[str, np.ndarray] = {}
            for cam_idx in range(num_cameras):
                sensor_key = f"{tls_id}_{cam_idx}"
                if sensor_key not in sensor_data:
                    continue
                image_data = sensor_data[sensor_key]
                image = self._get_sensor_image(image_data)
                if image is None:
                    continue
                direction = self._get_direction(tls_id, cam_idx)
                if direction not in DIRECTIONS:
                    continue
                frames[direction] = self._prepare_frame(image, direction)

            if not all(direction in frames for direction in DIRECTIONS):
                continue

            has_writers = (
                (self.export_composite_video and tls_id in self._composite_writers)
                or any((tls_id, direction) in self._direction_writers for direction in DIRECTIONS)
                or (self.export_direction_sequence and tls_id in self._sequence_writers)
            )
            if not has_writers:
                step_dir = self.output_root / f"step_{self._current_ctx.decision_step:04d}" / tls_id
                self._current_paths[tls_id] = {}
                if self.export_composite_video:
                    clip_path = step_dir / f"decision_window_2x2.{self.video_extension}"
                    comp = self._compose_2x2(frames)
                    self._composite_writers[tls_id] = self._open_writer(clip_path, (comp.shape[1], comp.shape[0]))
                    self._current_paths[tls_id]["composite"] = str(clip_path)
                if self.export_direction_sequence:
                    seq_path = step_dir / f"decision_window_direction_sequence.{self.video_extension}"
                    seq_frames = self._direction_sequence_frames(frames)
                    seq_h, seq_w = seq_frames[0].shape[:2]
                    self._sequence_shapes[tls_id] = (seq_h, seq_w)
                    self._sequence_writers[tls_id] = self._open_writer(
                        seq_path,
                        (seq_w, seq_h),
                    )
                    self._current_paths[tls_id]["direction_sequence"] = str(seq_path)
                if self.export_direction_videos:
                    for direction in DIRECTIONS:
                        dir_path = step_dir / f"decision_window_{direction}.{self.video_extension}"
                        self._direction_writers[(tls_id, direction)] = self._open_writer(
                            dir_path, (frames[direction].shape[1], frames[direction].shape[0])
                        )
                        self._current_paths[tls_id][direction] = str(dir_path)

            if self.export_composite_video:
                composite = self._compose_2x2(frames)
                self._composite_writers[tls_id].write(composite)
            if self.export_direction_sequence and tls_id in self._sequence_writers:
                target_shape = self._sequence_shapes[tls_id]
                for seq_frame in self._direction_sequence_frames(frames):
                    self._sequence_writers[tls_id].write(self._resize_to_shape(seq_frame, target_shape))
            if self.export_direction_videos:
                for direction in DIRECTIONS:
                    self._direction_writers[(tls_id, direction)].write(frames[direction])
            self._frame_counts[tls_id] = self._frame_counts.get(tls_id, 0) + 1
            if sim_time is not None:
                self._frame_times.setdefault(tls_id, []).append(float(sim_time))
            written += 1

        return written

    def finalize_interval(self, sim_end_sec: float, force_discard: bool = False,
                          return_details: bool = False):
        if self._current_ctx is None:
            return {}

        clip_paths = self._current_paths if not force_discard else {}
        details = {
            "paths": dict(clip_paths),
            "frame_counts": dict(self._frame_counts),
            "frame_times": {
                tls_id: list(times) for tls_id, times in self._frame_times.items()
            },
            "discarded": bool(force_discard),
        }

        for writer in self._composite_writers.values():
            writer.release()
        for writer in self._sequence_writers.values():
            writer.release()
        for writer in self._direction_writers.values():
            writer.release()

        if not force_discard and clip_paths:
            record = {
                "decision_step": self._current_ctx.decision_step,
                "sim_start_sec": self._current_ctx.sim_start_sec,
                "sim_end_sec": sim_end_sec,
                "fps": self.fps,
                "video_codec": self.video_codec,
                "video_extension": self.video_extension,
                "frame_view": self.frame_view,
                "tile_width": self.tile_width,
                "enable_preprocess": self.enable_preprocess,
                "record_mode": self.record_mode,
                "sim_interval": self.sim_interval,
                "sample_interval": self.sample_interval,
                "export_direction_sequence": self.export_direction_sequence,
                "export_direction_videos": self.export_direction_videos,
                "clips": [],
            }
            for tls_id, paths in clip_paths.items():
                frame_times = self._frame_times.get(tls_id, [])
                record["clips"].append(
                    {
                        "intersection_id": tls_id,
                        "frame_count": self._frame_counts.get(tls_id, 0),
                        "first_frame_sim_time": frame_times[0] if frame_times else None,
                        "last_frame_sim_time": frame_times[-1] if frame_times else None,
                        "sequence_frame_count": (
                            self._frame_counts.get(tls_id, 0) * len(DIRECTIONS)
                            if paths.get("direction_sequence") else 0
                        ),
                        "paths": paths,
                    }
                )
            with open(self.metadata_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        self._current_ctx = None
        self._composite_writers = {}
        self._direction_writers = {}
        self._sequence_writers = {}
        self._sequence_shapes = {}
        self._frame_counts = {}
        self._frame_times = {}
        self._current_paths = {}
        if self._writer_pool is not None:
            self._writer_pool.shutdown()
            self._writer_pool = None
        return details if return_details else clip_paths
