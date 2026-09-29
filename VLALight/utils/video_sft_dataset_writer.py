import json
import hashlib
import os
import re
from pathlib import Path
from typing import Any, Dict, List

import cv2

from .vehicle_position_snapshot import VehiclePositionSnapshotWriter
from .coordination_frame_selector import (
    OPPOSITE_DIRECTION,
    build_topology_coordination_plan,
    controlled_movements_for_entry,
)


def validate_video_sft_output_dir(output_dir: str) -> None:
    """Reject an output directory that already contains SFT records."""
    manifest_path = os.path.join(output_dir, "video_sft_raw.jsonl")
    complete_manifest_path = os.path.join(output_dir, "video_sft_complete.jsonl")
    audit_summary_path = os.path.join(
        output_dir, "video_sft_alignment_compact_summary_pretty.json"
    )
    records_path = os.path.join(output_dir, "json")
    if (os.path.isfile(manifest_path) or os.path.isfile(complete_manifest_path)
            or os.path.isfile(audit_summary_path)) or (
        os.path.isdir(records_path) and bool(os.listdir(records_path))
    ):
        raise FileExistsError(
            f"video SFT output already contains records: {output_dir}. "
            "Use a new --work-dir to avoid overwriting or mixing datasets."
        )


class VideoSFTDatasetWriter:
    """Stream synchronized SUMO histories and decision-window video paths."""

    def __init__(self, output_dir: str, view_distance_m: float,
                 video_sample_interval_s: float, sim_interval_s: float,
                 local_render_radius_m: float = None,
                 network_topology: Dict[str, Any] = None,
                 coordination_speed_mps: float = 11.0,
                 save_coordination_frames: bool = False,
                 resume_existing: bool = False) -> None:
        self.output_dir = output_dir
        self.video_sample_interval_s = float(video_sample_interval_s)
        self.sim_interval_s = float(sim_interval_s)
        self.view_distance_m = float(view_distance_m)
        self.local_render_radius_m = (
            float(local_render_radius_m) if local_render_radius_m is not None else None
        )
        self.network_topology = network_topology or {}
        self.coordination_speed_mps = float(coordination_speed_mps)
        self.save_coordination_frames = bool(save_coordination_frames)
        self._pending_coordination: Dict[int, Dict[str, List[Dict[str, Any]]]] = {}
        if not resume_existing:
            validate_video_sft_output_dir(output_dir)
        manifest_path = os.path.join(output_dir, "video_sft_raw.jsonl")
        records_path = os.path.join(output_dir, "json")
        complete_manifest_path = os.path.join(output_dir, "video_sft_complete.jsonl")
        os.makedirs(output_dir, exist_ok=True)
        self.records_dir = os.path.join(output_dir, "json")
        os.makedirs(self.records_dir, exist_ok=True)
        self.jsonl_path = manifest_path
        self.complete_jsonl_path = complete_manifest_path
        self.index_path = os.path.join(output_dir, "video_sft_raw_index.json")
        self.audit_summary_path = os.path.join(
            output_dir, "video_sft_alignment_compact_summary_pretty.json"
        )
        self._snapshot_builder = VehiclePositionSnapshotWriter(
            output_dir=os.path.join(output_dir, "_snapshot_schema"),
            view_distance_m=view_distance_m,
        )
        self._decision_step = None
        self._sim_start_s = None
        self._expected_tls_ids: List[str] = []
        self._collection_start_s = None
        self._frames: List[Dict[str, Any]] = []
        self._lane_groups_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._sample_count = 0
        self._valid_sample_count = 0
        self._audit_records: List[Dict[str, Any]] = []

    def begin_interval(self, decision_step: int, sim_start_s: float,
                       tls_ids: List[str] = None) -> None:
        if self._decision_step is not None:
            raise RuntimeError("previous video SFT interval was not finalized")
        self._decision_step = int(decision_step)
        self._sim_start_s = float(sim_start_s)
        if self._collection_start_s is None:
            self._collection_start_s = float(sim_start_s)
        self._expected_tls_ids = list(tls_ids or [])
        self._frames = []

    def capture_tick(self, inner_i: int, env: Any,
                     sim_time_s: float = None) -> None:
        if self._decision_step is None:
            return
        tick_number = int(inner_i) + 1
        stride = max(1, int(round(self.video_sample_interval_s / self.sim_interval_s)))
        env_sim_time_s = float(env.get_current_time())
        actual_sim_time_s = env_sim_time_s if sim_time_s is None else float(sim_time_s)
        expected_sim_time_s = self._sim_start_s + tick_number * self.sim_interval_s
        if abs(env_sim_time_s - actual_sim_time_s) > 1e-6:
            raise RuntimeError(
                "SUMO time changed between video capture and SFT snapshot: "
                f"video_time={actual_sim_time_s} snapshot_time={env_sim_time_s}"
            )
        if abs(actual_sim_time_s - expected_sim_time_s) > 1e-6:
            raise RuntimeError(
                "SUMO tick time is not aligned with the decision interval: "
                f"decision_step={self._decision_step} inner_i={inner_i} "
                f"expected={expected_sim_time_s} actual={actual_sim_time_s}"
            )
        intersections = [self._build_compact_intersection(env, inter) for inter in env.list_intersection]
        self._frames.append({
            "frame_index": int(inner_i),
            "sim_time_s": actual_sim_time_s,
            "scheduled_for_video": tick_number % stride == 0,
            "intersections": intersections,
        })

    def _build_compact_intersection(self, env: Any, inter: Any) -> Dict[str, Any]:
        tls_id = inter.inter_id
        lane_groups = self._lane_groups_cache.get(tls_id)
        if lane_groups is None:
            lane_groups = self._snapshot_builder._lane_groups_for_intersection(env, inter)
            self._lane_groups_cache[tls_id] = lane_groups
        try:
            lane_to_movements = env._v36_outbound_lane_movement_map(inter)
        except (AttributeError, TypeError, ValueError):
            lane_to_movements = {}
        camera_views = {}
        for orient in ("N", "E", "W", "S"):
            view_lanes = []
            visual_counter = 1
            for lane_group in lane_groups:
                if lane_group["orient"] != orient:
                    continue
                lane_record = {**lane_group, "vehicles": []}
                coordination_movements = sorted(set(
                    lane_to_movements.get(lane_group.get("lane_id"), [])
                    if lane_group.get("lane_role") == "outgoing" else []
                ))
                lane_record["coordination_movements"] = coordination_movements
                for vehicle in self._snapshot_builder._vehicles_for_lane(env, lane_group):
                    vehicle = dict(vehicle)
                    vehicle["visual_id"] = f"v{visual_counter}"
                    vehicle["camera_direction"] = orient
                    vehicle["coordination_movements"] = coordination_movements
                    visual_counter += 1
                    lane_record["vehicles"].append(vehicle)
                view_lanes.append(lane_record)
            camera_views[orient] = {
                "tls_id": tls_id,
                "intersection_name": getattr(inter, "inter_name", tls_id),
                "camera_direction": orient,
                "lanes": view_lanes,
            }
        return {
            "tls_id": tls_id,
            "intersection_name": getattr(inter, "inter_name", tls_id),
            "camera_views": camera_views,
        }

    def _save_coordination_frame(
            self, video_path: str, video_frame_index: int,
            frame_index: int, source_video_time_s: float,
            source_exit_direction: str = "") -> Dict[str, Any]:
        """Extract one already-written source-video frame for offline review."""
        result = {
            "coordination_frame_available": False,
            "coordination_frame_path": None,
            "coordination_frame_relative_path": None,
            "coordination_frame_error": None,
        }
        if not self.save_coordination_frames:
            return result
        if not video_path:
            result["coordination_frame_error"] = "source video path is unavailable"
            return result
        if int(video_frame_index) < 0:
            result["coordination_frame_error"] = (
                f"invalid source video frame index: {video_frame_index}"
            )
            return result

        capture = cv2.VideoCapture(os.fspath(video_path))
        try:
            if not capture.isOpened():
                result["coordination_frame_error"] = (
                    f"failed to open source video: {video_path}"
                )
                return result

            # Read from the beginning because random seeking is codec-dependent.
            frame = None
            for _ in range(int(video_frame_index) + 1):
                ok, frame = capture.read()
                if not ok or frame is None:
                    result["coordination_frame_error"] = (
                        "source video ended before requested frame "
                        f"{video_frame_index}"
                    )
                    return result
        finally:
            capture.release()

        output_dir = Path(video_path).parent / "coordination_frames"
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_time = (
            f"{float(source_video_time_s):.1f}"
            .replace("-", "m")
            .replace(".", "p")
        )
        direction = str(source_exit_direction or "").strip().upper()
        if direction not in {"N", "E", "W", "S"}:
            direction = "UNKNOWN"
        output_path = output_dir / (
            f"coordination_frame_{direction}_{int(frame_index):02d}_t{safe_time}s.jpg"
        )
        try:
            encoded_ok, encoded = cv2.imencode(".jpg", frame)
            if not encoded_ok:
                raise RuntimeError("cv2.imencode returned false")
            with open(output_path, "wb") as handle:
                handle.write(encoded.tobytes())
        except Exception as exc:
            result["coordination_frame_error"] = (
                f"failed to save coordination frame: {type(exc).__name__}: {exc}"
            )
            return result

        result.update({
            "coordination_frame_available": True,
            "coordination_frame_path": str(output_path),
            "coordination_frame_relative_path": os.path.relpath(
                output_path, self.output_dir
            ),
        })
        return result

    def finalize_interval(self, sim_end_s: float,
                          video_details: Dict[str, Any], status: str = "complete",
                          error: str = None) -> None:
        if self._decision_step is None:
            return
        scheduled_times = [
            frame["sim_time_s"] for frame in self._frames
            if frame["scheduled_for_video"]
        ]
        paths = (video_details or {}).get("paths", {})
        actual_times_by_tls = (video_details or {}).get("frame_times", {})
        intersections_by_frame = [
            {item["tls_id"]: item for item in frame["intersections"]}
            for frame in self._frames
        ]
        tls_ids = sorted(set(self._expected_tls_ids) | {
            intersection["tls_id"]
            for frame in self._frames
            for intersection in frame["intersections"]
        })
        coordination_by_target = self._pending_coordination.pop(
            self._decision_step, {})
        for target_tls in tls_ids:
            for selection in build_topology_coordination_plan(
                    self.network_topology, target_tls,
                    speed_mps=self.coordination_speed_mps):
                source_tls = selection["source_tls"]
                source_times = [float(value) for value in
                                actual_times_by_tls.get(source_tls, [])]
                video_index = int(selection["frame_index"]) - 1
                source_time = (
                    source_times[video_index]
                    if 0 <= video_index < len(source_times) else None
                )
                source_vehicles = []
                outbound_by_movement = {}
                movement_mapping_available = False
                if source_time is not None:
                    source_frame = next(
                        (frame_intersections.get(source_tls)
                         for frame, frame_intersections in zip(
                             self._frames, intersections_by_frame)
                         if abs(float(frame["sim_time_s"]) - source_time) <= 1e-6),
                        None,
                    )
                    source_view = (
                        (source_frame or {}).get("camera_views", {}).get(
                            selection["source_exit_direction"], {})
                    )
                    target_entry = str(
                        selection.get("target_entry_direction") or "").upper()
                    for lane in source_view.get("lanes", []):
                        if lane.get("lane_role") != "outgoing":
                            continue
                        lane_movements = sorted({
                            str(movement).upper()
                            for movement in lane.get(
                                "coordination_movements", [])
                            if str(movement).upper().startswith(target_entry)
                        })
                        movement_mapping_available = bool(
                            movement_mapping_available or lane_movements)
                        for vehicle in lane.get("vehicles", []):
                            distance = vehicle.get(
                                "signed_distance_from_observed_stopline_m")
                            if (vehicle.get("sumo_vehicle_id") is None
                                    or distance is None
                                    or not 0.0 <= float(distance) <= self.view_distance_m):
                                continue
                            vehicle_record = {
                                "sumo_vehicle_id": vehicle.get("sumo_vehicle_id"),
                                "lane_name": lane.get("lane_name"),
                                "current_lane_id": vehicle.get("current_lane_id"),
                                "distance_from_source_stopline_m": float(distance),
                                "speed_mps": vehicle.get("speed_mps"),
                                "coordination_movements": lane_movements,
                            }
                            source_vehicles.append(vehicle_record)
                            for movement in lane_movements:
                                outbound_by_movement.setdefault(
                                    movement, []).append(
                                        vehicle_record["sumo_vehicle_id"])
                    source_vehicles.sort(key=lambda item: item["sumo_vehicle_id"])
                outbound_by_movement = {
                    movement: sorted(set(vehicle_ids))
                    for movement, vehicle_ids in outbound_by_movement.items()
                }
                controlled_outbound_by_movement = {
                    movement: vehicle_ids
                    for movement, vehicle_ids in outbound_by_movement.items()
                    if movement[1:] in ("L", "T")
                }
                outbound_count_by_movement = {
                    movement: len(vehicle_ids)
                    for movement, vehicle_ids in outbound_by_movement.items()
                }
                controlled_outbound_count_by_movement = {
                    movement: len(vehicle_ids)
                    for movement, vehicle_ids in
                    controlled_outbound_by_movement.items()
                }
                source_vehicle_ids = [
                    item["sumo_vehicle_id"] for item in source_vehicles
                ]
                reference = {
                    **selection,
                    "source_decision_step": self._decision_step,
                    "source_observation_step": self._decision_step,
                    "target_observation_step": self._decision_step
                        + int(selection["injection_delay_cycles"]),
                    "injected_decision_step": self._decision_step
                        + int(selection["injection_delay_cycles"]) + 1,
                    "source_video_frame_index": video_index,
                    "source_video_frame_sim_time_s": source_time,
                    "source_video_path": (paths.get(source_tls, {}) or {}).get(
                        selection["source_exit_direction"]),
                    "outbound_vehicle_ids": source_vehicle_ids,
                    "outbound_vehicle_count": len(source_vehicle_ids),
                    "outbound_vehicles": source_vehicles,
                    "outbound_vehicle_ids_by_movement": outbound_by_movement,
                    "outbound_vehicle_count_by_movement": (
                        outbound_count_by_movement),
                    "controlled_outbound_vehicle_ids_by_movement": (
                        controlled_outbound_by_movement),
                    "controlled_outbound_vehicle_count_by_movement": (
                        controlled_outbound_count_by_movement),
                    "controlled_outbound_vehicle_count": len({
                        vehicle_id
                        for vehicle_ids in controlled_outbound_by_movement.values()
                        for vehicle_id in vehicle_ids
                    }),
                    "movement_mapping_available": bool(
                        movement_mapping_available),
                }
                reference["reference_valid"] = bool(
                    reference["source_video_path"]
                    and reference["source_video_frame_sim_time_s"] is not None
                )
                if reference["source_video_frame_sim_time_s"] is None:
                    reference.update({
                        "coordination_frame_available": False,
                        "coordination_frame_path": None,
                        "coordination_frame_relative_path": None,
                        "coordination_frame_error": (
                            "source video frame time is unavailable"
                        ),
                    })
                else:
                    reference.update(self._save_coordination_frame(
                        video_path=reference["source_video_path"],
                        video_frame_index=video_index,
                        frame_index=int(selection["frame_index"]),
                        source_video_time_s=float(
                            reference["source_video_frame_sim_time_s"]
                        ),
                        source_exit_direction=reference[
                            "source_exit_direction"
                        ],
                    ))
                if self.save_coordination_frames:
                    reference["reference_valid"] = bool(
                        reference["reference_valid"]
                        and reference["coordination_frame_available"]
                    )
                target_observation_step = reference["target_observation_step"]
                if target_observation_step == self._decision_step:
                    coordination_by_target.setdefault(target_tls, []).append(reference)
                else:
                    self._pending_coordination.setdefault(
                        target_observation_step, {}).setdefault(
                            target_tls, []).append(reference)
        for tls_id in tls_ids:
            actual_times = [float(value) for value in actual_times_by_tls.get(tls_id, [])]
            tls_history = []
            for frame, frame_intersections in zip(self._frames, intersections_by_frame):
                intersection = frame_intersections.get(tls_id)
                if intersection is None:
                    continue
                sim_time = float(frame["sim_time_s"])
                written_index = next(
                    (idx for idx, value in enumerate(actual_times)
                     if abs(value - sim_time) <= 1e-6),
                    None,
                )
                tls_history.append({
                    "frame_index": frame["frame_index"],
                    "sim_time_s": sim_time,
                    "scheduled_for_video": frame["scheduled_for_video"],
                    "written_to_video": written_index is not None,
                    "video_frame_index": written_index,
                    "intersection": intersection,
                })

            expected_final = scheduled_times[-1] if scheduled_times else None
            complete_sequence = (
                len(actual_times) == len(scheduled_times)
                and all(
                    abs(actual - scheduled) <= 1e-6
                    for actual, scheduled in zip(actual_times, scheduled_times)
                )
            )
            missing_times = [
                scheduled for scheduled in scheduled_times
                if not any(abs(actual - scheduled) <= 1e-6 for actual in actual_times)
            ]
            extra_times = [
                actual for actual in actual_times
                if not any(abs(actual - scheduled) <= 1e-6 for scheduled in scheduled_times)
            ]
            sample_status = status
            invalid_reason = error
            if sample_status == "complete" and not complete_sequence:
                sample_status = "invalid"
                invalid_reason = "actual video frame sequence does not match the scheduled sequence"
            coordination_sources = coordination_by_target.get(tls_id, [])
            if sample_status == "complete" and any(
                    not item.get("reference_valid", False)
                    for item in coordination_sources):
                sample_status = "invalid"
                invalid_reason = (
                    "selected upstream coordination video frame is unavailable"
                )
            record = {
                "decision_step": self._decision_step,
                "tls_id": tls_id,
                "status": sample_status,
                "invalid_reason": invalid_reason,
                "sim_start_s": self._sim_start_s,
                "sim_end_s": float(sim_end_s),
                "decision_interval_s": float(sim_end_s) - float(self._sim_start_s),
                "video_sample_interval_s": self.video_sample_interval_s,
                "scheduled_video_frame_sim_times": scheduled_times,
                "video_frame_sim_times": actual_times,
                "missing_video_frame_sim_times": missing_times,
                "extra_video_frame_sim_times": extra_times,
                "target_sim_time_s": actual_times[-1] if actual_times else None,
                "expected_target_sim_time_s": expected_final,
                "target_semantics": "vehicles and lane state at this TLS video's final written frame",
                "video_paths": paths.get(tls_id, {}),
                "sumo_tick_history": tls_history,
                "coordination_sources": coordination_sources,
            }
            self._append_audit_records(
                decision_step=self._decision_step,
                tls_id=tls_id,
                sample_status=sample_status,
                video_paths=paths.get(tls_id, {}),
                actual_times=actual_times,
                tls_history=tls_history,
                coordination_sources=coordination_sources,
            )
            safe_tls_id = self._safe_tls_filename(tls_id)
            record_name = f"decision_{self._decision_step:06d}_{safe_tls_id}.json"
            record_path = os.path.join(self.records_dir, record_name)
            with open(record_path, "w", encoding="utf-8") as handle:
                json.dump(record, handle, ensure_ascii=False, indent=2)
            manifest_item = {
                "decision_step": self._decision_step,
                "tls_id": tls_id,
                "status": sample_status,
                "target_sim_time_s": record["target_sim_time_s"],
                "record_path": os.path.relpath(record_path, self.output_dir),
            }
            with open(self.jsonl_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(manifest_item, ensure_ascii=False) + "\n")
            if sample_status == "complete":
                with open(self.complete_jsonl_path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(manifest_item, ensure_ascii=False) + "\n")
            self._sample_count += 1
            if sample_status == "complete":
                self._valid_sample_count += 1
        self._write_index()
        self._write_audit_summary()
        self._decision_step = None
        self._sim_start_s = None
        self._expected_tls_ids = []
        self._frames = []

    @staticmethod
    def _safe_tls_filename(tls_id: str) -> str:
        raw = str(tls_id)
        prefix = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip(" ._") or "tls"
        digest = hashlib.blake2b(raw.encode("utf-8"), digest_size=6).hexdigest()
        return f"{prefix[:80]}_{digest}"

    def _append_audit_records(self, decision_step: int, tls_id: str,
                              sample_status: str, video_paths: Dict[str, str],
                              actual_times: List[float],
                              tls_history: List[Dict[str, Any]],
                              coordination_sources: List[Dict[str, Any]] = None) -> None:
        history_by_time = {
            float(frame["sim_time_s"]): frame for frame in tls_history
        }
        for direction in ("N", "E", "W", "S"):
            video_path = video_paths.get(direction)
            frames = []
            for video_frame_index, sim_time in enumerate(actual_times):
                history_frame = next(
                    (frame for time_value, frame in history_by_time.items()
                     if abs(time_value - sim_time) <= 1e-6),
                    None,
                )
                if history_frame is None:
                    continue
                camera_view = (
                    history_frame.get("intersection", {})
                    .get("camera_views", {})
                    .get(direction, {})
                )
                sumo_lanes = {}
                vehicle_count = 0
                for lane in camera_view.get("lanes", []):
                    vehicles = []
                    for vehicle in lane.get("vehicles", []):
                        vehicles.append({
                            "sumo_vehicle_id": vehicle.get("sumo_vehicle_id"),
                            "visual_id": vehicle.get("visual_id"),
                            "lane_id": vehicle.get("lane_id"),
                            "current_lane_id": vehicle.get("current_lane_id"),
                            "distance_m": vehicle.get(
                                "signed_distance_from_observed_stopline_m"
                            ),
                            "speed_mps": vehicle.get("speed_mps"),
                            "position_xy": vehicle.get("position_xy"),
                            "coordination_movements": list(
                                vehicle.get("coordination_movements", []) or []),
                        })
                    sumo_lanes[lane.get("lane_name")] = vehicles
                    vehicle_count += len(vehicles)
                frames.append({
                    "video_frame_index": video_frame_index,
                    "sim_time_s": float(sim_time),
                    "collection_time_s": float(sim_time) - float(self._collection_start_s),
                    "relative_time_s": float(sim_time) - float(self._sim_start_s),
                    "sumo_lanes": sumo_lanes,
                    "vehicle_count": vehicle_count,
                })
            self._audit_records.append({
                "sample_id": f"decision_{decision_step:06d}_{tls_id}_{direction}",
                "decision_step": int(decision_step),
                "intersection": tls_id,
                "direction": direction,
                "status": sample_status,
                "sim_start_s": float(self._sim_start_s),
                "collection_start_s": float(self._sim_start_s) - float(self._collection_start_s),
                "video_name": os.path.basename(video_path) if video_path else None,
                "video_path": video_path,
                "coordination_sources": [
                    dict(item)
                    for item in (coordination_sources or [])
                    if item.get("target_entry_direction") == direction
                ],
                "frame_count": len(frames),
                "frames": frames,
            })

    def _write_audit_summary(self) -> None:
        complete_records = sum(
            1 for record in self._audit_records if record["status"] == "complete"
        )
        total_frames = sum(record["frame_count"] for record in self._audit_records)
        total_vehicles = sum(
            frame["vehicle_count"]
            for record in self._audit_records
            for frame in record["frames"]
        )
        coordination_references = [
            item
            for record in self._audit_records
            for item in record.get("coordination_sources", [])
        ]
        summary = {
            "source_dir": self.output_dir,
            "record_count": len(self._audit_records),
            "complete_record_count": complete_records,
            "total_video_frames": total_frames,
            "total_vehicle_observations": total_vehicles,
            "coordination_reference_count": len(coordination_references),
            "valid_coordination_reference_count": sum(
                bool(item.get("reference_valid", False))
                for item in coordination_references
            ),
            "coordination_frame_save_enabled": self.save_coordination_frames,
            "coordination_frame_available_count": sum(
                bool(item.get("coordination_frame_available", False))
                for item in coordination_references
            ),
            "format": (
                "one record per direction video; frames contain actual written "
                "video frame index, SUMO time, lane-grouped vehicle distance and speed"
            ),
            "distance_definition": (
                "signed distance to observed stop line; negative values indicate "
                "the vehicle has crossed the reference line"
            ),
            "camera_view_distance_m": self.view_distance_m,
            "coordination_definition": (
                "source intersection outbound vehicles within 0-150m; source frame "
                "and delayed target decision are selected by the shared topology "
                "coordination planner using ETA window [-5,15]s; V35 observation "
                "step k is consumed by LLMLight decision step k+1"
            ),
            "coordination_frame_definition": (
                "when enabled, the selected frame is decoded from the already-"
                "written upstream direction video using the same zero-based video "
                "frame index and recorded SUMO time; no extra SUMO step or render "
                "is executed"
            ),
            "coordination_outbound_distance_definition": (
                "distance after the source stop line along the selected outgoing "
                "lane; only 0 <= distance <= camera_view_distance_m is counted"
            ),
            "coordination_movement_definition": (
                "each vehicle is assigned from its current source outgoing lane "
                "to the static downstream movement exposed by SUMO lane links; "
                "future route and later lane changes are intentionally ignored; "
                "only through/left movements are injected into controlled phases"
            ),
            "coordination_direction_mapping": {
                source_exit: {
                    "target_entry_direction": target_entry,
                    "target_movements": list(
                        controlled_movements_for_entry(target_entry)),
                }
                for source_exit, target_entry in OPPOSITE_DIRECTION.items()
            },
            "local_render_radius_m": self.local_render_radius_m,
            "time_definition": (
                "sim_time_s uses the original SUMO clock for exact label alignment; "
                "collection_time_s is normalized to collection start; relative_time_s "
                "is within the 30s window"
            ),
            "records": self._audit_records,
        }
        with open(self.audit_summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)

    def _write_index(self) -> None:
        index = {
            "jsonl_path": self.jsonl_path,
            "complete_samples_jsonl_path": self.complete_jsonl_path,
            "audit_summary_json_path": self.audit_summary_path,
            "tls_sample_count": self._sample_count,
            "valid_sample_count": self._valid_sample_count,
            "record_granularity": "one formatted JSON file per TLS per 30-second decision window",
            "records_dir": self.records_dir,
            "sumo_history_interval_s": self.sim_interval_s,
            "video_sample_interval_s": self.video_sample_interval_s,
            "camera_view_distance_m": self.view_distance_m,
            "local_render_radius_m": self.local_render_radius_m,
            "final_frame_rule": "SFT outputs must contain only vehicles visible at target_sim_time_s",
            "vehicle_identity": "sumo_vehicle_id links the same vehicle across ticks",
            "coordination_movement_mapping": (
                "current outgoing lane to static downstream movement; no future route "
                "or later lane-change tracking"
            ),
            "downstream_filter_rule": "training dataset builders must read video_sft_complete.jsonl only",
        }
        with open(self.index_path, "w", encoding="utf-8") as handle:
            json.dump(index, handle, ensure_ascii=False, indent=2)
