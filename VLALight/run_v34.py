"""Collect V34 real-action transition data with V23-aligned video frames.

V34 runs one main SUMO simulation only. At every 30-second decision, it uses
the V2 policy exactly. The normal V23 video recorder writes frames at 5, 10, ..., 30s;
this script records the executed action's controlled-movement remaining-V at
those exact same SUMO ticks.
"""

import argparse
import json
import math
import os
import time
import xml.etree.ElementTree as ElementTree
from pathlib import Path
from typing import Any, Dict, List, Tuple

from utils.vlm_config import get_path_conf, get_traffic_env_conf
from utils.vlm_oneline import VLMOneLine
from utils.video_sft_dataset_writer import validate_video_sft_output_dir


PHASES = ("ETWT", "NTST", "ELWL", "NLSL")
DATASET_IDENTIFIERS = {
    "jinan": ("Jinan-3_4", "anon_3_4_jinan_real"),
    "hangzhou": ("Hangzhou-4_4", "anon_4_4_hangzhou_real"),
    "newyork": ("NewYork-28_7", "anon_28_7_newyork_real_double"),
    "newyork16x3": ("NewYork-16x3", "anon_16_3_newyork_real"),
    "newyork16x3_v1": ("NewYork-16x3-v1", "anon_16_3_newyork_real"),
    "newyork7x7_v1": ("NewYork-7x7-v1", "anon_7_7_newyork_wave_4000"),
}
# Keep scenario-level behavior aligned with the corresponding V28 setup.
# V34 differs only in data collection and action selection, not in the SUMO
# network, four-phase action space, or signal timing contract.
V28_SCENARIO_CONTRACTS = {
    "hangzhou": {
        "ROADNET_FILE": "hangzhou_phase.net.xml",
        "TRAFFIC_FILE": "hangzhou.rou.xml",
        "SUMOCFG_FILE": "hangzhou.sumocfg",
        "NUM_ROW": 4,
        "NUM_COL": 4,
        "NUM_INTERSECTIONS": 16,
        "NUM_AGENTS": 16,
    },
}
MOVEMENT_INDEX = {
    "WL": 0, "WT": 1, "WR": 2,
    "EL": 3, "ET": 4, "ER": 5,
    "NL": 6, "NT": 7, "NR": 8,
    "SL": 9, "ST": 10, "SR": 11,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect V34 V2-policy video transitions.")
    parser.add_argument("--dataset", default="newyork",
                        choices=sorted(DATASET_IDENTIFIERS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-counts", type=int, default=3600)
    parser.add_argument(
        "--traffic-file",
        default=None,
        help=(
            "Traffic input in the selected scenario data directory. A .rou.xml is used "
            "directly; a CityFlow .json is converted once to a same-stem .rou.xml."
        ),
    )
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--video-sample-interval", type=float, default=5.0)
    parser.add_argument("--camera-view-distance", type=float, default=150.0)
    parser.add_argument("--local-render-radius", type=float, default=180.0)
    parser.add_argument("--render-preset", default="1080P",
                        choices=["320P", "480P", "720P", "1080P"])
    parser.add_argument("--rendering-backend", default="p3headlessgl",
                        choices=["pandagl", "p3headlessgl"])
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--async-workers", type=int, default=4)
    parser.add_argument("--keep-batch-sensors", type=int, choices=[0, 1], default=0)
    parser.add_argument("--reuse-batch-sensors", type=int, choices=[0, 1], default=1)
    parser.add_argument("--step-task-manager", type=int, choices=[0, 1], default=0)
    return parser.parse_args()


def phase_movements(phase: str) -> List[str]:
    return [phase[index:index + 2] for index in range(0, len(phase), 2)]


def _convert_cityflow_flow_to_sumo_route(source_path: Path, route_path: Path) -> None:
    """Convert a CityFlow flow JSON into a SUMO route file without changing the source."""
    with source_path.open("r", encoding="utf-8") as handle:
        flows = json.load(handle)
    if not isinstance(flows, list):
        raise ValueError(f"CityFlow traffic JSON must contain a list: {source_path}")

    root = ElementTree.Element("routes")
    vehicle_types: Dict[Tuple[Tuple[str, str], ...], str] = {}
    for vehicle_index, flow in enumerate(flows):
        vehicle = flow.get("vehicle")
        route = flow.get("route")
        if not isinstance(vehicle, dict) or not isinstance(route, list) or not route:
            raise ValueError(
                f"Invalid CityFlow vehicle at index {vehicle_index} in {source_path}"
            )
        type_attrs = {
            "length": str(vehicle.get("length", 5.0)),
            "width": str(vehicle.get("width", 2.0)),
            "minGap": str(vehicle.get("minGap", 2.5)),
            "maxSpeed": str(vehicle.get("maxSpeed", 11.111)),
            "accel": str(vehicle.get("usualPosAcc", vehicle.get("maxPosAcc", 2.0))),
            "decel": str(vehicle.get("usualNegAcc", vehicle.get("maxNegAcc", 4.5))),
        }
        type_key = tuple(sorted(type_attrs.items()))
        type_id = vehicle_types.get(type_key)
        if type_id is None:
            type_id = f"cityflow_type_{len(vehicle_types)}"
            ElementTree.SubElement(root, "vType", id=type_id, **type_attrs)
            vehicle_types[type_key] = type_id
        depart = flow.get("startTime", 0)
        vehicle_element = ElementTree.SubElement(
            root, "vehicle", id=str(vehicle_index), depart=str(depart), type=type_id
        )
        ElementTree.SubElement(vehicle_element, "route", edges=" ".join(map(str, route)))

    ElementTree.indent(root, space="  ")
    ElementTree.ElementTree(root).write(route_path, encoding="utf-8", xml_declaration=True)
    print(f"[V34] converted {len(flows)} CityFlow vehicles: {source_path} -> {route_path}")


def _resolve_traffic_file(args: argparse.Namespace, path_conf: Dict[str, str], conf: Dict[str, Any]) -> str:
    """Set the SUMO route file selected at the command line, if one was requested."""
    if not args.traffic_file:
        return conf["TRAFFIC_FILE"]
    requested = Path(args.traffic_file)
    if requested.is_absolute() or ".." in requested.parts:
        raise ValueError("--traffic-file must be a filename inside the scenario data directory")
    source_path = Path(path_conf["PATH_TO_DATA"]) / requested
    if not source_path.is_file():
        raise FileNotFoundError(f"Traffic file not found: {source_path}")
    if requested.suffix.lower() == ".json":
        route_name = f"{requested.stem}.rou.xml"
        route_path = source_path.with_name(route_name)
        if not route_path.exists() or route_path.stat().st_mtime < source_path.stat().st_mtime:
            _convert_cityflow_flow_to_sumo_route(source_path, route_path)
    elif requested.name.endswith(".rou.xml"):
        route_name = requested.name
    else:
        raise ValueError("--traffic-file must end in .json or .rou.xml")
    conf["TRAFFIC_FILE"] = route_name
    conf["SUMOCFG_FILE"] = f"{Path(route_name).stem}.sumocfg"
    return route_name


class V34Collector(VLMOneLine):
    """V23 video collector with the exact V2 phase-selection policy."""

    def __init__(self, *args: Any, **kwargs: Any):
        self.phase_histories: Dict[str, Dict[str, List[int]]] = {}
        self.action_records: Dict[int, Dict[str, Dict[str, Any]]] = {}
        self.frame_remaining_v: Dict[int, Dict[str, Dict[float, int]]] = {}
        self.frame_remaining_queue: Dict[
            int, Dict[str, Dict[float, Dict[str, Any]]]
        ] = {}
        self._active_decision_step: int = -1
        super().__init__(*args, **kwargs)

    def _create_agents(self) -> None:
        # No VLM request is made while collecting this supervised dataset.
        self.agents = []

    @staticmethod
    def _phase_current_v(intersection: Any, phase: str) -> int:
        movement_sets = intersection.dic_feature.get("traffic_movement_vehicle_ids_150m", [])
        total = 0
        for movement in phase_movements(phase):
            index = MOVEMENT_INDEX.get(movement)
            if index is not None and index < len(movement_sets):
                total += len(movement_sets[index] or [])
        return int(total)

    @staticmethod
    def _controlled_remaining_v(intersection: Any) -> int:
        movement_sets = intersection.dic_feature.get("traffic_movement_vehicle_ids_150m", [])
        vehicle_ids = set()
        for phase in intersection.control_phases:
            for movement in phase_movements(phase):
                index = MOVEMENT_INDEX.get(movement)
                if index is not None and index < len(movement_sets):
                    vehicle_ids.update(movement_sets[index] or [])
        return len(vehicle_ids)

    @staticmethod
    def _controlled_queue_snapshot(intersection: Any) -> Dict[str, Any]:
        """Return the V30-compatible stopped-vehicle queue by movement/phase.

        A queue vehicle is a unique vehicle in a signal-controlled through or
        left-turn movement whose instantaneous SUMO speed is below 0.1 m/s.
        Right turns are deliberately excluded because they are uncontrolled.
        """
        movement_sets = intersection.dic_feature.get(
            "traffic_movement_vehicle_ids_150m", []
        )
        speeds = getattr(intersection, "dic_vehicle_speed_current_step", {})
        lane_queue = {
            movement: 0
            for phase in PHASES
            for movement in phase_movements(phase)
        }
        queued_ids_by_movement: Dict[str, set] = {
            movement: set() for movement in lane_queue
        }
        for movement in lane_queue:
            index = MOVEMENT_INDEX[movement]
            for vehicle_id in movement_sets[index] if index < len(movement_sets) else []:
                if speeds.get(vehicle_id, 1.0) < 0.1:
                    queued_ids_by_movement[movement].add(vehicle_id)
            lane_queue[movement] = len(queued_ids_by_movement[movement])
        phase_queue = {
            phase: sum(lane_queue[movement] for movement in phase_movements(phase))
            for phase in PHASES
        }
        return {
            "queue_by_lane": lane_queue,
            "queue_by_phase": phase_queue,
            "intersection_total_queue": sum(phase_queue.values()),
        }

    def _get_vlm_actions(self, saved_paths: Dict[str, Any], step_num: int,
                         state_action_log: Any) -> Dict[str, int]:
        actions: Dict[str, int] = {}
        step_actions: Dict[str, Dict[str, Any]] = {}
        for intersection in self.env.list_intersection:
            phases = list(intersection.control_phases)
            if tuple(phases) != PHASES:
                raise RuntimeError(
                    f"V34 requires standard four phases; {intersection.inter_id} has {phases}"
                )
            histories = self.phase_histories.setdefault(
                intersection.inter_id, {phase: [] for phase in phases}
            )
            current_v = {phase: self._phase_current_v(intersection, phase) for phase in phases}
            for phase in phases:
                histories[phase].append(current_v[phase])
            v2_action = max(
                range(len(phases)),
                key=lambda index: (
                    current_v[phases[index]],
                    sum(value > 0 for value in histories[phases[index]]),
                    -index,
                ),
            )
            action = v2_action
            target_sumo_phase = intersection.action_2_phase_index.get(
                action, intersection.current_phase_index
            )
            transition = target_sumo_phase != intersection.current_phase_index
            actions[intersection.inter_id] = action
            step_actions[intersection.inter_id] = {
                "v2_phase": phases[v2_action],
                "v2_action_idx": v2_action,
                "executed_phase": phases[action],
                "executed_action_idx": action,
                "action_source": "v2",
                "current_v_by_phase": current_v,
                "transition_expected": transition,
                "transition_time_s": 5.0 if transition else 0.0,
                "effective_green_time_s": 25.0 if transition else 30.0,
            }
            histories[phases[action]] = []
        self.action_records[int(step_num)] = step_actions
        self.frame_remaining_v[int(step_num)] = {
            intersection.inter_id: {} for intersection in self.env.list_intersection
        }
        self.frame_remaining_queue[int(step_num)] = {
            intersection.inter_id: {} for intersection in self.env.list_intersection
        }
        self._active_decision_step = int(step_num)
        return actions

    def _record_decision_interval_frame(self, inner_i: int, env: Any,
                                        sim_time_s: float = None) -> None:
        # Keep V23's renderer/video timing unchanged, then capture the matching
        # SUMO state on exactly the same callback tick.
        super()._record_decision_interval_frame(inner_i, env, sim_time_s=sim_time_s)
        sample_interval = float(self.dic_traffic_env_conf["VLM_CONFIG"].get(
            "VIDEO_FRAME_SAMPLE_INTERVAL", 5.0
        ))
        sim_interval = float(self.dic_traffic_env_conf.get("INTERVAL", 1.0))
        stride = max(1, int(round(sample_interval / sim_interval)))
        if (int(inner_i) + 1) % stride:
            return
        step_num = self._active_decision_step
        if step_num < 0 or step_num not in self.frame_remaining_v:
            return
        timestamp = float(env.get_current_time() if sim_time_s is None else sim_time_s)
        current_env_time = float(env.get_current_time())
        if abs(current_env_time - timestamp) > 1e-6:
            raise RuntimeError(
                "SUMO time changed between V34 video and state capture: "
                f"video_time={timestamp} state_time={current_env_time}"
            )
        for intersection in env.list_intersection:
            self.frame_remaining_v[step_num][intersection.inter_id][timestamp] = (
                self._controlled_remaining_v(intersection)
            )
            self.frame_remaining_queue[step_num][intersection.inter_id][timestamp] = (
                self._controlled_queue_snapshot(intersection)
            )


def build_collector(args: argparse.Namespace) -> Tuple[V34Collector, str]:
    if args.run_counts <= 0 or args.run_counts % 30:
        raise ValueError("--run-counts must be a positive multiple of 30")
    if args.video_sample_interval <= 0 or not math.isclose(
        30.0 / args.video_sample_interval,
        round(30.0 / args.video_sample_interval),
        abs_tol=1e-9,
    ):
        raise ValueError("--video-sample-interval must be a positive divisor of 30")
    if args.local_render_radius < args.camera_view_distance:
        raise ValueError("--local-render-radius must be >= --camera-view-distance")

    conf = get_traffic_env_conf(args.dataset, eightphase=False)
    scenario_contract = V28_SCENARIO_CONTRACTS.get(args.dataset)
    if scenario_contract:
        mismatches = {
            key: (conf.get(key), expected)
            for key, expected in scenario_contract.items()
            if conf.get(key) != expected
        }
        if mismatches:
            raise RuntimeError(
                f"V34 {args.dataset} configuration differs from V28: {mismatches}"
            )
        invalid_mappings = {
            inter_id: phases
            for inter_id, phases in conf["INTER_PHASE_MAPPING"].items()
            if tuple(phases) != PHASES
        }
        if invalid_mappings:
            raise RuntimeError(
                "V34 requires V28's standard four-phase order "
                f"{PHASES}; invalid mappings: {invalid_mappings}"
            )
    conf.update({
        "MODEL_NAME": "V34",
        "PROJECT_NAME": "V34-Transition-Collection",
        "RUN_COUNTS": int(args.run_counts),
        "MIN_ACTION_TIME": 30,
        "YELLOW_TIME": 5,
        "ALL_RED_TIME": 0,
        "SKIP_TRANSITION_PHASE": False,
        "METRIC_SAMPLE_INTERVAL": 30,
        "INTERVAL": 1.0,
        "SEED": int(args.seed),
        "CAMERA_VIEW_DISTANCE": float(args.camera_view_distance),
        "ENABLE_VIDEO_SFT_EXTRACTION": True,
        "RAISE_INNER_STEP_CALLBACK_ERRORS": True,
        "LIST_STATE_FEATURE": [
            "cur_phase", "time_this_phase", "traffic_movement_vehicle_ids_150m",
            "traffic_movement_vehicle_ids", "lane_num_vehicle",
        ],
    })
    vlm = conf["VLM_CONFIG"]
    vlm.update({
        "DECISION_INPUT_MODE": "video",
        "VIDEO_INCLUDE_CURRENT_IMAGES": False,
        "VIDEO_EXPORT_DIRECTIONS": True,
        "VIDEO_EXPORT_COMPOSITE": False,
        "VIDEO_EXPORT_DIRECTION_SEQUENCE": False,
        "VIDEO_RECORD_MODE": "sampled",
        "VIDEO_FRAME_SAMPLE_INTERVAL": float(args.video_sample_interval),
        "VIDEO_ASYNC_WORKERS": int(args.async_workers),
        "RENDER_PRESET": args.render_preset,
        "RENDERING_BACKEND": args.rendering_backend,
        "TLS_BATCH_SIZE": int(args.batch_size),
        "RENDER_KEEP_BATCH_SENSORS": bool(args.keep_batch_sensors),
        "RENDER_REUSE_BATCH_SENSORS": bool(args.reuse_batch_sensors),
        "RENDER_STEP_TASK_MANAGER": bool(args.step_task_manager),
        "LOCAL_RENDER_RADIUS_M": float(args.local_render_radius),
    })
    timestamp = time.strftime("%m_%d_%H_%M_%S", time.localtime())
    work_dir = args.work_dir or os.path.join(
        "records", "V34", f"{args.dataset}_seed{args.seed}_{timestamp}"
    )
    validate_video_sft_output_dir(os.path.join(work_dir, "video_sft_raw"))
    roadnet, trafficflow = DATASET_IDENTIFIERS[args.dataset]
    path_conf = get_path_conf(args.dataset, work_dir)
    selected_route = _resolve_traffic_file(args, path_conf, conf)
    if args.traffic_file:
        trafficflow = Path(selected_route).stem
    topology_path = os.path.join(path_conf["PATH_TO_DATA"], "network_topology.json")
    if os.path.isfile(topology_path):
        # Avoid VLMOneLine's legacy test_2x2 fallback when the active scenario
        # already provides its own topology file.
        conf["VLM_CONFIG"]["TOPOLOGY_JSON_PATH"] = topology_path
    collector = V34Collector(
        dic_agent_conf={}, dic_traffic_env_conf=conf, dic_path=path_conf,
        roadnet=roadnet, trafficflow=trafficflow,
    )
    return collector, work_dir


def attach_transition_labels(work_dir: str, collector: V34Collector, args: argparse.Namespace) -> None:
    raw_dir = Path(work_dir) / "video_sft_raw"
    labels_path = raw_dir / "v34_executed_transition_labels.jsonl"
    count = 0
    with labels_path.open("w", encoding="utf-8") as output:
        for record_path in sorted((raw_dir / "json").glob("decision_*.json")):
            with record_path.open("r", encoding="utf-8") as handle:
                record = json.load(handle)
            step = int(record["decision_step"])
            tls_id = record["tls_id"]
            if record.get("status") != "complete":
                # The V23 writer already detected an incomplete/extra frame
                # sequence. It must never become a V34 training label.
                continue
            action = collector.action_records[step][tls_id]
            values_by_time = collector.frame_remaining_v[step][tls_id]
            queues_by_time = collector.frame_remaining_queue[step][tls_id]
            actual_times = [float(value) for value in record.get("video_frame_sim_times", [])]
            expected_times = [float(record["sim_start_s"]) + offset for offset in (5, 10, 15, 20, 25, 30)]
            if len(actual_times) != len(expected_times) or any(
                    abs(actual - expected) > 1e-6
                    for actual, expected in zip(actual_times, expected_times)):
                raise RuntimeError(
                    f"Complete V34 sample has noncanonical video times at "
                    f"step={step}, tls={tls_id}: actual={actual_times}, expected={expected_times}"
                )
            frames = []
            for frame_index, sim_time in enumerate(actual_times):
                timestamp = float(sim_time)
                if timestamp not in values_by_time:
                    raise RuntimeError(
                        f"Missing V34 remaining-V at step={step}, tls={tls_id}, time={timestamp}"
                    )
                frames.append({
                    "video_frame_index": frame_index,
                    "sim_time_s": timestamp,
                    "relative_time_s": timestamp - float(record["sim_start_s"]),
                    "intersection_remaining_v": values_by_time[timestamp],
                    "intersection_queue": queues_by_time[timestamp],
                })
            final_value = next(
                item["intersection_remaining_v"]
                for item in frames
                if abs(item["relative_time_s"] - 30.0) <= 1e-6
            )
            final_queue = next(
                item["intersection_queue"]
                for item in frames
                if abs(item["relative_time_s"] - 30.0) <= 1e-6
            )
            label = {
                **action,
                "remaining_v_definition": (
                    "unique vehicles within 150m across all signal-controlled movements; "
                    "right turns are excluded"
                ),
                "intersection_remaining_v_by_video_frame": frames,
                "intersection_remaining_v_at_t30": final_value,
                "queue_definition": (
                    "unique vehicles within 150m in signal-controlled through or "
                    "left-turn movements whose instantaneous SUMO speed is below 0.1 m/s; "
                    "right turns are excluded"
                ),
                "intersection_queue_by_video_frame": [
                    {
                        "video_frame_index": item["video_frame_index"],
                        "sim_time_s": item["sim_time_s"],
                        "relative_time_s": item["relative_time_s"],
                        **item["intersection_queue"],
                    }
                    for item in frames
                ],
                "intersection_queue_at_t30": final_queue,
            }
            record["v34_executed_transition"] = label
            with record_path.open("w", encoding="utf-8") as handle:
                json.dump(record, handle, ensure_ascii=False, indent=2)
            output.write(json.dumps({
                "decision_step": step,
                "tls_id": tls_id,
                "status": record.get("status"),
                "record_path": str(record_path.relative_to(raw_dir)),
                "video_paths": record.get("video_paths", {}),
                "v34_executed_transition": label,
            }, ensure_ascii=False) + "\n")
            count += 1
    metadata = {
        "dataset": args.dataset,
        "seed": args.seed,
        "traffic_file": collector.dic_traffic_env_conf["TRAFFIC_FILE"],
        "decision_interval_s": 30,
        "yellow_time_s": 5,
        "video_frame_offsets_s": [5, 10, 15, 20, 25, 30],
        "policy": "V2: Current V, then effective nonzero history length, then phase order",
        "remaining_v_definition": (
            "unique 150m vehicles in signal-controlled movements only; right turns excluded"
        ),
        "queue_definition": (
            "unique 150m vehicles in signal-controlled through/left movements with "
            "instantaneous SUMO speed < 0.1 m/s; right turns excluded"
        ),
        "label_records": count,
    }
    with (raw_dir / "v34_collection_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    print(f"[V34] attached {count} executed-transition labels: {labels_path}")


def main() -> None:
    args = parse_args()
    os.environ.setdefault("WANDB_MODE", "disabled")
    collector = None
    try:
        collector, work_dir = build_collector(args)
        print(f"[V34] output: {work_dir}")
        print("[V34] policy: exact V2 (no random exploration)")
        collector.train(round=0)
        attach_transition_labels(work_dir, collector, args)
    finally:
        if collector is not None:
            try:
                collector.env.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
