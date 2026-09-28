"""Collect V34-aligned videos and transition labels under LLMLight control.

V35 keeps V34's four-direction video and six-frame SUMO state pipeline. The
executed phase is selected by LLMLight from structured SUMO telemetry; videos
are synchronized observations saved for later multimodal SFT construction and
are not shown to the teacher LLM during collection.
"""

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Tuple

from models.llmlight import LLMLightAgent
from run_v34 import (
    DATASET_IDENTIFIERS,
    PHASES,
    V28_SCENARIO_CONTRACTS,
    V34Collector,
    _resolve_traffic_file,
)
from utils.vlm_config import get_path_conf, get_traffic_env_conf, _load_phase_mapping
from utils.vlm_oneline import VLMOneLine
from utils.video_sft_dataset_writer import validate_video_sft_output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect V35 LLMLight-reasoning video transitions."
    )
    parser.add_argument("--dataset", default="newyork",
                        choices=sorted(DATASET_IDENTIFIERS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-counts", type=int, default=3600)
    parser.add_argument(
        "--traffic-file",
        default=None,
        help=(
            "Traffic input in the selected scenario data directory. A .rou.xml is "
            "used directly; a CityFlow .json is converted to a same-stem .rou.xml."
        ),
    )
    parser.add_argument("--work-dir", default=None)
    parser.add_argument(
        "--data-dir",
        default=None,
        help=(
            "Override the scenario data directory. Use this to run two variants "
            "of the same scenario independently, e.g. data/NewYork/16x3 or "
            "data/NewYork/16x3_v1. The directory must contain the configured "
            "network, SUMO config, topology and phase mapping files."
        ),
    )
    parser.add_argument("--video-sample-interval", type=float, default=5.0)
    parser.add_argument("--camera-view-distance", type=float, default=150.0)
    parser.add_argument("--local-render-radius", type=float, default=180.0)
    parser.add_argument("--render-preset", default="1080P",
                        choices=["320P", "480P", "720P", "1080P"])
    parser.add_argument("--rendering-backend", default="p3headlessgl",
                        choices=["pandagl", "p3headlessgl"])
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument(
        "--render-batch-mode",
        choices=["serial", "parallel"],
        default="serial",
        help=(
            "Batch-render execution mode. The default serial mode preserves "
            "the existing shared-renderer path; parallel uses one isolated "
            "renderer process per worker and is opt-in. "
            "This never makes concurrent calls on the shared renderer."
        ),
    )
    parser.add_argument(
        "--render-workers",
        type=int,
        default=1,
        help=(
            "Number of isolated render workers to use when parallel batch "
            "rendering is enabled (default: 1)."
        ),
    )
    parser.add_argument("--async-workers", type=int, default=4)
    parser.add_argument("--keep-batch-sensors", type=int, choices=[0, 1], default=0)
    parser.add_argument("--reuse-batch-sensors", type=int, choices=[0, 1], default=1)
    parser.add_argument("--step-task-manager", type=int, choices=[0, 1], default=0)
    parser.add_argument("--decision-api-url", default=None)
    parser.add_argument("--decision-api-key", default=None)
    parser.add_argument("--decision-model", default=None)
    parser.add_argument("--decision-temperature", type=float, default=None)
    parser.add_argument("--decision-max-tokens", type=int, default=None)
    parser.add_argument(
        "--stage1-sumo-only",
        type=int,
        choices=[0, 1],
        default=0,
        help=(
            "Bypass the Stage 1 VLM and construct its canonical perception "
            "directly from the current SUMO snapshot; Stage 2 still uses the model."
        ),
    )
    parser.add_argument(
        "--stage2-without-cooperation",
        type=int,
        choices=[0, 1],
        default=0,
        help=(
            "Stage 2 ablation: inject only <local_perception>; omit "
            "<cooperative_perception>. Default 0 preserves the full behavior."
        ),
    )
    parser.add_argument(
        "--decision-enable-thinking",
        type=int,
        choices=[0, 1],
        default=0,
        help="Enable the provider's hidden reasoning mode (default: 0).",
    )
    parser.add_argument(
        "--reasoning-mode",
        choices=["adaptive", "fast", "slow"],
        default="adaptive",
        help=(
            "Stage 2 reasoning mode: adaptive uses the learned router; "
            "fast/slow force the corresponding inference branch."
        ),
    )
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=None,
        help="Override LLMLight decision concurrency; 1 selects serial decisions.",
    )
    parser.add_argument(
        "--checkpoint-every-steps",
        type=int,
        default=0,
        help="Save a resumable checkpoint every N completed decision steps; 0 disables it.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest complete checkpoint under --work-dir.",
    )
    parser.add_argument(
        "--checkpoint-keep",
        type=int,
        default=2,
        help="Keep only the newest N complete resumable checkpoints (default: 2).",
    )
    return parser.parse_args()


class V35Collector(V34Collector):
    """V34 synchronized collection with LLMLight as the executed policy."""

    def _create_agents(self) -> None:
        VLMOneLine._create_agents(self)

    def _build_agent_input(self, tls_id: str, image_paths: Any) -> list:
        """Keep saved videos completely outside the SUMO-only teacher input."""
        return []

    def _get_vlm_actions(self, saved_paths: Dict[str, Any], step_num: int,
                         state_action_log: Any) -> Dict[str, int]:
        decision_sim_time_s = float(self.env.get_current_time())
        actions = VLMOneLine._get_vlm_actions(
            self, saved_paths, step_num, state_action_log
        )
        current_sim_time_s = float(self.env.get_current_time())
        if abs(current_sim_time_s - decision_sim_time_s) > 1e-6:
            raise RuntimeError(
                "SUMO advanced while LLMLight was deciding: "
                f"before={decision_sim_time_s} after={current_sim_time_s}"
            )
        agents_by_tls = {agent.tls_id: agent for agent in self.agents}
        step_actions: Dict[str, Dict[str, Any]] = {}

        for intersection in self.env.list_intersection:
            phases = list(intersection.control_phases)
            if tuple(phases) != PHASES:
                raise RuntimeError(
                    f"V35 requires standard four phases; "
                    f"{intersection.inter_id} has {phases}"
                )
            tls_id = intersection.inter_id
            action = int(actions[tls_id])
            agent = agents_by_tls[tls_id]
            decision = agent.get_last_decision_record()
            if not decision:
                raise RuntimeError(
                    f"LLMLight produced no structured decision record at "
                    f"step={step_num}, tls={tls_id}"
                )
            if int(decision.get("selected_action_idx", -1)) != action:
                raise RuntimeError(
                    f"LLMLight action/record mismatch at step={step_num}, tls={tls_id}: "
                    f"action={action}, record={decision.get('selected_action_idx')}"
                )
            if int(decision.get("decision_step", -1)) != int(step_num):
                raise RuntimeError(
                    f"LLMLight step/record mismatch at tls={tls_id}: "
                    f"expected={step_num}, record={decision.get('decision_step')}"
                )
            if decision.get("tls_id") != tls_id:
                raise RuntimeError(
                    f"LLMLight TLS/record mismatch at step={step_num}: "
                    f"expected={tls_id}, record={decision.get('tls_id')}"
                )
            if decision.get("selected_phase") != phases[action]:
                raise RuntimeError(
                    f"LLMLight phase/record mismatch at step={step_num}, tls={tls_id}: "
                    f"action_phase={phases[action]}, "
                    f"record={decision.get('selected_phase')}"
                )
            decision_source = decision.get("decision_source")
            if decision_source not in {"llm", "fallback_v25"}:
                raise RuntimeError(
                    f"Unexpected LLMLight decision source at step={step_num}, "
                    f"tls={tls_id}: {decision_source!r}"
                )
            expected_eligible = decision_source == "llm"
            if bool(decision.get("reasoning_training_eligible")) != expected_eligible:
                raise RuntimeError(
                    f"LLMLight eligibility/source mismatch at step={step_num}, "
                    f"tls={tls_id}"
                )
            decision["decision_sim_time_s"] = decision_sim_time_s

            target_sumo_phase = intersection.action_2_phase_index.get(
                action, intersection.current_phase_index
            )
            transition = target_sumo_phase != intersection.current_phase_index
            current_v = {
                phase: self._phase_current_v(intersection, phase)
                for phase in phases
            }
            step_actions[tls_id] = {
                "selected_phase": phases[action],
                "selected_action_idx": action,
                "executed_phase": phases[action],
                "executed_action_idx": action,
                "decision_source": decision_source,
                "decision_sim_time_s": decision_sim_time_s,
                "reasoning_training_eligible": bool(
                    decision.get("reasoning_training_eligible", False)
                ),
                "llmlight_decision": decision,
                "current_v_by_phase": current_v,
                "transition_expected": transition,
                "transition_time_s": 5.0 if transition else 0.0,
                "effective_green_time_s": 25.0 if transition else 30.0,
            }

        self.action_records[int(step_num)] = step_actions
        self.frame_remaining_v[int(step_num)] = {
            intersection.inter_id: {} for intersection in self.env.list_intersection
        }
        self.frame_remaining_queue[int(step_num)] = {
            intersection.inter_id: {} for intersection in self.env.list_intersection
        }
        self._active_decision_step = int(step_num)
        return actions


def _validate_args(args: argparse.Namespace) -> None:
    if args.run_counts <= 0 or args.run_counts % 30:
        raise ValueError("--run-counts must be a positive multiple of 30")
    if getattr(args, "batch_size", 1) <= 0:
        raise ValueError("--batch-size must be positive")
    if not math.isclose(args.video_sample_interval, 5.0, abs_tol=1e-9):
        raise ValueError(
            "V35 requires --video-sample-interval 5.0 so the six video frames "
            "match coordination times 0,5,10,15,20,25s"
        )
    if args.local_render_radius < args.camera_view_distance:
        raise ValueError("--local-render-radius must be >= --camera-view-distance")
    parallel_workers = getattr(args, "parallel_workers", None)
    if parallel_workers is not None and parallel_workers <= 0:
        raise ValueError("--parallel-workers must be positive")
    if getattr(args, "render_workers", 1) <= 0:
        raise ValueError("--render-workers must be positive")
    decision_max_tokens = getattr(args, "decision_max_tokens", None)
    if decision_max_tokens is not None and decision_max_tokens <= 0:
        raise ValueError("--decision-max-tokens must be positive")
    if getattr(args, "checkpoint_every_steps", 0) < 0:
        raise ValueError("--checkpoint-every-steps must be >= 0")
    if getattr(args, "checkpoint_keep", 1) < 1:
        raise ValueError("--checkpoint-keep must be >= 1")
    if getattr(args, "resume", False) and not getattr(args, "work_dir", None):
        raise ValueError("--resume requires the original explicit --work-dir")


def _validate_collector_runtime(collector: V35Collector) -> None:
    missing = []
    sumo_stage1_only = bool(
        (collector.dic_traffic_env_conf.get("VLM_CONFIG", {}) or {}).get(
            "VLM_STAGE1_SUMO_ONLY", False
        )
    )
    if not sumo_stage1_only:
        if collector.renderer is None:
            missing.append("3D renderer")
        if collector.decision_window_recorder is None:
            missing.append("direction video recorder")
        if collector.video_sft_dataset_writer is None:
            missing.append("video SFT dataset writer")
    if not (collector._network_topology.get("intersections") or {}):
        missing.append("network topology")
    if missing:
        raise RuntimeError(
            "V35 cannot start because required collection components are "
            f"unavailable: {', '.join(missing)}"
        )


def build_collector(args: argparse.Namespace) -> Tuple[V35Collector, str]:
    _validate_args(args)
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
                f"V35 {args.dataset} configuration differs from V28: {mismatches}"
            )
        invalid_mappings = {
            inter_id: phases
            for inter_id, phases in conf["INTER_PHASE_MAPPING"].items()
            if tuple(phases) != PHASES
        }
        if invalid_mappings:
            raise RuntimeError(
                "V35 requires V28's standard four-phase order "
                f"{PHASES}; invalid mappings: {invalid_mappings}"
            )

    conf.update({
        "MODEL_NAME": "V35",
        "PROJECT_NAME": "V35-LLMLight-Reasoning-Collection",
        "RUN_COUNTS": int(args.run_counts),
        "MIN_ACTION_TIME": 30,
        "YELLOW_TIME": 5,
        "ALL_RED_TIME": 0,
        "SKIP_TRANSITION_PHASE": False,
        "METRIC_SAMPLE_INTERVAL": 30,
        "INTERVAL": 1.0,
        "SEED": int(args.seed),
        "CAMERA_VIEW_DISTANCE": float(args.camera_view_distance),
        "ENABLE_VIDEO_SFT_EXTRACTION": not bool(args.stage1_sumo_only),
        "ENABLE_NEW_COORDINATION": True,
        "VLM_CONFIG": {
            **conf.get("VLM_CONFIG", {}),
            "VLM_STAGE2_INCLUDE_COOPERATION": not bool(args.stage2_without_cooperation),
            "RESUME_FROM_CHECKPOINT": bool(args.resume),
        },
        "V36_COORDINATION_SPEED_MPS": 11.0,
        "RAISE_INNER_STEP_CALLBACK_ERRORS": True,
        "CHECKPOINT_EVERY_STEPS": int(args.checkpoint_every_steps),
        "CHECKPOINT_KEEP": int(args.checkpoint_keep),
        "RESUME_FROM_CHECKPOINT": bool(args.resume),
        # SUMO saveState only persists its RNG when this launch option is set.
        "SAVE_STATE_RNG": bool(args.checkpoint_every_steps or args.resume),
        # Keep rendering enabled while selecting actions from SUMO-only telemetry.
        "SUMO_ONLY_MODE": bool(args.stage1_sumo_only),
        "SUMO_ONLY_AGENT_INPUT": True,
        "V9_TEMPORAL_FRAME_INTERVAL": 5,
        "V33_TEMPORAL_FRAME_INTERVAL": 5,
        "LIST_STATE_FEATURE": [
            "cur_phase",
            "time_this_phase",
            "traffic_movement_vehicle_ids_150m",
            "traffic_movement_vehicle_ids",
            "v3_counting_line_flow_history",
            "v9_cycle_150m_history",
            "v26_cycle_queue_history",
            "v32_cycle_vehicle_snapshots",
            "v36_cycle_outbound_snapshots",
            "vehicle_distance",
            "vehicle_speed",
            "waiting_vehicle_list",
            "traffic_movement_pressure_queue",
            "lane_num_vehicle",
        ],
    })
    vlm = conf["VLM_CONFIG"]
    vlm.update({
        "MIN_ACTION_TIME": 30,
        "CAMERA_VIEW_DISTANCE": float(args.camera_view_distance),
        "SEED": int(args.seed),
        "DECISION_INPUT_MODE": "video",
        "VIDEO_INCLUDE_CURRENT_IMAGES": False,
        "VIDEO_EXPORT_DIRECTIONS": not bool(args.stage1_sumo_only),
        "VIDEO_EXPORT_COMPOSITE": False,
        "VIDEO_EXPORT_DIRECTION_SEQUENCE": False,
        "VIDEO_RECORD_MODE": "sampled",
        "VIDEO_FRAME_SAMPLE_INTERVAL": float(args.video_sample_interval),
        "VIDEO_ASYNC_WORKERS": int(args.async_workers),
        "RENDER_PRESET": args.render_preset,
        "RENDERING_BACKEND": args.rendering_backend,
        "VLM_STAGE1_SUMO_ONLY": bool(getattr(args, "stage1_sumo_only", 0)),
        "TLS_BATCH_SIZE": int(args.batch_size),
        # Keep render execution controls separate from TLS_BATCH_SIZE:
        # the latter controls sensor partitioning, while these fields select
        # the execution backend. The default remains serial.
        "RENDER_BATCH_MODE": str(args.render_batch_mode),
        "RENDER_WORKERS": int(args.render_workers),
        "RENDER_KEEP_BATCH_SENSORS": bool(args.keep_batch_sensors),
        "RENDER_REUSE_BATCH_SENSORS": bool(args.reuse_batch_sensors),
        "RENDER_STEP_TASK_MANAGER": bool(args.step_task_manager),
        "LOCAL_RENDER_RADIUS_M": float(args.local_render_radius),
    })
    optional_overrides = {
        "DECISION_API_URL": args.decision_api_url,
        "DECISION_API_KEY": args.decision_api_key,
        "DECISION_MODEL": args.decision_model,
        "DECISION_TEMPERATURE": args.decision_temperature,
        "DECISION_MAX_TOKENS": args.decision_max_tokens,
        "DECISION_ENABLE_THINKING": bool(args.decision_enable_thinking),
        "VLM_REASONING_MODE": args.reasoning_mode,
    }
    vlm.update({key: value for key, value in optional_overrides.items()
                if value is not None})
    if args.parallel_workers is not None:
        vlm["PARALLEL_ENABLED"] = args.parallel_workers > 1
        vlm["PARALLEL_WORKERS"] = int(args.parallel_workers)

    timestamp = time.strftime("%m_%d_%H_%M_%S", time.localtime())
    work_dir = args.work_dir or os.path.join(
        "records", "V35", f"{args.dataset}_seed{args.seed}_{timestamp}"
    )
    if not args.resume:
        validate_video_sft_output_dir(os.path.join(work_dir, "video_sft_raw"))
    roadnet, trafficflow = DATASET_IDENTIFIERS[args.dataset]
    path_conf = get_path_conf(args.dataset, work_dir)
    if args.data_dir:
        data_dir = Path(args.data_dir).expanduser()
        if not data_dir.is_absolute():
            data_dir = Path(__file__).resolve().parent / data_dir
        data_dir = data_dir.resolve()
        if not data_dir.is_dir():
            raise FileNotFoundError(f"V35 data directory does not exist: {data_dir}")
        path_conf["PATH_TO_DATA"] = str(data_dir)
        phase_mapping_path = data_dir / "newyork_phase_mapping.json"
        if phase_mapping_path.is_file():
            with phase_mapping_path.open("r", encoding="utf-8") as f:
                phase_mapping = json.load(f)
            # Keep the same four-phase contract while using the selected
            # directory's mapping rather than the default scenario mapping.
            conf["INTER_PHASE_MAPPING"] = {
                inter_id: phases[:4] for inter_id, phases in phase_mapping.items()
            }
            conf["VLM_CONFIG"]["PHASE_MAPPING_PATH"] = str(phase_mapping_path)
        print(f"Using overridden scenario data directory: {data_dir}")
    selected_route = _resolve_traffic_file(args, path_conf, conf)
    if args.traffic_file:
        trafficflow = Path(selected_route).stem
    topology_path = os.path.join(path_conf["PATH_TO_DATA"], "network_topology.json")
    if not os.path.isfile(topology_path):
        raise FileNotFoundError(
            f"V35 coordination topology does not exist: {topology_path}"
        )
    vlm["TOPOLOGY_JSON_PATH"] = topology_path

    collector = V35Collector(
        dic_agent_conf={},
        dic_traffic_env_conf=conf,
        dic_path=path_conf,
        roadnet=roadnet,
        trafficflow=trafficflow,
        agent_class=LLMLightAgent,
    )
    try:
        _validate_collector_runtime(collector)
    except Exception:
        try:
            collector.env.close()
        except Exception:
            pass
        raise
    return collector, work_dir


def attach_reasoning_transition_labels(
        work_dir: str, collector: V35Collector, args: argparse.Namespace) -> None:
    raw_dir = Path(work_dir) / "video_sft_raw"
    labels_path = raw_dir / "v35_llm_reasoning_labels.jsonl"
    eligible_path = raw_dir / "v35_llm_reasoning_eligible.jsonl"
    total_count = 0
    llm_count = 0
    fallback_count = 0
    completed_records = {}

    with (
        labels_path.open("w", encoding="utf-8") as all_output,
        eligible_path.open("w", encoding="utf-8") as eligible_output,
    ):
        for record_path in sorted((raw_dir / "json").glob("decision_*.json")):
            with record_path.open("r", encoding="utf-8") as handle:
                record = json.load(handle)
            if record.get("status") != "complete":
                continue

            step = int(record["decision_step"])
            tls_id = record["tls_id"]
            action = collector.action_records[step][tls_id]
            decision_sim_time_s = float(action["decision_sim_time_s"])
            interval_start_s = float(record["sim_start_s"])
            if abs(decision_sim_time_s - interval_start_s) > 1e-6:
                raise RuntimeError(
                    f"V35 decision/video interval mismatch at step={step}, "
                    f"tls={tls_id}: decision={decision_sim_time_s}, "
                    f"video_start={interval_start_s}"
                )
            values_by_time = collector.frame_remaining_v[step][tls_id]
            queues_by_time = collector.frame_remaining_queue[step][tls_id]
            actual_times = [
                float(value) for value in record.get("video_frame_sim_times", [])
            ]
            expected_times = [
                float(record["sim_start_s"]) + offset
                for offset in (5, 10, 15, 20, 25, 30)
            ]
            if len(actual_times) != len(expected_times) or any(
                abs(actual - expected) > 1e-6
                for actual, expected in zip(actual_times, expected_times)
            ):
                raise RuntimeError(
                    f"Complete V35 sample has noncanonical video times at "
                    f"step={step}, tls={tls_id}: "
                    f"actual={actual_times}, expected={expected_times}"
                )

            frames = []
            for frame_index, sim_time in enumerate(actual_times):
                if sim_time not in values_by_time or sim_time not in queues_by_time:
                    raise RuntimeError(
                        f"Missing V35 SUMO frame label at step={step}, "
                        f"tls={tls_id}, time={sim_time}"
                    )
                frames.append({
                    "video_frame_index": frame_index,
                    "sim_time_s": sim_time,
                    "relative_time_s": sim_time - float(record["sim_start_s"]),
                    "intersection_remaining_v": values_by_time[sim_time],
                    "intersection_queue": queues_by_time[sim_time],
                })

            final_frame = next(
                item for item in frames
                if abs(item["relative_time_s"] - 30.0) <= 1e-6
            )
            label = {
                **action,
                "remaining_v_definition": (
                    "unique vehicles within 150m across all signal-controlled "
                    "movements; right turns are excluded"
                ),
                "intersection_remaining_v_by_video_frame": frames,
                "intersection_remaining_v_at_t30": final_frame[
                    "intersection_remaining_v"
                ],
                "queue_definition": (
                    "unique vehicles within 150m in signal-controlled through or "
                    "left-turn movements whose instantaneous SUMO speed is below "
                    "0.1 m/s; right turns are excluded"
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
                "intersection_queue_at_t30": final_frame["intersection_queue"],
            }
            record["v35_llm_reasoning_transition"] = label
            with record_path.open("w", encoding="utf-8") as handle:
                json.dump(record, handle, ensure_ascii=False, indent=2)

            index_record = {
                "decision_step": step,
                "tls_id": tls_id,
                "status": record.get("status"),
                "record_path": str(record_path.relative_to(raw_dir)),
                "video_paths": record.get("video_paths", {}),
                "decision_source": action["decision_source"],
                "reasoning_training_eligible": action[
                    "reasoning_training_eligible"
                ],
                "v35_llm_reasoning_transition": label,
            }
            line = json.dumps(index_record, ensure_ascii=False) + "\n"
            all_output.write(line)
            completed_records[(step, tls_id)] = (record, index_record)
            total_count += 1
            if action["reasoning_training_eligible"]:
                if action["decision_source"] != "llm":
                    raise RuntimeError(
                        f"Non-LLM sample marked training eligible at "
                        f"step={step}, tls={tls_id}"
                    )
            else:
                fallback_count += 1

        # A video from step k observes the transition after action k.  Its last
        # frame is the state used to make action k+1, so only the next decision
        # is a causally valid visual-reasoning target.
        for (step, tls_id), (source, source_index) in sorted(
                completed_records.items()):
            target_pair = completed_records.get((step + 1, tls_id))
            if target_pair is None:
                continue
            target, target_index = target_pair
            target_action = collector.action_records[step + 1][tls_id]
            if not target_action["reasoning_training_eligible"]:
                continue
            if target_action["decision_source"] != "llm":
                raise RuntimeError(
                    f"Non-LLM target marked training eligible at "
                    f"step={step + 1}, tls={tls_id}"
                )

            source_frame_times = [
                float(value) for value in source["video_frame_sim_times"]
            ]
            source_last_frame_s = source_frame_times[-1]
            source_end_s = float(source["sim_end_s"])
            target_start_s = float(target["sim_start_s"])
            target_decision_s = float(target_action["decision_sim_time_s"])
            boundary_times = (
                source_last_frame_s,
                source_end_s,
                target_start_s,
                target_decision_s,
            )
            if any(abs(value - boundary_times[0]) > 1e-6
                   for value in boundary_times[1:]):
                raise RuntimeError(
                    f"V35 cross-step reasoning mismatch at source_step={step}, "
                    f"target_step={step + 1}, tls={tls_id}: "
                    f"last_frame={source_last_frame_s}, source_end={source_end_s}, "
                    f"target_start={target_start_s}, "
                    f"target_decision={target_decision_s}"
                )

            reasoning_pair = {
                "source_decision_step": step,
                "target_decision_step": step + 1,
                "tls_id": tls_id,
                "observation_record_path": source_index["record_path"],
                "video_paths": source_index["video_paths"],
                "observation_start_s": float(source["sim_start_s"]),
                "observation_frame_sim_times": source_frame_times,
                "observation_end_s": source_end_s,
                "target_decision_sim_time_s": target_decision_s,
                "decision_source": "llm",
                "reasoning_training_eligible": True,
                "llmlight_decision_target": target_action,
                "alignment_rule": (
                    "source final video/state frame == source sim_end == "
                    "target sim_start == target decision time"
                ),
            }
            eligible_output.write(
                json.dumps(reasoning_pair, ensure_ascii=False) + "\n"
            )
            llm_count += 1

    metadata = {
        "dataset": args.dataset,
        "seed": args.seed,
        "traffic_file": collector.dic_traffic_env_conf["TRAFFIC_FILE"],
        "decision_interval_s": 30,
        "yellow_time_s": 5,
        "decision_state_offset_s": 0,
        "video_frame_offsets_s": [5, 10, 15, 20, 25, 30],
        "absolute_time_rule": (
            "decision_sim_time_s equals sim_start_s; each video/state frame time "
            "equals sim_start_s plus its video_frame_offsets_s value"
        ),
        "teacher_policy": "LLMLight structured-state LLM with V25 fallback",
        "teacher_visual_input": False,
        "saved_visual_observation": "four directional synchronized videos",
        "training_eligibility_rule": (
            "source video/state from step k is paired with the LLM decision at "
            "step k+1; source final frame, source sim_end, target sim_start, and "
            "target decision time must be identical"
        ),
        "same_step_labels_are_transition_audit_only": True,
        "label_records": total_count,
        "llm_reasoning_pairs": llm_count,
        "fallback_v25_records": fallback_count,
        "all_labels_path": labels_path.name,
        "eligible_labels_path": eligible_path.name,
    }
    metadata_path = raw_dir / "v35_collection_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    print(
        f"[V35] attached {total_count} labels: "
        f"llm={llm_count}, fallback_v25={fallback_count}"
    )
    print(f"[V35] all labels: {labels_path}")
    print(f"[V35] reasoning-training eligible only: {eligible_path}")


def main() -> None:
    args = parse_args()
    os.environ.setdefault("WANDB_MODE", "disabled")
    collector = None
    try:
        collector, work_dir = build_collector(args)
        print(f"[V35] output: {work_dir}")
        print("[V35] policy: LLMLight; every non-LLM action uses V25 fallback")
        collector.train(round=0)
        attach_reasoning_transition_labels(work_dir, collector, args)
    finally:
        if collector is not None:
            try:
                collector.env.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
