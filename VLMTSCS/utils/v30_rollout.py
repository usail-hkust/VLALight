"""Local one-intersection-at-a-time SUMO counterfactual rollouts for V30."""

import multiprocessing as mp
import os
import queue
import signal
import time
from copy import deepcopy

from .sumo_env import SUMOEnv
from .v28_rollout import _phase_vehicle_ids
from .v30_contract import (
    DEFAULT_ROLLOUT_WORKERS,
    ENDPOINT_DISTANCE_M,
    resolve_rollout_worker_count,
    validate_candidate_tick_trace,
    validate_restored_state_fingerprint,
    validate_rollout_contract,
    retain_recent_snapshot_group,
)


_MOVEMENT_INDEX = {
    "WL": 0, "WT": 1, "WR": 2,
    "EL": 3, "ET": 4, "ER": 5,
    "NL": 6, "NT": 7, "NR": 8,
    "SL": 9, "ST": 10, "SR": 11,
}


def _phase_movements(phase):
    if "_" in phase:
        return [item for item in phase.split("_") if item]
    return [phase[i:i + 2] for i in range(0, len(phase), 2)]


def _controlled_queue_snapshot(intersection):
    movement_sets = intersection.dic_feature.get(
        "traffic_movement_vehicle_ids_150m", [])
    speeds = intersection.dic_vehicle_speed_current_step
    controlled_movements = sorted({
        movement
        for phase in intersection.control_phases
        for movement in _phase_movements(phase)
    })
    queue_by_lane = {}
    for movement in controlled_movements:
        index = _MOVEMENT_INDEX.get(movement)
        vehicle_ids = (
            movement_sets[index]
            if index is not None and index < len(movement_sets)
            else []
        )
        queue_by_lane[movement] = sum(
            speeds.get(vehicle_id, 1.0) < 0.1
            for vehicle_id in set(vehicle_ids or []))
    queue_by_phase = {
        phase: sum(queue_by_lane.get(movement, 0)
                   for movement in _phase_movements(phase))
        for phase in intersection.control_phases
    }
    return {
        "queue_by_lane": queue_by_lane,
        "queue_by_phase": queue_by_phase,
        "intersection_total_queue": sum(queue_by_phase.values()),
    }


def _intersection_controlled_vehicle_ids(intersection):
    """Return the 150m vehicles in every movement controlled by this TLS."""
    vehicle_ids = set()
    for phase_name in intersection.control_phases:
        vehicle_ids.update(_phase_vehicle_ids(intersection, phase_name))
    return vehicle_ids


def _controlled_vehicle_ids_from_lane_snapshot(intersection, lane_snapshot):
    vehicle_ids = set()
    for phase_name in intersection.control_phases:
        vehicle_ids.update(
            intersection.phase_vehicle_ids_from_lane_snapshot(
                phase_name, lane_snapshot))
    return vehicle_ids


def _rollout_worker(worker_idx, config, path_config, work_dir, control_queue,
                    task_queue, result_queue):
    worker_path = deepcopy(path_config)
    worker_path["PATH_TO_WORK_DIRECTORY"] = work_dir
    worker_config = deepcopy(config)
    worker_config["USE_GUI"] = False
    # Counterfactual branches are deliberately SUMO-only. The main process
    # owns the V34-style renderer and video/state recorder; workers must never
    # create or inherit those outputs even when the parent config enables them.
    worker_config["ENABLE_VIDEO_SFT_EXTRACTION"] = False
    worker_config["ENABLE_VEHICLE_POSITION_SNAPSHOT"] = False
    worker_config["VLM_CONFIG"] = {}
    worker_config["ENABLE_COUNTERFACTUAL_DISCHARGE_LOG"] = False
    worker_config["RAISE_INNER_STEP_CALLBACK_ERRORS"] = True
    os.makedirs(work_dir, exist_ok=True)

    env = SUMOEnv(
        path_to_log=work_dir,
        path_to_work_directory=work_dir,
        dic_traffic_env_conf=worker_config,
        dic_path=worker_path,
        inter_phase_mapping=worker_config.get("INTER_PHASE_MAPPING", {}),
    )

    def handle_termination(signum, frame):
        env.close()
        raise SystemExit(128 + int(signum))

    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, handle_termination)
    context = None
    try:
        env.reset(use_gui=False, seed=worker_config.get("SEED"), verbose=False)
        result_queue.put({"type": "ready", "worker_idx": worker_idx})
        while True:
            try:
                control = control_queue.get_nowait()
            except queue.Empty:
                control = None
            if control is not None:
                if control.get("type") == "stop":
                    break
                if control.get("type") == "prepare":
                    context = control
                    result_queue.put({
                        "type": "prepared",
                        "worker_idx": worker_idx,
                        "task_id": context["task_id"],
                    })

            try:
                command = task_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if command.get("type") != "rollout":
                continue
            task_id = command["task_id"]
            target_tls = command["target_tls"]
            candidate_idx = int(command["candidate_idx"])
            if context is None or context.get("task_id") != task_id:
                result_queue.put({
                    "type": "error", "worker_idx": worker_idx,
                    "task_id": task_id, "target_tls": target_tls,
                    "candidate_idx": candidate_idx,
                    "error": "worker has no matching prepared decision context",
                })
                continue
            try:
                started = time.monotonic()
                env.load_from_file(
                    context["snapshot_path"], quiet=True, raise_on_error=True)
                env.restore_rollout_control_state(context["control_state"])
                restored_fingerprint = env.capture_rollout_state_fingerprint()
                expected_fingerprint = context["start_state_fingerprint"]
                restore_audit = validate_restored_state_fingerprint(
                    expected_fingerprint, restored_fingerprint)

                action_dict = dict(context["baseline_actions"])
                action_dict[target_tls] = candidate_idx
                inter = next(
                    item for item in env.list_intersection
                    if item.inter_id == target_tls)
                phase = inter.control_phases[candidate_idx]
                target_sumo_idx = inter.action_2_phase_index.get(
                    candidate_idx, inter.current_phase_index)
                start_sumo_idx = int(
                    env.traci_conn.trafficlight.getPhase(inter.tls_id))
                if inter.is_in_transition:
                    raise RuntimeError(
                        f"V30 candidate {target_tls}/{phase} starts while the "
                        "Python signal controller is still in transition"
                    )
                if start_sumo_idx != int(inter.current_phase_index):
                    raise RuntimeError(
                        f"V30 candidate {target_tls}/{phase} starts from "
                        "misaligned SUMO/Python phases: "
                        f"sumo={start_sumo_idx}, python={inter.current_phase_index}"
                    )
                transition_applied = target_sumo_idx != inter.current_phase_index
                candidate_phase_discharged = set()
                intersection_discharged = set()
                sampled_times = []
                sampled_sumo_phases = []
                sampled_transition_flags = []
                sampled_controller_phases = []
                counting_line = float(worker_config.get(
                    "DISCHARGE_COUNTING_LINE_DISTANCE", 50.0))

                def collect_discharged(inner_i, env):
                    sampled_times.append(float(env.get_current_time()))
                    sampled_sumo_phases.append(int(
                        env.traci_conn.trafficlight.getPhase(inter.tls_id)))
                    sampled_transition_flags.append(bool(inter.is_in_transition))
                    sampled_controller_phases.append(int(
                        inter.current_phase_index))
                    previous_ids = inter.phase_vehicle_ids_from_lane_snapshot(
                        phase, inter.dic_lane_vehicle_previous_step)
                    current_ids = inter.phase_vehicle_ids_from_lane_snapshot(
                        phase, inter.dic_lane_vehicle_current_step)
                    for vehicle_id in previous_ids - current_ids:
                        previous_distance = (
                            inter.dic_vehicle_distance_previous_step.get(vehicle_id)
                        )
                        if (previous_distance is not None
                                and previous_distance <= counting_line):
                            candidate_phase_discharged.add(vehicle_id)

                    previous_controlled_ids = (
                        _controlled_vehicle_ids_from_lane_snapshot(
                            inter, inter.dic_lane_vehicle_previous_step))
                    current_controlled_ids = (
                        _controlled_vehicle_ids_from_lane_snapshot(
                            inter, inter.dic_lane_vehicle_current_step))
                    for vehicle_id in previous_controlled_ids - current_controlled_ids:
                        previous_distance = (
                            inter.dic_vehicle_distance_previous_step.get(vehicle_id)
                        )
                        if (previous_distance is not None
                                and previous_distance <= counting_line):
                            intersection_discharged.add(vehicle_id)

                env.dic_traffic_env_conf["_SUPPRESS_SIGNAL_LOG"] = True
                env.step(
                    action_dict,
                    min_action_time=float(context["min_action_time"]),
                    inner_step_callback=collect_discharged,
                )
                env.dic_traffic_env_conf["_SUPPRESS_SIGNAL_LOG"] = False

                end_sim_time_s = float(env.get_current_time())
                expected_end_time_s = (
                    float(context["start_sim_time_s"])
                    + float(context["min_action_time"])
                )
                if abs(end_sim_time_s - expected_end_time_s) > 1e-6:
                    raise RuntimeError(
                        "V30 rollout horizon mismatch: "
                        f"start={context['start_sim_time_s']} "
                        f"end={end_sim_time_s} expected={expected_end_time_s}"
                    )

                timing_audit = validate_candidate_tick_trace(
                    start_time_s=context["start_sim_time_s"],
                    sample_times_s=sampled_times,
                    sampled_sumo_phases=sampled_sumo_phases,
                    sampled_transition_flags=sampled_transition_flags,
                    sampled_controller_phases=sampled_controller_phases,
                    start_sumo_phase=start_sumo_idx,
                    target_sumo_phase=target_sumo_idx,
                    transition_applied=transition_applied,
                )
                if inter.is_in_transition:
                    raise RuntimeError(
                        f"V30 candidate {target_tls}/{phase} still has an "
                        "active transition at t+30"
                    )
                end_sumo_idx = int(
                    env.traci_conn.trafficlight.getPhase(inter.tls_id))
                if (int(inter.current_phase_index) != int(target_sumo_idx)
                        or end_sumo_idx != int(target_sumo_idx)):
                    raise RuntimeError(
                        f"V30 candidate {target_tls}/{phase} did not end on "
                        f"the target green: target={target_sumo_idx}, "
                        f"sumo={end_sumo_idx}, python={inter.current_phase_index}"
                    )
                unsettled_actions = {}
                for active_inter in env.list_intersection:
                    action_idx = int(action_dict[active_inter.inter_id])
                    expected_phase_idx = int(
                        active_inter.action_2_phase_index[action_idx])
                    if (active_inter.is_in_transition
                            or int(active_inter.current_phase_index)
                            != expected_phase_idx):
                        unsettled_actions[active_inter.inter_id] = {
                            "action_idx": action_idx,
                            "expected_phase_idx": expected_phase_idx,
                            "current_phase_idx": int(
                                active_inter.current_phase_index),
                            "is_in_transition": bool(
                                active_inter.is_in_transition),
                        }
                if unsettled_actions:
                    raise RuntimeError(
                        "V30 candidate ended before every target/V25 action "
                        f"settled on green: {unsettled_actions}"
                    )

                phase_remaining_ids = _phase_vehicle_ids(inter, phase)
                intersection_remaining_ids = _intersection_controlled_vehicle_ids(
                    inter)
                phase_queue_count = sum(
                    inter.dic_vehicle_speed_current_step.get(vehicle_id, 1.0) < 0.1
                    for vehicle_id in phase_remaining_ids
                )
                intersection_queue_count = sum(
                    inter.dic_vehicle_speed_current_step.get(vehicle_id, 1.0) < 0.1
                    for vehicle_id in intersection_remaining_ids
                )
                intersection_queue = _controlled_queue_snapshot(inter)
                end_control_state = env.capture_rollout_control_state()
                end_state_fingerprint = env.capture_rollout_state_fingerprint()

                if (intersection_queue["intersection_total_queue"]
                        != intersection_queue_count):
                    raise RuntimeError(
                        f"V30 endpoint queue representations disagree for "
                        f"{target_tls}/{phase}: unique={intersection_queue_count}, "
                        f"by_movement={intersection_queue}"
                    )

                end_snapshot_path = os.path.join(
                    work_dir,
                    "end_states",
                    f"{task_id.replace(':', '_')}_{target_tls}_{candidate_idx}.xml",
                )
                os.makedirs(os.path.dirname(end_snapshot_path), exist_ok=True)
                if env.snapshot(end_snapshot_path) is None:
                    raise RuntimeError(
                        f"V30 failed to save counterfactual end state: "
                        f"{end_snapshot_path}"
                    )
                if (not os.path.isfile(end_snapshot_path)
                        or os.path.getsize(end_snapshot_path) <= 0):
                    raise RuntimeError(
                        f"V30 counterfactual end snapshot is missing or empty: "
                        f"{end_snapshot_path}"
                    )
                post_save_fingerprint = env.capture_rollout_state_fingerprint()
                if post_save_fingerprint["sha256"] != end_state_fingerprint["sha256"]:
                    raise RuntimeError(
                        "V30 state changed while saving the candidate endpoint: "
                        f"before={end_state_fingerprint}, "
                        f"after={post_save_fingerprint}"
                    )
                result_queue.put({
                    "type": "result",
                    "worker_idx": worker_idx,
                    "task_id": task_id,
                    "target_tls": target_tls,
                    "candidate_idx": candidate_idx,
                    "phase": phase,
                    "transition_applied": transition_applied,
                    "transition_time_s": (
                        float(worker_config.get("YELLOW_TIME", 0))
                        if transition_applied else 0.0
                    ),
                    "effective_green_time_s": (
                        float(context["min_action_time"])
                        - float(worker_config.get("YELLOW_TIME", 0))
                        if transition_applied
                        else float(context["min_action_time"])
                    ),
                    "discharged_30s": len(intersection_discharged),
                    "candidate_phase_discharged_30s": len(
                        candidate_phase_discharged),
                    "discharged_definition": (
                        "unique vehicles crossing the counting line from any "
                        "signal-controlled movement at the target intersection"
                    ),
                    "discharge_counting_line_distance_m": counting_line,
                    # Kept for backward-compatible audit logs only.  It is not
                    # comparable across phases because its movement set changes.
                    "remaining_v_30s": len(phase_remaining_ids),
                    "remaining_queue_30s": phase_queue_count,
                    "intersection_remaining_v_30s": len(
                        intersection_remaining_ids),
                    "intersection_remaining_queue_30s": intersection_queue_count,
                    "intersection_queue_at_t30": intersection_queue,
                    "endpoint_metric_distance_m": ENDPOINT_DISTANCE_M,
                    "remaining_v_definition": (
                        "unique vehicles within 150m across all signal-controlled "
                        "movements; right turns are excluded"
                    ),
                    "queue_definition": (
                        "unique vehicles within 150m in signal-controlled "
                        "movements with instantaneous SUMO speed below 0.1 m/s; "
                        "right turns are excluded"
                    ),
                    "target_phase_timing_audit": timing_audit,
                    "start_snapshot_restore_audit": restore_audit,
                    "end_snapshot_path": os.path.abspath(end_snapshot_path),
                    "end_control_state": end_control_state,
                    "end_state_fingerprint": end_state_fingerprint,
                    "end_sim_time_s": end_sim_time_s,
                    "elapsed_s": time.monotonic() - started,
                })
            except BaseException as exc:
                env.dic_traffic_env_conf["_SUPPRESS_SIGNAL_LOG"] = False
                result_queue.put({
                    "type": "error", "worker_idx": worker_idx,
                    "task_id": task_id, "target_tls": target_tls,
                    "candidate_idx": candidate_idx, "error": repr(exc),
                })
    finally:
        env.close()


class V30RolloutPool:
    """Evaluate each TLS candidate while all other TLS use a V25 baseline."""

    def __init__(self, config, path_config, work_dir,
                 worker_count=DEFAULT_ROLLOUT_WORKERS,
                 timeout_s=7200.0):
        validate_rollout_contract(config)
        self.config = config
        try:
            available_cpus = len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            available_cpus = os.cpu_count() or 1
        self.worker_count = resolve_rollout_worker_count(
            worker_count, available_cpus=available_cpus)
        self.timeout_s = float(timeout_s)
        for variable in (
                "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[variable] = "1"
        print(
            f"[V30] Starting {self.worker_count} parallel SUMO workers "
            f"(requested={int(worker_count)}, available_cpus={available_cpus})"
        )
        self.work_dir = os.path.abspath(work_dir)
        self.snapshot_dir = os.path.join(self.work_dir, "snapshots")
        os.makedirs(self.snapshot_dir, exist_ok=True)
        self.snapshot_retention_decisions = max(
            0, int(self.config.get("V30_SNAPSHOT_RETENTION_DECISIONS", 5)))
        # A completed decision owns one start snapshot and all of its endpoint
        # snapshots. Prune only whole decision groups after all candidates have
        # finished, so parallel workers never lose an input they still need.
        self._snapshot_history = []
        ctx = mp.get_context("spawn")
        self.task_queue = ctx.Queue()
        self.result_queue = ctx.Queue()
        self.control_queues = []
        self.processes = []
        for worker_idx in range(self.worker_count):
            control_queue = ctx.Queue()
            process = ctx.Process(
                target=_rollout_worker,
                args=(worker_idx, config, path_config,
                      os.path.join(work_dir, f"worker_{worker_idx}"),
                      control_queue, self.task_queue, self.result_queue),
                name=f"v30-sumo-{worker_idx}", daemon=True,
            )
            process.start()
            self.control_queues.append(control_queue)
            self.processes.append(process)
        try:
            self._wait_for("ready", "starting V30 SUMO workers")
        except BaseException:
            self.close()
            raise

    def _wait_for(self, message_type, operation, task_id=None):
        received = set()
        deadline = time.monotonic() + self.timeout_s
        while len(received) < self.worker_count:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Timed out while {operation}")
            try:
                message = self.result_queue.get(timeout=min(1.0, remaining))
            except queue.Empty:
                dead = [p.name for p in self.processes if not p.is_alive()]
                if dead:
                    raise RuntimeError(f"V30 SUMO workers exited: {dead}")
                continue
            if task_id is not None and message.get("task_id") != task_id:
                continue
            if message.get("type") == "error":
                raise RuntimeError(f"V30 worker failed: {message}")
            if message.get("type") == message_type:
                received.add(int(message["worker_idx"]))

    def evaluate(self, env, decision_step, min_action_time, baseline_actions):
        validate_rollout_contract(self.config, min_action_time=min_action_time)
        expected_tls_ids = {inter.inter_id for inter in env.list_intersection}
        if set(baseline_actions) != expected_tls_ids:
            raise RuntimeError(
                "V30 requires one V25 baseline action for every active TLS: "
                f"expected={sorted(expected_tls_ids)}, "
                f"actual={sorted(baseline_actions)}"
            )
        invalid_actions = {
            inter.inter_id: baseline_actions[inter.inter_id]
            for inter in env.list_intersection
            if (int(baseline_actions[inter.inter_id]) < 0
                or int(baseline_actions[inter.inter_id])
                >= len(inter.control_phases))
        }
        if invalid_actions:
            raise RuntimeError(
                f"V30 received invalid V25 baseline actions: {invalid_actions}"
            )
        start_sim_time_s = float(env.get_current_time())
        unstable = {
            inter.inter_id: {
                "sumo_phase": int(
                    env.traci_conn.trafficlight.getPhase(inter.tls_id)),
                "python_phase": int(inter.current_phase_index),
                "is_in_transition": bool(inter.is_in_transition),
            }
            for inter in env.list_intersection
            if (inter.is_in_transition
                or int(env.traci_conn.trafficlight.getPhase(inter.tls_id))
                != int(inter.current_phase_index))
        }
        if unstable:
            raise RuntimeError(
                "V30 decision snapshot must start from stable aligned greens: "
                f"{unstable}"
            )
        start_state_fingerprint = env.capture_rollout_state_fingerprint()
        snapshot_path = os.path.join(
            self.snapshot_dir,
            f"decision_{int(decision_step):04d}_{int(start_sim_time_s)}.xml",
        )
        if env.snapshot(snapshot_path) is None:
            raise RuntimeError("V30 failed to save the main SUMO snapshot")
        if not os.path.isfile(snapshot_path) or os.path.getsize(snapshot_path) <= 0:
            raise RuntimeError(
                f"V30 main SUMO snapshot is missing or empty: {snapshot_path}"
            )
        if abs(float(env.get_current_time()) - start_sim_time_s) > 1e-6:
            raise RuntimeError("SUMO time changed while saving the V30 start state")
        post_save_fingerprint = env.capture_rollout_state_fingerprint()
        if post_save_fingerprint["sha256"] != start_state_fingerprint["sha256"]:
            raise RuntimeError(
                "V30 main state changed while saving the decision snapshot: "
                f"before={start_state_fingerprint}, after={post_save_fingerprint}"
            )
        task_id = f"{decision_step}:{time.time_ns()}"
        context = {
            "type": "prepare", "task_id": task_id,
            "snapshot_path": os.path.abspath(snapshot_path),
            "start_sim_time_s": start_sim_time_s,
            "start_state_fingerprint": start_state_fingerprint,
            "control_state": env.capture_rollout_control_state(),
            "baseline_actions": dict(baseline_actions),
            "min_action_time": float(min_action_time),
        }
        try:
            for control_queue in self.control_queues:
                control_queue.put(context)
            self._wait_for("prepared", "preparing V30 decision", task_id)

            expected = 0
            for inter in env.list_intersection:
                for candidate_idx in range(len(inter.control_phases)):
                    self.task_queue.put({
                        "type": "rollout", "task_id": task_id,
                        "target_tls": inter.inter_id,
                        "candidate_idx": candidate_idx,
                    })
                    expected += 1

            results = {}
            deadline = time.monotonic() + self.timeout_s
            while len(results) < expected:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"V30 rollout timed out at decision {decision_step}; "
                        f"completed={len(results)}/{expected}")
                try:
                    message = self.result_queue.get(timeout=min(1.0, remaining))
                except queue.Empty:
                    dead = [p.name for p in self.processes if not p.is_alive()]
                    if dead:
                        raise RuntimeError(f"V30 SUMO workers exited: {dead}")
                    continue
                if message.get("task_id") != task_id:
                    continue
                if message.get("type") == "error":
                    raise RuntimeError(
                        f"V30 {message.get('target_tls')} candidate "
                        f"{message.get('candidate_idx')} failed: "
                        f"{message.get('error')}")
                key = (message["target_tls"], int(message["candidate_idx"]))
                results[key] = message
            self._retain_recent_snapshot_groups(
                decision_step=decision_step,
                snapshot_paths=[
                    snapshot_path,
                    *[
                        result["end_snapshot_path"]
                        for result in results.values()
                    ],
                ],
            )
            return (
                results,
                os.path.abspath(snapshot_path),
                deepcopy(context["control_state"]),
                deepcopy(start_state_fingerprint),
            )
        finally:
            if not self.config.get("V30_KEEP_ROLLOUT_SNAPSHOTS", True):
                try:
                    os.remove(snapshot_path)
                except OSError:
                    pass

    def _retain_recent_snapshot_groups(self, decision_step, snapshot_paths):
        """Keep complete snapshot sets for the latest N decisions."""
        self._snapshot_history = retain_recent_snapshot_group(
            work_dir=self.work_dir,
            snapshot_history=self._snapshot_history,
            decision_step=decision_step,
            snapshot_paths=snapshot_paths,
            retention_decisions=self.snapshot_retention_decisions,
        )

    def close(self):
        for control_queue in self.control_queues:
            try:
                control_queue.put({"type": "stop"})
            except Exception:
                pass
        deadline = time.monotonic() + 30.0
        for process in self.processes:
            process.join(timeout=max(0.0, deadline - time.monotonic()))
        for process in self.processes:
            if process.is_alive():
                process.terminate()
        for process in self.processes:
            process.join(timeout=10)
        for queue_object in (
                self.task_queue, self.result_queue, *self.control_queues):
            try:
                queue_object.close()
                queue_object.cancel_join_thread()
            except Exception:
                pass
