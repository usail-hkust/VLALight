"""Parallel one-cycle SUMO rollouts for the V28 oracle controller."""

import multiprocessing as mp
import os
import queue
import time
from copy import deepcopy

from .sumo_env import SUMOEnv


def _phase_vehicle_ids(intersection, phase_name):
    movement_sets = intersection.dic_feature.get(
        "traffic_movement_vehicle_ids_150m", [])
    vehicle_ids = set()
    for movement in intersection._parse_phase_movements(phase_name):
        slot_idx = intersection._MOVEMENT_TO_PRESSURE_IDX_MAP.get(movement)
        if slot_idx is not None and slot_idx < len(movement_sets):
            vehicle_ids.update(movement_sets[slot_idx] or [])
    return vehicle_ids


def _rollout_worker(worker_idx, config, path_config, work_dir, command_queue,
                    result_queue):
    worker_path = deepcopy(path_config)
    worker_path["PATH_TO_WORK_DIRECTORY"] = work_dir
    worker_config = deepcopy(config)
    worker_config["USE_GUI"] = False
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
    try:
        env.reset(use_gui=False, seed=worker_config.get("SEED"), verbose=False)
        result_queue.put({"type": "ready", "worker_idx": worker_idx})
        while True:
            command = command_queue.get()
            if command is None or command.get("type") == "stop":
                break

            task_id = command["task_id"]
            candidate_idx = int(command["candidate_idx"])
            try:
                rollout_started = time.monotonic()
                env.load_from_file(
                    command["snapshot_path"], quiet=True, raise_on_error=True)
                env.restore_rollout_control_state(command["control_state"])

                action_dict = {}
                candidate_phases = {}
                discharged = {}
                for inter in env.list_intersection:
                    phase_count = len(inter.control_phases)
                    action_idx = candidate_idx % phase_count if phase_count else 0
                    action_dict[inter.inter_id] = action_idx
                    phase = inter.control_phases[action_idx]
                    candidate_phases[inter.inter_id] = phase
                    discharged[inter.inter_id] = set()

                transition_applied = {}
                for inter in env.list_intersection:
                    action_idx = action_dict[inter.inter_id]
                    target_sumo_idx = inter.action_2_phase_index.get(
                        action_idx, inter.current_phase_index)
                    transition_applied[inter.inter_id] = (
                        target_sumo_idx != inter.current_phase_index)

                counting_line = float(worker_config.get(
                    "DISCHARGE_COUNTING_LINE_DISTANCE", 50.0))

                def collect_discharged(inner_i, env):
                    for inter in env.list_intersection:
                        phase = candidate_phases[inter.inter_id]
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
                                discharged[inter.inter_id].add(vehicle_id)

                env.dic_traffic_env_conf["_SUPPRESS_SIGNAL_LOG"] = True
                env.step(
                    action_dict,
                    min_action_time=float(command["min_action_time"]),
                    inner_step_callback=collect_discharged,
                )
                env.dic_traffic_env_conf["_SUPPRESS_SIGNAL_LOG"] = False

                intersections = {}
                for inter in env.list_intersection:
                    phase = candidate_phases[inter.inter_id]
                    remaining_ids = _phase_vehicle_ids(inter, phase)
                    queue_count = sum(
                        inter.dic_vehicle_speed_current_step.get(vehicle_id, 1.0) < 0.1
                        for vehicle_id in remaining_ids
                    )
                    intersections[inter.inter_id] = {
                        "phase": phase,
                        "action_idx": action_dict[inter.inter_id],
                        "transition_applied": transition_applied[inter.inter_id],
                        "transition_time_s": (
                            float(worker_config.get("YELLOW_TIME", 0))
                            if transition_applied[inter.inter_id] else 0.0
                        ),
                        "effective_green_time_s": (
                            float(command["min_action_time"])
                            - float(worker_config.get("YELLOW_TIME", 0))
                            if transition_applied[inter.inter_id]
                            else float(command["min_action_time"])
                        ),
                        "discharged_30s": len(discharged[inter.inter_id]),
                        "remaining_v_30s": len(remaining_ids),
                        "remaining_queue_30s": queue_count,
                    }

                result_queue.put({
                    "type": "result",
                    "worker_idx": worker_idx,
                    "task_id": task_id,
                    "candidate_idx": candidate_idx,
                    "elapsed_s": time.monotonic() - rollout_started,
                    "intersections": intersections,
                })
            except BaseException as exc:
                env.dic_traffic_env_conf["_SUPPRESS_SIGNAL_LOG"] = False
                result_queue.put({
                    "type": "error",
                    "worker_idx": worker_idx,
                    "task_id": task_id,
                    "candidate_idx": candidate_idx,
                    "error": repr(exc),
                })
    finally:
        env.close()


class V28RolloutPool:
    """Keep four SUMO workers alive and reload one shared state per decision."""

    def __init__(self, config, path_config, work_dir, worker_count=4,
                 timeout_s=180.0):
        self.worker_count = int(worker_count)
        self.timeout_s = float(timeout_s)
        self.work_dir = work_dir
        self.snapshot_dir = os.path.join(work_dir, "snapshots")
        os.makedirs(self.snapshot_dir, exist_ok=True)
        ctx = mp.get_context("spawn")
        self.result_queue = ctx.Queue()
        self.command_queues = []
        self.processes = []

        for worker_idx in range(self.worker_count):
            command_queue = ctx.Queue()
            worker_dir = os.path.join(work_dir, f"worker_{worker_idx}")
            process = ctx.Process(
                target=_rollout_worker,
                args=(worker_idx, config, path_config, worker_dir,
                      command_queue, self.result_queue),
                name=f"v28-sumo-{worker_idx}",
            )
            process.daemon = True
            process.start()
            self.command_queues.append(command_queue)
            self.processes.append(process)
        try:
            self._wait_ready()
        except BaseException:
            self.close()
            raise

    def _wait_ready(self):
        ready = set()
        deadline = time.monotonic() + self.timeout_s
        while len(ready) < self.worker_count:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Timed out while starting V28 SUMO workers")
            try:
                message = self.result_queue.get(timeout=min(1.0, remaining))
            except queue.Empty:
                dead = [p.name for p in self.processes if not p.is_alive()]
                if dead:
                    raise RuntimeError(f"V28 SUMO workers exited during startup: {dead}")
                continue
            if message.get("type") == "ready":
                ready.add(message["worker_idx"])
            elif message.get("type") == "error":
                raise RuntimeError(f"V28 worker startup failed: {message}")

    def evaluate(self, env, decision_step, min_action_time):
        snapshot_path = os.path.join(
            self.snapshot_dir,
            f"decision_{int(decision_step):04d}_{int(env.get_current_time())}.xml",
        )
        if env.snapshot(snapshot_path) is None:
            raise RuntimeError("V28 failed to save the main SUMO snapshot")
        control_state = env.capture_rollout_control_state()
        task_id = f"{decision_step}:{time.time_ns()}"

        try:
            for candidate_idx, command_queue in enumerate(self.command_queues):
                command_queue.put({
                    "type": "rollout",
                    "task_id": task_id,
                    "candidate_idx": candidate_idx,
                    "snapshot_path": os.path.abspath(snapshot_path),
                    "control_state": control_state,
                    "min_action_time": float(min_action_time),
                })

            results = {}
            deadline = time.monotonic() + self.timeout_s
            while len(results) < self.worker_count:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    pending = sorted(set(range(self.worker_count)) - set(results))
                    workers = {
                        idx: {
                            "alive": process.is_alive(),
                            "exitcode": process.exitcode,
                        }
                        for idx, process in enumerate(self.processes)
                    }
                    raise TimeoutError(
                        f"V28 rollout timed out at decision {decision_step}; "
                        f"pending_candidates={pending}, workers={workers}")
                try:
                    message = self.result_queue.get(timeout=min(1.0, remaining))
                except queue.Empty:
                    dead = [p.name for p in self.processes if not p.is_alive()]
                    if dead:
                        raise RuntimeError(f"V28 SUMO workers exited: {dead}")
                    continue
                if message.get("task_id") != task_id:
                    continue
                if message.get("type") == "error":
                    raise RuntimeError(
                        f"V28 candidate {message.get('candidate_idx')} failed: "
                        f"{message.get('error')}")
                results[int(message["candidate_idx"])] = message
            return [results[idx] for idx in range(self.worker_count)]
        finally:
            try:
                os.remove(snapshot_path)
            except OSError:
                pass

    def close(self):
        for command_queue in self.command_queues:
            try:
                command_queue.put({"type": "stop"})
            except Exception:
                pass
        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        for path in (self.snapshot_dir,):
            try:
                if os.path.isdir(path) and not os.listdir(path):
                    os.rmdir(path)
            except OSError:
                pass
