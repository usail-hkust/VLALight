"""Pure validation helpers for the V30 counterfactual data contract."""

import os


STANDARD_PHASES = ("ETWT", "NTST", "ELWL", "NLSL")
VIDEO_FRAME_OFFSETS_S = (5, 10, 15, 20, 25, 30)
ROLLOUT_HORIZON_S = 30.0
TRANSITION_TIME_S = 5.0
SIMULATION_INTERVAL_S = 1.0
ENDPOINT_DISTANCE_M = 150.0
DEFAULT_ROLLOUT_WORKERS = 32


def retain_recent_snapshot_group(
        work_dir, snapshot_history, decision_step, snapshot_paths,
        retention_decisions=5):
    """Retain complete snapshot groups for the latest decisions only."""
    work_dir = os.path.abspath(work_dir)
    retention_decisions = max(0, int(retention_decisions))
    normalized_paths = {
        os.path.abspath(path)
        for path in snapshot_paths
        if path
    }
    snapshot_history.append((int(decision_step), normalized_paths))
    while len(snapshot_history) > retention_decisions:
        _, stale_paths = snapshot_history.pop(0)
        for path in stale_paths:
            try:
                if os.path.commonpath([work_dir, path]) != work_dir:
                    continue
                os.remove(path)
            except FileNotFoundError:
                pass
            except OSError:
                # Snapshot cleanup must not interrupt a completed rollout.
                pass
    return snapshot_history


def validate_standard_phase_order(intersections):
    invalid = {
        inter.inter_id: list(inter.control_phases)
        for inter in intersections
        if tuple(inter.control_phases) != STANDARD_PHASES
    }
    if invalid:
        raise RuntimeError(
            "V30 requires the exact standard four-phase order "
            f"{STANDARD_PHASES}; invalid mappings: {invalid}"
        )


def resolve_rollout_worker_count(requested=None, available_cpus=None):
    """Return a positive V30 worker count within the scheduler CPU quota."""
    requested_count = (
        DEFAULT_ROLLOUT_WORKERS if requested is None else int(requested)
    )
    if requested_count <= 0:
        raise ValueError("V30 rollout worker count must be positive")
    if available_cpus is None:
        try:
            available_cpus = len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            available_cpus = os.cpu_count() or 1
    available_count = max(1, int(available_cpus))
    return min(requested_count, available_count)


def validate_rollout_contract(config, min_action_time=None):
    """Reject settings that would shift V30 candidate endpoint semantics."""
    horizon = float(
        config.get("MIN_ACTION_TIME", ROLLOUT_HORIZON_S)
        if min_action_time is None else min_action_time
    )
    interval = float(config.get("INTERVAL", SIMULATION_INTERVAL_S))
    yellow = float(config.get("YELLOW_TIME", TRANSITION_TIME_S))
    all_red = float(config.get("ALL_RED_TIME", 0.0) or 0.0)
    distance = float(config.get("CAMERA_VIEW_DISTANCE", ENDPOINT_DISTANCE_M))
    skip_transition = bool(config.get("SKIP_TRANSITION_PHASE", False))

    mismatches = {}
    expected = {
        "MIN_ACTION_TIME": ROLLOUT_HORIZON_S,
        "INTERVAL": SIMULATION_INTERVAL_S,
        "YELLOW_TIME": TRANSITION_TIME_S,
        "ALL_RED_TIME": 0.0,
        "CAMERA_VIEW_DISTANCE": ENDPOINT_DISTANCE_M,
        "SKIP_TRANSITION_PHASE": False,
    }
    actual = {
        "MIN_ACTION_TIME": horizon,
        "INTERVAL": interval,
        "YELLOW_TIME": yellow,
        "ALL_RED_TIME": all_red,
        "CAMERA_VIEW_DISTANCE": distance,
        "SKIP_TRANSITION_PHASE": skip_transition,
    }
    for key, expected_value in expected.items():
        if actual[key] != expected_value:
            mismatches[key] = {
                "actual": actual[key],
                "expected": expected_value,
            }
    if mismatches:
        raise RuntimeError(
            "V30 rollout timing/distance contract mismatch: "
            f"{mismatches}"
        )
    return actual


def validate_restored_state_fingerprint(expected, actual):
    """Validate identity/control equality across SUMO saveState/loadState.

    SUMO may reconstruct vehicle distance and speed with slightly different
    floating-point values after loading XML. The full numeric hash therefore
    remains audit-only, but identities, control state and every set used by
    V30's 150 m endpoint semantics must remain exact.
    """
    required_equal = (
        "sim_time_s",
        "vehicle_count",
        "signal_count",
        "vehicle_id_lanes_sha256",
        "signals_sha256",
        "movement_vehicles_150m_sha256",
        "movement_queues_150m_sha256",
    )
    mismatches = {
        key: {"expected": expected.get(key), "actual": actual.get(key)}
        for key in required_equal
        if (key not in expected or key not in actual
            or expected.get(key) != actual.get(key))
    }
    if mismatches:
        raise RuntimeError(
            "V30 restored state identity/control/150m semantic mismatch: "
            f"{mismatches}; expected={expected}; actual={actual}"
        )
    return {
        "identity_control_match": True,
        "numeric_vehicle_hash_match": (
            expected.get("vehicles_sha256") == actual.get("vehicles_sha256")
        ),
        "queue_threshold_hash_match": (
            expected.get("movement_queues_150m_sha256")
            == actual.get("movement_queues_150m_sha256")
        ),
        "movement_150m_hash_match": (
            expected.get("movement_vehicles_150m_sha256")
            == actual.get("movement_vehicles_150m_sha256")
        ),
        "expected_vehicles_sha256": expected.get("vehicles_sha256"),
        "restored_vehicles_sha256": actual.get("vehicles_sha256"),
        "expected_queues_150m_sha256": expected.get(
            "movement_queues_150m_sha256"),
        "restored_queues_150m_sha256": actual.get(
            "movement_queues_150m_sha256"),
        "expected_movement_vehicles_150m_sha256": expected.get(
            "movement_vehicles_150m_sha256"),
        "restored_movement_vehicles_150m_sha256": actual.get(
            "movement_vehicles_150m_sha256"),
    }


def validate_candidate_tick_trace(
        start_time_s, sample_times_s, sampled_sumo_phases,
        sampled_transition_flags, sampled_controller_phases,
        start_sumo_phase, target_sumo_phase, transition_applied):
    """Validate the completed ticks for one 30-second candidate.

    SUMO may expose the next phase exactly on the t+5 boundary even though the
    five elapsed intervals were transitional. Python controller state is the
    unambiguous source for interval timing; SUMO phase samples independently
    confirm that the target green is active for every completed tick after it.
    """
    expected_times = [
        float(start_time_s) + offset
        for offset in range(1, int(ROLLOUT_HORIZON_S) + 1)
    ]
    actual_times = [float(value) for value in sample_times_s]
    actual_phases = [int(value) for value in sampled_sumo_phases]
    transition_flags = [bool(value) for value in sampled_transition_flags]
    controller_phases = [int(value) for value in sampled_controller_phases]
    if len(actual_times) != len(expected_times) or any(
            abs(actual - expected) > 1e-6
            for actual, expected in zip(actual_times, expected_times)):
        raise RuntimeError(
            "V30 candidate tick times are not exactly t+1...t+30: "
            f"actual={actual_times}, expected={expected_times}"
        )
    trace_lengths = {
        "sumo_phases": len(actual_phases),
        "transition_flags": len(transition_flags),
        "controller_phases": len(controller_phases),
    }
    if any(length != len(expected_times) for length in trace_lengths.values()):
        raise RuntimeError(
            "V30 candidate phase trace length mismatch: "
            f"actual={trace_lengths}, expected={len(expected_times)}"
        )

    transition_ticks = int(TRANSITION_TIME_S) if transition_applied else 0
    target = int(target_sumo_phase)
    start = int(start_sumo_phase)
    expected_transition_flags = (
        [True] * transition_ticks
        + [False] * int(ROLLOUT_HORIZON_S - transition_ticks)
        if transition_applied
        else [False] * int(ROLLOUT_HORIZON_S)
    )
    expected_controller_phases = (
        [start] * transition_ticks
        + [target] * int(ROLLOUT_HORIZON_S - transition_ticks)
        if transition_applied
        else [target] * int(ROLLOUT_HORIZON_S)
    )
    if (transition_flags != expected_transition_flags
            or controller_phases != expected_controller_phases):
        raise RuntimeError(
            "V30 candidate did not realize the required transition/green "
            "timing: "
            f"transition_applied={bool(transition_applied)}, "
            f"transition_flags={transition_flags}, "
            f"controller_phases={controller_phases}"
        )

    first_stable_green_index = transition_ticks
    if any(
            phase != target
            for phase in actual_phases[first_stable_green_index:]
    ):
        raise RuntimeError(
            "V30 SUMO phase was not the target green after the transition: "
            f"target_phase={target}, actual_phases={actual_phases}"
        )
    # At the exact t+5 boundary SUMO may already report the next phase. Earlier
    # samples must still be non-target to rule out a shortened transition.
    if transition_applied and any(
            phase == target for phase in actual_phases[:transition_ticks - 1]
    ):
        raise RuntimeError(
            "V30 SUMO entered the target green before five transition seconds: "
            f"target_phase={target}, actual_phases={actual_phases}"
        )
    return {
        "sample_count": len(actual_times),
        "first_sample_time_s": actual_times[0],
        "last_sample_time_s": actual_times[-1],
        "target_green_ticks": int(ROLLOUT_HORIZON_S) - transition_ticks,
        "transition_ticks": transition_ticks,
        "first_target_green_offset_s": transition_ticks + 1,
        "end_sumo_phase_index": actual_phases[-1],
    }


def resolve_recorded_seed(decisions, requested_seed=None):
    recorded_seeds = {int(record["seed"]) for record in decisions.values()}
    if len(recorded_seeds) != 1:
        raise ValueError(f"V30 log contains inconsistent seeds: {recorded_seeds}")
    recorded_seed = next(iter(recorded_seeds))
    if requested_seed is not None and int(requested_seed) != recorded_seed:
        raise ValueError(
            f"--seed={requested_seed} does not match recorded V30 seed={recorded_seed}"
        )
    return recorded_seed


def validate_complete_video_record(record):
    """Return False for discarded clips; validate canonical complete clips."""
    if record.get("status") != "complete":
        return False
    actual_times = [
        float(value) for value in record.get("video_frame_sim_times", [])
    ]
    expected_times = [
        float(record["sim_start_s"]) + offset
        for offset in VIDEO_FRAME_OFFSETS_S
    ]
    if len(actual_times) != len(expected_times) or any(
            abs(actual - expected) > 1e-6
            for actual, expected in zip(actual_times, expected_times)):
        raise RuntimeError(
            "Complete V30 video has noncanonical frame times: "
            f"step={record.get('decision_step')}, tls={record.get('tls_id')}, "
            f"actual={actual_times}, expected={expected_times}"
        )
    return True
