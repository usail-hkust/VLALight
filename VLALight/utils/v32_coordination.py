"""Shared topology-aware arrival-demand aggregation for pseudo-video agents."""

from typing import Any, Dict


MOVEMENT_MAP = {
    "WL": 0, "WT": 1, "WR": 2,
    "EL": 3, "ET": 4, "ER": 5,
    "NL": 6, "NT": 7, "NR": 8,
    "SL": 9, "ST": 10, "SR": 11,
}


def merge_missing_boundary_coordination(
    coordination: Dict[str, Dict[str, Any]],
    legacy: Dict[str, Dict[str, Any]],
    phases: list[str],
    missing_directions,
) -> Dict[str, Dict[str, Any]]:
    """Add the legacy network-boundary estimate only for missing upstreams."""
    missing = {str(direction).upper() for direction in missing_directions}
    for phase in phases:
        phase_coordination = coordination.setdefault(phase, {
            "arrival_15s": 0, "internal": {}, "boundary": {}, "sources": {},
        })
        legacy_boundary = {
            movement: int(count)
            for movement, count in
            (legacy.get(phase, {}).get("boundary") or {}).items()
            if movement[:1] in missing
        }
        phase_coordination["boundary"] = legacy_boundary
        phase_coordination["arrival_15s"] = int(
            phase_coordination.get("arrival_15s", 0)) + sum(
                legacy_boundary.values())
        for movement in legacy_boundary:
            phase_coordination.setdefault("sources", {})[
                movement] = "network_boundary"
    coordination.setdefault("__meta__", {})[
        "boundary_directions"] = sorted(missing)
    coordination_meta = coordination["__meta__"]
    legacy_meta = dict(legacy.get("__meta__", {}) or {})
    has_usable_boundary = bool(
        missing and legacy_meta.get("boundary_estimate_available", False))
    coordination_meta["has_usable_boundary_coordination"] = (
        has_usable_boundary)
    coordination_meta["has_usable_coordination"] = bool(
        coordination_meta.get("has_usable_internal_coordination", False)
        or has_usable_boundary)
    return coordination


def build_v32_coordination_info(
    target_inter: Any,
    network_topology: Dict[str, Any],
    step_num: int,
    camera_distance: float = 150.0,
    decision_time: float = 30.0,
    yellow_time: float = 5.0,
) -> Dict[str, Dict[str, Any]]:
    """Match V32's final-frame, phase-aggregated coordination feature."""
    phases = list(target_inter.control_phases)
    target_tls = target_inter.inter_id
    result = {
        phase: {
            "arrival_15s": 0,
            "internal": {},
            "boundary": {},
            "sources": {},
        }
        for phase in phases
    }
    intersections = (network_topology or {}).get("intersections", {})
    upstream_directions = set()
    upstream_sources = {}
    for source_tls, source_cfg in intersections.items():
        for edge in (source_cfg.get("neighbors") or {}).values():
            if edge.get("neighbor_id") != target_tls:
                continue
            direction = str(edge.get("their_entry_direction") or "").upper()
            if direction in "EWNS":
                upstream_directions.add(direction)
                upstream_sources.setdefault(direction, str(source_tls))

    feature = target_inter.dic_feature or {}
    snapshots = sorted(
        feature.get("v32_cycle_vehicle_snapshots") or [],
        key=lambda item: float(item.get("time_s", 0.0)),
    )
    frame = snapshots[-1] if snapshots else {}
    frame_time = float(frame.get("time_s", decision_time))
    movement_ids = frame.get("movement_vehicle_ids") or []
    distances = frame.get("vehicle_distance") or {}
    speeds = frame.get("vehicle_speed") or {}

    active_phase = None
    cur_phase = feature.get("cur_phase") or []
    if cur_phase:
        try:
            active_idx = int(cur_phase[0])
            if 0 <= active_idx < len(phases):
                active_phase = phases[active_idx]
        except (TypeError, ValueError):
            pass

    eta_by_movement = {movement: [] for movement in MOVEMENT_MAP}
    for movement, movement_idx in MOVEMENT_MAP.items():
        if movement[0] not in upstream_directions or movement_idx >= len(movement_ids):
            continue
        for vehicle_id in movement_ids[movement_idx] or []:
            try:
                distance = float(distances.get(vehicle_id))
                speed = float(speeds.get(vehicle_id))
            except (TypeError, ValueError):
                continue
            if distance <= camera_distance or speed <= 0.1:
                continue
            eta_at_decision = (
                (distance - camera_distance) / speed
                - (decision_time - frame_time)
            )
            if eta_at_decision >= 0.0:
                eta_by_movement[movement].append(eta_at_decision)

    cycle_history = list(feature.get("v9_cycle_150m_history") or [])
    v15 = cycle_history[2] if len(cycle_history) >= 3 else []
    v30 = cycle_history[5] if len(cycle_history) >= 6 else (
        cycle_history[-1] if cycle_history else []
    )
    boundary_estimate_available = bool(
        len(cycle_history) >= 6 and len(v15) >= len(MOVEMENT_MAP)
        and len(v30) >= len(MOVEMENT_MAP))
    for phase in phases:
        horizon_s = 15.0 if phase == active_phase else 15.0 + yellow_time
        result[phase]["frame_time_s"] = frame_time
        result[phase]["horizon_s"] = horizon_s
        for movement in [phase[index:index + 2] for index in range(0, len(phase), 2)]:
            direction = movement[0]
            if direction in upstream_directions:
                value = sum(
                    eta <= horizon_s for eta in eta_by_movement.get(movement, [])
                )
                result[phase]["internal"][movement] = value
                result[phase]["sources"][movement] = upstream_sources[direction]
            else:
                movement_idx = MOVEMENT_MAP.get(movement)
                value = 0
                if (movement_idx is not None and movement_idx < len(v15)
                        and movement_idx < len(v30)):
                    value = max(0, int(v30[movement_idx]) - int(v15[movement_idx]))
                result[phase]["boundary"][movement] = value
                result[phase]["sources"][movement] = "network_boundary"
            result[phase]["arrival_15s"] += int(value)
    result["__meta__"] = {
        "step": int(step_num),
        "frame_time_s": frame_time,
        "active_phase": active_phase,
        "upstream_directions": sorted(upstream_directions),
        "boundary_directions": sorted(set("EWNS") - upstream_directions),
        "boundary_estimate_available": boundary_estimate_available,
    }
    return result
