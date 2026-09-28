"""Build same-step V35 Stage 2 perception from a live SUMO master.

The builder is deliberately independent of TraCI.  ``collect_observation_state``
is the only function that touches a local adapter/SUMO object; Ray actors call
that function inside the actor and return plain Python data.  A complete city
snapshot is then assembled from that one state payload, so local and neighbor
fields cannot accidentally come from different simulator times.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from numbers import Integral
from pathlib import Path
from typing import Any

from .online_rollout import CitySnapshot, IntersectionObservation


PHASE_MOVEMENTS: dict[str, tuple[str, str]] = {
    "ETWT": ("ET", "WT"),
    "NTST": ("NT", "ST"),
    "ELWL": ("EL", "WL"),
    "NLSL": ("NL", "SL"),
}
MOVEMENT_INDEX = {
    "WL": 0,
    "WT": 1,
    "WR": 2,
    "EL": 3,
    "ET": 4,
    "ER": 5,
    "NL": 6,
    "NT": 7,
    "NR": 8,
    "SL": 9,
    "ST": 10,
    "SR": 11,
}
SIDE_BY_ENTRY = {"N": "north", "S": "south", "E": "east", "W": "west"}
# Movement contract used by the cooperative prompt.  These are deliberately
# defined by the target-side direction labels in the prompt, rather than
# inferred from the source intersection's outbound route direction.
NEIGHBOR_MOVEMENTS = {
    "north": ("NT", "EL"),
    "south": ("ST", "WL"),
    "east": ("ET", "SL"),
    "west": ("WT", "NL"),
}


def _as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def _index_value(value: Any) -> int | None:
    """Normalize SUMO/numpy/string indices without accepting booleans."""
    if isinstance(value, bool):
        return None
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _indexed_mapping_value(mapping: Any, index: Any) -> tuple[bool, Any]:
    """Look up a phase mapping whose keys may be int or serialized strings."""
    if not isinstance(mapping, Mapping):
        return False, None
    normalized = _index_value(index)
    candidates = [index]
    if normalized is not None:
        candidates.extend((normalized, str(normalized)))
    for candidate in candidates:
        try:
            if candidate in mapping:
                return True, mapping[candidate]
        except TypeError:
            continue
    return False, None
def _movement_index(movement: str) -> int:
    try:
        return MOVEMENT_INDEX[movement]
    except KeyError as exc:
        raise ValueError(f"unsupported movement {movement!r}") from exc


def _phase_value(values: Any, phase: str, movement: str) -> int:
    if not isinstance(values, (list, tuple)):
        return 0
    index = _movement_index(movement)
    if index >= len(values):
        return 0
    return _as_int(values[index])


def _phase_total(values: Any, phase: str) -> int:
    return sum(_phase_value(values, phase, movement) for movement in PHASE_MOVEMENTS[phase])


def _history_delta(history: Any, index: int) -> int:
    """Return latest-minus-earliest for one movement slot."""
    if not isinstance(history, (list, tuple)) or len(history) < 2:
        return 0
    first = history[0]
    last = history[-1]
    try:
        return _as_int(last[index]) - _as_int(first[index])
    except (IndexError, TypeError):
        return 0


def _boundary_delta(history: Any, index: int) -> int:
    """Return V30-V15 used only for a network-boundary movement."""
    if not isinstance(history, (list, tuple)) or len(history) < 6:
        return 0
    try:
        return max(0, _as_int(history[-1][index]) - _as_int(history[2][index]))
    except (IndexError, TypeError):
        return 0


def _current_movement_counts(feature: Mapping[str, Any]) -> tuple[list[int], list[int]]:
    """Extract current V/Q in the canonical W,E,N,S movement slot order."""
    direct_v = feature.get("movement_v", feature.get("current_v"))
    direct_q = feature.get("movement_q", feature.get("current_q"))
    try:
        values_v = [_as_int(direct_v[index]) for index in range(12)]
    except (IndexError, KeyError, TypeError):
        movement_ids = feature.get("traffic_movement_vehicle_ids_150m") or []
        values_v = [len(movement_ids[index] or []) if index < len(movement_ids) else 0 for index in range(12)]

    # SUMOEnv's lane_num_waiting_vehicle_in is the authoritative queue vector.
    # Only use speed-derived stopped counts when a compatible queue vector is
    # unavailable (for example, in a small test double).
    authoritative_q = feature.get("lane_num_waiting_vehicle_in", direct_q)
    try:
        values_q = [_as_int(authoritative_q[index]) for index in range(12)]
    except (IndexError, KeyError, TypeError):
        movement_ids = feature.get("traffic_movement_vehicle_ids_150m") or []
        speeds = feature.get("vehicle_speed") or {}
        values_q = []
        for index in range(12):
            ids = movement_ids[index] if index < len(movement_ids) else []
            stopped = 0
            for vehicle_id in (ids or []):
                if isinstance(speeds, Mapping):
                    speed = speeds.get(vehicle_id, 1.0)
                else:
                    try:
                        speed = speeds[vehicle_id]
                    except (IndexError, KeyError, TypeError):
                        speed = 1.0
                try:
                    stopped += int(float(speed) < 0.1)
                except (TypeError, ValueError):
                    pass
            values_q.append(stopped)
    return values_v, values_q


def _phase_name(intersection: Any, signal_table: Mapping[str, str]) -> str:
    intersection_id = str(getattr(intersection, "inter_id", getattr(intersection, "tls_id", "")))
    direct = signal_table.get(intersection_id)
    if isinstance(direct, str) and direct in PHASE_MOVEMENTS:
        return direct
    feature = getattr(intersection, "dic_feature", {})
    phases = list(getattr(intersection, "control_phases", ()))
    # SUMOEnv exposes current_phase_index as the raw SUMO index and maps it to
    # a control action through phase_index_2_action.  Test doubles may expose
    # only cur_phase, which is already the control-phase index in SUMOEnv.
    raw_index = getattr(intersection, "current_phase_index", None)
    action_map = getattr(intersection, "phase_index_2_action", {})
    found, action_index = _indexed_mapping_value(action_map, raw_index)
    action_index = _index_value(action_index)
    if found and action_index is not None and 0 <= action_index < len(phases):
        if phases[action_index] in PHASE_MOVEMENTS:
            return str(phases[action_index])
    current = feature.get("cur_phase") if isinstance(feature, Mapping) else None
    if isinstance(current, (list, tuple)) and current:
        index = _index_value(current[0])
        if index is None:
            index = -1
        if 0 <= index < len(phases) and phases[index] in PHASE_MOVEMENTS:
            return str(phases[index])
    phase_name_map = getattr(intersection, "phase_index_2_phase_name", {})
    found, candidate = _indexed_mapping_value(phase_name_map, raw_index)
    if found and isinstance(candidate, str) and candidate in PHASE_MOVEMENTS:
        return candidate
    raise RuntimeError(
        f"cannot determine current phase for {intersection_id!r}; "
        "runtime phase index/mapping is missing or points to an invalid phase"
    )


def _age_values(
    intersection_id: str,
    current_phase: str,
    values_v: list[int],
    tracker: dict[str, dict[str, int]],
    step: int,
) -> dict[str, int]:
    previous = tracker.setdefault(intersection_id, {})
    last_step = previous.get("__step", -1)
    result: dict[str, int] = {}
    for phase in PHASE_MOVEMENTS:
        if phase == current_phase:
            result[phase] = 0
        elif _phase_total(values_v, phase) > 0:
            old = previous.get(phase, 0)
            result[phase] = old if last_step == step else old + 1
        else:
            result[phase] = 0
        previous[phase] = result[phase]
    previous["__step"] = int(step)
    return result


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        return value.item()
    except AttributeError:
        return str(value)


def collect_observation_state(source: Any, *, step: int | None = None) -> dict[str, Any]:
    """Collect serializable state from a local adapter or a SUMOEnv.

    This function is called inside a Ray actor for remote masters.  It also
    accepts an adapter directly, which is useful for local tests and debugging.
    """
    adapter = source
    env = getattr(source, "env", source)
    intersections = list(getattr(env, "list_intersection", ()))
    if not intersections:
        raise TypeError("observation source must expose env.list_intersection")
    try:
        current_time = float(getattr(env, "get_current_time")())
    except (AttributeError, TypeError):
        current_time = float(getattr(source, "current_time")())
    resolved_step = int(round(current_time / 30.0)) if step is None else int(step)
    signal_table: Mapping[str, str] = {}
    getter = getattr(adapter, "current_signal_table", None)
    if callable(getter):
        signal_table = getter()
    age_tracker = getattr(adapter, "_v35_age_tracker", None)
    if age_tracker is None:
        age_tracker = {}
        try:
            setattr(adapter, "_v35_age_tracker", age_tracker)
        except Exception:
            pass
    rows: list[dict[str, Any]] = []
    for intersection in intersections:
        intersection_id = str(getattr(intersection, "inter_id", getattr(intersection, "tls_id", "")))
        feature = getattr(intersection, "dic_feature", {})
        if not isinstance(feature, Mapping):
            feature = {}
        values_v, values_q = _current_movement_counts(feature)
        current_phase = _phase_name(intersection, signal_table)
        age = _age_values(intersection_id, current_phase, values_v, age_tracker, resolved_step)
        rows.append({
            "intersection_id": intersection_id,
            "current_phase": current_phase,
            "movement_v": values_v,
            "movement_q": values_q,
            "v_history": _json_safe(feature.get("v9_cycle_150m_history") or []),
            "q_history": _json_safe(feature.get("v26_cycle_queue_history") or []),
            "outbound_history": _json_safe(feature.get("v36_cycle_outbound_snapshots") or []),
            "age_by_phase": age,
        })
    return {"step": resolved_step, "current_time": current_time, "intersections": rows}


def _remote_or_local_state(source: Any, step: int | None) -> dict[str, Any]:
    if isinstance(source, Mapping):
        return dict(source)
    method = getattr(source, "observation_state", None)
    remote = getattr(method, "remote", None) if method is not None else None
    if callable(remote):
        try:
            import ray
        except ImportError as exc:  # pragma: no cover - Ray deployment only
            raise RuntimeError("Ray is required to read a remote master") from exc
        ref = remote(step=step) if step is not None else remote()
        return ray.get(ref)
    if callable(method):
        return method(step=step) if step is not None else method()
    return collect_observation_state(source, step=step)


def _route_table_path(city: str, routes_dir: str | Path | None) -> Path:
    city_key = str(city).lower()
    for candidate in ("jinan", "hangzhou", "newyork"):
        if city_key.startswith(candidate):
            city_key = candidate
            break
    root = Path(routes_dir) if routes_dir is not None else Path(__file__).resolve().parent / "artifacts"
    path = root / f"movement_routes_{city_key}.json"
    if not path.exists():
        raise FileNotFoundError(f"missing movement route table: {path}")
    return path


def _load_routes(city: str, routes_dir: str | Path | None) -> dict[str, Any]:
    path = _route_table_path(city, routes_dir)
    table = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(table, Mapping) or not isinstance(table.get("routes"), Mapping):
        raise ValueError(f"invalid movement route table: {path}")
    # Older generated artifacts omit link metrics. Enrich them from the source
    # topology so the online prompt retains distance and travel-time context.
    source_path = Path(str(table.get("source_topology", "")))
    candidates = [source_path]
    if not source_path.is_absolute():
        candidates.extend([Path.cwd() / source_path, Path(__file__).resolve().parents[4] / source_path])
    topology = None
    for candidate in candidates:
        if candidate.is_file():
            topology = json.loads(candidate.read_text(encoding="utf-8"))
            break
    if isinstance(topology, Mapping):
        for source_id, source_data in table["routes"].items():
            source_neighbors = ((topology.get("intersections") or {}).get(source_id) or {}).get("neighbors") or {}
            for movement, route in (source_data.get("movements") or {}).items():
                if not isinstance(route, dict) or route.get("receiver_id") is None:
                    continue
                edge = source_neighbors.get(route.get("exit_direction"), {})
                for key in ("distance_m", "speed_limit_mps", "travel_time_s"):
                    if key not in route and key in edge:
                        route[key] = edge[key]
    return dict(table)


def _state_rows(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = payload.get("intersections")
    if isinstance(rows, Mapping):
        result = {}
        for key, value in rows.items():
            if not isinstance(value, Mapping):
                raise ValueError(f"observation row {key!r} must be a mapping")
            row = dict(value)
            row.setdefault("intersection_id", str(key))
            row_id = str(row["intersection_id"])
            if row_id in result:
                raise ValueError(f"duplicate observation intersection ID: {row_id!r}")
            result[row_id] = row
        return result
    if isinstance(rows, (list, tuple)):
        result: dict[str, dict[str, Any]] = {}
        for index, value in enumerate(rows):
            if not isinstance(value, Mapping) or value.get("intersection_id") is None:
                raise ValueError(f"observation row {index} must contain intersection_id")
            row = dict(value)
            row_id = str(row["intersection_id"])
            if row_id in result:
                raise ValueError(f"duplicate observation intersection ID: {row_id!r}")
            result[row_id] = row
        return result
    raise ValueError("observation state must contain intersections")


def _movement_values(row: Mapping[str, Any], movement: str) -> tuple[int, int]:
    values_v = row.get("movement_v") or []
    values_q = row.get("movement_q") or []
    index = _movement_index(movement)
    return (
        _as_int(values_v[index]) if index < len(values_v) else 0,
        _as_int(values_q[index]) if index < len(values_q) else 0,
    )


def _coordination_outbound_measurement(
    row: Mapping[str, Any], movement: str, distance_m: float | None,
) -> tuple[int, dict[str, Any] | None]:
    history = row.get("outbound_history") or []
    if not isinstance(history, (list, tuple)):
        return 0, None
    frames = [item for item in history if isinstance(item, Mapping)]
    if not frames:
        return 0, None
    if distance_m is not None and distance_m > 0:
        from utils.coordination_frame_selector import select_coordination_frame
        selected, _ = select_coordination_frame(float(distance_m))
        frame_index = int(selected.frame_index)
        frame = next((item for item in frames if _as_int(item.get("frame_index")) == frame_index), None)
    else:
        frame = max(frames, key=lambda item: (_as_int(item.get("frame_index", 0)), float(item.get("sim_time_s", 0) or 0)))
    if frame is None:
        available = sorted(_as_int(item.get("frame_index")) for item in frames)
        raise RuntimeError(
            "selected coordination frame is missing from SUMO history: "
            f"movement={movement} distance_m={distance_m} "
            f"selected_frame_index={frame_index} available={available}"
        )
    by_movement = frame.get("outbound_vehicle_ids_by_movement")
    if not isinstance(by_movement, Mapping):
        return 0, None
    values = by_movement.get(movement) or []
    distances = frame.get("outbound_distance_from_source_m") or {}
    if isinstance(distances, Mapping) and distances:
        filtered = []
        for value in values:
            try:
                if float(distances.get(str(value), float("inf"))) <= 150.0:
                    filtered.append(value)
            except (TypeError, ValueError):
                continue
        values = filtered
    audit = {
        "frame_index": _as_int(frame.get("frame_index", 0)),
        "frame_time_s": float(frame.get("sim_time_s", 0.0) or 0.0),
        "source_movement": movement,
        "source_view_limit_m": 150.0,
    }
    return len(set(str(value) for value in values)), audit


def _coordination_outbound_count(
    row: Mapping[str, Any], movement: str, distance_m: float | None,
) -> int:
    return _coordination_outbound_measurement(row, movement, distance_m)[0]


def _target_coordination(
    target_id: str,
    movement: str,
    rows: Mapping[str, Mapping[str, Any]],
    routes: Mapping[str, Any],
) -> dict[str, Any]:
    entry = movement[0]
    upstream = []
    for source_id, source_data in routes.items():
        for source_movement, route in (source_data.get("movements") or {}).items():
            if not isinstance(route, Mapping):
                continue
            if route.get("receiver_id") == target_id and str(route.get("receiver_entry_direction", "")).upper() == entry:
                upstream.append((str(source_id), source_movement, route))
    if upstream:
        # The source movement is the movement observed at the upstream exit.
        # It is not necessarily the same label as the target entry movement.
        count = sum(
            _coordination_outbound_count(rows[source_id], source_movement,
                                          float(route.get("distance_m")) if route.get("distance_m") is not None else None)
            for source_id, source_movement, route in upstream
            if source_id in rows
        )
        route = upstream[0][2]
        return {"count": int(count), "is_boundary": "no"}
    # Boundary movement: V30-V15 from this target's own six-frame history.
    row = rows[target_id]
    index = _movement_index(movement)
    return {"count": _boundary_delta(row.get("v_history") or [], index), "is_boundary": "yes"}


def _target_coordination_audit(
    target_id: str,
    movement: str,
    rows: Mapping[str, Mapping[str, Any]],
    routes: Mapping[str, Any],
) -> list[dict[str, Any]]:
    entry = movement[0]
    selected: list[dict[str, Any]] = []
    for source_id, source_data in routes.items():
        for source_movement, route in (source_data.get("movements") or {}).items():
            if not isinstance(route, Mapping):
                continue
            if route.get("receiver_id") != target_id or str(route.get("receiver_entry_direction", "")).upper() != entry:
                continue
            if source_id not in rows:
                continue
            distance_m = float(route["distance_m"]) if route.get("distance_m") is not None else None
            _, audit = _coordination_outbound_measurement(rows[source_id], source_movement, distance_m)
            if audit is not None:
                selected.append({
                    **audit,
                    "source_intersection": str(source_id),
                    "target_intersection": target_id,
                    "target_movement": movement,
                    "route_distance_m": distance_m,
                })
    return selected


def _neighbor_context(
    target_id: str,
    rows: Mapping[str, Mapping[str, Any]],
    routes: Mapping[str, Any],
) -> tuple[dict[str, Any], set[str]]:
    neighbors: dict[str, Any] = {}
    expected: set[str] = set()
    candidates_by_side: dict[str, list[tuple[str, Mapping[str, Any]]]] = {}
    for source_id, source_data in routes.items():
        for route in (source_data.get("movements") or {}).values():
            if not isinstance(route, Mapping) or route.get("receiver_id") != target_id or route.get("is_boundary"):
                continue
            entry = str(route.get("receiver_entry_direction", "")).upper()
            side = SIDE_BY_ENTRY.get(entry)
            if side:
                candidates_by_side.setdefault(side, []).append((str(source_id), route))
    for side, candidates in candidates_by_side.items():
        source_id, route = sorted(candidates, key=lambda item: item[0])[0]
        expected.add(source_id)
        if source_id not in rows:
            continue
        upstream = {}
        for movement in NEIGHBOR_MOVEMENTS[side]:
            value_v, value_q = _movement_values(rows[source_id], movement)
            upstream[movement] = {"v": value_v, "q": value_q}
        neighbors[side] = {
            "source_intersection": source_id,
            "upstream_movements": upstream,
            "distance_m": route.get("distance_m"),
            "speed_limit_mps": route.get("speed_limit_mps"),
            "travel_time_s": route.get("travel_time_s"),
        }
    return neighbors, expected


def build_city_snapshot(
    source: Any,
    city: str,
    step: int | None = None,
    *,
    routes_dir: str | Path | None = None,
    simulator_snapshot: Any = None,
) -> CitySnapshot:
    """Build and validate one complete, same-step city snapshot."""
    payload = _remote_or_local_state(source, step)
    if not isinstance(payload, Mapping):
        raise TypeError("observation state must be a mapping")
    payload_step = payload.get("step", step)
    if payload_step is None:
        raise ValueError("observation state must contain a city-level step")
    resolved_step = _index_value(payload_step)
    if resolved_step is None or resolved_step < 0:
        raise ValueError(f"invalid city-level observation step: {payload_step!r}")
    rows = _state_rows(payload)
    sumo_time_s = float(payload.get("current_time")) if payload.get("current_time") is not None else None
    table = _load_routes(city, routes_dir)
    route_rows = table["routes"]
    row_ids = set(rows)
    route_ids = {str(value) for value in route_rows}
    if row_ids != route_ids:
        missing = sorted(route_ids - row_ids)
        unexpected = sorted(row_ids - route_ids)
        raise ValueError(
            "observation state does not match the route table: "
            f"missing={missing}, unexpected={unexpected}"
        )
    for source_id, source_data in route_rows.items():
        if not isinstance(source_data, Mapping):
            raise ValueError(f"route row {source_id!r} must be a mapping")
        movements = source_data.get("movements")
        if not isinstance(movements, Mapping):
            raise ValueError(f"route row {source_id!r} has no movements mapping")
        for movement, route in movements.items():
            if not isinstance(route, Mapping):
                raise ValueError(f"route {source_id!r}/{movement!r} must be a mapping")
            receiver_id = route.get("receiver_id")
            if receiver_id is not None and str(receiver_id) not in route_ids:
                raise ValueError(
                    f"route {source_id!r}/{movement!r} references missing "
                    f"receiver {receiver_id!r}"
                )
    observations: list[IntersectionObservation] = []
    required_neighbors: dict[str, set[str]] = {}
    for target_id in sorted(rows):
        row = rows[target_id]
        row_step = row.get("step", resolved_step)
        normalized_row_step = _index_value(row_step)
        if normalized_row_step != resolved_step:
            raise ValueError(
                f"observation row {target_id!r} has step {row_step!r}, "
                f"expected {resolved_step}"
            )
        phases: dict[str, Any] = {}
        for phase, movements in PHASE_MOVEMENTS.items():
            values_v = row.get("movement_v") or []
            values_q = row.get("movement_q") or []
            phase_v = [_as_int(values_v[_movement_index(movement)]) if _movement_index(movement) < len(values_v) else 0 for movement in movements]
            phase_q = [_as_int(values_q[_movement_index(movement)]) if _movement_index(movement) < len(values_q) else 0 for movement in movements]
            phases[phase] = {
                "v": phase_v,
                "q": phase_q,
                "dv": sum(_history_delta(row.get("v_history") or [], _movement_index(movement)) for movement in movements),
                "dq": sum(_history_delta(row.get("q_history") or [], _movement_index(movement)) for movement in movements),
                "age": _as_int((row.get("age_by_phase") or {}).get(phase, 0)),
            }
        current_phase = row.get("current_phase")
        if not isinstance(current_phase, str) or current_phase not in PHASE_MOVEMENTS:
            raise ValueError(
                f"observation row {target_id!r} has invalid current_phase: {current_phase!r}"
            )
        local = {"current_phase": current_phase, "phases": phases}
        coordination = {
            phase: {
                movement: _target_coordination(target_id, movement, rows, route_rows)
                for movement in movements
            }
            for phase, movements in PHASE_MOVEMENTS.items()
        }
        coordination_frames = {
            phase: {
                movement: _target_coordination_audit(target_id, movement, rows, route_rows)
                for movement in movements
            }
            for phase, movements in PHASE_MOVEMENTS.items()
        }
        neighbors, expected = _neighbor_context(target_id, rows, route_rows)
        required_neighbors[target_id] = expected
        observations.append(IntersectionObservation(
            intersection_id=target_id,
            step=resolved_step,
            local_perception=local,
            cooperative_perception={"local_coordination": coordination, "neighbors": neighbors},
            current_phase=current_phase,
            audit_metadata={
                "sumo_time_s": sumo_time_s,
                "observation_step": resolved_step,
                # Ray rollout actors restore SUMO's XML state, but this
                # controller-side history is not part of the XML snapshot.
                # Carry the SUMO truth so t1 age continues from t0.
                "sumo_age_by_phase": dict(row.get("age_by_phase") or {}),
                "coordination_frames": coordination_frames,
            },
        ))
    snapshot = CitySnapshot(
        city=str(city),
        step=resolved_step,
        observations=tuple(observations),
        simulator_snapshot=simulator_snapshot,
        required_neighbors=required_neighbors,
        sumo_time_s=sumo_time_s,
    )
    from .online_rollout import validate_city_snapshot

    validate_city_snapshot(snapshot)
    return snapshot


__all__ = ["PHASE_MOVEMENTS", "collect_observation_state", "build_city_snapshot"]
