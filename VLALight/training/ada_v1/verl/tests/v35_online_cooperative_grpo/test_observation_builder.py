from __future__ import annotations

import json
from pathlib import Path

import pytest

from v35_online_cooperative_grpo.observation_builder import build_city_snapshot


class _Intersection:
    def __init__(self, intersection_id: str) -> None:
        self.inter_id = intersection_id


class _Source:
    def __init__(self, rows: list[dict], signals: dict[str, str]) -> None:
        self.env = type("Env", (), {})()
        self.env.list_intersection = [_Intersection(row["intersection_id"]) for row in rows]
        self._rows = rows
        self._signals = signals

    def current_time(self) -> float:
        return 300.0

    def current_signal_table(self) -> dict[str, str]:
        return dict(self._signals)

    def observation_state(self, step: int | None = None) -> dict:
        return {"step": 10, "intersections": self._rows}


def _route_table(path: Path, ids: list[str], target_neighbors: dict[str, str]) -> None:
    routes = {}
    for intersection_id in ids:
        routes[intersection_id] = {
            "movements": {
                movement: {
                    "exit_direction": direction,
                    "receiver_id": None,
                    "receiver_entry_direction": None,
                    "is_boundary": True,
                }
                for movement, direction in {
                    "ET": "W", "WT": "E", "NT": "S", "ST": "N",
                    "NL": "E", "SL": "W", "EL": "S", "WL": "N",
                }.items()
            }
        }
    # Use one source movement per neighbor. The source label is deliberately
    # different from the target label to catch accidental same-name mapping.
    entries = {"north": ("N", "ST", "EL"), "south": ("S", "NT", "WL"),
               "east": ("E", "WT", "SL"), "west": ("W", "ET", "NL")}
    for side, source_id in target_neighbors.items():
        entry, source_movement, _ = entries[side]
        routes[source_id]["movements"][source_movement].update(
            receiver_id="target",
            receiver_entry_direction=entry,
            is_boundary=False,
            distance_m=100.0,
            speed_limit_mps=10.0,
            travel_time_s=10.0,
        )
    # A second movement from each source feeds the target too, so local
    # coordination has a measurable source-outgoing count.
    for side, source_id in target_neighbors.items():
        _, source_movement, _ = entries[side]
        routes[source_id]["movements"][source_movement]["receiver_id"] = "target"
    (path / "movement_routes_test.json").write_text(
        json.dumps({"routes": routes}), encoding="utf-8"
    )


def _row(intersection_id: str, *, step: int = 10, outbound_movement: str | None = None, incoming_values: dict[str, int] | None = None) -> dict:
    outbound = []
    if outbound_movement:
        outbound = [{
            "frame_index": 6,
            "sim_time_s": 30,
            "outbound_vehicle_ids_by_movement": {
                outbound_movement: ["veh-inside", "veh-outside"]
            },
            "outbound_distance_from_source_m": {
                "veh-inside": 149.9,
                "veh-outside": 150.1,
            },
        }]
    movement_v = [0] * 12
    movement_q = [0] * 12
    movement_index = {"ET": 4, "WT": 1, "NT": 7, "ST": 10, "EL": 3, "WL": 0, "NL": 6, "SL": 9}
    for movement, value in (incoming_values or {}).items():
        movement_v[movement_index[movement]] = value
        movement_q[movement_index[movement]] = value
    return {
        "intersection_id": intersection_id,
        "step": step,
        "current_phase": "NTST",
        "movement_v": movement_v,
        "movement_q": movement_q,
        "v_history": [],
        "q_history": [],
        "outbound_history": outbound,
        "age_by_phase": {},
    }


@pytest.mark.parametrize("neighbor_count", [2, 3, 4])
def test_build_snapshot_preserves_complete_topology_and_source_semantics(tmp_path: Path, neighbor_count: int):
    sides = ["north", "south", "east", "west"][:neighbor_count]
    neighbors = {side: f"n{index}" for index, side in enumerate(sides)}
    ids = ["target", *neighbors.values()]
    _route_table(tmp_path, ids, neighbors)
    rows = [_row("target")]
    rows.extend(_row(value, outbound_movement={"north": "ST", "south": "NT", "east": "WT", "west": "ET"}[side],
                     incoming_values={"NT": 5, "EL": 7} if side == "north" else {"ST": 2, "WL": 3} if side == "south" else {"ET": 4, "SL": 6} if side == "east" else {"WT": 8, "NL": 9})
                for side, value in neighbors.items())
    source = _Source(rows, {intersection_id: "NTST" for intersection_id in ids})

    snapshot = build_city_snapshot(source, "test", routes_dir=tmp_path)
    assert snapshot.step == 10
    target = next(item for item in snapshot.observations if item.intersection_id == "target")
    assert set(target.cooperative_perception["neighbors"]) == set(sides)
    assert target.cooperative_perception["local_coordination"]["NTST"]["NT"]["count"] == 1
    audit = target.audit_metadata["coordination_frames"]["NTST"]["NT"][0]
    assert audit["frame_index"] == 6
    assert audit["frame_time_s"] == 30.0
    assert audit["source_view_limit_m"] == 150.0
    assert target.cooperative_perception["neighbors"]["north"]["upstream_movements"] == {
        "NT": {"v": 5, "q": 5}, "EL": {"v": 7, "q": 7}
    }
    assert target.cooperative_perception["local_coordination"]["NTST"]["NT"]["count"] != target.cooperative_perception["neighbors"]["north"]["upstream_movements"]["NT"]["v"]


def test_build_snapshot_rejects_missing_rows_wrong_step_and_invalid_phase(tmp_path: Path):
    ids = ["target", "north"]
    _route_table(tmp_path, ids, {"north": "north"})
    signals = {intersection_id: "NTST" for intersection_id in ids}

    missing = _Source([_row("target")], {"target": "NTST"})
    with pytest.raises(ValueError, match="does not match the route table"):
        build_city_snapshot(missing, "test", routes_dir=tmp_path)

    wrong_step = _Source([_row("target"), _row("north", step=11)], signals)
    with pytest.raises(ValueError, match="expected 10"):
        build_city_snapshot(wrong_step, "test", routes_dir=tmp_path)

    invalid_phase = _Source([_row("target"), _row("north")], signals)
    invalid_phase._rows[0]["current_phase"] = "UNKNOWN"
    with pytest.raises(ValueError, match="invalid current_phase"):
        build_city_snapshot(invalid_phase, "test", routes_dir=tmp_path)


def test_build_snapshot_rejects_coordination_history_at_wrong_sample_times(tmp_path: Path):
    ids = ["target", "north"]
    _route_table(tmp_path, ids, {"north": "north"})
    rows = [_row("target"), _row("north", outbound_movement="ST")]
    rows[1]["outbound_history"][0]["frame_index"] = 3
    rows[1]["outbound_history"][0]["sim_time_s"] = 30
    source = _Source(rows, {intersection_id: "NTST" for intersection_id in ids})

    with pytest.raises(RuntimeError, match="selected_frame_index=6"):
        build_city_snapshot(source, "test", routes_dir=tmp_path)
