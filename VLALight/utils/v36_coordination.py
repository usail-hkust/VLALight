"""Topology-aware upstream outbound coordination for the V36 debug agent."""

from collections import defaultdict
from copy import deepcopy
import json
import os

from .coordination_frame_selector import build_topology_coordination_plan


_COORDINATION_EXCLUDED_MODELS = frozenset({"V30", "V30REPLAY"})


def is_new_coordination_enabled(config):
    """Return whether the shared V36 coordination pipeline is active.

    V30 is a counterfactual-only experiment.  It must remain independent of
    coordination even if a shared runner accidentally carries the opt-in
    configuration flag.
    """
    config = config or {}
    model_name = str(config.get("MODEL_NAME", "")).strip().upper()
    if model_name in _COORDINATION_EXCLUDED_MODELS:
        return False
    return model_name in {"V32", "V36"} or bool(
        config.get("ENABLE_NEW_COORDINATION", False))


class V36CoordinationManager:
    def __init__(self, topology, work_dir, *, speed_mps=11.0,
                 view_distance_m=150.0,
                 log_filename="v36_coordination_debug.jsonl"):
        self.intersections = (topology or {}).get("intersections", {})
        self.speed_mps = float(speed_mps)
        self.view_distance_m = float(view_distance_m)
        self.pending = defaultdict(list)
        self.log_path = os.path.join(work_dir, log_filename)
        os.makedirs(work_dir, exist_ok=True)

    @staticmethod
    def _movements_for_entry(phases, entry_direction):
        """Return controlled left/straight movements for an entry direction."""
        movements = []
        for phase in phases:
            for movement in [phase[i:i + 2] for i in range(0, len(phase), 2)]:
                if (movement.startswith(entry_direction)
                        and movement[1:] in ("L", "T")
                        and movement not in movements):
                    movements.append(movement)
        return movements

    @staticmethod
    def _phases_for_movement(phases, movement):
        return [
            phase for phase in phases
            if movement in [phase[i:i + 2] for i in range(0, len(phase), 2)]
        ]

    def _upstream_links(self, target_tls):
        return build_topology_coordination_plan(
            {"intersections": self.intersections},
            target_tls,
            speed_mps=self.speed_mps,
            view_distance_m=self.view_distance_m,
        )

    @staticmethod
    def _frame_movement_mapping_available(frame, source_exit_direction):
        by_direction = frame.get(
            "outbound_movement_mapping_available_by_direction")
        if isinstance(by_direction, dict):
            return bool(by_direction.get(source_exit_direction, False))
        if "outbound_lane_mapping_available" in frame:
            return bool(frame.get("outbound_lane_mapping_available"))
        return isinstance(frame.get("outbound_vehicle_ids_by_movement"), dict)

    def build(self, target_tls, target_phases, snapshots_by_tls, decision_step):
        result = {
            phase: {"arrival_15s": 0, "internal": {}, "boundary": {}, "sources": {}}
            for phase in target_phases
        }
        consumed = self.pending.pop((target_tls, decision_step), [])
        generated = []
        injected_current_count = 0
        missing_movement_mapping_count = 0

        for event in consumed:
            self._apply_event(result, target_phases, event)
            event["status"] = "consumed"
            self._write_log(event)

        for link in self._upstream_links(target_tls):
            source_frames = snapshots_by_tls.get(link["source_tls"], []) or []
            frame = next((item for item in source_frames
                          if int(item.get("frame_index", -1)) == link["frame_index"]), None)
            if frame is None:
                event = {**link, "target_tls": target_tls,
                         "generated_at_step": decision_step, "status": "missing_frame"}
                self._write_log(event)
                continue
            movement_snapshot = frame.get(
                "outbound_vehicle_ids_by_movement")
            if (not isinstance(movement_snapshot, dict)
                    or not self._frame_movement_mapping_available(
                        frame, link["source_exit_direction"])):
                direction_vehicle_ids = list(
                    (frame.get("outbound_vehicle_ids") or {}).get(
                        link["source_exit_direction"], []))
                event = {
                    **link,
                    "target_tls": target_tls,
                    "generated_at_step": decision_step,
                    "source_snapshot_sim_time_s": frame.get("sim_time_s"),
                    "outbound_vehicle_ids": direction_vehicle_ids,
                    "outbound_vehicle_count": len(direction_vehicle_ids),
                    "status": "missing_movement_mapping",
                }
                missing_movement_mapping_count += 1
                self._write_log(event)
                continue
            movement_specs = [
                (movement, list(movement_snapshot.get(movement, [])))
                for movement in self._movements_for_entry(
                    target_phases, link["target_entry_direction"])
            ]

            for target_movement, vehicle_ids in movement_specs:
                injection_step = decision_step + link["injection_delay_cycles"]
                event = {
                    **link,
                    "target_tls": target_tls,
                    "target_movement": target_movement,
                    "source_cycle": decision_step - 1,
                    "generated_at_step": decision_step,
                    "injected_at_step": injection_step,
                    "outbound_vehicle_ids": vehicle_ids,
                    "outbound_vehicle_count": len(vehicle_ids),
                    "source_snapshot_sim_time_s": frame.get("sim_time_s"),
                    "mapped_candidate_phases": self._phases_for_movement(
                        target_phases, target_movement),
                    "status": "generated",
                }
                generated.append(event)
                if link["injection_delay_cycles"] == 0:
                    self._apply_event(result, target_phases, event)
                    injected_current_count += 1
                    event["status"] = "injected_current"
                    self._write_log(event)
                else:
                    self.pending[(target_tls, injection_step)].append(
                        deepcopy(event))
                    event["status"] = "pending"
                    self._write_log(event)

        result["__meta__"] = {
            "decision_step": decision_step,
            "consumed_events": len(consumed),
            "generated_events": len(generated),
            "pending_events": sum(len(items) for items in self.pending.values()),
            "arrival_window_s": [-5.0, 15.0],
            "upstream_directions": sorted({
                link["target_entry_direction"]
                for link in self._upstream_links(target_tls)
            }),
            "boundary_directions": [],
            "has_internal_upstream": bool(self._upstream_links(target_tls)),
            "has_usable_internal_coordination": bool(
                consumed or injected_current_count),
            "usable_internal_event_count": (
                len(consumed) + injected_current_count),
            "missing_movement_mapping_count": missing_movement_mapping_count,
        }
        result["__meta__"]["has_usable_coordination"] = result[
            "__meta__"]["has_usable_internal_coordination"]
        return result

    @staticmethod
    def _apply_event(result, phases, event):
        entry = event["target_entry_direction"]
        movement = event.get("target_movement")
        if (not movement or len(movement) != 2
                or movement[0] != entry or movement[1] not in ("T", "L")):
            raise ValueError(
                "coordination event must contain a lane-level downstream "
                f"through/left movement: entry={entry!r} movement={movement!r}"
            )
        count = int(event.get("outbound_vehicle_count", 0))
        source = event["source_tls"]
        target_phases = V36CoordinationManager._phases_for_movement(
            phases, movement)
        for phase in target_phases:
            result[phase]["arrival_15s"] += count
            key = movement
            result[phase]["internal"][key] = (
                result[phase]["internal"].get(key, 0) + count)
            result[phase]["sources"][key] = source
            result[phase].setdefault("events", []).append(deepcopy(event))

    def _write_log(self, event):
        with open(self.log_path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
