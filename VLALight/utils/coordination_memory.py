import json
import math
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


_FULL_TO_SHORT_DIR = {
    "North": "N",
    "South": "S",
    "East": "E",
    "West": "W",
}


@dataclass
class CoordinationEvent:
    source_tls: str
    target_tls: str
    source_step: int
    effective_step: int
    expire_before_step: int
    source_direction: str
    movement_type: str
    target_entry_direction: str
    mass: float
    released_phase: str
    travel_time_s: float = 0.0


class CoordinationMemory:
    """
    Minimal coordination memory for SUMO-only experiments.

    This version only tracks delayed inflow arriving at a target entry direction.
    It does not infer left/straight/right intent at the downstream intersection.
    """

    def __init__(self,
                 enabled: bool = False,
                 region_structure_path: str = "",
                 active_horizon_steps: int = 1,
                 debug_dir: str = ""):
        self.enabled = bool(enabled)
        self.region_structure_path = region_structure_path or ""
        self.active_horizon_steps = max(1, int(active_horizon_steps))
        self.debug_dir = debug_dir or ""
        self._region_structure: Dict[str, Any] = {}
        self._events: List[CoordinationEvent] = []
        self._debug_log_path = ""

        if not self.enabled:
            return

        if not self.region_structure_path or not os.path.exists(self.region_structure_path):
            print(
                f"Warning: Coordination enabled but region structure not found: "
                f"{self.region_structure_path}"
            )
            self.enabled = False
            return

        with open(self.region_structure_path, "r", encoding="utf-8") as f:
            self._region_structure = json.load(f)

        if self.debug_dir:
            os.makedirs(self.debug_dir, exist_ok=True)
            self._debug_log_path = os.path.join(self.debug_dir, "coordination_events.jsonl")

    def expire_old_events(self, current_step: int) -> None:
        if not self.enabled:
            return
        self._events = [
            ev for ev in self._events
            if current_step < ev.expire_before_step
        ]

    def get_direction_incoming(self, tls_id: str, step_num: int,
                                 decision_interval: int = 35) -> Dict[str, Dict]:
        """
        Return per-direction incoming pressure for the macro direction-pair layer.
        
        Filters out events where the remaining travel time is < 5 seconds
        (those vehicles have likely already arrived and been counted in current V).
        
        Three-level remainder filter:
        - < 5s: vehicle already in current V → skip
        - 5~25s: vehicle will arrive this cycle → this_cycle (can be added to total)
        - > 25s: vehicle will arrive next cycle → next_cycle (preview only)

        Returns: {
            "this_cycle": {short_dir: {"mass": float, "sources": [str], "remaining_seconds": float}},
            "next_cycle": {short_dir: {"mass": float, "sources": [str], "remaining_seconds": float}}
        }
        """
        if not self.enabled:
            return {"this_cycle": {}, "next_cycle": {}}

        decision_interval = max(1, int(decision_interval))
        self.expire_old_events(step_num)
        this_cycle: Dict[str, Dict] = {}
        next_cycle: Dict[str, Dict] = {}
        for ev in self._events:
            if ev.target_tls != tls_id:
                continue
            if not (ev.effective_step <= step_num < ev.expire_before_step):
                continue
            remainder = ev.travel_time_s % decision_interval
            if remainder < 5:
                continue
            d = ev.target_entry_direction
            bucket = this_cycle if remainder <= 25 else next_cycle
            if d not in bucket:
                bucket[d] = {"mass": 0.0, "sources": [], "remaining_seconds": None}
            bucket[d]["mass"] += ev.mass
            bucket[d]["sources"].append(ev.source_tls)
            if bucket[d]["remaining_seconds"] is None or remainder < bucket[d]["remaining_seconds"]:
                bucket[d]["remaining_seconds"] = round(remainder, 1)
        # deduplicate sources
        for d in this_cycle:
            this_cycle[d]["sources"] = sorted(set(this_cycle[d]["sources"]))
        for d in next_cycle:
            next_cycle[d]["sources"] = sorted(set(next_cycle[d]["sources"]))
        return {"this_cycle": this_cycle, "next_cycle": next_cycle}

    def get_prompt_context(self, tls_id: str, step_num: int) -> str:
        if not self.enabled:
            return ""

        self.expire_old_events(step_num)
        return self._build_prompt_context_from_current_events(tls_id, step_num)

    def prepare_prompt_contexts(self, tls_ids: List[str], step_num: int) -> Dict[str, str]:
        """Prepare a read-only prompt-context snapshot for all target intersections."""
        if not self.enabled:
            return {tls_id: "" for tls_id in tls_ids}

        self.expire_old_events(step_num)
        return {
            tls_id: self._build_prompt_context_from_current_events(tls_id, step_num)
            for tls_id in tls_ids
        }

    def _build_prompt_context_from_current_events(self, tls_id: str, step_num: int) -> str:
        entry_mass: Dict[str, float] = {}
        entry_sources: Dict[str, List[str]] = {}
        for ev in self._events:
            if ev.target_tls != tls_id:
                continue
            if not (ev.effective_step <= step_num < ev.expire_before_step):
                continue
            entry_mass[ev.target_entry_direction] = entry_mass.get(ev.target_entry_direction, 0.0) + ev.mass
            entry_sources.setdefault(ev.target_entry_direction, []).append(ev.source_tls)

        if not entry_mass:
            return ""

        lines = [
            "== UPSTREAM COORDINATION MEMORY ==",
            "Potential delayed arrivals expected during this control interval:",
        ]
        for direction in ["N", "S", "E", "W"]:
            mass = entry_mass.get(direction, 0.0)
            if mass <= 0:
                continue
            unique_sources = sorted(set(entry_sources.get(direction, [])))
            source_text = ", ".join(unique_sources) if unique_sources else "upstream neighbor"
            lines.append(
                f"- {direction} approach: {mass:.0f} vehicles incoming from {source_text}"
            )

        lines.append(
            "Note: these vehicles will enter from the given direction, "
            "but their specific lane (left/straight/right) at this intersection is unknown."
        )
        return "\n".join(lines)

    def record_phase_release(self,
                             source_tls: str,
                             step_num: int,
                             phase_name: str,
                             allowed_lanes: Dict[str, str],
                             phase_stats: Dict[str, Dict[str, float]],
                             decision_interval: int) -> None:
        if not self.enabled:
            return

        intersections = self._region_structure.get("intersections", {})
        source_cfg = intersections.get(source_tls)
        if not source_cfg:
            return

        movements = source_cfg.get("movements", {})
        decision_interval = max(1, int(decision_interval))

        chosen_phase_stats = phase_stats.get(phase_name, {}) or {}
        for full_direction, movement_type in allowed_lanes.items():
            direction = _FULL_TO_SHORT_DIR.get(full_direction)
            if not direction or movement_type not in ("left", "straight"):
                continue

            lane_stats = chosen_phase_stats.get(full_direction, {}) or {}
            mass = float(lane_stats.get("V", 0) or 0)
            if mass <= 0:
                continue

            movement_info = movements.get(direction, {}).get(movement_type)
            if not movement_info:
                continue

            target_tls = movement_info.get("target_neighbor_id")
            target_entry_direction = movement_info.get("target_entry_direction")
            travel_time_s = movement_info.get("travel_time_s")
            is_boundary = movement_info.get("is_boundary", True)

            if is_boundary or not target_tls or not target_entry_direction:
                continue

            travel_time_s = float(travel_time_s or 0.0)
            delay_steps = max(1, math.floor(travel_time_s / decision_interval))
            effective_step = step_num + delay_steps
            expire_before_step = effective_step + self.active_horizon_steps

            event = CoordinationEvent(
                source_tls=source_tls,
                target_tls=target_tls,
                source_step=step_num,
                effective_step=effective_step,
                expire_before_step=expire_before_step,
                source_direction=direction,
                movement_type=movement_type,
                target_entry_direction=target_entry_direction,
                mass=mass,
                released_phase=str(phase_name),
                travel_time_s=travel_time_s,
            )
            self._events.append(event)
            self._append_debug_event(event)

    def _append_debug_event(self, event: CoordinationEvent) -> None:
        if not self._debug_log_path:
            return
        try:
            with open(self._debug_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Module-level singleton: all llmlight agents in the same process share
# one CoordinationMemory instance.  Avoids touching vlm_oneline.py.
# ---------------------------------------------------------------------------
_shared_instance: Optional["CoordinationMemory"] = None
_shared_init_kwargs: Dict[str, Any] = {}


def configure_shared(**kwargs) -> None:
    """Set init kwargs for the shared singleton (call once at startup)."""
    global _shared_init_kwargs
    _shared_init_kwargs = kwargs


def get_shared() -> "CoordinationMemory":
    """Return (and lazily create) the shared CoordinationMemory singleton."""
    global _shared_instance
    if _shared_instance is None:
        _shared_instance = CoordinationMemory(**_shared_init_kwargs)
    return _shared_instance
