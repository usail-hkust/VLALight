"""Select the upstream pseudo-video frame used by V36 coordination."""

from dataclasses import asdict, dataclass
from typing import Sequence
import argparse


OPPOSITE_DIRECTION = {
    "E": "W",
    "W": "E",
    "N": "S",
    "S": "N",
}
CONTROLLED_TURNS = ("T", "L")


def target_entry_for_source_exit(source_exit_direction: str) -> str:
    """Map a source outbound camera direction to the downstream entry."""
    source_exit = str(source_exit_direction or "").upper()
    try:
        return OPPOSITE_DIRECTION[source_exit]
    except KeyError as exc:
        raise ValueError(
            f"unsupported source exit direction: {source_exit_direction!r}"
        ) from exc


def controlled_movements_for_entry(entry_direction: str) -> tuple[str, str]:
    """Return the downstream through/left movements for one entry."""
    entry = str(entry_direction or "").upper()
    if entry not in OPPOSITE_DIRECTION:
        raise ValueError(f"unsupported target entry direction: {entry_direction!r}")
    return tuple(f"{entry}{turn}" for turn in CONTROLLED_TURNS)


@dataclass(frozen=True)
class FrameCandidate:
    frame_index: int
    frame_time_s: float
    injection_delay_cycles: int
    target_green_time_s: float
    eta_min_s: float
    eta_max_s: float
    overlap_min_s: float
    overlap_max_s: float
    overlap_s: float


def evaluate_frame_candidate(
    edge_distance_m: float,
    frame_time_s: float,
    injection_delay_cycles: int,
    *,
    frame_index: int = 0,
    cycle_s: float = 30.0,
    view_distance_m: float = 150.0,
    speed_mps: float = 11.0,
    arrival_window_s: tuple[float, float] = (-5.0, 15.0),
) -> FrameCandidate:
    """Score one source frame against the downstream green-relative ETA window."""
    if edge_distance_m <= 0 or view_distance_m <= 0 or speed_mps <= 0:
        raise ValueError("distance and speed values must be positive")
    target_green = cycle_s * (injection_delay_cycles + 1)
    travel_min = (edge_distance_m - 2.0 * view_distance_m) / speed_mps
    travel_max = (edge_distance_m - view_distance_m) / speed_mps
    eta_min = frame_time_s + travel_min - target_green
    eta_max = frame_time_s + travel_max - target_green
    window_min, window_max = arrival_window_s
    overlap_min = max(eta_min, window_min)
    overlap_max = min(eta_max, window_max)
    overlap = max(0.0, overlap_max - overlap_min)
    return FrameCandidate(
        frame_index=int(frame_index),
        frame_time_s=float(frame_time_s),
        injection_delay_cycles=int(injection_delay_cycles),
        target_green_time_s=float(target_green),
        eta_min_s=float(eta_min),
        eta_max_s=float(eta_max),
        overlap_min_s=float(overlap_min),
        overlap_max_s=float(overlap_max),
        overlap_s=float(overlap),
    )


def select_coordination_frame(
    edge_distance_m: float,
    *,
    # MasterVisualCapture records after completed SUMO ticks, so frame 1 is
    # t=5s and frame 6 is t=30s (not t=0s..25s).
    frame_times_s: Sequence[float] = (5, 10, 15, 20, 25, 30),
    max_injection_delay_cycles: int = 3,
    **kwargs,
) -> tuple[FrameCandidate, list[FrameCandidate]]:
    """Maximize ETA-window overlap; ties prefer the newest source frame."""
    candidates = [
        evaluate_frame_candidate(
            edge_distance_m,
            frame_time,
            delay,
            frame_index=index + 1,
            **kwargs,
        )
        for delay in range(max_injection_delay_cycles + 1)
        for index, frame_time in enumerate(frame_times_s)
    ]
    selected = max(
        candidates,
        key=lambda item: (
            item.overlap_s,
            item.frame_time_s,
            -item.injection_delay_cycles,
        ),
    )
    return selected, candidates


def build_topology_coordination_plan(
        network_topology, target_tls, *, view_distance_m: float = 150.0,
        **kwargs):
    """Return the canonical source/frame plan shared by SUMO and RGB video.

    Links within the target camera view are already represented by its local
    perception.  They must not be reintroduced as upstream coordination.
    """
    intersections = (network_topology or {}).get("intersections", {})
    plan = []
    for source_tls, source_cfg in intersections.items():
        for exit_direction, edge in (source_cfg.get("neighbors") or {}).items():
            if edge.get("neighbor_id") != target_tls:
                continue
            source_exit = str(
                edge.get("my_exit_direction") or exit_direction).upper()
            expected_target_entry = target_entry_for_source_exit(source_exit)
            configured_target_entry = str(
                edge.get("their_entry_direction") or "").upper()
            if (configured_target_entry
                    and configured_target_entry != expected_target_entry):
                raise ValueError(
                    "coordination topology direction mismatch: "
                    f"source={source_tls} exit={source_exit} "
                    f"target={target_tls} entry={configured_target_entry}; "
                    f"expected entry={expected_target_entry}"
                )
            target_entry = configured_target_entry or expected_target_entry
            distance_m = float(edge.get("distance_m", 0))
            if distance_m <= float(view_distance_m):
                continue
            selected, _ = select_coordination_frame(
                distance_m, view_distance_m=view_distance_m, **kwargs)
            plan.append({
                "source_tls": str(source_tls),
                "target_tls": str(target_tls),
                "source_exit_direction": source_exit,
                "target_entry_direction": target_entry,
                "target_movements": list(
                    controlled_movements_for_entry(target_entry)),
                "distance_m": distance_m,
                **asdict(selected),
            })
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("distances", nargs="+", type=float)
    args = parser.parse_args()
    for distance in args.distances:
        selected, _ = select_coordination_frame(distance)
        print({"edge_distance_m": distance, **asdict(selected)})


if __name__ == "__main__":
    main()
