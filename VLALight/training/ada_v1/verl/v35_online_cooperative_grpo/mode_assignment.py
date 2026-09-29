"""Balanced FAST/SLOW exploration assignments for city-level rollouts."""

from __future__ import annotations

import random
from typing import Sequence


MODES = ("fast", "slow")


def build_mode_assignment(
    intersection_ids: Sequence[str],
    *,
    num_rollouts: int = 6,
    seed: int | None = None,
) -> list[dict[str, str]]:
    """Return ``num_rollouts`` mode maps with exactly half of each mode per ID.

    The default six-rollout schedule is deterministic and gives every
    intersection three FAST and three SLOW trials.  A seeded permutation only
    changes which intersection starts with FAST; it does not change the 3:3
    guarantee.  For an odd number of intersections, each rollout row differs
    by at most one FAST/SLOW assignment.
    """
    ids = [str(value) for value in intersection_ids]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("intersection_ids must be non-empty and unique")
    if num_rollouts <= 0 or num_rollouts % 2:
        raise ValueError("num_rollouts must be a positive even number")

    order = list(range(len(ids)))
    if seed is not None:
        random.Random(seed).shuffle(order)
    return [
        {
            ids[index]: ("fast" if (position + rollout_id) % 2 == 0 else "slow")
            for position, index in enumerate(order)
        }
        for rollout_id in range(num_rollouts)
    ]


def validate_mode_assignment(
    assignment: Sequence[dict[str, str]],
    intersection_ids: Sequence[str],
    *,
    expected_per_mode: int | None = None,
) -> None:
    """Raise ``ValueError`` when a schedule violates the per-ID balance."""
    ids = [str(value) for value in intersection_ids]
    if not assignment:
        raise ValueError("mode assignment must contain at least one rollout")
    if set(assignment[0]) != set(ids):
        raise ValueError("assignment IDs do not match intersection IDs")
    for row in assignment:
        if set(row) != set(ids) or any(row[value] not in MODES for value in ids):
            raise ValueError("every rollout must assign fast or slow to every intersection")
    required = expected_per_mode if expected_per_mode is not None else len(assignment) // 2
    if len(assignment) != 2 * required:
        raise ValueError("expected_per_mode must be half the number of rollouts")
    for intersection_id in ids:
        counts = {mode: sum(row[intersection_id] == mode for row in assignment) for mode in MODES}
        if counts != {"fast": required, "slow": required}:
            raise ValueError(f"{intersection_id} is not balanced: {counts}")


def assignment_matrix(
    intersection_ids: Sequence[str], *, num_rollouts: int = 6, seed: int | None = None
) -> list[list[str]]:
    """Return the same schedule as a compact matrix ordered by ``intersection_ids``."""
    schedule = build_mode_assignment(intersection_ids, num_rollouts=num_rollouts, seed=seed)
    validate_mode_assignment(schedule, intersection_ids)
    return [[row[str(intersection_id)] for intersection_id in intersection_ids] for row in schedule]
