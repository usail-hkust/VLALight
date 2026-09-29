"""Synchronous city snapshot rollouts for cooperative Stage 2 GRPO."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence

from .mode_assignment import build_mode_assignment, validate_mode_assignment
from .online_reward import compute_endpoint_scores, compute_online_reward, mode_utility
from .stage2_protocol import (
    DecisionResponse, assistant_prefix, build_stage2_prompt, executable_signal,
    parse_decision_response,
)


class SimulatorAdapter(Protocol):
    """Minimal SUMO adapter required by the rollout coordinator."""

    def restore(self, snapshot: Any) -> None: ...
    def apply_signals(self, signals: Mapping[str, str]) -> None: ...
    def advance(self, decision_cycles: int) -> None: ...
    def queue_metrics(self) -> Mapping[str, float]: ...


@dataclass(frozen=True)
class IntersectionObservation:
    intersection_id: str
    step: int
    local_perception: Any
    cooperative_perception: Any
    current_phase: str = "ETWT"
    audit_metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CitySnapshot:
    city: str
    step: int
    observations: tuple[IntersectionObservation, ...]
    simulator_snapshot: Any = None
    required_neighbors: Mapping[str, Sequence[str]] | None = None
    sumo_time_s: float | None = None


@dataclass
class IntersectionRolloutResult:
    intersection_id: str
    forced_mode: str | None
    prompt: str
    response: str
    parsed: DecisionResponse
    reward: dict[str, Any]
    local_perception: Any = None
    cooperative_perception: Any = None
    perception_audit: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class CityRolloutResult:
    city: str
    step: int
    rollout_id: int
    results: list[IntersectionRolloutResult] = field(default_factory=list)
    before_queues: dict[str, float] = field(default_factory=dict)
    after_queues: dict[str, float] = field(default_factory=dict)
    # Temporal rollouts retain every decision for diagnostics/GDPO grouping.
    cycle_results: list[dict[str, Any]] = field(default_factory=list)
    cumulative_rewards: dict[str, dict[str, Any]] = field(default_factory=dict)


def flatten_for_gdpo(results: Sequence[CityRolloutResult]) -> list[dict[str, Any]]:
    """Flatten city results into one GDPO record per intersection decision.

    The group ID is city/step/intersection, so the six forced-mode candidates
    for one target are compared together.  Each record retains its full
    prompt, response, rollout ID, and independent reward components.
    """
    records: list[dict[str, Any]] = []
    for city_result in results:
        group_prefix = f"{city_result.city}:{city_result.step}"
        for row in city_result.results:
            reward = dict(row.reward)
            # Temporal rollouts expose trajectory-level sums for GDPO; the
            # legacy single-snapshot path keeps its per-window reward.
            reward.update(city_result.cumulative_rewards.get(row.intersection_id, {}))
            records.append({
                "group_id": f"{group_prefix}:{row.intersection_id}",
                "mode_group_id": f"{group_prefix}:{row.intersection_id}",
                "network_group_id": group_prefix,
                "episode_group_id": group_prefix,
                "city": city_result.city,
                "step": city_result.step,
                "rollout_id": city_result.rollout_id,
                "intersection_id": row.intersection_id,
                "prompt": row.prompt,
                "response": row.response,
                "mode": row.forced_mode,
                "parsed_mode": row.parsed.mode,
                "parsed_signal": row.parsed.signal,
                "reasoning": row.parsed.reasoning,
                "signal_valid": bool(row.parsed.signal_valid),
                "format_valid": bool(row.parsed.format_valid),
                "before_queues": dict(city_result.before_queues),
                "after_queues": dict(city_result.after_queues),
                "trajectory_signals": [
                    dict(item) for item in city_result.cycle_results
                    if str(item.get("intersection_id")) == row.intersection_id
                ],
                "local_perception": row.local_perception,
                "cooperative_perception": row.cooperative_perception,
                "perception_audit": dict(row.perception_audit),
                "candidate_intersections": tuple(item.intersection_id for item in city_result.results),
                # One network reward is shared by the episode's 12 decision
                # sequences; the trainer must average these rows per episode.
                "episode_reward": float(reward.get("network_reward", reward.get("global_score", 0.0))),
                "train_cycle": int(reward.get("train_cycle", 0)),
                **reward,
            })
    return records


def _neighbor_ids(value: Any) -> set[str]:
    if not isinstance(value, Mapping):
        return set()
    # Cooperative perception has an explicit ``neighbors`` section.  Never
    # inspect the whole object: fields such as local_coordination are not IDs.
    neighbors = value.get("neighbors")
    if isinstance(neighbors, Sequence) and not isinstance(neighbors, (str, bytes, bytearray)):
        found: set[str] = set()
        for item in neighbors:
            if isinstance(item, Mapping):
                candidate = item.get("intersection_id", item.get("neighbor_id", item.get("id")))
                if candidate is not None:
                    found.add(str(candidate))
        return found
    if not isinstance(neighbors, Mapping):
        return set()
    found: set[str] = set()
    for key, item in neighbors.items():
        if isinstance(item, str):
            found.add(item)
        elif isinstance(item, Mapping):
            candidate = item.get(
                "intersection_id",
                item.get("neighbor_id", item.get("source_intersection", item.get("id"))),
            )
            if candidate is not None:
                found.add(str(candidate))
    return found


def validate_city_snapshot(snapshot: CitySnapshot) -> None:
    if not snapshot.observations:
        raise ValueError("city snapshot must contain at least one intersection")
    ids = [row.intersection_id for row in snapshot.observations]
    if len(ids) != len(set(ids)):
        raise ValueError("city snapshot contains duplicate intersection IDs")
    if any(row.step != snapshot.step for row in snapshot.observations):
        raise ValueError("all observations in a rollout must have the same step")
    id_set = set(ids)
    for row in snapshot.observations:
        referenced = _neighbor_ids(row.cooperative_perception)
        unknown = referenced - id_set
        if unknown:
            raise ValueError(f"{row.intersection_id} references missing same-step neighbors: {sorted(unknown)}")
    if snapshot.required_neighbors is not None:
        for intersection_id in ids:
            expected = {str(v) for v in snapshot.required_neighbors.get(intersection_id, ())}
            actual = _neighbor_ids(next(row.cooperative_perception for row in snapshot.observations if row.intersection_id == intersection_id))
            if actual != expected:
                raise ValueError(f"topology mismatch for {intersection_id}: expected {sorted(expected)}, got {sorted(actual)}")


def _response_text(value: str | DecisionResponse) -> str:
    return value.raw if isinstance(value, DecisionResponse) else str(value)


def rollout_one_assignment(
    snapshot: CitySnapshot,
    simulator: SimulatorAdapter,
    policy_fn: Callable[[str, str, str], str | DecisionResponse],
    modes: Mapping[str, str],
    *,
    rollout_id: int = 0,
    prompt_template: str | None = None,
    decision_cycles: int = 3,
    concurrency: int | None = None,
    tokenizer: Any = None,
    global_weight: float = 0.5,
    local_weight: float = 1.0,
    reasoning_weight: float = 0.5,
    reasoning_free_tokens: int = 0,
    queue_scale: float = 1.0,
    global_queue_scale: float | None = None,
    local_queue_scale: float | None = None,
) -> CityRolloutResult:
    """Run exactly one synchronized candidate from a positioned simulator.

    The caller owns simulator positioning.  This is intentionally separate
    from ``rollout_city`` so a Ray actor can create one SUMO process, restore
    one private snapshot, and execute the same logic without sharing state.
    """
    if decision_cycles <= 0:
        raise ValueError("decision_cycles must be positive")
    validate_city_snapshot(snapshot)
    ids = [row.intersection_id for row in snapshot.observations]
    if set(modes) != set(ids) or any(modes[item] not in {"fast", "slow", None} for item in ids):
        raise ValueError("modes must assign fast, slow, or deployment mode None to every intersection")
    by_id = {row.intersection_id: row for row in snapshot.observations}
    workers = max(1, min(len(ids), concurrency or len(ids)))
    before = {str(k): float(v) for k, v in simulator.queue_metrics().items()}

    def ask(intersection_id: str) -> tuple[str, str, str | DecisionResponse]:
        row = by_id[intersection_id]
        mode = modes[intersection_id]
        prompt = build_stage2_prompt(
            row.local_perception,
            row.cooperative_perception,
            template=prompt_template,
            forced_mode=mode,
        )
        prefix = assistant_prefix(mode) if mode is not None else ""
        return intersection_id, prompt, policy_fn(intersection_id, prompt, prefix)

    prompts: dict[str, str] = {}
    responses: dict[str, str | DecisionResponse] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(ask, intersection_id) for intersection_id in ids]
        for future in as_completed(futures):
            intersection_id, prompt, response = future.result()
            prompts[intersection_id] = prompt
            responses[intersection_id] = response

    parsed = {
        intersection_id: parse_decision_response(
            _response_text(responses[intersection_id]),
            forced_mode=modes[intersection_id],
        )
        for intersection_id in ids
    }
    signals = {
        intersection_id: (
            executable_signal(parsed[intersection_id], by_id[intersection_id].current_phase)
        )
        for intersection_id in ids
    }
    simulator.apply_signals(signals)
    simulator.advance(decision_cycles)
    after = {str(k): float(v) for k, v in simulator.queue_metrics().items()}
    rows: list[IntersectionRolloutResult] = []
    for intersection_id in ids:
        reward = compute_online_reward(
            parsed[intersection_id],
            before_queues=before,
            after_queues=after,
            target_id=intersection_id,
            tokenizer=tokenizer,
            forced_mode=modes[intersection_id],
            global_weight=global_weight,
            local_weight=local_weight,
            reasoning_weight=reasoning_weight,
            reasoning_free_tokens=reasoning_free_tokens,
            queue_scale=queue_scale,
            global_queue_scale=global_queue_scale,
            local_queue_scale=local_queue_scale,
        )
        rows.append(
            IntersectionRolloutResult(
                intersection_id,
                modes[intersection_id],
                prompts[intersection_id],
                _response_text(responses[intersection_id]),
                parsed[intersection_id],
                reward,
                local_perception=by_id[intersection_id].local_perception,
                cooperative_perception=by_id[intersection_id].cooperative_perception,
                perception_audit=dict(by_id[intersection_id].audit_metadata),
            )
        )
    return CityRolloutResult(
        snapshot.city,
        snapshot.step,
        rollout_id,
        rows,
        before,
        after,
    )


def rollout_city(
    snapshot: CitySnapshot,
    simulator: SimulatorAdapter,
    policy_fn: Callable[[str, str, str], str | DecisionResponse],
    *,
    prompt_template: str | None = None,
    num_rollouts: int = 6,
    decision_cycles: int = 3,
    concurrency: int | None = None,
    seed: int | None = None,
    tokenizer: Any = None,
    global_weight: float = 0.5,
    local_weight: float = 1.0,
    reasoning_weight: float = 0.5,
    reasoning_free_tokens: int = 0,
    queue_scale: float = 1.0,
    global_queue_scale: float | None = None,
    local_queue_scale: float | None = None,
) -> list[CityRolloutResult]:
    """Run synchronized actions for all intersections from one snapshot.

    Every rollout restores the same simulator snapshot, gathers all responses,
    applies all signals together, advances exactly ``decision_cycles`` cycles,
    and then assigns the shared global reduction plus each target's local
    reduction to its own response.
    """
    if decision_cycles <= 0:
        raise ValueError("decision_cycles must be positive")
    if num_rollouts > 1 and snapshot.simulator_snapshot is None:
        raise ValueError(
            "multi-rollout city evaluation requires simulator_snapshot so every "
            "candidate starts from the same SUMO state"
        )
    validate_city_snapshot(snapshot)
    ids = [row.intersection_id for row in snapshot.observations]
    assignments = build_mode_assignment(ids, num_rollouts=num_rollouts, seed=seed)
    validate_mode_assignment(assignments, ids)
    workers = max(1, min(len(ids), concurrency or len(ids)))
    output: list[CityRolloutResult] = []

    for rollout_id, modes in enumerate(assignments):
        simulator.restore(snapshot.simulator_snapshot)
        output.append(
            rollout_one_assignment(
                snapshot,
                simulator,
                policy_fn,
                modes,
                rollout_id=rollout_id,
                prompt_template=prompt_template,
                decision_cycles=decision_cycles,
                concurrency=workers,
                tokenizer=tokenizer,
                global_weight=global_weight,
                local_weight=local_weight,
                reasoning_weight=reasoning_weight,
                reasoning_free_tokens=reasoning_free_tokens,
                queue_scale=queue_scale,
                global_queue_scale=global_queue_scale,
                local_queue_scale=local_queue_scale,
            )
        )
    return output


def rollout_one_assignment_temporal(
    snapshot: CitySnapshot,
    simulator: SimulatorAdapter,
    policy_fn: Callable[[str, str, str], str | DecisionResponse],
    modes: Mapping[str, str],
    *,
    rollout_id: int = 0,
    prompt_template: str | None = None,
    decision_cycles: int = 3,
    observation_fn: Callable[[SimulatorAdapter, int], CitySnapshot] | None = None,
    stage1_fn: Callable[[CitySnapshot], CitySnapshot] | None = None,
    router_fn: Callable[[CitySnapshot], CitySnapshot] | None = None,
    concurrency: int | None = None,
    tokenizer: Any = None,
    global_weight: float = 0.5,
    local_weight: float = 1.0,
    reasoning_weight: float = 0.5,
    reasoning_free_tokens: int = 0,
    queue_scale: float = 1.0,
    global_queue_scale: float | None = None,
    local_queue_scale: float | None = None,
) -> CityRolloutResult:
    """Run t0..tN decisions, with Stage 1 only at t0.

    ``decision_cycles=3`` means four decisions (t0, t1, t2, t3) and three
    one-cycle SUMO advances.  ``observation_fn`` must sample the live SUMO
    state for t1 onward; it may be ``build_city_snapshot`` partially applied.
    ``stage1_fn`` is an explicit hook for the perception model and is called
    exactly once on t0.  It produces per-intersection local ``<perception>``
    data; it must not manufacture the cross-intersection object.  ``router_fn``
    assembles that data with neighbors into each row's Stage 2 snapshot and is
    applied at t0 and after every live SUMO observation.
    """
    if decision_cycles <= 0:
        raise ValueError("decision_cycles must be positive")
    validate_city_snapshot(snapshot)
    ids = [row.intersection_id for row in snapshot.observations]
    if set(modes) != set(ids) or any(modes[item] not in {"fast", "slow"} for item in ids):
        raise ValueError("modes must assign fast or slow to every intersection")
    if observation_fn is None and decision_cycles > 0:
        candidate = getattr(simulator, "build_snapshot", None)
        if callable(candidate):
            observation_fn = candidate
        else:
            raise ValueError("temporal rollout requires observation_fn for t1 onward")

    current = stage1_fn(snapshot) if stage1_fn is not None else snapshot
    if router_fn is not None:
        current = router_fn(current)
    validate_city_snapshot(current)
    cumulative: dict[str, dict[str, Any]] = {
        item: {"global_queue_reward": 0.0, "local_queue_reward": 0.0,
               "reasoning_cost_reward": 0.0, "format_penalty": 0.0,
               "score": 0.0, "cycles": 0}
        for item in ids
    }
    cycle_records: list[dict[str, Any]] = []
    first_before: dict[str, float] = {}
    final_after: dict[str, float] = {}
    first_after: dict[str, float] = {}
    first_rows: dict[str, IntersectionRolloutResult] = {}
    last_rows: dict[str, IntersectionRolloutResult] = {}

    for cycle in range(decision_cycles + 1):
        validate_city_snapshot(current)
        by_id = {row.intersection_id: row for row in current.observations}
        before = {str(k): float(v) for k, v in simulator.queue_metrics().items()}
        if cycle == 0:
            first_before = dict(before)
        prompts: dict[str, str] = {}
        responses: dict[str, str | DecisionResponse] = {}

        def ask(intersection_id: str):
            row = by_id[intersection_id]
            mode = modes[intersection_id] if cycle == 0 else None
            prompt = build_stage2_prompt(row.local_perception, row.cooperative_perception,
                                         template=prompt_template, forced_mode=mode)
            return intersection_id, prompt, policy_fn(intersection_id, prompt, assistant_prefix(mode))

        workers = max(1, min(len(ids), concurrency or len(ids)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(ask, item) for item in ids]
            for future in as_completed(futures):
                item, prompt, response = future.result()
                prompts[item], responses[item] = prompt, response
        parsed = {
            item: parse_decision_response(
                _response_text(responses[item]), forced_mode=modes[item] if cycle == 0 else None
            )
            for item in ids
        }
        signals = {
            item: executable_signal(parsed[item], by_id[item].current_phase)
            for item in ids
        }
        simulator.apply_signals(signals)
        # decision_cycles denotes the number of transitions after t0.  The
        # final t3 decision is evaluated at t3 and must not create a t4 state.
        if cycle < decision_cycles:
            simulator.advance(1)
        after = {str(k): float(v) for k, v in simulator.queue_metrics().items()}
        if cycle == 0:
            first_after = dict(after)
        final_after = dict(after)
        cycle_rows = []
        for item in ids:
            if cycle == 0:
                reward = compute_online_reward(
                    parsed[item], before_queues=before, after_queues=after,
                    target_id=item, tokenizer=tokenizer, forced_mode=modes[item],
                    global_weight=global_weight, local_weight=local_weight,
                    reasoning_weight=reasoning_weight, reasoning_free_tokens=reasoning_free_tokens,
                    queue_scale=queue_scale, global_queue_scale=global_queue_scale,
                    local_queue_scale=local_queue_scale,
                )
                for key in ("global_queue_reward", "local_queue_reward", "reasoning_cost_reward", "format_penalty", "score"):
                    cumulative[item][key] = float(reward[key])
                cumulative[item]["cycles"] = 1
                row = IntersectionRolloutResult(
                    item, modes[item], prompts[item], _response_text(responses[item]), parsed[item], reward,
                    local_perception=by_id[item].local_perception,
                    cooperative_perception=by_id[item].cooperative_perception,
                    perception_audit=dict(by_id[item].audit_metadata),
                )
                last_rows[item] = row
                first_rows[item] = row
            cycle_rows.append({
                "cycle": cycle,
                "intersection_id": item,
                "signal": signals[item],
                "local_perception": by_id[item].local_perception,
                "cooperative_perception": by_id[item].cooperative_perception,
                "perception_audit": dict(by_id[item].audit_metadata),
            })
        cycle_records.extend(cycle_rows)
        if cycle < decision_cycles:
            current = observation_fn(simulator, snapshot.step + cycle + 1)  # type: ignore[misc]
            if router_fn is not None:
                current = router_fn(current)

    # Training uses the t0 response only.  t1..t3 are simulator evaluation
    # steps; their endpoint observations are attached to the t0 records.
    train_rows = first_rows or last_rows
    for item, row in train_rows.items():
        global_score, local_score = compute_endpoint_scores(
            first_before, final_after, item
        )
        local_t1 = compute_endpoint_scores(first_before, first_after, item)[1]
        penalty = float(row.reward.get("reasoning_penalty", 0.0))
        team_local = sum(
            compute_endpoint_scores(first_before, final_after, other)[1]
            for other in ids
        ) / len(ids)
        cumulative[item].update({
            "global_score": global_score,
            "local_score": local_t1,
            "local_long_term_score": local_score,
            "local_mean_score": team_local,
            "network_reward": global_score,
            "mode_utility": mode_utility(local_t1, global_score, penalty),
            "train_cycle": 0,
        })
        row.reward.update(cumulative[item])
    return CityRolloutResult(snapshot.city, snapshot.step, rollout_id, list(train_rows.values()),
                             first_before, final_after, cycle_records, cumulative)
