#!/usr/bin/env python3
"""CPU-only contract tests for the online cooperative Stage 2 path."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from v35_online_cooperative_grpo.mode_assignment import (  # noqa: E402
    assignment_matrix,
    build_mode_assignment,
    validate_mode_assignment,
)
from v35_online_cooperative_grpo.online_reward import (  # noqa: E402
    compute_online_reward,
    reasoning_penalty,
)
from v35_online_cooperative_grpo.online_rollout import (  # noqa: E402
    CitySnapshot,
    IntersectionObservation,
    flatten_for_gdpo,
    rollout_city,
    validate_city_snapshot,
)
from v35_online_cooperative_grpo.online_config import (  # noqa: E402
    CityConfig,
    OnlineConfig,
    SplitConfig,
)
from v35_online_cooperative_grpo.online_samples import (  # noqa: E402
    MasterStreamManager,
    iter_online_samples,
)
from v35_online_cooperative_grpo.stage2_protocol import (  # noqa: E402
    build_stage2_prompt,
    parse_decision_response,
)


def _perception(intersection_id: str, neighbors: list[str]) -> tuple[dict, dict]:
    local = {"current_phase": "ETWT", "phases": {phase: {} for phase in ("ETWT", "NTST", "ELWL", "NLSL")}}
    cooperative = {
        "local_coordination": {"ET": {"count": 1, "is_boundary": "no"}},
        "neighbors": {
            neighbor: {"intersection_id": neighbor, "upstream_movements": {"NT": {"count": 1}}, "travel_time_s": 20}
            for neighbor in neighbors
        },
    }
    return local, cooperative


class FakeSimulator:
    def __init__(self, ids: list[str]) -> None:
        self.ids = ids
        self.queues = {value: 10.0 for value in ids}
        self.restore_count = 0
        self.applied: list[dict[str, str]] = []
        self.advanced: list[int] = []

    def restore(self, snapshot) -> None:
        self.restore_count += 1
        self.queues = {value: 10.0 for value in self.ids}

    def apply_signals(self, signals) -> None:
        self.applied.append(dict(signals))

    def advance(self, decision_cycles: int) -> None:
        self.advanced.append(decision_cycles)
        self.queues = {key: value - 1.0 for key, value in self.queues.items()}

    def queue_metrics(self):
        return dict(self.queues)


class FakeMaster:
    """Small persistent-master double for lifecycle assertions."""

    def __init__(self) -> None:
        self.time = 0.0
        self.resets: list[int] = []
        self.commits: list[tuple[dict[str, str], int]] = []

    def current_time(self) -> float:
        return self.time

    def current_signal_table(self) -> dict[str, str]:
        return {"a": "ETWT"}

    def apply_signals(self, signals) -> None:
        self.signals = dict(signals)

    def advance(self, cycles: int) -> None:
        self.time += 30.0 * cycles

    def commit_signals(self, signals, decision_cycles: int) -> None:
        self.commits.append((dict(signals), decision_cycles))
        self.apply_signals(signals)
        self.advance(decision_cycles)

    def advance_v25(self, decision_cycles: int = 1) -> None:
        self.advance(decision_cycles)

    def reset(self, *, seed: int, use_gui: bool = False) -> None:
        self.resets.append(seed)
        self.time = 0.0

    def close(self) -> None:
        pass


def main() -> None:
    for count in (12, 16, 196):
        ids = [f"i{index}" for index in range(count)]
        assignment = build_mode_assignment(ids, num_rollouts=6, seed=7)
        validate_mode_assignment(assignment, ids)
        assert all(sum(row[item] == "fast" for row in assignment) == 3 for item in ids)
        assert len(assignment_matrix(ids)) == 6

    fast = parse_decision_response("<mode>fast</mode>\n<signal>NTST</signal>", forced_mode="fast")
    slow = parse_decision_response(
        "<mode>slow</mode>\n<reasoning>compare queues</reasoning>\n<signal>NTST</signal>",
        forced_mode="slow",
    )
    invalid_signal = parse_decision_response("<mode>fast</mode>\n<signal>BAD</signal>", forced_mode="fast")
    malformed = parse_decision_response("<mode>slow</mode>\n<signal>NTST</signal>", forced_mode="slow")
    assert fast.format_penalty == 0.0 and fast.format_valid
    assert slow.format_penalty == 0.0 and slow.format_valid
    assert invalid_signal.format_penalty == -1.0
    assert malformed.format_penalty == -0.5
    assert reasoning_penalty(1) > 0.0
    assert reasoning_penalty(300, free_tokens=300) == 0.0

    local_a, coop_a = _perception("a", ["b"])
    local_b, coop_b = _perception("b", ["a"])
    snapshot = CitySnapshot(
        city="test",
        step=42,
        observations=(
            IntersectionObservation("a", 42, local_a, coop_a),
            IntersectionObservation("b", 42, local_b, coop_b),
        ),
        simulator_snapshot="snapshot-42",
        required_neighbors={"a": ["b"], "b": ["a"]},
    )
    validate_city_snapshot(snapshot)
    prompt = build_stage2_prompt(local_a, coop_a, forced_mode="fast")
    assert "<local_perception>" in prompt and "<cooperative_perception>" in prompt

    simulator = FakeSimulator(["a", "b"])

    def policy(_intersection_id: str, _prompt: str, _prefix: str) -> str:
        return "<mode>fast</mode>\n<signal>ETWT</signal>"

    results = rollout_city(snapshot, simulator, policy, num_rollouts=6, decision_cycles=3)
    assert len(results) == 6
    assert all(len(result.results) == 2 for result in results)
    assert simulator.restore_count == 6
    assert simulator.advanced == [3] * 6
    assert all("global_queue_reward" in row.reward and "local_queue_reward" in row.reward for row in results[0].results)
    records = flatten_for_gdpo(results)
    assert len(records) == 12
    assert len({record["group_id"] for record in records}) == 2
    assert all(record["prompt"] and record["response"] for record in records)

    # Persistent stream lifecycle: warmup ends at step 6, a three-cycle
    # candidate advances 90 seconds, and a near-episode-end snapshot restarts
    # before the candidate horizon would cross 3600 seconds.
    config = OnlineConfig(
        cities={"jinan": CityConfig("jinan")},
        train=SplitConfig(("jinan",), batch_size=1, seed_base=11),
        val=SplitConfig(("jinan",), batch_size=1, seed_base=22),
        episode_seconds=3600,
        decision_cycle_seconds=30,
        warmup_steps=5,
        evaluation_decision_cycles=3,
    )
    master = FakeMaster()
    manager = MasterStreamManager(config, split="train", master_factory=lambda *_: master)
    spec = next(iter(iter_online_samples(config, "train", count=1)))

    def fake_snapshot(_master, _city, step):
        assert step == 6
        return CitySnapshot(
            city="jinan",
            step=step,
            observations=(IntersectionObservation("a", step, {}, {"neighbors": {}}),),
        )

    stream = manager.get_or_create(spec)
    first = stream.snapshot(fake_snapshot, decision_cycles=3)
    assert first.step == 6 and master.time == 180.0
    stream.commit({"a": "ETWT"}, decision_cycles=3)
    assert master.time == 270.0
    master.time = 3570.0
    restarted = stream.snapshot(fake_snapshot, decision_cycles=3)
    assert restarted.step == 6 and master.resets == [100011]
    assert manager.get_or_create(spec) is stream
    manager.close()
    print("PASS: online cooperative Stage 2 contracts")


if __name__ == "__main__":
    main()
