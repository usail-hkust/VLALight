from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

from v35_online_cooperative_grpo.online_config import CityConfig, OnlineConfig, ResourceConfig, SplitConfig
from v35_online_cooperative_grpo.online_rollout import CitySnapshot, IntersectionObservation, rollout_city
from v35_online_cooperative_grpo.online_runtime import OnlineCooperativeRuntime
from v35_online_cooperative_grpo.online_samples import (
    MasterStreamManager,
    OnlineSampleSpec,
    iter_online_samples,
)


IDS = ("a", "b")


def _snapshot(_source: Any, city: str, step: int) -> CitySnapshot:
    return CitySnapshot(
        city=city,
        step=step,
        observations=tuple(
            IntersectionObservation(
                intersection_id=intersection_id,
                step=step,
                local_perception={"current_phase": "ETWT", "phases": {}},
                cooperative_perception={"neighbors": {}},
            )
            for intersection_id in IDS
        ),
        simulator_snapshot=f"snapshot:{step}",
        required_neighbors={intersection_id: [] for intersection_id in IDS},
    )


class _Master:
    def __init__(self, _city: str, _seed: int, actor_id: str) -> None:
        self.actor_id = actor_id
        self.time = 0.0
        self.signals = {intersection_id: "ETWT" for intersection_id in IDS}
        self.commits: list[dict[str, str]] = []
        self.v25_advances = 0
        self.closed = False

    def current_time(self) -> float:
        return self.time

    def current_signal_table(self) -> Mapping[str, str]:
        return dict(self.signals)

    def apply_signals(self, signals: Mapping[str, str]) -> None:
        self.signals = dict(signals)

    def advance(self, cycles: int) -> None:
        self.time += 30.0 * int(cycles)

    def commit_signals(self, signals: Mapping[str, str], decision_cycles: int) -> None:
        self.commits.append(dict(signals))
        self.apply_signals(signals)
        self.advance(decision_cycles)

    def advance_v25(self, decision_cycles: int = 1) -> None:
        self.v25_advances += int(decision_cycles)
        self.advance(decision_cycles)

    def reset(self, *, seed: int, use_gui: bool = False) -> None:
        del seed, use_gui
        self.time = 0.0
        self.signals = {intersection_id: "ETWT" for intersection_id in IDS}

    def close(self) -> None:
        self.closed = True

    def save_snapshot(self, path: str) -> str:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"{self.time}|{self.signals}", encoding="utf-8")
        return str(target)

    def restore_snapshot(self, path: str) -> None:
        encoded = Path(path).read_text(encoding="utf-8")
        saved_time, saved_signals = encoded.split("|", 1)
        self.time = float(saved_time)
        self.signals = ast.literal_eval(saved_signals)

    def export_controller_history(self) -> dict[str, Any]:
        return {"age": int(self.time // 30)}

    def restore_controller_history(self, state: Mapping[str, Any]) -> None:
        self.restored_history = dict(state)


class _Candidate:
    def __init__(self) -> None:
        self.queues = {intersection_id: 10.0 for intersection_id in IDS}

    def restore(self, _snapshot: Any) -> None:
        self.queues = {intersection_id: 10.0 for intersection_id in IDS}

    def apply_signals(self, signals: Mapping[str, str]) -> None:
        self.signals = dict(signals)

    def advance(self, cycles: int) -> None:
        self.queues = {key: value - float(cycles) for key, value in self.queues.items()}

    def queue_metrics(self) -> Mapping[str, float]:
        return dict(self.queues)


class _Coordinator:
    def run_batch_from_master_actors(self, items, policy_factory, **kwargs):
        del kwargs
        return [
            rollout_city(
                snapshot,
                _Candidate(),
                policy_factory(),
                num_rollouts=6,
                decision_cycles=3,
            )
            for snapshot, _master, _snapshot_dir in items
        ]

    @staticmethod
    def signals_from_result(result):
        return {row.intersection_id: row.parsed.signal for row in result.results}


def test_runtime_keeps_master_timeline_and_flattens_six_candidates():
    config = OnlineConfig(
        cities={"jinan": CityConfig("jinan")},
        train=SplitConfig(("jinan",), batch_size=1, seed_base=11),
        val=SplitConfig(("jinan",), batch_size=1, seed_base=22),
        warmup_steps=5,
        evaluation_decision_cycles=3,
    )
    masters: list[_Master] = []

    def master_factory(city: str, seed: int, actor_id: str) -> _Master:
        master = _Master(city, seed, actor_id)
        masters.append(master)
        return master

    def policy_factory():
        def policy(_intersection_id: str, _prompt: str, prefix: str) -> str:
            if "<mode>slow</mode>" in prefix:
                return "<mode>slow</mode>\n<reasoning>queue comparison</reasoning>\n<signal>NTST</signal>"
            return "<mode>fast</mode>\n<signal>NTST</signal>"

        return policy

    runtime = OnlineCooperativeRuntime(
        config,
        split="train",
        master_factory=master_factory,
        coordinator=_Coordinator(),
        observation_builder=_snapshot,
        policy_factory=policy_factory,
        snapshot_root="runtime-test-snapshots",
    )
    first = runtime.rollout_and_commit()
    assert len(first.rollouts) == 1
    assert len(first.rollouts[0]) == 6
    assert len(first.gdpo_records) == 12
    assert {record["group_id"] for record in first.gdpo_records} == {"jinan:6:a", "jinan:6:b"}
    assert masters[0].actor_id == "train_master_online_slot_0000"
    assert masters[0].time == 210.0
    assert masters[0].commits == []
    assert masters[0].v25_advances == 7

    second = runtime.rollout_and_commit()
    assert {record["step"] for record in second.gdpo_records} == {7}
    assert masters[0].time == 240.0
    assert masters[0].commits == []
    assert masters[0].v25_advances == 8
    runtime.close()


def test_validation_masters_are_staggered_independently_per_city():
    config = OnlineConfig(
        cities={"jinan": CityConfig("jinan"), "hangzhou": CityConfig("hangzhou")},
        train=SplitConfig(("jinan", "hangzhou"), batch_size=4, seed_base=11),
        val=SplitConfig(
            ("jinan", "hangzhou"), batch_size=10, seed_base=22,
            stream_prefix="val", initial_step_stride=2,
        ),
        warmup_steps=5,
        evaluation_decision_cycles=3,
    )
    masters: list[_Master] = []

    def master_factory(city: str, seed: int, actor_id: str) -> _Master:
        master = _Master(city, seed, actor_id)
        masters.append(master)
        return master

    manager = MasterStreamManager(config, split="val", master_factory=master_factory)
    observed: dict[str, list[int]] = {"jinan": [], "hangzhou": []}
    for spec in iter_online_samples(config, "val"):
        stream = manager.get_or_create(spec)
        snapshot = stream.snapshot(_snapshot, decision_cycles=3)
        observed[spec.city].append(snapshot.step)

    assert observed == {
        "jinan": [6, 8, 10, 12, 14],
        "hangzhou": [6, 8, 10, 12, 14],
    }
    manager.close()


def test_validation_suites_repeat_the_same_five_decision_steps_per_city():
    config = OnlineConfig(
        cities={"jinan": CityConfig("jinan"), "hangzhou": CityConfig("hangzhou")},
        train=SplitConfig(("jinan", "hangzhou"), batch_size=4, seed_base=11),
        val=SplitConfig(
            ("jinan", "hangzhou"), batch_size=50, seed_base=22,
            stream_prefix="val", initial_step_stride=2,
        ),
        warmup_steps=5,
        val_metric_steps=5,
    )
    specs = list(iter_online_samples(config, "val"))

    assert len(specs) == 50
    assert len({spec.stream_id for spec in specs}) == 10
    assert len({spec.sample_id for spec in specs}) == 50
    for city in ("jinan", "hangzhou"):
        city_specs = [spec for spec in specs if spec.city == city]
        assert [spec.stream_id for spec in city_specs] == [
            f"val:{city}:stream:{lane:02d}"
            for _suite in range(5)
            for lane in range(5)
        ]
        assert [
            config.warmup_steps + 1 + int(spec.stream_id.rsplit(":", 1)[1]) * 2
            for spec in city_specs
        ] == [6, 8, 10, 12, 14] * 5


def test_validation_masters_continue_one_cycle_between_five_batches():
    config = OnlineConfig(
        cities={"jinan": CityConfig("jinan"), "hangzhou": CityConfig("hangzhou")},
        train=SplitConfig(("jinan", "hangzhou"), batch_size=4, seed_base=11),
        val=SplitConfig(
            ("jinan", "hangzhou"), batch_size=50, seed_base=22,
            stream_prefix="val", initial_step_stride=2,
        ),
        warmup_steps=5,
        val_metric_steps=5,
    )
    masters: list[_Master] = []

    def master_factory(city: str, seed: int, actor_id: str) -> _Master:
        master = _Master(city, seed, actor_id)
        masters.append(master)
        return master

    manager = MasterStreamManager(config, split="val", master_factory=master_factory)
    specs = list(iter_online_samples(config, "val"))
    observed: list[list[int]] = []
    for batch_start in range(0, len(specs), 10):
        steps: list[int] = []
        for spec in specs[batch_start : batch_start + 10]:
            stream = manager.get_or_create(spec)
            steps.append(stream.snapshot(_snapshot, decision_cycles=3).step)
        observed.append(steps)
        for spec in specs[batch_start : batch_start + 10]:
            manager.get_or_create(spec).advance_v25(1)

    assert observed == [
        [6, 6, 8, 8, 10, 10, 12, 12, 14, 14],
        [7, 7, 9, 9, 11, 11, 13, 13, 15, 15],
        [8, 8, 10, 10, 12, 12, 14, 14, 16, 16],
        [9, 9, 11, 11, 13, 13, 15, 15, 17, 17],
        [10, 10, 12, 12, 14, 14, 16, 16, 18, 18],
    ]
    assert len(masters) == 10
    manager.close()


def test_reset_validation_runtime_releases_only_validation_masters():
    config = OnlineConfig(
        cities={"jinan": CityConfig("jinan"), "hangzhou": CityConfig("hangzhou")},
        train=SplitConfig(("jinan", "hangzhou"), batch_size=4, seed_base=11),
        val=SplitConfig(
            ("jinan", "hangzhou"), batch_size=10, seed_base=22,
            stream_prefix="val", initial_step_stride=2,
        ),
        warmup_steps=0,
        val_metric_steps=5,
    )
    created: list[_Master] = []

    def master_factory(city: str, seed: int, actor_id: str) -> _Master:
        master = _Master(city, seed, actor_id)
        created.append(master)
        return master

    coordinator = _Coordinator()
    train = OnlineCooperativeRuntime(
        config, split="train", master_factory=master_factory, coordinator=coordinator,
        observation_builder=_snapshot, policy_factory=lambda: None, snapshot_root="runtime-test-snapshots",
    )
    val = OnlineCooperativeRuntime(
        config, split="val", master_factory=master_factory, coordinator=coordinator,
        observation_builder=_snapshot, policy_factory=lambda: None, snapshot_root="runtime-test-snapshots",
    )

    # This is a lifecycle test, not a Ray-materialization test.  Create the
    # fixed pools directly so local synchronous test doubles follow the same
    # ownership boundary as the remote actors.
    train_city_slots = {"jinan": 0, "hangzhou": 0}
    for spec in iter_online_samples(config, "train", count=4):
        city_slot = train_city_slots[spec.city] % 2
        train_city_slots[spec.city] += 1
        train.streams.get_or_create_lane(
            spec, f"train:{spec.city}:slot:{city_slot:02d}"
        )
    for spec in iter_online_samples(config, "val", count=10):
        val.streams.get_or_create_lane(
            spec, f"val:{spec.city}:lane:{spec.ordinal // 2:02d}"
        )
    assert len(train.streams._streams) == 4
    assert len(val.streams._streams) == 10

    train_masters = [stream.master_actor for stream in train.streams._streams.values()]
    val_masters = [stream.master_actor for stream in val.streams._streams.values()]
    val.reset()

    assert len(val.streams._streams) == 0
    assert len(train.streams._streams) == 4
    assert all(master.closed for master in val_masters)
    assert not any(master.closed for master in train_masters)
    train.close()


def test_train_masters_suspend_for_validation_and_resume_from_saved_state(tmp_path):
    config = OnlineConfig(
        cities={"jinan": CityConfig("jinan"), "hangzhou": CityConfig("hangzhou")},
        train=SplitConfig(("jinan", "hangzhou"), batch_size=4, seed_base=11),
        val=SplitConfig(("jinan", "hangzhou"), batch_size=10, seed_base=22),
        warmup_steps=0,
    )
    created: list[_Master] = []

    def master_factory(city: str, seed: int, actor_id: str) -> _Master:
        master = _Master(city, seed, actor_id)
        created.append(master)
        return master

    runtime = OnlineCooperativeRuntime(
        config, split="train", master_factory=master_factory, coordinator=_Coordinator(),
        observation_builder=_snapshot, policy_factory=lambda: None, snapshot_root=tmp_path,
    )
    for spec in iter_online_samples(config, "train", count=4):
        stream = runtime.streams.get_or_create(spec)
        stream.advance_v25(3)
        stream.accepted_snapshots = 7
    before = {
        key: (stream.master_actor, stream.current_time(), dict(stream.master_actor.signals), stream.accepted_snapshots)
        for key, stream in runtime.streams._streams.items()
    }

    assert runtime.suspend_for_validation() == 4
    assert not runtime.streams._streams
    assert all(actor.closed for actor, *_ in before.values())
    assert len(runtime.streams._suspended_streams) == 4

    assert runtime.resume_after_validation() == 4
    assert len(runtime.streams._streams) == 4
    for key, stream in runtime.streams._streams.items():
        old_actor, saved_time, saved_signals, saved_accepted = before[key]
        assert stream.master_actor is not old_actor
        assert stream.current_time() == saved_time
        assert stream.master_actor.signals == saved_signals
        assert stream.accepted_snapshots == saved_accepted
    runtime.close()


def test_train_rows_cycle_through_fixed_master_pool():
    config = OnlineConfig(
        cities={"jinan": CityConfig("jinan"), "hangzhou": CityConfig("hangzhou")},
        train=SplitConfig(
            ("jinan", "hangzhou"), batch_size=1000, seed_base=11,
            stream_prefix="train", persistent_stream_count=8,
        ),
        val=SplitConfig(("jinan", "hangzhou"), batch_size=10, seed_base=22),
    )
    specs = list(iter_online_samples(config, "train"))
    assert len({spec.stream_id for spec in specs}) == 8
    assert len({spec.sample_id for spec in specs}) == 1000
    assert [spec.stream_id for spec in specs[:8]] == [
        f"train:slot:{index:04d}" for index in range(8)
    ]
    assert specs[8].stream_id == "train:slot:0000"
    assert specs[8].city == specs[0].city


def test_four_master_train_layout_canonicalizes_old_bootstrap_stream_ids(monkeypatch):
    config = OnlineConfig(
        cities={"jinan": CityConfig("jinan"), "hangzhou": CityConfig("hangzhou")},
        train=SplitConfig(
            ("jinan", "hangzhou"), batch_size=4, seed_base=11,
            stream_prefix="train", persistent_stream_count=4,
        ),
        val=SplitConfig(("jinan", "hangzhou"), batch_size=10, seed_base=22),
        warmup_steps=0,
    )
    masters: list[_Master] = []

    def master_factory(city: str, seed: int, actor_id: str) -> _Master:
        master = _Master(city, seed, actor_id)
        masters.append(master)
        return master

    runtime = OnlineCooperativeRuntime(
        config,
        split="train",
        master_factory=master_factory,
        coordinator=_Coordinator(),
        observation_builder=_snapshot,
        policy_factory=lambda: None,
        snapshot_root="runtime-test-snapshots",
    )
    monkeypatch.setenv("V35_TRAIN_MATERIALIZE_CONCURRENCY", "4")
    # This reproduces the legacy JSONL that used one global train:slot ID per
    # row.  The fourth wave used to instantiate train:slot:0016..0019.
    specs = [
        OnlineSampleSpec(
            data_source="v35_online_sumo",
            city=("jinan" if ordinal % 2 == 0 else "hangzhou"),
            stream_id=f"train:slot:{ordinal:04d}",
            episode_id=0,
            seed=11 + ordinal,
            ordinal=ordinal,
        )
        for ordinal in range(20)
    ]

    for offset in range(0, len(specs), 4):
        runtime.collect_batch_specs(specs[offset : offset + 4])

    assert {master.actor_id for master in masters} == {
        "train_master_train_jinan_slot_00",
        "train_master_train_jinan_slot_01",
        "train_master_train_hangzhou_slot_00",
        "train_master_train_hangzhou_slot_01",
    }
    assert len(masters) == 4
    runtime.close()


def test_collect_batch_specs_submits_ray_masters_before_ordered_get(monkeypatch):
    config = OnlineConfig(
        cities={"jinan": CityConfig("jinan")},
        train=SplitConfig(("jinan",), batch_size=4, seed_base=11),
        val=SplitConfig(("jinan",), batch_size=1, seed_base=22),
        warmup_steps=0,
        resources=ResourceConfig(max_parallel_masters=4),
    )
    submitted: list[str] = []
    wait_sizes: list[int] = []

    class Ref:
        def __init__(self, value):
            self.value = value

    class RemoteMethod:
        def __init__(self, master):
            self.master = master

        def remote(self, **kwargs):
            submitted.append(self.master.actor_id)
            step = int(kwargs["warmup_steps"]) + 1 + int(kwargs["initial_step_offset"])
            from pathlib import Path
            snapshot_path = Path(kwargs["snapshot_path"])
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            snapshot_path.write_text("<snapshot />", encoding="utf-8")
            return Ref({
                "step": step,
                "restarted": False,
                "state": {"actor_id": self.master.actor_id},
                "video_details": {"actor_id": self.master.actor_id},
                "snapshot_path": str(snapshot_path),
            })

    class RayMaster(_Master):
        def __init__(self, city: str, seed: int, actor_id: str) -> None:
            super().__init__(city, seed, actor_id)
            self.materialize_snapshot_state = RemoteMethod(self)

    def ray_wait(refs, num_returns, timeout=None):
        del timeout
        wait_sizes.append(len(refs))
        return refs[:num_returns], refs[num_returns:]

    def ray_get(ref):
        if isinstance(ref, list):
            return [item.value for item in ref]
        return ref.value

    monkeypatch.setitem(sys.modules, "ray", SimpleNamespace(get=ray_get, wait=ray_wait))

    def observation_builder(source, city: str, step: int) -> CitySnapshot:
        snapshot = _snapshot(source, city, step)
        return CitySnapshot(
            city=snapshot.city,
            step=snapshot.step,
            observations=snapshot.observations,
            simulator_snapshot=source["actor_id"],
            required_neighbors=snapshot.required_neighbors,
        )

    runtime = OnlineCooperativeRuntime(
        config,
        split="train",
        master_factory=RayMaster,
        coordinator=_Coordinator(),
        observation_builder=observation_builder,
        policy_factory=lambda: None,
        snapshot_root="runtime-test-snapshots",
    )
    specs = list(iter_online_samples(config, "train", count=4))
    items = runtime.collect_batch_specs(specs)

    assert submitted == [f"train_master_online_slot_{index:04d}" for index in range(4)]
    assert wait_sizes == [2, 2]
    assert [item.spec.sample_id for item in items] == [spec.sample_id for spec in specs]
    assert [item.snapshot.simulator_snapshot for item in items] == [
        f"train_master_online_slot_{index:04d}" for index in range(4)
    ]
    runtime.close()
