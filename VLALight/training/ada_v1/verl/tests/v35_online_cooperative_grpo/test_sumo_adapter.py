from __future__ import annotations

import pytest

from v35_online_cooperative_grpo.sumo_adapter import (
    SUMOEnvAdapter,
    phase_name_to_action,
)


class _Intersection:
    def __init__(self, inter_id: str, phases: list[str], queues) -> None:
        self.inter_id = inter_id
        self.control_phases = phases
        self.dic_feature = {"lane_num_waiting_vehicle_in": queues}


class _FakeEnv:
    def __init__(self) -> None:
        self.list_intersection = [
            _Intersection("a", ["ETWT", "NTST"], [1, 2]),
            _Intersection("b", ["NLSL", "ELWL"], {"lane": 3}),
        ]
        self.calls: list[tuple[dict[str, int], float]] = []
        self.loaded = None

    def step(self, actions, min_action_time):
        self.calls.append((dict(actions), min_action_time))
        return (None, 0.0, False, {})

    def load_from_file(self, path, **kwargs):
        self.loaded = (path, kwargs)


def test_phase_mapping_and_queue_aggregation():
    env = _FakeEnv()
    assert phase_name_to_action(env, {"a": "NTST", "b": "NLSL"}) == {"a": 1, "b": 0}
    adapter = SUMOEnvAdapter(env)
    assert adapter.intersection_ids == ("a", "b")
    assert adapter.queue_metrics() == {"a": 3.0, "b": 3.0}


def test_advance_submits_one_synchronized_action_table():
    env = _FakeEnv()
    adapter = SUMOEnvAdapter(env, decision_cycle_seconds=30)
    adapter.apply_signals({"a": "NTST", "b": "NLSL"})
    adapter.advance(3)
    assert env.calls == [({"a": 1, "b": 0}, 90.0)]


def test_adapter_rejects_incomplete_or_invalid_tables():
    env = _FakeEnv()
    adapter = SUMOEnvAdapter(env)
    with pytest.raises(KeyError):
        adapter.apply_signals({"a": "NTST"})
    with pytest.raises(ValueError):
        adapter.apply_signals({"a": "BAD", "b": "NLSL"})
    with pytest.raises(RuntimeError):
        adapter.advance(1)


def test_restore_forwards_snapshot_with_strict_errors():
    env = _FakeEnv()
    adapter = SUMOEnvAdapter(env)
    adapter.restore("state.xml")
    assert env.loaded == ("state.xml", {"quiet": True, "raise_on_error": True})


def test_visual_renderer_can_be_released_without_losing_sumo_or_video_state():
    class VisualEnv(_FakeEnv):
        def __init__(self):
            super().__init__()
            self.time = 0.0

        def get_current_time(self):
            return self.time

        def step(self, actions, min_action_time, inner_step_callback=None):
            self.calls.append((dict(actions), min_action_time))
            self.time += min_action_time
            return (None, 0.0, False, {})

    class Capture:
        def __init__(self):
            self._decision_step = 0
            self.latest_video_details = {}
            self.closed = False
            self.recorder = type("Recorder", (), {})()

        def begin(self):
            self._decision_step += 1

        def callback(self, inner_i, env):
            return None

        def finish(self):
            self.latest_video_details = {"step": self._decision_step}
            return self.latest_video_details

        def close(self):
            self.closed = True

    env = VisualEnv()
    captures = []

    def factory():
        capture = Capture()
        captures.append(capture)
        return capture

    adapter = SUMOEnvAdapter(env, visual_capture_factory=factory)
    for expected_step in (1, 2):
        adapter.apply_signals({"a": "NTST", "b": "NLSL"})
        adapter.advance(1)
        adapter.release_visual_renderer()
        assert captures[-1].closed
        assert adapter.visual_capture is None
        assert adapter.latest_video_details() == {"step": expected_step}

    assert len(captures) == 2
    assert env.time == 60.0


def test_v25_warmup_updates_age_every_cycle_and_reset_clears_it(monkeypatch):
    import v35_online_cooperative_grpo.observation_builder as observation_builder

    env = _FakeEnv()
    env.time = 0.0
    env.get_current_time = lambda: env.time
    adapter = SUMOEnvAdapter(env, decision_cycle_seconds=30)
    monkeypatch.setattr(
        adapter, "v25_signal_table", lambda: {"a": "ETWT", "b": "NLSL"}
    )
    monkeypatch.setattr(adapter, "apply_signals", lambda signals: None)
    monkeypatch.setattr(
        adapter,
        "advance",
        lambda cycles: setattr(env, "time", env.time + 30.0 * cycles),
    )
    sampled_steps = []

    def collect(source, *, step=None):
        assert source is adapter
        sampled_steps.append(step)
        source._v35_age_tracker = {"sampled": {"__step": step}}

    monkeypatch.setattr(observation_builder, "collect_observation_state", collect)

    adapter.advance_v25(3)

    assert sampled_steps == [1, 2, 3]
    assert adapter._v35_age_tracker == {"sampled": {"__step": 3}}
    adapter.reset_v25_history()
    assert adapter._v35_age_tracker == {}
