"""Ray process-level SUMO rollouts for Stage 2 cooperative GRPO.

The master simulator is only used to materialize a same-time snapshot.  Each
Ray actor then owns one independent SUMO process and one private copy of that
snapshot.  No SUMO or TraCI state is shared between rollout candidates.
"""

from __future__ import annotations

import shutil
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .mode_assignment import build_mode_assignment, validate_mode_assignment
from .observation_builder import build_city_snapshot, collect_observation_state
from .online_rollout import (
    CityRolloutResult,
    CitySnapshot,
    IntersectionRolloutResult,
    SimulatorAdapter,
    rollout_one_assignment,
    validate_city_snapshot,
)
from .online_reward import compute_endpoint_scores, compute_online_reward, mode_utility
from .stage2_protocol import build_stage2_prompt, executable_signal, parse_decision_response

try:  # Ray is optional for CPU contract tests and local development.
    import ray
except ImportError:  # pragma: no cover - exercised only without Ray installed
    ray = None  # type: ignore[assignment]


def _dbg(message: str) -> None:
    if os.environ.get("V35_VERBOSE_DEBUG", "0") == "1":
        print(f"[ROLLOUT_DEBUG t={time.monotonic():.3f} pid={os.getpid()}] {message}", flush=True)


def _copy_snapshot(source: str | Path, target: str | Path) -> str:
    """Copy one master snapshot to an actor-private path on shared storage."""
    source_path = Path(source)
    target_path = Path(target)
    if not source_path.is_file():
        raise FileNotFoundError(f"SUMO snapshot does not exist: {source_path}")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_path)
    return str(target_path)


def _rollout_actor_id(snapshot_path: str | Path, city: str, rollout_id: int) -> str:
    """Build a work-directory-safe ID unique across concurrent batch slots."""
    path = Path(snapshot_path)
    # Coordinator paths are .../<sample_id>/step_<n>/rollout_<id>.xml. Keep
    # both sample and step so two same-city slots never share SUMO files.
    parts = [part for part in path.parts[-4:-1] if part]
    raw = "_".join([str(city), *parts, f"r{int(rollout_id):02d}"])
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in raw)


def _seed_temporal_age_tracker(simulator: Any, snapshot: CitySnapshot) -> None:
    """Restore controller-side age history omitted by SUMO XML snapshots."""
    tracker: dict[str, dict[str, int]] = {}
    for row in snapshot.observations:
        audit = row.audit_metadata if isinstance(row.audit_metadata, Mapping) else {}
        ages = audit.get("sumo_age_by_phase")
        if not isinstance(ages, Mapping):
            phases = (
                row.local_perception.get("phases", {})
                if isinstance(row.local_perception, Mapping)
                else {}
            )
            ages = {
                phase: (phases.get(phase, {}) or {}).get("age", 0)
                for phase in ("ETWT", "NTST", "ELWL", "NLSL")
            }
        tracker[row.intersection_id] = {
            phase: int(ages.get(phase, 0) or 0)
            for phase in ("ETWT", "NTST", "ELWL", "NLSL")
        }
        tracker[row.intersection_id]["__step"] = int(snapshot.step)
    setattr(simulator, "_v35_age_tracker", tracker)


@dataclass(frozen=True)
class PreparedCityRollout:
    """A synchronized SUMO snapshot ready for policy responses.

    Prompt generation is intentionally performed by VERL's own rollout
    manager on the training workers.  This object carries only the simulator
    state and the fixed fast/slow assignment into the later SUMO evaluation.
    """

    snapshot: CitySnapshot
    # The prepared rollout is file-backed.  Keeping this optional prevents
    # later Stage 2 code from accidentally issuing RPCs on the source master.
    master_actor: Any | None
    snapshot_dir: str
    snapshot_paths: tuple[str, ...]
    assignments: tuple[Mapping[str, str | None], ...]


@dataclass
class TemporalBatchSession:
    prepared: Sequence[PreparedCityRollout]
    actors: list[list[Any]]
    snapshots: list[list[CitySnapshot]]


def materialize_rollout_snapshots(
    master_simulator: Any,
    snapshot_dir: str | Path,
    *,
    num_rollouts: int = 6,
    master_name: str = "master_snapshot.xml",
) -> list[str]:
    """Save one t0 master snapshot and create one copy per rollout.

    ``master_simulator`` must be positioned at the desired city/step and
    expose ``save_snapshot``.  The returned paths must be visible to all Ray
    workers (for a multi-node Ray cluster, use a shared filesystem).
    """
    if num_rollouts <= 0:
        raise ValueError("num_rollouts must be positive")
    root = Path(snapshot_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    saved = master_simulator.save_snapshot(root / master_name)
    source = Path(saved)
    paths: list[str] = []
    for rollout_id in range(num_rollouts):
        suffix = source.suffix or ".xml"
        target = root / f"rollout_{rollout_id:02d}{suffix}"
        paths.append(_copy_snapshot(source, target))
    return paths


if ray is not None:

    @ray.remote(num_cpus=1)
    class SUMOMasterActor:
        """Ray-owned master SUMO process for one city sample.

        The master is positioned once by the caller/training loop.  It only
        writes the t0 snapshot while rollout actors run, and must not advance
        during candidate evaluation.
        """

        def __init__(
            self,
            env_factory: Callable[..., SimulatorAdapter],
            city: str,
            seed: int,
            actor_id: str = "master",
            egl_device_index: int = 0,
        ) -> None:
            # p3headlessgl enumerates EGL devices independently of CUDA.
            # Select the device before env_factory initializes ShowBase.
            self.actor_id = str(actor_id)
            self._master_phase = "initializing"
            self._master_phase_started = time.monotonic()
            os.environ["PANDA3D_EGL_DEVICE_INDEX"] = str(int(egl_device_index))
            print(
                f"[RENDER_GPU_ASSIGNMENT] actor={actor_id} city={city} "
                f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', '')} "
                f"egl_device_index={int(egl_device_index)}",
                flush=True,
            )
            self.simulator = env_factory(str(city), int(seed), actor_id=str(actor_id))
            self._master_phase = "idle"
            self._master_phase_started = time.monotonic()

        def debug_status(self) -> dict[str, Any]:
            """Return lightweight state for diagnosing a stuck RPC."""
            return {
                "actor_id": self.actor_id,
                "pid": os.getpid(),
                "phase": self._master_phase,
                "phase_elapsed_s": time.monotonic() - self._master_phase_started,
            }

        def save_snapshot(self, path: str) -> str:
            saver = getattr(self.simulator, "save_snapshot", None)
            if not callable(saver):
                raise TypeError("master simulator must expose save_snapshot")
            return str(saver(path))

        def restore_snapshot(self, path: str) -> None:
            restorer = getattr(self.simulator, "restore", None)
            if not callable(restorer):
                raise TypeError("master simulator must expose restore")
            restorer(str(path))

        def export_controller_history(self) -> dict[str, Any]:
            exporter = getattr(self.simulator, "export_controller_history", None)
            return dict(exporter()) if callable(exporter) else {}

        def restore_controller_history(self, state: Mapping[str, Any]) -> None:
            restorer = getattr(self.simulator, "restore_controller_history", None)
            if callable(restorer):
                restorer(dict(state))

        def reset(self, *, seed: int | None = None, use_gui: bool = False) -> None:
            resetter = getattr(self.simulator, "env", self.simulator)
            reset = getattr(resetter, "reset", None)
            if not callable(reset):
                raise TypeError("master simulator must expose reset")
            kwargs = {"use_gui": use_gui}
            if seed is not None:
                kwargs["seed"] = int(seed)
            try:
                reset(**kwargs, verbose=False)
            except TypeError:
                reset(**kwargs)
            reset_history = getattr(self.simulator, "reset_v25_history", None)
            if callable(reset_history):
                reset_history()

        def apply_signals(self, signals: Mapping[str, str]) -> None:
            self.simulator.apply_signals(signals)

        def advance(self, decision_cycles: int = 3) -> Any:
            try:
                return self.simulator.advance(decision_cycles)
            finally:
                releaser = getattr(self.simulator, "release_visual_renderer", None)
                if callable(releaser):
                    releaser()

        def advance_v25(self, decision_cycles: int = 1) -> Any:
            advance = getattr(self.simulator, "advance_v25", None)
            if not callable(advance):
                raise TypeError("master simulator must expose advance_v25")
            try:
                return advance(int(decision_cycles))
            finally:
                releaser = getattr(self.simulator, "release_visual_renderer", None)
                if callable(releaser):
                    releaser()

        def commit_signals(self, signals: Mapping[str, str], decision_cycles: int = 3) -> Any:
            """Commit the selected candidate to the master timeline."""
            self.simulator.apply_signals(signals)
            try:
                return self.simulator.advance(decision_cycles)
            finally:
                releaser = getattr(self.simulator, "release_visual_renderer", None)
                if callable(releaser):
                    releaser()

        def queue_metrics(self) -> Mapping[str, float]:
            return dict(self.simulator.queue_metrics())

        def current_signal_table(self) -> Mapping[str, str]:
            getter = getattr(self.simulator, "current_signal_table", None)
            if not callable(getter):
                raise TypeError("master simulator must expose current_signal_table")
            return dict(getter())

        def current_time(self) -> float:
            getter = getattr(self.simulator, "current_time", None)
            if not callable(getter):
                raise TypeError("master simulator must expose current_time")
            return float(getter())

        def observation_state(self, step: int | None = None) -> dict[str, Any]:
            """Extract a serializable same-time state inside the master actor."""
            return collect_observation_state(self.simulator, step=step)

        def materialize_snapshot_state(
            self,
            *,
            city: str,
            decision_cycles: int,
            episode_seconds: float,
            decision_cycle_seconds: float,
            warmup_steps: int,
            initial_step_offset: int,
            force_restart: bool,
            restart_seed: int,
            target_step: int | None = None,
            restore_path: str | None = None,
            snapshot_path: str | None = None,
        ) -> dict[str, Any]:
            """Atomically position one master and export its same-time state."""
            import time

            started_at = time.monotonic()
            self._master_phase = "materialize_start"
            self._master_phase_started = started_at
            print(
                f"[MASTER_PHASE] stream_actor={self.actor_id} phase=materialize_start "
                f"city={city} restore={bool(restore_path)} force_restart={bool(force_restart)}",
                flush=True,
            )
            if restore_path:
                self._master_phase = "restore"
                self._master_phase_started = time.monotonic()
                phase_started = time.monotonic()
                self.restore_snapshot(restore_path)
                print(
                    f"[MASTER_PHASE] stream_actor={self.actor_id} phase=restore_done "
                    f"elapsed_s={time.monotonic() - phase_started:.3f}",
                    flush=True,
                )
            current_time = self.current_time()
            horizon = float(decision_cycle_seconds) * int(decision_cycles)
            restarted = bool(force_restart or current_time + horizon > float(episode_seconds))
            if restarted:
                self._master_phase = "reset"
                self._master_phase_started = time.monotonic()
                phase_started = time.monotonic()
                print(
                    f"[MASTER_PHASE] stream_actor={self.actor_id} phase=reset_start "
                    f"seed={int(restart_seed)}",
                    flush=True,
                )
                self.reset(seed=int(restart_seed), use_gui=False)
                print(
                    f"[MASTER_PHASE] stream_actor={self.actor_id} phase=reset_done "
                    f"elapsed_s={time.monotonic() - phase_started:.3f}",
                    flush=True,
                )

            if target_step is None:
                target_step = int(warmup_steps) + 1 + int(initial_step_offset)
            else:
                target_step = int(target_step)
            current_step = int(round(self.current_time() / float(decision_cycle_seconds)))
            if current_step < target_step:
                self._master_phase = "advance_v25"
                self._master_phase_started = time.monotonic()
                phase_started = time.monotonic()
                print(
                    f"[MASTER_PHASE] stream_actor={self.actor_id} phase=advance_start "
                    f"from_step={current_step} to_step={target_step}",
                    flush=True,
                )
                self.advance_v25(target_step - current_step)
                print(
                    f"[MASTER_PHASE] stream_actor={self.actor_id} phase=advance_done "
                    f"elapsed_s={time.monotonic() - phase_started:.3f}",
                    flush=True,
                )

            current_time = self.current_time()
            step = int(round(current_time / float(decision_cycle_seconds)))
            if step <= int(warmup_steps):
                raise RuntimeError(
                    f"warm-up incomplete: current step={step}, required > {int(warmup_steps)}"
                )
            phase_started = time.monotonic()
            print(
                f"[MASTER_PHASE] stream_actor={self.actor_id} phase=state_extract_start step={step}",
                flush=True,
            )
            state = collect_observation_state(self.simulator, step=step)
            self._master_phase = "video_details"
            self._master_phase_started = time.monotonic()
            print(
                f"[MASTER_PHASE] stream_actor={self.actor_id} phase=state_extract_done "
                f"elapsed_s={time.monotonic() - phase_started:.3f}",
                flush=True,
            )
            getter = getattr(self.simulator, "latest_video_details", None)
            phase_started = time.monotonic()
            video_details = dict(getter()) if callable(getter) else {}
            print(
                f"[MASTER_PHASE] stream_actor={self.actor_id} phase=video_details_done "
                f"elapsed_s={time.monotonic() - phase_started:.3f}",
                flush=True,
            )
            exported_snapshot_path = None
            if snapshot_path:
                self._master_phase = "snapshot_export"
                self._master_phase_started = time.monotonic()
                phase_started = time.monotonic()
                print(
                    f"[MASTER_PHASE] stream_actor={self.actor_id} phase=snapshot_export_start "
                    f"path={snapshot_path}",
                    flush=True,
                )
                exported_snapshot_path = self.save_snapshot(str(snapshot_path))
                print(
                    f"[MASTER_PHASE] stream_actor={self.actor_id} phase=snapshot_export_done "
                    f"elapsed_s={time.monotonic() - phase_started:.3f}",
                    flush=True,
                )
            print(
                f"[MASTER_PHASE] stream_actor={self.actor_id} phase=materialize_done "
                f"step={step} total_elapsed_s={time.monotonic() - started_at:.3f}",
                flush=True,
            )
            self._master_phase = "idle"
            self._master_phase_started = time.monotonic()
            return {
                "city": str(city),
                "step": step,
                "current_time": current_time,
                "restarted": restarted,
                "state": state,
                "video_details": video_details,
                "snapshot_path": exported_snapshot_path,
            }

        def latest_video_details(self) -> dict[str, Any]:
            getter = getattr(self.simulator, "latest_video_details", None)
            if not callable(getter):
                raise TypeError("master simulator must expose latest_video_details")
            return dict(getter())

        def ensure_current_video_details(self) -> dict[str, Any]:
            getter = getattr(self.simulator, "ensure_current_video_details", None)
            if not callable(getter):
                return self.latest_video_details()
            return dict(getter())

        def export_video_details(self) -> dict[str, Any]:
            getter = getattr(self.simulator, "export_video_details", None)
            return dict(getter()) if callable(getter) else self.latest_video_details()

        def restore_video_details(self, details: Mapping[str, Any]) -> None:
            setter = getattr(self.simulator, "restore_video_details", None)
            if callable(setter):
                setter(dict(details or {}))

        def close(self) -> None:
            closer = getattr(self.simulator, "close", None)
            if callable(closer):
                closer()

    @ray.remote(num_cpus=0.25)
    class SUMORolloutActor:
        """One Ray process containing one actor-local SUMO environment."""

        def __init__(self, env_factory: Callable[..., SimulatorAdapter], policy_factory: Callable[[], Callable[..., Any]] | None) -> None:
            self.env_factory = env_factory
            self.policy_factory = policy_factory
            self.simulator = None
            self.temporal = None

        def start_temporal(
            self, snapshot: CitySnapshot, modes: Mapping[str, str | None], snapshot_path: str,
            *, rollout_id: int, seed: int | None = None, tokenizer: Any = None,
            reward_kwargs: Mapping[str, Any] | None = None,
        ) -> Mapping[str, float]:
            if self.simulator is not None:
                raise RuntimeError("temporal rollout is already active")
            self.simulator = self.env_factory(
                snapshot.city, int(seed if seed is not None else rollout_id),
                actor_id=_rollout_actor_id(snapshot_path, snapshot.city, rollout_id),
            )
            self.simulator.restore(snapshot_path)
            _seed_temporal_age_tracker(self.simulator, snapshot)
            before = {str(k): float(v) for k, v in self.simulator.queue_metrics().items()}
            self.temporal = {
                "snapshot": snapshot, "modes": dict(modes), "rollout_id": int(rollout_id),
                "tokenizer": tokenizer, "reward_kwargs": dict(reward_kwargs or {}),
                "first_before": before, "first_after": {}, "final_after": before,
                "first_rows": {}, "cycle_results": [], "cumulative": {
                    row.intersection_id: {"global_queue_reward": 0.0, "local_queue_reward": 0.0,
                        "reasoning_cost_reward": 0.0, "format_penalty": 0.0, "score": 0.0,
                        "cycles": 0}
                    for row in snapshot.observations
                },
            }
            return before

        def temporal_snapshot(self, step: int) -> CitySnapshot:
            if self.simulator is None or self.temporal is None:
                raise RuntimeError("temporal rollout is not active")
            base = self.temporal["snapshot"]
            return build_city_snapshot(self.simulator, base.city, int(step))

        def apply_temporal_cycle(
            self, snapshot: CitySnapshot, responses: Mapping[str, str], *, cycle: int,
            advance: bool,
        ) -> CitySnapshot | None:
            if self.simulator is None or self.temporal is None:
                raise RuntimeError("temporal rollout is not active")
            state = self.temporal
            modes = state["modes"]
            by_id = {row.intersection_id: row for row in snapshot.observations}
            if set(responses) != set(by_id):
                raise ValueError("temporal responses must cover every city intersection")
            before = {str(k): float(v) for k, v in self.simulator.queue_metrics().items()}
            prompts, parsed, signals = {}, {}, {}
            for intersection_id, row in by_id.items():
                # Only t0 is a forced exploration candidate. Future decisions
                # follow the deployed policy with its naturally generated mode.
                mode = modes[intersection_id] if cycle == 0 else None
                prompts[intersection_id] = build_stage2_prompt(
                    row.local_perception, row.cooperative_perception, forced_mode=mode
                )
                parsed[intersection_id] = parse_decision_response(
                    str(responses[intersection_id]), forced_mode=mode
                )
                signals[intersection_id] = executable_signal(
                    parsed[intersection_id], row.current_phase
                )
            self.simulator.apply_signals(signals)
            if advance:
                self.simulator.advance(1)
            after = {str(k): float(v) for k, v in self.simulator.queue_metrics().items()}
            if cycle == 0:
                state["first_after"] = after
            state["final_after"] = after
            for intersection_id in by_id:
                if cycle == 0:
                    reward = compute_online_reward(
                        parsed[intersection_id], before_queues=before, after_queues=after,
                        target_id=intersection_id, tokenizer=state["tokenizer"],
                        forced_mode=modes[intersection_id], **state["reward_kwargs"],
                    )
                    cumulative = state["cumulative"][intersection_id]
                    for key in ("global_queue_reward", "local_queue_reward", "reasoning_cost_reward", "format_penalty", "score"):
                        cumulative[key] = float(reward[key])
                    cumulative["cycles"] = 1
                    row = IntersectionRolloutResult(
                        intersection_id, modes[intersection_id], prompts[intersection_id],
                        str(responses[intersection_id]), parsed[intersection_id], reward,
                        local_perception=by_id[intersection_id].local_perception,
                        cooperative_perception=by_id[intersection_id].cooperative_perception,
                        perception_audit=dict(by_id[intersection_id].audit_metadata),
                    )
                    state["first_rows"][intersection_id] = row
                # Future cycles are environment continuation only. Retain the
                # applied signal for auditing, but do not attach token rewards.
                state["cycle_results"].append(
                    {
                        "cycle": int(cycle),
                        "intersection_id": intersection_id,
                        "signal": signals[intersection_id],
                        "local_perception": by_id[intersection_id].local_perception,
                        "cooperative_perception": by_id[intersection_id].cooperative_perception,
                        "perception_audit": dict(by_id[intersection_id].audit_metadata),
                    }
                )
            if not advance:
                return None
            return build_city_snapshot(self.simulator, snapshot.city, snapshot.step + 1)

        def finish_temporal(self) -> CityRolloutResult:
            if self.simulator is None or self.temporal is None:
                raise RuntimeError("temporal rollout is not active")
            state = self.temporal
            ids = list(state["first_rows"])
            for intersection_id, row in state["first_rows"].items():
                global_score, local_score = compute_endpoint_scores(
                    state["first_before"], state["final_after"], intersection_id
                )
                local_t1 = compute_endpoint_scores(
                    state["first_before"], state["first_after"], intersection_id
                )[1]
                team_local = sum(
                    compute_endpoint_scores(state["first_before"], state["final_after"], other)[1]
                    for other in ids
                ) / len(ids)
                penalty = float(row.reward.get("reasoning_penalty", 0.0))
                state["cumulative"][intersection_id].update({
                    "global_score": global_score, "local_score": local_t1,
                    "local_long_term_score": local_score,
                    "local_mean_score": team_local, "network_reward": global_score,
                    "mode_utility": mode_utility(local_t1, global_score, penalty), "train_cycle": 0,
                })
                row.reward.update(state["cumulative"][intersection_id])
            base = state["snapshot"]
            result = CityRolloutResult(
                base.city, base.step, state["rollout_id"], list(state["first_rows"].values()),
                state["first_before"], state["final_after"], state["cycle_results"], state["cumulative"],
            )
            closer = getattr(self.simulator, "close", None)
            if callable(closer):
                closer()
            self.simulator = None
            self.temporal = None
            return result

        def run_rollout(
            self,
            snapshot: CitySnapshot,
            modes: Mapping[str, str],
            snapshot_path: str,
            *,
            rollout_id: int,
            responses: Mapping[str, str] | None = None,
            prompt_template: str | None = None,
            decision_cycles: int = 3,
            concurrency: int | None = None,
            seed: int | None = None,
            tokenizer: Any = None,
            global_weight: float = 0.5,
            local_weight: float = 1.0,
            reasoning_weight: float = 0.5,
            reasoning_free_tokens: int = 0,
            queue_scale: float = 1.0,
        ) -> CityRolloutResult:
            _dbg(f"run_rollout start city={snapshot.city} rollout_id={rollout_id}")
            simulator = self.env_factory(
                snapshot.city,
                int(seed if seed is not None else rollout_id),
                actor_id=_rollout_actor_id(snapshot_path, snapshot.city, rollout_id),
            )
            try:
                _dbg("simulator ready; restoring snapshot")
                simulator.restore(snapshot_path)
                _dbg("snapshot restored; entering rollout")
                if responses is not None:
                    response_table = {str(key): str(value) for key, value in responses.items()}

                    def policy_fn(intersection_id: str, prompt: str, assistant_prefix: str = "") -> str:
                        del prompt, assistant_prefix
                        if intersection_id not in response_table:
                            raise KeyError(
                                f"missing in-process policy response for {intersection_id!r}"
                            )
                        return response_table[intersection_id]
                else:
                    policy_fn = self.policy_factory()
                result = rollout_one_assignment(
                    snapshot,
                    simulator,
                    policy_fn,
                    modes,
                    rollout_id=rollout_id,
                    prompt_template=prompt_template,
                    decision_cycles=decision_cycles,
                    concurrency=concurrency,
                    tokenizer=tokenizer,
                    global_weight=global_weight,
                    local_weight=local_weight,
                    reasoning_weight=reasoning_weight,
                    reasoning_free_tokens=reasoning_free_tokens,
                    queue_scale=queue_scale,
                )
                _dbg("rollout complete")
                return result
            finally:
                closer = getattr(simulator, "close", None)
                if callable(closer):
                    closer()

else:  # Keep imports and documentation usable on machines without Ray.
    SUMOMasterActor = None  # type: ignore[assignment,misc]
    SUMORolloutActor = None  # type: ignore[assignment,misc]


class RayCityRolloutCoordinator:
    """Launch six process-isolated SUMO candidates from one master snapshot."""

    def __init__(
        self,
        env_factory: Callable[..., SimulatorAdapter],
        *,
        num_rollouts: int = 6,
        ray_init_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if ray is None:
            raise ImportError("Ray is required for RayCityRolloutCoordinator")
        if num_rollouts <= 0:
            raise ValueError("num_rollouts must be positive")
        self.env_factory = env_factory
        self.num_rollouts = int(num_rollouts)
        self.ray_init_kwargs = dict(ray_init_kwargs or {})
        configured_devices = os.environ.get("V35_RENDER_EGL_DEVICES", "0,1,2,3")
        self.render_egl_devices = tuple(
            int(value.strip()) for value in configured_devices.split(",") if value.strip()
        )
        if not self.render_egl_devices or any(value < 0 for value in self.render_egl_devices):
            raise ValueError(
                "V35_RENDER_EGL_DEVICES must contain non-negative comma-separated EGL indices"
            )
        self._next_master_device = 0

    def create_master_actor(
        self,
        city: str,
        seed: int,
        *,
        actor_id: str = "master",
    ) -> Any:
        """Create the Ray-owned master SUMO for one city sample."""
        if ray is None:
            raise ImportError("Ray is required for RayCityRolloutCoordinator")
        if not ray.is_initialized():
            ray.init(**self.ray_init_kwargs)
        physical_device_index = self.render_egl_devices[
            self._next_master_device % len(self.render_egl_devices)
        ]
        self._next_master_device += 1
        resource_name = f"v35_render_gpu_{physical_device_index}"
        cluster_resources = ray.cluster_resources()
        if float(cluster_resources.get(resource_name, 0.0)) < 1.0:
            raise RuntimeError(
                f"Ray resource {resource_name!r} is not registered; initialize Ray "
                "through verl.trainer.main_ppo or register per-device render slots"
            )
        # The custom Ray resource selects the physical render GPU. CUDA device
        # visibility alone does not constrain EGL on hosts using a complete
        # user-space GLVND stack: EGL still enumerates all physical devices.
        # Such hosts must receive the physical EGL index. Containers exposing
        # only one EGL device use actor-local index zero instead.
        actor_env = {
            "CUDA_VISIBLE_DEVICES": str(physical_device_index),
            "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
        }
        render_env_map = {
            "V35_RENDER_LD_LIBRARY_PATH": "LD_LIBRARY_PATH",
            "V35_RENDER_LD_PRELOAD": "LD_PRELOAD",
            "V35_RENDER_EGL_VENDOR_LIBRARY_FILENAMES": "__EGL_VENDOR_LIBRARY_FILENAMES",
            "V35_RENDER_GLX_VENDOR_LIBRARY_NAME": "__GLX_VENDOR_LIBRARY_NAME",
            "V35_RENDER_EGL_PLATFORM": "EGL_PLATFORM",
        }
        for source, target in render_env_map.items():
            value = os.environ.get(source, "").strip()
            if value:
                actor_env[target] = value
        egl_index_mode = os.environ.get("V35_RENDER_EGL_INDEX_MODE", "").strip().lower()
        if not egl_index_mode:
            egl_index_mode = "physical" if actor_env.get("LD_PRELOAD") else "local"
        if egl_index_mode not in {"physical", "local"}:
            raise ValueError("V35_RENDER_EGL_INDEX_MODE must be 'physical' or 'local'")
        actor_egl_device_index = physical_device_index if egl_index_mode == "physical" else 0
        actor_env["V35_RENDER_EGL_INDEX_MODE"] = egl_index_mode
        actor_options: dict[str, Any] = {
            "resources": {resource_name: 1.0},
            "runtime_env": {"env_vars": actor_env},
        }
        egl_device_shim = os.environ.get("V35_EGL_DEVICE_SHIM", "").strip()
        if egl_device_shim:
            if not Path(egl_device_shim).is_file():
                raise FileNotFoundError(
                    f"EGL device compatibility shim does not exist: {egl_device_shim}"
                )
            # Apply the compatibility layer only to dedicated renderer actor
            # workers. Never preload it into VERL, vLLM, or PyTorch workers.
            # Some NVIDIA GLVND stacks expose EGL_EXT_device_base without the
            # legacy EGL_EXT_device_enumeration token Panda3D checks. In that
            # case the complete GLVND preload and this shim are both required;
            # the shim must appear first so it can augment eglQueryString.
            existing_preload = actor_env.get("LD_PRELOAD", "")
            preload_parts = [part for part in existing_preload.split(":") if part]
            if egl_device_shim not in preload_parts:
                actor_env["LD_PRELOAD"] = ":".join([egl_device_shim, *preload_parts])
        return SUMOMasterActor.options(**actor_options).remote(
            self.env_factory,
            str(city),
            int(seed),
            actor_id,
            actor_egl_device_index,
        )

    def start_temporal_batch(
        self, prepared: Sequence[PreparedCityRollout], *, tokenizer: Any = None,
        seed: int | None = None, reward_kwargs: Mapping[str, Any] | None = None,
    ) -> TemporalBatchSession:
        """Restore every candidate and keep its SUMO process alive across VLM calls."""
        timeout_s = float(os.environ.get("V35_TEMPORAL_RAY_TIMEOUT_S", "300"))
        max_attempts = max(1, int(os.environ.get("V35_TEMPORAL_START_ATTEMPTS", "2")))
        actor_concurrency = max(1, int(os.environ.get("V35_TEMPORAL_ACTOR_CONCURRENCY", "5")))
        # Temporal actors remain alive across all decision cycles.  Reserving
        # the normal per-actor CPU slice for every idle actor can deadlock the
        # later startup batches (all actors are created before they start).
        # Method calls are explicitly bounded by actor_concurrency below, so
        # temporal actors use no scheduler CPU reservation by default.
        temporal_actor_cpus = max(0.0, float(os.environ.get("V35_TEMPORAL_ACTOR_CPUS", "0")))
        last_error: BaseException | None = None
        for attempt in range(1, max_attempts + 1):
            actors: list[list[Any]] = []
            refs = []
            try:
                jobs: list[tuple[int, int, Any, Any]] = []
                for sample_index, item in enumerate(prepared):
                    sample_actors = []
                    actors.append(sample_actors)
                    for rollout_id, (modes, path) in enumerate(zip(item.assignments, item.snapshot_paths)):
                        actor = SUMORolloutActor.options(num_cpus=temporal_actor_cpus).remote(
                            self.env_factory, None
                        )
                        sample_actors.append(actor)
                        jobs.append((sample_index, rollout_id, actor, item))

                # Keep sessions alive, but limit concurrent startup load.  SUMO
                # restoration is CPU/process heavy and creating every actor at
                # once can leave Ray with a permanently pending tail.
                for batch_start in range(0, len(jobs), actor_concurrency):
                    batch = jobs[batch_start:batch_start + actor_concurrency]
                    batch_refs = []
                    print(
                        f"[TEMPORAL_START_BATCH] attempt={attempt}/{max_attempts} "
                        f"batch={batch_start // actor_concurrency + 1} "
                        f"actors={len(batch)} total={len(jobs)}",
                        flush=True,
                    )
                    for sample_index, rollout_id, actor, item in batch:
                        batch_refs.append(actor.start_temporal.remote(
                            item.snapshot, item.assignments[rollout_id], item.snapshot_paths[rollout_id],
                            rollout_id=rollout_id,
                            seed=None if seed is None else seed + sample_index * len(item.assignments) + rollout_id,
                            tokenizer=tokenizer, reward_kwargs=dict(reward_kwargs or {}),
                        ))
                    ready, pending = ray.wait(batch_refs, num_returns=len(batch_refs), timeout=timeout_s)
                    if pending:
                        raise TimeoutError(
                            f"temporal actor startup timed out after {timeout_s:.0f}s "
                            f"(ready={len(ready)} pending={len(pending)} attempt={attempt}/{max_attempts})"
                        )
                    ray.get(batch_refs)
                    print(
                        f"[TEMPORAL_START_BATCH_DONE] batch={batch_start // actor_concurrency + 1} "
                        f"actors={len(batch)}",
                        flush=True,
                    )
                snapshots = [[item.snapshot for _ in item.assignments] for item in prepared]
                return TemporalBatchSession(prepared, actors, snapshots)
            except BaseException as exc:
                last_error = exc
                print(f"[TEMPORAL_RETRY] phase=start attempt={attempt}/{max_attempts} error={exc}", flush=True)
                for actor in (actor for group in actors for actor in group):
                    ray.kill(actor, no_restart=True)
        assert last_error is not None
        raise last_error

    def apply_temporal_cycle(
        self, session: TemporalBatchSession,
        response_groups: Sequence[Sequence[Mapping[str, str]]], *, cycle: int, advance: bool,
    ) -> list[list[CitySnapshot]] | None:
        refs = []
        positions = []
        for sample_index, actors in enumerate(session.actors):
            for rollout_id, actor in enumerate(actors):
                refs.append(actor.apply_temporal_cycle.remote(
                    session.snapshots[sample_index][rollout_id],
                    dict(response_groups[sample_index][rollout_id]), cycle=int(cycle), advance=bool(advance),
                ))
                positions.append((sample_index, rollout_id))
        timeout_s = float(os.environ.get("V35_TEMPORAL_RAY_TIMEOUT_S", "300"))
        ready, pending = ray.wait(refs, num_returns=len(refs), timeout=timeout_s)
        if pending:
            raise TimeoutError(
                f"temporal cycle {cycle} timed out after {timeout_s:.0f}s "
                f"(ready={len(ready)} pending={len(pending)})"
            )
        values = ray.get(refs)
        if not advance:
            return None
        snapshots = [[None for _ in group] for group in session.actors]
        for (sample_index, rollout_id), value in zip(positions, values):
            snapshots[sample_index][rollout_id] = value
        session.snapshots = snapshots
        return snapshots

    def finish_temporal_batch(self, session: TemporalBatchSession) -> list[list[CityRolloutResult]]:
        refs, positions = [], []
        for sample_index, actors in enumerate(session.actors):
            for rollout_id, actor in enumerate(actors):
                refs.append(actor.finish_temporal.remote())
                positions.append((sample_index, rollout_id))
        grouped: list[list[CityRolloutResult]] = [[] for _ in session.actors]
        try:
            timeout_s = float(os.environ.get("V35_TEMPORAL_RAY_TIMEOUT_S", "300"))
            ready, pending = ray.wait(refs, num_returns=len(refs), timeout=timeout_s)
            if pending:
                raise TimeoutError(
                    f"temporal finish timed out after {timeout_s:.0f}s "
                    f"(ready={len(ready)} pending={len(pending)})"
                )
            for (sample_index, _), result in zip(positions, ray.get(refs)):
                grouped[sample_index].append(result)
            for values in grouped:
                values.sort(key=lambda item: item.rollout_id)
            return grouped
        finally:
            for actor in (actor for group in session.actors for actor in group):
                ray.kill(actor, no_restart=True)

    @staticmethod
    def abort_temporal_batch(session: TemporalBatchSession) -> None:
        for actor in (actor for group in session.actors for actor in group):
            ray.kill(actor, no_restart=True)

    def run_city_rollouts(
        self,
        snapshot: CitySnapshot,
        master_simulator: SimulatorAdapter,
        policy_factory: Callable[[], Callable[..., Any]],
        snapshot_dir: str | Path,
        *,
        prompt_template: str | None = None,
        decision_cycles: int = 3,
        concurrency: int | None = None,
        seed: int | None = None,
        tokenizer: Any = None,
        global_weight: float = 0.5,
        local_weight: float = 1.0,
        reasoning_weight: float = 0.5,
        reasoning_free_tokens: int = 0,
        queue_scale: float = 1.0,
    ) -> list[CityRolloutResult]:
        """Evaluate all forced-mode candidates from one synchronized t0 state."""
        validate_city_snapshot(snapshot)
        assignments = build_mode_assignment(
            [row.intersection_id for row in snapshot.observations],
            num_rollouts=self.num_rollouts,
            seed=seed,
        )
        validate_mode_assignment(assignments, [row.intersection_id for row in snapshot.observations])
        snapshot_paths = materialize_rollout_snapshots(
            master_simulator,
            snapshot_dir,
            num_rollouts=self.num_rollouts,
        )
        if not ray.is_initialized():
            ray.init(**self.ray_init_kwargs)

        actors = [
            SUMORolloutActor.remote(self.env_factory, policy_factory)
            for _ in range(self.num_rollouts)
        ]
        refs = [
            actor.run_rollout.remote(
                snapshot,
                assignments[rollout_id],
                snapshot_paths[rollout_id],
                rollout_id=rollout_id,
                prompt_template=prompt_template,
                decision_cycles=decision_cycles,
                concurrency=concurrency,
                seed=None if seed is None else seed + rollout_id,
                tokenizer=tokenizer,
                global_weight=global_weight,
                local_weight=local_weight,
                reasoning_weight=reasoning_weight,
                reasoning_free_tokens=reasoning_free_tokens,
                queue_scale=queue_scale,
            )
            for rollout_id, actor in enumerate(actors)
        ]
        try:
            results = list(ray.get(refs))
        finally:
            for actor in actors:
                ray.kill(actor, no_restart=True)
        return sorted(results, key=lambda item: item.rollout_id)

    def run_city_rollouts_from_master_actor(
        self,
        snapshot: CitySnapshot,
        master_actor: Any,
        policy_factory: Callable[[], Callable[..., Any]],
        snapshot_dir: str | Path,
        *,
        prompt_template: str | None = None,
        decision_cycles: int = 3,
        concurrency: int | None = None,
        seed: int | None = None,
        tokenizer: Any = None,
        global_weight: float = 0.5,
        local_weight: float = 1.0,
        reasoning_weight: float = 0.5,
        reasoning_free_tokens: int = 0,
        queue_scale: float = 1.0,
    ) -> list[CityRolloutResult]:
        """Variant using a Ray-owned master actor for one input sample.

        The master actor writes the t0 snapshot and remains untouched until
        all six rollout actors finish.  ``snapshot_dir`` must be visible to
        both the master and rollout workers.
        """
        if ray is None:
            raise ImportError("Ray is required for RayCityRolloutCoordinator")
        validate_city_snapshot(snapshot)
        root = Path(snapshot_dir).resolve()
        root.mkdir(parents=True, exist_ok=True)
        master_path = root / "master_snapshot.xml"
        # The formal AgentLoop path prepares this file before entering the
        # coordinator.  Reuse it here as well; only legacy callers that do not
        # provide a handoff file need the compatibility RPC.
        if master_path.is_file() and master_path.stat().st_size > 0:
            source = master_path
        else:
            saved = ray.get(master_actor.save_snapshot.remote(str(master_path)))
            source = Path(saved)
        paths = [
            _copy_snapshot(source, root / f"rollout_{rollout_id:02d}{source.suffix or '.xml'}")
            for rollout_id in range(self.num_rollouts)
        ]
        if not ray.is_initialized():
            ray.init(**self.ray_init_kwargs)
        assignments = build_mode_assignment(
            [row.intersection_id for row in snapshot.observations],
            num_rollouts=self.num_rollouts,
            seed=seed,
        )
        validate_mode_assignment(assignments, [row.intersection_id for row in snapshot.observations])
        actors = [
            SUMORolloutActor.remote(self.env_factory, policy_factory)
            for _ in range(self.num_rollouts)
        ]
        refs = [
            actor.run_rollout.remote(
                snapshot,
                assignments[rollout_id],
                paths[rollout_id],
                rollout_id=rollout_id,
                prompt_template=prompt_template,
                decision_cycles=decision_cycles,
                concurrency=concurrency,
                seed=None if seed is None else seed + rollout_id,
                tokenizer=tokenizer,
                global_weight=global_weight,
                local_weight=local_weight,
                reasoning_weight=reasoning_weight,
                reasoning_free_tokens=reasoning_free_tokens,
                queue_scale=queue_scale,
            )
            for rollout_id, actor in enumerate(actors)
        ]
        try:
            results = list(ray.get(refs))
        finally:
            for actor in actors:
                ray.kill(actor, no_restart=True)
        return sorted(results, key=lambda item: item.rollout_id)

    @staticmethod
    def signals_from_result(
        result: CityRolloutResult,
        fallback_signals: Mapping[str, str] | None = None,
    ) -> dict[str, str]:
        """Extract a complete signal table for the candidate to commit.

        Invalid model output is still retained for format/GDPO rewards.  When
        selecting such a candidate, use the snapshot's current phase as a
        deterministic no-op fallback instead of crashing the master stream.
        """
        signals: dict[str, str] = {}
        for row in result.results:
            signal = row.parsed.signal if row.parsed.signal_valid else None
            if signal is None:
                if fallback_signals is None or row.intersection_id not in fallback_signals:
                    raise ValueError(
                        f"rollout {result.rollout_id} has no valid signal for {row.intersection_id}"
                    )
                signal = str(fallback_signals[row.intersection_id])
            signals[row.intersection_id] = signal
        if not signals:
            raise ValueError("cannot commit an empty rollout result")
        return signals

    def run_batch_from_master_actors(
        self,
        items: Sequence[tuple[CitySnapshot, Any, str | Path]],
        policy_factory: Callable[[], Callable[..., Any]],
        *,
        prompt_template: str | None = None,
        decision_cycles: int = 3,
        concurrency: int | None = None,
        seed: int | None = None,
        tokenizer: Any = None,
        global_weight: float = 0.5,
        local_weight: float = 1.0,
        reasoning_weight: float = 0.5,
        reasoning_free_tokens: int = 0,
        queue_scale: float = 1.0,
    ) -> list[list[CityRolloutResult]]:
        """Run several samples concurrently, six SUMO processes per sample.

        Each item is ``(snapshot, master_actor, shared_snapshot_dir)``.  A
        sample's master is never advanced here; after selecting a candidate,
        call ``master_actor.commit_signals.remote(...)`` before the next
        global time step.
        """
        prepared = self.prepare_batch_from_master_actors(items, seed=seed)
        return self.run_prepared_batch(
            prepared,
            policy_factory=policy_factory,
            prompt_template=prompt_template,
            decision_cycles=decision_cycles,
            concurrency=concurrency,
            seed=seed,
            tokenizer=tokenizer,
            global_weight=global_weight,
            local_weight=local_weight,
            reasoning_weight=reasoning_weight,
            reasoning_free_tokens=reasoning_free_tokens,
            queue_scale=queue_scale,
        )

    def prepare_batch_from_master_actors(
        self,
        items: Sequence[tuple[CitySnapshot, Any, str | Path]],
        *,
        seed: int | None = None,
    ) -> list[PreparedCityRollout]:
        """Save synchronized snapshots and mode assignments without inference."""
        if not items:
            return []
        if ray is None:
            raise ImportError("Ray is required for RayCityRolloutCoordinator")
        if not ray.is_initialized():
            ray.init(**self.ray_init_kwargs)
        prepared: list[PreparedCityRollout] = []
        for sample_id, (snapshot, master_actor, snapshot_dir) in enumerate(items):
            validate_city_snapshot(snapshot)
            ids = [row.intersection_id for row in snapshot.observations]
            assignments = build_mode_assignment(
                ids,
                num_rollouts=self.num_rollouts,
                seed=None if seed is None else seed + sample_id,
            )
            paths_dir = Path(snapshot_dir).resolve()
            paths_dir.mkdir(parents=True, exist_ok=True)
            master_path = str(paths_dir / "master_snapshot.xml")
            # collect_batch_specs already exports the synchronized master
            # state. Reuse it instead of issuing a second remote save RPC.
            # This removes a serial TraCI/write operation from every sample.
            existing = Path(master_path)
            if existing.is_file() and existing.stat().st_size > 0:
                source = existing
                _dbg(f"reuse snapshot path={source}")
            else:
                raise FileNotFoundError(
                    f"prepared snapshot missing at {existing}; refusing a second master RPC"
                )
            paths = tuple(
                _copy_snapshot(source, paths_dir / f"rollout_{rid:02d}{source.suffix or '.xml'}")
                for rid in range(self.num_rollouts)
            )
            prepared.append(
                PreparedCityRollout(
                    snapshot=snapshot,
                    master_actor=None,
                    snapshot_dir=str(paths_dir),
                    snapshot_paths=paths,
                    assignments=tuple(assignments),
                )
            )
        return prepared

    def prepare_deployment_batch_from_master_actors(
        self, items: Sequence[tuple[CitySnapshot, Any, str | Path]]
    ) -> list[PreparedCityRollout]:
        """Prepare one autonomous, non-counterfactual trajectory per master."""
        if ray is None:
            raise ImportError("Ray is required for RayCityRolloutCoordinator")
        prepared: list[PreparedCityRollout] = []
        for snapshot, master_actor, snapshot_dir in items:
            validate_city_snapshot(snapshot)
            root = Path(snapshot_dir).resolve()
            root.mkdir(parents=True, exist_ok=True)
            existing = root / "master_snapshot.xml"
            if not existing.is_file() or existing.stat().st_size <= 0:
                raise FileNotFoundError(
                    f"deployment snapshot missing at {existing}; refusing a second master RPC"
                )
            source = existing
            path = _copy_snapshot(source, root / f"deployment{source.suffix or '.xml'}")
            modes = {row.intersection_id: None for row in snapshot.observations}
            prepared.append(PreparedCityRollout(snapshot, None, str(root), (path,), (modes,)))
        return prepared

    def run_prepared_batch(
        self,
        prepared: Sequence[PreparedCityRollout],
        *,
        policy_factory: Callable[[], Callable[..., Any]] | None = None,
        response_groups: Sequence[Sequence[Mapping[str, str]]] | None = None,
        prompt_template: str | None = None,
        decision_cycles: int = 3,
        concurrency: int | None = None,
        seed: int | None = None,
        tokenizer: Any = None,
        global_weight: float = 0.5,
        local_weight: float = 1.0,
        reasoning_weight: float = 0.5,
        reasoning_free_tokens: int = 0,
        queue_scale: float = 1.0,
    ) -> list[list[CityRolloutResult]]:
        """Evaluate prepared snapshots using either HTTP or fixed VERL responses.

        ``response_groups[sample][rollout][intersection_id]`` is produced by
        the trainer-side VERL rollout manager.  Passing it here keeps all
        policy generation on the trainable replicas while SUMO remains in Ray.
        """
        if not prepared:
            return []
        if ray is None:
            raise ImportError("Ray is required for RayCityRolloutCoordinator")
        if response_groups is not None and len(response_groups) != len(prepared):
            raise ValueError("response_groups must have one entry per prepared sample")
        if response_groups is None and policy_factory is None:
            raise ValueError("provide policy_factory or response_groups")
        if not ray.is_initialized():
            ray.init(**self.ray_init_kwargs)
        _dbg(f"run_prepared_batch start samples={len(prepared)}")
        grouped: list[list[CityRolloutResult]] = [[] for _ in prepared]
        # Deployment validation has one trajectory per independent master.
        # Ten such actors fit the configured CPU pool and should run together;
        # the sequential path below is retained for six-way training groups
        # so 4 x 6 actors cannot deadlock Ray resource scheduling.
        if all(len(item.assignments) == 1 for item in prepared):
            actors: list[Any] = []
            refs: list[Any] = []
            try:
                for sample_id, item in enumerate(prepared):
                    actor = SUMORolloutActor.remote(self.env_factory, policy_factory)
                    actors.append(actor)
                    kwargs: dict[str, Any] = {
                        "rollout_id": 0,
                        "prompt_template": prompt_template,
                        "decision_cycles": decision_cycles,
                        "concurrency": concurrency,
                        "seed": None if seed is None else seed + sample_id,
                        "tokenizer": tokenizer,
                        "global_weight": global_weight,
                        "local_weight": local_weight,
                        "reasoning_weight": reasoning_weight,
                        "reasoning_free_tokens": reasoning_free_tokens,
                        "queue_scale": queue_scale,
                    }
                    if response_groups is not None:
                        kwargs["responses"] = dict(response_groups[sample_id][0])
                    refs.append(actor.run_rollout.remote(
                        item.snapshot, item.assignments[0], item.snapshot_paths[0], **kwargs
                    ))
                results = list(ray.get(refs))
                for sample_id, result in enumerate(results):
                    grouped[sample_id] = [result]
                return grouped
            finally:
                for actor in actors:
                    ray.kill(actor, no_restart=True)
        # Run one sample at a time. Creating all sample x rollout actors at
        # once can exceed the Ray CPU pool; pending actors then hold the batch
        # at ray.get forever while already-finished actors still reserve CPUs.
        for sample_id, item in enumerate(prepared):
            rollout_count = len(item.assignments)
            if rollout_count <= 0 or len(item.snapshot_paths) != rollout_count:
                raise ValueError("prepared assignments and snapshot paths must be non-empty and aligned")
            actors: list[Any] = []
            refs: list[Any] = []
            try:
                for rollout_id in range(rollout_count):
                    actor = SUMORolloutActor.remote(self.env_factory, policy_factory)
                    actors.append(actor)
                    kwargs: dict[str, Any] = {
                        "rollout_id": rollout_id,
                        "prompt_template": prompt_template,
                        "decision_cycles": decision_cycles,
                        "concurrency": concurrency,
                        "seed": None if seed is None else seed + sample_id * rollout_count + rollout_id,
                        "tokenizer": tokenizer,
                        "global_weight": global_weight,
                        "local_weight": local_weight,
                        "reasoning_weight": reasoning_weight,
                        "reasoning_free_tokens": reasoning_free_tokens,
                        "queue_scale": queue_scale,
                    }
                    if response_groups is not None:
                        if len(response_groups[sample_id]) != rollout_count:
                            raise ValueError("each response group must match its prepared rollout count")
                        kwargs["responses"] = dict(response_groups[sample_id][rollout_id])
                    refs.append(actor.run_rollout.remote(
                        item.snapshot,
                        item.assignments[rollout_id],
                        item.snapshot_paths[rollout_id],
                        **kwargs,
                    ))
                _dbg(f"ray.get waiting sample={sample_id} refs={len(refs)}")
                grouped[sample_id] = list(ray.get(refs))
                _dbg(f"ray.get complete sample={sample_id}")
            finally:
                for actor in actors:
                    ray.kill(actor, no_restart=True)
            grouped[sample_id].sort(key=lambda result: result.rollout_id)
        return grouped


__all__ = [
    "PreparedCityRollout",
    "SUMOMasterActor",
    "SUMORolloutActor",
    "RayCityRolloutCoordinator",
    "materialize_rollout_snapshots",
]
