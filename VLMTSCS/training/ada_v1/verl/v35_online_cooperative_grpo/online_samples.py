"""Fake-row scheduling and live master lifecycle for online GRPO.

The JSONL rows generated here are bootstrap records for VERL's dataloader.
They are deliberately text-only and contain no stale perception or action.
``MasterStreamManager`` owns the mutable state associated with each row and
returns a same-step, topology-complete snapshot through a caller-supplied
observation builder.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .online_config import CityConfig, OnlineConfig, SplitConfig
from .online_rollout import CitySnapshot, validate_city_snapshot


def _call_master(master_actor: Any, method: str, *args: Any, **kwargs: Any) -> Any:
    """Call either a local master double or a Ray actor handle."""
    target = getattr(master_actor, method, None)
    if target is None:
        raise AttributeError(f"master does not expose {method}()")
    remote = getattr(target, "remote", None)
    if callable(remote):
        try:
            import ray
        except ImportError as exc:  # pragma: no cover - Ray deployment only
            raise RuntimeError("Ray actor handle received but Ray is unavailable") from exc
        return ray.get(remote(*args, **kwargs))
    return target(*args, **kwargs)


@dataclass(frozen=True)
class OnlineSampleSpec:
    data_source: str
    city: str
    stream_id: str
    episode_id: int
    seed: int
    ordinal: int

    @property
    def sample_id(self) -> str:
        return f"{self.stream_id}:{self.ordinal:08d}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "data_source": self.data_source,
            "prompt": [{"role": "user", "content": "ONLINE_SUMO_SNAPSHOT"}],
            "reward_model": {"ground_truth": "online"},
            "extra_info": {
                "index": self.sample_id,
                "city": self.city,
                "stream_id": self.stream_id,
                "episode_id": self.episode_id,
                "seed": self.seed,
                "ordinal": self.ordinal,
                "online": True,
            },
        }


def balanced_city_schedule(cities: Sequence[CityConfig], batch_size: int, *, offset: int = 0) -> list[str]:
    """Return a deterministic weighted round-robin batch schedule.

    With Jinan/Hangzhou both at weight 1 and batch size 4 this is exactly
    ``[jinan, hangzhou, jinan, hangzhou]``.  Weighted integer duplication is
    avoided so adding a third city does not starve it across successive batches.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    entries = [(str(city.name), float(city.weight)) for city in cities]
    if not entries or any(weight <= 0 for _, weight in entries):
        raise ValueError("cities must have positive weights")
    # Deficit round-robin: each city accrues its weight and the largest deficit
    # is selected.  Ties preserve registration order for reproducibility.
    total = sum(weight for _, weight in entries)
    deficits = [0.0] * len(entries)
    output: list[str] = []
    for _ in range(batch_size + max(0, offset)):
        for index, (_, weight) in enumerate(entries):
            deficits[index] += weight
        chosen = max(range(len(entries)), key=lambda index: (deficits[index], -index))
        output.append(entries[chosen][0])
        deficits[chosen] -= total
    return output[offset:]


def iter_online_samples(config: OnlineConfig, split: str = "train", *, count: int | None = None) -> Iterable[OnlineSampleSpec]:
    """Yield deterministic fake rows; live time is intentionally not persisted."""
    split_config: SplitConfig = getattr(config, split)
    city_by_name = config.cities
    cities = [city_by_name[name] for name in split_config.cities]
    total = split_config.batch_size if count is None else int(count)
    if total < 0:
        raise ValueError("count cannot be negative")
    schedule = balanced_city_schedule(cities, total)
    city_seen: dict[str, int] = {}
    for ordinal, city in enumerate(schedule):
        episode_id = 0
        # A validation call owns a fixed master pool.  A city has
        # ``val_metric_steps`` streams, with lanes 00..04 first deciding at
        # steps 6, 8, 10, 12, 14.  Later dataloader batches intentionally
        # reuse these stream IDs: their masters continue by one V25 cycle
        # after the preceding batch's Stage 2 rollout.  ``ordinal`` keeps the
        # bootstrap row/sample artifact identity unique across those batches.
        if split == "val":
            seen = city_seen.get(city, 0)
            suite_width = max(1, int(config.val_metric_steps))
            lane = seen % suite_width
            stream_id = f"{split_config.stream_prefix}:{city}:stream:{lane:02d}"
            city_seen[city] = seen + 1
        else:
            pool_size = split_config.persistent_stream_count
            if pool_size == 4 and len(cities) == 2:
                # Reserve two stable slots per city.  This prevents a global
                # ordinal schedule from rebinding one SUMO master to another
                # road network when the balanced city schedule alternates.
                local_slot = city_seen.get(city, 0) % 2
                city_seen[city] = city_seen.get(city, 0) + 1
                city_index = next(index for index, value in enumerate(cities) if value.name == city)
                slot = city_index * 2 + local_slot
                stream_id = f"{split_config.stream_prefix}:{city}:slot:{local_slot:02d}"
            else:
                slot = ordinal if pool_size is None else ordinal % pool_size
                stream_id = f"{split_config.stream_prefix}:slot:{slot:04d}"
        yield OnlineSampleSpec(
            data_source="v35_online_sumo",
            city=city,
            # A row identifies a persistent batch slot. The runtime reuses
            # this master on later iterations so its SUMO time can advance.
            stream_id=stream_id,
            episode_id=episode_id,
            seed=split_config.seed_base + ordinal * split_config.seed_stride,
            ordinal=ordinal,
        )


def write_fake_jsonl(config: OnlineConfig, path: str | Path, *, split: str = "train", count: int | None = None) -> int:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = list(iter_online_samples(config, split, count=count))
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row.as_dict(), ensure_ascii=False, separators=(",", ":")) + "\n")
    return len(rows)


@dataclass
class MasterStream:
    spec: OnlineSampleSpec
    city_config: CityConfig
    master_actor: Any
    episode_seconds: float
    decision_cycle_seconds: float
    warmup_steps: int
    initial_step_offset: int = 0
    episode_id: int = 0
    accepted_snapshots: int = 0
    needs_episode_reset: bool = False
    latest_snapshot_path: str | None = None

    @property
    def step(self) -> int:
        # SUMO starts at t=0 and each decision cycle is 30 seconds.
        return int(round(self.current_time() / self.decision_cycle_seconds))

    def current_time(self) -> float:
        return float(_call_master(self.master_actor, "current_time"))

    def needs_restart(self, decision_cycles: int = 1) -> bool:
        return self.current_time() + self.decision_cycle_seconds * int(decision_cycles) > self.episode_seconds

    def restart(self) -> None:
        self.episode_id += 1
        new_seed = int(self.spec.seed + self.episode_id * 100_000)
        _call_master(self.master_actor, "reset", seed=new_seed, use_gui=False)
        self.accepted_snapshots = 0
        self.needs_episode_reset = False

    def warmup(self) -> None:
        """Advance through sparse initial traffic without producing samples."""
        target_step = self.warmup_steps + 1 + self.initial_step_offset
        if self.step < target_step:
            # SUMOEnvAdapter requires a complete signal table.  Keep every
            # intersection on its currently active phase during warm-up.
            cycles = target_step - self.step
            _call_master(self.master_actor, "advance_v25", cycles)

    def snapshot(
        self,
        observation_builder: Callable[[Any, str, int], CitySnapshot],
        *,
        decision_cycles: int = 1,
    ) -> CitySnapshot:
        if decision_cycles <= 0:
            raise ValueError("decision_cycles must be positive")
        # Reserve room for the complete candidate horizon.  A snapshot near
        # the episode boundary must restart before rollouts would cross it.
        if self.needs_episode_reset or self.needs_restart(decision_cycles):
            self.restart()
        self.warmup()
        if self.step <= self.warmup_steps:
            raise RuntimeError(f"warm-up incomplete: current step={self.step}, required > {self.warmup_steps}")
        snapshot = observation_builder(self.master_actor, self.spec.city, self.step)
        validate_city_snapshot(snapshot)
        self.accepted_snapshots += 1
        return snapshot

    def commit(self, signals: Mapping[str, str], *, decision_cycles: int) -> None:
        _call_master(self.master_actor, "commit_signals", dict(signals), int(decision_cycles))
        if self.current_time() >= self.episode_seconds:
            self.needs_episode_reset = True

    def advance_v25(self, decision_cycles: int = 1) -> None:
        _call_master(self.master_actor, "advance_v25", int(decision_cycles))
        if self.current_time() >= self.episode_seconds:
            self.needs_episode_reset = True


def current_signal_table(master_actor: Any) -> dict[str, str]:
    """Read the adapter's managed signal table without guessing a phase."""
    try:
        values = _call_master(master_actor, "current_signal_table")
    except AttributeError:
        try:
            values = _call_master(master_actor, "signal_table")
        except AttributeError:
            return {}
    return {str(key): str(value) for key, value in dict(values).items()}


class MasterStreamManager:
    """Registry for train/validation master streams.

    The manager never shares a master between train and validation.  A caller
    can limit concurrency above this class (for example with Ray placement
    groups) using ``config.resources.max_parallel_masters``.
    """

    def __init__(self, config: OnlineConfig, *, split: str, master_factory: Callable[[str, int, str], Any]) -> None:
        if split not in {"train", "val"}:
            raise ValueError("split must be train or val")
        self.config = config
        self.split = split
        self.split_config: SplitConfig = getattr(config, split)
        self.master_factory = master_factory
        self._streams: dict[str, MasterStream] = {}
        self._aliases: dict[str, str] = {}
        # A validation pass can temporarily release the Train pool's native
        # SUMO/EGL resources.  Keep the logical streams (including episode
        # counters and lane aliases) here until their actors are recreated.
        self._suspended_streams: dict[str, tuple[MasterStream, str, dict[str, Any], dict[str, Any]]] = {}

    @staticmethod
    def _terminate_master(stream: MasterStream, *, reason: str) -> None:
        """Bound cleanup so a wedged native SUMO process cannot wedge Ray.

        ``close`` is useful when SUMO remains responsive, but it is not a
        reliable synchronization point after a renderer/TraCI deadlock.  In
        that case the actor is force-killed after a short grace period; this
        also releases Ray's actor reservation and terminates its process tree.
        """
        actor = stream.master_actor
        try:
            closer = getattr(actor, "close", None)
            if closer is not None and getattr(closer, "remote", None) is not None:
                import ray

                ray.get(closer.remote(), timeout=15)
            elif closer is not None:
                closer()
        except Exception as exc:
            print(
                f"[MASTER_STREAM_CLOSE_TIMEOUT] stream={stream.spec.stream_id} "
                f"reason={reason} error={type(exc).__name__}",
                flush=True,
            )
        try:
            import ray

            ray.kill(actor, no_restart=True)
            print(
                f"[MASTER_STREAM_KILL] stream={stream.spec.stream_id} reason={reason}",
                flush=True,
            )
        except Exception:
            pass

    def get_or_create(self, spec: OnlineSampleSpec) -> MasterStream:
        key = self._aliases.get(spec.stream_id, spec.stream_id)
        stream = self._streams.get(key)
        if stream is not None:
            return stream
        city_config = self.config.cities[spec.city]
        actor_id = key.replace(":", "_").replace("/", "_")
        initial_step_offset = 0
        if self.split_config.initial_step_stride:
            try:
                lane = int(spec.stream_id.rsplit(":", 1)[1])
            except (IndexError, ValueError) as exc:
                raise ValueError(
                    "step-staggered streams must end with a numeric lane, "
                    f"got {spec.stream_id!r}"
                ) from exc
            initial_step_offset = lane * self.split_config.initial_step_stride
        # SUMOEnvFactory uses the explicit role marker to attach the visual
        # capture pipeline only to persistent masters, never rollout actors.
        master = self.master_factory(
            spec.city,
            spec.seed,
            f"{self.split}_master_{actor_id}",
        )
        stream = MasterStream(
            spec=spec,
            city_config=city_config,
            master_actor=master,
            episode_seconds=self.config.episode_seconds,
            decision_cycle_seconds=self.config.decision_cycle_seconds,
            warmup_steps=self.config.warmup_steps,
            initial_step_offset=initial_step_offset,
        )
        self._streams[key] = stream
        self._aliases[spec.stream_id] = key
        return stream

    def get_or_create_lane(self, spec: OnlineSampleSpec, lane_id: str) -> MasterStream:
        """Return a persistent lane while allowing rows to change per batch."""
        self._aliases[spec.stream_id] = lane_id
        lane_spec = OnlineSampleSpec(
            data_source=spec.data_source,
            city=spec.city,
            seed=spec.seed,
            stream_id=lane_id,
            episode_id=spec.episode_id,
            ordinal=spec.ordinal,
        )
        return self.get_or_create(lane_spec)

    def replace(self, spec: OnlineSampleSpec) -> MasterStream:
        """Replace one unhealthy Ray master without disturbing other streams."""
        key = spec.stream_id
        key = self._aliases.get(key, key)
        previous = self._streams.pop(key, None)
        self._aliases.pop(spec.stream_id, None)
        if previous is not None:
            self._terminate_master(previous, reason="replace")
        return self.get_or_create(spec)

    def release(self, spec: OnlineSampleSpec) -> None:
        """Close and remove one completed stream.

        Validation snapshots are immutable handoff artifacts; once exported,
        keeping their SUMO/EGL master alive only consumes Ray resources.
        """
        key = spec.stream_id
        key = self._aliases.get(key, key)
        stream = self._streams.pop(key, None)
        if stream is None:
            return
        print(f"[MASTER_STREAM_RELEASE] split={self.split} stream={key}", flush=True)
        self._terminate_master(stream, reason="release")

    def close(self) -> None:
        for stream in list(self._streams.values()):
            self._terminate_master(stream, reason="reset")
        self._streams.clear()
        self._aliases.clear()
        self._suspended_streams.clear()

    def suspend(self, snapshot_root: str | Path) -> int:
        """Persist and release every live stream without losing its timeline.

        Snapshot all masters before closing any of them.  This makes a failed
        export non-destructive: the active Train pool remains usable and no
        partially-paused state leaks into validation.
        """
        if self._suspended_streams:
            raise RuntimeError(f"{self.split} master pool is already suspended")
        active = list(self._streams.items())
        if not active:
            return 0
        root = Path(snapshot_root)
        snapshots: dict[str, str] = {}
        histories: dict[str, dict[str, Any]] = {}
        videos: dict[str, dict[str, Any]] = {}
        stamp = f"pause_{time.time_ns()}"
        for key, stream in active:
            target = root / self.split / "paused" / key.replace(":", "_") / f"{stamp}.xml"
            target.parent.mkdir(parents=True, exist_ok=True)
            path = str(_call_master(stream.master_actor, "save_snapshot", str(target)))
            saved = Path(path)
            if not saved.is_file() or saved.stat().st_size <= 0:
                raise RuntimeError(f"invalid pause snapshot for stream={key}: {path}")
            snapshots[key] = path
            histories[key] = dict(_call_master(stream.master_actor, "export_controller_history"))
            exporter = getattr(stream.master_actor, "export_video_details", None)
            videos[key] = dict(_call_master(stream.master_actor, "export_video_details")) if exporter is not None else {}

        # All durable exports succeeded.  Only now release native actors.
        self._streams.clear()
        self._suspended_streams = {key: (stream, snapshots[key], histories[key], videos[key]) for key, stream in active}
        for _key, stream in active:
            self._terminate_master(stream, reason="validation_suspend")
        return len(active)

    def resume(self) -> int:
        """Recreate suspended actors and restore their exact saved SUMO state."""
        suspended = list(self._suspended_streams.items())
        if not suspended:
            return 0
        recreated: dict[str, MasterStream] = {}
        try:
            for key, (stream, snapshot_path, history, videos) in suspended:
                actor_id = key.replace(":", "_").replace("/", "_")
                master = self.master_factory(
                    stream.spec.city, stream.spec.seed, f"{self.split}_master_{actor_id}"
                )
                stream.master_actor = master
                _call_master(master, "restore_snapshot", snapshot_path)
                _call_master(master, "restore_controller_history", history)
                # The exported video files remain on shared storage. Restore
                # their metadata directly; do not synthesize a one-frame
                # replacement, because Stage 1 requires six temporal frames.
                if getattr(master, "restore_video_details", None) is not None:
                    _call_master(master, "restore_video_details", videos)
                recreated[key] = stream
        except Exception:
            for stream in recreated.values():
                self._terminate_master(stream, reason="validation_resume_failed")
            # Retain the saved metadata/snapshots so a later cleanup or retry
            # cannot silently turn the train timeline into a fresh episode.
            raise
        self._streams.update(recreated)
        self._suspended_streams.clear()
        return len(recreated)


__all__ = [
    "OnlineSampleSpec",
    "balanced_city_schedule",
    "iter_online_samples",
    "write_fake_jsonl",
    "MasterStream",
    "MasterStreamManager",
    "current_signal_table",
]
