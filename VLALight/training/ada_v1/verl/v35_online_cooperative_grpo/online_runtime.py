"""Runtime orchestration for online cooperative GRPO samples.

This module is intentionally below VERL's dataloader and above the existing
Ray rollout coordinator.  It turns one fake row into one mutable master, asks
an observation builder for a same-step complete city snapshot, evaluates six
citywide candidates, and commits the selected complete signal table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .online_config import OnlineConfig, SplitConfig
from .online_rollout import CityRolloutResult, CitySnapshot, flatten_for_gdpo
from .online_samples import (
    MasterStreamManager,
    OnlineSampleSpec,
    balanced_city_schedule,
    iter_online_samples,
)
from .ray_rollout import RayCityRolloutCoordinator


def _dbg(message: str) -> None:
    if os.environ.get("V35_VERBOSE_DEBUG", "0") == "1":
        print(f"[ONLINE_DEBUG t={time.monotonic():.3f} pid={os.getpid()}] {message}", flush=True)


def _safe_path_component(value: object) -> str:
    """Return a stable filename component for externally supplied stream IDs.

    Online sample IDs deliberately contain ``:`` to encode their logical
    identity (for example ``train:slot:0000:00000000``).  Keep that form in
    metadata, but never pass it through to SUMO/TraCI as part of a state-file
    path.  SUMO's option parsing has treated such paths as a port value in
    deployed builds after a save-state command.
    """
    return str(value).replace(":", "_").replace("/", "_").replace("\\", "_")


@dataclass(frozen=True)
class OnlineBatchItem:
    spec: OnlineSampleSpec
    snapshot: CitySnapshot
    master_actor: Any
    snapshot_dir: Path
    video_details: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class OnlineBatchResult:
    items: list[OnlineBatchItem]
    rollouts: list[list[CityRolloutResult]]
    gdpo_records: list[dict[str, Any]]
    selected_rollout_ids: list[int] = field(default_factory=list)


def select_best_city_rollout(results: Sequence[CityRolloutResult]) -> CityRolloutResult:
    """Select by total score across every intersection in a candidate."""
    if not results:
        raise ValueError("cannot select from an empty rollout list")

    def value(candidate: CityRolloutResult) -> float:
        return sum(float(row.reward.get("score", 0.0)) for row in candidate.results)

    # Stable tie-break: lower rollout ID wins, which keeps reproducibility when
    # queue changes are exactly equal.
    return max(results, key=lambda candidate: (value(candidate), -candidate.rollout_id))


class OnlineCooperativeRuntime:
    """Manage independent masters and synchronized city-level rollouts."""

    def __init__(
        self,
        config: OnlineConfig,
        *,
        split: str,
        master_factory: Callable[[str, int, str], Any],
        coordinator: RayCityRolloutCoordinator,
        observation_builder: Callable[[Any, str, int], CitySnapshot],
        policy_factory: Callable[[], Callable[..., Any]],
        prompt_template: str | None = None,
        snapshot_root: str | Path | None = None,
        select_candidate: Callable[[Sequence[CityRolloutResult]], CityRolloutResult] = select_best_city_rollout,
    ) -> None:
        if split not in {"train", "val"}:
            raise ValueError("split must be train or val")
        self.config = config
        self.split = split
        self.split_config: SplitConfig = getattr(config, split)
        self.streams = MasterStreamManager(config, split=split, master_factory=master_factory)
        self.coordinator = coordinator
        self.observation_builder = observation_builder
        self.policy_factory = policy_factory
        self.prompt_template = prompt_template
        self.snapshot_root = Path(snapshot_root or config.resources.snapshot_root)
        self.select_candidate = select_candidate
        # Fixed dataloader slots own persistent SUMO masters. A second call
        # for the same slot reads the next snapshot from that master.
        self._slot_specs: dict[int, OnlineSampleSpec] = {}
        self._val_lane_next_step: dict[str, int] = {}

    def _specs_for_batch(self, batch_size: int | None) -> list[OnlineSampleSpec]:
        size = self.split_config.batch_size if batch_size is None else int(batch_size)
        if size <= 0:
            raise ValueError("batch_size must be positive")
        # Materialize only missing slots. Existing slots retain their city,
        # seed, and master stream across training iterations.
        if any(index not in self._slot_specs for index in range(size)):
            generated = list(iter_online_samples(self.config, self.split, count=size))
            for index in range(size):
                self._slot_specs.setdefault(index, generated[index])
        return [self._slot_specs[index] for index in range(size)]

    def collect_batch(self, batch_size: int | None = None) -> list[OnlineBatchItem]:
        return self.collect_batch_specs(self._specs_for_batch(batch_size))

    def collect_batch_specs(self, specs: Sequence[OnlineSampleSpec]) -> list[OnlineBatchItem]:
        _dbg(f"collect_batch_specs start n={len(specs)}")
        started_at = time.monotonic()
        """Materialize snapshots for explicit dataloader slots.

        Online training rows are only bootstrap records.  The caller may pass
        the rows that VERL sampled so that the live master stream is keyed by
        the same ``stream_id`` rather than by a newly generated schedule.
        """
        specs = list(specs)
        if not specs:
            raise ValueError("at least one online sample spec is required")
        # A full validation call may contain 50 rows while owning only ten
        # persistent masters.  Rows 0..9 use the ten lanes, then rows 10..19
        # reuse them in the next materialization wave.  The wave scheduler
        # below never submits the same lane twice concurrently, so duplicate
        # stream IDs here are intentional and must not be rejected.
        for index, spec in enumerate(specs):
            # Explicit rows may be shuffled by the dataloader.  The durable
            # identity is ``stream_id`` in MasterStreamManager, not this
            # transient batch position; retain the mapping only for callers
            # that later request an implicit generated batch.
            self._slot_specs[index] = spec
        # Register actors first, then submit one composite RPC per master.  The
        # actor performs restart/warmup/video capture/state export atomically;
        # Ray, rather than local Python threads, provides process parallelism.
        # Create masters lazily per chunk.  Registering every stream up front
        # defeats max_parallel_masters: actors (and their SUMO/EGL processes)
        # would all start before ray.wait begins throttling RPCs.
        streams: list[Any | None] = [None] * len(specs)

        def build_item(spec: OnlineSampleSpec, stream: Any, result: Mapping[str, Any]) -> OnlineBatchItem:
            restarted = bool(result.get("restarted", False))
            if restarted:
                stream.episode_id += 1
                stream.accepted_snapshots = 0
                stream.needs_episode_reset = False
            snapshot = self.observation_builder(
                result["state"], spec.city, int(result["step"])
            )
            stream.accepted_snapshots += 1
            snapshot_dir = (
                self.snapshot_root
                / self.split
                / _safe_path_component(spec.sample_id)
                / f"step_{snapshot.step:06d}"
            )
            # The master export is the immutable handoff between online SUMO
            # materialization and Stage 1/2.  Put it at the coordinator's
            # canonical path now so later stages never need another actor RPC.
            source_path = Path(stream.latest_snapshot_path or "")
            if not source_path.is_file() or source_path.stat().st_size <= 0:
                raise RuntimeError(
                    f"missing exported SUMO snapshot for {spec.sample_id}: {source_path}"
                )
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            handoff_path = snapshot_dir / "master_snapshot.xml"
            shutil.copy2(source_path, handoff_path)
            return OnlineBatchItem(
                spec,
                snapshot,
                stream.master_actor,
                snapshot_dir,
                dict(result.get("video_details") or {}),
            )

        # Headless Panda3D/EGL rendering is substantially more sensitive to
        # fan-out during validation.  Keep Val conservative by default while
        # retaining the configured concurrency for Train; callers can raise it
        # explicitly after confirming the host is stable.
        configured_concurrency = self.config.resources.max_parallel_masters
        if self.split == "val":
            configured_concurrency = int(
                os.environ.get("V35_VAL_MASTER_CONCURRENCY", "10")
            )
        # Keep the full validation pool resident, but serialize the expensive
        # SUMO/EGL materialization work in small waves.  Ten simultaneous
        # render/export RPCs can exhaust EGL/TraCI resources even though the
        # actors themselves are healthy, causing all later lanes to time out.
        resident_concurrency = min(len(specs), max(1, int(configured_concurrency)))
        rpc_limit = int(
            os.environ.get(
                "V35_VAL_MATERIALIZE_CONCURRENCY" if self.split == "val"
                else "V35_TRAIN_MATERIALIZE_CONCURRENCY",
                # Persistent masters and active EGL materialization are
                # independent limits.  In particular, the eight Train lanes
                # are intentionally kept alive, but starting all eight native
                # render/TraCI calls at once can leave every RPC pending.  Use
                # the same conservative two-at-a-time default as Val; callers
                # may raise it through V35_TRAIN_MATERIALIZE_CONCURRENCY after
                # verifying their host's EGL capacity.
                "2",
            )
        )
        concurrency = min(resident_concurrency, max(1, rpc_limit))
        # Probe only the first lazily-created chunk.  Probing all streams here
        # would eagerly start every master and defeat the concurrency limit.
        first_chunk_specs = specs[:concurrency]
        def lane_for(spec: OnlineSampleSpec) -> str:
            # Validation rows already carry stable city/lane IDs (00..04).
            # Keep this explicit so a future bootstrap generator with unique
            # sample IDs still maps to the same ten persistent masters.
            if self.split != "val":
                # Train owns a fixed persistent pool.  Bootstrap rows may be
                # numbered 0..999, but their live SUMO state is keyed by the
                # eight dataloader slots, not by each row's unique ID.
                pool_size = self.split_config.persistent_stream_count
                if pool_size == 4 and len(self.split_config.cities) == 2:
                    # The sample generator emits city-local IDs for the
                    # four-master formal layout.  Do not accept the old
                    # ``train:slot:0016`` form here: it is a unique bootstrap
                    # row ID, not a persistent lane, and accepting it creates
                    # an unbounded number of SUMO masters.
                    slots_per_city = 2
                    city_prefix = f"{self.split_config.stream_prefix}:{spec.city}:slot:"
                    suffix = (
                        spec.stream_id[len(city_prefix):]
                        if spec.stream_id.startswith(city_prefix)
                        else ""
                    )
                    if suffix.isdigit() and int(suffix) < slots_per_city:
                        return f"{city_prefix}{int(suffix):02d}"

                    # Formal jobs can reuse a bootstrap JSONL produced before
                    # city-local train lanes existed.  Canonicalize those
                    # records by their deterministic city occurrence rather
                    # than preserving their global row ID.  Thus an old row
                    # such as ``train:slot:0016`` maps back to one of the
                    # two Jinan/Hangzhou lanes and continues its master.
                    try:
                        ordinal = max(0, int(spec.ordinal))
                    except (TypeError, ValueError):
                        ordinal = 0
                    cities = [self.config.cities[name] for name in self.split_config.cities]
                    scheduled = balanced_city_schedule(cities, ordinal + 1)
                    local_seen = sum(city == spec.city for city in scheduled[:-1])
                    return f"{city_prefix}{local_seen % slots_per_city:02d}"
                if pool_size:
                    try:
                        slot = int(spec.ordinal) % int(pool_size)
                    except (TypeError, ValueError):
                        slot = 0
                    return f"{self.split_config.stream_prefix}:slot:{slot:04d}"
                return spec.stream_id
            # Canonicalize any bootstrap naming scheme to exactly the
            # configured five lanes per city.  In particular, old bootstrap
            # files may contain stream:00..07; modulo keeps those rows on the
            # same five persistent masters instead of creating 16 actors.
            parts = spec.stream_id.rsplit(":", 1)
            try:
                ordinal = int(parts[-1])
            except (TypeError, ValueError):
                ordinal = int(spec.ordinal)
            width = max(1, int(self.config.val_metric_steps))
            return f"{self.split_config.stream_prefix}:{spec.city}:lane:{ordinal % width:02d}"

        def stream_for(spec: OnlineSampleSpec) -> Any:
            return self.streams.get_or_create_lane(spec, lane_for(spec))

        def stream_label(spec: OnlineSampleSpec) -> str:
            """Show the resident lane while retaining legacy-row provenance."""
            lane_id = lane_for(spec)
            if lane_id == spec.stream_id:
                return lane_id
            return f"{lane_id} bootstrap={spec.stream_id}"

        first_chunk_streams = [stream_for(spec) for spec in first_chunk_specs]
        streams[:concurrency] = first_chunk_streams
        composite_methods = [
            getattr(getattr(stream.master_actor, "materialize_snapshot_state", None), "remote", None)
            for stream in first_chunk_streams
        ]
        if all(callable(method) for method in composite_methods):
            import ray

            # A single SUMO materialization may include warmup, six-frame
            # rendering and state export.  Keep the bound finite but generous
            # enough for a loaded host; callers can still override it.
            timeout_s = float(os.environ.get("V35_MASTER_SNAPSHOT_TIMEOUT_S", "300"))
            wait_rounds = max(1, int(os.environ.get("V35_MASTER_SNAPSHOT_WAIT_ROUNDS", "1")))
            results_by_index: dict[int, Mapping[str, Any]] = {}
            for offset in range(0, len(specs), concurrency):
                chunk_started = time.monotonic()
                chunk_specs = specs[offset : offset + concurrency]
                chunk_streams = (
                    first_chunk_streams
                    if offset == 0
                    else [stream_for(spec) for spec in chunk_specs]
                )
                streams[offset : offset + concurrency] = chunk_streams
                refs: dict[Any, int] = {}

                def submit(local_index: int, stream: Any) -> Any:
                    """Submit one materialization while retaining its slot identity."""
                    stable_dir = (
                        self.snapshot_root / self.split
                        / _safe_path_component(chunk_specs[local_index].sample_id)
                    )
                    stable_dir.mkdir(parents=True, exist_ok=True)
                    stable_path = stable_dir / "latest_snapshot.xml"
                    method = stream.master_actor.materialize_snapshot_state.remote
                    lane_id = lane_for(chunk_specs[local_index])
                    if self.split == "val":
                        target_step = self._val_lane_next_step.get(lane_id)
                        # ``accepted_snapshots`` is committed only after the
                        # whole collection wave returns.  During a 50-row
                        # validation call, later rows for the same persistent
                        # lane therefore still observe zero here and used to
                        # repeat the initial warm-up target.  The lane cursor
                        # is reserved as soon as each materialization result
                        # arrives, so it is the authoritative first/next
                        # request discriminator.
                        if target_step is None:
                            target_step = int(stream.warmup_steps) + 1 + int(stream.initial_step_offset)
                        else:
                            target_step = int(target_step)
                    else:
                        target_step = None
                    return method(
                            city=chunk_specs[local_index].city,
                            decision_cycles=self.config.evaluation_decision_cycles,
                            episode_seconds=stream.episode_seconds,
                            decision_cycle_seconds=stream.decision_cycle_seconds,
                            warmup_steps=stream.warmup_steps,
                            initial_step_offset=stream.initial_step_offset,
                            target_step=target_step,
                            force_restart=stream.needs_episode_reset,
                            restart_seed=int(chunk_specs[local_index].seed + (stream.episode_id + 1) * 100_000),
                            # The persistent source master stays alive for a
                            # complete validation call.  Its in-memory SUMO
                            # state is authoritative; exported XML is only a
                            # handoff to Stage 1/2 rollout actors and must not
                            # rewind this master before the next Val batch.
                            restore_path=None,
                            snapshot_path=str(stable_path),
                    )

                for local_index, stream in enumerate(chunk_streams):
                    refs[submit(local_index, stream)] = local_index
                waiting = list(refs)
                replacement_attempts: dict[int, int] = {}
                for wait_round in range(1, wait_rounds + 1):
                    if not waiting:
                        break
                    ready, waiting = ray.wait(waiting, num_returns=len(waiting), timeout=timeout_s)
                    for ref in ready:
                        local_index = refs[ref]
                        try:
                            result = ray.get(ref)
                        except Exception as exc:
                            # A failed task can be returned as ready when Ray
                            # observes actor death.  Convert it into one
                            # bounded replacement attempt instead of dropping
                            # the sample or leaving a dead ref in the wait set.
                            if replacement_attempts.get(local_index, 0) >= 1:
                                raise RuntimeError(
                                    f"master materialization failed after replacement: "
                                    f"stream={stream_label(chunk_specs[local_index])}"
                                ) from exc
                            replacement_attempts[local_index] = 1
                            spec = chunk_specs[local_index]
                            replacement = self.streams.replace(spec)
                            chunk_streams[local_index] = replacement
                            streams[offset + local_index] = replacement
                            new_ref = submit(local_index, replacement)
                            refs[new_ref] = local_index
                            waiting.append(new_ref)
                            print(
                                f"[MASTER_SNAPSHOT_REPLACE] split={self.split} "
                                f"stream={stream_label(spec)} reason={type(exc).__name__}",
                                flush=True,
                            )
                            continue
                        results_by_index[offset + local_index] = result
                        stream = chunk_streams[local_index]
                        snapshot_path = result.get("snapshot_path")
                        if not snapshot_path:
                            raise RuntimeError(
                                f"master did not export a snapshot for "
                                f"{chunk_specs[local_index].sample_id}"
                            )
                        stream.latest_snapshot_path = str(snapshot_path)
                        if self.split == "val":
                            lane_id = lane_for(chunk_specs[local_index])
                            # commit_deployment_rollouts advances the source
                            # master by exactly one V25 cycle after this
                            # snapshot.  The next batch must target that
                            # resulting step, not skip an additional cycle.
                            self._val_lane_next_step[lane_id] = int(result["step"]) + 1
                    if waiting:
                        names = [stream_label(chunk_specs[refs[ref]]) for ref in waiting]
                        print(
                            f"[MASTER_SNAPSHOT_WAIT] split={self.split} batch={offset // concurrency + 1} "
                            f"round={wait_round}/{wait_rounds} ready={len(ready)} pending={len(waiting)} "
                            f"streams={names}",
                            flush=True,
                        )
                if waiting:
                    # ray.wait can leave a task pending forever when its actor
                    # dies before Ray publishes task completion. Probe the
                    # actor directly so this state converges deterministically.
                    dead: list[tuple[int, Exception]] = []
                    for ref in waiting:
                        local_index = refs[ref]
                        actor = chunk_streams[local_index].master_actor
                        try:
                            probe = actor.current_time.remote()
                            probe_ready, _ = ray.wait([probe], num_returns=1, timeout=5)
                            if not probe_ready:
                                raise TimeoutError("master health probe timed out")
                            ray.get(probe)
                        except Exception as exc:
                            dead.append((local_index, exc))
                    # A responsive health probe does not prove that the
                    # materialization RPC is making progress: the actor can
                    # be stuck inside SUMO/EGL while another RPC remains
                    # serviceable. Treat every still-pending materialization
                    # as unhealthy and replace it once, just like actor death.
                    pending_indices = {refs[ref] for ref in waiting}
                    # A timed-out RPC must not remain queued on the old actor.
                    # Cancel the Ray task before replacing the actor; otherwise
                    # the stale task can keep the native SUMO process alive
                    # and consume one of the limited EGL slots indefinitely.
                    for ref in waiting:
                        try:
                            ray.cancel(ref, force=True)
                        except Exception:
                            pass
                    replacement_refs: list[Any] = []
                    for local_index in sorted(pending_indices):
                        exc = next((err for idx, err in dead if idx == local_index),
                                   TimeoutError("master materialization timed out"))
                        spec = chunk_specs[local_index]
                        status = {}
                        try:
                            inspector = getattr(chunk_streams[local_index].master_actor, "debug_status", None)
                            if inspector is not None and getattr(inspector, "remote", None) is not None:
                                status_ref = inspector.remote()
                                status_ready, _ = ray.wait([status_ref], num_returns=1, timeout=5)
                                if status_ready:
                                    status = dict(ray.get(status_ref))
                        except Exception as status_exc:
                            status = {"status_error": type(status_exc).__name__}
                        print(
                            f"[MASTER_SNAPSHOT_TIMEOUT_DIAG] split={self.split} "
                            f"stream={stream_label(spec)} status={status}",
                            flush=True,
                        )
                        if replacement_attempts.get(local_index, 0) >= 1:
                            raise RuntimeError(
                                f"master actor unavailable after replacement: "
                                f"stream={stream_label(spec)}; no samples were dropped"
                            ) from exc
                        replacement_attempts[local_index] = 1
                        replacement = self.streams.replace(spec)
                        chunk_streams[local_index] = replacement
                        streams[offset + local_index] = replacement
                        new_ref = submit(local_index, replacement)
                        refs[new_ref] = local_index
                        replacement_refs.append(new_ref)
                        print(
                            f"[MASTER_SNAPSHOT_REPLACE] split={self.split} "
                            f"stream={stream_label(spec)} reason={type(exc).__name__}",
                            flush=True,
                        )
                    # Give all replacements one bounded wait round. Successful
                    # replacements are materialized normally; unresolved ones
                    # fail explicitly instead of being silently dropped.
                    # The original refs were force-cancelled above.  Waiting
                    # on them can never provide a result and was the source
                    # of apparent hangs after a timeout.  Only replacement
                    # refs belong in this bounded wait.
                    waiting = replacement_refs
                    ready, waiting = ray.wait(waiting, num_returns=len(waiting), timeout=timeout_s)
                    for ref in ready:
                        idx = refs[ref]
                        try:
                            result = ray.get(ref)
                        except Exception as replacement_exc:
                            raise RuntimeError(
                                f"master materialization failed after replacement: "
                                f"stream={stream_label(chunk_specs[idx])}"
                            ) from replacement_exc
                        results_by_index[offset + idx] = result
                        snapshot_path = result.get("snapshot_path")
                        if not snapshot_path:
                            raise RuntimeError(f"master did not export a snapshot for {chunk_specs[idx].sample_id}")
                        chunk_streams[idx].latest_snapshot_path = str(snapshot_path)
                        if self.split == "val":
                            lane_id = lane_for(chunk_specs[idx])
                            self._val_lane_next_step[lane_id] = int(result["step"]) + 1
                    if waiting:
                        names = [stream_label(chunk_specs[refs[ref]]) for ref in waiting]
                        raise TimeoutError(
                            f"master snapshot timed out after {timeout_s * wait_rounds:.0f}s; "
                            f"no samples were dropped; pending streams={names}"
                        )
                print(
                    f"[MASTER_CHUNK_TIMING] split={self.split} "
                    f"batch={offset // concurrency + 1} samples={len(chunk_specs)} "
                    f"elapsed_s={time.monotonic() - chunk_started:.3f}",
                    flush=True,
                )
            items = []
            for index, result in sorted(results_by_index.items()):
                item = build_item(specs[index], streams[index], result)
                items.append(item)
        else:
            # Local test doubles retain the synchronous compatibility path.
            items = []
            for spec, stream in zip(specs, streams):
                snapshot = stream.snapshot(
                    self.observation_builder,
                    decision_cycles=self.config.evaluation_decision_cycles,
                )
                getter = getattr(stream.master_actor, "latest_video_details", None)
                video_details = getter() if callable(getter) else {}
                snapshot_dir = (
                    self.snapshot_root / self.split / _safe_path_component(spec.sample_id)
                    / f"step_{snapshot.step:06d}"
                )
                items.append(OnlineBatchItem(
                    spec, snapshot, stream.master_actor, snapshot_dir, dict(video_details or {})
                ))
        _dbg(
            f"collect_batch_specs done n={len(items)} resident={resident_concurrency} "
            f"rpc_concurrency={concurrency} "
            f"elapsed_s={time.monotonic() - started_at:.3f}"
        )
        return items

    def rollout_and_commit(self, batch_size: int | None = None) -> OnlineBatchResult:
        return self.rollout_and_commit_specs(self._specs_for_batch(batch_size))

    def rollout_and_commit_specs(self, specs: Sequence[OnlineSampleSpec]) -> OnlineBatchResult:
        items = self.collect_batch_specs(specs)
        rollouts = self.coordinator.run_batch_from_master_actors(
            [(item.snapshot, item.master_actor, item.snapshot_dir) for item in items],
            self.policy_factory,
            prompt_template=self.prompt_template,
            decision_cycles=self.config.evaluation_decision_cycles,
            concurrency=None,
            seed=None,
        )
        return self.commit_rollouts(items, rollouts)

    def prepare_for_generation(
        self, specs: Sequence[OnlineSampleSpec]
    ) -> tuple[list[OnlineBatchItem], list[Any]]:
        """Materialize t0 observations and snapshots before policy inference."""
        _dbg(f"prepare_for_generation start n={len(specs)}")
        items = self.collect_batch_specs(specs)
        _dbg(f"prepare_for_generation done n={len(items)}")
        prepared = self.coordinator.prepare_batch_from_master_actors(
            [(item.snapshot, item.master_actor, item.snapshot_dir) for item in items],
            seed=None,
        )
        return items, prepared

    def release_exported_masters(self, items: Sequence[OnlineBatchItem]) -> None:
        """Keep validation masters alive through all batches of one Val call.

        The name remains for adapter compatibility.  Stage 1/2 consume the
        immutable exported XML, while each source master remains resident so
        the next ten-row validation batch can continue with one V25 cycle
        instead of reconstructing SUMO and repeating warm-up.  ``reset()``
        releases the whole pool after the final validation batch.
        """
        if self.split != "val":
            raise RuntimeError("only validation masters may be released after export")
        # ``get_or_create_lane`` stores the canonical ``val:*:lane:*`` key
        # while ``item.spec.stream_id`` retains the bootstrap
        # ``val:*:stream:*`` identifier.  Resolve the alias before counting;
        # otherwise the diagnostic incorrectly reports ``held=0`` even while
        # all ten persistent validation masters are still alive.
        held_keys = {
            self.streams._aliases.get(item.spec.stream_id, item.spec.stream_id)
            for item in items
        }
        held = sum(key in self.streams._streams for key in held_keys)
        print(
            f"[MASTER_EXPORT_HOLD] split=val held={held} "
            f"samples={len(items)}",
            flush=True,
        )

    def commit_rollouts(
        self,
        items: Sequence[OnlineBatchItem],
        rollouts: Sequence[Sequence[CityRolloutResult]],
    ) -> OnlineBatchResult:
        """Select and commit evaluated candidates to the persistent masters."""
        if len(rollouts) != len(items):
            raise RuntimeError(
                f"coordinator returned {len(rollouts)} rollout groups for {len(items)} snapshots"
            )
        selected_rollout_ids: list[int] = []
        for item, candidates in zip(items, rollouts):
            if len(candidates) != self.config.num_rollouts:
                raise RuntimeError(
                    f"coordinator returned {len(candidates)} candidates for {item.spec.sample_id}; "
                    f"expected {self.config.num_rollouts}"
                )
            selected = self.select_candidate(candidates)
            selected_rollout_ids.append(int(selected.rollout_id))
            stream = self.streams.get_or_create(item.spec)
            stream.advance_v25(1)
        records = [record for candidates in rollouts for record in flatten_for_gdpo(candidates)]
        return OnlineBatchResult(
            items=items,
            rollouts=rollouts,
            gdpo_records=records,
            selected_rollout_ids=selected_rollout_ids,
        )

    def commit_deployment_rollouts(
        self,
        items: Sequence[OnlineBatchItem],
        rollouts: Sequence[Sequence[CityRolloutResult]],
    ) -> OnlineBatchResult:
        """Commit each validation policy action without oracle candidate selection."""
        if len(items) != len(rollouts):
            raise RuntimeError("deployment rollout groups must align with validation streams")
        selected: list[int] = []
        for item, candidates in zip(items, rollouts):
            if len(candidates) != 1:
                raise RuntimeError("deployment validation requires exactly one trajectory per stream")
            result = candidates[0]
            # Validation rows for a persistent lane are materialized in order
            # by ``_val_lane_next_step``.  That cursor already targets the
            # next V25 cycle while collecting the batch (including duplicate
            # lanes in a 50-row validation call).  Advancing here as well
            # would double-step the source master and desynchronize it from
            # the cursor, causing subsequent snapshots to be repeated or
            # skipped.  Rollouts consume immutable exported snapshots, so no
            # additional actor mutation is needed at commit time.
            selected.append(int(result.rollout_id))
        records = [record for candidates in rollouts for record in flatten_for_gdpo(candidates)]
        return OnlineBatchResult(list(items), [list(x) for x in rollouts], records, selected)

    def close(self) -> None:
        self.streams.close()

    def suspend_for_validation(self) -> int:
        """Release Train SUMO/EGL actors while validation owns its lane pool."""
        if self.split != "train":
            raise RuntimeError("only the train runtime may be suspended for validation")
        started_at = time.monotonic()
        print(
            f"[TRAIN_MASTER_SUSPEND_BEGIN] streams={len(self.streams._streams)}",
            flush=True,
        )
        count = self.streams.suspend(self.snapshot_root)
        print(
            f"[TRAIN_MASTER_SUSPEND_DONE] streams={count} "
            f"elapsed_s={time.monotonic() - started_at:.3f}",
            flush=True,
        )
        return count

    def resume_after_validation(self) -> int:
        """Recreate Train SUMO/EGL actors from their pre-validation snapshots."""
        if self.split != "train":
            raise RuntimeError("only the train runtime may be resumed after validation")
        started_at = time.monotonic()
        print(
            f"[TRAIN_MASTER_RESUME_BEGIN] streams={len(self.streams._suspended_streams)}",
            flush=True,
        )
        count = self.streams.resume()
        print(
            f"[TRAIN_MASTER_RESUME_DONE] streams={count} "
            f"elapsed_s={time.monotonic() - started_at:.3f}",
            flush=True,
        )
        return count

    def reset(self) -> None:
        """Reset this split to its configured stream IDs and initial seeds."""
        self.streams.close()
        self._slot_specs.clear()
        self._val_lane_next_step.clear()


__all__ = [
    "OnlineBatchItem",
    "OnlineBatchResult",
    "OnlineCooperativeRuntime",
    "select_best_city_rollout",
]
