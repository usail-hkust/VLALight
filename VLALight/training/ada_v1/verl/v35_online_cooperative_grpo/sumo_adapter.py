"""Lightweight adapter between the Stage 2 rollout protocol and ``SUMOEnv``.

The online rollout code deliberately knows nothing about TraCI or the concrete
SUMO environment.  This module is the small integration boundary: phase labels
are converted to the action indices expected by ``SUMOEnv.step``, all signals
are submitted as one synchronized table, and queue metrics are exposed in the
mapping form consumed by :mod:`online_reward`.
"""

from __future__ import annotations

from copy import deepcopy
import json
from numbers import Integral
from pathlib import Path
from typing import Any, Callable, Mapping


PHASE_NAMES = ("ETWT", "NTST", "ELWL", "NLSL")
_PHASE_MOVEMENTS = {"ETWT": (4, 1), "NTST": (7, 10), "ELWL": (3, 0), "NLSL": (6, 9)}


def _intersections(env: Any) -> list[Any]:
    values = getattr(env, "list_intersection", None)
    if values is None:
        raise TypeError("SUMO environment must expose list_intersection")
    return list(values)


class MasterVisualCapture:
    """TransSimHub video capture used only by persistent master simulators."""

    def __init__(self, env: Any, config: Mapping[str, Any], work_dir: Path) -> None:
        import os
        import sys
        from loguru import logger

        # Training diagnostics (V35_VERBOSE_DEBUG) must not enable the very
        # noisy TransSimHub renderer INFO stream.  Renderer verbosity is an
        # independent opt-in so vehicle-model selection/removal messages do
        # not flood the training terminal during normal debug runs.
        if os.environ.get("V35_RENDER_VERBOSE", "0") != "1":
            logger.remove()
            logger.add(
                sys.stderr,
                level=os.environ.get("V35_RENDER_LOG_LEVEL", "WARNING"),
                colorize=False,
            )
        from TransSimHub.tshub.tshub_env3d.vis3d_renderer.tshub_render import TSHubRenderer
        from utils.decision_window_recorder import DecisionWindowRecorder

        self.env = env
        self.config = dict(config.get("VLM_CONFIG", {}))
        # Formal online training is headless.  The repository-wide VLM
        # defaults still select ``pandagl`` for interactive runs; on a node
        # without X that silently falls back to TinyDisplay, which cannot
        # decode the sRGB vehicle textures used by TransSimHub.
        self.config["RENDERING_BACKEND"] = "p3headlessgl"
        self.tls_ids = [str(item.inter_id) for item in _intersections(env)]
        mapping_path = Path(self.config["DIRECTION_MAPPING_PATH"])
        with mapping_path.open("r", encoding="utf-8") as handle:
            direction_mapping = json.load(handle)
        self._validate_video_contract()
        sensor_type = self.config.get("TLS_SENSOR_TYPE", "junction_front_all")
        sensor_config = {"tls": {
            tls_id: {
                "sensor_types": [sensor_type],
                "tls_camera_height": self.config.get("TLS_CAMERA_HEIGHT", 30),
            }
            for tls_id in self.tls_ids
        }}
        data_dir = Path(env.dic_path.get("PATH_TO_DATA", work_dir))
        netxml_path = data_dir / str(config["ROADNET_FILE"])
        self.renderer = TSHubRenderer(
            simid="sumo",
            sensor_config=sensor_config,
            preset=self.config.get("RENDER_PRESET", "1080P"),
            resolution=self.config.get("RENDER_RESOLUTION", 1.0),
            scenario_glb_dir=self.config["SCENARIO_GLB_DIR"],
            vehicle_model=self.config.get("VEHICLE_MODEL", "low"),
            render_mode="offscreen",
            rendering_backend=self.config.get("RENDERING_BACKEND", "p3headlessgl"),
            show_buildings=self.config.get("SHOW_BUILDINGS", False),
            tls_batch_size=self.config.get("TLS_BATCH_SIZE"),
            keep_batch_sensors=self.config.get("RENDER_KEEP_BATCH_SENSORS", False),
            reuse_batch_sensors=self.config.get("RENDER_REUSE_BATCH_SENSORS", True),
            step_task_manager=self.config.get("RENDER_STEP_TASK_MANAGER", False),
            netxml_path=str(netxml_path),
            show_arrows=self.config.get("SHOW_ARROWS", False),
        )
        self._validate_headless_renderer()
        self.renderer.reset(env.get_tls_init_info(self.tls_ids))
        self.recorder = DecisionWindowRecorder(
            session_dir=str(work_dir),
            direction_mapping=direction_mapping,
            fps=int(self.config.get("VIDEO_FPS", 1)),
            add_labels=bool(self.config.get("VIDEO_ADD_LABELS", True)),
            export_direction_videos=True,
            # Keep only the four directional clips required by Stage 1.  The
            # 2x2 composite is a debug artifact and adds substantial encoding
            # and disk-I/O cost during validation.
            export_composite_video=False,
            export_direction_sequence=False,
            enable_preprocess=True,
            left_crop=float(self.config.get("IMAGE_PREPROCESS_LEFT_CROP", 0.40)),
            right_crop=float(self.config.get("IMAGE_PREPROCESS_RIGHT_CROP", 0.30)),
            scale_mode=str(self.config.get("IMAGE_PREPROCESS_SCALE_MODE", "fit_width")),
            frame_view="legacy_crop",
            tile_width=512,
            sensor_type=sensor_type,
            record_mode="sampled",
            sim_interval=float(config.get("INTERVAL", 1.0)),
            sample_interval=5.0,
            preprocess_interpolation=str(self.config.get("VIDEO_PREPROCESS_INTERPOLATION", "linear")),
            async_video_write=False,
        )
        self.latest_video_details: dict[str, Any] = {}
        self._decision_step = 0

    def _validate_headless_renderer(self) -> None:
        showbase = getattr(self.renderer, "_showbase_instance", None)
        pipe = getattr(showbase, "pipe", None)
        pipe_name = pipe.getType().getName() if pipe is not None else ""
        window = getattr(showbase, "win", None)
        gsg = window.getGsg() if window is not None else None
        vendor = gsg.getDriverVendor().strip() if gsg is not None else ""
        renderer = gsg.getDriverRenderer().strip() if gsg is not None else ""
        version = gsg.getDriverVersion().strip() if gsg is not None else ""

        if pipe_name != "eglGraphicsPipe" or not vendor or not renderer or not version:
            raise RuntimeError(
                "master visual capture requires a valid p3headlessgl context; "
                f"got pipe={pipe_name!r}, vendor={vendor!r}, "
                f"renderer={renderer!r}, version={version!r}"
            )

    def _validate_video_contract(self) -> None:
        expected = {
            "RENDER_PRESET": "1080P",
            "VIDEO_FRAME_VIEW": "legacy_crop",
            "VIDEO_TILE_WIDTH": 512,
            "IMAGE_PREPROCESS_LEFT_CROP": 0.40,
            "IMAGE_PREPROCESS_RIGHT_CROP": 0.30,
            "IMAGE_PREPROCESS_SCALE_MODE": "fit_width",
        }
        for key, value in expected.items():
            actual = self.config.get(key)
            if actual != value:
                raise ValueError(f"master video contract requires VLM_CONFIG.{key}={value!r}, got {actual!r}")

    def begin(self) -> None:
        self._decision_step += 1
        self.recorder.begin_interval(self._decision_step, float(self.env.get_current_time()))

    def callback(self, inner_i: int, env: Any) -> None:
        if (int(inner_i) + 1) % 5 != 0:
            return
        sensor_data = self.renderer.step(
            env.get_tshub_obs(tls_ids=self.tls_ids,
                              radius=self.config.get("LOCAL_RENDER_RADIUS_M", 200.0)),
            should_count_vehicles=False,
        )
        if sensor_data:
            self.recorder.add_sensor_data(
                sensor_data, self.tls_ids, sim_time=float(env.get_current_time())
            )

    def finish(self) -> dict[str, Any]:
        details = self.recorder.finalize_interval(
            float(self.env.get_current_time()), return_details=True
        )
        self._validate_saved_dimensions(details)
        self.latest_video_details = details
        return details

    def capture_current_state(self) -> dict[str, Any]:
        """Export Stage-1 videos at the current SUMO time without advancing."""
        self.begin()
        sensor_data = self.renderer.step(
            self.env.get_tshub_obs(
                tls_ids=self.tls_ids,
                radius=self.config.get("LOCAL_RENDER_RADIUS_M", 200.0),
            ),
            should_count_vehicles=False,
        )
        if sensor_data:
            self.recorder.add_sensor_data(
                sensor_data, self.tls_ids, sim_time=float(self.env.get_current_time())
            )
        return self.finish()

    def _validate_saved_dimensions(self, details: Mapping[str, Any]) -> None:
        import cv2

        paths = details.get("paths", {})
        if not paths:
            raise RuntimeError("master visual capture produced no direction videos")
        missing_tls = sorted(set(self.tls_ids) - {str(tls_id) for tls_id in paths})
        if missing_tls:
            raise RuntimeError(
                "master visual capture omitted direction videos for TLS: "
                + ", ".join(missing_tls)
            )
        for tls_id, media in paths.items():
            for direction in ("N", "E", "W", "S"):
                path = media.get(direction)
                if not path:
                    raise RuntimeError(f"missing {direction} video for {tls_id}")
                capture = cv2.VideoCapture(str(path))
                try:
                    ok, frame = capture.read()
                finally:
                    capture.release()
                if not ok or frame is None:
                    raise RuntimeError(f"cannot read first frame from {path}")
                height, width = frame.shape[:2]
                if (width, height) != (512, 960):
                    raise RuntimeError(
                        f"invalid video frame size for {tls_id}/{direction}: "
                        f"{width}x{height}, expected 512x960"
                    )

    def close(self) -> None:
        closer = getattr(self.renderer, "close", None)
        if callable(closer):
            closer()
        self.renderer = None


def _index_value(value: Any) -> int | None:
    """Normalize SUMO/numpy/string indices without accepting booleans."""
    if isinstance(value, bool):
        return None
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _indexed_mapping_value(mapping: Any, index: Any) -> tuple[bool, Any]:
    """Look up a phase mapping whose keys may be int or serialized strings."""
    if not isinstance(mapping, Mapping):
        return False, None
    normalized = _index_value(index)
    candidates = [index]
    if normalized is not None:
        candidates.extend((normalized, str(normalized)))
    for candidate in candidates:
        try:
            if candidate in mapping:
                return True, mapping[candidate]
        except TypeError:
            continue
    return False, None


def _intersections(env: Any) -> list[Any]:
    values = getattr(env, "list_intersection", None)
    if values is None:
        raise TypeError("SUMO environment must expose list_intersection")
    return list(values)


def _intersection_id(intersection: Any) -> str:
    value = getattr(intersection, "inter_id", None)
    if value is None:
        value = getattr(intersection, "tls_id", None)
    if value is None:
        raise TypeError("intersection objects must expose inter_id or tls_id")
    return str(value)


def phase_name_to_action(
    env: Any,
    signals: Mapping[str, str],
    *,
    require_all: bool = True,
) -> dict[str, int]:
    """Convert ``{intersection_id: phase_name}`` to SUMO action indices.

    ``SUMOEnv.step`` consumes indices into each intersection's
    ``control_phases`` list, not the phase strings used by the Stage 2 prompt.
    By default the table must contain exactly every active intersection so a
    rollout cannot accidentally advance only part of the city.
    """
    if not isinstance(signals, Mapping) or not signals:
        raise ValueError("signals must be a non-empty mapping")
    intersections = _intersections(env)
    by_id = {_intersection_id(item): item for item in intersections}
    unknown = set(str(key) for key in signals) - set(by_id)
    if unknown:
        raise KeyError(f"signals contain unknown intersections: {sorted(unknown)}")
    if require_all:
        missing = set(by_id) - set(str(key) for key in signals)
        if missing:
            raise KeyError(f"signals omit intersections: {sorted(missing)}")

    actions: dict[str, int] = {}
    for raw_id, phase in signals.items():
        intersection_id = str(raw_id)
        if not isinstance(phase, str):
            raise TypeError(f"phase for {intersection_id!r} must be a string")
        control_phases = list(getattr(by_id[intersection_id], "control_phases", ()))
        if phase not in control_phases:
            raise ValueError(
                f"{intersection_id} cannot execute phase {phase!r}; "
                f"available phases: {control_phases}"
            )
        actions[intersection_id] = control_phases.index(phase)
    return actions


class SUMOEnvAdapter:
    """Adapter implementing ``online_rollout.SimulatorAdapter``.

    Parameters
    ----------
    env:
        An already-created ``utils.sumo_env.SUMOEnv`` (or a compatible test
        double).
    decision_cycle_seconds:
        Duration of one Stage 2 decision cycle.  V35 uses 25 seconds of green
        plus 5 seconds of transition, hence the default is 30 seconds.
    strict_signal_table:
        Require one action for every active intersection when advancing.
    """

    def __init__(
        self,
        env: Any,
        *,
        decision_cycle_seconds: float = 30.0,
        strict_signal_table: bool = True,
        owns_env: bool = False,
        allow_phase_fallback: bool = False,
        visual_capture: MasterVisualCapture | None = None,
        visual_capture_factory: Callable[[], MasterVisualCapture] | None = None,
    ) -> None:
        if decision_cycle_seconds <= 0:
            raise ValueError("decision_cycle_seconds must be positive")
        self.env = env
        self.decision_cycle_seconds = float(decision_cycle_seconds)
        self.strict_signal_table = bool(strict_signal_table)
        self.owns_env = owns_env
        # A missing runtime phase is an integration error in real SUMO.  A
        # fallback is available only for deliberately minimal test doubles.
        self.allow_phase_fallback = bool(allow_phase_fallback)
        self.visual_capture = visual_capture
        self.visual_capture_factory = visual_capture_factory
        self._latest_video_details: dict[str, Any] = {}
        self._visual_decision_step = int(
            getattr(visual_capture, "_decision_step", 0) if visual_capture is not None else 0
        )
        self._pending_signals: dict[str, str] | None = None
        self._v25_phase_histories: dict[str, dict[str, list[int]]] = {}

    @property
    def intersection_ids(self) -> tuple[str, ...]:
        return tuple(_intersection_id(item) for item in _intersections(self.env))

    def restore(self, snapshot: Any) -> None:
        """Restore a SUMO snapshot path before one candidate rollout.

        ``None`` means "keep the current state" and is useful when the caller
        has already positioned a fresh environment at the desired step.  A
        path is passed through ``SUMOEnv.load_from_file`` with strict error
        propagation when that signature is available.
        """
        self._pending_signals = None
        if snapshot is None:
            return
        path = str(snapshot) if isinstance(snapshot, (str, Path)) else snapshot
        loader = getattr(self.env, "load_from_file", None)
        if not callable(loader):
            raise TypeError("environment does not expose load_from_file")
        try:
            loader(path, quiet=True, raise_on_error=True)
        except TypeError:
            try:
                loader(path, quiet=True)
            except TypeError:
                loader(path)

    def save_snapshot(self, path: str | Path) -> str:
        """Save and return a snapshot path for reuse across the six trials."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        saver = getattr(self.env, "snapshot", None)
        if not callable(saver):
            raise TypeError("environment does not expose snapshot")
        saved = saver(str(target))
        if saved is None:
            raise RuntimeError(f"environment failed to save snapshot: {target}")
        return str(saved)

    def apply_signals(self, signals: Mapping[str, str]) -> None:
        """Validate and stage a complete synchronized signal table."""
        phase_name_to_action(self.env, signals, require_all=self.strict_signal_table)
        self._pending_signals = {str(key): str(value) for key, value in signals.items()}

    def advance(self, decision_cycles: int) -> Any:
        """Execute staged signals and advance the requested number of cycles."""
        if isinstance(decision_cycles, bool) or int(decision_cycles) != decision_cycles:
            raise ValueError("decision_cycles must be a positive integer")
        decision_cycles = int(decision_cycles)
        if decision_cycles <= 0:
            raise ValueError("decision_cycles must be positive")
        if self._pending_signals is None:
            raise RuntimeError("apply_signals must be called before advance")

        signals = self._pending_signals
        self._pending_signals = None
        actions = phase_name_to_action(
            self.env, signals, require_all=self.strict_signal_table
        )
        seconds = self.decision_cycle_seconds * decision_cycles
        # SUMOEnv uses a one-second (or configured INTERVAL) internal step and
        # accepts the requested wall-clock duration through min_action_time.
        if self.visual_capture is None and self.visual_capture_factory is not None:
            self.visual_capture = self.visual_capture_factory()
            self.visual_capture._decision_step = self._visual_decision_step
        if self.visual_capture is None:
            return self.env.step(actions, min_action_time=seconds)
        self.visual_capture.begin()
        try:
            result = self.env.step(
                actions,
                min_action_time=seconds,
                inner_step_callback=self.visual_capture.callback,
            )
        except BaseException:
            self.visual_capture.recorder.finalize_interval(
                float(self.env.get_current_time()), force_discard=True
            )
            raise
        self._latest_video_details = deepcopy(self.visual_capture.finish())
        self._visual_decision_step = int(self.visual_capture._decision_step)
        return result

    def release_visual_renderer(self) -> None:
        """Release EGL/Panda3D resources while preserving SUMO and video paths."""
        if self.visual_capture is None:
            return
        self.visual_capture.close()
        self.visual_capture = None

    def latest_video_details(self) -> dict[str, Any]:
        """Return the last complete 512x960 master video window."""
        if self.visual_capture is not None and self.visual_capture.latest_video_details:
            return deepcopy(self.visual_capture.latest_video_details)
        return deepcopy(self._latest_video_details)

    def export_video_details(self) -> dict[str, Any]:
        return deepcopy(self._latest_video_details)

    def restore_video_details(self, details: Mapping[str, Any]) -> None:
        self._latest_video_details = deepcopy(dict(details or {}))

    def ensure_current_video_details(self) -> dict[str, Any]:
        """Rebuild media after an actor was restored from XML."""
        if self._latest_video_details.get("paths"):
            return deepcopy(self._latest_video_details)
        if self.visual_capture is None and self.visual_capture_factory is not None:
            self.visual_capture = self.visual_capture_factory()
            self.visual_capture._decision_step = self._visual_decision_step
        if self.visual_capture is None:
            raise RuntimeError("cannot rebuild Stage-1 videos without visual capture")
        self._latest_video_details = deepcopy(self.visual_capture.capture_current_state())
        self._visual_decision_step = int(self.visual_capture._decision_step)
        return deepcopy(self._latest_video_details)

    def v25_signal_table(self) -> dict[str, str]:
        """Choose every master action with the repository's V25 rule."""
        signals: dict[str, str] = {}
        for intersection in _intersections(self.env):
            intersection_id = _intersection_id(intersection)
            phases = list(getattr(intersection, "control_phases", PHASE_NAMES))
            feature = getattr(intersection, "dic_feature", {}) or {}
            movement_ids = feature.get("traffic_movement_vehicle_ids_150m") or []
            direct = feature.get("movement_v") or []
            values = []
            for index in range(12):
                if index < len(movement_ids):
                    values.append(len(movement_ids[index] or []))
                else:
                    values.append(int(direct[index]) if index < len(direct) else 0)
            cycle_history = list(feature.get("v9_cycle_150m_history") or [])
            histories = self._v25_phase_histories.setdefault(
                intersection_id, {phase: [] for phase in phases}
            )

            def phase_sum(row: Any, phase: str) -> int:
                try:
                    return sum(int(row[index]) for index in _PHASE_MOVEMENTS[phase])
                except (IndexError, TypeError, ValueError):
                    return 0

            scores = []
            for order, phase in enumerate(phases):
                current_v = phase_sum(values, phase)
                histories.setdefault(phase, []).append(current_v)
                cycle_values = [phase_sum(row, phase) for row in cycle_history]
                delta = cycle_values[-1] - cycle_values[0] if len(cycle_values) >= 2 else 0
                scores.append((current_v, delta, sum(value > 0 for value in histories[phase]), -order))
            selected = max(range(len(phases)), key=lambda index: scores[index])
            phase = phases[selected]
            signals[intersection_id] = phase
            histories[phase] = []
        return signals

    def advance_v25(self, decision_cycles: int = 1) -> Any:
        result = None
        for _ in range(int(decision_cycles)):
            self.apply_signals(self.v25_signal_table())
            result = self.advance(1)
            # Age is controller-side history and is not reconstructed from a
            # single final SUMO state. Update it after every warm-up/committed
            # V25 cycle so the first trainable snapshot does not restart age.
            from .observation_builder import collect_observation_state
            step = int(round(self.current_time() / self.decision_cycle_seconds))
            collect_observation_state(self, step=step)
        return result

    def reset_v25_history(self) -> None:
        self._v25_phase_histories.clear()
        self._v35_age_tracker = {}

    def export_controller_history(self) -> dict[str, Any]:
        import copy
        return {"v25_phase_histories": copy.deepcopy(self._v25_phase_histories), "v35_age_tracker": copy.deepcopy(getattr(self, "_v35_age_tracker", {}))}

    def restore_controller_history(self, state: Mapping[str, Any]) -> None:
        import copy
        self._v25_phase_histories = copy.deepcopy(state.get("v25_phase_histories", {}))
        self._v35_age_tracker = copy.deepcopy(state.get("v35_age_tracker", {}))

    def queue_metrics(self) -> dict[str, float]:
        """Return total incoming stopped vehicles per intersection."""
        result: dict[str, float] = {}
        for intersection in _intersections(self.env):
            values = None
            feature = getattr(intersection, "dic_feature", None)
            if isinstance(feature, Mapping):
                values = feature.get("lane_num_waiting_vehicle_in")
            if values is None:
                values = getattr(intersection, "dic_lane_waiting_vehicle_count_current_step", None)
            if isinstance(values, Mapping):
                entering = getattr(intersection, "list_entering_lanes", None)
                if entering:
                    total = sum(float(values.get(lane, 0.0)) for lane in entering if lane is not None)
                else:
                    total = sum(float(value) for value in values.values())
            elif values is None:
                total = 0.0
            else:
                try:
                    total = sum(float(value) for value in values)
                except TypeError as exc:
                    raise TypeError(
                        f"invalid queue values for {_intersection_id(intersection)!r}"
                    ) from exc
            result[_intersection_id(intersection)] = float(total)
        return result

    def current_signal_table(self) -> dict[str, str]:
        """Return the managed green phase for every active intersection.

        This is used to warm a master before the first valid snapshot.  The
        lookup follows SUMOEnv's runtime phase mappings.  Real environments do
        not silently fall back to an arbitrary phase; set
        ``allow_phase_fallback=True`` only for a minimal test double.
        """
        result: dict[str, str] = {}
        for intersection in _intersections(self.env):
            intersection_id = _intersection_id(intersection)
            control_phases = list(getattr(intersection, "control_phases", ()))
            phase_name = None
            direct = getattr(intersection, "current_phase_name", None)
            if isinstance(direct, str) and direct in control_phases:
                phase_name = direct
            current_index = getattr(intersection, "current_phase_index", None)
            phase_map = getattr(intersection, "phase_index_2_phase_name", {})
            found, candidate = _indexed_mapping_value(phase_map, current_index)
            if phase_name is None and found and isinstance(candidate, str) and candidate in control_phases:
                phase_name = candidate
            action_map = getattr(intersection, "phase_index_2_action", {})
            found, action_index = _indexed_mapping_value(action_map, current_index)
            action_index = _index_value(action_index)
            if phase_name is None and found and action_index is not None and 0 <= action_index < len(control_phases):
                phase_name = control_phases[action_index]
            feature = getattr(intersection, "dic_feature", {})
            cur_phase = feature.get("cur_phase") if isinstance(feature, Mapping) else None
            feature_index = _index_value(cur_phase[0]) if isinstance(cur_phase, (list, tuple)) and cur_phase else None
            if phase_name is None and feature_index is not None and 0 <= feature_index < len(control_phases):
                phase_name = control_phases[feature_index]
            if phase_name is None and self.allow_phase_fallback and control_phases:
                phase_name = control_phases[0]
            if phase_name is None:
                raise RuntimeError(
                    f"cannot determine current runtime phase for {intersection_id!r}; "
                    "SUMO phase index/mapping is missing or points to a transition"
                )
            result[intersection_id] = str(phase_name)
        return result

    def current_time(self) -> float:
        getter = getattr(self.env, "get_current_time", None)
        if not callable(getter):
            raise TypeError("environment does not expose get_current_time")
        return float(getter())

    def close(self) -> None:
        self.release_visual_renderer()
        if self.owns_env:
            closer = getattr(self.env, "close", None)
            if callable(closer):
                closer()


class SUMOEnvFactory:
    """Construct actor-local ``SUMOEnvAdapter`` instances from city configs."""

    def __init__(
        self,
        config_by_city: Mapping[str, Mapping[str, Any]],
        paths_by_city: Mapping[str, Mapping[str, Any]],
        work_root: str | Path,
        media_root: str | Path | None = None,
        *,
        repo_root: str | Path | None = None,
        decision_cycle_seconds: float = 30.0,
    ) -> None:
        self.config_by_city = {str(key): deepcopy(value) for key, value in config_by_city.items()}
        self.paths_by_city = {str(key): deepcopy(value) for key, value in paths_by_city.items()}
        self.work_root = Path(work_root)
        self.media_root = Path(media_root) if media_root is not None else self.work_root / "media"
        self.repo_root = Path(repo_root).resolve() if repo_root else None
        self.decision_cycle_seconds = float(decision_cycle_seconds)

    def __call__(self, city: str, seed: int, actor_id: str = "actor") -> SUMOEnvAdapter:
        city = str(city)
        if city not in self.config_by_city or city not in self.paths_by_city:
            raise KeyError(f"no SUMO configuration registered for city {city!r}")
        if self.repo_root is not None:
            import sys

            root = str(self.repo_root)
            if root not in sys.path:
                sys.path.insert(0, root)
        from utils.sumo_env import SUMOEnv

        work_dir = self.work_root / city / str(actor_id)
        work_dir.mkdir(parents=True, exist_ok=True)
        config = deepcopy(self.config_by_city[city])
        paths = deepcopy(self.paths_by_city[city])
        config.update({
            "USE_GUI": False,
            "SEED": int(seed),
            # Required for SUMOEnv to maintain v36_cycle_outbound_snapshots,
            # which feed local_coordination in the Stage 2 prompt.
            "ENABLE_NEW_COORDINATION": True,
            # MasterVisualCapture renders after completed ticks at
            # 5,10,15,20,25,30s. V36 must sample the very same ticks so a
            # selected RGB frame and its SUMO coordination target share one
            # frame_index and simulator state.
            "V9_TEMPORAL_FRAME_INTERVAL": 5,
            # Visual capture is owned explicitly below by the master adapter;
            # VLMOneLine is not instantiated in the online runtime.
            "ENABLE_VIDEO_SFT_EXTRACTION": False,
            "ENABLE_COUNTERFACTUAL_DISCHARGE_LOG": False,
            "RAISE_INNER_STEP_CALLBACK_ERRORS": True,
        })
        # A fixed configured TraCI port would make concurrent Ray actors
        # collide. SUMOEnv allocates an OS-level free port when this key is
        # absent; each actor also has its own work directory.
        config.pop("SUMO_PORT", None)
        paths["PATH_TO_WORK_DIRECTORY"] = str(work_dir)
        env = SUMOEnv(
            str(work_dir),
            str(work_dir),
            config,
            paths,
            config.get("INTER_PHASE_MAPPING", {}),
        )
        reset = getattr(env, "reset", None)
        if not callable(reset):
            raise TypeError("SUMOEnv does not expose reset")
        try:
            reset(use_gui=False, seed=int(seed), verbose=False)
        except TypeError:
            reset(use_gui=False, seed=int(seed))
        visual_capture_factory = None
        if "master" in str(actor_id).lower():
            media_dir = self.media_root / city / str(actor_id)
            visual_capture_factory = lambda: MasterVisualCapture(env, config, media_dir)
        return SUMOEnvAdapter(
            env,
            decision_cycle_seconds=self.decision_cycle_seconds,
            owns_env=True,
            visual_capture_factory=visual_capture_factory,
        )


__all__ = ["PHASE_NAMES", "SUMOEnvAdapter", "SUMOEnvFactory", "phase_name_to_action"]
