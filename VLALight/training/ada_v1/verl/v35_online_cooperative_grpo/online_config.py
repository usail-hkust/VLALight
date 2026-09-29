"""Configuration for the online SUMO sample streams.

The online GRPO input is a stream of mutable SUMO masters.  This module keeps
city registration, episode timing, train/validation seeds, and Ray resource
limits in one small, serializable configuration object.  It intentionally does
not contain perception JSON: perception is extracted from the live master at a
validated same-step snapshot.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class CityConfig:
    name: str
    scenario: str | None = None
    weight: float = 1.0
    num_intersections: int | None = None
    traffic_env_conf: Mapping[str, Any] = field(default_factory=dict)
    path_conf: Mapping[str, Any] = field(default_factory=dict)
    topology_path: str | None = None
    data_root: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("city name must be non-empty")
        if self.weight <= 0:
            raise ValueError("city weight must be positive")
        if self.num_intersections is not None and self.num_intersections <= 0:
            raise ValueError("num_intersections must be positive")


@dataclass(frozen=True)
class SplitConfig:
    cities: tuple[str, ...]
    batch_size: int
    seed_base: int
    seed_stride: int = 100_000
    stream_prefix: str = "online"
    initial_step_stride: int = 0
    persistent_stream_count: int | None = None

    def __post_init__(self) -> None:
        if not self.cities:
            raise ValueError("split must contain at least one city")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.seed_stride <= 0:
            raise ValueError("seed_stride must be positive")
        if self.initial_step_stride < 0:
            raise ValueError("initial_step_stride cannot be negative")
        if self.persistent_stream_count is not None and self.persistent_stream_count <= 0:
            raise ValueError("persistent_stream_count must be positive")


@dataclass(frozen=True)
class ResourceConfig:
    max_parallel_masters: int = 4
    ray_num_cpus_per_master: float = 1.0
    ray_num_cpus_per_rollout: float = 1.0
    snapshot_root: str = "runtime/v35_online_snapshots"

    def __post_init__(self) -> None:
        if self.max_parallel_masters <= 0:
            raise ValueError("max_parallel_masters must be positive")
        if self.ray_num_cpus_per_master <= 0 or self.ray_num_cpus_per_rollout <= 0:
            raise ValueError("Ray CPU reservations must be positive")


@dataclass(frozen=True)
class PolicyConfig:
    """OpenAI-compatible policy service used for online candidate rollouts."""

    base_url: str = "http://127.0.0.1:8088"
    model: str = ""
    max_tokens: int = 512
    timeout: float = 300.0
    api_key: str = "EMPTY"

    def __post_init__(self) -> None:
        if not self.base_url:
            raise ValueError("policy base_url must be non-empty")
        if self.max_tokens <= 0:
            raise ValueError("policy max_tokens must be positive")
        if self.timeout <= 0:
            raise ValueError("policy timeout must be positive")


@dataclass(frozen=True)
class OnlineConfig:
    cities: Mapping[str, CityConfig]
    train: SplitConfig
    val: SplitConfig
    episode_seconds: float = 3600.0
    decision_cycle_seconds: float = 30.0
    warmup_steps: int = 5
    num_rollouts: int = 6
    evaluation_decision_cycles: int = 3
    exploration_temperature: float = 0.7
    evaluation_temperature: float = 0.0
    val_metric_steps: int = 5
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)

    def __post_init__(self) -> None:
        if self.episode_seconds <= 0 or self.decision_cycle_seconds <= 0:
            raise ValueError("episode and decision cycle durations must be positive")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps cannot be negative")
        if self.num_rollouts <= 0 or self.num_rollouts % 2:
            raise ValueError("num_rollouts must be a positive even number")
        if self.evaluation_decision_cycles <= 0:
            raise ValueError("evaluation_decision_cycles must be positive")
        if self.exploration_temperature < 0 or self.evaluation_temperature < 0:
            raise ValueError("temperatures cannot be negative")
        if self.val_metric_steps <= 0:
            raise ValueError("val_metric_steps must be positive")
        for split in (self.train, self.val):
            unknown = set(split.cities) - set(self.cities)
            if unknown:
                raise ValueError(f"{split.stream_prefix} references unknown cities: {sorted(unknown)}")

    @property
    def valid_step_min(self) -> int:
        """First accepted step; step 5 itself is still warm-up."""
        return self.warmup_steps + 1


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _city_from_dict(name: str, value: Mapping[str, Any]) -> CityConfig:
    return CityConfig(
        name=str(name),
        scenario=value.get("scenario", name),
        weight=float(value.get("weight", 1.0)),
        num_intersections=(None if value.get("num_intersections") is None else int(value["num_intersections"])),
        traffic_env_conf=_mapping(value.get("traffic_env_conf")),
        path_conf=_mapping(value.get("path_conf")),
        topology_path=value.get("topology_path"),
        data_root=value.get("data_root"),
    )


def _split_from_dict(value: Mapping[str, Any], *, default_prefix: str, default_seed: int, default_batch: int) -> SplitConfig:
    return SplitConfig(
        cities=tuple(str(item) for item in value.get("cities", ("jinan", "hangzhou"))),
        batch_size=int(value.get("batch_size", default_batch)),
        seed_base=int(value.get("seed_base", default_seed)),
        seed_stride=int(value.get("seed_stride", 100_000)),
        stream_prefix=str(value.get("stream_prefix", default_prefix)),
        initial_step_stride=int(value.get("initial_step_stride", 0)),
        persistent_stream_count=(
            None if value.get("persistent_stream_count") is None
            else int(value["persistent_stream_count"])
        ),
    )


def online_config_from_dict(data: Mapping[str, Any]) -> OnlineConfig:
    city_data = data.get("cities", {})
    if not isinstance(city_data, Mapping) or not city_data:
        raise ValueError("online config must register at least one city")
    cities = {str(name): _city_from_dict(str(name), _mapping(value)) for name, value in city_data.items()}
    train = _split_from_dict(_mapping(data.get("train")), default_prefix="train", default_seed=10_001, default_batch=4)
    val = _split_from_dict(_mapping(data.get("val")), default_prefix="val", default_seed=20_001, default_batch=2)
    resources = _mapping(data.get("resources"))
    policy = _mapping(data.get("policy"))
    return OnlineConfig(
        cities=cities,
        train=train,
        val=val,
        episode_seconds=float(data.get("episode_seconds", 3600.0)),
        decision_cycle_seconds=float(data.get("decision_cycle_seconds", 30.0)),
        warmup_steps=int(data.get("warmup_steps", 5)),
        num_rollouts=int(data.get("num_rollouts", 6)),
        evaluation_decision_cycles=int(data.get("evaluation_decision_cycles", 3)),
        exploration_temperature=float(data.get("exploration_temperature", 0.7)),
        evaluation_temperature=float(data.get("evaluation_temperature", 0.0)),
        val_metric_steps=int(data.get("val_metric_steps", 5)),
        resources=ResourceConfig(
            max_parallel_masters=int(resources.get("max_parallel_masters", 4)),
            ray_num_cpus_per_master=float(resources.get("ray_num_cpus_per_master", 1.0)),
            ray_num_cpus_per_rollout=float(resources.get("ray_num_cpus_per_rollout", 1.0)),
            snapshot_root=str(resources.get("snapshot_root", "runtime/v35_online_snapshots")),
        ),
        policy=PolicyConfig(
            base_url=str(policy.get("base_url", "http://127.0.0.1:8088")),
            model=str(policy.get("model", "")),
            max_tokens=int(policy.get("max_tokens", 512)),
            timeout=float(policy.get("timeout", 300.0)),
            api_key=str(policy.get("api_key", "EMPTY")),
        ),
    )


def load_online_config(path: str | Path) -> OnlineConfig:
    """Load YAML or JSON without requiring a project-specific config class."""
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    if source.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - deployment dependency
            raise ImportError("PyYAML is required to load online YAML config") from exc
        data = yaml.safe_load(text)
    if not isinstance(data, Mapping):
        raise ValueError(f"online config root must be a mapping: {source}")
    return online_config_from_dict(data)


def default_online_config() -> OnlineConfig:
    """Build the initial Jinan/Hangzhou configuration from repository helpers."""
    try:
        from utils.vlm_config import SCENARIO_DATA_MAP, get_path_conf, get_traffic_env_conf
    except ImportError as exc:  # pragma: no cover - import path is deployment-specific
        raise ImportError("run from the repository root so utils.vlm_config is importable") from exc

    city_specs: dict[str, CityConfig] = {}
    for name, count in (("jinan", 12), ("hangzhou", 16)):
        traffic = get_traffic_env_conf(name, eightphase=False)
        paths = get_path_conf(name)
        scenario = SCENARIO_DATA_MAP[name]
        city_specs[name] = CityConfig(
            name=name,
            scenario=name,
            num_intersections=count,
            traffic_env_conf=traffic,
            path_conf=paths,
            topology_path=str(Path(paths["PATH_TO_DATA"]) / Path(scenario["topology_json"]).name),
        )
    return OnlineConfig(
        cities=city_specs,
        train=SplitConfig(("jinan", "hangzhou"), batch_size=1000, seed_base=10_001, stream_prefix="train"),
        # Formal validation uses 25 snapshots per city (50 total).
        val=SplitConfig(("jinan", "hangzhou"), batch_size=50, seed_base=20_001, stream_prefix="val"),
    )


def hydrate_repository_defaults(config: OnlineConfig) -> OnlineConfig:
    """Fill missing SUMO/path fields from the repository's city registry.

    YAML is intentionally allowed to contain only city names and topology
    metadata.  At runtime the existing ``utils.vlm_config`` remains the source
    of truth for network, route, and phase-mapping files.
    """
    try:
        from utils.vlm_config import SCENARIO_DATA_MAP, get_path_conf, get_traffic_env_conf
    except ImportError as exc:  # pragma: no cover - deployment-specific path
        raise ImportError("run from the repository root so utils.vlm_config is importable") from exc

    hydrated: dict[str, CityConfig] = {}
    for name, city in config.cities.items():
        scenario_name = city.scenario or name
        if scenario_name not in SCENARIO_DATA_MAP:
            hydrated[name] = city
            continue
        scenario = SCENARIO_DATA_MAP[scenario_name]
        traffic = dict(city.traffic_env_conf) or get_traffic_env_conf(scenario_name, eightphase=False)
        paths = dict(city.path_conf) or get_path_conf(scenario_name)
        if city.data_root:
            paths["PATH_TO_DATA"] = str(Path(city.data_root).expanduser().resolve())
        topology = city.topology_path
        if not topology:
            topology = str(Path(paths["PATH_TO_DATA"]) / Path(scenario["topology_json"]).name)
        hydrated[name] = CityConfig(
            name=city.name,
            scenario=scenario_name,
            weight=city.weight,
            num_intersections=city.num_intersections,
            traffic_env_conf=traffic,
            path_conf=paths,
            topology_path=topology,
            data_root=city.data_root,
        )
    return OnlineConfig(
        cities=hydrated,
        train=config.train,
        val=config.val,
        episode_seconds=config.episode_seconds,
        decision_cycle_seconds=config.decision_cycle_seconds,
        warmup_steps=config.warmup_steps,
        num_rollouts=config.num_rollouts,
        evaluation_decision_cycles=config.evaluation_decision_cycles,
        exploration_temperature=config.exploration_temperature,
        evaluation_temperature=config.evaluation_temperature,
        val_metric_steps=config.val_metric_steps,
        resources=config.resources,
        policy=config.policy,
    )


__all__ = [
    "CityConfig",
    "SplitConfig",
    "ResourceConfig",
    "PolicyConfig",
    "OnlineConfig",
    "online_config_from_dict",
    "load_online_config",
    "default_online_config",
    "hydrate_repository_defaults",
]
