"""Building blocks for V35 Stage 2 online cooperative GRPO."""

from .mode_assignment import assignment_matrix, build_mode_assignment, validate_mode_assignment
from .online_reward import (
    compute_endpoint_scores, compute_online_reward, compute_queue_rewards,
    endpoint_score, mode_utility, reasoning_penalty,
)
from .online_rollout import (
    CityRolloutResult,
    CitySnapshot,
    IntersectionObservation,
    SimulatorAdapter,
    flatten_for_gdpo,
    rollout_city,
    rollout_one_assignment,
    rollout_one_assignment_temporal,
    validate_city_snapshot,
)
from .ray_rollout import (
    RayCityRolloutCoordinator,
    SUMOMasterActor,
    SUMORolloutActor,
    materialize_rollout_snapshots,
)
from .stage2_protocol import (
    DEFAULT_PROMPT,
    PHASES,
    DecisionResponse,
    assistant_prefix,
    build_stage2_prompt,
    build_stage2_prompt_from_cooperative_json, route_stage2_observation,
    parse_decision_response,
)
from .sumo_adapter import PHASE_NAMES, SUMOEnvAdapter, SUMOEnvFactory, phase_name_to_action
from .observation_builder import PHASE_MOVEMENTS, collect_observation_state, build_city_snapshot
from .verl_adapter import (
    OnlineVERLBatch,
    OnlineVERLCollector,
    records_for_verl,
    records_to_data_proto,
    specs_from_batch,
    validate_verl_records,
)
from .policy_client import OpenAICompatiblePolicy, make_openai_policy, make_policy_factory
from .online_runtime import OnlineBatchResult, OnlineCooperativeRuntime
from .online_runtime import OnlineBatchResult, OnlineCooperativeRuntime

__all__ = [
    "PHASES", "DEFAULT_PROMPT", "DecisionResponse", "CitySnapshot",
    "CityRolloutResult", "IntersectionObservation", "SimulatorAdapter", "flatten_for_gdpo",
    "assignment_matrix", "build_mode_assignment", "validate_mode_assignment",
    "build_stage2_prompt", "build_stage2_prompt_from_cooperative_json", "route_stage2_observation", "assistant_prefix", "parse_decision_response",
    "compute_online_reward", "compute_queue_rewards", "compute_endpoint_scores",
    "endpoint_score", "mode_utility", "reasoning_penalty",
    "rollout_city", "rollout_one_assignment", "rollout_one_assignment_temporal", "validate_city_snapshot", "PHASE_NAMES", "SUMOEnvAdapter",
    "SUMOEnvFactory", "phase_name_to_action",
    "PHASE_MOVEMENTS", "collect_observation_state", "build_city_snapshot",
    "RayCityRolloutCoordinator", "SUMOMasterActor", "SUMORolloutActor",
    "materialize_rollout_snapshots",
    "OnlineVERLBatch", "OnlineVERLCollector", "records_for_verl", "records_to_data_proto", "specs_from_batch",
    "validate_verl_records",
    "OpenAICompatiblePolicy", "make_openai_policy", "make_policy_factory",
    "OnlineBatchResult", "OnlineCooperativeRuntime",
    "OnlineBatchResult", "OnlineCooperativeRuntime",
]
