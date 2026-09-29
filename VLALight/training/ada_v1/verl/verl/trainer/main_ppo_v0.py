# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other mpain.
"""

import os
import socket
import sys
from pathlib import Path

import ray
from omegaconf import OmegaConf

from verl.trainer.distillation import is_distillation_enabled
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.utils import create_rl_dataset, create_rl_sampler, need_critic, need_reference_policy
from verl.utils.config import validate_config


class BaseTaskRunner:
    def __init__(self):
        self.role_worker_mapping = {}
        self.mapping = {}

    def add_actor_rollout_worker(self, config):
        """Add actor rollout worker using the unified model engine implementation."""
        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.ppo.ray_trainer import Role
        from verl.workers.engine_workers import ActorRolloutRefWorker

        actor_rollout_cls = ActorRolloutRefWorker
        ray_worker_group_cls = RayWorkerGroup

        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None
        # Ref policy is fused into ActorRolloutRefWorker unless LoRA is used with a dedicated ref model.
        if need_reference_policy(config) and not ref_in_actor:
            role = Role.ActorRolloutRef
        else:
            role = Role.ActorRollout
        self.role_worker_mapping[role] = ray.remote(actor_rollout_cls)
        self.mapping[role] = "global_pool"
        return actor_rollout_cls, ray_worker_group_cls

    def add_critic_worker(self, config):
        """Add critic worker to role mapping using the unified model engine implementation."""
        from verl.trainer.ppo.ray_trainer import Role
        from verl.workers.engine_workers import TrainingWorker

        # The model-engine TrainingWorker handles all critic backends (fsdp/fsdp2/megatron/...)
        # internally based on ``config.critic.strategy``.
        self.role_worker_mapping[Role.Critic] = ray.remote(TrainingWorker)
        self.mapping[Role.Critic] = "global_pool"

    def init_resource_pool_mgr(self, config):
        """Initialize resource pool manager."""

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }

        if config.reward.reward_model.enable_resource_pool:
            if config.reward.reward_model.n_gpus_per_node <= 0:
                raise ValueError("config.reward.reward_model.n_gpus_per_node must be greater than 0")
            if config.reward.reward_model.nnodes <= 0:
                raise ValueError("config.reward.reward_model.nnodes must be greater than 0")

            reward_pool = [config.reward.reward_model.n_gpus_per_node] * config.reward.reward_model.nnodes
            resource_pool_spec["reward_pool"] = reward_pool
        else:
            config.reward.reward_model.nnodes = config.trainer.nnodes
            config.reward.reward_model.n_gpus_per_node = config.trainer.n_gpus_per_node

        distillation_config = config.get("distillation")
        if is_distillation_enabled(distillation_config):
            if distillation_config.n_gpus_per_node <= 0:
                raise ValueError("config.distillation.n_gpus_per_node must be greater than 0")
            if distillation_config.nnodes <= 0:
                raise ValueError("config.distillation.nnodes must be greater than 0")

            teacher_pool = [distillation_config.n_gpus_per_node] * distillation_config.nnodes
            resource_pool_spec["teacher_pool"] = teacher_pool

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=self.mapping)
        return resource_pool_manager

    def add_reward_model_resource_pool(self, config):
        """Add reward model worker if enabled."""
        from verl.trainer.ppo.ray_trainer import Role

        if config.reward.reward_model.enable:
            # we do not use reward model workers, so we only register reward model in resource pool
            # without continue to register reward model worker in role mapping
            if config.reward.reward_model.enable_resource_pool:
                self.mapping[Role.RewardModel] = "reward_pool"
            else:
                self.mapping[Role.RewardModel] = "global_pool"

    def add_teacher_model_resource_pool(self, config):
        """Add teacher model worker if enabled."""
        from verl.trainer.ppo.ray_trainer import Role

        if is_distillation_enabled(config.get("distillation")):
            # we do not use teacher model workers, so we only register teacher model in resource pool
            # without registering a teacher model worker in role-worker mapping
            self.mapping[Role.TeacherModel] = "teacher_pool"

    def add_ref_policy_worker(self, config, ref_policy_cls):
        """Ref policy is fused into ActorRolloutRefWorker in the unified model engine.

        Kept for backward compatibility with subclasses that still invoke it; the method
        is now a no-op because the reference policy lives on the same worker group as
        the actor/rollout.
        """
        return

    def run(self, config):
        pass


@ray.remote
class TaskRunner(BaseTaskRunner):
    """Ray remote class for executing distributed PPO training tasks.

    This class encapsulates the main training logic and runs as a Ray remote actor
    to enable distributed execution across multiple nodes and GPUs.

    Attributes:
        role_worker_mapping: Dictionary mapping Role enums to Ray remote worker classes
        mapping: Dictionary mapping Role enums to resource pool IDs for GPU allocation
    """

    def __init__(self):
        super().__init__()

    def run(self, config):
        """Execute the main PPO training workflow.

        This method sets up the distributed training environment, initializes
        workers, datasets, and reward functions, then starts the training process.

        Args:
            config: Training configuration object containing all parameters needed
                   for setting up and running the PPO training process.
        """
        # Print the initial configuration. `resolve=True` will evaluate symbolic values.
        from pprint import pprint

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)

        self.add_reward_model_resource_pool(config)

        self.add_teacher_model_resource_pool(config)

        # Add a reference policy worker if KL loss or KL reward is used.
        self.add_ref_policy_worker(config, actor_rollout_cls)

        # validate config
        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(config),
            use_critic=need_critic(config),
        )

        # Instantiate the tokenizer and processor from the model config.
        from verl.utils.config import omega_conf_to_dataclass
        from verl.workers.config import HFModelConfig

        model_config: HFModelConfig = omega_conf_to_dataclass(config.actor_rollout_ref.model)
        tokenizer = model_config.tokenizer
        # Used for multimodal LLM, could be None
        processor = model_config.processor

        resource_pool_manager = self.init_resource_pool_mgr(config)

        from verl.utils.dataset.rl_dataset import collate_fn

        # Create training and validation datasets.
        train_dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            is_train=True,
            max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = create_rl_dataset(
            config.data.val_files,
            config.data,
            tokenizer,
            processor,
            is_train=False,
            max_samples=config.data.get("val_max_samples", -1),
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)

        # Initialize the PPO trainer.
        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        # Initialize the workers of the trainer.
        trainer.init_workers()

        # Attach the live SUMO/Ray collector before entering the training loop.
        # The bootstrap JSONL rows only provide persistent stream identities;
        # prompts, six candidates, and rewards are materialized from SUMO here.
        runtime = None
        val_runtime = None
        try:
            online_cfg_node = config.get("online_cooperative", {})
            if bool(online_cfg_node.get("enabled", False)):
                online_root = next(
                    (
                        path
                        for path in Path(__file__).resolve().parents
                        if (path / "v35_online_cooperative_grpo").is_dir()
                    ),
                    None,
                )
                if online_root is None:
                    raise RuntimeError("cannot locate v35_online_cooperative_grpo")
                if str(online_root) not in sys.path:
                    sys.path.insert(0, str(online_root))

                from v35_online_cooperative_grpo import (
                    OnlineCooperativeRuntime,
                    OnlineVERLCollector,
                    RayCityRolloutCoordinator,
                    SUMOEnvFactory,
                    build_city_snapshot,
                    make_policy_factory,
                )
                from v35_online_cooperative_grpo.online_config import (
                    hydrate_repository_defaults,
                    load_online_config,
                )

                online_config_path = online_cfg_node.get("config_path")
                if not online_config_path:
                    raise ValueError("online_cooperative.config_path is required when enabled")
                # ``hydrate_repository_defaults`` imports the repository's
                # existing ``utils.vlm_config``.  Add the repository root
                # before loading the YAML so this works from any launch cwd.
                repo_root = next(
                    (
                        path
                        for path in online_root.parents
                        if (path / "utils").is_dir() and (path / "data").is_dir()
                    ),
                    online_root.parent,
                )
                if str(repo_root) not in sys.path:
                    sys.path.insert(0, str(repo_root))
                online_cfg = hydrate_repository_defaults(load_online_config(online_config_path))

                config_by_city = {
                    name: city.traffic_env_conf
                    for name, city in online_cfg.cities.items()
                }
                paths_by_city = {
                    name: city.path_conf
                    for name, city in online_cfg.cities.items()
                }
                runtime_root = Path(
                    os.environ.get("V35_RUNTIME_ROOT", online_root / "runtime")
                )
                env_factory = SUMOEnvFactory(
                    config_by_city,
                    paths_by_city,
                    work_root=runtime_root / "sumo_work",
                    media_root=runtime_root / "media",
                    repo_root=repo_root,
                    decision_cycle_seconds=online_cfg.decision_cycle_seconds,
                )
                coordinator = RayCityRolloutCoordinator(
                    env_factory,
                    num_rollouts=online_cfg.num_rollouts,
                )

                def master_factory(city, seed, actor_id):
                    return coordinator.create_master_actor(city, seed, actor_id=actor_id)

                def observation_builder(master, city, step):
                    return build_city_snapshot(
                        master,
                        city,
                        step,
                        routes_dir=online_root / "artifacts",
                    )

                policy_factory = make_policy_factory(
                    base_url=online_cfg.policy.base_url,
                    model=online_cfg.policy.model,
                    temperature=online_cfg.exploration_temperature,
                    max_tokens=online_cfg.policy.max_tokens,
                    timeout=online_cfg.policy.timeout,
                    api_key=online_cfg.policy.api_key,
                )
                runtime = OnlineCooperativeRuntime(
                    online_cfg,
                    split="train",
                    master_factory=master_factory,
                    coordinator=coordinator,
                    observation_builder=observation_builder,
                    policy_factory=policy_factory,
                    prompt_template=None,
                    snapshot_root=runtime_root / "snapshots",
                )
                trainer.set_online_cooperative_collector(
                    OnlineVERLCollector(runtime, config=online_cfg, split="train")
                )
                # Validation must use a separate master-stream manager.  It
                # receives the same model and SUMO factory, but owns distinct
                # persistent masters and therefore never advances training
                # streams while collecting validation candidates.
                val_runtime = OnlineCooperativeRuntime(
                    online_cfg,
                    split="val",
                    master_factory=master_factory,
                    coordinator=coordinator,
                    observation_builder=observation_builder,
                    policy_factory=policy_factory,
                    prompt_template=None,
                    snapshot_root=runtime_root / "snapshots",
                )
                trainer.set_online_cooperative_val_collector(
                    OnlineVERLCollector(val_runtime, config=online_cfg, split="val")
                )

            # Start the training process.
            trainer.fit()
        finally:
            if val_runtime is not None:
                val_runtime.close()
            if runtime is not None:
                runtime.close()
