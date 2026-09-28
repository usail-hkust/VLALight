import json
import os
import time
import threading
from copy import deepcopy
from typing import List, Dict, Any, Tuple

import networkx as nx
import numpy as np
import wandb
from tqdm import tqdm

from .config import DIC_AGENTS
from .data_utils import merge, get_state, get_state_detail, eight_phase_list, dump_json
from .get_traffic_condition import get_lane_traffic_conditions
from .sumo_env import SUMOEnv
from .v30_contract import (
    DEFAULT_ROLLOUT_WORKERS,
    validate_standard_phase_order,
)
from .vehicle_position_snapshot import VehiclePositionSnapshotWriter
from .v32_coordination import merge_missing_boundary_coordination
from .v36_coordination import V36CoordinationManager
# Implementation note.

# Implementation note.
import shutil

def path_check(dic_path):
    """Create output directories when needed."""
    if os.path.exists(dic_path["PATH_TO_WORK_DIRECTORY"]):
        if dic_path["PATH_TO_WORK_DIRECTORY"] != "records/default":
            pass  # Implementation note.
    else:
        os.makedirs(dic_path["PATH_TO_WORK_DIRECTORY"])
    
    if not os.path.exists(dic_path["PATH_TO_MODEL"]):
        os.makedirs(dic_path["PATH_TO_MODEL"])

def copy_conf_file(dic_path, dic_agent_conf, dic_traffic_env_conf, path=None):
    """Copy configuration files to the work directory."""
    if path is None:
        path = dic_path["PATH_TO_WORK_DIRECTORY"]
    json.dump(dic_agent_conf, open(os.path.join(path, "agent.conf"), "w"), indent=4)
    json.dump(dic_traffic_env_conf, open(os.path.join(path, "traffic_env.conf"), "w"), indent=4)

def copy_cityflow_file(dic_path, dic_traffic_env_conf, path=None):
    """Copy traffic flow file to the work directory."""
    if path is None:
        path = dic_path["PATH_TO_WORK_DIRECTORY"]
    traffic_file_path = os.path.join(dic_path["PATH_TO_DATA"], dic_traffic_env_conf["TRAFFIC_FILE"])
    if os.path.exists(traffic_file_path):
        shutil.copy(traffic_file_path, os.path.join(path, dic_traffic_env_conf["TRAFFIC_FILE"]))


class OneLine:
    """
    Main training class for traffic light control using various AI agents.
    Manages the simulation environment, agents, and training process.
    """

    def __init__(self, dic_agent_conf: Dict[str, Any], dic_traffic_env_conf: Dict[str, Any],
                 dic_path: Dict[str, str], roadnet: str, trafficflow: str):
        """
        Initialize the OneLine training system.

        Args:
            dic_agent_conf: Agent configuration dictionary
            dic_traffic_env_conf: Traffic environment configuration dictionary
            dic_path: Path configuration dictionary
            roadnet: Road network identifier
            trafficflow: Traffic flow identifier
        """
        self.dic_agent_conf = dic_agent_conf
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_path = dic_path
        self.roadnet = roadnet
        self.trafficflow = trafficflow

        # Initialize containers
        self.agents: List[Any] = []
        self.env: SUMOEnv = None
        self.models: List[Any] = []
        self._network_topology: Dict[str, Any] = {}
        self.vehicle_position_snapshot_writer = None
        self.renderer = None
        self.image_saver = None
        self.tls_ids: List[str] = []
        self.decision_window_recorder = None
        self.video_sft_dataset_writer = None
        self._video_interval_active = False
        self._video_interval_start_s = None
        self._snapshot_serial = 0
        self._v28_rollout_pool = None
        self._v28_phase_histories = {}
        self._v30_rollout_pool = None
        self._v30_phase_histories = {}
        self._v36_coordination_manager = None

        # Setup the environment and agents
        self.initialize()

    def initialize(self) -> None:
        """
        Initialize the training environment and create agents for each intersection.
        Sets up file paths, copies necessary configuration files, and creates the simulation environment.
        """
        # Setup directories and copy configuration files
        path_check(self.dic_path)
        copy_conf_file(self.dic_path, self.dic_agent_conf, self.dic_traffic_env_conf)
        copy_cityflow_file(self.dic_path, self.dic_traffic_env_conf)

        # Initialize the CityFlow environment
        inter_phase_mapping = self.dic_traffic_env_conf.get('INTER_PHASE_MAPPING', {})
        self.env = SUMOEnv(
            path_to_log=self.dic_path["PATH_TO_WORK_DIRECTORY"],
            path_to_work_directory=self.dic_path["PATH_TO_WORK_DIRECTORY"],
            dic_traffic_env_conf=self.dic_traffic_env_conf,
            dic_path=self.dic_path,
            inter_phase_mapping=inter_phase_mapping
        )
        # Use GUI if specified in config
        use_gui = self.dic_traffic_env_conf.get("USE_GUI", False)
        self.env.reset(use_gui=use_gui, seed=self.dic_traffic_env_conf.get("SEED"))

        # Create assets
        self._load_network_topology()
        self.tls_ids = [inter.inter_id for inter in self.env.list_intersection]
        self._create_agents()
        self._create_road_network()
        self._init_vehicle_position_snapshot_writer()
        self._init_perception_image_capture()

    def _init_vehicle_position_snapshot_writer(self) -> None:
        if not self.dic_traffic_env_conf.get("ENABLE_VEHICLE_POSITION_SNAPSHOT", False):
            self.vehicle_position_snapshot_writer = None
            return
        output_dir = os.path.join(
            self.dic_path["PATH_TO_WORK_DIRECTORY"],
            self.dic_traffic_env_conf.get("VEHICLE_POSITION_SNAPSHOT_DIR", "vehicle_position_snapshots"),
        )
        view_distance_m = float(self.dic_traffic_env_conf.get("CAMERA_VIEW_DISTANCE", 150.0))
        self.vehicle_position_snapshot_writer = VehiclePositionSnapshotWriter(
            output_dir=output_dir,
            view_distance_m=view_distance_m,
        )
        print(f"VehiclePositionSnapshot: {output_dir} (view_distance={view_distance_m}m)")

    def _init_perception_image_capture(self) -> None:
        enable_snapshots = bool(
            self.dic_traffic_env_conf.get("ENABLE_VEHICLE_POSITION_SNAPSHOT", False)
        )
        enable_video = bool(
            self.dic_traffic_env_conf.get("ENABLE_VIDEO_SFT_EXTRACTION", False)
        )
        if not (enable_snapshots or enable_video):
            return
        vlm_config = self.dic_traffic_env_conf.get("VLM_CONFIG", {})
        if not vlm_config:
            raise RuntimeError(
                "VLM_CONFIG missing; perception/video capture cannot be enabled"
            )
        try:
            from .image_saver import ImageSaver
            from TransSimHub.tshub.tshub_env3d.vis3d_renderer.tshub_render import TSHubRenderer
        except Exception as exc:
            raise RuntimeError(f"image snapshot dependencies unavailable: {exc}") from exc

        direction_mapping_path = vlm_config.get("DIRECTION_MAPPING_PATH")
        scenario_glb_dir = vlm_config.get("SCENARIO_GLB_DIR")
        if not scenario_glb_dir or not os.path.exists(scenario_glb_dir):
            raise RuntimeError(f"SCENARIO_GLB_DIR invalid; image snapshots disabled: {scenario_glb_dir}")

        self.image_saver = ImageSaver(
            scenario=vlm_config.get("SCENARIO", "jinan"),
            direction_mapping_path=direction_mapping_path,
            base_dir=self.dic_path["PATH_TO_WORK_DIRECTORY"],
            session_id="",
            verbose=False,
            enable_preprocess=vlm_config.get("IMAGE_PREPROCESS_ENABLED", True),
            left_crop=vlm_config.get("IMAGE_PREPROCESS_LEFT_CROP", 0.40),
            right_crop=vlm_config.get("IMAGE_PREPROCESS_RIGHT_CROP", 0.30),
            scale_mode=vlm_config.get("IMAGE_PREPROCESS_SCALE_MODE", "fit_width"),
            strict=enable_video,
        )

        tls_sensor_type = vlm_config.get("TLS_SENSOR_TYPE", "junction_front_all")
        sensor_config = {
            "tls": {
                inter.inter_id: {
                    "sensor_types": [tls_sensor_type],
                    "tls_camera_height": vlm_config.get("TLS_CAMERA_HEIGHT", 30),
                }
                for inter in self.env.list_intersection
            }
        }
        netxml_path = os.path.join(
            self.dic_path.get("PATH_TO_DATA", self.dic_path["PATH_TO_WORK_DIRECTORY"]),
            self.dic_traffic_env_conf.get("ROADNET_FILE", ""),
        )
        self.renderer = TSHubRenderer(
            simid="sumo",
            sensor_config=sensor_config,
            preset=vlm_config.get("RENDER_PRESET", "1080P"),
            resolution=vlm_config.get("RENDER_RESOLUTION", 1.0),
            scenario_glb_dir=scenario_glb_dir,
            vehicle_model=vlm_config.get("VEHICLE_MODEL", "low"),
            render_mode="offscreen",
            rendering_backend=vlm_config.get("RENDERING_BACKEND", "pandagl"),
            show_buildings=vlm_config.get("SHOW_BUILDINGS", False),
            tls_batch_size=vlm_config.get("TLS_BATCH_SIZE", None),
            keep_batch_sensors=vlm_config.get("RENDER_KEEP_BATCH_SENSORS", True),
            reuse_batch_sensors=vlm_config.get("RENDER_REUSE_BATCH_SENSORS", False),
            step_task_manager=vlm_config.get("RENDER_STEP_TASK_MANAGER", True),
            netxml_path=netxml_path if os.path.exists(netxml_path) else None,
            show_arrows=vlm_config.get("SHOW_ARROWS", True),
        )
        init_obs = self.env.get_tls_init_info(tls_ids=self.tls_ids)
        self.renderer.reset(init_obs)
        self._validate_perception_renderer_ready("initial")
        print(f"Image snapshots: {self.image_saver.session_dir}")

        if enable_video and vlm_config.get("DECISION_INPUT_MODE", "image") == "video":
            from .decision_window_recorder import DecisionWindowRecorder

            self.decision_window_recorder = DecisionWindowRecorder(
                session_dir=self.dic_path["PATH_TO_WORK_DIRECTORY"],
                direction_mapping=self.image_saver.direction_mapping,
                fps=int(vlm_config.get("VIDEO_FPS", 1)),
                add_labels=bool(vlm_config.get("VIDEO_ADD_LABELS", True)),
                export_direction_videos=bool(
                    vlm_config.get("VIDEO_EXPORT_DIRECTIONS", False)
                ),
                export_composite_video=bool(
                    vlm_config.get("VIDEO_EXPORT_COMPOSITE", True)
                ),
                export_direction_sequence=bool(
                    vlm_config.get("VIDEO_EXPORT_DIRECTION_SEQUENCE", False)
                ),
                verbose=False,
                enable_preprocess=vlm_config.get(
                    "VIDEO_PREPROCESS_ENABLED",
                    vlm_config.get("IMAGE_PREPROCESS_ENABLED", True),
                ),
                left_crop=vlm_config.get("IMAGE_PREPROCESS_LEFT_CROP", 0.40),
                right_crop=vlm_config.get("IMAGE_PREPROCESS_RIGHT_CROP", 0.30),
                scale_mode=vlm_config.get("IMAGE_PREPROCESS_SCALE_MODE", "fit_width"),
                frame_view=str(vlm_config.get("VIDEO_FRAME_VIEW", "legacy_crop")),
                tile_width=int(vlm_config.get("VIDEO_TILE_WIDTH", 768)),
                sensor_type=vlm_config.get("TLS_SENSOR_TYPE", "junction_front_all"),
                record_mode=str(vlm_config.get("VIDEO_RECORD_MODE", "sampled")),
                sim_interval=float(
                    self.dic_traffic_env_conf.get("INTERVAL", 1.0) or 1.0
                ),
                sample_interval=float(
                    vlm_config.get("VIDEO_FRAME_SAMPLE_INTERVAL", 5.0) or 5.0
                ),
                preprocess_interpolation=vlm_config.get(
                    "VIDEO_PREPROCESS_INTERPOLATION", "linear"
                ),
                video_codec=vlm_config.get("VIDEO_CODEC", "mp4v"),
                video_extension=vlm_config.get("VIDEO_EXTENSION", "mp4"),
                async_video_write=vlm_config.get("VIDEO_ASYNC_WRITE", True),
                async_video_workers=vlm_config.get("VIDEO_ASYNC_WORKERS", 4),
                async_video_max_pending_writes=vlm_config.get(
                    "VIDEO_ASYNC_MAX_PENDING_WRITES"
                ),
            )

        if enable_video:
            from .video_sft_dataset_writer import VideoSFTDatasetWriter

            self.video_sft_dataset_writer = VideoSFTDatasetWriter(
                output_dir=os.path.join(
                    self.dic_path["PATH_TO_WORK_DIRECTORY"], "video_sft_raw"
                ),
                view_distance_m=float(
                    self.dic_traffic_env_conf.get("CAMERA_VIEW_DISTANCE", 150.0)
                ),
                video_sample_interval_s=float(
                    vlm_config.get("VIDEO_FRAME_SAMPLE_INTERVAL", 5.0)
                ),
                sim_interval_s=float(
                    self.dic_traffic_env_conf.get("INTERVAL", 1.0)
                ),
                network_topology=self._network_topology,
                coordination_speed_mps=float(
                    self.dic_traffic_env_conf.get(
                        "V36_COORDINATION_SPEED_MPS", 11.0
                    )
                ),
                local_render_radius_m=vlm_config.get("LOCAL_RENDER_RADIUS_M"),
                save_coordination_frames=bool(
                    vlm_config.get("VIDEO_SAVE_COORDINATION_FRAMES", False)
                ),
            )
            print(
                "V30/V34-style video recording enabled: "
                f"{self.dic_path['PATH_TO_WORK_DIRECTORY']}"
            )

    def _reset_perception_renderer(self) -> None:
        if self.renderer is None:
            return
        tls_ids = [inter.inter_id for inter in self.env.list_intersection]
        init_obs = self.env.get_tls_init_info(tls_ids=tls_ids)
        self.renderer.reset(init_obs)
        self._validate_perception_renderer_ready("train_reset")

    def _validate_perception_renderer_ready(self, stage: str) -> None:
        if self.renderer is None:
            return
        batch_info = self.renderer.get_batch_info() if hasattr(self.renderer, "get_batch_info") else {}
        initialized_tls = int(batch_info.get("initialized_tls", 0))
        total_tls = int(batch_info.get("total_tls", 0))
        if total_tls <= 0 or initialized_tls <= 0:
            raise RuntimeError(f"renderer TLS sensors not initialized at {stage}: {batch_info}")

    def _load_network_topology(self) -> None:
        """Load optional intersection-level topology for coordination agents."""
        topology_path = self.dic_traffic_env_conf.get("NETWORK_TOPOLOGY_PATH")
        if not topology_path:
            topology_path = os.path.join(self.dic_path.get("PATH_TO_DATA", ""), "network_topology.json")
        if not topology_path or not os.path.exists(topology_path):
            self._network_topology = {}
            return
        try:
            with open(topology_path, "r", encoding="utf-8") as f:
                self._network_topology = json.load(f)
            print(f"Loaded network topology from: {topology_path}")
        except Exception as e:
            print(f"Warning: failed to load network topology {topology_path}: {e}")
            self._network_topology = {}

    def _get_v36_coordination_manager(self):
        if self._v36_coordination_manager is None:
            model_name = self.dic_traffic_env_conf.get("MODEL_NAME", "V36").lower()
            self._v36_coordination_manager = V36CoordinationManager(
                self._network_topology,
                self.dic_path["PATH_TO_WORK_DIRECTORY"],
                speed_mps=self.dic_traffic_env_conf.get(
                    "V36_COORDINATION_SPEED_MPS", 11.0),
                view_distance_m=self.dic_traffic_env_conf.get(
                    "CAMERA_VIEW_DISTANCE", 150.0),
                log_filename=f"{model_name}_coordination_debug.jsonl",
            )
        return self._v36_coordination_manager

    def _create_agents(self) -> None:
        """Create appropriate agents based on the model configuration."""
        agent_name = self.dic_traffic_env_conf["MODEL_NAME"]
        num_intersections = self.dic_traffic_env_conf['NUM_INTERSECTIONS']

        for intersection_idx in range(num_intersections):
            intersection = self.env.list_intersection[intersection_idx]
            intersection_name = intersection.inter_name
            phase_count = len(intersection.phases)

            # Create agent based on model type
            if "ChatGPT" in agent_name:
                agent = self._create_chatgpt_agent(agent_name, intersection_name, phase_count)
            elif "open_llm" in agent_name:
                agent = self._create_open_llm_agent(agent_name, intersection_name, phase_count)
            else:
                agent = self._create_default_agent(agent_name, intersection_idx, intersection)

            self.agents.append(agent)

    def _create_road_network(self) -> None:
        """Constructs the directed road network graph from the environment configuration."""
        num_intersections = self.dic_traffic_env_conf['NUM_INTERSECTIONS']
        L_G = nx.DiGraph()
        LI_G = nx.DiGraph()

        for idx in range(num_intersections):
            intersection = self.env.list_intersection[idx]
            name = intersection.inter_name
            roads = self.env.intersection_dict[name]['roads']

            for road_id, road_info in roads.items():
                if road_info['type'] == 'outgoing':
                    for direction, lane_list in road_info['lanes'].items():
                        for lane in lane_list:
                            LI_G.add_edge(name, f"{road_id}_{lane}")
                else:
                    for direction, lane_list in road_info['lanes'].items():
                        for start_lane in lane_list:
                            next_road_id = road_info[direction]
                            next_road = roads[next_road_id]
                            LI_G.add_edge(f"{road_id}_{start_lane}", name)

                            for next_direction, next_lanes in next_road['lanes'].items():
                                for end_lane in next_lanes:
                                    L_G.add_edge(f"{road_id}_{start_lane}", f"{next_road_id}_{end_lane}")

        self.lane_graph = L_G
        self.lane_inter_graph = LI_G

    def _create_chatgpt_agent(self, agent_name: str, intersection_name: str, phase_count: int) -> Any:
        """Create a ChatGPT-based agent."""
        return DIC_AGENTS[agent_name.split("-")[0]](
            GPT_version=self.dic_agent_conf["GPT_VERSION"],
            intersection=self.env.intersection_dict[intersection_name],
            inter_name=intersection_name,
            phase_num=phase_count,
            log_dir=self.dic_agent_conf["LOG_DIR"],
            dataset=f"{self.roadnet}-{self.trafficflow}"
        )

    def _create_open_llm_agent(self, agent_name: str, intersection_name: str, phase_count: int) -> Any:
        """Create an open LLM-based agent."""
        return DIC_AGENTS[agent_name.split("-")[0]](
            ex_api=self.dic_agent_conf["WITH_EXTERNAL_API"],
            model=agent_name.split("-")[1],
            intersection=self.env.intersection_dict[intersection_name],
            inter_name=intersection_name,
            phase_num=phase_count,
            log_dir=self.dic_agent_conf["LOG_DIR"],
            dataset=f"{self.roadnet}-{self.trafficflow}"
        )

    def _create_default_agent(self, agent_name: str, intersection_idx: int, intersection: Any = None) -> Any:
        """Create a default (non-LLM) agent."""
        # Prepare extra parameters for agents that need phase information
        # All baseline agents need control_phases for correct phase selection
        extra_params = {}
        if agent_name in ['MaxPressure', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V20', 'V21', 'V24', 'V25', 'V26', 'V27', 'V28', 'V29', 'V30', 'V31', 'V32', 'V33', 'V36', 'AdvancedMaxPressure', 'EfficientMaxPressure', 'Random', 'Fixedtime', 'Webster']:
            if intersection is not None and hasattr(intersection, 'control_phases'):
                extra_params['control_phases'] = intersection.control_phases
        
        return DIC_AGENTS[agent_name](
            dic_agent_conf=self.dic_agent_conf,
            dic_traffic_env_conf=self.dic_traffic_env_conf,
            dic_path=self.dic_path,
            cnt_round=0,
            intersection_id=str(intersection_idx),
            **extra_params
        )

    def train(self, round: int) -> Dict[str, float]:
        """
        Execute one training round of the traffic simulation.

        Args:
            round: Current training round number

        Returns:
            Dictionary containing training results and metrics
        """
        print("================ start train ================")

        # Initialize training parameters
        total_run_cnt = self.dic_traffic_env_conf["RUN_COUNTS"]
        memory_file_path = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "memories.txt")

        # Reset environment and initialize tracking variables
        use_gui = self.dic_traffic_env_conf.get("USE_GUI", False)
        state = self.env.reset(use_gui=use_gui, seed=self.dic_traffic_env_conf.get("SEED"))
        if self.dic_traffic_env_conf.get("MODEL_NAME") == "V28":
            self._v28_phase_histories = {}
        if self.dic_traffic_env_conf.get("MODEL_NAME") == "V30":
            self._v30_phase_histories = {}
        if self.dic_traffic_env_conf.get("MODEL_NAME") in ("V32", "V36"):
            self._v36_coordination_manager = None
        self._reset_perception_renderer()
        training_metrics = self._initialize_training_metrics()
        self._next_queue_metric_time = None
        state_action_log = [[] for _ in range(len(state))]
        lane_state_log = {lane: [] for lane in self.lane_graph.nodes}

        print("end reset")

        # Setup logging
        logger = self._setup_wandb_logger(round)

        results = {}  # Implementation note.
        start_time = time.time()

        # Main training loop
        current_time = self.env.get_current_time()
        step_num = 0
        done = False
        overall_results = {"avg_travel_time": [], "avg_queue_len": [], "avg_waiting_time": []}

        with tqdm(total=total_run_cnt, desc="Simulation Progress", unit="s") as pbar:
            previous_time = 0
            while not done and current_time < total_run_cnt:
                # Get actions from all agents
                if self.dic_traffic_env_conf.get("MODEL_NAME") == "V28":
                    self._log_all_intersection_states(state_action_log)
                    action_list = self._get_v28_rollout_actions(state, step_num)
                elif self.dic_traffic_env_conf.get("MODEL_NAME") == "V30":
                    self._log_all_intersection_states(state_action_log)
                    action_list = self._get_v30_rollout_actions(state, step_num)
                else:
                    action_list = self._get_agent_actions(
                        state, step_num, state_action_log)
                if self.dic_traffic_env_conf.get("ENABLE_COUNTERFACTUAL_DISCHARGE_LOG", False):
                    cf_discharge = self.env.counterfactual_discharge_audit(
                        base_action_dict=action_list,
                        min_action_time=self.dic_traffic_env_conf.get("MIN_ACTION_TIME", 15))
                    self._attach_counterfactual_discharge(cf_discharge)

                # Get lane state
                for lane in self.lane_graph.nodes:
                    lane_state_log[lane].append(get_lane_traffic_conditions(lane, self.env))

                # Execute actions in environment
                action_time = self.dic_traffic_env_conf.get("MIN_ACTION_TIME", 15)
                self._begin_video_interval(step_num, current_time)
                try:
                    next_state, reward, done, _ = self.env.step(
                        action_list,
                        min_action_time=action_time,
                        inner_step_callback=self._build_perception_snapshot_callback(
                            step_num
                        ),
                    )
                except BaseException as exc:
                    self._finalize_video_interval(
                        self.env.get_current_time(),
                        status="incomplete",
                        error=f"{type(exc).__name__}: {exc}",
                        force_discard=True,
                    )
                    raise

                self._finalize_video_interval(self.env.get_current_time())

                # Log actions and update metrics
                self._log_step_data(state_action_log, action_list, memory_file_path,
                                    current_time, state, reward)
                self._update_training_metrics(training_metrics, reward)

                # Prepare for next step
                state = next_state
                step_num += 1
                current_time = self.env.get_current_time()

                # Update progress bar based on simulation time progressed
                time_delta = current_time - previous_time
                if time_delta > 0:
                    pbar.update(time_delta)
                    previous_time = current_time

                # Log intermediate results periodically (every 3600 seconds of simulation)
                if current_time > 0 and int(current_time) % 3600 == 0 and int(previous_time) < int(current_time):
                    intermediate_results = self._calculate_final_results(training_metrics)
                    # Log with timestep for monitoring
                    log_dict = {
                        "simulation_time": current_time,
                        "step_num": step_num,
                        "avg_travel_time_interim": intermediate_results["avg_travel_time"],
                        "avg_queue_len_interim": intermediate_results["avg_queue_len"],
                        "avg_waiting_time_interim": intermediate_results["avg_waiting_time"]
                    }
                    logger.log(log_dict)
                    overall_results["avg_travel_time"].append(intermediate_results["avg_travel_time"])
                    overall_results["avg_queue_len"].append(intermediate_results["avg_queue_len"])
                    overall_results["avg_waiting_time"].append(intermediate_results["avg_waiting_time"])

        for agent, final_state in zip(self.agents, state):
            flush_log = getattr(agent, "flush_pending_decision_log", None)
            if callable(flush_log):
                flush_log(final_state)

        # Calculate final results and log
        # Ensure we have at least one results entry
        if not overall_results["avg_travel_time"]:
            results = self._calculate_final_results(training_metrics)
            logger.log(results)
            overall_results["avg_travel_time"].append(results["avg_travel_time"])
            overall_results["avg_queue_len"].append(results["avg_queue_len"])
            overall_results["avg_waiting_time"].append(results["avg_waiting_time"])
            
        self._save_training_data(state_action_log, lane_state_log, training_metrics['global_waiting_times'])
        final_metrics = {
            "avg_travel_time": float(np.mean(overall_results["avg_travel_time"])),
            "avg_queue_len": float(np.mean(overall_results["avg_queue_len"])),
            "avg_waiting_time": float(np.mean(overall_results["avg_waiting_time"])),
        }
        self._save_final_metrics(final_metrics, step_num, current_time)
        print(f"============== End Testing ==============")
        print(f"Average Travel Time: {final_metrics['avg_travel_time']}\n"
              f"Average Queue Length: {final_metrics['avg_queue_len']}\n"
              f"Average Waiting Time: {final_metrics['avg_waiting_time']}")
        logger.log({"overall_avg_travel_time": final_metrics["avg_travel_time"],
                    "overall_avg_queue_len": final_metrics["avg_queue_len"],
                    "overall_avg_waiting_time": final_metrics["avg_waiting_time"]})
        # Upload Wandb
        wandb.finish()

        print("Training time: ", time.time() - start_time)
        self.env.batch_log()
        self._close_v28_rollout_pool()
        self._close_v30_rollout_pool()
        self._close_perception_image_capture()

        return final_metrics

    def _log_all_intersection_states(self, state_action_log):
        for intersection_idx in range(len(self.env.list_intersection)):
            self._log_intersection_state(intersection_idx, state_action_log)

    def _get_v28_rollout_actions(self, state, step_num):
        if self._v28_rollout_pool is None:
            from .v28_rollout import V28RolloutPool
            rollout_dir = os.path.join(
                self.dic_path["PATH_TO_WORK_DIRECTORY"], "v28_rollouts")
            self._v28_rollout_pool = V28RolloutPool(
                config=self.dic_traffic_env_conf,
                path_config=self.dic_path,
                work_dir=rollout_dir,
                worker_count=int(self.dic_traffic_env_conf.get(
                    "V28_ROLLOUT_WORKERS", 4)),
                timeout_s=float(self.dic_traffic_env_conf.get(
                    "V28_ROLLOUT_TIMEOUT_S", 300)),
            )

        min_action_time = float(
            self.dic_traffic_env_conf.get("MIN_ACTION_TIME", 30))
        rollouts = self._v28_rollout_pool.evaluate(
            self.env, step_num, min_action_time)
        actions = {}
        audit = {
            "decision_step": int(step_num),
            "sim_time": float(self.env.get_current_time()),
            "decision_rule": (
                "Current V > -Current V after 30s rollout > "
                "effective nonzero history length > phase order"
            ),
            "rollout_horizon_s": min_action_time,
            "transition_time_s": float(
                self.dic_traffic_env_conf.get("YELLOW_TIME", 0)),
            "intersections": {},
        }

        for inter_idx, inter in enumerate(self.env.list_intersection):
            candidates = []
            current_v_by_phase = {}
            movement_sets = state[inter_idx].get(
                "traffic_movement_vehicle_ids_150m", [])
            for phase in inter.control_phases:
                current_v_by_phase[phase] = len(
                    self._phase_vehicle_ids_from_movement_sets(
                        movement_sets, phase))

            phase_histories = self._v28_phase_histories.setdefault(
                inter.inter_id, {phase: [] for phase in inter.control_phases})
            for phase, current_v in current_v_by_phase.items():
                phase_histories.setdefault(phase, []).append(current_v)

            for rollout in rollouts:
                candidate = {
                    **rollout["intersections"][inter.inter_id],
                    "rollout_elapsed_s": float(rollout.get("elapsed_s", 0.0)),
                }
                phase = candidate["phase"]
                current_v = int(current_v_by_phase.get(phase, 0))
                effective_history_len = sum(
                    value > 0 for value in phase_histories.get(phase, [])
                )
                score = (
                    current_v,
                    -int(candidate["remaining_v_30s"]),
                    effective_history_len,
                    -int(candidate["action_idx"]),
                )
                candidates.append({
                    **candidate,
                    "current_v": current_v,
                    "effective_history_len": effective_history_len,
                    "score": list(score),
                })

            best = max(candidates, key=lambda item: tuple(item["score"]))
            actions[inter.inter_id] = int(best["action_idx"])
            audit["intersections"][inter.inter_id] = {
                "chosen_phase": best["phase"],
                "chosen_action_idx": int(best["action_idx"]),
                "candidates": candidates,
            }

            # Match V2/V18 semantics: serving a phase clears its accumulated
            # nonzero-demand history for the next decision cycle.
            phase_histories[best["phase"]] = []

        log_path = os.path.join(
            self.dic_path["PATH_TO_WORK_DIRECTORY"], "v28_rollout_log.jsonl")
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(audit, ensure_ascii=True) + "\n")
        return actions

    def _close_v28_rollout_pool(self):
        if self._v28_rollout_pool is not None:
            self._v28_rollout_pool.close()
            self._v28_rollout_pool = None

    def _get_v30_rollout_actions(self, state, step_num):
        validate_standard_phase_order(self.env.list_intersection)
        if self._v30_rollout_pool is None:
            from .v30_rollout import V30RolloutPool
            rollout_dir = os.path.join(
                self.dic_path["PATH_TO_WORK_DIRECTORY"], "v30_rollouts")
            self._v30_rollout_pool = V30RolloutPool(
                config=self.dic_traffic_env_conf,
                path_config=self.dic_path,
                work_dir=rollout_dir,
                worker_count=int(self.dic_traffic_env_conf.get(
                    "V30_ROLLOUT_WORKERS", DEFAULT_ROLLOUT_WORKERS)),
                timeout_s=float(self.dic_traffic_env_conf.get(
                    "V30_ROLLOUT_TIMEOUT_S", 7200)),
            )

        current_by_tls = {}
        baseline_actions = {}
        baseline_phases = {}
        baseline_audits = {}
        for inter_idx, inter in enumerate(self.env.list_intersection):
            movement_sets = state[inter_idx].get(
                "traffic_movement_vehicle_ids_150m", [])
            current_v = {
                phase: len(self._phase_vehicle_ids_from_movement_sets(
                    movement_sets, phase))
                for phase in inter.control_phases
            }
            histories = self._v30_phase_histories.setdefault(
                inter.inter_id, {phase: [] for phase in inter.control_phases})
            for phase, value in current_v.items():
                histories.setdefault(phase, []).append(value)
            cycle_history = list(
                state[inter_idx].get("v9_cycle_150m_history") or [])

            def cycle_phase_values(phase):
                return [
                    self._phase_sum_from_movement_values(values, phase)
                    for values in cycle_history
                ]

            phase_cycle_values = {
                phase: cycle_phase_values(phase)
                for phase in inter.control_phases
            }
            delta_current_v = {
                phase: (
                    values[-1] - values[0] if len(values) >= 2 else 0
                )
                for phase, values in phase_cycle_values.items()
            }
            baseline_idx = max(
                range(len(inter.control_phases)),
                key=lambda idx: (
                    current_v[inter.control_phases[idx]],
                    delta_current_v[inter.control_phases[idx]],
                    sum(value > 0 for value in histories[inter.control_phases[idx]]),
                    -idx,
                ),
            )
            current_by_tls[inter.inter_id] = current_v
            baseline_actions[inter.inter_id] = baseline_idx
            baseline_phases[inter.inter_id] = inter.control_phases[baseline_idx]
            baseline_audits[inter.inter_id] = {
                "rule": (
                    "Current V > delta Current V (t30-t5) > "
                    "effective unserved history length > phase order"
                ),
                "phases": {
                    phase: {
                        "current_v": current_v[phase],
                        "cycle_current_v": phase_cycle_values[phase],
                        "delta_current_v": delta_current_v[phase],
                        "effective_history_len": sum(
                            value > 0 for value in histories[phase]),
                    }
                    for phase in inter.control_phases
                },
            }

        min_action_time = float(
            self.dic_traffic_env_conf.get("MIN_ACTION_TIME", 30))
        (rollouts, snapshot_path, start_control_state,
         start_state_fingerprint) = (
            self._v30_rollout_pool.evaluate(
                self.env, step_num, min_action_time, baseline_actions)
        )
        actions = {}
        audit = {
            "decision_step": int(step_num),
            "sim_time": float(self.env.get_current_time()),
            "seed": int(self.env.actual_seed),
            "start_snapshot_path": snapshot_path,
            "start_control_state": start_control_state,
            "start_state_fingerprint": start_state_fingerprint,
            "snapshot_retention_decisions": int(
                self._v30_rollout_pool.snapshot_retention_decisions),
            "decision_rule": (
                "Mainline action is always V25; counterfactual results are offline labels"
            ),
            "counterfactual_design": (
                "Only the target TLS changes; every other TLS uses its V25 baseline"
            ),
            "rollout_horizon_s": min_action_time,
            "transition_time_s": float(
                self.dic_traffic_env_conf.get("YELLOW_TIME", 0)),
            "intersections": {},
        }
        for inter in self.env.list_intersection:
            histories = self._v30_phase_histories[inter.inter_id]
            candidates = []
            for action_idx, phase in enumerate(inter.control_phases):
                result = rollouts[(inter.inter_id, action_idx)]
                current_v = current_by_tls[inter.inter_id][phase]
                history_len = sum(value > 0 for value in histories[phase])
                score = (
                    current_v,
                    -int(result["intersection_remaining_v_30s"]),
                    history_len,
                    -action_idx,
                )
                candidates.append({
                    **result,
                    "current_v": current_v,
                    "effective_history_len": history_len,
                    "score": list(score),
                })
            best = max(candidates, key=lambda item: tuple(item["score"]))
            # Counterfactual ranking is for offline reward labels only. The
            # live/mainline simulation must continue under V25 at every TLS.
            actions[inter.inter_id] = int(baseline_actions[inter.inter_id])
            audit["intersections"][inter.inter_id] = {
                "baseline_phase": baseline_phases[inter.inter_id],
                "baseline_action_idx": baseline_actions[inter.inter_id],
                "baseline_v25_audit": baseline_audits[inter.inter_id],
                "executed_mainline_phase": baseline_phases[inter.inter_id],
                "executed_mainline_action_idx": baseline_actions[inter.inter_id],
                "heuristic_best_phase_for_audit": best["phase"],
                "heuristic_best_action_idx_for_audit": int(
                    best["candidate_idx"]),
                "heuristic_best_rule_for_audit": (
                    "Current V > negative intersection remaining V at t+30 > "
                    "effective unserved history length > phase order"
                ),
                "candidates": candidates,
            }
            histories[inter.control_phases[baseline_actions[inter.inter_id]]] = []

        log_path = os.path.join(
            self.dic_path["PATH_TO_WORK_DIRECTORY"], "v30_rollout_log.jsonl")
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(audit, ensure_ascii=True) + "\n")
        return actions

    def _close_v30_rollout_pool(self):
        if self._v30_rollout_pool is not None:
            self._v30_rollout_pool.close()
            self._v30_rollout_pool = None

    def _begin_video_interval(self, decision_step: int, sim_start_s: float) -> None:
        """Open the main-line V34-style video/state window for one action."""
        if self.decision_window_recorder is None and self.video_sft_dataset_writer is None:
            return
        if self._video_interval_active:
            raise RuntimeError("previous V30 video interval was not finalized")
        sim_start_s = float(sim_start_s)
        actual_start_s = float(self.env.get_current_time())
        if abs(actual_start_s - sim_start_s) > 1e-6:
            raise RuntimeError(
                "V30 video interval started from a stale SUMO clock: "
                f"loop_time={sim_start_s} actual_time={actual_start_s}"
            )

        self._video_interval_active = True
        self._video_interval_start_s = sim_start_s
        try:
            if self.decision_window_recorder is not None:
                self.decision_window_recorder.begin_interval(
                    decision_step=int(decision_step),
                    sim_start_sec=sim_start_s,
                )
            if self.video_sft_dataset_writer is not None:
                self.video_sft_dataset_writer.begin_interval(
                    int(decision_step), sim_start_s, tls_ids=self.tls_ids
                )
        except BaseException:
            self._finalize_video_interval(
                actual_start_s,
                status="incomplete",
                error="failed while opening the V30 video interval",
                force_discard=True,
            )
            raise

    def _finalize_video_interval(
        self,
        sim_end_s: float,
        status: str = "complete",
        error: str = None,
        force_discard: bool = False,
    ) -> Dict[str, Any]:
        """Close the main-line video/state window without touching rollouts."""
        if not self._video_interval_active:
            return {}

        video_details: Dict[str, Any] = {}
        timing_errors = []
        interval_start_s = self._video_interval_start_s
        if (
            status == "complete"
            and not force_discard
            and interval_start_s is not None
        ):
            expected_end_s = interval_start_s + float(
                self.dic_traffic_env_conf.get("MIN_ACTION_TIME", 30.0)
            )
            if abs(float(sim_end_s) - expected_end_s) > 1e-6:
                timing_errors.append(
                    "V30 video interval endpoint mismatch: "
                    f"start={interval_start_s} actual_end={float(sim_end_s)} "
                    f"expected_end={expected_end_s}"
                )
        try:
            if self.decision_window_recorder is not None:
                video_details = self.decision_window_recorder.finalize_interval(
                    sim_end_sec=float(sim_end_s),
                    force_discard=bool(force_discard),
                    return_details=True,
                )

            if (
                status == "complete"
                and not force_discard
                and self.decision_window_recorder is not None
                and interval_start_s is not None
            ):
                vlm_config = self.dic_traffic_env_conf.get("VLM_CONFIG", {})
                sample_interval_s = float(
                    vlm_config.get("VIDEO_FRAME_SAMPLE_INTERVAL", 5.0)
                )
                action_time_s = float(
                    self.dic_traffic_env_conf.get("MIN_ACTION_TIME", 30.0)
                )
                expected_frame_count = int(round(action_time_s / sample_interval_s))
                expected_frame_times = [
                    interval_start_s + sample_interval_s * index
                    for index in range(1, expected_frame_count + 1)
                ]
                actual_frame_times = (video_details or {}).get("frame_times", {})
                frame_count_errors = {}
                for tls_id in self.tls_ids:
                    actual = [
                        float(value)
                        for value in actual_frame_times.get(tls_id, [])
                    ]
                    if (
                        len(actual) != len(expected_frame_times)
                        or any(
                            abs(observed - expected) > 1e-6
                            for observed, expected in zip(
                                actual, expected_frame_times
                            )
                        )
                    ):
                        frame_count_errors[tls_id] = {
                            "actual": actual,
                            "expected": expected_frame_times,
                        }
                if frame_count_errors:
                    timing_errors.append(
                        "V30 video frame sequence mismatch: "
                        f"{frame_count_errors}"
                    )
        finally:
            try:
                if self.video_sft_dataset_writer is not None:
                    writer_status = "invalid" if timing_errors else status
                    writer_error = error
                    if timing_errors:
                        writer_error = "; ".join(timing_errors)
                    self.video_sft_dataset_writer.finalize_interval(
                        sim_end_s=float(sim_end_s),
                        video_details=video_details,
                        status=writer_status,
                        error=writer_error,
                    )
            finally:
                self._video_interval_active = False
                self._video_interval_start_s = None
        if timing_errors:
            raise RuntimeError("; ".join(timing_errors))
        return video_details

    def close_runtime_resources(self):
        """Release subprocess-backed resources on success or failure."""
        self._close_v28_rollout_pool()
        self._close_v30_rollout_pool()
        if self._video_interval_active:
            try:
                self._finalize_video_interval(
                    self.env.get_current_time() if self.env is not None else 0.0,
                    status="incomplete",
                    error="runtime closed before the decision interval was finalized",
                    force_discard=True,
                )
            except Exception as exc:
                print(f"Warning: failed to finalize incomplete video interval: {exc}")
        self._close_perception_image_capture()
        if self.env is not None:
            self.env.close()

    def _close_perception_image_capture(self) -> None:
        if self._video_interval_active:
            try:
                self._finalize_video_interval(
                    self.env.get_current_time() if self.env is not None else 0.0,
                    status="incomplete",
                    error="perception renderer closed before interval finalization",
                    force_discard=True,
                )
            except Exception as exc:
                print(f"Warning: failed to release incomplete video interval: {exc}")
        if self.renderer is not None:
            try:
                self.renderer.destroy()
            except Exception as exc:
                print(f"Warning: failed to destroy renderer: {exc}")
            self.renderer = None

    def _write_vehicle_position_snapshot(self, step_num: int, current_time: float,
                                         image_paths: Dict[str, List[str]] = None) -> None:
        if self.vehicle_position_snapshot_writer is None:
            return
        self.vehicle_position_snapshot_writer.write_step(
            step_num=step_num,
            sim_time=float(current_time),
            env=self.env,
            image_paths=image_paths or {},
        )

    def _build_perception_snapshot_callback(self, decision_step: int):
        if (
            self.vehicle_position_snapshot_writer is None
            and self.decision_window_recorder is None
            and self.video_sft_dataset_writer is None
        ):
            return None
        interval = float(self.dic_traffic_env_conf.get("INTERVAL", 1.0))
        sample_interval = float(self.dic_traffic_env_conf.get("VEHICLE_POSITION_SNAPSHOT_INTERVAL", 10.0))
        stride = max(1, int(round(sample_interval / interval)))

        def _callback(inner_i: int, env: SUMOEnv, sim_time_s: float = None) -> None:
            sim_time = (
                float(env.get_current_time())
                if sim_time_s is None else float(sim_time_s)
            )

            # The renderer and the SUMO sidecar consume this exact post-step
            # clock. The video recorder samples only 5, 10, ..., 30 s; the
            # sidecar keeps every SUMO tick for later audit and frame joins.
            if self.decision_window_recorder is not None:
                self._record_decision_interval_frame(
                    inner_i, env, sim_time_s=sim_time
                )
            if self.video_sft_dataset_writer is not None:
                self.video_sft_dataset_writer.capture_tick(
                    inner_i, env, sim_time_s=sim_time
                )

            if self.vehicle_position_snapshot_writer is None:
                return
            if (inner_i + 1) % stride != 0:
                return
            snapshot_step = self._snapshot_serial
            self._snapshot_serial += 1
            image_paths = self._save_perception_snapshot_images(snapshot_step)
            self._write_vehicle_position_snapshot(
                snapshot_step, sim_time, image_paths=image_paths
            )

        return _callback

    def _get_render_tshub_obs(self, tls_ids=None):
        vlm_config = self.dic_traffic_env_conf.get("VLM_CONFIG", {})
        radius = vlm_config.get("LOCAL_RENDER_RADIUS_M", 200.0)
        if radius is None:
            return self.env.get_tshub_obs()
        return self.env.get_tshub_obs(
            tls_ids=tls_ids, radius=float(radius)
        )

    def _record_decision_interval_frame(
        self, inner_i: int, env: SUMOEnv, sim_time_s: float = None
    ) -> None:
        """Record one V34-compatible main-line video frame."""
        if self.renderer is None or self.decision_window_recorder is None:
            return

        vlm_config = self.dic_traffic_env_conf.get("VLM_CONFIG", {})
        record_mode = str(
            vlm_config.get("VIDEO_RECORD_MODE", "sampled") or "sampled"
        ).lower()
        sample_interval = float(
            vlm_config.get("VIDEO_FRAME_SAMPLE_INTERVAL", 5.0)
        )
        sim_interval = float(
            self.dic_traffic_env_conf.get("INTERVAL", 1.0)
        )
        if record_mode not in ("continuous", "full", "all"):
            record_mode = "sampled"
        if record_mode == "sampled" and sample_interval > sim_interval:
            stride = max(1, int(round(sample_interval / sim_interval)))
            if (int(inner_i) + 1) % stride != 0:
                return

        sim_time = (
            float(env.get_current_time())
            if sim_time_s is None else float(sim_time_s)
        )
        current_env_time = float(env.get_current_time())
        if abs(current_env_time - sim_time) > 1e-6:
            raise RuntimeError(
                "SUMO time changed before V30 video rendering: "
                f"captured={sim_time} current={current_env_time}"
            )

        batch_info = (
            self.renderer.get_batch_info()
            if hasattr(self.renderer, "get_batch_info") else {"mode": "all"}
        )
        if batch_info.get("mode") == "batch":
            total_batches = batch_info.get("total_batches", 1)
            for batch_idx in range(total_batches):
                current_batch_tls = (
                    self.renderer.get_current_batch_tls_ids()
                    if hasattr(self.renderer, "get_current_batch_tls_ids")
                    else self.tls_ids
                )
                tshub_obs = self._get_render_tshub_obs(current_batch_tls)
                sensor_data = self.renderer.step(
                    tshub_obs, should_count_vehicles=False
                )
                if sensor_data:
                    self.decision_window_recorder.add_sensor_data(
                        sensor_data, current_batch_tls, sim_time=sim_time
                    )
                if (
                    batch_idx < total_batches - 1
                    and hasattr(self.renderer, "switch_to_next_batch")
                ):
                    self.renderer.switch_to_next_batch()
            if hasattr(self.renderer, "switch_to_batch"):
                self.renderer.switch_to_batch(0)
            if abs(float(env.get_current_time()) - sim_time) > 1e-6:
                raise RuntimeError(
                    "SUMO time changed during V30 batch video rendering: "
                    f"captured={sim_time} current={env.get_current_time()}"
                )
            return

        tshub_obs = self._get_render_tshub_obs(self.tls_ids)
        sensor_data = self.renderer.step(
            tshub_obs, should_count_vehicles=False
        )
        if sensor_data:
            self.decision_window_recorder.add_sensor_data(
                sensor_data, self.tls_ids, sim_time=sim_time
            )
        if abs(float(env.get_current_time()) - sim_time) > 1e-6:
            raise RuntimeError(
                "SUMO time changed during V30 video rendering: "
                f"captured={sim_time} current={env.get_current_time()}"
            )

    def _save_perception_snapshot_images(self, snapshot_step: int) -> Dict[str, List[str]]:
        if self.renderer is None or self.image_saver is None:
            raise RuntimeError("image snapshot requested but renderer/image_saver is not initialized")
        tls_ids = [inter.inter_id for inter in self.env.list_intersection]
        radius = float(self.dic_traffic_env_conf.get("CAMERA_VIEW_DISTANCE", 150.0))
        batch_info = self.renderer.get_batch_info() if hasattr(self.renderer, "get_batch_info") else {"mode": "all"}
        all_saved_paths = {}
        if batch_info.get("mode") == "batch":
            total_batches = batch_info.get("total_batches", 1)
            for batch_idx in range(total_batches):
                current_batch_tls = (
                    self.renderer.get_current_batch_tls_ids()
                    if hasattr(self.renderer, "get_current_batch_tls_ids") else tls_ids
                )
                tshub_obs = self.env.get_tshub_obs(tls_ids=current_batch_tls, radius=radius)
                sensor_data = self._capture_current_renderer_sensor_data(tshub_obs)
                if sensor_data:
                    saved_paths = self.image_saver.save_step_images(
                        step=snapshot_step,
                        sensor_data=sensor_data,
                        tls_ids=current_batch_tls,
                    )
                    for tls_id, paths in saved_paths.items():
                        all_saved_paths.setdefault(tls_id, [])
                        for path in paths:
                            if path not in all_saved_paths[tls_id]:
                                all_saved_paths[tls_id].append(path)
                if batch_idx < total_batches - 1 and hasattr(self.renderer, "switch_to_next_batch"):
                    self.renderer.switch_to_next_batch()
            if hasattr(self.renderer, "switch_to_batch"):
                self.renderer.switch_to_batch(0)
            self._validate_perception_image_paths(snapshot_step, all_saved_paths, tls_ids)
            return all_saved_paths

        tshub_obs = self.env.get_tshub_obs(tls_ids=tls_ids, radius=radius)
        sensor_data = self._capture_current_renderer_sensor_data(tshub_obs)
        if sensor_data:
            saved_paths = self.image_saver.save_step_images(
                step=snapshot_step,
                sensor_data=sensor_data,
                tls_ids=tls_ids,
            )
            self._validate_perception_image_paths(snapshot_step, saved_paths, tls_ids)
            return saved_paths
        raise RuntimeError(f"renderer returned no sensor data for perception snapshot {snapshot_step}")

    def _capture_current_renderer_sensor_data(self, tshub_obs: Dict[str, Any]) -> Dict[str, Any]:
        warmup_reads = int(self.dic_traffic_env_conf.get("PERCEPTION_RENDER_WARMUP_READS", 1))
        sensor_data = {}
        for _ in range(max(0, warmup_reads)):
            self.renderer.step(tshub_obs, should_count_vehicles=False)
        sensor_data = self.renderer.step(tshub_obs, should_count_vehicles=False)
        return sensor_data

    def _validate_perception_image_paths(self, snapshot_step: int, saved_paths: Dict[str, List[str]],
                                         tls_ids: List[str]) -> None:
        expected_per_tls = 4
        missing = []
        for tls_id in tls_ids:
            paths = saved_paths.get(tls_id, [])
            existing_paths = [path for path in paths if os.path.exists(path)]
            if len(existing_paths) < expected_per_tls:
                missing.append((tls_id, len(existing_paths), paths))
        if missing:
            preview = ", ".join(f"{tls}:{count}/4" for tls, count, _ in missing[:8])
            raise RuntimeError(
                f"perception snapshot {snapshot_step} image save incomplete; "
                f"expected 4 images per intersection, got {preview}"
            )

    def _initialize_training_metrics(self) -> Dict[str, Any]:
        """Initialize metrics tracking for training."""
        return {
            'total_reward': 0.0,
            'queue_length_episode': [],
            'waiting_time_episode': [],
            'global_waiting_times': []
        }

    def _setup_wandb_logger(self, round: int) -> Any:
        """Setup Weights & Biases logging for the training session."""
        all_config = merge(merge(self.dic_agent_conf, self.dic_path), self.dic_traffic_env_conf)
        phase_count = len(self.dic_traffic_env_conf['PHASE'])

        return wandb.init(
            project=self.dic_traffic_env_conf['PROJECT_NAME'],
            group=f"{self.dic_traffic_env_conf['MODEL_NAME']}-{self.roadnet}-{self.trafficflow}-{phase_count}_Phases",
            name=f"round_{round}",
            config=all_config,
        )

    def _get_agent_actions(self, state: List[Any], step_num: int,
                           state_action_log: List[List[Dict]]) -> Dict[str, int]:
        """
        Get actions from all agents, handling threading for LLM-based agents.

        Args:
            state: Current state for all intersections
            step_num: Current step number
            state_action_log: Log for state-action pairs

        Returns:
            Dictionary mapping inter_id to action for each intersection
        """
        action_dict = {}
        threads = []
        model_name = self.dic_traffic_env_conf["MODEL_NAME"]

        # Collect state information and prepare actions
        for i, agent_state in enumerate(state):
            # Log detailed state information
            self._log_intersection_state(i, state_action_log)

            if "ChatGPT" in model_name or "open_llm" in model_name:
                # Create thread for LLM-based agents
                thread = threading.Thread(target=self.agents[i].choose_action, args=(self.env,))
                threads.append(thread)
            else:
                # Get action directly for non-LLM agents
                if model_name == "V21":
                    neighbor_info = self._build_v21_neighbor_info(i, state, step_num)
                    action = self.agents[i].choose_action(
                        step_num, agent_state, neighbor_info=neighbor_info)
                elif model_name == "V32":
                    coordination_info = self._build_v32_coordination_info(
                        i, state, step_num)
                    action = self.agents[i].choose_action(
                        step_num, agent_state,
                        coordination_info=coordination_info)
                elif model_name == "V36":
                    snapshots_by_tls = {
                        self.env.list_intersection[index].inter_id:
                            (item.get("v36_cycle_outbound_snapshots") or [])
                        for index, item in enumerate(state)
                    }
                    target_inter = self.env.list_intersection[i]
                    coordination_info = self._get_v36_coordination_manager().build(
                        target_inter.inter_id,
                        list(target_inter.control_phases),
                        snapshots_by_tls,
                        step_num,
                    )
                    action = self.agents[i].choose_action(
                        step_num, agent_state,
                        coordination_info=coordination_info)
                else:
                    action = self.agents[i].choose_action(step_num, agent_state)
                inter_id = self.env.list_intersection[i].inter_id
                action_dict[inter_id] = action

        # Handle threading for LLM agents
        if threads:
            action_dict = self._execute_threaded_actions(threads, model_name)

        return action_dict

    def _phase_vehicle_ids_from_movement_sets(
            self, movement_vehicle_sets: List[List[str]], phase_name: str) -> List[str]:
        movement_map = {
            "WL": 0, "WT": 1, "WR": 2,
            "EL": 3, "ET": 4, "ER": 5,
            "NL": 6, "NT": 7, "NR": 8,
            "SL": 9, "ST": 10, "SR": 11,
        }
        vehicle_ids = []
        for movement in [phase_name[i:i + 2] for i in range(0, len(phase_name), 2)]:
            idx = movement_map.get(movement)
            if idx is not None and idx < len(movement_vehicle_sets):
                vehicle_ids.extend(movement_vehicle_sets[idx] or [])
        return sorted(set(vehicle_ids))

    def _phase_sum_from_movement_values(
            self, movement_values: List[int], phase_name: str) -> int:
        movement_map = {
            "WL": 0, "WT": 1, "WR": 2,
            "EL": 3, "ET": 4, "ER": 5,
            "NL": 6, "NT": 7, "NR": 8,
            "SL": 9, "ST": 10, "SR": 11,
        }
        movements = (
            [item for item in phase_name.split("_") if item]
            if "_" in phase_name
            else [phase_name[i:i + 2] for i in range(0, len(phase_name), 2)]
        )
        return int(sum(
            movement_values[idx]
            for movement in movements
            for idx in [movement_map.get(movement)]
            if idx is not None and idx < len(movement_values)
        ))

    def _target_trend_vehicle_sets(
            self, target_inter: Any, entry_dir: str, link_distance: float) -> List[List[str]]:
        lower = max(0.0, float(link_distance) - 150.0)
        upper = max(0.0, float(link_distance) - 50.0)
        filtered_entering_lanes = []
        for lane in target_inter.list_entering_lanes:
            if lane is None:
                filtered_entering_lanes.append([])
                continue
            road_id = target_inter.lane_to_road.get(lane)
            orient = target_inter.road_id_2_orient.get("incoming", {}).get(road_id)
            if orient != entry_dir:
                filtered_entering_lanes.append([])
                continue
            vehicle_ids = []
            for veh_id in target_inter.dic_lane_vehicle_current_step.get(lane, []):
                distance = target_inter.dic_vehicle_distance_current_step.get(veh_id)
                if distance is None:
                    continue
                if lower <= float(distance) <= upper:
                    vehicle_ids.append(veh_id)
            filtered_entering_lanes.append(vehicle_ids)
        return target_inter._get_traffic_movement_vehicle_sets(filtered_entering_lanes)

    def _eta_from_vehicle_ids(self, target_inter: Any, vehicle_ids: List[str]) -> float:
        vehicle_eta_s = self._eta_by_vehicle_ids(target_inter, vehicle_ids)
        return round(min(vehicle_eta_s.values()), 1) if vehicle_eta_s else None

    def _eta_by_vehicle_ids(self, target_inter: Any, vehicle_ids: List[str]) -> Dict[str, float]:
        camera_view_distance = float(self.dic_traffic_env_conf.get("CAMERA_VIEW_DISTANCE", 150.0))
        eta_speed_mps = float(self.dic_traffic_env_conf.get("V21_ETA_SPEED_MPS", 11.0))
        if eta_speed_mps <= 0:
            eta_speed_mps = 11.0
        vehicle_eta_s = {}
        for veh_id in vehicle_ids:
            distance = target_inter.dic_vehicle_distance_current_step.get(veh_id)
            try:
                distance = float(distance)
            except Exception:
                continue
            vehicle_eta_s[veh_id] = round(
                max(0.0, distance - camera_view_distance) / eta_speed_mps, 1)
        return vehicle_eta_s

    def _build_v21_neighbor_info(self, target_idx: int, state: List[Any], step_num: int) -> Dict[str, Dict[str, Any]]:
        """Build upstream-neighbor ETA metadata for V21 in the heuristic pipeline."""
        intersections = self._network_topology.get("intersections", {})
        if not intersections:
            return {}

        target_inter = self.env.list_intersection[target_idx]
        target_tls = self.env.list_intersection[target_idx].inter_id
        upstream_directions = set()
        for source_cfg in intersections.values():
            for edge in (source_cfg.get("neighbors") or {}).values():
                if edge.get("neighbor_id") != target_tls:
                    continue
                entry_dir = (edge.get("their_entry_direction") or "").upper()
                if entry_dir in ("E", "W", "N", "S"):
                    upstream_directions.add(entry_dir)
        if upstream_directions != {"E", "W", "N", "S"}:
            return {
                "__coordination_disabled__": {
                    "reason": "perimeter_incomplete_upstream",
                    "upstream_directions": sorted(upstream_directions),
                }
            }

        neighbor_info = {}
        for source_tls, source_cfg in intersections.items():
            for _, edge in (source_cfg.get("neighbors") or {}).items():
                if edge.get("neighbor_id") != target_tls:
                    continue
                entry_dir = edge.get("their_entry_direction", "")
                link_distance = float(edge.get("distance_m", 0) or 0)
                movement_sets = self._target_trend_vehicle_sets(
                    target_inter, entry_dir, link_distance)
                entry_phase_candidates = self._candidate_phases_for_entry_dir(
                    target_inter.control_phases, entry_dir)
                for target_phase in entry_phase_candidates:
                    vehicle_ids = self._phase_vehicle_ids_from_movement_sets(
                        movement_sets, target_phase)
                    vehicle_eta_s = self._eta_by_vehicle_ids(target_inter, vehicle_ids)
                    eta_s = round(min(vehicle_eta_s.values()), 1) if vehicle_eta_s else None
                    neighbor_info[f"{source_tls}:{target_phase}"] = {
                        "phase": target_phase,
                        "distance_m": link_distance,
                        "speed_mps": edge.get("speed_limit_mps", 0),
                        "eta_s": eta_s,
                        "their_entry": entry_dir,
                        "counts": {target_phase: len(vehicle_ids)},
                        "vehicle_ids": vehicle_ids,
                        "vehicle_eta_s": vehicle_eta_s,
                        "trend_zone": "50-150m_from_upstream_stopline",
                        "step": step_num,
                    }
        return neighbor_info

    def _build_v32_coordination_info(
            self, target_idx: int, state: List[Any], step_num: int
    ) -> Dict[str, Dict[str, Any]]:
        """Build V32 coordination from selected upstream outbound pseudo-video frames.

        The validated V36 selector is now the V32 production implementation:
        upstream outbound 150m, frames 0/5/10/15/20/25, ETA window
        [-5, 15], and explicit delayed consumption for long links.
        """
        snapshots_by_tls = {
            self.env.list_intersection[index].inter_id:
                (item.get("v36_cycle_outbound_snapshots") or [])
            for index, item in enumerate(state)
        }
        target_inter = self.env.list_intersection[target_idx]
        coordination = self._get_v36_coordination_manager().build(
            target_inter.inter_id,
            list(target_inter.control_phases),
            snapshots_by_tls,
            step_num,
        )
        upstream_directions = set(
            (coordination.get("__meta__") or {}).get(
                "upstream_directions", []))
        missing_directions = set("EWNS") - upstream_directions
        if missing_directions:
            legacy = self._build_v32_legacy_coordination_info(
                target_idx, state, step_num)
            merge_missing_boundary_coordination(
                coordination,
                legacy,
                list(target_inter.control_phases),
                missing_directions,
            )
        return coordination

    def _build_v32_legacy_coordination_info(
            self, target_idx: int, state: List[Any], step_num: int
    ) -> Dict[str, Dict[str, Any]]:
        """Retained reference for the pre-V36 final-frame implementation.

        The final (normally t=30s) frame is the common reference for all
        vehicles.  For a held phase, arrivals within the next 15 seconds are
        useful.  For a switched phase, the 5-second yellow interval is also
        useful because those vehicles wait inside the 150m view when green
        starts, so its horizon is 20 seconds.
        """
        target_inter = self.env.list_intersection[target_idx]
        target_tls = target_inter.inter_id
        phases = list(target_inter.control_phases)
        result = {
            phase: {
                "arrival_15s": 0,
                "internal": {},
                "boundary": {},
                "sources": {},
            }
            for phase in phases
        }
        intersections = self._network_topology.get("intersections", {})
        upstream_directions = set()
        upstream_sources = {}
        for source_tls, source_cfg in intersections.items():
            for edge in (source_cfg.get("neighbors") or {}).values():
                if edge.get("neighbor_id") != target_tls:
                    continue
                entry_dir = str(edge.get("their_entry_direction") or "").upper()
                if entry_dir in "EWNS":
                    upstream_directions.add(entry_dir)
                    upstream_sources.setdefault(entry_dir, str(source_tls))

        movement_map = {
            "WL": 0, "WT": 1, "WR": 2,
            "EL": 3, "ET": 4, "ER": 5,
            "NL": 6, "NT": 7, "NR": 8,
            "SL": 9, "ST": 10, "SR": 11,
        }
        camera_distance = float(
            self.dic_traffic_env_conf.get("CAMERA_VIEW_DISTANCE", 150.0))
        decision_time = float(
            self.dic_traffic_env_conf.get("MIN_ACTION_TIME", 30.0))
        yellow_time = float(self.dic_traffic_env_conf.get("YELLOW_TIME", 5.0))
        snapshots = sorted(
            state[target_idx].get("v32_cycle_vehicle_snapshots") or [],
            key=lambda item: float(item.get("time_s", 0)),
        )
        # One snapshot for the full candidate set prevents a vehicle platoon
        # from being assembled from several incompatible video timestamps.
        frame = snapshots[-1] if snapshots else {}
        frame_time = float(frame.get("time_s", decision_time))
        movement_ids = frame.get("movement_vehicle_ids") or []
        distances = frame.get("vehicle_distance") or {}
        speeds = frame.get("vehicle_speed") or {}

        active_phase = None
        cur_phase = state[target_idx].get("cur_phase") or []
        if cur_phase:
            try:
                active_idx = int(cur_phase[0])
                if 0 <= active_idx < len(phases):
                    active_phase = phases[active_idx]
            except (TypeError, ValueError):
                pass

        eta_by_movement = {movement: [] for movement in movement_map}
        for movement, movement_idx in movement_map.items():
            if movement[0] not in upstream_directions:
                continue
            if movement_idx >= len(movement_ids):
                continue
            for vehicle_id in movement_ids[movement_idx] or []:
                try:
                    distance = float(distances.get(vehicle_id))
                    speed = float(speeds.get(vehicle_id))
                except (TypeError, ValueError):
                    continue
                # At the reference frame, Current V owns vehicles already in
                # the 150m view.  Coordination owns only later arrivals.
                if distance <= camera_distance or speed <= 0.1:
                    continue
                eta_at_decision = (
                    (distance - camera_distance) / speed
                    - (decision_time - frame_time)
                )
                if eta_at_decision >= 0.0:
                    eta_by_movement[movement].append(eta_at_decision)
        cycle_history = list(
            state[target_idx].get("v9_cycle_150m_history") or [])
        v15 = cycle_history[2] if len(cycle_history) >= 3 else []
        v30 = cycle_history[5] if len(cycle_history) >= 6 else (
            cycle_history[-1] if cycle_history else [])
        boundary_estimate_available = bool(
            len(cycle_history) >= 6 and len(v15) >= len(movement_map)
            and len(v30) >= len(movement_map))

        for phase in phases:
            is_hold_phase = active_phase is not None and phase == active_phase
            horizon_s = 15.0 if is_hold_phase else 15.0 + yellow_time
            result[phase]["frame_time_s"] = frame_time
            result[phase]["horizon_s"] = horizon_s
            for movement in [phase[i:i + 2] for i in range(0, len(phase), 2)]:
                direction = movement[0]
                if direction in upstream_directions:
                    value = sum(
                        1 for eta in eta_by_movement.get(movement, [])
                        if eta <= horizon_s)
                    result[phase]["internal"][movement] = value
                    result[phase]["sources"][movement] = upstream_sources[direction]
                else:
                    movement_idx = movement_map.get(movement)
                    value = 0
                    if (movement_idx is not None
                            and movement_idx < len(v15)
                            and movement_idx < len(v30)):
                        value = max(
                            0, int(v30[movement_idx]) - int(v15[movement_idx]))
                    result[phase]["boundary"][movement] = value
                    result[phase]["sources"][movement] = "network_boundary"
                result[phase]["arrival_15s"] += value
        result["__meta__"] = {
            "step": int(step_num),
            "frame_time_s": frame_time,
            "active_phase": active_phase,
            "upstream_directions": sorted(upstream_directions),
            "boundary_directions": sorted(set("EWNS") - upstream_directions),
            "boundary_estimate_available": boundary_estimate_available,
        }
        return result

    def _candidate_phases_for_entry_dir(self, control_phases: List[str], entry_dir: str) -> List[str]:
        entry_dir = (entry_dir or "").upper()
        if entry_dir in ("E", "W"):
            prefixes = ("E", "W")
        elif entry_dir in ("N", "S"):
            prefixes = ("N", "S")
        else:
            return []
        candidates = []
        for phase in control_phases:
            movements = [phase[i:i + 2] for i in range(0, len(phase), 2)]
            if any(movement[:1] in prefixes for movement in movements):
                candidates.append(phase)
        return candidates

    def _attach_counterfactual_discharge(self, cf_discharge: Dict[str, Any]) -> None:
        if not cf_discharge:
            return
        for i, agent in enumerate(self.agents):
            inter_id = self.env.list_intersection[i].inter_id
            if hasattr(agent, "attach_counterfactual_discharge"):
                agent.attach_counterfactual_discharge(cf_discharge.get(inter_id))

    def _log_intersection_state(self, intersection_idx: int, state_action_log: List[List[Dict]]) -> None:
        """Log detailed state information for an intersection."""
        intersection = self.env.intersection_dict[self.env.list_intersection[intersection_idx].inter_name]
        roads = deepcopy(intersection["roads"])
        statistic_state, statistic_state_incoming, mean_speed = get_state_detail(roads, self.env)

        state_action_log[intersection_idx].append({
            "state": statistic_state,
            "state_incoming": statistic_state_incoming,
            "approaching_speed": mean_speed
        })

    def _execute_threaded_actions(self, threads: List[threading.Thread], model_name: str) -> Dict[str, int]:
        """Execute actions using threading for LLM-based agents and return as dict."""
        action_dict = {}

        if "ChatGPT" in model_name:
            # Start all threads and wait for completion
            for thread in threads:
                thread.start()

            for thread in tqdm(threads, desc="Waiting for ChatGPT responses"):
                thread.join()

        elif "open_llm" in model_name:
            # Manage thread pool for open LLM agents
            self._manage_llm_thread_pool(threads)

        # Collect actions from all agents and map to inter_id
        for i, agent in enumerate(self.agents):
            inter_id = self.env.list_intersection[i].inter_id
            action_dict[inter_id] = agent.temp_action_logger

        return action_dict

    def _manage_llm_thread_pool(self, threads: List[threading.Thread]) -> None:
        """Manage thread pool for open LLM agents to avoid overwhelming the API."""
        thread_limit = (self.dic_traffic_env_conf["LLM_API_THREAD_NUM"]
                        if not self.dic_agent_conf["WITH_EXTERNAL_API"] else 2)
        started_threads = []

        for i, thread in enumerate(tqdm(threads, desc="Processing LLM requests")):
            thread.start()
            started_threads.append(i)

            # Wait for batch completion
            if (i + 1) % thread_limit == 0:
                for thread_id in started_threads:
                    threads[thread_id].join()
                started_threads = []

        # Wait for remaining threads
        for thread_id in started_threads:
            threads[thread_id].join()

    def _log_step_data(self, state_action_log: List[List[Dict]], action_dict: Dict[str, int],
                       memory_file_path: str, current_time: float, state: List[Any],
                       reward: List[float]) -> None:
        """Log data for the current simulation step."""
        # Log actions in state-action log
        # action_dict maps inter_id to action, we iterate by intersection index
        for i, intersection in enumerate(self.env.list_intersection):
            inter_id = intersection.inter_id
            if inter_id in action_dict:
                action = action_dict[inter_id]
                # Use control_phases if available, otherwise fallback to eight_phase_list
                if hasattr(intersection, 'control_phases') and 0 <= action < len(intersection.control_phases):
                    action_label = intersection.control_phases[action]
                elif 0 <= action < len(eight_phase_list):
                    action_label = eight_phase_list[action]
                else:
                    action_label = f"action_{action}"
                state_action_log[i][-1]["action"] = action_label
                state_action_log[i][-1]["action_idx"] = int(action)  # Convert to native Python int for JSON serialization

        # Write to memory file
        current_phases = [state[i]["cur_phase"][0] for i in range(len(state))]
        memory_str = f'time = {current_time}\taction = {action_dict}\tcurrent_phase = {current_phases}\treward = {reward}'

        with open(memory_file_path, "a") as f_memory:
            f_memory.write(memory_str + "\n")

    def _update_training_metrics(self, metrics: Dict[str, Any], reward: List[float]) -> None:
        """Update training metrics with current step data."""
        metrics['total_reward'] += sum(reward)

        metric_interval = float(self.dic_traffic_env_conf.get(
            "METRIC_SAMPLE_INTERVAL",
            self.dic_traffic_env_conf.get("MIN_ACTION_TIME", 30.0)))
        current_time = float(self.env.get_current_time())
        if metric_interval <= 0.0:
            should_record_queue = True
        else:
            if getattr(self, "_next_queue_metric_time", None) is None:
                self._next_queue_metric_time = metric_interval
            should_record_queue = (
                current_time >= self._next_queue_metric_time - 1e-9)
        if should_record_queue:
            queue_length = sum(
                sum(intersection.dic_feature.get(
                    "lane_num_waiting_vehicle_in", [0]))
                for intersection in self.env.list_intersection
            )
            metrics['queue_length_episode'].append(queue_length)
            if metric_interval > 0.0:
                while self._next_queue_metric_time <= current_time + 1e-9:
                    self._next_queue_metric_time += metric_interval

        # Extract current-step waiting times with robust type checking.
        # SUMOEnv keeps waiting_vehicle_list as {veh_id: {time, link}} for
        # agent-facing features; final AWT uses get_all_vehicle_waiting_times().
        waiting_times = []
        for v_id, time_info in self.env.waiting_vehicle_list.items():
            try:
                # Handle different possible data types for robustness
                if isinstance(time_info, (int, float)):
                    waiting_times.append(float(time_info))
                elif isinstance(time_info, dict) and 'time' in time_info:
                    # Fallback in case implementation changes to dict format
                    waiting_times.append(float(time_info['time']))
                else:
                    # Skip malformed entries with warning
                    print(f"Warning: Unexpected waiting time format for vehicle {v_id}: {type(time_info)}")
                    continue
            except (ValueError, TypeError) as e:
                print(f"Error processing waiting time for vehicle {v_id}: {e}")
                continue
        
        avg_waiting_time = np.mean(waiting_times) if waiting_times else 0.0
        metrics['waiting_time_episode'].append(avg_waiting_time)
        metrics['global_waiting_times'].extend(waiting_times)

    def _calculate_final_results(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """Calculate final training results and metrics."""
        # Calculate travel times
        vehicle_travel_times = self._calculate_travel_times()
        total_travel_time = np.mean([sum(times) for times in vehicle_travel_times.values()])
        waiting_times = (
            self.env.get_all_vehicle_waiting_times()
            if hasattr(self.env, "get_all_vehicle_waiting_times")
            else {}
        )
        avg_waiting_time = (
            float(np.mean(list(waiting_times.values())))
            if waiting_times
            else 0.0
        )

        # Compile results
        return {
            "reward": metrics['total_reward'],
            "avg_queue_len": (np.mean(metrics['queue_length_episode'])
                              if metrics['queue_length_episode'] else 0),
            "queuing_vehicle": (np.sum(metrics['queue_length_episode'])
                                if metrics['queue_length_episode'] else 0),
            "avg_waiting_time": avg_waiting_time,
            "avg_travel_time": total_travel_time
        }

    def _save_final_metrics(self, metrics: Dict[str, float], step_num: int,
                            simulation_time: float) -> None:
        """Persist the three final comparison metrics independently of W&B."""
        work_dir = self.dic_path["PATH_TO_WORK_DIRECTORY"]
        os.makedirs(work_dir, exist_ok=True)
        payload = {
            "model_name": self.dic_traffic_env_conf.get("MODEL_NAME", ""),
            "simulation_time_s": float(simulation_time),
            "decision_steps": int(step_num),
            **metrics,
        }
        with open(os.path.join(work_dir, "final_metrics.json"), "w",
                  encoding="utf-8") as output_file:
            json.dump(payload, output_file, indent=2, ensure_ascii=False)
            output_file.write("\n")
        with open(os.path.join(work_dir, "final_metrics.txt"), "w",
                  encoding="utf-8") as output_file:
            output_file.write(
                f"Average Travel Time: {metrics['avg_travel_time']}\n"
                f"Average Queue Length: {metrics['avg_queue_len']}\n"
                f"Average Waiting Time: {metrics['avg_waiting_time']}\n"
            )

    def _calculate_travel_times(self) -> Dict[str, List[float]]:
        """
        Calculate travel times for all vehicles, including incomplete journeys.
        
        For vehicles that have not yet left the network, use simulation end time
        as their estimated departure time. This ensures all entering vehicles are
        included in the average travel time calculation, providing a more realistic
        measure of overall system efficiency.
        """
        vehicle_travel_times = {}
        run_count = self.dic_traffic_env_conf["RUN_COUNTS"]

        for intersection in self.env.list_intersection:
            arrive_leave_times = intersection.dic_vehicle_arrive_leave_time

            for vehicle_id, times in arrive_leave_times.items():
                # Skip shadow vehicles (artifacts of SUMO simulation)
                if "shadow" in vehicle_id:
                    continue

                enter_time = times["enter_time"]
                leave_time = times["leave_time"]

                # Only include vehicles that actually entered the intersection
                if np.isnan(enter_time):
                    continue

                # For vehicles that haven't left, use simulation end time
                # This is crucial for fair comparison: vehicles stuck in queue are counted
                actual_leave_time = leave_time if not np.isnan(leave_time) else run_count
                travel_time = actual_leave_time - enter_time

                if vehicle_id not in vehicle_travel_times:
                    vehicle_travel_times[vehicle_id] = [travel_time]
                else:
                    vehicle_travel_times[vehicle_id].append(travel_time)

        return vehicle_travel_times

    def _save_training_data(self, state_action_log: List[List[Dict]],
                            lane_state_log: Dict,
                            global_waiting_times: List[float]) -> None:
        """Save training data to files."""
        work_dir = self.dic_path["PATH_TO_WORK_DIRECTORY"]

        # Save state-action log
        state_action_file = os.path.join(work_dir, "state_action.json")
        dump_json(state_action_log, state_action_file, indent=4)

        # Save lane state log
        lane_state_file = os.path.join(work_dir, "lane_state.json")
        dump_json(lane_state_log, lane_state_file, indent=4)

        # Save global waiting times
        waiting_times_file = os.path.join(work_dir, "global_waiting_times.json")
        with open(waiting_times_file, "w") as f:
            json.dump(global_waiting_times, f)

