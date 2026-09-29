import copy
import json
import math
import os
import pickle
import shutil
import signal
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

import numpy as np
import wandb
from tqdm import tqdm

from .data_utils import merge, dump_json
from .sumo_env import SUMOEnv
from .v36_coordination import (
    V36CoordinationManager,
    is_new_coordination_enabled,
)
from .v32_coordination import (
    build_v32_coordination_info as build_legacy_v32_coordination,
    merge_missing_boundary_coordination,
)
# Deployment must not depend on the optional RL/VERL checkout.  This is the
# shared four-phase traffic contract; the RL observation builder uses the
# same mapping independently.
PHASE_MOVEMENTS = {
    "ETWT": ("ET", "WT"),
    "NTST": ("NT", "ST"),
    "ELWL": ("EL", "WL"),
    "NLSL": ("NL", "SL"),
}


def _extract_waiting_time(wait_info: Any) -> float:
    if isinstance(wait_info, dict):
        return float(wait_info.get("time", 0.0))
    return float(wait_info)


_MOVEMENT_SLOT_INDEX = {
    "WL": 0, "WT": 1, "WR": 2,
    "EL": 3, "ET": 4, "ER": 5,
    "NL": 6, "NT": 7, "NR": 8,
    "SL": 9, "ST": 10, "SR": 11,
}


def _build_current_movement_snapshot(intersection: Any) -> Dict[str, Dict[str, int]]:
    """Return current 150 m V/Q counts before controllers decide."""
    feature = getattr(intersection, "dic_feature", {}) or {}
    movement_ids = feature.get("traffic_movement_vehicle_ids_150m") or []
    vehicle_speed = (
        getattr(intersection, "dic_vehicle_speed_current_step", {})
        or feature.get("vehicle_speed")
        or {}
    )
    snapshot: Dict[str, Dict[str, int]] = {}
    for movement, slot_idx in _MOVEMENT_SLOT_INDEX.items():
        try:
            raw_vehicle_ids = movement_ids[slot_idx] if slot_idx < len(movement_ids) else []
        except (IndexError, KeyError, TypeError):
            raw_vehicle_ids = []
        if isinstance(raw_vehicle_ids, (str, bytes, dict)):
            raw_vehicle_ids = []
        try:
            vehicle_ids = list(raw_vehicle_ids or [])
        except TypeError:
            vehicle_ids = []
        if isinstance(vehicle_speed, dict):
            def _speed(vehicle_id: Any) -> float:
                try:
                    return float(vehicle_speed.get(vehicle_id, 1.0))
                except (TypeError, ValueError):
                    return 1.0
        elif isinstance(vehicle_speed, (list, tuple)):
            def _speed(vehicle_id: Any) -> float:
                try:
                    return float(vehicle_speed[int(vehicle_id)])
                except (IndexError, KeyError, TypeError, ValueError):
                    return 1.0
        else:
            def _speed(_vehicle_id: Any) -> float:
                return 1.0
        # V and Q must describe the exact same 150 m movement population.
        # ``lane_num_waiting_vehicle_in`` is lane-level and may include
        # vehicles outside this camera/movement slice, which can yield q > v.
        # Match MaxPressure: Q is the stopped subset of these vehicle IDs.
        stopped_count = sum(_speed(vehicle_id) < 0.1 for vehicle_id in vehicle_ids)
        snapshot[movement] = {
            "v": len(vehicle_ids),
            "q": stopped_count,
        }
    return snapshot


def _movement_history_value(history_frame: Any, movement: str) -> int:
    """Read one movement count from a SUMO history frame without guessing."""
    slot_idx = _MOVEMENT_SLOT_INDEX.get(movement)
    if slot_idx is None:
        return 0
    value: Any = 0
    if isinstance(history_frame, dict):
        for key in (movement, slot_idx, str(slot_idx)):
            if key in history_frame:
                value = history_frame[key]
                break
    elif isinstance(history_frame, (list, tuple, np.ndarray)):
        if slot_idx < len(history_frame):
            value = history_frame[slot_idx]
    try:
        if isinstance(value, bool):
            return 0
        return max(0, int(value))
    except (TypeError, ValueError, OverflowError):
        return 0


class VLMOneLine:
    """
    Main training class for traffic light control using various AI agents.
    Manages the simulation environment, agents, and training process.
    """

    def __init__(self, dic_agent_conf: Dict[str, Any], dic_traffic_env_conf: Dict[str, Any],
                 dic_path: Dict[str, str], roadnet: str, trafficflow: str, agent_class=None):
        """
        Initialize the OneLine training system.

        Args:
            dic_agent_conf: Agent configuration dictionary
            dic_traffic_env_conf: Traffic environment configuration dictionary
            dic_path: Path configuration dictionary
            roadnet: Road network identifier
            trafficflow: Traffic flow identifier
            agent_class: Optional agent class to use directly (优先级最高)
        """
        self.dic_agent_conf = dic_agent_conf
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_path = dic_path
        self.roadnet = roadnet
        self.trafficflow = trafficflow
        self.agent_class = agent_class  # Implementation note.

        # Initialize containers
        self.agents: List[Any] = []  # VLM Agents
        self.env: SUMOEnv = None
        
        # Implementation note.
        self.renderer = None  # Implementation note.
        self.tls_ids: List[str] = []  # Implementation note.
        self.image_saver = None  # Implementation note.
        self.decision_window_recorder = None  # Implementation note.
        self.previous_interval_video_paths: Dict[str, Dict[str, str]] = {}
        self.video_sft_dataset_writer = None
        self.neighbor_states: Dict[str, Dict] = {}  # {tls_id: {phase, counts, step}}
        self._network_topology: Dict[str, Any] = {}  # loaded from network_topology.json
        self._coordination_manager = None
        self._movement_route_table: Dict[str, Any] = {}

        # Load network topology JSON if available (path from VLM_CONFIG)
        vlm_config = self.dic_traffic_env_conf.get('VLM_CONFIG', {})
        topo_json_path = vlm_config.get('TOPOLOGY_JSON_PATH', None)
        if topo_json_path and os.path.exists(topo_json_path):
            try:
                with open(topo_json_path, 'r', encoding='utf-8') as f:
                    self._network_topology = json.load(f)
                print(f"Loaded network topology from: {topo_json_path}")
            except Exception as e:
                print(f"Warning: Failed to load topology {topo_json_path}: {e}")
        else:
            # Match OneLine: prefer the active scenario's data directory.
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            fallback = os.path.join(
                self.dic_path.get('PATH_TO_DATA', ''), 'network_topology.json'
            )
            if not os.path.exists(fallback):
                fallback = os.path.join(
                    project_root, 'data', 'test', '2_2', 'network_topology.json'
                )
            if os.path.exists(fallback):
                try:
                    with open(fallback, 'r', encoding='utf-8') as f:
                        self._network_topology = json.load(f)
                    print(f"Loaded network topology from fallback: {fallback}")
                except Exception as e:
                    print(f"Warning: Failed to load topology {fallback}: {e}")

        self._movement_route_table = self._load_movement_route_table()

        # Setup the environment and agents
        self.initialize()

        if is_new_coordination_enabled(self.dic_traffic_env_conf):
            self._coordination_manager = V36CoordinationManager(
                self._network_topology,
                self.session_work_dir,
                speed_mps=float(self.dic_traffic_env_conf.get(
                    "V36_COORDINATION_SPEED_MPS", 11.0)),
                view_distance_m=float(self.dic_traffic_env_conf.get(
                    "CAMERA_VIEW_DISTANCE", 150.0)),
                log_filename="llmlight_coordination_debug.jsonl",
            )

        if self.dic_traffic_env_conf.get("ENABLE_VIDEO_SFT_EXTRACTION", False):
            from .video_sft_dataset_writer import VideoSFTDatasetWriter
            vlm_config = self.dic_traffic_env_conf.get("VLM_CONFIG", {})
            self.video_sft_dataset_writer = VideoSFTDatasetWriter(
                output_dir=os.path.join(self.session_work_dir, "video_sft_raw"),
                view_distance_m=float(self.dic_traffic_env_conf.get("CAMERA_VIEW_DISTANCE", 150.0)),
                video_sample_interval_s=float(vlm_config.get("VIDEO_FRAME_SAMPLE_INTERVAL", 5.0)),
                sim_interval_s=float(self.dic_traffic_env_conf.get("INTERVAL", 1.0)),
                network_topology=self._network_topology,
                coordination_speed_mps=float(
                    self.dic_traffic_env_conf.get("V36_COORDINATION_SPEED_MPS", 11.0)),
                local_render_radius_m=vlm_config.get("LOCAL_RENDER_RADIUS_M"),
                save_coordination_frames=bool(
                    vlm_config.get("VIDEO_SAVE_COORDINATION_FRAMES", False)
                ),
                resume_existing=bool(
                    self.dic_traffic_env_conf.get("RESUME_FROM_CHECKPOINT", False)
                ),
            )

    def initialize(self) -> None:
        """
        Initialize the training environment, 3D renderer, and VLM agents.
        """
        # Setup directories
        os.makedirs(self.dic_path["PATH_TO_WORK_DIRECTORY"], exist_ok=True)

        sumo_only_mode = self.dic_traffic_env_conf.get("SUMO_ONLY_MODE", False)
        if sumo_only_mode:
            # Stage 1 is supplied directly by SUMO, so do not even construct
            # image/video writers.  Keep normal simulator logs in work_dir.
            self.session_work_dir = self.dic_path["PATH_TO_WORK_DIRECTORY"]
            self.image_saver = None
            self.decision_window_recorder = None
        else:
            # Implementation note.
            self._init_image_saver()

        # Implementation note.
        self.env = SUMOEnv(
            path_to_log=self.session_work_dir,
            path_to_work_directory=self.dic_path["PATH_TO_WORK_DIRECTORY"],
            dic_traffic_env_conf=self.dic_traffic_env_conf,
            dic_path=self.dic_path,
            inter_phase_mapping=self.dic_traffic_env_conf['INTER_PHASE_MAPPING']
        )
        
        # Implementation note.
        vlm_config = self.dic_traffic_env_conf.get('VLM_CONFIG', {})
        use_gui = vlm_config.get('SHOW_SUMO_GUI', False)
        self.env.reset(use_gui=use_gui, verbose=False)
        
        # Implementation note.
        self.tls_ids = [inter.inter_id for inter in self.env.list_intersection]
        
        # Implementation note.
        self._copy_static_sumo_configs()

        # Implementation note.
        self._configure_third_party_logging()
        
        # Implementation note.
        if sumo_only_mode:
            self.renderer = None
            self.image_saver = None
            self.decision_window_recorder = None
            print("SUMO-only mode: skip renderer and image saving.")
        else:
            self._init_renderer()
        
        # Implementation note.
        self._create_agents()

    def _init_image_saver(self) -> None:
        """
        初始化图片保存器。
        VLM 场景下必须保存图片，因为 VLM 需要从磁盘读取图片进行决策。
        """
        vlm_config = self.dic_traffic_env_conf.get('VLM_CONFIG', {})
        
        try:
            from .image_saver import ImageSaver
            from datetime import datetime
            
            scenario = vlm_config.get('SCENARIO', 'jinan')
            direction_mapping_path = vlm_config.get('DIRECTION_MAPPING_PATH', None)
            custom_session_id = self.dic_path.get("SESSION_ID")
            session_id = custom_session_id if custom_session_id is not None else datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
            
            # Implementation note.
            self.session_id = session_id
            
            # Implementation note.
            self.image_saver = ImageSaver(
                scenario=scenario,
                direction_mapping_path=direction_mapping_path,
                base_dir=self.dic_path["PATH_TO_WORK_DIRECTORY"],
                session_id="",
                verbose=False,
                enable_preprocess=vlm_config.get('IMAGE_PREPROCESS_ENABLED', True),
                left_crop=vlm_config.get('IMAGE_PREPROCESS_LEFT_CROP', 0.40),
                right_crop=vlm_config.get('IMAGE_PREPROCESS_RIGHT_CROP', 0.30),
                scale_mode=vlm_config.get('IMAGE_PREPROCESS_SCALE_MODE', 'fit_width'),
                output_width=vlm_config.get('IMAGE_SAVE_WIDTH'),
                output_height=vlm_config.get('IMAGE_SAVE_HEIGHT'),
            )
            
            # Implementation note.
            self.session_work_dir = self.dic_path["PATH_TO_WORK_DIRECTORY"]
            print(f"ImageSaver: {self.image_saver.session_dir}")

            # Implementation note.
            if vlm_config.get('DECISION_INPUT_MODE', 'image') == 'video':
                from .decision_window_recorder import DecisionWindowRecorder
                self.decision_window_recorder = DecisionWindowRecorder(
                    session_dir=self.session_work_dir,
                    direction_mapping=self.image_saver.direction_mapping,
                    fps=int(vlm_config.get('VIDEO_FPS', 5)),
                    add_labels=bool(vlm_config.get('VIDEO_ADD_LABELS', True)),
                    export_direction_videos=bool(vlm_config.get('VIDEO_EXPORT_DIRECTIONS', False)),
                    export_composite_video=bool(vlm_config.get('VIDEO_EXPORT_COMPOSITE', True)),
                    export_direction_sequence=bool(vlm_config.get('VIDEO_EXPORT_DIRECTION_SEQUENCE', False)),
                    verbose=False,
                    enable_preprocess=vlm_config.get('VIDEO_PREPROCESS_ENABLED', vlm_config.get('IMAGE_PREPROCESS_ENABLED', True)),
                    left_crop=vlm_config.get('IMAGE_PREPROCESS_LEFT_CROP', 0.40),
                    right_crop=vlm_config.get('IMAGE_PREPROCESS_RIGHT_CROP', 0.30),
                    scale_mode=vlm_config.get('IMAGE_PREPROCESS_SCALE_MODE', 'fit_width'),
                    frame_view=str(vlm_config.get('VIDEO_FRAME_VIEW', 'legacy_crop')),
                    tile_width=int(vlm_config.get('VIDEO_TILE_WIDTH', 768)),
                    sensor_type=vlm_config.get('TLS_SENSOR_TYPE', 'junction_front_all'),
                    record_mode=str(vlm_config.get('VIDEO_RECORD_MODE', 'sampled')),
                    sim_interval=float(self.dic_traffic_env_conf.get("INTERVAL", 1.0) or 1.0),
                    sample_interval=float(vlm_config.get('VIDEO_FRAME_SAMPLE_INTERVAL', 1.0) or 1.0),
                    preprocess_interpolation=vlm_config.get('VIDEO_PREPROCESS_INTERPOLATION', 'linear'),
                    video_codec=vlm_config.get('VIDEO_CODEC', 'mp4v'),
                    video_extension=vlm_config.get('VIDEO_EXTENSION', 'mp4'),
                    async_video_write=vlm_config.get('VIDEO_ASYNC_WRITE', True),
                    async_video_workers=vlm_config.get('VIDEO_ASYNC_WORKERS', 4),
                    async_video_max_pending_writes=vlm_config.get('VIDEO_ASYNC_MAX_PENDING_WRITES'),
                )
            else:
                self.decision_window_recorder = None
             
        except Exception as e:
            print(f"Warning: Failed to initialize ImageSaver: {e}")
            self.image_saver = None
            self.decision_window_recorder = None
            self.session_work_dir = self.dic_path["PATH_TO_WORK_DIRECTORY"]

    def _copy_static_sumo_configs(self) -> None:
        """
        复制静态SUMO配置文件到主目录（只需复制一次）。
        这些文件在每次运行中都是相同的，不需要保存在时间戳目录。
        """
        import shutil
        
        # Implementation note.
        sumo_config = self.dic_traffic_env_conf.get("SUMO_CONFIG_FILE")
        roadnet_file = self.dic_traffic_env_conf.get("ROADNET_FILE")
        route_file = self.dic_traffic_env_conf.get("FLOW_FILE")
        
        source_dir = self.dic_path.get("PATH_TO_DATA", self.env.path_to_work_directory)
        dest_dir = self.dic_path["PATH_TO_WORK_DIRECTORY"]
        
        # Implementation note.
        files_to_copy = []
        if sumo_config:
            files_to_copy.append(sumo_config)
        if roadnet_file:
            files_to_copy.append(roadnet_file)
        if route_file:
            files_to_copy.append(route_file)
        
        for filename in files_to_copy:
            source_path = os.path.join(source_dir, filename)
            dest_path = os.path.join(dest_dir, filename)
            
            # Implementation note.
            if os.path.exists(source_path) and not os.path.exists(dest_path):
                try:
                    shutil.copy2(source_path, dest_path)
                    print(f"Copied {filename} to {dest_dir}")
                except Exception as e:
                    print(f"Warning: Failed to copy {filename}: {e}")

    def _configure_third_party_logging(self) -> None:
        """降低TransSimHub的日志等级"""
        try:
            from loguru import logger
            # Implementation note.
            logger.remove()
            logger.add(sys.stdout, level="WARNING")
        except Exception:
            # Implementation note.
            pass

    def _init_renderer(self) -> None:
        """
        初始化 TSHubRenderer 3D 渲染器。
        支持大地图分批渲染（tls_batch_size）。
        """
        # Implementation note.
        vlm_config = self.dic_traffic_env_conf.get('VLM_CONFIG', {})
        scenario_glb_dir = vlm_config.get('SCENARIO_GLB_DIR', '')
        
        if not scenario_glb_dir:
            print("ERROR: SCENARIO_GLB_DIR is not configured in VLM_CONFIG")
            print("3D rendering will be disabled. VLM agent will fail without images.")
            self.renderer = None
            return
        
        if not os.path.exists(scenario_glb_dir):
            print(f"ERROR: SCENARIO_GLB_DIR directory does not exist:")
            print(f"  Expected: {scenario_glb_dir}")
            print(f"  Absolute path: {os.path.abspath(scenario_glb_dir)}")
            print("3D rendering will be disabled. VLM agent will fail without images.")
            self.renderer = None
            return
        
        print(f"Initializing 3D renderer with GLB dir: {scenario_glb_dir}")
        
        try:
            # Use SUMO roadnet (.net.xml) as the authoritative source for lane arrows.
            # Prefer the file that has been copied into PATH_TO_WORK_DIRECTORY for this run.
            netxml_name = self.dic_traffic_env_conf.get("ROADNET_FILE")
            netxml_path = None
            if netxml_name:
                candidate = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], netxml_name)
                if os.path.exists(candidate):
                    netxml_path = candidate
                else:
                    data_dir = self.dic_path.get("PATH_TO_DATA", self.dic_path["PATH_TO_WORK_DIRECTORY"])
                    candidate = os.path.join(data_dir, netxml_name)
                    if os.path.exists(candidate):
                        netxml_path = candidate
            
            # Implementation note.
            tls_sensor_type = vlm_config.get('TLS_SENSOR_TYPE', 'junction_front_all')
            sensor_config = {
                'tls': {
                    tls_id: {
                        'sensor_types': [tls_sensor_type],
                        'tls_camera_height': vlm_config.get('TLS_CAMERA_HEIGHT', 15)
                    }
                    for tls_id in self.tls_ids
                }
            }
            
            # Implementation note.
            tls_batch_size = vlm_config.get('TLS_BATCH_SIZE', None)
            render_batch_mode = str(
                vlm_config.get('RENDER_BATCH_MODE', 'serial') or 'serial'
            ).lower()
            render_workers = int(vlm_config.get('RENDER_WORKERS', 1) or 1)
            
            # Implementation note.
            show_3d_window = vlm_config.get('SHOW_3D_WINDOW', False)
            show_buildings = vlm_config.get('SHOW_BUILDINGS', False)
            show_arrows = vlm_config.get('SHOW_ARROWS', True)

            if render_batch_mode == 'parallel':
                if show_3d_window:
                    raise ValueError(
                        'parallel batch rendering requires offscreen rendering; '
                        'set SHOW_3D_WINDOW=false'
                    )
                if render_workers <= 0:
                    raise ValueError('RENDER_WORKERS must be positive')
                from .parallel_renderer import ParallelTSHubRenderer

                tls_init_info = self.env.get_tls_init_info(self.tls_ids)

                renderer_kwargs = {
                    'sensor_config': sensor_config,
                    'preset': vlm_config.get('RENDER_PRESET', '480P'),
                    'resolution': vlm_config.get('RENDER_RESOLUTION', 1.0),
                    'scenario_glb_dir': scenario_glb_dir,
                    'vehicle_model': vlm_config.get('VEHICLE_MODEL', 'low'),
                    'render_mode': 'offscreen',
                    'rendering_backend': vlm_config.get('RENDERING_BACKEND', 'pandagl'),
                    'show_buildings': show_buildings,
                    'keep_batch_sensors': vlm_config.get('RENDER_KEEP_BATCH_SENSORS', True),
                    'reuse_batch_sensors': vlm_config.get('RENDER_REUSE_BATCH_SENSORS', False),
                    'step_task_manager': vlm_config.get('RENDER_STEP_TASK_MANAGER', True),
                    'netxml_path': netxml_path,
                    'show_arrows': show_arrows,
                }
                self.renderer = ParallelTSHubRenderer(
                    tls_ids=self.tls_ids,
                    tls_init_info=tls_init_info,
                    batch_size=tls_batch_size,
                    workers=render_workers,
                    renderer_kwargs=renderer_kwargs,
                    timeout_s=float(vlm_config.get('RENDER_WORKER_TIMEOUT_S', 600.0)),
                )
                print(
                    'Parallel renderers initialized with '
                    f'{len(self.tls_ids)} intersections '
                    f'(batch_size={self.renderer.batch_size}, '
                    f'batches={len(self.renderer.batches)}, '
                    f'workers={self.renderer.workers}).'
                )
                return

            if render_batch_mode != 'serial':
                raise ValueError(
                    f"unsupported RENDER_BATCH_MODE={render_batch_mode!r}; "
                    "expected 'serial' or 'parallel'"
                )

            from TransSimHub.tshub.tshub_env3d.vis3d_renderer.tshub_render import TSHubRenderer

            # Implementation note.
            self.renderer = TSHubRenderer(
                simid='sumo',  # Implementation note.
                sensor_config=sensor_config,
                preset=vlm_config.get('RENDER_PRESET', '480P'),
                resolution=vlm_config.get('RENDER_RESOLUTION', 1.0),
                scenario_glb_dir=scenario_glb_dir,
                vehicle_model=vlm_config.get('VEHICLE_MODEL', 'low'),
                render_mode='onscreen' if show_3d_window else 'offscreen',
                rendering_backend=vlm_config.get('RENDERING_BACKEND', 'pandagl'),
                show_buildings=show_buildings,
                tls_batch_size=tls_batch_size,  # Implementation note.
                keep_batch_sensors=vlm_config.get('RENDER_KEEP_BATCH_SENSORS', True),
                reuse_batch_sensors=vlm_config.get('RENDER_REUSE_BATCH_SENSORS', False),
                step_task_manager=vlm_config.get('RENDER_STEP_TASK_MANAGER', True),
                netxml_path=netxml_path,
                show_arrows=show_arrows,
            )
            
            # Implementation note.
            tls_init_info = self.env.get_tls_init_info(self.tls_ids)
            self.renderer.reset(tls_init_info)
            
            # Implementation note.
            if tls_batch_size:
                print(f"TSHubRenderer initialized with {len(self.tls_ids)} intersections (batch_size={tls_batch_size}).")
            else:
                print(f"✓ TSHubRenderer initialized with {len(self.tls_ids)} intersections.")
            
        except ImportError as e:
            print(f"ERROR: Failed to import TSHubRenderer: {e}")
            print("Make sure TransSimHub is properly installed and in the Python path")
            self.renderer = None
        except Exception as e:
            print(f"ERROR: Failed to initialize TSHubRenderer:")
            print(f"  Exception type: {type(e).__name__}")
            print(f"  Exception message: {e}")
            import traceback
            traceback.print_exc()
            self.renderer = None

    def _create_agents(self) -> None:
        """
        创建 End-to-End VLM Agents（一体化决策，性能优化）。
        
        目录结构：
            {session}/conversations/  — SFT 训练数据 (.jsonl + .txt)
            {session}/api_debug/      — 原始 API 响应（每轮独立保存）
            {session}/tool_sandbox/   — 工具执行隔离区
        """
        # Implementation note.
        if self.agent_class is not None:
            SelectedAgent = self.agent_class
            agent_mode = SelectedAgent.__name__  # Implementation note.
            print(f"Using agent class from script: {SelectedAgent.__name__}")
        else:
            # Implementation note.
            agent_mode = self.dic_traffic_env_conf.get('VLM_CONFIG', {}).get('AGENT_MODE', 'e2e')
            if agent_mode == 'two_stage':
                from models.two_stages import TwoStageAgent as SelectedAgent
            else:
                from models.end_to_end_vlm_agent import EndToEndVLMAgent as SelectedAgent
            print(f"Using VLM agent mode from config: {agent_mode}")
        
        from utils.tool_executor import ToolExecutor
        
        vlm_config = self.dic_traffic_env_conf.get('VLM_CONFIG', {})
        scenario = vlm_config.get('SCENARIO', 'jinan')

        # Implementation note.
        conversations_dir = os.path.join(self.session_work_dir, "conversations")
        api_debug_dir = os.path.join(self.session_work_dir, "api_debug")
        sandbox_dir = os.path.join(self.session_work_dir, "tool_sandbox")
        os.makedirs(conversations_dir, exist_ok=True)
        os.makedirs(api_debug_dir, exist_ok=True)
        os.makedirs(sandbox_dir, exist_ok=True)
        print(
            f"[VIDEO_AGENT_LOG_DIR] conversations={conversations_dir} "
            f"stage1={os.path.join(conversations_dir, 'stage1')} "
            f"stage2={os.path.join(conversations_dir, 'stage2')}",
            flush=True,
        )

        # Implementation note.
        self.tool_executor = ToolExecutor(
            scenario=scenario,
            log_dir=self.session_work_dir,
            sandbox_dir=sandbox_dir,
        )
        print(f"ToolExecutor loaded tools: {self.tool_executor.list_tool_names()}")
        print(f"Using VLM agent mode: {agent_mode}")
        
        for intersection in self.env.list_intersection:
            tls_id = intersection.inter_id
            # Implementation note.
            phase_list = intersection.control_phases if intersection.control_phases else [f"Phase_{i}" for i in range(len(intersection.phases))]
            
            agent = SelectedAgent(
                tls_id=tls_id,
                phase_list=phase_list,
                api_url=vlm_config.get('VLM_API_URL', ''),
                api_key=vlm_config.get('VLM_API_KEY', ''),
                model_name=vlm_config.get('VLM_MODEL', 'gpt-4-vision-preview'),
                log_dir=self.session_work_dir,
                tool_executor=self.tool_executor,
                scenario=scenario,
                vlm_runtime_config=vlm_config,
                conversations_dir=conversations_dir,
                api_debug_dir=api_debug_dir,
            )
            self.agents.append(agent)

    def _checkpoint_root(self) -> str:
        return os.path.join(self.session_work_dir, "checkpoints")

    def _checkpoint_run_signature(self) -> Dict[str, Any]:
        vlm = self.dic_traffic_env_conf.get("VLM_CONFIG", {}) or {}
        return {
            "roadnet": self.roadnet,
            "trafficflow": self.trafficflow,
            "traffic_file": self.dic_traffic_env_conf.get("TRAFFIC_FILE"),
            "sumocfg_file": self.dic_traffic_env_conf.get("SUMOCFG_FILE"),
            "seed": self.dic_traffic_env_conf.get("SEED"),
            "run_counts": self.dic_traffic_env_conf.get("RUN_COUNTS"),
            "min_action_time": self.dic_traffic_env_conf.get("MIN_ACTION_TIME"),
            "data_path": os.path.abspath(self.dic_path.get("PATH_TO_DATA", "")),
            "tls_ids": list(self.tls_ids),
            "stage1_sumo_only": vlm.get("VLM_STAGE1_SUMO_ONLY"),
            "stage2_include_cooperation": vlm.get(
                "VLM_STAGE2_INCLUDE_COOPERATION"),
            "decision_model": vlm.get("DECISION_MODEL", vlm.get("VLM_MODEL")),
            "decision_api_url": vlm.get("DECISION_API_URL", vlm.get("VLM_API_URL")),
            "decision_temperature": vlm.get("DECISION_TEMPERATURE"),
            "decision_max_tokens": vlm.get("DECISION_MAX_TOKENS"),
            "decision_enable_thinking": vlm.get("DECISION_ENABLE_THINKING"),
            "reasoning_mode": vlm.get("VLM_REASONING_MODE", "adaptive"),
            "perception_model": vlm.get("PERCEPTION_MODEL", vlm.get("VLM_MODEL")),
            "perception_api_url": vlm.get("PERCEPTION_API_URL", vlm.get("VLM_API_URL")),
            "perception_temperature": vlm.get("PERCEPTION_TEMPERATURE"),
            "perception_max_tokens": vlm.get("PERCEPTION_MAX_TOKENS"),
            "perception_max_rounds": vlm.get("TWO_STAGE_PERCEPTION_MAX_ROUNDS"),
            "mode_threshold_path": vlm.get("VLM_MODE_THRESHOLD_PATH"),
            "mode_threshold_default": vlm.get("VLM_MODE_THRESHOLD_DEFAULT"),
            "parallel_enabled": vlm.get("PARALLEL_ENABLED"),
            "parallel_workers": vlm.get("PARALLEL_WORKERS"),
            "decision_input_mode": vlm.get("DECISION_INPUT_MODE"),
            "video_frame_sample_interval": vlm.get("VIDEO_FRAME_SAMPLE_INTERVAL"),
            "video_record_mode": vlm.get("VIDEO_RECORD_MODE"),
            "camera_view_distance": self.dic_traffic_env_conf.get(
                "CAMERA_VIEW_DISTANCE"),
            "local_render_radius_m": vlm.get("LOCAL_RENDER_RADIUS_M"),
            "render_preset": vlm.get("RENDER_PRESET"),
            "rendering_backend": vlm.get("RENDERING_BACKEND"),
            "render_batch_mode": vlm.get("RENDER_BATCH_MODE", "serial"),
            "render_workers": vlm.get("RENDER_WORKERS", 1),
            "render_batch_size": vlm.get("TLS_BATCH_SIZE"),
            "render_keep_batch_sensors": vlm.get(
                "RENDER_KEEP_BATCH_SENSORS"),
            "render_reuse_batch_sensors": vlm.get(
                "RENDER_REUSE_BATCH_SENSORS"),
            "render_step_task_manager": vlm.get(
                "RENDER_STEP_TASK_MANAGER"),
            "save_state_rng": self.dic_traffic_env_conf.get("SAVE_STATE_RNG", False),
        }

    @staticmethod
    def _validate_checkpoint_signature(
        saved: Dict[str, Any], current: Dict[str, Any]
    ) -> None:
        """Validate a checkpoint while remaining compatible with older manifests.

        Older checkpoints contain only the core run identity. Newer releases add
        runtime controls to the signature; those keys are intentionally allowed
        to be absent in an older checkpoint and use the current run's values.
        Values that are present in the checkpoint are always compared.
        """
        core_keys = (
            "roadnet", "trafficflow", "traffic_file", "sumocfg_file",
            "seed", "run_counts", "min_action_time", "data_path",
            "tls_ids", "stage1_sumo_only", "stage2_include_cooperation",
            "decision_model", "decision_api_url", "save_state_rng",
        )
        missing_core = [key for key in core_keys if key not in saved]
        if missing_core:
            raise RuntimeError(
                "Checkpoint is missing required run identity fields: "
                f"{missing_core!r}"
            )

        mismatches = {
            key: (saved[key], current.get(key))
            for key in saved
            if key in current and saved[key] != current.get(key)
        }
        if mismatches:
            raise RuntimeError(
                "Checkpoint does not match the current run configuration. "
                f"mismatches={mismatches!r}"
            )

        missing_runtime = [key for key in current if key not in saved]
        if missing_runtime:
            print(
                "[Resume] compatible older checkpoint: using current values "
                f"for missing runtime fields {missing_runtime!r}"
            )

    @staticmethod
    def _agent_checkpoint_state(agent: Any) -> Dict[str, Any]:
        fields = (
            "current_phase_idx", "action_history", "_history", "_last_current_v",
            "_current_v_history", "_last_decision_text", "_last_api_error",
            "_last_decision_source", "_last_decision_record", "_last_mode_routing",
            "_current_step_dir_counts",
        )
        return {name: copy.deepcopy(getattr(agent, name)) for name in fields
                if hasattr(agent, name)}

    def _collector_checkpoint_state(self) -> Dict[str, Any]:
        fields = (
            "previous_interval_video_paths", "neighbor_states", "action_records",
            "frame_remaining_v", "frame_remaining_queue", "_active_decision_step",
            "phase_histories",
        )
        state = {name: copy.deepcopy(getattr(self, name)) for name in fields
                 if hasattr(self, name)}
        if self.image_saver is not None:
            image_saver_fields = (
                "current_batch_index", "saved_count", "step_count",
            )
            state["image_saver"] = {
                name: copy.deepcopy(getattr(self.image_saver, name))
                for name in image_saver_fields
                if hasattr(self.image_saver, name)
            }
        writer = self.video_sft_dataset_writer
        if writer is not None:
            writer_fields = (
                "_pending_coordination", "_collection_start_s", "_sample_count",
                "_valid_sample_count", "_audit_records",
            )
            state["video_sft_writer"] = {
                name: copy.deepcopy(getattr(writer, name)) for name in writer_fields
                if hasattr(writer, name)
            }
        if self._coordination_manager is not None:
            state["coordination_pending"] = copy.deepcopy(
                self._coordination_manager.pending)
        return state

    def _checkpoint_append_file_sizes(self) -> Dict[str, int]:
        candidates = [getattr(self, "metrics_log_path", None)]
        candidates.extend(
            getattr(inter, "log_file_path", None)
            for inter in self.env.list_intersection
        )
        recorder = self.decision_window_recorder
        if recorder is not None:
            candidates.append(getattr(recorder, "metadata_path", None))
        writer = self.video_sft_dataset_writer
        if writer is not None:
            candidates.extend([
                getattr(writer, "jsonl_path", None),
                getattr(writer, "complete_jsonl_path", None),
            ])
        coordinator = self._coordination_manager
        if coordinator is not None:
            for name in ("log_path", "debug_log_path", "_log_path"):
                candidates.append(getattr(coordinator, name, None))
        for agent in self.agents:
            candidates.extend([
                getattr(agent, "log_path", None),
                getattr(agent, "stage1_log_path", None),
                getattr(agent, "stage2_log_path", None),
                (getattr(agent, "vlm_runtime_config", {}) or {}).get(
                    "VLM_MODE_ROUTING_LOG"),
            ])
        sizes = {}
        for path in candidates:
            if path and os.path.isfile(path):
                sizes[os.path.abspath(path)] = os.path.getsize(path)
        return sizes

    @staticmethod
    def _validate_append_file_sizes(sizes: Dict[str, int]) -> None:
        for path, size in sizes.items():
            if not os.path.isfile(path):
                if size:
                    raise RuntimeError(f"Checkpoint append-only file is missing: {path}")
                continue
            current_size = os.path.getsize(path)
            if current_size < size:
                raise RuntimeError(
                    f"Checkpoint append-only file is shorter than expected: {path} "
                    f"({current_size} < {size})")

    @classmethod
    def _restore_append_file_sizes(cls, sizes: Dict[str, int]) -> None:
        cls._validate_append_file_sizes(sizes)
        for path, size in sizes.items():
            if not os.path.isfile(path):
                continue
            current_size = os.path.getsize(path)
            if current_size > size:
                with open(path, "r+b") as handle:
                    handle.truncate(size)

    def _restore_collector_checkpoint_state(self, state: Dict[str, Any]) -> None:
        for name, value in state.items():
            if name not in {
                "image_saver", "video_sft_writer", "coordination_pending",
            }:
                setattr(self, name, copy.deepcopy(value))
        if self.image_saver is not None:
            for name, value in state.get("image_saver", {}).items():
                setattr(self.image_saver, name, copy.deepcopy(value))
        writer = self.video_sft_dataset_writer
        if writer is not None:
            for name, value in state.get("video_sft_writer", {}).items():
                setattr(writer, name, copy.deepcopy(value))
        if self._coordination_manager is not None and "coordination_pending" in state:
            self._coordination_manager.pending = copy.deepcopy(
                state["coordination_pending"])

    def _prune_resumable_checkpoints(self, keep: int) -> None:
        root = os.path.abspath(self._checkpoint_root())
        complete = []
        for name in os.listdir(root):
            path = os.path.abspath(os.path.join(root, name))
            if (not name.startswith("step_") or name.endswith(".tmp")
                    or not os.path.isdir(path)
                    or os.path.commonpath((root, path)) != root):
                continue
            if all(os.path.isfile(os.path.join(path, required)) for required in (
                    "sumo_state.xml", "runtime_state.pkl", "manifest.json")):
                complete.append(path)
        complete.sort(key=lambda path: (os.path.getmtime(path), path), reverse=True)
        for stale_path in complete[max(1, int(keep)):]:
            shutil.rmtree(stale_path)

    def _save_resumable_checkpoint(self, step_num: int,
                                   training_metrics: Dict[str, Any],
                                   state_action_log: List[List[Dict]]) -> None:
        root = self._checkpoint_root()
        os.makedirs(root, exist_ok=True)
        final_dir = os.path.join(root, f"step_{step_num:06d}")
        suffix = 1
        while os.path.exists(final_dir) or os.path.exists(final_dir + ".tmp"):
            final_dir = os.path.join(root, f"step_{step_num:06d}_{suffix}")
            suffix += 1
        temp_dir = final_dir + ".tmp"
        os.makedirs(temp_dir)

        try:
            sumo_path = os.path.join(temp_dir, "sumo_state.xml")
            if self.env.snapshot(sumo_path) is None:
                raise RuntimeError("SUMO refused to save the checkpoint state")
            checkpoint_time = float(self.env.get_current_time())
            payload = {
                "version": 1,
                "run_signature": self._checkpoint_run_signature(),
                "next_step": int(step_num),
                "sim_time": checkpoint_time,
                "environment": self.env.capture_runtime_state(),
                "training_metrics": copy.deepcopy(training_metrics),
                "state_action_log": copy.deepcopy(state_action_log),
                "agents": {
                    agent.tls_id: self._agent_checkpoint_state(agent)
                    for agent in self.agents
                },
                "collector": self._collector_checkpoint_state(),
                "append_file_sizes": self._checkpoint_append_file_sizes(),
            }
            with open(os.path.join(temp_dir, "runtime_state.pkl"), "wb") as handle:
                pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(temp_dir, "manifest.json"), "w", encoding="utf-8") as handle:
                json.dump({
                    "version": 1, "next_step": int(step_num),
                    "sim_time": checkpoint_time,
                }, handle, indent=2)
            os.replace(temp_dir, final_dir)
            latest_tmp = os.path.join(root, "latest.json.tmp")
            with open(latest_tmp, "w", encoding="utf-8") as handle:
                json.dump({"checkpoint": os.path.basename(final_dir)}, handle)
            os.replace(latest_tmp, os.path.join(root, "latest.json"))
            try:
                self._prune_resumable_checkpoints(
                    int(self.dic_traffic_env_conf.get("CHECKPOINT_KEEP", 2)))
            except OSError as exc:
                print(f"  [Resume checkpoint] warning: could not prune old checkpoints: {exc}")
        except BaseException:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise
        print(f"  [Resume checkpoint] saved step={step_num}, sim_time={self.env.get_current_time():.0f}s")

    def _load_resumable_checkpoint(self) -> Dict[str, Any]:
        root = self._checkpoint_root()
        candidates = []
        if os.path.isdir(root):
            for name in os.listdir(root):
                checkpoint_dir = os.path.abspath(os.path.join(root, name))
                if (not name.startswith("step_") or name.endswith(".tmp")
                        or not os.path.isdir(checkpoint_dir)
                        or os.path.commonpath((os.path.abspath(root), checkpoint_dir))
                        != os.path.abspath(root)):
                    continue
                required = (
                    "manifest.json", "runtime_state.pkl", "sumo_state.xml",
                )
                if not all(os.path.isfile(os.path.join(checkpoint_dir, item))
                           for item in required):
                    continue
                try:
                    with open(os.path.join(checkpoint_dir, "manifest.json"),
                              "r", encoding="utf-8") as handle:
                        candidate_manifest = json.load(handle)
                    candidates.append((
                        int(candidate_manifest["next_step"]),
                        os.path.getmtime(checkpoint_dir),
                        checkpoint_dir,
                    ))
                except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                    continue
        if not candidates:
            raise FileNotFoundError(
                f"No complete resumable checkpoint found under: {root}")
        _, _, checkpoint_dir = max(candidates)
        manifest_path = os.path.join(checkpoint_dir, "manifest.json")
        state_path = os.path.join(checkpoint_dir, "runtime_state.pkl")
        sumo_path = os.path.join(checkpoint_dir, "sumo_state.xml")
        for path in (manifest_path, state_path, sumo_path):
            if not os.path.isfile(path):
                raise RuntimeError(f"Incomplete resumable checkpoint: missing {path}")
        with open(state_path, "rb") as handle:
            payload = pickle.load(handle)
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if payload.get("version") != 1 or manifest.get("version") != 1:
            raise RuntimeError("Unsupported resumable checkpoint version")
        if manifest.get("next_step") != payload.get("next_step"):
            raise RuntimeError("Checkpoint manifest and runtime step do not match")
        if not math.isclose(float(manifest.get("sim_time", -1)),
                            float(payload.get("sim_time", -2)), abs_tol=1e-6):
            raise RuntimeError("Checkpoint manifest and runtime simulation time do not match")
        expected_signature = payload.get("run_signature")
        actual_signature = self._checkpoint_run_signature()
        if not isinstance(expected_signature, dict):
            raise RuntimeError("Checkpoint is missing a valid run signature")
        self._validate_checkpoint_signature(expected_signature, actual_signature)
        append_file_sizes = payload.get("append_file_sizes", {})
        self._validate_append_file_sizes(append_file_sizes)
        self.env.load_from_file(sumo_path, quiet=True, raise_on_error=True)
        loaded_time = float(self.env.get_current_time())
        if not math.isclose(loaded_time, float(payload["sim_time"]), abs_tol=1e-6):
            raise RuntimeError(
                "Loaded SUMO time does not match checkpoint runtime state: "
                f"{loaded_time} != {payload['sim_time']}")
        self.env.restore_runtime_state(payload["environment"])
        agents_by_tls = {agent.tls_id: agent for agent in self.agents}
        saved_agents = payload.get("agents", {})
        if set(saved_agents) != set(agents_by_tls):
            raise RuntimeError("Checkpoint agent IDs do not match the loaded network")
        for tls_id, agent_state in saved_agents.items():
            for name, value in agent_state.items():
                setattr(agents_by_tls[tls_id], name, copy.deepcopy(value))
        self._restore_collector_checkpoint_state(payload.get("collector", {}))
        self._restore_append_file_sizes(append_file_sizes)
        checkpoint_name = os.path.basename(checkpoint_dir)
        print(
            f"[Resume] restored {checkpoint_name}: next_step={payload['next_step']}, "
            f"sim_time={self.env.get_current_time():.0f}s"
        )
        return payload

    def train(self, round: int) -> Dict[str, float]:
        """
        执行 VLM 交通信号控制仿真。
        
        流程：渲染图片 → VLM 决策 → 执行动作 → 更新指标

        Args:
            round: 当前轮次

        Returns:
            包含训练结果和指标的字典
        """
        print("================ Start VLM Simulation ================")

        # Implementation note.
        total_run_cnt = self.dic_traffic_env_conf["RUN_COUNTS"]
        min_action_time = self.dic_traffic_env_conf.get("MIN_ACTION_TIME", 15)
        vlm_config = self.dic_traffic_env_conf.get('VLM_CONFIG', {})
        if vlm_config.get('DECISION_INPUT_MODE', 'image') == 'video':
            configured_window = float(vlm_config.get('VIDEO_WINDOW_SEC', min_action_time))
            if abs(configured_window - float(min_action_time)) > 1e-6:
                print(
                    f"Warning: VIDEO_WINDOW_SEC={configured_window} but MIN_ACTION_TIME={min_action_time}. "
                    "Current implementation records one clip per decision interval, so the effective window "
                    "length follows MIN_ACTION_TIME."
                )

        # Implementation note.
        vlm_config = self.dic_traffic_env_conf.get('VLM_CONFIG', {})
        use_gui = vlm_config.get('SHOW_SUMO_GUI', False)
        state = self.env.reset(
            use_gui=use_gui,
            seed=self.dic_traffic_env_conf.get("SEED"),
            verbose=False,
        )
        resume_enabled = bool(
            self.dic_traffic_env_conf.get("RESUME_FROM_CHECKPOINT", False))
        training_metrics = self._initialize_training_metrics(resume=resume_enabled)
        state_action_log = [[] for _ in range(len(self.tls_ids))]
        resume_payload = None
        if resume_enabled:
            resume_payload = self._load_resumable_checkpoint()
            training_metrics = resume_payload["training_metrics"]
            state_action_log = resume_payload["state_action_log"]

        # Setup logging
        logger = self._setup_wandb_logger(round)

        results = {}
        start_time = time.time()

        # Implementation note.
        sim_start_time = self.dic_traffic_env_conf.get("SIM_START_TIME", 0)
        if sim_start_time > 0 and resume_payload is None:
            print(f"[Warm-up] Fast-forwarding simulation from 0 to {sim_start_time}s ...")
            warmup_action = {tls_id: 0 for tls_id in self.tls_ids}  # Implementation note.
            warmup_step_size = min_action_time
            warmup_steps = 0
            while self.env.get_current_time() < sim_start_time:
                _, _, warmup_done, _ = self.env.step(warmup_action, warmup_step_size)
                warmup_steps += 1
                if warmup_done:
                    break
            cur_t = self.env.get_current_time()
            # Implementation note.
            try:
                veh_count = self.env.traci_conn.vehicle.getIDCount()
            except Exception:
                veh_count = "N/A"
            print(f"[Warm-up] Done. sim_time={cur_t:.0f}s, warm-up steps={warmup_steps}, vehicles on road={veh_count}")

        # Implementation note.
        current_time = self.env.get_current_time()
        step_num = int(resume_payload["next_step"]) if resume_payload else 0
        done = False
        overall_results = {"avg_travel_time": [], "avg_queue_len": [], "avg_waiting_time": []}
        last_checkpoint_time = current_time  # Implementation note.
        CHECKPOINT_INTERVAL = 30  # Implementation note.

        # Implementation note.
        _interrupted = [False]
        _orig_handler = signal.getsignal(signal.SIGINT)
        def _sigint_handler(signum, frame):
            if _interrupted[0]:
                print("\n强制退出")
                sys.exit(1)
            _interrupted[0] = True
            print("\n[SIGINT] 正在保存数据后退出，再次 Ctrl+C 强制退出...")
        signal.signal(signal.SIGINT, _sigint_handler)

        checkpoint_every_steps = int(
            self.dic_traffic_env_conf.get("CHECKPOINT_EVERY_STEPS", 0) or 0)
        progress_initial = max(0, int(current_time - sim_start_time))
        with tqdm(total=total_run_cnt - sim_start_time, initial=progress_initial,
                  desc="VLM Simulation", unit="step") as pbar:
            while not done and current_time < total_run_cnt and not _interrupted[0]:
                # Implementation note.
                try:
                    veh_count = self.env.traci_conn.vehicle.getIDCount()
                except Exception:
                    veh_count = "N/A"
                print(f"\n[Step {step_num}] sim_time={current_time:.0f}s, vehicles={veh_count}")

                # Implementation note.
                # Decisions consume only the previous window, which was
                # finalized after the preceding env.step. The current window
                # is opened after this call and cannot leak into this action.
                saved_paths = self._prepare_decision_visual_inputs(step_num)
                
                # Implementation note.
                action_list = self._get_vlm_actions(saved_paths, step_num, state_action_log)

                # Implementation note.
                inner_step_callback = None
                if self.decision_window_recorder is not None:
                    self.decision_window_recorder.begin_interval(
                        decision_step=step_num,
                        sim_start_sec=current_time,
                    )
                    inner_step_callback = self._record_decision_interval_frame
                if self.video_sft_dataset_writer is not None:
                    self.video_sft_dataset_writer.begin_interval(
                        step_num, current_time, tls_ids=self.tls_ids
                    )

                # Implementation note.
                # Implementation note.
                # Implementation note.
                _prior_callback = inner_step_callback
                _agents_ref = self.agents
                _intersections_ref = self.env.list_intersection

                def _inner_with_sumo_collect(inner_i, env,
                                             _cb=_prior_callback,
                                             _agents=_agents_ref,
                                             _inters=_intersections_ref):
                    # Read the post-simulationStep clock once and use it for
                    # the video frame, SFT snapshot and legacy agent history.
                    sim_time_s = float(env.get_current_time())
                    if _cb is not None:
                        _cb(inner_i=inner_i, env=env, sim_time_s=sim_time_s)
                    if self.video_sft_dataset_writer is not None:
                        self.video_sft_dataset_writer.capture_tick(
                            inner_i,
                            env,
                            sim_time_s=sim_time_s,
                        )
                    for agent, inter in zip(_agents, _inters):
                        if hasattr(agent, 'collect_sumo_frame'):
                            agent.collect_sumo_frame(
                                inter.dic_feature, sim_time_s,
                                inter.dic_vehicle_distance_current_step,
                                inter.dic_vehicle_speed_current_step)

                inner_step_callback = _inner_with_sumo_collect
                
                # Implementation note.
                try:
                    next_state, reward, done, _ = self.env.step(
                        action_list,
                        min_action_time,
                        inner_step_callback=inner_step_callback,
                    )
                except BaseException as exc:
                    sim_end_s = self.env.get_current_time()
                    video_details = {}
                    if self.decision_window_recorder is not None:
                        video_details = self.decision_window_recorder.finalize_interval(
                            sim_end_sec=sim_end_s,
                            force_discard=True,
                            return_details=True,
                        )
                    if self.video_sft_dataset_writer is not None:
                        self.video_sft_dataset_writer.finalize_interval(
                            sim_end_s=sim_end_s,
                            video_details=video_details,
                            status="incomplete",
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    raise

                video_details = {}
                if self.decision_window_recorder is not None:
                    video_details = self.decision_window_recorder.finalize_interval(
                        sim_end_sec=self.env.get_current_time(),
                        return_details=True,
                    )
                    self.previous_interval_video_paths = video_details.get("paths", {})
                if self.video_sft_dataset_writer is not None:
                    self.video_sft_dataset_writer.finalize_interval(
                        sim_end_s=self.env.get_current_time(),
                        video_details=video_details,
                    )
                
                # Implementation note.
                self._update_training_metrics(training_metrics, reward, step_num)
                
                # Implementation note.
                state = next_state
                step_num += 1
                current_time = self.env.get_current_time()

                if checkpoint_every_steps and step_num % checkpoint_every_steps == 0:
                    self._save_resumable_checkpoint(
                        step_num, training_metrics, state_action_log)

                # Implementation note.
                if current_time % 3600 == 0:
                    results = self._calculate_final_results(training_metrics)
                    logger.log(results)
                    overall_results["avg_travel_time"].append(results["avg_travel_time"])
                    overall_results["avg_queue_len"].append(results["avg_queue_len"])
                    overall_results["avg_waiting_time"].append(results["avg_waiting_time"])

                # Implementation note.
                if current_time - last_checkpoint_time >= CHECKPOINT_INTERVAL:
                    try:
                        self.env.batch_log()
                        print(f"  [Checkpoint] vehicle_logs 已保存 (sim_time={current_time:.0f}s)")
                    except Exception as e:
                        print(f"  [Checkpoint] 保存失败: {e}")
                    last_checkpoint_time = current_time

                # Implementation note.
                pbar.update(min_action_time)

        # Implementation note.
        signal.signal(signal.SIGINT, _orig_handler)
        if _interrupted[0]:
            print("\n[SIGINT] 仿真被中断，正在保存数据...")
            if checkpoint_every_steps:
                self._save_resumable_checkpoint(
                    step_num, training_metrics, state_action_log)

        # Implementation note.
        self._save_training_data(state_action_log, training_metrics['global_waiting_times'])
        
        # Implementation note.
        results = self._calculate_final_results(training_metrics)
        
        print(f"============== End VLM Simulation ==============")
        print(f"Average Travel Time: {results.get('avg_travel_time', 0):.2f}\n"
              f"Average Queue Length: {results.get('avg_queue_len', 0):.2f}\n"
              f"Average Waiting Time: {results.get('avg_waiting_time', 0):.2f}")
        em_count = results.get('emergency_count', 0)
        if em_count > 0:
            print(f"AETT (Emergency Travel Time): {results.get('aett', 0):.2f}\n"
                  f"AEWT (Emergency Waiting Time): {results.get('aewt', 0):.2f}\n"
                  f"Emergency Vehicles: {em_count}")
        
        # Implementation note.
        logger.log({
            "final_avg_travel_time": results.get("avg_travel_time", 0),
            "final_avg_queue_len": results.get("avg_queue_len", 0),
            "final_avg_waiting_time": results.get("avg_waiting_time", 0)
        })
        
        wandb.finish()
        print("Simulation time: ", time.time() - start_time)
        self.env.batch_log()
        
        # Implementation note.
        if self.renderer is not None:
            try:
                self.renderer.destroy()
            except SystemExit:
                pass  # Implementation note.
            except Exception:
                pass
            self.renderer = None

        return results

    def _initialize_training_metrics(self, resume: bool = False) -> Dict[str, Any]:
        """Initialize metrics tracking for training."""
        # Implementation note.
        metrics_log_dir = os.path.join(self.session_work_dir, "metrics")
        os.makedirs(metrics_log_dir, exist_ok=True)
        self.metrics_log_path = os.path.join(metrics_log_dir, "step_metrics.csv")
        if not resume or not os.path.isfile(self.metrics_log_path):
            with open(self.metrics_log_path, 'w', encoding='utf-8') as f:
                f.write("step,time,avg_queue_len,avg_waiting_time,reward\n")
        
        return {
            'total_reward': 0.0,
            'queue_length_episode': [],
            'waiting_time_episode': [],
            'global_waiting_times': [],
            'emergency_waiting_time_episode': [],
        }

    def _setup_wandb_logger(self, round: int) -> Any:
        """Setup Weights & Biases logging for the training session."""
        all_config = merge(merge(self.dic_agent_conf, self.dic_path), self.dic_traffic_env_conf)
        phase_count = len(self.dic_traffic_env_conf['PHASE'])

        try:
            return wandb.init(
                project=self.dic_traffic_env_conf['PROJECT_NAME'],
                group=f"{self.dic_traffic_env_conf['MODEL_NAME']}-{self.roadnet}-{self.trafficflow}-{phase_count}_Phases",
                name=f"round_{round}",
                config=all_config,
                settings=wandb.Settings(init_timeout=60),
            )
        except Exception as e:
            print(f"Warning: wandb init failed ({e}), falling back to disabled mode.")
            try:
                wandb.finish(quiet=True)
            except Exception:
                pass
            os.environ["WANDB_MODE"] = "disabled"
            return wandb.init(
                project=self.dic_traffic_env_conf['PROJECT_NAME'],
                group=f"{self.dic_traffic_env_conf['MODEL_NAME']}-{self.roadnet}-{self.trafficflow}-{phase_count}_Phases",
                name=f"round_{round}",
                config=all_config,
                mode="disabled",
            )

    def _get_render_tshub_obs(self, tls_ids=None):
        vlm_config = self.dic_traffic_env_conf.get('VLM_CONFIG', {})
        radius = vlm_config.get('LOCAL_RENDER_RADIUS_M', 200.0)
        if radius is None:
            return self.env.get_tshub_obs()
        return self.env.get_tshub_obs(tls_ids=tls_ids, radius=float(radius))

    def _prepare_decision_visual_inputs(self, step_num: int) -> Dict[str, List[str]]:
        if self.dic_traffic_env_conf.get("SUMO_ONLY_MODE", False):
            return {tls_id: [] for tls_id in self.tls_ids}

        vlm_config = self.dic_traffic_env_conf.get('VLM_CONFIG', {})
        is_video_mode = vlm_config.get('DECISION_INPUT_MODE', 'image') == 'video'
        include_current_images = vlm_config.get('VIDEO_INCLUDE_CURRENT_IMAGES', True)
        if is_video_mode and not include_current_images:
            return {tls_id: [] for tls_id in self.tls_ids}

        return self._render_and_save_images(step_num)

    def _record_decision_interval_frame(self, inner_i: int, env: SUMOEnv,
                                        sim_time_s: float = None) -> None:
        """Capture one inner-step frame for the next decision-window video clip."""
        if self.renderer is None or self.decision_window_recorder is None:
            return

        vlm_config = self.dic_traffic_env_conf.get('VLM_CONFIG', {})
        record_mode = str(vlm_config.get('VIDEO_RECORD_MODE', 'sampled') or 'sampled').lower()
        sample_interval = float(vlm_config.get('VIDEO_FRAME_SAMPLE_INTERVAL', 1.0))
        sim_interval = float(self.dic_traffic_env_conf.get("INTERVAL", 1.0))

        if record_mode not in ('continuous', 'full', 'all'):
            record_mode = 'sampled'

        if record_mode == 'sampled' and sample_interval > sim_interval:
            stride = max(1, int(round(sample_interval / sim_interval)))
            # inner_i is zero-based but the callback runs after each SUMO tick.
            # Sampling the completed tick keeps a 5 s window aligned to
            # 5, 10, ..., 30 s and makes the final video frame the label time.
            if (inner_i + 1) % stride != 0:
                return

        sim_time = (
            float(env.get_current_time())
            if sim_time_s is None else float(sim_time_s)
        )
        current_env_time = float(env.get_current_time())
        if abs(current_env_time - sim_time) > 1e-6:
            raise RuntimeError(
                "SUMO time changed before video rendering: "
                f"captured={sim_time} current={current_env_time}"
            )

        if getattr(self.renderer, 'is_parallel', False):
            batch_tls_ids = self.renderer.get_all_batch_tls_ids()
            batch_inputs = [
                (current_batch_tls, self._get_render_tshub_obs(current_batch_tls))
                for current_batch_tls in batch_tls_ids
            ]
            sensor_batches = self.renderer.step_all(
                batch_inputs,
                should_count_vehicles=False,
            )
            for current_batch_tls, sensor_data in zip(
                batch_tls_ids, sensor_batches, strict=True
            ):
                if sensor_data:
                    self.decision_window_recorder.add_sensor_data(
                        sensor_data,
                        current_batch_tls,
                        sim_time=sim_time,
                    )
            return

        batch_info = self.renderer.get_batch_info() if hasattr(self.renderer, 'get_batch_info') else {'mode': 'all'}

        if batch_info.get('mode') == 'batch':
            total_batches = batch_info.get('total_batches', 1)

            for batch_idx in range(total_batches):
                current_batch_tls = (
                    self.renderer.get_current_batch_tls_ids()
                    if hasattr(self.renderer, 'get_current_batch_tls_ids') else self.tls_ids
                )
                tshub_obs = self._get_render_tshub_obs(current_batch_tls)
                sensor_data = self.renderer.step(tshub_obs, should_count_vehicles=False)
                if sensor_data:
                    self.decision_window_recorder.add_sensor_data(
                        sensor_data,
                        current_batch_tls,
                        sim_time=sim_time,
                    )

                if batch_idx < total_batches - 1 and hasattr(self.renderer, 'switch_to_next_batch'):
                    self.renderer.switch_to_next_batch()

            if hasattr(self.renderer, 'switch_to_batch'):
                self.renderer.switch_to_batch(0)

            return

        tshub_obs = self._get_render_tshub_obs(self.tls_ids)
        sensor_data = self.renderer.step(tshub_obs, should_count_vehicles=False)
        if sensor_data:
            self.decision_window_recorder.add_sensor_data(
                sensor_data,
                self.tls_ids,
                sim_time=sim_time,
            )

    def _build_agent_input(self, tls_id: str, image_paths: List[str]) -> Any:
        """Build a decoupled decision input package for image or video VLM modes."""
        vlm_config = self.dic_traffic_env_conf.get('VLM_CONFIG', {})
        mode = vlm_config.get('DECISION_INPUT_MODE', 'image')

        if mode != 'video':
            return image_paths

        include_current_images = vlm_config.get('VIDEO_INCLUDE_CURRENT_IMAGES', True)
        current_images = image_paths if include_current_images else []
        video_paths = self.previous_interval_video_paths.get(tls_id, {})
        return {
            "mode": "video",
            "image_paths": current_images,
            "video_paths": video_paths,
            "video_window_sec": vlm_config.get('VIDEO_WINDOW_SEC', self.dic_traffic_env_conf.get("MIN_ACTION_TIME", 35)),
        }

    def _load_movement_route_table(self) -> Dict[str, Any]:
        """Load the per-scenario route table used to build Stage 2 context."""
        vlm_config = self.dic_traffic_env_conf.get('VLM_CONFIG', {}) or {}
        scenario = str(vlm_config.get('SCENARIO', self.dic_traffic_env_conf.get('SCENARIO', 'jinan'))).lower()
        for candidate in ('newyork7x7_v1', 'newyork16x3_v1', 'newyork16x3', 'jinan', 'hangzhou', 'newyork'):
            if scenario.startswith(candidate):
                scenario = candidate
                break
        candidates = []
        configured = vlm_config.get('MOVEMENT_ROUTE_TABLE_PATH') or vlm_config.get('ROUTE_TABLE_PATH')
        if configured:
            candidates.append(os.fspath(configured))
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidates.extend([
            os.path.join(project_root, 'scenario_assets', 'movement_routes', f'movement_routes_{scenario}.json'),
            os.path.join(project_root, 'training', 'ada_v1', 'verl', 'v35_online_cooperative_grpo', 'artifacts', f'movement_routes_{scenario}.json'),
        ])
        for path in candidates:
            if not path or not os.path.isfile(path):
                continue
            try:
                with open(path, 'r', encoding='utf-8') as handle:
                    table = json.load(handle)
                if isinstance(table, dict) and isinstance(table.get('routes'), dict):
                    print(f"Loaded movement route table from: {path}")
                    return table
            except Exception as exc:
                print(f"Warning: Failed to load movement route table {path}: {exc}")
        print(f"Warning: movement route table not found for scenario {scenario}; falling back to local neighbor state only")
        return {}

    def _sumo_stage1_perception(
        self, intersection: Any, agent: Any, step_num: int
    ) -> Dict[str, Any]:
        """Build the Stage 1 canonical schema from the current SUMO snapshot.

        This is intentionally a narrow recovery path for malformed structured
        model output.  Local fields come from SUMO directly; internal arrival
        fields reuse the same movement-level outbound snapshots as SFT gold.
        """
        feature = getattr(intersection, "dic_feature", {}) or {}
        current = _build_current_movement_snapshot(intersection)
        v_history = feature.get("v9_cycle_150m_history") or []
        q_history = feature.get("v26_cycle_queue_history") or []

        def _delta(history: Any, movement: str) -> int:
            if not isinstance(history, (list, tuple)) or len(history) < 2:
                return 0
            return _movement_history_value(history[-1], movement) - _movement_history_value(
                history[0], movement
            )

        def _boundary_delta(history: Any, movement: str) -> int:
            # The Stage 1 contract defines boundary arrivals as V30 - V15.
            if not isinstance(history, (list, tuple)) or len(history) < 6:
                return 0
            return max(
                0,
                _movement_history_value(history[-1], movement)
                - _movement_history_value(history[2], movement),
            )

        route_rows = self._movement_route_table.get("routes", {})
        route_rows = route_rows if isinstance(route_rows, dict) else {}
        # ``movement_routes`` describes a source's outbound movements.  For
        # Stage 1, boundary status is instead a property of the *target*
        # intersection entry direction: only a non-boundary source route that
        # terminates here provides an upstream coordination frame.
        tls_id = str(getattr(intersection, "inter_id", ""))
        available_entries = set()
        for source_data in route_rows.values():
            movements = source_data.get("movements", {}) if isinstance(source_data, dict) else {}
            for route in movements.values() if isinstance(movements, dict) else ():
                if (
                    isinstance(route, dict)
                    and not route.get("is_boundary")
                    and str(route.get("receiver_id", "")) == tls_id
                    and str(route.get("receiver_entry_direction", "")).upper()
                    in {"E", "W", "N", "S"}
                ):
                    available_entries.add(str(route["receiver_entry_direction"]).upper())

        movement_entry = {
            "ET": "E", "EL": "E", "WT": "W", "WL": "W",
            "NT": "N", "NL": "N", "ST": "S", "SL": "S",
        }

        intersections_by_id = {
            str(getattr(item, "inter_id", "")): item
            for item in getattr(getattr(self, "env", None), "list_intersection", []) or []
        }
        camera_view_distance = float(
            self.dic_traffic_env_conf.get("CAMERA_VIEW_DISTANCE", 150.0))

        def _internal_arrival_count(target_movement: str) -> int | None:
            """Match the SFT gold rule: sum routed upstream movement frames."""
            entry = target_movement[0]
            total = 0
            found_route = False
            for source_id, source_data in route_rows.items():
                movements = source_data.get("movements", {}) if isinstance(source_data, dict) else {}
                for source_movement, route in movements.items() if isinstance(movements, dict) else ():
                    if not (
                        isinstance(route, dict)
                        and str(route.get("receiver_id", "")) == tls_id
                        and str(route.get("receiver_entry_direction", "")).upper() == entry
                        and not route.get("is_boundary")
                    ):
                        continue
                    found_route = True
                    distance = route.get("distance_m")
                    try:
                        if float(distance) <= camera_view_distance:
                            # The upstream link is inside this intersection's
                            # local camera view, so its coordination count is
                            # explicitly zero rather than double-counted.
                            continue
                    except (TypeError, ValueError):
                        pass
                    source = intersections_by_id.get(str(source_id))
                    snapshots = (getattr(source, "dic_feature", {}) or {}).get(
                        "v36_cycle_outbound_snapshots", []
                    ) if source is not None else []
                    frames = [frame for frame in snapshots if isinstance(frame, dict)]
                    if not frames:
                        continue
                    frame = max(frames, key=lambda item: int(item.get("frame_index", 0) or 0))
                    if distance is not None:
                        try:
                            from .coordination_frame_selector import select_coordination_frame
                            selected, _ = select_coordination_frame(float(distance))
                            frame = next((item for item in frames
                                          if int(item.get("frame_index", -1)) == int(selected.frame_index)), None)
                        except (ImportError, TypeError, ValueError):
                            frame = None
                    if frame is None:
                        continue
                    by_movement = frame.get("outbound_vehicle_ids_by_movement")
                    if not isinstance(by_movement, dict):
                        continue
                    values = by_movement.get(source_movement) or []
                    distances = frame.get("outbound_distance_from_source_m") or {}
                    if isinstance(distances, dict) and distances:
                        def _distance(value: Any) -> float:
                            try:
                                return float(value)
                            except (TypeError, ValueError):
                                return float("inf")
                        values = [
                            value for value in values
                            if _distance(distances.get(value, distances.get(str(value), float("inf")))) <= 150.0
                        ]
                    total += len({str(value) for value in values})
            if not found_route:
                return None
            return total

        phase_list = getattr(agent, "phase_list", None) or list(PHASE_MOVEMENTS)
        phase_index = getattr(agent, "current_phase_idx", 0)
        current_phase = (
            phase_list[phase_index]
            if isinstance(phase_index, int) and 0 <= phase_index < len(phase_list)
            else next(iter(PHASE_MOVEMENTS))
        )
        history_age = getattr(agent, "_history", {}) or {}
        phases: Dict[str, Dict[str, Any]] = {}
        for phase, movements in PHASE_MOVEMENTS.items():
            values = [current.get(movement, {"v": 0, "q": 0}) for movement in movements]
            coord: Dict[str, Dict[str, Any]] = {}
            for movement in movements:
                is_boundary = movement_entry.get(movement) not in available_entries
                internal_count = None if is_boundary else _internal_arrival_count(movement)
                coord[movement] = {
                    "count": _boundary_delta(v_history, movement) if is_boundary else int(internal_count or 0),
                    "is_boundary": "yes" if is_boundary else "no",
                }
            phases[phase] = {
                "v": [int(item.get("v", 0) or 0) for item in values],
                "q": [int(item.get("q", 0) or 0) for item in values],
                "dv": sum(_delta(v_history, movement) for movement in movements),
                "dq": sum(_delta(q_history, movement) for movement in movements),
                "age": int(history_age.get(phase, 0) or 0),
                "coord": coord,
            }
        return {
            "current_phase": current_phase,
            "phases": phases,
            "fallback": {
                "source": "sumo",
                "step": int(step_num),
                "coordination_note": (
                    "internal target movement arrivals unavailable from local SUMO state"
                ),
            },
        }

    def _stage1_fallback_result(
        self, intersection: Any, agent: Any, step_num: int, exc: Exception
    ) -> Dict[str, Any]:
        """Return a traceable Stage 1 result after schema/parse failure."""
        return {
            "perception": self._sumo_stage1_perception(intersection, agent, step_num),
            "response": "",
            "prompt": None,
            "videos": [],
            "frames": [],
            "fallback": {
                "source": "sumo",
                "reason": f"{type(exc).__name__}: {exc}",
                "step": int(step_num),
            },
        }

    def _build_stage2_cooperative_perception(
        self,
        perception: Dict[str, Any],
        tls_id: str,
        stage1_perceptions: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build Stage 2 context from the same-step Stage 1 barrier.

        The offline Stage 2 generator routes each receiver direction to the
        first upstream source in ``movement_routes`` and reads that source's
        Stage 1 phase values.  Keeping this lookup on the Stage 1 barrier is
        important: SUMO's live snapshot may represent a different sampling
        point and can make deployment prompts disagree with training data.
        """
        local_coordination: Dict[str, Any] = {}
        phases = perception.get('phases', {}) if isinstance(perception, dict) else {}
        for phase in PHASE_MOVEMENTS:
            phase_data = phases.get(phase, {}) if isinstance(phases, dict) else {}
            coord = phase_data.get('coord', {}) if isinstance(phase_data, dict) else {}
            local_coordination[phase] = copy.deepcopy(coord) if isinstance(coord, dict) else {}

        neighbors: Dict[str, Any] = {}
        route_rows = self._movement_route_table.get('routes', {}) if isinstance(self._movement_route_table, dict) else {}
        grouped: Dict[str, list[tuple[str, str, Dict[str, Any]]]] = defaultdict(list)
        for source_id, source_data in route_rows.items():
            movements = source_data.get('movements', {}) if isinstance(source_data, dict) else {}
            for movement, route in movements.items():
                if not isinstance(route, dict) or route.get('is_boundary'):
                    continue
                if str(route.get('receiver_id', '')).strip() != tls_id:
                    continue
                entry = str(route.get('receiver_entry_direction', '')).upper()
                side = {'E': 'east', 'W': 'west', 'N': 'north', 'S': 'south'}.get(entry)
                if side:
                    grouped[side].append((str(source_id), str(movement), route))

        side_movements = {
            'north': ('NT', 'EL'),
            'south': ('ST', 'WL'),
            'east': ('ET', 'SL'),
            'west': ('WT', 'NL'),
        }
        for side, items in grouped.items():
            # Match the offline generator: choose one deterministic source per
            # receiver direction, then require both routed movements from it.
            source_id = sorted({item[0] for item in items})[0]
            route_by_movement = {
                movement: route
                for item_source, movement, route in items
                if item_source == source_id
            }

            source_perception = stage1_perceptions.get(source_id)
            if not isinstance(source_perception, dict):
                continue
            upstream_movements: Dict[str, Any] = {}
            for movement in side_movements[side]:
                if movement not in route_by_movement:
                    continue
                phase_index = {
                    'ET': ('ETWT', 0), 'WT': ('ETWT', 1),
                    'NT': ('NTST', 0), 'ST': ('NTST', 1),
                    'EL': ('ELWL', 0), 'WL': ('ELWL', 1),
                    'NL': ('NLSL', 0), 'SL': ('NLSL', 1),
                }.get(movement)
                if phase_index is None:
                    continue
                phase_name, index = phase_index
                source_phases = source_perception.get('phases')
                phase_data = source_phases.get(phase_name) if isinstance(source_phases, dict) else None
                values_v = phase_data.get('v') if isinstance(phase_data, dict) else None
                values_q = phase_data.get('q') if isinstance(phase_data, dict) else None
                if (
                    not isinstance(values_v, list)
                    or not isinstance(values_q, list)
                    or len(values_v) != 2
                    or len(values_q) != 2
                    or index >= len(values_v)
                    or index >= len(values_q)
                ):
                    continue
                value_v, value_q = values_v[index], values_q[index]
                if (
                    isinstance(value_v, bool) or not isinstance(value_v, (int, float))
                    or isinstance(value_q, bool) or not isinstance(value_q, (int, float))
                ):
                    continue
                upstream_movements[movement] = {'v': value_v, 'q': value_q}

            # Do not fabricate a partial neighbor or zero-filled movement.
            if len(upstream_movements) != len(side_movements[side]):
                continue
            route = route_by_movement[side_movements[side][0]]
            # ``movement_routes`` describes connectivity, but older route
            # tables do not carry the physical link timing.  Use the loaded
            # scenario topology as the authoritative fallback so V1's 300 m
            # links are not silently serialized as ``null`` in Stage 2.
            travel_time_s = route.get('travel_time_s')
            if not isinstance(travel_time_s, (int, float)) or isinstance(travel_time_s, bool):
                topo_intersections = self._network_topology.get('intersections', {})
                topo_inter = topo_intersections.get(tls_id, {}) if isinstance(topo_intersections, dict) else {}
                topo_neighbors = topo_inter.get('neighbors', {}) if isinstance(topo_inter, dict) else {}
                topo_side = {
                    'north': 'N', 'south': 'S', 'east': 'E', 'west': 'W',
                }.get(side)
                topo_link = topo_neighbors.get(topo_side, {}) if isinstance(topo_neighbors, dict) else {}
                candidate_time = topo_link.get('travel_time_s') if isinstance(topo_link, dict) else None
                if isinstance(candidate_time, (int, float)) and not isinstance(candidate_time, bool):
                    travel_time_s = candidate_time
                elif isinstance(topo_link, dict):
                    distance_m = topo_link.get('distance_m')
                    speed_mps = topo_link.get('speed_limit_mps')
                    if (
                        isinstance(distance_m, (int, float)) and not isinstance(distance_m, bool)
                        and isinstance(speed_mps, (int, float)) and not isinstance(speed_mps, bool)
                        and speed_mps > 0
                    ):
                        travel_time_s = round(distance_m / speed_mps, 2)
            neighbors[side] = {
                'total_v': sum(row['v'] for row in upstream_movements.values()),
                'total_q': sum(row['q'] for row in upstream_movements.values()),
                'travel_time_s': travel_time_s,
            }

        local_coordination = {
            phase: sum(
                movement.get('count', 0)
                for movement in coord.values()
                if isinstance(movement, dict)
                and isinstance(movement.get('count', 0), (int, float))
                and not isinstance(movement.get('count', 0), bool)
            ) if isinstance(coord, dict) else 0
            for phase, coord in local_coordination.items()
        }

        return {'local_coordination': local_coordination, 'neighbors': neighbors}

    def _get_vlm_actions(self, saved_paths: Dict[str, List[str]], step_num: int,
                          state_action_log: List[List[Dict]]) -> Dict[str, int]:
        """Get actions from VLM agents using the deployment two-stage barrier."""
        sumo_only_mode = (
            self.dic_traffic_env_conf.get("SUMO_ONLY_MODE", False)
            or self.dic_traffic_env_conf.get("SUMO_ONLY_AGENT_INPUT", False)
        )
        vlm_config = self.dic_traffic_env_conf.get('VLM_CONFIG', {}) or {}
        if not saved_paths and not sumo_only_mode:
            raise RuntimeError(
                "ERROR: No images were saved for VLM decision. This could be due to:\n"
                "  1. 3D renderer (TSHubRenderer) failed to initialize\n"
                "  2. SCENARIO_GLB_DIR configuration is missing or incorrect\n"
                "  3. TransSimHub is not properly installed\n"
                "  4. Image saver failed to initialize\n"
                "\n"
                "VLM traffic signal control requires 3D rendering images.\n"
                "Please check the initialization logs above for more details."
            )

        action_list: Dict[str, int] = {}
        num_agents = len(self.agents)
        current_movement_snapshots = {
            inter.inter_id: _build_current_movement_snapshot(inter)
            for inter in self.env.list_intersection
        }
        intersection_by_tls = {
            inter.inter_id: inter for inter in self.env.list_intersection
        }

        neighbor_info_map: Dict[str, Dict[str, Any]] = {}
        for i, _agent in enumerate(self.agents):
            tls_id = self.tls_ids[i]
            neighbor_info: Dict[str, Any] = {}
            topo_inter = self._network_topology.get('intersections', {}).get(tls_id, {})
            topo_neighbors = topo_inter.get('neighbors', {}) if isinstance(topo_inter, dict) else {}
            if isinstance(topo_neighbors, dict) and topo_neighbors:
                for direction, n_data in topo_neighbors.items():
                    n_id = n_data.get('neighbor_id')
                    if n_id:
                        neighbor_info[n_id] = {
                            **self.neighbor_states.get(n_id, {}),
                            'direction': direction,
                            'their_entry': n_data.get('their_entry_direction', '?'),
                            'distance_m': n_data.get('distance_m', 0),
                            'speed_mps': n_data.get('speed_limit_mps', 0),
                            'movement_state': current_movement_snapshots.get(n_id, {}),
                        }
            else:
                intersection = self.env.list_intersection[i]
                adj = getattr(intersection, 'adjacency_info', {}) or {}
                adj_row = adj.get('adjacency_row', [])
                id_map = adj.get('inter_id_to_index', {})
                idx_to_id = {v: k for k, v in id_map.items()} if id_map else {}
                for idx in adj_row[1:]:
                    n_id = idx_to_id.get(idx)
                    if n_id and n_id in self.neighbor_states:
                        neighbor_info[n_id] = self.neighbor_states[n_id]

            if sumo_only_mode and self._coordination_manager is not None:
                snapshots_by_tls = {
                    inter.inter_id: list(inter.dic_feature.get("v36_cycle_outbound_snapshots") or [])
                    for inter in self.env.list_intersection
                }
                target_inter = self.env.list_intersection[i]
                neighbor_info["__v32_coordination_info__"] = self._coordination_manager.build(
                    target_inter.inter_id,
                    list(target_inter.control_phases),
                    snapshots_by_tls,
                    step_num,
                )
            elif sumo_only_mode:
                neighbor_info["__v32_coordination_info__"] = {}

            if vlm_config.get('DECISION_INPUT_MODE', 'image') == 'video':
                coordination_video_paths = []
                previous_videos = self.previous_interval_video_paths
                if isinstance(topo_neighbors, dict):
                    for direction, n_data in topo_neighbors.items():
                        try:
                            if float(n_data.get('distance_m', 0)) <= float(
                                    self.dic_traffic_env_conf.get(
                                        "CAMERA_VIEW_DISTANCE", 150.0)):
                                continue
                        except (AttributeError, TypeError, ValueError):
                            continue
                        n_id = n_data.get('neighbor_id')
                        source_direction = n_data.get('my_exit_direction') or direction
                        target_direction = n_data.get('their_entry_direction') or direction
                        source_paths = previous_videos.get(n_id, {}) if n_id else {}
                        source_path = source_paths.get(source_direction)
                        if source_path and os.path.isfile(source_path):
                            coordination_video_paths.append({
                                'path': source_path,
                                'target_entry_direction': target_direction,
                                'source_exit_direction': source_direction,
                                'distance_m': n_data.get('distance_m', 0),
                            })
                neighbor_info['__v35_coordination_video_paths__'] = coordination_video_paths
            neighbor_info_map[tls_id] = neighbor_info

        stage_tasks = []
        for i, agent in enumerate(self.agents):
            tls_id = self.tls_ids[i]
            image_paths = saved_paths.get(tls_id, [])
            agent_input = self._build_agent_input(tls_id, image_paths)
            media_input_ok = bool(image_paths)
            if isinstance(agent_input, dict) and agent_input.get("mode") == "video":
                media_input_ok = True
            if not media_input_ok and not sumo_only_mode:
                raise RuntimeError(
                    f"ERROR: No images saved for intersection {tls_id} at step {step_num}.\n"
                    f"saved_paths keys: {list(saved_paths.keys())}\n"
                    "This could mean:\n"
                    "  1. The renderer failed to process this intersection\n"
                    "  2. The image saver failed to save images\n"
                    "  3. The intersection configuration is incorrect"
                )
            stage_tasks.append((i, agent, tls_id, image_paths, agent_input, neighbor_info_map.get(tls_id, {})))

        parallel_enabled = vlm_config.get('PARALLEL_ENABLED', False)
        parallel_workers = vlm_config.get('PARALLEL_WORKERS', 4)
        use_video_stage_barrier = (
            vlm_config.get('DECISION_INPUT_MODE', 'image') == 'video'
            and all(hasattr(task[1], 'run_stage1') and hasattr(task[1], 'run_stage2') for task in stage_tasks)
        )

        if use_video_stage_barrier:
            stage_workers = max(1, min(int(parallel_workers or 1), num_agents)) if parallel_enabled else 1
            print(f"  [VideoAgent] Stage 1 perception for {num_agents} intersections with {stage_workers} workers...")

            def _stage1(task):
                idx, agent, tls_id, image_paths, agent_input, neighbor_info = task
                if getattr(agent, "stage1_sumo_only", False):
                    intersection = intersection_by_tls.get(tls_id)
                    if intersection is None:
                        raise RuntimeError(
                            f"Forced Stage 1 SUMO perception cannot find intersection {tls_id}"
                        )
                    result = self._stage1_fallback_result(
                        intersection,
                        agent,
                        step_num,
                        RuntimeError("forced by VLM_STAGE1_SUMO_ONLY"),
                    )
                    result["fallback"].update({
                        "source": "sumo_forced",
                        "reason": "VLM_STAGE1_SUMO_ONLY=1",
                    })
                    print(
                        f"[STAGE1_SUMO_FORCED] tls={tls_id} step={step_num}",
                        flush=True,
                    )
                    return idx, tls_id, image_paths, agent_input, neighbor_info, result
                try:
                    result = agent.run_stage1(
                        agent_input, neighbor_info=neighbor_info, step_num=step_num
                    )
                except (ValueError, KeyError, TypeError) as exc:
                    intersection = intersection_by_tls.get(tls_id)
                    if intersection is None:
                        raise
                    result = self._stage1_fallback_result(intersection, agent, step_num, exc)
                    print(
                        f"[STAGE1_SUMO_FALLBACK] tls={tls_id} step={step_num} "
                        f"reason={type(exc).__name__}: {exc}",
                        flush=True,
                    )
                return idx, tls_id, image_paths, agent_input, neighbor_info, result

            stage1_results: Dict[str, Dict[str, Any]] = {}
            if stage_workers > 1:
                with ThreadPoolExecutor(max_workers=stage_workers) as executor:
                    futures = {executor.submit(_stage1, task): task[2] for task in stage_tasks}
                    for future in as_completed(futures):
                        tls_id_key = futures[future]
                        try:
                            idx, tls_id, image_paths, agent_input, neighbor_info, stage1 = future.result()
                            stage1_results[tls_id] = {
                                'idx': idx,
                                'image_paths': image_paths,
                                'agent_input': agent_input,
                                'neighbor_info': neighbor_info,
                                'stage1': stage1,
                            }
                            source = "sumo_fallback" if stage1.get("fallback") else "vlm"
                            print(f"    {tls_id} -> stage1=ok source={source}")
                        except Exception as exc:
                            print(f"    {tls_id_key} -> STAGE1 ERROR: {exc}")
                            raise
            else:
                for task in stage_tasks:
                    idx, tls_id, image_paths, agent_input, neighbor_info, stage1 = _stage1(task)
                    stage1_results[tls_id] = {
                        'idx': idx,
                        'image_paths': image_paths,
                        'agent_input': agent_input,
                        'neighbor_info': neighbor_info,
                        'stage1': stage1,
                    }
                    source = "sumo_fallback" if stage1.get("fallback") else "vlm"
                    print(f"    {tls_id} -> stage1=ok source={source}")

            if set(stage1_results) != set(self.tls_ids):
                missing = sorted(set(self.tls_ids) - set(stage1_results))
                raise RuntimeError(f"Stage 1 barrier failed; missing perceptions for {missing}")

            cooperative_map = {
                tls_id: self._build_stage2_cooperative_perception(
                    stage1_results[tls_id]['stage1']['perception'],
                    tls_id,
                    {
                        source_tls_id: source_result['stage1']['perception']
                        for source_tls_id, source_result in stage1_results.items()
                    },
                )
                for tls_id in self.tls_ids
            }
            print(f"  [VideoAgent] Stage 2 routed decision for {num_agents} intersections with {stage_workers} workers...")

            def _stage2(task):
                idx, agent, tls_id, image_paths, agent_input, neighbor_info = task
                stage1 = stage1_results[tls_id]['stage1']
                stage2 = agent.run_stage2(stage1['perception'], cooperative_map[tls_id], step_num=step_num)
                action = agent.phase_list.index(stage2['signal'])
                return idx, tls_id, action, image_paths, agent_input, stage1, cooperative_map[tls_id], stage2

            results = []
            if stage_workers > 1:
                with ThreadPoolExecutor(max_workers=stage_workers) as executor:
                    futures = {executor.submit(_stage2, task): task[2] for task in stage_tasks}
                    for future in as_completed(futures):
                        tls_id_key = futures[future]
                        try:
                            results.append(future.result())
                        except Exception as exc:
                            print(f"    {tls_id_key} -> STAGE2 ERROR: {exc}")
                            raise
            else:
                for task in stage_tasks:
                    results.append(_stage2(task))

            agent_by_tls = {tls_id: agent for _, agent, tls_id, *_ in stage_tasks}
            for idx, tls_id, action, image_paths, agent_input, stage1, cooperative, stage2 in results:
                print(f"    {tls_id} -> action={action}")
                action_list[tls_id] = action
                state_action_log[idx].append({
                    "step": step_num,
                    "tls_id": tls_id,
                    "action": action,
                    "image_paths": image_paths,
                    "decision_input_mode": vlm_config.get('DECISION_INPUT_MODE', 'image'),
                    "agent_input": agent_input if isinstance(agent_input, dict) else None,
                    "stage1_source": "sumo_fallback" if stage1.get("fallback") else "vlm",
                    "stage1_fallback": copy.deepcopy(stage1.get("fallback")),
                    "stage2_source": "v25_fallback" if stage2.get("fallback") else "vlm",
                    "stage2_fallback": copy.deepcopy(stage2.get("fallback")),
                })
                agent = agent_by_tls[tls_id]
                current_v = {
                    phase: sum(int(value or 0) for value in (stage1['perception']['phases'][phase].get('v') or []))
                    for phase in PHASE_MOVEMENTS
                }
                agent.current_phase_idx = action
                agent.action_history.append((step_num, action))
                agent._update_history(current_v, action)
                agent._last_decision_text = stage1.get('response', '') + "\n" + stage2.get('response', '')
                agent._last_api_error = None
                agent._last_decision_source = 'deployment_two_stage_global_route'
                agent._last_mode_routing = dict(stage2.get('diagnostic') or {})
                agent._last_mode_routing.update({
                    'stage1_prompt': stage1.get('prompt'),
                    'stage1_response': stage1.get('response'),
                    'stage1_perception': stage1.get('perception'),
                    'stage1_fallback': copy.deepcopy(stage1.get('fallback')),
                    'stage2_prompt': stage2.get('prompt'),
                    'stage2_response': stage2.get('response'),
                    'stage2_fallback': copy.deepcopy(stage2.get('fallback')),
                    'stage2_cooperative_perception': cooperative,
                })
                agent._last_decision_record = {
                    "step": step_num,
                    "action": action,
                    "phase": agent.phase_list[action],
                    "source": agent._last_decision_source,
                    "error": None,
                    "mode_routing": copy.deepcopy(agent._last_mode_routing),
                }
                agent.write_two_stage_logs(
                    step_num=step_num,
                    stage1=stage1,
                    stage2=stage2,
                    cooperative_perception=cooperative,
                    videos=stage1.get("videos"),
                    frames=stage1.get("frames"),
                )
        elif parallel_enabled and num_agents > 1:
            max_workers = min(parallel_workers, num_agents)
            print(f"  [Parallel] VLM deciding for {num_agents} intersections with {max_workers} workers...")

            def _vlm_decide(task):
                idx, agent, tls_id, image_paths, agent_input, neighbor_info = task
                action = agent.choose_action(agent_input, self.env, step_num=step_num, neighbor_info=neighbor_info)
                return idx, tls_id, action, image_paths, agent_input

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(_vlm_decide, t): t[2] for t in stage_tasks}
                for future in as_completed(futures):
                    tls_id_key = futures[future]
                    try:
                        idx, tls_id, action, image_paths, agent_input = future.result()
                        print(f"    {tls_id} -> action={action}")
                        action_list[tls_id] = action
                        state_action_log[idx].append({
                            "step": step_num,
                            "tls_id": tls_id,
                            "action": action,
                            "image_paths": image_paths,
                            "decision_input_mode": vlm_config.get('DECISION_INPUT_MODE', 'image'),
                            "agent_input": agent_input if isinstance(agent_input, dict) else None,
                        })
                    except Exception as e:
                        print(f"    {tls_id_key} -> ERROR: {e}")
                        raise
        else:
            for i, agent, tls_id, image_paths, agent_input, neighbor_info in stage_tasks:
                print(f"  VLM deciding for {tls_id} ({i+1}/{num_agents})...", end=" ", flush=True)
                action = agent.choose_action(agent_input, self.env, step_num=step_num, neighbor_info=neighbor_info)
                print(f"action={action}")
                action_list[tls_id] = action
                state_action_log[i].append({
                    "step": step_num,
                    "tls_id": tls_id,
                    "action": action,
                    "image_paths": image_paths,
                    "decision_input_mode": vlm_config.get('DECISION_INPUT_MODE', 'image'),
                    "agent_input": agent_input if isinstance(agent_input, dict) else None,
                })

        # Update neighbor_states for next step's coordination.
        for i, agent in enumerate(self.agents):
            tls_id = self.tls_ids[i]
            action = action_list.get(tls_id, 0)
            phase_name = agent.phase_list[action] if 0 <= action < len(agent.phase_list) else f"Phase_{action}"
            dir_counts = getattr(agent, '_current_step_dir_counts', {})
            self.neighbor_states[tls_id] = {
                'phase': phase_name,
                'action': action,
                'counts': dict(dir_counts),
                'step': step_num,
            }

        return action_list

    def _render_and_save_images(self, step_num: int) -> Dict[str, List[str]]:
        """
        渲染并保存图片到磁盘，支持分批渲染（大地图）。
        参考 TransSimHub/examples/NewYork/b_run_simulation.py 的实现。

        Args:
            step_num: 当前步数

        Returns:
            保存的图片路径字典 {tls_id: [path1, path2, ...]}
        """
        if self.renderer is None:
            return {}

        if getattr(self.renderer, 'is_parallel', False):
            all_saved_paths: Dict[str, List[str]] = {}
            batch_tls_ids = self.renderer.get_all_batch_tls_ids()
            batch_inputs = [
                (current_batch_tls, self._get_render_tshub_obs(current_batch_tls))
                for current_batch_tls in batch_tls_ids
            ]
            sensor_batches = self.renderer.step_all(
                batch_inputs,
                should_count_vehicles=False,
            )
            for current_batch_tls, sensor_data in zip(
                batch_tls_ids, sensor_batches, strict=True
            ):
                if not sensor_data or self.image_saver is None:
                    continue
                saved_paths = self.image_saver.save_step_images(
                    step=step_num,
                    sensor_data=sensor_data,
                    tls_ids=current_batch_tls,
                )
                for tls_id, paths in saved_paths.items():
                    if tls_id not in all_saved_paths:
                        all_saved_paths[tls_id] = []
                    for path in paths:
                        if path not in all_saved_paths[tls_id]:
                            all_saved_paths[tls_id].append(path)
            return all_saved_paths
        
        # Implementation note.
        all_saved_paths = {}
        
        # Implementation note.
        batch_info = self.renderer.get_batch_info() if hasattr(self.renderer, 'get_batch_info') else {'mode': 'all'}
        
        if batch_info.get('mode') == 'batch':
            # Implementation note.
            total_batches = batch_info.get('total_batches', 1)
            
            for batch_idx in range(total_batches):
                # Implementation note.
                current_batch_tls = self.renderer.get_current_batch_tls_ids() if hasattr(self.renderer, 'get_current_batch_tls_ids') else self.tls_ids
                
                # Implementation note.
                tshub_obs = self._get_render_tshub_obs(current_batch_tls)
                sensor_data = self.renderer.step(tshub_obs, should_count_vehicles=False)
                
                # Implementation note.
                if sensor_data and self.image_saver:
                    saved_paths = self.image_saver.save_step_images(
                        step=step_num,
                        sensor_data=sensor_data,
                        tls_ids=current_batch_tls
                    )
                    # Merge paths per intersection instead of overwriting.
                    # In some renderer batch modes, the same intersection may
                    # appear across multiple sub-batches (e.g., partial camera
                    # outputs), and `dict.update` would keep only the last
                    # sub-batch paths, causing the model to receive just one
                    # direction image.
                    for tls_id, paths in saved_paths.items():
                        if tls_id not in all_saved_paths:
                            all_saved_paths[tls_id] = []
                        for p in paths:
                            if p not in all_saved_paths[tls_id]:
                                all_saved_paths[tls_id].append(p)
                
                # Implementation note.
                if batch_idx < total_batches - 1 and hasattr(self.renderer, 'switch_to_next_batch'):
                    self.renderer.switch_to_next_batch()
            
            # Implementation note.
            if hasattr(self.renderer, 'switch_to_batch'):
                self.renderer.switch_to_batch(0)
            
            return all_saved_paths
        else:
            # Implementation note.
            tshub_obs = self._get_render_tshub_obs(self.tls_ids)
            sensor_data = self.renderer.step(tshub_obs)
            # Implementation note.
            if sensor_data and self.image_saver:
                saved_paths = self.image_saver.save_step_images(
                    step=step_num,
                    sensor_data=sensor_data,
                    tls_ids=self.tls_ids
                )
                return saved_paths
            
            return {}

    def _update_training_metrics(self, metrics: Dict[str, Any], reward: List[float], step_num: int = 0) -> None:
        """Update training metrics with current step data."""
        metrics['total_reward'] += sum(reward)

        queue_lengths = []
        for intersection in self.env.list_intersection:
            queue_length = sum(intersection.dic_feature.get('lane_num_waiting_vehicle_in', [0]))
            queue_lengths.append(queue_length)
        avg_queue_len = np.mean(queue_lengths) if queue_lengths else 0.0
        metrics['queue_length_episode'].append(sum(queue_lengths))

        # waiting_vehicle_list uses CoLLMLight-compatible {time, link} entries.
        waiting_times = [
            _extract_waiting_time(time_info)
            for v_id, time_info in self.env.waiting_vehicle_list.items()
        ]
        avg_waiting_time = np.mean(waiting_times) if waiting_times else 0.0
        metrics['waiting_time_episode'].append(avg_waiting_time)
        metrics['global_waiting_times'].extend(waiting_times)

        # AEWT: same as AWT, filtered to emergency vehicles currently in network
        if self.env._emergency_vehicle_ids:  # only when emergency vehicles are present
            em_waiting = [
                _extract_waiting_time(time_info)
                for v_id, time_info in self.env.waiting_vehicle_list.items()
                if v_id in self.env._emergency_vehicle_ids
            ]
            avg_em_waiting = np.mean(em_waiting) if em_waiting else 0.0
            metrics['emergency_waiting_time_episode'].append(avg_em_waiting)
        
        # Implementation note.
        current_time = self.env.get_current_time()
        step_reward = sum(reward)
        try:
            with open(self.metrics_log_path, 'a', encoding='utf-8') as f:
                f.write(f"{step_num},{current_time},{avg_queue_len:.4f},{avg_waiting_time:.4f},{step_reward:.4f}\n")
                f.flush()
        except Exception as e:
            pass  # Implementation note.

    def _calculate_final_results(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """Calculate final training results and metrics."""
        # Calculate travel times (per-intersection enter/leave for ALL vehicles)
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

        # --- AETT: same method as ATT, filtered to emergency vehicles ---
        emergency_vids = self.env._all_emergency_vids
        em_travel_times = {vid: times for vid, times in vehicle_travel_times.items()
                           if vid in emergency_vids}
        aett = (float(np.mean([sum(times) for times in em_travel_times.values()]))
                if em_travel_times else 0.0)
        emergency_count = len(em_travel_times)

        # --- AEWT: same method as AWT, filtered to emergency vehicles ---
        aewt = (float(np.mean(metrics['emergency_waiting_time_episode']))
                if metrics['emergency_waiting_time_episode'] else 0.0)

        # Compile results
        results = {
            "reward": metrics['total_reward'],
            "avg_queue_len": (np.mean(metrics['queue_length_episode'])
                              if metrics['queue_length_episode'] else 0),
            "queuing_vehicle": (np.sum(metrics['queue_length_episode'])
                                if metrics['queue_length_episode'] else 0),
            "avg_waiting_time": avg_waiting_time,
            "avg_travel_time": total_travel_time,
            "aett": aett,
            "aewt": aewt,
            "emergency_count": emergency_count,
        }

        # Log emergency metrics
        if emergency_count > 0:
            print(f"\n{'='*50}")
            print(f"Emergency Vehicle Metrics (same method as ATT/AWT, filtered by type):")
            print(f"  Completed: {emergency_count} vehicles")
            print(f"  ATT  (all vehicles):       {total_travel_time:.2f}s")
            print(f"  AETT (emergency only):     {aett:.2f}s")
            awt = results['avg_waiting_time']
            print(f"  AWT  (all vehicles):       {awt:.2f}s")
            print(f"  AEWT (emergency only):     {aewt:.2f}s")
            print(f"  Emergency vehicle IDs: {sorted(em_travel_times.keys())}")
            print(f"{'='*50}\n")

        return results

    def _calculate_travel_times(self) -> Dict[str, List[float]]:
        """Calculate travel times for all vehicles."""
        vehicle_travel_times = {}
        run_count = self.dic_traffic_env_conf["RUN_COUNTS"]

        for intersection in self.env.list_intersection:
            arrive_leave_times = intersection.dic_vehicle_arrive_leave_time

            for vehicle_id, times in arrive_leave_times.items():
                # Skip shadow vehicles
                if "shadow" in vehicle_id:
                    continue

                enter_time = times["enter_time"]
                leave_time = times["leave_time"]

                if np.isnan(enter_time):
                    continue

                # Use RUN_COUNTS if vehicle hasn't left, matching CoLLMLight utils/oneline.py.
                actual_leave_time = leave_time if not np.isnan(leave_time) else run_count
                travel_time = actual_leave_time - enter_time

                if vehicle_id not in vehicle_travel_times:
                    vehicle_travel_times[vehicle_id] = [travel_time]
                else:
                    vehicle_travel_times[vehicle_id].append(travel_time)

        return vehicle_travel_times

    def _save_training_data(self, state_action_log: List[List[Dict]],
                            global_waiting_times: List[float]) -> None:
        """Save training data to files."""
        work_dir = self.session_work_dir

        # Save state-action log
        state_action_file = os.path.join(work_dir, "state_action.json")
        dump_json(state_action_log, state_action_file, indent=4)

        # Save global waiting times
        waiting_times_file = os.path.join(work_dir, "global_waiting_times.json")
        with open(waiting_times_file, "w") as f:
            json.dump(global_waiting_times, f)
        
        print(f"Training data saved to: {work_dir}")
