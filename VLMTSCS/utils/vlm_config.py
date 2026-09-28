'''
@Description: VLM 交通信号控制配置文件
包含 VLM API、3D 渲染、图片保存等配置
'''
import os
import json


# Implementation note.
# Implementation note.
# 'newyork16x3_v1', 'newyork7x7_v1', 'test_2x2'
SCENARIO = 'test_2x2'

# Implementation note.
EMERGENCY_VEHICLE_CONFIG = {
    'ENABLED': False,  # Implementation note.
}

# Implementation note.
SCENARIO_DATA_MAP = {
    'jinan': {
        'data_dir': 'data/Jinan/3_4',
        'net_file': 'jinan_phase.net.xml',
        'sumocfg_file': 'jinan_emergency.sumocfg',  # Implementation note.
        'sumocfg_file_normal': 'jinan.sumocfg',       # Implementation note.
        'glb_dir': 'TransSimHub/examples/Jinan/3d_assets',
        'direction_mapping': 'output/direction_mapping_jinan.json',
        'phase_mapping': 'data/Jinan/3_4/jinan_phase_mapping.json',
        'topology_image': 'data/Jinan/3_4/network_topology.png',
        'topology_json': 'data/Jinan/3_4/network_topology.json',
        'num_row': 3,
        'num_col': 4,
    },
    'hangzhou': {
        'data_dir': 'data/Hangzhou/4_4',
        'net_file': 'hangzhou_phase.net.xml',
        'sumocfg_file': 'hangzhou.sumocfg',
        'glb_dir': 'TransSimHub/examples/Hangzhou/3d_assets',
        'direction_mapping': 'output/direction_mapping_hangzhou.json',
        'phase_mapping': 'data/Hangzhou/4_4/hangzhou_phase_mapping.json',
        'topology_image': None,
        'topology_json': 'data/Hangzhou/4_4/network_topology.json',
        'num_row': 4,
        'num_col': 4,
    },
    'newyork': {
        'data_dir': 'data/NewYork/28_7',
        'net_file': 'newyork_phase.net.xml',
        'sumocfg_file': 'newyork.sumocfg',
        'glb_dir': 'TransSimHub/examples/NewYork/3d_assets',
        'direction_mapping': 'output/direction_mapping_newyork.json',
        'phase_mapping': 'data/NewYork/28_7/newyork_phase_mapping.json',
        'topology_image': None,
        'topology_json': 'data/NewYork/28_7/network_topology.json',
        'num_row': 28,
        'num_col': 7,
    },
    'newyork16x3': {
        'data_dir': 'data/NewYork/16x3',
        'net_file': 'roadnet_16_3.net.xml',
        'sumocfg_file': 'roadnet_16_3.sumocfg',
        'traffic_file': 'anon_16_3_newyork_real.rou.xml',
        'glb_dir': 'TransSimHub/examples/NewYork16x3/3d_assets',
        'direction_mapping': 'output/direction_mapping_newyork16x3.json',
        'phase_mapping': 'data/NewYork/16x3/newyork_phase_mapping.json',
        'topology_image': None,
        'topology_json': 'data/NewYork/16x3_v1/network_topology.json',
        'num_row': 16,
        'num_col': 3,
        'num_intersections': 48,
    },
    'newyork16x3_v1': {
        'data_dir': 'data/NewYork/16x3_v1',
        'net_file': 'roadnet_16_3.net.xml',
        'sumocfg_file': 'roadnet_16_3.sumocfg',
        'traffic_file': 'anon_16_3_newyork_real.rou.xml',
        'glb_dir': 'TransSimHub/examples/NewYork16x3_v1/3d_assets',
        'direction_mapping': 'output/direction_mapping_newyork16x3.json',
        'phase_mapping': 'data/NewYork/16x3_v1/newyork_phase_mapping.json',
        'topology_image': None,
        'topology_json': 'data/NewYork/16x3_v1/network_topology.json',
        'num_row': 16,
        'num_col': 3,
        'num_intersections': 48,
    },
    'newyork7x7_v1': {
        'data_dir': 'data/NewYork/7x7_v1',
        'net_file': 'roadnet_7_7.net.xml',
        'sumocfg_file': 'roadnet_7_7.sumocfg',
        'traffic_file': 'anon_7_7_newyork_wave_4000.rou.xml',
        'glb_dir': 'TransSimHub/examples/NewYork7x7_v1/3d_assets',
        'direction_mapping': 'output/direction_mapping_newyork7x7_v1.json',
        'phase_mapping': 'data/NewYork/7x7_v1/newyork_phase_mapping.json',
        'topology_image': None,
        'topology_json': 'data/NewYork/7x7_v1/network_topology.json',
        'movement_route_table': 'data/NewYork/7x7_v1/movement_routes_newyork7x7_v1.json',
        'num_row': 7,
        'num_col': 7,
        'num_intersections': 49,
    },
    'test_2x2': {
        'data_dir': 'data/test/2_2',
        'net_file': 'grid.net.xml',
        'sumocfg_file': 'grid.sumocfg',
        'glb_dir': 'TransSimHub/examples/Test_2x2/3d_assets',
        'direction_mapping': 'output/direction_mapping_test_2x2.json',
        'phase_mapping': 'data/test/2_2/test_2x2_phase_mapping.json',
        'topology_image': 'data/test/2_2/network_topology.png',
        'topology_json': 'data/test/2_2/network_topology.json',
        'num_row': 2,
        'num_col': 2,
    },
}


# Implementation note.


VLM_API_CONFIG = {
    # Implementation note.
    'VLM_API_URL': 'http://localhost:8093/v1/chat/completions',
    'VLM_API_KEY': '',  # Implementation note.
    'VLM_MODEL': os.getenv('VLM_MODEL', ''),
    # Enable only for a local vLLM 0.18+ Chat Completions endpoint. The mode
    # router needs prompt_logprobs and prompt_token_ids; unsupported remote
    # OpenAI-compatible APIs safely fall back to the normal one-shot call.
    'VLM_ADAPTIVE_MODE_ROUTING': True,
    # Generate multimodal perception once, then run text-only cooperative
    # decision with FAST/SLOW routing in deployment.
    'VLM_DEPLOYMENT_TWO_STAGE': True,
    # Copy mode_threshold.json from the trained run next to the deployed
    # model, then set this path. An omitted/missing sidecar is logged and
    # uses the explicit default below instead of a hidden threshold.
    'VLM_MODE_THRESHOLD_PATH': '',
    'VLM_MODE_THRESHOLD_DEFAULT': 0.5,
    'VLM_MODE_ROUTING_LOG': '',

}


# Implementation note.
VLM_PARALLEL_CONFIG = {
    'PARALLEL_ENABLED': True,         # Implementation note.
    'PARALLEL_WORKERS': 2,          # Implementation note.
}


# Implementation note.
RENDER_CONFIG = {
    'RENDER_PRESET': '1080P',      # Implementation note.
    'RENDER_RESOLUTION': 1.0,     # Implementation note.
    'RENDERING_BACKEND': 'pandagl', # Implementation note.
    'TLS_SENSOR_TYPE': 'junction_front_all',
    'TLS_CAMERA_HEIGHT': 30,      # Implementation note.
    'LOCAL_RENDER_RADIUS_M': 200.0, # Implementation note.
    'VEHICLE_MODEL': 'low',        # Implementation note.
    'SHOW_ARROWS': False,        # Implementation note.
    'RENDER_KEEP_BATCH_SENSORS': False, # Implementation note.
    'RENDER_REUSE_BATCH_SENSORS': True, # Implementation note.
    'RENDER_STEP_TASK_MANAGER': False, # Implementation note.
}

# Implementation note.
IMAGE_PREPROCESS_CONFIG = {
    'ENABLED': True,            # Implementation note.
    'LEFT_CROP': 0.40,          # Implementation note.
    'RIGHT_CROP': 0.30,         # Implementation note.
    'SCALE_MODE': 'fit_width',  # Implementation note.
    'ADD_DIRECTION_LABEL': True, # Implementation note.
}


# Implementation note.
SENSOR_BATCH_CONFIG = {
    'jinan': None,      # Implementation note.
    'hangzhou': None,   # Implementation note.
    'newyork': 20,      # Implementation note.
    'newyork16x3': 16,  # Implementation note.
    'newyork16x3_v1': 16,
    'newyork7x7_v1': 16,
}


# Implementation note.
DISPLAY_CONFIG = {
    'SHOW_3D_WINDOW': False,      # Implementation note.
    'SHOW_SUMO_GUI': False,       # Implementation note.
    'SHOW_BUILDINGS': False,      # Implementation note.
}



DECISION_INTERVAL_SEC = 30  # Implementation note.

DECISION_INPUT_CONFIG = {
    'DECISION_INPUT_MODE': 'video',  # 'image' or 'video' — controls whether DecisionWindowRecorder is initialized
    'VIDEO_WINDOW_SEC': DECISION_INTERVAL_SEC, # Implementation note.
    'VIDEO_FPS': 1, # Implementation note.
    'VIDEO_EXPORT_DIRECTIONS': True, # V35 VideoAgent requires E/W/N/S direction videos
    'VIDEO_EXPORT_COMPOSITE': True, # Implementation note.
    'VIDEO_EXPORT_DIRECTION_SEQUENCE': False, # optional N/E/W/S sequence clip
    'VIDEO_ADD_LABELS': True, # Implementation note.
    'VIDEO_PREPROCESS_ENABLED': True, # Implementation note.
    'VIDEO_RECORD_MODE': 'sampled', # Implementation note.
    # Match the V30/V35 SFT convention: one completed post-tick frame at
    # 5, 10, 15, 20, 25 and 30 seconds in each 30-second decision window.
    'VIDEO_FRAME_SAMPLE_INTERVAL': 5.0,
    'VIDEO_FRAME_VIEW': 'legacy_crop', # raw / legacy_crop
    # The rendered approach view keeps its 2:3 aspect ratio, yielding 512x960.
    # This matches the 512x960 SFT model and avoids server-side re-scaling.
    'VIDEO_TILE_WIDTH': 512,
    'VIDEO_INCLUDE_CURRENT_IMAGES': False, # video mode: save current images but do not feed them to VLM
    'VIDEO_PREPROCESS_INTERPOLATION': 'linear', # Implementation note.
    'VIDEO_ASYNC_WRITE': True, # Implementation note.
    'VIDEO_ASYNC_WORKERS': 4, # Implementation note.
    'VIDEO_SAVE_COORDINATION_FRAMES': True,
}


REFERENCE_IMAGE_CONFIG = {
    'ENABLED': False,
    'PATH': 'output/reference.jpg',
}

TOPOLOGY_IMAGE_CONFIG = {
    'ENABLED': True,
    'PATH': None,  # Deprecated: now auto-resolved from SCENARIO_DATA_MAP[scenario]['topology_image']
}


MEMORY_CONFIG = {
    'ENABLED': False,
}


TWO_STAGE_CONFIG = {
    'PERCEPTION_API_URL': '',
    'PERCEPTION_API_KEY': '',
    'PERCEPTION_MODEL': '',
    'PERCEPTION_TEMPERATURE': 0.6,
    'PERCEPTION_MAX_TOKENS': 20480,
    'PERCEPTION_MAX_ROUNDS': 4,
    'DECISION_API_URL': 'http://localhost:8093/v1/chat/completions',
    'DECISION_API_KEY': '',
    'DECISION_MODEL': os.getenv('DECISION_MODEL', ''),
    'DECISION_TEMPERATURE': 0.6,
    'DECISION_MAX_TOKENS': 20480,
    'USE_REFERENCE_IMAGE_IN_PERCEPTION': True,
}
    
def get_project_root():
    """获取项目根目录"""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def get_vlm_config(scenario: str = None) -> dict:
    """
    获取完整的 VLM 配置字典，用于 VLMOneLine。
    
    Args:
        scenario: 场景名称，默认使用 SCENARIO
        
    Returns:
        VLM_CONFIG 字典
    """
    if scenario is None:
        scenario = SCENARIO
    
    project_root = get_project_root()
    scenario_data = SCENARIO_DATA_MAP.get(scenario, SCENARIO_DATA_MAP['jinan'])
    
    # Implementation note.
    if EMERGENCY_VEHICLE_CONFIG['ENABLED']:
        prompt_file = 'prompts/prompt_vlm_tsc.json'          # Implementation note.
    else:
        prompt_file = 'prompts/prompt_vlm_tsc_e2e.json'      # Implementation note.

    return {
        # Implementation note.
        'SCENARIO': scenario,
        'SCENARIO_GLB_DIR': os.path.join(project_root, scenario_data['glb_dir']),
        'DIRECTION_MAPPING_PATH': os.path.join(project_root, scenario_data['direction_mapping']),
        'MOVEMENT_ROUTE_TABLE_PATH': os.path.join(
            project_root,
            scenario_data.get(
                'movement_route_table',
                os.path.join('scenario_assets', 'movement_routes', f"movement_routes_{scenario}.json"),
            ),
        ),
        
        # VLM API
        'VLM_API_URL': VLM_API_CONFIG['VLM_API_URL'],
        'VLM_API_KEY': VLM_API_CONFIG['VLM_API_KEY'],
        'VLM_MODEL': VLM_API_CONFIG['VLM_MODEL'],
        'VLM_ADAPTIVE_MODE_ROUTING': VLM_API_CONFIG.get('VLM_ADAPTIVE_MODE_ROUTING', False),
        'VLM_MODE_THRESHOLD_PATH': VLM_API_CONFIG.get('VLM_MODE_THRESHOLD_PATH', ''),
        'VLM_MODE_THRESHOLD_DEFAULT': VLM_API_CONFIG.get('VLM_MODE_THRESHOLD_DEFAULT', 0.5),
        'VLM_MODE_ROUTING_LOG': VLM_API_CONFIG.get('VLM_MODE_ROUTING_LOG', ''),
        
        # Implementation note.
        'PROMPT_FILE': os.path.join(project_root, prompt_file),
        'EMERGENCY_ENABLED': EMERGENCY_VEHICLE_CONFIG['ENABLED'],
        
        # Implementation note.
        'RENDER_PRESET': RENDER_CONFIG['RENDER_PRESET'],
        'RENDER_RESOLUTION': RENDER_CONFIG['RENDER_RESOLUTION'],
        'RENDERING_BACKEND': RENDER_CONFIG.get('RENDERING_BACKEND', 'pandagl'),
        'TLS_CAMERA_HEIGHT': RENDER_CONFIG['TLS_CAMERA_HEIGHT'],
        'TLS_SENSOR_TYPE': RENDER_CONFIG.get('TLS_SENSOR_TYPE', 'junction_front_all'),
        'LOCAL_RENDER_RADIUS_M': RENDER_CONFIG.get('LOCAL_RENDER_RADIUS_M', 200.0),
        'TLS_BATCH_SIZE': SENSOR_BATCH_CONFIG.get(scenario, None),
        'VEHICLE_MODEL': RENDER_CONFIG['VEHICLE_MODEL'],
        'SHOW_ARROWS': RENDER_CONFIG.get('SHOW_ARROWS', False),
        'RENDER_KEEP_BATCH_SENSORS': RENDER_CONFIG.get('RENDER_KEEP_BATCH_SENSORS', True),
        'RENDER_REUSE_BATCH_SENSORS': RENDER_CONFIG.get('RENDER_REUSE_BATCH_SENSORS', False),
        'RENDER_STEP_TASK_MANAGER': RENDER_CONFIG.get('RENDER_STEP_TASK_MANAGER', True),

        # Implementation note.
        'DECISION_INPUT_MODE': DECISION_INPUT_CONFIG['DECISION_INPUT_MODE'],
        'VIDEO_WINDOW_SEC': DECISION_INPUT_CONFIG['VIDEO_WINDOW_SEC'],
        'VIDEO_FPS': DECISION_INPUT_CONFIG['VIDEO_FPS'],
        'VIDEO_EXPORT_DIRECTIONS': DECISION_INPUT_CONFIG['VIDEO_EXPORT_DIRECTIONS'],
        'VIDEO_EXPORT_COMPOSITE': DECISION_INPUT_CONFIG['VIDEO_EXPORT_COMPOSITE'],
        'VIDEO_EXPORT_DIRECTION_SEQUENCE': DECISION_INPUT_CONFIG.get('VIDEO_EXPORT_DIRECTION_SEQUENCE', False),
        'VIDEO_ADD_LABELS': DECISION_INPUT_CONFIG['VIDEO_ADD_LABELS'],
        'VIDEO_PREPROCESS_ENABLED': DECISION_INPUT_CONFIG.get('VIDEO_PREPROCESS_ENABLED', True),
        'VIDEO_RECORD_MODE': DECISION_INPUT_CONFIG.get('VIDEO_RECORD_MODE', 'sampled'),
        'VIDEO_FRAME_SAMPLE_INTERVAL': DECISION_INPUT_CONFIG['VIDEO_FRAME_SAMPLE_INTERVAL'],
        'VIDEO_FRAME_VIEW': DECISION_INPUT_CONFIG.get('VIDEO_FRAME_VIEW', 'legacy_crop'),
        # Keep the online/offline video contract identical to run_video.py.
        # A direction tile is always 512 px wide; after legacy_crop + fit_width
        # this yields the required 512x960 input frame.
        'VIDEO_TILE_WIDTH': DECISION_INPUT_CONFIG.get('VIDEO_TILE_WIDTH', 512),
        'VIDEO_INCLUDE_CURRENT_IMAGES': DECISION_INPUT_CONFIG['VIDEO_INCLUDE_CURRENT_IMAGES'],
        'VIDEO_PREPROCESS_INTERPOLATION': DECISION_INPUT_CONFIG.get('VIDEO_PREPROCESS_INTERPOLATION', 'lanczos4'),
        'VIDEO_ASYNC_WRITE': DECISION_INPUT_CONFIG.get('VIDEO_ASYNC_WRITE', True),
        'VIDEO_ASYNC_WORKERS': DECISION_INPUT_CONFIG.get('VIDEO_ASYNC_WORKERS', 4),
        'REFERENCE_IMAGE_ENABLED': REFERENCE_IMAGE_CONFIG['ENABLED'],
        'REFERENCE_IMAGE_PATH': os.path.join(project_root, REFERENCE_IMAGE_CONFIG['PATH']),
        'TOPOLOGY_IMAGE_ENABLED': TOPOLOGY_IMAGE_CONFIG['ENABLED'] and (scenario_data.get('topology_image') is not None),
        'TOPOLOGY_IMAGE_PATH': os.path.join(project_root, scenario_data['topology_image']) if scenario_data.get('topology_image') else None,
        'TOPOLOGY_JSON_PATH': os.path.join(project_root, scenario_data['topology_json']) if scenario_data.get('topology_json') else None,
        'MEMORY_ENABLED': MEMORY_CONFIG['ENABLED'],
        'PERCEPTION_API_URL': TWO_STAGE_CONFIG['PERCEPTION_API_URL'] or VLM_API_CONFIG['VLM_API_URL'],
        'PERCEPTION_API_KEY': TWO_STAGE_CONFIG['PERCEPTION_API_KEY'] or VLM_API_CONFIG['VLM_API_KEY'],
        'PERCEPTION_MODEL': TWO_STAGE_CONFIG['PERCEPTION_MODEL'] or VLM_API_CONFIG['VLM_MODEL'],
        'PERCEPTION_TEMPERATURE': TWO_STAGE_CONFIG['PERCEPTION_TEMPERATURE'],
        'PERCEPTION_MAX_TOKENS': TWO_STAGE_CONFIG['PERCEPTION_MAX_TOKENS'],
        'TWO_STAGE_PERCEPTION_MAX_ROUNDS': TWO_STAGE_CONFIG['PERCEPTION_MAX_ROUNDS'],
        'DECISION_API_URL': TWO_STAGE_CONFIG['DECISION_API_URL'] or VLM_API_CONFIG['VLM_API_URL'],
        'DECISION_API_KEY': TWO_STAGE_CONFIG['DECISION_API_KEY'] or VLM_API_CONFIG['VLM_API_KEY'],
        'DECISION_MODEL': TWO_STAGE_CONFIG['DECISION_MODEL'] or VLM_API_CONFIG['VLM_MODEL'],
        'DECISION_TEMPERATURE': TWO_STAGE_CONFIG['DECISION_TEMPERATURE'],
        'DECISION_MAX_TOKENS': TWO_STAGE_CONFIG['DECISION_MAX_TOKENS'],
        'USE_REFERENCE_IMAGE_IN_PERCEPTION': TWO_STAGE_CONFIG['USE_REFERENCE_IMAGE_IN_PERCEPTION'],

        # Implementation note.
        'SUMO_WINDOW_SIZE': DECISION_INTERVAL_SEC,  # Implementation note.
        'DECISION_INTERVAL_SEC': DECISION_INTERVAL_SEC,

        # Implementation note.
        'SHOW_3D_WINDOW': DISPLAY_CONFIG['SHOW_3D_WINDOW'],
        'SHOW_SUMO_GUI': DISPLAY_CONFIG['SHOW_SUMO_GUI'],
        'SHOW_BUILDINGS': DISPLAY_CONFIG.get('SHOW_BUILDINGS', False),
        
        # Implementation note.
        'PARALLEL_ENABLED': VLM_PARALLEL_CONFIG['PARALLEL_ENABLED'],
        'PARALLEL_WORKERS': VLM_PARALLEL_CONFIG['PARALLEL_WORKERS'],

        # Implementation note.
        'IMAGE_PREPROCESS_ENABLED': IMAGE_PREPROCESS_CONFIG['ENABLED'],
        'IMAGE_PREPROCESS_LEFT_CROP': IMAGE_PREPROCESS_CONFIG['LEFT_CROP'],
        'IMAGE_PREPROCESS_RIGHT_CROP': IMAGE_PREPROCESS_CONFIG['RIGHT_CROP'],
        'IMAGE_PREPROCESS_SCALE_MODE': IMAGE_PREPROCESS_CONFIG['SCALE_MODE'],
        'IMAGE_PREPROCESS_ADD_DIRECTION_LABEL': IMAGE_PREPROCESS_CONFIG['ADD_DIRECTION_LABEL'],
    }


def _load_phase_mapping(scenario: str, eightphase: bool = True) -> dict:
    """
    加载 phase_mapping.json 文件。
    
    Args:
        scenario: 场景名称
        eightphase: 是否使用 8 相位，False 则只取后 4 个相位（ELET, WLWT, NLNT, SLST）
    """
    project_root = get_project_root()
    scenario_data = SCENARIO_DATA_MAP.get(scenario, SCENARIO_DATA_MAP['jinan'])
    phase_mapping_path = os.path.join(project_root, scenario_data.get('phase_mapping', ''))
    
    if os.path.exists(phase_mapping_path):
        with open(phase_mapping_path, 'r', encoding='utf-8') as f:
            full_mapping = json.load(f)
        
        # Implementation note.
        if not eightphase:
            filtered_mapping = {}
            for inter_id, phases in full_mapping.items():
                filtered_mapping[inter_id] = phases[:4]
            return filtered_mapping
        
        return full_mapping
    return {}


def get_traffic_env_conf(scenario: str = None, eightphase: bool = False) -> dict:
    """
    获取交通环境配置，包含 VLM_CONFIG。
    
    Args:
        scenario: 场景名称
        eightphase: 是否使用 8 相位
        
    Returns:
        dic_traffic_env_conf 字典
    """
    if scenario is None:
        scenario = SCENARIO
    
    project_root = get_project_root()
    scenario_data = SCENARIO_DATA_MAP.get(scenario, SCENARIO_DATA_MAP['jinan'])
    
    # Implementation note.
    if EMERGENCY_VEHICLE_CONFIG['ENABLED']:
        sumocfg_file = scenario_data.get('sumocfg_file')
    else:
        sumocfg_file = scenario_data.get('sumocfg_file_normal', scenario_data.get('sumocfg_file'))
    
    # Implementation note.
    inter_phase_mapping = _load_phase_mapping(scenario, eightphase)
    
    return {
        'ROADNET_FILE': scenario_data['net_file'],
        'TRAFFIC_FILE': scenario_data.get(
            'traffic_file', sumocfg_file.replace('.sumocfg', '.rou.xml')
        ),
        'SUMOCFG_FILE': sumocfg_file,
        'MIN_ACTION_TIME': DECISION_INTERVAL_SEC,  # Implementation note.
        'YELLOW_TIME': 5,
        'ALL_RED_TIME': 0,
        'SKIP_TRANSITION_PHASE': False,
        'SIM_START_TIME':0,  # Implementation note.
        'RUN_COUNTS': 3600,  # Implementation note.
        'INTERVAL': 1.0,
        'NUM_INTERSECTIONS': scenario_data.get(
            'num_intersections',
            12 if scenario == 'jinan' else (16 if scenario == 'hangzhou' else (4 if scenario == 'test_2x2' else 196)),
        ),
        'NUM_ROW': scenario_data.get('num_row', 3),
        'NUM_COL': scenario_data.get('num_col', 4),
        'NUM_AGENTS': scenario_data.get(
            'num_intersections',
            12 if scenario == 'jinan' else (16 if scenario == 'hangzhou' else (4 if scenario == 'test_2x2' else 196)),
        ),
        'TOP_K_ADJACENCY': 5,
        'INTER_PHASE_MAPPING': inter_phase_mapping,
        'PHASE': [0, 1, 2, 3, 4, 5, 6, 7],
        'MODEL_NAME': 'VLM',
        'PROJECT_NAME': 'chatgpt-TSCS',
        'LIST_STATE_FEATURE': [
            "cur_phase",
            "time_this_phase",
            "traffic_movement_pressure_queue",
        ],
        'DIC_REWARD_INFO': {
            "pressure": 0
        },
        
        # Implementation note.
        'VLM_CONFIG': get_vlm_config(scenario),
    }


def get_path_conf(scenario: str = None, work_dir: str = None) -> dict:
    """
    获取路径配置。
    
    Args:
        scenario: 场景名称
        work_dir: 工作目录，默认为 VLM/output/{scenario}
        
    Returns:
        dic_path 字典
    """
    if scenario is None:
        scenario = SCENARIO
    
    project_root = get_project_root()
    scenario_data = SCENARIO_DATA_MAP.get(scenario, SCENARIO_DATA_MAP['jinan'])
    
    if work_dir is None:
        work_dir = os.path.join(project_root, 'records', 'VLM_online', scenario)
    
    os.makedirs(work_dir, exist_ok=True)
    
    return {
        'PATH_TO_DATA': os.path.join(project_root, scenario_data['data_dir']),
        'PATH_TO_WORK_DIRECTORY': work_dir,
    }


if __name__ == '__main__':
    print("=" * 60)
    print("VLM 交通信号控制配置")
    print("=" * 60)
    
    for scenario in ['jinan', 'hangzhou', 'newyork']:
        print(f"\n场景: {scenario}")
        vlm_config = get_vlm_config(scenario)
        print(f"  3D 资源目录: {vlm_config['SCENARIO_GLB_DIR']}")
        print(f"  方向映射文件: {vlm_config['DIRECTION_MAPPING_PATH']}")
        print(f"  分批大小: {vlm_config['TLS_BATCH_SIZE']}")
        print(f"  VLM 模型: {vlm_config['VLM_MODEL']}")
