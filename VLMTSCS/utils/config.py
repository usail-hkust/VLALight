# Implementation note.
def _get_agent_class(agent_name):
    """寤惰繜瀵煎叆 Agent 绫伙紝閬垮厤涓嶅繀瑕佺殑渚濊禆"""
    if agent_name == "Random":
        from models.random_agent import RandomAgent
        return RandomAgent
    elif agent_name == "Fixedtime":
        from models.fixedtime_agent import FixedtimeAgent
        return FixedtimeAgent
    elif agent_name == "Webster":
        from models.webster_agent import WebsterAgent
        return WebsterAgent
    elif agent_name == "MaxPressure":
        from models.maxpressure_agent import MaxPressureAgent
        return MaxPressureAgent
    elif agent_name == "V1":
        from models.v1_agent import V1Agent
        return V1Agent
    elif agent_name == "V2":
        from models.v2_agent import V2Agent
        return V2Agent
    elif agent_name == "V3":
        from models.v3_agent import V3Agent
        return V3Agent
    elif agent_name == "V4":
        from models.v4_agent import V4Agent
        return V4Agent
    elif agent_name == "V5":
        from models.v5_agent import V5Agent
        return V5Agent
    elif agent_name == "V6":
        from models.v6_agent import V6Agent
        return V6Agent
    elif agent_name == "V7":
        from models.v7_agent import V7Agent
        return V7Agent
    elif agent_name == "V8":
        from models.v8_agent import V8Agent
        return V8Agent
    elif agent_name == "V9":
        from models.v9_agent import V9Agent
        return V9Agent
    elif agent_name == "V10":
        from models.v10_agent import V10Agent
        return V10Agent
    elif agent_name == "V11":
        from models.v11_agent import V11Agent
        return V11Agent
    elif agent_name == "V12":
        from models.v12_agent import V12Agent
        return V12Agent
    elif agent_name == "V13":
        from models.v13_agent import V13Agent
        return V13Agent
    elif agent_name == "V14":
        from models.v14_agent import V14Agent
        return V14Agent
    elif agent_name == "V15":
        from models.v15_agent import V15Agent
        return V15Agent
    elif agent_name == "V16":
        from models.v16_agent import V16Agent
        return V16Agent
    elif agent_name == "V17":
        from models.v17_agent import V17Agent
        return V17Agent
    elif agent_name == "V18":
        from models.v18_agent import V18Agent
        return V18Agent
    elif agent_name == "V20":
        from models.v20_agent import V20Agent
        return V20Agent
    elif agent_name == "V21":
        from models.v21_agent import V21Agent
        return V21Agent
    elif agent_name == "V24":
        from models.v24_agent import V24Agent
        return V24Agent
    elif agent_name == "V25":
        from models.v25_agent import V25Agent
        return V25Agent
    elif agent_name == "V26":
        from models.v26_agent import V26Agent
        return V26Agent
    elif agent_name == "V27":
        from models.v27_agent import V27Agent
        return V27Agent
    elif agent_name == "V28":
        from models.v28_agent import V28Agent
        return V28Agent
    elif agent_name == "V29":
        from models.v29_agent import V29Agent
        return V29Agent
    elif agent_name == "V30":
        from models.v30_agent import V30Agent
        return V30Agent
    elif agent_name == "V31":
        from models.v31_agent import V31Agent
        return V31Agent
    elif agent_name == "V32":
        from models.v32_agent import V32Agent
        return V32Agent
    elif agent_name == "V33":
        from models.v33_agent import V33Agent
        return V33Agent
    elif agent_name == "V36":
        from models.v36_agent import V36Agent
        return V36Agent
    elif agent_name == "EfficientMaxPressure":
        from models.efficient_maxpressure_agent import EfficientMaxPressureAgent
        return EfficientMaxPressureAgent
    elif agent_name == "AdvancedMaxPressure":
        from models.advanced_maxpressure_agent import AdvancedMaxPressureAgent
        return AdvancedMaxPressureAgent
    elif agent_name == "EfficientPressLight":
        from models.presslight_one import PressLightAgentOne
        return PressLightAgentOne
    elif agent_name in ["EfficientColight", "Colight", "AdvancedColight"]:
        from models.colight_agent import CoLightAgent
        return CoLightAgent
    elif agent_name in ["EfficientMPLight", "MPLight"]:
        from models.mplight_agent import MPLightAgent
        return MPLightAgent
    elif agent_name == "AdvancedMPLight":
        from models.advanced_mplight_agent import AdvancedMPLightAgent
        return AdvancedMPLightAgent
    elif agent_name == "AdvancedDQN":
        from models.simple_dqn_one import SimpleDQNAgentOne
        return SimpleDQNAgentOne
    elif agent_name == "Attend":
        from models.attendlight_agent import AttendLightAgent
        return AttendLightAgent
    elif agent_name == "ChatGPTTLCSWaitTimeForecast":
        from models.chatgpt import ChatGPTTLCS_Wait_Time_Forecast
        return ChatGPTTLCS_Wait_Time_Forecast
    elif agent_name == "ChatGPTTLCSCommonsense":
        from models.chatgpt import ChatGPTTLCS_Commonsense
        return ChatGPTTLCS_Commonsense
    else:
        raise ValueError(f"Unknown agent: {agent_name}")

# Implementation note.
class _AgentDict(dict):
    def __getitem__(self, key):
        return _get_agent_class(key)
    
    def get(self, key, default=None):
        try:
            return self[key]
        except (ValueError, KeyError):
            return default

DIC_AGENTS = _AgentDict()

DIC_PATH = {
    "PATH_TO_MODEL": "model/default",
    "PATH_TO_WORK_DIRECTORY": "records/default",
    "PATH_TO_DATA": "data/template",
    "PATH_TO_PRETRAIN_MODEL": "model/default",
    "PATH_TO_ERROR": "errors/default",
}

dic_traffic_env_conf = {

    "LIST_MODEL": ["Random", "Fixedtime", "Webster", "MaxPressure", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V20", "V21", "V24", "V25", "V26", "V27", "V28", "V29", "V30", "V31", "V32", "V33", "V36", "EfficientMaxPressure", "AdvancedMaxPressure",
                   "EfficientPressLight", "EfficientColight", "EfficientMPLight",
                   "AdvancedMPLight", "AdvancedColight", "AdvancedDQN", "Attend"],
    "LIST_MODEL_NEED_TO_UPDATE": ["EfficientPressLight", "EfficientColight", "EfficientMPLight",
                                  "AdvancedMPLight", "AdvancedColight", "AdvancedDQN", "Attend"],

    "NUM_LANE": 12,
    # 'WT_ET', 'NT_ST', 'WL_EL', 'NL_SL'/ 'WL_WT', 'EL_ET', 'SL_ST', 'NL_NT'
    "PHASE_MAP": [[1, 4, 12, 13, 14, 15, 16, 17], [7, 10, 18, 19, 20, 21, 22, 23], [0, 3, 18, 19, 20, 21, 22, 23], [6, 9, 12, 13, 14, 15, 16, 17]],
                  # [0, 1, 15, 16, 17, 18, 19, 20], [3, 4, 12, 13, 14, 21, 22, 23], [9, 10, 18, 19, 20, 12, 13, 14], [6, 7, 21, 22, 23, 15, 16, 17]],
    "FORGET_ROUND": 20,
    "RUN_COUNTS": 3600,
    "MODEL_NAME": None,
    "TOP_K_ADJACENCY": 5,

    "ACTION_PATTERN": "set",
    "NUM_INTERSECTIONS": 1,

    "OBS_LENGTH": 167,
    "MIN_ACTION_TIME": 30,
    "MEASURE_TIME": 30,
    "V33_TEMPORAL_FRAME_INTERVAL": 5,
    "V33_QUEUE_MATCH_DISTANCE_M": 5.0,

    "BINARY_PHASE_EXPANSION": True,

    "YELLOW_TIME": 5,
    "ALL_RED_TIME": 0,
    "NUM_PHASES": 4,
    "NUM_LANES": [3, 3, 3, 3],

    "INTERVAL": 1,

    "LIST_STATE_FEATURE": [
        "cur_phase",
        "time_this_phase",
        "lane_num_vehicle",
        "lane_num_vehicle_downstream",
        "traffic_movement_pressure_num",
        "traffic_movement_pressure_queue",
        "traffic_movement_pressure_queue_efficient",
        "pressure",
        "adjacency_matrix"
    ],
    "DIC_REWARD_INFO": {
        "queue_length": 0,
        "pressure": 0,
    },
    "PHASE": {
        1: [0, 1, 0, 1, 0, 0, 0, 0],
        2: [0, 0, 0, 0, 0, 1, 0, 1],
        3: [1, 0, 1, 0, 0, 0, 0, 0],
        4: [0, 0, 0, 0, 1, 0, 1, 0]
        },
    "list_lane_order": ["WL", "WT", "EL", "ET", "NL", "NT", "SL", "ST"],
    "PHASE_LIST": ['WT_ET', 'NT_ST', 'WL_EL', 'NL_SL'],

}

DIC_BASE_AGENT_CONF = {
    "D_DENSE": 20,
    "LEARNING_RATE": 0.001,
    "PATIENCE": 10,
    "BATCH_SIZE": 20,
    "EPOCHS": 100,
    "SAMPLE_SIZE": 3000,
    "MAX_MEMORY_LEN": 12000,

    "UPDATE_Q_BAR_FREQ": 5,
    "UPDATE_Q_BAR_EVERY_C_ROUND": False,

    "GAMMA": 0.8,
    "NORMAL_FACTOR": 20,

    "EPSILON": 0.8,
    "EPSILON_DECAY": 0.95,
    "MIN_EPSILON": 0.2,
    "LOSS_FUNCTION": "mean_squared_error",
}

DIC_CHATGPT_AGENT_CONF = {
    "GPT_VERSION": "gpt-4",
    "LOG_DIR": "../GPT_logs"
}

DIC_FIXEDTIME_AGENT_CONF = {
    "FIXED_TIME": [30, 30, 30, 30]
}

DIC_MAXPRESSURE_AGENT_CONF = {
    "FIXED_TIME": [30, 30, 30, 30]
}

