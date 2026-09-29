'''
@Description: MZW 路网传感器配置文件
包含所有信号灯路口的传感器配置
'''

# MZW 路网信号灯路口 ID 列表
MZW_TLS_IDS = [
    "5719196463",  # 路口1
    "6467143011",  # 路口2
    "6467143013",  # 路口3
    "6467143016",  # 路口4
    "6467143017",  # 路口5
]

# 传感器配置
MZW_SENSOR_CONFIG = {
    'tls': {
        "5719196463": {  # 路口1
            'sensor_types': ['junction_front_all'],
            'tls_camera_height': 15
        },
        "6467143011": {  # 路口2
            'sensor_types': ['junction_front_all'],
            'tls_camera_height': 15
        },
        "6467143013": {  # 路口3
            'sensor_types': ['junction_front_all'],
            'tls_camera_height': 15
        },
        "6467143016": {  # 路口4
            'sensor_types': ['junction_front_all'],
            'tls_camera_height': 15
        },
        "6467143017": {  # 路口5
            'sensor_types': ['junction_front_all'],
            'tls_camera_height': 15
        },
    }
}

# 可选：不同传感器配置方案
SENSOR_CONFIGS = {
    # 方案1：所有路口前方视角
    'all_front': MZW_SENSOR_CONFIG,
    
    # 方案2：只监控关键路口
    'key_junctions': {
        'tls': {
            "5719196463": {
                'sensor_types': ['junction_front_all'],
                'tls_camera_height': 15
            },
            "6467143011": {
                'sensor_types': ['junction_front_all'],
                'tls_camera_height': 15
            },
        }
    },
    
    # 方案3：多角度监控（前方+后方）
    'multi_angle': {
        'tls': {
            "5719196463": {
                'sensor_types': ['junction_front_all', 'junction_back_all'],
                'tls_camera_height': 15
            },
            "6467143011": {
                'sensor_types': ['junction_front_all', 'junction_back_all'],
                'tls_camera_height': 15
            },
        }
    }
}

# 显示配置
DISPLAY_CONFIG = {
    'show_image_window': True,   # 是否显示Image Window
    'show_3d_window': True,      # 是否显示3D Window (render_mode)
    'show_sumo_gui': True,       # 是否显示SUMO GUI
    'show_buildings': False,      # 是否显示3D建筑
}

def get_sensor_config(config_name='all_front'):
    """
    获取指定的传感器配置
    
    Args:
        config_name (str): 配置方案名称，可选：
            - 'all_front': 所有路口前方视角（默认）
            - 'key_junctions': 只监控关键路口
            - 'multi_angle': 多角度监控
    
    Returns:
        dict: 传感器配置字典
        list: 路口ID列表
    """
    if config_name not in SENSOR_CONFIGS:
        raise ValueError(f"未知的配置方案: {config_name}. 可选: {list(SENSOR_CONFIGS.keys())}")
    
    sensor_config = SENSOR_CONFIGS[config_name]
    tls_ids = list(sensor_config['tls'].keys())
    
    return sensor_config, tls_ids

def get_display_config():
    """
    获取显示配置
    
    Returns:
        dict: 显示配置字典
    """
    return DISPLAY_CONFIG

if __name__ == '__main__':
    # 测试配置
    print("MZW 路网传感器配置:")
    print(f"路口数量: {len(MZW_TLS_IDS)}")
    print(f"路口ID: {MZW_TLS_IDS}")
    
    # 测试不同配置方案
    for config_name in SENSOR_CONFIGS.keys():
        sensor_config, tls_ids = get_sensor_config(config_name)
        print(f"\n配置方案 '{config_name}':")
        print(f"  监控路口: {len(tls_ids)} 个")
        print(f"  路口ID: {tls_ids}")
