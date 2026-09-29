'''
@Description: 济南路网传感器配置文件
自动从路网文件读取信号灯路口ID
'''
import os
import xml.etree.ElementTree as ET

def get_tls_ids_from_net(net_file):
    """从 SUMO 路网文件中自动读取所有信号灯路口 ID"""
    if not os.path.exists(net_file):
        raise FileNotFoundError(f"路网文件不存在: {net_file}")
    
    tree = ET.parse(net_file)
    root = tree.getroot()
    tls_ids = [tl.get('id') for tl in root.findall('.//tlLogic')]
    return list(set(tls_ids))  # 去重

# 自动获取路网文件路径
DATA_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), 
    "../../../data/Jinan/3_4"
))
NET_FILE = os.path.join(DATA_DIR, "jinan_phase.net.xml")

# 自动从路网读取路口 ID
JINAN_TLS_IDS = get_tls_ids_from_net(NET_FILE) if os.path.exists(NET_FILE) else []

# 传感器配置 - 为每个路口配置前方视角摄像头
def create_sensor_config(tls_ids):
    return {
        'tls': {
            tls_id: {
                'sensor_types': ['junction_front_all'],
                'tls_camera_height': 15
            }
            for tls_id in tls_ids
        }
    }

# 所有路口配置
SENSOR_CONFIG = create_sensor_config(JINAN_TLS_IDS)

# 显示配置
DISPLAY_CONFIG = {
    'show_image_window': False,  # 是否显示Image Window
    'show_3d_window': False,     # 是否显示3D Window (render_mode) - 关闭后SUMO GUI可交互
    'show_sumo_gui': True,       # 是否显示SUMO GUI
    'show_buildings': False,     # 是否显示3D建筑
}

# 图片保存配置
IMAGE_SAVE_CONFIG = {
    'save_images': True,         # 是否保存图片
    'save_interval': 10,          # 保存间隔（每隔多少步保存一次）
    'scenario': 'jinan',         # 场景名称
}

def get_sensor_config():
    """获取传感器配置（所有路口）"""
    return SENSOR_CONFIG, JINAN_TLS_IDS

def get_display_config():
    """获取显示配置"""
    return DISPLAY_CONFIG

def get_image_save_config():
    """获取图片保存配置"""
    return IMAGE_SAVE_CONFIG

if __name__ == '__main__':
    print("济南路网传感器配置:")
    print(f"路口数量: {len(JINAN_TLS_IDS)}")
    print(f"路口ID: {JINAN_TLS_IDS}")
