'''
@Description: 纽约路网传感器配置文件
自动从路网文件读取信号灯路口ID
注意：纽约路网有 196 个路口，建议使用 key_junctions 配置以节省资源
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
    "../../../data/NewYork/28_7"
))
NET_FILE = os.path.join(DATA_DIR, "newyork_phase.net.xml")

# 自动从路网读取路口 ID
NEWYORK_TLS_IDS = get_tls_ids_from_net(NET_FILE) if os.path.exists(NET_FILE) else []

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
SENSOR_CONFIG = create_sensor_config(NEWYORK_TLS_IDS)

# 显示配置
DISPLAY_CONFIG = {
    'show_image_window': False,
    'show_3d_window': False,   # WSL环境下关闭3D窗口避免GLSL兼容性问题
    'show_sumo_gui': True,
    'show_buildings': False,
}

# 图片保存配置
IMAGE_SAVE_CONFIG = {
    'save_images': True,          # 是否保存图片
    'save_interval': 1,           # 保存间隔（每隔多少步保存一次）
    'scenario': 'newyork',        # 场景名称
}

# 传感器分批初始化配置（解决大规模路口初始化卡顿问题）
SENSOR_BATCH_CONFIG = {
    'tls_batch_size': 20,         # 每批初始化的路口数量，None表示一次性全部初始化
}

def get_sensor_config():
    """获取传感器配置（所有路口）"""
    return SENSOR_CONFIG, NEWYORK_TLS_IDS

def get_display_config():
    """获取显示配置"""
    return DISPLAY_CONFIG

def get_image_save_config():
    """获取图片保存配置"""
    return IMAGE_SAVE_CONFIG

def get_sensor_batch_config():
    """获取传感器分批初始化配置"""
    return SENSOR_BATCH_CONFIG

if __name__ == '__main__':
    print("纽约路网传感器配置:")
    print(f"路口数量: {len(NEWYORK_TLS_IDS)}")
    print(f"路口ID: {NEWYORK_TLS_IDS}")
