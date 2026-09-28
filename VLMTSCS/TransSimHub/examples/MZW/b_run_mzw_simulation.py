'''
@Description: 运行 MZW 路网的 3D 仿真（仅路口传感器，节省资源）
'''
import random
from tshub.utils.init_log import set_logger
from tshub.utils.get_abs_path import get_abs_path
from tshub.tshub_env3d.tshub_env3d import Tshub3DEnvironment
from tshub.tshub_env3d.show_sensor_images import show_sensor_images

# 导入传感器配置
from sensor_config import get_sensor_config, get_display_config

path_convert = get_abs_path(__file__)
set_logger(path_convert('./'), terminal_log_level='INFO')

if __name__ == '__main__':
    sumo_cfg = path_convert("./mzwmap.sumocfg")
    scenario_glb_dir = path_convert("./3d_assets/")
    
    # 从配置文件获取传感器配置和显示配置
    sensor_config, tls_ids = get_sensor_config('all_front')
    display_config = get_display_config()
    
    # 提取显示配置
    show_image_window = display_config['show_image_window']
    show_3d_window = display_config['show_3d_window']
    show_sumo_gui = display_config['show_sumo_gui']
    show_buildings = display_config.get('show_buildings', True)  # 默认显示建筑
    
    print(f"使用传感器配置: 监控 {len(tls_ids)} 个路口")
    print(f"路口ID: {tls_ids}")
    print(f"Image Window: {'显示' if show_image_window else '隐藏'}")
    print(f"3D Window: {'显示' if show_3d_window else '隐藏'}")
    print(f"SUMO GUI: {'显示' if show_sumo_gui else '隐藏'}")
    print(f"Buildings: {'显示' if show_buildings else '隐藏'}")
    
    tshub_env3d = Tshub3DEnvironment(
        sumo_cfg=sumo_cfg,
        scenario_glb_dir=scenario_glb_dir,
        is_map_builder_initialized=False,
        is_aircraft_builder_initialized=False,  # 不使用飞行器
        is_vehicle_builder_initialized=True,
        is_traffic_light_builder_initialized=True,  # 启用信号灯
        tls_ids=tls_ids,  # 从配置文件动态获取的路口ID列表
        use_gui=show_sumo_gui,  # 根据配置显示 SUMO GUI
        num_seconds=86400,  # 仿真到路由文件结束（86325秒，设置86400秒确保完整）
        delta_time=100,  # 每步延迟100ms
        # 渲染参数
        preset="480P",  # 使用较低分辨率节省资源
        resolution=0.5,  # 进一步降低分辨率
        vehicle_model='low',  # 使用低精度车辆模型
        render_mode="onscreen" if show_3d_window else "offscreen",  # 根据配置显示3D窗口
        debuger_spin_camera=True,  # 显示旋转相机
        sensor_config=sensor_config,
        show_buildings=show_buildings,  # 是否显示建筑
    )

    # 只运行一次仿真，直到结束
    obs = tshub_env3d.reset()
    done = False
    i_steps = 0
    while not done:
        actions = {
            'vehicle': dict(),
            # 如果有信号灯控制，在这里添加
            # 'tls': {
            #     '12716389007': random.randint(0, 3),
            # },
        }
        obs, reward, info, done, sensor_data = tshub_env3d.step(actions=actions)
        i_steps += 1

        # 根据配置决定是否显示传感器图像
        if sensor_data and show_image_window:
            images = []
            for sensor_id, sensor_output in sensor_data.items():
                for sensor_type, image in sensor_output.items():
                    images.append(image)
            
            if images:
                show_sensor_images(images, scale=0.5, images_per_row=4)
        elif sensor_data and not show_image_window:
            # 传感器数据生成但不显示（可用于数据分析）
            pass

    tshub_env3d.close()
