'''
@Description: 运行 MZW 路网仿真 - 只监控关键路口（节省更多资源）
'''
import random
from tshub.utils.init_log import set_logger
from tshub.utils.get_abs_path import get_abs_path
from tshub.tshub_env3d.tshub_env3d import Tshub3DEnvironment
from tshub.tshub_env3d.show_sensor_images import show_sensor_images

# 导入传感器配置
from sensor_config import get_sensor_config

path_convert = get_abs_path(__file__)
set_logger(path_convert('./'), terminal_log_level='INFO')

if __name__ == '__main__':
    sumo_cfg = path_convert("./mzwmap.sumocfg")
    scenario_glb_dir = path_convert("./3d_assets/")
    
    # 使用关键路口配置（只监控2个路口）
    sensor_config, tls_ids = get_sensor_config('key_junctions')
    
    print(f"使用关键路口配置: 监控 {len(tls_ids)} 个路口")
    print(f"路口ID: {tls_ids}")
    
    tshub_env3d = Tshub3DEnvironment(
        sumo_cfg=sumo_cfg,
        scenario_glb_dir=scenario_glb_dir,
        is_map_builder_initialized=False,
        is_aircraft_builder_initialized=False,
        is_vehicle_builder_initialized=True,
        is_traffic_light_builder_initialized=True,
        tls_ids=tls_ids,
        use_gui=True,
        num_seconds=86400,
        delta_time=100,
        # 更高性能设置
        preset="320P",  # 更低分辨率
        resolution=0.3,  # 更低分辨率
        vehicle_model='low',
        render_mode="offscreen",
        debuger_spin_camera=True,
        sensor_config=sensor_config
    )

    # 运行仿真
    obs = tshub_env3d.reset()
    done = False
    i_steps = 0
    while not done:
        actions = {
            'vehicle': dict(),
        }
        obs, reward, info, done, sensor_data = tshub_env3d.step(actions=actions)
        i_steps += 1

        # 显示传感器图像
        if sensor_data:
            images = []
            for sensor_id, sensor_output in sensor_data.items():
                for sensor_type, image in sensor_output.items():
                    images.append(image)
            
            if images:
                show_sensor_images(images, scale=0.5, images_per_row=2)

    tshub_env3d.close()
