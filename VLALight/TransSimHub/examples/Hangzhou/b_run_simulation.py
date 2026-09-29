'''
@Description: 运行杭州路网的 3D 仿真
'''
import os
import sys
from tshub.utils.init_log import set_logger
from tshub.utils.get_abs_path import get_abs_path
from tshub.tshub_env3d.tshub_env3d import Tshub3DEnvironment
from tshub.tshub_env3d.show_sensor_images import show_sensor_images

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, project_root)
from utils.image_saver import ImageSaver

from sensor_config import get_sensor_config, get_display_config, get_image_save_config

path_convert = get_abs_path(__file__)
set_logger(path_convert('./'), terminal_log_level='INFO')

if __name__ == '__main__':
    data_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), 
        "../../../data/Hangzhou/4_4"
    ))
    
    sumo_cfg = os.path.join(data_dir, "hangzhou.sumocfg")
    scenario_glb_dir = path_convert("./3d_assets/")
    
    print(f"SUMO配置文件: {sumo_cfg}")
    print(f"3D资源目录: {scenario_glb_dir}")
    
    sensor_config, tls_ids = get_sensor_config()
    display_config = get_display_config()
    image_save_config = get_image_save_config()
    
    show_image_window = display_config['show_image_window']
    show_3d_window = display_config['show_3d_window']
    show_sumo_gui = display_config['show_sumo_gui']
    show_buildings = display_config.get('show_buildings', False)
    
    save_images = image_save_config['save_images']
    save_interval = image_save_config['save_interval']
    scenario = image_save_config['scenario']
    
    print(f"监控 {len(tls_ids)} 个路口")
    
    # 初始化图片保存器
    image_saver = None
    if save_images:
        image_saver = ImageSaver(scenario=scenario)
        print(f"图片保存目录: {image_saver.session_dir}")
    
    tshub_env3d = Tshub3DEnvironment(
        sumo_cfg=sumo_cfg,
        scenario_glb_dir=scenario_glb_dir,
        is_map_builder_initialized=False,
        is_aircraft_builder_initialized=False,
        is_vehicle_builder_initialized=True,
        is_traffic_light_builder_initialized=True,
        tls_ids=tls_ids,
        use_gui=show_sumo_gui,
        num_seconds=3600,
        delta_time=5,
        preset="480P",
        resolution=0.5,
        vehicle_model='low',
        render_mode="onscreen" if show_3d_window else "offscreen",
        debuger_spin_camera=True,
        sensor_config=sensor_config,
        show_buildings=show_buildings,
    )

    obs = tshub_env3d.reset()
    done = False
    i_steps = 0
    
    print("\n开始仿真，按 Ctrl+C 停止...")
    try:
        while not done:
            actions = {'vehicle': dict()}
            obs, reward, info, done, sensor_data = tshub_env3d.step(actions=actions)
            
            # 保存图片
            if save_images and image_saver and sensor_data and i_steps % save_interval == 0:
                saved_paths = image_saver.save_step_images(
                    step=i_steps, 
                    sensor_data=sensor_data,
                    tls_ids=tls_ids
                )
                print(f"Step {i_steps}: 保存了 {sum(len(v) for v in saved_paths.values())} 张图片")
            
            if sensor_data and show_image_window:
                images = [img for so in sensor_data.values() for img in so.values()]
                if images:
                    show_sensor_images(images, scale=0.5, images_per_row=4)
            
            i_steps += 1
    except KeyboardInterrupt:
        print("\n用户中断仿真")

    tshub_env3d.close()
    print(f"仿真完成，共运行 {i_steps} 步")
    if image_saver:
        print(f"图片保存目录: {image_saver.session_dir}")
