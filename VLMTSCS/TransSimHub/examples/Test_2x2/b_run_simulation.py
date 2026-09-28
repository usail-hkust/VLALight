'''
@Description: 运行 Test 2x2 路网的 3D 仿真并保存图片
'''
import os
import sys
from tshub.utils.init_log import set_logger
from tshub.utils.get_abs_path import get_abs_path
from tshub.tshub_env3d.tshub_env3d import Tshub3DEnvironment

# 添加项目路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)
from utils.image_saver import ImageSaver

# 导入传感器配置
from sensor_config import get_sensor_config, get_display_config, get_image_save_config

path_convert = get_abs_path(__file__)
set_logger(path_convert('./'), terminal_log_level='INFO')

if __name__ == '__main__':
    # 使用 sumocfg 文件
    data_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), 
        "../../../data/test/2_2"
    ))
    
    sumo_cfg = os.path.join(data_dir, "grid.sumocfg")
    scenario_glb_dir = path_convert("./3d_assets/")
    
    print(f"SUMO配置文件: {sumo_cfg}")
    print(f"3D资源目录: {scenario_glb_dir}")
    
    # 从配置文件获取配置
    sensor_config, tls_ids = get_sensor_config()
    display_config = get_display_config()
    image_save_config = get_image_save_config()
    
    # 提取显示配置
    show_3d_window = display_config['show_3d_window']
    show_sumo_gui = display_config['show_sumo_gui']
    show_buildings = display_config.get('show_buildings', False)
    
    # 提取图片保存配置
    save_images = image_save_config['save_images']
    save_interval = image_save_config['save_interval']
    scenario = image_save_config['scenario']
    
    print(f"使用传感器配置: 监控 {len(tls_ids)} 个路口")
    print(f"路口ID: {tls_ids}")
    
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
        num_seconds=3600,  # 仿真1小时
        delta_time=5,
        # 渲染参数
        preset="480P",
        resolution=0.5,
        vehicle_model='low',
        render_mode="onscreen" if show_3d_window else "offscreen",
        debuger_spin_camera=True,
        sensor_config=sensor_config,
        show_buildings=show_buildings,
    )

    # 运行仿真
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
                total_saved = sum(len(v) for v in saved_paths.values())
                print(f"Step {i_steps}: 保存了 {total_saved} 张图片")
            
            i_steps += 1
    except KeyboardInterrupt:
        print("\n\n检测到 Ctrl+C，正在停止仿真...")

    tshub_env3d.close()
    
    # 打印保存统计
    if image_saver:
        stats = image_saver.get_stats()
        print(f"\n仿真完成，共运行 {i_steps} 步")
        print(f"图片保存统计:")
        print(f"  保存目录: {stats['session_dir']}")
        print(f"  总步数: {stats['step_count']}")
        print(f"  总图片数: {stats['saved_count']}")
    else:
        print(f"仿真完成，共运行 {i_steps} 步")
