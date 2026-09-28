'''
@Description: 运行纽约路网的 3D 仿真
注意：纽约路网较大，建议使用 'key_junctions' 配置以节省资源
'''
import os
import sys

# 添加项目根目录和 TransSimHub 到路径（优先使用本地代码）
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
transsimhub_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)
sys.path.insert(0, transsimhub_root)

from tshub.utils.init_log import set_logger
from tshub.utils.get_abs_path import get_abs_path
from tshub.tshub_env3d.tshub_env3d import Tshub3DEnvironment
from tshub.tshub_env3d.show_sensor_images import show_sensor_images
from utils.image_saver import ImageSaver

from sensor_config import get_sensor_config, get_display_config, get_image_save_config, get_sensor_batch_config

path_convert = get_abs_path(__file__)
set_logger(path_convert('./'), terminal_log_level='INFO')

if __name__ == '__main__':
    data_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), 
        "../../../data/NewYork/28_7"
    ))
    
    sumo_cfg = os.path.join(data_dir, "newyork.sumocfg")
    scenario_glb_dir = path_convert("./3d_assets/")
    
    print(f"SUMO配置文件: {sumo_cfg}")
    print(f"3D资源目录: {scenario_glb_dir}")
    
    sensor_config, tls_ids = get_sensor_config()
    display_config = get_display_config()
    image_save_config = get_image_save_config()
    sensor_batch_config = get_sensor_batch_config()
    
    show_image_window = display_config['show_image_window']
    show_3d_window = display_config['show_3d_window']
    show_sumo_gui = display_config['show_sumo_gui']
    show_buildings = display_config.get('show_buildings', False)
    
    # 图片保存配置
    save_images = image_save_config['save_images']
    save_interval = image_save_config['save_interval']
    scenario = image_save_config['scenario']
    
    # 传感器分批初始化配置
    tls_batch_size = sensor_batch_config.get('tls_batch_size', None)
    
    print(f"监控 {len(tls_ids)} 个路口")
    print(f"路口ID: {tls_ids}")
    
    # 初始化图片保存器（不需要 batch_size，因为渲染已经分批了）
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
        tls_batch_size=tls_batch_size,  # 分批初始化传感器
    )

    obs = tshub_env3d.reset()
    done = False
    i_steps = 0
    
    # 打印批次信息
    batch_info = tshub_env3d.get_batch_info()
    print(f"传感器批次模式: {batch_info['mode']}")
    if batch_info['mode'] == 'batch':
        print(f"  每批路口数: {batch_info['batch_size']}")
        print(f"  总批次数: {batch_info['total_batches']}")
        print(f"  当前批次路口: {batch_info['current_batch_tls']}")
    
    while not done:
        actions = {'vehicle': dict()}
        obs, reward, info, done, sensor_data = tshub_env3d.step(actions=actions)
        i_steps += 1
        
        # 保存所有路口的图片（分批进行）
        if image_saver and i_steps % save_interval == 0:
            batch_info = tshub_env3d.get_batch_info()
            total_batches = batch_info.get('total_batches', 1)
            
            # 遍历所有批次，每批：渲染 → 保存 → 切换
            for batch_idx in range(total_batches):
                # 获取当前批次的路口ID
                current_batch_tls = tshub_env3d.get_current_batch_tls_ids()
                
                # 渲染当前批次（需要再次调用 step 来获取传感器数据，但不推进仿真时间）
                # 注意：这里直接使用 tshub_render.step 来只渲染不推进仿真
                sensor_data = tshub_env3d.tshub_render.step(obs, should_count_vehicles=False)
                
                # 保存当前批次的图片
                if sensor_data:
                    image_saver.save_step_images(step=i_steps, sensor_data=sensor_data, tls_ids=current_batch_tls)
                
                # 切换到下一批次传感器
                if batch_idx < total_batches - 1:
                    tshub_env3d.switch_to_next_batch()
            
            # 重置回第一批次，为下一个 step 做准备
            tshub_env3d.switch_to_batch(0)
            
            # 打印进度
            if i_steps % 10 == 0:
                print(f"Step {i_steps}: 已保存 {image_saver.saved_count} 张图片 (所有 {total_batches} 批次)")

        if sensor_data and show_image_window:
            images = [img for so in sensor_data.values() for img in so.values()]
            if images:
                show_sensor_images(images, scale=0.5, images_per_row=3)

    tshub_env3d.close()
    print(f"仿真完成，共运行 {i_steps} 步")
    if image_saver:
        print(f"图片保存统计: {image_saver.get_stats()}")
