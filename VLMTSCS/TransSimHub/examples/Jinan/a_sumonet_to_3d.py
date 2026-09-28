'''
@Description: 将济南 SUMO Net 转换为 3D 资源
'''
import os
from tshub.utils.init_log import set_logger
from tshub.utils.get_abs_path import get_abs_path
from tshub.tshub_env3d.vis3d_sumonet_convert.sumonet_to_tshub3d import SumoNet3D

path_convert = get_abs_path(__file__)
set_logger(path_convert('./'), terminal_log_level='INFO')

if __name__ == '__main__':
    # 指定济南路网文件（使用带相位名称的版本）
    netxml = os.path.join(
        os.path.dirname(__file__), 
        "../../../data/Jinan/3_4/jinan_phase.net.xml"
    )
    netxml = os.path.abspath(netxml)
    
    print(f"正在转换路网文件: {netxml}")
    sumonet_to_3d = SumoNet3D(net_file=netxml)
    
    # 创建 3d_assets 目录（如果不存在）
    glb_dir = path_convert(f"./3d_assets/")
    os.makedirs(glb_dir, exist_ok=True)
    
    # 生成 3D 资源到 3d_assets 目录
    print(f"正在生成 3D 资源到: {glb_dir}")
    sumonet_to_3d.to_glb(glb_dir=glb_dir)
    print("3D 资源生成完成！")
