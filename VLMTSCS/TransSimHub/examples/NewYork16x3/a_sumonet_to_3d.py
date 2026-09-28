"""Generate 3D GLB assets for the NewYork 16x3 VideoAgent scene."""

import os

from tshub.utils.get_abs_path import get_abs_path
from tshub.utils.init_log import set_logger
from tshub.tshub_env3d.vis3d_sumonet_convert.sumonet_to_tshub3d import SumoNet3D


path_convert = get_abs_path(__file__)
set_logger(path_convert("./"), terminal_log_level="INFO")


if __name__ == "__main__":
    netxml = os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        "../../../data/NewYork/16x3_v1/roadnet_16_3.net.xml",
    ))
    glb_dir = path_convert("../NewYork16x3_v1/3d_assets/")
    os.makedirs(glb_dir, exist_ok=True)
    print(f"Converting NewYork 16x3 SUMO network: {netxml}")
    SumoNet3D(net_file=netxml).to_glb(glb_dir=glb_dir)
    print(f"3D assets written to: {glb_dir}")
