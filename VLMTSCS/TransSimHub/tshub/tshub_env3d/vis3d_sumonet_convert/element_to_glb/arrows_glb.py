'''
@Author: Copilot
@Date: 2026-01-20
@Description: 生成车道内交通标线导向箭头的 glb 文件
'''
import math
import numpy as np
import warnings
from typing import List, Tuple

from ..sumonet_convert_utils.glb_data import GLBData

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message="Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.",
        category=DeprecationWarning,
    )
    import trimesh
    from trimesh.exchange import gltf
    from trimesh.creation import box
    from trimesh.visual.material import PBRMaterial


ARROW_SCALE = 2.2
# 箭头条带粗细（米）。调小更细、调大更粗；直行/左转/右转一致。建议 0.15～0.3
ARROW_THICKNESS = 0.18


def _pos_and_heading_from_lane(
    shape: List[Tuple[float, float]],
    turn_direction: str,
    lane_width: float,
    lane_index: int = 0,
    num_lanes: int = 1,
) -> Tuple[Tuple[float, float, float], float]:
    """
    与 add_arrows_from_netxml 完全一致的位置与朝向逻辑。
    shape: 车道中心线点串 [(x,y), ...]，从起点到终点（终点靠近路口）
    turn_direction: 'left'|'straight'|'right'|'uturn'|'unknown'
    返回: (pos_in_lane, arrow_heading_deg)
    """
    pts = shape
    if len(pts) < 2:
        return ((0, 0, 1.0), 0.0)

    # heading: 从 shape 末段方向，与 get_heading_from_shape(use_end=True) 一致
    (x0, y0), (x1, y1) = pts[-2], pts[-1]
    dx, dy = x1 - x0, y1 - y0
    heading = math.degrees(math.atan2(dx, dy))

    # 箭头朝向 = 车道行驶方向（箭头几何体本身已包含左/右转弯折形状）
    # 所有箭头类型统一使用 heading
    arrow_heading = heading

    # EW 方向（heading 接近 ±90°）的箭头需要翻转 180°，
    # 使四个方向的箭头在摄像头视角下都一致地指向路口方向
    norm_h = ((heading + 180) % 360) - 180
    is_ew = 45 <= abs(norm_h) <= 135
    if is_ew:
        arrow_heading += 180

    # pos: 车道末端，Z=0.001（与路面 Z=0 对齐，略高避免 Z-fighting），再往车道内退（2.9m，略往后挪）
    pos = (float(pts[-1][0]), float(pts[-1][1]), 0.001)
    heading_rad = math.radians(heading)
    fx, fy = math.sin(heading_rad), math.cos(heading_rad)
    pos_in_lane = (pos[0] - fx * 2.9, pos[1] - fy * 2.9, pos[2])

    # 所有箭头根据车道位置向中间车道方向平移
    # lane_index: 0=最右车道, num_lanes-1=最左车道
    if num_lanes > 1:
        perp_rad = heading_rad + math.pi / 2
        perp_fx, perp_fy = math.sin(perp_rad), math.cos(perp_rad)
        # 计算偏移：最右车道向左，最左车道向右
        # 中间车道偏移量为 0
        center_lane = (num_lanes - 1) / 2.0
        offset_from_center = lane_index - center_lane
        # 偏移量：每偏离中心车道 1 个车道，向中心平移 0.4m
        offset_dist = offset_from_center * 0.4
        pos_in_lane = (
            pos_in_lane[0] + perp_fx * offset_dist,
            pos_in_lane[1] + perp_fy * offset_dist,
            pos_in_lane[2],
        )

    return (pos_in_lane, arrow_heading)


def _arrow_segments_for_turn(scale: float, turn_direction: str):
    """
    按转向返回线段列表，局部 z=0。顶端箭头用两条线段组成 V 形。
    中间：线段 + 两条线段组成向下箭头（尖端 (0,s)，底在 y=s-d）；
    左：直线 + 折线 + 两条线段组成向左箭头（底边中点在折线尾，尖端在更左）；
    右：镜像。
    """
    s = scale
    z = 0.0
    w, d = 0.28 * s, 0.35 * s   # 中间箭头：底半宽、尖到底
    h = 0.2 * s                 # 左/右：底半高（底边 y 方向 ±h）；尖端相对底边中点再伸出 0.3s

    if turn_direction in ("straight", "uturn", "unknown"):
        # 线段 + 两条线段：尖端 (0,s) 到底边两点，形成向下箭头
        return [
            [(0, 0, z), (0, s, z)],
            [(0, s, z), (-w, s - d, z)],
            [(0, s, z), (w, s - d, z)],
        ]

    if turn_direction == "left":
        # 折线缩短、偏折角 |Δx|/Δy=0.52/0.15 不变：|base_x|=0.44，Δy=0.44/r，cy≈0.677。往两侧：offset 0.52
        r = 0.52 / 0.15
        base_x = -0.44 * s
        dy = (0.44 / r) * s
        cy = s * 0.55 + dy
        tip_x = base_x - 0.3 * s
        return [
            [(0, 0, z), (0, s * 0.55, z)],
            [(0, s * 0.55, z), (base_x, cy, z)],
            [(tip_x, cy, z), (base_x, cy - h, z)],
            [(tip_x, cy, z), (base_x, cy + h, z)],
            [(base_x, cy - h, z), (base_x, cy + h, z)],
        ]

    if turn_direction == "right":
        r = 0.52 / 0.15
        base_x = 0.44 * s
        dy = (0.44 / r) * s
        cy = s * 0.55 + dy
        tip_x = base_x + 0.3 * s
        return [
            [(0, 0, z), (0, s * 0.55, z)],
            [(0, s * 0.55, z), (base_x, cy, z)],
            [(tip_x, cy, z), (base_x, cy - h, z)],
            [(tip_x, cy, z), (base_x, cy + h, z)],
            [(base_x, cy - h, z), (base_x, cy + h, z)],
        ]

    return [
        [(0, 0, z), (0, s, z)],
        [(0, s, z), (-w, s - d, z)],
        [(0, s, z), (w, s - d, z)],
    ]


def make_arrows_glb(
    lane_arrow_data: List[Tuple[List[Tuple[float, float]], str, float, int, int]],
) -> GLBData:
    """
    生成与 add_arrows_from_netxml 形状、位置、朝向一致的箭头 glb。
    中间：线段+两条线段组成向下箭头；左/右：直线+折线+两条线段组成向左/右箭头（底边中点在折线尾端）。
    粗细由 ARROW_THICKNESS 统一控制。

    lane_arrow_data: List[Tuple[shape, turn_direction, lane_width, lane_index, num_lanes]]
        - shape: 车道中心线 [(x,y), ...]
        - turn_direction: 'left'|'straight'|'right'|'uturn'|'unknown'
        - lane_width: 车道宽度（米）
        - lane_index: 车道索引（0=最右）
        - num_lanes: 该 edge 的总车道数

    调节粗细：修改本文件顶部 ARROW_THICKNESS（单位米），建议 0.25～0.5。
    """
    scene = trimesh.Scene()
    t = ARROW_THICKNESS
    # 使用 PBR 材质的 baseColorFactor 保证 glTF 中为纯白，Panda3D 加载后不会变黑
    white_mat = PBRMaterial(
        baseColorFactor=[1.0, 1.0, 1.0, 1.0],
        metallicFactor=0.0,
        roughnessFactor=0.7,
    )

    # 与 map/ground/lane/road 一致的 (x,y,z) -> glb 的变换：rotation_matrix(pi/2, [-1,0,0])
    R_x = trimesh.transformations.rotation_matrix(math.pi / 2, [-1, 0, 0])

    for shape, turn_direction, lane_width, lane_index, num_lanes in lane_arrow_data:
        if len(shape) < 2:
            continue

        pos_in_lane, arrow_heading = _pos_and_heading_from_lane(
            shape, turn_direction, lane_width, lane_index, num_lanes
        )
        x, y, z = pos_in_lane

        R_z = trimesh.transformations.rotation_matrix(
            math.radians(arrow_heading), [0, 0, 1]
        )
        T = trimesh.transformations.translation_matrix([x, z, -y])
        M = T @ R_x @ R_z

        for seg in _arrow_segments_for_turn(ARROW_SCALE, turn_direction):
            a = np.array(seg[0], dtype=np.float64)
            b = np.array(seg[1], dtype=np.float64)
            d = b - a
            L = np.linalg.norm(d)
            if L < 1e-6:
                continue
            D = d / L
            bar = box(extents=[t, L, t])
            R_align = trimesh.geometry.align_vectors(np.array([0.0, 1.0, 0.0]), D)
            bar.apply_transform(R_align)
            bar.apply_transform(trimesh.transformations.translation_matrix((a + b) * 0.5))
            bar.apply_transform(M)
            try:
                bar.visual.material = white_mat
            except (AttributeError, TypeError):
                from trimesh.visual import TextureVisuals
                bar.visual = TextureVisuals(material=white_mat)
            scene.add_geometry(bar)

    return GLBData(gltf.export_glb(scene, include_normals=True))
