import json
import os
import math
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple
import numpy as np

def calculate_direction(heading_deg: float) -> str:
    """
    根据道路的朝向角度，计算摄像头所在位置的方向（N, S, E, W）
    
    命名方式：按摄像头所在位置命名（路口上边的摄像头叫N，下边叫S，左边叫W，右边叫E）
    
    参数:
        heading_deg: 道路的朝向角度（度），表示车辆行驶方向
                     0度=向东, 90度=向北, 180度=向西, 270度=向南
    
    返回:
        str: 摄像头所在位置的方向 ('N', 'S', 'E', 'W')
        - 道路heading=0°（车从西向东行驶）-> 摄像头在路口西边，命名为 W
        - 道路heading=90°（车从南向北行驶）-> 摄像头在路口南边，命名为 S
        - 道路heading=180°（车从东向西行驶）-> 摄像头在路口东边，命名为 E
        - 道路heading=270°（车从北向南行驶）-> 摄像头在路口北边，命名为 N
    """
    normalized_angle = heading_deg % 360
    if 315 <= normalized_angle or normalized_angle < 45:
        return 'E'
    elif 45 <= normalized_angle < 135:
        return 'N'
    elif 135 <= normalized_angle < 225:
        return 'W'
    else:
        return 'S'

def save_direction_mapping(mapping: Dict, output_path: str) -> None:
    """
    将方向映射保存到JSON文件
    
    参数:
        mapping: 方向映射字典
        output_path: 输出文件路径
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)

def load_direction_mapping(file_path: str) -> Dict:
    """
    从JSON文件加载方向映射
    
    参数:
        file_path: 输入文件路径
    
    返回:
        Dict: 方向映射字典
    """
    if not os.path.exists(file_path):
        return {}
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def parse_shape(shape_str: str) -> List[Tuple[float, float]]:
    """
    解析 SUMO 的 shape 字符串为坐标点列表
    
    参数:
        shape_str: 形如 "x1,y1 x2,y2 x3,y3" 的字符串
    
    返回:
        List[Tuple[float, float]]: 坐标点列表
    """
    points = []
    for point_str in shape_str.strip().split():
        coords = point_str.split(',')
        if len(coords) >= 2:
            points.append((float(coords[0]), float(coords[1])))
    return points

def calculate_heading_from_shape(shape_points: List[Tuple[float, float]]) -> float:
    """
    根据 shape 的最后两个点计算道路的朝向角度
    
    参数:
        shape_points: 坐标点列表
    
    返回:
        float: 朝向角度（度），0度表示东，90度表示北
    """
    if len(shape_points) < 2:
        return 0.0
    p1 = shape_points[-2]
    p2 = shape_points[-1]
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)
    if angle_deg < 0:
        angle_deg += 360
    return angle_deg

def parse_sumo_net_file(net_file: str) -> Dict[str, List[Tuple[str, float]]]:
    """
    从 SUMO 路网文件解析每个路口的进入道路及其朝向角度
    
    重要：保持 incLanes 的出现顺序，因为 TransSimHub 使用这个顺序分配摄像头索引
    
    参数:
        net_file: SUMO 路网文件路径 (.net.xml)
    
    返回:
        Dict[str, List[Tuple[str, float]]]: {junction_id: [(edge_id, heading_angle), ...]}
        列表顺序与 incLanes 中边的出现顺序一致
    """
    tree = ET.parse(net_file)
    root = tree.getroot()
    edge_shapes = {}
    for edge in root.findall('edge'):
        edge_id = edge.get('id')
        if edge_id and not edge_id.startswith(':'):
            shape_str = edge.get('shape')
            if not shape_str:
                lane = edge.find('lane')
                if lane is not None:
                    shape_str = lane.get('shape')
            if shape_str:
                edge_shapes[edge_id] = parse_shape(shape_str)
    result = {}
    for junction in root.findall('junction'):
        junction_id = junction.get('id')
        junction_type = junction.get('type')
        if junction_type and junction_type.startswith('traffic_light'):
            inc_lanes = junction.get('incLanes', '')
            if inc_lanes:
                seen_edges = []
                for lane_id in inc_lanes.split():
                    edge_id = lane_id.rsplit('_', 1)[0]
                    if edge_id in edge_shapes and edge_id not in seen_edges:
                        seen_edges.append(edge_id)
                in_roads_with_heading = []
                for edge_id in seen_edges:
                    heading = calculate_heading_from_shape(edge_shapes[edge_id])
                    in_roads_with_heading.append((edge_id, heading))
                if in_roads_with_heading:
                    result[junction_id] = in_roads_with_heading
    return result

def generate_direction_mapping_from_net(net_file: str) -> Dict[str, Dict[int, str]]:
    """
    从 SUMO 路网文件生成摄像头索引到方向的映射
    
    重要：摄像头索引按 incLanes 的出现顺序分配（与 TransSimHub 一致）
    
    参数:
        net_file: SUMO 路网文件路径 (.net.xml)
    
    返回:
        Dict[str, Dict[int, str]]: 映射字典，格式为 {junction_id: {camera_index: direction}}
    """
    junction_headings = parse_sumo_net_file(net_file)
    direction_mapping = {}
    for junction_id, in_roads_with_heading in junction_headings.items():
        junction_mapping = {}
        for camera_index, (road_id, heading) in enumerate(in_roads_with_heading):
            direction = calculate_direction(heading)
            junction_mapping[camera_index] = direction
        direction_mapping[junction_id] = junction_mapping
    return direction_mapping

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='从 SUMO 路网文件生成方向映射')
    parser.add_argument('--net', type=str, default=None,
                        help='SUMO 路网文件路径 (.net.xml)')
    parser.add_argument('--output', type=str, default=None,
                        help='输出 JSON 文件路径（默认按场景命名）')
    parser.add_argument('--scenario', type=str, default='jinan',
                        choices=['jinan', 'hangzhou', 'newyork'],
                        help='预设场景名称')
    args = parser.parse_args()
    SCENARIO_PATHS = {
        'jinan': 'data/Jinan/3_4/jinan.net.xml',
        'hangzhou': 'data/Hangzhou/4_4/hangzhou.net.xml',
        'newyork': 'data/NewYork/28_7/newyork.net.xml',
    }
    if args.net:
        net_file = args.net
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.normpath(os.path.join(script_dir, '..'))
        net_file = os.path.normpath(os.path.join(project_root, SCENARIO_PATHS[args.scenario]))
    print(f"解析路网文件: {net_file}")
    if not os.path.exists(net_file):
        print(f"错误: 路网文件不存在: {net_file}")
        exit(1)
    mapping = generate_direction_mapping_from_net(net_file)
    if args.output:
        output_path = args.output
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.normpath(os.path.join(script_dir, '..'))
        output_path = os.path.join(project_root, 'output', f'direction_mapping_{args.scenario}.json')
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    save_direction_mapping(mapping, output_path)
    print(f"\n找到 {len(mapping)} 个信号灯路口:")
    for junction_id, cam_mapping in mapping.items():
        print(f"  {junction_id}: {cam_mapping}")
    print(f"\n映射已保存到: {output_path}")
