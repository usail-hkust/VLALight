"""
生成路网宏观图，显示摄像头方向映射
"""
import os
import sys
import xml.etree.ElementTree as ET

# Implementation note.
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.normpath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root)

from utils.direction_mapper import generate_direction_mapping_from_net

def create_macro_view(net_file: str, direction_mapping: dict, output_path: str, scenario: str = 'jinan'):
    """
    创建宏观俯视图，显示路网结构和摄像头方向
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    tree = ET.parse(net_file)
    root = tree.getroot()
    
    if scenario == 'newyork':
        fig, ax = plt.subplots(1, 1, figsize=(24, 60))
        show_cam_labels = False
        arrow_scale = 0.8
    else:
        fig, ax = plt.subplots(1, 1, figsize=(16, 14))
        show_cam_labels = True
        arrow_scale = 1.0
    
    edge_labels = {}
    for edge in root.findall('edge'):
        edge_id = edge.get('id')
        if edge_id and edge_id.startswith(':'):
            continue
        all_points = []
        for lane in edge.findall('lane'):
            shape_str = lane.get('shape')
            if shape_str:
                points = []
                for point_str in shape_str.strip().split():
                    coords = point_str.split(',')
                    if len(coords) >= 2:
                        points.append((float(coords[0]), float(coords[1])))
                if len(points) >= 2:
                    xs = [p[0] for p in points]
                    ys = [p[1] for p in points]
                    ax.plot(xs, ys, 'gray', linewidth=1, alpha=0.6)
                    all_points.extend(points)
        if all_points:
            mid_idx = len(all_points) // 2
            edge_labels[edge_id] = all_points[mid_idx]
    
    junction_positions = {}
    for junction in root.findall('junction'):
        junction_id = junction.get('id')
        junction_type = junction.get('type')
        x = float(junction.get('x', 0))
        y = float(junction.get('y', 0))
        
        if junction_type and junction_type.startswith('traffic_light'):
            junction_positions[junction_id] = (x, y)
            ax.plot(x, y, 'ro', markersize=12, zorder=10)
            ax.annotate(junction_id, (x, y), textcoords="offset points", 
                       xytext=(0, -20), fontsize=8, color='black', fontweight='bold',
                       ha='center', va='top',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.9, edgecolor='red'))
            
            if junction_id in direction_mapping:
                cam_mapping = direction_mapping[junction_id]
                arrow_start = 15 * arrow_scale
                arrow_end = 80 * arrow_scale
                direction_vectors = {
                    'N': (0, 1),
                    'S': (0, -1),
                    'E': (1, 0),
                    'W': (-1, 0),
                }
                colors = {'N': 'blue', 'S': 'green', 'E': 'orange', 'W': 'purple'}
                
                for cam_idx, direction in cam_mapping.items():
                    vx, vy = direction_vectors.get(direction, (0, 0))
                    start_x = x + vx * arrow_start
                    start_y = y + vy * arrow_start
                    end_x = x + vx * arrow_end
                    end_y = y + vy * arrow_end
                    
                    ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                               arrowprops=dict(arrowstyle='-|>', color=colors.get(direction, 'black'), 
                                              lw=2.5 * arrow_scale, mutation_scale=20 * arrow_scale),
                               zorder=5)
                    if show_cam_labels:
                        label_x = x + vx * (arrow_end + 15)
                        label_y = y + vy * (arrow_end + 15)
                        ax.text(label_x, label_y, f'Cam{cam_idx}', fontsize=6, 
                               color=colors.get(direction, 'black'), ha='center', va='center',
                               fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none'))
    
    ax.set_aspect('equal')
    ax.set_title('Road Network Macro View\n(Arrows show camera facing directions, Red dots are traffic light junctions)', fontsize=12)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.grid(True, alpha=0.3)
    
    legend_elements = [
        patches.Patch(facecolor='blue', label='N - Camera facing North'),
        patches.Patch(facecolor='green', label='S - Camera facing South'),
        patches.Patch(facecolor='orange', label='E - Camera facing East'),
        patches.Patch(facecolor='purple', label='W - Camera facing West'),
        patches.Patch(facecolor='red', label='Traffic Light Junction'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"宏观图已保存: {output_path}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='生成路网宏观图')
    parser.add_argument('--scenario', type=str, default='jinan',
                        choices=['jinan', 'hangzhou', 'newyork'],
                        help='场景名称')
    parser.add_argument('--output', type=str, default='./output/verify_images',
                        help='输出图片目录')
    args = parser.parse_args()
    
    SCENARIO_CONFIGS = {
        'jinan': {
            'data_dir': 'data/Jinan/3_4',
            'net_file': 'jinan.net.xml',
        },
        'hangzhou': {
            'data_dir': 'data/Hangzhou/4_4',
            'net_file': 'hangzhou.net.xml',
        },
        'newyork': {
            'data_dir': 'data/NewYork/28_7',
            'net_file': 'newyork.net.xml',
        },
    }
    
    config = SCENARIO_CONFIGS[args.scenario]
    data_dir = os.path.join(project_root, config['data_dir'])
    net_file = os.path.join(data_dir, config['net_file'])
    
    print(f"场景: {args.scenario}")
    print(f"路网文件: {net_file}")
    
    print("\n生成方向映射...")
    direction_mapping = generate_direction_mapping_from_net(net_file)
    print(f"找到 {len(direction_mapping)} 个路口")
    
    output_dir = os.path.join(project_root, args.output, args.scenario)
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n生成宏观俯视图...")
    macro_view_path = os.path.join(output_dir, f"macro_view_{args.scenario}.png")
    create_macro_view(net_file, direction_mapping, macro_view_path, scenario=args.scenario)
    
    print("\n完成！")
