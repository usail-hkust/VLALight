"""向济南路由文件注入紧急车辆。

策略：每 30 分钟（1800s）从现有长路径车辆中随机抽取 1~10 辆，
改为 emergency / fire_engine / police 类型。

用法:
  python utils/inject_emergency_vehicles.py
  python utils/inject_emergency_vehicles.py --min_per_period 3 --max_per_period 8
  python utils/inject_emergency_vehicles.py --input data/Jinan/3_4/jinan.rou.xml --output data/Jinan/3_4/jinan_emergency.rou.xml
"""
import os
import sys
import random
import argparse
import xml.etree.ElementTree as ET
from copy import deepcopy

# Implementation note.
EMERGENCY_VTYPES = {
    'emergency': {
        'length': '6.5', 'width': '2.2', 'minGap': '2.0',
        'maxSpeed': '16.67', 'accel': '3.0', 'decel': '5.0',
        'color': '255,165,0', 'tau': '1.0',
    },
    'fire_engine': {
        'length': '7.1', 'width': '2.5', 'minGap': '2.0',
        'maxSpeed': '16.67', 'accel': '2.5', 'decel': '5.0',
        'color': '255,0,0', 'tau': '1.0',
    },
    'police': {
        'length': '5.0', 'width': '2.0', 'minGap': '2.0',
        'maxSpeed': '16.67', 'accel': '3.5', 'decel': '5.0',
        'color': '0,0,255', 'tau': '1.0',
    },
}

# Implementation note.
TYPE_WEIGHTS = {'emergency': 0.5, 'fire_engine': 0.25, 'police': 0.25}


def pick_emergency_type():
    """按权重随机选择紧急车辆类型。"""
    r = random.random()
    cumulative = 0.0
    for vtype, weight in TYPE_WEIGHTS.items():
        cumulative += weight
        if r <= cumulative:
            return vtype
    return 'emergency'


def inject_emergency_vehicles(
    input_path: str,
    output_path: str,
    period_seconds: int = 1800,
    min_per_period: int = 1,
    max_per_period: int = 10,
    min_route_edges: int = 4,
    seed: int = 42,
):
    """向路由文件注入紧急车辆。
    
    Args:
        input_path: 原始 .rou.xml 路径
        output_path: 输出 .rou.xml 路径
        period_seconds: 每个时间段的秒数（默认 1800 = 30分钟）
        min_per_period: 每个时间段最少紧急车辆数
        max_per_period: 每个时间段最多紧急车辆数
        min_route_edges: 最少 edge 数（确保穿过足够路口）
        seed: 随机种子
    """
    random.seed(seed)
    
    print(f"输入文件: {input_path}")
    print(f"输出文件: {output_path}")
    print(f"时间段: 每 {period_seconds}s ({period_seconds//60} 分钟)")
    print(f"每段紧急车辆: {min_per_period}~{max_per_period} 辆")
    print(f"最少路径 edge 数: {min_route_edges}")
    print(f"随机种子: {seed}")
    print()
    
    # Implementation note.
    tree = ET.parse(input_path)
    root = tree.getroot()
    
    # Implementation note.
    vehicles = []
    for veh in root.findall('vehicle'):
        route_elem = veh.find('route')
        if route_elem is None:
            continue
        edges = route_elem.get('edges', '').split()
        depart = float(veh.get('depart', '0'))
        vehicles.append({
            'element': veh,
            'id': veh.get('id'),
            'depart': depart,
            'edges': edges,
            'num_edges': len(edges),
        })
    
    print(f"总车辆数: {len(vehicles)}")
    
    # Implementation note.
    candidates = [v for v in vehicles if v['num_edges'] >= min_route_edges]
    print(f"长路径候选车辆 (≥{min_route_edges} edges): {len(candidates)}")
    
    if not candidates:
        print("ERROR: 没有满足条件的候选车辆！")
        return
    
    # Implementation note.
    max_depart = max(v['depart'] for v in candidates)
    num_periods = int(max_depart // period_seconds) + 1
    print(f"仿真时长: {max_depart:.0f}s, 分为 {num_periods} 个时间段")
    
    period_buckets = {i: [] for i in range(num_periods)}
    for v in candidates:
        period_idx = int(v['depart'] // period_seconds)
        period_idx = min(period_idx, num_periods - 1)
        period_buckets[period_idx].append(v)
    
    # Implementation note.
    selected = []
    for period_idx in range(num_periods):
        bucket = period_buckets[period_idx]
        if not bucket:
            continue
        
        count = random.randint(min_per_period, min(max_per_period, len(bucket)))
        chosen = random.sample(bucket, count)
        
        for v in chosen:
            etype = pick_emergency_type()
            selected.append({
                'vehicle': v,
                'emergency_type': etype,
                'period': period_idx,
            })
    
    print(f"\n总共选中 {len(selected)} 辆紧急车辆:")
    
    # Implementation note.
    type_counts = {}
    period_counts = {}
    for s in selected:
        t = s['emergency_type']
        p = s['period']
        type_counts[t] = type_counts.get(t, 0) + 1
        period_counts[p] = period_counts.get(p, 0) + 1
    
    for t, c in sorted(type_counts.items()):
        print(f"  {t}: {c} 辆")
    
    print(f"\n各时间段分布:")
    for p in range(num_periods):
        start = p * period_seconds
        end = min((p + 1) * period_seconds, max_depart)
        c = period_counts.get(p, 0)
        print(f"  {start:5.0f}s ~ {end:5.0f}s: {c} 辆")
    
    # Implementation note.
    # Implementation note.
    existing_vtypes = root.findall('vType')
    insert_pos = 0
    if existing_vtypes:
        # Implementation note.
        for i, child in enumerate(root):
            if child.tag == 'vType':
                insert_pos = i + 1
    
    for vtype_id, attrs in EMERGENCY_VTYPES.items():
        # Implementation note.
        if any(vt.get('id') == vtype_id for vt in existing_vtypes):
            print(f"  vType '{vtype_id}' 已存在，跳过")
            continue
        
        vtype_elem = ET.Element('vType', id=vtype_id, **attrs)
        root.insert(insert_pos, vtype_elem)
        insert_pos += 1
    
    # Implementation note.
    for s in selected:
        veh_elem = s['vehicle']['element']
        veh_elem.set('type', s['emergency_type'])
    
    # Implementation note.
    tree.write(output_path, encoding='utf-8', xml_declaration=True)
    
    print(f"\n输出已保存到: {output_path}")
    
    # Implementation note.
    print(f"\n选中车辆列表:")
    print(f"  {'ID':>6s}  {'depart':>8s}  {'类型':>12s}  {'路径edges':>8s}  路径")
    print(f"  {'-'*6}  {'-'*8}  {'-'*12}  {'-'*8}  {'-'*40}")
    for s in sorted(selected, key=lambda x: x['vehicle']['depart']):
        v = s['vehicle']
        route_str = ' → '.join(v['edges'][:4])
        if v['num_edges'] > 4:
            route_str += f' → ... ({v["num_edges"]} edges)'
        print(f"  {v['id']:>6s}  {v['depart']:>8.0f}s  {s['emergency_type']:>12s}  {v['num_edges']:>8d}  {route_str}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='向路由文件注入紧急车辆')
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    default_input = os.path.join(project_root, 'data', 'Jinan', '3_4', 'jinan.rou.xml')
    default_output = os.path.join(project_root, 'data', 'Jinan', '3_4', 'jinan_emergency.rou.xml')
    
    parser.add_argument('--input', type=str, default=default_input, help='输入路由文件')
    parser.add_argument('--output', type=str, default=default_output, help='输出路由文件')
    parser.add_argument('--period', type=int, default=1800, help='时间段秒数 (默认1800)')
    parser.add_argument('--min_per_period', type=int, default=1, help='每段最少紧急车辆')
    parser.add_argument('--max_per_period', type=int, default=10, help='每段最多紧急车辆')
    parser.add_argument('--min_edges', type=int, default=4, help='最少路径edge数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    
    inject_emergency_vehicles(
        input_path=args.input,
        output_path=args.output,
        period_seconds=args.period,
        min_per_period=args.min_per_period,
        max_per_period=args.max_per_period,
        min_route_edges=args.min_edges,
        seed=args.seed,
    )
