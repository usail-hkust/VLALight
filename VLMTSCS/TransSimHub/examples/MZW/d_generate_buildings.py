'''
@Description: 为MZW路网生成3D建筑物
程序化生成道路周边的小房子和建筑
'''
import os
import random
import numpy as np
import trimesh
from shapely.geometry import Polygon, Point, MultiPolygon, LineString
from shapely.ops import unary_union
from tshub.utils.init_log import set_logger
from tshub.utils.get_abs_path import get_abs_path
import xml.etree.ElementTree as ET

path_convert = get_abs_path(__file__)
set_logger(path_convert('./'), terminal_log_level='INFO')

class BuildingGenerator:
    """
    参考 ImportAndGenerate.cs 的建筑生成逻辑：
    - 沿道路放置建筑，垂直于道路方向偏移
    - 使用碰撞检测避免建筑重叠
    """
    
    # 建筑放置参数（参考 ImportAndGenerate.cs）
    LENGTH_FROM_ROAD_TO_BUILDING = 30.0    # 建筑距道路中心的距离（米）
    LENGTH_BETWEEN_BUILDINGS = 30.0         # 沿道路方向建筑间距（米）
    MAX_LANES_FOR_BUILDINGS = 6             # 最多6车道的道路才放建筑
    MIN_BUILDING_DISTANCE = 20.0            # 建筑间最小距离（米）
    
    def __init__(self, net_file):
        self.net_file = net_file
        self.buildings = []
        self.lanes = []  # 存储车道信息
        self.placed_buildings = []  # 已放置建筑的位置，用于碰撞检测
        
    def parse_sumo_network(self):
        """解析SUMO网络文件，获取车道信息（参考 ImportAndGenerate.cs 的 parseXMLfiles）"""
        tree = ET.parse(self.net_file)
        root = tree.getroot()
        
        # 统计每条edge的车道数
        edge_lane_count = {}
        
        for edge in root.findall('edge'):
            edge_id = edge.get('id')
            if edge_id and not edge_id.startswith(':'):  # 排除内部边
                lane_count = len(edge.findall('lane'))
                edge_lane_count[edge_id] = lane_count
                
                for lane in edge.findall('lane'):
                    shape = lane.get('shape')
                    if shape:
                        # 解析车道形状坐标
                        points = []
                        for point_str in shape.split():
                            x, y = map(float, point_str.split(','))
                            points.append([x, y])
                        
                        self.lanes.append({
                            'id': lane.get('id'),
                            'edge_id': edge_id,
                            'points': points,
                            'width': float(lane.get('width', 3.2)),
                            'lane_count': lane_count  # 该edge的车道数
                        })
        
        print(f"解析到 {len(self.lanes)} 条车道")
    
    def create_simple_house(self, x, y, width=15, length=12, height=8):
        """创建简单的房子模型（小型住宅）"""
        # 主体立方体
        box = trimesh.creation.box(extents=[width, length, height])
        
        # 三角屋顶
        roof_height = 4
        roof_vertices = np.array([
            [-width/2, -length/2, height/2],  # 底部四个角
            [width/2, -length/2, height/2],
            [width/2, length/2, height/2],
            [-width/2, length/2, height/2],
            [0, -length/2, height/2 + roof_height],  # 屋顶顶点
            [0, length/2, height/2 + roof_height]
        ])
        
        roof_faces = np.array([
            [0, 1, 4],  # 前面三角形
            [2, 3, 5],  # 后面三角形
            [1, 2, 4],  # 右侧梯形
            [2, 5, 4],
            [3, 0, 5],  # 左侧梯形
            [0, 4, 5]
        ])
        
        roof = trimesh.Trimesh(vertices=roof_vertices, faces=roof_faces)
        
        # 组合房子
        house = trimesh.util.concatenate([box, roof])
        
        # 设置位置
        house.apply_translation([x, y, height/2])
        
        # 随机颜色（暖色调）
        colors = [
            [200, 180, 160, 255],  # 米色
            [180, 160, 140, 255],  # 棕色
            [220, 200, 180, 255],  # 浅米色
            [160, 140, 120, 255],  # 深棕色
        ]
        house.visual.face_colors = random.choice(colors)
        
        return house
    
    def create_office_building(self, x, y, width=20, length=15, height=25):
        """创建办公楼模型（小型办公楼）"""
        # 主体
        building = trimesh.creation.box(extents=[width, length, height])
        building.apply_translation([x, y, height/2])
        
        # 现代建筑颜色（冷色调）
        colors = [
            [150, 150, 170, 255],  # 灰蓝色
            [140, 140, 140, 255],  # 灰色
            [160, 170, 180, 255],  # 浅灰蓝
            [120, 130, 140, 255],  # 深灰
        ]
        building.visual.face_colors = random.choice(colors)
        
        return building
    
    def create_shop(self, x, y, width=18, length=14, height=6):
        """创建商店模型（小型商铺）"""
        # 扁平的商业建筑
        shop = trimesh.creation.box(extents=[width, length, height])
        shop.apply_translation([x, y, height/2])
        
        # 商业建筑颜色（明亮色调）
        colors = [
            [180, 200, 220, 255],  # 浅蓝色
            [200, 180, 200, 255],  # 浅紫色
            [180, 220, 180, 255],  # 浅绿色
            [220, 200, 160, 255],  # 浅黄色
        ]
        shop.visual.face_colors = random.choice(colors)
        
        return shop
    
    def calculate_building_positions(self):
        """
        参考 ImportAndGenerate.cs 第302-351行的建筑放置逻辑：
        1. 遍历每条车道的形状点
        2. 计算道路方向和垂直方向
        3. 沿道路每隔一定距离放置建筑
        4. 建筑向道路垂直方向偏移
        5. 碰撞检测避免重叠
        """
        building_positions = []
        
        # 收集所有道路线段用于碰撞检测
        all_road_lines = []
        for lane in self.lanes:
            points = lane['points']
            for i in range(len(points) - 1):
                all_road_lines.append(LineString([points[i], points[i+1]]))
        
        # 合并道路区域（用于碰撞检测）
        road_buffer = unary_union([line.buffer(15) for line in all_road_lines])  # 道路缓冲15米
        
        for lane in self.lanes:
            # 只在车道数较少的道路放建筑（参考 maxLanesForBuildings）
            if lane['lane_count'] > self.MAX_LANES_FOR_BUILDINGS:
                continue
            
            points = lane['points']
            
            # 遍历车道的每个线段
            for i in range(len(points) - 1):
                x1, y1 = points[i]
                x2, y2 = points[i + 1]
                
                # 计算线段长度
                dx = x2 - x1
                dy = y2 - y1
                length = np.sqrt(dx*dx + dy*dy)
                
                # 线段太短则跳过
                if length < self.LENGTH_BETWEEN_BUILDINGS:
                    continue
                
                # 计算道路方向角度（参考 ImportAndGenerate.cs）
                angle = np.arctan2(dy, dx)
                
                # 计算垂直方向（道路右侧）
                perp_x = -np.sin(angle)
                perp_y = np.cos(angle)
                
                # 计算沿道路可放置的建筑数量
                num_buildings = int(length / self.LENGTH_BETWEEN_BUILDINGS)
                length_between = length / (num_buildings + 1)
                
                for building_num in range(num_buildings):
                    # 计算沿道路的位置（参考 ImportAndGenerate.cs 第314-316行）
                    ratio = (length_between * (building_num + 1)) / length
                    road_x = (1 - ratio) * x1 + ratio * x2
                    road_y = (1 - ratio) * y1 + ratio * y2
                    
                    # 在道路两侧都放置建筑
                    for side in [1, -1]:  # 1=右侧, -1=左侧
                        building_x = road_x + perp_x * self.LENGTH_FROM_ROAD_TO_BUILDING * side
                        building_y = road_y + perp_y * self.LENGTH_FROM_ROAD_TO_BUILDING * side
                        
                        # 碰撞检测：检查是否与道路重叠
                        building_point = Point(building_x, building_y)
                        if road_buffer.contains(building_point):
                            continue  # 在道路上，跳过
                        
                        # 碰撞检测：检查是否与已放置的建筑重叠
                        is_overlapping = False
                        for placed in self.placed_buildings:
                            dist = np.sqrt((building_x - placed[0])**2 + (building_y - placed[1])**2)
                            if dist < self.MIN_BUILDING_DISTANCE:
                                is_overlapping = True
                                break
                        
                        if not is_overlapping:
                            building_positions.append({
                                'x': building_x,
                                'y': building_y,
                                'angle': angle,
                                'type': random.choice(['house', 'house', 'shop'])
                            })
                            self.placed_buildings.append((building_x, building_y))
        
        print(f"有效建筑位置: {len(building_positions)}")
        return building_positions
    
    def generate_buildings(self):
        """生成所有建筑"""
        print("开始生成建筑...")
        
        # 解析SUMO网络文件获取车道信息
        self.parse_sumo_network()
        
        # 计算建筑位置
        positions = self.calculate_building_positions()
        print(f"计算出 {len(positions)} 个建筑位置")
        
        # 生成建筑模型
        all_buildings = []
        
        for i, pos in enumerate(positions):
            x, y = pos['x'], pos['y']
            building_type = pos['type']
            
            if building_type == 'house':
                building = self.create_simple_house(x, y)
            elif building_type == 'office':
                building = self.create_office_building(x, y)
            else:  # shop
                building = self.create_shop(x, y)
            
            all_buildings.append(building)
            
            if (i + 1) % 20 == 0:
                print(f"已生成 {i + 1} 个建筑...")
        
        # 合并所有建筑
        if all_buildings:
            combined_buildings = trimesh.util.concatenate(all_buildings)
            return combined_buildings
        else:
            print("没有生成任何建筑")
            return None
    
    def save_buildings(self, output_dir):
        """保存建筑到GLB文件"""
        buildings = self.generate_buildings()
        
        if buildings:
            # GLB uses Y-up coordinate system, but Panda3D uses Z-up
            # We need to rotate the model: swap Y and Z, and negate the new Z
            # This is equivalent to rotating -90 degrees around X axis
            vertices = buildings.vertices.copy()
            new_vertices = vertices.copy()
            new_vertices[:, 1] = vertices[:, 2]   # new Y = old Z (height goes to Y)
            new_vertices[:, 2] = -vertices[:, 1]  # new Z = -old Y (Y goes to -Z)
            buildings.vertices = new_vertices
            
            output_file = os.path.join(output_dir, 'buildings.glb')
            buildings.export(output_file)
            print(f"建筑已保存到: {output_file}")
            print(f"生成了 {len(buildings.faces)} 个面片的建筑群")
        else:
            print("没有建筑可保存")

if __name__ == '__main__':
    # 输入文件 - 使用当前目录的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    net_file = os.path.join(current_dir, "mzwmap_phase.net.xml")
    output_dir = os.path.join(current_dir, "3d_assets")
    
    # 检查文件是否存在
    if not os.path.exists(net_file):
        print(f"错误：找不到网络文件 {net_file}")
        print(f"当前目录: {current_dir}")
        print("目录内容:")
        for f in os.listdir(current_dir):
            print(f"  {f}")
        exit(1)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成建筑（参考 ImportAndGenerate.cs 的逻辑）
    generator = BuildingGenerator(net_file)
    generator.save_buildings(output_dir)
    
    print("建筑生成完成！")
    print("现在可以运行 b_run_mzw_simulation.py 查看效果")
