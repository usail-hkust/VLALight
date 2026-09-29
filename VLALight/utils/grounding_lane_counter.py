"""
基于 VLM Grounding 的精准车道级别计数模块

核心功能：
1. VLM Grounding 检测车辆 bbox
2. 结合车道边界线进行精确车道分配
3. 输出车道级别的车辆计数

作者：Cascade AI
日期：2025-03-14
"""
import json
import numpy as np
from typing import List, Dict, Tuple, Optional


class GroundingLaneCounter:
    """基于 Grounding 的车道级别计数器"""
    
    def __init__(self):
        self.y_threshold = 400  # Implementation note.
        self.single_lane_x_range = 50  # Implementation note.
        self.lane_gap_threshold = 50  # Implementation note.
    
    def parse_grounding_response(self, vlm_response: str) -> List[Dict]:
        """
        解析 VLM Grounding 响应
        
        Args:
            vlm_response: VLM 返回的文本（可能包含 JSON）
            
        Returns:
            检测结果列表 [{"bbox_2d": [x1,y1,x2,y2], "label": "car"}, ...]
        """
        # Implementation note.
        text = vlm_response.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        
        text = text.strip()
        
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and 'bbox_2d' in data:
                return [data]
        except json.JSONDecodeError:
            # Implementation note.
            import re
            match = re.search(r'\[.*\]', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
        
        return []
    
    def get_vehicle_centers(self, detections: List[Dict]) -> List[Dict]:
        """
        提取车辆中心点坐标
        
        Args:
            detections: Grounding 检测结果
            
        Returns:
            车辆中心点列表 [{"x": cx, "y": cy, "label": "car", "bbox": [...]}, ...]
        """
        centers = []
        for det in detections:
            bbox = det.get('bbox_2d', [])
            if len(bbox) != 4:
                continue
            
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            centers.append({
                'x': cx,
                'y': cy,
                'label': det.get('label', 'vehicle'),
                'bbox': bbox
            })
        return centers
    
    def detect_lane_structure_by_anchor(self, centers: List[Dict]) -> Dict:
        """
        使用锚点策略检测车道结构
        
        策略：
        1. 找到靠近路口的车辆（y > threshold）作为锚点
        2. 分析锚点车辆的 x 坐标分布确定车道数和车道中心
        3. 避免 3D 透视导致的 x 坐标偏移
        
        Args:
            centers: 车辆中心点列表
            
        Returns:
            {
                'lane_count': int,
                'lane_centers': [x1, x2, ...],
                'method': str,
                'anchor_count': int
            }
        """
        if not centers:
            return {
                'lane_count': 0,
                'lane_centers': [],
                'method': 'no_vehicles',
                'anchor_count': 0
            }
        
        # Implementation note.
        anchor_vehicles = [v for v in centers if v['y'] > self.y_threshold]
        
        if not anchor_vehicles:
            # Implementation note.
            y_threshold = np.percentile([v['y'] for v in centers], 50)
            anchor_vehicles = [v for v in centers if v['y'] > y_threshold]
        else:
            y_threshold = self.y_threshold
        
        # Implementation note.
        anchor_x = [v['x'] for v in anchor_vehicles]
        x_min, x_max = min(anchor_x), max(anchor_x)
        x_range = x_max - x_min
        
        # Implementation note.
        if x_range < self.single_lane_x_range:
            # Implementation note.
            lane_centers = [np.mean(anchor_x)]
            lane_count = 1
            method = 'single_lane_anchor'
        else:
            # Implementation note.
            sorted_anchor_x = sorted(anchor_x)
            
            # Implementation note.
            gaps = []
            for i in range(len(sorted_anchor_x) - 1):
                gap = sorted_anchor_x[i+1] - sorted_anchor_x[i]
                gaps.append((i, gap, sorted_anchor_x[i], sorted_anchor_x[i+1]))
            
            # Implementation note.
            lane_splits = [g for g in gaps if g[1] > self.lane_gap_threshold]
            
            if not lane_splits:
                # Implementation note.
                lane_centers = [np.mean(anchor_x)]
                lane_count = 1
                method = 'single_lane_no_gap'
            else:
                # Implementation note.
                lane_centers = []
                start_idx = 0
                
                for split in lane_splits:
                    split_idx = split[0]
                    lane_x = sorted_anchor_x[start_idx:split_idx+1]
                    lane_centers.append(np.mean(lane_x))
                    start_idx = split_idx + 1
                
                # Implementation note.
                lane_x = sorted_anchor_x[start_idx:]
                lane_centers.append(np.mean(lane_x))
                
                lane_count = len(lane_centers)
                method = 'multi_lane_anchor'
        
        return {
            'lane_count': lane_count,
            'lane_centers': lane_centers,
            'method': method,
            'anchor_count': len(anchor_vehicles),
            'y_threshold': y_threshold
        }
    
    def assign_vehicles_to_lanes(
        self,
        centers: List[Dict],
        lane_centers: List[float]
    ) -> Dict[int, List[Dict]]:
        """
        将车辆分配到最近的车道
        
        Args:
            centers: 车辆中心点列表
            lane_centers: 车道中心 x 坐标列表
            
        Returns:
            {lane_id: [vehicle1, vehicle2, ...], ...}
        """
        lanes = {i: [] for i in range(len(lane_centers))}
        
        for vehicle in centers:
            # Implementation note.
            distances = [abs(vehicle['x'] - lc) for lc in lane_centers]
            lane_id = distances.index(min(distances))
            lanes[lane_id].append(vehicle)
        
        return lanes
    
    def assign_vehicles_with_lane_boundaries(
        self,
        centers: List[Dict],
        lane_boundaries: Optional[List[Tuple[float, float]]] = None
    ) -> Dict[int, List[Dict]]:
        """
        使用车道边界线进行精确车道分配
        
        Args:
            centers: 车辆中心点列表
            lane_boundaries: 车道边界 [(x_min, x_max), ...] for each lane
            
        Returns:
            {lane_id: [vehicle1, vehicle2, ...], ...}
        """
        if not lane_boundaries:
            # Implementation note.
            lane_structure = self.detect_lane_structure_by_anchor(centers)
            return self.assign_vehicles_to_lanes(centers, lane_structure['lane_centers'])
        
        lanes = {i: [] for i in range(len(lane_boundaries))}
        
        for vehicle in centers:
            vx = vehicle['x']
            
            # Implementation note.
            assigned = False
            for lane_id, (x_min, x_max) in enumerate(lane_boundaries):
                if x_min <= vx <= x_max:
                    lanes[lane_id].append(vehicle)
                    assigned = True
                    break
            
            # Implementation note.
            if not assigned:
                distances = [min(abs(vx - x_min), abs(vx - x_max)) 
                           for x_min, x_max in lane_boundaries]
                lane_id = distances.index(min(distances))
                lanes[lane_id].append(vehicle)
        
        return lanes
    
    def count_vehicles_per_lane(
        self,
        detections: List[Dict],
        lane_boundaries: Optional[List[Tuple[float, float]]] = None,
        direction: str = "Unknown"
    ) -> Dict:
        """
        车道级别计数主函数
        
        Args:
            detections: VLM Grounding 检测结果
            lane_boundaries: 可选的车道边界线
            direction: 方向名称（用于日志）
            
        Returns:
            {
                'total_vehicles': int,
                'lane_count': int,
                'per_lane_count': [count1, count2, ...],
                'lanes': {lane_id: [vehicle_info, ...]},
                'method': str
            }
        """
        if not detections:
            return {
                'total_vehicles': 0,
                'lane_count': 0,
                'per_lane_count': [],
                'lanes': {},
                'method': 'no_detections'
            }
        
        # Implementation note.
        centers = self.get_vehicle_centers(detections)
        
        if not centers:
            return {
                'total_vehicles': 0,
                'lane_count': 0,
                'per_lane_count': [],
                'lanes': {},
                'method': 'invalid_detections'
            }
        
        # Implementation note.
        if lane_boundaries:
            # Implementation note.
            lanes = self.assign_vehicles_with_lane_boundaries(centers, lane_boundaries)
            method = 'lane_boundaries'
            lane_count = len(lane_boundaries)
        else:
            # Implementation note.
            lane_structure = self.detect_lane_structure_by_anchor(centers)
            lanes = self.assign_vehicles_to_lanes(centers, lane_structure['lane_centers'])
            method = lane_structure['method']
            lane_count = lane_structure['lane_count']
        
        # Implementation note.
        per_lane_count = [len(lanes[i]) for i in sorted(lanes.keys())]
        
        return {
            'total_vehicles': len(centers),
            'lane_count': lane_count,
            'per_lane_count': per_lane_count,
            'lanes': lanes,
            'method': method,
            'direction': direction
        }
    
    def format_lane_count_result(self, result: Dict) -> str:
        """
        格式化车道计数结果为文本描述
        
        Args:
            result: count_vehicles_per_lane 的返回值
            
        Returns:
            文本描述
        """
        if result['total_vehicles'] == 0:
            return "No vehicles detected."
        
        lines = []
        lines.append(f"Direction: {result.get('direction', 'Unknown')}")
        lines.append(f"Total vehicles: {result['total_vehicles']}")
        lines.append(f"Number of lanes: {result['lane_count']}")
        lines.append(f"Method: {result['method']}")
        
        if result['lane_count'] > 0:
            lines.append("\nPer-lane breakdown:")
            for lane_id, count in enumerate(result['per_lane_count']):
                lane_name = self._get_lane_name(lane_id, result['lane_count'])
                lines.append(f"  {lane_name}: {count} vehicles")
        
        return "\n".join(lines)
    
    def _get_lane_name(self, lane_id: int, total_lanes: int) -> str:
        """获取车道名称"""
        if total_lanes == 1:
            return "Lane 0 (Center)"
        elif total_lanes == 2:
            return ["Lane 0 (Left)", "Lane 1 (Right)"][lane_id]
        elif total_lanes == 3:
            return ["Lane 0 (Left/Right-turn)", 
                   "Lane 1 (Center/Straight)", 
                   "Lane 2 (Right/Left-turn)"][lane_id]
        else:
            return f"Lane {lane_id}"


# Implementation note.
def count_vehicles_from_grounding(
    vlm_response: str,
    lane_boundaries: Optional[List[Tuple[float, float]]] = None,
    direction: str = "Unknown"
) -> Dict:
    """
    从 VLM Grounding 响应中计数车辆
    
    Args:
        vlm_response: VLM 返回的文本
        lane_boundaries: 可选的车道边界
        direction: 方向名称
        
    Returns:
        车道级别计数结果
    """
    counter = GroundingLaneCounter()
    detections = counter.parse_grounding_response(vlm_response)
    return counter.count_vehicles_per_lane(detections, lane_boundaries, direction)
