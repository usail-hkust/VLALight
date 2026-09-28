"""
优化版 Grounding 车道计数器

改进点：
1. 多阈值分层分析
2. 加权聚类（靠近路口的车辆权重更高）
3. DBSCAN 聚类算法
4. 透视校正
"""
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.cluster import DBSCAN


class GroundingLaneCounterV2:
    """优化版车道级别计数器"""
    
    def __init__(self):
        # Implementation note.
        self.y_thresholds = [400, 300, 200]  # Implementation note.
        self.min_anchor_vehicles = 3  # Implementation note.
        
        # Implementation note.
        self.single_lane_x_range = 50
        self.lane_gap_threshold = 50
        self.dbscan_eps = 60  # Implementation note.
        self.dbscan_min_samples = 1
        
        # Implementation note.
        self.perspective_correction = True
        self.y_reference = 500  # Implementation note.
        self.correction_strength = 0.3  # Implementation note.
    
    def parse_grounding_response(self, vlm_response: str) -> List[Dict]:
        """解析 VLM Grounding 响应"""
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
            import re
            match = re.search(r'\[.*\]', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
        
        return []
    
    def get_vehicle_centers(self, detections: List[Dict]) -> List[Dict]:
        """提取车辆中心点坐标"""
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
    
    def correct_perspective(self, centers: List[Dict]) -> List[Dict]:
        """
        透视校正：根据 y 坐标调整 x 坐标
        
        原理：远处的车辆（y 小）的 x 坐标会向中心收缩
        校正方法：根据 y 坐标对 x 坐标进行线性调整
        """
        if not self.perspective_correction or not centers:
            return centers
        
        # Implementation note.
        all_x = [v['x'] for v in centers]
        center_x = np.mean(all_x)
        
        corrected_centers = []
        for v in centers:
            # Implementation note.
            correction_factor = 1 + (self.y_reference - v['y']) / 1000 * self.correction_strength
            
            # Implementation note.
            x_offset = v['x'] - center_x
            corrected_x = center_x + x_offset * correction_factor
            
            corrected_centers.append({
                'x': corrected_x,
                'y': v['y'],
                'label': v['label'],
                'bbox': v['bbox'],
                'original_x': v['x']
            })
        
        return corrected_centers
    
    def get_anchor_vehicles_adaptive(self, centers: List[Dict]) -> Tuple[List[Dict], float]:
        """
        自适应选择锚点车辆
        
        策略：从高到低尝试多个阈值，直到找到足够的锚点车辆
        """
        for threshold in self.y_thresholds:
            anchor_vehicles = [v for v in centers if v['y'] > threshold]
            
            if len(anchor_vehicles) >= self.min_anchor_vehicles:
                return anchor_vehicles, threshold
        
        # Implementation note.
        if centers:
            sorted_by_y = sorted(centers, key=lambda v: v['y'], reverse=True)
            n_anchor = max(self.min_anchor_vehicles, len(sorted_by_y) // 2)
            anchor_vehicles = sorted_by_y[:n_anchor]
            threshold = anchor_vehicles[-1]['y'] if anchor_vehicles else 0
            return anchor_vehicles, threshold
        
        return [], 0
    
    def detect_lanes_by_dbscan(self, centers: List[Dict]) -> Dict:
        """
        使用 DBSCAN 聚类检测车道
        
        优点：
        - 自动确定车道数
        - 对噪声鲁棒
        - 不需要预设车道数
        """
        if not centers:
            return {
                'lane_count': 0,
                'lane_centers': [],
                'method': 'no_vehicles',
                'labels': []
            }
        
        # Implementation note.
        X = np.array([[v['x']] for v in centers])
        
        # Implementation note.
        clustering = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples)
        labels = clustering.fit_predict(X)
        
        # Implementation note.
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)  # Implementation note.
        
        lane_centers = []
        for label in sorted(unique_labels):
            cluster_x = [centers[i]['x'] for i in range(len(centers)) if labels[i] == label]
            lane_centers.append(np.mean(cluster_x))
        
        lane_count = len(lane_centers)
        
        return {
            'lane_count': lane_count,
            'lane_centers': sorted(lane_centers),
            'method': 'dbscan',
            'labels': labels.tolist()
        }
    
    def detect_lanes_by_weighted_anchor(self, centers: List[Dict]) -> Dict:
        """
        加权锚点策略
        
        改进：
        1. 自适应选择锚点车辆
        2. 靠近路口的车辆权重更高
        3. 加权计算车道中心
        """
        if not centers:
            return {
                'lane_count': 0,
                'lane_centers': [],
                'method': 'no_vehicles',
                'anchor_count': 0
            }
        
        # Implementation note.
        anchor_vehicles, y_threshold = self.get_anchor_vehicles_adaptive(centers)
        
        if not anchor_vehicles:
            return {
                'lane_count': 0,
                'lane_centers': [],
                'method': 'no_anchor',
                'anchor_count': 0
            }
        
        # Implementation note.
        anchor_x = [v['x'] for v in anchor_vehicles]
        x_min, x_max = min(anchor_x), max(anchor_x)
        x_range = x_max - x_min
        
        # Implementation note.
        if x_range < self.single_lane_x_range:
            # Implementation note.
            weights = [1.0 + (v['y'] - y_threshold) / 1000 for v in anchor_vehicles]
            lane_centers = [np.average(anchor_x, weights=weights)]
            lane_count = 1
            method = 'single_lane_weighted'
        else:
            # Implementation note.
            sorted_anchor = sorted(anchor_vehicles, key=lambda v: v['x'])
            sorted_x = [v['x'] for v in sorted_anchor]
            
            # Implementation note.
            gaps = []
            for i in range(len(sorted_x) - 1):
                gap = sorted_x[i+1] - sorted_x[i]
                gaps.append((i, gap))
            
            # Implementation note.
            lane_splits = [g for g in gaps if g[1] > self.lane_gap_threshold]
            
            if not lane_splits:
                # Implementation note.
                weights = [1.0 + (v['y'] - y_threshold) / 1000 for v in anchor_vehicles]
                lane_centers = [np.average(anchor_x, weights=weights)]
                lane_count = 1
                method = 'single_lane_no_gap_weighted'
            else:
                # Implementation note.
                lane_centers = []
                start_idx = 0
                
                for split_idx, gap in lane_splits:
                    # Implementation note.
                    lane_vehicles = sorted_anchor[start_idx:split_idx+1]
                    lane_x = [v['x'] for v in lane_vehicles]
                    weights = [1.0 + (v['y'] - y_threshold) / 1000 for v in lane_vehicles]
                    lane_centers.append(np.average(lane_x, weights=weights))
                    start_idx = split_idx + 1
                
                # Implementation note.
                lane_vehicles = sorted_anchor[start_idx:]
                lane_x = [v['x'] for v in lane_vehicles]
                weights = [1.0 + (v['y'] - y_threshold) / 1000 for v in lane_vehicles]
                lane_centers.append(np.average(lane_x, weights=weights))
                
                lane_count = len(lane_centers)
                method = 'multi_lane_weighted'
        
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
        """将车辆分配到最近的车道"""
        lanes = {i: [] for i in range(len(lane_centers))}
        
        for vehicle in centers:
            distances = [abs(vehicle['x'] - lc) for lc in lane_centers]
            lane_id = distances.index(min(distances))
            lanes[lane_id].append(vehicle)
        
        return lanes
    
    def count_vehicles_per_lane(
        self,
        detections: List[Dict],
        lane_boundaries: Optional[List[Tuple[float, float]]] = None,
        direction: str = "Unknown",
        method: str = "weighted_anchor"  # 'weighted_anchor', 'dbscan', 'original'
    ) -> Dict:
        """
        车道级别计数主函数
        
        Args:
            detections: VLM Grounding 检测结果
            lane_boundaries: 可选的车道边界线
            direction: 方向名称
            method: 检测方法 ('weighted_anchor', 'dbscan', 'original')
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
        corrected_centers = self.correct_perspective(centers)
        
        # Implementation note.
        if lane_boundaries:
            # Implementation note.
            lanes = self._assign_with_boundaries(corrected_centers, lane_boundaries)
            detection_method = 'lane_boundaries'
            lane_count = len(lane_boundaries)
        elif method == 'dbscan':
            # Implementation note.
            lane_structure = self.detect_lanes_by_dbscan(corrected_centers)
            lanes = self.assign_vehicles_to_lanes(corrected_centers, lane_structure['lane_centers'])
            detection_method = lane_structure['method']
            lane_count = lane_structure['lane_count']
        else:  # weighted_anchor
            # Implementation note.
            lane_structure = self.detect_lanes_by_weighted_anchor(corrected_centers)
            lanes = self.assign_vehicles_to_lanes(corrected_centers, lane_structure['lane_centers'])
            detection_method = lane_structure['method']
            lane_count = lane_structure['lane_count']
        
        # Implementation note.
        per_lane_count = [len(lanes[i]) for i in sorted(lanes.keys())]
        
        return {
            'total_vehicles': len(centers),
            'lane_count': lane_count,
            'per_lane_count': per_lane_count,
            'lanes': lanes,
            'method': detection_method,
            'direction': direction
        }
    
    def _assign_with_boundaries(
        self,
        centers: List[Dict],
        lane_boundaries: List[Tuple[float, float]]
    ) -> Dict[int, List[Dict]]:
        """使用车道边界线进行精确车道分配"""
        lanes = {i: [] for i in range(len(lane_boundaries))}
        
        for vehicle in centers:
            vx = vehicle['x']
            
            assigned = False
            for lane_id, (x_min, x_max) in enumerate(lane_boundaries):
                if x_min <= vx <= x_max:
                    lanes[lane_id].append(vehicle)
                    assigned = True
                    break
            
            if not assigned:
                distances = [min(abs(vx - x_min), abs(vx - x_max)) 
                           for x_min, x_max in lane_boundaries]
                lane_id = distances.index(min(distances))
                lanes[lane_id].append(vehicle)
        
        return lanes
    
    def format_lane_count_result(self, result: Dict) -> str:
        """格式化车道计数结果为文本描述"""
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
def count_vehicles_from_grounding_v2(
    vlm_response: str,
    lane_boundaries: Optional[List[Tuple[float, float]]] = None,
    direction: str = "Unknown",
    method: str = "weighted_anchor"
) -> Dict:
    """
    从 VLM Grounding 响应中计数车辆（优化版）
    
    Args:
        vlm_response: VLM 返回的文本
        lane_boundaries: 可选的车道边界
        direction: 方向名称
        method: 检测方法 ('weighted_anchor', 'dbscan')
    """
    counter = GroundingLaneCounterV2()
    detections = counter.parse_grounding_response(vlm_response)
    return counter.count_vehicles_per_lane(detections, lane_boundaries, direction, method)
