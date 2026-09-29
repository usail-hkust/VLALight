"""
Traffic Difference Comparison Tool
对比相邻两轮的交通检测结果，可视化车辆的新增和消失。
"""

import os
import cv2
import numpy as np
from typing import Dict, List, Tuple, Any, Optional


class TrafficDiffTool:
    """
    交通差异对比工具
    对比两轮检测结果，标注新增和消失的车辆。
    """
    
    # Implementation note.
    COLOR_NEW = (0, 255, 0)      # Implementation note.
    COLOR_GONE = (0, 0, 255)     # Implementation note.
    COLOR_KEPT = (0, 255, 255)   # Implementation note.
    
    def __init__(self):
        """初始化差异对比工具"""
        pass
    
    def calculate_iou(self, box1: List[int], box2: List[int]) -> float:
        """
        计算两个边界框的 IoU（交并比）
        
        Args:
            box1: [x1, y1, x2, y2]
            box2: [x1, y1, x2, y2]
            
        Returns:
            IoU 值 (0-1)
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)
        
        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0
        
        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def match_vehicles(
        self, 
        prev_detections: List[Dict], 
        curr_detections: List[Dict],
        iou_threshold: float = 0.3
    ) -> Tuple[set, set]:
        """
        匹配两轮之间的车辆（基于 IoU）
        
        Args:
            prev_detections: 上一轮检测结果
            curr_detections: 本轮检测结果
            iou_threshold: IoU 阈值
            
        Returns:
            (matched_prev_indices, matched_curr_indices)
        """
        matched_prev = set()
        matched_curr = set()
        
        for i, prev_det in enumerate(prev_detections):
            for j, curr_det in enumerate(curr_detections):
                iou = self.calculate_iou(prev_det['bbox'], curr_det['bbox'])
                if iou > iou_threshold:
                    matched_prev.add(i)
                    matched_curr.add(j)
                    break
        
        return matched_prev, matched_curr
    
    def classify_changes(
        self,
        prev_detections: List[Dict],
        curr_detections: List[Dict],
        matched_prev: set,
        matched_curr: set
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        分类车辆变化
        
        Returns:
            (new_vehicles, gone_vehicles, kept_vehicles)
        """
        new_vehicles = [
            curr_detections[j] for j in range(len(curr_detections))
            if j not in matched_curr
        ]
        
        gone_vehicles = [
            prev_detections[i] for i in range(len(prev_detections))
            if i not in matched_prev
        ]
        
        kept_vehicles = [
            curr_detections[j] for j in range(len(curr_detections))
            if j in matched_curr
        ]
        
        return new_vehicles, gone_vehicles, kept_vehicles
    
    def draw_dashed_box(
        self,
        image: np.ndarray,
        bbox: List[int],
        color: Tuple[int, int, int],
        thickness: int = 2,
        dash_length: int = 10
    ):
        """绘制虚线边界框"""
        x1, y1, x2, y2 = map(int, bbox)
        
        for x in range(x1, x2, dash_length * 2):
            cv2.line(image, (x, y1), (min(x + dash_length, x2), y1), color, thickness)
        for x in range(x1, x2, dash_length * 2):
            cv2.line(image, (x, y2), (min(x + dash_length, x2), y2), color, thickness)
        for y in range(y1, y2, dash_length * 2):
            cv2.line(image, (x1, y), (x1, min(y + dash_length, y2)), color, thickness)
        for y in range(y1, y2, dash_length * 2):
            cv2.line(image, (x2, y), (x2, min(y + dash_length, y2)), color, thickness)
    
    def visualize_diff(
        self,
        curr_image: np.ndarray,
        new_vehicles: List[Dict],
        gone_vehicles: List[Dict],
        kept_vehicles: List[Dict],
        prev_total: int,
        curr_total: int
    ) -> np.ndarray:
        """在本轮图像上可视化差异"""
        result = curr_image.copy()
        
        # Implementation note.
        for det in gone_vehicles:
            self.draw_dashed_box(result, det['bbox'], self.COLOR_GONE, thickness=3)
            x1, y1, x2, y2 = map(int, det['bbox'])
            label_pos = (x1, y1 - 10 if y1 > 30 else y2 + 20)
            cv2.putText(result, "GONE", label_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(result, "GONE", label_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLOR_GONE, 2, cv2.LINE_AA)
        
        # Implementation note.
        for det in kept_vehicles:
            x1, y1, x2, y2 = map(int, det['bbox'])
            cv2.rectangle(result, (x1, y1), (x2, y2), self.COLOR_KEPT, 3)
        
        # Implementation note.
        for det in new_vehicles:
            x1, y1, x2, y2 = map(int, det['bbox'])
            cv2.rectangle(result, (x1, y1), (x2, y2), self.COLOR_NEW, 3)
            label_pos = (x1, y1 - 10 if y1 > 30 else y2 + 20)
            cv2.putText(result, "NEW", label_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(result, "NEW", label_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLOR_NEW, 2, cv2.LINE_AA)
        
        # Implementation note.
        h, w = result.shape[:2]
        total_change = curr_total - prev_total
        
        stats_lines = [
            "=== Traffic Changes ===",
            f"Total: {prev_total} -> {curr_total} ({total_change:+d})",
            "",
            f"Green Box: New vehicles ({len(new_vehicles)})",
            f"Red Dash: Gone vehicles ({len(gone_vehicles)})",
            f"Cyan Box: Kept vehicles ({len(kept_vehicles)})"
        ]
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        line_height = 25
        padding = 10
        
        max_width = 0
        for line in stats_lines:
            (tw, th), _ = cv2.getTextSize(line, font, font_scale, thickness)
            max_width = max(max_width, tw)
        
        panel_width = max_width + padding * 2
        panel_height = len(stats_lines) * line_height + padding * 2
        
        # Implementation note.
        panel_x = w - panel_width - 10
        panel_y = 10
        
        overlay = result.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height),
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, result, 0.3, 0, result)
        
        cv2.rectangle(result, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height),
                     (255, 255, 255), 2)
        
        y_offset = panel_y + padding + 15
        for line in stats_lines:
            if line:
                cv2.putText(result, line, (panel_x + padding, y_offset),
                           font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            y_offset += line_height
        
        return result
    
    def load_image_with_chinese_path(self, image_path: str) -> np.ndarray:
        """加载图像（支持中文路径）"""
        try:
            with open(image_path, 'rb') as f:
                image_data = np.frombuffer(f.read(), np.uint8)
                image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Failed to decode image")
            return image
        except Exception as e:
            raise FileNotFoundError(f"Cannot load image: {image_path}, Error: {e}")
    
    def compare_traffic(
        self,
        curr_image_path: str,
        prev_step: int,
        curr_step: int,
        output_dir: str,
        iou_threshold: float = 0.3,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ) -> Dict[str, Any]:
        """
        对比两轮交通检测结果并生成可视化图像。
        内部自动调用 VehicleDetectionTool（复用全局 DINO 缓存），
        无需 VLM 手动传入 detections。

        prev_image_path 由 curr_image_path 推导：
            将路径中的 step_{curr:04d} 替换为 step_{prev:04d}

        Args:
            curr_image_path: 当前步的单方向图像路径（由 context_lookup 自动注入）
            prev_step: 上一步步号
            curr_step: 当前步步号
            output_dir: 输出目录（由 tool_executor 的 output_dir 机制提供）
            iou_threshold: IoU 匹配阈值
            box_threshold: DINO 检测置信度阈值
            text_threshold: DINO 文本相似度阈值

        Returns:
            差异统计结果字典
        """
        from utils.vehicle_detection import VehicleDetectionTool, _draw_lane_overlay

        # Implementation note.
        prev_image_path = curr_image_path.replace(
            f"step_{curr_step:04d}", f"step_{prev_step:04d}"
        )
        if not os.path.exists(prev_image_path):
            return {
                "success": False,
                "summary": (
                    f"Previous step image not found: {prev_image_path}. "
                    f"Cannot compute diff between step {prev_step} and {curr_step}."
                ),
            }

        # Implementation note.
        detector = VehicleDetectionTool()

        prev_result = detector.detect_vehicles(
            prev_image_path,
            draw_boxes=False,
            output_dir=output_dir,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        curr_result = detector.detect_vehicles(
            curr_image_path,
            draw_boxes=False,
            output_dir=output_dir,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

        prev_detections = prev_result.get('detections', [])
        curr_detections = curr_result.get('detections', [])

        # Implementation note.
        curr_image_raw = self.load_image_with_chinese_path(curr_image_path)
        curr_image = _draw_lane_overlay(curr_image_raw.copy())
        output_path = os.path.join(output_dir, "traffic_diff.jpg")
        
        # Implementation note.
        prev_total = len(prev_detections)
        curr_total = len(curr_detections)
        
        # Implementation note.
        matched_prev, matched_curr = self.match_vehicles(
            prev_detections, curr_detections, iou_threshold
        )
        
        # Implementation note.
        new_vehicles, gone_vehicles, kept_vehicles = self.classify_changes(
            prev_detections, curr_detections, matched_prev, matched_curr
        )
        
        # Implementation note.
        diff_image = self.visualize_diff(
            curr_image, new_vehicles, gone_vehicles, kept_vehicles,
            prev_total, curr_total
        )
        
        # Implementation note.
        os.makedirs(output_dir, exist_ok=True)
        success, buf = cv2.imencode('.jpg', diff_image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        with open(output_path, 'wb') as f:
            f.write(buf.tobytes())
        
        # Implementation note.
        # Implementation note.
        result = {
            "success": True,
            "total_change": curr_total - prev_total,
            "prev_total": prev_total,
            "curr_total": curr_total,
            "new_count": len(new_vehicles),
            "gone_count": len(gone_vehicles),
            "kept_count": len(kept_vehicles),
            "visualization_image_path": output_path,
            "summary": (
                f"Step diff: {prev_total} → {curr_total} vehicles "
                f"(+{len(new_vehicles)} new, -{len(gone_vehicles)} gone, "
                f"{len(kept_vehicles)} unchanged). "
                f"Net change: {curr_total - prev_total:+d}."
            ),
        }
        
        return result
