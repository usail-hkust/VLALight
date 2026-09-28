"""Distance Zone Segmentation Tool for Traffic Signal Control.

提取路口梯形区域后，按纵向切分为远端（上半部分）和近端（下半部分），
输出分区截图供 VLM 自主分析交通压力与来车趋势。
"""
import os
import cv2
from typing import Dict, List

from utils.lane_segmentation import LaneSegmentationTool


class DistanceZoneSegmentationTool:
    """路口远近端分区工具。"""

    def __init__(self,
                 direction: str,
                 scenario: str = 'jinan',
                 split_ratio: float = 0.4,
                 debug_mode: bool = False):
        """初始化分区工具。

        Args:
            direction: 摄像头方向 ('N', 'S', 'E', 'W')
            scenario: 场景 ('jinan', 'hangzhou', 'newyork')
            split_ratio: 上半部分占比，默认 0.4
            debug_mode: 是否保存中间调试图像
        """
        self.direction = direction
        self.scenario = scenario
        self.split_ratio = min(max(split_ratio, 0.2), 0.8)
        self.debug_mode = debug_mode

    def split_distance_zones(self,
                             direction_image_path: str = '',
                             lane_image_path: str = '',
                             source: str = 'direction',
                             output_dir: str = None) -> Dict:
        """提取梯形 ROI 并切分远端/近端。"""
        image_path = ''
        if source == 'lane':
            image_path = lane_image_path or direction_image_path
        else:
            image_path = direction_image_path or lane_image_path

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")

        debug_dir = None
        if self.debug_mode and output_dir:
            debug_dir = os.path.join(output_dir, 'debug')
            os.makedirs(debug_dir, exist_ok=True)

        lane_tool = LaneSegmentationTool(
            direction=self.direction,
            scenario=self.scenario,
            debug_mode=self.debug_mode,
        )

        trapezoid_info = lane_tool._detect_trapezoid_region(image, debug_dir)
        trapezoid_points = trapezoid_info['points']
        roi = lane_tool._extract_trapezoid(image, trapezoid_points, debug_dir)

        height, width = roi.shape[:2]
        split_y = int(height * self.split_ratio)
        split_y = min(max(split_y, 1), height - 1)

        far_zone_raw = roi[:split_y, :].copy()
        near_zone_raw = roi[split_y:, :].copy()

        # Implementation note.
        target_w = width
        target_h = max(height // 2, 1)
        far_zone = self._resize_with_padding(far_zone_raw, target_w, target_h)
        near_zone = self._resize_with_padding(near_zone_raw, target_w, target_h)

        far_path = None
        near_path = None
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            far_path = os.path.join(output_dir, f"{self.direction}_zone_far.jpg")
            near_path = os.path.join(output_dir, f"{self.direction}_zone_near.jpg")
            cv2.imwrite(far_path, far_zone)
            cv2.imwrite(near_path, near_zone)

            if self.debug_mode:
                roi_path = os.path.join(output_dir, f"{self.direction}_roi_trapezoid.jpg")
                cv2.imwrite(roi_path, roi)

        zones: List[Dict] = [
            {
                'zone_index': 0,
                'zone_key': 'far',
                'zone_name': '路口远端',
                'zone_image_path': far_path,
            },
            {
                'zone_index': 1,
                'zone_key': 'near',
                'zone_name': '路口近端',
                'zone_image_path': near_path,
            },
        ]

        return {
            'direction': self.direction,
            'zones': zones,
            'metadata': {
                'source': source,
                'input_image_path': image_path,
                'roi_shape': (height, width),
                'split_ratio': self.split_ratio,
                'split_y': split_y,
                'far_raw_shape': far_zone_raw.shape[:2],
                'near_raw_shape': near_zone_raw.shape[:2],
                'normalized_zone_shape': (target_h, target_w),
                'trapezoid_points': trapezoid_points,
            }
        }

    @staticmethod
    def _resize_with_padding(image, target_w: int, target_h: int):
        """保持纵横比缩放后补边到目标尺寸。"""
        src_h, src_w = image.shape[:2]
        if src_h <= 0 or src_w <= 0:
            return image

        scale = min(target_w / src_w, target_h / src_h)
        new_w = max(int(src_w * scale), 1)
        new_h = max(int(src_h * scale), 1)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        canvas = cv2.copyMakeBorder(
            resized,
            top=(target_h - new_h) // 2,
            bottom=target_h - new_h - (target_h - new_h) // 2,
            left=(target_w - new_w) // 2,
            right=target_w - new_w - (target_w - new_w) // 2,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
        return canvas
