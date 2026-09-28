"""Congestion Analysis Tool for Traffic Signal Control.

基于颜色分割统计车辆像素占道路面积比，输出各方向拥堵度排名
和方向组比较（N-S vs W-E），为 VLM 的相位决策提供数值辅助。

返回纯文本摘要，不生成图像。
"""
import os
import cv2
import numpy as np
from typing import Dict, List


class CongestionAnalysisTool:
    """路口拥堵度分析工具。"""

    def __init__(self, scenario: str = 'jinan'):
        self.scenario = scenario

    # ------------------------------------------------------------------ #
    #  Public API (called by ToolExecutor)                                #
    # ------------------------------------------------------------------ #

    def analyze(self, image_paths_dict: Dict[str, str],
                output_dir: str = None) -> Dict:
        """分析路口 4 个方向的拥堵度。

        Args:
            image_paths_dict: {direction: image_path} e.g. {'N': '/path/N.jpg', ...}
            output_dir: 输出目录（本工具不生成图像，仅用于兼容）

        Returns:
            dict with keys:
              - directions: per-direction results
              - pair_comparison: N-S vs W-E group totals
              - ranking: directions sorted by density desc
              - text_report: human-readable summary for VLM
        """
        dir_names = {'N': 'North', 'S': 'South', 'E': 'East', 'W': 'West'}
        directions = {}

        for d in ['N', 'S', 'E', 'W']:
            path = image_paths_dict.get(d, '')
            if not path or not os.path.exists(path):
                continue
            img = cv2.imread(path)
            if img is None:
                continue
            directions[d] = self._analyze_single(img)

        if not directions:
            return {
                'directions': {},
                'pair_comparison': {},
                'ranking': [],
                'text_report': 'No direction images found.',
            }

        # Ranking by density
        ranking = sorted(directions.keys(),
                         key=lambda d: directions[d]['density'], reverse=True)

        # Pair comparison
        ns_density = sum(directions.get(d, {}).get('density', 0) for d in ['N', 'S'])
        we_density = sum(directions.get(d, {}).get('density', 0) for d in ['E', 'W'])
        ns_regions = sum(directions.get(d, {}).get('region_count', 0) for d in ['N', 'S'])
        we_regions = sum(directions.get(d, {}).get('region_count', 0) for d in ['E', 'W'])

        pair_comparison = {
            'NS_density': ns_density,
            'WE_density': we_density,
            'NS_regions': ns_regions,
            'WE_regions': we_regions,
            'busier_pair': 'N-S' if ns_density >= we_density else 'W-E',
        }

        # Text report
        text_report = self._build_text_report(
            directions, dir_names, ranking, pair_comparison)

        return {
            'directions': directions,
            'pair_comparison': pair_comparison,
            'ranking': ranking,
            'text_report': text_report,
        }

    # ------------------------------------------------------------------ #
    #  Internal                                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _analyze_single(img: np.ndarray) -> dict:
        """分析单个方向的拥堵度。"""
        vehicle_mask = CongestionAnalysisTool._extract_vehicle_mask(img)
        road_mask = CongestionAnalysisTool._extract_road_mask(img, vehicle_mask)

        total_vehicle = int(np.sum(vehicle_mask > 0))
        total_road = int(np.sum(road_mask > 0))

        density = total_vehicle / total_road if total_road > 0 else 0.0

        contours, _ = cv2.findContours(
            vehicle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        region_count = len([c for c in contours if cv2.contourArea(c) > 800])

        if density < 0.05:
            level = 'Low'
        elif density < 0.15:
            level = 'Medium'
        else:
            level = 'High'

        return {
            'density': round(density, 4),
            'level': level,
            'region_count': region_count,
        }

    @staticmethod
    def _extract_vehicle_mask(img: np.ndarray) -> np.ndarray:
        """HSV 排除法提取车辆像素 mask。"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        road = cv2.inRange(hsv, (0, 0, 60), (180, 50, 160))
        grass = cv2.inRange(hsv, (30, 40, 60), (85, 255, 255))
        marking = cv2.inRange(hsv, (0, 0, 200), (180, 40, 255))
        non_vehicle = cv2.bitwise_or(road, cv2.bitwise_or(grass, marking))
        mask = cv2.bitwise_not(non_vehicle)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        clean = np.zeros_like(mask)
        for c in contours:
            if cv2.contourArea(c) > 500:
                cv2.drawContours(clean, [c], -1, 255, -1)
        return clean

    @staticmethod
    def _extract_road_mask(img: np.ndarray, vehicle_mask: np.ndarray) -> np.ndarray:
        """提取道路区域 mask。"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        road = cv2.inRange(hsv, (0, 0, 40), (180, 60, 180))
        marking = cv2.inRange(hsv, (0, 0, 200), (180, 40, 255))
        road = cv2.bitwise_or(road, cv2.bitwise_or(marking, vehicle_mask))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        road = cv2.morphologyEx(road, cv2.MORPH_CLOSE, kernel, iterations=3)
        return road

    @staticmethod
    def _build_text_report(directions: dict, dir_names: dict,
                           ranking: list, pair_comparison: dict) -> str:
        """构建纯文本拥堵度报告。"""
        lines = ['=== Congestion Analysis (pixel-based) ===', '']

        # Per-direction ranking
        lines.append('Direction Ranking (by vehicle density on road):')
        for rank, d in enumerate(ranking, 1):
            r = directions[d]
            lines.append(
                f"  {rank}. {dir_names[d]:5s}: {r['level']:6s} | "
                f"density={r['density']:5.1%} | "
                f"~{r['region_count']} vehicle regions"
            )
        lines.append('')

        # Pair comparison
        pc = pair_comparison
        lines.append('Direction Pair Comparison:')
        lines.append(
            f"  N-S Group: density={pc['NS_density']:.1%}, "
            f"~{pc['NS_regions']} regions")
        lines.append(
            f"  W-E Group: density={pc['WE_density']:.1%}, "
            f"~{pc['WE_regions']} regions")
        lines.append(f"  Busier pair: {pc['busier_pair']}")

        return '\n'.join(lines)
