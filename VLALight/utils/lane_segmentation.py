"""Lane Segmentation Tool for Traffic Signal Control.

将交通摄像头图像分割为独立车道，并返回各车道图像及方向。
"""
import os
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional


class LaneSegmentationTool:
    """车道分割工具。
    将图像分割为独立车道，并标注每个车道的车流方向。
    """
    
    def __init__(self, direction: str, scenario: str = 'jinan', debug_mode: bool = False,
                 split_ratio: float = 0.3):
        """初始化车道分割工具。

        Args:
            direction: 摄像头方向 ('N', 'S', 'E', 'W')
            scenario: 场景 ('jinan', 'hangzhou', 'newyork')
            debug_mode: 是否保存中间调试图像
            split_ratio: 远端占比（0.3 = 远端30% / 近端70%）
        """
        self.direction = direction
        self.scenario = scenario
        self.debug_mode = debug_mode
        self.split_ratio = min(max(split_ratio, 0.2), 0.8)
        
        # Implementation note.
        self.lane_direction_map = {
            'N': {'left': '右转', 'center': '直行', 'right': '左转'},
            'S': {'left': '右转', 'center': '直行', 'right': '左转'},
            'E': {'left': '右转', 'center': '直行', 'right': '左转'},
            'W': {'left': '右转', 'center': '直行', 'right': '左转'},
        }
        # Implementation note.
        self._direction_en = {'右转': 'right_turn', '直行': 'straight', '左转': 'left_turn'}
        
        # Implementation note.
        self.white_lower = np.array([0, 0, 180])
        self.white_upper = np.array([180, 30, 255])
        self.yellow_lower = np.array([20, 50, 150])
        self.yellow_upper = np.array([40, 180, 255])
        self.orange_lower = np.array([10, 80, 160])
        self.orange_upper = np.array([40, 255, 255])

        # Current-frame geometry cache, populated during segment_lanes.
        self._current_roi_points = None
        self._current_y_bottom = None
    
    @staticmethod
    def _safe_imwrite(path: str, img: np.ndarray) -> bool:
        """cv2.imwrite 的中文路径安全版本。"""
        try:
            ext = os.path.splitext(path)[1] if os.path.splitext(path)[1] else '.jpg'
            success, buf = cv2.imencode(ext, img)
            if success:
                with open(path, 'wb') as f:
                    f.write(buf.tobytes())
                return True
        except Exception:
            pass
        return False

    @staticmethod
    def _crop_black_margins(image: np.ndarray, padding: int = 24, min_size: int = 96) -> np.ndarray:
        """轻量裁剪纯黑边，降低 token，但保留一定上下文。"""
        if image is None or image.size == 0:
            return image

        non_black = np.any(image > 0, axis=2)
        ys, xs = np.where(non_black)
        if len(xs) == 0 or len(ys) == 0:
            return image

        h, w = image.shape[:2]
        x1 = max(0, int(xs.min()) - padding)
        y1 = max(0, int(ys.min()) - padding)
        x2 = min(w, int(xs.max()) + 1 + padding)
        y2 = min(h, int(ys.max()) + 1 + padding)

        cropped = image[y1:y2, x1:x2]
        ch, cw = cropped.shape[:2]
        if ch < min_size or cw < min_size:
            # Implementation note.
            return image
        return cropped

    @staticmethod
    def _compute_vehicle_pixel_ratio(lane_img: np.ndarray) -> float:
        """Estimate vehicle pixel occupancy in a lane crop.

        Uses HSV exclusion (road + markings) with morphological
        open/close and contour filtering to build a clean vehicle mask,
        then returns vehicle_pixels / lane_pixels.  [0.0, 1.0].

        Approach adapted from congestion_analysis.py.
        """
        if lane_img is None or lane_img.size == 0:
            return 0.0
        non_black = np.any(lane_img > 10, axis=2)
        lane_area = int(non_black.sum())
        if lane_area == 0:
            return 0.0

        hsv = cv2.cvtColor(lane_img, cv2.COLOR_BGR2HSV)
        s, v = hsv[:, :, 1], hsv[:, :, 2]
        h_img, w_img = lane_img.shape[:2]

        # Road candidates: tight gray band + white markings
        road_cand = np.zeros((h_img, w_img), dtype=np.uint8)
        road_cand[((s < 35) & (v >= 75) & (v <= 105) & non_black) |
                  ((s < 30) & (v > 170) & non_black)] = 255

        # Erode road candidates to break thin road strips between vehicles
        k_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        road_eroded = cv2.morphologyEx(road_cand, cv2.MORPH_ERODE, k_erode, iterations=1)

        # Connected-component: only eroded road reachable from edges is real road
        num_labels, labels = cv2.connectedComponents(road_eroded)
        bottom_rows = max(1, h_img // 10)
        bottom_labels = set(np.unique(labels[-bottom_rows:, :]))
        bottom_labels.discard(0)
        edge_labels = set(np.unique(labels[:, :3]))
        edge_labels |= set(np.unique(labels[:, -3:]))
        edge_labels.discard(0)
        real_road_labels = bottom_labels | edge_labels

        # Dilate back to original road extent (undo erosion for connected regions)
        real_road_eroded = np.isin(labels, list(real_road_labels)).astype(np.uint8) * 255
        real_road = cv2.morphologyEx(real_road_eroded, cv2.MORPH_DILATE, k_erode, iterations=1)
        # Clip to original road candidates (don't expand beyond)
        real_road = cv2.bitwise_and(real_road, road_cand)

        # Vehicle = non-black minus real road
        vehicle_area = int(non_black.sum()) - int((real_road > 0).sum())
        return round(vehicle_area / lane_area, 3)

    @staticmethod
    def _polygon_mask(shape: Tuple[int, int], polygon: np.ndarray) -> np.ndarray:
        mask = np.zeros(shape, dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 255)
        return mask

    @staticmethod
    def _line_x_at_y(line: Tuple[int, int, int, int], y: float):
        x1, y1, x2, y2 = line
        if y1 == y2:
            return None
        t = (y - y1) / (y2 - y1)
        return x1 + t * (x2 - x1)

    @staticmethod
    def _interpolate_point(p1: Tuple[int, int], p2: Tuple[int, int], alpha: float) -> Tuple[int, int]:
        x = int(round(p1[0] + alpha * (p2[0] - p1[0])))
        y = int(round(p1[1] + alpha * (p2[1] - p1[1])))
        return x, y
    
    def segment_lanes(self, image_path: str, output_dir: str = None,
                       split_ratio: float = None) -> Dict:
        """分割图像中的车道，并按纵向切分为近/远两段。

        流程：黄线确定梯形区域 → 提取并放大 → 检测白色分割线 → 切分车道 → 纵向分段(近/远)。
        不再调用 DINO 进行车辆计数，由 VLM 从切分图像自行计数。
        """
        if split_ratio is not None:
            self.split_ratio = min(max(split_ratio, 0.2), 0.8)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Implementation note.
        try:
            with open(image_path, 'rb') as f:
                img_data = np.frombuffer(f.read(), np.uint8)
                image = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        except Exception as e:
            raise ValueError(f"Failed to read image: {image_path}, error: {e}")
        
        if image is None:
            raise ValueError(f"Failed to decode image: {image_path}")
        
        height, width = image.shape[:2]
        debug_dir = None
        if self.debug_mode and output_dir:
            debug_dir = os.path.join(output_dir, 'debug')
            os.makedirs(debug_dir, exist_ok=True)
        
        # Implementation note.
        trapezoid_info = self._detect_trapezoid_region(image, debug_dir)
        trapezoid_points = trapezoid_info['points']
        self._current_roi_points = trapezoid_points
        self._current_y_bottom = trapezoid_info.get('y_bottom', height - 1)
        
        # Implementation note.
        trapezoid_image = self._extract_trapezoid(image, trapezoid_points, debug_dir)
        
        # Implementation note.
        white_lines = self._detect_lane_dividers(trapezoid_image, debug_dir)
        
        # Implementation note.
        lane_images = self._split_into_lanes(trapezoid_image, white_lines, debug_dir)

        # Implementation note.
        lane_zones = self._split_lanes_into_zones(lane_images, output_dir, debug_dir)

        # Implementation note.
        lanes = []
        for i, lane_img in enumerate(lane_images):
            lane_key = ['left', 'center', 'right'][i] if i < 3 else f'extra_{i}'
            lane_direction = self.lane_direction_map[self.direction].get(lane_key, '未知')

            lane_image_path = None
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                dir_en = self._direction_en.get(lane_direction, f'lane{i}')
                lane_image_path = os.path.join(output_dir, f"{self.direction}_lane_{i}_{dir_en}.jpg")
                self._safe_imwrite(lane_image_path, lane_img)

            lane_entry = {
                'lane_id': i,
                'lane_direction': lane_direction,
                'lane_image': lane_img,
                'lane_image_path': lane_image_path,
                'zones': lane_zones[i],  # [{zone:'far',...}, {zone:'near',...}]
            }
            lanes.append(lane_entry)

        # Implementation note.
        result = {
            'direction': self.direction,
            'lanes': lanes,
            'trapezoid_image': trapezoid_image,
            'dividers': white_lines,
            'metadata': {
                'image_shape': (height, width),
                'lane_count': len(lanes),
                'trapezoid_points': trapezoid_points,
                'divider_count': len(white_lines),
                'split_ratio': self.split_ratio,
            }
        }
        
        return result
    
    
    def _detect_trapezoid_region(self, image: np.ndarray, debug_dir: str = None) -> Dict:
        """优先通过橙色进口框检测 ROI，失败时回退到旧的黄线检测。"""
        orange_info = self._detect_orange_box_region(image, debug_dir)
        if orange_info is not None:
            return orange_info

        return self._detect_trapezoid_region_yellow(image, debug_dir)

    def _detect_orange_box_region(self, image: np.ndarray, debug_dir: str = None):
        """通过橙色进口框检测完整进口道 ROI。"""
        height, width = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        orange_mask = cv2.inRange(hsv, self.orange_lower, self.orange_upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        edges = cv2.Canny(orange_mask, 50, 150)
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=50,
            minLineLength=max(60, width // 8),
            maxLineGap=30,
        )
        if lines is None:
            return None

        raw_lines = [tuple(map(int, line[0])) for line in lines]
        horizontal = []
        slanted = []
        for line in raw_lines:
            x1, y1, x2, y2 = line
            dx, dy = x2 - x1, y2 - y1
            length = float(np.hypot(dx, dy))
            if length < 40:
                continue
            if abs(dy) <= 8 and max(y1, y2) > height * 0.55:
                horizontal.append((length, line))
            else:
                angle = abs(np.degrees(np.arctan2(dy, dx)))
                if 35 <= angle <= 85:
                    slanted.append((length, line))

        if not horizontal:
            return None

        horizontal.sort(key=lambda x: (x[0], max(x[1][1], x[1][3])), reverse=True)
        bottom = horizontal[0][1]
        bx1, by1, bx2, by2 = bottom
        if bx1 > bx2:
            bx1, bx2 = bx2, bx1
            by1, by2 = by2, by1
        y_bottom = float((by1 + by2) / 2.0)

        left_candidates = []
        right_candidates = []
        for _, line in slanted:
            x_at_bottom = self._line_x_at_y(line, y_bottom)
            if x_at_bottom is None:
                continue
            if x_at_bottom < (bx1 + bx2) / 2:
                left_candidates.append((abs(x_at_bottom - bx1), line))
            else:
                right_candidates.append((abs(x_at_bottom - bx2), line))

        if not left_candidates or not right_candidates:
            return None

        left_line = min(left_candidates, key=lambda x: x[0])[1]
        right_line = min(right_candidates, key=lambda x: x[0])[1]

        left_top_x = self._line_x_at_y(left_line, 0.0)
        right_top_x = self._line_x_at_y(right_line, 0.0)
        left_bottom_x = self._line_x_at_y(left_line, y_bottom)
        right_bottom_x = self._line_x_at_y(right_line, y_bottom)

        if None in (left_top_x, right_top_x, left_bottom_x, right_bottom_x):
            return None

        trapezoid_points = [
            (int(left_top_x), 0),
            (int(right_top_x), 0),
            (int(right_bottom_x), int(y_bottom)),
            (int(left_bottom_x), int(y_bottom))
        ]

        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(trapezoid_points, dtype=np.int32)], 255)

        if debug_dir:
            vis = image.copy()
            cv2.polylines(vis, [np.array(trapezoid_points, dtype=np.int32)], True, (0, 255, 0), 3)
            self._safe_imwrite(os.path.join(debug_dir, 'step0_trapezoid.jpg'), vis)

        return {
            'points': trapezoid_points,
            'mask': mask,
            'y_bottom': int(y_bottom),
            'source': 'orange_box',
            'left_line': left_line,
            'right_line': right_line,
            'bottom_line': (int(bx1), int(y_bottom), int(bx2), int(y_bottom)),
        }

    def _detect_trapezoid_region_yellow(self, image: np.ndarray, debug_dir: str = None) -> Dict:
        """旧版：通过黄线检测梯形道路区域。"""
        height, width = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Implementation note.
        yellow_mask = cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
        
        # Implementation note.
        edges = cv2.Canny(yellow_mask, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=80, maxLineGap=30)
        
        trapezoid_points = None
        
        if lines is not None and len(lines) > 0:
            # Implementation note.
            clustered_lines = self._cluster_lines([line[0] for line in lines])
            vertical_lines = []
            
            for x1, y1, x2, y2 in clustered_lines:
                dx, dy = x2 - x1, y2 - y1
                angle = 90 if dx == 0 else abs(np.arctan(dy / dx) * 180 / np.pi)
                if angle > 60:
                    vertical_lines.append((x1, y1, x2, y2))
            
            if len(vertical_lines) >= 2:
                vertical_lines.sort(key=lambda l: min(l[0], l[2]))
                left_line, right_line = vertical_lines[0], vertical_lines[-1]
                
                # Implementation note.
                left_top_x = left_line[0] if left_line[1] < left_line[3] else left_line[2]
                left_bottom_x = left_line[0] if left_line[1] > left_line[3] else left_line[2]
                left_bottom_y = max(left_line[1], left_line[3])
                
                right_top_x = right_line[0] if right_line[1] < right_line[3] else right_line[2]
                right_bottom_x = right_line[0] if right_line[1] > right_line[3] else right_line[2]
                right_bottom_y = max(right_line[1], right_line[3])
                
                trapezoid_points = [
                    (int(left_top_x), 0),
                    (int(right_top_x), 0),
                    (int(right_bottom_x), int(right_bottom_y)),
                    (int(left_bottom_x), int(left_bottom_y))
                ]
        
        if trapezoid_points:
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(mask, [np.array(trapezoid_points, dtype=np.int32)], 255)
            
            if debug_dir:
                vis = image.copy()
                cv2.polylines(vis, [np.array(trapezoid_points, dtype=np.int32)], True, (0, 255, 0), 3)
                self._safe_imwrite(os.path.join(debug_dir, 'step0_trapezoid.jpg'), vis)
            
            return {'points': trapezoid_points, 'mask': mask, 'y_bottom': height - 1, 'source': 'yellow'}
        
        # Implementation note.
        return {
            'points': [(0, 0), (width, 0), (width, height), (0, height)],
            'mask': np.ones((height, width), dtype=np.uint8) * 255,
            'y_bottom': height - 1,
            'source': 'full',
        }
    
    def _cluster_lines(self, lines: List[Tuple], dist_thresh: int = 30, angle_thresh: float = 5.0) -> List[Tuple]:
        """聚类相似线段。"""
        if not lines:
            return []
        
        line_info = []
        for x1, y1, x2, y2 in lines:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            dx, dy = x2 - x1, y2 - y1
            angle = 90 if dx == 0 else abs(np.arctan(dy / dx) * 180 / np.pi)
            line_info.append({'line': (x1, y1, x2, y2), 'center': (mx, my), 'angle': angle, 'used': False})
        
        clusters = []
        for i, info in enumerate(line_info):
            if info['used']:
                continue
            
            cluster = [info['line']]
            info['used'] = True
            
            for j in range(i + 1, len(line_info)):
                if line_info[j]['used']:
                    continue
                
                dist = np.sqrt((info['center'][0] - line_info[j]['center'][0])**2 + 
                              (info['center'][1] - line_info[j]['center'][1])**2)
                angle_diff = abs(info['angle'] - line_info[j]['angle'])
                
                if dist < dist_thresh and angle_diff < angle_thresh:
                    cluster.append(line_info[j]['line'])
                    line_info[j]['used'] = True
            
            # Implementation note.
            avg = tuple(int(np.mean([l[k] for l in cluster])) for k in range(4))
            clusters.append(avg)
        
        return clusters
    
    def _extract_trapezoid(self, image: np.ndarray, points: List[Tuple], debug_dir: str = None) -> np.ndarray:
        """保留完整进口道区域，外部填黑，不再做 bbox 拉伸。"""
        height, width = image.shape[:2]
        
        # Implementation note.
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(points, dtype=np.int32)], 255)
        
        result = image.copy()
        result[mask == 0] = 0
        
        if debug_dir:
            self._safe_imwrite(os.path.join(debug_dir, 'step1_trapezoid_enlarged.jpg'), result)
        
        return result
    
    def _detect_lane_dividers(self, image: np.ndarray, debug_dir: str = None) -> List[Tuple]:
        """在 ROI 内检测白色虚线，并延伸到底部，失败时回退到三等分。"""
        height, width = image.shape[:2]
        if not self._current_roi_points:
            third = width // 3
            return [(third, 0, third, height-1), (2*third, 0, 2*third, height-1)]

        roi_poly = np.array(self._current_roi_points, dtype=np.int32)
        roi_mask = self._polygon_mask((height, width), roi_poly)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Implementation note.
        white_mask = cv2.inRange(hsv, self.white_lower, self.white_upper)
        white_mask = cv2.bitwise_and(white_mask, roi_mask)

        # Implementation note.
        search_bottom = max(0, int((self._current_y_bottom or (height - 1)) - 120))
        white_mask[search_bottom:, :] = 0

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 9))
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Implementation note.
        edges = cv2.Canny(white_mask, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=max(40, height // 12), maxLineGap=30)
        
        if lines is None or len(lines) == 0:
            # Implementation note.
            tl, tr, br, bl = self._current_roi_points
            top_1 = self._interpolate_point(tl, tr, 1 / 3)
            top_2 = self._interpolate_point(tl, tr, 2 / 3)
            bottom_1 = self._interpolate_point(bl, br, 1 / 3)
            bottom_2 = self._interpolate_point(bl, br, 2 / 3)
            return [
                (top_1[0], top_1[1], bottom_1[0], bottom_1[1]),
                (top_2[0], top_2[1], bottom_2[0], bottom_2[1]),
            ]

        tl, tr, br, bl = self._current_roi_points
        left_boundary = (tl[0], tl[1], bl[0], bl[1])
        right_boundary = (tr[0], tr[1], br[0], br[1])
        y_top = 0.0
        y_eval = float(search_bottom)
        y_bottom = float(self._current_y_bottom or (height - 1))

        left_top = self._line_x_at_y(left_boundary, y_top)
        right_top = self._line_x_at_y(right_boundary, y_top)
        left_eval = self._line_x_at_y(left_boundary, y_eval)
        right_eval = self._line_x_at_y(right_boundary, y_eval)
        if None in (left_top, right_top, left_eval, right_eval):
            third = width // 3
            return [(third, 0, third, height-1), (2*third, 0, 2*third, height-1)]

        expected = []
        for alpha in (1 / 3, 2 / 3):
            exp_top = left_top + alpha * (right_top - left_top)
            exp_eval = left_eval + alpha * (right_eval - left_eval)
            expected.append((int(round(exp_top)), 0, int(round(exp_eval)), int(round(y_eval))))

        candidates = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx, dy = x2 - x1, y2 - y1
            
            if abs(dy) < 20:
                continue
            angle = abs(np.degrees(np.arctan2(dy, dx)))
            if angle < 65:
                continue

            x_top = self._line_x_at_y((x1, y1, x2, y2), y_top)
            x_eval = self._line_x_at_y((x1, y1, x2, y2), y_eval)
            if x_top is None or x_eval is None:
                continue
            if not (left_eval + 10 < x_eval < right_eval - 10):
                continue

            candidates.append({
                'line': (int(round(x_top)), 0, int(round(x_eval)), int(round(y_eval))),
                'x_eval': x_eval,
                'length': float(np.hypot(dx, dy)),
            })

        result = []
        for exp in expected:
            exp_x = exp[2]
            close = [c for c in candidates if abs(c['x_eval'] - exp_x) < 80]
            if not close:
                xt = exp[0]
                xb = int(round(self._line_x_at_y((exp[0], exp[1], exp[2], exp[3]), y_bottom)))
                result.append((xt, 0, xb, int(round(y_bottom))))
                continue
            total_len = sum(c['length'] for c in close)
            avg_top = sum(c['line'][0] * c['length'] for c in close) / total_len
            avg_eval = sum(c['line'][2] * c['length'] for c in close) / total_len
            xb = self._line_x_at_y((avg_top, 0, avg_eval, y_eval), y_bottom)
            if xb is None:
                xb = avg_eval
            result.append((int(round(avg_top)), 0, int(round(xb)), int(round(y_bottom))))

        result.sort(key=lambda l: (l[0] + l[2]) / 2.0)
        
        if debug_dir:
            vis = image.copy()
            for i, (x1, y1, x2, y2) in enumerate(result):
                cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(vis, f"L{i+1}", ((x1+x2)//2, height//2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            self._safe_imwrite(os.path.join(debug_dir, 'step2_dividers.jpg'), vis)
        
        return result
    
    def _split_into_lanes(self, image: np.ndarray, dividers: List[Tuple], debug_dir: str = None) -> List[np.ndarray]:
        """按完整进口道 ROI 切分为 3 条道路块，不裁掉靠近路口的小梯形。"""
        height, width = image.shape[:2]

        if self._current_roi_points:
            tl, tr, br, bl = self._current_roi_points
        else:
            tl, tr, br, bl = (0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)

        if len(dividers) >= 2:
            line1 = dividers[0]
            line2 = dividers[1]
            top_1 = (line1[0], line1[1])
            bottom_1 = (line1[2], line1[3])
            top_2 = (line2[0], line2[1])
            bottom_2 = (line2[2], line2[3])
        else:
            top_1 = self._interpolate_point(tl, tr, 1 / 3)
            top_2 = self._interpolate_point(tl, tr, 2 / 3)
            bottom_1 = self._interpolate_point(bl, br, 1 / 3)
            bottom_2 = self._interpolate_point(bl, br, 2 / 3)

        polygons = [
            np.array([tl, top_1, bottom_1, bl], dtype=np.int32),
            np.array([top_1, top_2, bottom_2, bottom_1], dtype=np.int32),
            np.array([top_2, tr, br, bottom_2], dtype=np.int32),
        ]
        
        lane_images = []
        for i, poly in enumerate(polygons):
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(mask, [poly], 255)
            
            lane_img = image.copy()
            lane_img[mask == 0] = 0
            lane_img = self._crop_black_margins(lane_img)
            lane_images.append(lane_img)
            
            if debug_dir:
                self._safe_imwrite(os.path.join(debug_dir, f'step3_lane_{i}.jpg'), lane_img)
        
        return lane_images

    @staticmethod
    def _add_zone_label(image: np.ndarray, text: str) -> np.ndarray:
        """在图像右上角添加半透明标签。

        Args:
            image: 输入图像
            text: 标签文本，如 "E-right_turn far"

        Returns:
            添加标签后的图像副本
        """
        if image is None or image.size == 0:
            return image

        img = image.copy()
        h, w = img.shape[:2]

        # Implementation note.
        font_scale = max(0.5, min(h, w) / 400)
        thickness = max(1, int(font_scale * 1.5))
        font = cv2.FONT_HERSHEY_SIMPLEX

        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        # Implementation note.
        pad = 4
        x1 = w - tw - pad * 2 - baseline
        y1 = 0
        x2 = w
        y2 = th + pad * 2 + baseline

        # Implementation note.
        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        # Implementation note.
        text_x = x1 + pad
        text_y = y2 - pad - baseline // 2
        cv2.putText(img, text, (text_x, text_y), font, font_scale,
                    (255, 255, 255), thickness, cv2.LINE_AA)

        return img

    def _split_lanes_into_zones(self, lane_images: List[np.ndarray],
                                 output_dir: str = None,
                                 debug_dir: str = None) -> List[List[Dict]]:
        """对每条 lane 纵向切分为近/远两段，远端放大到与近端等高。

        Args:
            lane_images: 3 条 lane crop 图像
            output_dir: 保存 zone 图像的目录
            debug_dir: 调试图像目录

        Returns:
            List[List[Dict]]: 每条 lane 对应 [{zone:'far', ...}, {zone:'near', ...}]
        """
        all_lane_zones = []

        for lane_idx, lane_img in enumerate(lane_images):
            if lane_img is None or lane_img.size == 0:
                all_lane_zones.append([
                    {'zone': 'far', 'zone_image': None, 'zone_image_path': None},
                    {'zone': 'near', 'zone_image': None, 'zone_image_path': None},
                ])
                continue

            h, w = lane_img.shape[:2]
            split_y = int(h * self.split_ratio)
            split_y = min(max(split_y, 1), h - 1)

            # Implementation note.
            far_raw = lane_img[:split_y, :].copy()
            # Implementation note.
            near_raw = lane_img[split_y:, :].copy()

            # Implementation note.
            far_raw = self._crop_black_margins(far_raw)
            near_raw = self._crop_black_margins(near_raw)

            # Implementation note.
            near_h = near_raw.shape[0] if near_raw.size > 0 else 1
            far_h = far_raw.shape[0] if far_raw.size > 0 else 1
            far_w = far_raw.shape[1] if far_raw.size > 0 else 1
            if far_raw.size > 0 and far_h > 0 and near_h > far_h:
                # Implementation note.
                scale = near_h / far_h
                new_w = int(round(far_w * scale))
                far_upscaled = cv2.resize(far_raw, (new_w, near_h),
                                          interpolation=cv2.INTER_LANCZOS4)
            else:
                far_upscaled = far_raw

            # Implementation note.
            # Implementation note.
            lane_functions = ['right-turn', 'straight', 'left-turn']
            lane_key = lane_functions[lane_idx] if lane_idx < 3 else f'lane{lane_idx}'
            label_text = f"{self.direction} {lane_key}"

            # Implementation note.
            far_labeled = self._add_zone_label(far_upscaled, f"{label_text} far")
            near_labeled = self._add_zone_label(near_raw, f"{label_text} near")

            # Implementation note.
            far_path = None
            near_path = None
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                far_path = os.path.join(output_dir,
                                        f"{self.direction}_lane{lane_idx}_far.jpg")
                near_path = os.path.join(output_dir,
                                         f"{self.direction}_lane{lane_idx}_near.jpg")
                self._safe_imwrite(far_path, far_labeled)
                self._safe_imwrite(near_path, near_labeled)

            if debug_dir:
                self._safe_imwrite(
                    os.path.join(debug_dir, f'step4_lane{lane_idx}_far.jpg'), far_labeled)
                self._safe_imwrite(
                    os.path.join(debug_dir, f'step4_lane{lane_idx}_near.jpg'), near_labeled)

            zones = [
                {
                    'zone': 'far',
                    'zone_image': far_upscaled,
                    'zone_image_path': far_path,
                    'raw_shape': (far_h, far_raw.shape[1]) if far_raw.size > 0 else (0, 0),
                    'upscaled_shape': far_upscaled.shape[:2] if far_upscaled.size > 0 else (0, 0),
                },
                {
                    'zone': 'near',
                    'zone_image': near_raw,
                    'zone_image_path': near_path,
                    'raw_shape': near_raw.shape[:2] if near_raw.size > 0 else (0, 0),
                    'upscaled_shape': None,  # near zone not upscaled
                },
            ]
            all_lane_zones.append(zones)

        return all_lane_zones


def segment_lanes(image_path: str, direction: str, scenario: str = 'jinan',
                  output_dir: str = None, debug_mode: bool = False) -> List[Dict]:
    """Agent接口：分割指定方向的车道图像。"""
    tool = LaneSegmentationTool(direction=direction, scenario=scenario, debug_mode=debug_mode)
    result = tool.segment_lanes(image_path, output_dir)
    return result['lanes']


if __name__ == '__main__':
    print("Lane Segmentation Tool")
    print("=" * 50)
    print("使用方法：")
    print("  from utils.lane_segmentation import segment_lanes")
    print("  lanes = segment_lanes(image_path, direction='E')")
    print("  ")
    print("返回值：")
    print("  [{'lane_id': 0, 'lane_direction': '右转', 'lane_image': ...},")
    print("   {'lane_id': 1, 'lane_direction': '直行', 'lane_image': ...},")
    print("   {'lane_id': 2, 'lane_direction': '左转', 'lane_image': ...}]")
