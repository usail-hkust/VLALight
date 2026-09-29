"""Queue length estimation tool for single-lane traffic images.

This tool is designed to be called after lane segmentation. It estimates the
farthest queued vehicle distance from the stop-line on one lane image.

Core idea:
1. Extract a lane mask (non-black region in lane crop).
2. Extract a vehicle mask using HSV exclusion of road-like colors.
3. Build a longitudinal axis (PCA projection on lane mask by default).
4. Measure queue length from stop-line to queue tail.
   - Optional slicing continuity to avoid outlier pixels.
"""

import os
from typing import Dict, Tuple, List

import cv2
import numpy as np


class QueueLengthEstimationTool:
    """Estimate queue length on a single lane image."""

    def __init__(self, direction: str, lane_direction: str, scenario: str = "jinan"):
        self.direction = direction
        self.lane_direction = self._normalize_lane_direction(lane_direction)
        self.scenario = scenario

    def estimate_queue_length(
        self,
        source: str = "lane",
        lane_image_from_context: str = "",
        lane_image_path: str = "",
        lane_index: int = 0,
        reference_line: str = "center",
        use_projection: bool = True,
        use_slicing: bool = True,
        slice_bin_size_px: float = 8,
        occupancy_ratio_threshold: float = 0.015,
        max_gap_bins: int = 2,
        output_dir: str = "",
    ) -> Dict:
        """Estimate queue length in pixels.

        Args:
            source: "lane" to use context lane image, "path" to use lane_image_path.
            lane_image_from_context: lane image path from tool context (latest lane crops).
            lane_image_path: explicit lane image path.
            lane_index: lane index for metadata.
            reference_line: lane reference line used for queue tracing: center/left/right.
            use_projection: legacy switch for PCA projection mode (only used when reference_line='pca').
            use_slicing: if True, apply longitudinal occupancy slicing continuity.
            slice_bin_size_px: bin size for slicing.
            occupancy_ratio_threshold: occupied-bin threshold.
            max_gap_bins: allowed tiny gaps when tracing queue from stop-line.
            output_dir: optional output directory for debug visualization.
        """
        img_path = self._select_image_path(source, lane_image_from_context, lane_image_path)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Lane image not found: {img_path}")

        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Failed to read image: {img_path}")

        lane_mask = self._extract_lane_mask(image)
        vehicle_mask = self._extract_vehicle_mask(image, lane_mask)

        lane_points = np.column_stack(np.where(lane_mask > 0))

        if lane_points.shape[0] < 50:
            return self._empty_result("Lane mask too small.", img_path, lane_index)

        reference_line = (reference_line or "center").strip().lower()

        if reference_line in ("center", "left", "right"):
            ref = self._build_row_reference_profile(lane_mask, reference_line)
            stopline_point = ref["stopline_point"]
            axis_unit = np.array([0.0, 1.0], dtype=np.float32)
            axis_mode = f"row_reference_{reference_line}"
            lane_dist = self._distance_from_stopline_row(lane_points, int(stopline_point[1]))
            vehicle_points = np.column_stack(np.where(vehicle_mask > 0))
            if vehicle_points.shape[0] < 10:
                return self._empty_result("No clear vehicle pixels found.", img_path, lane_index)
            vehicle_dist = self._distance_from_stopline_row(vehicle_points, int(stopline_point[1]))
            component_intervals = self._extract_component_intervals_row(
                vehicle_mask=vehicle_mask,
                ref_x_by_y=ref["x_by_y"],
                lane_width_by_y=ref["lane_width_by_y"],
                stop_y=int(stopline_point[1]),
                reference_line=reference_line,
            )
            reference_polyline = ref["polyline"]
        else:
            axis_unit, stopline_point, axis_mode = self._build_lane_axis(lane_points, use_projection)

            vehicle_points = np.column_stack(np.where(vehicle_mask > 0))
            if vehicle_points.shape[0] < 10:
                return self._empty_result("No clear vehicle pixels found.", img_path, lane_index)

            lane_dist = self._distance_from_stopline(lane_points, stopline_point, axis_unit)
            vehicle_dist = self._distance_from_stopline(vehicle_points, stopline_point, axis_unit)

            component_intervals = self._extract_component_intervals(
                vehicle_mask=vehicle_mask,
                stopline_xy=stopline_point,
                axis_xy=axis_unit,
            )
            reference_polyline = None

        lane_max = float(np.max(lane_dist)) if lane_dist.size else 0.0
        raw_tail = float(np.max(vehicle_dist)) if vehicle_dist.size else 0.0

        queue_mode = "raw_farthest"
        queue_length = raw_tail
        occupancy_profile = None

        if use_slicing and lane_max > 0:
            queue_mode = "slicing"
            queue_length, occupancy_profile = self._estimate_with_slices(
                lane_dist=lane_dist,
                vehicle_dist=vehicle_dist,
                lane_max=lane_max,
                component_intervals=component_intervals,
                bin_size=max(float(slice_bin_size_px), 2.0),
                occ_threshold=max(float(occupancy_ratio_threshold), 0.001),
                max_gap_bins=max(int(max_gap_bins), 0),
                continuity_bridge_px=55.0 if self.lane_direction in ("left_turn", "right_turn") else 70.0,
            )

            continuity_tail = 0.0
            slice_tail = 0.0
            if occupancy_profile:
                continuity_tail = float(occupancy_profile.get("continuity_tail_px", 0.0))
                slice_tail = float(occupancy_profile.get("slice_tail_px", 0.0))

            # Guard against over-extended tails on turn lanes where slice occupancy
            # can be polluted by elongated artifacts.
            if self.lane_direction in ("left_turn", "right_turn"):
                if continuity_tail > 0 and slice_tail > continuity_tail * 1.35:
                    queue_length = continuity_tail
                    queue_mode = "slicing_with_continuity_guard"

            # Generic fallback: when slice tail is far beyond continuity evidence and
            # components are sparse, prefer continuity tail.
            if continuity_tail > 0 and slice_tail > continuity_tail * 1.8 and len(component_intervals) <= 4:
                queue_length = continuity_tail
                queue_mode = "slicing_with_sparse_component_guard"

            queue_length = min(queue_length, raw_tail if raw_tail > 0 else queue_length)

        debug_path = None
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            debug_path = os.path.join(output_dir, f"{self.direction}_lane_{lane_index}_queue_debug.jpg")
            self._draw_debug_visualization(
                image=image,
                lane_mask=lane_mask,
                vehicle_mask=vehicle_mask,
                stopline_point=stopline_point,
                axis_unit=axis_unit,
                queue_length=queue_length,
                reference_polyline=reference_polyline,
                out_path=debug_path,
            )

        text_report = self._build_text_report(
            queue_length=queue_length,
            raw_tail=raw_tail,
            lane_max=lane_max,
            queue_mode=queue_mode,
            axis_mode=axis_mode,
        )

        return {
            "queue_length_px": round(float(queue_length), 2),
            "queue_tail_distance_raw_px": round(float(raw_tail), 2),
            "queue_tail_mode": queue_mode,
            "lane_axis": {
                "mode": axis_mode,
                "unit_vector_xy": [round(float(axis_unit[0]), 5), round(float(axis_unit[1]), 5)],
                "stopline_point_xy": [int(stopline_point[0]), int(stopline_point[1])],
            },
            "metadata": {
                "direction": self.direction,
                "lane_direction": self.lane_direction,
                "lane_index": int(lane_index),
                "source": source,
                "input_image_path": img_path,
                "lane_pixels": int(lane_points.shape[0]),
                "vehicle_pixels": int(vehicle_points.shape[0]),
                "lane_axis_length_px": round(float(lane_max), 2),
                "use_projection": bool(use_projection),
                "reference_line": reference_line,
                "use_slicing": bool(use_slicing),
                "slice_bin_size_px": round(float(max(slice_bin_size_px, 2)), 2),
                "occupancy_ratio_threshold": float(max(occupancy_ratio_threshold, 0.001)),
                "max_gap_bins": int(max(max_gap_bins, 0)),
                "debug_image_path": debug_path,
                "occupancy_profile": occupancy_profile,
                "component_count": len(component_intervals),
            },
            "text_report": text_report,
        }

    @staticmethod
    def _select_image_path(source: str, lane_image_from_context: str, lane_image_path: str) -> str:
        if source == "path":
            return lane_image_path or lane_image_from_context
        return lane_image_from_context or lane_image_path

    @staticmethod
    def _normalize_lane_direction(value: str) -> str:
        mapping = {
            "right_turn": "right_turn",
            "straight": "straight",
            "left_turn": "left_turn",
            "右转": "right_turn",
            "直行": "straight",
            "左转": "left_turn",
        }
        return mapping.get((value or "").strip(), "straight")

    @staticmethod
    def _extract_lane_mask(image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, lane_mask = cv2.threshold(gray, 8, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        return lane_mask

    @staticmethod
    def _extract_vehicle_mask(image: np.ndarray, lane_mask: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 140)
        road = cv2.inRange(hsv, (0, 0, 45), (180, 65, 185))
        grass = cv2.inRange(hsv, (30, 40, 60), (85, 255, 255))
        marking_raw = cv2.inRange(hsv, (0, 0, 190), (180, 48, 255))
        # Keep only thin bright structures as lane markings; avoid removing white vehicles.
        mk_v = cv2.morphologyEx(
            marking_raw, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 23)))
        mk_h = cv2.morphologyEx(
            marking_raw, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (23, 3)))
        marking = cv2.bitwise_or(mk_v, mk_h)
        non_vehicle = cv2.bitwise_or(road, cv2.bitwise_or(grass, marking))
        vehicle = cv2.bitwise_not(non_vehicle)
        vehicle = cv2.bitwise_and(vehicle, lane_mask)

        # Drop pixels too close to lane boundary to suppress edge lines/markings.
        dist_edge = cv2.distanceTransform(lane_mask, cv2.DIST_L2, 3)
        vehicle[dist_edge <= 1.5] = 0

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        vehicle = cv2.morphologyEx(vehicle, cv2.MORPH_OPEN, kernel, iterations=1)
        vehicle = cv2.morphologyEx(vehicle, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(vehicle, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        clean = np.zeros_like(vehicle)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 120:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            short_side = max(min(w, h), 1)
            long_side = max(w, h)
            elongation = long_side / short_side
            fill_ratio = area / max(w * h, 1)

            # Filter long thin artifacts (lane markings / boundaries).
            if elongation > 10.0 and fill_ratio < 0.5:
                continue

            comp_mask = np.zeros_like(vehicle)
            cv2.drawContours(comp_mask, [cnt], -1, 255, -1)
            comp_px = max(int(np.sum(comp_mask > 0)), 1)
            sat_mean = float(cv2.mean(hsv[:, :, 1], mask=comp_mask)[0])
            val_mean = float(cv2.mean(hsv[:, :, 2], mask=comp_mask)[0])
            edge_ratio = float(np.sum((edges > 0) & (comp_mask > 0)) / comp_px)

            # Reject large smooth road-like blobs that survived color exclusion.
            if area > 4000 and sat_mean < 60.0 and edge_ratio < 0.045:
                continue
            if sat_mean < 22.0 and edge_ratio < 0.03 and val_mean > 70.0:
                continue

            cv2.drawContours(clean, [cnt], -1, 255, -1)
        return clean

    @staticmethod
    def _extract_component_intervals(
        vehicle_mask: np.ndarray,
        stopline_xy: np.ndarray,
        axis_xy: np.ndarray,
    ) -> List[Tuple[float, float]]:
        """Convert each valid vehicle blob into robust [d_min, d_max] axis interval.

        For very large merged blobs, cap the interval span to avoid one artifact
        stretching the queue tail across the whole lane.
        """
        contours, _ = cv2.findContours(vehicle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        intervals: List[Tuple[float, float]] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 120:
                continue

            pts = cnt.reshape(-1, 2).astype(np.float32)
            delta = stopline_xy.reshape(1, 2) - pts
            d = np.sum(delta * axis_xy.reshape(1, 2), axis=1)
            d = np.maximum(d, 0.0)
            if d.size == 0:
                continue

            d_min = float(np.min(d))
            d_max = float(np.max(d))
            d_center = float(np.mean(d))
            if d_max <= 1.0:
                continue

            raw_span = max(d_max - d_min, 1.0)
            capped_span = min(raw_span, 85.0)
            half = 0.5 * capped_span
            robust_min = max(0.0, d_center - half)
            robust_max = max(robust_min, d_center + half)
            intervals.append((robust_min, robust_max))

        intervals.sort(key=lambda x: x[0])
        return intervals

    @staticmethod
    def _build_row_reference_profile(lane_mask: np.ndarray, reference_line: str) -> Dict:
        h, w = lane_mask.shape[:2]
        ratio_map = {"left": 0.22, "center": 0.5, "right": 0.78}
        ratio = ratio_map.get(reference_line, 0.5)

        x_by_y = {}
        lane_width_by_y = {}
        polyline = []

        valid_rows = np.where(np.sum(lane_mask > 0, axis=1) >= 6)[0]
        if valid_rows.size == 0:
            stopline = np.array([w // 2, h - 1], dtype=np.float32)
            return {
                "stopline_point": stopline,
                "x_by_y": x_by_y,
                "lane_width_by_y": lane_width_by_y,
                "polyline": polyline,
            }

        for y in valid_rows:
            xs = np.where(lane_mask[y] > 0)[0]
            if xs.size < 2:
                continue
            x_left = int(xs.min())
            x_right = int(xs.max())
            lane_w = max(x_right - x_left + 1, 1)
            x_ref = int(round(x_left + ratio * (lane_w - 1)))
            x_by_y[int(y)] = x_ref
            lane_width_by_y[int(y)] = lane_w
            if y % 6 == 0:
                polyline.append((x_ref, int(y)))

        stop_y = int(valid_rows.max())
        stop_x = int(x_by_y.get(stop_y, int(np.mean(list(x_by_y.values()))) if x_by_y else w // 2))
        stopline = np.array([stop_x, stop_y], dtype=np.float32)
        if (stop_x, stop_y) not in polyline:
            polyline.append((stop_x, stop_y))
        polyline = sorted(polyline, key=lambda p: p[1])

        return {
            "stopline_point": stopline,
            "x_by_y": x_by_y,
            "lane_width_by_y": lane_width_by_y,
            "polyline": polyline,
        }

    @staticmethod
    def _distance_from_stopline_row(points_rc: np.ndarray, stop_y: int) -> np.ndarray:
        ys = points_rc[:, 0].astype(np.float32)
        return np.maximum(float(stop_y) - ys, 0.0)

    @staticmethod
    def _extract_component_intervals_row(
        vehicle_mask: np.ndarray,
        ref_x_by_y: Dict[int, int],
        lane_width_by_y: Dict[int, int],
        stop_y: int,
        reference_line: str,
    ) -> List[Tuple[float, float]]:
        contours, _ = cv2.findContours(vehicle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        intervals: List[Tuple[float, float]] = []

        # Keep components around selected reference line.
        band_ratio = 0.46 if reference_line == "center" else 0.35

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 120:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            roi = vehicle_mask[y:y + h, x:x + w]
            ys, xs = np.where(roi > 0)
            if ys.size == 0:
                continue
            ys_full = ys + y
            xs_full = xs + x

            ref_d = []
            for yy, xx in zip(ys_full, xs_full):
                yy_i = int(yy)
                if yy_i not in ref_x_by_y:
                    continue
                lane_w = max(int(lane_width_by_y.get(yy_i, 1)), 1)
                ref_x = int(ref_x_by_y[yy_i])
                ref_d.append(abs(int(xx) - ref_x) / lane_w)

            if not ref_d:
                continue

            if float(np.percentile(ref_d, 35)) > band_ratio:
                continue

            d_vals = np.maximum(float(stop_y) - ys_full.astype(np.float32), 0.0)
            if d_vals.size == 0:
                continue
            d_min = float(np.percentile(d_vals, 12))
            d_max = float(np.percentile(d_vals, 92))
            if d_max <= 1.0:
                continue
            intervals.append((max(0.0, d_min), max(d_min, d_max)))

        intervals.sort(key=lambda t: t[0])
        return intervals

    @staticmethod
    def _build_lane_axis(lane_points_rc: np.ndarray, use_projection: bool) -> Tuple[np.ndarray, np.ndarray, str]:
        points_xy = lane_points_rc[:, ::-1].astype(np.float32)

        if not use_projection:
            axis = np.array([0.0, 1.0], dtype=np.float32)
            stopline = np.array([float(np.mean(points_xy[:, 0])), float(np.max(points_xy[:, 1]))], dtype=np.float32)
            return axis, stopline, "vertical_fallback"

        mean = np.mean(points_xy, axis=0)
        centered = points_xy - mean
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        axis = eigvecs[:, int(np.argmax(eigvals))]
        axis = axis / (np.linalg.norm(axis) + 1e-8)

        proj = centered @ axis
        min_idx = int(np.argmin(proj))
        max_idx = int(np.argmax(proj))
        end_a = points_xy[min_idx]
        end_b = points_xy[max_idx]

        # Enforce a consistent orientation: axis points toward larger image y.
        if end_a[1] > end_b[1]:
            near_end = end_a
            far_end = end_b
        else:
            near_end = end_b
            far_end = end_a

        axis = near_end - far_end
        axis = axis / (np.linalg.norm(axis) + 1e-8)
        stopline = near_end.astype(np.float32)
        return axis.astype(np.float32), stopline, "pca_projection"

    @staticmethod
    def _distance_from_stopline(points_rc: np.ndarray, stopline_xy: np.ndarray, axis_xy: np.ndarray) -> np.ndarray:
        points_xy = points_rc[:, ::-1].astype(np.float32)
        delta = stopline_xy.reshape(1, 2) - points_xy
        dist = np.sum(delta * axis_xy.reshape(1, 2), axis=1)
        return np.maximum(dist, 0.0)

    @staticmethod
    def _estimate_with_slices(
        lane_dist: np.ndarray,
        vehicle_dist: np.ndarray,
        lane_max: float,
        component_intervals: List[Tuple[float, float]],
        bin_size: float,
        occ_threshold: float,
        max_gap_bins: int,
        continuity_bridge_px: float,
    ) -> Tuple[float, Dict]:
        bins = int(np.ceil(lane_max / bin_size)) + 1
        if bins <= 1:
            return 0.0, {"bins": 0, "occupied_bins": 0}

        lane_hist, _ = np.histogram(lane_dist, bins=bins, range=(0, lane_max))
        veh_hist, _ = np.histogram(vehicle_dist, bins=bins, range=(0, lane_max))

        ratio = np.zeros_like(lane_hist, dtype=np.float32)
        valid = lane_hist > 0
        ratio[valid] = veh_hist[valid] / lane_hist[valid]

        occupied = (veh_hist >= 20) & (ratio >= occ_threshold)

        # Reinforce occupancy with component intervals to bridge tiny pixel gaps.
        for d_min, d_max in component_intervals:
            i0 = int(np.floor(d_min / bin_size))
            i1 = int(np.floor(d_max / bin_size))
            i0 = max(0, min(i0, len(occupied) - 1))
            i1 = max(0, min(i1, len(occupied) - 1))
            occupied[i0:i1 + 1] = True

        # Morphological close in 1D: allow tiny holes within a queue.
        max_hole = max(1, max_gap_bins)
        occ_u8 = occupied.astype(np.uint8).reshape(1, -1)
        kernel = np.ones((1, max_hole * 2 + 1), dtype=np.uint8)
        occupied = (cv2.morphologyEx(occ_u8, cv2.MORPH_CLOSE, kernel).reshape(-1) > 0)

        # Prefer component-interval continuity from the nearest detected vehicle.
        continuity_tail = 0.0
        merged_intervals = []
        if component_intervals:
            gap_px = max(bin_size * max_gap_bins + continuity_bridge_px, 20.0)
            cur_start, cur_end = component_intervals[0]
            merged_intervals.append([cur_start, cur_end])
            for s, e in component_intervals[1:]:
                if s - cur_end <= gap_px:
                    cur_end = max(cur_end, e)
                    merged_intervals[-1][1] = cur_end
                else:
                    break
            continuity_tail = float(merged_intervals[-1][1]) if merged_intervals else 0.0

        farthest_bin = -1
        gap_run = 0
        for i in range(len(occupied)):
            if occupied[i]:
                farthest_bin = i
                gap_run = 0
            else:
                gap_run += 1
                if farthest_bin >= 0 and gap_run > max_gap_bins:
                    break

        if farthest_bin < 0:
            queue_length_slice = 0.0
        else:
            queue_length_slice = min((farthest_bin + 1) * bin_size, lane_max)

        if continuity_tail > 0:
            queue_length = min(max(queue_length_slice, continuity_tail), lane_max)
        else:
            queue_length = queue_length_slice

        profile = {
            "bins": int(len(occupied)),
            "occupied_bins": int(np.count_nonzero(occupied)),
            "farthest_occupied_bin": int(farthest_bin),
            "occupancy_ratio_max": round(float(np.max(ratio)) if ratio.size else 0.0, 4),
            "bin_size_px": round(float(bin_size), 2),
            "component_intervals": [[round(a, 2), round(b, 2)] for a, b in component_intervals],
            "continuity_tail_px": round(float(continuity_tail), 2),
            "slice_tail_px": round(float(queue_length_slice), 2),
        }
        return float(queue_length), profile

    @staticmethod
    def _draw_debug_visualization(
        image: np.ndarray,
        lane_mask: np.ndarray,
        vehicle_mask: np.ndarray,
        stopline_point: np.ndarray,
        axis_unit: np.ndarray,
        queue_length: float,
        reference_polyline: List[Tuple[int, int]],
        out_path: str,
    ) -> None:
        vis = image.copy()

        lane_overlay = np.zeros_like(vis)
        lane_overlay[:, :, 1] = lane_mask
        vehicle_overlay = np.zeros_like(vis)
        vehicle_overlay[:, :, 2] = vehicle_mask
        vis = cv2.addWeighted(vis, 1.0, lane_overlay, 0.2, 0)
        vis = cv2.addWeighted(vis, 1.0, vehicle_overlay, 0.35, 0)

        sx, sy = int(stopline_point[0]), int(stopline_point[1])
        ex = int(round(sx - axis_unit[0] * queue_length))
        ey = int(round(sy - axis_unit[1] * queue_length))

        cv2.circle(vis, (sx, sy), 5, (0, 255, 255), -1)
        if reference_polyline and len(reference_polyline) >= 2:
            pts = np.array(reference_polyline, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(vis, [pts], False, (255, 255, 0), 2)
            tip_y = max(0, ey)
            # Use nearest polyline point by y as queue tail marker.
            tail_pt = min(reference_polyline, key=lambda p: abs(p[1] - tip_y))
            ex, ey = int(tail_pt[0]), int(tail_pt[1])
            cv2.line(vis, (sx, sy), (ex, ey), (255, 255, 0), 3)
        else:
            cv2.line(vis, (sx, sy), (ex, ey), (255, 255, 0), 3)
        cv2.putText(
            vis,
            f"queue={queue_length:.1f}px",
            (max(10, min(sx, ex)), max(20, min(sy, ey) - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )

        cv2.imwrite(out_path, vis)

    def _build_text_report(
        self,
        queue_length: float,
        raw_tail: float,
        lane_max: float,
        queue_mode: str,
        axis_mode: str,
    ) -> str:
        ratio = 0.0 if lane_max <= 1e-6 else queue_length / lane_max
        return (
            "=== Queue Length Estimation (single lane) ===\n"
            f"Direction: {self.direction}, lane movement: {self.lane_direction}\n"
            f"Axis mode: {axis_mode}, tail mode: {queue_mode}\n"
            f"Queue length: {queue_length:.1f}px (raw farthest: {raw_tail:.1f}px)\n"
            f"Lane-axis length: {lane_max:.1f}px, queue ratio: {ratio:.1%}"
        )

    def _empty_result(self, reason: str, img_path: str, lane_index: int) -> Dict:
        return {
            "queue_length_px": 0.0,
            "queue_tail_distance_raw_px": 0.0,
            "queue_tail_mode": "none",
            "lane_axis": {
                "mode": "none",
                "unit_vector_xy": [0.0, 1.0],
                "stopline_point_xy": [0, 0],
            },
            "metadata": {
                "direction": self.direction,
                "lane_direction": self.lane_direction,
                "lane_index": int(lane_index),
                "input_image_path": img_path,
                "reason": reason,
            },
            "text_report": f"Queue length estimation unavailable: {reason}",
        }
