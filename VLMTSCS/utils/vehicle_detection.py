"""
Vehicle Detection Tool using Grounding DINO
Detects and visualizes vehicles in traffic camera images.
"""

import os
import tempfile
import cv2
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
import torch
from PIL import Image
import time

# Implementation note.
project_root = os.path.dirname(os.path.dirname(__file__))

# Implementation note.
os.environ['TRANSFORMERS_CACHE'] = os.path.join(project_root, "weights")
os.environ['HF_HOME'] = os.path.join(project_root, "weights")
os.environ['HUGGINGFACE_HUB_CACHE'] = os.path.join(project_root, "weights")
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Implementation note.
import warnings
import logging
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", category=UserWarning, module="torch") 
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")
warnings.filterwarnings("ignore", message=".*resume_download.*deprecated.*")
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*deprecated.*")

# Implementation note.
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

# Implementation note.

# Implementation note.
import threading
_MODEL_LOCK = threading.Lock()
_GLOBAL_MODEL_CACHE = {
    'model': None,
    'load_image_fn': None,
    'predict_fn': None,
    'device': None
}


def _line_x_at_y(line: Tuple[int, int, int, int], y: float) -> Optional[float]:
    x1, y1, x2, y2 = line
    if y1 == y2:
        return None
    t = (y - y1) / (y2 - y1)
    return x1 + t * (x2 - x1)


def _polygon_mask(shape: Tuple[int, int], polygon: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    return mask


def _interpolate_point(p1: Tuple[int, int], p2: Tuple[int, int], alpha: float) -> Tuple[int, int]:
    x = int(round(p1[0] + alpha * (p2[0] - p1[0])))
    y = int(round(p1[1] + alpha * (p2[1] - p1[1])))
    return x, y


def _detect_orange_box_geometry(image: np.ndarray) -> Optional[Dict[str, Any]]:
    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mouth_extend = 70

    orange_mask = cv2.inRange(
        hsv,
        np.array([10, 80, 160], dtype=np.uint8),
        np.array([40, 255, 255], dtype=np.uint8),
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    edges = cv2.Canny(orange_mask, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=50,
        minLineLength=max(60, w // 8),
        maxLineGap=30,
    )
    if lines is None:
        return None

    raw_lines = [tuple(map(int, line[0])) for line in lines]
    horizontal = []
    slanted = []
    for line in raw_lines:
        x1, y1, x2, y2 = line
        dx = x2 - x1
        dy = y2 - y1
        length = float(np.hypot(dx, dy))
        if length < 40:
            continue
        if abs(dy) <= 8 and max(y1, y2) > h * 0.55:
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
        x_at_bottom = _line_x_at_y(line, y_bottom)
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

    left_top_x = _line_x_at_y(left_line, 0.0)
    right_top_x = _line_x_at_y(right_line, 0.0)
    left_bottom_x = _line_x_at_y(left_line, y_bottom)
    right_bottom_x = _line_x_at_y(right_line, y_bottom)
    if None in (left_top_x, right_top_x, left_bottom_x, right_bottom_x):
        return None

    extended_bottom_y = min(h - 1, int(y_bottom + mouth_extend))
    left_bottom_ext = _line_x_at_y(left_line, extended_bottom_y)
    right_bottom_ext = _line_x_at_y(right_line, extended_bottom_y)
    if left_bottom_ext is None:
        left_bottom_ext = left_bottom_x
    if right_bottom_ext is None:
        right_bottom_ext = right_bottom_x

    roi_polygon = [
        (int(left_top_x), 0),
        (int(right_top_x), 0),
        (int(right_bottom_ext), int(extended_bottom_y)),
        (int(left_bottom_ext), int(extended_bottom_y)),
    ]
    return {
        "roi_polygon": roi_polygon,
        "y_bottom": int(extended_bottom_y),
        "reference_y_bottom": int(y_bottom),
    }


def _detect_white_dividers(image: np.ndarray, geometry: Dict[str, Any]) -> List[Tuple[int, int, int, int]]:
    h, w = image.shape[:2]
    roi_poly = np.array(geometry["roi_polygon"], dtype=np.int32)
    roi_mask = _polygon_mask((h, w), roi_poly)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(
        hsv,
        np.array([0, 0, 180], dtype=np.uint8),
        np.array([180, 40, 255], dtype=np.uint8),
    )
    white_mask = cv2.bitwise_and(white_mask, roi_mask)
    search_bottom = max(0, geometry["y_bottom"] - 120)
    white_mask[search_bottom:, :] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 9))
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    edges = cv2.Canny(white_mask, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=30,
        minLineLength=max(40, h // 12),
        maxLineGap=30,
    )
    if lines is None:
        return []

    tl, tr, br, bl = geometry["roi_polygon"]
    left_boundary = (tl[0], tl[1], bl[0], bl[1])
    right_boundary = (tr[0], tr[1], br[0], br[1])
    y_top = 0.0
    y_eval = float(search_bottom)
    y_bottom = float(geometry["y_bottom"])
    lt = _line_x_at_y(left_boundary, y_top)
    rt = _line_x_at_y(right_boundary, y_top)
    le = _line_x_at_y(left_boundary, y_eval)
    re = _line_x_at_y(right_boundary, y_eval)
    if None in (lt, rt, le, re):
        return []

    expected = []
    for alpha in (1 / 3, 2 / 3):
        expected.append((
            int(round(lt + alpha * (rt - lt))),
            0,
            int(round(le + alpha * (re - le))),
            int(round(y_eval)),
        ))

    candidates = []
    for raw in lines:
        x1, y1, x2, y2 = map(int, raw[0])
        dx = x2 - x1
        dy = y2 - y1
        if abs(dy) < 20:
            continue
        angle = abs(np.degrees(np.arctan2(dy, dx)))
        if angle < 65:
            continue
        x0 = _line_x_at_y((x1, y1, x2, y2), y_top)
        xe = _line_x_at_y((x1, y1, x2, y2), y_eval)
        if x0 is None or xe is None:
            continue
        if not (le + 10 < xe < re - 10):
            continue
        candidates.append({
            "line": (int(round(x0)), 0, int(round(xe)), int(round(y_eval))),
            "x_eval": xe,
            "length": float(np.hypot(dx, dy)),
        })

    result = []
    for exp in expected:
        exp_x = exp[2]
        close = [c for c in candidates if abs(c["x_eval"] - exp_x) < 80]
        if not close:
            xt = exp[0]
            xb = _line_x_at_y((exp[0], exp[1], exp[2], exp[3]), y_bottom)
            xb = exp[2] if xb is None else xb
            result.append((xt, 0, int(round(xb)), int(round(y_bottom))))
            continue
        total_len = sum(c["length"] for c in close)
        avg_top = sum(c["line"][0] * c["length"] for c in close) / total_len
        avg_eval = sum(c["line"][2] * c["length"] for c in close) / total_len
        xb = _line_x_at_y((avg_top, 0, avg_eval, y_eval), y_bottom)
        xb = avg_eval if xb is None else xb
        result.append((int(round(avg_top)), 0, int(round(xb)), int(round(y_bottom))))

    result.sort(key=lambda l: (l[0] + l[2]) / 2.0)
    return result


def _build_lane_polygons(
    geometry: Dict[str, Any],
    dividers: List[Tuple[int, int, int, int]],
) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]:
    tl, tr, br, bl = geometry["roi_polygon"]

    if len(dividers) >= 2:
        d1, d2 = dividers[:2]
        lane_polygons = [
            np.array([tl, (d1[0], d1[1]), (d1[2], d1[3]), bl], dtype=np.int32),
            np.array([(d1[0], d1[1]), (d2[0], d2[1]), (d2[2], d2[3]), (d1[2], d1[3])], dtype=np.int32),
            np.array([(d2[0], d2[1]), tr, br, (d2[2], d2[3])], dtype=np.int32),
        ]
        return lane_polygons, dividers[:2]

    top_1 = _interpolate_point(tl, tr, 1 / 3)
    top_2 = _interpolate_point(tl, tr, 2 / 3)
    bottom_1 = _interpolate_point(bl, br, 1 / 3)
    bottom_2 = _interpolate_point(bl, br, 2 / 3)
    fallback_dividers = [
        (top_1[0], top_1[1], bottom_1[0], bottom_1[1]),
        (top_2[0], top_2[1], bottom_2[0], bottom_2[1]),
    ]
    lane_polygons = [
        np.array([tl, top_1, bottom_1, bl], dtype=np.int32),
        np.array([top_1, top_2, bottom_2, bottom_1], dtype=np.int32),
        np.array([top_2, tr, br, bottom_2], dtype=np.int32),
    ]
    return lane_polygons, fallback_dividers


def _draw_lane_overlay(image: np.ndarray) -> np.ndarray:
    geometry = _detect_orange_box_geometry(image)
    if geometry is None:
        return image

    dividers = _detect_white_dividers(image, geometry)
    lane_polygons, draw_dividers = _build_lane_polygons(geometry, dividers)

    overlay = image.copy()
    colors = [(255, 80, 80), (80, 255, 255), (255, 80, 255)]
    for poly, color in zip(lane_polygons, colors):
        cv2.fillPoly(overlay, [poly], color)

    vis = cv2.addWeighted(overlay, 0.15, image, 0.85, 0)
    cv2.polylines(vis, [np.array(geometry["roi_polygon"], dtype=np.int32)], True, (0, 255, 0), 2)
    for i, (x1, y1, x2, y2) in enumerate(draw_dividers):
        cv2.line(vis, (x1, y1), (x2, y2), (255, 255, 0), 2)
        label_x = (x1 + x2) // 2
        label_y = max(40, (y1 + y2) // 2)
        cv2.putText(vis, f"d{i}", (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    names = ["right turn", "straight", "left turn"]
    arrow_anchor_y = max(geometry["y_bottom"] - 22, 40)
    x_offsets = [-70, -35, 10]
    for name, poly, color, x_offset in zip(names, lane_polygons, colors, x_offsets):
        xs = poly[:, 0]
        x_pos = int(xs.mean()) + x_offset
        cv2.putText(
            vis,
            name,
            (x_pos, arrow_anchor_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
    return vis


class VehicleDetectionTool:
    """
    Vehicle detection tool using Grounding DINO for open-set object detection.
    Detects vehicles (car, truck, bus) and generates visualization with bounding boxes.
    """
    
    def __init__(self, direction: str = "N", scenario: str = "default"):
        """
        Initialize the vehicle detection tool.
        
        Args:
            direction: Camera direction (N/S/E/W)
            scenario: Scenario name for logging
        """
        self.direction = direction
        self.scenario = scenario
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Implementation note.
        self._ensure_model_loaded()
        
    def _ensure_model_loaded(self):
        """确保全局模型已加载（使用缓存以提升性能）。"""
        global _GLOBAL_MODEL_CACHE, _MODEL_LOCK

        # Implementation note.
        if _GLOBAL_MODEL_CACHE['model'] is not None:
            return

        # Implementation note.
        with _MODEL_LOCK:
            # Implementation note.
            if _GLOBAL_MODEL_CACHE['model'] is not None:
                return

            print(f"[VehicleDetection] Loading Grounding DINO model...")
            start_time = time.time()

            try:
                import transformers
                transformers.utils.logging.set_verbosity_error()

                import io
                from contextlib import redirect_stdout, redirect_stderr

                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    from groundingdino.util.inference import load_model, load_image, predict

                model_config_path = os.path.join(
                    os.path.dirname(__file__), "..", "GroundingDINO",
                    "groundingdino", "config", "GroundingDINO_SwinB_cfg.py"
                )
                model_checkpoint_path = os.path.join(
                    os.path.dirname(__file__), "..", "weights", "groundingdino_swinb_cogcoor.pth"
                )

                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    _GLOBAL_MODEL_CACHE['model'] = load_model(model_config_path, model_checkpoint_path)
                    _GLOBAL_MODEL_CACHE['load_image_fn'] = load_image
                    _GLOBAL_MODEL_CACHE['predict_fn'] = predict
                    _GLOBAL_MODEL_CACHE['device'] = self.device

                load_time = time.time() - start_time
                print(f"[VehicleDetection] Model loaded in {load_time:.1f}s on {self.device}")

            except ImportError as e:
                raise ImportError(
                    "Grounding DINO not installed. Please install it:\n"
                    "cd GroundingDINO && pip install -e .\n"
                    "See: https://github.com/IDEA-Research/GroundingDINO"
                ) from e
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load Grounding DINO model: {e}\n"
                    f"Please check model files integrity."
                ) from e
    
    def detect_vehicles(
        self,
        image_path: str,
        text_prompt: str = "vehicle . car . automobile . truck",
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        draw_boxes: bool = True,
        output_dir: str = "./output"
    ) -> Dict[str, Any]:
        """
        Detect vehicles in the given image using Grounding DINO.
        
        Args:
            image_path: Path to input image
            text_prompt: Text prompt for detection (e.g., "car . truck . bus")
            box_threshold: Confidence threshold for bounding boxes
            text_threshold: Text similarity threshold
            draw_boxes: Whether to generate visualization image
            output_dir: Directory to save output images
            
        Returns:
            Dict containing:
                - detections: List of detected vehicles with bbox, confidence, label
                - vehicle_count: Total number of vehicles
                - vehicle_types: Count by vehicle type
                - visualization_image_path: Path to visualization (if draw_boxes=True)
                - metadata: Detection metadata
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Implementation note.
        global _GLOBAL_MODEL_CACHE
        model = _GLOBAL_MODEL_CACHE['model']
        load_image_fn = _GLOBAL_MODEL_CACHE['load_image_fn']
        predict_fn = _GLOBAL_MODEL_CACHE['predict_fn']
        
        if model is None:
            raise RuntimeError("Failed to access cached Grounding DINO model")
        
        # Build lane overlay on the original image first, then feed the overlaid image to DINO.
        original_bgr = cv2.imread(image_path)
        if original_bgr is None:
            raise FileNotFoundError(f"Failed to read image for overlay: {image_path}")
        overlay_source = _draw_lane_overlay(original_bgr.copy())

        tmp_overlay_path = None
        try:
            tmp_fd, tmp_path = tempfile.mkstemp(suffix='.jpg')
            os.close(tmp_fd)
            ok, buf = cv2.imencode('.jpg', overlay_source)
            if not ok:
                raise RuntimeError('Failed to encode overlay image')
            with open(tmp_path, 'wb') as f:
                f.write(buf.tobytes())
            tmp_overlay_path = tmp_path

            # Load and preprocess the overlaid image for DINO
            image_source, image = load_image_fn(tmp_overlay_path)
            
            # Implementation note.
            global _MODEL_LOCK
            start_inference = time.time()
            with _MODEL_LOCK:
                boxes, logits, phrases = predict_fn(
                    model=model,
                    image=image,
                    caption=text_prompt,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                    device=self.device
                )
            inference_time = time.time() - start_inference
            # Implementation note.
            
            # Convert boxes to pixel coordinates
            h, w, _ = overlay_source.shape
            boxes_xyxy = self._box_cxcywh_to_xyxy(boxes) * torch.Tensor([w, h, w, h])
        
            # Build detection results
            raw_detections = []
            for box, confidence, label in zip(boxes_xyxy, logits, phrases):
                x1, y1, x2, y2 = box.cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                raw_detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float(confidence),
                    "label": label.strip()
                })
        
            # Apply Non-Maximum Suppression to remove duplicate detections
            detections = self._apply_nms(raw_detections, iou_threshold=0.5)
        
            # Simplified: all detected objects are counted as vehicles
            vehicle_types = {"vehicle": len(detections)}
        
            # Update detection labels for consistency
            for det in detections:
                det["label"] = "vehicle"
        
            vehicle_count = len(detections)
        
            # Generate visualization if requested
            visualization_path = None
            if draw_boxes and vehicle_count > 0:
                visualization_path = self._draw_bounding_boxes(
                    overlay_source, detections, output_dir
                )
            elif draw_boxes and vehicle_count == 0:
                os.makedirs(output_dir, exist_ok=True)
                overlay_only_path = os.path.join(output_dir, f"overlay_{self.direction}.jpg")
                cv2.imwrite(overlay_only_path, overlay_source)
                if os.path.exists(overlay_only_path):
                    visualization_path = overlay_only_path
        
            # Build result
            result = {
                "detections": detections,
                "vehicle_count": vehicle_count,
                "vehicle_types": vehicle_types,
                "direction": self.direction,
                "metadata": {
                    "image_path": image_path,
                    "image_size": [w, h],
                    "text_prompt": text_prompt,
                    "box_threshold": box_threshold,
                    "text_threshold": text_threshold,
                    "model": "Grounding DINO",
                    "device": self.device
                }
            }
            
            if visualization_path:
                result["visualization_image_path"] = visualization_path
            
            return result
        finally:
            if tmp_overlay_path and os.path.exists(tmp_overlay_path):
                try:
                    os.remove(tmp_overlay_path)
                except Exception:
                    pass
    
    @staticmethod
    def _prepare_image_from_array(img_bgr: np.ndarray):
        """将 BGR numpy array 直接转为 DINO 输入，跳过磁盘 I/O。

        等价于 load_image_fn(path) 但省去 JPEG 编解码和磁盘读写。
        """
        import groundingdino.datasets.transforms as T
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        image_transformed, _ = transform(pil_img, None)
        return img_bgr, image_transformed

    def detect_vehicles_batch(
        self,
        images: List[Union[str, np.ndarray]],
        text_prompt: str = "vehicle . car . automobile . truck",
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ) -> List[Dict[str, Any]]:
        """Batch detect vehicles in multiple images using Grounding DINO.

        Acquires the model lock ONCE and processes all images sequentially
        inside it, avoiding repeated lock acquisition overhead.  Each image
        is still processed individually (Grounding DINO does not natively
        support batched tensors), but the single-lock session eliminates
        context-switching and lock-wait costs.

        Args:
            images: List of image inputs. Each element can be either:
                - str: file path (ASCII-safe)
                - np.ndarray: BGR image array (skips disk I/O)
            text_prompt: Text prompt for detection.
            box_threshold: Confidence threshold for bounding boxes.
            text_threshold: Text similarity threshold.

        Returns:
            List of dicts, one per input image, each containing:
                - vehicle_count: int
                - detections: list of detection dicts
        """
        # Implementation note.
        self._ensure_model_loaded()

        global _GLOBAL_MODEL_CACHE, _MODEL_LOCK
        model = _GLOBAL_MODEL_CACHE['model']
        load_image_fn = _GLOBAL_MODEL_CACHE['load_image_fn']
        predict_fn = _GLOBAL_MODEL_CACHE['predict_fn']

        if model is None:
            raise RuntimeError("Failed to access cached Grounding DINO model")

        results = []
        # Single lock acquisition for the entire batch
        with _MODEL_LOCK:
            for img_input in images:
                try:
                    # Implementation note.
                    if isinstance(img_input, np.ndarray):
                        if img_input is None or img_input.size == 0:
                            results.append({"vehicle_count": 0, "detections": []})
                            continue
                        image_source, image = self._prepare_image_from_array(img_input)
                        h, w = image_source.shape[:2]
                    elif isinstance(img_input, str):
                        if not os.path.exists(img_input):
                            results.append({"vehicle_count": 0, "detections": []})
                            continue
                        image_source, image = load_image_fn(img_input)
                        h, w = image_source.shape[:2] if hasattr(image_source, 'shape') else (0, 0)
                    else:
                        results.append({"vehicle_count": 0, "detections": []})
                        continue

                    boxes, logits, phrases = predict_fn(
                        model=model,
                        image=image,
                        caption=text_prompt,
                        box_threshold=box_threshold,
                        text_threshold=text_threshold,
                        device=self.device,
                    )

                    boxes_xyxy = self._box_cxcywh_to_xyxy(boxes) * torch.Tensor([w, h, w, h])
                    raw_detections = []
                    for box, confidence, label in zip(boxes_xyxy, logits, phrases):
                        x1, y1, x2, y2 = box.cpu().numpy()
                        raw_detections.append({
                            "bbox": [int(x1), int(y1), int(x2), int(y2)],
                            "confidence": float(confidence),
                            "label": label.strip(),
                        })
                    detections = self._apply_nms(raw_detections, iou_threshold=0.5)
                    for det in detections:
                        det["label"] = "vehicle"
                    results.append({
                        "vehicle_count": len(detections),
                        "detections": detections,
                    })
                except Exception as e:
                    label = img_input if isinstance(img_input, str) else '<ndarray>'
                    print(f"Warning: batch DINO detection failed for {label}: {e}")
                    results.append({"vehicle_count": 0, "detections": []})

        return results

    def _box_cxcywh_to_xyxy(self, boxes):
        """Convert boxes from center format to corner format."""
        x_c, y_c, w, h = boxes.unbind(-1)
        boxes_xyxy = torch.stack([
            x_c - 0.5 * w, y_c - 0.5 * h,
            x_c + 0.5 * w, y_c + 0.5 * h
        ], dim=-1)
        return boxes_xyxy
    
    def _apply_nms(self, detections, iou_threshold=0.5):
        """Apply Non-Maximum Suppression to remove duplicate detections."""
        if not detections:
            return detections
        
        # Sort by confidence (highest first)
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        filtered_detections = []
        
        for detection in detections:
            # Check if this detection overlaps significantly with any already kept detection
            should_keep = True
            
            for kept_detection in filtered_detections:
                iou = self._calculate_iou(detection['bbox'], kept_detection['bbox'])
                if iou > iou_threshold:
                    should_keep = False
                    break
            
            if should_keep:
                filtered_detections.append(detection)
        
        return filtered_detections
    
    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)
        
        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0
        
        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def _draw_bounding_boxes(
        self,
        image: np.ndarray,
        detections: List[Dict],
        output_dir: str
    ) -> str:
        """
        Draw bounding boxes on image and save.
        
        Args:
            image: Original image (numpy array)
            detections: List of detection dicts with bbox, confidence, label
            output_dir: Output directory
            
        Returns:
            Path to saved visualization image
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Copy image to preserve original background
        vis_image = image.copy()
        
        # Draw clean bounding boxes for each detection
        for det in detections:
            bbox = det["bbox"]
            x1, y1, x2, y2 = map(int, bbox)
            
            # Use bright cyan color for all vehicles (highly visible)
            color = (0, 255, 255)  # Cyan - stands out well against most backgrounds
            
            # Draw clean rectangle outline
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 3)
        
        # Add clean summary text with background (top-right corner to avoid overlap)
        summary = f"Vehicles: {len(detections)}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        
        # Get text size and image dimensions
        h, w = vis_image.shape[:2]
        (text_w, text_h), baseline = cv2.getTextSize(summary, font, font_scale, thickness)
        
        # Position in top-right corner with margin
        margin = 10
        text_x = w - text_w - margin - 5
        text_y = margin
        
        # Draw semi-transparent background rectangle
        cv2.rectangle(
            vis_image,
            (text_x - 5, text_y),
            (text_x + text_w + 5, text_y + text_h + baseline + 5),
            (0, 0, 0),  # Black background
            -1
        )
        
        # Draw text
        cv2.putText(
            vis_image,
            summary,
            (text_x, text_y + text_h + 2),
            font,
            font_scale,
            (0, 255, 255),  # Cyan text to match boxes
            thickness
        )
        
        # Save image
        output_filename = f"{self.direction}_vehicle_detection.jpg"
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, vis_image)
        
        return output_path


# Standalone test
if __name__ == "__main__":
    tool = VehicleDetectionTool(direction="N")
    
    # Test detection
    result = tool.detect_vehicles(
        image_path="path/to/test_image.jpg",
        text_prompt="vehicle . car . automobile . truck",
        box_threshold=0.35,
        text_threshold=0.25,
        draw_boxes=True,
        output_dir="./test_output"
    )
    
    print(f"Detected {result['vehicle_count']} vehicles")
    print(f"Vehicle types: {result['vehicle_types']}")
    if result.get('visualization_image_path'):
        print(f"Visualization saved to: {result['visualization_image_path']}")
