"""
Image Analysis Library — 图像分析库

为每个 conversation 维护一个 JSON 文件，存储单图 VLM 分析结果和工具调用结果。
支持渐进式更新：初始单图分析 → 工具补充 → 推理轮次记录。

核心思想：将视觉感知（单图分析，准确）与高层推理（文本决策）解耦，
避免多图推理时 VLM 注意力稀释导致的车辆计数偏差。
"""
import os
import re
import json
from datetime import datetime
from typing import Dict, Optional, List


class ImageAnalysisLibrary:
    """管理图像分析库的创建、更新和文本总结生成。"""

    def __init__(self, base_dir: str):
        """
        Args:
            base_dir: 对话日志基础目录（conversations 所在目录）
        """
        self.base_dir = base_dir
        self.analysis_dir = os.path.join(base_dir, 'image_analysis')
        os.makedirs(self.analysis_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Implementation note.
    # ------------------------------------------------------------------ #

    def _filepath(self, conversation_id: int, step_num: int) -> str:
        return os.path.join(
            self.analysis_dir,
            f"step_{step_num:04d}_conv_{conversation_id}.json")

    # ------------------------------------------------------------------ #
    # Implementation note.
    # ------------------------------------------------------------------ #

    def create_or_load(self,
                       conversation_id: int,
                       step_num: int,
                       intersection_id: str,
                       image_paths: Dict[str, str]) -> Dict:
        """创建新的分析库或加载已有分析库。

        Returns:
            (analysis_data dict, filepath str)
        """
        filepath = self._filepath(conversation_id, step_num)

        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f), filepath

        analysis_data = {
            "metadata": {
                "conversation_id": conversation_id,
                "step_num": step_num,
                "intersection_id": intersection_id,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "analysis_version": "1.0",
            },
            "image_analysis": {},
            "reasoning_rounds": [],
        }

        for direction in ['N', 'S', 'E', 'W']:
            analysis_data["image_analysis"][direction] = {
                "image_path": image_paths.get(direction, ''),
                "analyzed_at": None,
                "method": None,
                "visual_observation": None,
                "vehicle_count": {
                    "total": None,
                    "by_lane": {},
                },
                "traffic_assessment": {},
                "tool_results": {},
                "raw_response": None,
            }

        self._save(filepath, analysis_data)
        return analysis_data, filepath

    # ------------------------------------------------------------------ #
    # Implementation note.
    # ------------------------------------------------------------------ #

    def update_image_analysis(self, filepath: str, direction: str,
                              analysis_result: Dict) -> None:
        """更新某方向的单图 VLM 分析结果。"""
        data = self._load(filepath)
        entry = data["image_analysis"][direction]
        entry["analyzed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry["method"] = "single_image_vlm"
        entry["visual_observation"] = analysis_result.get("visual_observation", "")
        entry["vehicle_count"] = {
            "total": analysis_result.get("total_vehicles", 0),
            "by_lane": analysis_result.get("lane_distribution", {}),
        }
        entry["traffic_assessment"] = {
            "queue_length": analysis_result.get("queue_length", "unknown"),
            "traffic_density": analysis_result.get("traffic_density", "unknown"),
            "most_congested_lane": analysis_result.get("most_congested_lane", "None"),
        }
        entry["raw_response"] = analysis_result.get("raw_response", "")
        self._save(filepath, data)

    # ------------------------------------------------------------------ #
    # Implementation note.
    # ------------------------------------------------------------------ #

    @staticmethod
    def _clean_tool_result(tool_result: Dict) -> Dict:
        """清理工具结果中的大字段（numpy 数组序列化的字符串等），避免 JSON 文件过大。"""
        import copy
        # Implementation note.
        large_keys = {'lane_image', 'trapezoid_image', 'zone_image'}
        
        cleaned = copy.deepcopy(tool_result)
        
        # Implementation note.
        for key in large_keys:
            if key in cleaned:
                del cleaned[key]
        
        # Implementation note.
        if 'lanes' in cleaned and isinstance(cleaned['lanes'], list):
            for lane in cleaned['lanes']:
                for key in large_keys:
                    if key in lane:
                        del lane[key]
        
        # Implementation note.
        if 'zones' in cleaned and isinstance(cleaned['zones'], list):
            for zone in cleaned['zones']:
                for key in large_keys:
                    if key in zone:
                        del zone[key]
        
        return cleaned

    # ------------------------------------------------------------------ #
    # Implementation note.
    # ------------------------------------------------------------------ #

    def update_tool_result(self, filepath: str, direction: str,
                           tool_name: str, tool_result: Dict) -> None:
        """更新工具分析结果。direction='all' 表示全局工具。
        
        对于 lane_segmentation 带 enable_vehicle_counting=true 的结果，
        自动将精确的车道级别计数覆盖到 vehicle_count.by_lane 中。
        """
        data = self._load(filepath)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Clean large fields (numpy arrays serialized as strings) before storing
        clean_result = self._clean_tool_result(tool_result)

        if direction == 'all':
            for d in ['N', 'S', 'E', 'W']:
                dir_data = clean_result.get('directions', {}).get(d)
                if dir_data:
                    data["image_analysis"][d]["tool_results"][tool_name] = {
                        "timestamp": timestamp, **dir_data}
        else:
            data["image_analysis"][direction]["tool_results"][tool_name] = {
                "timestamp": timestamp, **clean_result}
            
            # Implementation note.
            if tool_name == "lane_segmentation" and "lane_counts" in tool_result:
                lane_counts = tool_result["lane_counts"]
                total = tool_result.get("total_vehicles", 0)
                
                entry = data["image_analysis"][direction]
                # Implementation note.
                entry["vehicle_count"]["by_lane"] = {
                    "left_lane": {
                        "count": lane_counts.get("Left", 0),
                        "movement": "Right-turn",
                    },
                    "center_lane": {
                        "count": lane_counts.get("Center", 0),
                        "movement": "Straight",
                    },
                    "right_lane": {
                        "count": lane_counts.get("Right", 0),
                        "movement": "Left-turn",
                    },
                }
                # Implementation note.
                if total > 0:
                    entry["vehicle_count"]["total"] = total
                entry["vehicle_count"]["method"] = "lane_segmentation_vlm_counting"

        self._save(filepath, data)

    # ------------------------------------------------------------------ #
    # Implementation note.
    # ------------------------------------------------------------------ #

    def add_reasoning_round(self, filepath: str, round_num: int,
                            action: str, summary: str,
                            api_calls: int = 0) -> None:
        data = self._load(filepath)
        data["reasoning_rounds"].append({
            "round": round_num,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "action": action,
            "summary": summary,
            "api_calls": api_calls,
        })
        self._save(filepath, data)

    # ------------------------------------------------------------------ #
    # Implementation note.
    # ------------------------------------------------------------------ #

    def is_analyzed(self, data: Dict) -> bool:
        """检查 4 个方向是否都已完成初始分析。"""
        for d in ['N', 'S', 'E', 'W']:
            if data["image_analysis"].get(d, {}).get("analyzed_at") is None:
                return False
        return True

    # ------------------------------------------------------------------ #
    # Implementation note.
    # ------------------------------------------------------------------ #

    def get_text_summary(self, data: Dict) -> str:
        """根据分析数据生成结构化文本总结，用于替代图片输入。
        
        Phase 1 只提供总车数。分车道数据仅在 lane_segmentation 工具
        被调用后（带 enable_vehicle_counting=true）才可用。
        """
        direction_names = {'N': 'North', 'S': 'South', 'E': 'East', 'W': 'West'}
        lines: List[str] = []
        lines.append("== PRE-ANALYZED IMAGE DATA ==")
        lines.append("The following TOTAL vehicle counts were obtained through "
                      "VLM Grounding with bounding box detection "
                      "(one image at a time for maximum accuracy).")
        lines.append("NOTE: Lane-level counts are NOT available yet. "
                      "Call lane_segmentation with enable_vehicle_counting=true "
                      "to get precise per-lane counts for the busiest direction(s).\n")

        for idx, direction in enumerate(['N', 'S', 'E', 'W'], start=1):
            img = data["image_analysis"].get(direction, {})
            full_name = direction_names[direction]
            vc = img.get("vehicle_count", {})
            total = vc.get("total", "?")
            assessment = img.get("traffic_assessment", {})
            method = vc.get("method", "")

            lines.append(f"Image {idx} ({full_name} / {direction}):")
            lines.append(f"  Total vehicles: {total}")

            # Implementation note.
            by_lane = vc.get("by_lane", {})
            if by_lane and method == "lane_segmentation_vlm_counting":
                lane_order = [
                    ("left_lane", "Right-turn (R)"),
                    ("center_lane", "Straight (T)"),
                    ("right_lane", "Left-turn (L)"),
                ]
                lines.append(f"  Per-lane counts (from lane_segmentation tool):")
                for lane_key, lane_label in lane_order:
                    lane = by_lane.get(lane_key, {})
                    count = lane.get("count", 0)
                    lines.append(f"  - {lane_label}: {count} vehicles")
            else:
                lines.append(f"  Per-lane counts: not yet available (call lane_segmentation)")

            lines.append(f"  Queue length: {assessment.get('queue_length', '?')}")
            lines.append(f"  Traffic density: {assessment.get('traffic_density', '?')}")

            # Implementation note.
            tool_results = img.get("tool_results", {})
            if tool_results:
                for tname, tdata in tool_results.items():
                    # Implementation note.
                    skip_keys = {'timestamp', 'dividers', 'metadata', 'trapezoid_image',
                                 'lane_image', 'lanes', 'zone_image', 'direction'}
                    ts_copy = {k: v for k, v in tdata.items() if k not in skip_keys}
                    if ts_copy:
                        lines.append(f"  [{tname}]: {json.dumps(ts_copy, ensure_ascii=False)}")

            lines.append("")

        return '\n'.join(lines)

    # ------------------------------------------------------------------ #
    # Implementation note.
    # ------------------------------------------------------------------ #

    @staticmethod
    def parse_single_image_response(content: str) -> Dict:
        """解析单图 VLM 分析的原始文本输出为结构化 dict。"""
        result: Dict = {
            "visual_observation": "",
            "total_vehicles": 0,
            "lane_distribution": {},
            "queue_length": "unknown",
            "traffic_density": "unknown",
            "most_congested_lane": "None",
            "raw_response": content,
        }

        # Implementation note.
        obs_match = re.search(
            r'\[Visual Observation\]\s*\n(.*?)(?=\n\[|\Z)', content, re.S)
        if obs_match:
            result["visual_observation"] = obs_match.group(1).strip()

        # Implementation note.
        total_match = re.search(r'Total\s*Count:\s*(\d+)', content, re.I)
        if total_match:
            result["total_vehicles"] = int(total_match.group(1))

        # Implementation note.
        lane_patterns = [
            (r'Left\s+lane\s*\(Right-turn\)\s*:\s*(\d+)\s*vehicles?\s*(?:\[(.*?)\])?',
             "left_lane", "Right-turn"),
            (r'Center\s+lane\s*\(Straight\)\s*:\s*(\d+)\s*vehicles?\s*(?:\[(.*?)\])?',
             "center_lane", "Straight"),
            (r'Right\s+lane\s*\(Left-turn\)\s*:\s*(\d+)\s*vehicles?\s*(?:\[(.*?)\])?',
             "right_lane", "Left-turn"),
        ]
        for pattern, key, movement in lane_patterns:
            m = re.search(pattern, content, re.I)
            if m:
                count = int(m.group(1))
                raw_colors = m.group(2) or ''
                colors = [c.strip() for c in raw_colors.split(',') if c.strip()]
                result["lane_distribution"][key] = {
                    "count": count,
                    "colors": colors,
                    "movement": movement,
                }

        # Implementation note.
        if result["total_vehicles"] == 0 and result["lane_distribution"]:
            result["total_vehicles"] = sum(
                v.get("count", 0) for v in result["lane_distribution"].values())

        # Implementation note.
        q_match = re.search(r'Queue\s+length:\s*(\w+)', content, re.I)
        if q_match:
            result["queue_length"] = q_match.group(1).lower()

        d_match = re.search(r'Traffic\s+density:\s*(\w+)', content, re.I)
        if d_match:
            result["traffic_density"] = d_match.group(1)

        c_match = re.search(r'Most\s+congested\s+lane:\s*(\w+)', content, re.I)
        if c_match:
            result["most_congested_lane"] = c_match.group(1)

        return result

    # ------------------------------------------------------------------ #
    # Implementation note.
    # ------------------------------------------------------------------ #

    @staticmethod
    def _load(filepath: str) -> Dict:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def _save(filepath: str, data: Dict) -> None:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
