"""V35 four-video traffic-signal agent backed by an OpenAI-compatible API."""

from __future__ import annotations

import base64
import copy
import json
import math
import mimetypes
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

from utils.v35_video_prompt import SYSTEM_PROMPT, build_v35_prompt


PHASES: Tuple[str, ...] = ("ETWT", "NTST", "ELWL", "NLSL")
VIDEO_DIRECTIONS: Tuple[str, ...] = ("E", "W", "N", "S")
MOVEMENT_MAP = {
    "WL": 0, "WT": 1, "WR": 2, "EL": 3, "ET": 4, "ER": 5,
    "NL": 6, "NT": 7, "NR": 8, "SL": 9, "ST": 10, "SR": 11,
}


DEPLOYMENT_STAGE1_SYSTEM = """You are the perception stage of a traffic signal controller.
Read the supplied four approach videos and coordination frames. Return exactly one
<perception>...</perception> block containing the required JSON schema. Do not emit
<mode>, <reasoning>, <signal>, markdown, or text outside the perception block."""

DEPLOYMENT_STAGE2_SYSTEM = """You are the decision stage of a traffic signal controller.
The perception JSON below was produced by Stage 1 and has already been routed across
the intersection. Choose exactly one phase from ETWT, NTST, ELWL, NLSL.
Return either:
<mode>fast</mode>\n<signal>PHASE</signal>
or:
<mode>slow</mode>\n<reasoning>brief comparison</reasoning>\n<signal>PHASE</signal>
Do not emit markdown or text outside the required tags."""


class VideoAgent:
    """Online inference agent matching the V35 SFT multimodal contract."""

    @staticmethod
    def _as_bool(value: Any, default: bool = False) -> bool:
        """Parse bool-like config values without treating ``"false"`` as true."""
        if value is None:
            return bool(default)
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        normalized = str(value).strip().lower()
        if normalized in {"1", "true", "yes", "y", "on", "enable", "enabled"}:
            return True
        if normalized in {"0", "false", "no", "n", "off", "disable", "disabled", ""}:
            return False
        return bool(default)

    def __init__(self, tls_id: str, phase_list: list, api_url: str = "",
                 api_key: str = "", model_name: str = "", log_dir: str = "",
                 scenario: str = "jinan", tool_executor=None,
                 vlm_runtime_config: Optional[Dict[str, Any]] = None,
                 conversations_dir: str = "", api_debug_dir: str = "",
                 seed: Optional[int] = None):
        if tuple(phase_list) != PHASES:
            raise ValueError(f"VideoAgent requires phase order {PHASES}, got {phase_list}")
        self.tls_id = tls_id
        self.phase_list = list(phase_list)
        self.num_phases = len(phase_list)
        self.scenario = scenario
        self.vlm_runtime_config = vlm_runtime_config or {}
        # VideoAgent is the deployment decision agent.  Its Stage 1 and
        # Stage 2 calls intentionally share the deployed decision model, so
        # command-line --decision-* overrides must take precedence over the
        # generic VLM defaults used by collection-only agents.
        self.api_url = self.vlm_runtime_config.get(
            "DECISION_API_URL", self.vlm_runtime_config.get("VLM_API_URL", api_url)
        )
        self.api_key = self.vlm_runtime_config.get(
            "DECISION_API_KEY", self.vlm_runtime_config.get("VLM_API_KEY", api_key)
        )
        self.model_name = self.vlm_runtime_config.get(
            "DECISION_MODEL", self.vlm_runtime_config.get("VLM_MODEL", model_name)
        )
        self.temperature = float(self.vlm_runtime_config.get(
            "DECISION_TEMPERATURE", self.vlm_runtime_config.get("VLM_TEMPERATURE", 0.0)
        ))
        self.max_tokens = int(self.vlm_runtime_config.get(
            "DECISION_MAX_TOKENS", self.vlm_runtime_config.get("VLM_MAX_TOKENS", 4096)
        ))
        self.timeout = float(self.vlm_runtime_config.get("VLM_TIMEOUT", 900.0))
        self.retries = int(self.vlm_runtime_config.get("VLM_RETRIES", 2))
        self.adaptive_mode_routing = self._as_bool(
            self.vlm_runtime_config.get("VLM_ADAPTIVE_MODE_ROUTING", False)
        )
        self.reasoning_mode = str(
            self.vlm_runtime_config.get("VLM_REASONING_MODE", "adaptive")
        ).strip().lower()
        if self.reasoning_mode not in {"adaptive", "fast", "slow"}:
            raise ValueError(
                "VLM_REASONING_MODE must be one of: adaptive, fast, slow"
            )
        # Deployment defaults to the same two-stage contract as online RL:
        # multimodal perception is generated once, then the text-only decision
        # branch is routed and continued.  Object-created test doubles retain
        # the legacy path through getattr(..., False) in choose_action.
        self.deployment_two_stage = self._as_bool(
            self.vlm_runtime_config.get("VLM_DEPLOYMENT_TWO_STAGE", True)
        )
        # Diagnostic deployment mode: retain the normal global Stage 1
        # barrier and cooperative router, but replace visual perception with
        # the canonical SUMO-backed perception used by the existing Stage 1
        # recovery path.  Stage 2 remains the only model inference call.
        self.stage1_sumo_only = self._as_bool(
            self.vlm_runtime_config.get("VLM_STAGE1_SUMO_ONLY", False)
        )
        # Ablation switch: keep the default cooperative Stage 2 contract
        # unchanged, but allow a local-only Stage 2 input for controlled tests.
        self.stage2_include_cooperation = self._as_bool(
            self.vlm_runtime_config.get("VLM_STAGE2_INCLUDE_COOPERATION", True),
            default=True,
        )
        self.camera_view_distance = float(
            self.vlm_runtime_config.get("CAMERA_VIEW_DISTANCE", 150.0)
        )
        self.mode_threshold_path = str(
            self.vlm_runtime_config.get("VLM_MODE_THRESHOLD_PATH", "") or ""
        )
        self.mode_threshold_default = float(
            self.vlm_runtime_config.get("VLM_MODE_THRESHOLD_DEFAULT", 0.5)
        )
        self.seed = self.vlm_runtime_config.get("SEED", seed)
        self.current_phase_idx = 0
        self.action_history: List[Tuple[int, int]] = []
        self._history = {phase: 0 for phase in PHASES}
        self._last_current_v = {phase: 0 for phase in PHASES}
        self._current_v_history = {phase: [] for phase in PHASES}
        self._last_decision_text = ""
        self._last_api_error: Optional[str] = None
        self._last_decision_source = "uninitialized"
        self._last_decision_record: Dict[str, Any] = {}
        self._last_mode_routing: Dict[str, Any] = {}
        self._current_step_dir_counts: Dict[str, int] = {}
        self.conversations_dir = conversations_dir or os.path.join(log_dir, "conversations")
        self.stage1_conversations_dir = os.path.join(self.conversations_dir, "stage1")
        self.stage2_conversations_dir = os.path.join(self.conversations_dir, "stage2")
        self.api_debug_dir = api_debug_dir or os.path.join(log_dir, "api_debug")
        os.makedirs(self.conversations_dir, exist_ok=True)
        os.makedirs(self.stage1_conversations_dir, exist_ok=True)
        os.makedirs(self.stage2_conversations_dir, exist_ok=True)
        os.makedirs(self.api_debug_dir, exist_ok=True)
        self.log_path = os.path.join(self.conversations_dir, f"{tls_id}_video_agent.txt")
        self.stage1_log_path = os.path.join(self.stage1_conversations_dir, f"{tls_id}_stage1.txt")
        self.stage2_log_path = os.path.join(self.stage2_conversations_dir, f"{tls_id}_stage2.txt")
        resume_existing = self._as_bool(
            self.vlm_runtime_config.get("RESUME_FROM_CHECKPOINT", False), False)
        log_mode = "a" if resume_existing and os.path.isfile(self.log_path) else "w"
        with open(self.log_path, log_mode, encoding="utf-8") as handle:
            if log_mode == "a":
                handle.write("\n[RESUME] VideoAgent process restarted; checkpoint restore pending.\n")
            else:
                handle.write(f"[VIDEO] Traffic Control Agent Log -- {self.tls_id}\n")
                handle.write("=" * 72 + "\n")
                handle.write(f"Scenario:           {self.scenario}\n")
                handle.write("Agent type:         video_agent\n")
                handle.write(f"Experiment seed:    {self.seed}\n")
                handle.write(f"Available phases:   {self.phase_list}\n")
                handle.write(f"Decision model:     {self.model_name}\n")
                handle.write(
                    "Stage 1 source:      SUMO canonical fallback (forced)\n"
                    if self.stage1_sumo_only
                    else "Stage 1 source:      four approach videos and coordination frames\n"
                )
                handle.write("=" * 72 + "\n\n")

    @staticmethod
    def _data_url(path: str) -> str:
        mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
        encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
        return f"data:{mime};base64,{encoded}"

    @staticmethod
    def _direction_key(value: Any) -> Optional[str]:
        text = str(value or "").upper()
        match = re.search(r"(?:^|[_\-/])([EWNS])(?:$|[_\-.])", text)
        return match.group(1) if match else (text if text in VIDEO_DIRECTIONS else None)

    def _video_paths(self, agent_input: Any) -> Dict[str, str]:
        if not isinstance(agent_input, dict) or agent_input.get("mode") != "video":
            raise ValueError("VideoAgent requires the video input package from VLMOneLine")
        raw = agent_input.get("video_paths") or {}
        found: Dict[str, str] = {}
        if isinstance(raw, dict):
            for key, path in raw.items():
                direction = self._direction_key(key) or self._direction_key(path)
                if direction and path and os.path.isfile(path):
                    found[direction] = os.fspath(path)
        missing = [direction for direction in VIDEO_DIRECTIONS if direction not in found]
        if missing:
            raise ValueError(
                f"Missing direction videos {missing} for {self.tls_id}; "
                "set VIDEO_EXPORT_DIRECTIONS=True"
            )
        return found

    def _coordination_frames(self, agent_input: Any, neighbor_info: Any,
                             step_num: int = 0) -> List[Tuple[str, str]]:
        candidates: List[Any] = []
        if isinstance(agent_input, dict):
            candidates.extend(agent_input.get("coordination_frames") or [])
        if isinstance(neighbor_info, dict):
            candidates.extend(neighbor_info.get("__v35_coordination_frames__") or [])
            candidates.extend(neighbor_info.get("coordination_frames") or [])
            candidates.extend(neighbor_info.get("__v35_coordination_video_paths__") or [])
        frames: Dict[str, str] = {}
        for item in candidates:
            if isinstance(item, str):
                path, direction = item, self._direction_key(item)
            elif isinstance(item, dict):
                path = item.get("path") or item.get("coordination_frame_path") or item.get("image_path")
                direction = self._direction_key(
                    item.get("direction") or item.get("target_entry_direction") or path)
                try:
                    if float(item.get("distance_m")) <= self.camera_view_distance:
                        continue
                except (TypeError, ValueError):
                    pass
            else:
                continue
            if not path or not direction or not os.path.isfile(path):
                continue
            if str(path).lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
                frame_index = item.get("frame_index") if isinstance(item, dict) else None
                distance = item.get("distance_m") if isinstance(item, dict) else None
                if frame_index is None and distance is not None:
                    frame_index = self._select_coordination_frame_index(float(distance))
                path = self._extract_frame(path, direction, frame_index, step_num)
            if path and os.path.isfile(path):
                frames[direction] = os.fspath(path)
        return [(direction, frames[direction]) for direction in VIDEO_DIRECTIONS if direction in frames]

    @staticmethod
    def _select_coordination_frame_index(distance_m: float) -> int:
        from utils.coordination_frame_selector import select_coordination_frame
        selected, _ = select_coordination_frame(distance_m)
        return int(selected.frame_index)

    def _extract_frame(self, video_path: str, target_direction: str,
                       frame_index: Optional[int] = None,
                       step_num: int = 0) -> Optional[str]:
        """Extract the V30-selected frame from an upstream sampled video."""
        suffix = f"f{frame_index}" if frame_index is not None else "last"
        # Keep reusable coordination evidence in the run record, separate
        # from transient API response debugging files.
        run_dir = Path(self.conversations_dir).parent
        # The decision at step k consumes the video window finalized after
        # step k-1.  Coordination frames are extracted from that same source
        # video.  Preserve the source window's step in the artifact path
        # instead of placing it under the current decision step, which made
        # otherwise aligned inputs look like they came from different steps.
        source_step = None
        match = re.search(r"(?:^|[\\/])step_(\d+)(?:[\\/])", os.fspath(video_path))
        if match:
            source_step = int(match.group(1))
        artifact_step = source_step if source_step is not None else int(step_num)
        output = run_dir / "images" / f"step_{artifact_step:04d}" / self.tls_id / (
            f"{self.tls_id}_coord_{target_direction}_{suffix}_{Path(video_path).stem}.jpg")
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.is_file():
            return os.fspath(output)
        try:
            import cv2
            capture = cv2.VideoCapture(video_path)
            if not capture.isOpened():
                return None
            frame = None
            target = None if frame_index is None else max(0, int(frame_index) - 1)
            index = 0
            while True:
                ok, candidate = capture.read()
                if not ok:
                    break
                frame = candidate
                if target is not None and index >= target:
                    break
                index += 1
            capture.release()
            if frame is None or not cv2.imwrite(os.fspath(output), frame):
                return None
            return os.fspath(output)
        except Exception:
            return None

    def _user_prompt(self, coordination_directions: Iterable[str]) -> str:
        directions = list(coordination_directions)
        history = "\n".join(
            ["Controller-provided persistent-demand history:",
             "nonzero_v_history_length_since_last_service"]
            + [f"- {phase}: {self._history[phase]}" for phase in PHASES]
        )
        return build_v35_prompt(
            self.tls_id,
            self.phase_list[self.current_phase_idx],
            history,
            directions,
        )[0]["content"]

    def _messages(self, videos: Dict[str, str], frames: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = [{"type": "text", "text": self._user_prompt(d for d, _ in frames)}]
        for direction in VIDEO_DIRECTIONS:
            content.append({"type": "video_url", "video_url": {"url": self._data_url(videos[direction])}})
        for _, path in frames:
            content.append({"type": "image_url", "image_url": {"url": self._data_url(path)}})
        return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}]

    @staticmethod
    def _text_from_multimodal_message(message: Dict[str, Any]) -> str:
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(
                str(item.get("text", "")) for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            )
        return str(content)

    def _stage1_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build the deployment request from the exact Stage 1 SFT sample template."""
        result = copy.deepcopy(messages)
        baseline = self._stage1_baseline_template()
        if baseline is None:
            raise FileNotFoundError(
                "Stage 1 SFT template train_sample_1.txt is required for deployment"
            )
        if len(result) < 2 or not isinstance(result[1].get("content"), list):
            raise ValueError("Stage 1 requires a multimodal user message")
        baseline = list(baseline)

        source_text = self._text_from_multimodal_message(result[1])
        intersection = re.search(r"^Intersection: (.+)$", source_text, re.MULTILINE)
        current_phase = re.search(r"^Current phase: (.+)$", source_text, re.MULTILINE)
        if intersection:
            baseline[1] = re.sub(r"^Intersection: .*?$", f"Intersection: {intersection.group(1)}", baseline[1], count=1, flags=re.MULTILINE)
        if current_phase:
            baseline[1] = re.sub(r"^Current phase: .*?$", f"Current phase: {current_phase.group(1)}", baseline[1], count=1, flags=re.MULTILINE)
        for phase in PHASES:
            age = re.search(rf"^- {phase}: (\d+)$", source_text, re.MULTILINE)
            if age:
                baseline[1] = re.sub(rf"^- {phase}: .*?$", f"- {phase}: {age.group(1)}", baseline[1], count=1, flags=re.MULTILINE)
        directions = re.search(r"Coordination frame inputs, in this exact order: (.*?)\.", source_text)
        if directions:
            baseline[1] = re.sub(r"Coordination frame inputs, in this exact order: .*?\.", f"Coordination frame inputs, in this exact order: {directions.group(1)}.", baseline[1], count=1)

        media = [item for item in result[1]["content"] if isinstance(item, dict) and item.get("type") in {"video_url", "image_url"}]
        image_count = sum(item.get("type") == "image_url" for item in media)
        baseline[1] = re.sub(r"<image>(?:<image>)*", "<image>" * image_count, baseline[1], count=1)
        parts = re.split(r"(<video>|<image>)", baseline[1])
        content: List[Dict[str, Any]] = []
        media_iter = iter(media)
        for part in parts:
            if part in {"<video>", "<image>"}:
                try:
                    content.append(next(media_iter))
                except StopIteration as exc:
                    raise ValueError("Stage 1 SFT template has more media placeholders than inputs") from exc
            elif part:
                content.append({"type": "text", "text": part})
        if next(media_iter, None) is not None:
            raise ValueError("Stage 1 SFT template has fewer media placeholders than inputs")
        result[0] = {"role": "system", "content": baseline[0]}
        result[1] = {"role": "user", "content": content}
        return result

    @staticmethod
    def _stage1_baseline_template() -> Optional[Tuple[str, str]]:
        """Read the checked-in human-readable Stage 1 SFT sample."""
        source = Path(__file__).resolve().parents[1] / "sft_v35_local_perception_dataset_reduced_pixels" / "train_sample_1.txt"
        try:
            text = source.read_text(encoding="utf-8")
            system, rest = text.split("[USER]\n", 1)
            user = rest.split("\n[ASSISTANT]", 1)[0]
            return system.removeprefix("[SYSTEM]\n").rstrip(), user.rstrip()
        except (OSError, ValueError):
            return None

    @staticmethod
    def _parse_perception_payload(text: str) -> Dict[str, Any]:
        blocks = re.findall(r"<perception>\s*(\{.*?\})\s*</perception>", text or "", re.I | re.S)
        if len(blocks) != 1:
            raise ValueError("Stage 1 deployment output must contain one perception block")
        payload = json.loads(blocks[0])
        if not isinstance(payload, dict):
            raise ValueError("Stage 1 perception must be a JSON object")
        candidates = payload.get("candidate_phases")
        if candidates is None and isinstance(payload.get("phases"), dict):
            phases = payload["phases"]
            if set(phases) != set(PHASES):
                raise ValueError("Stage 1 canonical perception must contain all four phases")
            normalized = {}
            for expected in PHASES:
                candidate = phases[expected]
                if not isinstance(candidate, dict):
                    raise ValueError(f"Stage 1 phase {expected} must be an object")
                values_v, values_q = candidate.get("v") or [0, 0], candidate.get("q") or [0, 0]
                if not isinstance(values_v, (list, tuple)) or not isinstance(values_q, (list, tuple)):
                    raise ValueError(f"Stage 1 phase {expected} has invalid v/q arrays")
                normalized[expected] = {
                    "v": [int(values_v[i] or 0) if i < len(values_v) else 0 for i in range(2)],
                    "q": [int(values_q[i] or 0) if i < len(values_q) else 0 for i in range(2)],
                    "dv": int(candidate.get("dv", 0) or 0),
                    "dq": int(candidate.get("dq", 0) or 0),
                    "age": int(candidate.get("age", 0) or 0),
                    "coord": candidate.get("coord") or {},
                }
            return {"current_phase": payload.get("current_phase"), "phases": normalized}
        if not isinstance(candidates, list) or len(candidates) != len(PHASES):
            raise ValueError("Stage 1 perception must contain four candidate phases")
        phases = {}
        for expected, candidate in zip(PHASES, candidates, strict=True):
            if not isinstance(candidate, dict) or candidate.get("signal") != expected:
                raise ValueError("Stage 1 phases must be ordered ETWT, NTST, ELWL, NLSL")
            current_v = candidate.get("current_v") or {}
            current_q = candidate.get("current_q") or {}
            movements = [expected[i:i + 2] for i in range(0, len(expected), 2)]
            phases[expected] = {
                "v": [int(current_v.get(name, 0) or 0) for name in movements],
                "q": [int(current_q.get(name, 0) or 0) for name in movements],
                "dv": int(candidate.get("demand_trend_v30_minus_v5", 0) or 0),
                "dq": int(candidate.get("queue_trend_q30_minus_q5", 0) or 0),
                "age": int(candidate.get("nonzero_v_history_length_since_last_service", 0) or 0),
                "coord": candidate.get("coordinated_arrivals") or {},
            }
        return {"current_phase": payload.get("current_phase"), "phases": phases}

    @staticmethod
    def _perception_block(text: str) -> str:
        """Return the sole Stage 1 block, excluding any post-stop spillover."""
        matches = re.findall(r"<perception>\s*\{.*?\}\s*</perception>", text or "", re.I | re.S)
        if len(matches) != 1:
            raise ValueError("Stage 1 deployment output must contain one perception block")
        return matches[0].strip()

    @staticmethod
    def _sum_numeric_values(values: Any) -> int | float | None:
        if not isinstance(values, (list, tuple)) or len(values) != 2:
            return None
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
            return None
        total = sum(values)
        return int(total) if isinstance(total, float) and total.is_integer() else total

    @staticmethod
    def _coordination_count(value: Any) -> int | float:
        if isinstance(value, bool):
            return 0
        if isinstance(value, (int, float)):
            return value
        if not isinstance(value, dict):
            return 0
        counts = []
        for movement in value.values():
            count = movement.get("count", 0) if isinstance(movement, dict) else movement
            if isinstance(count, (int, float)) and not isinstance(count, bool):
                counts.append(count)
        return sum(counts)

    @staticmethod
    def _stage2_cooperative_perception(perception: Dict[str, Any], neighbor_info: Any) -> Dict[str, Any]:
        local_coordination: Dict[str, Any] = {}
        phases = perception.get("phases", {}) if isinstance(perception, dict) else {}
        for phase in PHASES:
            phase_data = phases.get(phase, {}) if isinstance(phases, dict) else {}
            coord = phase_data.get("coord", {}) if isinstance(phase_data, dict) else {}
            local_coordination[phase] = VideoAgent._coordination_count(coord)
        if not isinstance(neighbor_info, dict):
            return {"local_coordination": local_coordination, "neighbors": {}}
        explicit = (
            neighbor_info.get("__v35_stage2_cooperative_perception__")
            or neighbor_info.get("cooperative_perception")
        )
        if isinstance(explicit, dict):
            explicit_coord = explicit.get("local_coordination")
            if isinstance(explicit_coord, dict):
                local_coordination = {
                    phase: VideoAgent._coordination_count(explicit_coord.get(phase, 0))
                    for phase in PHASES
                }
            explicit_neighbors = explicit.get("neighbors", {})
            neighbors: Dict[str, Any] = {}
            if isinstance(explicit_neighbors, dict):
                for direction, value in explicit_neighbors.items():
                    if not isinstance(value, dict):
                        continue
                    movements = value.get("upstream_movements", value.get("movements"))
                    if isinstance(movements, dict):
                        total_v = sum(
                            row.get("v", 0) for row in movements.values()
                            if isinstance(row, dict) and isinstance(row.get("v", 0), (int, float))
                            and not isinstance(row.get("v", 0), bool)
                        )
                        total_q = sum(
                            row.get("q", 0) for row in movements.values()
                            if isinstance(row, dict) and isinstance(row.get("q", 0), (int, float))
                            and not isinstance(row.get("q", 0), bool)
                        )
                    else:
                        total_v, total_q = value.get("total_v"), value.get("total_q")
                    if not all(isinstance(item, (int, float)) and not isinstance(item, bool)
                               for item in (total_v, total_q)):
                        continue
                    item = {"total_v": total_v, "total_q": total_q}
                    if "travel_time_s" in value:
                        item["travel_time_s"] = copy.deepcopy(value["travel_time_s"])
                    neighbors[str(direction).lower()] = item
            return {"local_coordination": local_coordination, "neighbors": neighbors}
        neighbors: Dict[str, Any] = {}
        entries = [
            (key, value) for key, value in neighbor_info.items()
            if not str(key).startswith("__") and isinstance(value, dict)
        ]
        order = {"north": 0, "east": 1, "south": 2, "west": 3}
        entries.sort(key=lambda pair: order.get(str(pair[0]).lower(), 99))
        for key, value in entries:
            movements = value.get("upstream_movements", value.get("movements", {}))
            direction = str(key).lower()
            if not isinstance(movements, dict):
                continue
            item = {
                "total_v": sum(row.get("v", 0) for row in movements.values() if isinstance(row, dict)),
                "total_q": sum(row.get("q", 0) for row in movements.values() if isinstance(row, dict)),
            }
            if "travel_time_s" in value:
                item["travel_time_s"] = copy.deepcopy(value["travel_time_s"])
            neighbors[direction] = item
        return {"local_coordination": local_coordination, "neighbors": neighbors}

    @staticmethod
    def _stage2_local_perception(perception: Dict[str, Any]) -> Dict[str, Any]:
        """Match the offline Stage 2 generator's local/coordinated split."""
        phases = perception.get("phases", {}) if isinstance(perception, dict) else {}
        local: Dict[str, Any] = {
            "current_phase": perception.get("current_phase") if isinstance(perception, dict) else None,
            "phases": {},
        }
        for phase in PHASES:
            phase_data = phases.get(phase, {}) if isinstance(phases, dict) else {}
            if not isinstance(phase_data, dict):
                phase_data = {}
            local["phases"][phase] = {
                "current_v": VideoAgent._sum_numeric_values(phase_data.get("v")),
                "current_q": VideoAgent._sum_numeric_values(phase_data.get("q")),
                "dv": copy.deepcopy(phase_data.get("dv")),
                "dq": copy.deepcopy(phase_data.get("dq")),
                "unserved_age": copy.deepcopy(phase_data.get("age")),
            }
        return local

    @staticmethod
    def _cooperative_field_meanings() -> str:
        return """Cooperative-field meanings:
- local_coordination contains one potential-arrival count for each target
  phase. It is direct future-pressure evidence: selecting that phase for the
  next service interval can serve this approaching pressure as it reaches the
  target intersection. These approaching vehicles are not yet included in the
  target intersection's local perception. Keep local_coordination separate
  from current_v, and never interpret it as or add it to current_q. Use it only
  as supplementary future-arrival pressure for the corresponding phase.
- neighbors contains broad directional pressure from actual neighboring
  intersections. For each included direction, total_v and total_q already sum
  the available upstream movements at that neighbor. Do not infer a specific
  target movement from these totals, and do not treat them as vehicles that
  will definitely arrive or as direct phase-level local_coordination.
- Use neighbors only as broad directional context, because their eventual
  releases depend on their unknown next signal decisions. North and south
  neighbor pressure can increase the general priority of the target's
  north-south phase group (NTST and NLSL). East and west neighbor pressure can
  increase the general priority of the target's east-west phase group (ETWT
  and ELWL). Within each group, choose using the target's own local demand,
  queue, trends, unserved_age, and local_coordination.
- travel_time_s estimates when broad neighbor pressure may reach the target.
  Compare it with the next 5-second transition plus 25-second green interval;
  pressure with a longer travel time should have less influence.
- Only directions with an actual neighboring intersection are included. Missing
  directions are boundary directions and should be ignored.
"""

    def _formal_stage2_template(self) -> str:
        """Load the SFT Stage-2 prompt and convert its embedded examples to slots.

        The flattened prompt is the deployment default and retains the formal
        fast/slow output grammar. A configured path allows controlled prompt
        experiments without changing routing or decision code.
        """
        configured = str(getattr(self, "vlm_runtime_config", {}).get("VLM_STAGE2_PROMPT_PATH", "") or "").strip()
        source = Path(configured).expanduser() if configured else (
            Path(__file__).resolve().parents[1]
            / "stage2_prompt_flattened_experiment"
            / "stage2_cooperative_decision_prompt_formal.txt"
        )
        if not source.is_file():
            raise FileNotFoundError(f"formal Stage 2 prompt is missing: {source}")
        template = source.read_text(encoding="utf-8").strip()
        if (
            "{{LOCAL_PERCEPTION_JSON}}" not in template
            or "{{COOPERATIVE_PERCEPTION_BLOCK}}" not in template
            or "{{COOPERATIVE_FIELD_MEANINGS}}" not in template
        ):
            raise ValueError("SFT Stage 2 formal prompt is missing perception placeholders")
        return template

    def _stage2_messages(self, perception: Dict[str, Any], neighbor_info: Any) -> List[Dict[str, Any]]:
        """Build the byte-compatible Stage 2 text contract from routed JSON."""
        cooperative = self._stage2_cooperative_perception(perception, neighbor_info)
        local = self._stage2_local_perception(perception)
        local_json = json.dumps(local, ensure_ascii=False, separators=(",", ":"))
        if self.stage2_include_cooperation:
            cooperative_json = json.dumps(
                cooperative, ensure_ascii=False, separators=(",", ":")
            )
            cooperative_block = "<cooperative_perception>\n" + cooperative_json + "\n</cooperative_perception>"
            cooperative_meanings = self._cooperative_field_meanings()
        else:
            cooperative_block = ""
            cooperative_meanings = ""
        user = self._formal_stage2_template()
        user = user.replace("{{LOCAL_PERCEPTION_JSON}}", local_json)
        user = user.replace("{{COOPERATIVE_PERCEPTION_BLOCK}}", cooperative_block)
        user = user.replace("{{COOPERATIVE_FIELD_MEANINGS}}", cooperative_meanings)
        return [{"role": "system", "content": DEPLOYMENT_STAGE2_SYSTEM}, {"role": "user", "content": user}]

    def _call_deployment_two_stage(
        self, messages: List[Dict[str, Any]], neighbor_info: Any, step_num: int
    ) -> Tuple[Dict[str, Any], str, Dict[str, Any]]:
        """Run perception, binary mode routing, and decision continuation."""
        stage1_prompt_messages = self._stage1_messages(messages)
        stage1 = self._call_api(
            stage1_prompt_messages, step_num,
            overrides={"temperature": 0.0, "stop": ["</perception>"],
                       "include_stop_str_in_output": True, "max_tokens": self.max_tokens},
            debug_label="stage1_perception",
        )
        # Some OpenAI-compatible servers still return text after a stop string.
        # Keep only the structured Stage 1 result so it cannot duplicate mode or
        # signal tags in the final combined response.
        perception_text = self._perception_block(self._assistant_text(stage1))
        perception = self._parse_perception_payload(perception_text)
        stage2_messages = self._stage2_messages(perception, neighbor_info)
        routed_body, routed_text, diagnostic = self._call_with_adaptive_mode_routing(
            stage2_messages, step_num
        )
        decision_text = routed_text if routed_text is not None else self._assistant_text(routed_body)
        # Stage 2 receives perception as input and must not reproduce it.  A
        # permissive server may nevertheless echo the block; remove that echo
        # before composing the canonical deployment response.
        decision_text = re.sub(
            r"\s*<perception>.*?</perception>\s*", "\n", decision_text or "", flags=re.I | re.S
        )
        text = perception_text + "\n" + decision_text.lstrip()
        diagnostic = {**diagnostic, "deployment_two_stage": True,
                      "stage1_prompt": stage1_prompt_messages,
                      "stage1_response": perception_text,
                      "stage1_perception": perception,
                      "stage2_prompt": stage2_messages[1]["content"],
                      "stage2_response": decision_text.strip()}
        return routed_body, text, diagnostic

    def _call_api(self, messages: List[Dict[str, Any]], step_num: int, *,
                  overrides: Optional[Dict[str, Any]] = None,
                  debug_label: str = "response") -> Dict[str, Any]:
        payload: Dict[str, Any] = {"model": self.model_name, "messages": messages,
                                  "temperature": self.temperature, "max_tokens": self.max_tokens}
        if overrides:
            payload.update(overrides)
        if self.seed is not None:
            payload["seed"] = int(self.seed)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        error: Optional[Exception] = None
        for attempt in range(self.retries + 1):
            try:
                response = requests.post(self.api_url, headers=headers, json=payload, timeout=self.timeout)
                response.raise_for_status()
                body = response.json()
                Path(self.api_debug_dir, f"{self.tls_id}_step{step_num}_{debug_label}.json").write_text(
                    json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
                return body
            except Exception as exc:
                error = exc
                if attempt < self.retries:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"VLM API failed after {self.retries + 1} attempts: {error}")

    @staticmethod
    def _assistant_text(body: Dict[str, Any]) -> str:
        content = body["choices"][0]["message"].get("content") or ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(item.get("text", "") for item in content if isinstance(item, dict))
        return str(content)

    @staticmethod
    def _completion_token_ids(body: Dict[str, Any]) -> List[int]:
        if not body.get("choices"):
            return []
        token_ids = body["choices"][0].get("token_ids") or []
        return [int(token_id) for token_id in token_ids]

    @staticmethod
    def _mode_top_logprobs(body: Dict[str, Any], generated_mode: str) -> Dict[str, float]:
        try:
            content = body["choices"][0]["logprobs"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("API response is missing generated-token top_logprobs") from exc
        mode_item = next(
            (
                item for item in reversed(content)
                if str(item.get("token", "")).strip().lower() == generated_mode
            ),
            None,
        )
        if not isinstance(mode_item, dict):
            raise ValueError("could not align the generated mode token with its top_logprobs")
        candidates = mode_item.get("top_logprobs")
        if not isinstance(candidates, list):
            raise ValueError("generated mode token has no top_logprobs")
        scores: Dict[str, float] = {}
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            token = str(candidate.get("token", "")).strip().lower()
            logprob = candidate.get("logprob")
            if token in {"fast", "slow"} and isinstance(logprob, (int, float)):
                scores[token] = float(logprob)
        if set(scores) != {"fast", "slow"}:
            raise ValueError("fast and slow are not both present in next-token top_logprobs")
        return scores

    def _write_mode_routing_diagnostic(self, diagnostic: Dict[str, Any]) -> None:
        path = str(self.vlm_runtime_config.get("VLM_MODE_ROUTING_LOG", "") or "").strip()
        if not path:
            return
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(diagnostic, ensure_ascii=False) + "\n")

    def _mode_routing_threshold(self) -> float:
        """Read the deployment-calibrated slow-mode threshold defensively."""
        default = float(getattr(self, "mode_threshold_default", 0.5))
        path = str(getattr(self, "mode_threshold_path", "") or "").strip()
        if not path:
            return min(1.0, max(0.0, default))
        try:
            value = float(json.loads(Path(path).read_text(encoding="utf-8"))["threshold"])
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            value = default
        return min(1.0, max(0.0, value))

    def _call_with_adaptive_mode_routing(
        self, messages: List[Dict[str, Any]], step_num: int
    ) -> Tuple[Dict[str, Any], Optional[str], Dict[str, Any]]:
        """Generate perception, deterministically route mode, then continue the answer."""
        diagnostic: Dict[str, Any] = {
            "enabled": True,
            "step": int(step_num),
            "threshold": self._mode_routing_threshold(),
            "tie_break": "fast",
            "requested_mode": self.reasoning_mode,
        }
        try:
            if self.reasoning_mode in {"fast", "slow"}:
                # Forced ablations bypass the classifier call entirely.  This
                # makes their token/latency cost comparable to the selected
                # continuation branch, rather than charging for adaptive routing.
                selected_mode = self.reasoning_mode
                routed_prefix = f"<mode>{selected_mode}</mode>"
                suffix_messages = list(messages) + [
                    {"role": "assistant", "content": routed_prefix}
                ]
                suffix_body = self._call_api(
                    suffix_messages,
                    step_num,
                    overrides={
                        "max_tokens": self.max_tokens,
                        "continue_final_message": True,
                        "add_generation_prompt": False,
                        "stop": ["</signal>"],
                        "include_stop_str_in_output": True,
                    },
                    debug_label=f"forced_{selected_mode}",
                )
                suffix = self._assistant_text(suffix_body)
                text = routed_prefix + suffix
                diagnostic.update({
                    "decode": "forced_mode_continuation",
                    "route_method": "cli_override",
                    "selected_mode": selected_mode,
                    "adaptive_selected_mode": None,
                    "mode_override": True,
                    "prefix_token_count": 0,
                    "candidate_token_count": 0,
                    "suffix_max_tokens": self.max_tokens,
                })
                self._write_mode_routing_diagnostic(diagnostic)
                return suffix_body, text, diagnostic
            prefix_body = self._call_api(
                messages,
                step_num,
                overrides={
                    "temperature": 0.0,
                    "max_tokens": self.max_tokens,
                    "stop": ["</mode>"],
                    "include_stop_str_in_output": True,
                    "logprobs": True,
                    "top_logprobs": 20,
                },
                debug_label="perception_mode",
            )
            generated_prefix = self._assistant_text(prefix_body).strip()
            if not generated_prefix.lower().endswith("</mode>"):
                raise ValueError("generated prefix did not close with </mode>")
            modes = re.findall(r"<mode>\s*(fast|slow)\s*</mode>", generated_prefix, re.I)
            if len(modes) != 1:
                raise ValueError("generated prefix must contain exactly one valid mode block")
            generated_mode = modes[0].lower()
            mode_scores = self._mode_top_logprobs(prefix_body, generated_mode)
            fast_logprob = mode_scores["fast"]
            slow_logprob = mode_scores["slow"]
            maximum = max(fast_logprob, slow_logprob)
            fast_weight = math.exp(fast_logprob - maximum)
            slow_weight = math.exp(slow_logprob - maximum)
            p_slow = slow_weight / (fast_weight + slow_weight)

            # The threshold is trained/calibrated from paired FAST/SLOW
            # utilities.  Strict comparison preserves the documented FAST
            # tie-break at p(slow)==threshold.
            adaptive_mode = "slow" if p_slow > diagnostic["threshold"] else "fast"
            selected_mode = (
                adaptive_mode
                if self.reasoning_mode == "adaptive"
                else self.reasoning_mode
            )
            routed_prefix = re.sub(
                r"<mode>\s*(?:fast|slow)\s*</mode>",
                f"<mode>{selected_mode}</mode>",
                generated_prefix,
                count=1,
                flags=re.I,
            )
            suffix_messages = list(messages) + [{"role": "assistant", "content": routed_prefix}]
            prefix_token_count = len(self._completion_token_ids(prefix_body))
            candidate_token_count = 1
            suffix_max_tokens = max(
                1, self.max_tokens - prefix_token_count
            )
            suffix_body = self._call_api(
                suffix_messages,
                step_num,
                overrides={
                    "max_tokens": suffix_max_tokens,
                    "continue_final_message": True,
                    "add_generation_prompt": False,
                    "stop": ["</signal>"],
                    "include_stop_str_in_output": True,
                },
                debug_label=f"suffix_{selected_mode}",
            )
            suffix = self._assistant_text(suffix_body)
            text = routed_prefix + suffix
            diagnostic.update({
                "decode": "two_stage_generated_mode_top_logprobs",
                "route_method": "argmax",
                "generated_mode": generated_mode,
                "selected_mode": selected_mode,
                "adaptive_selected_mode": adaptive_mode,
                "mode_override": self.reasoning_mode != "adaptive",
                "p_slow_model": p_slow,
                "p_fast_model": 1.0 - p_slow,
                "fast_logprob": fast_logprob,
                "slow_logprob": slow_logprob,
                "fast_token_logprobs": [fast_logprob],
                "slow_token_logprobs": [slow_logprob],
                "prefix_token_count": prefix_token_count,
                "candidate_token_count": candidate_token_count,
                "suffix_max_tokens": suffix_max_tokens,
            })
            self._write_mode_routing_diagnostic(diagnostic)
            return suffix_body, text, diagnostic
        except Exception as exc:
            diagnostic.update({
                "decode": "one_stage_fallback",
                "routing_fallback_reason": f"{type(exc).__name__}: {exc}",
            })
            fallback = self._call_api(messages, step_num, debug_label="one_stage_fallback")
            self._write_mode_routing_diagnostic(diagnostic)
            return fallback, None, diagnostic

    @staticmethod
    def _parse_signal(text: str) -> str:
        signals = re.findall(r"<signal>\s*(ETWT|NTST|ELWL|NLSL)\s*</signal>", text, re.I)
        if len(signals) != 1:
            raise ValueError("response must contain exactly one signal block")
        return signals[0].upper()

    @staticmethod
    def _parse_perception(text: str) -> Dict[str, int]:
        perception_blocks = re.findall(r"<perception>\s*(.*?)\s*</perception>", text, re.I | re.S)
        if len(perception_blocks) != 1:
            raise ValueError("response must contain exactly one perception block")
        perception = json.loads(perception_blocks[0])
        candidates = perception.get("candidate_phases")
        if candidates is None and isinstance(perception.get("phases"), dict):
            normalized = VideoAgent._parse_perception_payload(text)
            return {
                phase: sum(int(value or 0) for value in (normalized["phases"][phase].get("v") or []))
                for phase in PHASES
            }
        if not isinstance(candidates, list) or len(candidates) != len(PHASES):
            raise ValueError("perception.candidate_phases must contain exactly four entries")
        current_v: Dict[str, int] = {}
        for expected_phase, candidate in zip(PHASES, candidates, strict=True):
            if not isinstance(candidate, dict):
                raise ValueError("every perception candidate must be an object")
            phase = candidate.get("signal")
            if phase not in PHASES or phase in current_v:
                raise ValueError("perception candidates must contain each phase exactly once")
            if phase != expected_phase:
                raise ValueError("perception candidates must be ordered ETWT, NTST, ELWL, NLSL")
            value = candidate.get("current_v", {})
            if not isinstance(value, dict) or "total" not in value:
                raise ValueError(f"perception candidate {phase} is missing current_v.total")
            total = value["total"]
            if isinstance(total, bool) or not isinstance(total, (int, float)) or total < 0:
                raise ValueError(f"perception candidate {phase} has invalid current_v.total")
            current_v[phase] = int(total)
        if set(current_v) != set(PHASES):
            raise ValueError("perception candidates must contain ETWT, NTST, ELWL, and NLSL")
        return current_v

    @classmethod
    def _parse_output(cls, text: str) -> Tuple[str, Dict[str, int]]:
        return cls._parse_signal(text), cls._parse_perception(text)

    @staticmethod
    def _phase_movements(phase: str) -> List[str]:
        return [phase[i:i + 2] for i in range(0, len(phase), 2)]

    @classmethod
    def _phase_total(cls, values: Any, phase: str) -> int:
        total = 0
        for movement in cls._phase_movements(phase):
            index = MOVEMENT_MAP.get(movement)
            if index is not None and isinstance(values, (list, tuple)) and index < len(values):
                try:
                    total += len(values[index]) if isinstance(values[index], (list, tuple, set, dict)) else int(values[index])
                except (TypeError, ValueError):
                    pass
        return max(0, total)

    def _sumo_v25_state(self, env) -> Optional[Tuple[Dict[str, int], Dict[str, List[int]]]]:
        """Read the live intersection features used by V25 from SUMO."""
        state = None
        for inter in getattr(env, "list_intersection", []) or []:
            if getattr(inter, "inter_id", None) == self.tls_id:
                state = getattr(inter, "dic_feature", None)
                break
        if not isinstance(state, dict):
            return None
        movement_values = state.get("traffic_movement_vehicle_ids_150m")
        if movement_values is None:
            movement_values = state.get("traffic_movement_vehicle_ids")
        if movement_values is None:
            movement_values = state.get("lane_num_vehicle")
        if movement_values is None:
            movement_values = state.get("traffic_movement_pressure_queue")
        if not isinstance(movement_values, (list, tuple)):
            return None
        current = {phase: self._phase_total(movement_values, phase) for phase in PHASES}
        raw_history = state.get("v9_cycle_150m_history") or []
        histories = {phase: [] for phase in PHASES}
        if isinstance(raw_history, (list, tuple)):
            for snapshot in raw_history:
                for phase in PHASES:
                    histories[phase].append(self._phase_total(snapshot, phase))
        return current, histories

    def _fallback(self, env) -> int:
        # V25-equivalent fallback: Current V first, then accumulated demand
        # history, then the fixed phase order. Do not use SUMO's
        # best_action_idx/MaxPressure shortcut here.
        live = self._sumo_v25_state(env)
        if live is not None:
            current_by_phase, cycle_histories = live
        else:
            current_by_phase = dict(self._last_current_v)
            cycle_histories = self._current_v_history
        scores = []
        for index, phase in enumerate(self.phase_list):
            current_v = int(current_by_phase.get(phase, 0) or 0)
            values = cycle_histories.get(phase, [])
            delta_v = int(values[-1] - values[0]) if len(values) >= 2 else 0
            history_len = int(self._history.get(phase, 0) or 0)
            scores.append((current_v, delta_v, history_len, -index))
        return max(range(self.num_phases), key=lambda index: scores[index])

    def run_stage1(self, image_paths: Any, neighbor_info: Any = None,
                   step_num: int = 0) -> Dict[str, Any]:
        """Run only the multimodal perception stage for one intersection.

        This is the first half of the global two-stage deployment barrier.  It
        intentionally does not call the Stage 2 router or choose a signal.
        """
        videos = self._video_paths(image_paths)
        frames = self._coordination_frames(image_paths, neighbor_info, step_num)
        messages = self._messages(videos, frames)
        prompt_messages = self._stage1_messages(messages)
        body = self._call_api(
            prompt_messages, step_num,
            overrides={"temperature": 0.0, "stop": ["</perception>"],
                       "include_stop_str_in_output": True,
                       "max_tokens": self.max_tokens},
            debug_label="stage1_perception",
        )
        response = self._perception_block(self._assistant_text(body))
        perception = self._parse_perception_payload(response)
        return {
            "perception": perception,
            "response": response,
            "prompt": prompt_messages,
            "videos": videos,
            "frames": frames,
        }

    def run_stage2(self, perception: Dict[str, Any], cooperative_perception: Any,
                   step_num: int = 0) -> Dict[str, Any]:
        """Run only the text decision stage after global routing."""
        messages = self._stage2_messages(perception, {
            "__v35_stage2_cooperative_perception__": cooperative_perception,
        })
        body, routed_text, diagnostic = self._call_with_adaptive_mode_routing(
            messages, step_num
        )
        decision = routed_text if routed_text is not None else self._assistant_text(body)
        decision = re.sub(r"\s*<perception>.*?</perception>\s*", "\n", decision,
                          flags=re.I | re.S).strip()
        fallback = None
        try:
            signal = self._parse_signal(decision)
        except ValueError as exc:
            # Stage 2 schema failure only: preserve V25's deterministic
            # pressure ordering without hiding API/transport failures.
            scores = []
            phases = perception.get("phases", {}) if isinstance(perception, dict) else {}
            for index, phase in enumerate(self.phase_list):
                item = phases.get(phase, {}) if isinstance(phases, dict) else {}
                values = item.get("v", []) if isinstance(item, dict) else []
                current_v = sum(int(value or 0) for value in values[:2]) if isinstance(values, (list, tuple)) else 0
                delta_v = int(item.get("dv", 0) or 0) if isinstance(item, dict) else 0
                age = int(item.get("age", 0) or 0) if isinstance(item, dict) else 0
                scores.append((current_v, delta_v, age, -index))
            action = max(range(len(self.phase_list)), key=lambda i: scores[i])
            signal = self.phase_list[action]
            fallback = {
                "source": "v25_perception_fallback",
                "reason": f"{type(exc).__name__}: {exc}",
                "step": int(step_num),
            }
            diagnostic = {**diagnostic, "stage2_fallback": fallback}
        return {
            "response": decision,
            "body": body,
            "prompt": messages,
            "diagnostic": diagnostic,
            "signal": signal,
            "fallback": fallback,
        }

    @staticmethod
    def _prompt_log_text(prompt: Any) -> str:
        """Render a prompt for logs without dumping base64 media payloads."""
        if not isinstance(prompt, list):
            return str(prompt or "")
        sections: List[str] = []
        for message in prompt:
            if not isinstance(message, dict):
                sections.append(str(message))
                continue
            role = message.get("role", "unknown")
            content = message.get("content", "")
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                parts: List[str] = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            parts.append(str(item.get("text", "")))
                        else:
                            parts.append(f"[{item.get('type', 'media')} omitted from log]")
                    else:
                        parts.append(str(item))
                text = "\n".join(parts)
            else:
                text = str(content)
            sections.append(f"[{role}]\n{text}")
        return "\n\n".join(sections)

    def write_two_stage_logs(self, step_num: int, stage1: Dict[str, Any],
                             stage2: Dict[str, Any], cooperative_perception: Any,
                             videos: Optional[Dict[str, str]] = None,
                             frames: Optional[List[Tuple[str, str]]] = None) -> None:
        """Persist transcripts for the global two-stage deployment barrier.

        ``choose_action`` is not used by the barrier path, so its legacy log
        block cannot record these calls.  Keep this writer on the agent so both
        paths use the same per-session ``conversations/stage*`` directories.
        """
        try:
            os.makedirs(self.stage1_conversations_dir, exist_ok=True)
            os.makedirs(self.stage2_conversations_dir, exist_ok=True)
            videos = videos or {}
            frames = frames or []
            with open(self.stage1_log_path, "a", encoding="utf-8") as handle:
                handle.write("=" * 88 + "\n")
                handle.write(f"Step: {step_num}\nStage: 1 perception\n")
                if videos:
                    handle.write("Videos:\n")
                    for direction in VIDEO_DIRECTIONS:
                        handle.write(f"  {direction}: {videos.get(direction, '')}\n")
                if frames:
                    handle.write("Coordination frames:\n")
                    for direction, path in frames:
                        handle.write(f"  {direction}: {path}\n")
                handle.write("Prompt:\n")
                handle.write(self._prompt_log_text(stage1.get("prompt")) + "\n")
                handle.write("Response:\n")
                handle.write(str(stage1.get("response", "")) + "\n")
                handle.write("Perception:\n")
                handle.write(json.dumps(stage1.get("perception", {}), ensure_ascii=False) + "\n")
                handle.write("Fallback:\n")
                handle.write(json.dumps(stage1.get("fallback"), ensure_ascii=False) + "\n\n")

            with open(self.stage2_log_path, "a", encoding="utf-8") as handle:
                handle.write("=" * 88 + "\n")
                handle.write(f"Step: {step_num}\nStage: 2 routed decision\n")
                handle.write("Cooperative perception:\n")
                handle.write(json.dumps(cooperative_perception, ensure_ascii=False) + "\n")
                handle.write("Prompt:\n")
                handle.write(self._prompt_log_text(stage2.get("prompt")) + "\n")
                handle.write("Response:\n")
                handle.write(str(stage2.get("response", "")) + "\n")
                handle.write(f"Signal: {stage2.get('signal', '')}\n")
                handle.write("Fallback:\n")
                handle.write(json.dumps(stage2.get("fallback"), ensure_ascii=False) + "\n\n")
        except Exception as exc:
            # A diagnostic transcript must never abort a simulation step.
            print(f"[VIDEO_AGENT_LOG_WARNING] tls={self.tls_id} step={step_num}: {exc}", flush=True)

    def _update_history(self, current_v: Dict[str, int], action: int) -> None:
        selected = self.phase_list[action]
        for phase in PHASES:
            if phase == selected:
                self._history[phase] = 0
            elif current_v.get(phase, 0) > 0:
                self._history[phase] += 1
        self._last_current_v = dict(current_v)
        for phase in PHASES:
            values = self._current_v_history.setdefault(phase, [])
            values.append(int(current_v.get(phase, 0) or 0))
            del values[:-6]

    def choose_action(self, image_paths, env, step_num: int = 0, neighbor_info=None) -> int:
        self._last_api_error = None
        messages: List[Dict[str, Any]] = []
        videos: Dict[str, str] = {}
        frames: List[Tuple[str, str]] = []
        try:
            videos = self._video_paths(image_paths)
            frames = self._coordination_frames(image_paths, neighbor_info, step_num)
            messages = self._messages(videos, frames)
            if getattr(self, "deployment_two_stage", False):
                body, text, mode_routing = self._call_deployment_two_stage(
                    messages, neighbor_info, step_num
                )
                self._last_mode_routing = mode_routing
                source = (
                    "deployment_two_stage_route"
                    if mode_routing.get("decode") != "one_stage_fallback"
                    else "deployment_two_stage_router_fallback"
                )
            elif self.adaptive_mode_routing:
                body, routed_text, mode_routing = self._call_with_adaptive_mode_routing(
                    messages, step_num
                )
                text = routed_text if routed_text is not None else self._assistant_text(body)
                self._last_mode_routing = mode_routing
                source = (
                    "adaptive_mode_route"
                    if routed_text is not None
                    else "adaptive_mode_one_stage_fallback"
                )
            else:
                body = self._call_api(messages, step_num)
                text = self._assistant_text(body)
                self._last_mode_routing = {}
                source = "final_answer"
            try:
                signal = self._parse_signal(text)
                action = self.phase_list.index(signal)
            except Exception as exc:
                self._last_api_error = f"signal parse: {type(exc).__name__}: {exc}"
                action = self._fallback(env)
                source = "v25_fallback"
            try:
                current_v = self._parse_perception(text)
            except Exception as exc:
                perception_error = f"perception parse: {type(exc).__name__}: {exc}"
                self._last_api_error = (self._last_api_error + "; " if self._last_api_error else "") + perception_error
                live = self._sumo_v25_state(env)
                current_v = live[0] if live is not None else dict(self._last_current_v)
        except Exception as exc:
            self._last_api_error = f"{type(exc).__name__}: {exc}"
            text = ""
            live = self._sumo_v25_state(env)
            current_v = live[0] if live is not None else dict(self._last_current_v)
            action = self._fallback(env)
            source = "v25_fallback"
        self.current_phase_idx = action
        self.action_history.append((step_num, action))
        self._update_history(current_v, action)
        self._last_decision_text = text
        self._last_decision_source = source
        self._last_decision_record = {"step": step_num, "action": action,
                                      "phase": self.phase_list[action], "source": source,
                                      "error": self._last_api_error,
                                      "mode_routing": copy.deepcopy(self._last_mode_routing)}
        with open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write("=" * 88 + "\n")
            handle.write(f"Step: {step_num}\n")
            handle.write(f"Decision source: {source}\n")
            handle.write(f"Selected action: {action} ({self.phase_list[action]})\n")
            if self._last_api_error:
                handle.write(f"Fallback error: {self._last_api_error}\n")
            if self._last_mode_routing:
                handle.write("Mode routing: " + json.dumps(
                    self._last_mode_routing, ensure_ascii=False
                ) + "\n")
            if videos:
                handle.write("Videos:\n")
                for direction in VIDEO_DIRECTIONS:
                    handle.write(f"  {direction}: {videos.get(direction, '')}\n")
            if frames:
                handle.write("Coordination frames:\n")
                for direction, path in frames:
                    handle.write(f"  {direction}: {path}\n")
            if messages:
                handle.write("\nSystem prompt:\n")
                handle.write(str(messages[0].get("content", "")) + "\n")
                user_content = messages[1].get("content", [])
                user_text = next(
                    (item.get("text", "") for item in user_content
                     if isinstance(item, dict) and item.get("type") == "text"),
                    "",
                )
                handle.write("\nUser prompt:\n")
                handle.write(user_text + "\n")
            handle.write("Model response:\n")
            handle.write((text or "<no model response; fallback used>") + "\n\n")
        # Keep auditable stage-specific transcripts separate from the legacy
        # combined conversation log.  Stage 1 contains video perception only;
        # Stage 2 contains routed JSON and the decision continuation only.
        routing = self._last_mode_routing or {}
        stage1_prompt = routing.get("stage1_prompt")
        stage2_prompt = routing.get("stage2_prompt")
        if stage1_prompt or routing.get("stage1_response"):
            with open(self.stage1_log_path, "a", encoding="utf-8") as handle:
                handle.write("=" * 88 + "\n")
                handle.write(f"Step: {step_num}\n")
                handle.write("Stage: 1 perception\n")
                if videos:
                    handle.write("Videos:\n")
                    for direction in VIDEO_DIRECTIONS:
                        handle.write(f"  {direction}: {videos.get(direction, '')}\n")
                handle.write("Prompt:\n")
                handle.write(json.dumps(stage1_prompt, ensure_ascii=False) + "\n")
                handle.write("Response:\n")
                handle.write(str(routing.get("stage1_response", "")) + "\n")
        if stage2_prompt or routing.get("stage2_response"):
            with open(self.stage2_log_path, "a", encoding="utf-8") as handle:
                handle.write("=" * 88 + "\n")
                handle.write(f"Step: {step_num}\n")
                handle.write("Stage: 2 routed decision\n")
                handle.write("Prompt:\n")
                handle.write(str(stage2_prompt or "") + "\n")
                handle.write("Response:\n")
                handle.write(str(routing.get("stage2_response", "")) + "\n")
        return action

    def get_last_decision_record(self) -> Dict[str, Any]:
        """Return the latest decision in the interface expected by V35 runners."""
        return copy.deepcopy(self._last_decision_record)


V35VideoAgent = VideoAgent
