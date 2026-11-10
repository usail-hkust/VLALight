"""
Pure SUMO-Only Traffic Signal Control Agent

ARCHITECTURE:
  - Single-stage: SUMO telemetry data -> LLM decision
  - NO perception stage, NO image processing
  - Direct JSON-based input from SUMO simulation interface

INPUT:
  - SUMO telemetry window (vehicle counts, waiting queues per lane)
  - Historical decision context
  - Phase availability and timing constraints

OUTPUT:
  - Single traffic signal phase decision
  - Based purely on numerical traffic data analysis

This module provides a clean text-only baseline for comparing against
vision-based agents, using only structured simulation data as input.
"""

import copy
import hashlib
import json
import os
import random
import re
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple
import requests


def _extract_waiting_time(wait_info: Any) -> float:
    if isinstance(wait_info, dict):
        return float(wait_info.get('time', 0.0))
    if isinstance(wait_info, (int, float)):
        return float(wait_info)
    return 0.0


class SUMOOnlyAgent:
    """
    ===============================================
    NO PERCEPTION STAGE - SUMO-ONLY AGENT
    ===============================================

    Single-stage decision agent that operates purely on SUMO telemetry data:
      - Input: SUMO vehicle counts and queue data (JSON format)
      - Processing: LLM analysis of numerical traffic patterns
      - Output: Direct signal phase decision

    NO image processing, NO vision models, NO perception stage.
    Pure text-based traffic signal control using simulation telemetry.
    """

    # Canonical 12-slot lane names matching dic_feature ordering:
    # W[0-2], E[3-5], N[6-8], S[9-11]; within each approach: left/straight/right
    _SLOT_NAMES: Tuple[str, ...] = (
        'W_left', 'W_straight', 'W_right',
        'E_left', 'E_straight', 'E_right',
        'N_left', 'N_straight', 'N_right',
        'S_left', 'S_straight', 'S_right',
    )

    _SUMO_WINDOW_SIZE: int = 35      # number of simulation steps per decision window
    _SUMO_SAMPLE_INTERVAL: int = 5   # sample every N steps for the frame sequence
    _CAMERA_VIEW_DISTANCE: float = 150.0  # metres from stop-line visible to camera
    _DEFAULT_VEH_SPEED: float = 11.0
    _QUEUE_SPEED_THRESHOLD: float = 0.1



    def __init__(self,
                 tls_id: str,
                 phase_list: list,
                 api_url: str = '',
                 api_key: str = '',
                 model_name: str = '',
                 log_dir: str = '',
                 scenario: str = 'jinan',
                 tool_executor=None,  # Unused in SUMO-only mode, kept for interface compatibility
                 vlm_runtime_config: Optional[Dict[str, Any]] = None,
                 conversations_dir: str = '',
                 api_debug_dir: str = '',
                 seed: Optional[int] = None):

        self.tls_id = tls_id
        self.phase_list = phase_list
        self.num_phases = len(phase_list)
        self.api_url = api_url
        self.api_key = api_key
        self.model_name = model_name
        self.log_dir = log_dir
        self.scenario = scenario
        self.tool_executor = tool_executor
        self.vlm_runtime_config = vlm_runtime_config or {}
        configured_seed = self.vlm_runtime_config.get('SEED', seed)
        self.seed = None if configured_seed is None else int(configured_seed)
        if self.seed is None:
            self._rng = random.Random()
        else:
            seed_material = f'{self.seed}:{self.tls_id}'.encode('utf-8')
            agent_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], 'big')
            self._rng = random.Random(agent_seed)

        # ========================================================================
        # PERCEPTION STAGE PERMANENTLY DISABLED - SUMO-ONLY MODE
        # ========================================================================
        # This agent ALWAYS skips perception and uses only SUMO telemetry data.
        # No perception API configuration needed.
        # ========================================================================
        self.skip_perception_stage: bool = True  # Permanently true - no image processing

        self.decision_api_url = self.vlm_runtime_config.get('DECISION_API_URL', api_url)
        self.decision_api_key = self.vlm_runtime_config.get('DECISION_API_KEY', api_key)
        self.decision_model = self.vlm_runtime_config.get('DECISION_MODEL', model_name)
        self.decision_temperature = self.vlm_runtime_config.get('DECISION_TEMPERATURE', 0.2)
        self.decision_max_tokens = self.vlm_runtime_config.get('DECISION_MAX_TOKENS', 1200)
        self.decision_enable_thinking = bool(
            self.vlm_runtime_config.get('DECISION_ENABLE_THINKING', False)
        )
        # Image-related configurations removed (not needed for SUMO-only mode)
        self.current_phase_idx = 0
        self.action_history: List[tuple] = []
        self._last_phase_totals: Dict[str, int] = {}  # Vehicle counts from last step per phase
        self._last_chosen_phase: str = ''  # Phase chosen at last step
        self._phase_v_since_served: Dict[str, List[int]] = {}  # V history per phase since last served
        self._latest_coordination: Dict[str, Any] = {}
        self._latest_neighbor_info: Dict[str, Any] = {}
        self._last_decision_messages: List[Dict[str, Any]] = []
        self._last_decision_source: str = 'final_answer'
        self._last_decision_response_json: Optional[Dict[str, Any]] = None
        self._last_decision_text: str = ''
        self._last_decision_reasoning: str = ''
        self._last_api_error: Optional[str] = None
        self._last_decision_record: Dict[str, Any] = {}
        self._decision_step_count: int = 0
        self._latest_agent_state: Dict[str, Any] = {}

        self.conversations_dir = conversations_dir or os.path.join(log_dir, 'conversations')
        self.api_debug_dir = api_debug_dir or os.path.join(log_dir, 'api_debug')
        os.makedirs(self.conversations_dir, exist_ok=True)
        os.makedirs(self.api_debug_dir, exist_ok=True)
        self.log_file_path = os.path.join(self.conversations_dir, f'{self.tls_id}_sumo_only.txt')
        # Write file header once at init
        # Experiment variant: SUMO-only agent (no perception stage, no image processing)
        experiment_variant = 'sumo_only_agent'  # Always SUMO-only mode
        with open(self.log_file_path, 'w', encoding='utf-8') as f:
            f.write(f'[SUMO-ONLY] Traffic Control Agent Log -- {self.tls_id}\n')
            f.write(f'{"=" * 72}\n')
            f.write(f'Scenario:           {self.scenario}\n')
            f.write(f'Agent type:         {experiment_variant}\n')
            f.write(f'Experiment seed:     {self.seed}\n')
            f.write(f'Available phases:   {self.phase_list}\n')
            f.write(f'Decision model:     {self.decision_model}\n')
            f.write(f'Input data source:  SUMO telemetry only (no images)\n')
            f.write(f'Perception stage:   DISABLED (text-only mode)\n')
            f.write(f'{"=" * 72}\n\n')

        # SUMO telemetry sliding window: each frame is a dict with keys
        #   'step', 'num_vehicle'[12], 'num_waiting'[12]
        # Both lists use the SEMANTIC 12-slot ordering:
        #   W[0-2], E[3-5], N[6-8], S[9-11], within each: left / straight / right
        # Source fields:
        #   traffic_movement_pressure_num   -> num_vehicle  (total vehicles, semantic order)
        #   traffic_movement_pressure_queue -> num_waiting  (halting vehicles, semantic order)
        # NOT lane_num_vehicle / lane_num_waiting_vehicle_in -- those follow sorted-lane-ID
        # order and do NOT guarantee left/straight/right alignment.
        sumo_window_size = int(
            self.vlm_runtime_config.get('SUMO_WINDOW_SIZE', self._SUMO_WINDOW_SIZE))
        self._sumo_window: Deque[Dict[str, Any]] = deque(maxlen=sumo_window_size)
        self._sumo_telemetry_summary: Optional[Dict[str, Any]] = None

    def collect_sumo_frame(self,
                           dic_feature: Dict[str, Any],
                           step_num: int,
                           vehicle_distances: Optional[Dict[str, float]] = None,
                           vehicle_speeds: Optional[Dict[str, float]] = None) -> None:
        """
        Called every simulation step to push one frame into the sliding window.

        Counts only vehicles within CAMERA_VIEW_DISTANCE metres of the stop-line,
        mimicking the spatial field-of-view of a real intersection camera.

        A vehicle is counted as "waiting" (halting) iff speed < 0.1 m/s,
        consistent with SUMO's own waiting-vehicle definition.

        vehicle_distances : dict vehicle_id -> distance from stop-line (metres).
                            Provided by inter.dic_vehicle_distance_current_step.
        vehicle_speeds    : dict vehicle_id -> speed (m/s).
                            Provided by inter.dic_vehicle_speed_current_step.
        If either is None, falls back to full-lane pressure fields (no FOV clipping).

        Uses traffic_movement_vehicle_ids (semantic 12-slot order:
        W-Left/Thru/Right, E-Left/Thru/Right, N-Left/Thru/Right, S-Left/Thru/Right).
        """

        camera_range = float(
            self.vlm_runtime_config.get('CAMERA_VIEW_DISTANCE', self._CAMERA_VIEW_DISTANCE))

        vehicle_ids_by_slot: List[List[str]] = dic_feature.get(
            'traffic_movement_vehicle_ids') or [[] for _ in range(12)]
        while len(vehicle_ids_by_slot) < 12:
            vehicle_ids_by_slot.append([])

        raw_num   = dic_feature.get('traffic_movement_pressure_num')   or [0.0] * 12
        raw_queue = dic_feature.get('traffic_movement_pressure_queue') or [0.0] * 12

        num_vehicle: List[float] = []
        num_waiting: List[float] = []

        for slot_idx in range(12):
            veh_ids = vehicle_ids_by_slot[slot_idx] or []
            if vehicle_distances is not None and vehicle_speeds is not None and veh_ids:
                # Per-vehicle FOV filter + speed < 0.1 m/s halting check
                visible_count   = 0.0
                visible_waiting = 0.0
                for v in veh_ids:
                    dist = vehicle_distances.get(v, 999.0)
                    if dist <= camera_range:
                        visible_count += 1.0
                        if vehicle_speeds.get(v, 1.0) < 0.1:
                            visible_waiting += 1.0
            else:
                # Fallback: no per-vehicle info, use full-lane pressure fields
                visible_count   = float(raw_num[slot_idx])   if slot_idx < len(raw_num)   else 0.0
                visible_waiting = float(raw_queue[slot_idx]) if slot_idx < len(raw_queue) else 0.0

            num_vehicle.append(visible_count)
            num_waiting.append(visible_waiting)



        frame: Dict[str, Any] = {
            'step':         step_num,
            'num_vehicle':  num_vehicle,
            'num_waiting':  num_waiting,
            'camera_range': camera_range,
        }
        self._sumo_window.append(frame)
        # Persist raw frame to JSONL for post-hoc analysis.
        raw_path = os.path.join(self.api_debug_dir, f'{self.tls_id}_sumo_frames.jsonl')
        try:
            with open(raw_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({
                    **frame,
                    'slot_names': list(self._SLOT_NAMES),
                }, ensure_ascii=False) + '\n')
        except Exception:
            pass

    def _record_selected_action(self, step_num: int, action: int) -> None:
        """Synchronize action state for both API and fallback decisions."""
        self.current_phase_idx = action
        self.action_history.append((step_num, action))
        phase_name = self.phase_list[action] if 0 <= action < self.num_phases else ''
        self._last_chosen_phase = phase_name
        if hasattr(self, '_current_phase_totals_snapshot'):
            self._last_phase_totals = self._current_phase_totals_snapshot
        if phase_name in self._phase_v_since_served:
            self._phase_v_since_served[phase_name] = []

    def choose_action(self, image_paths, env, step_num: int = 0, neighbor_info=None) -> int:
        """
        SUMO-ONLY DECISION: No image processing, pure telemetry-based choice.

        Args:
            image_paths: IGNORED (not used in SUMO-only mode)
            env: SUMO environment providing intersection data
            step_num: Current simulation step

        Returns:
            Selected phase index based on SUMO telemetry analysis
        """

        self._last_decision_messages = []
        self._last_decision_response_json = None
        self._last_decision_text = ''
        self._last_decision_reasoning = ''
        self._last_api_error = None
        self._last_decision_record = {}

        intersection = None
        for inter in env.list_intersection:
            if inter.inter_id == self.tls_id:
                intersection = inter
                break
        if intersection is None:
            raise ValueError(f'Intersection {self.tls_id} not found')

        # Image paths are ignored in SUMO-only mode
        try:
            self._decision_step_count += 1
            self._latest_agent_state = self._build_agent_state(intersection, env)
            self._latest_neighbor_info = dict(neighbor_info or {})
            self._latest_coordination = dict(
                (neighbor_info or {}).get("__v32_coordination_info__") or {}
            )
            self._sumo_telemetry_summary = self._aggregate_sumo_window(step_num)
            # Persist aggregated telemetry snapshot for this decision step.
            if self._sumo_telemetry_summary and self._sumo_telemetry_summary.get('available'):
                snap_path = os.path.join(
                    self.api_debug_dir,
                    f'{self.tls_id}_sumo_telemetry_step{step_num}.json')
                try:
                    with open(snap_path, 'w', encoding='utf-8') as f:
                        json.dump(self._sumo_telemetry_summary, f,
                                  ensure_ascii=False, indent=2)
                except Exception:
                    pass



            # ============================================================
            # NO PERCEPTION STAGE - DIRECT SUMO TELEMETRY INPUT
            # ============================================================
            # Generate decision input directly from SUMO telemetry data.
            # No image processing, no vision model calls.

            telemetry_block = self._format_sumo_telemetry_block(self._sumo_telemetry_summary)
            if not telemetry_block:
                # First decision step: no frames collected yet; proceed with a placeholder.
                sumo_input_report = (
                    '[SUMO telemetry not yet available; sensor window is empty at step 0. '
                    'Select initial phase based on default priors.]'
                )
            else:
                sumo_input_report = telemetry_block

            # Direct decision stage: SUMO telemetry -> LLM -> phase choice
            decision_text = self._run_decision_stage(sumo_input_report, step_num)
            action, decision_source = self._resolve_decision_action(decision_text, step_num)
            self._last_decision_source = decision_source
            self._record_selected_action(step_num, action)
            decision_error = None
            if decision_source != 'final_answer':
                decision_error = (
                    'Assistant response did not contain a valid <signal> phase; '
                    'the coordination-aware fallback selected the executed action.'
                )
            self._store_last_decision_record(
                step_num=step_num,
                action=action,
                decision_text=decision_text,
                decision_source=decision_source,
                decision_error=decision_error,
            )

        except Exception as e:
            import traceback
            err_msg = traceback.format_exc()
            print(f'Warning: SUMO-only agent failed for {self.tls_id}: {e}')
            action = self._fallback_decision()
            self._last_decision_source = 'fallback_v25'
            self._record_selected_action(step_num, action)
            self._store_last_decision_record(
                step_num=step_num,
                action=action,
                decision_text=getattr(self, '_last_decision_text', ''),
                decision_source='fallback_v25',
                decision_error=str(e),
                error_traceback=err_msg,
                api_error=getattr(self, '_last_api_error', None),
            )
            try:
                self._save_fallback_log(step_num, action, err_msg)
            except Exception as log_error:
                print(f'Warning: failed to save fallback log for {self.tls_id}: {log_error}')
            return action

        try:
            self._save_sumo_only_log(step_num, decision_text, action)
        except Exception as log_error:
            print(f'Warning: failed to save decision log for {self.tls_id}: {log_error}')
        return action

    # ========================================================================
    # DECISION STAGE (the only stage in SUMO-only mode)
    # ========================================================================
    def _run_decision_stage(self, sumo_data: str, step_num: int) -> str:
        messages = [
            {'role': 'system', 'content': self._build_decision_system_prompt()},
            {'role': 'user', 'content': self._build_decision_user_prompt(sumo_data, step_num)},
        ]
        self._last_decision_messages = messages
        try:
            response_json = self._call_chat_api(
                messages=messages,
                api_url=self.decision_api_url,
                api_key=self.decision_api_key,
                model_name=self.decision_model,
                temperature=self.decision_temperature,
                max_tokens=self.decision_max_tokens,
                debug_tag=f'{self.tls_id}_step{step_num}_decision',
            )
        except Exception as exc:
            self._last_api_error = str(exc)
            raise
        self._last_decision_response_json = copy.deepcopy(response_json)
        assistant_message = self._extract_assistant_message(response_json)
        decision_text = self._extract_message_text(assistant_message).strip()
        reasoning_content = assistant_message.get('reasoning_content')
        self._last_decision_reasoning = (
            reasoning_content.strip()
            if isinstance(reasoning_content, str)
            else ''
        )
        self._last_decision_text = decision_text
        return decision_text

    @staticmethod
    def _extract_reasoning_text(decision_text: str) -> str:
        """Return the assistant reasoning without the final signal tag."""
        reasoning = re.sub(
            r'<signal>.*?</signal>',
            '',
            decision_text or '',
            flags=re.I | re.S,
        )
        return reasoning.strip()

    @staticmethod
    def _normalize_decision_source(decision_source: str) -> str:
        return 'llm' if decision_source == 'final_answer' else 'fallback_v25'

    @classmethod
    def _json_safe(cls, value: Any) -> Any:
        """Convert runtime containers/scalars into stable JSON values."""
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(key): cls._json_safe(item) for key, item in value.items()}
        if isinstance(value, set):
            return [cls._json_safe(item) for item in sorted(value, key=str)]
        if isinstance(value, (list, tuple, deque)):
            return [cls._json_safe(item) for item in value]
        tolist_method = getattr(value, 'tolist', None)
        if callable(tolist_method):
            try:
                return cls._json_safe(tolist_method())
            except Exception:
                pass
        item_method = getattr(value, 'item', None)
        if callable(item_method):
            try:
                return cls._json_safe(item_method())
            except Exception:
                pass
        return str(value)

    def _store_last_decision_record(
            self,
            step_num: int,
            action: int,
            decision_text: str,
            decision_source: str,
            decision_error: Optional[str] = None,
            error_traceback: Optional[str] = None,
            api_error: Optional[str] = None) -> None:
        """Capture the latest decision in a stable JSON-serializable form."""
        normalized_source = self._normalize_decision_source(decision_source)
        phase_name = (
            self.phase_list[action]
            if 0 <= action < self.num_phases
            else f'Phase_{action}'
        )
        messages = self._json_safe(getattr(self, '_last_decision_messages', []))
        system_prompt = next(
            (message.get('content', '') for message in messages
             if message.get('role') == 'system'),
            '',
        )
        user_prompt = next(
            (message.get('content', '') for message in messages
             if message.get('role') == 'user'),
            '',
        )
        self._last_decision_record = {
            'schema_version': 'v35_llmlight_decision_v1',
            'decision_step': int(step_num),
            'tls_id': self.tls_id,
            'model': getattr(self, 'decision_model', ''),
            'messages': messages,
            'system_prompt': system_prompt,
            'user_prompt': user_prompt,
            'raw_response': self._json_safe(
                getattr(self, '_last_decision_response_json', None)
            ),
            'assistant_response': decision_text or '',
            'reasoning': (
                getattr(self, '_last_decision_reasoning', '')
                or self._extract_reasoning_text(decision_text)
            ),
            'selected_phase': phase_name,
            'selected_action_idx': int(action),
            'decision_source': normalized_source,
            'reasoning_training_eligible': normalized_source == 'llm',
            'decision_error': decision_error,
            'api_error': api_error,
            'error_traceback': error_traceback,
            'structured_state': self._json_safe(
                getattr(self, '_latest_agent_state', {})
            ),
            'telemetry_summary': self._json_safe(
                getattr(self, '_sumo_telemetry_summary', None)
            ),
            'coordination': self._json_safe(
                getattr(self, '_latest_coordination', {})
            ),
        }

    def get_last_decision_record(self) -> Dict[str, Any]:
        """Return an isolated copy of the most recent structured decision."""
        return copy.deepcopy(getattr(self, '_last_decision_record', {}))

    # _build_perception_system_prompt removed: no perception stage.

    # _build_perception_prompt removed: no perception stage.

    def _build_decision_system_prompt(self) -> str:
        return self.vlm_runtime_config.get(
            'SUMO_ONLY_DECISION_SYSTEM_PROMPT',
            (
                'You are a traffic signal controller for a four-way intersection.\n\n'
                'The intersection has four sections (North, South, East, West). '
                'Each section contains lanes for through traffic, left-turns, and right-turns. '
                'Right-turn lanes have permanent permissive green and do not require signal control.'
            ),
        )

    def _build_decision_user_prompt(self, sumo_data: str, step_num: int) -> str:
        # Refresh the exact V25-compatible state and then record this decision
        # interval before rendering the compact Stage-2 representation.
        self._organize_data_by_phase(sumo_data)
        snapshot = getattr(self, '_current_phase_totals_snapshot', {})
        for phase_name in self.phase_list:
            if phase_name not in self._phase_v_since_served:
                self._phase_v_since_served[phase_name] = []
            self._phase_v_since_served[phase_name].append(snapshot.get(phase_name, 0))
        local = self._build_stage2_local_perception()
        cooperative = self._build_stage2_cooperative_perception()
        return self._render_stage2_prompt(local, cooperative)

    def _build_stage2_local_perception(self) -> Dict[str, Any]:
        """Render local SUMO telemetry in the Stage-2 SFT contract."""
        phases: Dict[str, Any] = {}
        for phase_name in self.phase_list:
            feature = self._phase_pseudo_video_feature(phase_name)
            state = self._latest_agent_state or {}
            movement_ids = state.get('traffic_movement_vehicle_ids_150m') or []
            vehicle_speed = state.get('vehicle_speed') or {}
            values_v, values_q = [], []
            for slot_idx in self._phase_slots(phase_name):
                vehicle_ids = list(movement_ids[slot_idx] or []) if slot_idx < len(movement_ids) else []
                values_v.append(len(vehicle_ids))
                values_q.append(sum(
                    float(vehicle_speed.get(vehicle_id, 1.0)) < self._QUEUE_SPEED_THRESHOLD
                    for vehicle_id in vehicle_ids
                ))
            # Preserve the canonical two-value shape even for unexpected phase names.
            while len(values_v) < 2:
                values_v.append(0)
                values_q.append(0)
            phases[phase_name] = {
                'v': values_v[:2],
                'q': values_q[:2],
                'dv': int(feature['delta_v']),
                'dq': int(feature['delta_q']),
                'age': sum(value > 0 for value in self._phase_v_since_served.get(phase_name, [])),
            }
        return {'current_phase': self._current_phase_name(), 'phases': phases}

    def _build_stage2_local_coordination(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        coordination: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for phase_name in self.phase_list:
            phase_info = dict(self._latest_coordination.get(phase_name, {}) or {})
            internal = dict(phase_info.get('internal', {}) or {})
            boundary = dict(phase_info.get('boundary', {}) or {})
            movements: Dict[str, Dict[str, Any]] = {}
            for movement in self._parse_phase_movements(phase_name):
                if movement in internal:
                    movements[movement] = {'count': int(internal[movement]), 'is_boundary': 'no'}
                elif movement in boundary:
                    movements[movement] = {'count': int(boundary[movement]), 'is_boundary': 'yes'}
                else:
                    movements[movement] = {'count': 0, 'is_boundary': 'no'}
            coordination[phase_name] = movements
        return coordination

    def _build_stage2_neighbors(self) -> Dict[str, Dict[str, Any]]:
        """Return broad directional neighbour pressure, never target movement matches."""
        movement_by_target_direction = {
            'N': ('NT', 'EL'), 'S': ('ST', 'WL'),
            'E': ('ET', 'SL'), 'W': ('WT', 'NL'),
        }
        neighbors: Dict[str, Dict[str, Any]] = {}
        for source_tls, data in (getattr(self, '_latest_neighbor_info', {}) or {}).items():
            if str(source_tls).startswith('__') or not isinstance(data, dict):
                continue
            # ``direction`` is the target-relative cardinal location of this
            # neighbor (N means the neighbor north of the target).  In
            # contrast, ``their_entry`` is the road entry direction at that
            # neighbor and must not be used to select its pressure movements.
            direction = str(data.get('direction') or '').upper()
            movements = movement_by_target_direction.get(direction)
            if not movements:
                continue
            movement_state = data.get('movement_state') or {}
            upstream = {
                movement: {
                    'v': int((movement_state.get(movement) or {}).get('v', 0)),
                    'q': int((movement_state.get(movement) or {}).get('q', 0)),
                }
                for movement in movements
            }
            distance_m = float(data.get('distance_m') or 0.0)
            speed_mps = float(data.get('speed_mps') or 0.0)
            neighbors[direction] = {
                'source_intersection': str(source_tls),
                'upstream_movements': upstream,
                'distance_m': round(distance_m, 2),
                'speed_limit_mps': round(speed_mps, 2),
                'travel_time_s': round(distance_m / speed_mps, 2)
                if distance_m > 0 and speed_mps > 0 else None,
            }
        return neighbors

    def _build_stage2_cooperative_perception(self) -> Dict[str, Any]:
        return {
            'local_coordination': self._build_stage2_local_coordination(),
            'neighbors': self._build_stage2_neighbors(),
        }

    @staticmethod
    def _render_stage2_prompt(local: Dict[str, Any], cooperative: Dict[str, Any]) -> str:
        local_json = json.dumps(local, ensure_ascii=False, separators=(',', ':'))
        cooperative_json = json.dumps(cooperative, ensure_ascii=False, separators=(',', ':'))
        return f'''You are an intelligent traffic signal controller for a four-phase intersection.
You receive compact perception from the current intersection and, when available,
perception summaries from neighboring intersections. Use both sources to decide
which candidate phase should be served next.

The controller operates in 30-second cycles: 25 seconds of green followed by a
5-second signal-transition interval. The decision is made at the end of the
current 25-second green interval, immediately before the 5-second transition.
The selected phase receives the next 25-second green interval after that
transition.

Let's think step by step.

<local_perception>
{local_json}
</local_perception>

<cooperative_perception>
{cooperative_json}
</cooperative_perception>

Local-field meanings:
- The phase movement order is ETWT=[ET, WT], NTST=[NT, ST], ELWL=[EL, WL], and NLSL=[NL, SL].
- v is the current vehicle count on the two movements, in the order above.
- q is the current queue count on the two movements, in the order above.
- dv is the recent vehicle-demand change (last observation minus earlier observation); dq is the corresponding queue change.
- age is the persistence, in decision intervals, since the phase last received service. A larger value means longer waiting, but it is not an independent measurement of traffic volume.
- Positive dv or dq indicates increasing pressure; negative values indicate decreasing pressure.

Cooperative-field meanings:
- local_coordination is movement-level potential arrival information for this target intersection. It comes from vehicles observed in upstream outgoing lanes. A count for ET is already a direct estimate of future pressure on this intersection's ET movement, regardless of which phase the upstream intersection subsequently selects. Use it directly to strengthen ET/WT -> ETWT, NT/ST -> NTST, EL/WL -> ELWL, and NL/SL -> NLSL. These vehicles are not yet visible locally, so do not add their counts to local v or q.
- is_boundary="no" means local_coordination uses an available upstream outgoing-lane frame; is_boundary="yes" means the direction is at a network boundary and the count is estimated from local V30-V15 instead.
- neighbors contains neighboring intersections' current incoming-lane pressure. Its upstream_movements labels belong to the neighbor, not to this target intersection. Never match a neighbor movement name directly to a same-named target movement, and do not treat neighbor counts as vehicles that will definitely arrive or as direct movement-level coordination counts.
- Use neighbors only as broad directional context because their eventual releases depend on their unknown next signal decisions. North and south neighbor pressure can increase the general priority of NTST and NLSL; east and west neighbor pressure can increase the general priority of ETWT and ELWL. Within each group, choose using this target's local demand, queue, trends, age, and local_coordination.
- distance_m, speed_limit_mps, and travel_time_s describe when broad neighbor pressure may matter. Compare travel_time_s with the next 5-second transition plus 25-second green interval; distant pressure should have less influence. Missing directions are network boundaries.

Please answer:
Which is the most effective traffic signal that will most significantly improve the traffic condition?

Reasoning guidance:
- Prefer phases that can release the largest visible demand within the current camera view, with the MOST significant impact.
- Prioritize sustained and worsening traffic pressure over brief fluctuations. Use dv to judge whether visible demand is accumulating or dissipating, and dq to judge whether queues are forming or clearing.
- When local visible demand and trends are comparable across phases, first use direct target-movement local_coordination, then persistent unmet demand and queue urgency to break the tie. Do not add coordination counts to locally visible vehicles.
- Use neighbors only to adjust broad north-south or east-west priority, not to infer a specific target movement or fixed target phase. The neighbor's future release is unknown, so its influence must remain weaker than direct local_coordination and this target's own observed demand.

Do not make up traffic measurements. First analyze the local and cooperative perception, then make the traffic-control decision. At the end, output exactly one selected phase:
<signal>SELECTED_PHASE</signal>
Replace SELECTED_PHASE with exactly one of ETWT, NTST, ELWL, or NLSL.'''

    def _build_agent_state(self, intersection, env) -> Dict[str, Any]:
        dic_feature = getattr(intersection, 'dic_feature', {}) or {}
        return {
            'traffic_movement_vehicle_ids_150m': dic_feature.get('traffic_movement_vehicle_ids_150m'),
            'traffic_movement_vehicle_ids': dic_feature.get('traffic_movement_vehicle_ids'),
            'v3_counting_line_flow_history': dic_feature.get('v3_counting_line_flow_history'),
            'v9_cycle_150m_history': dic_feature.get('v9_cycle_150m_history'),
            'v26_cycle_queue_history': dic_feature.get('v26_cycle_queue_history'),
            'v32_cycle_vehicle_snapshots': dic_feature.get('v32_cycle_vehicle_snapshots'),
            'vehicle_distance': getattr(intersection, 'dic_vehicle_distance_current_step', {}) or dic_feature.get('vehicle_distance') or {},
            'vehicle_speed': getattr(intersection, 'dic_vehicle_speed_current_step', {}) or dic_feature.get('vehicle_speed') or {},
            'waiting_vehicle_list': getattr(env, 'waiting_vehicle_list', {}) or dic_feature.get('waiting_vehicle_list') or {},
        }

    def _parse_phase_movements(self, phase_name: str) -> List[str]:
        if '_' in phase_name:
            return [movement for movement in phase_name.split('_') if movement]
        return [phase_name[i:i + 2] for i in range(0, len(phase_name), 2)]

    def _phase_slots(self, phase_name: str) -> List[int]:
        movement_map = {
            'WL': 0, 'WT': 1, 'WR': 2,
            'EL': 3, 'ET': 4, 'ER': 5,
            'NL': 6, 'NT': 7, 'NR': 8,
            'SL': 9, 'ST': 10, 'SR': 11,
        }
        slots = []
        for movement in self._parse_phase_movements(phase_name):
            idx = movement_map.get(movement)
            if idx is not None:
                slots.append(idx)
        return slots

    def _effective_distance(self) -> float:
        signal_time = float(self.vlm_runtime_config.get('MIN_ACTION_TIME', 30.0))
        camera_distance = float(
            self.vlm_runtime_config.get('CAMERA_VIEW_DISTANCE', self._CAMERA_VIEW_DISTANCE))
        avg_speed = float(
            self.vlm_runtime_config.get('V12_AVG_VEH_SPEED', self._DEFAULT_VEH_SPEED))
        return min(camera_distance, signal_time * avg_speed)

    def _phase_service_feature(self, phase_name: str, current_v: int) -> Dict[str, Any]:
        state = self._latest_agent_state or {}
        movement_ids = state.get('traffic_movement_vehicle_ids_150m') or state.get('traffic_movement_vehicle_ids') or []
        vehicle_distance = state.get('vehicle_distance') or {}
        vehicle_speed = state.get('vehicle_speed') or {}
        waiting_times = state.get('waiting_vehicle_list') or {}
        effective_distance = self._effective_distance()

        effective_vehicle_ids = []
        queued_wait_times = []
        for slot_idx in self._phase_slots(phase_name):
            if slot_idx >= len(movement_ids):
                continue
            for veh_id in movement_ids[slot_idx] or []:
                distance = float(vehicle_distance.get(veh_id, effective_distance))
                if distance > effective_distance:
                    continue
                effective_vehicle_ids.append(veh_id)
                speed = float(vehicle_speed.get(veh_id, 1.0))
                if speed < self._QUEUE_SPEED_THRESHOLD:
                    queued_wait_times.append(_extract_waiting_time(waiting_times.get(veh_id, 0.0)))

        n = len(effective_vehicle_ids)
        q = len(queued_wait_times)
        avg_wait_time = sum(queued_wait_times) / q if q > 0 else 0.0
        return {
            'current_v': current_v,
            'n': n,
            'q': q,
            'avg_wait_time': avg_wait_time,
            'effective_distance': effective_distance,
            'effective_vehicle_count': len(effective_vehicle_ids),
        }

    def _phase_pseudo_video_feature(self, phase_name: str) -> Dict[str, Any]:
        """Aggregate the exact six-frame V/Q features used by heuristics."""
        state = self._latest_agent_state or {}
        slots = self._phase_slots(phase_name)
        v_history = list(state.get("v9_cycle_150m_history") or [])
        q_history = list(state.get("v26_cycle_queue_history") or [])

        def phase_values(history: List[Any]) -> List[int]:
            values = []
            for frame in history:
                try:
                    values.append(int(sum(float(frame[index]) for index in slots
                                          if index < len(frame))))
                except (TypeError, ValueError):
                    values.append(0)
            return values

        v_values = phase_values(v_history)
        q_values = phase_values(q_history)
        # Keep instantaneous V/Q identical to V2/V25: count the current
        # 150 m movement lists directly. The six-frame buffers are used only
        # for temporal deltas.
        movement_ids = state.get("traffic_movement_vehicle_ids_150m") or []
        vehicle_speed = state.get("vehicle_speed") or {}
        current_ids = [
            vehicle_id
            for index in slots
            if index < len(movement_ids)
            for vehicle_id in (movement_ids[index] or [])
        ]
        current_v = len(current_ids)
        current_q = sum(
            float(vehicle_speed.get(vehicle_id, 1.0)) < self._QUEUE_SPEED_THRESHOLD
            for vehicle_id in current_ids
        )
        coordination = dict(self._latest_coordination.get(phase_name, {}) or {})
        return {
            "v_frames": v_values,
            "q_frames": q_values,
            "current_v": current_v,
            "current_q": current_q,
            "delta_v": v_values[-1] - v_values[0] if len(v_values) >= 2 else 0,
            "delta_q": q_values[-1] - q_values[0] if len(q_values) >= 2 else 0,
            "coordination_arrivals": int(coordination.get("arrival_15s", 0) or 0),
            "coordination_internal": dict(coordination.get("internal", {}) or {}),
            "coordination_boundary": dict(coordination.get("boundary", {}) or {}),
            "coordination_sources": dict(coordination.get("sources", {}) or {}),
            "coordination_events": list(coordination.get("events", []) or []),
            "has_internal_upstream": bool(
                (self._latest_coordination.get("__meta__") or {}).get(
                    "has_internal_upstream", False)),
            "has_usable_internal_coordination": bool(
                (self._latest_coordination.get("__meta__") or {}).get(
                    "has_usable_internal_coordination",
                    (self._latest_coordination.get("__meta__") or {}).get(
                        "has_internal_upstream", False))),
            "coordination_frame_time_s": coordination.get("frame_time_s"),
            "coordination_horizon_s": coordination.get("horizon_s"),
        }



    def _organize_data_by_phase(self, sumo_data: str) -> str:
        """
        Organize the current V18 state by phase.

        All phase-level quantities are derived from _latest_agent_state so that
        the LLM prompt uses the same state source as the V18 score.
        """
        state = self._latest_agent_state or {}
        if not (state.get('traffic_movement_vehicle_ids_150m') or
                state.get('traffic_movement_vehicle_ids')):
            return self._fallback_phase_format()

        phase_blocks = []
        current_phase_totals = {}
        self._current_phase_pseudo_features = {}

        for phase_name in self.phase_list:
            allowed_lanes = self._get_phase_lanes(phase_name)
            phase_stats = self._calculate_phase_stats_from_state(phase_name, allowed_lanes)
            pseudo_feature = self._phase_pseudo_video_feature(phase_name)
            self._current_phase_pseudo_features[phase_name] = pseudo_feature
            current_phase_totals[phase_name] = pseudo_feature["current_v"]

            phase_feature = self._phase_service_feature(
                phase_name, current_phase_totals[phase_name])
            phase_block = self._format_phase_block(
                phase_name, allowed_lanes, phase_stats, phase_feature, pseudo_feature)
            phase_blocks.append(phase_block)

        self._current_phase_totals_snapshot = current_phase_totals
        return json.dumps({
            'current_phase': self._current_phase_name(),
            'candidate_phases': phase_blocks,
        }, ensure_ascii=False, indent=2)



    def _get_phase_lanes(self, phase_name: str) -> Dict[str, str]:
        """
        Map phase name to the lanes it serves.
        Returns dict: {direction: lane_type} e.g. {'North': 'left', 'South': 'left'}
        """
        phase_upper = str(phase_name).upper()

        # Standard phase naming: ETWT, NTST, ELWL, NLSL
        if 'ETWT' in phase_upper or 'EWNS' in phase_upper:
            return {'East': 'straight', 'West': 'straight'}
        elif 'NTST' in phase_upper or 'NSEW' in phase_upper:
            return {'North': 'straight', 'South': 'straight'}
        elif 'ELWL' in phase_upper:
            return {'East': 'left', 'West': 'left'}
        elif 'NLSL' in phase_upper:
            return {'North': 'left', 'South': 'left'}
        else:
            # Fallback: try to parse from phase name
            lanes = {}
            if 'N' in phase_upper:
                lanes['North'] = 'left' if 'L' in phase_upper else 'straight'
            if 'S' in phase_upper:
                lanes['South'] = 'left' if 'L' in phase_upper else 'straight'
            if 'E' in phase_upper:
                lanes['East'] = 'left' if 'L' in phase_upper else 'straight'
            if 'W' in phase_upper:
                lanes['West'] = 'left' if 'L' in phase_upper else 'straight'
            return lanes

    def _current_phase_name(self) -> Optional[str]:
        if 0 <= self.current_phase_idx < self.num_phases:
            return self.phase_list[self.current_phase_idx]
        return None



    def _calculate_phase_stats(self, phase_name: str, approaches: Dict, allowed_lanes: Dict) -> Dict:
        """
        Calculate vehicle statistics for a phase.
        Returns: {direction: {'V': total_vehicles, 'H': halting_vehicles}}
        """
        # Map full direction names to SUMO abbreviations
        direction_map = {
            'North': 'N',
            'South': 'S',
            'East': 'E',
            'West': 'W',
        }

        stats = {}

        for direction, lane_type in allowed_lanes.items():
            # Convert "North" -> "N" to match SUMO data format
            sumo_direction = direction_map.get(direction, direction)
            approach_data = approaches.get(sumo_direction, {})
            lane_data = approach_data.get(lane_type, {'V': 0, 'H': 0})

            stats[direction] = {
                'V': lane_data.get('V', 0),
                'H': lane_data.get('H', 0),
            }

        return stats

    def _calculate_phase_stats_from_state(self, phase_name: str, allowed_lanes: Dict) -> Dict:
        """
        Calculate phase stats directly from _latest_agent_state.

        This mirrors the V12Agent state path and avoids mixing the SUMO-window
        JSON snapshot with the service-pressure feature state.
        """
        state = self._latest_agent_state or {}
        movement_ids = state.get('traffic_movement_vehicle_ids_150m') or state.get('traffic_movement_vehicle_ids') or []
        vehicle_speed = state.get('vehicle_speed') or {}
        direction_to_slot = {
            ('West', 'left'): 0,
            ('West', 'straight'): 1,
            ('West', 'right'): 2,
            ('East', 'left'): 3,
            ('East', 'straight'): 4,
            ('East', 'right'): 5,
            ('North', 'left'): 6,
            ('North', 'straight'): 7,
            ('North', 'right'): 8,
            ('South', 'left'): 9,
            ('South', 'straight'): 10,
            ('South', 'right'): 11,
        }

        stats = {}
        for direction, lane_type in allowed_lanes.items():
            slot_idx = direction_to_slot.get((direction, lane_type))
            veh_ids = []
            if slot_idx is not None and slot_idx < len(movement_ids):
                veh_ids = list(movement_ids[slot_idx] or [])
            stats[direction] = {
                'V': len(veh_ids),
                'H': sum(
                    1 for veh_id in veh_ids
                    if float(vehicle_speed.get(veh_id, 1.0)) < self._QUEUE_SPEED_THRESHOLD
                ),
            }
        return stats



    def _format_phase_block(self, phase_name: str, allowed_lanes: Dict, stats: Dict,
                            feature: Optional[Dict[str, Any]] = None,
                            pseudo_feature: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Format a single phase block in the template style.
        """
        # Build lane description
        lane_desc_parts = []
        for direction, lane_type in allowed_lanes.items():
            lane_desc_parts.append(f'{direction} {lane_type}')
        lane_desc = ' and '.join(lane_desc_parts) + ' lanes'

        # Calculate totals
        total_h = sum(s['H'] for s in stats.values())
        total_v = sum(s['V'] for s in stats.values())

        if feature is None or pseudo_feature is None:
            return {
                'signal': phase_name,
                'allowed_lanes': lane_desc,
                'current_v': total_v,
                'current_q': total_h,
                'data_available': False,
            }

        breakdown = {}
        for movement, count in pseudo_feature[
                'coordination_internal'].items():
            breakdown[movement] = {
                'count': int(count),
                'is_boundary': 'no',
            }
        for movement, count in pseudo_feature[
                'coordination_boundary'].items():
            if movement in breakdown:
                breakdown[movement]['count'] += int(count)
            else:
                breakdown[movement] = {
                    'count': int(count),
                    'is_boundary': 'yes',
                }
        coordinated_arrivals = {
            'total': pseudo_feature['coordination_arrivals'],
            'breakdown': breakdown,
        }

        phase_summary = {
            'signal': phase_name,
            'allowed_lanes': lane_desc,
            'current_v': pseudo_feature['current_v'],
            'current_q': pseudo_feature['current_q'],
            'demand_trend_v30_minus_v5': pseudo_feature['delta_v'],
            'queue_trend_q30_minus_q5': pseudo_feature['delta_q'],
            'coordinated_arrivals': coordinated_arrivals,
            'nonzero_v_history_length_since_last_service': sum(
                value > 0 for value in self._phase_v_since_served.get(phase_name, [])
            ),
        }
        return phase_summary



    def _fallback_phase_format(self) -> str:
        """Fallback when SUMO data is unavailable."""
        blocks = [
            {
                'signal': phase_name,
                'allowed_lanes': 'data unavailable',
                'current_v': 0,
                'current_q': 0,
                'data_available': False,
            }
            for phase_name in self.phase_list
        ]
        return json.dumps({
            'current_phase': self._current_phase_name(),
            'candidate_phases': blocks,
        }, ensure_ascii=False, indent=2)

    def _format_action_history(self, step_num: int) -> str:
        if not self.action_history:
            return 'No previous decisions.'
        lines = []
        for hist_step, action in self.action_history[-5:]:
            phase_name = self.phase_list[action] if 0 <= action < self.num_phases else f'Phase_{action}'
            gap = step_num - hist_step
            lines.append(f'- Step {hist_step}: {phase_name} ({gap} step(s) ago)')
        return '\n'.join(lines)



    def _format_steps_since_last_served(self, step_num: int) -> str:
        """Format steps since each phase was last served."""
        last_served: Dict[int, int] = {}
        for hist_step, action in self.action_history:
            if 0 <= action < self.num_phases:
                last_served[action] = hist_step

        lines = []
        for phase_idx in range(self.num_phases):
            phase_name = self.phase_list[phase_idx]
            if phase_idx in last_served:
                steps_ago = step_num - last_served[phase_idx]
                lines.append(f'- {phase_name}: {steps_ago} step(s) ago')
            else:
                lines.append(f'- {phase_name}: never served')

        return '\n'.join(lines) if lines else 'No phase history available.'

    def _format_phase_v_since_served(self) -> str:
        """Format vehicle count history for each phase since it was last served."""
        lines = []
        for phase_name in self.phase_list:
            history = self._phase_v_since_served.get(phase_name, [])
            if not history:
                lines.append(f'- {phase_name}: (served this step)')
            else:
                active_history = [value for value in history if value > 0]
                active_str = ' -> '.join(str(v) for v in active_history) if active_history else 'none'
                active_steps = len(active_history)
                lines.append(
                    f'- {phase_name}: active_unserved_steps={active_steps} | '
                    f'active V: {active_str}')
        return '\n'.join(lines) if lines else 'No history available.'

    def _format_decision_feedback(self) -> str:
        """
        Compare current phase vehicle counts with last step's counts.
        Shows the effect of the last decision on each phase's queue.
        """
        if not self._last_phase_totals or not self._last_chosen_phase:
            return ''

        # Get current totals (saved during _organize_data_by_phase)
        current_totals = getattr(self, '_current_phase_totals_snapshot', {})
        if not current_totals:
            return ''

        lines = [f'You chose {self._last_chosen_phase}.']
        for phase_name in self.phase_list:
            prev = self._last_phase_totals.get(phase_name, 0)
            curr = current_totals.get(phase_name, 0)
            diff = curr - prev
            if diff > 0:
                tag = 'worsened'
                diff_str = f'+{diff}'
            elif diff < 0:
                tag = 'improved'
                diff_str = str(diff)
            else:
                tag = 'no change'
                diff_str = '0'
            lines.append(f'- {phase_name}: {prev} -> {curr} ({diff_str}, {tag})')

        return '\n'.join(lines)

    # _build_perception_history_context removed: no perception stage.

    # _build_repeated_phase_check_for_perception removed: no perception stage.

    def _parse_decision_or_raise(self, answer: str) -> int:
        matches = re.findall(r'<signal>(.*?)</signal>', answer, flags=re.I | re.S)
        if not matches:
            raise RuntimeError(f'No <signal> tag found in decision output: {answer[:300]}')
        phase_name = matches[-1].strip()
        for idx, phase in enumerate(self.phase_list):
            if str(phase).strip().upper() == phase_name.upper():
                return idx
        raise RuntimeError(f'Unknown phase from decision stage: {phase_name}')

    def _has_valid_signal_tag(self, answer: str) -> bool:
        matches = re.findall(r'<signal>(.*?)</signal>', answer, flags=re.I | re.S)
        if not matches:
            return False
        phase_name = matches[-1].strip().upper()
        return any(str(phase).strip().upper() == phase_name for phase in self.phase_list)

    def _resolve_decision_action(self, decision_text: str, step_num: int) -> tuple[int, str]:
        if self._has_valid_signal_tag(decision_text):
            return self._parse_decision_or_raise(decision_text), 'final_answer'
        return self._fallback_decision(), 'fallback_v25'

    def _get_v25_based_fallback(self) -> int:
        """Use new coordination when available; otherwise preserve V25 exactly."""
        best_idx = 0
        best_score = None
        phase_totals = getattr(self, '_current_phase_totals_snapshot', {}) or {}
        pseudo_features = getattr(self, '_current_phase_pseudo_features', {}) or {}
        coordination_meta = (
            (getattr(self, '_latest_coordination', {}) or {}).get('__meta__', {})
            or {}
        )
        has_usable_coordination = bool(
            coordination_meta.get(
                'has_usable_coordination',
                coordination_meta.get(
                    'has_usable_internal_coordination',
                    coordination_meta.get('has_internal_upstream', False),
                ),
            )
        )

        for phase_idx, phase_name in enumerate(self.phase_list):
            current_v = int(phase_totals.get(phase_name, 0))
            pseudo_feature = pseudo_features.get(phase_name)
            if pseudo_feature is None:
                pseudo_feature = self._phase_pseudo_video_feature(phase_name)
            secondary = (
                int(pseudo_feature.get('coordination_arrivals', 0))
                if has_usable_coordination
                else int(pseudo_feature.get('delta_v', 0))
            )
            history = self._phase_v_since_served.get(phase_name, [])
            effective_history_len = sum(1 for value in history if value > 0)
            score = (current_v, secondary, effective_history_len, -phase_idx)
            if best_score is None or score > best_score:
                best_score = score
                best_idx = phase_idx
        return best_idx

    # _extract_perception_output removed: no perception stage.

    # _build_perception_raw_dump removed: no perception stage.

    # ------------------------------------------------------------------
    # SUMO telemetry: sliding-window aggregation
    # ------------------------------------------------------------------
    def _aggregate_sumo_window(self, step_num: int) -> Dict[str, Any]:
        """
        Extract ONLY the current frame state from the sliding window.

        Simplified output structure (current state only):
          current_state: {lane_name: {vehicle_count, waiting_count, state}}
        """
        frames = list(self._sumo_window)
        if len(frames) == 0:
            return {'available': False, 'reason': 'no frames collected yet'}

        # Get the newest (current) frame only
        current_frame = frames[-1]

        # Build per-lane current state
        current_state: Dict[str, Any] = {}
        for slot_idx, lane_name in enumerate(self._SLOT_NAMES):
            v = current_frame['num_vehicle'][slot_idx]
            w = current_frame['num_waiting'][slot_idx]
            current_state[lane_name] = {
                'vehicle_count': round(v),
                'waiting_count': round(w),
                'state':         self._quantize_state(v, w),
            }

        return {
            'available':        True,
            'timestamp_step':   step_num,
            'intersection_id':  self.tls_id,
            'current_state':    current_state,
        }

    # -- quantization helpers (only _quantize_state is used by _aggregate_sumo_window) --
    @staticmethod
    def _quantize_state(avg_vehicle: float, avg_waiting: float) -> str:
        if avg_vehicle < 0.5:
            return 'all_moving'
        ratio = avg_waiting / avg_vehicle if avg_vehicle > 0 else 0.0
        if ratio < 0.2:
            return 'all_moving'
        if ratio < 0.5:
            return 'mixed'
        if ratio < 0.85:
            return 'mostly_queued'
        return 'fully_stopped'

    def _format_sumo_telemetry_block(self, summary: Optional[Dict[str, Any]]) -> str:
        """
        Format ONLY the current state as a simple JSON object.
        No historical frames, no time sequence.
        """
        if not summary or not summary.get('available'):
            return ''

        current_state = summary.get('current_state', {})
        if not current_state:
            return ''



        # Build compact per-approach current state
        approaches: Dict[str, Any] = {}
        for approach in ['N', 'E', 'W', 'S']:
            lanes_in = [ln for ln in self._SLOT_NAMES if ln.startswith(approach + '_')]
            lanes_data: Dict[str, Any] = {}
            for ln in lanes_in:
                lf = current_state[ln]
                if lf['vehicle_count'] > 0 or lf['waiting_count'] > 0:
                    # use short lane suffix: left / straight / right
                    short = ln.split('_', 1)[1]
                    lanes_data[short] = {
                        'V': lf['vehicle_count'],
                        'H': lf['waiting_count'],
                        'state': lf['state'],
                    }
            if lanes_data:
                approaches[approach] = lanes_data

        output = {
            'step': summary.get('timestamp_step', 0),
            'intersection': summary.get('intersection_id', ''),
            'approaches': approaches,
        }

        header_lines = [
            '== SUMO CURRENT STATE (snapshot at decision time) ==',
            'V=vehicle count, H=halting count (speed<0.1 m/s), state=dominant lane state.',
            'IMPORTANT: Only lanes with at least one vehicle are shown. Lanes not listed have V=0 and H=0.',
            '',
        ]
        json_block = json.dumps(output, ensure_ascii=False, indent=2)
        return '\n'.join(header_lines) + json_block

    def _call_chat_api(self,
                       messages: List[Dict[str, Any]],
                       api_url: str,
                       api_key: str,
                       model_name: str,
                       temperature: float,
                       max_tokens: int,
                       debug_tag: str,
                       tools: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        if not api_url:
            raise RuntimeError('API URL is empty')
        payload = {
            'model': model_name,
            'messages': messages,
            'temperature': temperature,
            'max_tokens': max_tokens,
            'enable_thinking': self.decision_enable_thinking,
        }
        if self.seed is not None:
            payload['seed'] = self.seed
        if tools:
            payload['tools'] = tools
            payload['tool_choice'] = 'auto'
        headers = {'Content-Type': 'application/json'}
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'
        response = requests.post(api_url, headers=headers, data=json.dumps(payload), timeout=180)
        response.raise_for_status()
        response_json = response.json()
        self._last_decision_response_json = copy.deepcopy(response_json)
        debug_path = os.path.join(self.api_debug_dir, f'{debug_tag}.json')
        try:
            with open(debug_path, 'w', encoding='utf-8') as f:
                json.dump(response_json, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        if isinstance(response_json, dict) and 'error' in response_json:
            raise RuntimeError(f"API error: {response_json['error']}")
        return response_json

    def _extract_text_response(self, response_json: Dict[str, Any]) -> str:
        message = self._extract_assistant_message(response_json)
        return self._extract_message_text(message)

    def _extract_assistant_message(self, response_json: Dict[str, Any]) -> Dict[str, Any]:
        choices = response_json.get('choices') or []
        if not choices:
            raise RuntimeError('Empty API response choices')
        return choices[0].get('message') or {}

    def _extract_message_text(self, message: Dict[str, Any]) -> str:
        content = message.get('content')
        if content is None:
            return ''
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get('type') == 'text':
                    text_parts.append(item.get('text', ''))
            return '\n'.join(text_parts)
        raise RuntimeError('Unsupported response content format')

    # _extract_tool_calls removed: no perception stage / no tool calling.

    # _execute_tool_call removed: no perception stage / no tool calling.

    # _create_simple_composite_image removed: no image processing.

    # _append_single_image removed: no image processing.

    # _resolve_composite_output_dir removed: no image processing.

    def _save_sumo_only_log(self, step_num: int, decision_text: str, action: int):
        """Log the actual LLM conversation, response, and selected action."""
        phase_name = self.phase_list[action] if 0 <= action < self.num_phases else f'Phase_{action}'
        with open(self.log_file_path, 'a', encoding='utf-8') as f:
            f.write(f"{'=' * 72}\n")
            f.write(f'[SUMO-ONLY] Step {step_num}  [NO PERCEPTION STAGE]\n')
            f.write(f"{'=' * 72}\n\n")

            # ---- Decision stage input (text only) ----
            if self._last_decision_messages:
                f.write(f"{'-' * 72}\n")
                f.write('[LLM INPUT] Text-Only Decision Prompt\n')
                f.write(f"{'-' * 72}\n")
                for msg in self._last_decision_messages:
                    role = msg.get('role', 'unknown').upper()
                    content = msg.get('content', '')
                    f.write(f'[{role}]\n')
                    f.write(str(content) + '\n\n')

            # ---- Decision stage output ----
            f.write(f"{'-' * 72}\n")
            f.write('[LLM OUTPUT] Decision Response\n')
            f.write(f"{'-' * 72}\n")
            f.write(decision_text + '\n\n')

            # ---- Final result ----
            f.write(f'[FINAL ACTION]      {phase_name}\n')
            f.write(f'[DECISION SOURCE]   {getattr(self, "_last_decision_source", "sumo_only_decision")}\n')
            f.write(f'[INPUT DATA TYPE]   SUMO telemetry only (no images)\n\n')

    def _save_fallback_log(self, step_num: int, fallback_action: int, error_trace: str):
        phase_name = (self.phase_list[fallback_action]
                      if 0 <= fallback_action < self.num_phases else f'Phase_{fallback_action}')
        with open(self.log_file_path, 'a', encoding='utf-8') as f:
            f.write(f"{'=' * 72}\n")
            f.write(f'Step {step_num}  [FALLBACK -- EXCEPTION]\n')
            f.write(f"{'=' * 72}\n\n")
            f.write('[ERROR TRACEBACK]\n')
            f.write(error_trace + '\n\n')
            f.write(f'[FALLBACK ACTION] {phase_name}\n\n')

    def _fallback_decision(self) -> int:
        return self._get_v25_based_fallback()


# Backward-compatible alias for external scripts that import the old class name
LLMLightAgent = SUMOOnlyAgent


if __name__ == '__main__':
    print('SUMO-Only Traffic Signal Control Agent')
    print('  - Input:  SUMO telemetry data (JSON)')
    print('  - Output: Signal phase decision')
    print('  - Mode:   No perception stage, no image processing')
