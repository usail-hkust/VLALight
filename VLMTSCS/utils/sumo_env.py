import os
import sys
import time
import math
import random
import json
import hashlib
import pickle
import subprocess
import shutil
import socket
import numpy as np
import pandas as pd
import traceback
from multiprocessing import Process
from collections import defaultdict, deque
from functools import reduce
from copy import deepcopy

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    if tools not in sys.path:
        sys.path.append(tools)
else:
    default_tools = "/usr/share/sumo/tools"
    if os.path.exists(default_tools) and default_tools not in sys.path:
        sys.path.append(default_tools)

import warnings
import traci
import sumolib
from .v36_coordination import is_new_coordination_enabled

# Suppress deprecation from traci/libsumo about getAllProgramLogics (we use getCompleteRedYellowGreenDefinition)
warnings.filterwarnings("ignore", message=".*getAllProgramLogics.*", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*getAllProgramLogics.*", category=UserWarning)

# Global dictionaries (can be part of config or discovered)
location_dict = {"North": "N", "South": "S", "East": "E", "West": "W"}
location_dict_reverse = {v: k for k, v in location_dict.items()}
direction_dict = {"go_straight": "T", "turn_left": "L", "turn_right": "R"}

# Angles represent the direction of travel (Heading) towards the intersection (calculated via atan2(dy, dx))
angles = [0, math.pi / 2, math.pi, 3 * math.pi / 2, 2 * math.pi]  # Eastbound, Northbound, Westbound, Southbound, Eastbound
# Orients map these Headings to their Origin (Standard Convention)
orients = ['W', 'S', 'E', 'N', 'W', 'S', 'E', 'N']

DEFAULT_YELLOW_TIME = 3  # Default yellow time if not specified

class Intersection:
    """
    Represents a single intersection in the simulation environment, adapted for SUMO.
    Handles state updates, feature calculation, and signal control for this intersection.
    It dynamically discovers its topology from a SUMO network and controls via TraCI.
    """
    _MOVEMENT_TO_PRESSURE_IDX_MAP = {
        'WL': 0, 'WT': 1, 'WR': 2,  # West: Left, Through, Right
        'EL': 3, 'ET': 4, 'ER': 5,  # East: Left, Through, Right
        'NL': 6, 'NT': 7, 'NR': 8,  # North: Left, Through, Right
        'SL': 9, 'ST': 10, 'SR': 11  # South: Left, Through, Right
    }

    def __init__(self, tls_id, dic_traffic_env_conf, traci_conn, sumo_net, path_to_log, adjacency_info, custom_phase_list=None):
        """
        Initializes an Intersection object based on a SUMO traffic light system (TLS).

        Args:
            tls_id (str): The ID of the traffic light system in SUMO.
            dic_traffic_env_conf (dict): The traffic environment configuration dictionary.
            traci_conn (traci.connection): The active TraCI connection object.
            sumo_net (sumolib.net.Net): The pre-parsed sumolib network object.
            path_to_log (str): Path to the directory for logging.
            adjacency_info (dict): Information about neighboring intersections.
            custom_phase_list (list, optional): A list of phase name strings restricting the agent's choices.
        """
        # 1. Store injected dependencies and set compatibility attributes
        self.tls_id = tls_id
        self.inter_id = tls_id
        self.inter_name = tls_id
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.traci_conn = traci_conn
        self.sumo_net = sumo_net
        self.path_to_log = path_to_log
        self.custom_phase_list = custom_phase_list
        self.adjacency_info = adjacency_info

        # 2. Initial Validation: Ensure the TLS ID is valid
        try:
            if self.tls_id not in self.traci_conn.trafficlight.getIDList():
                raise ValueError(f"TLS ID '{self.tls_id}' not found in the running SUMO simulation.")
        except Exception as e:
             raise ValueError(f"Failed to validate TLS ID '{self.tls_id}': {e}") from e

        # --- Declare attributes that will be populated by _build_conceptual_model ---
        self.virtual = False
        self.point = {}
        self.phases = [] # Raw SUMO phase definitions
        self.control_phases = [] # Final list of phase NAMES available to agent
        self.all_control_phase_names = [] # All detected phase NAMES
        self.phase_name_2_cityflow_idx = {} # Map name string to actual SUMO phase index
        self.green_phases = [] # List of actual SUMO green phase indices
        
        self.incoming_roads = {}
        self.outgoing_roads = {}
        self.list_entering_lanes = [] # Padded, canonical lists
        self.list_exiting_lanes = [] # Padded, canonical lists
        self.lane_to_road = {}
        self.list_lanes = []
        self.road_id_2_orient = {}
        self.action_2_phase_index = {} # action_idx (0,1..) -> SUMO phase_idx

        self.yellow_time = self.dic_traffic_env_conf.get("YELLOW_TIME", DEFAULT_YELLOW_TIME)
        self.yellow_phase_index = -1 # Placeholder for logging/internal state

        # 3. Build the conceptual model from SUMO data
        self._build_conceptual_model()

        # --- State Variables ---
        self.dic_lane_vehicle_previous_step = defaultdict(list)
        self.dic_lane_waiting_vehicle_count_previous_step = defaultdict(int)
        self.dic_vehicle_speed_previous_step = {}
        self.dic_vehicle_distance_previous_step = {}
        self.dic_lane_vehicle_current_step = defaultdict(list)
        self.dic_lane_waiting_vehicle_count_current_step = defaultdict(int)
        self.dic_vehicle_speed_current_step = {}
        self.dic_vehicle_distance_current_step = {}
        self.dic_lane_vehicle_previous_step_in = defaultdict(list)
        self.dic_lane_vehicle_current_step_in = defaultdict(list)
        self.list_lane_vehicle_previous_step_in = []
        self.list_lane_vehicle_current_step_in = []
        self.dic_vehicle_arrive_leave_time = {}
        self._v3_slot_flow_accumulator = [0] * 12
        self._v3_slot_flow_history = [deque(maxlen=30) for _ in range(12)]
        self._v3_flow_sample_counter = 0
        self._v3_accumulated_150m_total = [0] * 12
        self._v3_discharged_0m_total = [0] * 12

        # --- Feature Storage ---
        self.dic_feature = {}
        self.dic_feature_previous_step = {}

        # --- Signal Timing ---
        self.current_phase_index = 0
        try:
             # Get the initial phase from SUMO
             self.current_phase_index = self.traci_conn.trafficlight.getPhase(self.tls_id)
        except traci.TraCIException as e:
            print(f"Warning: Could not get initial phase for {self.tls_id}. Defaulting to 0. Error: {e}")
        
        self.default_phase_index = self.current_phase_index
        self.previous_phase_index = self.current_phase_index
        self.next_phase_to_set_index = self.current_phase_index
        self.current_phase_duration = 0
        self.time_since_last_change = 0
        
        # for SUMO to complete its internal yellow/red transition.
        self.is_in_transition = False
        self._transition_stage = 'yellow'  # 'yellow' or 'allred'
        self._transition_timer = 0         # countdown timer for current transition stage (in sim steps)
        self._pending_green_phase = -1     # target green phase index after transition ends
        self._action_already_set = False   # flag to prevent re-triggering yellow in same action period
        
        self.fixed_time_cycle_index = 0

        # Lock the initial phase so SUMO does not auto-advance
        try:
            self.traci_conn.trafficlight.setPhaseDuration(self.tls_id, 9999)
        except Exception:
            pass

        # Implementation note.
        signal_log_dir = os.path.join(path_to_log, "signals")
        os.makedirs(signal_log_dir, exist_ok=True)
        self.log_file_path = os.path.join(signal_log_dir, f"signal_{self.inter_name}.txt")
        resume_existing = bool(self.dic_traffic_env_conf.get(
            "RESUME_FROM_CHECKPOINT", False))
        if not (resume_existing and os.path.isfile(self.log_file_path)):
            with open(self.log_file_path, "w") as f:
                f.write("time,phase_index\n")
            self._log_signal_state()

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]

        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        for k, v in self.__dict__.items():
            # Skip non-serializable or shared attributes
            if k in ['traci_conn', 'sumo_net', 'incoming_roads', 'outgoing_roads']:
                continue
            setattr(result, k, deepcopy(v, memo))

        result.traci_conn = None
        result.sumo_net = self.sumo_net
        result.incoming_roads = self.incoming_roads.copy()
        result.outgoing_roads = self.outgoing_roads.copy()

        return result
    
    def _build_conceptual_model(self):
        """
        Orchestrates the discovery of intersection topology (lanes, roads, orientations)
        and traffic light logic (phases, movements) from the SUMO simulation.
        """
        # Get basic intersection info from the pre-parsed network object
        try:
            tls_node = self.sumo_net.getNode(self.tls_id)
            self.point = {"x": tls_node.getCoord()[0], "y": tls_node.getCoord()[1]}
        except KeyError:
            # Fallback if TLS node is not found in the static network file
            self.point = {"x": 0, "y": 0}

        # Step 1: Discover all connected lanes and roads
        self._discover_lanes_and_roads()

        # Step 2: Determine road orientations based on geometry
        self._calculate_road_orientations()

        # Step 3: Build the legacy `road_links` structure.
        self._build_road_links()

        # Step 4: Create padded, canonical lane lists for feature calculation.
        self._create_canonical_lane_lists()

        # Step 5: Discover SUMO phases and map them to canonical names
        self._discover_phases_and_movements()

        # Step 6: Apply custom phase filtering if provided.
        self._apply_custom_phase_mapping()


    def _print_intersection_debug_info(self, intersection_obj):
        """
        打印指定Intersection对象的详细调试信息，包括相位和车道方向。
        """
        inter_id = intersection_obj.inter_id
        print("\n" + "="*80)
        print(f"DEBUG INFO FOR INTERSECTION: {inter_id}")
        print("="*80)

        # Implementation note.
        # Implementation note.
        print("\n[1] 可用信号灯相位 (Control Phases):")
        if intersection_obj.control_phases:
            print(f"    {intersection_obj.control_phases}")
        else:
            print("    - No control phases found.")

        # Implementation note.
        print("\n[2] 道路地理方向 (Road Orientations):")
        print("  入口道路 (Incoming):")
        if intersection_obj.road_id_2_orient.get('incoming'):
            for road, orient in intersection_obj.road_id_2_orient['incoming'].items():
                print(f"    - Road '{road}': {orient} approach")
        else:
            print("    - None found.")

        print("\n  出口道路 (Outgoing):")
        if intersection_obj.road_id_2_orient.get('outgoing'):
            for road, orient in intersection_obj.road_id_2_orient['outgoing'].items():
                print(f"    - Road '{road}': {orient} exit")
        else:
            print("    - None found.")

        # Implementation note.
        # Implementation note.
        print("\n[3] 解析出的车道转向 (Parsed Lane Movements / Road Links):")
        if intersection_obj.road_links:
            for link in intersection_obj.road_links:
                start_road = link.get('startRoad')
                end_road = link.get('endRoad')
                turn_type = link.get('type')
                print(f"  - FROM '{start_road}' TO '{end_road}', Turn: {turn_type}")
        else:
            print("    - No road links found.")

        print("="*80 + "\n")

    def _discover_lanes_and_roads(self):
        """
        Populates lane and road lists using TraCI and the sumolib network object.
        """
        # Get all lanes controlled by this traffic light
        try:
            # Note: getControlledLanes returns ALL lanes, including internal ones.
            self.list_entering_lanes = list(set(self.traci_conn.trafficlight.getControlledLanes(self.tls_id)))
        except traci.TraCIException as e:
            print(f"Warning: Could not get controlled lanes for {self.tls_id}. Error: {e}")
            self.list_entering_lanes = []
            return

        if not self.list_entering_lanes:
            return

        # Discover incoming/outgoing roads and all lanes
        temp_outgoing_lanes = set()
        
        for lane_id in self.list_entering_lanes:
            try:
                lane_obj = self.sumo_net.getLane(lane_id)
                road_obj = lane_obj.getEdge()
                road_id = road_obj.getID()

                # Ignore internal edges
                if not road_id.startswith(":"):
                    self.incoming_roads[road_id] = road_obj
                    self.lane_to_road[lane_id] = road_id

                # Find corresponding outgoing lanes from this incoming lane (sumolib)
                outgoing_lanes = lane_obj.getOutgoingLanes()  # list[sumolib.net.Lane]

                for outgoing_lane in outgoing_lanes:
                    outgoing_lane_id = outgoing_lane.getID()
                    outgoing_road_obj = outgoing_lane.getEdge()
                    outgoing_road_id = outgoing_road_obj.getID()

                    # Ensure it's not an internal junction edge
                    if not outgoing_road_id.startswith(":"):
                        temp_outgoing_lanes.add(outgoing_lane_id)
                        self.outgoing_roads[outgoing_road_id] = outgoing_road_obj
                        self.lane_to_road[outgoing_lane_id] = outgoing_road_id
            except (KeyError, AttributeError) as e:
                pass

        self.list_exiting_lanes = sorted(list(temp_outgoing_lanes))
        self.list_lanes = sorted(list(set(self.list_entering_lanes + self.list_exiting_lanes)))

    def _calculate_road_orientations(self):
        """
        Calculates and assigns canonical orientations (N, S, E, W) to incoming and
        outgoing roads based on their geometry from the sumolib network.
        """
        def _calculate_for(roads_dict, is_incoming):
            center_x, center_y = self.point['x'], self.point['y']
            roads_with_angle = []
            
            for road_id, road_obj in roads_dict.items():
                shape = road_obj.getShape()
                if len(shape) < 2:
                    continue

                point_x, point_y = shape[-2] if is_incoming else shape[1]

                dx = center_x - point_x
                dy = center_y - point_y
                angle = math.atan2(dy, dx)
                if angle < 0: angle += 2 * math.pi

                # Find the closest cardinal direction
                orient_angle_diffs = np.abs(np.subtract(angles, angle))
                orient_index = np.argmin(orient_angle_diffs)
                orient = orients[orient_index]
                orient_angle_diff = orient_angle_diffs[orient_index]
                
                roads_with_angle.append(
                    {'id': road_id, 'angle': angle, 'orient': orient, 'angle_diff': orient_angle_diff}
                )

            if not roads_with_angle: 
                return {}

            roads_with_angle.sort(key=lambda x: x['angle'])
            # Use the min angle diff for starting point
            min_orient_road = min(roads_with_angle, key=lambda x: x['angle_diff'])
            min_orient_road_index = roads_with_angle.index(min_orient_road)
            roads_with_angle = roads_with_angle[min_orient_road_index:] + roads_with_angle[:min_orient_road_index]

            if not self._decide_road_orient(roads_with_angle):
                # Implementation note.
                return {road['id']: road.get('orient') for road in roads_with_angle}

            result = {road['id']: road.get('orient') for road in roads_with_angle}
            return result

        self.road_id_2_orient['incoming'] = _calculate_for(self.incoming_roads, is_incoming=True)
        self.road_id_2_orient['outgoing'] = _calculate_for(self.outgoing_roads, is_incoming=False)

    def _build_road_links(self):
        """
        Builds the `road_links` structure by relying on SUMO's connection definitions (via sumolib)
        """
        self.road_links = []
        links_grouped = defaultdict(list)

        try:
            controlled_links = self.traci_conn.trafficlight.getControlledLinks(self.tls_id)
        except traci.TraCIException as e:
            print(f"Warning: getControlledLinks failed for {self.tls_id}: {e}")
            return

        # Mapping SUMO internal directions ('s', 'l', 'r', etc.) to standardized turn types
        SUMO_DIR_MAP = {
            's': 'go_straight',
            't': 'turn_left',  # 't' (turn) is often treated as slight left/through
            'l': 'turn_left',
            'L': 'turn_left',
            'r': 'turn_right',
            'R': 'turn_right',
        }

        for link_group in controlled_links:
            if not link_group:
                continue
            from_lane_id, to_lane_id, _via = link_group[0]
            
            if from_lane_id.startswith(":") or to_lane_id.startswith(":"):
                continue

            try:
                from_lane_obj = self.sumo_net.getLane(from_lane_id)
                to_lane_obj = self.sumo_net.getLane(to_lane_id)
                from_road_id = from_lane_obj.getEdge().getID()
                to_road_id = to_lane_obj.getEdge().getID()

                # Find the specific connection object between these two lanes in sumolib
                connection = None
                for conn in from_lane_obj.getOutgoing():
                    if conn.getToLane() == to_lane_obj:
                        connection = conn
                        break

                if connection is None:
                    continue

                # Get the direction attribute from the connection
                sumo_direction = connection.getDirection()
                turn_type = SUMO_DIR_MAP.get(sumo_direction)

                if turn_type is None:
                    continue # Skip unknown/unsupported types (e.g., U-turns 'u')

                # We still need the orientation (now correctly calculated) to categorize the movement
                f_or = self.road_id_2_orient['incoming'].get(from_road_id)
                                
                if not f_or:
                    continue
                    
                start_lane_idx = from_lane_obj.getIndex()
                end_lane_idx = to_lane_obj.getIndex()
                
                links_grouped[(from_road_id, to_road_id, turn_type)].append({
                    "startLaneIndex": start_lane_idx,
                    "endLaneIndex": end_lane_idx
                })
                
            except (KeyError, AttributeError) as e:
                continue

        for (start_road, end_road, turn_type), lane_links in links_grouped.items():
            # Remove duplicate lane links and sort for stability
            unique_lane_links = [dict(t) for t in {tuple(d.items()) for d in lane_links}]
            # Sort by startLaneIndex primarily for consistency
            unique_lane_links.sort(key=lambda x: (x['startLaneIndex'], x.get('endLaneIndex', -1)))
            self.road_links.append({
                "startRoad": start_road,
                "endRoad": end_road,
                "type": turn_type,
                "laneLinks": unique_lane_links
            })

    def _create_canonical_lane_lists(self):
        """
        Creates 12-element, canonically ordered (W,E,N,S approach, 3 lanes each)
        lists for entering and exiting lanes. Non-existent lanes are filled with None.
        This is required for compatibility with feature calculation methods that expect a fixed-size input.
        """
        # Ensure we have a valid road_id_2_orient map
        if not self.road_id_2_orient or not 'incoming' in self.road_id_2_orient:
            print(f"Warning: {self.inter_id} Missing orientation data for canonical lane list creation. Using default empty lists.")
            self.list_entering_lanes = [None] * 12
            self.list_exiting_lanes = [None] * 12
            return

        padded_entering_lanes = [None] * 12
        padded_exiting_lanes = [None] * 12

        # --- Pad Entering Lanes ---
        lanes_by_orient = defaultdict(list)
        for road_id, orient in self.road_id_2_orient.get('incoming', {}).items():
            road_obj = self.incoming_roads.get(road_id)
            if road_obj:
                for lane_obj in road_obj.getLanes():
                    lanes_by_orient[orient].append(lane_obj.getID())

        orient_map = {'W': 0, 'E': 3, 'N': 6, 'S': 9}
        for orient, offset in orient_map.items():
            lanes = sorted(lanes_by_orient.get(orient, []))
            for i in range(3):
                if i < len(lanes):
                    padded_entering_lanes[offset + i] = lanes[i]

        # --- Pad Exiting Lanes ---
        exiting_lanes_by_orient = defaultdict(list)
        for road_id, orient in self.road_id_2_orient.get('outgoing', {}).items():
             road_obj = self.outgoing_roads.get(road_id)
             if road_obj:
                for lane_obj in road_obj.getLanes():
                    exiting_lanes_by_orient[orient].append(lane_obj.getID())

        for orient, offset in orient_map.items():
            lanes = sorted(exiting_lanes_by_orient.get(orient, []))
            for i in range(3):
                if i < len(lanes):
                    padded_exiting_lanes[offset + i] = lanes[i]

        # Overwrite the instance attributes with the padded, canonical lists
        self.list_entering_lanes = padded_entering_lanes
        self.list_exiting_lanes = padded_exiting_lanes
        
        # Update the list_lanes to include all padded lanes
        self.list_lanes = self.list_entering_lanes + self.list_exiting_lanes
        
        # Remove None values from the list_lanes
        self.list_lanes = [lane for lane in self.list_lanes if lane is not None]


    def _discover_phases_and_movements(self):
        """
        Discovers the traffic light program from SUMO, identifies green phases,
        and translates them into canonical movement-based names (e.g., "ETWT")
        by READING THE PHASE NAME ATTRIBUTE.
        """
        try:
            # Get the first (active) program definition
            logic = self.traci_conn.trafficlight.getCompleteRedYellowGreenDefinition(self.tls_id)[0]
            self.phases = logic.getPhases()
        except (traci.TraCIException, IndexError) as e:
            print(f"Warning: Could not get TLS logic for '{self.tls_id}'. Phase control disabled. Error: {e}")
            return

        phase_dict = {}  # cityflow_idx -> set(movement_names_like_WT)
        phase_name_to_idx_map = {}
        temp_control_phase_names = []

        for phase_idx, phase in enumerate(self.phases):
            # A phase is green if it has a 'G' or 'g' and is not a short transition phase.
            if ('G' in phase.state or 'g' in phase.state) and phase.minDur > 2:
                self.green_phases.append(phase_idx)
                # 1. Read the phase name directly from the object. This is the name
                #    set by convert_sumo_roadnet_phases.py script.
                phase_name_str = getattr(phase, 'name', '')

                # 2. Skip phases that are unnamed or are the specific yellow phase.
                #    This makes the logic robust and ignores transitional phases.
                if not phase_name_str or phase_name_str == "YELLOW_ALL_RED":
                    continue

                # 3. Create a set of movements from the name for the legacy `phase_dict`
                #    This assumes names are pairs of characters, e.g., "ETWT" -> {"ET", "WT"}
                movement_names = set()
                if len(phase_name_str) % 2 == 0:
                    for i in range(0, len(phase_name_str), 2):
                        movement_names.add(phase_name_str[i:i + 2])
                
                if not movement_names:
                    continue

                phase_dict[phase_idx] = movement_names

                # 4. Use the clean, correct name for agent control.
                if phase_name_str not in phase_name_to_idx_map:
                    temp_control_phase_names.append(phase_name_str)
                    phase_name_to_idx_map[phase_name_str] = phase_idx


        self.phase_index_2_phase_name = phase_dict
        self.all_control_phase_names = sorted(list(set(temp_control_phase_names)))
        self.phase_name_2_cityflow_idx = phase_name_to_idx_map
        if self.green_phases:
            # Set a reasonable default green phase
            if self.all_control_phase_names:
                first_phase_name = self.all_control_phase_names[0]
                self.default_phase_index = self.phase_name_2_cityflow_idx.get(first_phase_name, self.green_phases[0])
            else:
                 self.default_phase_index = self.green_phases[0]

    # Helper methods for road orientation calculation (used in _calculate_road_orientations)
    def _get_opposite_road(self, cur_road, roads_with_angle, orients_taken={}):
        """
        Finds the opposite road for a given road based on its orientation.
        """
        for orient, road_idx in orients_taken.items():
            road = roads_with_angle[road_idx]
            if road['orient'] and cur_road['id'] != road['id'] and abs(
                    abs(cur_road['angle'] - road['angle']) - math.pi) < math.pi / 8:
                return road_idx
        return -1

    def _decide_road_orient(self, roads_with_angle, last_road=None, orients_taken={}, cur_index=0):
        if cur_index == len(roads_with_angle):
            return True

        cur_road = roads_with_angle[cur_index]

        my_possible_orients = []
        cur_possible_dir_index = 0 if last_road is None else (orients.index(last_road['orient']) + 1)
        for i in range(cur_possible_dir_index, cur_possible_dir_index + 4):
            if orients[i % 4] in orients_taken or (not last_road is None and orients[i % 4] == last_road['orient']):
                break
            my_possible_orients.append(orients[i % 4])

        if len(my_possible_orients) == 0:
            return False

        my_fav_orient_index = my_possible_orients.index(cur_road['orient']) if cur_road[
                                                                                   'orient'] in my_possible_orients else 0
        opposite_road_index = self._get_opposite_road(cur_road, roads_with_angle, orients_taken)
        if opposite_road_index != -1:
            my_fav_orient_index = my_possible_orients.index(
                orients[(orients.index(roads_with_angle[opposite_road_index]['orient']) + 2) % 4])
        my_possible_orients = my_possible_orients[my_fav_orient_index:] + my_possible_orients[:my_fav_orient_index]

        for dir in my_possible_orients:
            cur_road['orient'] = dir
            _orients_taken = orients_taken.copy()
            _orients_taken[dir] = cur_index
            if self._decide_road_orient(roads_with_angle, cur_road, _orients_taken, cur_index + 1):
                return True

        return False

    def _apply_custom_phase_mapping(self):
        """
        Filters the detected phases based on self.custom_phase_list.
        Sets the final self.control_phases and self.action_2_phase_index.
        """
        source_phase_names = self.all_control_phase_names
        name_to_idx_map = self.phase_name_2_cityflow_idx
        final_control_phase_names = []

        if self.custom_phase_list is not None:
            available_names_set = set(source_phase_names)
            for name in self.custom_phase_list:
                if name in available_names_set:
                    if name not in final_control_phase_names:
                        final_control_phase_names.append(name)
                else:
                    print(f"  Warning: Custom phase '{name}' not found in detected phases {source_phase_names}.")

        else:
            final_control_phase_names = source_phase_names

        if not final_control_phase_names and self.green_phases:
            print(final_control_phase_names, self.green_phases)
            print(f"Intersection {self.inter_id}: No standard phase names detected, but green phases exist. Allowing control by index.")
            final_control_phase_names = []
            for idx in self.green_phases:
                dummy_name = f"Index_{idx}"
                final_control_phase_names.append(dummy_name)
                name_to_idx_map[dummy_name] = idx

        self.control_phases = final_control_phase_names
        self.action_2_phase_index = {}
        self.phase_index_2_action = {}  # Reverse mapping: SUMO phase index -> control_phases index
        allowed_sumo_indices = set()

        # Implementation note.
        # Implementation note.
        # Implementation note.
        for i, phase_name in enumerate(self.control_phases):
            sumo_idx = name_to_idx_map.get(phase_name)
            if sumo_idx is not None:
                self.action_2_phase_index[i] = sumo_idx
                self.phase_index_2_action[sumo_idx] = i  # Reverse mapping
                allowed_sumo_indices.add(sumo_idx)

        original_green_phases = list(self.green_phases)
        self.green_phases = [idx for idx in original_green_phases if idx in allowed_sumo_indices]


    def _log_signal_state(self):
        """Logs the current time and phase index."""
        if self.dic_traffic_env_conf.get("_SUPPRESS_SIGNAL_LOG", False):
            return
        try:
            current_time = self.get_current_time()
            # Log a special index like -1 to indicate the start of a transition
            log_phase_index = -1 if self.is_in_transition else self.current_phase_index
            with open(self.log_file_path, "a") as f:
                f.write(f"{current_time},{log_phase_index}\n")
        except Exception as e:
            print(f"Error logging signal state for {self.inter_name}: {e}")

    def _get_yellow_phase_index(self, green_phase_index):
        """
        Return the index of the yellow phase that follows the given green phase.

        New phase structure (per green phase):
            green_phase_index + 0 : green  (named e.g. ETWT, state has 'G')
            green_phase_index + 1 : yellow (unnamed, state has 'y')
            green_phase_index + 2 : all-red (unnamed, state all 'r'/'g')

        Strategy:
          1. Check green_phase_index + 1: if it exists and has 'y', treat it
             as the yellow phase (yellow may also have 'G' for right-turn).
          2. If no 'y', check if it has no 'G' (could be all-red or permissive-only).
          3. Legacy fallback: search for a phase named "YELLOW_ALL_RED".
          4. Last resort: return the first phase that has no 'G'.
          Returns -1 if no valid yellow/all-red phase is found.
        """
        # Primary: next phase after the current green is the dedicated yellow
        if self.phases and 0 <= green_phase_index < len(self.phases):
            next_idx = green_phase_index + 1
            if next_idx < len(self.phases):
                next_state = getattr(self.phases[next_idx], 'state', '')
                if next_state:
                    # Yellow phase has 'y' character (may also have 'G' for right-turn)
                    if 'y' in next_state:
                        return next_idx
                    # All-red or permissive-only phase has no 'G' (only 'r'/'g')
                    if 'G' not in next_state:
                        return next_idx

        # Legacy fallback: find by name "YELLOW_ALL_RED"
        for idx, phase in enumerate(self.phases):
            name = getattr(phase, 'name', '')
            if name == "YELLOW_ALL_RED":
                return idx

        # Last resort: first phase with no 'G'
        for idx, phase in enumerate(self.phases):
            state_str = getattr(phase, 'state', '')
            if state_str and 'G' not in state_str:
                return idx
        return -1

    def set_signal(self, action, action_pattern):
        """
        Sets the traffic signal phase for the intersection in SUMO.

        Key fix: SUMO type="static" traffic lights auto-advance through their
        phase program. We must call setPhaseDuration() after setPhase() to lock
        the phase and prevent SUMO from auto-cycling.

        Transition logic (two-stage, managed by internal timer):
        - Stage 'yellow': setPhase(yellow_idx), locked for configured yellow duration.
          When timer expires, advance to stage 'allred'.
        - Stage 'allred': setPhase(allred_idx), locked only when ALL_RED_TIME > 0.
          When timer expires, setPhase(target_green) and exit transition.
        - A transition is started only on an actual phase change. Keeping the
          same phase continues green without inserting yellow.
        """
        if not self.phases:
            return

        # --- Phase 1: Handle ongoing transition (yellow or all-red stage) ---
        if self.is_in_transition:
            self._transition_timer -= 1
            if self._transition_timer <= 0:
                if self._transition_stage == 'yellow':
                    # Yellow done → enter all-red stage
                    yellow_idx = self._get_yellow_phase_index(self.current_phase_index)
                    allred_idx = yellow_idx + 1 if yellow_idx != -1 else -1
                    # Validate all-red phase: must exist and have no 'G'
                    if (allred_idx != -1 and allred_idx < len(self.phases)):
                        allred_state = getattr(self.phases[allred_idx], 'state', '')
                        if 'G' in allred_state:
                            allred_idx = -1  # not a real all-red, skip
                    else:
                        allred_idx = -1

                    configured_allred_duration = int(
                        self.dic_traffic_env_conf.get("ALL_RED_TIME", 0) or 0)
                    if allred_idx != -1 and configured_allred_duration > 0:
                        allred_obj = self.phases[allred_idx]
                        allred_duration = configured_allred_duration
                        if allred_duration <= 0:
                            allred_duration = configured_allred_duration
                        try:
                            self.traci_conn.trafficlight.setPhase(self.tls_id, allred_idx)
                            self.traci_conn.trafficlight.setPhaseDuration(self.tls_id, allred_duration)
                        except traci.TraCIException:
                            pass
                        self._transition_stage = 'allred'
                        self._transition_timer = allred_duration
                    else:
                        # No all-red phase — go directly to target green
                        try:
                            target_green = self._pending_green_phase
                            self.traci_conn.trafficlight.setPhase(self.tls_id, target_green)
                            self.traci_conn.trafficlight.setPhaseDuration(self.tls_id, 9999)
                            self.current_phase_index = target_green
                            self.current_phase_duration = 0
                            self.is_in_transition = False
                            self._transition_stage = 'yellow'
                            self._log_signal_state()
                        except traci.TraCIException as e:
                            print(f"TraCIException setting green phase {self._pending_green_phase} for {self.tls_id}: {e}")
                            self.is_in_transition = False
                            self._transition_stage = 'yellow'

                elif self._transition_stage == 'allred':
                    # All-red done → enter target green phase
                    try:
                        target_green = self._pending_green_phase
                        self.traci_conn.trafficlight.setPhase(self.tls_id, target_green)
                        self.traci_conn.trafficlight.setPhaseDuration(self.tls_id, 9999)
                        self.current_phase_index = target_green
                        self.current_phase_duration = 0
                        self.is_in_transition = False
                        self._transition_stage = 'yellow'
                        self._log_signal_state()
                    except traci.TraCIException as e:
                        print(f"TraCIException setting green phase {self._pending_green_phase} for {self.tls_id}: {e}")
                        self.is_in_transition = False
                        self._transition_stage = 'yellow'
            else:
                # Still in current transition stage — keep phase locked
                try:
                    self.traci_conn.trafficlight.setPhaseDuration(self.tls_id, self._transition_timer)
                except traci.TraCIException:
                    pass
            return

        # --- Phase 2: Keep current green phase locked (prevent SUMO auto-advance) ---
        self.current_phase_duration += 1
        self.time_since_last_change += 1
        try:
            self.traci_conn.trafficlight.setPhaseDuration(self.tls_id, 9999)
        except traci.TraCIException:
            pass

        # --- Phase 3: Determine target phase from action ---
        target_phase_index = -1

        if action_pattern == "set":
            if action == -1:
                target_phase_index = self.current_phase_index
            elif not self.control_phases or not self.action_2_phase_index:
                target_phase_index = self.current_phase_index
            elif action >= 0:
                num_actions = len(self.control_phases)
                if num_actions > 0:
                    actual_action_index = action % num_actions
                    default_sumo_idx = self.action_2_phase_index.get(0, self.current_phase_index)
                    target_phase_index = self.action_2_phase_index.get(actual_action_index, default_sumo_idx)
                else:
                    target_phase_index = self.current_phase_index
            else:
                target_phase_index = self.current_phase_index

        elif action_pattern == "switch":
            if action == 0:
                target_phase_index = self.current_phase_index
            elif action == 1:
                target_phase_index = (self.current_phase_index + 1) % len(self.phases)
            else:
                target_phase_index = self.current_phase_index
        else:
            target_phase_index = self.current_phase_index

        if self.dic_traffic_env_conf.get("SKIP_TRANSITION_PHASE", False):
            if target_phase_index != -1 and not self._action_already_set:
                try:
                    self.traci_conn.trafficlight.setPhase(self.tls_id, target_phase_index)
                    self.traci_conn.trafficlight.setPhaseDuration(self.tls_id, 9999)
                    self.current_phase_index = target_phase_index
                    self.current_phase_duration = 0
                    self.is_in_transition = False
                    self._transition_stage = 'yellow'
                    self._pending_green_phase = -1
                    self._action_already_set = True
                    self.previous_phase_index = self.current_phase_index
                    self.time_since_last_change = 0
                    self._log_signal_state()
                except traci.TraCIException as e:
                    print(f"TraCIException setting phase for {self.tls_id}: {e}")
            return

        # --- Phase 4: Start yellow+all-red transition only on actual phase changes ---
        # Same-phase decisions keep the current green running (no clearance needed).
        if target_phase_index != -1 and not self._action_already_set and target_phase_index != self.current_phase_index:
            try:
                # Find the yellow phase that follows the CURRENT green phase
                yellow_idx = self._get_yellow_phase_index(self.current_phase_index)

                if yellow_idx != -1:
                    # Use configured yellow duration so experiments match CoLLMLight.
                    yellow_duration = int(self.yellow_time or DEFAULT_YELLOW_TIME)
                    if yellow_duration <= 0:
                        yellow_duration = self.yellow_time

                    # Start yellow stage
                    self.traci_conn.trafficlight.setPhase(self.tls_id, yellow_idx)
                    self.traci_conn.trafficlight.setPhaseDuration(self.tls_id, yellow_duration)

                    self.is_in_transition = True
                    self._transition_stage = 'yellow'
                    self._transition_timer = yellow_duration
                    self._pending_green_phase = target_phase_index
                    self._action_already_set = True
                else:
                    # No yellow phase found — switch directly
                    self.traci_conn.trafficlight.setPhase(self.tls_id, target_phase_index)
                    self.traci_conn.trafficlight.setPhaseDuration(self.tls_id, 9999)
                    self.current_phase_index = target_phase_index
                    self.current_phase_duration = 0
                    self._action_already_set = True

                self.previous_phase_index = self.current_phase_index
                self.time_since_last_change = 0
                self._log_signal_state()
            except traci.TraCIException as e:
                print(f"TraCIException setting phase for {self.tls_id}: {e}")

    def update_previous_measurements(self):
        """Copies current measurements to previous step's variables."""
        self.previous_phase_index = self.current_phase_index

        self.dic_lane_vehicle_previous_step = self.dic_lane_vehicle_current_step.copy()
        self.dic_lane_vehicle_previous_step_in = self.dic_lane_vehicle_current_step_in.copy()
        self.dic_lane_waiting_vehicle_count_previous_step = self.dic_lane_waiting_vehicle_count_current_step.copy()
        self.dic_vehicle_speed_previous_step = self.dic_vehicle_speed_current_step.copy()
        self.dic_vehicle_distance_previous_step = self.dic_vehicle_distance_current_step.copy()
        self.list_lane_vehicle_previous_step_in = self.list_lane_vehicle_current_step_in[:]

    def update_current_measurements(self, simulator_state):
        """
        Updates intersection state based on the global simulator state.
        Calculates features for the current step.
        """
        # --- Update Phase Duration ---
        # IMPORTANT: Do NOT sync current_phase_index from SUMO here.
        # set_signal() now fully manages current_phase_index internally.
        # Syncing from SUMO would overwrite our managed state (e.g., during
        # yellow transitions) and break the phase control logic.
        # Note: current_phase_duration is already managed in set_signal() method

        # --- Update Vehicle/Lane States ---
        self.dic_lane_vehicle_current_step = defaultdict(list, simulator_state["get_lane_vehicles"])
        self.dic_lane_waiting_vehicle_count_current_step = defaultdict(int, simulator_state["get_lane_waiting_vehicle_count"])
        self.dic_vehicle_speed_current_step = simulator_state["get_vehicle_speed"]
        self.dic_vehicle_distance_current_step = simulator_state["get_vehicle_distance"]

        # Update states specifically for incoming lanes (using padded lists)
        self.dic_lane_vehicle_current_step_in = defaultdict(list)
        for lane_id in self.list_entering_lanes:
            if lane_id is not None:
                self.dic_lane_vehicle_current_step_in[lane_id] = self.dic_lane_vehicle_current_step.get(lane_id, [])

        # --- Track Vehicle Arrivals/Departures ---
        current_vehicles_in = set()
        for vehicles in self.dic_lane_vehicle_current_step_in.values():
            current_vehicles_in.update(vehicles)
        self.list_lane_vehicle_current_step_in = list(current_vehicles_in)

        previous_vehicles_in = set(self.list_lane_vehicle_previous_step_in)

        list_vehicle_new_arrive = list(current_vehicles_in - previous_vehicles_in)
        list_vehicle_new_left = list(previous_vehicles_in - current_vehicles_in)

        self._update_arrive_time(list_vehicle_new_arrive)
        self._update_left_time(list_vehicle_new_left)

        # --- Update Features ---
        self._update_feature()

    def _update_arrive_time(self, list_vehicle_arrive):
        """Records the arrival time for vehicles entering the intersection's approach lanes."""
        ts = self.get_current_time()
        for vehicle in list_vehicle_arrive:
            if vehicle not in self.dic_vehicle_arrive_leave_time:
                self.dic_vehicle_arrive_leave_time[vehicle] = {"enter_time": ts, "leave_time": np.nan}

    def _update_left_time(self, list_vehicle_left):
        """Records the departure time for vehicles leaving the intersection's approach lanes."""
        ts = self.get_current_time()
        for vehicle in list_vehicle_left:
            if vehicle in self.dic_vehicle_arrive_leave_time:
                if np.isnan(self.dic_vehicle_arrive_leave_time[vehicle]["leave_time"]):
                    self.dic_vehicle_arrive_leave_time[vehicle]["leave_time"] = ts

    def _update_feature(self):
        """Calculates and stores various state features for the current time step."""
        dic_feature = {}

        # --- Basic Features ---
        # Use transition index (-1) if in transition
        # Convert SUMO phase index to control_phases index for agent
        if self.is_in_transition:
            active_phase_index = -1
        else:
            # Map SUMO phase index to control_phases index
            active_phase_index = self.phase_index_2_action.get(self.current_phase_index, 0)
        dic_feature["cur_phase"] = [active_phase_index]
        dic_feature["time_this_phase"] = [self.current_phase_duration]

        # Use the padded lists for feature calculation (handles non-existent lanes with None)
        dic_feature["lane_num_vehicle"] = [
            len(self.dic_lane_vehicle_current_step.get(lane, [])) if lane is not None else 0
            for lane in self.list_entering_lanes
        ]
        dic_feature["lane_num_vehicle_downstream"] = [
            len(self.dic_lane_vehicle_current_step.get(lane, [])) if lane is not None else 0
            for lane in self.list_exiting_lanes
        ]
        dic_feature["lane_num_waiting_vehicle_in"] = [
            self.dic_lane_waiting_vehicle_count_current_step.get(lane, 0) if lane is not None else 0
            for lane in self.list_entering_lanes
        ]
        dic_feature["lane_num_waiting_vehicle_out"] = [
            self.dic_lane_waiting_vehicle_count_current_step.get(lane, 0) if lane is not None else 0
            for lane in self.list_exiting_lanes
        ]

        # Pressure calculated based on waiting vehicles
        dic_feature["pressure"] = dic_feature["lane_num_waiting_vehicle_in"] + [-count for count in dic_feature["lane_num_waiting_vehicle_out"]]

        # Calculate pressures using the canonical 12-lane methods
        dic_feature["traffic_movement_pressure_queue"] = self._get_traffic_movement_pressure_general(
            dic_feature["lane_num_waiting_vehicle_in"], dic_feature["lane_num_waiting_vehicle_out"])

        dic_feature["best_action_idx"] = self.get_max_pressure_phase_action(
            dic_feature["traffic_movement_pressure_queue"])

        dic_feature["traffic_movement_pressure_num"] = self._get_traffic_movement_pressure_general(
            dic_feature["lane_num_vehicle"], dic_feature["lane_num_vehicle_downstream"])

        dic_feature["entering_lane_vehicle_list"] = [
            self.dic_lane_vehicle_current_step.get(lane, []) if lane is not None else []
            for lane in self.list_entering_lanes
        ]

        dic_feature["traffic_movement_vehicle_ids"] = self._get_traffic_movement_vehicle_sets(
            dic_feature["entering_lane_vehicle_list"])
        self._update_v3_counting_line_flow(dic_feature["traffic_movement_vehicle_ids"])
        self._update_v3_discharge_counter(dic_feature["traffic_movement_vehicle_ids"])
        camera_view_distance = self.dic_traffic_env_conf.get("CAMERA_VIEW_DISTANCE", 150.0)
        entering_lane_vehicle_list_150m = [
            [
                veh_id for veh_id in vehicle_ids
                if self.dic_vehicle_distance_current_step.get(veh_id, float("inf")) <= camera_view_distance
            ]
            for vehicle_ids in dic_feature["entering_lane_vehicle_list"]
        ]
        dic_feature["traffic_movement_vehicle_ids_150m"] = self._get_traffic_movement_vehicle_sets(
            entering_lane_vehicle_list_150m)
        dic_feature["v8_upstream_arrival_30s"] = self._get_v8_upstream_arrival_preview(
            dic_feature["entering_lane_vehicle_list"])
        dic_feature["vehicle_distance"] = dict(self.dic_vehicle_distance_current_step)
        dic_feature["vehicle_speed"] = dict(self.dic_vehicle_speed_current_step)
        dic_feature["v3_counting_line_flow_history"] = [
            list(history) for history in self._v3_slot_flow_history
        ]
        dic_feature["v3_consequence_totals"] = {
            "accumulated_150m": list(self._v3_accumulated_150m_total),
            "discharged_0m": list(self._v3_discharged_0m_total),
        }

        dic_feature["v9_cycle_150m_history"] = list(
            self.dic_traffic_env_conf.get("_CURRENT_V9_CYCLE_HISTORY", {}).get(
                self.inter_id, []))
        dic_feature["v32_cycle_vehicle_snapshots"] = list(
            self.dic_traffic_env_conf.get(
                "_CURRENT_V32_CYCLE_VEHICLE_SNAPSHOTS", {}).get(
                self.inter_id, []))
        dic_feature["v36_cycle_outbound_snapshots"] = list(
            self.dic_traffic_env_conf.get(
                "_CURRENT_V36_CYCLE_OUTBOUND_SNAPSHOTS", {}).get(
                self.inter_id, []))
        dic_feature["v33_cycle_vehicle_snapshots"] = list(
            self.dic_traffic_env_conf.get(
                "_CURRENT_V33_CYCLE_VEHICLE_SNAPSHOTS", {}).get(
                self.inter_id, []))
        dic_feature["v26_cycle_queue_history"] = list(
            self.dic_traffic_env_conf.get("_CURRENT_V26_CYCLE_QUEUE_HISTORY", {}).get(
                self.inter_id, []))
        dic_feature["v10_cycle_zone_history"] = list(
            self.dic_traffic_env_conf.get("_CURRENT_V10_CYCLE_ZONE_HISTORY", {}).get(
                self.inter_id, []))
        self.dic_feature = dic_feature

    def _get_v8_upstream_arrival_preview(self, entering_lane_vehicle_list):
        """Counts vehicles outside 150m that can enter the camera view within 30s."""
        camera_view_distance = float(self.dic_traffic_env_conf.get(
            "CAMERA_VIEW_DISTANCE", 150.0))
        preview_horizon = float(self.dic_traffic_env_conf.get(
            "V8_UPSTREAM_PREVIEW_HORIZON", 30.0))
        min_speed = float(self.dic_traffic_env_conf.get(
            "V8_UPSTREAM_MIN_SPEED", 0.1))

        preview_lane_vehicle_list = []
        for vehicle_ids in entering_lane_vehicle_list:
            preview_vehicle_ids = []
            for veh_id in vehicle_ids:
                distance = self.dic_vehicle_distance_current_step.get(veh_id)
                speed = self.dic_vehicle_speed_current_step.get(veh_id, 0.0)
                if distance is None or distance <= camera_view_distance or speed <= min_speed:
                    continue
                eta_to_view = (distance - camera_view_distance) / speed
                if 0.0 <= eta_to_view <= preview_horizon:
                    preview_vehicle_ids.append(veh_id)
            preview_lane_vehicle_list.append(preview_vehicle_ids)
        return self._get_traffic_movement_vehicle_sets(preview_lane_vehicle_list)

    def _parse_phase_movements(self, phase_name):
        return [phase_name[i:i + 2] for i in range(0, len(phase_name), 2)]

    def phase_sum_from_slots(self, values, phase_name):
        total = 0
        for movement in self._parse_phase_movements(phase_name):
            idx = self._MOVEMENT_TO_PRESSURE_IDX_MAP.get(movement)
            if idx is not None and idx < len(values):
                total += values[idx]
        return total

    def phase_vehicle_ids_from_lane_snapshot(self, phase_name, lane_vehicle_snapshot):
        entering_lane_vehicle_list = [
            lane_vehicle_snapshot.get(lane, []) if lane is not None else []
            for lane in self.list_entering_lanes
        ]
        vehicle_ids_by_slot = self._get_traffic_movement_vehicle_sets(
            entering_lane_vehicle_list)
        phase_vehicle_ids = set()
        for movement in self._parse_phase_movements(phase_name):
            idx = self._MOVEMENT_TO_PRESSURE_IDX_MAP.get(movement)
            if idx is not None and idx < len(vehicle_ids_by_slot):
                phase_vehicle_ids.update(vehicle_ids_by_slot[idx] or [])
        return phase_vehicle_ids

    def capture_runtime_state(self):
        return {
            "current_phase_index": self.current_phase_index,
            "previous_phase_index": self.previous_phase_index,
            "next_phase_to_set_index": self.next_phase_to_set_index,
            "current_phase_duration": self.current_phase_duration,
            "is_in_transition": self.is_in_transition,
            "transition_stage": self._transition_stage,
            "transition_timer": self._transition_timer,
            "action_already_set": self._action_already_set,
            "dic_feature": deepcopy(self.dic_feature),
            "dic_feature_previous_step": deepcopy(self.dic_feature_previous_step),
            "fixed_time_cycle_index": self.fixed_time_cycle_index,
            "dic_lane_vehicle_previous_step": deepcopy(self.dic_lane_vehicle_previous_step),
            "dic_lane_vehicle_current_step": deepcopy(self.dic_lane_vehicle_current_step),
            "dic_lane_vehicle_previous_step_in": deepcopy(
                self.dic_lane_vehicle_previous_step_in),
            "dic_lane_vehicle_current_step_in": deepcopy(
                self.dic_lane_vehicle_current_step_in),
            "list_lane_vehicle_previous_step_in": list(
                self.list_lane_vehicle_previous_step_in),
            "list_lane_vehicle_current_step_in": list(
                self.list_lane_vehicle_current_step_in),
            "dic_lane_waiting_vehicle_count_current_step": deepcopy(
                self.dic_lane_waiting_vehicle_count_current_step),
            "dic_vehicle_speed_previous_step": deepcopy(self.dic_vehicle_speed_previous_step),
            "dic_vehicle_distance_previous_step": deepcopy(self.dic_vehicle_distance_previous_step),
            "dic_vehicle_speed_current_step": deepcopy(self.dic_vehicle_speed_current_step),
            "dic_vehicle_distance_current_step": deepcopy(self.dic_vehicle_distance_current_step),
            "v3_slot_flow_accumulator": list(self._v3_slot_flow_accumulator),
            "v3_slot_flow_history": [deque(history, maxlen=history.maxlen) for history in self._v3_slot_flow_history],
            "v3_flow_sample_counter": self._v3_flow_sample_counter,
            "v3_accumulated_150m_total": list(self._v3_accumulated_150m_total),
            "v3_discharged_0m_total": list(self._v3_discharged_0m_total),
            "dic_vehicle_arrive_leave_time": deepcopy(self.dic_vehicle_arrive_leave_time),
        }

    def restore_runtime_state(self, state):
        self.current_phase_index = state["current_phase_index"]
        self.previous_phase_index = state["previous_phase_index"]
        self.next_phase_to_set_index = state["next_phase_to_set_index"]
        self.current_phase_duration = state["current_phase_duration"]
        self.is_in_transition = state["is_in_transition"]
        self._transition_stage = state["transition_stage"]
        self._transition_timer = state["transition_timer"]
        self._action_already_set = state["action_already_set"]
        self.dic_feature = deepcopy(state["dic_feature"])
        self.dic_feature_previous_step = deepcopy(state.get(
            "dic_feature_previous_step", self.dic_feature_previous_step))
        self.fixed_time_cycle_index = int(state.get(
            "fixed_time_cycle_index", self.fixed_time_cycle_index))
        self.dic_lane_vehicle_previous_step = deepcopy(state["dic_lane_vehicle_previous_step"])
        self.dic_lane_vehicle_current_step = deepcopy(state["dic_lane_vehicle_current_step"])
        self.dic_lane_vehicle_previous_step_in = deepcopy(state.get(
            "dic_lane_vehicle_previous_step_in",
            self.dic_lane_vehicle_previous_step_in,
        ))
        self.dic_lane_vehicle_current_step_in = deepcopy(state.get(
            "dic_lane_vehicle_current_step_in",
            self.dic_lane_vehicle_current_step_in,
        ))
        self.list_lane_vehicle_previous_step_in = list(state.get(
            "list_lane_vehicle_previous_step_in",
            self.list_lane_vehicle_previous_step_in,
        ))
        self.list_lane_vehicle_current_step_in = list(state.get(
            "list_lane_vehicle_current_step_in",
            self.list_lane_vehicle_current_step_in,
        ))
        self.dic_lane_waiting_vehicle_count_current_step = deepcopy(
            state["dic_lane_waiting_vehicle_count_current_step"])
        self.dic_vehicle_speed_previous_step = deepcopy(state["dic_vehicle_speed_previous_step"])
        self.dic_vehicle_distance_previous_step = deepcopy(state["dic_vehicle_distance_previous_step"])
        self.dic_vehicle_speed_current_step = deepcopy(state["dic_vehicle_speed_current_step"])
        self.dic_vehicle_distance_current_step = deepcopy(state["dic_vehicle_distance_current_step"])
        self._v3_slot_flow_accumulator = list(state["v3_slot_flow_accumulator"])
        self._v3_slot_flow_history = [
            deque(history, maxlen=history.maxlen) for history in state["v3_slot_flow_history"]
        ]
        self._v3_flow_sample_counter = state["v3_flow_sample_counter"]
        self._v3_accumulated_150m_total = list(state["v3_accumulated_150m_total"])
        self._v3_discharged_0m_total = list(state["v3_discharged_0m_total"])
        self.dic_vehicle_arrive_leave_time = deepcopy(state["dic_vehicle_arrive_leave_time"])

    def _update_v3_counting_line_flow(self, vehicle_ids_by_slot):
        counting_line = float(self.dic_traffic_env_conf.get("COUNTING_LINE_DISTANCE", 150.0))
        for slot_idx in range(min(12, len(vehicle_ids_by_slot))):
            for veh_id in vehicle_ids_by_slot[slot_idx] or []:
                prev_dist = self.dic_vehicle_distance_previous_step.get(veh_id)
                curr_dist = self.dic_vehicle_distance_current_step.get(veh_id, float("inf"))
                if prev_dist is not None and prev_dist > counting_line and curr_dist <= counting_line:
                    self._v3_slot_flow_accumulator[slot_idx] += 1
                    self._v3_accumulated_150m_total[slot_idx] += 1

        self._v3_flow_sample_counter += 1
        if self._v3_flow_sample_counter >= 10:
            self._v3_flow_sample_counter = 0
            for slot_idx in range(12):
                self._v3_slot_flow_history[slot_idx].append(
                    self._v3_slot_flow_accumulator[slot_idx])
                self._v3_slot_flow_accumulator[slot_idx] = 0

    def _update_v3_discharge_counter(self, current_vehicle_ids_by_slot):
        previous_entering_lane_vehicle_list = [
            self.dic_lane_vehicle_previous_step.get(lane, []) if lane is not None else []
            for lane in self.list_entering_lanes
        ]
        previous_vehicle_ids_by_slot = self._get_traffic_movement_vehicle_sets(
            previous_entering_lane_vehicle_list)
        near_stopline_distance = float(self.dic_traffic_env_conf.get(
            "DISCHARGE_COUNTING_LINE_DISTANCE", 50.0))

        for slot_idx in range(min(12, len(previous_vehicle_ids_by_slot))):
            previous_ids = set(previous_vehicle_ids_by_slot[slot_idx] or [])
            current_ids = (
                set(current_vehicle_ids_by_slot[slot_idx] or [])
                if slot_idx < len(current_vehicle_ids_by_slot) else set()
            )
            for veh_id in previous_ids - current_ids:
                prev_dist = self.dic_vehicle_distance_previous_step.get(veh_id)
                if prev_dist is not None and prev_dist <= near_stopline_distance:
                    self._v3_discharged_0m_total[slot_idx] += 1


    def _orgnize_several_segments_attend(self, queue_in, queue_out):
        """ Prepares input for Attend model, assumes 12 lanes in/out. """
        if len(queue_in) != 12 or len(queue_out) != 12:
            print(
                f"Error ({self.inter_id}): _orgnize_several_segments_attend called with incorrect lane counts ({len(queue_in)}, {len(queue_out)}). Requires 12.")
            return [0] * (12 * 4 + 12 * 4)

        part1, part2, part3 = self._get_several_segments_attend(
            lane_vehicles=self.dic_lane_vehicle_current_step,
            vehicle_distance=self.dic_vehicle_distance_current_step,
            vehicle_speed=self.dic_vehicle_speed_current_step,
            list_lanes=self.list_entering_lanes + self.list_exiting_lanes
        )
        
        # Generate 12-element lists using the padded lane lists to ensure index safety
        run_in_part1 = [float(len(part1.get(lane, []))) if lane is not None else 0 for lane in self.list_entering_lanes]
        run_in_part2 = [float(len(part2.get(lane, []))) if lane is not None else 0 for lane in self.list_entering_lanes]
        run_in_part3 = [float(len(part3.get(lane, []))) if lane is not None else 0 for lane in self.list_entering_lanes]
        
        run_out_part1 = [float(len(part1.get(lane, []))) if lane is not None else 0 for lane in self.list_exiting_lanes]
        run_out_part2 = [float(len(part2.get(lane, []))) if lane is not None else 0 for lane in self.list_exiting_lanes]
        run_out_part3 = [float(len(part3.get(lane, []))) if lane is not None else 0 for lane in self.list_exiting_lanes]

        total_in, total_out = [], []
        for i in range(12):
            # Now these indices are safe since we're using the padded 12-element lists
            total_in.extend([run_in_part1[i], run_in_part2[i], run_in_part3[i], queue_in[i]])
            total_out.extend([run_out_part1[i], run_out_part2[i], run_out_part3[i], queue_out[i]])
        return total_in + total_out
    
    # ... (Rest of _get_several_segments_attend, _get_traffic_movement_pressure_efficient, etc.)

    def _get_several_segments_attend(self, lane_vehicles, vehicle_distance, vehicle_speed, list_lanes):
        """ Divides lanes into segments for Attend model features. """
        obs_length = 100
        part1, part2, part3 = defaultdict(list), defaultdict(list), defaultdict(list)
        for lane in list_lanes:
            if lane is None: continue
            
            # Use lane_length from CityFlowEnv's cache
            lane_len = self.dic_traffic_env_conf["lane_length"].get(lane, obs_length * 3)
            if lane_len <= 0: continue

            for vehicle in lane_vehicles.get(lane, []):
                if "shadow" in vehicle: continue
                v_speed = vehicle_speed.get(vehicle, 0)
                v_dist = vehicle_distance.get(vehicle, 0)

                if v_speed > 0.1:
                    if v_dist > lane_len - obs_length:
                        part1[lane].append(vehicle)
                    elif lane_len - 2 * obs_length < v_dist <= lane_len - obs_length:
                        part2[lane].append(vehicle)
                    elif lane_len - 3 * obs_length < v_dist <= lane_len - 2 * obs_length:
                        part3[lane].append(vehicle)
        return part1, part2, part3

    def _get_traffic_movement_pressure_efficient(self, enterings, exitings):
        """ Calculates pressure using downstream/3, assumes 12 lanes in WENSLR order. """
        if len(enterings) != 12 or len(exitings) != 12:
            return [0] * 12

        index_maps = {"W": [0, 1, 2], "E": [3, 4, 5], "N": [6, 7, 8], "S": [9, 10, 11]}
        turn_maps = ["S", "W", "N", "N", "E", "S", "W", "N", "E", "E", "S", "W"]

        outs_maps = {}
        for approach, indices in index_maps.items():
            outs_maps[approach] = sum(exitings[i] for i in indices) / 3.0

        t_m_p = [enterings[j] - outs_maps[turn_maps[j]] for j in range(12)]
        return t_m_p

    def _get_traffic_movement_vehicle_sets(self, entering_vehicle_lists):
        """
        Groups entering vehicles by movement (W-L, W-T, etc.).
        Returns a list of 12 sets (or lists), one for each movement.
        """
        movement_vehicle_sets = [set() for _ in range(12)]

        conceptual_slots = [
            {'orient': 'W', 'turn_char': 'L', 'turn_type_str': 'turn_left'},
            {'orient': 'W', 'turn_char': 'T', 'turn_type_str': 'go_straight'},
            {'orient': 'W', 'turn_char': 'R', 'turn_type_str': 'turn_right'},
            {'orient': 'E', 'turn_char': 'L', 'turn_type_str': 'turn_left'},
            {'orient': 'E', 'turn_char': 'T', 'turn_type_str': 'go_straight'},
            {'orient': 'E', 'turn_char': 'R', 'turn_type_str': 'turn_right'},
            {'orient': 'N', 'turn_char': 'L', 'turn_type_str': 'turn_left'},
            {'orient': 'N', 'turn_char': 'T', 'turn_type_str': 'go_straight'},
            {'orient': 'N', 'turn_char': 'R', 'turn_type_str': 'turn_right'},
            {'orient': 'S', 'turn_char': 'L', 'turn_type_str': 'turn_left'},
            {'orient': 'S', 'turn_char': 'T', 'turn_type_str': 'go_straight'},
            {'orient': 'S', 'turn_char': 'R', 'turn_type_str': 'turn_right'},
        ]

        # Calculate vehicle sets for each of the 12 conceptual slots
        for i, slot_info in enumerate(conceptual_slots):
            target_incoming_orient = slot_info['orient']
            target_turn_type_str = slot_info['turn_type_str']

            for road_link in self.road_links:
                start_road_id = road_link.get('startRoad')
                link_turn_type = road_link.get('type')

                if not start_road_id or not link_turn_type: continue

                actual_incoming_orient = self.road_id_2_orient.get('incoming', {}).get(start_road_id)

                if actual_incoming_orient == target_incoming_orient and link_turn_type == target_turn_type_str:
                    for lane_link_detail in road_link.get('laneLinks', []):
                        start_lane_idx_on_road = lane_link_detail.get('startLaneIndex')
                        if start_lane_idx_on_road is None:
                            continue
                        
                        full_lane_id = f"{start_road_id}_{start_lane_idx_on_road}"

                        if full_lane_id in self.list_entering_lanes:
                            try:
                                master_lane_list_idx = self.list_entering_lanes.index(full_lane_id)
                                if master_lane_list_idx < len(entering_vehicle_lists):
                                    movement_vehicle_sets[i].update(entering_vehicle_lists[master_lane_list_idx])
                            except ValueError:
                                pass
        
        # Convert sets to sorted lists for consistency if needed, but sets are fine for now.
        # But for serialization or usage, list is better? 
        # The agent expects a list of things.
        return [list(s) for s in movement_vehicle_sets]

    def _get_traffic_movement_pressure_general(self, enterings, exitings):
        """
        Calculates pressure based on dynamically determined road orientations and movements.
        Output is a fixed 12-element vector: W(L,T,R), E(L,T,R), N(L,T,R), S(L,T,R).
        """
        pressure_vector = [0.0] * 12

        conceptual_slots = [
            {'orient': 'W', 'turn_char': 'L', 'turn_type_str': 'turn_left'},
            {'orient': 'W', 'turn_char': 'T', 'turn_type_str': 'go_straight'},
            {'orient': 'W', 'turn_char': 'R', 'turn_type_str': 'turn_right'},
            {'orient': 'E', 'turn_char': 'L', 'turn_type_str': 'turn_left'},
            {'orient': 'E', 'turn_char': 'T', 'turn_type_str': 'go_straight'},
            {'orient': 'E', 'turn_char': 'R', 'turn_type_str': 'turn_right'},
            {'orient': 'N', 'turn_char': 'L', 'turn_type_str': 'turn_left'},
            {'orient': 'N', 'turn_char': 'T', 'turn_type_str': 'go_straight'},
            {'orient': 'N', 'turn_char': 'R', 'turn_type_str': 'turn_right'},
            {'orient': 'S', 'turn_char': 'L', 'turn_type_str': 'turn_left'},
            {'orient': 'S', 'turn_char': 'T', 'turn_type_str': 'go_straight'},
            {'orient': 'S', 'turn_char': 'R', 'turn_type_str': 'turn_right'},
        ]
        fixed_destination_map = {
            # Orient is the incoming approach side; destinations follow the
            # physical SUMO connection direction (left/right are not swapped).
            ('W', 'L'): 'N', ('W', 'T'): 'E', ('W', 'R'): 'S',
            ('E', 'L'): 'S', ('E', 'T'): 'W', ('E', 'R'): 'N',
            ('N', 'L'): 'E', ('N', 'T'): 'S', ('N', 'R'): 'W',
            ('S', 'L'): 'W', ('S', 'T'): 'N', ('S', 'R'): 'E',
        }

        # 1. Pre-calculate total exiting vehicles per orientation
        exiting_vehicles_by_orient = defaultdict(float)
        
        # Use the padded exiting lanes list for indexing consistency
        for lane_idx, exiting_lane_id in enumerate(self.list_exiting_lanes):
            if exiting_lane_id is None: continue
            road_id = self.lane_to_road.get(exiting_lane_id)
            if road_id in self.road_id_2_orient.get('outgoing', {}):
                orient = self.road_id_2_orient['outgoing'][road_id]
                if lane_idx < len(exitings):
                    exiting_vehicles_by_orient[orient] += exitings[lane_idx]

        # 2. Calculate pressure for each of the 12 conceptual slots
        for i, slot_info in enumerate(conceptual_slots):
            target_incoming_orient = slot_info['orient']
            target_turn_char = slot_info['turn_char']
            target_turn_type_str = slot_info['turn_type_str']

            current_slot_entering_vehicles = 0.0

            for road_link in self.road_links:
                start_road_id = road_link.get('startRoad')
                link_turn_type = road_link.get('type')

                if not start_road_id or not link_turn_type: continue

                actual_incoming_orient = self.road_id_2_orient.get('incoming', {}).get(start_road_id)

                if actual_incoming_orient == target_incoming_orient and link_turn_type == target_turn_type_str:
                    for lane_link_detail in road_link.get('laneLinks', []):
                        start_lane_idx_on_road = lane_link_detail.get('startLaneIndex')
                        if start_lane_idx_on_road is None:
                            continue
                        
                        full_lane_id = f"{start_road_id}_{start_lane_idx_on_road}"

                        # Use the padded list for indexing consistency
                        if full_lane_id in self.list_entering_lanes:
                            try:
                                master_lane_list_idx = self.list_entering_lanes.index(full_lane_id)
                                if master_lane_list_idx < len(enterings):
                                    current_slot_entering_vehicles += enterings[master_lane_list_idx]
                            except ValueError:
                                pass

            destination_key = (target_incoming_orient, target_turn_char)
            canonical_dest_orient = fixed_destination_map.get(destination_key)

            current_slot_exiting_vehicles = 0.0
            if canonical_dest_orient:
                current_slot_exiting_vehicles = exiting_vehicles_by_orient.get(canonical_dest_orient, 0.0)

            pressure_vector[i] = current_slot_entering_vehicles - current_slot_exiting_vehicles

        return pressure_vector

    def _get_part_traffic_movement_features(self):
        """ Calculates features based on vehicles in specific segments of the lanes. Assumes 12 lanes. """
        if len(self.list_entering_lanes) != 12 or len(self.list_exiting_lanes) != 12:
            num_in = len(self.list_entering_lanes)
            return [0] * num_in, [0] * num_in, [0] * num_in, [0] * num_in, [0] * num_in

        obs_length = self.dic_traffic_env_conf.get("OBS_LENGTH", 100)

        f_p_num, l_p_num, l_p_q = self._get_part_observations(
            lane_vehicles=self.dic_lane_vehicle_current_step,
            vehicle_distance=self.dic_vehicle_distance_current_step,
            vehicle_speed=self.dic_vehicle_speed_current_step,
            # lane_length=self.lane_length, # Note: using self.lane_length from CityFlowEnv
            obs_length=obs_length,
            list_lanes=self.list_entering_lanes + self.list_exiting_lanes
        )

        list_entering_part_queue = [len(l_p_q.get(lane, [])) if lane is not None else 0 for lane in self.list_entering_lanes]
        list_exiting_part_queue = [len(l_p_q.get(lane, [])) if lane is not None else 0 for lane in self.list_exiting_lanes]

        tmp_queue_efficient_part = self._get_traffic_movement_pressure_efficient(list_entering_part_queue,
                                                                                 list_exiting_part_queue)
        tmp_queue_part = self._get_traffic_movement_pressure_general(list_entering_part_queue, list_exiting_part_queue)

        list_entering_num_f = [len(f_p_num.get(lane, [])) if lane is not None else 0 for lane in self.list_entering_lanes]
        list_entering_num_l = [len(l_p_num.get(lane, [])) if lane is not None else 0 for lane in self.list_entering_lanes]
        entering_num = np.array(list_entering_num_f) + np.array(list_entering_num_l)

        list_exiting_num_f = [len(f_p_num.get(lane, [])) if lane is not None else 0 for lane in self.list_exiting_lanes]
        list_exiting_num_l = [len(l_p_num.get(lane, [])) if lane is not None else 0 for lane in self.list_exiting_lanes]
        exiting_num = np.array(list_exiting_num_f) + np.array(list_exiting_num_l)

        traffic_movement_pressure_nums = self._get_traffic_movement_pressure_general(entering_num.tolist(),
                                                                                     exiting_num.tolist())

        part_entering_running = np.array(list_entering_num_l) - np.array(list_entering_part_queue)

        return traffic_movement_pressure_nums, tmp_queue_part, tmp_queue_efficient_part, part_entering_running.tolist(), list_entering_part_queue


    def _get_part_observations(self, lane_vehicles, vehicle_distance, vehicle_speed, obs_length, list_lanes):
        """ Identifies vehicles in the first/last segments of lanes and waiting vehicles in the last segment. """
        first_part_num_vehicle = defaultdict(list)
        last_part_num_vehicle = defaultdict(list)
        last_part_queue_vehicle = defaultdict(list)

        for lane in list_lanes:
            if lane is None: continue
            
            # Now self is available since this is no longer a static method
            lane_len = self.dic_traffic_env_conf["lane_length"].get(lane, obs_length)
            
            if lane_len <= 0: continue

            last_part_obs_boundary = max(0, lane_len - obs_length)

            for vehicle in lane_vehicles.get(lane, []):
                if "shadow" in vehicle: continue

                v_dist = vehicle_distance.get(vehicle, 0)
                v_speed = vehicle_speed.get(vehicle, 0)

                if v_dist <= obs_length:
                    first_part_num_vehicle[lane].append(vehicle)

                if v_dist >= last_part_obs_boundary:
                    last_part_num_vehicle[lane].append(vehicle)
                    if v_speed <= 0.1:
                        last_part_queue_vehicle[lane].append(vehicle)

        return first_part_num_vehicle, last_part_num_vehicle, last_part_queue_vehicle

    def get_current_time(self):
        """Returns the current simulation time."""
        return self.traci_conn.simulation.getTime()

    def get_dic_vehicle_arrive_leave_time(self):
        """Returns the dictionary tracking vehicle arrival and departure times."""
        return self.dic_vehicle_arrive_leave_time

    def get_feature(self):
        """Returns the calculated dictionary of features for the current step."""
        return self.dic_feature

    def get_state(self, list_state_features):
        """
        Returns a dictionary containing the specified state features.
        """
        dic_state = {}
        for feature_name in list_state_features:
            dic_state[feature_name] = self.dic_feature.get(feature_name)
            if dic_state[feature_name] is None:
                print(f"Warning ({self.inter_id}): Requested state feature '{feature_name}' not found.")
                if "num" in feature_name or "pressure" in feature_name or "vehicle" in feature_name or "matrix" in feature_name or "attend" in feature_name:
                    dic_state[feature_name] = []
                elif "time" in feature_name:
                    dic_state[feature_name] = [0.0]
                elif "phase" in feature_name:
                    dic_state[feature_name] = [0]
                else:
                    dic_state[feature_name] = None
        return dic_state

    def get_reward(self, dic_reward_info):
        """
        Calculates the reward for the current step based on the configured metrics.
        """
        reward = 0.0
        if dic_reward_info.get("pressure", 0) != 0:
            pressure_val = np.sum(np.abs(self.dic_feature.get("pressure", [0])))
            reward += dic_reward_info["pressure"] * pressure_val

        if dic_reward_info.get("queue_length", 0) != 0:
            queue_val = np.sum(self.dic_feature.get("lane_num_waiting_vehicle_in", [0]))
            reward += dic_reward_info["queue_length"] * queue_val
        
        return reward

    def get_max_pressure_phase_action(self, pressures):
        """
        Calculates the action index corresponding to the traffic light phase
        that relieves the maximum pressure.
        """
        if not pressures or len(pressures) != 12:
            return 0

        if not self.control_phases:
            return 0

        max_pressure_sum = -float('inf')
        best_action_idx = 0

        for action_idx, phase_definition_str in enumerate(self.control_phases):
            current_phase_pressure_sum = 0.0
            movements_in_phase = []
            if len(phase_definition_str) % 2 == 0:
                for i in range(0, len(phase_definition_str), 2):
                    movements_in_phase.append(phase_definition_str[i:i + 2])
            else:
                continue

            for movement_code in movements_in_phase:
                pressure_idx = Intersection._MOVEMENT_TO_PRESSURE_IDX_MAP.get(movement_code)
                if pressure_idx is not None:
                    current_phase_pressure_sum += pressures[pressure_idx]

            if current_phase_pressure_sum > max_pressure_sum:
                max_pressure_sum = current_phase_pressure_sum
                best_action_idx = action_idx

        return best_action_idx


class SUMOEnv:
    """
    Generic SUMO Simulation Environment.

    Handles the overall simulation lifecycle, interacts with the SUMO engine via TraCI,
    manages multiple Intersection objects, and provides an API compatible with
    the original CityFlow-based environment.
    """

    def __init__(self, path_to_log, path_to_work_directory, dic_traffic_env_conf, dic_path, inter_phase_mapping=None):
        """
        Initializes the SUMO Environment.

        Args:
            path_to_log (str): Path to the logging directory.
            path_to_work_directory (str): Path to the working directory containing SUMO config files.
            dic_traffic_env_conf (dict): Environment configuration dictionary.
            dic_path (dict): Dictionary containing paths to data files.
            inter_phase_mapping (dict, optional): Dictionary mapping intersection_id to a list
                                   of allowed phase name strings.
        """
        self.path_to_log = path_to_log
        self.path_to_work_directory = path_to_work_directory
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_path = dic_path
        self.inter_phase_mapping = inter_phase_mapping if inter_phase_mapping is not None else {}

        # --- SUMO Process and TraCI Connection Management ---
        self.sumo_process = None
        self.traci_conn = None
        self._simulation_running = False
        self.sumo_net = None  # Cache for the parsed sumolib network object
        # --- End of SUMO Management ---

        self.roadnet = None # Kept for compatibility, but self.sumo_net is the primary source
        self.roads_data = {}  # Legacy format: road_id -> road_json
        self.intersections_data = {}  # Legacy format: tls_id -> inter_json

        self.list_intersection = []  # List of Intersection objects
        self.intersection_dict = {}  # Compatibility: inter_id -> parsed info dict
        self.id_to_index = {}  # Maps intersection id to index in list_intersection
        self.list_inter_log = []

        self.list_lanes = []  # List of all unique lane IDs in the network
        self.lane_length = {}  # lane_id -> length

        self.system_states = {}  # Stores results from bulk TraCI API calls
        self.current_time = 0.0
        self.actual_seed = None
        self._v9_cycle_150m_history_by_intersection = {}
        self._v32_cycle_vehicle_snapshots_by_intersection = {}
        self._v36_cycle_outbound_snapshots_by_intersection = {}
        self._v36_outbound_lane_movement_maps = {}
        self._v33_cycle_vehicle_snapshots_by_intersection = {}
        self._v26_cycle_queue_history_by_intersection = {}
        self._v10_cycle_zone_history_by_intersection = {}
        self.waiting_vehicle_list = {}
        self._seen_vehicle_ids = set()
        self._total_waiting_time_by_vehicle = {}
        # --- Travel-time aggregator (fallback for SUMO versions without getArrivedMeanTravelTime) ---
        self._depart_time_by_vehicle = {}   # vid -> depart_time
        self._arrived_tt_sum = 0.0          # sum of travel times of arrived vehicles
        self._arrived_count = 0             # count of arrived vehicles
        self._arrived_vehicle_tt = {}       # optional: vid -> travel time (for debugging/analysis)

        # --- Emergency vehicle tracking (AETT / AEWT) ---
        self._emergency_types = {'emergency', 'fire_engine', 'police'}  # vType IDs considered emergency
        self._all_emergency_vids = set()             # all emergency vehicle IDs ever seen (never cleared during sim)
        self._emergency_vehicle_ids = set()          # currently in-network emergency vehicle IDs
        self._emergency_depart_time = {}             # vid -> depart_time (for JSON detail)
        self._emergency_arrived_tt = {}              # vid -> travel_time (for JSON detail)
        self._emergency_waiting_times = {}           # vid -> accumulated_waiting_time (for JSON detail)
        self._emergency_cur_waiting = {}             # vid -> running accumulated waiting time (while in network)

        configured_port = self.dic_traffic_env_conf.get("SUMO_PORT")
        # Never allow a snapshot/path value to reach SUMO's --remote-port.
        # This can happen when legacy callers pass a load-state path through
        # a loosely typed configuration mapping; SUMO otherwise exits with
        # "Given port number ... is not numeric".
        try:
            self._default_port = (
                int(configured_port) if configured_port is not None
                else self._find_available_tcp_port()
            )
        except (TypeError, ValueError):
            self._default_port = self._find_available_tcp_port()
        if not isinstance(self._default_port, int) or self._default_port <= 0:
            self._default_port = self._find_available_tcp_port()
        if os.environ.get("V35_SUMO_DEBUG", "0") == "1":
            print(
                f"[SUMO_DEBUG pid={os.getpid()}] initialized port={self._default_port}",
                flush=True,
            )
        # --- Configuration Validation ---
        if self.dic_traffic_env_conf.get("MIN_ACTION_TIME", 15) <= self.dic_traffic_env_conf.get("YELLOW_TIME", DEFAULT_YELLOW_TIME):
            print("Warning: MIN_ACTION_TIME should ideally be greater than YELLOW_TIME.")

        # --- Ensure Log Directory Exists ---
        os.makedirs(self.path_to_log, exist_ok=True)

    @staticmethod
    def _find_available_tcp_port():
        """Ask the OS for an unused local TCP port for the TraCI server."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])


    def _load_roadnet(self):
        """
        Loads and parses the SUMO network file (.net.xml) using sumolib.
        Translates the SUMO network topology into the legacy dictionary formats
        and caches the results. This acts as the Anti-Corruption Layer for static data.
        """
        net_file_name = self.dic_traffic_env_conf.get("ROADNET_FILE", "map.net.xml")
        net_file_path = os.path.join(self.dic_path.get("PATH_TO_DATA", self.path_to_work_directory), net_file_name)

        if not os.path.exists(net_file_path):
            raise FileNotFoundError(f"SUMO network file '{net_file_path}' not found.")

        if getattr(self, '_verbose_reset', True):
            print(f"Loading SUMO network from: {net_file_path}")
        try:
            # 1. One-Time Parsing and Caching of the sumolib object
            self.sumo_net = sumolib.net.readNet(net_file_path)
        except Exception as e:
            # Catches XML parsing errors and other sumolib issues
            raise ValueError(f"Failed to parse SUMO network file '{net_file_path}': {e}")

        # 2. Translate SUMO topology into legacy formats
        # --- Translate Intersections (Traffic Light Systems) ---
        self.intersections_data = {}
        
        # NEW: TLS↔node mappings for robust planning
        self.tls_to_nodes = {}   # TLS id -> [node ids]
        self.node_to_tls = {}    # node id -> TLS id
        
        for tls in self.sumo_net.getTrafficLights():
            tls_id = tls.getID()

            # Nodes controlled by this TLS (empty for some nets / versions)
            nodes = getattr(tls, "getNodes", lambda: [])()
            node_ids = [n.getID() for n in nodes] if nodes else []

            # Coordinates (your existing code)
            nodes_for_xy = nodes
            if nodes_for_xy:
                xs, ys = zip(*(n.getCoord() for n in nodes_for_xy))
                x, y = sum(xs) / len(xs), sum(ys) / len(ys)
            else:
                try:
                    node = self.sumo_net.getNode(tls_id)
                    x, y = node.getCoord()
                except Exception:
                    x, y = 0.0, 0.0

            # Save mapping + some useful metadata
            self.tls_to_nodes[tls_id] = node_ids
            for nid in node_ids:
                self.node_to_tls[nid] = tls_id

            self.intersections_data[tls_id] = {
                "id": tls_id,
                "virtual": False,
                "point": {"x": x, "y": y},
                "controlled_nodes": node_ids,         # NEW
                "graph_node_ids": node_ids or [tls_id]# NEW: how planners can address this TLS in graphs
            }

        # --- Translate Roads (Edges) ---
        self.roads_data = {}
        for edge in self.sumo_net.getEdges():
            edge_id = edge.getID()
            if edge_id.startswith(":"): # Ignore internal edges
                continue
            
            points = [{"x": x, "y": y} for x, y in edge.getShape()]
            lanes_info = [{"maxSpeed": lane.getSpeed()} for lane in edge.getLanes()]

            self.roads_data[edge_id] = {
                "id": edge_id,
                "lanes": lanes_info,
                "startIntersection": edge.getFromNode().getID(),
                "endIntersection": edge.getToNode().getID(),
                "points": points,
            }

        self.dic_traffic_env_conf["NUM_INTERSECTIONS"] = len(self.intersections_data)
        if getattr(self, '_verbose_reset', True):
            print(f"Found {len(self.roads_data)} roads (edges) and {len(self.intersections_data)} signalized intersections (TLS).")

    def get_lane_speed(self, lane_id):
        """Returns the speed limit of the specified lane."""
        return self.sumo_net.getLane(lane_id).getSpeed()

    def _get_lane_length(self):
        """Calculates and caches the length of each lane from the parsed sumolib network."""
        self.lane_length = {}
        if not self.sumo_net:
            print("Warning: Cannot calculate lane lengths, SUMO network not loaded.")
            return

        for edge in self.sumo_net.getEdges():
            # Include all lanes, even internal junction lanes
            for lane in edge.getLanes():
                self.lane_length[lane.getID()] = lane.getLength()

        if getattr(self, '_verbose_reset', True):
            print(f"Cached lengths for {len(self.lane_length)} lanes.")

    def _adjacency_extraction(self):
        """
        Extracts adjacency information based on geometric proximity.
        """
        if not self.intersections_data:
            return {}

        adjacency_results = {}
        inter_ids = list(self.intersections_data.keys())
        inter_id_to_idx_map = {inter_id: i for i, inter_id in enumerate(inter_ids)}
        num_intersections = len(inter_ids)
        top_k = min(self.dic_traffic_env_conf.get("TOP_K_ADJACENCY", 5), num_intersections)

        if getattr(self, '_verbose_reset', True):
            print(f"Calculating adjacency for {num_intersections} intersections (top_k={top_k})...")

        for i, inter_id in enumerate(inter_ids):
            loc_i = self.intersections_data[inter_id]["point"]
            distances = np.full(num_intersections, np.inf)

            for j, other_inter_id in enumerate(inter_ids):
                if i == j:
                    distances[j] = 0
                    continue
                loc_j = self.intersections_data[other_inter_id]["point"]
                distances[j] = self._cal_distance(loc_i, loc_j)
            
            if num_intersections <= top_k:
                neighbor_indices = [idx for idx in range(num_intersections) if idx != i]
                neighbor_distances = distances[neighbor_indices]
                sorted_neighbor_indices = np.array(neighbor_indices)[np.argsort(neighbor_distances)]
                adjacency_row = [i] + sorted_neighbor_indices.tolist()
            else:
                partitioned_indices = np.argpartition(distances, top_k)
                candidate_indices = partitioned_indices[:top_k + 1]
                candidate_distances = distances[candidate_indices]
                sorted_candidate_indices_local = np.argsort(candidate_distances)
                sorted_candidate_indices_global = candidate_indices[sorted_candidate_indices_local]
                neighbor_indices = [idx for idx in sorted_candidate_indices_global if idx != i][:top_k]
                adjacency_row = [i] + neighbor_indices

            adjacency_results[inter_id] = {
                "adjacency_row": adjacency_row,
                "total_inter_num": num_intersections,
                "inter_id_to_index": inter_id_to_idx_map,
            }

        return adjacency_results

    @staticmethod
    def _cal_distance(loc_dict1, loc_dict2):
        """Calculates Euclidean distance between two points."""
        x1, y1 = loc_dict1.get('x', 0), loc_dict1.get('y', 0)
        x2, y2 = loc_dict2.get('x', 0), loc_dict2.get('y', 0)
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def reset(self, use_gui=False, seed=None, load_state_path=None, verbose=True):
        """
        Resets the simulation environment.
        - Shuts down any existing SUMO simulation.
        - Dynamically creates the .sumocfg file in the work directory.
        - Launches a new SUMO instance (with or without GUI).
        - Establishes a TraCI connection.
        - Initializes Intersection objects.
        - Retrieves initial state.
        
        Args:
            use_gui (bool): Whether to use sumo-gui.
            seed (int, optional): Random seed for the simulation.
            load_state_path (str, optional): If provided, starts the simulation from this snapshot file.
        """
        self._verbose_reset = verbose
        if verbose:
            print("================ Starting Environment Reset ================")
        self.close()

        self.current_time = 0.0
        self.waiting_vehicle_list = {}
        self._seen_vehicle_ids = set()
        self._total_waiting_time_by_vehicle = {}
        # Reset travel-time aggregate
        self._depart_time_by_vehicle = {}
        self._arrived_tt_sum = 0.0
        self._arrived_count = 0
        self._arrived_vehicle_tt = {}
        # Reset emergency vehicle tracking
        self._all_emergency_vids = set()
        self._emergency_vehicle_ids = set()
        self._emergency_depart_time = {}
        self._emergency_arrived_tt = {}
        self._emergency_waiting_times = {}
        self._emergency_cur_waiting = {}

        # 1. Load static network data once
        if self.sumo_net is None:
            self._load_roadnet()
            self._get_lane_length()

        # 2. Prepare SUMO configuration files in the work directory
        data_path = self.dic_path.get("PATH_TO_DATA", self.path_to_work_directory)

        # Get source file names and paths
        net_file_name = self.dic_traffic_env_conf.get("ROADNET_FILE", "map.net.xml")
        flow_file_name = self.dic_traffic_env_conf.get("TRAFFIC_FILE", "route.rou.xml")
        
        source_net_path = os.path.join(data_path, net_file_name)
        source_flow_path = os.path.join(data_path, flow_file_name)

        if not os.path.exists(source_net_path):
            raise FileNotFoundError(f"SUMO network file not found at source: {source_net_path}")
        if not os.path.exists(source_flow_path):
            raise FileNotFoundError(f"SUMO route file not found at source: {source_flow_path}")

        # Define destination paths in the work directory
        dest_net_path = os.path.join(self.path_to_work_directory, net_file_name)
        dest_flow_path = os.path.join(self.path_to_work_directory, flow_file_name)

        # Copy files to the work directory
        shutil.copy(source_net_path, dest_net_path)
        shutil.copy(source_flow_path, dest_flow_path)
        
        # Generate and write the sumocfg file
        sumocfg_file_name = self.dic_traffic_env_conf.get("SUMOCFG_FILE", "map.sumocfg")
        sumocfg_path = os.path.join(self.path_to_work_directory, sumocfg_file_name)

        sumocfg_content = f"""<configuration>
    <input>
        <net-file value="{net_file_name}"/>
        <route-files value="{flow_file_name}"/>
    </input>
</configuration>
"""
        with open(sumocfg_path, 'w') as f:
            f.write(sumocfg_content)
        
        # 3. Prepare and Launch SUMO
        # Use full path to avoid conda wrapper conflicts
        sumo_home = os.environ.get("SUMO_HOME", "")
        # SUMO binary names differ by platform.  The previous implementation
        # always selected the Windows ``.exe`` name, which fails on HPC/Linux
        # even when SUMO_HOME is correctly configured.
        if sumo_home and os.path.exists(os.path.join(sumo_home, "bin")):
            binary_name = "sumo-gui" if use_gui else "sumo"
            if os.name == "nt":
                binary_name += ".exe"
            sumo_binary = os.path.join(sumo_home, "bin", binary_name)
        else:
            sumo_binary = "sumo-gui" if use_gui else "sumo"
        
        if seed is None:
            seed = int(np.random.randint(0, 10000))
        self.actual_seed = int(seed)

        # Allocate a fresh TraCI port for every SUMO process.  A SUMO process
        # from a previous reset may still be in TIME_WAIT (or another Ray
        # actor may have raced for the same ephemeral port), so reusing the
        # initialization-time port can fail with "Address already in use".
        # Keep an explicitly configured SUMO_PORT stable for legacy callers.
        if self.dic_traffic_env_conf.get("SUMO_PORT") is None:
            self._default_port = self._find_available_tcp_port()
        elif not isinstance(self._default_port, int) or self._default_port <= 0:
            self._default_port = int(self.dic_traffic_env_conf["SUMO_PORT"])

        # Implementation note.
        sim_time = self.dic_traffic_env_conf.get("RUN_COUNTS", 3600)
        sumo_cmd = [
            sumo_binary, "-c", sumocfg_path,
            "--remote-port", str(self._default_port),
            "--seed", str(seed),
            "--step-length", str(self.dic_traffic_env_conf.get("INTERVAL", 1.0)),
            "--no-warnings", "true",
            # Implementation note.
            # Implementation note.
            "--ignore-route-errors", "true",
            "--start",  # Implementation note.
            # "--end", str(sim_time)
        ]
        if self.dic_traffic_env_conf.get("SAVE_STATE_RNG", False):
            sumo_cmd.extend(["--save-state.rng", "true"])
        save_state_precision = self.dic_traffic_env_conf.get(
            "SAVE_STATE_PRECISION")
        if save_state_precision is not None:
            sumo_cmd.extend([
                "--save-state.precision", str(int(save_state_precision))
            ])

        # MODIFICATION: Add the --load-state argument if a path is provided
        if load_state_path and os.path.exists(load_state_path):
            if getattr(self, '_verbose_reset', True):
                print(f"Attempting to load simulation from state: {load_state_path}")
            sumo_cmd.extend(["--load-state", load_state_path])
        elif load_state_path and getattr(self, '_verbose_reset', True):
            print(f"Warning: Snapshot file for loading not found at {load_state_path}. Starting new simulation.")

        if getattr(self, '_verbose_reset', True):
            print(f"Launching SUMO with command: {' '.join(sumo_cmd)}")
        sumo_stdout_log = os.path.join(self.path_to_log, "sumo_stdout.log")
        sumo_stderr_log = os.path.join(self.path_to_log, "sumo_stderr.log")
        sumo_command_log = os.path.join(self.path_to_log, "sumo_command.txt")

        # Persist exact argv. This is intentionally one argument per line so
        # paths containing spaces remain unambiguous in Ray worker logs.
        with open(sumo_command_log, "w", encoding="utf-8") as f_command:
            f_command.write("\n".join(str(argument) for argument in sumo_cmd))
            f_command.write("\n")

        try:
            # Port probing and process bind are inherently racy across Ray
            # actors. Retry with a newly allocated port when SUMO loses that
            # race, instead of failing the whole rollout.
            for _port_attempt in range(5):
                sumo_cmd[sumo_cmd.index("--remote-port") + 1] = str(self._default_port)
                with open(sumo_stdout_log, 'w') as f_out, open(sumo_stderr_log, 'w') as f_err:
                    self.sumo_process = subprocess.Popen(sumo_cmd, stdout=f_out, stderr=f_err)
                time.sleep(1)
                if self.sumo_process.poll() is None:
                    if os.environ.get("V35_SUMO_DEBUG", "0") == "1":
                        print(f"[SUMO_DEBUG pid={os.getpid()}] process alive port={self._default_port}", flush=True)
                    break
                try:
                    with open(sumo_stderr_log, 'r') as f_err_read:
                        _sumo_err = f_err_read.read()
                except IOError:
                    _sumo_err = ""
                if "Address already in use" not in _sumo_err or _port_attempt == 4:
                    break
                self._default_port = self._find_available_tcp_port()

            # Check if the SUMO process terminated prematurely
            if self.sumo_process.poll() is not None:
                error_message = f"SUMO process terminated unexpectedly. Check SUMO logs for details:\n"
                error_message += f"  - STDERR: {sumo_stderr_log}\n"
                try:
                    with open(sumo_stderr_log, 'r') as f_err_read:
                        error_details = f_err_read.read()
                        if error_details.strip():
                            error_message += f"\n--- SUMO Error Log Content ---\n{error_details}\n----------------------------"
                except IOError:
                    error_message += "(Could not read error log file.)"
                self.close()
                raise RuntimeError(error_message)

            # 4. Establish TraCI Connection
            if os.environ.get("V35_SUMO_DEBUG", "0") == "1":
                print(f"[SUMO_DEBUG pid={os.getpid()}] connecting TraCI port={self._default_port}", flush=True)
            self.traci_conn = traci.connect(port=self._default_port, numRetries=10)
            if os.environ.get("V35_SUMO_DEBUG", "0") == "1":
                print(f"[SUMO_DEBUG pid={os.getpid()}] TraCI connected port={self._default_port}", flush=True)
            
            # Subscribe to lane-based information once
            self._subscribe_lanes()

            self._simulation_running = True
            if getattr(self, '_verbose_reset', True):
                print(f"Successfully connected to SUMO (seed: {seed}).")

        except FileNotFoundError:
            raise EnvironmentError(f"'{sumo_binary}' not found. Please ensure SUMO is installed and in your system's PATH.")
        except traci.TraCIException as e:
            error_message = (f"Failed to connect to SUMO via TraCI. This often means SUMO crashed on startup. "
                             f"Please check the SUMO log files for errors:\n"
                             f"  - STDERR: {sumo_stderr_log}\n"
                             f"Original TraCI error: {e}")
            try:
                with open(sumo_stderr_log, 'r') as f_err_read:
                     error_details = f_err_read.read()
                     if error_details.strip():
                         error_message += f"\n\n--- SUMO Error Log Content ---\n{error_details}\n----------------------------"
            except IOError:
                pass
            self.close()
            raise EnvironmentError(error_message) from e
        except Exception as e:
            self.close()
            raise

        # 5. Calculate Adjacency
        self.adjacency_map = self._adjacency_extraction()

        # 6. Initialize Intersection Objects
        self.list_intersection = []
        self.id_to_index = {}
        self.list_inter_log = []
        if getattr(self, '_verbose_reset', True):
            print(f"Creating Intersection objects for {len(self.intersections_data)} intersections...")
        
        # Pass the global lane_length dict to the intersection for use in feature calcs
        self.dic_traffic_env_conf["lane_length"] = self.lane_length
        
        for idx, (inter_id, _) in enumerate(self.intersections_data.items()):
            adjacency_info = self.adjacency_map.get(inter_id, {})
            custom_phases = self.inter_phase_mapping.get(inter_id)
            try:
                intersection = Intersection(
                    tls_id=inter_id,
                    dic_traffic_env_conf=self.dic_traffic_env_conf,
                    traci_conn=self.traci_conn,
                    sumo_net=self.sumo_net,
                    path_to_log=self.path_to_log,
                    adjacency_info=adjacency_info,
                    custom_phase_list=custom_phases
                )
                self.list_intersection.append(intersection)
                self.id_to_index[inter_id] = idx
                self.list_inter_log.append([])
            except Exception as e:
                print(f"ERROR: Failed to initialize Intersection object for {inter_id}: {e}")
                self.close()
                raise

        # Intersection construction discovers the canonical entering/exiting
        # lanes used by phase-level features. Subscribe again after that
        # discovery; the earlier connection-time subscription can miss them.
        self._subscribe_lanes()

        # 7. Get Initial State from Simulator
        # MODIFICATION: If loading from state, the first step is not needed as SUMO is already at that time.
        # Otherwise, take one step to populate the network with initial vehicles.
        if getattr(self, '_verbose_reset', True):
            print("Getting initial simulator state...")
        if not load_state_path:
            self.traci_conn.simulationStep()
        self._update_system_states()

        # 8. Update Intersection Measurements
        if getattr(self, '_verbose_reset', True):
            print("Updating initial intersection measurements...")
        for inter in self.list_intersection:
            inter.update_current_measurements(self.system_states)

        # 9. Create intersection_dict (for compatibility)
        if getattr(self, '_verbose_reset', True):
            print("Creating compatibility intersection_dict...")
        self.create_intersection_dict()

        # 10. Get Formatted Initial State
        state, _ = self.get_state()

        if getattr(self, '_verbose_reset', True):
            print("================ Environment Reset Complete ================")
        self._verbose_reset = True  # restore for later close() or next reset
        return state

    def _update_system_states(self, force_direct_vehicle_queries=False):
        """
        Subscribes to dynamic vehicle data and retrieves all subscription
        results in a single batch.
        """
        if not self._simulation_running:
            return

        try:
            # 1. Dynamic Vehicle Subscriptions (per step)
            current_vehicle_ids = tuple(
                self.traci_conn.vehicle.getIDList())
            current_vehicle_id_set = set(current_vehicle_ids)
            for vehicle_id in current_vehicle_ids:
                self.traci_conn.vehicle.subscribe(vehicle_id, [
                    traci.constants.VAR_SPEED,
                    traci.constants.VAR_LANE_ID,
                    traci.constants.VAR_LANEPOSITION
                ])

            # 2. Batch Data Retrieval - Use domain-specific calls
            raw_vehicle_results = (
                self.traci_conn.vehicle.getAllSubscriptionResults() or {})
            lane_results = (
                {} if force_direct_vehicle_queries
                else (self.traci_conn.lane.getAllSubscriptionResults() or {})
            )

            # 3. Data Transformation
            self.system_states = {
                "get_lane_vehicles": defaultdict(list),
                "get_lane_waiting_vehicle_count": defaultdict(int),
                "get_vehicle_speed": {},
                "get_vehicle_lane": {},
                "get_vehicle_distance": {},
            }
            
            # Process lane subscription results to get the list of vehicles per lane
            for lane_id, data in lane_results.items():
                # TraCI normally returns a list, but after ``loadState`` some
                # versions return a tuple.  Later code supplements this lane
                # collection with dynamically subscribed vehicles, so it must
                # always be mutable.
                self.system_states["get_lane_vehicles"][lane_id] = [
                    vehicle_id
                    for vehicle_id in data.get(
                        traci.constants.LAST_STEP_VEHICLE_ID_LIST, ())
                    if vehicle_id in current_vehicle_id_set
                ]
            
            # Process vehicle subscription results to get speeds and calculate queue length
            for vehicle_id in current_vehicle_ids:
                data = (
                    {} if force_direct_vehicle_queries
                    else (raw_vehicle_results.get(vehicle_id) or {})
                )
                try:
                    speed = data.get(traci.constants.VAR_SPEED)
                    if speed is None:
                        speed = self.traci_conn.vehicle.getSpeed(vehicle_id)
                    lane_id = data.get(traci.constants.VAR_LANE_ID)
                    if lane_id is None:
                        lane_id = self.traci_conn.vehicle.getLaneID(vehicle_id)
                    sumo_position = data.get(traci.constants.VAR_LANEPOSITION)
                    if sumo_position is None:
                        sumo_position = self.traci_conn.vehicle.getLanePosition(
                            vehicle_id)
                except traci.TraCIException:
                    # The vehicle may arrive between getIDList() and the direct
                    # fallback query. It must not survive in the cached state.
                    continue
                
                self.system_states["get_vehicle_speed"][vehicle_id] = speed
                self.system_states["get_vehicle_lane"][vehicle_id] = lane_id
                lane_vehicle_ids = self.system_states["get_lane_vehicles"][lane_id]
                if vehicle_id not in lane_vehicle_ids:
                    lane_vehicle_ids.append(vehicle_id)
                
                # If the vehicle's speed is below the threshold, increment the waiting count for its lane
                if speed < 0.1:
                    self.system_states["get_lane_waiting_vehicle_count"][lane_id] += 1
                
                # The rest of the logic for distance remains the same
                if lane_id not in self.lane_length:
                    try:
                        self.lane_length[lane_id] = self.traci_conn.lane.getLength(lane_id)
                    except traci.TraCIException: continue
                
                # Convert VAR_LANEPOSITION (distance from lane start) to
                # distance from stop-line so that two_stages.py camera FOV
                # filter (dist <= 50 m) correctly keeps vehicles near the stop-line.
                lane_len = self.lane_length.get(lane_id, 0.0)
                self.system_states["get_vehicle_distance"][vehicle_id] = max(0.0, lane_len - sumo_position)

            cached_vehicle_ids = set(
                self.system_states["get_vehicle_speed"])
            if cached_vehicle_ids != current_vehicle_id_set:
                raise RuntimeError(
                    "SUMO vehicle cache does not match vehicle.getIDList(): "
                    f"missing={sorted(current_vehicle_id_set - cached_vehicle_ids)}, "
                    f"stale={sorted(cached_vehicle_ids - current_vehicle_id_set)}"
                )

            # Get current sim time once
            now_time = self.traci_conn.simulation.getTime()

            # --- Update travel-time aggregate ---
            try:
                departed_ids = self.traci_conn.simulation.getDepartedIDList()
                arrived_ids  = self.traci_conn.simulation.getArrivedIDList()
            except traci.TraCIException:
                departed_ids, arrived_ids = [], []

            # Record depart times for vehicles that started this step
            for vid in departed_ids:
                self._depart_time_by_vehicle[vid] = now_time
                # Check if this is an emergency vehicle
                try:
                    vtype = self.traci_conn.vehicle.getTypeID(vid)
                    if vtype in self._emergency_types:
                        self._all_emergency_vids.add(vid)
                        self._emergency_vehicle_ids.add(vid)
                        self._emergency_depart_time[vid] = now_time
                except traci.TraCIException:
                    pass

            # Update accumulated waiting time for emergency vehicles still in network
            # (must be done BEFORE processing arrivals, because arrived vehicles are already removed)
            interval = self.dic_traffic_env_conf.get("INTERVAL", 1.0)
            current_speeds = self.system_states.get("get_vehicle_speed", {})
            for vid in list(self._emergency_vehicle_ids):
                speed = current_speeds.get(vid, 1.0)  # default > 0.1 if not found
                if speed < 0.1:
                    self._emergency_cur_waiting[vid] = self._emergency_cur_waiting.get(vid, 0.0) + interval

            # For vehicles that finished this step, accumulate travel times
            for vid in arrived_ids:
                depart_time = self._depart_time_by_vehicle.pop(vid, None)
                if depart_time is not None:
                    tt = now_time - depart_time
                    self._arrived_tt_sum += tt
                    self._arrived_count += 1
                    self._arrived_vehicle_tt[vid] = tt

                    # Emergency vehicle: record AETT and AEWT
                    if vid in self._emergency_vehicle_ids:
                        self._emergency_arrived_tt[vid] = tt
                        self._emergency_waiting_times[vid] = self._emergency_cur_waiting.pop(vid, 0.0)
                        self._emergency_vehicle_ids.discard(vid)
                        self._emergency_depart_time.pop(vid, None)

            # Commit the time for the environment
            self.current_time = now_time

        except traci.TraCIException as e:
            print(f"TraCIException during state update: {e}. Simulation may have ended.")
            self.close()
            raise RuntimeError(f"TraCI connection lost: {e}") from e

    def create_intersection_dict(self):
        """
        Creates the `intersection_dict` attribute for API compatibility, based on the
        dynamically discovered properties of the Intersection objects.
        """
        self.intersection_dict = {}
        if getattr(self, '_verbose_reset', True):
            print(f"Populating intersection_dict for {len(self.list_intersection)} intersections...")

        for intersection_obj in self.list_intersection:
            inter_id = intersection_obj.inter_id
            
            agent_intersection_info = {
                "id": inter_id,
                "phases": {},
                "roads": {},
                "control_phases": intersection_obj.control_phases
            }

            for phase_name, sumo_idx in intersection_obj.phase_name_2_cityflow_idx.items():
                if sumo_idx < len(intersection_obj.phases):
                    phase_def = intersection_obj.phases[sumo_idx]
                    agent_intersection_info["phases"][phase_name] = {"time": phase_def.duration, "idx": sumo_idx}

            all_roads = {**intersection_obj.incoming_roads, **intersection_obj.outgoing_roads}
            for road_id, road_obj in all_roads.items():
                is_incoming = road_id in intersection_obj.incoming_roads
                road_type = "incoming" if is_incoming else "outgoing"
                
                location_code = intersection_obj.road_id_2_orient.get(road_type, {}).get(road_id)
                location = location_dict_reverse.get(location_code)

                road_info = {
                    "location": location, "type": road_type,
                    "length": road_obj.getLength(),
                    "max_speed": road_obj.getSpeed(),
                    "num_lanes": len(road_obj.getLanes()),
                    "lanes": defaultdict(list), "go_straight": None,
                    "turn_left": None, "turn_right": None
                }
                
                if is_incoming:
                    for link in intersection_obj.road_links:
                        if link["startRoad"] == road_id:
                            turn_type = link["type"]
                            end_road_id = link["endRoad"]

                            if turn_type == "go_straight":
                                road_info["go_straight"] = end_road_id
                            elif turn_type == "turn_left":
                                road_info["turn_left"] = end_road_id
                            elif turn_type == "turn_right":
                                road_info["turn_right"] = end_road_id

                            for lane_link in link.get("laneLinks", []):
                                start_lane_idx = lane_link.get("startLaneIndex")
                                if start_lane_idx is not None and start_lane_idx not in road_info["lanes"][turn_type]:
                                    road_info["lanes"][turn_type].append(start_lane_idx)
                else:  # This is an outgoing road
                    # Expanded logic for outgoing roads
                    for link in intersection_obj.road_links:
                        if link["endRoad"] == road_id:
                            turn_type = link["type"]
                            end_road_id = link["endRoad"] # This is the same as road_id

                            # This makes the data structure consistent with incoming roads.
                            # The value will be the ID of the outgoing road itself.
                            if turn_type == "go_straight":
                                road_info["go_straight"] = end_road_id
                            elif turn_type == "turn_left":
                                road_info["turn_left"] = end_road_id
                            elif turn_type == "turn_right":
                                road_info["turn_right"] = end_road_id
                            
                            # Populate the lanes field based on the receiving lane index
                            for lane_link in link.get("laneLinks", []):
                                end_lane_idx = lane_link.get("endLaneIndex")
                                if end_lane_idx is not None and end_lane_idx not in road_info["lanes"][turn_type]:
                                    road_info["lanes"][turn_type].append(end_lane_idx)
                
                # Finalize and sort the lane lists for consistency
                road_info["lanes"] = {k: sorted(v) for k, v in road_info["lanes"].items()}
                agent_intersection_info["roads"][road_id] = road_info

            self.intersection_dict[inter_id] = agent_intersection_info

    def _v36_outbound_lane_movement_map(self, source_intersection):
        """Map source outgoing lanes to target movements at the next TLS.

        This is deliberately a frame-time lane mapping. It does not inspect a
        vehicle's future route, so later lane changes do not alter the saved
        coordination label.
        """
        cached = self._v36_outbound_lane_movement_maps.get(
            source_intersection.inter_id)
        if cached is not None:
            return cached

        turn_chars = {
            "go_straight": "T",
            "turn_left": "L",
            "turn_right": "R",
        }
        lane_to_movements = {}
        for lane_id in source_intersection.list_exiting_lanes:
            if lane_id is None:
                continue
            try:
                lane = source_intersection.sumo_net.getLane(lane_id)
                edge_id = lane.getEdge().getID()
                lane_index = int(lane.getIndex())
            except (KeyError, AttributeError, TypeError, ValueError):
                continue

            movements = set()
            for target_intersection in self.list_intersection:
                if target_intersection.inter_id == source_intersection.inter_id:
                    continue
                target_entry = target_intersection.road_id_2_orient.get(
                    "incoming", {}).get(edge_id)
                if not target_entry:
                    continue
                for road_link in target_intersection.road_links:
                    if road_link.get("startRoad") != edge_id:
                        continue
                    turn_char = turn_chars.get(road_link.get("type"))
                    if turn_char is None:
                        continue
                    if any(
                            int(detail.get("startLaneIndex")) == lane_index
                            for detail in road_link.get("laneLinks", [])
                            if detail.get("startLaneIndex") is not None):
                        movements.add(f"{target_entry}{turn_char}")
            if movements:
                lane_to_movements[lane_id] = sorted(movements)

        self._v36_outbound_lane_movement_maps[
            source_intersection.inter_id] = lane_to_movements
        return lane_to_movements

    def step(self, action_dict, min_action_time=15, inner_step_callback=None):
        """
        Controls intersections and advances the simulation for a specified duration.
        """
        if not self._simulation_running:
            next_state, _ = self.get_state()
            return next_state, self.get_reward(), True, {"error": "Simulation not running."}

        # Reset action flags at the start of each new VLM decision cycle
        for inter in self.list_intersection:
            inter._action_already_set = False

        interval = self.dic_traffic_env_conf.get("INTERVAL", 1.0)
        num_inner_steps = int(min_action_time / interval)
        v9_sample_interval = int(self.dic_traffic_env_conf.get(
            "V9_TEMPORAL_FRAME_INTERVAL", 10))
        self._v9_cycle_150m_history_by_intersection = {
            inter.inter_id: [] for inter in self.list_intersection
        }
        self._v32_cycle_vehicle_snapshots_by_intersection = {
            inter.inter_id: [] for inter in self.list_intersection
        }
        self._v36_cycle_outbound_snapshots_by_intersection = {
            inter.inter_id: [] for inter in self.list_intersection
        }
        self._v36_outbound_lane_movement_maps = {}
        self._v33_cycle_vehicle_snapshots_by_intersection = {
            inter.inter_id: [] for inter in self.list_intersection
        }
        self._v26_cycle_queue_history_by_intersection = {
            inter.inter_id: [] for inter in self.list_intersection
        }
        self._v10_cycle_zone_history_by_intersection = {
            inter.inter_id: [] for inter in self.list_intersection
        }

        for inner_i in range(num_inner_steps):
            if not self._simulation_running: break

            for inter in self.list_intersection:
                inter.update_previous_measurements()

            for inter in self.list_intersection:
                action = action_dict.get(inter.inter_id, -1)
                inter.set_signal(action, action_pattern="set")

            try:
                self.traci_conn.simulationStep()
            except traci.TraCIException as e:
                print(f'TraCIException during step: {e}. Simulation may have ended.')
                self.close()
                next_state, _ = self.get_state()
                return next_state, self.get_reward(), True, {"error": str(e)}

            self._update_system_states()

            for inter in self.list_intersection:
                inter.update_current_measurements(self.system_states)

            if v9_sample_interval > 0 and (inner_i + 1) % v9_sample_interval == 0:
                for inter in self.list_intersection:
                    camera_distance = float(self.dic_traffic_env_conf.get(
                        "CAMERA_VIEW_DISTANCE", 150.0))
                    movement_ids = inter.dic_feature.get(
                        "traffic_movement_vehicle_ids_150m", [])
                    movement_counts = [len(vehicle_ids) for vehicle_ids in movement_ids]
                    self._v9_cycle_150m_history_by_intersection.setdefault(
                        inter.inter_id, []).append(movement_counts)
                    snapshot_vehicle_ids = [
                        list(vehicle_ids)
                        for vehicle_ids in inter.dic_feature.get(
                            "traffic_movement_vehicle_ids", [])
                    ]
                    snapshot_ids = {
                        vehicle_id
                        for vehicle_ids in snapshot_vehicle_ids
                        for vehicle_id in vehicle_ids
                    }
                    self._v32_cycle_vehicle_snapshots_by_intersection.setdefault(
                        inter.inter_id, []).append({
                            "time_s": int(inner_i + 1),
                            "movement_vehicle_ids": snapshot_vehicle_ids,
                            "vehicle_distance": {
                                vehicle_id: inter.dic_vehicle_distance_current_step.get(vehicle_id)
                                for vehicle_id in snapshot_ids
                            },
                            "vehicle_speed": {
                                vehicle_id: inter.dic_vehicle_speed_current_step.get(vehicle_id)
                                for vehicle_id in snapshot_ids
                            },
                        })
                    if is_new_coordination_enabled(self.dic_traffic_env_conf):
                        outbound_by_direction = {}
                        outbound_by_movement = defaultdict(list)
                        mapping_available_by_direction = {}
                        outbound_distance_by_vehicle = {}
                        lane_to_movements = self._v36_outbound_lane_movement_map(
                            inter)
                        for direction, offset in (("W", 0), ("E", 3), ("N", 6), ("S", 9)):
                            direction_ids = []
                            direction_lanes = inter.list_exiting_lanes[
                                offset:offset + 3]
                            mapping_available_by_direction[direction] = any(
                                lane_id is not None
                                and bool(lane_to_movements.get(lane_id))
                                for lane_id in direction_lanes
                            )
                            for lane_id in direction_lanes:
                                if lane_id is None:
                                    continue
                                try:
                                    lane_length = float(inter.sumo_net.getLane(lane_id).getLength())
                                except (KeyError, AttributeError, TypeError, ValueError):
                                    continue
                                for vehicle_id in inter.dic_lane_vehicle_current_step.get(lane_id, []):
                                    distance_to_lane_end = inter.dic_vehicle_distance_current_step.get(vehicle_id)
                                    if distance_to_lane_end is None:
                                        continue
                                    distance_from_source = lane_length - float(distance_to_lane_end)
                                    if 0.0 <= distance_from_source <= camera_distance:
                                        direction_ids.append(vehicle_id)
                                        outbound_distance_by_vehicle[vehicle_id] = distance_from_source
                                        for movement in lane_to_movements.get(
                                                lane_id, []):
                                            outbound_by_movement[movement].append(
                                                vehicle_id)
                            outbound_by_direction[direction] = sorted(set(direction_ids))
                        outbound_by_movement = {
                            movement: sorted(set(vehicle_ids))
                            for movement, vehicle_ids in outbound_by_movement.items()
                        }
                        self._v36_cycle_outbound_snapshots_by_intersection.setdefault(
                            inter.inter_id, []).append({
                                # This snapshot is captured after the completed SUMO
                                # tick, so its timestamp must match the video frame
                                # timestamp (5, 10, ..., 30s), not a zero-based index.
                                "frame_index": len(
                                    self._v36_cycle_outbound_snapshots_by_intersection.get(
                                        inter.inter_id, [])) + 1,
                                "frame_time_s": float(self.get_current_time()),
                                "sim_time_s": float(self.get_current_time()),
                                "outbound_vehicle_ids": outbound_by_direction,
                                "outbound_vehicle_ids_by_movement": (
                                    outbound_by_movement),
                                "outbound_lane_mapping_available": bool(
                                    lane_to_movements),
                                "outbound_movement_mapping_available_by_direction": (
                                    mapping_available_by_direction),
                                "outbound_distance_from_source_m": outbound_distance_by_vehicle,
                            })
                    queue_counts = [
                        sum(
                            inter.dic_vehicle_speed_current_step.get(veh_id, 1.0) < 0.1
                            for veh_id in vehicle_ids
                        )
                        for vehicle_ids in movement_ids
                    ]
                    self._v26_cycle_queue_history_by_intersection.setdefault(
                        inter.inter_id, []).append(queue_counts)
                    near_counts = []
                    far_counts = []
                    near_distance = float(self.dic_traffic_env_conf.get(
                        "V10_NEAR_DISTANCE", 50.0))
                    for vehicle_ids in movement_ids:
                        near = 0
                        far = 0
                        for veh_id in vehicle_ids:
                            distance = inter.dic_vehicle_distance_current_step.get(
                                veh_id, float("inf"))
                            if distance <= near_distance:
                                near += 1
                            elif distance <= camera_distance:
                                far += 1
                        near_counts.append(near)
                        far_counts.append(far)
                    self._v10_cycle_zone_history_by_intersection.setdefault(
                        inter.inter_id, []).append({
                            "near": near_counts,
                            "far": far_counts,
                        })

            v33_sample_interval = int(self.dic_traffic_env_conf.get(
                "V33_TEMPORAL_FRAME_INTERVAL", 5))
            if v33_sample_interval > 0 and (inner_i + 1) % v33_sample_interval == 0:
                for inter in self.list_intersection:
                    snapshot_vehicle_ids = [
                        list(vehicle_ids)
                        for vehicle_ids in inter.dic_feature.get(
                            "traffic_movement_vehicle_ids", [])
                    ]
                    snapshot_ids = {
                        vehicle_id
                        for vehicle_ids in snapshot_vehicle_ids
                        for vehicle_id in vehicle_ids
                    }
                    self._v33_cycle_vehicle_snapshots_by_intersection.setdefault(
                        inter.inter_id, []).append({
                            "time_s": int(inner_i + 1),
                            "movement_vehicle_ids": snapshot_vehicle_ids,
                            "vehicle_distance": {
                                vehicle_id: inter.dic_vehicle_distance_current_step.get(vehicle_id)
                                for vehicle_id in snapshot_ids
                            },
                            "vehicle_speed": {
                                vehicle_id: inter.dic_vehicle_speed_current_step.get(vehicle_id)
                                for vehicle_id in snapshot_ids
                            },
                        })

            self._update_waiting_vehicles()

            if inner_step_callback is not None:
                try:
                    inner_step_callback(inner_i=inner_i, env=self)
                except Exception as e:
                    print(f"Warning: inner_step_callback failed at inner step {inner_i}: {e}")
                    if self.dic_traffic_env_conf.get("RAISE_INNER_STEP_CALLBACK_ERRORS", False):
                        raise

            if self.traci_conn.simulation.getMinExpectedNumber() == 0:
                print("Warning: No more vehicles expected, but continuing simulation...")
                # Implementation note.

        self.dic_traffic_env_conf["_CURRENT_V9_CYCLE_HISTORY"] = deepcopy(
            self._v9_cycle_150m_history_by_intersection)
        self.dic_traffic_env_conf["_CURRENT_V32_CYCLE_VEHICLE_SNAPSHOTS"] = deepcopy(
            self._v32_cycle_vehicle_snapshots_by_intersection)
        self.dic_traffic_env_conf["_CURRENT_V36_CYCLE_OUTBOUND_SNAPSHOTS"] = deepcopy(
            self._v36_cycle_outbound_snapshots_by_intersection)
        self.dic_traffic_env_conf["_CURRENT_V33_CYCLE_VEHICLE_SNAPSHOTS"] = deepcopy(
            self._v33_cycle_vehicle_snapshots_by_intersection)
        self.dic_traffic_env_conf["_CURRENT_V26_CYCLE_QUEUE_HISTORY"] = deepcopy(
            self._v26_cycle_queue_history_by_intersection)
        # update_current_measurements() runs before the completed cycle history
        # is published above. Refresh this feature explicitly so get_state()
        # returns the just-finished 5s...30s samples instead of a one-cycle lag.
        for inter in self.list_intersection:
            inter.dic_feature["v9_cycle_150m_history"] = deepcopy(
                self._v9_cycle_150m_history_by_intersection.get(
                    inter.inter_id, []))
            inter.dic_feature["v32_cycle_vehicle_snapshots"] = deepcopy(
                self._v32_cycle_vehicle_snapshots_by_intersection.get(
                    inter.inter_id, []))
            inter.dic_feature["v36_cycle_outbound_snapshots"] = deepcopy(
                self._v36_cycle_outbound_snapshots_by_intersection.get(
                    inter.inter_id, []))
            inter.dic_feature["v33_cycle_vehicle_snapshots"] = deepcopy(
                self._v33_cycle_vehicle_snapshots_by_intersection.get(
                    inter.inter_id, []))
            inter.dic_feature["v26_cycle_queue_history"] = deepcopy(
                self._v26_cycle_queue_history_by_intersection.get(
                    inter.inter_id, []))
        self.dic_traffic_env_conf["_CURRENT_V10_CYCLE_ZONE_HISTORY"] = deepcopy(
            self._v10_cycle_zone_history_by_intersection)
        next_state, done = self.get_state()
        reward = self.get_reward()
        if not self._simulation_running: done = True
        
        info = {"reward": reward}
        return next_state, reward, done, info

    def counterfactual_discharge_audit(self, base_action_dict=None, min_action_time=30):
        """
        Roll out each phase from the current SUMO state and measure discharge.

        This is a strict diagnostic helper: it saves the current SUMO state,
        tests each candidate action for one intersection while other
        intersections keep their real selected action, restores the original
        state, and then returns per-intersection discharge estimates.
        The real simulation trajectory is not advanced by this audit.
        """
        if not self._simulation_running:
            return {}
        if not self.dic_traffic_env_conf.get("ENABLE_COUNTERFACTUAL_DISCHARGE_LOG", False):
            return {}

        snapshot_dir = os.path.join(self.path_to_log, "counterfactual_snapshots")
        os.makedirs(snapshot_dir, exist_ok=True)
        snapshot_path = os.path.join(snapshot_dir, f"cf_{int(self.get_current_time())}.xml")

        py_state = self._capture_counterfactual_python_state()
        if self.snapshot(snapshot_path) is None:
            return {}

        audit = {
            inter.inter_id: {
                "phases": list(inter.control_phases),
                "discharged_if_served": {},
            }
            for inter in self.list_intersection
        }
        base_action_dict = dict(base_action_dict or {})

        try:
            for target_inter in self.list_intersection:
                for test_idx, phase in enumerate(target_inter.control_phases):
                    self.load_from_file(snapshot_path, quiet=True)
                    self._restore_counterfactual_python_state(py_state)
                    discharged_ids = set()
                    action_dict = dict(base_action_dict)
                    action_dict[target_inter.inter_id] = test_idx
                    self.dic_traffic_env_conf["_SUPPRESS_SIGNAL_LOG"] = True
                    self.step(
                        action_dict,
                        min_action_time=min_action_time,
                        inner_step_callback=self._build_counterfactual_discharge_callback(
                            target_inter.inter_id, phase, discharged_ids),
                    )
                    self.dic_traffic_env_conf["_SUPPRESS_SIGNAL_LOG"] = False
                    audit[target_inter.inter_id]["discharged_if_served"][phase] = len(discharged_ids)
        finally:
            self.dic_traffic_env_conf["_SUPPRESS_SIGNAL_LOG"] = False
            self.load_from_file(snapshot_path, quiet=True)
            self._restore_counterfactual_python_state(py_state)
            try:
                os.remove(snapshot_path)
            except OSError:
                pass

        return audit

    def _build_counterfactual_discharge_callback(self, target_inter_id, phase, discharged_ids):
        def _callback(inner_i, env):
            target = None
            for inter in env.list_intersection:
                if inter.inter_id == target_inter_id:
                    target = inter
                    break
            if target is None:
                return
            previous_ids = target.phase_vehicle_ids_from_lane_snapshot(
                phase, target.dic_lane_vehicle_previous_step)
            current_ids = target.phase_vehicle_ids_from_lane_snapshot(
                phase, target.dic_lane_vehicle_current_step)
            near_stopline_distance = float(target.dic_traffic_env_conf.get(
                "DISCHARGE_COUNTING_LINE_DISTANCE", 50.0))
            for veh_id in previous_ids - current_ids:
                prev_dist = target.dic_vehicle_distance_previous_step.get(veh_id)
                if prev_dist is not None and prev_dist <= near_stopline_distance:
                    discharged_ids.add(veh_id)
        return _callback

    def _capture_counterfactual_python_state(self):
        return {
            "current_time": self.current_time,
            "system_states": deepcopy(self.system_states),
            "waiting_vehicle_list": deepcopy(self.waiting_vehicle_list),
            "list_inter_log": deepcopy(self.list_inter_log),
            "intersections": [inter.capture_runtime_state() for inter in self.list_intersection],
        }

    def _restore_counterfactual_python_state(self, state):
        intersection_states = state.get("intersections", [])
        if len(intersection_states) != len(self.list_intersection):
            raise RuntimeError(
                "Checkpoint intersection state count does not match the loaded network: "
                f"{len(intersection_states)} != {len(self.list_intersection)}")
        self.current_time = state["current_time"]
        self.system_states = deepcopy(state["system_states"])
        self.waiting_vehicle_list = deepcopy(state["waiting_vehicle_list"])
        self.list_inter_log = deepcopy(state.get("list_inter_log", self.list_inter_log))
        for inter, inter_state in zip(self.list_intersection, intersection_states):
            inter.restore_runtime_state(inter_state)

    def capture_runtime_state(self):
        """Capture Python-side state that SUMO saveState does not contain."""
        state = self._capture_counterfactual_python_state()
        state["checkpoint_accumulators"] = {
            "seen_vehicle_ids": set(self._seen_vehicle_ids),
            "total_waiting_time_by_vehicle": deepcopy(self._total_waiting_time_by_vehicle),
            "depart_time_by_vehicle": deepcopy(self._depart_time_by_vehicle),
            "arrived_tt_sum": self._arrived_tt_sum,
            "arrived_count": self._arrived_count,
            "arrived_vehicle_tt": deepcopy(self._arrived_vehicle_tt),
            "all_emergency_vids": set(self._all_emergency_vids),
            "emergency_vehicle_ids": set(self._emergency_vehicle_ids),
            "emergency_depart_time": deepcopy(self._emergency_depart_time),
            "emergency_arrived_tt": deepcopy(self._emergency_arrived_tt),
            "emergency_waiting_times": deepcopy(self._emergency_waiting_times),
            "emergency_cur_waiting": deepcopy(self._emergency_cur_waiting),
            "v9_cycle_150m_history_by_intersection": deepcopy(
                self._v9_cycle_150m_history_by_intersection),
            "v32_cycle_vehicle_snapshots_by_intersection": deepcopy(
                self._v32_cycle_vehicle_snapshots_by_intersection),
            "v36_cycle_outbound_snapshots_by_intersection": deepcopy(
                self._v36_cycle_outbound_snapshots_by_intersection),
            "v33_cycle_vehicle_snapshots_by_intersection": deepcopy(
                self._v33_cycle_vehicle_snapshots_by_intersection),
            "v26_cycle_queue_history_by_intersection": deepcopy(
                self._v26_cycle_queue_history_by_intersection),
            "v10_cycle_zone_history_by_intersection": deepcopy(
                self._v10_cycle_zone_history_by_intersection),
            "intersection_checkpoint_fields": [
                {
                    "pending_green_phase": inter._pending_green_phase,
                    "time_since_last_change": inter.time_since_last_change,
                    "dic_lane_waiting_vehicle_count_previous_step": deepcopy(
                        inter.dic_lane_waiting_vehicle_count_previous_step),
                }
                for inter in self.list_intersection
            ],
        }
        return state

    def restore_runtime_state(self, state):
        """Restore Python-side state after loading the matching SUMO state."""
        self._restore_counterfactual_python_state(state)
        accumulators = state.get("checkpoint_accumulators", {})
        self._seen_vehicle_ids = set(accumulators.get("seen_vehicle_ids", ()))
        self._total_waiting_time_by_vehicle = deepcopy(
            accumulators.get("total_waiting_time_by_vehicle", {}))
        self._depart_time_by_vehicle = deepcopy(
            accumulators.get("depart_time_by_vehicle", {}))
        self._arrived_tt_sum = float(accumulators.get("arrived_tt_sum", 0.0))
        self._arrived_count = int(accumulators.get("arrived_count", 0))
        self._arrived_vehicle_tt = deepcopy(accumulators.get("arrived_vehicle_tt", {}))
        self._all_emergency_vids = set(accumulators.get("all_emergency_vids", ()))
        self._emergency_vehicle_ids = set(accumulators.get("emergency_vehicle_ids", ()))
        self._emergency_depart_time = deepcopy(accumulators.get("emergency_depart_time", {}))
        self._emergency_arrived_tt = deepcopy(accumulators.get("emergency_arrived_tt", {}))
        self._emergency_waiting_times = deepcopy(
            accumulators.get("emergency_waiting_times", {}))
        self._emergency_cur_waiting = deepcopy(accumulators.get("emergency_cur_waiting", {}))
        for key in (
            "v9_cycle_150m_history_by_intersection",
            "v32_cycle_vehicle_snapshots_by_intersection",
            "v36_cycle_outbound_snapshots_by_intersection",
            "v33_cycle_vehicle_snapshots_by_intersection",
            "v26_cycle_queue_history_by_intersection",
            "v10_cycle_zone_history_by_intersection",
        ):
            if key in accumulators:
                setattr(self, f"_{key}", deepcopy(accumulators[key]))
        intersection_fields = accumulators.get("intersection_checkpoint_fields", [])
        if len(intersection_fields) != len(self.list_intersection):
            raise RuntimeError(
                "Checkpoint intersection accumulator count does not match the loaded network: "
                f"{len(intersection_fields)} != {len(self.list_intersection)}")
        for inter, extra in zip(self.list_intersection, intersection_fields):
            inter._pending_green_phase = extra["pending_green_phase"]
            inter.time_since_last_change = extra["time_since_last_change"]
            inter.dic_lane_waiting_vehicle_count_previous_step = deepcopy(
                extra["dic_lane_waiting_vehicle_count_previous_step"])

    def capture_rollout_control_state(self):
        """Capture only Python-side signal fields not stored in SUMO state."""
        fields = (
            "current_phase_index",
            "previous_phase_index",
            "next_phase_to_set_index",
            "current_phase_duration",
            "is_in_transition",
            "_transition_stage",
            "_transition_timer",
            "_pending_green_phase",
            "_action_already_set",
            "time_since_last_change",
        )
        return {
            inter.inter_id: {
                field: getattr(inter, field, None) for field in fields
            }
            for inter in self.list_intersection
        }

    def restore_rollout_control_state(self, state):
        """Restore signal-controller fields after loading a rollout snapshot."""
        for inter in self.list_intersection:
            inter_state = state.get(inter.inter_id, {})
            for field, value in inter_state.items():
                setattr(inter, field, value)

    def capture_rollout_state_fingerprint(self):
        """Return a stable fingerprint for binding videos to rollout states."""
        speeds = self.system_states.get("get_vehicle_speed", {})
        lanes = self.system_states.get("get_vehicle_lane", {})
        distances = self.system_states.get("get_vehicle_distance", {})
        vehicle_ids = sorted(set(speeds) | set(lanes) | set(distances))
        # SUMO saveState/loadState may move a speed by a sub-micro floating
        # point amount around the 0.1 m/s queue threshold. Use the same
        # precision for both vehicle and queue fingerprints so semantically
        # identical restored states cannot disagree only at that boundary.
        normalized_speeds = {
            vehicle_id: round(float(speeds.get(vehicle_id, 0.0)), 6)
            for vehicle_id in vehicle_ids
        }
        vehicles = [
            {
                "vehicle_id": vehicle_id,
                "lane_id": lanes.get(vehicle_id),
                "distance_to_stopline_m": round(
                    float(distances.get(vehicle_id, 0.0)), 6),
                "speed_mps": normalized_speeds[vehicle_id],
            }
            for vehicle_id in vehicle_ids
        ]
        vehicle_id_lanes = [
            {
                "vehicle_id": vehicle_id,
                "lane_id": lanes.get(vehicle_id),
            }
            for vehicle_id in vehicle_ids
        ]
        movement_vehicles_150m = []
        movement_queues_150m = []
        for inter in sorted(self.list_intersection, key=lambda item: item.inter_id):
            movement_sets = inter.dic_feature.get(
                "traffic_movement_vehicle_ids_150m", [])
            normalized_sets = [
                sorted(set(ids or [])) for ids in movement_sets
            ]
            movement_vehicles_150m.append({
                "tls_id": inter.inter_id,
                "movement_vehicle_ids": normalized_sets,
            })
            movement_queues_150m.append({
                "tls_id": inter.inter_id,
                "movement_queue_vehicle_ids": [
                    sorted(
                        vehicle_id for vehicle_id in ids
                        if normalized_speeds.get(vehicle_id, 1.0) < 0.1
                    )
                    for ids in normalized_sets
                ],
            })

        control_state = self.capture_rollout_control_state()
        signals = []
        for inter in sorted(self.list_intersection, key=lambda item: item.inter_id):
            try:
                sumo_phase_index = int(
                    self.traci_conn.trafficlight.getPhase(inter.tls_id))
            except Exception:
                sumo_phase_index = None
            signals.append({
                "tls_id": inter.inter_id,
                "sumo_phase_index": sumo_phase_index,
                "control_state": control_state.get(inter.inter_id, {}),
            })

        def digest(value):
            encoded = json.dumps(
                value, sort_keys=True, separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
            return hashlib.sha256(encoded).hexdigest()

        payload = {
            "sim_time_s": round(float(self.get_current_time()), 6),
            "vehicles": vehicles,
            "signals": signals,
        }
        return {
            "sha256": digest(payload),
            "vehicles_sha256": digest(vehicles),
            "vehicle_id_lanes_sha256": digest(vehicle_id_lanes),
            "movement_vehicles_150m_sha256": digest(movement_vehicles_150m),
            "movement_queues_150m_sha256": digest(movement_queues_150m),
            "signals_sha256": digest(signals),
            "sim_time_s": payload["sim_time_s"],
            "vehicle_count": len(vehicles),
            "signal_count": len(signals),
        }

    def _update_waiting_vehicles(self):
        """Updates waiting vehicles using CoLLMLight's time/link semantics."""
        interval = self.dic_traffic_env_conf.get("INTERVAL", 1.0)
        current_vehicle_speeds = self.system_states.get("get_vehicle_speed", {})
        current_vehicle_lanes = self.system_states.get("get_vehicle_lane", {})
        self._seen_vehicle_ids.update(current_vehicle_speeds)

        for v_id in list(self.waiting_vehicle_list.keys()):
            if v_id not in current_vehicle_speeds or current_vehicle_speeds[v_id] >= 0.1:
                del self.waiting_vehicle_list[v_id]

        for v_id, speed in current_vehicle_speeds.items():
            self._total_waiting_time_by_vehicle.setdefault(v_id, 0.0)
            if speed >= 0.1:
                continue
            self._total_waiting_time_by_vehicle[v_id] += interval

            current_link = current_vehicle_lanes.get(v_id)
            wait_info = self.waiting_vehicle_list.get(v_id)
            if wait_info is None:
                self.waiting_vehicle_list[v_id] = {"time": interval, "link": current_link}
                continue

            if not isinstance(wait_info, dict):
                wait_info = {"time": float(wait_info or 0.0), "link": current_link}

            if wait_info.get("link") != current_link:
                self.waiting_vehicle_list[v_id] = {"time": interval, "link": current_link}
            else:
                self.waiting_vehicle_list[v_id] = {
                    "time": float(wait_info.get("time", 0.0)) + interval,
                    "link": current_link,
                }

    def get_all_vehicle_waiting_times(self):
        """Return cumulative stopped time for every vehicle observed in this run."""
        return {
            vehicle_id: float(
                self._total_waiting_time_by_vehicle.get(vehicle_id, 0.0)
            )
            for vehicle_id in self._seen_vehicle_ids
        }
    
    def close(self):
        """
        Closes the TraCI connection and terminates the SUMO subprocess.
        """
        if self._simulation_running and self.traci_conn:
            try:
                self.traci_conn.close()
            except traci.TraCIException: pass
            finally:
                self.traci_conn = None
                self._simulation_running = False

        if self.sumo_process:
            try:
                if self.sumo_process.poll() is None:
                    self.sumo_process.terminate()
                    self.sumo_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.sumo_process.kill()
            except Exception: pass
            finally:
                self.sumo_process = None
        
        if (
            getattr(self, '_verbose_reset', True)
            and os.environ.get("V35_SUMO_DEBUG", "0") == "1"
        ):
            print("SUMO Environment closed.")

    def __del__(self):
        """Ensures the simulation is closed when the object is garbage collected."""
        self.close()

    # ==========================================================================
    # API Methods (State-aware wrappers)
    # ==========================================================================

    def get_feature(self):
        """Returns a list of feature dictionaries, one for each intersection."""
        return [inter.get_feature() for inter in self.list_intersection]

    def get_state(self, list_state_feature=None):
        """Returns the current state observation for all intersections."""
        if list_state_feature is None:
            list_state_feature = self.dic_traffic_env_conf.get("LIST_STATE_FEATURE", [])

        if "waiting_vehicle_list" in list_state_feature:
            waiting_snapshot = dict(getattr(self, "waiting_vehicle_list", {}))
            for inter in self.list_intersection:
                inter.dic_feature["waiting_vehicle_list"] = waiting_snapshot

        list_state = [inter.get_state(list_state_feature) for inter in self.list_intersection]

        run_counts = self.dic_traffic_env_conf.get("RUN_COUNTS", 3600)
        # Implementation note.
        actual_time = self.get_current_time()
        done = actual_time >= run_counts or not self._simulation_running

        return list_state, done

    def get_reward(self):
        """Returns a list of reward values, one for each intersection."""
        reward_info = self.dic_traffic_env_conf.get("DIC_REWARD_INFO", {})
        return [inter.get_reward(reward_info) for inter in self.list_intersection]

    def get_current_time(self):
        """Returns the current simulation time."""
        return self.current_time

    def get_vehicle_count(self):
        """Gets the total number of running vehicles from SUMO."""
        if not self._simulation_running: return 0
        try: return self.traci_conn.vehicle.getIDCount()
        except traci.TraCIException: return 0

    def get_vehicles(self, include_waiting=False):
        """Gets a list of all vehicle IDs from SUMO."""
        if not self._simulation_running: return []
        try: return self.traci_conn.vehicle.getIDList()
        except traci.TraCIException: return []

    def get_lane_vehicle_count(self):
        """Gets vehicle count per lane from the last step's cached state."""
        if not self._simulation_running: return {}
        return {
            lane: len(vehicles)
            for lane, vehicles in self.system_states.get("get_lane_vehicles", {}).items()
        }

    def get_lane_waiting_vehicle_count(self):
        """Gets waiting vehicle count per lane from the last step's cached state."""
        if not self._simulation_running: return {}
        return self.system_states.get("get_lane_waiting_vehicle_count", {}).copy()

    def get_lane_vehicles(self):
        """Gets vehicle IDs per lane from the last step's cached state."""
        if not self._simulation_running: return {}
        return self.system_states.get("get_lane_vehicles", {}).copy()

    def get_vehicle_info(self, vehicle_id):
        """Gets detailed information for a specific vehicle from SUMO."""
        default_info = {"running": "false"}
        if not self._simulation_running or vehicle_id not in self.system_states.get("get_vehicle_speed", {}):
            return default_info
        try:
            return {"running": "true", "drivable": self.traci_conn.vehicle.getLaneID(vehicle_id)}
        except traci.TraCIException: return default_info

    def get_vehicle_speed(self):
        """Gets speed for all vehicles from the last step's cached state."""
        if not self._simulation_running: return {}
        return self.system_states.get("get_vehicle_speed", {}).copy()

    def get_vehicle_distance(self):
        """Gets distance from lane end for all vehicles from the last step's cached state."""
        if not self._simulation_running: return {}
        return self.system_states.get("get_vehicle_distance", {}).copy()

    def get_leader(self, vehicle_id):
        """Gets the leader of a specific vehicle from SUMO."""
        if not self._simulation_running: return ""
        try:
            leader_info = self.traci_conn.vehicle.getLeader(vehicle_id)
            return leader_info[0] if leader_info else ""
        except traci.TraCIException: return ""

    def get_average_travel_time(self):
        """Gets the average travel time of vehicles that have finished their trips (version-agnostic)."""
        # If simulation API supports it, use it
        sim = getattr(self.traci_conn, "simulation", None) if self.traci_conn else None
        if sim is not None and hasattr(sim, "getArrivedMeanTravelTime"):
            try:
                return sim.getArrivedMeanTravelTime()
            except (traci.TraCIException, AttributeError):
                # Fall through to internal aggregate
                pass

        # Fallback: use the internal aggregate (works even after simulation is closed)
        return float(self._arrived_tt_sum / self._arrived_count) if self._arrived_count > 0 else 0.0

    def get_emergency_metrics(self):
        """Gets per-vehicle detail for emergency vehicles (for JSON output).

        Note: The official AETT/AEWT metrics are calculated in vlm_oneline using
        the same methods as ATT/AWT, filtered by vehicle type. This method provides
        supplementary per-vehicle detail data.

        Returns:
            dict with keys:
              - all_emergency_vids: list of all emergency vehicle IDs ever seen
              - emergency_count: number of emergency vehicles that completed trips
              - emergency_travel_times: {vid: tt} per-vehicle depart→arrive time
              - emergency_waiting_times: {vid: wt} per-vehicle accumulated waiting time
        """
        return {
            'all_emergency_vids': sorted(self._all_emergency_vids),
            'emergency_count': len(self._emergency_arrived_tt),
            'emergency_travel_times': dict(self._emergency_arrived_tt),
            'emergency_waiting_times': dict(self._emergency_waiting_times),
        }

    def get_arrived_vehicle_travel_times(self):
        """Returns {vehicle_id: travel_time} for all vehicles that have arrived so far."""
        return dict(self._arrived_vehicle_tt)

    def set_tl_phase(self, intersection_id, phase_id):
        """Sets the traffic light phase for a specific intersection ID."""
        if not self._simulation_running: return
        try: self.traci_conn.trafficlight.setPhase(intersection_id, phase_id)
        except traci.TraCIException as e: print(f"TraCIException setting phase for {intersection_id}: {e}")

    def set_vehicle_speed(self, vehicle_id, speed):
        """Sets the speed for a specific vehicle."""
        if not self._simulation_running: return
        try: self.traci_conn.vehicle.setSpeed(vehicle_id, speed)
        except traci.TraCIException as e: print(f"TraCIException setting speed for vehicle {vehicle_id}: {e}")

    def set_vehicle_route(self, vehicle_id, route):
        """Changes the route of a specific vehicle."""
        if not self._simulation_running: return False
        try:
            self.traci_conn.vehicle.setRoute(vehicle_id, route)
            return True
        except traci.TraCIException as e:
            print(f"TraCIException setting route for vehicle {vehicle_id}: {e}")
            return False

    def set_random_seed(self, seed):
        """No-op for SUMO. The seed must be set at simulation start via reset()."""
        print("Warning: `set_random_seed` has no effect after the simulation has started. Provide a seed to `env.reset()`.")

    def snapshot(self, path=None):
        """Takes a snapshot of the current simulation state."""
        if not self._simulation_running:
            return None
        try:
            snapshot_path = path if path else os.path.join(self.path_to_log, f"snapshot_{self.current_time}.xml")
            self.traci_conn.simulation.saveState(snapshot_path)
            return snapshot_path
        except traci.TraCIException as e:
            print(f"TraCIException taking snapshot: {e}")
            return None

    def load_from_file(self, path, quiet=False, raise_on_error=False):
        """Loads a simulation state from a snapshot file."""
        if not self._simulation_running:
            print("Warning: Cannot load state. Simulation not running.")
            return
        try:
            # CMD_LOAD_SIMSTATE itself appends to TraCI's private command queue.
            # Normalize before issuing it and again before re-subscribing below.
            if not isinstance(getattr(self.traci_conn, "_queue", None), list):
                self.traci_conn._queue = []
            self.traci_conn.simulation.loadState(path)
            # Some TraCI/SUMO combinations leave the private command queue as a
            # tuple after CMD_LOAD_SIMSTATE.  The following resubscription uses
            # ``append()``, so normalize it before rebuilding subscriptions.
            if not isinstance(getattr(self.traci_conn, "_queue", None), list):
                self.traci_conn._queue = []
            self._post_load_reset(quiet=quiet)
            if not quiet:
                print(f"Simulation state loaded from file: {path}")
        except Exception as e:
            print(f"Error loading snapshot from file {path}: {e}")
            if raise_on_error:
                traceback.print_exc()
            if raise_on_error:
                raise RuntimeError(
                    f"Failed to restore SUMO snapshot {path}") from e
            
    def _post_load_reset(self, quiet=False):
        """Resets internal state after loading a snapshot."""
        if not quiet:
            print("Resetting internal state after loading snapshot...")
        self.current_time = self.traci_conn.simulation.getTime()
        self._subscribe_lanes()
        # Subscription result caches can retain vehicles and lane membership
        # from the worker state that existed before loadState(). Rebuild this
        # first restored frame directly from SUMO's current vehicle domain.
        self._update_system_states(force_direct_vehicle_queries=True)
        for inter in self.list_intersection:
            inter.update_current_measurements(self.system_states)
        self._update_waiting_vehicles()

    def _subscribe_lanes(self):
        """Re-subscribes lane variables, which may be cleared by SUMO loadState."""
        if not self._simulation_running or self.traci_conn is None:
            return
        for lane_id in self.lane_length.keys():
            try:
                self.traci_conn.lane.subscribe(lane_id, [
                    traci.constants.LAST_STEP_VEHICLE_HALTING_NUMBER,
                    traci.constants.LAST_STEP_VEHICLE_ID_LIST
                ])
            except traci.TraCIException:
                continue

    def log(self, cur_time, before_action_feature, action):
        """Logs the state and action for each intersection."""
        for inter_ind in range(len(self.list_intersection)):
            self.list_inter_log[inter_ind].append({
                "time": cur_time,
                "state": before_action_feature[inter_ind],
                "action": action[inter_ind]
            })

    def batch_log(self, start=0, stop=None):
        """Logs vehicle times and intersection state-action logs."""
        if stop is None: stop = len(self.list_intersection)
        
        # Implementation note.
        vehicle_log_dir = os.path.join(self.path_to_log, "vehicle_logs")
        os.makedirs(vehicle_log_dir, exist_ok=True)
        
        for inter_ind in range(start, stop):
            inter = self.list_intersection[inter_ind]
            # Implementation note.
            path_to_vehicle_log = os.path.join(vehicle_log_dir, f"vehicle_{inter.inter_name}.csv")
            dic_vehicle = inter.get_dic_vehicle_arrive_leave_time()
            if dic_vehicle:
                pd.DataFrame.from_dict(dic_vehicle, orient="index").to_csv(path_to_vehicle_log, na_rep="nan", index_label="vehicle_id")
            if self.dic_traffic_env_conf.get("SAVE_INTERSECTION_PKL", False):
                path_to_state_log = os.path.join(
                    self.path_to_log, f"inter_{inter_ind}.pkl")
                with open(path_to_state_log, "wb") as f:
                    pickle.dump(self.list_inter_log[inter_ind], f)

        # Save emergency vehicle metrics (AETT / AEWT)
        emergency_metrics = self.get_emergency_metrics()
        if emergency_metrics['emergency_count'] > 0 or self._emergency_arrived_tt:
            em_path = os.path.join(self.path_to_log, "emergency_metrics.json")
            try:
                with open(em_path, 'w', encoding='utf-8') as f:
                    json.dump(emergency_metrics, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

    @staticmethod
    def end_engine():
        """Placeholder method indicating simulation end."""
        print("================ SUMO Process End ================")

    def __deepcopy__(self, memo):
        """
        Custom deepcopy implementation for SUMOEnv.
        This method creates a copy of the Python-side environment state,
        while nullifying attributes that cannot or should not be copied,
        such as the live TraCI connection and the SUMO process handle.
        The static network object (`sumo_net`) is shared by reference.
        """
        if id(self) in memo:
            return memo[id(self)]

        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        for k, v in self.__dict__.items():
            # Skip non-serializable or shared attributes
            if k in ['sumo_process', 'traci_conn', 'sumo_net']:
                continue
            # Perform a deepcopy on all other attributes
            setattr(result, k, deepcopy(v, memo))

        # Manually handle the skipped attributes for the new copy
        result.sumo_process = None
        result.traci_conn = None
        result.sumo_net = self.sumo_net  # Share the reference to the static network data

        return result

    # Implementation note.
    
    def _get_tls_render_center(self, tls_id):
        return getattr(self, "_tls_render_centers", {}).get(tls_id)

    def _update_tls_render_centers(self, tls_info):
        centers = getattr(self, "_tls_render_centers", {}).copy()
        for tls_id, info in tls_info.items():
            points = []
            for stop_line_points in info.get("in_road_stop_line", {}).values():
                points.extend(stop_line_points or [])
            if not points:
                continue
            centers[tls_id] = (
                sum(point[0] for point in points) / len(points),
                sum(point[1] for point in points) / len(points),
            )
        self._tls_render_centers = centers

    def _vehicle_within_tls_radius(self, position, tls_ids, radius):
        if not tls_ids or radius is None:
            return True
        radius_sq = float(radius) * float(radius)
        for tls_id in tls_ids:
            center = self._get_tls_render_center(tls_id)
            if center is None:
                continue
            dx = float(position[0]) - float(center[0])
            dy = float(position[1]) - float(center[1])
            if dx * dx + dy * dy <= radius_sq:
                return True
        return False

    def get_tshub_obs(self, tls_ids=None, radius=None):
        """
        获取当前仿真步骤的车辆信息，用于 TSHubRenderer.step()。
        返回格式与 TshubEnvironment 的 VehicleBuilder.get_objects_infos() 兼容。
        
        Returns:
            dict: {'vehicle': {veh_id: {vehicle_type, position, heading, length}, ...}}
        """
        if isinstance(tls_ids, str):
            tls_ids = [tls_ids]
        vehicle_obs = {}
        for veh_id in self.traci_conn.vehicle.getIDList():
            # Implementation note.
            pos = self.traci_conn.vehicle.getPosition(veh_id)  # Implementation note.
            if not self._vehicle_within_tls_radius(pos, tls_ids, radius):
                continue
            heading = self.traci_conn.vehicle.getAngle(veh_id)  # Implementation note.
            veh_type = self.traci_conn.vehicle.getTypeID(veh_id)  # Implementation note.
            length = self.traci_conn.vehicle.getLength(veh_id)  # Implementation note.
            vehicle_obs[veh_id] = {
                'vehicle_type': veh_type,
                'position': pos,
                'heading': heading,
                'length': length
            }
        return {'vehicle': vehicle_obs}
    
    def get_tls_init_info(self, tls_ids=None):
        """
        获取路口初始化信息，用于 TSHubRenderer.reset()。
        包含每个路口的进入道路朝向和停止线位置。
        
        Args:
            tls_ids: 需要获取信息的路口 ID 列表，None 表示所有路口
            
        Returns:
            dict: {'tls': {tls_id: {in_roads_heading, in_road_stop_line}, ...}}
        """
        if tls_ids is None:
            tls_ids = list(self.intersections_data.keys())
        
        tls_info = {}
        for tls_id in tls_ids:
            if tls_id not in self.intersections_data:
                continue
            
            # Implementation note.
            in_roads = []
            in_roads_heading = {}
            in_road_stop_line = {}
            
            # Implementation note.
            try:
                # Implementation note.
                # Implementation note.
                controlled_links = self.traci_conn.trafficlight.getControlledLinks(tls_id)
                in_out_lanes = [(link[0][0], link[0][1]) for link in controlled_links if link]
                in_lanes = [lanes[0] for lanes in in_out_lanes]  # Implementation note.
                
                # Implementation note.
                road_ids_set = set()
                for lane_id in in_lanes:
                    road_id = self.traci_conn.lane.getEdgeID(lane_id)
                    road_ids_set.add(road_id)
                in_roads = sorted(list(road_ids_set))  # Implementation note.
                
                # Implementation note.
                in_roads_heading = {road_id: self.traci_conn.edge.getAngle(road_id) for road_id in in_roads}
                
                # Implementation note.
                in_road_stop_line = {road_id: [] for road_id in in_roads}
                for lane_id in in_lanes:
                    road_name = self.traci_conn.lane.getEdgeID(lane_id)
                    lane_end_position = self.traci_conn.lane.getShape(lane_id)[-1]  # Implementation note.
                    in_road_stop_line[road_name].append(lane_end_position)
                
                tls_info[tls_id] = {
                    'in_roads_heading': in_roads_heading,
                    'in_road_stop_line': in_road_stop_line
                }
            except Exception as e:
                print(f"Warning: Failed to get TLS info for {tls_id}: {e}")
                continue
        self._update_tls_render_centers(tls_info)
        return {'tls': tls_info}
