import xml.etree.ElementTree as ET
import numpy as np
import math
from collections import defaultdict
import json
import os
import argparse

# --- Constants adapted from the CityFlow script ---
# (Your existing constants remain unchanged)
# ... (all your existing constants are here) ...

location_dict = {"North": "N", "South": "S", "East": "E", "West": "W"}
location_dict_reverse = {"N": "North", "S": "South", "E": "East", "W": "West"}

SUMO_DIR_TO_SYMBOLIC_TURN = {
    's': 'T', 't': 'L', 'l': 'L', 'L': 'L', 'r': 'R', 'R': 'R'
}
SYMBOLIC_MOVEMENT_TO_CITYFLOW_TYPE = {
    "T": "go_straight", "L": "turn_left", "R": "turn_right"
}

# Angles represent the direction of travel (Heading) towards the intersection
angles_cityflow = [0, math.pi / 2, math.pi, 3 * math.pi / 2, 2 * math.pi]
# Orients map these Headings to their Origin (Standard Convention)
orients = ['W', 'S', 'E', 'N', 'W', 'S', 'E', 'N']
# -----------------------------------------------------------------------------------
DEFAULT_YELLOW_TIME_CONFIG = 5
DEFAULT_GREEN_TIME_CONFIG = 30
TARGET_PHASE_SYMBOLS_TEMPLATE = [
    {"ET", "WT"},  # ETWT - East & West Through (standard 4-phase)
    {"NT", "ST"},  # NTST - North & South Through
    {"EL", "WL"},  # ELWL - East & West Left
    {"NL", "SL"},  # NLSL - North & South Left
    # Removed single-direction phases (ELET, WLWT, NLNT, SLST) for standard 4-phase control
]


# --- Helper functions ---
def parse_sumo_net_xml(file_path):
    """Parses a SUMO .net.xml file."""
    try:
        tree = ET.parse(file_path)
        return tree.getroot()
    except ET.ParseError as e:
        print(f"Error parsing XML file {file_path}: {e}")
        return None
    except FileNotFoundError:
        print(f"Error: File not found {file_path}")
        return None

def write_sumo_net_xml(root_element, file_path):
    """Writes the XML tree to a .net.xml file."""
    tree = ET.ElementTree(root_element)
    # ET.indent(tree, space="  ") # For pretty printing, requires Python 3.9+
    tree.write(file_path, encoding="utf-8", xml_declaration=True)


# ==============================================================================
# --- NEW FUNCTION TO FIX THE ERROR ---
# ==============================================================================
def remove_unwanted_vclasses(root_element):
    """
    Finds all <lane> elements and removes specific, unwanted vehicle class
    permissions from their 'allow' and 'disallow' attributes. This prevents
    SUMO errors for undefined vehicle classes like 'subway', 'tram', etc.
    """
    # Define the set of vehicle classes you want to remove.
    # You can easily add others here, e.g., 'tram', 'rail', 'bus'
    VCLASSES_TO_REMOVE = {'cable_car', 'subway', 'tram', 'rail_urban', 'rail', 'rail_electric', 'rail_fast', 'ship'}
    
    print("\nCleaning lane permissions...")
    lanes_modified_count = 0
    
    # Use .//lane to find all lane elements anywhere in the tree
    for lane in root_element.findall(".//lane"):
        modified = False
        
        # --- Clean the 'allow' attribute ---
        allowed_vclasses_str = lane.get('allow')
        if allowed_vclasses_str:
            # Split the string into a list of classes
            original_vclasses = allowed_vclasses_str.split()
            # Filter the list, keeping only the ones NOT in our removal set
            filtered_vclasses = [vc for vc in original_vclasses if vc not in VCLASSES_TO_REMOVE]
            
            # If the list has changed, update the attribute
            if len(filtered_vclasses) != len(original_vclasses):
                modified = True
                if filtered_vclasses:
                    # Join the cleaned list back into a string and set it
                    lane.set('allow', ' '.join(filtered_vclasses))
                else:
                    # If no allowed classes are left, remove the attribute entirely
                    del lane.attrib['allow']

        # --- Clean the 'disallow' attribute (less common but good practice) ---
        disallowed_vclasses_str = lane.get('disallow')
        if disallowed_vclasses_str:
            original_vclasses = disallowed_vclasses_str.split()
            filtered_vclasses = [vc for vc in original_vclasses if vc not in VCLASSES_TO_REMOVE]
            
            if len(filtered_vclasses) != len(original_vclasses):
                modified = True
                if filtered_vclasses:
                    lane.set('disallow', ' '.join(filtered_vclasses))
                else:
                    del lane.attrib['disallow']
                    
        if modified:
            lanes_modified_count += 1
            
    if lanes_modified_count > 0:
        print(f"Cleaned unwanted vehicle class permissions from {lanes_modified_count} lanes.")
    else:
        print("No lanes with unwanted vehicle class permissions were found.")
# ==============================================================================
# --- END OF NEW FUNCTION ---
# ==============================================================================

# ==============================================================================
# --- NEW FUNCTION TO SET GLOBAL SPEED ---
# ==============================================================================
def set_global_speed(root_element, new_speed):
    """
    Sets the 'speed' attribute for all <lane> and <type> elements in the
    .net.xml file to a specified value.

    Args:
        root_element: The root element of the parsed XML tree.
        new_speed (float): The new speed value in m/s.
    """
    print(f"\nSetting global speed to {new_speed} m/s...")
    speed_str = str(new_speed)
    
    # --- Update all <lane> elements ---
    lanes_updated = 0
    for lane in root_element.findall(".//lane"):
        lane.set("speed", speed_str)
        lanes_updated += 1
        
    # --- Update all <type> elements (vehicle types) ---
    types_updated = 0
    for vtype in root_element.findall(".//type"):
        vtype.set("speed", speed_str)
        types_updated += 1

    print(f"Updated speed for {lanes_updated} lanes and {types_updated} vehicle types.")
# ==============================================================================
# --- END OF NEW FUNCTION ---
# ==============================================================================


class SumoRoadDirectionFinder:
    # ... (Your existing class remains unchanged) ...
    """
    Adapts logic from CityFlow's RoadDirectionFinder for SUMO networks.
    Determines cardinal orientations for incoming edges/lanes at a junction.
    """
    def __init__(self, junction_element, edges_map, lanes_map):
        self.junction_id = junction_element.get("id")
        self.junction_pos = (float(junction_element.get("x")), float(junction_element.get("y")))
        self.edges_map = edges_map # {edge_id: edge_element}
        self.lanes_map = lanes_map # {lane_id: lane_element}
        
        self.incoming_approaches_info = [] # List of dicts for each approach
        self.approach_to_orient = {} # approach_id (e.g. edge_id) -> 'N'/'S'/'E'/'W'

        self._parse_incoming_approaches(junction_element)
        if self.incoming_approaches_info:
            self._determine_orientations()

    def _parse_incoming_approaches(self, junction_element):
        """
        Identifies incoming edges/lanes that form approaches to the junction.
        The orientation is determined by the geometry of its lanes near the junction,
        calculated consistently with env_auto.py's method.
        """
        inc_lanes_str = junction_element.get("incLanes", "")
        if not inc_lanes_str:
            return

        incoming_lane_ids = inc_lanes_str.split()
        
        edge_to_lanes_map = defaultdict(list)
        for lane_id in incoming_lane_ids:
            if lane_id in self.lanes_map:
                # Extract edge_id by removing the last "_index" suffix from lane_id
                # e.g., "road_1_1_0_0" -> "road_1_1_0", "601909765#1_0" -> "601909765#1"
                edge_id = "_".join(lane_id.rsplit("_", 1)[:-1]) if "_" in lane_id else lane_id
                edge_to_lanes_map[edge_id].append(self.lanes_map[lane_id])
        
        center_x, center_y = self.junction_pos[0], self.junction_pos[1]

        for edge_id, lanes_on_edge in edge_to_lanes_map.items():
            if not lanes_on_edge:
                continue

            # Use the shape of the first lane listed for that edge.
            # Its second-to-last point will serve as the 'start_point' from env_auto.py
            lane_shape_str = lanes_on_edge[0].get("shape")
            if not lane_shape_str:
                # print(f"Warning (Junction {self.junction_id}): Lane {lanes_on_edge[0].get('id')} has no shape.")
                continue
            
            shape_points_str = lane_shape_str.split()
            if len(shape_points_str) < 2:
                # print(f"Warning (Junction {self.junction_id}): Lane {lanes_on_edge[0].get('id')} shape has < 2 points.")
                continue

            # p_prev_coords is the second-to-last point of the incoming lane's shape.
            # This is analogous to 'start_point' in env_auto.py's logic.
            p_prev_coords_list = list(map(float, shape_points_str[-2].split(',')))
            start_point_x, start_point_y = p_prev_coords_list[0], p_prev_coords_list[1]

            # Calculate dx and dy for the vector from start_point (on road/lane) to center (intersection).
            # This directly matches the dx, dy calculation in env_auto.py:
            # dx_env_auto = center_x - start_point_on_road_x
            # dy_env_auto = center_y - start_point_on_road_y
            dx = center_x - start_point_x
            dy = center_y - start_point_y
            
            # --- CORRECTION START: Align calculation with sumo_env.py ---
            # Calculate the Heading angle.
            angle_rad = math.atan2(dy, dx)
            if angle_rad < 0:
                angle_rad += 2 * math.pi
            # --- CORRECTION END ---
            
            # The result of (atan2(dy, dx) + math.pi) will be in [0, 2*pi].
            # This range is directly comparable with CityFlow's 'angles' array
            # [0, math.pi/2, math.pi, 3*math.pi/2, 2*pi].
            # No further modulo is strictly needed here if the comparison logic handles 2*pi correctly.

            # Find closest cardinal direction based on this angle
            # (E=0, N=pi/2, W=pi, S=3pi/2, E=2*pi for comparison wrap-around)
            # Using CityFlow's 'angles' for direct comparison.
            # Note: angles_cityflow is [0, math.pi / 2, math.pi, math.pi + math.pi / 2, math.pi + math.pi]
            # which simplifies to [0, pi/2, pi, 3pi/2, 2pi]
            
            # Calculate absolute differences to each cardinal angle
            # The orients list is ['E', 'N', 'W', 'S', 'E', ...]
            # The angles_cityflow list is [0, pi/2, pi, 3pi/2, 2pi]
            orient_angle_diffs = [abs(angle_rad - a) for a in angles_cityflow[:4]] # Compare with E, N, W, S
            
            # Find closest cardinal direction using the standardized definitions
            angle_diffs_for_argmin = np.abs(np.subtract(angles_cityflow, angle_rad))
            orient_idx_guess_raw = np.argmin(angle_diffs_for_argmin)
            # This now correctly maps Heading to Origin using the updated 'orients' list
            initial_orient_guess = orients[orient_idx_guess_raw % 4]

            orient_angle_diff = angle_diffs_for_argmin[orient_idx_guess_raw]

            self.incoming_approaches_info.append({
                'id': edge_id, 
                'angle': angle_rad,
                'initial_orient_guess': initial_orient_guess,
                'angle_diff': orient_angle_diff,
                'orient': None 
            })

    def _get_opposite_road(self, cur_road_angle_info, roads_with_angle, orients_taken_map):
        # This function is from the original CityFlow script.
        # It finds if there's an already oriented road that is opposite to the current one.
        for orient, road_idx_in_sorted_list in orients_taken_map.items():
            road_info_taken = roads_with_angle[road_idx_in_sorted_list]
            if cur_road_angle_info['id'] != road_info_taken['id'] and \
               abs(abs(cur_road_angle_info['angle'] - road_info_taken['angle']) - math.pi) < math.pi / 8: # Tolerance
                return road_idx_in_sorted_list
        return -1

    def _decide_road_orient_recursive(self, roads_with_angle_list, current_processing_idx, assigned_orientations_map, last_assigned_orient_info=None):
        # This function is from the original CityFlow script.
        if current_processing_idx == len(roads_with_angle_list):
            return True 

        current_road_info = roads_with_angle_list[current_processing_idx]
        possible_local_orients = []
        start_orient_idx = 0
        if last_assigned_orient_info:
            try:
                start_orient_idx = (orients.index(last_assigned_orient_info['orient']) + 1) % 4
            except ValueError: pass

        for i in range(4):
            candidate_orient = orients[(start_orient_idx + i) % 4]
            if candidate_orient not in assigned_orientations_map:
                if last_assigned_orient_info and candidate_orient == last_assigned_orient_info['orient']:
                    if len(roads_with_angle_list) == 2: pass
                    else: continue
                possible_local_orients.append(candidate_orient)
        
        if not possible_local_orients and len(roads_with_angle_list) > 4:
             possible_local_orients = [current_road_info['initial_orient_guess']]

        preferred_orient = current_road_info['initial_orient_guess']
        opposite_road_sorted_idx = self._get_opposite_road(current_road_info, roads_with_angle_list, assigned_orientations_map)

        if opposite_road_sorted_idx != -1:
            opposite_road_assigned_orient = roads_with_angle_list[opposite_road_sorted_idx]['orient']
            try:
                preferred_orient_idx = (orients.index(opposite_road_assigned_orient) + 2) % 4
                preferred_orient = orients[preferred_orient_idx]
            except ValueError: pass

        if preferred_orient in possible_local_orients:
            possible_local_orients.pop(possible_local_orients.index(preferred_orient))
            possible_local_orients.insert(0, preferred_orient)
        elif not possible_local_orients and preferred_orient not in assigned_orientations_map:
            possible_local_orients.insert(0, preferred_orient)

        for trial_orient in possible_local_orients:
            current_road_info['orient'] = trial_orient
            newly_assigned_orientations_map = assigned_orientations_map.copy()
            newly_assigned_orientations_map[trial_orient] = current_processing_idx
            if self._decide_road_orient_recursive(roads_with_angle_list, current_processing_idx + 1, newly_assigned_orientations_map, current_road_info):
                return True
        
        current_road_info['orient'] = None
        return False

    def _determine_orientations(self):
        """Assigns cardinal directions (N,S,E,W) to incoming approaches."""
        if not self.incoming_approaches_info:
            return

        # Sort approaches by angle (similar to CityFlow script)
        self.incoming_approaches_info.sort(key=lambda x: x['angle'])
        
        # Attempt recursive assignment
        if not self._decide_road_orient_recursive(self.incoming_approaches_info, 0, {}):
            # print(f"Warning (Junction {self.junction_id}): Could not assign unique cardinal directions using heuristic.")
            # Fallback: Assign based purely on initial geometric guess
            for r_info in self.incoming_approaches_info:
                r_info['orient'] = r_info['initial_orient_guess']
                # print(f"  Fallback assigning approach {r_info['id']} as {r_info['orient']}")

        self.approach_to_orient = {r['id']: r['orient'] for r in self.incoming_approaches_info if r['orient'] is not None}

    def get_approach_orientations(self):
        """Returns {approach_id (edge_id): 'N'/'S'/'E'/'W'}."""
        return self.approach_to_orient



def modify_sumo_roadnet_phases(sumo_net_file, output_net_file, output_mapping_file, speed=None):
    """
    Modifies traffic light phases and optionally sets a global speed in a SUMO .net.xml file.
    
    Args:
        sumo_net_file (str): Path to the input .net.xml file.
        output_net_file (str): Path to save the modified .net.xml file.
        output_mapping_file (str): Path to save the intersection-phase mapping JSON.
        speed (float, optional): If provided, sets a new global speed for all lanes and types.
    """
    root = parse_sumo_net_xml(sumo_net_file)
    if root is None:
        return

    # --- NEW: Call the cleanup function right after parsing ---
    remove_unwanted_vclasses(root)
    
    # NEW: If a speed value is provided, call the function to update the XML tree
    if speed is not None:
        set_global_speed(root, speed)

    # --- Pre-collect all necessary elements for efficiency ---
    edges_map = {edge.get("id"): edge for edge in root.findall("edge")}
    lanes_map = {}
    for edge in edges_map.values():
        for lane in edge.findall("lane"):
            lanes_map[lane.get("id")] = lane
    
    junctions_map = {junc.get("id"): junc for junc in root.findall("junction")}
    
    connections_by_tl_id = defaultdict(list)
    max_link_indices = defaultdict(lambda: -1)

    for conn in root.findall("connection"):
        tl_id = conn.get("tl")
        if tl_id:
            connections_by_tl_id[tl_id].append(conn)
            try:
                link_idx = int(conn.get("linkIndex"))
                if link_idx > max_link_indices[tl_id]:
                    max_link_indices[tl_id] = link_idx
            except (ValueError, TypeError):
                pass

    tl_logic_elements = root.findall("tlLogic")
    intersection_to_new_symbolic_phases_map = {}

    print("\nProcessing Traffic Light Logic...")
    for tl_logic in tl_logic_elements:
        tl_id = tl_logic.get("id")
        
        if not connections_by_tl_id[tl_id]:
            continue

        num_controlled_signals = max_link_indices[tl_id] + 1
        if num_controlled_signals <= 0:
            continue

        junction_element = junctions_map.get(tl_id)
        if not junction_element:
            continue
        
        direction_finder = SumoRoadDirectionFinder(junction_element, edges_map, lanes_map)
        approach_orientations = direction_finder.get_approach_orientations()
        
        movements_to_link_indices = defaultdict(list)
        all_right_turn_link_indices = []

        for conn in connections_by_tl_id[tl_id]:
            from_edge_id = conn.get("from")
            approach_orient = approach_orientations.get(from_edge_id)
            sumo_dir = conn.get("dir")
            symbolic_turn = SUMO_DIR_TO_SYMBOLIC_TURN.get(sumo_dir)
            
            link_index_str = conn.get("linkIndex")
            if link_index_str is None: continue
            link_index = int(link_index_str)

            if approach_orient and symbolic_turn:
                movement_symbol = approach_orient + symbolic_turn
                movements_to_link_indices[movement_symbol].append(link_index)
                if symbolic_turn == "R":
                    all_right_turn_link_indices.append(link_index)
        
        all_right_turn_link_indices = sorted(list(set(all_right_turn_link_indices)))

        # ======================================================================
        # --- START: SAFER MODIFICATION LOGIC ---
        # ======================================================================
        
        # 1. Generate new phases in a temporary list first, without touching the XML tree.
        new_phases_to_add = []
        actual_symbolic_phases_strings_for_tl = []

        # Generate Yellow/All-Red Phase (fully red, including right turns)
        yellow_state_list = ['r'] * num_controlled_signals
        
        yellow_phase_attrs = {
            "duration": str(DEFAULT_YELLOW_TIME_CONFIG),
            "state": "".join(yellow_state_list),
            "name": "YELLOW_ALL_RED"
        }
        new_phases_to_add.append(yellow_phase_attrs)

        # Generate Green Phases
        for target_phase_group in TARGET_PHASE_SYMBOLS_TEMPLATE:
            current_green_phase_link_indices = []
            actual_movements_in_this_phase = []

            for individual_movement_symbol in target_phase_group:
                if individual_movement_symbol in movements_to_link_indices:
                    current_green_phase_link_indices.extend(movements_to_link_indices[individual_movement_symbol])
                    actual_movements_in_this_phase.append(individual_movement_symbol)
            
            if actual_movements_in_this_phase:
                combined_link_indices = list(set(current_green_phase_link_indices + all_right_turn_link_indices))
                
                green_state_list = ['r'] * num_controlled_signals
                for idx in combined_link_indices:
                    if 0 <= idx < num_controlled_signals:
                        if idx in current_green_phase_link_indices:
                            green_state_list[idx] = 'G'
                        elif idx in all_right_turn_link_indices:
                             green_state_list[idx] = 'g'

                actual_symbolic_phase_str = "".join(sorted(list(actual_movements_in_this_phase)))
                green_phase_attrs = {
                    "duration": str(DEFAULT_GREEN_TIME_CONFIG),
                    "state": "".join(green_state_list),
                    "name": actual_symbolic_phase_str
                }
                new_phases_to_add.append(green_phase_attrs)
                actual_symbolic_phases_strings_for_tl.append(actual_symbolic_phase_str)
        
        # 2. Validate: Check if we generated any green phases.
        #    The length of `new_phases_to_add` will be > 1 if green phases were added.
        if len(new_phases_to_add) > 1:
            # 3. Commit: If valid, clear the old phases and add the new ones.
            # print(f"Successfully generated {len(actual_symbolic_phases_strings_for_tl)} green phases for TL '{tl_id}'. Applying changes.")
            
            # Remove old phases
            existing_phases = tl_logic.findall("phase")
            for p in existing_phases:
                tl_logic.remove(p)
            
            # Add all the new phases we generated
            for phase_attrs in new_phases_to_add:
                ET.SubElement(tl_logic, "phase", attrib=phase_attrs)
            
            intersection_to_new_symbolic_phases_map[tl_id] = actual_symbolic_phases_strings_for_tl
        else:
            # 4. Abort: If not valid, do nothing and warn the user.
            print(f"WARNING: Could not generate any valid green phases for TL '{tl_id}'. "
                  f"Skipping modification for this intersection. Its original phasing is preserved.")

        # ======================================================================
        # --- END: SAFER MODIFICATION LOGIC ---
        # ======================================================================
    
    print(f"\nSaving modified SUMO roadnet to: {output_net_file}")
    write_sumo_net_xml(root, output_net_file)

    print(f"Saving intersection-phase mapping to: {output_mapping_file}")
    with open(output_mapping_file, "w") as f:
        json.dump(intersection_to_new_symbolic_phases_map, f, indent=2)

    print("\nSUMO processing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Modify traffic light phases in a SUMO .net.xml file and optionally set a global speed.')
    parser.add_argument('--input', type=str, required=True,
                        help='Input SUMO .net.xml file path')
    parser.add_argument('--output', type=str, required=True,
                        help='Output modified SUMO .net.xml file path')
    parser.add_argument('--mapping', type=str, required=True,
                        help='Output phase mapping JSON file path')
    # NEW ARGUMENT
    parser.add_argument('--speed', type=float, required=False,
                        help='Optional: Global speed in m/s to set for all lanes and vehicle types.')
    
    args = parser.parse_args()
    
    # Pass the new argument to the main function
    modify_sumo_roadnet_phases(args.input, args.output, args.mapping, speed=args.speed)