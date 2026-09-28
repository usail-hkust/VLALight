import numpy as np
from typing import Dict, Any


def get_lane_traffic_conditions(lane_id: str, env) -> Dict[str, Any]:
    """
    Retrieve detailed traffic conditions for a specific lane.

    Args:
        lane_id: The identifier of the SUMO lane (e.g., "edgeID_0")
        env: An instance of SUMOEnv

    Returns:
        Dictionary containing comprehensive lane traffic information:
        - vehicle_count: Total number of vehicles in the lane
        - queue_length: Number of waiting vehicles (speed <= 0.1 m/s)
        - moving_vehicles: Number of moving vehicles (speed > 0.1 m/s)
        - average_speed: Average speed of all vehicles in the lane
        - average_waiting_time: Average waiting time of queued vehicles
        - cell_occupancy: Vehicle distribution across 4 lane cells (0-3, 0 is nearest to intersection)
        - vehicle_details: List of detailed information for each vehicle
        - lane_density: Vehicles per unit length (approx. using 8 m per vehicle)
        - throughput_potential: Estimated throughput based on moving vehicles ratio
        - queue_density: Waiting vehicles per unit length (approx. using 8 m per vehicle)
        - lane_length: Lane length in meters
    """
    try:
        # --- Basic lane data from SUMOEnv caches ---
        lane_vehicles = env.get_lane_vehicles().get(lane_id, [])
        lane_waiting_count = env.get_lane_waiting_vehicle_count().get(lane_id, 0)

        # Lane length from SUMOEnv cache
        lane_length = _get_lane_length(lane_id, env)

        # Initialize result structure
        traffic_conditions = {
            'lane_id': lane_id,
            'vehicle_count': len(lane_vehicles),
            'queue_length': lane_waiting_count,
            'queue_density': 0.0,
            'moving_vehicles': 0,
            'average_speed': env.get_lane_speed(lane_id),
            'average_waiting_time': 0.0,
            'cell_occupancy': [0, 0, 0, 0],  # cells: 0 (near intersection) -> 3 (far)
            'vehicle_details': [],
            'lane_density': 0.0,
            'throughput_potential': 0.0,
            'lane_length': lane_length,
        }

        # Early return if empty
        if not lane_vehicles:
            return traffic_conditions

        # Cached per-vehicle kinematics from SUMOEnv
        speed_map = env.get_vehicle_speed()       # {veh_id: speed (m/s)}
        pos_map = env.get_vehicle_distance()      # {veh_id: lanePosition (m from lane START)}

        # Collect detailed vehicle information
        vehicle_speeds = []
        waiting_times = []

        for vehicle_id in lane_vehicles:
            vehicle_info = _get_vehicle_traffic_info(
                vehicle_id=vehicle_id,
                lane_length=lane_length,
                speed_map=speed_map,
                pos_map=pos_map,
                env=env
            )
            traffic_conditions['vehicle_details'].append(vehicle_info)

            # Update cell occupancy (0..3)
            cell_index = vehicle_info['cell_position']
            if 0 <= cell_index <= 3:
                traffic_conditions['cell_occupancy'][cell_index] += 1

            # Accumulate stats
            vehicle_speeds.append(vehicle_info['speed'])
            if vehicle_info['is_moving']:
                traffic_conditions['moving_vehicles'] += 1
            else:
                waiting_times.append(vehicle_info['waiting_time'])

        # Aggregate statistics
        traffic_conditions['average_speed'] = float(np.mean(vehicle_speeds)) if vehicle_speeds else 0.0
        traffic_conditions['average_waiting_time'] = float(np.mean(waiting_times)) if waiting_times else 0.0
        traffic_conditions['lane_density'] = (len(lane_vehicles) * 8.0 / lane_length) if lane_length > 0 else 0.0
        traffic_conditions['queue_density'] = (lane_waiting_count * 8.0 / lane_length) if lane_length > 0 else 0.0
        traffic_conditions['lane_length'] = lane_length

        # Throughput potential: proportion of moving vehicles
        if traffic_conditions['vehicle_count'] > 0:
            traffic_conditions['throughput_potential'] = (
                traffic_conditions['moving_vehicles'] / traffic_conditions['vehicle_count']
            )

        return traffic_conditions

    except Exception as e:
        # Robust error return
        return {
            'lane_id': lane_id,
            'error': f"Failed to get traffic conditions: {str(e)}",
            'vehicle_count': 0,
            'queue_length': 0,
            'queue_density': 0.0,
            'moving_vehicles': 0,
            'average_speed': 0.0,
            'average_waiting_time': 0.0,
            'cell_occupancy': [0, 0, 0, 0],
            'vehicle_details': [],
            'lane_density': 0.0,
            'throughput_potential': 0.0,
            'lane_length': 0.0,
        }


def _get_vehicle_traffic_info(
    vehicle_id: str,
    lane_length: float,
    speed_map: Dict[str, float],
    pos_map: Dict[str, float],
    env
) -> Dict[str, Any]:
    """
    Detailed traffic info for a single vehicle (SUMOEnv only).

    Args:
        vehicle_id: SUMO vehicle ID
        lane_length: Length of the lane the vehicle is in (meters)
        speed_map: Cached speeds from SUMOEnv.get_vehicle_speed()
        pos_map: Cached lane positions from SUMOEnv.get_vehicle_distance() (distance from lane START)
        env: SUMOEnv instance (used for waiting time cache)

    Returns:
        Dict with:
        - speed (m/s)
        - position_from_start: distance to intersection (lane END), so smaller = nearer to intersection
        - position_from_end: distance from lane START (SUMO lanePosition)
        - cell_position: 0..3 (0 is nearest to intersection)
        - is_moving: speed > 0.1 m/s
        - waiting_time: accumulated waiting time from env.waiting_vehicle_list
        - relative_position: position_from_start / lane_length (0..1)
    """
    try:
        # SUMO caches
        speed = float(speed_map.get(vehicle_id, 0.0))
        lane_pos_from_start = float(pos_map.get(vehicle_id, 0.0))  # distance from lane START
        # Convert to "distance to intersection (lane END)"
        # Keep the original module's semantics: smaller value -> closer to intersection
        position_from_start = max(0.0, lane_length - lane_pos_from_start)
        position_from_end = lane_pos_from_start

        is_moving = speed > 0.1

        wait_info = env.waiting_vehicle_list.get(vehicle_id, 0.0)
        if isinstance(wait_info, dict):
            wait_info = wait_info.get("time", 0.0)
        waiting_time = 0.0 if is_moving else float(wait_info)

        cell_position = _calculate_cell_position(position_from_start, lane_length)

        return {
            'vehicle_id': vehicle_id,
            'speed': speed,
            'position_from_start': position_from_start,       # distance to intersection (lane END)
            'position_from_end': position_from_end,           # distance from lane START (SUMO lanePosition)
            'cell_position': cell_position,
            'is_moving': is_moving,
            'waiting_time': waiting_time,
            'relative_position': (position_from_start / lane_length) if lane_length > 0 else 0.0
        }

    except Exception as e:
        return {
            'vehicle_id': vehicle_id,
            'speed': 0.0,
            'position_from_start': 0.0,
            'position_from_end': 0.0,
            'cell_position': 0,
            'is_moving': False,
            'waiting_time': 0.0,
            'relative_position': 0.0,
            'error': str(e)
        }


def _calculate_cell_position(position_from_start: float, lane_length: float) -> int:
    """
    Map a vehicle's distance to the intersection (lane END) to a cell index [0..3].
    Cell 0 is closest to the intersection; Cell 3 is farthest.

    Args:
        position_from_start: distance to intersection (meters)
        lane_length: total lane length (meters)

    Returns:
        int in [0, 3]
    """
    if lane_length <= 0:
        return 0

    cell_size = lane_length / 4.0

    if position_from_start <= cell_size:
        return 0
    elif position_from_start <= 2 * cell_size:
        return 1
    elif position_from_start <= 3 * cell_size:
        return 2
    else:
        return 3


def _get_lane_length(lane_id: str, env) -> float:
    """
    Get lane length directly from SUMOEnv cache.

    Args:
        lane_id: SUMO lane ID
        env: SUMOEnv instance

    Returns:
        Lane length in meters
    """
    try:
        # Use SUMOEnv's cached lane lengths
        return float(env.lane_length.get(lane_id, 100.0))
    except Exception:
        return 100.0
