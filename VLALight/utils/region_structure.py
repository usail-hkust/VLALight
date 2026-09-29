import argparse
import json
import math
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET


def _load_xml(xml_path: str) -> ET.Element:
    tree = ET.parse(xml_path)
    return tree.getroot()


def _read_json(json_path: str) -> Dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_float(value: Optional[str], default: float = 0.0) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _first_existing_file(base_dir: str, candidates: List[str]) -> Optional[str]:
    for name in candidates:
        path = os.path.join(base_dir, name)
        if os.path.exists(path):
            return path
    return None


def _resolve_sumocfg_paths(sumocfg_path: str) -> Dict[str, Any]:
    sumocfg_path = os.path.abspath(sumocfg_path)
    base_dir = os.path.dirname(sumocfg_path)
    root = _load_xml(sumocfg_path)

    input_node = root.find("input")
    if input_node is None:
        raise ValueError(f"Invalid SUMO config without <input>: {sumocfg_path}")

    net_file_value = None
    net_file_elem = input_node.find("net-file")
    if net_file_elem is not None:
        net_file_value = net_file_elem.get("value") or (net_file_elem.text or "").strip()
    route_files_value = None
    route_files_elem = input_node.find("route-files")
    if route_files_elem is not None:
        route_files_value = route_files_elem.get("value")

    if not net_file_value:
        raise ValueError(f"SUMO config missing net-file value: {sumocfg_path}")

    net_path = os.path.join(base_dir, net_file_value)
    if not os.path.exists(net_path):
        raise FileNotFoundError(f"Referenced net file not found: {net_path}")

    route_paths = []
    missing_route_paths = []
    if route_files_value:
        for part in route_files_value.split(","):
            candidate = os.path.join(base_dir, part.strip())
            if os.path.exists(candidate):
                route_paths.append(candidate)
            else:
                missing_route_paths.append(candidate)

    phase_mapping_path = _first_existing_file(
        base_dir,
        [name for name in os.listdir(base_dir) if name.endswith("_phase_mapping.json")]
        + [name for name in os.listdir(base_dir) if name.endswith("phase_mapping.json")]
    )

    return {
        "sumocfg_path": sumocfg_path,
        "base_dir": base_dir,
        "net_path": os.path.abspath(net_path),
        "route_paths": [os.path.abspath(p) for p in route_paths],
        "missing_route_paths": [os.path.abspath(p) for p in missing_route_paths],
        "phase_mapping_path": os.path.abspath(phase_mapping_path) if phase_mapping_path else None,
    }


def _direction_from_delta(dx: float, dy: float) -> str:
    if abs(dx) >= abs(dy):
        return "E" if dx > 0 else "W"
    return "N" if dy > 0 else "S"


def _parse_net(net_path: str) -> Dict[str, Any]:
    root = _load_xml(net_path)

    junctions: Dict[str, Dict[str, Any]] = {}
    edges: Dict[str, Dict[str, Any]] = {}
    connections: List[Dict[str, Any]] = []

    for junction in root.findall("junction"):
        junction_id = junction.get("id")
        if not junction_id:
            continue
        junctions[junction_id] = {
            "id": junction_id,
            "type": junction.get("type", ""),
            "x": _safe_float(junction.get("x")),
            "y": _safe_float(junction.get("y")),
        }

    for edge in root.findall("edge"):
        edge_id = edge.get("id")
        if not edge_id or edge_id.startswith(":"):
            continue
        lanes = edge.findall("lane")
        lane_lengths = [_safe_float(lane.get("length")) for lane in lanes]
        lane_speeds = [_safe_float(lane.get("speed")) for lane in lanes]
        edges[edge_id] = {
            "id": edge_id,
            "from": edge.get("from"),
            "to": edge.get("to"),
            "priority": edge.get("priority"),
            "num_lanes": len(lanes),
            "lane_ids": [lane.get("id") for lane in lanes if lane.get("id")],
            "length": lane_lengths[0] if lane_lengths else 0.0,
            "avg_speed": lane_speeds[0] if lane_speeds else 0.0,
        }

    for conn in root.findall("connection"):
        from_edge = conn.get("from")
        to_edge = conn.get("to")
        tl_id = conn.get("tl")
        if not from_edge or not to_edge or not tl_id:
            continue
        if from_edge.startswith(":") or to_edge.startswith(":"):
            continue
        connections.append({
            "intersection_id": tl_id,
            "from_edge": from_edge,
            "to_edge": to_edge,
            "from_lane": conn.get("fromLane"),
            "to_lane": conn.get("toLane"),
            "turn_dir": conn.get("dir"),
            "link_index": conn.get("linkIndex"),
            "state": conn.get("state"),
        })

    return {
        "junctions": junctions,
        "edges": edges,
        "connections": connections,
    }


def _load_phase_mapping(phase_mapping_path: Optional[str]) -> Dict[str, List[str]]:
    if not phase_mapping_path or not os.path.exists(phase_mapping_path):
        return {}
    raw = _read_json(phase_mapping_path)
    result: Dict[str, List[str]] = {}
    for key, value in raw.items():
        if isinstance(value, list):
            result[key] = value
    return result


def _build_intersection_set(
    junctions: Dict[str, Dict[str, Any]],
    phase_mapping: Dict[str, List[str]]
) -> List[str]:
    controlled = []
    for junction_id, info in junctions.items():
        jtype = info.get("type", "")
        if jtype.startswith("traffic_light") or junction_id in phase_mapping:
            controlled.append(junction_id)
    return sorted(controlled)


def _build_directional_neighbors(
    intersection_id: str,
    junctions: Dict[str, Dict[str, Any]],
    edges: Dict[str, Dict[str, Any]],
    controlled_ids: set,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, str], Dict[str, str]]:
    center = junctions[intersection_id]
    upstream: Dict[str, Any] = {}
    downstream: Dict[str, Any] = {}
    incoming_edge_to_direction: Dict[str, str] = {}
    outgoing_edge_to_direction: Dict[str, str] = {}

    cx = center["x"]
    cy = center["y"]

    for edge_id, edge in edges.items():
        from_id = edge.get("from")
        to_id = edge.get("to")
        if not from_id or not to_id:
            continue

        if to_id == intersection_id and from_id in junctions:
            src = junctions[from_id]
            direction = _direction_from_delta(src["x"] - cx, src["y"] - cy)
            incoming_edge_to_direction[edge_id] = direction
            upstream[direction] = {
                "neighbor_id": from_id if from_id in controlled_ids else None,
                "distance_m": round(edge.get("length", 0.0), 3),
                "avg_speed_mps": round(edge.get("avg_speed", 0.0), 3),
                "travel_time_s": round(
                    edge.get("length", 0.0) / edge.get("avg_speed", 1.0),
                    3,
                ) if edge.get("avg_speed", 0.0) > 0 else None,
                "is_boundary": from_id not in controlled_ids,
            }

        if from_id == intersection_id and to_id in junctions:
            dst = junctions[to_id]
            direction = _direction_from_delta(dst["x"] - cx, dst["y"] - cy)
            outgoing_edge_to_direction[edge_id] = direction
            downstream[direction] = {
                "neighbor_id": to_id if to_id in controlled_ids else None,
                "distance_m": round(edge.get("length", 0.0), 3),
                "avg_speed_mps": round(edge.get("avg_speed", 0.0), 3),
                "travel_time_s": round(
                    edge.get("length", 0.0) / edge.get("avg_speed", 1.0),
                    3,
                ) if edge.get("avg_speed", 0.0) > 0 else None,
                "is_boundary": to_id not in controlled_ids,
            }

    for direction in ("N", "S", "E", "W"):
        upstream.setdefault(direction, None)
        downstream.setdefault(direction, None)

    return upstream, downstream, incoming_edge_to_direction, outgoing_edge_to_direction


def _build_movements(
    intersection_id: str,
    connections: List[Dict[str, Any]],
    edges: Dict[str, Dict[str, Any]],
    junctions: Dict[str, Dict[str, Any]],
    controlled_ids: set,
    incoming_edge_to_direction: Dict[str, str],
    outgoing_edge_to_direction: Dict[str, str],
) -> Dict[str, Dict[str, Any]]:
    movements: Dict[str, Dict[str, Any]] = {}
    turn_map = {"l": "left", "s": "straight", "r": "right"}

    for conn in connections:
        if conn["intersection_id"] != intersection_id:
            continue

        from_edge = conn["from_edge"]
        to_edge = conn["to_edge"]
        incoming_direction = incoming_edge_to_direction.get(from_edge)
        movement_key = turn_map.get(conn.get("turn_dir", ""))
        if not incoming_direction or not movement_key:
            continue

        if incoming_direction not in movements:
            movements[incoming_direction] = {}

        out_edge = edges.get(to_edge, {})
        target_node_id = out_edge.get("to")
        target_node = junctions.get(target_node_id, {})

        target_incoming_direction = None
        if target_node_id and target_node_id in controlled_ids and to_edge in edges:
            tx = target_node.get("x", 0.0)
            ty = target_node.get("y", 0.0)
            source_node_id = out_edge.get("from")
            source_node = junctions.get(source_node_id, {})
            sx = source_node.get("x", 0.0)
            sy = source_node.get("y", 0.0)
            target_incoming_direction = _direction_from_delta(sx - tx, sy - ty)

        movements[incoming_direction][movement_key] = {
            "target_neighbor_id": target_node_id if target_node_id in controlled_ids else None,
            "target_entry_direction": target_incoming_direction,
            "target_exit_direction": outgoing_edge_to_direction.get(to_edge),
            "distance_m": round(out_edge.get("length", 0.0), 3),
            "avg_speed_mps": round(out_edge.get("avg_speed", 0.0), 3),
            "travel_time_s": round(
                out_edge.get("length", 0.0) / out_edge.get("avg_speed", 1.0),
                3,
            ) if out_edge.get("avg_speed", 0.0) > 0 else None,
            "is_boundary": target_node_id not in controlled_ids,
        }

    for incoming_direction in ("N", "S", "E", "W"):
        if incoming_direction not in movements:
            continue
        for movement_key in ("left", "straight", "right"):
            movements[incoming_direction].setdefault(movement_key, None)

    return movements


def generate_region_structure(sumocfg_path: str) -> Dict[str, Any]:
    paths = _resolve_sumocfg_paths(sumocfg_path)
    phase_mapping = _load_phase_mapping(paths["phase_mapping_path"])
    net = _parse_net(paths["net_path"])

    junctions = net["junctions"]
    edges = net["edges"]
    connections = net["connections"]
    controlled_ids = set(_build_intersection_set(junctions, phase_mapping))

    intersections: Dict[str, Any] = {}
    for intersection_id in sorted(controlled_ids):
        upstream, downstream, incoming_edge_to_direction, outgoing_edge_to_direction = _build_directional_neighbors(
            intersection_id,
            junctions,
            edges,
            controlled_ids,
        )
        movements = _build_movements(
            intersection_id,
            connections,
            edges,
            junctions,
            controlled_ids,
            incoming_edge_to_direction,
            outgoing_edge_to_direction,
        )

        intersections[intersection_id] = {
            "coordinates": {
                "x": junctions[intersection_id]["x"],
                "y": junctions[intersection_id]["y"],
            },
            "junction_type": junctions[intersection_id].get("type", ""),
            "phases": phase_mapping.get(intersection_id, []),
            "upstream": upstream,
            "downstream": downstream,
            "movements": movements,
        }

    scenario_name = os.path.splitext(os.path.basename(paths["sumocfg_path"]))[0]
    return {
        "schema_version": "1.0",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "scenario_name": scenario_name,
        "source": {
            "sumocfg": paths["sumocfg_path"],
            "net": paths["net_path"],
            "phase_mapping": paths["phase_mapping_path"],
            "route_files": paths["route_paths"],
            "missing_route_files": paths["missing_route_paths"],
        },
        "units": {
            "distance": "meter",
            "speed": "meter_per_second",
            "travel_time": "second",
        },
        "intersections": intersections,
    }


def save_region_structure(region_structure: Dict[str, Any], output_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(region_structure, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate region structure JSON from a SUMO sumocfg/net.xml scene."
    )
    parser.add_argument(
        "--sumocfg",
        required=True,
        help="Path to SUMO .sumocfg file.",
    )
    parser.add_argument(
        "--output",
        required=False,
        help="Output JSON path. Defaults to <scene_dir>/<sumocfg_name>_region_structure.json",
    )
    args = parser.parse_args()

    sumocfg_path = os.path.abspath(args.sumocfg)
    default_output = os.path.join(
        os.path.dirname(sumocfg_path),
        f"{os.path.splitext(os.path.basename(sumocfg_path))[0]}_region_structure.json",
    )
    output_path = os.path.abspath(args.output) if args.output else default_output

    region_structure = generate_region_structure(sumocfg_path)
    save_region_structure(region_structure, output_path)

    print(f"Generated region structure for {len(region_structure['intersections'])} intersections")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
