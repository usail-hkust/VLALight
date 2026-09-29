import json
import os
from typing import Any, Dict, List, Optional


ORIENT_ORDER = ["W", "E", "N", "S"]
ROLE_ORDER = ["incoming", "outgoing"]
ROLE_DISTANCE_MIN_M = {
    "incoming": -5.0,
    "outgoing": -1.0,
}


class VehiclePositionSnapshotWriter:
    """Write SUMO ground-truth vehicle positions for later DINO/VLM alignment."""

    def __init__(self, output_dir: str, view_distance_m: float = 150.0) -> None:
        self.output_dir = output_dir
        self.view_distance_m = float(view_distance_m)
        os.makedirs(self.output_dir, exist_ok=True)
        self.jsonl_path = os.path.join(self.output_dir, "vehicle_position_snapshots.jsonl")
        self.index_path = os.path.join(self.output_dir, "vehicle_position_snapshot_index.json")
        self._count = 0

    def write_step(self, *, step_num: int, sim_time: float, env: Any,
                   image_paths: Optional[Dict[str, List[str]]] = None) -> None:
        record = {
            "step": int(step_num),
            "sim_time_s": float(sim_time),
            "view_distance_m": self.view_distance_m,
            "intersections": [],
        }

        for inter in env.list_intersection:
            record["intersections"].append(self._build_intersection_record(env, inter, image_paths or {}))

        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        self._count += 1
        self._write_index()

    def _write_index(self) -> None:
        index = {
            "jsonl_path": self.jsonl_path,
            "snapshot_count": self._count,
            "view_distance_m": self.view_distance_m,
            "distance_windows_m": {
                "incoming": [ROLE_DISTANCE_MIN_M["incoming"], self.view_distance_m],
                "outgoing": [ROLE_DISTANCE_MIN_M["outgoing"], self.view_distance_m],
            },
            "vehicle_order": "per camera view: lane_role incoming,outgoing; lane index; signed distance from observed stop line",
            "camera_views": "Each direction image has its own v1,v2,... labels under intersections[].camera_views.<N/E/W/S>.vehicles.",
            "distance_definition": {
                "incoming": "diagnose_camera_view_distance.py model: observed stop line is lane_end; signed_distance = lane_length_m - lane_pos_m; negative means after crossing lane_end into the intersection.",
                "outgoing": "diagnose_camera_view_distance.py model: observed stop line is lane_start; signed_distance = lane_pos_m; negative means before entering lane_start from the intersection.",
            },
            "image_alignment": "camera_views.<direction> corresponds to saved image <direction>.jpg from ImageSaver direction_mapping, e.g. camera_views.S -> S.jpg.",
        }
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)

    def _build_intersection_record(self, env: Any, inter: Any,
                                   image_paths: Dict[str, List[str]]) -> Dict[str, Any]:
        lane_groups = self._lane_groups_for_intersection(env, inter)
        intersection_record = {
            "tls_id": inter.inter_id,
            "intersection_name": getattr(inter, "inter_name", inter.inter_id),
            "vehicles": [],
            "lanes": [],
            "camera_views": {},
        }

        visual_counter = 1
        for lane_group in lane_groups:
            lane_vehicles = self._vehicles_for_lane(env, lane_group)
            lane_record = {
                **lane_group,
                "vehicles": [],
            }
            for vehicle in lane_vehicles:
                vehicle["visual_id"] = f"v{visual_counter}"
                visual_counter += 1
                lane_record["vehicles"].append(vehicle)
                intersection_record["vehicles"].append(vehicle)
            intersection_record["lanes"].append(lane_record)

        for orient in ORIENT_ORDER:
            intersection_record["camera_views"][orient] = self._build_camera_view_record(
                env=env,
                inter=inter,
                orient=orient,
                image_paths=image_paths,
            )

        return intersection_record

    def _build_camera_view_record(self, env: Any, inter: Any, orient: str,
                                  image_paths: Dict[str, List[str]]) -> Dict[str, Any]:
        lane_groups = [
            group
            for group in self._lane_groups_for_intersection(env, inter)
            if group["orient"] == orient and group["lane_role"] in ("incoming", "outgoing")
        ]
        image_filename = f"{orient}.jpg"
        image_path = self._image_path_for_direction(image_paths, inter.inter_id, image_filename)
        camera_record = {
            "tls_id": inter.inter_id,
            "intersection_name": getattr(inter, "inter_name", inter.inter_id),
            "camera_direction": orient,
            "image_filename": image_filename,
            "image_path": image_path,
            "vehicles": [],
            "lanes": [],
        }

        visual_counter = 1
        for lane_group in lane_groups:
            lane_vehicles = self._vehicles_for_lane(env, lane_group)
            lane_record = {
                **lane_group,
                "vehicles": [],
            }
            for vehicle in lane_vehicles:
                vehicle = dict(vehicle)
                vehicle["visual_id"] = f"v{visual_counter}"
                vehicle["camera_direction"] = orient
                visual_counter += 1
                lane_record["vehicles"].append(vehicle)
                camera_record["vehicles"].append(vehicle)
            camera_record["lanes"].append(lane_record)
        return camera_record

    @staticmethod
    def _image_path_for_direction(image_paths: Dict[str, List[str]], tls_id: str,
                                  image_filename: str) -> Optional[str]:
        for path in image_paths.get(tls_id, []):
            if os.path.basename(path) == image_filename:
                return path
        return None

    def _opposite_visible_lane_groups(self, env: Any, inter: Any, orient: str) -> List[Dict[str, Any]]:
        incoming_map = getattr(inter, "road_id_2_orient", {}).get("incoming", {}) or {}
        incoming_edges = sorted(edge_id for edge_id, edge_orient in incoming_map.items() if edge_orient == orient)
        groups = []
        intersection_name = getattr(inter, "inter_name", inter.inter_id)
        for incoming_edge_id in incoming_edges:
            reverse_edge_id, reverse_source = self._find_reverse_edge(env, incoming_edge_id)
            if not reverse_edge_id:
                continue
            for lane_id in self._lanes_for_edge(env, reverse_edge_id):
                lane_idx = self._lane_index(lane_id)
                groups.append({
                    "tls_id": inter.inter_id,
                    "intersection_name": intersection_name,
                    "orient": orient,
                    "lane_role": "opposite_visible",
                    "edge_id": reverse_edge_id,
                    "source_incoming_edge_id": incoming_edge_id,
                    "opposite_source": reverse_source,
                    "lane_id": lane_id,
                    "lane_index": lane_idx,
                    "lane_name": f"{orient}_opposite_visible_lane{lane_idx}",
                    "distance_reference": "to_upstream_stopline",
                })
        return groups

    def _find_reverse_edge(self, env: Any, edge_id: str) -> (Optional[str], str):
        reverse_edge_id = self._find_reverse_edge_from_topology(env, edge_id)
        if reverse_edge_id:
            return reverse_edge_id, "topology"

        reverse_edge_id = self._find_reverse_edge_by_id(env, edge_id)
        if reverse_edge_id:
            return reverse_edge_id, "edge_id"

        return None, "not_found"

    def _find_reverse_edge_from_topology(self, env: Any, edge_id: str) -> Optional[str]:
        try:
            edge = env.sumo_net.getEdge(edge_id)
        except Exception:
            edge = None
        if edge is None:
            return None

        candidates = set()
        try:
            for incoming_edge in edge.getIncoming().keys():
                candidates.add(incoming_edge.getID())
        except Exception:
            pass

        for candidate in sorted(candidates):
            if candidate != edge_id:
                return candidate
        return None

    def _find_reverse_edge_by_id(self, env: Any, edge_id: str) -> Optional[str]:
        candidates = []
        if edge_id.startswith("-"):
            candidates.append(edge_id[1:])
        else:
            candidates.append(f"-{edge_id}")
        if edge_id.startswith("road_"):
            parts = edge_id.split("_")
            if len(parts) == 5:
                candidates.append("_".join(["road", parts[3], parts[4], parts[1], parts[2]]))

        for candidate in candidates:
            if candidate == edge_id:
                continue
            try:
                env.traci_conn.edge.getLaneNumber(candidate)
            except Exception:
                continue
            return candidate
        return None

    def _lane_groups_for_intersection(self, env: Any, inter: Any) -> List[Dict[str, Any]]:
        groups = []
        intersection_name = getattr(inter, "inter_name", inter.inter_id)
        for orient in ORIENT_ORDER:
            for role in ROLE_ORDER:
                road_map = getattr(inter, "road_id_2_orient", {}).get(role, {}) or {}
                edge_ids = sorted(edge_id for edge_id, edge_orient in road_map.items() if edge_orient == orient)
                for edge_id in edge_ids:
                    for lane_id in self._lanes_for_edge(env, edge_id):
                        lane_idx = self._lane_index(lane_id)
                        groups.append({
                            "tls_id": inter.inter_id,
                            "intersection_name": intersection_name,
                            "orient": orient,
                            "lane_role": role,
                            "edge_id": edge_id,
                            "lane_id": lane_id,
                            "internal_lane_ids": self._internal_lanes_for_observed_lane(env, lane_id, role),
                            "lane_index": lane_idx,
                            "lane_name": f"{orient}_{role}_lane{lane_idx}",
                            "distance_reference": "to_current_stopline" if role == "incoming" else "from_current_stopline",
                        })
        return groups

    def _internal_lanes_for_observed_lane(self, env: Any, lane_id: str, role: str) -> List[str]:
        if role == "incoming":
            return self._link_internal_lane_ids(env, lane_id)

        internal_lane_ids = []
        edge_id = lane_id.rsplit("_", 1)[0]
        for candidate_lane_id in self._candidate_lanes_linking_to_edge(env, edge_id):
            for link in self._lane_links(env, candidate_lane_id):
                if self._link_to_lane_id(link) != lane_id:
                    continue
                internal_lane_ids.extend(self._internal_lane_ids_from_link(link))
        return sorted(set(internal_lane_ids))

    def _candidate_lanes_linking_to_edge(self, env: Any, edge_id: str) -> List[str]:
        candidate_lanes = []
        try:
            edge = env.sumo_net.getEdge(edge_id)
            incoming_edges = sorted(item.getID() for item in edge.getIncoming().keys())
        except Exception:
            incoming_edges = []
        for incoming_edge_id in incoming_edges:
            candidate_lanes.extend(self._lanes_for_edge(env, incoming_edge_id))
        return candidate_lanes

    def _link_internal_lane_ids(self, env: Any, lane_id: str) -> List[str]:
        internal_lane_ids = []
        for link in self._lane_links(env, lane_id):
            internal_lane_ids.extend(self._internal_lane_ids_from_link(link))
        return sorted(set(internal_lane_ids))

    def _lane_links(self, env: Any, lane_id: str) -> List[Any]:
        try:
            return list(env.traci_conn.lane.getLinks(lane_id))
        except Exception:
            return []

    @staticmethod
    def _link_to_lane_id(link: Any) -> Optional[str]:
        if isinstance(link, dict):
            value = link.get("approachedLane") or link.get("toLane") or link.get("lane")
            return value if isinstance(value, str) else None
        if isinstance(link, (list, tuple)) and link and isinstance(link[0], str):
            return link[0]
        return None

    @staticmethod
    def _internal_lane_ids_from_link(link: Any) -> List[str]:
        if isinstance(link, dict):
            values = link.values()
        elif isinstance(link, (list, tuple)):
            values = link
        else:
            values = []
        return sorted({value for value in values if isinstance(value, str) and value.startswith(":")})

    def _lanes_for_edge(self, env: Any, edge_id: str) -> List[str]:
        lanes = []
        try:
            lane_count = int(env.traci_conn.edge.getLaneNumber(edge_id))
        except Exception:
            return lanes
        for lane_idx in range(lane_count):
            lane_id = f"{edge_id}_{lane_idx}"
            try:
                env.traci_conn.lane.getLength(lane_id)
            except Exception:
                continue
            lanes.append(lane_id)
        return lanes

    def _vehicles_for_lane(self, env: Any, lane_group: Dict[str, Any]) -> List[Dict[str, Any]]:
        lane_id = lane_group["lane_id"]
        try:
            vehicle_ids = list(env.traci_conn.lane.getLastStepVehicleIDs(lane_id))
        except Exception:
            vehicle_ids = list(env.system_states.get("get_lane_vehicles", {}).get(lane_id, []))
        for internal_lane_id in lane_group.get("internal_lane_ids", []):
            try:
                vehicle_ids.extend(env.traci_conn.lane.getLastStepVehicleIDs(internal_lane_id))
            except Exception:
                continue
        vehicle_ids = sorted(set(vehicle_ids))

        vehicles = []
        for vehicle_id in vehicle_ids:
            info = self._vehicle_info(env, vehicle_id, lane_group)
            if info is not None and self._within_role_distance_window(info):
                vehicles.append(info)

        vehicles.sort(key=lambda item: (
            item["signed_distance_from_observed_stopline_m"],
            item["lane_pos_m"],
            item["sumo_vehicle_id"],
        ))
        return vehicles

    def _within_role_distance_window(self, vehicle_info: Dict[str, Any]) -> bool:
        role = vehicle_info.get("lane_role", "")
        distance = float(vehicle_info.get("signed_distance_from_observed_stopline_m", float("inf")))
        min_distance = ROLE_DISTANCE_MIN_M.get(role, 0.0)
        return min_distance <= distance <= self.view_distance_m

    def _vehicle_info(self, env: Any, vehicle_id: str, lane_group: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        lane_id = lane_group["lane_id"]
        role = lane_group["lane_role"]
        try:
            current_lane = env.traci_conn.vehicle.getLaneID(vehicle_id)
            lane_pos = float(env.traci_conn.vehicle.getLanePosition(vehicle_id))
            lane_len = float(env.traci_conn.lane.getLength(lane_id))
            speed = float(env.traci_conn.vehicle.getSpeed(vehicle_id))
            position_xy = env.traci_conn.vehicle.getPosition(vehicle_id)
        except Exception:
            return None

        if current_lane != lane_id and current_lane not in set(lane_group.get("internal_lane_ids", [])):
            return None

        if role == "outgoing":
            signed_distance_from_observed = self._signed_distance_from_lane_start(
                env,
                lane_id,
                position_xy,
            ) if current_lane != lane_id else lane_pos
            observed_stopline_endpoint = "lane_start"
        else:
            signed_distance_from_observed = self._signed_distance_to_lane_end(
                env,
                lane_id,
                position_xy,
            ) if current_lane != lane_id else lane_len - lane_pos
            observed_stopline_endpoint = "lane_end"

        distance_from_observed = max(0.0, signed_distance_from_observed)

        eta_s = None
        if speed > 0.1:
            eta_s = distance_from_observed / speed

        return {
            "sumo_vehicle_id": vehicle_id,
            "tls_id": lane_group.get("tls_id"),
            "intersection_name": lane_group.get("intersection_name"),
            "lane_id": lane_id,
            "current_lane_id": current_lane,
            "lane_name": lane_group["lane_name"],
            "distance_reference": lane_group.get("distance_reference"),
            "observed_stopline_endpoint": observed_stopline_endpoint,
            "orient": lane_group["orient"],
            "lane_role": role,
            "lane_index": lane_group["lane_index"],
            "lane_length_m": round(lane_len, 3),
            "lane_pos_m": round(lane_pos, 3),
            "signed_distance_from_observed_stopline_m": round(signed_distance_from_observed, 3),
            "distance_from_observed_stopline_m": round(distance_from_observed, 3),
            "speed_mps": round(speed, 3),
            "eta_to_observed_stopline_s": round(eta_s, 3) if eta_s is not None else None,
            "position_xy": [round(float(position_xy[0]), 3), round(float(position_xy[1]), 3)],
        }

    def _signed_distance_to_lane_end(self, env: Any, lane_id: str, position_xy: Any) -> float:
        shape = env.traci_conn.lane.getShape(lane_id)
        if len(shape) < 2:
            return 0.0
        prev_xy = shape[-2]
        end_xy = shape[-1]
        ux, uy = self._unit_vector(prev_xy, end_xy)
        dx = float(position_xy[0]) - float(end_xy[0])
        dy = float(position_xy[1]) - float(end_xy[1])
        return -(dx * ux + dy * uy)

    def _signed_distance_from_lane_start(self, env: Any, lane_id: str, position_xy: Any) -> float:
        shape = env.traci_conn.lane.getShape(lane_id)
        if len(shape) < 2:
            return 0.0
        start_xy = shape[0]
        next_xy = shape[1]
        ux, uy = self._unit_vector(start_xy, next_xy)
        dx = float(position_xy[0]) - float(start_xy[0])
        dy = float(position_xy[1]) - float(start_xy[1])
        return dx * ux + dy * uy

    @staticmethod
    def _unit_vector(start_xy: Any, end_xy: Any) -> (float, float):
        dx = float(end_xy[0]) - float(start_xy[0])
        dy = float(end_xy[1]) - float(start_xy[1])
        norm = (dx * dx + dy * dy) ** 0.5
        if norm <= 1e-6:
            return 0.0, 0.0
        return dx / norm, dy / norm

    @staticmethod
    def _lane_index(lane_id: str) -> int:
        try:
            return int(lane_id.rsplit("_", 1)[-1])
        except Exception:
            return -1
