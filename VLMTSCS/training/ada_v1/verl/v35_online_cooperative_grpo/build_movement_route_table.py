#!/usr/bin/env python3
"""Build and audit sender-movement routes for online cooperative GRPO.

The source topology is authoritative for physical neighbor and receiver-entry
directions. Movement semantics are fixed by the traffic-controller convention:
ET exits west, WT exits east, NT exits south, ST exits north; left turns exit
east/west/south/north for NL/SL/EL/WL respectively.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


MOVEMENT_EXIT = {
    "ET": "W", "WT": "E", "NT": "S", "ST": "N",
    "NL": "E", "SL": "W", "EL": "S", "WL": "N",
}
OPPOSITE = {"E": "W", "W": "E", "N": "S", "S": "N"}
CITY_TOPOLOGIES = {
    "jinan": "data/Jinan/3_4/network_topology.json",
    "hangzhou": "data/Hangzhou/4_4/network_topology.json",
    "newyork": "data/NewYork/28_7/network_topology.json",
    "newyork16x3": "data/NewYork/16x3/network_topology.json",
}


def ab_group(intersection_id: str) -> str:
    """Checkerboard grouping from stable ``intersection_<column>_<row>`` IDs."""
    match = re.fullmatch(r"intersection_(\d+)_(\d+)", intersection_id)
    if match is None:
        raise ValueError(f"Unsupported intersection ID for A/B grouping: {intersection_id}")
    column, row = (int(value) for value in match.groups())
    return "A" if (row + column) % 2 == 0 else "B"


def build_city(city: str, source: Path) -> tuple[dict[str, Any], list[str]]:
    topology = json.loads(source.read_text(encoding="utf-8"))
    intersections = topology["intersections"]
    issues: list[str] = []
    routes: dict[str, Any] = {}

    for sender_id, sender in sorted(intersections.items()):
        sender_group = ab_group(sender_id)
        rows: dict[str, Any] = {}
        for movement, exit_direction in MOVEMENT_EXIT.items():
            edge = sender["neighbors"].get(exit_direction)
            if edge is None:
                issues.append(f"{sender_id} {movement}: missing {exit_direction} neighbor definition")
                rows[movement] = {"is_boundary": True, "receiver_id": None}
                continue
            receiver_id = edge.get("neighbor_id")
            is_boundary = bool(edge.get("is_boundary", False) or receiver_id is None)
            route = {
                "exit_direction": exit_direction,
                "receiver_id": receiver_id,
                "receiver_entry_direction": edge.get("their_entry_direction"),
                "is_boundary": is_boundary,
            }
            if receiver_id is not None:
                if receiver_id not in intersections:
                    issues.append(f"{sender_id} {movement}: unknown receiver {receiver_id}")
                else:
                    receiver = intersections[receiver_id]
                    route["receiver_group"] = ab_group(receiver_id)
                    reverse = receiver["neighbors"].get(OPPOSITE[exit_direction], {})
                    if reverse.get("neighbor_id") != sender_id:
                        issues.append(f"{sender_id} {movement}: reverse edge from {receiver_id} is inconsistent")
                    if route["receiver_group"] == sender_group:
                        issues.append(f"{sender_id} {movement}: A/B grouping collision with {receiver_id}")
                    expected_entry = OPPOSITE[exit_direction]
                    if edge.get("their_entry_direction") != expected_entry:
                        issues.append(
                            f"{sender_id} {movement}: receiver entry is {edge.get('their_entry_direction')}, "
                            f"expected {expected_entry}"
                        )
            rows[movement] = route
        routes[sender_id] = {"sender_group": sender_group, "movements": rows}

    return {
        "city": city,
        "source_topology": str(source.as_posix()),
        "movement_exit_directions": MOVEMENT_EXIT,
        "routes": routes,
    }, issues


def write_audit(path: Path, cities: dict[str, dict[str, Any]], issues: dict[str, list[str]]) -> None:
    lines = ["V35 online cooperative GRPO movement routing audit", ""]
    for city, table in cities.items():
        rows = table["routes"]
        counts = Counter(
            "boundary" if route["is_boundary"] else "routable"
            for sender in rows.values() for route in sender["movements"].values()
        )
        lines.extend([
            f"## {city}",
            f"- intersections: {len(rows)}",
            f"- movement routes: {counts['routable']}",
            f"- boundary movements: {counts['boundary']}",
            f"- audit issues: {len(issues[city])}",
            "",
            "| Sender | Group | Movement | Exit | Receiver | Receiver entry | Boundary |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ])
        for sender_id, sender in rows.items():
            for movement, route in sender["movements"].items():
                lines.append(
                    f"| {sender_id} | {sender['sender_group']} | {movement} | "
                    f"{route.get('exit_direction', '')} | {route.get('receiver_id') or '-'} | "
                    f"{route.get('receiver_entry_direction') or '-'} | {str(route['is_boundary']).lower()} |"
                )
        if issues[city]:
            lines.extend(["", "Issues:", *[f"- {item}" for item in issues[city]]])
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    city_tables: dict[str, dict[str, Any]] = {}
    city_issues: dict[str, list[str]] = {}
    for city, relative in CITY_TOPOLOGIES.items():
        table, issues = build_city(city, args.repo_root / relative)
        city_tables[city] = table
        city_issues[city] = issues
        (args.output_dir / f"movement_routes_{city}.json").write_text(
            json.dumps(table, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    write_audit(args.output_dir / "movement_routes_audit.md", city_tables, city_issues)
    total = sum(len(items) for items in city_issues.values())
    print(f"wrote route tables for {len(city_tables)} cities; audit issues={total}")
    if total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
