#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def fail(message: str) -> None:
    raise ValueError(message)


def bounds(value: Any, label: str) -> tuple[float, float, float, float]:
    if not isinstance(value, list) or len(value) != 4:
        fail(f"{label} must be [xmin,ymin,xmax,ymax]")
    result = tuple(float(v) for v in value)
    if not all(math.isfinite(v) for v in result) or result[0] >= result[2] or result[1] >= result[3]:
        fail(f"{label} is not a finite positive rectangle")
    return result  # type: ignore[return-value]


def overlap_area(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def touches(a: tuple[float, ...], b: tuple[float, ...], eps: float = 1e-9) -> bool:
    vertical = (abs(a[2] - b[0]) <= eps or abs(b[2] - a[0]) <= eps) and min(a[3], b[3]) - max(a[1], b[1]) >= 0.9
    horizontal = (abs(a[3] - b[1]) <= eps or abs(b[3] - a[1]) <= eps) and min(a[2], b[2]) - max(a[0], b[0]) >= 0.9
    return vertical or horizontal


def zone_parts(record: dict[str, Any], label: str) -> list[tuple[float, float, float, float]]:
    if "bounds_xy_parts" in record:
        raw = record["bounds_xy_parts"]
        if not isinstance(raw, list) or not raw:
            fail(f"{label}.bounds_xy_parts must be a nonempty list")
        return [bounds(value, f"{label}.bounds_xy_parts[{index}]") for index, value in enumerate(raw)]
    return [bounds(record.get("bounds_xy"), f"{label}.bounds_xy")]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.contract.read_bytes()
    data = json.loads(raw)
    if data.get("contract_id") != "food_factory_full_quality_20260713":
        fail("unexpected contract_id")
    building = data["building"]
    if [building.get("width_m"), building.get("depth_m"), building.get("clear_height_m")] != [44.0, 32.0, 5.0]:
        fail("building dimensions must be exactly 44x32x5 m")
    shell = bounds(building["shell_bounds_xy"], "shell")
    rooms = data.get("rooms")
    order = data.get("room_order")
    if not isinstance(rooms, dict) or not isinstance(order, list) or len(rooms) != 14 or set(order) != set(rooms):
        fail("contract must contain exactly the ordered 14 enclosed rooms")
    rectangles = {name: bounds(record.get("bounds_xy"), f"rooms.{name}.bounds_xy") for name, record in rooms.items()}
    for name, rect in rectangles.items():
        if rect[0] < shell[0] or rect[1] < shell[1] or rect[2] > shell[2] or rect[3] > shell[3]:
            fail(f"{name} lies outside the shell")
        inventory = rooms[name].get("inventory")
        if not isinstance(inventory, list) or not inventory or len(inventory) != len(set(inventory)):
            fail(f"{name} inventory must be nonempty and distinct")
    names = list(rectangles)
    overlaps = []
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            area = overlap_area(rectangles[left], rectangles[right])
            if area > 1e-9:
                overlaps.append({"left": left, "right": right, "area_m2": area})
    if overlaps:
        fail(f"room overlap: {overlaps}")
    zones = {name: zone_parts(record, f"common_zones.{name}") for name, record in data["common_zones"].items()}
    all_parts = {name: [rect] for name, rect in rectangles.items()} | zones
    connections = data.get("required_connections")
    if not isinstance(connections, list) or len(connections) != len({tuple(edge) for edge in connections}):
        fail("required_connections must be a distinct list")
    bad_edges = []
    for edge in connections:
        if not isinstance(edge, list) or len(edge) != 2 or edge[0] not in all_parts or edge[1] not in all_parts or not any(touches(left, right) or overlap_area(left, right) >= 0.9 for left in all_parts[edge[0]] for right in all_parts[edge[1]]):
            bad_edges.append(edge)
    if bad_edges:
        fail(f"connections lack a shared >=0.9m boundary: {bad_edges}")
    required_window_rooms = {"ingredient_receiving","dry_storage","cold_storage","washing_preparation","qc_laboratory","office_administration","finished_goods_storage","maintenance","changing_room","break_room","boys_toilet","girls_toilet"}
    missing_windows = sorted(name for name in required_window_rooms if not rooms[name].get("windows"))
    if missing_windows:
        fail(f"rooms missing mandatory windows: {missing_windows}")
    global_req = data["global_requirements"]
    if global_req.get("minimum_artiverse_survivors") != 1 or global_req.get("max_collision_hulls_per_object") != 32 or global_req.get("no_hssd") is not True:
        fail("full-quality Artiverse/collision/no-HSSD invariants changed")
    result = {
        "schema_version": 1,
        "status": "pass",
        "contract_id": data["contract_id"],
        "contract_sha256": hashlib.sha256(raw).hexdigest(),
        "room_count": len(rooms),
        "room_order": order,
        "positive_area_overlaps": overlaps,
        "required_connection_count": len(connections),
        "inventory_family_count_by_room": {name: len(rooms[name]["inventory"]) for name in order},
        "articulated_roles": sorted(data["articulated_roles"]),
        "minimum_artiverse_survivors": 1,
        "max_collision_hulls_per_object": 32
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
