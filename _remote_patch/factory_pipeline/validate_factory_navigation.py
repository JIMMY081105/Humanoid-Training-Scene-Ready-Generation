#!/usr/bin/env python3
"""Fail-closed pedestrian/forklift navigation proof for the final food factory.

The proof deliberately uses only immutable scene artifacts: the validated
``house_layout.json``, every final room state, and the assembled
``combined_house/house_state.json``.  It does not trust connectivity booleans.
Instead, it reconstructs door portals, transforms object AABBs from room-local
coordinates into house coordinates, inflates them by the humanoid radius, and
runs deterministic grid searches through explicitly named circulation zones.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = 1
ALGORITHM = "factory_navigation_grid_v1"
VERIFICATION_SCHEMA_ID = "scenesmith_factory_navigation_verification_v1"
EXPECTED_ROOM_IDS = (
    "ingredient_receiving",
    "dry_storage",
    "cold_storage",
    "washing_preparation",
    "qc_laboratory",
    "office_administration",
    "processing_hall",
    "packaging_hall",
    "finished_goods_storage",
    "maintenance",
    "changing_room",
    "break_room",
    "boys_toilet",
    "girls_toilet",
)
EXPECTED_ROOM_SET = frozenset(EXPECTED_ROOM_IDS)
CORE_COMMON_ZONES = frozenset(
    {"entrance_transition", "internal_circulation", "toilet_foyer", "loading_dock"}
)
TURNING_ROOM_IDS = frozenset(EXPECTED_ROOM_IDS)
STRUCTURAL_TYPES = frozenset({"wall", "floor", "ceiling"})
MINIMUM_DOOR_WIDTH_M = 0.9
MINIMUM_ENTRANCE_WIDTH_M = 1.6
DEFAULT_HUMANOID_RADIUS_M = 0.35
DEFAULT_GRID_RESOLUTION_M = 0.15
DEFAULT_TURNING_DIAMETER_M = 1.5
DEFAULT_HUMANOID_HEIGHT_M = 2.0
DEFAULT_FORKLIFT_RADIUS_M = 1.5
FORKLIFT_TARGETS = frozenset(
    {"ingredient_receiving", "dry_storage", "cold_storage", "finished_goods_storage"}
)
EPSILON = 1e-7

Cell = tuple[int, int]
Bounds = tuple[float, float, float, float]
Node = tuple[str, Cell]


@dataclass(frozen=True)
class Zone:
    zone_id: str
    bounds: Bounds
    kind: str
    carved_from: tuple[str, ...] = ()


@dataclass(frozen=True)
class Portal:
    portal_id: str
    zone_a: str
    zone_b: str | None
    width_m: float
    center: tuple[float, float]
    orientation: str
    source: str


@dataclass(frozen=True)
class Obstacle:
    room_id: str
    object_id: str
    raw_bounds: Bounds
    inflated_bounds: Bounds
    z_range: tuple[float, float]
    rotation_source: str


class ValidationError(RuntimeError):
    """Raised for a malformed required input, never for ordinary no-path results."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValidationError(f"JSON root must be an object: {path}")
    return value


def _number(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise ValidationError(f"{label} must be finite")
    return result


def _vector(value: Any, length: int, label: str) -> tuple[float, ...]:
    if isinstance(value, Mapping):
        axes = ("x", "y", "z", "w")[:length]
        raw = [value.get(axis) for axis in axes]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) < length:
            raise ValidationError(f"{label} needs {length} components")
        raw = list(value[:length])
    else:
        raise ValidationError(f"{label} must be a vector")
    return tuple(_number(component, f"{label}[{index}]") for index, component in enumerate(raw))


def _room_entries(layout: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = layout.get("placed_rooms")
    if isinstance(raw, dict):
        entries = list(raw.values())
    elif isinstance(raw, list):
        entries = raw
    else:
        raise ValidationError("house_layout.placed_rooms must be a list or mapping")
    if not all(isinstance(entry, dict) for entry in entries):
        raise ValidationError("house_layout.placed_rooms contains a non-object entry")
    return entries


def _room_bounds(room: Mapping[str, Any], room_id: str) -> Bounds:
    position = room.get("position")
    if isinstance(position, Mapping):
        x, y = _vector(position, 2, f"{room_id}.position")
    else:
        x, y = _vector(position, 2, f"{room_id}.position")
    width = _number(room.get("width"), f"{room_id}.width")
    depth = _number(room.get("depth", room.get("length")), f"{room_id}.depth")
    if width <= 0.0 or depth <= 0.0:
        raise ValidationError(f"{room_id} dimensions must be positive")
    return (x, y, x + width, y + depth)


def _rooms(layout: Mapping[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, Bounds]]:
    records: dict[str, dict[str, Any]] = {}
    bounds: dict[str, Bounds] = {}
    for room in _room_entries(layout):
        room_id = str(room.get("room_id") or room.get("id") or "").strip()
        if not room_id:
            raise ValidationError("placed room is missing room_id")
        if room_id in records:
            raise ValidationError(f"duplicate placed room ID: {room_id}")
        records[room_id] = room
        bounds[room_id] = _room_bounds(room, room_id)
    actual = set(records)
    if actual != EXPECTED_ROOM_SET:
        raise ValidationError(
            f"expected exactly the 14 factory rooms; missing={sorted(EXPECTED_ROOM_SET - actual)}, "
            f"unexpected={sorted(actual - EXPECTED_ROOM_SET)}"
        )
    return records, bounds


def _bounds_overlap(first: Bounds, second: Bounds) -> Bounds | None:
    overlap = (
        max(first[0], second[0]),
        max(first[1], second[1]),
        min(first[2], second[2]),
        min(first[3], second[3]),
    )
    return overlap if overlap[2] - overlap[0] > EPSILON and overlap[3] - overlap[1] > EPSILON else None


def _point_in_bounds(point: tuple[float, float], bounds: Bounds, margin: float = 0.0) -> bool:
    return (
        bounds[0] + margin - EPSILON <= point[0] <= bounds[2] - margin + EPSILON
        and bounds[1] + margin - EPSILON <= point[1] <= bounds[3] - margin + EPSILON
    )


def _common_zones(
    layout: Mapping[str, Any], room_bounds: Mapping[str, Bounds]
) -> tuple[dict[str, Zone], list[dict[str, Any]]]:
    zones = {
        room_id: Zone(room_id, room_bounds[room_id], "room")
        for room_id in EXPECTED_ROOM_IDS
    }
    raw_zones = layout.get("navigation_common_zones", [])
    if raw_zones is None:
        raw_zones = []
    if not isinstance(raw_zones, list) or not all(isinstance(item, dict) for item in raw_zones):
        raise ValidationError("navigation_common_zones must be a list of objects")
    evidence = []
    for item in raw_zones:
        zone_id = str(item.get("id") or "").strip()
        if not zone_id or zone_id in zones:
            raise ValidationError(f"invalid or duplicate common-zone ID: {zone_id!r}")
        reason = str(item.get("reason") or "").strip()
        if not reason:
            raise ValidationError(f"common zone {zone_id} must state its safe-circulation reason")
        bounds = _room_bounds(item, f"navigation_common_zones[{zone_id}]")
        carved_raw = item.get("carved_from", [])
        if not isinstance(carved_raw, list):
            raise ValidationError(f"common zone {zone_id}.carved_from must be a list")
        carved = tuple(sorted({str(value) for value in carved_raw}))
        invalid_carves = sorted(set(carved) - EXPECTED_ROOM_SET)
        if invalid_carves:
            raise ValidationError(f"common zone {zone_id} carves unknown rooms: {invalid_carves}")
        overlaps = sorted(
            room_id
            for room_id, candidate in room_bounds.items()
            if _bounds_overlap(bounds, candidate) is not None
        )
        if set(overlaps) != set(carved):
            raise ValidationError(
                f"common zone {zone_id} overlap/carve mismatch: overlaps={overlaps}, "
                f"carved_from={list(carved)}"
            )
        for other in zones.values():
            if other.kind == "common" and _bounds_overlap(bounds, other.bounds):
                raise ValidationError(f"common zones {zone_id} and {other.zone_id} overlap")
        zones[zone_id] = Zone(zone_id, bounds, "common", carved)
        evidence.append(
            {
                "id": zone_id,
                "bounds": list(bounds),
                "reason": reason,
                "carved_from": list(carved),
            }
        )
    return zones, evidence


def _wall_opening(room: Mapping[str, Any], opening_id: str) -> tuple[dict[str, Any], dict[str, Any]] | None:
    walls = room.get("walls", [])
    if not isinstance(walls, list):
        return None
    for wall in walls:
        if not isinstance(wall, dict):
            continue
        openings = wall.get("openings", [])
        if not isinstance(openings, list):
            continue
        for opening in openings:
            if isinstance(opening, dict) and str(opening.get("opening_id") or opening.get("id") or "") == opening_id:
                return wall, opening
    return None


def _opening_width(opening: Mapping[str, Any], label: str) -> float:
    width = _number(opening.get("width"), f"{label}.width")
    if width < MINIMUM_DOOR_WIDTH_M:
        raise ValidationError(
            f"{label} width {width:.3f} m is below {MINIMUM_DOOR_WIDTH_M:.1f} m"
        )
    opening_type = str(opening.get("opening_type") or opening.get("type") or "").split(".")[-1].lower()
    if opening_type != "door":
        raise ValidationError(f"{label} is not a DOOR opening")
    return width


def _shared_boundary(first: Bounds, second: Bounds) -> tuple[str, float, float, float] | None:
    """Return orientation, fixed coordinate, overlap start, overlap end."""

    y0, y1 = max(first[1], second[1]), min(first[3], second[3])
    if y1 - y0 > EPSILON:
        if abs(first[2] - second[0]) <= 0.25:
            return ("vertical", (first[2] + second[0]) / 2.0, y0, y1)
        if abs(second[2] - first[0]) <= 0.25:
            return ("vertical", (second[2] + first[0]) / 2.0, y0, y1)
    x0, x1 = max(first[0], second[0]), min(first[2], second[2])
    if x1 - x0 > EPSILON:
        if abs(first[3] - second[1]) <= 0.25:
            return ("horizontal", (first[3] + second[1]) / 2.0, x0, x1)
        if abs(second[3] - first[1]) <= 0.25:
            return ("horizontal", (second[3] + first[1]) / 2.0, x0, x1)
    return None


def _portal_center_from_wall(
    wall: Mapping[str, Any], opening: Mapping[str, Any]
) -> tuple[float, float] | None:
    start, end = wall.get("start_point"), wall.get("end_point")
    if start is None or end is None:
        return None
    sx, sy = _vector(start, 2, "wall.start_point")
    ex, ey = _vector(end, 2, "wall.end_point")
    length = math.hypot(ex - sx, ey - sy)
    if length <= EPSILON:
        raise ValidationError("door wall has zero length")
    offset = _number(
        opening.get("position_along_wall", opening.get("position_exact", 0.0)),
        "opening.position_along_wall",
    )
    width = _number(opening.get("width"), "opening.width")
    distance = offset + width / 2.0
    return (sx + (ex - sx) * distance / length, sy + (ey - sy) * distance / length)


def _require_center_on_shared_boundary(
    door_id: str,
    center: tuple[float, float],
    shared: tuple[str, float, float, float],
    width: float,
) -> None:
    orientation, fixed, start, end = shared
    normal = center[0] if orientation == "vertical" else center[1]
    along = center[1] if orientation == "vertical" else center[0]
    if abs(normal - fixed) > 0.25 + EPSILON:
        raise ValidationError(f"door {door_id} center is not on its shared boundary")
    if along - width / 2.0 < start - EPSILON or along + width / 2.0 > end + EPSILON:
        raise ValidationError(f"door {door_id} does not fit inside its shared boundary")


def _require_common_portal_boundary(
    portal_id: str,
    point: tuple[float, float],
    zone: Zone,
    orientation: str,
    width: float,
) -> None:
    if orientation == "vertical":
        on_boundary = min(abs(point[0] - zone.bounds[0]), abs(point[0] - zone.bounds[2])) <= 0.25
        fits = (
            point[1] - width / 2.0 >= zone.bounds[1] - EPSILON
            and point[1] + width / 2.0 <= zone.bounds[3] + EPSILON
        )
    else:
        on_boundary = min(abs(point[1] - zone.bounds[1]), abs(point[1] - zone.bounds[3])) <= 0.25
        fits = (
            point[0] - width / 2.0 >= zone.bounds[0] - EPSILON
            and point[0] + width / 2.0 <= zone.bounds[2] + EPSILON
        )
    if not on_boundary or not fits:
        raise ValidationError(
            f"common portal {portal_id} is not a width-preserving threshold on {zone.zone_id}"
        )


def _portal_contains(backing: Portal, claimed: Portal) -> bool:
    if backing.orientation != claimed.orientation:
        return False
    normal_axis = 0 if claimed.orientation == "vertical" else 1
    tangent_axis = 1 if claimed.orientation == "vertical" else 0
    return (
        abs(backing.center[normal_axis] - claimed.center[normal_axis]) <= 0.1
        and abs(backing.center[tangent_axis] - claimed.center[tangent_axis])
        + claimed.width_m / 2.0
        <= backing.width_m / 2.0 + 0.05
    )


def _portal_inside_zone(portal: Portal, zone: Zone) -> bool:
    if portal.orientation == "vertical":
        return (
            zone.bounds[0] - 0.1 <= portal.center[0] <= zone.bounds[2] + 0.1
            and portal.center[1] - portal.width_m / 2.0 >= zone.bounds[1] - 0.05
            and portal.center[1] + portal.width_m / 2.0 <= zone.bounds[3] + 0.05
        )
    return (
        zone.bounds[1] - 0.1 <= portal.center[1] <= zone.bounds[3] + 0.1
        and portal.center[0] - portal.width_m / 2.0 >= zone.bounds[0] - 0.05
        and portal.center[0] + portal.width_m / 2.0 <= zone.bounds[2] + 0.05
    )


def _common_zone_physical_backing(
    portals: Sequence[Portal], zones: Mapping[str, Zone]
) -> dict[str, Any]:
    real = [
        portal
        for portal in portals
        if portal.source in {"layout_door", "layout_open_connection"}
        and portal.zone_b is not None
    ]
    evidence: dict[str, Any] = {}
    for zone in sorted(
        (candidate for candidate in zones.values() if candidate.kind == "common"),
        key=lambda candidate: candidate.zone_id,
    ):
        carved = set(zone.carved_from)
        carved_graph: dict[str, set[str]] = {room_id: set() for room_id in carved}
        cross_openings: list[str] = []
        for portal in real:
            if (
                portal.zone_a in carved
                and portal.zone_b in carved
                and _portal_inside_zone(portal, zone)
            ):
                carved_graph[portal.zone_a].add(portal.zone_b)
                carved_graph[portal.zone_b].add(portal.zone_a)
                cross_openings.append(portal.portal_id)
        reached: set[str] = set()
        if carved:
            reached.add(sorted(carved)[0])
            queue = deque(reached)
            while queue:
                room_id = queue.popleft()
                for neighbor in carved_graph.get(room_id, set()):
                    if neighbor not in reached:
                        reached.add(neighbor)
                        queue.append(neighbor)
        disconnected = sorted(carved - reached)
        if disconnected:
            raise ValidationError(
                f"common zone {zone.zone_id} crosses room walls without real openings: {disconnected}"
            )
        declared = [portal for portal in portals if portal.zone_a == zone.zone_id and portal.source.startswith("explicit_common_zone")]
        backed: list[dict[str, Any]] = []
        for portal in declared:
            backing_id = portal.source.partition(":")[2]
            if backing_id:
                backing = next((candidate for candidate in real if candidate.portal_id == backing_id), None)
                if backing is None or not _portal_contains(backing, portal):
                    raise ValidationError(
                        f"common portal {portal.portal_id} is not contained by backing door {backing_id}"
                    )
                backed.append(
                    {
                        "portal_id": portal.portal_id,
                        "backing_connection_id": backing_id,
                        "target_zone_id": portal.zone_b,
                    }
                )
            elif portal.zone_b not in CORE_COMMON_ZONES:
                raise ValidationError(
                    f"functional-room common portal {portal.portal_id} lacks a real backing door"
                )
        if not declared:
            raise ValidationError(f"common zone {zone.zone_id} has no explicit circulation portals")
        evidence[zone.zone_id] = {
            "status": "pass",
            "carved_rooms": sorted(carved),
            "carved_rooms_reachable_through_real_openings": sorted(reached),
            "cross_room_opening_ids": sorted(cross_openings),
            "backed_core_portals": backed,
            "declared_portal_ids": sorted(portal.portal_id for portal in declared),
        }
    return evidence


def _door_portals(
    layout: Mapping[str, Any],
    rooms: Mapping[str, Mapping[str, Any]],
    zones: Mapping[str, Zone],
) -> tuple[list[Portal], str, list[dict[str, Any]], dict[str, Any]]:
    raw_doors = layout.get("doors")
    if not isinstance(raw_doors, list) or not all(isinstance(item, dict) for item in raw_doors):
        raise ValidationError("house_layout.doors must be a list of objects")
    portals: list[Portal] = []
    evidence: list[dict[str, Any]] = []
    entrance_id: str | None = None
    seen_ids: set[str] = set()
    for door in raw_doors:
        door_id = str(door.get("id") or "").strip()
        if not door_id or door_id in seen_ids:
            raise ValidationError(f"missing or duplicate door ID: {door_id!r}")
        seen_ids.add(door_id)
        room_a = str(door.get("room_a") or "").strip()
        room_b_raw = door.get("room_b")
        room_b = str(room_b_raw).strip() if room_b_raw is not None else None
        if room_a not in rooms or (room_b is not None and room_b not in rooms):
            raise ValidationError(f"door {door_id} has an unknown endpoint")
        width = _number(door.get("width"), f"door {door_id}.width")
        if width < MINIMUM_DOOR_WIDTH_M:
            raise ValidationError(
                f"door {door_id} width {width:.3f} m is below {MINIMUM_DOOR_WIDTH_M:.1f} m"
            )
        opening_a_record = _wall_opening(rooms[room_a], door_id)
        if opening_a_record is None:
            raise ValidationError(f"door {door_id} lacks a matching opening in {room_a}")
        wall_a, opening_a = opening_a_record
        opening_a_width = _opening_width(opening_a, f"{room_a}/{door_id}")
        if abs(opening_a_width - width) > 0.05:
            raise ValidationError(f"door {door_id} width disagrees with {room_a} opening")
        if room_b is not None:
            if str(door.get("door_type") or "").strip().lower() != "interior":
                raise ValidationError(f"two-room door {door_id} must have door_type=interior")
            opening_b_record = _wall_opening(rooms[room_b], door_id)
            if opening_b_record is None:
                raise ValidationError(f"door {door_id} lacks a matching opening in {room_b}")
            wall_b, opening_b = opening_b_record
            opening_b_width = _opening_width(opening_b, f"{room_b}/{door_id}")
            if abs(opening_b_width - width) > 0.05:
                raise ValidationError(f"door {door_id} width disagrees with {room_b} opening")
            shared = _shared_boundary(zones[room_a].bounds, zones[room_b].bounds)
            if shared is None:
                raise ValidationError(f"door {door_id} endpoints are not physically adjacent")
            center = _portal_center_from_wall(wall_a, opening_a)
            if center is None:
                orientation, fixed, start, end = shared
                segment = str(door.get("position_segment") or "center").lower()
                fraction = {"left": 0.25, "bottom": 0.25, "center": 0.5, "right": 0.75, "top": 0.75}.get(segment, 0.5)
                along = start + fraction * (end - start)
                center = (fixed, along) if orientation == "vertical" else (along, fixed)
            _require_center_on_shared_boundary(door_id, center, shared, width)
            center_b = _portal_center_from_wall(wall_b, opening_b)
            if center_b is not None and math.dist(center, center_b) > 0.05:
                raise ValidationError(f"door {door_id} openings disagree across its two walls")
            portals.append(
                Portal(
                    door_id,
                    room_a,
                    room_b,
                    width,
                    center,
                    shared[0],
                    "layout_door",
                )
            )
        else:
            if str(door.get("door_type") or "").strip().lower() != "exterior":
                raise ValidationError("the exterior entrance must have door_type=exterior")
            leaf_count = int(_number(door.get("leaf_count"), f"door {door_id}.leaf_count"))
            direction = str(wall_a.get("direction") or "").split(".")[-1].lower()
            center = _portal_center_from_wall(wall_a, opening_a)
            if center is None:
                bounds = zones[room_a].bounds
                center = ((bounds[0] + bounds[2]) / 2.0, bounds[1] if direction == "south" else bounds[3])
            portal_zone = room_a
            if door_id == "main_entrance":
                if entrance_id is not None:
                    raise ValidationError("the factory must have exactly one main entrance")
                if room_a != "finished_goods_storage" or direction != "south":
                    raise ValidationError("main entrance must be on the finished-goods south wall")
                if width < MINIMUM_ENTRANCE_WIDTH_M or leaf_count != 2:
                    raise ValidationError("main entrance must be at least 1.6 m with two leaves")
                entrance_zone = zones.get("entrance_transition")
                if entrance_zone is None or not _point_in_bounds(center, entrance_zone.bounds):
                    raise ValidationError("main entrance is not co-located with entrance_transition")
                portal_zone = "entrance_transition"
                entrance_id = door_id
            elif door_id == "loading_dock_door":
                if room_a != "finished_goods_storage" or direction != "east" or width < 3.0:
                    raise ValidationError("loading-dock door must be a >=3.0 m east roll-up threshold")
                dock_zone = zones.get("loading_dock")
                if dock_zone is None or not _point_in_bounds(center, dock_zone.bounds):
                    raise ValidationError("loading-dock door is not co-located with loading_dock")
                portal_zone = "loading_dock"
            else:
                raise ValidationError(f"unexpected factory exterior door: {door_id}")
            portals.append(
                Portal(
                    door_id,
                    portal_zone,
                    None,
                    width,
                    center,
                    "horizontal" if direction in {"north", "south"} else "vertical",
                    "exterior_factory_door",
                )
            )
        evidence.append(
            {
                "id": door_id,
                "zone_a": portals[-1].zone_a,
                "zone_b": portals[-1].zone_b,
                "width_m": width,
                "center": [round(center[0], 6), round(center[1], 6)],
                "orientation": portals[-1].orientation,
                "leaf_count": door.get("leaf_count"),
                "source": "layout_door",
            }
        )
    if entrance_id is None:
        raise ValidationError("missing exterior double entrance")

    # SceneSmith may represent a genuinely open-plan threshold on RoomSpec
    # instead of emitting a Door.  Treat the full shared boundary as the
    # portal, while still requiring physical adjacency and a >=0.9 m span.
    open_pairs: set[tuple[str, str]] = set()
    raw_room_specs = layout.get("rooms", [])
    if raw_room_specs is None:
        raw_room_specs = []
    if not isinstance(raw_room_specs, list):
        raise ValidationError("house_layout.rooms must be a list when supplied")
    door_pairs = {
        frozenset({portal.zone_a, portal.zone_b})
        for portal in portals
        if portal.zone_b is not None
    }
    for room_spec in raw_room_specs:
        if not isinstance(room_spec, dict):
            raise ValidationError("house_layout.rooms contains a malformed RoomSpec")
        first = str(room_spec.get("room_id") or room_spec.get("id") or "").strip()
        connections = room_spec.get("connections", {})
        if not isinstance(connections, dict):
            raise ValidationError(f"RoomSpec {first or '<missing>'}.connections must be a mapping")
        for other_raw, connection_type in connections.items():
            if str(connection_type).split(".")[-1].lower() != "open":
                continue
            other = str(other_raw).strip()
            if first not in rooms or other not in rooms or first == other:
                raise ValidationError(f"OPEN connection {first}<->{other} has an invalid endpoint")
            pair = tuple(sorted((first, other)))
            if pair in open_pairs or frozenset(pair) in door_pairs:
                continue
            open_pairs.add(pair)
            shared = _shared_boundary(zones[first].bounds, zones[other].bounds)
            if shared is None:
                raise ValidationError(f"OPEN connection {first}<->{other} is not physically adjacent")
            orientation, fixed, start, end = shared
            width = end - start
            if width < MINIMUM_DOOR_WIDTH_M:
                raise ValidationError(f"OPEN connection {first}<->{other} is narrower than 0.9 m")
            center = (fixed, (start + end) / 2.0) if orientation == "vertical" else ((start + end) / 2.0, fixed)
            portal_id = f"open_{pair[0]}_{pair[1]}"
            if portal_id in seen_ids:
                raise ValidationError(f"duplicate portal ID: {portal_id}")
            seen_ids.add(portal_id)
            portals.append(
                Portal(
                    portal_id,
                    first,
                    other,
                    width,
                    center,
                    orientation,
                    "layout_open_connection",
                )
            )
            evidence.append(
                {
                    "id": portal_id,
                    "zone_a": first,
                    "zone_b": other,
                    "width_m": width,
                    "center": [round(center[0], 6), round(center[1], 6)],
                    "orientation": orientation,
                    "leaf_count": None,
                    "source": "layout_open_connection",
                }
            )

    raw_common = layout.get("navigation_common_zones", []) or []
    for zone_record in raw_common:
        zone_id = str(zone_record.get("id"))
        connections = zone_record.get("connections")
        if not isinstance(connections, list) or not connections:
            raise ValidationError(f"common zone {zone_id} must have explicit connections")
        for index, connection in enumerate(connections):
            if not isinstance(connection, dict):
                raise ValidationError(f"common zone {zone_id} connection {index} is malformed")
            other = str(connection.get("to") or connection.get("room_id") or connection.get("zone_id") or "").strip()
            if other not in zones or other == zone_id:
                raise ValidationError(f"common zone {zone_id} connection has invalid endpoint {other!r}")
            portal_id = str(connection.get("id") or f"{zone_id}_to_{other}").strip()
            if portal_id in seen_ids:
                raise ValidationError(f"duplicate portal ID: {portal_id}")
            seen_ids.add(portal_id)
            backing_id = str(connection.get("backing_door_id") or "").strip()
            backing = next(
                (
                    portal
                    for portal in portals
                    if portal.portal_id == backing_id
                    and portal.source in {"layout_door", "layout_open_connection"}
                ),
                None,
            ) if backing_id else None
            if backing_id and backing is None:
                raise ValidationError(
                    f"common portal {portal_id} names missing real backing door {backing_id}"
                )
            width = (
                backing.width_m
                if backing is not None
                else _number(connection.get("width"), f"common portal {portal_id}.width")
            )
            if width < MINIMUM_DOOR_WIDTH_M:
                raise ValidationError(
                    f"common portal {portal_id} width {width:.3f} m is below 0.9 m"
                )
            point = (
                backing.center
                if backing is not None
                else _vector(connection.get("position"), 2, f"common portal {portal_id}.position")
            )
            orientation = (
                backing.orientation
                if backing is not None
                else str(connection.get("orientation") or "").strip().lower()
            )
            if orientation not in {"horizontal", "vertical"}:
                raise ValidationError(
                    f"common portal {portal_id} must declare horizontal/vertical orientation"
                )
            if backing is None:
                _require_common_portal_boundary(
                    portal_id, point, zones[zone_id], orientation, width
                )
            # A declared threshold must physically join its endpoints.  A room
            # listed in carved_from meets the foyer on an internal carve
            # boundary; every other pair must share a normal room boundary.
            if backing is not None:
                if other not in {backing.zone_a, backing.zone_b}:
                    raise ValidationError(
                        f"common portal {portal_id} target is not an endpoint of {backing_id}"
                    )
                if not _portal_inside_zone(backing, zones[zone_id]):
                    raise ValidationError(
                        f"backing door {backing_id} is not co-located with common zone {zone_id}"
                    )
            elif other in zones[zone_id].carved_from:
                if not _point_in_bounds(point, zones[other].bounds):
                    raise ValidationError(
                        f"common portal {portal_id} is outside carved room {other}"
                    )
                tangent = point[1] if orientation == "vertical" else point[0]
                tangent_start = zones[other].bounds[1] if orientation == "vertical" else zones[other].bounds[0]
                tangent_end = zones[other].bounds[3] if orientation == "vertical" else zones[other].bounds[2]
                if tangent - width / 2.0 < tangent_start - EPSILON or tangent + width / 2.0 > tangent_end + EPSILON:
                    raise ValidationError(
                        f"common portal {portal_id} does not fit carved room {other}"
                    )
            else:
                shared = _shared_boundary(zones[zone_id].bounds, zones[other].bounds)
                if shared is None or shared[0] != orientation:
                    raise ValidationError(
                        f"common portal {portal_id} endpoints do not share its declared boundary"
                    )
                _require_center_on_shared_boundary(portal_id, point, shared, width)
            portals.append(
                Portal(
                    portal_id,
                    zone_id,
                    other,
                    width,
                    point,
                    orientation,
                    f"explicit_common_zone:{backing_id}" if backing_id else "explicit_common_zone",
                )
            )
            evidence.append(
                {
                    "id": portal_id,
                    "zone_a": zone_id,
                    "zone_b": other,
                    "width_m": width,
                    "center": [round(point[0], 6), round(point[1], 6)],
                    "orientation": orientation,
                    "leaf_count": None,
                    "source": "explicit_common_zone",
                    "backing_door_id": backing_id or None,
                }
            )
    physical_backing = _common_zone_physical_backing(portals, zones)
    return portals, entrance_id, evidence, physical_backing


def _translation(obj: Mapping[str, Any], label: str) -> tuple[float, float, float]:
    transform = obj.get("transform")
    raw: Any = None
    if isinstance(transform, Mapping):
        raw = transform.get("translation", transform.get("position"))
    if raw is None:
        raw = obj.get("translation", obj.get("position"))
    return _vector(raw, 3, f"{label}.translation")  # type: ignore[return-value]


def _bbox(obj: Mapping[str, Any], label: str) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    minimum = obj.get("bbox_min", obj.get("bounding_box_min"))
    maximum = obj.get("bbox_max", obj.get("bounding_box_max"))
    for nested_key in ("bbox", "aabb", "bounding_box"):
        nested = obj.get(nested_key)
        if isinstance(nested, Mapping):
            minimum = minimum if minimum is not None else nested.get("min", nested.get("minimum"))
            maximum = maximum if maximum is not None else nested.get("max", nested.get("maximum"))
        elif (
            minimum is None
            and maximum is None
            and isinstance(nested, Sequence)
            and not isinstance(nested, (str, bytes))
        ):
            if len(nested) == 2 and all(isinstance(value, (list, tuple, dict)) for value in nested):
                minimum, maximum = nested[0], nested[1]
            elif len(nested) >= 6:
                minimum, maximum = nested[:3], nested[3:6]
    low = _vector(minimum, 3, f"{label}.bbox_min")
    high = _vector(maximum, 3, f"{label}.bbox_max")
    if any(high[index] <= low[index] for index in range(3)):
        raise ValidationError(f"{label} has a degenerate or inverted bounding box")
    return low, high  # type: ignore[return-value]


def _rotation_matrix(obj: Mapping[str, Any], label: str) -> tuple[tuple[float, float, float], ...]:
    transform = obj.get("transform")
    transform = transform if isinstance(transform, Mapping) else {}
    raw_matrix = transform.get("rotation_matrix", transform.get("rotation"))
    if isinstance(raw_matrix, Sequence) and len(raw_matrix) == 3 and all(
        isinstance(row, Sequence) and len(row) >= 3 for row in raw_matrix
    ):
        matrix = tuple(_vector(row, 3, f"{label}.rotation_matrix") for row in raw_matrix)
        row_norms = [math.sqrt(sum(value * value for value in row)) for row in matrix]
        row_dots = [
            sum(matrix[first][axis] * matrix[second][axis] for axis in range(3))
            for first, second in ((0, 1), (0, 2), (1, 2))
        ]
        determinant = (
            matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
            - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
            + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
        )
        if (
            any(abs(norm - 1.0) > 1e-3 for norm in row_norms)
            or any(abs(dot) > 1e-3 for dot in row_dots)
            or abs(determinant - 1.0) > 1e-3
        ):
            raise ValidationError(f"{label}.rotation_matrix is not a proper rotation")
        return matrix
    if raw_matrix is not None:
        try:
            roll, pitch, yaw = _vector(raw_matrix, 3, f"{label}.rotation_euler")
        except ValidationError as exc:
            raise ValidationError(f"{label}.rotation has an unsupported format") from exc
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        return (
            (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
            (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
            (-sp, cp * sr, cp * cr),
        )
    quaternion = transform.get("rotation_wxyz", obj.get("rotation_wxyz"))
    order = "wxyz"
    if quaternion is None:
        quaternion = transform.get("rotation_xyzw", obj.get("rotation_xyzw"))
        order = "xyzw"
    if quaternion is None:
        return ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    if isinstance(quaternion, Mapping):
        axes = ("w", "x", "y", "z") if order == "wxyz" else ("x", "y", "z", "w")
        values = tuple(
            _number(quaternion.get(axis), f"{label}.rotation_{order}.{axis}")
            for axis in axes
        )
    else:
        values = _vector(quaternion, 4, f"{label}.rotation_{order}")
    w, x, y, z = values if order == "wxyz" else (values[3], values[0], values[1], values[2])
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= EPSILON:
        raise ValidationError(f"{label} has a zero quaternion")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )


def _objects(state: Mapping[str, Any], room_id: str) -> list[tuple[str, dict[str, Any]]]:
    raw = state.get("objects")
    if isinstance(raw, dict):
        items = list(raw.items())
    elif isinstance(raw, list):
        items = [(str(item.get("object_id", index)), item) for index, item in enumerate(raw) if isinstance(item, dict)]
    else:
        raise ValidationError(f"{room_id} state objects must be a mapping or list")
    result = []
    for object_id, obj in items:
        if not isinstance(obj, dict):
            raise ValidationError(f"{room_id}/{object_id} object record is malformed")
        result.append((str(object_id), obj))
    return result


def _obstacles(
    states: Mapping[str, Mapping[str, Any]],
    room_bounds: Mapping[str, Bounds],
    radius: float,
    humanoid_height: float,
) -> tuple[list[Obstacle], list[dict[str, Any]]]:
    obstacles: list[Obstacle] = []
    ignored: list[dict[str, Any]] = []
    for room_id in EXPECTED_ROOM_IDS:
        center_offset = (
            (room_bounds[room_id][0] + room_bounds[room_id][2]) / 2.0,
            (room_bounds[room_id][1] + room_bounds[room_id][3]) / 2.0,
        )
        for object_id, obj in _objects(states[room_id], room_id):
            object_type = str(obj.get("object_type") or "").split(".")[-1].lower()
            if object_type in STRUCTURAL_TYPES:
                ignored.append({"room_id": room_id, "object_id": object_id, "reason": "structural"})
                continue
            label = f"{room_id}/{object_id}"
            translation = _translation(obj, label)
            low, high = _bbox(obj, label)
            rotation = _rotation_matrix(obj, label)
            local_center = tuple((low[index] + high[index]) / 2.0 for index in range(3))
            half = tuple((high[index] - low[index]) / 2.0 for index in range(3))
            rotated_center = tuple(
                translation[row] + sum(rotation[row][column] * local_center[column] for column in range(3))
                for row in range(3)
            )
            world_center = (
                rotated_center[0] + center_offset[0],
                rotated_center[1] + center_offset[1],
                rotated_center[2],
            )
            extent = tuple(
                sum(abs(rotation[row][column]) * half[column] for column in range(3))
                for row in range(3)
            )
            z_range = (world_center[2] - extent[2], world_center[2] + extent[2])
            if z_range[0] >= humanoid_height or z_range[1] <= 0.02:
                ignored.append({"room_id": room_id, "object_id": object_id, "reason": "vertical_clearance"})
                continue
            raw = (
                world_center[0] - extent[0],
                world_center[1] - extent[1],
                world_center[0] + extent[0],
                world_center[1] + extent[1],
            )
            inflated = (raw[0] - radius, raw[1] - radius, raw[2] + radius, raw[3] + radius)
            obstacles.append(
                Obstacle(room_id, object_id, raw, inflated, z_range, "exact_rotation_aabb")
            )
    return obstacles, ignored


def _final_state_paths(scene_dir: Path) -> dict[str, Path]:
    return {
        room_id: scene_dir
        / f"room_{room_id}"
        / "scene_states"
        / "final_scene"
        / "scene_state.json"
        for room_id in EXPECTED_ROOM_IDS
    }


def _load_states(
    scene_dir: Path,
    combined: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    combined_rooms = combined.get("rooms")
    if not isinstance(combined_rooms, dict) or set(combined_rooms) != EXPECTED_ROOM_SET:
        raise ValidationError("combined house_state rooms must be exactly the 14 factory rooms")
    states: dict[str, dict[str, Any]] = {}
    evidence: dict[str, dict[str, Any]] = {}
    for room_id, path in _final_state_paths(scene_dir).items():
        if not path.is_file():
            raise ValidationError(f"missing final room state: {path}")
        final = _read_object(path)
        assembled = combined_rooms.get(room_id)
        if not isinstance(assembled, dict):
            raise ValidationError(f"combined state for {room_id} is malformed")
        final_objects = final.get("objects")
        assembled_objects = assembled.get("objects")
        if _canonical_hash(final_objects) != _canonical_hash(assembled_objects):
            raise ValidationError(f"combined object map does not match final state for {room_id}")
        states[room_id] = assembled
        evidence[room_id] = {
            "path": str(path.resolve()),
            "sha256": _file_hash(path),
            "objects_sha256": _canonical_hash(final_objects),
            "object_count": len(_objects(final, room_id)),
        }
    return states, evidence


class Grid:
    def __init__(self, zones: Mapping[str, Zone], resolution: float):
        self.resolution = resolution
        self.min_x = math.floor(min(zone.bounds[0] for zone in zones.values()) / resolution) * resolution
        self.min_y = math.floor(min(zone.bounds[1] for zone in zones.values()) / resolution) * resolution
        self.max_x = math.ceil(max(zone.bounds[2] for zone in zones.values()) / resolution) * resolution
        self.max_y = math.ceil(max(zone.bounds[3] for zone in zones.values()) / resolution) * resolution
        self.width = int(round((self.max_x - self.min_x) / resolution))
        self.height = int(round((self.max_y - self.min_y) / resolution))
        if self.width <= 0 or self.height <= 0 or self.width * self.height > 2_000_000:
            raise ValidationError("navigation grid dimensions are unsafe or invalid")

    def center(self, cell: Cell) -> tuple[float, float]:
        return (
            self.min_x + (cell[0] + 0.5) * self.resolution,
            self.min_y + (cell[1] + 0.5) * self.resolution,
        )

    def cells_for_bounds(self, bounds: Bounds) -> Iterable[Cell]:
        ix0 = max(0, int(math.floor((bounds[0] - self.min_x) / self.resolution)))
        iy0 = max(0, int(math.floor((bounds[1] - self.min_y) / self.resolution)))
        ix1 = min(self.width - 1, int(math.ceil((bounds[2] - self.min_x) / self.resolution)))
        iy1 = min(self.height - 1, int(math.ceil((bounds[3] - self.min_y) / self.resolution)))
        for ix in range(ix0, ix1 + 1):
            for iy in range(iy0, iy1 + 1):
                yield (ix, iy)


def _blocked(point: tuple[float, float], bounds: Iterable[Bounds]) -> bool:
    return any(_point_in_bounds(point, candidate) for candidate in bounds)


def _free_cells(
    grid: Grid,
    zones: Mapping[str, Zone],
    common_zone_ids: frozenset[str],
    obstacles: Sequence[Obstacle],
    radius: float,
) -> dict[str, set[Cell]]:
    obstacle_bounds = [obstacle.inflated_bounds for obstacle in obstacles]
    carved_bounds: dict[str, list[Bounds]] = {room_id: [] for room_id in EXPECTED_ROOM_IDS}
    for zone_id in common_zone_ids - CORE_COMMON_ZONES:
        zone = zones[zone_id]
        for room_id in zone.carved_from:
            carved_bounds[room_id].append(
                (
                    zone.bounds[0] - radius,
                    zone.bounds[1] - radius,
                    zone.bounds[2] + radius,
                    zone.bounds[3] + radius,
                )
            )
    result: dict[str, set[Cell]] = {}
    for zone_id, zone in zones.items():
        candidates: set[Cell] = set()
        inner = (
            zone.bounds[0] + radius,
            zone.bounds[1] + radius,
            zone.bounds[2] - radius,
            zone.bounds[3] - radius,
        )
        if inner[2] <= inner[0] or inner[3] <= inner[1]:
            raise ValidationError(f"zone {zone_id} is too narrow for the humanoid radius")
        exclusions = [*obstacle_bounds, *carved_bounds.get(zone_id, [])]
        for cell in grid.cells_for_bounds(inner):
            point = grid.center(cell)
            if _point_in_bounds(point, inner) and not _blocked(point, exclusions):
                candidates.add(cell)
        if not candidates:
            raise ValidationError(f"zone {zone_id} has no humanoid free-space cells")
        result[zone_id] = candidates
    return result


def _nearest_portal_cell(
    grid: Grid,
    free: set[Cell],
    point: tuple[float, float],
    width: float,
    radius: float,
    orientation: str,
) -> Cell | None:
    half_clear_span = width / 2.0 - radius
    if half_clear_span < -EPSILON:
        return None
    normal_limit = radius + 2.5 * grid.resolution
    tangent_limit = max(0.0, half_clear_span) + grid.resolution / 2.0
    tangent_axis = 1 if orientation == "vertical" else 0
    normal_axis = 0 if orientation == "vertical" else 1
    candidates = [
        cell
        for cell in free
        if abs(grid.center(cell)[normal_axis] - point[normal_axis])
        <= normal_limit + EPSILON
        and abs(grid.center(cell)[tangent_axis] - point[tangent_axis])
        <= tangent_limit + EPSILON
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda cell: (math.dist(grid.center(cell), point), cell))


def _portal_transitions(
    grid: Grid,
    portals: Sequence[Portal],
    free: Mapping[str, set[Cell]],
    radius: float,
    entrance_portal_id: str,
) -> tuple[dict[Node, list[Node]], dict[str, dict[str, Any]], Node]:
    transitions: dict[Node, list[Node]] = {}
    evidence: dict[str, dict[str, Any]] = {}
    entrance_node: Node | None = None
    for portal in portals:
        first = _nearest_portal_cell(
            grid,
            free[portal.zone_a],
            portal.center,
            portal.width_m,
            radius,
            portal.orientation,
        )
        if first is None:
            raise ValidationError(f"portal {portal.portal_id} is blocked on {portal.zone_a} side")
        second: Cell | None = None
        if portal.zone_b is not None:
            second = _nearest_portal_cell(
                grid,
                free[portal.zone_b],
                portal.center,
                portal.width_m,
                radius,
                portal.orientation,
            )
            if second is None:
                raise ValidationError(f"portal {portal.portal_id} is blocked on {portal.zone_b} side")
            first_node, second_node = (portal.zone_a, first), (portal.zone_b, second)
            transitions.setdefault(first_node, []).append(second_node)
            transitions.setdefault(second_node, []).append(first_node)
        elif portal.portal_id == entrance_portal_id:
            entrance_node = (portal.zone_a, first)
        evidence[portal.portal_id] = {
            "zone_a_cell": list(first),
            "zone_b_cell": list(second) if second is not None else None,
            "effective_clear_width_m": round(portal.width_m - 2.0 * radius, 6),
            "status": "pass",
        }
    if entrance_node is None:
        raise ValidationError(f"could not locate exterior grid cell for {entrance_portal_id}")
    return transitions, evidence, entrance_node


def _turning_candidates(
    grid: Grid,
    zone: Zone,
    free: set[Cell],
    raw_obstacles: Sequence[Bounds],
    carve_bounds: Sequence[Bounds],
    turning_radius: float,
) -> list[Cell]:
    candidates = []
    for cell in free:
        point = grid.center(cell)
        if not _point_in_bounds(point, zone.bounds, turning_radius):
            continue
        turning_square = (
            point[0] - turning_radius,
            point[1] - turning_radius,
            point[0] + turning_radius,
            point[1] + turning_radius,
        )
        if any(_bounds_overlap(turning_square, bounds) for bounds in (*raw_obstacles, *carve_bounds)):
            continue
        candidates.append(cell)
    center = ((zone.bounds[0] + zone.bounds[2]) / 2.0, (zone.bounds[1] + zone.bounds[3]) / 2.0)
    return sorted(candidates, key=lambda cell: (math.dist(grid.center(cell), center), cell))


def _bfs(
    start: Node,
    goals: set[Node],
    allowed_zones: frozenset[str],
    free: Mapping[str, set[Cell]],
    transitions: Mapping[Node, Sequence[Node]],
) -> list[Node] | None:
    if start[0] not in allowed_zones or start[1] not in free[start[0]]:
        return None
    queue = deque([start])
    parent: dict[Node, Node | None] = {start: None}
    while queue:
        node = queue.popleft()
        if node in goals:
            path = []
            cursor: Node | None = node
            while cursor is not None:
                path.append(cursor)
                cursor = parent[cursor]
            return list(reversed(path))
        zone_id, (ix, iy) = node
        neighbors = [
            (zone_id, (ix + 1, iy)),
            (zone_id, (ix - 1, iy)),
            (zone_id, (ix, iy + 1)),
            (zone_id, (ix, iy - 1)),
            *transitions.get(node, []),
        ]
        for neighbor in neighbors:
            if neighbor[0] not in allowed_zones or neighbor in parent:
                continue
            if neighbor[1] not in free[neighbor[0]]:
                continue
            parent[neighbor] = node
            queue.append(neighbor)
    return None


def _compress_waypoints(grid: Grid, path: Sequence[Node]) -> list[dict[str, Any]]:
    if not path:
        return []
    keep = [0]
    previous_direction: tuple[Any, ...] | None = None
    for index in range(1, len(path)):
        previous, current = path[index - 1], path[index]
        direction = (
            current[0] if current[0] != previous[0] else None,
            current[1][0] - previous[1][0],
            current[1][1] - previous[1][1],
        )
        if previous_direction is not None and direction != previous_direction:
            keep.append(index - 1)
        previous_direction = direction
    keep.append(len(path) - 1)
    return [
        {
            "zone": path[index][0],
            "cell": list(path[index][1]),
            "point_m": [round(value, 6) for value in grid.center(path[index][1])],
        }
        for index in sorted(set(keep))
    ]


def _topology_checks(
    portals: Sequence[Portal], common_zone_ids: frozenset[str]
) -> tuple[list[str], dict[str, Any]]:
    issues: list[str] = []
    graph: dict[str, set[str]] = {zone: set() for zone in common_zone_ids}
    direct: dict[str, list[str]] = {room_id: [] for room_id in EXPECTED_ROOM_IDS}
    for portal in portals:
        if portal.zone_b is None:
            continue
        if portal.zone_a in common_zone_ids and portal.zone_b in common_zone_ids:
            graph[portal.zone_a].add(portal.zone_b)
            graph[portal.zone_b].add(portal.zone_a)
        if portal.zone_a in common_zone_ids and portal.zone_b in EXPECTED_ROOM_SET:
            direct[portal.zone_b].append(portal.portal_id)
        if portal.zone_b in common_zone_ids and portal.zone_a in EXPECTED_ROOM_SET:
            direct[portal.zone_a].append(portal.portal_id)
    reached = {"entrance_transition"}
    queue = deque(["entrance_transition"])
    while queue:
        zone = queue.popleft()
        for neighbor in graph.get(zone, set()):
            if neighbor not in reached:
                reached.add(neighbor)
                queue.append(neighbor)
    disconnected_common = sorted(common_zone_ids - reached)
    if disconnected_common:
        issues.append(f"common circulation zones are disconnected from the core: {disconnected_common}")
    for room_id in EXPECTED_ROOM_IDS:
        if room_id in common_zone_ids:
            continue
        if not direct[room_id]:
            issues.append(
                f"{room_id} lacks a direct threshold to explicit factory circulation"
            )
    return issues, {
        "common_zone_ids": sorted(common_zone_ids),
        "common_zones_reachable_from_core": sorted(reached),
        "direct_common_portals_by_room": {key: sorted(value) for key, value in direct.items()},
        "forbidden_transit_policy": (
            "Each target route may use only explicit hash-bound circulation common zones "
            "and the target room; no other functional factory zone."
        ),
    }


def evaluate(
    scene_dir: Path,
    *,
    humanoid_radius_m: float = DEFAULT_HUMANOID_RADIUS_M,
    grid_resolution_m: float = DEFAULT_GRID_RESOLUTION_M,
    turning_diameter_m: float = DEFAULT_TURNING_DIAMETER_M,
    humanoid_height_m: float = DEFAULT_HUMANOID_HEIGHT_M,
    forklift_radius_m: float = DEFAULT_FORKLIFT_RADIUS_M,
) -> dict[str, Any]:
    scene_dir = scene_dir.resolve()
    for value, name in (
        (humanoid_radius_m, "humanoid_radius_m"),
        (grid_resolution_m, "grid_resolution_m"),
        (turning_diameter_m, "turning_diameter_m"),
        (humanoid_height_m, "humanoid_height_m"),
        (forklift_radius_m, "forklift_radius_m"),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValidationError(f"{name} must be a positive finite number")
    if grid_resolution_m > humanoid_radius_m:
        raise ValidationError("grid_resolution_m must not exceed humanoid_radius_m")
    if turning_diameter_m < 2.0 * humanoid_radius_m:
        raise ValidationError("turning diameter cannot be smaller than the humanoid")

    layout_path = scene_dir / "house_layout.json"
    house_state_path = scene_dir / "combined_house" / "house_state.json"
    if not layout_path.is_file() or not house_state_path.is_file():
        raise ValidationError("scene is missing house_layout.json or combined_house/house_state.json")
    layout = _read_object(layout_path)
    combined = _read_object(house_state_path)
    rooms, room_bounds = _rooms(layout)
    zones, common_zone_evidence = _common_zones(layout, room_bounds)
    common_zone_ids = frozenset(
        zone.zone_id for zone in zones.values() if zone.kind == "common"
    )
    if not CORE_COMMON_ZONES.issubset(common_zone_ids):
        raise ValidationError(
            f"missing required factory common zones: {sorted(CORE_COMMON_ZONES - common_zone_ids)}"
        )
    states, final_state_evidence = _load_states(scene_dir, combined)

    combined_layout = combined.get("layout")
    if not isinstance(combined_layout, dict):
        raise ValidationError("combined house_state is missing its layout object")
    _combined_rooms, combined_bounds = _rooms(combined_layout)
    if _canonical_hash(combined_bounds) != _canonical_hash(room_bounds):
        raise ValidationError("combined house layout bounds do not match house_layout.json")

    (
        portals,
        entrance_id,
        portal_evidence,
        common_zone_physical_backing,
    ) = _door_portals(layout, rooms, zones)
    consumed_real_portal_ids = {
        str(portal_id)
        for backing in common_zone_physical_backing.values()
        for portal_id in backing.get("cross_room_opening_ids", [])
    } | {
        str(record.get("backing_connection_id"))
        for backing in common_zone_physical_backing.values()
        for record in backing.get("backed_core_portals", [])
    }
    routing_portals = [
        portal
        for portal in portals
        if portal.portal_id not in consumed_real_portal_ids
    ]
    topology_issues, topology = _topology_checks(
        routing_portals, common_zone_ids
    )
    obstacles, ignored_objects = _obstacles(
        states, room_bounds, humanoid_radius_m, humanoid_height_m
    )
    grid = Grid(zones, grid_resolution_m)
    free = _free_cells(grid, zones, common_zone_ids, obstacles, humanoid_radius_m)
    transitions, transition_evidence, entrance_node = _portal_transitions(
        grid, routing_portals, free, humanoid_radius_m, "main_entrance"
    )

    turning_radius = turning_diameter_m / 2.0
    raw_obstacles = [obstacle.raw_bounds for obstacle in obstacles]
    turning: dict[str, dict[str, Any]] = {}
    turning_goals: dict[str, set[Node]] = {}
    for room_id in sorted(TURNING_ROOM_IDS):
        carve_bounds = [
            zone.bounds
            for zone in zones.values()
            if zone.kind == "common" and room_id in zone.carved_from
        ]
        candidates = _turning_candidates(
            grid,
            zones[room_id],
            free[room_id],
            raw_obstacles,
            carve_bounds,
            turning_radius,
        )
        turning_goals[room_id] = {(room_id, cell) for cell in candidates}
        turning[room_id] = {
            "status": "pass" if candidates else "fail",
            "candidate_count": len(candidates),
            "selected_cell": list(candidates[0]) if candidates else None,
            "selected_point_m": (
                [round(value, 6) for value in grid.center(candidates[0])]
                if candidates
                else None
            ),
            "clear_area_model": "1.5 m axis-aligned square (conservative for a 1.5 m turning circle)",
        }

    routes: dict[str, dict[str, Any]] = {}
    route_issues: list[str] = []
    for target in EXPECTED_ROOM_IDS:
        allowed = frozenset({*common_zone_ids, target})
        if target in turning_goals:
            goals = turning_goals[target]
        else:
            target_center = (
                (zones[target].bounds[0] + zones[target].bounds[2]) / 2.0,
                (zones[target].bounds[1] + zones[target].bounds[3]) / 2.0,
            )
            ordered = sorted(
                free[target], key=lambda cell: (math.dist(grid.center(cell), target_center), cell)
            )
            goals = {(target, cell) for cell in ordered[: max(1, min(25, len(ordered)))]}
        path = _bfs(entrance_node, goals, allowed, free, transitions) if goals else None
        if path is None:
            route_issues.append(f"no compliant humanoid route from entrance to {target}")
            routes[target] = {
                "status": "fail",
                "allowed_zones": sorted(allowed),
                "cells": [],
                "waypoints": [],
                "route_sha256": None,
            }
            continue
        cells = [{"zone": zone, "cell": list(cell)} for zone, cell in path]
        waypoints = _compress_waypoints(grid, path)
        route_length = sum(
            math.dist(grid.center(path[index - 1][1]), grid.center(path[index][1]))
            for index in range(1, len(path))
        )
        routes[target] = {
            "status": "pass",
            "allowed_zones": sorted(allowed),
            "traversed_zones": list(dict.fromkeys(zone for zone, _cell in path)),
            "cells": cells,
            "waypoints": waypoints,
            "route_length_m": round(route_length, 6),
            "route_sha256": _canonical_hash({"cells": cells, "waypoints": waypoints}),
        }

    forklift_common_ids = frozenset(
        common_zone_ids - {"entrance_transition", "toilet_foyer"}
    )
    forklift_obstacles, forklift_ignored = _obstacles(
        states, room_bounds, forklift_radius_m, humanoid_height_m
    )
    forklift_free = _free_cells(
        grid, zones, common_zone_ids, forklift_obstacles, forklift_radius_m
    )
    forklift_portals = [
        portal
        for portal in routing_portals
        if portal.width_m + EPSILON >= 2.0 * forklift_radius_m
        and (
            portal.zone_b is None
            or portal.zone_a in forklift_common_ids
            or portal.zone_b in forklift_common_ids
        )
    ]
    forklift_transitions, forklift_portal_evidence, loading_node = _portal_transitions(
        grid,
        forklift_portals,
        forklift_free,
        forklift_radius_m,
        "loading_dock_door",
    )
    forklift_routes: dict[str, dict[str, Any]] = {}
    forklift_issues: list[str] = []
    for target in sorted(FORKLIFT_TARGETS):
        allowed = frozenset({*forklift_common_ids, target})
        center = (
            (zones[target].bounds[0] + zones[target].bounds[2]) / 2.0,
            (zones[target].bounds[1] + zones[target].bounds[3]) / 2.0,
        )
        candidates = sorted(
            forklift_free[target],
            key=lambda cell: (math.dist(grid.center(cell), center), cell),
        )
        goals = {(target, cell) for cell in candidates[: max(1, min(25, len(candidates)))]}
        path = _bfs(
            loading_node,
            goals,
            allowed,
            forklift_free,
            forklift_transitions,
        ) if goals else None
        if path is None:
            forklift_issues.append(
                f"no obstacle-inflated >= {2.0 * forklift_radius_m:.1f} m forklift route from loading dock to {target}"
            )
            forklift_routes[target] = {
                "status": "fail",
                "allowed_zones": sorted(allowed),
                "cells": [],
                "waypoints": [],
                "route_sha256": None,
            }
            continue
        cells = [{"zone": zone, "cell": list(cell)} for zone, cell in path]
        waypoints = _compress_waypoints(grid, path)
        forklift_routes[target] = {
            "status": "pass",
            "minimum_clear_width_m": round(2.0 * forklift_radius_m, 6),
            "allowed_zones": sorted(allowed),
            "traversed_zones": list(dict.fromkeys(zone for zone, _ in path)),
            "cells": cells,
            "waypoints": waypoints,
            "route_sha256": _canonical_hash({"cells": cells, "waypoints": waypoints}),
        }
    prohibited_forklift_zones = {"entrance_transition", "toilet_foyer"}
    separation_violations = sorted(
        f"{target}:{zone}"
        for target, route in forklift_routes.items()
        for zone in route.get("traversed_zones", [])
        if zone in prohibited_forklift_zones
    )
    if separation_violations:
        forklift_issues.append(
            f"forklift routes enter pedestrian-only zones: {separation_violations}"
        )

    turning_issues = [
        f"{room_id} lacks a verified {turning_diameter_m:.1f} m turning area"
        for room_id, record in turning.items()
        if record["status"] != "pass"
    ]
    issues = [*topology_issues, *turning_issues, *route_issues, *forklift_issues]
    input_evidence = {
        "house_layout": {"path": str(layout_path.resolve()), "sha256": _file_hash(layout_path)},
        "house_state": {"path": str(house_state_path.resolve()), "sha256": _file_hash(house_state_path)},
        "final_room_states": final_state_evidence,
    }
    obstacle_evidence = [
        {
            "room_id": obstacle.room_id,
            "object_id": obstacle.object_id,
            "raw_bounds": [round(value, 6) for value in obstacle.raw_bounds],
            "inflated_bounds": [round(value, 6) for value in obstacle.inflated_bounds],
            "z_range": [round(value, 6) for value in obstacle.z_range],
            "rotation_source": obstacle.rotation_source,
        }
        for obstacle in obstacles
    ]
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "algorithm": ALGORITHM,
        "status": "pass" if not issues else "fail",
        "parameters": {
            "humanoid_radius_m": humanoid_radius_m,
            "humanoid_height_m": humanoid_height_m,
            "grid_resolution_m": grid_resolution_m,
            "minimum_door_width_m": MINIMUM_DOOR_WIDTH_M,
            "minimum_entrance_width_m": MINIMUM_ENTRANCE_WIDTH_M,
            "turning_diameter_m": turning_diameter_m,
            "forklift_radius_m": forklift_radius_m,
            "minimum_forklift_clear_width_m": 2.0 * forklift_radius_m,
        },
        "inputs": input_evidence,
        "topology": {
            **topology,
            "common_zones": common_zone_evidence,
            "entrance_portal_id": entrance_id,
            "portals": portal_evidence,
            "common_zone_physical_backing": common_zone_physical_backing,
            "real_portals_consumed_by_common_zones": sorted(
                consumed_real_portal_ids
            ),
            "portal_grid_evidence": transition_evidence,
        },
        "occupancy": {
            "coordinate_convention": "room-local object pose + placed-room center -> house coordinates",
            "grid_origin_m": [grid.min_x, grid.min_y],
            "grid_dimensions": [grid.width, grid.height],
            "free_cell_counts": {zone_id: len(cells) for zone_id, cells in sorted(free.items())},
            "obstacle_count": len(obstacle_evidence),
            "obstacles": obstacle_evidence,
            "obstacles_sha256": _canonical_hash(obstacle_evidence),
            "ignored_objects": ignored_objects,
        },
        "turning_areas": turning,
        "routes": routes,
        "routes_sha256": _canonical_hash(routes),
        "forklift_navigation": {
            "start_zone": "loading_dock",
            "target_rooms": sorted(FORKLIFT_TARGETS),
            "pedestrian_only_zones": sorted(prohibited_forklift_zones),
            "pedestrian_forklift_separable": not separation_violations,
            "portal_grid_evidence": forklift_portal_evidence,
            "obstacle_count": len(forklift_obstacles),
            "ignored_objects": forklift_ignored,
            "routes": forklift_routes,
            "routes_sha256": _canonical_hash(forklift_routes),
        },
        "critical_issues": issues,
    }
    result["attestation"] = {
        "algorithm": "sha256(canonical JSON excluding attestation)",
        "payload_sha256": _canonical_hash(result),
        "self_attested_status": result["status"],
        "all_inputs_hashed": True,
        "all_routes_hashed": all(
            route.get("route_sha256") for route in routes.values()
        ) and all(route.get("route_sha256") for route in forklift_routes.values()),
    }
    return result


def verify_output(
    output: Path,
    scene_dir: Path | None = None,
    *,
    verification_output: Path | None = None,
) -> dict[str, Any]:
    output = output.resolve()
    if (
        verification_output is not None
        and verification_output.resolve() == output
    ):
        raise ValidationError(
            "verification output must differ from the source navigation-gate output"
        )
    source_sha256 = _file_hash(output)
    source_size_bytes = output.stat().st_size
    saved = _read_object(output)
    attestation = saved.get("attestation")
    if not isinstance(attestation, dict):
        raise ValidationError("navigation evidence has no attestation")
    unsigned = dict(saved)
    unsigned.pop("attestation", None)
    if attestation.get("payload_sha256") != _canonical_hash(unsigned):
        raise ValidationError("navigation evidence self-attestation is invalid")
    if saved.get("schema_version") != SCHEMA_VERSION or saved.get("algorithm") != ALGORITHM:
        raise ValidationError("navigation evidence schema/algorithm is unsupported")
    parameters = saved.get("parameters")
    if not isinstance(parameters, dict):
        raise ValidationError("navigation evidence parameters are malformed")
    if scene_dir is None:
        layout_record = saved.get("inputs", {}).get("house_layout", {})
        layout_path = Path(str(layout_record.get("path", ""))).resolve()
        if not layout_path.name:
            raise ValidationError("navigation evidence does not bind house_layout.json")
        scene_dir = layout_path.parent
    recomputed = evaluate(
        scene_dir,
        humanoid_radius_m=_number(parameters.get("humanoid_radius_m"), "saved humanoid_radius_m"),
        humanoid_height_m=_number(parameters.get("humanoid_height_m"), "saved humanoid_height_m"),
        grid_resolution_m=_number(parameters.get("grid_resolution_m"), "saved grid_resolution_m"),
        turning_diameter_m=_number(parameters.get("turning_diameter_m"), "saved turning_diameter_m"),
        forklift_radius_m=_number(parameters.get("forklift_radius_m"), "saved forklift_radius_m"),
    )
    if _canonical_hash(recomputed) != _canonical_hash(saved):
        raise ValidationError("navigation evidence is stale or its inputs/routes changed")
    if saved.get("status") != "pass":
        raise ValidationError("saved navigation evidence is not passing")
    recomputed_attestation = recomputed.get("attestation")
    result: dict[str, Any] = {
        "verification_schema_id": VERIFICATION_SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "mode": "verify-only",
        "scene_dir": str(scene_dir.resolve()),
        "verified_output": str(output),
        "source_gate": {
            "path": str(output),
            "sha256": source_sha256,
            "size_bytes": source_size_bytes,
            "payload_sha256": attestation["payload_sha256"],
            "routes_sha256": saved.get("routes_sha256"),
        },
        "payload_sha256": attestation["payload_sha256"],
        "routes_sha256": saved.get("routes_sha256"),
        "recomputed_gate": {
            "result": recomputed,
            "status": recomputed.get("status"),
            "payload_sha256": (
                recomputed_attestation.get("payload_sha256")
                if isinstance(recomputed_attestation, dict)
                else None
            ),
            "routes_sha256": recomputed.get("routes_sha256"),
            "attestation": recomputed_attestation,
            "result_sha256": _canonical_hash(recomputed),
            "route_count": (
                len(recomputed.get("routes", {}))
                if isinstance(recomputed.get("routes"), dict)
                else 0
            ),
        },
        "navigation_recomputed": True,
        "critical_issues": [],
        "message": "navigation evidence and every input/route hash recomputed exactly",
    }
    result["attestation"] = {
        "algorithm": "sha256(canonical JSON excluding attestation)",
        "payload_sha256": _canonical_hash(result),
        "self_attested_status": "pass",
    }
    if verification_output is not None:
        _write_json(verification_output.resolve(), result)
    return result


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--verification-output",
        type=Path,
        help=(
            "Atomically save the hash-bound repeat-execution receipt; valid only "
            "with --verify-only."
        ),
    )
    parser.add_argument("--humanoid-radius", type=float, default=DEFAULT_HUMANOID_RADIUS_M)
    parser.add_argument("--grid-resolution", type=float, default=DEFAULT_GRID_RESOLUTION_M)
    parser.add_argument("--turning-diameter", type=float, default=DEFAULT_TURNING_DIAMETER_M)
    parser.add_argument("--humanoid-height", type=float, default=DEFAULT_HUMANOID_HEIGHT_M)
    parser.add_argument("--forklift-radius", type=float, default=DEFAULT_FORKLIFT_RADIUS_M)
    args = parser.parse_args(argv)
    if args.verification_output is not None and not args.verify_only:
        parser.error("--verification-output requires --verify-only")
    try:
        if args.verify_only:
            result = verify_output(
                args.output,
                args.scene_dir.resolve(),
                verification_output=args.verification_output,
            )
        else:
            result = evaluate(
                args.scene_dir,
                humanoid_radius_m=args.humanoid_radius,
                grid_resolution_m=args.grid_resolution,
                turning_diameter_m=args.turning_diameter,
                humanoid_height_m=args.humanoid_height,
                forklift_radius_m=args.forklift_radius,
            )
            _write_json(args.output.resolve(), result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result.get("status") == "pass" else 2
    except ValidationError as exc:
        failure = {
            "verification_schema_id": (
                VERIFICATION_SCHEMA_ID if args.verify_only else None
            ),
            "schema_version": SCHEMA_VERSION,
            "algorithm": ALGORITHM,
            "status": "fail",
            "mode": "verify-only" if args.verify_only else "evaluate",
            "navigation_recomputed": False,
            "critical_issues": [str(exc)],
        }
        if args.verify_only:
            failure["attestation"] = {
                "algorithm": "sha256(canonical JSON excluding attestation)",
                "payload_sha256": _canonical_hash(failure),
                "self_attested_status": "fail",
            }
            if args.verification_output is not None:
                _write_json(args.verification_output.resolve(), failure)
        else:
            _write_json(args.output.resolve(), failure)
        print(json.dumps(failure, indent=2, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
