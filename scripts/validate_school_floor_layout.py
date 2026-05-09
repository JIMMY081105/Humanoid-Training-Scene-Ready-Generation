#!/usr/bin/env python3
"""Reject a generated school floor plan that diverges from the reference contract."""

from __future__ import annotations

import argparse
import itertools
import json
import math

from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any


EXPECTED_IDS = {
    "classroom_01",
    "classroom_02",
    "classroom_03",
    "classroom_04",
    "classroom_05",
    "classroom_06",
    "library",
    "boys_toilet",
    "girls_toilet",
    "storage_room",
    "main_corridor",
}

AREA_RANGES = {
    **{f"classroom_{index:02d}": (58.0, 85.0) for index in range(1, 7)},
    "library": (75.0, 115.0),
    "boys_toilet": (11.0, 24.0),
    "girls_toilet": (11.0, 24.0),
    "storage_room": (12.0, 28.0),
    "main_corridor": (170.0, 280.0),
}

TARGET_DIMENSIONS = {
    **{f"classroom_{index:02d}": (9.0, 7.5) for index in range(1, 7)},
    "library": (10.0, 9.0),
    "boys_toilet": (4.0, 4.0),
    "girls_toilet": (4.0, 4.0),
    "storage_room": (5.0, 3.7),
    "main_corridor": (12.0, 20.0),
}

DIMENSION_RELATIVE_TOLERANCE = 0.20
ASPECT_RATIO_RELATIVE_TOLERANCE = 0.25
MAXIMUM_CONNECTION_GAP = 0.25
MINIMUM_CONNECTION_SPAN = 0.8
MINIMUM_NAVIGATION_DOOR_WIDTH = 0.9
GEOMETRY_TOLERANCE = 0.05
BOUNDARY_TOLERANCE = 0.10
CENTER_SEGMENT_START_FRACTION = 0.33
CENTER_SEGMENT_END_FRACTION = 0.67
MINIMUM_COMMON_ZONE_WIDTH = 0.9
MINIMUM_CARVED_ROOM_REMAINDER = 1.5
MAXIMUM_CARVED_ROOM_AREA_FRACTION = 0.45
RESTROOM_ROW_TOLERANCE = 1.0
LIBRARY_CENTER_TOLERANCE = 1.0
CLASSROOM_ROW_ALIGNMENT_TOLERANCE = 1.0
MAXIMUM_RESTROOM_LIBRARY_HORIZONTAL_GAP = 2.0
MAXIMUM_RESTROOM_LIBRARY_VERTICAL_GAP = 8.0
CORE_COMMON_ROOM_IDS = frozenset({"library", "main_corridor"})
WINDOW_REQUIRED_ROOM_IDS = (
    *(f"classroom_{index:02d}" for index in range(1, 7)),
    "library",
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _placed_room_entries(layout: dict[str, Any]) -> list[dict[str, Any]]:
    rooms = layout.get("placed_rooms", [])
    if not isinstance(rooms, list):
        return []
    return [room for room in rooms if isinstance(room, dict)]


def _room_map(layout: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return the first placed room per ID; duplicate detection is separate."""

    result: dict[str, dict[str, Any]] = {}
    for room in _placed_room_entries(layout):
        room_id = str(room.get("room_id") or "").strip()
        if room_id and room_id not in result:
            result[room_id] = room
    return result


def _float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _center(room: dict[str, Any]) -> tuple[float, float]:
    position = room.get("position", [0.0, 0.0])
    x = (
        _float_or_nan(position[0])
        if isinstance(position, (list, tuple)) and position
        else math.nan
    )
    y = (
        _float_or_nan(position[1])
        if isinstance(position, (list, tuple)) and len(position) > 1
        else math.nan
    )
    return (
        x + _float_or_nan(room.get("width")) / 2,
        y + _float_or_nan(room.get("depth")) / 2,
    )


def _bounds(room: dict[str, Any]) -> tuple[float, float, float, float]:
    position = room.get("position", [0.0, 0.0])
    x = (
        _float_or_nan(position[0])
        if isinstance(position, (list, tuple)) and position
        else math.nan
    )
    y = (
        _float_or_nan(position[1])
        if isinstance(position, (list, tuple)) and len(position) > 1
        else math.nan
    )
    return (
        x,
        y,
        x + _float_or_nan(room.get("width")),
        y + _float_or_nan(room.get("depth")),
    )


def _all_finite(values: tuple[float, ...]) -> bool:
    return all(math.isfinite(value) for value in values)


def _overlap_area(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    overlap_x = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    overlap_y = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    return overlap_x * overlap_y


def _are_adjacent(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
    *,
    maximum_gap: float = 0.25,
    minimum_shared_span: float = 1.0,
) -> bool:
    horizontal_gap = max(first[0] - second[2], second[0] - first[2], 0.0)
    vertical_overlap = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    vertical_gap = max(first[1] - second[3], second[1] - first[3], 0.0)
    horizontal_overlap = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    return (
        horizontal_gap <= maximum_gap and vertical_overlap >= minimum_shared_span
    ) or (
        vertical_gap <= maximum_gap and horizontal_overlap >= minimum_shared_span
    )


def _adjacency_evidence(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
    *,
    maximum_gap: float = MAXIMUM_CONNECTION_GAP,
    minimum_shared_span: float = MINIMUM_CONNECTION_SPAN,
) -> dict[str, Any]:
    """Describe a shared horizontal or vertical boundary between two rooms."""

    if not _all_finite(first) or not _all_finite(second):
        return {
            "adjacent": False,
            "orientation": None,
            "gap_m": None,
            "shared_span_m": 0.0,
        }

    horizontal_gap = min(abs(first[2] - second[0]), abs(second[2] - first[0]))
    vertical_overlap = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    vertical_gap = min(abs(first[3] - second[1]), abs(second[3] - first[1]))
    horizontal_overlap = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    candidates = []
    if horizontal_gap <= maximum_gap and vertical_overlap >= minimum_shared_span:
        candidates.append((horizontal_gap, -vertical_overlap, "east_west", vertical_overlap))
    if vertical_gap <= maximum_gap and horizontal_overlap >= minimum_shared_span:
        candidates.append((vertical_gap, -horizontal_overlap, "north_south", horizontal_overlap))
    if not candidates:
        return {
            "adjacent": False,
            "orientation": None,
            "gap_m": min(horizontal_gap, vertical_gap),
            "shared_span_m": max(horizontal_overlap, vertical_overlap),
        }
    gap, _negative_span, orientation, shared_span = min(candidates)
    return {
        "adjacent": True,
        "orientation": orientation,
        "gap_m": gap,
        "shared_span_m": shared_span,
    }


def _wall_for_direction(
    room: dict[str, Any], direction: str
) -> dict[str, Any] | None:
    for wall in room.get("walls", []):
        if (
            isinstance(wall, dict)
            and str(wall.get("direction", "")).split(".")[-1].lower() == direction
        ):
            return wall
    return None


def _wall_opening(
    room: dict[str, Any], opening_id: str
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    for wall in room.get("walls", []):
        if not isinstance(wall, dict):
            continue
        for opening in wall.get("openings", []):
            if isinstance(opening, dict) and str(opening.get("opening_id")) == opening_id:
                return wall, opening
    return None


def _wall_openings(
    room: dict[str, Any], opening_id: str
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Return every wall opening with ``opening_id`` in one placed room."""

    matches = []
    for wall in room.get("walls", []):
        if not isinstance(wall, dict):
            continue
        for opening in wall.get("openings", []):
            if isinstance(opening, dict) and str(opening.get("opening_id")) == opening_id:
                matches.append((wall, opening))
    return matches


def _opening_geometry_evidence(
    wall: dict[str, Any],
    opening: dict[str, Any],
    *,
    require_positive_height: bool,
) -> dict[str, Any]:
    """Validate finite opening geometry against the wall segment that owns it."""

    issues: list[str] = []
    start = wall.get("start_point")
    end = wall.get("end_point")
    if not (
        isinstance(start, (list, tuple))
        and len(start) >= 2
        and isinstance(end, (list, tuple))
        and len(end) >= 2
    ):
        issues.append("wall start/end geometry is missing")
        start_xy = (math.nan, math.nan)
        end_xy = (math.nan, math.nan)
    else:
        start_xy = (_float_or_nan(start[0]), _float_or_nan(start[1]))
        end_xy = (_float_or_nan(end[0]), _float_or_nan(end[1]))
    wall_length = math.hypot(end_xy[0] - start_xy[0], end_xy[1] - start_xy[1])
    position = _float_or_nan(opening.get("position_along_wall"))
    width = _float_or_nan(opening.get("width"))
    height = _float_or_nan(opening.get("height"))
    if not _all_finite((*start_xy, *end_xy)) or wall_length <= GEOMETRY_TOLERANCE:
        issues.append("wall start/end geometry must be finite and nondegenerate")
    if not math.isfinite(position) or position < -GEOMETRY_TOLERANCE:
        issues.append("opening position_along_wall must be finite and nonnegative")
    if not math.isfinite(width) or width <= 0.0:
        issues.append("opening width must be a positive finite number")
    if require_positive_height and (not math.isfinite(height) or height <= 0.0):
        issues.append("opening height must be a positive finite number")
    elif not require_positive_height and not math.isfinite(height):
        issues.append("opening height must be finite")
    if (
        math.isfinite(position)
        and math.isfinite(width)
        and math.isfinite(wall_length)
        and position + width > wall_length + GEOMETRY_TOLERANCE
    ):
        issues.append("opening does not fit inside its wall segment")
    center = _opening_center(wall, opening)
    if center is None:
        issues.append("opening center cannot be derived from finite wall geometry")
    return {
        "valid": not issues,
        "position_along_wall_m": position if math.isfinite(position) else None,
        "width_m": width if math.isfinite(width) else None,
        "height_m": height if math.isfinite(height) else None,
        "wall_length_m": wall_length if math.isfinite(wall_length) else None,
        "center": center,
        "issues": issues,
    }


def _opening_fits_room_side(
    center: tuple[float, float] | None,
    width: float,
    room_bounds: tuple[float, float, float, float],
    direction: str,
) -> bool:
    """Return whether an opening lies on and fits the named room boundary."""

    if center is None or not _all_finite((*center, width, *room_bounds)) or width <= 0.0:
        return False
    if direction == "north":
        return (
            abs(center[1] - room_bounds[3]) <= BOUNDARY_TOLERANCE
            and center[0] - width / 2.0 >= room_bounds[0] - GEOMETRY_TOLERANCE
            and center[0] + width / 2.0 <= room_bounds[2] + GEOMETRY_TOLERANCE
        )
    if direction == "south":
        return (
            abs(center[1] - room_bounds[1]) <= BOUNDARY_TOLERANCE
            and center[0] - width / 2.0 >= room_bounds[0] - GEOMETRY_TOLERANCE
            and center[0] + width / 2.0 <= room_bounds[2] + GEOMETRY_TOLERANCE
        )
    if direction == "east":
        return (
            abs(center[0] - room_bounds[2]) <= BOUNDARY_TOLERANCE
            and center[1] - width / 2.0 >= room_bounds[1] - GEOMETRY_TOLERANCE
            and center[1] + width / 2.0 <= room_bounds[3] + GEOMETRY_TOLERANCE
        )
    if direction == "west":
        return (
            abs(center[0] - room_bounds[0]) <= BOUNDARY_TOLERANCE
            and center[1] - width / 2.0 >= room_bounds[1] - GEOMETRY_TOLERANCE
            and center[1] + width / 2.0 <= room_bounds[3] + GEOMETRY_TOLERANCE
        )
    return False


def _wall_matches_room_side(
    wall: dict[str, Any],
    room_bounds: tuple[float, float, float, float],
    direction: str,
) -> bool:
    """Require a serialized wall segment to be the named AABB boundary.

    SceneSmith serializes each placed rectangular room with four concrete wall
    segments.  Checking their endpoints prevents a forged direction label from
    turning an arbitrary segment into exterior/opening evidence.
    """

    start = wall.get("start_point")
    end = wall.get("end_point")
    if not (
        isinstance(start, (list, tuple))
        and len(start) >= 2
        and isinstance(end, (list, tuple))
        and len(end) >= 2
        and _all_finite((*room_bounds, _float_or_nan(start[0]), _float_or_nan(start[1]),
                         _float_or_nan(end[0]), _float_or_nan(end[1])))
    ):
        return False
    x0, y0, x1, y1 = room_bounds
    expected_by_direction = {
        "north": ((x0, y1), (x1, y1)),
        "south": ((x0, y0), (x1, y0)),
        "east": ((x1, y0), (x1, y1)),
        "west": ((x0, y0), (x0, y1)),
    }
    expected = expected_by_direction.get(direction)
    if expected is None:
        return False
    actual = (
        (_float_or_nan(start[0]), _float_or_nan(start[1])),
        (_float_or_nan(end[0]), _float_or_nan(end[1])),
    )

    def close(first: tuple[float, float], second: tuple[float, float]) -> bool:
        return math.dist(first, second) <= BOUNDARY_TOLERANCE

    return (
        close(actual[0], expected[0]) and close(actual[1], expected[1])
    ) or (
        close(actual[0], expected[1]) and close(actual[1], expected[0])
    )


def _opening_in_center_segment(geometry: dict[str, Any]) -> bool:
    """Match the center-segment semantics used by FloorPlanTools.

    ``position_segment='center'`` causes the tool to sample the opening center
    from the middle 33%-67% of the wall, not necessarily at its exact midpoint.
    """

    position = _float_or_nan(geometry.get("position_along_wall_m"))
    width = _float_or_nan(geometry.get("width_m"))
    wall_length = _float_or_nan(geometry.get("wall_length_m"))
    if not _all_finite((position, width, wall_length)) or wall_length <= 0.0:
        return False
    center = position + width / 2.0
    return (
        wall_length * CENTER_SEGMENT_START_FRACTION - GEOMETRY_TOLERANCE
        <= center
        <= wall_length * CENTER_SEGMENT_END_FRACTION + GEOMETRY_TOLERANCE
    )


def _side_exposure_blockers(
    bounds: dict[str, tuple[float, float, float, float]],
    room_id: str,
    direction: str,
) -> list[str]:
    room_bounds = bounds[room_id]
    blockers = []
    for other_id, other in bounds.items():
        if other_id == room_id or not _all_finite(other):
            continue
        if direction == "north":
            gap = abs(room_bounds[3] - other[1])
            shared = max(0.0, min(room_bounds[2], other[2]) - max(room_bounds[0], other[0]))
        elif direction == "south":
            gap = abs(room_bounds[1] - other[3])
            shared = max(0.0, min(room_bounds[2], other[2]) - max(room_bounds[0], other[0]))
        elif direction == "east":
            gap = abs(room_bounds[2] - other[0])
            shared = max(0.0, min(room_bounds[3], other[3]) - max(room_bounds[1], other[1]))
        else:  # west
            gap = abs(room_bounds[0] - other[2])
            shared = max(0.0, min(room_bounds[3], other[3]) - max(room_bounds[1], other[1]))
        if gap <= MAXIMUM_CONNECTION_GAP and shared > 1e-4:
            blockers.append(other_id)
    return sorted(blockers)


def _exterior_window_evidence(
    layout: dict[str, Any],
    rooms: dict[str, dict[str, Any]],
    bounds: dict[str, tuple[float, float, float, float]],
    room_id: str,
) -> list[dict[str, Any]]:
    evidence = []
    for window in layout.get("windows", []):
        if not isinstance(window, dict) or str(window.get("room_id")) != room_id:
            continue
        window_id = str(window.get("id") or "").strip()
        opening_records = _wall_openings(rooms[room_id], window_id) if window_id else []
        opening_wall, opening = opening_records[0] if len(opening_records) == 1 else (None, None)
        declared_direction = str(window.get("wall_direction") or "").split(".")[-1].lower()
        opening_direction = (
            str(opening_wall.get("direction") or "").split(".")[-1].lower()
            if opening_wall
            else ""
        )
        direction = declared_direction or opening_direction
        wall = opening_wall
        blockers = (
            _side_exposure_blockers(bounds, room_id, direction)
            if direction in {"north", "south", "east", "west"}
            else []
        )
        issues = []
        if not window_id:
            issues.append("window id is missing")
        if len(opening_records) != 1:
            issues.append("window must have exactly one matching wall opening")
        if direction not in {"north", "south", "east", "west"}:
            issues.append("wall direction is missing or invalid")
        if declared_direction and opening_direction and declared_direction != opening_direction:
            issues.append("window wall_direction disagrees with its wall opening")
        if (
            opening is None
            or str(opening.get("opening_type", "")).split(".")[-1].lower()
            != "window"
        ):
            issues.append("matching WINDOW wall opening is missing")
        if wall is None or wall.get("is_exterior") is not True:
            issues.append("window wall is not marked exterior")
        if (
            wall is None
            or direction not in {"north", "south", "east", "west"}
            or not _wall_matches_room_side(wall, bounds[room_id], direction)
        ):
            issues.append("window wall geometry does not match its declared room boundary")
        if blockers:
            issues.append(f"window wall is physically blocked by rooms {blockers}")
        record_position = _float_or_nan(window.get("position_along_wall"))
        record_width = _float_or_nan(window.get("width"))
        record_height = _float_or_nan(window.get("height"))
        record_sill = _float_or_nan(window.get("sill_height"))
        if not math.isfinite(record_position) or record_position < -GEOMETRY_TOLERANCE:
            issues.append("window position_along_wall must be finite and nonnegative")
        if not math.isfinite(record_width) or record_width <= 0.0:
            issues.append("window width must be a positive finite number")
        if not math.isfinite(record_height) or record_height <= 0.0:
            issues.append("window height must be a positive finite number")
        if not math.isfinite(record_sill) or record_sill < 0.0:
            issues.append("window sill_height must be finite and nonnegative")

        opening_geometry = None
        if opening_wall is not None and opening is not None:
            opening_geometry = _opening_geometry_evidence(
                opening_wall, opening, require_positive_height=True
            )
            issues.extend(opening_geometry["issues"])
            opening_position = _float_or_nan(opening.get("position_along_wall"))
            opening_width = _float_or_nan(opening.get("width"))
            opening_height = _float_or_nan(opening.get("height"))
            opening_sill = _float_or_nan(opening.get("sill_height"))
            comparisons = (
                (record_position, opening_position, "position"),
                (record_width, opening_width, "width"),
                (record_height, opening_height, "height"),
                (record_sill, opening_sill, "sill height"),
            )
            for declared, generated, label in comparisons:
                if (
                    not _all_finite((declared, generated))
                    or abs(declared - generated) > GEOMETRY_TOLERANCE
                ):
                    issues.append(f"window record and wall opening {label} disagree")
            if (
                direction in {"north", "south", "east", "west"}
                and not _opening_fits_room_side(
                    opening_geometry.get("center"),
                    opening_width,
                    bounds[room_id],
                    direction,
                )
            ):
                issues.append("window opening does not fit its declared room boundary")
        evidence.append(
            {
                "window_id": window_id,
                "direction": direction or None,
                "opening_geometry": opening_geometry,
                "valid_exterior_window": not issues,
                "issues": issues,
            }
        )
    return evidence


def _entrance_door_evidence(
    door: dict[str, Any],
    rooms: dict[str, dict[str, Any]],
    bounds: dict[str, tuple[float, float, float, float]],
) -> dict[str, Any]:
    """Validate the one exterior entrance against its generated wall cutout."""

    issues: list[str] = []
    door_id = str(door.get("id") or "").strip()
    room_id = str(door.get("room_a") or "").strip()
    width = _float_or_nan(door.get("width"))
    height = _float_or_nan(door.get("height"))
    position_exact = _float_or_nan(door.get("position_exact"))
    leaf_count = door.get("leaf_count")
    if not door_id:
        issues.append("exterior door id is missing")
    if door.get("room_b") is not None:
        issues.append("exterior entrance must have room_b=null")
    if room_id != "library":
        issues.append("exterior entrance must belong to library")
    if str(door.get("door_type") or "").lower() != "exterior":
        issues.append("exterior entrance must have door_type=exterior")
    if str(door.get("position_segment") or "").lower() != "center":
        issues.append("exterior entrance must declare position_segment=center")
    if not math.isfinite(width) or width < 1.6:
        issues.append("exterior double entrance width must be a finite >=1.6 m")
    if not math.isfinite(height) or height <= 0.0:
        issues.append("exterior entrance height must be a positive finite number")
    # FloorPlanTools writes this native integer field when it creates a wide
    # exterior Door.  There is no separate per-leaf layout schema to invent.
    if type(leaf_count) is not int or leaf_count != 2:
        issues.append("exterior entrance must serialize exactly two physical leaves")
    if not math.isfinite(position_exact) or position_exact < -GEOMETRY_TOLERANCE:
        issues.append("exterior entrance position_exact must be finite and nonnegative")

    opening_records = (
        _wall_openings(rooms[room_id], door_id)
        if room_id in rooms and door_id
        else []
    )
    if len(opening_records) != 1:
        issues.append("exterior entrance must have exactly one generated wall opening")
    opening_geometry = None
    direction = None
    if len(opening_records) == 1:
        wall, opening = opening_records[0]
        direction = str(wall.get("direction") or "").split(".")[-1].lower()
        opening_geometry = _opening_geometry_evidence(
            wall, opening, require_positive_height=True
        )
        issues.extend(opening_geometry["issues"])
        opening_width = _float_or_nan(opening.get("width"))
        opening_height = _float_or_nan(opening.get("height"))
        opening_position = _float_or_nan(opening.get("position_along_wall"))
        opening_sill = _float_or_nan(opening.get("sill_height"))
        if str(opening.get("opening_type") or "").split(".")[-1].lower() != "door":
            issues.append("exterior entrance wall opening must have type=door")
        if direction != "south":
            issues.append("exterior entrance wall opening must be on the library south wall")
        if wall.get("is_exterior") is not True:
            issues.append("exterior entrance wall must be marked exterior")
        if room_id not in bounds or not _wall_matches_room_side(
            wall, bounds[room_id], direction
        ):
            issues.append("exterior entrance wall geometry does not match the room boundary")
        if not _all_finite((width, opening_width)) or abs(width - opening_width) > GEOMETRY_TOLERANCE:
            issues.append("exterior entrance record and wall opening width disagree")
        if not _all_finite((height, opening_height)) or abs(height - opening_height) > GEOMETRY_TOLERANCE:
            issues.append("exterior entrance record and wall opening height disagree")
        if not _all_finite((position_exact, opening_position)) or abs(
            position_exact - opening_position
        ) > GEOMETRY_TOLERANCE:
            issues.append("exterior entrance position_exact disagrees with its wall opening")
        if not math.isfinite(opening_sill) or abs(opening_sill) > GEOMETRY_TOLERANCE:
            issues.append("exterior entrance wall opening must start at floor level")
        if (
            room_id not in bounds
            or not _opening_fits_room_side(
                opening_geometry.get("center"),
                opening_width,
                bounds[room_id],
                direction,
            )
        ):
            issues.append("exterior entrance opening does not fit the library south boundary")
        if not _opening_in_center_segment(opening_geometry):
            issues.append(
                "exterior entrance opening center is outside the generated center segment"
            )
        if room_id in bounds and direction == "south" and _side_exposure_blockers(
            bounds, room_id, direction
        ):
            issues.append("exterior entrance south side is physically blocked by another room")

    return {
        "door_id": door_id,
        "room_id": room_id or None,
        "width_m": width if math.isfinite(width) else None,
        "height_m": height if math.isfinite(height) else None,
        "leaf_count": leaf_count,
        "leaf_width_m": (
            width / 2.0
            if type(leaf_count) is int and leaf_count == 2 and math.isfinite(width)
            else None
        ),
        "direction": direction,
        "opening_geometry": opening_geometry,
        "valid": not issues,
        "issues": issues,
    }


def _connection_records(layout: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for door in layout.get("doors", []):
        if not isinstance(door, dict):
            continue
        first = door.get("room_a")
        second = door.get("room_b")
        if first and second:
            records.append(
                {
                    "room_a": str(first),
                    "room_b": str(second),
                    "connection_type": "door",
                    "source": "door",
                    "source_id": str(door.get("id") or ""),
                    "width": _float_or_nan(door.get("width")),
                    "door_type": str(door.get("door_type") or "").lower(),
                    "reciprocal": True,
                }
            )
    # Open-plan connections are serialized on RoomSpec rather than as Door
    # records. Door-valued RoomSpec connections are design metadata backed by
    # the authoritative Door records above and must not become duplicate fake
    # door records. OPEN connections must be reciprocal and are deduplicated.
    raw_room_specs = layout.get("rooms", [])
    room_specs = [room for room in raw_room_specs if isinstance(room, dict)] if isinstance(raw_room_specs, list) else []
    spec_connections: dict[str, dict[str, str]] = {}
    for room in room_specs:
        room_id = str(room.get("room_id") or room.get("id") or "").strip()
        raw_connections = room.get("connections", {})
        if room_id and isinstance(raw_connections, dict):
            spec_connections[room_id] = {
                str(other_id).strip(): str(connection_type).split(".")[-1].lower()
                for other_id, connection_type in raw_connections.items()
                if str(other_id).strip()
            }
    processed_open_pairs: set[tuple[str, str]] = set()
    for room in room_specs:
        if not isinstance(room, dict):
            continue
        room_id = str(room.get("room_id") or room.get("id") or "").strip()
        connections = room.get("connections", {})
        if not room_id or not isinstance(connections, dict):
            continue
        for other_id, connection_type in connections.items():
            normalized = str(connection_type).split(".")[-1].lower()
            other = str(other_id).strip()
            if other and normalized == "open":
                pair = tuple(sorted((room_id, other)))
                if pair in processed_open_pairs:
                    continue
                processed_open_pairs.add(pair)
                records.append(
                    {
                        "room_a": pair[0],
                        "room_b": pair[1],
                        "connection_type": "open",
                        "source": "room_spec",
                        "source_id": f"open_{pair[0]}_{pair[1]}",
                        "width": None,
                        "door_type": None,
                        "reciprocal": (
                            spec_connections.get(room_id, {}).get(other) == "open"
                            and spec_connections.get(other, {}).get(room_id) == "open"
                        ),
                    }
                )
    return records


def _opening_direction(room: dict[str, Any], opening_id: str) -> str | None:
    record = _wall_opening(room, opening_id)
    if record is None:
        return None
    return str(record[0].get("direction", "")).split(".")[-1].lower() or None


def _opening_center(
    wall: dict[str, Any], opening: dict[str, Any]
) -> tuple[float, float] | None:
    start = wall.get("start_point")
    end = wall.get("end_point")
    if not (
        isinstance(start, (list, tuple))
        and len(start) >= 2
        and isinstance(end, (list, tuple))
        and len(end) >= 2
    ):
        return None
    sx, sy = _float_or_nan(start[0]), _float_or_nan(start[1])
    ex, ey = _float_or_nan(end[0]), _float_or_nan(end[1])
    position = _float_or_nan(opening.get("position_along_wall"))
    width = _float_or_nan(opening.get("width"))
    length = math.hypot(ex - sx, ey - sy)
    if not _all_finite((sx, sy, ex, ey, position, width)) or length <= 1e-6:
        return None
    distance = position + width / 2.0
    return (
        sx + (ex - sx) * distance / length,
        sy + (ey - sy) * distance / length,
    )


def _generated_opening_ids_match(
    first_id: str, second_id: str, first_room: str, second_room: str
) -> bool:
    """Recognize the paired IDs emitted by OpenPlanMixin.

    The first wall gets ``open_A_B`` and the reciprocal wall gets
    ``open_A_B_b``.  RoomSpec iteration order may choose either endpoint first,
    so both endpoint orders are valid, but unrelated co-located cutouts are not.
    """

    for stem in (
        f"open_{first_room}_{second_room}",
        f"open_{second_room}_{first_room}",
    ):
        if {first_id, second_id} == {stem, f"{stem}_b"}:
            return True
    return False


def _open_threshold_evidence(
    first: str,
    second: str,
    rooms: dict[str, dict[str, Any]],
    bounds: dict[str, tuple[float, float, float, float]],
) -> dict[str, Any]:
    """Require co-located finite OPEN wall cutouts on both adjacent rooms."""

    shared = _shared_boundary_geometry(bounds[first], bounds[second])
    issues: list[str] = []
    candidates_by_room: dict[str, list[dict[str, Any]]] = {first: [], second: []}
    for room_id in (first, second):
        for wall in rooms[room_id].get("walls", []):
            if not isinstance(wall, dict):
                continue
            for opening in wall.get("openings", []):
                if (
                    not isinstance(opening, dict)
                    or str(opening.get("opening_type") or "").split(".")[-1].lower()
                    != "open"
                ):
                    continue
                geometry = _opening_geometry_evidence(
                    wall, opening, require_positive_height=False
                )
                width = _float_or_nan(opening.get("width"))
                center = geometry.get("center")
                local_issues = list(geometry["issues"])
                direction = str(wall.get("direction") or "").split(".")[-1].lower()
                if not _wall_matches_room_side(wall, bounds[room_id], direction):
                    local_issues.append(
                        "OPEN wall geometry does not match its owning room boundary"
                    )
                if (
                    shared is None
                    or not math.isfinite(width)
                    or width < MINIMUM_NAVIGATION_DOOR_WIDTH
                    or center is None
                    or not _point_on_threshold(center, shared, width)
                ):
                    local_issues.append(
                        "OPEN wall cutout does not fit the shared room boundary"
                    )
                candidates_by_room[room_id].append(
                    {
                        "opening_id": str(opening.get("opening_id") or ""),
                        "center": center,
                        "width_m": width if math.isfinite(width) else None,
                        "valid": not local_issues,
                        "issues": local_issues,
                    }
                )

    valid_pairs = []
    for first_candidate in candidates_by_room[first]:
        for second_candidate in candidates_by_room[second]:
            first_center = first_candidate.get("center")
            second_center = second_candidate.get("center")
            first_width = _float_or_nan(first_candidate.get("width_m"))
            second_width = _float_or_nan(second_candidate.get("width_m"))
            if (
                first_candidate["valid"]
                and second_candidate["valid"]
                and isinstance(first_center, (list, tuple))
                and isinstance(second_center, (list, tuple))
                and math.dist(first_center, second_center) <= GEOMETRY_TOLERANCE
                and abs(first_width - second_width) <= GEOMETRY_TOLERANCE
                and _generated_opening_ids_match(
                    str(first_candidate.get("opening_id") or ""),
                    str(second_candidate.get("opening_id") or ""),
                    first,
                    second,
                )
            ):
                valid_pairs.append((first_candidate, second_candidate))
    if not valid_pairs:
        issues.append(
            "OPEN connection requires reciprocal co-located generated wall cutouts "
            "on both rooms"
        )
        return {
            "valid": False,
            "orientation": shared[0] if shared else None,
            "center": None,
            "width_m": None,
            "wall_openings": candidates_by_room,
            "issues": issues,
        }
    selected_first, selected_second = valid_pairs[0]
    return {
        "valid": True,
        "orientation": shared[0] if shared else None,
        "center": selected_first["center"],
        "width_m": selected_first["width_m"],
        "wall_openings": candidates_by_room,
        "paired_opening_ids": [
            selected_first["opening_id"],
            selected_second["opening_id"],
        ],
        "issues": [],
    }


def _physical_threshold_evidence(
    record: dict[str, Any],
    rooms: dict[str, dict[str, Any]],
    bounds: dict[str, tuple[float, float, float, float]],
) -> dict[str, Any]:
    first = record["room_a"]
    second = record["room_b"]
    shared = (
        _shared_boundary_geometry(bounds[first], bounds[second])
        if first in bounds and second in bounds
        else None
    )
    issues: list[str] = []
    if shared is None:
        issues.append("room endpoints lack a shared physical threshold")
        return {
            "valid": False,
            "orientation": None,
            "center": None,
            "width_m": None,
            "issues": issues,
        }

    if record["connection_type"] == "open":
        return _open_threshold_evidence(first, second, rooms, bounds)

    opening_id = str(record.get("source_id") or "")
    opening_records = [
        _wall_opening(rooms[room_id], opening_id) if room_id in rooms else None
        for room_id in (first, second)
    ]
    if any(item is None for item in opening_records):
        issues.append("door lacks matching wall openings on both rooms")
        return {
            "valid": False,
            "orientation": shared[0],
            "center": None,
            "width_m": record.get("width"),
            "issues": issues,
        }
    centers = []
    for room_id, opening_record in zip((first, second), opening_records):
        assert opening_record is not None
        wall, opening = opening_record
        opening_type = str(opening.get("opening_type") or "").split(".")[-1].lower()
        opening_width = _float_or_nan(opening.get("width"))
        if opening_type != "door":
            issues.append(f"{room_id} matching opening is not a door")
        if (
            not math.isfinite(opening_width)
            or abs(opening_width - _float_or_nan(record.get("width"))) > 0.05
        ):
            issues.append(f"{room_id} opening width disagrees with the door")
        center = _opening_center(wall, opening)
        if center is None:
            issues.append(f"{room_id} opening lacks finite wall geometry/position")
        else:
            centers.append(center)
    center = centers[0] if centers else None
    if len(centers) == 2 and math.dist(centers[0], centers[1]) > 0.05:
        issues.append("paired door openings disagree in house coordinates")
    width = _float_or_nan(record.get("width"))
    if center is not None and (
        shared[0] not in {"horizontal", "vertical"}
        or not _point_on_threshold(center, shared, width)
    ):
        issues.append("door opening does not fit its shared room boundary")
    return {
        "valid": not issues,
        "orientation": shared[0],
        "center": center,
        "width_m": width if math.isfinite(width) else None,
        "issues": issues,
    }


def _reachable(graph: dict[str, set[str]], start: str) -> set[str]:
    seen = {start}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for neighbor in graph.get(node, set()):
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    return seen


def _finite_bounds(
    value: tuple[float, float, float, float]
) -> bool:
    return _all_finite(value) and value[2] > value[0] and value[3] > value[1]


def _navigation_zone_bounds(record: dict[str, Any]) -> tuple[float, float, float, float]:
    position = record.get("position")
    if not isinstance(position, (list, tuple)) or len(position) < 2:
        return (math.nan, math.nan, math.nan, math.nan)
    x = _float_or_nan(position[0])
    y = _float_or_nan(position[1])
    width = _float_or_nan(record.get("width"))
    depth = _float_or_nan(record.get("depth"))
    return (x, y, x + width, y + depth)


def _shared_boundary_geometry(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> tuple[str, float, float, float] | None:
    """Return orientation, fixed coordinate, tangent start, tangent end."""

    if not _finite_bounds(first) or not _finite_bounds(second):
        return None
    y_start = max(first[1], second[1])
    y_end = min(first[3], second[3])
    if y_end - y_start > 1e-4:
        if abs(first[2] - second[0]) <= MAXIMUM_CONNECTION_GAP:
            return ("vertical", (first[2] + second[0]) / 2.0, y_start, y_end)
        if abs(second[2] - first[0]) <= MAXIMUM_CONNECTION_GAP:
            return ("vertical", (second[2] + first[0]) / 2.0, y_start, y_end)
    x_start = max(first[0], second[0])
    x_end = min(first[2], second[2])
    if x_end - x_start > 1e-4:
        if abs(first[3] - second[1]) <= MAXIMUM_CONNECTION_GAP:
            return ("horizontal", (first[3] + second[1]) / 2.0, x_start, x_end)
        if abs(second[3] - first[1]) <= MAXIMUM_CONNECTION_GAP:
            return ("horizontal", (second[3] + first[1]) / 2.0, x_start, x_end)
    return None


def _point_on_threshold(
    point: tuple[float, float],
    shared: tuple[str, float, float, float],
    width: float,
) -> bool:
    orientation, fixed, start, end = shared
    normal = point[0] if orientation == "vertical" else point[1]
    tangent = point[1] if orientation == "vertical" else point[0]
    return (
        abs(normal - fixed) <= MAXIMUM_CONNECTION_GAP
        and tangent - width / 2.0 >= start - 1e-4
        and tangent + width / 2.0 <= end + 1e-4
    )


def _point_in_bounds(
    point: tuple[float, float],
    room_bounds: tuple[float, float, float, float],
) -> bool:
    return (
        room_bounds[0] - 1e-4 <= point[0] <= room_bounds[2] + 1e-4
        and room_bounds[1] - 1e-4 <= point[1] <= room_bounds[3] + 1e-4
    )


def _threshold_contains(
    backing: dict[str, Any],
    point: tuple[float, float],
    orientation: str,
    width: float,
) -> bool:
    center = backing.get("center")
    backing_width = _float_or_nan(backing.get("width_m"))
    if (
        backing.get("valid") is not True
        or backing.get("orientation") != orientation
        or not isinstance(center, (list, tuple))
        or len(center) < 2
        or not _all_finite((point[0], point[1], center[0], center[1], width, backing_width))
    ):
        return False
    normal_axis = 0 if orientation == "vertical" else 1
    tangent_axis = 1 if orientation == "vertical" else 0
    return (
        abs(point[normal_axis] - center[normal_axis]) <= 0.1
        and abs(point[tangent_axis] - center[tangent_axis]) + width / 2.0
        <= backing_width / 2.0 + 0.05
    )


def _threshold_inside_bounds(
    threshold: dict[str, Any],
    container: tuple[float, float, float, float],
) -> bool:
    center = threshold.get("center")
    width = _float_or_nan(threshold.get("width_m"))
    orientation = threshold.get("orientation")
    if (
        threshold.get("valid") is not True
        or orientation not in {"horizontal", "vertical"}
        or not isinstance(center, (list, tuple))
        or len(center) < 2
        or not _all_finite((center[0], center[1], width))
    ):
        return False
    if orientation == "vertical":
        return (
            container[0] - 0.1 <= center[0] <= container[2] + 0.1
            and center[1] - width / 2.0 >= container[1] - 0.05
            and center[1] + width / 2.0 <= container[3] + 0.05
        )
    return (
        container[1] - 0.1 <= center[1] <= container[3] + 0.1
        and center[0] - width / 2.0 >= container[0] - 0.05
        and center[0] + width / 2.0 <= container[2] + 0.05
    )


def _carve_evidence(
    room_bounds: tuple[float, float, float, float],
    zone_bounds: tuple[float, float, float, float],
) -> dict[str, Any]:
    overlap = (
        max(room_bounds[0], zone_bounds[0]),
        max(room_bounds[1], zone_bounds[1]),
        min(room_bounds[2], zone_bounds[2]),
        min(room_bounds[3], zone_bounds[3]),
    )
    overlap_width = max(0.0, overlap[2] - overlap[0])
    overlap_depth = max(0.0, overlap[3] - overlap[1])
    room_width = room_bounds[2] - room_bounds[0]
    room_depth = room_bounds[3] - room_bounds[1]
    area_fraction = (
        overlap_width * overlap_depth / (room_width * room_depth)
        if room_width > 0.0 and room_depth > 0.0
        else math.inf
    )
    full_width = (
        abs(overlap[0] - room_bounds[0]) <= 1e-4
        and abs(overlap[2] - room_bounds[2]) <= 1e-4
    )
    full_depth = (
        abs(overlap[1] - room_bounds[1]) <= 1e-4
        and abs(overlap[3] - room_bounds[3]) <= 1e-4
    )
    touches_horizontal_edge = (
        abs(overlap[1] - room_bounds[1]) <= 1e-4
        or abs(overlap[3] - room_bounds[3]) <= 1e-4
    )
    touches_vertical_edge = (
        abs(overlap[0] - room_bounds[0]) <= 1e-4
        or abs(overlap[2] - room_bounds[2]) <= 1e-4
    )
    boundary_strip = (full_width and touches_horizontal_edge) or (
        full_depth and touches_vertical_edge
    )
    interior_threshold = None
    if full_width and touches_horizontal_edge:
        fixed = (
            overlap[3]
            if abs(overlap[1] - room_bounds[1]) <= 1e-4
            else overlap[1]
        )
        interior_threshold = ("horizontal", fixed, overlap[0], overlap[2])
    elif full_depth and touches_vertical_edge:
        fixed = (
            overlap[2]
            if abs(overlap[0] - room_bounds[0]) <= 1e-4
            else overlap[0]
        )
        interior_threshold = ("vertical", fixed, overlap[1], overlap[3])
    remainder = (
        room_depth - overlap_depth if full_width else room_width - overlap_width
        if full_depth
        else 0.0
    )
    return {
        "overlap_bounds": overlap,
        "overlap_area_fraction": area_fraction,
        "boundary_strip": boundary_strip,
        "interior_threshold": interior_threshold,
        "remaining_depth_or_width_m": remainder,
        "valid": (
            overlap_width > 0.0
            and overlap_depth > 0.0
            and boundary_strip
            and area_fraction <= MAXIMUM_CARVED_ROOM_AREA_FRACTION
            and remainder >= MINIMUM_CARVED_ROOM_REMAINDER
        ),
    }


def _validate_navigation_common_topology(
    layout: dict[str, Any],
    bounds: dict[str, tuple[float, float, float, float]],
    physical_connections: list[dict[str, Any]],
) -> tuple[list[str], dict[str, Any]]:
    """Validate routes that never need another classroom/restroom as transit."""

    issues: list[str] = []
    raw_zones = layout.get("navigation_common_zones", [])
    if raw_zones is None:
        raw_zones = []
    if not isinstance(raw_zones, list):
        return ["navigation_common_zones must be a list"], {
            "status": "fail",
            "common_zone_ids": sorted(CORE_COMMON_ROOM_IDS),
            "explicit_common_zones": [],
        }

    zone_records: dict[str, dict[str, Any]] = {}
    zone_bounds: dict[str, tuple[float, float, float, float]] = {}
    zone_evidence: list[dict[str, Any]] = []
    portal_ids = {
        str(door.get("id") or "")
        for door in layout.get("doors", [])
        if isinstance(door, dict) and str(door.get("id") or "")
    }
    for index, raw_zone in enumerate(raw_zones):
        if not isinstance(raw_zone, dict):
            issues.append(
                f"navigation_common_zones[{index}] must be an object"
            )
            continue
        zone_id = str(raw_zone.get("id") or "").strip()
        local_issues: list[str] = []
        if not zone_id or zone_id in EXPECTED_IDS or zone_id in zone_records:
            local_issues.append("ID is missing, duplicated, or collides with a room ID")
        candidate_bounds = _navigation_zone_bounds(raw_zone)
        width = candidate_bounds[2] - candidate_bounds[0]
        depth = candidate_bounds[3] - candidate_bounds[1]
        if (
            not _finite_bounds(candidate_bounds)
            or width < MINIMUM_COMMON_ZONE_WIDTH
            or depth < MINIMUM_COMMON_ZONE_WIDTH
        ):
            local_issues.append(
                f"bounds must be finite with width/depth >= {MINIMUM_COMMON_ZONE_WIDTH:.1f} m"
            )
        reason = str(raw_zone.get("reason") or "").strip()
        if len(reason) < 20:
            local_issues.append("safe-circulation reason must be explicit (>=20 characters)")
        carved_raw = raw_zone.get("carved_from", [])
        if not isinstance(carved_raw, list):
            local_issues.append("carved_from must be a room-ID list")
            carved = []
        else:
            carved = sorted({str(value).strip() for value in carved_raw if str(value).strip()})
            unknown = sorted(set(carved) - EXPECTED_IDS)
            if unknown:
                local_issues.append(f"carved_from contains unknown rooms {unknown}")
        overlaps = sorted(
            room_id
            for room_id, room_bounds in bounds.items()
            if _finite_bounds(candidate_bounds)
            and _finite_bounds(room_bounds)
            and _overlap_area(candidate_bounds, room_bounds) > 1e-4
        )
        if set(overlaps) != set(carved):
            local_issues.append(
                f"physical overlaps {overlaps} must exactly match carved_from {carved}"
            )
        zone_area = width * depth if math.isfinite(width) and math.isfinite(depth) else math.nan
        covered_area = sum(
            _overlap_area(candidate_bounds, bounds[room_id]) for room_id in overlaps
        )
        coverage_fraction = (
            covered_area / zone_area
            if math.isfinite(zone_area) and zone_area > 0.0
            else 0.0
        )
        if coverage_fraction < 0.999:
            local_issues.append(
                "the entire common-zone floor must be physically carved from placed rooms"
            )
        carve_records = {}
        for room_id in overlaps:
            carve = _carve_evidence(bounds[room_id], candidate_bounds)
            carve_records[room_id] = carve
            if not carve["valid"]:
                local_issues.append(
                    f"carve of {room_id} must be a <=45% boundary strip leaving >=1.5 m"
                )
        for other_id, other_bounds in zone_bounds.items():
            if _overlap_area(candidate_bounds, other_bounds) > 1e-4:
                local_issues.append(f"overlaps common zone {other_id}")
        if zone_id and zone_id not in zone_records:
            zone_records[zone_id] = raw_zone
            zone_bounds[zone_id] = candidate_bounds
        if local_issues:
            issues.extend(
                f"Invalid navigation common zone {zone_id or index}: {item}"
                for item in local_issues
            )
        zone_evidence.append(
            {
                "id": zone_id or None,
                "bounds": candidate_bounds,
                "reason": reason,
                "carved_from": carved,
                "physical_overlaps": overlaps,
                "placed_room_coverage_fraction": coverage_fraction,
                "carves": carve_records,
                "valid": not local_issues,
                "issues": local_issues,
            }
        )

    common_ids = set(CORE_COMMON_ROOM_IDS) | set(zone_records)
    valid_common_edges: list[dict[str, Any]] = []
    direct_by_room: dict[str, list[str]] = {
        room_id: [] for room_id in sorted(EXPECTED_IDS - CORE_COMMON_ROOM_IDS)
    }
    common_graph: dict[str, set[str]] = defaultdict(set)

    for record in physical_connections:
        if record.get("valid") is not True:
            continue
        first = str(record.get("room_a") or "")
        second = str(record.get("room_b") or "")
        width = record.get("width")
        if record.get("connection_type") == "door" and (
            not isinstance(width, float)
            or not math.isfinite(width)
            or width < MINIMUM_NAVIGATION_DOOR_WIDTH
        ):
            continue
        connection_id = str(record.get("source_id") or f"{first}<->{second}")
        if first in common_ids and second in common_ids:
            common_graph[first].add(second)
            common_graph[second].add(first)
            valid_common_edges.append(
                {"id": connection_id, "zone_a": first, "zone_b": second, "source": record.get("source")}
            )
        if first in common_ids and second in direct_by_room:
            direct_by_room[second].append(connection_id)
        if second in common_ids and first in direct_by_room:
            direct_by_room[first].append(connection_id)

    valid_zone_portals: list[dict[str, Any]] = []
    connected_carves: dict[str, set[str]] = {
        zone_id: set() for zone_id in zone_records
    }
    for zone_id, zone in zone_records.items():
        raw_connections = zone.get("connections")
        if not isinstance(raw_connections, list) or not raw_connections:
            issues.append(
                f"Invalid navigation common zone {zone_id}: connections must be a nonempty list"
            )
            continue
        for index, connection in enumerate(raw_connections):
            local_issues: list[str] = []
            if not isinstance(connection, dict):
                issues.append(
                    f"Invalid common-zone portal {zone_id}[{index}]: record must be an object"
                )
                continue
            other = str(
                connection.get("to")
                or connection.get("room_id")
                or connection.get("zone_id")
                or ""
            ).strip()
            portal_id = str(connection.get("id") or f"{zone_id}_to_{other}").strip()
            if not portal_id or portal_id in portal_ids:
                local_issues.append("portal ID is missing or duplicated")
            portal_ids.add(portal_id)
            if other not in EXPECTED_IDS and other not in zone_records:
                local_issues.append(f"endpoint {other!r} is unknown")
            width = _float_or_nan(connection.get("width"))
            if not math.isfinite(width) or width < MINIMUM_NAVIGATION_DOOR_WIDTH:
                local_issues.append(
                    f"width must be >= {MINIMUM_NAVIGATION_DOOR_WIDTH:.1f} m"
                )
            position = connection.get("position")
            if isinstance(position, (list, tuple)) and len(position) >= 2:
                point = (_float_or_nan(position[0]), _float_or_nan(position[1]))
            else:
                point = (math.nan, math.nan)
            if not _all_finite(point):
                local_issues.append("position must contain two finite coordinates")
            orientation = str(connection.get("orientation") or "").strip().lower()
            if orientation not in {"horizontal", "vertical"}:
                local_issues.append("orientation must be horizontal or vertical")

            if not local_issues and other in zone_records[zone_id].get("carved_from", []):
                carve = _carve_evidence(bounds[other], zone_bounds[zone_id])
                shared = carve.get("interior_threshold")
                if (
                    shared is None
                    or shared[0] != orientation
                    or not _point_in_bounds(point, bounds[other])
                    or not _point_on_threshold(point, shared, width)
                ):
                    local_issues.append(
                        "carved-room portal must fit a common-zone boundary inside that room"
                    )
                else:
                    connected_carves[zone_id].add(other)
            elif not local_issues and other in zone_records:
                local_issues.append(
                    "common-zone-to-common-zone portals are unsupported; back each zone by a real core-room threshold"
                )
            elif not local_issues and other not in CORE_COMMON_ROOM_IDS:
                local_issues.append(
                    "endpoint must be a carved room, library, or main_corridor"
                )
            elif not local_issues and other in CORE_COMMON_ROOM_IDS:
                other_bounds = (
                    zone_bounds[other] if other in zone_bounds else bounds.get(other)
                )
                shared = (
                    _shared_boundary_geometry(zone_bounds[zone_id], other_bounds)
                    if other_bounds is not None
                    else None
                )
                if (
                    shared is None
                    or shared[0] != orientation
                    or not _point_on_threshold(point, shared, width)
                ):
                    local_issues.append(
                        "endpoints do not share the declared width-preserving boundary"
                    )

            record = {
                "id": portal_id,
                "zone_a": zone_id,
                "zone_b": other,
                "width_m": width if math.isfinite(width) else None,
                "position": point if _all_finite(point) else None,
                "orientation": orientation or None,
                "valid": not local_issues,
                "issues": local_issues,
            }
            valid_zone_portals.append(record)
            if local_issues:
                issues.extend(
                    f"Invalid common-zone portal {portal_id}: {item}"
                    for item in local_issues
                )
                continue

    zone_physical_backing: dict[str, Any] = {}
    for zone_id, zone in zone_records.items():
        carved = set(zone.get("carved_from", []))
        missing_carved_portals = sorted(carved - connected_carves[zone_id])
        if missing_carved_portals:
            issues.append(
                f"Navigation common zone {zone_id} lacks a safe portal to carved rooms {missing_carved_portals}"
            )

        carved_graph: dict[str, set[str]] = defaultdict(set)
        cross_room_openings: list[dict[str, Any]] = []
        for physical in physical_connections:
            if physical.get("valid") is not True:
                continue
            first = str(physical.get("room_a") or "")
            second = str(physical.get("room_b") or "")
            if first not in carved or second not in carved:
                continue
            threshold = physical.get("physical_threshold", {})
            if not _threshold_inside_bounds(threshold, zone_bounds[zone_id]):
                continue
            carved_graph[first].add(second)
            carved_graph[second].add(first)
            cross_room_openings.append(
                {
                    "connection_id": physical.get("source_id"),
                    "room_a": first,
                    "room_b": second,
                    "threshold": threshold,
                }
            )
        if carved:
            reached_carves = _reachable(carved_graph, sorted(carved)[0])
        else:
            reached_carves = set()
        disconnected_carves = sorted(carved - reached_carves)
        if disconnected_carves:
            issues.append(
                f"Navigation common zone {zone_id} is divided by placed-room walls without real >=0.9 m openings: {disconnected_carves}"
            )

        backed_core_portals: list[dict[str, Any]] = []
        for portal in valid_zone_portals:
            if (
                portal.get("valid") is not True
                or portal.get("zone_a") != zone_id
                or portal.get("zone_b") not in CORE_COMMON_ROOM_IDS
            ):
                continue
            core_id = str(portal["zone_b"])
            candidates = []
            for physical in physical_connections:
                if physical.get("valid") is not True:
                    continue
                endpoints = {
                    str(physical.get("room_a") or ""),
                    str(physical.get("room_b") or ""),
                }
                overlapping_rooms = sorted((endpoints - {core_id}) & carved)
                if core_id not in endpoints or len(overlapping_rooms) != 1:
                    continue
                threshold = physical.get("physical_threshold", {})
                if _threshold_contains(
                    threshold,
                    portal["position"],
                    str(portal["orientation"]),
                    float(portal["width_m"]),
                ) and _threshold_inside_bounds(threshold, zone_bounds[zone_id]):
                    candidates.append((physical, overlapping_rooms[0]))
            if not candidates:
                portal["valid"] = False
                portal["issues"].append(
                    "portal is not backed by a co-located real Door/OPEN threshold from a carved room"
                )
                issues.append(
                    f"Invalid common-zone portal {portal['id']}: {portal['issues'][-1]}"
                )
                continue
            backing, backing_room = candidates[0]
            portal["backing_connection_id"] = backing.get("source_id")
            portal["backing_room_id"] = backing_room
            backed_core_portals.append(
                {
                    "portal_id": portal["id"],
                    "core_room_id": core_id,
                    "backing_room_id": backing_room,
                    "backing_connection_id": backing.get("source_id"),
                    "physical_threshold": backing.get("physical_threshold"),
                }
            )
            common_graph[zone_id].add(core_id)
            common_graph[core_id].add(zone_id)
            valid_common_edges.append(
                {
                    "id": portal["id"],
                    "zone_a": zone_id,
                    "zone_b": core_id,
                    "source": "physically_backed_navigation_common_zone",
                }
            )

        for portal in valid_zone_portals:
            if (
                portal.get("valid") is True
                and portal.get("zone_a") == zone_id
                and portal.get("zone_b") in carved
            ):
                direct_by_room[str(portal["zone_b"])].append(str(portal["id"]))

        zone_physical_backing[zone_id] = {
            "carved_rooms": sorted(carved),
            "carved_rooms_reachable_through_real_openings": sorted(reached_carves),
            "cross_room_openings_inside_zone": cross_room_openings,
            "backed_core_portals": backed_core_portals,
            "status": (
                "pass"
                if not disconnected_carves and backed_core_portals
                else "fail"
            ),
        }
        if not backed_core_portals:
            issues.append(
                f"Navigation common zone {zone_id} has no physically backed threshold to library/main_corridor"
            )

    reachable_common = _reachable(common_graph, "library")
    disconnected_common = sorted(common_ids - reachable_common)
    if disconnected_common:
        issues.append(
            "Common circulation is disconnected from the library entrance: "
            f"{disconnected_common}"
        )
    for room_id, connection_ids in direct_by_room.items():
        if not connection_ids:
            issues.append(
                f"{room_id} must have a direct >=0.9 m threshold to library, "
                "main_corridor, or an explicit connected navigation common zone"
            )

    return issues, {
        "status": "pass" if not issues else "fail",
        "policy": (
            "Every functional room has a direct common-circulation threshold; "
            "no route may require traversing another classroom, restroom, or storage room."
        ),
        "common_zone_ids": sorted(common_ids),
        "common_zones_reachable_from_library_entrance": sorted(reachable_common),
        "explicit_common_zones": zone_evidence,
        "explicit_common_zone_portals": valid_zone_portals,
        "common_zone_physical_backing": zone_physical_backing,
        "valid_common_edges": valid_common_edges,
        "direct_common_connections_by_room": {
            room_id: sorted(connection_ids)
            for room_id, connection_ids in direct_by_room.items()
        },
    }


def validate(layout: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    evidence: dict[str, Any] = {}

    raw_placed_rooms = layout.get("placed_rooms", [])
    if not isinstance(raw_placed_rooms, list):
        errors.append("placed_rooms must be a list")
        raw_placed_rooms = []
    placed_rooms = _placed_room_entries(layout)
    malformed_room_entries = len(raw_placed_rooms) - len(placed_rooms)
    if malformed_room_entries:
        errors.append(
            f"placed_rooms contains {malformed_room_entries} non-object entrie(s)"
        )

    placed_ids = [
        str(room.get("room_id") or "").strip()
        for room in placed_rooms
        if str(room.get("room_id") or "").strip()
    ]
    missing_id_entries = len(placed_rooms) - len(placed_ids)
    if missing_id_entries:
        errors.append(
            f"placed_rooms contains {missing_id_entries} room entrie(s) without room_id"
        )
    duplicate_ids = sorted(
        room_id for room_id, count in Counter(placed_ids).items() if count > 1
    )
    if duplicate_ids:
        errors.append(f"Duplicate placed room IDs are forbidden: {duplicate_ids}")

    rooms = _room_map(layout)
    actual_ids = set(rooms)
    missing = sorted(EXPECTED_IDS - actual_ids)
    unexpected = sorted(actual_ids - EXPECTED_IDS)
    if missing:
        errors.append(f"Missing required room IDs: {missing}")
    if unexpected:
        errors.append(f"Unexpected room IDs: {unexpected}")
    if len(raw_placed_rooms) != 11:
        errors.append(
            f"Expected exactly 11 placed room entries, found {len(raw_placed_rooms)}"
        )
    evidence["placed_room_entry_count"] = len(raw_placed_rooms)
    evidence["unique_room_id_count"] = len(actual_ids)
    evidence["duplicate_room_ids"] = duplicate_ids

    raw_room_specs = layout.get("rooms", [])
    if not isinstance(raw_room_specs, list):
        errors.append("rooms must be a list of the exact required RoomSpec entries")
        raw_room_specs = []
    room_specs = [item for item in raw_room_specs if isinstance(item, dict)]
    malformed_room_specs = len(raw_room_specs) - len(room_specs)
    if malformed_room_specs:
        errors.append(f"rooms contains {malformed_room_specs} non-object entrie(s)")
    room_spec_ids = [
        str(item.get("id") or item.get("room_id") or "").strip()
        for item in room_specs
        if str(item.get("id") or item.get("room_id") or "").strip()
    ]
    missing_room_spec_ids = len(room_specs) - len(room_spec_ids)
    if missing_room_spec_ids:
        errors.append(
            f"rooms contains {missing_room_spec_ids} RoomSpec entrie(s) without an ID"
        )
    duplicate_room_spec_ids = sorted(
        room_id for room_id, count in Counter(room_spec_ids).items() if count > 1
    )
    if duplicate_room_spec_ids:
        errors.append(f"Duplicate RoomSpec IDs are forbidden: {duplicate_room_spec_ids}")
    room_spec_id_set = set(room_spec_ids)
    missing_room_specs = sorted(EXPECTED_IDS - room_spec_id_set)
    unexpected_room_specs = sorted(room_spec_id_set - EXPECTED_IDS)
    if missing_room_specs:
        errors.append(f"Missing required RoomSpec IDs: {missing_room_specs}")
    if unexpected_room_specs:
        errors.append(f"Unexpected RoomSpec IDs: {unexpected_room_specs}")
    if len(raw_room_specs) != len(EXPECTED_IDS):
        errors.append(
            f"Expected exactly {len(EXPECTED_IDS)} RoomSpec entries, "
            f"found {len(raw_room_specs)}"
        )
    evidence["room_spec_entry_count"] = len(raw_room_specs)
    evidence["room_spec_ids"] = sorted(room_spec_id_set)
    evidence["duplicate_room_spec_ids"] = duplicate_room_spec_ids

    centers = {room_id: _center(room) for room_id, room in rooms.items()}
    bounds = {room_id: _bounds(room) for room_id, room in rooms.items()}
    evidence["centers"] = centers
    evidence["bounds"] = bounds
    dimension_evidence: dict[str, Any] = {}
    for room_id, (minimum, maximum) in AREA_RANGES.items():
        room = rooms.get(room_id)
        if room is None:
            continue
        width = _float_or_nan(room.get("width"))
        depth = _float_or_nan(room.get("depth"))
        position = room.get("position")
        position_values = (
            tuple(_float_or_nan(value) for value in position[:2])
            if isinstance(position, (list, tuple)) and len(position) >= 2
            else (math.nan, math.nan)
        )
        if not _all_finite((width, depth, *position_values)):
            errors.append(f"{room_id} position, width, and depth must be finite numbers")
            continue
        if width <= 0.0 or depth <= 0.0:
            errors.append(f"{room_id} must have positive width and depth")
        area = width * depth
        if not minimum <= area <= maximum:
            errors.append(
                f"{room_id} area {area:.2f} m^2 outside required range {minimum}-{maximum}"
            )

        target_width, target_depth = TARGET_DIMENSIONS[room_id]
        width_range = (
            target_width * (1.0 - DIMENSION_RELATIVE_TOLERANCE),
            target_width * (1.0 + DIMENSION_RELATIVE_TOLERANCE),
        )
        depth_range = (
            target_depth * (1.0 - DIMENSION_RELATIVE_TOLERANCE),
            target_depth * (1.0 + DIMENSION_RELATIVE_TOLERANCE),
        )
        target_aspect = target_width / target_depth
        actual_aspect = width / depth if depth > 0.0 else math.inf
        relative_aspect_error = abs(actual_aspect - target_aspect) / target_aspect
        if not width_range[0] <= width <= width_range[1]:
            errors.append(
                f"{room_id} width {width:.2f} m outside target tolerance "
                f"{width_range[0]:.2f}-{width_range[1]:.2f} m"
            )
        if not depth_range[0] <= depth <= depth_range[1]:
            errors.append(
                f"{room_id} depth {depth:.2f} m outside target tolerance "
                f"{depth_range[0]:.2f}-{depth_range[1]:.2f} m"
            )
        if relative_aspect_error > ASPECT_RATIO_RELATIVE_TOLERANCE:
            errors.append(
                f"{room_id} aspect ratio {actual_aspect:.3f} diverges from "
                f"target {target_aspect:.3f} by more than "
                f"{ASPECT_RATIO_RELATIVE_TOLERANCE:.0%}"
            )
        dimension_evidence[room_id] = {
            "width_m": width,
            "depth_m": depth,
            "area_m2": area,
            "aspect_ratio": actual_aspect,
            "target_width_m": target_width,
            "target_depth_m": target_depth,
        }
    evidence["dimensions"] = dimension_evidence

    overlapping_pairs = []
    for first_id, second_id in itertools.combinations(sorted(bounds), 2):
        if not _all_finite(bounds[first_id]) or not _all_finite(bounds[second_id]):
            continue
        area = _overlap_area(bounds[first_id], bounds[second_id])
        if area > 1e-4:
            overlapping_pairs.append(
                {"room_a": first_id, "room_b": second_id, "overlap_area_m2": area}
            )
            errors.append(
                f"Rooms overlap: {first_id} and {second_id} share {area:.2f} m^2"
            )
    evidence["overlapping_room_pairs"] = overlapping_pairs

    if "main_corridor" in rooms and _all_finite(centers["main_corridor"]):
        corridor_x, _ = centers["main_corridor"]
        left_rooms = ("classroom_01", "classroom_03", "classroom_04")
        right_rooms = ("classroom_02", "classroom_06", "classroom_05")
        for room_id in left_rooms:
            if (
                room_id in centers
                and _all_finite(centers[room_id])
                and centers[room_id][0] >= corridor_x
            ):
                errors.append(f"{room_id} must be left/west of main_corridor")
        for room_id in right_rooms:
            if (
                room_id in centers
                and _all_finite(centers[room_id])
                and centers[room_id][0] <= corridor_x
            ):
                errors.append(f"{room_id} must be right/east of main_corridor")

        if all(room_id in centers for room_id in left_rooms) and not (
            centers["classroom_01"][1]
            < centers["classroom_03"][1]
            < centers["classroom_04"][1]
        ):
            errors.append("Left classrooms must run south-to-north as 01, 03, 04")
        if all(room_id in centers for room_id in right_rooms) and not (
            centers["classroom_02"][1]
            < centers["classroom_06"][1]
            < centers["classroom_05"][1]
        ):
            errors.append("Right classrooms must run south-to-north as 02, 06, 05")
        classroom_row_alignment = {}
        for left_id, right_id in (
            ("classroom_01", "classroom_02"),
            ("classroom_03", "classroom_06"),
            ("classroom_04", "classroom_05"),
        ):
            if {left_id, right_id} <= centers.keys():
                delta = abs(centers[left_id][1] - centers[right_id][1])
                classroom_row_alignment[f"{left_id}<->{right_id}"] = delta
                if delta > CLASSROOM_ROW_ALIGNMENT_TOLERANCE:
                    errors.append(
                        f"{left_id} and {right_id} must remain aligned in the same "
                        f"reference row within {CLASSROOM_ROW_ALIGNMENT_TOLERANCE:.1f} m"
                    )
        evidence["classroom_row_alignment_delta_m"] = classroom_row_alignment
        if (
            {"library", "classroom_03"} <= centers.keys()
            and centers["library"][1] >= centers["classroom_03"][1]
        ):
            errors.append("library must remain in the lower/south central zone")
        if (
            {"boys_toilet", "library"} <= centers.keys()
            and centers["boys_toilet"][0] >= centers["library"][0]
        ):
            errors.append("boys_toilet must be left/west of library")
        if (
            {"girls_toilet", "library"} <= centers.keys()
            and centers["girls_toilet"][0] >= centers["library"][0]
        ):
            errors.append("girls_toilet must be left/west of library")
        if {"boys_toilet", "girls_toilet"} <= centers.keys():
            if centers["boys_toilet"][0] >= centers["girls_toilet"][0]:
                errors.append("boys_toilet must be left/west of girls_toilet")
            if abs(centers["boys_toilet"][1] - centers["girls_toilet"][1]) > RESTROOM_ROW_TOLERANCE:
                errors.append("boys_toilet and girls_toilet must remain in the same horizontal row")
        if {"boys_toilet", "girls_toilet"} <= bounds.keys():
            restroom_adjacency = _adjacency_evidence(
                bounds["boys_toilet"], bounds["girls_toilet"]
            )
            evidence["restroom_adjacency"] = restroom_adjacency
            if not restroom_adjacency["adjacent"]:
                errors.append("boys_toilet and girls_toilet must be directly adjacent")
        if {"boys_toilet", "girls_toilet", "library"} <= bounds.keys():
            restroom_zone = (
                min(bounds["boys_toilet"][0], bounds["girls_toilet"][0]),
                min(bounds["boys_toilet"][1], bounds["girls_toilet"][1]),
                max(bounds["boys_toilet"][2], bounds["girls_toilet"][2]),
                max(bounds["boys_toilet"][3], bounds["girls_toilet"][3]),
            )
            library_bounds = bounds["library"]
            horizontal_gap = library_bounds[0] - restroom_zone[2]
            vertical_gap = restroom_zone[1] - library_bounds[3]
            evidence["restroom_library_proximity"] = {
                "horizontal_gap_m": horizontal_gap,
                "vertical_gap_m": vertical_gap,
            }
            if not 0.0 <= horizontal_gap <= MAXIMUM_RESTROOM_LIBRARY_HORIZONTAL_GAP:
                errors.append(
                    "restroom zone must sit immediately west of library with a "
                    f"0-{MAXIMUM_RESTROOM_LIBRARY_HORIZONTAL_GAP:.1f} m horizontal gap"
                )
            if not 0.0 <= vertical_gap <= MAXIMUM_RESTROOM_LIBRARY_VERTICAL_GAP:
                errors.append(
                    "restroom zone must sit northwest of library with a "
                    f"0-{MAXIMUM_RESTROOM_LIBRARY_VERTICAL_GAP:.1f} m vertical gap"
                )
            if (
                "classroom_03" in centers
                and max(centers["boys_toilet"][1], centers["girls_toilet"][1])
                >= centers["classroom_03"][1]
            ):
                errors.append("restroom zone must remain south of classroom_03")
        if (
            {"classroom_02", "storage_room", "classroom_06"} <= centers.keys()
            and not centers["classroom_02"][1]
            < centers["storage_room"][1]
            < centers["classroom_06"][1]
        ):
            errors.append("storage_room must lie vertically between classroom_02 and classroom_06")
        if "storage_room" in centers and centers["storage_room"][0] <= corridor_x:
            errors.append("storage_room must be right/east of main_corridor")
        if {"classroom_02", "storage_room", "classroom_06"} <= bounds.keys():
            storage_south = _adjacency_evidence(
                bounds["classroom_02"], bounds["storage_room"]
            )
            storage_north = _adjacency_evidence(
                bounds["storage_room"], bounds["classroom_06"]
            )
            evidence["storage_vertical_adjacency"] = {
                "classroom_02": storage_south,
                "classroom_06": storage_north,
            }
            if not (
                storage_south["adjacent"]
                and storage_south["orientation"] == "north_south"
                and abs(bounds["classroom_02"][3] - bounds["storage_room"][1])
                <= MAXIMUM_CONNECTION_GAP
            ):
                errors.append("storage_room south wall must directly meet classroom_02")
            if not (
                storage_north["adjacent"]
                and storage_north["orientation"] == "north_south"
                and abs(bounds["storage_room"][3] - bounds["classroom_06"][1])
                <= MAXIMUM_CONNECTION_GAP
            ):
                errors.append("storage_room north wall must directly meet classroom_06")

        if "library" in bounds:
            corridor_bounds = bounds["main_corridor"]
            library_bounds = bounds["library"]
            corridor_center_x = centers["main_corridor"][0]
            library_center_x = centers["library"][0]
            if abs(library_center_x - corridor_center_x) > LIBRARY_CENTER_TOLERANCE:
                errors.append("library must be horizontally centered on the main circulation spine")
            library_corridor_adjacency = _adjacency_evidence(
                library_bounds,
                corridor_bounds,
                minimum_shared_span=0.8
                * min(
                    library_bounds[2] - library_bounds[0],
                    corridor_bounds[2] - corridor_bounds[0],
                ),
            )
            evidence["library_corridor_adjacency"] = library_corridor_adjacency
            if (
                abs(library_bounds[3] - corridor_bounds[1])
                > MAXIMUM_CONNECTION_GAP
                or not library_corridor_adjacency["adjacent"]
            ):
                errors.append(
                    "library north wall must directly meet the south end of main_corridor"
                )

    exterior_windows: dict[str, Any] = {}
    for room_id in WINDOW_REQUIRED_ROOM_IDS:
        if room_id not in rooms or room_id not in bounds:
            continue
        window_evidence = _exterior_window_evidence(
            layout, rooms, bounds, room_id
        )
        exterior_windows[room_id] = window_evidence
        if not any(item["valid_exterior_window"] for item in window_evidence):
            detail = [issue for item in window_evidence for issue in item["issues"]]
            suffix = f": {sorted(set(detail))}" if detail else ""
            errors.append(
                f"{room_id} must have at least one verified exterior-facing window{suffix}"
            )
    evidence["exterior_windows"] = exterior_windows

    graph: dict[str, set[str]] = defaultdict(set)
    physical_connections = []
    for record in _connection_records(layout):
        first = record["room_a"]
        second = record["room_b"]
        connection_type = record["connection_type"]
        connection_errors = []
        if first == second:
            connection_errors.append("self-connection is invalid")
        if first not in EXPECTED_IDS or second not in EXPECTED_IDS:
            connection_errors.append("connection endpoint is not a required room")
        if connection_type not in {"door", "open"}:
            connection_errors.append(
                f"unsupported connection type {connection_type!r}"
            )
        if connection_type == "open":
            if record.get("reciprocal") is not True:
                connection_errors.append(
                    "OPEN connection must be declared reciprocally by both RoomSpecs"
                )
            if {first, second} != CORE_COMMON_ROOM_IDS:
                connection_errors.append(
                    "functional rooms must use real doors; OPEN is restricted to "
                    "library<->main_corridor"
                )
        if record["source"] == "door" and record.get("door_type") not in {
            "",
            "interior",
        }:
            connection_errors.append("two-room door must have door_type=interior")

        width = record.get("width")
        minimum_span = max(
            MINIMUM_CONNECTION_SPAN, MINIMUM_NAVIGATION_DOOR_WIDTH
        )
        if isinstance(width, float):
            if not math.isfinite(width) or width <= 0.0:
                connection_errors.append("door width must be a positive finite number")
            elif (
                record["connection_type"] == "door"
                and width < MINIMUM_NAVIGATION_DOOR_WIDTH
            ):
                connection_errors.append(
                    f"door width must be >= {MINIMUM_NAVIGATION_DOOR_WIDTH:.1f} m "
                    "for humanoid circulation"
                )
            else:
                minimum_span = max(minimum_span, width)
        adjacency = (
            _adjacency_evidence(
                bounds[first],
                bounds[second],
                minimum_shared_span=minimum_span,
            )
            if first in bounds and second in bounds
            else {
                "adjacent": False,
                "orientation": None,
                "gap_m": None,
                "shared_span_m": 0.0,
            }
        )
        if not adjacency["adjacent"]:
            connection_errors.append(
                "declared connection does not join physically adjacent rooms"
            )
        threshold = (
            _physical_threshold_evidence(record, rooms, bounds)
            if (
                first in bounds
                and second in bounds
                and connection_type in {"door", "open"}
            )
            else {
                "valid": False,
                "orientation": None,
                "center": None,
                "width_m": None,
                "issues": ["cannot derive physical threshold"],
            }
        )
        if not threshold["valid"]:
            connection_errors.extend(threshold["issues"])
        if not connection_errors:
            graph[first].add(second)
            graph[second].add(first)
        else:
            errors.append(
                f"Invalid {record['source']} connection {first}<->{second} "
                f"({record['source_id']}): " + "; ".join(connection_errors)
            )
        serializable_record = dict(record)
        if isinstance(serializable_record.get("width"), float) and not math.isfinite(
            serializable_record["width"]
        ):
            serializable_record["width"] = None
        physical_connections.append(
            {
                **serializable_record,
                "physical_adjacency": adjacency,
                "physical_threshold": threshold,
                "valid": not connection_errors,
                "issues": connection_errors,
            }
        )
    evidence["physical_connections"] = physical_connections

    navigation_issues, navigation_evidence = _validate_navigation_common_topology(
        layout, bounds, physical_connections
    )
    errors.extend(navigation_issues)
    evidence["navigation_common_topology"] = navigation_evidence

    if "main_corridor" in rooms:
        reachable = _reachable(graph, "main_corridor")
        disconnected = sorted(EXPECTED_IDS - reachable)
        if disconnected:
            errors.append(f"Rooms disconnected from main circulation: {disconnected}")
        evidence["reachable_from_main_corridor"] = sorted(reachable)

    raw_doors = layout.get("doors", [])
    if not isinstance(raw_doors, list):
        errors.append("doors must be a list")
        raw_doors = []
    door_entries = [door for door in raw_doors if isinstance(door, dict)]
    malformed_door_entries = len(raw_doors) - len(door_entries)
    if malformed_door_entries:
        errors.append(f"doors contains {malformed_door_entries} non-object entrie(s)")
    exterior_doors = [
        door
        for door in door_entries
        if door.get("room_b") is None
        or str(door.get("door_type") or "").lower() == "exterior"
    ]
    entrance_evidence = [
        _entrance_door_evidence(door, rooms, bounds) for door in exterior_doors
    ]
    entrance_candidates = [
        item for item in entrance_evidence if item["valid"] is True
    ]
    if len(exterior_doors) != 1:
        errors.append(
            "Expected exactly one exterior door (the library main entrance), "
            f"found {len(exterior_doors)}"
        )
    for item in entrance_evidence:
        if item["issues"]:
            errors.append(
                f"Invalid exterior door {item['door_id'] or '<missing>'}: "
                + "; ".join(item["issues"])
            )
    if not entrance_candidates:
        errors.append(
            "Missing centered >=1.6 m south-facing exterior double entrance door "
            "on the library south threshold"
        )
    evidence["exterior_door_count"] = len(exterior_doors)
    evidence["exterior_door_ids"] = [door.get("id") for door in exterior_doors]
    evidence["entrance_geometry"] = entrance_evidence
    evidence["entrance_door_ids"] = [item["door_id"] for item in entrance_candidates]
    evidence["entrance_leaf_counts"] = [
        item["leaf_count"] for item in entrance_candidates
    ]
    evidence["entrance_room_id"] = "library" if entrance_candidates else None

    if layout.get("placement_valid") is not True:
        errors.append("Floor-plan placement_valid is not true")
    if layout.get("connectivity_valid") is not True:
        errors.append("Floor-plan connectivity_valid is not true")

    return {
        "status": "pass" if not errors else "fail",
        "required_room_count": 11,
        "actual_room_count": len(raw_placed_rooms),
        "actual_unique_room_count": len(actual_ids),
        "missing_room_ids": missing,
        "unexpected_room_ids": unexpected,
        "critical_issues": errors,
        "evidence": evidence,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate(_read_json(args.layout.resolve()))
    if args.output:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
