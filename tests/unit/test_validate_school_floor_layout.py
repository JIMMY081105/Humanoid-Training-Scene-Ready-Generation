from __future__ import annotations

from copy import deepcopy

from scripts.validate_school_floor_layout import EXPECTED_IDS, validate


DIRECTIONS = ("north", "south", "east", "west")


def _room(
    room_id: str,
    x: float,
    y: float,
    width: float,
    depth: float,
    *,
    exterior_directions: tuple[str, ...] = (),
) -> dict:
    wall_points = {
        "north": ([x, y + depth], [x + width, y + depth]),
        "south": ([x, y], [x + width, y]),
        "east": ([x + width, y], [x + width, y + depth]),
        "west": ([x, y], [x, y + depth]),
    }
    return {
        "room_id": room_id,
        "position": [x, y],
        "width": width,
        "depth": depth,
        "walls": [
            {
                "wall_id": f"{room_id}_{direction}",
                "room_id": room_id,
                "direction": direction,
                "start_point": wall_points[direction][0],
                "end_point": wall_points[direction][1],
                "length": width if direction in {"north", "south"} else depth,
                "is_exterior": direction in exterior_directions,
                "openings": [],
            }
            for direction in DIRECTIONS
        ],
    }


def _placed_room(layout: dict, room_id: str) -> dict:
    return next(
        room for room in layout["placed_rooms"] if room["room_id"] == room_id
    )


def _wall(layout: dict, room_id: str, direction: str) -> dict:
    room = _placed_room(layout, room_id)
    return next(wall for wall in room["walls"] if wall["direction"] == direction)


def _add_opening(
    layout: dict,
    room_id: str,
    direction: str,
    opening_id: str,
    opening_type: str,
    *,
    width: float = 1.0,
    position_along_wall: float | None = None,
    height: float | None = None,
    sill_height: float = 0.0,
) -> None:
    wall = _wall(layout, room_id, direction)
    if position_along_wall is None:
        position_along_wall = (wall["length"] - width) / 2.0
    if height is None:
        height = 2.1 if opening_type == "door" else 1.2
    wall["openings"].append(
        {
            "opening_id": opening_id,
            "opening_type": opening_type,
            "width": width,
            "position_along_wall": position_along_wall,
            "height": height,
            "sill_height": sill_height,
        }
    )


def _interior_door(room_a: str, room_b: str) -> dict:
    return {
        "id": f"door_{room_a}_{room_b}",
        "room_a": room_a,
        "room_b": room_b,
        "width": 1.0,
        "height": 2.1,
        "door_type": "interior",
        "position_segment": "center",
    }


def _attach_interior_door_openings(
    layout: dict,
    door: dict,
    *,
    center: tuple[float, float] | None = None,
) -> None:
    first = _placed_room(layout, door["room_a"])
    second = _placed_room(layout, door["room_b"])
    first_bounds = (
        first["position"][0],
        first["position"][1],
        first["position"][0] + first["width"],
        first["position"][1] + first["depth"],
    )
    second_bounds = (
        second["position"][0],
        second["position"][1],
        second["position"][0] + second["width"],
        second["position"][1] + second["depth"],
    )
    if abs(first_bounds[2] - second_bounds[0]) < 1e-6:
        directions = ("east", "west")
        start, end = max(first_bounds[1], second_bounds[1]), min(
            first_bounds[3], second_bounds[3]
        )
        center = center or (first_bounds[2], (start + end) / 2.0)
    elif abs(second_bounds[2] - first_bounds[0]) < 1e-6:
        directions = ("west", "east")
        start, end = max(first_bounds[1], second_bounds[1]), min(
            first_bounds[3], second_bounds[3]
        )
        center = center or (first_bounds[0], (start + end) / 2.0)
    elif abs(first_bounds[3] - second_bounds[1]) < 1e-6:
        directions = ("north", "south")
        start, end = max(first_bounds[0], second_bounds[0]), min(
            first_bounds[2], second_bounds[2]
        )
        center = center or ((start + end) / 2.0, first_bounds[3])
    else:
        directions = ("south", "north")
        start, end = max(first_bounds[0], second_bounds[0]), min(
            first_bounds[2], second_bounds[2]
        )
        center = center or ((start + end) / 2.0, first_bounds[1])
    for room_id, direction in zip((door["room_a"], door["room_b"]), directions):
        wall = _wall(layout, room_id, direction)
        wall_start = wall["start_point"]
        tangent = center[1] if direction in {"east", "west"} else center[0]
        wall_tangent_start = (
            wall_start[1] if direction in {"east", "west"} else wall_start[0]
        )
        position = tangent - wall_tangent_start - door["width"] / 2.0
        _add_opening(
            layout,
            room_id,
            direction,
            door["id"],
            "door",
            width=door["width"],
            position_along_wall=position,
        )
        if room_id == door["room_a"]:
            door["position_exact"] = position


def _attach_open_connection(layout: dict, room_a: str, room_b: str) -> None:
    """Mirror OpenPlanMixin's paired full-shared-edge OPEN serialization."""

    first = _placed_room(layout, room_a)
    second = _placed_room(layout, room_b)
    first_bounds = (
        first["position"][0],
        first["position"][1],
        first["position"][0] + first["width"],
        first["position"][1] + first["depth"],
    )
    second_bounds = (
        second["position"][0],
        second["position"][1],
        second["position"][0] + second["width"],
        second["position"][1] + second["depth"],
    )
    if abs(first_bounds[2] - second_bounds[0]) < 1e-6:
        directions = ("east", "west")
        tangent_start = max(first_bounds[1], second_bounds[1])
        tangent_end = min(first_bounds[3], second_bounds[3])
    elif abs(second_bounds[2] - first_bounds[0]) < 1e-6:
        directions = ("west", "east")
        tangent_start = max(first_bounds[1], second_bounds[1])
        tangent_end = min(first_bounds[3], second_bounds[3])
    elif abs(first_bounds[3] - second_bounds[1]) < 1e-6:
        directions = ("north", "south")
        tangent_start = max(first_bounds[0], second_bounds[0])
        tangent_end = min(first_bounds[2], second_bounds[2])
    else:
        directions = ("south", "north")
        tangent_start = max(first_bounds[0], second_bounds[0])
        tangent_end = min(first_bounds[2], second_bounds[2])
    width = tangent_end - tangent_start
    stem = f"open_{room_a}_{room_b}"
    for room_id, direction, opening_id in (
        (room_a, directions[0], stem),
        (room_b, directions[1], f"{stem}_b"),
    ):
        wall = _wall(layout, room_id, direction)
        tangent_wall_start = (
            wall["start_point"][1]
            if direction in {"east", "west"}
            else wall["start_point"][0]
        )
        _add_opening(
            layout,
            room_id,
            direction,
            opening_id,
            "open",
            width=width,
            position_along_wall=tangent_start - tangent_wall_start,
            height=0.0,
        )


def _valid_layout() -> dict:
    # The central spine begins at y=0. The library meets its south edge, while
    # lower classrooms occupy the southwest/southeast wings, as in the reference.
    rooms = [
        _room("classroom_01", 0, 0, 9, 7.5, exterior_directions=("west",)),
        _room("classroom_03", 0, 11.5, 9, 7.5, exterior_directions=("west",)),
        _room("classroom_04", 0, 20, 9, 7.5, exterior_directions=("west",)),
        _room("classroom_02", 21, 0, 9, 7.5, exterior_directions=("east",)),
        _room("classroom_06", 21, 11.2, 9, 7.5, exterior_directions=("east",)),
        _room("classroom_05", 21, 20, 9, 7.5, exterior_directions=("east",)),
        _room("library", 10, -9, 10, 9, exterior_directions=("south",)),
        _room("boys_toilet", 1, 7.5, 4, 4),
        _room("girls_toilet", 5, 7.5, 4, 4),
        _room("storage_room", 21, 7.5, 5, 3.7),
        _room("main_corridor", 9, 0, 12, 22.5),
    ]
    layout = {
        "placed_rooms": rooms,
        "rooms": [
            {
                "id": room["room_id"],
                "type": "room",
                "position": list(room["position"]),
                "width": room["width"],
                "length": room["depth"],
                "prompt": f"Canonical prompt for {room['room_id']}",
                "connections": {},
                "exterior_walls": [],
            }
            for room in rooms
        ],
        "doors": [
            _interior_door("main_corridor", room_id)
            for room_id in (
                "classroom_01",
                "classroom_02",
                "classroom_03",
                "classroom_04",
                "classroom_05",
                "classroom_06",
                "library",
                "girls_toilet",
                "storage_room",
            )
        ]
        + [_interior_door("girls_toilet", "boys_toilet")],
        "windows": [],
        "navigation_common_zones": [
            {
                "id": "restroom_foyer",
                "position": [1.0, 7.5],
                "width": 8.0,
                "depth": 1.2,
                "reason": (
                    "A separate public foyer gives both gendered restrooms "
                    "independent access without crossing either restroom."
                ),
                "carved_from": ["boys_toilet", "girls_toilet"],
                "connections": [
                    {
                        "id": "foyer_corridor",
                        "to": "main_corridor",
                        "width": 1.0,
                        "position": [9.0, 8.1],
                        "orientation": "vertical",
                    },
                    {
                        "id": "foyer_boys",
                        "to": "boys_toilet",
                        "width": 1.0,
                        "position": [3.0, 8.7],
                        "orientation": "horizontal",
                    },
                    {
                        "id": "foyer_girls",
                        "to": "girls_toilet",
                        "width": 1.0,
                        "position": [7.0, 8.7],
                        "orientation": "horizontal",
                    },
                ],
            }
        ],
        "placement_valid": True,
        "connectivity_valid": True,
    }

    foyer_centers = {
        frozenset({"main_corridor", "girls_toilet"}): (9.0, 8.1),
        frozenset({"girls_toilet", "boys_toilet"}): (5.0, 8.1),
    }
    for door in layout["doors"]:
        pair = frozenset({door["room_a"], door["room_b"]})
        _attach_interior_door_openings(
            layout, door, center=foyer_centers.get(pair)
        )

    entrance = {
        "id": "main_entrance",
        "room_a": "library",
        "room_b": None,
        "width": 1.8,
        "height": 2.2,
        "door_type": "exterior",
        "leaf_count": 2,
        "position_segment": "center",
        "position_exact": 4.1,
    }
    layout["doors"].append(entrance)
    _add_opening(
        layout,
        "library",
        "south",
        "main_entrance",
        "door",
        width=1.8,
        height=2.2,
    )

    window_directions = {
        "classroom_01": "west",
        "classroom_02": "east",
        "classroom_03": "west",
        "classroom_04": "west",
        "classroom_05": "east",
        "classroom_06": "east",
        "library": "south",
    }
    for room_id, direction in window_directions.items():
        window_id = f"window_{room_id}"
        layout["windows"].append(
            {
                "id": window_id,
                "room_id": room_id,
                "wall_direction": direction,
                "position_along_wall": 2.0,
                "width": 1.5,
                "height": 1.2,
                "sill_height": 0.9,
            }
        )
        _add_opening(
            layout,
            room_id,
            direction,
            window_id,
            "window",
            width=1.5,
            position_along_wall=2.0,
            height=1.2,
            sill_height=0.9,
        )
    return layout


def test_reference_layout_contract_passes_with_physical_evidence() -> None:
    result = validate(_valid_layout())
    assert result["status"] == "pass", result["critical_issues"]
    assert result["actual_room_count"] == 11
    assert result["actual_unique_room_count"] == 11
    assert result["evidence"]["entrance_room_id"] == "library"
    entrance = result["evidence"]["entrance_geometry"]
    assert len(entrance) == 1
    assert entrance[0]["valid"] is True
    assert entrance[0]["leaf_count"] == 2
    assert entrance[0]["leaf_width_m"] == 0.9
    assert result["evidence"]["library_corridor_adjacency"]["adjacent"] is True
    assert set(result["evidence"]["reachable_from_main_corridor"]) == EXPECTED_IDS
    topology = result["evidence"]["navigation_common_topology"]
    assert topology["status"] == "pass"
    assert topology["direct_common_connections_by_room"]["boys_toilet"] == [
        "foyer_boys"
    ]
    assert "girls_toilet" not in topology["direct_common_connections_by_room"][
        "boys_toilet"
    ]
    assert all(
        any(window["valid_exterior_window"] for window in windows)
        for windows in result["evidence"]["exterior_windows"].values()
    )


def test_duplicate_placed_room_id_cannot_hide_twelfth_room() -> None:
    layout = _valid_layout()
    layout["placed_rooms"].append(deepcopy(layout["placed_rooms"][0]))
    result = validate(layout)
    assert result["status"] == "fail"
    assert result["actual_room_count"] == 12
    assert result["evidence"]["duplicate_room_ids"] == ["classroom_01"]
    assert any("Duplicate placed room IDs" in issue for issue in result["critical_issues"])


def test_unused_extra_room_spec_is_rejected() -> None:
    layout = _valid_layout()
    layout["rooms"].append(
        {
            "id": "unused_bonus_room",
            "type": "room",
            "position": [100.0, 100.0],
            "width": 4.0,
            "length": 4.0,
            "prompt": "This RoomSpec was never placed.",
            "connections": {},
            "exterior_walls": [],
        }
    )

    result = validate(layout)

    assert result["status"] == "fail"
    assert result["evidence"]["room_spec_entry_count"] == 12
    assert any(
        "Unexpected RoomSpec IDs: ['unused_bonus_room']" in issue
        for issue in result["critical_issues"]
    )


def test_one_by_two_hundred_meter_corridor_is_rejected() -> None:
    layout = _valid_layout()
    corridor = _placed_room(layout, "main_corridor")
    corridor.update(position=[14.5, 0], width=1.0, depth=200.0)
    result = validate(layout)
    assert result["status"] == "fail"
    assert any("main_corridor width" in issue for issue in result["critical_issues"])
    assert any("main_corridor depth" in issue for issue in result["critical_issues"])
    assert any("main_corridor aspect ratio" in issue for issue in result["critical_issues"])


def test_dimension_tolerance_still_accepts_small_corridor_adjustment() -> None:
    layout = _valid_layout()
    corridor = _placed_room(layout, "main_corridor")
    corridor["depth"] = 22.0
    result = validate(layout)
    assert result["status"] == "pass", result["critical_issues"]


def test_aspect_ratio_is_independently_enforced() -> None:
    layout = _valid_layout()
    corridor = _placed_room(layout, "main_corridor")
    corridor.update(width=14.4, depth=16.0)
    result = validate(layout)
    assert result["status"] == "fail"
    assert any("main_corridor aspect ratio" in issue for issue in result["critical_issues"])


def test_missing_room_and_mirrored_layout_fail() -> None:
    layout = _valid_layout()
    layout["placed_rooms"] = [
        room for room in layout["placed_rooms"] if room["room_id"] != "library"
    ]
    room = _placed_room(layout, "classroom_01")
    room["position"][0] = 30
    result = validate(layout)
    assert result["status"] == "fail"
    assert "library" in result["missing_room_ids"]
    assert any("classroom_01 must be left" in issue for issue in result["critical_issues"])


def test_entrance_must_be_south_facing_double_door_on_library() -> None:
    layout = _valid_layout()
    entrance = next(door for door in layout["doors"] if door["id"] == "main_entrance")
    entrance["width"] = 1.0
    result = validate(layout)
    assert result["status"] == "fail"
    assert any("double entrance" in issue for issue in result["critical_issues"])

    one_leaf = _valid_layout()
    entrance = next(
        door for door in one_leaf["doors"] if door["id"] == "main_entrance"
    )
    entrance["leaf_count"] = 1
    result = validate(one_leaf)
    assert result["status"] == "fail"
    assert any("double entrance" in issue for issue in result["critical_issues"])

    non_native_leaf_count = _valid_layout()
    entrance = next(
        door
        for door in non_native_leaf_count["doors"]
        if door["id"] == "main_entrance"
    )
    entrance["leaf_count"] = 2.0
    result = validate(non_native_leaf_count)
    assert result["status"] == "fail"
    assert any(
        "exactly two physical leaves" in issue
        for issue in result["critical_issues"]
    )


def test_entrance_record_cannot_forge_centered_or_fitting_opening_geometry() -> None:
    off_center = _valid_layout()
    entrance = next(
        door for door in off_center["doors"] if door["id"] == "main_entrance"
    )
    opening = next(
        opening
        for opening in _wall(off_center, "library", "south")["openings"]
        if opening["opening_id"] == "main_entrance"
    )
    entrance["position_exact"] = 0.0
    opening["position_along_wall"] = 0.0
    result = validate(off_center)
    assert result["status"] == "fail"
    assert any(
        "opening center is outside the generated center segment" in issue
        for issue in result["critical_issues"]
    )

    outside_wall = _valid_layout()
    entrance = next(
        door for door in outside_wall["doors"] if door["id"] == "main_entrance"
    )
    opening = next(
        opening
        for opening in _wall(outside_wall, "library", "south")["openings"]
        if opening["opening_id"] == "main_entrance"
    )
    entrance["position_exact"] = 9.0
    opening["position_along_wall"] = 9.0
    result = validate(outside_wall)
    assert result["status"] == "fail"
    assert any(
        "opening does not fit inside its wall segment" in issue
        for issue in result["critical_issues"]
    )

    width_mismatch = _valid_layout()
    opening = next(
        opening
        for opening in _wall(width_mismatch, "library", "south")["openings"]
        if opening["opening_id"] == "main_entrance"
    )
    opening["width"] = 1.6
    result = validate(width_mismatch)
    assert result["status"] == "fail"
    assert any(
        "record and wall opening width disagree" in issue
        for issue in result["critical_issues"]
    )


def test_unexpected_additional_exterior_door_is_rejected() -> None:
    layout = _valid_layout()
    layout["doors"].append(
        {
            "id": "unexpected_exterior_exit",
            "room_a": "classroom_04",
            "room_b": None,
            "width": 1.0,
            "height": 2.1,
            "door_type": "exterior",
            "leaf_count": 1,
            "position_segment": "center",
            "position_exact": 3.25,
        }
    )
    _add_opening(
        layout,
        "classroom_04",
        "west",
        "unexpected_exterior_exit",
        "door",
        width=1.0,
        position_along_wall=3.25,
    )

    result = validate(layout)

    assert result["status"] == "fail"
    assert result["evidence"]["exterior_door_count"] == 2
    assert any(
        "Expected exactly one exterior door" in issue
        for issue in result["critical_issues"]
    )


def test_corridor_door_cannot_substitute_for_reference_library_entrance() -> None:
    layout = _valid_layout()
    entrance = next(door for door in layout["doors"] if door["id"] == "main_entrance")
    entrance["room_a"] = "main_corridor"
    _wall(layout, "library", "south")["openings"] = []
    _add_opening(
        layout, "main_corridor", "south", "main_entrance", "door", width=1.8
    )
    result = validate(layout)
    assert result["status"] == "fail"
    assert any("library south threshold" in issue for issue in result["critical_issues"])


def test_library_must_physically_meet_corridor_south_end() -> None:
    layout = _valid_layout()
    _placed_room(layout, "library")["position"][1] = -9.5
    result = validate(layout)
    assert result["status"] == "fail"
    assert any(
        "library north wall must directly meet" in issue
        for issue in result["critical_issues"]
    )


def test_room_overlap_fails_even_when_upstream_flag_is_true() -> None:
    layout = _valid_layout()
    _placed_room(layout, "storage_room")["position"] = [21, 6]
    result = validate(layout)
    assert result["status"] == "fail"
    assert any("Rooms overlap" in issue for issue in result["critical_issues"])
    assert result["evidence"]["overlapping_room_pairs"]


def test_toilets_must_be_adjacent_and_in_left_to_right_order() -> None:
    layout = _valid_layout()
    boys = _placed_room(layout, "boys_toilet")
    girls = _placed_room(layout, "girls_toilet")
    boys["position"], girls["position"] = girls["position"], boys["position"]
    result = validate(layout)
    assert result["status"] == "fail"
    assert any(
        "boys_toilet must be left/west of girls_toilet" in issue
        for issue in result["critical_issues"]
    )

    separated = _valid_layout()
    _placed_room(separated, "girls_toilet")["position"] = [5, 12.0]
    result = validate(separated)
    assert result["status"] == "fail"
    assert any("must be directly adjacent" in issue for issue in result["critical_issues"])


def test_reference_rows_and_service_room_proximity_are_geometric() -> None:
    misaligned_row = _valid_layout()
    _placed_room(misaligned_row, "classroom_05")["position"][1] += 2.0
    result = validate(misaligned_row)
    assert result["status"] == "fail"
    assert any(
        "classroom_04 and classroom_05 must remain aligned" in issue
        for issue in result["critical_issues"]
    )

    distant_restrooms = _valid_layout()
    for room_id in ("boys_toilet", "girls_toilet"):
        _placed_room(distant_restrooms, room_id)["position"][0] -= 3.0
    result = validate(distant_restrooms)
    assert result["status"] == "fail"
    assert any(
        "restroom zone must sit immediately west of library" in issue
        for issue in result["critical_issues"]
    )

    detached_storage = _valid_layout()
    _placed_room(detached_storage, "storage_room")["position"][0] = 29.5
    result = validate(detached_storage)
    assert result["status"] == "fail"
    assert any(
        "storage_room south wall must directly meet classroom_02" in issue
        for issue in result["critical_issues"]
    )


def test_library_and_entrance_must_be_centered() -> None:
    layout = _valid_layout()
    _placed_room(layout, "library")["position"] = [8.5, -9]
    entrance = next(door for door in layout["doors"] if door["id"] == "main_entrance")
    entrance["position_segment"] = "left"
    result = validate(layout)
    assert result["status"] == "fail"
    assert any(
        "library must be horizontally centered" in issue
        for issue in result["critical_issues"]
    )
    assert any("Missing centered" in issue for issue in result["critical_issues"])


def test_window_record_on_interior_wall_does_not_count_as_exterior() -> None:
    layout = _valid_layout()
    window = next(
        item for item in layout["windows"] if item["room_id"] == "classroom_01"
    )
    _wall(layout, "classroom_01", "west")["openings"] = []
    window["wall_direction"] = "east"
    # Even forged wall metadata cannot override the adjacent corridor geometry.
    _wall(layout, "classroom_01", "east")["is_exterior"] = True
    _add_opening(
        layout,
        "classroom_01",
        "east",
        window["id"],
        "window",
        width=window["width"],
    )
    result = validate(layout)
    assert result["status"] == "fail"
    assert any(
        "classroom_01 must have at least one verified exterior-facing window" in issue
        for issue in result["critical_issues"]
    )
    evidence = result["evidence"]["exterior_windows"]["classroom_01"][0]
    assert any("physically blocked" in issue for issue in evidence["issues"])


def test_window_dimensions_must_be_finite_and_match_generated_opening() -> None:
    nonfinite = _valid_layout()
    window = next(
        item for item in nonfinite["windows"] if item["room_id"] == "classroom_01"
    )
    opening = next(
        opening
        for opening in _wall(nonfinite, "classroom_01", "west")["openings"]
        if opening["opening_id"] == window["id"]
    )
    window["width"] = float("nan")
    opening["width"] = float("nan")
    result = validate(nonfinite)
    assert result["status"] == "fail"
    issues = result["evidence"]["exterior_windows"]["classroom_01"][0]["issues"]
    assert "window width must be a positive finite number" in issues
    assert "opening width must be a positive finite number" in issues

    mismatch = _valid_layout()
    window = next(
        item for item in mismatch["windows"] if item["room_id"] == "classroom_01"
    )
    window["sill_height"] = 0.5
    result = validate(mismatch)
    assert result["status"] == "fail"
    issues = result["evidence"]["exterior_windows"]["classroom_01"][0]["issues"]
    assert "window record and wall opening sill height disagree" in issues


def test_declared_door_requires_physical_room_adjacency() -> None:
    layout = _valid_layout()
    _placed_room(layout, "classroom_03")["position"][0] = -1.0
    result = validate(layout)
    assert result["status"] == "fail"
    assert any(
        "main_corridor<->classroom_03" in issue
        and "physically adjacent" in issue
        for issue in result["critical_issues"]
    )
    assert "classroom_03" not in result["evidence"]["reachable_from_main_corridor"]


def test_open_room_spec_connection_counts_when_rooms_are_physically_adjacent() -> None:
    layout = _valid_layout()
    layout["doors"] = [
        door
        for door in layout["doors"]
        if not (
            door.get("room_a") == "main_corridor"
            and door.get("room_b") == "library"
        )
    ]
    layout["rooms"] = [
        {
            "id": room_id,
            "connections": (
                {"library": "OPEN"}
                if room_id == "main_corridor"
                else {"main_corridor": "OPEN"}
                if room_id == "library"
                else {}
            ),
        }
        for room_id in sorted(EXPECTED_IDS)
    ]
    _attach_open_connection(layout, "library", "main_corridor")
    result = validate(layout)
    assert result["status"] == "pass", result["critical_issues"]
    assert "library" in result["evidence"]["reachable_from_main_corridor"]
    assert any(
        record["connection_type"] == "open" and record["valid"]
        for record in result["evidence"]["physical_connections"]
    )


def test_open_connection_requires_reciprocal_generated_wall_gap() -> None:
    def layout_with_open_gap() -> dict:
        layout = _valid_layout()
        layout["doors"] = [
            door
            for door in layout["doors"]
            if {door.get("room_a"), door.get("room_b")}
            != {"main_corridor", "library"}
        ]
        for room_spec in layout["rooms"]:
            if room_spec["id"] == "main_corridor":
                room_spec["connections"] = {"library": "OPEN"}
            elif room_spec["id"] == "library":
                room_spec["connections"] = {"main_corridor": "OPEN"}
        _attach_open_connection(layout, "library", "main_corridor")
        return layout

    one_sided = layout_with_open_gap()
    _wall(one_sided, "main_corridor", "south")["openings"] = [
        opening
        for opening in _wall(one_sided, "main_corridor", "south")["openings"]
        if opening["opening_type"] != "open"
    ]
    result = validate(one_sided)
    assert result["status"] == "fail"
    assert any(
        "reciprocal co-located generated wall cutouts" in issue
        for issue in result["critical_issues"]
    )

    unrelated_ids = layout_with_open_gap()
    for room_id, replacement in (
        ("library", "forged_gap_a"),
        ("main_corridor", "forged_gap_b"),
    ):
        opening = next(
            opening
            for wall in _placed_room(unrelated_ids, room_id)["walls"]
            for opening in wall["openings"]
            if opening["opening_type"] == "open"
        )
        opening["opening_id"] = replacement
    result = validate(unrelated_ids)
    assert result["status"] == "fail"
    assert any(
        "reciprocal co-located generated wall cutouts" in issue
        for issue in result["critical_issues"]
    )


def test_nonadjacent_open_room_spec_connection_is_rejected() -> None:
    layout = _valid_layout()
    layout["rooms"] = [
        {
            "id": "classroom_01",
            "connections": {"classroom_05": "OPEN"},
        }
    ]
    result = validate(layout)
    assert result["status"] == "fail"
    assert any(
        "classroom_01<->classroom_05" in issue
        and "physically adjacent" in issue
        for issue in result["critical_issues"]
    )


def test_indirect_boys_restroom_route_is_rejected_before_room_generation() -> None:
    layout = _valid_layout()
    layout.pop("navigation_common_zones")

    result = validate(layout)

    assert result["status"] == "fail"
    assert any(
        "boys_toilet must have a direct >=0.9 m threshold" in issue
        for issue in result["critical_issues"]
    )
    topology = result["evidence"]["navigation_common_topology"]
    assert topology["direct_common_connections_by_room"]["boys_toilet"] == []
    assert topology["direct_common_connections_by_room"]["girls_toilet"] == [
        "door_main_corridor_girls_toilet"
    ]


def test_explicit_foyer_must_connect_independently_to_both_restrooms() -> None:
    layout = _valid_layout()
    foyer = layout["navigation_common_zones"][0]
    foyer["connections"] = [
        connection
        for connection in foyer["connections"]
        if connection["to"] != "boys_toilet"
    ]

    result = validate(layout)

    assert result["status"] == "fail"
    assert any(
        "lacks a safe portal to carved rooms ['boys_toilet']" in issue
        for issue in result["critical_issues"]
    )
    assert any(
        "boys_toilet must have a direct >=0.9 m threshold" in issue
        for issue in result["critical_issues"]
    )


def test_common_foyer_cannot_be_a_forged_overlap_or_exterior_exit() -> None:
    oversized = _valid_layout()
    foyer = oversized["navigation_common_zones"][0]
    foyer["depth"] = 2.0
    # Keep portal positions on the new internal edge so the excessive carve is
    # the independent reason this plan fails.
    for connection in foyer["connections"]:
        if connection["to"] in {"boys_toilet", "girls_toilet"}:
            connection["position"][1] = 9.5
    result = validate(oversized)
    assert result["status"] == "fail"
    assert any(
        "must be a <=45% boundary strip" in issue
        for issue in result["critical_issues"]
    )

    exterior_exit = _valid_layout()
    foyer = exterior_exit["navigation_common_zones"][0]
    boys = next(
        connection
        for connection in foyer["connections"]
        if connection["to"] == "boys_toilet"
    )
    boys["position"][1] = 7.5
    result = validate(exterior_exit)
    assert result["status"] == "fail"
    assert any(
        "carved-room portal must fit" in issue
        for issue in result["critical_issues"]
    )


def test_common_foyer_schema_and_portal_width_fail_closed() -> None:
    layout = _valid_layout()
    foyer = layout["navigation_common_zones"][0]
    foyer["reason"] = "foyer"
    foyer["carved_from"] = ["boys_toilet"]
    foyer["connections"][0]["width"] = 0.8
    foyer["connections"][0]["position"] = [8.0, 8.1]

    result = validate(layout)

    assert result["status"] == "fail"
    assert any("reason must be explicit" in issue for issue in result["critical_issues"])
    assert any("physical overlaps" in issue for issue in result["critical_issues"])
    assert any("width must be >= 0.9 m" in issue for issue in result["critical_issues"])


def test_every_classroom_and_storage_room_also_needs_direct_common_access() -> None:
    layout = _valid_layout()
    layout["doors"] = [
        door
        for door in layout["doors"]
        if door.get("room_b") != "classroom_05"
    ]

    result = validate(layout)

    assert result["status"] == "fail"
    assert any(
        "classroom_05 must have a direct >=0.9 m threshold" in issue
        for issue in result["critical_issues"]
    )


def test_sub_90cm_interior_door_is_not_navigation_evidence() -> None:
    layout = _valid_layout()
    door = next(
        door for door in layout["doors"] if door.get("room_b") == "storage_room"
    )
    door["width"] = 0.8

    result = validate(layout)

    assert result["status"] == "fail"
    assert any(
        "door width must be >= 0.9 m" in issue
        for issue in result["critical_issues"]
    )
    assert any(
        "storage_room must have a direct >=0.9 m threshold" in issue
        for issue in result["critical_issues"]
    )


def test_foyer_core_portal_requires_colocated_real_door_geometry() -> None:
    layout = _valid_layout()
    door = next(
        door
        for door in layout["doors"]
        if {door.get("room_a"), door.get("room_b")}
        == {"main_corridor", "girls_toilet"}
    )
    # Move the real door to y=9.5, outside the y=7.5..8.7 foyer strip,
    # while leaving the claimed foyer portal at y=8.1.
    for room_id, position in (("main_corridor", 9.0), ("girls_toilet", 1.5)):
        opening = next(
            opening
            for wall in _placed_room(layout, room_id)["walls"]
            for opening in wall["openings"]
            if opening["opening_id"] == door["id"]
        )
        opening["position_along_wall"] = position

    result = validate(layout)

    assert result["status"] == "fail"
    assert any(
        "not backed by a co-located real Door/OPEN threshold" in issue
        for issue in result["critical_issues"]
    )
    backing = result["evidence"]["navigation_common_topology"][
        "common_zone_physical_backing"
    ]["restroom_foyer"]
    assert backing["backed_core_portals"] == []


def test_foyer_cannot_cross_room_divider_without_real_opening_in_strip() -> None:
    layout = _valid_layout()
    door = next(
        door
        for door in layout["doors"]
        if {door.get("room_a"), door.get("room_b")}
        == {"girls_toilet", "boys_toilet"}
    )
    # Keep the paired openings co-located but move them north of the foyer.
    for room_id in ("girls_toilet", "boys_toilet"):
        opening = next(
            opening
            for wall in _placed_room(layout, room_id)["walls"]
            for opening in wall["openings"]
            if opening["opening_id"] == door["id"]
        )
        opening["position_along_wall"] = 1.5

    result = validate(layout)

    assert result["status"] == "fail"
    assert any(
        "divided by placed-room walls without real >=0.9 m openings" in issue
        for issue in result["critical_issues"]
    )


def test_door_metadata_without_two_generated_wall_openings_fails() -> None:
    layout = _valid_layout()
    door = next(
        door for door in layout["doors"] if door.get("room_b") == "classroom_02"
    )
    for wall in _placed_room(layout, "classroom_02")["walls"]:
        wall["openings"] = [
            opening
            for opening in wall["openings"]
            if opening["opening_id"] != door["id"]
        ]

    result = validate(layout)

    assert result["status"] == "fail"
    assert any(
        "door lacks matching wall openings on both rooms" in issue
        for issue in result["critical_issues"]
    )
