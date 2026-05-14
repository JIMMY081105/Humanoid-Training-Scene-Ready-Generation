from __future__ import annotations

import hashlib
import json

from copy import deepcopy
from pathlib import Path

import pytest

from scripts.validate_school_navigation import (
    EXPECTED_ROOM_IDS,
    ValidationError,
    evaluate,
    main,
    verify_output,
)


DIRECTIONS = ("north", "south", "east", "west")


def _room(room_id: str, x: float, y: float, width: float, depth: float) -> dict:
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
                "openings": [],
                "is_exterior": direction == "south" and room_id == "library",
            }
            for direction in DIRECTIONS
        ],
    }


def _room_map(layout: dict) -> dict[str, dict]:
    return {room["room_id"]: room for room in layout["placed_rooms"]}


def _wall(layout: dict, room_id: str, direction: str) -> dict:
    return next(
        wall
        for wall in _room_map(layout)[room_id]["walls"]
        if wall["direction"] == direction
    )


def _opening(
    layout: dict,
    room_id: str,
    direction: str,
    door_id: str,
    width: float,
    *,
    position_along_wall: float | None = None,
) -> None:
    wall = _wall(layout, room_id, direction)
    if position_along_wall is None:
        position_along_wall = (wall["length"] - width) / 2.0
    wall["openings"].append(
        {
            "opening_id": door_id,
            "opening_type": "DOOR",
            "position_along_wall": position_along_wall,
            "width": width,
            "height": 2.1,
        }
    )


def _door(
    layout: dict,
    room_a: str,
    room_b: str,
    direction_a: str,
    direction_b: str,
    *,
    width: float = 1.1,
    center: tuple[float, float] | None = None,
) -> None:
    door_id = f"door_{room_a}_{room_b}"
    layout["doors"].append(
        {
            "id": door_id,
            "room_a": room_a,
            "room_b": room_b,
            "door_type": "interior",
            "position_segment": "center",
            "width": width,
            "height": 2.1,
            "leaf_count": 1,
        }
    )
    if center is None:
        first, second = _room_map(layout)[room_a], _room_map(layout)[room_b]
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
        if direction_a in {"east", "west"}:
            center = (
                first_bounds[2] if direction_a == "east" else first_bounds[0],
                (
                    max(first_bounds[1], second_bounds[1])
                    + min(first_bounds[3], second_bounds[3])
                )
                / 2.0,
            )
        else:
            center = (
                (
                    max(first_bounds[0], second_bounds[0])
                    + min(first_bounds[2], second_bounds[2])
                )
                / 2.0,
                first_bounds[3] if direction_a == "north" else first_bounds[1],
            )
    for room_id, direction in ((room_a, direction_a), (room_b, direction_b)):
        wall = _wall(layout, room_id, direction)
        tangent = center[1] if direction in {"east", "west"} else center[0]
        tangent_start = (
            wall["start_point"][1]
            if direction in {"east", "west"}
            else wall["start_point"][0]
        )
        _opening(
            layout,
            room_id,
            direction,
            door_id,
            width,
            position_along_wall=tangent - tangent_start - width / 2.0,
        )


def _layout() -> dict:
    layout = {
        "placed_rooms": [
            _room("classroom_01", 0, 0, 9, 7.5),
            _room("classroom_03", 0, 11.5, 9, 7.5),
            _room("classroom_04", 0, 20, 9, 7.5),
            _room("classroom_02", 21, 0, 9, 7.5),
            _room("classroom_06", 21, 11.2, 9, 7.5),
            _room("classroom_05", 21, 20, 9, 7.5),
            _room("library", 10, -9, 10, 9),
            _room("boys_toilet", 1, 7.5, 4, 4),
            _room("girls_toilet", 5, 7.5, 4, 4),
            _room("storage_room", 21, 7.5, 5, 3.7),
            _room("main_corridor", 9, 0, 12, 22.5),
        ],
        "doors": [],
        "navigation_common_zones": [
            {
                "id": "restroom_foyer",
                "position": [1.0, 7.5],
                "width": 8.0,
                "depth": 1.2,
                "reason": (
                    "A separately modelled public foyer gives each gendered restroom "
                    "its own route without entering the other restroom."
                ),
                "carved_from": ["boys_toilet", "girls_toilet"],
                "connections": [
                    {
                        "id": "foyer_corridor",
                        "to": "main_corridor",
                        "width": 1.1,
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
    }
    for room_id, direction_a, direction_b in (
        ("classroom_01", "west", "east"),
        ("classroom_02", "east", "west"),
        ("classroom_03", "west", "east"),
        ("classroom_04", "west", "east"),
        ("classroom_05", "east", "west"),
        ("classroom_06", "east", "west"),
        ("storage_room", "east", "west"),
    ):
        _door(layout, "main_corridor", room_id, direction_a, direction_b)
    _door(layout, "main_corridor", "library", "south", "north", width=1.8)
    # These legacy physical edges exist, but the proof must not use the girls'
    # room as a passage to the boys' room; both use restroom_foyer instead.
    _door(
        layout,
        "main_corridor",
        "girls_toilet",
        "west",
        "east",
        center=(9.0, 8.1),
    )
    _door(
        layout,
        "girls_toilet",
        "boys_toilet",
        "west",
        "east",
        center=(5.0, 8.1),
    )

    entrance = {
        "id": "main_entrance",
        "room_a": "library",
        "room_b": None,
        "door_type": "exterior",
        "position_segment": "center",
        "width": 1.8,
        "height": 2.2,
        "leaf_count": 2,
    }
    layout["doors"].append(entrance)
    _opening(layout, "library", "south", "main_entrance", 1.8)
    return layout


def _object(
    *,
    translation=(0.0, 0.0, 0.5),
    bbox_min=(-0.2, -0.2, -0.5),
    bbox_max=(0.2, 0.2, 0.5),
) -> dict:
    return {
        "object_id": "fixture",
        "object_type": "FURNITURE",
        "name": "fixture",
        "transform": {
            "translation": list(translation),
            "rotation_wxyz": [1.0, 0.0, 0.0, 0.0],
        },
        "bbox_min": list(bbox_min),
        "bbox_max": list(bbox_max),
    }


def _write_scene(tmp_path: Path, layout: dict | None = None) -> Path:
    scene = tmp_path / "scene_000"
    scene.mkdir(parents=True)
    layout = deepcopy(layout or _layout())
    (scene / "house_layout.json").write_text(json.dumps(layout), encoding="utf-8")
    rooms = {}
    for room_id in EXPECTED_ROOM_IDS:
        state = {"room_geometry": {}, "objects": {}, "text_description": room_id}
        rooms[room_id] = deepcopy(state)
        path = (
            scene
            / f"room_{room_id}"
            / "scene_states"
            / "final_scene"
            / "scene_state.json"
        )
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(state), encoding="utf-8")
    combined = scene / "combined_house"
    combined.mkdir()
    (combined / "house_state.json").write_text(
        json.dumps({"layout": layout, "rooms": rooms}), encoding="utf-8"
    )
    return scene


def _replace_room_objects(scene: Path, room_id: str, objects: dict) -> None:
    final_path = (
        scene
        / f"room_{room_id}"
        / "scene_states"
        / "final_scene"
        / "scene_state.json"
    )
    final = json.loads(final_path.read_text())
    final["objects"] = objects
    final_path.write_text(json.dumps(final), encoding="utf-8")
    house_path = scene / "combined_house" / "house_state.json"
    house = json.loads(house_path.read_text())
    house["rooms"][room_id]["objects"] = deepcopy(objects)
    house_path.write_text(json.dumps(house), encoding="utf-8")


def test_full_school_navigation_passes_with_hash_bound_routes_and_turning_areas(
    tmp_path: Path,
) -> None:
    scene = _write_scene(tmp_path)
    # Exercise dict translations, nested bounding boxes, and xyzw rotation.
    robust_fixture = {
        "object_type": "FURNITURE",
        "transform": {
            "translation": {"x": 1.5, "y": 0.0, "z": 0.5},
            "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "aabb": {"min": [-0.2, -0.2, -0.5], "max": [0.2, 0.2, 0.5]},
    }
    _replace_room_objects(scene, "storage_room", {"robust_fixture": robust_fixture})

    result = evaluate(scene)

    assert result["status"] == "pass", result["critical_issues"]
    assert set(result["routes"]) == set(EXPECTED_ROOM_IDS)
    assert all(route["status"] == "pass" for route in result["routes"].values())
    assert all(route["route_sha256"] for route in result["routes"].values())
    assert result["routes"]["boys_toilet"]["traversed_zones"] == [
        "library",
        "main_corridor",
        "restroom_foyer",
        "boys_toilet",
    ]
    assert "girls_toilet" not in result["routes"]["boys_toilet"]["traversed_zones"]
    assert all(record["status"] == "pass" for record in result["turning_areas"].values())
    assert result["attestation"]["all_inputs_hashed"] is True
    assert result["attestation"]["all_routes_hashed"] is True


def test_gendered_restroom_cannot_be_used_as_an_indirect_corridor(tmp_path: Path) -> None:
    layout = _layout()
    layout.pop("navigation_common_zones")
    scene = _write_scene(tmp_path, layout)

    result = evaluate(scene)

    assert result["status"] == "fail"
    assert any("boys_toilet lacks a direct threshold" in issue for issue in result["critical_issues"])
    assert result["routes"]["boys_toilet"]["status"] == "fail"
    assert "girls_toilet" not in result["routes"]["boys_toilet"]["allowed_zones"]


def test_narrow_door_or_single_leaf_entrance_fails_closed(tmp_path: Path) -> None:
    narrow = _layout()
    door = next(item for item in narrow["doors"] if item["room_b"] == "classroom_01")
    door["width"] = 0.8
    opening_id = door["id"]
    for room_id in ("main_corridor", "classroom_01"):
        record = next(
            opening
            for wall in _room_map(narrow)[room_id]["walls"]
            for opening in wall["openings"]
            if opening["opening_id"] == opening_id
        )
        record["width"] = 0.8
    with pytest.raises(ValidationError, match="below 0.9 m"):
        evaluate(_write_scene(tmp_path / "narrow", narrow))

    one_leaf = _layout()
    next(item for item in one_leaf["doors"] if item["room_b"] is None)["leaf_count"] = 1
    with pytest.raises(ValidationError, match="exactly two leaves"):
        evaluate(_write_scene(tmp_path / "leaf", one_leaf))


def test_inflated_object_aabb_blocks_corridor_routes(tmp_path: Path) -> None:
    scene = _write_scene(tmp_path)
    # Corridor-local x spans -6..6.  This low barrier covers its complete
    # width near y=10, leaving cells on each side but no cross-corridor route.
    barrier = _object(
        translation=(0.0, 0.0, 0.6),
        bbox_min=(-6.0, -0.25, -0.6),
        bbox_max=(6.0, 0.25, 0.6),
    )
    _replace_room_objects(scene, "main_corridor", {"barrier": barrier})

    result = evaluate(scene)

    assert result["status"] == "fail"
    assert result["occupancy"]["obstacle_count"] == 1
    assert result["routes"]["classroom_04"]["status"] == "fail"
    assert any("classroom_04" in issue for issue in result["critical_issues"])


def test_missing_bbox_and_stale_combined_state_are_rejected(tmp_path: Path) -> None:
    scene = _write_scene(tmp_path / "bbox")
    _replace_room_objects(
        scene,
        "storage_room",
        {"bad": {"object_type": "FURNITURE", "transform": {"translation": [0, 0, 0]}}},
    )
    with pytest.raises(ValidationError, match="bbox_min"):
        evaluate(scene)

    stale = _write_scene(tmp_path / "stale")
    final_path = (
        stale
        / "room_library"
        / "scene_states"
        / "final_scene"
        / "scene_state.json"
    )
    final = json.loads(final_path.read_text())
    final["objects"]["new"] = _object()
    final_path.write_text(json.dumps(final), encoding="utf-8")
    with pytest.raises(ValidationError, match="does not match final state"):
        evaluate(stale)


def test_verify_only_recomputes_inputs_routes_and_self_attestation(tmp_path: Path) -> None:
    scene = _write_scene(tmp_path)
    output = tmp_path / "navigation.json"
    result = evaluate(scene)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    verification_output = tmp_path / "navigation_verification.json"
    verified = verify_output(
        output,
        scene,
        verification_output=verification_output,
    )
    assert verified["status"] == "pass"
    assert verified["routes_sha256"] == result["routes_sha256"]
    assert verified["navigation_recomputed"] is True
    assert verified["source_gate"]["sha256"] == hashlib.sha256(
        output.read_bytes()
    ).hexdigest()
    assert verified["recomputed_gate"]["route_count"] == 11
    assert json.loads(verification_output.read_text(encoding="utf-8")) == verified
    assert not list(tmp_path.glob(".navigation_verification.json.*.tmp"))

    state_path = (
        scene
        / "room_storage_room"
        / "scene_states"
        / "final_scene"
        / "scene_state.json"
    )
    state = json.loads(state_path.read_text())
    state["text_description"] = "mutated after proof"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(ValidationError, match="stale or its inputs/routes changed"):
        verify_output(output, scene)

    failed_receipt = tmp_path / "failed_navigation_verification.json"
    assert main(
        [
            "--scene-dir",
            str(scene),
            "--output",
            str(output),
            "--verify-only",
            "--verification-output",
            str(failed_receipt),
        ]
    ) == 2
    failed = json.loads(failed_receipt.read_text(encoding="utf-8"))
    assert failed["status"] == "fail"
    assert failed["navigation_recomputed"] is False


def test_physical_open_plan_library_corridor_threshold_is_routable(tmp_path: Path) -> None:
    layout = _layout()
    door = next(
        item
        for item in layout["doors"]
        if {item.get("room_a"), item.get("room_b")} == {"library", "main_corridor"}
    )
    layout["doors"].remove(door)
    for room_id in ("library", "main_corridor"):
        for wall in _room_map(layout)[room_id]["walls"]:
            wall["openings"] = [
                opening
                for opening in wall["openings"]
                if opening["opening_id"] != door["id"]
            ]
    layout["rooms"] = [
        {
            "id": "library",
            "connections": {"main_corridor": "OPEN"},
        },
        {
            "id": "main_corridor",
            "connections": {"library": "OPEN"},
        },
    ]

    result = evaluate(_write_scene(tmp_path, layout))

    assert result["status"] == "pass", result["critical_issues"]
    assert any(
        portal["source"] == "layout_open_connection"
        and {portal["zone_a"], portal["zone_b"]}
        == {"library", "main_corridor"}
        for portal in result["topology"]["portals"]
    )


def test_common_zone_metadata_cannot_forge_missing_physical_openings(
    tmp_path: Path,
) -> None:
    layout = _layout()
    door = next(
        item
        for item in layout["doors"]
        if {item.get("room_a"), item.get("room_b")}
        == {"main_corridor", "girls_toilet"}
    )
    # Move the real door north of the claimed foyer portal/strip.
    for room_id, position in (("main_corridor", 8.95), ("girls_toilet", 1.45)):
        opening = next(
            opening
            for wall in _room_map(layout)[room_id]["walls"]
            for opening in wall["openings"]
            if opening["opening_id"] == door["id"]
        )
        opening["position_along_wall"] = position

    with pytest.raises(
        ValidationError, match="lacks a co-located real Door/OPEN threshold"
    ):
        evaluate(_write_scene(tmp_path, layout))
