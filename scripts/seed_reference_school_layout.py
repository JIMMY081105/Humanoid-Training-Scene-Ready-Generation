#!/usr/bin/env python3
"""Build the exact reference-school floor plan with native SceneSmith geometry.

This is a deterministic replacement for stochastic LLM room placement for the
single, immutable ``school_reference_20260710`` production profile.  It does
not fabricate a JSON preview: the script constructs SceneSmith ``RoomSpec``,
``PlacedRoom``, ``Door``, and ``Window`` objects, applies openings through the
real floor-plan tools, generates the real SDF/GLTF room geometry, and exports
the normal Drake directive and Blender floor plan.

All generated artifacts are first built and validated in a same-filesystem
transaction directory.  Publication refuses to overwrite an existing layout
and rolls back its own moves if any final verification fails.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import stat
import sys
import uuid

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit


PROFILE = "school_reference_20260710"
SEED_SCHEMA_VERSION = 2
WALL_HEIGHT_M = 3.2


@dataclass(frozen=True)
class RoomSeed:
    room_id: str
    room_type: str
    x: float
    y: float
    x_size: float
    y_size: float
    exterior_walls: tuple[str, ...] = ()


@dataclass(frozen=True)
class DoorSeed:
    door_id: str
    room_a: str
    room_b: str
    center_world: tuple[float, float] | None = None
    width: float = 1.0
    height: float = 2.1


ROOMS: tuple[RoomSeed, ...] = (
    RoomSeed("classroom_01", "classroom", 0.0, 0.0, 9.0, 7.5, ("west",)),
    RoomSeed("classroom_03", "classroom", 0.0, 11.5, 9.0, 7.5, ("west",)),
    RoomSeed("classroom_04", "classroom", 0.0, 20.0, 9.0, 7.5, ("west",)),
    RoomSeed("classroom_02", "classroom", 21.0, 0.0, 9.0, 7.5, ("east",)),
    RoomSeed("classroom_06", "classroom", 21.0, 11.2, 9.0, 7.5, ("east",)),
    RoomSeed("classroom_05", "classroom", 21.0, 20.0, 9.0, 7.5, ("east",)),
    RoomSeed("library", "library", 10.0, -9.0, 10.0, 9.0, ("south",)),
    RoomSeed("boys_toilet", "restroom", 1.0, 7.5, 4.0, 4.0),
    RoomSeed("girls_toilet", "restroom", 5.0, 7.5, 4.0, 4.0),
    RoomSeed("storage_room", "storage", 21.0, 7.5, 5.0, 3.7),
    RoomSeed("main_corridor", "corridor", 9.0, 0.0, 12.0, 22.5),
)

DOORS: tuple[DoorSeed, ...] = (
    DoorSeed("door_main_corridor_classroom_01", "main_corridor", "classroom_01"),
    DoorSeed("door_main_corridor_classroom_02", "main_corridor", "classroom_02"),
    DoorSeed("door_main_corridor_classroom_03", "main_corridor", "classroom_03"),
    DoorSeed("door_main_corridor_classroom_04", "main_corridor", "classroom_04"),
    DoorSeed("door_main_corridor_classroom_05", "main_corridor", "classroom_05"),
    DoorSeed("door_main_corridor_classroom_06", "main_corridor", "classroom_06"),
    DoorSeed("door_main_corridor_library", "main_corridor", "library"),
    DoorSeed(
        "door_main_corridor_girls_toilet",
        "main_corridor",
        "girls_toilet",
        center_world=(9.0, 8.1),
    ),
    DoorSeed("door_main_corridor_storage_room", "main_corridor", "storage_room"),
    DoorSeed(
        "door_girls_toilet_boys_toilet",
        "girls_toilet",
        "boys_toilet",
        center_world=(5.0, 8.1),
    ),
)

WINDOW_WALLS: dict[str, str] = {
    "classroom_01": "west",
    "classroom_02": "east",
    "classroom_03": "west",
    "classroom_04": "west",
    "classroom_05": "east",
    "classroom_06": "east",
    "library": "south",
}

ROOM_CONTEXT: dict[str, str] = {
    "classroom_01": "Reference west-wing lower classroom with balanced teaching rows.",
    "classroom_02": "Reference east-wing lower classroom with grouped seating.",
    "classroom_03": "Reference west-wing middle classroom with active learning clusters.",
    "classroom_04": "Reference west-wing upper classroom with tidy daylight-focused layout.",
    "classroom_05": "Reference east-wing upper classroom with reading and art identity.",
    "classroom_06": "Reference east-wing middle classroom with practical dense seating.",
    "library": "Warm central library immediately inside the south double entrance.",
    "boys_toilet": "Compact boys school restroom west of the shared public foyer.",
    "girls_toilet": "Compact girls school restroom beside the boys restroom and foyer.",
    "storage_room": "School supply storage between the lower and middle east classrooms.",
    "main_corridor": "Broad central circulation spine connecting every functional room.",
}


class SeedError(RuntimeError):
    """Raised when deterministic layout generation cannot fail closed."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _is_link_like(path: Path) -> bool:
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(is_junction is not None and is_junction())


def _inside(root: Path, candidate: Path) -> bool:
    try:
        return os.path.commonpath((os.fspath(root), os.fspath(candidate))) == os.fspath(
            root
        )
    except ValueError:
        return False


def _regular_file(path: Path, *, label: str) -> None:
    if _is_link_like(path):
        raise SeedError(f"{label} must not be a symlink or junction: {path}")
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise SeedError(f"Cannot inspect {label}: {path}: {exc}") from exc
    if not stat.S_ISREG(mode):
        raise SeedError(f"{label} must be a regular file: {path}")


def _tree_manifest(root: Path, relative_roots: tuple[str, ...]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for relative_root in relative_roots:
        target = root / relative_root
        if not target.exists():
            raise SeedError(f"Generated artifact is missing: {target}")
        paths = [target] if target.is_file() else sorted(target.rglob("*"))
        for path in paths:
            if path.is_dir() and not _is_link_like(path):
                continue
            _regular_file(path, label="generated artifact")
            records.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
    return sorted(records, key=lambda item: item["path"])


def _manifest_sha256(records: list[dict[str, Any]]) -> str:
    return hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _transaction_path(scene_dir: Path, *, pid: int, nonce: str) -> Path:
    """Return a same-depth sibling transaction for path-stable GLTF URIs.

    SceneSmith writes material references relative to each generated GLTF.  A
    transaction nested below the final run directory adds one path component,
    so the references become invalid after publication.  Keeping the staged
    run beside the final run preserves the exact relative depth.
    """

    run_dir = scene_dir.parent
    return run_dir.parent / (
        f".{run_dir.name}.reference_layout_txn.{pid}.{nonce}"
    )


def _gltf_local_uri_manifest(
    repo: Path,
    scene_root: Path,
    relative_roots: tuple[str, ...] = ("floor_plans", "room_geometry"),
) -> list[dict[str, Any]]:
    """Resolve every local GLTF buffer/image URI and fail closed if it is broken."""

    records: list[dict[str, Any]] = []
    resolved_scene_root = scene_root.resolve(strict=True)
    for relative_root in relative_roots:
        root = scene_root / relative_root
        if not root.is_dir() or _is_link_like(root):
            raise SeedError(f"Generated GLTF root is missing or linked: {root}")
        for gltf_path in sorted(root.rglob("*.gltf")):
            _regular_file(gltf_path, label="generated GLTF")
            try:
                document = json.loads(gltf_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise SeedError(f"Cannot parse generated GLTF {gltf_path}: {exc}") from exc
            for section in ("buffers", "images"):
                entries = document.get(section, [])
                if not isinstance(entries, list):
                    raise SeedError(f"GLTF {gltf_path} has non-list {section}")
                for index, entry in enumerate(entries):
                    if not isinstance(entry, dict) or "uri" not in entry:
                        continue
                    uri = entry["uri"]
                    if not isinstance(uri, str) or not uri:
                        raise SeedError(
                            f"GLTF {gltf_path} has invalid {section}[{index}] URI"
                        )
                    parsed = urlsplit(uri)
                    if parsed.scheme == "data":
                        continue
                    if (
                        parsed.scheme
                        or parsed.netloc
                        or parsed.query
                        or parsed.fragment
                    ):
                        raise SeedError(
                            f"GLTF {gltf_path} has non-local {section}[{index}] URI: {uri}"
                        )
                    decoded = unquote(parsed.path)
                    if not decoded or "\\" in decoded or "\x00" in decoded:
                        raise SeedError(
                            f"GLTF {gltf_path} has unsafe {section}[{index}] URI: {uri}"
                        )
                    relative_target = Path(decoded)
                    if relative_target.is_absolute():
                        raise SeedError(
                            f"GLTF {gltf_path} has absolute {section}[{index}] URI: {uri}"
                        )
                    try:
                        target = (gltf_path.parent / relative_target).resolve(strict=True)
                    except OSError as exc:
                        raise SeedError(
                            f"GLTF {gltf_path} has unresolved {section}[{index}] URI "
                            f"{uri}: {exc}"
                        ) from exc
                    if not _inside(repo, target):
                        raise SeedError(
                            f"GLTF {gltf_path} {section}[{index}] URI escapes the repo: {uri}"
                        )
                    _regular_file(target, label="resolved GLTF dependency")
                    if _inside(resolved_scene_root, target):
                        target_scope = "scene"
                        target_path = target.relative_to(resolved_scene_root).as_posix()
                    else:
                        target_scope = "repo"
                        target_path = target.relative_to(repo).as_posix()
                    records.append(
                        {
                            "gltf": gltf_path.relative_to(scene_root).as_posix(),
                            "section": section,
                            "index": index,
                            "uri": uri,
                            "target_scope": target_scope,
                            "target": target_path,
                            "size_bytes": target.stat().st_size,
                            "sha256": _sha256_file(target),
                        }
                    )
    return sorted(
        records,
        key=lambda item: (item["gltf"], item["section"], item["index"]),
    )


def _structural_layout_projection(document: dict[str, Any]) -> dict[str, Any]:
    """Return the layout fields that prompt binding is not allowed to change.

    ``school_room_contract.py bind-layout`` deliberately rewrites the prompt in
    every RoomSpec and adds the same prompt to each raw PlacedRoom record.  The
    deterministic seed receipt therefore binds all other layout bytes while
    excluding only those two expected prompt locations.
    """

    projected = json.loads(json.dumps(document))
    rooms = projected.get("rooms", [])
    if isinstance(rooms, list):
        for room in rooms:
            if isinstance(room, dict):
                room.pop("prompt", None)
    placed_rooms = projected.get("placed_rooms", [])
    if isinstance(placed_rooms, list):
        for room in placed_rooms:
            if isinstance(room, dict):
                room.pop("prompt", None)
    return projected


def _structural_layout_sha256(document: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            _structural_layout_projection(document),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _seed_spec_evidence() -> dict[str, Any]:
    """Return a pure-Python summary used by local tests and the receipt."""

    return {
        "profile": PROFILE,
        "room_ids": [room.room_id for room in ROOMS],
        "room_bounds": {
            room.room_id: [
                room.x,
                room.y,
                room.x + room.x_size,
                room.y + room.y_size,
            ]
            for room in ROOMS
        },
        "interior_door_ids": [door.door_id for door in DOORS],
        "window_rooms": sorted(WINDOW_WALLS),
        "entrance": {
            "id": "main_entrance",
            "room": "library",
            "wall": "south",
            "position_exact": 4.1,
            "width": 1.8,
            "height": 2.2,
            "leaf_count": 2,
        },
    }


def _boundary_for_pair(layout: Any, first: str, second: str) -> str:
    for label, (room_a, room_b, _direction) in layout.boundary_labels.items():
        if {room_a, room_b} == {first, second}:
            return label
    raise SeedError(f"No SceneSmith boundary joins {first} and {second}")


def _boundary_for_exterior(layout: Any, room_id: str, direction: str) -> str:
    for label, (room_a, room_b, wall_direction) in layout.boundary_labels.items():
        if room_a == room_id and room_b is None and wall_direction == direction:
            return label
    raise SeedError(f"No exterior {direction} boundary exists for {room_id}")


def _door_position(
    layout: Any,
    seed: DoorSeed,
    *,
    get_shared_edge: Any,
) -> float:
    room_a = layout.get_placed_room(seed.room_a)
    room_b = layout.get_placed_room(seed.room_b)
    if room_a is None or room_b is None:
        raise SeedError(f"Door {seed.door_id} references an unplaced room")
    shared = get_shared_edge(room_a, room_b)
    if shared is None or shared.width + 1e-9 < seed.width:
        raise SeedError(f"Door {seed.door_id} has no sufficiently wide shared edge")
    if seed.center_world is None:
        return shared.position_along_wall + (shared.width - seed.width) / 2.0
    wall = next(
        (candidate for candidate in room_a.walls if candidate.direction == shared.wall_direction),
        None,
    )
    if wall is None:
        raise SeedError(f"Door {seed.door_id} cannot resolve its room-a wall")
    tangent = (
        seed.center_world[1]
        if shared.wall_direction.value in {"east", "west"}
        else seed.center_world[0]
    )
    wall_start = (
        wall.start_point[1]
        if shared.wall_direction.value in {"east", "west"}
        else wall.start_point[0]
    )
    return tangent - wall_start - seed.width / 2.0


def _build_native_layout(stage_scene: Path, prompt: str, cfg: Any) -> tuple[Any, Any]:
    # Imports are deliberately lazy: local contract tests can inspect the exact
    # seed without requiring the Linux-only Drake/Blender SceneSmith runtime.
    from scenesmith.agent_utils.house import (
        ConnectionType,
        Door,
        HouseLayout,
        RoomSpec,
        WallDirection,
        Window,
    )
    from scenesmith.floor_plan_agents.tools.ascii_generator import (
        generate_ascii_floor_plan,
    )
    from scenesmith.floor_plan_agents.tools.floor_plan_tools import (
        DoorWindowConfig,
        FloorPlanTools,
    )
    from scenesmith.floor_plan_agents.tools.room_placement import (
        _create_placed_room,
        _update_wall_connectivity,
        get_shared_edge,
    )

    specs = []
    for room in ROOMS:
        specs.append(
            RoomSpec(
                room_id=room.room_id,
                room_type=room.room_type,
                prompt=ROOM_CONTEXT[room.room_id],
                position=(room.x, room.y),
                width=room.y_size,
                length=room.x_size,
                connections={},
                exterior_walls={WallDirection(value) for value in room.exterior_walls},
            )
        )
    layout = HouseLayout(
        wall_height=WALL_HEIGHT_M,
        house_prompt=prompt,
        room_specs=specs,
        house_dir=stage_scene,
    )
    layout.placed_rooms = [
        _create_placed_room(
            spec,
            (room.x, room.y),
            room_width=room.x_size,
            room_depth=room.y_size,
        )
        for room, spec in zip(ROOMS, specs, strict=True)
    ]
    _update_wall_connectivity(layout.placed_rooms)
    ascii_plan = generate_ascii_floor_plan(layout.placed_rooms)
    layout.boundary_labels = ascii_plan.boundary_labels
    layout.placement_valid = True

    width_range = tuple(float(value) for value in cfg.doors.width_range)
    height_range = tuple(float(value) for value in cfg.doors.height_range)
    window_width_range = tuple(float(value) for value in cfg.windows.width_range)
    window_height_range = tuple(float(value) for value in cfg.windows.height_range)
    door_window_config = DoorWindowConfig(
        door_width_min=width_range[0],
        door_width_max=width_range[1],
        door_height_min=height_range[0],
        door_height_max=height_range[1],
        door_default_width=float(cfg.doors.default_width),
        door_default_height=float(cfg.doors.default_height),
        window_width_min=window_width_range[0],
        window_width_max=window_width_range[1],
        window_height_min=window_height_range[0],
        window_height_max=window_height_range[1],
        window_default_width=float(cfg.windows.default_width),
        window_default_height=float(cfg.windows.default_height),
        window_default_sill_height=float(cfg.windows.default_sill_height),
        window_segment_margin=float(cfg.windows.segment_margin),
        exterior_door_clearance_m=float(cfg.doors.exterior_clearance),
    )
    # Only native geometry/manipulation methods are needed; bypassing __init__
    # avoids constructing OpenAI function wrappers or a material retriever.
    tools = object.__new__(FloorPlanTools)
    tools.layout = layout
    tools.mode = "house"
    tools.min_opening_separation = float(cfg.room_placement.min_opening_separation)
    tools.door_window_config = door_window_config

    for seed in DOORS:
        label = _boundary_for_pair(layout, seed.room_a, seed.room_b)
        position = _door_position(layout, seed, get_shared_edge=get_shared_edge)
        door = Door(
            id=seed.door_id,
            boundary_label=label,
            position_segment="center",
            position_exact=position,
            door_type="interior",
            room_a=seed.room_a,
            room_b=seed.room_b,
            width=seed.width,
            height=seed.height,
            leaf_count=1,
        )
        error = tools._apply_door_to_wall(door)
        if error:
            raise SeedError(f"SceneSmith rejected {seed.door_id}: {error}")
        layout.doors.append(door)
        spec_a = layout.get_room_spec(seed.room_a)
        spec_b = layout.get_room_spec(seed.room_b)
        if spec_a is None or spec_b is None:
            raise SeedError(f"Door {seed.door_id} lost a RoomSpec endpoint")
        spec_a.connections[seed.room_b] = ConnectionType.DOOR
        spec_b.connections[seed.room_a] = ConnectionType.DOOR

    entrance = Door(
        id="main_entrance",
        boundary_label=_boundary_for_exterior(layout, "library", "south"),
        position_segment="center",
        position_exact=4.1,
        door_type="exterior",
        room_a="library",
        room_b=None,
        width=1.8,
        height=2.2,
        leaf_count=2,
    )
    entrance_error = tools._apply_door_to_wall(entrance)
    if entrance_error:
        raise SeedError(f"SceneSmith rejected the main entrance: {entrance_error}")
    layout.doors.append(entrance)

    for room_id, direction in WINDOW_WALLS.items():
        wall_direction = WallDirection(direction)
        window = Window(
            id=f"window_{room_id}",
            boundary_label=_boundary_for_exterior(layout, room_id, direction),
            position_along_wall=2.0,
            room_id=room_id,
            wall_direction=wall_direction,
            width=1.5,
            height=1.2,
            sill_height=0.9,
        )
        placed = layout.get_placed_room(room_id)
        wall = next(
            (candidate for candidate in placed.walls if candidate.direction == wall_direction),
            None,
        ) if placed is not None else None
        if wall is None or not wall.is_exterior:
            raise SeedError(f"Window for {room_id} is not on a real exterior wall")
        error = tools._apply_window_to_wall(window)
        if error:
            raise SeedError(f"SceneSmith rejected {window.id}: {error}")
        layout.windows.append(window)

    common_zones = [
        {
            "id": "restroom_foyer",
            "position": [1.0, 7.5],
            "width": 8.0,
            "depth": 1.2,
            "reason": (
                "A separate public foyer gives both gendered restrooms independent "
                "access without crossing either restroom."
            ),
            "carved_from": ["boys_toilet", "girls_toilet"],
            "connections": [
                {
                    "id": "foyer_corridor",
                    "to": "main_corridor",
                    "backing_door_id": "door_main_corridor_girls_toilet",
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
    ]
    common_result = tools._set_navigation_common_zones_impl(
        json.dumps(common_zones, sort_keys=True)
    )
    if not common_result.success:
        raise SeedError(f"SceneSmith rejected restroom foyer: {common_result.message}")

    native_validation = tools._validate_impl()
    if native_validation.layout != "ok" or native_validation.connectivity != "ok":
        raise SeedError(
            "Native SceneSmith validation failed: "
            f"layout={native_validation.layout}; connectivity={native_validation.connectivity}"
        )
    layout.connectivity_valid = True
    return layout, ascii_plan


def _generate_native_geometry(layout: Any, stage_scene: Path, cfg: Any) -> None:
    from scenesmith.floor_plan_agents.stateful_floor_plan_agent import (
        StatefulFloorPlanAgent,
    )
    from scenesmith.floor_plan_agents.tools.geometry_cache import GeometryCache
    from scenesmith.utils.logging import ConsoleLogger

    agent = object.__new__(StatefulFloorPlanAgent)
    agent.layout = layout
    agent.cfg = cfg
    agent.logger = ConsoleLogger(stage_scene)
    agent._geometry_cache = GeometryCache(stage_scene / ".geometry_cache")
    agent._generate_all_room_geometries(output_dir=stage_scene / "floor_plans")


def _validate_document(document: dict[str, Any]) -> dict[str, Any]:
    from validate_school_floor_layout import validate

    result = validate(document)
    if result.get("status") != "pass":
        raise SeedError(
            "Reference-school validator rejected deterministic native layout: "
            + "; ".join(str(item) for item in result.get("critical_issues", []))
        )
    return result


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _publication_names() -> tuple[str, ...]:
    return (
        "floor_plans",
        "room_geometry",
        *(f"room_{room.room_id}" for room in ROOMS),
        "house_layout.json",
    )


def _publish(stage_scene: Path, scene_dir: Path) -> list[str]:
    stage_package = stage_scene / "package.xml"
    final_package = scene_dir / "package.xml"
    names = list(_publication_names())
    if final_package.exists():
        _regular_file(final_package, label="existing scene package.xml")
        if final_package.read_bytes() != stage_package.read_bytes():
            raise SeedError("Existing scene package.xml differs from native staged package")
    else:
        # Publish package.xml before house_layout.json, which is the commit marker.
        names.insert(-1, "package.xml")
    for name in names:
        if (scene_dir / name).exists() or (scene_dir / name).is_symlink():
            raise SeedError(f"Refusing to overwrite existing layout artifact: {scene_dir / name}")

    moved: list[str] = []
    try:
        for name in names:
            os.replace(stage_scene / name, scene_dir / name)
            moved.append(name)
    except BaseException:
        for name in reversed(moved):
            final = scene_dir / name
            staged = stage_scene / name
            if final.exists() and not staged.exists():
                os.replace(final, staged)
        raise
    return moved


def _rollback_publication(stage_scene: Path, scene_dir: Path, moved: list[str]) -> None:
    for name in reversed(moved):
        final = scene_dir / name
        staged = stage_scene / name
        if final.exists() and not staged.exists():
            os.replace(final, staged)


def _verify_existing(
    *,
    repo: Path,
    scene_dir: Path,
    config_path: Path,
    prompt_path: Path,
    prompt_sha256: str,
) -> dict[str, Any]:
    layout_path = scene_dir / "house_layout.json"
    receipt_path = scene_dir / "quality_gates" / "reference_school_layout_seed.json"
    _regular_file(layout_path, label="existing house layout")
    _regular_file(receipt_path, label="reference-school seed receipt")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected_header = {
        "schema_version": SEED_SCHEMA_VERSION,
        "status": "pass",
        "profile": PROFILE,
        "implementation": "native_scenesmith_deterministic_reference_layout",
    }
    for key, expected in expected_header.items():
        if receipt.get(key) != expected:
            raise SeedError(f"Existing seed receipt has invalid {key}: {receipt.get(key)!r}")
    script_record = receipt.get("script", {})
    script_path = Path(__file__).resolve()
    if (
        script_record.get("path") != script_path.relative_to(repo).as_posix()
        or script_record.get("sha256") != _sha256_file(script_path)
    ):
        raise SeedError("Existing seed receipt does not bind the current materializer")
    if receipt.get("prompt") != {"path": str(prompt_path), "sha256": prompt_sha256}:
        raise SeedError("Existing seed receipt does not bind the immutable prompt")
    expected_config = {"path": str(config_path), "sha256": _sha256_file(config_path)}
    if receipt.get("resolved_config") != expected_config:
        raise SeedError("Existing seed receipt does not bind the current resolved config")
    if receipt.get("seed_spec") != _seed_spec_evidence():
        raise SeedError("Existing seed receipt does not bind the current exact seed")

    layout_document = json.loads(layout_path.read_text(encoding="utf-8"))
    validation = _validate_document(layout_document)
    if receipt.get("structural_layout_sha256") != _structural_layout_sha256(
        layout_document
    ):
        raise SeedError("Existing house layout differs structurally from the native seed")
    artifact_manifest = _tree_manifest(
        scene_dir, ("floor_plans", "room_geometry", "package.xml")
    )
    if receipt.get("artifacts") != artifact_manifest:
        raise SeedError("Existing native floor geometry differs from the seed receipt")
    if receipt.get("artifact_manifest_sha256") != _manifest_sha256(artifact_manifest):
        raise SeedError("Existing seed artifact-manifest attestation is invalid")
    if receipt.get("artifact_count") != len(artifact_manifest):
        raise SeedError("Existing seed artifact count is invalid")
    gltf_uri_manifest = _gltf_local_uri_manifest(repo, scene_dir)
    if receipt.get("gltf_local_uris") != gltf_uri_manifest:
        raise SeedError("Existing GLTF dependency evidence differs from the seed receipt")
    if receipt.get("gltf_local_uri_manifest_sha256") != _manifest_sha256(
        gltf_uri_manifest
    ):
        raise SeedError("Existing GLTF dependency attestation is invalid")
    for room in ROOMS:
        room_dir = scene_dir / f"room_{room.room_id}"
        if not room_dir.is_dir() or _is_link_like(room_dir):
            raise SeedError(f"Required native room output directory is missing: {room_dir}")

    from scenesmith.agent_utils.house import HouseLayout

    HouseLayout.from_dict(layout_document, house_dir=scene_dir)
    receipt["native_scene_validation"] = validation
    receipt["reuse_verified"] = True
    return receipt


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo_dir.expanduser().resolve(strict=True)
    scene_dir = args.scene_dir.expanduser().resolve(strict=True)
    config_path = args.config.expanduser().resolve(strict=True)
    prompt_path = args.prompt.expanduser().resolve(strict=True)
    if not _inside(repo, scene_dir) or not _inside(repo, config_path) or not _inside(
        repo, prompt_path
    ):
        raise SeedError("Repository, scene, config, and prompt paths must share one checkout")
    for path, label in ((config_path, "resolved config"), (prompt_path, "prompt")):
        _regular_file(path, label=label)
    if _is_link_like(scene_dir):
        raise SeedError(f"Scene directory must not be a link: {scene_dir}")
    if args.profile != PROFILE:
        raise SeedError(f"Unsupported deterministic seed profile: {args.profile}")
    prompt_sha256 = _sha256_file(prompt_path)
    if prompt_sha256 != args.expected_prompt_sha256:
        raise SeedError(
            f"Prompt hash mismatch: expected {args.expected_prompt_sha256}, got {prompt_sha256}"
        )
    if (scene_dir / "house_layout.json").exists():
        return _verify_existing(
            repo=repo,
            scene_dir=scene_dir,
            config_path=config_path,
            prompt_path=prompt_path,
            prompt_sha256=prompt_sha256,
        )

    from omegaconf import OmegaConf

    cfg_document = OmegaConf.load(config_path)
    if "floor_plan_agent" not in cfg_document:
        raise SeedError("Resolved config has no floor_plan_agent section")
    cfg = cfg_document.floor_plan_agent

    transaction = _transaction_path(
        scene_dir,
        pid=os.getpid(),
        nonce=uuid.uuid4().hex,
    )
    if transaction.exists():
        raise SeedError(f"Transaction path unexpectedly exists: {transaction}")
    transaction.mkdir(parents=False)
    stage_scene = transaction / scene_dir.name
    stage_scene.mkdir()

    previous_cwd = Path.cwd()
    try:
        os.chdir(repo)
        layout, ascii_plan = _build_native_layout(stage_scene, prompt_path.read_text(encoding="utf-8"), cfg)
        _generate_native_geometry(layout, stage_scene, cfg)
        for room in ROOMS:
            (stage_scene / f"room_{room.room_id}").mkdir()

        layout_document = layout.to_dict(scene_dir=stage_scene)
        layout_path = stage_scene / "house_layout.json"
        _atomic_json(layout_path, layout_document)
        validation = _validate_document(layout_document)

        # Round-trip through the actual SceneSmith loader before Blender export.
        from scenesmith.agent_utils.house import HouseLayout
        from scenesmith.floor_plan_agents.stateful_floor_plan_agent import StatefulFloorPlanAgent

        HouseLayout.from_dict(layout_document, house_dir=stage_scene)
        export_agent = object.__new__(StatefulFloorPlanAgent)
        export_agent.layout = layout
        export_agent._export_floor_plan(output_dir=stage_scene / "floor_plans")
        final_floor = stage_scene / "floor_plans" / "final_floor_plan"
        for required in (final_floor / "floor_plan.dmd.yaml", final_floor / "floor_plan.blend"):
            _regular_file(required, label="native floor-plan export")
            if required.stat().st_size == 0:
                raise SeedError(f"Native floor-plan export is empty: {required}")

        cache_dir = stage_scene / ".geometry_cache"
        if cache_dir.exists():
            resolved_cache = cache_dir.resolve(strict=True)
            if not _inside(stage_scene.resolve(strict=True), resolved_cache):
                raise SeedError("Geometry cache escaped the transaction scene")
            shutil.rmtree(resolved_cache)

        seed_validation_path = transaction / "seed_validation.json"
        _atomic_json(seed_validation_path, validation)
        staged_manifest = _tree_manifest(
            stage_scene,
            ("floor_plans", "room_geometry", "package.xml"),
        )
        staged_gltf_uri_manifest = _gltf_local_uri_manifest(repo, stage_scene)
        manifest_sha256 = _manifest_sha256(staged_manifest)
        gltf_uri_manifest_sha256 = _manifest_sha256(staged_gltf_uri_manifest)
        structural_layout_sha256 = _structural_layout_sha256(layout_document)

        moved = _publish(stage_scene, scene_dir)
        receipt_written = False
        try:
            final_document = json.loads(
                (scene_dir / "house_layout.json").read_text(encoding="utf-8")
            )
            final_validation = _validate_document(final_document)
            HouseLayout.from_dict(final_document, house_dir=scene_dir)
            final_manifest = _tree_manifest(
                scene_dir, ("floor_plans", "room_geometry", "package.xml")
            )
            final_gltf_uri_manifest = _gltf_local_uri_manifest(repo, scene_dir)
            if final_manifest != staged_manifest:
                raise SeedError("Published layout bytes differ from validated staging bytes")
            if final_gltf_uri_manifest != staged_gltf_uri_manifest:
                raise SeedError(
                    "Published GLTF dependencies differ from validated staging dependencies"
                )
            if _structural_layout_sha256(final_document) != structural_layout_sha256:
                raise SeedError("Published structural layout differs from validated staging bytes")

            receipt = {
                "schema_version": SEED_SCHEMA_VERSION,
                "status": "pass",
                "profile": PROFILE,
                "implementation": "native_scenesmith_deterministic_reference_layout",
                "script": {
                    "path": Path(__file__).resolve().relative_to(repo).as_posix(),
                    "sha256": _sha256_file(Path(__file__).resolve()),
                },
                "prompt": {"path": str(prompt_path), "sha256": prompt_sha256},
                "resolved_config": {
                    "path": str(config_path),
                    "sha256": _sha256_file(config_path),
                },
                "seed_spec": _seed_spec_evidence(),
                "ascii_floor_plan_sha256": hashlib.sha256(
                    ascii_plan.ascii_art.encode("utf-8")
                ).hexdigest(),
                "structural_layout_sha256": structural_layout_sha256,
                "layout_sha256_at_materialization": _sha256_file(
                    scene_dir / "house_layout.json"
                ),
                "native_scene_validation": final_validation,
                "artifact_manifest_sha256": manifest_sha256,
                "artifact_count": len(final_manifest),
                "artifacts": final_manifest,
                "gltf_local_uri_manifest_sha256": gltf_uri_manifest_sha256,
                "gltf_local_uris": final_gltf_uri_manifest,
            }
            _atomic_json(
                scene_dir / "quality_gates" / "reference_school_layout_seed.json",
                receipt,
            )
            receipt_written = True
        except BaseException:
            if not receipt_written:
                _rollback_publication(stage_scene, scene_dir, moved)
            raise

        # Delete only the unique transaction directory created above.
        resolved_transaction = transaction.resolve(strict=True)
        if not _inside(scene_dir.parent.parent.resolve(strict=True), resolved_transaction):
            raise SeedError("Transaction cleanup path escaped the output-date directory")
        shutil.rmtree(resolved_transaction)
        return receipt
    finally:
        os.chdir(previous_cwd)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--expected-prompt-sha256", required=True)
    parser.add_argument("--profile", default=PROFILE)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        receipt = run(parse_args(argv))
    except (OSError, ValueError, SeedError) as exc:
        print(f"reference-school seed failed: {exc}", file=sys.stderr)
        return 1
    print(
        "reference-school native layout passed: "
        f"rooms={len(ROOMS)} artifacts={receipt['artifact_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
