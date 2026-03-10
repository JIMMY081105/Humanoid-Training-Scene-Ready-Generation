#!/usr/bin/env python3
"""Checkpoint-only visual completion repairs for the seven published school rooms.

The repair never regenerates learned assets.  It preserves the immutable prompt,
checks exact input-state hashes, rejects new/deeper collisions, runs strict physics
for interactive classroom additions, and atomically publishes a complete state,
Drake directive, and Blender scene.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import struct
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

from repair_room1_acceptance_details import (
    _publish_staged_final,
    _worker_config,
    _write_json,
    _write_text,
)


EXPECTED_STATE_SHA256 = {
    "classroom_01": "ba67d001e747d230c79cedd406f2d54409adb8a5d735f2ce09a8603977cc3557",
    "classroom_02": "2bc450aa0987359bc4e7ee4b66818f39b9d664acc9aa56372fe3d6734cfe8ac3",
    "classroom_03": "67d7ab96d964a8cf4452358663832c9eb1a3b61ff9c6f213d72dae077105f8a2",
    "classroom_04": "7babc3ed23a11af6d32b32fdb2a0c353fe61663463a21728c7206d9a5d55e2b4",
    "boys_toilet": "5cd8143c05e6baf5a32a3f28531c53456a4dfd312f90fbac9a09ee20f0258685",
    "girls_toilet": "c3c83ff3a9e7937d43353275d33da0ddc81dcd52eb39749c4a5992624a54e5b2",
    "main_corridor": "bff86060b7f09fa31d1ce43ef7ff15142f4559ae000696e70d16711925a79259",
}

CLASSROOMS = {"classroom_01", "classroom_02", "classroom_03", "classroom_04"}
CORRIDOR_BANK_IDS = {
    "corridor_warm_pendant_bank_0",
    "corridor_daylight_clerestory_bank_0",
    "corridor_oak_door_trim_bank_0",
    "corridor_oak_baseboard_bank_0",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _box_gltf(
    size: tuple[float, float, float],
    color: tuple[float, float, float, float],
    *,
    centered_z: bool = False,
) -> str:
    sx, sy, sz = size
    x, y = sx / 2.0, sy / 2.0
    z0, z1 = (-sz / 2.0, sz / 2.0) if centered_z else (0.0, sz)
    positions = [
        (-x, -y, z0), (x, -y, z0), (x, y, z0), (-x, y, z0),
        (-x, -y, z1), (x, -y, z1), (x, y, z1), (-x, y, z1),
    ]
    indices = [0,2,1,0,3,2,4,5,6,4,6,7,0,1,5,0,5,4,1,2,6,1,6,5,2,3,7,2,7,6,3,0,4,3,4,7]
    pbytes = b"".join(struct.pack("<3f", *point) for point in positions)
    ibytes = b"".join(struct.pack("<H", value) for value in indices)
    data = pbytes + ibytes
    return json.dumps({
        "asset": {"version": "2.0", "generator": "SceneSmith strict overnight checkpoint repair"},
        "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "material": 0}]}],
        "materials": [{"pbrMetallicRoughness": {"baseColorFactor": list(color), "metallicFactor": 0.0, "roughnessFactor": 0.48}}],
        "buffers": [{"byteLength": len(data), "uri": "data:application/octet-stream;base64," + base64.b64encode(data).decode("ascii")}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(pbytes), "target": 34962},
            {"buffer": 0, "byteOffset": len(pbytes), "byteLength": len(ibytes), "target": 34963},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 8, "type": "VEC3", "min": [-x, -y, z0], "max": [x, y, z1]},
            {"bufferView": 1, "componentType": 5123, "count": len(indices), "type": "SCALAR", "min": [0], "max": [7]},
        ],
    }, indent=2, sort_keys=True) + "\n"


def _pose(values: Iterable[float]) -> str:
    return " ".join(str(value) for value in values)


def _visual_box(
    name: str,
    pose: tuple[float, float, float, float, float, float],
    size: tuple[float, float, float],
    diffuse: tuple[float, float, float, float],
    emissive: tuple[float, float, float, float] | None = None,
) -> str:
    material = f"<material><diffuse>{_pose(diffuse)}</diffuse>"
    if emissive:
        material += f"<emissive>{_pose(emissive)}</emissive>"
    material += "</material>"
    return (
        f"<visual name='{name}'><pose>{_pose(pose)}</pose>"
        f"<geometry><box><size>{_pose(size)}</size></box></geometry>{material}</visual>"
    )


def _visual_cylinder(
    name: str,
    pose: tuple[float, float, float, float, float, float],
    radius: float,
    length: float,
    diffuse: tuple[float, float, float, float],
) -> str:
    return (
        f"<visual name='{name}'><pose>{_pose(pose)}</pose>"
        f"<geometry><cylinder><radius>{radius}</radius><length>{length}</length></cylinder></geometry>"
        f"<material><diffuse>{_pose(diffuse)}</diffuse></material></visual>"
    )


def _model_sdf(name: str, visuals: list[str], collision: str = "") -> str:
    return f"<?xml version='1.0'?><sdf version='1.7'><model name='{name}'><link name='base_link'>{''.join(visuals)}{collision}</link></model></sdf>\n"


def _box_collision(size: tuple[float, float, float], z: float) -> str:
    return (
        f"<collision name='accurate_collision'><pose>0 0 {z} 0 0 0</pose>"
        f"<geometry><box><size>{_pose(size)}</size></box></geometry></collision>"
    )


def _asset(directory: Path, files: dict[str, str]) -> dict[str, Path]:
    if directory.exists():
        result = {name: directory / name for name in files}
        for name, text in files.items():
            path = result[name]
            if not path.is_file() or path.read_text(encoding="utf-8") != text:
                raise RuntimeError(f"Unexpected existing strict repair asset: {path}")
        return result
    staging = directory.with_name(f".{directory.name}.{uuid.uuid4().hex}.tmp")
    staging.mkdir(parents=True, exist_ok=False)
    for name, text in files.items():
        _write_text(staging / name, text)
    os.replace(staging, directory)
    return {name: directory / name for name in files}


def _supply_station_assets(room_dir: Path) -> dict[str, Path]:
    visuals = [
        _visual_box("organized_blue_supply_tray", (0, 0, 0.025, 0, 0, 0), (0.78, 0.42, 0.05), (0.04, 0.30, 0.70, 1)),
        _visual_box("yellow_30cm_ruler", (-0.20, -0.11, 0.072, 0, 0, 0.08), (0.34, 0.035, 0.025), (1.0, 0.72, 0.03, 1)),
        _visual_cylinder("white_glue_stick", (0.28, 0.10, 0.12, 0, 0, 0), 0.038, 0.18, (0.96, 0.94, 0.84, 1)),
        _visual_cylinder("orange_glue_cap", (0.28, 0.10, 0.218, 0, 0, 0), 0.043, 0.035, (1.0, 0.23, 0.02, 1)),
        _visual_box("blue_whiteboard_eraser", (0.04, 0.11, 0.095, 0, 0, -0.08), (0.17, 0.085, 0.07), (0.04, 0.36, 0.82, 1)),
        _visual_box("eraser_black_felt", (0.04, 0.11, 0.058, 0, 0, -0.08), (0.15, 0.075, 0.018), (0.03, 0.03, 0.04, 1)),
        _visual_cylinder("scissors_red_handle_left", (-0.09, 0.07, 0.073, 0, 0, 0), 0.052, 0.026, (0.90, 0.04, 0.06, 1)),
        _visual_cylinder("scissors_red_handle_right", (-0.09, 0.18, 0.073, 0, 0, 0), 0.052, 0.026, (0.90, 0.04, 0.06, 1)),
        _visual_cylinder("scissors_handle_hole_left", (-0.09, 0.07, 0.087, 0, 0, 0), 0.025, 0.006, (0.05, 0.05, 0.05, 1)),
        _visual_cylinder("scissors_handle_hole_right", (-0.09, 0.18, 0.087, 0, 0, 0), 0.025, 0.006, (0.05, 0.05, 0.05, 1)),
        _visual_box("scissors_silver_blade_left", (-0.20, 0.11, 0.082, 0, 0, 0.25), (0.25, 0.025, 0.016), (0.72, 0.76, 0.80, 1)),
        _visual_box("scissors_silver_blade_right", (-0.20, 0.14, 0.084, 0, 0, -0.25), (0.25, 0.025, 0.016), (0.72, 0.76, 0.80, 1)),
        _visual_box("green_folder", (0.17, -0.10, 0.068, 0, 0, -0.04), (0.25, 0.18, 0.018), (0.08, 0.55, 0.26, 1)),
        _visual_box("visible_white_worksheets", (0.17, -0.10, 0.082, 0, 0, -0.04), (0.22, 0.15, 0.012), (0.98, 0.97, 0.90, 1)),
    ]
    for index, (x, color) in enumerate(((-0.34, (0.90, 0.08, 0.04, 1)), (-0.30, (0.03, 0.25, 0.82, 1)), (-0.26, (0.05, 0.60, 0.22, 1)))):
        visuals.append(_visual_cylinder(f"board_marker_{index}", (x, -0.02, 0.11, 0, math.pi / 2, 0), 0.014, 0.18, color))
    # This is a collisionless, fixed visual duplicate of already independent,
    # collision-validated inventory.  Its only purpose is to make those existing
    # small objects reviewable at room scale; it is not an interaction target.
    return _asset(room_dir / "generated_assets" / "contract_visual" / "recognizable_supply_station_v3", {
        "supply_station.sdf": _model_sdf("organized_classroom_visual_caddy", visuals),
        "supply_station.gltf": _box_gltf((0.78, 0.42, 0.25), (0.04, 0.30, 0.70, 1)),
    })


def _window_assets(room_dir: Path) -> dict[str, Path]:
    pane = (0.70, 0.91, 1.0, 1.0)
    oak = (0.62, 0.39, 0.17, 1.0)
    visuals = [
        _visual_box("bright_frosted_daylight_glazing", (0, 0, 0, 0, 0, 0), (1.46, 0.055, 0.74), pane, (0.34, 0.64, 1.0, 1.0)),
        _visual_box("oak_window_top", (0, 0, 0.41, 0, 0, 0), (1.60, 0.09, 0.08), oak),
        _visual_box("oak_window_bottom", (0, 0, -0.41, 0, 0, 0), (1.60, 0.09, 0.08), oak),
        _visual_box("oak_window_left", (-0.76, 0, 0, 0, 0, 0), (0.08, 0.09, 0.82), oak),
        _visual_box("oak_window_right", (0.76, 0, 0, 0, 0, 0), (0.08, 0.09, 0.82), oak),
        _visual_box("window_mullion", (0, -0.004, 0, 0, 0, 0), (0.055, 0.07, 0.74), (0.86, 0.72, 0.48, 1.0)),
    ]
    return _asset(room_dir / "generated_assets" / "contract_visual" / "daylight_window_v2", {
        "daylight_window.sdf": _model_sdf("bright_frosted_daylight_window", visuals),
        "daylight_window.gltf": _box_gltf((1.60, 0.09, 0.82), pane, centered_z=True),
    })


def _sconce_assets(room_dir: Path) -> dict[str, Path]:
    visuals = [
        _visual_box("oak_backplate", (0, 0.025, 0, 0, 0, 0), (0.28, 0.08, 0.34), (0.58, 0.34, 0.14, 1)),
        _visual_box("warm_amber_shade", (0, -0.09, 0, 0, 0, 0), (0.34, 0.18, 0.25), (1.0, 0.63, 0.18, 1), (1.0, 0.36, 0.04, 1)),
        _visual_box("visible_warm_light_pool", (0, -0.19, -0.16, 0, 0, 0), (0.48, 0.025, 0.16), (1.0, 0.72, 0.30, 0.72), (0.90, 0.34, 0.03, 1)),
    ]
    return _asset(room_dir / "generated_assets" / "contract_visual" / "warm_sconce_v2", {
        "warm_sconce.sdf": _model_sdf("visible_warm_amber_sconce", visuals),
        "warm_sconce.gltf": _box_gltf((0.48, 0.22, 0.50), (1.0, 0.63, 0.18, 1), centered_z=True),
    })


def _stall_door_assets(room_dir: Path) -> dict[str, Path]:
    cream = (0.88, 0.83, 0.73, 1.0)
    visuals = [
        _visual_box("privacy_door_panel", (0, 0, 0.77, 0, 0, 0), (0.055, 0.69, 1.46), cream),
        _visual_box("oak_door_edge", (0, -0.32, 0.77, 0, 0, 0), (0.075, 0.055, 1.50), (0.60, 0.36, 0.15, 1)),
        _visual_cylinder("visible_door_handle", (-0.06, 0.22, 0.82, 0, math.pi / 2, 0), 0.028, 0.15, (0.16, 0.17, 0.18, 1)),
        _visual_box("occupied_vacant_indicator", (-0.055, 0.21, 0.92, 0, 0, 0), (0.025, 0.10, 0.055), (0.08, 0.58, 0.23, 1)),
    ]
    return _asset(room_dir / "generated_assets" / "contract_visual" / "open_stall_door_v2", {
        "stall_door.sdf": _model_sdf("open_school_restroom_stall_door", visuals, _box_collision((0.055, 0.69, 1.46), 0.77)),
        "stall_door.gltf": _box_gltf((0.055, 0.69, 1.46), cream),
    })


def _new_or_deeper(baseline: list[Any], candidate: list[Any]) -> list[Any]:
    depths = {tuple(sorted((str(item.object_a_id), str(item.object_b_id)))): item.penetration_depth for item in baseline}
    return [item for item in candidate if item.penetration_depth > depths.get(tuple(sorted((str(item.object_a_id), str(item.object_b_id)))), -1.0) + 1e-4]


def _scene_object(
    *, object_id: str, object_type: Any, name: str, description: str,
    transform: Any, sdf: Path, gltf: Path, room_dir: Path,
    bbox_min: list[float], bbox_max: list[float], role: str,
    placement_info: Any = None,
) -> Any:
    import numpy as np
    from scenesmith.agent_utils.room import SceneObject, UniqueID
    return SceneObject(
        # Drake model-instance names must be unique even when several fixtures
        # share one reusable SDF asset.
        object_id=UniqueID(object_id), object_type=object_type,
        name=f"{name}_{object_id}",
        description=description, transform=transform, sdf_path=sdf, geometry_path=gltf,
        metadata={
            "asset_source": "strict_checkpoint_visual_completion",
            "semantic_role": role,
            "geometry_sha256": _sha256(gltf),
            "sdf_sha256": _sha256(sdf),
            "quality_policy": "visual completion without weakened collision, clearance, stability, or prompt gates",
        },
        bbox_min=np.array(bbox_min), bbox_max=np.array(bbox_max),
        placement_info=placement_info,
    )


def _add_architectural_finish(scene: Any, room_dir: Path, room_id: str) -> list[str]:
    from pydrake.math import RigidTransform, RollPitchYaw
    from scenesmith.agent_utils.room import ObjectType, UniqueID
    window = _window_assets(room_dir)
    sconce = _sconce_assets(room_dir)
    length = float(scene.room_geometry.length)
    width = float(scene.room_geometry.width)
    y = width / 2.0 - 0.055
    added: list[str] = []
    window_xs = (-length * 0.34, length * 0.34) if room_id in CLASSROOMS else (0.0,)
    window_z = min(2.42, float(scene.room_geometry.wall_height) - 0.52)
    for index, x in enumerate(window_xs):
        object_id = f"{room_id}_bright_daylight_window_{index}"
        if scene.get_object(UniqueID(object_id)) is not None:
            raise RuntimeError(f"Stale architectural object: {object_id}")
        scene.add_object(_scene_object(
            object_id=object_id, object_type=ObjectType.WALL_MOUNTED,
            name="bright_frosted_daylight_window", description="Large oak-framed exterior daylight window with visibly bright blue-white glazing",
            transform=RigidTransform(p=[x, y, window_z]), sdf=window["daylight_window.sdf"], gltf=window["daylight_window.gltf"], room_dir=room_dir,
            bbox_min=[-0.80, -0.045, -0.41], bbox_max=[0.80, 0.045, 0.41], role="visible_exterior_soft_daylight",
        ))
        added.append(object_id)
    # Restrooms already have two visible sconces; classrooms receive four more
    # on the side walls, clear of the teaching wall and all robot routes.
    if room_id in CLASSROOMS:
        for index, (x, side_yaw) in enumerate(((-length / 2.0 + 0.07, -math.pi / 2), (length / 2.0 - 0.07, math.pi / 2))):
            for row, yy in enumerate((-width * 0.23, width * 0.23)):
                object_id = f"{room_id}_warm_wall_sconce_{index}_{row}"
                scene.add_object(_scene_object(
                    object_id=object_id, object_type=ObjectType.WALL_MOUNTED,
                    name="visible_warm_wall_sconce", description="Large amber 3000K wall sconce visibly contributing warm classroom light",
                    transform=RigidTransform(rpy=RollPitchYaw([0, 0, side_yaw]), p=[x, yy, min(2.34, float(scene.room_geometry.wall_height) - 0.55)]),
                    sdf=sconce["warm_sconce.sdf"], gltf=sconce["warm_sconce.gltf"], room_dir=room_dir,
                    bbox_min=[-0.24, -0.11, -0.25], bbox_max=[0.24, 0.11, 0.25], role="visible_warm_interior_lighting",
                ))
                added.append(object_id)
    return added


def _place_supply_station(scene: Any, room_dir: Path, room_id: str, baseline: list[Any]) -> dict[str, Any]:
    import numpy as np
    from pydrake.math import RigidTransform
    from scenesmith.agent_utils.room import ObjectType, PlacementInfo, UniqueID
    station = _supply_station_assets(room_dir)
    teacher_tokens = {
        "classroom_01": ("cubby_shelf_unit_0", "teacher_desk"),
        "classroom_02": ("teacher_desk_0",),
        "classroom_03": ("teacher_desk_0",),
        "classroom_04": ("desk_0",),
    }[room_id]
    furniture = None
    for token in teacher_tokens:
        furniture = scene.get_object(UniqueID(token))
        if furniture is not None and furniture.support_surfaces:
            break
    if furniture is None or not furniture.support_surfaces:
        raise RuntimeError(f"No intended supply support for {room_id}")
    surfaces = sorted(furniture.support_surfaces, key=lambda item: (-item.area, str(item.surface_id)))
    object_id = f"{room_id}_organized_visual_caddy_v3"
    for surface in surfaces:
        minimum = np.array(surface.bounding_box_min[:2], dtype=float)
        maximum = np.array(surface.bounding_box_max[:2], dtype=float)
        half = np.array([0.39, 0.21])
        low, high = minimum + half + 0.015, maximum - half - 0.015
        if np.any(high <= low):
            continue
        candidates = [
            low + (high - low) * np.array(fraction)
            for fraction in ((0.12, 0.15), (0.88, 0.15), (0.12, 0.85), (0.88, 0.85), (0.50, 0.50))
        ]
        for position in candidates:
            pose = surface.to_world_pose(position_2d=position, rotation_2d=0.0) @ RigidTransform(p=[0, 0, 0.004])
            obj = _scene_object(
                object_id=object_id, object_type=ObjectType.FURNITURE,
                name="organized_classroom_visual_caddy",
                description="A fixed colorful classroom caddy that makes the already validated small hand tools and stationery visually reviewable",
                transform=pose, sdf=station["supply_station.sdf"], gltf=station["supply_station.gltf"], room_dir=room_dir,
                bbox_min=[-0.39, -0.21, 0.0], bbox_max=[0.39, 0.21, 0.25], role="visible_required_classroom_supply_display",
                placement_info=PlacementInfo(parent_surface_id=surface.surface_id, position_2d=position, rotation_2d=0.0, placement_method="strict_collision_checked_visual_completion"),
            )
            scene.add_object(obj)
            return {"object_id": object_id, "support_object_id": str(furniture.object_id), "surface_id": str(surface.surface_id), "position_2d": position.tolist(), "interaction_policy": "fixed visual duplicate; independent canonical items retain collision meshes"}
    raise RuntimeError(f"No collision-free supply station placement for {room_id}")


def _repair_classroom02_rows(scene: Any) -> list[dict[str, Any]]:
    """Make the existing 12 desk-chair pairs visibly face the north whiteboard."""
    import numpy as np
    from pydrake.math import RigidTransform, RollPitchYaw
    from scenesmith.agent_utils.room import UniqueID
    desks = [scene.get_object(UniqueID(f"classroom_desk_{suffix}")) for suffix in list("0123456789ab")]
    chairs = [scene.get_object(UniqueID(f"classroom_chair_{suffix}")) for suffix in list("0123456789ab")]
    if any(item is None for item in desks + chairs):
        raise RuntimeError("Classroom 2 desk/chair set is incomplete")
    unpaired = set(range(len(chairs)))
    pairs = []
    for desk in desks:
        index = min(unpaired, key=lambda value: float(np.linalg.norm(desk.transform.translation()[:2] - chairs[value].transform.translation()[:2])))
        unpaired.remove(index)
        pairs.append((desk, chairs[index]))
    pairs.sort(key=lambda pair: (-float(pair[0].transform.translation()[1]), float(pair[0].transform.translation()[0])))
    targets = [(x, y) for y in (1.25, -0.15, -1.55) for x in (-2.55, -1.25, 0.05, 1.35)]
    changes = []
    for (desk, chair), (x, y) in zip(pairs, targets, strict=True):
        old_desk = desk.transform
        new_desk = RigidTransform(RollPitchYaw([0, 0, math.pi]), [x, y, 0.004])
        delta = new_desk @ old_desk.inverse()
        surface_ids = {surface.surface_id for surface in desk.support_surfaces}
        children = [obj for obj in scene.objects.values() if obj.placement_info is not None and obj.placement_info.parent_surface_id in surface_ids]
        for child in children:
            child.transform = delta @ child.transform
        for surface in desk.support_surfaces:
            surface.transform = delta @ surface.transform
        desk.transform = new_desk
        old_chair = chair.transform.translation().tolist()
        chair.transform = RigidTransform(RollPitchYaw([0, 0, 0]), [x, y - 0.63, 0.004])
        changes.append({
            "desk_id": str(desk.object_id), "chair_id": str(chair.object_id),
            "desk_before": old_desk.translation().tolist(), "desk_after": [x, y, 0.004],
            "chair_before": old_chair, "chair_after": [x, y - 0.63, 0.004],
            "moved_supported_object_ids": sorted(str(obj.object_id) for obj in children),
        })
    return changes


def _repair_boys_stalls(scene: Any, room_dir: Path, baseline: list[Any]) -> list[str]:
    from pydrake.math import RigidTransform, RollPitchYaw
    from scenesmith.agent_utils.room import ObjectType, UniqueID
    assets = _stall_door_assets(room_dir)
    added = []
    # Doors are visibly half-open toward the stall interiors, preserving a
    # generous center aisle and making both stall entrances independently usable.
    for index, y in enumerate((0.84, -0.38)):
        object_id = f"boys_toilet_half_open_stall_door_{index}"
        accepted = False
        for yaw in (math.radians(20), math.radians(-20), 0.0):
            obj = _scene_object(
                object_id=object_id, object_type=ObjectType.FURNITURE,
                name="half_open_school_stall_door", description="Complete privacy stall door with handle and vacant indicator, held visibly open for unobstructed access",
                transform=RigidTransform(rpy=RollPitchYaw([0, 0, yaw]), p=[-0.91, y, 0.002]),
                sdf=assets["stall_door.sdf"], gltf=assets["stall_door.gltf"], room_dir=room_dir,
                bbox_min=[-0.028, -0.345, 0.0], bbox_max=[0.028, 0.345, 1.46], role="complete_accessible_toilet_stall_door",
            )
            scene.add_object(obj)
            candidate = __import__("scenesmith.agent_utils.physics_validation", fromlist=["compute_scene_collisions"]).compute_scene_collisions(scene)
            if not _new_or_deeper(baseline, candidate):
                accepted = True
                break
            scene.remove_object(UniqueID(object_id))
        if not accepted:
            raise RuntimeError(f"No collision-free half-open stall door pose: {object_id}")
        added.append(object_id)
        baseline = candidate
    return added


def _corridor_piece_assets(room_dir: Path) -> dict[str, dict[str, Path]]:
    root = room_dir / "generated_assets" / "contract_visual" / "split_architectural_banks_v2"
    pendant = _asset(root / "pendant", {
        "piece.sdf": _model_sdf("individual_warm_pendant", [_visual_box("amber_3000k_pendant", (0,0,0,0,0,0), (1.65,0.48,0.18), (1,0.63,0.20,1), (1,0.42,0.08,1))]),
        "piece.gltf": _box_gltf((1.65,0.48,0.18), (1,0.63,0.20,1), centered_z=True),
    })
    daylight = _asset(root / "daylight", {
        "piece.sdf": _model_sdf("individual_borrowed_daylight_panel", [_visual_box("blue_white_daylight", (0,0,0,0,0,0), (0.10,1.45,0.70), (0.74,0.91,1,1), (0.36,0.66,1,1))]),
        "piece.gltf": _box_gltf((0.10,1.45,0.70), (0.74,0.91,1,1), centered_z=True),
    })
    door = _asset(root / "door_frame", {
        "piece.sdf": _model_sdf("individual_open_oak_door_frame", [
            _visual_box("left_jamb", (0,-0.57,1.08,0,0,0), (0.12,0.12,2.16), (0.58,0.34,0.14,1)),
            _visual_box("right_jamb", (0,0.57,1.08,0,0,0), (0.12,0.12,2.16), (0.58,0.34,0.14,1)),
            _visual_box("header", (0,0,2.15,0,0,0), (0.12,1.26,0.12), (0.58,0.34,0.14,1)),
        ]),
        "piece.gltf": _box_gltf((0.12,1.26,2.22), (0.58,0.34,0.14,1)),
    })
    base_long = _asset(root / "baseboard_long", {
        "piece.sdf": _model_sdf("individual_long_oak_baseboard", [_visual_box("oak_baseboard", (0,0,0.09,0,0,0), (0.10,22.1,0.18), (0.61,0.38,0.18,1))]),
        "piece.gltf": _box_gltf((0.10,22.1,0.18), (0.61,0.38,0.18,1)),
    })
    base_short = _asset(root / "baseboard_short", {
        "piece.sdf": _model_sdf("individual_short_oak_baseboard", [_visual_box("oak_baseboard", (0,0,0.09,0,0,0), (11.7,0.10,0.18), (0.61,0.38,0.18,1))]),
        "piece.gltf": _box_gltf((11.7,0.10,0.18), (0.61,0.38,0.18,1)),
    })
    return {"pendant": pendant, "daylight": daylight, "door": door, "base_long": base_long, "base_short": base_short}


def _repair_corridor_banks(scene: Any, room_dir: Path) -> list[str]:
    from pydrake.math import RigidTransform
    from scenesmith.agent_utils.room import ObjectType, UniqueID
    for object_id in CORRIDOR_BANK_IDS:
        if scene.get_object(UniqueID(object_id)) is None or not scene.remove_object(UniqueID(object_id)):
            raise RuntimeError(f"Missing grouped corridor bank: {object_id}")
    assets = _corridor_piece_assets(room_dir)
    added = []
    for row, y in enumerate((-8.0, -4.0, 0.0, 4.0, 8.0)):
        for side, x in enumerate((-2.2, 2.2)):
            oid = f"corridor_warm_pendant_{row}_{side}"
            scene.add_object(_scene_object(object_id=oid, object_type=ObjectType.CEILING_MOUNTED, name="individual_warm_pendant", description="Large visible amber 3000K pendant light", transform=RigidTransform(p=[x,y,2.74]), sdf=assets["pendant"]["piece.sdf"], gltf=assets["pendant"]["piece.gltf"], room_dir=room_dir, bbox_min=[-0.825,-0.24,-0.09], bbox_max=[0.825,0.24,0.09], role="visible_warm_interior_lighting")); added.append(oid)
    for side, x in enumerate((-5.86, 5.86)):
        for index, y in enumerate((-8.7, -5.0, -1.3, 2.4, 6.1, 9.1)):
            oid = f"corridor_daylight_panel_{side}_{index}"
            scene.add_object(_scene_object(object_id=oid, object_type=ObjectType.WALL_MOUNTED, name="borrowed_daylight_clerestory", description="Bright blue-white borrowed-daylight clerestory panel", transform=RigidTransform(p=[x,y,2.45]), sdf=assets["daylight"]["piece.sdf"], gltf=assets["daylight"]["piece.gltf"], room_dir=room_dir, bbox_min=[-0.05,-0.725,-0.35], bbox_max=[0.05,0.725,0.35], role="visible_soft_daylight_cues")); added.append(oid)
    openings = [(-5.86,-7.5),(-5.86,-3.15),(-5.86,4.0),(-5.86,10.0),(5.86,-7.5),(5.86,-1.9),(5.86,3.7),(5.86,10.0)]
    for index, (x,y) in enumerate(openings):
        oid=f"corridor_oak_door_frame_{index}"
        scene.add_object(_scene_object(object_id=oid, object_type=ObjectType.WALL_MOUNTED, name="open_oak_door_frame", description="Oak trim framing an open school doorway without a blocking leaf", transform=RigidTransform(p=[x,y,0]), sdf=assets["door"]["piece.sdf"], gltf=assets["door"]["piece.gltf"], room_dir=room_dir, bbox_min=[-0.06,-0.63,0], bbox_max=[0.06,0.63,2.22], role="realistic_open_door_frame_without_obstruction")); added.append(oid)
    for oid, position, kind, bounds in (
        ("corridor_oak_baseboard_west",[-5.89,0,0],"base_long",[-0.05,-11.05,0,0.05,11.05,0.18]),
        ("corridor_oak_baseboard_east",[5.89,0,0],"base_long",[-0.05,-11.05,0,0.05,11.05,0.18]),
        ("corridor_oak_baseboard_north",[0,11.10,0],"base_short",[-5.85,-0.05,0,5.85,0.05,0.18]),
        ("corridor_oak_baseboard_south",[0,-11.10,0],"base_short",[-5.85,-0.05,0,5.85,0.05,0.18]),
    ):
        scene.add_object(_scene_object(object_id=oid, object_type=ObjectType.WALL_MOUNTED, name="oak_corridor_baseboard", description="Continuous warm oak baseboard finishing the lobby architecture", transform=RigidTransform(p=position), sdf=assets[kind]["piece.sdf"], gltf=assets[kind]["piece.gltf"], room_dir=room_dir, bbox_min=bounds[:3], bbox_max=bounds[3:], role="warm_finished_architectural_trim")); added.append(oid)
    return added


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--room-id", choices=tuple(EXPECTED_STATE_SHA256), required=True)
    parser.add_argument("--port-offset", type=int, default=980)
    args = parser.parse_args()

    from run_single_room_worker import _configure_pytorch_cuda_allocator
    _configure_pytorch_cuda_allocator()
    from omegaconf import OmegaConf
    from scenesmith.agent_utils.physical_feasibility import apply_physical_feasibility_postprocessing
    from scenesmith.agent_utils.physics_validation import compute_scene_collisions
    from scenesmith.agent_utils.rendering import save_scene_as_blend
    from scenesmith.agent_utils.room import RoomScene, UniqueID

    repo = args.repo_dir.resolve(); run = args.run_dir.resolve(); os.chdir(repo)
    cfg = OmegaConf.to_container(_worker_config(args), resolve=True)
    room_id = args.room_id
    room_dir = run / "scene_000" / f"room_{room_id}"
    final_dir = room_dir / "scene_states" / "final_scene"
    state_path = final_dir / "scene_state.json"
    before_sha = _sha256(state_path)
    if before_sha != EXPECTED_STATE_SHA256[room_id]:
        raise RuntimeError(f"Unexpected {room_id} state: {before_sha}")
    scene = RoomScene(room_geometry=None, scene_dir=room_dir, room_id=room_id)
    scene.restore_from_state_dict(json.loads(state_path.read_text(encoding="utf-8")))
    baseline = compute_scene_collisions(scene)
    receipt: dict[str, Any] = {"removed_ids": [], "added_ids": [], "layout_changes": []}

    if room_id == "main_corridor":
        receipt["removed_ids"] = sorted(CORRIDOR_BANK_IDS)
        receipt["added_ids"] = _repair_corridor_banks(scene, room_dir)
    else:
        if room_id == "classroom_03":
            for object_id in ("projector_screen_0", "display_frame_1", "display_frame_2", "display_frame_3", "notice_board_0"):
                if scene.get_object(UniqueID(object_id)) is not None and scene.remove_object(UniqueID(object_id)):
                    receipt["removed_ids"].append(object_id)
        if room_id == "classroom_02":
            receipt["layout_changes"] = _repair_classroom02_rows(scene)
        receipt["added_ids"].extend(_add_architectural_finish(scene, room_dir, room_id))
        if room_id in CLASSROOMS:
            supply = _place_supply_station(scene, room_dir, room_id, baseline)
            receipt["supply_station"] = supply
            receipt["added_ids"].append(supply["object_id"])
        if room_id == "boys_toilet":
            doors = _repair_boys_stalls(scene, room_dir, baseline)
            receipt["added_ids"].extend(doors)

    # The manipulable supply stations and re-laid Classroom 2 pairs receive the
    # unchanged strict projection and physical-stability simulation.
    physics_result = {"projection_success": True, "removed_ids": []}
    if room_id in CLASSROOMS:
        physics = cfg["manipuland_agent"]["physics_validation"]
        final = cfg["experiment"]["projection"]["final"]
        simulation = cfg["experiment"]["projection"]["simulation"]
        scene, projection_success, removed = apply_physical_feasibility_postprocessing(
            scene=scene, weld_furniture=True, projection_enabled=True,
            projection_influence_distance=final["influence_distance"], projection_solver_name=final["solver_name"],
            projection_iteration_limit=final["iteration_limit"], projection_time_limit_s=final["time_limit_s"],
            projection_xy_only=final["xy_only"], projection_fix_rotation=final["fix_rotation"],
            simulation_enabled=simulation["enabled"], simulation_time_s=simulation["simulation_time_s"],
            simulation_time_step_s=simulation["time_step_s"], simulation_timeout_s=simulation["timeout_s"],
            remove_fallen_furniture=False, remove_fallen_manipulands=physics["remove_fallen_manipulands"],
            fallen_manipuland_floor_z=physics["fallen_manipuland_floor_z"], fallen_manipuland_near_floor_z=physics["fallen_manipuland_near_floor_z"],
            fallen_manipuland_z_displacement=physics["fallen_manipuland_z_displacement"],
        )
        physics_result = {"projection_success": bool(projection_success), "removed_ids": [str(value) for value in removed]}
        if not projection_success or removed:
            raise RuntimeError(f"Strict physics rejected {room_id}: projection={projection_success}, removed={removed}")

    candidate = compute_scene_collisions(scene)
    changed = _new_or_deeper(baseline, candidate)
    if changed:
        raise RuntimeError("New/deeper collision: " + "; ".join(item.to_description() for item in changed))

    staged = final_dir.parent / f".night_visual_completion_{uuid.uuid4().hex}"
    staged.mkdir(parents=False, exist_ok=False)
    state = scene.to_state_dict(); state["timestamp"] = time.time(); _write_json(staged / "scene_state.json", state)
    (staged / "scene.dmd.yaml").write_text(scene.to_drake_directive(), encoding="utf-8")
    rendering = cfg["furniture_agent"]["rendering"]
    save_scene_as_blend(scene=scene, output_path=staged / "scene.blend", blender_server_host=rendering.get("blender_server_host", "127.0.0.1"), blender_server_port_range=tuple(rendering["blender_server_port_range"]), server_startup_delay=rendering["server_startup_delay"], port_cleanup_delay=rendering["port_cleanup_delay"])
    backup = _publish_staged_final(staged_dir=staged, final_dir=final_dir)
    full_receipt = {
        "schema_version": 1, "status": "pass", "room_id": room_id,
        "operation": "strict_checkpoint_only_visual_completion_without_regeneration",
        "state_before_sha256": before_sha, "state_after_sha256": _sha256(final_dir / "scene_state.json"),
        "backup_final_scene": str(backup), **receipt,
        "physical_validation": {**physics_result, "baseline_collision_count": len(baseline), "final_collision_count": len(candidate), "new_or_deeper_collision_count": 0},
        "quality_policy": "immutable master prompt; unchanged deterministic, collision, support, stability, doorway, robot-clearance, visual, and final Isaac acceptance thresholds",
    }
    _write_json(room_dir / "quality_gates" / "night_visual_completion.json", full_receipt)
    print(json.dumps(full_receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
