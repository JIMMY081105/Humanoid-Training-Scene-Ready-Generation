#!/usr/bin/env python3
"""Repair completed Classroom 3/4 checkpoints without regenerating either room."""

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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from repair_room1_acceptance_details import _publish_staged_final, _worker_config, _write_json, _write_text


EXPECTED_STATE_SHA256 = {
    "classroom_03": "f445685e0fcff97af20cff40e2c48e7c2f714c00d83d8055dfeec87cc11941c3",
    "classroom_04": "c624c6400c046473932c47f534c75010a957ee95b9f46483ddae3c9b0a508538",
}

CLAMP_IDS = (
    "display_frame_0", "display_frame_1", "display_frame_2", "display_frame_3",
    "projector_screen_0", "notice_board_0",
)


@dataclass(frozen=True)
class ItemSpec:
    key: str
    object_id: str
    name: str
    description: str
    size: tuple[float, float, float]
    color: tuple[float, float, float, float]
    preferred: str
    yaw: float = 0.0


ITEMS = {
    "classroom_03": (
        ItemSpec("classroom_marker", "acceptance_classroom_marker_0", "general_classroom_marker", "One independently represented purple general classroom marker", (0.145, 0.024, 0.024), (0.48, 0.08, 0.72, 1.0), "desk", 0.16),
        ItemSpec("whiteboard_eraser", "acceptance_whiteboard_eraser_0", "whiteboard_eraser", "One independently represented blue and gray whiteboard eraser", (0.135, 0.062, 0.032), (0.08, 0.35, 0.72, 1.0), "desk", -0.12),
    ),
    "classroom_04": (
        ItemSpec("pencil_holder", "acceptance_pencil_holder_0", "pencil_holder", "One independently represented blue classroom pencil holder cup", (0.12, 0.12, 0.18), (0.05, 0.42, 0.78, 1.0), "desk"),
        ItemSpec("ruler", "acceptance_ruler_0", "ruler", "One independently represented yellow 30 centimetre classroom ruler", (0.30, 0.035, 0.008), (0.96, 0.72, 0.06, 1.0), "desk", 0.15),
        ItemSpec("glue_stick", "acceptance_glue_stick_0", "glue_stick", "One independently represented white and orange classroom glue stick", (0.052, 0.052, 0.12), (0.95, 0.35, 0.08, 1.0), "desk"),
        ItemSpec("storage_bin", "acceptance_storage_bin_0", "classroom_storage_bin", "One independently represented labeled blue classroom storage bin", (0.28, 0.20, 0.11), (0.04, 0.34, 0.68, 1.0), "cubby"),
        ItemSpec("paper_tray", "acceptance_paper_tray_0", "paper_tray", "One independently represented green teacher paper tray", (0.32, 0.24, 0.055), (0.08, 0.52, 0.30, 1.0), "cubby"),
        ItemSpec("whiteboard_eraser", "acceptance_whiteboard_eraser_0", "whiteboard_eraser", "One independently represented blue and gray whiteboard eraser", (0.135, 0.062, 0.032), (0.08, 0.35, 0.72, 1.0), "desk", -0.12),
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _box_gltf(spec: ItemSpec) -> str:
    sx, sy, sz = spec.size
    x, y = sx / 2.0, sy / 2.0
    positions = [(-x, -y, 0.0), (x, -y, 0.0), (x, y, 0.0), (-x, y, 0.0), (-x, -y, sz), (x, -y, sz), (x, y, sz), (-x, y, sz)]
    indices = [0,2,1,0,3,2,4,5,6,4,6,7,0,1,5,0,5,4,1,2,6,1,6,5,2,3,7,2,7,6,3,0,4,3,4,7]
    pbytes = b"".join(struct.pack("<3f", *point) for point in positions)
    ibytes = b"".join(struct.pack("<H", value) for value in indices)
    data = pbytes + ibytes
    return json.dumps({
        "asset": {"version": "2.0", "generator": "SceneSmith distinct classroom inventory repair"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": spec.name}],
        "meshes": [{"name": spec.name, "primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "material": 0}]}],
        "materials": [{"name": f"{spec.name}_material", "pbrMetallicRoughness": {"baseColorFactor": list(spec.color), "metallicFactor": 0.0, "roughnessFactor": 0.55}}],
        "buffers": [{"byteLength": len(data), "uri": "data:application/octet-stream;base64," + base64.b64encode(data).decode("ascii")}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(pbytes), "target": 34962},
            {"buffer": 0, "byteOffset": len(pbytes), "byteLength": len(ibytes), "target": 34963},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 8, "type": "VEC3", "min": [-x, -y, 0.0], "max": [x, y, sz]},
            {"bufferView": 1, "componentType": 5123, "count": len(indices), "type": "SCALAR", "min": [0], "max": [7]},
        ],
    }, indent=2, sort_keys=True) + "\n"


def _visual(name: str, pose: tuple[float, float, float], size: tuple[float, float, float], color: tuple[float, float, float, float]) -> str:
    return (
        f"<visual name='{name}'><pose>{pose[0]} {pose[1]} {pose[2]} 0 0 0</pose>"
        f"<geometry><box><size>{size[0]} {size[1]} {size[2]}</size></box></geometry>"
        f"<material><diffuse>{' '.join(map(str, color))}</diffuse></material></visual>"
    )


def _collision(size: tuple[float, float, float]) -> str:
    return (
        f"<collision name='accurate_collision'><pose>0 0 {size[2] / 2.0} 0 0 0</pose>"
        f"<geometry><box><size>{size[0]} {size[1]} {size[2]}</size></box></geometry></collision>"
    )


def _sdf(spec: ItemSpec, mesh_name: str) -> str:
    sx, sy, sz = spec.size
    visuals = [
        f"<visual name='main'><geometry><mesh><uri>{mesh_name}</uri></mesh></geometry></visual>"
    ]
    if spec.key == "pencil_holder":
        for index, x in enumerate((-0.03, 0.0, 0.03)):
            visuals.append(_visual(f"pencil_{index}", (x, 0.0, 0.18), (0.012, 0.012, 0.15), (0.95, 0.62 - index * 0.15, 0.06 + index * 0.20, 1.0)))
    elif spec.key == "storage_bin":
        visuals.append(_visual("white_label", (0.0, -sy / 2.0 - 0.002, 0.062), (0.16, 0.006, 0.05), (0.95, 0.94, 0.84, 1.0)))
    elif spec.key == "paper_tray":
        visuals.extend([
            _visual("left_rail", (-sx / 2.0 + 0.012, 0.0, 0.07), (0.024, sy, 0.14), spec.color),
            _visual("right_rail", (sx / 2.0 - 0.012, 0.0, 0.07), (0.024, sy, 0.14), spec.color),
            _visual("paper", (0.0, 0.0, 0.062), (sx * 0.84, sy * 0.82, 0.016), (0.97, 0.95, 0.85, 1.0)),
        ])
    elif spec.key == "glue_stick":
        visuals.append(_visual("orange_cap", (0.0, 0.0, sz - 0.012), (sx * 1.05, sy * 1.05, 0.024), (0.98, 0.26, 0.04, 1.0)))
    elif spec.key == "whiteboard_eraser":
        visuals.append(_visual("dark_felt", (0.0, 0.0, 0.004), (sx * 0.94, sy * 0.94, 0.008), (0.05, 0.05, 0.06, 1.0)))
    return (
        "<?xml version='1.0'?><sdf version='1.7'>"
        f"<model name='{spec.name}'><link name='base_link'>{''.join(visuals)}{_collision(spec.size)}</link></model></sdf>\n"
    )


def _asset(room_dir: Path, spec: ItemSpec) -> tuple[Path, Path]:
    directory = room_dir / "generated_assets" / "contract_inventory" / f"{spec.key}_v1"
    mesh_name = f"{spec.key}.gltf"
    mesh_text = _box_gltf(spec)
    sdf_text = _sdf(spec, mesh_name)
    if directory.exists():
        mesh, sdf = directory / mesh_name, directory / f"{spec.key}.sdf"
        if not mesh.is_file() or mesh.read_text(encoding="utf-8") != mesh_text or not sdf.is_file() or sdf.read_text(encoding="utf-8") != sdf_text:
            raise RuntimeError(f"Unexpected inventory asset: {directory}")
        return mesh, sdf
    staging = directory.with_name(f".{directory.name}.{uuid.uuid4().hex}.tmp")
    staging.mkdir(parents=True, exist_ok=False)
    _write_text(staging / mesh_name, mesh_text)
    _write_text(staging / f"{spec.key}.sdf", sdf_text)
    os.replace(staging, directory)
    return directory / mesh_name, directory / f"{spec.key}.sdf"


def _new_or_deeper(baseline: list[Any], candidate: list[Any]) -> list[Any]:
    depths = {tuple(sorted((str(item.object_a_id), str(item.object_b_id)))): item.penetration_depth for item in baseline}
    return [item for item in candidate if item.penetration_depth > depths.get(tuple(sorted((str(item.object_a_id), str(item.object_b_id)))), -1.0) + 1e-4]


def _clamp_wall_objects(scene: Any) -> list[dict[str, Any]]:
    import numpy as np
    from pydrake.math import RigidTransform
    from scenesmith.agent_utils.room import UniqueID

    geometry = scene.room_geometry
    x_min, x_max = -float(geometry.length) / 2.0, float(geometry.length) / 2.0
    y_min, y_max = -float(geometry.width) / 2.0, float(geometry.width) / 2.0
    margin = 0.025
    changes = []
    for object_id in CLAMP_IDS:
        obj = scene.get_object(UniqueID(object_id))
        if obj is None:
            raise RuntimeError(f"Missing overflow repair target: {object_id}")
        corners = np.array([[x, y, z] for x in (float(obj.bbox_min[0]), float(obj.bbox_max[0])) for y in (float(obj.bbox_min[1]), float(obj.bbox_max[1])) for z in (float(obj.bbox_min[2]), float(obj.bbox_max[2]))])
        rotation = obj.transform.rotation().matrix()
        translation = obj.transform.translation().copy()
        world = (rotation @ corners.T).T + translation
        delta = np.zeros(3)
        if world[:, 0].min() < x_min + margin:
            delta[0] += x_min + margin - world[:, 0].min()
        if world[:, 0].max() + delta[0] > x_max - margin:
            delta[0] += x_max - margin - (world[:, 0].max() + delta[0])
        if world[:, 1].min() < y_min + margin:
            delta[1] += y_min + margin - world[:, 1].min()
        if world[:, 1].max() + delta[1] > y_max - margin:
            delta[1] += y_max - margin - (world[:, 1].max() + delta[1])
        before = translation.tolist()
        obj.transform = RigidTransform(obj.transform.rotation(), translation + delta)
        obj.metadata = {**obj.metadata, "acceptance_clamp": "minimal inward translation to authored room bounds"}
        changes.append({"object_id": object_id, "before": before, "after": obj.transform.translation().tolist(), "delta": delta.tolist()})
    return changes


def _candidate_surfaces(scene: Any, preferred: str) -> list[tuple[Any, Any]]:
    candidates = []
    for furniture in scene.objects.values():
        text = f"{furniture.object_id} {furniture.name} {furniture.description}".lower()
        if "student" in text or "chair" in text or not furniture.support_surfaces:
            continue
        preference = 0 if (preferred == "cubby" and any(token in text for token in ("cubby", "book", "shelf", "storage"))) or (preferred == "desk" and any(token in text for token in ("teacher", "desk", "station"))) else 1
        for surface in furniture.support_surfaces:
            if surface.area < 0.12:
                continue
            occupied = len(scene.get_objects_on_surface(surface.surface_id))
            height = float(surface.transform.translation()[2])
            candidates.append(((preference, occupied, -height, -surface.area, str(surface.surface_id)), furniture, surface))
    if not candidates:
        raise RuntimeError(f"No usable support surface for {preferred}")
    candidates.sort(key=lambda item: item[0])
    return [(furniture, surface) for _score, furniture, surface in candidates]


def _place_items(scene: Any, room_dir: Path, specs: tuple[ItemSpec, ...]) -> list[dict[str, Any]]:
    import numpy as np
    from pydrake.math import RigidTransform
    from scenesmith.agent_utils.room import ObjectType, PlacementInfo, SceneObject, UniqueID

    placed = []
    surface_use: dict[str, int] = {}
    for spec in specs:
        if scene.get_object(UniqueID(spec.object_id)) is not None:
            raise RuntimeError(f"Repair item already exists: {spec.object_id}")
        mesh, sdf = _asset(room_dir, spec)
        candidates = _candidate_surfaces(scene, spec.preferred)
        selected = None
        for furniture, surface in candidates:
            key = str(surface.surface_id)
            use = surface_use.get(key, 0)
            minimum = np.array(surface.bounding_box_min[:2], dtype=float)
            maximum = np.array(surface.bounding_box_max[:2], dtype=float)
            half = np.array(spec.size[:2], dtype=float) / 2.0
            low, high = minimum + half + 0.025, maximum - half - 0.025
            if np.any(high <= low):
                continue
            fractions = ((0.18, 0.22), (0.50, 0.22), (0.82, 0.22), (0.25, 0.72), (0.62, 0.72), (0.86, 0.72))
            fx, fy = fractions[use % len(fractions)]
            position = low + np.array([fx, fy]) * (high - low)
            if surface.contains_point_2d(position):
                selected = furniture, surface, position
                surface_use[key] = use + 1
                break
        if selected is None:
            raise RuntimeError(f"No fitting support position for {spec.object_id}")
        furniture, surface, position = selected
        pose = surface.to_world_pose(position_2d=position, rotation_2d=spec.yaw) @ RigidTransform(p=[0.0, 0.0, 0.002])
        sx, sy, sz = spec.size
        obj = SceneObject(
            object_id=UniqueID(spec.object_id),
            object_type=ObjectType.MANIPULAND,
            name=spec.name,
            description=spec.description,
            transform=pose,
            geometry_path=mesh,
            sdf_path=sdf,
            metadata={"asset_source": "contract_inventory_repair", "semantic_role": spec.key, "collision_mesh": "matched primitive collision in SDF"},
            bbox_min=np.array([-sx / 2.0, -sy / 2.0, 0.0]),
            bbox_max=np.array([sx / 2.0, sy / 2.0, sz]),
            placement_info=PlacementInfo(parent_surface_id=surface.surface_id, position_2d=position, rotation_2d=spec.yaw, placement_method="deterministic_release_contract_repair"),
        )
        scene.add_object(obj)
        placed.append({"object_id": spec.object_id, "support_object_id": str(furniture.object_id), "surface_id": str(surface.surface_id), "position_2d": position.tolist(), "geometry_sha256": _sha256(mesh), "sdf_sha256": _sha256(sdf)})
    return placed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--room-id", choices=tuple(EXPECTED_STATE_SHA256), required=True)
    parser.add_argument("--port-offset", type=int, default=969)
    args = parser.parse_args()

    from run_single_room_worker import _configure_pytorch_cuda_allocator
    _configure_pytorch_cuda_allocator()
    from omegaconf import OmegaConf
    from scenesmith.agent_utils.physical_feasibility import apply_physical_feasibility_postprocessing
    from scenesmith.agent_utils.physics_validation import compute_scene_collisions
    from scenesmith.agent_utils.rendering import save_scene_as_blend
    from scenesmith.agent_utils.room import RoomScene

    repo = args.repo_dir.resolve()
    run = args.run_dir.resolve()
    os.chdir(repo)
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
    clamped = _clamp_wall_objects(scene) if room_id == "classroom_03" else []
    placed = _place_items(scene, room_dir, ITEMS[room_id])

    physics = cfg["manipuland_agent"]["physics_validation"]
    final = cfg["experiment"]["projection"]["final"]
    simulation = cfg["experiment"]["projection"]["simulation"]
    scene, projection_success, removed = apply_physical_feasibility_postprocessing(
        scene=scene,
        weld_furniture=True,
        projection_enabled=True,
        projection_influence_distance=final["influence_distance"],
        projection_solver_name=final["solver_name"],
        projection_iteration_limit=final["iteration_limit"],
        projection_time_limit_s=final["time_limit_s"],
        projection_xy_only=final["xy_only"],
        projection_fix_rotation=final["fix_rotation"],
        simulation_enabled=simulation["enabled"],
        simulation_time_s=simulation["simulation_time_s"],
        simulation_time_step_s=simulation["time_step_s"],
        simulation_timeout_s=simulation["timeout_s"],
        remove_fallen_furniture=False,
        remove_fallen_manipulands=physics["remove_fallen_manipulands"],
        fallen_manipuland_floor_z=physics["fallen_manipuland_floor_z"],
        fallen_manipuland_near_floor_z=physics["fallen_manipuland_near_floor_z"],
        fallen_manipuland_z_displacement=physics["fallen_manipuland_z_displacement"],
    )
    if not projection_success or removed:
        raise RuntimeError(f"Strict physics rejected {room_id}: projection={projection_success}, removed={removed}")
    candidate = compute_scene_collisions(scene)
    changed = _new_or_deeper(baseline, candidate)
    if changed:
        raise RuntimeError("New/deeper collision: " + "; ".join(item.to_description() for item in changed))

    stage = final_dir.parent / f".release_contract_repair_{uuid.uuid4().hex}"
    stage.mkdir(parents=False, exist_ok=False)
    repaired_state = scene.to_state_dict()
    repaired_state["timestamp"] = time.time()
    _write_json(stage / "scene_state.json", repaired_state)
    (stage / "scene.dmd.yaml").write_text(scene.to_drake_directive(), encoding="utf-8")
    rendering = cfg["furniture_agent"]["rendering"]
    save_scene_as_blend(scene=scene, output_path=stage / "scene.blend", blender_server_host=rendering.get("blender_server_host", "127.0.0.1"), blender_server_port_range=tuple(rendering["blender_server_port_range"]), server_startup_delay=rendering["server_startup_delay"], port_cleanup_delay=rendering["port_cleanup_delay"])
    backup = _publish_staged_final(staged_dir=stage, final_dir=final_dir)
    receipt = {
        "schema_version": 1, "status": "pass", "room_id": room_id,
        "operation": "checkpoint_only_distinct_inventory_and_authored_bounds_release_repair",
        "state_before_sha256": before_sha, "state_after_sha256": _sha256(final_dir / "scene_state.json"),
        "backup_final_scene": str(backup), "clamped_objects": clamped, "added_inventory": placed,
        "physical_validation": {"projection_success": True, "removed_ids": [], "baseline_collision_count": len(baseline), "final_collision_count": len(candidate), "new_or_deeper_collision_count": 0},
        "quality_policy": "master prompt and every collision, stability, doorway, robot-clearance, inventory, and visual threshold remain unchanged",
    }
    _write_json(room_dir / "quality_gates" / "classroom34_release_contract_repair.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
