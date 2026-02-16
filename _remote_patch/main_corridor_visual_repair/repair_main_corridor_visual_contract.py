#!/usr/bin/env python3
"""Repair the real main-corridor scene after its strict visual gate failed.

The transaction warms the authored wall materials, makes the prompt-required
double entrance and welcome zone visually explicit, adds visible warm fixtures,
and turns two existing perimeter benches/planters into a spacious lobby island.
It regenerates no learned asset and publishes only after room-wide projection,
simulation, and collision comparison succeed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

from repair_room1_acceptance_details import (
    _publish_staged_final,
    _worker_config,
    _write_json,
    _write_text,
)


EXPECTED_STATE_SHA256 = "5ea67cb496733a018ffcd2c08f3dbe26ce18e0b8c28bedeee0ae8ddc1690886b"
ENTRANCE_ID = "main_entrance_double_door_welcome_0"
MAT_ID = "welcome_threshold_mat_0"
SCONCE_IDS = tuple(f"corridor_warm_sconce_{index}" for index in range(4))
WARM_CREAM = [1.0, 0.86, 0.72, 1.0]


LETTER_BITMAPS = {
    "W": ("10001", "10001", "10001", "10101", "10101", "11011", "10001"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "C": ("01111", "10000", "10000", "10000", "10000", "10000", "01111"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _box_visual(
    name: str,
    pose: tuple[float, float, float, float, float, float],
    size: tuple[float, float, float],
    diffuse: str,
    *,
    emissive: str | None = None,
) -> str:
    emission = f"<emissive>{emissive}</emissive>" if emissive else ""
    return (
        f"<visual name='{name}'>"
        f"<pose>{' '.join(str(value) for value in pose)}</pose>"
        f"<geometry><box><size>{' '.join(str(value) for value in size)}</size></box></geometry>"
        f"<material><diffuse>{diffuse}</diffuse>{emission}</material>"
        "</visual>"
    )


def _welcome_letters() -> str:
    word = "WELCOME"
    cell_x = 0.055
    cell_z = 0.064
    pitch = 0.34
    left = -pitch * (len(word) - 1) / 2.0
    visuals: list[str] = []
    for letter_index, letter in enumerate(word):
        for row, line in enumerate(LETTER_BITMAPS[letter]):
            for column, enabled in enumerate(line):
                if enabled != "1":
                    continue
                x = left + letter_index * pitch + (column - 2) * cell_x
                z = 2.48 + (3 - row) * cell_z
                visuals.append(
                    _box_visual(
                        f"letter_{letter_index}_{row}_{column}",
                        (x, 0.145, z, 0, 0, 0),
                        (cell_x * 0.92, 0.035, cell_z * 0.92),
                        "0.05 0.16 0.27 1",
                    )
                )
    return "".join(visuals)


def _entrance_sdf() -> str:
    visuals = [
        _box_visual("left_jamb", (-0.96, 0.04, 1.15, 0, 0, 0), (0.12, 0.16, 2.30), "0.44 0.22 0.07 1"),
        _box_visual("right_jamb", (0.96, 0.04, 1.15, 0, 0, 0), (0.12, 0.16, 2.30), "0.44 0.22 0.07 1"),
        _box_visual("door_header", (0, 0.04, 2.30, 0, 0, 0), (2.04, 0.16, 0.14), "0.44 0.22 0.07 1"),
        _box_visual("welcome_panel", (0, 0.08, 2.48, 0, 0, 0), (3.10, 0.12, 0.62), "0.96 0.76 0.42 1"),
        # Two visibly distinct glass-and-oak leaves are opened toward the exterior,
        # so the 1.8 m interior threshold and its robot route remain unobstructed.
        _box_visual("left_open_leaf", (-1.08, -0.30, 1.12, 0, 0, -1.02), (0.80, 0.07, 2.16), "0.49 0.27 0.09 1"),
        _box_visual("right_open_leaf", (1.08, -0.30, 1.12, 0, 0, 1.02), (0.80, 0.07, 2.16), "0.49 0.27 0.09 1"),
        _box_visual("left_glass", (-1.08, -0.34, 1.35, 0, 0, -1.02), (0.56, 0.025, 1.30), "0.45 0.76 0.90 0.48", emissive="0.08 0.18 0.24 1"),
        _box_visual("right_glass", (1.08, -0.34, 1.35, 0, 0, 1.02), (0.56, 0.025, 1.30), "0.45 0.76 0.90 0.48", emissive="0.08 0.18 0.24 1"),
        _box_visual("left_pull", (-0.80, 0.01, 1.05, 0, 0, 0), (0.04, 0.06, 0.28), "0.62 0.64 0.66 1"),
        _box_visual("right_pull", (0.80, 0.01, 1.05, 0, 0, 0), (0.04, 0.06, 0.28), "0.62 0.64 0.66 1"),
    ]
    return (
        "<?xml version='1.0'?><sdf version='1.7'>"
        "<model name='main_entrance_double_door_welcome'><link name='base_link'>"
        + "".join(visuals)
        + _welcome_letters()
        # Boundary-mounted visual architecture intentionally adds no duplicate
        # collision over the existing cut-out wall and exterior entrance portal.
        + "</link></model></sdf>"
    )


def _mat_sdf() -> str:
    return (
        "<?xml version='1.0'?><sdf version='1.7'><model name='welcome_threshold_mat'>"
        "<link name='base_link'>"
        + _box_visual("mat", (0, 0, 0.008, 0, 0, 0), (2.20, 1.15, 0.016), "0.12 0.31 0.36 1")
        + _box_visual("mat_inner", (0, 0, 0.017, 0, 0, 0), (1.75, 0.76, 0.006), "0.86 0.63 0.22 1")
        + "</link></model></sdf>"
    )


def _sconce_sdf() -> str:
    return (
        "<?xml version='1.0'?><sdf version='1.7'><model name='corridor_warm_sconce'>"
        "<link name='base_link'>"
        + _box_visual("oak_backplate", (0, 0.02, 0.26, 0, 0, 0), (0.30, 0.06, 0.52), "0.36 0.17 0.05 1")
        + _box_visual("amber_shade", (0, -0.12, 0.27, 0, 0, 0), (0.24, 0.20, 0.28), "1.0 0.55 0.12 1", emissive="1.0 0.34 0.04 1")
        + _box_visual("light_pool", (0, -0.16, 0.08, 0, 0, 0), (0.42, 0.02, 0.08), "1.0 0.72 0.25 0.72", emissive="0.9 0.38 0.04 1")
        + "</link></model></sdf>"
    )


def _style_walls(paths: list[Path]) -> list[dict[str, str]]:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    changes: list[dict[str, str]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        materials = payload.get("materials")
        if not isinstance(materials, list) or not materials:
            raise RuntimeError(f"Wall GLTF has no materials: {path}")
        for material in materials:
            if not isinstance(material, dict):
                raise RuntimeError(f"Malformed wall material: {path}")
            pbr = material.setdefault("pbrMetallicRoughness", {})
            if not isinstance(pbr, dict):
                raise RuntimeError(f"Malformed wall PBR payload: {path}")
            pbr["baseColorFactor"] = WARM_CREAM
            pbr["metallicFactor"] = 0.0
            pbr["roughnessFactor"] = 0.86
            material["emissiveFactor"] = [0.045, 0.022, 0.006]
        backup = path.with_name(f"{path.name}.backup_corridor_visual_{stamp}")
        if backup.exists():
            raise RuntimeError(f"Refusing to overwrite wall backup: {backup}")
        shutil.copy2(path, backup)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
        changes.append({
            "path": str(path),
            "backup": str(backup),
            "before_sha256": _sha256(backup),
            "after_sha256": _sha256(path),
        })
    return changes


def _restore_walls(changes: list[dict[str, str]]) -> None:
    for record in changes:
        shutil.copy2(record["backup"], record["path"])


def _asset(room_dir: Path, name: str, filename: str, content: str) -> Path:
    directory = room_dir / "generated_assets" / "architectural" / name
    if directory.exists():
        candidate = directory / filename
        if not candidate.is_file() or candidate.read_text(encoding="utf-8") != content:
            raise RuntimeError(f"Unexpected existing architectural asset: {directory}")
        return candidate
    staging = directory.with_name(f".{directory.name}.{uuid.uuid4().hex}.tmp")
    staging.mkdir(parents=True, exist_ok=False)
    _write_text(staging / filename, content)
    os.replace(staging, directory)
    return directory / filename


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--csv", default="inputs/full_school_floor_20260703.csv")
    parser.add_argument("--room-id", default="main_corridor")
    parser.add_argument("--port-offset", type=int, default=929)
    args = parser.parse_args()

    from run_single_room_worker import _configure_pytorch_cuda_allocator

    _configure_pytorch_cuda_allocator()
    import numpy as np
    from omegaconf import OmegaConf
    from pydrake.math import RigidTransform, RollPitchYaw
    from scenesmith.agent_utils.physical_feasibility import apply_physical_feasibility_postprocessing
    from scenesmith.agent_utils.physics_validation import compute_scene_collisions
    from scenesmith.agent_utils.rendering import save_scene_as_blend
    from scenesmith.agent_utils.room import ObjectType, RoomScene, SceneObject, UniqueID

    repo = args.repo_dir.resolve()
    run = args.run_dir.resolve()
    os.chdir(repo)
    room_dir = run / "scene_000" / f"room_{args.room_id}"
    final_dir = room_dir / "scene_states" / "final_scene"
    state_path = final_dir / "scene_state.json"
    if _sha256(state_path) != EXPECTED_STATE_SHA256:
        raise RuntimeError(f"Unexpected main-corridor state: {_sha256(state_path)}")
    wall_paths = sorted((run / "scene_000" / "floor_plans" / args.room_id / "walls").glob("**/wall.gltf"))
    expected_wall_ids = {"east_wall", "north_wall", "north_wall_exterior", "south_wall", "west_wall"}
    actual_wall_ids = {path.parent.name for path in wall_paths}
    if actual_wall_ids != expected_wall_ids:
        raise RuntimeError(
            "Unexpected main-corridor wall set: "
            f"expected={sorted(expected_wall_ids)}, actual={sorted(actual_wall_ids)}"
        )

    cfg = OmegaConf.to_container(_worker_config(args), resolve=True)
    stage_dir: Path | None = None
    wall_changes: list[dict[str, str]] = []
    published = False
    try:
        original = json.loads(state_path.read_text(encoding="utf-8"))
        scene = RoomScene(room_geometry=None, scene_dir=room_dir, room_id=args.room_id)
        scene.restore_from_state_dict(original)
        required = {"bench_1", "bench_3", "planter_0", "planter_5", "sign_stand_0", "console_table_0"}
        existing = {str(object_id) for object_id in scene.objects}
        missing = sorted(required - existing)
        unexpected = sorted({ENTRANCE_ID, MAT_ID, *SCONCE_IDS} & existing)
        if missing or unexpected:
            raise RuntimeError(f"Stale corridor input: missing={missing}, unexpected={unexpected}")

        baseline = compute_scene_collisions(scene)
        baseline_depths = {
            tuple(sorted((item.object_a_id, item.object_b_id))): item.penetration_depth
            for item in baseline
        }

        entrance_sdf = _asset(room_dir, "main_entrance_welcome_v1", "main_entrance.sdf", _entrance_sdf())
        mat_sdf = _asset(room_dir, "welcome_threshold_mat_v1", "welcome_mat.sdf", _mat_sdf())
        sconce_sdf = _asset(room_dir, "corridor_warm_sconce_v1", "warm_sconce.sdf", _sconce_sdf())

        scene.add_object(SceneObject(
            object_id=UniqueID(ENTRANCE_ID),
            object_type=ObjectType.WALL_MOUNTED,
            name="exterior_double_entrance_welcome",
            description="Clearly visible 1.8 metre exterior double entrance with glass oak leaves and raised WELCOME sign",
            transform=RigidTransform(p=[0.0, -11.16, 0.0]),
            sdf_path=entrance_sdf,
            metadata={
                "asset_source": "architectural_contract_repair",
                "semantic_role": "main_exterior_double_entrance",
                "leaf_count": 2,
                "clear_width_m": 1.8,
                "door_state": "open_outward",
                "collision_policy": "existing_room_and_global_entrance_portal_owns_boundary_collision",
            },
            bbox_min=np.array([-1.60, -0.75, 0.0]),
            bbox_max=np.array([1.60, 0.18, 2.80]),
        ))
        scene.add_object(SceneObject(
            object_id=UniqueID(MAT_ID),
            object_type=ObjectType.FURNITURE,
            name="welcome_threshold_mat",
            description="Low-profile teal and gold welcome mat defining the entrance arrival zone",
            transform=RigidTransform(p=[0.0, -10.25, 0.0]),
            sdf_path=mat_sdf,
            metadata={
                "asset_source": "architectural_contract_repair",
                "semantic_role": "entrance_threshold_welcome_zone",
                "collision_policy": "sixteen_millimetre_compressible_floor_finish",
            },
            bbox_min=np.array([-1.10, -0.575, 0.0]),
            bbox_max=np.array([1.10, 0.575, 0.025]),
        ))
        for object_id, x in zip(SCONCE_IDS, (-4.4, -2.4, 2.4, 4.4), strict=True):
            scene.add_object(SceneObject(
                object_id=UniqueID(object_id),
                object_type=ObjectType.WALL_MOUNTED,
                name="warm_corridor_wall_sconce",
                description="Visible amber wall sconce providing warm welcoming lobby light",
                transform=RigidTransform(rpy=RollPitchYaw([0.0, 0.0, math.pi]), p=[x, 11.13, 2.20]),
                sdf_path=sconce_sdf,
                metadata={
                    "asset_source": "architectural_contract_repair",
                    "lighting_type": "warm_amber_wall_sconce",
                    "collision_policy": "high_wall_fixture_outside_robot_contact_volume",
                },
                bbox_min=np.array([-0.22, -0.20, 0.0]),
                bbox_max=np.array([0.22, 0.06, 0.54]),
            ))

        # Form a balanced lobby island from existing generated furniture while
        # keeping a roughly 4.4 m unobstructed center lane and every wall portal clear.
        moved: list[dict[str, Any]] = []
        targets = {
            "bench_1": ([-2.65, 2.8, 0.004], math.pi / 2.0),
            "bench_3": ([2.65, 2.8, 0.004], math.pi / 2.0),
            "planter_0": ([-3.35, 1.75, 0.004], 0.0),
            "planter_5": ([3.35, 1.75, 0.004], 0.0),
            "sign_stand_0": ([0.0, -8.65, 0.004], 0.0),
        }
        for object_id, (translation, yaw) in targets.items():
            obj = scene.get_object(UniqueID(object_id))
            assert obj is not None
            moved.append({
                "object_id": object_id,
                "before_translation": obj.transform.translation().tolist(),
                "after_translation": translation,
                "after_yaw_radians": yaw,
            })
            obj.transform = RigidTransform(rpy=RollPitchYaw([0.0, 0.0, yaw]), p=translation)
            obj.metadata = {
                **obj.metadata,
                "visual_contract_repair": "spacious_central_lobby_island_or_arrival_wayfinding",
            }

        wall_changes = _style_walls(wall_paths)
        physics = cfg["manipuland_agent"]["physics_validation"]
        projection = cfg["experiment"]["projection"]
        final = projection["final"]
        simulation = projection["simulation"]
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
            raise RuntimeError(f"Corridor physics repair rejected: projection={projection_success}, removed={removed}")
        collisions = compute_scene_collisions(scene)
        new_or_deeper = [
            item for item in collisions
            if item.penetration_depth
            > baseline_depths.get(tuple(sorted((item.object_a_id, item.object_b_id))), -1.0) + 1e-4
        ]
        if new_or_deeper:
            raise RuntimeError(
                "Corridor repair introduced collision: "
                + "; ".join(item.to_description() for item in new_or_deeper)
            )

        stage_dir = final_dir.parent / f".main_corridor_visual_{uuid.uuid4().hex}"
        stage_dir.mkdir(parents=False, exist_ok=False)
        repaired_state = scene.to_state_dict()
        repaired_state["timestamp"] = time.time()
        _write_json(stage_dir / "scene_state.json", repaired_state)
        (stage_dir / "scene.dmd.yaml").write_text(scene.to_drake_directive(), encoding="utf-8")
        rendering = cfg["furniture_agent"]["rendering"]
        save_scene_as_blend(
            scene=scene,
            output_path=stage_dir / "scene.blend",
            blender_server_host=rendering.get("blender_server_host", "127.0.0.1"),
            blender_server_port_range=tuple(rendering["blender_server_port_range"]),
            server_startup_delay=rendering["server_startup_delay"],
            port_cleanup_delay=rendering["port_cleanup_delay"],
        )
        backup = _publish_staged_final(staged_dir=stage_dir, final_dir=final_dir)
        stage_dir = None
        published = True
        receipt = {
            "schema_version": 1,
            "status": "pass",
            "room_id": args.room_id,
            "operation": "warm_walls_visible_double_entrance_lighting_and_lobby_island",
            "state_before_sha256": EXPECTED_STATE_SHA256,
            "state_after_sha256": _sha256(final_dir / "scene_state.json"),
            "backup_final_scene": str(backup),
            "entrance": {
                "object_id": ENTRANCE_ID,
                "sdf": str(entrance_sdf),
                "sdf_sha256": _sha256(entrance_sdf),
                "leaf_count": 2,
                "clear_width_m": 1.8,
            },
            "welcome_mat": {"object_id": MAT_ID, "sdf_sha256": _sha256(mat_sdf)},
            "warm_sconces": list(SCONCE_IDS),
            "moved_existing_objects": moved,
            "wall_materials": wall_changes,
            "physical_validation": {
                "projection_success": projection_success,
                "removed_ids": [str(value) for value in removed],
                "baseline_collision_count": len(baseline),
                "final_collision_count": len(collisions),
                "new_or_deeper_collision_count": len(new_or_deeper),
            },
            "quality_policy": "collision, stability, doorway, robot-clearance, inventory, and visual thresholds unchanged",
        }
        _write_json(room_dir / "quality_gates" / "main_corridor_visual_contract_repair.json", receipt)
        print(json.dumps(receipt, indent=2, sort_keys=True))
    finally:
        if wall_changes and not published:
            _restore_walls(wall_changes)
        # Unpublished staging is retained for forensic diagnosis.


if __name__ == "__main__":
    main()
