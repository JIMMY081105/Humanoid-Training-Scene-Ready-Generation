#!/usr/bin/env python3
"""Add the remaining real Room 1 acceptance details transactionally.

This does not regenerate furniture or manipulands.  It keeps the existing
generated inventory, builds only the missing architectural open-door assembly,
repositions the existing clock and bulletin board onto a visible teaching wall,
warms the existing wall materials, reruns the normal physics post-processing,
and atomically publishes a new final checkpoint on success.
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
from types import SimpleNamespace
from typing import Any


DOOR_OBJECT_ID = "open_classroom_door_0"
REQUIRED_IDS = ("teacher_desk_0", "clock_0", "bulletin_board_0", "whiteboard_1")
WARM_CREAM = [1.0, 0.83, 0.64, 1.0]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _publish_staged_final(*, staged_dir: Path, final_dir: Path) -> Path:
    for name in ("scene_state.json", "scene.dmd.yaml", "scene.blend"):
        candidate = staged_dir / name
        if not candidate.is_file() or candidate.stat().st_size <= 0:
            raise RuntimeError(f"Staged final artifact is missing or empty: {candidate}")
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    backup_dir = final_dir.with_name(f"final_scene.backup_acceptance_details_{stamp}")
    if backup_dir.exists():
        raise RuntimeError(f"Refusing to overwrite final-scene backup: {backup_dir}")
    os.replace(final_dir, backup_dir)
    try:
        os.replace(staged_dir, final_dir)
    except Exception:
        os.replace(backup_dir, final_dir)
        raise
    return backup_dir


def _worker_config(args: argparse.Namespace):
    from run_single_room_worker import _configure_ports, _load_cfg

    worker_args = SimpleNamespace(
        repo_dir=str(args.repo_dir),
        run_dir=str(args.run_dir),
        csv=str(args.csv),
        run_name="room1_acceptance_details_repair",
        start_stage="manipuland",
        stop_stage="manipuland",
        asset_pipeline="generated_sam3d",
        port_offset=args.port_offset,
        artiverse_data="data/artiverse",
        artiverse_embeddings="data/artiverse/embeddings",
        artvip_data="data/artvip_sdf",
        artvip_embeddings="data/artvip_sdf/embeddings",
        materials_data="data/materials",
        materials_embeddings="data/materials_full_quality_contract/embeddings",
    )
    cfg = _load_cfg(worker_args)
    _configure_ports(cfg, args.port_offset)
    return cfg


def _open_door_sdf() -> str:
    """A full-height, outward-open classroom door with a frame and lever.

    The leaf pivots entirely into the exterior/corridor side, leaving the one
    metre room opening clear.  The visual frame makes the doorway readable;
    collision is intentionally limited to the real leaf rather than duplicating
    the existing wall collision with decorative jambs.
    """
    return """<?xml version='1.0'?>
<sdf version='1.7'>
  <model name='open_classroom_door'>
    <link name='base_link'>
      <visual name='left_jamb'>
        <pose>0 -0.48 1.05 0 0 0</pose>
        <geometry><box><size>0.08 0.08 2.10</size></box></geometry>
        <material><diffuse>0.34 0.17 0.06 1</diffuse><specular>0.08 0.04 0.01 1</specular></material>
      </visual>
      <visual name='right_jamb'>
        <pose>0 0.48 1.05 0 0 0</pose>
        <geometry><box><size>0.08 0.08 2.10</size></box></geometry>
        <material><diffuse>0.34 0.17 0.06 1</diffuse><specular>0.08 0.04 0.01 1</specular></material>
      </visual>
      <visual name='header'>
        <pose>0 0 2.05 0 0 0</pose>
        <geometry><box><size>0.08 1.04 0.10</size></box></geometry>
        <material><diffuse>0.34 0.17 0.06 1</diffuse><specular>0.08 0.04 0.01 1</specular></material>
      </visual>
      <visual name='open_oak_leaf'>
        <pose>0.45 -0.43 1.0 0 0 -1.57079632679</pose>
        <geometry><box><size>0.05 0.86 2.0</size></box></geometry>
        <material><diffuse>0.48 0.25 0.09 1</diffuse><specular>0.10 0.05 0.02 1</specular></material>
      </visual>
      <visual name='upper_panel'>
        <pose>0.45 -0.43 1.48 0 0 -1.57079632679</pose>
        <geometry><box><size>0.057 0.62 0.62</size></box></geometry>
        <material><diffuse>0.38 0.18 0.055 1</diffuse></material>
      </visual>
      <visual name='lower_panel'>
        <pose>0.45 -0.43 0.55 0 0 -1.57079632679</pose>
        <geometry><box><size>0.057 0.62 0.50</size></box></geometry>
        <material><diffuse>0.38 0.18 0.055 1</diffuse></material>
      </visual>
      <visual name='brushed_metal_lever'>
        <pose>0.72 -0.40 1.08 0 0 0</pose>
        <geometry><box><size>0.15 0.045 0.045</size></box></geometry>
        <material><diffuse>0.55 0.57 0.58 1</diffuse><specular>0.45 0.45 0.45 1</specular></material>
      </visual>
      <collision name='open_leaf_collision'>
        <pose>0.45 -0.43 1.0 0 0 -1.57079632679</pose>
        <geometry><box><size>0.05 0.86 2.0</size></box></geometry>
      </collision>
    </link>
  </model>
</sdf>
"""


def _style_walls(wall_paths: list[Path]) -> list[dict[str, str]]:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    changed: list[dict[str, str]] = []
    for path in wall_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        materials = payload.get("materials")
        if not isinstance(materials, list) or not materials:
            raise RuntimeError(f"Wall GLTF has no material payload: {path}")
        for material in materials:
            if not isinstance(material, dict):
                raise RuntimeError(f"Wall GLTF has malformed material payload: {path}")
            pbr = material.setdefault("pbrMetallicRoughness", {})
            if not isinstance(pbr, dict):
                raise RuntimeError(f"Wall GLTF has malformed PBR payload: {path}")
            pbr["baseColorFactor"] = WARM_CREAM
            pbr["metallicFactor"] = 0.0
            pbr["roughnessFactor"] = 0.82
            material["emissiveFactor"] = [0.035, 0.020, 0.006]
        backup = path.with_name(f"{path.name}.backup_acceptance_details_{stamp}")
        if backup.exists():
            raise RuntimeError(f"Refusing to overwrite wall backup: {backup}")
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        _write_json(temporary, payload)
        shutil.copy2(path, backup)
        os.replace(temporary, path)
        changed.append(
            {
                "path": str(path),
                "backup": str(backup),
                "prior_sha256": _sha256_file(backup),
                "final_sha256": _sha256_file(path),
            }
        )
    return changed


def _restore_walls(changed: list[dict[str, str]]) -> None:
    for record in changed:
        shutil.copy2(record["backup"], record["path"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--csv", default="inputs/full_school_floor_20260703.csv")
    parser.add_argument("--room-id", default="classroom_01")
    parser.add_argument("--port-offset", type=int, default=72)
    args = parser.parse_args()

    from run_single_room_worker import _configure_pytorch_cuda_allocator

    _configure_pytorch_cuda_allocator()
    import numpy as np
    from omegaconf import OmegaConf
    from pydrake.math import RigidTransform, RollPitchYaw

    from scenesmith.agent_utils.physical_feasibility import (
        apply_physical_feasibility_postprocessing,
    )
    from scenesmith.agent_utils.physics_validation import compute_scene_collisions
    from scenesmith.agent_utils.rendering import save_scene_as_blend
    from scenesmith.agent_utils.room import ObjectType, RoomScene, SceneObject, UniqueID

    repo_dir = args.repo_dir.resolve()
    run_dir = args.run_dir.resolve()
    os.chdir(repo_dir)
    room_dir = run_dir / "scene_000" / f"room_{args.room_id}"
    final_dir = room_dir / "scene_states" / "final_scene"
    state_path = final_dir / "scene_state.json"
    if not state_path.is_file():
        raise FileNotFoundError(f"Missing Room 1 final state: {state_path}")
    wall_paths = sorted(
        (run_dir / "scene_000" / "floor_plans" / args.room_id / "walls").glob("**/wall.gltf")
    )
    if len(wall_paths) < 4:
        raise RuntimeError(f"Expected Room 1 wall GLTFs, found {len(wall_paths)}")

    cfg_dict = OmegaConf.to_container(_worker_config(args), resolve=True)
    original_state_sha256 = _sha256_file(state_path)
    stage_dir: Path | None = None
    asset_dir: Path | None = None
    wall_changes: list[dict[str, str]] = []
    published = False
    try:
        original_state = json.loads(state_path.read_text(encoding="utf-8"))
        scene = RoomScene(room_geometry=None, scene_dir=room_dir, room_id=args.room_id)
        scene.restore_from_state_dict(original_state)
        initial_ids = {str(object_id) for object_id in scene.objects}
        missing = sorted(set(REQUIRED_IDS) - initial_ids)
        if missing:
            raise RuntimeError(f"Room 1 acceptance repair input is stale; missing={missing}")
        if DOOR_OBJECT_ID in initial_ids:
            raise RuntimeError("Room 1 already has the acceptance-detail door; refusing replay")

        baseline_collisions = compute_scene_collisions(scene)
        baseline_depths = {
            tuple(sorted((collision.object_a_id, collision.object_b_id))): collision.penetration_depth
            for collision in baseline_collisions
        }

        # Keep the failed v1 asset as forensic evidence.  Version two removes
        # the static-model flag so Drake honours the scene object's world pose
        # when welding the outward-open leaf to the doorway.
        asset_dir = room_dir / "generated_assets" / "architectural" / "open_classroom_door_v2"
        if asset_dir.exists():
            raise RuntimeError(f"Refusing to overwrite existing architectural asset: {asset_dir}")
        staged_asset_dir = asset_dir.with_name(f".{asset_dir.name}.{uuid.uuid4().hex}.tmp")
        staged_asset_dir.mkdir(parents=True, exist_ok=False)
        _write_text(staged_asset_dir / "open_classroom_door.sdf", _open_door_sdf())
        os.replace(staged_asset_dir, asset_dir)
        door_sdf = asset_dir / "open_classroom_door.sdf"

        scene.add_object(
            SceneObject(
                object_id=UniqueID(DOOR_OBJECT_ID),
                object_type=ObjectType.WALL_MOUNTED,
                name="open_classroom_door",
                description=(
                    "Full-height oak classroom door, open outward into the corridor, "
                    "with a visible frame and brushed-metal lever; the interior doorway "
                    "clearance remains unobstructed."
                ),
                transform=RigidTransform(p=[4.51, 0.0, 0.0]),
                geometry_path=None,
                sdf_path=door_sdf,
                metadata={
                    "asset_source": "architectural_repair",
                    "door_state": "open_outward",
                    "clear_width_m": 0.95,
                    "repair_contract": "room1_visual_acceptance",
                },
                bbox_min=np.array([-0.05, -0.52, 0.0]),
                bbox_max=np.array([0.92, 0.52, 2.10]),
            )
        )

        # Both assets already exist in the accepted state.  Put them on the
        # clear side of the east teaching wall, above and beside the real door.
        # This is a physical scene move, not a review-image annotation.
        board = scene.get_object(UniqueID("bulletin_board_0"))
        clock = scene.get_object(UniqueID("clock_0"))
        assert board is not None and clock is not None
        board.transform = RigidTransform(
            rpy=RollPitchYaw([0.0, 0.0, math.pi / 2.0]),
            p=[4.39, 2.00, 1.58],
        )
        clock.transform = RigidTransform(
            rpy=RollPitchYaw([0.0, 0.0, math.pi / 2.0]),
            p=[4.39, 3.42, 2.67],
        )
        for object_id in ("bulletin_board_0", "clock_0"):
            obj = scene.get_object(UniqueID(object_id))
            assert obj is not None
            obj.metadata = {
                **obj.metadata,
                "acceptance_relocation": "east_teaching_wall_visible_and_clear",
            }

        wall_changes = _style_walls(wall_paths)

        physics_cfg = cfg_dict["manipuland_agent"]["physics_validation"]
        projection_cfg = cfg_dict["experiment"]["projection"]
        final_cfg = projection_cfg["final"]
        simulation_cfg = projection_cfg["simulation"]
        scene, projection_success, physics_removed = apply_physical_feasibility_postprocessing(
            scene=scene,
            weld_furniture=True,
            projection_enabled=True,
            projection_influence_distance=final_cfg["influence_distance"],
            projection_solver_name=final_cfg["solver_name"],
            projection_iteration_limit=final_cfg["iteration_limit"],
            projection_time_limit_s=final_cfg["time_limit_s"],
            projection_xy_only=final_cfg["xy_only"],
            projection_fix_rotation=final_cfg["fix_rotation"],
            simulation_enabled=simulation_cfg["enabled"],
            simulation_time_s=simulation_cfg["simulation_time_s"],
            simulation_time_step_s=simulation_cfg["time_step_s"],
            simulation_timeout_s=simulation_cfg["timeout_s"],
            remove_fallen_furniture=False,
            remove_fallen_manipulands=physics_cfg["remove_fallen_manipulands"],
            fallen_manipuland_floor_z=physics_cfg["fallen_manipuland_floor_z"],
            fallen_manipuland_near_floor_z=physics_cfg["fallen_manipuland_near_floor_z"],
            fallen_manipuland_z_displacement=physics_cfg["fallen_manipuland_z_displacement"],
        )
        if not projection_success or physics_removed:
            raise RuntimeError(
                "Final physics validation did not retain the repaired room: "
                f"projection_success={projection_success}, removed={physics_removed}"
            )
        collisions = compute_scene_collisions(scene)
        new_or_deeper = [
            collision
            for collision in collisions
            if collision.penetration_depth
            > baseline_depths.get(tuple(sorted((collision.object_a_id, collision.object_b_id))), -1.0)
            + 1e-4
        ]
        if new_or_deeper:
            raise RuntimeError(
                "Room-wide collision validation found a new/deeper collision: "
                + "; ".join(collision.to_description() for collision in new_or_deeper)
            )

        stage_dir = final_dir.parent / f".room1_acceptance_details_{uuid.uuid4().hex}"
        stage_dir.mkdir(parents=False, exist_ok=False)
        repaired_state = scene.to_state_dict()
        repaired_state["timestamp"] = time.time()
        _write_json(stage_dir / "scene_state.json", repaired_state)
        (stage_dir / "scene.dmd.yaml").write_text(scene.to_drake_directive(), encoding="utf-8")
        rendering_cfg = cfg_dict["furniture_agent"]["rendering"]
        save_scene_as_blend(
            scene=scene,
            output_path=stage_dir / "scene.blend",
            blender_server_host=rendering_cfg.get("blender_server_host", "127.0.0.1"),
            blender_server_port_range=tuple(rendering_cfg["blender_server_port_range"]),
            server_startup_delay=rendering_cfg["server_startup_delay"],
            port_cleanup_delay=rendering_cfg["port_cleanup_delay"],
        )
        backup_dir = _publish_staged_final(staged_dir=stage_dir, final_dir=final_dir)
        stage_dir = None
        published = True
        receipt = {
            "status": "pass",
            "room_id": args.room_id,
            "repair": "open_architectural_door_visible_teaching_display_clock_warm_walls",
            "prior_final_state_sha256": original_state_sha256,
            "final_state_sha256": _sha256_file(final_dir / "scene_state.json"),
            "backup_final_scene": str(backup_dir),
            "door": {"object_id": DOOR_OBJECT_ID, "sdf": str(door_sdf), "sha256": _sha256_file(door_sdf)},
            "relocated_existing_assets": ["bulletin_board_0", "clock_0"],
            "wall_materials": wall_changes,
            "physical_validation": {
                "final_projection_success": projection_success,
                "removed_ids": [str(object_id) for object_id in physics_removed],
                "baseline_collision_count": len(baseline_collisions),
                "collision_count": len(collisions),
                "new_or_deeper_collision_count": len(new_or_deeper),
            },
        }
        _write_json(room_dir / "quality_gates" / "room1_acceptance_details_repair.json", receipt)
        print(json.dumps(receipt, indent=2, sort_keys=True))
    finally:
        if wall_changes and not published:
            _restore_walls(wall_changes)
        # Keep unpublished scene/asset staging for forensic recovery on failure.


if __name__ == "__main__":
    main()
