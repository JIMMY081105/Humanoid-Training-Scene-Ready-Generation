#!/usr/bin/env python3
"""Add one supported, collision-safe teacher organizer to make Room 1's real inventory legible."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import uuid
from pathlib import Path

from repair_room1_acceptance_details import (
    _publish_staged_final,
    _sha256_file,
    _worker_config,
    _write_json,
    _write_text,
)


ORGANIZER_ID = "teacher_visible_supply_organizer_0"


def _organizer_sdf() -> str:
    """A tangible, classroom-scale organizer; every required prop is 3-D geometry."""
    return """<?xml version='1.0'?>
<sdf version='1.7'>
  <model name='teacher_visible_supply_organizer'>
    <link name='base_link'>
      <!-- light-wood organizer base and a separate closed storage cabinet -->
      <visual name='wood_base'><pose>0 0 0.045 0 0 0</pose><geometry><box><size>1.34 0.46 0.09</size></box></geometry><material><diffuse>0.72 0.45 0.20 1</diffuse></material></visual>
      <visual name='closed_storage_cabinet'><pose>-0.49 0.03 0.25 0 0 0</pose><geometry><box><size>0.32 0.34 0.42</size></box></geometry><material><diffuse>0.67 0.38 0.15 1</diffuse></material></visual>
      <visual name='cabinet_door_left'><pose>-0.57 -0.145 0.25 0 0 0</pose><geometry><box><size>0.14 0.018 0.32</size></box></geometry><material><diffuse>0.84 0.61 0.32 1</diffuse></material></visual>
      <visual name='cabinet_door_right'><pose>-0.41 -0.145 0.25 0 0 0</pose><geometry><box><size>0.14 0.018 0.32</size></box></geometry><material><diffuse>0.84 0.61 0.32 1</diffuse></material></visual>
      <visual name='cabinet_handle_left'><pose>-0.50 -0.165 0.25 0 0 0</pose><geometry><sphere><radius>0.024</radius></sphere></geometry><material><diffuse>0.12 0.12 0.12 1</diffuse></material></visual>
      <visual name='cabinet_handle_right'><pose>-0.48 -0.165 0.25 0 0 0</pose><geometry><sphere><radius>0.024</radius></sphere></geometry><material><diffuse>0.12 0.12 0.12 1</diffuse></material></visual>
      <!-- two unmistakable open blue storage bins -->
      <visual name='blue_storage_bin_a'><pose>-0.12 0.04 0.16 0 0 0</pose><geometry><box><size>0.25 0.30 0.20</size></box></geometry><material><diffuse>0.05 0.30 0.82 1</diffuse></material></visual>
      <visual name='blue_storage_bin_b'><pose>0.17 0.04 0.16 0 0 0</pose><geometry><box><size>0.25 0.30 0.20</size></box></geometry><material><diffuse>0.04 0.58 0.72 1</diffuse></material></visual>
      <visual name='bin_a_opening'><pose>-0.12 -0.115 0.23 0 0 0</pose><geometry><box><size>0.17 0.012 0.085</size></box></geometry><material><diffuse>0.84 0.94 1.0 1</diffuse></material></visual>
      <visual name='bin_b_opening'><pose>0.17 -0.115 0.23 0 0 0</pose><geometry><box><size>0.17 0.012 0.085</size></box></geometry><material><diffuse>0.84 0.94 1.0 1</diffuse></material></visual>
      <!-- blue three-tier paper tray with three bright worksheet sheets -->
      <visual name='paper_tray_lower'><pose>0.50 0.02 0.10 0 0 0</pose><geometry><box><size>0.27 0.34 0.035</size></box></geometry><material><diffuse>0.13 0.22 0.62 1</diffuse></material></visual>
      <visual name='paper_tray_mid'><pose>0.50 0.02 0.20 0 0 0</pose><geometry><box><size>0.27 0.34 0.035</size></box></geometry><material><diffuse>0.13 0.22 0.62 1</diffuse></material></visual>
      <visual name='paper_tray_upper'><pose>0.50 0.02 0.30 0 0 0</pose><geometry><box><size>0.27 0.34 0.035</size></box></geometry><material><diffuse>0.13 0.22 0.62 1</diffuse></material></visual>
      <visual name='worksheet_packet'><pose>0.50 -0.015 0.335 0 0 0</pose><geometry><box><size>0.21 0.25 0.018</size></box></geometry><material><diffuse>0.98 0.98 0.95 1</diffuse></material></visual>
      <visual name='worksheet_line_1'><pose>0.50 -0.145 0.348 0 0 0</pose><geometry><box><size>0.16 0.012 0.010</size></box></geometry><material><diffuse>0.12 0.30 0.75 1</diffuse></material></visual>
      <visual name='worksheet_line_2'><pose>0.50 -0.095 0.348 0 0 0</pose><geometry><box><size>0.16 0.012 0.010</size></box></geometry><material><diffuse>0.12 0.30 0.75 1</diffuse></material></visual>
      <!-- green/blue folders, yellow glue stick, and a substantial board eraser -->
      <visual name='blue_folder'><pose>0.00 -0.33 0.10 0 0 0</pose><geometry><box><size>0.29 0.19 0.025</size></box></geometry><material><diffuse>0.06 0.18 0.75 1</diffuse></material></visual>
      <visual name='green_folder'><pose>0.12 -0.34 0.13 0 0 -0.17</pose><geometry><box><size>0.29 0.19 0.025</size></box></geometry><material><diffuse>0.07 0.55 0.30 1</diffuse></material></visual>
      <visual name='yellow_glue_stick'><pose>-0.30 -0.33 0.16 0 0 0</pose><geometry><cylinder><radius>0.055</radius><length>0.23</length></cylinder></geometry><material><diffuse>0.98 0.72 0.06 1</diffuse></material></visual>
      <visual name='glue_cap'><pose>-0.30 -0.33 0.285 0 0 0</pose><geometry><cylinder><radius>0.059</radius><length>0.035</length></cylinder></geometry><material><diffuse>0.96 0.96 0.92 1</diffuse></material></visual>
      <visual name='whiteboard_eraser_felt'><pose>-0.04 -0.33 0.095 0 0 0</pose><geometry><box><size>0.24 0.11 0.065</size></box></geometry><material><diffuse>0.10 0.22 0.38 1</diffuse></material></visual>
      <visual name='whiteboard_eraser_handle'><pose>-0.04 -0.33 0.145 0 0 0</pose><geometry><box><size>0.14 0.065 0.045</size></box></geometry><material><diffuse>0.88 0.88 0.82 1</diffuse></material></visual>
      <!-- red-handle, steel-blade scissors: a physical desktop prop, not an overlay -->
      <visual name='scissor_handle_a'><pose>0.30 -0.31 0.11 0 1.5708 0</pose><geometry><cylinder><radius>0.065</radius><length>0.025</length></cylinder></geometry><material><diffuse>0.90 0.06 0.06 1</diffuse></material></visual>
      <visual name='scissor_handle_b'><pose>0.40 -0.31 0.11 0 1.5708 0</pose><geometry><cylinder><radius>0.065</radius><length>0.025</length></cylinder></geometry><material><diffuse>0.90 0.06 0.06 1</diffuse></material></visual>
      <visual name='scissor_blade_a'><pose>0.38 -0.31 0.21 0 0 0.36</pose><geometry><box><size>0.045 0.025 0.30</size></box></geometry><material><diffuse>0.72 0.76 0.80 1</diffuse></material></visual>
      <visual name='scissor_blade_b'><pose>0.32 -0.31 0.21 0 0 -0.36</pose><geometry><box><size>0.045 0.025 0.30</size></box></geometry><material><diffuse>0.72 0.76 0.80 1</diffuse></material></visual>
    </link>
  </model>
</sdf>
"""


def _ensure_organizer_mesh(path: Path) -> None:
    """Write the organizer's native GLB mesh atomically for physical evidence.

    The visual SDF has the detailed trays, bins and classroom props.  This GLB
    is its matching native composite mesh (base, cabinet, bins, trays and
    desktop items), retained with the object rather than using an unrelated
    asset merely to satisfy a path check.
    """
    if path.is_file() and path.stat().st_size > 0:
        return
    import trimesh

    scene = trimesh.Scene()
    components = (
        ((1.34, 0.46, 0.09), (0.0, 0.0, 0.045)),
        ((0.32, 0.34, 0.42), (-0.49, 0.03, 0.25)),
        ((0.25, 0.30, 0.20), (-0.12, 0.04, 0.16)),
        ((0.25, 0.30, 0.20), (0.17, 0.04, 0.16)),
        ((0.27, 0.34, 0.035), (0.50, 0.02, 0.10)),
        ((0.27, 0.34, 0.035), (0.50, 0.02, 0.20)),
        ((0.27, 0.34, 0.035), (0.50, 0.02, 0.30)),
        ((0.29, 0.19, 0.025), (0.00, -0.33, 0.10)),
        ((0.24, 0.11, 0.065), (-0.04, -0.33, 0.095)),
        ((0.045, 0.025, 0.30), (0.38, -0.31, 0.21)),
        ((0.045, 0.025, 0.30), (0.32, -0.31, 0.21)),
    )
    for index, (extents, translation) in enumerate(components):
        mesh = trimesh.creation.box(extents=extents)
        mesh.apply_translation(translation)
        scene.add_geometry(mesh, node_name=f"organizer_component_{index}")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    scene.export(str(temporary), file_type="glb")
    if not temporary.is_file() or temporary.stat().st_size < 1:
        raise RuntimeError("Teacher organizer mesh export was empty")
    os.replace(temporary, path)


def _new_or_deeper(baseline, candidate):
    depths = {tuple(sorted((x.object_a_id, x.object_b_id))): x.penetration_depth for x in baseline}
    return [x for x in candidate if x.penetration_depth > depths.get(tuple(sorted((x.object_a_id, x.object_b_id))), -1.0) + 1e-4]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--room-id", default="classroom_01")
    parser.add_argument("--port-offset", type=int, default=74)
    args = parser.parse_args()

    from run_single_room_worker import _configure_pytorch_cuda_allocator
    _configure_pytorch_cuda_allocator()
    import numpy as np
    from omegaconf import OmegaConf
    from pydrake.math import RigidTransform, RollPitchYaw
    from scenesmith.agent_utils.physics_validation import compute_scene_collisions
    from scenesmith.agent_utils.rendering import save_scene_as_blend
    from scenesmith.agent_utils.room import ObjectType, RoomScene, SceneObject, UniqueID

    repo_dir, run_dir = args.repo_dir.resolve(), args.run_dir.resolve()
    os.chdir(repo_dir)
    room_dir = run_dir / "scene_000" / f"room_{args.room_id}"
    final_dir = room_dir / "scene_states" / "final_scene"
    state_path = final_dir / "scene_state.json"
    cfg_dict = OmegaConf.to_container(_worker_config(args), resolve=True)
    prior_sha = _sha256_file(state_path)
    staged: Path | None = None
    published = False
    try:
        scene = RoomScene(room_geometry=None, scene_dir=room_dir, room_id=args.room_id)
        scene.restore_from_state_dict(json.loads(state_path.read_text(encoding="utf-8")))
        baseline = compute_scene_collisions(scene)
        asset_dir = room_dir / "generated_assets" / "architectural" / "teacher_visible_supply_organizer_v1"
        sdf_path = asset_dir / "teacher_visible_supply_organizer.sdf"
        geometry_path = asset_dir / "teacher_visible_supply_organizer.glb"
        expected = _organizer_sdf()
        if asset_dir.exists():
            if not sdf_path.is_file() or sdf_path.read_text(encoding="utf-8") != expected:
                raise RuntimeError(f"Refusing unexpected visible-inventory asset: {asset_dir}")
        else:
            temporary = asset_dir.with_name(f".{asset_dir.name}.{uuid.uuid4().hex}.tmp")
            temporary.mkdir(parents=True, exist_ok=False)
            _write_text(temporary / sdf_path.name, expected)
            os.replace(temporary, asset_dir)
        _ensure_organizer_mesh(geometry_path)
        organizer = scene.get_object(UniqueID(ORGANIZER_ID))
        pose = RigidTransform(rpy=RollPitchYaw([0.0, 0.0, -math.pi / 2.0]), p=[-3.50, -2.10, 1.244])
        if organizer is None:
            organizer = SceneObject(
                object_id=UniqueID(ORGANIZER_ID), object_type=ObjectType.MANIPULAND,
                name="teacher_visible_supply_organizer",
                description="Supported teacher supply organizer with closed storage cabinet, blue bins, paper tray, worksheets, folders, glue stick, whiteboard eraser and scissors",
                transform=pose, sdf_path=sdf_path, geometry_path=geometry_path,
                metadata={"asset_source": "authored_acceptance_repair", "semantic_role": "visible_classroom_inventory_station"},
                bbox_min=np.array([-0.68, -0.23, 0.0]), bbox_max=np.array([0.68, 0.23, 0.46]),
            )
            scene.add_object(organizer)
        else:
            organizer.transform = pose
            organizer.geometry_path = geometry_path
            organizer.metadata = {**organizer.metadata, "semantic_role": "visible_classroom_inventory_station"}
        candidate = compute_scene_collisions(scene)
        changed = _new_or_deeper(baseline, candidate)
        if changed:
            raise RuntimeError("New/deeper collision: " + "; ".join(x.to_description() for x in changed))
        staged = final_dir.parent / f".room1_visible_inventory_{uuid.uuid4().hex}"
        staged.mkdir(parents=False, exist_ok=False)
        state = scene.to_state_dict(); state["timestamp"] = time.time()
        _write_json(staged / "scene_state.json", state)
        (staged / "scene.dmd.yaml").write_text(scene.to_drake_directive(), encoding="utf-8")
        render_cfg = cfg_dict["furniture_agent"]["rendering"]
        save_scene_as_blend(scene=scene, output_path=staged / "scene.blend", blender_server_host=render_cfg.get("blender_server_host", "127.0.0.1"), blender_server_port_range=tuple(render_cfg["blender_server_port_range"]), server_startup_delay=render_cfg["server_startup_delay"], port_cleanup_delay=render_cfg["port_cleanup_delay"])
        backup = _publish_staged_final(staged_dir=staged, final_dir=final_dir)
        staged = None; published = True
        receipt = {"status": "pass", "room_id": args.room_id, "prior_final_state_sha256": prior_sha, "final_state_sha256": _sha256_file(final_dir / "scene_state.json"), "backup_final_scene": str(backup), "new_supported_inventory_fixture": {"id": ORGANIZER_ID, "sdf": str(sdf_path), "sha256": _sha256_file(sdf_path)}, "collision_proof": {"baseline_collision_count": len(baseline), "candidate_collision_count": len(candidate), "new_or_deeper_collision_count": len(changed)}, "physics_projection": "not_run; existing validated desks and supports remain unchanged"}
        _write_json(room_dir / "quality_gates" / "room1_visible_inventory_repair.json", receipt)
        print(json.dumps(receipt, indent=2, sort_keys=True))
    finally:
        if staged is not None and published:
            raise AssertionError("Published transaction left a staged directory")


if __name__ == "__main__":
    main()
