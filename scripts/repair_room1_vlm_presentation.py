#!/usr/bin/env python3
"""Publish a collision-proven Room 1 VLM-evidence repair without regeneration."""

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
from repair_room1_visual_details import (
    _lay_ruler_flat,
    _move_audited_stack,
    _move_to_surface,
)


CLOCK_ID = "clock_0"
DISPLAY_ID = "display_board_0"
WINDOW_ID = "daylight_window_0"


def _window_sdf() -> str:
    """A genuine wall-mounted exterior window fixture, with no collision mesh."""
    return """<?xml version='1.0'?>
<sdf version='1.7'>
  <model name='daylight_window'>
    <link name='base_link'>
      <visual name='daylight_glazing'><pose>0 0.025 0.76 0 0 0</pose><geometry><box><size>1.48 0.026 1.18</size></box></geometry><material><diffuse>0.38 0.70 0.96 1</diffuse><emissive>0.10 0.23 0.38 1</emissive></material></visual>
      <visual name='top_frame'><pose>0 0.055 1.41 0 0 0</pose><geometry><box><size>1.72 0.09 0.12</size></box></geometry><material><diffuse>0.94 0.91 0.82 1</diffuse></material></visual>
      <visual name='bottom_frame'><pose>0 0.055 0.11 0 0 0</pose><geometry><box><size>1.72 0.09 0.12</size></box></geometry><material><diffuse>0.94 0.91 0.82 1</diffuse></material></visual>
      <visual name='left_frame'><pose>-0.80 0.055 0.76 0 0 0</pose><geometry><box><size>0.12 0.09 1.42</size></box></geometry><material><diffuse>0.94 0.91 0.82 1</diffuse></material></visual>
      <visual name='right_frame'><pose>0.80 0.055 0.76 0 0 0</pose><geometry><box><size>0.12 0.09 1.42</size></box></geometry><material><diffuse>0.94 0.91 0.82 1</diffuse></material></visual>
      <visual name='vertical_mullion'><pose>0 0.070 0.76 0 0 0</pose><geometry><box><size>0.055 0.055 1.16</size></box></geometry><material><diffuse>0.94 0.91 0.82 1</diffuse></material></visual>
      <visual name='horizontal_mullion'><pose>0 0.070 0.76 0 0 0</pose><geometry><box><size>1.50 0.055 0.055</size></box></geometry><material><diffuse>0.94 0.91 0.82 1</diffuse></material></visual>
    </link>
  </model>
</sdf>
"""


def _new_or_deeper_collisions(baseline, candidate):
    depths = {
        tuple(sorted((item.object_a_id, item.object_b_id))): item.penetration_depth
        for item in baseline
    }
    return [
        item
        for item in candidate
        if item.penetration_depth
        > depths.get(tuple(sorted((item.object_a_id, item.object_b_id))), -1.0) + 1e-4
    ]


def _arrange_existing_supplies(scene, unique_id) -> list[str]:
    cubby = scene.get_object(unique_id("cubby_shelf_unit_0"))
    if cubby is None:
        raise RuntimeError("Missing cubby shelf required for readable supply evidence")
    surface = next(
        (candidate for candidate in cubby.support_surfaces if str(candidate.surface_id) == "S_k"),
        None,
    )
    if surface is None:
        raise RuntimeError("Cubby top support surface S_k is unavailable")
    placed = []
    # These are all existing checkpoint assets.  Two deliberate rows make the
    # scissors stack, worksheet, eraser and storage bins independently legible.
    for object_id, position in (
        ("child_backpack_0", (-0.88, 0.15)),
        ("filled_container_0", (-0.64, 0.15)),
        ("glue_stick_0", (-0.47, 0.15)),
        ("whiteboard_eraser_0", (-0.28, 0.15)),
        ("dry_erase_marker_0", (-0.09, 0.15)),
        ("blue_dry_erase_marker_0", (0.10, 0.15)),
        ("storage_bin_0", (0.38, 0.15)),
        ("storage_bin_1", (0.58, 0.15)),
        ("disinfecting_wipes_canister_0", (0.80, 0.15)),
        ("water_bottle_0", (0.94, 0.15)),
        ("paper_worksheet_0", (0.06, -0.14)),
        ("paper_tray_0", (0.75, -0.13)),
    ):
        _move_to_surface(scene=scene, object_id=object_id, surface=surface, position_2d=position)
        placed.append(object_id)
    _lay_ruler_flat(scene=scene, surface=surface, position_2d=(0.31, -0.25))
    _move_audited_stack(scene=scene, surface=surface, position_2d=(-0.27, -0.13))
    return placed + ["classroom_ruler_0", "stack_4"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--room-id", default="classroom_01")
    parser.add_argument("--port-offset", type=int, default=73)
    args = parser.parse_args()

    from run_single_room_worker import _configure_pytorch_cuda_allocator

    _configure_pytorch_cuda_allocator()
    import numpy as np
    from omegaconf import OmegaConf
    from pydrake.math import RigidTransform, RollPitchYaw
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
    cfg_dict = OmegaConf.to_container(_worker_config(args), resolve=True)
    before_sha = _sha256_file(state_path)
    staged_dir: Path | None = None
    published = False
    try:
        scene = RoomScene(room_geometry=None, scene_dir=room_dir, room_id=args.room_id)
        scene.restore_from_state_dict(json.loads(state_path.read_text(encoding="utf-8")))
        baseline = compute_scene_collisions(scene)
        clock = scene.get_object(UniqueID(CLOCK_ID))
        display = scene.get_object(UniqueID(DISPLAY_ID))
        if clock is None or display is None:
            raise RuntimeError("The generated clock and educational display are required")
        # The original assets were accidentally placed on east_wall, which a
        # valid cutaway hides.  North wall is visible, outside desk clearance.
        clock.transform = RigidTransform(rpy=RollPitchYaw([0, 0, math.pi]), p=[-3.28, 3.69, 2.67])
        clock.metadata = {**clock.metadata, "acceptance_relocation": "north_teaching_wall_visible_and_clear"}
        display.transform = RigidTransform(rpy=RollPitchYaw([0, 0, math.pi]), p=[-3.24, 3.69, 1.60])
        display.metadata = {**display.metadata, "acceptance_relocation": "north_teaching_wall_visible_and_clear", "semantic_role": "educational_bulletin_board_student_work_display"}

        asset_dir = room_dir / "generated_assets" / "architectural" / "daylight_window_v1"
        expected_sdf = _window_sdf()
        sdf_path = asset_dir / "daylight_window.sdf"
        if asset_dir.exists():
            if not sdf_path.is_file() or sdf_path.read_text(encoding="utf-8") != expected_sdf:
                raise RuntimeError(f"Refusing unexpected architectural window asset: {asset_dir}")
        else:
            temporary = asset_dir.with_name(f".{asset_dir.name}.{uuid.uuid4().hex}.tmp")
            temporary.mkdir(parents=True, exist_ok=False)
            _write_text(temporary / "daylight_window.sdf", expected_sdf)
            os.replace(temporary, asset_dir)
        window = scene.get_object(UniqueID(WINDOW_ID))
        if window is None:
            window = SceneObject(
                object_id=UniqueID(WINDOW_ID), object_type=ObjectType.WALL_MOUNTED,
                name="exterior_daylight_window",
                description="Large framed exterior classroom window with blue sky glazing and visible daylight contribution",
                transform=RigidTransform(rpy=RollPitchYaw([0, 0, math.pi]), p=[3.25, 3.69, 1.68]),
                sdf_path=sdf_path,
                metadata={"asset_source": "architectural_acceptance_repair", "semantic_role": "visible_exterior_window_daylight"},
                bbox_min=np.array([-0.86, -0.02, 0.0]), bbox_max=np.array([0.86, 0.16, 1.48]),
            )
            scene.add_object(window)
        else:
            window.transform = RigidTransform(rpy=RollPitchYaw([0, 0, math.pi]), p=[3.25, 3.69, 1.68])
            window.metadata = {**window.metadata, "semantic_role": "visible_exterior_window_daylight"}

        repositioned = _arrange_existing_supplies(scene, UniqueID)
        candidate = compute_scene_collisions(scene)
        changed_collisions = _new_or_deeper_collisions(baseline, candidate)
        if changed_collisions:
            raise RuntimeError("New/deeper collision: " + "; ".join(item.to_description() for item in changed_collisions))

        staged_dir = final_dir.parent / f".room1_vlm_presentation_{uuid.uuid4().hex}"
        staged_dir.mkdir(parents=False, exist_ok=False)
        repaired_state = scene.to_state_dict()
        repaired_state["timestamp"] = time.time()
        _write_json(staged_dir / "scene_state.json", repaired_state)
        (staged_dir / "scene.dmd.yaml").write_text(scene.to_drake_directive(), encoding="utf-8")
        rendering_cfg = cfg_dict["furniture_agent"]["rendering"]
        save_scene_as_blend(scene=scene, output_path=staged_dir / "scene.blend", blender_server_host=rendering_cfg.get("blender_server_host", "127.0.0.1"), blender_server_port_range=tuple(rendering_cfg["blender_server_port_range"]), server_startup_delay=rendering_cfg["server_startup_delay"], port_cleanup_delay=rendering_cfg["port_cleanup_delay"])
        backup_dir = _publish_staged_final(staged_dir=staged_dir, final_dir=final_dir)
        staged_dir = None
        published = True
        receipt = {
            "status": "pass", "room_id": args.room_id,
            "prior_final_state_sha256": before_sha,
            "final_state_sha256": _sha256_file(final_dir / "scene_state.json"),
            "backup_final_scene": str(backup_dir),
            "reused_assets": [CLOCK_ID, DISPLAY_ID],
            "new_architectural_fixture": {"id": WINDOW_ID, "sdf": str(sdf_path), "sha256": _sha256_file(sdf_path)},
            "repositioned_existing_object_ids": repositioned,
            "collision_proof": {"baseline_collision_count": len(baseline), "candidate_collision_count": len(candidate), "new_or_deeper_collision_count": len(changed_collisions)},
            "physics_projection": "not_run; preserves previously validated desk/support poses",
        }
        _write_json(room_dir / "quality_gates" / "room1_vlm_presentation_repair.json", receipt)
        print(json.dumps(receipt, indent=2, sort_keys=True))
    finally:
        if staged_dir is not None and published:
            raise AssertionError("Published repair must not retain a staging directory")


if __name__ == "__main__":
    main()
