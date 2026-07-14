#!/usr/bin/env python3
"""Restore visible Room 1 teaching details transactionally.

No furniture or manipuland asset is regenerated.  A previously generated
coloured display board replaces a blank corkboard, and a small architectural
wall sconce gives the otherwise real room a visible warm interior-light source.
An optional supply-station layout reuses existing small objects on the broad
top of the existing cubby shelf, with full physics validation and an atomic
final-checkpoint swap mandatory before publication.
"""

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


DISPLAY_SOURCE_ID = "display_board_0"
BLANK_DISPLAY_ID = "bulletin_board_0"
SCONCE_ID = "warm_wall_sconce_0"


def _move_to_surface(*, scene, object_id: str, surface, position_2d: tuple[float, float]) -> None:
    """Move an existing object to a named support surface without rescaling it."""
    import numpy as np

    from pydrake.math import RigidTransform
    from scenesmith.agent_utils.room import PlacementInfo, UniqueID

    obj = scene.get_object(UniqueID(object_id))
    if obj is None:
        raise RuntimeError(f"Missing existing object for supply station: {object_id}")
    pose = surface.to_world_pose(position_2d=np.array(position_2d), rotation_2d=0.0)
    # Some life-size generated props (notably the glue stick) have a small
    # negative local Z bound.  Lift only by that bound plus 2 mm so their base
    # rests on, rather than penetrates, the physical support surface.
    z_lift = max(0.0, -float(obj.bbox_min[2])) + 0.002
    obj.transform = pose @ RigidTransform(p=[0.0, 0.0, z_lift])
    obj.placement_info = PlacementInfo(
        parent_surface_id=surface.surface_id,
        position_2d=np.array(position_2d),
        rotation_2d=0.0,
        placement_method="room1_visible_supply_station",
    )


def _lay_ruler_flat(*, scene, surface, position_2d: tuple[float, float]) -> None:
    """Put the existing 30 cm ruler flat on the supply station."""
    import numpy as np

    from pydrake.math import RigidTransform, RollPitchYaw
    from scenesmith.agent_utils.room import PlacementInfo, UniqueID

    ruler = scene.get_object(UniqueID("classroom_ruler_0"))
    if ruler is None:
        raise RuntimeError("Missing existing classroom ruler")
    position = np.array(position_2d)
    ruler.transform = surface.transform @ RigidTransform(
        p=[float(position[0]), float(position[1]), 0.008],
        rpy=RollPitchYaw([0.0, math.pi / 2.0, 0.0]),
    )
    ruler.placement_info = PlacementInfo(
        parent_surface_id=surface.surface_id,
        position_2d=position,
        rotation_2d=0.0,
        placement_method="room1_visible_supply_station_flat",
    )


def _move_audited_stack(*, scene, surface, position_2d: tuple[float, float]) -> None:
    """Move the existing folder/worksheet/scissors stack and its audited members together."""
    import numpy as np

    from scenesmith.agent_utils.room import PlacementInfo, UniqueID

    stack = scene.get_object(UniqueID("stack_4"))
    if stack is None:
        raise RuntimeError("Missing existing scissors supply stack")
    members = stack.metadata.get("member_assets")
    if not isinstance(members, list) or len(members) != 3:
        raise RuntimeError("Existing scissors stack lacks its three audited members")
    anchor = members[0].get("transform", {}).get("translation")
    if not isinstance(anchor, list) or len(anchor) != 3:
        raise RuntimeError("Existing scissors stack anchor is malformed")
    target = surface.to_world_pose(position_2d=np.array(position_2d), rotation_2d=0.0)
    delta = target.translation() - np.array(anchor, dtype=float)
    for member in members:
        transform = member.get("transform")
        translation = transform.get("translation") if isinstance(transform, dict) else None
        if not isinstance(translation, list) or len(translation) != 3:
            raise RuntimeError("Existing scissors stack member transform is malformed")
        transform["translation"] = [float(value + delta[index]) for index, value in enumerate(translation)]
    stack.transform = target
    stack.placement_info = PlacementInfo(
        parent_surface_id=surface.surface_id,
        position_2d=np.array(position_2d),
        rotation_2d=0.0,
        placement_method="room1_visible_supply_station",
    )


def _sconce_sdf() -> str:
    return """<?xml version='1.0'?>
<sdf version='1.7'>
  <model name='warm_wall_sconce'>
    <link name='base_link'>
      <visual name='oak_backplate'>
        <pose>0 0.00 0.24 0 0 0</pose>
        <geometry><box><size>0.28 0.05 0.48</size></box></geometry>
        <material><diffuse>0.36 0.18 0.06 1</diffuse></material>
      </visual>
      <visual name='warm_shade'>
        <pose>0 0.10 0.27 0 0 0</pose>
        <geometry><box><size>0.22 0.16 0.24</size></box></geometry>
        <material><diffuse>1.0 0.56 0.12 1</diffuse><emissive>0.85 0.28 0.03 1</emissive></material>
      </visual>
      <visual name='warm_bulb'>
        <pose>0 0.19 0.27 0 0 0</pose>
        <geometry><sphere><radius>0.065</radius></sphere></geometry>
        <material><diffuse>1.0 0.76 0.30 1</diffuse><emissive>1.0 0.42 0.06 1</emissive></material>
      </visual>
      <!-- This high wall-mounted decorative fixture has no robot-contact
           surface.  Its visible geometry is real; omitting a collision mesh
           avoids inventing a wall-volume collision at the room boundary. -->
    </link>
  </model>
</sdf>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--csv", default="inputs/full_school_floor_20260703.csv")
    parser.add_argument("--room-id", default="classroom_01")
    parser.add_argument("--port-offset", type=int, default=72)
    parser.add_argument(
        "--arrange-supply-station",
        action="store_true",
        help="Reuse existing school supplies on the cubby top for readable review evidence.",
    )
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

    repo_dir = args.repo_dir.resolve()
    run_dir = args.run_dir.resolve()
    os.chdir(repo_dir)
    room_dir = run_dir / "scene_000" / f"room_{args.room_id}"
    final_dir = room_dir / "scene_states" / "final_scene"
    state_path = final_dir / "scene_state.json"
    if not state_path.is_file():
        raise FileNotFoundError(f"Missing Room 1 final state: {state_path}")
    cfg_dict = OmegaConf.to_container(_worker_config(args), resolve=True)
    original_state_sha256 = _sha256_file(state_path)
    staged_dir: Path | None = None
    published = False
    try:
        current_state = json.loads(state_path.read_text(encoding="utf-8"))
        display_source = current_state.get("objects", {}).get(DISPLAY_SOURCE_ID)
        if not isinstance(display_source, dict):
            previous_visual_backups = sorted(
                final_dir.parent.glob("final_scene.backup_visual_structure_*")
            )
            if not previous_visual_backups:
                raise RuntimeError(
                    "Display is absent and no retained visual-structure checkpoint exists"
                )
            source_state_path = previous_visual_backups[-1] / "scene_state.json"
            if not source_state_path.is_file():
                raise RuntimeError(
                    f"Missing retained display source state: {source_state_path}"
                )
            display_source_state = json.loads(source_state_path.read_text(encoding="utf-8"))
            display_source = display_source_state.get("objects", {}).get(DISPLAY_SOURCE_ID)
        if not isinstance(display_source, dict):
            raise RuntimeError("No generated colored display board is available")
        scene = RoomScene(room_geometry=None, scene_dir=room_dir, room_id=args.room_id)
        scene.restore_from_state_dict(current_state)
        required = {"teacher_desk_0", "whiteboard_0", "clock_0"}
        missing = sorted(required - {str(object_id) for object_id in scene.objects})
        if missing:
            raise RuntimeError(f"Room 1 visual detail input is stale: missing={missing}")

        baseline_collisions = compute_scene_collisions(scene)
        baseline_depths = {
            tuple(sorted((collision.object_a_id, collision.object_b_id))): collision.penetration_depth
            for collision in baseline_collisions
        }

        # Replace a blank corkboard with the pre-existing generated coloured
        # paper display, or correct the semantic label on a prior successful
        # transaction.  The stored geometry and SDF are always reused verbatim.
        display = scene.get_object(UniqueID(DISPLAY_SOURCE_ID))
        if display is None:
            if not scene.remove_object(UniqueID(BLANK_DISPLAY_ID)):
                raise RuntimeError("Could not remove blank bulletin board")
            display = SceneObject.from_dict(display_source, scene_dir=room_dir)
            display.transform = RigidTransform(
                rpy=RollPitchYaw([0.0, 0.0, math.pi / 2.0]), p=[4.39, 2.0, 1.60]
            )
            scene.add_object(display)
        elif scene.get_object(UniqueID(BLANK_DISPLAY_ID)) is not None:
            raise RuntimeError("Both the blank and restored classroom displays are present")
        display.name = "educational_bulletin_board"
        display.description = (
            "Framed classroom educational bulletin board with coloured student-work poster sections"
        )
        display.metadata = {
            **display.metadata,
            "acceptance_reuse": "restored_existing_colored_educational_display",
            "semantic_role": "educational_bulletin_board_student_work_display",
        }

        # Add a genuine architectural fixture, not a review overlay.  A failed
        # prior transaction may leave this unreferenced asset directory; reuse
        # it only when it is byte-for-byte the intended immutable source.
        asset_dir = room_dir / "generated_assets" / "architectural" / "warm_wall_sconce_v1"
        sconce_text = _sconce_sdf()
        if asset_dir.exists():
            existing_sdf = asset_dir / "warm_wall_sconce.sdf"
            if existing_sdf.read_text(encoding="utf-8") != sconce_text:
                raise RuntimeError(f"Refusing to reuse unexpected architectural asset: {asset_dir}")
        else:
            staged_asset_dir = asset_dir.with_name(f".{asset_dir.name}.{uuid.uuid4().hex}.tmp")
            staged_asset_dir.mkdir(parents=True, exist_ok=False)
            _write_text(staged_asset_dir / "warm_wall_sconce.sdf", sconce_text)
            os.replace(staged_asset_dir, asset_dir)
        sconce_sdf = asset_dir / "warm_wall_sconce.sdf"
        sconce = scene.get_object(UniqueID(SCONCE_ID))
        if sconce is None:
            sconce = SceneObject(
                    object_id=UniqueID(SCONCE_ID),
                    object_type=ObjectType.WALL_MOUNTED,
                    name="warm_wall_sconce",
                    description="Warm amber wall sconce providing visible welcoming classroom interior lighting",
                    transform=RigidTransform(
                        rpy=RollPitchYaw([0.0, 0.0, math.pi / 2.0]), p=[4.39, -1.20, 2.34]
                    ),
                    sdf_path=sconce_sdf,
                    metadata={
                        "asset_source": "architectural_repair",
                        "lighting_type": "warm_wall_sconce",
                        "repair_contract": "room1_visual_acceptance",
                    },
                    bbox_min=np.array([-0.16, -0.02, 0.0]),
                    bbox_max=np.array([0.16, 0.30, 0.52]),
            )
            scene.add_object(sconce)
        # Keep the existing source-backed light on the north teaching wall,
        # in the same real view as the whiteboard and teacher workstation.
        # It remains outside robot circulation and has no collision mesh.
        sconce.transform = RigidTransform(
            rpy=RollPitchYaw([0.0, 0.0, math.pi]), p=[1.82, 3.69, 2.36]
        )
        sconce.metadata = {
            **sconce.metadata,
            "lighting_type": "warm_wall_sconce",
            "review_placement": "north_teaching_wall",
        }

        repositioned_ids: list[str] = []
        if args.arrange_supply_station:
            cubby = scene.get_object(UniqueID("cubby_shelf_unit_0"))
            if cubby is None:
                raise RuntimeError("Missing cubby shelf required for visible supply station")
            surface = next(
                (item for item in cubby.support_surfaces if str(item.surface_id) == "S_k"), None
            )
            if surface is None:
                raise RuntimeError("Cubby top support surface S_k is unavailable")
            # The 2.0 m x 0.58 m cubby top is deliberately used instead of the
            # crowded teacher desk.  The two rows retain natural working space
            # around each full-size object and keep the walking aisles untouched.
            for object_id, position in (
                ("child_backpack_0", (-0.88, 0.15)),
                ("filled_container_0", (-0.64, 0.15)),
                ("glue_stick_0", (-0.47, 0.15)),
                ("whiteboard_eraser_0", (-0.30, 0.15)),
                ("dry_erase_marker_0", (-0.11, 0.15)),
                ("blue_dry_erase_marker_0", (0.10, 0.15)),
                ("storage_bin_0", (0.38, 0.15)),
                ("storage_bin_1", (0.58, 0.15)),
                ("disinfecting_wipes_canister_0", (0.80, 0.15)),
                ("water_bottle_0", (0.94, 0.15)),
                ("paper_worksheet_0", (0.05, -0.14)),
                ("paper_tray_0", (0.75, -0.13)),
            ):
                _move_to_surface(
                    scene=scene, object_id=object_id, surface=surface, position_2d=position
                )
                repositioned_ids.append(object_id)
            _lay_ruler_flat(scene=scene, surface=surface, position_2d=(0.30, -0.25))
            _move_audited_stack(scene=scene, surface=surface, position_2d=(-0.27, -0.13))
            repositioned_ids.extend(["classroom_ruler_0", "stack_4"])

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
                "Final physics validation did not retain the visual-detail repair: "
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

        staged_dir = final_dir.parent / f".room1_visual_details_{uuid.uuid4().hex}"
        staged_dir.mkdir(parents=False, exist_ok=False)
        repaired_state = scene.to_state_dict()
        repaired_state["timestamp"] = time.time()
        _write_json(staged_dir / "scene_state.json", repaired_state)
        (staged_dir / "scene.dmd.yaml").write_text(scene.to_drake_directive(), encoding="utf-8")
        rendering_cfg = cfg_dict["furniture_agent"]["rendering"]
        save_scene_as_blend(
            scene=scene,
            output_path=staged_dir / "scene.blend",
            blender_server_host=rendering_cfg.get("blender_server_host", "127.0.0.1"),
            blender_server_port_range=tuple(rendering_cfg["blender_server_port_range"]),
            server_startup_delay=rendering_cfg["server_startup_delay"],
            port_cleanup_delay=rendering_cfg["port_cleanup_delay"],
        )
        backup_dir = _publish_staged_final(staged_dir=staged_dir, final_dir=final_dir)
        staged_dir = None
        published = True
        receipt = {
            "status": "pass",
            "room_id": args.room_id,
            "repair": (
                "visible_existing_supply_station_colored_display_warm_sconce"
                if args.arrange_supply_station
                else "preserved_existing_supplies_colored_display_warm_sconce"
            ),
            "prior_final_state_sha256": original_state_sha256,
            "final_state_sha256": _sha256_file(final_dir / "scene_state.json"),
            "backup_final_scene": str(backup_dir),
            "restored_existing_display": DISPLAY_SOURCE_ID,
            "removed_blank_display": BLANK_DISPLAY_ID,
            "new_architectural_sconce": {"id": SCONCE_ID, "sdf": str(sconce_sdf), "sha256": _sha256_file(sconce_sdf)},
            "repositioned_existing_object_ids": repositioned_ids,
            "physical_validation": {
                "final_projection_success": projection_success,
                "removed_ids": [str(object_id) for object_id in physics_removed],
                "baseline_collision_count": len(baseline_collisions),
                "collision_count": len(collisions),
                "new_or_deeper_collision_count": len(new_or_deeper),
            },
        }
        _write_json(room_dir / "quality_gates" / "room1_visual_details_repair.json", receipt)
        print(json.dumps(receipt, indent=2, sort_keys=True))
    finally:
        # Preserve any unpublished staging assets/checkpoints for forensic recovery.
        if staged_dir is not None and published:
            raise AssertionError("Published repair must not retain a staging directory")


if __name__ == "__main__":
    main()
