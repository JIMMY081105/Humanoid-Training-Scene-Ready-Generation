#!/usr/bin/env python3
"""Repair Room 1's visual structure without regenerating any scene assets.

This narrow transactional repair removes redundant wall-mounted panels that
obscure the classroom and read as unsupported in review views.  It keeps the
freestanding teaching board, one cork display, clock, all required furniture,
and every manipuland.  It also upgrades the *existing* floor-plan window's
glass material to an opaque daylight-blue glazing so the generated window is
visibly legible.  The source window is backed up, the complete room is run
through the normal final physics gate, and a staged final scene is atomically
published only after the blend export succeeds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
import uuid

from pathlib import Path
from types import SimpleNamespace
from typing import Any


REDUNDANT_WALL_MOUNT_IDS = (
    "bulletin_board_1",
    "bulletin_board_2",
    "calendar_board_0",
    "chart_board_0",
    "display_board_0",
    "display_board_1",
    "pocket_chart_organizer_0",
    "room_plaque_0",
    "wall_sign_0",
)
REQUIRED_RETAINED_IDS = (
    "teacher_desk_0",
    "office_chair_0",
    "whiteboard_0",
    "whiteboard_1",
    "bulletin_board_0",
    "clock_0",
    "storage_cabinet_0",
    "classroom_ruler_0",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _publish_staged_final(*, staged_dir: Path, final_dir: Path) -> Path:
    for name in ("scene_state.json", "scene.dmd.yaml", "scene.blend"):
        candidate = staged_dir / name
        if not candidate.is_file() or candidate.stat().st_size <= 0:
            raise RuntimeError(f"Staged final artifact is missing or empty: {candidate}")
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    backup_dir = final_dir.with_name(f"final_scene.backup_visual_structure_{stamp}")
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
        run_name="room1_visual_structure_repair",
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


def _styled_window_payload(window_path: Path) -> dict[str, Any]:
    payload = json.loads(window_path.read_text(encoding="utf-8"))
    materials = payload.get("materials")
    if not isinstance(materials, list):
        raise RuntimeError(f"Window GLTF has no material list: {window_path}")
    glass = next(
        (
            material
            for material in materials
            if isinstance(material, dict) and material.get("name") == "window_glass"
        ),
        None,
    )
    if glass is None:
        raise RuntimeError("Generated window GLTF has no window_glass material")
    pbr = glass.setdefault("pbrMetallicRoughness", {})
    if not isinstance(pbr, dict):
        raise RuntimeError("Generated window glass PBR material is malformed")
    # The room has a real exterior opening and frame.  Opaque sky-blue glazing
    # gives that existing asset readable daylight semantics in an enclosed
    # standalone review blend without adding untracked scenery or geometry.
    pbr["baseColorFactor"] = [0.18, 0.52, 0.88, 1.0]
    pbr["roughnessFactor"] = 0.18
    pbr["metallicFactor"] = 0.0
    glass["alphaMode"] = "OPAQUE"
    glass["doubleSided"] = True
    glass["emissiveFactor"] = [0.06, 0.16, 0.32]
    return payload


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
    from omegaconf import OmegaConf

    from scenesmith.agent_utils.physical_feasibility import (
        apply_physical_feasibility_postprocessing,
    )
    from scenesmith.agent_utils.physics_validation import compute_scene_collisions
    from scenesmith.agent_utils.rendering import save_scene_as_blend
    from scenesmith.agent_utils.room import RoomScene, UniqueID

    repo_dir = args.repo_dir.resolve()
    run_dir = args.run_dir.resolve()
    os.chdir(repo_dir)
    room_dir = run_dir / "scene_000" / f"room_{args.room_id}"
    final_dir = room_dir / "scene_states" / "final_scene"
    state_path = final_dir / "scene_state.json"
    window_path = (
        run_dir
        / "scene_000"
        / "floor_plans"
        / args.room_id
        / "windows"
        / f"window_{args.room_id}"
        / "window.gltf"
    )
    if not state_path.is_file() or not window_path.is_file():
        raise FileNotFoundError(
            f"Missing Room 1 final state or floor-plan window: {state_path}, {window_path}"
        )

    cfg_dict = OmegaConf.to_container(_worker_config(args), resolve=True)
    original_state_sha256 = _sha256_file(state_path)
    original_window_sha256 = _sha256_file(window_path)
    stage_dir: Path | None = None
    window_backup: Path | None = None
    window_replaced = False
    published = False
    try:
        original_state = json.loads(state_path.read_text(encoding="utf-8"))
        scene = RoomScene(room_geometry=None, scene_dir=room_dir, room_id=args.room_id)
        scene.restore_from_state_dict(original_state)

        # The final room intentionally has supported manipulands in contact
        # with desks, shelves, and the whiteboard tray.  Preserve that audited
        # baseline and reject only a new/deeper collision caused by this repair.
        baseline_collisions = compute_scene_collisions(scene)
        baseline_collision_depths = {
            tuple(sorted((collision.object_a_id, collision.object_b_id))): collision.penetration_depth
            for collision in baseline_collisions
        }

        initial_ids = {str(object_id) for object_id in scene.objects}
        missing_removals = sorted(set(REDUNDANT_WALL_MOUNT_IDS) - initial_ids)
        missing_required = sorted(set(REQUIRED_RETAINED_IDS) - initial_ids)
        if missing_removals or missing_required:
            raise RuntimeError(
                "Room 1 visual repair input is stale; "
                f"missing removals={missing_removals}, missing required={missing_required}"
            )
        removed_wall_mounts: list[str] = []
        for object_id in REDUNDANT_WALL_MOUNT_IDS:
            if not scene.remove_object(UniqueID(object_id)):
                raise RuntimeError(f"Could not remove redundant wall mount: {object_id}")
            removed_wall_mounts.append(object_id)
        if any(scene.get_object(UniqueID(object_id)) is None for object_id in REQUIRED_RETAINED_IDS):
            raise RuntimeError("A required Room 1 teaching/inventory object was removed")

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
        new_or_deeper_collisions = [
            collision
            for collision in collisions
            if collision.penetration_depth
            > baseline_collision_depths.get(
                tuple(sorted((collision.object_a_id, collision.object_b_id))),
                -1.0,
            )
            + 1e-4
        ]
        if new_or_deeper_collisions:
            raise RuntimeError(
                "Room-wide collision validation found a new/deeper collision after "
                "visual repair: "
                + "; ".join(
                    collision.to_description() for collision in new_or_deeper_collisions
                )
            )

        stage_dir = final_dir.parent / f".room1_visual_structure_{uuid.uuid4().hex}"
        stage_dir.mkdir(parents=False, exist_ok=False)
        repaired_state = scene.to_state_dict()
        repaired_state["timestamp"] = time.time()
        _write_json(stage_dir / "scene_state.json", repaired_state)
        (stage_dir / "scene.dmd.yaml").write_text(
            scene.to_drake_directive(), encoding="utf-8"
        )

        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        window_backup = window_path.with_name(
            f"window.gltf.backup_visual_structure_{stamp}"
        )
        if window_backup.exists():
            raise RuntimeError(f"Refusing to overwrite window backup: {window_backup}")
        styled_window = window_path.with_name(f".window.gltf.{uuid.uuid4().hex}.tmp")
        _write_json(styled_window, _styled_window_payload(window_path))
        shutil.copy2(window_path, window_backup)
        os.replace(styled_window, window_path)
        window_replaced = True

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
            "repair": "remove_redundant_wall_mounts_and_style_existing_exterior_window",
            "prior_final_state_sha256": original_state_sha256,
            "final_state_sha256": _sha256_file(final_dir / "scene_state.json"),
            "backup_final_scene": str(backup_dir),
            "removed_wall_mount_ids": removed_wall_mounts,
            "retained_required_ids": list(REQUIRED_RETAINED_IDS),
            "window": {
                "path": str(window_path),
                "prior_sha256": original_window_sha256,
                "final_sha256": _sha256_file(window_path),
                "backup": str(window_backup),
                "style": "opaque_daylight_blue_existing_window_glazing",
            },
            "physical_validation": {
                "final_projection_success": projection_success,
                "removed_ids": [str(object_id) for object_id in physics_removed],
                "baseline_collision_count": len(baseline_collisions),
                "collision_count": len(collisions),
                "new_or_deeper_collision_count": len(new_or_deeper_collisions),
            },
        }
        _write_json(room_dir / "quality_gates" / "room1_visual_structure_repair.json", receipt)
        print(json.dumps(receipt, indent=2, sort_keys=True))
    finally:
        if window_replaced and not published and window_backup is not None:
            shutil.copy2(window_backup, window_path)
        # Preserve an unpublished staging directory and all backups on failure.


if __name__ == "__main__":
    main()
