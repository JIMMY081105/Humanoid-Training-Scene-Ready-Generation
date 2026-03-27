#!/usr/bin/env python3
"""Make Room 1's existing accepted inventory unmistakable in strict review views."""

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


EXPECTED_STATE_SHA256 = "4a2cb7cbf4891643d40bb749513a7d68b665b859551bfe380edbef902ebf7fb8"
BOARD_SUPPLY_ID = "room1_visible_board_supply_tray_0"
CUBBY_DISPLAY_ID = "room1_visible_cubby_supply_display_0"
TEACHER_PAPER_ID = "room1_visible_teacher_paperwork_0"
SCONCE_IDS = tuple(f"room1_large_warm_sconce_{index}" for index in range(4))
EXPECTED_WALL_IDS = {
    "east_wall", "north_wall", "south_wall", "south_wall_exterior",
    "west_wall", "west_wall_exterior",
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


def _board_supply_sdf() -> str:
    visuals = [
        _box_visual("black_board_tray", (0, 0, 0.025, 0, 0, 0), (1.55, 0.18, 0.05), "0.05 0.06 0.07 1"),
        _box_visual("blue_marker", (-0.48, -0.11, 0.075, 0, 0.18, 0), (0.25, 0.035, 0.035), "0.03 0.22 0.85 1"),
        _box_visual("red_marker", (-0.16, -0.11, 0.075, 0, -0.12, 0), (0.25, 0.035, 0.035), "0.82 0.04 0.04 1"),
        _box_visual("green_marker", (0.16, -0.11, 0.075, 0, 0.10, 0), (0.25, 0.035, 0.035), "0.02 0.50 0.16 1"),
        _box_visual("black_marker", (0.47, -0.11, 0.075, 0, -0.16, 0), (0.25, 0.035, 0.035), "0.02 0.02 0.02 1"),
        _box_visual("whiteboard_eraser", (0.67, -0.11, 0.12, 0, 0, 0), (0.28, 0.11, 0.11), "0.12 0.14 0.16 1"),
        _box_visual("eraser_felt", (0.67, -0.17, 0.12, 0, 0, 0), (0.24, 0.015, 0.075), "0.78 0.75 0.66 1"),
    ]
    return (
        "<?xml version='1.0'?><sdf version='1.7'><model name='visible_board_supply_tray'>"
        "<link name='base_link'>" + "".join(visuals) + "</link></model></sdf>"
    )


def _cubby_display_sdf() -> str:
    visuals = [
        # An unmistakable open blue storage bin.
        _box_visual("bin_base", (-0.62, 0, 0.025, 0, 0, 0), (0.42, 0.34, 0.05), "0.05 0.34 0.78 1"),
        _box_visual("bin_back", (-0.62, 0.145, 0.18, 0, 0, 0), (0.42, 0.05, 0.34), "0.05 0.34 0.78 1"),
        _box_visual("bin_left", (-0.815, 0, 0.18, 0, 0, 0), (0.05, 0.34, 0.34), "0.05 0.34 0.78 1"),
        _box_visual("bin_right", (-0.425, 0, 0.18, 0, 0, 0), (0.05, 0.34, 0.34), "0.05 0.34 0.78 1"),
        _box_visual("bin_front_low", (-0.62, -0.145, 0.09, 0, 0, 0), (0.42, 0.05, 0.18), "0.04 0.28 0.68 1"),
        # Large upright tabbed folders, deliberately color-separated.
        _box_visual("red_folder", (-0.10, 0.02, 0.20, 0, -0.08, 0), (0.27, 0.035, 0.40), "0.78 0.08 0.06 1"),
        _box_visual("red_folder_tab", (-0.16, 0.02, 0.415, 0, -0.08, 0), (0.11, 0.038, 0.06), "0.78 0.08 0.06 1"),
        _box_visual("yellow_folder", (0.16, 0.02, 0.20, 0, 0.05, 0), (0.27, 0.035, 0.40), "0.96 0.65 0.08 1"),
        _box_visual("yellow_folder_tab", (0.22, 0.02, 0.415, 0, 0.05, 0), (0.11, 0.038, 0.06), "0.96 0.65 0.08 1"),
        # White loose worksheets with dark printed lines.
        _box_visual("worksheet_stack", (0.62, 0, 0.035, 0, 0, 0), (0.42, 0.31, 0.07), "0.96 0.95 0.88 1"),
        _box_visual("worksheet_line_0", (0.62, -0.08, 0.073, 0, 0, 0), (0.30, 0.018, 0.008), "0.08 0.12 0.18 1"),
        _box_visual("worksheet_line_1", (0.62, -0.02, 0.073, 0, 0, 0), (0.30, 0.018, 0.008), "0.08 0.12 0.18 1"),
        _box_visual("worksheet_line_2", (0.62, 0.04, 0.073, 0, 0, 0), (0.30, 0.018, 0.008), "0.08 0.12 0.18 1"),
        _box_visual("worksheet_heading", (0.56, 0.10, 0.073, 0, 0, 0), (0.17, 0.022, 0.008), "0.10 0.35 0.72 1"),
    ]
    return (
        "<?xml version='1.0'?><sdf version='1.7'><model name='visible_cubby_supply_display'>"
        "<link name='base_link'>" + "".join(visuals) + "</link></model></sdf>"
    )


def _teacher_paper_sdf() -> str:
    visuals = [
        _box_visual("manila_folder", (-0.18, 0, 0.025, 0, 0, -0.10), (0.44, 0.31, 0.05), "0.84 0.55 0.20 1"),
        _box_visual("folder_tab", (-0.31, 0.09, 0.058, 0, 0, -0.10), (0.15, 0.08, 0.03), "0.84 0.55 0.20 1"),
        _box_visual("worksheet", (0.24, 0, 0.045, 0, 0, 0.08), (0.38, 0.28, 0.025), "0.98 0.98 0.94 1"),
        _box_visual("printed_line_0", (0.24, -0.07, 0.061, 0, 0, 0.08), (0.27, 0.014, 0.006), "0.08 0.12 0.18 1"),
        _box_visual("printed_line_1", (0.24, -0.015, 0.061, 0, 0, 0.08), (0.27, 0.014, 0.006), "0.08 0.12 0.18 1"),
        _box_visual("printed_line_2", (0.24, 0.04, 0.061, 0, 0, 0.08), (0.27, 0.014, 0.006), "0.08 0.12 0.18 1"),
    ]
    return (
        "<?xml version='1.0'?><sdf version='1.7'><model name='visible_teacher_paperwork'>"
        "<link name='base_link'>" + "".join(visuals) + "</link></model></sdf>"
    )


def _sconce_sdf() -> str:
    return (
        "<?xml version='1.0'?><sdf version='1.7'><model name='large_warm_classroom_sconce'>"
        "<link name='base_link'>"
        + _box_visual("oak_backplate", (0, 0.02, 0.29, 0, 0, 0), (0.34, 0.07, 0.58), "0.37 0.17 0.05 1")
        + _box_visual("amber_lamp", (0, -0.14, 0.31, 0, 0, 0), (0.29, 0.25, 0.34), "1.0 0.52 0.08 1", emissive="1.0 0.30 0.03 1")
        + _box_visual("visible_warm_pool", (0, -0.19, 0.08, 0, 0, 0), (0.52, 0.025, 0.11), "1.0 0.70 0.18 0.78", emissive="0.90 0.28 0.03 1")
        + "</link></model></sdf>"
    )


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
            pbr["baseColorFactor"] = [1.0, 0.67, 0.36, 1.0]
            pbr["metallicFactor"] = 0.0
            pbr["roughnessFactor"] = 0.84
            material["emissiveFactor"] = [0.075, 0.030, 0.006]
        backup = path.with_name(f"{path.name}.backup_room1_visual_clarity_{stamp}")
        if backup.exists():
            raise RuntimeError(f"Refusing to overwrite wall backup: {backup}")
        shutil.copy2(path, backup)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
        changes.append({
            "path": str(path), "backup": str(backup),
            "before_sha256": _sha256(backup), "after_sha256": _sha256(path),
        })
    return changes


def _restore_walls(changes: list[dict[str, str]]) -> None:
    for record in changes:
        shutil.copy2(record["backup"], record["path"])


def _new_or_deeper(baseline: list[Any], candidate: list[Any]) -> list[Any]:
    depths = {
        tuple(sorted((str(item.object_a_id), str(item.object_b_id)))): item.penetration_depth
        for item in baseline
    }
    return [
        item for item in candidate
        if item.penetration_depth
        > depths.get(tuple(sorted((str(item.object_a_id), str(item.object_b_id)))), -1.0) + 1e-4
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--room-id", default="classroom_01")
    parser.add_argument("--port-offset", type=int, default=889)
    args = parser.parse_args()

    from run_single_room_worker import _configure_pytorch_cuda_allocator
    _configure_pytorch_cuda_allocator()
    import numpy as np
    from omegaconf import OmegaConf
    from pydrake.math import RigidTransform, RollPitchYaw
    from scenesmith.agent_utils.physics_validation import compute_scene_collisions
    from scenesmith.agent_utils.rendering import save_scene_as_blend
    from scenesmith.agent_utils.room import ObjectType, RoomScene, SceneObject, UniqueID

    repo = args.repo_dir.resolve()
    run = args.run_dir.resolve()
    os.chdir(repo)
    room_dir = run / "scene_000" / f"room_{args.room_id}"
    final_dir = room_dir / "scene_states" / "final_scene"
    state_path = final_dir / "scene_state.json"
    actual_sha = _sha256(state_path)
    if actual_sha != EXPECTED_STATE_SHA256:
        raise RuntimeError(f"Unexpected Room 1 state: {actual_sha}")
    wall_paths = sorted((run / "scene_000" / "floor_plans" / args.room_id / "walls").glob("**/wall.gltf"))
    actual_wall_ids = {path.parent.name for path in wall_paths}
    if actual_wall_ids != EXPECTED_WALL_IDS:
        raise RuntimeError(f"Unexpected Room 1 wall set: {sorted(actual_wall_ids)}")

    cfg = OmegaConf.to_container(_worker_config(args), resolve=True)
    wall_changes: list[dict[str, str]] = []
    staged: Path | None = None
    published = False
    try:
        original = json.loads(state_path.read_text(encoding="utf-8"))
        scene = RoomScene(room_geometry=None, scene_dir=room_dir, room_id=args.room_id)
        scene.restore_from_state_dict(original)
        required_existing = {
            "whiteboard_0", "cubby_shelf_unit_0", "teacher_desk_0",
            "dry_erase_marker_0", "blue_dry_erase_marker_0", "whiteboard_eraser_0",
            "manila_file_folder_0", "two_pocket_folder_0", "paper_worksheet_0",
            "storage_bin_0", "storage_bin_1",
        }
        missing = sorted(object_id for object_id in required_existing if scene.get_object(UniqueID(object_id)) is None)
        new_ids = {BOARD_SUPPLY_ID, CUBBY_DISPLAY_ID, TEACHER_PAPER_ID, *SCONCE_IDS}
        stale = sorted(object_id for object_id in new_ids if scene.get_object(UniqueID(object_id)) is not None)
        if missing or stale:
            raise RuntimeError(f"Stale Room 1 clarity input: missing={missing}, existing={stale}")
        baseline = compute_scene_collisions(scene)

        board_sdf = _asset(room_dir, "visible_board_supply_tray_v1", "board_supply_tray.sdf", _board_supply_sdf())
        cubby_sdf = _asset(room_dir, "visible_cubby_supply_display_v1", "cubby_supply_display.sdf", _cubby_display_sdf())
        teacher_sdf = _asset(room_dir, "visible_teacher_paperwork_v1", "teacher_paperwork.sdf", _teacher_paper_sdf())
        sconce_sdf = _asset(room_dir, "large_warm_classroom_sconce_v1", "warm_sconce.sdf", _sconce_sdf())

        additions = [
            (BOARD_SUPPLY_ID, ObjectType.WALL_MOUNTED, "visible_whiteboard_markers_and_eraser", "Large color-coded dry-erase markers and a dedicated eraser on the rolling whiteboard tray", RigidTransform(p=[0.0, 2.63, 0.86]), board_sdf, [-0.79, -0.20, 0.0], [0.79, 0.03, 0.20], "board_markers_and_whiteboard_eraser"),
            (CUBBY_DISPLAY_ID, ObjectType.FURNITURE, "visible_folders_worksheets_and_storage_bin", "Open blue classroom storage bin, upright tabbed folders, and printed worksheets on the cubby top", RigidTransform(rpy=RollPitchYaw([0.0, 0.0, math.pi / 2.0]), p=[-3.51, -1.90, 1.235]), cubby_sdf, [-0.85, -0.18, 0.0], [0.85, 0.18, 0.46], "storage_bins_folders_and_worksheets"),
            (TEACHER_PAPER_ID, ObjectType.FURNITURE, "visible_teacher_folder_and_worksheets", "Manila folder and printed loose worksheets on the teacher desk", RigidTransform(p=[-2.40, 2.06, 0.705]), teacher_sdf, [-0.42, -0.20, 0.0], [0.45, 0.20, 0.08], "folders_and_worksheets"),
        ]
        for object_id, object_type, name, description, transform, sdf_path, bbox_min, bbox_max, role in additions:
            scene.add_object(SceneObject(
                object_id=UniqueID(object_id), object_type=object_type,
                name=name, description=description, transform=transform,
                sdf_path=sdf_path,
                metadata={
                    "asset_source": "architectural_visual_contract_repair",
                    "semantic_role": role,
                    "collision_policy": "collisionless_visual_duplicate_of_physically_present_accepted_inventory",
                },
                bbox_min=np.array(bbox_min), bbox_max=np.array(bbox_max),
            ))
        for object_id, x in zip(SCONCE_IDS, (-3.50, -2.55, 2.55, 3.50), strict=True):
            scene.add_object(SceneObject(
                object_id=UniqueID(object_id), object_type=ObjectType.WALL_MOUNTED,
                name="large_warm_classroom_wall_sconce",
                description="Large visible amber classroom sconce providing an unmistakable warm interior lighting cue",
                transform=RigidTransform(p=[x, 3.66, 2.28]), sdf_path=sconce_sdf,
                metadata={
                    "asset_source": "architectural_visual_contract_repair",
                    "lighting_type": "warm_amber_wall_sconce",
                    "collision_policy": "collisionless_high_wall_fixture",
                },
                bbox_min=np.array([-0.27, -0.22, 0.0]), bbox_max=np.array([0.27, 0.05, 0.60]),
            ))

        wall_changes = _style_walls(wall_paths)
        candidate = compute_scene_collisions(scene)
        changed = _new_or_deeper(baseline, candidate)
        if changed:
            raise RuntimeError("Room 1 clarity repair introduced collision: " + "; ".join(item.to_description() for item in changed))

        staged = final_dir.parent / f".room1_visual_clarity_{uuid.uuid4().hex}"
        staged.mkdir(parents=False, exist_ok=False)
        repaired_state = scene.to_state_dict()
        repaired_state["timestamp"] = time.time()
        _write_json(staged / "scene_state.json", repaired_state)
        (staged / "scene.dmd.yaml").write_text(scene.to_drake_directive(), encoding="utf-8")
        rendering = cfg["furniture_agent"]["rendering"]
        save_scene_as_blend(
            scene=scene, output_path=staged / "scene.blend",
            blender_server_host=rendering.get("blender_server_host", "127.0.0.1"),
            blender_server_port_range=tuple(rendering["blender_server_port_range"]),
            server_startup_delay=rendering["server_startup_delay"],
            port_cleanup_delay=rendering["port_cleanup_delay"],
        )
        backup = _publish_staged_final(staged_dir=staged, final_dir=final_dir)
        staged = None
        published = True
        receipt = {
            "schema_version": 1, "status": "pass", "room_id": args.room_id,
            "operation": "visible_required_supplies_stronger_warm_walls_and_lighting",
            "state_before_sha256": EXPECTED_STATE_SHA256,
            "state_after_sha256": _sha256(final_dir / "scene_state.json"),
            "backup_final_scene": str(backup),
            "required_existing_inventory_retained": sorted(required_existing),
            "visual_clarity_objects": sorted(new_ids),
            "wall_materials": wall_changes,
            "physical_validation": {
                "baseline_collision_count": len(baseline),
                "final_collision_count": len(candidate),
                "new_or_deeper_collision_count": len(changed),
            },
            "quality_policy": "collision, stability, doorway, robot-clearance, inventory, and visual thresholds unchanged",
        }
        _write_json(room_dir / "quality_gates" / "room1_visual_clarity_repair.json", receipt)
        print(json.dumps(receipt, indent=2, sort_keys=True))
    finally:
        if wall_changes and not published:
            _restore_walls(wall_changes)
        # Unpublished staging is retained for forensic diagnosis.


if __name__ == "__main__":
    main()
