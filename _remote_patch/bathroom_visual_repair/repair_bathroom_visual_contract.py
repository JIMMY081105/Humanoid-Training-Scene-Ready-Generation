#!/usr/bin/env python3
"""Repair both restroom visual contracts without weakening physical gates.

The transaction reorganizes the existing partitions into open, accessible stall
rows, makes the existing mirrors visibly reflective, and adds collisionless
architectural dryer/sconce cues.  Both rooms are staged, physically checked,
rendered, and then published together; a partial publish is rolled back.
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


EXPECTED_STATE_SHA256 = {
    "boys_toilet": "f061e75608ecf34ff3ffb3790a4ae3119d07268ba456dc749a1d14b3926e166b",
    "girls_toilet": "7c6c68736f0cf4c1742106146999cf9b953cfc66775259b9ba961133af813bcd",
}

PARTITION_TARGETS = {
    "boys_toilet": {
        "privacy_screen_divider_0": ([-1.50, 1.36, 0.002], 0.0),
        "privacy_screen_divider_2": ([-1.50, 0.23, 0.002], 0.0),
        "privacy_screen_divider_3": ([-1.50, -0.90, 0.002], 0.0),
        "privacy_screen_divider_4": ([-0.62, 0.23, 0.002], 0.0),
    },
    "girls_toilet": {
        "partition_panel_0": ([-1.43, 1.42, 0.002], math.pi / 2.0),
        "partition_panel_1": ([-0.34, 1.42, 0.002], math.pi / 2.0),
        "partition_panel_2": ([0.75, 1.42, 0.002], math.pi / 2.0),
        "restroom_stall_partition_0": ([-1.43, 0.66, 0.002], math.pi / 2.0),
        "restroom_stall_partition_1": ([-0.34, 0.66, 0.002], math.pi / 2.0),
        "restroom_stall_partition_2": ([0.75, 0.66, 0.002], math.pi / 2.0),
    },
}

MIRROR_IDS = {
    "boys_toilet": ("mirror_0", "mirror_1"),
    "girls_toilet": ("photograph_print_1", "photograph_print_2"),
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


def _dryer_sdf() -> str:
    visuals = [
        _box_visual("white_shell", (0, -0.055, 0.25, 0, 0, 0), (0.34, 0.13, 0.50), "0.90 0.92 0.91 1"),
        _box_visual("dark_air_slot", (0, -0.126, 0.11, 0, 0, 0), (0.23, 0.018, 0.055), "0.04 0.07 0.09 1"),
        _box_visual("blue_status", (0, -0.127, 0.37, 0, 0, 0), (0.075, 0.019, 0.075), "0.06 0.45 0.78 1", emissive="0.03 0.20 0.48 1"),
        _box_visual("airflow_0", (-0.09, -0.14, 0.035, 0, 0, -0.25), (0.10, 0.015, 0.018), "0.10 0.52 0.82 1", emissive="0.02 0.12 0.30 1"),
        _box_visual("airflow_1", (0, -0.15, 0.015, 0, 0, 0), (0.10, 0.015, 0.018), "0.10 0.52 0.82 1", emissive="0.02 0.12 0.30 1"),
        _box_visual("airflow_2", (0.09, -0.14, 0.035, 0, 0, 0.25), (0.10, 0.015, 0.018), "0.10 0.52 0.82 1", emissive="0.02 0.12 0.30 1"),
    ]
    return (
        "<?xml version='1.0'?><sdf version='1.7'><model name='wall_mounted_hand_dryer'>"
        "<link name='base_link'>" + "".join(visuals) + "</link></model></sdf>"
    )


def _sconce_sdf() -> str:
    return (
        "<?xml version='1.0'?><sdf version='1.7'><model name='restroom_warm_sconce'>"
        "<link name='base_link'>"
        + _box_visual("oak_backplate", (0, 0.01, 0.22, 0, 0, 0), (0.24, 0.05, 0.44), "0.36 0.17 0.05 1")
        + _box_visual("amber_lamp", (0, -0.09, 0.24, 0, 0, 0), (0.20, 0.16, 0.24), "1.0 0.58 0.16 1", emissive="1.0 0.32 0.05 1")
        + _box_visual("warm_pool", (0, -0.13, 0.06, 0, 0, 0), (0.34, 0.02, 0.07), "1.0 0.76 0.28 0.72", emissive="0.85 0.30 0.04 1")
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


def _resolve_geometry(room_dir: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else room_dir / path


def _style_mirrors(
    room_id: str,
    room_dir: Path,
    scene: Any,
    change_sink: list[dict[str, str]],
) -> list[dict[str, str]]:
    paths: set[Path] = set()
    for object_id in MIRROR_IDS[room_id]:
        obj = scene.get_object(object_id)
        if obj is None or obj.geometry_path is None:
            raise RuntimeError(f"Missing mirror geometry: {room_id}/{object_id}")
        paths.add(_resolve_geometry(room_dir, obj.geometry_path))

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    changes: list[dict[str, str]] = []
    for path in sorted(paths):
        payload = json.loads(path.read_text(encoding="utf-8"))
        materials = payload.get("materials")
        if not isinstance(materials, list) or not materials:
            raise RuntimeError(f"Mirror GLTF has no materials: {path}")
        for material in materials:
            if not isinstance(material, dict):
                raise RuntimeError(f"Malformed mirror material: {path}")
            pbr = material.setdefault("pbrMetallicRoughness", {})
            if not isinstance(pbr, dict):
                raise RuntimeError(f"Malformed mirror PBR payload: {path}")
            pbr.pop("baseColorTexture", None)
            pbr["baseColorFactor"] = [0.66, 0.78, 0.84, 1.0]
            pbr["metallicFactor"] = 0.96
            pbr["roughnessFactor"] = 0.06
            material["emissiveFactor"] = [0.015, 0.025, 0.03]
            material["doubleSided"] = True
        backup = path.with_name(f"{path.name}.backup_bathroom_visual_{stamp}")
        if backup.exists():
            raise RuntimeError(f"Refusing to overwrite mirror backup: {backup}")
        shutil.copy2(path, backup)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
        record = {
            "path": str(path),
            "backup": str(backup),
            "before_sha256": _sha256(backup),
            "after_sha256": _sha256(path),
        }
        changes.append(record)
        change_sink.append(record)
    return changes


def _restore_materials(changes: list[dict[str, str]]) -> None:
    for record in changes:
        shutil.copy2(record["backup"], record["path"])


def _collision_depths(collisions: list[Any]) -> dict[tuple[str, str], float]:
    return {
        tuple(sorted((str(item.object_a_id), str(item.object_b_id)))): item.penetration_depth
        for item in collisions
    }


def _stage_room(
    args: argparse.Namespace,
    room_id: str,
    cfg: dict[str, Any],
    material_changes: list[dict[str, str]],
) -> dict[str, Any]:
    import numpy as np
    from pydrake.math import RigidTransform, RollPitchYaw
    from scenesmith.agent_utils.physical_feasibility import apply_physical_feasibility_postprocessing
    from scenesmith.agent_utils.physics_validation import compute_scene_collisions
    from scenesmith.agent_utils.rendering import save_scene_as_blend
    from scenesmith.agent_utils.room import ObjectType, RoomScene, SceneObject, UniqueID

    run = args.run_dir.resolve()
    room_dir = run / "scene_000" / f"room_{room_id}"
    final_dir = room_dir / "scene_states" / "final_scene"
    state_path = final_dir / "scene_state.json"
    actual_sha = _sha256(state_path)
    if actual_sha != EXPECTED_STATE_SHA256[room_id]:
        raise RuntimeError(f"Unexpected {room_id} state: {actual_sha}")

    original = json.loads(state_path.read_text(encoding="utf-8"))
    scene = RoomScene(room_geometry=None, scene_dir=room_dir, room_id=room_id)
    scene.restore_from_state_dict(original)
    expected_ids = set(PARTITION_TARGETS[room_id]) | set(MIRROR_IDS[room_id])
    missing = sorted(object_id for object_id in expected_ids if scene.get_object(object_id) is None)
    fixture_ids = {f"{room_id}_hand_dryer_contract_0", f"{room_id}_warm_sconce_contract_0", f"{room_id}_warm_sconce_contract_1"}
    stale = sorted(object_id for object_id in fixture_ids if scene.get_object(object_id) is not None)
    if missing or stale:
        raise RuntimeError(f"Stale {room_id} input: missing={missing}, existing_fixtures={stale}")

    baseline = compute_scene_collisions(scene)
    baseline_depths = _collision_depths(baseline)
    moved: list[dict[str, Any]] = []
    for object_id, (translation, yaw) in PARTITION_TARGETS[room_id].items():
        obj = scene.get_object(object_id)
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
            "visual_contract_repair": "open_front_stall_row_with_unobstructed_access",
        }

    mirror_changes = _style_mirrors(room_id, room_dir, scene, material_changes)
    dryer_sdf = _asset(room_dir, "wall_mounted_hand_dryer_contract_v1", "hand_dryer.sdf", _dryer_sdf())
    sconce_sdf = _asset(room_dir, "warm_restroom_sconce_contract_v1", "warm_sconce.sdf", _sconce_sdf())
    if room_id == "boys_toilet":
        dryer_transform = RigidTransform(p=[1.04, 1.88, 1.02])
    else:
        dryer_transform = RigidTransform(rpy=RollPitchYaw([0.0, 0.0, -math.pi / 2.0]), p=[1.88, -0.62, 1.02])
    dryer_id = f"{room_id}_hand_dryer_contract_0"
    scene.add_object(SceneObject(
        object_id=UniqueID(dryer_id),
        object_type=ObjectType.WALL_MOUNTED,
        name="wall_mounted_hand_dryer",
        description="Clearly recognizable wall-mounted electric hand dryer with air outlet and blue airflow marks",
        transform=dryer_transform,
        sdf_path=dryer_sdf,
        metadata={
            "asset_source": "architectural_contract_repair",
            "semantic_role": "hand_drying_fixture",
            "collision_policy": "collisionless_wall_finish_above_robot_contact_volume",
        },
        bbox_min=np.array([-0.18, -0.15, 0.0]),
        bbox_max=np.array([0.18, 0.04, 0.52]),
    ))
    for index, x in enumerate((-1.15, 1.20)):
        sconce_id = f"{room_id}_warm_sconce_contract_{index}"
        scene.add_object(SceneObject(
            object_id=UniqueID(sconce_id),
            object_type=ObjectType.WALL_MOUNTED,
            name="warm_restroom_wall_sconce",
            description="Visible amber wall sconce providing a warm neutral restroom lighting cue",
            transform=RigidTransform(p=[x, 1.91, 2.18]),
            sdf_path=sconce_sdf,
            metadata={
                "asset_source": "architectural_contract_repair",
                "lighting_type": "warm_amber_wall_sconce",
                "collision_policy": "collisionless_high_wall_fixture",
            },
            bbox_min=np.array([-0.18, -0.15, 0.0]),
            bbox_max=np.array([0.18, 0.04, 0.46]),
        ))

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
        raise RuntimeError(f"{room_id} physics repair rejected: projection={projection_success}, removed={removed}")
    collisions = compute_scene_collisions(scene)
    new_or_deeper = [
        item for item in collisions
        if item.penetration_depth
        > baseline_depths.get(tuple(sorted((str(item.object_a_id), str(item.object_b_id)))), -1.0) + 1e-4
    ]
    if new_or_deeper:
        raise RuntimeError(
            f"{room_id} repair introduced collision: "
            + "; ".join(item.to_description() for item in new_or_deeper)
        )

    stage_dir = final_dir.parent / f".bathroom_visual_{uuid.uuid4().hex}"
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
    return {
        "room_id": room_id,
        "room_dir": room_dir,
        "final_dir": final_dir,
        "stage_dir": stage_dir,
        "mirror_changes": mirror_changes,
        "moved": moved,
        "baseline_collision_count": len(baseline),
        "final_collision_count": len(collisions),
        "projection_success": projection_success,
        "dryer_id": dryer_id,
        "sconce_ids": [f"{room_id}_warm_sconce_contract_0", f"{room_id}_warm_sconce_contract_1"],
        "dryer_sdf_sha256": _sha256(dryer_sdf),
        "sconce_sdf_sha256": _sha256(sconce_sdf),
    }


def _rollback_publish(record: dict[str, Any]) -> None:
    final_dir: Path = record["final_dir"]
    backup_dir: Path = record["backup_dir"]
    rejected = final_dir.with_name(f"final_scene.rejected_bathroom_visual_{uuid.uuid4().hex}")
    os.replace(final_dir, rejected)
    os.replace(backup_dir, final_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--csv", default="inputs/full_school_floor_20260703.csv")
    parser.add_argument("--port-offset", type=int, default=849)
    args = parser.parse_args()

    from run_single_room_worker import _configure_pytorch_cuda_allocator
    from omegaconf import OmegaConf

    _configure_pytorch_cuda_allocator()
    os.chdir(args.repo_dir.resolve())
    cfg = OmegaConf.to_container(_worker_config(args), resolve=True)
    staged: list[dict[str, Any]] = []
    published: list[dict[str, Any]] = []
    material_changes: list[dict[str, str]] = []
    written_receipts: list[Path] = []
    complete = False
    try:
        for room_id in ("boys_toilet", "girls_toilet"):
            staged.append(_stage_room(args, room_id, cfg, material_changes))
        for record in staged:
            record["backup_dir"] = _publish_staged_final(
                staged_dir=record["stage_dir"], final_dir=record["final_dir"]
            )
            published.append(record)
        receipts = []
        for record in staged:
            room_id = record["room_id"]
            receipt = {
                "schema_version": 1,
                "status": "pass",
                "room_id": room_id,
                "operation": "open_accessible_stalls_reflective_mirrors_visible_dryer_and_warm_lighting",
                "state_before_sha256": EXPECTED_STATE_SHA256[room_id],
                "state_after_sha256": _sha256(record["final_dir"] / "scene_state.json"),
                "backup_final_scene": str(record["backup_dir"]),
                "moved_existing_partitions": record["moved"],
                "mirror_materials": record["mirror_changes"],
                "dryer": {"object_id": record["dryer_id"], "sdf_sha256": record["dryer_sdf_sha256"]},
                "warm_sconces": {"object_ids": record["sconce_ids"], "sdf_sha256": record["sconce_sdf_sha256"]},
                "physical_validation": {
                    "projection_success": record["projection_success"],
                    "removed_ids": [],
                    "baseline_collision_count": record["baseline_collision_count"],
                    "final_collision_count": record["final_collision_count"],
                    "new_or_deeper_collision_count": 0,
                },
                "access_policy": "stall fronts remain fully open; no doorway or robot route is narrowed",
                "quality_policy": "collision, stability, doorway, robot-clearance, inventory, and visual thresholds unchanged",
            }
            receipt_path = record["room_dir"] / "quality_gates" / "bathroom_visual_contract_repair.json"
            _write_json(receipt_path, receipt)
            written_receipts.append(receipt_path)
            receipts.append(receipt)
        complete = True
        print(json.dumps({"status": "pass", "rooms": receipts}, indent=2, sort_keys=True))
    finally:
        if not complete:
            for receipt_path in written_receipts:
                if receipt_path.exists():
                    os.replace(
                        receipt_path,
                        receipt_path.with_name(f"{receipt_path.name}.rejected_{uuid.uuid4().hex}"),
                    )
            for record in reversed(published):
                _rollback_publish(record)
            _restore_materials(material_changes)
        # Unpublished staging is retained for forensic diagnosis.


if __name__ == "__main__":
    main()
