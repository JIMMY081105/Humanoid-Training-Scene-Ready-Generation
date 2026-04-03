#!/usr/bin/env python3
"""Reuse one verified generated fixture to close a required room inventory gap.

Only two audited gaps are supported: Classroom 2 greenery and Classroom 6's
trash bin.  The source scene/object bytes are pinned, assets and registry evidence
are copied into the target room, a collision-free perimeter pose is selected, and
the final state plus Blender scene are published atomically before release gates.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Any


SPECS = {
    "classroom_02": {
        "source_room": "classroom_03",
        "source_object": "potted_plant_0",
        "source_state_sha256": "fb305b6047abf8fab08c889705872f94cc711444561477979277125957bad2c8",
        "source_object_sha256": "bfb318077adf6397ce89ffe2c0bdec1456e0e965d22b8a859355bbff0812cbd4",
        "required_role": "plant_greenery",
    },
    "classroom_06": {
        "source_room": "classroom_05",
        "source_object": "trash_bin_0",
        "source_state_sha256": "2fdb7dcd9f90011e58a5fc75a9adc25c157c1043488b958b17eef597f6c290fb",
        "source_object_sha256": "f0598321be773af7d343b88b4aa223e470b23955b6f7a45e849a8f816bd8933b",
        "required_role": "classroom_trash_bin",
    },
}


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def tree_manifest(root: Path) -> list[dict[str, Any]]:
    return [
        {"path": str(path.relative_to(root)), "size": path.stat().st_size, "sha256": sha_file(path)}
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink()
    ]


def atomic_json(path: Path, value: object, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def copy_file_verified(source: Path, target: Path) -> None:
    if target.exists():
        if not target.is_file() or target.is_symlink() or sha_file(target) != sha_file(source):
            raise RuntimeError(f"conflicting reused asset file: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    shutil.copy2(source, temporary, follow_symlinks=False)
    if sha_file(temporary) != sha_file(source):
        raise RuntimeError(f"copied file digest mismatch: {source}")
    os.replace(temporary, target)


def copy_tree_verified(source: Path, target: Path) -> list[dict[str, Any]]:
    source_manifest = tree_manifest(source)
    if target.exists():
        if not target.is_dir() or target.is_symlink() or tree_manifest(target) != source_manifest:
            raise RuntimeError(f"conflicting reused asset directory: {target}")
        return source_manifest
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    shutil.copytree(source, temporary, symlinks=False)
    if tree_manifest(temporary) != source_manifest:
        raise RuntimeError("copied asset directory manifest mismatch")
    os.replace(temporary, target)
    return source_manifest


def rewrite_prefix(value: Any, source_prefix: str, target_prefix: str) -> Any:
    if isinstance(value, str):
        return target_prefix + value[len(source_prefix) :] if value.startswith(source_prefix) else value
    if isinstance(value, list):
        return [rewrite_prefix(item, source_prefix, target_prefix) for item in value]
    if isinstance(value, dict):
        return {key: rewrite_prefix(item, source_prefix, target_prefix) for key, item in value.items()}
    return value


def changed_collisions(baseline: list[Any], candidate: list[Any]) -> list[Any]:
    old = {
        tuple(sorted((str(item.object_a_id), str(item.object_b_id)))): float(item.penetration_depth)
        for item in baseline
    }
    return [
        item
        for item in candidate
        if float(item.penetration_depth)
        > old.get(tuple(sorted((str(item.object_a_id), str(item.object_b_id)))), -1.0) + 1e-4
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--room-id", required=True, choices=sorted(SPECS))
    parser.add_argument("--port-offset", required=True, type=int)
    args = parser.parse_args()

    repo_dir = args.repo_dir.resolve(strict=True)
    run_dir = args.run_dir.resolve(strict=True)
    sys.path.insert(0, str(repo_dir / "scripts"))
    os.chdir(repo_dir)

    from run_single_room_worker import _configure_pytorch_cuda_allocator

    _configure_pytorch_cuda_allocator()
    from omegaconf import OmegaConf
    from scenesmith.agent_utils.physics_validation import compute_scene_collisions
    from scenesmith.agent_utils.rendering import save_scene_as_blend
    from scenesmith.agent_utils.room import RoomScene
    from repair_room1_acceptance_details import (
        _publish_staged_final,
        _worker_config,
    )

    spec = SPECS[args.room_id]
    scene_dir = run_dir / "scene_000"
    source_room = scene_dir / f"room_{spec['source_room']}"
    target_room = scene_dir / f"room_{args.room_id}"
    source_state_path = source_room / "scene_states/scene_after_ceiling_objects/scene_state.json"
    target_final = target_room / "scene_states/final_scene"
    target_state_path = target_final / "scene_state.json"
    if sha_file(source_state_path) != spec["source_state_sha256"]:
        raise RuntimeError("source scene state digest changed")
    source_state = json.loads(source_state_path.read_text(encoding="utf-8"))
    source_object = source_state["objects"][spec["source_object"]]
    if sha_bytes(canonical(source_object)) != spec["source_object_sha256"]:
        raise RuntimeError("source object digest changed")
    target_state = json.loads(target_state_path.read_text(encoding="utf-8"))
    if spec["source_object"] in target_state.get("objects", {}):
        existing = target_state["objects"][spec["source_object"]]
        binding = (existing.get("metadata") or {}).get("required_fixture_reuse") or {}
        if binding.get("source_object_sha256") != spec["source_object_sha256"]:
            raise RuntimeError("target object ID exists without the required reuse binding")
        print("REQUIRED_ROOM_FIXTURE_ALREADY_PRESENT", args.room_id, spec["source_object"])
        return 0

    geometry_relative = Path(source_object["geometry_path"])
    sdf_relative = Path(source_object["sdf_path"])
    image_relative = Path(source_object["image_path"])
    if geometry_relative.parent != sdf_relative.parent:
        raise RuntimeError("source geometry and SDF directories differ")
    source_asset_dir = source_room / geometry_relative.parent
    target_asset_dir = target_room / geometry_relative.parent
    asset_manifest = copy_tree_verified(source_asset_dir, target_asset_dir)
    copy_file_verified(source_room / image_relative, target_room / image_relative)

    source_registry_path = source_room / "generated_assets/furniture/asset_registry.json"
    target_registry_path = target_room / "generated_assets/furniture/asset_registry.json"
    source_registry = json.loads(source_registry_path.read_text(encoding="utf-8"))
    target_registry = json.loads(target_registry_path.read_text(encoding="utf-8"))
    source_entry = source_registry[spec["source_object"]]
    rewritten_entry = rewrite_prefix(source_entry, str(source_room), str(target_room))
    existing_entry = target_registry.get(spec["source_object"])
    if existing_entry is not None and existing_entry != rewritten_entry:
        raise RuntimeError("target asset registry contains a conflicting object ID")
    if existing_entry is None:
        registry_before = sha_file(target_registry_path)
        backup = target_registry_path.with_name(
            f"asset_registry.before_required_fixture.{registry_before}.json"
        )
        if not backup.exists():
            shutil.copy2(target_registry_path, backup, follow_symlinks=False)
        if sha_file(backup) != registry_before:
            raise RuntimeError("asset registry backup digest mismatch")
        target_registry[spec["source_object"]] = rewritten_entry
        atomic_json(target_registry_path, target_registry, target_registry_path.stat().st_mode & 0o777)

    base_scene = RoomScene(room_geometry=None, scene_dir=target_room, room_id=args.room_id)
    base_scene.restore_from_state_dict(target_state)
    baseline_collisions = compute_scene_collisions(base_scene)
    cloned = copy.deepcopy(source_object)
    cloned["metadata"] = {
        **(cloned.get("metadata") or {}),
        "required_fixture_reuse": {
            "source_room": spec["source_room"],
            "source_object_id": spec["source_object"],
            "source_state_sha256": spec["source_state_sha256"],
            "source_object_sha256": spec["source_object_sha256"],
            "required_role": spec["required_role"],
            "reuse_policy": "verified_generated_asset_cache_reuse",
        },
    }
    candidates = [
        (4.0, 3.30), (-4.0, 3.30), (4.0, -3.30), (-4.0, -3.30),
        (3.85, 2.85), (-3.85, 2.85), (3.85, -2.85), (-3.85, -2.85),
        (4.0, 1.80), (-4.0, 1.80), (4.0, -1.80), (-4.0, -1.80),
    ]
    selected_scene = None
    selected_pose = None
    selected_collisions = None
    candidate_failures: list[dict[str, Any]] = []
    for x, y in candidates:
        state = copy.deepcopy(target_state)
        candidate_object = copy.deepcopy(cloned)
        candidate_object["transform"] = {
            "translation": [x, y, 0.0],
            "rotation_wxyz": [1.0, 0.0, 0.0, 0.0],
        }
        state["objects"][spec["source_object"]] = candidate_object
        scene = RoomScene(room_geometry=None, scene_dir=target_room, room_id=args.room_id)
        scene.restore_from_state_dict(state)
        collisions = compute_scene_collisions(scene)
        changed = changed_collisions(baseline_collisions, collisions)
        if not changed:
            selected_scene, selected_pose, selected_collisions = scene, [x, y, 0.0], collisions
            break
        candidate_failures.append(
            {
                "pose": [x, y, 0.0],
                "new_or_deeper_collisions": [item.to_description() for item in changed],
            }
        )
    if selected_scene is None or selected_collisions is None:
        raise RuntimeError("no collision-free perimeter pose: " + json.dumps(candidate_failures))

    cfg = OmegaConf.to_container(_worker_config(args), resolve=True)
    staged = target_final.parent / f".required_fixture_{uuid.uuid4().hex}"
    staged.mkdir(parents=False, exist_ok=False)
    published = False
    try:
        final_state = selected_scene.to_state_dict()
        final_state["timestamp"] = time.time()
        atomic_json(staged / "scene_state.json", final_state, 0o600)
        (staged / "scene.dmd.yaml").write_text(
            selected_scene.to_drake_directive(), encoding="utf-8"
        )
        rendering = cfg["furniture_agent"]["rendering"]
        save_scene_as_blend(
            scene=selected_scene,
            output_path=staged / "scene.blend",
            blender_server_host=rendering.get("blender_server_host", "127.0.0.1"),
            blender_server_port_range=tuple(rendering["blender_server_port_range"]),
            server_startup_delay=rendering["server_startup_delay"],
            port_cleanup_delay=rendering["port_cleanup_delay"],
        )
        prior_state_sha256 = sha_file(target_state_path)
        backup_final = _publish_staged_final(staged_dir=staged, final_dir=target_final)
        published = True
        receipt = {
            "schema_version": 1,
            "status": "pass",
            "room_id": args.room_id,
            "operation": "reuse_verified_generated_required_fixture",
            "required_role": spec["required_role"],
            "object_id": spec["source_object"],
            "source_room": spec["source_room"],
            "source_state_sha256": spec["source_state_sha256"],
            "source_object_sha256": spec["source_object_sha256"],
            "asset_manifest": asset_manifest,
            "selected_pose_xyz": selected_pose,
            "candidate_failures_before_selection": candidate_failures,
            "baseline_collision_count": len(baseline_collisions),
            "candidate_collision_count": len(selected_collisions),
            "new_or_deeper_collision_count": 0,
            "prior_final_state_sha256": prior_state_sha256,
            "final_state_sha256": sha_file(target_final / "scene_state.json"),
            "backup_final_scene": str(backup_final),
            "asset_registry_sha256": sha_file(target_registry_path),
        }
        receipt["attestation"] = sha_bytes(canonical(receipt))
        atomic_json(
            target_room / "quality_gates/required_fixture_reuse_repair.json", receipt
        )
        print(json.dumps(receipt, sort_keys=True))
    finally:
        if staged.exists() and not published:
            shutil.rmtree(staged)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
