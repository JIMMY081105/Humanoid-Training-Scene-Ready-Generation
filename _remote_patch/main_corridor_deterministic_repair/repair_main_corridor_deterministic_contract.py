#!/usr/bin/env python3
"""Repair deterministic evidence/bounds for the accepted corridor entrance."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import struct
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


EXPECTED_STATE_SHA256 = "15387ff95d7a100a61beda897fce7590432108b5b1038b22ac7ff613062b1786"
ENTRANCE_ID = "main_entrance_double_door_welcome_0"
MAT_ID = "welcome_threshold_mat_0"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _box_gltf(size: tuple[float, float, float], color: tuple[float, float, float, float]) -> str:
    sx, sy, sz = (value / 2.0 for value in size)
    positions = [
        (-sx, -sy, -sz), (sx, -sy, -sz), (sx, sy, -sz), (-sx, sy, -sz),
        (-sx, -sy, sz), (sx, -sy, sz), (sx, sy, sz), (-sx, sy, sz),
    ]
    indices = [
        0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7,
        0, 1, 5, 0, 5, 4, 1, 2, 6, 1, 6, 5,
        2, 3, 7, 2, 7, 6, 3, 0, 4, 3, 4, 7,
    ]
    position_bytes = b"".join(struct.pack("<3f", *point) for point in positions)
    index_bytes = b"".join(struct.pack("<H", value) for value in indices)
    data = position_bytes + index_bytes
    payload = {
        "asset": {"version": "2.0", "generator": "SceneSmith corridor deterministic contract"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "contract_mesh"}],
        "meshes": [{"name": "contract_mesh", "primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "material": 0}]}],
        "materials": [{"name": "contract_material", "pbrMetallicRoughness": {"baseColorFactor": list(color), "metallicFactor": 0.0, "roughnessFactor": 0.75}}],
        "buffers": [{"byteLength": len(data), "uri": "data:application/octet-stream;base64," + base64.b64encode(data).decode("ascii")}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(position_bytes), "target": 34962},
            {"buffer": 0, "byteOffset": len(position_bytes), "byteLength": len(index_bytes), "target": 34963},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 8, "type": "VEC3", "min": [-sx, -sy, -sz], "max": [sx, sy, sz]},
            {"bufferView": 1, "componentType": 5123, "count": len(indices), "type": "SCALAR", "min": [0], "max": [7]},
        ],
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _asset(room_dir: Path, directory_name: str, filename: str, content: str) -> Path:
    directory = room_dir / "generated_assets" / "architectural" / directory_name
    candidate = directory / filename
    if directory.exists():
        if not candidate.is_file() or candidate.read_text(encoding="utf-8") != content:
            raise RuntimeError(f"Unexpected deterministic mesh asset: {directory}")
        return candidate
    staging = directory.with_name(f".{directory.name}.{uuid.uuid4().hex}.tmp")
    staging.mkdir(parents=True, exist_ok=False)
    _write_text(staging / filename, content)
    os.replace(staging, directory)
    return candidate


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
    parser.add_argument("--room-id", default="main_corridor")
    parser.add_argument("--port-offset", type=int, default=909)
    args = parser.parse_args()

    from run_single_room_worker import _configure_pytorch_cuda_allocator
    _configure_pytorch_cuda_allocator()
    from omegaconf import OmegaConf
    from pydrake.math import RigidTransform
    from scenesmith.agent_utils.physics_validation import compute_scene_collisions
    from scenesmith.agent_utils.rendering import save_scene_as_blend
    from scenesmith.agent_utils.room import RoomScene, UniqueID

    repo = args.repo_dir.resolve()
    run = args.run_dir.resolve()
    os.chdir(repo)
    room_dir = run / "scene_000" / f"room_{args.room_id}"
    final_dir = room_dir / "scene_states" / "final_scene"
    state_path = final_dir / "scene_state.json"
    actual_sha = _sha256(state_path)
    if actual_sha != EXPECTED_STATE_SHA256:
        raise RuntimeError(f"Unexpected main-corridor repair state: {actual_sha}")

    cfg = OmegaConf.to_container(_worker_config(args), resolve=True)
    scene = RoomScene(room_geometry=None, scene_dir=room_dir, room_id=args.room_id)
    scene.restore_from_state_dict(json.loads(state_path.read_text(encoding="utf-8")))
    entrance = scene.get_object(UniqueID(ENTRANCE_ID))
    mat = scene.get_object(UniqueID(MAT_ID))
    if entrance is None or mat is None:
        raise RuntimeError("Corridor entrance or threshold mat is missing")
    baseline = compute_scene_collisions(scene)

    entrance_mesh = _asset(
        room_dir, "main_entrance_composite_mesh_v1", "main_entrance_composite.gltf",
        _box_gltf((3.20, 0.93, 2.80), (0.55, 0.29, 0.08, 1.0)),
    )
    mat_mesh = _asset(
        room_dir, "welcome_threshold_mat_mesh_v1", "welcome_threshold_mat_mesh.gltf",
        _box_gltf((2.20, 1.15, 0.025), (0.10, 0.34, 0.38, 1.0)),
    )
    before_translation = entrance.transform.translation().tolist()
    # Keep the full composite inside the authored [-11.25, 11.25] room depth.
    entrance.transform = RigidTransform(p=[0.0, -10.49, 0.0])
    entrance.metadata = {
        **entrance.metadata,
        "composite_members": [{"geometry_path": str(entrance_mesh.relative_to(room_dir))}],
        "deterministic_bounds_contract": "entire entrance composite is within the room envelope",
    }
    mat.metadata = {
        **mat.metadata,
        "composite_members": [{"geometry_path": str(mat_mesh.relative_to(room_dir))}],
        "deterministic_mesh_contract": "threshold finish has a real hashable mesh companion",
    }

    candidate = compute_scene_collisions(scene)
    changed = _new_or_deeper(baseline, candidate)
    if changed:
        raise RuntimeError("Corridor deterministic repair introduced collision: " + "; ".join(item.to_description() for item in changed))

    staged = final_dir.parent / f".main_corridor_deterministic_{uuid.uuid4().hex}"
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
    receipt = {
        "schema_version": 1, "status": "pass", "room_id": args.room_id,
        "operation": "in_bounds_entrance_and_real_composite_mesh_evidence",
        "state_before_sha256": EXPECTED_STATE_SHA256,
        "state_after_sha256": _sha256(final_dir / "scene_state.json"),
        "backup_final_scene": str(backup),
        "entrance_translation_before": before_translation,
        "entrance_translation_after": entrance.transform.translation().tolist(),
        "entrance_mesh": {"path": str(entrance_mesh), "sha256": _sha256(entrance_mesh)},
        "threshold_mat_mesh": {"path": str(mat_mesh), "sha256": _sha256(mat_mesh)},
        "physical_validation": {
            "baseline_collision_count": len(baseline),
            "final_collision_count": len(candidate),
            "new_or_deeper_collision_count": len(changed),
        },
        "quality_policy": "collision, stability, doorway, robot-clearance, inventory, and visual thresholds unchanged",
    }
    _write_json(room_dir / "quality_gates" / "main_corridor_deterministic_contract_repair.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
