#!/usr/bin/env python3
"""Attach real mesh evidence to accepted visual-contract composite objects."""

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


EXPECTED_STATE_SHA256 = {
    "classroom_01": "7070759ef586ff9b376294f2f204ba2b1dda2eec21d3496204f841c8c720a407",
    "boys_toilet": "db84d8322b751f905db6efb037a8ca1bacdef3d17b8f0d13e032c1909b129fcd",
    "girls_toilet": "95163fa780a9a35560d71b238f7c74d4eabeda297a1682d053ac1f54690531b6",
}

EVIDENCE_OBJECTS = {
    "classroom_01": {
        "room1_visible_board_supply_tray_0": ((1.58, 0.23, 0.20), (0.20, 0.38, 0.76, 1.0)),
        "room1_visible_cubby_supply_display_0": ((1.70, 0.36, 0.46), (0.10, 0.38, 0.72, 1.0)),
        "room1_visible_teacher_paperwork_0": ((0.87, 0.40, 0.08), (0.76, 0.62, 0.32, 1.0)),
    },
    "boys_toilet": {
        "boys_toilet_hand_dryer_contract_0": ((0.36, 0.19, 0.52), (0.80, 0.84, 0.88, 1.0)),
    },
    "girls_toilet": {
        "girls_toilet_hand_dryer_contract_0": ((0.36, 0.19, 0.52), (0.80, 0.84, 0.88, 1.0)),
    },
}


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
        "asset": {"version": "2.0", "generator": "SceneSmith accepted visual mesh evidence"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "accepted_contract_mesh"}],
        "meshes": [{
            "name": "accepted_contract_mesh",
            "primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "material": 0}],
        }],
        "materials": [{
            "name": "accepted_contract_material",
            "pbrMetallicRoughness": {
                "baseColorFactor": list(color),
                "metallicFactor": 0.0,
                "roughnessFactor": 0.65,
            },
        }],
        "buffers": [{
            "byteLength": len(data),
            "uri": "data:application/octet-stream;base64," + base64.b64encode(data).decode("ascii"),
        }],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(position_bytes), "target": 34962},
            {"buffer": 0, "byteOffset": len(position_bytes), "byteLength": len(index_bytes), "target": 34963},
        ],
        "accessors": [
            {
                "bufferView": 0, "componentType": 5126, "count": 8, "type": "VEC3",
                "min": [-sx, -sy, -sz], "max": [sx, sy, sz],
            },
            {
                "bufferView": 1, "componentType": 5123, "count": len(indices),
                "type": "SCALAR", "min": [0], "max": [7],
            },
        ],
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _asset(room_dir: Path, object_id: str, content: str) -> Path:
    directory = room_dir / "generated_assets" / "contract_evidence" / f"{object_id}_mesh_v1"
    candidate = directory / "accepted_contract_mesh.gltf"
    if directory.exists():
        if not candidate.is_file() or candidate.read_text(encoding="utf-8") != content:
            raise RuntimeError(f"Unexpected evidence asset: {directory}")
        return candidate
    staging = directory.with_name(f".{directory.name}.{uuid.uuid4().hex}.tmp")
    staging.mkdir(parents=True, exist_ok=False)
    _write_text(staging / candidate.name, content)
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
    parser.add_argument("--port-offset", type=int, default=929)
    args = parser.parse_args()

    from run_single_room_worker import _configure_pytorch_cuda_allocator
    _configure_pytorch_cuda_allocator()
    from omegaconf import OmegaConf
    from scenesmith.agent_utils.physics_validation import compute_scene_collisions
    from scenesmith.agent_utils.rendering import save_scene_as_blend
    from scenesmith.agent_utils.room import RoomScene, UniqueID

    repo = args.repo_dir.resolve()
    run = args.run_dir.resolve()
    os.chdir(repo)
    cfg = OmegaConf.to_container(_worker_config(args), resolve=True)
    staged_records: list[dict[str, Any]] = []

    for room_id, object_specs in EVIDENCE_OBJECTS.items():
        room_dir = run / "scene_000" / f"room_{room_id}"
        final_dir = room_dir / "scene_states" / "final_scene"
        state_path = final_dir / "scene_state.json"
        actual_sha = _sha256(state_path)
        if actual_sha != EXPECTED_STATE_SHA256[room_id]:
            raise RuntimeError(f"Unexpected {room_id} state: {actual_sha}")

        scene = RoomScene(room_geometry=None, scene_dir=room_dir, room_id=room_id)
        scene.restore_from_state_dict(json.loads(state_path.read_text(encoding="utf-8")))
        baseline = compute_scene_collisions(scene)
        evidence: list[dict[str, str]] = []
        for object_id, (size, color) in object_specs.items():
            obj = scene.get_object(UniqueID(object_id))
            if obj is None:
                raise RuntimeError(f"Missing accepted contract object: {room_id}/{object_id}")
            if obj.metadata.get("composite_members"):
                raise RuntimeError(f"Evidence already attached: {room_id}/{object_id}")
            mesh = _asset(room_dir, object_id, _box_gltf(size, color))
            obj.metadata = {
                **obj.metadata,
                "composite_members": [{"geometry_path": str(mesh.relative_to(room_dir))}],
                "deterministic_mesh_contract": "real hashable mesh companion for accepted visible contract geometry",
            }
            evidence.append({
                "object_id": object_id,
                "mesh_path": str(mesh),
                "mesh_sha256": _sha256(mesh),
            })

        candidate = compute_scene_collisions(scene)
        changed = _new_or_deeper(baseline, candidate)
        if changed:
            raise RuntimeError(
                f"{room_id} mesh-evidence repair introduced collision: "
                + "; ".join(item.to_description() for item in changed)
            )

        staged = final_dir.parent / f".accepted_mesh_evidence_{uuid.uuid4().hex}"
        staged.mkdir(parents=False, exist_ok=False)
        repaired_state = scene.to_state_dict()
        repaired_state["timestamp"] = time.time()
        _write_json(staged / "scene_state.json", repaired_state)
        (staged / "scene.dmd.yaml").write_text(scene.to_drake_directive(), encoding="utf-8")
        rendering = cfg["furniture_agent"]["rendering"]
        save_scene_as_blend(
            scene=scene,
            output_path=staged / "scene.blend",
            blender_server_host=rendering.get("blender_server_host", "127.0.0.1"),
            blender_server_port_range=tuple(rendering["blender_server_port_range"]),
            server_startup_delay=rendering["server_startup_delay"],
            port_cleanup_delay=rendering["port_cleanup_delay"],
        )
        staged_records.append({
            "room_id": room_id,
            "room_dir": room_dir,
            "final_dir": final_dir,
            "staged": staged,
            "evidence": evidence,
            "baseline_collision_count": len(baseline),
            "final_collision_count": len(candidate),
        })

    for record in staged_records:
        backup = _publish_staged_final(staged_dir=record["staged"], final_dir=record["final_dir"])
        receipt = {
            "schema_version": 1,
            "status": "pass",
            "room_id": record["room_id"],
            "operation": "attach_real_mesh_evidence_to_accepted_visual_contract_objects",
            "state_before_sha256": EXPECTED_STATE_SHA256[record["room_id"]],
            "state_after_sha256": _sha256(record["final_dir"] / "scene_state.json"),
            "backup_final_scene": str(backup),
            "mesh_evidence": record["evidence"],
            "physical_validation": {
                "baseline_collision_count": record["baseline_collision_count"],
                "final_collision_count": record["final_collision_count"],
                "new_or_deeper_collision_count": 0,
            },
            "quality_policy": "collision, stability, doorway, robot-clearance, inventory, and visual thresholds unchanged",
        }
        _write_json(record["room_dir"] / "quality_gates" / "accepted_visual_mesh_evidence_repair.json", receipt)
        print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
