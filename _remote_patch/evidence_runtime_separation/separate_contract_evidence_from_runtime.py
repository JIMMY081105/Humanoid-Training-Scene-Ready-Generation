#!/usr/bin/env python3
"""Keep deterministic mesh evidence without importing it as duplicate runtime geometry."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

from repair_room1_acceptance_details import _publish_staged_final, _worker_config, _write_json


EXPECTED_STATE_SHA256 = {
    "classroom_01": "cb82eaa10f91a468b2ad830663597f53f75dc8d48f63f5ff73cb734be87e50dc",
    "boys_toilet": "88e8460ede0dd0d0c7aa2b32e2cd49869300f6114f49d1f335d4afa262bb8a73",
    "girls_toilet": "a63dd0faf972903dbe06ad9fb11a556a0ac845a7b137327035bc7db52c6a1cde",
    "main_corridor": "117d91e1dfcac76d22ba1fdb06885186060b26a6a7bab406eda1e26bd9130145",
}

TARGET_IDS = {
    "classroom_01": (
        "room1_visible_board_supply_tray_0",
        "room1_visible_cubby_supply_display_0",
        "room1_visible_teacher_paperwork_0",
    ),
    "boys_toilet": ("boys_toilet_hand_dryer_contract_0",),
    "girls_toilet": ("girls_toilet_hand_dryer_contract_0",),
    "main_corridor": (
        "corridor_central_lobby_rug_0",
        "corridor_daylight_clerestory_bank_0",
        "corridor_oak_baseboard_bank_0",
        "corridor_oak_door_trim_bank_0",
        "corridor_warm_pendant_bank_0",
        "main_entrance_double_door_welcome_0",
        "welcome_threshold_mat_0",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _rollback(record: dict[str, Any]) -> None:
    final_dir: Path = record["final_dir"]
    backup: Path = record["backup"]
    rejected = final_dir.with_name(f"final_scene.rejected_evidence_separation_{uuid.uuid4().hex}")
    os.replace(final_dir, rejected)
    os.replace(backup, final_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--port-offset", type=int, default=959)
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
    staged: list[dict[str, Any]] = []
    published: list[dict[str, Any]] = []
    complete = False
    try:
        for room_id, object_ids in TARGET_IDS.items():
            room_dir = run / "scene_000" / f"room_{room_id}"
            final_dir = room_dir / "scene_states" / "final_scene"
            state_path = final_dir / "scene_state.json"
            actual_sha = _sha256(state_path)
            if actual_sha != EXPECTED_STATE_SHA256[room_id]:
                raise RuntimeError(f"Unexpected {room_id} state: {actual_sha}")

            scene = RoomScene(room_geometry=None, scene_dir=room_dir, room_id=room_id)
            scene.restore_from_state_dict(json.loads(state_path.read_text(encoding="utf-8")))
            baseline = compute_scene_collisions(scene)
            moved: list[dict[str, Any]] = []
            for object_id in object_ids:
                obj = scene.get_object(UniqueID(object_id))
                if obj is None:
                    raise RuntimeError(f"Missing object: {room_id}/{object_id}")
                members = obj.metadata.get("composite_members")
                if not isinstance(members, list) or not members:
                    raise RuntimeError(f"Missing composite evidence: {room_id}/{object_id}")
                obj.metadata = dict(obj.metadata)
                obj.metadata.pop("composite_members", None)
                obj.metadata["deterministic_mesh_evidence"] = {
                    "purpose": "hashable mesh evidence only; never instantiated as duplicate scene geometry",
                    "members": members,
                }
                moved.append({"object_id": object_id, "members": members})

            candidate = compute_scene_collisions(scene)
            changed = _new_or_deeper(baseline, candidate)
            if changed:
                raise RuntimeError(
                    f"{room_id} evidence separation introduced collision: "
                    + "; ".join(item.to_description() for item in changed)
                )

            stage = final_dir.parent / f".evidence_runtime_separation_{uuid.uuid4().hex}"
            stage.mkdir(parents=False, exist_ok=False)
            repaired_state = scene.to_state_dict()
            repaired_state["timestamp"] = time.time()
            _write_json(stage / "scene_state.json", repaired_state)
            (stage / "scene.dmd.yaml").write_text(scene.to_drake_directive(), encoding="utf-8")
            rendering = cfg["furniture_agent"]["rendering"]
            save_scene_as_blend(
                scene=scene,
                output_path=stage / "scene.blend",
                blender_server_host=rendering.get("blender_server_host", "127.0.0.1"),
                blender_server_port_range=tuple(rendering["blender_server_port_range"]),
                server_startup_delay=rendering["server_startup_delay"],
                port_cleanup_delay=rendering["port_cleanup_delay"],
            )
            staged.append({
                "room_id": room_id,
                "room_dir": room_dir,
                "final_dir": final_dir,
                "stage": stage,
                "moved": moved,
                "baseline_collision_count": len(baseline),
                "final_collision_count": len(candidate),
            })

        for record in staged:
            record["backup"] = _publish_staged_final(
                staged_dir=record["stage"], final_dir=record["final_dir"]
            )
            published.append(record)

        for record in staged:
            receipt = {
                "schema_version": 1,
                "status": "pass",
                "room_id": record["room_id"],
                "operation": "separate_deterministic_mesh_evidence_from_runtime_scene_geometry",
                "state_before_sha256": EXPECTED_STATE_SHA256[record["room_id"]],
                "state_after_sha256": _sha256(record["final_dir"] / "scene_state.json"),
                "backup_final_scene": str(record["backup"]),
                "objects": record["moved"],
                "physical_validation": {
                    "baseline_collision_count": record["baseline_collision_count"],
                    "final_collision_count": record["final_collision_count"],
                    "new_or_deeper_collision_count": 0,
                },
                "quality_policy": "all deterministic, collision, clearance, accessibility, stability, and visual thresholds unchanged",
            }
            _write_json(
                record["room_dir"] / "quality_gates" / "evidence_runtime_separation.json",
                receipt,
            )
            print(json.dumps(receipt, indent=2, sort_keys=True))
        complete = True
    finally:
        if not complete:
            for record in reversed(published):
                _rollback(record)


if __name__ == "__main__":
    main()
