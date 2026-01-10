#!/usr/bin/env python3
"""Restore the one Room 1 desk pair lifted by an unsuccessful projection.

The pre-repair checkpoint is the authoritative spatial baseline.  This tool
restores only the six transforms that the projection lifted by more than 10 cm,
keeps the independently repositioned wall sconce, and refuses publication if
the candidate introduces a new or deeper Drake collision.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import time

from pathlib import Path


RESTORE_IDS = frozenset(
    {
        "student_desk_0",
        "composition_notebook_1",
        "pencil_holder_cup_0",
        "science_textbook_1",
        "vinyl_eraser_0",
        "water_0",
    }
)
INTENTIONAL_CHANGED_IDS = frozenset({"warm_wall_sconce_0"})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _translation(value: dict) -> tuple[float, float, float]:
    translation = value.get("transform", {}).get("translation")
    if not isinstance(translation, list) or len(translation) != 3:
        raise RuntimeError("Object transform is missing a 3D translation")
    return tuple(float(item) for item in translation)


def _significantly_changed(current: dict, baseline: dict) -> bool:
    now = _translation(current)
    old = _translation(baseline)
    return sum((now[index] - old[index]) ** 2 for index in range(3)) ** 0.5 > 0.1


def _collision_depths(scene) -> dict[tuple[str, str], float]:
    from scenesmith.agent_utils.physics_validation import compute_scene_collisions

    return {
        tuple(sorted((str(collision.object_a_id), str(collision.object_b_id)))): float(
            collision.penetration_depth
        )
        for collision in compute_scene_collisions(scene)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--room-dir", type=Path, required=True)
    parser.add_argument("--baseline-state", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    room_dir = args.room_dir.resolve()
    state_path = room_dir / "scene_states" / "final_scene" / "scene_state.json"
    baseline_path = args.baseline_state.resolve()
    current = json.loads(state_path.read_text(encoding="utf-8"))
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    current_objects = current.get("objects")
    baseline_objects = baseline.get("objects")
    if not isinstance(current_objects, dict) or not isinstance(baseline_objects, dict):
        raise RuntimeError("Room state object collections are malformed")
    if set(current_objects) != set(baseline_objects):
        raise RuntimeError("Projection recovery state object sets differ")

    significant = {
        object_id
        for object_id in current_objects
        if _significantly_changed(current_objects[object_id], baseline_objects[object_id])
    }
    expected = RESTORE_IDS | INTENTIONAL_CHANGED_IDS
    if significant != expected:
        raise RuntimeError(
            "Unexpected significant projection changes: "
            f"actual={sorted(significant)}, expected={sorted(expected)}"
        )

    candidate = copy.deepcopy(current)
    for object_id in RESTORE_IDS:
        candidate["objects"][object_id]["transform"] = copy.deepcopy(
            baseline_objects[object_id]["transform"]
        )

    from scenesmith.agent_utils.room import RoomScene

    baseline_scene = RoomScene(room_geometry=None, scene_dir=room_dir, room_id="classroom_01")
    baseline_scene.restore_from_state_dict(baseline)
    candidate_scene = RoomScene(room_geometry=None, scene_dir=room_dir, room_id="classroom_01")
    candidate_scene.restore_from_state_dict(candidate)
    baseline_depths = _collision_depths(baseline_scene)
    candidate_depths = _collision_depths(candidate_scene)
    new_or_deeper = {
        pair: depth
        for pair, depth in candidate_depths.items()
        if depth > baseline_depths.get(pair, -1.0) + 1e-4
    }
    if new_or_deeper:
        raise RuntimeError(f"Projection recovery introduced collision regressions: {new_or_deeper}")

    temporary = state_path.with_name(f".{state_path.name}.projection_recovery.tmp")
    temporary.write_text(json.dumps(candidate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    backup = state_path.with_name(f"{state_path.name}.projection_anomaly_{int(time.time())}")
    shutil.copy2(state_path, backup)
    os.replace(temporary, state_path)
    payload = {
        "status": "pass",
        "operation": "restore_pre_projection_desk_pair_transforms",
        "prior_state_sha256": sha256_file(backup),
        "final_state_sha256": sha256_file(state_path),
        "baseline_state": str(baseline_path),
        "backup": str(backup),
        "restored_object_ids": sorted(RESTORE_IDS),
        "retained_intentional_changed_ids": sorted(INTENTIONAL_CHANGED_IDS),
        "baseline_collision_count": len(baseline_depths),
        "collision_count": len(candidate_depths),
        "new_or_deeper_collision_count": len(new_or_deeper),
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    temp_receipt = args.receipt.with_name(f".{args.receipt.name}.tmp")
    temp_receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp_receipt, args.receipt)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
