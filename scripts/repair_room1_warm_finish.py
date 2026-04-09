#!/usr/bin/env python3
"""Add a physical warm-cream teaching-wall finish to Room 1, collision-proven."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import uuid
from pathlib import Path

from repair_room1_acceptance_details import _publish_staged_final, _sha256_file, _worker_config, _write_json, _write_text


FINISH_ID = "warm_cream_teaching_wall_finish_0"
OBSCURING_ORGANIZER_ID = "teacher_visible_supply_organizer_0"


def _sdf() -> str:
    # A real thin architectural wall lining, deliberately behind the existing
    # board, clock, display, window, and sconce rather than a render overlay.
    return """<?xml version='1.0'?>
<sdf version='1.7'><model name='warm_cream_teaching_wall_finish'><link name='base_link'>
<visual name='warm_cream_plaster'><geometry><box><size>8.82 0.018 3.12</size></box></geometry><material><diffuse>0.94 0.84 0.68 1</diffuse><specular>0.04 0.03 0.02 1</specular></material></visual>
<visual name='lower_wainscot'><pose>0 -0.014 0.52 0 0 0</pose><geometry><box><size>8.84 0.022 1.04</size></box></geometry><material><diffuse>0.76 0.54 0.30 1</diffuse></material></visual>
</link></model></sdf>
"""


def _new_or_deeper(baseline, candidate):
    old = {tuple(sorted((x.object_a_id, x.object_b_id))): x.penetration_depth for x in baseline}
    return [x for x in candidate if x.penetration_depth > old.get(tuple(sorted((x.object_a_id, x.object_b_id))), -1.0) + 1e-4]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--room-id", default="classroom_01")
    parser.add_argument("--port-offset", type=int, default=79)
    args = parser.parse_args()

    from run_single_room_worker import _configure_pytorch_cuda_allocator
    _configure_pytorch_cuda_allocator()
    import numpy as np
    from omegaconf import OmegaConf
    from pydrake.math import RigidTransform, RollPitchYaw
    from scenesmith.agent_utils.physics_validation import compute_scene_collisions
    from scenesmith.agent_utils.rendering import save_scene_as_blend
    from scenesmith.agent_utils.room import ObjectType, RoomScene, SceneObject, UniqueID

    repo_dir, run_dir = args.repo_dir.resolve(), args.run_dir.resolve()
    os.chdir(repo_dir)
    room_dir = run_dir / "scene_000" / f"room_{args.room_id}"
    final_dir = room_dir / "scene_states" / "final_scene"
    state_path = final_dir / "scene_state.json"
    if not state_path.is_file():
        raise FileNotFoundError(state_path)
    prior_sha = _sha256_file(state_path)
    cfg = OmegaConf.to_container(_worker_config(args), resolve=True)
    staged: Path | None = None
    published = False
    try:
        scene = RoomScene(room_geometry=None, scene_dir=room_dir, room_id=args.room_id)
        scene.restore_from_state_dict(json.loads(state_path.read_text(encoding="utf-8")))
        baseline = compute_scene_collisions(scene)
        # The prior acceptance organizer duplicated and visually obscured the
        # genuine checkpoint supplies already arranged on this cubby.  Remove
        # only that authored stand-in; retain the actual reusable assets.
        removed_organizer = scene.remove_object(UniqueID(OBSCURING_ORGANIZER_ID))
        asset_dir = room_dir / "generated_assets" / "architectural" / "warm_cream_teaching_wall_finish_v1"
        sdf_path = asset_dir / "warm_cream_teaching_wall_finish.sdf"
        expected = _sdf()
        if asset_dir.exists():
            if not sdf_path.is_file() or sdf_path.read_text(encoding="utf-8") != expected:
                raise RuntimeError(f"Refusing unexpected architectural finish: {asset_dir}")
        else:
            temporary = asset_dir.with_name(f".{asset_dir.name}.{uuid.uuid4().hex}.tmp")
            temporary.mkdir(parents=True, exist_ok=False)
            _write_text(temporary / sdf_path.name, expected)
            os.replace(temporary, asset_dir)
        finish = scene.get_object(UniqueID(FINISH_ID))
        pose = RigidTransform(rpy=RollPitchYaw([0.0, 0.0, math.pi]), p=[0.0, 3.69, 1.56])
        if finish is None:
            scene.add_object(SceneObject(
                object_id=UniqueID(FINISH_ID), object_type=ObjectType.WALL_MOUNTED,
                name="warm_cream_teaching_wall_finish",
                description="Physical warm cream plaster teaching wall with low wood wainscot, behind the classroom board and daylight window",
                transform=pose, sdf_path=sdf_path,
                metadata={"asset_source": "architectural_acceptance_repair", "semantic_role": "warm_cream_classroom_wall_finish"},
                bbox_min=np.array([-4.42, -0.03, 0.0]), bbox_max=np.array([4.42, 0.03, 3.12]),
            ))
        else:
            finish.transform = pose
            finish.metadata = {**finish.metadata, "semantic_role": "warm_cream_classroom_wall_finish"}
        candidate = compute_scene_collisions(scene)
        changed = _new_or_deeper(baseline, candidate)
        if changed:
            raise RuntimeError("New/deeper collision: " + "; ".join(x.to_description() for x in changed))
        staged = final_dir.parent / f".room1_warm_finish_{uuid.uuid4().hex}"
        staged.mkdir(parents=False, exist_ok=False)
        state = scene.to_state_dict(); state["timestamp"] = time.time()
        _write_json(staged / "scene_state.json", state)
        (staged / "scene.dmd.yaml").write_text(scene.to_drake_directive(), encoding="utf-8")
        rendering = cfg["furniture_agent"]["rendering"]
        save_scene_as_blend(scene=scene, output_path=staged / "scene.blend", blender_server_host=rendering.get("blender_server_host", "127.0.0.1"), blender_server_port_range=tuple(rendering["blender_server_port_range"]), server_startup_delay=rendering["server_startup_delay"], port_cleanup_delay=rendering["port_cleanup_delay"])
        backup = _publish_staged_final(staged_dir=staged, final_dir=final_dir)
        staged = None; published = True
        _write_json(room_dir / "quality_gates" / "room1_warm_finish_repair.json", {
            "status": "pass", "room_id": args.room_id, "prior_final_state_sha256": prior_sha,
            "final_state_sha256": _sha256_file(final_dir / "scene_state.json"), "backup_final_scene": str(backup),
            "physical_architectural_fixture": {"id": FINISH_ID, "sdf": str(sdf_path), "sha256": _sha256_file(sdf_path)},
            "removed_obscuring_organizer": bool(removed_organizer),
            "collision_proof": {"baseline_collision_count": len(baseline), "candidate_collision_count": len(candidate), "new_or_deeper_collision_count": len(changed)},
        })
    finally:
        if staged is not None and published:
            raise AssertionError("Published transaction left staging")


if __name__ == "__main__":
    main()
