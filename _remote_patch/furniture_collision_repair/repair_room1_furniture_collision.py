#!/usr/bin/env python3
"""Atomically repair the verified Room 1 furniture-origin collisions.

This is deliberately a narrow recovery tool, not a gate bypass.  It accepts only
the canonical Classroom 01 state, preserves every superseded checkpoint under a
timestamped archive, applies the four independently collision-tested transforms,
and refuses publication unless the complete, unfiltered physics check is clean.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
from scenesmith.agent_utils.physics_tools import check_physics_violations
from scenesmith.agent_utils.physics_validation import compute_scene_collisions
from scenesmith.agent_utils.room import RoomScene, UniqueID


STAGES = (
    "scene_after_furniture",
    "scene_after_wall_objects",
    "scene_after_ceiling_objects",
)

# These absolute targets are the smallest candidates that passed the complete
# Drake collision query on the canonical Room 1 ceiling checkpoint.  Absolute
# values make the recovery idempotent after an interrupted publication. Artiverse
# cabinet mesh origins are below their visual bases, so the vertical values seat
# their collision meshes on the floor rather than making the rendered cabinets
# float.
TARGET_TRANSLATIONS = {
    "filing_cabinet_0": (-4.1791529999999995, 2.045819618804612, 0.60),
    "storage_cabinet_0": (3.1891150635718146, -3.384332099425, 0.90),
    "cubby_shelf_unit_0": (-3.6520664499999995, -1.9009348399533974, 0.0),
    "potted_plant_0": (-3.72171382341105, -3.2432431551374132, 0.0),
}


class RepairError(RuntimeError):
    """Raised when the narrow repair contract is not satisfied."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def read_state(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RepairError(f"State is not a regular file: {path}")
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RepairError(f"Cannot parse state: {path}: {exc}") from exc
    if not isinstance(state, dict) or not isinstance(state.get("objects"), dict):
        raise RepairError(f"Malformed state: {path}")
    return state


def object_transform(state: dict[str, Any], object_id: str) -> dict[str, Any]:
    try:
        transform = state["objects"][object_id]["transform"]
    except (KeyError, TypeError) as exc:
        raise RepairError(f"Missing transform for {object_id}") from exc
    if not isinstance(transform, dict):
        raise RepairError(f"Malformed transform for {object_id}")
    translation = transform.get("translation")
    rotation = transform.get("rotation_wxyz")
    if (
        not isinstance(translation, list)
        or len(translation) != 3
        or not all(isinstance(value, (int, float)) for value in translation)
        or not isinstance(rotation, list)
        or len(rotation) != 4
    ):
        raise RepairError(f"Malformed rigid transform for {object_id}")
    return transform


def repaired_state(state: dict[str, Any]) -> dict[str, Any]:
    repaired = json.loads(json.dumps(state))
    # A manipuland stage was previously interrupted after mutating its in-memory
    # scene, and that stale downstream inventory leaked into the reusable stage
    # snapshots.  Those objects are not valid without their matching checkpoint
    # receipts, which are archived below.  Furniture, wall, and ceiling stages
    # must therefore resume with no manipulands at all.
    repaired["objects"] = {
        object_id: object_data
        for object_id, object_data in repaired["objects"].items()
        if object_data.get("object_type") != "manipuland"
    }
    for object_id, target in TARGET_TRANSLATIONS.items():
        transform = object_transform(repaired, object_id)
        transform["translation"] = [float(value) for value in target]
    return repaired


def build_scene(room_dir: Path, state: dict[str, Any]) -> RoomScene:
    scene = RoomScene(room_geometry=None, scene_dir=room_dir, room_id="classroom_01")
    scene.restore_from_state_dict(state)
    return scene


def validate_state(room_dir: Path, state: dict[str, Any], config: Any) -> dict[str, Any]:
    scene = build_scene(room_dir, state)
    expected_ids = {
        "filing_cabinet_0",
        "storage_cabinet_0",
        "cubby_shelf_unit_0",
        "potted_plant_0",
    }
    if not expected_ids.issubset({str(object_id) for object_id in scene.objects}):
        raise RepairError("Repair state has an incomplete furniture inventory")

    collisions = compute_scene_collisions(
        scene,
        penetration_threshold=0.001,
        floor_penetration_tolerance=0.05,
    )
    if collisions:
        detail = [
            (pair.object_a_id, pair.object_b_id, pair.penetration_depth)
            for pair in collisions
        ]
        raise RepairError(f"Repair leaves collisions: {detail}")

    physics = check_physics_violations(scene=scene, cfg=config)
    if physics != "No physics violations detected. All objects are properly placed.":
        raise RepairError(f"Repair fails complete physics validation: {physics}")
    return {
        "collision_count": 0,
        "full_physics": physics,
        "object_count": len(scene.objects),
    }


def write_new_stage(stage_dir: Path, state: dict[str, Any]) -> None:
    stage_dir.mkdir(mode=0o755)
    output = stage_dir / "scene_state.json"
    with output.open("xb") as stream:
        stream.write(canonical_json(state))
        stream.flush()
        os.fsync(stream.fileno())


def run(args: argparse.Namespace) -> None:
    room_dir = args.room_dir.resolve(strict=True)
    if room_dir.name != "room_classroom_01" or room_dir.is_symlink():
        raise RepairError("This recovery is authorized only for Room 1 classroom_01")
    config_path = args.config.resolve(strict=True)
    if config_path.is_symlink() or not config_path.is_file():
        raise RepairError("Config must be a regular file")
    config = OmegaConf.load(config_path)
    if "furniture_agent" not in config:
        raise RepairError("Resolved configuration has no furniture-agent physics policy")
    physics_config = config.furniture_agent
    states_root = room_dir / "scene_states"
    if states_root.is_symlink() or not states_root.is_dir():
        raise RepairError("Room state directory is missing or linked")
    if (states_root / "final_scene").exists():
        raise RepairError("Refusing to mutate a completed Room 1")

    paths = {stage: states_root / stage / "scene_state.json" for stage in STAGES}
    original_states = {stage: read_state(path) for stage, path in paths.items()}
    reference_prompt = original_states[STAGES[0]].get("text_description")
    if not isinstance(reference_prompt, str) or not reference_prompt:
        raise RepairError("Furniture checkpoint has no prompt")
    for stage, state in original_states.items():
        if state.get("text_description") != reference_prompt:
            raise RepairError(f"Prompt differs at {stage}")
    for object_id in TARGET_TRANSLATIONS:
            if object_transform(state, object_id) != object_transform(
                original_states[STAGES[0]], object_id
            ):
                raise RepairError(f"Furniture transform differs before repair at {stage}")

    outputs = {stage: repaired_state(state) for stage, state in original_states.items()}
    validation = validate_state(
        room_dir, outputs["scene_after_ceiling_objects"], physics_config
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = states_root / f"prior_furniture_collision_repair_{stamp}"
    archive.mkdir(mode=0o755)
    old_hashes = {stage: sha256(path) for stage, path in paths.items()}
    new_hashes = {
        stage: hashlib.sha256(canonical_json(output)).hexdigest()
        for stage, output in outputs.items()
    }
    receipt = {
        "schema_version": 1,
        "status": "validated_pending_atomic_publication",
        "room_id": "classroom_01",
        "created_at_utc": stamp,
        "cause": {
            "detected_collisions": [
                "filing_cabinet_0 floor penetration 0.57183m",
                "storage_cabinet_0 floor penetration 0.49870m",
                "storage_cabinet_0 south-wall penetration 0.09181m",
                "cubby_shelf_unit_0/potted_plant_0 penetration 0.02060m",
                "cubby_shelf_unit_0 blocked west-window clearance",
            ],
            "policy": "full unfiltered collision and clearance validation required",
            "stale_manipulands": "removed before a new receipt-bound manipuland run",
        },
        "target_translations": {
            key: list(value) for key, value in TARGET_TRANSLATIONS.items()
        },
        "validation": validation,
        "prior_state_sha256": old_hashes,
        "published_state_sha256": new_hashes,
        "archived_entries": [],
    }

    selected = [*STAGES]
    selected.extend(
        item.name
        for item in sorted(states_root.iterdir())
        if item.name.startswith(("manipuland_checkpoint_", "manipuland_furniture_", "manipuland_registry_"))
    )
    for name in selected:
        source = states_root / name
        if not source.exists():
            continue
        destination = archive / name
        os.replace(source, destination)
        receipt["archived_entries"].append(name)

    for stage in STAGES:
        write_new_stage(states_root / stage, outputs[stage])
    receipt["status"] = "published"
    receipt_path = archive / "furniture_collision_repair_receipt.json"
    with receipt_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())

    published = read_state(
        states_root / "scene_after_ceiling_objects" / "scene_state.json"
    )
    validate_state(room_dir, published, physics_config)
    print(json.dumps({"status": "published", "archive": str(archive), **validation}))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--room-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    try:
        run(parser.parse_args())
    except RepairError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
