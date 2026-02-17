#!/usr/bin/env python3
"""Rebind unchanged manipuland plans to relocated ParaCloud ceiling states."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
from pathlib import Path

from scenesmith.agent_utils.room import RoomScene


ROOM_IDS = (
    "classroom_04", "storage_room", "classroom_05", "library",
    "boys_toilet", "classroom_06", "classroom_02", "classroom_03",
    "main_corridor", "girls_toilet",
)
EXPECTED_RUNTIME_SHA256 = "8ad862129a21f236986573cb1d851223f8fb28e8246ddaff9a3ae27761bec4b7"


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def attest(document: dict[str, object]) -> str:
    return sha(canonical({key: value for key, value in document.items() if key != "attestation"}))


def atomic_json(path: Path, document: dict[str, object], mode: int) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(document, stream, indent=2, sort_keys=True, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-dir", required=True, type=Path)
    parser.add_argument("--backup-dir", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    scene_dir = args.scene_dir.resolve(strict=True)
    records = []

    for room_id in ROOM_IDS:
        room = scene_dir / f"room_{room_id}"
        states = room / "scene_states"
        state_path = states / "scene_after_ceiling_objects" / "scene_state.json"
        plan_path = states / "manipuland_furniture_plan.json"
        with (states / ".manipuland_checkpoint.lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            before_bytes = plan_path.read_bytes()
            before = sha(before_bytes)
            document = json.loads(before_bytes)
            selections = document.get("selections")
            if (
                document.get("status") != "pass"
                or document.get("room_id") != room_id
                or document.get("checkpoint_runtime_sha256") != EXPECTED_RUNTIME_SHA256
                or document.get("attestation") != attest(document)
                or not isinstance(selections, list)
                or document.get("selections_sha256") != sha(canonical(selections))
            ):
                raise SystemExit(f"plan contract mismatch for {room_id}")
            state = json.loads(state_path.read_text(encoding="utf-8"))
            loaded = RoomScene(room_geometry=None, scene_dir=room, room_id=room_id)
            loaded.restore_from_state_dict(state)
            current_hash = loaded.content_hash()
            old_hash = document.get("input_scene_content_hash")
            if old_hash == current_hash:
                records.append({"room_id": room_id, "status": "already_bound", "plan_sha256": before})
                continue
            output = dict(document)
            output["input_scene_content_hash"] = current_hash
            output["attestation"] = attest(output)
            args.backup_dir.mkdir(parents=True, exist_ok=True)
            backup = args.backup_dir / f"{room_id}.manipuland_furniture_plan.{before}.json"
            if not backup.exists():
                shutil.copy2(plan_path, backup, follow_symlinks=False)
            elif sha(backup.read_bytes()) != before:
                raise SystemExit(f"backup mismatch for {room_id}")
            atomic_json(plan_path, output, plan_path.stat().st_mode & 0o777)
            records.append({
                "room_id": room_id,
                "status": "rebound",
                "old_input_scene_content_hash": old_hash,
                "new_input_scene_content_hash": current_hash,
                "plan_before_sha256": before,
                "plan_after_sha256": sha(plan_path.read_bytes()),
                "scene_state_sha256": sha(state_path.read_bytes()),
            })

    receipt = {
        "schema_version": 1,
        "status": "pass",
        "operation": "rebind_migrated_manipuland_plans_to_unchanged_ceiling_states",
        "records": records,
        "scene_content_modified": False,
        "selection_content_modified": False,
    }
    receipt["attestation"] = attest(receipt)
    atomic_json(args.receipt, receipt, 0o600)
    print("MIGRATED_PLAN_REBIND_PASS", len(records))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
