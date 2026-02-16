#!/usr/bin/env python3
"""Rebind valid migrated checkpoints to current path/config identity."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
from pathlib import Path

from scenesmith.agent_utils.room import RoomScene


EXPECTED_RUNTIME_SHA256 = "32bd6b5869a774f155eda5093f9879df2f1c62e18f2497caee6d604dc551fefe"


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def attest(document: dict[str, object]) -> str:
    return sha(canonical({key: value for key, value in document.items() if key != "attestation"}))


def file_record(path: Path) -> dict[str, object]:
    return {"name": path.name, "size_bytes": path.stat().st_size, "sha256": sha(path.read_bytes())}


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


def load_scene(room: Path, state_path: Path, room_id: str) -> RoomScene:
    scene = RoomScene(room_geometry=None, scene_dir=room, room_id=room_id)
    scene.restore_from_state_dict(json.loads(state_path.read_text(encoding="utf-8")))
    return scene


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-dir", required=True, type=Path)
    parser.add_argument("--backup-dir", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    root = args.scene_dir.resolve(strict=True)
    records = []
    for room in sorted(root.glob("room_*")):
        states = room / "scene_states"
        checkpoints = sorted(states.glob("manipuland_checkpoint_*"))
        if not checkpoints:
            continue
        room_id = room.name.removeprefix("room_")
        with (states / ".manipuland_checkpoint.lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            plan = json.loads((states / "manipuland_furniture_plan.json").read_text(encoding="utf-8"))
            if plan.get("status") != "pass" or plan.get("attestation") != attest(plan):
                raise SystemExit(f"plan contract mismatch for {room_id}")
            selections = plan.get("selections")
            if not isinstance(selections, list):
                raise SystemExit(f"plan selections missing for {room_id}")
            live = load_scene(room, states / "scene_after_ceiling_objects/scene_state.json", room_id)
            for checkpoint in checkpoints:
                receipt_path = checkpoint / "completion_receipt.json"
                before_bytes = receipt_path.read_bytes()
                before = sha(before_bytes)
                document = json.loads(before_bytes)
                index = document.get("furniture_index")
                if not isinstance(index, int) or index >= len(selections):
                    raise SystemExit(f"checkpoint index mismatch: {checkpoint}")
                selection = selections[index]
                state_path = checkpoint / "scene_state.json"
                dmd_path = checkpoint / "scene.dmd.yaml"
                output_scene = load_scene(room, state_path, room_id)
                input_hash = live.content_hash()
                output_hash = output_scene.content_hash()
                if (
                    document.get("schema_version") != 3
                    or document.get("status") != "pass"
                    or document.get("furniture_id") != selection.get("furniture_id")
                    or document.get("selection") != selection
                    or document.get("selection_sha256") != sha(canonical(selection))
                    or document.get("checkpoint_runtime_sha256") != EXPECTED_RUNTIME_SHA256
                    or document.get("output_scene_content_hash") != output_hash
                    or document.get("legacy_source") is not None
                    or document.get("artifacts", {}).get("scene_state") != file_record(state_path)
                    or document.get("artifacts", {}).get("drake_directive") != file_record(dmd_path)
                    or document.get("attestation") != attest(document)
                ):
                    raise SystemExit(f"checkpoint evidence mismatch: {checkpoint}")
                output = dict(document)
                output["config_sha256"] = plan["config_sha256"]
                output["input_scene_content_hash"] = input_hash
                output["attestation"] = attest(output)
                args.backup_dir.mkdir(parents=True, exist_ok=True)
                backup = args.backup_dir / f"{room_id}.{checkpoint.name}.{before}.json"
                if not backup.exists():
                    shutil.copy2(receipt_path, backup, follow_symlinks=False)
                elif sha(backup.read_bytes()) != before:
                    raise SystemExit(f"backup mismatch: {checkpoint}")
                atomic_json(receipt_path, output, receipt_path.stat().st_mode & 0o777)
                records.append({
                    "room_id": room_id, "checkpoint": checkpoint.name,
                    "receipt_before_sha256": before,
                    "receipt_after_sha256": sha(receipt_path.read_bytes()),
                    "input_scene_content_hash": input_hash,
                    "output_scene_content_hash": output_hash,
                    "config_sha256": plan["config_sha256"],
                })
                live = output_scene
    receipt = {
        "schema_version": 1, "status": "pass",
        "operation": "rebind_migrated_manipuland_checkpoint_receipts",
        "records": records, "checkpoint_scene_content_modified": False,
        "checkpoint_artifacts_modified": False,
    }
    receipt["attestation"] = attest(receipt)
    atomic_json(args.receipt, receipt, 0o600)
    print("MANIPULAND_CHECKPOINT_REBIND_PASS", len(records))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
