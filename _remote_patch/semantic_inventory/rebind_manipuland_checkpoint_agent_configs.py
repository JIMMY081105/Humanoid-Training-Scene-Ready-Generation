#!/usr/bin/env python3
"""Rebind migrated checkpoints to the exact agent configs validated at restore."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
from pathlib import Path

from scenesmith.agent_utils.room import RoomScene


def canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def attest(document: dict[str, object]) -> str:
    payload = {key: value for key, value in document.items() if key != "attestation"}
    return sha(canonical(payload))


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
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


def file_record_matches(path: Path, record: object) -> bool:
    return (
        isinstance(record, dict)
        and set(record) == {"name", "size_bytes", "sha256"}
        and record.get("name") == path.name
        and record.get("size_bytes") == path.stat().st_size
        and record.get("sha256") == sha(path.read_bytes())
    )


def load_scene_hash(state_path: Path, room_dir: Path, room_id: str) -> str:
    scene = RoomScene(room_geometry=None, scene_dir=room_dir, room_id=room_id)
    scene.restore_from_state_dict(json.loads(state_path.read_text(encoding="utf-8")))
    return scene.content_hash()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-dir", required=True, type=Path)
    parser.add_argument("--plan-config-receipt", required=True, type=Path)
    parser.add_argument("--backup-dir", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()

    scene_dir = args.scene_dir.resolve(strict=True)
    config_receipt = json.loads(args.plan_config_receipt.read_text(encoding="utf-8"))
    if (
        config_receipt.get("status") != "pass"
        or config_receipt.get("operation")
        != "rebind_plans_to_exact_manipuland_agent_configs_and_runtime"
        or config_receipt.get("attestation") != attest(config_receipt)
    ):
        raise SystemExit("agent-config receipt contract mismatch")
    expected_configs = {
        record["room_id"]: record["new_agent_config_sha256"]
        for record in config_receipt["records"]
    }
    expected_runtime = config_receipt.get("checkpoint_runtime_sha256")
    if not isinstance(expected_runtime, str) or len(expected_runtime) != 64:
        raise SystemExit("agent-config receipt runtime hash is missing")
    args.backup_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []

    for room_id, expected_config in expected_configs.items():
        room_dir = scene_dir / f"room_{room_id}"
        states = room_dir / "scene_states"
        plan_path = states / "manipuland_furniture_plan.json"
        with (states / ".manipuland_checkpoint.lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            selections = plan.get("selections")
            if (
                plan.get("status") != "pass"
                or plan.get("room_id") != room_id
                or plan.get("config_sha256") != expected_config
                or plan.get("checkpoint_runtime_sha256") != expected_runtime
                or plan.get("attestation") != attest(plan)
                or not isinstance(selections, list)
                or plan.get("selections_sha256") != sha(canonical(selections))
            ):
                raise SystemExit(f"plan/config contract mismatch for {room_id}")

            expected_input = plan["input_scene_content_hash"]
            checkpoints = sorted(states.glob("manipuland_checkpoint_*"))
            for checkpoint in checkpoints:
                receipt_path = checkpoint / "completion_receipt.json"
                state_path = checkpoint / "scene_state.json"
                directive_path = checkpoint / "scene.dmd.yaml"
                before_bytes = receipt_path.read_bytes()
                before_sha = sha(before_bytes)
                document = json.loads(before_bytes)
                index = document.get("furniture_index")
                if (
                    not isinstance(index, int)
                    or index < 0
                    or index >= len(selections)
                    or document.get("status") != "pass"
                    or document.get("selection") != selections[index]
                    or document.get("selection_sha256")
                    != sha(canonical(selections[index]))
                    or document.get("input_scene_content_hash") != expected_input
                    or document.get("attestation") != attest(document)
                ):
                    raise SystemExit(f"checkpoint context mismatch: {checkpoint}")
                artifacts = document.get("artifacts")
                if (
                    not isinstance(artifacts, dict)
                    or not file_record_matches(state_path, artifacts.get("scene_state"))
                    or not file_record_matches(
                        directive_path, artifacts.get("drake_directive")
                    )
                    or load_scene_hash(state_path, room_dir, room_id)
                    != document.get("output_scene_content_hash")
                ):
                    raise SystemExit(f"checkpoint artifact mismatch: {checkpoint}")

                backup = (
                    args.backup_dir
                    / f"{room_id}.{checkpoint.name}.{before_sha}.completion_receipt.json"
                )
                if not backup.exists():
                    shutil.copy2(receipt_path, backup, follow_symlinks=False)
                elif sha(backup.read_bytes()) != before_sha:
                    raise SystemExit(f"backup mismatch: {backup}")

                output = dict(document)
                output["config_sha256"] = expected_config
                output["checkpoint_runtime_sha256"] = expected_runtime
                output["attestation"] = attest(output)
                atomic_json(
                    receipt_path, output, receipt_path.stat().st_mode & 0o777
                )
                records.append(
                    {
                        "room_id": room_id,
                        "checkpoint": checkpoint.name,
                        "old_config_sha256": document.get("config_sha256"),
                        "new_agent_config_sha256": expected_config,
                        "old_checkpoint_runtime_sha256": document.get(
                            "checkpoint_runtime_sha256"
                        ),
                        "new_checkpoint_runtime_sha256": expected_runtime,
                        "receipt_before_sha256": before_sha,
                        "receipt_after_sha256": sha(receipt_path.read_bytes()),
                        "input_scene_content_hash": expected_input,
                        "output_scene_content_hash": document[
                            "output_scene_content_hash"
                        ],
                    }
                )
                expected_input = document["output_scene_content_hash"]

    receipt: dict[str, object] = {
        "schema_version": 1,
        "status": "pass",
        "operation": "rebind_checkpoints_to_exact_manipuland_agent_configs",
        "records": records,
        "checkpoint_scene_content_modified": False,
        "selection_content_modified": False,
    }
    receipt["attestation"] = attest(receipt)
    atomic_json(args.receipt, receipt, 0o600)
    print("MANIPULAND_CHECKPOINT_AGENT_CONFIG_REBIND_PASS", len(records))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
