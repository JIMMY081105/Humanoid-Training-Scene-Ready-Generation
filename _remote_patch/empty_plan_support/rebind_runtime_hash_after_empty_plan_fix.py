#!/usr/bin/env python3
"""Rebind valid plans/checkpoints to the tested empty-plan runtime fix."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
from pathlib import Path


OLD_RUNTIME_SHA256 = "8ad862129a21f236986573cb1d851223f8fb28e8246ddaff9a3ae27761bec4b7"
NEW_RUNTIME_SHA256 = "32bd6b5869a774f155eda5093f9879df2f1c62e18f2497caee6d604dc551fefe"


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


def update(path: Path, backup_dir: Path) -> dict[str, object]:
    before_bytes = path.read_bytes()
    before = sha(before_bytes)
    document = json.loads(before_bytes)
    if document.get("status") != "pass" or document.get("attestation") != attest(document):
        raise SystemExit(f"invalid attested document: {path}")
    if document.get("checkpoint_runtime_sha256") not in {OLD_RUNTIME_SHA256, NEW_RUNTIME_SHA256}:
        raise SystemExit(f"unexpected runtime binding: {path}")
    output = dict(document)
    output["checkpoint_runtime_sha256"] = NEW_RUNTIME_SHA256
    output["attestation"] = attest(output)
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup = backup_dir / f"{path.parent.name}.{path.name}.{before}"
    if not backup.exists():
        shutil.copy2(path, backup, follow_symlinks=False)
    elif sha(backup.read_bytes()) != before:
        raise SystemExit(f"backup mismatch: {path}")
    atomic_json(path, output, path.stat().st_mode & 0o777)
    return {"path": str(path), "before_sha256": before, "after_sha256": sha(path.read_bytes())}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-dir", required=True, type=Path)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--backup-dir", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    if sha(args.runtime.resolve(strict=True).read_bytes()) != NEW_RUNTIME_SHA256:
        raise SystemExit("tested runtime digest mismatch")
    root = args.scene_dir.resolve(strict=True)
    records = []
    for room in sorted(root.glob("room_*")):
        states = room / "scene_states"
        plan = states / "manipuland_furniture_plan.json"
        if not plan.is_file():
            continue
        with (states / ".manipuland_checkpoint.lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            records.append(update(plan, args.backup_dir))
            for checkpoint in sorted(states.glob("manipuland_checkpoint_*")):
                records.append(update(checkpoint / "completion_receipt.json", args.backup_dir))
    receipt = {
        "schema_version": 1, "status": "pass",
        "operation": "rebind_checkpoint_runtime_after_empty_plan_support_fix",
        "old_runtime_sha256": OLD_RUNTIME_SHA256,
        "new_runtime_sha256": NEW_RUNTIME_SHA256,
        "records": records,
        "scene_content_modified": False,
        "selection_content_modified": False,
    }
    receipt["attestation"] = attest(receipt)
    atomic_json(args.receipt, receipt, 0o600)
    print("EMPTY_PLAN_RUNTIME_REBIND_PASS", len(records))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
