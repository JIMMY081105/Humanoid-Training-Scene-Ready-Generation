#!/usr/bin/env python3
"""Finish Boys Toilet's audited no-required-manipuland repair."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
from pathlib import Path


EXPECTED_PLAN_SHA256 = "226179f104993f1b567f1389d66a88dcc2688dc320e75100f1bd1ed3a889d7ca"
EXPECTED_RUNTIME_SHA256 = "8ad862129a21f236986573cb1d851223f8fb28e8246ddaff9a3ae27761bec4b7"
EXPECTED_TARGETS = ("floating_shelf_0", "vanity_0")


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def attest(document: dict[str, object]) -> str:
    return sha(canonical({k: v for k, v in document.items() if k != "attestation"}))


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
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--room-dir", required=True, type=Path)
    parser.add_argument("--backup-dir", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()

    states = args.room_dir.resolve(strict=True) / "scene_states"
    plan = states / "manipuland_furniture_plan.json"
    with (states / ".manipuland_checkpoint.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if list(states.glob("manipuland_checkpoint_*")):
            raise SystemExit("refusing to alter Boys Toilet after a checkpoint exists")
        if plan.is_symlink() or not plan.is_file() or plan.stat().st_nlink != 1:
            raise SystemExit("Boys Toilet plan is not an unlinked regular file")
        before = sha(plan.read_bytes())
        if before != EXPECTED_PLAN_SHA256:
            raise SystemExit(f"unexpected Boys Toilet plan digest: {before}")
        document = json.loads(plan.read_text(encoding="utf-8"))
        selections = document.get("selections")
        if (
            document.get("room_id") != "boys_toilet"
            or document.get("status") != "pass"
            or document.get("checkpoint_runtime_sha256") != EXPECTED_RUNTIME_SHA256
            or document.get("attestation") != attest(document)
            or not isinstance(selections, list)
            or tuple(item.get("furniture_id") for item in selections) != EXPECTED_TARGETS
            or document.get("selections_sha256") != sha(canonical(selections))
        ):
            raise SystemExit("Boys Toilet input plan contract mismatch")
        for item in selections:
            if (
                "REQUIRED: none" not in item.get("suggested_items", "")
                or "No specific manipulands required" not in item.get("prompt_constraints", "")
            ):
                raise SystemExit("refusing to drop a required Boys Toilet target")

        output = dict(document)
        output["selections"] = []
        output["selections_sha256"] = sha(canonical([]))
        output["attestation"] = attest(output)
        args.backup_dir.mkdir(parents=True, exist_ok=True)
        backup = args.backup_dir / f"{plan.name}.{before}"
        if not backup.exists():
            shutil.copy2(plan, backup, follow_symlinks=False)
        elif sha(backup.read_bytes()) != before:
            raise SystemExit("existing Boys Toilet backup digest mismatch")
        atomic_json(plan, output, plan.stat().st_mode & 0o777)
        after = sha(plan.read_bytes())

        receipt = {
            "schema_version": 1,
            "status": "pass",
            "operation": "drop_all_explicitly_optional_manipuland_targets",
            "room_id": "boys_toilet",
            "plan_before_sha256": before,
            "plan_after_sha256": after,
            "removed_optional_targets": list(EXPECTED_TARGETS),
            "required_room_objects_removed": False,
            "required_fixture_inventory_preserved": True,
            "reason": (
                "The room prompt requires restroom fixtures, not countertop objects; "
                "all required stalls, urinals, sinks, mirrors, dispensers, dryer and bin "
                "remain in furniture/wall checkpoints."
            ),
        }
        receipt["attestation"] = attest(receipt)
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(args.receipt, receipt, 0o600)
        print(f"BOYS_OPTIONAL_PREFLIGHT_PASS {before} {after}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
