#!/usr/bin/env python3
"""Remove an optional wall display from the boys-toilet surface plan."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
from pathlib import Path


EXPECTED_PLAN_SHA256 = "ddb974e6d9b3df31a82267a27d51968dfe4a38d2104d3ed169066cde8bc93ee9"


def canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest_file(path: Path) -> str:
    return digest_bytes(path.read_bytes())


def attestation(document: dict[str, object]) -> str:
    return digest_bytes(canonical({k: v for k, v in document.items() if k != "attestation"}))


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

    room_dir = args.room_dir.resolve(strict=True)
    states = room_dir / "scene_states"
    plan = states / "manipuland_furniture_plan.json"
    lock_path = states / ".manipuland_checkpoint.lock"
    if plan.is_symlink() or not plan.is_file() or plan.stat().st_nlink != 1:
        raise SystemExit(f"plan is not an unlinked regular file: {plan}")

    with lock_path.open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        before_sha256 = digest_file(plan)
        if before_sha256 != EXPECTED_PLAN_SHA256:
            raise SystemExit(f"unexpected plan digest: {before_sha256}")
        if any(states.glob("manipuland_checkpoint_*")):
            raise SystemExit("unexpected completed manipuland checkpoint")

        document = json.loads(plan.read_text(encoding="utf-8"))
        if document.get("attestation") != attestation(document):
            raise SystemExit("input plan attestation mismatch")
        selections = document.get("selections")
        if not isinstance(selections, list) or len(selections) != 3:
            raise SystemExit("unexpected input selection inventory")
        if document.get("selections_sha256") != digest_bytes(canonical(selections)):
            raise SystemExit("input selection inventory digest mismatch")

        removed = selections[0]
        if (
            removed.get("furniture_id") != "bulletin_board_0"
            or "required: none" not in str(removed.get("suggested_items", "")).casefold()
            or "no specific requirements" not in str(
                removed.get("prompt_constraints", "")
            ).casefold()
        ):
            raise SystemExit("bulletin-board selection is not explicitly optional")

        repaired = selections[1:]
        output = dict(document)
        output["selections"] = repaired
        output["selections_sha256"] = digest_bytes(canonical(repaired))
        output["attestation"] = attestation(output)

        args.backup_dir.mkdir(parents=True, exist_ok=True)
        backup = args.backup_dir / f"{plan.name}.{before_sha256}"
        shutil.copy2(plan, backup, follow_symlinks=False)
        atomic_json(plan, output, plan.stat().st_mode & 0o777)
        after_sha256 = digest_file(plan)

        receipt = {
            "schema_version": 1,
            "status": "pass",
            "operation": "remove_optional_zero_surface_target",
            "room_id": "boys_toilet",
            "plan": str(plan),
            "before_sha256": before_sha256,
            "after_sha256": after_sha256,
            "removed_target": "bulletin_board_0",
            "required_items_removed": [],
            "remaining_targets": [item["furniture_id"] for item in repaired],
            "reason": "vertical bulletin board has no physical support surface",
        }
        receipt["attestation"] = attestation(receipt)
        atomic_json(args.receipt, receipt, 0o600)
        print(f"BOYS_PLAN_REPAIR_PASS {before_sha256} {after_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
