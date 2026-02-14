#!/usr/bin/env python3
"""Remove optional-only Library manipuland scopes while preserving visible furniture."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
from pathlib import Path


EXPECTED_PLAN_SHA256 = "a5dd5ef297938a1083d4cf826430daa00e09ec782eb2d28ec4fac91b49d4b0bb"
EXPECTED_RUNTIME_SHA256 = "8ad862129a21f236986573cb1d851223f8fb28e8246ddaff9a3ae27761bec4b7"
DROP_IDS = {
    "reading_table_0",
    "reading_table_1",
    "coffee_table_0",
    "storage_cubbies_unit_0",
}
REQUIRED_IDS = {
    "circulation_counter_0",
    "book_display_rack_0",
    "book_display_rack_1",
    "bookcase_cabinet_0",
    *(f"bookcase_{index}" for index in range(9)),
    "reading_table_2",
}


def canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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

    room = args.room_dir.resolve(strict=True)
    states = room / "scene_states"
    plan = states / "manipuland_furniture_plan.json"
    with (states / ".manipuland_checkpoint.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if list(states.glob("manipuland_checkpoint_*")):
            raise SystemExit("refusing to alter Library after a furniture checkpoint exists")
        if plan.is_symlink() or not plan.is_file() or plan.stat().st_nlink != 1:
            raise SystemExit("Library plan is not an unlinked regular file")

        before = sha(plan.read_bytes())
        if before != EXPECTED_PLAN_SHA256:
            raise SystemExit(f"unexpected Library plan digest: {before}")
        document = json.loads(plan.read_text(encoding="utf-8"))
        selections = document.get("selections")
        if (
            document.get("room_id") != "library"
            or document.get("status") != "pass"
            or document.get("checkpoint_runtime_sha256") != EXPECTED_RUNTIME_SHA256
            or document.get("attestation") != attest(document)
            or not isinstance(selections, list)
            or len(selections) != 18
            or document.get("selections_sha256") != sha(canonical(selections))
        ):
            raise SystemExit("Library input plan contract mismatch")

        by_id = {item.get("furniture_id"): item for item in selections}
        if set(by_id) != REQUIRED_IDS | DROP_IDS:
            raise SystemExit("Library target inventory differs from the audited plan")
        for furniture_id in DROP_IDS:
            suggested = by_id[furniture_id].get("suggested_items", "")
            if "REQUIRED:" in suggested or not suggested.startswith("Optional:"):
                raise SystemExit(f"refusing to remove non-optional target {furniture_id}")
        for furniture_id in REQUIRED_IDS:
            if "REQUIRED:" not in by_id[furniture_id].get("suggested_items", ""):
                raise SystemExit(f"required inventory marker missing from {furniture_id}")

        repaired = [item for item in selections if item.get("furniture_id") not in DROP_IDS]
        if len(repaired) != 14 or {item.get("furniture_id") for item in repaired} != REQUIRED_IDS:
            raise SystemExit("Library consolidation result mismatch")
        output = dict(document)
        output["selections"] = repaired
        output["selections_sha256"] = sha(canonical(repaired))
        output["attestation"] = attest(output)

        args.backup_dir.mkdir(parents=True, exist_ok=True)
        backup = args.backup_dir / f"{plan.name}.{before}"
        if not backup.exists():
            shutil.copy2(plan, backup, follow_symlinks=False)
        elif sha(backup.read_bytes()) != before:
            raise SystemExit("existing Library backup digest mismatch")
        atomic_json(plan, output, plan.stat().st_mode & 0o777)
        after = sha(plan.read_bytes())

        receipt = {
            "schema_version": 1,
            "status": "pass",
            "operation": "drop_optional_only_library_manipuland_targets",
            "room_id": "library",
            "plan_before_sha256": before,
            "plan_after_sha256": after,
            "removed_optional_targets": sorted(DROP_IDS),
            "retained_required_targets": sorted(REQUIRED_IDS),
            "required_room_objects_removed": False,
            "visible_furniture_removed": False,
            "reason": (
                "All removed scopes are optional-only. Their visible tables and cubbies "
                "remain in the room; books, magazines, notices, and checkout inventory "
                "remain assigned to fourteen required targets."
            ),
        }
        receipt["attestation"] = attest(receipt)
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(args.receipt, receipt, 0o600)
        print(f"LIBRARY_OPTIONAL_CONSOLIDATION_PASS {before} {after}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
