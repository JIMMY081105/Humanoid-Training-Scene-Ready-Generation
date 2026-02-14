#!/usr/bin/env python3
"""Drop audited zero-surface bulletin boards from Library's manipuland plan."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
from pathlib import Path


EXPECTED_PLAN_SHA256 = "641191d7ac3892c367bd8631013fec6db4043b2505411ab964f06ad99ccedf38"
EXPECTED_RUNTIME_SHA256 = "8ad862129a21f236986573cb1d851223f8fb28e8246ddaff9a3ae27761bec4b7"
DROP_IDS = ("bulletin_board_0", "bulletin_board_1")


def canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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
            or len(selections) != 20
            or document.get("selections_sha256") != sha(canonical(selections))
        ):
            raise SystemExit("Library input plan contract mismatch")
        if tuple(item.get("furniture_id") for item in selections[-2:]) != DROP_IDS:
            raise SystemExit("zero-surface targets are not the audited trailing selections")
        reading_table = next(
            (item for item in selections if item.get("furniture_id") == "reading_table_2"),
            None,
        )
        if not reading_table or "flyers/handouts" not in reading_table.get("suggested_items", ""):
            raise SystemExit("required display-notice inventory was not rerouted")
        for item in selections[-2:]:
            if "REQUIRED: none" not in item.get("suggested_items", ""):
                raise SystemExit("refusing to drop a required manipuland target")

        repaired = selections[:-2]
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
            "operation": "drop_optional_zero_surface_manipuland_targets",
            "room_id": "library",
            "plan_before_sha256": before,
            "plan_after_sha256": after,
            "removed_optional_targets": list(DROP_IDS),
            "notice_inventory_rerouted_to": "reading_table_2",
            "required_room_objects_removed": False,
            "reason": (
                "The visible bulletin-board furniture remains in the scene; only its "
                "impossible horizontal manipuland scope is removed. Required notices "
                "remain assigned to reading_table_2."
            ),
        }
        receipt["attestation"] = attest(receipt)
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(args.receipt, receipt, 0o600)
        print(f"LIBRARY_ZERO_SURFACE_PREFLIGHT_PASS {before} {after}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
