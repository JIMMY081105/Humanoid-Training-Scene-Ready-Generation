#!/usr/bin/env python3
"""Repair Classroom 5's unphysical final cubby scope without touching checkpoints."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
from pathlib import Path


EXPECTED_PLAN_SHA256 = "3864964de8d3db0cf0c9074367505c641b921cc1f18ca781150bef16af837482"
EXPECTED_REPAIRED_PLAN_SHA256 = "9f5c53878c88dbddd861e57c65490b915821108f1e2ac9d3b966847fc0ae61da"
ROOM_ID = "classroom_05"
TARGET_INDEX = 9
TARGET_FURNITURE = "cubby_bookshelf_0"
REQUIRED_CHECKPOINTS = tuple(
    f"manipuland_checkpoint_{index:03d}_{name}"
    for index, name in enumerate(
        (
            "teacher_desk_0",
            "drawer_pedestal_0",
            "student_desk_0",
            "student_desk_1",
            "student_desk_2",
            "student_desk_3",
            "student_desk_4",
            "student_desk_5",
            "activity_table_0",
        )
    )
)

SUGGESTED_ITEMS = (
    "REQUIRED: one compact school backpack laid flat and two small labeled storage "
    "bins. Do not add loose books, folders, or tall vertical objects here; those "
    "prompt categories are already covered by the passed desk and activity-table scopes."
)
PROMPT_CONSTRAINTS = (
    "This furniture exposes one authored top support surface only. Keep the backpack "
    "fully on that top surface, laid flat and centered away from all edges; place one "
    "low labeled bin to each side with visible separation. Do not infer cubby interiors, "
    "hang anything over an edge, or put objects on the floor. This scope supplies the "
    "global backpack and storage-bin categories while preserving clear circulation."
)


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
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


def verify_checkpoint_inventory(states: Path, selections: list[dict[str, object]]) -> None:
    checkpoint_dirs = sorted(path.name for path in states.glob("manipuland_checkpoint_*"))
    if checkpoint_dirs != list(REQUIRED_CHECKPOINTS):
        raise SystemExit(f"unexpected checkpoint inventory: {checkpoint_dirs}")
    for index, directory_name in enumerate(REQUIRED_CHECKPOINTS):
        receipt_path = states / directory_name / "completion_receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            receipt.get("status") != "pass"
            or receipt.get("furniture_index") != index
            or receipt.get("selection") != selections[index]
            or receipt.get("attestation") != attest(receipt)
        ):
            raise SystemExit(f"checkpoint contract mismatch: {directory_name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--room-dir", required=True, type=Path)
    parser.add_argument("--backup-dir", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()

    room = args.room_dir.resolve(strict=True)
    states = room / "scene_states"
    plan_path = states / "manipuland_furniture_plan.json"
    lock_path = states / ".manipuland_checkpoint.lock"
    args.receipt.parent.mkdir(parents=True, exist_ok=True)

    with lock_path.open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        before_bytes = plan_path.read_bytes()
        before_sha256 = sha(before_bytes)
        if before_sha256 not in {EXPECTED_PLAN_SHA256, EXPECTED_REPAIRED_PLAN_SHA256}:
            raise SystemExit(f"unexpected plan digest: {before_sha256}")
        plan = json.loads(before_bytes)
        selections = plan.get("selections")
        if (
            plan.get("room_id") != ROOM_ID
            or plan.get("status") != "pass"
            or plan.get("attestation") != attest(plan)
            or not isinstance(selections, list)
            or len(selections) != TARGET_INDEX + 1
            or plan.get("selections_sha256") != sha(canonical(selections))
            or selections[TARGET_INDEX].get("furniture_id") != TARGET_FURNITURE
        ):
            raise SystemExit("plan contract mismatch")
        verify_checkpoint_inventory(states, selections)

        args.backup_dir.mkdir(parents=True, exist_ok=True)
        backup = args.backup_dir / f"{plan_path.name}.{EXPECTED_PLAN_SHA256}"
        receipt_recovered = before_sha256 == EXPECTED_REPAIRED_PLAN_SHA256
        if receipt_recovered:
            target = selections[TARGET_INDEX]
            if (
                target.get("suggested_items") != SUGGESTED_ITEMS
                or target.get("prompt_constraints") != PROMPT_CONSTRAINTS
                or not backup.exists()
                or sha(backup.read_bytes()) != EXPECTED_PLAN_SHA256
            ):
                raise SystemExit("incomplete-repair recovery contract mismatch")
            after_sha256 = before_sha256
        else:
            if not backup.exists():
                shutil.copy2(plan_path, backup, follow_symlinks=False)
            elif sha(backup.read_bytes()) != before_sha256:
                raise SystemExit("existing backup digest mismatch")
            repaired_selections = [dict(selection) for selection in selections]
            target = repaired_selections[TARGET_INDEX]
            target["suggested_items"] = SUGGESTED_ITEMS
            target["prompt_constraints"] = PROMPT_CONSTRAINTS
            repaired = dict(plan)
            repaired["selections"] = repaired_selections
            repaired["selections_sha256"] = sha(canonical(repaired_selections))
            repaired["attestation"] = attest(repaired)
            atomic_json(plan_path, repaired, plan_path.stat().st_mode & 0o777)
            after_sha256 = sha(plan_path.read_bytes())
            if after_sha256 != EXPECTED_REPAIRED_PLAN_SHA256:
                raise SystemExit(f"unexpected repaired plan digest: {after_sha256}")

        receipt = {
            "schema_version": 1,
            "status": "pass",
            "operation": "repair_unphysical_cubby_scope_to_authored_top_surface_only",
            "room_id": ROOM_ID,
            "plan_before_sha256": EXPECTED_PLAN_SHA256,
            "plan_after_sha256": after_sha256,
            "preserved_checkpoint_inventory": list(REQUIRED_CHECKPOINTS),
            "preserved_checkpoint_count": len(REQUIRED_CHECKPOINTS),
            "target_furniture_id": TARGET_FURNITURE,
            "target_index": TARGET_INDEX,
            "quality_configuration_modified": False,
            "selection_content_modified": True,
            "receipt_publication_recovered": receipt_recovered,
            "required_inventory_retained_elsewhere": [
                "textbooks", "notebooks", "folders", "worksheets", "paper tray",
                "pencil/marker holders", "erasers", "rulers", "glue", "scissors",
                "water bottles", "board marker", "whiteboard eraser",
            ],
            "target_inventory": ["backpack", "labeled storage bins"],
        }
        receipt["attestation"] = attest(receipt)
        atomic_json(args.receipt, receipt, 0o600)
        print(f"CLASSROOM05_CUBBY_SCOPE_REPAIR_PASS {before_sha256} {after_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
