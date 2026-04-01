#!/usr/bin/env python3
"""Bind every remaining classroom manipuland category as REQUIRED.

This is a checkpoint-safe, digest-bound plan refinement.  It does not add
duplicate furniture scopes or alter completed checkpoint selections; it only
removes ambiguity between required prompt objects and optional decoration.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
from pathlib import Path


EXPECTED_RUNTIME_SHA256 = "8ad862129a21f236986573cb1d851223f8fb28e8246ddaff9a3ae27761bec4b7"
EXPECTED_PLANS = {
    "classroom_02": "c9aaadc86f2105fc5e18d0ec27582ef2dac8b4c164c2c61476847f654c2eade8",
    "classroom_03": "811cf160903a4d73b8395695321108d2f1e863d90248dbca931dc5bc73fe4cf4",
    "classroom_04": "1b6ee8a31c18ad60e12004343ba7a5d23e8f589a5634d3ebb81817a808bc312f",
    "classroom_05": "410b0e970606bbf8634ca87e2ab45c9d2b4fdd4e49a575e7774039c90ba4cb13",
    "classroom_06": "886cbd354af7e8b01ae0a2abada53b3e4c42ba645d05db763627f1fde1dd154d",
}
EXPECTED_CHECKPOINTS = {
    "classroom_02": [],
    "classroom_03": [
        ("manipuland_checkpoint_000_teacher_desk_0", "e5fca7929e4fb8417dca7b3700d78622ab3d967bd549d097b7ad5a73e72939a5")
    ],
    "classroom_04": [],
    "classroom_05": [
        ("manipuland_checkpoint_000_teacher_desk_0", "990d0c21c8082b165d01328a53f2f1dcc2a65342c10e09b7823a79a6bc245030"),
        ("manipuland_checkpoint_001_drawer_pedestal_0", "c615e5333bd8d1cd343519b656c7275ca8bd5e04b439344d56143c4b489ce269"),
    ],
    "classroom_06": [
        ("manipuland_checkpoint_000_desk_0", "f913910620ba3b5f03289d6284d8b06aab21276597f709705558335d1927f1b0")
    ],
}


UPDATES = {
    "classroom_02": {
        "teacher_desk_0": (
            "REQUIRED: one paper tray, one folder, one separate worksheet, one pencil holder, one separate colored classroom marker, one dry-erase board marker, and one whiteboard eraser. Optional: teacher planner/notebook",
            "Use separate physical objects for the folder, worksheet, pencil holder, colored classroom marker, dry-erase board marker, and whiteboard eraser; a marker tray or holder must not substitute for the markers themselves.",
        ),
        "classroom_desk_0": (
            "REQUIRED: notebook, pencil eraser, ruler, water bottle. Optional: pencil",
            "Keep four compact student items clearly separated and fully supported on the desk.",
        ),
    },
    "classroom_03": {
        "cubby_bookshelf_0": (
            "REQUIRED: low-profile labeled storage bin(s), textbooks, one folder, one separate worksheet, backpack(s), glue stick, scissors, one separate colored classroom marker, one dry-erase board marker, and one whiteboard eraser. Optional: pencil cases",
            "Use separate physical objects for folder/worksheet and colored/board markers. Distribute everything inside supported cubbies without protruding into circulation.",
        ),
        "student_desk_0": (
            "REQUIRED: notebook, pencil eraser, ruler. Optional: pencil, water bottle",
            "Use separate compact notebook, eraser, and ruler objects with clear spacing.",
        ),
    },
    "classroom_04": {
        "desk_0": (
            "REQUIRED: one paper tray, one folder, one separate worksheet, one dry-erase board marker, and one whiteboard eraser. Optional: gradebook/notebook, pen cup, stapler",
            "Folder, worksheet, board marker, and whiteboard eraser must be separate visible physical objects; retain clear teacher writing space.",
        ),
        "collaboration_table_0": (
            "REQUIRED: shared pencil/marker holder, one separate colored classroom marker, glue stick, scissors, worksheet, and ruler. Optional: small stack of colored paper",
            "The colored marker must be a separate object rather than part of the holder; group the craft tools compactly.",
        ),
        "student_desk_0": (
            "REQUIRED: notebook, pencil eraser, ruler. Optional: pencil",
            "Keep notebook, eraser, and ruler separate and fully supported.",
        ),
    },
    "classroom_05": {
        "student_desk_0": (
            "REQUIRED: notebook, pencil, dry-erase board marker, whiteboard eraser. Optional: water bottle",
            "The board marker and whiteboard eraser must be separate visible physical objects.",
        ),
        "student_desk_1": (
            "REQUIRED: textbook, one folder, one separate worksheet. Optional: ruler",
            "Use separate folder and worksheet objects so each required category remains independently auditable.",
        ),
        "activity_table_0": (
            "REQUIRED: construction paper, worksheet, separate colored classroom markers, glue, scissors, small storage bin/caddy, and one paper tray. Optional: crayons/colored pencils",
            "Keep the paper tray and storage caddy distinct; markers must be separate visible objects rather than only a holder label.",
        ),
    },
    "classroom_06": {
        "classroom_storage_cabinet_0": (
            "REQUIRED: extra paper tray(s), one separate colored classroom marker, one dry-erase board marker, and one whiteboard eraser. Optional: stack of worksheets, tissue box",
            "Use separate physical objects for the colored marker, board marker, and whiteboard eraser; keep them low and stable on the cabinet support.",
        ),
        "cubby_bookshelf_0": (
            "REQUIRED: labeled storage bins, textbooks, one folder, one separate worksheet, two backpacks fully stored in lower cubbies, and glue/scissors caddy. Optional: pencil box set",
            "Use separate folder and worksheet objects and leave all backpacks fully inside lower cubbies.",
        ),
        "student_desk_0": (
            "REQUIRED: notebook, pencil eraser, water bottle. Optional: worksheet",
            "Use separate compact notebook, eraser, and water bottle objects with clear support.",
        ),
    },
}


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest_file(path: Path) -> str:
    return digest_bytes(path.read_bytes())


def attest(document: dict) -> str:
    return digest_bytes(canonical({k: v for k, v in document.items() if k != "attestation"}))


def atomic_json(path: Path, document: dict, mode: int) -> None:
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


def harden(room_dir: Path, backup_dir: Path) -> dict:
    room_id = room_dir.name.removeprefix("room_")
    states = room_dir / "scene_states"
    plan_path = states / "manipuland_furniture_plan.json"
    if plan_path.is_symlink() or not plan_path.is_file() or plan_path.stat().st_nlink != 1:
        raise SystemExit(f"unsafe plan path: {plan_path}")
    expected_checkpoints = EXPECTED_CHECKPOINTS[room_id]
    actual_checkpoint_names = sorted(path.name for path in states.glob("manipuland_checkpoint_*"))
    if actual_checkpoint_names != [name for name, _digest in expected_checkpoints]:
        raise SystemExit(f"{room_id}: checkpoint inventory changed: {actual_checkpoint_names}")

    with (states / ".manipuland_checkpoint.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        before = digest_file(plan_path)
        if before != EXPECTED_PLANS[room_id]:
            raise SystemExit(f"{room_id}: unexpected plan digest {before}")
        document = json.loads(plan_path.read_text(encoding="utf-8"))
        selections = document.get("selections")
        if (
            document.get("room_id") != room_id
            or document.get("status") != "pass"
            or document.get("checkpoint_runtime_sha256") != EXPECTED_RUNTIME_SHA256
            or document.get("attestation") != attest(document)
            or not isinstance(selections, list)
            or document.get("selections_sha256") != digest_bytes(canonical(selections))
        ):
            raise SystemExit(f"{room_id}: input plan contract mismatch")

        original_by_id = {item["furniture_id"]: dict(item) for item in selections}
        updated_by_id = UPDATES[room_id]
        if not set(updated_by_id).issubset(original_by_id):
            raise SystemExit(f"{room_id}: expected furniture selection missing")
        repaired = [dict(item) for item in selections]
        for item in repaired:
            furniture_id = item["furniture_id"]
            if furniture_id in updated_by_id:
                items, constraint = updated_by_id[furniture_id]
                item["suggested_items"] = items
                item["prompt_constraints"] = constraint

        for index, (checkpoint_name, expected_digest) in enumerate(expected_checkpoints):
            receipt_path = states / checkpoint_name / "completion_receipt.json"
            if digest_file(receipt_path) != expected_digest:
                raise SystemExit(f"{room_id}: checkpoint receipt digest changed")
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if (
                receipt.get("furniture_index") != index
                or receipt.get("selection") != selections[index]
                or repaired[index] != selections[index]
            ):
                raise SystemExit(f"{room_id}: completed checkpoint selection was modified")

        output = dict(document)
        output["selections"] = repaired
        output["selections_sha256"] = digest_bytes(canonical(repaired))
        output["attestation"] = attest(output)

        room_backup = backup_dir / room_id
        room_backup.mkdir(parents=True, exist_ok=True)
        backup = room_backup / f"{plan_path.name}.{before}.backup"
        if not backup.exists():
            shutil.copy2(plan_path, backup, follow_symlinks=False)
        if digest_file(backup) != before:
            raise SystemExit(f"{room_id}: backup digest mismatch")
        atomic_json(plan_path, output, plan_path.stat().st_mode & 0o777)
        after = digest_file(plan_path)
        return {
            "room_id": room_id,
            "status": "pass",
            "plan_before_sha256": before,
            "plan_after_sha256": after,
            "preserved_checkpoints": [name for name, _digest in expected_checkpoints],
            "updated_furniture_ids": sorted(updated_by_id),
            "required_categories": [
                "textbooks", "notebooks", "folders", "worksheets", "pencil holders",
                "general markers", "erasers", "rulers", "glue sticks", "scissors",
                "backpacks", "water bottles", "storage bins", "paper trays",
                "board markers", "whiteboard erasers",
            ],
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-dir", required=True, type=Path)
    parser.add_argument("--backup-dir", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument(
        "--room-ids",
        nargs="+",
        choices=sorted(EXPECTED_PLANS),
        default=sorted(EXPECTED_PLANS),
    )
    args = parser.parse_args()
    records = [
        harden(args.scene_dir / f"room_{room_id}", args.backup_dir)
        for room_id in args.room_ids
    ]
    receipt = {
        "schema_version": 1,
        "status": "pass",
        "operation": "bind_all_remaining_classroom_inventory_as_required",
        "rooms": records,
        "checkpoint_policy": "completed selections byte-identical",
        "quality_policy": "no inventory category removed or made optional",
    }
    receipt["attestation"] = attest(receipt)
    atomic_json(args.receipt, receipt, 0o600)
    for record in records:
        print(record["room_id"], record["plan_before_sha256"], record["plan_after_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
