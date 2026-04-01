#!/usr/bin/env python3
"""Checkpoint-safe C3 cubby recovery, used only if its active strict pass fails."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import stat
from pathlib import Path


EXPECTED_PLAN_SHA256 = "a762f00deff8a0f5ccce8e7508c89c9e0b78db8933098f5201a7e629004cbcbf"
EXPECTED_RUNTIME_SHA256 = "2b39c7bd572c1e0e607eb13cd8ed68f3b5a1e76db004f9f323d6c52da4ef9f1a"
CHECKPOINT = (
    "manipuland_checkpoint_000_teacher_desk_0",
    "1d7b00eba22c02d580c9b32aea6a01b731898316838cd4e6ef2b05a42c5ed51f",
)


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def attest(document: dict[str, object]) -> str:
    return digest(canonical({key: value for key, value in document.items() if key != "attestation"}))


def require_regular(path: Path) -> None:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise SystemExit(f"unsafe regular file: {path}")


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
    plan_path = states / "manipuland_furniture_plan.json"
    if args.receipt.exists():
        raise SystemExit(f"refusing to overwrite receipt: {args.receipt}")

    with (states / ".manipuland_checkpoint.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        require_regular(plan_path)
        before = plan_path.read_bytes()
        before_sha256 = digest(before)
        if before_sha256 != EXPECTED_PLAN_SHA256:
            raise SystemExit(f"unexpected Classroom 3 plan digest: {before_sha256}")
        actual_checkpoints = sorted(path.name for path in states.glob("manipuland_checkpoint_*"))
        if actual_checkpoints != [CHECKPOINT[0]]:
            raise SystemExit(f"Classroom 3 checkpoint inventory changed: {actual_checkpoints}")

        plan = json.loads(before)
        selections = plan.get("selections")
        if (
            plan.get("schema_version") != 1
            or plan.get("status") != "pass"
            or plan.get("room_id") != "classroom_03"
            or plan.get("checkpoint_runtime_sha256") != EXPECTED_RUNTIME_SHA256
            or plan.get("attestation") != attest(plan)
            or not isinstance(selections, list)
            or plan.get("selections_sha256") != digest(canonical(selections))
        ):
            raise SystemExit("Classroom 3 plan contract mismatch")

        checkpoint_path = states / CHECKPOINT[0] / "completion_receipt.json"
        require_regular(checkpoint_path)
        checkpoint_bytes = checkpoint_path.read_bytes()
        if digest(checkpoint_bytes) != CHECKPOINT[1]:
            raise SystemExit("Classroom 3 checkpoint receipt changed")
        checkpoint = json.loads(checkpoint_bytes)
        if (
            checkpoint.get("schema_version") != 3
            or checkpoint.get("status") != "pass"
            or checkpoint.get("furniture_index") != 0
            or checkpoint.get("selection") != selections[0]
            or checkpoint.get("selection_sha256") != digest(canonical(selections[0]))
            or checkpoint.get("checkpoint_runtime_sha256") != EXPECTED_RUNTIME_SHA256
            or checkpoint.get("attestation") != attest(checkpoint)
        ):
            raise SystemExit("Classroom 3 checkpoint contract mismatch")

        repaired = [dict(selection) for selection in selections]
        by_id = {selection.get("furniture_id"): selection for selection in repaired}
        required = {"cubby_bookshelf_0", "student_desk_0", "student_desk_6"}
        if len(by_id) != len(repaired) or not required.issubset(by_id):
            raise SystemExit("Classroom 3 recovery targets are missing")

        by_id["cubby_bookshelf_0"].update({
            "suggested_items": (
                "REQUIRED: lidded low-profile labeled storage totes with broad flat bases; "
                "textbooks in short flat stacks; one folder and one separate worksheet laid "
                "flat; backpacks fully supported inside lower cubbies. Optional: one closed "
                "rectangular pencil case"
            ),
            "prompt_constraints": (
                "Preserve labeled storage, textbooks, separate folder/worksheet, and backpacks. "
                "Do not use open bins, filled-container composites, loose cylindrical markers, "
                "glue, or scissors in this cubby. Center every complete base inside a compartment "
                "with no leaning, hanging, overhang, or circulation intrusion."
            ),
            "style_notes": (
                "Use broad-base lidded totes and flat book/paper stacks in separate cubbies; "
                "keep backpacks in the lower row and preserve empty compartments and aisle access."
            ),
        })
        by_id["student_desk_0"].update({
            "suggested_items": (
                "REQUIRED: one closed notebook, one pencil eraser, one ruler, one rectangular "
                "whiteboard eraser laid flat, and one broad shallow rectangular marker tray "
                "holding one separate colored classroom marker and one separate dry-erase marker"
            ),
            "prompt_constraints": (
                "Markers remain separate visible physical objects but must lie horizontally in "
                "the broad shallow tray with bottom and side support. Keep the ruler and both "
                "erasers fully supported, separated, and well away from every desk edge."
            ),
            "style_notes": (
                "Compact supported writing station with clear chair approach and most of the "
                "desktop unobstructed."
            ),
        })
        by_id["student_desk_6"].update({
            "suggested_items": (
                "REQUIRED: one closed notebook, one pencil, one capped glue stick laid "
                "horizontally, and one pair of closed school scissors laid flat"
            ),
            "prompt_constraints": (
                "This desk carries the glue and scissors rerouted from the cubby. Use separate "
                "visible objects with full broad contact, clear gaps, and generous edge distance."
            ),
            "style_notes": (
                "Minimal safe-looking craft set with the writing area and chair approach clear."
            ),
        })
        if repaired[0] != selections[0]:
            raise SystemExit("completed Classroom 3 selection changed")

        output = dict(plan)
        output["selections"] = repaired
        output["selections_sha256"] = digest(canonical(repaired))
        output["attestation"] = attest(output)
        args.backup_dir.mkdir(parents=True, exist_ok=True)
        backup = args.backup_dir / f"{plan_path.name}.{before_sha256}.json"
        if not backup.exists():
            shutil.copy2(plan_path, backup, follow_symlinks=False)
        require_regular(backup)
        if digest(backup.read_bytes()) != before_sha256:
            raise SystemExit("Classroom 3 plan backup mismatch")
        atomic_json(plan_path, output, plan_path.stat().st_mode & 0o777)
        after_sha256 = digest(plan_path.read_bytes())

    receipt: dict[str, object] = {
        "schema_version": 1,
        "status": "pass",
        "operation": "harden_failed_cubby_and_reroute_rolling_supplies",
        "room_id": "classroom_03",
        "plan_before_sha256": before_sha256,
        "plan_after_sha256": after_sha256,
        "preserved_checkpoints": [CHECKPOINT[0]],
        "preserved_checkpoint_receipt_sha256": [CHECKPOINT[1]],
        "changed_targets": sorted(required),
        "required_inventory_preserved": [
            "storage bins", "textbooks", "folder", "worksheet", "backpacks",
            "colored marker", "dry-erase marker", "whiteboard eraser", "glue", "scissors",
        ],
        "quality_policy": "strict stability, collision, inventory, visual, and access thresholds unchanged",
    }
    receipt["attestation"] = attest(receipt)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.receipt, receipt, 0o600)
    print("CLASSROOM03_CUBBY_HARDENING_PASS", before_sha256, after_sha256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
