#!/usr/bin/env python3
"""Reroute Classroom 5 board supplies from a trayless rolling whiteboard."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
from pathlib import Path


EXPECTED_PLAN_SHA256 = "29016d8c95704eba9a2d764459e2213ceb3b958fe0995401f85c8cb898c3e165"
EXPECTED_ZERO_SURFACE_SELECTION_SHA256 = (
    "b6963c24ca42c5da81014fbf6ab32aeec44fbc5779e80f7321949c1718d068ba"
)
EXPECTED_REROUTE_SELECTION_SHA256 = (
    "843dc018ee1e31fce4a12ba894a29787c5e2af1dd4c2f47df52c9be1504efb81"
)
PRESERVED_CHECKPOINTS = (
    "manipuland_checkpoint_000_teacher_desk_0",
    "manipuland_checkpoint_001_drawer_pedestal_0",
)


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
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(temporary, flags, mode)
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
        document = json.loads(plan.read_text(encoding="utf-8"))
        if document.get("attestation") != attestation(document):
            raise SystemExit("input plan attestation mismatch")
        selections = document.get("selections")
        if not isinstance(selections, list) or len(selections) < 4:
            raise SystemExit("input plan selections are incomplete")
        if document.get("selections_sha256") != digest_bytes(canonical(selections)):
            raise SystemExit("input selection inventory digest mismatch")

        for index, name in enumerate(PRESERVED_CHECKPOINTS):
            receipt_path = states / name / "completion_receipt.json"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if receipt.get("status") != "pass" or receipt.get("furniture_index") != index:
                raise SystemExit(f"invalid completed checkpoint: {receipt_path}")
            if receipt.get("selection") != selections[index]:
                raise SystemExit(f"completed selection changed before reroute: {receipt_path}")
            if receipt.get("selection_sha256") != digest_bytes(canonical(selections[index])):
                raise SystemExit(f"completed selection digest mismatch: {receipt_path}")
            if receipt.get("attestation") != attestation(receipt):
                raise SystemExit(f"completed checkpoint attestation mismatch: {receipt_path}")

        zero_surface = selections[2]
        reroute = selections[3]
        if (
            zero_surface.get("furniture_id") != "rolling_whiteboard_0"
            or digest_bytes(canonical(zero_surface))
            != EXPECTED_ZERO_SURFACE_SELECTION_SHA256
            or reroute.get("furniture_id") != "student_desk_0"
            or digest_bytes(canonical(reroute)) != EXPECTED_REROUTE_SELECTION_SHA256
        ):
            raise SystemExit("zero-surface/reroute selections do not match the audited plan")

        repaired_reroute = dict(reroute)
        repaired_reroute["suggested_items"] = (
            "REQUIRED: notebook, pencil, dry-erase marker, whiteboard eraser. "
            "Optional: water bottle"
        )
        repaired_reroute["prompt_constraints"] = (
            'Prompt: "Add believable school-use details including textbooks, '
            'notebooks, pencils/markers, board markers, and a whiteboard eraser." '
            "The generated rolling board has no physical tray, so stage its marker "
            "and eraser together on the nearest front-row desk for accessible board use."
        )
        repaired_reroute["style_notes"] = (
            "Student-ready and tidy; keep the marker and eraser together at the rear "
            "edge nearest the board while preserving clear writing space."
        )
        repaired = [*selections[:2], repaired_reroute, *selections[4:]]
        if len({item["furniture_id"] for item in repaired}) != len(repaired):
            raise SystemExit("repaired furniture plan contains duplicate targets")

        output = dict(document)
        output["selections"] = repaired
        output["selections_sha256"] = digest_bytes(canonical(repaired))
        output["attestation"] = attestation(output)

        args.backup_dir.mkdir(parents=True, exist_ok=True)
        backup = args.backup_dir / f"{plan.name}.{before_sha256}"
        if not backup.exists():
            shutil.copy2(plan, backup, follow_symlinks=False)
        elif digest_file(backup) != before_sha256:
            raise SystemExit(f"existing backup digest mismatch: {backup}")
        atomic_json(plan, output, plan.stat().st_mode & 0o777)
        after_sha256 = digest_file(plan)

        receipt = {
            "schema_version": 1,
            "status": "pass",
            "operation": "reroute_required_manipulands_from_zero_surface_target",
            "room_id": "classroom_05",
            "plan": str(plan),
            "before_sha256": before_sha256,
            "after_sha256": after_sha256,
            "removed_target": "rolling_whiteboard_0",
            "reroute_target": "student_desk_0",
            "preserved_checkpoint_directories": list(PRESERVED_CHECKPOINTS),
            "required_items_preserved": ["dry-erase marker", "whiteboard eraser"],
            "reason": "generated rolling whiteboard has no physical support surface",
        }
        receipt["attestation"] = attestation(receipt)
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(args.receipt, receipt, 0o600)
        print(f"PLAN_REPAIR_PASS {before_sha256} {after_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
