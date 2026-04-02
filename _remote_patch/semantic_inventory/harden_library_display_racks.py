#!/usr/bin/env python3
"""Preserve Library checkpoint 000 and remove unstable thin media from angled racks."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import stat
from pathlib import Path


EXPECTED_PLAN_SHA256 = "f398d2710da8a6a10d2a1e60175ca0f9c00b1e2a6c0b277b781284ed4cab45f8"
EXPECTED_RUNTIME_SHA256 = "2b39c7bd572c1e0e607eb13cd8ed68f3b5a1e76db004f9f323d6c52da4ef9f1a"
EXPECTED_CHECKPOINT = (
    "manipuland_checkpoint_000_circulation_counter_0",
    "5754d4c1da9c90a380139be0beb217b19f9bb9e915996d8c61ac7c61a9325437",
)
TARGETS = ("book_display_rack_0", "book_display_rack_1")


def canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def attest(document: dict[str, object]) -> str:
    return sha(canonical({key: value for key, value in document.items() if key != "attestation"}))


def require_regular(path: Path) -> None:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise SystemExit(f"unsafe regular-file contract: {path}")


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
    receipt_path = states / EXPECTED_CHECKPOINT[0] / "completion_receipt.json"
    if args.receipt.exists():
        raise SystemExit(f"refusing to overwrite receipt: {args.receipt}")

    with (states / ".manipuland_checkpoint.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        require_regular(plan_path)
        require_regular(receipt_path)
        before_bytes = plan_path.read_bytes()
        before_sha256 = sha(before_bytes)
        if before_sha256 != EXPECTED_PLAN_SHA256:
            raise SystemExit(f"unexpected Library plan digest: {before_sha256}")
        if sha(receipt_path.read_bytes()) != EXPECTED_CHECKPOINT[1]:
            raise SystemExit("Library checkpoint receipt changed")
        checkpoint_names = sorted(path.name for path in states.glob("manipuland_checkpoint_*"))
        if checkpoint_names != [EXPECTED_CHECKPOINT[0]]:
            raise SystemExit(f"Library checkpoint inventory changed: {checkpoint_names}")

        plan = json.loads(before_bytes)
        selections = plan.get("selections")
        if (
            plan.get("schema_version") != 1
            or plan.get("status") != "pass"
            or plan.get("room_id") != "library"
            or plan.get("checkpoint_runtime_sha256") != EXPECTED_RUNTIME_SHA256
            or plan.get("attestation") != attest(plan)
            or not isinstance(selections, list)
            or plan.get("selections_sha256") != sha(canonical(selections))
        ):
            raise SystemExit("Library plan contract mismatch")
        checkpoint = json.loads(receipt_path.read_bytes())
        if (
            checkpoint.get("schema_version") != 3
            or checkpoint.get("status") != "pass"
            or checkpoint.get("furniture_index") != 0
            or checkpoint.get("selection") != selections[0]
            or checkpoint.get("selection_sha256") != sha(canonical(selections[0]))
            or checkpoint.get("checkpoint_runtime_sha256") != EXPECTED_RUNTIME_SHA256
            or checkpoint.get("attestation") != attest(checkpoint)
        ):
            raise SystemExit("Library checkpoint contract mismatch")

        repaired = [dict(selection) for selection in selections]
        changed: list[str] = []
        for selection in repaired:
            furniture_id = selection.get("furniture_id")
            if furniture_id not in TARGETS:
                continue
            if "featured books" not in selection.get("suggested_items", ""):
                raise SystemExit(f"unexpected display-rack selection: {furniture_id}")
            selection["suggested_items"] = (
                "REQUIRED: rigid hardcover and paperback featured books, face-out. "
                "Optional: none; no magazines, pamphlets, flyers, or other thin media"
            )
            selection["prompt_constraints"] = (
                "Preserve the required visible book-display role using rigid books only. "
                "Thin magazines and pamphlets are forbidden on this angled support because "
                "strict simulation proved they slide or flip; the required magazines remain "
                "assigned to reading_table_2 on a flat support."
            )
            selection["style_notes"] = (
                "Fill 70-90% of the slots with varied rigid books, aligned face-out with "
                "their complete lower edge supported and nothing overhanging."
            )
            changed.append(str(furniture_id))
        if changed != list(TARGETS):
            raise SystemExit(f"display-rack target mismatch: {changed}")
        if repaired[0] != selections[0]:
            raise SystemExit("completed checkpoint selection was modified")
        reading = next(
            (selection for selection in repaired if selection.get("furniture_id") == "reading_table_2"),
            None,
        )
        if reading is None or "two magazines" not in reading.get("suggested_items", ""):
            raise SystemExit("required magazines are not preserved on reading_table_2")

        output = dict(plan)
        output["selections"] = repaired
        output["selections_sha256"] = sha(canonical(repaired))
        output["attestation"] = attest(output)
        args.backup_dir.mkdir(parents=True, exist_ok=True)
        backup = args.backup_dir / f"{plan_path.name}.{before_sha256}.json"
        if not backup.exists():
            shutil.copy2(plan_path, backup, follow_symlinks=False)
        require_regular(backup)
        if sha(backup.read_bytes()) != before_sha256:
            raise SystemExit("Library plan backup mismatch")
        atomic_json(plan_path, output, plan_path.stat().st_mode & 0o777)
        after_sha256 = sha(plan_path.read_bytes())

    receipt: dict[str, object] = {
        "schema_version": 1,
        "status": "pass",
        "operation": "replace_unstable_thin_media_on_angled_library_displays",
        "room_id": "library",
        "plan_before_sha256": before_sha256,
        "plan_after_sha256": after_sha256,
        "preserved_checkpoint": EXPECTED_CHECKPOINT[0],
        "preserved_checkpoint_receipt_sha256": EXPECTED_CHECKPOINT[1],
        "changed_targets": changed,
        "required_magazine_target": "reading_table_2",
        "quality_policy": (
            "book displays remain 70-90 percent full; magazine inventory preserved on a "
            "flat table; strict stability and collision thresholds unchanged"
        ),
    }
    receipt["attestation"] = attest(receipt)
    atomic_json(args.receipt, receipt, 0o600)
    print("LIBRARY_DISPLAY_RACK_HARDENING_PASS", before_sha256, after_sha256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
