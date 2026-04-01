#!/usr/bin/env python3
"""Preserve C6 checkpoints and reroute unstable cubby supplies to flat supports."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import stat
from pathlib import Path


EXPECTED_PLAN_SHA256 = "b83f54e11259c0d3782f3fb6e54bce680e6cc3a2a4165cd3db0f68c5a1e71f9e"
EXPECTED_RUNTIME_SHA256 = "2b39c7bd572c1e0e607eb13cd8ed68f3b5a1e76db004f9f323d6c52da4ef9f1a"
CHECKPOINTS = (
    ("manipuland_checkpoint_000_desk_0", "bf78537d1689f7514ee626d355cb47be1afefe44105c811c7c7ec6f85fd01cf2"),
    ("manipuland_checkpoint_001_classroom_storage_cabinet_0", "4a35cd4982d831ef7d63f4c907626a93f14e0268e292cf93c810e86fc876198e"),
)


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


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
    if args.receipt.exists():
        raise SystemExit(f"refusing to overwrite receipt: {args.receipt}")

    with (states / ".manipuland_checkpoint.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        require_regular(plan_path)
        before_bytes = plan_path.read_bytes()
        before_sha256 = sha(before_bytes)
        if before_sha256 != EXPECTED_PLAN_SHA256:
            raise SystemExit(f"unexpected Classroom 6 plan digest: {before_sha256}")
        checkpoint_names = sorted(path.name for path in states.glob("manipuland_checkpoint_*"))
        if checkpoint_names != [name for name, _digest in CHECKPOINTS]:
            raise SystemExit(f"Classroom 6 checkpoint inventory changed: {checkpoint_names}")

        plan = json.loads(before_bytes)
        selections = plan.get("selections")
        if (
            plan.get("schema_version") != 1
            or plan.get("status") != "pass"
            or plan.get("room_id") != "classroom_06"
            or plan.get("checkpoint_runtime_sha256") != EXPECTED_RUNTIME_SHA256
            or plan.get("attestation") != attest(plan)
            or not isinstance(selections, list)
            or plan.get("selections_sha256") != sha(canonical(selections))
        ):
            raise SystemExit("Classroom 6 plan contract mismatch")
        for index, (name, expected_digest) in enumerate(CHECKPOINTS):
            receipt_path = states / name / "completion_receipt.json"
            require_regular(receipt_path)
            receipt_bytes = receipt_path.read_bytes()
            if sha(receipt_bytes) != expected_digest:
                raise SystemExit(f"Classroom 6 checkpoint receipt changed at {index}")
            receipt = json.loads(receipt_bytes)
            if (
                receipt.get("schema_version") != 3
                or receipt.get("status") != "pass"
                or receipt.get("furniture_index") != index
                or receipt.get("selection") != selections[index]
                or receipt.get("selection_sha256") != sha(canonical(selections[index]))
                or receipt.get("checkpoint_runtime_sha256") != EXPECTED_RUNTIME_SHA256
                or receipt.get("attestation") != attest(receipt)
            ):
                raise SystemExit(f"Classroom 6 checkpoint contract mismatch at {index}")

        repaired = [dict(selection) for selection in selections]
        by_id = {selection.get("furniture_id"): selection for selection in repaired}
        if not {"cubby_bookshelf_0", "student_desk_7"}.issubset(by_id):
            raise SystemExit("Classroom 6 reroute targets are missing")
        cubby = by_id["cubby_bookshelf_0"]
        cubby["suggested_items"] = (
            "REQUIRED: lidded low-profile labeled storage totes with broad flat bases; "
            "textbooks in short flat stacks; one folder and one separate worksheet laid "
            "flat; two backpacks fully supported inside lower cubbies. Optional: one "
            "closed rectangular pencil box"
        )
        cubby["prompt_constraints"] = (
            "Preserve labeled storage, textbooks, separate folder/worksheet, and two "
            "backpacks. Do not use open bins, filled-container composites, caddies, or "
            "loose glue/scissors in this cubby. Every base must be centered fully inside "
            "a compartment with no leaning, hanging, or overhang."
        )
        cubby["style_notes"] = (
            "Use broad-base lidded totes and flat book/paper stacks in separate cubbies; "
            "keep backpacks in the lower row and preserve empty space and aisle access."
        )
        craft = by_id["student_desk_7"]
        craft["suggested_items"] = (
            "REQUIRED: one closed notebook, one capped glue stick laid horizontally, and "
            "one pair of closed school scissors laid flat. Optional: one pencil"
        )
        craft["prompt_constraints"] = (
            "This flat desk is the required glue/scissors evidence target rerouted from "
            "the cubby. Use separate visible objects with full broad contact, clear gaps, "
            "and generous distance from every desk edge."
        )
        craft["style_notes"] = (
            "Minimal safe-looking craft set: notebook flat, scissors closed and flat, "
            "capped glue horizontal, with the writing area and chair approach clear."
        )
        if repaired[: len(CHECKPOINTS)] != selections[: len(CHECKPOINTS)]:
            raise SystemExit("completed Classroom 6 selection changed")

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
            raise SystemExit("Classroom 6 plan backup mismatch")
        atomic_json(plan_path, output, plan_path.stat().st_mode & 0o777)
        after_sha256 = sha(plan_path.read_bytes())

    receipt: dict[str, object] = {
        "schema_version": 1,
        "status": "pass",
        "operation": "replace_unstable_cubby_containers_and_reroute_craft_supplies",
        "room_id": "classroom_06",
        "plan_before_sha256": before_sha256,
        "plan_after_sha256": after_sha256,
        "preserved_checkpoints": [name for name, _digest in CHECKPOINTS],
        "preserved_checkpoint_receipt_sha256": [digest for _name, digest in CHECKPOINTS],
        "changed_targets": ["cubby_bookshelf_0", "student_desk_7"],
        "required_inventory_preserved": [
            "storage bins", "textbooks", "folders", "worksheets", "backpacks",
            "glue sticks", "scissors", "notebooks",
        ],
        "quality_policy": "strict stability, collision, inventory, and access thresholds unchanged",
    }
    receipt["attestation"] = attest(receipt)
    atomic_json(args.receipt, receipt, 0o600)
    print("CLASSROOM06_CUBBY_HARDENING_PASS", before_sha256, after_sha256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
