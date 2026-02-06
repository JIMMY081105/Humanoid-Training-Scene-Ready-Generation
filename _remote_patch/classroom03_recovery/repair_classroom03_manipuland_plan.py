#!/usr/bin/env python3
"""Preserve Classroom 3's checkpoint and consolidate the remaining details."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
from pathlib import Path


EXPECTED_PLAN_SHA256 = "50c8aab780c9be20d54786498e06ec596bcbe226421bcf5ed2c172c555b16bbf"
EXPECTED_CHECKPOINT_SHA256 = "e5fca7929e4fb8417dca7b3700d78622ab3d967bd549d097b7ad5a73e72939a5"
EXPECTED_RUNTIME_SHA256 = "8ad862129a21f236986573cb1d851223f8fb28e8246ddaff9a3ae27761bec4b7"
EXPECTED_SELECTION_SHA256 = [
    "9a92ea4269a6877d4b669586b6e65b85877db2dc2a80d1624691648ab98b274d",
    "7e51a2f9600cdbdd6b7e8a7284c870791a8fa4bb90e46c0c91903f8c11773e1d",
    "ffe496e0f201127c1021cde3f17c7ccd469826fbeae7d5fb6cb88af00e587916",
    "73df112933062f51b5a0ac65ba9156b8fa94dbdc496e4c99df3d5f08f5becefc",
    "cf6b2f2d5b6dfc366fd73b85cadb68b98b21972bac4ad6d66cc762b5dfd26d8d",
    "cbb0fb6788a51f6b39ace886d0d9e5c302044f232f8a33019566e810d178d40f",
    "f25303c33c19615abc0323eac3e617687ae3241aaa624afb5904610266a17322",
    "8c7b06dcea626d95918c9e6fac8d3aa6773b37b01f612bef518cfe2db5be4017",
    "b1bec2c0e23a28d35c5512c90f17cdbccc98080645ac6be384b8171a1f2ddbd6",
    "98362b1c0a170c1bef56c03222d55d27ec46a56c0a75692ab3fb69a644dc0700",
    "97c758c43365a3907d2c90d8c6987c9eacafc793d2d0764bf5eb324b8850d3ed",
    "0b7f4a9160067742f5e88311a3de20931896199a9bb3c15be80e32d581c6fc18",
    "d641fcdaf5edb333bf0d477b156b1ae1931a7ac6bb728522f6d5b96ae9f6029d",
    "dabd30f4509fa71b8d4ac250994b6658ad754c17a6370cee50daa5d42247d025",
    "56afda8343b361f77f74b68ec8b842cb4d6499103362f9056f5d765724114e14",
    "40aab48f06cce7648636a4c592944f283f7e509b9a3fddabe97f2e4db607767f",
    "56e862abed963b9bb6991e66d4c286307280e3ea5741c0de086ec446bf19cc77",
    "f05e7f90df7b965e321f5ba51b9d77dd199d453484226e6b3006b680b8478ec3",
    "540c8ab900fc4e482a3836d1f404bc72d45b34da0c3a1408d013ab906a258352",
]
KEEP_INDICES = (0, 3, 5, 8, 11)


def canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


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

    room = args.room_dir.resolve(strict=True)
    states = room / "scene_states"
    plan_path = states / "manipuland_furniture_plan.json"
    checkpoint_path = (
        states
        / "manipuland_checkpoint_000_teacher_desk_0"
        / "completion_receipt.json"
    )
    checkpoint_dirs = sorted(states.glob("manipuland_checkpoint_*"))
    if [path.name for path in checkpoint_dirs] != [
        "manipuland_checkpoint_000_teacher_desk_0"
    ]:
        raise SystemExit("unexpected checkpoint inventory")

    with (states / ".manipuland_checkpoint.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        before = sha(plan_path.read_bytes())
        checkpoint_sha256 = sha(checkpoint_path.read_bytes())
        if before != EXPECTED_PLAN_SHA256:
            raise SystemExit(f"unexpected plan digest: {before}")
        if checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256:
            raise SystemExit(f"unexpected checkpoint digest: {checkpoint_sha256}")

        document = json.loads(plan_path.read_text(encoding="utf-8"))
        selections = document.get("selections")
        if (
            document.get("room_id") != "classroom_03"
            or document.get("status") != "pass"
            or document.get("checkpoint_runtime_sha256") != EXPECTED_RUNTIME_SHA256
            or document.get("attestation") != attest(document)
            or not isinstance(selections, list)
            or len(selections) != len(EXPECTED_SELECTION_SHA256)
            or document.get("selections_sha256") != sha(canonical(selections))
        ):
            raise SystemExit("input plan contract mismatch")
        for index, expected in enumerate(EXPECTED_SELECTION_SHA256):
            if sha(canonical(selections[index])) != expected:
                raise SystemExit(f"unexpected selection at index {index}")

        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if (
            checkpoint.get("status") != "pass"
            or checkpoint.get("furniture_index") != 0
            or checkpoint.get("furniture_id") != "teacher_desk_0"
            or checkpoint.get("selection") != selections[0]
            or checkpoint.get("selection_sha256") != EXPECTED_SELECTION_SHA256[0]
            or checkpoint.get("checkpoint_runtime_sha256") != EXPECTED_RUNTIME_SHA256
            or checkpoint.get("attestation") != attest(checkpoint)
        ):
            raise SystemExit("checkpoint contract mismatch")

        repaired = [dict(selections[index]) for index in KEEP_INDICES]
        repaired[1]["suggested_items"] = (
            "REQUIRED: low-profile labeled storage bin(s), textbooks, "
            "folders/worksheets, backpack(s), glue stick, scissors, dry-erase "
            "markers, whiteboard eraser. Optional: rulers, pencil cases"
        )
        repaired[1]["prompt_constraints"] = (
            "Prompt requires textbooks, folders/worksheets, backpacks, storage bins, "
            "glue/scissors, marker supplies and a whiteboard eraser. Distribute these "
            "inside supported cubbies; do not use the unstable supply-cart containers "
            "or the rolling board's trayless visual fixture."
        )
        repaired[1]["style_notes"] = (
            "Use separate cubbies as organized zones; keep bins low-profile and leave "
            "several cubbies open so nothing protrudes into circulation."
        )

        output = dict(document)
        output["selections"] = repaired
        output["selections_sha256"] = sha(canonical(repaired))
        output["attestation"] = attest(output)

        args.backup_dir.mkdir(parents=True, exist_ok=True)
        backup = args.backup_dir / f"{plan_path.name}.{before}"
        if not backup.exists():
            shutil.copy2(plan_path, backup, follow_symlinks=False)
        elif sha(backup.read_bytes()) != before:
            raise SystemExit("existing backup digest mismatch")
        atomic_json(plan_path, output, plan_path.stat().st_mode & 0o777)
        after = sha(plan_path.read_bytes())

        receipt = {
            "schema_version": 1,
            "status": "pass",
            "operation": "preserve_checkpoint_and_consolidate_remaining_inventory",
            "room_id": "classroom_03",
            "plan_before_sha256": before,
            "plan_after_sha256": after,
            "preserved_checkpoint": "manipuland_checkpoint_000_teacher_desk_0",
            "preserved_checkpoint_sha256": checkpoint_sha256,
            "kept_targets": [item["furniture_id"] for item in repaired],
            "removed_failed_target": "supply_cart_0",
            "reroute_target": "cubby_bookshelf_0",
            "required_categories_preserved": [
                "textbooks",
                "notebooks",
                "folders/worksheets",
                "pencil/marker holders",
                "paper trays",
                "storage bins",
                "glue",
                "scissors",
                "markers",
                "whiteboard eraser",
                "erasers/rulers",
                "water bottles",
                "backpacks",
            ],
            "reason": (
                "The supply-cart container set failed strict forward stability; the "
                "supported cubby compartments preserve every required category with "
                "less duplicated desk clutter."
            ),
        }
        receipt["attestation"] = attest(receipt)
        atomic_json(args.receipt, receipt, 0o600)
        print(f"CLASSROOM03_PLAN_REPAIR_PASS {before} {after}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
