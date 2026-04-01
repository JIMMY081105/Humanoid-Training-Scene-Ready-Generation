#!/usr/bin/env python3
"""Checkpoint-safe repair of three manipuland plans after strict stability failures."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import stat
from contextlib import ExitStack
from pathlib import Path


EXPECTED_RUNTIME_SHA256 = "2b39c7bd572c1e0e607eb13cd8ed68f3b5a1e76db004f9f323d6c52da4ef9f1a"
ROOMS = {
    "classroom_04": {
        "plan_sha256": "3749452aaac3b2c6221021f87de81e1c861c1619d98946aa1c365681fed03087",
        "checkpoints": [
            ("manipuland_checkpoint_000_desk_0", "95a9752591460fa09bd84325a1dc57a5fb24af1775c80bed55870a49749ab96d"),
            ("manipuland_checkpoint_001_collaboration_table_0", "e4de08773d2436bac224d4891e939750ecc78f305ea64133e5525fa1639b25db"),
            ("manipuland_checkpoint_002_student_desk_0", "7273ce84b130962f34ef1904637bf1158367435c6af8bf7b453a156c07e4e91c"),
            ("manipuland_checkpoint_003_student_desk_1", "66d24289416334d0fcd9367fcf794cffaf4d00b7f3cdd0867fd6c33125673d74"),
            ("manipuland_checkpoint_004_student_desk_3", "c5cbdcc362a2ca9953eec3ab7771a5d2a3f25a8b1b9e3e247e2543056c2e3149"),
        ],
        "updates": {
            "cubby_bookshelf_0": {
                "suggested_items": (
                    "REQUIRED: textbooks laid flat in one or two short stacks; lidded, "
                    "low-profile labeled storage bins with broad flat bases; backpacks "
                    "resting fully inside lower cubbies. Optional: one horizontal binder stack"
                ),
                "prompt_constraints": (
                    "Preserve the required textbooks, labeled storage bins, and backpacks. "
                    "Exclude open or unsupported storage-bin assets, gradebooks, and loose "
                    "spiral notebooks from this target. Every object must have its complete "
                    "base inside a shelf or lower cubby; nothing may hang, lean, or overhang."
                ),
                "style_notes": (
                    "Use separate cubbies for flat book stacks, broad-base lidded bins, and "
                    "backpacks. Keep the shelf front and nearby aisle clear."
                ),
            }
        },
    },
    "classroom_05": {
        "plan_sha256": "b905655e263822ffb77993ab1ad8b1e3f4c20919dc08f51cdcd3b11ba1561fea",
        "checkpoints": [
            ("manipuland_checkpoint_000_teacher_desk_0", "10840fd8185cfe914d77744cf94572c6af6db55a37642ddb47609326471dcd88"),
            ("manipuland_checkpoint_001_drawer_pedestal_0", "439643bff4bec3acc1e5242d6759326bfaa1a2ca8474214aefdb07a71d3da8b6"),
        ],
        "updates": {
            "student_desk_0": {
                "suggested_items": (
                    "REQUIRED: one closed notebook, one pencil, and one rectangular "
                    "whiteboard eraser placed flat. Optional: one flat folder"
                ),
                "prompt_constraints": (
                    "Keep the separate whiteboard eraser visible here. Do not place a loose "
                    "cylindrical dry-erase marker on this desk; the required separate board "
                    "marker is assigned to activity_table_0 in a supporting caddy."
                ),
                "style_notes": (
                    "Use only compact broad-contact objects, fully supported away from every "
                    "desk edge, with the writing area and chair approach clear."
                ),
            },
            "activity_table_0": {
                "suggested_items": (
                    "REQUIRED: construction paper, one worksheet, separate colored classroom "
                    "markers, one separate dry-erase board marker, glue, closed scissors, one "
                    "broad shallow storage caddy, and one paper tray. Optional: crayons"
                ),
                "prompt_constraints": (
                    "Keep the paper tray and caddy distinct. Markers remain separate visible "
                    "physical objects but must lie horizontally within the broad shallow "
                    "caddy with bottom and side support; never leave a loose cylinder on the "
                    "tabletop. Preserve clear access and all prompt inventory categories."
                ),
                "style_notes": (
                    "Compact supported art station: flat paper stacks, closed scissors, and "
                    "contained markers, with most of the table and its approach unobstructed."
                ),
            },
        },
    },
    "storage_room": {
        "plan_sha256": "06ede7ff3b587ef3e74937533c859c8bb430be093eb0fee45f7a0944febbe17b",
        "checkpoints": [
            ("manipuland_checkpoint_000_shelving_unit_0", "6783851d0a0a370af8f62cf2a532f43a28db90df992974989fa4b3e1535c3673"),
        ],
        "updates": {
            "shelving_unit_1": {
                "suggested_items": (
                    "REQUIRED: sealed reams of copy paper laid flat; binders and folders laid "
                    "horizontally in short low stacks. Optional: one sealed broad-base toner "
                    "or printer-supply box centered on a shelf"
                ),
                "prompt_constraints": (
                    "Preserve paper, binder/folder, and school-storage semantics. Do not "
                    "generate loose markers here; they are assigned to the bin organizer and "
                    "art-supply shelving. No binder, folder, or box may stand upright, lean, "
                    "overhang, or touch a shelf edge."
                ),
                "style_notes": (
                    "Use only low, flat, broad-contact stacks centered on separate shelves; "
                    "leave clear negative space and an unobstructed retrieval edge."
                ),
            }
        },
    },
}


def canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def attest(document: dict[str, object]) -> str:
    return sha_bytes(
        canonical({key: value for key, value in document.items() if key != "attestation"})
    )


def require_regular(path: Path) -> None:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise SystemExit(f"unsafe regular-file contract: {path}")


def atomic_bytes(path: Path, payload: bytes, mode: int) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
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


def encoded_json(document: dict[str, object]) -> bytes:
    return (
        json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def validate_room(scene: Path, room_id: str, contract: dict[str, object]) -> dict[str, object]:
    states = scene / f"room_{room_id}" / "scene_states"
    plan_path = states / "manipuland_furniture_plan.json"
    require_regular(plan_path)
    before_bytes = plan_path.read_bytes()
    before_sha256 = sha_bytes(before_bytes)
    if before_sha256 != contract["plan_sha256"]:
        raise SystemExit(f"{room_id}: unexpected plan digest {before_sha256}")
    plan = json.loads(before_bytes)
    selections = plan.get("selections")
    if (
        set(plan) != {
            "schema_version", "status", "room_id", "input_scene_content_hash",
            "config_sha256", "checkpoint_runtime_sha256", "selections",
            "selections_sha256", "attestation",
        }
        or plan.get("schema_version") != 1
        or plan.get("status") != "pass"
        or plan.get("room_id") != room_id
        or plan.get("checkpoint_runtime_sha256") != EXPECTED_RUNTIME_SHA256
        or plan.get("attestation") != attest(plan)
        or not isinstance(selections, list)
        or plan.get("selections_sha256") != sha_bytes(canonical(selections))
    ):
        raise SystemExit(f"{room_id}: plan contract mismatch")

    expected_checkpoints = contract["checkpoints"]
    actual = sorted(path.name for path in states.glob("manipuland_checkpoint_*"))
    if actual != [name for name, _receipt_sha in expected_checkpoints]:
        raise SystemExit(f"{room_id}: checkpoint inventory changed: {actual}")
    for index, (name, expected_receipt_sha) in enumerate(expected_checkpoints):
        receipt_path = states / name / "completion_receipt.json"
        require_regular(receipt_path)
        if sha_file(receipt_path) != expected_receipt_sha:
            raise SystemExit(f"{room_id}: checkpoint receipt changed at {index}")
        receipt = json.loads(receipt_path.read_bytes())
        if (
            receipt.get("schema_version") != 3
            or receipt.get("status") != "pass"
            or receipt.get("furniture_index") != index
            or receipt.get("selection") != selections[index]
            or receipt.get("selection_sha256") != sha_bytes(canonical(selections[index]))
            or receipt.get("checkpoint_runtime_sha256") != EXPECTED_RUNTIME_SHA256
            or receipt.get("attestation") != attest(receipt)
        ):
            raise SystemExit(f"{room_id}: checkpoint contract mismatch at {index}")

    updates = contract["updates"]
    completed_ids = {item["furniture_id"] for item in selections[: len(expected_checkpoints)]}
    if completed_ids & set(updates):
        raise SystemExit(f"{room_id}: update overlaps completed checkpoint")
    by_id = {item.get("furniture_id"): item for item in selections}
    if len(by_id) != len(selections) or not set(updates).issubset(by_id):
        raise SystemExit(f"{room_id}: target selection inventory mismatch")

    repaired = [dict(item) for item in selections]
    for selection in repaired:
        update = updates.get(selection["furniture_id"])
        if update is not None:
            selection.update(update)
    if repaired[: len(expected_checkpoints)] != selections[: len(expected_checkpoints)]:
        raise SystemExit(f"{room_id}: completed selection changed")
    output = dict(plan)
    output["selections"] = repaired
    output["selections_sha256"] = sha_bytes(canonical(repaired))
    output["attestation"] = attest(output)
    return {
        "room_id": room_id,
        "states": states,
        "plan_path": plan_path,
        "mode": plan_path.stat().st_mode & 0o777,
        "before_bytes": before_bytes,
        "before_sha256": before_sha256,
        "output": output,
        "updated_furniture_ids": sorted(updates),
        "preserved_checkpoints": actual,
        "preserved_checkpoint_receipts": [value for _name, value in expected_checkpoints],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-dir", required=True, type=Path)
    parser.add_argument("--backup-dir", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    scene = args.scene_dir.resolve(strict=True)
    if args.receipt.exists():
        raise SystemExit(f"refusing to overwrite receipt: {args.receipt}")
    args.backup_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    with ExitStack() as stack:
        for room_id in sorted(ROOMS):
            states = scene / f"room_{room_id}" / "scene_states"
            lock = stack.enter_context((states / ".manipuland_checkpoint.lock").open("a+b"))
            fcntl.flock(lock, fcntl.LOCK_EX)
        validated = [
            validate_room(scene, room_id, ROOMS[room_id]) for room_id in sorted(ROOMS)
        ]
        for record in validated:
            backup = args.backup_dir / (
                f"{record['room_id']}.manipuland_furniture_plan."
                f"{record['before_sha256']}.json"
            )
            if not backup.exists():
                shutil.copy2(record["plan_path"], backup, follow_symlinks=False)
            require_regular(backup)
            if sha_file(backup) != record["before_sha256"]:
                raise SystemExit(f"backup mismatch: {backup}")

        published: list[dict[str, object]] = []
        try:
            for record in validated:
                atomic_bytes(
                    record["plan_path"], encoded_json(record["output"]), record["mode"]
                )
                after_sha256 = sha_file(record["plan_path"])
                published.append(record)
                records.append(
                    {
                        "room_id": record["room_id"],
                        "plan_before_sha256": record["before_sha256"],
                        "plan_after_sha256": after_sha256,
                        "updated_furniture_ids": record["updated_furniture_ids"],
                        "preserved_checkpoints": record["preserved_checkpoints"],
                        "preserved_checkpoint_receipt_sha256": record[
                            "preserved_checkpoint_receipts"
                        ],
                    }
                )
        except BaseException:
            for record in published:
                atomic_bytes(
                    record["plan_path"], record["before_bytes"], record["mode"]
                )
            raise

    receipt: dict[str, object] = {
        "schema_version": 1,
        "status": "pass",
        "operation": "harden_only_failed_uncompleted_manipuland_targets",
        "rooms": records,
        "completed_checkpoint_policy": "all completed selections and receipts preserved byte-for-byte",
        "quality_policy": (
            "all prompt categories preserved; only unstable placement instructions changed; "
            "physical, collision, visual, clearance, and inventory thresholds unchanged"
        ),
    }
    receipt["attestation"] = attest(receipt)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    atomic_bytes(args.receipt, encoded_json(receipt), 0o600)
    print("FAILED_MANIPULAND_TARGET_HARDENING_PASS", len(records))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
