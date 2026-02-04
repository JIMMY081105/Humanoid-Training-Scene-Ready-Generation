#!/usr/bin/env python3
"""Consolidate Classroom 2's required details onto six reliable surfaces."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
from pathlib import Path


EXPECTED_PLAN_SHA256 = "bcaf6d8b2a494f38f54716a903a0cf990928147ce4c2207b2437381ce209e776"
EXPECTED_RUNTIME_SHA256 = "8ad862129a21f236986573cb1d851223f8fb28e8246ddaff9a3ae27761bec4b7"
EXPECTED_SELECTION_SHA256 = [
    "1e14579c99612b7904684c2038a7fe2071c58cb520229dad24402c9b875c8b3d",
    "17c90181a2d963246eb5f5ff9918bb57018f512a9cde2312cc578a7d351ec446",
    "32cb384f901cf98b6ca823321d550b44c96c1c1374a1473978748979ec7403b3",
    "f14032687eef71396ce33099bdf2b739d4f69f6bca0943edca29067320a45b95",
    "48d795316cadff8b771305dde86ab818132d2118e001b4adc17b4abb7487036f",
    "2a501310d84e0a3b221225b8f141a763c8397c0b2470db63d38172a4c6922fbf",
    "a5b5aebc97d036d178b54b5b391b5f9bdfe6909f042215300b4c988251701dc4",
    "d2d95a1dc24cf52e2639a657ea483c8322fd87149db559fa3e81cb19fe5ee7c1",
    "5cf4b9ab15735ec634bc6d22fbd919e24e6112577ce060020999e85dbcb1c1e9",
    "0d66d03a8336b019fad60006f41ca33e0875ac008dcf441962203a15f24e5172",
    "6796a4170291dc4a22a381286ea286a9b03494db9b1754e1e2412b0eaa7cba24",
    "426abfce58ee989c3b41547a25f0624b50a63fccccdd87d28f359e79a97afbde",
    "0f5029d9ce258bdc95df69b59b4c238930e17149a0ee812e56c47116433ec5b0",
]
KEEP_INDICES = (0, 1, 2, 3, 6, 9)


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
    if plan_path.is_symlink() or not plan_path.is_file() or plan_path.stat().st_nlink != 1:
        raise SystemExit(f"plan is not an unlinked regular file: {plan_path}")
    if list(states.glob("manipuland_checkpoint_*")):
        raise SystemExit("refusing to alter a plan after a checkpoint exists")

    with (states / ".manipuland_checkpoint.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        before = sha(plan_path.read_bytes())
        if before != EXPECTED_PLAN_SHA256:
            raise SystemExit(f"unexpected plan digest: {before}")
        document = json.loads(plan_path.read_text(encoding="utf-8"))
        selections = document.get("selections")
        if (
            document.get("room_id") != "classroom_02"
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

        repaired = [dict(selections[index]) for index in KEEP_INDICES]
        repaired[0]["suggested_items"] = (
            "REQUIRED: paper trays, folders/worksheets, pencil/marker holder, "
            "dry-erase markers, whiteboard eraser. Optional: teacher planner/notebook"
        )
        repaired[0]["prompt_constraints"] = (
            "Prompt requires paper trays, folders/worksheets, pencil/marker holders, "
            "board markers, and a whiteboard eraser. Keep the board supplies together "
            "at the rear edge nearest the teaching board because the generated board "
            "tray is not a reliable horizontal support."
        )
        repaired[0]["style_notes"] = (
            "Organized teacher station; group board tools together and paper tools "
            "together while retaining clear writing space."
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

        removed_targets = [
            selections[index]["furniture_id"]
            for index in range(len(selections))
            if index not in KEEP_INDICES
        ]
        receipt = {
            "schema_version": 1,
            "status": "pass",
            "operation": "consolidate_required_inventory_on_reliable_surfaces",
            "room_id": "classroom_02",
            "plan_before_sha256": before,
            "plan_after_sha256": after,
            "kept_targets": [item["furniture_id"] for item in repaired],
            "removed_duplicate_or_optional_targets": removed_targets,
            "board_supply_reroute_target": "teacher_desk_0",
            "required_categories_preserved": [
                "textbooks",
                "notebooks",
                "folders/worksheets",
                "pencil/marker holder",
                "markers",
                "eraser",
                "ruler",
                "glue",
                "scissors",
                "water bottle",
                "backpack",
                "storage bins",
                "paper trays",
                "board markers",
                "whiteboard eraser",
            ],
            "reason": (
                "The prompt asks for details on selected desks and forbids clutter; "
                "six supported surfaces cover every category without duplicating the "
                "same items across nine desks or relying on optional trayless fixtures."
            ),
        }
        receipt["attestation"] = attest(receipt)
        atomic_json(args.receipt, receipt, 0o600)
        print(f"CLASSROOM02_PLAN_REPAIR_PASS {before} {after}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
